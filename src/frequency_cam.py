#!/usr/bin/env python3
# -----------------------------------------------------------------------------
# Copyright 2022 Bernd Pfrommer <bernd.pfrommer@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
#
"""Python implementation of the Frequency Cam filter algorithm."""

import numpy as np
import math
from event_types import EventCD

#
# structure to hold the filter state and aux variables
#

State = np.dtype(
    {'names': ['t_flip_up_down', 't_flip_down_up',
               'L',  'L_lag', 'period', 'p'],
     'formats': ['<i8', '<i8', '<f4', '<f4', '<f4', '<i1'],
     'offsets': [0,       8,     16,   20,     24,     28],
     'itemsize': 29})


class FrequencyCam():
    def __init__(self, width, height,
                 min_freq, max_freq,
                 cutoff_period=20,
                 frame_timestamps=None,
                 frame_timeslice=None,
                 timeout_cycles=2.0,
                 debug_pixels=(-1, -1),
                 extra_args={}):
        self._res = (width, height)
        self._freq_range = np.array((min_freq, max_freq))
        self._extra_args = extra_args
        self._frame_number = 0
        self._dt_min = 1.0 / max_freq
        self._dt_max = 1.0 / min_freq
        self._timeout_cycles = float(timeout_cycles)
        self._debug_x = debug_pixels[0]
        self._debug_y = debug_pixels[1]

        if frame_timestamps is not None:
            ts = np.loadtxt(frame_timestamps, dtype=np.int64) // 1000
            self._timestamps = ts[:, 0] if ts.ndim > 1 else ts
        else:
            self._timestamps = None

        self._frame_timeslice = None if frame_timeslice is None \
            else int(frame_timeslice * 1e6)  # convert to us
        self._current_frame_end_time = None
        self._events = []  # no pending events yet
        self._width = width
        self._height = height
        self._state = None
        if cutoff_period > 0:
            # compute alpha, beta, and IIR coefficients (see paper)
            omega_cut = 2 * math.pi / cutoff_period
            alpha = (1 - math.sin(omega_cut)) / math.cos(omega_cut)
            phi = 2 - math.cos(omega_cut)
            beta = phi - math.sqrt(phi**2 - 1)
            self._c = np.array((alpha + beta, -alpha * beta, 0.5 * (1 + beta)),
                               dtype=np.float32)
        else:
            self._c = None
        self._merged_events = 0
        self._processed_events = 0
        self._events_this_frame = 0
        if self._debug_x != -1 and self._debug_y != -1:
            self._debug = open("freq.txt", "w")
            self._readout = open("readout.txt", "w")

    def get_frame_number(self):
        return self._frame_number

    def get_number_merged(self):
        return self._merged_events

    def get_number_events(self):
        return self._processed_events

    def initialize_state(self, t):
        self._state = np.empty((self._height, self._width), dtype=State)
        self._state['t_flip_up_down'] = t
        self._state['t_flip_down_up'] = t
        self._state['L'] = 0
        self._state['L_lag'] = 0
        self._state['period'] = -1
        self._state['p'] = -1  # should set to zero?
        self._last_event_time = t

    def set_output_callback(self, cb):
        self._callback = cb

    def update_state_filter(self, events):
        """Update state with slow loop."""
        if self._state is None:
            self.initialize_state(events['t'][0])
        self._processed_events += events.shape[0]
        self._events_this_frame += events.shape[0]
        self._last_event_time = events['t'][-1]
        for e in events:
            x, y, p, t = e['x'], e['y'], e['p'] * 2 - 1, e['t']
            # run the filter for approx reconstruction (see paper)
            L_km1 = self._state['L'][y, x]
            L_km2 = self._state['L_lag'][y, x]
            dp = p - self._state['p'][y, x]
            L_k = self._c[0] * L_km1 + self._c[1] * L_km2 + self._c[2] * dp

            if L_k < 0 and L_km1 > 0:
                # ---------------------------------------------
                # signal crosses the zero line from above
                dt_ud = (t - self._state['t_flip_up_down'][y, x]) * 1.0e-6
                if dt_ud >= self._dt_min and dt_ud <= self._dt_max:
                    # period is within valid range, use it
                    self._state['period'][y, x] = dt_ud
                else:
                    dt_du = (t - self._state['t_flip_down_up'][y, x]) * 1.0e-6
                    if self._state['period'][y, x] > 0:
                        to = self._state['period'][y, x] * self._timeout_cycles
                        if dt_ud > to and dt_du > 0.5 * to:
                            self._state['period'][y, x] = 0  # stale pixel
                        elif (dt_du >= 0.5 * self._dt_min) and \
                             (dt_du <= 0.5 * self._dt_max):
                            # don't have a valid period, init from half cycle
                            self._state['period'][y, x] = 2 * dt_du
                # update the flip time
                self._state['t_flip_up_down'][y, x] = t
            elif L_k > 0 and L_km1 < 0:
                # ---------------------------------------------
                # signal crosses the zero line from below
                dt_du = (t - self._state['t_flip_down_up'][y, x]) * 1.0e-6
                if self._state['period'][y, x] <= 0 and \
                   dt_du >= self._dt_min and dt_du <= self._dt_max:
                    # period is within valid range and no period available yet
                    self._state['period'][y, x] = dt_du
                else:
                    # can half-cycle transition be used ?
                    dt_ud = (t - self._state['t_flip_up_down'][y, x]) * 1.0e-6
                    if self._state['period'][y, x] > 0:
                        # have valid period, may have to time it out
                        to = self._state['period'][y, x] * self._timeout_cycles
                        if dt_du > to and dt_ud > 0.5 * to:
                            self._state['period'][y, x] = 0  # stale pixel
                    elif (dt_ud >= 0.5 * self._dt_min) and \
                         (dt_ud <= 0.5 * self._dt_max):
                        # have invalid period, initialize from half-cycle
                        self._state['period'][y, x] = 2 * dt_ud
                # update the flip time
                self._state['t_flip_down_up'][y, x] = t

            if x == self._debug_x and y == self._debug_y:
                dt = (t - max(self._state['t_flip_up_down'][y, x],
                              self._state['t_flip_down_up'][y, x])) * 1e-6
                self._debug.write(f"{t + self.time_offset} {dp} {L_k}"
                                  + f" {L_km1} {L_km2} {dt}"
                                  + f" {self._state['period'][y, x]}"
                                  + f" {self._dt_min} {self._dt_max}\n")
                self._debug.flush()

            # update twice lagged signal
            self._state['L_lag'][y, x] = L_km1
            # update once lagged signal
            self._state['L'][y, x] = L_k
            # remember polarity
            self._state['p'][y, x] = p

    def make_frequency_map(self, t_now):
        if self._state is None:
            return np.zeros((self._height, self._width), dtype=np.float32)
        period = self._state['period']
        # filter out:
        #   - pixels where no period has been detected yet
        #   - pixels that have not flipped for two periods

        dt = (t_now - np.maximum(self._state['t_flip_up_down'],
                                 self._state['t_flip_down_up'])) * 1e-6
        fm = np.divide(1.0, period,
                       out=np.zeros_like(period),
                       where=((period > 0)
                              & (dt < period * self._timeout_cycles)
                              & (dt < self._dt_max * self._timeout_cycles)))

        if self._debug_x != -1 and self._debug_y != -1:
            self._readout.write(
                f'{(t_now  + self.time_offset) * 1e-6} ' +
                f'{fm[self._debug_y, self._debug_x]}\n')
            self._readout.flush()
        return fm

    def update_state_from_list(self, event_list):
        if len(event_list) > 0:
            event_camera = np.array(event_list, dtype=EventCD)
            self.update_state(event_camera)

    def update_state(self, event_camera):
        self.update_state_filter(event_camera)
        self._events.append(event_camera)

    def process_events(self, events):
        if self._timestamps is not None:
            self.process_events_with_timestamps(events)
        else:
            self.process_events_to_frames(events)

    def process_events_with_timestamps(self, events):
        if self._frame_number >= self._timestamps.shape[0]:
            return  # already processed all frames

        if events[-1]['t'] < self._timestamps[self._frame_number]:
            # in the not unusual case where all events in this call
            # belong to the current frame and there is no cross-over
            # into the next one, simply update the state and store the events
            self.update_state(events)
        else:
            # need to split
            event_list = []
            for e in events:
                while (self._frame_number < self._timestamps.shape[0] and
                       e['t'] > self._timestamps[self._frame_number]):
                    # write events and bump frame number until caught
                    # up with the current event time
                    self.update_state_from_list(event_list)
                    event_list = []
                    fmap = self.make_frequency_map(self._last_event_time)
                    self._callback(
                        self._frame_number, self._events, fmap,
                        self._freq_range, self._extra_args)
                    self._events = []
                    print(f"frame {self._frame_number} has events: ",
                          f"{self._events_this_frame}")
                    self._events_this_frame = 0
                    self._frame_number += 1
                event_list.append(e)
            # process accumulated events
            event_list = self.update_state_from_list(event_list)

    def process_events_no_callback(self, events, time_offset):
        self.time_offset = time_offset
        self.update_state(events)

    def process_events_to_frames(self, events):
        # handle case of first event received
        if self._current_frame_end_time is None:
            self._current_frame_end_time = events[0]['t'] \
                + self._frame_timeslice

        if events[-1]['t'] < self._current_frame_end_time:
            # in the not unusual case where all events in this call
            # belong to the current frame and there is no cross-over
            # into the next one, simply update the state and store the events
            self.update_state(events)
        else:
            # need to split
            event_list = []
            for e in events:
                while e['t'] > self._current_frame_end_time:
                    # write events and bump frame number until caught
                    # up with the current event time
                    self.update_state_from_list(event_list)
                    event_list = []
                    fmap = self.make_frequency_map(self._last_event_time)
                    self._callback(
                        self._frame_number, self._events, fmap,
                        self._freq_range, self._extra_args)
                    self._events = []
                    print(f"frame {self._frame_number} has events: "
                          + f"{self._events_this_frame}")
                    self._events_this_frame = 0
                    self._frame_number += 1
                    self._current_frame_end_time += self._frame_timeslice
                event_list.append(e)
            # process accumulated events
            event_list = self.update_state_from_list(event_list)
