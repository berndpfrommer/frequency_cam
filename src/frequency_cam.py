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
"""
frequency camera algorithm
"""

import numpy as np
import math
from event_types import EventCD

# for the temporal filter
TimeAndPolarity = np.dtype({'names': ['t', 'p'],
                            'formats': ['<u4', '<i1'],
                            'offsets': [0, 4], 'itemsize': 5})

debug = open("freq.txt", "w")
#
# structure to hold the filter state and counters for keeping a clean average
#

State = np.dtype(
    {'names': ['t_flip', 'L',  'L_lag', 'period_avg', 'p', 'skip', 'idx', 'tp'],
     'formats': ['<f4', '<f4', '<f4',     '<f4',    '<i1', '<u1', '<u1',
                 TimeAndPolarity * 4],
     'offsets': [0,       4,      8,      12,     16,    17,     18,   19],
     'itemsize': 39})


class FrequencyCam():
    def __init__(self, width, height,
                 min_freq, max_freq,
                 cutoff_period=20,
                 frame_timestamps=None,
                 frame_timeslice=None,
                 period_averaging_alpha=0.1,
                 reset_threshold=0.2,
                 extra_args={}):
        self._res = (width, height)
        self._freq_range = np.array((min_freq, max_freq))
        self._extra_args = extra_args
        self._frame_number = 0
        self._dt_min = 1.0 / max_freq
        self._dt_max = 1.0 / min_freq
        self._dt_mix = period_averaging_alpha
        self._one_minus_dt_mix = 1.0 - self._dt_mix
        self._reset_thresh = reset_threshold # percent off before average is reset
        self._max_num_diff_period = int(3)

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

        if self._frame_timeslice is None and self._timestamps is None:
            raise Exception(
                "must specify frame_timeslice or frame_timestamps")

    def get_frame_number(self):
        return self._frame_number

    def get_number_merged(self):
        return self._merged_events

    def get_number_events(self):
        return self._processed_events

    def initialize_state(self, t):
        self._state = np.empty((self._height, self._width), dtype=State)
        self._state['t_flip'] = t
        self._state['L'] = 0
        self._state['L_lag'] = 0
        self._state['period_avg'] = -1
        self._state['p'] = -1  # should set to zero?
        self._state['skip'] = 0
        self._state['idx'] = 0
        for lag in range(4):
            self._state['tp'][lag]['t'] = 0
            self._state['tp'][lag]['p'] = -1  # should set to zero ?
        self._last_event_time = t
        
    def set_output_callback(self, cb):
        self._callback = cb

    def update_state_baseline(self, events):
        """updating state using baseline method (with loop ,slow)"""
        if self._state is None:
            self.initialize_state(events[0]['t'])
        self._processed_events += events.shape[0]
        self._events_this_frame += events.shape[0]
        self._last_event_time = events['t'][-1]
        for e in events:
            x, y, p, t = e['x'], e['y'], e['p'] * 2 - 1, e['t']
            if self._state['p'][y, x] == 1 and p == -1:
                dt = (t - self._state['t_flip'][y, x]) * 1.0e-6
                # update the flip time
                self._state['t_flip'][y, x] = t
                if dt >= self._dt_min and dt <= self._dt_max:
                    # measured period is within acceptable freq range
                    curr_avg = self._state['period_avg'][y, x]
                    if curr_avg < 0 or self._dt_mix >= 1.0:
                        # initialize period average
                        self._state['period_avg'][y, x] = dt
                    else:
                        if abs(dt - curr_avg) > curr_avg * self._reset_thresh:
                            # measured period is too far off from current estim
                            self._state[y, x]['skip'] += 1
                            if self._state[y, x]['skip'] \
                               > self._max_num_diff_period:
                                # got too many bad cycles
                                self._state['skip'][y, x] = 0
                                curr_avg = dt
                        else:
                            # period passes sanity check, compound into average
                            self._state['period_avg'][y, x] = \
                                curr_avg * self._one_minus_dt_mix \
                                + dt * self._dt_mix
            # remember polarity
            self._state['p'][y, x] = p

    def update_state_filter(self, events):
        """updating state with loop (slow)"""
        if self._state is None:
            self.initialize_state(events[0]['t'])
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

            if (L_k < 0) & (L_km1 >= 0):
                # signal crosses the zero line from above
                dt = (t - self._state['t_flip'][y, x]) * 1.0e-6
                # update the flip time
                self._state['t_flip'][y, x] = t
                if dt >= self._dt_min and dt <= self._dt_max:
                    # measured period is within acceptable freq range
                    curr_avg = self._state['period_avg'][y, x]
                    if curr_avg < 0 or self._dt_mix >= 1.0:
                        # initialize period average
                        self._state['period_avg'][y, x] = dt
                    else:
                        if abs(dt - curr_avg) > curr_avg * self._reset_thresh:
                            # measured period is too far off from current estim
                            self._state[y, x]['skip'] += 1
                            if self._state[y, x]['skip'] \
                               > self._max_num_diff_period:
                                # got too many bad cycles
                                self._state['skip'][y, x] = 0
                                curr_avg = dt
                        else:
                            # period passes sanity check, compound into average
                            self._state['period_avg'][y, x] = \
                                curr_avg * self._one_minus_dt_mix \
                                + dt * self._dt_mix
            if x == 319 and y == 239:
                dt = (t - self._state['t_flip'][y, x]) * 1.0e-6
                debug.write(f"{t} {dp} {L_k} {L_km1} {L_km2} {dt}" 
                            + f" {self._state['period_avg'][y, x]}\n")

            # update twice lagged signal
            self._state['L_lag'][y, x] = L_km1
            # update once lagged signal
            self._state['L'][y, x] = L_k
            # remember polarity
            self._state['p'][y, x] = p

    def make_frequency_map(self, t_now):
        period = self._state['period_avg']
        # filter out:
        #   - pixels where no period has been detected yet
        #   - pixels that have not flipped for two periods
        fm = np.where(
            (period > 0)
            & (t_now - self._state['t_flip'] < 2e6 * period),
            1.0 / self._state['period_avg'], 0)
        
        return fm

    def update_state_from_list(self, event_list):
        if len(event_list) > 0:
            event_array = np.array(event_list, dtype=EventCD)
            self.update_state(event_array)

    def update_state(self, event_array):
        if self._c is None:
            self.update_state_baseline(event_array)
        else:
            self.update_state_filter(event_array)
        self._events.append(event_array)

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
                    print(f"frame {self._frame_number} has events: {self._events_this_frame}")
                    self._events_this_frame = 0
                    self._frame_number += 1
                event_list.append(e)
            # process accumulated events
            event_list = self.update_state_from_list(event_list)

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
