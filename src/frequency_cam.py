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

TimeAndPolarity = np.dtype({'names': ['t', 'p'],
                            'formats': ['<u4', '<i1'],
                            'offsets': [0, 4], 'itemsize': 5})

State = np.dtype(
    {'names': ['t_flip', 'L',  'L_lag', 'period_avg', 'p', 'skip', 'idx', 'tp'],
     'formats': ['<f4', '<f4', '<f4',     '<f4',    '<i1', '<u1', '<u1',
                 TimeAndPolarity * 4],
     'offsets': [0,       4,      8,      12,     16,    17,     18,   19],
     'itemsize': 39})

debug_file = open("debug_file.txt", "w")


class FrequencyCam():
    def __init__(self, width, height,
                 min_freq, max_freq,
                 cutoff_period=20,
                 frame_timestamps=None,
                 period_averaging_alpha=0.1,
                 extra_args={}):
        self._res = (width, height)
        self._freq_range = (min_freq, max_freq)
        self._extra_args = extra_args
        self._frame_number = 0
        self._dt_min = 1.0 / max_freq
        self._dt_max = 1.0 / min_freq
        self._dt_mix = period_averaging_alpha
        if frame_timestamps is not None:
            ts = np.loadtxt(frame_timestamps, dtype=np.int64) // 1000
            self._timestamps = ts[:, 0] if ts.ndim > 1 else ts
        else:
            self._timestamps = None
        self._events = []  # no pending events yet
        self._width = width
        self._height = height
        self._state = None
        # compute alpha, beta, and IIR coefficients (see paper)
        omega_cut = 2 * math.pi / cutoff_period
        alpha = (1 - math.sin(omega_cut)) / math.cos(omega_cut)
        phi = 2 - math.cos(omega_cut)
        beta = phi - math.sqrt(phi**2 - 1)
        self._c_ = np.array((alpha + beta, -alpha * beta, 0.5 * (1 + beta)),
                            dtype=np.float32)
        self._merged_events = 0
        self._processed_events = 0

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
        self._state['p'] = 0
        self._state['skip'] = 0
        self._state['idx'] = 0
        for lag in range(4):
            self._state['tp'][lag]['t'] = 0
            self._state['tp'][lag]['p'] = 0
        self._last_event_time = t
        
    def set_output_callback(self, cb):
        self._callback = cb

    def update_state_vectorized(self, events):
        """updating state in a vectorized way (not reliable)"""
        if self._state is None:
            self.initialize_state(events[0]['t'])
        # change in polarity
        x = events['x']
        y = events['y']
        p = events['p'] * 2 - 1
        p_tmp = np.zeros((self._height, self._width), dtype=np.int8)
        np.add.at(p_tmp, (y, x), p)          # will add dup event polarities!
        yx_u = np.unique(np.column_stack((y, x)), axis=0)     # unique indices
        self._merged_events += x.shape[0] - yx_u.shape[0]
        self._processed_events += x.shape[0]
        y_u = yx_u[:, 0]
        x_u = yx_u[:, 1]
        p_u = p_tmp[y_u, x_u]  # unique polarities
        t_tmp = np.zeros((self._height, self._width), dtype=np.int64)
        np.maximum.at(t_tmp, (y, x), events['t']) # should find max time stamp
        t_u = t_tmp[y_u, x_u]  # unique time stamps
        self._last_event_time = events['t'][-1]
        
        # run the filter for approx reconstruction (see paper)
        L_km1 = self._state['L'][y_u, x_u]
        L_km2 = self._state['L_lag'][y_u, x_u]
        L_k = self._c_[0] * L_km1 \
            + self._c_[1] * L_km2 \
            + self._c_[2] * p_u

        # find locations where signal crosses zero line from above
        flip_idx = (L_k < 0) & (L_km1 >= 0)  # indices where sign flips
        y_f, x_f = y_u[flip_idx], x_u[flip_idx] # 2d indices where sign flips

        t_uf = t_u[flip_idx] # times where sign flips
        # time difference where sign flips
        period = (t_uf - self._state['t_flip'][y_f, x_f]) * 1.0e-6

        # update the flip time
        self._state['t_flip'][y_f, x_f] = t_uf

        good_update = (period >= self._dt_min) & (period <= self._dt_max)
        y_g, x_g = y_f[good_update], x_f[good_update]
        
        # compound period into average
        self._state['period_avg'][y_g, x_g] = period[good_update]
        
        # update twice lagged signal
        self._state['L_lag'][y_u, x_u] = L_km1

        # update once lagged signal
        self._state['L'][y_u, x_u] = L_k

        # remember polarity
        self._state['p'][y_u, x_u] = p_u

    def update_state_slow(self, events):
        """updating state with loop"""
        if self._state is None:
            self.initialize_state(events[0]['t'])
        self._processed_events += events.shape[0]
        self._last_event_time = events['t'][-1]
        for e in events:
            x = e['x']
            y = e['y']
            p = e['p'] * 2 - 1
            t = e['t']
            # run the filter for approx reconstruction (see paper)
            L_km1 = self._state['L'][y, x]
            L_km2 = self._state['L_lag'][y, x]
            L_k = self._c_[0] * L_km1 + self._c_[1] * L_km2 + self._c_[2] * p

            # test if signal crossed zero from above
            if (L_k < 0) & (L_km1 >= 0):
                period = (t - self._state['t_flip'][y, x]) * 1.0e-6
                # update the flip time
                self._state['t_flip'][y, x] = t
                # if period passes sanity check, compound into average
                if period >= self._dt_min and period <= self._dt_max:
                    # compound period into average
                    self._state['period_avg'][y, x] = period
            # update twice lagged signal
            self._state['L_lag'][y, x] = L_km1
            # update once lagged signal
            self._state['L'][y, x] = L_k
            # remember polarity
            self._state['p'][y, x] = p
            if y == 327 and x == 3:
                debug_file.write(f"{t} {p} {L_k}\n")

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
        #self.update_state_vectorized(event_array)
        self.update_state_slow(event_array)
        self._events.append(event_array)
        
    def process_events(self, events):
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
                    self._frame_number += 1
                event_list.append(e)
            # process accumulated events
            event_list = self.update_state_from_list(event_list)
