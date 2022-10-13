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
"""Compute frequency image using the metavision SDK's analytics."""

import cv2
import numpy as np
import argparse
from metavision_sdk_analytics import FrequencyMapAsyncAlgorithm
from pathlib import Path
from read_bag_ros2 import read_bag, EventCDConverter

# global variables to keep track of current frame, events etc

output_dir = ""  # output directory name
last_events = []
t_curr = 0   # current time stamp
frame_count = 0  # current frame
frame_time_stamps = []
freq_range = np.array([0, 0])
use_log_scale = False


def make_bg_image(img, events):
    """Make background image of events."""
    for evts in events:
        x = evts[:]['x']
        y = evts[:]['y']
        img[y, x, :] = 128  # set all events gray
    return img


def write_image_cb(ts, freq_map):
    global last_events
    global frame_count
    # print(ts, t_curr)
    img = np.zeros([freq_map.shape[0], freq_map.shape[1], 3], dtype=np.uint8)
    img = make_bg_image(img, last_events)
    last_events = []  # clear out all events
    nz_idx = freq_map > 0  # indices of non-zero elements of frequency map
    fname = str(Path(output_dir) / f"frame_{frame_count:05d}.jpg")

    if nz_idx.sum() > 0:
        fr_tf = np.log10(freq_range) if use_log_scale else freq_range
        freq_map_tf = \
            np.where(freq_map > 0, np.log10(freq_map), 0) if use_log_scale \
            else freq_map
        r = fr_tf[1] - fr_tf[0]
        # scale image into range of 0..255
        scaled = cv2.convertScaleAbs(freq_map_tf, alpha=255.0 / r,
                                     beta=-fr_tf[0] * 255.0 / r)
        img_scaled = cv2.applyColorMap(scaled, cv2.COLORMAP_JET)
        img[nz_idx, :] = img_scaled[nz_idx, :]
        if frame_count % 10 == 0:
            print('writing image: ', frame_count)
        cv2.imwrite(fname, img)
    else:
        print('writing empty image: ', fname)
        cv2.imwrite(fname, np.zeros_like(freq_map, dtype=np.uint8))

    frame_time_stamps.append((t_curr * 1000, frame_count))
    frame_count += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='compute frequency image with metavision SDK.')
    parser.add_argument('--bag', '-b', action='store', default=None,
                        required=True, help='bag file to read events from')
    parser.add_argument('--topic', help='Event topic to read',
                        default='/event_camera/events', type=str)
    parser.add_argument('--freq_min', help='min frequency (hz)',
                        default=200, type=float)
    parser.add_argument('--freq_max', help='max frequency (hz)',
                        default=300, type=float)
    parser.add_argument('--update_freq', help='callback frequency (hz)',
                        default=100, type=float)
    parser.add_argument('--diff_thresh', help='time period diff thresh (usec)',
                        default=100, type=int)
    parser.add_argument('--filter_length',
                        help='min num periods for valid detection',
                        default=1, type=int)
    parser.add_argument('--timestamp_file',
                        help='name of file to write time stamps to',
                        default='mv_timestamps.txt')
    parser.add_argument('--output_dir', help='name of output directory',
                        default='frames')
    parser.add_argument('--log_scale', action='store_true',
                        required=False, help='color frequency on log scale')
    parser.set_defaults(log_scale=False)

    args = parser.parse_args()

    use_log_scale = args.log_scale

    events, res = read_bag(args.bag, args.topic, converter=EventCDConverter())

    algo = FrequencyMapAsyncAlgorithm(
        width=res[0], height=res[1], filter_length=args.filter_length,
        min_freq=args.freq_min, max_freq=args.freq_max,
        diff_thresh_us=args.diff_thresh)
    algo.update_frequency = args.update_freq
    algo.set_output_callback(write_image_cb)

    output_dir = args.output_dir
    freq_range = np.array([args.freq_min, args.freq_max])

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    for evs in events:
        if evs.size > 0:
            last_events.append(evs)
            t_curr = evs[-1][3]
            # update the algo with events. Sometimes no callback happens,
            # I believe when the frequency map does not change.
            # The callback has a "ts" time stamp argument but it seems
            # that a few events are included that are later than
            # that time stamp
            algo.process_events(evs)

    np.savetxt(args.timestamp_file, np.array(frame_time_stamps).astype(np.uint64),
               fmt='%d')
