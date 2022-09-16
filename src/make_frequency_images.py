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
makes sequence of frequency images from ros2 bag
"""

import cv2
import numpy as np
import argparse

from pathlib import Path
from frequency_cam import FrequencyCam
from read_bag_ros2 import read_bag, EventCDConverter


global out_ts_file
out_ts_file = None  # global variable


def make_bg_image(img, events):
    """make_bg_image():  make background image based on events"""
    num_events = 0
    for evts in events:
        x = evts[:]['x']
        y = evts[:]['y']
        img[y, x, :] = 128  # set all events gray
        num_events += evts.shape[0]
    return img


def write_image_cb(frame_number, events, freq_map, freq_range, extra_args):
    img = np.zeros([freq_map.shape[0], freq_map.shape[1], 3], dtype=np.uint8)
    if extra_args['make_bg_image']:
        img = make_bg_image(img, events)
    nz_idx = freq_map > 0  # indices of non-zero elements of frequency map
    fname = str(extra_args['output_dir'] / f"frame_{frame_number:05d}.jpg")
    # print(f'fmap min: {np.min(freq_map)}  max: {np.max(freq_map)}')
    use_log = extra_args['log_scale']
    if nz_idx.sum() > 0:
        fr_tf = np.log10(freq_range) if use_log else freq_range
        freq_map_tf = \
            np.where(freq_map > 0, np.log10(freq_map), 0) if use_log \
            else freq_map
        r = fr_tf[1] - fr_tf[0]
        # scale image into range of 0..255
        scaled = cv2.convertScaleAbs(freq_map_tf, alpha=255.0 / r,
                                     beta=-fr_tf[0] * 255.0 / r)
        img_scaled = cv2.applyColorMap(scaled, cv2.COLORMAP_JET)
        # only color those points where valid events happen
        img[nz_idx, :] = img_scaled[nz_idx, :]
        cv2.imwrite(fname, img)
    else:
        cv2.imwrite(fname, img)

    global out_ts_file
    if out_ts_file is not None:
        out_ts_file.write(f"{events[-1][-1]['t']*1000:d} {frame_number:d}\n")

    if frame_number % 10 == 0:
        print('writing image: ', frame_number)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='compute frequency image with frequency cam algorithm.')
    parser.add_argument('--bag', '-b', action='store', default=None,
                        required=True, help='bag file to read events from')
    parser.add_argument('--topic', help='Event topic to read',
                        default='/event_camera/events', type=str)
    parser.add_argument('--freq_min', help='min frequency (hz)',
                        default=200, type=float)
    parser.add_argument('--freq_max', help='max frequency (hz)',
                        default=300, type=float)
    parser.add_argument('--cutoff_period',
                        help='number of events for cutoff period (>4)',
                        default=5, type=int)
    parser.add_argument('--period_averaging_alpha',
                        help='how much new period to mix in avg (0..1)',
                        default=0.2, type=float)
    parser.add_argument('--timestamp_file', required=False,
                        help='name of file to read frame time stamps from',
                        default=None)
    parser.add_argument('--output_timestamp_file', required=False,
                        help='name of file to write time stamps to',
                        default=None)
    parser.add_argument('--timeslice', required=False, type=float,
                        help='timeslice for each frame [seconds]',
                        default=None)
    parser.add_argument('--timeout_cycles', required=False, type=int,
                        help='number of cycles before pixel timeout',
                        default=2)
    parser.add_argument('--reset_threshold', required=False, type=float,
                        help='relative error at which to restart averaging',
                        default=1e6)
    parser.add_argument('--max_frames',
                        help='maximum number of frames to compute', type=int,
                        default=1000000)
    parser.add_argument('--output_dir', help='name of output directory',
                        default='frames')
    parser.add_argument('--log_scale', action='store_true',
                        required=False, help='plot frequency on log scale')
    parser.set_defaults(log_scale=False)
    parser.add_argument('--no_bg', action='store_true',
                        required=False, help='do not show background noise')
    parser.set_defaults(no_bg=False)

    args = parser.parse_args()
    
    events, res = read_bag(args.bag, args.topic, converter=EventCDConverter())

    if args.output_timestamp_file is not None:
        out_ts_file = open(args.output_timestamp_file, 'w')

    # make directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    algo = FrequencyCam(width=res[0], height=res[1],
                        min_freq=args.freq_min, max_freq=args.freq_max,
                        cutoff_period=args.cutoff_period,
                        frame_timestamps=args.timestamp_file,
                        frame_timeslice=args.timeslice,
                        period_averaging_alpha=args.period_averaging_alpha,
                        reset_threshold=args.reset_threshold,
                        timeout_cycles=args.timeout_cycles,
                        extra_args={'output_dir': Path(args.output_dir),
                                    'log_scale': args.log_scale,
                                    'make_bg_image': not args.no_bg})

    algo.set_output_callback(write_image_cb)

    for evs in events:
        if evs.size > 0:
            algo.process_events(evs)
            if algo.get_frame_number() >= args.max_frames:
                print('reached maximum number of frames!')
                break
    
    print(f'merged {algo.get_number_merged()} events of ',
          f'{algo.get_number_events()} ',
          f'({algo.get_number_merged() * 100 / algo.get_number_events():.2f}%)')
