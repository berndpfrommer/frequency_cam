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
    img = make_bg_image(img, events)
    nz_idx = freq_map > 0  # indices of non-zero elements of frequency map
    fname = str(extra_args['output_dir'] / f"frame_{frame_number:05d}.jpg")
    # print(f'fmap min: {np.min(freq_map)}  max: {np.max(freq_map)}')
    if nz_idx.sum() > 0:
        r = freq_range[1] - freq_range[0]
        # scale image into range of 0..255
        scaled = cv2.convertScaleAbs(freq_map, alpha=255.0 / r,
                                     beta=-freq_range[0] * 255.0 / r)
        img_scaled = cv2.applyColorMap(scaled, cv2.COLORMAP_JET)
        # only color those points where valid events happen
        img[nz_idx, :] = img_scaled[nz_idx, :]
        cv2.imwrite(fname, img)
    else:
        cv2.imwrite(fname, img)

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
                        default=0.1, type=float)
    parser.add_argument('--timestamp_file',
                        help='name of file to read frame time stamps from',
                        default=None)
    parser.add_argument('--output_dir', help='name of output directory',
                        default='frames')

    args = parser.parse_args()
    
    events, res = read_bag(args.bag, args.topic, converter=EventCDConverter())

    # make directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    algo = FrequencyCam(width=res[0], height=res[1],
                        min_freq=args.freq_min, max_freq=args.freq_max,
                        cutoff_period=args.cutoff_period,
                        frame_timestamps=args.timestamp_file,
                        period_averaging_alpha=args.period_averaging_alpha,
                        extra_args={'output_dir': Path(args.output_dir)})

    algo.set_output_callback(write_image_cb)

    for evs in events:
        if evs.size > 0:
            algo.process_events(evs)
            if algo.get_frame_number() > 150:
                print('XXX premature exit!')
                break
    
    print(f'merged {algo.get_number_merged()} events of ',
          f'{algo.get_number_events()} ',
          f'({algo.get_number_merged() * 100 / algo.get_number_events():.2f}%)')
