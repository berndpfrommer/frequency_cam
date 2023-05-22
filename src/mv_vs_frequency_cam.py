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
"""Compare frequency image between metavision SDK and frequency camera."""

import cv2
import numpy as np
import argparse
from metavision_sdk_analytics import FrequencyMapAsyncAlgorithm
from frequency_cam import FrequencyCam
from pathlib import Path
from read_bag_ros2 import read_bag, EventCDConverter

# global variables to keep track of current frame, events etc

args = None

last_events = []
last_time = None

frame_count = 0     # current frame
freq_range = np.array([0, 0])
use_log_scale = False
fc_algo = None
labels = None
offset = None


def make_bg_image(events, res):
    """Make grey pixel background image based on events."""
    img = 255 * np.ones([res[1], res[0], 3], dtype=np.uint8)

    for evts in events:
        x = evts[:]['x']
        y = evts[:]['y']
        img[y, x, :] = 128  # set all events gray
    return img


def make_color_image(freq_map, freq_range, use_log_scale):
    fr_tf = np.log10(freq_range) if use_log_scale else freq_range
    freq_map_tf = np.log10(freq_map, out=np.zeros_like(freq_map),
                           where=(freq_map > 0)) if use_log_scale else freq_map.copy()
    r = fr_tf[1] - fr_tf[0]
    # scale image into range of 0..255
    scaled = cv2.convertScaleAbs(freq_map_tf, alpha=255.0 / r,
                                 beta=-fr_tf[0] * 255.0 / r)
    img_scaled = cv2.applyColorMap(scaled, cv2.COLORMAP_JET)
    return img_scaled


def make_fc_image(freq_map, use_log, bg_img):
    img = bg_img.copy()
    nz_idx = freq_map > 0  # indices of non-zero elements of frequency map
    if nz_idx.sum() > 0:
        img_scaled = make_color_image(freq_map, freq_range, use_log)
        # only color those points where valid events happen
        img[nz_idx, :] = img_scaled[nz_idx, :]
    return img


def draw_labeled_rectangle(img, x, dx, patch_height, text_height, text_label,
                           color):
    rec = (x, 0, dx, patch_height)
    y_margin = 10

    cv2.rectangle(img=img, rec=rec, color=color.tolist(),
                  thickness=-1, lineType=cv2.LINE_8)
    font = cv2.FONT_HERSHEY_PLAIN
    font_scale = args.font_scale
    font_thickness = args.font_thickness
    textsize = cv2.getTextSize(text_label, font, fontScale=font_scale,
                               thickness=font_thickness)[0]
    pos = (int(x + (dx - textsize[0]) / 2),
           int(patch_height + y_margin + text_height / 2))

    cv2.putText(img=img, text=text_label, org=pos,
                fontFace=font, fontScale=font_scale, color=(0, 0, 0),
                thickness=font_thickness, lineType=cv2.LINE_AA)


def make_legend(res, patch_height, text_height,
                text_labels, legend_values, color_map):
    img = 255 * np.ones([patch_height + text_height, res[0], 3],
                        dtype=np.uint8)

    patches = np.zeros(len(legend_values), dtype=np.float32)
    lv_tf = np.log10(legend_values) if use_log_scale else legend_values
    for i, v in enumerate(lv_tf):
        patches[i] = v
    # scale patch frequencies to colors the same way as image
    fr_tf = np.log10(freq_range) if use_log_scale else freq_range
    r = fr_tf[1] - fr_tf[0]
    scaled = cv2.convertScaleAbs(patches, alpha=255.0 / r,
                                 beta=-fr_tf[0] * 255.0 / r)
    color_code = cv2.applyColorMap(scaled, cv2.COLORMAP_JET)

    dx = int(res[0] / color_code.shape[0])
    for i, c in enumerate(color_code):
        x = dx * i
        draw_labeled_rectangle(
            img, x, dx, patch_height=patch_height, text_height=text_height,
            text_label=text_labels[i], color=color_code[i, :].squeeze())

    return img


def crop_map(freq_map):
    if args.tlx is None or args.tly is None \
       or args.brx is None or args.bry is None:
        return freq_map
    return freq_map[args.tly:args.bry, args.tlx:args.brx]


def scale_img(img, s):
    return cv2.resize(img, dsize=(int(s*img.shape[1]),  int(s*img.shape[0])))


def find_events_in_slice(ts, events):
    sl = []
    remainder = []
    for evs in events:
        t = evs['t']
        if t[-1] < ts:
            sl.append(evs)
        else:
            if t[0] >= ts:
                remainder.append(evs)
            else:
                sl.append(evs[t < ts].copy())
                remainder.append(evs[t >= ts].copy())
    return remainder, sl


def mv_write_image_cb(ts, freq_map_orig):
    global last_events
    global last_time
    global frame_count
    scale = args.scale
    orig_res = (freq_map_orig.shape[1], freq_map_orig.shape[0])
    freq_map = scale_img(crop_map(freq_map_orig), scale)
    res = (freq_map.shape[1], freq_map.shape[0])
    text_height = int(args.text_height * args.scale)
    patch_height = int(args.patch_height * args.scale)
    legend_height = 0 if labels is None else patch_height + text_height
    img = np.zeros([res[1] * 2 + legend_height, res[0], 3], dtype=np.uint8)

    if labels is not None:
        legend = make_legend(
            res, patch_height, text_height,
            [f"{int(round(a)):d}" for a in labels],
            labels, use_log_scale)
        img[res[1]:(res[1] + legend_height), :] = legend
    # feed accumulated events into frequency cam algo
    get_fc_image = True
    remainder, events_this_slice = find_events_in_slice(ts, last_events)
    if get_fc_image:
        for evs in events_this_slice:
            fc_algo.process_events_no_callback(evs, offset)
    # pull resultant frequency map
    last_time = last_time if len(events_this_slice) == 0 \
        else events_this_slice[-1]['t'][-1]
    fc_raw_freq_map = fc_algo.make_frequency_map(last_time)
    fc_freq_map = scale_img(crop_map(fc_raw_freq_map), scale)

    raw_bg_img = make_bg_image(events_this_slice, orig_res)
    bg_img = scale_img(crop_map(raw_bg_img), scale)

    # write fc image into full frame
    img[0:res[1], :] = make_fc_image(fc_freq_map, use_log_scale, bg_img)

    nz_idx = freq_map > 0  # indices of non-zero elements of frequency map
    fname = str(Path(args.output_dir) / f"frame_{frame_count:05d}.jpg")

    if nz_idx.sum() > 0 or np.count_nonzero(raw_bg_img) > 0:
        img_scaled = make_color_image(freq_map, freq_range, use_log_scale)
        bg_img[nz_idx, :] = img_scaled[nz_idx, :]  # paint over bg image
        # lower half is metavision image
        img[(res[1] + legend_height):(2 * res[1] + legend_height), :] = bg_img
        if frame_count % 1 == 0:
            print('writing image: ', frame_count)
        cv2.imwrite(fname, img)
    else:
        print('writing empty image: ', fname)
        cv2.imwrite(fname, np.zeros_like(freq_map, dtype=np.uint8))

    last_events = remainder  # remove consumed events
    frame_count += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='compare Metavision SDK vs frequency cam.')
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
    parser.add_argument('--diff_thresh',
                        help='MV time period diff thresh (usec)',
                        default=100, type=int)
    parser.add_argument('--filter_length',
                        help='MV min num periods for valid detection',
                        default=1, type=int)
    parser.add_argument('--output_dir',
                        help='name of output directory',
                        default='mv_fc_frames')
    parser.add_argument('--log_scale', action='store_true',
                        required=False, help='color frequency on log scale')
    parser.set_defaults(log_scale=False)
    parser.add_argument('--debug_x', required=False, type=int,
                        help='x-coordinate of debug pixel', default=-1)
    parser.add_argument('--debug_y', required=False, type=int,
                        help='y-coordinate of debug pixel', default=-1)
    parser.add_argument('--timeout_cycles', required=False, type=int,
                        help='freq cam number of cycles before pixel timeout',
                        default=2)
    parser.add_argument('--cutoff_period',
                        help='number of events for cutoff period (>4)',
                        default=5, type=int)
    parser.add_argument('--labels', '-l', nargs='+',
                        help='labels for frequency', type=float)
    parser.add_argument('--tlx', type=int, default=None,
                        help='top left crop x')
    parser.add_argument('--tly', type=int, default=None,
                        help='top left crop y')
    parser.add_argument('--brx', type=int, default=None,
                        help='bottom right crop x')
    parser.add_argument('--bry', type=int, default=None,
                        help='bottom right crop y')
    parser.add_argument('--text_height', type=int, default=30,
                        help='height of text label field')
    parser.add_argument('--patch_height', type=int, default=50,
                        help='height of sample color patches')
    parser.add_argument('--scale', type=float, default=1.0,
                        help='how much to scale for better resolution')
    parser.add_argument('--font_scale', type=float, default=2.0,
                        help='how much to scale font size')
    parser.add_argument('--font_thickness', type=int, default=5,
                        help='how thick to make fonts')
    args = parser.parse_args()

    use_log_scale = args.log_scale
    labels = args.labels

    events, res, offset, _, _ = read_bag(
        args.bag, args.topic, use_sensor_time=False,
        converter=EventCDConverter())

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    mv_algo = FrequencyMapAsyncAlgorithm(
        width=res[0], height=res[1], filter_length=args.filter_length,
        min_freq=args.freq_min, max_freq=args.freq_max,
        diff_thresh_us=args.diff_thresh)
    mv_algo.update_frequency = args.update_freq
    mv_algo.set_output_callback(mv_write_image_cb)

    freq_range = np.array([args.freq_min, args.freq_max])

    fc_algo = FrequencyCam(width=res[0], height=res[1],
                           min_freq=args.freq_min, max_freq=args.freq_max,
                           cutoff_period=args.cutoff_period,
                           frame_timestamps=None,
                           frame_timeslice=None,
                           timeout_cycles=args.timeout_cycles,
                           debug_pixels=(args.debug_x, args.debug_y),
                           extra_args={})

    for evs in events:
        if evs.size > 0:
            last_events.append(evs)
            # update the algo with events. Sometimes no callback happens,
            # I believe when the frequency map does not change.
            # The callback has a "ts" time stamp argument but it seems
            # that a few events are included that are later than
            # that time stamp
            mv_algo.process_events(evs)
