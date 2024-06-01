#!/usr/bin/env python3
# -----------------------------------------------------------------------------
# Copyright 2024 Bernd Pfrommer <bernd.pfrommer@gmail.com>
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
"""Test python implementation of the Frequency Cam filter algorithm."""

import argparse
from pathlib import Path

import cv2  # noqa: I100
import event_bag_reader
import numpy as np

from frequency_cam import FrequencyCam  # noqa: I100  # noqa: I100

# global variables to keep track of current frame, events etc

args = None

last_events = []
last_time = None

frame_count = 0  # current frame
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


def make_color_image(freq_map, freq_range, use_log):
    fr_tf = np.log10(freq_range) if use_log else freq_range
    freq_map_tf = (
        np.log10(freq_map, out=np.zeros_like(freq_map), where=(freq_map > 0))
        if use_log
        else freq_map.copy()
    )
    r = fr_tf[1] - fr_tf[0]
    # scale image into range of 0..255
    scaled = cv2.convertScaleAbs(freq_map_tf, alpha=255.0 / r, beta=-fr_tf[0] * 255.0 / r)
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


def draw_labeled_rectangle(img, x, dx, patch_height, text_height, text_label, color):
    rec = (x, 0, dx, patch_height)
    y_margin = 10

    cv2.rectangle(img=img, rec=rec, color=color.tolist(), thickness=-1, lineType=cv2.LINE_8)
    font = cv2.FONT_HERSHEY_PLAIN
    font_scale = args.font_scale
    font_thickness = args.font_thickness
    textsize = cv2.getTextSize(text_label, font, fontScale=font_scale, thickness=font_thickness)[0]
    pos = (
        int(x + (dx - textsize[0]) / 2),
        int(patch_height + y_margin + text_height / 2),
    )

    cv2.putText(
        img=img,
        text=text_label,
        org=pos,
        fontFace=font,
        fontScale=font_scale,
        color=(0, 0, 0),
        thickness=font_thickness,
        lineType=cv2.LINE_AA,
    )


def make_legend(res, patch_height, text_height, text_labels, legend_values, color_map):
    img = 255 * np.ones([patch_height + text_height, res[0], 3], dtype=np.uint8)

    patches = np.zeros(len(legend_values), dtype=np.float32)
    lv_tf = np.log10(legend_values) if use_log_scale else legend_values
    for i, v in enumerate(lv_tf):
        patches[i] = v
    # scale patch frequencies to colors the same way as image
    fr_tf = np.log10(freq_range) if use_log_scale else freq_range
    r = fr_tf[1] - fr_tf[0]
    scaled = cv2.convertScaleAbs(patches, alpha=255.0 / r, beta=-fr_tf[0] * 255.0 / r)
    color_code = cv2.applyColorMap(scaled, cv2.COLORMAP_JET)

    dx = int(res[0] / color_code.shape[0])
    for i, c in enumerate(color_code):
        x = dx * i
        draw_labeled_rectangle(
            img,
            x,
            dx,
            patch_height=patch_height,
            text_height=text_height,
            text_label=text_labels[i],
            color=color_code[i, :].squeeze(),
        )

    return img


def crop_map(freq_map):
    if args.tlx is None or args.tly is None or args.brx is None or args.bry is None:
        return freq_map
    return freq_map[args.tly : args.bry, args.tlx : args.brx]  # noqa: 203


def scale_img(img, s):
    return cv2.resize(img, dsize=(int(s * img.shape[1]), int(s * img.shape[0])))


def freq_cam_frame_callback(
    frame_number, events_this_slice, freq_map_orig, freq_range, extra_args
):
    compare_to_other_method = False
    scale = args.scale
    orig_res = (freq_map_orig.shape[1], freq_map_orig.shape[0])
    freq_map = scale_img(crop_map(freq_map_orig), scale)
    res = (freq_map.shape[1], freq_map.shape[0])
    text_height = int(args.text_height * args.scale)
    patch_height = int(args.patch_height * args.scale)
    legend_height = 0 if labels is None else patch_height + text_height
    vert_res = res[1] * (2 if compare_to_other_method else 1)
    img = np.zeros([vert_res + legend_height, res[0], 3], dtype=np.uint8)

    if labels is not None:
        legend = make_legend(
            res,
            patch_height,
            text_height,
            [f'{int(round(a)):d}' for a in labels],
            labels,
            use_log_scale,
        )
        img[res[1] : (res[1] + legend_height), :] = legend  # noqa: E203

    raw_bg_img = make_bg_image(events_this_slice, orig_res)
    bg_img = scale_img(crop_map(raw_bg_img), scale)

    # write fc image into top half of frame
    img[0 : res[1], :] = make_fc_image(freq_map, use_log_scale, bg_img)  # noqa: E203

    nz_idx = freq_map > 0  # indices of non-zero elements of frequency map
    fname = str(Path(args.output_dir) / f'frame_{frame_number:05d}.jpg')

    if nz_idx.sum() > 0 or np.count_nonzero(raw_bg_img) > 0:
        img_scaled = make_color_image(freq_map, freq_range, use_log_scale)
        bg_img[nz_idx, :] = img_scaled[nz_idx, :]  # paint over bg image
        if compare_to_other_method:
            img[(res[1] + legend_height) : (2 * res[1] + legend_height), :] = bg_img  # noqa: E203
        if frame_number % 1 == 0:
            num_events = sum(ev.shape[0] for ev in events_this_slice)
            print(f'writing image with {num_events} events to {fname}')
        cv2.imwrite(fname, img)
    else:
        print('writing empty image: ', fname)
        cv2.imwrite(fname, np.zeros_like(freq_map, dtype=np.uint8))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='run frequency camera from bag.')
    parser.add_argument('--bag', '-b', required=True, help='bag file to read events from')
    parser.add_argument('--topic', help='Event topic to read', default='/event_camera/events')
    parser.add_argument('--freq_min', help='min frequency (hz)', default=200, type=float)
    parser.add_argument('--freq_max', help='max frequency (hz)', default=300, type=float)
    parser.add_argument('--update_freq', help='callback frequency (hz)', default=100, type=float)
    parser.add_argument('--output_dir', help='name of output directory', default='fc_frames')
    parser.add_argument(
        '--log_scale', action='store_true', required=False, help='color on log scale'
    )
    parser.set_defaults(log_scale=False)
    parser.add_argument('--debug_x', type=int, help='x-coordinate of debug pixel', default=-1)
    parser.add_argument('--debug_y', type=int, help='y-coordinate of debug pixel', default=-1)
    parser.add_argument(
        '--timeout_cycles',
        type=int,
        help='freq cam number of cycles before pixel timeout',
        default=2,
    )
    parser.add_argument(
        '--cutoff_period', help='number of events for cutoff period (>4)', default=5, type=int
    )
    parser.add_argument(
        '--frame_timeslice', help='frame time slice (in seconds)', default=0.01, type=float
    )
    parser.add_argument('--labels', '-l', nargs='+', help='labels for frequency', type=float)
    parser.add_argument('--tlx', type=int, default=None, help='top left crop x')
    parser.add_argument('--tly', type=int, default=None, help='top left crop y')
    parser.add_argument('--brx', type=int, default=None, help='bottom right crop x')
    parser.add_argument('--bry', type=int, default=None, help='bottom right crop y')
    parser.add_argument('--text_height', type=int, default=30, help='height of text label field')
    parser.add_argument(
        '--patch_height', type=int, default=50, help='height of sample color patches'
    )
    parser.add_argument(
        '--scale', type=float, default=1.0, help='how much to scale image for better resolution'
    )
    parser.add_argument(
        '--font_scale', type=float, default=2.0, help='how much to scale font size'
    )
    parser.add_argument('--font_thickness', type=int, default=5, help='how thick to make fonts')
    args = parser.parse_args()

    use_log_scale = args.log_scale
    labels = args.labels

    events, res, _ = event_bag_reader.read_events(args.bag, args.topic)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    freq_range = np.array([args.freq_min, args.freq_max])

    freq_cam = FrequencyCam(
        width=res[0],
        height=res[1],
        min_freq=args.freq_min,
        max_freq=args.freq_max,
        cutoff_period=args.cutoff_period,
        frame_timestamps=None,
        frame_timeslice=args.frame_timeslice,
        timeout_cycles=args.timeout_cycles,
        debug_pixels=(args.debug_x, args.debug_y),
        extra_args={},
    )

    freq_cam.set_frame_callback(freq_cam_frame_callback)

    for evs in events:
        if evs.size > 0:
            last_events.append(evs)
            freq_cam.process_events_to_frames(evs)
