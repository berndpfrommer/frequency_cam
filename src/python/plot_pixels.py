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
"""Plot filter transfer functions etc."""

import argparse
import math
import sys

import core_filtering
import event_bag_reader
import matplotlib.pyplot as plt
import numpy as np


def compute_H_alpha_detrend_sq(omega, alpha):
    return (2 - 2 * math.cos(omega)) / (1 - 2 * alpha * math.cos(omega) + alpha * alpha)


def compute_H_beta_norm_sq(omega, beta):
    return (1 - beta) ** 2 / (1 + beta**2 - 2 * beta * np.cos(omega))


def compute_alpha_for_max(omega):
    co = math.cos(omega)
    alpha = 2 - co - math.sqrt(3 - 4 * co + co**2)
    return alpha


def compute_cos_omega_max(alpha, beta):
    a = alpha + 1 / alpha
    b = beta + 1 / beta
    cos_omega_max = 1 - math.sqrt(1 - 0.5 * a - 0.5 * b + 0.25 * a * b)
    return cos_omega_max


def compute_H_max_sq(alpha, beta):
    y = 2 * compute_cos_omega_max(alpha, beta)
    a = alpha + 1 / alpha
    b = beta + 1 / beta
    H_max_sq = (1 + beta) ** 2 / (4 * alpha * beta) * (2 - y) / ((y - a) * (y - b))
    return H_max_sq


def set_tick_font_size(ax, fs):
    """Set tick font sizes for x and y."""
    for tl in ax.get_xticklabels() + ax.get_yticklabels():
        tl.set_fontsize(fs)


def set_font_size(axs, fontsize):
    for ax in axs:
        ax.title.set_fontsize(fontsize)
        ax.yaxis.label.set_fontsize(fontsize)
        ax.xaxis.label.set_fontsize(fontsize)
        set_tick_font_size(ax, fontsize)


def plot_pixel(args, data):
    """Plot pixel data."""
    # if filter parameters are specified, apply high-noise filter
    if args.filter_pass_dt > 0:
        data = core_filtering.filter_noise(data, args.filter_pass_dt, args.filter_dead_dt)
    num_on = np.count_nonzero(data[:, 1] > 0)
    num_off = max(data.shape[0] - num_on, 1)
    on_ratio = args.on_ratio if args.on_ratio else num_on / num_off
    dx = 2 * data[:, 1].astype(np.float64) - 1
    dx_detrend = np.where(data[:, 1] == 1, 1, -on_ratio)
    x = np.cumsum(dx)
    x_detrend = np.cumsum(dx_detrend)
    num_graphs = 3
    fig, axs = plt.subplots(
        nrows=num_graphs, ncols=1, sharex=True, gridspec_kw={'height_ratios': [1, 2, 2]}
    )
    fontsize = 30
    fontsize_legend = int(fontsize * 0.55)
    t = (data[:, 0] - (data[0, 0] if args.t_at_zero else 0)) * 1e-6
    # ---- polarity
    axs[0].plot(t, dx, 'x', color='r', label='raw events')
    axs[0].yaxis.set_ticks((-1, 1))
    axs[0].set_ylabel('polarity')
    axs[0].set_ylim(-1.4, 1.4)
    axs[0].legend(loc='center right', prop={'size': fontsize_legend})

    # ---- naive integration
    axs[1].plot(t, x, 'o-', label='integrated: $C_{ON} = 1$, $C_{OFF} = 1$')
    axs[1].set_ylabel(r'$\tilde{L}(t)$')
    axs[1].legend(loc='lower right', prop={'size': fontsize_legend})
    # ---- detrended integration
    axs[2].plot(
        t, x_detrend, 'o-', label='integrated: $C_{ON}$ = 1, $C_{OFF}$ = ' + f'{on_ratio:.2f}'
    )
    axs[2].set_ylabel(r'$\tilde{L}(t)$')
    axs[2].set_xlabel('time [s]')
    axs[2].legend(loc='lower right', prop={'size': fontsize_legend})
    # set font size for title, axis labels and ticks
    set_font_size(axs, fontsize)
    # plt.subplots_adjust(left=0.13, right=0.98, top=0.97, bottom=0.1)
    # plt.tight_layout()
    fig.suptitle('naive pixel reconstruction', fontsize=fontsize)
    plt.show()


def filter_pixel(args, data):
    """Apply filter and plot data."""
    # if filter parameters are specified, apply high-noise filter
    if args.filter_pass_dt > 0:
        data = core_filtering.filter_noise(data, args.filter_pass_dt, args.filter_dead_dt)
    num_graphs = 1
    fig, axs = plt.subplots(
        nrows=num_graphs, ncols=1, sharex=True, gridspec_kw={'height_ratios': [1]}
    )
    axs = (axs,)
    fontsize = 40
    # t = (data[:, 0] - (data[0, 0] if args.t_at_zero else 0)) * 1e-6
    t = data[:, 0] * 1e-6
    N = data.shape[0]
    num_on = np.count_nonzero(data[:, 1])
    num_off = np.count_nonzero(data[:, 1] == 0)
    print(f'num on:   {num_on:5d} {num_on / N * 100:.2f}%')
    print(f'num off:  {num_off:5d} {num_off / N * 100:.2f}%')
    T = args.cutoff_period
    alpha = core_filtering.compute_alpha_for_cutoff(T)
    beta = core_filtering.compute_beta_for_cutoff(T)
    dx = np.where(data[:, 1] == 0, -1, 1)
    L = core_filtering.filter_iir(dx, alpha, beta, (0, 0))
    label = r'$T_{cut}=$' + f'{int(T):d}'
    axs[0].step(t, L, 'o-', where='post', label=label, linewidth=3)
    axs[0].set_ylabel(r'$\tilde{L}(t)$')

    axs[-1].legend(loc='upper right', prop={'size': int(fontsize * 0.75)})
    axs[-1].plot((t[0], t[-1]), (0, 0), '-', color='k')
    axs[-1].set_xlabel('time [s]')
    # plt.subplots_adjust(left=0.13, right=0.98, top=0.97, bottom=0.15)
    fig.suptitle('reconstructed pixel brightness', fontsize=fontsize)
    # set font size for title, axis labels and ticks
    set_font_size(axs, fontsize)
    plt.show()


def plot_pixels(args, pixel_events, res):
    """Plot pixels with naive integration."""
    for i, ev in enumerate(pixel_events):
        print(f'plotting pixel {i} with {ev.shape[0]} events')
        s = args.skip_events_plot
        n = min(s + args.num_events_plot, ev.shape[0])
        plot_pixel(args, ev[s:n, ...])


def apply_filter(args, pixel_events, res):
    """Plot reconstructed pixel signal."""
    for ev in pixel_events:
        s = args.skip_events_plot
        n = min(s + args.num_events_plot, ev.shape[0])
        filter_pixel(args, ev[s:n, :])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='visualize how filter digital works.')
    parser.add_argument('--bag', '-b', default=None, required=False, help='name of rosbag')
    parser.add_argument('--topic', '-t', default='/event_camera/events', help='ros topic')
    parser.add_argument(
        '--px',
        action='append',
        default=[],
        required=False,
        type=int,
        help='x coordinate of pixel to plot (usable multiple times)',
    )
    parser.add_argument(
        '--py',
        action='append',
        default=[],
        required=False,
        type=int,
        help='y coordinate of pixel to plot (usable multiple times)',
    )
    parser.add_argument(
        '--max_read',
        '-m',
        default=sys.maxsize,
        required=False,
        type=int,
        help='how many events to read (total)',
    )
    parser.add_argument(
        '--t_at_zero', '-0', action='store_true', required=False, help='start time at zero'
    )
    parser.set_defaults(t_at_zero=False)

    parser.add_argument(
        '--filter', action='store_true', required=False, help='apply filter to signal'
    )
    parser.set_defaults(filter=False)

    parser.add_argument(
        '--filter_pass_dt', default=0e-6, type=float, help='passing dt between OFF followed by ON'
    )
    parser.add_argument(
        '--filter_dead_dt',
        default=None,
        type=float,
        help='filter dt between event preceeding OFF/ON pair',
    )
    parser.add_argument(
        '--on_ratio',
        default=None,
        type=float,
        help='fraction of ON/OFF events for detrending',
    )
    parser.add_argument('--skip', default=0, type=int, help='number of events to skip on read')
    parser.add_argument('--cutoff_period', default=30.0, type=float, help='T_cut, see paper')
    parser.add_argument(
        '--num_events_plot',
        '-n',
        default=10000000000,
        type=int,
        help='number of events to plot',
    )
    parser.add_argument(
        '--skip_events_plot',
        '-s',
        default=0,
        type=int,
        help='number of events to skip on plot',
    )

    args = parser.parse_args()
    pixels = list(zip(args.px, args.py))

    if len(pixels) == 0:
        raise Exception('must specify some pixels!')
    if args.bag is None:
        raise Exception('must specify bag!')
    if args.filter_dead_dt is None:
        args.filter_dead_dt = args.filter_pass_dt

    print('reading event packets...')
    events, res, _ = event_bag_reader.read_events(args.bag, args.topic, args.max_read, args.skip)
    if len(events) == 0:
        raise Exception('no messages found in bag, is topic correct?')
    print(f'number of event packets read: {len(events)}')
    print(f'filtering events for pixels: {pixels}')
    filtered_events = event_bag_reader.filter_events(events, pixels)
    print(f'number of event packets for selected pixels: {len(filtered_events)} event packets')
    if args.filter:
        apply_filter(args, filtered_events, res)
    else:
        plot_pixels(args, filtered_events, res)
