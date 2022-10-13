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
"""Script for plotting pixel debugging data."""

import numpy as np
import matplotlib.pyplot as plt
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='plot debugging data')
    parser.add_argument(
        '--file', '-f', nargs='+',
        required=True, help='frequency input file(s) with data (can be multiple files)')
    parser.add_argument(
        '--readout', '-r', nargs='+',
        required=True, help='readout input file(s) with data (can be multiple files)')
    parser.add_argument('--freq_min', help='min frequency (hz)',
                        default=200, type=float)
    parser.add_argument('--freq_max', help='max frequency (hz)',
                        default=300, type=float)

    args = parser.parse_args()

    dt_max = 1.0 / args.freq_min
    dt_min = 1.0 / args.freq_max

    _, axs = plt.subplots(nrows=2, ncols=1, sharex=True)
    for f, r in zip(args.file, args.readout):
        freq = np.loadtxt(f)
        readout = np.loadtxt(r)
        t = freq[:, 0] * 1e-6
        # ------- reconstruction
        axs[0].plot(t, freq[:, 2], '-o', label=f'{f}')
        axs[0].plot((t[0], t[-1]), (0, 0), color='black')

        # -------- period and readout
        period = freq[:, 6]
        axs[1].plot(t, period, '-x', label=f'period {f}')

        t_r = readout[:, 0]
        vr = np.where(readout[:, 1] < 1e-6, 1e10, readout[:, 1])
        dt_r = np.zeros_like(t_r)
        np.divide(1.0, readout[:, 1], out=dt_r, where=readout[:, 1] > 1e-6)
        axs[1].plot(t_r, dt_r, '-.o', markerfacecolor='none',
                    label=f'dt(readout) {f}')

    axs[0].legend()
    axs[1].legend()
    axs[1].set_ylim((dt_min, dt_max))
    plt.show()
