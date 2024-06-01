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

import math

import numpy as np


def filter_noise(d, dt_cutoff, dt_dead):
    """Remove high-freq temporal noise from cam ultra-fast speed settings."""
    t = d[:, 0] * 1e-9
    p = d[:, 1].astype(np.int32) * 2 - 1
    d_f = []
    skip_counter = 0
    for i in range(4, d.shape[0]):
        if skip_counter == 0:
            d_f.append((d[i - 4, 0], d[i - 4, 1]))
        else:
            skip_counter -= 1
        if (
            p[i - 2] < 0
            and p[i - 1] > 0
            and t[i - 1] - t[i - 2] < dt_cutoff
            and t[i - 2] - t[i - 3] > dt_dead
        ):
            skip_counter = 4
    n_filt = d.shape[0] - len(d_f)
    print(f'filtered {n_filt} of {d.shape[0]} events ({n_filt/d.shape[0]}%)')
    return np.array(d_f)


def find_periods_baseline(data, L, T):
    last_t = [None, None]
    last_p = None
    period = [[], []]
    times = [[], []]
    cnt = int(0)
    period_omit = 2 * int(round(T))
    for i, d in enumerate(data):
        t = d[0]
        p = d[1]
        if last_p is not None and last_p != p:
            if p:
                if last_t[0] is not None and cnt > period_omit:
                    period[0].append(t - last_t[0])
                    times[0].append((t, last_t[0], 1e9 * L[i]))
                last_t[0] = t
            else:
                if last_t[1] is not None and cnt > period_omit:
                    period[1].append(t - last_t[1])
                    times[1].append((t, last_t[1], 1e9 * L[i]))
                last_t[1] = t
        last_p = p
        cnt += 1
    return (
        1e-9 * np.array(times[0]),
        1e-9 * np.array(period[0]),
        1e-9 * np.array(times[1]),
        1e-9 * np.array(period[1]),
    )


def compute_alpha_for_cutoff(cutoff_period):
    omega = 2 * np.pi / cutoff_period
    return (1 - math.sin(omega)) / math.cos(omega)


def compute_beta_for_cutoff(cutoff_period):
    omega = 2 * np.pi / cutoff_period
    x = 2 - math.cos(omega)
    return x - math.sqrt(x**2 - 1)


def filter_iir(x, alpha, beta, start_x):
    # x is polarity, should be +1 or -1
    a1 = alpha + beta
    a2 = -alpha * beta
    a3 = 0.5 * (1 + beta)
    x_cum = []
    last_x = start_x[0]  # one lag back
    last_last_x = start_x[1]  # two lags back
    last_p = 0  # could also use x[0]
    for p in x:
        x_cum.append(a1 * last_x + a2 * last_last_x + a3 * (p - last_p))
        last_last_x = last_x
        last_x = x_cum[-1]
        last_p = p
    return np.array(x_cum)


def reconstruct(data, T):
    alpha = compute_alpha_for_cutoff(T)
    beta = compute_beta_for_cutoff(T)
    dL = np.where(data[:, 1] == 0, -1, 1)
    return filter_iir(dL, alpha, beta, (0, 0))


def find_periods_filtered(data, L_all, T):
    t_all = data[:, 0]

    # drop the first 2 * cutoff period
    period_omit = 2 * int(round(T))
    t_prev, L_prev = None, None
    t_flip, t_flip_interp = [], []
    dt, dt_interp = [], []
    upper_half = False
    cnt = 0
    for t, L in zip(t_all, L_all):
        if upper_half and L < 0:
            if cnt > period_omit:
                # regular periods
                if t_flip:  # has element
                    dt.append(t - t_flip[-1][0])
                    t_flip.append((t, t_flip[-1][0], L * 1e9))
                else:
                    t_flip.append((t, t, L * 1e9))
                # ------ interpolated periods
                if L_prev:
                    dt_int = t - t_prev
                    dL_int = L - L_prev
                    t_interp = t_prev - dt_int * L_prev / dL_int
                    if t_flip_interp:
                        dt_interp.append(t_interp - t_flip_interp[-1][0])
                        t_flip_interp.append((t_interp, t_flip_interp[-1][0], 0))
                    else:
                        t_flip_interp.append((t_interp, t_interp, 0))
        upper_half = L > 0
        t_prev = t  # for interpolation
        L_prev = L  # for interpolation
        cnt += 1

    return (
        (
            1e-9 * np.array(t_flip),
            1e-9 * np.array(dt),
            1e-9 * np.array(t_flip_interp),
            1e-9 * np.array(dt_interp),
        ),
        t_all,
        L_all,
    )
