from numba import njit
import numpy as np
from math import sin, cos, sqrt, copysign
from soft_nudge.cpu_utils import lerp, general_sine_wave


@njit(cache=True)
def fsx(x, y, period, amplitude, w, h, t):
    return amplitude * abs(h / 2) * (sin(x / (w / 2 / period) + t) * sin(t)) + x + h / 2


@njit(cache=True)
def fsy(x, y, period, amplitude, w, h, t):
    return amplitude * abs(w / 2) * (sin(y / (h / 2 / period) + t) * sin(t)) + y + w / 2


@njit(cache=True)
def fx(x, y, period, amplitude, w, h, t):
    return min(
        abs(fsx(x, y, period, amplitude, w, -h, t) - (x + y)),
        abs(fsx(x, y, period, amplitude, w, h, t) - (x + y)),
    )


@njit(cache=True)
def fy(x, y, period, amplitude, w, h, t):
    return min(
        abs(fsy(x, y, period, amplitude, -w, h, t) - (x + y)),
        abs(fsy(x, y, period, amplitude, w, h, t) - (x + y)),
    )


@njit(cache=True)
def border_effect_f(x, y, period, amplitude, w, h, t):
    if (
        fsy(x, y, period, amplitude, -w, h, t) - y < x
        and fsy(x, y, period, amplitude, w, h, t) - y > x
        and fsx(x, y, period, amplitude, w, -h, t) - x < y
        and fsx(x, y, period, amplitude, w, h, t) - x > y
    ):
        return fx(x, y, period, amplitude, w, h, t) * fy(
            x, y, period, amplitude, w, h, t
        )
    return 0.0


@njit(cache=True)
def slope(x, s):
    return min(2 ** (5 * ((x / s) ** 3)), 1)


@njit(cache=True)
def render_frame(
    w,
    h,
    rgba,
    period,
    amplitude,
    undulation_frequency,
    duration,
    trend_split,
    flat_time_pct,
    t,
    pop_in_outside,
    pop_in_inside,
    limit_rect_x1,
    limit_rect_y1,
    limit_rect_x2,
    limit_rect_y2,
):
    frame = np.zeros((h, w, 4), dtype=np.uint8)

    tseconds = t / 1_000_000_000
    undulation_time = tseconds * 2 * np.pi * undulation_frequency
    xt = tseconds
    duration_a = duration * trend_split
    duration_ms = (duration - duration_a) * flat_time_pct
    duration_b = duration - duration_a - duration_ms

    # progress formula: https://www.desmos.com/calculator/orf5s78po2
    fa = slope(xt - duration_a, duration_a)
    fb = -slope(xt - duration, duration_b) + 1
    xfade = -slope(xt - duration, duration_b + duration_ms) + 1

    progress = lerp(fb, fa, xfade)

    pcx = w / 2
    pcy = h / 2

    # Limit rect debug code
    # progress = 1
    # undulation_time = (np.pi/2)/(2*np.pi*undulation_frequency)

    for y in range(h):
        for x in range(w):
            px = x
            py = y

            if x == 0 and y == 0 and tseconds >= duration:
                # Trigger kill with this color: 101 110 100 are the codes for e n d in ascii: http://sticksandstones.kstrom.com/appen.html
                frame[y, x, 0] = 101
                frame[y, x, 1] = 110
                frame[y, x, 2] = 100
                frame[y, x, 3] = 255
                continue

            if (px > limit_rect_x1 and px < limit_rect_x2) and (
                py > limit_rect_y1 and py < limit_rect_y2
            ):
                # limit rect debug code
                # frame[y, x, 0] = 255
                # frame[y, x, 1] = 0
                # frame[y, x, 2] = 0
                # frame[y, x, 3] = 20
                continue

            pixel_r = 0
            pixel_g = 0
            pixel_b = 0
            pixel_a = 0

            a = border_effect_f(
                (px - pcx),
                (py - pcy),
                period,
                amplitude,
                w * lerp(pop_in_outside, pop_in_inside, progress),
                h * lerp(pop_in_outside, pop_in_inside, progress),
                undulation_time,
            )
            if a < 50:
                pixel_r = rgba[0]
                pixel_g = rgba[1]
                pixel_b = rgba[2]
                pixel_a = rgba[3]

            frame[y, x, 0] = pixel_r
            frame[y, x, 1] = pixel_g
            frame[y, x, 2] = pixel_b
            frame[y, x, 3] = pixel_a
    return frame


def get_bmp_data(
    w,
    h,
    rgba,
    period,
    amplitude,
    undulation_frequency,
    duration,
    trend_split,
    flat_time_pct,
    t,
    pop_in_outside,
    pop_in_inside,
    limit_rect_x1,
    limit_rect_y1,
    limit_rect_x2,
    limit_rect_y2,
):
    frame = render_frame(
        w,
        h,
        rgba,
        period,
        amplitude,
        undulation_frequency,
        duration,
        trend_split,
        flat_time_pct,
        t,
        pop_in_outside,
        pop_in_inside,
        limit_rect_x1,
        limit_rect_y1,
        limit_rect_x2,
        limit_rect_y2,
    )

    cdata = frame[:, :, 0:3]
    adata = frame[:, :, 3]

    return cdata, adata
