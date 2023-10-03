from numba import njit
from math import sin, cos, sqrt, copysign, pi


@njit(cache=True)
def general_sine_wave(x, period, lower, upper):
    # https://www.desmos.com/calculator/bd0o21n3ht
    return lower + (sin(pi / period * x) + 1) / 2 * (upper - lower)


@njit(cache=True)
def lerp(a, b, t):
    return a + (b - a) * t


@njit(cache=True)
def recursive_lerp(a, b, t, n):
    i = 1
    out = lerp(a, b, t)
    while i < n:
        out = lerp(a, b, out)
        i += 1
    return out


@njit(cache=True)
def calc_circle(px, py, r, ox, oy):
    d = sqrt((ox - px) ** 2 + (oy - py) ** 2)
    res = d <= r
    dnorm = 1.0 - d / r
    return res, dnorm
