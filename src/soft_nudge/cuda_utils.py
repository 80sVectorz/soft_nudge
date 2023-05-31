from numba import cuda
from math import sin, cos, sqrt, copysign, pi


@cuda.jit(device=True)
def general_sine_wave(x, period, lower, upper):
    # https://www.desmos.com/calculator/bd0o21n3ht
    return lower + (sin(pi / period * x) + 1) / 2 * (upper - lower)


@cuda.jit(device=True)
def lerp(a, b, t):
    return a + (b - a) * t


@cuda.jit(device=True)
def recursive_lerp(a, b, t, n):
    i = 1
    out = lerp(a, b, t)
    while i < n:
        out = lerp(a, b, out)
        i += 1
    return out


@cuda.jit(device=True)
def calc_circle(px, py, r, ox, oy):
    d = sqrt((ox - px) ** 2 + (oy - py) ** 2)
    res = d <= r
    dnorm = 1.0 - d / r
    return res, dnorm


@cuda.jit()
def render_graphics_cuda_test(image, w, h, t):
    height = image.shape[0]
    width = image.shape[1]

    startX = cuda.blockDim.x * cuda.blockIdx.x + cuda.threadIdx.x  # type: ignore
    startY = cuda.blockDim.y * cuda.blockIdx.y + cuda.threadIdx.y  # type: ignore
    gridX = cuda.gridDim.x * cuda.blockDim.x  # type: ignore
    gridY = cuda.gridDim.y * cuda.blockDim.y  # type: ignore

    for x in range(startX, width, gridX):
        for y in range(startY, height, gridY):
            px = x
            py = y

            pixel_r = 0
            pixel_g = 0
            pixel_b = 0
            pixel_a = 0

            pcenter_x = w / 2
            pcenter_y = h / 2
            aa = 1.0
            t1 = t / 1_000_000_00
            c1r = 100
            c1ir = c1r * 0.75
            c1, d1 = calc_circle(px, py, c1r, pcenter_x, pcenter_y)
            c1i, d1i = calc_circle(px, py, c1ir, pcenter_x, pcenter_y)

            c2r = c1ir * 0.5
            c2offset_x, c2offset_y = cos(t1), sin(t1)
            ox = pcenter_x + c2offset_x * c2r
            oy = pcenter_y + c2offset_y * c2r

            c2, d2 = calc_circle(px, py, c2r, ox, oy)

            # Formulas: https://www.desmos.com/calculator/krzaba0bno
            if c1:
                aa_margin = 1.0 / float(c1r) * aa
                aa_alpha = min(d1, aa_margin) / aa_margin
                alpha_buffer = int(160 * (aa_alpha))
                pixel_r = 36
                pixel_g = 173
                pixel_b = 243
                pixel_a = alpha_buffer
            if c1i:
                aa_margin = 1.0 / float(c1ir) * aa
                aa_alpha = min(d1i, aa_margin) / aa_margin
                pixel_a = int(pixel_a * (1.0 - aa_alpha))
            if c2:
                aa_margin = 1.0 / float(c2r) * aa
                aa_alpha = min((d2, aa_margin)) / aa_margin
                alpha_buffer = max(int(160 * (aa_alpha)), pixel_a)
                pixel_r = 36
                pixel_g = 173
                pixel_b = 243
                pixel_a = alpha_buffer
            image[y, x, 0] = pixel_r
            image[y, x, 1] = pixel_g
            image[y, x, 2] = pixel_b
            image[y, x, 3] = pixel_a
