import wx
import numpy as np
from numba import cuda
from soft_nudge.utils import calculate_limit_rect, split_frame_into_ud_lr
from soft_nudge.animation_caching import BakedAnimationFormat, BakedAnimationData
from soft_nudge.soft_nudge_cpu import render_frame as render_frame_cpu
from soft_nudge.soft_nudge_cuda import render_frame as render_frame_gpu


class BakedAnimation:
    def __init__(
        self,
        fps: int,
        frame_dimensions: tuple[int, int],
        frame_count: int,
        data_object: BakedAnimationData,
    ) -> None:
        self.fps = fps
        self.frame_dimensions = frame_dimensions
        self.frame_count = frame_count
        self.data_object = data_object

    @classmethod
    def load_from_file(cls, binary_file_object):
        data_object = BakedAnimationData.create_from_file(binary_file_object)
        frame_count = data_object.frame_count
        fps = data_object.fps
        frame_dimensions = (data_object.width, data_object.height)
        return cls(fps, frame_dimensions, frame_count, data_object)

    def write_to_file(self, binary_file_object) -> None:
        self.data_object.write_to_file(binary_file_object)

    def get_frame(self, frame_index: int) -> tuple[np.ndarray, np.ndarray]:
        frame = self.data_object.get_raw_frame(frame_index)
        cdata = frame[:, :, 0:3]
        adata = frame[:, :, 3]
        return cdata, adata


def bake_animation(
    color_rgba: tuple[int, int, int, int],
    size: tuple[int, int],
    fps: int,
    duration: float,
    anim_period: float,
    anim_amplitude: float,
    anim_undulation_frequency: float,
    trend_split: float = 0.6,
    flat_time_pct: float = 0.4,
    pop_in_outside: float = 1.15,
    pop_in_inside: float = 0.9,
    compression_method: BakedAnimationFormat = BakedAnimationFormat.stencil_format,
    force_cpu: bool = False,
) -> BakedAnimation:
    """Creates a baked soft-nudge animation that can be replayed later.
    The device used depends on the availability of CUDA and the value of `force_cpu`.
    """
    render_method = render_frame_cpu
    if cuda.is_available() and not force_cpu:
        render_method = render_frame_gpu

    frame_count = int(duration * fps)
    limit_rect = calculate_limit_rect(
        *size,
        anim_amplitude,
        anim_period,
        anim_undulation_frequency,
        pop_in_inside,
    )

    seconds_per_frame = 1 / fps
    nano_seconds_per_frame = seconds_per_frame * 1_000_000_000

    raw_frames_ud = np.zeros((frame_count, int(limit_rect[1] * 2), int(size[0]), 4))
    raw_frames_lr = np.zeros(
        (frame_count, int(size[1] - limit_rect[1] * 2), int(limit_rect[0] * 2), 4)
    )

    for f in range(frame_count):
        frame = render_method(
            *size,
            color_rgba,
            anim_period,
            anim_amplitude,
            anim_undulation_frequency,
            duration,
            trend_split,
            flat_time_pct,
            f * nano_seconds_per_frame,
            pop_in_outside,
            pop_in_inside,
            *limit_rect,
        )

        frame_ud, frame_lr = split_frame_into_ud_lr(
            frame,
            *limit_rect,
        )

        raw_frames_ud[f, :, :, :] = frame_ud
        raw_frames_lr[f, :, :, :] = frame_lr

    data_object = BakedAnimationData.create_from_raw_frames(
        raw_frames_ud,raw_frames_lr, fps, compression_method
    )
    return BakedAnimation(fps, size, frame_count, data_object)
