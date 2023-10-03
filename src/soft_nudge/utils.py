import numpy as np


def calculate_limit_rect(
    width: int,
    height: int,
    anim_amplitude: float,
    anim_period,
    anim_undulation_frequency: float,
    pop_in_inside: float,
) -> tuple[int, int, int, int]:
    t = (np.pi / 2) / (2 * np.pi * anim_undulation_frequency)
    limit_rect_x1 = (
        anim_amplitude * abs(width * pop_in_inside) * (np.sin(t) * np.sin(t))
    )
    limit_rect_y1 = (
        anim_amplitude * abs(height * pop_in_inside) * (np.sin(t) * np.sin(t))
    )
    limit_rect_x2 = (width * pop_in_inside) - limit_rect_x1
    limit_rect_y2 = (height * pop_in_inside) - limit_rect_y1

    offset_x = (width / 2) * (pop_in_inside - 1)
    offset_y = (height / 2) * (pop_in_inside - 1)

    limit_rect_x1 -= offset_x
    limit_rect_y1 -= offset_y
    limit_rect_x2 -= offset_x
    limit_rect_y2 -= offset_y
    return (
        int(limit_rect_x1),
        int(limit_rect_y1),
        int(limit_rect_x2),
        int(limit_rect_y2),
    )


def split_frame_into_ud_lr(
    frame: np.ndarray,
    limit_rect_x1: int,
    limit_rect_y1: int,
    limit_rect_x2: int,
    limit_rect_y2: int,
) -> tuple[np.ndarray, np.ndarray]:
    frame_ud = np.zeros(
        (int(limit_rect_y1 * 2), int(frame.shape[1]), 4), dtype=np.uint8
    )

    frame_lr = np.zeros(
        (int(frame.shape[0] - limit_rect_y1 * 2), int(limit_rect_x1 * 2), 4),
        dtype=np.uint8,
    )

    frame_ud[0:limit_rect_y1, :, :] = frame[0:limit_rect_y1, :, :]
    frame_ud[limit_rect_y1:, :, :] = frame[-limit_rect_y1:, :, :]

    frame_lr[:, 0:limit_rect_x1, :] = frame[
        limit_rect_y1:-limit_rect_y1, 0:limit_rect_x1, :
    ]
    frame_lr[:, limit_rect_x1:, :] = frame[
        limit_rect_y1:-limit_rect_y1, -limit_rect_x1:, :
    ]

    return frame_ud, frame_lr


def combine_ud_lr_into_frame(frame_ud:np.ndarray, frame_lr:np.ndarray) -> np.ndarray:
    width = frame_ud.shape[1]
    height = frame_ud.shape[0] + frame_lr.shape[0]

    limit_rect_x1 = frame_lr.shape[1] // 2
    limit_rect_y1 = frame_ud.shape[0] // 2

    frame = np.zeros((height, width, 4),dtype=np.uint8)

    frame[0:limit_rect_y1, :, :] = frame_ud[0:limit_rect_y1, :, :]
    frame[-limit_rect_y1:, :, :] = frame_ud[limit_rect_y1:, :, :]

    frame[limit_rect_y1:-limit_rect_y1, 0:limit_rect_x1, :] = frame_lr[
        :, 0:limit_rect_x1, :
    ]
    frame[limit_rect_y1:-limit_rect_y1:, -limit_rect_x1:, :] = frame_lr[
        :, limit_rect_x1:, :
    ]

    return frame
