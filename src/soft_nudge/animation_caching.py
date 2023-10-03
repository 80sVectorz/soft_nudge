import numpy as np
from enum import Enum
from soft_nudge.utils import combine_ud_lr_into_frame
from .baked_animation_data_processors.data_processors import (
    BakedAnimationDataProcessorRaw,
    BakedAnimationDataProcessor,
)
from .baked_animation_data_processors.data_processor_stencil_format import (
    BakedAnimationDataProcessorStencilFormat,
)


class BakedAnimationFormat(Enum):
    raw_format = 0
    stencil_format = 1


BAKED_ANIMATION_DATA_PROCESSOR_LUT: dict[
    BakedAnimationFormat, BakedAnimationDataProcessor
] = {
    BakedAnimationFormat.raw_format: BakedAnimationDataProcessorRaw,
    BakedAnimationFormat.stencil_format: BakedAnimationDataProcessorStencilFormat,
}


class BakedAnimationData:
    def __init__(
        self,
        compressed_frame_data_ud: np.ndarray,
        compressed_frame_data_lr: np.ndarray,
        data_processor: BakedAnimationDataProcessor,
        header_values: dict[str, int | float],
    ) -> None:
        self.compressed_frame_data_ud = compressed_frame_data_ud
        self.compressed_frame_data_lr = compressed_frame_data_lr
        self.data_processor = data_processor
        self.header_values = header_values
        self.width = header_values["width"]
        self.height = header_values["height"]
        self.fps = header_values["fps"]
        self.frame_count = header_values["frame_count"]

    @classmethod
    def create_from_raw_frames(
        cls,
        raw_frames_ud: np.ndarray,
        raw_frames_lr: np.ndarray,
        fps: int,
        target_data_processor: BakedAnimationFormat,
    ):
        data_processor = BAKED_ANIMATION_DATA_PROCESSOR_LUT[target_data_processor]
        compressed_frame_data, header_values = data_processor.compress_raw_frames(
            raw_frames_ud, raw_frames_lr, fps
        )
        return cls(*compressed_frame_data, data_processor, header_values)

    @classmethod
    def create_from_bytes(cls, bytes_object):
        target_data_format = BakedAnimationFormat(bytes_object[0])

        data_processor = BAKED_ANIMATION_DATA_PROCESSOR_LUT[target_data_format]
        (
            compressed_frame_data,
            header_values,
        ) = data_processor.read_compressed_animation_data(bytes_object)
        return cls(*compressed_frame_data, data_processor, header_values)

    @classmethod
    def create_from_file(cls, binary_file_object):
        data_bytes = binary_file_object.read()
        target_data_format = BakedAnimationFormat(data_bytes[0])

        data_processor = BAKED_ANIMATION_DATA_PROCESSOR_LUT[target_data_format]
        (
            compressed_frame_data,
            header_values,
        ) = data_processor.read_compressed_animation_data(data_bytes)
        return cls(*compressed_frame_data, data_processor, header_values)

    def get_raw_frame(self, frame_index: int) -> np.ndarray:
        return combine_ud_lr_into_frame(
            *self.data_processor.decompress_frame(
                frame_index,
                self.compressed_frame_data_ud,
                self.compressed_frame_data_lr,
                self.header_values,
            )
        )

    def write_to_file(self, binary_file_object):
        binary_file_object.write(
            self.data_processor.compressed_animation_to_bytes(
                self.compressed_frame_data_ud,
                self.compressed_frame_data_lr,
                self.header_values,
            )
        )
