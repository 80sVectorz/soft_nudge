import numpy as np
from .data_processors import BakedAnimationDataProcessor


class BakedAnimationDataProcessorStencilFormat(BakedAnimationDataProcessor):
    """Stencil based baked animation data processor class.
    Assumes single color for entire animation.
    Each pixel is either transparent or the given color.
    This should drastically improve storage costs.

    Base file structure:

    HEADER:
    - format: U-INT 8
    - width: U-INT 16
    - height: U-INT 16
    - limit_rect_x1: U-INT 16
    - limit_rect_y1: U-INT 16
    - limit_rect_x2: U-INT 16
    - limit_rect_y2: U-INT 16
    - frame_count: U-INT 16
    - fps: U-INT 8

    - R: U-INT 8
    - G: U-INT 8
    - B: U-INT 8
    - A: U-INT 8

    FRAME DATA:

    FRAME DATA:
    "Frame data is represented in bits. The mapping is defined below"

    - F_0...F_n
        - Y_UD_0...Y_UD_n
            - X_UD_0...X_UD_n
                - S: 1-BIT
        - Y_LR_0...Y_LR_n
            - X_LR_0...X_LR_n
                - S: 1-BIT

    "Every '1' maps to the RGBA color defined in the header and every '0' maps to fully transparent cutout"
    """

    frame_data_offset = 20
    data_format_id = 1

    @classmethod
    def read_format_header(cls, bytes_object) -> dict[str, int]:
        header_values = cls.read_base_header(bytes_object)
        header_values["R"] = bytes_object[16]
        header_values["G"] = bytes_object[17]
        header_values["B"] = bytes_object[18]
        header_values["A"] = bytes_object[19]

        return header_values

    @classmethod
    def create_format_header(
        cls, raw_frames_ud: np.ndarray, raw_frames_lr: np.ndarray, fps: int
    ) -> bytes:
        header_values = cls.create_base_header(raw_frames_ud, raw_frames_lr, fps)
        for chunk in [raw_frames_lr, raw_frames_ud]:
            for f in range(chunk.shape[0]):
                for y in range(chunk.shape[1]):
                    for x in range(chunk.shape[2]):
                        if chunk[f, y, x, 3] != 0:
                            header_values["R"] = chunk[f, y, x, 0]
                            header_values["G"] = chunk[f, y, x, 1]
                            header_values["B"] = chunk[f, y, x, 2]
                            header_values["A"] = chunk[f, y, x, 3]
                            return header_values

        header_values["R"] = 255
        header_values["G"] = 0
        header_values["B"] = 0
        header_values["A"] = 20
        return header_values

    @classmethod
    def format_header_to_bytes(cls, header_values: dict[str, int]):
        header_bytes = bytearray()
        header_bytes += cls.base_header_to_bytes(header_values)

        header_bytes.append(int(header_values["R"]))
        header_bytes.append(int(header_values["G"]))
        header_bytes.append(int(header_values["B"]))
        header_bytes.append(int(header_values["A"]))

        return bytes(header_bytes)

    @classmethod
    def compressed_animation_to_bytes(
        cls,
        compressed_frame_data_ud: np.ndarray,
        compressed_frame_data_lr: np.ndarray,
        header_values: dict[str, int],
    ) -> bytes:
        header_values_bytes = cls.format_header_to_bytes(header_values)
        output_bytes = bytearray()
        output_bytes += header_values_bytes

        for frame in range(header_values["frame_count"]):
            output_bytes += (
                np.packbits(compressed_frame_data_ud[frame].flatten("C"))
                .astype(np.uint8)
                .tobytes("C")
            )
            output_bytes += (
                np.packbits(compressed_frame_data_lr[frame].flatten("C"))
                .astype(np.uint8)
                .tobytes("C")
            )

        # I'm proud of my first swing at bit based operations so this code block stays
        # Even though numpy probably does it 10 times better
        # EDIT: This is code from before I implemented te limit rect system
        #
        # bit_position = 0
        # byte_buffer = 0<<8
        # i = 0
        # i_final = stencil.shape[0]*stencil.shape[1]*stencil.shape[2]-1
        #
        # for f in range(frame_count):
        #     for y in range(height):
        #         for x in range(width):
        #             byte_buffer = byte_buffer | (stencil[f,y,x]>>bit_position)
        #             bit_position += 1
        #             if bit_position == 8 or i == i_final:
        #                 output_bytes+=byte_buffer
        #                 bit_position = 0
        #                 byte_buffer = 0<<8
        #             i+=1

        return output_bytes

    @classmethod
    def compress_raw_frames(
        cls, raw_frames_ud: np.ndarray, raw_frames_lr: np.ndarray, fps: int
    ) -> tuple[tuple[np.ndarray, np.ndarray], dict[str, int]]:
        header_values = cls.create_format_header(raw_frames_ud, raw_frames_lr, fps)

        compressed_frames_ud = np.where(raw_frames_ud[:, :, :, 3] > 0, 1, 0).astype(
            np.uint8
        )
        compressed_frames_lr = np.where(raw_frames_lr[:, :, :, 3] > 0, 1, 0).astype(
            np.uint8
        )

        return (compressed_frames_ud, compressed_frames_lr), header_values

    @classmethod
    def read_compressed_animation_data(
        cls, bytes_object
    ) -> tuple[np.ndarray, dict[str, int]]:
        header_values = cls.read_format_header(bytes_object)

        frame_count = header_values["frame_count"]
        width = header_values["width"]
        height = header_values["height"]

        compressed_frames_ud = np.zeros(
            (frame_count, header_values["limit_rect_y1"] * 2, width), dtype=np.uint8
        )
        compressed_frames_lr = np.zeros(
            (
                frame_count,
                height - header_values["limit_rect_y1"] * 2,
                header_values["limit_rect_x1"] * 2,
            ),
            dtype=np.uint8,
        )

        bit_length_ud = compressed_frames_ud.shape[1] * compressed_frames_ud.shape[2]
        bit_length_lr = compressed_frames_lr.shape[1] * compressed_frames_lr.shape[2]
        bit_length_frame = bit_length_ud + bit_length_lr

        # This is the raw format so technically the frames aren't compressed at all, except for the limit rect based segmentation.

        frames_data = np.frombuffer(
            bytes_object, offset=cls.frame_data_offset, dtype=np.uint8
        )

        bit_i = 0
        calc_byte_i_l = lambda bit_i: (int(np.floor(bit_i / 8)), bit_i % 8)
        calc_byte_i_u = lambda bit_i: (int(np.ceil(bit_i / 8)), bit_i % 8)

        for frame in range(frame_count):
            byte_i_l, remainder_l = calc_byte_i_l(bit_i)
            bit_i += bit_length_ud
            byte_i_u, remainder_u = calc_byte_i_u(bit_i)

            compressed_frames_ud[frame, :, :] = np.unpackbits(
                frames_data[byte_i_l:byte_i_u]
            )[remainder_l : remainder_u if remainder_u > 0 else None].reshape(
                compressed_frames_ud.shape[1:], order="C"
            )

            byte_i_l, remainder_l = calc_byte_i_l(bit_i)
            bit_i += bit_length_lr
            byte_i_u, remainder_u = calc_byte_i_u(bit_i)

            compressed_frames_lr[frame, :, :] = np.unpackbits(
                frames_data[byte_i_l:byte_i_u]
            )[remainder_l : remainder_u if remainder_u > 0 else None].reshape(
                compressed_frames_lr.shape[1:], order="C"
            )

        return (compressed_frames_ud, compressed_frames_lr), header_values

    @classmethod
    def decompress_frame(
        cls,
        frame_index: int,
        compressed_frame_data_ud: np.ndarray,
        compressed_frame_data_lr: np.ndarray,
        header_values: dict[str, int],
    ) -> tuple[np.ndarray, np.ndarray]:
        width = header_values["width"]
        height = header_values["height"]

        R = header_values["R"]
        G = header_values["G"]
        B = header_values["B"]
        A = header_values["A"]
        stencil_ud = compressed_frame_data_ud[frame_index, :, :, np.newaxis]
        stencil_lr = compressed_frame_data_lr[frame_index, :, :, np.newaxis]
        decompressed_ud = stencil_ud * (
            np.ones(stencil_ud.shape, dtype=np.uint8)
            * np.array([R, G, B, A], dtype=np.uint8)
        )
        decompressed_lr = stencil_lr * (
            np.ones(stencil_lr.shape, dtype=np.uint8)
            * np.array([R, G, B, A], dtype=np.uint8)
        )

        return (decompressed_ud, decompressed_lr)
