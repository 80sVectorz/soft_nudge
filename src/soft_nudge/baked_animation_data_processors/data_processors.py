import numpy as np


class BakedAnimationDataProcessor:
    """Base animation data processor class

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

    Frame segmentation & Segment chunking:
    +----------------------------+      +----------------------------+
    |             U              |      |      U                     |
    +-+------------------------+-+      +-------------UD-------------+
    |L|       limit rect       |R|      |      D                     |
    +-+------------------------+-+      +----------------------------+
    |             D              |
    +----------------------------+      +-------+-------+
                                        |  L    |   R   |
    Segments are combined into 2 chunks |       |       |
    Chunk UD and chunk LR               |       |       |
                                        |       LR      |
                                        |       |       |
                                        |       |       |
                                        +-------+-------+
    FRAME DATA:
    - F_0...F_n
        - Y_UD_0...Y_UD_n
            - X_UD_0...X_UD_n
                - R: U-INT 8
                - G: U-INT 8
                - B: U-INT 8
                - A: U-INT 8
        - Y_LR_0...Y_LR_n
            - X_LR_0...X_LR_n
                - R: U-INT 8
                - G: U-INT 8
                - B: U-INT 8
                - A: U-INT 8
    """

    frame_data_offset = 16
    data_format_id = 0

    @classmethod
    def read_base_header(cls, bytes_object) -> dict[str, int]:
        data_format = bytes_object[0]

        width = (bytes_object[1] << 8) | bytes_object[2]
        height = (bytes_object[3] << 8) | bytes_object[4]

        limit_rect_x1 = (bytes_object[5] << 8) | bytes_object[6]
        limit_rect_y1 = (bytes_object[7] << 8) | bytes_object[8]
        limit_rect_x2 = (bytes_object[9] << 8) | bytes_object[10]
        limit_rect_y2 = (bytes_object[11] << 8) | bytes_object[12]

        frame_count = (bytes_object[13] << 8) | bytes_object[14]
        fps = bytes_object[15]

        header_values = {
            "format": data_format,
            "width": width,
            "height": height,
            "limit_rect_x1": limit_rect_x1,
            "limit_rect_y1": limit_rect_y1,
            "limit_rect_x2": limit_rect_x2,
            "limit_rect_y2": limit_rect_y2,
            "frame_count": frame_count,
            "fps": fps,
        }
        return header_values

    @classmethod
    def create_base_header(
        cls, raw_frames_ud: np.ndarray, raw_frames_lr: np.ndarray, fps: int
    ) -> bytes:
        header_values = {
            "format": cls.data_format_id,
            "width": raw_frames_ud.shape[2],
            "height": raw_frames_lr.shape[1] + raw_frames_ud.shape[1],
            "limit_rect_x1": raw_frames_lr.shape[2] // 2,
            "limit_rect_y1": raw_frames_ud.shape[1] // 2,
            "limit_rect_x2": raw_frames_ud.shape[2] - raw_frames_lr.shape[2] // 2,
            "limit_rect_y2": raw_frames_lr.shape[1]
            + raw_frames_ud.shape[1]
            - raw_frames_ud.shape[1] // 2,
            "frame_count": raw_frames_ud.shape[0],
            "fps": fps,
        }
        return header_values

    @classmethod
    def base_header_to_bytes(cls, header_values: dict[str, int]):
        format_uint8 = int(header_values["format"]).to_bytes(1)
        width_uint16 = int(header_values["width"]).to_bytes(2)
        height_uint16 = int(header_values["height"]).to_bytes(2)
        limit_rect_x1_uint16 = int(header_values["limit_rect_x1"]).to_bytes(2)
        limit_rect_y1_uint16 = int(header_values["limit_rect_y1"]).to_bytes(2)
        limit_rect_x2_uint16 = int(header_values["limit_rect_x2"]).to_bytes(2)
        limit_rect_y2_uint16 = int(header_values["limit_rect_y2"]).to_bytes(2)
        frame_count_uint16 = int(header_values["frame_count"]).to_bytes(2)
        fps_uint8 = int(header_values["fps"]).to_bytes(1)

        standard_header_bytes = bytearray()

        standard_header_bytes.append(format_uint8[0])
        standard_header_bytes.append(width_uint16[0])
        standard_header_bytes.append(width_uint16[1])
        standard_header_bytes.append(height_uint16[0])
        standard_header_bytes.append(height_uint16[1])

        standard_header_bytes.append(limit_rect_x1_uint16[0])
        standard_header_bytes.append(limit_rect_x1_uint16[1])
        standard_header_bytes.append(limit_rect_y1_uint16[0])
        standard_header_bytes.append(limit_rect_y1_uint16[1])
        standard_header_bytes.append(limit_rect_x2_uint16[0])
        standard_header_bytes.append(limit_rect_x2_uint16[1])
        standard_header_bytes.append(limit_rect_y2_uint16[0])
        standard_header_bytes.append(limit_rect_y2_uint16[1])

        standard_header_bytes.append(frame_count_uint16[0])
        standard_header_bytes.append(frame_count_uint16[1])
        standard_header_bytes.append(fps_uint8[0])

        return bytes(standard_header_bytes)

    @classmethod
    def read_format_header(cls, bytes_object) -> dict[str, int]:
        header_values = cls.read_base_header(bytes_object)
        return header_values

    @classmethod
    def create_format_header(
        cls, raw_frames_ud: np.ndarray, raw_frames_lr: np.ndarray, fps: int
    ) -> bytes:
        header_values = cls.create_base_header(raw_frames_ud, raw_frames_lr, fps)
        return header_values

    @classmethod
    def format_header_to_bytes(cls, header_values: dict[str, int]):
        header_bytes = bytearray()
        header_bytes += cls.base_header_to_bytes(header_values)
        return bytes(header_bytes)

    @classmethod
    def compress_raw_frames_to_bytes(
        cls, raw_frames_ud: np.ndarray, raw_frames_lr: np.ndarray, fps: int
    ) -> bytes:
        compressed_frames, header_values = cls.compress_raw_frames(
            raw_frames_ud, raw_frames_lr, fps
        )
        return cls.compressed_animation_to_bytes(*compressed_frames, header_values)

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
                compressed_frame_data_ud[frame].astype(np.uint8).tobytes("C")
            )
            output_bytes += (
                compressed_frame_data_lr[frame].astype(np.uint8).tobytes("C")
            )

        return bytes(output_bytes)

    @classmethod
    def compress_raw_frames(
        cls, raw_frames_ud: np.ndarray, raw_frames_lr: np.ndarray, fps: int
    ) -> tuple[tuple[np.ndarray, np.ndarray], dict[str, int]]:
        header_values = cls.create_format_header(raw_frames_ud, raw_frames_lr, fps)

        compressed_frames_ud = raw_frames_ud
        compressed_frames_lr = raw_frames_lr

        return (compressed_frames_ud, compressed_frames_lr), header_values

    @classmethod
    def read_compressed_animation_data(
        cls, bytes_object
    ) -> tuple[tuple[np.ndarray, np.ndarray], dict[str, int]]:
        header_values = cls.read_format_header(bytes_object)

        frame_count = header_values["frame_count"]
        width = header_values["width"]
        height = header_values["height"]

        compressed_frames_ud = np.zeros(
            (frame_count, header_values["limit_rect_y1"] * 2, width, 4), dtype=np.uint8
        )
        compressed_frames_lr = np.zeros(
            (
                frame_count,
                height - header_values["limit_rect_y1"] * 2,
                header_values["limit_rect_x1"] * 2,
                4,
            ),
            dtype=np.uint8,
        )

        byte_length_ud = (
            compressed_frames_ud.shape[1] * compressed_frames_ud.shape[2] * 4
        )
        byte_length_lr = (
            compressed_frames_lr.shape[1] * compressed_frames_lr.shape[2] * 4
        )
        byte_length_frame = byte_length_ud + byte_length_lr

        # This is the raw format so technically the frames aren't compressed at all, except for the limit rect based segmentation.

        frames_data = np.frombuffer(
            bytes_object, offset=cls.frame_data_offset, dtype=np.uint8
        )
        i = 0
        for frame in range(frame_count):
            compressed_frames_ud[frame, :, :, :] = frames_data[
                i : i + byte_length_ud
            ].reshape(compressed_frames_ud.shape[1:], order="C")
            i += byte_length_ud
            compressed_frames_lr[frame, :, :, :] = frames_data[
                i : i + byte_length_lr
            ].reshape(compressed_frames_lr.shape[1:], order="C")
            i += byte_length_lr

        return (compressed_frames_ud, compressed_frames_lr), header_values

    @classmethod
    def decompress_frame(
        cls,
        frame_index: int,
        compressed_frame_data_ud: np.ndarray,
        compressed_frame_data_lr: np.ndarray,
        header_values: dict[str, int],
    ) -> tuple[np.ndarray, np.ndarray]:
        return (
            compressed_frame_data_ud[frame_index, :, :, :],
            compressed_frame_data_lr[frame_index, :, :, :],
        )


class BakedAnimationDataProcessorRaw(BakedAnimationDataProcessor):
    """Raw frame data processor class (No compression)
    Not recommended as the result size is worse than unreasonable.
    An animation at 1920x1080 10fps 10s ends up being bigger than 6Gb!

    File structure:

    HEADER:
    - format: U-INT 8
    - width: U-INT I16
    - height: U-INT 16
    - frame_count: U-INT 16
    - fps: U-INT 8

    FRAME DATA:
    - F0...Fn
        - Y0...Yn
            - X0...Xn
                - R: U-INT 8
                - G: U-INT 8
                - B: U-INT 8
                - A: U-INT 8
    """

    pass
