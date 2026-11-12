from __future__ import annotations

from dataclasses import dataclass

from frame_proxy.domain.errors import ErrorCode, MediaValidationError
from frame_proxy.media.avi import AviDescriptor


@dataclass(frozen=True, slots=True)
class Frame:
    index: int
    width: int
    height: int
    rgb: bytes


def decode_frame_at(
    buffer: bytes,
    descriptor: AviDescriptor,
    frame_index: int,
) -> Frame:
    if frame_index < 0 or frame_index >= descriptor.frame_count:
        raise MediaValidationError(
            "frame index is outside the clip",
            code=ErrorCode.MEDIA_BOUNDS,
            details={"frame_index": frame_index},
        )
    if frame_index >= len(descriptor.frame_offsets):
        raise MediaValidationError(
            "frame offset was not found",
            code=ErrorCode.FRAME_DECODE,
        )
    offset = descriptor.frame_offsets[frame_index]
    if offset < 8:
        raise MediaValidationError(
            "frame offset is outside the container",
            code=ErrorCode.FRAME_DECODE,
        )
    frame_end = offset + descriptor.frame_payload_size
    if frame_end > len(buffer):
        raise MediaValidationError(
            "frame extent is invalid",
            code=ErrorCode.FRAME_DECODE,
        )
    tag = buffer[offset - 8 : offset - 4]
    declared_size = int.from_bytes(buffer[offset - 4 : offset], "little")
    if tag != b"00db" or declared_size != descriptor.frame_payload_size:
        raise MediaValidationError(
            "frame extent disagrees with its chunk",
            code=ErrorCode.FRAME_DECODE,
        )

    rgb_rows: list[bytes] = []
    for row_index in range(descriptor.height):
        source_row = row_index
        if not descriptor.top_down:
            source_row = descriptor.height - 1 - row_index
        row_start = offset + source_row * descriptor.row_stride
        row = buffer[row_start : row_start + descriptor.width * 3]
        converted = bytearray(len(row))
        converted[0::3] = row[2::3]
        converted[1::3] = row[1::3]
        converted[2::3] = row[0::3]
        rgb_rows.append(bytes(converted))
    return Frame(
        index=frame_index,
        width=descriptor.width,
        height=descriptor.height,
        rgb=b"".join(rgb_rows),
    )

