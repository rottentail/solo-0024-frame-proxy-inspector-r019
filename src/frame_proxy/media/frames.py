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
    selected_offset: int | None = None
    preceding_end = 0
    for ordinal, candidate_offset in enumerate(descriptor.frame_offsets):
        if candidate_offset < 8:
            raise MediaValidationError(
                "frame offset is outside the container",
                code=ErrorCode.FRAME_DECODE,
            )
        candidate_end = candidate_offset + descriptor.frame_payload_size
        if candidate_end > len(buffer) or candidate_offset < preceding_end:
            raise MediaValidationError(
                "frame extent is invalid",
                code=ErrorCode.FRAME_DECODE,
            )
        tag = buffer[candidate_offset - 8 : candidate_offset - 4]
        declared_size = int.from_bytes(
            buffer[candidate_offset - 4 : candidate_offset], "little"
        )
        if tag != b"00db" or declared_size != descriptor.frame_payload_size:
            raise MediaValidationError(
                "frame extent disagrees with its chunk",
                code=ErrorCode.FRAME_DECODE,
            )
        preceding_end = candidate_end
        if ordinal == frame_index:
            selected_offset = candidate_offset
            break
    if selected_offset is None:
        raise MediaValidationError(
            "frame offset was not found",
            code=ErrorCode.FRAME_DECODE,
        )
    offset = selected_offset

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

