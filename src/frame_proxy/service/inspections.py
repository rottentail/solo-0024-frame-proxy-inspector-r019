from __future__ import annotations

import hashlib

from frame_proxy.config import Settings
from frame_proxy.domain.errors import (
    ApplicationError,
    ErrorCode,
    MediaValidationError,
)
from frame_proxy.domain.models import InspectionRecord
from frame_proxy.domain.policies import (
    InspectionRequest,
    validate_clip_id,
    validate_identifier,
)
from frame_proxy.domain.transitions import transition_inspection
from frame_proxy.media.avi import probe_avi
from frame_proxy.media.detector import analyze_frames
from frame_proxy.media.fingerprint import fingerprint_frame
from frame_proxy.media.frames import decode_frame_at
from frame_proxy.storage.repository import FileRepository


class InspectionService:
    def __init__(self, repository: FileRepository, settings: Settings) -> None:
        self._repository = repository
        self._settings = settings

    def create(
        self,
        clip_id: str,
        request: InspectionRequest,
    ) -> InspectionRecord:
        normalized_clip_id = validate_clip_id(clip_id)
        clip = self._repository.get_clip(normalized_clip_id)
        if clip.state != "registered":
            raise MediaValidationError(
                "clip is not ready for inspection",
                code=ErrorCode.INVALID_STATE,
            )

        record = InspectionRecord(
            inspection_id=request.inspection_id,
            clip_id=clip.clip_id,
            state="pending",
            sensitivity=request.sensitivity,
            sample_stride=request.sample_stride,
            min_scene_frames=request.min_scene_frames,
        )
        self._repository.create_inspection(record)

        transition_inspection(record, "scanning", reason="frame scan started")
        self._repository.update_inspection(record)
        try:
            media = self._repository.get_clip_media(clip.clip_id)
            if hashlib.sha256(media).hexdigest() != clip.digest:
                raise MediaValidationError(
                    "stored clip digest does not match its record",
                    code=ErrorCode.FRAME_DECODE,
                )
            descriptor = probe_avi(media)
            sampled_frames = tuple(
                range(0, descriptor.frame_count, request.sample_stride)
            )
            fingerprints = tuple(
                fingerprint_frame(decode_frame_at(media, descriptor, frame_index))
                for frame_index in sampled_frames
            )
            result = analyze_frames(
                descriptor.frame_count,
                sampled_frames,
                fingerprints,
                sensitivity=request.sensitivity,
                min_scene_frames=request.min_scene_frames,
            )
            record.sampled_frames = list(result.sampled_frames)
            record.cut_frames = list(result.cut_frames)
            record.scenes = list(result.scenes)
            transition_inspection(record, "indexed", reason="scene ranges verified")
            self._repository.update_inspection(record)
            return record
        except ApplicationError as exc:
            self._mark_failed(record, exc.code.value, exc.message)
            raise
        except Exception as exc:
            self._mark_failed(record, ErrorCode.INTERNAL.value, str(exc))
            raise ApplicationError(
                ErrorCode.INTERNAL,
                "shot indexing failed",
                status=500,
            ) from exc

    def _mark_failed(
        self,
        record: InspectionRecord,
        code: str,
        message: str,
    ) -> None:
        record.error_code = code
        record.error_message = message
        if record.state in {"pending", "scanning"}:
            transition_inspection(record, "failed", reason=code)
        try:
            self._repository.update_inspection(record)
        except Exception:
            pass

    def get(self, inspection_id: str) -> InspectionRecord:
        normalized = validate_identifier(inspection_id, "inspection_id")
        return self._repository.get_inspection(normalized)
