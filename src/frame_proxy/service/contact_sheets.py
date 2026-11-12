from __future__ import annotations

import hashlib

from frame_proxy.config import Settings
from frame_proxy.domain.errors import (
    ApplicationError,
    ErrorCode,
    MediaValidationError,
    StateError,
)
from frame_proxy.domain.models import ClipRecord, ContactSheetRecord
from frame_proxy.domain.policies import ContactSheetRequest, validate_identifier
from frame_proxy.domain.transitions import transition_contact_sheet
from frame_proxy.media.avi import AviDescriptor, descriptor_from_geometry, probe_avi
from frame_proxy.media.contact import compose_grid, plan_grid
from frame_proxy.media.frames import Frame, decode_frame_at
from frame_proxy.media.ppm import parse_ppm_header
from frame_proxy.storage.repository import FileRepository


def _verified_descriptor(clip: ClipRecord, media: bytes) -> AviDescriptor:
    """Resolve the frame geometry recorded at intake for this exact media.

    The persisted offsets are the intake probe's own output, so rendering
    reuses the same frame interpretation instead of re-walking the container.
    Records written without usable offsets fall back to a fresh probe of the
    digest-verified bytes.
    """
    try:
        return descriptor_from_geometry(
            width=clip.width,
            height=clip.height,
            frame_count=clip.frame_count,
            fps_numerator=clip.fps_numerator,
            fps_denominator=clip.fps_denominator,
            row_stride=clip.row_stride,
            frame_payload_size=clip.frame_payload_size,
            top_down=clip.top_down,
            frame_offsets=clip.frame_offsets,
            handler=clip.handler,
        )
    except MediaValidationError:
        return probe_avi(media)


class ContactSheetService:
    def __init__(self, repository: FileRepository, settings: Settings) -> None:
        self._repository = repository
        self._settings = settings

    def create(
        self,
        inspection_id: str,
        request: ContactSheetRequest,
    ) -> ContactSheetRecord:
        normalized_id = validate_identifier(inspection_id, "inspection_id")
        inspection = self._repository.get_inspection(normalized_id)
        if inspection.state != "indexed":
            raise StateError(
                "contact sheets require an indexed inspection",
                details={"state": inspection.state},
            )
        clip = self._repository.get_clip(inspection.clip_id)
        plan = plan_grid(
            scene_count=len(inspection.scenes),
            source_width=clip.width,
            source_height=clip.height,
            request=request,
            max_scenes=self._settings.max_sheet_scenes,
            max_pixels=self._settings.max_sheet_pixels,
        )

        record = ContactSheetRecord(
            sheet_id=request.sheet_id,
            inspection_id=inspection.inspection_id,
            state="planned",
            columns=request.columns,
            tile_width=request.tile_width,
            tile_height=plan.tile_height,
            gutter=request.gutter,
            background_rgb=request.background_rgb,
            selection=request.selection,
            rows=plan.rows,
            image_width=plan.image_width,
            image_height=plan.image_height,
            artifact_name="sheet.ppm",
            artifact_digest=None,
            tile_map=[],
        )
        self._repository.create_contact_sheet(record)
        transition_contact_sheet(record, "compositing", reason="layout reserved")
        self._repository.update_contact_sheet(record)

        try:
            media = self._repository.get_clip_media(clip.clip_id)
            if hashlib.sha256(media).hexdigest() != clip.digest:
                raise MediaValidationError(
                    "stored clip digest does not match its record",
                    code=ErrorCode.FRAME_DECODE,
                )
            descriptor = _verified_descriptor(clip, media)
            frames: dict[int, Frame] = {}
            for scene in inspection.scenes:
                if request.selection == "first":
                    frame_index = scene.start_frame
                elif request.selection == "midpoint":
                    frame_index = scene.start_frame + (
                        scene.end_frame - scene.start_frame
                    ) // 2
                else:
                    frame_index = scene.representative_frame
                if frame_index not in frames:
                    frames[frame_index] = decode_frame_at(
                        media,
                        descriptor,
                        frame_index,
                    )
            result = compose_grid(
                plan=plan,
                scenes=tuple(inspection.scenes),
                frames=frames,
                background_rgb=request.background_rgb,
                selection=request.selection,
            )
            header = parse_ppm_header(result.payload)
            if (
                header.width != record.image_width
                or header.height != record.image_height
            ):
                raise ValueError("composed PPM dimensions disagree with the plan")
            ready_record = ContactSheetRecord.from_dict(record.as_dict())
            ready_record.artifact_digest = result.digest
            ready_record.tile_map = list(result.tile_map)
            transition_contact_sheet(
                ready_record,
                "ready",
                reason="artifact verified",
            )
            self._repository.commit_contact_sheet(ready_record, result.payload)
            return ready_record
        except ApplicationError as exc:
            self._mark_failed(record, exc.code.value, exc.message)
            raise
        except Exception as exc:
            self._mark_failed(record, ErrorCode.INTERNAL.value, str(exc))
            raise ApplicationError(
                ErrorCode.INTERNAL,
                "contact sheet composition failed",
                status=500,
            ) from exc

    def _mark_failed(
        self,
        record: ContactSheetRecord,
        code: str,
        message: str,
    ) -> None:
        record.error_code = code
        record.error_message = message
        if record.state in {"planned", "compositing"}:
            transition_contact_sheet(record, "failed", reason=code)
        try:
            self._repository.update_contact_sheet(record)
        except Exception:
            pass

    def get(self, sheet_id: str) -> ContactSheetRecord:
        normalized = validate_identifier(sheet_id, "sheet_id")
        return self._repository.get_contact_sheet(normalized)

    def get_artifact(self, sheet_id: str) -> bytes:
        normalized = validate_identifier(sheet_id, "sheet_id")
        record = self._repository.get_contact_sheet(normalized)
        if record.state != "ready":
            raise StateError(
                "contact sheet is not ready",
                details={"state": record.state},
            )
        return self._repository.get_contact_sheet_artifact(
            record.sheet_id,
            record.artifact_name,
        )
