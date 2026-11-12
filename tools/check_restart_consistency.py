"""Restart consistency for registered clips, scenes, and rendered sheets.

Proves that a registered clip, the persisted scene frame coordinates, and
the rendered contact sheet keep referring to the same verified source bytes
across a service restart, without re-registering the clip or re-running the
inspection:

- phase one registers a clip, indexes it, and renders a first sheet;
- a fresh service instance on the same data root (standing in for a
  restarted process) must return identical records and render a second
  sheet with identical tile map, digest, and PPM bytes;
- after the stored media is altered on disk, rendering must refuse with the
  stable frame-decode error instead of silently reading frames the intake
  never verified.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from tools.harness import RunningService, require, require_status, success

INSPECTION_BODY = {
    "inspection_id": "inspection-restart",
    "sensitivity": 0.55,
    "sample_stride": 1,
    "min_scene_frames": 2,
}
SHEET_BODY = {
    "columns": 3,
    "tile_width": 96,
    "gutter": 4,
    "selection": "stored",
    "background": "#20252c",
}


def create_sheet(service: RunningService, sheet_id: str) -> dict:
    response = service.request_json(
        "POST",
        "/v1/inspections/inspection-restart/contact-sheets",
        {"sheet_id": sheet_id, **SHEET_BODY},
    )
    require_status(response, 201, f"contact sheet {sheet_id}")
    return response.json()["contact_sheet"]


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="frame-proxy-restart-") as data_dir:
        with RunningService(data_dir=data_dir) as first:
            clip_response = first.upload_clip("clip-restart")
            require_status(clip_response, 201, "clip intake")
            clip_record = clip_response.json()["clip"]
            inspection_response = first.request_json(
                "POST",
                "/v1/clips/clip-restart/inspections",
                INSPECTION_BODY,
            )
            require_status(inspection_response, 201, "inspection")
            inspection_record = inspection_response.json()["inspection"]
            sheet_one = create_sheet(first, "sheet-restart-one")
            artifact_one = first.request(
                "GET", "/v1/contact-sheets/sheet-restart-one/artifact"
            )
            require_status(artifact_one, 200, "first artifact fetch")

        # Second instance on the same data root: no re-registration, no
        # re-inspection; records and rendered bytes must line up exactly.
        with RunningService(data_dir=data_dir) as second:
            reloaded_clip = second.request("GET", "/v1/clips/clip-restart")
            require_status(reloaded_clip, 200, "clip reload")
            require(
                reloaded_clip.json()["clip"] == clip_record,
                "clip record changed across restart",
            )
            require(
                len(clip_record["frame_offsets"]) == clip_record["frame_count"],
                "clip record lost its frame offsets",
            )
            reloaded_inspection = second.request(
                "GET", "/v1/inspections/inspection-restart"
            )
            require_status(reloaded_inspection, 200, "inspection reload")
            require(
                reloaded_inspection.json()["inspection"] == inspection_record,
                "inspection record changed across restart",
            )

            sheet_two = create_sheet(second, "sheet-restart-two")
            require(
                sheet_two["tile_map"] == sheet_one["tile_map"],
                "tile map changed across restart",
            )
            require(
                sheet_two["artifact_digest"] == sheet_one["artifact_digest"],
                "artifact digest changed across restart",
            )
            artifact_two = second.request(
                "GET", "/v1/contact-sheets/sheet-restart-two/artifact"
            )
            require_status(artifact_two, 200, "second artifact fetch")
            require(
                artifact_two.body == artifact_one.body,
                "PPM bytes changed across restart",
            )
            scenes = inspection_record["scenes"]
            for tile in sheet_two["tile_map"]:
                scene = scenes[tile["scene_index"]]
                require(
                    tile["source_frame"] == scene["representative_frame"],
                    "tile source frame disagrees with the stored scene",
                )

    # Tampered media must be refused, not reinterpreted.
    with tempfile.TemporaryDirectory(prefix="frame-proxy-tamper-") as data_dir:
        with RunningService(data_dir=data_dir) as service:
            require_status(service.upload_clip("clip-tamper"), 201, "clip intake")
            response = service.request_json(
                "POST",
                "/v1/clips/clip-tamper/inspections",
                INSPECTION_BODY,
            )
            require_status(response, 201, "inspection")
        media_path = Path(data_dir) / "clips" / "clip-tamper" / "clip.avi"
        payload = bytearray(media_path.read_bytes())
        midpoint = len(payload) // 2
        payload[midpoint] ^= 0xFF
        payload[-16] ^= 0xFF
        media_path.write_bytes(bytes(payload))
        with RunningService(data_dir=data_dir) as restarted:
            refused = restarted.request_json(
                "POST",
                "/v1/inspections/inspection-restart/contact-sheets",
                {"sheet_id": "sheet-tamper", **SHEET_BODY},
            )
            require_status(refused, 422, "tampered media refusal")
            require(
                refused.json()["error"]["code"] == "frame_decode_failed",
                "tampered media returned the wrong error code",
            )
            record = restarted.request("GET", "/v1/contact-sheets/sheet-tamper")
            require_status(record, 200, "failed sheet record")
            require(
                record.json()["contact_sheet"]["state"] == "failed",
                "tampered render did not end in the failed state",
            )

    success(
        "restart-consistency",
        {
            "digest": sheet_one["artifact_digest"],
            "tiles": len(sheet_one["tile_map"]),
            "tamper_status": refused.status,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
