"""Scale comparison for the intake and contact-sheet workflows.

Builds small-frame, repetitive-content AVI fixtures at two frame counts
(B = 2 x A) through the production AVI writer, then times three cost
centers separately at each scale:

- container handling (AVI probe, stored-media read, full clip intake);
- source-frame extraction (on-demand decode of a frame near the end);
- image composition (grid layout and PPM encoding of a fixed tile set).

Doubling the frame count must roughly double the container costs and leave
per-frame extraction and fixed-grid composition unchanged. The limits below
sit halfway between that linear expectation and the quadratic behavior this
check guards against; they are intentionally loose so timing noise on a
loaded machine cannot flip a healthy build, while a regression to
scan-per-frame or join-per-block behavior (ratio near 4x) fails reliably.
"""

from __future__ import annotations

import statistics
import tempfile
import time
from pathlib import Path

from frame_proxy.domain.models import SceneRecord
from frame_proxy.domain.policies import ContactSheetRequest
from frame_proxy.media.avi import probe_avi
from frame_proxy.media.avi_writer import build_sample_avi
from frame_proxy.media.contact import compose_grid, plan_grid
from frame_proxy.media.frames import decode_frame_at
from frame_proxy.storage.atomic import atomic_write_bytes
from frame_proxy.storage.repository import FileRepository
from tools.harness import RunningService, require, require_status, success

FRAME_WIDTH = 64
FRAME_HEIGHT = 36
FRAMES_A = 400
FRAMES_B = 800

# Ratio limits for a 2x frame-count increase. Linear stages trend to 2.0,
# fixed-work stages to 1.0; the pre-fix quadratic stages trended to 4.0.
LINEAR_LIMIT = 3.0
FIXED_WORK_LIMIT = 1.6

INSPECTION_BODY = {
    "sensitivity": 0.55,
    "sample_stride": 1,
    "min_scene_frames": 2,
}
SHEET_BODY = {
    "columns": 2,
    "tile_width": 96,
    "gutter": 4,
    "selection": "stored",
    "background": "#20252c",
}


def median_timing(repetitions: int, function) -> float:
    samples = []
    for _ in range(repetitions):
        start = time.perf_counter()
        function()
        samples.append(time.perf_counter() - start)
    return statistics.median(samples)


def build_fixture(frame_count: int) -> bytes:
    quarter = frame_count // 4
    return build_sample_avi(
        width=FRAME_WIDTH,
        height=FRAME_HEIGHT,
        fps=12,
        scene_lengths=(quarter, quarter, quarter, frame_count - 3 * quarter),
    )


def fixture_scenes(frame_count: int) -> tuple[SceneRecord, ...]:
    quarter = frame_count // 4
    bounds = [0, quarter, 2 * quarter, 3 * quarter, frame_count]
    return tuple(
        SceneRecord(
            index=index,
            start_frame=bounds[index],
            end_frame=bounds[index + 1],
            representative_frame=(bounds[index] + bounds[index + 1]) // 2,
            mean_luma=0.0,
            contrast=0.0,
            dominant_rgb=(0, 0, 0),
        )
        for index in range(4)
    )


def compose_job(payload: bytes, frame_count: int):
    """Prepare a fixed four-tile grid and return a compose-only callable."""
    descriptor = probe_avi(payload)
    scenes = fixture_scenes(frame_count)
    frames = {
        scene.representative_frame: decode_frame_at(
            payload, descriptor, scene.representative_frame
        )
        for scene in scenes
    }
    request = ContactSheetRequest(
        sheet_id="sheet-scale",
        columns=SHEET_BODY["columns"],
        tile_width=SHEET_BODY["tile_width"],
        gutter=SHEET_BODY["gutter"],
        selection=SHEET_BODY["selection"],
        background_rgb=(32, 37, 44),
    )
    plan = plan_grid(
        scene_count=len(scenes),
        source_width=FRAME_WIDTH,
        source_height=FRAME_HEIGHT,
        request=request,
        max_scenes=36,
        max_pixels=250_000,
    )

    def compose() -> None:
        compose_grid(
            plan=plan,
            scenes=scenes,
            frames=frames,
            background_rgb=(32, 37, 44),
            selection="stored",
        )

    return compose


def main() -> int:
    fixtures = {FRAMES_A: build_fixture(FRAMES_A), FRAMES_B: build_fixture(FRAMES_B)}
    descriptors = {count: probe_avi(payload) for count, payload in fixtures.items()}
    rows: list[tuple[str, float, float, float]] = []

    def record(label: str, scale_a: float, scale_b: float, limit: float) -> None:
        rows.append((label, scale_a, scale_b, limit))

    # --- container handling: probe and stored-media read ---
    record(
        "container_probe",
        median_timing(5, lambda: probe_avi(fixtures[FRAMES_A])),
        median_timing(5, lambda: probe_avi(fixtures[FRAMES_B])),
        LINEAR_LIMIT,
    )
    with tempfile.TemporaryDirectory(prefix="frame-proxy-scale-") as media_dir:
        repository = FileRepository(Path(media_dir))
        for count, payload in fixtures.items():
            clip_dir = repository.paths.clip_dir(f"clip-{count}")
            clip_dir.mkdir(parents=True, exist_ok=True)
            atomic_write_bytes(repository.paths.clip_media(f"clip-{count}"), payload)
        record(
            "stored_media_read",
            median_timing(
                5, lambda: repository.get_clip_media(f"clip-{FRAMES_A}")
            ),
            median_timing(
                5, lambda: repository.get_clip_media(f"clip-{FRAMES_B}")
            ),
            LINEAR_LIMIT,
        )

    # --- source-frame extraction: last frame, on demand ---
    record(
        "frame_extraction_last",
        median_timing(
            300,
            lambda: decode_frame_at(
                fixtures[FRAMES_A], descriptors[FRAMES_A], FRAMES_A - 1
            ),
        ),
        median_timing(
            300,
            lambda: decode_frame_at(
                fixtures[FRAMES_B], descriptors[FRAMES_B], FRAMES_B - 1
            ),
        ),
        FIXED_WORK_LIMIT,
    )

    # --- image composition: fixed four-tile grid, decode already done ---
    record(
        "grid_composition",
        median_timing(20, compose_job(fixtures[FRAMES_A], FRAMES_A)),
        median_timing(20, compose_job(fixtures[FRAMES_B], FRAMES_B)),
        FIXED_WORK_LIMIT,
    )

    # --- workflows over HTTP: intake, then contact sheet after a restart ---
    with tempfile.TemporaryDirectory(prefix="frame-proxy-scale-") as data_dir:
        with RunningService(data_dir=data_dir) as service:
            intake_times = {}
            for count, payload in fixtures.items():
                samples = []
                for attempt in range(3):
                    clip_id = f"clip-scale-{count}-{attempt}"
                    start = time.perf_counter()
                    response = service.upload_clip(clip_id, payload)
                    samples.append(time.perf_counter() - start)
                    require_status(response, 201, f"clip intake {count}")
                intake_times[count] = statistics.median(samples)
            record(
                "workflow_clip_intake",
                intake_times[FRAMES_A],
                intake_times[FRAMES_B],
                LINEAR_LIMIT,
            )
            for count in fixtures:
                response = service.request_json(
                    "POST",
                    f"/v1/clips/clip-scale-{count}-0/inspections",
                    {"inspection_id": f"inspection-scale-{count}", **INSPECTION_BODY},
                )
                require_status(response, 201, f"inspection setup {count}")

        # A fresh service instance on the same data root stands in for a new
        # process: the clips and inspections above must be reusable as-is.
        with RunningService(data_dir=data_dir) as restarted:
            sheet_times = {}
            for count in fixtures:
                samples = []
                for attempt in range(3):
                    start = time.perf_counter()
                    response = restarted.request_json(
                        "POST",
                        f"/v1/inspections/inspection-scale-{count}/contact-sheets",
                        {"sheet_id": f"sheet-scale-{count}-{attempt}", **SHEET_BODY},
                    )
                    samples.append(time.perf_counter() - start)
                    require_status(response, 201, f"contact sheet {count}")
                sheet_times[count] = statistics.median(samples)
            record(
                "workflow_contact_sheet",
                sheet_times[FRAMES_A],
                sheet_times[FRAMES_B],
                LINEAR_LIMIT,
            )

    print(
        f"scale: {FRAMES_A} -> {FRAMES_B} frames "
        f"({FRAME_WIDTH}x{FRAME_HEIGHT}, repetitive scenes)"
    )
    print(f"{'stage':<24}{'scale_a':>12}{'scale_b':>12}{'ratio':>8}{'limit':>8}  verdict")
    failures = []
    for label, scale_a, scale_b, limit in rows:
        ratio = scale_b / scale_a if scale_a > 0 else 0.0
        verdict = "PASS" if ratio <= limit else "FAIL"
        if verdict == "FAIL":
            failures.append(label)
        print(
            f"{label:<24}{scale_a:>11.4f}s{scale_b:>11.4f}s"
            f"{ratio:>8.2f}{limit:>8.2f}  {verdict}"
        )
    require(not failures, f"scaling limits exceeded for: {', '.join(failures)}")
    success(
        "scale-comparison",
        {label: round(scale_b / scale_a, 2) for label, scale_a, scale_b, _ in rows},
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
