"""Item 20: run-manifest provenance for the visual stage.

A result is never claimed reproducible without the anchor input hash, the tracking
config it was produced by, and the OCR engine/version/language/status.
"""

from __future__ import annotations

from manuscript_reviewer.models.run import RunManifest
from manuscript_reviewer.models.validation import RunStatus
from manuscript_reviewer.tracking.tracker import TRACKING_VERSION, tracking_config


def test_tracking_config_is_pinned_and_versioned() -> None:
    cfg = tracking_config()
    assert set(cfg) == {
        "track_threshold", "occlusion_max", "lost_giveup", "min_template",
        "ambiguity_margin", "max_disp_factor", "max_disp_floor",
        "metric_width", "metric_height",
    }
    assert cfg["track_threshold"] == 0.55
    assert TRACKING_VERSION  # non-empty; bumps when tracking behaviour changes


def test_manifest_carries_visual_provenance() -> None:
    manifest = RunManifest(
        run_id="R1", source_video_path="v.mp4", source_video_sha256="v-sha",
        source_visual_anchors_path="anchors.json", source_visual_anchors_sha256="a-sha",
        visual_intelligence_version="0.4.0", ocr_status="AVAILABLE",
        ocr_engine="tesseract", ocr_version="5.3.0", ocr_language="eng",
        tracking_version=TRACKING_VERSION, tracking_config=tracking_config(),
        app_version="0.1.0", rules_version="1.3.0",
        ffmpeg_version="ffmpeg 8", ffprobe_version="ffprobe 8",
        started_at_utc="2026-08-12T00:00:00Z", ended_at_utc="2026-08-12T00:00:01Z",
        analysis_duration_seconds=1.0, stage_timings_seconds={}, artifacts=[],
        validation_status=RunStatus.REVIEW_REQUIRED,
    )
    dumped = manifest.model_dump(mode="json")
    assert dumped["source_visual_anchors_sha256"] == "a-sha"
    assert dumped["tracking_version"] == TRACKING_VERSION
    assert dumped["tracking_config"]["track_threshold"] == 0.55
    assert (dumped["ocr_engine"], dumped["ocr_version"], dumped["ocr_language"]) == (
        "tesseract", "5.3.0", "eng")
