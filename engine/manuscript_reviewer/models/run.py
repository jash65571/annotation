"""Run-level models: the project run and the reproducibility manifest."""

from __future__ import annotations

from .common import StrictModel
from .validation import RunStatus


class ArtifactEntry(StrictModel):
    """One generated artifact, with its hash for tamper-evidence."""

    path: str  # relative to the run folder
    sha256: str
    size_bytes: int


class RunManifest(StrictModel):
    """manifest.json — everything needed to reproduce and trust the run."""

    run_id: str
    source_video_path: str
    source_video_sha256: str
    source_seed_path: str | None = None
    source_seed_sha256: str | None = None
    #: Phase 4 inputs (additive; None when the stage did not run).
    source_feedback_path: str | None = None
    source_feedback_sha256: str | None = None
    review_decisions_path: str | None = None
    review_decisions_sha256: str | None = None
    #: Visual-anchor provenance: reproducibility is never claimed without the exact
    #: anchor bytes the tracks were seeded from (item 20).
    source_visual_anchors_path: str | None = None
    source_visual_anchors_sha256: str | None = None
    #: Phase 5 inputs/outputs (additive; None when the stage did not run).
    human_facts_path: str | None = None
    human_facts_sha256: str | None = None
    final_review_path: str | None = None
    final_review_sha256: str | None = None
    caption_brain_version: str | None = None
    caption_final_status: str | None = None
    visual_intelligence_version: str | None = None
    ocr_status: str | None = None
    #: Engine provenance for the visual stage (item 20).
    ocr_engine: str | None = None
    ocr_version: str | None = None
    ocr_language: str | None = None
    tracking_version: str | None = None
    tracking_config: dict[str, float] = {}
    app_version: str
    rules_version: str
    ffmpeg_version: str
    ffprobe_version: str
    started_at_utc: str
    ended_at_utc: str
    analysis_duration_seconds: float
    #: Wall-clock timing of individual pipeline stages (benchmarks).
    stage_timings_seconds: dict[str, float]
    artifacts: list[ArtifactEntry]
    validation_status: RunStatus


class ProjectRun(StrictModel):
    """In-memory description of one audit run (not persisted as a whole;
    the manifest + artifacts are the persistent record)."""

    run_id: str
    video_path: str
    seed_path: str | None = None
    run_dir: str
    extract_frames: bool = False
