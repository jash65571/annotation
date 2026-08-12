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
