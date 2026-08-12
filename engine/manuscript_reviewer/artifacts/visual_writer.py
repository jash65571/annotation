"""Phase 4 visual artifact writing: frame observations and the enriched ledger.

Phase 1 ``frames.csv``/``frames.jsonl`` are never overwritten. The enriched
ledger is a separate, human-friendly export; every derived column is labelled by
source (deterministic vs candidate) so machine candidates are never mistaken for
factual truth.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

from ..media.timestamps import seconds_to_decimal
from ..models.frame import FrameLedger
from ..models.review_intelligence import FrameObservation
from .writer import ArtifactWriteError

_OBS_COLUMNS = [
    "frame_index",
    "source_pts",
    "source_pts_time",
    "annotation_time",
    "shot_number",
    "brightness",
    "contrast",
    "sharpness",
    "motion_magnitude",
    "global_camera_motion",
    "foreground_motion",
    "text_region_count",
    "visual_concern_candidates",
]


def _time(value: object) -> str:
    if value is None:
        return ""
    from fractions import Fraction

    if isinstance(value, Fraction):
        return str(seconds_to_decimal(value))
    return str(value)


def _obs_row(obs: FrameObservation) -> list[object]:
    return [
        obs.frame_index,
        obs.source_pts if obs.source_pts is not None else "",
        _time(obs.source_pts_time_exact),
        _time(obs.annotation_time_exact),
        obs.shot_number if obs.shot_number is not None else "",
        obs.brightness,
        obs.contrast,
        obs.sharpness,
        obs.motion_magnitude,
        obs.global_camera_motion,
        obs.foreground_motion,
        obs.text_region_count,
        ";".join(obs.visual_concern_codes),
    ]


def write_frame_observations_csv(visual_dir: Path, observations: list[FrameObservation]) -> Path:
    visual_dir.mkdir(parents=True, exist_ok=True)
    path = visual_dir / "frame_observations.csv"
    try:
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(_OBS_COLUMNS)
            for obs in observations:
                writer.writerow(_obs_row(obs))
    except OSError as exc:
        raise ArtifactWriteError(f"Failed to write {path}: {exc}") from exc
    return path


def write_frame_observations_jsonl(visual_dir: Path, observations: list[FrameObservation]) -> Path:
    visual_dir.mkdir(parents=True, exist_ok=True)
    path = visual_dir / "frame_observations.jsonl"
    try:
        with path.open("w", encoding="utf-8", newline="\n") as handle:
            for obs in observations:
                handle.write(json.dumps(obs.model_dump(mode="json"), sort_keys=True) + "\n")
    except OSError as exc:
        raise ArtifactWriteError(f"Failed to write {path}: {exc}") from exc
    return path


#: Column -> provenance tag (documented in docs/08).
_ENRICHED_SOURCES = {
    "frame_index": "DETERMINISTIC",
    "source_pts_time": "DETERMINISTIC",
    "annotation_time": "DETERMINISTIC",
    "key_frame": "DETERMINISTIC",
    "shot_number": "DETERMINISTIC",
    "brightness": "DETERMINISTIC",
    "contrast": "DETERMINISTIC",
    "sharpness": "DETERMINISTIC",
    "motion_magnitude": "DETERMINISTIC",
    "global_camera_motion": "CANDIDATE",
    "foreground_motion": "CANDIDATE",
    "visual_concern_candidates": "CANDIDATE",
}


def write_enriched_frame_ledger(
    visual_dir: Path, ledger: FrameLedger, observations: list[FrameObservation]
) -> Path:
    """Combine Phase 1 identity/timing with Phase 4 observations. Phase 1 files
    remain untouched; this is a derived export with per-column source tags."""
    visual_dir.mkdir(parents=True, exist_ok=True)
    path = visual_dir / "enriched_frame_ledger.csv"
    columns = list(_ENRICHED_SOURCES.keys())
    obs_by_index = {o.frame_index: o for o in observations}
    try:
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(columns)
            # Second row documents the provenance of each column.
            writer.writerow([_ENRICHED_SOURCES[c] for c in columns])
            for record in ledger.frames:
                obs = obs_by_index.get(record.frame_index)
                writer.writerow(
                    [
                        record.frame_index,
                        _time(record.pts_time_seconds),
                        _time(obs.annotation_time_exact) if obs else "",
                        int(record.key_frame),
                        obs.shot_number if obs and obs.shot_number is not None else "",
                        obs.brightness if obs else "",
                        obs.contrast if obs else "",
                        obs.sharpness if obs else "",
                        obs.motion_magnitude if obs else "",
                        obs.global_camera_motion if obs else "",
                        obs.foreground_motion if obs else "",
                        ";".join(obs.visual_concern_codes) if obs else "",
                    ]
                )
    except OSError as exc:
        raise ArtifactWriteError(f"Failed to write {path}: {exc}") from exc
    return path
