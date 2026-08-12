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

from ..camera.global_motion import PairMotion
from ..media.timestamps import seconds_to_decimal
from ..models.frame import FrameLedger
from ..models.review_intelligence import (
    ActionCandidate,
    CameraMotionCandidate,
    CharacterHypothesis,
    ContactEvent,
    ContinuityLink,
    EntityTrack,
    FinalStateCheck,
    FrameObservation,
    ObjectHypothesis,
    OCREngineInfo,
    OCRObservation,
    TextTrack,
    WatermarkCandidate,
)
from .writer import ArtifactWriteError


def _write_json(path: Path, payload: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, ensure_ascii=False)
            handle.write("\n")
    except OSError as exc:
        raise ArtifactWriteError(f"Failed to write {path}: {exc}") from exc
    return path

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
    "raw_interframe_motion",
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
        obs.raw_interframe_motion,
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


# --- OCR artifacts (ensure_ascii=False so non-Latin text is never corrupted) ---


def write_ocr_status(ocr_dir: Path, info: OCREngineInfo) -> Path:
    return _write_json(ocr_dir / "ocr_status.json", info.model_dump(mode="json"))


def write_ocr_observations(ocr_dir: Path, observations: list[OCRObservation]) -> Path:
    return _write_json(
        ocr_dir / "ocr_observations.json",
        {"observations": [o.model_dump(mode="json") for o in observations]},
    )


def write_text_tracks(ocr_dir: Path, tracks: list[TextTrack]) -> Path:
    return _write_json(
        ocr_dir / "text_tracks.json",
        {"text_tracks": [t.model_dump(mode="json") for t in tracks]},
    )


def write_watermark_candidates(ocr_dir: Path, candidates: list[WatermarkCandidate]) -> Path:
    return _write_json(
        ocr_dir / "watermark_candidates.json",
        {"watermark_candidates": [c.model_dump(mode="json") for c in candidates]},
    )


def write_entity_tracks(entities_dir: Path, tracks: list[EntityTrack]) -> Path:
    return _write_json(
        entities_dir / "tracks.json",
        {"tracks": [t.model_dump(mode="json") for t in tracks]},
    )


def write_character_hypotheses(entities_dir: Path, hyps: list[CharacterHypothesis]) -> Path:
    return _write_json(
        entities_dir / "character_hypotheses.json",
        {"character_hypotheses": [h.model_dump(mode="json") for h in hyps]},
    )


def write_object_hypotheses(entities_dir: Path, hyps: list[ObjectHypothesis]) -> Path:
    return _write_json(
        entities_dir / "object_hypotheses.json",
        {"object_hypotheses": [h.model_dump(mode="json") for h in hyps]},
    )


def write_continuity_links(entities_dir: Path, links: list[ContinuityLink]) -> Path:
    return _write_json(
        entities_dir / "continuity_links.json",
        {"continuity_links": [link.model_dump(mode="json") for link in links]},
    )


def write_final_state_checks(entities_dir: Path, checks: list[FinalStateCheck]) -> Path:
    return _write_json(
        entities_dir / "final_state_checks.json",
        {"final_state_checks": [c.model_dump(mode="json") for c in checks]},
    )


def write_contact_events(actions_dir: Path, events: list[ContactEvent]) -> Path:
    return _write_json(
        actions_dir / "contact_events.json",
        {"contact_events": [e.model_dump(mode="json") for e in events]},
    )


def write_action_candidates(actions_dir: Path, candidates: list[ActionCandidate]) -> Path:
    return _write_json(
        actions_dir / "action_candidates.json",
        {"action_candidates": [c.model_dump(mode="json") for c in candidates]},
    )


def write_camera_events(camera_dir: Path, candidates: list[CameraMotionCandidate]) -> Path:
    return _write_json(
        camera_dir / "camera_events.json",
        {"camera_events": [c.model_dump(mode="json") for c in candidates]},
    )


_PAIR_COLUMNS = ["left_frame", "right_frame", "dx", "dy", "response", "scale", "scale_response"]


def write_camera_pair_metrics(camera_dir: Path, pairs: list[PairMotion]) -> Path:
    """Raw per-pair global-motion metrics (kept separate so smoothing never loses
    the sensitive evidence)."""
    camera_dir.mkdir(parents=True, exist_ok=True)
    path = camera_dir / "camera_pair_metrics.csv"
    try:
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(_PAIR_COLUMNS)
            for pm in pairs:
                writer.writerow(
                    [
                        pm.left_frame,
                        pm.right_frame,
                        round(pm.dx, 4),
                        round(pm.dy, 4),
                        round(pm.response, 5),
                        round(pm.scale, 5),
                        round(pm.scale_response, 5),
                    ]
                )
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
    "raw_interframe_motion": "CANDIDATE",
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
                        obs.raw_interframe_motion if obs else "",
                        ";".join(obs.visual_concern_codes) if obs else "",
                    ]
                )
    except OSError as exc:
        raise ArtifactWriteError(f"Failed to write {path}: {exc}") from exc
    return path
