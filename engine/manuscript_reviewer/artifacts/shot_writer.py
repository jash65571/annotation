"""Phase 2 artifact writing: adjacent metrics, candidates, shots, transition
evidence, shot QC. Same determinism rules as the Phase 1 writer."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from ..media.timestamps import seconds_to_decimal
from ..models.shot_truth import (
    BlendEvidence,
    BoundaryCandidate,
    FadeEvidence,
    PairMetrics,
    ShotProposal,
    ShotTruthResult,
)
from .writer import ArtifactWriteError

PAIR_CSV_COLUMNS = [
    "left_frame_index",
    "right_frame_index",
    "left_pts",
    "right_pts",
    "left_pts_time",
    "right_pts_time",
    "mean_abs_diff",
    "hist_distance",
    "phash_hamming",
    "edge_change",
    "luma_delta",
    "flow_mean_mag",
    "flow_coherence",
    "scdet_score",
    "scdet_mafd",
]


def _time_str(pair: PairMetrics, side: str) -> str:
    value = pair.left_pts_time_seconds if side == "left" else pair.right_pts_time_seconds
    return str(seconds_to_decimal(value)) if value is not None else ""


def write_adjacent_metrics_csv(run_dir: Path, pairs: list[PairMetrics]) -> Path:
    path = run_dir / "adjacent_metrics.csv"
    try:
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(PAIR_CSV_COLUMNS)
            for pair in pairs:
                writer.writerow(
                    [
                        pair.left_frame_index,
                        pair.right_frame_index,
                        pair.left_pts if pair.left_pts is not None else "",
                        pair.right_pts if pair.right_pts is not None else "",
                        _time_str(pair, "left"),
                        _time_str(pair, "right"),
                        pair.mean_abs_diff,
                        pair.hist_distance,
                        pair.phash_hamming,
                        pair.edge_change,
                        pair.luma_delta,
                        pair.flow_mean_mag,
                        pair.flow_coherence,
                        pair.scdet_score if pair.scdet_score is not None else "",
                        pair.scdet_mafd if pair.scdet_mafd is not None else "",
                    ]
                )
    except OSError as exc:
        raise ArtifactWriteError(f"Failed to write {path}: {exc}") from exc
    return path


def write_adjacent_metrics_jsonl(run_dir: Path, pairs: list[PairMetrics]) -> Path:
    path = run_dir / "adjacent_metrics.jsonl"
    try:
        with path.open("w", encoding="utf-8", newline="\n") as handle:
            for pair in pairs:
                handle.write(json.dumps(pair.model_dump(mode="json"), sort_keys=True) + "\n")
    except OSError as exc:
        raise ArtifactWriteError(f"Failed to write {path}: {exc}") from exc
    return path


def _write_json(path: Path, payload: object) -> Path:
    try:
        with path.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, ensure_ascii=False)
            handle.write("\n")
    except OSError as exc:
        raise ArtifactWriteError(f"Failed to write {path}: {exc}") from exc
    return path


def write_cut_candidates(run_dir: Path, candidates: list[BoundaryCandidate]) -> Path:
    return _write_json(
        run_dir / "cut_candidates.json",
        {"candidates": [c.model_dump(mode="json") for c in candidates]},
    )


def write_boundary_evidence(run_dir: Path, candidates: list[BoundaryCandidate]) -> Path:
    return _write_json(
        run_dir / "boundary_evidence.json",
        {"boundaries": [c.model_dump(mode="json") for c in candidates]},
    )


def write_shots_proposed(run_dir: Path, shots: list[ShotProposal]) -> Path:
    return _write_json(
        run_dir / "shots_proposed.json",
        {"shots": [s.model_dump(mode="json") for s in shots]},
    )


def write_transition_evidence(
    run_dir: Path,
    candidates: list[BoundaryCandidate],
    fades: list[FadeEvidence],
    blends: list[BlendEvidence],
) -> Path:
    return _write_json(
        run_dir / "transition_evidence.json",
        {
            "boundary_transitions": {
                c.candidate_id: c.transition.model_dump(mode="json")
                for c in candidates
                if c.transition is not None
            },
            "fades": [f.model_dump(mode="json") for f in fades],
            "blends": [b.model_dump(mode="json") for b in blends],
        },
    )


def write_shot_qc(run_dir: Path, result: ShotTruthResult) -> Path:
    return _write_json(run_dir / "shot_qc.json", result.model_dump(mode="json"))
