"""Candidate generation: recall-first, multi-signal, merged per adjacent pair.

Missing a real cut is worse than producing an extra candidate for review, so
generators use permissive thresholds; the adversarial verifier is responsible
for precision. Candidates for the same adjacent pair merge, preserving every
source.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..media.timestamps import format_manuscript_display
from ..models.frame import FrameLedger
from ..models.shot_truth import (
    BoundaryCandidate,
    CandidateStatus,
    LocalBaseline,
    PairMetrics,
)
from .regions import FlashRegion


@dataclass(frozen=True)
class Sensitivity:
    """Threshold scale for candidate generation. Lower factors = more candidates."""

    diff_z: float
    hist_z: float
    phash_min: int
    scdet_min: float
    abs_diff_floor: float

    @staticmethod
    def from_name(name: str) -> Sensitivity:
        presets = {
            # recall-first defaults
            "high": Sensitivity(diff_z=3.5, hist_z=3.5, phash_min=10, scdet_min=6.0,
                                abs_diff_floor=3.0),
            "normal": Sensitivity(diff_z=5.0, hist_z=5.0, phash_min=14, scdet_min=8.0,
                                  abs_diff_floor=4.0),
            "low": Sensitivity(diff_z=8.0, hist_z=8.0, phash_min=20, scdet_min=12.0,
                               abs_diff_floor=6.0),
        }
        if name not in presets:
            raise ValueError(f"Unknown sensitivity {name!r}; use high/normal/low")
        return presets[name]


def _raw_sources(
    pair: PairMetrics, base: LocalBaseline, sens: Sensitivity
) -> list[str]:
    """Which signal families flag this pair as a possible boundary.

    Every family judges INDEPENDENTLY with its own criteria — a cut with a low
    luma difference (similar palettes, structural change only) must still be
    nominated by histogram/phash/scdet. The MAD floor gates ONLY the
    difference family; recall is the generator's job, precision the verifier's.
    """
    sources: list[str] = []
    if pair.mean_abs_diff >= sens.abs_diff_floor and base.diff_z >= sens.diff_z:
        sources.append("internal_difference")
    if base.hist_z >= sens.hist_z and pair.hist_distance > 0.05:
        sources.append("internal_histogram")
    if pair.phash_hamming >= sens.phash_min:
        sources.append("internal_phash")
    if pair.scdet_score is not None and pair.scdet_score >= sens.scdet_min:
        sources.append("ffmpeg_scdet")
    return sources


def generate_candidates(
    pairs: list[PairMetrics],
    baselines: list[LocalBaseline],
    flash_regions: list[FlashRegion],
    ledger: FrameLedger,
    sensitivity: Sensitivity,
) -> tuple[list[BoundaryCandidate], int]:
    """Generate merged boundary candidates.

    Returns (merged candidates, raw pre-merge candidate count). Flash regions
    contribute candidates at both their entry and exit pairs so a flash used as
    an editorial transition stays visible to review.
    """
    raw_count = 0
    by_pair: dict[int, list[str]] = {}

    for i, (pair, base) in enumerate(zip(pairs, baselines, strict=True)):
        sources = _raw_sources(pair, base, sensitivity)
        raw_count += len(sources)
        if sources:
            by_pair.setdefault(i, []).extend(sources)

    for region in flash_regions:
        entry_pair = region.start_frame - 1  # pair (start-1 → start)
        exit_pair = region.end_frame  # pair (end → end+1)
        for pair_index in (entry_pair, exit_pair):
            if 0 <= pair_index < len(pairs):
                raw_count += 1
                by_pair.setdefault(pair_index, []).append("internal_flash")

    candidates: list[BoundaryCandidate] = []
    for seq, pair_index in enumerate(sorted(by_pair), start=1):
        pair = pairs[pair_index]
        base = baselines[pair_index]
        sources = sorted(set(by_pair[pair_index]))
        right = ledger.frames[pair.right_frame_index]
        score = max(base.diff_z, base.hist_z, float(pair.phash_hamming) / 8.0)
        candidates.append(
            BoundaryCandidate(
                candidate_id=f"cand_{seq:04d}",
                left_frame_index=pair.left_frame_index,
                right_frame_index=pair.right_frame_index,
                left_pts=pair.left_pts,
                right_pts=pair.right_pts,
                boundary_time_exact=right.pts_time_seconds,
                boundary_time_manuscript=(
                    format_manuscript_display(right.pts_time_seconds)
                    if right.pts_time_seconds is not None
                    else None
                ),
                candidate_sources=sources,
                metric_snapshot=pair,
                local_baseline=base,
                candidate_score=round(score, 4),
                status=CandidateStatus.CANDIDATE,
            )
        )
    return candidates, raw_count
