"""Deterministic adversarial verification of boundary candidates.

For every candidate the verifier actively tries to explain the discontinuity
WITHOUT an edit (motion, flash, occlusion, zoom, fade, blend). Only candidates
that survive every challenge become SUPPORTED. Ambiguity is never resolved to
make output neat — it stays REVIEW_REQUIRED.

All decisions carry structured :class:`ReasonCode` values.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import cv2
import numpy as np

from ..media.timestamps import format_manuscript_display
from ..models.frame import FrameLedger
from ..models.shot_truth import (
    BlendEvidence,
    BoundaryCandidate,
    CandidateStatus,
    FadeEvidence,
    LocalBaseline,
    PairMetrics,
    ReasonCode,
    TransitionEvidence,
    TransitionStatus,
)
from .decode import GrayFrames
from .metrics import _bhattacharyya, _hamming64, _histogram, _phash64
from .regions import FlashRegion

logger = logging.getLogger(__name__)

# --- verifier thresholds (documented in docs/06) ---
SUPPORT_DIFF_Z = 7.0
SUPPORT_STRUCTURE_PHASH = 18
COHERENT_MOTION_MIN = 0.70
MOTION_MAG_MIN = 1.2
SUSTAINED_MOTION_MIN = 0.8
CONTINUITY_PHASH_MAX = 11
CONTINUITY_HIST_MAX = 0.12
EXTREME_DIFF_Z = 14.0
JUMP_CUT_PHASH_MAX = 14


@dataclass
class VerifierContext:
    frames: GrayFrames
    ledger: FrameLedger
    pairs: list[PairMetrics]
    baselines: list[LocalBaseline]
    flash_regions: list[FlashRegion]
    fades: list[FadeEvidence]
    blends: list[BlendEvidence]
    has_audio: bool


def _frame_similarity(ctx: VerifierContext, a: int, b: int) -> tuple[int, float]:
    """(phash hamming, histogram distance) between two arbitrary ledger frames."""
    fa, fb = ctx.frames[a], ctx.frames[b]
    return (
        _hamming64(_phash64(fa), _phash64(fb)),
        _bhattacharyya(_histogram(fa), _histogram(fb)),
    )


def _in_region(pair_index: int, start: int, end: int) -> bool:
    """Pair i spans frames i → i+1; inside [start, end] frame region."""
    return start <= pair_index and pair_index + 1 <= end + 1 and pair_index >= start - 1


def _flash_for(ctx: VerifierContext, candidate: BoundaryCandidate) -> FlashRegion | None:
    for region in ctx.flash_regions:
        entry_pair = region.start_frame - 1
        exit_pair = region.end_frame
        if candidate.left_frame_index in (entry_pair, exit_pair):
            return region
    return None


def _fade_for(ctx: VerifierContext, candidate: BoundaryCandidate) -> FadeEvidence | None:
    for fade in ctx.fades:
        if (
            fade.transition_start_frame - 1
            <= candidate.left_frame_index
            <= fade.transition_end_frame
        ):
            return fade
    return None


def _blend_for(ctx: VerifierContext, candidate: BoundaryCandidate) -> BlendEvidence | None:
    for blend in ctx.blends:
        if blend.start_frame <= candidate.left_frame_index < blend.end_frame:
            return blend
    return None


def _zoom_signature(ctx: VerifierContext, pair_index: int) -> bool:
    """Radial (zoom/scope) motion: high magnitude, low coherence, but flow
    vectors point away from / toward the frame center consistently."""
    pair = ctx.pairs[pair_index]
    if pair.flow_mean_mag < MOTION_MAG_MIN:
        return False
    left = cv2.resize(ctx.frames[pair.left_frame_index], (80, 45))
    right = cv2.resize(ctx.frames[pair.right_frame_index], (80, 45))
    flow = cv2.calcOpticalFlowFarneback(  # type: ignore[call-overload]
        left, right, None, 0.5, 3, 15, 2, 5, 1.1, 0
    )
    h, w = flow.shape[:2]
    ys, xs = np.mgrid[0:h, 0:w]
    rx = (xs - w / 2.0).astype(np.float32)
    ry = (ys - h / 2.0).astype(np.float32)
    norm = np.sqrt(rx**2 + ry**2) + 1e-6
    radial = (flow[..., 0] * rx + flow[..., 1] * ry) / norm
    mags = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2) + 1e-6
    radial_ratio = float(np.abs(radial).mean() / mags.mean())
    return radial_ratio > 0.7


def _frames_match(ctx: VerifierContext, a: int, b: int, gap: int) -> bool:
    """Scene-level similarity between frames ``a`` and ``b`` that are ``gap``
    frames apart. NOTE: this deliberately CANNOT distinguish "same shot,
    moments later" from "jump cut back to a similar composition" — pixel
    similarity cannot prove temporal continuity, which is why a match leads to
    REVIEW_REQUIRED, never to automatic rejection."""
    phash_d, hist_d = _frame_similarity(ctx, a, b)
    if phash_d > 24 or hist_d > 0.09:
        return False
    frame_a = ctx.frames[a].astype(np.int16)
    frame_b = ctx.frames[b].astype(np.int16)
    mad = float(np.abs(frame_b - frame_a).mean())
    return mad <= max(3.0 * gap, 12.0)


def _anomaly_length(ctx: VerifierContext, candidate: BoundaryCandidate) -> int | None:
    """If content shortly after the boundary matches content shortly before it
    (or vice versa), return the anomalous run length in frames; else None."""
    n = ctx.ledger.frame_count
    left = candidate.left_frame_index
    right = candidate.right_frame_index
    # Forward probe: frame LEFT vs frames RIGHT+m-1 (anomaly = m frames R..R+m-1).
    for m in range(1, 9):
        probe = right + m
        if probe >= n:
            break
        if _frames_match(ctx, left, probe, gap=m + 1):
            return m
    # Backward probe: frame RIGHT vs frames LEFT-m (anomaly before the boundary).
    for m in range(1, 9):
        probe = left - m
        if probe < 0:
            break
        if _frames_match(ctx, probe, right, gap=m + 1):
            return m
    return None


def verify_candidate(ctx: VerifierContext, candidate: BoundaryCandidate) -> BoundaryCandidate:
    """Run the adversarial challenge sequence for one candidate."""
    pair_index = candidate.left_frame_index
    pair = candidate.metric_snapshot
    base = candidate.local_baseline
    reasons: list[ReasonCode] = []
    notes: list[str] = list(candidate.notes)
    n = ctx.ledger.frame_count

    # Challenge: part of a multi-frame fade pattern?
    fade = _fade_for(ctx, candidate)
    if fade is not None:
        reasons.append(ReasonCode.MULTIFRAME_FADE_PATTERN)
        notes.append(
            f"Pair lies inside a fade-{fade.direction} ({fade.target_color}) region "
            f"F{fade.transition_start_frame}-F{fade.transition_end_frame}."
        )
        return candidate.model_copy(
            update={
                "status": CandidateStatus.REJECTED,
                "reason_codes": reasons,
                "notes": notes,
            }
        )

    # Challenge: part of a sustained multi-frame blend (possible dissolve)?
    blend = _blend_for(ctx, candidate)
    if blend is not None:
        peak_index = max(
            range(max(blend.start_frame, 0), min(blend.end_frame, len(ctx.pairs))),
            key=lambda k: ctx.pairs[k].mean_abs_diff,
        )
        reasons.append(ReasonCode.MULTIFRAME_BLEND_PATTERN)
        if pair_index == peak_index:
            transition = TransitionEvidence(
                manuscript_type=None,
                status=TransitionStatus.REVIEW_REQUIRED,
                blend=blend,
                notes=["Sustained multi-frame blend: possible cross dissolve."],
            )
            return candidate.model_copy(
                update={
                    "status": CandidateStatus.REVIEW_REQUIRED,
                    "reason_codes": reasons,
                    "transition": transition,
                    "notes": notes,
                }
            )
        notes.append(f"Folded into blend region peak pair F{peak_index}.")
        return candidate.model_copy(
            update={
                "status": CandidateStatus.REJECTED,
                "reason_codes": reasons,
                "notes": notes,
            }
        )

    # Challenge: is this a flash frame?
    flash = _flash_for(ctx, candidate)
    if flash is not None:
        reasons.append(ReasonCode.FLASH_FRAME)
        before = flash.start_frame - 1
        after = flash.end_frame + 1
        if before >= 0 and after < n:
            phash_d, hist_d = _frame_similarity(ctx, before, after)
            if phash_d <= CONTINUITY_PHASH_MAX and hist_d <= CONTINUITY_HIST_MAX:
                reasons.append(ReasonCode.RETURN_TO_PREVIOUS_STATE)
                notes.append(
                    f"Content before F{before} and after F{after} the "
                    f"{flash.color} flash match (phash {phash_d}, hist {hist_d:.3f}); "
                    "flash is likely an in-shot effect, but a deliberate flash "
                    "transition cannot be ruled out deterministically."
                )
            else:
                notes.append(
                    f"Content differs across the {flash.color} flash "
                    f"(phash {phash_d}, hist {hist_d:.3f}); may be an editorial "
                    "transition."
                )
        return candidate.model_copy(
            update={
                "status": CandidateStatus.REVIEW_REQUIRED,
                "reason_codes": reasons,
                "notes": notes,
            }
        )

    # Challenge: coherent camera motion (pan/whip/shake)?
    coherent = (
        pair.flow_coherence >= COHERENT_MOTION_MIN
        and pair.flow_mean_mag >= MOTION_MAG_MIN
    )
    sustained = base.neighbor_motion >= SUSTAINED_MOTION_MIN
    if coherent and (sustained or pair.phash_hamming < SUPPORT_STRUCTURE_PHASH):
        if base.diff_z >= EXTREME_DIFF_Z and pair.phash_hamming >= 28:
            reasons.extend(
                [ReasonCode.COHERENT_CAMERA_MOTION, ReasonCode.HIGH_DISCONTINUITY]
            )
            notes.append(
                "Coherent motion present but discontinuity is extreme; a cut "
                "during fast motion cannot be excluded."
            )
            return candidate.model_copy(
                update={
                    "status": CandidateStatus.REVIEW_REQUIRED,
                    "reason_codes": reasons,
                    "notes": notes,
                }
            )
        reasons.append(ReasonCode.COHERENT_CAMERA_MOTION)
        notes.append(
            f"Coherent global motion (|flow| {pair.flow_mean_mag:.2f}, coherence "
            f"{pair.flow_coherence:.2f}); large difference explained by camera movement."
        )
        return candidate.model_copy(
            update={
                "status": CandidateStatus.REJECTED,
                "reason_codes": reasons,
                "notes": notes,
            }
        )

    # Challenge: zoom / scope-style radial motion?
    if sustained and _zoom_signature(ctx, pair_index):
        reasons.append(ReasonCode.SCOPE_OR_ZOOM_CONTINUITY)
        notes.append("Radial flow signature with sustained neighborhood motion: zoom/scope.")
        return candidate.model_copy(
            update={
                "status": CandidateStatus.REJECTED,
                "reason_codes": reasons,
                "notes": notes,
            }
        )

    # Challenge: does content shortly after the boundary resemble content
    # shortly before it (occlusion, effect, short insert — or a jump cut)?
    # Pixel similarity CANNOT prove temporal continuity, so a match is never
    # auto-rejected: real jump cuts and 1-frame editorial inserts must stay
    # visible to the reviewer (false-negative cuts are high risk).
    anomaly_len = _anomaly_length(ctx, candidate)
    if anomaly_len is not None:
        reasons.append(ReasonCode.RETURN_TO_PREVIOUS_STATE)
        if anomaly_len <= 2:
            reasons.append(ReasonCode.POSSIBLE_JUMP_CUT)
            notes.append(
                f"Content matches across a {anomaly_len}-frame anomaly: may be an "
                "in-shot effect, a very short editorial insert, or a jump cut. "
                "Deterministic evidence cannot decide."
            )
        else:
            reasons.extend([ReasonCode.LARGE_OCCLUSION, ReasonCode.EFFECT_CONTINUITY])
            notes.append(
                f"Content returns to a similar state after a {anomaly_len}-frame "
                "deviation: likely transient occlusion/effect, but a jump cut "
                "back to a similar composition cannot be excluded."
            )
        return candidate.model_copy(
            update={
                "status": CandidateStatus.REVIEW_REQUIRED,
                "reason_codes": reasons,
                "notes": notes,
            }
        )

    # Support evaluation: single-pair spike with quiet or coherent sides.
    left_quiet = right_quiet = True
    if pair_index - 1 >= 0:
        left_quiet = ctx.pairs[pair_index - 1].mean_abs_diff <= max(
            0.5 * pair.mean_abs_diff, base.diff_median * 3 + 1.0
        )
    if pair_index + 1 < len(ctx.pairs):
        right_quiet = ctx.pairs[pair_index + 1].mean_abs_diff <= max(
            0.5 * pair.mean_abs_diff, base.diff_median * 3 + 1.0
        )
    single_spike = left_quiet and right_quiet
    strong_outlier = base.diff_z >= SUPPORT_DIFF_Z or base.hist_z >= SUPPORT_DIFF_Z
    structural = pair.phash_hamming >= SUPPORT_STRUCTURE_PHASH
    multi_source = len(candidate.candidate_sources) >= 2

    if strong_outlier and single_spike and (structural or multi_source):
        reasons.extend([ReasonCode.HIGH_DISCONTINUITY, ReasonCode.LOCAL_OUTLIER])
        if single_spike:
            reasons.append(ReasonCode.SINGLE_PAIR_SPIKE)
            reasons.append(ReasonCode.SIDES_LOCALLY_STABLE)
        if multi_source:
            reasons.append(ReasonCode.MULTI_DETECTOR_SUPPORT)
        transition_status = TransitionStatus.PROPOSED
        manuscript_type: str | None = "Hard cut"
        transition_notes: list[str] = []
        if pair.phash_hamming <= JUMP_CUT_PHASH_MAX:
            # Similar composition across a supported discontinuity: possible jump cut.
            reasons.append(ReasonCode.POSSIBLE_JUMP_CUT)
            transition_status = TransitionStatus.REVIEW_REQUIRED
            manuscript_type = None
            transition_notes.append(
                "Composition is similar across the boundary; possible jump cut. "
                "Semantic verification required."
            )
        transition = TransitionEvidence(
            manuscript_type=manuscript_type,
            status=transition_status,
            audio_verification_required=ctx.has_audio,
            notes=transition_notes
            + (
                ["L-cut/J-cut cannot be ruled out without the audio engine (Phase 3)."]
                if ctx.has_audio
                else []
            ),
        )
        return candidate.model_copy(
            update={
                "status": CandidateStatus.SUPPORTED,
                "reason_codes": reasons,
                "transition": transition,
                "notes": notes,
            }
        )

    # Weak candidate: reject only when structure barely changed AND no
    # corroborating source; otherwise keep for human review (recall-first).
    if pair.phash_hamming < 10 and not multi_source and base.diff_z < SUPPORT_DIFF_Z:
        reasons.append(ReasonCode.INSUFFICIENT_EVIDENCE)
        notes.append("Low structural change, single weak signal; no edit evidence.")
        return candidate.model_copy(
            update={
                "status": CandidateStatus.REJECTED,
                "reason_codes": reasons,
                "notes": notes,
            }
        )

    reasons.append(ReasonCode.INSUFFICIENT_EVIDENCE)
    notes.append("Discontinuity is suspicious but no deterministic explanation fits.")
    return candidate.model_copy(
        update={
            "status": CandidateStatus.REVIEW_REQUIRED,
            "reason_codes": reasons,
            "notes": notes,
        }
    )


def build_fade_boundaries(
    ctx: VerifierContext, next_candidate_seq: int
) -> list[BoundaryCandidate]:
    """Compose supported fade-out→fade-in sequences into boundary candidates.

    Per Manuscript rules content visible during the outgoing fade belongs to
    the outgoing shot, so black/white hold frames attach to the outgoing shot
    and the boundary's right frame is the first frame of the incoming fade-in.
    A fade-out at end of media (or fade-in at start) is transition evidence but
    creates no boundary.
    """
    boundaries: list[BoundaryCandidate] = []
    outs = [f for f in ctx.fades if f.direction == "out"]
    ins = [f for f in ctx.fades if f.direction == "in"]
    seq = next_candidate_seq
    for fade_out in outs:
        following = [
            f
            for f in ins
            if f.target_color == fade_out.target_color
            and f.transition_start_frame > fade_out.transition_end_frame
        ]
        if not following:
            continue
        fade_in = min(following, key=lambda f: f.transition_start_frame)
        right_index = fade_in.transition_start_frame
        left_index = right_index - 1
        if left_index < 0:
            continue
        left_rec = ctx.ledger.frames[left_index]
        right_rec = ctx.ledger.frames[right_index]
        pair = ctx.pairs[left_index]
        base = ctx.baselines[left_index]
        transition = TransitionEvidence(
            manuscript_type="Fade in",
            status=TransitionStatus.PROPOSED,
            audio_verification_required=ctx.has_audio,
            fade=fade_in,
            notes=[
                f"Preceding fade-out to {fade_out.target_color} "
                f"F{fade_out.transition_start_frame}-F{fade_out.transition_end_frame}; "
                "hold frames attached to the outgoing shot per Manuscript "
                "outgoing-ownership rule.",
            ],
        )
        boundaries.append(
            BoundaryCandidate(
                candidate_id=f"cand_{seq:04d}",
                left_frame_index=left_index,
                right_frame_index=right_index,
                left_pts=left_rec.pts,
                right_pts=right_rec.pts,
                boundary_time_exact=right_rec.pts_time_seconds,
                boundary_time_manuscript=(
                    format_manuscript_display(right_rec.pts_time_seconds)
                    if right_rec.pts_time_seconds is not None
                    else None
                ),
                candidate_sources=["internal_fade"],
                metric_snapshot=pair,
                local_baseline=base,
                candidate_score=0.0,
                status=CandidateStatus.SUPPORTED,
                reason_codes=[ReasonCode.MULTIFRAME_FADE_PATTERN],
                transition=transition,
            )
        )
        seq += 1
    return boundaries


def _build_blend_candidates(
    ctx: VerifierContext, verified: list[BoundaryCandidate], next_seq: int
) -> list[BoundaryCandidate]:
    """A blend region with no surviving candidate could hide a real dissolve —
    synthesize a REVIEW_REQUIRED candidate at the blend's peak pair so the
    uncertainty stays visible (never optimize candidate count by hiding it)."""
    extras: list[BoundaryCandidate] = []
    seq = next_seq
    active_pairs = {
        c.left_frame_index for c in verified if c.status != CandidateStatus.REJECTED
    }
    for blend in ctx.blends:
        covered = any(
            blend.start_frame <= pair_index < blend.end_frame for pair_index in active_pairs
        )
        if covered:
            continue
        peak_index = max(
            range(blend.start_frame, min(blend.end_frame, len(ctx.pairs))),
            key=lambda k: ctx.pairs[k].mean_abs_diff,
        )
        pair = ctx.pairs[peak_index]
        right_rec = ctx.ledger.frames[pair.right_frame_index]
        extras.append(
            BoundaryCandidate(
                candidate_id=f"cand_{seq:04d}",
                left_frame_index=pair.left_frame_index,
                right_frame_index=pair.right_frame_index,
                left_pts=pair.left_pts,
                right_pts=pair.right_pts,
                boundary_time_exact=right_rec.pts_time_seconds,
                boundary_time_manuscript=(
                    format_manuscript_display(right_rec.pts_time_seconds)
                    if right_rec.pts_time_seconds is not None
                    else None
                ),
                candidate_sources=["internal_blend"],
                metric_snapshot=pair,
                local_baseline=ctx.baselines[peak_index],
                candidate_score=0.0,
                status=CandidateStatus.REVIEW_REQUIRED,
                reason_codes=[ReasonCode.MULTIFRAME_BLEND_PATTERN],
                transition=TransitionEvidence(
                    manuscript_type=None,
                    status=TransitionStatus.REVIEW_REQUIRED,
                    blend=blend,
                    notes=["Sustained multi-frame blend: possible cross dissolve."],
                ),
            )
        )
        seq += 1
    return extras


def verify_all(
    ctx: VerifierContext, candidates: list[BoundaryCandidate]
) -> list[BoundaryCandidate]:
    """Verify every candidate, then append composed fade and blend boundaries."""
    verified = [verify_candidate(ctx, candidate) for candidate in candidates]
    # Fade boundaries replace any raw candidates that fell inside fade regions
    # (those were rejected with MULTIFRAME_FADE_PATTERN above).
    fade_boundaries = build_fade_boundaries(ctx, len(verified) + 1)
    existing_pairs = {c.left_frame_index for c in verified if c.status != CandidateStatus.REJECTED}
    for boundary in fade_boundaries:
        if boundary.left_frame_index not in existing_pairs:
            verified.append(boundary)
    verified.extend(_build_blend_candidates(ctx, verified, len(verified) + 1))
    return sorted(verified, key=lambda c: c.left_frame_index)
