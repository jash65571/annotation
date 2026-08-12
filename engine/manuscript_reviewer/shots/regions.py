"""Flash, fade, and blend region detection from per-frame stats + pair metrics.

These detectors produce EVIDENCE (regions with frame anchors), never final
transitions. A single bright frame may be a muzzle flash, an ability effect, a
white transition frame, or a corrupted frame — the deterministic system records
the pattern and lets the verifier / human decide.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..models.frame import FrameLedger
from ..models.shot_truth import (
    BlendEvidence,
    CandidateStatus,
    FadeEvidence,
    FrameStats,
    PairMetrics,
)

#: Minimum frames of monotonic luma trend to call a fade pattern.
MIN_FADE_FRAMES = 3
#: Mean-luma slope per frame (0-255) considered a deliberate fade trend.
MIN_FADE_STEP = 4.0
#: Sustained elevated-change run length for a blend/dissolve pattern.
MIN_BLEND_PAIRS = 3


@dataclass(frozen=True)
class FlashRegion:
    """A run of 1+ consecutive flat near-black/near-white frames with normal
    frames on both sides."""

    start_frame: int
    end_frame: int  # inclusive
    color: str  # "black" | "white"


def detect_flash_regions(stats: list[FrameStats]) -> list[FlashRegion]:
    regions: list[FlashRegion] = []
    n = len(stats)
    i = 0
    while i < n:
        flat_black = stats[i].near_black
        flat_white = stats[i].near_white
        if not (flat_black or flat_white):
            i += 1
            continue
        color = "black" if flat_black else "white"
        j = i
        while j + 1 < n and (
            (color == "black" and stats[j + 1].near_black)
            or (color == "white" and stats[j + 1].near_white)
        ):
            j += 1
        has_left = i > 0 and not (stats[i - 1].near_black or stats[i - 1].near_white)
        has_right = j < n - 1 and not (stats[j + 1].near_black or stats[j + 1].near_white)
        # A "flash" needs normal content on at least one side; a clip that is
        # entirely black is not a flash.
        if has_left or has_right:
            regions.append(FlashRegion(start_frame=i, end_frame=j, color=color))
        i = j + 1
    return regions


def detect_fades(
    stats: list[FrameStats], ledger: FrameLedger
) -> list[FadeEvidence]:
    """Multi-frame monotonic luma trends into/out of near-black/near-white."""
    fades: list[FadeEvidence] = []
    n = len(stats)

    def pts(idx: int) -> int | None:
        return ledger.frames[idx].pts

    i = 0
    while i < n - MIN_FADE_FRAMES:
        # Fade OUT: monotonic decrease (to black) or increase (to white)
        # ending in a flat frame.
        for direction, target in (("out", "black"), ("out", "white")):
            sign = -1.0 if target == "black" else 1.0
            j = i
            while (
                j + 1 < n
                and sign * (stats[j + 1].luma_mean - stats[j].luma_mean) >= MIN_FADE_STEP
            ):
                j += 1
            run = j - i
            if run >= MIN_FADE_FRAMES - 1:
                ends_flat = (
                    stats[j].near_black if target == "black" else stats[j].near_white
                )
                if ends_flat:
                    fades.append(
                        FadeEvidence(
                            direction=direction,
                            target_color=target,
                            transition_start_frame=i,
                            transition_end_frame=j,
                            start_pts=pts(i),
                            end_pts=pts(j),
                            status=CandidateStatus.SUPPORTED,
                        )
                    )
                    i = j
                    break
        else:
            # Fade IN: starts flat, monotonic away from the flat color.
            for target in ("black", "white"):
                starts_flat = (
                    stats[i].near_black if target == "black" else stats[i].near_white
                )
                if not starts_flat:
                    continue
                sign = 1.0 if target == "black" else -1.0
                j = i
                while (
                    j + 1 < n
                    and sign * (stats[j + 1].luma_mean - stats[j].luma_mean)
                    >= MIN_FADE_STEP
                ):
                    j += 1
                if j - i >= MIN_FADE_FRAMES - 1:
                    fades.append(
                        FadeEvidence(
                            direction="in",
                            target_color=target,
                            transition_start_frame=i,
                            transition_end_frame=j,
                            start_pts=pts(i),
                            end_pts=pts(j),
                            status=CandidateStatus.SUPPORTED,
                        )
                    )
                    i = j
                    break
        i += 1
    return fades


def detect_blends(
    pairs: list[PairMetrics], baselines_diff_median: list[float], ledger: FrameLedger
) -> list[BlendEvidence]:
    """Sustained elevated-change runs without a single dominant spike —
    the conservative cross-dissolve signature. Always REVIEW_REQUIRED.

    The elevation reference is the CLIP-GLOBAL median pair difference, not the
    local window median: a sustained dissolve raises its own local baseline,
    so a local reference can never fire inside the very pattern it should
    detect (baseline self-contamination).
    """
    blends: list[BlendEvidence] = []
    n = len(pairs)
    if n == 0:
        return blends
    ordered = sorted(p.mean_abs_diff for p in pairs)
    global_median = ordered[n // 2]
    threshold = max(2.5 * global_median, 4.0)
    i = 0
    while i < n:
        if pairs[i].mean_abs_diff < threshold or pairs[i].flow_coherence > 0.75:
            i += 1
            continue
        j = i
        while (
            j + 1 < n
            and pairs[j + 1].mean_abs_diff >= threshold
            and pairs[j + 1].flow_coherence <= 0.75
        ):
            j += 1
        run = j - i + 1
        if run >= MIN_BLEND_PAIRS:
            run_values = [pairs[k].mean_abs_diff for k in range(i, j + 1)]
            peak = max(run_values)
            mean_run = sum(run_values) / run
            # A dominant single spike is hard-cut-like, not a blend.
            if peak < 2.5 * mean_run:
                blends.append(
                    BlendEvidence(
                        start_frame=pairs[i].left_frame_index,
                        end_frame=pairs[j].right_frame_index,
                        start_pts=ledger.frames[pairs[i].left_frame_index].pts,
                        end_pts=ledger.frames[pairs[j].right_frame_index].pts,
                        sustained_pairs=run,
                        status=CandidateStatus.REVIEW_REQUIRED,
                    )
                )
        i = j + 1
    return blends
