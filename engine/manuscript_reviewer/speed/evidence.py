"""Per-shot playback-speed evidence conclusions (Y, item 9 + final-fix 1).

Evidence-honest: uniform PTS/frame cadence proves the ENCODED OUTPUT cadence, NOT
the original playback speed — a 2x/0.5x/speed-ramped source re-encodes to perfectly
uniform CFR timestamps. So the video path NEVER asserts REGULAR_SUPPORTED; its
honest best is REGULAR_CANDIDATE (output cadence consistent with regular, original
speed unconfirmed, never factual). REGULAR_SUPPORTED is reachable only through
human/source confirmation (a PLAYBACK_SPEED decision). Insufficient/ambiguous
evidence defaults to REVIEW_REQUIRED. Fast motion is never accelerated; motion blur
is never slow motion. ACCELERATED_CANDIDATE needs sustained retiming evidence.
"""

from __future__ import annotations

from itertools import pairwise

from ..models.frame import FrameLedger
from ..models.review_intelligence import PlaybackSpeedEvidence, SpeedConclusion
from ..models.shot_truth import ShotTruthResult
from ..shots.decode import GrayFrames
from .cadence import ShotCadence, shot_cadence

_SLOW_DUP_RATIO = 0.4
_CHANGE_DELTA = 0.35


def _pts_regular(ledger: FrameLedger, start: int, end: int) -> bool:
    """Positive-evidence check: frame durations are uniform (CFR) across the shot.

    Uses exact PTS-time deltas — uniform spacing is positive evidence of regular
    playback; VFR / irregular spacing is not (so it cannot confirm 'regular')."""
    times = [
        ledger.frames[i].pts_time_seconds
        for i in range(start, end + 1)
        if 0 <= i < ledger.frame_count and ledger.frames[i].pts_time_seconds is not None
    ]
    if len(times) < 3:
        return False
    deltas = [b - a for a, b in pairwise(times)]  # type: ignore[operator]
    first = deltas[0]
    if first <= 0:
        return False
    tol = first / 20  # 5% tolerance
    return all(abs(d - first) <= tol for d in deltas)


def conclude_speed(
    cadence: ShotCadence, pts_regular: bool, accelerated_evidence: bool = False
) -> tuple[SpeedConclusion, bool]:
    """Pure decision over cadence metrics (unit-testable, incl. ACCELERATED)."""
    if cadence.pair_count < 2:
        return SpeedConclusion.REVIEW_REQUIRED, True
    if cadence.duplicate_ratio >= _SLOW_DUP_RATIO and cadence.motion_present:
        # Frames held between real motion look like slow motion — candidate only.
        return SpeedConclusion.SLOW_MOTION_CANDIDATE, True
    if accelerated_evidence:
        return SpeedConclusion.ACCELERATED_CANDIDATE, True
    # Uniform PTS cadence (and no slow-motion duplication under motion) is consistent
    # with regular playback but only proves the ENCODED OUTPUT cadence — a retimed
    # source re-encodes to uniform CFR too. The video path therefore concludes
    # REGULAR_CANDIDATE, never REGULAR_SUPPORTED (that needs human/source evidence).
    # It is not review-required on its own, so a clean clip is not flagged; a seed
    # 'regular' claim it is compared against becomes PARTIALLY_SUPPORTED, not proven.
    if pts_regular and (not cadence.motion_present or cadence.duplicate_ratio < _SLOW_DUP_RATIO):
        return SpeedConclusion.REGULAR_CANDIDATE, False
    return SpeedConclusion.REVIEW_REQUIRED, True


def build_playback_speed_evidence(
    gray: GrayFrames, shot_truth: ShotTruthResult | None, ledger: FrameLedger | None = None
) -> list[PlaybackSpeedEvidence]:
    if shot_truth is None:
        return []
    out: list[PlaybackSpeedEvidence] = []
    frame_count = gray.shape[0]
    for shot in shot_truth.shots:
        start = max(0, shot.start_frame_index)
        end = min(frame_count - 1, shot.end_frame_index)
        cadence = shot_cadence(gray, start, end)
        pts_regular = _pts_regular(ledger, start, end) if ledger is not None else False
        conclusion, review = conclude_speed(cadence, pts_regular)
        change_frames = _speed_change_frames(cadence, start)
        out.append(
            PlaybackSpeedEvidence(
                shot_number=shot.shot_index,
                duplicate_frame_ratio=cadence.duplicate_ratio,
                frame_spacing_regular=pts_regular,
                sustained_retiming=bool(change_frames),
                conclusion=conclusion,
                review_required=review,
                speed_change_frames=change_frames,
                notes=[
                    "REGULAR needs positive PTS cadence evidence; fast motion is "
                    "not accelerated playback; motion blur is not slow motion",
                ],
            )
        )
    return out


def _speed_change_frames(cadence: ShotCadence, start: int) -> list[int]:
    if cadence.pair_count < 4:
        return []
    if abs(cadence.first_half_dup - cadence.second_half_dup) >= _CHANGE_DELTA:
        return [start + cadence.pair_count // 2]
    return []
