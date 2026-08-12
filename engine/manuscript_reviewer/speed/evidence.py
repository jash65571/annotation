"""Per-shot playback-speed evidence conclusions (Y).

Conservative and safety-first: sustained frame near-duplication ALONGSIDE real
content motion is a SLOW_MOTION_CANDIDATE; irregular/absent evidence is
REVIEW_REQUIRED; otherwise REGULAR_SUPPORTED. Accelerated playback is NEVER
concluded from fast motion, and slow motion is NEVER concluded from motion blur —
both are left for review. A mid-shot speed change is only proposed with sustained
evidence (a durable duplicate-cadence shift between shot halves).
"""

from __future__ import annotations

from ..models.review_intelligence import PlaybackSpeedEvidence, SpeedConclusion
from ..models.shot_truth import ShotTruthResult
from ..shots.decode import GrayFrames
from .cadence import ShotCadence, shot_cadence

_SLOW_DUP_RATIO = 0.4
#: A sustained half-to-half duplicate-ratio shift needed to flag a speed change.
_CHANGE_DELTA = 0.35


def build_playback_speed_evidence(
    gray: GrayFrames, shot_truth: ShotTruthResult | None
) -> list[PlaybackSpeedEvidence]:
    if shot_truth is None:
        return []
    out: list[PlaybackSpeedEvidence] = []
    frame_count = gray.shape[0]
    for shot in shot_truth.shots:
        start = max(0, shot.start_frame_index)
        end = min(frame_count - 1, shot.end_frame_index)
        cadence = shot_cadence(gray, start, end)
        conclusion, review = _conclude(cadence)
        change_frames = _speed_change_frames(cadence, start)
        out.append(
            PlaybackSpeedEvidence(
                shot_number=shot.shot_index,
                duplicate_frame_ratio=cadence.duplicate_ratio,
                frame_spacing_regular=cadence.pair_count > 0,
                sustained_retiming=bool(change_frames),
                conclusion=conclusion,
                review_required=review,
                speed_change_frames=change_frames,
                notes=[
                    "fast motion is not accelerated playback; "
                    "motion blur is not slow motion",
                ],
            )
        )
    return out


def _conclude(cadence: ShotCadence) -> tuple[SpeedConclusion, bool]:
    if cadence.pair_count < 2:
        return SpeedConclusion.REVIEW_REQUIRED, True
    if cadence.duplicate_ratio >= _SLOW_DUP_RATIO and cadence.motion_present:
        # Frames held between real motion look like slow motion — candidate only.
        return SpeedConclusion.SLOW_MOTION_CANDIDATE, True
    return SpeedConclusion.REGULAR_SUPPORTED, False


def _speed_change_frames(cadence: ShotCadence, start: int) -> list[int]:
    if cadence.pair_count < 4:
        return []
    if abs(cadence.first_half_dup - cadence.second_half_dup) >= _CHANGE_DELTA:
        # Sustained cadence shift at the shot midpoint (candidate boundary).
        return [start + cadence.pair_count // 2]
    return []
