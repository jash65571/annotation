"""Segment per-pair global motion into per-shot camera-motion phases.

A phase is a maximal run of consecutive pairs sharing the same motion class and
direction. A direction reversal (screen-right then screen-left) therefore always
becomes two separate phases (P4-CAMERA-002); every phase stays inside one shot
(P4-CAMERA-001).
"""

from __future__ import annotations

from fractions import Fraction

from ..media.clock import AnnotationClock
from ..models.frame import FrameLedger
from ..models.review_intelligence import CameraMotionCandidate, CameraMotionClass
from ..models.shot_truth import ShotTruthResult
from ..shots.decode import GrayFrames
from .classify import PairClassification, classify_pair
from .global_motion import estimate_pair_motion, hanning_for


def _time(ledger: FrameLedger, clock: AnnotationClock, frame_index: int) -> Fraction | None:
    if 0 <= frame_index < ledger.frame_count:
        src = ledger.frames[frame_index].pts_time_seconds
        if src is not None:
            return clock.to_annotation(src)
    return None


def _shot_ranges(
    shot_result: ShotTruthResult | None, frame_count: int
) -> list[tuple[int | None, int, int]]:
    if shot_result is None or not shot_result.shots:
        return [(None, 0, frame_count - 1)]
    ranges: list[tuple[int | None, int, int]] = []
    for shot in shot_result.shots:
        start = max(0, shot.start_frame_index)
        end = min(frame_count - 1, shot.end_frame_index)
        if end > start:
            ranges.append((shot.shot_index, start, end))
    return ranges


def analyze_camera_motion(
    gray: GrayFrames,
    ledger: FrameLedger,
    clock: AnnotationClock,
    shot_result: ShotTruthResult | None,
) -> list[CameraMotionCandidate]:
    """Per-shot camera-motion phases from the shared gray metric grid."""
    frame_count = gray.shape[0]
    if frame_count < 2:
        return []
    window = hanning_for((gray.shape[1], gray.shape[2]))
    candidates: list[CameraMotionCandidate] = []
    counter = 0

    for shot_number, start, end in _shot_ranges(shot_result, frame_count):
        classifications: list[tuple[int, int, PairClassification]] = []
        for left in range(start, end):
            right = left + 1
            pm = estimate_pair_motion(gray[left], gray[right], left, right, window)
            classifications.append((left, right, classify_pair(pm)))
        for phase in _group_phases(classifications):
            counter += 1
            candidates.append(_to_candidate(counter, shot_number, phase, ledger, clock))
    return candidates


def _group_phases(
    classifications: list[tuple[int, int, PairClassification]],
) -> list[list[tuple[int, int, PairClassification]]]:
    phases: list[list[tuple[int, int, PairClassification]]] = []
    current: list[tuple[int, int, PairClassification]] = []
    for item in classifications:
        _, _, cls = item
        if not current:
            current = [item]
            continue
        _, _, prev_cls = current[-1]
        same = (
            cls.motion_class == prev_cls.motion_class
            and cls.direction == prev_cls.direction
        )
        if same:
            current.append(item)
        else:
            phases.append(current)
            current = [item]
    if current:
        phases.append(current)
    return phases


def _to_candidate(
    counter: int,
    shot_number: int | None,
    phase: list[tuple[int, int, PairClassification]],
    ledger: FrameLedger,
    clock: AnnotationClock,
) -> CameraMotionCandidate:
    start_frame = phase[0][0]
    end_frame = phase[-1][1]
    cls = phase[0][2]
    strengths = [c.strength for _, _, c in phase]
    responses = [c.response for _, _, c in phase]
    avg_strength = sum(strengths) / len(strengths)
    avg_response = sum(responses) / len(responses)
    movement = cls.motion_class not in (CameraMotionClass.STATIC, CameraMotionClass.UNRESOLVED)
    return CameraMotionCandidate(
        candidate_id=f"CAM-{counter:04d}",
        shot_number=shot_number,
        start_frame=start_frame,
        end_frame=end_frame,
        start_exact=_time(ledger, clock, start_frame),
        end_exact=_time(ledger, clock, end_frame),
        motion_class=cls.motion_class,
        direction=cls.direction,
        strength=round(avg_strength, 5),
        inlier_ratio=round(avg_response, 5),
        supporting_pair_ids=[f"{left}-{right}" for left, right, _ in phase],
        review_required=movement,
    )
