"""Segment per-pair global motion into coherent per-shot camera-motion phases.

Raw per-pair classifications stay sensitive (kept in ``camera_pair_metrics.csv``);
reviewer-facing phases are smoothed with hysteresis (O) so a 246-frame clip does
not explode into dozens of one-pair phases — WITHOUT suppressing a genuine short
camera reversal. A direction reversal is always two phases (P4-CAMERA-002); every
phase stays inside one shot (P4-CAMERA-001). Phase interval ends use the
frame-after-last-supporting boundary, never the last supporting frame's own start
(P).
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

from ..media.clock import AnnotationClock
from ..models.frame import FrameLedger
from ..models.review_intelligence import CameraMotionCandidate, CameraMotionClass
from ..models.shot_truth import ShotTruthResult
from ..shots.decode import GrayFrames
from .classify import PairClassification, classify_pair
from .global_motion import PairMotion, estimate_pair_motion, hanning_for

#: A reviewer-facing phase needs at least this many supporting pairs, UNLESS it
#: is a genuine reversal (see below).
_MIN_PHASE_PAIRS = 3
#: A direction reversal survives smoothing at this many pairs (short pans are
#: real and must not be absorbed).
_REVERSAL_MIN_PAIRS = 2

_MOVEMENT = frozenset(
    {
        CameraMotionClass.HORIZONTAL_GLOBAL_MOTION,
        CameraMotionClass.VERTICAL_GLOBAL_MOTION,
        CameraMotionClass.DIAGONAL_GLOBAL_MOTION,
        CameraMotionClass.SCALE_INCREASE,
        CameraMotionClass.SCALE_DECREASE,
        CameraMotionClass.ROTATION,
    }
)


@dataclass
class _Run:
    """A phase run. Its resolved class/direction are stored EXPLICITLY at
    construction — never re-derived from ``items[0]`` — so absorbing a short run
    into a neighbour (which reorders items) can never flip the target's class."""

    items: list[tuple[int, int, PairClassification]]
    motion_class: CameraMotionClass
    direction: str | None

    def __len__(self) -> int:
        return len(self.items)


def _new_run(items: list[tuple[int, int, PairClassification]]) -> _Run:
    cls = items[0][2]
    return _Run(items=items, motion_class=cls.motion_class, direction=cls.direction)


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
) -> tuple[list[CameraMotionCandidate], list[PairMotion]]:
    """Return (smoothed reviewer phases, raw per-pair motion)."""
    frame_count = gray.shape[0]
    if frame_count < 2:
        return [], []
    window = hanning_for((gray.shape[1], gray.shape[2]))
    candidates: list[CameraMotionCandidate] = []
    raw_pairs: list[PairMotion] = []
    counter = 0
    shots_by_index = (
        {s.shot_index: s for s in shot_result.shots} if shot_result is not None else {}
    )
    media_end = shot_result.annotation_endpoint_exact if shot_result is not None else None

    for shot_number, start, end in _shot_ranges(shot_result, frame_count):
        classifications: list[tuple[int, int, PairClassification]] = []
        for left in range(start, end):
            right = left + 1
            pm = estimate_pair_motion(gray[left], gray[right], left, right, window)
            raw_pairs.append(pm)
            classifications.append((left, right, classify_pair(pm)))
        runs = _smooth(_group_runs(classifications))
        proposal = shots_by_index.get(shot_number) if shot_number is not None else None
        shot_end_exact = proposal.end_exact if proposal is not None else media_end
        for run in runs:
            counter += 1
            candidates.append(
                _to_candidate(
                    counter, shot_number, run, ledger, clock, frame_count, end, shot_end_exact
                )
            )
    return candidates, raw_pairs


def _group_runs(
    classifications: list[tuple[int, int, PairClassification]],
) -> list[_Run]:
    runs: list[_Run] = []
    for item in classifications:
        cls = item[2]
        if runs and (
            cls.motion_class == runs[-1].motion_class and cls.direction == runs[-1].direction
        ):
            runs[-1].items.append(item)
        else:
            runs.append(_new_run([item]))
    return runs


def _is_protected_reversal(runs: list[_Run], index: int) -> bool:
    """A short movement run that reverses a neighbouring movement's direction is
    a real reversal and must survive smoothing."""
    run = runs[index]
    if run.motion_class not in _MOVEMENT or run.direction is None:
        return False
    if len(run) < _REVERSAL_MIN_PAIRS:
        return False
    for neighbour in (index - 1, index + 1):
        if 0 <= neighbour < len(runs):
            other = runs[neighbour]
            if (
                other.motion_class == run.motion_class
                and other.direction is not None
                and other.direction != run.direction
            ):
                return True
    return False


def _smooth(runs: list[_Run]) -> list[_Run]:
    """Absorb sub-threshold runs into neighbours until stable, preserving true
    reversals. Deterministic (no RNG, fixed iteration order)."""
    changed = True
    while changed and len(runs) > 1:
        changed = False
        for i, run in enumerate(runs):
            if len(run) >= _MIN_PHASE_PAIRS or _is_protected_reversal(runs, i):
                continue
            # Merge this short run into its longer neighbour (adopting its class).
            prev_len = len(runs[i - 1]) if i > 0 else -1
            next_len = len(runs[i + 1]) if i + 1 < len(runs) else -1
            if prev_len < 0 and next_len < 0:
                continue
            target = i - 1 if prev_len >= next_len else i + 1
            # Absorb the short run's items into the target but PRESERVE the
            # target's resolved class/direction (item 10 fix).
            runs[target].items = sorted(runs[target].items + run.items, key=lambda it: it[0])
            del runs[i]
            changed = True
            break
    return runs


def _to_candidate(
    counter: int,
    shot_number: int | None,
    run: _Run,
    ledger: FrameLedger,
    clock: AnnotationClock,
    frame_count: int,
    shot_end_frame: int,
    shot_end_exact: Fraction | None,
) -> CameraMotionCandidate:
    start_frame = run.items[0][0]
    last_supporting = run.items[-1][1]
    # P/11: interval end = presentation start of the frame AFTER the last
    # supporting frame; BUT when the phase runs through the shot's final owned
    # frame, the interval end is the shot's canonical end (item 11), not the
    # final frame's own start time.
    end_exact: Fraction | None
    if last_supporting >= shot_end_frame and shot_end_exact is not None:
        end_boundary = shot_end_frame
        end_exact = shot_end_exact
    else:
        end_boundary = min(last_supporting + 1, frame_count - 1)
        end_exact = _time(ledger, clock, end_boundary)
    strengths = [c.strength for _, _, c in run.items]
    responses = [c.response for _, _, c in run.items]
    movement = run.motion_class in _MOVEMENT
    return CameraMotionCandidate(
        candidate_id=f"CAM-{counter:04d}",
        shot_number=shot_number,
        start_frame=start_frame,
        last_supporting_frame=last_supporting,
        end_frame=end_boundary,
        start_exact=_time(ledger, clock, start_frame),
        end_exact=end_exact,
        motion_class=run.motion_class,
        direction=run.direction,
        strength=round(sum(strengths) / len(strengths), 5),
        inlier_ratio=round(sum(responses) / len(responses), 5),
        supporting_pair_ids=[f"{left}-{right}" for left, right, _ in run.items],
        review_required=movement,
    )
