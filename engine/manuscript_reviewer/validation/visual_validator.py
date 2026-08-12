"""Phase 4 visual validators (P4-OBS-*).

Enforce that the visual observation ledger stays exactly aligned with the Phase 1
frame ledger and that every observation carries exact frame identity — the
invariant behind "exact frame identity controls all timing".
"""

from __future__ import annotations

import itertools

from ..models.frame import FrameLedger
from ..models.review_intelligence import (
    CameraMotionCandidate,
    CameraMotionClass,
    CharacterHypothesis,
    EntityTrack,
    FrameObservation,
    ObjectHypothesis,
    TrackStatus,
)
from ..models.shot_truth import ShotTruthResult
from ..models.validation import Severity, ValidatorIssue

_MOVEMENT_MIN_RESPONSE = 0.10


def validate_frame_observations(
    observations: list[FrameObservation], ledger: FrameLedger
) -> list[ValidatorIssue]:
    issues: list[ValidatorIssue] = []
    if len(observations) != ledger.frame_count:
        issues.append(
            ValidatorIssue(
                rule_id="P4-OBS-001",
                severity=Severity.FAIL,
                location="frame_observations",
                message=(
                    f"Observation count {len(observations)} != ledger frame count "
                    f"{ledger.frame_count}; refusing to mislabel visual evidence."
                ),
            )
        )
        return issues
    for i, obs in enumerate(observations):
        if obs.frame_index != i:
            issues.append(
                ValidatorIssue(
                    rule_id="P4-OBS-002",
                    severity=Severity.FAIL,
                    location=f"observation {i}",
                    message=f"Observation out of order: frame_index {obs.frame_index} at row {i}.",
                )
            )
        # Exact frame identity: a frame with a PTS must retain its exact time.
        record = ledger.frames[i]
        if record.pts_time_seconds is not None and obs.source_pts_time_exact is None:
            issues.append(
                ValidatorIssue(
                    rule_id="P4-OBS-003",
                    severity=Severity.FAIL,
                    location=f"observation {i}",
                    message="Observation lost its exact source frame time.",
                )
            )
    return issues


_MOVEMENT_CLASSES = frozenset(
    {
        CameraMotionClass.HORIZONTAL_GLOBAL_MOTION,
        CameraMotionClass.VERTICAL_GLOBAL_MOTION,
        CameraMotionClass.DIAGONAL_GLOBAL_MOTION,
        CameraMotionClass.SCALE_INCREASE,
        CameraMotionClass.SCALE_DECREASE,
        CameraMotionClass.ROTATION,
    }
)


def validate_camera_candidates(
    candidates: list[CameraMotionCandidate],
    shot_result: ShotTruthResult | None,
    frame_count: int,
) -> list[ValidatorIssue]:
    issues: list[ValidatorIssue] = []
    shot_ranges: dict[int, tuple[int, int]] = {}
    if shot_result is not None:
        shot_ranges = {
            s.shot_index: (s.start_frame_index, s.end_frame_index) for s in shot_result.shots
        }

    for cand in candidates:
        # Valid ordered frames: start <= last supporting, and the interval end
        # boundary is the frame after the last supporting frame (P).
        if not (0 <= cand.start_frame <= cand.last_supporting_frame < frame_count):
            issues.append(
                ValidatorIssue(
                    rule_id="P4-CAMERA-001",
                    severity=Severity.FAIL,
                    location=cand.candidate_id,
                    message=f"Camera phase frame range invalid: "
                    f"{cand.start_frame}->{cand.last_supporting_frame}.",
                )
            )
        # P4-CAMERA-001: the SUPPORTING frames must stay inside one shot (the
        # interval-end boundary may legitimately be the next shot's first frame).
        if cand.shot_number is not None and cand.shot_number in shot_ranges:
            lo, hi = shot_ranges[cand.shot_number]
            if cand.start_frame < lo or cand.last_supporting_frame > hi:
                issues.append(
                    ValidatorIssue(
                        rule_id="P4-CAMERA-001",
                        severity=Severity.FAIL,
                        location=cand.candidate_id,
                        message=f"Camera phase crosses shot {cand.shot_number} boundary.",
                    )
                )
        # P4-CAMERA-003: a movement phase must be supported by a real global
        # correlation (foreground-only motion cannot become camera motion).
        if cand.motion_class in _MOVEMENT_CLASSES and cand.inlier_ratio < _MOVEMENT_MIN_RESPONSE:
            issues.append(
                ValidatorIssue(
                    rule_id="P4-CAMERA-003",
                    severity=Severity.WARN,
                    location=cand.candidate_id,
                    message="Camera movement phase has weak global-motion support.",
                )
            )
    return issues


def validate_tracks(tracks: list[EntityTrack], frame_count: int) -> list[ValidatorIssue]:
    issues: list[ValidatorIssue] = []
    for track in tracks:
        # P4-ENTITY-001: track frame ranges stay inside valid frames.
        for obs in track.observations:
            if not (0 <= obs.frame_index < frame_count):
                issues.append(
                    ValidatorIssue(
                        rule_id="P4-ENTITY-001",
                        severity=Severity.FAIL,
                        location=track.track_id,
                        message=f"Track observation frame {obs.frame_index} out of range.",
                    )
                )
        # P4-ENTITY-002: chronological observations are ordered.
        frames = [o.frame_index for o in track.observations]
        if frames != sorted(frames):
            issues.append(
                ValidatorIssue(
                    rule_id="P4-ENTITY-002",
                    severity=Severity.FAIL,
                    location=track.track_id,
                    message="Track observations are not in chronological order.",
                )
            )
        # P4-ENTITY-003: a reacquired identity cannot silently become verified.
        if track.reacquired and track.status == TrackStatus.TRACKED:
            issues.append(
                ValidatorIssue(
                    rule_id="P4-ENTITY-003",
                    severity=Severity.FAIL,
                    location=track.track_id,
                    message="Reacquired track marked TRACKED (must stay REVIEW_REQUIRED).",
                )
            )
    return issues


def validate_hypotheses(
    characters: list[CharacterHypothesis], objects: list[ObjectHypothesis]
) -> list[ValidatorIssue]:
    issues: list[ValidatorIssue] = []
    # P4-ENTITY-004: C IDs proposed in verified first-appearance order.
    firsts = [c.first_visible_frame for c in characters if c.first_visible_frame is not None]
    if firsts != sorted(firsts):
        issues.append(
            ValidatorIssue(
                rule_id="P4-ENTITY-004",
                severity=Severity.FAIL,
                location="character_hypotheses",
                message="Character labels not in verified first-appearance order.",
            )
        )
    # P4-OBJECT-001: one physical track cannot silently map to two O IDs.
    track_to_labels: dict[str, set[str]] = {}
    for obj in objects:
        for tid in obj.track_ids:
            track_to_labels.setdefault(tid, set()).add(obj.proposed_label or obj.hypothesis_id)
    for tid, labels in track_to_labels.items():
        if len(labels) > 1:
            issues.append(
                ValidatorIssue(
                    rule_id="P4-OBJECT-001",
                    severity=Severity.FAIL,
                    location=tid,
                    message=f"One physical track maps to multiple O IDs: {sorted(labels)}.",
                )
            )
    return issues


def count_direction_reversals(candidates: list[CameraMotionCandidate]) -> int:
    """Count screen-left <-> screen-right reversals within each shot (P4-CAMERA-002
    guarantees these are already separate phases)."""
    reversals = 0
    by_shot: dict[int | None, list[CameraMotionCandidate]] = {}
    for cand in candidates:
        by_shot.setdefault(cand.shot_number, []).append(cand)
    for phases in by_shot.values():
        horiz = [
            c for c in phases
            if c.motion_class == CameraMotionClass.HORIZONTAL_GLOBAL_MOTION and c.direction
        ]
        for prev, nxt in itertools.pairwise(horiz):
            if prev.direction != nxt.direction:
                reversals += 1
    return reversals
