"""Phase 4 visual validators (P4-OBS-*).

Enforce that the visual observation ledger stays exactly aligned with the Phase 1
frame ledger and that every observation carries exact frame identity — the
invariant behind "exact frame identity controls all timing".
"""

from __future__ import annotations

import itertools

from ..models.frame import FrameLedger
from ..models.review_intelligence import (
    ActionCandidate,
    CameraMotionCandidate,
    CameraMotionClass,
    CharacterHypothesis,
    ContactEvent,
    ContinuityLink,
    EntityTrack,
    FinalStateCheck,
    FrameObservation,
    ObjectHypothesis,
    PlaybackSpeedEvidence,
    SpeedConclusion,
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


def _shot_of_frame(shot_result: ShotTruthResult | None, frame: int) -> int | None:
    if shot_result is None:
        return None
    for shot in shot_result.shots:
        if shot.start_frame_index <= frame <= shot.end_frame_index:
            return shot.shot_index
    return None


def validate_tracks(
    tracks: list[EntityTrack], frame_count: int, shot_result: ShotTruthResult | None = None
) -> list[ValidatorIssue]:
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
        # P4-ENTITY-007: an identity-ambiguous track cannot silently be verified
        # (FAIL only on the contradiction, never on ambiguity itself).
        if track.identity_ambiguous and track.status == TrackStatus.TRACKED:
            issues.append(
                ValidatorIssue(
                    rule_id="P4-ENTITY-007",
                    severity=Severity.FAIL,
                    location=track.track_id,
                    message="Identity-ambiguous track marked TRACKED (must stay REVIEW_REQUIRED).",
                )
            )
        # P4-ENTITY-005: a single track must not silently span a shot cut. Tracks are
        # shot-bounded; cross-shot continuity belongs to a review-required continuity
        # link, never one track carrying the same identity across a cut.
        if shot_result is not None:
            shots_touched = {
                s for o in track.observations
                if (s := _shot_of_frame(shot_result, o.frame_index)) is not None
            }
            if len(shots_touched) > 1:
                issues.append(
                    ValidatorIssue(
                        rule_id="P4-ENTITY-005",
                        severity=Severity.FAIL,
                        location=track.track_id,
                        message=(
                            f"Track spans {len(shots_touched)} shots "
                            f"{sorted(shots_touched)}; cross-shot continuity must be a "
                            "review-required link, not one track."
                        ),
                    )
                )
    return issues


def validate_continuity_links(links: list[ContinuityLink]) -> list[ValidatorIssue]:
    """P4-ENTITY-006: similar tracks are never auto-merged. A SAME_ENTITY_CANDIDATE
    (or REACQUISITION) link must stay review-required until a human decides."""
    issues: list[ValidatorIssue] = []
    for link in links:
        if link.relationship in ("SAME_ENTITY_CANDIDATE", "REACQUISITION") \
                and not link.review_required:
            issues.append(
                ValidatorIssue(
                    rule_id="P4-ENTITY-006",
                    severity=Severity.FAIL,
                    location=link.link_id,
                    message=(
                        f"{link.relationship} link marked resolved without review "
                        "(similar tracks must never be auto-merged)."
                    ),
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


def validate_action_candidates(
    candidates: list[ActionCandidate], frame_count: int, shots_exist: bool = False
) -> list[ValidatorIssue]:
    issues: list[ValidatorIssue] = []
    for cand in candidates:
        # P4-ACTION-001/002: valid, ordered exact frames.
        if not (0 <= cand.start_frame <= cand.end_frame < frame_count):
            issues.append(
                ValidatorIssue(
                    rule_id="P4-ACTION-001",
                    severity=Severity.FAIL,
                    location=cand.candidate_id,
                    message=f"Action frame range invalid: {cand.start_frame}->{cand.end_frame}.",
                )
            )
        # P4-ACTION-003: a semantic label may never be asserted as verified from
        # generic motion — a labelled candidate must still be review-required.
        if cand.semantic_label is not None and not cand.review_required:
            issues.append(
                ValidatorIssue(
                    rule_id="P4-ACTION-003",
                    severity=Severity.FAIL,
                    location=cand.candidate_id,
                    message="Semantic action label marked verified from generic motion.",
                )
            )
        # P4-ACTION-004: shot_number must be set when Shot Truth exists.
        if shots_exist and cand.shot_number is None:
            issues.append(
                ValidatorIssue(
                    rule_id="P4-ACTION-004",
                    severity=Severity.FAIL,
                    location=cand.candidate_id,
                    message="Action candidate missing shot_number while Shot Truth exists.",
                )
            )
        # P4-ACTION-005: a candidate must carry evidence refs.
        if not cand.evidence_refs:
            issues.append(
                ValidatorIssue(
                    rule_id="P4-ACTION-005",
                    severity=Severity.FAIL,
                    location=cand.candidate_id,
                    message="Action candidate has no evidence references.",
                )
            )
    return issues


def validate_contact_events(events: list[ContactEvent]) -> list[ValidatorIssue]:
    """P4-OBJECT-003: ownership/contact transitions are chronological per object."""
    issues: list[ValidatorIssue] = []
    by_obj: dict[str | None, list[int]] = {}
    for ev in events:
        by_obj.setdefault(ev.object_track_id, []).append(ev.frame_index)
    for obj, frames in by_obj.items():
        if frames != sorted(frames):
            issues.append(
                ValidatorIssue(
                    rule_id="P4-OBJECT-003",
                    severity=Severity.FAIL,
                    location=str(obj),
                    message="Contact/ownership events are not chronological.",
                )
            )
    return issues


def validate_final_states(
    checks: list[FinalStateCheck],
    object_tracks: list[EntityTrack],
    shot_result: ShotTruthResult | None,
) -> list[ValidatorIssue]:
    """P4-FINAL-001: coverage + evidence integrity for final states.

    * exactly one FinalStateCheck per anchored object that appears in a shot;
    * a resolved state must carry graded (frame/artifact) evidence;
    * removal/exit is never resolved without evidence.
    """
    issues: list[ValidatorIssue] = []
    have = {(c.shot_number, c.entity_id) for c in checks}
    counts: dict[tuple[int, str], int] = {}
    for c in checks:
        counts[(c.shot_number, c.entity_id)] = counts.get((c.shot_number, c.entity_id), 0) + 1

    if shot_result is not None:
        for shot in shot_result.shots:
            lo, hi = shot.start_frame_index, shot.end_frame_index
            for track in object_tracks:
                appears = any(lo <= o.frame_index <= hi for o in track.observations)
                if appears and (shot.shot_index, track.track_id) not in have:
                    issues.append(
                        ValidatorIssue(
                            rule_id="P4-FINAL-001",
                            severity=Severity.FAIL,
                            location=f"{track.track_id}@shot{shot.shot_index}",
                            message="Missing final-state check for an object in a shot.",
                        )
                    )
    for (shot_num, entity), n in counts.items():
        if n != 1:
            issues.append(
                ValidatorIssue(
                    rule_id="P4-FINAL-001",
                    severity=Severity.FAIL,
                    location=f"{entity}@shot{shot_num}",
                    message=f"{n} final-state checks for one object/shot (expected 1).",
                )
            )
    for check in checks:
        if check.resolved and not _graded_refs(check.evidence_refs):
            issues.append(
                ValidatorIssue(
                    rule_id="P4-FINAL-001",
                    severity=Severity.FAIL,
                    location=f"{check.entity_id}@shot{check.shot_number}",
                    message="Resolved final state without graded evidence.",
                )
            )
    return issues


def _graded_refs(refs: list) -> bool:  # type: ignore[type-arg]
    return any(
        r.start_frame is not None
        or r.start_pts is not None
        or bool(r.artifact_paths)
        or r.is_factual
        for r in refs
    )


def validate_speed_evidence(evidence: list[PlaybackSpeedEvidence]) -> list[ValidatorIssue]:
    issues: list[ValidatorIssue] = []
    for ev in evidence:
        # P4-SPEED-001: fast motion alone cannot prove accelerated playback.
        if ev.conclusion == SpeedConclusion.ACCELERATED_CANDIDATE and not ev.sustained_retiming:
            issues.append(
                ValidatorIssue(
                    rule_id="P4-SPEED-001",
                    severity=Severity.FAIL,
                    location=f"shot {ev.shot_number}",
                    message="ACCELERATED concluded without sustained retiming evidence.",
                )
            )
        # P4-SPEED-002: motion blur alone cannot prove slow motion — a slow-motion
        # candidate must rest on frame duplication cadence.
        if ev.conclusion == SpeedConclusion.SLOW_MOTION_CANDIDATE and (
            ev.duplicate_frame_ratio is None or ev.duplicate_frame_ratio <= 0.0
        ):
            issues.append(
                ValidatorIssue(
                    rule_id="P4-SPEED-002",
                    severity=Severity.FAIL,
                    location=f"shot {ev.shot_number}",
                    message="SLOW_MOTION concluded without frame-duplication evidence.",
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
