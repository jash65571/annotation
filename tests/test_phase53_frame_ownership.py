"""Phase 5.3 frame-ownership regressions: ShotProposal frame ownership is
INCLUSIVE, so shot.end_frame_index + 1 belongs to the next shot and is never an
event-owned frame. Temporal intervals may still end exactly at shot.end_exact —
exact boundaries and frame ownership are separate concepts."""

from __future__ import annotations

from fractions import Fraction
from pathlib import Path

from manuscript_reviewer.caption import eligibility as elig
from manuscript_reviewer.caption.eligibility import ShotBounds
from manuscript_reviewer.models.caption_brain import (
    CaptionEligibility,
    CaptionFactType,
    HumanCaptionFact,
)
from manuscript_reviewer.models.review_intelligence import (
    ActionCandidate,
    ActionStateClass,
    DecisionOutcome,
    DecisionType,
    TextTrack,
)
from manuscript_reviewer.review.decisions import DecisionTargets, apply_decisions

from .phase5_helpers import RULES_VERSION, VIDEO_SHA, factual_ref, human_decision

#: Shot 1 owns frames 0-23 (0.0s-1.0s); Shot 2 owns frames 24-47 (1.0s-2.0s).
_SHOT_RANGES = {1: (0, 23), 2: (24, 47)}
_BOUNDS = {
    1: ShotBounds(start_exact=Fraction(0), end_exact=Fraction(1), start_frame=0, end_frame=23),
    2: ShotBounds(start_exact=Fraction(1), end_exact=Fraction(2), start_frame=24, end_frame=47),
}


def _frame_time(i: int) -> Fraction | None:
    return Fraction(i, 24) if 0 <= i < 48 else None


def _apply_boundary(value: str) -> tuple[DecisionOutcome, ActionCandidate]:
    candidate = ActionCandidate(
        candidate_id="AC-1", shot_number=1,
        action_class=ActionStateClass.CONTACT_BEGINS, start_frame=0, end_frame=5,
    )
    apps = apply_decisions(
        [human_decision("D-AB", "AC-1", DecisionType.ACTION_BOUNDARY, value)],
        DecisionTargets(
            action_candidates={"AC-1": candidate},
            frame_to_time=_frame_time,
            shot_frame_ranges=_SHOT_RANGES,
        ),
        VIDEO_SHA, RULES_VERSION,
    )
    return apps[0].outcome, candidate


def test_action_boundary_claiming_next_shots_frame_is_invalid() -> None:
    outcome, candidate = _apply_boundary("10-24")
    assert outcome == DecisionOutcome.INVALID_VALUE
    assert (candidate.start_frame, candidate.end_frame) == (0, 5)


def test_action_boundary_to_last_owned_frame_is_valid() -> None:
    outcome, candidate = _apply_boundary("10-23")
    assert outcome == DecisionOutcome.APPLIED
    assert (candidate.start_frame, candidate.end_frame) == (10, 23)


def _apply_text_timing(value: str) -> tuple[DecisionOutcome, TextTrack]:
    track = TextTrack(track_id="TT-1", first_candidate_frame=0)
    apps = apply_decisions(
        [human_decision("D-TT", "TT-1", DecisionType.TEXT_TIMING, value)],
        DecisionTargets(
            text_tracks={"TT-1": track},
            frame_to_time=_frame_time,
            shot_frame_ranges=_SHOT_RANGES,
        ),
        VIDEO_SHA, RULES_VERSION,
    )
    return apps[0].outcome, track


def test_text_timing_claiming_next_shots_frame_is_invalid() -> None:
    outcome, track = _apply_text_timing("10-24")
    assert outcome == DecisionOutcome.INVALID_VALUE
    assert track.first_stable_frame is None


def test_text_timing_to_last_owned_frame_is_valid() -> None:
    outcome, track = _apply_text_timing("10-23")
    assert outcome == DecisionOutcome.APPLIED
    assert (track.first_stable_frame, track.last_stable_frame) == (10, 23)


def _camera_fact(start_frame: int, end_frame: int) -> HumanCaptionFact:
    return HumanCaptionFact(
        fact_id="HF-CAM",
        fact_type=CaptionFactType.CAMERA_MOVEMENT,
        text_value="The camera view moves screen-left.",
        shot_number=1,
        start_frame=start_frame,
        end_frame=end_frame,
        evidence_refs=[factual_ref("EV-HF-CAM")],
        decided_by="reviewer@test",
        bound_video_sha256=VIDEO_SHA,
        bound_rules_version=RULES_VERSION,
    )


def test_human_camera_fact_claiming_next_shots_frame_is_blocked() -> None:
    status, _, reason = elig.assess_human_fact(_camera_fact(10, 24), _BOUNDS, _frame_time)
    assert status == CaptionEligibility.REVIEW_REQUIRED
    assert "frame ownership is inclusive" in reason


def test_human_camera_fact_to_last_owned_frame_is_allowed() -> None:
    status, _, _ = elig.assess_human_fact(_camera_fact(10, 23), _BOUNDS, _frame_time)
    assert status == CaptionEligibility.ELIGIBLE


def test_temporal_boundary_may_equal_shot_end_without_claiming_next_frame(
    tmp_path: Path,
) -> None:
    """An event whose last owned frame is 23 may still END temporally at Shot 1
    end_exact (== Shot 2 start_exact) — the boundary lives on the clock, never
    in the next shot's frame ownership."""
    fact = HumanCaptionFact(
        fact_id="HF-END",
        fact_type=CaptionFactType.VISUAL_ACTION,
        text_value="C1 holds the pose.",
        shot_number=1,
        character_ids=["C1"],
        start_frame=10,
        end_frame=23,
        start_exact=Fraction(10, 24),
        end_exact=Fraction(1),  # exactly Shot 1 end_exact / Shot 2 start_exact
        evidence_refs=[factual_ref("EV-HF-END")],
        decided_by="reviewer@test",
        bound_video_sha256=VIDEO_SHA,
        bound_rules_version=RULES_VERSION,
    )
    status, _, reason = elig.assess_human_fact(fact, _BOUNDS, _frame_time)
    assert status == CaptionEligibility.ELIGIBLE, reason
