"""Phase 5 caption-eligibility tests (§100): CANDIDATE != FINAL FACT, enforced
structurally, with provenance inspection (§4)."""

from __future__ import annotations

from fractions import Fraction

from manuscript_reviewer.caption import eligibility as elig
from manuscript_reviewer.caption.eligibility import EligibilityContext
from manuscript_reviewer.models.audio import SourceVerificationStatus
from manuscript_reviewer.models.caption_brain import CaptionEligibility, EligibilityBasis
from manuscript_reviewer.models.review_intelligence import (
    ActionCandidate,
    ActionStateClass,
    CameraMotionCandidate,
    CameraMotionClass,
    DecisionType,
    SeedClaimType,
    SourceTextVerificationStatus,
    SpeedConclusion,
    TextConsensus,
    TextTrack,
)
from manuscript_reviewer.models.shot_truth import TransitionStatus

from .phase5_helpers import (
    build_ctx,
    human_decision,
    make_shot,
    make_speech_region,
    make_speed_evidence,
)


def test_unverified_asr_speech_is_ineligible() -> None:
    region = make_speech_region("SR-1", Fraction(0), Fraction(1), "hello there")
    status, basis, _ = elig.assess_speech(region)
    assert status == CaptionEligibility.INELIGIBLE
    assert basis is None


def test_human_verified_speech_is_eligible() -> None:
    region = make_speech_region(
        "SR-1", Fraction(0), Fraction(1), "hello there",
        status=SourceVerificationStatus.HUMAN_VERIFIED,
    )
    status, basis, _ = elig.assess_speech(region)
    assert status == CaptionEligibility.ELIGIBLE
    assert basis == EligibilityBasis.HUMAN_VERIFICATION_EVIDENCE


def test_human_corrected_speech_uses_corrected_text_only() -> None:
    region = make_speech_region(
        "SR-1", Fraction(0), Fraction(1), "machine words",
        status=SourceVerificationStatus.HUMAN_CORRECTED,
        corrected="the corrected words",
    )
    status, _, _ = elig.assess_speech(region)
    assert status == CaptionEligibility.ELIGIBLE
    assert region.caption_text == "the corrected words"


def test_rejected_asr_is_never_used() -> None:
    region = make_speech_region(
        "SR-1", Fraction(0), Fraction(1), "bad guess",
        status=SourceVerificationStatus.REJECTED,
    )
    status, _, _ = elig.assess_speech(region)
    assert status == CaptionEligibility.REJECTED


def _text_track(status: SourceTextVerificationStatus) -> TextTrack:
    return TextTrack(
        track_id="TT-1",
        first_candidate_frame=0,
        consensus=TextConsensus(consensus_text="best Drone that i've ever owned", support_frames=5),
        verification_status=status,
    )


def test_machine_ocr_is_ineligible() -> None:
    status, _, _ = elig.assess_on_screen_text(
        _text_track(SourceTextVerificationStatus.UNVERIFIED)
    )
    assert status == CaptionEligibility.INELIGIBLE


def test_human_verified_ocr_is_eligible() -> None:
    status, basis, _ = elig.assess_on_screen_text(
        _text_track(SourceTextVerificationStatus.HUMAN_VERIFIED)
    )
    assert status == CaptionEligibility.ELIGIBLE
    assert basis == EligibilityBasis.HUMAN_VERIFICATION_EVIDENCE


def test_regular_candidate_is_never_final_speed() -> None:
    evidence = make_speed_evidence(1, SpeedConclusion.REGULAR_CANDIDATE)
    status, _, reason = elig.assess_playback_speed(evidence, EligibilityContext())
    assert status == CaptionEligibility.REVIEW_REQUIRED
    assert "candidate" in reason


def test_mutated_regular_supported_enum_alone_is_not_trusted() -> None:
    """§4: an enum flipped to REGULAR_SUPPORTED without an applied human
    decision must NOT become a final caption fact."""
    evidence = make_speed_evidence(1, SpeedConclusion.REGULAR_SUPPORTED)
    status, _, reason = elig.assess_playback_speed(evidence, EligibilityContext())
    assert status == CaptionEligibility.REVIEW_REQUIRED
    assert "provenance" in reason


def test_applied_human_speed_decision_is_eligible() -> None:
    evidence = make_speed_evidence(1, SpeedConclusion.REGULAR_CANDIDATE)
    ctx = build_ctx(
        [human_decision("D-1", "SPEED-1", DecisionType.PLAYBACK_SPEED, "regular")],
        [evidence],
    )
    status, basis, _ = elig.assess_playback_speed(evidence, ctx)
    assert status == CaptionEligibility.ELIGIBLE
    assert basis == EligibilityBasis.APPLIED_HUMAN_DECISION


def _pickup_candidate(label: str | None = None) -> ActionCandidate:
    return ActionCandidate(
        candidate_id="AC-1",
        shot_number=1,
        action_class=ActionStateClass.CONTACT_BEGINS,
        semantic_label=label,
        start_frame=10,
        end_frame=20,
    )


def test_generic_action_candidate_is_ineligible() -> None:
    status, _, _ = elig.assess_visual_action(_pickup_candidate(), EligibilityContext())
    assert status == CaptionEligibility.INELIGIBLE


def test_semantic_label_without_applied_decision_is_not_final() -> None:
    """A semantic_label mutated onto the model without decision provenance is
    review-required, never final (§4/§37)."""
    status, _, reason = elig.assess_visual_action(
        _pickup_candidate("C1 picks up O1."), EligibilityContext()
    )
    assert status == CaptionEligibility.REVIEW_REQUIRED
    assert "not verification" in reason


def test_human_action_semantics_decision_is_eligible() -> None:
    candidate = _pickup_candidate("C1 places the right hand on O1.")
    from manuscript_reviewer.review.decisions import DecisionTargets, apply_decisions

    decision = human_decision(
        "D-2", "AC-1", DecisionType.ACTION_SEMANTICS, "C1 places the right hand on O1."
    )
    applications = apply_decisions(
        [decision],
        DecisionTargets(action_candidates={"AC-1": candidate}),
        "a" * 64,
        "1.3.0",
    )
    ctx = EligibilityContext.build([decision], applications)
    status, basis, _ = elig.assess_visual_action(candidate, ctx)
    assert status == CaptionEligibility.ELIGIBLE
    assert basis == EligibilityBasis.APPLIED_HUMAN_DECISION


def test_ambiguous_identity_blocks_action() -> None:
    candidate = _pickup_candidate("C1 picks up O1.")
    candidate.subject_track_ids = ["T-9"]
    ctx = EligibilityContext(ambiguous_track_ids=frozenset({"T-9"}))
    status, _, reason = elig.assess_visual_action(candidate, ctx)
    assert status == CaptionEligibility.INELIGIBLE
    assert "ambiguous" in reason


def test_unresolved_transition_blocks_final() -> None:
    shot = make_shot(
        2, Fraction(1), Fraction(2), None, TransitionStatus.UNRESOLVED
    )
    status, _, reason = elig.assess_transition(shot, EligibilityContext())
    assert status == CaptionEligibility.REVIEW_REQUIRED
    assert "Hard cut" in reason


def test_l_cut_requires_human_verification() -> None:
    """Audio crossing a boundary never proves an L-cut (§23)."""
    shot = make_shot(2, Fraction(1), Fraction(2), "L-cut", TransitionStatus.PROPOSED)
    status, _, reason = elig.assess_transition(shot, EligibilityContext())
    assert status == CaptionEligibility.REVIEW_REQUIRED
    assert "source relationship" in reason


def test_machine_language_guess_is_never_a_final_language_fact() -> None:
    from manuscript_reviewer.models.audio import LanguageEvidence, LanguageReviewStatus

    region = make_speech_region(
        "SR-1", Fraction(0), Fraction(1), "hola",
        status=SourceVerificationStatus.HUMAN_VERIFIED,
    )
    region.language = LanguageEvidence(
        language_candidate="es",
        language_probability=0.99,
        language_source="faster_whisper",
        language_review_status=LanguageReviewStatus.SUPPORTED_BY_ASR,
    )
    assert region.caption_language_eligible is False


def test_camera_motion_needs_human_classification() -> None:
    candidate = CameraMotionCandidate(
        candidate_id="CAM-1",
        shot_number=1,
        start_frame=0,
        last_supporting_frame=10,
        end_frame=11,
        motion_class=CameraMotionClass.HORIZONTAL_GLOBAL_MOTION,
        direction="screen-left",
        strength=1.0,
        inlier_ratio=0.9,
    )
    status, _, _ = elig.assess_camera_movement(candidate, EligibilityContext())
    assert status == CaptionEligibility.REVIEW_REQUIRED


def test_protected_trait_never_inferred() -> None:
    status, _, reason = elig.assess_seed_claim(
        SeedClaimType.PROTECTED_TRAIT, None, None, []
    )
    assert status == CaptionEligibility.INELIGIBLE
    assert "never inferred" in reason


def test_transient_candidate_never_becomes_semantic_sound() -> None:
    status, _, _ = elig.assess_sound_semantics(EligibilityContext(), "TRANS-1")
    assert status == CaptionEligibility.INELIGIBLE
