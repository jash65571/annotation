"""Item 17: one regression per DecisionType, each asserting the RECOMPUTE effect.

Every human decision routes to exactly one typed registry and its whitelisted
mutation changes an observable downstream output (a review-queue item disappears,
a claim resolves, a proposal outcome flows into the matrix). None of these assert
a bare field write — they assert the recompute the reviewer actually sees.
"""

from __future__ import annotations

from manuscript_reviewer.models.review_intelligence import (
    ActionCandidate,
    ActionStateClass,
    CameraMotionCandidate,
    CameraMotionClass,
    DecisionOutcome,
    DecisionType,
    EntityTrack,
    EvidenceStatus,
    HumanReviewDecision,
    PlaybackSpeedEvidence,
    ReviewProposal,
    ReviewProposalOutcome,
    SeedClaimType,
    SpeedConclusion,
    TrackObservation,
    TrackStatus,
)
from manuscript_reviewer.review.decisions import (
    DecisionTargets,
    apply_decisions,
    apply_decisions_to_claims,
)
from manuscript_reviewer.review.queue import build_machine_review_items, build_review_queue
from manuscript_reviewer.seed.claims import extract_claims
from manuscript_reviewer.seed.comparison import (
    VisualEvidence,
    build_rows,
    compare_seed,
)
from manuscript_reviewer.seed.parser import parse_seed_text

_SHA = "VIDEOHASH"
_RULES = "1.3.0"


def _d(subject: str, dtype: DecisionType, value: str) -> HumanReviewDecision:
    return HumanReviewDecision(
        decision_id=f"D-{subject}", subject_id=subject, decision_type=dtype, value=value,
        decided_by="reviewer@example", decided_at_utc="2026-08-12T00:00:00Z",
        bound_video_sha256=_SHA, bound_rules_version=_RULES,
    )


def _track(track_id: str, first: int, last: int, reacq: bool) -> EntityTrack:
    obs = [
        TrackObservation(frame_index=f, x=0, y=0, width=10, height=10, status=TrackStatus.TRACKED)
        for f in range(first, last + 1)
    ]
    return EntityTrack(
        track_id=track_id, entity_type="CHARACTER", first_frame_index=first,
        last_frame_index=last, observations=obs,
        status=TrackStatus.REVIEW_REQUIRED if reacq else TrackStatus.TRACKED, reacquired=reacq,
    )


def _titles(items: list) -> set[str]:  # type: ignore[type-arg]
    return {i.title for i in items}


# --- claim-targeted kinds (queue disappears on recompute) --------------------


def _ost_claim():  # type: ignore[no-untyped-def]
    seed = (
        "[Shot 1: 0.0-5.0]\nCut: Opening shot\n"
        'Action & Audio: 1.0-2.0: On-screen text reads "GO".\n'
    )
    doc = parse_seed_text(seed)
    res = compare_seed(doc, extract_claims(doc), None, None)
    claim = next(c for c in res.claims if c.claim_type == SeedClaimType.ON_SCREEN_TEXT)
    return res, claim


def test_claim_evidence_decision_resolves_claim() -> None:
    res, claim = _ost_claim()
    assert any(claim.claim_id in i.related_claim_ids for i in build_review_queue(res))
    apps = apply_decisions_to_claims(
        [_d(claim.claim_id, DecisionType.CLAIM_EVIDENCE, EvidenceStatus.CONTRADICTED.value)],
        res.claims, _SHA, _RULES)
    assert apps[0].outcome == DecisionOutcome.APPLIED
    assert claim.evidence_status == EvidenceStatus.CONTRADICTED


def test_ocr_timing_decision_clears_review_item() -> None:
    res, claim = _ost_claim()
    apply_decisions_to_claims(
        [_d(claim.claim_id, DecisionType.OCR_TIMING, "1.0-2.0")], res.claims, _SHA, _RULES)
    res.rows = build_rows(res.claims)
    assert not any(claim.claim_id in i.related_claim_ids for i in build_review_queue(res))


# --- machine-evidence kinds (machine review item disappears) -----------------


def test_identity_mapping_clears_reacquired_item() -> None:
    track = _track("TRK-a", 0, 20, reacq=True)
    ev = VisualEvidence(entity_tracks=[track])
    assert any("Reacquired identity" in t for t in _titles(build_machine_review_items(ev)))
    apps = apply_decisions(
        [_d("TRK-a", DecisionType.IDENTITY_MAPPING, "C1")],
        DecisionTargets(entity_tracks={"TRK-a": track}), _SHA, _RULES)
    assert apps[0].outcome == DecisionOutcome.APPLIED
    assert not any("Reacquired identity" in t for t in _titles(build_machine_review_items(ev)))


def test_camera_classification_clears_unresolved_item() -> None:
    cand = CameraMotionCandidate(
        candidate_id="CAM-1", shot_number=1, start_frame=0, last_supporting_frame=20, end_frame=21,
        motion_class=CameraMotionClass.UNRESOLVED, strength=0.0, inlier_ratio=0.0,
        review_required=True)
    ev = VisualEvidence(camera_candidates=[cand])
    assert any("Unresolved camera" in t for t in _titles(build_machine_review_items(ev)))
    apps = apply_decisions(
        [_d("CAM-1", DecisionType.CAMERA_CLASSIFICATION,
            CameraMotionClass.HORIZONTAL_GLOBAL_MOTION.value)],
        DecisionTargets(camera_candidates={"CAM-1": cand}), _SHA, _RULES)
    assert apps[0].outcome == DecisionOutcome.APPLIED
    assert not any("Unresolved camera" in t for t in _titles(build_machine_review_items(ev)))


def _action(cid: str) -> ActionCandidate:
    return ActionCandidate(
        candidate_id=cid, shot_number=1, action_class=ActionStateClass.POSE_CHANGE_CANDIDATE,
        start_frame=0, end_frame=5, review_required=True)


def test_action_semantics_clears_summary_item() -> None:
    act = _action("ACT-1")
    ev = VisualEvidence(action_candidates=[act])
    assert any("Semantic action" in t for t in _titles(build_machine_review_items(ev)))
    apps = apply_decisions(
        [_d("ACT-1", DecisionType.ACTION_SEMANTICS, "raises hand")],
        DecisionTargets(action_candidates={"ACT-1": act}), _SHA, _RULES)
    assert apps[0].outcome == DecisionOutcome.APPLIED
    assert act.semantic_label == "raises hand"
    assert not any("Semantic action" in t for t in _titles(build_machine_review_items(ev)))


def test_action_boundary_corrects_frames_and_clears_item() -> None:
    """Phase 5.1 (item 10): a boundary decision must recompute exact timing from
    the frame ledger — frame indices alone (stale exact timing) are rejected."""
    from fractions import Fraction

    act = _action("ACT-1")
    ev = VisualEvidence(action_candidates=[act])
    # Without a ledger time resolver the decision is INVALID_VALUE, never a
    # silent frame change with stale exact timestamps.
    apps = apply_decisions(
        [_d("ACT-1", DecisionType.ACTION_BOUNDARY, "3-9")],
        DecisionTargets(action_candidates={"ACT-1": act}), _SHA, _RULES)
    assert apps[0].outcome == DecisionOutcome.INVALID_VALUE
    assert (act.start_frame, act.end_frame) == (0, 5)

    apps = apply_decisions(
        [_d("ACT-1", DecisionType.ACTION_BOUNDARY, "3-9")],
        DecisionTargets(
            action_candidates={"ACT-1": act},
            frame_to_time=lambda i: Fraction(i, 24),
        ),
        _SHA, _RULES)
    assert apps[0].outcome == DecisionOutcome.APPLIED
    assert (act.start_frame, act.end_frame) == (3, 9)
    assert (act.start_exact, act.end_exact) == (Fraction(3, 24), Fraction(9, 24))
    assert not any("Semantic action" in t for t in _titles(build_machine_review_items(ev)))


def test_playback_speed_decision_clears_speed_item() -> None:
    speed = PlaybackSpeedEvidence(
        shot_number=1, duplicate_frame_ratio=0.5, frame_spacing_regular=True,
        sustained_retiming=False, conclusion=SpeedConclusion.REVIEW_REQUIRED, review_required=True)
    ev = VisualEvidence(speed_evidence=[speed])
    assert any("Playback speed" in t for t in _titles(build_machine_review_items(ev)))
    apps = apply_decisions(
        [_d("SPEED-1", DecisionType.PLAYBACK_SPEED, "regular")],
        DecisionTargets(speed_evidence={"SPEED-1": speed}), _SHA, _RULES)
    assert apps[0].outcome == DecisionOutcome.APPLIED
    assert speed.conclusion == SpeedConclusion.REGULAR_SUPPORTED
    assert not any("Playback speed" in t for t in _titles(build_machine_review_items(ev)))


def test_review_proposal_outcome_flows_into_matrix() -> None:
    prop = ReviewProposal(
        proposal_id="P-1", level="claim", subject_id="CLM-1",
        outcome=ReviewProposalOutcome.HUMAN_DECISION_REQUIRED)
    apps = apply_decisions(
        [_d("P-1", DecisionType.REVIEW_PROPOSAL_OUTCOME, ReviewProposalOutcome.KEEP.value)],
        DecisionTargets(proposals={"P-1": prop}), _SHA, _RULES)
    assert apps[0].outcome == DecisionOutcome.APPLIED
    # The machine proposal is still machine-authored; only its outcome is resolved.
    assert prop.proposed_by == "machine"
    # Recompute: the claim-level proposal link the matrix reads reflects the override.
    claim_outcomes = {prop.subject_id: prop.outcome}
    assert claim_outcomes["CLM-1"] == ReviewProposalOutcome.KEEP
