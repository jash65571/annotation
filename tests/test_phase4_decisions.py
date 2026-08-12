"""Phase 4 human-decision application tests (F) + review-queue frame ranges (AA)."""

from __future__ import annotations

import json
from fractions import Fraction
from pathlib import Path

from manuscript_reviewer.models.review_intelligence import (
    ClaimReviewStatus,
    DecisionOutcome,
    DecisionType,
    EvidenceStatus,
    HumanReviewDecision,
    SeedClaimType,
)
from manuscript_reviewer.review.decisions import (
    DecisionLoadError,
    apply_decisions_to_claims,
    load_decisions,
)
from manuscript_reviewer.review.queue import build_review_queue
from manuscript_reviewer.seed.claims import extract_claims
from manuscript_reviewer.seed.comparison import build_rows, compare_seed
from manuscript_reviewer.seed.parser import parse_seed_text

_SHA = "VIDEOHASH"
_RULES = "1.3.0"


def _ost_claim():  # type: ignore[no-untyped-def]
    seed = (
        "[Shot 1: 0.0-5.0]\nCut: Opening shot\n"
        'Action & Audio: 1.0-2.0: On-screen text reads "ENEMY KILLED".\n'
    )
    doc = parse_seed_text(seed)
    claims = extract_claims(doc)
    res = compare_seed(doc, claims, None, None)  # no OCR -> ON_SCREEN_TEXT UNRESOLVED
    claim = next(c for c in res.claims if c.claim_type == SeedClaimType.ON_SCREEN_TEXT)
    return res, claim


def _decision(subject_id: str, dtype: DecisionType, value: str,
              sha: str = _SHA, rules: str = _RULES) -> HumanReviewDecision:
    return HumanReviewDecision(
        decision_id="D1",
        subject_id=subject_id,
        decision_type=dtype,
        value=value,
        decided_by="reviewer@example",
        decided_at_utc="2026-08-12T00:00:00Z",
        bound_video_sha256=sha,
        bound_rules_version=rules,
    )


def test_human_decision_makes_unresolved_claim_supported_and_clears_review_item() -> None:
    res, claim = _ost_claim()
    assert claim.evidence_status == EvidenceStatus.UNRESOLVED
    before = build_review_queue(res)
    assert any(claim.claim_id in i.related_claim_ids for i in before)

    apps = apply_decisions_to_claims(
        [_decision(claim.claim_id, DecisionType.OCR_TEXT, "ENEMY KILLED")],
        res.claims, _SHA, _RULES,
    )
    assert apps[0].outcome == DecisionOutcome.APPLIED
    assert claim.evidence_status == EvidenceStatus.SUPPORTED
    assert claim.review_status == ClaimReviewStatus.HUMAN_RESOLVED
    # A human-supported claim carries graded HUMAN_VERIFICATION evidence.
    assert any(e.is_factual for e in claim.evidence)

    # Recompute: the review item for this claim is gone.
    res.rows = build_rows(res.claims)
    after = build_review_queue(res)
    assert not any(claim.claim_id in i.related_claim_ids for i in after)


def test_decision_from_other_video_is_stale_and_unchanged() -> None:
    res, claim = _ost_claim()
    apps = apply_decisions_to_claims(
        [_decision(claim.claim_id, DecisionType.OCR_TEXT, "X", sha="OTHERVIDEO")],
        res.claims, _SHA, _RULES,
    )
    assert apps[0].outcome == DecisionOutcome.STALE
    assert claim.evidence_status == EvidenceStatus.UNRESOLVED  # unchanged


def test_conflicting_decisions_are_flagged_not_applied() -> None:
    res, claim = _ost_claim()
    d1 = _decision(claim.claim_id, DecisionType.OCR_TEXT, "ENEMY KILLED")
    d2 = HumanReviewDecision(
        decision_id="D2", subject_id=claim.claim_id, decision_type=DecisionType.OCR_TEXT,
        value="ALLY DOWN", decided_by="reviewer2", decided_at_utc="2026-08-12T00:00:00Z",
        bound_video_sha256=_SHA, bound_rules_version=_RULES,
    )
    apps = apply_decisions_to_claims([d1, d2], res.claims, _SHA, _RULES)
    assert all(a.outcome == DecisionOutcome.CONFLICT for a in apps)
    assert claim.evidence_status == EvidenceStatus.UNRESOLVED


def test_invalid_value_for_typed_decision() -> None:
    res, claim = _ost_claim()
    apps = apply_decisions_to_claims(
        [_decision(claim.claim_id, DecisionType.PLAYBACK_SPEED, "super_fast")],
        res.claims, _SHA, _RULES,
    )
    assert apps[0].outcome == DecisionOutcome.INVALID_VALUE


def test_invalid_subject_decision() -> None:
    res, _ = _ost_claim()
    apps = apply_decisions_to_claims(
        [_decision("CLM-9999", DecisionType.OCR_TEXT, "X")], res.claims, _SHA, _RULES,
    )
    assert apps[0].outcome == DecisionOutcome.INVALID_SUBJECT


def test_unbound_decision_rejected_at_load(tmp_path: Path) -> None:
    path = tmp_path / "d.json"
    payload = {"decisions": [{
        "decision_id": "D1", "subject_id": "CLM-0001", "decision_type": "OCR_TEXT",
        "value": "X", "decided_by": "reviewer@example",
        # no bound_video_sha256 / bound_rules_version -> unbound
    }]}
    path.write_text(json.dumps(payload), encoding="utf-8")
    try:
        load_decisions(path)
    except DecisionLoadError:
        return
    raise AssertionError("unbound decision must be rejected")


def test_machine_decision_rejected_at_load(tmp_path: Path) -> None:
    path = tmp_path / "d.json"
    payload = {"decisions": [{
        "decision_id": "D1", "subject_id": "CLM-0001", "decision_type": "OCR_TEXT",
        "value": "X", "decided_by": "machine",
        "bound_video_sha256": _SHA, "bound_rules_version": _RULES,
    }]}
    path.write_text(json.dumps(payload), encoding="utf-8")
    try:
        load_decisions(path)
    except DecisionLoadError:
        return
    raise AssertionError("machine-authored decision must be rejected")


def test_review_queue_frame_ranges_resolved() -> None:
    res, claim = _ost_claim()

    def frame_for_time(t: Fraction) -> int | None:
        return int(t * 10)  # 0.1 s per frame -> frame index

    items = build_review_queue(res, frame_for_time=frame_for_time)
    ost_item = next(i for i in items if claim.claim_id in i.related_claim_ids)
    assert ost_item.start_frame == 10  # 1.0 s -> frame 10
    assert ost_item.end_frame == 20  # 2.0 s -> frame 20
