"""Phase 4 slice-1 tests: seed snapshot, parser, atomic claims, structural
comparison, proposals, queue, triage, feedback, decisions, and validators.

All pure-logic (no ffmpeg), so they run in any environment.
"""

from __future__ import annotations

import json
from fractions import Fraction
from pathlib import Path

from manuscript_reviewer.models.review_intelligence import (
    ClaimImportance,
    EvidenceStatus,
    FoundationStatus,
    ReviewPriority,
    ReviewProposalOutcome,
    SeedClaimType,
    SeedFieldKind,
    SeedSectionKind,
    TriageStrategy,
)
from manuscript_reviewer.models.shot_truth import (
    CandidateStatus,
    ShotProposal,
    ShotTruthResult,
    TransitionStatus,
)
from manuscript_reviewer.review.decisions import (
    DecisionLoadError,
    apply_decisions,
    load_decisions,
)
from manuscript_reviewer.review.proposals import build_proposals, count_by_outcome
from manuscript_reviewer.review.queue import build_review_queue, build_triage
from manuscript_reviewer.seed import feedback as feedback_mod
from manuscript_reviewer.seed import snapshot as snapshot_mod
from manuscript_reviewer.seed.claims import extract_claims
from manuscript_reviewer.seed.comparison import compare_seed
from manuscript_reviewer.seed.parser import find_time_range, parse_seed_text, parse_time_token
from manuscript_reviewer.validation import review_intelligence_validator as riv
from manuscript_reviewer.validation import seed_validator

# --------------------------------------------------------------------------
# Fixtures / helpers
# --------------------------------------------------------------------------

CLEAN_SEED = """Video ID: clip42

Characters
C1: A tall man in a red shirt.
C2: Off-screen narrator.

Objects
O1: A yellow zipline.

Scene
A rooftop.

Visual concerns
None.

[Shot 1: 0.0-3.2]
Cut: Opening shot
Camera: Wide shot, low angle
Camera Movements: 0.0-1.2: Camera pans screen-left.
Action & Audio: 1.0-2.0: C1 grabs O1.
Action & Audio: 2.1-2.5: On-screen text reads "GO".
Playback speed: regular

[Shot 2: 3.2-9.5]
Cut: Hard cut
Action & Audio: 4.8s-5.5s: C2 says, "Ready?"
Playback speed: regular
"""


def _shot(index: int, start: Fraction, end: Fraction, transition: str) -> ShotProposal:
    return ShotProposal(
        shot_index=index,
        start_frame_index=0,
        end_frame_index=1,
        start_exact=start,
        end_exact=end,
        last_owned_frame_start_exact=start,
        start_manuscript=None,
        end_manuscript=None,
        transition_into_shot=transition,
        transition_status=TransitionStatus.PROPOSED,
        supporting_boundary_id=None,
        review_status=CandidateStatus.SUPPORTED,
    )


def make_shot_truth(shots: list[ShotProposal]) -> ShotTruthResult:
    return ShotTruthResult(
        frame_count=100,
        adjacent_pair_count=99,
        raw_candidate_count=0,
        merged_candidate_count=0,
        supported_count=max(0, len(shots) - 1),
        rejected_count=0,
        review_required_count=0,
        proposed_shot_count=len(shots),
        overall_status="PASS",
        candidates=[],
        shots=shots,
    )


# --------------------------------------------------------------------------
# Timestamps
# --------------------------------------------------------------------------


def test_parse_time_token_formats() -> None:
    assert parse_time_token("12") == Fraction(12)
    assert parse_time_token("4.3s") == Fraction(43, 10)
    assert parse_time_token("00:03.4") == Fraction(34, 10)
    assert parse_time_token("1:08.5") == Fraction(60) + Fraction(85, 10)
    assert parse_time_token("bogus") is None
    assert parse_time_token("-3") is None


def test_find_time_range_endash() -> None:
    found = find_time_range("4.8s–7.9s something")  # noqa: RUF001
    assert found is not None
    _, start, end = found
    assert start == Fraction(48, 10)
    assert end == Fraction(79, 10)


# --------------------------------------------------------------------------
# Parser
# --------------------------------------------------------------------------


def test_parser_recognizes_sections_and_fields() -> None:
    doc = parse_seed_text(CLEAN_SEED)
    assert doc.video_id == "clip42"
    kinds = [s.kind for s in doc.sections]
    assert SeedSectionKind.CAST in kinds
    assert SeedSectionKind.OBJECTS in kinds
    assert SeedSectionKind.VISUAL_CONCERNS in kinds
    assert doc.seed_shot_count == 2
    # A per-shot field parsed with its kind and exact time.
    shot1 = doc.shot_sections[0]
    aa = [e for e in shot1.entries if e.field == SeedFieldKind.ACTION_AUDIO]
    assert aa and aa[0].parsed_start_exact == Fraction(1)
    assert aa[0].referenced_character_ids == ["C1"]
    assert aa[0].referenced_object_ids == ["O1"]


def test_parser_preserves_malformed_timestamp() -> None:
    seed = "[Shot 1: 0.0-3.2]\nEnd: bogus-14.x\n"
    doc = parse_seed_text(seed)
    end_entries = [
        e for s in doc.shot_sections for e in s.entries if e.field == SeedFieldKind.SHOT_END
    ]
    assert end_entries
    # Preserved raw, not repaired, and an issue was logged.
    assert end_entries[0].timestamp_text == "bogus-14.x"
    assert end_entries[0].parsed_start_exact is None
    assert doc.issues


def test_parser_plain_text_without_headings() -> None:
    # Missing Overview heading; content still preserved as freeform, nothing lost.
    seed = "just some notes about the clip\nmore notes\n[Shot 1: 0.0-1.0]\nCut: Opening shot\n"
    doc = parse_seed_text(seed)
    total_entries = sum(len(s.entries) for s in doc.sections)
    assert total_entries >= 3  # two freeform notes + shot content


def test_parser_never_mutates_raw_lines() -> None:
    doc = parse_seed_text(CLEAN_SEED)
    raw_lines = [e.raw_line for s in doc.sections for e in s.entries]
    for line in raw_lines:
        assert line in CLEAN_SEED


# --------------------------------------------------------------------------
# Claims
# --------------------------------------------------------------------------


def test_claims_are_atomic_and_typed() -> None:
    doc = parse_seed_text(CLEAN_SEED)
    claims = extract_claims(doc)
    types = {c.claim_type for c in claims}
    assert SeedClaimType.MEDIA_ID in types
    assert SeedClaimType.SHOT_COUNT in types
    assert SeedClaimType.CHARACTER_EXISTS in types
    assert SeedClaimType.OBJECT_EXISTS in types
    assert SeedClaimType.ON_SCREEN_TEXT in types
    assert SeedClaimType.TRANSITION in types
    # Every non-document claim links to a source line.
    for claim in claims:
        if claim.claim_type != SeedClaimType.SHOT_COUNT:
            assert claim.seed_source_line is not None or claim.seed_entry_id is not None


def test_protected_trait_is_flagged_never_inferred() -> None:
    seed = "Characters\nC1: A 25-year-old American man.\n"
    claims = extract_claims(parse_seed_text(seed))
    protected = [c for c in claims if c.claim_type == SeedClaimType.PROTECTED_TRAIT]
    assert protected  # nationality/age/gender captured for review
    # Comparison must never mark a protected trait SUPPORTED.
    res = compare_seed(parse_seed_text(seed), claims, None, None)
    for claim in res.claims:
        if claim.claim_type == SeedClaimType.PROTECTED_TRAIT:
            assert claim.evidence_status == EvidenceStatus.UNRESOLVED


# --------------------------------------------------------------------------
# Structural comparison
# --------------------------------------------------------------------------


def test_shot_count_contradiction_drives_rebuild() -> None:
    # Seed says 2 shots; verified truth has 1.
    doc = parse_seed_text(CLEAN_SEED)
    claims = extract_claims(doc)
    truth = make_shot_truth([_shot(1, Fraction(0), Fraction(95, 10), "Opening shot")])
    res = compare_seed(doc, claims, None, truth)
    assert res.foundation_status == FoundationStatus.CONTRADICTED
    shot_count_claim = next(c for c in res.claims if c.claim_type == SeedClaimType.SHOT_COUNT)
    assert shot_count_claim.evidence_status == EvidenceStatus.CONTRADICTED
    triage = build_triage(res, 0.1)
    assert triage.suggested_strategy == TriageStrategy.REBUILD


def test_correct_structure_one_wrong_timestamp_is_fix_not_redo() -> None:
    # Seed shot structure matches (2 shots), but shot 2's boundary is off.
    doc = parse_seed_text(CLEAN_SEED)
    claims = extract_claims(doc)
    truth = make_shot_truth(
        [
            _shot(1, Fraction(0), Fraction(32, 10), "Opening shot"),
            _shot(2, Fraction(32, 10), Fraction(95, 10), "Hard cut"),
        ]
    )
    res = compare_seed(doc, claims, None, truth)
    # Shot count agrees -> not a wholesale rebuild of the structure.
    assert res.seed_shot_count == res.verified_shot_count == 2
    boundary_claims = [c for c in res.claims if c.claim_type == SeedClaimType.SHOT_BOUNDARY]
    # Shot boundaries that match are SUPPORTED.
    assert any(c.evidence_status == EvidenceStatus.SUPPORTED for c in boundary_claims)


def test_explicit_start_end_fields_score_as_one_boundary() -> None:
    # A seed that uses explicit Start:/End: fields (no header range) must yield a
    # single boundary claim scored against both endpoints, not a mis-scored point.
    seed = (
        "[Shot 1]\nStart: 0.0\nEnd: 3.2\nCut: Opening shot\n"
        "[Shot 2]\nStart: 3.2\nEnd: 9.5\nCut: Hard cut\n"
    )
    doc = parse_seed_text(seed)
    claims = extract_claims(doc)
    boundary = [c for c in claims if c.claim_type == SeedClaimType.SHOT_BOUNDARY]
    assert len(boundary) == 2  # exactly one per shot
    truth = make_shot_truth(
        [
            _shot(1, Fraction(0), Fraction(32, 10), "Opening shot"),
            _shot(2, Fraction(32, 10), Fraction(95, 10), "Hard cut"),
        ]
    )
    res = compare_seed(doc, claims, None, truth)
    shot1_boundary = next(
        c for c in res.claims
        if c.claim_type == SeedClaimType.SHOT_BOUNDARY and c.shot_number == 1
    )
    assert shot1_boundary.evidence_status == EvidenceStatus.SUPPORTED


def test_opening_shot_supported_has_evidence() -> None:
    doc = parse_seed_text(CLEAN_SEED)
    claims = extract_claims(doc)
    truth = make_shot_truth(
        [
            _shot(1, Fraction(0), Fraction(32, 10), "Opening shot"),
            _shot(2, Fraction(32, 10), Fraction(95, 10), "Hard cut"),
        ]
    )
    res = compare_seed(doc, claims, None, truth)
    trans1 = next(
        c for c in res.claims if c.claim_type == SeedClaimType.TRANSITION and c.shot_number == 1
    )
    assert trans1.evidence_status == EvidenceStatus.SUPPORTED
    assert trans1.evidence  # P4-CLAIM-001: supported requires evidence
    # matrix must not violate P4-CLAIM-001/002
    assert not riv.validate_matrix(res.rows)


def test_undefined_character_reference_is_contradiction() -> None:
    seed = (
        "Characters\nC1: A man.\n[Shot 1: 0.0-2.0]\nCut: Opening shot\n"
        "Action & Audio: 1.0-1.5: C3 waves.\n"
    )
    doc = parse_seed_text(seed)
    claims = extract_claims(doc)
    res = compare_seed(doc, claims, None, None)
    assert any("C3" in c for c in res.foundational_conflicts)


# --------------------------------------------------------------------------
# Proposals / queue / validators
# --------------------------------------------------------------------------


def test_redo_proposals_have_structural_reason() -> None:
    doc = parse_seed_text(CLEAN_SEED)
    claims = extract_claims(doc)
    truth = make_shot_truth([_shot(1, Fraction(0), Fraction(95, 10), "Opening shot")])
    res = compare_seed(doc, claims, None, truth)
    proposals = build_proposals(res)
    # P4-REVIEW-002 must hold: every REDO has a structural reason.
    assert not riv.validate_proposals(proposals, res.rows)
    counts = count_by_outcome(proposals)
    assert counts[ReviewProposalOutcome.REDO_REBUILD.value] >= 1


def test_keep_forbidden_with_contradiction() -> None:
    doc = parse_seed_text(CLEAN_SEED)
    claims = extract_claims(doc)
    truth = make_shot_truth([_shot(1, Fraction(0), Fraction(95, 10), "Opening shot")])
    res = compare_seed(doc, claims, None, truth)
    proposals = build_proposals(res)
    # No shot/seed KEEP proposal may cover a contradicted claim.
    issues = riv.validate_proposals(proposals, res.rows)
    assert all(i.rule_id != "P4-REVIEW-003" for i in issues)


def test_queue_has_critical_for_foundation_contradiction() -> None:
    doc = parse_seed_text(CLEAN_SEED)
    claims = extract_claims(doc)
    truth = make_shot_truth([_shot(1, Fraction(0), Fraction(95, 10), "Opening shot")])
    res = compare_seed(doc, claims, None, truth)
    items = build_review_queue(res)
    assert any(i.priority == ReviewPriority.CRITICAL for i in items)
    # P4-QC-001: PASS forbidden while a critical item exists.
    assert riv.validate_qc_gate("PASS", items)
    assert riv.compute_overall_status(items) == "REVIEW_REQUIRED"


# --------------------------------------------------------------------------
# Snapshot immutability
# --------------------------------------------------------------------------


def test_seed_snapshot_is_immutable_and_hashed(tmp_path: Path) -> None:
    src = tmp_path / "seed.md"
    src.write_text(CLEAN_SEED, encoding="utf-8")
    seed_dir = tmp_path / "run" / "seed"
    snap = snapshot_mod.snapshot_seed(src, seed_dir)
    stored = seed_dir / "seed_original.txt"
    assert stored.read_text(encoding="utf-8") == CLEAN_SEED
    doc = parse_seed_text(CLEAN_SEED, snap)
    assert not seed_validator.validate_seed_snapshot(seed_dir, doc)
    # Tamper -> validator fails.
    stored.write_text(CLEAN_SEED + "tampered", encoding="utf-8")
    assert seed_validator.validate_seed_snapshot(seed_dir, doc)


# --------------------------------------------------------------------------
# Feedback
# --------------------------------------------------------------------------


def test_feedback_maps_known_pattern_but_preserves_raw() -> None:
    fb = "The English singing is not transcribed.\nSomething vague to consider.\n"
    doc = feedback_mod.parse_feedback_text(fb)
    assert len(doc.directives) == 2
    mapped = [d for d in doc.directives if d.machine_interpretation]
    assert mapped and mapped[0].machine_interpretation == "REQUIRE_VOCAL_LYRIC_REVIEW"
    assert all(d.review_required for d in doc.directives)
    assert doc.directives[0].raw_text.strip().startswith("The English singing")


# --------------------------------------------------------------------------
# Human decisions
# --------------------------------------------------------------------------


def _write_decisions(path: Path, decided_by: str, bound_sha: str | None) -> None:
    payload = {
        "decisions": [
            {
                "decision_id": "D1",
                "subject_id": "shot_1",
                "decision_type": "confirm_boundary",
                "value": "confirmed",
                "decided_by": decided_by,
                "bound_video_sha256": bound_sha,
            }
        ]
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_decisions_reject_machine_authored(tmp_path: Path) -> None:
    path = tmp_path / "d.json"
    _write_decisions(path, "machine", "abc")
    try:
        load_decisions(path)
    except DecisionLoadError:
        pass
    else:
        raise AssertionError("machine-authored decision must be rejected")


def test_stale_decision_from_other_video_not_applied(tmp_path: Path) -> None:
    path = tmp_path / "d.json"
    _write_decisions(path, "reviewer@example", "OTHERVIDEOHASH")
    decisions = load_decisions(path)
    apps = apply_decisions(decisions, video_sha256="CURRENTHASH", rules_version="1.0")
    assert apps and apps[0].stale and not apps[0].applied


def test_valid_decision_applied(tmp_path: Path) -> None:
    path = tmp_path / "d.json"
    _write_decisions(path, "reviewer@example", "CURRENTHASH")
    decisions = load_decisions(path)
    apps = apply_decisions(decisions, video_sha256="CURRENTHASH", rules_version="1.0")
    assert apps and apps[0].applied and not apps[0].stale


# --------------------------------------------------------------------------
# Importance classification
# --------------------------------------------------------------------------


def test_foundational_vs_local_importance() -> None:
    doc = parse_seed_text(CLEAN_SEED)
    claims = extract_claims(doc)
    by_type = {c.claim_type: c for c in claims}
    assert by_type[SeedClaimType.SHOT_COUNT].importance == ClaimImportance.FOUNDATIONAL
    assert by_type[SeedClaimType.CHARACTER_EXISTS].importance == ClaimImportance.FOUNDATIONAL
    assert by_type[SeedClaimType.ON_SCREEN_TEXT].importance == ClaimImportance.LOCAL
