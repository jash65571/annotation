"""Phase 5 readiness / signoff / ready-file safety / finalize tests
(§103/§118/§119/§94) plus golden-gate behavior fixtures (§115) and the static
safety sweep (§127)."""

from __future__ import annotations

import json
import re
from fractions import Fraction
from pathlib import Path

from manuscript_reviewer.caption_brain import finalize_run
from manuscript_reviewer.models.caption_brain import (
    CaptionReadiness,
    FinalReviewSignoff,
)
from manuscript_reviewer.models.review_intelligence import SeedClaimType
from manuscript_reviewer.models.shot_truth import TransitionStatus
from manuscript_reviewer.validation.final_caption_validator import check_signoff

from .phase5_helpers import (
    RULES_VERSION,
    VIDEO_SHA,
    make_shot,
    make_shot_truth,
    supported_claim,
    write_json,
    write_run_dir,
)

ENGINE_DIR = Path(__file__).parent.parent / "engine" / "manuscript_reviewer"

#: Every media-factual human fact must carry traceable evidence (§5.1-11).
_EV = [
    {
        "evidence_id": "EV-HF",
        "evidence_type": "FRAME_RANGE",
        "start_frame": 0,
        "end_frame": 24,
        "source": "reviewer@test",
    }
]


def _ready_run_dir(tmp_path: Path) -> Path:
    """A run whose caption reaches READY_FOR_FINAL_REVIEW: one supported shot,
    verified speed, referenced character, eligible scene."""
    shots = make_shot_truth([make_shot(1, Fraction(0), Fraction(2), "Opening shot")])
    claims = [
        supported_claim(
            "CLM-C1", SeedClaimType.CHARACTER_EXISTS,
            "A person in a dark green jacket.", subject_ids=["C1"],
        ),
        supported_claim(
            "CLM-SC", SeedClaimType.SCENE_STATE,
            "A flat dirt path beside a canal.", source_field="SCENE",
        ),
    ]
    run_dir = write_run_dir(tmp_path, shots, seed_claims=claims)
    # Machine speed evidence exists; a human decision verifies it.
    write_json(
        run_dir / "visual" / "speed" / "playback_speed_evidence.json",
        {
            "playback_speed_evidence": [
                {
                    "shot_number": 1,
                    "conclusion": "REGULAR_CANDIDATE",
                    "review_required": True,
                }
            ]
        },
    )
    return run_dir


def _decisions_file(tmp_path: Path) -> Path:
    return write_json(
        tmp_path / "decisions.json",
        {
            "decisions": [
                {
                    "decision_id": "D-SPEED",
                    "subject_id": "SPEED-1",
                    "decision_type": "PLAYBACK_SPEED",
                    "value": "regular",
                    "decided_by": "reviewer@test",
                    "decided_at_utc": "2026-08-12T00:00:00Z",
                    "bound_video_sha256": VIDEO_SHA,
                    "bound_rules_version": RULES_VERSION,
                }
            ]
        },
    )


def _human_facts_file(tmp_path: Path) -> Path:
    return write_json(
        tmp_path / "human_facts.json",
        {
            "facts": [
                {
                    "fact_id": "HF-ACT",
                    "fact_type": "VISUAL_ACTION",
                    "text_value": "C1 stands beside the canal.",
                    "shot_number": 1,
                    "character_ids": ["C1"],
                    "start_exact": "0",
                    "end_exact": "1",
                    "evidence_refs": _EV,
                    "decided_by": "reviewer@test",
                    "bound_video_sha256": VIDEO_SHA,
                    "bound_rules_version": RULES_VERSION,
                }
            ]
        },
    )


def test_no_signoff_stops_at_ready_for_final_review(tmp_path: Path) -> None:
    run_dir = _ready_run_dir(tmp_path)
    output = finalize_run(
        run_dir,
        review_decisions_path=_decisions_file(tmp_path),
        human_facts_path=_human_facts_file(tmp_path),
    )
    assert output.result.readiness == CaptionReadiness.READY_FOR_FINAL_REVIEW
    # §119: no ready file without READY_TO_ENTER; the draft name is honest.
    assert not (run_dir / "caption" / "ready_to_enter.md").exists()
    assert (run_dir / "caption" / "draft_review_only.md").exists()


def test_valid_signoff_reaches_ready_to_enter(tmp_path: Path) -> None:
    run_dir = _ready_run_dir(tmp_path)
    decisions = _decisions_file(tmp_path)
    facts = _human_facts_file(tmp_path)
    first = finalize_run(run_dir, review_decisions_path=decisions, human_facts_path=facts)
    assert first.result.readiness == CaptionReadiness.READY_FOR_FINAL_REVIEW
    manifest = json.loads(
        (run_dir / "caption" / "caption_manifest.json").read_text(encoding="utf-8")
    )
    caption_sha = manifest["rendered_caption_sha256"]
    signoff = write_json(
        tmp_path / "final_review.json",
        {
            "video_sha256": VIDEO_SHA,
            "rules_version": RULES_VERSION,
            "caption_sha256": caption_sha,
            "reviewer": "reviewer@test",
            "reviewed_at_utc": "2026-08-12T01:00:00Z",
            "golden_example_comparison_complete": True,
            "platform_semantic_pass_complete": True,
            "final_adversarial_read_complete": True,
            "no_known_omissions_confirmed": True,
            "no_known_hallucinations_confirmed": True,
        },
    )
    second = finalize_run(
        run_dir,
        review_decisions_path=decisions,
        human_facts_path=facts,
        final_review_path=signoff,
    )
    assert second.result.readiness == CaptionReadiness.READY_TO_ENTER
    assert (run_dir / "caption" / "ready_to_enter.md").exists()
    assert (run_dir / "caption" / "ready_to_enter.json").exists()
    assert not (run_dir / "caption" / "draft_review_only.md").exists()
    # Fast re-finalization (§94/§122): no media analysis, sub-second.
    total = second.result.stage_timings_seconds["caption_brain_total"]
    assert total < 5.0


def test_signoff_wrong_video_is_stale() -> None:
    signoff = FinalReviewSignoff(
        video_sha256="f" * 64,
        rules_version=RULES_VERSION,
        caption_sha256="c" * 64,
        reviewer="reviewer@test",
        golden_example_comparison_complete=True,
        platform_semantic_pass_complete=True,
        final_adversarial_read_complete=True,
        no_known_omissions_confirmed=True,
        no_known_hallucinations_confirmed=True,
    )
    check = check_signoff(signoff, VIDEO_SHA, RULES_VERSION, "c" * 64)
    assert check.stale and not check.valid


def test_signoff_wrong_caption_hash_is_stale() -> None:
    signoff = FinalReviewSignoff(
        video_sha256=VIDEO_SHA,
        rules_version=RULES_VERSION,
        caption_sha256="old" + "0" * 61,
        reviewer="reviewer@test",
        golden_example_comparison_complete=True,
        platform_semantic_pass_complete=True,
        final_adversarial_read_complete=True,
        no_known_omissions_confirmed=True,
        no_known_hallucinations_confirmed=True,
    )
    check = check_signoff(signoff, VIDEO_SHA, RULES_VERSION, "new" + "1" * 61)
    assert check.stale
    assert any("changed after final review" in r for r in check.reasons)


def test_caption_change_invalidates_old_signoff(tmp_path: Path) -> None:
    run_dir = _ready_run_dir(tmp_path)
    decisions = _decisions_file(tmp_path)
    facts = _human_facts_file(tmp_path)
    first = finalize_run(run_dir, review_decisions_path=decisions, human_facts_path=facts)
    manifest = json.loads(
        (run_dir / "caption" / "caption_manifest.json").read_text(encoding="utf-8")
    )
    signoff = write_json(
        tmp_path / "final_review.json",
        {
            "video_sha256": VIDEO_SHA,
            "rules_version": RULES_VERSION,
            "caption_sha256": manifest["rendered_caption_sha256"],
            "reviewer": "reviewer@test",
            "golden_example_comparison_complete": True,
            "platform_semantic_pass_complete": True,
            "final_adversarial_read_complete": True,
            "no_known_omissions_confirmed": True,
            "no_known_hallucinations_confirmed": True,
        },
    )
    # A new human fact changes the caption content → old signoff is stale.
    more_facts = write_json(
        tmp_path / "human_facts2.json",
        json.loads(facts.read_text(encoding="utf-8"))
        | {
            "facts": json.loads(facts.read_text(encoding="utf-8"))["facts"]
            + [
                {
                    "fact_id": "HF-ACT2",
                    "fact_type": "VISUAL_ACTION",
                    "text_value": "C1 crouches beside the canal.",
                    "shot_number": 1,
                    "character_ids": ["C1"],
                    "start_exact": "1",
                    "end_exact": "2",
                    "evidence_refs": _EV,
                    "decided_by": "reviewer@test",
                    "bound_video_sha256": VIDEO_SHA,
                    "bound_rules_version": RULES_VERSION,
                }
            ]
        },
    )
    changed = finalize_run(
        run_dir,
        review_decisions_path=decisions,
        human_facts_path=more_facts,
        final_review_path=signoff,
    )
    assert changed.result.readiness == CaptionReadiness.READY_FOR_FINAL_REVIEW
    assert changed.result.signoff_stale
    assert not (run_dir / "caption" / "ready_to_enter.md").exists()
    assert first.result.readiness == CaptionReadiness.READY_FOR_FINAL_REVIEW


def test_unresolved_transition_blocks_ready(tmp_path: Path) -> None:
    """§103: seeded 5 shots but only 2 verified → 2 final shots; unresolved
    shot-2 transition → not ready."""
    shots = make_shot_truth(
        [
            make_shot(1, Fraction(0), Fraction(1), "Opening shot"),
            make_shot(2, Fraction(1), Fraction(2), None, TransitionStatus.UNRESOLVED),
        ]
    )
    run_dir = write_run_dir(tmp_path, shots)
    output = finalize_run(run_dir)
    assert output.result.readiness in (
        CaptionReadiness.REVIEW_REQUIRED,
        CaptionReadiness.BLOCKED,
    )
    assert output.plan is not None
    assert len(output.plan.shot_plans) == 2
    assert any("transition" in b for b in output.result.blockers)
    draft = (run_dir / "caption" / "draft_review_only.md").read_text(encoding="utf-8")
    assert "Hard cut" not in draft


def test_stale_human_fact_from_other_video_rejected(tmp_path: Path) -> None:
    shots = make_shot_truth([make_shot(1, Fraction(0), Fraction(2), "Opening shot")])
    run_dir = write_run_dir(tmp_path, shots)
    stale_facts = write_json(
        tmp_path / "stale.json",
        {
            "facts": [
                {
                    "fact_id": "HF-STALE",
                    "fact_type": "VISUAL_ACTION",
                    "text_value": "C1 waves.",
                    "shot_number": 1,
                    "character_ids": ["C1"],
                    "decided_by": "reviewer@test",
                    "bound_video_sha256": "9" * 64,
                    "bound_rules_version": RULES_VERSION,
                }
            ]
        },
    )
    output = finalize_run(run_dir, human_facts_path=stale_facts)
    record = json.loads(
        (run_dir / "caption" / "human_facts_applied.json").read_text(encoding="utf-8")
    )
    assert record["accepted"] == []
    assert record["rejected"][0]["fact_id"] == "HF-STALE"
    assert "C1 waves." not in (run_dir / "caption" / "draft_review_only.md").read_text(
        encoding="utf-8"
    )
    assert output.result.readiness != CaptionReadiness.READY_TO_ENTER


def test_golden_gate_keeps_short_events_and_overlap(tmp_path: Path) -> None:
    """§115: a 0.1 s event is retained; overlapping events keep separate
    entries; camera stays separate."""
    run_dir = _ready_run_dir(tmp_path)
    facts = write_json(
        tmp_path / "hf.json",
        {
            "facts": [
                {
                    "fact_id": "HF-SHORT",
                    "fact_type": "VISUAL_ACTION",
                    "text_value": "C1 releases the grip with the right hand.",
                    "shot_number": 1,
                    "character_ids": ["C1"],
                    "start_exact": "73/10",
                    "end_exact": "74/10",
                    "evidence_refs": _EV,
                    "decided_by": "reviewer@test",
                    "bound_video_sha256": VIDEO_SHA,
                    "bound_rules_version": RULES_VERSION,
                },
                {
                    "fact_id": "HF-OVER",
                    "fact_type": "VISUAL_ACTION",
                    "text_value": "C1 leans toward the container.",
                    "shot_number": 1,
                    "character_ids": ["C1"],
                    "start_exact": "7",
                    "end_exact": "8",
                    "evidence_refs": _EV,
                    "decided_by": "reviewer@test",
                    "bound_video_sha256": VIDEO_SHA,
                    "bound_rules_version": RULES_VERSION,
                },
            ]
        },
    )
    # Widen the single shot so the 7.3-7.4 event is inside it.
    shots = make_shot_truth([make_shot(1, Fraction(0), Fraction(10), "Opening shot")])
    (run_dir / "shot_qc.json").write_text(shots.model_dump_json(), encoding="utf-8")
    output = finalize_run(
        run_dir,
        review_decisions_path=_decisions_file(tmp_path),
        human_facts_path=facts,
    )
    golden = json.loads(
        (run_dir / "caption" / "golden_gate.json").read_text(encoding="utf-8")
    )
    by_category = {c["category"]: c["status"] for c in golden["categories"]}
    assert by_category["EVENT_GRANULARITY"] == "PASS"
    assert by_category["TRUTHFUL_OVERLAP"] == "PASS"
    assert by_category["CAMERA_SEPARATION"] == "PASS"
    draft = (run_dir / "caption" / "draft_review_only.md").read_text(encoding="utf-8")
    assert "[7.3s-7.4s]" in draft
    assert output.result.readiness == CaptionReadiness.READY_FOR_FINAL_REVIEW


# --- §127 static safety sweep ---------------------------------------------


def _phase5_sources() -> list[Path]:
    files = [ENGINE_DIR / "caption_brain.py"]
    files += sorted((ENGINE_DIR / "caption").glob("*.py"))
    files += [
        ENGINE_DIR / "validation" / "caption_validator.py",
        ENGINE_DIR / "validation" / "platform_semantic_validator.py",
        ENGINE_DIR / "validation" / "golden_validator.py",
        ENGINE_DIR / "validation" / "final_caption_validator.py",
        ENGINE_DIR / "artifacts" / "caption_writer.py",
        ENGINE_DIR / "models" / "caption_brain.py",
    ]
    return files


def test_no_cloud_or_submission_code_in_caption_brain() -> None:
    """§52/§98: no media upload, no platform submission, no result-code
    generation anywhere in Phase 5 code."""
    forbidden = re.compile(
        r"requests\.post|httpx|import openai|import anthropic|import google|"
        r"\bdescript\b|\bupload\s*\(|\.upload\b|result_code|submit_task|"
        r"shell=True|subprocess\.run",
        re.IGNORECASE,
    )
    for path in _phase5_sources():
        text = path.read_text(encoding="utf-8")
        match = forbidden.search(text)
        assert match is None, f"{path.name}: forbidden pattern {match.group(0)!r}"


def test_no_local_rounding_logic_in_caption_brain() -> None:
    """§48: no ad-hoc float rounding; all display timing goes through
    to_manuscript_display / format_manuscript_display."""
    pattern = re.compile(r"round\(\s*[a-z_]*\s*\*\s*10")
    for path in _phase5_sources():
        assert not pattern.search(path.read_text(encoding="utf-8")), path.name


def test_machine_code_never_creates_human_records() -> None:
    """HumanCaptionFact / FinalReviewSignoff are only ever instantiated from
    human-supplied files (model_validate), never constructed by engine code."""
    constructor = re.compile(r"(?<!class )(HumanCaptionFact|FinalReviewSignoff)\(")
    for path in _phase5_sources():
        text = path.read_text(encoding="utf-8")
        for match in constructor.finditer(text):
            # Direct keyword construction would be `HumanCaptionFact(fact_id=...`
            # or `FinalReviewSignoff(video_sha256=...`; loading human files goes
            # through model_validate instead.
            following = text[match.end() : match.end() + 40].strip()
            assert not following.startswith(("fact_id=", "video_sha256=")), (
                f"{path.name}: machine-side construction of {match.group(1)}"
            )
