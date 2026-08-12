"""Phase 5.2 pre-lock micro-hardening regressions: enrichment evidence gate,
feedback-resolution gate, timed human-fact shot containment, decision shot
containment, manifest-exists semantics, and speech decision provenance."""

from __future__ import annotations

import json
from fractions import Fraction
from pathlib import Path

import pytest

from manuscript_reviewer.caption_brain import CaptionBrainError
from manuscript_reviewer.models.audio import SourceVerificationStatus
from manuscript_reviewer.models.caption_brain import CaptionReadiness
from manuscript_reviewer.models.review_intelligence import (
    ActionCandidate,
    ActionStateClass,
    DecisionOutcome,
    DecisionType,
    SeedClaimType,
    TextTrack,
)
from manuscript_reviewer.review.decisions import DecisionTargets, apply_decisions

from .phase5_helpers import (
    RULES_VERSION,
    VIDEO_SHA,
    finalize,
    human_decision,
    make_audio_truth,
    make_shot,
    make_shot_truth,
    make_speech_region,
    supported_claim,
    write_json,
    write_run_dir,
)

_EV = [
    {
        "evidence_id": "EV-HF",
        "evidence_type": "FRAME_RANGE",
        "start_frame": 0,
        "end_frame": 24,
        "source": "reviewer@test",
    }
]


def _speech_run(tmp_path: Path, verified: bool = True) -> Path:
    shots = make_shot_truth([make_shot(1, Fraction(0), Fraction(2), "Opening shot")])
    status = (
        SourceVerificationStatus.HUMAN_VERIFIED
        if verified
        else SourceVerificationStatus.UNVERIFIED
    )
    audio = make_audio_truth(
        [make_speech_region("SR-1", Fraction(0), Fraction(1), "hello world", status=status)]
    )
    claims = [
        supported_claim("CLM-C1", SeedClaimType.CHARACTER_EXISTS,
                        "A person in a coat.", subject_ids=["C1"]),
    ]
    return write_run_dir(tmp_path, shots, audio_truth=audio, seed_claims=claims)


def _enrichment_fact(with_evidence: bool, extra: dict[str, str] | None = None) -> dict[str, object]:
    fact: dict[str, object] = {
        "fact_id": "HF-SPK",
        "fact_type": "SPEECH",
        "semantic_value": {"region_id": "SR-1", "speaker_id": "C1", **(extra or {})},
        "character_ids": ["C1"],
        "decided_by": "reviewer@test",
        "bound_video_sha256": VIDEO_SHA,
        "bound_rules_version": RULES_VERSION,
    }
    if with_evidence:
        fact["evidence_refs"] = _EV
    return fact


# --- item 1: enrichment must pass the evidence gate ------------------------


def test_evidence_free_enrichment_never_finalizes_speech(tmp_path: Path) -> None:
    run_dir = _speech_run(tmp_path, verified=True)
    facts = write_json(tmp_path / "hf.json", {"facts": [_enrichment_fact(False)]})
    output = finalize(run_dir, human_facts_path=facts)
    # The enrichment was rejected: no speaker reaches the fact, so the verified
    # region still cannot become final dialogue through it.
    assert output.result.speech_verified_count == 0
    assert output.result.speech_blocked_count == 1
    draft = (run_dir / "caption" / "draft_review_only.md").read_text(encoding="utf-8")
    assert "hello world" not in draft
    caption_facts = json.loads(
        (run_dir / "caption" / "caption_facts.json").read_text(encoding="utf-8")
    )
    speech = next(
        f for f in caption_facts["facts"] if f["fact_type"] == "SPEECH"
    )
    assert "speaker" in speech["eligibility_reason"]
    assert any("enrichment HF-SPK rejected" in n for n in speech["notes"])


def test_evidence_backed_enrichment_still_works(tmp_path: Path) -> None:
    run_dir = _speech_run(tmp_path, verified=True)
    facts = write_json(tmp_path / "hf.json", {"facts": [_enrichment_fact(True)]})
    output = finalize(run_dir, human_facts_path=facts)
    assert output.result.speech_verified_count == 1


# --- item 2: HIGH feedback needs a VALIDATED human fact --------------------


def _feedback_run(tmp_path: Path) -> Path:
    run_dir = _speech_run(tmp_path, verified=True)
    write_json(
        run_dir / "feedback" / "feedback_directives.json",
        {"directives": [{
            "directive_id": "FBK-0001",
            "raw_text": "CRITICAL: music lyrics are missing",
            "source_line": 1,
            "priority": "HIGH",
            "machine_interpretation": "REQUIRE_VOCAL_LYRIC_REVIEW",
            "interpretation_status": "MAPPED",
            "review_required": True,
        }]},
    )
    return run_dir


def test_evidence_free_fact_cannot_resolve_high_feedback(tmp_path: Path) -> None:
    run_dir = _feedback_run(tmp_path)
    facts = write_json(
        tmp_path / "hf.json",
        {"facts": [{
            "fact_id": "HF-LYR",
            "fact_type": "SOUND",
            "text_value": "Soft singing continues in the background.",
            "shot_number": 1,
            "start_exact": "0",
            "end_exact": "1",
            "semantic_value": {"resolves_directive": "FBK-0001"},
            "decided_by": "reviewer@test",
            "bound_video_sha256": VIDEO_SHA,
            "bound_rules_version": RULES_VERSION,
        }]},
    )
    output = finalize(run_dir, human_facts_path=facts)
    assert output.result.unresolved_feedback_high == 1
    assert any("FBK-0001" in b for b in output.result.blockers)


def test_validated_fact_resolves_high_feedback(tmp_path: Path) -> None:
    run_dir = _feedback_run(tmp_path)
    facts = write_json(
        tmp_path / "hf.json",
        {"facts": [{
            "fact_id": "HF-LYR",
            "fact_type": "SOUND",
            "text_value": "Soft singing continues in the background.",
            "shot_number": 1,
            "start_exact": "0",
            "end_exact": "1",
            "evidence_refs": _EV,
            "semantic_value": {"resolves_directive": "FBK-0001"},
            "decided_by": "reviewer@test",
            "bound_video_sha256": VIDEO_SHA,
            "bound_rules_version": RULES_VERSION,
        }]},
    )
    output = finalize(run_dir, human_facts_path=facts)
    assert output.result.unresolved_feedback_high == 0


# --- item 3: timed human facts validated against Shot Truth ----------------


def _timed_fact(fact_type: str, start: str, end: str) -> dict[str, object]:
    return {
        "fact_id": "HF-T",
        "fact_type": fact_type,
        "text_value": "The camera view moves screen-left."
        if fact_type == "CAMERA_MOVEMENT"
        else "Playback slows briefly.",
        "shot_number": 1,
        "start_exact": start,
        "end_exact": end,
        "evidence_refs": _EV,
        "decided_by": "reviewer@test",
        "bound_video_sha256": VIDEO_SHA,
        "bound_rules_version": RULES_VERSION,
    }


@pytest.mark.parametrize(
    "fact_type,start,end,reason_fragment",
    [
        ("CAMERA_MOVEMENT", "3", "4", "outside verified shot"),
        ("SPEED_CHANGE", "5/2", "3", "outside verified shot"),
        ("CAMERA_MOVEMENT", "3/2", "1", "start > end"),
    ],
)
def test_timed_human_fact_shot_containment(
    tmp_path: Path, fact_type: str, start: str, end: str, reason_fragment: str
) -> None:
    shots = make_shot_truth([make_shot(1, Fraction(0), Fraction(2), "Opening shot")])
    run_dir = write_run_dir(tmp_path, shots)
    facts = write_json(
        tmp_path / "hf.json", {"facts": [_timed_fact(fact_type, start, end)]}
    )
    output = finalize(run_dir, human_facts_path=facts)
    assert output.result.readiness != CaptionReadiness.READY_FOR_FINAL_REVIEW
    assert any(reason_fragment in b for b in output.result.blockers)
    draft = (run_dir / "caption" / "draft_review_only.md").read_text(encoding="utf-8")
    assert "screen-left" not in draft and "slows briefly" not in draft


# --- item 4: decision boundaries stay inside their shot --------------------


def test_action_boundary_outside_shot_is_invalid(tmp_path: Path) -> None:
    candidate = ActionCandidate(
        candidate_id="AC-1", shot_number=1,
        action_class=ActionStateClass.CONTACT_BEGINS, start_frame=0, end_frame=5,
    )
    apps = apply_decisions(
        [human_decision("D-AB", "AC-1", DecisionType.ACTION_BOUNDARY, "40-60")],
        DecisionTargets(
            action_candidates={"AC-1": candidate},
            frame_to_time=lambda i: Fraction(i, 24) if i < 96 else None,
            shot_frame_ranges={1: (0, 23)},
        ),
        VIDEO_SHA, RULES_VERSION,
    )
    assert apps[0].outcome == DecisionOutcome.INVALID_VALUE
    assert "outside shot 1" in (apps[0].reason or "")
    assert (candidate.start_frame, candidate.end_frame) == (0, 5)


def test_text_timing_cannot_span_shots(tmp_path: Path) -> None:
    track = TextTrack(track_id="TT-1", first_candidate_frame=0)
    apps = apply_decisions(
        [human_decision("D-TT", "TT-1", DecisionType.TEXT_TIMING, "10-30")],
        DecisionTargets(
            text_tracks={"TT-1": track},
            frame_to_time=lambda i: Fraction(i, 24) if i < 96 else None,
            shot_frame_ranges={1: (0, 23), 2: (24, 47)},
        ),
        VIDEO_SHA, RULES_VERSION,
    )
    assert apps[0].outcome == DecisionOutcome.INVALID_VALUE
    assert "spans beyond shot 1" in (apps[0].reason or "")
    assert track.first_stable_frame is None


# --- item 5: manifest exists with empty artifacts = strict mode ------------


def test_manifest_with_empty_artifacts_rejects_evidence(tmp_path: Path) -> None:
    shots = make_shot_truth([make_shot(1, Fraction(0), Fraction(2), "Opening shot")])
    run_dir = write_run_dir(tmp_path, shots)
    (run_dir / "manifest.json").write_text(
        json.dumps(
            {
                "source_video_sha256": VIDEO_SHA,
                "source_video_path": "C:/videos/x.mp4",
                "rules_version": RULES_VERSION,
                "artifacts": [],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(CaptionBrainError, match="not listed in manifest"):
        finalize(run_dir)


# --- item 6: speech decision provenance on the fact ------------------------


def test_speech_decision_provenance_on_fact(tmp_path: Path) -> None:
    run_dir = _speech_run(tmp_path, verified=False)
    decisions = write_json(
        tmp_path / "decisions.json",
        {"decisions": [{
            "decision_id": "D-SV",
            "subject_id": "SR-1",
            "decision_type": "SPEECH_VERIFICATION",
            "value": "verified",
            "decided_by": "reviewer@test",
            "decided_at_utc": "2026-08-12T00:00:00Z",
            "bound_video_sha256": VIDEO_SHA,
            "bound_rules_version": RULES_VERSION,
        }]},
    )
    facts = write_json(tmp_path / "hf.json", {"facts": [_enrichment_fact(True)]})
    finalize(run_dir, review_decisions_path=decisions, human_facts_path=facts)
    caption_facts = json.loads(
        (run_dir / "caption" / "caption_facts.json").read_text(encoding="utf-8")
    )
    speech = next(f for f in caption_facts["facts"] if f["fact_type"] == "SPEECH")
    # Traceable to the applied decision, not only a mutated enum.
    assert speech["human_decision_ids"] == ["D-SV"]
    assert any(
        ref["evidence_type"] == "HUMAN_VERIFICATION"
        and ref["evidence_id"] == "EV-HUMAN-D-SV"
        for ref in speech["evidence_refs"]
    )
