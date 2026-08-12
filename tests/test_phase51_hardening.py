"""Phase 5.1 hardening regressions: canonical media id, evidence tampering,
Phase 4 review carry-forward, speech/OCR/transition/identity/action-boundary
human flows, human-fact evidence rules, blocked-material Golden gates, and
reviewed_caption parity."""

from __future__ import annotations

import hashlib
import json
from fractions import Fraction
from pathlib import Path

import pytest

from manuscript_reviewer.caption_brain import CaptionBrainError
from manuscript_reviewer.models.caption_brain import CaptionReadiness
from manuscript_reviewer.models.review_intelligence import SeedClaimType
from manuscript_reviewer.models.shot_truth import TransitionStatus

from .phase5_helpers import (
    RULES_VERSION,
    VIDEO_ID,
    VIDEO_SHA,
    finalize,
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


def _decision(
    decision_id: str, subject: str, dtype: str, value: str
) -> dict[str, str]:
    return {
        "decision_id": decision_id,
        "subject_id": subject,
        "decision_type": dtype,
        "value": value,
        "decided_by": "reviewer@test",
        "decided_at_utc": "2026-08-12T00:00:00Z",
        "bound_video_sha256": VIDEO_SHA,
        "bound_rules_version": RULES_VERSION,
    }


def _decisions_file(tmp_path: Path, decisions: list[dict[str, str]]) -> Path:
    return write_json(tmp_path / "decisions.json", {"decisions": decisions})


def _one_shot_run(tmp_path: Path, **kwargs: object) -> Path:
    shots = make_shot_truth([make_shot(1, Fraction(0), Fraction(2), "Opening shot")])
    return write_run_dir(tmp_path, shots, **kwargs)  # type: ignore[arg-type]


# --- canonical media id (§5.1-2/16) ---------------------------------------


def test_wrong_seed_video_id_fails_m2_media(tmp_path: Path) -> None:
    """Actual video ABC, seed says XYZ → the caption begins with ABC and
    M2-MEDIA-004 FAILS. The seed can never validate against itself."""
    run_dir = _one_shot_run(tmp_path, video_id="XYZ_totally_wrong")
    output = finalize(run_dir)
    draft = (run_dir / "caption" / "draft_review_only.md").read_text(encoding="utf-8")
    assert draft.splitlines()[0] == VIDEO_ID  # canonical id from media truth
    assert "XYZ_totally_wrong" not in draft
    assert any("M2-MEDIA-004" in b for b in output.result.blockers)
    assert output.result.readiness == CaptionReadiness.BLOCKED


def test_no_seed_video_id_caption_still_knows_actual_id(tmp_path: Path) -> None:
    run_dir = _one_shot_run(tmp_path, video_id=None)
    output = finalize(run_dir)
    draft = (run_dir / "caption" / "draft_review_only.md").read_text(encoding="utf-8")
    assert draft.splitlines()[0] == VIDEO_ID
    assert not any("M2-MEDIA-004" in b for b in output.result.blockers)


# --- evidence tampering (§5.1-3) ------------------------------------------


def _manifest_with(run_dir: Path, entries: dict[str, str]) -> None:
    (run_dir / "manifest.json").write_text(
        json.dumps(
            {
                "source_video_sha256": VIDEO_SHA,
                "source_video_path": f"C:/videos/{VIDEO_ID}.mp4",
                "rules_version": RULES_VERSION,
                "artifacts": [
                    {"path": rel, "sha256": sha} for rel, sha in entries.items()
                ],
            }
        ),
        encoding="utf-8",
    )


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _hash_all_consumed(run_dir: Path) -> dict[str, str]:
    entries: dict[str, str] = {}
    for path in run_dir.rglob("*"):
        if path.is_file() and path.name != "manifest.json":
            entries[path.relative_to(run_dir).as_posix()] = _sha(path)
    return entries


@pytest.mark.parametrize(
    "rel_path,payload",
    [
        (
            "visual/actions/action_candidates.json",
            {"action_candidates": [{
                "candidate_id": "AC-1", "shot_number": 1,
                "action_class": "CONTACT_BEGINS", "start_frame": 0, "end_frame": 5,
            }]},
        ),
        (
            "visual/ocr/text_tracks.json",
            {"text_tracks": [{"track_id": "TT-1", "first_candidate_frame": 0}]},
        ),
        (
            "visual/entities/tracks.json",
            {"tracks": [{
                "track_id": "T-1", "entity_type": "CHARACTER",
                "first_frame_index": 0, "last_frame_index": 5,
            }]},
        ),
    ],
)
def test_tampered_evidence_never_finalizes(
    tmp_path: Path, rel_path: str, payload: dict[str, object]
) -> None:
    run_dir = _one_shot_run(tmp_path)
    target = run_dir / rel_path
    write_json(target, payload)
    _manifest_with(run_dir, _hash_all_consumed(run_dir))
    finalize(run_dir)  # untampered baseline passes

    tampered = dict(payload)
    tampered["injected"] = "changed after analysis"
    target.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(CaptionBrainError, match="manifest hash"):
        finalize(run_dir)


def test_unmanifested_evidence_file_is_rejected(tmp_path: Path) -> None:
    run_dir = _one_shot_run(tmp_path)
    entries = _hash_all_consumed(run_dir)
    _manifest_with(run_dir, entries)
    # A file the manifest never recorded appears later: never trusted.
    write_json(
        run_dir / "visual" / "actions" / "action_candidates.json",
        {"action_candidates": []},
    )
    with pytest.raises(CaptionBrainError, match="not listed in manifest"):
        finalize(run_dir)


# --- Phase 4 review carry-forward (§5.1-4) + identity decision (§5.1-9) ----


def _ambiguous_track_run(tmp_path: Path) -> Path:
    run_dir = _one_shot_run(tmp_path)
    write_json(
        run_dir / "visual" / "entities" / "tracks.json",
        {"tracks": [{
            "track_id": "T-AMB", "entity_type": "CHARACTER",
            "first_frame_index": 0, "last_frame_index": 10,
            "identity_ambiguous": True,
        }]},
    )
    return run_dir


def test_phase4_high_identity_review_gates_phase5(tmp_path: Path) -> None:
    run_dir = _ambiguous_track_run(tmp_path)
    output = finalize(run_dir)
    assert output.result.readiness == CaptionReadiness.REVIEW_REQUIRED
    assert any("Ambiguous identity: T-AMB" in b for b in output.result.blockers)


def test_identity_decision_resolves_ambiguity_and_clears_gate(tmp_path: Path) -> None:
    run_dir = _ambiguous_track_run(tmp_path)
    decisions = _decisions_file(
        tmp_path, [_decision("D-ID", "T-AMB", "IDENTITY_MAPPING", "C1")]
    )
    output = finalize(run_dir, review_decisions_path=decisions)
    assert not any("Ambiguous identity" in b for b in output.result.blockers)


# --- speech verification / correction E2E (§5.1-6) -------------------------


def _speech_run(tmp_path: Path) -> Path:
    shots = make_shot_truth([make_shot(1, Fraction(0), Fraction(2), "Opening shot")])
    audio = make_audio_truth(
        [make_speech_region("SR-1", Fraction(0), Fraction(1), "hello world")]
    )
    claims = [
        supported_claim("CLM-C1", SeedClaimType.CHARACTER_EXISTS,
                        "A person in a coat.", subject_ids=["C1"]),
    ]
    return write_run_dir(tmp_path, shots, audio_truth=audio, seed_claims=claims)


def _speaker_facts(tmp_path: Path) -> Path:
    return write_json(
        tmp_path / "hf.json",
        {"facts": [{
            "fact_id": "HF-SPK",
            "fact_type": "SPEECH",
            "semantic_value": {"region_id": "SR-1", "speaker_id": "C1"},
            "character_ids": ["C1"],
            "evidence_refs": _EV,
            "decided_by": "reviewer@test",
            "bound_video_sha256": VIDEO_SHA,
            "bound_rules_version": RULES_VERSION,
        }]},
    )


def test_speech_verification_e2e(tmp_path: Path) -> None:
    run_dir = _speech_run(tmp_path)
    blocked = finalize(run_dir, human_facts_path=_speaker_facts(tmp_path))
    assert blocked.result.speech_blocked_count == 1
    draft = (run_dir / "caption" / "draft_review_only.md").read_text(encoding="utf-8")
    assert "hello world" not in draft

    decisions = _decisions_file(
        tmp_path, [_decision("D-SV", "SR-1", "SPEECH_VERIFICATION", "verified")]
    )
    verified = finalize(
        run_dir,
        review_decisions_path=decisions,
        human_facts_path=_speaker_facts(tmp_path),
    )
    assert verified.result.speech_verified_count == 1
    assert verified.result.speech_blocked_count == 0
    draft = (run_dir / "caption" / "draft_review_only.md").read_text(encoding="utf-8")
    assert 'C1 says, "hello world"' in draft
    golden = json.loads(
        (run_dir / "caption" / "golden_gate.json").read_text(encoding="utf-8")
    )
    by_cat = {c["category"]: c["status"] for c in golden["categories"]}
    assert by_cat["DIALOGUE_COVERAGE"] == "PASS"


def test_speech_correction_e2e_preserves_asr_original(tmp_path: Path) -> None:
    run_dir = _speech_run(tmp_path)
    decisions = _decisions_file(
        tmp_path,
        [_decision("D-SC", "SR-1", "SPEECH_CORRECTION", "hello there, world")],
    )
    output = finalize(
        run_dir,
        review_decisions_path=decisions,
        human_facts_path=_speaker_facts(tmp_path),
    )
    assert output.result.speech_verified_count == 1
    draft = (run_dir / "caption" / "draft_review_only.md").read_text(encoding="utf-8")
    assert 'C1 says, "hello there, world"' in draft
    # The original ASR transcript artifact is untouched on disk.
    audio_qc = json.loads(
        (run_dir / "audio" / "audio_qc.json").read_text(encoding="utf-8")
    )
    assert audio_qc["speech_regions"][0]["text_candidate"] == "hello world"


# --- OCR verification / correction E2E (§5.1-7) ----------------------------


def _write_frames_jsonl(run_dir: Path, frame_count: int, fps: int = 24) -> None:
    lines = [
        json.dumps(
            {
                "record_type": "ledger_header",
                "stream_index": 0,
                "time_base": f"1/{fps}",
                "frame_count": frame_count,
            }
        )
    ]
    for i in range(frame_count):
        lines.append(
            json.dumps(
                {
                    "frame_index": i,
                    "pts": i,
                    "pts_time_seconds": f"{i}/{fps}",
                    "key_frame": i == 0,
                }
            )
        )
    (run_dir / "frames.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _ocr_run(tmp_path: Path) -> Path:
    run_dir = _one_shot_run(tmp_path)
    _write_frames_jsonl(run_dir, 48)
    write_json(
        run_dir / "visual" / "ocr" / "text_tracks.json",
        {"text_tracks": [{
            "track_id": "TT-1",
            "first_candidate_frame": 2,
            "first_stable_frame": 2,
            "last_stable_frame": 40,
            "total_support_frames": 12,
            "consensus": {
                "consensus_text": "best Drone that i've ever owned",
                "support_frames": 12,
            },
        }]},
    )
    return run_dir


def test_material_unverified_ocr_blocks_and_gates_golden(tmp_path: Path) -> None:
    """§5.1-15: 'no eligible OCR facts' is never PASS while a material overlay
    awaits verification."""
    run_dir = _ocr_run(tmp_path)
    output = finalize(run_dir)
    assert output.result.readiness == CaptionReadiness.REVIEW_REQUIRED
    golden = json.loads(
        (run_dir / "caption" / "golden_gate.json").read_text(encoding="utf-8")
    )
    by_cat = {c["category"]: c["status"] for c in golden["categories"]}
    assert by_cat["TEXT_COVERAGE"] == "REVIEW_REQUIRED"


def test_ocr_verification_e2e(tmp_path: Path) -> None:
    run_dir = _ocr_run(tmp_path)
    decisions = _decisions_file(
        tmp_path, [_decision("D-TV", "TT-1", "TEXT_VERIFICATION", "verified")]
    )
    output = finalize(run_dir, review_decisions_path=decisions)
    assert output.result.text_verified_count == 1
    draft = (run_dir / "caption" / "draft_review_only.md").read_text(encoding="utf-8")
    # Exact odd capitalization preserved; the original machine TextTrack became
    # eligible — no duplicate human overlay fact required.
    assert 'On-screen text reads "best Drone that i\'ve ever owned"' in draft


def test_ocr_correction_e2e_never_overwrites_observations(tmp_path: Path) -> None:
    run_dir = _ocr_run(tmp_path)
    decisions = _decisions_file(
        tmp_path,
        [_decision("D-TC", "TT-1", "TEXT_CORRECTION", "best Drone that I have ever owned")],
    )
    output = finalize(run_dir, review_decisions_path=decisions)
    assert output.result.text_verified_count == 1
    draft = (run_dir / "caption" / "draft_review_only.md").read_text(encoding="utf-8")
    assert "best Drone that I have ever owned" in draft
    # Raw machine consensus on disk is untouched.
    tracks = json.loads(
        (run_dir / "visual" / "ocr" / "text_tracks.json").read_text(encoding="utf-8")
    )
    assert (
        tracks["text_tracks"][0]["consensus"]["consensus_text"]
        == "best Drone that i've ever owned"
    )


# --- transition classification E2E (§5.1-8) --------------------------------


def test_transition_classification_e2e(tmp_path: Path) -> None:
    shots = make_shot_truth(
        [
            make_shot(1, Fraction(0), Fraction(1), "Opening shot"),
            make_shot(2, Fraction(1), Fraction(2), None, TransitionStatus.UNRESOLVED),
        ]
    )
    run_dir = write_run_dir(tmp_path, shots)
    unresolved = finalize(run_dir)
    assert any("transition" in b for b in unresolved.result.blockers)

    decisions = _decisions_file(
        tmp_path,
        [_decision("D-TR", "TRANSITION-2", "TRANSITION_CLASSIFICATION", "Hard cut")],
    )
    resolved = finalize(run_dir, review_decisions_path=decisions)
    assert not any("transition unresolved" in b for b in resolved.result.blockers)
    draft = (run_dir / "caption" / "draft_review_only.md").read_text(encoding="utf-8")
    assert "Hard cut" in draft


def test_transition_decision_cannot_make_later_shot_opening(tmp_path: Path) -> None:
    shots = make_shot_truth(
        [
            make_shot(1, Fraction(0), Fraction(1), "Opening shot"),
            make_shot(2, Fraction(1), Fraction(2), None, TransitionStatus.UNRESOLVED),
        ]
    )
    run_dir = write_run_dir(tmp_path, shots)
    decisions = _decisions_file(
        tmp_path,
        [_decision("D-TR", "TRANSITION-2", "TRANSITION_CLASSIFICATION", "Opening shot")],
    )
    output = finalize(run_dir, review_decisions_path=decisions)
    # INVALID_VALUE: the transition stays unresolved; never applied.
    assert any("transition unresolved" in b for b in output.result.blockers)


# --- action boundary exact-time recomputation E2E (§5.1-10) ----------------


def test_action_boundary_changes_rendered_timestamps(tmp_path: Path) -> None:
    run_dir = _one_shot_run(tmp_path)
    _write_frames_jsonl(run_dir, 48)
    write_json(
        run_dir / "visual" / "actions" / "action_candidates.json",
        {"action_candidates": [{
            "candidate_id": "AC-1", "shot_number": 1,
            "action_class": "CONTACT_BEGINS",
            "start_frame": 0, "end_frame": 5,
            "start_exact": "0", "end_exact": "5/24",
        }]},
    )
    seed_claims = [
        supported_claim("CLM-C1", SeedClaimType.CHARACTER_EXISTS,
                        "A person.", subject_ids=["C1"]),
    ]
    (run_dir / "seed" / "seed_claims.json").write_text(
        json.dumps({"claims": [c.model_dump(mode="json") for c in seed_claims]}),
        encoding="utf-8",
    )
    decisions = _decisions_file(
        tmp_path,
        [
            _decision("D-AS", "AC-1", "ACTION_SEMANTICS", "C1 touches the counter."),
            _decision("D-AB", "AC-1", "ACTION_BOUNDARY", "12-24"),
        ],
    )
    output = finalize(run_dir, review_decisions_path=decisions)
    assert output.result.action_verified_count == 1
    draft = (run_dir / "caption" / "draft_review_only.md").read_text(encoding="utf-8")
    # 12/24 = 0.5s, 24/24 = 1.0s — the rendered stamps visibly moved with the
    # human boundary; stale exact timing is impossible.
    assert "[0.5s-1.0s] C1 touches the counter." in draft


# --- human-fact evidence + protected traits (§5.1-11/13) -------------------


def test_human_fact_without_evidence_cannot_reach_ready(tmp_path: Path) -> None:
    run_dir = _one_shot_run(tmp_path)
    facts = write_json(
        tmp_path / "hf.json",
        {"facts": [{
            "fact_id": "HF-NOEV",
            "fact_type": "VISUAL_ACTION",
            "text_value": "C1 waves at the camera.",
            "shot_number": 1,
            "character_ids": ["C1"],
            "start_exact": "0",
            "end_exact": "1",
            "decided_by": "reviewer@test",
            "bound_video_sha256": VIDEO_SHA,
            "bound_rules_version": RULES_VERSION,
        }]},
    )
    output = finalize(run_dir, human_facts_path=facts)
    assert output.result.readiness == CaptionReadiness.REVIEW_REQUIRED
    assert any("no evidence reference" in b for b in output.result.blockers)
    draft = (run_dir / "caption" / "draft_review_only.md").read_text(encoding="utf-8")
    assert "C1 waves at the camera." not in draft


def test_protected_trait_human_fact_guard(tmp_path: Path) -> None:
    run_dir = _one_shot_run(tmp_path)
    facts = write_json(
        tmp_path / "hf.json",
        {"facts": [{
            "fact_id": "HF-TRAIT",
            "fact_type": "CHARACTER",
            "text_value": "C1 is a 25-year-old American man.",
            "character_ids": ["C1"],
            "shot_number": None,
            "evidence_refs": _EV,
            "decided_by": "reviewer@test",
            "bound_video_sha256": VIDEO_SHA,
            "bound_rules_version": RULES_VERSION,
        }]},
    )
    output = finalize(run_dir, human_facts_path=facts)
    draft = (run_dir / "caption" / "draft_review_only.md").read_text(encoding="utf-8")
    assert "25-year-old" not in draft
    assert any("protected" in b for b in output.result.blockers)


def test_blocked_material_action_gates_detail_coverage(tmp_path: Path) -> None:
    run_dir = _one_shot_run(tmp_path)
    facts = write_json(
        tmp_path / "hf.json",
        {"facts": [{
            "fact_id": "HF-NOEV",
            "fact_type": "VISUAL_ACTION",
            "text_value": "C1 waves at the camera.",
            "shot_number": 1,
            "character_ids": ["C1"],
            "start_exact": "0",
            "end_exact": "1",
            "decided_by": "reviewer@test",
            "bound_video_sha256": VIDEO_SHA,
            "bound_rules_version": RULES_VERSION,
        }]},
    )
    finalize(run_dir, human_facts_path=facts)
    golden = json.loads(
        (run_dir / "caption" / "golden_gate.json").read_text(encoding="utf-8")
    )
    by_cat = {c["category"]: c["status"] for c in golden["categories"]}
    assert by_cat["DETAIL_COVERAGE"] == "REVIEW_REQUIRED"


# --- reviewed_caption parity (§5.1-14) -------------------------------------


def test_reviewed_caption_parity(tmp_path: Path) -> None:
    """Playback speed, speech, OCR and camera-adjacent structure in
    reviewed_caption.json must mirror the rendered caption."""
    shots = make_shot_truth([make_shot(1, Fraction(0), Fraction(2), "Opening shot")])
    audio = make_audio_truth(
        [make_speech_region("SR-1", Fraction(0), Fraction(1), "hello world")]
    )
    claims = [
        supported_claim("CLM-C1", SeedClaimType.CHARACTER_EXISTS,
                        "A person in a coat.", subject_ids=["C1"]),
    ]
    run_dir = write_run_dir(tmp_path, shots, audio_truth=audio, seed_claims=claims)
    _write_frames_jsonl(run_dir, 48)
    write_json(
        run_dir / "visual" / "speed" / "playback_speed_evidence.json",
        {"playback_speed_evidence": [{
            "shot_number": 1, "conclusion": "REGULAR_CANDIDATE", "review_required": True,
        }]},
    )
    write_json(
        run_dir / "visual" / "ocr" / "text_tracks.json",
        {"text_tracks": [{
            "track_id": "TT-1", "first_candidate_frame": 2,
            "first_stable_frame": 2, "last_stable_frame": 40,
            "total_support_frames": 12,
            "consensus": {"consensus_text": "LEVEL UP", "support_frames": 12},
        }]},
    )
    decisions = _decisions_file(
        tmp_path,
        [
            _decision("D-SP", "SPEED-1", "PLAYBACK_SPEED", "regular"),
            _decision("D-SV", "SR-1", "SPEECH_VERIFICATION", "verified"),
            _decision("D-TV", "TT-1", "TEXT_VERIFICATION", "verified"),
        ],
    )
    finalize(
        run_dir,
        review_decisions_path=decisions,
        human_facts_path=_speaker_facts(tmp_path),
    )
    reviewed = json.loads(
        (run_dir / "caption" / "reviewed_caption.json").read_text(encoding="utf-8")
    )
    draft = (run_dir / "caption" / "draft_review_only.md").read_text(encoding="utf-8")

    # Speed parity.
    assert reviewed["shots"][0]["playback_speed"] == "regular"
    assert "Video playback speed: regular" in draft
    # Speech parity: verbatim text + speaker, timing in time_range not text.
    speech = reviewed["speech_events"]
    assert len(speech) == 1
    assert speech[0]["verbatim_text"] == "hello world"
    assert speech[0]["speaker_id"] == "C1"
    # OCR parity.
    texts = reviewed["on_screen_text_events"]
    assert len(texts) == 1
    assert texts[0]["text"] == "LEVEL UP"
    assert 'On-screen text reads "LEVEL UP"' in draft
    # Action & Audio parity: one structured record per rendered event line,
    # with no duplicated timestamp inside the text.
    action_lines = [
        ln for ln in draft.splitlines() if ln.startswith("[") and "ACTION" not in ln
    ]
    assert len(reviewed["action_audio_events"]) >= 1
    for event in reviewed["action_audio_events"]:
        assert not event["text"].startswith("[")
        assert any(event["text"] in line for line in action_lines)
