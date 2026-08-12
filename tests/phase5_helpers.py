"""Shared Phase 5 test factories: synthetic shot/audio/visual evidence and
human inputs. Pure-logic — no ffmpeg required."""

from __future__ import annotations

import json
from fractions import Fraction
from pathlib import Path

from manuscript_reviewer.caption.eligibility import EligibilityContext
from manuscript_reviewer.caption.facts import FactBuildInputs
from manuscript_reviewer.caption_brain import CaptionBrainOutput, finalize_run
from manuscript_reviewer.media.timestamps import format_manuscript_display
from manuscript_reviewer.models.audio import (
    AlignmentStatus,
    ASRStatus,
    AudioQCResult,
    AudioStatus,
    SourceVerificationStatus,
    SpeechRegion,
)
from manuscript_reviewer.models.caption import SeedClaim
from manuscript_reviewer.models.evidence import EvidenceReference, EvidenceType
from manuscript_reviewer.models.review_intelligence import (
    ClaimReviewStatus,
    DecisionType,
    EvidenceStatus,
    HumanReviewDecision,
    PlaybackSpeedEvidence,
    SeedClaimType,
    SpeedConclusion,
)
from manuscript_reviewer.models.shot_truth import (
    CandidateStatus,
    ShotProposal,
    ShotTruthResult,
    TransitionStatus,
)
from manuscript_reviewer.review.decisions import DecisionTargets, apply_decisions

VIDEO_SHA = "a" * 64
RULES_VERSION = "1.3.0"
VIDEO_ID = "747e3e2754d8_0.0_2.0"


def make_shot(
    index: int,
    start: Fraction,
    end: Fraction,
    transition: str | None,
    transition_status: TransitionStatus = TransitionStatus.PROPOSED,
    fps: int = 24,
) -> ShotProposal:
    return ShotProposal(
        shot_index=index,
        start_frame_index=int(start * fps),
        end_frame_index=int(end * fps) - 1,
        start_exact=start,
        end_exact=end,
        last_owned_frame_start_exact=end - Fraction(1, fps),
        start_manuscript=format_manuscript_display(start),
        end_manuscript=format_manuscript_display(end),
        transition_into_shot=transition,
        transition_status=transition_status,
        supporting_boundary_id=None,
        review_status=CandidateStatus.SUPPORTED,
    )


def make_shot_truth(shots: list[ShotProposal]) -> ShotTruthResult:
    endpoint = shots[-1].end_exact if shots else None
    return ShotTruthResult(
        frame_count=0,
        adjacent_pair_count=0,
        raw_candidate_count=0,
        merged_candidate_count=0,
        supported_count=max(0, len(shots) - 1),
        rejected_count=0,
        review_required_count=0,
        proposed_shot_count=len(shots),
        overall_status="PASS",
        annotation_timeline_origin=Fraction(0),
        annotation_endpoint_exact=endpoint,
        annotation_endpoint_method="test",
        candidates=[],
        shots=shots,
    )


def make_speech_region(
    region_id: str,
    start: Fraction,
    end: Fraction,
    text: str | None,
    status: SourceVerificationStatus = SourceVerificationStatus.UNVERIFIED,
    corrected: str | None = None,
) -> SpeechRegion:
    return SpeechRegion(
        region_id=region_id,
        start_exact=start,
        end_exact=end,
        start_manuscript=format_manuscript_display(start),
        end_manuscript=format_manuscript_display(end),
        sources=["test"],
        text_candidate=text,
        corrected_text=corrected,
        source_verification_status=status,
    )


def make_audio_truth(regions: list[SpeechRegion]) -> AudioQCResult:
    return AudioQCResult(
        audio_status=AudioStatus.ANALYZED,
        asr_status=ASRStatus.PASS,
        alignment_status=AlignmentStatus.ALIGNED,
        overall_status="REVIEW_REQUIRED",
        speech_region_count=len(regions),
        speech_regions=regions,
    )


def factual_ref(ref_id: str, start_frame: int = 0) -> EvidenceReference:
    return EvidenceReference(
        evidence_id=ref_id,
        evidence_type=EvidenceType.FRAME_RANGE,
        start_frame=start_frame,
        end_frame=start_frame + 1,
        source="test",
    )


def supported_claim(
    claim_id: str,
    claim_type: SeedClaimType,
    text: str,
    subject_ids: list[str] | None = None,
    object_ids: list[str] | None = None,
    shot_number: int | None = None,
    source_field: str = "Characters",
) -> SeedClaim:
    return SeedClaim(
        claim_id=claim_id,
        source_field=source_field,
        text=text,
        claim_type=claim_type,
        subject_ids=subject_ids or [],
        object_ids=object_ids or [],
        shot_number=shot_number,
        evidence_status=EvidenceStatus.SUPPORTED,
        review_status=ClaimReviewStatus.MACHINE_ONLY,
        evidence=[factual_ref(f"EV-{claim_id}")],
    )


def human_decision(
    decision_id: str, subject_id: str, decision_type: DecisionType, value: str
) -> HumanReviewDecision:
    return HumanReviewDecision(
        decision_id=decision_id,
        subject_id=subject_id,
        decision_type=decision_type,
        value=value,
        decided_by="reviewer@test",
        decided_at_utc="2026-08-12T00:00:00Z",
        bound_video_sha256=VIDEO_SHA,
        bound_rules_version=RULES_VERSION,
    )


def make_speed_evidence(
    shot_number: int, conclusion: SpeedConclusion = SpeedConclusion.REGULAR_CANDIDATE
) -> PlaybackSpeedEvidence:
    return PlaybackSpeedEvidence(
        shot_number=shot_number, conclusion=conclusion, review_required=True
    )


def build_ctx(
    decisions: list[HumanReviewDecision],
    speed_evidence: list[PlaybackSpeedEvidence] | None = None,
) -> EligibilityContext:
    """Apply decisions through the REAL Phase 4 application layer so the
    eligibility context only ever sees genuinely APPLIED decisions."""
    targets = DecisionTargets(
        speed_evidence={f"SPEED-{s.shot_number}": s for s in (speed_evidence or [])}
    )
    applications = apply_decisions(decisions, targets, VIDEO_SHA, RULES_VERSION)
    return EligibilityContext.build(decisions, applications)


def base_inputs(**overrides: object) -> FactBuildInputs:
    defaults: dict[str, object] = {
        "video_id": VIDEO_ID,
        "video_sha256": VIDEO_SHA,
        "rules_version": RULES_VERSION,
    }
    defaults.update(overrides)
    return FactBuildInputs(**defaults)  # type: ignore[arg-type]


def write_run_dir(
    tmp_path: Path,
    shot_truth: ShotTruthResult,
    audio_truth: AudioQCResult | None = None,
    seed_claims: list[SeedClaim] | None = None,
    video_id: str | None = VIDEO_ID,
    seed_sha: str | None = "b" * 64,
) -> Path:
    """A minimal synthetic run directory the Caption Brain can finalize from
    (no media analysis, §94/§123)."""
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True, exist_ok=True)
    # No manifest.json: a manifest's mere existence forces STRICT hash mode
    # (§5.2-5), so synthetic dirs run in the in-process trust mode and pass
    # video sha / canonical id explicitly via finalize(). Tamper regressions
    # write explicit full-hash manifests instead.
    (run_dir / "shot_qc.json").write_text(
        shot_truth.model_dump_json(), encoding="utf-8"
    )
    if audio_truth is not None:
        audio_dir = run_dir / "audio"
        audio_dir.mkdir(exist_ok=True)
        (audio_dir / "audio_qc.json").write_text(
            audio_truth.model_dump_json(), encoding="utf-8"
        )
    seed_dir = run_dir / "seed"
    seed_dir.mkdir(exist_ok=True)
    if video_id is not None:
        (seed_dir / "seed_parse.json").write_text(
            json.dumps({"video_id": video_id}), encoding="utf-8"
        )
    if seed_sha is not None:
        (seed_dir / "seed_sha256.txt").write_text(seed_sha, encoding="utf-8")
    if seed_claims:
        (seed_dir / "seed_claims.json").write_text(
            json.dumps(
                {"claims": [c.model_dump(mode="json") for c in seed_claims]},
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
    return run_dir


def write_json(path: Path, payload: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return path


def finalize(
    run_dir: Path,
    review_decisions_path: Path | None = None,
    human_facts_path: Path | None = None,
    final_review_path: Path | None = None,
) -> CaptionBrainOutput:
    """finalize_run with the synthetic run's provenance supplied explicitly
    (manifest-less trust mode)."""
    return finalize_run(
        run_dir,
        review_decisions_path=review_decisions_path,
        human_facts_path=human_facts_path,
        final_review_path=final_review_path,
        video_sha256=VIDEO_SHA,
        canonical_video_id=VIDEO_ID,
    )
