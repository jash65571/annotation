"""Phase 3 audio validators (P3-*)."""

from __future__ import annotations

import itertools
from fractions import Fraction

from ..models.audio import (
    RESOLVED_SOURCE_VERIFICATION,
    AlignmentResult,
    AlignmentStatus,
    ASRResult,
    ASRStatus,
    AudioEnergyBin,
    AudioFrameRecord,
    AudioReviewItem,
    AudioReviewReason,
    AudioTimeline,
    AudioVerificationStatus,
    BestWord,
    BoundaryAudioEvidence,
    BoundaryAudioStatus,
    SampleAnchorStatus,
    SourceVerificationStatus,
    SpeechRegion,
)
from ..models.shot_truth import BoundaryCandidate, CandidateStatus
from ..models.validation import Severity, ValidatorIssue


def _issue(rule_id: str, severity: Severity, location: str, message: str) -> ValidatorIssue:
    return ValidatorIssue(rule_id=rule_id, severity=severity, location=location, message=message)


def validate_audio_timeline(
    timeline: AudioTimeline, audio_frames: list[AudioFrameRecord]
) -> list[ValidatorIssue]:
    issues: list[ValidatorIssue] = []
    if timeline.evidence_sample_count <= 0:
        issues.append(
            _issue("P3-AUDIO-002", Severity.FAIL, "audio", "Decoded sample count is 0.")
        )
    pts_values = [f.pts for f in audio_frames if f.pts is not None]
    if any(b <= a for a, b in itertools.pairwise(pts_values)):
        issues.append(
            _issue(
                "P3-AUDIO-003",
                Severity.FAIL,
                "audio frames",
                "Audio frame timeline is not strictly monotonic.",
            )
        )
    if timeline.annotation_audio_offset < Fraction(-5):
        issues.append(
            _issue(
                "P3-AUDIO-004",
                Severity.WARN,
                "audio timeline",
                f"Audio anchor precedes the annotation origin by "
                f"{float(-timeline.annotation_audio_offset):.3f}s — unexpectedly large.",
            )
        )
    if timeline.sample_anchor_status == SampleAnchorStatus.ANCHOR_REVIEW_REQUIRED:
        issues.append(
            _issue(
                "P3-AUDIO-009",
                Severity.WARN,
                "audio timeline",
                "AUDIO_SAMPLE_ANCHOR_REVIEW_REQUIRED: codec priming/skip metadata "
                f"(skip_samples={timeline.codec_skip_samples}, "
                f"initial_padding={timeline.initial_padding_samples}) means decoded "
                "PCM sample 0 cannot be proven to anchor the first encoded packet; "
                "sample-perfect source anchoring is not claimed.",
            )
        )
    return issues


def validate_energy_bins(
    bins: list[AudioEnergyBin], timeline: AudioTimeline
) -> list[ValidatorIssue]:
    issues: list[ValidatorIssue] = []
    for a, b in itertools.pairwise(bins):
        if b.start_sample != a.end_sample:
            issues.append(
                _issue(
                    "P3-AUDIO-005",
                    Severity.FAIL,
                    f"energy bin {b.bin_index}",
                    "Energy bins are not sequential/non-overlapping.",
                )
            )
            break
    if bins:
        covered = bins[-1].end_sample
        bin_samples = timeline.evidence_sample_rate // 100
        if timeline.evidence_sample_count - covered >= bin_samples:
            issues.append(
                _issue(
                    "P3-AUDIO-006",
                    Severity.WARN,
                    "energy bins",
                    f"Energy coverage ends at sample {covered} of "
                    f"{timeline.evidence_sample_count}; more than one full bin uncovered.",
                )
            )
    return issues


def validate_asr(result: ASRResult, timeline: AudioTimeline) -> list[ValidatorIssue]:
    issues: list[ValidatorIssue] = []
    if result.status == ASRStatus.PASS:
        audio_start = timeline.annotation_audio_offset
        audio_end = timeline.annotation_audio_offset + timeline.evidence_duration_seconds
        tolerance = Fraction(1, 4)
        for segment in result.segments:
            words = segment.words
            timed = [
                w
                for w in words
                if w.start_annotation_time is not None and w.end_annotation_time is not None
            ]
            for a, b in itertools.pairwise(timed):
                assert a.start_annotation_time is not None
                assert b.start_annotation_time is not None
                if b.start_annotation_time < a.start_annotation_time:
                    issues.append(
                        _issue(
                            "P3-ASR-001",
                            Severity.FAIL,
                            f"segment {segment.segment_id}",
                            "ASR word timestamps are not ordered.",
                        )
                    )
                    break
            for word in timed:
                assert word.start_annotation_time is not None
                assert word.end_annotation_time is not None
                if (
                    word.start_annotation_time < audio_start - tolerance
                    or word.end_annotation_time > audio_end + tolerance
                ):
                    issues.append(
                        _issue(
                            "P3-ASR-002",
                            Severity.WARN,
                            f"segment {segment.segment_id}",
                            f"Word '{word.text}' timing falls outside source audio bounds.",
                        )
                    )
                    break
    elif result.status in (ASRStatus.FAILED, ASRStatus.UNAVAILABLE):
        if not result.runtime.failure_reason:
            issues.append(
                _issue(
                    "P3-ASR-004",
                    Severity.FAIL,
                    "asr",
                    "ASR failure has no recorded reason.",
                )
            )
    return issues


def validate_alignment(
    alignment: AlignmentResult | None,
) -> list[ValidatorIssue]:
    issues: list[ValidatorIssue] = []
    if alignment is None:
        return issues
    if alignment.status == AlignmentStatus.TEXT_MISMATCH:
        issues.append(
            _issue(
                "P3-ASR-003",
                Severity.WARN,
                "alignment",
                "ALIGNMENT_TEXT_MISMATCH: aligned transcript differs from the "
                "faster-whisper source text; faster-whisper wording is preserved "
                "as transcript_best.",
            )
        )
    if alignment.status == AlignmentStatus.PARTIAL:
        coverage = alignment.alignment_coverage
        issues.append(
            _issue(
                "P3-ASR-007",
                Severity.WARN,
                "alignment",
                "ALIGNMENT_PARTIAL: only "
                f"{coverage if coverage is not None else 0:.0%} of source words "
                "received WhisperX timing; the rest keep faster-whisper timing. "
                "No source word was dropped; the region requires review.",
            )
        )
    return issues


def validate_best_words(
    best_words: list[BestWord], asr_result: ASRResult
) -> list[ValidatorIssue]:
    """P3-ASR-006: the best-transcript word sequence MUST match the faster-whisper
    source word sequence exactly (count AND wording). Timing source may differ
    per word, but a word may never be dropped, invented, or re-worded."""
    issues: list[ValidatorIssue] = []
    if asr_result.status != ASRStatus.PASS:
        return issues
    source = [w.text for s in asr_result.segments for w in s.words]
    best = [w.text for w in best_words]
    if source != best:
        issues.append(
            _issue(
                "P3-ASR-006",
                Severity.FAIL,
                "transcript_best",
                f"best word sequence ({len(best)} words) differs from the "
                f"faster-whisper source sequence ({len(source)} words); a source "
                "word was dropped, invented, or re-worded.",
            )
        )
    return issues


def validate_source_verification(
    speech_regions: list[SpeechRegion], review_items: list[AudioReviewItem]
) -> list[ValidatorIssue]:
    """P3-SPEECH-004: every machine speech region that is not human-verified MUST
    have an unresolved mandatory-source-verification review item referencing it.
    Machine-aligned dialogue can never silently skip a human listen."""
    issues: list[ValidatorIssue] = []
    verified_refs = {
        ref
        for item in review_items
        if AudioReviewReason.MANDATORY_SOURCE_AUDIO_VERIFICATION in item.reasons
        for ref in item.evidence_refs
    }
    for region in speech_regions:
        # Verified / corrected / rejected are all resolved dispositions.
        if region.source_verification_status in RESOLVED_SOURCE_VERIFICATION:
            continue
        if region.region_id not in verified_refs:
            issues.append(
                _issue(
                    "P3-SPEECH-004",
                    Severity.FAIL,
                    region.region_id,
                    "Source-unverified speech region has no mandatory "
                    "source-audio-verification review item.",
                )
            )
    return issues


def validate_boundary_continuity(
    boundary_evidence: list[BoundaryAudioEvidence],
) -> list[ValidatorIssue]:
    """P3-BOUNDARY-003: CONTINUOUS requires meaningful continuity evidence — an
    actual word spanning the boundary or proven spectral continuity. Energy
    presence on both sides alone never justifies CONTINUOUS (§13)."""
    issues: list[ValidatorIssue] = []
    for boundary in boundary_evidence:
        if boundary.audio_continuity_status != BoundaryAudioStatus.CONTINUOUS:
            continue
        codes = set(boundary.continuity_evidence_codes)
        if not codes & {"ASR_WORD_SPANS_BOUNDARY", "SPECTRAL_CONTINUITY"}:
            issues.append(
                _issue(
                    "P3-BOUNDARY-003",
                    Severity.FAIL,
                    boundary.boundary_candidate_id,
                    "Boundary marked CONTINUOUS without meaningful continuity "
                    "evidence (no crossing word, no spectral continuity).",
                )
            )
    return issues


def validate_speech_regions(
    regions: list[SpeechRegion], timeline: AudioTimeline
) -> list[ValidatorIssue]:
    issues: list[ValidatorIssue] = []
    audio_start = timeline.annotation_audio_offset
    audio_end = timeline.annotation_audio_offset + timeline.evidence_duration_seconds
    tolerance = Fraction(1, 4)
    for region in regions:
        if region.start_exact >= region.end_exact:
            issues.append(
                _issue(
                    "P3-SPEECH-001",
                    Severity.FAIL,
                    region.region_id,
                    "Speech region start is not before its end.",
                )
            )
        if (
            region.start_exact < audio_start - tolerance
            or region.end_exact > audio_end + tolerance
        ):
            issues.append(
                _issue(
                    "P3-SPEECH-002",
                    Severity.WARN,
                    region.region_id,
                    "Speech region extends beyond source audio bounds.",
                )
            )
    return issues


def validate_boundaries(
    candidates: list[BoundaryCandidate],
    boundary_evidence: list[BoundaryAudioEvidence],
    has_audio: bool,
) -> list[ValidatorIssue]:
    issues: list[ValidatorIssue] = []
    if not has_audio:
        return issues
    covered = {b.boundary_candidate_id for b in boundary_evidence}
    for candidate in candidates:
        if candidate.status == CandidateStatus.SUPPORTED and candidate.candidate_id not in covered:
            issues.append(
                _issue(
                    "P3-BOUNDARY-001",
                    Severity.FAIL,
                    candidate.candidate_id,
                    "Supported visual boundary received no audio-continuity evidence.",
                )
            )
    return issues


def compute_audio_status(
    issues: list[ValidatorIssue],
    review_items: list[AudioReviewItem],
    boundary_evidence: list[BoundaryAudioEvidence],
    speech_regions: list[SpeechRegion],
    no_audio: bool,
) -> str:
    """P3-QC-001: overall PASS is forbidden while high-risk unresolved audio
    regions (review items, crossing boundaries) remain — AND, explicitly, while
    ANY machine speech region is still source-unverified (§4). A clean aligned
    transcript never yields a top-level audio PASS on its own."""
    if no_audio:
        return "NO_AUDIO_STREAM"
    if any(i.severity == Severity.FAIL for i in issues):
        return "FAILED"
    source_unverified_speech = any(
        r.source_verification_status == SourceVerificationStatus.UNVERIFIED
        for r in speech_regions
    )
    crossing = any(
        b.audio_verification_status == AudioVerificationStatus.CROSSING_REVIEW_REQUIRED
        for b in boundary_evidence
    )
    if review_items or crossing or source_unverified_speech:
        return "REVIEW_REQUIRED"
    return "PASS"
