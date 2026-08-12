"""Phase 3 audio validators (P3-*)."""

from __future__ import annotations

import itertools
from fractions import Fraction

from ..models.audio import (
    AlignmentResult,
    AlignmentStatus,
    ASRResult,
    ASRStatus,
    AudioEnergyBin,
    AudioFrameRecord,
    AudioReviewItem,
    AudioTimeline,
    AudioVerificationStatus,
    BoundaryAudioEvidence,
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
            for a, b in itertools.pairwise(words):
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
            for word in words:
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
    if alignment is not None and alignment.status == AlignmentStatus.TEXT_MISMATCH:
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
    no_audio: bool,
) -> str:
    """P3-QC-001: overall PASS is forbidden while high-risk unresolved audio
    regions (review items, crossing boundaries) remain."""
    if no_audio:
        return "NO_AUDIO_STREAM"
    if any(i.severity == Severity.FAIL for i in issues):
        return "FAILED"
    crossing = any(
        b.audio_verification_status == AudioVerificationStatus.CROSSING_REVIEW_REQUIRED
        for b in boundary_evidence
    )
    if review_items or crossing:
        return "REVIEW_REQUIRED"
    return "PASS"
