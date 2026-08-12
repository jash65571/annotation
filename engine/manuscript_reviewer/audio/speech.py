"""Speech-region builder: ASR words → defensible speech regions + recall defense.

Regions split at real word gaps (rule: timing.speech_pause_split_seconds),
never at punctuation; separate vocal acts are never merged just because one
ASR segment contains both. Absence of ASR output is NEVER treated as silence
truth — meaningful audio without ASR coverage becomes REVIEW_REQUIRED
(AUDIO_WITHOUT_ASR_COVERAGE), and likely vocal material without trustworthy
text stays visible as VOCAL_REVIEW_REQUIRED.
"""

from __future__ import annotations

import logging
from fractions import Fraction

from ..media.timestamps import format_manuscript_display
from ..models.audio import (
    AlignmentStatus,
    ASRResult,
    ASRStatus,
    AudioRegion,
    AudioRegionKind,
    AudioReviewReason,
    AudioTimeline,
    BestWord,
    EvidenceState,
    SourceVerificationStatus,
    SpeechRegion,
    WordTimingStatus,
)
from ..rules.loader import load_rules

logger = logging.getLogger(__name__)

LOW_WORD_PROBABILITY = 0.5
#: Uncovered active audio must last at least this long to demand review.
MIN_UNCOVERED_SECONDS = Fraction(2, 5)
#: Speech within this distance of a clip edge is flagged as possibly clipped.
CLIP_EDGE_SECONDS = Fraction(3, 20)


def _pause_threshold() -> Fraction:
    value = load_rules().get("timing.speech_pause_split_seconds", 0.5)
    return Fraction(str(value))


def build_speech_regions(
    result: ASRResult,
    best_words: list[BestWord],
    best_status: AlignmentStatus,
    alignment_coverage: float | None,
    timeline: AudioTimeline,
    annotation_endpoint: Fraction | None,
) -> list[SpeechRegion]:
    """Group reconciled best words into speech regions split at real audio gaps.

    EVERY machine speech region defaults to ``source_verification_status =
    UNVERIFIED`` and always carries ``MANDATORY_SOURCE_AUDIO_VERIFICATION``: a
    clean, high-confidence, fully-aligned result never bypasses a human listen.
    The machine ``EvidenceState`` still records processing quality
    (ALIGNED_EVIDENCE vs ASR_EVIDENCE) but NEVER suppresses that review (§2/§3).
    """
    if result.status != ASRStatus.PASS:
        return []
    if not best_words:
        return []
    pause = _pause_threshold()

    groups: list[list[BestWord]] = [[best_words[0]]]
    for word in best_words[1:]:
        gap = word.start_annotation_time - groups[-1][-1].end_annotation_time
        if gap > pause:
            groups.append([word])
        else:
            groups[-1].append(word)

    regions: list[SpeechRegion] = []
    for index, group in enumerate(groups, start=1):
        start = group[0].start_annotation_time
        end = group[-1].end_annotation_time
        probabilities = [w.probability for w in group if w.probability is not None]
        mean_probability = (
            sum(probabilities) / len(probabilities) if probabilities else None
        )
        # Mandatory human source verification is ALWAYS required for machine speech.
        reasons: list[AudioReviewReason] = [
            AudioReviewReason.MANDATORY_SOURCE_AUDIO_VERIFICATION
        ]
        if mean_probability is not None and mean_probability < LOW_WORD_PROBABILITY:
            reasons.append(AudioReviewReason.LOW_ASR_CONFIDENCE)
        if best_status == AlignmentStatus.PARTIAL:
            reasons.append(AudioReviewReason.PARTIAL_ALIGNMENT)
        elif best_status not in (
            AlignmentStatus.ALIGNED,
            AlignmentStatus.NOT_ATTEMPTED,
        ):
            reasons.append(AudioReviewReason.MISSING_ALIGNMENT)
        if start <= CLIP_EDGE_SECONDS:
            reasons.append(AudioReviewReason.SPEECH_AT_CLIP_START)
        if annotation_endpoint is not None and annotation_endpoint - end <= CLIP_EDGE_SECONDS:
            reasons.append(AudioReviewReason.SPEECH_AT_CLIP_END)

        # Machine processing state — NOT a human-verification claim.
        state = (
            EvidenceState.ALIGNED_EVIDENCE
            if best_status == AlignmentStatus.ALIGNED
            else EvidenceState.ASR_EVIDENCE
        )
        has_aligned = any(
            w.timing_status == WordTimingStatus.WHISPERX_ALIGNED for w in group
        )
        regions.append(
            SpeechRegion(
                region_id=f"speech_{index:04d}",
                start_exact=start,
                end_exact=end,
                start_manuscript=format_manuscript_display(start),
                end_manuscript=format_manuscript_display(end),
                sources=["asr_faster_whisper"]
                + (["whisperx_alignment"] if has_aligned else []),
                text_candidate=" ".join(w.text for w in group),
                language=result.language,
                mean_word_probability=(
                    round(mean_probability, 4) if mean_probability is not None else None
                ),
                alignment_status=best_status,
                alignment_coverage=alignment_coverage,
                state=state,
                source_verification_status=SourceVerificationStatus.UNVERIFIED,
                review_reasons=reasons,
            )
        )
    return regions


def find_uncovered_audio(
    active_regions: list[AudioRegion],
    speech_regions: list[SpeechRegion],
    silence_regions: list[AudioRegion],
) -> list[tuple[Fraction, Fraction]]:
    """Meaningful active-audio windows with no ASR coverage (VAD recall defense).

    Decision (docs/07): instead of a second no-VAD transcription pass over the
    whole clip, uncovered regions go to human review — cheaper, and a human
    listen is the only trustworthy resolution for VAD-missed material anyway.
    """
    uncovered: list[tuple[Fraction, Fraction]] = []
    for region in active_regions:
        if region.kind != AudioRegionKind.ACTIVE_AUDIO:
            continue
        cursor = region.start_annotation_time
        end = region.end_annotation_time
        overlaps = sorted(
            (
                (max(s.start_exact, cursor), min(s.end_exact, end))
                for s in speech_regions
                if s.end_exact > cursor and s.start_exact < end
            ),
            key=lambda pair: pair[0],
        )
        for overlap_start, overlap_end in overlaps:
            if overlap_start - cursor >= MIN_UNCOVERED_SECONDS:
                uncovered.append((cursor, overlap_start))
            cursor = max(cursor, overlap_end)
        if end - cursor >= MIN_UNCOVERED_SECONDS:
            uncovered.append((cursor, end))
    return uncovered
