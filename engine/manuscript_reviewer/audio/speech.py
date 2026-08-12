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
    AlignmentResult,
    AlignmentStatus,
    ASRResult,
    ASRStatus,
    ASRWord,
    AudioRegion,
    AudioRegionKind,
    AudioReviewReason,
    AudioTimeline,
    EvidenceState,
    SpeechRegion,
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


def _flatten_words(result: ASRResult, alignment: AlignmentResult | None) -> list[ASRWord]:
    """Words with the best available timing: WhisperX when validly aligned,
    faster-whisper otherwise. Wording always comes from faster-whisper."""
    if alignment is not None and alignment.status == AlignmentStatus.ALIGNED:
        words = [w for s in alignment.segments for w in s.words]
        if words:
            return words
    return [w for s in result.segments for w in s.words]


def build_speech_regions(
    result: ASRResult,
    alignment: AlignmentResult | None,
    timeline: AudioTimeline,
    annotation_endpoint: Fraction | None,
) -> list[SpeechRegion]:
    """Group ASR words into speech regions split at real audio gaps."""
    if result.status != ASRStatus.PASS:
        return []
    words = _flatten_words(result, alignment)
    if not words:
        return []
    pause = _pause_threshold()
    alignment_status = (
        alignment.status if alignment is not None else AlignmentStatus.NOT_ATTEMPTED
    )

    groups: list[list[ASRWord]] = [[words[0]]]
    for word in words[1:]:
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
        reasons: list[AudioReviewReason] = []
        if mean_probability is not None and mean_probability < LOW_WORD_PROBABILITY:
            reasons.append(AudioReviewReason.LOW_ASR_CONFIDENCE)
        if alignment_status not in (AlignmentStatus.ALIGNED, AlignmentStatus.NOT_ATTEMPTED):
            reasons.append(AudioReviewReason.MISSING_ALIGNMENT)
        if start <= CLIP_EDGE_SECONDS:
            reasons.append(AudioReviewReason.SPEECH_AT_CLIP_START)
        if annotation_endpoint is not None and annotation_endpoint - end <= CLIP_EDGE_SECONDS:
            reasons.append(AudioReviewReason.SPEECH_AT_CLIP_END)

        state = (
            EvidenceState.REVIEW_REQUIRED
            if reasons
            else (
                EvidenceState.ALIGNED_EVIDENCE
                if alignment_status == AlignmentStatus.ALIGNED
                else EvidenceState.ASR_EVIDENCE
            )
        )
        regions.append(
            SpeechRegion(
                region_id=f"speech_{index:04d}",
                start_exact=start,
                end_exact=end,
                start_manuscript=format_manuscript_display(start),
                end_manuscript=format_manuscript_display(end),
                sources=["asr_faster_whisper"]
                + (["whisperx_alignment"] if alignment_status == AlignmentStatus.ALIGNED else []),
                text_candidate=" ".join(w.text for w in group),
                language=result.language,
                mean_word_probability=(
                    round(mean_probability, 4) if mean_probability is not None else None
                ),
                alignment_status=alignment_status,
                state=state,
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
