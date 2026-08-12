"""Manual audio review queue: every item explains WHY a human should listen.

Items carry exact annotation-clock windows plus a padded playback window and
optional local review WAV clips — enough data for the future Tauri UI to seek,
loop, and play at 0.5x/1x/2x.
"""

from __future__ import annotations

import logging
from fractions import Fraction
from pathlib import Path

from ..models.audio import (
    RESOLVED_SOURCE_VERIFICATION,
    AudioRegion,
    AudioRegionKind,
    AudioReviewItem,
    AudioReviewReason,
    AudioTimeline,
    AudioTransientCandidate,
    AudioVerificationStatus,
    BoundaryAudioEvidence,
    LanguageReviewStatus,
    ReviewPriority,
    SpeechRegion,
)
from .decode import DecodedWav, write_wav_slice
from .timeline import annotation_to_sample

logger = logging.getLogger(__name__)

PLAYBACK_PADDING = Fraction(1, 2)
#: A transient at least this loud (rise, dB) carries significant semantics.
SIGNIFICANT_TRANSIENT_RISE_DB = 20.0

_CRITICAL_REASONS = frozenset(
    {
        AudioReviewReason.MANDATORY_SOURCE_AUDIO_VERIFICATION,
        AudioReviewReason.POSSIBLE_OVERLAP,
        AudioReviewReason.SPEECH_AT_CLIP_START,
        AudioReviewReason.SPEECH_AT_CLIP_END,
    }
)
_HIGH_REASONS = frozenset(
    {
        AudioReviewReason.UNKNOWN_LANGUAGE,
        AudioReviewReason.LOW_LANGUAGE_CONFIDENCE,
        AudioReviewReason.MISSING_ALIGNMENT,
        AudioReviewReason.PARTIAL_ALIGNMENT,
        AudioReviewReason.AUDIO_WITHOUT_ASR_COVERAGE,
        AudioReviewReason.VOCAL_REVIEW_REQUIRED,
        AudioReviewReason.ASR_UNAVAILABLE,
    }
)


def _priority(reasons: list[AudioReviewReason]) -> ReviewPriority:
    """Structured priority — never hides evidence, only orders it (§20)."""
    rset = set(reasons)
    if rset & _CRITICAL_REASONS:
        return ReviewPriority.CRITICAL
    if rset & _HIGH_REASONS:
        return ReviewPriority.HIGH
    if rset & {
        AudioReviewReason.SHOT_BOUNDARY_AUDIO_CONTINUITY,
        AudioReviewReason.TRANSIENT_SEMANTICS_UNKNOWN,
    }:
        return ReviewPriority.NORMAL
    return ReviewPriority.LOW


def build_review_queue(
    timeline: AudioTimeline | None,
    speech_regions: list[SpeechRegion],
    uncovered: list[tuple[Fraction, Fraction]],
    transients: list[AudioTransientCandidate],
    boundary_evidence: list[BoundaryAudioEvidence],
    regions: list[AudioRegion],
    asr_unavailable: bool,
) -> list[AudioReviewItem]:
    items: list[AudioReviewItem] = []
    seq = 1

    audio_lo: Fraction | None = None
    audio_hi: Fraction | None = None
    if timeline is not None:
        audio_lo = timeline.annotation_audio_offset
        audio_hi = timeline.annotation_audio_offset + timeline.evidence_duration_seconds

    def _clamp(value: Fraction) -> Fraction:
        """Clamp a playback edge to available source-audio bounds (§18) so the
        future UI can never seek outside the media."""
        lo = audio_lo if audio_lo is not None else Fraction(0)
        result = max(lo, value)
        if audio_hi is not None:
            result = min(audio_hi, result)
        return result

    def add(
        reasons: list[AudioReviewReason],
        start: Fraction,
        end: Fraction,
        text: str | None = None,
        refs: list[str] | None = None,
        notes: list[str] | None = None,
        priority: ReviewPriority | None = None,
    ) -> None:
        nonlocal seq
        p_start = _clamp(start - PLAYBACK_PADDING)
        p_end = _clamp(end + PLAYBACK_PADDING)
        items.append(
            AudioReviewItem(
                item_id=f"areview_{seq:04d}",
                reasons=reasons,
                priority=priority or _priority(reasons),
                start_exact=start,
                end_exact=end,
                playback_start=p_start,
                playback_end=p_end,
                asr_text_candidate=text,
                evidence_refs=refs or [],
                notes=notes or [],
            )
        )
        seq += 1

    # One review item per speech act, carrying ALL its reasons (§3/§20). A machine
    # speech region that is not human-verified ALWAYS produces a mandatory-listen
    # item — a clean aligned result never bypasses it. Language review is decided
    # PER region, never globally suppressed once one language issue exists (§19).
    for region in speech_regions:
        # A human has listened (verified / corrected / rejected) → resolved, no
        # mandatory listen. Only UNVERIFIED speech demands review.
        if region.source_verification_status in RESOLVED_SOURCE_VERIFICATION:
            continue
        reasons = list(region.review_reasons)
        if (
            region.language is not None
            and region.language.language_review_status
            in (LanguageReviewStatus.REVIEW_REQUIRED, LanguageReviewStatus.UNKNOWN)
        ):
            language_reason = (
                AudioReviewReason.UNKNOWN_LANGUAGE
                if region.language.language_candidate is None
                else AudioReviewReason.LOW_LANGUAGE_CONFIDENCE
            )
            if language_reason not in reasons:
                reasons.append(language_reason)
        add(
            reasons,
            region.start_exact,
            region.end_exact,
            text=region.text_candidate,
            refs=[region.region_id],
        )

    for start, end in uncovered:
        reasons = [AudioReviewReason.AUDIO_WITHOUT_ASR_COVERAGE]
        if asr_unavailable:
            reasons.append(AudioReviewReason.ASR_UNAVAILABLE)
        # Possible vocal material without trustworthy text (singing defense):
        # tonal sustained audio overlapping the uncovered window stays visible.
        if any(
            r.kind == AudioRegionKind.SUSTAINED_TONAL_AUDIO
            and r.end_annotation_time > start
            and r.start_annotation_time < end
            for r in regions
        ):
            reasons.append(AudioReviewReason.VOCAL_REVIEW_REQUIRED)
        add(reasons, start, end,
            notes=["Meaningful audio energy with no ASR coverage; do not assume silence."])

    for transient in transients:
        priority = (
            ReviewPriority.NORMAL
            if transient.rise_db >= SIGNIFICANT_TRANSIENT_RISE_DB
            else ReviewPriority.LOW
        )
        add(
            [AudioReviewReason.TRANSIENT_SEMANTICS_UNKNOWN],
            transient.start_annotation_time,
            transient.end_annotation_time,
            refs=[transient.candidate_id],
            notes=[
                f"Transient peak {transient.peak_dbfs:.1f} dBFS, "
                f"rise {transient.rise_db:.1f} dB."
            ],
            priority=priority,
        )

    for boundary in boundary_evidence:
        if boundary.audio_verification_status == AudioVerificationStatus.CROSSING_REVIEW_REQUIRED:
            add(
                [AudioReviewReason.SHOT_BOUNDARY_AUDIO_CONTINUITY],
                boundary.visual_boundary_exact - Fraction(1, 2),
                boundary.visual_boundary_exact + Fraction(1, 2),
                refs=[boundary.boundary_candidate_id],
                notes=boundary.notes,
            )

    return items


def render_review_clips(
    items: list[AudioReviewItem],
    source: DecodedWav,
    timeline: AudioTimeline,
    clips_dir: Path,
) -> list[Path]:
    """Exact-sample review WAVs with context padding, never outside the source."""
    written: list[Path] = []
    for index, item in enumerate(items, start=1):
        start_sample = annotation_to_sample(timeline, item.playback_start)
        end_sample = annotation_to_sample(timeline, item.playback_end)
        if end_sample <= start_sample:
            continue
        path = clips_dir / f"review_{index:04d}.wav"
        write_wav_slice(source, path, start_sample, end_sample)
        item.review_clip = f"review_clips/{path.name}"
        written.append(path)
    return written
