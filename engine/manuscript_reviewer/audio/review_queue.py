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
    AudioRegion,
    AudioRegionKind,
    AudioReviewItem,
    AudioReviewReason,
    AudioTimeline,
    AudioTransientCandidate,
    AudioVerificationStatus,
    BoundaryAudioEvidence,
    EvidenceState,
    LanguageReviewStatus,
    SpeechRegion,
)
from .decode import DecodedWav, write_wav_slice
from .timeline import annotation_to_sample

logger = logging.getLogger(__name__)

PLAYBACK_PADDING = Fraction(1, 2)


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

    def add(
        reasons: list[AudioReviewReason],
        start: Fraction,
        end: Fraction,
        text: str | None = None,
        refs: list[str] | None = None,
        notes: list[str] | None = None,
    ) -> None:
        nonlocal seq
        items.append(
            AudioReviewItem(
                item_id=f"areview_{seq:04d}",
                reasons=reasons,
                start_exact=start,
                end_exact=end,
                playback_start=max(Fraction(0), start - PLAYBACK_PADDING),
                playback_end=end + PLAYBACK_PADDING,
                asr_text_candidate=text,
                evidence_refs=refs or [],
                notes=notes or [],
            )
        )
        seq += 1

    for region in speech_regions:
        if region.state == EvidenceState.REVIEW_REQUIRED and region.review_reasons:
            add(
                region.review_reasons,
                region.start_exact,
                region.end_exact,
                text=region.text_candidate,
                refs=[region.region_id],
            )
        if (
            region.language is not None
            and region.language.language_review_status
            in (LanguageReviewStatus.REVIEW_REQUIRED, LanguageReviewStatus.UNKNOWN)
            and not any(
                r in (AudioReviewReason.UNKNOWN_LANGUAGE, AudioReviewReason.LOW_LANGUAGE_CONFIDENCE)
                for item in items
                for r in item.reasons
            )
        ):
            reason = (
                AudioReviewReason.UNKNOWN_LANGUAGE
                if region.language.language_candidate is None
                else AudioReviewReason.LOW_LANGUAGE_CONFIDENCE
            )
            add([reason], region.start_exact, region.end_exact,
                text=region.text_candidate, refs=[region.region_id])

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
        add(
            [AudioReviewReason.TRANSIENT_SEMANTICS_UNKNOWN],
            transient.start_annotation_time,
            transient.end_annotation_time,
            refs=[transient.candidate_id],
            notes=[
                f"Transient peak {transient.peak_dbfs:.1f} dBFS, "
                f"rise {transient.rise_db:.1f} dB."
            ],
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
