"""Shot-boundary audio continuity: evidence for every SUPPORTED visual boundary.

Resolves Phase 2's ``audio_verification_required`` flag at the EVIDENCE level
only — the final transition selector is never changed automatically, and L-cut
vs J-cut is never decided from waveform continuity alone (that needs semantic
source relation, P3-BOUNDARY-002).

Continuity is EVIDENCE-GRADED, conservatively (§11/§13):

- Audio energy present on BOTH sides proves only ``AUDIO_PRESENT_BOTH_SIDES`` —
  NOT that one sound source crosses the cut. Two equal-energy but different
  sources (440 Hz → 880 Hz, or a tone → broadband noise at equal RMS) must NOT
  read CONTINUOUS: the spectrum moves even when energy does not.
- CONTINUOUS requires MEANINGFUL continuity evidence: an actual vocal word that
  spans the boundary, or strong local spectral continuity (centroid + flatness
  stable, no energy step, no silence gap).
- Audio on both sides whose source continuity cannot be proven is UNCERTAIN,
  never CONTINUOUS.
"""

from __future__ import annotations

import logging
from fractions import Fraction

from ..media.clock import AnnotationClock
from ..models.audio import (
    ASRSegment,
    AudioEnergyBin,
    AudioRegion,
    AudioRegionKind,
    AudioTimeline,
    AudioVerificationStatus,
    BestWord,
    BoundaryAudioEvidence,
    BoundaryAudioStatus,
    SpeechRegion,
)
from ..models.shot_truth import BoundaryCandidate, CandidateStatus

logger = logging.getLogger(__name__)

WINDOW_SECONDS = Fraction(1, 2)
SILENT_DBFS = -55.0
#: A dBFS step this large across the boundary is an energy discontinuity.
ENERGY_STEP_DB = 20.0
#: For spectral continuity we also require the energy to be roughly stable.
ENERGY_CONTINUITY_DB = 6.0
#: Combined spectral-change score at or below this reads as spectral continuity.
#: Score = relative centroid shift + absolute flatness shift; a pure tone that
#: continues unchanged scores ~0, while a 440→880 Hz or tone→noise switch (which
#: preserves energy) scores well above this because centroid and/or flatness
#: move. Chosen from the physics of the signal, not tuned to fixtures (§14).
SPECTRAL_CONTINUITY_MAX = 0.20


def _window_energy(
    bins: list[AudioEnergyBin], start: Fraction, end: Fraction
) -> float | None:
    selected = [
        b.dbfs
        for b in bins
        if b.end_annotation_time > start and b.start_annotation_time < end
    ]
    if not selected:
        return None
    return round(sum(selected) / len(selected), 3)


def _window_spectral(
    bins: list[AudioEnergyBin], start: Fraction, end: Fraction
) -> tuple[float | None, float | None]:
    """Mean spectral centroid (Hz) and flatness over a window, ignoring bins
    that are effectively silent (their spectrum is noise-floor artefact)."""
    selected = [
        b
        for b in bins
        if b.end_annotation_time > start
        and b.start_annotation_time < end
        and b.dbfs > SILENT_DBFS
    ]
    if not selected:
        return None, None
    centroid = round(sum(b.spectral_centroid_hz for b in selected) / len(selected), 2)
    flatness = round(sum(b.spectral_flatness for b in selected) / len(selected), 6)
    return centroid, flatness


def _spectral_change_score(
    cb: float | None, ca: float | None, fb: float | None, fa: float | None
) -> float | None:
    if cb is None or ca is None or fb is None or fa is None:
        return None
    rel_centroid = abs(cb - ca) / max(cb, ca, 1.0)
    flatness_shift = abs(fb - fa)
    return round(rel_centroid + flatness_shift, 6)


def _no_audio_evidence(
    candidate_id: str, boundary: Fraction
) -> BoundaryAudioEvidence:
    return BoundaryAudioEvidence(
        boundary_candidate_id=candidate_id,
        visual_boundary_exact=boundary,
        window_seconds=float(WINDOW_SECONDS),
        audio_present_before=False,
        audio_present_after=False,
        silence_spans_boundary=False,
        speech_region_spans_boundary=False,
        asr_segment_spans_boundary=False,
        asr_word_spans_boundary=False,
        energy_before_dbfs=None,
        energy_after_dbfs=None,
        energy_delta_db=None,
        continuity_evidence_codes=[],
        audio_continuity_status=BoundaryAudioStatus.NO_AUDIO,
        audio_verification_status=AudioVerificationStatus.UNAVAILABLE,
    )


def analyze_boundary_audio(
    candidates: list[BoundaryCandidate],
    clock: AnnotationClock,
    timeline: AudioTimeline | None,
    bins: list[AudioEnergyBin],
    regions: list[AudioRegion],
    speech_regions: list[SpeechRegion],
    best_words: list[BestWord],
    asr_segments: list[ASRSegment],
) -> list[BoundaryAudioEvidence]:
    evidence: list[BoundaryAudioEvidence] = []
    for candidate in candidates:
        if candidate.status != CandidateStatus.SUPPORTED:
            continue
        if candidate.boundary_time_exact is None:
            continue
        boundary = clock.to_annotation(candidate.boundary_time_exact)

        if timeline is None:
            evidence.append(_no_audio_evidence(candidate.candidate_id, boundary))
            continue

        before = _window_energy(bins, boundary - WINDOW_SECONDS, boundary)
        after = _window_energy(bins, boundary, boundary + WINDOW_SECONDS)
        audio_present_before = before is not None and before > SILENT_DBFS
        audio_present_after = after is not None and after > SILENT_DBFS
        energy_delta = (
            round(abs(before - after), 3)
            if before is not None and after is not None
            else None
        )

        silence_spans = bool(
            any(
                r.kind == AudioRegionKind.SILENCE_CANDIDATE
                and r.start_annotation_time <= boundary <= r.end_annotation_time
                for r in regions
            )
        )
        # Three DISTINCT crossing facts — each computed from its own evidence,
        # never cloned from another (§12).
        speech_region_spans = any(
            s.start_exact < boundary < s.end_exact for s in speech_regions
        )
        asr_segment_spans = any(
            s.start_annotation_time is not None
            and s.end_annotation_time is not None
            and s.start_annotation_time < boundary < s.end_annotation_time
            for s in asr_segments
        )
        asr_word_spans = any(
            w.start_annotation_time < boundary < w.end_annotation_time
            for w in best_words
        )

        centroid_before, sf_before = _window_spectral(
            bins, boundary - WINDOW_SECONDS, boundary
        )
        centroid_after, sf_after = _window_spectral(
            bins, boundary, boundary + WINDOW_SECONDS
        )
        change_score = _spectral_change_score(
            centroid_before, centroid_after, sf_before, sf_after
        )

        codes: list[str] = []
        notes: list[str] = []

        if before is None or after is None:
            status = BoundaryAudioStatus.NO_AUDIO
            verification = AudioVerificationStatus.UNAVAILABLE
        elif silence_spans and not audio_present_before and not audio_present_after:
            status = BoundaryAudioStatus.DISCONTINUOUS
            verification = AudioVerificationStatus.CHECKED_NO_CROSSING
            codes.append("SILENCE_SPANS_BOUNDARY")
            notes.append("Silence spans the visual boundary; no source crosses.")
        elif asr_word_spans:
            status = BoundaryAudioStatus.CONTINUOUS
            verification = AudioVerificationStatus.CROSSING_REVIEW_REQUIRED
            codes.append("ASR_WORD_SPANS_BOUNDARY")
            notes.append(
                "A machine speech word spans the visual edit. An overlapping "
                "audio/video transition needs semantic source relation — not "
                "decidable from continuity alone (no automatic transition type)."
            )
        elif not audio_present_before or not audio_present_after:
            status = BoundaryAudioStatus.DISCONTINUOUS
            verification = AudioVerificationStatus.CHECKED_NO_CROSSING
            codes.append("AUDIO_ONE_SIDE_ONLY")
            notes.append("Audible energy on only one side of the boundary.")
        elif energy_delta is not None and energy_delta >= ENERGY_STEP_DB:
            status = BoundaryAudioStatus.DISCONTINUOUS
            verification = AudioVerificationStatus.CHECKED_NO_CROSSING
            codes.append("ENERGY_STEP")
            notes.append(f"Energy steps {before:.1f} -> {after:.1f} dBFS across the boundary.")
        elif (
            change_score is not None
            and change_score <= SPECTRAL_CONTINUITY_MAX
            and energy_delta is not None
            and energy_delta <= ENERGY_CONTINUITY_DB
        ):
            status = BoundaryAudioStatus.CONTINUOUS
            verification = AudioVerificationStatus.CROSSING_REVIEW_REQUIRED
            codes.append("SPECTRAL_CONTINUITY")
            codes.append("AUDIO_PRESENT_BOTH_SIDES")
            notes.append(
                "Stable spectrum and energy across the boundary suggest one "
                "sound source continues; source/semantic relation still requires "
                "review (no automatic overlapping-transition type)."
            )
        else:
            status = BoundaryAudioStatus.UNCERTAIN
            verification = AudioVerificationStatus.CROSSING_REVIEW_REQUIRED
            codes.append("AUDIO_PRESENT_BOTH_SIDES")
            notes.append(
                "Audio present on both sides, but the spectrum/energy shift is "
                "consistent with a source SWITCH — continuity is not proven."
            )

        if speech_region_spans and not asr_word_spans:
            codes.append("SPEECH_REGION_SPANS_NO_WORD")
        if asr_segment_spans:
            codes.append("ASR_SEGMENT_SPANS_BOUNDARY")

        evidence.append(
            BoundaryAudioEvidence(
                boundary_candidate_id=candidate.candidate_id,
                visual_boundary_exact=boundary,
                window_seconds=float(WINDOW_SECONDS),
                audio_present_before=audio_present_before,
                audio_present_after=audio_present_after,
                silence_spans_boundary=silence_spans,
                speech_region_spans_boundary=speech_region_spans,
                asr_segment_spans_boundary=asr_segment_spans,
                asr_word_spans_boundary=asr_word_spans,
                energy_before_dbfs=before,
                energy_after_dbfs=after,
                energy_delta_db=energy_delta,
                spectral_centroid_before=centroid_before,
                spectral_centroid_after=centroid_after,
                spectral_flatness_before=sf_before,
                spectral_flatness_after=sf_after,
                spectral_change_score=change_score,
                continuity_evidence_codes=codes,
                audio_continuity_status=status,
                audio_verification_status=verification,
                notes=notes,
            )
        )
    return evidence
