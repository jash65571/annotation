"""Deterministic audio regions: silence, active audio, transients, signal classes.

These are SIGNAL classes, never semantic claims — silence regions are internal
evidence only (never "No speech" lines), transients are TRANSIENT_CANDIDATE
until semantic verification, tonal/noise classes describe the spectrum only.
"""

from __future__ import annotations

import logging
from fractions import Fraction

from ..models.audio import (
    AudioEnergyBin,
    AudioRegion,
    AudioRegionKind,
    AudioTimeline,
    AudioTransientCandidate,
)
from .timeline import sample_to_annotation

logger = logging.getLogger(__name__)

#: Conservative default thresholds (docs/07).
SILENCE_DBFS = -55.0
ACTIVE_DBFS = -40.0
LOW_LEVEL_DBFS = -50.0
MIN_REGION_BINS = 30  # 300 ms
TRANSIENT_RISE_DB = 14.0
TONAL_FLATNESS_MAX = 0.02
NOISE_FLATNESS_MIN = 0.3


def _mean_dbfs(bins: list[AudioEnergyBin]) -> float:
    return round(sum(b.dbfs for b in bins) / len(bins), 3) if bins else -120.0


def _runs(flags: list[bool], min_len: int) -> list[tuple[int, int]]:
    """(start, end-exclusive) index runs of True at least min_len long."""
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for i, flag in enumerate([*flags, False]):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            if i - start >= min_len:
                runs.append((start, i))
            start = None
    return runs


def detect_regions(
    bins: list[AudioEnergyBin], timeline: AudioTimeline
) -> list[AudioRegion]:
    regions: list[AudioRegion] = []
    seq = 1

    def add(kind: AudioRegionKind, start_bin: int, end_bin: int) -> None:
        nonlocal seq
        chunk = bins[start_bin:end_bin]
        start_sample = bins[start_bin].start_sample
        end_sample = bins[end_bin - 1].end_sample
        regions.append(
            AudioRegion(
                region_id=f"aregion_{seq:04d}",
                kind=kind,
                start_sample=start_sample,
                end_sample=end_sample,
                start_annotation_time=sample_to_annotation(timeline, start_sample),
                end_annotation_time=sample_to_annotation(timeline, end_sample),
                mean_dbfs=_mean_dbfs(chunk),
            )
        )
        seq += 1

    silence_flags = [b.dbfs <= SILENCE_DBFS for b in bins]
    for start, end in _runs(silence_flags, MIN_REGION_BINS):
        add(AudioRegionKind.SILENCE_CANDIDATE, start, end)

    active_flags = [b.dbfs >= ACTIVE_DBFS for b in bins]
    for start, end in _runs(active_flags, MIN_REGION_BINS):
        chunk = bins[start:end]
        flatness = sorted(b.spectral_flatness for b in chunk)[len(chunk) // 2]
        add(AudioRegionKind.ACTIVE_AUDIO, start, end)
        if flatness <= TONAL_FLATNESS_MAX:
            add(AudioRegionKind.SUSTAINED_TONAL_AUDIO, start, end)
        elif flatness >= NOISE_FLATNESS_MIN:
            add(AudioRegionKind.BROADBAND_NOISE, start, end)

    background_flags = [
        SILENCE_DBFS < b.dbfs < LOW_LEVEL_DBFS for b in bins
    ]
    for start, end in _runs(background_flags, MIN_REGION_BINS * 2):
        add(AudioRegionKind.LOW_LEVEL_BACKGROUND_AUDIO, start, end)

    return regions


def detect_transients(
    bins: list[AudioEnergyBin], timeline: AudioTimeline
) -> list[AudioTransientCandidate]:
    """High-energy onsets: dBFS rise vs the trailing local median."""
    transients: list[AudioTransientCandidate] = []
    seq = 1
    window = 20  # 200 ms of trailing context
    i = 0
    while i < len(bins):
        item = bins[i]
        history = bins[max(0, i - window) : i]
        if not history:
            i += 1
            continue
        baseline = sorted(b.dbfs for b in history)[len(history) // 2]
        rise = item.dbfs - baseline
        if rise >= TRANSIENT_RISE_DB and item.dbfs > SILENCE_DBFS:
            # Extend through the elevated run to find peak and end.
            j = i
            peak_index = i
            while j + 1 < len(bins) and bins[j + 1].dbfs > baseline + TRANSIENT_RISE_DB / 2:
                j += 1
                if bins[j].dbfs > bins[peak_index].dbfs:
                    peak_index = j
            start_sample = item.start_sample
            end_sample = bins[j].end_sample
            peak_sample = bins[peak_index].start_sample
            transients.append(
                AudioTransientCandidate(
                    candidate_id=f"transient_{seq:04d}",
                    start_sample=start_sample,
                    peak_sample=peak_sample,
                    end_sample=end_sample,
                    start_annotation_time=sample_to_annotation(timeline, start_sample),
                    peak_annotation_time=sample_to_annotation(timeline, peak_sample),
                    end_annotation_time=sample_to_annotation(timeline, end_sample),
                    peak_dbfs=bins[peak_index].dbfs,
                    rise_db=round(rise, 2),
                )
            )
            seq += 1
            i = j + 1
        else:
            i += 1
    return transients


def active_coverage(
    regions: list[AudioRegion],
) -> list[tuple[Fraction, Fraction]]:
    """Annotation-time windows of ACTIVE_AUDIO regions."""
    return [
        (r.start_annotation_time, r.end_annotation_time)
        for r in regions
        if r.kind == AudioRegionKind.ACTIVE_AUDIO
    ]
