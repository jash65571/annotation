"""Deterministic 10 ms audio energy metrics.

Bin size is exactly ``sample_rate / 100`` samples when the rate is divisible
by 100 (true for 44.1 kHz → 441 and 48 kHz → 480); otherwise ``floor`` is used
and the actual bin duration (still sample-exact) is recorded.

Metric purposes / blind spots (docs/07):
- RMS/dBFS: overall energy — cannot distinguish speech from music/noise.
- peak: transient headroom — fooled by single-sample clicks.
- zero-crossing rate: crude voiced/unvoiced + noisiness cue — fooled by tonal
  high frequencies.
- spectral centroid: brightness — fooled by broadband noise.
- spectral flatness: tonal (≈0) vs noise-like (≈1) — fooled by dense music.
"""

from __future__ import annotations

import logging
from fractions import Fraction

import numpy as np
import numpy.typing as npt

from ..models.audio import AudioEnergyBin, AudioTimeline
from .timeline import sample_to_annotation

logger = logging.getLogger(__name__)

_DBFS_FLOOR = -120.0


def _dbfs(rms: float) -> float:
    if rms <= 0.0:
        return _DBFS_FLOOR
    return max(_DBFS_FLOOR, float(20.0 * np.log10(rms)))


def compute_energy_bins(
    mono: npt.NDArray[np.float64], timeline: AudioTimeline
) -> list[AudioEnergyBin]:
    rate = timeline.evidence_sample_rate
    bin_samples = rate // 100
    if bin_samples == 0:
        return []
    n_bins = len(mono) // bin_samples
    bins: list[AudioEnergyBin] = []
    for i in range(n_bins):
        start = i * bin_samples
        end = start + bin_samples
        chunk = mono[start:end]
        rms = float(np.sqrt(np.mean(chunk**2)))
        peak = float(np.max(np.abs(chunk))) if len(chunk) else 0.0
        signs = np.signbit(chunk)
        zcr = float(np.mean(signs[1:] != signs[:-1])) if len(chunk) > 1 else 0.0
        spectrum = np.abs(np.fft.rfft(chunk))
        power = spectrum**2
        total = float(power.sum())
        if total > 0:
            freqs = np.fft.rfftfreq(len(chunk), d=1.0 / rate)
            centroid = float((freqs * power).sum() / total)
            geometric = float(np.exp(np.mean(np.log(power + 1e-20))))
            arithmetic = float(np.mean(power))
            flatness = geometric / arithmetic if arithmetic > 0 else 0.0
        else:
            centroid = 0.0
            flatness = 0.0
        bins.append(
            AudioEnergyBin(
                bin_index=i,
                start_sample=start,
                end_sample=end,
                start_annotation_time=sample_to_annotation(timeline, start),
                end_annotation_time=sample_to_annotation(timeline, end),
                rms=round(rms, 8),
                peak=round(peak, 8),
                dbfs=round(_dbfs(rms), 3),
                zero_crossing_rate=round(zcr, 5),
                spectral_centroid_hz=round(centroid, 2),
                spectral_flatness=round(min(flatness, 1.0), 6),
            )
        )
    return bins


def bin_duration_seconds(timeline: AudioTimeline) -> Fraction:
    """Exact duration of one energy bin."""
    return Fraction(timeline.evidence_sample_rate // 100, timeline.evidence_sample_rate)
