"""Waveform, spectrogram and energy PNG rendering (cv2, no extra dependencies).

All X axes use the ANNOTATION clock — never raw source PTS — matching the
clock the Manuscript caption uses. Rendering parameters are deterministic and
documented here:

- waveform: per-column min/max envelope of the mono mix, 1600x400 px.
- spectrogram: STFT with 1024-sample Hann window, hop 256, magnitude in dB
  (floor -90 dB), linear frequency axis, MAGMA colormap, 1600x400 px.
- energy: dBFS curve of the 10 ms bins, silence threshold line drawn.
"""

from __future__ import annotations

import itertools
import logging
from fractions import Fraction
from pathlib import Path

import cv2
import numpy as np
import numpy.typing as npt

from ..models.audio import AudioEnergyBin, AudioTimeline
from .timeline import sample_to_annotation

logger = logging.getLogger(__name__)

WIDTH = 1600
HEIGHT = 400
MARGIN_BOTTOM = 28
SPEC_WINDOW = 1024
SPEC_HOP = 256
SPEC_DB_FLOOR = -90.0


def _time_axis(image: npt.NDArray[np.uint8], start: Fraction, end: Fraction) -> None:
    """Burn annotation-clock second ticks onto the bottom margin."""
    height, width = image.shape[:2]
    span = float(end - start)
    if span <= 0:
        return
    cv2.rectangle(image, (0, height - MARGIN_BOTTOM), (width, height), (20, 20, 20), -1)
    first_tick = int(np.ceil(float(start)))
    for second in range(first_tick, int(np.floor(float(end))) + 1):
        x = int((second - float(start)) / span * width)
        cv2.line(image, (x, height - MARGIN_BOTTOM), (x, height - MARGIN_BOTTOM + 6),
                 (200, 200, 200), 1)
        cv2.putText(image, f"{second}s", (x + 3, height - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (220, 220, 220), 1, cv2.LINE_AA)


def render_waveform(
    mono: npt.NDArray[np.float64], timeline: AudioTimeline, out_path: Path
) -> Path:
    plot_height = HEIGHT - MARGIN_BOTTOM
    image = np.full((HEIGHT, WIDTH, 3), 12, dtype=np.uint8)
    n = len(mono)
    if n > 0:
        per_column = max(1, n // WIDTH)
        mid = plot_height // 2
        for x in range(WIDTH):
            start = x * per_column
            end = min(n, start + per_column)
            if start >= n:
                break
            chunk = mono[start:end]
            low = int(mid - float(chunk.max()) * (mid - 4))
            high = int(mid - float(chunk.min()) * (mid - 4))
            cv2.line(image, (x, low), (x, high), (90, 200, 120), 1)
        cv2.line(image, (0, mid), (WIDTH, mid), (60, 60, 60), 1)
    start_t = sample_to_annotation(timeline, 0)
    end_t = sample_to_annotation(timeline, n)
    _time_axis(image, start_t, end_t)
    cv2.imwrite(str(out_path), image)
    return out_path


def render_spectrogram(
    mono: npt.NDArray[np.float64], timeline: AudioTimeline, out_path: Path
) -> Path:
    n = len(mono)
    if n < SPEC_WINDOW:
        columns = np.zeros((SPEC_WINDOW // 2 + 1, 1))
    else:
        window = np.hanning(SPEC_WINDOW)
        frames = []
        for start in range(0, n - SPEC_WINDOW + 1, SPEC_HOP):
            segment = mono[start : start + SPEC_WINDOW] * window
            frames.append(np.abs(np.fft.rfft(segment)))
        columns = np.array(frames).T  # (freq_bins, time_frames)
    db = 20.0 * np.log10(columns + 1e-9)
    db = np.clip(db, SPEC_DB_FLOOR, 0.0)
    normalized = ((db - SPEC_DB_FLOOR) / -SPEC_DB_FLOOR * 255.0).astype(np.uint8)
    normalized = np.flipud(normalized)  # low frequencies at the bottom
    plot = cv2.resize(normalized, (WIDTH, HEIGHT - MARGIN_BOTTOM),
                      interpolation=cv2.INTER_AREA)
    colored = cv2.applyColorMap(plot, cv2.COLORMAP_MAGMA)
    image = np.full((HEIGHT, WIDTH, 3), 12, dtype=np.uint8)
    image[: HEIGHT - MARGIN_BOTTOM] = colored
    nyquist = timeline.evidence_sample_rate // 2
    cv2.putText(image, f"0-{nyquist} Hz linear, {SPEC_WINDOW}pt Hann hop {SPEC_HOP}",
                (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (230, 230, 230), 1, cv2.LINE_AA)
    _time_axis(image, sample_to_annotation(timeline, 0), sample_to_annotation(timeline, n))
    cv2.imwrite(str(out_path), image)
    return out_path


def render_energy(
    bins: list[AudioEnergyBin],
    timeline: AudioTimeline,
    out_path: Path,
    silence_threshold_dbfs: float,
) -> Path:
    plot_height = HEIGHT - MARGIN_BOTTOM
    image = np.full((HEIGHT, WIDTH, 3), 12, dtype=np.uint8)
    if bins:
        db_min, db_max = -80.0, 0.0

        def y_for(dbfs: float) -> int:
            clamped = min(max(dbfs, db_min), db_max)
            return int((db_max - clamped) / (db_max - db_min) * (plot_height - 8)) + 4

        threshold_y = y_for(silence_threshold_dbfs)
        cv2.line(image, (0, threshold_y), (WIDTH, threshold_y), (80, 80, 200), 1)
        cv2.putText(image, f"silence {silence_threshold_dbfs:.0f} dBFS",
                    (6, threshold_y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                    (120, 120, 230), 1, cv2.LINE_AA)
        points = []
        for i, item in enumerate(bins):
            x = int(i / max(1, len(bins) - 1) * (WIDTH - 1))
            points.append((x, y_for(item.dbfs)))
        for a, b in itertools.pairwise(points):
            cv2.line(image, a, b, (120, 220, 220), 1)
        start_t = bins[0].start_annotation_time
        end_t = bins[-1].end_annotation_time
        _time_axis(image, start_t, end_t)
    else:
        _time_axis(image, sample_to_annotation(timeline, 0), sample_to_annotation(timeline, 0))
    cv2.imwrite(str(out_path), image)
    return out_path
