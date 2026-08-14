"""Measured prosody: the evidence path for speech delivery. Stdlib only.

The source of truth (§10 Rule 4) requires every speech line to carry a
*supported audible tone*, and the Aug 2026 evaluator feedback names missing
tone, pitch and pace as a scoring failure. A vision model given frames and
transcript text has no basis for any of that — so an earlier pass "solved" the
problem by forbidding tone outright. That traded an invention risk for a
guaranteed rule violation.

The correct fix is to measure the delivery and hand it over as fact:

  - **loudness** — span RMS against the clip's own speech median, so a shout and
    a murmur are relative to this speaker in this recording, not to an absolute
    scale that varies with mic gain;
  - **pitch** — fundamental frequency by autocorrelation over the voiced band
    (80-400 Hz), reported in Hz and as a band relative to the same speaker's
    other spans;
  - **pace** — words per second from word-level ASR timestamps.

Every measure has an ``unresolved`` state. Unvoiced, noisy, or too-short audio
yields "unresolved" rather than a guess, and the prompt tells the model to omit
what is unresolved instead of inventing it.
"""

from __future__ import annotations

import array
import wave
from dataclasses import dataclass
from pathlib import Path

from .transcribe import Transcript

#: Human voiced fundamental frequency, in Hz.
F0_MIN, F0_MAX = 80.0, 400.0
#: Autocorrelation peak below this is not periodic enough to call voiced.
VOICING_THRESHOLD = 0.30
#: Shortest span worth measuring.
MIN_SPAN = 0.20

UNRESOLVED = "unresolved"


@dataclass(frozen=True)
class Prosody:
    start: float
    end: float
    loudness: str
    pitch: str
    pace: str
    #: Raw measurements, kept so a human can audit the classification.
    f0_hz: float | None = None
    words_per_second: float | None = None

    def describe(self) -> str:
        parts: list[str] = []
        if self.loudness != UNRESOLVED:
            parts.append(f"loudness={self.loudness}")
        if self.pitch != UNRESOLVED:
            hz = f" ({self.f0_hz:.0f} Hz)" if self.f0_hz else ""
            parts.append(f"pitch={self.pitch}{hz}")
        if self.pace != UNRESOLVED:
            wps = f" ({self.words_per_second:.1f} words/s)" if self.words_per_second else ""
            parts.append(f"pace={self.pace}{wps}")
        measured = ", ".join(parts) if parts else "no delivery measurement possible"
        return f"{self.start:.1f}s-{self.end:.1f}s: {measured}"

    @property
    def all_unresolved(self) -> bool:
        return (
            self.loudness == UNRESOLVED
            and self.pitch == UNRESOLVED
            and self.pace == UNRESOLVED
        )


def _read_mono(wav_path: Path) -> tuple[array.array[int], int]:
    with wave.open(str(wav_path), "rb") as w:
        sr, ch, sw, n = w.getframerate(), w.getnchannels(), w.getsampwidth(), w.getnframes()
        raw = w.readframes(n)
    if sw != 2:
        return array.array("h"), sr
    samples = array.array("h", raw)
    if ch > 1:
        samples = samples[::ch]
    return samples, sr


def _rms(chunk: array.array[int]) -> float:
    if not chunk:
        return 0.0
    return float((sum(s * s for s in chunk) / len(chunk)) ** 0.5)


def _estimate_f0(chunk: array.array[int], sr: int) -> float | None:
    """Fundamental frequency by normalized autocorrelation, or None if unvoiced.

    Only the voiced band is searched, and a weak peak returns None — speech
    over music, whispering and pure noise must not produce a pitch number.
    """
    n = len(chunk)
    if n < sr // 20:  # need at least ~50 ms
        return None
    mean = sum(chunk) / n
    dev = [s - mean for s in chunk]
    energy = sum(d * d for d in dev)
    if energy <= 0:
        return None
    lag_min = max(1, int(sr / F0_MAX))
    lag_max = min(int(sr / F0_MIN), n // 2)
    if lag_max <= lag_min:
        return None
    best_lag, best_score = 0, 0.0
    for lag in range(lag_min, lag_max + 1):
        acc = 0.0
        for i in range(lag, n):
            acc += dev[i] * dev[i - lag]
        score = acc / energy
        if score > best_score:
            best_score, best_lag = score, lag
    if best_lag == 0 or best_score < VOICING_THRESHOLD:
        return None
    return sr / best_lag


def _downsample(chunk: array.array[int], factor: int) -> array.array[int]:
    return chunk if factor <= 1 else chunk[::factor]


def _median(values: list[float]) -> float:
    """True median, interpolated for an even count.

    ``sorted(v)[len(v) // 2]`` is the upper of the two middle values, so with
    exactly two spans the louder one sits *at* the median and reads as "level"
    no matter how much louder it is. With two spans — a common case in a short
    clip — that silently disabled the whole loudness comparison.
    """
    if not values:
        return 0.0
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2


def analyze(
    wav_path: Path,
    transcript: Transcript,
    spans: list[tuple[float, float]],
) -> list[Prosody]:
    """Measure delivery for each speech span. Never raises on empty input."""
    if not spans:
        return []
    samples, sr = _read_mono(wav_path)
    if not samples or sr <= 0:
        return [
            Prosody(s, e, UNRESOLVED, UNRESOLVED, UNRESOLVED) for s, e in spans
        ]

    def slice_of(start: float, end: float) -> array.array[int]:
        a, b = max(0, int(start * sr)), min(len(samples), int(end * sr))
        return samples[a:b] if b > a else array.array("h")

    # Loudness is relative to this clip's own speech, so mic gain cancels out.
    energies = [
        _rms(slice_of(s, e)) for s, e in spans if e - s >= MIN_SPAN
    ]
    median_rms = _median([x for x in energies if x > 0])

    # Pitch band is likewise relative: the same voice measured against itself.
    pitches: list[float] = []
    raw_f0: dict[tuple[float, float], float | None] = {}
    factor = max(1, sr // 8000)  # autocorrelation is fine at 8 kHz and 2x faster
    for s, e in spans:
        if e - s < MIN_SPAN:
            raw_f0[(s, e)] = None
            continue
        mid = (s + e) / 2
        window = slice_of(max(s, mid - 0.25), min(e, mid + 0.25))
        f0 = _estimate_f0(_downsample(window, factor), sr // factor)
        raw_f0[(s, e)] = f0
        if f0:
            pitches.append(f0)
    median_f0 = _median(pitches)

    out: list[Prosody] = []
    for s, e in spans:
        duration = e - s
        loudness = UNRESOLVED
        if duration >= MIN_SPAN and median_rms > 0:
            rms = _rms(slice_of(s, e))
            if rms > 0:
                ratio = rms / median_rms
                if ratio >= 1.6:
                    loudness = "louder than the rest of the speech"
                elif ratio <= 0.6:
                    loudness = "quieter than the rest of the speech"
                else:
                    loudness = "level with the rest of the speech"

        f0 = raw_f0.get((s, e))
        pitch = UNRESOLVED
        if f0 and median_f0 > 0:
            ratio = f0 / median_f0
            if ratio >= 1.15:
                pitch = "higher than this speaker's norm"
            elif ratio <= 0.87:
                pitch = "lower than this speaker's norm"
            else:
                pitch = "at this speaker's norm"

        pace = UNRESOLVED
        wps: float | None = None
        if transcript.words and duration >= MIN_SPAN:
            count = sum(
                1 for w in transcript.words if s <= (w.start + w.end) / 2 < e
            )
            if count:
                wps = count / duration
                if wps >= 3.6:
                    pace = "fast"
                elif wps <= 1.8:
                    pace = "slow"
                else:
                    pace = "measured"

        out.append(Prosody(
            round(s, 1), round(e, 1), loudness, pitch, pace,
            f0_hz=f0, words_per_second=wps,
        ))
    return out


def describe(items: list[Prosody]) -> str:
    return "\n".join(f"  {p.describe()}" for p in items)


def for_range(items: list[Prosody], start: float, end: float) -> list[Prosody]:
    """Prosody entries whose midpoint falls inside [start, end]."""
    return [p for p in items if start <= (p.start + p.end) / 2 < end]
