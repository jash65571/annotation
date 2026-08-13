"""Signal-measured audio timeline. Stdlib only — no numpy/scipy.

Motivation (verified failure): the vision passes were told to infer audio from
what is *visible*, which deleted real non-diegetic music and let hallucinated
narration cover the whole clip. This module measures the waveform itself:

  - RMS energy envelope  -> quiet vs. audible regions;
  - onset-autocorrelation rhythmicity -> "music-like" (steady beat) vs.
    "non-rhythmic" (crowd noise, laughter, ambience);
  - speech spans come from word-level ASR timestamps (transcribe.Word), which
    are far more accurate than whisper segment bounds.

The result is a list of labelled spans that are stated to the vision model as
measured FACTS, so it never invents audio and never deletes real music because
no performer is on screen.
"""

from __future__ import annotations

import array
import wave
from dataclasses import dataclass
from pathlib import Path

from .transcribe import Transcript

HOP = 0.05  # envelope hop in seconds


@dataclass(frozen=True)
class AudioSpan:
    start: float
    end: float
    #: 'speech' | 'music' | 'sound' | 'quiet'
    label: str

    def describe(self) -> str:
        names = {
            "speech": "speech (see verbatim transcript)",
            "music": "strongly rhythmic audio — music with a steady beat "
                     "(non-diegetic music needs NO visible source)",
            "sound": "non-rhythmic sound — crowd noise, laughter, ambience or "
                     "effects (audible, but not music with a steady beat and "
                     "not intelligible speech)",
            "quiet": "near-silence",
        }
        return f"{self.start:.1f}s-{self.end:.1f}s: {names[self.label]}"


def _envelope(wav_path: Path) -> tuple[list[float], float]:
    """Mono RMS energy per HOP seconds. Returns (envelope, duration)."""
    with wave.open(str(wav_path), "rb") as w:
        sr, ch, sw, n = w.getframerate(), w.getnchannels(), w.getsampwidth(), w.getnframes()
        raw = w.readframes(n)
    if sw != 2:  # pipeline always extracts pcm_s16le; anything else -> no analysis
        return [], n / sr if sr else 0.0
    samples = array.array("h", raw)
    if ch > 1:
        samples = samples[::ch]
    hop_n = max(1, int(sr * HOP))
    env: list[float] = []
    for i in range(0, len(samples), hop_n):
        chunk = samples[i:i + hop_n]
        if not chunk:
            break
        env.append((sum(s * s for s in chunk) / len(chunk)) ** 0.5)
    return env, len(samples) / sr


def _rhythmicity(env: list[float], lo_lag: float = 0.25, hi_lag: float = 1.0) -> float:
    """Peak normalized autocorrelation of onset strength at musical-beat lags.

    A steady beat repeats every 0.25-1.0s, producing a strong self-similar
    onset pattern; speech and crowd noise do not. Returns 0..1."""
    onsets = [max(0.0, env[i] - env[i - 1]) for i in range(1, len(env))]
    n = len(onsets)
    if n < 20:
        return 0.0
    mean = sum(onsets) / n
    dev = [o - mean for o in onsets]
    var = sum(d * d for d in dev)
    if var <= 0:
        return 0.0
    best = 0.0
    for lag in range(int(lo_lag / HOP), min(int(hi_lag / HOP) + 1, n // 2)):
        num = sum(dev[i] * dev[i - lag] for i in range(lag, n))
        best = max(best, num / var)
    return best


def _speech_spans(transcript: Transcript, gap: float = 0.6) -> list[tuple[float, float]]:
    """Merge word timestamps (reliable segments only, when words are absent fall
    back to reliable segment bounds) into speech spans, splitting at real pauses."""
    if transcript.words:
        points = [(w.start, w.end) for w in transcript.words]
    else:
        points = [(s.start, s.end) for s in transcript.reliable_segments]
    spans: list[tuple[float, float]] = []
    for s, e in sorted(points):
        if spans and s - spans[-1][1] <= gap:
            spans[-1] = (spans[-1][0], max(spans[-1][1], e))
        else:
            spans.append((s, e))
    return spans


BLOCK = 0.5  # labelling resolution in seconds


def analyze(wav_path: Path, transcript: Transcript) -> list[AudioSpan]:
    """Label the full clip as speech / music / sound / quiet spans.

    Verified failures this version fixes (Captain 6 clip): 1s blocks reported
    music ending at 4.0s when the beat runs to 5.5s; a mid-music rhythmicity dip
    became a phantom 'laughter' span; a 1s rhythmic burst of clapping/laughter
    became 'new music' at 15.0s; whisper onset drift made speech start at 5.1s
    inside the music. Fixes: 0.5s blocks, short-sound-after-music absorption,
    a 2s minimum before a NEW music region is believed, and trimming a short
    speech-onset overlap against the measured music region."""
    env, duration = _envelope(wav_path)
    if not env or duration <= 0:
        return []
    peak = sorted(env)[int(len(env) * 0.95)] or 1.0
    speech = _speech_spans(transcript)

    # 1. Signal-only labels per BLOCK seconds (speech is overlaid at the end).
    labels: list[str] = []
    block = max(1, int(BLOCK / HOP))
    win = int(4.0 / HOP)  # rhythm needs ~4s of context
    for b in range(0, len(env), block):
        chunk = env[b:b + block]
        if (sum(chunk) / len(chunk)) < 0.06 * peak:
            labels.append("quiet")
            continue
        ctx = env[max(0, b - win // 2): b + win // 2 + block]
        labels.append("music" if _rhythmicity(ctx) >= 0.35 else "sound")

    # 2. Merge into regions.
    regions: list[list[float | str]] = []  # [start, end, label]
    for i, lab in enumerate(labels):
        s, e = i * BLOCK, min((i + 1) * BLOCK, duration)
        if regions and regions[-1][2] == lab:
            regions[-1][1] = e
        else:
            regions.append([s, e, lab])

    # 3a. A NEW music region needs sustained beat evidence (>= 2s); shorter
    # rhythmic bursts (clapping, pulsed laughter) are 'sound'.
    for r in regions:
        if r[2] == "music" and r[1] - r[0] < 2.0:
            r[2] = "sound"
    # 3b. A short 'sound' dip (< 1.5s) directly after music is the music
    # continuing through a beat break — absorb it.
    for i in range(1, len(regions)):
        if (regions[i][2] == "sound" and regions[i][1] - regions[i][0] < 1.5
                and regions[i - 1][2] == "music"):
            regions[i][2] = "music"
    merged: list[list[float | str]] = []
    for r in regions:
        if merged and merged[-1][2] == r[2]:
            merged[-1][1] = r[1]
        else:
            merged.append(list(r))
    # 3c. Boundary slack: the block grid only resolves a music end to within one
    # BLOCK, and the rhythm window dilutes the final beats near a transition —
    # extend a music tail one block into an adjacent 'sound' region (verified:
    # beat audibly runs to 5.45s but blocks reported 5.0s).
    for i in range(len(merged) - 1):
        if merged[i][2] == "music" and merged[i + 1][2] == "sound":
            shift = min(BLOCK, float(merged[i + 1][1]) - float(merged[i + 1][0]) - 0.25)
            if shift > 0:
                merged[i][1] = float(merged[i][1]) + shift
                merged[i + 1][0] = merged[i][1]

    # 4. Trim a short speech-onset overlap into a music region (ASR onset
    # drift). A long overlap is left alone — that is singing over music.
    trimmed: list[tuple[float, float]] = []
    for s, e in speech:
        for r in merged:
            if r[2] == "music" and r[0] <= s < r[1] and float(r[1]) - s <= 0.8:
                s = float(r[1])
        if e - s >= 0.15:
            trimmed.append((s, e))

    # 5. Compose: non-speech regions minus speech, plus the speech spans.
    out: list[AudioSpan] = []
    for r in merged:
        cursor = float(r[0])
        for s, e in trimmed:
            if e <= cursor or s >= float(r[1]):
                continue
            if s > cursor:
                out.append(AudioSpan(round(cursor, 1), round(s, 1), str(r[2])))
            cursor = max(cursor, e)
        if float(r[1]) > cursor:
            out.append(AudioSpan(round(cursor, 1), round(float(r[1]), 1), str(r[2])))
    out += [AudioSpan(round(s, 1), round(e, 1), "speech") for s, e in trimmed]
    out.sort(key=lambda sp: sp.start)
    return [sp for sp in out if sp.end - sp.start >= 0.25]


def music_spans(spans: list[AudioSpan]) -> list[tuple[float, float]]:
    return [(sp.start, sp.end) for sp in spans if sp.label == "music"]


def describe(spans: list[AudioSpan]) -> str:
    return "\n".join(f"  {sp.describe()}" for sp in spans)
