"""Signal-measured audio timeline. Stdlib only — no numpy/scipy.

Motivation (verified failure): the vision passes were told to infer audio from
what is *visible*, which deleted real non-diegetic music and let hallucinated
narration cover the whole clip. This module measures the waveform itself:

  - RMS energy envelope  -> quiet vs. audible regions;
  - onset-autocorrelation rhythmicity -> "music-like" (steady beat) vs.
    "non-rhythmic" (crowd noise, laughter, ambience);
  - speech spans come from word-level ASR timestamps (transcribe.Word), which
    are far more accurate than whisper segment bounds.

Two properties matter for correctness:

*Tracks overlap.* Audio is not a partition. Music playing under dialogue is
both music and dialogue, and the previous version subtracted speech out of the
music region — which is exactly how "music under dialogue" vanished from
captions. Spans are now layered and may overlap; ``analyze`` returns every
audible layer that was measured.

*Uncertainty is a label, not a coin flip.* A rhythmicity score in the ambiguous
middle band is reported as ``unresolved`` rather than forced to music or sound,
and the coarse ``sound`` class is explicitly documented as undifferentiated
(ambience vs. SFX vs. crowd reaction is NOT decided here). Callers turn both
into review items instead of stating them as fact.
"""

from __future__ import annotations

import array
import wave
from dataclasses import dataclass
from pathlib import Path

from .blockers import WARNING, BlockerLog
from .transcribe import Transcript

HOP = 0.05  # envelope hop in seconds

#: Rhythmicity thresholds. Between them the class is genuinely undecided.
MUSIC_THRESHOLD = 0.35
SOUND_THRESHOLD = 0.20


@dataclass(frozen=True)
class AudioSpan:
    start: float
    end: float
    #: 'speech' | 'music' | 'sound' | 'quiet' | 'unresolved'
    label: str

    def describe(self) -> str:
        names = {
            "speech": "speech (see verbatim transcript)",
            "music": "strongly rhythmic audio — music with a steady beat "
                     "(non-diegetic music needs NO visible source)",
            "sound": "non-speech audio of an UNDETERMINED kind — this may be "
                     "ambience, a sound effect, or a crowd reaction (laughter, "
                     "gasps, cheers); the signal analysis does not distinguish "
                     "them, so describe only what the frames actually support",
            "quiet": "near-silence",
            "unresolved": "audible non-speech audio whose kind could NOT be "
                          "determined (rhythmicity was ambiguous) — do not "
                          "state what it is",
        }
        return f"{self.start:.1f}s-{self.end:.1f}s: {names[self.label]}"


@dataclass
class _Region:
    """Mutable working region during labelling (typed, unlike the old
    ``list[float | str]`` which mypy could not check at all)."""

    start: float
    end: float
    label: str


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


def analyze(
    wav_path: Path, transcript: Transcript, blockers: BlockerLog | None = None
) -> list[AudioSpan]:
    """Label the clip as layered speech / music / sound / unresolved / quiet spans.

    Spans MAY OVERLAP: a music bed and the dialogue over it are both returned in
    full. Callers must not assume the list partitions the timeline.

    Verified failures this version fixes (Captain 6 clip): 1s blocks reported
    music ending at 4.0s when the beat runs to 5.5s; a mid-music rhythmicity dip
    became a phantom 'laughter' span; a 1s rhythmic burst of clapping/laughter
    became 'new music' at 15.0s; whisper onset drift made speech start at 5.1s
    inside the music; and speech was SUBTRACTED from music, deleting the bed.
    """
    env, duration = _envelope(wav_path)
    if not env or duration <= 0:
        if blockers is not None:
            blockers.add(
                "AUDIO_NOT_ANALYZED",
                "The audio envelope could not be measured, so no music, ambience, "
                "sound effects or reactions were detected. Audio content in this "
                "caption is unverified.",
            )
        return []
    peak = sorted(env)[int(len(env) * 0.95)] or 1.0
    speech = _speech_spans(transcript)

    # 1. Signal-only labels per BLOCK seconds. Speech is layered on top later,
    #    never subtracted from these.
    labels: list[str] = []
    block = max(1, int(BLOCK / HOP))
    win = int(4.0 / HOP)  # rhythm needs ~4s of context
    for b in range(0, len(env), block):
        chunk = env[b:b + block]
        if (sum(chunk) / len(chunk)) < 0.06 * peak:
            labels.append("quiet")
            continue
        ctx = env[max(0, b - win // 2): b + win // 2 + block]
        score = _rhythmicity(ctx)
        if score >= MUSIC_THRESHOLD:
            labels.append("music")
        elif score <= SOUND_THRESHOLD:
            labels.append("sound")
        else:
            labels.append("unresolved")

    # 2. Merge into regions.
    regions: list[_Region] = []
    for i, lab in enumerate(labels):
        s, e = i * BLOCK, min((i + 1) * BLOCK, duration)
        if regions and regions[-1].label == lab:
            regions[-1].end = e
        else:
            regions.append(_Region(s, e, lab))

    # 3a. A NEW music region needs sustained beat evidence (>= 2s); shorter
    # rhythmic bursts (clapping, pulsed laughter) are 'sound'.
    for r in regions:
        if r.label == "music" and r.end - r.start < 2.0:
            r.label = "sound"
    # 3b. A short 'sound' dip (< 1.5s) directly after music is the music
    # continuing through a beat break — absorb it.
    for i in range(1, len(regions)):
        if (regions[i].label == "sound" and regions[i].end - regions[i].start < 1.5
                and regions[i - 1].label == "music"):
            regions[i].label = "music"
    merged: list[_Region] = []
    for r in regions:
        if merged and merged[-1].label == r.label:
            merged[-1].end = r.end
        else:
            merged.append(_Region(r.start, r.end, r.label))
    # 3c. Boundary slack: the block grid only resolves a music end to within one
    # BLOCK, and the rhythm window dilutes the final beats near a transition —
    # extend a music tail one block into an adjacent 'sound' region (verified:
    # beat audibly runs to 5.45s but blocks reported 5.0s).
    for i in range(len(merged) - 1):
        if merged[i].label == "music" and merged[i + 1].label == "sound":
            shift = min(BLOCK, merged[i + 1].end - merged[i + 1].start - 0.25)
            if shift > 0:
                merged[i].end += shift
                merged[i + 1].start = merged[i].end

    # 4. Trim a short speech-onset overlap into a music region (ASR onset
    # drift). A long overlap is left alone — that is singing/dialogue over music.
    trimmed: list[tuple[float, float]] = []
    for s, e in speech:
        for r in merged:
            if r.label == "music" and r.start <= s < r.end and r.end - s <= 0.8:
                s = r.end
        if e - s >= 0.15:
            trimmed.append((s, e))

    # 5. Compose LAYERS. Non-speech regions are emitted whole — a music bed is
    #    not cut into pieces by the dialogue on top of it — and speech spans are
    #    added as their own layer.
    out: list[AudioSpan] = [
        AudioSpan(round(r.start, 1), round(r.end, 1), r.label) for r in merged
    ]
    out += [AudioSpan(round(s, 1), round(e, 1), "speech") for s, e in trimmed]
    out = [sp for sp in out if sp.end - sp.start >= 0.25]
    out.sort(key=lambda sp: (sp.start, sp.label))

    if blockers is not None:
        for sp in out:
            if sp.label == "unresolved":
                blockers.add(
                    "AUDIO_CLASS_UNRESOLVED",
                    "Audible non-speech audio whose kind could not be determined "
                    "(ambiguous rhythmicity).",
                    start=sp.start, end=sp.end,
                )
            elif sp.label == "sound":
                blockers.add(
                    "AUDIO_CLASS_COARSE",
                    "Non-speech audio detected but not classified — ambience, sound "
                    "effect and crowd reaction are not distinguished by the signal "
                    "analysis. Confirm what this actually is.",
                    severity=WARNING, start=sp.start, end=sp.end,
                )
    return out


def music_spans(spans: list[AudioSpan]) -> list[tuple[float, float]]:
    return [(sp.start, sp.end) for sp in spans if sp.label == "music"]


def describe(spans: list[AudioSpan]) -> str:
    header = (
        "  (Layers may OVERLAP — music under speech is listed as both.)\n"
        if any(sp.label == "speech" for sp in spans) else ""
    )
    return header + "\n".join(f"  {sp.describe()}" for sp in spans)
