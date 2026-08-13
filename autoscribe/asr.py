"""Speech → words with timestamps, via faster-whisper (already in this repo's env).

Returns word- and segment-level timings so the assembler can place spoken text on
the same fine-grained timeline as the visual description. If the clip has no speech
(e.g. music only), this returns an empty list — that is a valid, honest result.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from .ffbin import find_tool


@dataclass(frozen=True)
class Word:
    start: float
    end: float
    text: str
    probability: float


@dataclass(frozen=True)
class SpeechSegment:
    start: float
    end: float
    text: str
    words: list[Word] = field(default_factory=list)


def extract_audio(video: Path, out_wav: Path) -> Path:
    """16 kHz mono WAV — what Whisper wants."""
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg = find_tool("ffmpeg")
    subprocess.run(
        [ffmpeg, "-v", "error", "-i", str(video),
         "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", "-y", str(out_wav)],
        check=True,
    )
    return out_wav


def transcribe(
    wav: Path, model_name: str = "large-v3-turbo", language: str | None = None
) -> list[SpeechSegment]:
    """Transcribe with faster-whisper. Empty list means no speech detected."""
    from faster_whisper import WhisperModel  # type: ignore[import-not-found]  # heavy dep

    model = WhisperModel(model_name, device="auto", compute_type="auto")
    segments, _info = model.transcribe(
        str(wav), language=language, vad_filter=True, word_timestamps=True,
    )
    result: list[SpeechSegment] = []
    for seg in segments:
        words = [
            Word(start=w.start, end=w.end, text=w.word.strip(), probability=w.probability)
            for w in (seg.words or [])
        ]
        text = seg.text.strip()
        if text:
            result.append(SpeechSegment(start=seg.start, end=seg.end, text=text, words=words))
    return result
