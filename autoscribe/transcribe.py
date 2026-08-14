"""Real audio transcription via OpenAI speech-to-text (whisper-1).

Returns verbatim segments + detected language so speech and sung lyrics can be
placed on the timeline and attributed to characters — satisfying Manuscript
Rule 6 (transcribe lyrics) and Rule 7 (foreign language: transcribe if the model
is confident of the language, which Whisper reports).
"""

from __future__ import annotations

import json
import os
import urllib.request
import uuid
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class Word:
    start: float
    end: float
    text: str


@dataclass(frozen=True)
class Segment:
    start: float
    end: float
    text: str
    #: Whisper confidence signals (verbose_json). Used to catch hallucinated
    #: transcripts — Whisper invents plausible words over unclear music/noise.
    avg_logprob: float = 0.0
    no_speech_prob: float = 0.0
    compression_ratio: float = 1.0

    @property
    def reliable(self) -> bool:
        """Standard Whisper hallucination heuristics: low log-prob, high
        no-speech probability, or repetitive output all mean the words cannot
        be trusted as verbatim (Manuscript: use [inaudible], never guess).

        Thresholds calibrated on real clips: correct transcripts (EN/PL) sit at
        avg_logprob >= -0.59 and no_speech_prob <= 0.29; a garbled Russian
        hallucination sat at -0.75 / 0.34. -0.70 separates them cleanly."""
        return (
            self.avg_logprob >= -0.70
            and self.no_speech_prob <= 0.5
            and self.compression_ratio <= 2.4
        )


@dataclass(frozen=True)
class Transcript:
    language: str
    text: str
    segments: list[Segment]
    #: Word-level timestamps (whisper word granularity). Far more accurate than
    #: segment bounds — segment timing drifts badly over music intros, which is
    #: how narration got stamped at 0.0s on a clip whose speech starts at 5.5s.
    words: list[Word] = field(default_factory=list)

    @property
    def has_speech(self) -> bool:
        return bool(self.text.strip())

    @property
    def reliable_segments(self) -> list[Segment]:
        return [s for s in self.segments if s.reliable]

    @property
    def language_confident(self) -> bool:
        """Name the language (Rule 7) only when the transcription itself is
        trustworthy; a hallucinated transcript means a guessed language.

        An ABSENT language is unestablished by definition. Without this, a
        transcript carrying real speech but no language value (the API can
        return none) reported ``language_confident=True`` — the state that
        skips the §17 fallback entirely, so speech shipped with no declaration
        at all.
        """
        if not self.language.strip():
            return False
        if not self.segments:
            return False
        return len(self.reliable_segments) * 2 >= len(self.segments)


def _multipart(fields: list[tuple[str, str]], file_bytes: bytes, boundary: str) -> bytes:
    parts: list[bytes] = []
    for name, value in fields:
        parts.append(
            f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n'
            f"{value}\r\n".encode()
        )
    parts.append(
        f'--{boundary}\r\nContent-Disposition: form-data; name="file"; '
        f'filename="audio.wav"\r\nContent-Type: audio/wav\r\n\r\n'.encode()
    )
    parts.append(file_bytes)
    parts.append(f"\r\n--{boundary}--\r\n".encode())
    return b"".join(parts)


def transcribe(wav: Path, model: str = "whisper-1") -> Transcript:
    """Transcribe a WAV with verbatim text + segment timestamps + language."""
    key = os.environ.get("OPENAI_API_KEY", "")
    if not key:
        raise RuntimeError("OPENAI_API_KEY is not set for audio transcription.")
    boundary = "----autoscribe" + uuid.uuid4().hex
    body = _multipart(
        [
            ("model", model),
            ("response_format", "verbose_json"),
            ("timestamp_granularities[]", "segment"),
            ("timestamp_granularities[]", "word"),
        ],
        wav.read_bytes(),
        boundary,
    )
    req = urllib.request.Request(
        "https://api.openai.com/v1/audio/transcriptions", data=body,
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": f"multipart/form-data; boundary={boundary}",
        },
    )
    with urllib.request.urlopen(req, timeout=240) as resp:
        data = json.loads(resp.read())
    segments = [
        Segment(
            float(s["start"]), float(s["end"]), str(s["text"]).strip(),
            avg_logprob=float(s.get("avg_logprob", 0.0)),
            no_speech_prob=float(s.get("no_speech_prob", 0.0)),
            compression_ratio=float(s.get("compression_ratio", 1.0)),
        )
        for s in data.get("segments", [])
        if str(s.get("text", "")).strip()
    ]
    words = [
        Word(float(w["start"]), float(w["end"]), str(w["word"]).strip())
        for w in data.get("words", [])
        if str(w.get("word", "")).strip()
    ]
    return Transcript(
        language=str(data.get("language", "")),
        text=str(data.get("text", "")).strip(),
        segments=segments,
        words=words,
    )
