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
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Segment:
    start: float
    end: float
    text: str


@dataclass(frozen=True)
class Transcript:
    language: str
    text: str
    segments: list[Segment]

    @property
    def has_speech(self) -> bool:
        return bool(self.text.strip())


def _multipart(fields: dict[str, str], file_bytes: bytes, boundary: str) -> bytes:
    parts: list[bytes] = []
    for name, value in fields.items():
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
        {
            "model": model,
            "response_format": "verbose_json",
            "timestamp_granularities[]": "segment",
        },
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
        Segment(float(s["start"]), float(s["end"]), str(s["text"]).strip())
        for s in data.get("segments", [])
        if str(s.get("text", "")).strip()
    ]
    return Transcript(
        language=str(data.get("language", "")),
        text=str(data.get("text", "")).strip(),
        segments=segments,
    )
