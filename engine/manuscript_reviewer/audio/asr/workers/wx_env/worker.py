"""WhisperX forced-alignment worker: JSON request in, JSON result out.

ALIGNMENT ONLY — receives the faster-whisper transcript and must preserve its
wording; it refines word boundaries. It never re-transcribes and never
replaces text. Runs in its own isolated uv environment. Local-only I/O.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", required=True)
    parser.add_argument("--response", required=True)
    args = parser.parse_args()
    request = json.loads(Path(args.request).read_text(encoding="utf-8"))
    response: dict[str, Any] = {"engine": "whisperx"}
    try:
        import importlib.metadata

        import whisperx

        try:
            response["package_version"] = importlib.metadata.version("whisperx")
        except importlib.metadata.PackageNotFoundError:
            response["package_version"] = getattr(whisperx, "__version__", None)

        language = request["language"]
        device = request.get("device", "cpu")

        load_start = time.perf_counter()
        align_model, metadata = whisperx.load_align_model(
            language_code=language, device=device
        )
        load_seconds = time.perf_counter() - load_start

        # Input transcript: faster-whisper segments (wording is authoritative).
        segments = [
            {"start": float(s["start"]), "end": float(s["end"]), "text": s["text"]}
            for s in request["segments"]
        ]

        align_start = time.perf_counter()
        audio = whisperx.load_audio(request["audio_path"])
        aligned = whisperx.align(
            segments,
            align_model,
            metadata,
            audio,
            device,
            return_char_alignments=False,
        )
        align_seconds = time.perf_counter() - align_start

        out_segments = []
        for i, segment in enumerate(aligned.get("segments", [])):
            words = []
            for word in segment.get("words", []):
                words.append(
                    {
                        "text": word.get("word", ""),
                        "start": str(word["start"]) if "start" in word else None,
                        "end": str(word["end"]) if "end" in word else None,
                        "probability": word.get("score"),
                    }
                )
            out_segments.append(
                {
                    "id": i,
                    "start": str(segment.get("start")),
                    "end": str(segment.get("end")),
                    "text": segment.get("text", ""),
                    "words": words,
                }
            )

        response.update(
            {
                "status": "ok",
                "language": language,
                "device": device,
                "align_model": str(getattr(metadata, "get", lambda *_: None)("language_code"))
                if isinstance(metadata, dict)
                else language,
                "load_seconds": round(load_seconds, 3),
                "align_seconds": round(align_seconds, 3),
                "segments": out_segments,
            }
        )
    except Exception as exc:
        response.update(
            {
                "status": "error",
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc()[-4000:],
            }
        )
    Path(args.response).write_text(
        json.dumps(response, ensure_ascii=False), encoding="utf-8"
    )
    return 0 if response.get("status") == "ok" else 1


if __name__ == "__main__":
    sys.exit(main())
