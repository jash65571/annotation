"""faster-whisper worker: JSON request in, JSON result out. Runs in its own
isolated uv environment — the core engine never imports these packages.

Protocol: ``python worker.py --request req.json --response res.json``.
Local-only: reads a local WAV, writes a local JSON. task is ALWAYS
"transcribe" (never translate). Word timestamps enabled. All floats are
serialized via repr(str) so the core can parse them with Decimal.
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
    response: dict[str, Any] = {"engine": "faster_whisper"}
    try:
        import faster_whisper

        response["package_version"] = getattr(faster_whisper, "__version__", None)
        from faster_whisper import WhisperModel

        model_name = request["model"]
        device = request.get("device", "auto")
        compute_type = request.get("compute_type", "auto")

        def _load_and_run(dev: str, ctype: str) -> tuple[Any, Any, float, float]:
            load_start = time.perf_counter()
            model = WhisperModel(model_name, device=dev, compute_type=ctype)
            load_secs = time.perf_counter() - load_start
            transcribe_start = time.perf_counter()
            seg_iter, run_info = model.transcribe(
                request["audio_path"],
                task="transcribe",  # NEVER translate
                language=request.get("language"),
                word_timestamps=True,
                vad_filter=bool(request.get("vad", True)),
            )
            return seg_iter, run_info, load_secs, transcribe_start

        try:
            segments_iter, info, load_seconds, transcribe_start = _load_and_run(
                device, compute_type
            )
        except RuntimeError as gpu_exc:
            # Missing local CUDA runtime (cublas/cudnn) is an environment
            # condition, not an ASR failure — retry on CPU and record it.
            if device != "cpu" and any(
                marker in str(gpu_exc).lower()
                for marker in ("cublas", "cudnn", "cuda")
            ):
                response["device_fallback"] = f"gpu unavailable: {gpu_exc}"
                device, compute_type = "cpu", "int8"
                segments_iter, info, load_seconds, transcribe_start = _load_and_run(
                    device, compute_type
                )
            else:
                raise
        segments = []
        for segment in segments_iter:
            words = []
            for word in segment.words or []:
                words.append(
                    {
                        "text": word.word,
                        "start": str(word.start),
                        "end": str(word.end),
                        "probability": word.probability,
                    }
                )
            segments.append(
                {
                    "id": segment.id,
                    "start": str(segment.start),
                    "end": str(segment.end),
                    "text": segment.text,
                    "avg_logprob": segment.avg_logprob,
                    "no_speech_prob": segment.no_speech_prob,
                    "words": words,
                }
            )
        transcribe_seconds = time.perf_counter() - transcribe_start

        response.update(
            {
                "status": "ok",
                "model": model_name,
                "device": device,
                "compute_type": compute_type,
                "load_seconds": round(load_seconds, 3),
                "transcribe_seconds": round(transcribe_seconds, 3),
                "language": info.language,
                "language_probability": info.language_probability,
                "duration": str(info.duration),
                "segments": segments,
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
