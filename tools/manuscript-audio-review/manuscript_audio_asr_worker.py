"""Secondary ASR worker (Phase 3A / hardened in 3A.1).

Runs under `.venv-whisperx` as an isolated subprocess so the heavy
faster-whisper/ctranslate2 stack never has to be imported by the base
`.venv-review` interpreter (design rule 5: worker subprocess, JSON in/out).

Deliberately independent of the primary WhisperX path:
- primary:   whisperx CLI -> faster-whisper large-v3 + forced alignment
- secondary: faster-whisper large-v3-turbo, called directly, no alignment

`large-v3-turbo` is a distinct distilled model (different weights/decoding),
already cached locally (see README), so this adds no new download and no new
dependency -- it reuses faster-whisper, which whisperx already pulls in.

Two modes:

1. One-shot (original 3A behaviour, kept for manual/CLI use):
       python manuscript_audio_asr_worker.py AUDIO.wav [--start S --end E]
   Loads the model, transcribes once, prints one JSON object, exits.

2. Persistent server (3A.1 -- avoids reloading the model per window):
       python manuscript_audio_asr_worker.py --serve --model large-v3-turbo
   Loads the model once, then reads one JSON request per line from stdin and
   writes one JSON response per line to stdout:
       request:  {"audio_path": "...", "start": 3.2, "end": 6.8}
       request:  {"cmd": "shutdown"}
   The decoded audio for a given audio_path is cached in memory across
   requests within the same server process (typically one path per pipeline
   run), so a window rerun only re-transcribes a numpy slice -- no ffmpeg
   subprocess and no disk I/O per request.
"""

import argparse
import json
import sys
import wave
from pathlib import Path


def build_segment_payload(segments, offset=0.0):
    out_segments = []

    for seg in segments:
        words = [
            {
                "word": w.word.strip(),
                "start": round(float(w.start) + offset, 3),
                "end": round(float(w.end) + offset, 3),
                "score": round(float(w.probability), 3),
            }
            for w in (seg.words or [])
        ]

        out_segments.append({
            "start": round(float(seg.start) + offset, 3),
            "end": round(float(seg.end) + offset, 3),
            "text": seg.text.strip(),
            "words": words,
            # Advisory hallucination-risk inputs (spec 3A.1-4); present only
            # when faster-whisper reports them.
            "avg_logprob": (
                round(float(seg.avg_logprob), 4)
                if seg.avg_logprob is not None
                else None
            ),
            "compression_ratio": (
                round(float(seg.compression_ratio), 4)
                if seg.compression_ratio is not None
                else None
            ),
            "no_speech_prob": (
                round(float(seg.no_speech_prob), 4)
                if seg.no_speech_prob is not None
                else None
            ),
        })

    return out_segments


class Transcriber:
    """Loads the model once; keeps a small audio-array cache across calls."""

    def __init__(self, model_name, compute_type="int8"):
        from faster_whisper import WhisperModel

        self.model = WhisperModel(model_name, device="cpu", compute_type=compute_type)
        self.model_name = model_name
        self._audio_cache = {}

    def _load_audio(self, audio_path):
        audio_path = str(audio_path)

        if audio_path in self._audio_cache:
            return self._audio_cache[audio_path]

        # Reads with the stdlib `wave` module rather than soundfile/librosa
        # so this worker adds no new dependency to `.venv-whisperx` --
        # requires 16-bit PCM WAV, which is exactly what the pipeline's
        # ffmpeg extraction step produces (`-c:a pcm_s16le`).
        import numpy as np

        with wave.open(audio_path, "rb") as wf:
            n_channels = wf.getnchannels()
            sample_width = wf.getsampwidth()
            sample_rate = wf.getframerate()
            raw = wf.readframes(wf.getnframes())

        if sample_width != 2:
            raise ValueError(
                f"expected 16-bit PCM WAV, got sample width {sample_width} "
                f"bytes ({audio_path})"
            )

        audio = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0

        if n_channels > 1:
            audio = audio.reshape(-1, n_channels).mean(axis=1)

        # Cache only the most recent source file -- a pipeline run always
        # transcribes the same analysis WAV, so this stays O(1) memory.
        self._audio_cache = {audio_path: (audio, sample_rate)}
        return audio, sample_rate

    def transcribe(self, audio_path, start=None, end=None, options=None):
        audio, sample_rate = self._load_audio(audio_path)

        offset = 0.0

        if start is not None and end is not None:
            s = max(0, int(float(start) * sample_rate))
            e = min(len(audio), int(float(end) * sample_rate))
            clip = audio[s:e]
            offset = s / sample_rate
        else:
            clip = audio

        # Accuracy knobs (faster-whisper 1.x). Defaults match whisper's
        # proven values; callers may tighten them (e.g. strict mode for
        # reruns: no conditioning on previous text + a repetition penalty
        # suppress the long hallucination chains whisper emits on quiet or
        # degraded audio).
        options = dict(options or {})
        options.setdefault("word_timestamps", True)
        options.setdefault("vad_filter", False)

        segments, info = self.model.transcribe(
            clip,
            **options,
        )

        out_segments = build_segment_payload(list(segments), offset=offset)

        return {
            "status": "complete",
            "model": self.model_name,
            "language": info.language,
            "language_probability": round(float(info.language_probability), 3),
            "segments": out_segments,
            "window": (
                [round(offset, 3), round(offset + (len(clip) / sample_rate), 3)]
                if start is not None and end is not None
                else None
            ),
        }


def serve(model_name, compute_type):
    try:
        transcriber = Transcriber(model_name, compute_type=compute_type)
    except Exception as exc:  # noqa: BLE001 -- report, do not crash silently
        sys.stdout.write(
            json.dumps({
                "status": "failed",
                "error": f"model load failed: {type(exc).__name__}: {exc}",
            })
            + "\n"
        )
        sys.stdout.flush()
        return

    sys.stdout.write(json.dumps({"status": "ready", "model": model_name}) + "\n")
    sys.stdout.flush()

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue

        try:
            request = json.loads(line)
        except json.JSONDecodeError as exc:
            sys.stdout.write(
                json.dumps({"status": "failed", "error": f"bad request JSON: {exc}"})
                + "\n"
            )
            sys.stdout.flush()
            continue

        if request.get("cmd") == "shutdown":
            break

        try:
            result = transcriber.transcribe(
                request["audio_path"],
                start=request.get("start"),
                end=request.get("end"),
                options=request.get("options"),
            )
        except Exception as exc:  # noqa: BLE001 -- fail soft per request
            result = {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}

        sys.stdout.write(json.dumps(result, ensure_ascii=False) + "\n")
        sys.stdout.flush()


def run_one_shot(audio_path, model_name, compute_type, start, end, strict=False):
    options = None
    if strict:
        options = {
            "condition_on_previous_text": False,
            "repetition_penalty": 1.15,
        }
    try:
        transcriber = Transcriber(model_name, compute_type=compute_type)
        result = transcriber.transcribe(
            audio_path, start=start, end=end, options=options
        )
    except Exception as exc:  # noqa: BLE001 -- fail soft, report to caller
        result = {
            "status": "failed",
            "error": f"{type(exc).__name__}: {exc}",
            "segments": [],
        }

    json.dump(result, sys.stdout, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("audio_path", nargs="?")
    parser.add_argument("--serve", action="store_true")
    parser.add_argument("--model", default="large-v3-turbo")
    parser.add_argument("--compute-type", default="int8")
    parser.add_argument("--start", type=float, default=None)
    parser.add_argument("--end", type=float, default=None)
    parser.add_argument(
        "--strict",
        action="store_true",
        help="anti-hallucination mode: no conditioning on previous text, "
        "repetition penalty 1.15 (recommended for the primary pass and "
        "reruns on quiet/degraded audio)",
    )
    args = parser.parse_args()

    if args.serve:
        serve(args.model, args.compute_type)
        return

    if not args.audio_path:
        print("usage: manuscript_audio_asr_worker.py AUDIO_PATH [--start S --end E]")
        sys.exit(1)

    run_one_shot(
        Path(args.audio_path),
        args.model,
        args.compute_type,
        args.start,
        args.end,
        strict=args.strict,
    )


if __name__ == "__main__":
    main()
