"""Lossless local PCM extraction: source evidence WAV + ASR WAV.

Two distinct products (documented conversions, source never overwritten):

- ``audio/source.wav`` — factual source evidence: PCM s16le at the SOURCE
  sample rate and channel layout. Conversion: lossy-codec → PCM decode only
  (the codec's decoded output IS the evidence; 16-bit quantization of the
  decoder's float output is the sole precision change).
- ``audio/asr.wav`` — ASR worker input: PCM s16le, mono, 16 kHz (the input
  contract of the Whisper family). Conversion: channel downmix + resample.
"""

from __future__ import annotations

import logging
import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt

from ..media.ffmpeg_tools import find_tool, run_tool

logger = logging.getLogger(__name__)

ASR_SAMPLE_RATE = 16000


class AudioDecodeError(RuntimeError):
    """PCM extraction failed or produced an unreadable WAV."""


@dataclass(frozen=True)
class DecodedWav:
    path: Path
    sample_rate: int
    channels: int
    sample_count: int  # frames (per-channel samples)
    #: Interleaved int16 samples reshaped to (frames, channels).
    samples: npt.NDArray[np.int16]


def extract_wav(
    video_path: Path,
    out_path: Path,
    stream_index: int = 0,
    sample_rate: int | None = None,
    mono: bool = False,
) -> DecodedWav:
    """Decode one audio stream to a PCM s16le WAV and load it."""
    ffmpeg = find_tool("ffmpeg")
    args = [
        "-v", "error",
        "-i", str(video_path),
        "-map", f"0:a:{stream_index}",
        "-c:a", "pcm_s16le",
    ]
    if sample_rate is not None:
        args += ["-ar", str(sample_rate)]
    if mono:
        args += ["-ac", "1"]
    args += ["-y", str(out_path)]
    run_tool(ffmpeg, args, timeout=1800.0)
    return read_wav(out_path)


def read_wav(path: Path) -> DecodedWav:
    """Read a PCM s16le WAV via the stdlib wave module."""
    try:
        with wave.open(str(path), "rb") as handle:
            channels = handle.getnchannels()
            rate = handle.getframerate()
            width = handle.getsampwidth()
            frames = handle.getnframes()
            raw = handle.readframes(frames)
    except (wave.Error, OSError) as exc:
        raise AudioDecodeError(f"Unreadable WAV {path}: {exc}") from exc
    if width != 2:
        raise AudioDecodeError(f"{path} is not 16-bit PCM (width={width})")
    samples = np.frombuffer(raw, dtype=np.int16).reshape(-1, channels)
    return DecodedWav(
        path=path,
        sample_rate=rate,
        channels=channels,
        sample_count=samples.shape[0],
        samples=samples,
    )


def write_wav_slice(
    source: DecodedWav, out_path: Path, start_sample: int, end_sample: int
) -> Path:
    """Write an exact sample range of the source WAV (for review clips)."""
    start = max(0, start_sample)
    end = min(source.sample_count, end_sample)
    if end <= start:
        raise AudioDecodeError(f"Empty review-clip range {start_sample}-{end_sample}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(out_path), "wb") as handle:
        handle.setnchannels(source.channels)
        handle.setsampwidth(2)
        handle.setframerate(source.sample_rate)
        handle.writeframes(source.samples[start:end].tobytes())
    return out_path


def mono_float(source: DecodedWav) -> npt.NDArray[np.float64]:
    """Channel-averaged float samples in [-1, 1] for metric computation."""
    result: npt.NDArray[np.float64] = source.samples.astype(np.float64).mean(axis=1) / 32768.0
    return result
