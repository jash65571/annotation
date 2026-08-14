"""Measured prosody — the evidence path that makes a SUPPORTED tone possible.

The source of truth (§10 Rule 4) requires a supported audible tone on every
speech line, and the Aug 2026 evaluator feedback names missing tone, pitch and
pace as a scoring failure. An earlier pass "solved" the no-evidence problem by
forbidding tone outright, which traded an invention risk for a guaranteed rule
violation. These tests pin the measurement instead.
"""

from __future__ import annotations

import math
import struct
import wave
from pathlib import Path

from autoscribe import prosody
from autoscribe.transcribe import Transcript, Word

SR = 16000


def _write_wav(path: Path, segments: list[tuple[float, float, float]]) -> Path:
    """segments = [(duration_s, frequency_hz, amplitude_0_to_1)]"""
    frames = bytearray()
    for duration, freq, amp in segments:
        for i in range(int(SR * duration)):
            value = 0.0
            if freq > 0:
                # A few harmonics so autocorrelation sees a real voiced pitch.
                for h, weight in ((1, 1.0), (2, 0.5), (3, 0.25)):
                    value += weight * math.sin(2 * math.pi * freq * h * i / SR)
                value /= 1.75
            frames += struct.pack("<h", int(max(-1.0, min(1.0, value)) * amp * 32000))
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(bytes(frames))
    return path


def _transcript(words: list[tuple[float, float, str]]) -> Transcript:
    return Transcript(
        language="english", text=" ".join(t for _s, _e, t in words), segments=[],
        words=[Word(s, e, t) for s, e, t in words],
    )


def test_louder_span_is_measured_as_louder(tmp_path: Path) -> None:
    wav = _write_wav(tmp_path / "a.wav", [(1.0, 150, 0.2), (1.0, 150, 1.0)])
    out = prosody.analyze(wav, _transcript([]), [(0.0, 1.0), (1.0, 2.0)])
    assert "quieter" in out[0].loudness
    assert "louder" in out[1].loudness


def test_pitch_is_measured_relative_to_the_speaker(tmp_path: Path) -> None:
    """Two spans, one an octave up: the higher must read as higher."""
    wav = _write_wav(tmp_path / "b.wav", [(1.0, 120, 0.8), (1.0, 240, 0.8)])
    out = prosody.analyze(wav, _transcript([]), [(0.0, 1.0), (1.0, 2.0)])
    assert out[0].f0_hz is not None and out[1].f0_hz is not None
    assert out[1].f0_hz > out[0].f0_hz
    assert "higher" in out[1].pitch or "lower" in out[0].pitch


def test_measured_f0_is_close_to_the_real_frequency(tmp_path: Path) -> None:
    wav = _write_wav(tmp_path / "c.wav", [(1.0, 200, 0.8)])
    out = prosody.analyze(wav, _transcript([]), [(0.0, 1.0)])
    assert out[0].f0_hz is not None
    assert abs(out[0].f0_hz - 200) < 20, f"measured {out[0].f0_hz}"


def test_silence_yields_unresolved_pitch_not_a_guess(tmp_path: Path) -> None:
    wav = _write_wav(tmp_path / "d.wav", [(1.0, 0, 0.0)])
    out = prosody.analyze(wav, _transcript([]), [(0.0, 1.0)])
    assert out[0].pitch == prosody.UNRESOLVED
    assert out[0].f0_hz is None


def test_pace_is_measured_from_word_density(tmp_path: Path) -> None:
    wav = _write_wav(tmp_path / "e.wav", [(2.0, 150, 0.8)])
    fast = _transcript([(0.0 + i * 0.2, 0.15 + i * 0.2, f"w{i}") for i in range(5)])
    out = prosody.analyze(wav, fast, [(0.0, 1.0)])
    assert out[0].pace == "fast", out[0].describe()

    slow = _transcript([(0.0, 0.4, "one"), (0.9, 1.4, "two")])
    out2 = prosody.analyze(wav, slow, [(0.0, 2.0)])
    assert out2[0].pace == "slow", out2[0].describe()


def test_pace_unresolved_without_word_timestamps(tmp_path: Path) -> None:
    wav = _write_wav(tmp_path / "f.wav", [(1.0, 150, 0.8)])
    out = prosody.analyze(wav, Transcript("en", "hi", []), [(0.0, 1.0)])
    assert out[0].pace == prosody.UNRESOLVED


def test_too_short_span_is_not_measured(tmp_path: Path) -> None:
    wav = _write_wav(tmp_path / "g.wav", [(1.0, 150, 0.8)])
    out = prosody.analyze(wav, _transcript([]), [(0.0, 0.05)])
    assert out[0].all_unresolved


def test_describe_omits_unresolved_attributes() -> None:
    p = prosody.Prosody(0.0, 1.0, "loud", prosody.UNRESOLVED, prosody.UNRESOLVED)
    text = p.describe()
    assert "loudness=loud" in text
    assert "pitch" not in text and "pace" not in text


def test_describe_reports_no_measurement_when_all_unresolved() -> None:
    p = prosody.Prosody(0.0, 1.0, prosody.UNRESOLVED, prosody.UNRESOLVED,
                        prosody.UNRESOLVED)
    assert "no delivery measurement possible" in p.describe()


def test_for_range_selects_by_midpoint() -> None:
    items = [
        prosody.Prosody(0.0, 1.0, "a", "b", "c"),
        prosody.Prosody(5.0, 6.0, "a", "b", "c"),
    ]
    assert len(prosody.for_range(items, 0.0, 2.0)) == 1
    assert len(prosody.for_range(items, 0.0, 10.0)) == 2


def test_missing_audio_yields_unresolved_not_a_crash(tmp_path: Path) -> None:
    empty = tmp_path / "empty.wav"
    _write_wav(empty, [])
    out = prosody.analyze(empty, _transcript([]), [(0.0, 1.0)])
    assert out and out[0].all_unresolved


def test_prompt_requires_tone_from_measurement_not_from_frames() -> None:
    """Regression: tone was banned outright, violating source-of-truth §10."""
    from autoscribe.structured import _SHOT_PROMPT

    assert "supported audible" in _SHOT_PROMPT
    assert "DO NOT state delivery or tone" not in _SHOT_PROMPT
    assert "MEASURED DELIVERY" in _SHOT_PROMPT or "measured loudness" in _SHOT_PROMPT
    assert "never infer tone from a facial expression" in _SHOT_PROMPT
