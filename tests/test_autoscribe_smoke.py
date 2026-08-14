"""End-to-end smoke test: real FFmpeg, real frame ledger, stubbed model.

CI previously never imported ``autoscribe`` at all. This exercises the actual
media path — probing encoded-frame timestamps, extracting frames, resolving
shots, rendering, and validating — with the vision/ASR calls stubbed so no key
or network is needed. It is the test that would have caught "the documented
install produces a tool whose pipeline cannot run".
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest

from autoscribe import cuts, frames, render, structured, transcribe
from autoscribe.validate import validate_caption

pytestmark = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="FFmpeg/ffprobe not available",
)


@pytest.fixture(scope="module")
def clip(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A 2-second synthetic clip with a hard cut at 1.0s (colour change).

    It carries a real audio stream so the audio path is genuinely exercised —
    a silent-video fixture would skip straight past extraction and never test
    the transcription failure path at all.
    """
    out = tmp_path_factory.mktemp("autoscribe_smoke") / "clip.mp4"
    part_a = out.parent / "a.mp4"
    part_b = out.parent / "b.mp4"
    listing = out.parent / "list.txt"
    for path, colour, freq in ((part_a, "red", 440), (part_b, "blue", 880)):
        subprocess.run(
            ["ffmpeg", "-v", "error", "-y",
             "-f", "lavfi", "-i", f"color=c={colour}:s=320x240:r=25:d=1",
             "-f", "lavfi", "-i", f"sine=frequency={freq}:sample_rate=16000:duration=1",
             "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest",
             str(path)],
            check=True,
        )
    listing.write_text(f"file '{part_a.as_posix()}'\nfile '{part_b.as_posix()}'\n")
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0",
         "-i", str(listing), "-c", "copy", str(out)],
        check=True,
    )
    return out


def test_probe_returns_a_real_frame_ledger(clip: Path) -> None:
    times = frames.probe_frame_times(clip)
    assert len(times) >= 40, "expected ~50 frames from 2s at 25fps"
    assert times == sorted(times)
    # The first encoded frame need NOT sit at 0.0 — container priming and
    # concat offsets routinely push it later. That is exactly the discrepancy
    # the old `index / hz` grid papered over by asserting 0.0 regardless.
    assert 0.0 <= times[0] < 0.2
    assert len(set(times)) == len(times), "duplicate presentation timestamps"


def test_extracted_frames_carry_source_pts(clip: Path, tmp_path: Path) -> None:
    grid = frames.extract_grid(clip, tmp_path / "frames", hz=10.0)
    assert grid, "no frames extracted"
    ledger = frames.probe_frame_times(clip)
    for f in grid:
        assert f.source_index >= 0, "frame is not anchored to the source ledger"
        assert f.time_seconds == pytest.approx(ledger[f.source_index], abs=1e-3)
        assert f.path.exists()


def test_pts_timing_needs_no_blocker_when_ledger_is_available(
    clip: Path, tmp_path: Path
) -> None:
    from autoscribe.blockers import BlockerLog

    log = BlockerLog()
    frames.extract_grid(clip, tmp_path / "frames", hz=10.0, blockers=log)
    assert not any(b.code == "TIMING_NOT_PTS_ANCHORED" for b in log.entries)


class _StubBackend:
    """Answers every model call with fixed, schema-valid JSON."""

    def __init__(self) -> None:
        self.calls = 0

    def complete(self, content: list[dict[str, Any]], **_k: Any) -> str:
        self.calls += 1
        text = " ".join(
            str(c.get("text", "")) for c in content if c.get("type") == "text"
        )
        if "POSSIBLE shot boundary" in text:
            return json.dumps({"is_cut": True, "cut": "Hard cut"})
        if "[Overview]" in text or "CHARACTERS:" in text:
            return json.dumps({
                "characters": [{"id": "C1", "description": "A person in a red jacket."}],
                "objects": [],
                # Written to the 3D-reconstruction standard: the validator now
                # rejects object inventories, so a stub that emits one would
                # fail the pipeline for the wrong reason.
                "scene": (
                    "A flat colour field fills the frame edge to edge. In the "
                    "foreground no object intervenes; the middle ground is the "
                    "unbroken saturated plane itself, with no visible texture or "
                    "seam; the background is the same plane continuing behind it. "
                    "No furniture, terrain or structure is present, and no horizon "
                    "divides the space above or below centre."
                ),
                "style": (
                    "Flat even illumination with no directional key light from any "
                    "side; shadows are absent entirely across the frame. Colour "
                    "temperature reads neutral. No depth of field, digital capture, "
                    "no non-standard aspect ratio."
                ),
                "audio": "No measured audio.",
                "visual_concerns": "None.",
                "audio_concerns": "None.",
            })
        return json.dumps({
            "shot_type": "medium shot",
            "camera": "Medium, eye-level, static.",
            "camera_movements": [],
            "scene": "No changes from overview.",
            "actions": [{"start": 0.0, "end": 0.5, "text": "C1 raises the right hand."}],
            "playback_speed": "regular",
            "speed_changes": [],
        })


def test_full_structured_pipeline_renders_a_valid_caption(
    clip: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stub = _StubBackend()
    monkeypatch.setattr(structured, "OpenAIVisionBackend", lambda *a, **k: stub)
    monkeypatch.setattr(cuts, "OpenAIVisionBackend", lambda *a, **k: stub)
    # No API key in CI: transcription must fail loudly, not silently.
    monkeypatch.setattr(
        transcribe, "transcribe",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no API key")),
    )

    ann = structured.analyze(clip, tmp_path / "out", hz=10.0)

    assert ann.shots, "pipeline produced no shots"
    assert stub.calls > 0, "the vision backend was never called"

    caption = render.render(ann)
    assert "[Overview]" in caption and "Cast:" in caption
    assert "[Shot 1:" in caption

    log = validate_caption(caption)
    assert log.blocking == [], [b.describe() for b in log.blocking]


def test_transcription_failure_becomes_a_blocker_not_silence(
    clip: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The defect this pins: a swallowed ASR exception rendered as 'no speech'."""
    stub = _StubBackend()
    monkeypatch.setattr(structured, "OpenAIVisionBackend", lambda *a, **k: stub)
    monkeypatch.setattr(cuts, "OpenAIVisionBackend", lambda *a, **k: stub)
    monkeypatch.setattr(
        transcribe, "transcribe",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no API key")),
    )

    ann = structured.analyze(clip, tmp_path / "out", hz=10.0)
    codes = {b.code for b in ann.blockers.blocking}
    assert "TRANSCRIPTION_FAILED" in codes

    ready, _reason = ann.blockers.readiness()
    assert ready is False


def test_empty_transcript_with_audible_audio_blocks(
    clip: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An ASR that returns EMPTY *without raising* loses speech just as
    completely as one that raises — and the shot prompt then tells the model
    there is no dialogue. That path must BLOCK, not merely warn."""
    from autoscribe.transcribe import Transcript

    stub = _StubBackend()
    monkeypatch.setattr(structured, "OpenAIVisionBackend", lambda *a, **k: stub)
    monkeypatch.setattr(cuts, "OpenAIVisionBackend", lambda *a, **k: stub)
    # Succeeds, returns nothing — the silent-loss path.
    monkeypatch.setattr(
        transcribe, "transcribe",
        lambda *a, **k: Transcript(language="", text="", segments=[]),
    )

    ann = structured.analyze(clip, tmp_path / "out", hz=10.0)

    blocking = {b.code for b in ann.blockers.blocking}
    assert "TRANSCRIPTION_FAILED" not in blocking, "ASR did not raise here"
    assert "NO_TRANSCRIPT_DESPITE_AUDIO" in blocking, (
        "an empty transcript over audible audio must block, not warn"
    )


def test_evidence_summary_reports_measured_facts(
    clip: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stub = _StubBackend()
    monkeypatch.setattr(structured, "OpenAIVisionBackend", lambda *a, **k: stub)
    monkeypatch.setattr(cuts, "OpenAIVisionBackend", lambda *a, **k: stub)
    monkeypatch.setattr(
        transcribe, "transcribe",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no API key")),
    )
    ann = structured.analyze(clip, tmp_path / "out", hz=10.0)
    summary = ann.evidence_summary()
    assert "SHOT BOUNDARIES (measured)" in summary
    assert "UNRESOLVED DURING ANALYSIS" in summary
