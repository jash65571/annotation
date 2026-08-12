"""Integration tests: full audit runs, manifest, hashing, seed, failure paths."""

from __future__ import annotations

import json
from pathlib import Path

from manuscript_reviewer.artifacts.writer import sha256_file
from manuscript_reviewer.models.validation import RunStatus
from manuscript_reviewer.pipeline import run_audit
from manuscript_reviewer.rules.loader import load_rules
from tests.conftest import requires_ffmpeg


@requires_ffmpeg
def test_full_audit_pass(clip_60fps_audio: Path, tmp_path: Path) -> None:
    result = run_audit(clip_60fps_audio, artifacts_root=tmp_path)
    assert result.status == RunStatus.PASS
    assert result.fatal_error is None

    run_dir = result.run_dir
    for name in ("media.json", "frames.csv", "frames.jsonl", "manifest.json", "qc.json", "run.log"):
        assert (run_dir / name).is_file(), f"missing artifact {name}"

    # frames.csv: header + one row per enumerated frame, starting at index 0.
    lines = (run_dir / "frames.csv").read_text(encoding="utf-8").strip().splitlines()
    assert result.ledger is not None
    assert len(lines) == result.ledger.frame_count + 1
    assert lines[1].startswith("0,")
    assert lines[-1].startswith(f"{result.ledger.frame_count - 1},")

    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["run_id"] == result.run_id
    assert manifest["source_video_sha256"] == sha256_file(clip_60fps_audio)
    assert manifest["validation_status"] == "PASS"
    assert manifest["rules_version"] == load_rules().version
    assert manifest["ffprobe_version"].startswith("ffprobe")
    assert manifest["analysis_duration_seconds"] > 0
    assert "enumerate_frames" in manifest["stage_timings_seconds"]

    # Artifact hashes in the manifest match the files on disk.
    for entry in manifest["artifacts"]:
        assert sha256_file(run_dir / entry["path"]) == entry["sha256"]

    qc = json.loads((run_dir / "qc.json").read_text(encoding="utf-8"))
    assert qc["status"] == "PASS"
    signals = {s["method"]: s["count"] for s in qc["frame_count_signals"]}
    assert signals["ffprobe -show_frames enumeration"] == result.ledger.frame_count
    assert signals["ffprobe -count_frames decode"] == result.ledger.frame_count


@requires_ffmpeg
def test_audit_with_seed_hashes_and_copies(clip_24fps: Path, tmp_path: Path) -> None:
    seed = tmp_path / "seed.json"
    seed.write_text('{"caption": "hypothesis"}', encoding="utf-8")
    result = run_audit(clip_24fps, artifacts_root=tmp_path / "runs", seed_path=seed)
    assert result.status == RunStatus.PASS
    assert result.manifest is not None
    assert result.manifest.source_seed_sha256 == sha256_file(seed)
    assert (result.run_dir / "seed.json").is_file()


@requires_ffmpeg
def test_audit_corrupt_file_fails_cleanly(corrupt_file: Path, tmp_path: Path) -> None:
    result = run_audit(corrupt_file, artifacts_root=tmp_path)
    assert result.status == RunStatus.FAILED
    assert result.qc is not None
    assert any(i.rule_id == "P1-MEDIA-000" for i in result.qc.issues)
    # Even failed runs leave an auditable trail.
    assert (result.run_dir / "qc.json").is_file()
    assert (result.run_dir / "manifest.json").is_file()


@requires_ffmpeg
def test_extract_frames_flag(clip_24fps: Path, tmp_path: Path) -> None:
    result = run_audit(clip_24fps, artifacts_root=tmp_path, extract_frames=True)
    assert result.status == RunStatus.PASS
    assert result.ledger is not None
    frames_dir = result.run_dir / "frames"
    pngs = list(frames_dir.glob("F*.png"))
    assert len(pngs) == result.ledger.frame_count
