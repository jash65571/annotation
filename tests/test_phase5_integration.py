"""Phase 5 end-to-end integration: full audit (Phase 1→5) over a real synthetic
clip, then fast re-finalization of the same run directory."""

from __future__ import annotations

import json
from pathlib import Path

from manuscript_reviewer.caption_brain import finalize_run
from manuscript_reviewer.models.caption_brain import CaptionReadiness
from manuscript_reviewer.pipeline import run_audit

from .conftest import requires_ffmpeg


@requires_ffmpeg
def test_full_audit_with_caption_brain(clip_24fps: Path, tmp_path: Path) -> None:
    seed = tmp_path / "seed.md"
    seed.write_text(
        "Video ID: clip_24fps\n\n"
        "[Overview]\n"
        "CHARACTERS\n"
        "C1: A test-pattern figure.\n"
        "SCENE\n"
        "A synthetic color test pattern.\n",
        encoding="utf-8",
    )
    result = run_audit(
        video_path=clip_24fps,
        artifacts_root=tmp_path / "artifacts",
        seed_path=seed,
        shot_analysis=True,
        extract_shot_evidence=False,
        audio_analysis=False,
        visual_intelligence=True,
        ocr_enabled=False,
        caption_brain=True,
    )
    assert result.caption_brain is not None
    cb = result.caption_brain.result
    # Nothing is human-verified → the caption is honestly not ready.
    assert cb.readiness in (CaptionReadiness.REVIEW_REQUIRED, CaptionReadiness.BLOCKED)
    caption_dir = result.run_dir / "caption"
    assert (caption_dir / "caption_facts.json").exists()
    assert (caption_dir / "caption_plan.json").exists()
    assert (caption_dir / "m2_validator.json").exists()
    assert (caption_dir / "final_status.json").exists()
    assert (caption_dir / "caption_manifest.json").exists()
    assert not (caption_dir / "ready_to_enter.md").exists()
    assert (result.run_dir / "review_report.md").exists()
    # The manifest records the Phase 5 outcome.
    manifest = json.loads((result.run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["caption_final_status"] == cb.readiness.value
    # The run never claims PASS over an unready caption.
    assert result.status.value in ("REVIEW_REQUIRED", "PARTIAL", "FAILED")

    # Fast re-finalization from the SAME run dir (no media re-analysis).
    output = finalize_run(result.run_dir)
    assert output.result.readiness in (
        CaptionReadiness.REVIEW_REQUIRED,
        CaptionReadiness.BLOCKED,
    )
    assert output.result.stage_timings_seconds["caption_brain_total"] < 10.0
