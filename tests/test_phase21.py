"""Phase 2.1 hardening tests: shot timeline semantics, annotation endpoint,
rounding edges, low-MAD candidate recall, subprocess wrapper, scdet error
narrowing, top-level status propagation, and current rule keys."""

from __future__ import annotations

from fractions import Fraction
from pathlib import Path

import pytest

from manuscript_reviewer.media.endpoint import compute_annotation_endpoint
from manuscript_reviewer.media.ffmpeg_tools import (
    ToolExecutionError,
    find_tool,
    run_tool_binary,
)
from manuscript_reviewer.models.frame import FrameLedger, FrameRecord
from manuscript_reviewer.models.media import MediaInfo, VideoStreamInfo
from manuscript_reviewer.models.shot_truth import CandidateStatus
from manuscript_reviewer.models.validation import RunStatus
from manuscript_reviewer.pipeline import run_audit
from manuscript_reviewer.rules.loader import load_rules
from manuscript_reviewer.shots import scdet
from tests.conftest import requires_ffmpeg
from tests.test_shot_truth import SIZE, _lavfi_concat, _lavfi_filtered, analyze

# ---------------------------------------------------------------- FIX 1: rules

def test_current_controlling_rule_keys_load() -> None:
    rules = load_rules()
    assert rules.version == "1.3.0"
    assert rules.get("source_hierarchy.newest_official_workflow_outranks_older") is True
    assert rules.get("source_hierarchy.actual_media_is_factual_truth") is True
    assert rules.get("source_hierarchy.golden_examples_are_floor_not_ceiling") is True
    assert rules.get("action_audio_atomicity.strict_atomicity") is True
    assert rules.get("action_audio_atomicity.one_line_one_defensible_event") is True
    assert rules.get("action_audio_atomicity.hidden_second_action_forbidden") is True
    assert "while" in rules.get("action_audio_atomicity.hidden_action_connectives")
    assert rules.get("action_audio_atomicity.multiple_speech_acts_per_line_forbidden") is True
    assert rules.get("action_audio_atomicity.mixed_event_sources_per_line_forbidden") is True
    assert rules.get("timing_integrity.fake_timestamp_nudges_forbidden") is True
    assert rules.get("timing_integrity.final_object_state_check_required") is True
    assert (
        rules.get("timing_integrity.incorrect_annotation_endpoint_is_permanent_failure")
        is True
    )
    assert rules.get("validation_scope.machine_validator_replaces_platform_validation") is False
    assert rules.get("audio_tooling.descript_enabled") is False
    assert rules.get("audio_tooling.local_asr_failure_authorizes_cloud_fallback") is False
    assert rules.get("audio_tooling.local_asr_failure_authorizes_descript_fallback") is False
    assert rules.get("audio_tooling.asr_is_evidence_only") is True
    assert rules.get("audio_tooling.full_audit_continues_after_asr_failure") is True


# ------------------------------------------------- FIX 2: timeline semantics

@requires_ffmpeg
def test_shot_timeline_boundary_equality(tmp_path: Path) -> None:
    """Both shots share the SAME exact boundary time; ownership stays split."""
    clip = _lavfi_concat(
        "st_hardcut.mp4",
        [f"testsrc2=duration=1:rate=24:size={SIZE}",
         f"smptebars=duration=1:rate=24:size={SIZE}"],
    )
    result = analyze(clip, tmp_path)
    shot1, shot2 = result.shots
    # Continuous annotation interval: shared boundary time = PTS(F24) = 1 s.
    assert shot1.end_exact == Fraction(1)
    assert shot2.start_exact == Fraction(1)
    assert shot1.end_exact == shot2.start_exact
    # Inclusive frame ownership is a separate concept and stays F23 / F24.
    assert shot1.end_frame_index == 23
    assert shot2.start_frame_index == 24
    assert shot1.last_owned_frame_start_exact == Fraction(23, 24)
    # Shot 1 starts at the media timeline start.
    assert shot1.start_exact == Fraction(0)


@requires_ffmpeg
def test_final_shot_uses_annotation_endpoint_not_final_frame_start(tmp_path: Path) -> None:
    clip = _lavfi_concat(
        "st_hardcut.mp4",
        [f"testsrc2=duration=1:rate=24:size={SIZE}",
         f"smptebars=duration=1:rate=24:size={SIZE}"],
    )
    result = analyze(clip, tmp_path)
    final = result.shots[-1]
    final_frame_start = Fraction(47, 24)
    assert final.last_owned_frame_start_exact == final_frame_start
    # Endpoint = final frame presentation end = exactly 2 s, NOT 47/24.
    assert final.end_exact == Fraction(2)
    assert final.end_exact != final_frame_start
    assert result.annotation_endpoint_exact == Fraction(2)
    assert result.annotation_endpoint_conflict is False


# --------------------------------------------- FIX 2B: rounding regression

@requires_ffmpeg
def test_rounding_edge_shared_boundary_display(tmp_path: Path) -> None:
    """Cut at 0.55 s (60 fps): outgoing final-frame START (0.5333 s) rounds to
    0.5 but the boundary (0.55 s) rounds to 0.6. Both Manuscript windows must
    share the SAME 0.6 boundary — never 0.5 / 0.6."""
    clip = _lavfi_concat(
        "st_round_edge.mp4",
        [f"testsrc2=duration=0.55:rate=60:size={SIZE}",
         f"smptebars=duration=1:rate=60:size={SIZE}"],
    )
    result = analyze(clip, tmp_path)
    boundaries = [c for c in result.candidates if c.status == CandidateStatus.SUPPORTED]
    assert len(boundaries) == 1
    assert boundaries[0].right_frame_index == 33
    assert boundaries[0].boundary_time_exact == Fraction(33, 60)
    shot1, shot2 = result.shots
    # The old bug: using F32's start (0.5333→"0.5s") as shot 1's end.
    assert shot1.last_owned_frame_start_exact == Fraction(32, 60)
    assert shot1.end_manuscript == "0.6s"
    assert shot2.start_manuscript == "0.6s"
    assert shot1.end_manuscript == shot2.start_manuscript


# ------------------------------------------------ FIX 2: endpoint helper

def _ledger_24fps(n: int) -> FrameLedger:
    tb = Fraction(1, 12288)
    return FrameLedger(
        stream_index=0,
        time_base=tb,
        frames=[
            FrameRecord(
                frame_index=i,
                pts=i * 512,
                pts_time_seconds=Fraction(i, 24),
                duration=512,
                duration_seconds=Fraction(1, 24),
                key_frame=i == 0,
            )
            for i in range(n)
        ],
    )


def _media(duration: Fraction | None) -> MediaInfo:
    return MediaInfo(
        file_name="x.mp4",
        file_size_bytes=1,
        container_format="mp4",
        container_duration_seconds=duration,
        video_streams=[
            VideoStreamInfo(
                stream_index=0,
                codec_name="h264",
                width=320,
                height=240,
                time_base=Fraction(1, 12288),
                declared_duration_seconds=duration,
            )
        ],
        audio_streams=[],
    )


def test_endpoint_prefers_final_frame_presentation_end() -> None:
    result = compute_annotation_endpoint(_media(Fraction(2)), _ledger_24fps(48), "clip")
    assert result.endpoint == Fraction(2)
    assert result.method == "final_frame_presentation_end"
    assert result.conflict is False


def test_endpoint_material_conflict_is_exposed() -> None:
    # Declared durations claim 3 s but frames end at 2 s → material conflict.
    result = compute_annotation_endpoint(_media(Fraction(3)), _ledger_24fps(48), "clip")
    assert result.endpoint == Fraction(2)  # media evidence wins
    assert result.conflict is True


def test_endpoint_filename_segment_signal() -> None:
    result = compute_annotation_endpoint(
        _media(Fraction(2)), _ledger_24fps(48), "abcdef123456_408.0_410.0"
    )
    assert result.signals["filename_segment_length"] == Fraction(2)
    assert result.conflict is False


# ------------------------------------------------ FIX 3: low-MAD recall

@requires_ffmpeg
def test_low_mad_structural_cut_not_lost(tmp_path: Path) -> None:
    """A real cut with tiny luma difference but structural change must surface
    as SUPPORTED or REVIEW_REQUIRED — never vanish below the MAD floor."""
    clip = _lavfi_concat(
        "st_lowmad.mp4",
        [
            f"color=c=0x808080:duration=1:rate=24:size={SIZE},"
            "drawbox=x=30:y=40:w=100:h=140:color=0x888888:t=fill",
            f"color=c=0x808080:duration=1:rate=24:size={SIZE},"
            "drawbox=x=190:y=40:w=100:h=140:color=0x888888:t=fill",
        ],
    )
    result = analyze(clip, tmp_path)
    at_cut = [c for c in result.candidates if (c.left_frame_index, c.right_frame_index) == (23, 24)]
    assert at_cut, "low-MAD structural cut produced no candidate at all"
    candidate = at_cut[0]
    # Prove the fixture actually sits below the difference-family floor.
    assert candidate.metric_snapshot.mean_abs_diff < 4.0
    assert candidate.status in (CandidateStatus.SUPPORTED, CandidateStatus.REVIEW_REQUIRED)


@requires_ffmpeg
def test_low_mad_same_shot_noise_not_supported(tmp_path: Path) -> None:
    clip = _lavfi_filtered(
        "st_noise.mp4",
        f"color=c=0x808080:duration=2:rate=24:size={SIZE}",
        "noise=alls=6:allf=t",
    )
    result = analyze(clip, tmp_path)
    assert [c for c in result.candidates if c.status == CandidateStatus.SUPPORTED] == []


# ------------------------------------------- FIX 4: binary subprocess layer

@requires_ffmpeg
def test_run_tool_binary_rawvideo() -> None:
    ffmpeg = find_tool("ffmpeg")
    result = run_tool_binary(
        ffmpeg,
        [
            "-v", "error",
            "-f", "lavfi", "-i", "color=c=black:duration=0.25:rate=8:size=16x16",
            "-f", "rawvideo", "-pix_fmt", "gray", "-",
        ],
    )
    assert isinstance(result.stdout, bytes)
    assert len(result.stdout) == 2 * 16 * 16  # 2 frames of 16x16 gray
    assert result.stdout == b"\x00" * len(result.stdout)


@requires_ffmpeg
def test_run_tool_binary_failure_raises() -> None:
    ffmpeg = find_tool("ffmpeg")
    with pytest.raises(ToolExecutionError) as exc_info:
        run_tool_binary(ffmpeg, ["-v", "error", "-i", "does_not_exist.mp4", "-f", "null", "-"])
    assert "does_not_exist" in str(exc_info.value)


def test_no_direct_subprocess_outside_wrapper() -> None:
    """All media subprocess calls flow through media/ffmpeg_tools.py."""
    engine_root = Path(__file__).parent.parent / "engine" / "manuscript_reviewer"
    offenders = []
    for path in engine_root.rglob("*.py"):
        if path.name == "ffmpeg_tools.py" or ".venv" in path.parts:
            continue
        text = path.read_text(encoding="utf-8")
        for needle in ("subprocess.run(", "subprocess.Popen(", "os.system(", "shell=True"):
            if needle in text:
                offenders.append(f"{path.name}: {needle}")
    assert offenders == []


# --------------------------------------------- FIX 4A: scdet error handling

def test_scdet_expected_tool_failure_degrades(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(*args: object, **kwargs: object) -> object:
        raise ToolExecutionError(["ffmpeg"], 1, "simulated scdet failure")

    monkeypatch.setattr(scdet, "run_tool", boom)
    assert scdet.scdet_scores(Path("whatever.mp4")) == {}


def test_scdet_unexpected_bug_propagates(monkeypatch: pytest.MonkeyPatch) -> None:
    def bug(*args: object, **kwargs: object) -> object:
        raise ValueError("programming error")

    monkeypatch.setattr(scdet, "run_tool", bug)
    with pytest.raises(ValueError, match="programming error"):
        scdet.scdet_scores(Path("whatever.mp4"))


# --------------------------------------- FIX 5: overall status propagation

@requires_ffmpeg
def test_overall_status_review_required_propagates(tmp_path: Path) -> None:
    clip = _lavfi_filtered(
        "st_whiteflash.mp4",
        f"testsrc2=duration=2:rate=24:size={SIZE}",
        "drawbox=enable='eq(n\\,24)':color=white:t=fill",
    )
    result = run_audit(clip, artifacts_root=tmp_path, shot_analysis=True,
                       extract_shot_evidence=False)
    assert result.shot_truth is not None
    assert result.shot_truth.overall_status == "REVIEW_REQUIRED"
    assert result.status == RunStatus.REVIEW_REQUIRED
    assert result.qc is not None
    assert result.qc.status == RunStatus.REVIEW_REQUIRED


@requires_ffmpeg
def test_overall_status_pass_when_shot_truth_passes(tmp_path: Path) -> None:
    clip = _lavfi_concat(
        "st_hardcut.mp4",
        [f"testsrc2=duration=1:rate=24:size={SIZE}",
         f"smptebars=duration=1:rate=24:size={SIZE}"],
    )
    result = run_audit(clip, artifacts_root=tmp_path, shot_analysis=True,
                       extract_shot_evidence=False)
    assert result.shot_truth is not None
    assert result.shot_truth.overall_status == "PASS"
    assert result.status == RunStatus.PASS
