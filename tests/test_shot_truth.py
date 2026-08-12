"""Shot Truth Engine tests: exact-frame boundary recovery, short-shot survival,
flash/pan/occlusion/zoom false-positive defense, fades, dissolves, fps/VFR.

All fixtures are synthetic lavfi clips with known edit points; boundary
assertions are EXACT frame indexes and exact rational PTS times, never ±0.1 s.
"""

from __future__ import annotations

import itertools
from fractions import Fraction
from pathlib import Path

import pytest

from manuscript_reviewer.media.ffmpeg_tools import find_tool, run_tool
from manuscript_reviewer.media.frames import enumerate_frames
from manuscript_reviewer.media.probe import probe_media
from manuscript_reviewer.models.shot_truth import (
    CandidateStatus,
    ReasonCode,
    ShotTruthResult,
)
from manuscript_reviewer.shots.engine import run_shot_analysis
from tests.conftest import FIXTURES_DIR, requires_ffmpeg

pytestmark = requires_ffmpeg

SIZE = "320x240"


def _make(name: str, args: list[str]) -> Path:
    path = FIXTURES_DIR / name
    if path.exists():
        return path
    FIXTURES_DIR.mkdir(exist_ok=True)
    ffmpeg = find_tool("ffmpeg")
    run_tool(ffmpeg, ["-v", "error", *args, "-c:v", "libx264", "-pix_fmt", "yuv420p",
                      "-y", str(path)])
    return path


def _lavfi_concat(name: str, sources: list[str]) -> Path:
    inputs: list[str] = []
    for source in sources:
        inputs += ["-f", "lavfi", "-i", source]
    labels = "".join(f"[{i}:v]" for i in range(len(sources)))
    return _make(
        name,
        [*inputs, "-filter_complex", f"{labels}concat=n={len(sources)}:v=1:a=0"],
    )


def _lavfi_filtered(name: str, source: str, vf: str) -> Path:
    return _make(name, ["-f", "lavfi", "-i", source, "-vf", vf])


def analyze(path: Path, tmp_path: Path, extract_evidence: bool = False) -> ShotTruthResult:
    media, _ = probe_media(path)
    ledger = enumerate_frames(path, media.video_streams[0].time_base)
    run_dir = tmp_path / path.stem
    run_dir.mkdir(parents=True, exist_ok=True)
    output = run_shot_analysis(
        path, run_dir, media, ledger, extract_evidence=extract_evidence
    )
    assert output.result is not None, f"shot analysis failed: {output.issues}"
    assert not [i for i in output.issues if i.severity.value == "FAIL"], output.issues
    return output.result


def supported(result: ShotTruthResult) -> list:
    return [c for c in result.candidates if c.status == CandidateStatus.SUPPORTED]


def review(result: ShotTruthResult) -> list:
    return [c for c in result.candidates if c.status == CandidateStatus.REVIEW_REQUIRED]


# ---------------------------------------------------------------- fixtures

def test_static_shot_no_boundaries(tmp_path: Path) -> None:
    clip = _make(
        "st_static.mp4",
        ["-f", "lavfi", "-i", f"color=c=0x406080:duration=2:rate=24:size={SIZE}"],
    )
    result = analyze(clip, tmp_path)
    assert result.adjacent_pair_count == result.frame_count - 1
    assert supported(result) == []
    assert result.proposed_shot_count == 1
    assert result.overall_status == "PASS"


def test_continuous_motion_no_boundaries(tmp_path: Path) -> None:
    clip = _make(
        "st_motion.mp4",
        ["-f", "lavfi", "-i", f"testsrc2=duration=2:rate=24:size={SIZE}"],
    )
    result = analyze(clip, tmp_path)
    assert supported(result) == []


def test_clean_hard_cut_exact_frames(tmp_path: Path) -> None:
    clip = _lavfi_concat(
        "st_hardcut.mp4",
        [f"testsrc2=duration=1:rate=24:size={SIZE}",
         f"smptebars=duration=1:rate=24:size={SIZE}"],
    )
    result = analyze(clip, tmp_path)
    assert result.frame_count == 48
    assert result.adjacent_pair_count == 47
    boundaries = supported(result)
    assert len(boundaries) == 1
    boundary = boundaries[0]
    assert boundary.left_frame_index == 23
    assert boundary.right_frame_index == 24
    assert boundary.boundary_time_exact == Fraction(1)
    assert boundary.transition is not None
    assert boundary.transition.manuscript_type == "Hard cut"
    assert result.proposed_shot_count == 2
    shots = result.shots
    assert shots[0].transition_into_shot == "Opening shot"
    assert shots[0].start_frame_index == 0 and shots[0].end_frame_index == 23
    assert shots[1].start_frame_index == 24 and shots[1].end_frame_index == 47
    assert result.overall_status == "PASS"


def test_two_hard_cuts_both_recovered(tmp_path: Path) -> None:
    clip = _lavfi_concat(
        "st_twocuts.mp4",
        [f"testsrc2=duration=1:rate=24:size={SIZE}",
         f"smptebars=duration=1:rate=24:size={SIZE}",
         f"rgbtestsrc=duration=1:rate=24:size={SIZE}"],
    )
    result = analyze(clip, tmp_path)
    boundaries = sorted(supported(result), key=lambda c: c.left_frame_index)
    assert [(b.left_frame_index, b.right_frame_index) for b in boundaries] == [
        (23, 24),
        (47, 48),
    ]
    assert boundaries[0].boundary_time_exact == Fraction(1)
    assert boundaries[1].boundary_time_exact == Fraction(2)
    assert result.proposed_shot_count == 3


def test_very_short_shot_not_suppressed(tmp_path: Path) -> None:
    clip = _lavfi_concat(
        "st_shortshot.mp4",
        [f"testsrc2=duration=1:rate=24:size={SIZE}",
         f"smptebars=duration=0.125:rate=24:size={SIZE}",  # 3 frames
         f"rgbtestsrc=duration=1:rate=24:size={SIZE}"],
    )
    result = analyze(clip, tmp_path)
    non_rejected = [c for c in result.candidates if c.status != CandidateStatus.REJECTED]
    pairs = {(c.left_frame_index, c.right_frame_index) for c in non_rejected}
    assert (23, 24) in pairs, "entry boundary of 3-frame shot was suppressed"
    assert (26, 27) in pairs, "exit boundary of 3-frame shot was suppressed"


def test_white_flash_not_blindly_hard_cut(tmp_path: Path) -> None:
    clip = _lavfi_filtered(
        "st_whiteflash.mp4",
        f"testsrc2=duration=2:rate=24:size={SIZE}",
        "drawbox=enable='eq(n\\,24)':color=white:t=fill",
    )
    result = analyze(clip, tmp_path)
    assert supported(result) == []
    flagged = review(result)
    assert flagged, "white flash produced no review candidate"
    assert any(ReasonCode.FLASH_FRAME in c.reason_codes for c in flagged)
    assert result.overall_status == "REVIEW_REQUIRED"


def test_black_flash_not_blindly_hard_cut(tmp_path: Path) -> None:
    clip = _lavfi_filtered(
        "st_blackflash.mp4",
        f"testsrc2=duration=2:rate=24:size={SIZE}",
        "drawbox=enable='eq(n\\,24)':color=black:t=fill",
    )
    result = analyze(clip, tmp_path)
    assert supported(result) == []
    assert any(ReasonCode.FLASH_FRAME in c.reason_codes for c in review(result))


def test_fade_to_black_evidence(tmp_path: Path) -> None:
    clip = _lavfi_filtered(
        "st_fadeout.mp4",
        f"testsrc2=duration=2:rate=24:size={SIZE}",
        "fade=t=out:st=1.25:d=0.5",
    )
    result = analyze(clip, tmp_path)
    outs = [f for f in result.fades if f.direction == "out" and f.target_color == "black"]
    assert outs, "no fade-out evidence produced"
    fade = outs[0]
    assert 28 <= fade.transition_start_frame <= 34
    assert fade.transition_end_frame > fade.transition_start_frame
    assert supported(result) == []  # end-of-media fade creates no boundary
    assert result.proposed_shot_count == 1


def test_fade_from_black_evidence(tmp_path: Path) -> None:
    clip = _lavfi_filtered(
        "st_fadein.mp4",
        f"testsrc2=duration=2:rate=24:size={SIZE}",
        "fade=t=in:st=0:d=0.5",
    )
    result = analyze(clip, tmp_path)
    ins = [f for f in result.fades if f.direction == "in" and f.target_color == "black"]
    assert ins, "no fade-in evidence produced"
    assert ins[0].transition_start_frame <= 3
    assert supported(result) == []
    assert result.shots[0].transition_into_shot == "Opening shot"


def test_fade_out_then_in_creates_fade_boundary(tmp_path: Path) -> None:
    clip = _make(
        "st_fadecut.mp4",
        [
            "-f", "lavfi", "-i", f"testsrc2=duration=1:rate=24:size={SIZE}",
            "-f", "lavfi", "-i", f"smptebars=duration=1:rate=24:size={SIZE}",
            "-filter_complex",
            "[0:v]fade=t=out:st=0.6:d=0.35[a];"
            "[1:v]fade=t=in:st=0:d=0.35[b];[a][b]concat=n=2:v=1:a=0",
        ],
    )
    result = analyze(clip, tmp_path)
    boundaries = supported(result)
    assert len(boundaries) == 1
    boundary = boundaries[0]
    assert boundary.transition is not None
    assert boundary.transition.manuscript_type == "Fade in"
    assert result.proposed_shot_count == 2
    assert result.shots[1].transition_into_shot == "Fade in"


def test_cross_dissolve_conservative(tmp_path: Path) -> None:
    clip = _make(
        "st_dissolve.mp4",
        [
            "-f", "lavfi", "-i", f"testsrc2=duration=1.5:rate=24:size={SIZE}",
            "-f", "lavfi", "-i", f"smptebars=duration=1.5:rate=24:size={SIZE}",
            "-filter_complex", "[0:v][1:v]xfade=transition=fade:duration=0.5:offset=0.75",
        ],
    )
    result = analyze(clip, tmp_path)
    hard_cuts = [
        c
        for c in supported(result)
        if c.transition is not None and c.transition.manuscript_type == "Hard cut"
    ]
    assert hard_cuts == [], "dissolve must not be classified as a hard cut"
    assert review(result) or result.blends, "dissolve left no evidence at all"
    if review(result):
        assert result.overall_status == "REVIEW_REQUIRED"


def test_rapid_pan_not_a_cut(tmp_path: Path) -> None:
    clip = _lavfi_filtered(
        "st_pan.mp4",
        "testsrc2=duration=2:rate=24:size=1280x240",
        "crop=320:240:x='(iw-320)*t/2':y=0",
    )
    result = analyze(clip, tmp_path)
    assert supported(result) == [], "rapid pan misclassified as cut"


def test_camera_shake_not_a_cut(tmp_path: Path) -> None:
    clip = _lavfi_filtered(
        "st_shake.mp4",
        "testsrc2=duration=2:rate=24:size=640x480",
        "crop=320:240:x='160+140*sin(40*t)':y='120+100*sin(31*t)'",
    )
    result = analyze(clip, tmp_path)
    assert supported(result) == [], "camera shake misclassified as cut"


def test_large_occlusion_not_a_cut(tmp_path: Path) -> None:
    clip = _lavfi_filtered(
        "st_occlusion.mp4",
        f"testsrc2=duration=2:rate=24:size={SIZE}",
        "drawbox=enable='between(n\\,24\\,28)':color=orange:t=fill",
    )
    result = analyze(clip, tmp_path)
    assert supported(result) == [], "occlusion misclassified as cut"
    # Pixel evidence cannot distinguish an occlusion from a jump cut back to a
    # similar composition, so the anomaly stays visible as REVIEW_REQUIRED.
    assert any(
        ReasonCode.RETURN_TO_PREVIOUS_STATE in c.reason_codes for c in review(result)
    )


def test_zoom_not_a_cut(tmp_path: Path) -> None:
    clip = _lavfi_filtered(
        "st_zoom.mp4",
        f"testsrc2=duration=2:rate=24:size={SIZE}",
        "zoompan=z='min(1+in/12,5)':d=1:x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)'"
        f":s={SIZE}:fps=24",
    )
    result = analyze(clip, tmp_path)
    assert supported(result) == [], "zoom misclassified as cut"


def test_similar_composition_cut_survives(tmp_path: Path) -> None:
    clip = _make(
        "st_jumpcut.mp4",
        [
            "-f", "lavfi", "-i", f"testsrc2=duration=6:rate=24:size={SIZE}",
            "-filter_complex",
            "[0:v]split[a][b];[a]trim=0:1,setpts=PTS-STARTPTS[s0];"
            "[b]trim=5:6,setpts=PTS-STARTPTS+1/TB[s1];[s0][s1]concat=n=2:v=1:a=0",
        ],
    )
    result = analyze(clip, tmp_path)
    at_boundary = [
        c
        for c in result.candidates
        if (c.left_frame_index, c.right_frame_index) == (23, 24)
        and c.status != CandidateStatus.REJECTED
    ]
    assert at_boundary, "similar-composition cut was suppressed"


def test_one_frame_editorial_insert_stays_visible(tmp_path: Path) -> None:
    clip = _lavfi_filtered(
        "st_oneframe.mp4",
        f"testsrc2=duration=2:rate=24:size={SIZE}",
        "drawbox=enable='eq(n\\,24)':color=red:t=fill",
    )
    result = analyze(clip, tmp_path)
    visible = [c for c in result.candidates if c.status != CandidateStatus.REJECTED]
    pairs = {(c.left_frame_index, c.right_frame_index) for c in visible}
    assert (23, 24) in pairs or (24, 25) in pairs, "1-frame insert was discarded"
    assert result.overall_status == "REVIEW_REQUIRED"


@pytest.mark.parametrize(
    ("rate", "seg_frames"),
    [
        ("24000/1001", 24),
        ("30000/1001", 30),
        ("60000/1001", 60),
        ("60", 60),
    ],
)
def test_hard_cut_exact_at_standard_rates(rate: str, seg_frames: int, tmp_path: Path) -> None:
    safe = rate.replace("/", "_")
    clip = _lavfi_concat(
        f"st_rate_{safe}.mp4",
        [f"testsrc2=duration=1:rate={rate}:size={SIZE}",
         f"smptebars=duration=1:rate={rate}:size={SIZE}"],
    )
    result = analyze(clip, tmp_path)
    boundaries = supported(result)
    assert len(boundaries) == 1
    boundary = boundaries[0]
    assert boundary.left_frame_index == seg_frames - 1
    assert boundary.right_frame_index == seg_frames
    rate_fraction = Fraction(rate) if "/" not in rate else Fraction(
        int(rate.split("/")[0]), int(rate.split("/")[1])
    )
    assert boundary.boundary_time_exact == seg_frames / rate_fraction


def test_vfr_hard_cut_exact_pts(tmp_path: Path) -> None:
    """Drop frames 10-13 before the cut: indexes shift, exact PTS must not."""
    clip = _make(
        "st_vfr.mp4",
        [
            "-f", "lavfi", "-i", f"testsrc2=duration=1:rate=24:size={SIZE}",
            "-f", "lavfi", "-i", f"smptebars=duration=1:rate=24:size={SIZE}",
            "-filter_complex",
            "[0:v][1:v]concat=n=2:v=1:a=0,select='not(between(n\\,10\\,13))'",
            "-fps_mode", "passthrough",
        ],
    )
    result = analyze(clip, tmp_path)
    assert result.frame_count == 44
    assert result.adjacent_pair_count == 43
    boundaries = supported(result)
    assert len(boundaries) == 1
    boundary = boundaries[0]
    # Original frame 24 is now enumeration index 20; its exact PTS time is
    # still exactly 1 s.
    assert boundary.left_frame_index == 19
    assert boundary.right_frame_index == 20
    assert boundary.boundary_time_exact == Fraction(1)


# ---------------------------------------------------------------- structure

def test_pair_records_are_complete_and_consecutive(tmp_path: Path) -> None:
    clip = _lavfi_concat(
        "st_hardcut.mp4",
        [f"testsrc2=duration=1:rate=24:size={SIZE}",
         f"smptebars=duration=1:rate=24:size={SIZE}"],
    )
    media, _ = probe_media(clip)
    ledger = enumerate_frames(clip, media.video_streams[0].time_base)
    run_dir = tmp_path / "pairs"
    run_dir.mkdir()
    output = run_shot_analysis(clip, run_dir, media, ledger, extract_evidence=False)
    assert output.result is not None
    csv_lines = (run_dir / "adjacent_metrics.csv").read_text().strip().splitlines()
    assert len(csv_lines) == ledger.frame_count  # header + N-1 rows
    jsonl_lines = (run_dir / "adjacent_metrics.jsonl").read_text().strip().splitlines()
    assert len(jsonl_lines) == ledger.frame_count - 1


def test_shot_proposals_gapless_and_evidence_bundles(tmp_path: Path) -> None:
    clip = _lavfi_concat(
        "st_twocuts.mp4",
        [f"testsrc2=duration=1:rate=24:size={SIZE}",
         f"smptebars=duration=1:rate=24:size={SIZE}",
         f"rgbtestsrc=duration=1:rate=24:size={SIZE}"],
    )
    media, _ = probe_media(clip)
    ledger = enumerate_frames(clip, media.video_streams[0].time_base)
    run_dir = tmp_path / "gapless"
    run_dir.mkdir()
    output = run_shot_analysis(clip, run_dir, media, ledger, extract_evidence=True)
    result = output.result
    assert result is not None
    shots = result.shots
    assert shots[0].start_frame_index == 0
    assert shots[-1].end_frame_index == ledger.frame_count - 1
    for prev, current in itertools.pairwise(shots):
        assert current.start_frame_index == prev.end_frame_index + 1
    for candidate in supported(result):
        assert candidate.evidence_refs, "supported boundary missing evidence bundle"
        bundle = run_dir / candidate.evidence_refs[0]
        assert bundle.is_file()
    assert (run_dir / "shot_qc.json").is_file()
    assert (run_dir / "shots_proposed.json").is_file()
    assert (run_dir / "transition_evidence.json").is_file()
