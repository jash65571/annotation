"""Timing, shot-preservation, audio-layering and rendering guarantees.

Each test pins a property the audit found structurally violated: short shots
deleted by design, timestamps derived from a resampled grid, music erased by the
dialogue over it, and a renderer emitting non-canonical field names.
"""

from __future__ import annotations

from pathlib import Path

from autoscribe import cuts, render
from autoscribe.audio_timeline import AudioSpan, describe
from autoscribe.blockers import BlockerLog
from autoscribe.frames import GridFrame, pick_source_frames
from autoscribe.structured import Annotation, Entity, Globals, Shot, Timed


# --------------------------------------------------------------------------
# short shots survive
# --------------------------------------------------------------------------
def test_candidate_clustering_keeps_two_close_boundaries() -> None:
    """A 0.25s shot has boundaries 0.25s apart. The old 0.4s merge collapsed
    them into one, which DELETED the shot no matter what the footage showed."""
    raw = [1.00, 1.25]
    merged: list[float] = []
    for t in raw:
        if not merged or t - merged[-1] > cuts.FRAME_EPSILON:
            merged.append(t)
    assert merged == raw


def test_frame_epsilon_is_below_any_plausible_shot_length() -> None:
    assert cuts.FRAME_EPSILON <= 0.1


def test_audio_crossing_a_boundary_is_detected() -> None:
    spans = [AudioSpan(0.0, 5.0, "music"), AudioSpan(0.0, 1.0, "quiet")]
    assert cuts.audio_crosses(spans, 2.5) is True


def test_quiet_does_not_count_as_audio_crossing() -> None:
    assert cuts.audio_crosses([AudioSpan(0.0, 5.0, "quiet")], 2.5) is False


def test_audio_not_crossing_is_not_detected() -> None:
    spans = [AudioSpan(0.0, 2.0, "music"), AudioSpan(2.6, 5.0, "music")]
    assert cuts.audio_crosses(spans, 2.3) is False


def test_detectors_do_not_impose_a_minimum_shot_length() -> None:
    """PySceneDetect defaults every detector to min_scene_len=15 FRAMES (0.6s at
    25fps). With that floor a one-frame flash yields only its ENTRY boundary and
    never its exit, so the shot is silently absorbed into the next one — no
    downstream care can recover a candidate that was never proposed."""
    assert cuts.MIN_SCENE_LEN == 1


def test_densify_pulls_real_frames_around_a_candidate() -> None:
    """A ~10Hz grid cannot contain a one-frame event at 25fps, so verifying
    against the grid alone shows the model frames that never held the shot."""
    ledger = [i * 0.04 for i in range(50)]
    grid = [
        GridFrame(index=k, time_seconds=ledger[i], path=Path(f"{i}.png"),
                  source_index=i)
        for k, i in enumerate(range(0, 50, 3))
    ]
    # Frame 25 (1.00s) is absent from a 3-frame-stride grid.
    assert 25 not in {f.source_index for f in grid}

    calls: list[list[int]] = []

    def fake_extract(_v: Path, _o: Path, indices: list[int], times: list[float],
                     width: int = 768) -> list[GridFrame]:
        calls.append(indices)
        return [
            GridFrame(index=-1, time_seconds=times[i], path=Path(f"d{i}.png"),
                      source_index=i)
            for i in indices
        ]

    original = cuts.extract_indices
    try:
        cuts.extract_indices = fake_extract  # type: ignore[assignment]
        dense = cuts.densify(Path("v.mp4"), grid, [1.00], ledger, Path("w"), radius=3)
    finally:
        cuts.extract_indices = original  # type: ignore[assignment]

    assert calls, "no on-demand extraction was requested"
    assert 25 in calls[0], "the candidate's own frame was not fetched"
    assert 25 in {f.source_index for f in dense}
    assert dense == sorted(dense, key=lambda f: f.time_seconds)


def test_densify_is_a_noop_without_a_ledger() -> None:
    grid = [GridFrame(index=0, time_seconds=0.0, path=Path("a.png"), source_index=0)]
    assert cuts.densify(Path("v.mp4"), grid, [1.0], [], Path("w")) is grid
    assert cuts.densify(Path("v.mp4"), grid, [], [0.0, 0.04], Path("w")) is grid


def test_lj_cuts_are_not_offerable_to_the_vision_model() -> None:
    """The frame verifier only sees images, so it must not be able to answer
    with a transition that is defined by sound."""
    assert "L-cut" not in cuts.PICTURE_CUT_TYPES
    assert "J-cut" not in cuts.PICTURE_CUT_TYPES
    assert "L-cut" in cuts.CUT_TYPES and "J-cut" in cuts.CUT_TYPES


# --------------------------------------------------------------------------
# PTS-anchored timing
# --------------------------------------------------------------------------
def test_picked_frames_are_real_source_frames() -> None:
    """Every chosen index must name an actual encoded frame, and the times must
    come from the ledger — never from index/hz arithmetic."""
    frame_times = [0.0, 0.033, 0.067, 0.1, 0.133, 0.167, 0.2]
    picked = pick_source_frames(frame_times, duration=0.2, hz=10.0)
    assert picked, "expected at least one frame"
    assert all(0 <= i < len(frame_times) for i in picked)
    chosen = [frame_times[i] for i in picked]
    assert chosen == sorted(set(chosen)), "frames must be unique and ordered"


def test_variable_frame_rate_times_are_not_invented() -> None:
    """On VFR media a 1/hz grid names moments that do not exist. The sampler
    must snap to real frames instead."""
    frame_times = [0.0, 0.5, 0.52, 0.54, 2.0]
    picked = pick_source_frames(frame_times, duration=2.0, hz=10.0)
    chosen = [frame_times[i] for i in picked]
    assert all(t in frame_times for t in chosen)


def test_no_duplicate_frames_when_source_is_slower_than_grid() -> None:
    frame_times = [0.0, 1.0, 2.0]
    picked = pick_source_frames(frame_times, duration=2.0, hz=10.0)
    assert len(picked) == len(set(picked))


def test_grid_frame_carries_an_evidence_pointer() -> None:
    f = GridFrame(index=3, time_seconds=1.234, path=Path("x.png"), source_index=37)
    assert "n=37" in f.evidence()
    assert "1.234" in f.evidence()


# --------------------------------------------------------------------------
# FFmpeg command construction
# --------------------------------------------------------------------------
def test_modern_and_legacy_extract_commands_are_both_well_formed() -> None:
    """Regression: the legacy fallback was built by splicing a list in place and
    replaced the wrong two elements, emitting `-fps_mode -vsync 0 0` — a command
    that fails on every FFmpeg ever built, so old-FFmpeg users got no fallback
    at all, only a confusing second error."""
    from autoscribe.frames import _extract_cmd

    modern = _extract_cmd("ffmpeg", Path("in.mp4"), Path("o%06d.png"), [0, 2], 768,
                          legacy=False)
    legacy = _extract_cmd("ffmpeg", Path("in.mp4"), Path("o%06d.png"), [0, 2], 768,
                          legacy=True)

    assert "-fps_mode" in modern and modern[modern.index("-fps_mode") + 1] == "passthrough"
    assert "-vsync" not in modern

    assert "-vsync" in legacy and legacy[legacy.index("-vsync") + 1] == "0"
    assert "-fps_mode" not in legacy, "legacy command must not keep the removed flag"

    for cmd in (modern, legacy):
        assert cmd[-1].endswith(".png")
        assert cmd[cmd.index("-start_number") + 1] == "0"
        assert len(cmd) == len(modern), "both variants take the same argument count"


# --------------------------------------------------------------------------
# frame-period boundary resolution
# --------------------------------------------------------------------------
def test_frame_epsilon_is_derived_from_the_media() -> None:
    """A fixed 0.08s constant erases a real one-frame shot at 25fps (0.04s)."""
    at_25fps = cuts.frame_epsilon([i * 0.04 for i in range(50)])
    assert at_25fps < 0.04, "must be below one frame period so 1-frame shots survive"
    at_60fps = cuts.frame_epsilon([i / 60 for i in range(100)])
    assert at_60fps < at_25fps, "higher frame rate must resolve finer"


def test_frame_epsilon_uses_the_ledger_not_the_sampled_grid() -> None:
    """Regression: epsilon was computed from the 10 Hz grid handed to the vision
    model, so on 25 fps footage it returned 0.072s instead of the true 0.040s —
    one-frame shots stayed undetectable despite the fix that was meant to save
    them."""
    ledger = [i * 0.04 for i in range(100)]
    sampled = [ledger[i] for i in pick_source_frames(ledger, 3.96, 10.0)]

    from_ledger = cuts.frame_epsilon(ledger)
    from_grid = cuts.frame_epsilon(sampled)

    assert from_ledger < 0.04
    assert from_grid > 0.04, "the sampled grid cannot see the real frame period"
    assert from_ledger < from_grid


def test_frame_epsilon_falls_back_when_ledger_is_unusable() -> None:
    assert cuts.frame_epsilon([]) == cuts.FRAME_EPSILON
    assert cuts.frame_epsilon([0.0]) == cuts.FRAME_EPSILON


def test_frame_epsilon_survives_one_anomalous_gap() -> None:
    """A single duplicated/jittered timestamp must not collapse the threshold."""
    ledger = [i * 0.04 for i in range(60)]
    ledger.append(ledger[10] + 0.0001)  # container jitter
    assert cuts.frame_epsilon(sorted(ledger)) > 0.01


def test_one_frame_shot_survives_deduplication_at_25fps() -> None:
    """Two boundaries one frame apart are two boundaries, not one."""
    eps = cuts.frame_epsilon([i * 0.04 for i in range(60)])
    confirmed = [(1.00, "Hard cut"), (1.04, "Hard cut")]
    deduped: list[tuple[float, str]] = []
    for c in confirmed:
        if deduped and c[0] - deduped[-1][0] < eps:
            continue
        deduped.append(c)
    assert len(deduped) == 2, "a genuine one-frame shot was erased"


# --------------------------------------------------------------------------
# timestamps snap to frames that exist
# --------------------------------------------------------------------------
def test_timestamps_snap_to_real_frames() -> None:
    from autoscribe.structured import _snap

    frames = [0.0, 0.064, 0.128, 0.192]
    assert _snap(0.07, frames) == 0.064
    assert _snap(0.13, frames) == 0.128
    assert _snap(0.0, frames) == 0.0


def test_snap_falls_back_to_rounding_without_a_ledger() -> None:
    from autoscribe.structured import _snap

    assert _snap(1.2345, []) == 1.2


# --------------------------------------------------------------------------
# audio layers overlap
# --------------------------------------------------------------------------
def test_music_and_speech_coexist_in_the_timeline() -> None:
    """Music under dialogue must appear as BOTH; the old composer subtracted
    speech out of the music region and the bed vanished."""
    spans = [AudioSpan(0.0, 10.0, "music"), AudioSpan(2.0, 4.0, "speech")]
    text = describe(spans)
    assert "OVERLAP" in text
    assert "0.0s-10.0s" in text and "2.0s-4.0s" in text


def test_unresolved_audio_class_states_uncertainty() -> None:
    assert "could NOT be determined" in AudioSpan(0.0, 1.0, "unresolved").describe()


def test_coarse_sound_class_does_not_claim_a_kind() -> None:
    desc = AudioSpan(0.0, 1.0, "sound").describe()
    assert "UNDETERMINED" in desc
    assert "does not distinguish" in desc


# --------------------------------------------------------------------------
# canonical rendering
# --------------------------------------------------------------------------
def _annotation() -> Annotation:
    shot = Shot(
        index=1, cut="Opening shot", start=0.0, end=2.0, shot_type="medium shot",
        camera="Medium, eye-level, handheld",
        camera_movements=[Timed(0.0, 1.0, "Camera pans left")],
        scene="No changes from overview",
        actions=[Timed(0.0, 1.0, "C1 raises the right hand")],
        playback_speed="regular",
        speed_changes=[Timed(1.0, 2.0, "Footage ramps into slow motion")],
    )
    return Annotation(
        video_name="clip.mp4", duration=2.0,
        globals=Globals(
            characters=[Entity("C1", "A person in a red jacket")],
            scene=(
                "A domestic kitchen shot along its length. In the foreground a "
                "rectangular oak table with turned legs occupies the lower third. "
                "In the middle ground C1 stands behind the table facing camera, "
                "with open pine shelving on screen-left and a steel refrigerator "
                "on screen-right. The background is a plastered wall with a deep "
                "sash window above the counter run, beyond which a brick garden "
                "wall is visible."
            ),
            style=(
                "Daylight key from the sash window at screen-left with soft "
                "overhead fill; shadows are soft-edged and shallow. Colour "
                "temperature is cool toward the window and warmer near the "
                "shelving. Shallow depth of field, digital capture, no "
                "non-standard aspect ratio."
            ),
            audio="Music throughout",
        ),
        shots=[shot],
    )


def test_renderer_uses_canonical_field_names() -> None:
    text = render.render(_annotation())
    assert "Cast:" in text and "Characters:" not in text
    assert "Camera Movements:" in text and "Camera movements:" not in text
    assert "Playback Speed:" in text and "Video playback speed:" not in text
    assert "Speed Changes:" in text
    # Source-of-truth §26 capitalises both words.
    assert "Visual Concerns:" in text and "Visual concerns:" not in text
    assert "Audio Concerns:" in text and "Audio concerns:" not in text


def test_renderer_uses_seconds_in_shot_headers() -> None:
    """Headers used clock time while every action line used seconds — one
    caption speaking two time languages."""
    text = render.render(_annotation())
    assert "[Shot 1: 0.0s–2.0s]" in text
    assert "00:00:" not in text


def test_renderer_adds_terminal_punctuation_deterministically() -> None:
    text = render.render(_annotation())
    assert "(0.0s–1.0s) C1 raises the right hand." in text
    assert "Cut: Opening shot." in text


def test_renderer_omits_empty_action_section() -> None:
    ann = _annotation()
    ann.shots[0].actions = []
    text = render.render(ann)
    assert "Action & Audio:" not in text


def test_renderer_omits_speed_changes_when_absent() -> None:
    ann = _annotation()
    ann.shots[0].speed_changes = []
    assert "Speed Changes:" not in render.render(ann)


def test_rendered_output_passes_its_own_validator() -> None:
    """The renderer and the gate must agree; otherwise every clean run reports
    formatting blockers against itself."""
    from autoscribe.validate import validate_caption

    log: BlockerLog = validate_caption(render.render(_annotation()))
    assert log.blocking == [], [b.describe() for b in log.blocking]
