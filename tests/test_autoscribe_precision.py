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
        scene="A kitchen",
        actions=[Timed(0.0, 1.0, "C1 raises the right hand")],
        playback_speed="regular",
        speed_changes=[Timed(1.0, 2.0, "Footage ramps into slow motion")],
    )
    return Annotation(
        video_name="clip.mp4", duration=2.0,
        globals=Globals(
            characters=[Entity("C1", "A person in a red jacket")],
            scene="A kitchen", style="Natural light", audio="Music throughout",
        ),
        shots=[shot],
    )


def test_renderer_uses_canonical_field_names() -> None:
    text = render.render(_annotation())
    assert "Cast:" in text and "Characters:" not in text
    assert "Camera Movements:" in text and "Camera movements:" not in text
    assert "Playback Speed:" in text and "Video playback speed:" not in text
    assert "Speed Changes:" in text


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
