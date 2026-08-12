"""Phase 4 camera-motion tests on deterministic synthetic clips.

Proves the camera-vs-subject defense and phase segmentation:
- global background shift  -> camera-motion candidate
- moving object, static bg -> NOT a pan
- direction reversal       -> two separate phases
- uniform scale            -> scale-change candidate (never a pan)
- static                   -> STATIC only

Assertions are on the derived motion class / direction / exact phase frames —
never on raw float magnitudes — so they survive cross-platform FFT differences.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from manuscript_reviewer.camera.segmentation import analyze_camera_motion
from manuscript_reviewer.media import frames as frames_mod
from manuscript_reviewer.media import probe as probe_mod
from manuscript_reviewer.media.clock import AnnotationClock
from manuscript_reviewer.models.frame import FrameLedger
from manuscript_reviewer.models.review_intelligence import CameraMotionCandidate, CameraMotionClass
from manuscript_reviewer.validation.visual_validator import count_direction_reversals
from manuscript_reviewer.visual.decode import FrameCache

from .conftest import requires_ffmpeg, synth_clip

_W, _H = 320, 180
_MOVEMENT = {
    CameraMotionClass.HORIZONTAL_GLOBAL_MOTION,
    CameraMotionClass.VERTICAL_GLOBAL_MOTION,
    CameraMotionClass.DIAGONAL_GLOBAL_MOTION,
}


def _texture() -> np.ndarray:
    rng = np.random.default_rng(7)
    gray = rng.integers(0, 256, size=(_H, _W), dtype=np.uint8)
    return np.stack([gray, gray, gray], axis=2)


def _to_bgr(gray2d: np.ndarray) -> np.ndarray:
    return np.stack([gray2d, gray2d, gray2d], axis=2)


def _build_ledger(path: Path) -> FrameLedger:
    media, _ = probe_mod.probe_media(path)
    stream = media.video_streams[0]
    return frames_mod.enumerate_frames(path, stream.time_base, stream_index=0)


def _analyze(path: Path) -> list[CameraMotionCandidate]:
    ledger = _build_ledger(path)
    cache = FrameCache(path, ledger)
    clock = AnnotationClock.from_ledger(ledger)
    candidates, _raw_pairs = analyze_camera_motion(cache.gray_frames(), ledger, clock, None)
    return candidates


@requires_ffmpeg
def test_global_horizontal_shift_is_camera_motion(tmp_path: Path) -> None:
    base = _texture()[:, :, 0]
    frames = [_to_bgr(np.roll(base, shift=i * 3, axis=1)) for i in range(18)]
    clip = synth_clip(tmp_path / "hshift.mp4", frames)
    candidates = _analyze(clip)
    horiz = [c for c in candidates if c.motion_class == CameraMotionClass.HORIZONTAL_GLOBAL_MOTION]
    assert horiz, [c.motion_class.value for c in candidates]
    # A movement phase must be supported by a real global correlation.
    assert all(c.inlier_ratio > 0 for c in horiz)


@requires_ffmpeg
def test_moving_object_static_background_is_not_a_pan(tmp_path: Path) -> None:
    base = _texture()[:, :, 0]
    frames = []
    for i in range(18):
        frame = base.copy()
        x = 10 + i * 12
        frame[70:110, x : x + 30] = 255  # small bright object crossing frame
        frames.append(_to_bgr(frame))
    clip = synth_clip(tmp_path / "object.mp4", frames)
    candidates = _analyze(clip)
    movement = [c for c in candidates if c.motion_class in _MOVEMENT]
    assert not movement, [(c.motion_class.value, c.direction) for c in candidates]


@requires_ffmpeg
def test_direction_reversal_splits_into_two_phases(tmp_path: Path) -> None:
    base = _texture()[:, :, 0]
    frames = []
    for i in range(20):
        shift = i * 3 if i < 10 else (10 * 3 - (i - 9) * 3)
        frames.append(_to_bgr(np.roll(base, shift=shift, axis=1)))
    clip = synth_clip(tmp_path / "reversal.mp4", frames)
    candidates = _analyze(clip)
    horiz = [c for c in candidates if c.motion_class == CameraMotionClass.HORIZONTAL_GLOBAL_MOTION]
    directions = {c.direction for c in horiz}
    assert len(horiz) >= 2, [c.direction for c in horiz]
    assert len(directions) >= 2  # opposite directions -> separate phases
    assert count_direction_reversals(candidates) >= 1


@requires_ffmpeg
def test_uniform_scale_is_scale_change_not_pan(tmp_path: Path) -> None:
    import cv2

    base = _texture()[:, :, 0]
    frames = []
    for i in range(18):
        zoom = 1.0 + i * 0.05  # sustained zoom that survives phase smoothing
        cw, ch = int(_W / zoom), int(_H / zoom)
        x0, y0 = (_W - cw) // 2, (_H - ch) // 2
        crop = base[y0 : y0 + ch, x0 : x0 + cw]
        resized = cv2.resize(crop, (_W, _H), interpolation=cv2.INTER_LINEAR)
        frames.append(_to_bgr(resized))
    clip = synth_clip(tmp_path / "zoom.mp4", frames)
    candidates = _analyze(clip)
    classes = {c.motion_class for c in candidates}
    # Never a pan; a scale-change candidate is expected.
    assert CameraMotionClass.HORIZONTAL_GLOBAL_MOTION not in classes
    assert CameraMotionClass.SCALE_INCREASE in classes, [c.motion_class.value for c in candidates]


@requires_ffmpeg
def test_short_reversal_survives_smoothing(tmp_path: Path) -> None:
    base = _texture()[:, :, 0]
    positions = []
    pos = 0
    for i in range(22):
        if i < 8:
            pos += 3  # pan one way
        elif i < 12:
            pos -= 3  # short reversal (4 frames)
        else:
            pos += 3  # pan back
        positions.append(pos)
    frames = [_to_bgr(np.roll(base, shift=p, axis=1)) for p in positions]
    clip = synth_clip(tmp_path / "short_reversal.mp4", frames)
    candidates = _analyze(clip)
    horiz = [c for c in candidates if c.motion_class == CameraMotionClass.HORIZONTAL_GLOBAL_MOTION]
    directions = {c.direction for c in horiz}
    assert len(directions) >= 2  # the short reversal was not smoothed away


@requires_ffmpeg
def test_phase_endpoint_is_frame_after_last_supporting(tmp_path: Path) -> None:
    base = _texture()[:, :, 0]
    frames = [_to_bgr(np.roll(base, shift=i * 3, axis=1)) for i in range(18)]
    clip = synth_clip(tmp_path / "endpoint.mp4", frames)
    ledger = _build_ledger(clip)
    cache = FrameCache(clip, ledger)
    clock = AnnotationClock.from_ledger(ledger)
    candidates, _ = analyze_camera_motion(cache.gray_frames(), ledger, clock, None)
    for cand in candidates:
        if cand.last_supporting_frame < ledger.frame_count - 1:
            # Interval end boundary is the frame AFTER the last supporting frame.
            assert cand.end_frame == cand.last_supporting_frame + 1


def _camera_candidate(motion_class: CameraMotionClass, direction: str | None) -> object:
    return CameraMotionCandidate(
        candidate_id="CAM-0001",
        shot_number=1,
        start_frame=0,
        last_supporting_frame=4,
        end_frame=5,
        motion_class=motion_class,
        direction=direction,
        strength=1.0,
        inlier_ratio=0.9,
    )


def test_seed_pan_partially_supported_by_horizontal_evidence() -> None:
    from manuscript_reviewer.models.review_intelligence import EvidenceStatus, SeedClaimType
    from manuscript_reviewer.seed.claims import extract_claims
    from manuscript_reviewer.seed.comparison import compare_seed
    from manuscript_reviewer.seed.parser import parse_seed_text

    seed = (
        "[Shot 1: 0.0-5.0]\nCut: Opening shot\n"
        "Camera Movements: 0.0-1.0: Camera pans screen-left.\n"
    )
    doc = parse_seed_text(seed)
    claims = extract_claims(doc)
    cand = _camera_candidate(CameraMotionClass.HORIZONTAL_GLOBAL_MOTION, "screen-left")
    from manuscript_reviewer.seed.comparison import VisualEvidence
    res = compare_seed(doc, claims, None, None, VisualEvidence(camera_candidates=[cand]))
    cm = next(c for c in res.claims if c.claim_type == SeedClaimType.CAMERA_MOVEMENT)
    assert cm.evidence_status == EvidenceStatus.PARTIALLY_SUPPORTED


def test_seed_dolly_never_proven_by_2d_motion() -> None:
    from manuscript_reviewer.models.review_intelligence import EvidenceStatus, SeedClaimType
    from manuscript_reviewer.seed.claims import extract_claims
    from manuscript_reviewer.seed.comparison import compare_seed
    from manuscript_reviewer.seed.parser import parse_seed_text

    seed = (
        "[Shot 1: 0.0-5.0]\nCut: Opening shot\n"
        "Camera Movements: 0.0-1.0: Camera dollies in toward the subject.\n"
    )
    doc = parse_seed_text(seed)
    claims = extract_claims(doc)
    cand = _camera_candidate(CameraMotionClass.SCALE_INCREASE, None)
    from manuscript_reviewer.seed.comparison import VisualEvidence
    res = compare_seed(doc, claims, None, None, VisualEvidence(camera_candidates=[cand]))
    cm = next(c for c in res.claims if c.claim_type == SeedClaimType.CAMERA_MOVEMENT)
    # 2D motion cannot prove a dolly -> stays UNRESOLVED (never SUPPORTED).
    assert cm.evidence_status == EvidenceStatus.UNRESOLVED


def _synthetic_motion(n: int = 12) -> np.ndarray:
    rng = np.random.default_rng(11)
    base = rng.integers(0, 256, size=(90, 160), dtype=np.uint8)
    return np.stack([np.roll(base, i * 3, axis=1) for i in range(n)], axis=0)


def _ledger_at(fps: tuple[int, int], n: int):  # type: ignore[no-untyped-def]
    from fractions import Fraction

    from manuscript_reviewer.models.frame import FrameLedger, FrameRecord
    tb = Fraction(fps[1], fps[0])  # seconds per frame as time_base-ish
    frames = [
        FrameRecord(frame_index=i, pts=i, pts_time_seconds=i * tb, key_frame=(i == 0),
                    width=160, height=90)
        for i in range(n)
    ]
    return FrameLedger(stream_index=0, time_base=tb, frames=frames)


def _shot_truth_end(n: int, end_exact) -> object:  # type: ignore[no-untyped-def]
    from manuscript_reviewer.models.shot_truth import (
        CandidateStatus,
        ShotProposal,
        ShotTruthResult,
        TransitionStatus,
    )
    prop = ShotProposal(
        shot_index=1, start_frame_index=0, end_frame_index=n - 1,
        start_exact=None, end_exact=end_exact, last_owned_frame_start_exact=None,
        start_manuscript=None, end_manuscript=None, transition_into_shot="Opening shot",
        transition_status=TransitionStatus.PROPOSED, supporting_boundary_id=None,
        review_status=CandidateStatus.SUPPORTED,
    )
    return ShotTruthResult(
        frame_count=n, adjacent_pair_count=0, raw_candidate_count=0, merged_candidate_count=0,
        supported_count=0, rejected_count=0, review_required_count=0, proposed_shot_count=1,
        overall_status="PASS", candidates=[], shots=[prop],
        annotation_endpoint_exact=end_exact,
    )


def test_camera_endpoint_uses_shot_end_not_final_frame_start() -> None:
    from fractions import Fraction

    from manuscript_reviewer.camera.segmentation import analyze_camera_motion
    from manuscript_reviewer.media.clock import AnnotationClock

    for fps in [(24, 1), (60, 1), (30000, 1001)]:  # 24, 60, 59.94
        n = 12
        gray = _synthetic_motion(n)
        ledger = _ledger_at(fps, n)
        clock = AnnotationClock.from_ledger(ledger)
        # Shot end is the media endpoint = presentation start of a frame AFTER the
        # last owned frame (distinct from the last frame's own start time).
        shot_end = ledger.frames[n - 1].pts_time_seconds + Fraction(fps[1], fps[0])
        candidates, _ = analyze_camera_motion(gray, ledger, clock, _shot_truth_end(n, shot_end))
        movement = [c for c in candidates if c.last_supporting_frame >= n - 1]
        assert movement, fps
        # A phase running through the final frame ends at the shot end, not the
        # final frame's own start time (would be one frame early).
        last = movement[-1]
        assert last.end_exact == shot_end
        assert last.end_exact != ledger.frames[n - 1].pts_time_seconds


def test_smoothing_preserves_target_class_when_absorbing_short_run() -> None:
    from manuscript_reviewer.camera.classify import PairClassification
    from manuscript_reviewer.camera.segmentation import _group_runs, _smooth

    def pair(left: int, cls: CameraMotionClass, direction: str | None) -> tuple:
        return (left, left + 1, PairClassification(cls, direction, 1.0, 0.9))

    static = CameraMotionClass.STATIC
    horiz = CameraMotionClass.HORIZONTAL_GLOBAL_MOTION

    # A: 1-pair STATIC then B: 8-pair screen-right movement. Absorbing A into B
    # must keep the phase screen-right (item 10 regression).
    seq_ab = [pair(0, static, None)] + [pair(i, horiz, "screen-right") for i in range(1, 9)]
    runs_ab = _smooth(_group_runs(seq_ab))
    assert len(runs_ab) == 1
    assert runs_ab[0].motion_class == horiz
    assert runs_ab[0].direction == "screen-right"

    # Inverse: 8-pair movement then 1-pair static -> still screen-right movement.
    seq_ba = [pair(i, horiz, "screen-right") for i in range(0, 8)] + [pair(8, static, None)]
    runs_ba = _smooth(_group_runs(seq_ba))
    assert len(runs_ba) == 1
    assert runs_ba[0].motion_class == horiz
    assert runs_ba[0].direction == "screen-right"


@requires_ffmpeg
def test_static_scene_is_static(tmp_path: Path) -> None:
    base = _texture()
    frames = [base.copy() for _ in range(12)]
    clip = synth_clip(tmp_path / "static.mp4", frames)
    candidates = _analyze(clip)
    movement = [c for c in candidates if c.motion_class in _MOVEMENT]
    assert not movement
