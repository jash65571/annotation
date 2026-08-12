"""Phase 4 OCR-slice tests (deterministic, mock adapter — no system OCR).

Covers the sequential-OCR track model, temporal consensus, the one-frame
defense, the caption-eligibility gate (P4-OCR-001 / P4-TEXT-001), watermark
candidates, Unicode preservation, and safe degradation when OCR is unavailable.
"""

from __future__ import annotations

from fractions import Fraction

import numpy as np

from manuscript_reviewer.media.clock import AnnotationClock
from manuscript_reviewer.models.frame import FrameLedger, FrameRecord
from manuscript_reviewer.models.review_intelligence import (
    OCRObservation,
    OCRStatus,
    SourceTextVerificationStatus,
)
from manuscript_reviewer.ocr.adapter import MockOCRAdapter, OCRWord
from manuscript_reviewer.ocr.consensus import temporal_consensus
from manuscript_reviewer.ocr.pipeline import run_ocr
from manuscript_reviewer.ocr.timing import build_text_tracks, detect_watermark_candidates
from manuscript_reviewer.validation.review_intelligence_validator import validate_text_tracks


def _ledger(n: int) -> FrameLedger:
    frames = [
        FrameRecord(
            frame_index=i,
            pts=i * 100,
            pts_time_seconds=Fraction(i, 10),
            key_frame=(i == 0),
        )
        for i in range(n)
    ]
    return FrameLedger(stream_index=0, time_base=Fraction(1, 1000), frames=frames)


def _obs(frame: int, text: str, box: tuple[int, int, int, int] = (10, 10, 40, 12),
         conf: float | None = 90.0) -> OCRObservation:
    x, y, w, h = box
    return OCRObservation(
        observation_id=f"OCR-{frame}",
        frame_index=frame,
        source_pts_time_exact=Fraction(frame, 10),
        annotation_time_exact=Fraction(frame, 10),
        x=x, y=y, width=w, height=h,
        raw_text=text,
        confidence=conf,
    )


# --------------------------------------------------------------------------
# Consensus
# --------------------------------------------------------------------------


def test_temporal_consensus_majority_without_correcting_words() -> None:
    obs = [
        _obs(0, "ENEMY KILLED"),
        _obs(1, "ENEMY KILLED"),
        _obs(2, "ENEMY K1LLED"),
        _obs(3, "ENEMY KILLED"),
    ]
    consensus = temporal_consensus(obs)
    assert consensus is not None
    assert consensus.consensus_text == "ENEMY KILLED"
    assert consensus.support_frames == 3
    # Raw observations are retained by the caller (never mutated).
    assert obs[2].raw_text == "ENEMY K1LLED"


# --------------------------------------------------------------------------
# Text tracks: persistence, one-frame defense, caption-eligibility gate
# --------------------------------------------------------------------------


def test_persistent_text_track_has_exact_first_last_frames() -> None:
    obs = [_obs(f, "GO") for f in range(5, 25)]  # frames 5..24
    tracks = build_text_tracks(obs)
    assert len(tracks) == 1
    track = tracks[0]
    assert track.first_candidate_frame == 5
    assert track.first_stable_frame == 5
    assert track.last_stable_frame == 24
    assert track.disappearance_frame == 25
    # Never caption-eligible without human verification.
    assert track.caption_text_eligible is False
    assert track.verification_status == SourceTextVerificationStatus.UNVERIFIED
    assert not validate_text_tracks(tracks)


def test_one_frame_text_is_review_required_not_stable() -> None:
    tracks = build_text_tracks([_obs(7, "FLASH")])
    assert len(tracks) == 1
    track = tracks[0]
    assert track.first_stable_frame is None  # one isolated frame -> not stable
    assert track.last_stable_frame is None
    assert track.review_required is True


def test_caption_eligibility_gate_is_enforced() -> None:
    # Simulate a bug: an unverified track flagged caption-eligible must fail.
    tracks = build_text_tracks([_obs(f, "GO") for f in range(5, 10)])
    tracks[0].caption_text_eligible = True  # illegal without verification
    issues = validate_text_tracks(tracks)
    assert any(i.rule_id == "P4-OCR-001" for i in issues)


def test_unicode_text_is_preserved() -> None:
    obs = [_obs(f, "日本語テスト") for f in range(3)]
    tracks = build_text_tracks(obs)
    assert tracks[0].consensus is not None
    assert tracks[0].consensus.consensus_text == "日本語テスト"


# --------------------------------------------------------------------------
# Watermark candidates
# --------------------------------------------------------------------------


def test_persistent_corner_region_is_watermark_candidate() -> None:
    obs = [_obs(f, "LOGO", box=(0, 0, 30, 10)) for f in range(0, 20)]
    tracks = build_text_tracks(obs)
    candidates = detect_watermark_candidates(tracks, total_frames=20, min_persistence=0.8)
    assert candidates and candidates[0].review_required is True


def test_brief_text_is_not_watermark() -> None:
    obs = [_obs(f, "GO", box=(0, 0, 30, 10)) for f in range(0, 3)]
    tracks = build_text_tracks(obs)
    candidates = detect_watermark_candidates(tracks, total_frames=20, min_persistence=0.8)
    assert not candidates


# --------------------------------------------------------------------------
# Pipeline with the mock adapter + unavailable degradation
# --------------------------------------------------------------------------


def test_run_ocr_with_mock_builds_tracks() -> None:
    ledger = _ledger(6)
    clock = AnnotationClock.from_ledger(ledger)
    word = OCRWord(text="GO", x=10, y=10, width=20, height=10, confidence=95.0)
    adapter = MockOCRAdapter(scripted=[[word]])  # stable overlay every frame
    img = np.zeros((20, 40), dtype=np.uint8)
    stream = [(i, img) for i in range(6)]
    result = run_ocr(stream, ledger, clock, adapter, total_frames=6)
    assert result.engine_info.status == OCRStatus.AVAILABLE
    assert result.text_tracks
    assert result.text_tracks[0].consensus is not None
    # Every observation carries exact frame identity.
    for obs in result.observations:
        assert obs.source_pts_time_exact is not None


def test_run_ocr_unavailable_degrades() -> None:
    ledger = _ledger(3)
    clock = AnnotationClock.from_ledger(ledger)

    class _Unavailable:
        def engine_info(self):  # type: ignore[no-untyped-def]
            from manuscript_reviewer.models.review_intelligence import OCREngineInfo

            return OCREngineInfo(engine="tesseract", status=OCRStatus.UNAVAILABLE)

        def recognize(self, image, language="eng"):  # type: ignore[no-untyped-def]
            raise AssertionError("must not be called when unavailable")

    img = np.zeros((10, 10), dtype=np.uint8)
    result = run_ocr([(i, img) for i in range(3)], ledger, clock, _Unavailable())
    assert result.engine_info.status == OCRStatus.UNAVAILABLE
    assert not result.text_tracks
    assert not result.observations
