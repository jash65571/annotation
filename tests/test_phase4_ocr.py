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
    tracks = build_text_tracks(obs, last_inspected_frame=40)
    assert len(tracks) == 1
    track = tracks[0]
    assert track.first_candidate_frame == 5
    assert track.first_stable_frame == 5
    assert track.last_stable_frame == 24
    assert track.consecutive_support_frames == 20  # a real consecutive run (L)
    # Disappearance is NOT invented as last+1 (K): text ended before the last
    # inspected frame but we have no region-absence evidence -> UNRESOLVED.
    assert track.disappearance_frame is None
    assert track.disappearance_status.value == "UNRESOLVED"
    # Never caption-eligible without human verification.
    assert track.caption_text_eligible is False
    assert track.verification_status == SourceTextVerificationStatus.UNVERIFIED
    assert not validate_text_tracks(tracks)


def test_text_persisting_to_last_frame_is_not_a_disappearance() -> None:
    # Text present through the last inspected frame -> persists, no disappearance.
    obs = [_obs(f, "SCORE") for f in range(0, 10)]
    tracks = build_text_tracks(obs, last_inspected_frame=9)
    track = tracks[0]
    assert track.disappearance_frame is None
    assert track.text_persists_to_shot_end is True
    assert track.disappearance_status.value == "PERSISTS_TO_END"


def test_same_box_text_change_splits_tracks() -> None:
    # Same HUD box; text changes ROUND WON -> NEXT ROUND: two tracks, not one.
    box = (100, 20, 80, 15)
    obs = [_obs(f, "ROUND WON", box=box) for f in range(0, 4)]
    obs += [_obs(f, "NEXT ROUND", box=box) for f in range(4, 8)]
    tracks = build_text_tracks(obs, last_inspected_frame=8)
    texts = {t.consensus.consensus_text for t in tracks if t.consensus}
    assert texts == {"ROUND WON", "NEXT ROUND"}


def test_stability_requires_consecutive_run_not_two_gapped_observations() -> None:
    # Two observations far apart (frames 0 and 9) are not a stable run.
    obs = [_obs(0, "X"), _obs(9, "X")]
    tracks = build_text_tracks(obs, last_inspected_frame=20)
    # They do not even link (gap > max), so each is its own one-frame track.
    assert all(t.first_stable_frame is None for t in tracks)


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


def test_ocr_engine_failures_are_accounted_as_degraded() -> None:
    ledger = _ledger(10)
    clock = AnnotationClock.from_ledger(ledger)

    class _Flaky:
        def engine_info(self):  # type: ignore[no-untyped-def]
            from manuscript_reviewer.models.review_intelligence import OCREngineInfo

            return OCREngineInfo(engine="tesseract", status=OCRStatus.AVAILABLE)

        def recognize(self, image, language="eng"):  # type: ignore[no-untyped-def]
            raise RuntimeError("engine crashed")

    stream = [(i, np.zeros((10, 10), dtype=np.uint8)) for i in range(10)]
    result = run_ocr(stream, ledger, clock, _Flaky(), total_frames=10)
    # A total OCR failure is DEGRADED, never AVAILABLE-with-no-text.
    assert result.engine_info.status == OCRStatus.DEGRADED
    assert result.failed_frame_count == 10
    assert all(v == "OCR_ENGINE_FAILED" for v in result.frame_status.values())


def test_seed_on_screen_text_matched_by_ocr_is_partial_not_final() -> None:
    from manuscript_reviewer.models.review_intelligence import EvidenceStatus, SeedClaimType
    from manuscript_reviewer.seed.claims import extract_claims
    from manuscript_reviewer.seed.comparison import compare_seed
    from manuscript_reviewer.seed.parser import parse_seed_text

    tracks = build_text_tracks([_obs(f, "ENEMY KILLED") for f in range(5)], last_inspected_frame=5)
    seed = (
        "[Shot 1: 0.0-5.0]\nCut: Opening shot\n"
        'Action & Audio: 1.0-2.0: On-screen text reads "ENEMY KILLED".\n'
    )
    doc = parse_seed_text(seed)
    claims = extract_claims(doc)
    res = compare_seed(doc, claims, None, None, tracks)
    ost = next(c for c in res.claims if c.claim_type == SeedClaimType.ON_SCREEN_TEXT)
    # Machine OCR agreement is PARTIAL + review-required, never final SUPPORTED.
    assert ost.evidence_status == EvidenceStatus.PARTIALLY_SUPPORTED
    assert ost.review_status is not None and ost.review_status.value == "REVIEW_REQUIRED"


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
