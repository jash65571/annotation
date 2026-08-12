"""Phase 3.1 audio evidence-hardening regressions.

These are deliberately ffmpeg-free pure-unit tests: they exercise the evidence
data model and pure functions directly, so they run in CI and locally without
ASR models or media. End-to-end ffmpeg fixtures live in test_audio_truth.py.
"""

from __future__ import annotations

from fractions import Fraction
from typing import Any

from manuscript_reviewer.audio.asr import runtime as asr_runtime
from manuscript_reviewer.audio.asr.runtime import (
    parse_alignment,
    parse_transcription,
    reconcile_best_transcript,
    resolve_cache_state,
)
from manuscript_reviewer.audio.boundary_audio import analyze_boundary_audio
from manuscript_reviewer.audio.review_queue import build_review_queue
from manuscript_reviewer.audio.speech import build_speech_regions
from manuscript_reviewer.media.clock import AnnotationClock
from manuscript_reviewer.models.audio import (
    AlignmentStatus,
    AudioEnergyBin,
    AudioReviewReason,
    AudioTimeline,
    BestWord,
    BoundaryAudioStatus,
    EvidenceState,
    LanguageEvidence,
    LanguageReviewStatus,
    ReviewPriority,
    SampleAnchorStatus,
    SourceVerificationStatus,
    SpeechRegion,
    WordTimingStatus,
)
from manuscript_reviewer.models.shot_truth import (
    BoundaryCandidate,
    CandidateStatus,
    LocalBaseline,
    PairMetrics,
)
from manuscript_reviewer.models.validation import Severity
from manuscript_reviewer.rules.loader import load_rules
from manuscript_reviewer.validation import audio_validator

# --------------------------------------------------------------- test helpers


def _timeline(offset: float = 0.0, duration: float = 3.0) -> AudioTimeline:
    return AudioTimeline(
        source_stream_index=0,
        source_time_base=Fraction(1, 48000),
        source_start_pts=0,
        source_start_time=Fraction(0),
        first_decoded_audio_pts=0,
        annotation_timeline_origin=Fraction(0),
        annotation_audio_offset=Fraction(offset),
        source_sample_rate=48000,
        source_channels=1,
        evidence_sample_rate=48000,
        evidence_channels=1,
        evidence_sample_count=int(duration * 48000),
        evidence_duration_seconds=Fraction(duration),
    )


def _fw_response(words: list[tuple[str, float, float]], *, language: str = "en",
                 language_probability: float = 0.97) -> dict[str, Any]:
    return {
        "status": "ok",
        "model": "mock",
        "language": language,
        "language_probability": language_probability,
        "segments": [
            {
                "id": 0,
                "start": str(words[0][1]),
                "end": str(words[-1][2]),
                "text": " ".join(w[0] for w in words),
                "words": [
                    {"text": w[0], "start": str(w[1]), "end": str(w[2]),
                     "probability": 0.95}
                    for w in words
                ],
            }
        ],
    }


def _wx_response(words: list[tuple[str, Any, Any]], full_text: str,
                 seg_start: Any = "0.0", seg_end: Any = "1.0") -> dict[str, Any]:
    """WhisperX response: segment text is the PRESERVED full FW text even when
    some words fail to align (that is exactly how WhisperX behaves)."""
    return {
        "status": "ok",
        "language": "en",
        "segments": [
            {
                "id": 0,
                "start": seg_start,
                "end": seg_end,
                "text": full_text,
                "words": [
                    {"text": w[0],
                     "start": None if w[1] is None else str(w[1]),
                     "end": None if w[2] is None else str(w[2]),
                     "probability": 0.99}
                    for w in words
                ],
            }
        ],
    }


def _bins(start: float, end: float, dbfs: float, centroid: float,
          flatness: float) -> list[AudioEnergyBin]:
    bins: list[AudioEnergyBin] = []
    step = 0.01
    n = round((end - start) / step)
    for i in range(n):
        s = Fraction(start) + Fraction(i, 100)
        e = s + Fraction(1, 100)
        bins.append(
            AudioEnergyBin(
                bin_index=i,
                start_sample=int(s * 48000),
                end_sample=int(e * 48000),
                start_annotation_time=s,
                end_annotation_time=e,
                rms=0.1,
                peak=0.1,
                dbfs=dbfs,
                zero_crossing_rate=0.1,
                spectral_centroid_hz=centroid,
                spectral_flatness=flatness,
            )
        )
    return bins


def _supported_candidate(boundary: float) -> BoundaryCandidate:
    metrics = PairMetrics(
        left_frame_index=0, right_frame_index=1, left_pts=0, right_pts=1,
        left_pts_time_seconds=Fraction(0), right_pts_time_seconds=Fraction(boundary),
        mean_abs_diff=10.0, hist_distance=0.5, phash_hamming=10, edge_change=0.1,
        luma_delta=1.0, flow_mean_mag=0.1, flow_coherence=0.1,
    )
    baseline = LocalBaseline(
        window_pairs=8, diff_median=1.0, diff_mad=1.0, diff_z=5.0,
        hist_median=0.1, hist_mad=0.1, hist_z=5.0, neighbor_motion=0.1,
    )
    return BoundaryCandidate(
        candidate_id="cand_0001", left_frame_index=0, right_frame_index=1,
        left_pts=0, right_pts=1, boundary_time_exact=Fraction(boundary),
        boundary_time_manuscript=f"{boundary:.1f}s", candidate_sources=["scdet"],
        metric_snapshot=metrics, local_baseline=baseline, candidate_score=5.0,
        status=CandidateStatus.SUPPORTED,
    )


def _best_word(seg: int, idx: int, text: str, start: float, end: float,
               status: WordTimingStatus = WordTimingStatus.WHISPERX_ALIGNED) -> BestWord:
    return BestWord(
        segment_id=seg, source_word_index=idx, text=text,
        start_annotation_time=Fraction(start), end_annotation_time=Fraction(end),
        asr_start_seconds=str(start), asr_end_seconds=str(end),
        probability=0.95, timing_source="whisperx", timing_status=status,
    )


def _analyze_boundary(bins: list[AudioEnergyBin], *, best_words: list[BestWord] | None = None,
                      speech_regions: list[SpeechRegion] | None = None,
                      asr_segments: list[Any] | None = None,
                      regions: list[Any] | None = None) -> Any:
    clock = AnnotationClock(origin=Fraction(0))
    evidence = analyze_boundary_audio(
        [_supported_candidate(1.0)], clock, _timeline(), bins,
        regions or [], speech_regions or [], best_words or [], asr_segments or [],
    )
    assert len(evidence) == 1
    return evidence[0]


# --------------------------------------------- §2/§3 mandatory source verification


def _speech_regions_from(fw: dict[str, Any], wx: dict[str, Any] | None = None,
                         offset: float = 0.0) -> tuple[list[SpeechRegion], Any]:
    result = parse_transcription(fw, Fraction(offset))
    alignment = (
        parse_alignment(wx, result.segments, Fraction(offset)) if wx is not None else None
    )
    best = reconcile_best_transcript(result, alignment)
    regions = build_speech_regions(
        result, best.best_words, best.status, best.coverage, _timeline(offset), None
    )
    return regions, best


def test_every_machine_speech_region_requires_source_verification() -> None:
    fw = _fw_response([("Hello", 0.3, 0.6), ("there", 0.7, 1.0)])
    wx = _wx_response([("Hello", 0.31, 0.62), ("there", 0.71, 1.02)], "Hello there")
    regions, best = _speech_regions_from(fw, wx)
    assert best.status == AlignmentStatus.ALIGNED  # clean, high-confidence
    assert regions
    for region in regions:
        assert region.source_verification_status == SourceVerificationStatus.UNVERIFIED
        assert (
            AudioReviewReason.MANDATORY_SOURCE_AUDIO_VERIFICATION in region.review_reasons
        )


def test_aligned_high_confidence_speech_still_enters_review_queue() -> None:
    fw = _fw_response([("Hello", 0.3, 0.6), ("there", 0.7, 1.0)])
    wx = _wx_response([("Hello", 0.31, 0.62), ("there", 0.71, 1.02)], "Hello there")
    regions, _ = _speech_regions_from(fw, wx)
    items = build_review_queue(_timeline(), regions, [], [], [], [], asr_unavailable=False)
    assert len(items) == 1
    item = items[0]
    assert AudioReviewReason.MANDATORY_SOURCE_AUDIO_VERIFICATION in item.reasons
    assert item.priority == ReviewPriority.CRITICAL
    # P3-SPEECH-004: the mandatory item satisfies the source-verification rule.
    assert not audio_validator.validate_source_verification(regions, items)


def test_top_level_pass_forbidden_with_unverified_dialogue() -> None:
    fw = _fw_response([("Hello", 0.3, 0.6), ("there", 0.7, 1.0)])
    wx = _wx_response([("Hello", 0.31, 0.62), ("there", 0.71, 1.02)], "Hello there")
    regions, _ = _speech_regions_from(fw, wx)
    # Even with NO review items and NO crossing boundaries, unverified speech
    # forbids PASS.
    status = audio_validator.compute_audio_status([], [], [], regions, no_audio=False)
    assert status == "REVIEW_REQUIRED"


def test_source_verified_speech_clears_review_condition() -> None:
    fw = _fw_response([("Hello", 0.3, 0.6), ("there", 0.7, 1.0)])
    wx = _wx_response([("Hello", 0.31, 0.62), ("there", 0.71, 1.02)], "Hello there")
    regions, _ = _speech_regions_from(fw, wx)
    for region in regions:
        region.source_verification_status = SourceVerificationStatus.HUMAN_VERIFIED
    items = build_review_queue(_timeline(), regions, [], [], [], [], asr_unavailable=False)
    assert items == []  # verified speech no longer demands a listen
    assert not audio_validator.validate_source_verification(regions, items)
    status = audio_validator.compute_audio_status([], items, [], regions, no_audio=False)
    assert status == "PASS"


def test_missing_source_verification_item_is_p3_speech_004_failure() -> None:
    fw = _fw_response([("Hello", 0.3, 0.6)])
    regions, _ = _speech_regions_from(fw)
    issues = audio_validator.validate_source_verification(regions, [])  # no items
    assert any(i.rule_id == "P3-SPEECH-004" and i.severity == Severity.FAIL for i in issues)


# --------------------------------------------------- §4/§23 caption-eligibility gate


def test_caption_text_eligibility_contract() -> None:
    region = SpeechRegion(
        region_id="s1", start_exact=Fraction(0), end_exact=Fraction(1),
        start_manuscript="0.0s", end_manuscript="1.0s", sources=["asr_faster_whisper"],
        text_candidate="hello there",
        language=LanguageEvidence(language_candidate="en", language_probability=0.99,
                                  language_source="faster_whisper",
                                  language_review_status=LanguageReviewStatus.SUPPORTED_BY_ASR),
    )
    # UNVERIFIED machine ASR text is evidence only — never caption-eligible.
    assert region.source_verification_status == SourceVerificationStatus.UNVERIFIED
    assert region.caption_text_eligible is False
    assert region.caption_text is None
    assert region.caption_language_eligible is False  # machine language guess

    region.source_verification_status = SourceVerificationStatus.HUMAN_VERIFIED
    assert region.caption_text_eligible is True
    assert region.caption_text == "hello there"  # human confirmed the ASR text

    region.source_verification_status = SourceVerificationStatus.HUMAN_CORRECTED
    assert region.caption_text_eligible is False  # no corrected text supplied yet
    region.corrected_text = "hello, there!"
    assert region.caption_text_eligible is True
    assert region.caption_text == "hello, there!"  # only the corrected text

    region.source_verification_status = SourceVerificationStatus.REJECTED
    assert region.caption_text_eligible is False
    assert region.caption_text is None


def test_caption_language_requires_human_verification() -> None:
    region = SpeechRegion(
        region_id="s1", start_exact=Fraction(0), end_exact=Fraction(1),
        start_manuscript="0.0s", end_manuscript="1.0s", sources=["asr_faster_whisper"],
        language=LanguageEvidence(language_candidate="en", language_probability=0.999,
                                  language_source="faster_whisper",
                                  language_review_status=LanguageReviewStatus.SUPPORTED_BY_ASR),
    )
    assert region.caption_language_eligible is False  # high prob is not verification
    assert region.language is not None
    region.language.language_review_status = LanguageReviewStatus.HUMAN_VERIFIED
    assert region.caption_language_eligible is True


# --------------------------------------------- §5-§10 partial alignment / best words


def test_partial_alignment_missing_first_word() -> None:
    fw = _fw_response([("one", 0.0, 0.2), ("two", 0.2, 0.4),
                       ("three", 0.4, 0.6), ("four", 0.6, 0.8)])
    wx = _wx_response(
        [("two", 0.21, 0.41), ("three", 0.41, 0.61), ("four", 0.61, 0.81)],
        "one two three four",
    )
    result = parse_transcription(fw, Fraction(0))
    alignment = parse_alignment(wx, result.segments, Fraction(0))
    best = reconcile_best_transcript(result, alignment)
    assert [w.text for w in best.best_words] == ["one", "two", "three", "four"]
    assert best.status == AlignmentStatus.PARTIAL
    assert best.coverage == 0.75
    first = best.best_words[0]
    assert first.timing_status == WordTimingStatus.FASTER_WHISPER_FALLBACK
    assert first.start_annotation_time == Fraction(0)  # FW timing, not dropped, not 0-invented
    assert best.best_words[1].timing_status == WordTimingStatus.WHISPERX_ALIGNED


def test_partial_alignment_missing_middle_word() -> None:
    fw = _fw_response([("one", 0.0, 0.2), ("two", 0.2, 0.4),
                       ("three", 0.4, 0.6), ("four", 0.6, 0.8)])
    wx = _wx_response(
        [("one", 0.01, 0.2), ("three", 0.41, 0.61), ("four", 0.61, 0.81)],
        "one two three four",
    )
    result = parse_transcription(fw, Fraction(0))
    best = reconcile_best_transcript(result, parse_alignment(wx, result.segments, Fraction(0)))
    assert [w.text for w in best.best_words] == ["one", "two", "three", "four"]
    assert best.best_words[1].timing_status == WordTimingStatus.FASTER_WHISPER_FALLBACK
    assert best.best_words[1].start_annotation_time == Fraction("0.2")
    assert best.coverage == 0.75


def test_partial_alignment_missing_final_word() -> None:
    fw = _fw_response([("one", 0.0, 0.2), ("two", 0.2, 0.4),
                       ("three", 0.4, 0.6), ("four", 0.6, 0.8)])
    wx = _wx_response(
        [("one", 0.01, 0.2), ("two", 0.21, 0.41), ("three", 0.41, 0.61)],
        "one two three four",
    )
    result = parse_transcription(fw, Fraction(0))
    best = reconcile_best_transcript(result, parse_alignment(wx, result.segments, Fraction(0)))
    assert [w.text for w in best.best_words] == ["one", "two", "three", "four"]
    assert best.best_words[-1].timing_status == WordTimingStatus.FASTER_WHISPER_FALLBACK
    assert best.best_words[-1].end_annotation_time == Fraction("0.8")


def test_missing_whisperx_segment_timestamp_falls_back_to_faster_whisper() -> None:
    fw = _fw_response([("one", 0.10, 0.40), ("two", 0.50, 0.90)])
    wx = _wx_response(
        [("one", 0.11, 0.41), ("two", 0.51, 0.91)],
        "one two", seg_start=None, seg_end=None,
    )
    result = parse_transcription(fw, Fraction(0))
    alignment = parse_alignment(wx, result.segments, Fraction(0))
    seg = alignment.segments[0]
    # Segment timestamps came from the preserved faster-whisper segment, NOT 0.
    assert seg.start_annotation_time == Fraction("0.10")
    assert seg.end_annotation_time == Fraction("0.90")
    assert seg.start_annotation_time != Fraction(0)


def test_missing_asr_time_never_becomes_zero() -> None:
    # A WhisperX word with no timing is KEPT with None timing, never mapped to 0.
    words = asr_runtime._words_from_raw(
        [{"text": "ghost", "start": None, "end": None, "probability": 0.5}],
        Fraction(0), "whisperx", 0,
    )
    assert len(words) == 1  # not dropped
    assert words[0].start_annotation_time is None  # missing, not 0
    assert words[0].asr_start_seconds is None
    assert asr_runtime._decimal_str(None) is None
    assert asr_runtime._decimal_str("None") is None


def test_best_word_sequence_matches_faster_whisper_source() -> None:
    fw = _fw_response([("one", 0.0, 0.2), ("two", 0.2, 0.4), ("three", 0.4, 0.6)])
    result = parse_transcription(fw, Fraction(0))
    best = reconcile_best_transcript(result, None)
    assert [w.text for w in best.best_words] == ["one", "two", "three"]
    # P3-ASR-006 passes for a faithful sequence...
    assert not audio_validator.validate_best_words(best.best_words, result)
    # ...and FAILS if a word is dropped.
    dropped = best.best_words[:-1]
    issues = audio_validator.validate_best_words(dropped, result)
    assert any(i.rule_id == "P3-ASR-006" and i.severity == Severity.FAIL for i in issues)


def test_unicode_word_matching_all_scripts() -> None:
    """§Review-1: WhisperX↔FW matching must work for non-Latin scripts, not
    collapse them to an empty key and fall back for every foreign word."""
    scripts = [
        ("hello", "world"),          # English
        ("café", "déjà"),            # accented Latin
        ("नमस्ते", "दुनिया"),          # हिन्दी
        ("مرحبا", "عالم"),           # العربية
        ("Привет", "мир"),           # Русский
        ("こんにちは", "世界"),         # 日本語
        ("你好", "世界"),              # 中文
    ]
    for w1, w2 in scripts:
        fw = _fw_response([(w1, 0.0, 0.4), (w2, 0.5, 0.9)])
        wx = _wx_response([(w1, 0.01, 0.41), (w2, 0.51, 0.91)], f"{w1} {w2}")
        result = parse_transcription(fw, Fraction(0))
        best = reconcile_best_transcript(result, parse_alignment(wx, result.segments, Fraction(0)))
        assert [w.text for w in best.best_words] == [w1, w2], f"{w1}/{w2} sequence"
        assert best.coverage == 1.0, f"{w1}/{w2} should fully align, not fall back"
        assert all(
            w.timing_status == WordTimingStatus.WHISPERX_ALIGNED for w in best.best_words
        ), f"{w1}/{w2} lost WhisperX timing"


def test_match_key_unicode_normalization() -> None:
    from manuscript_reviewer.audio.asr.runtime import _match_key
    # Non-Latin scripts produce non-empty keys (the old ASCII regex returned "").
    for token in ("नमस्ते", "مرحبا", "Привет", "世界", "café"):
        assert _match_key(token) != ""
    # NFKC folds composed vs decomposed accents; casefold folds case/script case.
    assert _match_key("café") == _match_key("café")  # é vs e + combining acute
    assert _match_key("ПРИВЕТ") == _match_key("привет")
    # Punctuation/whitespace are still stripped.
    assert _match_key(" Hello, ") == _match_key("hello")


def test_rejected_speech_is_resolved_and_leaves_queue() -> None:
    """§Review-2: a human who REJECTED the ASR claim has listened — the region is
    resolved: no caption text, no lingering mandatory item, P3-SPEECH-004 passes."""
    fw = _fw_response([("hello", 0.3, 0.7)])
    regions, _ = _speech_regions_from(fw)
    region = regions[0]
    region.source_verification_status = SourceVerificationStatus.REJECTED
    assert region.caption_text is None
    assert region.caption_text_eligible is False
    items = build_review_queue(_timeline(), regions, [], [], [], [], asr_unavailable=False)
    assert items == []  # rejected speech does not return to the queue
    assert not audio_validator.validate_source_verification(regions, items)
    # Rejected speech no longer blocks PASS (it is resolved).
    assert audio_validator.compute_audio_status([], items, [], regions, no_audio=False) == "PASS"


def test_evidence_state_has_no_human_dispositions() -> None:
    """§Review-3: EvidenceState is the machine axis only — human verified/rejected
    live solely in SourceVerificationStatus."""
    members = {e.value for e in EvidenceState}
    assert members == {
        "MACHINE_CANDIDATE", "ASR_EVIDENCE", "ALIGNED_EVIDENCE", "REVIEW_REQUIRED",
    }
    assert not hasattr(EvidenceState, "HUMAN_VERIFIED")
    assert not hasattr(EvidenceState, "REJECTED")


def test_text_mismatch_keeps_faster_whisper_timing() -> None:
    fw = _fw_response([("hello", 0.0, 0.3), ("there", 0.4, 0.7)])
    wx = _wx_response([("hello", 0.01, 0.31), ("here", 0.41, 0.71)], "hello here")
    result = parse_transcription(fw, Fraction(0))
    alignment = parse_alignment(wx, result.segments, Fraction(0))
    assert alignment.status == AlignmentStatus.TEXT_MISMATCH
    best = reconcile_best_transcript(result, alignment)
    assert [w.text for w in best.best_words] == ["hello", "there"]  # FW wording
    assert all(w.timing_source == "faster_whisper" for w in best.best_words)


# ------------------------------------------- §11-§14 boundary continuity semantics


def test_continuous_same_tone_across_cut() -> None:
    """A (continuous 440 Hz): stable spectrum + energy → CONTINUOUS."""
    bins = _bins(0.5, 1.0, -12.0, 440.0, 0.001) + _bins(1.0, 1.5, -12.0, 440.0, 0.001)
    ev = _analyze_boundary(bins)
    assert ev.audio_continuity_status == BoundaryAudioStatus.CONTINUOUS
    assert "SPECTRAL_CONTINUITY" in ev.continuity_evidence_codes
    assert ev.asr_word_spans_boundary is False  # no word — continuity is spectral


def test_equal_energy_frequency_switch_is_not_continuous() -> None:
    """B (440→880, EQUAL energy): energy alone must not read CONTINUOUS."""
    bins = _bins(0.5, 1.0, -12.0, 440.0, 0.001) + _bins(1.0, 1.5, -12.0, 880.0, 0.001)
    ev = _analyze_boundary(bins)
    assert ev.energy_delta_db == 0.0  # identical energy both sides
    assert ev.audio_continuity_status != BoundaryAudioStatus.CONTINUOUS
    assert ev.audio_continuity_status == BoundaryAudioStatus.UNCERTAIN
    assert "AUDIO_PRESENT_BOTH_SIDES" in ev.continuity_evidence_codes


def test_equal_energy_tone_to_noise_is_not_continuous() -> None:
    """C (tone→broadband noise, EQUAL RMS): spectrum moves → not CONTINUOUS."""
    bins = _bins(0.5, 1.0, -12.0, 300.0, 0.001) + _bins(1.0, 1.5, -12.0, 8000.0, 0.6)
    ev = _analyze_boundary(bins)
    assert ev.energy_delta_db == 0.0
    assert ev.audio_continuity_status != BoundaryAudioStatus.CONTINUOUS


def test_silence_gap_across_cut_is_checked_no_crossing() -> None:
    """D (silence spans the cut): DISCONTINUOUS / CHECKED_NO_CROSSING."""
    from manuscript_reviewer.models.audio import AudioRegion, AudioRegionKind
    bins = _bins(0.5, 1.5, -90.0, 0.0, 0.0)
    silence = AudioRegion(
        region_id="r1", kind=AudioRegionKind.SILENCE_CANDIDATE,
        start_sample=0, end_sample=1, start_annotation_time=Fraction(0),
        end_annotation_time=Fraction(2), mean_dbfs=-90.0,
    )
    ev = _analyze_boundary(bins, regions=[silence])
    assert ev.silence_spans_boundary is True
    assert ev.audio_continuity_status == BoundaryAudioStatus.DISCONTINUOUS
    assert ev.audio_verification_status.value == "CHECKED_NO_CROSSING"


def test_true_word_spanning_cut() -> None:
    """E: an actual best-word interval spans the boundary."""
    bins = _bins(0.5, 1.0, -12.0, 440.0, 0.3) + _bins(1.0, 1.5, -12.0, 900.0, 0.3)
    word = _best_word(0, 0, "crossing", 0.9, 1.1)
    ev = _analyze_boundary(bins, best_words=[word])
    assert ev.asr_word_spans_boundary is True
    assert ev.audio_continuity_status == BoundaryAudioStatus.CONTINUOUS
    assert "ASR_WORD_SPANS_BOUNDARY" in ev.continuity_evidence_codes


def test_speech_region_spans_without_word_spanning() -> None:
    """F: a merged speech region spans the cut but NO word does — distinct facts."""
    bins = _bins(0.5, 1.0, -12.0, 440.0, 0.3) + _bins(1.0, 1.5, -12.0, 460.0, 0.3)
    # Two words with a <0.5 s gap merged into one region straddling the cut.
    words = [_best_word(0, 0, "before", 0.6, 0.95), _best_word(0, 1, "after", 1.05, 1.4)]
    region = SpeechRegion(
        region_id="s1", start_exact=Fraction("0.6"), end_exact=Fraction("1.4"),
        start_manuscript="0.6s", end_manuscript="1.4s", sources=["asr_faster_whisper"],
    )
    ev = _analyze_boundary(bins, best_words=words, speech_regions=[region])
    assert ev.speech_region_spans_boundary is True
    assert ev.asr_word_spans_boundary is False  # NOT cloned from the region fact
    assert "SPEECH_REGION_SPANS_NO_WORD" in ev.continuity_evidence_codes


def test_asr_segment_spans_but_no_word_spans_are_distinct() -> None:
    """G: segment spans the cut but no word does — segment and word facts differ."""
    from manuscript_reviewer.models.audio import ASRSegment
    bins = _bins(0.5, 1.0, -12.0, 440.0, 0.3) + _bins(1.0, 1.5, -12.0, 460.0, 0.3)
    words = [_best_word(0, 0, "before", 0.6, 0.95), _best_word(0, 1, "after", 1.05, 1.4)]
    segment = ASRSegment(
        segment_id=0, text="before after",
        start_annotation_time=Fraction("0.6"), end_annotation_time=Fraction("1.4"),
    )
    ev = _analyze_boundary(bins, best_words=words, asr_segments=[segment])
    assert ev.asr_segment_spans_boundary is True
    assert ev.asr_word_spans_boundary is False
    assert "ASR_SEGMENT_SPANS_BOUNDARY" in ev.continuity_evidence_codes


def test_p3_boundary_003_rejects_unjustified_continuous() -> None:
    ev = _analyze_boundary(
        _bins(0.5, 1.0, -12.0, 440.0, 0.001) + _bins(1.0, 1.5, -12.0, 440.0, 0.001)
    )
    # Tamper: claim CONTINUOUS but strip the evidence codes.
    ev.continuity_evidence_codes = []
    issues = audio_validator.validate_boundary_continuity([ev])
    assert any(i.rule_id == "P3-BOUNDARY-003" and i.severity == Severity.FAIL for i in issues)


# --------------------------------------------------------- §18 review playback bounds


def test_review_playback_bounds_clamped_to_source_audio() -> None:
    # Speech right at the clip edges; padding must not seek outside [offset, end].
    region = SpeechRegion(
        region_id="s1", start_exact=Fraction("0.05"), end_exact=Fraction("2.95"),
        start_manuscript="0.05s", end_manuscript="2.95s", sources=["asr_faster_whisper"],
        review_reasons=[AudioReviewReason.MANDATORY_SOURCE_AUDIO_VERIFICATION],
    )
    timeline = _timeline(offset=0.0, duration=3.0)
    items = build_review_queue(timeline, [region], [], [], [], [], asr_unavailable=False)
    item = items[0]
    assert item.playback_start >= Fraction(0)
    assert item.playback_end <= Fraction(3)


def test_review_playback_bounds_respect_nonzero_offset() -> None:
    region = SpeechRegion(
        region_id="s1", start_exact=Fraction("5.05"), end_exact=Fraction("5.5"),
        start_manuscript="5.05s", end_manuscript="5.5s", sources=["asr_faster_whisper"],
        review_reasons=[AudioReviewReason.MANDATORY_SOURCE_AUDIO_VERIFICATION],
    )
    timeline = _timeline(offset=5.0, duration=3.0)  # audio spans [5, 8]
    items = build_review_queue(timeline, [region], [], [], [], [], asr_unavailable=False)
    assert items[0].playback_start >= Fraction(5)


# --------------------------------------------------- §19 multilingual language review


def _lang_region(region_id: str, start: float, end: float, candidate: str | None,
                 status: LanguageReviewStatus) -> SpeechRegion:
    return SpeechRegion(
        region_id=region_id, start_exact=Fraction(start), end_exact=Fraction(end),
        start_manuscript=f"{start}s", end_manuscript=f"{end}s",
        sources=["asr_faster_whisper"], text_candidate="...",
        language=LanguageEvidence(language_candidate=candidate, language_probability=0.4,
                                  language_source="faster_whisper",
                                  language_review_status=status),
        review_reasons=[AudioReviewReason.MANDATORY_SOURCE_AUDIO_VERIFICATION],
    )


def test_two_uncertain_language_regions_not_globally_suppressed() -> None:
    r1 = _lang_region("s1", 0.0, 1.0, "fr", LanguageReviewStatus.REVIEW_REQUIRED)
    r2 = _lang_region("s2", 2.0, 3.0, None, LanguageReviewStatus.UNKNOWN)
    items = build_review_queue(_timeline(), [r1, r2], [], [], [], [], asr_unavailable=False)
    assert len(items) == 2  # each region reviewed independently
    assert AudioReviewReason.LOW_LANGUAGE_CONFIDENCE in items[0].reasons
    assert AudioReviewReason.UNKNOWN_LANGUAGE in items[1].reasons


# ------------------------------------------------------------- §20 review priority


def test_review_queue_priority_levels() -> None:
    from manuscript_reviewer.models.audio import AudioTransientCandidate
    speech = SpeechRegion(
        region_id="s1", start_exact=Fraction(1), end_exact=Fraction(2),
        start_manuscript="1.0s", end_manuscript="2.0s", sources=["asr_faster_whisper"],
        review_reasons=[AudioReviewReason.MANDATORY_SOURCE_AUDIO_VERIFICATION],
    )
    weak = AudioTransientCandidate(
        candidate_id="t1", start_sample=0, peak_sample=1, end_sample=2,
        start_annotation_time=Fraction(1), peak_annotation_time=Fraction(1),
        end_annotation_time=Fraction(1), peak_dbfs=-30.0, rise_db=15.0,
    )
    items = build_review_queue(_timeline(), [speech], [], [weak], [], [], asr_unavailable=False)
    by_reason = {r: it.priority for it in items for r in it.reasons}
    mandatory = AudioReviewReason.MANDATORY_SOURCE_AUDIO_VERIFICATION
    assert by_reason[mandatory] == ReviewPriority.CRITICAL
    assert by_reason[AudioReviewReason.TRANSIENT_SEMANTICS_UNKNOWN] == ReviewPriority.LOW


# ------------------------------------------------------------- §21 model cache state


def test_model_downloaded_this_run_metadata(monkeypatch: Any) -> None:
    # Absent before, present after a worker run → downloaded_this_run (not not_cached).
    monkeypatch.setattr(asr_runtime, "_model_in_cache", lambda model: True)
    assert resolve_cache_state("m", "not_cached", worker_ran=True) == "downloaded_this_run"
    # Present before and after → cached.
    assert resolve_cache_state("m", "cached", worker_ran=True) == "cached"
    # Absent before and after a real run → unavailable.
    monkeypatch.setattr(asr_runtime, "_model_in_cache", lambda model: False)
    assert resolve_cache_state("m", "not_cached", worker_ran=True) == "unavailable"


# ------------------------------------------------- §16/§17 AAC sample-anchor policy


def test_sample_anchor_review_required_warns_and_never_claims_perfect() -> None:
    timeline = _timeline()
    timeline.codec_skip_samples = 1024  # AAC encoder delay
    timeline.sample_anchor_status = SampleAnchorStatus.ANCHOR_REVIEW_REQUIRED
    issues = audio_validator.validate_audio_timeline(timeline, [])
    assert any(i.rule_id == "P3-AUDIO-009" for i in issues)
    assert any("AUDIO_SAMPLE_ANCHOR_REVIEW_REQUIRED" in i.message for i in issues)


# ------------------------------------------------- §22 no Hard-cut default transition


def test_rule_file_has_no_hard_cut_default() -> None:
    rules = load_rules()
    assert rules.get("shots.default_transition") is None  # not "Hard cut"
    assert rules.get("shots.unresolved_transition_is_not_hard_cut") is True
    # Hard cut is still a valid MENU option — just never an unresolved default.
    assert "Hard cut" in rules.get("shots.allowed_transition_types")


def test_unresolved_transition_never_defaults_to_hard_cut() -> None:
    """A boundary with no resolved incoming transition stays UNRESOLVED with no
    manuscript type — a builder can never emit Hard cut from an unresolved edit."""
    from manuscript_reviewer.models.shot_truth import TransitionEvidence, TransitionStatus
    unresolved = TransitionEvidence(status=TransitionStatus.UNRESOLVED)
    assert unresolved.manuscript_type is None
    assert unresolved.manuscript_type != "Hard cut"
