"""Audio Truth Engine core tests (CI-safe: no ASR models, mock adapters only)."""

from __future__ import annotations

import itertools
import json
from fractions import Fraction
from pathlib import Path
from typing import Any

import pytest

from manuscript_reviewer.audio.asr import runtime as asr_runtime
from manuscript_reviewer.audio.asr.runtime import (
    ASRConfig,
    normalize_text,
    parse_alignment,
    parse_transcription,
)
from manuscript_reviewer.audio.engine import run_audio_analysis
from manuscript_reviewer.audio.probe import probe_audio_priming
from manuscript_reviewer.audio.timeline import annotation_to_sample, sample_to_annotation
from manuscript_reviewer.media.clock import AnnotationClock
from manuscript_reviewer.media.endpoint import compute_annotation_endpoint
from manuscript_reviewer.media.ffmpeg_tools import find_tool, run_tool
from manuscript_reviewer.media.frames import enumerate_frames
from manuscript_reviewer.media.probe import probe_media
from manuscript_reviewer.models.audio import (
    AlignmentStatus,
    ASRStatus,
    AudioRegionKind,
    AudioReviewReason,
    AudioStatus,
    AudioVerificationStatus,
    BoundaryAudioStatus,
    SampleAnchorStatus,
)
from manuscript_reviewer.models.validation import RunStatus
from manuscript_reviewer.pipeline import run_audit
from manuscript_reviewer.shots.engine import run_shot_analysis
from tests.conftest import FIXTURES_DIR, requires_ffmpeg

pytestmark = requires_ffmpeg

SIZE = "320x240"


def _make_av(name: str, video_src: str, audio_filter_args: list[str],
             extra: list[str] | None = None) -> Path:
    path = FIXTURES_DIR / name
    if path.exists():
        return path
    FIXTURES_DIR.mkdir(exist_ok=True)
    ffmpeg = find_tool("ffmpeg")
    args = ["-v", "error", "-f", "lavfi", "-i", video_src, *audio_filter_args,
            "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p",
            *(extra or []), "-y", str(path)]
    run_tool(ffmpeg, args)
    return path


def _analyze(path: Path, tmp_path: Path, with_shots: bool = False,
             asr_enabled: bool = False, **kwargs: Any) -> Any:
    media, _ = probe_media(path)
    ledger = enumerate_frames(path, media.video_streams[0].time_base)
    run_dir = tmp_path / path.stem
    run_dir.mkdir(parents=True, exist_ok=True)
    shot_result = None
    endpoint = None
    if with_shots:
        shot_output = run_shot_analysis(path, run_dir, media, ledger,
                                        extract_evidence=False)
        shot_result = shot_output.result
        assert shot_result is not None
        endpoint = shot_result.annotation_endpoint_exact
    else:
        clock = AnnotationClock.from_ledger(ledger)
        endpoint = compute_annotation_endpoint(media, ledger, path.stem, clock).annotation_endpoint
    output = run_audio_analysis(
        path, run_dir, media, ledger, shot_result, endpoint,
        asr_enabled=asr_enabled, render_clips=True, **kwargs
    )
    return output


# ------------------------------------------------------------ no audio / basics

def test_no_audio_video_is_valid(tmp_path: Path, clip_24fps: Path) -> None:
    output = _analyze(clip_24fps, tmp_path)
    assert output.result is not None
    assert output.result.audio_status == AudioStatus.NO_AUDIO_STREAM
    assert output.result.overall_status == "NO_AUDIO_STREAM"
    # No fake artifacts: no source.wav for silent media.
    assert not (tmp_path / clip_24fps.stem / "audio" / "source.wav").exists()
    assert not [i for i in output.issues if i.severity.value == "FAIL"]


def test_sine_48k_stereo_full_pipeline(tmp_path: Path) -> None:
    clip = _make_av(
        "au_sine48k.mp4",
        f"testsrc2=duration=2:rate=24:size={SIZE}",
        ["-f", "lavfi", "-i", "sine=frequency=440:duration=2:sample_rate=48000"],
        ["-c:a", "aac", "-ac", "2", "-ar", "48000", "-shortest"],
    )
    output = _analyze(clip, tmp_path)
    qc = output.result
    assert qc is not None
    timeline = qc.timeline
    assert timeline is not None
    assert timeline.evidence_sample_rate == 48000
    assert timeline.evidence_channels == 2
    # 10ms bins are exactly 480 samples at 48 kHz.
    assert qc.energy_bin_count > 190
    audio_dir = tmp_path / clip.stem / "audio"
    for name in ("source.wav", "audio_frames.csv", "audio_frames.jsonl",
                 "audio_timeline.json", "audio_energy_10ms.csv", "waveform.png",
                 "spectrogram.png", "audio_energy.png", "audio_regions.json",
                 "transient_candidates.json", "speech_regions.csv",
                 "boundary_audio_evidence.json", "audio_review_queue.json",
                 "audio_qc.json"):
        assert (audio_dir / name).is_file(), f"missing {name}"
    # Tonal signal is classed, never semantically labeled.
    kinds = {r.kind for r in qc.regions}
    assert AudioRegionKind.SUSTAINED_TONAL_AUDIO in kinds
    assert not (audio_dir / "asr.wav").exists()  # ASR disabled → no asr.wav


def test_44k_mono_energy_bins_exact(tmp_path: Path) -> None:
    clip = _make_av(
        "au_sine44k.mp4",
        f"testsrc2=duration=1:rate=24:size={SIZE}",
        ["-f", "lavfi", "-i", "sine=frequency=300:duration=1:sample_rate=44100"],
        ["-c:a", "aac", "-ac", "1", "-ar", "44100", "-shortest"],
    )
    output = _analyze(clip, tmp_path)
    qc = output.result
    assert qc is not None and qc.timeline is not None
    assert qc.timeline.evidence_sample_rate == 44100
    csv_lines = (tmp_path / clip.stem / "audio" / "audio_energy_10ms.csv").read_text().splitlines()
    first = csv_lines[1].split(",")
    assert first[1] == "0" and first[2] == "441"  # exactly 441 samples per bin


def test_silence_tone_silence_regions(tmp_path: Path) -> None:
    clip = _make_av(
        "au_sts.mp4",
        f"testsrc2=duration=3:rate=24:size={SIZE}",
        ["-f", "lavfi", "-i",
         "sine=frequency=600:duration=3,volume=enable='between(t,1,2)':volume=1:eval=frame,"
         "volume=enable='not(between(t,1,2))':volume=0:eval=frame"],
        ["-c:a", "aac", "-shortest"],
    )
    output = _analyze(clip, tmp_path)
    qc = output.result
    assert qc is not None
    silences = [r for r in qc.regions if r.kind == AudioRegionKind.SILENCE_CANDIDATE]
    actives = [r for r in qc.regions if r.kind == AudioRegionKind.ACTIVE_AUDIO]
    assert silences and actives
    active = actives[0]
    assert abs(float(active.start_annotation_time) - 1.0) < 0.15
    assert abs(float(active.end_annotation_time) - 2.0) < 0.15
    # Silence is internal evidence only — never caption text.
    qc_json = (tmp_path / clip.stem / "audio" / "audio_qc.json").read_text()
    assert "No speech" not in qc_json and "No music" not in qc_json


def test_transient_impulses_detected(tmp_path: Path) -> None:
    clip = _make_av(
        "au_clicks.mp4",
        f"testsrc2=duration=2:rate=24:size={SIZE}",
        ["-f", "lavfi", "-i",
         "aevalsrc=exprs='if(lt(mod(t\\,0.7)\\,0.02)\\,0.9*sin(2*PI*2500*t)\\,0.001*sin(2*PI*200*t))'"
         ":duration=2:sample_rate=48000"],
        ["-c:a", "aac", "-shortest"],
    )
    output = _analyze(clip, tmp_path)
    qc = output.result
    assert qc is not None
    assert qc.transient_count >= 2
    for transient in qc.transients:
        assert transient.start_sample <= transient.peak_sample <= transient.end_sample
    reasons = [r for i in qc.review_items for r in i.reasons]
    assert AudioReviewReason.TRANSIENT_SEMANTICS_UNKNOWN in reasons


# ------------------------------------------------------------ timeline / offsets

def test_nonzero_video_pts_annotation_clock(tmp_path: Path) -> None:
    """Fix A/A2: video starting at PTS 5 s must normalize to annotation 0."""
    clip = FIXTURES_DIR / "st_offset5.mp4"
    if not clip.exists():
        ffmpeg = find_tool("ffmpeg")
        run_tool(ffmpeg, [
            "-v", "error",
            "-f", "lavfi", "-i", f"testsrc2=duration=1:rate=24:size={SIZE}",
            "-f", "lavfi", "-i", f"smptebars=duration=1:rate=24:size={SIZE}",
            "-filter_complex", "[0:v][1:v]concat=n=2:v=1:a=0",
            "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p",
            "-output_ts_offset", "5", "-y", str(clip),
        ])
    media, _ = probe_media(clip)
    ledger = enumerate_frames(clip, media.video_streams[0].time_base)
    clock = AnnotationClock.from_ledger(ledger)
    assert clock.origin == Fraction(5)
    endpoint = compute_annotation_endpoint(media, ledger, clip.stem, clock)
    # Duration signals (2 s) must become origin+2=7 s source, 2 s annotation —
    # never 7 s annotation and never -3 s.
    assert endpoint.endpoint == Fraction(7)
    assert endpoint.annotation_endpoint == Fraction(2)
    assert endpoint.conflict is False

    run_dir = tmp_path / "offset"
    run_dir.mkdir()
    shot_output = run_shot_analysis(clip, run_dir, media, ledger, extract_evidence=False)
    result = shot_output.result
    assert result is not None
    assert result.annotation_timeline_origin == Fraction(5)
    boundary = next(c for c in result.candidates if c.status.value == "SUPPORTED")
    assert boundary.boundary_time_exact == Fraction(6)  # raw source PTS preserved
    assert boundary.boundary_annotation_time == Fraction(1)
    assert boundary.boundary_time_manuscript == "1.0s"
    shot1, shot2 = result.shots
    assert shot1.start_exact == Fraction(0) and shot1.end_exact == Fraction(1)
    assert shot2.start_exact == Fraction(1) and shot2.end_exact == Fraction(2)
    assert shot1.source_start_exact == Fraction(5)
    assert shot2.source_end_exact == Fraction(7)
    assert result.overall_status == "PASS"


def test_audio_starting_after_video(tmp_path: Path) -> None:
    clip = _make_av(
        "au_delayed.mp4",
        f"testsrc2=duration=3:rate=24:size={SIZE}",
        ["-f", "lavfi", "-i",
         "sine=frequency=500:duration=2:sample_rate=48000,adelay=1000|1000"],
        ["-c:a", "aac", "-shortest"],
    )
    output = _analyze(clip, tmp_path)
    qc = output.result
    assert qc is not None and qc.timeline is not None
    # Sample↔annotation mapping is exact and invertible.
    timeline = qc.timeline
    t = sample_to_annotation(timeline, 48000)
    assert t == timeline.annotation_audio_offset + Fraction(1)
    assert annotation_to_sample(timeline, t) == 48000


def test_audio_shorter_than_video_cross_check(tmp_path: Path) -> None:
    clip = _make_av(
        "au_short.mp4",
        f"testsrc2=duration=3:rate=24:size={SIZE}",
        ["-f", "lavfi", "-i", "sine=frequency=500:duration=1:sample_rate=48000"],
        ["-c:a", "aac"],
    )
    output = _analyze(clip, tmp_path)
    assert output.result is not None
    # Shorter audio is fine; P1 already warns on stream-duration mismatch.
    assert output.result.audio_status == AudioStatus.ANALYZED


# ------------------------------------------------------------ boundary continuity

def _hardcut_with_audio(name: str, audio_expr_args: list[str]) -> Path:
    path = FIXTURES_DIR / name
    if path.exists():
        return path
    ffmpeg = find_tool("ffmpeg")
    run_tool(ffmpeg, [
        "-v", "error",
        "-f", "lavfi", "-i", f"testsrc2=duration=1:rate=24:size={SIZE}",
        "-f", "lavfi", "-i", f"smptebars=duration=1:rate=24:size={SIZE}",
        *audio_expr_args,
        "-filter_complex", "[0:v][1:v]concat=n=2:v=1:a=0[v]",
        "-map", "[v]", "-map", "2:a",
        "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-shortest", "-y", str(path),
    ])
    return path


def _hardcut_two_audio(name: str, a0: str, a1: str) -> Path:
    """Video hard cut at t=1 s with two DIFFERENT 1 s audio sources concatenated
    at the cut, so the audio source genuinely switches exactly at the boundary."""
    path = FIXTURES_DIR / name
    if path.exists():
        return path
    FIXTURES_DIR.mkdir(exist_ok=True)
    ffmpeg = find_tool("ffmpeg")
    run_tool(ffmpeg, [
        "-v", "error",
        "-f", "lavfi", "-i", f"testsrc2=duration=1:rate=24:size={SIZE}",
        "-f", "lavfi", "-i", f"smptebars=duration=1:rate=24:size={SIZE}",
        "-f", "lavfi", "-i", a0,
        "-f", "lavfi", "-i", a1,
        "-filter_complex",
        "[0:v][1:v]concat=n=2:v=1:a=0[v];[2:a][3:a]concat=n=2:v=0:a=1[a]",
        "-map", "[v]", "-map", "[a]",
        "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-shortest", "-y", str(path),
    ])
    return path


def test_boundary_equal_energy_frequency_switch_not_continuous(tmp_path: Path) -> None:
    """§14 B: 440 Hz → 880 Hz at equal amplitude must NOT read CONTINUOUS just
    because energy is present on both sides — the spectrum moved."""
    clip = _hardcut_two_audio(
        "au_cut_freqswitch.mp4",
        "sine=frequency=440:duration=1:sample_rate=48000",
        "sine=frequency=880:duration=1:sample_rate=48000",
    )
    output = _analyze(clip, tmp_path, with_shots=True)
    qc = output.result
    assert qc is not None and qc.boundaries_checked == 1
    boundary = qc.boundary_evidence[0]
    assert boundary.audio_present_before and boundary.audio_present_after
    assert boundary.audio_continuity_status != BoundaryAudioStatus.CONTINUOUS


def test_boundary_equal_energy_tone_to_noise_not_continuous(tmp_path: Path) -> None:
    """§14 C: a tone → broadband noise at similar level must NOT read CONTINUOUS."""
    clip = _hardcut_two_audio(
        "au_cut_tonenoise.mp4",
        "sine=frequency=400:duration=1:sample_rate=48000",
        "anoisesrc=color=white:duration=1:sample_rate=48000:amplitude=0.5",
    )
    output = _analyze(clip, tmp_path, with_shots=True)
    qc = output.result
    assert qc is not None and qc.boundaries_checked == 1
    assert qc.boundary_evidence[0].audio_continuity_status != BoundaryAudioStatus.CONTINUOUS


def test_aac_priming_skip_samples_detected(tmp_path: Path) -> None:
    """§16/§17: AAC exposes encoder-delay skip_samples; sample-0 anchoring is not
    claimed and an AUDIO_SAMPLE_ANCHOR_REVIEW_REQUIRED concern is recorded."""
    clip = _make_av(
        "au_sine48k.mp4",
        f"testsrc2=duration=2:rate=24:size={SIZE}",
        ["-f", "lavfi", "-i", "sine=frequency=440:duration=2:sample_rate=48000"],
        ["-c:a", "aac", "-ac", "2", "-ar", "48000", "-shortest"],
    )
    priming = probe_audio_priming(clip)
    assert priming.codec_name == "aac"
    assert priming.has_priming  # skip_samples > 0 for AAC
    output = _analyze(clip, tmp_path)
    qc = output.result
    assert qc is not None and qc.timeline is not None
    assert qc.timeline.sample_anchor_status == SampleAnchorStatus.ANCHOR_REVIEW_REQUIRED
    assert qc.timeline.codec_skip_samples is not None and qc.timeline.codec_skip_samples > 0
    codes = {c.concern_code for c in qc.concerns}
    assert "AUDIO_SAMPLE_ANCHOR_REVIEW_REQUIRED" in codes
    assert any(i.rule_id == "P3-AUDIO-009" for i in output.issues)


def test_boundary_with_continuous_audio_crossing(tmp_path: Path) -> None:
    clip = _hardcut_with_audio(
        "au_cut_continuous.mp4",
        ["-f", "lavfi", "-i", "sine=frequency=440:duration=2:sample_rate=48000"],
    )
    output = _analyze(clip, tmp_path, with_shots=True)
    qc = output.result
    assert qc is not None
    assert qc.boundaries_checked == 1
    boundary = qc.boundary_evidence[0]
    assert boundary.visual_boundary_exact == Fraction(1)
    assert boundary.audio_continuity_status == BoundaryAudioStatus.CONTINUOUS
    assert (
        boundary.audio_verification_status
        == AudioVerificationStatus.CROSSING_REVIEW_REQUIRED
    )
    # L/J-cut is never auto-finalized (P3-BOUNDARY-002).
    qc_text = json.dumps(qc.model_dump(mode="json"))
    assert "L-cut" not in qc_text and "J-cut" not in qc_text
    assert qc.overall_status == "REVIEW_REQUIRED"


def test_boundary_with_silence_gap(tmp_path: Path) -> None:
    clip = _hardcut_with_audio(
        "au_cut_gap.mp4",
        ["-f", "lavfi", "-i",
         "sine=frequency=440:duration=2:sample_rate=48000,"
         "volume=enable='between(t,0.5,1.5)':volume=0:eval=frame"],
    )
    output = _analyze(clip, tmp_path, with_shots=True)
    qc = output.result
    assert qc is not None
    assert qc.boundaries_checked == 1
    boundary = qc.boundary_evidence[0]
    assert boundary.silence_spans_boundary is True
    assert boundary.audio_continuity_status == BoundaryAudioStatus.DISCONTINUOUS
    assert boundary.audio_verification_status == AudioVerificationStatus.CHECKED_NO_CROSSING


# ------------------------------------------------------------ mock ASR behavior

MOCK_TRANSCRIPTION = {
    "status": "ok",
    "package_version": "1.2.1-mock",
    "model": "mock",
    "device": "cpu",
    "compute_type": "int8",
    "transcribe_seconds": 0.1,
    "language": "en",
    "language_probability": 0.97,
    "duration": "2.0",
    "segments": [
        {
            "id": 0,
            "start": "0.30",
            "end": "1.10",
            "text": " Hello there.",
            "avg_logprob": -0.2,
            "no_speech_prob": 0.02,
            "words": [
                {"text": " Hello", "start": "0.30", "end": "0.65", "probability": 0.95},
                {"text": " there.", "start": "0.70", "end": "1.10", "probability": 0.92},
            ],
        },
        {
            "id": 1,
            "start": "1.90",
            "end": "2.40",
            "text": " General Kenobi.",
            "avg_logprob": -0.3,
            "no_speech_prob": 0.05,
            "words": [
                {"text": " General", "start": "1.90", "end": "2.10", "probability": 0.9},
                {"text": " Kenobi.", "start": "2.15", "end": "2.40", "probability": 0.88},
            ],
        },
    ],
}


class MockTranscriber:
    def __init__(self, response: dict[str, Any] | Exception) -> None:
        self.response = response

    def transcribe(self, asr_wav: Path, config: ASRConfig, scratch: Path) -> dict[str, Any]:
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


class MockAligner:
    def __init__(self, response: dict[str, Any] | Exception) -> None:
        self.response = response

    def align(self, asr_wav: Path, language: str, segments: list[dict[str, Any]],
              config: ASRConfig, scratch: Path) -> dict[str, Any]:
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


def _mock_alignment_response(text_change: bool = False) -> dict[str, Any]:
    return {
        "status": "ok",
        "package_version": "3.4.3-mock",
        "language": "en",
        "device": "cpu",
        "align_seconds": 0.05,
        "segments": [
            {
                "id": 0,
                "start": "0.31",
                "end": "1.08",
                "text": " Hello there." if not text_change else " Hello here.",
                "words": [
                    {"text": "Hello", "start": "0.31", "end": "0.63", "probability": 0.99},
                    {"text": "there.", "start": "0.71", "end": "1.08", "probability": 0.97},
                ],
            },
            {
                "id": 1,
                "start": "1.91",
                "end": "2.38",
                "text": " General Kenobi.",
                "words": [
                    {"text": "General", "start": "1.91", "end": "2.09", "probability": 0.98},
                    {"text": "Kenobi.", "start": "2.16", "end": "2.38", "probability": 0.96},
                ],
            },
        ],
    }


def test_mock_asr_speech_regions_and_pause_split(tmp_path: Path) -> None:
    clip = _make_av(
        "au_mockspeech.mp4",
        f"testsrc2=duration=3:rate=24:size={SIZE}",
        ["-f", "lavfi", "-i", "sine=frequency=350:duration=3:sample_rate=48000"],
        ["-c:a", "aac", "-shortest"],
    )
    output = _analyze(
        clip, tmp_path, with_shots=False, asr_enabled=True,
        transcriber=MockTranscriber(MOCK_TRANSCRIPTION),
        aligner=MockAligner(_mock_alignment_response()),
    )
    qc = output.result
    assert qc is not None
    assert qc.asr_status == ASRStatus.PASS
    assert qc.alignment_status == AlignmentStatus.ALIGNED
    # Word gap 1.08→1.91 (> 0.5 s pause rule) splits into two regions.
    assert qc.speech_region_count == 2
    region1, region2 = qc.speech_regions
    assert region1.text_candidate == "Hello there."
    assert region2.text_candidate == "General Kenobi."
    # Best timing comes from WhisperX (aligned start 0.31, not fw 0.30).
    assert region1.start_exact == Fraction("0.31")
    # Language is machine evidence, not fact.
    assert qc.language is not None
    assert qc.language.language_review_status.value == "SUPPORTED_BY_ASR"
    asr_dir = tmp_path / clip.stem / "audio" / "asr"
    best = (asr_dir / "transcript_best.txt").read_text(encoding="utf-8")
    assert "ASR_EVIDENCE_ONLY" in best
    assert "Hello there." in best


def test_alignment_text_mismatch_preserves_fw(tmp_path: Path) -> None:
    clip = _make_av(
        "au_mockspeech.mp4",
        f"testsrc2=duration=3:rate=24:size={SIZE}",
        ["-f", "lavfi", "-i", "sine=frequency=350:duration=3:sample_rate=48000"],
        ["-c:a", "aac", "-shortest"],
    )
    output = _analyze(
        clip, tmp_path, asr_enabled=True,
        transcriber=MockTranscriber(MOCK_TRANSCRIPTION),
        aligner=MockAligner(_mock_alignment_response(text_change=True)),
    )
    qc = output.result
    assert qc is not None
    assert qc.alignment_status == AlignmentStatus.TEXT_MISMATCH
    # transcript_best falls back to faster-whisper wording AND timing.
    best = (tmp_path / clip.stem / "audio" / "asr" / "transcript_best.txt").read_text(
        encoding="utf-8"
    )
    assert "Hello there." in best and "Hello here." not in best
    assert any(i.rule_id == "P3-ASR-003" for i in output.issues)


def test_asr_worker_failure_degrades_and_continues(tmp_path: Path) -> None:
    clip = _make_av(
        "au_mockspeech.mp4",
        f"testsrc2=duration=3:rate=24:size={SIZE}",
        ["-f", "lavfi", "-i", "sine=frequency=350:duration=3:sample_rate=48000"],
        ["-c:a", "aac", "-shortest"],
    )
    output = _analyze(
        clip, tmp_path, asr_enabled=True,
        transcriber=MockTranscriber(RuntimeError("simulated worker crash")),
    )
    qc = output.result
    assert qc is not None
    assert qc.asr_status == ASRStatus.FAILED
    # Evidence pipeline continued: waveform/energy/spectrogram all present.
    audio_dir = tmp_path / clip.stem / "audio"
    assert (audio_dir / "waveform.png").is_file()
    assert (audio_dir / "spectrogram.png").is_file()
    assert (audio_dir / "audio_energy.png").is_file()
    # Failure has a recorded reason; uncovered audio demands review.
    status = json.loads((audio_dir / "asr" / "asr_status.json").read_text())
    assert "simulated worker crash" in (status["failure_reason"] or "")
    reasons = [r for i in qc.review_items for r in i.reasons]
    assert AudioReviewReason.AUDIO_WITHOUT_ASR_COVERAGE in reasons
    assert AudioReviewReason.ASR_UNAVAILABLE in reasons


def test_asr_unavailable_when_bootstrap_disabled(tmp_path: Path,
                                                 monkeypatch: pytest.MonkeyPatch) -> None:
    clip = _make_av(
        "au_mockspeech.mp4",
        f"testsrc2=duration=3:rate=24:size={SIZE}",
        ["-f", "lavfi", "-i", "sine=frequency=350:duration=3:sample_rate=48000"],
        ["-c:a", "aac", "-shortest"],
    )
    # Simulate a missing worker env with bootstrap disabled.
    monkeypatch.setattr(asr_runtime, "FW_ENV", tmp_path / "missing_env")
    output = _analyze(
        clip, tmp_path, asr_enabled=True,
        asr_config=ASRConfig(bootstrap=False),
    )
    qc = output.result
    assert qc is not None
    assert qc.asr_status == ASRStatus.UNAVAILABLE
    assert (tmp_path / clip.stem / "audio" / "waveform.png").is_file()


def test_vad_recall_defense_uncovered_audio(tmp_path: Path) -> None:
    """Strong audio with NO ASR output must become review — never silence truth."""
    clip = _make_av(
        "au_mockspeech.mp4",
        f"testsrc2=duration=3:rate=24:size={SIZE}",
        ["-f", "lavfi", "-i", "sine=frequency=350:duration=3:sample_rate=48000"],
        ["-c:a", "aac", "-shortest"],
    )
    empty = dict(MOCK_TRANSCRIPTION, segments=[])
    output = _analyze(clip, tmp_path, asr_enabled=True,
                      transcriber=MockTranscriber(empty))
    qc = output.result
    assert qc is not None
    assert qc.asr_status == ASRStatus.PASS
    assert qc.speech_region_count == 0
    reasons = [r for i in qc.review_items for r in i.reasons]
    assert AudioReviewReason.AUDIO_WITHOUT_ASR_COVERAGE in reasons
    assert qc.overall_status == "REVIEW_REQUIRED"


def test_clip_edge_speech_flags(tmp_path: Path) -> None:
    clip = _make_av(
        "au_edge.mp4",
        f"testsrc2=duration=2:rate=24:size={SIZE}",
        ["-f", "lavfi", "-i", "sine=frequency=350:duration=2:sample_rate=48000"],
        ["-c:a", "aac", "-shortest"],
    )
    edge_response = {
        **MOCK_TRANSCRIPTION,
        "segments": [
            {
                "id": 0, "start": "0.02", "end": "1.98", "text": " edge words",
                "avg_logprob": -0.2, "no_speech_prob": 0.02,
                "words": [
                    {"text": " edge", "start": "0.02", "end": "0.90", "probability": 0.9},
                    {"text": " words", "start": "1.00", "end": "1.98", "probability": 0.9},
                ],
            }
        ],
    }
    output = _analyze(clip, tmp_path, asr_enabled=True,
                      transcriber=MockTranscriber(edge_response))
    qc = output.result
    assert qc is not None
    region = qc.speech_regions[0]
    assert AudioReviewReason.SPEECH_AT_CLIP_START in region.review_reasons
    assert AudioReviewReason.SPEECH_AT_CLIP_END in region.review_reasons
    codes = {c.concern_code for c in qc.concerns}
    assert "CLIPPED_OPENING_WORD" in codes and "CLIPPED_FINAL_WORD" in codes


# ------------------------------------------------------------ parsing units

def test_parse_transcription_decimal_mapping() -> None:
    result = parse_transcription(MOCK_TRANSCRIPTION, Fraction(1, 2))
    word = result.segments[0].words[0]
    assert word.asr_start_seconds == "0.30"
    assert word.start_annotation_time == Fraction(1, 2) + Fraction("0.30")
    assert result.task == "transcribe"


def test_parse_transcription_failure_keeps_reason() -> None:
    result = parse_transcription({"status": "error", "error": "boom"}, Fraction(0))
    assert result.status == ASRStatus.FAILED
    assert result.runtime.failure_reason == "boom"


def test_normalize_text_policy() -> None:
    assert normalize_text("  Hello   there. ") == "Hello there."
    assert normalize_text("Hello there.") != normalize_text("hello there.")  # case preserved


def test_parse_alignment_word_order_validation() -> None:
    source = parse_transcription(MOCK_TRANSCRIPTION, Fraction(0)).segments
    aligned = parse_alignment(_mock_alignment_response(), source, Fraction(0))
    assert aligned.status == AlignmentStatus.ALIGNED
    words = [w for s in aligned.segments for w in s.words]
    assert all(
        b.start_annotation_time >= a.start_annotation_time
        for a, b in itertools.pairwise(words)
    )


# ------------------------------------------------------------ pipeline / status

def test_full_audit_audio_status_propagation(tmp_path: Path) -> None:
    clip = _make_av(
        "au_sine48k.mp4",
        f"testsrc2=duration=2:rate=24:size={SIZE}",
        ["-f", "lavfi", "-i", "sine=frequency=440:duration=2:sample_rate=48000"],
        ["-c:a", "aac", "-ac", "2", "-ar", "48000", "-shortest"],
    )
    result = run_audit(
        clip, artifacts_root=tmp_path, shot_analysis=True,
        extract_shot_evidence=False, audio_analysis=True, asr_enabled=False,
    )
    assert result.audio_truth is not None
    # Uncovered tonal audio without ASR → review → top-level REVIEW_REQUIRED.
    assert result.audio_truth.overall_status == "REVIEW_REQUIRED"
    assert result.status == RunStatus.REVIEW_REQUIRED
    # All audio artifacts are manifest-hashed.
    assert result.manifest is not None
    manifest_paths = {a.path for a in result.manifest.artifacts}
    assert "audio/source.wav" in manifest_paths
    assert "audio/audio_qc.json" in manifest_paths
    assert "audio/waveform.png" in manifest_paths


def test_no_audio_video_full_audit_still_passes(tmp_path: Path, clip_24fps: Path) -> None:
    result = run_audit(
        clip_24fps, artifacts_root=tmp_path, shot_analysis=True,
        extract_shot_evidence=False, audio_analysis=True, asr_enabled=True,
    )
    assert result.audio_truth is not None
    assert result.audio_truth.overall_status == "NO_AUDIO_STREAM"
    assert result.status == RunStatus.PASS


def test_worker_env_pyprojects_are_pinned() -> None:
    fw = (Path("engine/manuscript_reviewer/audio/asr/workers/fw_env/pyproject.toml")
          .read_text(encoding="utf-8"))
    wx = (Path("engine/manuscript_reviewer/audio/asr/workers/wx_env/pyproject.toml")
          .read_text(encoding="utf-8"))
    assert "faster-whisper==" in fw
    assert "whisperx==" in wx


def test_no_cloud_or_descript_paths() -> None:
    """Privacy sweep: no cloud transcription or Descript code path exists."""
    import re

    engine_root = Path(__file__).parent.parent / "engine" / "manuscript_reviewer"
    # Word-bounded so "description"/"describes" never match "descript".
    banned = [
        re.compile(r"\bdescript\b"),
        re.compile(r"\bopenai\b"),
        re.compile(r"\bgemini\b"),
        re.compile(r"requests\.post"),
        re.compile(r"httpx\.post"),
        re.compile(r"drive\.google"),
        re.compile(r"\bupload\b"),
    ]
    offenders = []
    for path in engine_root.rglob("*.py"):
        if ".venv" in path.parts:
            continue
        text = path.read_text(encoding="utf-8").lower()
        for pattern in banned:
            match = pattern.search(text)
            if match and "never" not in text[max(0, match.start() - 200) : match.start()]:
                offenders.append(f"{path.name}: {pattern.pattern}")
    assert offenders == []


def test_worker_uses_transcribe_never_translate() -> None:
    worker = (Path("engine/manuscript_reviewer/audio/asr/workers/fw_env/worker.py")
              .read_text(encoding="utf-8"))
    assert 'task="transcribe"' in worker
    assert 'task="translate"' not in worker


@requires_ffmpeg
def test_scdet_error_handling_unchanged(tmp_path: Path) -> None:
    """Regression guard: audio stage must not break --no-audio-analysis runs."""
    clip = _make_av(
        "au_sine48k.mp4",
        f"testsrc2=duration=2:rate=24:size={SIZE}",
        ["-f", "lavfi", "-i", "sine=frequency=440:duration=2:sample_rate=48000"],
        ["-c:a", "aac", "-ac", "2", "-ar", "48000", "-shortest"],
    )
    result = run_audit(clip, artifacts_root=tmp_path, shot_analysis=True,
                       extract_shot_evidence=False, audio_analysis=False)
    assert result.audio_truth is None
    assert result.status == RunStatus.PASS
