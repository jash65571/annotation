"""Audio Truth Engine orchestration.

VIDEO → stream verification → audio frame ledger → lossless PCM → waveform /
10 ms energy / spectrogram → regions → faster-whisper → WhisperX alignment →
speech regions + VAD-recall defense → boundary audio continuity → review
queue → audio QC. ASR failure at any point degrades to evidence-only analysis
and the pipeline continues — never a cloud/Descript fallback.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path

from ..artifacts import audio_writer
from ..media.clock import AnnotationClock
from ..media.ffmpeg_tools import ToolExecutionError
from ..models.audio import (
    AlignmentResult,
    AlignmentStatus,
    ASRResult,
    ASRRuntimeInfo,
    ASRSegment,
    ASRStatus,
    AudioConcernCandidate,
    AudioQCResult,
    AudioReviewReason,
    AudioStatus,
)
from ..models.frame import FrameLedger
from ..models.media import MediaInfo
from ..models.shot_truth import ShotTruthResult
from ..models.validation import Severity, ValidatorIssue
from ..validation import audio_validator
from . import boundary_audio as boundary_mod
from . import regions as regions_mod
from .asr.runtime import (
    AlignmentAdapter,
    ASRConfig,
    FasterWhisperAdapter,
    TranscriptionAdapter,
    WhisperXAdapter,
    WorkerUnavailableError,
    model_cache_state,
    parse_alignment,
    parse_transcription,
)
from .decode import AudioDecodeError, extract_wav, mono_float
from .metrics import compute_energy_bins
from .probe import AudioProbeError, enumerate_audio_frames, stream_initial_padding
from .render import render_energy, render_spectrogram, render_waveform
from .review_queue import build_review_queue, render_review_clips
from .speech import build_speech_regions, find_uncovered_audio
from .timeline import build_audio_timeline, cross_check_sample_count

logger = logging.getLogger(__name__)


@dataclass
class AudioAnalysisOutput:
    result: AudioQCResult | None
    issues: list[ValidatorIssue]
    artifact_paths: list[Path]
    stage_timings: dict[str, float] = field(default_factory=dict)


def _timed(sink: dict[str, float], name: str, start: float) -> None:
    sink[name] = round(time.perf_counter() - start, 4)


def run_audio_analysis(
    video_path: Path,
    run_dir: Path,
    media: MediaInfo,
    ledger: FrameLedger,
    shot_result: ShotTruthResult | None,
    annotation_endpoint: Fraction | None,
    asr_enabled: bool = True,
    asr_config: ASRConfig | None = None,
    transcriber: TranscriptionAdapter | None = None,
    aligner: AlignmentAdapter | None = None,
    render_clips: bool = True,
) -> AudioAnalysisOutput:
    """Run the full Phase 3 audio stage and write artifacts into run_dir/audio."""
    issues: list[ValidatorIssue] = []
    artifacts: list[Path] = []
    timings: dict[str, float] = {}
    clock = AnnotationClock.from_ledger(ledger)
    config = asr_config or ASRConfig()

    # --- NO-AUDIO media is valid: no fake artifacts, no failure. ---
    if not media.audio_streams:
        qc = AudioQCResult(
            audio_status=AudioStatus.NO_AUDIO_STREAM,
            asr_status=ASRStatus.DISABLED,
            alignment_status=AlignmentStatus.NOT_ATTEMPTED,
            overall_status="NO_AUDIO_STREAM",
        )
        audio_dir = run_dir / "audio"
        audio_dir.mkdir(parents=True, exist_ok=True)
        artifacts.append(audio_writer.write_audio_qc(audio_dir, qc))
        return AudioAnalysisOutput(result=qc, issues=issues, artifact_paths=artifacts,
                                   stage_timings=timings)

    stream = media.audio_streams[0]
    audio_dir = run_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    # --- audio frame ledger ---
    start = time.perf_counter()
    try:
        time_base = stream.time_base
        if time_base is None:
            raise AudioProbeError("Audio stream has no time base")
        audio_frames = enumerate_audio_frames(video_path, time_base, clock.origin)
        initial_padding = stream_initial_padding(video_path)
    except AudioProbeError as exc:
        issues.append(
            ValidatorIssue(
                rule_id="P3-AUDIO-001",
                severity=Severity.FAIL,
                location=video_path.name,
                message=f"Audio frame enumeration failed: {exc}",
            )
        )
        return AudioAnalysisOutput(result=None, issues=issues, artifact_paths=artifacts,
                                   stage_timings=timings)
    _timed(timings, "audio_frame_enumeration", start)
    artifacts.extend(audio_writer.write_audio_frames(audio_dir, audio_frames))

    # --- PCM extraction: source evidence + ASR input ---
    start = time.perf_counter()
    try:
        source_wav = extract_wav(video_path, audio_dir / "source.wav")
    except (AudioDecodeError, ToolExecutionError) as exc:
        issues.append(
            ValidatorIssue(
                rule_id="P3-AUDIO-001",
                severity=Severity.FAIL,
                location=video_path.name,
                message=f"Source audio extraction failed: {exc}",
            )
        )
        return AudioAnalysisOutput(result=None, issues=issues, artifact_paths=artifacts,
                                   stage_timings=timings)
    artifacts.append(source_wav.path)
    asr_wav = None
    if asr_enabled:
        asr_wav = extract_wav(
            video_path, audio_dir / "asr.wav", sample_rate=16000, mono=True
        )
        artifacts.append(asr_wav.path)
    _timed(timings, "audio_pcm_extraction", start)

    # --- timeline + cross-check ---
    start = time.perf_counter()
    timeline = build_audio_timeline(
        stream, audio_frames, source_wav, clock, initial_padding, asr_wav
    )
    issues.extend(audio_validator.validate_audio_timeline(timeline, audio_frames))
    issues.extend(cross_check_sample_count(timeline, audio_frames, stream))
    artifacts.append(audio_writer.write_audio_timeline(audio_dir, timeline))
    _timed(timings, "audio_timeline", start)

    # --- metrics / renders ---
    start = time.perf_counter()
    mono = mono_float(source_wav)
    bins = compute_energy_bins(mono, timeline)
    issues.extend(audio_validator.validate_energy_bins(bins, timeline))
    artifacts.append(audio_writer.write_energy_csv(audio_dir, bins))
    _timed(timings, "audio_metrics", start)

    start = time.perf_counter()
    artifacts.append(render_waveform(mono, timeline, audio_dir / "waveform.png"))
    _timed(timings, "audio_waveform", start)
    start = time.perf_counter()
    artifacts.append(render_spectrogram(mono, timeline, audio_dir / "spectrogram.png"))
    _timed(timings, "audio_spectrogram", start)
    start = time.perf_counter()
    artifacts.append(
        render_energy(bins, timeline, audio_dir / "audio_energy.png",
                      regions_mod.SILENCE_DBFS)
    )
    _timed(timings, "audio_energy_plot", start)

    # --- deterministic regions ---
    start = time.perf_counter()
    audio_regions = regions_mod.detect_regions(bins, timeline)
    transients = regions_mod.detect_transients(bins, timeline)
    _timed(timings, "audio_regions", start)

    # --- ASR: faster-whisper (primary lead) ---
    asr_result: ASRResult
    alignment: AlignmentResult | None = None
    if not asr_enabled:
        asr_result = ASRResult(
            status=ASRStatus.DISABLED,
            runtime=ASRRuntimeInfo(engine="faster_whisper",
                                   failure_reason="ASR disabled by --no-asr"),
        )
    else:
        start = time.perf_counter()
        adapter = transcriber or FasterWhisperAdapter()
        cache_state = model_cache_state(config.model)
        try:
            response = adapter.transcribe(
                audio_dir / "asr.wav", config, run_dir / "audio" / "_scratch"
            )
            asr_result = parse_transcription(response, timeline.annotation_audio_offset)
        except WorkerUnavailableError as exc:
            asr_result = ASRResult(
                status=ASRStatus.UNAVAILABLE,
                runtime=ASRRuntimeInfo(engine="faster_whisper",
                                       failure_reason=str(exc)),
            )
        except (RuntimeError, OSError, ValueError) as exc:
            asr_result = ASRResult(
                status=ASRStatus.FAILED,
                runtime=ASRRuntimeInfo(engine="faster_whisper",
                                       failure_reason=f"{type(exc).__name__}: {exc}"),
            )
        asr_result.runtime.model_cache_state = cache_state
        _timed(timings, "audio_faster_whisper", start)

        # --- WhisperX forced alignment (wording preserved, never replaced) ---
        if (
            asr_result.status == ASRStatus.PASS
            and asr_result.segments
            and asr_result.language is not None
            and asr_result.language.language_candidate
        ):
            start = time.perf_counter()
            align_adapter = aligner or WhisperXAdapter()
            fw_segments = [
                {
                    "start": s.asr_start_seconds,
                    "end": s.asr_end_seconds,
                    "text": s.text,
                }
                for s in asr_result.segments
            ]
            try:
                align_response = align_adapter.align(
                    audio_dir / "asr.wav",
                    asr_result.language.language_candidate,
                    fw_segments,
                    config,
                    run_dir / "audio" / "_scratch",
                )
                alignment = parse_alignment(
                    align_response, asr_result.segments, timeline.annotation_audio_offset
                )
            except WorkerUnavailableError as exc:
                alignment = AlignmentResult(
                    status=AlignmentStatus.UNAVAILABLE,
                    runtime=ASRRuntimeInfo(engine="whisperx", failure_reason=str(exc)),
                    failure_reason=str(exc),
                )
            except (RuntimeError, OSError, ValueError) as exc:
                alignment = AlignmentResult(
                    status=AlignmentStatus.FAILED,
                    runtime=ASRRuntimeInfo(
                        engine="whisperx",
                        failure_reason=f"{type(exc).__name__}: {exc}",
                    ),
                    failure_reason=str(exc),
                )
            _timed(timings, "audio_whisperx", start)

    issues.extend(audio_validator.validate_asr(asr_result, timeline))
    issues.extend(audio_validator.validate_alignment(alignment))

    # --- transcript_best: faster-whisper wording + best valid timing ---
    best_segments: list[ASRSegment]
    if alignment is not None and alignment.status == AlignmentStatus.ALIGNED:
        best_segments = alignment.segments
    else:
        best_segments = asr_result.segments
    artifacts.extend(
        audio_writer.write_asr_artifacts(audio_dir / "asr", asr_result, alignment,
                                         best_segments)
    )

    # --- speech regions + VAD recall defense ---
    start = time.perf_counter()
    speech_regions = build_speech_regions(
        asr_result, alignment, timeline, annotation_endpoint
    )
    issues.extend(audio_validator.validate_speech_regions(speech_regions, timeline))
    uncovered = find_uncovered_audio(audio_regions, speech_regions, audio_regions)
    artifacts.append(audio_writer.write_speech_regions_csv(audio_dir, speech_regions))
    _timed(timings, "audio_speech_regions", start)

    # --- shot-boundary audio continuity ---
    start = time.perf_counter()
    shot_candidates = shot_result.candidates if shot_result is not None else []
    boundary_evidence = boundary_mod.analyze_boundary_audio(
        shot_candidates, clock, timeline, bins, audio_regions, speech_regions
    )
    issues.extend(
        audio_validator.validate_boundaries(shot_candidates, boundary_evidence, True)
    )
    artifacts.append(audio_writer.write_boundary_evidence(audio_dir, boundary_evidence))
    _timed(timings, "audio_boundary_continuity", start)

    # --- review queue + clips ---
    start = time.perf_counter()
    asr_unavailable = asr_result.status in (ASRStatus.UNAVAILABLE, ASRStatus.FAILED)
    review_items = build_review_queue(
        timeline, speech_regions, uncovered, transients, boundary_evidence,
        audio_regions, asr_unavailable
    )
    if render_clips and review_items:
        clips = render_review_clips(
            review_items, source_wav, timeline, audio_dir / "review_clips"
        )
        artifacts.extend(clips)
    artifacts.append(audio_writer.write_review_queue(audio_dir, review_items))
    _timed(timings, "audio_review_queue", start)

    # --- concerns (structured leads, never prose) ---
    concerns: list[AudioConcernCandidate] = []
    if asr_unavailable:
        concerns.append(
            AudioConcernCandidate(
                concern_code="ASR_UNAVAILABLE",
                detail=asr_result.runtime.failure_reason,
            )
        )
    for item in review_items:
        if AudioReviewReason.SPEECH_AT_CLIP_START in item.reasons:
            concerns.append(
                AudioConcernCandidate(
                    concern_code="CLIPPED_OPENING_WORD",
                    start_exact=item.start_exact,
                    end_exact=item.end_exact,
                )
            )
        if AudioReviewReason.SPEECH_AT_CLIP_END in item.reasons:
            concerns.append(
                AudioConcernCandidate(
                    concern_code="CLIPPED_FINAL_WORD",
                    start_exact=item.start_exact,
                    end_exact=item.end_exact,
                )
            )

    overall = audio_validator.compute_audio_status(
        issues, review_items, boundary_evidence, no_audio=False
    )
    qc = AudioQCResult(
        audio_status=AudioStatus.ANALYZED,
        asr_status=asr_result.status,
        alignment_status=alignment.status if alignment else AlignmentStatus.NOT_ATTEMPTED,
        overall_status=overall,
        timeline=timeline,
        energy_bin_count=len(bins),
        region_count=len(audio_regions),
        transient_count=len(transients),
        speech_region_count=len(speech_regions),
        review_item_count=len(review_items),
        boundaries_checked=len(boundary_evidence),
        language=asr_result.language,
        regions=audio_regions,
        transients=transients,
        speech_regions=speech_regions,
        boundary_evidence=boundary_evidence,
        review_items=review_items,
        concerns=concerns,
    )
    artifacts.extend(audio_writer.write_regions_json(audio_dir, qc))
    artifacts.append(audio_writer.write_audio_qc(audio_dir, qc))
    return AudioAnalysisOutput(result=qc, issues=issues, artifact_paths=artifacts,
                               stage_timings=timings)
