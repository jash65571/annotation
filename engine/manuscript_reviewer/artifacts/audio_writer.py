"""Phase 3 audio artifact writing. Same determinism rules as the other writers.
No empty fake artifacts: stages that did not run write nothing (their absence
is recorded in asr_status.json / audio_qc.json)."""

from __future__ import annotations

import csv
import json
from fractions import Fraction
from pathlib import Path
from typing import Any

from ..media.timestamps import seconds_to_decimal
from ..models.audio import (
    AlignmentResult,
    ASRResult,
    ASRSegment,
    AudioEnergyBin,
    AudioFrameRecord,
    AudioQCResult,
    AudioReviewItem,
    AudioTimeline,
    BoundaryAudioEvidence,
    SpeechRegion,
)
from .writer import ArtifactWriteError


def _write_json(path: Path, payload: Any) -> Path:
    try:
        with path.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, ensure_ascii=False)
            handle.write("\n")
    except OSError as exc:
        raise ArtifactWriteError(f"Failed to write {path}: {exc}") from exc
    return path


def _dec(value: Fraction | None) -> str:
    return str(seconds_to_decimal(value)) if value is not None else ""


def write_audio_frames(
    audio_dir: Path, frames: list[AudioFrameRecord]
) -> list[Path]:
    csv_path = audio_dir / "audio_frames.csv"
    try:
        with csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                ["audio_frame_index", "pts", "pts_time_source", "annotation_time",
                 "duration", "duration_time", "nb_samples"]
            )
            for frame in frames:
                writer.writerow(
                    [
                        frame.audio_frame_index,
                        frame.pts if frame.pts is not None else "",
                        _dec(frame.pts_time_source),
                        _dec(frame.annotation_time),
                        frame.duration if frame.duration is not None else "",
                        _dec(frame.duration_time),
                        frame.nb_samples if frame.nb_samples is not None else "",
                    ]
                )
    except OSError as exc:
        raise ArtifactWriteError(f"Failed to write {csv_path}: {exc}") from exc

    jsonl_path = audio_dir / "audio_frames.jsonl"
    try:
        with jsonl_path.open("w", encoding="utf-8", newline="\n") as handle:
            for frame in frames:
                handle.write(json.dumps(frame.model_dump(mode="json"), sort_keys=True) + "\n")
    except OSError as exc:
        raise ArtifactWriteError(f"Failed to write {jsonl_path}: {exc}") from exc
    return [csv_path, jsonl_path]


def write_audio_timeline(audio_dir: Path, timeline: AudioTimeline) -> Path:
    return _write_json(audio_dir / "audio_timeline.json", timeline.model_dump(mode="json"))


def write_energy_csv(audio_dir: Path, bins: list[AudioEnergyBin]) -> Path:
    path = audio_dir / "audio_energy_10ms.csv"
    try:
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                ["bin_index", "start_sample", "end_sample", "start_annotation_time",
                 "end_annotation_time", "rms", "peak", "dbfs", "zero_crossing_rate",
                 "spectral_centroid_hz", "spectral_flatness"]
            )
            for item in bins:
                writer.writerow(
                    [
                        item.bin_index, item.start_sample, item.end_sample,
                        _dec(item.start_annotation_time), _dec(item.end_annotation_time),
                        item.rms, item.peak, item.dbfs, item.zero_crossing_rate,
                        item.spectral_centroid_hz, item.spectral_flatness,
                    ]
                )
    except OSError as exc:
        raise ArtifactWriteError(f"Failed to write {path}: {exc}") from exc
    return path


def write_regions_json(audio_dir: Path, qc: AudioQCResult) -> list[Path]:
    paths = [
        _write_json(
            audio_dir / "audio_regions.json",
            {"regions": [r.model_dump(mode="json") for r in qc.regions]},
        ),
        _write_json(
            audio_dir / "transient_candidates.json",
            {"transients": [t.model_dump(mode="json") for t in qc.transients]},
        ),
    ]
    return paths


def write_speech_regions_csv(audio_dir: Path, regions: list[SpeechRegion]) -> Path:
    path = audio_dir / "speech_regions.csv"
    try:
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                ["region_id", "start_exact", "end_exact", "start_manuscript",
                 "end_manuscript", "sources", "text_candidate", "language_candidate",
                 "mean_word_probability", "alignment_status", "state",
                 "possible_overlap", "review_reasons"]
            )
            for region in regions:
                writer.writerow(
                    [
                        region.region_id,
                        _dec(region.start_exact),
                        _dec(region.end_exact),
                        region.start_manuscript,
                        region.end_manuscript,
                        "|".join(region.sources),
                        region.text_candidate or "",
                        region.language.language_candidate if region.language else "",
                        region.mean_word_probability
                        if region.mean_word_probability is not None
                        else "",
                        region.alignment_status.value,
                        region.state.value,
                        int(region.possible_overlap),
                        "|".join(r.value for r in region.review_reasons),
                    ]
                )
    except OSError as exc:
        raise ArtifactWriteError(f"Failed to write {path}: {exc}") from exc
    return path


def write_boundary_evidence(
    audio_dir: Path, evidence: list[BoundaryAudioEvidence]
) -> Path:
    return _write_json(
        audio_dir / "boundary_audio_evidence.json",
        {"boundaries": [b.model_dump(mode="json") for b in evidence]},
    )


def write_review_queue(audio_dir: Path, items: list[AudioReviewItem]) -> Path:
    return _write_json(
        audio_dir / "audio_review_queue.json",
        {"items": [i.model_dump(mode="json") for i in items]},
    )


def write_audio_qc(audio_dir: Path, qc: AudioQCResult) -> Path:
    return _write_json(audio_dir / "audio_qc.json", qc.model_dump(mode="json"))


def _segments_csv(path: Path, segments: list[ASRSegment]) -> Path:
    try:
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                ["segment_id", "start_annotation_time", "end_annotation_time",
                 "asr_start", "asr_end", "avg_logprob", "no_speech_prob", "text"]
            )
            for segment in segments:
                writer.writerow(
                    [
                        segment.segment_id,
                        _dec(segment.start_annotation_time),
                        _dec(segment.end_annotation_time),
                        segment.asr_start_seconds,
                        segment.asr_end_seconds,
                        segment.avg_logprob if segment.avg_logprob is not None else "",
                        segment.no_speech_prob if segment.no_speech_prob is not None else "",
                        segment.text,
                    ]
                )
    except OSError as exc:
        raise ArtifactWriteError(f"Failed to write {path}: {exc}") from exc
    return path


def _words_csv(path: Path, segments: list[ASRSegment]) -> Path:
    try:
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                ["segment_id", "word_index", "start_annotation_time",
                 "end_annotation_time", "asr_start", "asr_end", "probability",
                 "timing_source", "text"]
            )
            for segment in segments:
                for word in segment.words:
                    writer.writerow(
                        [
                            segment.segment_id,
                            word.word_index,
                            _dec(word.start_annotation_time),
                            _dec(word.end_annotation_time),
                            word.asr_start_seconds,
                            word.asr_end_seconds,
                            word.probability if word.probability is not None else "",
                            word.timing_source,
                            word.text,
                        ]
                    )
    except OSError as exc:
        raise ArtifactWriteError(f"Failed to write {path}: {exc}") from exc
    return path


def write_asr_artifacts(
    asr_dir: Path,
    result: ASRResult,
    alignment: AlignmentResult | None,
    best_segments: list[ASRSegment],
) -> list[Path]:
    """Write ASR/alignment/best artifacts. transcript_best carries
    verification_status = ASR_EVIDENCE_ONLY — never implies human verification."""
    asr_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    paths.append(
        _write_json(
            asr_dir / "asr_status.json",
            {
                "asr_status": result.status.value,
                "alignment_status": alignment.status.value if alignment else "NOT_ATTEMPTED",
                "task": result.task,
                "vad_enabled": result.vad_enabled,
                "failure_reason": result.runtime.failure_reason,
                "alignment_failure_reason": alignment.failure_reason if alignment else None,
                "verification_status": "ASR_EVIDENCE_ONLY",
            },
        )
    )
    paths.append(
        _write_json(
            asr_dir / "runtime.json",
            {
                "faster_whisper": result.runtime.model_dump(mode="json"),
                "whisperx": alignment.runtime.model_dump(mode="json") if alignment else None,
            },
        )
    )

    if result.segments:
        paths.append(
            _write_json(
                asr_dir / "transcript_faster_whisper.json",
                {
                    "language": result.language.model_dump(mode="json")
                    if result.language
                    else None,
                    "segments": [s.model_dump(mode="json") for s in result.segments],
                },
            )
        )
        text = "\n".join(s.text for s in result.segments)
        (asr_dir / "transcript_faster_whisper.txt").write_text(text + "\n", encoding="utf-8")
        paths.append(asr_dir / "transcript_faster_whisper.txt")
        paths.append(_segments_csv(asr_dir / "segments_faster_whisper.csv", result.segments))
        paths.append(_words_csv(asr_dir / "words_faster_whisper.csv", result.segments))

    if alignment is not None and alignment.segments:
        paths.append(
            _write_json(
                asr_dir / "transcript_whisperx_aligned.json",
                {"segments": [s.model_dump(mode="json") for s in alignment.segments]},
            )
        )
        text = "\n".join(s.text for s in alignment.segments)
        (asr_dir / "transcript_whisperx_aligned.txt").write_text(text + "\n", encoding="utf-8")
        paths.append(asr_dir / "transcript_whisperx_aligned.txt")
        paths.append(
            _segments_csv(asr_dir / "segments_whisperx_aligned.csv", alignment.segments)
        )
        paths.append(_words_csv(asr_dir / "words_whisperx_aligned.csv", alignment.segments))

    if best_segments:
        header = "# verification_status = ASR_EVIDENCE_ONLY (machine evidence, not verified)\n"
        text = "\n".join(s.text for s in best_segments)
        (asr_dir / "transcript_best.txt").write_text(header + text + "\n", encoding="utf-8")
        paths.append(asr_dir / "transcript_best.txt")
        paths.append(_segments_csv(asr_dir / "segments_best.csv", best_segments))
        paths.append(_words_csv(asr_dir / "words_best.csv", best_segments))

    return paths
