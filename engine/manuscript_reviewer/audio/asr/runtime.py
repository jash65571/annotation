"""ASR worker orchestration: isolated uv environments, JSON protocol, typed results.

The core engine invokes workers with structured argv (no shell mode, ever) via
``uv run --project <worker_env> python worker.py --request ... --response ...``.
uv creates/syncs the isolated environment on first use — that is the permitted
bootstrap network traffic (packages/models only; task media is NEVER uploaded).

With bootstrap disabled, a missing worker environment records ASR_UNAVAILABLE
and the evidence pipeline continues.
"""

from __future__ import annotations

import json
import logging
import re
import shutil
import time
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from fractions import Fraction
from pathlib import Path
from typing import Any, Protocol

from ...media.ffmpeg_tools import ToolExecutionError, run_tool
from ...models.audio import (
    AlignmentResult,
    AlignmentStatus,
    ASRResult,
    ASRRuntimeInfo,
    ASRSegment,
    ASRStatus,
    ASRWord,
    LanguageEvidence,
    LanguageReviewStatus,
)

logger = logging.getLogger(__name__)

WORKERS_DIR = Path(__file__).parent / "workers"
FW_ENV = WORKERS_DIR / "fw_env"
WX_ENV = WORKERS_DIR / "wx_env"

#: Language probability below this stays REVIEW_REQUIRED.
LANGUAGE_CONFIDENCE_THRESHOLD = 0.8


@dataclass(frozen=True)
class ASRConfig:
    model: str = "large-v3-turbo"
    device: str = "auto"
    compute_type: str = "auto"
    language: str | None = None
    bootstrap: bool = True
    timeout_seconds: float = 3600.0


class WorkerUnavailableError(RuntimeError):
    """Worker environment missing and bootstrap disabled."""


def _decimal_str(value: Any) -> str:
    """Normalize an ASR-reported float time via Decimal(str(...)). Estimates only."""
    try:
        return str(Decimal(str(value)))
    except (InvalidOperation, ValueError, TypeError):
        return "0"


def decimal_to_fraction(text: str) -> Fraction:
    return Fraction(Decimal(text))


def _uv_executable() -> Path:
    found = shutil.which("uv")
    if not found:
        raise WorkerUnavailableError("uv not found on PATH; cannot run ASR workers")
    return Path(found)


def _run_worker(
    env_dir: Path, request: dict[str, Any], config: ASRConfig, scratch: Path
) -> dict[str, Any]:
    """Invoke one worker; returns its parsed JSON response (or raises)."""
    uv = _uv_executable()
    if not config.bootstrap and not (env_dir / ".venv").exists():
        raise WorkerUnavailableError(
            f"Worker env {env_dir.name} not bootstrapped and --no-asr-bootstrap set"
        )
    scratch.mkdir(parents=True, exist_ok=True)
    request_path = scratch / f"{env_dir.name}_request.json"
    response_path = scratch / f"{env_dir.name}_response.json"
    request_path.write_text(json.dumps(request, ensure_ascii=False), encoding="utf-8")
    args = [
        "run",
        "--project", str(env_dir),
        *(["--no-sync"] if not config.bootstrap else []),
        "python", str(env_dir / "worker.py"),
        "--request", str(request_path),
        "--response", str(response_path),
    ]
    # Structured argv through the single safe subprocess wrapper; no shell mode.
    try:
        run_tool(uv, args, timeout=config.timeout_seconds)
    except ToolExecutionError as exc:
        # Workers exit non-zero on handled errors but still write a response
        # JSON carrying the structured failure reason.
        if not response_path.exists():
            raise RuntimeError(
                f"Worker {env_dir.name} produced no response: {exc.stderr[-2000:]}"
            ) from exc
    if not response_path.exists():
        raise RuntimeError(f"Worker {env_dir.name} produced no response")
    return json.loads(response_path.read_text(encoding="utf-8"))  # type: ignore[no-any-return]


class TranscriptionAdapter(Protocol):
    """Injectable transcription interface (mocked in tests)."""

    def transcribe(
        self, asr_wav: Path, config: ASRConfig, scratch: Path
    ) -> dict[str, Any]: ...


class AlignmentAdapter(Protocol):
    def align(
        self,
        asr_wav: Path,
        language: str,
        segments: list[dict[str, Any]],
        config: ASRConfig,
        scratch: Path,
    ) -> dict[str, Any]: ...


class FasterWhisperAdapter:
    """Real faster-whisper worker invocation."""

    def transcribe(
        self, asr_wav: Path, config: ASRConfig, scratch: Path
    ) -> dict[str, Any]:
        request = {
            "audio_path": str(asr_wav),
            "model": config.model,
            "device": config.device,
            "compute_type": config.compute_type,
            "language": config.language,
            "vad": True,
        }
        return _run_worker(FW_ENV, request, config, scratch)


class WhisperXAdapter:
    """Real WhisperX forced-alignment worker invocation."""

    def align(
        self,
        asr_wav: Path,
        language: str,
        segments: list[dict[str, Any]],
        config: ASRConfig,
        scratch: Path,
    ) -> dict[str, Any]:
        request = {
            "audio_path": str(asr_wav),
            "language": language,
            "device": "cpu",
            "segments": segments,
        }
        return _run_worker(WX_ENV, request, config, scratch)


def _words_from_raw(
    raw_words: list[dict[str, Any]], offset: Fraction, timing_source: str
) -> list[ASRWord]:
    words: list[ASRWord] = []
    for index, word in enumerate(raw_words):
        if word.get("start") is None or word.get("end") is None:
            continue
        start_text = _decimal_str(word["start"])
        end_text = _decimal_str(word["end"])
        words.append(
            ASRWord(
                word_index=index,
                text=str(word.get("text", "")).strip(),
                asr_start_seconds=start_text,
                asr_end_seconds=end_text,
                start_annotation_time=offset + decimal_to_fraction(start_text),
                end_annotation_time=offset + decimal_to_fraction(end_text),
                probability=word.get("probability"),
                timing_source=timing_source,
            )
        )
    return words


def parse_transcription(
    response: dict[str, Any], annotation_audio_offset: Fraction
) -> ASRResult:
    """Typed ASRResult from a worker response. ASR times are estimates mapped
    onto the annotation clock via Decimal(str(...)) → Fraction."""
    runtime = ASRRuntimeInfo(
        engine="faster_whisper",
        package_version=response.get("package_version"),
        model_name=response.get("model"),
        device=response.get("device"),
        compute_type=response.get("compute_type"),
        runtime_seconds=response.get("transcribe_seconds"),
        failure_reason=response.get("error"),
    )
    if response.get("status") != "ok":
        return ASRResult(status=ASRStatus.FAILED, runtime=runtime)

    probability = response.get("language_probability")
    language = LanguageEvidence(
        language_candidate=response.get("language"),
        language_probability=probability,
        language_source="faster_whisper",
        language_review_status=(
            LanguageReviewStatus.SUPPORTED_BY_ASR
            if probability is not None and probability >= LANGUAGE_CONFIDENCE_THRESHOLD
            else LanguageReviewStatus.REVIEW_REQUIRED
            if response.get("language")
            else LanguageReviewStatus.UNKNOWN
        ),
    )
    segments: list[ASRSegment] = []
    for raw in response.get("segments", []):
        start_text = _decimal_str(raw.get("start"))
        end_text = _decimal_str(raw.get("end"))
        segments.append(
            ASRSegment(
                segment_id=int(raw.get("id", len(segments))),
                text=str(raw.get("text", "")).strip(),
                asr_start_seconds=start_text,
                asr_end_seconds=end_text,
                start_annotation_time=annotation_audio_offset
                + decimal_to_fraction(start_text),
                end_annotation_time=annotation_audio_offset + decimal_to_fraction(end_text),
                avg_logprob=raw.get("avg_logprob"),
                no_speech_prob=raw.get("no_speech_prob"),
                words=_words_from_raw(
                    raw.get("words", []), annotation_audio_offset, "faster_whisper"
                ),
            )
        )
    return ASRResult(status=ASRStatus.PASS, runtime=runtime, language=language,
                     segments=segments)


_NORMALIZE_RE = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    """Documented normalization for the ALIGNMENT text-preservation check:
    whitespace collapsed, surrounding space stripped, case preserved."""
    return _NORMALIZE_RE.sub(" ", text).strip()


def parse_alignment(
    response: dict[str, Any],
    source_segments: list[ASRSegment],
    annotation_audio_offset: Fraction,
) -> AlignmentResult:
    """Typed AlignmentResult; wording MUST match the faster-whisper source."""
    runtime = ASRRuntimeInfo(
        engine="whisperx",
        package_version=response.get("package_version"),
        model_name=response.get("align_model"),
        device=response.get("device"),
        runtime_seconds=response.get("align_seconds"),
        failure_reason=response.get("error"),
    )
    if response.get("status") != "ok":
        return AlignmentResult(
            status=AlignmentStatus.FAILED,
            runtime=runtime,
            failure_reason=response.get("error"),
        )

    segments: list[ASRSegment] = []
    for raw in response.get("segments", []):
        start_text = _decimal_str(raw.get("start"))
        end_text = _decimal_str(raw.get("end"))
        segments.append(
            ASRSegment(
                segment_id=int(raw.get("id", len(segments))),
                text=str(raw.get("text", "")).strip(),
                asr_start_seconds=start_text,
                asr_end_seconds=end_text,
                start_annotation_time=annotation_audio_offset
                + decimal_to_fraction(start_text),
                end_annotation_time=annotation_audio_offset + decimal_to_fraction(end_text),
                words=_words_from_raw(
                    raw.get("words", []), annotation_audio_offset, "whisperx"
                ),
            )
        )

    source_text = normalize_text(" ".join(s.text for s in source_segments))
    aligned_text = normalize_text(" ".join(s.text for s in segments))
    preserved = source_text == aligned_text
    return AlignmentResult(
        status=AlignmentStatus.ALIGNED if preserved else AlignmentStatus.TEXT_MISMATCH,
        runtime=runtime,
        segments=segments,
        text_preserved=preserved,
        failure_reason=None if preserved else "aligned text differs from source transcript",
    )


def model_cache_state(model: str) -> str:
    """Best-effort: was the faster-whisper model already in the HF cache?"""
    cache_root = Path.home() / ".cache" / "huggingface" / "hub"
    marker = f"models--Systran--faster-whisper-{model}"
    turbo_marker = f"models--mobiuslabsgmbh--faster-whisper-{model}"
    for name in (marker, turbo_marker):
        if (cache_root / name).exists():
            return "cached"
    return "not_cached"


def measure_runtime(start: float) -> float:
    return round(time.perf_counter() - start, 3)
