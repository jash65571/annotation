"""ASR worker orchestration: isolated uv environments, JSON protocol, typed results.

The core engine invokes workers with structured argv (no shell mode, ever) via
``uv run --project <worker_env> python worker.py --request ... --response ...``.
uv creates/syncs the isolated environment on first use — that is the permitted
bootstrap network traffic (packages/models only; task media is NEVER uploaded).

With bootstrap disabled, a missing worker environment records ASR_UNAVAILABLE
and the evidence pipeline continues.
"""

from __future__ import annotations

import itertools
import json
import logging
import re
import shutil
import time
import unicodedata
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
    BestWord,
    LanguageEvidence,
    LanguageReviewStatus,
    WordTimingStatus,
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


def _decimal_str(value: Any) -> str | None:
    """Normalize an ASR-reported float time via Decimal(str(...)). Estimates only.

    Returns ``None`` for a missing/unparseable time. Missing timing is MISSING
    EVIDENCE — it is NEVER coerced to ``"0"`` (that would fabricate a 0 s anchor,
    §8). Callers decide the honest fallback (faster-whisper timing, review).
    """
    if value is None:
        return None
    text = str(value).strip()
    if text == "" or text.lower() == "none":
        return None
    try:
        return str(Decimal(text))
    except (InvalidOperation, ValueError, TypeError):
        return None


def decimal_to_fraction(text: str) -> Fraction:
    return Fraction(Decimal(text))


def _maybe_fraction(text: str | None, offset: Fraction) -> Fraction | None:
    return offset + decimal_to_fraction(text) if text is not None else None


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
    raw_words: list[dict[str, Any]],
    offset: Fraction,
    timing_source: str,
    segment_id: int,
) -> list[ASRWord]:
    """Build ASRWords WITHOUT dropping any. A word with missing start/end is kept
    with ``None`` timing (missing evidence) so no source word silently disappears
    and no fake 0 s timestamp is invented (§5/§8)."""
    words: list[ASRWord] = []
    for index, word in enumerate(raw_words):
        start_text = _decimal_str(word.get("start"))
        end_text = _decimal_str(word.get("end"))
        words.append(
            ASRWord(
                word_index=index,
                segment_id=segment_id,
                source_word_index=index,
                text=str(word.get("text", "")).strip(),
                asr_start_seconds=start_text,
                asr_end_seconds=end_text,
                start_annotation_time=_maybe_fraction(start_text, offset),
                end_annotation_time=_maybe_fraction(end_text, offset),
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
        segment_id = int(raw.get("id", len(segments)))
        start_text = _decimal_str(raw.get("start"))
        end_text = _decimal_str(raw.get("end"))
        segments.append(
            ASRSegment(
                segment_id=segment_id,
                text=str(raw.get("text", "")).strip(),
                asr_start_seconds=start_text,
                asr_end_seconds=end_text,
                start_annotation_time=_maybe_fraction(start_text, annotation_audio_offset),
                end_annotation_time=_maybe_fraction(end_text, annotation_audio_offset),
                avg_logprob=raw.get("avg_logprob"),
                no_speech_prob=raw.get("no_speech_prob"),
                words=_words_from_raw(
                    raw.get("words", []), annotation_audio_offset,
                    "faster_whisper", segment_id,
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
    for position, raw in enumerate(response.get("segments", [])):
        # A missing WhisperX segment timestamp falls back to the preserved
        # faster-whisper segment timing (§8) — never to 0 s.
        source = source_segments[position] if position < len(source_segments) else None
        segment_id = int(raw.get("id", position))
        start_text = _decimal_str(raw.get("start"))
        end_text = _decimal_str(raw.get("end"))
        start_frac = _maybe_fraction(start_text, annotation_audio_offset)
        end_frac = _maybe_fraction(end_text, annotation_audio_offset)
        if start_frac is None and source is not None:
            start_text, start_frac = source.asr_start_seconds, source.start_annotation_time
        if end_frac is None and source is not None:
            end_text, end_frac = source.asr_end_seconds, source.end_annotation_time
        segments.append(
            ASRSegment(
                segment_id=segment_id,
                text=str(raw.get("text", "")).strip(),
                asr_start_seconds=start_text,
                asr_end_seconds=end_text,
                start_annotation_time=start_frac,
                end_annotation_time=end_frac,
                words=_words_from_raw(
                    raw.get("words", []), annotation_audio_offset, "whisperx", segment_id
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


#: Unicode major categories kept in a match key: Letters, Numbers, and Marks
#: (combining marks are part of the word in Arabic/Devanagari/etc.). Punctuation
#: (P), separators/whitespace (Z), symbols (S) and control/format (C) are dropped.
_MATCH_KEY_KEEP = ("L", "N", "M")


def _match_key(text: str) -> str:
    """Normalized token identity for FW↔WhisperX word matching, Unicode-aware so
    ALL scripts (Latin, Cyrillic, Arabic, Devanagari, CJK, …) match — an
    ASCII-only key would collapse non-Latin words to '' and force faster-whisper
    fallback for every foreign-language word. NFKC + casefold, then keep only
    letters/numbers/marks. Deliberately conservative: when the key cannot safely
    match, faster-whisper timing is kept rather than guessing a mapping (§6)."""
    normalized = unicodedata.normalize("NFKC", text).casefold()
    return "".join(
        ch for ch in normalized if unicodedata.category(ch)[0] in _MATCH_KEY_KEEP
    )


@dataclass(frozen=True)
class BestTranscript:
    """Reconciliation product: exactly one best word per faster-whisper source
    word, plus alignment coverage bookkeeping."""

    best_words: list[BestWord]
    status: AlignmentStatus
    coverage: float | None
    partial_alignment: bool
    source_word_count: int
    aligned_word_count: int


def _fw_best_word(word: ASRWord, segment: ASRSegment, status: WordTimingStatus) -> BestWord:
    """A faster-whisper-timed best word. Uses the FW word timing, falling back to
    the FW segment timing only if the word itself has none (never fabricated)."""
    start = word.start_annotation_time
    end = word.end_annotation_time
    start_str = word.asr_start_seconds
    end_str = word.asr_end_seconds
    resolved = status
    if start is None or end is None or start_str is None or end_str is None:
        # Pathological: FW word without timing. Use the real segment timing and
        # force review rather than inventing 0 s.
        start = segment.start_annotation_time
        end = segment.end_annotation_time
        start_str = segment.asr_start_seconds
        end_str = segment.asr_end_seconds
        resolved = WordTimingStatus.ALIGNMENT_UNRESOLVED
    if start is None or end is None or start_str is None or end_str is None:
        raise ValueError(
            f"faster-whisper word '{word.text}' has no usable timing evidence"
        )
    return BestWord(
        segment_id=word.segment_id,
        source_word_index=word.source_word_index,
        text=word.text,
        start_annotation_time=start,
        end_annotation_time=end,
        asr_start_seconds=start_str,
        asr_end_seconds=end_str,
        probability=word.probability,
        timing_source="faster_whisper",
        timing_status=resolved,
    )


def _reconcile_segment(
    fw_words: list[ASRWord], wx_words: list[ASRWord], segment: ASRSegment
) -> list[BestWord]:
    """Map WhisperX timing back onto each FW source word by identity. WhisperX
    only ever improves timing: every FW source word yields exactly one best word,
    and none is dropped, invented, or re-worded (§5/§6)."""
    best: list[BestWord] = []
    j = 0
    for fw in fw_words:
        key = _match_key(fw.text)
        matched: ASRWord | None = None
        k = j
        while k < len(wx_words):
            if _match_key(wx_words[k].text) == key and key != "":
                matched = wx_words[k]
                j = k + 1
                break
            k += 1
        if (
            matched is not None
            and matched.start_annotation_time is not None
            and matched.end_annotation_time is not None
            and matched.end_annotation_time >= matched.start_annotation_time
            and matched.asr_start_seconds is not None
            and matched.asr_end_seconds is not None
        ):
            best.append(
                BestWord(
                    segment_id=fw.segment_id,
                    source_word_index=fw.source_word_index,
                    text=fw.text,  # wording ALWAYS faster-whisper
                    start_annotation_time=matched.start_annotation_time,
                    end_annotation_time=matched.end_annotation_time,
                    asr_start_seconds=matched.asr_start_seconds,
                    asr_end_seconds=matched.asr_end_seconds,
                    probability=fw.probability,
                    timing_source="whisperx",
                    timing_status=WordTimingStatus.WHISPERX_ALIGNED,
                )
            )
        else:
            best.append(
                _fw_best_word(fw, segment, WordTimingStatus.FASTER_WHISPER_FALLBACK)
            )

    # Ordering safety policy (§7): if per-word WhisperX timings produce a
    # non-monotonic segment, fall the WHOLE segment back to faster-whisper timing
    # rather than emit unsafe ordering. Accuracy over aligned-word count.
    starts = [w.start_annotation_time for w in best]
    if any(b < a for a, b in itertools.pairwise(starts)):
        return [
            _fw_best_word(fw, segment, WordTimingStatus.FASTER_WHISPER_FALLBACK)
            for fw in fw_words
        ]
    return best


def reconcile_best_transcript(
    asr_result: ASRResult, alignment: AlignmentResult | None
) -> BestTranscript:
    """Produce the best transcript word sequence: faster-whisper wording with the
    best available per-word timing. Guarantees the best word sequence exactly
    matches the faster-whisper source sequence (P3-ASR-006)."""
    fw_segments = asr_result.segments
    source_word_count = sum(len(s.words) for s in fw_segments)

    aligned_ok = alignment is not None and alignment.status == AlignmentStatus.ALIGNED
    best_words: list[BestWord] = []
    if not aligned_ok:
        for segment in fw_segments:
            best_words.extend(
                _fw_best_word(w, segment, WordTimingStatus.FASTER_WHISPER_FALLBACK)
                for w in segment.words
            )
        status = alignment.status if alignment is not None else AlignmentStatus.NOT_ATTEMPTED
        return BestTranscript(
            best_words=best_words,
            status=status,
            coverage=0.0 if source_word_count else None,
            partial_alignment=bool(source_word_count),
            source_word_count=source_word_count,
            aligned_word_count=0,
        )

    assert alignment is not None
    aligned_segments = alignment.segments
    for position, segment in enumerate(fw_segments):
        wx_words = (
            aligned_segments[position].words if position < len(aligned_segments) else []
        )
        best_words.extend(_reconcile_segment(segment.words, wx_words, segment))

    aligned_word_count = sum(
        1 for w in best_words if w.timing_status == WordTimingStatus.WHISPERX_ALIGNED
    )
    coverage = (aligned_word_count / source_word_count) if source_word_count else None
    fully_aligned = aligned_word_count == source_word_count and source_word_count > 0
    return BestTranscript(
        best_words=best_words,
        status=AlignmentStatus.ALIGNED if fully_aligned else AlignmentStatus.PARTIAL,
        coverage=coverage,
        partial_alignment=not fully_aligned,
        source_word_count=source_word_count,
        aligned_word_count=aligned_word_count,
    )


def best_words_to_segments(best_words: list[BestWord]) -> list[ASRSegment]:
    """Group best words back into segments (for segments_best.csv / transcript)."""
    by_segment: dict[int, list[BestWord]] = {}
    order: list[int] = []
    for word in best_words:
        if word.segment_id not in by_segment:
            by_segment[word.segment_id] = []
            order.append(word.segment_id)
        by_segment[word.segment_id].append(word)
    segments: list[ASRSegment] = []
    for segment_id in order:
        group = by_segment[segment_id]
        segments.append(
            ASRSegment(
                segment_id=segment_id,
                text=" ".join(w.text for w in group).strip(),
                asr_start_seconds=group[0].asr_start_seconds,
                asr_end_seconds=group[-1].asr_end_seconds,
                start_annotation_time=group[0].start_annotation_time,
                end_annotation_time=group[-1].end_annotation_time,
                words=[
                    ASRWord(
                        word_index=i,
                        segment_id=w.segment_id,
                        source_word_index=w.source_word_index,
                        text=w.text,
                        asr_start_seconds=w.asr_start_seconds,
                        asr_end_seconds=w.asr_end_seconds,
                        start_annotation_time=w.start_annotation_time,
                        end_annotation_time=w.end_annotation_time,
                        probability=w.probability,
                        timing_source=w.timing_source,
                    )
                    for i, w in enumerate(group)
                ],
            )
        )
    return segments


def _model_in_cache(model: str) -> bool:
    """Best-effort: is the faster-whisper model present in the local HF cache?"""
    cache_root = Path.home() / ".cache" / "huggingface" / "hub"
    marker = f"models--Systran--faster-whisper-{model}"
    turbo_marker = f"models--mobiuslabsgmbh--faster-whisper-{model}"
    return any((cache_root / name).exists() for name in (marker, turbo_marker))


def model_cache_state(model: str) -> str:
    """Snapshot BEFORE the worker runs: 'cached' | 'not_cached'."""
    return "cached" if _model_in_cache(model) else "not_cached"


def resolve_cache_state(model: str, state_before: str, worker_ran: bool) -> str:
    """Reconcile the pre-run snapshot with the post-run cache (§21).

    A model that was absent before and present after was downloaded during this
    run — recording it as ``not_cached`` would be wrong. Returns one of
    ``cached`` | ``downloaded_this_run`` | ``unavailable``.
    """
    present_after = _model_in_cache(model)
    if state_before == "cached" and present_after:
        return "cached"
    if present_after:
        return "downloaded_this_run"
    # Not in cache after a worker run that should have fetched it → unavailable.
    return "unavailable" if worker_ran else "not_cached"


def measure_runtime(start: float) -> float:
    return round(time.perf_counter() - start, 3)
