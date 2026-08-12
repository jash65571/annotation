"""Writes the audit run folder: media.json, frames.csv/jsonl, manifest.json, qc.json.

All JSON is emitted deterministically (sorted keys, exact rationals as
"num/den" strings) so identical inputs produce byte-identical artifacts apart
from the manifest's timing fields.
"""

from __future__ import annotations

import csv
import hashlib
import json
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ..media.timestamps import seconds_to_decimal
from ..models.frame import FrameLedger
from ..models.media import MediaInfo
from ..models.run import ArtifactEntry
from ..models.validation import QCReport


class ArtifactWriteError(RuntimeError):
    """An artifact could not be written or verified."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def new_run_id() -> str:
    """Sortable, collision-safe run id: UTC timestamp + short random suffix."""
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}-{uuid.uuid4().hex[:8]}"


def create_run_dir(artifacts_root: Path, video_path: Path, run_id: str) -> Path:
    run_dir = artifacts_root / video_path.stem / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def _write_json(path: Path, payload: Any) -> None:
    try:
        with path.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, ensure_ascii=False)
            handle.write("\n")
    except OSError as exc:
        raise ArtifactWriteError(f"Failed to write {path}: {exc}") from exc


def write_media_json(run_dir: Path, media: MediaInfo, raw_probe: dict[str, Any]) -> Path:
    path = run_dir / "media.json"
    _write_json(path, {"media": media.model_dump(mode="json"), "raw_ffprobe": raw_probe})
    return path


FRAME_CSV_COLUMNS = [
    "frame_index",
    "pts",
    "pts_time",
    "dts",
    "duration",
    "duration_time",
    "key_frame",
    "pict_type",
    "width",
    "height",
]


def write_frames_csv(run_dir: Path, ledger: FrameLedger) -> Path:
    """One row per enumerated frame. pts_time/duration_time are exact values
    rendered at microsecond precision (matching ffprobe's own rendering)."""
    path = run_dir / "frames.csv"
    try:
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(FRAME_CSV_COLUMNS)
            for frame in ledger.frames:
                writer.writerow(
                    [
                        frame.frame_index,
                        frame.pts if frame.pts is not None else "",
                        (
                            str(seconds_to_decimal(frame.pts_time_seconds))
                            if frame.pts_time_seconds is not None
                            else ""
                        ),
                        frame.dts if frame.dts is not None else "",
                        frame.duration if frame.duration is not None else "",
                        (
                            str(seconds_to_decimal(frame.duration_seconds))
                            if frame.duration_seconds is not None
                            else ""
                        ),
                        int(frame.key_frame),
                        frame.pict_type or "",
                        frame.width if frame.width is not None else "",
                        frame.height if frame.height is not None else "",
                    ]
                )
    except OSError as exc:
        raise ArtifactWriteError(f"Failed to write {path}: {exc}") from exc
    return path


def write_frames_jsonl(run_dir: Path, ledger: FrameLedger) -> Path:
    """Lossless ledger: every FrameRecord as JSON, exact rationals preserved."""
    path = run_dir / "frames.jsonl"
    header = {
        "record_type": "ledger_header",
        "stream_index": ledger.stream_index,
        "time_base": f"{ledger.time_base.numerator}/{ledger.time_base.denominator}",
        "frame_count": ledger.frame_count,
    }
    try:
        with path.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(header, sort_keys=True) + "\n")
            for frame in ledger.frames:
                handle.write(
                    json.dumps(frame.model_dump(mode="json"), sort_keys=True) + "\n"
                )
    except OSError as exc:
        raise ArtifactWriteError(f"Failed to write {path}: {exc}") from exc
    return path


def write_qc_json(run_dir: Path, report: QCReport) -> Path:
    path = run_dir / "qc.json"
    _write_json(path, report.model_dump(mode="json"))
    return path


def collect_artifact_entries(run_dir: Path, paths: list[Path]) -> list[ArtifactEntry]:
    entries = []
    for path in paths:
        entries.append(
            ArtifactEntry(
                path=path.relative_to(run_dir).as_posix(),
                sha256=sha256_file(path),
                size_bytes=path.stat().st_size,
            )
        )
    return entries


def write_manifest_json(run_dir: Path, manifest_payload: dict[str, Any]) -> Path:
    path = run_dir / "manifest.json"
    _write_json(path, manifest_payload)
    return path
