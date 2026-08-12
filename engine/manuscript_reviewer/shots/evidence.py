"""Candidate evidence bundles: labeled frame pairs and strips from EXACT frames.

Evidence images are decoded by frame identity (``select='eq(n,...)'`` under
``-fps_mode passthrough``), never by approximate ``-ss`` timestamp seeking, so
image N is guaranteed to be ledger frame N. Every image carries burned-in
labels (frame index, exact PTS time, Manuscript display time) and the same
identities are stored in ``evidence.json``.
"""

from __future__ import annotations

import json
import logging
import tempfile
from fractions import Fraction
from pathlib import Path

import cv2
import numpy as np
import numpy.typing as npt

from ..media.ffmpeg_tools import ToolExecutionError, find_tool, run_tool
from ..media.timestamps import format_manuscript_display, seconds_to_decimal
from ..models.frame import FrameLedger
from ..models.shot_truth import BoundaryCandidate

logger = logging.getLogger(__name__)

PAIR_FRAME_WIDTH = 480
STRIP_FRAME_WIDTH = 160
CONTEXT_SECONDS = Fraction(1, 2)
CONTEXT_MAX_FRAMES_PER_SIDE = 10
SHORT_STRIP_RADIUS = 3

BGRImage = npt.NDArray[np.uint8]


def extract_frames_by_index(
    video_path: Path, indexes: list[int], width: int, stream_index: int = 0
) -> dict[int, BGRImage]:
    """Decode exactly the requested ledger frames, scaled to ``width``."""
    if not indexes:
        return {}
    unique = sorted(set(indexes))
    ffmpeg = find_tool("ffmpeg")
    images: dict[int, BGRImage] = {}
    # Chunk the select expression to keep command lines manageable.
    for start in range(0, len(unique), 120):
        chunk = unique[start : start + 120]
        expr = "+".join(f"eq(n\\,{i})" for i in chunk)
        with tempfile.TemporaryDirectory(prefix="mr_evidence_") as tmp:
            out_pattern = Path(tmp) / "f%06d.png"
            result = run_tool(
                ffmpeg,
                [
                    "-v", "error",
                    "-i", str(video_path),
                    "-map", f"0:v:{stream_index}",
                    "-fps_mode", "passthrough",
                    "-vf", f"select='{expr}',scale={width}:-2",
                    "-start_number", "0",
                    str(out_pattern),
                ],
                timeout=1800.0,
            )
            produced = sorted(Path(tmp).glob("f??????.png"))
            if len(produced) != len(chunk):
                raise ToolExecutionError(
                    result.command, 0,
                    f"select produced {len(produced)} frames, expected {len(chunk)}",
                )
            for frame_index, path in zip(chunk, produced, strict=True):
                image = cv2.imread(str(path), cv2.IMREAD_COLOR)
                if image is None:
                    raise ToolExecutionError(result.command, 0, f"unreadable frame {path}")
                images[frame_index] = image.astype(np.uint8)
    return images


def _label(image: BGRImage, lines: list[str]) -> BGRImage:
    """Burn compact label lines onto the bottom of an image."""
    labeled = image.copy()
    bar_height = 14 * len(lines) + 6
    h, w = labeled.shape[:2]
    cv2.rectangle(labeled, (0, h - bar_height), (w, h), (0, 0, 0), -1)
    for i, text in enumerate(lines):
        cv2.putText(
            labeled,
            text,
            (4, h - bar_height + 12 + 14 * i),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    return labeled


def _frame_identity(ledger: FrameLedger, index: int) -> dict[str, object]:
    record = ledger.frames[index]
    time_value = record.pts_time_seconds
    return {
        "pts": record.pts,
        "pts_time": str(seconds_to_decimal(time_value)) if time_value is not None else None,
    }


def _frame_labels(ledger: FrameLedger, index: int) -> list[str]:
    record = ledger.frames[index]
    if record.pts_time_seconds is not None:
        exact = str(seconds_to_decimal(record.pts_time_seconds))
        display = format_manuscript_display(record.pts_time_seconds)
    else:
        exact, display = "?", "?"
    return [f"F{index:06d} pts={record.pts}", f"t={exact}s  M2={display}"]


def _hstack(images: list[BGRImage]) -> BGRImage:
    height = min(img.shape[0] for img in images)
    resized = [
        cv2.resize(img, (int(img.shape[1] * height / img.shape[0]), height))
        for img in images
    ]
    return np.hstack(resized)


def _context_indexes(ledger: FrameLedger, candidate: BoundaryCandidate) -> list[int]:
    """Time-based context window resolved to actual ledger frames (VFR-safe)."""
    n = ledger.frame_count
    center_time = ledger.frames[candidate.right_frame_index].pts_time_seconds
    if center_time is None:
        low = max(0, candidate.left_frame_index - CONTEXT_MAX_FRAMES_PER_SIDE)
        high = min(n, candidate.right_frame_index + CONTEXT_MAX_FRAMES_PER_SIDE + 1)
        return list(range(low, high))
    selected = [candidate.left_frame_index, candidate.right_frame_index]
    for count, i in enumerate(range(candidate.left_frame_index - 1, -1, -1)):
        t = ledger.frames[i].pts_time_seconds
        if t is None or center_time - t > CONTEXT_SECONDS or count >= CONTEXT_MAX_FRAMES_PER_SIDE:
            break
        selected.append(i)
    for count, i in enumerate(range(candidate.right_frame_index + 1, n)):
        t = ledger.frames[i].pts_time_seconds
        if t is None or t - center_time > CONTEXT_SECONDS or count >= CONTEXT_MAX_FRAMES_PER_SIDE:
            break
        selected.append(i)
    return sorted(set(selected))


def render_candidate_evidence(
    video_path: Path,
    ledger: FrameLedger,
    candidates: list[BoundaryCandidate],
    evidence_root: Path,
) -> dict[str, list[str]]:
    """Render evidence bundles for the given candidates.

    Returns {candidate_id: [relative artifact paths]}. One decode pass per
    image size fetches every needed frame across all candidates.
    """
    needed_pair: set[int] = set()
    needed_strip: set[int] = set()
    per_candidate_short: dict[str, list[int]] = {}
    per_candidate_context: dict[str, list[int]] = {}
    n = ledger.frame_count

    for candidate in candidates:
        needed_pair.update([candidate.left_frame_index, candidate.right_frame_index])
        short = [
            i
            for i in range(
                candidate.left_frame_index - SHORT_STRIP_RADIUS,
                candidate.right_frame_index + SHORT_STRIP_RADIUS + 1,
            )
            if 0 <= i < n
        ]
        context = _context_indexes(ledger, candidate)
        per_candidate_short[candidate.candidate_id] = short
        per_candidate_context[candidate.candidate_id] = context
        needed_strip.update(short)
        needed_strip.update(context)

    pair_images = extract_frames_by_index(video_path, sorted(needed_pair), PAIR_FRAME_WIDTH)
    strip_images = extract_frames_by_index(video_path, sorted(needed_strip), STRIP_FRAME_WIDTH)

    refs: dict[str, list[str]] = {}
    for candidate in candidates:
        bundle_dir = evidence_root / f"candidate_{candidate.candidate_id.split('_')[-1]}"
        bundle_dir.mkdir(parents=True, exist_ok=True)
        artifacts: list[str] = []

        left = _label(
            pair_images[candidate.left_frame_index],
            ["LEFT (outgoing)", *_frame_labels(ledger, candidate.left_frame_index)],
        )
        right = _label(
            pair_images[candidate.right_frame_index],
            ["RIGHT (incoming)", *_frame_labels(ledger, candidate.right_frame_index)],
        )
        pair_path = bundle_dir / "pair.png"
        cv2.imwrite(str(pair_path), _hstack([left, right]))
        artifacts.append(pair_path.name)

        for name, indexes in (
            ("strip_short.png", per_candidate_short[candidate.candidate_id]),
            ("strip_context.png", per_candidate_context[candidate.candidate_id]),
        ):
            labeled = [
                _label(strip_images[i], _frame_labels(ledger, i)) for i in indexes
            ]
            strip_path = bundle_dir / name
            cv2.imwrite(str(strip_path), _hstack(labeled))
            artifacts.append(strip_path.name)

        evidence_payload = {
            "candidate": candidate.model_dump(mode="json"),
            "images": {
                "pair.png": [candidate.left_frame_index, candidate.right_frame_index],
                "strip_short.png": per_candidate_short[candidate.candidate_id],
                "strip_context.png": per_candidate_context[candidate.candidate_id],
            },
            "frame_identities": {
                str(i): _frame_identity(ledger, i)
                for i in sorted(
                    set(per_candidate_short[candidate.candidate_id])
                    | set(per_candidate_context[candidate.candidate_id])
                )
            },
        }
        json_path = bundle_dir / "evidence.json"
        json_path.write_text(
            json.dumps(evidence_payload, indent=2, sort_keys=True), encoding="utf-8"
        )
        artifacts.append(json_path.name)
        refs[candidate.candidate_id] = [
            (bundle_dir.relative_to(evidence_root.parent) / a).as_posix() for a in artifacts
        ]
    return refs
