"""High-risk visual evidence bundles (Z).

For each CRITICAL/HIGH review item with a resolved frame range, render
before/start/middle/end/after frames plus a context strip and an evidence.json.
Every image records its exact frame index, annotation time, and role. Evidence
shows the contradiction context, not only supporting frames, and no misleading
semantic label is burned into the source images.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from fractions import Fraction
from pathlib import Path

import cv2
import numpy as np

from ..models.review_intelligence import ReviewPriority, ReviewQueueItem
from ..visual.decode import FrameCache

_ROLES = ("before", "start", "middle", "end", "after")


def extract_evidence_bundles(
    queue_items: list[ReviewQueueItem],
    cache: FrameCache,
    frame_count: int,
    frame_time: Callable[[int], Fraction | None],
    out_dir: Path,
) -> list[Path]:
    written: list[Path] = []
    high_risk = [
        i for i in queue_items
        if i.priority in (ReviewPriority.CRITICAL, ReviewPriority.HIGH)
        and _bundle_range(i) is not None
    ]
    for index, item in enumerate(high_risk, start=1):
        bundle = out_dir / f"review_{index:04d}"
        written.extend(_write_bundle(item, cache, frame_count, frame_time, bundle))
        item.evidence_bundle_dir = f"visual_evidence/review_{index:04d}"
    return written


def _bundle_range(item: ReviewQueueItem) -> tuple[int, int] | None:
    """Prefer the item's own frame range; else fall back to the widest frame range
    across its evidence refs (item 19 — precise ranges come from evidence, not a
    single anchor). Returns None when no frame anchor exists at all."""
    if item.start_frame is not None:
        end = item.end_frame if item.end_frame is not None else item.start_frame
        return item.start_frame, end
    frames = [
        (r.start_frame, r.end_frame if r.end_frame is not None else r.start_frame)
        for r in (*item.supporting_evidence_refs, *item.contradicting_evidence_refs)
        if r.start_frame is not None
    ]
    if not frames:
        return None
    return min(f[0] for f in frames), max(f[1] for f in frames)


def _role_frames(item: ReviewQueueItem, frame_count: int) -> dict[str, int]:
    span = _bundle_range(item)
    raw_start, raw_end = span if span is not None else (0, 0)
    start = max(0, min(frame_count - 1, raw_start))
    end = max(start, min(frame_count - 1, raw_end))
    middle = (start + end) // 2
    roles = {
        "before": max(0, start - 1),
        "start": start,
        "middle": middle,
        "end": end,
        "after": min(frame_count - 1, end + 1),
    }
    return roles


def _write_bundle(
    item: ReviewQueueItem,
    cache: FrameCache,
    frame_count: int,
    frame_time: Callable[[int], Fraction | None],
    bundle: Path,
) -> list[Path]:
    bundle.mkdir(parents=True, exist_ok=True)
    roles = _role_frames(item, frame_count)
    images: list[dict[str, object]] = []
    strip_parts: list[np.ndarray] = []
    written: list[Path] = []
    for role, frame_index in roles.items():
        rgb = cache.color_frame(frame_index)
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        path = bundle / f"{role}.png"
        cv2.imwrite(str(path), bgr)
        written.append(path)
        strip_parts.append(bgr)
        t = frame_time(frame_index)
        images.append({
            "role": role,
            "frame_index": frame_index,
            "annotation_time": str(t) if t is not None else None,
        })
    if strip_parts:
        strip = np.hstack([cv2.resize(p, (160, 90)) for p in strip_parts])
        strip_path = bundle / "strip.png"
        cv2.imwrite(str(strip_path), strip)
        written.append(strip_path)
    evidence = {
        "item_id": item.item_id,
        "priority": item.priority.value,
        "title": item.title,
        "reason": item.reason,
        "recommended_action": item.recommended_action.value,
        "shows_contradiction_context": True,
        "images": images,
        # The frame ranges the bundle was derived from (item 19).
        "evidence_ref_frames": [
            {"evidence_id": r.evidence_id, "start_frame": r.start_frame,
             "end_frame": r.end_frame, "notes": r.notes}
            for r in (*item.supporting_evidence_refs, *item.contradicting_evidence_refs)
            if r.start_frame is not None
        ],
    }
    ev_path = bundle / "evidence.json"
    with ev_path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(evidence, handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.write("\n")
    written.append(ev_path)
    return written
