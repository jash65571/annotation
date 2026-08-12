"""Deterministic per-shot cadence metrics from the shared gray grid."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..shots.decode import GrayFrames

#: Adjacent mean-abs-diff below this (0..1) counts as a near-duplicate frame.
_NEAR_DUP = 0.005
#: Adjacent mean-abs-diff above this indicates real content motion in the shot.
_REAL_MOTION = 0.02


@dataclass(frozen=True)
class ShotCadence:
    pair_count: int
    duplicate_ratio: float
    motion_present: bool
    first_half_dup: float
    second_half_dup: float
    diffs: list[float]


def shot_cadence(gray: GrayFrames, start: int, end: int) -> ShotCadence:
    diffs: list[float] = []
    for i in range(start, end):
        a = gray[i].astype(np.float32)
        b = gray[i + 1].astype(np.float32)
        diffs.append(float(np.abs(a - b).mean()) / 255.0)
    if not diffs:
        return ShotCadence(0, 0.0, False, 0.0, 0.0, [])
    dup = sum(1 for d in diffs if d < _NEAR_DUP)
    motion = any(d > _REAL_MOTION for d in diffs)
    mid = len(diffs) // 2
    first = diffs[:mid] or diffs
    second = diffs[mid:] or diffs
    return ShotCadence(
        pair_count=len(diffs),
        duplicate_ratio=round(dup / len(diffs), 4),
        motion_present=motion,
        first_half_dup=round(sum(1 for d in first if d < _NEAR_DUP) / len(first), 4),
        second_half_dup=round(sum(1 for d in second if d < _NEAR_DUP) / len(second), 4),
        diffs=diffs,
    )
