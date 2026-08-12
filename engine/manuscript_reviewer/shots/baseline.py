"""Local robust baselines: judge each pair against its temporal neighborhood.

Fast-motion footage naturally has large adjacent differences, so no single
global threshold exists. Each pair is compared against the median/MAD of pair
metrics in a time-based window around it (excluding the pair itself), yielding
robust z-like scores. Time-based windows use real PTS times, so VFR media gets
correct context; when PTS is missing the window falls back to pair counts.
"""

from __future__ import annotations

from fractions import Fraction

from ..models.shot_truth import LocalBaseline, PairMetrics

#: Half-window duration around a pair (seconds of real media time).
WINDOW_SECONDS = Fraction(1, 2)
#: Fallback half-window in pairs when PTS is unavailable.
FALLBACK_WINDOW_PAIRS = 12
#: Guard so a perfectly static neighborhood (MAD=0) cannot yield infinite z.
MAD_FLOOR_DIFF = 0.35
MAD_FLOOR_HIST = 0.004
_MAD_TO_SIGMA = 1.4826


def _median(values: list[float]) -> float:
    ordered = sorted(values)
    n = len(ordered)
    if n == 0:
        return 0.0
    mid = n // 2
    if n % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2.0


def _window_indexes(pairs: list[PairMetrics], center: int) -> list[int]:
    """Indexes of neighboring pairs within WINDOW_SECONDS of the center pair."""
    center_time = pairs[center].right_pts_time_seconds
    if center_time is None:
        low = max(0, center - FALLBACK_WINDOW_PAIRS)
        high = min(len(pairs), center + FALLBACK_WINDOW_PAIRS + 1)
        return [i for i in range(low, high) if i != center]
    selected: list[int] = []
    for i in range(center - 1, -1, -1):
        t = pairs[i].right_pts_time_seconds
        if t is None or center_time - t > WINDOW_SECONDS:
            break
        selected.append(i)
    for i in range(center + 1, len(pairs)):
        t = pairs[i].right_pts_time_seconds
        if t is None or t - center_time > WINDOW_SECONDS:
            break
        selected.append(i)
    return selected


def compute_local_baselines(pairs: list[PairMetrics]) -> list[LocalBaseline]:
    baselines: list[LocalBaseline] = []
    for center in range(len(pairs)):
        neighbors = _window_indexes(pairs, center)
        diffs = [pairs[i].mean_abs_diff for i in neighbors]
        hists = [pairs[i].hist_distance for i in neighbors]
        motions = [pairs[i].flow_mean_mag for i in neighbors]

        diff_median = _median(diffs)
        hist_median = _median(hists)
        diff_mad = max(_median([abs(d - diff_median) for d in diffs]), MAD_FLOOR_DIFF)
        hist_mad = max(_median([abs(h - hist_median) for h in hists]), MAD_FLOOR_HIST)

        pair = pairs[center]
        baselines.append(
            LocalBaseline(
                window_pairs=len(neighbors),
                diff_median=round(diff_median, 4),
                diff_mad=round(diff_mad, 4),
                diff_z=round(
                    (pair.mean_abs_diff - diff_median) / (diff_mad * _MAD_TO_SIGMA), 4
                ),
                hist_median=round(hist_median, 6),
                hist_mad=round(hist_mad, 6),
                hist_z=round(
                    (pair.hist_distance - hist_median) / (hist_mad * _MAD_TO_SIGMA), 4
                ),
                neighbor_motion=round(_median(motions), 4),
            )
        )
    return baselines
