"""MetricEngine: deterministic per-frame stats and adjacent-pair metrics.

Metric choices, what they catch, what fools them, cost and ranges are
documented in docs/06-shot-truth-engine.md. All metrics are computed on the
160x90 grayscale metric grid; optical flow runs on a further-halved 80x45 grid.
"""

from __future__ import annotations

import logging

import cv2
import numpy as np
import numpy.typing as npt

from ..models.frame import FrameLedger
from ..models.shot_truth import FrameStats, PairMetrics
from .decode import GrayFrames

logger = logging.getLogger(__name__)

#: Near-black / near-white flat-frame thresholds (0-255 luma).
NEAR_BLACK_MEAN = 24.0
NEAR_WHITE_MEAN = 231.0
FLAT_STD = 16.0

_HIST_BINS = 64


def _phash64(gray: npt.NDArray[np.uint8]) -> np.uint64:
    """64-bit DCT perceptual hash (structure-aware, illumination-tolerant)."""
    small = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA)
    dct = cv2.dct(small.astype(np.float32))
    low = dct[:8, :8].flatten()
    low_ac = low[1:]  # drop the DC coefficient
    median = float(np.median(low_ac))
    bits = np.uint64(0)
    for i, value in enumerate(low_ac):
        if value > median:
            bits |= np.uint64(1) << np.uint64(i)
    return bits


def _hamming64(a: np.uint64, b: np.uint64) -> int:
    return int(bin(int(a) ^ int(b)).count("1"))


def _histogram(gray: npt.NDArray[np.uint8]) -> npt.NDArray[np.float32]:
    hist = cv2.calcHist([gray], [0], None, [_HIST_BINS], [0, 256])
    total = float(hist.sum())
    result: npt.NDArray[np.float32] = (hist / total).astype(np.float32)
    return result


def _bhattacharyya(h1: npt.NDArray[np.float32], h2: npt.NDArray[np.float32]) -> float:
    return float(cv2.compareHist(h1, h2, cv2.HISTCMP_BHATTACHARYYA))


def compute_frame_stats(frames: GrayFrames) -> list[FrameStats]:
    """Per-frame luma statistics, edge density, sharpness, flat-frame flags."""
    stats: list[FrameStats] = []
    for index in range(frames.shape[0]):
        gray = frames[index]
        mean = float(gray.mean())
        std = float(gray.std())
        edges = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3) ** 2
        edges += cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3) ** 2
        edge_density = float(np.sqrt(edges).mean()) / 255.0
        sharpness = float(cv2.Laplacian(gray, cv2.CV_32F).var())
        stats.append(
            FrameStats(
                frame_index=index,
                luma_mean=round(mean, 4),
                luma_std=round(std, 4),
                edge_density=round(edge_density, 6),
                sharpness=round(sharpness, 4),
                near_black=mean < NEAR_BLACK_MEAN and std < FLAT_STD,
                near_white=mean > NEAR_WHITE_MEAN and std < FLAT_STD,
            )
        )
    return stats


def _flow_summary(
    left_small: npt.NDArray[np.uint8], right_small: npt.NDArray[np.uint8]
) -> tuple[float, float]:
    """Farneback optical flow on the 80x45 grid → (mean magnitude, coherence).

    Coherence = |mean flow vector| / mean |flow|: ~1.0 for coherent global
    camera motion (pan/tilt), low for incoherent change (cuts, noise).
    """
    flow = cv2.calcOpticalFlowFarneback(  # type: ignore[call-overload]
        left_small,
        right_small,
        None,
        pyr_scale=0.5,
        levels=3,
        winsize=15,
        iterations=2,
        poly_n=5,
        poly_sigma=1.1,
        flags=0,
    )
    magnitudes = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
    mean_mag = float(magnitudes.mean())
    if mean_mag < 1e-6:
        return 0.0, 0.0
    mean_vec = flow.reshape(-1, 2).mean(axis=0)
    coherence = float(np.sqrt(mean_vec[0] ** 2 + mean_vec[1] ** 2) / mean_mag)
    return mean_mag, min(coherence, 1.0)


def compute_pair_metrics(frames: GrayFrames, ledger: FrameLedger) -> list[PairMetrics]:
    """One metric record for every adjacent frame pair (N frames → N-1 records)."""
    n = frames.shape[0]
    if n != ledger.frame_count:
        raise ValueError(
            f"Metric frames ({n}) do not match ledger ({ledger.frame_count})"
        )
    halves = [
        cv2.resize(frames[i], (80, 45), interpolation=cv2.INTER_AREA) for i in range(n)
    ]
    hashes = [_phash64(frames[i]) for i in range(n)]
    hists = [_histogram(frames[i]) for i in range(n)]
    edge_maps: list[float] = []
    for i in range(n):
        gx = cv2.Sobel(frames[i], cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(frames[i], cv2.CV_32F, 0, 1, ksize=3)
        edge_maps.append(float(np.sqrt(gx**2 + gy**2).mean()) / 255.0)

    pairs: list[PairMetrics] = []
    for i in range(n - 1):
        left = frames[i].astype(np.int16)
        right = frames[i + 1].astype(np.int16)
        mad = float(np.abs(right - left).mean())
        flow_mag, flow_coh = _flow_summary(halves[i], halves[i + 1])
        left_rec = ledger.frames[i]
        right_rec = ledger.frames[i + 1]
        pairs.append(
            PairMetrics(
                left_frame_index=i,
                right_frame_index=i + 1,
                left_pts=left_rec.pts,
                right_pts=right_rec.pts,
                left_pts_time_seconds=left_rec.pts_time_seconds,
                right_pts_time_seconds=right_rec.pts_time_seconds,
                mean_abs_diff=round(mad, 4),
                hist_distance=round(_bhattacharyya(hists[i], hists[i + 1]), 6),
                phash_hamming=_hamming64(hashes[i], hashes[i + 1]),
                edge_change=round(abs(edge_maps[i + 1] - edge_maps[i]), 6),
                luma_delta=round(float(right.mean() - left.mean()), 4),
                flow_mean_mag=round(flow_mag, 4),
                flow_coherence=round(flow_coh, 4),
            )
        )
    return pairs
