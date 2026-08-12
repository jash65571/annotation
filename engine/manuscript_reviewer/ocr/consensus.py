"""Temporal OCR consensus.

Several consecutive frames of the same text region are combined by exact-string
majority vote. A one-character misread ("ENEMY K1LLED") loses to the frequent
correct reading ("ENEMY KILLED") without ever *inventing* or silently correcting
words — the consensus is always one of the raw observed strings. Raw
observations are always retained by the caller.
"""

from __future__ import annotations

from collections import Counter

from ..models.review_intelligence import OCRObservation, TextConsensus


def temporal_consensus(observations: list[OCRObservation]) -> TextConsensus | None:
    """Majority-vote consensus over the raw OCR strings of one text track."""
    texts = [o.raw_text.strip() for o in observations if o.raw_text.strip()]
    if not texts:
        return None
    counts = Counter(texts)
    top_count = max(counts.values())
    # Break ties deterministically: among the most frequent readings, prefer the
    # one with the highest total confidence, then lexicographic order.
    candidates = sorted(t for t, c in counts.items() if c == top_count)
    if len(candidates) > 1:
        conf_by_text: dict[str, float] = {}
        for obs in observations:
            key = obs.raw_text.strip()
            if key in candidates:
                conf_by_text[key] = conf_by_text.get(key, 0.0) + (obs.confidence or 0.0)
        candidates.sort(key=lambda t: (-conf_by_text.get(t, 0.0), t))
    consensus_text = candidates[0]
    confidence = top_count / len(texts)
    return TextConsensus(
        consensus_text=consensus_text,
        support_frames=top_count,
        confidence=round(confidence, 4),
    )
