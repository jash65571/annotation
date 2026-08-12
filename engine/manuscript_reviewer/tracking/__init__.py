"""Tracking subsystem (Phase 4): anchor-seeded, explainable local tracking.

No cloud detector, no general multi-object tracking — a human/detector anchor
seeds forward/backward local tracking (template + appearance histogram) on the
shared gray metric grid (no new decode). A similarity score is NEVER identity
truth; a reacquisition is never silently accepted; two visually similar
entities are never merged without continuity evidence.
"""
