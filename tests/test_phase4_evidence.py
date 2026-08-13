"""Phase 4 evidence-bundle tests (Z) + visual-reasoner contract tests (AD)."""

from __future__ import annotations

import json
from fractions import Fraction
from pathlib import Path

import numpy as np

from manuscript_reviewer.media import frames as frames_mod
from manuscript_reviewer.media import probe as probe_mod
from manuscript_reviewer.media.clock import AnnotationClock
from manuscript_reviewer.models.review_intelligence import (
    ReviewerAction,
    ReviewPriority,
    ReviewQueueItem,
)
from manuscript_reviewer.review.evidence_bundles import extract_evidence_bundles
from manuscript_reviewer.shots.evidence import SELECT_CHUNK_SIZE, extract_frames_by_index
from manuscript_reviewer.visual.decode import FrameCache
from manuscript_reviewer.visual.reasoner import (
    VisualReasonerAdapter,
    VisualReasonerRequest,
    VisualReasonerResponse,
)

from .conftest import requires_ffmpeg, synth_clip

# --------------------------------------------------------------------------
# AD: visual-reasoner contract (design only)
# --------------------------------------------------------------------------


def test_visual_reasoner_response_has_no_timestamp_field() -> None:
    fields = set(VisualReasonerResponse.model_fields)
    # The contract cites frames, never timestamps.
    assert "supporting_frame_ids" in fields
    assert "contradicting_frame_ids" in fields
    assert not any("time" in f or "timestamp" in f for f in fields)


def test_mock_reasoner_conforms_to_protocol() -> None:
    class _Mock:
        def reason(self, request: VisualReasonerRequest) -> VisualReasonerResponse:
            return VisualReasonerResponse(
                proposal="candidate", supporting_frame_ids=list(request.frame_ids),
                uncertainty=0.5,
            )

    adapter: VisualReasonerAdapter = _Mock()
    resp = adapter.reason(VisualReasonerRequest(question="what?", frame_ids=[101, 108]))
    assert isinstance(adapter, VisualReasonerAdapter)
    assert resp.supporting_frame_ids == [101, 108]
    assert resp.review_required is True


# --------------------------------------------------------------------------
# Z: high-risk evidence bundles
# --------------------------------------------------------------------------


def _build_ledger(path: Path):  # type: ignore[no-untyped-def]
    media, _ = probe_mod.probe_media(path)
    stream = media.video_streams[0]
    return frames_mod.enumerate_frames(path, stream.time_base, stream_index=0)


@requires_ffmpeg
def test_evidence_bundle_has_role_frames_with_identity(clip_24fps: Path, tmp_path: Path) -> None:
    ledger = _build_ledger(clip_24fps)
    cache = FrameCache(clip_24fps, ledger)
    clock = AnnotationClock.from_ledger(ledger)

    def frame_time(i: int) -> Fraction | None:
        rec = ledger.frames[i]
        return clock.to_annotation(rec.pts_time_seconds) if rec.pts_time_seconds else None

    item = ReviewQueueItem(
        item_id="RQ-0001", priority=ReviewPriority.CRITICAL, title="foundation contradiction",
        reason="seed shots != verified", start_frame=5, end_frame=10,
        recommended_action=ReviewerAction.REBUILD_SECTION,
    )
    out = tmp_path / "visual_evidence"
    written = extract_evidence_bundles([item], cache, ledger.frame_count, frame_time, out)
    bundle = out / "review_0001"
    for role in ("before", "start", "middle", "end", "after"):
        assert (bundle / f"{role}.png").is_file()
    assert (bundle / "strip.png").is_file()
    ev = json.loads((bundle / "evidence.json").read_text(encoding="utf-8"))
    # Every image carries exact frame identity + role.
    roles = {img["role"]: img for img in ev["images"]}
    assert roles["start"]["frame_index"] == 5
    assert roles["end"]["frame_index"] == 10
    assert all(img["annotation_time"] is not None for img in ev["images"])
    assert item.evidence_bundle_dir == "visual_evidence/review_0001"
    assert written


@requires_ffmpeg
def test_extract_frames_by_index_survives_large_index_set(tmp_path: Path) -> None:
    """Regression: one candidate's context window can request >100 frames.

    FFmpeg's expression parser refuses an ``eq(n\\,X)+...`` select chain past
    ~100 terms (ENOMEM). ``extract_frames_by_index`` must chunk below that
    ceiling, so a large single request splits across invocations instead of
    crashing. Before the fix the chunk size (120) exceeded the ceiling and this
    request died with exit -12 on a real clip.
    """
    # Guardrail: the chunk size must stay under the empirically-confirmed
    # ffmpeg ceiling (n8.1.2 and pinned 9.0 both fail at ~105, pass at 96).
    assert SELECT_CHUNK_SIZE <= 96

    n = 130  # > 100, so a single 120-term chunk (old behaviour) would crash
    frames = [np.full((64, 64, 3), i % 251, dtype=np.uint8) for i in range(n)]
    clip = synth_clip(tmp_path / "long.mp4", frames, fps=25)

    images = extract_frames_by_index(clip, list(range(n)), width=32)

    assert set(images) == set(range(n))
    assert len(images) == n
    assert all(img.shape[2] == 3 for img in images.values())
