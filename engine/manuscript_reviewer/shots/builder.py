"""Shot proposals: gapless, ordered, non-overlapping coverage from supported
boundaries.

Frame ownership: a boundary's left frame is the final frame of the outgoing
shot; its right frame is the first frame of the incoming shot.

Annotation timeline: the SAME exact boundary time — the incoming/right frame's
PTS — is both the outgoing shot's interval end and the incoming shot's
interval start, expressed on the ANNOTATION clock (source minus origin). Raw
source values are preserved separately. The final shot's interval end is the
canonical media annotation endpoint (media/endpoint.py), never the final
frame's start PTS.
"""

from __future__ import annotations

from fractions import Fraction

from ..media.clock import AnnotationClock
from ..media.timestamps import format_manuscript_display
from ..models.frame import FrameLedger
from ..models.shot_truth import (
    BoundaryCandidate,
    CandidateStatus,
    ShotProposal,
    TransitionStatus,
)


def _display(value: Fraction | None) -> str | None:
    return format_manuscript_display(value) if value is not None else None


def build_shot_proposals(
    ledger: FrameLedger,
    candidates: list[BoundaryCandidate],
    source_annotation_endpoint: Fraction | None,
    clock: AnnotationClock,
) -> list[ShotProposal]:
    """Build gapless shot proposals. ``source_annotation_endpoint`` is the
    canonical endpoint on the SOURCE clock; annotation values derive via
    ``clock``."""
    supported = sorted(
        (c for c in candidates if c.status == CandidateStatus.SUPPORTED),
        key=lambda c: c.right_frame_index,
    )
    n = ledger.frame_count
    shots: list[ShotProposal] = []
    start_index = 0
    boundaries: list[BoundaryCandidate | None] = [*supported, None]
    #: Timeline cursor on the SOURCE clock; starts at the first frame's PTS.
    source_cursor = ledger.frames[0].pts_time_seconds if ledger.frames else None

    for shot_number, boundary in enumerate(boundaries, start=1):
        end_index = boundary.left_frame_index if boundary is not None else n - 1
        end_rec = ledger.frames[end_index]

        if boundary is not None:
            # Interval end = incoming/right frame PTS = the boundary exact time.
            source_end = ledger.frames[boundary.right_frame_index].pts_time_seconds
        else:
            source_end = source_annotation_endpoint

        annotation_start = (
            clock.to_annotation(source_cursor) if source_cursor is not None else None
        )
        annotation_end = clock.to_annotation(source_end) if source_end is not None else None

        transition_into: str | None
        if shot_number == 1:
            transition_into = "Opening shot"
            transition_status = TransitionStatus.PROPOSED
            supporting_id: str | None = None
            review_status = CandidateStatus.SUPPORTED
        else:
            incoming = boundaries[shot_number - 2]
            assert incoming is not None
            supporting_id = incoming.candidate_id
            review_status = incoming.status
            if incoming.transition is not None:
                transition_into = incoming.transition.manuscript_type
                transition_status = incoming.transition.status
            else:
                transition_into = None
                transition_status = TransitionStatus.UNRESOLVED

        shots.append(
            ShotProposal(
                shot_index=shot_number,
                start_frame_index=start_index,
                end_frame_index=end_index,
                start_exact=annotation_start,
                end_exact=annotation_end,
                source_start_exact=source_cursor,
                source_end_exact=source_end,
                last_owned_frame_start_exact=end_rec.pts_time_seconds,
                start_manuscript=_display(annotation_start),
                end_manuscript=_display(annotation_end),
                transition_into_shot=transition_into,
                transition_status=transition_status,
                supporting_boundary_id=supporting_id,
                review_status=review_status,
            )
        )
        if boundary is not None:
            start_index = boundary.right_frame_index
            source_cursor = source_end
    return shots
