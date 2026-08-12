"""Shot proposals: gapless, ordered, non-overlapping coverage from supported
boundaries. Frame ownership: a boundary's left frame is the final frame of the
outgoing shot; its right frame is the first frame of the incoming shot."""

from __future__ import annotations

from ..media.timestamps import format_manuscript_display
from ..models.frame import FrameLedger
from ..models.shot_truth import (
    BoundaryCandidate,
    CandidateStatus,
    ShotProposal,
    TransitionStatus,
)


def build_shot_proposals(
    ledger: FrameLedger, candidates: list[BoundaryCandidate]
) -> list[ShotProposal]:
    supported = sorted(
        (c for c in candidates if c.status == CandidateStatus.SUPPORTED),
        key=lambda c: c.right_frame_index,
    )
    n = ledger.frame_count
    shots: list[ShotProposal] = []
    start_index = 0
    boundaries: list[BoundaryCandidate | None] = [*supported, None]

    for shot_number, boundary in enumerate(boundaries, start=1):
        end_index = (boundary.left_frame_index if boundary is not None else n - 1)
        start_rec = ledger.frames[start_index]
        end_rec = ledger.frames[end_index]
        transition_into: str | None
        if shot_number == 1:
            transition_into = "Opening shot"
            transition_status = TransitionStatus.PROPOSED
            supporting_id: str | None = None
        else:
            incoming = boundaries[shot_number - 2]
            assert incoming is not None
            supporting_id = incoming.candidate_id
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
                start_exact=start_rec.pts_time_seconds,
                end_exact=end_rec.pts_time_seconds,
                start_manuscript=(
                    format_manuscript_display(start_rec.pts_time_seconds)
                    if start_rec.pts_time_seconds is not None
                    else None
                ),
                end_manuscript=(
                    format_manuscript_display(end_rec.pts_time_seconds)
                    if end_rec.pts_time_seconds is not None
                    else None
                ),
                transition_into_shot=transition_into,
                transition_status=transition_status,
                supporting_boundary_id=supporting_id,
                review_status=(
                    CandidateStatus.SUPPORTED
                    if shot_number == 1 or boundaries[shot_number - 2] is None
                    else boundaries[shot_number - 2].status  # type: ignore[union-attr]
                ),
            )
        )
        if boundary is not None:
            start_index = boundary.right_frame_index
    return shots
