"""Atomic action-boundary candidates from defensible state changes (X).

Each candidate records the first/last supporting frame, the pre/post state, and
its evidence — it never carries a forced semantic label (picks up / throws /
raises hand). A semantic label stays ``None`` unless independently supported by a
later human/visual-reasoner pass; timing always comes from frame identity.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable
from fractions import Fraction

from ..models.review_intelligence import (
    ActionCandidate,
    ActionStateClass,
    ContactEvent,
    ContactEventKind,
    EntityTrack,
    TrackObservation,
    TrackStatus,
)

_MOTION_EPS = 2.0

_CONTACT_TO_ACTION = {
    ContactEventKind.CONTACT_BEGINS: ActionStateClass.CONTACT_BEGINS,
    ContactEventKind.CONTACT_ENDS: ActionStateClass.CONTACT_ENDS,
    ContactEventKind.OBJECT_RELEASE: ActionStateClass.OBJECT_SEPARATES,
}

FrameTime = Callable[[int], Fraction | None]


def _center(o: TrackObservation) -> tuple[float, float]:
    return o.x + o.width / 2.0, o.y + o.height / 2.0


def _moving(a: TrackObservation, b: TrackObservation) -> bool:
    ca, cb = _center(a), _center(b)
    return bool(((ca[0] - cb[0]) ** 2 + (ca[1] - cb[1]) ** 2) ** 0.5 > _MOTION_EPS)


def build_action_candidates(
    character_tracks: list[EntityTrack],
    object_tracks: list[EntityTrack],
    contact_events: list[ContactEvent] | None = None,
    frame_time: FrameTime | None = None,
) -> list[ActionCandidate]:
    candidates: list[ActionCandidate] = []
    counter = 0

    def emit(
        action_class: ActionStateClass,
        subject: str | None,
        obj: str | None,
        start: int,
        end: int,
        pre: str | None,
        post: str | None,
    ) -> None:
        nonlocal counter
        counter += 1
        candidates.append(
            ActionCandidate(
                candidate_id=f"ACT-{counter:04d}",
                shot_number=None,
                subject_track_ids=[subject] if subject else [],
                object_track_ids=[obj] if obj else [],
                action_class=action_class,
                semantic_label=None,  # never forced from generic motion
                start_frame=start,
                end_frame=end,
                start_exact=frame_time(start) if frame_time else None,
                end_exact=frame_time(end) if frame_time else None,
                pre_state=pre,
                post_state=post,
                supporting_observation_frames=sorted({start, end}),
                review_required=True,
            )
        )

    for track in character_tracks + object_tracks:
        _track_boundaries(track, emit)

    for ev in contact_events or []:
        action = _CONTACT_TO_ACTION.get(ev.kind)
        if action is not None:
            emit(action, ev.character_track_id, ev.object_track_id, ev.frame_index,
                 ev.frame_index, pre=None, post=None)

    return candidates


def _track_boundaries(
    track: EntityTrack,
    emit: Callable[..., None],
) -> None:
    subject = track.track_id
    # ENTITY_APPEARS at the first observed frame.
    emit(ActionStateClass.ENTITY_APPEARS, subject, None,
         track.first_frame_index, track.first_frame_index, pre="absent", post="present")

    obs = track.observations
    was_moving: bool | None = None
    for prev, cur in itertools.pairwise(obs):
        # Occlusion boundaries from status transitions.
        if prev.status == TrackStatus.TRACKED and cur.status == TrackStatus.OCCLUDED:
            emit(ActionStateClass.OCCLUSION_BEGINS, subject, None,
                 prev.frame_index, cur.frame_index, pre="visible", post="occluded")
        elif prev.status == TrackStatus.OCCLUDED and cur.status in (
            TrackStatus.TRACKED, TrackStatus.REACQUIRED
        ):
            emit(ActionStateClass.OCCLUSION_ENDS, subject, None,
                 prev.frame_index, cur.frame_index, pre="occluded", post="visible")
        # Motion begin/end boundaries.
        moving = _moving(prev, cur)
        if was_moving is not None:
            if moving and not was_moving:
                emit(ActionStateClass.MOTION_BEGINS, subject, None,
                     prev.frame_index, cur.frame_index, pre="static", post="moving")
            elif not moving and was_moving:
                emit(ActionStateClass.MOTION_ENDS, subject, None,
                     prev.frame_index, cur.frame_index, pre="moving", post="static")
        was_moving = moving
