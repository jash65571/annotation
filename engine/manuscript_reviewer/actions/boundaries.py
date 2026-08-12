"""Atomic action-boundary candidates from defensible state changes (X, item 7).

Tracker loss is NOT proof of absence: ENTITY_APPEARS/EXITS are never inferred
from a track simply starting/ending. Motion boundaries are computed only over
genuinely consecutive trusted-visible (TRACKED) observations — never through
OCCLUDED frames, an unverified reacquisition, or a multi-frame gap. Every
candidate carries its shot number, frame-anchored evidence, exact first/last
supporting frame, and pre/post state. A semantic label stays ``None`` (never
forced from generic motion). Tracks are already shot-bounded (item 3), so a
candidate never spans an editorial cut.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable
from fractions import Fraction

from ..models.evidence import EvidenceReference, EvidenceType
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
FrameShot = Callable[[int], int | None]


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
    frame_shot: FrameShot | None = None,
) -> list[ActionCandidate]:
    candidates: list[ActionCandidate] = []
    counter = 0
    time_of = frame_time or (lambda _f: None)
    shot_of = frame_shot or (lambda _f: None)

    def emit(
        action_class: ActionStateClass,
        subject: str | None,
        obj: str | None,
        start: int,
        end: int,
        pre: str | None,
        post: str | None,
        refs: list[EvidenceReference],
    ) -> None:
        nonlocal counter
        counter += 1
        candidates.append(
            ActionCandidate(
                candidate_id=f"ACT-{counter:04d}",
                shot_number=shot_of(start),
                subject_track_ids=[subject] if subject else [],
                object_track_ids=[obj] if obj else [],
                action_class=action_class,
                semantic_label=None,  # never forced from generic motion
                start_frame=start,
                end_frame=end,
                start_exact=time_of(start),
                end_exact=time_of(end),
                pre_state=pre,
                post_state=post,
                supporting_observation_frames=sorted({start, end}),
                review_required=True,
                evidence_refs=refs,
            )
        )

    for track in character_tracks + object_tracks:
        _track_boundaries(track, emit)

    for ev in contact_events or []:
        action = _CONTACT_TO_ACTION.get(ev.kind)
        if action is not None:
            emit(action, ev.character_track_id, ev.object_track_id, ev.frame_index,
                 ev.frame_index, None, None, list(ev.evidence_refs))

    return candidates


def _ref(track_id: str, start: int, end: int) -> EvidenceReference:
    return EvidenceReference(
        evidence_id=f"EV-ACT-{track_id}-{start}-{end}",
        evidence_type=EvidenceType.ACTION_STATE_CHANGE,
        start_frame=start,
        end_frame=end,
        source=track_id,
    )


def _track_boundaries(
    track: EntityTrack,
    emit: Callable[..., None],
) -> None:
    subject = track.track_id
    obs = track.observations
    was_moving: bool | None = None
    for prev, cur in itertools.pairwise(obs):
        # Occlusion boundaries from status transitions (defensible: a verified
        # visible frame adjacent to an occluded one).
        if prev.status == TrackStatus.TRACKED and cur.status == TrackStatus.OCCLUDED:
            emit(ActionStateClass.OCCLUSION_BEGINS, subject, None,
                 prev.frame_index, cur.frame_index, "visible", "occluded",
                 [_ref(subject, prev.frame_index, cur.frame_index)])
        elif prev.status == TrackStatus.OCCLUDED and cur.status in (
            TrackStatus.TRACKED, TrackStatus.REACQUIRED
        ):
            emit(ActionStateClass.OCCLUSION_ENDS, subject, None,
                 prev.frame_index, cur.frame_index, "occluded", "visible",
                 [_ref(subject, prev.frame_index, cur.frame_index)])
        # Motion boundaries ONLY over consecutive trusted-visible observations.
        if not (
            prev.status == TrackStatus.TRACKED
            and cur.status == TrackStatus.TRACKED
            and cur.frame_index - prev.frame_index == 1
        ):
            was_moving = None  # break the motion state across any gap/occlusion
            continue
        moving = _moving(prev, cur)
        if was_moving is not None:
            if moving and not was_moving:
                emit(ActionStateClass.MOTION_BEGINS, subject, None,
                     prev.frame_index, cur.frame_index, "static", "moving",
                     [_ref(subject, prev.frame_index, cur.frame_index)])
            elif not moving and was_moving:
                emit(ActionStateClass.MOTION_ENDS, subject, None,
                     prev.frame_index, cur.frame_index, "moving", "static",
                     [_ref(subject, prev.frame_index, cur.frame_index)])
        was_moving = moving
