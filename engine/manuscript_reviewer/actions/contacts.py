"""Ownership / contact events between character and object tracks (V, item 4).

Contact is inferred ONLY from trusted-visible (TRACKED) observations on BOTH
entities — never from OCCLUDED / LOST / REVIEW_REQUIRED / untrusted-REACQUIRED
boxes. Motion-together requires genuinely consecutive frames (a multi-frame gap
is never treated as one step); contact breaks across missing evidence. Every
event carries its shot number and frame-anchored evidence.

A semantic "picks up" is only proposed when the state sequence (object separate +
static -> contact -> object moves with the character) supports it; otherwise the
neutral state change is retained.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable

from ..models.evidence import EvidenceReference, EvidenceType
from ..models.review_intelligence import (
    ContactEvent,
    ContactEventKind,
    EntityTrack,
    ObjectStateKind,
    TrackObservation,
    TrackStatus,
)

_CONTACT_FACTOR = 0.6
_HELD_MIN = 3
_STATIC_EPS = 2.0


def _center(obs: TrackObservation) -> tuple[float, float]:
    return obs.x + obs.width / 2.0, obs.y + obs.height / 2.0


def _in_contact(a: TrackObservation, b: TrackObservation) -> bool:
    ax, ay = _center(a)
    bx, by = _center(b)
    dist = ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5
    scale = (a.width + a.height + b.width + b.height) / 4.0
    return bool(dist <= _CONTACT_FACTOR * scale)


def _visible_by_frame(track: EntityTrack) -> dict[int, TrackObservation]:
    """Only TRACKED observations are trustworthy geometry (item 4)."""
    return {o.frame_index: o for o in track.observations if o.status == TrackStatus.TRACKED}


def _ref(track_id: str, frame_index: int) -> EvidenceReference:
    return EvidenceReference(
        evidence_id=f"EV-CON-{track_id}-{frame_index}",
        evidence_type=EvidenceType.CONTACT_STATE,
        start_frame=frame_index,
        source=track_id,
    )


def build_contact_events(
    character_tracks: list[EntityTrack],
    object_tracks: list[EntityTrack],
    frame_to_shot: Callable[[int], int | None] | None = None,
) -> list[ContactEvent]:
    events: list[ContactEvent] = []
    counter = 0
    shot_of = frame_to_shot or (lambda _f: None)
    for obj in object_tracks:
        obj_frames = _visible_by_frame(obj)
        for char in character_tracks:
            char_frames = _visible_by_frame(char)
            shared = sorted(set(obj_frames) & set(char_frames))
            if len(shared) < 2:
                continue
            counter = _pair_events(
                obj, char, obj_frames, char_frames, shared, events, counter, shot_of
            )
    return events


def _pair_events(
    obj: EntityTrack,
    char: EntityTrack,
    obj_frames: dict[int, TrackObservation],
    char_frames: dict[int, TrackObservation],
    shared: list[int],
    events: list[ContactEvent],
    counter: int,
    shot_of: Callable[[int], int | None],
) -> int:
    contact_prev = False
    held = False
    held_run = 0
    obj_static_before = _is_static(obj_frames, shared[0])

    for idx, f in enumerate(shared):
        contact = _in_contact(obj_frames[f], char_frames[f])
        # Motion-together only over genuinely consecutive visible frames.
        prev_f = shared[idx - 1] if idx > 0 else None
        moving_together = (
            prev_f is not None
            and f - prev_f == 1
            and _moving_together(obj_frames, char_frames, prev_f, f)
        )

        if contact and not contact_prev:
            counter += 1
            events.append(_event(counter, ContactEventKind.CONTACT_BEGINS, obj, char, f,
                                 ObjectStateKind.UNASSIGNED,
                                 ObjectStateKind.IN_CONTACT_WITH_CHARACTER, shot_of))
        elif not contact and contact_prev:
            counter += 1
            post = ObjectStateKind.MOVING_INDEPENDENTLY if not held else ObjectStateKind.ON_SURFACE
            events.append(_event(counter, ContactEventKind.CONTACT_ENDS, obj, char, f,
                                 ObjectStateKind.IN_CONTACT_WITH_CHARACTER, post, shot_of))
            if held:
                counter += 1
                events.append(_event(counter, ContactEventKind.OBJECT_RELEASE, obj, char, f,
                                     ObjectStateKind.HELD_BY_CHARACTER,
                                     ObjectStateKind.MOVING_INDEPENDENTLY, shot_of))
                held = False
                held_run = 0

        if contact and moving_together:
            held_run += 1
            if held_run >= _HELD_MIN and not held:
                held = True
                begin_frame = shared[max(0, idx - held_run + 1)]
                counter += 1
                events.append(_event(counter, ContactEventKind.HELD_STATE_BEGINS, obj, char,
                                     begin_frame, ObjectStateKind.IN_CONTACT_WITH_CHARACTER,
                                     ObjectStateKind.HELD_BY_CHARACTER, shot_of))
                if obj_static_before:
                    counter += 1
                    events.append(_event(counter, ContactEventKind.OBJECT_PICKUP_CANDIDATE,
                                         obj, char, begin_frame, ObjectStateKind.ON_SURFACE,
                                         ObjectStateKind.HELD_BY_CHARACTER, shot_of))
        elif contact:
            held_run = 0
        else:
            held_run = 0  # contact broke across missing/updated evidence
        contact_prev = contact
    return counter


def _is_static(frames: dict[int, TrackObservation], start: int) -> bool:
    keys = sorted(k for k in frames if k >= start)[: _HELD_MIN + 1]
    for a, b in itertools.pairwise(keys):
        if b - a != 1:  # non-consecutive: cannot claim static
            return False
        ca, cb = _center(frames[a]), _center(frames[b])
        if ((ca[0] - cb[0]) ** 2 + (ca[1] - cb[1]) ** 2) ** 0.5 > _STATIC_EPS:
            return False
    return True


def _moving_together(
    obj_frames: dict[int, TrackObservation],
    char_frames: dict[int, TrackObservation],
    prev_f: int,
    f: int,
) -> bool:
    ov = _delta(obj_frames[prev_f], obj_frames[f])
    cv = _delta(char_frames[prev_f], char_frames[f])
    if ov[0] ** 2 + ov[1] ** 2 < _STATIC_EPS**2:
        return False
    return abs(ov[0] - cv[0]) <= 3.0 and abs(ov[1] - cv[1]) <= 3.0


def _delta(a: TrackObservation, b: TrackObservation) -> tuple[float, float]:
    ca, cb = _center(a), _center(b)
    return cb[0] - ca[0], cb[1] - ca[1]


def _event(
    counter: int,
    kind: ContactEventKind,
    obj: EntityTrack,
    char: EntityTrack,
    frame: int,
    pre: ObjectStateKind,
    post: ObjectStateKind,
    shot_of: Callable[[int], int | None],
) -> ContactEvent:
    return ContactEvent(
        event_id=f"CON-{counter:04d}",
        kind=kind,
        shot_number=shot_of(frame),
        frame_index=frame,
        object_track_id=obj.track_id,
        character_track_id=char.track_id,
        pre_state=pre,
        post_state=post,
        review_required=True,
        evidence_refs=[_ref(obj.track_id, frame), _ref(char.track_id, frame)],
    )
