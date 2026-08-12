"""Ownership / contact events between character and object tracks (V).

Exact frame candidate transitions: CONTACT_BEGINS/ENDS, HELD_STATE_BEGINS/ENDS,
OBJECT_RELEASE, OBJECT_PICKUP_CANDIDATE, CONTACT_STATE_CHANGE. A semantic "picks
up" is only proposed when the state sequence (object separate+static → contact →
object moves with character) supports it; otherwise the neutral
CONTACT_STATE_CHANGE is retained.
"""

from __future__ import annotations

import itertools

from ..models.review_intelligence import (
    ContactEvent,
    ContactEventKind,
    EntityTrack,
    ObjectStateKind,
    TrackObservation,
)

#: Boxes closer than this (as a fraction of the mean box size) are "in contact".
_CONTACT_FACTOR = 0.6
#: Frames of moving-together needed to call a HELD state.
_HELD_MIN = 3
#: Per-frame center movement below this (px) is "static".
_STATIC_EPS = 2.0


def _center(obs: TrackObservation) -> tuple[float, float]:
    return obs.x + obs.width / 2.0, obs.y + obs.height / 2.0


def _in_contact(a: TrackObservation, b: TrackObservation) -> bool:
    ax, ay = _center(a)
    bx, by = _center(b)
    dist = ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5
    scale = (a.width + a.height + b.width + b.height) / 4.0
    return bool(dist <= _CONTACT_FACTOR * scale)


def _by_frame(track: EntityTrack) -> dict[int, TrackObservation]:
    return {o.frame_index: o for o in track.observations}


def build_contact_events(
    character_tracks: list[EntityTrack], object_tracks: list[EntityTrack]
) -> list[ContactEvent]:
    events: list[ContactEvent] = []
    counter = 0
    for obj in object_tracks:
        obj_frames = _by_frame(obj)
        for char in character_tracks:
            char_frames = _by_frame(char)
            shared = sorted(set(obj_frames) & set(char_frames))
            if len(shared) < 2:
                continue
            counter = _pair_events(
                obj, char, obj_frames, char_frames, shared, events, counter
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
) -> int:
    contact_prev = False
    held = False
    held_run = 0
    obj_static_before = _is_static(obj_frames, shared[0])

    for idx, f in enumerate(shared):
        contact = _in_contact(obj_frames[f], char_frames[f])
        moving_together = idx > 0 and _moving_together(
            obj_frames, char_frames, shared[idx - 1], f
        )

        if contact and not contact_prev:
            counter += 1
            events.append(_event(counter, ContactEventKind.CONTACT_BEGINS, obj, char, f,
                                  ObjectStateKind.UNASSIGNED,
                                  ObjectStateKind.IN_CONTACT_WITH_CHARACTER))
        elif not contact and contact_prev:
            counter += 1
            post = ObjectStateKind.MOVING_INDEPENDENTLY if not held else ObjectStateKind.ON_SURFACE
            events.append(_event(counter, ContactEventKind.CONTACT_ENDS, obj, char, f,
                                 ObjectStateKind.IN_CONTACT_WITH_CHARACTER, post))
            if held:
                counter += 1
                events.append(_event(counter, ContactEventKind.OBJECT_RELEASE, obj, char, f,
                                     ObjectStateKind.HELD_BY_CHARACTER,
                                     ObjectStateKind.MOVING_INDEPENDENTLY))
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
                                     ObjectStateKind.HELD_BY_CHARACTER))
                # Pickup candidate only if the object was static & separate first.
                if obj_static_before:
                    counter += 1
                    events.append(_event(counter, ContactEventKind.OBJECT_PICKUP_CANDIDATE,
                                         obj, char, begin_frame, ObjectStateKind.ON_SURFACE,
                                         ObjectStateKind.HELD_BY_CHARACTER))
        elif contact:
            held_run = 0
        contact_prev = contact
    return counter


def _is_static(frames: dict[int, TrackObservation], start: int) -> bool:
    keys = sorted(k for k in frames if k >= start)[:_HELD_MIN + 1]
    for a, b in itertools.pairwise(keys):
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
    # Both must actually be moving, and by a similar vector.
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
) -> ContactEvent:
    return ContactEvent(
        event_id=f"CON-{counter:04d}",
        kind=kind,
        frame_index=frame,
        object_track_id=obj.track_id,
        character_track_id=char.track_id,
        pre_state=pre,
        post_state=post,
        review_required=True,
    )
