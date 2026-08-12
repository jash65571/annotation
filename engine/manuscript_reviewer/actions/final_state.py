"""Final object-state checker (W) — mandatory for every important object/shot.

Never infers release/removal because the shot ends: a track that stops before the
shot end with no leaving evidence is UNRESOLVED, not "released". A track present
at the shot's final frame persists to shot end.
"""

from __future__ import annotations

from ..models.review_intelligence import (
    ContactEvent,
    ContactEventKind,
    EntityTrack,
    FinalStateCheck,
    ObjectStateKind,
    TrackStatus,
)
from ..models.shot_truth import ShotTruthResult


def _held_at(events: list[ContactEvent], object_track_id: str, frame: int) -> bool:
    """Was the object held (HELD begun and not released) at/near ``frame``?"""
    held = False
    for ev in sorted(
        (e for e in events if e.object_track_id == object_track_id),
        key=lambda e: e.frame_index,
    ):
        if ev.frame_index > frame:
            break
        if ev.kind == ContactEventKind.HELD_STATE_BEGINS:
            held = True
        elif ev.kind in (ContactEventKind.OBJECT_RELEASE, ContactEventKind.HELD_STATE_ENDS):
            held = False
    return held


def build_final_state_checks(
    object_tracks: list[EntityTrack],
    shot_truth: ShotTruthResult | None,
    contact_events: list[ContactEvent] | None = None,
) -> list[FinalStateCheck]:
    if shot_truth is None:
        return []
    events = contact_events or []
    checks: list[FinalStateCheck] = []
    for shot in shot_truth.shots:
        lo, hi = shot.start_frame_index, shot.end_frame_index
        for track in object_tracks:
            in_shot = [o for o in track.observations if lo <= o.frame_index <= hi]
            if not in_shot:
                continue
            last = max(in_shot, key=lambda o: o.frame_index)
            checks.append(_check(shot.shot_index, track, last, hi, events))
    return checks


def _check(
    shot_number: int,
    track: EntityTrack,
    last_obs: object,
    shot_end: int,
    events: list[ContactEvent],
) -> FinalStateCheck:
    frame = last_obs.frame_index  # type: ignore[attr-defined]
    status = last_obs.status  # type: ignore[attr-defined]
    reaches_end = frame >= shot_end
    if status == TrackStatus.OCCLUDED:
        return FinalStateCheck(
            shot_number=shot_number, entity_id=track.track_id, final_visible_frame=frame,
            final_state=ObjectStateKind.OCCLUDED, resolved=False,
            review_reason="occluded at last observation; final state unresolved",
        )
    if reaches_end:
        if _held_at(events, track.track_id, frame):
            state = ObjectStateKind.HELD_BY_CHARACTER
        else:
            state = ObjectStateKind.ON_SURFACE
        return FinalStateCheck(
            shot_number=shot_number, entity_id=track.track_id, final_visible_frame=frame,
            final_state=state, resolved=True,
            review_reason=None if state == ObjectStateKind.ON_SURFACE else "still held at shot end",
        )
    # Track ended before the shot ended: do NOT infer removal/release.
    return FinalStateCheck(
        shot_number=shot_number, entity_id=track.track_id, final_visible_frame=frame,
        final_state=ObjectStateKind.REVIEW_REQUIRED, resolved=False,
        review_reason="track ended before shot end; removal/exit not inferred",
    )
