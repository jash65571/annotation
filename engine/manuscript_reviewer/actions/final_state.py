"""Final object-state checker (W, item 5) — mandatory per anchored object/shot.

Only evidence-supported states are asserted. "Not held" is NEVER ON_SURFACE:
an object merely visible at shot end is VISIBLE (present, contact/surface state
undetermined). Removal/exit is never inferred because the shot or track ends.
``final_visible_frame`` is an actually-visible (TRACKED) frame — an OCCLUDED
observation is not visible.
"""

from __future__ import annotations

from ..models.evidence import EvidenceReference, EvidenceType
from ..models.review_intelligence import (
    ContactEvent,
    ContactEventKind,
    EntityTrack,
    FinalStateCheck,
    ObjectStateKind,
    TrackObservation,
    TrackStatus,
)
from ..models.shot_truth import ShotTruthResult


def _held_at(events: list[ContactEvent], object_track_id: str, frame: int) -> bool:
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
            checks.append(_check(shot.shot_index, track, in_shot, hi, events))
    return checks


def _check(
    shot_number: int,
    track: EntityTrack,
    in_shot: list[TrackObservation],
    shot_end: int,
    events: list[ContactEvent],
) -> FinalStateCheck:
    visible = [o for o in in_shot if o.status == TrackStatus.TRACKED]
    last_obs_frame = max(o.frame_index for o in in_shot)
    last_vis_frame = max((o.frame_index for o in visible), default=None)

    entity = track.track_id
    if last_vis_frame is None:
        # Only ever occluded/uncertain within the shot: unresolved.
        return FinalStateCheck(
            shot_number=shot_number, entity_id=entity, final_visible_frame=None,
            last_observation_frame=last_obs_frame, still_visible_at_shot_end=False,
            final_state=ObjectStateKind.OCCLUDED, resolved=False,
            review_reason="never verified visible in this shot; final state unresolved",
        )

    reaches_end = last_vis_frame >= shot_end
    ref = EvidenceReference(
        evidence_id=f"EV-FINAL-{entity}-{shot_number}",
        evidence_type=EvidenceType.FINAL_STATE,
        start_frame=last_vis_frame,
        source=entity,
    )
    if reaches_end and _held_at(events, entity, last_vis_frame):
        return FinalStateCheck(
            shot_number=shot_number, entity_id=entity, final_visible_frame=last_vis_frame,
            last_observation_frame=last_obs_frame, still_visible_at_shot_end=True,
            final_state=ObjectStateKind.HELD_BY_CHARACTER, resolved=True,
            evidence_refs=[ref], review_reason="held at shot end",
        )
    if reaches_end:
        # Visible at shot end but contact/surface state undetermined — VISIBLE,
        # NEVER ON_SURFACE (item 5). Presence is resolved; the state is flagged.
        return FinalStateCheck(
            shot_number=shot_number, entity_id=entity, final_visible_frame=last_vis_frame,
            last_observation_frame=last_obs_frame, still_visible_at_shot_end=True,
            final_state=ObjectStateKind.VISIBLE, resolved=True, evidence_refs=[ref],
            review_reason="visible at shot end; contact/surface state undetermined",
        )
    # Last verified visible before shot end: do NOT infer removal/exit.
    return FinalStateCheck(
        shot_number=shot_number, entity_id=entity, final_visible_frame=last_vis_frame,
        last_observation_frame=last_obs_frame, still_visible_at_shot_end=False,
        final_state=ObjectStateKind.REVIEW_REQUIRED, resolved=False, evidence_refs=[ref],
        review_reason="last verified visible before shot end; removal/exit not inferred",
    )
