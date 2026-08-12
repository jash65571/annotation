"""Character / object continuity hypotheses from local tracks.

C/O labels are proposed in verified first-appearance order. Two visually similar
entities are NEVER merged into one hypothesis — each track gets its own
hypothesis, and any same-entity guess is a review-required ContinuityLink, never
an automatic merge. Protected traits (race/ethnicity/nationality/gender/age/
accent) are never inferred from imagery here.
"""

from __future__ import annotations

from ..models.review_intelligence import (
    CharacterHypothesis,
    ContinuityLink,
    EntityTrack,
    IdentityEvidence,
    ObjectHypothesis,
    TrackStatus,
)


def _identity_evidence(track: EntityTrack) -> list[IdentityEvidence]:
    evidence = [
        IdentityEvidence(
            kind="bbox_continuity",
            detail=f"tracked frames {track.first_frame_index}-{track.last_frame_index}",
            supports=True,
        )
    ]
    if track.reacquired:
        evidence.append(
            IdentityEvidence(
                kind="reacquisition",
                detail="track was lost and reacquired; identity not proven by similarity",
                supports=False,
            )
        )
    if track.identity_ambiguous:
        evidence.append(
            IdentityEvidence(
                kind="identity_ambiguous",
                detail="a near-equal competitor / implausible jump made identity "
                "ambiguous; the track may have wanted to hop to a similar entity",
                supports=False,
            )
        )
    return evidence


def build_continuity(
    tracks: list[EntityTrack],
) -> tuple[list[CharacterHypothesis], list[ObjectHypothesis], list[ContinuityLink]]:
    characters = sorted(
        (t for t in tracks if t.entity_type.upper() == "CHARACTER"),
        key=lambda t: t.first_frame_index,
    )
    objects = sorted(
        (t for t in tracks if t.entity_type.upper() == "OBJECT"),
        key=lambda t: t.first_frame_index,
    )

    char_hyps: list[CharacterHypothesis] = []
    for i, track in enumerate(characters, start=1):
        char_hyps.append(
            CharacterHypothesis(
                hypothesis_id=f"H-C{i}",
                proposed_label=f"C{i}",  # candidate label, review_required
                track_ids=[track.track_id],
                first_visible_frame=track.first_frame_index,
                last_visible_frame=track.last_frame_index,
                identity_evidence=_identity_evidence(track),
                review_required=True,
            )
        )

    obj_hyps: list[ObjectHypothesis] = []
    for i, track in enumerate(objects, start=1):
        obj_hyps.append(
            ObjectHypothesis(
                hypothesis_id=f"H-O{i}",
                proposed_label=f"O{i}",
                track_ids=[track.track_id],
                first_visible_frame=track.first_frame_index,
                last_visible_frame=track.last_frame_index,
                identity_evidence=_identity_evidence(track),
                review_required=True,
            )
        )

    # Same-entity guesses are review-required links, NEVER automatic merges.
    links: list[ContinuityLink] = []
    counter = 0
    for group in (characters, objects):
        for a, b in _pairs(group):
            if _temporally_disjoint(a, b) and (a.reacquired or b.reacquired):
                counter += 1
                links.append(
                    ContinuityLink(
                        link_id=f"LNK-{counter:03d}",
                        from_track_id=a.track_id,
                        to_track_id=b.track_id,
                        relationship="SAME_ENTITY_CANDIDATE",
                        review_required=True,
                        notes=["temporally disjoint tracks; possible reacquisition — review"],
                    )
                )
    return char_hyps, obj_hyps, links


def _pairs(items: list[EntityTrack]) -> list[tuple[EntityTrack, EntityTrack]]:
    out: list[tuple[EntityTrack, EntityTrack]] = []
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            out.append((items[i], items[j]))
    return out


def _temporally_disjoint(a: EntityTrack, b: EntityTrack) -> bool:
    return a.last_frame_index < b.first_frame_index or b.last_frame_index < a.first_frame_index


def status_summary(tracks: list[EntityTrack]) -> dict[str, int]:
    counts = {s.value: 0 for s in TrackStatus}
    for track in tracks:
        counts[track.status.value] += 1
    return counts
