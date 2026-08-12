"""Caption planning (§12/§13): a complete structured plan before rendering.

The plan is assembled ONLY from the eligibility-gated fact graph plus the
verified shot structure. The renderer consumes the plan and never rediscovers
facts. Shot structure always comes from Shot Truth — never from seed shot
numbers (§22); a REDO disposition never preserves bad seeded structure.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from ..media.timestamps import format_manuscript_display
from ..models.caption_brain import (
    CaptionEligibility,
    CaptionFact,
    CaptionFactType,
    CaptionPlan,
    CaptionReadiness,
    CharacterPlanEntry,
    ObjectPlanEntry,
    OverviewPlan,
    SeedChangeEntry,
    ShotPlan,
)
from ..models.review_intelligence import (
    FeedbackDirective,
    FeedbackPriority,
    ReviewProposal,
)
from ..models.shot_truth import ShotTruthResult
from .facts import FactGraph

_ID_NUM = re.compile(r"^[CO](\d+)$")


@dataclass
class PlanInputs:
    graph: FactGraph
    shot_truth: ShotTruthResult | None = None
    video_id: str | None = None
    video_sha256: str | None = None
    rules_version: str | None = None
    proposals: list[ReviewProposal] = field(default_factory=list)
    feedback: list[FeedbackDirective] = field(default_factory=list)
    #: Directive ids a human explicitly resolved (via decision or human fact).
    resolved_directive_ids: frozenset[str] = frozenset()


def _id_sort_key(entity_id: str) -> tuple[int, str]:
    match = _ID_NUM.match(entity_id)
    return (int(match.group(1)) if match else 10_000, entity_id)


def _eligible_ids(facts: list[CaptionFact]) -> list[str]:
    return [f.fact_id for f in facts if f.eligibility == CaptionEligibility.ELIGIBLE]


def build_caption_plan(inputs: PlanInputs) -> CaptionPlan:
    graph = inputs.graph
    facts = graph.facts
    blockers: list[str] = []

    overview = _build_overview_plan(graph)
    shot_plans = _build_shot_plans(inputs, blockers)

    eligible = [f.fact_id for f in facts if f.eligibility == CaptionEligibility.ELIGIBLE]
    review = [
        f.fact_id for f in facts if f.eligibility == CaptionEligibility.REVIEW_REQUIRED
    ]
    blocked = [
        f.fact_id
        for f in facts
        if f.eligibility in (CaptionEligibility.INELIGIBLE, CaptionEligibility.REJECTED)
    ]

    for fact in facts:
        unresolved_required = (
            fact.eligibility == CaptionEligibility.REVIEW_REQUIRED
            and fact.materiality.value == "REQUIRED"
        )
        # §5.1-5: material media content that is currently INELIGIBLE (audible
        # unverified speech, a material unverified overlay...) also blocks —
        # eligibility alone never decides materiality.
        blocked_material = (
            fact.resolution_required
            and fact.eligibility != CaptionEligibility.ELIGIBLE
        )
        if unresolved_required or blocked_material:
            blockers.append(
                f"required fact {fact.fact_id} ({fact.fact_type.value}) unresolved: "
                f"{fact.eligibility_reason}"
            )

    for directive in inputs.feedback:
        if (
            directive.priority == FeedbackPriority.HIGH
            and directive.directive_id not in inputs.resolved_directive_ids
        ):
            blockers.append(
                f"unresolved HIGH task-feedback directive {directive.directive_id}"
            )

    readiness = (
        CaptionReadiness.REVIEW_REQUIRED if blockers else CaptionReadiness.READY_FOR_FINAL_REVIEW
    )
    if inputs.shot_truth is None or not inputs.shot_truth.shots:
        readiness = CaptionReadiness.BLOCKED
        blockers.append("no verified shot structure — caption cannot be planned")

    return CaptionPlan(
        video_id=inputs.video_id,
        video_sha256=inputs.video_sha256,
        rules_version=inputs.rules_version,
        overview_plan=overview,
        shot_plans=shot_plans,
        eligible_fact_ids=eligible,
        omitted_fact_ids=[],
        blocked_fact_ids=blocked,
        review_required_fact_ids=review,
        seed_change_summary=_seed_change_summary(inputs),
        readiness=readiness,
        blockers=blockers,
    )


def _build_overview_plan(graph: FactGraph) -> OverviewPlan:
    characters: dict[str, CharacterPlanEntry] = {}
    for fact in graph.of_type(CaptionFactType.CHARACTER):
        if fact.eligibility != CaptionEligibility.ELIGIBLE:
            continue
        for char_id in fact.character_ids:
            entry = characters.setdefault(char_id, CharacterPlanEntry(character_id=char_id))
            entry.description_fact_ids.append(fact.fact_id)
            if fact.semantic_value.get("off_screen") == "true":
                entry.off_screen_only = True

    objects: dict[str, ObjectPlanEntry] = {}
    for fact in graph.of_type(CaptionFactType.OBJECT):
        if fact.eligibility != CaptionEligibility.ELIGIBLE:
            continue
        for obj_id in fact.object_ids:
            obj_entry = objects.setdefault(obj_id, ObjectPlanEntry(object_id=obj_id))
            obj_entry.description_fact_ids.append(fact.fact_id)
    for fact in graph.of_type(CaptionFactType.FINAL_OBJECT_STATE):
        if fact.eligibility != CaptionEligibility.ELIGIBLE:
            continue
        for obj_id in fact.object_ids:
            if obj_id in objects:
                objects[obj_id].final_state_fact_ids.append(fact.fact_id)

    return OverviewPlan(
        characters=[characters[c] for c in sorted(characters, key=_id_sort_key)],
        objects=[objects[o] for o in sorted(objects, key=_id_sort_key)],
        scene_fact_ids=_eligible_ids(
            [f for f in graph.of_type(CaptionFactType.SCENE) if f.shot_number is None]
        ),
        style_fact_ids=_eligible_ids(graph.of_type(CaptionFactType.STYLE)),
        overview_audio_fact_ids=_eligible_ids(graph.of_type(CaptionFactType.OVERVIEW_AUDIO)),
        visual_concern_fact_ids=_eligible_ids(graph.of_type(CaptionFactType.VISUAL_CONCERN)),
        audio_concern_fact_ids=_eligible_ids(graph.of_type(CaptionFactType.AUDIO_CONCERN)),
    )


#: Action & Audio event union membership (§39): camera movement is NOT an event.
_EVENT_TYPES = (
    CaptionFactType.VISUAL_ACTION,
    CaptionFactType.SPEECH,
    CaptionFactType.SOUND,
    CaptionFactType.ON_SCREEN_TEXT,
)


def _build_shot_plans(inputs: PlanInputs, blockers: list[str]) -> list[ShotPlan]:
    if inputs.shot_truth is None:
        return []
    graph = inputs.graph
    plans: list[ShotPlan] = []
    for shot in inputs.shot_truth.shots:
        if shot.start_exact is None or shot.end_exact is None:
            blockers.append(f"shot {shot.shot_index} lacks exact interval times")
            continue
        shot_facts = [f for f in graph.facts if f.shot_number == shot.shot_index]

        transition = next(
            (f for f in shot_facts if f.fact_type == CaptionFactType.TRANSITION), None
        )
        transition_resolved = (
            transition is not None and transition.eligibility == CaptionEligibility.ELIGIBLE
        )
        if not transition_resolved:
            blockers.append(f"shot {shot.shot_index} transition unresolved")

        speed = next(
            (f for f in shot_facts if f.fact_type == CaptionFactType.PLAYBACK_SPEED), None
        )
        speed_resolved = (
            speed is not None
            and speed.eligibility == CaptionEligibility.ELIGIBLE
            and speed.text_value is not None
        )
        if not speed_resolved:
            blockers.append(f"shot {shot.shot_index} playback speed unverified")

        # Truthful overlap is preserved: events are ordered by exact start then
        # end, never flattened or nudged (§45).
        events = sorted(
            (
                f
                for f in shot_facts
                if f.fact_type in _EVENT_TYPES
                and f.eligibility == CaptionEligibility.ELIGIBLE
            ),
            key=lambda f: (
                f.start_exact if f.start_exact is not None else 0,
                f.end_exact if f.end_exact is not None else 0,
                f.fact_id,
            ),
        )

        plans.append(
            ShotPlan(
                shot_number=shot.shot_index,
                start_exact=shot.start_exact,
                end_exact=shot.end_exact,
                display_start=format_manuscript_display(shot.start_exact),
                display_end=format_manuscript_display(shot.end_exact),
                transition_fact_id=transition.fact_id if transition is not None else None,
                transition_resolved=transition_resolved,
                camera_framing_fact_ids=_eligible_ids(
                    [f for f in shot_facts if f.fact_type == CaptionFactType.CAMERA_FRAMING]
                ),
                camera_movement_fact_ids=_eligible_ids(
                    [f for f in shot_facts if f.fact_type == CaptionFactType.CAMERA_MOVEMENT]
                ),
                scene_fact_ids=_eligible_ids(
                    [f for f in shot_facts if f.fact_type == CaptionFactType.SCENE]
                ),
                event_fact_ids=[f.fact_id for f in events],
                playback_speed_fact_id=speed.fact_id if speed is not None else None,
                playback_speed_resolved=speed_resolved,
                speed_change_fact_ids=_eligible_ids(
                    [f for f in shot_facts if f.fact_type == CaptionFactType.SPEED_CHANGE]
                ),
            )
        )
    return plans


def _seed_change_summary(inputs: PlanInputs) -> list[SeedChangeEntry]:
    """Final dispositions per section (§58). A machine proposal alone is not a
    human decision — ``decided_by_human`` distinguishes them (§13)."""
    entries: list[SeedChangeEntry] = []
    for proposal in inputs.proposals:
        if proposal.level not in ("seed", "overview", "shot", "character", "object"):
            continue
        disposition = proposal.outcome.value
        if disposition == "HUMAN_DECISION_REQUIRED":
            disposition = "REDO_REBUILD" if proposal.foundational else "FIX_ENRICH"
        entries.append(
            SeedChangeEntry(
                section=proposal.level,
                subject_id=proposal.subject_id,
                disposition=disposition,
                decided_by_human=proposal.proposed_by != "machine",
                reasons=[c.value for c in proposal.reason_codes],
            )
        )
    return entries
