"""Golden Example behavior gate (§83-§86).

Compares BEHAVIOR, never word/sentence/entry counts: is the caption at least
as evidence-complete, as carefully separated, as timing-disciplined, and as
overlap-faithful as the Golden Examples? Rules come from the committed derived
rubric ``rules/golden_behavior_v1.yaml`` (each rule carries source/page
provenance); the raw Golden PDF stays local-only.
"""

from __future__ import annotations

from fractions import Fraction
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from ..caption.coverage import CoverageResult
from ..media.timestamps import format_manuscript_display
from ..models.caption_brain import (
    CaptionEligibility,
    CaptionFact,
    CaptionFactType,
    CaptionPlan,
    CaptionSection,
    GoldenGateCategoryResult,
    GoldenGateResult,
    GoldenGateStatus,
    RenderedCaption,
)
from ..validation.platform_semantic_validator import PlatformSemanticReport

GOLDEN_RULES_FILE = Path(__file__).parent.parent / "rules" / "golden_behavior_v1.yaml"


@lru_cache(maxsize=1)
def load_golden_rules() -> dict[str, Any]:
    with GOLDEN_RULES_FILE.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict) or "golden_behavior_version" not in data:
        raise ValueError("golden_behavior_v1.yaml is missing golden_behavior_version")
    return data


def _rule_ids(rules: dict[str, Any], category: str) -> list[str]:
    entries = rules.get("categories", {}).get(category, [])
    return [str(e.get("id")) for e in entries if isinstance(e, dict)]


def run_golden_gate(
    plan: CaptionPlan,
    caption: RenderedCaption,
    facts: list[CaptionFact],
    coverage: CoverageResult,
    platform_report: PlatformSemanticReport,
) -> GoldenGateResult:
    rules = load_golden_rules()
    facts_by_id = {f.fact_id: f for f in facts}
    eligible = [f for f in facts if f.eligibility == CaptionEligibility.ELIGIBLE]
    rendered_fact_ids = {fid for ln in caption.lines for fid in ln.fact_ids}
    action_lines = [
        ln for ln in caption.lines if ln.section == CaptionSection.ACTION_AUDIO
    ]
    categories: list[GoldenGateCategoryResult] = []

    def add(category: str, status: GoldenGateStatus, detail: str) -> None:
        categories.append(
            GoldenGateCategoryResult(
                category=category,
                status=status,
                detail=detail,
                rule_ids=_rule_ids(rules, category),
            )
        )

    # DETAIL_COVERAGE
    if coverage.missing_required_fact_ids:
        add(
            "DETAIL_COVERAGE",
            GoldenGateStatus.FAIL,
            f"{len(coverage.missing_required_fact_ids)} material eligible fact(s) "
            "missing without a valid omission reason",
        )
    else:
        add("DETAIL_COVERAGE", GoldenGateStatus.PASS, "all material eligible facts represented")

    # EVENT_GRANULARITY: short eligible events must survive rendering.
    short_events = [
        f
        for f in eligible
        if f.fact_type
        in (CaptionFactType.VISUAL_ACTION, CaptionFactType.SPEECH, CaptionFactType.SOUND)
        and f.start_exact is not None
        and f.end_exact is not None
        and (f.end_exact - f.start_exact) <= Fraction(2, 10)
    ]
    dropped_short = [f for f in short_events if f.fact_id not in rendered_fact_ids]
    if dropped_short:
        add(
            "EVENT_GRANULARITY",
            GoldenGateStatus.FAIL,
            f"{len(dropped_short)} short (≤0.2 s) eligible event(s) dropped — the "
            "Golden standard keeps 0.1 s events",
        )
    else:
        add("EVENT_GRANULARITY", GoldenGateStatus.PASS, "short events retained; no quota applied")

    # TIMESTAMP_DISCIPLINE
    bad_stamps = [
        ln.line_id
        for ln in action_lines
        if ln.start_exact is not None
        and ln.display_start is not None
        and ln.display_start != format_manuscript_display(ln.start_exact)
    ]
    add(
        "TIMESTAMP_DISCIPLINE",
        GoldenGateStatus.FAIL if bad_stamps else GoldenGateStatus.PASS,
        (
            "all stamps are canonical projections"
            if not bad_stamps
            else f"non-canonical: {bad_stamps}"
        ),
    )

    # TRUTHFUL_OVERLAP: overlapping eligible events must each keep a line.
    overlap_violations = 0
    timed = [
        f
        for f in eligible
        if f.start_exact is not None
        and f.end_exact is not None
        and f.fact_type
        in (
            CaptionFactType.VISUAL_ACTION,
            CaptionFactType.SPEECH,
            CaptionFactType.SOUND,
            CaptionFactType.ON_SCREEN_TEXT,
        )
    ]
    for i, a in enumerate(timed):
        for b in timed[i + 1 :]:
            overlapping = (
                a.start_exact < b.end_exact and b.start_exact < a.end_exact  # type: ignore[operator]
            )
            if overlapping and (
                a.fact_id not in rendered_fact_ids or b.fact_id not in rendered_fact_ids
            ):
                overlap_violations += 1
    add(
        "TRUTHFUL_OVERLAP",
        GoldenGateStatus.FAIL if overlap_violations else GoldenGateStatus.PASS,
        "overlapping events preserved as separate entries"
        if not overlap_violations
        else f"{overlap_violations} overlapping event pair(s) lost a member",
    )

    # CHARACTER_CONTINUITY / OBJECT_CONTINUITY
    defined_c = {e.character_id for e in plan.overview_plan.characters}
    speech_missing_speaker = [
        f
        for f in eligible
        if f.fact_type == CaptionFactType.SPEECH
        and f.semantic_value.get("speaker_id") not in defined_c
    ]
    add(
        "CHARACTER_CONTINUITY",
        GoldenGateStatus.REVIEW_REQUIRED if speech_missing_speaker else GoldenGateStatus.PASS,
        "every speaker has a defined C ID"
        if not speech_missing_speaker
        else f"{len(speech_missing_speaker)} speech act(s) without a defined speaker C ID",
    )
    unresolved_final = [
        f
        for f in facts
        if f.fact_type == CaptionFactType.FINAL_OBJECT_STATE
        and f.eligibility == CaptionEligibility.REVIEW_REQUIRED
    ]
    add(
        "OBJECT_CONTINUITY",
        GoldenGateStatus.REVIEW_REQUIRED if unresolved_final else GoldenGateStatus.PASS,
        "object states reconciled"
        if not unresolved_final
        else f"{len(unresolved_final)} object final-state check(s) unresolved",
    )

    # CAMERA_SEPARATION (structural)
    camera_in_action = [
        ln.line_id
        for ln in action_lines
        for fid in ln.fact_ids
        if facts_by_id.get(fid) is not None
        and facts_by_id[fid].fact_type == CaptionFactType.CAMERA_MOVEMENT
    ]
    add(
        "CAMERA_SEPARATION",
        GoldenGateStatus.FAIL if camera_in_action else GoldenGateStatus.PASS,
        "camera movement independent of Action & Audio"
        if not camera_in_action
        else f"camera facts rendered as events: {camera_in_action}",
    )

    # DIALOGUE_COVERAGE: eligible speech rendered; blocked speech → review.
    speech_facts = [f for f in facts if f.fact_type == CaptionFactType.SPEECH]
    blocked_speech = [
        f
        for f in speech_facts
        if f.eligibility
        in (CaptionEligibility.INELIGIBLE, CaptionEligibility.REVIEW_REQUIRED)
    ]
    missing_speech = [
        f
        for f in speech_facts
        if f.eligibility == CaptionEligibility.ELIGIBLE
        and f.fact_id not in rendered_fact_ids
    ]
    if missing_speech:
        add("DIALOGUE_COVERAGE", GoldenGateStatus.FAIL, "verified dialogue missing from caption")
    elif blocked_speech:
        add(
            "DIALOGUE_COVERAGE",
            GoldenGateStatus.REVIEW_REQUIRED,
            f"{len(blocked_speech)} speech act(s) await source verification",
        )
    else:
        add("DIALOGUE_COVERAGE", GoldenGateStatus.PASS, "all dialogue verified and rendered")

    # AUDIO_COVERAGE
    audio_facts = [
        f
        for f in eligible
        if f.fact_type in (CaptionFactType.SOUND, CaptionFactType.OVERVIEW_AUDIO)
    ]
    missing_audio = [f for f in audio_facts if f.fact_id not in rendered_fact_ids]
    add(
        "AUDIO_COVERAGE",
        GoldenGateStatus.FAIL if missing_audio else GoldenGateStatus.PASS,
        "eligible audio facts rendered"
        if not missing_audio
        else f"{len(missing_audio)} eligible audio fact(s) missing",
    )

    # TEXT_COVERAGE
    text_facts = [f for f in eligible if f.fact_type == CaptionFactType.ON_SCREEN_TEXT]
    missing_text = [f for f in text_facts if f.fact_id not in rendered_fact_ids]
    add(
        "TEXT_COVERAGE",
        GoldenGateStatus.FAIL if missing_text else GoldenGateStatus.PASS,
        "verified on-screen text rendered"
        if not missing_text
        else f"{len(missing_text)} verified text track(s) missing",
    )

    # FINAL_STATE mirrors OBJECT_CONTINUITY at gate level.
    add(
        "FINAL_STATE",
        GoldenGateStatus.REVIEW_REQUIRED if unresolved_final else GoldenGateStatus.PASS,
        "final object states reconciled"
        if not unresolved_final
        else "unresolved final-state checks remain",
    )

    # SCENE_RECONSTRUCTABILITY: evidence readiness, never word count (§61).
    has_scene = bool(plan.overview_plan.scene_fact_ids)
    has_cast = bool(plan.overview_plan.characters)
    if has_scene and has_cast:
        add(
            "SCENE_RECONSTRUCTABILITY",
            GoldenGateStatus.PASS,
            "overview has eligible identity + layout facts",
        )
    else:
        add(
            "SCENE_RECONSTRUCTABILITY",
            GoldenGateStatus.REVIEW_REQUIRED,
            "overview lacks eligible scene and/or character facts; do not pad "
            "with invented detail",
        )

    # EXPORT_SAFETY from the platform-semantic pass.
    platform_status = platform_report.status
    add(
        "EXPORT_SAFETY",
        {
            "PASS": GoldenGateStatus.PASS,
            "REVIEW_REQUIRED": GoldenGateStatus.REVIEW_REQUIRED,
            "FAIL": GoldenGateStatus.FAIL,
        }[platform_status],
        f"platform-semantic pass: {platform_status}",
    )

    if any(c.status == GoldenGateStatus.FAIL for c in categories):
        overall = GoldenGateStatus.FAIL
    elif any(c.status == GoldenGateStatus.REVIEW_REQUIRED for c in categories):
        overall = GoldenGateStatus.REVIEW_REQUIRED
    else:
        overall = GoldenGateStatus.PASS
    return GoldenGateResult(
        rules_file=GOLDEN_RULES_FILE.name,
        rules_version=str(rules["golden_behavior_version"]),
        overall=overall,
        categories=categories,
    )
