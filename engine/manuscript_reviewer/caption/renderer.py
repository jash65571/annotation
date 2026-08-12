"""Deterministic caption renderer (§50/§55).

Consumes the CaptionPlan + fact graph and emits the Manuscript II structure.
It combines approved facts into sentences; it cannot invent adjectives,
objects, traits, C/O IDs, timing, speakers, quoted text, transitions, speeds
or actions. Every rendered assertion maps back to CaptionFact IDs; the
assertion map is produced structurally as the renderer writes each line.

No LLM composes final facts. A future CaptionWordingAdapter (§51) may propose
wording from caption-eligible facts only, but this deterministic renderer
remains sufficient for CI and is the only implemented path.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

from ..models.caption_brain import (
    AssertionStatus,
    CaptionAssertionRecord,
    CaptionFact,
    CaptionFactType,
    CaptionPlan,
    CaptionSection,
    LanguageRenderLevel,
    RenderedCaption,
    RenderedCaptionLine,
)
from ..rules.loader import load_rules


@dataclass
class RenderResult:
    caption: RenderedCaption
    assertions: list[CaptionAssertionRecord] = field(default_factory=list)
    #: fact_id -> line_id for every rendered fact (coverage input).
    rendered_fact_lines: dict[str, str] = field(default_factory=dict)


class _LineBuilder:
    def __init__(self) -> None:
        self.lines: list[RenderedCaptionLine] = []
        self.assertions: list[CaptionAssertionRecord] = []
        self.rendered: dict[str, str] = {}
        self._n = 0
        self._a = 0

    def add(
        self,
        section: CaptionSection,
        text: str,
        fact_ids: list[str] | None = None,
        shot_number: int | None = None,
        fact: CaptionFact | None = None,
        structural: bool = False,
    ) -> RenderedCaptionLine:
        self._n += 1
        line = RenderedCaptionLine(
            line_id=f"L-{self._n:04d}",
            section=section,
            shot_number=shot_number,
            text=text,
            display_start=fact.display_start if fact is not None else None,
            display_end=fact.display_end if fact is not None else None,
            start_exact=fact.start_exact if fact is not None else None,
            end_exact=fact.end_exact if fact is not None else None,
            fact_ids=list(fact_ids or []),
            structural=structural,
        )
        self.lines.append(line)
        for fid in line.fact_ids:
            self.rendered.setdefault(fid, line.line_id)
        if not structural:
            self._a += 1
            self.assertions.append(
                CaptionAssertionRecord(
                    assertion_id=f"A-{self._a:04d}",
                    line_id=line.line_id,
                    assertion_text=text,
                    fact_ids=list(line.fact_ids),
                    status=(
                        AssertionStatus.MAPPED if line.fact_ids else AssertionStatus.UNMAPPED
                    ),
                )
            )
        return line


def _sentence(text: str) -> str:
    """Terminal punctuation only; wording is never altered."""
    stripped = text.strip()
    if not stripped:
        return stripped
    if stripped[-1] in '.!?"”':
        return stripped
    return stripped + "."


def _join_descriptions(facts: list[CaptionFact]) -> str:
    return " ".join(_sentence(f.text_value) for f in facts if f.text_value)


def render_speech_text(fact: CaptionFact) -> str:
    """The speech ladder (§27-§32). Uncertainty was resolved structurally
    before rendering; this function never guesses a word or a language."""
    speaker = fact.semantic_value.get("speaker_id", "")
    off_screen = " off-screen" if fact.semantic_value.get("off_screen") == "true" else ""
    tone = fact.semantic_value.get("tone")
    tone_part = f" in {tone}" if tone else ""
    level = LanguageRenderLevel(
        fact.semantic_value.get("language_level", LanguageRenderLevel.VERBATIM.value)
    )
    if level == LanguageRenderLevel.INDISCERNIBLE:
        return f"{speaker} speaks{off_screen}; the words are indiscernible."
    if level == LanguageRenderLevel.FOREIGN_LANGUAGE:
        return f"{speaker} speaks{off_screen}{tone_part} in a foreign language."
    if level in (LanguageRenderLevel.NAMED_LANGUAGE, LanguageRenderLevel.LANGUAGE_FAMILY):
        language = fact.semantic_value.get("language_name", "")
        if fact.text_value:
            # Verified verbatim words in a named language: original language
            # is preserved, never translated (§30 rung 1).
            return f'{speaker} says{off_screen}{tone_part} in {language}, "{fact.text_value}"'
        return f"{speaker} speaks{off_screen}{tone_part} in {language}."
    text = fact.text_value or ""
    return f'{speaker} says{off_screen}{tone_part}, "{text}"'


#: Conservative wording for human-classified 2D camera motion. 2D global
#: motion never claims tracking/dolly/truck (§25); a human fact may supply
#: stronger verified wording via text_value.
_CAMERA_WORDING: dict[str, str] = {
    "HORIZONTAL_GLOBAL_MOTION": "The camera view moves {direction}.",
    "VERTICAL_GLOBAL_MOTION": "The camera view moves {direction}.",
    "DIAGONAL_GLOBAL_MOTION": "The camera view moves {direction}.",
    "SCALE_INCREASE": "The framing tightens on the scene.",
    "SCALE_DECREASE": "The framing widens on the scene.",
    "ROTATION": "The camera view rotates.",
    "HANDHELD_DRIFT": "The camera drifts with handheld movement.",
    "HANDHELD_SHAKE": "The camera shakes with handheld movement.",
    "STATIC": "The camera holds static.",
}


def render_camera_movement_text(fact: CaptionFact) -> str:
    if fact.text_value:
        return _sentence(fact.text_value)
    motion_class = fact.semantic_value.get("motion_class", "")
    direction = fact.semantic_value.get("direction", "")
    template = _CAMERA_WORDING.get(motion_class, "The camera view changes.")
    return template.format(direction=direction or "across the frame")


def render_event_text(fact: CaptionFact) -> str:
    if fact.fact_type == CaptionFactType.SPEECH:
        return render_speech_text(fact)
    if fact.fact_type == CaptionFactType.ON_SCREEN_TEXT:
        text = (fact.text_value or "").replace("\n", " / ")
        return f'On-screen text reads "{text}"'
    return _sentence(fact.text_value or "")


def render_caption(plan: CaptionPlan, facts_by_id: dict[str, CaptionFact]) -> RenderResult:
    rules = load_rules()
    none_literal = str(rules.get("fields.concerns_none_literal", "None."))
    builder = _LineBuilder()

    if plan.video_id:
        media_fact_ids = [
            f.fact_id
            for f in facts_by_id.values()
            if f.fact_type == CaptionFactType.MEDIA and f.text_value == plan.video_id
        ]
        builder.add(CaptionSection.VIDEO_ID, plan.video_id, media_fact_ids)

    overview = plan.overview_plan
    for entry in overview.characters:
        facts = [facts_by_id[fid] for fid in entry.description_fact_ids if fid in facts_by_id]
        description = _join_descriptions(facts)
        if description:
            builder.add(
                CaptionSection.CAST,
                f"{entry.character_id}: {description}",
                entry.description_fact_ids,
            )
    for obj in overview.objects:
        facts = [facts_by_id[fid] for fid in obj.description_fact_ids if fid in facts_by_id]
        description = _join_descriptions(facts)
        if description:
            builder.add(
                CaptionSection.OBJECTS,
                f"{obj.object_id}: {description}",
                obj.description_fact_ids,
            )

    _render_fact_section(builder, CaptionSection.SCENE, overview.scene_fact_ids, facts_by_id)
    _render_fact_section(builder, CaptionSection.STYLE, overview.style_fact_ids, facts_by_id)
    _render_fact_section(
        builder, CaptionSection.OVERVIEW_AUDIO, overview.overview_audio_fact_ids, facts_by_id
    )
    _render_concerns(
        builder,
        CaptionSection.VISUAL_CONCERNS,
        overview.visual_concern_fact_ids,
        facts_by_id,
        none_literal,
    )
    _render_concerns(
        builder,
        CaptionSection.AUDIO_CONCERNS,
        overview.audio_concern_fact_ids,
        facts_by_id,
        none_literal,
    )

    for shot in plan.shot_plans:
        builder.add(
            CaptionSection.SHOT_HEADER,
            # The en dash is the Manuscript II shot-header range separator.
            f"[Shot {shot.shot_number}: {shot.display_start}–{shot.display_end}]",  # noqa: RUF001
            [
                f.fact_id
                for f in facts_by_id.values()
                if f.fact_type == CaptionFactType.SHOT_BOUNDARY
                and f.shot_number == shot.shot_number
            ],
            shot_number=shot.shot_number,
        )
        if shot.transition_resolved and shot.transition_fact_id is not None:
            transition = facts_by_id[shot.transition_fact_id]
            builder.add(
                CaptionSection.CUT,
                transition.text_value or "",
                [transition.fact_id],
                shot_number=shot.shot_number,
            )
        for fid in shot.camera_framing_fact_ids:
            fact = facts_by_id[fid]
            builder.add(
                CaptionSection.CAMERA,
                _sentence(fact.text_value or ""),
                [fid],
                shot_number=shot.shot_number,
            )
        for fid in shot.camera_movement_fact_ids:
            fact = facts_by_id[fid]
            builder.add(
                CaptionSection.CAMERA_MOVEMENTS,
                f"[{fact.display_start}-{fact.display_end}] "
                + render_camera_movement_text(fact),
                [fid],
                shot_number=shot.shot_number,
                fact=fact,
            )
        for fid in shot.scene_fact_ids:
            fact = facts_by_id[fid]
            builder.add(
                CaptionSection.SHOT_SCENE,
                _sentence(fact.text_value or ""),
                [fid],
                shot_number=shot.shot_number,
            )
        for fid in shot.event_fact_ids:
            fact = facts_by_id[fid]
            builder.add(
                CaptionSection.ACTION_AUDIO,
                f"[{fact.display_start}-{fact.display_end}] " + render_event_text(fact),
                [fid],
                shot_number=shot.shot_number,
                fact=fact,
            )
        if shot.playback_speed_resolved and shot.playback_speed_fact_id is not None:
            fact = facts_by_id[shot.playback_speed_fact_id]
            builder.add(
                CaptionSection.PLAYBACK_SPEED,
                f"Video playback speed: {fact.text_value}",
                [fact.fact_id],
                shot_number=shot.shot_number,
            )
        for fid in shot.speed_change_fact_ids:
            fact = facts_by_id[fid]
            builder.add(
                CaptionSection.SPEED_CHANGES,
                f"[{fact.display_start}-{fact.display_end}] "
                + _sentence(fact.text_value or ""),
                [fid],
                shot_number=shot.shot_number,
                fact=fact,
            )

    markdown = _assemble_markdown(plan, builder.lines)
    caption = RenderedCaption(
        video_id=plan.video_id,
        lines=builder.lines,
        markdown=markdown,
        caption_sha256=hashlib.sha256(markdown.encode("utf-8")).hexdigest(),
    )
    return RenderResult(
        caption=caption, assertions=builder.assertions, rendered_fact_lines=builder.rendered
    )


def _render_fact_section(
    builder: _LineBuilder,
    section: CaptionSection,
    fact_ids: list[str],
    facts_by_id: dict[str, CaptionFact],
) -> None:
    for fid in fact_ids:
        fact = facts_by_id.get(fid)
        if fact is not None and fact.text_value:
            builder.add(section, _sentence(fact.text_value), [fid])


def _render_concerns(
    builder: _LineBuilder,
    section: CaptionSection,
    fact_ids: list[str],
    facts_by_id: dict[str, CaptionFact],
    none_literal: str,
) -> None:
    if not fact_ids:
        builder.add(section, none_literal, structural=True)
        return
    _render_fact_section(builder, section, fact_ids, facts_by_id)


_SECTION_HEADINGS: dict[CaptionSection, str] = {
    CaptionSection.CAST: "CHARACTERS",
    CaptionSection.OBJECTS: "OBJECTS",
    CaptionSection.SCENE: "SCENE",
    CaptionSection.STYLE: "STYLE",
    CaptionSection.OVERVIEW_AUDIO: "AUDIO",
    CaptionSection.VISUAL_CONCERNS: "VISUAL CONCERNS",
    CaptionSection.AUDIO_CONCERNS: "AUDIO CONCERNS",
}

_SHOT_FIELD_HEADINGS: dict[CaptionSection, str] = {
    CaptionSection.CUT: "CUT",
    CaptionSection.CAMERA: "CAMERA",
    CaptionSection.CAMERA_MOVEMENTS: "CAMERA MOVEMENTS",
    CaptionSection.SHOT_SCENE: "SCENE",
    CaptionSection.ACTION_AUDIO: "ACTION & AUDIO",
    CaptionSection.SPEED_CHANGES: "SPEED CHANGES",
}


def _assemble_markdown(plan: CaptionPlan, lines: list[RenderedCaptionLine]) -> str:
    """Current live-tool layout (§55): video id, [Overview] sections, then one
    block per verified shot. Deterministic; identical inputs → identical bytes."""
    out: list[str] = []
    video_lines = [ln for ln in lines if ln.section == CaptionSection.VIDEO_ID]
    for ln in video_lines:
        out.append(ln.text)
        out.append("")
    out.append("[Overview]")
    for section in (
        CaptionSection.CAST,
        CaptionSection.OBJECTS,
        CaptionSection.SCENE,
        CaptionSection.STYLE,
        CaptionSection.OVERVIEW_AUDIO,
        CaptionSection.VISUAL_CONCERNS,
        CaptionSection.AUDIO_CONCERNS,
    ):
        section_lines = [ln for ln in lines if ln.section == section]
        if not section_lines:
            continue
        out.append("")
        out.append(_SECTION_HEADINGS[section])
        out.extend(ln.text for ln in section_lines)

    for shot in plan.shot_plans:
        header = next(
            (
                ln
                for ln in lines
                if ln.section == CaptionSection.SHOT_HEADER
                and ln.shot_number == shot.shot_number
            ),
            None,
        )
        out.append("")
        if header is not None:
            out.append(header.text)
        for section in (
            CaptionSection.CUT,
            CaptionSection.CAMERA,
            CaptionSection.CAMERA_MOVEMENTS,
            CaptionSection.SHOT_SCENE,
            CaptionSection.ACTION_AUDIO,
        ):
            section_lines = [
                ln
                for ln in lines
                if ln.section == section and ln.shot_number == shot.shot_number
            ]
            if not section_lines:
                continue
            out.append("")
            out.append(_SHOT_FIELD_HEADINGS[section])
            out.extend(ln.text for ln in section_lines)
        speed_lines = [
            ln
            for ln in lines
            if ln.section == CaptionSection.PLAYBACK_SPEED
            and ln.shot_number == shot.shot_number
        ]
        for ln in speed_lines:
            out.append("")
            out.append(ln.text)
        change_lines = [
            ln
            for ln in lines
            if ln.section == CaptionSection.SPEED_CHANGES
            and ln.shot_number == shot.shot_number
        ]
        if change_lines:
            out.append("")
            out.append(_SHOT_FIELD_HEADINGS[CaptionSection.SPEED_CHANGES])
            out.extend(ln.text for ln in change_lines)
    return "\n".join(out) + "\n"
