"""M2 caption validator (§62-§77): versioned rule groups over the rendered
caption + plan + fact graph.

Severity mapping: FAIL findings force BLOCKED; WARN findings force
REVIEW_REQUIRED. The validator is a machine pre-check — it never replaces the
platform-semantic pass or the manual final review (rules
``validation_scope.machine_validator_replaces_platform_validation: false``).
"""

from __future__ import annotations

import itertools
import re
from dataclasses import dataclass, field
from fractions import Fraction

from ..caption.coverage import AssertionCheckResult, CoverageResult
from ..caption.textcheck import find_quote_spans, pronoun_hits, strip_quotes
from ..media.timestamps import format_manuscript_display
from ..models.caption_brain import (
    CaptionFact,
    CaptionFactType,
    CaptionPlan,
    CaptionSection,
    RenderedCaption,
    RenderedCaptionLine,
)
from ..models.shot_truth import ShotTruthResult
from ..models.validation import Severity, ValidatorIssue
from ..rules.loader import load_rules

M2_VALIDATOR_VERSION = "1.0.0"

_ID_REF = re.compile(r"\b([CO]\d+)\b")
_CAMERA_MOVEMENT_WORDS = re.compile(
    r"\b(pans?|tilts?|zooms?|dolly|dollies|trucks?|tracking shot|push-in|pushes in|"
    r"pull-back|pulls back|whip pan|camera (?:moves|movement|rotates|drifts|shakes))\b",
    re.IGNORECASE,
)
_DYNAMIC_ACTION_WORDS = re.compile(
    r"\b(walks|runs|jumps|throws|grabs|picks up|puts down|raises|lowers|opens|closes|"
    r"pours|turns around|enters|exits|drops|swings)\b",
    re.IGNORECASE,
)
_REVIEWER_NOTE_WORDS = re.compile(
    r"\b(needs verification|seed says|evaluator|validator|review required|provisional|"
    r"review later|to be confirmed|TODO)\b",
    re.IGNORECASE,
)
_BARE_LEFT_RIGHT = re.compile(
    r"\b(?<!screen-)(left|right)\b(?!\s*(hand|arm|foot|leg|shoulder|side|eye|ear|knee))",
    re.IGNORECASE,
)
_SPEECH_VERB = re.compile(r"\b(says|speaks|shouts|whispers|sings)\b", re.IGNORECASE)


@dataclass
class M2Inputs:
    plan: CaptionPlan
    caption: RenderedCaption
    facts_by_id: dict[str, CaptionFact]
    shot_truth: ShotTruthResult | None = None
    expected_video_id: str | None = None
    coverage: CoverageResult | None = None
    assertions: AssertionCheckResult | None = None
    #: Annotation-clock endpoint the final shot must end on.
    annotation_endpoint: Fraction | None = None
    unresolved_high_feedback: list[str] = field(default_factory=list)


def validate_caption(inputs: M2Inputs) -> list[ValidatorIssue]:
    issues: list[ValidatorIssue] = []
    issues += _media_rules(inputs)
    issues += _struct_rules(inputs)
    issues += _transition_rules(inputs)
    issues += _cast_object_rules(inputs)
    issues += _time_rules(inputs)
    issues += _action_rules(inputs)
    issues += _speech_rules(inputs)
    issues += _audio_rules(inputs)
    issues += _text_rules(inputs)
    issues += _camera_rules(inputs)
    issues += _speed_rules(inputs)
    issues += _field_rules(inputs)
    issues += _source_rules(inputs)
    return issues


def _issue(
    rule_id: str, severity: Severity, location: str, message: str, fix: str | None = None
) -> ValidatorIssue:
    return ValidatorIssue(
        rule_id=rule_id, severity=severity, location=location, message=message,
        suggested_fix=fix,
    )


def _caption_lines(inputs: M2Inputs) -> list[RenderedCaptionLine]:
    return [ln for ln in inputs.caption.lines if not ln.structural]


def _action_lines(inputs: M2Inputs) -> list[RenderedCaptionLine]:
    return [
        ln for ln in inputs.caption.lines if ln.section == CaptionSection.ACTION_AUDIO
    ]


# --- M2-MEDIA (§63) --------------------------------------------------------


def _media_rules(inputs: M2Inputs) -> list[ValidatorIssue]:
    issues: list[ValidatorIssue] = []
    if inputs.expected_video_id is not None:
        if inputs.caption.video_id != inputs.expected_video_id:
            issues.append(
                _issue(
                    "M2-MEDIA-001",
                    Severity.FAIL,
                    "video_id",
                    f"Caption video id {inputs.caption.video_id!r} does not match the "
                    f"verified video id {inputs.expected_video_id!r} — wrong-video "
                    "captions are an auto-reject.",
                )
            )
    elif inputs.caption.video_id is None:
        issues.append(
            _issue(
                "M2-MEDIA-002", Severity.WARN, "video_id",
                "No video id in the caption; the exact video id must open the caption.",
            )
        )
    return issues


# --- M2-STRUCT (§64) -------------------------------------------------------


def _struct_rules(inputs: M2Inputs) -> list[ValidatorIssue]:
    issues: list[ValidatorIssue] = []
    if inputs.shot_truth is None:
        return issues
    verified = {s.shot_index for s in inputs.shot_truth.shots}
    planned = [sp.shot_number for sp in inputs.plan.shot_plans]
    if set(planned) - verified:
        issues.append(
            _issue(
                "M2-STRUCT-001", Severity.FAIL, "shots",
                f"Invented shots not in Shot Truth: {sorted(set(planned) - verified)}",
            )
        )
    if verified - set(planned):
        issues.append(
            _issue(
                "M2-STRUCT-002", Severity.FAIL, "shots",
                f"Verified shots missing from the caption: {sorted(verified - set(planned))}",
            )
        )
    if planned != sorted(planned):
        issues.append(
            _issue("M2-STRUCT-003", Severity.FAIL, "shots", "Shot order is incorrect.")
        )
    plans = inputs.plan.shot_plans
    for prev, nxt in itertools.pairwise(plans):
        if prev.end_exact != nxt.start_exact:
            issues.append(
                _issue(
                    "M2-STRUCT-004", Severity.FAIL,
                    f"shot {prev.shot_number}->{nxt.shot_number}",
                    "Illegal gap/overlap: adjacent shot intervals must meet exactly.",
                )
            )
    if plans and inputs.annotation_endpoint is not None:
        final = plans[-1]
        if final.end_exact != inputs.annotation_endpoint:
            issues.append(
                _issue(
                    "M2-STRUCT-005", Severity.FAIL, f"shot {final.shot_number}",
                    "Final shot does not end at the canonical annotation endpoint.",
                )
            )
    return issues


# --- M2-TRANSITION (§65) ---------------------------------------------------


def _transition_rules(inputs: M2Inputs) -> list[ValidatorIssue]:
    issues: list[ValidatorIssue] = []
    rules = load_rules()
    allowed = set(rules.get("shots.allowed_transition_types", []))
    opening = str(rules.get("shots.shot_one_transition", "Opening shot"))
    for shot in inputs.plan.shot_plans:
        if not shot.transition_resolved:
            issues.append(
                _issue(
                    "M2-TRANSITION-001", Severity.WARN, f"shot {shot.shot_number}",
                    "Unresolved transition blocks ready status; it is never "
                    "defaulted to Hard cut.",
                )
            )
            continue
        fact = inputs.facts_by_id.get(shot.transition_fact_id or "")
        label = fact.text_value if fact is not None else None
        if label not in allowed:
            issues.append(
                _issue(
                    "M2-TRANSITION-002", Severity.FAIL, f"shot {shot.shot_number}",
                    f"Transition {label!r} is not in the rule-file menu.",
                )
            )
            continue
        if shot.shot_number == 1 and label != opening:
            issues.append(
                _issue(
                    "M2-TRANSITION-003", Severity.FAIL, "shot 1",
                    f"Shot 1 must be {opening!r}.",
                )
            )
        if shot.shot_number > 1 and label == opening:
            issues.append(
                _issue(
                    "M2-TRANSITION-004", Severity.FAIL, f"shot {shot.shot_number}",
                    f"{opening!r} may appear nowhere except Shot 1.",
                )
            )
    return issues


# --- M2-CAST / M2-OBJECT (§66/§67) + visibility consistency (§81) ----------

_LOWER_BODY_SENTENCE = "Lower body and shoes are not visible."
_LOWER_BODY_WORDS = re.compile(
    r"\b(trousers|pants|skirt|shorts|shoes|footwear|jeans|sneakers|boots|socks)\b",
    re.IGNORECASE,
)


def _cast_object_rules(inputs: M2Inputs) -> list[ValidatorIssue]:
    issues: list[ValidatorIssue] = []
    rules = load_rules()
    blocklist = [str(p) for p in rules.get("characters.pronoun_blocklist", [])]

    defined_c = [e.character_id for e in inputs.plan.overview_plan.characters]
    defined_o = [e.object_id for e in inputs.plan.overview_plan.objects]
    referenced: set[str] = set()
    for line in inputs.caption.lines:
        if line.section in (CaptionSection.CAST, CaptionSection.OBJECTS):
            continue
        referenced.update(_ID_REF.findall(line.text))

    for cid in sorted(referenced):
        if cid.startswith("C") and cid not in defined_c:
            issues.append(
                _issue(
                    "M2-CAST-001", Severity.FAIL, cid,
                    f"Undefined character reference {cid} (ghost reference).",
                )
            )
        if cid.startswith("O") and cid not in defined_o:
            issues.append(
                _issue(
                    "M2-OBJECT-001", Severity.FAIL, cid,
                    f"Undefined object reference {cid} (ghost reference).",
                )
            )
    for cid in defined_c:
        if cid not in referenced:
            issues.append(
                _issue(
                    "M2-CAST-002", Severity.WARN, cid,
                    f"{cid} is defined but never referenced (ghost character).",
                )
            )
    for oid in defined_o:
        if oid not in referenced:
            issues.append(
                _issue(
                    "M2-OBJECT-002", Severity.WARN, oid,
                    f"{oid} is defined but never referenced (ghost object).",
                )
            )

    expected = [f"C{i}" for i in range(1, len(defined_c) + 1)]
    if defined_c != expected:
        issues.append(
            _issue(
                "M2-CAST-003", Severity.WARN, "cast",
                f"Character ids {defined_c} are not C1..C{len(defined_c)} in "
                "first-appearance order.",
            )
        )

    for line in _caption_lines(inputs):
        if line.section == CaptionSection.VIDEO_ID:
            continue
        hits = pronoun_hits(line.text, blocklist)
        if hits:
            issues.append(
                _issue(
                    "M2-CAST-004", Severity.FAIL, line.line_id,
                    f"Pronoun(s) outside quoted dialogue: {', '.join(sorted(set(hits)))}. "
                    "Use C# ids.",
                )
            )

    for line in inputs.caption.lines:
        if line.section != CaptionSection.CAST:
            continue
        if _LOWER_BODY_SENTENCE.lower() in line.text.lower():
            stripped = line.text.replace(_LOWER_BODY_SENTENCE, "")
            match = _LOWER_BODY_WORDS.search(stripped)
            if match:
                issues.append(
                    _issue(
                        "M2-CAST-005", Severity.FAIL, line.line_id,
                        f"Visibility contradiction: description says lower body is "
                        f"not visible but also mentions {match.group(0)!r}.",
                    )
                )
    return issues


# --- M2-TIME (§69, §46) ----------------------------------------------------


def _time_rules(inputs: M2Inputs) -> list[ValidatorIssue]:
    issues: list[ValidatorIssue] = []
    shot_bounds = {
        sp.shot_number: (sp.start_exact, sp.end_exact) for sp in inputs.plan.shot_plans
    }
    seen_pairs: dict[tuple[int, str, str], list[RenderedCaptionLine]] = {}
    for line in _action_lines(inputs):
        if line.start_exact is None or line.end_exact is None:
            issues.append(
                _issue(
                    "M2-TIME-001", Severity.FAIL, line.line_id,
                    "Action & Audio line lacks exact source timing.",
                )
            )
            continue
        if line.start_exact > line.end_exact:
            issues.append(
                _issue("M2-TIME-002", Severity.FAIL, line.line_id, "start > end.")
            )
        # Display honesty: fake nudging is impossible because display MUST be
        # the canonical ROUND_HALF_UP projection of the exact time.
        for exact, display in (
            (line.start_exact, line.display_start),
            (line.end_exact, line.display_end),
        ):
            if display is not None and display != format_manuscript_display(exact):
                issues.append(
                    _issue(
                        "M2-TIME-003", Severity.FAIL, line.line_id,
                        f"Display {display} is not the canonical projection of the "
                        "exact time — arbitrary timestamp nudges are forbidden.",
                    )
                )
        bounds = shot_bounds.get(line.shot_number or -1)
        if bounds is not None and (
            line.start_exact < bounds[0] or line.end_exact > bounds[1]
        ):
            issues.append(
                _issue(
                    "M2-TIME-004", Severity.FAIL, line.line_id,
                    "Timestamps fall outside the shot window.",
                )
            )
        if line.shot_number is not None and line.display_start and line.display_end:
            key = (line.shot_number, line.display_start, line.display_end)
            seen_pairs.setdefault(key, []).append(line)

    for (shot_number, d_start, d_end), lines in seen_pairs.items():
        if len(lines) < 2:
            continue
        exact_ranges = {(ln.start_exact, ln.end_exact) for ln in lines}
        if len(exact_ranges) > 1:
            issues.append(
                _issue(
                    "M2-TIME-COLLISION", Severity.WARN, f"shot {shot_number}",
                    f"{len(lines)} events display the same 0.1 s pair "
                    f"({d_start}-{d_end}) with different exact ranges. Never "
                    "nudge timestamps; merge only if truly inseparable, refine "
                    "boundaries only with evidence, else human review.",
                )
            )
        else:
            issues.append(
                _issue(
                    "M2-TIME-005", Severity.WARN, f"shot {shot_number}",
                    f"Two entries share the identical exact window ({d_start}-{d_end}); "
                    "merge only if they are one true event.",
                )
            )
    return issues


# --- M2-ACTION (§70) -------------------------------------------------------


def _action_rules(inputs: M2Inputs) -> list[ValidatorIssue]:
    issues: list[ValidatorIssue] = []
    for line in _action_lines(inputs):
        if len(line.fact_ids) != 1:
            issues.append(
                _issue(
                    "M2-ACTION-001", Severity.FAIL, line.line_id,
                    "An Action & Audio line must carry exactly one defensible "
                    f"event/source (has {len(line.fact_ids)}).",
                )
            )
        if _CAMERA_MOVEMENT_WORDS.search(line.text):
            issues.append(
                _issue(
                    "M2-ACTION-002", Severity.FAIL, line.line_id,
                    "Camera movement inside Action & Audio; it belongs in "
                    "Camera Movements.",
                )
            )
        for fid in line.fact_ids:
            fact = inputs.facts_by_id.get(fid)
            if fact is not None and fact.fact_type == CaptionFactType.CAMERA_MOVEMENT:
                issues.append(
                    _issue(
                        "M2-ACTION-003", Severity.FAIL, line.line_id,
                        "A camera fact is structurally excluded from the Action & "
                        "Audio event union.",
                    )
                )
    return issues


# --- M2-SPEECH (§71) -------------------------------------------------------


def _speech_rules(inputs: M2Inputs) -> list[ValidatorIssue]:
    issues: list[ValidatorIssue] = []
    for line in inputs.caption.lines:
        if "<unintelligible>" in line.text:
            issues.append(
                _issue(
                    "M2-SPEECH-001", Severity.FAIL, line.line_id,
                    "Deprecated <unintelligible> token; the current token is "
                    "[inaudible].",
                )
            )
    for line in _action_lines(inputs):
        fact = inputs.facts_by_id.get(line.fact_ids[0]) if line.fact_ids else None
        if fact is None or fact.fact_type != CaptionFactType.SPEECH:
            continue
        speaker = fact.semantic_value.get("speaker_id")
        if not speaker:
            issues.append(
                _issue(
                    "M2-SPEECH-002", Severity.FAIL, line.line_id,
                    "Speech line without a verified speaker C-ID.",
                )
            )
        if fact.text_value == "[inaudible]":
            issues.append(
                _issue(
                    "M2-SPEECH-003", Severity.WARN, line.line_id,
                    'A fully-inaudible quote ("[inaudible]") should instead be an '
                    "indiscernible-speech description.",
                )
            )
        if "continues speaking about" in line.text.lower():
            issues.append(
                _issue(
                    "M2-SPEECH-004", Severity.FAIL, line.line_id,
                    "Vague dialogue filler is forbidden; transcribe verbatim.",
                )
            )
    return issues


# --- M2-AUDIO (§72) --------------------------------------------------------


def _audio_rules(inputs: M2Inputs) -> list[ValidatorIssue]:
    issues: list[ValidatorIssue] = []
    rules = load_rules()
    fillers = [str(f) for f in rules.get("audio.empty_audio_fillers_forbidden", [])]
    for line in _caption_lines(inputs):
        for filler in fillers:
            if line.text.strip().rstrip(".").lower() == filler.rstrip(".").lower():
                issues.append(
                    _issue(
                        "M2-AUDIO-001", Severity.FAIL, line.line_id,
                        f"Empty-audio filler {filler!r} is forbidden.",
                    )
                )
    for line in inputs.caption.lines:
        if line.section == CaptionSection.OVERVIEW_AUDIO and _SPEECH_VERB.search(
            line.text
        ):
            quoted = find_quote_spans(line.text)
            if quoted:
                issues.append(
                    _issue(
                        "M2-AUDIO-002", Severity.FAIL, line.line_id,
                        "Quoted speech in Overview Audio; speech lives in shots.",
                    )
                )
        if line.section == CaptionSection.AUDIO_CONCERNS and re.search(
            r"\btranscripts?\b", line.text, re.IGNORECASE
        ):
            issues.append(
                _issue(
                    "M2-AUDIO-003", Severity.FAIL, line.line_id,
                    'The word "transcript" is forbidden inside Audio concerns.',
                )
            )
    return issues


# --- M2-TEXT (§73) ---------------------------------------------------------


def _text_rules(inputs: M2Inputs) -> list[ValidatorIssue]:
    issues: list[ValidatorIssue] = []
    for line in _action_lines(inputs):
        fact = inputs.facts_by_id.get(line.fact_ids[0]) if line.fact_ids else None
        if fact is None or fact.fact_type != CaptionFactType.ON_SCREEN_TEXT:
            continue
        spans = find_quote_spans(line.text)
        if len(spans) != 1:
            issues.append(
                _issue(
                    "M2-TEXT-001", Severity.FAIL, line.line_id,
                    "On-screen text must be one quoted string (simultaneous lines "
                    "combine into one quote; different-time overlays are separate "
                    "entries).",
                )
            )
    return issues


# --- M2-CAMERA (§74) -------------------------------------------------------


def _camera_rules(inputs: M2Inputs) -> list[ValidatorIssue]:
    issues: list[ValidatorIssue] = []
    for line in inputs.caption.lines:
        if line.section == CaptionSection.CAMERA and _CAMERA_MOVEMENT_WORDS.search(
            line.text
        ):
            issues.append(
                _issue(
                    "M2-CAMERA-001", Severity.FAIL, line.line_id,
                    "The Camera field carries framing/angle only; movement goes to "
                    "Camera Movements.",
                )
            )
        if line.section in (
            CaptionSection.SCENE,
            CaptionSection.SHOT_SCENE,
            CaptionSection.STYLE,
        ) and _CAMERA_MOVEMENT_WORDS.search(line.text):
            issues.append(
                _issue(
                    "M2-CAMERA-002", Severity.FAIL, line.line_id,
                    "Camera movement inside Scene/Style.",
                )
            )
    return issues


# --- M2-SPEED (§75) --------------------------------------------------------


def _speed_rules(inputs: M2Inputs) -> list[ValidatorIssue]:
    issues: list[ValidatorIssue] = []
    rules = load_rules()
    allowed = {str(v) for v in rules.get("playback_speed.allowed_values", [])}
    for shot in inputs.plan.shot_plans:
        if not shot.playback_speed_resolved:
            issues.append(
                _issue(
                    "M2-SPEED-001", Severity.WARN, f"shot {shot.shot_number}",
                    "Playback speed is unverified: candidates "
                    "(REGULAR/SLOW_MOTION/ACCELERATED_CANDIDATE) are never final.",
                )
            )
            continue
        fact = inputs.facts_by_id.get(shot.playback_speed_fact_id or "")
        if fact is None or fact.text_value not in allowed:
            issues.append(
                _issue(
                    "M2-SPEED-002", Severity.FAIL, f"shot {shot.shot_number}",
                    f"Playback speed {fact.text_value if fact else None!r} is not an "
                    "allowed value.",
                )
            )
    return issues


# --- M2-FIELD placement (§76/§77) + positioning (§68) ----------------------


def _field_rules(inputs: M2Inputs) -> list[ValidatorIssue]:
    issues: list[ValidatorIssue] = []
    for line in _caption_lines(inputs):
        if _REVIEWER_NOTE_WORDS.search(line.text):
            issues.append(
                _issue(
                    "M2-FIELD-001", Severity.FAIL, line.line_id,
                    "Reviewer rationale/process text inside a caption field.",
                )
            )
        if line.section in (CaptionSection.SCENE, CaptionSection.STYLE) and (
            _DYNAMIC_ACTION_WORDS.search(line.text)
        ):
            issues.append(
                _issue(
                    "M2-FIELD-002", Severity.FAIL, line.line_id,
                    "Dynamic action inside Scene/Style; it belongs in Action & Audio.",
                )
            )
        if line.section in (
            CaptionSection.ACTION_AUDIO,
            CaptionSection.SHOT_SCENE,
            CaptionSection.SCENE,
        ) and _BARE_LEFT_RIGHT.search(strip_quotes(line.text)):
            issues.append(
                _issue(
                    "M2-FIELD-003", Severity.WARN, line.line_id,
                    "Bare left/right without perspective; use screen-left / "
                    "screen-right (anatomical sides need body-side evidence).",
                )
            )
    return issues


# --- M2-SOURCE (§60 gate surfaced as validator rules) ----------------------


def _source_rules(inputs: M2Inputs) -> list[ValidatorIssue]:
    issues: list[ValidatorIssue] = []
    if inputs.assertions is not None:
        for record in inputs.assertions.unmapped:
            issues.append(
                _issue(
                    "M2-SOURCE-001", Severity.FAIL, record.line_id,
                    "Caption assertion with no CaptionFact source (hallucination gate).",
                )
            )
        for assertion_id, fid in inputs.assertions.ineligible_refs:
            issues.append(
                _issue(
                    "M2-SOURCE-002", Severity.FAIL, assertion_id,
                    f"Caption assertion references non-eligible fact {fid}.",
                )
            )
    if inputs.coverage is not None:
        for fid in inputs.coverage.missing_required_fact_ids:
            issues.append(
                _issue(
                    "M2-SOURCE-003", Severity.FAIL, fid,
                    "Material eligible fact missing from the caption with no valid "
                    "omission reason (omission gate).",
                )
            )
    for directive_id in inputs.unresolved_high_feedback:
        issues.append(
            _issue(
                "M2-SOURCE-004", Severity.WARN, directive_id,
                "Unresolved HIGH task-feedback directive blocks ready status.",
            )
        )
    return issues
