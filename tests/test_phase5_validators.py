"""Phase 5 M2 + platform-semantic validator tests (§101/§102/§107/§108/§109/
§111/§112/§114) including timing and NTSC collision regressions."""

from __future__ import annotations

from fractions import Fraction

from manuscript_reviewer.caption.facts import build_fact_graph
from manuscript_reviewer.caption.planning import PlanInputs, build_caption_plan
from manuscript_reviewer.caption.renderer import render_caption
from manuscript_reviewer.media.timestamps import to_manuscript_display
from manuscript_reviewer.models.caption_brain import (
    CaptionFactType,
    CaptionSection,
    HumanCaptionFact,
    RenderedCaption,
    RenderedCaptionLine,
)
from manuscript_reviewer.models.review_intelligence import SeedClaimType
from manuscript_reviewer.models.validation import Severity
from manuscript_reviewer.validation.caption_validator import M2Inputs, validate_caption
from manuscript_reviewer.validation.platform_semantic_validator import (
    validate_platform_semantics,
)

from .phase5_helpers import (
    RULES_VERSION,
    VIDEO_SHA,
    base_inputs,
    make_shot,
    make_shot_truth,
    supported_claim,
)


def _hf(
    fact_id: str,
    text: str,
    start: Fraction,
    end: Fraction,
    fact_type: CaptionFactType = CaptionFactType.VISUAL_ACTION,
    shot_number: int = 1,
    character_ids: list[str] | None = None,
) -> HumanCaptionFact:
    return HumanCaptionFact(
        fact_id=fact_id,
        fact_type=fact_type,
        text_value=text,
        shot_number=shot_number,
        character_ids=character_ids or [],
        start_exact=start,
        end_exact=end,
        decided_by="reviewer@test",
        bound_video_sha256=VIDEO_SHA,
        bound_rules_version=RULES_VERSION,
    )


def _pipeline(human_facts: list[HumanCaptionFact], seed_claims: list[object] | None = None):
    shots = make_shot_truth([make_shot(1, Fraction(0), Fraction(2), "Opening shot")])
    claims = seed_claims if seed_claims is not None else [
        supported_claim("CLM-C1", SeedClaimType.CHARACTER_EXISTS,
                        "A person in a dark coat.", subject_ids=["C1"]),
    ]
    inputs = base_inputs(shot_truth=shots, seed_claims=claims, human_facts=human_facts)
    graph = build_fact_graph(inputs)
    plan = build_caption_plan(PlanInputs(graph=graph, shot_truth=shots))
    render = render_caption(plan, graph.by_id())
    return shots, graph, plan, render


def _m2(shots, graph, plan, render, video_id: str | None = None):
    return validate_caption(
        M2Inputs(
            plan=plan,
            caption=render.caption,
            facts_by_id=graph.by_id(),
            shot_truth=shots,
            expected_video_id=video_id,
            annotation_endpoint=Fraction(2),
        )
    )


# --- §101 timing / rounding ------------------------------------------------


def test_round_half_up_display_grid() -> None:
    cases = {
        Fraction(1, 20): "0.1",   # 0.05
        Fraction(3, 20): "0.2",   # 0.15
        Fraction(1, 4): "0.3",    # 0.25
        Fraction(9, 20): "0.5",   # 0.45
        Fraction(11, 20): "0.6",  # 0.55
    }
    for exact, expected in cases.items():
        assert str(to_manuscript_display(exact)) == expected


def test_display_projection_across_frame_rates() -> None:
    for num, den in ((24, 1), (24000, 1001), (30, 1), (30000, 1001), (60, 1), (60000, 1001)):
        rate = Fraction(num, den)
        for frame in (0, 7, 29, 59):
            exact = frame / rate
            display = to_manuscript_display(exact)
            # The projection is always within 0.05 s of the exact time.
            assert abs(Fraction(str(display)) - exact) <= Fraction(1, 20)


def test_non_zero_source_pts_uses_annotation_clock() -> None:
    from manuscript_reviewer.media.clock import AnnotationClock

    clock = AnnotationClock(origin=Fraction(5))
    assert clock.to_annotation(Fraction(6)) == Fraction(1)
    assert str(to_manuscript_display(clock.to_annotation(Fraction(6)))) == "1.0"


# --- §102 NTSC collision ---------------------------------------------------


def test_ntsc_display_collision_is_review_not_nudge() -> None:
    """Two real events with different 30000/1001 exact ranges that round to the
    same 0.1 pair → M2-TIME-COLLISION, never a fabricated offset."""
    base = Fraction(1001, 30000)
    a_start, a_end = 30 * base, 60 * base      # 1.001 .. 2.002 -> 1.0-2.0
    b_start, b_end = 30 * base + Fraction(1, 100), 60 * base - Fraction(1, 100)
    shots = make_shot_truth([make_shot(1, Fraction(0), Fraction(3), "Opening shot")])
    inputs = base_inputs(
        shot_truth=shots,
        seed_claims=[
            supported_claim("CLM-C1", SeedClaimType.CHARACTER_EXISTS,
                            "A person.", subject_ids=["C1"]),
        ],
        human_facts=[
            _hf("HF-A", "C1 raises the right hand.", a_start, a_end),
            _hf("HF-B", "C1 lowers the right hand.", b_start, b_end),
        ],
    )
    graph = build_fact_graph(inputs)
    plan = build_caption_plan(PlanInputs(graph=graph, shot_truth=shots))
    render = render_caption(plan, graph.by_id())
    issues = _m2(shots, graph, plan, render)
    collisions = [i for i in issues if i.rule_id == "M2-TIME-COLLISION"]
    assert len(collisions) == 1
    assert collisions[0].severity == Severity.WARN
    # Displays were NOT nudged: both lines keep the canonical projection.
    action_lines = [
        ln for ln in render.caption.lines if ln.section == CaptionSection.ACTION_AUDIO
    ]
    assert {ln.display_start for ln in action_lines} == {"1.0s"}


# --- §107 atomicity --------------------------------------------------------


def test_two_subject_actions_stay_separate_lines_with_overlap() -> None:
    _shots, graph, _plan, render = _pipeline(
        [
            _hf("HF-A", "C1 walks forward.", Fraction(0), Fraction(1)),
            _hf("HF-B", "C2 turns away.", Fraction(1, 2), Fraction(3, 2)),
        ],
        seed_claims=[
            supported_claim("CLM-C1", SeedClaimType.CHARACTER_EXISTS,
                            "First person.", subject_ids=["C1"]),
            supported_claim("CLM-C2", SeedClaimType.CHARACTER_EXISTS,
                            "Second person.", subject_ids=["C2"]),
        ],
    )
    action_lines = [
        ln for ln in render.caption.lines if ln.section == CaptionSection.ACTION_AUDIO
    ]
    assert len(action_lines) == 2  # truthful overlap, never flattened
    report = validate_platform_semantics(render.caption, graph.by_id())
    assert report.status == "PASS"


def test_hidden_second_action_is_caught() -> None:
    _shots, graph, _plan, render = _pipeline(
        [_hf("HF-A", "C1 raises O1 and C2 turns away.", Fraction(0), Fraction(1))],
        seed_claims=[
            supported_claim("CLM-C1", SeedClaimType.CHARACTER_EXISTS,
                            "First person.", subject_ids=["C1"]),
            supported_claim("CLM-C2", SeedClaimType.CHARACTER_EXISTS,
                            "Second person.", subject_ids=["C2"]),
            supported_claim("CLM-O1", SeedClaimType.OBJECT_EXISTS,
                            "A bucket.", object_ids=["O1"], source_field="Objects"),
        ],
    )
    report = validate_platform_semantics(render.caption, graph.by_id())
    assert any(i.rule_id == "M2-PLATFORM-004" for i in report.issues)


def test_valid_connective_is_not_blindly_banned() -> None:
    _shots, graph, _plan, render = _pipeline(
        [_hf("HF-A", "C1 holds O1 with both hands.", Fraction(0), Fraction(1))],
        seed_claims=[
            supported_claim("CLM-C1", SeedClaimType.CHARACTER_EXISTS,
                            "A person.", subject_ids=["C1"]),
            supported_claim("CLM-O1", SeedClaimType.OBJECT_EXISTS,
                            "A bucket.", object_ids=["O1"], source_field="Objects"),
        ],
    )
    report = validate_platform_semantics(render.caption, graph.by_id())
    assert report.status == "PASS"


def test_multiple_sentences_in_one_entry_fail() -> None:
    _shots, graph, _plan, render = _pipeline(
        [_hf("HF-A", "C1 walks forward. O1 swings at the side.", Fraction(0), Fraction(1))],
        seed_claims=[
            supported_claim("CLM-C1", SeedClaimType.CHARACTER_EXISTS,
                            "A person.", subject_ids=["C1"]),
            supported_claim("CLM-O1", SeedClaimType.OBJECT_EXISTS,
                            "A bucket.", object_ids=["O1"], source_field="Objects"),
        ],
    )
    report = validate_platform_semantics(render.caption, graph.by_id())
    assert any(i.rule_id == "M2-PLATFORM-001" for i in report.issues)


def test_multiple_quote_spans_fail() -> None:
    line = RenderedCaptionLine(
        line_id="L-0001",
        section=CaptionSection.ACTION_AUDIO,
        shot_number=1,
        text='[0.0s-1.0s] C1 says, "Stop" and then says, "Go"',
        display_start="0.0s",
        display_end="1.0s",
        start_exact=Fraction(0),
        end_exact=Fraction(1),
        fact_ids=["CF-0001"],
    )
    report = validate_platform_semantics(
        RenderedCaption(lines=[line], markdown="x", caption_sha256="y"), {}
    )
    rule_ids = {i.rule_id for i in report.issues}
    assert "M2-PLATFORM-002" in rule_ids
    assert "M2-PLATFORM-003" in rule_ids


# --- §109 character rules --------------------------------------------------


def test_pronoun_outside_dialogue_fails() -> None:
    shots, graph, plan, render = _pipeline(
        [_hf("HF-A", "C1 raises his hand.", Fraction(0), Fraction(1))]
    )
    issues = _m2(shots, graph, plan, render)
    assert any(i.rule_id == "M2-CAST-004" and i.severity == Severity.FAIL for i in issues)


def test_lower_body_contradiction_fails() -> None:
    shots, graph, plan, render = _pipeline(
        [_hf("HF-A", "C1 stands still.", Fraction(0), Fraction(1))],
        seed_claims=[
            supported_claim(
                "CLM-C1", SeedClaimType.CHARACTER_EXISTS,
                "A person in dark trousers. Lower body and shoes are not visible.",
                subject_ids=["C1"],
            ),
        ],
    )
    issues = _m2(shots, graph, plan, render)
    assert any(i.rule_id == "M2-CAST-005" for i in issues)


def test_lower_body_sentence_without_contradiction_passes() -> None:
    shots, graph, plan, render = _pipeline(
        [_hf("HF-A", "C1 stands still.", Fraction(0), Fraction(1))],
        seed_claims=[
            supported_claim(
                "CLM-C1", SeedClaimType.CHARACTER_EXISTS,
                "A person in a green jacket. Lower body and shoes are not visible.",
                subject_ids=["C1"],
            ),
        ],
    )
    issues = _m2(shots, graph, plan, render)
    assert not any(i.rule_id == "M2-CAST-005" for i in issues)


def test_ghost_character_reference_fails() -> None:
    shots, graph, plan, render = _pipeline(
        [_hf("HF-A", "C7 waves at the camera.", Fraction(0), Fraction(1))]
    )
    issues = _m2(shots, graph, plan, render)
    assert any(i.rule_id == "M2-CAST-001" for i in issues)


# --- §114 field placement --------------------------------------------------


def test_camera_movement_in_scene_fails() -> None:
    shots, graph, plan, render = _pipeline(
        [],
        seed_claims=[
            supported_claim("CLM-C1", SeedClaimType.CHARACTER_EXISTS,
                            "A person.", subject_ids=["C1"]),
            supported_claim(
                "CLM-SC", SeedClaimType.SCENE_STATE,
                "The camera pans across a dirt path.", source_field="SCENE",
            ),
        ],
    )
    issues = _m2(shots, graph, plan, render)
    assert any(i.rule_id == "M2-CAMERA-002" for i in issues)


def test_transcript_in_audio_concerns_fails() -> None:
    shots, graph, plan, render = _pipeline(
        [],
        seed_claims=[
            supported_claim(
                "CLM-AC", SeedClaimType.AUDIO_CONCERN,
                "The transcript is uncertain in places.", source_field="AUDIO_CONCERNS",
            ),
        ],
    )
    issues = _m2(shots, graph, plan, render)
    assert any(i.rule_id == "M2-AUDIO-003" for i in issues)


def test_reviewer_note_in_caption_fails() -> None:
    shots, graph, plan, render = _pipeline(
        [],
        seed_claims=[
            supported_claim(
                "CLM-SC", SeedClaimType.SCENE_STATE,
                "A canal bank; speaker needs verification.", source_field="SCENE",
            ),
        ],
    )
    issues = _m2(shots, graph, plan, render)
    assert any(i.rule_id == "M2-FIELD-001" for i in issues)


def test_empty_concerns_render_none_literal() -> None:
    _shots, _graph, _plan, render = _pipeline([])
    concern_lines = [
        ln
        for ln in render.caption.lines
        if ln.section in (CaptionSection.VISUAL_CONCERNS, CaptionSection.AUDIO_CONCERNS)
    ]
    assert len(concern_lines) == 2
    assert all(ln.text == "None." and ln.structural for ln in concern_lines)


def test_deprecated_unintelligible_token_fails() -> None:
    shots, graph, plan, render = _pipeline(
        [_hf("HF-A", "C1 mutters <unintelligible> words.", Fraction(0), Fraction(1))],
        seed_claims=[
            supported_claim("CLM-C1", SeedClaimType.CHARACTER_EXISTS,
                            "A person.", subject_ids=["C1"]),
        ],
    )
    issues = _m2(shots, graph, plan, render)
    assert any(i.rule_id == "M2-SPEECH-001" for i in issues)


def test_shot_one_must_be_opening_shot() -> None:
    shots = make_shot_truth([make_shot(1, Fraction(0), Fraction(2), "Hard cut")])
    inputs = base_inputs(shot_truth=shots)
    graph = build_fact_graph(inputs)
    plan = build_caption_plan(PlanInputs(graph=graph, shot_truth=shots))
    render = render_caption(plan, graph.by_id())
    issues = _m2(shots, graph, plan, render)
    assert any(i.rule_id == "M2-TRANSITION-003" for i in issues)
