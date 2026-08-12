"""Phase 5 fact-graph / plan / renderer tests: speech ladder + foreign
language (§104/§105), songs (§106), coverage & hallucination gates
(§116/§117), quote handling (§108), character rules (§109)."""

from __future__ import annotations

from fractions import Fraction

from manuscript_reviewer.caption.coverage import build_coverage, check_assertions
from manuscript_reviewer.caption.facts import build_fact_graph
from manuscript_reviewer.caption.planning import PlanInputs, build_caption_plan
from manuscript_reviewer.caption.renderer import render_caption
from manuscript_reviewer.caption.textcheck import (
    find_quote_spans,
    pronoun_hits,
    sentence_count,
)
from manuscript_reviewer.models.audio import SourceVerificationStatus
from manuscript_reviewer.models.caption_brain import (
    AssertionStatus,
    CaptionAssertionRecord,
    CaptionEligibility,
    CaptionFact,
    CaptionFactType,
    CaptionSection,
    EligibilityBasis,
    FactMateriality,
    FactSourceKind,
    HumanCaptionFact,
    LanguageRenderLevel,
)
from manuscript_reviewer.models.review_intelligence import (
    FeedbackDirective,
    FeedbackPriority,
    InterpretationStatus,
    SeedClaimType,
)
from manuscript_reviewer.models.shot_truth import TransitionStatus

from .phase5_helpers import (
    RULES_VERSION,
    VIDEO_SHA,
    base_inputs,
    factual_ref,
    make_audio_truth,
    make_shot,
    make_shot_truth,
    make_speech_region,
    supported_claim,
)


def _human_fact(
    fact_id: str,
    fact_type: CaptionFactType,
    text: str | None = None,
    shot_number: int | None = 1,
    start: Fraction | None = None,
    end: Fraction | None = None,
    character_ids: list[str] | None = None,
    semantic: dict[str, str] | None = None,
) -> HumanCaptionFact:
    return HumanCaptionFact(
        fact_id=fact_id,
        fact_type=fact_type,
        text_value=text,
        semantic_value=semantic or {},
        shot_number=shot_number,
        character_ids=character_ids or [],
        start_exact=start,
        end_exact=end,
        evidence_refs=[factual_ref(f"EV-{fact_id}")],
        decided_by="reviewer@test",
        bound_video_sha256=VIDEO_SHA,
        bound_rules_version=RULES_VERSION,
    )


def _one_shot_truth() -> object:
    return make_shot_truth([make_shot(1, Fraction(0), Fraction(2), "Opening shot")])


def test_verified_speech_renders_verbatim_with_speaker() -> None:
    region = make_speech_region(
        "SR-1", Fraction(1, 2), Fraction(1),
        "I'm gonna, I'm just gonna put them in here where they--",
        status=SourceVerificationStatus.HUMAN_VERIFIED,
    )
    enrichment = _human_fact(
        "HF-SPK", CaptionFactType.SPEECH,
        semantic={"region_id": "SR-1", "speaker_id": "C1", "tone": "a casual tone"},
        character_ids=["C1"],
    )
    inputs = base_inputs(
        shot_truth=_one_shot_truth(),
        audio_truth=make_audio_truth([region]),
        seed_claims=[
            supported_claim("CLM-C1", SeedClaimType.CHARACTER_EXISTS,
                            "A stocky figure in a black T-shirt.", subject_ids=["C1"]),
        ],
        human_facts=[enrichment],
    )
    graph = build_fact_graph(inputs)
    speech = [f for f in graph.facts if f.fact_type == CaptionFactType.SPEECH]
    assert len(speech) == 1
    assert speech[0].eligibility == CaptionEligibility.ELIGIBLE
    plan = build_caption_plan(PlanInputs(graph=graph, shot_truth=inputs.shot_truth))
    render = render_caption(plan, graph.by_id())
    line = next(
        ln for ln in render.caption.lines if ln.section == CaptionSection.ACTION_AUDIO
    )
    # Verbatim: the cutoff is preserved; exactly one quote span; tone included.
    assert 'where they--"' in line.text
    assert line.text.count("C1 says") == 1
    assert len(find_quote_spans(line.text)) == 1


def test_unverified_asr_dialogue_never_appears_in_final_output() -> None:
    region = make_speech_region("SR-1", Fraction(0), Fraction(1), "never render me")
    inputs = base_inputs(
        shot_truth=_one_shot_truth(), audio_truth=make_audio_truth([region])
    )
    graph = build_fact_graph(inputs)
    plan = build_caption_plan(PlanInputs(graph=graph, shot_truth=inputs.shot_truth))
    render = render_caption(plan, graph.by_id())
    assert "never render me" not in render.caption.markdown


def test_speech_without_speaker_is_blocked() -> None:
    region = make_speech_region(
        "SR-1", Fraction(0), Fraction(1), "who said this",
        status=SourceVerificationStatus.HUMAN_VERIFIED,
    )
    graph = build_fact_graph(
        base_inputs(shot_truth=_one_shot_truth(), audio_truth=make_audio_truth([region]))
    )
    speech = next(f for f in graph.facts if f.fact_type == CaptionFactType.SPEECH)
    assert speech.eligibility == CaptionEligibility.REVIEW_REQUIRED
    assert "speaker" in speech.eligibility_reason


def test_unicode_verbatim_speech_preserved_no_translation() -> None:
    """Hindi/Arabic/Chinese/Japanese/Cyrillic/accented-Latin fixtures survive
    end-to-end without ASCII collapse or translation (§105)."""
    samples = ["नमस्ते दोस्तों", "مرحبا بكم", "你好世界", "こんにちは", "Привет мир", "café čeština"]
    for i, text in enumerate(samples):
        region = make_speech_region(
            f"SR-{i}", Fraction(i, 10), Fraction(i + 1, 10), text,
            status=SourceVerificationStatus.HUMAN_VERIFIED,
        )
        enrichment = _human_fact(
            f"HF-{i}", CaptionFactType.SPEECH,
            semantic={"region_id": f"SR-{i}", "speaker_id": "C1"},
            character_ids=["C1"],
        )
        inputs = base_inputs(
            shot_truth=_one_shot_truth(),
            audio_truth=make_audio_truth([region]),
            seed_claims=[
                supported_claim("CLM-C1", SeedClaimType.CHARACTER_EXISTS,
                                "A speaker.", subject_ids=["C1"])
            ],
            human_facts=[enrichment],
        )
        graph = build_fact_graph(inputs)
        plan = build_caption_plan(PlanInputs(graph=graph, shot_truth=inputs.shot_truth))
        render = render_caption(plan, graph.by_id())
        assert text in render.caption.markdown


def test_foreign_language_fallback_never_guesses() -> None:
    fact = CaptionFact(
        fact_id="CF-X",
        fact_type=CaptionFactType.SPEECH,
        semantic_value={
            "speaker_id": "C2",
            "off_screen": "true",
            "language_level": LanguageRenderLevel.FOREIGN_LANGUAGE.value,
        },
        eligibility=CaptionEligibility.ELIGIBLE,
        eligibility_basis=EligibilityBasis.HUMAN_ADDED_FACT,
        eligibility_reason="test",
        source_kind=FactSourceKind.HUMAN_FACT,
    )
    from manuscript_reviewer.caption.renderer import render_speech_text

    text = render_speech_text(fact)
    assert text == "C2 speaks off-screen in a foreign language."
    assert "possibly" not in text and "maybe" not in text and "probably" not in text


def test_indiscernible_speech_wording() -> None:
    from manuscript_reviewer.caption.renderer import render_speech_text

    fact = CaptionFact(
        fact_id="CF-Y",
        fact_type=CaptionFactType.SPEECH,
        semantic_value={
            "speaker_id": "C1",
            "off_screen": "true",
            "language_level": LanguageRenderLevel.INDISCERNIBLE.value,
        },
        eligibility=CaptionEligibility.ELIGIBLE,
        eligibility_reason="test",
        source_kind=FactSourceKind.HUMAN_FACT,
    )
    assert render_speech_text(fact) == "C1 speaks off-screen; the words are indiscernible."


def test_cross_shot_speech_requires_human_split() -> None:
    """Speech crossing a visual cut is never rendered across shots (§34)."""
    shots = make_shot_truth(
        [
            make_shot(1, Fraction(0), Fraction(1), "Opening shot"),
            make_shot(2, Fraction(1), Fraction(2), "Hard cut"),
        ]
    )
    region = make_speech_region(
        "SR-1", Fraction(1, 2), Fraction(3, 2), "crossing words",
        status=SourceVerificationStatus.HUMAN_VERIFIED,
    )
    graph = build_fact_graph(
        base_inputs(shot_truth=shots, audio_truth=make_audio_truth([region]))
    )
    speech = next(f for f in graph.facts if f.fact_type == CaptionFactType.SPEECH)
    assert speech.eligibility == CaptionEligibility.REVIEW_REQUIRED
    assert "crosses a visual cut" in speech.eligibility_reason

    # Human split facts supersede the machine region: one entry per shot.
    splits = [
        _human_fact("HF-A", CaptionFactType.SPEECH, "crossing", 1,
                    Fraction(1, 2), Fraction(1), ["C1"],
                    {"splits_region_id": "SR-1", "speaker_id": "C1"}),
        _human_fact("HF-B", CaptionFactType.SPEECH, "words", 2,
                    Fraction(1), Fraction(3, 2), ["C1"],
                    {"splits_region_id": "SR-1", "speaker_id": "C1"}),
    ]
    graph2 = build_fact_graph(
        base_inputs(
            shot_truth=shots, audio_truth=make_audio_truth([region]), human_facts=splits
        )
    )
    speech2 = [f for f in graph2.facts if f.fact_type == CaptionFactType.SPEECH]
    assert len(speech2) == 2
    assert {f.shot_number for f in speech2} == {1, 2}


def test_unresolved_lyric_feedback_blocks_ready() -> None:
    """§106: a HIGH lyric directive keeps the plan in review until resolved."""
    directive = FeedbackDirective(
        directive_id="FBK-0001",
        raw_text="CRITICAL: English singing is not transcribed",
        source_line=1,
        priority=FeedbackPriority.HIGH,
        machine_interpretation="REQUIRE_VOCAL_LYRIC_REVIEW",
        interpretation_status=InterpretationStatus.MAPPED,
    )
    graph = build_fact_graph(base_inputs(shot_truth=_one_shot_truth()))
    plan = build_caption_plan(
        PlanInputs(
            graph=graph,
            shot_truth=make_shot_truth(
                [make_shot(1, Fraction(0), Fraction(2), "Opening shot")]
            ),
            feedback=[directive],
        )
    )
    assert any("FBK-0001" in b for b in plan.blockers)


def test_omission_gate_catches_missing_fact() -> None:
    """§116: 10 eligible material facts, one omitted without reason → FAIL."""
    facts = [
        CaptionFact(
            fact_id=f"CF-{i:04d}",
            fact_type=CaptionFactType.SCENE,
            text_value=f"Fact {i}.",
            eligibility=CaptionEligibility.ELIGIBLE,
            eligibility_basis=EligibilityBasis.DETERMINISTIC_EVIDENCE,
            eligibility_reason="test",
            source_kind=FactSourceKind.SEED_SUPPORTED,
            materiality=FactMateriality.MATERIAL,
        )
        for i in range(10)
    ]
    rendered = {f.fact_id: f"L-{i}" for i, f in enumerate(facts[:-1])}
    coverage = build_coverage(facts, rendered)
    assert not coverage.passed
    assert coverage.missing_required_fact_ids == ["CF-0009"]


def test_hallucination_gate_catches_unmapped_assertion() -> None:
    """§117: a phrase with no CaptionFact source fails the assertion map."""
    assertion = CaptionAssertionRecord(
        assertion_id="A-0001",
        line_id="L-0001",
        assertion_text="C1 wears a red scarf.",
        fact_ids=[],
        status=AssertionStatus.UNMAPPED,
    )
    result = check_assertions([assertion], {})
    assert not result.passed
    assert result.unmapped


def test_contractions_are_not_quote_spans() -> None:
    assert find_quote_spans("C1 says I've got it, no quotes here") == []
    assert sentence_count("[0.0s-1.0s] C1 raises the right hand to approx. head height.") == 1
    assert sentence_count("C1 moves at 4.3 seconds exactly.") == 1


def test_pronoun_detection_ignores_quotes() -> None:
    blocklist = ["he", "she", "they", "his", "her", "him", "them"]
    assert pronoun_hits('C1 says, "I told him everything."', blocklist) == []
    assert pronoun_hits("C1 raises his hand.", blocklist) == ["his"]


def test_unsupported_seed_traits_are_dropped_supported_kept() -> None:
    """§14: 'red shirt' supported → kept; age/nationality unsupported → out."""
    from manuscript_reviewer.models.caption import SeedClaim

    supported = supported_claim(
        "CLM-1", SeedClaimType.CHARACTER_TRAIT, "C1 wears a red shirt.",
        subject_ids=["C1"],
    )
    unsupported = SeedClaim(
        claim_id="CLM-2",
        source_field="Characters",
        text="C1 is a 25-year-old American man.",
        claim_type=SeedClaimType.PROTECTED_TRAIT,
        subject_ids=["C1"],
    )
    exists = supported_claim(
        "CLM-3", SeedClaimType.CHARACTER_EXISTS, "A man in a red shirt.",
        subject_ids=["C1"],
    )
    inputs = base_inputs(
        shot_truth=_one_shot_truth(), seed_claims=[supported, unsupported, exists]
    )
    graph = build_fact_graph(inputs)
    plan = build_caption_plan(PlanInputs(graph=graph, shot_truth=inputs.shot_truth))
    render = render_caption(plan, graph.by_id())
    assert "red shirt" in render.caption.markdown
    assert "25-year-old" not in render.caption.markdown
    assert "American" not in render.caption.markdown


def test_transition_from_unresolved_shot_never_rendered_as_hard_cut() -> None:
    shots = make_shot_truth(
        [
            make_shot(1, Fraction(0), Fraction(1), "Opening shot"),
            make_shot(2, Fraction(1), Fraction(2), None, TransitionStatus.UNRESOLVED),
        ]
    )
    graph = build_fact_graph(base_inputs(shot_truth=shots))
    plan = build_caption_plan(PlanInputs(graph=graph, shot_truth=shots))
    render = render_caption(plan, graph.by_id())
    assert "Hard cut" not in render.caption.markdown
    shot2 = plan.shot_plans[1]
    assert not shot2.transition_resolved


def test_multiline_overlay_renders_as_one_quote() -> None:
    """§36: one simultaneous overlay with several lines = ONE quoted string."""
    hf = _human_fact(
        "HF-UI", CaptionFactType.ON_SCREEN_TEXT,
        "SHOTGUN EXPERT\n500 Shotgun damage in a match\n+110 XP",
        1, Fraction(0), Fraction(1),
    )
    inputs = base_inputs(shot_truth=_one_shot_truth(), human_facts=[hf])
    graph = build_fact_graph(inputs)
    plan = build_caption_plan(PlanInputs(graph=graph, shot_truth=inputs.shot_truth))
    render = render_caption(plan, graph.by_id())
    line = next(
        ln for ln in render.caption.lines if ln.section == CaptionSection.ACTION_AUDIO
    )
    assert len(find_quote_spans(line.text)) == 1
    assert "SHOTGUN EXPERT / 500 Shotgun damage in a match / +110 XP" in line.text
