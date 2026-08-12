"""Phase 5 — Caption Brain orchestrator.

    CAPTION ELIGIBILITY → CAPTION FACT GRAPH → CAPTION PLAN → BUILDERS →
    DETERMINISTIC RENDER → M2 VALIDATOR → PLATFORM-SEMANTIC VALIDATOR →
    GOLDEN BEHAVIOR GATE → OMISSION/HALLUCINATION PASS → ADVERSARIAL QC →
    READY STATE

Runs entirely from existing Phase 1-4 evidence artifacts — the video is never
re-decoded (§123), so re-finalization after a human correction is fast (§94).
No cloud upload, no platform submission, no result-code generation (§52/§98).
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ValidationError

from .artifacts import caption_writer
from .artifacts.writer import sha256_file
from .caption import human_facts as human_facts_mod
from .caption.coverage import (
    AssertionCheckResult,
    CoverageResult,
    build_coverage,
    check_assertions,
)
from .caption.eligibility import EligibilityContext
from .caption.facts import (
    FactBuildInputs,
    FactGraph,
    ambiguous_track_ids,
    build_fact_graph,
)
from .caption.finalizer import (
    ReadinessInputs,
    SignoffLoadError,
    compute_readiness,
    load_signoff,
)
from .caption.planning import PlanInputs, build_caption_plan
from .caption.renderer import RenderResult, render_caption
from .models.audio import AudioQCResult
from .models.caption import (
    ActionAudioEvent,
    Character,
    ExactTimeRange,
    ObjectEntity,
    ReviewedCaption,
    SeedClaim,
    Shot,
    Transition,
)
from .models.caption_brain import (
    CaptionBrainResult,
    CaptionEligibility,
    CaptionFactType,
    CaptionPlan,
    CaptionReadiness,
    FinalReviewSignoff,
    HumanCaptionFact,
)
from .models.review_intelligence import (
    ActionCandidate,
    CameraMotionCandidate,
    ContinuityLink,
    DecisionApplication,
    EntityTrack,
    FeedbackDirective,
    FinalStateCheck,
    HumanReviewDecision,
    PlaybackSpeedEvidence,
    ReviewProposal,
    TextTrack,
)
from .models.shot_truth import ShotTruthResult
from .models.validation import Severity, ValidatorIssue
from .review.decisions import (
    DecisionLoadError,
    DecisionTargets,
    apply_decisions,
    load_decisions,
)
from .rules.loader import load_rules
from .validation.caption_validator import (
    M2_VALIDATOR_VERSION,
    M2Inputs,
    validate_caption,
)
from .validation.final_caption_validator import (
    AdversarialQCResult,
    SignoffCheck,
    check_signoff,
    run_adversarial_qc,
)
from .validation.golden_validator import run_golden_gate
from .validation.platform_semantic_validator import (
    PlatformSemanticReport,
    validate_platform_semantics,
)

logger = logging.getLogger(__name__)

CAPTION_BRAIN_VERSION = "0.5.0"


class CaptionBrainError(RuntimeError):
    """The Caption Brain could not run (missing/invalid run evidence)."""


@dataclass
class CaptionBrainOutput:
    result: CaptionBrainResult
    plan: CaptionPlan | None = None
    issues: list[ValidatorIssue] = field(default_factory=list)
    artifact_paths: list[Path] = field(default_factory=list)
    stage_timings: dict[str, float] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Evidence loading (no media re-analysis, §123)
# ---------------------------------------------------------------------------


def _load_model_list[M: BaseModel](path: Path, key: str, model: type[M]) -> list[M]:
    if not path.exists():
        return []
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        return [model.model_validate(entry) for entry in raw.get(key, [])]
    except (OSError, json.JSONDecodeError, ValidationError) as exc:
        raise CaptionBrainError(f"Cannot load {path.name}: {exc}") from exc


def _load_model[M: BaseModel](path: Path, model: type[M]) -> M | None:
    if not path.exists():
        return None
    try:
        return model.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValidationError) as exc:
        raise CaptionBrainError(f"Cannot load {path.name}: {exc}") from exc


@dataclass
class RunEvidence:
    video_sha256: str | None = None
    rules_version: str | None = None
    seed_sha256: str | None = None
    video_id: str | None = None
    shot_truth: ShotTruthResult | None = None
    audio_truth: AudioQCResult | None = None
    seed_claims: list[SeedClaim] = field(default_factory=list)
    proposals: list[ReviewProposal] = field(default_factory=list)
    feedback: list[FeedbackDirective] = field(default_factory=list)
    camera_candidates: list[CameraMotionCandidate] = field(default_factory=list)
    speed_evidence: list[PlaybackSpeedEvidence] = field(default_factory=list)
    text_tracks: list[TextTrack] = field(default_factory=list)
    action_candidates: list[ActionCandidate] = field(default_factory=list)
    final_state_checks: list[FinalStateCheck] = field(default_factory=list)
    entity_tracks: list[EntityTrack] = field(default_factory=list)
    continuity_links: list[ContinuityLink] = field(default_factory=list)


def load_run_evidence(run_dir: Path) -> RunEvidence:
    """Reload Phase 1-4 evidence artifacts. Verifies manifest hashes for the
    inputs it consumes (§94) — tampered evidence never silently finalizes."""
    evidence = RunEvidence()
    manifest_path = run_dir / "manifest.json"
    manifest_hashes: dict[str, str] = {}
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise CaptionBrainError(f"Cannot read manifest.json: {exc}") from exc
        evidence.video_sha256 = manifest.get("source_video_sha256")
        evidence.rules_version = manifest.get("rules_version")
        for entry in manifest.get("artifacts", []):
            manifest_hashes[str(entry.get("path", ""))] = str(entry.get("sha256", ""))

    def _verify(path: Path) -> None:
        rel = path.relative_to(run_dir).as_posix()
        recorded = manifest_hashes.get(rel)
        if recorded is not None and path.exists() and sha256_file(path) != recorded:
            raise CaptionBrainError(
                f"Evidence artifact {rel} does not match its manifest hash — "
                "the run evidence changed after analysis."
            )

    shot_qc = run_dir / "shot_qc.json"
    _verify(shot_qc)
    evidence.shot_truth = _load_model(shot_qc, ShotTruthResult)
    audio_qc = run_dir / "audio" / "audio_qc.json"
    _verify(audio_qc)
    evidence.audio_truth = _load_model(audio_qc, AudioQCResult)

    seed_dir = run_dir / "seed"
    evidence.seed_claims = _load_model_list(
        seed_dir / "seed_claims.json", "claims", SeedClaim
    )
    sha_file = seed_dir / "seed_sha256.txt"
    if sha_file.exists():
        evidence.seed_sha256 = sha_file.read_text(encoding="utf-8").strip()
    parse_path = seed_dir / "seed_parse.json"
    if parse_path.exists():
        try:
            parsed = json.loads(parse_path.read_text(encoding="utf-8"))
            evidence.video_id = parsed.get("video_id")
        except (OSError, json.JSONDecodeError):
            evidence.video_id = None

    evidence.proposals = _load_model_list(
        run_dir / "review" / "review_proposals.json", "proposals", ReviewProposal
    )
    evidence.feedback = _load_model_list(
        run_dir / "feedback" / "feedback_directives.json", "directives", FeedbackDirective
    )
    visual = run_dir / "visual"
    evidence.camera_candidates = _load_model_list(
        visual / "camera" / "camera_events.json", "camera_events", CameraMotionCandidate
    )
    evidence.speed_evidence = _load_model_list(
        visual / "speed" / "playback_speed_evidence.json",
        "playback_speed_evidence",
        PlaybackSpeedEvidence,
    )
    evidence.text_tracks = _load_model_list(
        visual / "ocr" / "text_tracks.json", "text_tracks", TextTrack
    )
    evidence.action_candidates = _load_model_list(
        visual / "actions" / "action_candidates.json", "action_candidates", ActionCandidate
    )
    evidence.final_state_checks = _load_model_list(
        visual / "entities" / "final_state_checks.json",
        "final_state_checks",
        FinalStateCheck,
    )
    evidence.entity_tracks = _load_model_list(
        visual / "entities" / "tracks.json", "tracks", EntityTrack
    )
    evidence.continuity_links = _load_model_list(
        visual / "entities" / "continuity_links.json", "continuity_links", ContinuityLink
    )
    return evidence


# ---------------------------------------------------------------------------
# Finalization
# ---------------------------------------------------------------------------


def finalize_run(
    run_dir: Path,
    review_decisions_path: Path | None = None,
    human_facts_path: Path | None = None,
    final_review_path: Path | None = None,
    video_sha256: str | None = None,
) -> CaptionBrainOutput:
    """Fast re-finalization (§94): existing evidence + current human inputs →
    facts → plan → render → validate → gates → final status. No FFmpeg.

    ``video_sha256`` overrides the manifest value when the pipeline calls this
    before the run manifest exists (same-process integration)."""
    timings: dict[str, float] = {}
    total_start = time.perf_counter()
    issues: list[ValidatorIssue] = []
    artifacts: list[Path] = []
    rules = load_rules()

    start = time.perf_counter()
    evidence = load_run_evidence(run_dir)
    if video_sha256 is not None:
        evidence.video_sha256 = video_sha256
    timings["load_evidence"] = round(time.perf_counter() - start, 4)
    rules_version = evidence.rules_version or rules.version

    # --- human decisions (Phase 4 authoritative layer) ---
    decisions: list[HumanReviewDecision] = []
    applications: list[DecisionApplication] = []
    if review_decisions_path is not None:
        try:
            decisions = load_decisions(review_decisions_path)
        except DecisionLoadError as exc:
            issues.append(
                ValidatorIssue(
                    rule_id="M2-SOURCE-005",
                    severity=Severity.FAIL,
                    location=review_decisions_path.name,
                    message=f"Review decisions rejected: {exc}",
                )
            )
        if decisions and evidence.video_sha256 is not None:
            targets = DecisionTargets(
                claims={c.claim_id: c for c in evidence.seed_claims},
                entity_tracks={t.track_id: t for t in evidence.entity_tracks},
                camera_candidates={
                    c.candidate_id: c for c in evidence.camera_candidates
                },
                action_candidates={
                    a.candidate_id: a for a in evidence.action_candidates
                },
                speed_evidence={
                    f"SPEED-{s.shot_number}": s for s in evidence.speed_evidence
                },
                proposals={p.proposal_id: p for p in evidence.proposals},
            )
            applications = apply_decisions(
                decisions, targets, evidence.video_sha256, rules_version
            )

    # --- human-added facts (§9/§10) ---
    accepted_facts: list[HumanCaptionFact] = []
    rejected_facts: list[tuple[str, str]] = []
    if human_facts_path is not None and evidence.video_sha256 is not None:
        try:
            loaded = human_facts_mod.load_human_facts(
                human_facts_path,
                evidence.video_sha256,
                rules_version,
                evidence.seed_sha256,
            )
            accepted_facts = loaded.accepted
            rejected_facts = loaded.rejected
        except human_facts_mod.HumanFactLoadError as exc:
            issues.append(
                ValidatorIssue(
                    rule_id="M2-SOURCE-006",
                    severity=Severity.FAIL,
                    location=human_facts_path.name,
                    message=f"Human facts rejected: {exc}",
                )
            )

    # --- fact graph ---
    start = time.perf_counter()
    ctx = EligibilityContext.build(
        decisions,
        applications,
        ambiguous_track_ids(evidence.entity_tracks, evidence.continuity_links),
    )
    graph = build_fact_graph(
        FactBuildInputs(
            video_id=evidence.video_id,
            video_sha256=evidence.video_sha256,
            rules_version=rules_version,
            shot_truth=evidence.shot_truth,
            audio_truth=evidence.audio_truth,
            seed_claims=evidence.seed_claims,
            speed_evidence=evidence.speed_evidence,
            camera_candidates=evidence.camera_candidates,
            text_tracks=evidence.text_tracks,
            action_candidates=evidence.action_candidates,
            final_state_checks=evidence.final_state_checks,
            entity_tracks=evidence.entity_tracks,
            continuity_links=evidence.continuity_links,
            human_facts=accepted_facts,
            ctx=ctx,
        )
    )
    timings["fact_graph"] = round(time.perf_counter() - start, 4)

    # --- feedback resolution (§89) ---
    resolved_directives = _resolved_directives(evidence.feedback, ctx, accepted_facts)
    unresolved_high = [
        d.directive_id
        for d in evidence.feedback
        if d.priority.value == "HIGH" and d.directive_id not in resolved_directives
    ]

    # --- plan ---
    start = time.perf_counter()
    plan = build_caption_plan(
        PlanInputs(
            graph=graph,
            shot_truth=evidence.shot_truth,
            video_id=evidence.video_id,
            video_sha256=evidence.video_sha256,
            rules_version=rules_version,
            proposals=evidence.proposals,
            feedback=evidence.feedback,
            resolved_directive_ids=resolved_directives,
        )
    )
    timings["planning"] = round(time.perf_counter() - start, 4)

    # --- render ---
    start = time.perf_counter()
    facts_by_id = graph.by_id()
    render = render_caption(plan, facts_by_id)
    timings["render"] = round(time.perf_counter() - start, 4)

    # --- coverage + assertions (§59/§60) ---
    start = time.perf_counter()
    coverage = build_coverage(graph.facts, render.rendered_fact_lines)
    assertion_check = check_assertions(render.assertions, facts_by_id)
    timings["coverage"] = round(time.perf_counter() - start, 4)

    # --- M2 + platform + golden + adversarial ---
    start = time.perf_counter()
    endpoint = (
        evidence.shot_truth.annotation_endpoint_exact
        if evidence.shot_truth is not None
        else None
    )
    m2_issues = validate_caption(
        M2Inputs(
            plan=plan,
            caption=render.caption,
            facts_by_id=facts_by_id,
            shot_truth=evidence.shot_truth,
            expected_video_id=evidence.video_id,
            coverage=coverage,
            assertions=assertion_check,
            annotation_endpoint=Fraction(endpoint) if endpoint is not None else None,
            unresolved_high_feedback=unresolved_high,
        )
    )
    timings["m2_validation"] = round(time.perf_counter() - start, 4)
    start = time.perf_counter()
    platform_report = validate_platform_semantics(render.caption, facts_by_id)
    timings["platform_validation"] = round(time.perf_counter() - start, 4)
    start = time.perf_counter()
    golden = run_golden_gate(plan, render.caption, graph.facts, coverage, platform_report)
    timings["golden_gate"] = round(time.perf_counter() - start, 4)
    start = time.perf_counter()
    adversarial = run_adversarial_qc(
        plan, render.caption, graph.facts, coverage, assertion_check, m2_issues
    )
    timings["adversarial_qc"] = round(time.perf_counter() - start, 4)

    # --- signoff (§54): loaded only from a human-supplied file ---
    signoff: FinalReviewSignoff | None = None
    if final_review_path is not None:
        try:
            signoff = load_signoff(final_review_path)
        except SignoffLoadError as exc:
            issues.append(
                ValidatorIssue(
                    rule_id="M2-SOURCE-007",
                    severity=Severity.FAIL,
                    location=final_review_path.name,
                    message=str(exc),
                )
            )
    signoff_check = check_signoff(
        signoff, evidence.video_sha256, rules_version, render.caption.caption_sha256
    )

    # --- readiness (§90) ---
    readiness = compute_readiness(
        ReadinessInputs(
            plan=plan,
            m2_issues=m2_issues + issues,
            platform_report=platform_report,
            golden=golden,
            adversarial=adversarial,
            signoff_check=signoff_check,
            unresolved_high_feedback=unresolved_high,
        )
    )

    result = _summarize(
        graph, plan, m2_issues, platform_report, golden, coverage, assertion_check,
        unresolved_high, signoff_check, readiness.readiness, readiness.blockers,
        evidence,
    )
    timings["caption_brain_total"] = round(time.perf_counter() - total_start, 4)
    result.stage_timings_seconds = dict(timings)

    # --- artifacts (§92) ---
    caption_dir = run_dir / "caption"
    artifacts += caption_writer.write_caption_facts(caption_dir, graph.facts)
    artifacts.append(caption_writer.write_eligibility_report(caption_dir, graph.facts))
    artifacts += caption_writer.write_caption_plan(caption_dir, plan)
    artifacts.append(
        caption_writer.write_reviewed_caption(
            caption_dir, _build_reviewed_caption(plan, graph, render)
        )
    )
    artifacts += caption_writer.write_rendered_caption(
        caption_dir, render.caption, readiness.readiness
    )
    artifacts.append(caption_writer.write_assertion_map(caption_dir, render.assertions))
    artifacts.append(caption_writer.write_coverage(caption_dir, coverage.items))
    artifacts += caption_writer.write_seed_change_log(caption_dir, plan.seed_change_summary)
    artifacts += caption_writer.write_m2_report(caption_dir, m2_issues, M2_VALIDATOR_VERSION)
    artifacts.append(caption_writer.write_platform_report(caption_dir, platform_report))
    artifacts.append(caption_writer.write_golden_gate(caption_dir, golden))
    artifacts.append(caption_writer.write_adversarial_qc(caption_dir, adversarial))
    artifacts.append(caption_writer.write_final_review_checklist(caption_dir, signoff_check))
    if accepted_facts or rejected_facts:
        artifacts.append(
            caption_writer.write_human_facts_record(caption_dir, accepted_facts, rejected_facts)
        )
    artifacts.append(caption_writer.write_final_status(caption_dir, result))
    artifacts.append(
        caption_writer.write_review_report(
            run_dir,
            _review_report_sections(result, plan, m2_issues, golden, adversarial),
        )
    )
    artifacts.append(
        caption_writer.write_caption_manifest(
            caption_dir,
            {
                "caption_brain_version": CAPTION_BRAIN_VERSION,
                "rules_version": rules_version,
                "golden_behavior_version": golden.rules_version,
                "video_sha256": evidence.video_sha256,
                "seed_sha256": evidence.seed_sha256,
                "review_decisions_sha256": (
                    sha256_file(review_decisions_path)
                    if review_decisions_path is not None and review_decisions_path.exists()
                    else None
                ),
                "human_facts_sha256": (
                    sha256_file(human_facts_path)
                    if human_facts_path is not None and human_facts_path.exists()
                    else None
                ),
                "final_review_sha256": (
                    sha256_file(final_review_path)
                    if final_review_path is not None and final_review_path.exists()
                    else None
                ),
                "rendered_caption_sha256": render.caption.caption_sha256,
                "final_status": readiness.readiness.value,
                "stage_timings_seconds": timings,
            },
            artifacts,
        )
    )
    return CaptionBrainOutput(
        result=result,
        plan=plan,
        issues=issues + m2_issues,
        artifact_paths=artifacts,
        stage_timings={f"caption_{k}": v for k, v in timings.items()},
    )


def _resolved_directives(
    feedback: list[FeedbackDirective],
    ctx: EligibilityContext,
    human_facts: list[HumanCaptionFact],
) -> frozenset[str]:
    """A directive is RESOLVED/NOT_APPLICABLE only via explicit human input:
    an applied decision on the directive subject, or a human fact naming it."""
    resolved: set[str] = set()
    for (_, subject_id), _decision in ctx.applied_decisions.items():
        resolved.add(subject_id)
    for hf in human_facts:
        target = hf.semantic_value.get("resolves_directive")
        if target:
            resolved.add(target)
    return frozenset(d.directive_id for d in feedback if d.directive_id in resolved)


def _summarize(
    graph: FactGraph,
    plan: CaptionPlan,
    m2_issues: list[ValidatorIssue],
    platform_report: PlatformSemanticReport,
    golden: Any,
    coverage: CoverageResult,
    assertion_check: AssertionCheckResult,
    unresolved_high: list[str],
    signoff_check: SignoffCheck,
    readiness: CaptionReadiness,
    blockers: list[str],
    evidence: RunEvidence,
) -> CaptionBrainResult:
    def count(fact_type: CaptionFactType, statuses: tuple[CaptionEligibility, ...]) -> int:
        return sum(
            1
            for f in graph.facts
            if f.fact_type == fact_type and f.eligibility in statuses
        )

    blocked_statuses = (
        CaptionEligibility.REVIEW_REQUIRED,
        CaptionEligibility.INELIGIBLE,
        CaptionEligibility.REJECTED,
    )
    return CaptionBrainResult(
        caption_brain_version=CAPTION_BRAIN_VERSION,
        video_id=evidence.video_id,
        fact_count=len(graph.facts),
        eligible_count=sum(
            1 for f in graph.facts if f.eligibility == CaptionEligibility.ELIGIBLE
        ),
        review_required_count=sum(
            1 for f in graph.facts if f.eligibility == CaptionEligibility.REVIEW_REQUIRED
        ),
        ineligible_count=sum(
            1 for f in graph.facts if f.eligibility == CaptionEligibility.INELIGIBLE
        ),
        rejected_count=sum(
            1 for f in graph.facts if f.eligibility == CaptionEligibility.REJECTED
        ),
        shot_count=len(plan.shot_plans),
        character_count=len(plan.overview_plan.characters),
        object_count=len(plan.overview_plan.objects),
        speech_verified_count=count(CaptionFactType.SPEECH, (CaptionEligibility.ELIGIBLE,)),
        speech_blocked_count=count(CaptionFactType.SPEECH, blocked_statuses),
        text_verified_count=count(
            CaptionFactType.ON_SCREEN_TEXT, (CaptionEligibility.ELIGIBLE,)
        ),
        text_blocked_count=count(CaptionFactType.ON_SCREEN_TEXT, blocked_statuses),
        action_verified_count=count(
            CaptionFactType.VISUAL_ACTION, (CaptionEligibility.ELIGIBLE,)
        ),
        action_blocked_count=count(CaptionFactType.VISUAL_ACTION, blocked_statuses),
        m2_fail_count=sum(1 for i in m2_issues if i.severity == Severity.FAIL),
        m2_review_count=sum(1 for i in m2_issues if i.severity == Severity.WARN),
        platform_semantic_status=platform_report.status,
        golden_gate_status=golden.overall.value,
        coverage_missing_required=len(coverage.missing_required_fact_ids),
        unmapped_assertions=len(assertion_check.unmapped),
        unresolved_feedback_high=len(unresolved_high),
        signoff_present=signoff_check.present,
        signoff_stale=signoff_check.stale,
        readiness=readiness,
        blockers=blockers,
    )


def _build_reviewed_caption(
    plan: CaptionPlan, graph: FactGraph, render: RenderResult
) -> ReviewedCaption:
    """Structured caption record (§92 reviewed_caption.json)."""
    facts_by_id = graph.by_id()
    characters = []
    for entry in plan.overview_plan.characters:
        description = " ".join(
            facts_by_id[fid].text_value or ""
            for fid in entry.description_fact_ids
            if fid in facts_by_id
        ).strip()
        characters.append(
            Character(
                character_id=entry.character_id,
                description=description,
                off_screen_only=entry.off_screen_only,
            )
        )
    objects = []
    for obj in plan.overview_plan.objects:
        description = " ".join(
            facts_by_id[fid].text_value or ""
            for fid in obj.description_fact_ids
            if fid in facts_by_id
        ).strip()
        objects.append(ObjectEntity(object_id=obj.object_id, description=description))

    def _section_text(fact_ids: list[str]) -> str | None:
        parts = [
            facts_by_id[fid].text_value
            for fid in fact_ids
            if fid in facts_by_id and facts_by_id[fid].text_value
        ]
        return " ".join(p for p in parts if p) or None

    shots = []
    action_events = []
    for shot_plan in plan.shot_plans:
        transition_fact = facts_by_id.get(shot_plan.transition_fact_id or "")
        shots.append(
            Shot(
                shot_number=shot_plan.shot_number,
                time_range=ExactTimeRange(
                    start_seconds=shot_plan.start_exact, end_seconds=shot_plan.end_exact
                ),
                transition=Transition(
                    transition_type=(
                        transition_fact.text_value
                        if shot_plan.transition_resolved
                        and transition_fact is not None
                        and transition_fact.text_value
                        else "UNRESOLVED"
                    )
                ),
                playback_speed=None,
            )
        )
        for fid in shot_plan.event_fact_ids:
            fact = facts_by_id[fid]
            if fact.start_exact is None or fact.end_exact is None:
                continue
            line_id = render.rendered_fact_lines.get(fid)
            text = next(
                (ln.text for ln in render.caption.lines if ln.line_id == line_id), ""
            )
            action_events.append(
                ActionAudioEvent(
                    shot_number=shot_plan.shot_number,
                    time_range=ExactTimeRange(
                        start_seconds=fact.start_exact, end_seconds=fact.end_exact
                    ),
                    text=text,
                    character_ids=fact.character_ids,
                    object_ids=fact.object_ids,
                    evidence=fact.evidence_refs,
                )
            )
    return ReviewedCaption(
        characters=characters,
        objects=objects,
        scene=_section_text(plan.overview_plan.scene_fact_ids),
        style=_section_text(plan.overview_plan.style_fact_ids),
        overview_audio=_section_text(plan.overview_plan.overview_audio_fact_ids),
        visual_concerns=_section_text(plan.overview_plan.visual_concern_fact_ids),
        audio_concerns=_section_text(plan.overview_plan.audio_concern_fact_ids),
        shots=shots,
        action_audio_events=action_events,
    )


def _review_report_sections(
    result: CaptionBrainResult,
    plan: CaptionPlan,
    m2_issues: list[ValidatorIssue],
    golden: Any,
    adversarial: AdversarialQCResult,
) -> list[tuple[str, list[str]]]:
    """The reviewer packet (§96/§57): everything that remains, outside caption
    fields."""
    blockers = [f"- {b}" for b in result.blockers]
    m2 = [
        f"- {i.severity.value} {i.rule_id} {i.location}: {i.message}" for i in m2_issues
    ]
    gate = [f"- {c.category}: {c.status.value} — {c.detail}" for c in golden.categories]
    qc = [
        f"- [{'BLOCKING' if f.blocking else 'review'}] {f.category}: {f.detail}"
        for f in adversarial.findings
    ]
    seed = [
        f"- {e.section} {e.subject_id or ''}: {e.disposition}"
        + (" (human)" if e.decided_by_human else " (machine proposal)")
        for e in plan.seed_change_summary
    ]
    return [
        (f"Final status: {result.readiness.value}", blockers),
        ("Seed KEEP / FIX_ENRICH / REDO_REBUILD", seed),
        ("M2 validator findings", m2),
        ("Golden behavior gate", gate),
        ("Final adversarial QC", qc),
    ]
