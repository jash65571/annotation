"""Phase 5 caption artifacts (§92). Deterministic JSON (sorted keys, exact
rationals as "num/den", ensure_ascii=False so non-Latin text survives).

Filename safety (§53/§119): ``ready_to_enter.md``/``.json`` exist ONLY when
final status is truly READY_TO_ENTER; otherwise the draft is written as
``draft_review_only.md``. Both are removed first so a stale ready file can
never survive a downgrade. No fake artifacts are created for stages that did
not run.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..models.caption import ReviewedCaption
from ..models.caption_brain import (
    CaptionAssertionRecord,
    CaptionBrainResult,
    CaptionCoverageItem,
    CaptionFact,
    CaptionPlan,
    CaptionReadiness,
    GoldenGateResult,
    HumanCaptionFact,
    RenderedCaption,
    SeedChangeEntry,
)
from ..models.validation import ValidatorIssue
from ..validation.final_caption_validator import AdversarialQCResult, SignoffCheck
from ..validation.platform_semantic_validator import PlatformSemanticReport
from .writer import ArtifactWriteError, sha256_file


def _write_json(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, ensure_ascii=False)
            handle.write("\n")
    except OSError as exc:
        raise ArtifactWriteError(f"Failed to write {path}: {exc}") from exc
    return path


def _write_text(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        path.write_text(text, encoding="utf-8", newline="\n")
    except OSError as exc:
        raise ArtifactWriteError(f"Failed to write {path}: {exc}") from exc
    return path


def write_caption_facts(caption_dir: Path, facts: list[CaptionFact]) -> list[Path]:
    json_path = _write_json(
        caption_dir / "caption_facts.json",
        {"facts": [f.model_dump(mode="json") for f in facts]},
    )
    jsonl_path = caption_dir / "caption_facts.jsonl"
    with jsonl_path.open("w", encoding="utf-8", newline="\n") as handle:
        for fact in facts:
            handle.write(
                json.dumps(fact.model_dump(mode="json"), sort_keys=True, ensure_ascii=False)
                + "\n"
            )
    return [json_path, jsonl_path]


def write_eligibility_report(caption_dir: Path, facts: list[CaptionFact]) -> Path:
    return _write_json(
        caption_dir / "eligibility_report.json",
        {
            "facts": [
                {
                    "fact_id": f.fact_id,
                    "fact_type": f.fact_type.value,
                    "eligibility": f.eligibility.value,
                    "eligibility_basis": (
                        f.eligibility_basis.value if f.eligibility_basis else None
                    ),
                    "eligibility_reason": f.eligibility_reason,
                    "source_kind": f.source_kind.value,
                    "human_decision_ids": f.human_decision_ids,
                    "human_fact_id": f.human_fact_id,
                }
                for f in facts
            ]
        },
    )


def write_caption_plan(caption_dir: Path, plan: CaptionPlan) -> list[Path]:
    return [
        _write_json(caption_dir / "caption_plan.json", plan.model_dump(mode="json")),
        _write_json(
            caption_dir / "overview_plan.json", plan.overview_plan.model_dump(mode="json")
        ),
        _write_json(
            caption_dir / "shot_plans.json",
            {"shots": [s.model_dump(mode="json") for s in plan.shot_plans]},
        ),
    ]


def write_reviewed_caption(caption_dir: Path, reviewed: ReviewedCaption) -> Path:
    return _write_json(
        caption_dir / "reviewed_caption.json", reviewed.model_dump(mode="json")
    )


def write_rendered_caption(
    caption_dir: Path, caption: RenderedCaption, readiness: CaptionReadiness
) -> list[Path]:
    """Filename honesty: the draft filename never overstates readiness."""
    ready_md = caption_dir / "ready_to_enter.md"
    ready_json = caption_dir / "ready_to_enter.json"
    draft_md = caption_dir / "draft_review_only.md"
    for stale in (ready_md, ready_json, draft_md):
        if stale.exists():
            stale.unlink()
    paths: list[Path] = []
    if readiness == CaptionReadiness.READY_TO_ENTER:
        paths.append(_write_text(ready_md, caption.markdown))
        paths.append(_write_json(ready_json, caption.model_dump(mode="json")))
    else:
        paths.append(_write_text(draft_md, caption.markdown))
    return paths


def write_assertion_map(
    caption_dir: Path, assertions: list[CaptionAssertionRecord]
) -> Path:
    return _write_json(
        caption_dir / "caption_assertion_map.json",
        {"assertions": [a.model_dump(mode="json") for a in assertions]},
    )


def write_coverage(caption_dir: Path, items: list[CaptionCoverageItem]) -> Path:
    return _write_json(
        caption_dir / "caption_coverage.json",
        {"items": [i.model_dump(mode="json") for i in items]},
    )


def write_seed_change_log(
    caption_dir: Path, entries: list[SeedChangeEntry]
) -> list[Path]:
    json_path = _write_json(
        caption_dir / "seed_change_log.json",
        {"entries": [e.model_dump(mode="json") for e in entries]},
    )
    lines = ["# Seed change log", ""]
    for entry in entries:
        source = "human decision" if entry.decided_by_human else "machine proposal"
        reasons = f" ({', '.join(entry.reasons)})" if entry.reasons else ""
        lines.append(
            f"- **{entry.section}** `{entry.subject_id or '-'}`: "
            f"{entry.disposition} [{source}]{reasons}"
        )
    md_path = _write_text(caption_dir / "seed_change_log.md", "\n".join(lines) + "\n")
    return [json_path, md_path]


def write_m2_report(caption_dir: Path, issues: list[ValidatorIssue], version: str) -> list[Path]:
    json_path = _write_json(
        caption_dir / "m2_validator.json",
        {
            "m2_validator_version": version,
            "issues": [i.model_dump(mode="json") for i in issues],
        },
    )
    lines = [f"M2 validator {version}", ""]
    for issue in issues:
        lines.append(
            f"{issue.severity.value:5} {issue.rule_id:20} {issue.location}: {issue.message}"
        )
    if not issues:
        lines.append("no findings")
    txt_path = _write_text(caption_dir / "m2_validator.txt", "\n".join(lines) + "\n")
    return [json_path, txt_path]


def write_platform_report(caption_dir: Path, report: PlatformSemanticReport) -> Path:
    return _write_json(
        caption_dir / "platform_semantic_report.json",
        {
            "status": report.status,
            "issues": [i.model_dump(mode="json") for i in report.issues],
        },
    )


def write_golden_gate(caption_dir: Path, result: GoldenGateResult) -> Path:
    return _write_json(caption_dir / "golden_gate.json", result.model_dump(mode="json"))


def write_adversarial_qc(caption_dir: Path, result: AdversarialQCResult) -> Path:
    return _write_json(
        caption_dir / "final_adversarial_qc.json",
        {
            "findings": [f.model_dump(mode="json") for f in result.findings],
            "missing_required_facts": result.missing_required_facts,
            "blocking_count": result.blocking_count,
        },
    )


def write_final_review_checklist(caption_dir: Path, check: SignoffCheck) -> Path:
    return _write_json(
        caption_dir / "final_review_checklist.json",
        {
            "signoff_present": check.present,
            "signoff_valid": check.valid,
            "signoff_stale": check.stale,
            "reasons": check.reasons,
        },
    )


def write_human_facts_record(
    caption_dir: Path,
    accepted: list[HumanCaptionFact],
    rejected: list[tuple[str, str]],
) -> Path:
    return _write_json(
        caption_dir / "human_facts_applied.json",
        {
            "accepted": [f.model_dump(mode="json") for f in accepted],
            "rejected": [{"fact_id": fid, "reason": reason} for fid, reason in rejected],
        },
    )


def write_final_status(caption_dir: Path, result: CaptionBrainResult) -> Path:
    return _write_json(caption_dir / "final_status.json", result.model_dump(mode="json"))


def write_review_report(run_dir: Path, sections: list[tuple[str, list[str]]]) -> Path:
    """review_report.md — reviewer rationale lives OUTSIDE caption fields (§57)."""
    lines = ["# Review report", ""]
    for title, body in sections:
        lines.append(f"## {title}")
        lines.extend(body or ["(nothing to report)"])
        lines.append("")
    return _write_text(run_dir / "review_report.md", "\n".join(lines) + "\n")


def write_caption_manifest(
    caption_dir: Path,
    entries: dict[str, str | float | dict[str, float] | dict[str, str] | None],
    artifact_paths: list[Path],
) -> Path:
    """caption_manifest.json (§93): hash every Phase 5 artifact + record input
    provenance and stage timings."""
    hashes = {}
    for path in sorted(set(artifact_paths)):
        if path.exists() and path.name != "caption_manifest.json":
            hashes[path.name] = sha256_file(path)
    return _write_json(
        caption_dir / "caption_manifest.json", {**entries, "artifact_sha256": hashes}
    )
