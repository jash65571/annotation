"""Phase 4 seed / feedback / review artifact writing.

Same determinism rules as the Phase 1/2/3 writers: UTF-8, sorted keys, exact
rationals preserved, no fake artifacts for stages that did not run.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

from ..media.timestamps import seconds_to_decimal
from ..models.caption import SeedClaim
from ..models.review_intelligence import (
    ClaimEvidenceRow,
    FeedbackDocument,
    ReviewProposal,
    ReviewQueueItem,
    SeedDocument,
    SeedTriage,
    VisualIntelligenceResult,
)
from .writer import ArtifactWriteError


def _write_json(path: Path, payload: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, ensure_ascii=False)
            handle.write("\n")
    except OSError as exc:
        raise ArtifactWriteError(f"Failed to write {path}: {exc}") from exc
    return path


def write_seed_parse(seed_dir: Path, doc: SeedDocument) -> Path:
    return _write_json(seed_dir / "seed_parse.json", doc.model_dump(mode="json"))


def write_seed_parse_issues(seed_dir: Path, doc: SeedDocument) -> Path:
    return _write_json(
        seed_dir / "seed_parse_issues.json",
        {"issues": [i.model_dump(mode="json") for i in doc.issues]},
    )


def write_seed_claims(seed_dir: Path, claims: list[SeedClaim]) -> Path:
    return _write_json(
        seed_dir / "seed_claims.json",
        {"claims": [c.model_dump(mode="json") for c in claims]},
    )


def write_seed_triage(seed_dir: Path, triage: SeedTriage) -> Path:
    return _write_json(seed_dir / "seed_triage.json", triage.model_dump(mode="json"))


def write_feedback_directives(feedback_dir: Path, doc: FeedbackDocument) -> Path:
    return _write_json(
        feedback_dir / "feedback_directives.json",
        {"directives": [d.model_dump(mode="json") for d in doc.directives]},
    )


_MATRIX_COLUMNS = [
    "claim_id",
    "claim_type",
    "seed_text",
    "seed_shot",
    "seed_source_line",
    "seed_start",
    "seed_end",
    "evidence_status",
    "importance",
    "foundational",
    "supporting_evidence_refs",
    "contradicting_evidence_refs",
    "unresolved_reasons",
    "review_proposal",
]


def _fmt(value: object) -> str:
    if value is None:
        return ""
    return str(value)


def write_claim_evidence_matrix_csv(review_dir: Path, rows: list[ClaimEvidenceRow]) -> Path:
    review_dir.mkdir(parents=True, exist_ok=True)
    path = review_dir / "claim_evidence_matrix.csv"
    try:
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(_MATRIX_COLUMNS)
            for row in rows:
                writer.writerow(
                    [
                        row.claim_id,
                        row.claim_type.value,
                        (row.seed_text or "").replace("\n", " ").strip(),
                        _fmt(row.seed_shot),
                        _fmt(row.seed_source_line),
                        str(seconds_to_decimal(row.seed_start_exact))
                        if row.seed_start_exact is not None
                        else "",
                        str(seconds_to_decimal(row.seed_end_exact))
                        if row.seed_end_exact is not None
                        else "",
                        row.evidence_status.value,
                        row.importance.value,
                        int(row.foundational),
                        ";".join(e.evidence_id for e in row.supporting_evidence_refs),
                        ";".join(e.evidence_id for e in row.contradicting_evidence_refs),
                        " | ".join(row.unresolved_reasons),
                        row.review_proposal.value if row.review_proposal is not None else "",
                    ]
                )
    except OSError as exc:
        raise ArtifactWriteError(f"Failed to write {path}: {exc}") from exc
    return path


def write_claim_evidence_matrix_json(review_dir: Path, rows: list[ClaimEvidenceRow]) -> Path:
    return _write_json(
        review_dir / "claim_evidence_matrix.json",
        {"rows": [r.model_dump(mode="json") for r in rows]},
    )


def write_review_proposals(review_dir: Path, proposals: list[ReviewProposal]) -> Path:
    return _write_json(
        review_dir / "review_proposals.json",
        {"proposals": [p.model_dump(mode="json") for p in proposals]},
    )


def write_visual_review_queue(review_dir: Path, items: list[ReviewQueueItem]) -> Path:
    return _write_json(
        review_dir / "visual_review_queue.json",
        {"items": [i.model_dump(mode="json") for i in items]},
    )


def write_visual_qc(review_dir: Path, result: VisualIntelligenceResult) -> Path:
    return _write_json(review_dir / "visual_qc.json", result.model_dump(mode="json"))
