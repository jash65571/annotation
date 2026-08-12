"""Phase 4 seed validators (P4-SEED-*).

* P4-SEED-001: the stored seed snapshot hash matches the source.
* P4-SEED-002: every parsed entry retains its raw source location.
* P4-SEED-003: every claim links back to a seed source location.
"""

from __future__ import annotations

from pathlib import Path

from ..artifacts.writer import sha256_file
from ..models.caption import SeedClaim
from ..models.review_intelligence import SeedClaimType, SeedDocument
from ..models.validation import Severity, ValidatorIssue

#: Claim types that are document-level derivations and legitimately have no
#: single source line.
_DOCUMENT_LEVEL = frozenset({SeedClaimType.SHOT_COUNT})


def validate_seed_snapshot(seed_dir: Path, doc: SeedDocument) -> list[ValidatorIssue]:
    issues: list[ValidatorIssue] = []
    snapshot = doc.snapshot
    if snapshot is None:
        return issues
    stored = seed_dir / "seed_original.txt"
    if not stored.is_file():
        issues.append(
            ValidatorIssue(
                rule_id="P4-SEED-001",
                severity=Severity.FAIL,
                location=snapshot.stored_relative_path,
                message="Seed snapshot file is missing.",
            )
        )
        return issues
    actual = sha256_file(stored)
    if actual != snapshot.sha256:
        issues.append(
            ValidatorIssue(
                rule_id="P4-SEED-001",
                severity=Severity.FAIL,
                location=snapshot.stored_relative_path,
                message=(
                    f"Seed snapshot hash mismatch: recorded {snapshot.sha256[:12]}, "
                    f"file {actual[:12]}."
                ),
            )
        )
    return issues


def validate_seed_document(doc: SeedDocument) -> list[ValidatorIssue]:
    issues: list[ValidatorIssue] = []
    for section in doc.sections:
        for entry in section.entries:
            if entry.source_line <= 0 or not entry.raw_line:
                issues.append(
                    ValidatorIssue(
                        rule_id="P4-SEED-002",
                        severity=Severity.FAIL,
                        location=f"entry {entry.entry_id}",
                        message="Parsed entry lost its raw source location.",
                    )
                )
    return issues


def validate_claims_source_links(claims: list[SeedClaim]) -> list[ValidatorIssue]:
    issues: list[ValidatorIssue] = []
    for claim in claims:
        if claim.claim_type in _DOCUMENT_LEVEL:
            continue
        if claim.seed_source_line is None and claim.seed_entry_id is None:
            issues.append(
                ValidatorIssue(
                    rule_id="P4-SEED-003",
                    severity=Severity.WARN,
                    location=f"claim {claim.claim_id}",
                    message="Claim does not link to a seed source entry.",
                )
            )
    return issues
