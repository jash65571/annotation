# References (local only — not committed)

Raw Manuscript II project documents are **not distributed with this
repository** (distribution rights unclear). They must be supplied locally into
this directory. The engine's rule system commits only derived rule definitions
plus source metadata (`engine/manuscript_reviewer/rules/manuscript_v1.yaml`).

## Expected local documents

All seven controlling sources are supplied on this machine (as of the Phase 3.1
finalization, 2026-08-12) and have been read in full; the rule file's provenance
is re-mapped to the actual controlling documents (see `manuscript_v1.yaml`).
Two files carry download-artifact `(n)` suffixes — the exact on-disk names are:

- `manuscript-ii-master-reference.md` — Master Reference (Learning Hub compilation)
- `MANUSCRIPT-II-REVIEWER-HANDBOOK-v1.2 (1).md` — Reviewer Handbook v1.2
- `Manuscript II.pdf` — Golden Examples export (31 pages, 2026-08-10)
- `MANUSCRIPT-II-MASTER-FRAME-AUDIT-PROTOCOL-v1.5(4).txt` — Master Frame Audit Protocol v1.5
- `MANUSCRIPT-II-PROJECT-RULES-CURRENT.md` — Standing Project Memory Rules
- `MANUSCRIPT_AUDIT_README-v3.2(2).md` — Audit Pipeline v3.2 README
- `MANUSCRIPT-II-CURRENT-SOURCES.md` — Current Project Sources index

When a newer document lands, follow docs/01-rule-hierarchy.md: diff against
the rule file, resolve conflicts by the priority order, record them under
`known_conflicts`, bump `rules_version`.
