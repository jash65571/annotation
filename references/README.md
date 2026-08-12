# References (local only — not committed)

Raw Manuscript II project documents are **not distributed with this
repository** (distribution rights unclear). They must be supplied locally into
this directory. The engine's rule system commits only derived rule definitions
plus source metadata (`engine/manuscript_reviewer/rules/manuscript_v1.yaml`).

## Expected local documents

Currently supplied on this machine:

- `manuscript-ii-master-reference.md` — Master Reference (Learning Hub compilation)
- `MANUSCRIPT-II-REVIEWER-HANDBOOK-v1.2 (1).md` — Reviewer Handbook v1.2
- `Manuscript II.pdf` — Golden Examples export (31 pages, 2026-08-10)

Named controlling documents **not yet supplied** (rules derived from
task-specific workflow feedback until they arrive; supply them here and re-run
the rule extraction review):

- `MANUSCRIPT II Master Frame Audit Protocol v1.5`
- `MANUSCRIPT-II-PROJECT-RULES-CURRENT.md`
- `MANUSCRIPT_AUDIT_README-v3.2.md`
- `MANUSCRIPT-II-CURRENT-SOURCES.md`

When a newer document lands, follow docs/01-rule-hierarchy.md: diff against
the rule file, resolve conflicts by the priority order, record them under
`known_conflicts`, bump `rules_version`.
