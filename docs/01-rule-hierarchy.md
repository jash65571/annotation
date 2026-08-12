# 01 — Rule Hierarchy

## Priority order (highest wins)

1. **Current official workflow / live-tool rules / task-specific feedback** — what the
   annotation tool actually enforces today, plus reviewer feedback on specific tasks.
2. **Actual video frames and audio** — factual truth. The media decides what happened.
3. **Newest Master Frame Audit Protocol** — Master Frame Audit Protocol v1.5, now
   supplied locally and read in full; rules it supports carry FRAME-AUDIT provenance.
4. **Current Reviewer Handbook / current project rules** — HANDBOOK v1.2, MASTER-REF,
   plus MANUSCRIPT-II-PROJECT-RULES-CURRENT.md, MANUSCRIPT_AUDIT_README-v3.2.md and
   MANUSCRIPT-II-CURRENT-SOURCES.md (all now supplied locally — see
   `references/README.md`).
5. **Golden Examples** — quality level only, not syntax authority.
6. **Seed caption** — a hypothesis to verify, never truth.
7. **AI / evaluator suggestions** — leads, never commands.

Raw source documents are **local-only and gitignored** (distribution rights unclear);
the repository commits only derived rules with provenance metadata.

An older example never overrides a newer live-tool rule.

## How the hierarchy is encoded

- Machine-readable rules live in `engine/manuscript_reviewer/rules/manuscript_v1.yaml`,
  each with a provenance comment naming its source document and section.
- Conflicts between sources are recorded under `known_conflicts` with the winning rule.
- The rule file is versioned (`rules_version`); every audit run's `manifest.json`
  records the rule version used, so past runs remain interpretable after rule updates.
- Current version: **v1.3.0** (Phase 3.1). It replaced the caption-facing
  `default_transition: "Hard cut"` with `default_transition: null` +
  `unresolved_transition_is_not_hard_cut: true`: "Hard cut" stays a valid menu
  option but is never emitted merely because a transition is unresolved
  (recorded as `CONFLICT-005`, grounded in FRAME-AUDIT §12/§13/§30A.1 and
  check 11). All four named controlling documents are now supplied and read, and
  provenance has been re-mapped from `TASK-FEED` to the actual controlling
  sources wherever one directly supports the rule.

## Conflict-resolution procedure

When a new source document arrives:
1. Diff its requirements against `manuscript_v1.yaml`.
2. For each disagreement, decide by the priority order above.
3. Record the conflict in `known_conflicts` (id, topic, older, newer, winner, provenance).
4. Bump `rules_version` (semver: patch = clarification, minor = new rule,
   major = behavior reversal).
5. Never edit history — old rule files stay resolvable for old runs.

## Evidence trumps everything below line 2

Even a current-tool suggestion (Final Review, evaluator hints) is judged against the
media: **Resolve** when the video supports it, **Ignore** when it does not, and never
invent an ID, word, speaker, object, timestamp, or sound to satisfy a suggestion
(HANDBOOK §26).
