# 09 — Caption Brain (Phase 5)

Turns the reviewed Phase 1-4 evidence into a complete Manuscript II caption
while preserving the evidence-first safety model. The first phase allowed to
create final caption prose — and the phase where **CANDIDATE ≠ FINAL FACT** is
enforced in code, not comments.

```
SHOT TRUTH + AUDIO TRUTH + VISUAL TRUTH + SEED CLAIMS + TASK FEEDBACK
+ HUMAN REVIEW DECISIONS + HUMAN CAPTION FACTS
  → CAPTION ELIGIBILITY (one central policy)
  → CAPTION FACT GRAPH → CAPTION PLAN → DETERMINISTIC RENDER
  → M2-* VALIDATOR → PLATFORM-SEMANTIC VALIDATOR → GOLDEN BEHAVIOR GATE
  → COVERAGE/OMISSION + ASSERTION/HALLUCINATION PASS → ADVERSARIAL QC
  → BLOCKED | REVIEW_REQUIRED | READY_FOR_FINAL_REVIEW | READY_TO_ENTER
```

Orchestrator: `caption_brain.py::finalize_run` — wired into `pipeline.py`
behind `--caption-brain` (CLI default on; library opt-in) and exposed directly
as `manuscript-reviewer finalize RUN_DIR`.

## Eligibility (caption/eligibility.py)

`CaptionEligibility` (ELIGIBLE / REVIEW_REQUIRED / INELIGIBLE / REJECTED)
answers "may this become final caption text?" — deliberately distinct from
Phase 4 `EvidenceStatus` ("what does the evidence say?"). One resolver holds
the whole matrix; builders never decide eligibility.

Every ELIGIBLE fact names an `EligibilityBasis`: deterministic factual
evidence, an APPLIED bound `HumanReviewDecision`, explicit HUMAN_VERIFICATION
evidence, a human-added fact, or a controlling-source rule.

**Provenance over enums (§4).** Phase 4 human decisions mutate models (e.g. a
PLAYBACK_SPEED decision writes `REGULAR_SUPPORTED`). Phase 5 never trusts the
mutated enum alone: a `REGULAR_SUPPORTED` conclusion without a matching APPLIED
decision (or human-verification evidence) stays REVIEW_REQUIRED. The same holds
for `ActionCandidate.semantic_label`, camera classes, and claim statuses.

Never final on their own: REGULAR/SLOW_MOTION/ACCELERATED_CANDIDATE,
OBJECT_PICKUP_CANDIDATE and every generic ActionCandidate, machine OCR text,
ASR text/language/diarization, identity-ambiguous or reacquired tracks,
SAME_ENTITY_CANDIDATE links, unresolved camera semantics (2D scale ≠
zoom/dolly), audio-continuity evidence as L-cut/J-cut, visual expression
guesses, protected traits inferred from appearance.

## Fact graph (caption/facts.py) and human facts (caption/human_facts.py)

`CaptionFact` is a typed record (type, shot, C/O ids, exact interval, display
projections, text/semantic values, evidence refs, eligibility + basis + reason,
source kind, decision/fact provenance, materiality). Prose starts from facts.

`HumanCaptionFact` (`--human-facts`) supports REDO cases where required facts
were missing from the seed. Loaded only from a human-supplied file; must carry
a human `decided_by` and be bound to the video SHA-256 + rules version
(optionally seed SHA); stale entries are rejected and recorded. A human SPEECH
fact can also *enrich* a machine speech region (`semantic_value.region_id`:
speaker/tone/off-screen/language level) or *split* one at a shot boundary
(`splits_region_id`) — cross-shot speech is otherwise blocked, never rendered
across a cut.

Timing authority (§49): shots = Shot Truth exact intervals; speech/sound =
source-audio verified timing; OCR = verified TextTrack frames; actions/camera =
verified frame boundaries. Seed timestamps are context only, never timing. All
display values are `to_manuscript_display` (ROUND_HALF_UP) projections of exact
annotation times — no local rounding exists in Phase 5 (swept by test).

## Plan (caption/planning.py)

`CaptionPlan` = `OverviewPlan` (characters/objects in id order, scene, style,
overview audio, concerns) + per-shot `ShotPlan`s (exact interval, transition
resolution, camera framing vs camera movements, scene, the ordered Action &
Audio event union, playback speed, verified speed changes) + fact dispositions
+ the seed KEEP/FIX_ENRICH/REDO_REBUILD summary (machine proposals marked as
such — a proposal is never a human decision). Shot structure comes from Shot
Truth only; an unresolved transition is never defaulted to Hard cut; the event
union structurally excludes camera movement; overlap is preserved, never
flattened or nudged.

## Renderer (caption/renderer.py)

Deterministic; consumes the plan and never rediscovers facts. It can join
approved facts into sentences but cannot invent adjectives/objects/traits,
change ids/timing/speakers/quotes/transitions/speeds, or add actions. Speech
follows the language ladder structurally (VERBATIM → NAMED_LANGUAGE →
LANGUAGE_FAMILY → "a foreign language" → indiscernible description); verbatim
text preserves stutters/cutoffs and the original language (never translated);
`[inaudible]` is the only recoverable-gap token. Sung, verified lyrics render
like speech; unverified vocals never gain invented lyrics. On-screen text is
one quoted string per simultaneous overlay, exact wording preserved (no case or
punctuation "fixes"). Empty concerns render the exact configured literal
`None.`. Every non-structural line records its fact ids → the assertion map is
produced structurally as the caption is written.

A future `CaptionWordingAdapter` (§51) may propose wording from
caption-eligible facts only; no network/cloud provider is implemented, and the
deterministic renderer remains the CI path. Phase 5 uploads nothing anywhere.

## Validators

- **M2-*** (`validation/caption_validator.py`, versioned): MEDIA (video id /
  wrong-video), STRUCT (all verified shots, order, gapless coverage, canonical
  endpoint), TRANSITION (menu from `manuscript_v1.yaml`, Opening shot rules,
  unresolved blocks, L/J need human basis), CAST/OBJECT (ghost/undefined ids,
  first-appearance order, pronoun blocklist outside quotes, lower-body
  visibility contradiction), TIME (canonical ROUND_HALF_UP display honesty —
  fake nudges are structurally impossible, windows inside shots,
  `M2-TIME-COLLISION` for same-display different-exact pairs), ACTION (one
  fact per line, no camera movement), SPEECH (deprecated `<unintelligible>`,
  speaker required, no vague filler), AUDIO (filler blocklist, no speech in
  Overview Audio, no "transcript" in Audio concerns), TEXT (one quote), CAMERA
  (framing vs movement separation), SPEED (allowed values, human-verified
  only), FIELD (reviewer notes, dynamic action in Scene/Style, bare
  left/right), SOURCE (assertion + omission gates, feedback). FAIL ⇒ BLOCKED;
  WARN ⇒ REVIEW_REQUIRED.
- **Platform-semantic** (`validation/platform_semantic_validator.py`):
  export-blocker defense using the structured plan first, text second — multi
  sentence (abbreviation/decimal/quote-safe splitter), multiple quote spans
  (apostrophes never confuse the parser), multiple speech acts, two independent
  subject actions (connective defense — the words are not blindly banned),
  mixed events, duplicate displayed pairs, artificial nudging.
- **Golden behavior gate** (`validation/golden_validator.py` +
  `rules/golden_behavior_v1.yaml`): behavior rules derived from the full
  31-page Golden PDF with per-rule provenance (raw PDF stays local-only).
  Categories: DETAIL_COVERAGE, EVENT_GRANULARITY (0.1 s events retained, no
  quota), TIMESTAMP_DISCIPLINE, TRUTHFUL_OVERLAP, CHARACTER/OBJECT_CONTINUITY,
  CAMERA_SEPARATION, DIALOGUE/AUDIO/TEXT_COVERAGE, FINAL_STATE,
  SCENE_RECONSTRUCTABILITY (evidence readiness, never word count),
  EXPORT_SAFETY → PASS / REVIEW_REQUIRED / FAIL. Floor, not ceiling.
- **Coverage & hallucination** (`caption/coverage.py`): every material
  ELIGIBLE fact is rendered or carries a valid explicit `OmissionReason`
  (REQUIRED facts can never be omitted); every assertion maps to eligible
  facts. "The writer forgot it" fails.
- **Final adversarial QC** (`validation/final_caption_validator.py`): the
  finished caption is treated as potentially wrong — unmapped assertions,
  ineligible evidence, omissions, unresolved speech/transition/speed riding
  under the caption, every M2 FAIL → `final_adversarial_qc.json`.

## Signoff and ready states

`FinalReviewSignoff` (§54) binds reviewer identity + confirmations to the video
SHA-256, rules version, and the rendered caption's SHA-256. Machine code never
fabricates it; it loads from `--final-review`. Any change to caption content,
rules, or video makes an old signoff stale (§91).

- **BLOCKED** — any M2/platform FAIL, wrong media, invalid timeline.
- **REVIEW_REQUIRED** — unresolved required facts, unverified
  speech/OCR/speed/transition, unresolved HIGH task feedback, golden gate not
  passing, blocking adversarial findings.
- **READY_FOR_FINAL_REVIEW** — all machine gates pass; no signoff yet.
- **READY_TO_ENTER** — plus a valid bound signoff whose caption hash matches.

`ready_to_enter.md`/`.json` are written ONLY at READY_TO_ENTER; otherwise the
draft is `draft_review_only.md` (stale ready files are deleted on downgrade).
The CLI never prints EXPORT READY unless READY_TO_ENTER.

## Artifacts (run_dir/caption/) and manifest

caption_facts.json/.jsonl, eligibility_report.json, caption_plan.json,
overview_plan.json, shot_plans.json, reviewed_caption.json,
draft_review_only.md | ready_to_enter.md/.json, caption_assertion_map.json,
caption_coverage.json, seed_change_log.json/.md, m2_validator.json/.txt,
platform_semantic_report.json, golden_gate.json, final_adversarial_qc.json,
final_review_checklist.json, human_facts_applied.json (only when supplied),
final_status.json, caption_manifest.json — plus run-level review_report.md
(reviewer rationale lives outside caption fields). `caption_manifest.json`
hashes every Phase 5 artifact and records caption-brain/rules/golden versions,
input SHAs (video/seed/decisions/human-facts/signoff), the rendered caption
SHA, final status, and stage timings. The run manifest additionally records the
Phase 5 inputs and `caption_final_status`.

## Phase 5.1 hardening

- **Canonical video identity comes from media truth** (manifest
  `source_video_path` / the pipeline's video file), never the seed. The seed's
  claimed id is kept separately; a mismatch is `M2-MEDIA-004` FAIL — a
  wrong-video seed can never validate against itself.
- **Every consumed Phase 1-4 evidence artifact is hash-verified** against
  `manifest.json` before parsing (shot/audio QC, seed claims/parse, proposals,
  feedback, camera/speed/OCR/action/track/link files, frames.jsonl). A hash
  mismatch or a present-but-unmanifested evidence file raises; the consumed
  hashes are recorded in `caption_manifest.json → evidence_sha256`.
- **Phase 4 review carry-forward**: machine review items are recomputed from
  the CURRENT (post-decision) evidence at finalize time; CRITICAL/HIGH items
  gate readiness (a resolved item disappears; NORMAL/LOW items never demand
  one click per weak candidate).
- **`resolution_required`** on CaptionFact: material media content (audible
  unverified speech, a material multi-frame unverified overlay, unresolved
  transition/speed, any human-added fact) blocks readiness even when
  INELIGIBLE — eligibility alone never decides materiality. The Golden gate's
  TEXT_COVERAGE / DETAIL_COVERAGE downgrade on blocked material.
- **New typed human decisions** resolving REAL evidence records (no duplicate
  human fact needed): `SPEECH_VERIFICATION`/`SPEECH_CORRECTION` (SpeechRegion;
  originals preserved), `TEXT_VERIFICATION`/`TEXT_CORRECTION`/`TEXT_TIMING`
  (TextTrack; raw OCR never overwritten), `TRANSITION_CLASSIFICATION`
  (ShotProposal; menu-validated, Opening-shot rules enforced).
  `IDENTITY_MAPPING` now resolves BOTH `reacquired` and `identity_ambiguous`
  with recorded previous state + human evidence. `ACTION_BOUNDARY` recomputes
  `start_exact`/`end_exact` from the frame ledger — frame indices with stale
  exact timing are INVALID_VALUE.
- **Human facts must carry evidence**: a bound `HumanCaptionFact` with no
  evidence reference is REVIEW_REQUIRED, plus type-specific validation (shot
  exists, timed facts carry exact ranges, speed/transition values from the
  menu, speech has a speaker, protected traits need an explicit allowed source
  with human verification — appearance alone is never sufficient).
- **`reviewed_caption.json` mirrors the render**: playback speed, camera
  events, speech/sound/on-screen-text event records, per-shot camera/scene
  descriptions; ActionAudioEvent text carries no duplicated timestamps.

## Fast finalize, performance, failure routes

`manuscript-reviewer finalize RUN_DIR [--review-decisions ...] [--human-facts
...] [--final-review ...]` reloads existing Phase 1-4 evidence (verifying
manifest hashes for the artifacts it consumes), applies current human inputs
through the Phase 4 typed decision layer, rebuilds facts/plan/caption, and
re-runs every gate. No FFmpeg, no frame decoding (§123); measured re-finalization
of the CI fixtures runs in well under a second, with per-stage timings recorded
(`load_evidence`, `fact_graph`, `planning`, `render`, `coverage`,
`m2_validation`, `platform_validation`, `golden_gate`, `adversarial_qc`,
`caption_brain_total`).

Failures are truthful: unverified dialogue/speed/transition → REVIEW_REQUIRED;
wrong media / structural M2 FAIL → BLOCKED; tampered evidence (manifest hash
mismatch) → `CaptionBrainError`. A missing field is never filled with a guess
to reach READY. Unresolved review never crashes the pipeline — it produces
REVIEW_REQUIRED plus the reviewer packet and a draft when safe.

## Known limitations

- Speech pause-splitting relies on Phase 3 region granularity; a >0.5 s pause
  inside one verified region requires a human split (regions are never merged).
- Speaker attribution always requires human input (diarization labels are never
  C-IDs) — by design, every dialogue line is blocked until attributed.
- Camera wording for human-classified 2D motion is deliberately conservative;
  stronger verified wording must come from a human fact.
- The seed→fact reuse layer works at the atomic-claim level; sentence-level
  polish of KEEP wording beyond claim text is left to the reviewer/Phase 6.
- Golden gate categories are behavior heuristics over the structured plan; the
  mandatory human Golden comparison remains part of the signoff checklist.
