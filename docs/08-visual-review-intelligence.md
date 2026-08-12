# 08 — Visual Review Intelligence (Phase 4)

**The seed is never truth. Visual models are never truth. OCR is never truth.
Tracking is never truth. Similarity is never identity truth. Actual media
remains factual truth.** Phase 4 turns the deterministic Phase 1–3 evidence plus
the seed and task feedback into structured *reviewer intelligence*. It produces
no final caption prose, makes no cloud/media upload, and its default runtime is
fully local.

Pipeline:

```
MEDIA + FRAME LEDGER + SHOT TRUTH + AUDIO TRUTH + SEED + TASK FEEDBACK
  → SEED SNAPSHOT → SEED PARSE → ATOMIC SEED CLAIMS
  → FRAME-LEVEL VISUAL OBSERVATIONS
  → STRUCTURAL COMPARISON (seed vs media)
  → CLAIM ↔ EVIDENCE MATRIX
  → KEEP / FIX_ENRICH / REDO_REBUILD PROPOSALS
  → REVIEW QUEUE → SEED TRIAGE → VISUAL QC
```

The orchestrator is `visual_intelligence.py::run_visual_intelligence`, wired into
`pipeline.py` behind `--visual-intelligence` (default on). It runs after media,
shots, and audio; it never upgrades a run to PASS and never prints
`CAPTION VERIFIED` / `EXPORT READY`.

## Three vocabularies, never conflated

Phase 4 keeps three concepts strictly separate (`models/review_intelligence.py`):

| Concept | Type | Values |
| --- | --- | --- |
| What the media proves about a claim | `EvidenceStatus` | SUPPORTED / CONTRADICTED / PARTIALLY_SUPPORTED / UNRESOLVED / NOT_APPLICABLE |
| The machine's *proposal* for a section | `ReviewProposalOutcome` | KEEP / FIX_ENRICH / REDO_REBUILD / HUMAN_DECISION_REQUIRED |
| An actual human decision | `caption.ReviewDecision` / `HumanReviewDecision` | recorded only from a human-supplied file |

Evidence status is never KEEP/FIX/REDO. A machine proposal is never a human
decision (`proposed_by == "machine"`, validator P4-REVIEW-001).

## Seed snapshot (immutable evidence)

`seed/snapshot.py` copies the seed byte-for-byte to `seed/seed_original.txt`,
writes `seed/seed_sha256.txt`, and records `SeedSnapshot`. The original is never
normalized, repaired, or reordered. Parsing works from a decoded *copy* of these
bytes; validator **P4-SEED-001** re-hashes the stored file against the recorded
digest. Task feedback is snapshotted the same way (`feedback/feedback_original.txt`).

## Seed parser

`seed/parser.py` accepts Markdown, plain text, or copied editor output and
produces a recoverable `SeedDocument` (`SeedSection` → `SeedEntry`, plus
`SeedParseIssue`). It recognizes the Overview sections (Characters/Cast, Objects,
Scene, Style, Audio, Visual concerns, Audio concerns) and per-shot fields
(Start, End, Cut/Transition, Camera, Camera Movements, Scene, Action & Audio,
Playback speed, Speed Changes), the `[Shot N: start–end]` header, and the
`Video ID`.

Robustness contract:

- The original bytes are never mutated.
- A line is never discarded because parsing failed — malformed content is kept
  as a FREEFORM entry (validator **P4-SEED-002**: every entry keeps its raw
  source line + line number).
- Malformed timestamp syntax is **not** silently repaired: `timestamp_text` is
  preserved, `parsed_*` stays `None`, and a `SeedParseIssue` is recorded.

Timestamp parsing (`parse_time_token` / `find_time_range`) accepts `12`, `12.3`,
`4.3s`, `00:03.4`, `1:08.5`, en-dash/em-dash/hyphen/`to` ranges, and leading
bracketed point events `(00:03.4)`. All times are exact `Fraction`s — never
floats. Bare numbers inside prose are deliberately not treated as timestamps
(false-positive defense).

## Atomic seed claims

`seed/claims.py` extends `caption.SeedClaim` (additive, optional fields) and
extracts *atomic* claims — a paragraph is never one giant claim. Claim types:
MEDIA_ID, SHOT_COUNT, SHOT_BOUNDARY, TRANSITION, CHARACTER_EXISTS,
CHARACTER_TRAIT, OBJECT_EXISTS, SCENE_STATE, STYLE_STATE, CAMERA_FRAMING,
CAMERA_MOVEMENT, ACTION, SPEECH, SOUND, ON_SCREEN_TEXT, PLAYBACK_SPEED,
VISUAL_CONCERN, AUDIO_CONCERN, and PROTECTED_TRAIT. Each claim keeps its source
line, shot number, exact seed time range, subject/object IDs, and quoted text.

**Foundational vs local importance** drives rebuild-vs-patch decisions
(`ClaimImportance`). Foundational: media identity, shot count, shot boundaries,
transitions, major character/object identity. Local: one clothing detail, one
overlay line, one camera phase, one concern, one action wording.

**Protected/unsupported traits** (nationality, ethnicity, race, exact age,
gender, accent) are captured as `PROTECTED_TRAIT` claims so they can be flagged
unsupported. They are **never** visually inferred — the CV side builds no such
classifier (semantic-safety guarantee), and comparison always leaves them
`UNRESOLVED`.

## Structural comparison (the high-value, zero-CV wins)

`seed/comparison.py` verifies the seed against deterministic Phase 1–3 truth:

- **Shot count**: seed shot count vs `ShotTruthResult.proposed_shot_count`. A
  mismatch is `CONTRADICTED` (reason `SHOT_COUNT_CONTRADICTION`) and drives a
  shot-section REDO_REBUILD.
- **Shot boundaries**: seed shot N vs the verified `ShotProposal` N (compared at
  the Manuscript 0.1 s display precision). SUPPORTED / PARTIALLY_SUPPORTED /
  CONTRADICTED; a seed shot with no verified counterpart is CONTRADICTED
  (`SHOT_BOUNDARY_FOUNDATION_INVALID`).
- **Transitions**: shot 1 must be `Opening shot`; shots > 1 compare to the
  verified transition. An unresolved transition stays UNRESOLVED — it is **never**
  confirmed as Hard cut.
- **Timestamp containment**: a shot-scoped claim whose seed time falls outside
  the verified shot interval is CONTRADICTED (consistency is necessary, not
  sufficient — it can only contradict, never upgrade a semantic claim).
- **C/O reference integrity**: characters/objects referenced but never defined
  (foundation contradiction) and defined-but-never-referenced ghost IDs.

Semantic claim types are deliberately left `UNRESOLVED` here — they require the
later visual slices or human verification. Every SUPPORTED/CONTRADICTED claim
carries an `EvidenceReference` (validators **P4-CLAIM-001/002**).

## Claim ↔ evidence matrix, proposals, queue, triage

- `review/claim_evidence_matrix.{csv,json}`: one row per claim with status,
  importance, foundational flag, supporting/contradicting evidence, unresolved
  reasons, and the machine proposal.
- `review/proposals.py`: KEEP / FIX_ENRICH / REDO_REBUILD / HUMAN_DECISION_REQUIRED
  at claim/shot/overview/seed levels with structured reason codes. **KEEP** is
  only proposed with a supported foundation and no material contradiction
  (validators **P4-REVIEW-003 / P4-CLAIM-003**); **REDO_REBUILD** always carries
  a structural reason code (**P4-REVIEW-002**).
- `review/queue.py`: the visual review queue (CRITICAL / HIGH / NORMAL / LOW).
  Low-priority items are never hidden. Every item states what/why/where +
  supporting and contradicting evidence + a recommended reviewer action.
- Seed triage (`seed_triage.json`): the early PATCH / REBUILD / REVIEW_REQUIRED
  recommendation ("minute-8" whole-seed keep/fix/rebuild judgment). An early
  recommendation only; it never blocks later correction.

Task feedback is higher priority than seed content and is never over-interpreted:
only exact known patterns map to a check code (e.g. `REQUIRE_VOCAL_LYRIC_REVIEW`);
everything else stays `REVIEW_REQUIRED`.

## Human decisions (re-run durable)

`review/decisions.py` loads human decisions bound to the video SHA-256 and rules
version. A decision from another video or after a media/rules change is detected
as **stale** and never applied. Machine code never fabricates a human decision:
a decision file with a machine/empty `decided_by` is rejected.

## Frame observations and the shared frame cache

`visual/decode.py::FrameCache` is the single bounded frame-access layer: the gray
metric grid is decoded once (reusing the Phase 2 count==ledger guarantee so image
N is always ledger frame N), and full-resolution colour frames are decoded on
demand into a bounded LRU. No consumer performs an independent full decode.

`visual/observations.py` emits one `FrameObservation` per source frame
(`visual/frame_observations.{csv,jsonl}`) with exact identity (frame_index →
source PTS → annotation time), deterministic brightness/contrast/sharpness/motion
metrics, an RNG-free FFT phase-correlation global-motion estimate, and
deterministic visual-concern candidates (`visual/concerns.py`). Validators
**P4-OBS-\*** keep the observation ledger exactly aligned with the frame ledger.

`visual/enriched_frame_ledger.csv` is a derived, human-friendly export that
combines Phase 1 identity/timing with the Phase 4 observations; every column is
labelled by provenance (DETERMINISTIC vs CANDIDATE). **Phase 1 `frames.csv` /
`frames.jsonl` are never overwritten.**

## Safety invariants (executable validators)

- **P4-PRIVACY-001** — no cloud media path exists. Phase 4 performs no network
  I/O; all shell-outs go through the Phase 1 safe `run_tool` layer.
- **P4-REVIEW-001** — a machine proposal can never be stored as a human decision.
- **P4-OCR-001 / P4-TEXT-001** — machine OCR text is never caption-eligible
  without human source verification (`SourceTextVerificationStatus`).
- **P4-QC-001** — top-level PASS is forbidden while a CRITICAL review item
  remains. Phase 4 is a review-preparation stage: with any review item it reports
  REVIEW_REQUIRED.

## Future visual-reasoner contract (design only)

A future `VisualReasonerAdapter` receives only selected evidence (exact frame
IDs, crops, short strips, a structured question, existing hypotheses) and returns
a proposal with supporting/contradicting **frame IDs** and an uncertainty. **It
can never supply a timestamp** — timestamps come from frame identity, and the
engine derives display time. No cloud call is implemented; the interface is a
contract only.

## Delivered vs planned

**Delivered (slices 1–2):** seed snapshot/parser/claims/feedback, structural
comparison, claim↔evidence matrix, proposals, review queue, triage, human-decision
persistence, the frame cache, the frame-observation ledger + concerns, the P4
validators listed above, CLI + manifest integration.

**Designed, staged for later slices:** OCR (adapter + Tesseract + temporal
consensus + timing + caption-eligibility gate), local anchor-assisted
tracking / character & object continuity, ownership/contact events, final-state
checks, camera global-motion segmentation, action boundaries, playback-speed
evidence, and high-risk evidence bundles. The data model for all of these already
exists in `models/review_intelligence.py`; each slice plugs into the tested
orchestrator harness.
