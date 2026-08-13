# 04 — Development Roadmap

## Phase 1 — Deterministic foundation (DONE)
Video → ffprobe verification → every-frame ledger → evidence manifest → validation →
audit artifacts. Exact rational timing, SHA-256 provenance, independent frame-count
cross-check, hard FAIL/WARN semantics, full test/lint/typecheck gates.

## Phase 2 — Shot truth (DONE — see docs/06)
- Adjacent-pair deterministic metrics on every frame pair (MAD, histogram,
  phash, edges, Farneback flow) + ffmpeg scdet as independent evidence.
- Recall-first multi-signal candidate generation, ±0.5 s robust local
  baselines, flash/fade/blend region evidence.
- Deterministic adversarial verifier with structured reason codes →
  SUPPORTED / REJECTED / REVIEW_REQUIRED; no minimum shot length anywhere.
- Gapless shot proposals with exact frame ownership; Hard cut / Fade in /
  Fade out proposed automatically, dissolves conservative, jump/match/smash/
  L/J-cut deferred; labeled evidence bundles from exact frame identity.
- Deferred from the original Phase 2 sketch: audio waveform/energy extraction
  (moved into Phase 3 with the rest of the audio engine).

## Phase 3 — Audio truth (DONE — see docs/07)
- Annotation timeline origin + non-zero-PTS normalization (clock.py).
- Audio frame ledger, exact sample↔PTS↔annotation mapping, PCM cross-checks.
- source.wav / asr.wav, 10 ms energy, waveform/spectrogram/energy renders.
- faster-whisper (1.2.1, default large-v3-turbo) + WhisperX (3.4.3) forced
  alignment in isolated uv worker envs; transcribe-only; wording preserved;
  full local failure routes, zero cloud/Descript paths.
- Speech regions with pause splitting, VAD recall defense, language safety,
  singing/overlap review states, clip-edge flags.
- Shot-boundary audio continuity resolving Phase 2's audio flags (L/J never
  auto-finalized); review queue + exact-sample review clips.
- Deferred to Phase 3.x/4: diarization (adapter contract fixed), OCR,
  camera-movement phases, playback-speed analysis.

## Phase 4 — Visual review intelligence (DONE — see docs/08)
Under an independent review-findings pass; not declared complete until every
hardening item passes review (evidence→claim reconciliation, typed decision
registries, anchor/tracking/OCR provenance, expanded validators). Delivered:
- Immutable seed snapshot → robust recoverable parser → atomic `SeedClaim`
  extraction (typed, foundational/local) → structural comparison against Shot
  Truth / media → claim↔evidence matrix → KEEP / FIX_ENRICH / REDO_REBUILD /
  HUMAN_DECISION_REQUIRED proposals with structured reason codes.
- Task-feedback structuring, review queue, seed triage, media/rules-bound
  human-decision persistence.
- Shared bounded frame cache + per-frame visual observation ledger + concerns +
  provenance-tagged enriched ledger.
- P4-SEED/CLAIM/REVIEW/OCR/QC/OBS validators; three separate status vocabularies
  (evidence status ≠ machine proposal ≠ human decision).

- Streaming exact-frame decoder; OCR (adapter + Tesseract + region detection +
  temporal consensus + timing + failure accounting + caption-eligibility gate).
- Anchor-seeded local tracking, character/object continuity, ownership/contact
  events, mandatory final-state checks, atomic action-boundary candidates.
- Camera global-motion segmentation with hysteresis, playback-speed evidence,
  high-risk visual evidence bundles, and the design-only `VisualReasonerAdapter`.
- The full P4-* validator suite (SEED/CLAIM/REVIEW/OCR/QC/OBS/CAMERA/ENTITY/
  OBJECT/ACTION/FINAL/SPEED/PRIVACY/TEXT); three separate status vocabularies
  (evidence status ≠ machine proposal ≠ human decision).

Phase 4 produces structured evidence and reviewer proposals only — no final
caption prose.

## Phase 5 — Caption Brain (DONE — see docs/09)
- Central caption-eligibility policy (`CaptionEligibility` distinct from
  `EvidenceStatus`); CANDIDATE ≠ FINAL FACT enforced structurally, with
  provenance inspection (a mutated enum alone never proves human verification).
- Typed `CaptionFact` graph → `CaptionPlan` → deterministic renderer producing
  the Manuscript II structure; every rendered assertion maps back to fact IDs.
- Human-added `HumanCaptionFact` input (`--human-facts`), bound to video
  SHA-256 + rules version; stale facts rejected; machine code never creates one.
- M2-* caption validator, platform-semantic validator, derived
  `golden_behavior_v1.yaml` gate, coverage/omission ledger, assertion
  (hallucination) map, final adversarial QC, task-feedback gate.
- Ready states BLOCKED / REVIEW_REQUIRED / READY_FOR_FINAL_REVIEW /
  READY_TO_ENTER; bound `FinalReviewSignoff` (never fabricated); honest
  filenames (`ready_to_enter.md` only when truly ready, else
  `draft_review_only.md`).
- Fast `manuscript-reviewer finalize RUN_DIR` re-finalization from existing
  Phase 1-4 evidence — no media re-analysis; sub-second to a few seconds.
- No cloud upload, no platform submission, no result-code generation.

## Phase 6 — Reviewer cockpit (IN REVIEW — see docs/10)
- Tauri 2 + React + TypeScript desktop UI over the Python engine: timeline, frame
  stepping, evidence panels, per-section KEEP/FIX/REBUILD workflow, human-decision
  and human-fact capture, final-review signoff, final checklist.
- Export of the finished caption for manual paste into the live tool. The app never
  claims tasks, never submits, never touches credentials.
- Implemented on branch `phase-6-reviewer-cockpit`: JSONL UI bridge (protocol v1)
  with additive progress hooks, Rust process/job control with least-privilege
  capabilities, full review workstation, packaged sidecar + FFmpeg + uv bundling
  scripts. Marked complete only at final lock after independent review.
