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

## Phase 4 — Review intelligence
- Seed parsing → `SeedClaim` extraction → claim-vs-evidence comparison →
  KEEP / FIX_ENRICH / REDO_REBUILD proposals with evidence.
- Character/object continuity tracking (C#/O# candidate maps).
- Caption generation via VLM/LLM adapters (propose-only; timing injected from the
  ledger, never from the model).
- M2-* caption validator (rule registry from docs/00 §29).
- Golden Example behavior gate.

## Phase 5 — Reviewer cockpit
- Tauri 2 + React + TypeScript desktop UI over the Python engine: timeline, frame
  stepping, evidence panels, per-section KEEP/FIX/REBUILD workflow, final checklist.
- Export of the finished caption for manual paste into the live tool. The app never
  claims tasks, never submits, never touches credentials.
