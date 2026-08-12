# 04 — Development Roadmap

## Phase 1 — Deterministic foundation (DONE)
Video → ffprobe verification → every-frame ledger → evidence manifest → validation →
audit artifacts. Exact rational timing, SHA-256 provenance, independent frame-count
cross-check, hard FAIL/WARN semantics, full test/lint/typecheck gates.

## Phase 2 — Shot truth
- Adjacent-frame difference pass (OpenCV/PyAV): per-pair deterministic metrics
  written as evidence artifacts.
- Cut candidate generation with exact frame anchors; adversarial verification step
  that must look at the actual adjacent frames (flash-frame, fade, whip-pan traps).
- True `Shot` records with transition classification from the fixed menu; frame-strip
  evidence for every boundary.
- Audio extraction: waveform/energy artifacts, exact sample↔PTS anchoring.

## Phase 3 — Event evidence
- ASR adapter interface (faster-whisper first, local-only default); word-level
  timestamps snapped to media time; `[inaudible]` handling per current rules.
- Diarization + overlap preservation.
- Camera-movement phases from optical flow (timestamped, direction-segmented).
- OCR for on-screen text with frame anchors.
- Playback-speed analysis.

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
