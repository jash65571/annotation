# 10 — Reviewer Cockpit (Phase 6)

The desktop application a Manuscript II reviewer uses daily: drop a video +
seed + task feedback, analyze, resolve evidence-gated review items, finalize,
sign off, copy the exact ready caption. The app never submits, never claims
tasks, never touches platform credentials — the human remains the submission
authority.

## Architecture

```
React (desktop/src)  — screens, review workflow, typed DTOs
   │  typed Tauri commands (no shell, no broad fs)
Rust (desktop/src-tauri) — process control, jobs, path validation
   │  JSONL over stdin/stdout (UI_BRIDGE_PROTOCOL_VERSION = 1)
Python engine worker (engine/manuscript_reviewer/ui_bridge)
   │  direct calls
run_audit() / finalize_run() + run-directory artifacts (the source of truth)
```

- **Rust is transport/process control.** It owns engine worker startup,
  handshake, job lifecycle, cancellation (process-tree kill), path
  validation, and the recent-runs index. There is no `run_command(string)`
  and no arbitrary executable path — only the packaged sidecar or the dev
  `uv run` module invocation can ever be spawned.
- **Python remains the factual authority.** The bridge serializes existing
  artifacts and calls existing APIs; it never re-implements caption or
  evidence logic, and machine code never fabricates human decisions.
- **React never parses CLI text.** All data arrives as typed JSON.

## Bridge protocol

One JSON object per line on stdin/stdout. Requests carry
`request_id / command / payload / protocol_version`; responses echo the id
with `status: ok|error`, a payload, or a typed error
(`ENGINE_NOT_FOUND`, `PROTOCOL_VERSION_MISMATCH`, `RUN_NOT_FOUND`,
`RUN_LOCKED`, `MANIFEST_MISMATCH`, `INVALID_DECISION`, `NOT_READY`, …).
Long jobs (`start_audit`) interleave `{"event":"progress"}` lines fed by the
additive `ProgressReporter` hooks in `pipeline.py` (default `NoOp`; CLI
unchanged; no factual algorithm may depend on it).

Commands: `health`, `engine_info`, `get_rules`, `start_audit`, `load_run`,
`get_run_summary`, `get_review_queue`, `get_review_item`, `get_shots`,
`get_frame_record`, `get_exact_frame`, `get_evidence_bundle`,
`get_audio_review_clip`, `get_waveform_metadata`, `save_review_decisions`,
`save_human_facts`, `get_review_inputs`, `save_visual_anchors`, `finalize`,
`get_caption_state`, `create_final_signoff`, `validate_final_signoff`,
`export_caption`. The Rust layer proxies only this allow-list.

Handshake: at startup (and worker restart) Rust sends `health` and refuses to
proceed on a protocol version mismatch with a blocking typed error.

## Two engine modes

- **Development**: `uv run --project <repo> python -m
  manuscript_reviewer.ui_bridge.worker`, repo located by walking up from the
  working directory to a `pyproject.toml` + `engine/manuscript_reviewer`
  marker. `npm run tauri dev` inside `desktop/`.
- **Packaged**: a PyInstaller **onedir** sidecar
  (`binaries/manuscript-engine-worker/`) built by
  `scripts/build_engine_sidecar.ps1` from the *same* worker module
  (`scripts/sidecar_entry.py`) — the engine is never forked. Bundled
  resources are declared in `tauri.packaged.conf.json`, applied only during
  `scripts/package_windows.ps1` so development builds never require them.

Packaged runtime environment (set by Rust when spawning the sidecar):

- `MANUSCRIPT_FFMPEG_DIR` → the bundled `resources/ffmpeg/bin` (the packaged
  FFmpeg always wins over an arbitrary system one; its version is recorded in
  `ffmpeg/VERSION.txt`).
- `MANUSCRIPT_ASR_WORKERS_DIR` → `<app-data>/asr_workers`. The engine copies
  the pinned worker templates (`pyproject.toml`, `worker.py`, `uv.lock`)
  there on first use and uv materializes the isolated envs in writable
  app-local data (the install dir stays read-only).
- Bundled `uv.exe` is prepended to `PATH` for the worker process only.

ASR remains local-only: bootstrap may download packages/models (clearly
surfaced in the UI); task audio never leaves the machine; there is no cloud
transcription path. "Continue without ASR" keeps waveform/spectrogram/energy
and manual audio review. OCR: if Tesseract is unavailable the app shows
"OCR unavailable — text review remains manual" (no fatal error, no silent
binary downloads).

## Security model

- Tauri capabilities (`capabilities/default.json`): `core:default`,
  `dialog:allow-open`, `dialog:allow-save`,
  `clipboard-manager:allow-write-text`. No fs scope, no shell, no HTTP, no
  updater, no clipboard read.
- Run files (evidence images, waveform PNGs, clips) are served as data URLs
  through `read_run_file`, which only serves paths inside run directories the
  engine validated **this session**, after canonicalization (traversal is
  refused). Known file types only.
- Context video playback uses the asset protocol scoped dynamically to the
  one selected video (`allow_video_playback`), never a directory.
- No telemetry, no analytics SDKs, no accounts. Crash detail goes to local
  logs only. The app works offline once dependencies/models are cached.

## Run lifecycle

`start_audit` runs in a dedicated worker process (one analysis at a time).
Progress stages stream as events; **Cancel Analysis** kills the process tree
(`taskkill /T /F`), leaving already-written artifacts auditable and the app
ready for a new job. Human-input files are written atomically (temp + fsync +
rename) into `<run>/ui/`; a per-run lock file (with stale-pid recovery)
prevents audit/finalize racing on one run. A run directory without
`manifest.json` shows as **INCOMPLETE** in recent runs and is never claimed
verified.

## Review workflow

- **Three media modes**: context playback (labeled "CONTEXT PLAYBACK — not
  frame truth"; never timing authority), exact frame mode (ledger identity:
  Phase 1 extracted PNG or on-demand single-frame extraction by decode index,
  cached under `<run>/ui/frame_cache`), and audio review (waveform +
  exact review clips).
- Exact frame stepping: ←/→ = ±1, Shift = ±10, with overlay
  `F{index} · exact time · PTS · Shot N`. UI shows decimal projections of
  exact `"num/den"` strings via BigInt math; stored values remain exact.
- **Review queue** merges visual/caption items and audio items; CRITICAL →
  HIGH → timeline order; priority filters; J/K navigation; SKIPPED is
  visually distinct and never counts as resolved.
- **Evidence panel** answers "why am I looking at this": reason, supporting
  and contradicting evidence as clickable typed cards, machine proposal
  (labeled MACHINE LEAD), current human decision, readiness effect.
- **Typed editors** produce only engine vocabulary:
  speech (verify / correct — original ASR text preserved read-only / reject),
  transition (menu loaded from `shots.allowed_transition_types` in the rules
  file; Shot 1 fixed to the opening value), playback speed
  (`regular|slow_motion|accelerated`, subject `SPEED-<shot>`), OCR text +
  timing (`first-last` frames inside one shot), identity
  (same/different/unresolved), action boundary (`start-end` frames from exact
  frame mode; ledger recomputes exact timing) + semantics (separate save),
  camera (engine `CameraMotionClass` values), proposal outcome
  (KEEP / FIX_ENRICH / REDO_REBUILD).
- **Human facts** require fact type + evidence references (current frame /
  frame range as HUMAN_VERIFICATION evidence) and a human reviewer name;
  no evidence ⇒ cannot save.
- Every decision **auto-saves and re-finalizes** immediately (engine
  `finalize_run`, no media re-analysis); the caption preview updates, a
  line-diff popover shows what changed, and gate badges (M2 / platform /
  Golden / coverage) refresh. Undo restores the previous decision snapshot,
  persists it, and re-finalizes; a local structured audit trail records every
  change.
- **Anchors**: the anchor editor stores SOURCE-pixel boxes (letterbox-aware
  conversion, unit-tested) to `<run>/ui/visual_anchors.json`. The locked
  engine's anchor-assisted tracking slice is reserved/not implemented, so
  "re-run with anchors" currently means a new analysis pass — documented as a
  known limitation.

## Final review & export

`READY_FOR_FINAL_REVIEW` opens the final screen: full rendered caption, gate
results, remaining blockers, and the FinalReviewSignoff checklist — never
pre-checked; every confirmation plus the reviewer name is human input. The
signoff binds to video SHA-256, rules version and caption SHA-256; the engine
re-validates staleness on every finalize, and any later change disables
readiness until re-signed.

**Copy Ready Caption** is enabled only at `READY_TO_ENTER`, copies the exact
`ready_to_enter.md` content via write-only clipboard (no reformatting — the
export path hashes the bytes), and "Save Caption As…" writes the same exact
artifact through the native save dialog. Draft copies are impossible: the
engine refuses `export_caption` with `NOT_READY` below `READY_TO_ENTER`.

## Status language

The UI uses the engine's terms verbatim: BLOCKED / REVIEW REQUIRED / READY
FOR FINAL REVIEW / READY TO ENTER, and distinguishes MACHINE EVIDENCE /
HUMAN VERIFIED / HUMAN CORRECTED / UNRESOLVED. There is no "Approve all",
no auto-resolve, and no confidence-percentage acceptance UI.

## Testing

- Python: full engine suite + `tests/test_ui_bridge.py` (protocol handshake,
  bad JSON, invalid command, request-id echo, human-input guards, lock
  conflict/stale recovery, atomic writes, full audit→finalize worker flow on
  a synthetic clip) + `tests/test_asr_workers_dir_override.py`.
- Rust: `cargo test` (path allow-list + traversal, recent-runs index,
  data-URL types, sidecar resolution and packaged env), `cargo fmt --check`,
  `cargo clippy -- -D warnings`.
- Frontend: Vitest + Testing Library (state machine, queue ordering/filters,
  editor vocabulary, anchor coordinate transforms, exact-time rounding).
- CI: `python-engine`, `frontend`, `rust`, `tauri-build` (Windows) stages.
  Nothing downloads Whisper models in CI.

## Known limitations (v1)

- Anchor-assisted tracking rerun requires a full new analysis (engine's
  tracking slice is reserved in the locked baseline).
- Visual queue items reference machine records heuristically (counter ids);
  the editor panel lets the reviewer pick the editor type when a mapping is
  ambiguous.
- Desktop E2E (tauri-driver) is not yet wired into CI; the packaged smoke
  test covers launch + engine + synthetic run.
