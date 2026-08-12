# 02 — System Architecture

## Prime directive: AI must not own the clock

Language/vision models may **classify and describe** ("this looks like a cut",
"C1 begins raising the hand here"). Deterministic media evidence determines the exact
candidate frame, the neighboring frames, the actual frame timestamps, and the final
boundary. Every timing value in the system traces back to integer PTS × rational
time_base from the media pipeline — never to model output.

## Monorepo layout

```
manuscript-reviewer/
├── references/            # authoritative Manuscript II source documents
├── docs/                  # product + engineering documentation
├── engine/manuscript_reviewer/   # Python 3.12 analysis engine (UI-independent)
│   ├── cli.py             # Typer CLI (thin; no logic)
│   ├── pipeline.py        # audit orchestration
│   ├── media/             # ffmpeg_tools (single subprocess wrapper), probe, frames, timestamps
│   ├── models/            # Pydantic v2 persistent schemas (common, media, frame, evidence, caption, validation, run)
│   ├── rules/             # versioned YAML rule set + typed loader
│   ├── validation/        # media_validator, ledger_validator
│   └── artifacts/         # deterministic artifact writer
├── tests/                 # pytest suite; synthetic fixtures generated with ffmpeg
├── scripts/               # developer utilities
└── artifacts/             # run outputs: artifacts/<video_stem>/<run_id>/
```

Structural changes vs the brief (documented here as required):
- Added `media/ffmpeg_tools.py` — the single safe subprocess wrapper (brief requires one).
- Added `pipeline.py` — orchestration lives outside the CLI so a future Tauri UI can
  call the engine without Typer.
- Added `models/common.py` (ExactFraction serialization) and `models/run.py`
  (RunManifest/ProjectRun) — run/reproducibility models didn't fit the other files.
- `tests/test_pipeline.py` added beyond the four listed test files.

## Phase 1 vertical slice (implemented)

```
VIDEO
  → ffprobe MEDIA VERIFICATION      (probe.py: container + per-stream facts, kept separately)
  → EVERY ENCODED FRAME             (frames.py: ffprobe -show_frames, decoder order)
  → FRAME LEDGER                    (FrameRecord[]: index, pts, exact rational time, ...)
  → EVIDENCE MANIFEST               (manifest.json: hashes, versions, timings)
  → VALIDATION                      (media_validator + ledger_validator + cross-check)
  → AUDIT ARTIFACTS                 (media.json, frames.csv/jsonl, qc.json, run.log)
```

## Full future pipeline

```
INGEST
→ MEDIA VERIFY
→ FRAME LEDGER
→ ADJACENT-FRAME PASS          (deterministic frame-difference metrics per adjacent pair)
→ CUT CANDIDATES               (thresholded candidates with exact frame anchors)
→ ADVERSARIAL CUT VERIFICATION (each candidate challenged against actual adjacent frames)
→ TRUE SHOTS                   (Shot records with PTS-exact boundaries + transition types)
→ AUDIO EXTRACTION             (waveform/energy artifacts, exact sample anchoring)
→ SPEECH / ASR                 (adapter: faster-whisper / WhisperX / cloud; opt-in only)
→ CAMERA ANALYSIS              (movement phases from optical flow, timestamped)
→ CHARACTER ANALYSIS           (C-ID candidates, appearance windows, continuity)
→ OBJECT CONTINUITY            (O-ID candidates, contact/release/transfer states)
→ ACTION BOUNDARIES            (event start/end pinned to frames)
→ ON-SCREEN TEXT               (OCR with frame anchors)
→ SEED COMPARISON              (SeedClaim extraction; claims vs evidence)
→ KEEP / FIX / REBUILD         (per-section ReviewDecision)
→ CAPTION GENERATION           (LLM/VLM adapters propose text; evidence supplies timing)
→ GOLDEN EXAMPLE QUALITY GATE  (behavior comparison, not word count)
→ REVIEWER CHECKLIST           (9 questions + 16 Additional Checks, automated where possible)
→ VALIDATOR                    (M2-* rule engine over the caption)
→ FINAL ADVERSARIAL QC         (independent pass attacking every claim)
→ HUMAN REVIEW                 (final authority; app never submits anywhere)
```

Each later module consumes **evidence records** (frames, PTS ranges, audio ranges),
never another model's prose. `EvidenceReference` is the interchange type;
`MODEL_OBSERVATION` evidence can propose but never verify (see `models/evidence.py`).

## Module boundaries planned, not implemented

Cut detection, adjacent-frame similarity, optical flow, OCR, ASR, diarization, audio
energy, waveform/spectrogram, character/object tracking, camera movement analysis,
playback-speed analysis, LLM/VLM reasoning, caption generation, seed comparison,
Golden Example scoring, caption validator, desktop timeline UI (Tauri 2 + React + TS).
No fake implementations exist — the data model (`models/caption.py`) defines their
output contracts.

AI providers will be **adapters** behind a narrow interface (propose-only), never
embedded throughout the code. Audio engines (faster-whisper, WhisperX, cloud) plug in
the same way. Descript is excluded. MoviePy is never a timing authority.

## Determinism, privacy, performance

- All processing is local; zero network calls, no telemetry, no accounts, no DB.
- Artifacts are deterministic: sorted-key JSON, exact rationals as `"num/den"` strings.
- Frame images are extracted lazily (`--extract-frames` opt-in in Phase 1;
  on-demand per evidence request later). The ledger represents every frame without
  materializing images.
- `manifest.json` records per-stage wall-clock timings as benchmarks; later phases add
  triage so only informative frames reach AI adapters.
