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

## Phase 2 Shot Truth Engine (implemented — see docs/06)

```
FRAME LEDGER
  → METRIC DECODE                   (shots/decode.py: 160x90 gray, ledger-aligned)
  → ADJACENT-PAIR METRICS           (shots/metrics.py: MAD, histogram, phash, edges, flow)
  → ffmpeg scdet EVIDENCE           (shots/scdet.py: independent signal, evidence only)
  → LOCAL BASELINES                 (shots/baseline.py: ±0.5 s robust median/MAD windows)
  → FLASH / FADE / BLEND REGIONS    (shots/regions.py)
  → CANDIDATE GENERATION + MERGE    (shots/candidates.py: recall-first, multi-signal)
  → ADVERSARIAL VERIFICATION        (shots/verifier.py: SUPPORTED/REJECTED/REVIEW_REQUIRED)
  → SHOT PROPOSALS                  (shots/builder.py: gapless, exact frame ownership)
  → EVIDENCE BUNDLES                (shots/evidence.py: labeled pairs/strips by frame identity)
  → P2 VALIDATION + shot_qc.json    (validation/shot_validator.py)
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

## Phase 3 Audio Truth Engine (implemented — see docs/07)

```
AUDIO STREAM VERIFICATION
  → AUDIO FRAME LEDGER              (audio/probe.py: every decoded audio frame, exact PTS)
  → LOSSLESS PCM                    (audio/decode.py: source.wav evidence + asr.wav)
  → EXACT SAMPLE ANCHOR             (audio/timeline.py: sample ↔ PTS ↔ annotation time)
  → 10ms ENERGY / WAVEFORM / SPECTROGRAM  (metrics.py, render.py — annotation clock)
  → SILENCE / TRANSIENT / SIGNAL REGIONS  (regions.py — signal classes, never semantics)
  → faster-whisper WORKER           (isolated uv env, pinned 1.2.1, transcribe-only)
  → WhisperX ALIGNMENT WORKER       (isolated uv env, pinned 3.4.3, wording preserved)
  → SPEECH REGIONS + VAD RECALL DEFENSE   (speech.py)
  → SHOT-BOUNDARY AUDIO CONTINUITY  (boundary_audio.py — resolves Phase 2 audio flags)
  → REVIEW QUEUE + CLIPS            (review_queue.py)
  → P3 VALIDATION + audio_qc.json   (validation/audio_validator.py)
```

The annotation clock (`media/clock.py`) is the single source↔annotation time
mapping used by shots, endpoint, and audio alike.

## Phase 5 Caption Brain (implemented — see docs/09)

```
PHASE 1-4 EVIDENCE ARTIFACTS (+ human decisions / human facts / signoff)
  → CAPTION ELIGIBILITY            (caption/eligibility.py: ONE central policy)
  → CAPTION FACT GRAPH             (caption/facts.py: typed, eligibility-gated)
  → CAPTION PLAN                   (caption/planning.py: overview + shot plans)
  → DETERMINISTIC RENDER           (caption/renderer.py: Manuscript II structure)
  → COVERAGE + ASSERTION MAP       (caption/coverage.py: omission/hallucination)
  → M2-* / PLATFORM / GOLDEN GATES (validation/caption_validator.py + friends)
  → ADVERSARIAL QC + SIGNOFF       (final_caption_validator.py, finalizer.py)
  → BLOCKED | REVIEW_REQUIRED | READY_FOR_FINAL_REVIEW | READY_TO_ENTER
```

Orchestrated by `caption_brain.py::finalize_run`, which reloads run artifacts
(no media re-decoding) — the same path serves both `audit --caption-brain` and
the fast `finalize RUN_DIR` command. `ready_to_enter.md` exists only when the
status is truly READY_TO_ENTER.

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
