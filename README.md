# Manuscript Reviewer

A frame-accurate **Manuscript II review and evidence engine**. Give it a video and it
produces a provably correct frame ledger with exact media metadata, SHA-256
provenance, independent frame-count cross-checks, and audit artifacts — plus the
**Shot Truth Engine** (deterministic adjacent-frame metrics, adversarially verified
cut candidates, gapless shot proposals — `docs/06`) and the **Audio Truth Engine**
(exact sample-anchored PCM evidence, waveform/spectrogram/energy, local
faster-whisper + WhisperX alignment in isolated uv environments, boundary audio
continuity, manual review queue — `docs/07`), **Visual Review Intelligence**
(seed claims, structural comparison, KEEP/FIX/REDO proposals — `docs/08`), and
the **Caption Brain** (eligibility-gated caption facts, deterministic
Manuscript II rendering, M2/platform/Golden gates, honest ready states —
`docs/09`).

Not a generic captioning app. Not a task-submission bot. All processing is local:
no network calls, no telemetry, no accounts, no database.

## Requirements

- **OS**: developed on Windows 11; the engine is pathlib/subprocess-portable and
  should run on macOS/Linux (untested there).
- **Python**: 3.12+ (managed automatically if you use [uv](https://docs.astral.sh/uv/)).
- **FFmpeg**: `ffmpeg` and `ffprobe` on `PATH`, or set `MANUSCRIPT_FFMPEG_DIR` to
  their bin directory. Any recent build works (developed against FFmpeg 9.0).

## Installation

```powershell
git clone <repo> manuscript-reviewer
cd manuscript-reviewer
uv sync          # creates .venv with Python 3.12 and all dependencies
```

Without uv: create a Python 3.12 venv and `pip install -e . --group dev`.

## Usage

```powershell
uv run manuscript-reviewer audit input.mp4
uv run manuscript-reviewer audit input.mp4 --seed seed.json      # hash+copy seed into the run
uv run manuscript-reviewer audit input.mp4 --extract-frames      # every frame as PNG evidence
uv run manuscript-reviewer audit input.mp4 --no-shot-analysis    # Phase 1 ledger only
uv run manuscript-reviewer audit input.mp4 --candidate-sensitivity high   # max recall
uv run manuscript-reviewer audit input.mp4 --no-asr              # audio evidence without ASR
uv run manuscript-reviewer audit input.mp4 --asr-model tiny --asr-device cpu
uv run manuscript-reviewer audit input.mp4 --seed seed.md --feedback feedback.txt `
    --review-decisions decisions.json --human-facts human_facts.json   # Phase 1→5
uv run manuscript-reviewer finalize artifacts\input\<run_id> `
    --review-decisions decisions.json --human-facts human_facts.json `
    --final-review final_review.json    # fast re-finalization, no media re-analysis
uv run manuscript-reviewer version
uv run python scripts/benchmark.py                                # Phase 2 runtime benchmark
uv run pytest -m asr_integration                                  # real local ASR tests (needs models)
```

Audio analysis runs by default. ASR runs locally in isolated uv worker
environments (faster-whisper 1.2.1, default model `large-v3-turbo`; WhisperX
3.4.3 alignment). First use downloads packages/models (`--no-asr-bootstrap`
to forbid); task media is never uploaded anywhere and there is no cloud or
Descript fallback — an ASR failure degrades to waveform/spectrogram/energy
evidence and manual review.

Shot analysis runs by default and reports supported boundaries, rejected false
positives, and review-required candidates with structured reason codes. Expert
thresholds are intentionally not CLI flags; only `--candidate-sensitivity
high|normal|low` is exposed.

Example output:

```
MANUSCRIPT REVIEWER
-------------------

Media verified
Resolution: 320x240
Codec: h264 (High)
FPS: 60
Duration: 2.000000s
Frames declared: 120
Frames enumerated: 120
Audio: AAC 44100Hz stereo

Frame ledger: PASS
Timestamp monotonicity: PASS
Frame accounting: PASS

Artifacts:
artifacts\input\20260812T005001Z-6c015bcd

Overall audit status: PASS
```

Statuses: `PASS`, `REVIEW_REQUIRED` (unresolved possible cuts or unverified
endpoint — deterministic checks passed but a human must review),
`PARTIAL` (exact timing incomplete), `FAILED`. Exit codes: `0`
PASS/REVIEW_REQUIRED/PARTIAL, `1` FAILED (details in qc.json), `2` fatal
environment error (missing ffmpeg, unwritable artifacts).

## Artifacts

Each run writes `artifacts/<video_stem>/<run_id>/`:

| File | Contents |
|---|---|
| `media.json` | typed media facts + the raw ffprobe JSON verbatim |
| `frames.csv` | one row per enumerated frame: index, pts, pts_time, dts, duration, key_frame, pict_type, geometry |
| `frames.jsonl` | lossless ledger — exact rational times (`"1/24"`), time_base header |
| `qc.json` | validation status, every issue with rule ID, all frame-count signals |
| `manifest.json` | run id, source/seed SHA-256, app+rules+ffmpeg versions, timings, artifact hashes |
| `run.log` | full debug log of the run |
| `frames/` | (only with `--extract-frames`) `F000042_1.750000.png` — index + exact time |
| `adjacent_metrics.csv/.jsonl` | one record per adjacent frame pair (N−1 rows): diff, histogram, phash, edges, optical flow, scdet |
| `cut_candidates.json` | merged pre-verification candidates with sources |
| `boundary_evidence.json` | post-verification candidates with status + reason codes |
| `shots_proposed.json` | gapless provisional shots with exact frame ownership |
| `transition_evidence.json` | per-boundary transition proposals, fades, blends |
| `shot_qc.json` | full Shot Truth result: counts, statuses, every candidate |
| `shot_evidence/candidate_NNNN/` | labeled `pair.png`, `strip_short.png`, `strip_context.png`, `evidence.json` |
| `audio/` | `source.wav`, `asr.wav`, audio frame ledger, `audio_timeline.json`, 10 ms energy CSV, waveform/spectrogram/energy PNGs, regions, transients, `speech_regions.csv`, `boundary_audio_evidence.json`, `audio_review_queue.json`, `review_clips/`, `audio_qc.json` |
| `audio/asr/` | status + runtime metadata, faster-whisper transcript/segments/words, WhisperX-aligned versions, `transcript_best.*` (ASR_EVIDENCE_ONLY) |
| `caption/` | Phase 5: caption facts + eligibility, plan, `draft_review_only.md` **or** `ready_to_enter.md` (only when truly ready), assertion map, coverage, seed change log, M2/platform/Golden reports, adversarial QC, `final_status.json`, `caption_manifest.json` |
| `review_report.md` | reviewer packet: blockers, KEEP/FIX/REDO, validator + gate findings (rationale never enters caption fields) |

## Development

```powershell
uv sync                  # install everything (dev group included)
uv run pytest            # tests (generates tiny synthetic clips with ffmpeg)
uv run ruff check .      # lint
uv run mypy              # strict type checking
```

Tests that need media are skipped with a clear message if ffmpeg is missing.

GitHub Actions CI (`.github/workflows/ci.yml`) runs the same three gates on
Ubuntu / Python 3.12 / uv with FFmpeg (libx264) on every push and pull
request. Tested dependency versions: opencv-python-headless 5.0.0.93, numpy
2.5.2 (bounds in pyproject reflect these).

Note: the raw Manuscript II reference documents are local-only and gitignored
(`references/README.md`); the repo ships only derived rules.

## Documentation

- `docs/00-product-source-of-truth.md` — Manuscript II requirements, from `references/`
- `docs/01-rule-hierarchy.md` — source priority and conflict handling
- `docs/02-system-architecture.md` — current slice + full future pipeline
- `docs/03-data-model.md` — the typed schema (Phase 1 and future)
- `docs/04-development-roadmap.md` — Phases 2–6
- `docs/05-phase-1-verification.md` — validators, thresholds, cross-check rationale
- `docs/06-shot-truth-engine.md` — Phase 2: metrics, verifier, transition policy, failure modes
- `docs/07-audio-truth-engine.md` — Phase 3: audio timeline, ASR workers, language safety, review queue
- `docs/08-visual-review-intelligence.md` — Phase 4: seed parsing/claims, structural comparison, KEEP/FIX/REDO proposals, review queue, frame observations
- `docs/09-caption-brain.md` — Phase 5: caption eligibility, fact graph, deterministic renderer, M2/platform/Golden gates, signoff and ready states
