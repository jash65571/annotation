# Manuscript Reviewer

A frame-accurate **Manuscript II review and evidence engine**. Give it a video and it
produces a provably correct frame ledger with exact media metadata, SHA-256
provenance, independent frame-count cross-checks, and audit artifacts — the
deterministic foundation for evidence-backed Manuscript II reviewer captions.

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
uv run manuscript-reviewer version
```

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

Overall Phase 1 status: PASS
```

Exit codes: `0` PASS/PARTIAL, `1` FAILED (details in qc.json), `2` fatal environment
error (missing ffmpeg, unwritable artifacts).

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

## Development

```powershell
uv sync                  # install everything (dev group included)
uv run pytest            # tests (generates tiny synthetic clips with ffmpeg)
uv run ruff check .      # lint
uv run mypy              # strict type checking
```

Tests that need media are skipped with a clear message if ffmpeg is missing.

## Documentation

- `docs/00-product-source-of-truth.md` — Manuscript II requirements, from `references/`
- `docs/01-rule-hierarchy.md` — source priority and conflict handling
- `docs/02-system-architecture.md` — current slice + full future pipeline
- `docs/03-data-model.md` — the typed schema (Phase 1 and future)
- `docs/04-development-roadmap.md` — Phases 2–5
- `docs/05-phase-1-verification.md` — validators, thresholds, cross-check rationale
