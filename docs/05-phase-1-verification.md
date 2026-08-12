# 05 — Phase 1 Verification

## What a PASS means

For `manuscript-reviewer audit VIDEO`:
- The file probed successfully and has a video stream.
- Every decoded frame appears in the ledger, indexes 0..N-1, no duplicates, strictly
  increasing PTS.
- The independently measured decode count (`ffprobe -count_frames`) equals the
  enumerated count.
- The source file's SHA-256 was identical before and after analysis.
- All artifacts wrote successfully and are hashed in `manifest.json`.

`PARTIAL`: enumeration succeeded but some frames lack a usable PTS (exact timing
incomplete). `FAILED`: any FAIL-severity issue.

## Frame-counting strategy

Primary (authoritative): `ffprobe -select_streams v:0 -show_frames` — walks the real
decoder output in presentation order and reports each frame's integer `pts` (falling
back to `best_effort_timestamp`), `duration`, `key_frame`, `pict_type`, geometry.
This is the same decode path a player uses, so it reflects what a reviewer actually
sees, and it works identically for CFR and VFR media.

Independent cross-check: a **separate** `ffprobe -count_frames` invocation fully
decodes the stream and reports `nb_read_frames`. Two separate decode passes through
two different ffprobe code paths must agree; disagreement is a FAIL (`P1-COUNT-002`).

Third signal (metadata claim): container-declared `nb_frames`. Disagreement is a WARN
(`P1-COUNT-003`) because container metadata routinely lies; the enumerated count is
authoritative. All three signals are recorded in `qc.json → frame_count_signals` —
the engine never silently selects a convenient number.

PyAV traversal was considered as the second signal and rejected for Phase 1: it adds
a heavy binary dependency while exercising the same libavcodec decode path, so it is
not meaningfully more independent than `-count_frames`. It remains a candidate third
measurement if a container class ever defeats ffprobe.

## Timestamp strategy

- Integer PTS in stream `time_base` units is the source of truth.
- Exact seconds = `pts × time_base` as `fractions.Fraction`; no float ever
  accumulates. (`media/timestamps.py` is the only conversion site.)
- `frames.jsonl` stores exact rationals (`"1/24"`); `frames.csv` renders microsecond
  fixed-point for humans (matching ffprobe's own rendering).
- Manuscript 0.1 s display values are produced by `to_manuscript_display`
  (Decimal quantize, ROUND_HALF_UP) and never flow back into internal state.
- `frame_index / fps` is used only as a CFR cross-check (`cfr_expected_time`); VFR
  media is detected via nominal≠average rate (WARN `P1-MEDIA-004`) and handled purely
  by PTS.

## Validators and rule IDs

| Rule | Severity | Meaning |
|---|---|---|
| P1-MEDIA-000 | FAIL | file cannot be probed |
| P1-MEDIA-001 | FAIL | no video stream |
| P1-MEDIA-002 | WARN | multiple video streams (only v:0 audited) |
| P1-MEDIA-003 | WARN | container vs stream duration differ beyond threshold |
| P1-MEDIA-004 | WARN | nominal ≠ average frame rate (VFR signal) |
| P1-MEDIA-005 | WARN | audio vs video duration differ > 0.5 s |
| P1-LEDGER-000 | FAIL | frame enumeration failed |
| P1-LEDGER-001 | FAIL | ledger has zero frames |
| P1-LEDGER-002 | FAIL | frame indexes not sequential from 0 |
| P1-LEDGER-003 | FAIL | duplicate frame indexes |
| P1-LEDGER-004 | WARN | frames missing PTS (run at best PARTIAL) |
| P1-LEDGER-005 | FAIL | PTS not strictly increasing |
| P1-COUNT-001 | WARN | independent decode count unavailable |
| P1-COUNT-002 | FAIL | enumerated ≠ independent decode count |
| P1-COUNT-003 | WARN | container-declared nb_frames ≠ enumerated |
| P1-SOURCE-001 | FAIL | source SHA-256 changed during processing |

Artifact write failures raise `ArtifactWriteError` → fatal (exit code 2).

## Warning thresholds (documented per brief)

- Container vs video-stream duration: one nominal frame duration (fallback 0.1 s).
  Rationale: container start offsets/headers legitimately shift duration by ≲1 frame.
- Nominal vs average FPS: any inequality warns — it is the primary VFR indicator.
- Audio vs video duration: 0.5 s. AAC priming/padding commonly adds tens of ms;
  0.5 s separates codec padding from genuinely mismatched streams.

## Verification checklist for this phase

Run from the repo root:

```
uv run pytest        # 55 tests: timing math, probe, ledger, validators, integration
uv run ruff check .  # lint
uv run mypy          # strict typing over the engine package
uv run manuscript-reviewer audit tests/fixtures/clip_60fps_audio.mp4
```

The audit prints the media summary, three PASS/FAIL check lines, the frame-count
signal table, warnings/failures with rule IDs, the artifact path, and the overall
status; exit code 0 = PASS/PARTIAL, 1 = FAILED, 2 = fatal environment error.
