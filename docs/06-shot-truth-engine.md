# 06 — Shot Truth Engine (Phase 2)

## Candidate vs truth

A **candidate** is a hypothesis produced by a detector. A detector — including
PySceneDetect-style detectors, ffmpeg `scdet`, and our own metrics — is never
factual truth; a threshold crossing is never factual truth. Every candidate
passes through a deterministic **adversarial verifier** that actively tries to
explain the discontinuity without an edit. The automated output distinguishes:

- `CANDIDATE` — generated, not yet verified
- `SUPPORTED` — survived every adversarial challenge
- `REJECTED` — a benign deterministic explanation fits (with reason codes)
- `REVIEW_REQUIRED` — evidence is ambiguous; a human must decide

`REVIEW_REQUIRED` is never resolved just to make output neat, and overall shot
status can never be `PASS` while unresolved potential cuts remain (P2-QC-001).
**False-negative cuts are high risk**: candidate generation favors recall, and
rejection requires a positive benign explanation.

## Boundary-between-frames model

A boundary is evidence BETWEEN two exact ledger frames, never a lone timestamp:

- `left_frame_index` — final frame before the visual boundary
- `right_frame_index` — first frame after it (always left+1)
- `boundary_time_exact` — the incoming/right frame's PTS time (exact `Fraction`)
- `boundary_time_manuscript` — 0.1 s display projection (presentation only)

Phase 1's rational timestamp infrastructure is the only clock. Floats never
carry authoritative time.

## Shot timeline semantics (Phase 2.1)

Two concepts are kept strictly separate on every `ShotProposal`:

- **Inclusive frame ownership** — `start_frame_index`/`end_frame_index`; the
  boundary's left frame belongs to the outgoing shot, its right frame opens the
  incoming shot.
- **Continuous annotation interval** — `start_exact`/`end_exact`. The SAME
  exact boundary time (the incoming/right frame's PTS) is both the outgoing
  shot's interval end and the incoming shot's interval start, so adjacent
  shots always satisfy `prev.end_exact == next.start_exact` — including after
  0.1 s display rounding (regression-tested at a half-tenth boundary).
  The final owned frame's own start time is preserved separately as
  `last_owned_frame_start_exact` and is never used as an interval end.

The **final shot's** `end_exact` is the canonical annotation endpoint from
`media/endpoint.py` (final-frame presentation end preferred, then stream
duration, container duration, filename segment length), never the final
frame's start PTS. Material endpoint conflicts surface as WARN `P2-END-001`
and force `REVIEW_REQUIRED` — an incorrect annotation endpoint is a permanent
failure class (rules v1.1.0). Timeline validators: `P2-TIME-001…005`.

## Frame terminology (engineering clarification)

The Manuscript sources say "every encoded frame". Technically our pipeline
distinguishes: container **packets** (not enumerated), **encoded frames**
(compressed pictures, incl. B/P reordering), and **decoded/enumerated video
frames** in presentation order — which is what `ffprobe -show_frames` reports,
what a reviewer sees, and what the ledger and this engine operate on. The
Manuscript rule quote is unchanged; our implementation term is
"decoded/enumerated frames".

## Metric decode

One extra full decode scaled to a deterministic metric grid: **160×90
grayscale** (`scale=160:90:flags=area`, `-fps_mode passthrough`). Decoded image
N is ledger frame N; a count mismatch aborts the stage (no silent
misalignment). Optical flow runs on a further-halved 80×45 grid.

## Adjacent-pair metrics (N frames → exactly N−1 records)

| Metric | Detects well | Fooled by | Cost | Normalization / range |
|---|---|---|---|---|
| `mean_abs_diff` (luma MAD) | any abrupt visual change | fast motion, flashes, noise | trivial | 0–255, luma units on metric grid |
| `hist_distance` (Bhattacharyya, 64-bin gray) | content/palette change; robust to motion | luminance-only changes, similar-palette cuts | cheap | 0 (identical) – 1 (disjoint) |
| `phash_hamming` (64-bit DCT hash) | structural/layout change | flat frames, heavy motion blur; flips bits on ordinary motion (measured: 20 bits for same-scene frames 6 apart) | cheap | 0–64 bits |
| `edge_change` (Sobel edge-density delta) | structure appearing/vanishing (fades, cuts to flat) | texture-dense motion | cheap | 0–~1 |
| `luma_delta` / per-frame `luma_mean`, `luma_std` | flashes, fades, exposure spikes | slow lighting drift | trivial | 0–255 |
| `sharpness` (Laplacian variance, per frame) | blur events (whip pans, focus loss) | noise | cheap | unbounded, relative |
| `flow_mean_mag` + `flow_coherence` (Farneback, 80×45) | distinguishes coherent camera motion from incoherent change | very large displacement (> grid), pure zoom (coherence ≈ 0 by design → radial check) | ~1–2 ms/pair | mag: px on 80×45 grid; coherence: 0–1 |
| `scdet_score` / `scdet_mafd` (ffmpeg scdet) | independent implementation cross-check | same things that fool MAFD | 1 extra decode (~0.1–0.2 s/15 s clip) | scdet's 0–100-ish score |

**Considered and rejected:** MSE (redundant with MAD), full HSV per-channel
histograms (gray histogram + luma cover the need at lower cost), SSIM
(needs scikit-image or a hand-rolled window implementation; phash+edge+hist
already cover structure — revisit if real-world regressions show a gap), dense
full-resolution optical flow (cost without added decision power).

## Local baselines

No single global threshold: each pair is judged against the **median/MAD of
pair metrics in a ±0.5 s time window** around it (excluding itself), using real
PTS times (VFR-safe; falls back to ±12 pairs if PTS is missing). Robust
z-scores `(x − median)/(MAD·1.4826)` with MAD floors (0.35 luma / 0.004 hist)
so a perfectly static neighborhood cannot produce infinite z. `neighbor_motion`
(median flow magnitude in the window) feeds the motion defense.

## Candidate generation (recall-first)

Signal families (sources are preserved through merging):

- `internal_difference` — diff z ≥ 5 (normal) + absolute floor 4.0
- `internal_histogram` — hist z ≥ 5 + hist > 0.05
- `internal_phash` — hamming ≥ 14
- `internal_flash` — entry/exit pairs of detected flash regions (always)
- `internal_fade` / `internal_blend` — synthesized from region evidence
- `ffmpeg_scdet` — scdet score ≥ 8

`--candidate-sensitivity high|normal|low` scales these (high = max recall).
Candidates on the same adjacent pair merge into one, preserving all sources.
There is **no minimum shot length anywhere** — adjacent boundaries (1-frame
shots) are representable and tested.

## Adversarial verifier (deterministic, ordered challenges)

1. **Fade pattern** — pair inside a multi-frame fade region → not a standalone
   hard cut (fade boundaries are composed separately).
2. **Blend pattern** — pair inside a sustained multi-frame blend → folded to
   the blend's peak pair as one REVIEW_REQUIRED possible-dissolve.
3. **Flash frame** — pair borders a 1+-frame flat near-black/near-white run →
   REVIEW_REQUIRED; pre/post continuity is measured and reported, but a
   deliberate flash transition is never ruled out deterministically.
4. **Coherent camera motion** — flow coherence ≥ 0.70 and magnitude ≥ 1.2 with
   sustained neighborhood motion → REJECTED (pan/whip/shake), unless the
   discontinuity is extreme (z ≥ 14 and phash ≥ 28) → REVIEW_REQUIRED.
5. **Zoom/scope** — radial-flow signature with sustained motion → REJECTED.
6. **Return-to-previous-state** — content shortly after the boundary matches
   content shortly before it. Measured fact: pixel similarity CANNOT
   distinguish "same shot moments later" from "jump cut to similar
   composition" (fixture measurements: jump-cut cross-boundary phash 14 /
   hist 0.034 vs same-shot 6-frames-apart phash 20 / hist 0.069). Therefore a
   match is NEVER auto-rejected → REVIEW_REQUIRED with
   `RETURN_TO_PREVIOUS_STATE` (+ `POSSIBLE_JUMP_CUT` for ≤2-frame anomalies,
   `LARGE_OCCLUSION`/`EFFECT_CONTINUITY` for longer ones). This also keeps
   1-frame editorial inserts visible.
7. **Support** — strong local outlier (z ≥ 7) + single-pair spike with locally
   stable sides + (structural change ≥ 18 phash bits OR ≥ 2 independent
   sources) → SUPPORTED. Similar-composition supported boundaries (phash ≤ 14)
   get transition status REVIEW_REQUIRED (`POSSIBLE_JUMP_CUT`) instead of a
   Hard cut label.
8. Anything left: rejected only if structure barely changed (phash < 10) with a
   single weak source; otherwise REVIEW_REQUIRED.

## Fade detection

Multi-frame monotonic luma trends (≥3 frames, ≥4 luma/frame) ending/starting in
a flat near-black/near-white frame → `FadeEvidence` (direction, target color,
frame range, PTS range). A mid-clip fade-out→fade-in pair composes a SUPPORTED
boundary whose right frame is the first frame of the incoming fade-in;
black/white hold frames attach to the outgoing shot per the Manuscript
outgoing-ownership rule. Edge-of-media fades produce evidence but no boundary.
Proposed transitions: `Fade in` / `Fade out` only when the pattern is proven.

## Cross-dissolve detection (conservative)

Sustained elevated-change runs (≥3 pairs above 2.5× the CLIP-GLOBAL median
pair difference — global, because a sustained dissolve raises its own local
baseline and would hide from a local reference) without a dominant single
spike (peak < 2.5× run mean) and without coherent motion →
`BlendEvidence`, always REVIEW_REQUIRED. If no generator flagged the region, a
review candidate is synthesized at the blend's peak pair so a missed dissolve
can never produce a false PASS. Dissolves are never auto-classified.

## Wipe / iris

Not detected in Phase 2 (rare; low value now). The evidence model supports
spatially progressive transitions later via difference masks; unexplained
sustained changes surface as blend/REVIEW_REQUIRED rather than being hidden.

## Transition classification policy

- Automated proposals: **Hard cut, Fade in, Fade out** (evidence-backed only).
- Conservative/review: **Cross dissolve** (blend evidence, REVIEW_REQUIRED).
- Deferred to semantic/audio phases: **Jump cut, Match cut, Smash cut, Whip/
  Swish pan as transition, Wipe, Iris** — flagged (e.g. `POSSIBLE_JUMP_CUT`)
  but never finalized from pixels.
- **L-cut / J-cut**: require audio evidence; Phase 2 sets
  `audio_verification_required=true` on every supported boundary of a clip
  with an audio stream and never emits these labels (P2-TRANS-002 enforces).
- Internal states `UNRESOLVED` / `REVIEW_REQUIRED` never silently become a
  Manuscript transition type; the final menu is validated against the rule
  file (P2-TRANS-001).

## External detectors

**PySceneDetect: evaluated, not integrated.** Its detectors (Content/Adaptive/
Histogram/Hash) recompute the same signal families we already produce with
exact ledger identity; integrating it would add a large dependency, its own
decode path, its own timestamp model to reconcile, and a min-scene-len default
that historically hides short shots. Our design goal ("use detector metrics as
evidence rather than trusting its scene list") is already satisfied
internally. Revisit if real-world regressions show a recall gap.

**FFmpeg `scdet`: integrated as evidence only.** One extra decode with
`scdet=s=0,metadata=mode=print` parses per-frame `lavfi.scd.score`/`mafd`;
frame identity maps 1:1 to the ledger via decode order under
`-fps_mode passthrough`. Its threshold/decision output is ignored; scores
attach to pair records and act as one more candidate source
(`MULTI_DETECTOR_SUPPORT`). Benchmarked at ~0.08–0.18 s per 15 s clip.
Disable with `--no-scdet`.

## VFR behavior

All windows (baselines, evidence context) are time-based over real PTS, so VFR
media gets correct context. Boundary identity is frame-pair + exact PTS —
tested with a synthetic VFR clip (dropped frames): enumeration indexes shift,
exact boundary PTS does not.

## Short-shot policy

No minimum scene length exists anywhere in the pipeline. Tested: a 3-frame
inserted shot yields both boundaries; a 1-frame insert stays visible as
REVIEW_REQUIRED (deterministically indistinguishable from an effect frame).

## Evidence artifacts

Per supported/review candidate: `shot_evidence/candidate_NNNN/` containing
`pair.png` (labeled LEFT/RIGHT frames), `strip_short.png` (±3 frames),
`strip_context.png` (time-based ±0.5 s window resolved to actual ledger
frames), `evidence.json` (candidate record + frame identities). All images are
decoded by frame identity (`select='eq(n,…)'` + `-fps_mode passthrough`),
never by approximate `-ss` seeks, and carry burned-in labels: frame index,
exact PTS time, Manuscript display time. Strips use thumbnails (160 px) to
avoid redundant full-resolution dumps.

## Validator rules (P2-*)

| Rule | Severity | Meaning |
|---|---|---|
| P2-DECODE-001 | FAIL | metric decode failed / misaligned |
| P2-PAIR-001 | FAIL | pair record count ≠ frame_count − 1 |
| P2-PAIR-002 | FAIL | pair not exactly consecutive ledger frames |
| P2-PAIR-003 | FAIL | pair timestamps go backward |
| P2-CAND-001 | FAIL | candidate references invalid adjacent frames |
| P2-CAND-002 | FAIL | boundary exact time ≠ right-frame PTS |
| P2-SHOT-001/002 | FAIL | shot 1 not at media start / final shot not at media end |
| P2-SHOT-003/004 | FAIL | gap / overlap between shot proposals |
| P2-SHOT-005 | FAIL | supported boundaries ≠ shot boundaries (1:1) |
| P2-SHOT-006/007 | FAIL | Shot 1 not Opening shot / later shot Opening shot |
| P2-TRANS-001 | FAIL | transition outside the Manuscript rule menu |
| P2-TRANS-002 | FAIL | L-cut/J-cut auto-finalized without audio |
| P2-EVID-001/002 | FAIL | supported/review candidate without evidence artifacts |
| P2-QC-001 | — | encoded in `compute_shot_status`: PASS forbidden with unresolved candidates |

## Known failure modes

1. **Jump cuts with near-identical composition** may surface only as
   REVIEW_REQUIRED (`RETURN_TO_PREVIOUS_STATE`), or — if the discontinuity is
   weak at every metric — not at all. Semantic analysis (Phase 4) is the real
   fix; sensitivity `high` narrows the gap.
2. **Cuts between visually similar scenes** (same palette, same layout) can
   fall below generation thresholds. Multiple signal families + scdet reduce
   but don't eliminate this.
3. **A cut hidden inside fast coherent motion** in the same direction may be
   rejected as camera motion unless extreme (guarded by the extreme-z
   REVIEW_REQUIRED escape hatch).
4. **Dissolves under heavy motion** can be masked by the coherence gate in
   blend detection.
5. **Long flash/effect sequences** (> ~8 frames) exceed the return-probe range
   and surface as multiple review candidates rather than one tidy region.
6. **Verifier thresholds** were set against synthetic fixtures and one real
   clip; the regression-fixture architecture (tests/regression/) exists to
   tune them against completed Manuscript tasks without overfitting to one
   clip.
