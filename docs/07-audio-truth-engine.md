# 07 — Audio Truth Engine (Phase 3)

**Actual source audio is factual truth. ASR is evidence only.** Nothing in this
stage produces final caption prose; it produces defensible audio evidence for
the later caption layer.

## Annotation timeline origin (Pre-flight A)

`media/clock.py::AnnotationClock` is the ONE mapping between clocks:
`annotation_time = source_media_time - origin`, where origin = the first
presented video frame's source PTS time. Both clocks are preserved everywhere
(`source_*` fields keep raw PTS; `start_exact`/`end_exact`/Manuscript displays
use annotation time). Endpoint duration signals (stream/container/filename)
are converted as `origin + duration` before comparison (Pre-flight A2) — an
absolute PTS is never compared against an unshifted duration. Tested with
`-output_ts_offset 5` fixtures: origin 5 s, boundary raw PTS 6 s → annotation
1.0 s, endpoint annotation 2 s (never 7, never −3).

## Audio frame ledger and sample anchor

`ffprobe -show_frames` over `a:0` enumerates every decoded audio frame (pts,
duration, nb_samples) → `audio_frames.csv/.jsonl`. `annotation time of sample N
= annotation_audio_offset + N/sample_rate` (exact Fraction math,
`audio/timeline.py`, no float accumulation).

**AAC priming / skip-sample evidence (`audio/probe.py::probe_audio_priming`).**
Sample 0 of the decoded WAV is NOT assumed to equal the first encoded packet's
PTS without evidence. We collect the raw priming metadata separately:

- stream `initial_padding` (`AudioTimeline.initial_padding_samples`);
- packet side-data `skip_samples` / `discard_padding` — the AAC encoder delay
  the decoder trims — from the first packet's `side_data_list`
  (`codec_skip_samples` / `codec_discard_padding`);
- stream `start_pts` (`codec_delay`).

Three distinct concepts stay distinct: the **encoded packet timeline** (first
packet PTS, often negative for AAC, e.g. −1024), the **decoder-applied
skip/priming** (e.g. `skip_samples = 1024`), and the **decoded PCM sample-0
anchor**. When ANY priming/skip is declared, `sample_anchor_status` becomes
`AUDIO_SAMPLE_ANCHOR_REVIEW_REQUIRED`: a `P3-AUDIO-009` warning is emitted and an
`AUDIO_SAMPLE_ANCHOR_REVIEW_REQUIRED` concern is recorded — sample-perfect source
anchoring is never silently claimed (only `ANCHORED` when no priming exists).

**PCM cross-check**: decoded WAV sample count vs (a) sum of enumerated frame
nb_samples and (b) declared duration × rate, with a documented 50 ms priming
tolerance (AAC priming is typically 1024–2112 samples). Disagreements surface
as `P3-AUDIO-007/008` warnings — never silently resolved.

## Source WAV vs ASR WAV

- `audio/source.wav` — PCM s16le at SOURCE rate/channels. The only precision
  change is 16-bit quantization of the decoder output. Evidence for listening,
  waveform, spectrogram, review clips. Never overwritten.
- `audio/asr.wav` — PCM s16le mono 16 kHz (Whisper input contract). Conversion:
  channel downmix + resample. ASR input only.

## Energy metrics (10 ms)

Bins are exactly `rate/100` samples (480 @ 48 kHz, 441 @ 44.1 kHz — both
exact). Per bin: RMS, peak, dBFS, zero-crossing rate, spectral centroid,
spectral flatness — purposes and blind spots documented in
`audio/metrics.py`. Energy ≠ speech, ever.

## Waveform / spectrogram / energy plot

cv2-rendered (no new dependencies), X axis ALWAYS the annotation clock with
second ticks. Spectrogram: 1024-pt Hann STFT, hop 256, dB floor −90, linear
frequency, MAGMA colormap. The spectrogram is signal evidence — no sound is
ever semantically labeled (music/gunshot/applause) from a spectral pattern.

## Deterministic regions

- SILENCE_CANDIDATE: ≤ −55 dBFS sustained ≥ 300 ms. Internal evidence only —
  NEVER "No speech"/"No music" lines (rules: empty_audio_fillers_forbidden).
- ACTIVE_AUDIO ≥ −40 dBFS sustained; SUSTAINED_TONAL_AUDIO (flatness ≤ 0.02)
  and BROADBAND_NOISE (≥ 0.3) are signal classes, not semantics.
- TRANSIENT_CANDIDATE: ≥ 14 dB rise over the trailing 200 ms median; start/
  peak/end samples + annotation times; semantics stay
  TRANSIENT_SEMANTICS_UNKNOWN for review.

## faster-whisper (primary transcription lead)

Isolated uv worker (`audio/asr/workers/fw_env`, pinned
**faster-whisper==1.2.1**), JSON file protocol, structured argv through the
single safe subprocess wrapper. `task=transcribe` ALWAYS (never translate),
word timestamps on, VAD on for the main pass. Default model
**large-v3-turbo** (AUDIT-NOTES §5), configurable via `--asr-model/`
`--asr-device/--asr-compute-type/--asr-language`. A missing local CUDA
runtime (cublas/cudnn) triggers a recorded CPU int8 fallback inside the
worker — local-only, honest metadata (`device_fallback`). Low-confidence
output is stored and flagged, never discarded.

Tested locally: faster-whisper 1.2.1 (CTranslate2), models `tiny` (plumbing
verification + integration test) and `large-v3-turbo` (real-clip run, CPU
int8, ~23 s for a 16 s clip). **Model cache state is snapshotted before AND
after the worker runs** and reconciled to `cached` / `downloaded_this_run` /
`unavailable` — a model fetched during the run is recorded as
`downloaded_this_run`, never left as `not_cached`.

## Mandatory source verification + caption-eligibility gate

**ASR speech is never human-verified by machine.** Text, word timing, language
and speaker labels are all unverified evidence (Master Frame Audit Protocol).
Two orthogonal axes are stored separately:

- `EvidenceState` (machine processing quality): `ASR_EVIDENCE` /
  `ALIGNED_EVIDENCE`. `ALIGNED_EVIDENCE` means only that WhisperX produced valid
  timing — NOT that a human confirmed the words.
- `SpeechRegion.source_verification_status` (human-listen axis): defaults to
  `UNVERIFIED` for ALL ASR speech and is moved off `UNVERIFIED` ONLY by a human
  (`HUMAN_VERIFIED` / `HUMAN_CORRECTED` / `REJECTED`). ASR probability and
  WhisperX alignment never change it.

Every source-unverified speech region gets a
`MANDATORY_SOURCE_AUDIO_VERIFICATION` review item — a clean, high-confidence,
fully-aligned result never bypasses a human listen (P3-SPEECH-004). Top-level
audio PASS is forbidden while any speech region is `UNVERIFIED` (P3-QC-001).

**Caption-eligibility safety contract (for later Phase 4, built now, not used
yet).** `SpeechRegion.caption_text_eligible` / `.caption_text`:
`HUMAN_VERIFIED` → the ASR text may be quoted; `HUMAN_CORRECTED` → only the
human `corrected_text`; `UNVERIFIED` → ASR text is evidence only and MUST NOT
become quoted caption dialogue; `REJECTED` → not usable.
`caption_language_eligible` is false until the language is human-verified — a
high Whisper language probability never creates a caption language claim. No
final captions are built in Phase 3.1; only the safety contract.

## VAD recall defense

Absence of VAD/ASR output NEVER proves silence. ACTIVE_AUDIO windows without
ASR coverage (≥ 0.4 s) become `AUDIO_WITHOUT_ASR_COVERAGE` review items.
**Decision**: no automatic second no-VAD transcription pass — uncovered
regions go to human review instead, because a human listen is the only
trustworthy resolution for VAD-missed material and double transcription of
every clip costs more than it recovers. Revisit if review queues show a
pattern of recoverable misses.

## WhisperX (alignment only) + word reconciliation

Isolated uv worker (`audio/asr/workers/wx_env`, pinned **whisperx==3.4.3**,
CPU-only torch via the pytorch-cpu index for reproducibility). It receives the
faster-whisper transcript and must PRESERVE wording; the text-preservation
check (whitespace-collapsed, case-preserving normalization) demotes any
change to `ALIGNMENT_TEXT_MISMATCH` — the changed transcript is never
silently accepted. Alignment failure records the reason, keeps faster-whisper as
`transcript_best`, and the pipeline continues.

**WhisperX supplies timing improvements only, never words.** Reconciliation
(`reconcile_best_transcript`) produces **exactly one best word per faster-whisper
source word** — no source word is ever dropped, invented, or re-worded
(P3-ASR-006). Faster-whisper words carry stable identity (`segment_id`,
`source_word_index`); WhisperX timing is mapped back by conservative normalized-
text matching within a segment, not by output-list position. Per word:

- match found with valid timing → `WHISPERX_ALIGNED` (WhisperX timing);
- no match, or WhisperX emitted no timing for that word → `FASTER_WHISPER_FALLBACK`
  (faster-whisper timing kept);
- ambiguous mapping → conservative faster-whisper fallback, review required.

**Missing timing is missing evidence, never 0 s.** A WhisperX word with no
start/end is kept (never dropped) with `None` timing; the source word falls back
to faster-whisper timing. A missing WhisperX *segment* timestamp falls back to
the preserved faster-whisper segment timing. **Ordering-safety policy:** if
per-word WhisperX timings would make a segment non-monotonic, the whole segment
falls back to faster-whisper timing — accuracy over aligned-word count.

**Coverage & status.** `ALIGNED` requires text preserved AND every source word
WhisperX-aligned AND ordered AND in-bounds; otherwise `ALIGNMENT_PARTIAL` with
`alignment_coverage` recorded (e.g. 3/4 = 0.75) and a `P3-ASR-007` warning. If
3 of 4 words align, those three use WhisperX timing and the fourth keeps
faster-whisper timing; the region requires review either way.

`transcript_best` / `words_best.csv`: wording = faster-whisper; timing = best per
word with `timing_status`. `words_best.csv` preserves the exact faster-whisper
source word sequence (P3-ASR-006 fails a drop/invention/re-word). Header carries
`verification_status = ASR_EVIDENCE_ONLY` — "best" never implies verified.

## ASR float time policy

Worker timestamps are parsed via `Decimal(str(value))`, stored verbatim
(`asr_start_seconds`), and mapped onto the annotation clock as exact
Fractions with `timing_source` recorded. ASR boundaries are estimates —
never sample truth.

## Language safety

`LanguageEvidence` stores candidate + probability + source + review status
(SUPPORTED_BY_ASR ≥ 0.8, else REVIEW_REQUIRED/UNKNOWN; HUMAN_VERIFIED only by
a human). A Whisper language guess never becomes caption fact; the later
caption layer degrades specificity structurally (family/region/"foreign
language") instead of hedging.

## Singing / vocals

Sustained tonal audio without trustworthy ASR text stays visible as
`VOCAL_REVIEW_REQUIRED` — "music detected" never erases vocals, and lyrics
are never guessed.

## Overlap and diarization

`SpeechRegion.possible_overlap`/`overlap_status` support overlap even when
automation cannot resolve it (REVIEW_REQUIRED, never flattened into one
speaker as fact). Diarization is OPTIONAL and not implemented in Phase 3; the
`SpeakerEvidence` model fixes the contract now: labels are `SPEAKER_NN` only
and NEVER become C# identities without audio/visual continuity or human
verification. No Hugging Face token is required for the normal audit.

## Shot-boundary audio continuity

Every SUPPORTED visual boundary gets a ±0.5 s window. **Each fact is computed and
stored separately — never cloned from another** (`boundary_audio_evidence.json`):

- presence: `audio_present_before`, `audio_present_after`, `silence_spans_boundary`;
- crossing (three DISTINCT facts): `speech_region_spans_boundary` (weakest — two
  nearby words merge into one region), `asr_segment_spans_boundary`, and
  `asr_word_spans_boundary` (strongest — an actual best-word interval satisfies
  `word.start < boundary < word.end`). The old `asr_word_crossing =
  speech_crossing` clone is removed;
- energy: `energy_before/after_dbfs`, `energy_delta_db`;
- spectral: `spectral_centroid_before/after`, `spectral_flatness_before/after`,
  `spectral_change_score`.

**Energy on both sides proves only `AUDIO_PRESENT_BOTH_SIDES`, not that one
source crosses the cut.** A 440→880 Hz switch or a tone→broadband-noise switch at
EQUAL energy moves the spectrum, so it reads `UNCERTAIN`, never `CONTINUOUS`.
`CONTINUOUS` requires meaningful evidence — an actual word spanning the boundary
(`ASR_WORD_SPANS_BOUNDARY`) or proven spectral continuity (`SPECTRAL_CONTINUITY`:
stable centroid + flatness, small energy step, no silence gap); P3-BOUNDARY-003
fails a `CONTINUOUS` verdict lacking either code. Audio present on both sides
without proven source continuity is `UNCERTAIN`. Silence spanning the boundary is
`DISCONTINUOUS` / `CHECKED_NO_CROSSING`. Phase 2's `audio_verification_required`
resolves to CHECKED_NO_CROSSING / CROSSING_REVIEW_REQUIRED / UNAVAILABLE.
**L-cut/J-cut is never auto-finalized** — even proven continuity cannot decide
the semantic source relation (P3-BOUNDARY-002); the boundary stays
`CROSSING_REVIEW_REQUIRED`.

## Review queue and clips

`audio_review_queue.json`: every item names WHY to listen (reason enum), the
exact annotation window, a padded playback window, evidence refs, and the ASR
text candidate. **One item per speech act carries ALL its reasons** (mandatory
source verification + low confidence + partial alignment + clip-edge + language …)
— duplicate listening items for the same speech act are not created. **Language
review is decided per relevant speech region** and never globally suppressed once
one language issue exists (a multilingual clip yields several language items).
**Playback windows are clamped to the available source-audio bounds** in the
record itself — the future UI can never seek outside the media.

**Structured priority** (never hides evidence, only orders it): `CRITICAL`
(mandatory source verification, possible overlap, clipped words) · `HIGH`
(unknown/low-confidence language, missing/partial alignment, uncovered meaningful
audio) · `NORMAL` (shot-boundary continuity, significant transients) · `LOW`
(weak generic transient candidates).

Optional exact-sample `review_clips/*.wav` (never sliced outside source audio).
Records carry enough for the future Tauri UI: seek to exact annotation time,
loop, 0.5x/1x/2x playback.

## Failure routes

faster-whisper/WhisperX failure → exact reason recorded
(`asr_status.json`) → FFmpeg audio evidence continues (waveform, spectrogram,
energy, regions, review queue) → full pipeline continues. A local ASR failure
NEVER authorizes Descript, OpenAI/Gemini/cloud transcription, or any media
upload — there is no such code path (tested by a repository sweep test).
Descript is disabled by default with no code path at all.

## Model bootstrap and privacy

`--asr-bootstrap` (default on) lets uv create worker envs and lets
faster-whisper/WhisperX download models to the local HF cache — permitted
bootstrap traffic (packages/models INTO the machine). Task media, audio, and
transcripts are NEVER uploaded anywhere. With `--no-asr-bootstrap` and a
missing env: `ASR_UNAVAILABLE`, evidence pipeline continues. `--no-asr`
disables ASR but never skips audio analysis.

## Batch-mode architecture note

Workers load the model per invocation today (single-clip CLI). The JSON
protocol deliberately separates request from process lifetime so a future
batch mode can hold a warm worker process (stdin/stdout framing) and reuse
the loaded model across tasks without changing the core engine contract.

## Validators

P3-AUDIO-001…009 (extraction, sample count, monotonicity, anchor sanity, bin
sequence/coverage, PCM cross-checks, **009 sample-anchor review required**),
P3-ASR-001…007 (word order, bounds, text preservation, recorded failure reasons,
**006 best word sequence == faster-whisper source**, **007 partial-alignment
coverage**), P3-SPEECH-001…004 (**004 every source-unverified speech region has an
unresolved mandatory-verification item**), P3-BOUNDARY-001/002/003 (**003
CONTINUOUS requires a crossing word or spectral continuity**), P3-TIME-001/002
(clock consistency, non-zero-PTS normalization), P3-QC-001 (PASS forbidden with
unresolved high-risk audio OR any source-unverified speech; review items,
crossing boundaries and unverified dialogue force REVIEW_REQUIRED, which
propagates to the top-level audit status).

## Known failure modes

1. ASR word timing on accented/overlapping/musical speech is unreliable —
   regions carry probabilities and land in review; timing is never presented
   as sample truth.
2. The silence threshold (−55 dBFS) can mask very quiet speech into
   LOW_LEVEL_BACKGROUND_AUDIO; the uncovered-audio defense only fires above
   the ACTIVE threshold. Human review of the waveform remains the backstop.
3. Boundary continuity uses energy + spectrum + actual word/segment crossing.
   Equal-energy source switches (frequency or tone→noise) read UNCERTAIN, not
   CONTINUOUS. A remaining edge case: two genuinely different sources that share
   both energy AND spectral shape across the cut could read CONTINUOUS — it stays
   review-required regardless, and L/J transitions are never auto-selected.
4. Windows SAPI TTS integration fixture verifies plumbing, not transcription
   quality on real-world audio.
5. WhisperX alignment models are per-language; unsupported languages fall
   back to faster-whisper timing (recorded).
