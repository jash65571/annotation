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
duration, nb_samples) → `audio_frames.csv/.jsonl`. Codec priming: the stream's
declared `initial_padding` is captured into `AudioTimeline
.initial_padding_samples`; AAC is never assumed to start at sample zero — the
first decoded audio frame's PTS anchors the PCM, and
`annotation time of sample N = annotation_audio_offset + N/sample_rate`
(exact Fraction math, `audio/timeline.py`, no float accumulation).

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
int8, ~23 s for a 16 s clip). Model cache state (cached/downloaded) is
recorded per run.

## VAD recall defense

Absence of VAD/ASR output NEVER proves silence. ACTIVE_AUDIO windows without
ASR coverage (≥ 0.4 s) become `AUDIO_WITHOUT_ASR_COVERAGE` review items.
**Decision**: no automatic second no-VAD transcription pass — uncovered
regions go to human review instead, because a human listen is the only
trustworthy resolution for VAD-missed material and double transcription of
every clip costs more than it recovers. Revisit if review queues show a
pattern of recoverable misses.

## WhisperX (alignment only)

Isolated uv worker (`audio/asr/workers/wx_env`, pinned **whisperx==3.4.3**,
CPU-only torch via the pytorch-cpu index for reproducibility). It receives the
faster-whisper transcript and must PRESERVE wording; the text-preservation
check (whitespace-collapsed, case-preserving normalization) demotes any
change to `ALIGNMENT_TEXT_MISMATCH` — the changed transcript is never
silently accepted. Alignment failure (missing language model, dependency,
audio issue) records the reason, keeps the faster-whisper result as
`transcript_best`, and the pipeline continues.

`transcript_best`: wording = faster-whisper; timing = WhisperX when validly
aligned, else faster-whisper. Header carries
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

Every SUPPORTED visual boundary gets a ±0.5 s audio window: energy
before/after, silence overlap, speech-region/word crossing →
CONTINUOUS / DISCONTINUOUS / UNCERTAIN / NO_AUDIO, and Phase 2's
`audio_verification_required` resolves to CHECKED_NO_CROSSING /
CROSSING_REVIEW_REQUIRED / UNAVAILABLE (`boundary_audio_evidence.json`).
**L-cut/J-cut is never auto-finalized** — waveform continuity alone cannot
prove the semantic source relation (P3-BOUNDARY-002).

## Review queue and clips

`audio_review_queue.json`: every item names WHY to listen (reason enum), the
exact annotation window, a padded playback window, evidence refs, and the ASR
text candidate. Optional exact-sample `review_clips/*.wav` (never sliced
outside source audio). Records carry enough for the future Tauri UI: seek to
exact annotation time, loop, 0.5x/1x/2x playback.

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

P3-AUDIO-001…008 (extraction, sample count, monotonicity, anchor sanity, bin
sequence/coverage, PCM cross-checks), P3-ASR-001…005 (word order, bounds,
text preservation, recorded failure reasons, no-silent-skip), P3-SPEECH-001…003,
P3-BOUNDARY-001/002, P3-TIME-001/002 (clock consistency, non-zero-PTS
normalization — enforced by construction through AnnotationClock + tests),
P3-QC-001 (PASS forbidden with unresolved high-risk audio; review items and
crossing boundaries force REVIEW_REQUIRED, which propagates to the top-level
audit status).

## Known failure modes

1. ASR word timing on accented/overlapping/musical speech is unreliable —
   regions carry probabilities and land in review; timing is never presented
   as sample truth.
2. The silence threshold (−55 dBFS) can mask very quiet speech into
   LOW_LEVEL_BACKGROUND_AUDIO; the uncovered-audio defense only fires above
   the ACTIVE threshold. Human review of the waveform remains the backstop.
3. Boundary continuity uses energy + region crossing; a hard audio edit that
   happens to match levels reads CONTINUOUS (review-required anyway when
   audible material crosses).
4. Windows SAPI TTS integration fixture verifies plumbing, not transcription
   quality on real-world audio.
5. WhisperX alignment models are per-language; unsupported languages fall
   back to faster-whisper timing (recorded).
