# Manuscript II Audio Review Pipeline

Windows setup for the Manuscript II audio-review helper.

The tool takes a source video and creates audio evidence for human review. Automatic transcript, pitch, emotion, and speaker-change signals are leads only. The reviewer must still listen to the actual audio.

## What the pipeline does

1. Extracts a 16 kHz mono WAV with FFmpeg.
2. Runs WhisperX 3.8.6 with `large-v3`, CPU, and INT8 compute.
3. Reads word-level aligned transcript evidence.
4. Flags low-confidence transcript words.
5. Extracts acoustic features with openSMILE.
6. Runs Wav2Vec2 emotion evidence.
7. Checks large pitch changes between adjacent WhisperX segments.
8. Builds a human-review queue.
9. Creates short WAV clips around flagged regions.
10. Validates generated review-clip durations.

The system never automatically chooses final tone or speaker identity.

## Requirements

Install these before running setup:

- Git for Windows
- Python 3.11, 64-bit, including the `py` launcher
- FFmpeg and FFprobe on Windows PATH
- Internet access for Python packages and first-run model downloads

Check them in PowerShell:

```powershell
git --version
py -3.11 --version
ffmpeg -version
ffprobe -version
```

## Fresh laptop setup

Clone the `autoscribe` branch:

```powershell
cd "$env:USERPROFILE\Desktop"
git clone --branch autoscribe https://github.com/jash65571/annotation.git
cd ".\annotation\tools\manuscript-audio-review"
```

Run the installer:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File ".\setup_windows.ps1"
```

A successful setup ends with:

```text
SETUP COMPLETE
```

The installer creates two isolated environments:

```text
.venv-review\
.venv-whisperx\
```

`.venv-review` contains the acoustic and emotion analysis stack.

`.venv-whisperx` contains WhisperX and its dependency stack.

You do not need to activate either environment manually.

## Run a video

From the `tools\manuscript-audio-review` directory:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File ".\review_manuscript_audio.ps1" "C:\path\to\your\video.mp4"
```

If local scripts are already allowed, the shorter form also works:

```powershell
.\review_manuscript_audio.ps1 "C:\path\to\your\video.mp4"
```

Video filenames may contain spaces.

### Auto-ingest the locked task seed (Phase 3.5)

Pass a **seed file** as the second argument and the pipeline parses the real
Manuscript task (C# cast, O# objects, locked shot ranges) into
`task_context.json` **before** analysis. Every stage then becomes shot-aware:
ASR words, sound candidates, transients, masking checks, review windows, and
speaker evidence all carry `shot: N`.

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File ".\review_manuscript_audio.ps1" "C:\path\to\your\video.mp4" "C:\path\to\seed.txt"
```

Accepted seed forms (same parser as `manuscript_audio_seed.py`):

```text
C1: a man in a red shirt
O2 - a wooden chair
Shot 1: 0.0 - 4.0
Shot 2 4.00-10.50
```

If no seed is passed, an existing `task_context.json` bound to the current
video is reused; otherwise the pipeline runs shot-less (every finding stays
`shot: null`) and `task_context.json` can be written later and the pipeline
re-run.

A successful run ends with:

```text
SOURCE PREPROCESSING: PASS
PIPELINE STATUS: PASS
AUDIO REVIEW PIPELINE COMPLETE
```

## Outputs

Generated files stay inside the tool directory and are ignored by Git.

```text
output\VIDEO.json
analysis\audio.wav
analysis\manuscript_audio_evidence.json
analysis\audio_review_queue.json
analysis\review_clips\
analysis\review_clips\review_clips_manifest.json
```

`output\VIDEO.json` is the stable internal WhisperX transcript path. A source-named WhisperX JSON is copied into that stable path after each run.

`manuscript_audio_evidence.json` contains transcript, acoustic, emotion, continuity, and review-synthesis evidence.

`audio_review_queue.json` identifies regions that deserve closer listening.

`review_clips` contains short WAV files around those flagged regions.

Phase 3.5 hardening (see `3B_VERIFICATION.md` for the earlier phases):

- **Sequence-aware ASR matching** -- the same lexical word heard by both
  ASR models a few hundred ms apart counts as agreement (normalized text +
  center-time tolerance), instead of being misread as a coverage gap or a
  weak unrelated word.
- **Targeted reruns are hypotheses** -- rerun output reads as "Targeted rerun
  hypothesis", never recovered truth, and is marked `CONFLICT` when it
  disagrees with surrounding evidence (no independent speech signal, or
  content that contradicts the primary transcript).
- **Transient/SFX detector** -- an independent detector (short-time RMS,
  spectral flux, onset strength, broadband energy change, crest factor)
  turns strong unexplained transients into high-priority review windows
  even when no model can name the sound.
- **Split door classes** -- `doorbell_chime`, `door_open_close`,
  `door_latch_click`, and `door_knock` are separate candidate classes; the
  physical door and the electronic chime are never collapsed.
- **Stricter masking** -- a masking warning requires a real overlapping
  source AND intelligibility loss. Anything else is just
  "Low-confidence speech: re-listen".
- **Deduplicated review report** -- findings are grouped by time window, so
  one questionable word no longer generates five repetitive review items.
- **Music mix_role left blank** -- recorded level may be estimated, but the
  mix role (e.g. `Foreground`) is never auto-filled for human review.
- **Honest media metadata** -- both `source_sample_rate` (original video)
  and `analysis_sample_rate` (the resampled 16 kHz WAV) are reported.

Phase 3.6 fixes (from real-task failures):

- **ASR multi-stream divergence gate** -- when the two ASR models align to
  DIFFERENT concurrent vocal content (foreground speech vs lyrics/background
  vocals), the word-level match stops and ONE `MULTI_STREAM_ASR_DIVERGENCE`
  region is reported instead of dozens of fake per-word conflicts. Words in
  the region never become individual risks.
- **Proper-noun filter repaired** -- only mid-sentence capitalized tokens
  (real name-like words) are flagged; sentence-start capitalization and
  lowercase common words like `percussion`, `Check`, `him` are never called
  proper nouns.
- **Masking queue depends on final masking evidence** -- the review queue
  only creates `masking_check` from `masking_overlap_evidence.json`; the
  sound-fusion layer never independently infers masking, and WEAK sounds
  can never drive a masking claim.
- **WEAK sounds never explain transients** -- only MEDIUM/STRONG named
  sounds can explain or demote a transient; a WEAK cheering guess (CLAP
  0.218) leaves the transient unexplained and STRONG.
- **Cross-shot evidence carries `shots: [...]`** -- whole-clip music, long
  reruns, and multi-shot transients get a `shots` list instead of one
  forced (wrong) shot; whole-clip music is marked `scope: whole_clip`.
- **Vision environment installed by default setup** -- `setup_windows.ps1`
  now runs `setup_vision_windows.ps1` best-effort, so Phase 3B face
  tracking works on a fresh clone instead of requiring a manual install.
- **Unavailable face evidence is not "no face"** -- when face tracking
  did not run, the packet says "Visible-speaker evidence unavailable",
  never "No visible face during this speech window".
- **Long transient spans are regions** -- merged peaks > 1 s are labeled
  `high_energy_acoustic_region` with the individual peak times underneath,
  not one giant "transient".
- **REVIEW_ME wording** -- STRONG findings are "HIGH PRIORITY — strong
  evidence", never "safe defaults".

Real-task follow-up hardening:

- **Tokenization-equivalent ASR matching** -- concatenated token groups such as
  `shitface` versus `shit` + `face`, `goodbye` versus `good bye`, and similar
  contractions are classified as `tokenization_equivalent`, not conflicts.
- **Speech-gated clip tails** -- a long tail triggers an ASR rerun only when
  VAD/diarization independently supports continued speech; otherwise it gets a
  listen-only `clip_tail_check`.
- **Separate ASR metrics** -- reports distinguish lexical agreement from
  high-confidence cross-model confirmation; neither is presented as transcript
  accuracy.
- **Peak-level transient explanation** -- mixed detector regions split at
  speech/source boundaries, preserving an unexplained chair scrape or impact
  beside speech-associated energy. `chair_scrape` and `furniture_scrape` are
  controlled sound classes with dedicated CLAP prompts.
- **Contraction-safe proper nouns** -- first-person and common contractions such
  as `I'm`, `We're`, and `You're` never become name-risk findings.

## Reviewer environment

Pinned direct packages:

```text
numpy==2.4.6
opensmile==2.6.0
soundfile==0.14.0
torch==2.13.0
transformers==5.15.1
```

They are installed from `requirements-review.txt`. `silero-vad==5.1.2` is
installed separately, best-effort, by `setup_windows.ps1`: it is the optional
independent speech-presence fallback for untranscribed-speech detection when
diarization is skipped. If it fails to install, the pipeline still runs and
coverage falls back to diarization (or is honestly reported as UNKNOWN).

## WhisperX environment

Pinned package:

```text
whisperx==3.8.6
```

It is installed from `requirements-whisperx.txt`.

## Vision environment (Phase 3B, optional)

Pinned packages (`requirements-vision.txt`):

```text
mediapipe==0.10.21
opencv-python-headless==4.10.0.84
numpy<2
```

Installed into `.venv-vision` by `setup_vision_windows.ps1`, which the main
`setup_windows.ps1` now runs best-effort (step 8/8). A failure here does NOT
block the core review stack: face tracking just degrades to unavailable
speaker evidence. If it was skipped during setup, install it later with:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File ".\setup_vision_windows.ps1"
```

Without it, Phase 1.6 face tracking is skipped and speech windows are
reported as "Visible-speaker evidence unavailable" rather than claiming no
face is visible.

## Troubleshooting

### Python 3.11 not found

Run:

```powershell
py -0p
```

Install Python 3.11 if it is missing, then reopen PowerShell.

### FFmpeg not found

Run:

```powershell
Get-Command ffmpeg
Get-Command ffprobe
```

If either command fails, add the FFmpeg `bin` directory to Windows PATH and reopen PowerShell.

### PowerShell blocks scripts

Run the script through a one-process bypass:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File ".\setup_windows.ps1"
```

or:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File ".\review_manuscript_audio.ps1" "C:\path\to\video.mp4"
```

### Verify the reviewer environment

```powershell
.\.venv-review\Scripts\python.exe -c "import numpy, opensmile, soundfile, torch, transformers; print('REVIEW PASS')"
```

### Verify WhisperX

```powershell
.\.venv-whisperx\Scripts\python.exe -c "import whisperx; print('WHISPERX PASS')"
.\.venv-whisperx\Scripts\python.exe -m whisperx --version
```

### First run is slow

The first WhisperX or Hugging Face run may download model files. Later runs reuse the local cache.

## Updating on another laptop

From the cloned repository:

```powershell
git checkout autoscribe
git pull origin autoscribe
cd ".\tools\manuscript-audio-review"
```

If dependency files changed, rerun:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File ".\setup_windows.ps1"
```

## Git safety

The tool-level `.gitignore` excludes:

- `.venv-review`
- `.venv-whisperx`
- `analysis`
- `output`
- video files
- generated audio files
- model/cache files
- Python cache files

Do not force-add task videos, transcripts, or generated evidence.

## Manuscript II review rule

This pipeline is an evidence helper, not annotation authority.

- ASR text must be checked by listening.
- Low confidence means review the word, not automatically mark it wrong.
- Emotion output does not decide final tone.
- Pitch changes do not identify speakers.
- Actual clip audio remains the factual source of truth.

## Version 2 Audio Reviewer

Version 2 adds a much stronger Manuscript II audio-review workflow.

### What Version 2 does

- Runs WhisperX large-v3 transcription.
- Creates word-level timing evidence.
- Checks speech across shot boundaries.
- Merges duplicate transcript-review regions.
- Runs optional speaker diarization.
- Flags suspicious speaker clusters.
- Supports human-confirmed speaker-to-character mapping.
- Builds Character Voice evidence.
- Detects possible non-speech sounds.
- Checks sound continuity across shots.
- Checks possible clipping, echo, and wind noise.
- Separates ordinary overlap from possible masking.
- Creates targeted WAV clips for manual listening.
- Creates a readable Manuscript reviewer report.
- Runs a final validator preflight.
- Protects against old task data being reused on a new video.

### Normal use

Put the video in this folder and run:

```powershell
.\review_manuscript_audio.ps1 ".\video.mp4"
```

## Version 2.1 — consolidated review packet + companion tools

Version 2.1 adds one consolidated packet so you no longer reconcile a dozen
JSON files by hand, plus three companion tools.

### One packet to read

The pipeline now ends with a master aggregator (`manuscript_audio_master.py`,
Phase 7). It reads every per-stage evidence file and writes:

```text
analysis\manuscript_audio_review_packet.json   # master: all sections + ranked findings
analysis\REVIEW_ME.md                          # human-readable summary
analysis\manuscript_audio_ui_suggestions.json  # sparse, MEDIUM+ UI field suggestions
```

Every conclusion carries one shared confidence tier — STRONG / MEDIUM / WEAK /
CONFLICT / UNKNOWN — plus the signals it came from. `REVIEW_ME.md` groups
findings into "STRONG (safe defaults)", "NEEDS REVIEW", and "DO NOT
AUTO-ASSERT". UI suggestions stay sparse on purpose: a field appears only with
MEDIUM+ evidence; pitch, tone, clarity, texture, and speaking level are left
blank for human listening.

### Untranscribed-speech detection

The aggregator diffs an independent speech-presence signal against the ASR
segments. Where speech exists but ASR produced no words, it emits
`UNTRANSCRIBED_SPEECH` with a coverage ratio, so late or dropped speech is
never silently lost. The signal is diarization turns when available, otherwise
Silero VAD (`manuscript_audio_vad.py`, added to `requirements-review.txt`). If
neither is available, coverage is honestly reported as UNKNOWN.

### Task-seed parser

```powershell
.\.venv-review\Scripts\python.exe manuscript_audio_seed.py .\seed.txt
```

Parses the pasted live task into `task_context.json` (C#/O# ids + descriptions,
locked shot boundaries) and records the original cast/object baseline under
`seed_meta`. The live locked task then drives audio review.

### Pasted-back QA (predict the blockers)

A separate surface for after you fill the live UI. Paste the filled event
fields + generated caption / Final Audio Text into a state JSON and run:

```powershell
.\.venv-review\Scripts\python.exe manuscript_audio_qa.py .\state.json
```

It predicts export blockers before Handshake: `STRUCTURED_FIELD_MISSING` (prose
claims "moderate recorded level" while the field is blank), a Not-observed
character used as a Speech source, a silent object used as a sound source,
deletion of original cast, numeric timestamps or past tense in Final Audio
Text, and dialogue missing from the final prose.

### Regression lock

`test_regression_clip.py` locks the reference applause clip's ground truth and
every discipline rule (weak music stays weak, overlap is not masking, defects
stay UNKNOWN, late speech is recovered). Run it any time — no video, no models:

```powershell
.\.venv-review\Scripts\python.exe test_regression_clip.py
```
