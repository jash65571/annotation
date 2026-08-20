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
