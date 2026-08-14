# AutoScribe

Drop a video → get a **draft Manuscript-II caption** (cast, shots, cuts, camera
moves, timed Action & Audio, verbatim foreign-language lyrics) to review and
tighten by hand.

> ### AutoScribe output is a DRAFT, not a deliverable
>
> AutoScribe does **not** produce a Ready-To-Deliver caption and never marks one.
> It is a standalone tool that does not use the review-gated Manuscript engine,
> so it has no encoded-frame evidence ledger, no validator chain over structured
> facts, and no human signoff. Every run ends "NOT READY TO DELIVER" with a list
> of unresolved items, by design — a caption becomes RTD when a person has
> checked it against the video, not when a model stops objecting.
>
> ### Media leaves your machine
>
> In the default `structured` mode AutoScribe **uploads extracted audio and video
> frames to OpenAI** for transcription and vision analysis. This is the opposite
> of the Manuscript Reviewer engine's local-only guarantee. The web UI shows the
> notice for the active mode on every run. Do not use cloud mode for material you
> are not permitted to send to a third party.

---

## Setup on a new machine

### 1. Prerequisites
- **Python** (managed by [uv](https://docs.astral.sh/uv/) — install uv first).
- **ffmpeg + ffprobe** on the machine (FFmpeg 5.0 or newer):
  - Windows: download a static build, put `ffmpeg.exe`/`ffprobe.exe` in a folder
    (e.g. `C:\ffbin`) and point `AUTOSCRIBE_FFMPEG_DIR` at it, **or** add to PATH.
  - macOS: `brew install ffmpeg`   ·   Linux: `sudo apt install ffmpeg` (then it's on PATH).
- An **OpenAI API key** (used for both vision and audio transcription).

### 2. Install
```bash
git clone https://github.com/jash65571/annotation.git
cd annotation
uv sync --extra autoscribe      # engine + AutoScribe deps (scenedetect, faster-whisper)
```

### 3. Configure
```bash
cp .env.example .env
# edit .env: set OPENAI_API_KEY, and AUTOSCRIBE_FFMPEG_DIR if ffmpeg isn't on PATH
```
`.env` is gitignored. Any of these can also be real environment variables instead of a file.

| Variable | Required | Default | Purpose |
|----------|----------|---------|---------|
| `OPENAI_API_KEY` | ✅ | — | vision + audio transcription |
| `AUTOSCRIBE_FFMPEG_DIR` | if ffmpeg not on PATH | — | folder with ffmpeg/ffprobe |
| `AUTOSCRIBE_MODE` | — | `structured` | `structured` (Manuscript) or `flat` |
| `AUTOSCRIBE_VISION` | — | `openai` | `openai` \| `cloud` \| `ollama` \| `manual` |
| `OPENAI_MODEL` | — | `gpt-4o` | any vision-capable model |
| `AUTOSCRIBE_UNCLEAR_TOKEN` | — | `<unintelligible>` | unclear-speech token; set to `[inaudible]` only if the task UI asks |
| `AUTOSCRIBE_MAX_UPLOAD_MB` | — | `512` | upload size cap |
| `AUTOSCRIBE_MAX_JOBS` | — | `2` | concurrent analyses |

### 4. Run
```bash
uv run python -m autoscribe.webapp      # → http://localhost:8765
```
Open the URL, drop a video, watch progress, review the **unresolved list**, then
verify the draft against the video before using it. Output is also written to
`<video-dir>/out/<name>.manuscript.md`.

Programmatic use:
```python
from pathlib import Path
from autoscribe import structured, render
from autoscribe.validate import validate_caption

ann = structured.analyze(Path("clip.mp4"), Path("out"), hz=10.0)
caption = render.render(ann)
validate_caption(caption, ann.blockers)       # deterministic gate
ready, reason = ann.blockers.readiness()      # ready is always False: needs a human
print(ann.blockers.describe())
Path("clip.manuscript.md").write_text(caption, encoding="utf-8")
```

---

## How it works
1. **frames** — probe the encoded-frame ledger (`ffprobe` → real `pts_time` per
   frame) and extract the source frames nearest a 1/hz cadence. Every timestamp
   names a frame that actually exists, so the tool is VFR-safe.
2. **audio** (`transcribe.py`) — OpenAI Whisper transcribes verbatim
   speech/lyrics with word-level timestamps and detects the language.
   `audio_timeline.py` measures the waveform into **overlapping layers**
   (speech / music / sound / unresolved / quiet) — music under dialogue is both.
3. **shots** (`cuts.py`) — PySceneDetect (Adaptive + Content + fade) proposes
   candidates; the vision model verifies each. Candidates are clustered only at
   frame resolution, so genuinely short shots survive. A candidate that cannot be
   verified becomes a **blocking unresolved item**, never a silent "no cut".
   L-cuts and J-cuts are decided from audio crossing a boundary, never from frames.
4. **structured** — two vision passes (cast/global, then per-shot) produce timed
   Action & Audio + camera movements + speed changes.
5. **render** — canonical `[Overview]` / `[Shot N]` caption text (`Cast:`,
   `Camera Movements:`, `Playback Speed:`, `Speed Changes:`).
6. **validate** (`validate.py`) — deterministic gate over the rendered text:
   field names, quote balance, punctuation, C/O-IDs, pronouns, timestamps,
   transitions, protected traits. Runs on the draft **and again** after any
   reviewer rewrite.

## Source-of-truth compliance
AutoScribe targets `references/MANUSCRIPT-II-COMPLETE-SOURCE-OF-TRUTH.md`
(effective 13 Aug 2026) and the Aug 2026 evaluator feedback. Two of those
requirements are enforced by machinery, not just prompt text:

- **Descriptive depth** (§6.3/§6.4). The validator rejects a Scene that reads as
  an object inventory: it requires ~70 words, explicit spatial relationships,
  and named foreground / middle ground / background. Style must name light
  source **and direction**, **shadow quality**, and **colour temperature**.
  "A kitchen with a wooden table." and "Natural light." are blocking failures.
- **Supported tone** (§10 Rule 4). Speech lines must carry a tone, and that tone
  comes from `prosody.py` — measured loudness (vs. the clip's own speech
  median), pitch (autocorrelation f0, vs. the speaker's own norm) and pace
  (words/second). Any attribute that cannot be measured is `unresolved` and is
  omitted rather than guessed.

## Reviewer mode
Paste an attempter's seed caption (and optional evaluator feedback) to get an
audit, a 1-5 score, feedback, and a reviewed draft. The reviewer is given
**frames from the clip** plus the **measured facts** (shot boundaries, audio
timeline, unresolved items) alongside both captions, and applies the §1 source
hierarchy: the media outranks both captions; frames are sampled, so they can
prove an event happened but not that one did not. Conflicts it cannot settle
are returned as `unresolved` rather than guessed. Its rewrite is re-validated
before being written to disk.

## Known limits (read before trusting output)
- **No shared evidence layer with the engine.** AutoScribe measures its own
  frames/audio rather than consuming the Manuscript engine's ledger, so its
  claims carry frame/PTS pointers but not the engine's full proof trail.
- **The reviewer sees sampled frames, not the clip.** It cannot hear the audio
  and does not receive every frame, so it is a strong comparator — not a
  substitute for the human watch-through the source of truth requires.
- **Audio classes are coarse.** Ambience, sound effects and crowd reactions are
  not distinguished; that span is reported as undetermined, not guessed.
- **Speaker attribution has no diarization.** Speech is attributed
  conservatively and falls back to an off-screen voice C-ID when ambiguous.
- **Prosody is not emotion.** Loudness, pitch and pace are measured; inferring
  "angry" or "teasing" from them remains a human judgement.
- **Only the language DECLARATION is enforced, not the audio prose.** The
  `Spoken Language:` field is tool-owned and checked exactly. The model's
  `Audio:` prose is outside that check: an unhedged guess written there
  (e.g. "The voice speaks Pashto.") cannot be detected without a language list
  that would never be complete, and four attempts at inferring it from prose
  failed in both directions. A hedge in that prose raises a non-blocking
  advisory; anything stronger needs the human read-through.
- **Cost:** ~1 audio + 1 cast + N cut-verify + 1 per-shot vision call per video.

## Development
```bash
uv sync --dev --extra autoscribe
uv run pytest tests/test_autoscribe_*.py
uv run ruff check .
uv run mypy                     # autoscribe is in the strict package list
```

- **Security:** never commit `.env`; rotate a key if it's ever exposed.
