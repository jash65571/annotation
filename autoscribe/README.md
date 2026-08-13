# AutoScribe

Drop a video → get a precise, copy-paste **Manuscript-II caption** (cast, shots,
cuts, camera moves, timed Action & Audio, verbatim foreign-language lyrics).
Automatic — a first-draft generator, verified against the Manuscript rules.

Standalone tool; it does **not** touch the review-gated Manuscript engine.

---

## Setup on a new machine

### 1. Prerequisites
- **Python** (managed by [uv](https://docs.astral.sh/uv/) — install uv first).
- **ffmpeg + ffprobe** on the machine:
  - Windows: download a static build, put `ffmpeg.exe`/`ffprobe.exe` in a folder
    (e.g. `C:\ffbin`) and point `AUTOSCRIBE_FFMPEG_DIR` at it, **or** add to PATH.
  - macOS: `brew install ffmpeg`   ·   Linux: `sudo apt install ffmpeg` (then it's on PATH).
- An **OpenAI API key** (used for both vision and audio transcription).

### 2. Install
```bash
git clone https://github.com/jash65571/annotation.git
cd annotation
uv sync --extra autoscribe      # installs the engine + AutoScribe deps (incl. scenedetect)
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

### 4. Run
```bash
uv run python -m autoscribe.webapp      # → http://localhost:8765
```
Open the URL, drop a video, watch progress, click **Copy**. Output is also written to
`<video-dir>/out/<name>.manuscript.md`.

Programmatic use:
```python
from pathlib import Path
from autoscribe import structured, render
ann = structured.analyze(Path("clip.mp4"), Path("out"), hz=10.0)
Path("clip.manuscript.md").write_text(render.render(ann), encoding="utf-8")
```

---

## How it works
1. **frames** — extract a 0.1 s frame grid.
2. **audio** (`transcribe.py`) — OpenAI Whisper transcribes verbatim speech/lyrics with
   timestamps and detects the language (Rules 6 & 7).
3. **shots** (`cuts.py`) — PySceneDetect (Adaptive + sensitive Content) proposes candidate
   cuts; the vision model verifies each to reject motion/lighting false-cuts and confirm
   real ones → correct shot **count** + cut type.
4. **structured** — two vision passes (cast/global, then per-shot) produce timed Action &
   Audio + camera movements; enforces no-pronouns, distinct 0.1 s timestamps, Manuscript
   field structure.
5. **render** — canonical `[Overview]` / `[Shot N]` caption text.

## Notes
- **Cost:** ~1 audio + 1 cast + N cut-verify + 1 per-shot vision call per video (a few cents).
- **Accuracy:** cast, shot count, cuts, and foreign-language lyrics are strong; action
  *timestamps* are close but not frame-exact — that last bit is what human review tightens.
- **Security:** never commit `.env`; rotate a key if it's ever exposed.
