# AutoScribe — plan

**Goal:** Drop a video → get one precise, complete, copy-paste description
(spoken words + on-screen action, timestamped). No review gate. Clean one-screen UI.

Standalone tool, separate from the locked Manuscript engine (different philosophy:
automatic best-effort description, clearly labeled — never claims "verified").

## Pipeline (upload → output)
1. **Probe / normalize** — ffprobe metadata; extract audio (`asr.wav`).
2. **Audio → words** — faster-whisper (large-v3-turbo) with word timestamps + VAD.
   (Already proven working in this repo's ASR workers.)
3. **Frames** — extract a dense grid (default **10 Hz = every 0.1 s**) plus
   ffmpeg scene-change detection → pick keyframes.
4. **Frames → action** — a pluggable **VisionBackend** describes keyframes; the
   description is carried across the 0.1 s grid and re-queried on scene change.
   Backends: cloud API (best) | local Ollama VLM (offline) | manual (fallback).
5. **Assemble** — merge speech + per-timestamp visual state into one markdown:
   overview, on-screen text, and a fine-grained timeline. One copy button.

## On "every 0.1s / every millisecond"
Frames ARE extracted at 0.1 s (that's the timeline resolution; per-ms is below the
25 fps source's real frame period of 40 ms, so it's meaningless). But running a
vision model on every near-identical frame is slow and adds nothing — so we describe
**keyframes + scene changes densely** and carry state between them. That is both more
precise (change-aligned) and fast. Adjustable: `--hz` and scene threshold.

## Decision that gates the build
Vision backend = the one external dependency. See the question in chat.

## Files
- `frames.py` — grid + scene-change extraction (done)
- `asr.py` — faster-whisper wrapper (done)
- `vision.py` — VisionBackend protocol + Cloud/Ollama/Manual (interface done)
- `assemble.py` — merge → markdown (done)
- `pipeline.py` — orchestrator (next)
- `webapp.py` + `static/index.html` — clean UI (next)
