"""Structured Manuscript-style annotation via multi-pass vision.

Pass 1 (cast/global): from frames spanning the whole clip, establish a stable
cast (C1, C2, ... / O1, O2, ...) with detailed descriptors, plus scene, style,
audio, and concerns.

Pass 2 (per shot): for each detected shot, describe camera framing, timed camera
movements, and fine-grained timed Action & Audio lines that reference the cast
IDs from pass 1.

Requires the OpenAI vision backend (multi-image + JSON mode).
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import asr
from . import cuts as cuts_mod
from . import frames as frames_mod
from . import transcribe as tr
from .vision import OpenAIVisionBackend, image_content, text_content


# --------------------------------------------------------------------------
# data model
# --------------------------------------------------------------------------
@dataclass
class Entity:
    id: str
    description: str


@dataclass
class Globals:
    characters: list[Entity] = field(default_factory=list)
    objects: list[Entity] = field(default_factory=list)
    scene: str = ""
    style: str = ""
    audio: str = ""
    visual_concerns: str = ""
    audio_concerns: str = ""


@dataclass
class Timed:
    start: float
    end: float
    text: str


@dataclass
class Shot:
    index: int
    cut: str
    start: float
    end: float
    shot_type: str
    camera: str
    camera_movements: list[Timed]
    scene: str
    actions: list[Timed]
    playback_speed: str


@dataclass
class Annotation:
    video_name: str
    duration: float
    globals: Globals
    shots: list[Shot]


# --------------------------------------------------------------------------
# frame sampling → labelled content
# --------------------------------------------------------------------------
def _sample(
    grid: list[frames_mod.GridFrame], start: float, end: float, step: float
) -> list[frames_mod.GridFrame]:
    """Grid frames within [start, end] at ~step-second spacing (deduped)."""
    picked: dict[int, frames_mod.GridFrame] = {}
    t = start
    while t <= end + 1e-6:
        near = min(grid, key=lambda f: abs(f.time_seconds - t))
        if start - 1e-6 <= near.time_seconds <= end + 1e-6:
            picked[near.index] = near
        t += step
    return [picked[i] for i in sorted(picked)]


def _labelled(
    frames: list[frames_mod.GridFrame],
    shot_bounds: list[tuple[float, float, str]] | None = None,
) -> list[dict[str, object]]:
    content: list[dict[str, object]] = []
    for f in frames:
        label = f"Frame at t={f.time_seconds:.1f}s"
        if shot_bounds:
            for i, (s, e, _c) in enumerate(shot_bounds):
                if s <= f.time_seconds < e or (i == len(shot_bounds) - 1 and f.time_seconds >= s):
                    label += f" (Shot {i + 1})"
                    break
        content.append(text_content(label + ":"))
        content.append(image_content(f.path))
    return content


# --------------------------------------------------------------------------
# prompts
# --------------------------------------------------------------------------
_CAST_PROMPT = (
    "You are writing the [Overview] of a Manuscript-II video caption. The images are "
    "timestamped frames spanning the whole clip; each label says which SHOT it belongs to. "
    "Describe ONLY what is genuinely visible — never invent, never guess. If something is "
    "uncertain, omit it.\n\n"
    "CHARACTERS: assign C1, C2, ... in order of appearance to each DISTINCT person "
    "(disambiguate by appearance). Give a full head-to-toe description: apparent "
    "race/ethnicity (only if confident), apparent age range, build, hair "
    "(length/color/style), facial hair, and EVERY visible clothing item and accessory "
    "(jacket, shirt, tie/bow tie, dress cut/color, hat, glasses, watch, pocket square, "
    "lanyard, in-ear monitor). When the lower body or shoes are not visible, state "
    "'Lower body and shoes not visible.'. Use ONE aggregate character for an "
    "indistinguishable crowd. Do NOT invent names.\n"
    "IDENTITY RULES (critical — get these right):\n"
    "- ONE C-ID per real person. The SAME person often reappears across shots of a vlog "
    "or montage in different clothing or locations: judge by the FACE — if it is the same "
    "person, keep ONE C-ID and mention the outfit changes inside that one description. "
    "Never split one person into several IDs just because the outfit or room changed.\n"
    "- Different shots may show completely UNRELATED scenes or people (e.g. two different "
    "games, two different events). NEVER reuse a C-ID across shots for different people. "
    "When the clip has multiple shots, say in each description which shot(s) that "
    "character appears in (e.g. 'Appears in Shot 2 only.').\n"
    "- Every off-screen voice in the audio (narrator, commentator, unseen vocalist) gets "
    "its OWN C-ID with a description like 'Off-screen male commentator, never visible.' "
    "Shots must attribute speech to that C-ID — never to a bracket placeholder.\n"
    "OBJECTS: assign O1, O2, ... to LARGE or prominent objects referenced repeatedly "
    "(describe color/material/shape and any legible branding/text EXACTLY as written). "
    "Skip formal IDs for small common props. Only describe an object's placement in terms "
    "you can verify from the frames; do not overstate fixed positions when the camera moves.\n"
    "SCENE: the setting/space — spatial layout, positions relative to camera, lighting, "
    "background, foliage/decor. Describe what each character holds and their posture "
    "(static initial state), per character.\n"
    "STYLE: lighting, color, depth of field, film look, and aspect ratio (say 'No "
    "non-standard aspect ratio.' for standard widescreen).\n"
    "AUDIO: background audio, music, ambient sound ONLY (speech belongs in the shots). "
    "You CANNOT hear the audio — describe only what visible sources imply (a live band, a "
    "singer at a mic) and label sung words as [inaudible]; never fabricate lyrics.\n"
    "VISUAL CONCERNS: watermarks, letterbox bars, burned-in text, heavy shake, artifacts; "
    "else 'None.'. AUDIO CONCERNS: 'None.' unless a defect is visibly implied.\n\n"
    "Return STRICT JSON: {\"characters\":[{\"id\":\"C1\",\"description\":\"...\"}], "
    "\"objects\":[{\"id\":\"O1\",\"description\":\"...\"}], \"scene\":\"...\", "
    "\"style\":\"...\", \"audio\":\"...\", \"visual_concerns\":\"...\", "
    "\"audio_concerns\":\"...\"}."
)

_SHOT_PROMPT = (
    "Annotate ONE shot of a video in EXTREME detail to Manuscript-II standard. The images "
    "are closely-spaced timestamped frames of THIS shot only; use their timestamps to "
    "place every event precisely (0.1s precision).\n"
    "CAST (use these exact IDs consistently — never renumber):\n{cast}\n\n"
    "HARD RULES:\n"
    "1. NO PRONOUNS. Never write he/she/they/his/her/them. Refer to people only as C1, "
    "C2, ... and to objects as O1, O2, .... Use 'the left hand'/'the right arm', not "
    "'his hand'. List IDs individually (C1, C2, C3), never a range (not 'C1-C3').\n"
    "2. Every timestamp range must be DISTINCT — never reuse the same start/end for two "
    "entries. Overlapping actions get their own precise, different ranges.\n"
    "3. Camera motion goes ONLY in camera_movements — never in actions.\n"
    "4. Describe only what is visible; never invent. Only transcribe words you can truly "
    "read/lip-read — otherwise write [inaudible]. Never fabricate lyrics or dialogue.\n"
    "5. Reference ONLY cast members actually visible in THIS shot (plus off-screen voice "
    "C-IDs). Do not mention characters who belong to other shots.\n"
    "6. SINGER/SPEAKER ATTRIBUTION: attribute singing or speech to an on-screen character "
    "ONLY if that character is visibly performing it (at or holding a microphone, mouth "
    "clearly performing vocals). A person who is dancing is NOT the singer. If no visible "
    "performer exists, attribute to the off-screen voice C-ID from the cast.\n"
    "7. ON-SCREEN TEXT: transcribe ALL legible on-screen text VERBATIM in quotes — "
    "captions, subtitles, tweets, user comments, headlines, scoreboards, signs, logos "
    "with words. Put static text in the scene field; text that appears/changes gets its "
    "own timed action line (e.g. 'On-screen text appears: \\\"...\\\".'). Never write "
    "merely that text is present without quoting it; summarize only when many near-"
    "identical comments repeat the same words (quote the repeated words once).\n\n"
    "Produce, referencing cast IDs:\n"
    "- shot_type: e.g. 'long shot', 'medium-wide shot', 'close-up'.\n"
    "- camera: base framing + angle + field of view.\n"
    "- camera_movements: a SEPARATE {{start,end,text}} entry per distinct pan/tilt/zoom/"
    "drift/hold, split into consecutive intervals; note what becomes visible or hidden "
    "(e.g. 'Camera pans left; C2 and part of C3 come into frame.').\n"
    "- scene: the physical setting THIS shot, or 'No changes from overview.'.\n"
    "- actions: the core — as MANY fine-grained {{start,end,text}} lines as the footage "
    "supports (aim 25-40 for a 10s shot). Break EACH character's movements into separate "
    "short lines (~0.3-1.5s), one discrete action per line, e.g. 'C2 raises both arms "
    "overhead.', 'C3 extends the right arm toward screen-right.'. Track every visible "
    "character in parallel — overlapping/interleaved ranges are REQUIRED. Add separate "
    "AUDIO lines with their own ranges: ambient (e.g. 'Loud reception music continues "
    "throughout the shot.'), and speech/singing as verbatim quotes with tone when "
    "readable (e.g. 'C1 sings into O1: \\\"...\\\".') or '[inaudible]' when not. Every "
    "timestamp must lie within [{start:.1f}s, {end:.1f}s].\n"
    "- playback_speed: exactly one of 'regular', 'slow_motion', 'accelerated'.\n"
    "Return STRICT JSON: {{\"shot_type\":\"...\",\"camera\":\"...\","
    "\"camera_movements\":[{{\"start\":0.0,\"end\":0.5,\"text\":\"...\"}}],"
    "\"scene\":\"...\",\"actions\":[{{\"start\":0.0,\"end\":1.5,\"text\":\"...\"}}],"
    "\"playback_speed\":\"regular\"}}."
)


# --------------------------------------------------------------------------
# passes
# --------------------------------------------------------------------------
def _lang_label(transcript: tr.Transcript) -> str:
    """Rule 7 ladder: name the language only when the transcript is trustworthy;
    otherwise the safe, always-accepted answer is 'a foreign language'."""
    if transcript.language_confident and transcript.language:
        return transcript.language.capitalize()
    return "a foreign language"


def _cast_pass(
    backend: OpenAIVisionBackend, grid: list[frames_mod.GridFrame], duration: float,
    transcript: tr.Transcript, shot_bounds: list[tuple[float, float, str]],
) -> Globals:
    frames = _sample(grid, 0.0, duration, step=max(0.6, duration / 14))
    prompt = _CAST_PROMPT
    if len(shot_bounds) > 1:
        shot_list = "; ".join(
            f"Shot {i + 1} [{s:.1f}s-{e:.1f}s]" for i, (s, e, _c) in enumerate(shot_bounds)
        )
        prompt += (
            f"\n\nSHOT STRUCTURE: this clip has {len(shot_bounds)} shots: {shot_list}. "
            f"Apply the identity rules across them."
        )
    if transcript.has_speech:
        lang = _lang_label(transcript)
        if transcript.language_confident:
            prompt += (
                f"\n\nAUDIO TRANSCRIPT (verbatim, detected language = {lang}): "
                f'"{transcript.text}". In the Audio field, describe it accurately (note if '
                f"a voice is singing and that it is {lang}); the verbatim words belong in "
                f"the shots, not here. Add a C-ID for the performer only if one is visibly "
                f"performing vocals (at/holding a microphone); otherwise add an off-screen "
                f"voice C-ID."
            )
        else:
            prompt += (
                f"\n\nAUDIO: vocals/speech are audible but the words are NOT intelligible "
                f"(automatic transcription was unreliable — do not treat it as verbatim). "
                f"Describe the audio as vocals in {lang}; never quote specific words. Add "
                f"an off-screen voice C-ID unless a character is visibly performing vocals."
            )
    content = [text_content(prompt), *_labelled(frames, shot_bounds)]
    data = _complete_json(backend, content, max_tokens=2000)
    return Globals(
        characters=[Entity(e["id"], e["description"]) for e in data.get("characters", [])],
        objects=[Entity(e["id"], e["description"]) for e in data.get("objects", [])],
        scene=data.get("scene", ""), style=data.get("style", ""), audio=data.get("audio", ""),
        visual_concerns=data.get("visual_concerns", ""),
        audio_concerns=data.get("audio_concerns", ""),
    )


def _cast_text(g: Globals) -> str:
    lines = [f"{e.id}: {e.description}" for e in g.characters]
    lines += [f"{e.id}: {e.description}" for e in g.objects]
    return "\n".join(lines)


def _complete_json(
    backend: OpenAIVisionBackend, content: list[dict[str, object]],
    max_tokens: int, attempts: int = 3,
) -> dict[str, Any]:
    """Call the model in JSON mode, retrying transient empty/invalid responses."""
    last = ""
    for _ in range(attempts):
        last = backend.complete(content, json_mode=True, max_tokens=max_tokens)
        if last:
            try:
                parsed = json.loads(last)
                if isinstance(parsed, dict):
                    return parsed
            except json.JSONDecodeError:
                pass
    raise RuntimeError(f"no valid JSON after {attempts} attempts: {last[:200]!r}")


_LABELS = re.compile(r"\b([CO]\d+)\b")
_PRON = re.compile(r"\b(he|she|they|him|them|his|her|their|its|it)\b", re.IGNORECASE)
_POSSESSIVE = {"his", "her", "their", "its"}


def _depronoun(text: str) -> str:
    """Rule 5: replace pronouns with a character label. Only fires when the line
    references exactly ONE C#/O#, so the substitution is unambiguous — quoted
    speech is left untouched."""
    if '"' in text:  # never rewrite inside verbatim quotes
        return text
    labels = set(_LABELS.findall(text))
    if len(labels) != 1:
        return text
    label = next(iter(labels))
    return _PRON.sub(
        lambda m: f"{label}'s" if m.group(0).lower() in _POSSESSIVE else label, text
    )


def _clean(items: list[Timed]) -> list[Timed]:
    return [Timed(t.start, t.end, _depronoun(t.text)) for t in items]


def _timed_items(items: object) -> list[Timed]:
    out: list[Timed] = []
    if not isinstance(items, list):
        return out
    for x in items:
        if isinstance(x, dict) and "start" in x and "end" in x and "text" in x:
            out.append(Timed(float(x["start"]), float(x["end"]), str(x["text"])))
    return out


def _norm_speed(value: object) -> str:
    v = str(value).lower().replace(" ", "_")
    if "slow" in v:
        return "slow_motion"
    if "fast" in v or "accel" in v or "sped" in v:
        return "accelerated"
    return "regular"


def _shot_pass(
    backend: OpenAIVisionBackend, grid: list[frames_mod.GridFrame],
    cast: str, index: int, cut: str, start: float, end: float,
    transcript: tr.Transcript,
) -> Shot:
    frames = _sample(grid, start, end, step=0.25)
    prompt = _SHOT_PROMPT.format(cast=cast, start=start, end=end)
    segs = [s for s in transcript.segments if s.end > start and s.start < end]
    if segs:
        lang = _lang_label(transcript)
        good = [s for s in segs if s.reliable]
        bad = [s for s in segs if not s.reliable]
        if good:
            lines = "\n".join(f'  ({s.start:.1f}s-{s.end:.1f}s) "{s.text}"' for s in good)
            prompt += (
                f"\n\nVERBATIM AUDIO TRANSCRIPT for this shot (language = {lang}); these "
                f"are the ACTUAL spoken/sung words with real timestamps:\n{lines}\n"
                f"You MUST add each as an Action & Audio line with those exact timestamps "
                f"and EXACT words, no translation: '(start-end) C# sings in {lang}: "
                f"\\\"exact words\\\".' (use 'says' if spoken). Attribution follows HARD "
                f"RULE 6: only a character visibly performing vocals, else the off-screen "
                f"voice C-ID — never a dancer."
            )
        if bad:
            spans = "; ".join(f"({s.start:.1f}s-{s.end:.1f}s)" for s in bad)
            prompt += (
                f"\n\nUNRELIABLE AUDIO at {spans}: vocals are audible in these ranges but "
                f"the words are NOT intelligible (automatic transcription failed its "
                f"confidence check). Add an Action & Audio line per range in the form "
                f"'(start-end) C# sings in {lang}; the words are [inaudible].' — attribute "
                f"per HARD RULE 6 and NEVER invent or guess words for these ranges."
            )
    content = [text_content(prompt), *_labelled(frames)]
    data = _complete_json(backend, content, max_tokens=4000)
    return Shot(
        index=index, cut=cut, start=start, end=end,
        shot_type=data.get("shot_type", "shot"), camera=data.get("camera", ""),
        camera_movements=_clean(_timed_items(data.get("camera_movements", []))),
        scene=data.get("scene", ""),
        actions=sorted(_clean(_timed_items(data.get("actions", []))), key=lambda t: t.start),
        playback_speed=_norm_speed(data.get("playback_speed", "regular")),
    )


def analyze(
    video: Path, out_dir: Path, *, hz: float = 10.0,
    progress: Callable[[str, float], None] = lambda _s, _f: None,
) -> Annotation:
    video = Path(video)
    work = out_dir / video.stem
    progress("frames", 0.1)
    duration = frames_mod.probe_duration(video)
    grid = frames_mod.extract_grid(video, work / "frames", hz=hz)
    backend = OpenAIVisionBackend()

    progress("audio", 0.2)
    try:
        wav = asr.extract_audio(video, work / "audio.wav")
        transcript = tr.transcribe(wav)
    except Exception:  # audio is optional; never block the visual pass
        transcript = tr.Transcript(language="", text="", segments=[])

    # Shots FIRST: the cast pass needs the shot structure to keep identities
    # straight across cuts (different shots may show unrelated people).
    progress("shots", 0.3)
    resolved = cuts_mod.resolve_shots(backend, video, grid, duration)
    progress("cast", 0.4)
    g = _cast_pass(backend, grid, duration, transcript, resolved)
    cast_text = _cast_text(g)
    shots: list[Shot] = []
    for i, (s, e, cut) in enumerate(resolved):
        progress(f"shot {i + 1}/{len(resolved)}", 0.45 + 0.5 * i / len(resolved))
        shots.append(_shot_pass(backend, grid, cast_text, i + 1, cut, s, e, transcript))
    progress("done", 1.0)
    return Annotation(video_name=video.name, duration=duration, globals=g, shots=shots)
