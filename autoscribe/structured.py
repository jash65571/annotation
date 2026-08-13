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
import os
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import asr
from . import audio_timeline
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
    "EXACTLY: 'Lower body and shoes are not visible.'. Use ONE aggregate character for "
    "an indistinguishable crowd. Do NOT invent names.\n"
    "NO PRONOUNS ANYWHERE IN THE OVERVIEW: never write he/she/him/her/his/hers/they/"
    "them/their in any description or the scene. Repeat the C-ID or use neutral "
    "phrasing: 'C2 wears a black cap.', 'The hair is shoulder-length.', 'C3 raises the "
    "right hand.' — never 'He wears', 'her hair', 'their table'.\n"
    "IDENTITY RULES (critical — get these right):\n"
    "- ONE C-ID per real person. The SAME person may reappear across shots of a vlog "
    "or montage in different clothing or locations: judge by the FACE — if it is the same "
    "person, keep ONE C-ID and mention the outfit changes inside that one description.\n"
    "- CLOTHING LOCK: within footage of the SAME visit/scene (same location, minutes "
    "apart), a person CANNOT change clothes. If two people wear clearly different tops "
    "(e.g. a gray short-sleeve tee vs a black high-neck top) in nearby shots of the same "
    "place, they are TWO different people — give them separate C-IDs even if hair and "
    "build look similar. Merging two people into one C-ID is a critical error; when "
    "unsure, keep them separate.\n"
    "- Never write 'cannot be determined', 'not verifiable', or similar hedges in a "
    "description. State what IS supported and simply omit unknown traits.\n"
    "- Different shots may show completely UNRELATED scenes or people (e.g. two different "
    "games, two different events). NEVER reuse a C-ID across shots for different people. "
    "When the clip has multiple shots, say in each description which shot(s) that "
    "character appears in (e.g. 'Appears in Shot 2 only.').\n"
    "- Every off-screen voice in the audio (narrator, commentator, unseen vocalist) gets "
    "its OWN C-ID with a description like 'Off-screen male commentator, never visible.' "
    "Shots must attribute speech to that C-ID — never to a bracket placeholder.\n"
    "OBJECTS: assign O1, O2, ... to LARGE or prominent objects referenced repeatedly "
    "(describe color/material/shape and any legible branding/text EXACTLY as written). "
    "ONE O-ID = ONE object type: never combine distinct types under a single ID (water "
    "carafes and utensil cups are separate O-IDs or stay inline; grill grates and "
    "tabletop burners are separate). Skip formal IDs for small common props. Only "
    "describe an object's placement in terms you can verify from the frames; do not "
    "overstate fixed positions when the camera moves.\n"
    "SCENE: STABLE facts only — the setting/space, spatial layout, positions relative "
    "to camera, lighting, background, decor, each character's initial posture and "
    "initially held objects. NO actions, NO gestures, NO dialogue or narration quotes, "
    "NO shot-by-shot progression — every timed event belongs in the shots, never here.\n"
    "STYLE: lighting, color, depth of field, film look, and aspect ratio (say 'No "
    "non-standard aspect ratio.' for standard widescreen).\n"
    "AUDIO: background audio, music, ambient sound ONLY (speech belongs in the shots). "
    "You will be given a MEASURED AUDIO TIMELINE (signal analysis of the real "
    "soundtrack). Treat it as fact: describe exactly the music/ambience/laughter it "
    "reports, with its time ranges. Non-diegetic music needs NO visible source — never "
    "delete or omit measured music because no performer is on screen, and never invent "
    "audio the timeline does not show. Never fabricate lyrics; unheard sung words are "
    "[inaudible].\n"
    "VISUAL CONCERNS: real source defects only — watermarks, letterbox bars, burned-in "
    "text, heavy shake, compression artifacts, glare; else 'None.'. Collage layouts, "
    "borders, and split-screen presentation are STYLE, not concerns; motion blur is a "
    "concern only when it blocks useful detail. AUDIO CONCERNS: 'None.' unless the "
    "measured audio shows a real defect (silence, clipping, dropouts) — a description "
    "of audible content is NOT a concern.\n\n"
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
    "3. Camera motion goes ONLY in camera_movements — never in actions. GRAPHIC motion "
    "is NOT camera motion: sliding collage panels, expanding circular masks, moving "
    "text/overlays are EDITS — describe them as timed action lines, and put an entry in "
    "camera_movements only when the real viewpoint itself moves.\n"
    "3b. TRUE EVENT TIMING: each line's start is the first frame where the event is "
    "actually visible and its end is the last. NEVER emit sliding/rolling template "
    "windows (e.g. 0.4-0.7, 0.5-0.9, 0.6-1.0, 0.7-1.1 ...) — timestamps must follow "
    "real event boundaries, not a fixed stride. A static or continuing state gets ONE "
    "line spanning its true duration, not a new line every 0.1s. Write as many lines as "
    "there are REAL distinct events — no more, no fewer. SPLIT TRIGGERS: if a line "
    "contains 'then', 'followed by', 'afterward', 'before', 'while', 'as', more than "
    "one quoted speech act, an action and a sound with different timing, or two people "
    "acting independently — split it, each part with its own true range.\n"
    "4. Describe only what is visible; never invent. Only transcribe words you can truly "
    "read/lip-read — otherwise write [inaudible]. Never fabricate lyrics or dialogue.\n"
    "5. Reference ONLY cast members actually visible in THIS shot (plus off-screen voice "
    "C-IDs). Do not mention characters who belong to other shots.\n"
    "6. SINGER/SPEAKER ATTRIBUTION: attribute singing or speech to an on-screen character "
    "ONLY if that character is visibly performing it (at or holding a microphone, mouth "
    "clearly performing vocals). A person who is dancing is NOT the singer. If no visible "
    "performer exists, attribute to the off-screen voice C-ID from the cast.\n"
    "7a. SPEECH INTEGRITY: every speech line needs the correct C-ID, off-screen status "
    "when the speaker is not visible ('C1 says off-screen ...'), audible delivery/tone "
    "ONLY when clearly supportable (firm, teasing, loud, urgent), and the EXACT words — "
    "preserve fillers, repeats, stutters, false starts, and cutoffs verbatim. The "
    "timestamp covers the audible words only, never surrounding silence.\n"
    "7b. NO TIMING FILLER: never write 'near the end', 'around 5 seconds', 'partway "
    "through', 'in the final frames' — the timestamps carry the timing.\n"
    "7c. NO FILLER LINES: never write 'No speech.', 'No dialogue.', 'No music.' or "
    "similar — absence is expressed by omission.\n"
    "7d. STABLE STATE: a fact that holds for the WHOLE shot (e.g. 'C1 holds a thumbs-up "
    "at screen-right throughout.') belongs in the scene field, not as an action line "
    "and never as repeated near-identical lines.\n"
    "8. ON-SCREEN TEXT: transcribe ALL legible on-screen text VERBATIM in quotes — "
    "captions, subtitles, tweets, user comments, headlines, scoreboards, signs, logos "
    "with words. Put static text in the scene field; text that appears/changes gets its "
    "own timed action line (e.g. 'On-screen text appears: \\\"...\\\".'). Never write "
    "merely that text is present without quoting it; summarize only when many near-"
    "identical comments repeat the same words (quote the repeated words once).\n\n"
    "Produce, referencing cast IDs:\n"
    "- shot_type: e.g. 'long shot', 'medium-wide shot', 'close-up'.\n"
    "- camera: shot size + angle + framing + stability (e.g. 'Medium-wide, eye-level, "
    "handheld with subtle organic drift.'). Never write 'locked static' when ANY drift "
    "or movement is visible; static means truly static.\n"
    "- camera_movements: a SEPARATE {{start,end,text}} entry per distinct pan/tilt/zoom/"
    "drift/hold, split into consecutive intervals; note what becomes visible or hidden "
    "(e.g. 'Camera pans left; C2 and part of C3 come into frame.').\n"
    "- scene: STABLE shot-specific facts only (setting, layout, whole-shot states), or "
    "'No changes from overview.'. No timed motion, no speech, no narration here.\n"
    "- actions: the core — one fine-grained {{start,end,text}} line per REAL distinct "
    "event, timed to its true first and last frame (rule 3b). Break EACH character's "
    "movements into separate lines, one discrete action per line, e.g. 'C2 raises both "
    "arms overhead.', 'C3 extends the right arm toward screen-right.'. Track every "
    "visible character in parallel — truthful overlapping/interleaved ranges are "
    "REQUIRED, but never pad with template windows or re-describe an unchanged state. "
    "Add separate AUDIO lines with their own ranges ONLY for audio reported by the "
    "measured audio facts for this shot (music, ambience, laughter) and the verbatim "
    "transcript words given below. If the audio facts show no speech in this shot, do "
    "NOT write any narration or dialogue line. Every timestamp must lie within "
    "[{start:.1f}s, {end:.1f}s].\n"
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


def _audio_facts(spans: list[audio_timeline.AudioSpan], start: float, end: float) -> str:
    """The measured audio timeline clipped to [start, end], as prompt lines."""
    rows = []
    for sp in spans:
        s, e = max(sp.start, start), min(sp.end, end)
        if e - s >= 0.15:
            rows.append("  " + audio_timeline.AudioSpan(round(s, 1), round(e, 1), sp.label).describe())
    return "\n".join(rows)


def _cast_pass(
    backend: OpenAIVisionBackend, grid: list[frames_mod.GridFrame], duration: float,
    transcript: tr.Transcript, shot_bounds: list[tuple[float, float, str]],
    spans: list[audio_timeline.AudioSpan],
) -> Globals:
    frames = _sample(grid, 0.0, duration, step=max(0.6, duration / 14))
    prompt = _CAST_PROMPT
    if spans:
        prompt += (
            "\n\nMEASURED AUDIO TIMELINE (signal analysis of the actual soundtrack — "
            "treat these as facts):\n" + audio_timeline.describe(spans) +
            "\nWrite the Audio field from these facts. Report the music/ambience/"
            "laughter spans with their measured times; do not invent audio outside "
            "them and do not drop measured music for lacking a visible source."
        )
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
    prompt = prompt.replace("[inaudible]", _unclear_token())
    content = [text_content(prompt), *_labelled(frames, shot_bounds)]
    data = _complete_json(backend, content, max_tokens=2000)
    return Globals(
        characters=[Entity(e["id"], _entity_depronoun(e["description"], e["id"]))
                    for e in data.get("characters", [])],
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


# he/she pronouns for a single known entity; excludes it/its, which usually
# refer to a garment or object inside a description, not the person.
_PERSON_PRON = re.compile(
    r"\b(he|she|they|him|them|his|her|hers|their|theirs)\b", re.IGNORECASE
)
_REQUIRED_LOWER = re.compile(
    r"Lower body and shoes (?:are )?not visible\.?", re.IGNORECASE
)


def _entity_depronoun(text: str, label: str) -> str:
    """Overview descriptions reference exactly one entity, so pronoun
    substitution is unambiguous (verified regression: 'He wears', 'her hair'
    all over the cast list). 'her' is possessive when a word follows."""
    def sub(m: re.Match[str]) -> str:
        w = m.group(0).lower()
        if w in ("his", "hers", "their", "theirs"):
            return f"{label}'s"
        if w == "her":
            rest = text[m.end():m.end() + 2]
            return f"{label}'s" if re.match(r"\s\w", rest) else label
        return label
    out = _PERSON_PRON.sub(sub, text)
    return _REQUIRED_LOWER.sub("Lower body and shoes are not visible.", out)


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


def _unclear_token() -> str:
    """Playbook 'Current syntax conflicts': the live tool decides the unclear-
    speech token ([inaudible] vs <unintelligible>). Configurable, never mixed."""
    return os.environ.get("AUTOSCRIBE_UNCLEAR_TOKEN", "[inaudible]")


def _shot_speech(
    transcript: tr.Transcript, start: float, end: float,
) -> tuple[list[tuple[float, float, str, bool]], list[tuple[float, float]]]:
    """Speech genuinely inside [start, end], clipped at WORD level.

    Verified failure this fixes: whisper SEGMENT bounds drift (a segment claimed
    0.0s on a clip whose speech starts at 5.5s), and injecting any overlapping
    segment whole made every shot repeat the same full sentence. Words are
    grouped into runs split at >0.5s pauses; only words whose midpoint lies in
    the shot are used. Returns (verbatim_runs, unreliable_ranges); each verbatim
    run carries a continues-past-this-shot flag (playbook Rule 4 cut lines)."""
    def _seg_reliable(w: tr.Word) -> bool:
        mid = (w.start + w.end) / 2
        for s in transcript.segments:
            if s.start - 0.2 <= mid <= s.end + 0.2:
                return s.reliable
        return True

    good: list[tuple[float, float, str, bool]] = []
    bad: list[tuple[float, float]] = []
    if transcript.words:
        words = sorted(transcript.words, key=lambda w: w.start)

        def chains(i: int, j: int) -> bool:  # consecutive words joined by <=0.5s gaps
            return all(words[k + 1].start - words[k].end <= 0.5 for k in range(i, j))

        selected: list[tr.Word] = []
        for i, w in enumerate(words):
            mid = (w.start + w.end) / 2
            if start <= mid < end:
                # Onset-drift fix at cuts (verified: 'Come' stamped 0.4s before
                # the 5.5s cut it belongs after): a trailing word within 0.5s of
                # the shot end whose sentence continues into the next shot is
                # attributed to the next shot, not this one.
                if (mid >= end - 0.5 and i + 1 < len(words)
                        and chains(i, i + 1)
                        and (words[i + 1].start + words[i + 1].end) / 2 >= end):
                    continue
                selected.append(w)
            elif mid < start and w.end > start - 0.5 and i + 1 < len(words):
                # Mirror rule: a word just before this shot's start that chains
                # into words inside it belongs here (clamped to the boundary).
                nxt = (words[i + 1].start + words[i + 1].end) / 2
                if chains(i, i + 1) and start <= nxt < end:
                    selected.append(w)

        run: list[tr.Word] | None = None
        run_rel = True

        def flush() -> None:
            if run:
                s = round(max(run[0].start, start), 1)  # clamp drift to the shot
                e = round(min(max(run[-1].end, s + 0.1), end), 1)
                if run_rel:
                    # Does this sentence continue past the shot? True when the
                    # next word chains on (<=0.5s gap) and lands at or beyond
                    # the boundary — including words the trailing rule already
                    # reassigned to the next shot (their midpoint sits within
                    # 0.5s of the boundary).
                    last = words.index(run[-1])
                    cont = (last + 1 < len(words)
                            and words[last + 1].start - run[-1].end <= 0.5
                            and (words[last + 1].start + words[last + 1].end) / 2
                            >= end - 0.5)
                    good.append((s, e, " ".join(w.text for w in run), cont))
                else:
                    bad.append((s, e))

        for w in selected:
            rel = _seg_reliable(w)
            if run and rel == run_rel and w.start - run[-1].end <= 0.5:
                run.append(w)
            else:
                flush()
                run, run_rel = [w], rel
        flush()
        return good, bad
    # No word timestamps: only segments FULLY inside the shot keep their words;
    # partial overlaps become unreliable ranges (we cannot know which words fall
    # inside, and repeating the whole sentence across shots is the known bug).
    for s in transcript.segments:
        if s.end <= start or s.start >= end:
            continue
        if s.reliable and s.start >= start - 0.2 and s.end <= end + 0.2:
            good.append((round(max(s.start, start), 1), round(min(s.end, end), 1), s.text, False))
        else:
            bad.append((round(max(s.start, start), 1), round(min(s.end, end), 1)))
    return good, bad


_FILLER_RX = re.compile(r"^no\s+(?:speech|dialogue|music|audio|sound)\b", re.IGNORECASE)


def _polish(shot: Shot) -> Shot:
    """Playbook Rule 8 enforced deterministically: terminal punctuation on every
    line, no 'No speech/dialogue/music' filler lines, no exact-duplicate lines,
    every range clamped inside the shot, and no trailing 'locked static' claim
    once timed camera movement exists."""
    def tidy(items: list[Timed]) -> list[Timed]:
        out: list[Timed] = []
        seen: set[tuple[float, float, str]] = set()
        for t in items:
            text = " ".join(t.text.split())
            if not text or _FILLER_RX.match(text):
                continue
            if text[-1] not in '.!?"’”':
                text += "."
            s = round(max(shot.start, min(t.start, shot.end)), 1)
            e = round(max(s, min(t.end, shot.end)), 1)
            key = (s, e, text.lower())
            if key in seen:
                continue
            seen.add(key)
            out.append(Timed(s, e, text))
        # The live validator blocks two entries sharing an identical full
        # start/end pair even with different text — merge them into one
        # tightly-linked line rather than faking a 0.1s offset.
        by_window: dict[tuple[float, float], int] = {}
        merged: list[Timed] = []
        for t in out:
            w = (t.start, t.end)
            if w in by_window:
                prev = merged[by_window[w]]
                nxt = t.text
                if not re.match(r'[CO]\d|"|On-screen', nxt):  # keep IDs/quotes cased
                    nxt = nxt[0].lower() + nxt[1:]
                merged[by_window[w]] = Timed(prev.start, prev.end,
                                             prev.text.rstrip(".") + "; " + nxt)
            else:
                by_window[w] = len(merged)
                merged.append(t)
        return merged

    shot.actions = tidy(shot.actions)
    shot.camera_movements = tidy(shot.camera_movements)
    if shot.camera_movements:
        # Strip only a standalone trailing static claim ('..., locked static.');
        # 'settles to locked static' after a real move is a truthful phrase.
        shot.camera = re.sub(r",\s*locked static\s*\.?\s*$", ".", shot.camera).strip()
    for attr in ("camera", "scene"):
        val = " ".join(getattr(shot, attr, "").split())
        if val and val[-1] not in '.!?"’”':
            val += "."
        setattr(shot, attr, val)
    return shot


def _is_ladder(items: list[Timed], run: int = 5) -> bool:
    """Detect templated sliding-window timestamps (verified failure: 0.4-0.7,
    0.5-0.9, 0.6-1.0, 0.7-1.1 ... for dozens of lines): `run` consecutive
    entries whose starts advance by a near-EXACT constant stride while durations
    stay similar. Real event boundaries never form such ladders — their starts
    land wherever the footage changes."""
    ts = sorted((t.start, t.end) for t in items)
    for i in range(len(ts) - run + 1):
        win = ts[i:i + run]
        strides = [win[j + 1][0] - win[j][0] for j in range(run - 1)]
        durs = [e - s for s, e in win]
        if (min(strides) > 0.0 and max(strides) - min(strides) <= 0.021
                and max(durs) - min(durs) <= 0.2):
            return True
    return False


def _shot_pass(
    backend: OpenAIVisionBackend, grid: list[frames_mod.GridFrame],
    cast: str, index: int, cut: str, start: float, end: float,
    transcript: tr.Transcript, spans: list[audio_timeline.AudioSpan],
) -> Shot:
    frames = _sample(grid, start, end, step=0.25)
    prompt = _SHOT_PROMPT.format(cast=cast, start=start, end=end)
    facts = _audio_facts(spans, start, end)
    if facts:
        prompt += (
            "\n\nMEASURED AUDIO FACTS for this shot (signal analysis of the real "
            "soundtrack — treat as ground truth):\n" + facts +
            "\nAdd audio lines ONLY for these facts (music/ambience/laughter with the "
            "measured ranges). If no speech span is listed here, there is NO narration "
            "or dialogue in this shot — write none."
        )
    good, bad = _shot_speech(transcript, start, end)
    # Verified failure: an unreliable-ASR sliver inside the measured MUSIC span
    # became a fabricated '(5.1s-5.5s) C4 says "[inaudible]"' line. If the
    # signal says music, unintelligible-vocal claims from the ASR are noise —
    # the music line already covers that audio.
    music = audio_timeline.music_spans(spans)
    def _mostly_music(s: float, e: float) -> bool:
        overlap = sum(max(0.0, min(e, me) - max(s, ms)) for ms, me in music)
        return e > s and overlap / (e - s) >= 0.6
    bad = [(s, e) for s, e in bad if not _mostly_music(s, e)]
    # (Reliable words over music are kept — that is real singing.)
    token = _unclear_token()
    if good:
        lang = _lang_label(transcript)
        lines = "\n".join(
            f'  ({s:.1f}s-{e:.1f}s) "{txt}"'
            + ("  <- sentence continues in the NEXT shot" if cont else "")
            for s, e, txt, cont in good
        )
        prompt += (
            f"\n\nVERBATIM WORDS AUDIBLE IN THIS SHOT ONLY (word-level timestamps, "
            f"language = {lang}):\n{lines}\n"
            f"Add each as an Action & Audio line with those exact timestamps and EXACT "
            f"words, no translation: '(start-end) C# says in {lang}: \\\"exact "
            f"words\\\".' (use 'sings' for singing). A sentence that continues across a "
            f"cut is SPLIT at the cut: this shot contains ONLY the words above — never "
            f"complete the sentence, never repeat words that belong to another shot, "
            f"and when marked as continuing, do NOT add editorial notes like 'the line "
            f"ends mid-sentence' — the split itself is the record. Attribution follows "
            f"HARD RULE 6: only a character visibly performing vocals, else the "
            f"off-screen voice C-ID — never a dancer."
        )
    if bad:
        lang = _lang_label(transcript)
        ranges = "; ".join(f"({s:.1f}s-{e:.1f}s)" for s, e in bad)
        prompt += (
            f"\n\nUNRELIABLE VOCALS at {ranges}: vocals are audible in these ranges but "
            f"the words are NOT intelligible (transcription failed its confidence "
            f"check). Add an Action & Audio line per range in the form '(start-end) C# "
            f"sings in {lang}; the words are {token}.' — attribute per HARD RULE 6 "
            f"and NEVER invent or guess words for these ranges."
        )
    prompt = prompt.replace("[inaudible]", token)  # live-tool token consistency
    content = [text_content(prompt), *_labelled(frames)]
    data = _complete_json(backend, content, max_tokens=4000)
    actions = sorted(_clean(_timed_items(data.get("actions", []))), key=lambda t: t.start)
    if _is_ladder(actions):
        # Templated windows detected — one corrective retry with the defect named.
        retry = content[:1] + [text_content(
            "\n\nQC REJECTION: a previous attempt produced sliding template windows "
            "(constant stride/duration ladders like 0.4-0.7, 0.5-0.9, 0.6-1.0). That "
            "violates rule 3b. Re-time every line to the true first/last frame of its "
            "event, merge unchanged states into single spanning lines, and never use a "
            "fixed stride.")] + content[1:]
        data2 = _complete_json(backend, retry, max_tokens=4000)
        actions2 = sorted(_clean(_timed_items(data2.get("actions", []))), key=lambda t: t.start)
        if actions2 and not _is_ladder(actions2):
            data, actions = data2, actions2
    return _polish(Shot(
        index=index, cut=cut, start=start, end=end,
        shot_type=data.get("shot_type", "shot"), camera=data.get("camera", ""),
        camera_movements=_clean(_timed_items(data.get("camera_movements", []))),
        scene=data.get("scene", ""),
        actions=actions,
        playback_speed=_norm_speed(data.get("playback_speed", "regular")),
    ))


_QUOTE_RX = re.compile(r'"([^"]{8,})"')
_SPEECH_VERB_RX = re.compile(r"\b(says|sings|shouts|whispers|asks|replies)\b", re.IGNORECASE)


def _dedup_dialogue(shots: list[Shot]) -> list[Shot]:
    """Output sanity check (verified failure: the same narration sentence was
    emitted in four different shots): an identical spoken/sung quote may appear
    ONCE across the whole caption — word-level clipping already assigns each
    word to exactly one shot, so any later identical quote is a duplicate."""
    seen: set[str] = set()
    for shot in shots:
        keep: list[Timed] = []
        for a in shot.actions:
            m = _QUOTE_RX.search(a.text)
            if m and _SPEECH_VERB_RX.search(a.text):
                key = " ".join(m.group(1).lower().split())
                if key in seen:
                    continue
                seen.add(key)
            keep.append(a)
        shot.actions = keep
    return shots


_TRANSITION_RX = re.compile(
    r"\biris\b|circular (?:window|mask|wipe)|expanding (?:circle|circular|mask)"
    r"|\bwipes?\b|dissolv|fade[sd]? (?:in|out|to|from)",
    re.IGNORECASE,
)


def _contradiction_qc(
    backend: OpenAIVisionBackend, grid: list[frames_mod.GridFrame], cast: str,
    shots: list[Shot], transcript: tr.Transcript,
    spans: list[audio_timeline.AudioSpan],
    progress: Callable[[str, float], None],
) -> list[Shot]:
    """Catch self-contradictions: a shot whose OWN text describes a transition
    (iris, wipe, dissolve, fade) mid-shot that the boundary pass missed.

    Verified failure this fixes: the model described a 'white-edged circular
    window expanding' at 3.1s inside a single shot — a genuine Iris boundary —
    while the shot list said no cut existed there. Each suspect time is
    re-verified on the straddling frames; confirmed boundaries split the shot
    and both halves are re-annotated."""
    out: list[Shot] = []
    for shot in shots:
        suspects: list[float] = []
        for item in [*shot.camera_movements, *shot.actions]:
            if _TRANSITION_RX.search(item.text):
                t = round(item.start, 1)
                if (shot.start + 0.4 < t < shot.end - 0.4
                        and all(abs(t - u) > 0.5 for u in suspects)):
                    suspects.append(t)
        split_at: tuple[float, str] | None = None
        for t in suspects:
            is_cut, cut_type = cuts_mod._verify(
                backend,
                cuts_mod._near(grid, t, before=True),
                cuts_mod._near(grid, t, before=False),
            )
            if is_cut:
                split_at = (t, cut_type)
                break
        if split_at is None:
            out.append(shot)
            continue
        t, cut_type = split_at
        progress(f"qc: splitting shot {shot.index} at {t:.1f}s ({cut_type})", 0.94)
        out.append(_shot_pass(backend, grid, cast, shot.index, shot.cut,
                              shot.start, t, transcript, spans))
        out.append(_shot_pass(backend, grid, cast, shot.index, cut_type,
                              t, shot.end, transcript, spans))
    for i, shot in enumerate(out):
        shot.index = i + 1
    return out


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
    spans: list[audio_timeline.AudioSpan] = []
    try:
        wav = asr.extract_audio(video, work / "audio.wav")
        transcript = tr.transcribe(wav)
        spans = audio_timeline.analyze(wav, transcript)
    except Exception:  # audio is optional; never block the visual pass
        transcript = tr.Transcript(language="", text="", segments=[])

    # Shots FIRST: the cast pass needs the shot structure to keep identities
    # straight across cuts (different shots may show unrelated people).
    progress("shots", 0.3)
    resolved = cuts_mod.resolve_shots(backend, video, grid, duration)
    progress("cast", 0.4)
    g = _cast_pass(backend, grid, duration, transcript, resolved, spans)
    cast_text = _cast_text(g)
    shots: list[Shot] = []
    for i, (s, e, cut) in enumerate(resolved):
        progress(f"shot {i + 1}/{len(resolved)}", 0.45 + 0.45 * i / len(resolved))
        shots.append(_shot_pass(backend, grid, cast_text, i + 1, cut, s, e, transcript, spans))
    progress("qc: contradictions", 0.92)
    shots = _contradiction_qc(backend, grid, cast_text, shots, transcript, spans, progress)
    shots = _dedup_dialogue(shots)
    progress("done", 1.0)
    return Annotation(video_name=video.name, duration=duration, globals=g, shots=shots)
