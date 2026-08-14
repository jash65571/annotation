"""Reviewer pass: audit a seeded (attempter) caption against AutoScribe's own
fresh annotation of the same video, then produce a reviewed DRAFT caption, a
1-5 score for the original attempt, and attempter feedback.

Three things this pass is NOT allowed to pretend:

*It is not comparing evidence to text.* The reviewer model sees two captions and
some measured facts — not the video. It can tell which caption is internally
consistent with the measured timeline; it cannot see the footage. So the source
hierarchy is stated honestly: the MEDIA is truth, and both captions plus every
automated measurement are leads. Where the two captions disagree on something no
measurement covers, the disagreement is reported as unresolved rather than
silently decided in the fresh caption's favour.

*Its output is not automatically valid.* The rewritten caption is put back
through the full deterministic validator. Previously a reviewer model could
introduce a broken timestamp, an unbalanced quote or a ghost C-ID and the web app
would write it straight to disk as the final file.

*A failure is not a pass.* If the model returns nothing usable, that is recorded
as a blocking failure instead of quietly returning the fresh caption as if it
had been reviewed.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .blockers import WARNING, BlockerLog
from .validate import validate_caption
from .vision import OpenAIVisionBackend, image_content, text_content

_REVIEW_PROMPT = (
    "You are a Manuscript-II REVIEWER. You are given:\n"
    "1. FRESH CAPTION — a machine annotation of the same video. Its shot boundaries "
    "and timestamps were measured from the media, so they are the best available "
    "evidence for TIMING — but it is a machine draft and can still be wrong about "
    "what it describes.\n"
    "2. SEED CAPTION — the original attempter's submission, which must be audited. "
    "Automated evaluations passing it prove nothing; there is always something to fix.\n"
    "{evidence_clause}"
    "{feedback_clause}"
    "\nSOURCE HIERARCHY (this is the rule that decides every conflict):\n"
    "- The actual MEDIA is the only truth. NEITHER caption is truth.\n"
    "- FRAMES from the clip, when supplied below, are the media. They outrank both "
    "captions on anything visible in them. They are SAMPLED, so they can prove an "
    "event happened but cannot prove one did not.\n"
    "- MEASURED FACTS (shot boundaries, the audio timeline, word-level transcript "
    "timings) are direct observations of the media and outrank both captions.\n"
    "- For anything NOT covered by a measured fact — what an object is, what a "
    "gesture means, who a person is — the two captions are competing claims and "
    "NEITHER automatically wins. Prefer the more specific claim only when it is "
    "consistent with the measured facts. When they genuinely conflict and no "
    "measurement settles it, keep the more conservative wording and list the "
    "conflict in 'unresolved'. Do NOT invent a tiebreak.\n"
    "\nREVIEWER RULES:\n"
    "- C-IDs ARE LABELS, NOT IDENTITIES: the seed and the fresh caption may number "
    "the same people differently. Before comparing them, match characters by "
    "appearance, clothing, and actions — renumbering is NEVER an issue and must "
    "never appear in 'issues' or lower the score.\n"
    "- Audit the ENTIRE seed: cast identity, objects, scene, style, audio, concerns, "
    "shot count/boundaries, cuts, camera moves, playback speed, every action line, "
    "every dialogue line, every timestamp.\n"
    "- Decide per area: KEEP (already meets standard), FIX/ENRICH (usable, needs "
    "corrections), or REDO/REBUILD (foundation wrong). Never protect a bad seed.\n"
    "- FINAL CAPTION: start from the FRESH caption's structure and timestamps. "
    "NEVER import a timestamp, shot boundary, cut label, or playback speed from the "
    "seed. You MAY merge extra descriptive detail from the seed (names of garments, "
    "colors, verbatim on-screen text, dialogue wording) when it is consistent "
    "with the measured facts. Keep the exact section format of the fresh caption "
    "([Overview], Cast:, [Shot N: ...], Cut:, Camera:, Camera Movements:, Scene:, "
    "Action & Audio:, Playback Speed:, Speed Changes:). No pronouns outside quotes; "
    "C#/O# labels; one event per action line; never mention the seed, reviewer, or "
    "review process inside the caption.\n"
    "- NEVER state race, ethnicity, nationality or apparent age of any person, and "
    "remove any such claim inherited from the seed — those are not observable from "
    "footage.\n"
    "- NEVER state tone or delivery of speech unless the measured facts support it; "
    "you cannot hear this video.\n"
    "- QUALITY GATE (audit-flag playbook — check every one):\n"
    "  * Split any line whose events start or end at different times (watch for "
    "'then', 'followed by', 'while', 'as', two quotes in one line, action+sound "
    "with different timing, two people acting independently).\n"
    "  * Two genuinely SIMULTANEOUS events keep their identical range on SEPARATE "
    "lines — never merge them into one line and never fake an offset.\n"
    "  * No timing-filler phrases ('near the end', 'around 5 seconds', 'partway "
    "through') — timestamps carry the timing.\n"
    "  * No filler lines ('No speech.', 'No music.'); genuine silence belongs in "
    "Audio concerns.\n"
    "  * Speech lines need C-ID + off-screen status when unseen + exact words with "
    "fillers/stutters/false starts preserved; timestamp covers the audible words "
    "only; pauses > 0.5s split the line; overlapping speakers keep separate "
    "overlapping ranges; a sentence severed by a cut is tagged [mid-sentence cut].\n"
    "  * Camera setup in Camera; every timed pan/tilt/zoom/push/track split by "
    "phase with direction in Camera Movements; never 'locked static' when any "
    "drift is visible; graphic/collage panel motion is an edit, NOT camera motion.\n"
    "  * A stable whole-shot state belongs in Scene, not repeated action lines.\n"
    "  * Audio completeness: dialogue alone is incomplete — music, ambience, SFX, "
    "and reactions (laughter, gasps, cheers) need truthful ranges; non-diegetic "
    "music needs no visible source; music under dialogue is BOTH and neither one "
    "removes the other.\n"
    "  * Every relevant voice gets a C-ID (narrator, singer, crowd as one grouped "
    "ID, filmer, partial figure); no ghost C/O-IDs; one O-ID never combines "
    "distinct object types; never write 'cannot be determined' — omit instead.\n"
    "  * Direction: screen-left/right for viewer position, character-left/right "
    "only when the body side is clear.\n"
    "  * Formatting: balanced quotes, terminal punctuation everywhere, 'None.' for "
    "empty concerns, no stray separators, no ghost references.\n"
    "  * Transitions: verify each boundary's defining evidence (Wipe = incoming "
    "pushes across a line; Iris = circular opening/closing; jump cut = time "
    "removed, same framing); a short shot is still a shot. L-cut and J-cut are "
    "defined by SOUND crossing a picture boundary — never assign them from "
    "description alone.\n"
    "- SCORE the ORIGINAL attempt (not the final caption): 5 = no issues; 4 = one "
    "minor issue; 3 = several minor issues; 2 = one major issue or many minor "
    "(meaningful rework); 1 = multiple major issues / rebuild. Major issues: missing "
    "speech, wrong speaker, missed camera move, missed slow motion, broadly wrong "
    "timing.\n"
    "- FEEDBACK for the attempter: clear, factual, material issues only — what was "
    "wrong and what to check next time. Bullet lines.\n"
    "- UNRESOLVED: list every question you could NOT settle from the measured facts. "
    "An empty list means you are certain, so do not pad it — but never resolve a "
    "genuine conflict by guessing just to keep it empty.\n"
    "{feedback_rules}"
    "\nReturn STRICT JSON:\n"
    '{{"verdict": "KEEP" | "FIX / ENRICH" | "REDO / REBUILD",\n'
    '  "score": 1-5,\n'
    '  "score_reason": "one sentence",\n'
    '  "issues": ["each material issue found in the seed, with timestamps"],\n'
    '  "unresolved": ["each conflict the measured facts could not settle"],\n'
    '  "feedback": "attempter feedback, bullet lines separated by \\n",\n'
    '  "final_caption": "the complete reviewed caption text"}}\n'
    "\n=== FRESH CAPTION (machine draft) ===\n{fresh}\n"
    "\n=== SEED CAPTION (attempter submission under review) ===\n{seed}\n"
    "{evidence_block}"
    "{feedback_block}"
)

_EVIDENCE_CLAUSE = (
    "3. MEASURED FACTS — direct signal measurements of the actual media (shot "
    "boundaries with real frame timestamps, and the audio timeline). These outrank "
    "both captions.\n"
)
_FEEDBACK_CLAUSE = (
    "4. EVALUATOR / FINAL-REVIEW FEEDBACK — suggestions attached to the task. They "
    "describe the ORIGINAL SEED, not the fresh caption. Judge each against the "
    "measured facts; never invent content to satisfy one.\n"
)
_FEEDBACK_RULES = (
    "- FEEDBACK VERDICTS: each evaluator item concerns the SEED. In 'issues', mark it "
    "'confirmed in seed — fixed in final' when the seed truly had the problem and the "
    "final caption corrects it, or 'not supported by the media' when the evidence "
    "disproves it. NEVER mark an item rejected merely because the rebuilt final "
    "caption no longer contains the problem — that is the fix working, not the "
    "feedback being wrong.\n"
)


def review(
    fresh_caption: str,
    seed_caption: str,
    evaluator_feedback: str = "",
    evidence: str = "",
    blockers: BlockerLog | None = None,
    frames: list[tuple[float, Path]] | None = None,
    detected_language: str = "",
) -> dict[str, Any]:
    """Run the reviewer pass over a caption and return the reviewed draft.

    ``frames`` are ``(timestamp_seconds, png_path)`` stills spanning the clip.
    They are what let the reviewer weigh a disputed visual claim against the
    PICTURE rather than against whichever caption sounds more confident, and
    the timestamps are what let it check the claim at the right *moment*.
    Without them this pass is only a prose comparator, so their absence is
    recorded as a blocker.

    The result is ALWAYS a draft: ``ready`` is never True, and ``blockers``
    carries everything that must be resolved by a human first.
    """
    log = blockers if blockers is not None else BlockerLog()
    backend = OpenAIVisionBackend()
    fb = evaluator_feedback.strip()
    ev = evidence.strip()
    prompt = _REVIEW_PROMPT.format(
        evidence_clause=_EVIDENCE_CLAUSE if ev else "",
        feedback_clause=_FEEDBACK_CLAUSE if fb else "",
        feedback_rules=_FEEDBACK_RULES if fb else "",
        fresh=fresh_caption.strip(),
        seed=seed_caption.strip(),
        evidence_block=f"\n=== MEASURED FACTS ===\n{ev}\n" if ev else "",
        feedback_block=f"\n=== EVALUATOR FEEDBACK ===\n{fb}\n" if fb else "",
    )
    content: list[dict[str, object]] = [text_content(prompt)]
    if frames:
        content.append(text_content(
            "\n=== FRAMES FROM THE ACTUAL VIDEO ===\nThese are stills from the clip "
            "under review, each labelled with the TIMESTAMP it was taken at. They "
            "outrank BOTH captions on anything visible. Use the labels to check a "
            "disputed claim against the right moment — a claim at 4.2s is tested "
            "against the frames nearest 4.2s, not against the overall impression. "
            "They are sampled, so absence from these stills is not proof an event "
            "did not happen."
        ))
        # An unlabelled image can settle "is there a red jacket?" but not
        # "does C1 raise a hand at 4.2s" — which is most of what a caption
        # review actually disputes.
        for time_s, frame_path in frames:
            content.append(text_content(f"t={time_s:.2f}s:"))
            content.append(image_content(frame_path))
    else:
        log.add(
            "REVIEW_WITHOUT_PICTURE",
            "The reviewer pass received no frames, so visual disagreements between "
            "the seed and the fresh caption were settled from text alone.",
            severity=WARNING,
        )

    data: dict[str, Any] = {}
    last_error = "no response"
    for _ in range(3):
        try:
            raw = backend.complete(content, json_mode=True, max_tokens=8000)
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            continue
        if not raw:
            last_error = "empty response"
            continue
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as exc:
            last_error = f"invalid JSON: {exc}"
            continue
        if isinstance(parsed, dict):
            data = parsed
            break

    final = str(data.get("final_caption") or "").strip()
    if not final:
        # A failed review is NOT a passed review. The fresh caption is returned
        # so the work is not lost, but it is explicitly marked un-reviewed.
        log.add(
            "REVIEW_FAILED",
            f"The reviewer pass produced no caption ({last_error}). The text below is "
            f"the UNREVIEWED machine draft — no audit of the seed was performed.",
        )
        final = fresh_caption

    # The reviewer's rewrite is re-validated. Nothing reaches a file unchecked.
    validate_caption(final, log, detected_language=detected_language)

    score = data.get("score")
    unresolved = [str(u) for u in data.get("unresolved", []) if str(u).strip()]
    for item in unresolved:
        log.add("REVIEW_UNRESOLVED", item)

    ready, reason = log.readiness()
    return {
        "verdict": str(data.get("verdict") or "FIX / ENRICH"),
        "score": int(score) if isinstance(score, (int, float)) and 1 <= score <= 5 else 0,
        "score_reason": str(data.get("score_reason") or ""),
        "issues": [str(i) for i in data.get("issues", []) if str(i).strip()],
        "unresolved": unresolved,
        "feedback": str(data.get("feedback") or ""),
        "final_caption": final,
        "ready": ready,
        "readiness_reason": reason,
        "blockers": log.as_dicts(),
    }
