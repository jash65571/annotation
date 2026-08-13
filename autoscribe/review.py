"""Reviewer pass: audit a seeded (attempter) caption against AutoScribe's own
fresh annotation of the same video, then produce the final RTD caption, a
1-5 score for the original attempt, and attempter feedback.

The fresh machine annotation is the evidence baseline: its shot boundaries and
timestamps come from actual media analysis and are never overwritten by the
seed ("AI never owns the clock" — the seed's clock is even less trustworthy).
The seed may only contribute extra *descriptive* detail that is consistent
with the fresh annotation.
"""

from __future__ import annotations

import json
from typing import Any

from .vision import OpenAIVisionBackend, text_content

_REVIEW_PROMPT = (
    "You are a Manuscript-II REVIEWER. You are given:\n"
    "1. FRESH CAPTION — a machine annotation built directly from the actual video "
    "frames and audio. Its shot boundaries and timestamps are measured from the "
    "media and are the factual baseline.\n"
    "2. SEED CAPTION — the original attempter's submission, which must be audited. "
    "Automated evaluations passing it prove nothing; there is always something to fix.\n"
    "{feedback_clause}"
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
    "colors, verbatim on-screen text, dialogue wording) ONLY when it is consistent "
    "with the fresh caption; when they conflict, the fresh caption wins. Keep the "
    "exact section format of the fresh caption ([Overview], Characters:, [Shot N: "
    "...], Cut:, Camera:, Scene:, Action & Audio:, Video playback speed:). No "
    "pronouns outside quotes; C#/O# labels; one sentence per action line; no "
    "duplicate identical timestamp ranges; never mention the seed, reviewer, or "
    "review process inside the caption.\n"
    "- RTD QUALITY GATE (audit-flag playbook — check every one):\n"
    "  * Split any line whose events start or end at different times (watch for "
    "'then', 'followed by', 'while', 'as', two quotes in one line, action+sound "
    "with different timing, two people acting independently).\n"
    "  * No timing-filler phrases ('near the end', 'around 5 seconds', 'partway "
    "through') — timestamps carry the timing.\n"
    "  * No filler lines ('No speech.', 'No music.'); genuine silence belongs in "
    "Audio concerns.\n"
    "  * Speech lines need C-ID + off-screen status when unseen + supported tone + "
    "exact words with fillers/stutters/false starts preserved; timestamp covers the "
    "audible words only; pauses > 0.5s split the line; overlapping speakers keep "
    "separate overlapping ranges.\n"
    "  * Camera setup in Camera; every timed pan/tilt/zoom/push/track split by "
    "phase with direction in Camera Movements; never 'locked static' when any "
    "drift is visible; graphic/collage panel motion is an edit, NOT camera motion.\n"
    "  * A stable whole-shot state belongs in Scene, not repeated action lines.\n"
    "  * Audio completeness: dialogue alone is incomplete — music, ambience, SFX, "
    "and reactions (laughter, gasps, cheers) need truthful ranges; non-diegetic "
    "music needs no visible source.\n"
    "  * Every relevant voice gets a C-ID (narrator, singer, crowd as one grouped "
    "ID, filmer, partial figure); no ghost C/O-IDs; one O-ID never combines "
    "distinct object types; never write 'cannot be determined' — omit instead.\n"
    "  * Direction: screen-left/right for viewer position, character-left/right "
    "only when the body side is clear.\n"
    "  * Formatting: balanced quotes, terminal punctuation everywhere, 'None.' for "
    "empty concerns, no stray separators, no ghost references.\n"
    "  * Transitions: verify each boundary's defining evidence (Wipe = incoming "
    "pushes across a line; Iris = circular opening/closing; jump cut = time "
    "removed, same framing); a short shot is still a shot.\n"
    "- SCORE the ORIGINAL attempt (not the final caption): 5 = no issues; 4 = one "
    "minor issue; 3 = several minor issues; 2 = one major issue or many minor "
    "(meaningful rework); 1 = multiple major issues / rebuild. Major issues: missing "
    "speech, wrong speaker, missed camera move, missed slow motion, broadly wrong "
    "timing.\n"
    "- FEEDBACK for the attempter: clear, factual, material issues only — what was "
    "wrong and what to check next time. Bullet lines.\n"
    "{feedback_rules}"
    "\nReturn STRICT JSON:\n"
    '{{"verdict": "KEEP" | "FIX / ENRICH" | "REDO / REBUILD",\n'
    '  "score": 1-5,\n'
    '  "score_reason": "one sentence",\n'
    '  "issues": ["each material issue found in the seed, with timestamps"],\n'
    '  "feedback": "attempter feedback, bullet lines separated by \\n",\n'
    '  "final_caption": "the complete final RTD caption text"}}\n'
    "\n=== FRESH CAPTION (evidence baseline) ===\n{fresh}\n"
    "\n=== SEED CAPTION (attempter submission under review) ===\n{seed}\n"
    "{feedback_block}"
)

_FEEDBACK_CLAUSE = (
    "3. EVALUATOR / FINAL-REVIEW FEEDBACK — suggestions attached to the task. They "
    "describe the ORIGINAL SEED, not the fresh caption. Judge each against the media "
    "evidence; never invent content to satisfy one.\n"
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
) -> dict[str, Any]:
    """Run the reviewer pass. Returns verdict/score/issues/feedback/final_caption;
    falls back to the fresh caption if the model returns no usable final text."""
    backend = OpenAIVisionBackend()
    fb = evaluator_feedback.strip()
    prompt = _REVIEW_PROMPT.format(
        feedback_clause=_FEEDBACK_CLAUSE if fb else "",
        feedback_rules=_FEEDBACK_RULES if fb else "",
        fresh=fresh_caption.strip(),
        seed=seed_caption.strip(),
        feedback_block=f"\n=== EVALUATOR FEEDBACK ===\n{fb}\n" if fb else "",
    )
    data: dict[str, Any] = {}
    for _ in range(3):
        raw = backend.complete([text_content(prompt)], json_mode=True, max_tokens=8000)
        if not raw:
            continue
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            data = parsed
            break
    final = str(data.get("final_caption") or "").strip()
    score = data.get("score")
    return {
        "verdict": str(data.get("verdict") or "FIX / ENRICH"),
        "score": int(score) if isinstance(score, (int, float)) and 1 <= score <= 5 else 0,
        "score_reason": str(data.get("score_reason") or ""),
        "issues": [str(i) for i in data.get("issues", []) if str(i).strip()],
        "feedback": str(data.get("feedback") or ""),
        "final_caption": final if final else fresh_caption,
    }
