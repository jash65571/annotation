"""Task-seed parser for the Manuscript II audio reviewer.

Turns the pasted live task / seed into the tool's task_context.json so the
locked structure (C# cast, O# objects, shot boundaries) drives audio review
instead of a hand-authored stub. The live locked task always wins over machine
shot detection.

Output is backward compatible with the existing task_context.json:

    {
      "characters": ["C1", "C2"],          # ids only (existing consumers)
      "objects":    ["O1"],                # ids only
      "shots":      [{"shot": 1, "start": 0.0, "end": 4.0}],
      "video_sha256": "<preserved if already present>",
      "seed_meta": {                       # richer baseline for new checks
        "characters": [{"id": "C1", "description": "...", "original": true}],
        "objects":    [{"id": "O1", "description": "...", "original": true}],
        "original_character_ids": ["C1", "C2"],
        "original_object_ids":    ["O1"],
        "shot_count_declared": 2,
        "parse_issues": [...]
      }
    }

The seed_meta baseline records what the ORIGINAL task contained. Detecting
later added / deleted cast (spec 34, 35) is a comparison against this baseline
and lives in the pasted-back QA tool, which sees the edited state. The parser's
job is only to establish the immutable baseline.

Accepted entity forms (case-insensitive, optional leading @):
    C1: a man in a red shirt
    O2 - a wooden chair
    @C3) off-screen narrator

Accepted shot forms:
    Shot 1: 0.0 - 4.0
    Shot 2  4.00–10.50
    (a "Shots: 3" style count line is recorded but not required)

Usage:
    python manuscript_audio_seed.py path/to/seed.txt
    python manuscript_audio_seed.py < seed.txt      # via stdin
"""

from pathlib import Path
import json
import re
import sys


ROOT = Path(__file__).resolve().parent
CONTEXT = ROOT / "task_context.json"

# "C1:", "O2 -", "@C3)" ... an inline definition line binds an id to a
# description on the same line.
_DEF_RE = re.compile(
    r"^\s*@?(?P<id>[CO]\d+)\s*[:\-)–—]\s*(?P<desc>.*\S)?",
    re.IGNORECASE,
)

# A bare id on its own line: "C1", "@O2". The live UI dump puts the description
# on the following line, so this is handled separately from _DEF_RE.
_ID_ONLY_RE = re.compile(r"^\s*@?(?P<id>[CO]\d+)\s*$", re.IGNORECASE)

# Inline shot with plain seconds: "Shot 1: 0.0 - 4.0" / "Shot 2 4.00–10.50".
_SHOT_RE = re.compile(
    r"^\s*shot\s*(?P<num>\d+)\s*[:\-)]?\s*"
    r"(?P<start>\d+(?:\.\d+)?)\s*[\-–—to]+\s*(?P<end>\d+(?:\.\d+)?)",
    re.IGNORECASE,
)

# Live UI shot header: "Shot 1of 3", "Shot 2 of 3". Times arrive on a later
# timecode line, not here.
_SHOT_HEADER_RE = re.compile(
    r"^\s*shot\s*(?P<num>\d+)\s*of\s*\d+\s*$",
    re.IGNORECASE,
)

# A timecode range line: "00:00:00.0–00:00:10.1" (HH:MM:SS.s, any dash).
_TIMECODE_RANGE_RE = re.compile(
    r"^\s*(?P<start>\d{1,2}:\d{2}:\d{2}(?:\.\d+)?)\s*[\-–—]\s*"
    r"(?P<end>\d{1,2}:\d{2}:\d{2}(?:\.\d+)?)\s*$",
)

# "Shots: 3" declared-count line.
_SHOT_COUNT_RE = re.compile(
    r"^\s*(?:number\s+of\s+)?shots?\s*[:=]\s*(?P<count>\d+)\s*$",
    re.IGNORECASE,
)

# Lines that are never a valid entity description (live UI noise / labels).
_DESC_NOISE = re.compile(
    r"^\s*(add\b|select\b|describe\b|tone\b|pitch\b|speaking level\b|"
    r"recorded level\b|mix role\b|clarity\b|speed\b|delivery\b|source\b|"
    r"transcription\b|voice\b|relationship\b|quick add\b|caption\b|"
    r"regenerate\b|make (sound|speech)\b|off-screen\b|None\.?\s*$|\d+\s*$)",
    re.IGNORECASE,
)

_ENTITY_STATEMENT_RE = re.compile(r"^(?:voice|sound)\s*:", re.IGNORECASE)
_LOCKED_DESCRIPTION_HINT_RE = re.compile(
    r"\b(?:visual appearance|appearance is|looks?\b|wearing\b|visible\b)",
    re.IGNORECASE,
)


def _description_after_bare_id(lines, start_index):
    """Recover a locked description from a verbose live-UI entity block."""
    fallback = ""
    option_noise = {
        "male", "female", "unclear", "observed", "observed briefly",
        "not observed", "mid", "normal", "ordinary", "moderate",
        "foreground", "supporting", "clear", "high", "breathy", "rough",
        "nasal", "squeaky", "makes a sound", "silent",
    }
    for follow in lines[start_index + 1:]:
        candidate = follow.strip()
        if not candidate:
            continue
        if (
            _ID_ONLY_RE.match(candidate)
            or _SHOT_HEADER_RE.match(candidate)
            or _SHOT_RE.match(candidate)
        ):
            break
        if _LOCKED_DESCRIPTION_HINT_RE.search(candidate):
            return candidate
        if (
            candidate.lower() in option_noise
            or _DESC_NOISE.match(candidate)
            or _ENTITY_STATEMENT_RE.match(candidate)
        ):
            continue
        # Compact seeds commonly put prose immediately after the id. Require
        # several words so a UI option cannot become a locked description.
        if not fallback and len(candidate.split()) >= 4:
            fallback = candidate
    return fallback


def _seed_listening_targets(text):
    """Return seed-named non-speech checks, never machine confirmations."""
    lines = [line.strip().lower() for line in text.splitlines() if line.strip()]

    def affirmative(pattern, subject):
        for line in lines:
            if not re.search(pattern, line):
                continue
            # UI answers such as "No wind noise is audible" are explicit
            # rejections, not listening targets. Limit negation to the nearby
            # phrase so unrelated "no clipping" text does not hide a target.
            if re.search(
                rf"\b(?:no|not|without)\b.{{0,30}}\b{subject}", line
            ):
                continue
            return True
        return False

    targets = []
    if affirmative(
        r"\b(?:chew\w*|mastication|eating sounds?)\b",
        r"(?:chew\w*|mastication|eating sounds?)",
    ):
        targets.append({"class": "chewing", "label": "chewing/eating sounds"})
    if affirmative(
        r"\b(?:obvious\s+wind|audible\s+wind|wind\s+(?:noise|sound)\s+(?:is\s+)?(?:audible|present|obvious)|wind\s+(?:noise|sound)\s+throughout)\b",
        r"wind(?:\s+(?:noise|sound))?",
    ):
        targets.append({"class": "wind_noise", "label": "wind noise"})
    if affirmative(
        r"\bbottles?\b.*\b(?:table|contact|hit\w*|sat|set|clink\w*)\b|"
        r"\b(?:table|contact|hit\w*|sat|set|clink\w*)\b.*\bbottles?\b",
        r"bottles?",
    ):
        targets.append({
            "class": "bottle_table_contact",
            "label": "bottle/table contact",
        })
    return targets


def _timecode_to_seconds(value):
    hours, minutes, seconds = value.split(":")
    return int(hours) * 3600 + int(minutes) * 60 + float(seconds)


def parse_seed_text(text):
    characters = []
    objects = []
    shots = []
    issues = []
    declared_shot_count = None

    seen_ids = {}
    lines = text.splitlines()

    # Shot boundary that separates the Cast section from the event section:
    # entity definitions before the first shot header are the real cast, the
    # bare ids after it are just event sources (already captured above).
    first_shot_line = None
    for i, raw in enumerate(lines):
        if _SHOT_HEADER_RE.match(raw.strip()) or _SHOT_RE.match(raw.strip()):
            first_shot_line = i
            break

    pending_shot = None  # a shot header awaiting its timecode line

    def add_entity(entity_id, description, line_no):
        entity_id = entity_id.upper()

        if entity_id in seen_ids:
            issues.append(
                f"line {line_no}: duplicate definition of {entity_id}; "
                "keeping the first"
            )
            return

        seen_ids[entity_id] = line_no
        record = {
            "id": entity_id,
            "description": description.strip(),
            "original": True,
        }
        (characters if entity_id.startswith("C") else objects).append(record)

        if not description.strip():
            issues.append(f"line {line_no}: {entity_id} has no description")

    for line_no, raw in enumerate(lines, start=1):
        line = raw.strip()
        index = line_no - 1

        if not line:
            continue

        shot_count = _SHOT_COUNT_RE.match(line)
        if shot_count:
            declared_shot_count = int(shot_count.group("count"))
            continue

        # Inline shot with plain seconds.
        shot = _SHOT_RE.match(line)
        if shot:
            start = float(shot.group("start"))
            end = float(shot.group("end"))
            if end <= start:
                issues.append(
                    f"line {line_no}: shot {shot.group('num')} has "
                    f"end <= start; skipped"
                )
            else:
                shots.append({
                    "shot": int(shot.group("num")),
                    "start": round(start, 3),
                    "end": round(end, 3),
                })
            continue

        # Live UI shot header -> remember it; the next timecode line has times.
        header = _SHOT_HEADER_RE.match(line)
        if header:
            pending_shot = int(header.group("num"))
            continue

        timecode = _TIMECODE_RANGE_RE.match(line)
        if timecode and pending_shot is not None:
            start = _timecode_to_seconds(timecode.group("start"))
            end = _timecode_to_seconds(timecode.group("end"))
            if end > start:
                shots.append({
                    "shot": pending_shot,
                    "start": round(start, 3),
                    "end": round(end, 3),
                })
            pending_shot = None
            continue

        # Inline "C1: description".
        definition = _DEF_RE.match(line)
        if definition and definition.group("desc"):
            add_entity(
                definition.group("id"),
                definition.group("desc"),
                line_no,
            )
            continue

        # Bare id on its own line (live UI): description is the next real line.
        # Only trust these in the Cast section (before the first shot).
        id_only = _ID_ONLY_RE.match(line)
        if id_only and (first_shot_line is None or index < first_shot_line):
            description = _description_after_bare_id(lines, index)
            add_entity(id_only.group("id"), description, line_no)
            continue

    characters.sort(key=lambda r: int(r["id"][1:]))
    objects.sort(key=lambda r: int(r["id"][1:]))
    shots.sort(key=lambda r: r["shot"])

    character_ids = [r["id"] for r in characters]
    object_ids = [r["id"] for r in objects]

    if declared_shot_count is not None and declared_shot_count != len(shots):
        issues.append(
            f"declared shot count {declared_shot_count} does not match "
            f"{len(shots)} parsed shot boundaries"
        )

    if not shots:
        issues.append("no shot boundaries parsed from seed")

    return {
        "characters": character_ids,
        "objects": object_ids,
        "shots": shots,
        "seed_meta": {
            "characters": characters,
            "objects": objects,
            "original_character_ids": character_ids,
            "original_object_ids": object_ids,
            "shot_count_declared": declared_shot_count,
            "human_listening_targets": _seed_listening_targets(text),
            "parse_issues": issues,
        },
    }


def write_task_context(parsed, context_path=CONTEXT, preserve_sha=True):
    context_path = Path(context_path)

    result = dict(parsed)

    # Bind the task to the CURRENT video. Prefer the fingerprint that the
    # pipeline wrote for the analyzed video (video_identity.json); only fall
    # back to an existing task_context fingerprint. This prevents a stale
    # fingerprint from a previous clip from lingering on a new video.
    identity_path = context_path.parent / "analysis" / "video_identity.json"
    current_sha = None

    if identity_path.exists():
        try:
            with identity_path.open("r", encoding="utf-8-sig") as f:
                current_sha = json.load(f).get("video_sha256")
        except (json.JSONDecodeError, OSError):
            pass

    if current_sha:
        result["video_sha256"] = current_sha
    elif preserve_sha and context_path.exists():
        try:
            with context_path.open("r", encoding="utf-8-sig") as f:
                existing = json.load(f)
            if existing.get("video_sha256"):
                result["video_sha256"] = existing["video_sha256"]
        except (json.JSONDecodeError, OSError):
            pass

    with context_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    return result


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)

    if argv:
        text = Path(argv[0]).read_text(encoding="utf-8-sig")
    else:
        text = sys.stdin.read()

    if not text.strip():
        raise SystemExit(
            "No seed text provided. Pass a file path or pipe seed text in."
        )

    parsed = parse_seed_text(text)
    result = write_task_context(parsed)

    meta = result["seed_meta"]

    print("=== TASK SEED PARSER ===")
    print("Characters:", ", ".join(result["characters"]) or "(none)")
    print("Objects:", ", ".join(result["objects"]) or "(none)")
    print("Shots:", len(result["shots"]))

    if meta["parse_issues"]:
        print("Parse issues:")
        for issue in meta["parse_issues"]:
            print("  -", issue)

    print("Wrote:", CONTEXT)
    return result


if __name__ == "__main__":
    main()
