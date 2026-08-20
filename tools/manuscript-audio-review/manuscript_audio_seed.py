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

# "C1:", "O2 -", "@C3)" ... a definition line binds an id to a description.
_DEF_RE = re.compile(
    r"^\s*@?(?P<id>[CO]\d+)\s*[:\-)–—]\s*(?P<desc>.*\S)?",
    re.IGNORECASE,
)

# "Shot 1: 0.0 - 4.0" / "Shot 2 4.00–10.50"
_SHOT_RE = re.compile(
    r"^\s*shot\s*(?P<num>\d+)\s*[:\-)]?\s*"
    r"(?P<start>\d+(?:\.\d+)?)\s*[\-–—to]+\s*(?P<end>\d+(?:\.\d+)?)",
    re.IGNORECASE,
)

# "Shots: 3" declared-count line.
_SHOT_COUNT_RE = re.compile(
    r"^\s*(?:number\s+of\s+)?shots?\s*[:=]\s*(?P<count>\d+)\s*$",
    re.IGNORECASE,
)


def parse_seed_text(text):
    characters = []
    objects = []
    shots = []
    issues = []
    declared_shot_count = None

    seen_ids = {}

    for line_no, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()

        if not line:
            continue

        shot_count = _SHOT_COUNT_RE.match(line)
        if shot_count:
            declared_shot_count = int(shot_count.group("count"))
            continue

        shot = _SHOT_RE.match(line)
        if shot:
            start = float(shot.group("start"))
            end = float(shot.group("end"))

            if end <= start:
                issues.append(
                    f"line {line_no}: shot {shot.group('num')} has "
                    f"end <= start ({start} -> {end}); skipped"
                )
                continue

            shots.append({
                "shot": int(shot.group("num")),
                "start": round(start, 3),
                "end": round(end, 3),
            })
            continue

        definition = _DEF_RE.match(line)
        if definition:
            entity_id = definition.group("id").upper()
            description = (definition.group("desc") or "").strip()

            if entity_id in seen_ids:
                issues.append(
                    f"line {line_no}: duplicate definition of {entity_id}; "
                    "keeping the first"
                )
                continue

            seen_ids[entity_id] = line_no

            record = {
                "id": entity_id,
                "description": description,
                "original": True,
            }

            if entity_id.startswith("C"):
                characters.append(record)
            else:
                objects.append(record)

            if not description:
                issues.append(
                    f"line {line_no}: {entity_id} has no description"
                )

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
            "parse_issues": issues,
        },
    }


def write_task_context(parsed, context_path=CONTEXT, preserve_sha=True):
    context_path = Path(context_path)

    result = dict(parsed)

    # Preserve an already-bound video fingerprint so the identity guards keep
    # working; the seed itself carries no fingerprint.
    if preserve_sha and context_path.exists():
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
