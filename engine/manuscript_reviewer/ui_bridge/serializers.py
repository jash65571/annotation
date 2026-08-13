"""Run-directory readers and safe persistence for the UI bridge.

The run directory remains the single source of truth. Readers return the
existing artifact JSON (already typed, deterministic, sorted-key) wrapped in
thin DTO envelopes — no parallel truth models. Writers persist only the three
human-input files the engine already consumes (review decisions, human facts,
final signoff), validated against the existing typed models and written
atomically (temp file + rename) so a crash can never corrupt them.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from ..media.ffmpeg_tools import FFmpegNotFoundError, find_tool, run_tool
from ..models.caption_brain import FinalReviewSignoff, HumanCaptionFact
from ..models.review_intelligence import HumanReviewDecision
from .protocol import BridgeCommandError, BridgeErrorCode

#: Human-input files owned by the app, inside the run directory. The engine
#: validates them again on finalize — the bridge never applies them itself.
UI_DIR_NAME = "ui"
REVIEW_DECISIONS_FILE = "review_decisions.json"
HUMAN_FACTS_FILE = "human_facts.json"
FINAL_REVIEW_FILE = "final_review.json"
FRAME_CACHE_DIR = "frame_cache"


def ui_dir(run_dir: Path) -> Path:
    return run_dir / UI_DIR_NAME


def review_decisions_path(run_dir: Path) -> Path | None:
    path = ui_dir(run_dir) / REVIEW_DECISIONS_FILE
    return path if path.exists() else None


def human_facts_path(run_dir: Path) -> Path | None:
    path = ui_dir(run_dir) / HUMAN_FACTS_FILE
    return path if path.exists() else None


def final_review_path(run_dir: Path) -> Path | None:
    path = ui_dir(run_dir) / FINAL_REVIEW_FILE
    return path if path.exists() else None


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------


def require_run_dir(raw: str) -> Path:
    run_dir = Path(raw)
    if not run_dir.is_dir() or not (run_dir / "manifest.json").exists():
        raise BridgeCommandError(
            BridgeErrorCode.RUN_NOT_FOUND,
            f"Not an audit run directory: {raw}",
        )
    return run_dir


def read_artifact(run_dir: Path, relative: str, required: bool = True) -> Any:
    path = run_dir / relative
    if not path.exists():
        if required:
            raise BridgeCommandError(
                BridgeErrorCode.ARTIFACT_NOT_FOUND,
                f"Run artifact missing: {relative}",
            )
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BridgeCommandError(
            BridgeErrorCode.ARTIFACT_NOT_FOUND,
            f"Run artifact unreadable: {relative}",
            detail=str(exc),
        ) from exc


def read_text_artifact(run_dir: Path, relative: str, required: bool = True) -> str | None:
    path = run_dir / relative
    if not path.exists():
        if required:
            raise BridgeCommandError(
                BridgeErrorCode.ARTIFACT_NOT_FOUND,
                f"Run artifact missing: {relative}",
            )
        return None
    return path.read_text(encoding="utf-8")


def run_summary(run_dir: Path) -> dict[str, Any]:
    """The reviewer-facing summary of one run: identity, statuses, counts."""
    manifest = read_artifact(run_dir, "manifest.json")
    qc = read_artifact(run_dir, "qc.json", required=False)
    final_status = read_artifact(run_dir, "caption/final_status.json", required=False)
    audio_qc = read_artifact(run_dir, "audio/audio_qc.json", required=False)
    visual_qc = read_artifact(run_dir, "visual/visual_qc.json", required=False)
    shot_qc = read_artifact(run_dir, "shot_qc.json", required=False)
    return {
        "run_dir": str(run_dir),
        "manifest": manifest,
        "qc": qc,
        "caption_final_status": final_status,
        "audio_qc": audio_qc,
        "visual_qc": visual_qc,
        "shot_qc": shot_qc,
        "has_extracted_frames": (run_dir / "frames").is_dir(),
        "ui_inputs": {
            "review_decisions": review_decisions_path(run_dir) is not None,
            "human_facts": human_facts_path(run_dir) is not None,
            "final_review": final_review_path(run_dir) is not None,
        },
    }


def review_queue(run_dir: Path) -> dict[str, Any]:
    """The merged review queue: visual/caption items + audio items, tagged by
    source. Items pass through exactly as the engine wrote them."""
    visual = read_artifact(run_dir, "review/visual_review_queue.json", required=False)
    audio = read_artifact(run_dir, "audio/audio_review_queue.json", required=False)
    return {
        "visual_items": _as_item_list(visual),
        "audio_items": _as_item_list(audio),
    }


def _as_item_list(raw: Any) -> list[dict[str, Any]]:
    if raw is None:
        return []
    if isinstance(raw, dict):
        for key in ("items", "queue"):
            value = raw.get(key)
            if isinstance(value, list):
                return [entry for entry in value if isinstance(entry, dict)]
        return []
    if isinstance(raw, list):
        return [entry for entry in raw if isinstance(entry, dict)]
    return []


def shots(run_dir: Path) -> dict[str, Any]:
    return {
        "shots_proposed": read_artifact(run_dir, "shots_proposed.json", required=False),
        "transition_evidence": read_artifact(
            run_dir, "transition_evidence.json", required=False
        ),
        "cut_candidates": read_artifact(run_dir, "cut_candidates.json", required=False),
        "shot_qc": read_artifact(run_dir, "shot_qc.json", required=False),
    }


def frame_record(run_dir: Path, frame_index: int) -> dict[str, Any]:
    """One exact frame-ledger record, straight from frames.jsonl."""
    path = run_dir / "frames.jsonl"
    if not path.exists():
        raise BridgeCommandError(
            BridgeErrorCode.ARTIFACT_NOT_FOUND, "Run has no frames.jsonl ledger"
        )
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if isinstance(record, dict) and record.get("frame_index") == frame_index:
                return record
    raise BridgeCommandError(
        BridgeErrorCode.INVALID_INPUT,
        f"Frame {frame_index} is not in the run's frame ledger",
    )


def exact_frame(run_dir: Path, frame_index: int) -> dict[str, Any]:
    """The exact decoded source frame for a ledger index.

    Serves the Phase 1 extracted evidence image when present; otherwise
    extracts that single frame on demand (ffmpeg select by decode index, the
    same identity the ledger uses) into a UI cache inside the run directory.
    """
    record = frame_record(run_dir, frame_index)
    frames_dir = run_dir / "frames"
    if frames_dir.is_dir():
        matches = sorted(frames_dir.glob(f"F{frame_index:06d}_*.png"))
        if matches:
            return {"frame_index": frame_index, "record": record, "path": str(matches[0])}
    cache_dir = ui_dir(run_dir) / FRAME_CACHE_DIR
    cache_path = cache_dir / f"F{frame_index:06d}.png"
    if cache_path.exists():
        return {"frame_index": frame_index, "record": record, "path": str(cache_path)}
    manifest = read_artifact(run_dir, "manifest.json")
    video_path = Path(str(manifest.get("source_video_path", "")))
    if not video_path.exists():
        raise BridgeCommandError(
            BridgeErrorCode.ARTIFACT_NOT_FOUND,
            "Source video is no longer at its recorded path; cannot extract "
            "an exact frame",
            detail=str(video_path),
        )
    cache_dir.mkdir(parents=True, exist_ok=True)
    try:
        ffmpeg = find_tool("ffmpeg")
    except FFmpegNotFoundError as exc:
        raise BridgeCommandError(
            BridgeErrorCode.FFMPEG_UNAVAILABLE, "FFmpeg is unavailable", detail=str(exc)
        ) from exc
    # -fps_mode passthrough + select by decode index n: identical frame
    # identity to the Phase 1 ledger/extractor (no dup/drop resampling).
    run_tool(
        ffmpeg,
        [
            "-v", "error",
            "-i", str(video_path),
            "-map", "0:v:0",
            "-fps_mode", "passthrough",
            "-vf", f"select='eq(n\\,{frame_index})'",
            "-frames:v", "1",
            "-y",
            str(cache_path),
        ],
        timeout=120.0,
    )
    if not cache_path.exists():
        raise BridgeCommandError(
            BridgeErrorCode.ENGINE_CRASH,
            f"FFmpeg produced no image for frame {frame_index}",
        )
    return {"frame_index": frame_index, "record": record, "path": str(cache_path)}


def caption_state(run_dir: Path) -> dict[str, Any]:
    """Everything the caption panel needs: readiness, draft/ready text, gates."""
    final_status = read_artifact(run_dir, "caption/final_status.json", required=False)
    reviewed = read_artifact(run_dir, "caption/reviewed_caption.json", required=False)
    caption_manifest = read_artifact(
        run_dir, "caption/caption_manifest.json", required=False
    )
    ready_md = read_text_artifact(run_dir, "caption/ready_to_enter.md", required=False)
    draft_md = read_text_artifact(
        run_dir, "caption/draft_review_only.md", required=False
    )
    rendered_sha = (
        caption_manifest.get("rendered_caption_sha256")
        if isinstance(caption_manifest, dict)
        else None
    )
    return {
        "final_status": final_status,
        "reviewed_caption": reviewed,
        "caption_manifest": caption_manifest,
        #: The signoff-binding hash of the current rendered caption.
        "rendered_caption_sha256": rendered_sha,
        "ready_markdown": ready_md,
        "draft_markdown": draft_md,
        "m2_validator": read_artifact(run_dir, "caption/m2_validator.json", required=False),
        "platform_semantic": read_artifact(
            run_dir, "caption/platform_semantic_report.json", required=False
        ),
        "golden_gate": read_artifact(run_dir, "caption/golden_gate.json", required=False),
        "coverage": read_artifact(run_dir, "caption/caption_coverage.json", required=False),
        "assertion_map": read_artifact(
            run_dir, "caption/caption_assertion_map.json", required=False
        ),
        "seed_change_log": read_artifact(
            run_dir, "caption/seed_change_log.json", required=False
        ),
        "caption_facts": read_artifact(
            run_dir, "caption/caption_facts.json", required=False
        ),
        "eligibility_report": read_artifact(
            run_dir, "caption/eligibility_report.json", required=False
        ),
        "final_review_checklist": read_artifact(
            run_dir, "caption/final_review_checklist.json", required=False
        ),
    }


def waveform_metadata(run_dir: Path) -> dict[str, Any]:
    audio_dir = run_dir / "audio"
    if not audio_dir.is_dir():
        raise BridgeCommandError(
            BridgeErrorCode.ARTIFACT_NOT_FOUND, "Run has no audio artifacts"
        )

    def _existing(name: str) -> str | None:
        path = audio_dir / name
        return str(path) if path.exists() else None

    return {
        "timeline": read_artifact(run_dir, "audio/audio_timeline.json", required=False),
        "waveform_png": _existing("waveform.png"),
        "spectrogram_png": _existing("spectrogram.png"),
        "energy_png": _existing("energy.png"),
        "source_wav": _existing("source.wav"),
        "audio_qc": read_artifact(run_dir, "audio/audio_qc.json", required=False),
    }


def audio_review_clip(run_dir: Path, item_id: str) -> dict[str, Any]:
    queue = review_queue(run_dir)["audio_items"]
    for item in queue:
        if item.get("item_id") == item_id:
            clip_rel = item.get("review_clip")
            clip_path = run_dir / str(clip_rel) if clip_rel else None
            return {
                "item": item,
                "clip_path": (
                    str(clip_path) if clip_path is not None and clip_path.exists() else None
                ),
                "source_wav": (
                    str(run_dir / "audio" / "source.wav")
                    if (run_dir / "audio" / "source.wav").exists()
                    else None
                ),
            }
    raise BridgeCommandError(
        BridgeErrorCode.INVALID_INPUT, f"Unknown audio review item: {item_id}"
    )


def evidence_bundle(run_dir: Path, bundle_dir: str) -> dict[str, Any]:
    """Contents of one shot-evidence bundle directory (paths + evidence.json)."""
    bundle = (run_dir / bundle_dir).resolve()
    if not str(bundle).startswith(str(run_dir.resolve())):
        raise BridgeCommandError(
            BridgeErrorCode.INVALID_INPUT, "Evidence bundle path escapes the run directory"
        )
    if not bundle.is_dir():
        raise BridgeCommandError(
            BridgeErrorCode.ARTIFACT_NOT_FOUND, f"Evidence bundle missing: {bundle_dir}"
        )
    evidence_json = bundle / "evidence.json"
    return {
        "bundle_dir": str(bundle),
        "images": sorted(str(p) for p in bundle.glob("*.png")),
        "evidence": (
            json.loads(evidence_json.read_text(encoding="utf-8"))
            if evidence_json.exists()
            else None
        ),
    }


# ---------------------------------------------------------------------------
# Writing (atomic, validated)
# ---------------------------------------------------------------------------


def atomic_write_json(path: Path, data: Any) -> None:
    """Write JSON via temp file + fsync + atomic rename; never leaves a
    half-written human-input file behind."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        dir=str(path.parent), prefix=path.name, suffix=".tmp"
    )
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(data, handle, indent=2, sort_keys=True, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        tmp_path.replace(path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)


_NON_HUMAN_AUTHORS = {"", "machine", "ai", "system"}


def save_review_decisions(run_dir: Path, decisions: list[dict[str, Any]]) -> dict[str, Any]:
    """Validate + persist the run's human review decisions (typed, bound)."""
    validated: list[HumanReviewDecision] = []
    for entry in decisions:
        try:
            decision = HumanReviewDecision.model_validate(entry)
        except ValidationError as exc:
            raise BridgeCommandError(
                BridgeErrorCode.INVALID_DECISION,
                "Review decision failed validation",
                detail=str(exc),
            ) from exc
        if (decision.decided_by or "").strip().lower() in _NON_HUMAN_AUTHORS:
            raise BridgeCommandError(
                BridgeErrorCode.INVALID_DECISION,
                f"Decision {decision.decision_id} has non-human decided_by",
            )
        if not decision.bound_video_sha256 or not decision.bound_rules_version:
            raise BridgeCommandError(
                BridgeErrorCode.INVALID_DECISION,
                f"Decision {decision.decision_id} is missing binding keys",
            )
        validated.append(decision)
    path = ui_dir(run_dir) / REVIEW_DECISIONS_FILE
    atomic_write_json(
        path, {"decisions": [d.model_dump(mode="json") for d in validated]}
    )
    return {"path": str(path), "count": len(validated)}


def save_human_facts(run_dir: Path, facts: list[dict[str, Any]]) -> dict[str, Any]:
    """Validate + persist reviewer-added caption facts (must carry evidence)."""
    validated: list[HumanCaptionFact] = []
    for entry in facts:
        try:
            fact = HumanCaptionFact.model_validate(entry)
        except ValidationError as exc:
            raise BridgeCommandError(
                BridgeErrorCode.INVALID_DECISION,
                "Human fact failed validation",
                detail=str(exc),
            ) from exc
        if (fact.decided_by or "").strip().lower() in _NON_HUMAN_AUTHORS:
            raise BridgeCommandError(
                BridgeErrorCode.INVALID_DECISION,
                f"Human fact {fact.fact_id} has non-human decided_by",
            )
        if not fact.evidence_refs:
            raise BridgeCommandError(
                BridgeErrorCode.INVALID_DECISION,
                f"Human fact {fact.fact_id} carries no evidence reference; "
                "facts without evidence cannot be saved as verified",
            )
        validated.append(fact)
    path = ui_dir(run_dir) / HUMAN_FACTS_FILE
    atomic_write_json(path, {"facts": [f.model_dump(mode="json") for f in validated]})
    return {"path": str(path), "count": len(validated)}


def read_review_inputs(run_dir: Path) -> dict[str, Any]:
    """The app-owned human-input files, for restoring UI state on reopen."""
    decisions_file = review_decisions_path(run_dir)
    facts_file = human_facts_path(run_dir)
    decisions_raw: Any = None
    facts_raw: Any = None
    if decisions_file is not None:
        decisions_raw = json.loads(decisions_file.read_text(encoding="utf-8"))
    if facts_file is not None:
        facts_raw = json.loads(facts_file.read_text(encoding="utf-8"))
    return {
        "decisions": (decisions_raw or {}).get("decisions", [])
        if isinstance(decisions_raw, dict)
        else [],
        "facts": (facts_raw or {}).get("facts", []) if isinstance(facts_raw, dict) else [],
    }


VISUAL_ANCHORS_FILE = "visual_anchors.json"


def save_visual_anchors(run_dir: Path, anchors: list[dict[str, Any]]) -> dict[str, Any]:
    """Persist reviewer-drawn anchors (source-pixel boxes). Structural check
    only — the engine validates content when the anchors are consumed."""
    required = {"frame_index", "x", "y", "width", "height", "entity_type", "label"}
    for anchor in anchors:
        if not isinstance(anchor, dict) or not required.issubset(anchor):
            raise BridgeCommandError(
                BridgeErrorCode.INVALID_INPUT,
                f"Anchor entries need fields {sorted(required)}",
            )
    path = ui_dir(run_dir) / VISUAL_ANCHORS_FILE
    atomic_write_json(path, {"anchors": anchors})
    return {"path": str(path), "count": len(anchors)}


def save_final_signoff(run_dir: Path, signoff_raw: dict[str, Any]) -> dict[str, Any]:
    """Validate + persist the human final-review signoff. The bridge never
    fabricates or pre-checks anything: every field comes from the UI form the
    human filled in, and the engine re-validates staleness on finalize."""
    try:
        signoff = FinalReviewSignoff.model_validate(signoff_raw)
    except ValidationError as exc:
        raise BridgeCommandError(
            BridgeErrorCode.INVALID_DECISION,
            "Final signoff failed validation",
            detail=str(exc),
        ) from exc
    if (signoff.reviewer or "").strip().lower() in _NON_HUMAN_AUTHORS:
        raise BridgeCommandError(
            BridgeErrorCode.INVALID_DECISION, "Final signoff requires a human reviewer name"
        )
    path = ui_dir(run_dir) / FINAL_REVIEW_FILE
    atomic_write_json(path, signoff.model_dump(mode="json"))
    return {"path": str(path)}
