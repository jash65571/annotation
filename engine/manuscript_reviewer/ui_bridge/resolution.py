"""Engine-owned review targeting and resolution for the UI.

The frontend must never infer a decision target from title strings or first
evidence ids. This module rebuilds the SAME typed registries the engine's
decision appliers use (via ``load_run_evidence``), replays the run's current
human decisions through ``apply_decisions``, and reports for every review
item / decidable subject:

- a typed ``decision_target`` (kind + subject id + admissible decision types),
  matched only by EXACT registry-key identity — no fuzzy matching;
- an engine-computed ``resolution_status`` (OPEN / RESOLVED) derived from the
  actual DecisionApplication outcomes, never from local UI state.

Nothing here mutates artifacts: registries are in-memory reconstructions and
the replay is discarded after reporting.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..caption_brain import CaptionBrainError, load_run_evidence
from ..models.review_intelligence import DecisionType
from ..review.decisions import (
    DecisionLoadError,
    DecisionTargets,
    apply_decisions,
    load_decisions,
)
from .protocol import BridgeCommandError, BridgeErrorCode
from .serializers import read_artifact, review_decisions_path, review_queue

#: Admissible decision types per target kind — mirrors the engine's
#: DecisionType → registry dispatch (review/decisions.py).
TARGET_DECISION_TYPES: dict[str, list[str]] = {
    "SPEECH_REGION": [
        DecisionType.SPEECH_VERIFICATION.value,
        DecisionType.SPEECH_CORRECTION.value,
    ],
    "TEXT_TRACK": [
        DecisionType.TEXT_VERIFICATION.value,
        DecisionType.TEXT_CORRECTION.value,
        DecisionType.TEXT_TIMING.value,
        DecisionType.OCR_TEXT.value,
        DecisionType.OCR_TIMING.value,
    ],
    "ENTITY_TRACK": [DecisionType.IDENTITY_MAPPING.value],
    "ACTION_CANDIDATE": [
        DecisionType.ACTION_BOUNDARY.value,
        DecisionType.ACTION_SEMANTICS.value,
    ],
    "CAMERA_EVENT": [DecisionType.CAMERA_CLASSIFICATION.value],
    "SHOT_TRANSITION": [DecisionType.TRANSITION_CLASSIFICATION.value],
    "PLAYBACK_SPEED": [DecisionType.PLAYBACK_SPEED.value],
    "REVIEW_PROPOSAL": [DecisionType.REVIEW_PROPOSAL_OUTCOME.value],
    "SEED_CLAIM": [DecisionType.CLAIM_EVIDENCE.value],
}


def _build_registries(run_dir: Path) -> tuple[DecisionTargets, Any]:
    try:
        evidence, _hashes = load_run_evidence(run_dir)
    except CaptionBrainError as exc:
        raise BridgeCommandError(
            BridgeErrorCode.MANIFEST_MISMATCH,
            "Run evidence could not be verified for review resolution",
            detail=str(exc),
        ) from exc
    targets = DecisionTargets(
        claims={c.claim_id: c for c in evidence.seed_claims},
        entity_tracks={t.track_id: t for t in evidence.entity_tracks},
        camera_candidates={c.candidate_id: c for c in evidence.camera_candidates},
        action_candidates={a.candidate_id: a for a in evidence.action_candidates},
        speed_evidence={f"SPEED-{s.shot_number}": s for s in evidence.speed_evidence},
        proposals={p.proposal_id: p for p in evidence.proposals},
        speech_regions=(
            {r.region_id: r for r in evidence.audio_truth.speech_regions}
            if evidence.audio_truth is not None
            else {}
        ),
        text_tracks={t.track_id: t for t in evidence.text_tracks},
        transitions=(
            {f"TRANSITION-{s.shot_index}": s for s in evidence.shot_truth.shots}
            if evidence.shot_truth is not None
            else {}
        ),
        frame_to_time=evidence.frame_to_time,
        shot_frame_ranges=(
            {
                s.shot_index: (s.start_frame_index, s.end_frame_index)
                for s in evidence.shot_truth.shots
            }
            if evidence.shot_truth is not None
            else {}
        ),
    )
    return targets, evidence


def _registry_kinds(targets: DecisionTargets) -> list[tuple[str, dict[str, Any]]]:
    return [
        ("SPEECH_REGION", targets.speech_regions),
        ("TEXT_TRACK", targets.text_tracks),
        ("ENTITY_TRACK", targets.entity_tracks),
        ("ACTION_CANDIDATE", targets.action_candidates),
        ("CAMERA_EVENT", targets.camera_candidates),
        ("SHOT_TRANSITION", targets.transitions),
        ("PLAYBACK_SPEED", targets.speed_evidence),
        ("REVIEW_PROPOSAL", targets.proposals),
        ("SEED_CLAIM", targets.claims),
    ]


def _target_dto(kind: str, subject_id: str) -> dict[str, Any]:
    return {
        "target_kind": kind,
        "subject_id": subject_id,
        "allowed_decision_types": TARGET_DECISION_TYPES[kind],
    }


def _item_candidate_ids(item: dict[str, Any]) -> list[str]:
    """Every id-shaped string the ENGINE attached to this review item. Only
    exact registry-key matches ever become targets."""
    ids: list[str] = []
    ids.extend(str(v) for v in item.get("related_claim_ids", []) or [])
    for key in ("supporting_evidence_refs", "contradicting_evidence_refs"):
        for ref in item.get(key, []) or []:
            if isinstance(ref, dict):
                if ref.get("evidence_id"):
                    ids.append(str(ref["evidence_id"]))
                if ref.get("source"):
                    ids.append(str(ref["source"]))
    # Audio review items carry raw evidence-ref id strings.
    for ref in item.get("evidence_refs", []) or []:
        if isinstance(ref, str):
            ids.append(ref)
    return ids


def review_resolution(run_dir: Path) -> dict[str, Any]:
    """Typed targets + engine resolution for every review item and decidable
    subject in the run."""
    targets, evidence = _build_registries(run_dir)

    decisions = []
    decisions_file = review_decisions_path(run_dir)
    if decisions_file is not None:
        try:
            decisions = load_decisions(decisions_file)
        except DecisionLoadError as exc:
            raise BridgeCommandError(
                BridgeErrorCode.INVALID_DECISION,
                "Saved review decisions failed engine validation",
                detail=str(exc),
            ) from exc

    rules_version = evidence.rules_version or ""
    video_sha = evidence.video_sha256 or ""
    applications = (
        apply_decisions(decisions, targets, video_sha, rules_version)
        if decisions and video_sha
        else []
    )
    application_dtos = [
        {
            "decision_id": app.decision_id,
            "outcome": app.outcome.value,
            "applied": app.applied,
            "stale": app.stale,
            "reason": app.reason,
        }
        for app in applications
    ]
    applied_by_decision = {
        app.decision_id: app.applied for app in applications
    }
    #: subject -> APPLIED decision ids (engine truth: only APPLIED resolves).
    applied_subjects: dict[str, list[str]] = {}
    for decision in decisions:
        if applied_by_decision.get(decision.decision_id):
            applied_subjects.setdefault(decision.subject_id, []).append(
                decision.decision_id
            )

    registries = _registry_kinds(targets)

    def resolve_item(item: dict[str, Any]) -> dict[str, Any]:
        matches: list[tuple[str, str]] = []
        for candidate in _item_candidate_ids(item):
            for kind, registry in registries:
                if candidate in registry and (kind, candidate) not in matches:
                    matches.append((kind, candidate))
        target = _target_dto(*matches[0]) if len(matches) == 1 else None
        resolved_ids = [
            decision_id
            for kind, subject in matches
            for decision_id in applied_subjects.get(subject, [])
        ]
        return {
            "item_id": item.get("item_id"),
            "decision_target": target,
            "candidate_targets": [_target_dto(k, s) for k, s in matches],
            "resolution_status": "RESOLVED" if resolved_ids else "OPEN",
            "resolved_by_decision_ids": resolved_ids,
        }

    queue = review_queue(run_dir)
    item_resolutions = [
        resolve_item(item)
        for item in [*queue["visual_items"], *queue["audio_items"]]
    ]

    # Standalone decidable subjects the engine says still need humans —
    # derived from registry state, never from blocker strings.
    speed_targets = [
        {
            **_target_dto("PLAYBACK_SPEED", subject_id),
            "shot_number": entry.shot_number,
            "machine_conclusion": str(entry.conclusion.value),
            "review_required": bool(entry.review_required),
            "resolution_status": (
                "RESOLVED" if applied_subjects.get(subject_id) else "OPEN"
            ),
            "resolved_by_decision_ids": applied_subjects.get(subject_id, []),
        }
        for subject_id, entry in sorted(targets.speed_evidence.items())
    ]
    transition_targets = [
        {
            **_target_dto("SHOT_TRANSITION", subject_id),
            "shot_index": shot.shot_index,
            "transition_status": str(shot.transition_status.value),
            "transition_into_shot": shot.transition_into_shot,
            "resolution_status": (
                "RESOLVED" if applied_subjects.get(subject_id) else "OPEN"
            ),
            "resolved_by_decision_ids": applied_subjects.get(subject_id, []),
        }
        for subject_id, shot in sorted(targets.transitions.items())
    ]

    return {
        "items": item_resolutions,
        "speed_targets": speed_targets,
        "transition_targets": transition_targets,
        "applications": application_dtos,
    }


def media_dimensions(run_dir: Path) -> dict[str, Any]:
    """The verified Phase 1 media dimensions (source-pixel truth for the
    anchor editor). Never inferred from CSS or the video element."""
    media = read_artifact(run_dir, "media.json")
    # media.json wraps the MediaInfo record under "media" (with raw probe
    # alongside); accept a bare MediaInfo dump too.
    record = media.get("media", media) if isinstance(media, dict) else None
    streams = record.get("video_streams") if isinstance(record, dict) else None
    if not isinstance(streams, list) or not streams:
        raise BridgeCommandError(
            BridgeErrorCode.ARTIFACT_NOT_FOUND,
            "Run media.json has no video stream record",
        )
    first = streams[0]
    width = first.get("width")
    height = first.get("height")
    if not isinstance(width, int) or not isinstance(height, int):
        raise BridgeCommandError(
            BridgeErrorCode.ARTIFACT_NOT_FOUND,
            "Run media.json lacks integer width/height",
        )
    return {"width": width, "height": height}
