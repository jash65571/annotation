/**
 * Camera classification editor.
 *
 * CAMERA_CLASSIFICATION values must be valid engine CameraMotionClass enum
 * values. If the rules expose a camera class list we use it; otherwise we fall
 * back to the engine's own enum (models/review_intelligence.py
 * CameraMotionClass) — mirrored here, never invented. 2D pixel evidence can
 * show global motion, but it cannot prove dolly/track/push/pull.
 */

import { useEffect, useState, type ReactElement } from "react";
import { getRules } from "../../../api/bridge";
import type { DecisionsStore } from "../decisionsStore";
import { getReviewerName, ReviewerNameField } from "../reviewerName";

export interface CameraEditorProps {
  candidateId: string;
  machineClass?: string;
  machineDirection?: string;
  store: DecisionsStore;
  onResolved: (outcome: unknown) => void;
}

/** Exact engine vocabulary: CameraMotionClass in models/review_intelligence.py. */
const ENGINE_CAMERA_CLASSES: string[] = [
  "STATIC",
  "HORIZONTAL_GLOBAL_MOTION",
  "VERTICAL_GLOBAL_MOTION",
  "DIAGONAL_GLOBAL_MOTION",
  "SCALE_INCREASE",
  "SCALE_DECREASE",
  "ROTATION",
  "HANDHELD_DRIFT",
  "HANDHELD_SHAKE",
  "UNRESOLVED",
];

function cameraClassesFromRules(rules: Record<string, unknown>): string[] | null {
  // Look for a rules-provided camera class list under a few plausible homes;
  // the current rules file does not define one, so the engine enum is the
  // usual source.
  const candidates: unknown[] = [
    rules["camera_motion_classes"],
    (rules["camera"] as Record<string, unknown> | undefined)?.["motion_classes"],
    (rules["camera"] as Record<string, unknown> | undefined)?.["allowed_classes"],
  ];
  for (const candidate of candidates) {
    if (
      Array.isArray(candidate) &&
      candidate.length > 0 &&
      candidate.every((value): value is string => typeof value === "string")
    ) {
      return candidate;
    }
  }
  return null;
}

export function CameraEditor({
  candidateId,
  machineClass,
  machineDirection,
  store,
  onResolved,
}: CameraEditorProps): ReactElement {
  const [classes, setClasses] = useState<string[]>(ENGINE_CAMERA_CLASSES);
  const [selected, setSelected] = useState("");
  const [note, setNote] = useState("");
  const [reviewerName, setReviewerNameState] = useState(getReviewerName());
  const nameMissingAtMount = useState(getReviewerName() === "")[0];
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    getRules()
      .then((payload) => {
        if (cancelled) return;
        const fromRules = cameraClassesFromRules(payload.rules);
        if (fromRules != null) setClasses(fromRules);
      })
      .catch(() => {
        // Rules unavailable — keep the engine enum fallback.
      });
    return () => {
      cancelled = true;
    };
  }, []);

  async function save(): Promise<void> {
    if (selected === "") return;
    setError(null);
    setSaving(true);
    try {
      const outcome = await store.applyDecision(
        store.makeDecision({
          subject_id: candidateId,
          decision_type: "CAMERA_CLASSIFICATION",
          value: selected,
          decided_by: reviewerName,
          ...(note.trim() !== "" ? { reviewer_note: note.trim() } : {}),
        }),
      );
      onResolved(outcome);
    } catch (err) {
      setError(String(err));
    } finally {
      setSaving(false);
    }
  }

  return (
    <div className="panel col" aria-label={`Camera editor for ${candidateId}`}>
      <h3 style={{ margin: 0 }}>Camera — {candidateId}</h3>
      {nameMissingAtMount && <ReviewerNameField onChange={setReviewerNameState} />}
      {(machineClass != null || machineDirection != null) && (
        <p style={{ margin: 0 }}>
          <span className="badge machine">
            MACHINE LEAD: {machineClass ?? "?"}
            {machineDirection != null ? ` · ${machineDirection}` : ""}
          </span>
        </p>
      )}
      <p className="muted" style={{ margin: 0 }}>
        Do not force dolly/track/push/pull wording from 2D evidence — pixel
        motion cannot prove physical camera rigs.
      </p>
      <label className="col">
        <span className="muted">Final camera classification</span>
        <select
          aria-label="Final camera classification"
          value={selected}
          onChange={(event) => setSelected(event.target.value)}
        >
          <option value="">— select —</option>
          {classes.map((value) => (
            <option key={value} value={value}>
              {value}
            </option>
          ))}
        </select>
      </label>
      <label className="col">
        <span className="muted">Reviewer note (optional)</span>
        <textarea
          aria-label="Reviewer note"
          value={note}
          rows={2}
          onChange={(event) => setNote(event.target.value)}
        />
      </label>
      <button
        type="button"
        className="primary"
        disabled={reviewerName === "" || saving || selected === ""}
        onClick={() => void save()}
      >
        SAVE CAMERA CLASSIFICATION
      </button>
      {reviewerName === "" && (
        <p className="muted">Enter your reviewer name to save decisions.</p>
      )}
      {error != null && (
        <p className="badge fail" role="alert">
          {error}
        </p>
      )}
    </div>
  );
}
