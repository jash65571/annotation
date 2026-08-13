/**
 * Playback speed editor.
 *
 * Machine speed evidence is only ever a candidate (e.g. REGULAR_CANDIDATE) —
 * it never becomes a final fact without a human PLAYBACK_SPEED decision.
 * Nothing is pre-selected; the reviewer must choose explicitly.
 */

import { useState, type ReactElement } from "react";
import type { DecisionsStore } from "../decisionsStore";
import { getReviewerName, ReviewerNameField } from "../reviewerName";

export interface SpeedEditorProps {
  shotNumber: number;
  subjectId: string;
  machineEvidence?: string;
  store: DecisionsStore;
  onResolved: (outcome: unknown) => void;
}

/** Engine vocabulary for PLAYBACK_SPEED (review/decisions.py). */
const SPEED_OPTIONS: Array<{ value: "regular" | "slow_motion" | "accelerated"; label: string }> = [
  { value: "regular", label: "Regular" },
  { value: "slow_motion", label: "Slow motion" },
  { value: "accelerated", label: "Accelerated" },
];

export function SpeedEditor({
  shotNumber,
  subjectId,
  machineEvidence,
  store,
  onResolved,
}: SpeedEditorProps): ReactElement {
  const [selected, setSelected] = useState<"" | "regular" | "slow_motion" | "accelerated">("");
  const [note, setNote] = useState("");
  const [reviewerName, setReviewerNameState] = useState(getReviewerName());
  const nameMissingAtMount = useState(getReviewerName() === "")[0];
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function save(): Promise<void> {
    if (selected === "") return;
    setError(null);
    setSaving(true);
    try {
      const outcome = await store.applyDecision(
        store.makeDecision({
          subject_id: subjectId,
          decision_type: "PLAYBACK_SPEED",
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
    <div className="panel col" aria-label={`Playback speed editor for shot ${shotNumber}`}>
      <h3 style={{ margin: 0 }}>Playback speed — shot {shotNumber}</h3>
      {nameMissingAtMount && <ReviewerNameField onChange={setReviewerNameState} />}
      {machineEvidence != null && (
        <p style={{ margin: 0 }}>
          <span className="badge machine">
            {machineEvidence} — machine evidence, never final
          </span>
        </p>
      )}
      <p className="muted" style={{ margin: 0 }}>
        Required human verification: playback speed is only a fact after your
        explicit decision.
      </p>
      <fieldset className="col" style={{ border: "none", margin: 0, padding: 0 }}>
        <legend className="muted">Playback speed</legend>
        {SPEED_OPTIONS.map((option) => (
          <label key={option.value} className="row">
            <input
              type="radio"
              name={`playback-speed-${subjectId}`}
              value={option.value}
              checked={selected === option.value}
              aria-label={option.label}
              onChange={() => setSelected(option.value)}
            />
            <span>{option.label}</span>
          </label>
        ))}
      </fieldset>
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
        SAVE PLAYBACK SPEED
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
