/**
 * Identity mapping editor.
 *
 * Visual similarity alone never proves identity — the machine only pairs
 * lookalike tracks as candidates. The human maps the track to a verified
 * entity id, declares it a different entity ("different"), or leaves it
 * unresolved ("unresolved").
 */

import { useState, type ReactElement } from "react";
import type { DecisionsStore } from "../decisionsStore";
import { getReviewerName, ReviewerNameField } from "../reviewerName";

export interface IdentityEditorProps {
  subjectTrackId: string;
  candidates: Array<{ id: string; label: string }>;
  store: DecisionsStore;
  onResolved: (outcome: unknown) => void;
}

export function IdentityEditor({
  subjectTrackId,
  candidates,
  store,
  onResolved,
}: IdentityEditorProps): ReactElement {
  const [selectedEntity, setSelectedEntity] = useState("");
  const [note, setNote] = useState("");
  const [reviewerName, setReviewerNameState] = useState(getReviewerName());
  const nameMissingAtMount = useState(getReviewerName() === "")[0];
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function save(value: string): Promise<void> {
    setError(null);
    setSaving(true);
    try {
      const outcome = await store.applyDecision(
        store.makeDecision({
          subject_id: subjectTrackId,
          decision_type: "IDENTITY_MAPPING",
          value,
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

  const disabled = reviewerName === "" || saving;

  return (
    <div className="panel col" aria-label={`Identity editor for ${subjectTrackId}`}>
      <h3 style={{ margin: 0 }}>Identity — {subjectTrackId}</h3>
      {nameMissingAtMount && <ReviewerNameField onChange={setReviewerNameState} />}
      <p className="muted" style={{ margin: 0 }}>
        Visual similarity alone never proves identity. Confirm only what the
        media itself proves.
      </p>

      <div className="row">
        <label className="col" style={{ flex: 1 }}>
          <span className="muted">Verified entity</span>
          <select
            aria-label="Verified entity"
            value={selectedEntity}
            onChange={(event) => setSelectedEntity(event.target.value)}
          >
            <option value="">— select entity —</option>
            {candidates.map((candidate) => (
              <option key={candidate.id} value={candidate.id}>
                {candidate.label} ({candidate.id})
              </option>
            ))}
          </select>
        </label>
        <button
          type="button"
          className="primary"
          disabled={disabled || selectedEntity === ""}
          onClick={() => void save(selectedEntity)}
        >
          SAME IDENTITY
        </button>
      </div>

      <div className="row">
        <button type="button" disabled={disabled} onClick={() => void save("different")}>
          DIFFERENT IDENTITY
        </button>
        <button type="button" disabled={disabled} onClick={() => void save("unresolved")}>
          UNRESOLVED
        </button>
      </div>

      <label className="col">
        <span className="muted">Reviewer note (optional)</span>
        <textarea
          aria-label="Reviewer note"
          value={note}
          rows={2}
          onChange={(event) => setNote(event.target.value)}
        />
      </label>

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
