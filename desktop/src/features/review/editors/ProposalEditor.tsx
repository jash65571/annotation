/**
 * Review proposal outcome editor.
 *
 * The human decides the fate of a machine proposal: KEEP, FIX_ENRICH, or
 * REDO_REBUILD (REVIEW_PROPOSAL_OUTCOME vocabulary). Nothing pre-selected.
 */

import { useState, type ReactElement } from "react";
import type { DecisionsStore } from "../decisionsStore";
import { getReviewerName, ReviewerNameField } from "../reviewerName";

export interface ProposalEditorProps {
  proposalId: string;
  proposalKind: string;
  reasonCodes: string[];
  store: DecisionsStore;
  onResolved: (outcome: unknown) => void;
}

const OUTCOMES: Array<{ value: "KEEP" | "FIX_ENRICH" | "REDO_REBUILD"; label: string }> = [
  { value: "KEEP", label: "KEEP — the proposal stands as-is" },
  { value: "FIX_ENRICH", label: "FIX_ENRICH — keep but fix or enrich it" },
  { value: "REDO_REBUILD", label: "REDO_REBUILD — rebuild this section" },
];

export function ProposalEditor({
  proposalId,
  proposalKind,
  reasonCodes,
  store,
  onResolved,
}: ProposalEditorProps): ReactElement {
  const [selected, setSelected] = useState<"" | "KEEP" | "FIX_ENRICH" | "REDO_REBUILD">("");
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
          subject_id: proposalId,
          decision_type: "REVIEW_PROPOSAL_OUTCOME",
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
    <div className="panel col" aria-label={`Proposal editor for ${proposalId}`}>
      <h3 style={{ margin: 0 }}>
        Proposal — {proposalId} <span className="muted">({proposalKind})</span>
      </h3>
      {nameMissingAtMount && <ReviewerNameField onChange={setReviewerNameState} />}
      {reasonCodes.length > 0 && (
        <div className="row" style={{ flexWrap: "wrap" }}>
          {reasonCodes.map((code) => (
            <span key={code} className="badge unresolved mono">
              {code}
            </span>
          ))}
        </div>
      )}
      <fieldset className="col" style={{ border: "none", margin: 0, padding: 0 }}>
        <legend className="muted">Outcome</legend>
        {OUTCOMES.map((outcome) => (
          <label key={outcome.value} className="row">
            <input
              type="radio"
              name={`proposal-outcome-${proposalId}`}
              value={outcome.value}
              checked={selected === outcome.value}
              aria-label={outcome.value}
              onChange={() => setSelected(outcome.value)}
            />
            <span>{outcome.label}</span>
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
        SAVE OUTCOME
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
