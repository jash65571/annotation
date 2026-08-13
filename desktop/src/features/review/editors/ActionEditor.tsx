/**
 * Action editor: boundary and semantics are two separate decisions.
 *
 * ACTION_BOUNDARY value is the dash-separated inclusive frame range
 * "start-end" the engine parses (value.split("-", 1)); frames must stay
 * inside the owning shot's inclusive range — the engine validates this.
 * ACTION_SEMANTICS is the human wording; wording never changes the boundary.
 */

import { useState, type ReactElement } from "react";
import type { DecisionType } from "../../../api/types";
import type { DecisionsStore } from "../decisionsStore";
import { getReviewerName, ReviewerNameField } from "../reviewerName";

export interface ActionEditorProps {
  candidateId: string;
  startFrame: number;
  endFrame: number;
  currentFrame: number;
  machineClass?: string;
  store: DecisionsStore;
  onResolved: (outcome: unknown) => void;
}

export function ActionEditor({
  candidateId,
  startFrame,
  endFrame,
  currentFrame,
  machineClass,
  store,
  onResolved,
}: ActionEditorProps): ReactElement {
  const [start, setStart] = useState<string>(String(startFrame));
  const [end, setEnd] = useState<string>(String(endFrame));
  const [sentence, setSentence] = useState("");
  const [note, setNote] = useState("");
  const [reviewerName, setReviewerNameState] = useState(getReviewerName());
  const nameMissingAtMount = useState(getReviewerName() === "")[0];
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function save(decisionType: DecisionType, value: string): Promise<void> {
    setError(null);
    setSaving(true);
    try {
      const outcome = await store.applyDecision(
        store.makeDecision({
          subject_id: candidateId,
          decision_type: decisionType,
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
  const startNum = Number.parseInt(start, 10);
  const endNum = Number.parseInt(end, 10);
  const boundaryValid =
    Number.isInteger(startNum) && Number.isInteger(endNum) && startNum >= 0 && endNum >= startNum;

  return (
    <div className="panel col" aria-label={`Action editor for ${candidateId}`}>
      <h3 style={{ margin: 0 }}>Action — {candidateId}</h3>
      {nameMissingAtMount && <ReviewerNameField onChange={setReviewerNameState} />}
      {machineClass != null && (
        <p style={{ margin: 0 }}>
          <span className="badge machine">MACHINE LEAD: {machineClass}</span>
        </p>
      )}

      <section className="col">
        <h4 style={{ margin: 0 }}>Boundary (inclusive frames)</h4>
        <p className="muted" style={{ margin: 0 }}>
          Frames must stay inside the owning shot's inclusive frame range.
        </p>
        <div className="row">
          <label className="col">
            <span className="muted">Start frame</span>
            <input
              type="number"
              min={0}
              aria-label="Start frame"
              value={start}
              onChange={(event) => setStart(event.target.value)}
            />
          </label>
          <button
            type="button"
            aria-label="Use current exact frame as start frame"
            onClick={() => setStart(String(currentFrame))}
          >
            Use current exact frame
          </button>
        </div>
        <div className="row">
          <label className="col">
            <span className="muted">End frame</span>
            <input
              type="number"
              min={0}
              aria-label="End frame"
              value={end}
              onChange={(event) => setEnd(event.target.value)}
            />
          </label>
          <button
            type="button"
            aria-label="Use current exact frame as end frame"
            onClick={() => setEnd(String(currentFrame))}
          >
            Use current exact frame
          </button>
        </div>
        <button
          type="button"
          className="primary"
          disabled={disabled || !boundaryValid}
          onClick={() => void save("ACTION_BOUNDARY", `${startNum}-${endNum}`)}
        >
          SAVE BOUNDARY
        </button>
      </section>

      <section className="col">
        <h4 style={{ margin: 0 }}>Semantics (separate decision)</h4>
        <p className="muted" style={{ margin: 0 }}>
          Wording never changes the boundary — boundary and semantics save
          separately.
        </p>
        <label className="col">
          <span className="muted">Human action sentence</span>
          <input
            type="text"
            aria-label="Human action sentence"
            placeholder='e.g. "C1 places the right hand on O1."'
            value={sentence}
            onChange={(event) => setSentence(event.target.value)}
          />
        </label>
        <button
          type="button"
          className="primary"
          disabled={disabled || sentence.trim() === ""}
          onClick={() => void save("ACTION_SEMANTICS", sentence.trim())}
        >
          SAVE SEMANTICS
        </button>
      </section>

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
