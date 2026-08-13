/**
 * On-screen text (OCR) editor.
 *
 * Machine OCR text is a MACHINE LEAD. The human verifies (TEXT_VERIFICATION
 * "verified"), corrects (TEXT_CORRECTION with the exact text), rejects
 * (TEXT_VERIFICATION "rejected"), or corrects timing (TEXT_TIMING with the
 * dash-separated "first-last" frame value the engine parses; the whole range
 * must lie inside one verified shot). Frame timing must be picked in exact
 * frame mode — the parent passes the current exact frame for convenience.
 */

import { useState, type ReactElement } from "react";
import type { DecisionType } from "../../../api/types";
import type { DecisionsStore } from "../decisionsStore";
import { getReviewerName, ReviewerNameField } from "../reviewerName";

export interface OcrEditorProps {
  trackId: string;
  machineText?: string;
  currentFrame?: number;
  store: DecisionsStore;
  onResolved: (outcome: unknown) => void;
}

export function OcrEditor({
  trackId,
  machineText,
  currentFrame,
  store,
  onResolved,
}: OcrEditorProps): ReactElement {
  const [reviewerName, setReviewerNameState] = useState(getReviewerName());
  const nameMissingAtMount = useState(getReviewerName() === "")[0];
  const [note, setNote] = useState("");
  const [correcting, setCorrecting] = useState(false);
  const [correctedText, setCorrectedText] = useState("");
  const [timing, setTiming] = useState(false);
  const [firstFrame, setFirstFrame] = useState<string>("");
  const [lastFrame, setLastFrame] = useState<string>("");
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function save(decisionType: DecisionType, value: string): Promise<void> {
    setError(null);
    setSaving(true);
    try {
      const outcome = await store.applyDecision(
        store.makeDecision({
          subject_id: trackId,
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
  const first = Number.parseInt(firstFrame, 10);
  const last = Number.parseInt(lastFrame, 10);
  const timingValid =
    Number.isInteger(first) && Number.isInteger(last) && first >= 0 && last >= first;

  return (
    <div className="panel col" aria-label={`On-screen text editor for ${trackId}`}>
      <h3 style={{ margin: 0 }}>On-screen text — {trackId}</h3>
      {nameMissingAtMount && <ReviewerNameField onChange={setReviewerNameState} />}

      <section className="col">
        <span className="badge machine">MACHINE LEAD — unverified</span>
        <textarea
          readOnly
          aria-label="Machine OCR text (machine lead, unverified)"
          value={machineText ?? "(no machine text)"}
          rows={2}
        />
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

      <div className="row" style={{ flexWrap: "wrap" }}>
        <button
          type="button"
          className="primary"
          disabled={disabled}
          onClick={() => void save("TEXT_VERIFICATION", "verified")}
        >
          VERIFY TEXT
        </button>
        <button
          type="button"
          disabled={disabled}
          aria-pressed={correcting}
          onClick={() => setCorrecting((prev) => !prev)}
        >
          CORRECT TEXT
        </button>
        <button
          type="button"
          disabled={disabled}
          onClick={() => void save("TEXT_VERIFICATION", "rejected")}
        >
          REJECT TEXT
        </button>
        <button
          type="button"
          disabled={disabled}
          aria-pressed={timing}
          onClick={() => setTiming((prev) => !prev)}
        >
          CORRECT TIMING
        </button>
      </div>

      {correcting && (
        <div className="col">
          <label className="col">
            <span className="muted">Corrected on-screen text (verbatim)</span>
            <textarea
              aria-label="Corrected on-screen text"
              value={correctedText}
              rows={2}
              onChange={(event) => setCorrectedText(event.target.value)}
            />
          </label>
          <button
            type="button"
            className="primary"
            disabled={disabled || correctedText.trim() === ""}
            onClick={() => void save("TEXT_CORRECTION", correctedText.trim())}
          >
            SAVE CORRECTED TEXT
          </button>
        </div>
      )}

      {timing && (
        <div className="col">
          <p className="muted" style={{ margin: 0 }}>
            Pick frames in exact frame mode. The whole range must lie inside one
            verified shot.
          </p>
          <div className="row">
            <label className="col">
              <span className="muted">First stable frame</span>
              <input
                type="number"
                min={0}
                aria-label="First stable frame"
                value={firstFrame}
                onChange={(event) => setFirstFrame(event.target.value)}
              />
            </label>
            {currentFrame != null && (
              <button
                type="button"
                aria-label="Use current frame as first stable frame"
                onClick={() => setFirstFrame(String(currentFrame))}
              >
                Use current frame
              </button>
            )}
          </div>
          <div className="row">
            <label className="col">
              <span className="muted">Last stable frame</span>
              <input
                type="number"
                min={0}
                aria-label="Last stable frame"
                value={lastFrame}
                onChange={(event) => setLastFrame(event.target.value)}
              />
            </label>
            {currentFrame != null && (
              <button
                type="button"
                aria-label="Use current frame as last stable frame"
                onClick={() => setLastFrame(String(currentFrame))}
              >
                Use current frame
              </button>
            )}
          </div>
          <button
            type="button"
            className="primary"
            disabled={disabled || !timingValid}
            onClick={() => void save("TEXT_TIMING", `${first}-${last}`)}
          >
            SAVE TIMING
          </button>
        </div>
      )}

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
