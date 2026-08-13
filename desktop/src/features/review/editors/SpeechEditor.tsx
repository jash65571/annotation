/**
 * Speech verification editor.
 *
 * The ASR text is a MACHINE LEAD — never final wording. The human either
 * verifies it (SPEECH_VERIFICATION "verified"), corrects it verbatim
 * (SPEECH_CORRECTION with the corrected wording as the value), or rejects it
 * (SPEECH_VERIFICATION "rejected" — the exact vocabulary the engine accepts).
 */

import { useState, type ReactElement } from "react";
import type { AudioReviewItem, DecisionType } from "../../../api/types";
import type { DecisionsStore } from "../decisionsStore";
import { getReviewerName, ReviewerNameField } from "../reviewerName";

export interface SpeechEditorProps {
  item: AudioReviewItem;
  store: DecisionsStore;
  onResolved: (outcome: unknown) => void;
}

export function SpeechEditor({ item, store, onResolved }: SpeechEditorProps): ReactElement {
  // The decision subject is the SPEECH REGION id ("speech_NNNN"), carried in
  // the review item's evidence_refs — item_id itself is the queue row id
  // ("areview_NNNN") and is never a valid decision subject
  // (engine audio/review_queue.py + review/decisions.py speech_regions registry).
  const subjectId =
    (item.evidence_refs ?? []).find((ref) => ref.startsWith("speech_")) ?? item.item_id;

  const [reviewerName, setReviewerNameState] = useState(getReviewerName());
  const nameMissingAtMount = useState(getReviewerName() === "")[0];
  const [note, setNote] = useState("");
  const [correcting, setCorrecting] = useState(false);
  const [correctedText, setCorrectedText] = useState("");
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function save(decisionType: DecisionType, value: string): Promise<void> {
    setError(null);
    setSaving(true);
    try {
      const outcome = await store.applyDecision(
        store.makeDecision({
          subject_id: subjectId,
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

  return (
    <div className="panel col" aria-label={`Speech editor for ${subjectId}`}>
      <h3 style={{ margin: 0 }}>Speech review — {subjectId}</h3>
      {nameMissingAtMount && <ReviewerNameField onChange={setReviewerNameState} />}

      <section className="col">
        <span className="badge machine">MACHINE LEAD — unverified</span>
        <textarea
          readOnly
          aria-label="ASR text candidate (machine lead, unverified)"
          value={item.asr_text_candidate ?? "(no ASR candidate)"}
          rows={3}
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

      <div className="row">
        <button
          type="button"
          className="primary"
          disabled={disabled}
          onClick={() => void save("SPEECH_VERIFICATION", "verified")}
        >
          VERIFY WORDING
        </button>
        <button
          type="button"
          disabled={disabled}
          aria-pressed={correcting}
          onClick={() => setCorrecting((prev) => !prev)}
        >
          CORRECT WORDING
        </button>
        <button
          type="button"
          disabled={disabled}
          onClick={() => void save("SPEECH_VERIFICATION", "rejected")}
        >
          REJECT ASR
        </button>
      </div>

      {correcting && (
        <div className="col">
          <label className="col">
            <span className="muted">Verified source wording</span>
            <textarea
              aria-label="Verified source wording"
              value={correctedText}
              rows={3}
              onChange={(event) => setCorrectedText(event.target.value)}
            />
          </label>
          <button
            type="button"
            className="primary"
            disabled={disabled || correctedText.trim() === ""}
            onClick={() => void save("SPEECH_CORRECTION", correctedText.trim())}
          >
            SAVE CORRECTED WORDING
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
