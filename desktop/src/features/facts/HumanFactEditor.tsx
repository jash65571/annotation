/**
 * ADD VERIFIED FACT panel.
 *
 * A human-added caption fact carries HUMAN_VERIFICATION evidence created at
 * the exact frame(s) the reviewer inspected, and is bound to the run's video
 * SHA-256 and rules version through the DecisionsStore. Save stays disabled
 * until fact type, text, and at least one evidence reference exist.
 */

import { useState, type ReactElement } from "react";
import type { EvidenceReference, HumanCaptionFact } from "../../api/types";
import type { DecisionsStore } from "../review/decisionsStore";
import { getReviewerName, ReviewerNameField } from "../review/reviewerName";

export interface HumanFactEditorProps {
  runDir: string;
  store: DecisionsStore;
  shots: Array<{ shot_number: number; start_frame: number; end_frame: number }>;
  currentFrame: number;
  onSaved: () => void;
  onCancel: () => void;
}

/** CaptionFactType vocabulary (models/caption_brain.py). */
const CAPTION_FACT_TYPES: string[] = [
  "MEDIA",
  "CHARACTER",
  "OBJECT",
  "SCENE",
  "STYLE",
  "OVERVIEW_AUDIO",
  "VISUAL_CONCERN",
  "AUDIO_CONCERN",
  "SHOT_BOUNDARY",
  "TRANSITION",
  "CAMERA_FRAMING",
  "CAMERA_MOVEMENT",
  "VISUAL_ACTION",
  "SPEECH",
  "SOUND",
  "ON_SCREEN_TEXT",
  "PLAYBACK_SPEED",
  "SPEED_CHANGE",
  "FINAL_OBJECT_STATE",
];

let evidenceCounter = 0;

function parseIdList(raw: string): string[] {
  return raw
    .split(",")
    .map((part) => part.trim())
    .filter((part) => part !== "");
}

export function HumanFactEditor({
  runDir,
  store,
  shots,
  currentFrame,
  onSaved,
  onCancel,
}: HumanFactEditorProps): ReactElement {
  const [factType, setFactType] = useState("");
  const [shotNumber, setShotNumber] = useState<string>("");
  const [textValue, setTextValue] = useState("");
  const [evidence, setEvidence] = useState<EvidenceReference[]>([]);
  const [rangeStart, setRangeStart] = useState<string>("");
  const [rangeEnd, setRangeEnd] = useState<string>("");
  const [characterIdsRaw, setCharacterIdsRaw] = useState("");
  const [objectIdsRaw, setObjectIdsRaw] = useState("");
  const [reviewerName, setReviewerNameState] = useState(getReviewerName());
  const nameMissingAtMount = useState(getReviewerName() === "")[0];
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);

  function makeHumanEvidence(startFrame: number, endFrame: number): EvidenceReference {
    return {
      evidence_id: `HV-${Date.now()}-${(evidenceCounter += 1)}`,
      evidence_type: "HUMAN_VERIFICATION",
      start_frame: startFrame,
      end_frame: endFrame,
      source: getReviewerName(),
    };
  }

  function addCurrentFrame(): void {
    setEvidence((prev) => [...prev, makeHumanEvidence(currentFrame, currentFrame)]);
  }

  function addFrameRange(): void {
    const start = Number.parseInt(rangeStart, 10);
    const end = Number.parseInt(rangeEnd, 10);
    if (!Number.isInteger(start) || !Number.isInteger(end) || start < 0 || end < start) return;
    setEvidence((prev) => [...prev, makeHumanEvidence(start, end)]);
  }

  const canSave =
    factType !== "" && textValue.trim() !== "" && evidence.length > 0 && reviewerName !== "";

  async function save(): Promise<void> {
    if (!canSave) return;
    setError(null);
    setSaving(true);
    try {
      const characterIds = parseIdList(characterIdsRaw);
      const objectIds = parseIdList(objectIdsRaw);
      const parsedShot = shotNumber === "" ? null : Number.parseInt(shotNumber, 10);
      const fact: HumanCaptionFact = {
        fact_id: `HF-${Date.now()}`,
        fact_type: factType,
        text_value: textValue.trim(),
        ...(parsedShot != null && Number.isInteger(parsedShot)
          ? { shot_number: parsedShot }
          : {}),
        ...(characterIds.length > 0 ? { character_ids: characterIds } : {}),
        ...(objectIds.length > 0 ? { object_ids: objectIds } : {}),
        evidence_refs: evidence,
        decided_by: reviewerName,
        decided_at_utc: new Date().toISOString(),
        bound_video_sha256: store.videoSha256,
        bound_rules_version: store.rulesVersion,
      };
      await store.applyFact(fact);
      onSaved();
    } catch (err) {
      setError(String(err));
    } finally {
      setSaving(false);
    }
  }

  return (
    <div className="panel col" role="dialog" aria-label="Add verified fact">
      <h3 style={{ margin: 0 }}>ADD VERIFIED FACT</h3>
      <p className="faint mono" style={{ margin: 0 }}>
        Run: {runDir}
      </p>
      {nameMissingAtMount && <ReviewerNameField onChange={setReviewerNameState} />}

      <label className="col">
        <span className="muted">Fact type (required)</span>
        <select
          aria-label="Fact type"
          value={factType}
          onChange={(event) => setFactType(event.target.value)}
        >
          <option value="">— select fact type —</option>
          {CAPTION_FACT_TYPES.map((value) => (
            <option key={value} value={value}>
              {value}
            </option>
          ))}
        </select>
      </label>

      <label className="col">
        <span className="muted">Shot (when applicable)</span>
        <select
          aria-label="Shot"
          value={shotNumber}
          onChange={(event) => setShotNumber(event.target.value)}
        >
          <option value="">— no specific shot —</option>
          {shots.map((shot) => (
            <option key={shot.shot_number} value={String(shot.shot_number)}>
              Shot {shot.shot_number} (frames {shot.start_frame}–{shot.end_frame})
            </option>
          ))}
        </select>
      </label>

      <label className="col">
        <span className="muted">Fact text (required)</span>
        <textarea
          aria-label="Fact text"
          value={textValue}
          rows={2}
          onChange={(event) => setTextValue(event.target.value)}
        />
      </label>

      <section className="col">
        <h4 style={{ margin: 0 }}>Evidence (required)</h4>
        <div className="row">
          <button type="button" aria-label="Use current frame as evidence" onClick={addCurrentFrame}>
            Use current frame ({currentFrame})
          </button>
        </div>
        <div className="row">
          <label className="col">
            <span className="muted">Range start frame</span>
            <input
              type="number"
              min={0}
              aria-label="Range start frame"
              value={rangeStart}
              onChange={(event) => setRangeStart(event.target.value)}
            />
          </label>
          <label className="col">
            <span className="muted">Range end frame</span>
            <input
              type="number"
              min={0}
              aria-label="Range end frame"
              value={rangeEnd}
              onChange={(event) => setRangeEnd(event.target.value)}
            />
          </label>
          <button type="button" aria-label="Use frame range as evidence" onClick={addFrameRange}>
            Use frame range
          </button>
        </div>
        {evidence.length === 0 ? (
          <p className="faint" style={{ margin: 0 }}>
            No evidence yet — a human fact needs at least one HUMAN_VERIFICATION
            reference.
          </p>
        ) : (
          <ul className="col" style={{ listStyle: "none", margin: 0, padding: 0 }}>
            {evidence.map((reference) => (
              <li key={reference.evidence_id} className="row">
                <span className="badge human">HUMAN_VERIFICATION</span>
                <span className="mono">
                  frames {reference.start_frame}–{reference.end_frame}
                </span>
                <button
                  type="button"
                  aria-label={`Remove evidence ${reference.evidence_id}`}
                  onClick={() =>
                    setEvidence((prev) =>
                      prev.filter((item) => item.evidence_id !== reference.evidence_id),
                    )
                  }
                >
                  Remove
                </button>
              </li>
            ))}
          </ul>
        )}
      </section>

      <div className="row">
        <label className="col" style={{ flex: 1 }}>
          <span className="muted">Character ids (comma-separated)</span>
          <input
            type="text"
            aria-label="Character ids"
            placeholder="C1, C2"
            value={characterIdsRaw}
            onChange={(event) => setCharacterIdsRaw(event.target.value)}
          />
        </label>
        <label className="col" style={{ flex: 1 }}>
          <span className="muted">Object ids (comma-separated)</span>
          <input
            type="text"
            aria-label="Object ids"
            placeholder="O1, O2"
            value={objectIdsRaw}
            onChange={(event) => setObjectIdsRaw(event.target.value)}
          />
        </label>
      </div>

      <div className="row">
        <button
          type="button"
          className="primary"
          disabled={!canSave || saving}
          onClick={() => void save()}
        >
          SAVE VERIFIED FACT
        </button>
        <button type="button" onClick={onCancel}>
          Cancel
        </button>
      </div>

      {reviewerName === "" && (
        <p className="muted">Enter your reviewer name to save facts.</p>
      )}
      {error != null && (
        <p className="badge fail" role="alert">
          {error}
        </p>
      )}
    </div>
  );
}
