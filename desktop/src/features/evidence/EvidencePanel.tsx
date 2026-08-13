/**
 * Evidence panel for the selected queue entry.
 *
 * Every evidence reference is rendered exactly as the engine wrote it: exact
 * "num/den" rational times are shown verbatim in a tooltip with a decimal
 * display next to them — never re-derived, never rounded into new "truth".
 * Machine output is always labeled MACHINE LEAD, never "Detected"/"Verified".
 */

import type { ReactElement } from "react";
import type {
  AudioReviewItem,
  EvidenceReference,
  HumanReviewDecision,
  VisualReviewItem,
} from "../../api/types";
import type { QueueEntry } from "../review/QueuePanel";

export interface EvidencePanelProps {
  entry: QueueEntry | null;
  runDir: string;
  onShowFrame: (frameIndex: number) => void;
  onShowBundle: (bundleDir: string) => void;
  onPlayClip: (itemId: string) => void;
  currentDecision?: HumanReviewDecision | null;
}

const FRAME_TYPES = new Set(["FRAME", "FRAME_RANGE", "FRAME_STRIP", "CROP", "FRAME_OBSERVATION"]);
const AUDIO_TYPES = new Set([
  "AUDIO_RANGE",
  "WAVEFORM_RANGE",
  "SPECTROGRAM_RANGE",
  "ASR_SEGMENT",
]);

/** Decimal display for an exact "num/den" string; the raw string stays the truth. */
function exactToDecimal(exact: string | null | undefined): string | null {
  if (exact == null) return null;
  const match = /^(-?\d+)\/(\d+)$/.exec(exact.trim());
  if (match) {
    const num = Number(match[1]);
    const den = Number(match[2]);
    if (Number.isFinite(num) && Number.isFinite(den) && den !== 0) {
      return (num / den).toFixed(3);
    }
    return null;
  }
  const plain = Number(exact);
  return Number.isFinite(plain) ? plain.toFixed(3) : null;
}

function ExactTime({ label, exact }: { label: string; exact: string | null | undefined }): ReactElement | null {
  const decimal = exactToDecimal(exact);
  if (exact == null || decimal == null) return null;
  return (
    <span className="mono muted" title={`Exact: ${exact}`}>
      {label} {decimal}s
    </span>
  );
}

function EvidenceCard({
  reference,
  bundleDir,
  entryId,
  onShowFrame,
  onShowBundle,
  onPlayClip,
}: {
  reference: EvidenceReference;
  bundleDir: string | null;
  entryId: string;
  onShowFrame: (frameIndex: number) => void;
  onShowBundle: (bundleDir: string) => void;
  onPlayClip: (itemId: string) => void;
}): ReactElement {
  let action: (() => void) | null = null;
  let actionHint = "";
  if (FRAME_TYPES.has(reference.evidence_type) && reference.start_frame != null) {
    const frame = reference.start_frame;
    action = () => onShowFrame(frame);
    actionHint = `Show exact frame ${frame}`;
  } else if (AUDIO_TYPES.has(reference.evidence_type)) {
    action = () => onPlayClip(entryId);
    actionHint = "Play review clip";
  } else if (bundleDir != null && bundleDir !== "") {
    action = () => onShowBundle(bundleDir);
    actionHint = "Open evidence bundle";
  }

  const body = (
    <span className="col" style={{ alignItems: "flex-start", gap: 2 }}>
      <span className="row">
        <span className="badge machine">{reference.evidence_type}</span>
        <span className="mono">{reference.evidence_id}</span>
      </span>
      <span className="row">
        {reference.start_frame != null && (
          <span className="mono muted">
            frames {reference.start_frame}
            {reference.end_frame != null ? `–${reference.end_frame}` : ""}
          </span>
        )}
        <ExactTime label="start" exact={reference.start_time_seconds} />
        <ExactTime label="end" exact={reference.end_time_seconds} />
      </span>
      {reference.notes != null && reference.notes !== "" && (
        <span className="muted">{reference.notes}</span>
      )}
      {actionHint !== "" && <span className="faint">{actionHint}</span>}
    </span>
  );

  if (action == null) {
    return (
      <div className="panel" role="listitem">
        {body}
      </div>
    );
  }
  return (
    <button
      type="button"
      className="panel"
      style={{ textAlign: "left" }}
      aria-label={`Evidence ${reference.evidence_id}: ${actionHint}`}
      onClick={action}
    >
      {body}
    </button>
  );
}

function EvidenceList({
  heading,
  references,
  bundleDir,
  entryId,
  onShowFrame,
  onShowBundle,
  onPlayClip,
}: {
  heading: string;
  references: EvidenceReference[];
  bundleDir: string | null;
  entryId: string;
  onShowFrame: (frameIndex: number) => void;
  onShowBundle: (bundleDir: string) => void;
  onPlayClip: (itemId: string) => void;
}): ReactElement {
  return (
    <section className="col">
      <h3 style={{ margin: 0 }}>{heading}</h3>
      {references.length === 0 && <p className="faint">None recorded.</p>}
      <div className="col" role="list">
        {references.map((reference) => (
          <EvidenceCard
            key={reference.evidence_id}
            reference={reference}
            bundleDir={bundleDir}
            entryId={entryId}
            onShowFrame={onShowFrame}
            onShowBundle={onShowBundle}
            onPlayClip={onPlayClip}
          />
        ))}
      </div>
    </section>
  );
}

export function EvidencePanel({
  entry,
  runDir,
  onShowFrame,
  onShowBundle,
  onPlayClip,
  currentDecision,
}: EvidencePanelProps): ReactElement {
  if (entry == null) {
    return (
      <div className="panel col">
        <p className="muted">Select a queue item to inspect its evidence.</p>
        <p className="faint mono">{runDir}</p>
      </div>
    );
  }

  const isVisual = entry.source === "visual";
  const visual = isVisual ? (entry.item as VisualReviewItem) : null;
  const audio = isVisual ? null : (entry.item as AudioReviewItem);
  const bundleDir = visual?.evidence_bundle_dir ?? null;
  const supporting = visual?.supporting_evidence_refs ?? [];
  const contradicting = visual?.contradicting_evidence_refs ?? [];
  const critical = entry.priority === "CRITICAL" || entry.priority === "HIGH";

  return (
    <div className="panel col" aria-label={`Evidence for ${entry.title}`}>
      <h2 style={{ margin: 0 }}>{entry.title}</h2>

      <section className="col">
        <h3 style={{ margin: 0 }}>Why this needs review</h3>
        <p style={{ margin: 0 }}>{entry.reason}</p>
        <div className="row">
          <span className={`badge ${critical ? "fail" : "neutral"}`}>{entry.priority}</span>
          <ExactTime
            label="start"
            exact={(entry.item as { start_exact?: string | null }).start_exact}
          />
          <ExactTime
            label="end"
            exact={(entry.item as { end_exact?: string | null }).end_exact}
          />
        </div>
      </section>

      {isVisual ? (
        <>
          <EvidenceList
            heading="Supporting evidence"
            references={supporting}
            bundleDir={bundleDir}
            entryId={entry.id}
            onShowFrame={onShowFrame}
            onShowBundle={onShowBundle}
            onPlayClip={onPlayClip}
          />
          <EvidenceList
            heading="Contradicting evidence"
            references={contradicting}
            bundleDir={bundleDir}
            entryId={entry.id}
            onShowFrame={onShowFrame}
            onShowBundle={onShowBundle}
            onPlayClip={onPlayClip}
          />
          {bundleDir != null && bundleDir !== "" && (
            <button
              type="button"
              aria-label="Open evidence bundle"
              onClick={() => onShowBundle(bundleDir)}
            >
              Open evidence bundle
            </button>
          )}
        </>
      ) : (
        <section className="col">
          <h3 style={{ margin: 0 }}>Supporting evidence</h3>
          {(audio?.evidence_refs ?? []).length === 0 && (
            <p className="faint">None recorded.</p>
          )}
          <div className="row" style={{ flexWrap: "wrap" }}>
            {(audio?.evidence_refs ?? []).map((refId) => (
              <span key={refId} className="badge machine mono">
                {refId}
              </span>
            ))}
          </div>
          <button
            type="button"
            aria-label="Play audio review clip"
            onClick={() => onPlayClip(entry.id)}
          >
            Play review clip
          </button>
        </section>
      )}

      <section className="col">
        <h3 style={{ margin: 0 }}>Machine proposal</h3>
        {isVisual ? (
          <p style={{ margin: 0 }}>
            <span className="badge machine">MACHINE LEAD</span>{" "}
            Recommended action: <span className="mono">{visual?.recommended_action}</span>
          </p>
        ) : (
          <p style={{ margin: 0 }}>
            <span className="badge machine">MACHINE LEAD — unverified</span>{" "}
            <span className="mono">
              {audio?.asr_text_candidate ?? "(no ASR candidate)"}
            </span>
          </p>
        )}
      </section>

      <section className="col">
        <h3 style={{ margin: 0 }}>Current human decision</h3>
        {currentDecision == null ? (
          <p className="faint" style={{ margin: 0 }}>
            No human decision recorded for this item yet.
          </p>
        ) : (
          <p style={{ margin: 0 }}>
            <span className="badge human">{currentDecision.decision_type}</span>{" "}
            <span className="mono">{currentDecision.value}</span>{" "}
            <span className="muted">by {currentDecision.decided_by}</span>
          </p>
        )}
      </section>

      {critical && (
        <p className="muted" style={{ margin: 0 }}>
          Effect on caption readiness: while this {entry.priority} item is
          unresolved, the caption stays in REVIEW REQUIRED.
        </p>
      )}
    </div>
  );
}
