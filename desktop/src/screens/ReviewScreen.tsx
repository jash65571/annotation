/** Review workstation shell — queue, evidence, caption preview (spec §29+).
 * Built out feature-by-feature; this shell owns layout + data loading. */

import { useEffect, useState } from "react";
import { useApp } from "../state/context";
import type { Screen } from "../App";
import { getCaptionState, getReviewQueue } from "../api/bridge";
import type { CaptionStatePayload, ReviewQueuePayload } from "../api/types";

export function ReviewScreen({ onNavigate }: { onNavigate: (screen: Screen) => void }) {
  const { state } = useApp();
  const [queue, setQueue] = useState<ReviewQueuePayload | null>(null);
  const [caption, setCaption] = useState<CaptionStatePayload | null>(null);
  const [error, setError] = useState<string | null>(null);
  const runDir = state.runDir;

  useEffect(() => {
    if (!runDir) return;
    Promise.all([getReviewQueue(runDir), getCaptionState(runDir)])
      .then(([q, c]) => {
        setQueue(q);
        setCaption(c);
      })
      .catch((e) => setError(String(e)));
  }, [runDir]);

  if (!runDir) {
    return (
      <div className="col" style={{ padding: "var(--gap-lg)" }}>
        <p className="muted">No run loaded. Start a new review from Home.</p>
        <button onClick={() => onNavigate("home")} style={{ alignSelf: "flex-start" }}>
          Home
        </button>
      </div>
    );
  }

  const readiness = caption?.final_status?.readiness ?? "REVIEW_REQUIRED";

  return (
    <div className="col" style={{ padding: "var(--gap-lg)", gap: "var(--gap-lg)" }}>
      <div className="row" style={{ justifyContent: "space-between" }}>
        <h1 style={{ fontSize: 18, margin: 0 }}>Review</h1>
        <span
          className={`badge ${
            readiness === "READY_TO_ENTER"
              ? "pass"
              : readiness === "BLOCKED"
                ? "blocked"
                : "review"
          }`}
        >
          {readiness.replaceAll("_", " ")}
        </span>
      </div>
      {error && <span className="badge fail">{error}</span>}
      {queue && (
        <section className="panel col">
          <h2 style={{ fontSize: 14, margin: 0 }}>Review queue</h2>
          <span className="muted">
            {queue.visual_items.length} visual/caption items, {queue.audio_items.length} audio
            items
          </span>
        </section>
      )}
      {caption?.draft_markdown && (
        <section className="panel col">
          <h2 style={{ fontSize: 14, margin: 0 }}>
            Caption draft <span className="badge review">REVIEW DRAFT — NOT FINAL</span>
          </h2>
          <pre className="mono" style={{ whiteSpace: "pre-wrap", margin: 0 }}>
            {caption.draft_markdown}
          </pre>
        </section>
      )}
    </div>
  );
}
