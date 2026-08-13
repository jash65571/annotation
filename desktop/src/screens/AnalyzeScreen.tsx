/**
 * Analyze: pre-flight summary → ANALYZE → structured stage progress →
 * honest cancellation. Advanced options stay collapsed (spec §22–26).
 */

import { useState } from "react";
import { useApp } from "../state/context";
import type { Screen } from "../App";
import { cancelAnalysis, startAnalysis } from "../api/bridge";
import type { StartAuditOptions } from "../api/types";

const STAGE_LABELS: Record<string, string> = {
  preparing: "Preparing",
  media_verification: "Media verification",
  frame_ledger: "Frame ledger",
  frame_extraction: "Frame extraction",
  shot_analysis: "Shot analysis",
  audio_analysis: "Audio analysis (incl. ASR)",
  visual_intelligence: "Visual observations",
  caption_brain: "Caption planning + validation",
  done: "Done",
};

export function AnalyzeScreen({ onNavigate }: { onNavigate: (screen: Screen) => void }) {
  const { state, dispatch } = useApp();
  const [advancedOpen, setAdvancedOpen] = useState(false);
  const [options, setOptions] = useState<StartAuditOptions>({});
  const task = state.task;

  async function analyze() {
    if (!task.videoPath) return;
    dispatch({ type: "ANALYSIS_STARTED" });
    try {
      await startAnalysis({
        video_path: task.videoPath,
        ...(task.seedPath ? { seed_path: task.seedPath } : {}),
        ...(!task.seedPath && task.seedText.trim() ? { seed_text: task.seedText } : {}),
        ...(task.feedbackPath ? { feedback_path: task.feedbackPath } : {}),
        ...(!task.feedbackPath && task.feedbackText.trim()
          ? { feedback_text: task.feedbackText }
          : {}),
        options,
      });
    } catch (error) {
      dispatch({
        type: "ANALYSIS_FAILED",
        code: "ENGINE_CRASH",
        message: String(error),
      });
    }
  }

  async function cancel() {
    await cancelAnalysis();
    dispatch({ type: "ANALYSIS_CANCELLED" });
  }

  const analyzing = state.phase === "ANALYZING";
  const latestByStage = new Map<string, (typeof state.progress)[number]>();
  for (const event of state.progress) latestByStage.set(event.stage, event);

  return (
    <div className="col" style={{ padding: "var(--gap-lg)", maxWidth: 860, margin: "0 auto" }}>
      <h1 style={{ fontSize: 18, margin: 0 }}>Analysis</h1>

      <section className="panel col">
        <h2 style={{ fontSize: 14, margin: 0 }}>Task</h2>
        <dl className="col" style={{ gap: 4, margin: 0 }}>
          <div className="row">
            <dt className="muted" style={{ width: 90 }}>
              Video
            </dt>
            <dd className="mono" style={{ margin: 0 }}>
              {task.videoPath ?? "—"}
            </dd>
          </div>
          <div className="row">
            <dt className="muted" style={{ width: 90 }}>
              Seed
            </dt>
            <dd style={{ margin: 0 }}>
              {task.seedPath ?? (task.seedText.trim() ? "pasted text" : "none")}
            </dd>
          </div>
          <div className="row">
            <dt className="muted" style={{ width: 90 }}>
              Feedback
            </dt>
            <dd style={{ margin: 0 }}>
              {task.feedbackPath ?? (task.feedbackText.trim() ? "pasted text" : "none")}
            </dd>
          </div>
          <div className="row">
            <dt className="muted" style={{ width: 90 }}>
              ASR
            </dt>
            <dd style={{ margin: 0 }}>{options.asr === false ? "off" : "on (local)"}</dd>
          </div>
          <div className="row">
            <dt className="muted" style={{ width: 90 }}>
              OCR
            </dt>
            <dd style={{ margin: 0 }}>{options.ocr === false ? "off" : "on (local)"}</dd>
          </div>
        </dl>

        <details open={advancedOpen} onToggle={(e) => setAdvancedOpen(e.currentTarget.open)}>
          <summary>Advanced</summary>
          <div className="col" style={{ paddingTop: "var(--gap)" }}>
            <label className="row">
              <input
                type="checkbox"
                checked={options.asr !== false}
                onChange={(e) => setOptions({ ...options, asr: e.target.checked })}
              />
              Local ASR (never cloud)
            </label>
            <label className="row">
              ASR model
              <input
                value={options.asr_model ?? "large-v3-turbo"}
                onChange={(e) => setOptions({ ...options, asr_model: e.target.value })}
              />
            </label>
            <label className="row">
              ASR device
              <select
                value={options.asr_device ?? "auto"}
                onChange={(e) => setOptions({ ...options, asr_device: e.target.value })}
              >
                <option value="auto">auto</option>
                <option value="cpu">cpu</option>
                <option value="cuda">cuda</option>
              </select>
            </label>
            <label className="row">
              ASR compute type
              <select
                value={options.asr_compute_type ?? "auto"}
                onChange={(e) => setOptions({ ...options, asr_compute_type: e.target.value })}
              >
                <option value="auto">auto</option>
                <option value="int8">int8</option>
                <option value="float16">float16</option>
              </select>
            </label>
            <label className="row">
              <input
                type="checkbox"
                checked={options.asr_bootstrap !== false}
                onChange={(e) => setOptions({ ...options, asr_bootstrap: e.target.checked })}
              />
              Allow local ASR bootstrap (env/model download)
            </label>
            <label className="row">
              <input
                type="checkbox"
                checked={options.ocr !== false}
                onChange={(e) => setOptions({ ...options, ocr: e.target.checked })}
              />
              OCR text evidence
            </label>
            <label className="row">
              Shot sensitivity
              <select
                value={options.shot_sensitivity ?? "normal"}
                onChange={(e) =>
                  setOptions({
                    ...options,
                    shot_sensitivity: e.target.value as "high" | "normal" | "low",
                  })
                }
              >
                <option value="high">high (max recall)</option>
                <option value="normal">normal</option>
                <option value="low">low</option>
              </select>
            </label>
            <label className="row">
              <input
                type="checkbox"
                checked={options.extract_shot_evidence !== false}
                onChange={(e) =>
                  setOptions({ ...options, extract_shot_evidence: e.target.checked })
                }
              />
              Extract shot evidence bundles
            </label>
            <label className="row">
              <input
                type="checkbox"
                checked={options.extract_frames === true}
                onChange={(e) => setOptions({ ...options, extract_frames: e.target.checked })}
              />
              Extract every frame as evidence PNG
            </label>
          </div>
        </details>

        <div className="row">
          {!analyzing && (
            <button className="primary" disabled={!task.videoPath} onClick={() => void analyze()}>
              Analyze
            </button>
          )}
          {analyzing && (
            <button onClick={() => void cancel()}>Cancel analysis</button>
          )}
          <button onClick={() => onNavigate("home")} disabled={analyzing}>
            Back
          </button>
        </div>
      </section>

      {(analyzing || state.progress.length > 0) && (
        <section className="panel col" aria-live="polite">
          <h2 style={{ fontSize: 14, margin: 0 }}>Progress</h2>
          {[...latestByStage.values()].map((event) => (
            <div key={event.stage} className="row" style={{ justifyContent: "space-between" }}>
              <span>{STAGE_LABELS[event.stage] ?? event.stage}</span>
              <span className="row">
                {event.total != null && (
                  <span className="faint mono">
                    {event.current != null ? `${event.current}/${event.total}` : event.total}
                  </span>
                )}
                <span
                  className={`badge ${
                    event.status === "COMPLETED"
                      ? "pass"
                      : event.status === "FAILED"
                        ? "fail"
                        : "neutral"
                  }`}
                >
                  {event.status}
                </span>
              </span>
            </div>
          ))}
        </section>
      )}

      {state.phase === "ANALYSIS_CANCELLED" && (
        <section className="panel">
          <span className="badge review">CANCELLED</span> Analysis was cancelled. Artifacts
          already written remain on disk; start a new analysis when ready.
        </section>
      )}

      {state.error && (
        <section className="panel col">
          <span className="badge fail">{state.error.code}</span>
          <p style={{ margin: 0 }}>{state.error.message}</p>
          {state.error.detail && (
            <details>
              <summary>Details</summary>
              <pre className="mono" style={{ whiteSpace: "pre-wrap" }}>{state.error.detail}</pre>
            </details>
          )}
        </section>
      )}
    </div>
  );
}
