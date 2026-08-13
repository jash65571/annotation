/**
 * Final review: rendered caption, gate results, the human-only signoff
 * checklist (never pre-checked), and copy/export gated on READY_TO_ENTER
 * (spec §65–69, §117–118). Copy uses write-only clipboard access and the
 * exact ready artifact bytes — no reformatting.
 */

import { useCallback, useEffect, useState } from "react";
import { writeText } from "@tauri-apps/plugin-clipboard-manager";
import { save } from "@tauri-apps/plugin-dialog";
import { useApp } from "../state/context";
import type { Screen } from "../App";
import {
  createFinalSignoff,
  exportCaption,
  finalize,
  getCaptionState,
  getRunSummary,
  openArtifactFolder,
  validateFinalSignoff,
} from "../api/bridge";
import type { CaptionStatePayload, SignoffValidationPayload } from "../api/types";
import { GateBadge } from "../components/GateBadge";

interface ChecklistState {
  golden_example_comparison_complete: boolean;
  platform_semantic_pass_complete: boolean;
  final_adversarial_read_complete: boolean;
  no_known_omissions_confirmed: boolean;
  no_known_hallucinations_confirmed: boolean;
}

const EMPTY_CHECKLIST: ChecklistState = {
  golden_example_comparison_complete: false,
  platform_semantic_pass_complete: false,
  final_adversarial_read_complete: false,
  no_known_omissions_confirmed: false,
  no_known_hallucinations_confirmed: false,
};

const CHECK_LABELS: Record<keyof ChecklistState, string> = {
  golden_example_comparison_complete:
    "I compared the caption against the Golden Examples myself.",
  platform_semantic_pass_complete:
    "I completed the platform-semantic pass on the full caption.",
  final_adversarial_read_complete:
    "I completed a final adversarial read of every sentence.",
  no_known_omissions_confirmed: "I confirm there are no known omissions.",
  no_known_hallucinations_confirmed: "I confirm there are no known hallucinations.",
};

export function FinalScreen({ onNavigate }: { onNavigate: (screen: Screen) => void }) {
  const { state, dispatch } = useApp();
  const runDir = state.runDir;
  const [caption, setCaption] = useState<CaptionStatePayload | null>(null);
  const [signoffState, setSignoffState] = useState<SignoffValidationPayload | null>(null);
  const [checklist, setChecklist] = useState<ChecklistState>(EMPTY_CHECKLIST);
  const [reviewer, setReviewer] = useState("");
  const [comments, setComments] = useState("");
  const [busy, setBusy] = useState(false);
  const [copied, setCopied] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const reload = useCallback(async () => {
    if (!runDir) return;
    const [captionState, signoff] = await Promise.all([
      getCaptionState(runDir),
      validateFinalSignoff(runDir),
    ]);
    setCaption(captionState);
    setSignoffState(signoff);
  }, [runDir]);

  useEffect(() => {
    reload().catch((e) => setError(String(e)));
  }, [reload]);

  if (!runDir) {
    return (
      <div className="col" style={{ padding: "var(--gap-lg)" }}>
        <p className="muted">No run loaded.</p>
        <button onClick={() => onNavigate("home")} style={{ alignSelf: "flex-start" }}>
          Home
        </button>
      </div>
    );
  }

  const finalStatus = caption?.final_status;
  const readiness = finalStatus?.readiness ?? "REVIEW_REQUIRED";
  const ready = readiness === "READY_TO_ENTER";
  const blockers = (finalStatus?.blockers as string[] | undefined) ?? [];
  const captionSha = caption?.rendered_caption_sha256 ?? "";
  const manifest = state.runSummary?.manifest as
    | { source_video_sha256?: string; rules_version?: string }
    | undefined;
  const allChecked = Object.values(checklist).every(Boolean);
  const canSign =
    allChecked && reviewer.trim().length > 0 && captionSha.length > 0 && !busy;

  async function signAndFinalize() {
    if (!runDir || !manifest?.source_video_sha256 || !manifest.rules_version) {
      setError("Run manifest is missing identity fields; cannot bind a signoff.");
      return;
    }
    setBusy(true);
    setError(null);
    try {
      await createFinalSignoff(runDir, {
        video_sha256: manifest.source_video_sha256,
        rules_version: manifest.rules_version,
        caption_sha256: captionSha,
        reviewer: reviewer.trim(),
        reviewed_at_utc: new Date().toISOString(),
        ...checklist,
        comments: comments.trim() ? comments.trim() : null,
      });
      dispatch({ type: "FINALIZE_STARTED" });
      const result = await finalize(runDir);
      const summary = await getRunSummary(runDir);
      const newReadiness =
        (result.result as { readiness?: string }).readiness ?? "REVIEW_REQUIRED";
      dispatch({
        type: "READINESS_CHANGED",
        readiness: newReadiness as never,
        summary,
      });
      await reload();
    } catch (e) {
      setError(String(e));
    } finally {
      setBusy(false);
    }
  }

  async function copyReadyCaption() {
    if (!runDir) return;
    setError(null);
    try {
      // Copy exactly what the engine exported — never the React-rendered text.
      const exported = await exportCaption(runDir);
      await writeText(exported.markdown);
      setCopied(true);
      setTimeout(() => setCopied(false), 2500);
    } catch (e) {
      setError(String(e));
    }
  }

  async function saveCaptionAs() {
    if (!runDir) return;
    setError(null);
    try {
      const exported = await exportCaption(runDir);
      const target = await save({
        defaultPath: "ready_to_enter.md",
        filters: [{ name: "Markdown", extensions: ["md"] }],
      });
      if (target) {
        const { invoke } = await import("@tauri-apps/api/core");
        await invoke("save_text_file", { path: target, contents: exported.markdown });
      }
    } catch (e) {
      setError(String(e));
    }
  }

  return (
    <div
      className="col"
      style={{ padding: "var(--gap-lg)", maxWidth: 980, margin: "0 auto", gap: "var(--gap-lg)" }}
    >
      <div className="row" style={{ justifyContent: "space-between" }}>
        <h1 style={{ fontSize: 18, margin: 0 }}>Final review</h1>
        <span className={`badge ${ready ? "pass" : readiness === "BLOCKED" ? "blocked" : "review"}`}>
          {readiness.replaceAll("_", " ")}
        </span>
      </div>

      {error && (
        <section className="panel">
          <span className="badge fail">ERROR</span> {error}
        </section>
      )}

      <section className="panel col">
        <h2 style={{ fontSize: 14, margin: 0 }}>Gates</h2>
        <div className="row" style={{ flexWrap: "wrap" }}>
          <GateBadge
            label="M2 validator"
            failCount={Number(finalStatus?.m2_fail_count ?? 0)}
            reviewCount={Number(finalStatus?.m2_review_count ?? 0)}
          />
          <GateBadge label="Platform semantic" status={String(finalStatus?.platform_semantic_status ?? "NOT_RUN")} />
          <GateBadge label="Golden gate" status={String(finalStatus?.golden_gate_status ?? "NOT_RUN")} />
          <GateBadge
            label="Coverage"
            failCount={Number(finalStatus?.coverage_missing_required ?? 0)}
            reviewCount={0}
          />
          <GateBadge
            label="Assertions"
            failCount={Number(finalStatus?.unmapped_assertions ?? 0)}
            reviewCount={0}
          />
          <GateBadge
            label="Feedback"
            failCount={0}
            reviewCount={Number(finalStatus?.unresolved_feedback_high ?? 0)}
          />
        </div>
        {blockers.length > 0 && (
          <div className="col" style={{ gap: 4 }}>
            <span className="muted">Remaining before ready:</span>
            <ul style={{ margin: 0, paddingLeft: 18 }}>
              {blockers.map((blocker) => (
                <li key={blocker}>{blocker}</li>
              ))}
            </ul>
          </div>
        )}
      </section>

      <section className="panel col">
        <h2 style={{ fontSize: 14, margin: 0 }}>
          Caption{" "}
          {ready ? (
            <span className="badge pass">READY TO ENTER</span>
          ) : (
            <span className="badge review">REVIEW DRAFT — NOT FINAL</span>
          )}
        </h2>
        <pre
          className="mono"
          style={{
            whiteSpace: "pre-wrap",
            margin: 0,
            background: "var(--bg-inset)",
            padding: "var(--gap-lg)",
            borderRadius: "var(--radius)",
            maxHeight: 360,
            overflow: "auto",
          }}
        >
          {ready ? caption?.ready_markdown : caption?.draft_markdown ?? "No caption rendered."}
        </pre>
        <div className="row">
          <button
            className="primary"
            disabled={!ready}
            onClick={() => void copyReadyCaption()}
            aria-label="Copy ready caption"
          >
            {copied ? "Copied" : "Copy ready caption"}
          </button>
          <button disabled={!ready} onClick={() => void saveCaptionAs()}>
            Save caption as…
          </button>
          <button onClick={() => void openArtifactFolder(runDir)}>Open artifact folder</button>
        </div>
        {!ready && (
          <span className="faint">
            Copy is enabled only when the engine reports READY TO ENTER.
          </span>
        )}
      </section>

      <section className="panel col">
        <h2 style={{ fontSize: 14, margin: 0 }}>Final review checklist</h2>
        {signoffState?.present && signoffState.stale && (
          <div>
            <span className="badge fail">SIGNOFF STALE</span>{" "}
            {signoffState.reasons.join("; ")}
          </div>
        )}
        {signoffState?.present && signoffState.valid && (
          <div>
            <span className="badge pass">SIGNOFF VALID</span>
          </div>
        )}
        <p className="muted" style={{ margin: 0 }}>
          These confirmations are yours alone — nothing here is pre-checked, and the engine
          re-verifies the signoff against the exact video, rules and caption content.
        </p>
        {(Object.keys(CHECK_LABELS) as Array<keyof ChecklistState>).map((key) => (
          <label key={key} className="row">
            <input
              type="checkbox"
              checked={checklist[key]}
              onChange={(e) => setChecklist({ ...checklist, [key]: e.target.checked })}
            />
            {CHECK_LABELS[key]}
          </label>
        ))}
        <div className="row">
          <label className="row">
            Reviewer
            <input
              value={reviewer}
              onChange={(e) => setReviewer(e.target.value)}
              placeholder="your name"
              aria-label="Reviewer name"
            />
          </label>
        </div>
        <textarea
          rows={2}
          placeholder="Comments (optional)"
          aria-label="Signoff comments"
          value={comments}
          onChange={(e) => setComments(e.target.value)}
        />
        <div className="row">
          <button className="primary" disabled={!canSign} onClick={() => void signAndFinalize()}>
            {busy ? "Finalizing…" : "Sign off and finalize"}
          </button>
          <button onClick={() => onNavigate("review")}>Back to review</button>
        </div>
      </section>
    </div>
  );
}
