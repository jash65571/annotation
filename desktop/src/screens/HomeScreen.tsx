/**
 * Home: NEW REVIEW intake (video + seed + task feedback) and recent runs.
 * Seed/feedback accept either a file or pasted text (snapshotted immutably
 * into the run by the engine).
 */

import { useEffect, useState } from "react";
import { open } from "@tauri-apps/plugin-dialog";
import { useApp } from "../state/context";
import type { Screen } from "../App";
import {
  forgetRecentRun,
  getRunSummary,
  listRecentRuns,
  registerIntakeVideo,
} from "../api/bridge";
import type { RecentRun } from "../api/types";

export function HomeScreen({ onNavigate }: { onNavigate: (screen: Screen) => void }) {
  const { state, dispatch } = useApp();
  const [recent, setRecent] = useState<RecentRun[]>([]);
  const task = state.task;

  useEffect(() => {
    listRecentRuns().then(setRecent).catch(() => setRecent([]));
  }, []);

  async function pickVideo() {
    const path = await open({
      multiple: false,
      filters: [{ name: "Video", extensions: ["mp4", "mov", "mkv", "webm", "avi"] }],
    });
    if (typeof path === "string") {
      // Register the dialog choice with Rust: only this intake video (or a
      // validated run's source) is ever playable in the webview.
      const canonical = await registerIntakeVideo(path);
      dispatch({ type: "TASK_INPUT_CHANGED", task: { videoPath: canonical } });
    }
  }

  async function pickText(kind: "seed" | "feedback") {
    const path = await open({
      multiple: false,
      filters: [{ name: "Text", extensions: ["txt", "md"] }],
    });
    if (typeof path === "string") {
      dispatch({
        type: "TASK_INPUT_CHANGED",
        task: kind === "seed" ? { seedPath: path } : { feedbackPath: path },
      });
    }
  }

  async function openRun(runDir: string) {
    try {
      const summary = await getRunSummary(runDir);
      dispatch({ type: "RUN_LOADED", runDir, summary });
      onNavigate("review");
    } catch {
      dispatch({
        type: "ERROR",
        code: "RUN_NOT_FOUND",
        message: `The run at ${runDir} could not be opened.`,
      });
    }
  }

  return (
    <div className="col" style={{ padding: "var(--gap-lg)", maxWidth: 860, margin: "0 auto" }}>
      <h1 style={{ fontSize: 18, margin: 0 }}>Manuscript Reviewer Cockpit</h1>

      <section className="panel col" aria-labelledby="new-review-heading">
        <h2 id="new-review-heading" style={{ fontSize: 14, margin: 0 }}>
          New review
        </h2>

        <div className="row">
          <button onClick={pickVideo}>Select video…</button>
          <span className={task.videoPath ? "mono" : "faint"}>
            {task.videoPath ?? "No video selected"}
          </span>
        </div>

        <div className="col">
          <div className="row">
            <button onClick={() => pickText("seed")}>Seed file…</button>
            <span className={task.seedPath ? "mono" : "faint"}>
              {task.seedPath ?? "or paste seed text below"}
            </span>
          </div>
          <textarea
            aria-label="Seed caption text"
            rows={4}
            placeholder="Paste seed caption text (optional if a file is selected)"
            value={task.seedText}
            onChange={(e) =>
              dispatch({ type: "TASK_INPUT_CHANGED", task: { seedText: e.target.value } })
            }
          />
        </div>

        <div className="col">
          <div className="row">
            <button onClick={() => pickText("feedback")}>Feedback file…</button>
            <span className={task.feedbackPath ? "mono" : "faint"}>
              {task.feedbackPath ?? "or paste task feedback below"}
            </span>
          </div>
          <textarea
            aria-label="Task feedback text"
            rows={3}
            placeholder="Paste task-specific evaluator feedback (optional)"
            value={task.feedbackText}
            onChange={(e) =>
              dispatch({ type: "TASK_INPUT_CHANGED", task: { feedbackText: e.target.value } })
            }
          />
        </div>

        <div className="row">
          <button
            className="primary"
            disabled={state.phase !== "TASK_READY"}
            onClick={() => onNavigate("analyze")}
          >
            Continue to analysis
          </button>
          {state.phase !== "TASK_READY" && (
            <span className="faint">Select a video to continue.</span>
          )}
        </div>
      </section>

      <section className="panel col" aria-labelledby="recent-heading">
        <h2 id="recent-heading" style={{ fontSize: 14, margin: 0 }}>
          Recent reviews
        </h2>
        {recent.length === 0 && <span className="faint">No recent reviews.</span>}
        {recent.map((entry) => (
          <div key={entry.run_dir} className="row" style={{ justifyContent: "space-between" }}>
            <div className="col" style={{ gap: 2 }}>
              <span>{entry.video_name}</span>
              <span className="faint mono">{entry.run_dir}</span>
            </div>
            <div className="row">
              <span
                className={`badge ${
                  entry.readiness === "READY_TO_ENTER"
                    ? "pass"
                    : entry.readiness === "INCOMPLETE"
                      ? "fail"
                      : "review"
                }`}
              >
                {entry.readiness === "INCOMPLETE" ? "INCOMPLETE" : entry.display_status}
              </span>
              <button onClick={() => void openRun(entry.run_dir)}>Open</button>
              <button
                onClick={() => {
                  void forgetRecentRun(entry.run_dir).then(() =>
                    setRecent((prev) => prev.filter((r) => r.run_dir !== entry.run_dir)),
                  );
                }}
              >
                Remove
              </button>
            </div>
          </div>
        ))}
      </section>
    </div>
  );
}
