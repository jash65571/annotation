import { describe, expect, it } from "vitest";
import {
  appReducer,
  initialState,
  phaseForReadiness,
  type AppState,
} from "../state/appState";
import type { RunSummaryPayload } from "../api/types";

function summaryWithReadiness(readiness: string): RunSummaryPayload {
  return {
    run_dir: "C:/runs/x",
    manifest: {},
    qc: null,
    caption_final_status: { readiness: readiness as never },
    audio_qc: null,
    visual_qc: null,
    shot_qc: null,
    has_extracted_frames: false,
    ui_inputs: { review_decisions: false, human_facts: false, final_review: false },
  };
}

describe("app state machine", () => {
  it("starts IDLE and becomes TASK_READY only with a video", () => {
    expect(initialState.phase).toBe("IDLE");
    const withSeedOnly = appReducer(initialState, {
      type: "TASK_INPUT_CHANGED",
      task: { seedText: "seed" },
    });
    expect(withSeedOnly.phase).toBe("IDLE");
    const withVideo = appReducer(withSeedOnly, {
      type: "TASK_INPUT_CHANGED",
      task: { videoPath: "C:/clips/a.mp4" },
    });
    expect(withVideo.phase).toBe("TASK_READY");
  });

  it("maps engine readiness verbatim to phases — never invents readiness", () => {
    expect(phaseForReadiness("BLOCKED")).toBe("BLOCKED");
    expect(phaseForReadiness("REVIEW_REQUIRED")).toBe("REVIEWING");
    expect(phaseForReadiness("READY_FOR_FINAL_REVIEW")).toBe("READY_FOR_FINAL_REVIEW");
    expect(phaseForReadiness("READY_TO_ENTER")).toBe("READY_TO_ENTER");
  });

  it("RUN_LOADED adopts the engine-reported readiness", () => {
    const loaded = appReducer(initialState, {
      type: "RUN_LOADED",
      runDir: "C:/runs/x",
      summary: summaryWithReadiness("READY_FOR_FINAL_REVIEW"),
    });
    expect(loaded.phase).toBe("READY_FOR_FINAL_REVIEW");
    expect(loaded.runDir).toBe("C:/runs/x");
  });

  it("progress events only accumulate while ANALYZING", () => {
    let state: AppState = appReducer(initialState, { type: "ANALYSIS_STARTED" });
    state = appReducer(state, {
      type: "ANALYSIS_PROGRESS",
      event: { stage: "probe", status: "STARTED" },
    });
    expect(state.progress).toHaveLength(1);
    state = appReducer(state, { type: "ANALYSIS_CANCELLED" });
    state = appReducer(state, {
      type: "ANALYSIS_PROGRESS",
      event: { stage: "probe", status: "COMPLETED" },
    });
    expect(state.progress).toHaveLength(1);
    expect(state.phase).toBe("ANALYSIS_CANCELLED");
  });

  it("analysis failure lands in ERROR with a typed code", () => {
    let state = appReducer(initialState, { type: "ANALYSIS_STARTED" });
    state = appReducer(state, {
      type: "ANALYSIS_FAILED",
      code: "FFMPEG_UNAVAILABLE",
      message: "Bundled FFmpeg could not start.",
    });
    expect(state.phase).toBe("ERROR");
    expect(state.error?.code).toBe("FFMPEG_UNAVAILABLE");
  });
});
