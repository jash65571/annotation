/**
 * The explicit UI state machine (spec §79). State is never inferred from
 * boolean soup; every transition is a typed action. Caption readiness values
 * come verbatim from the engine and are never invented client-side.
 */

import type {
  CaptionReadiness,
  ProgressEventPayload,
  RunSummaryPayload,
} from "../api/types";

export type AppPhase =
  | "IDLE"
  | "TASK_READY"
  | "ANALYZING"
  | "ANALYSIS_CANCELLED"
  | "REVIEWING"
  | "FINALIZING"
  | "READY_FOR_FINAL_REVIEW"
  | "READY_TO_ENTER"
  | "BLOCKED"
  | "ERROR";

export interface TaskInputs {
  videoPath: string | null;
  seedPath: string | null;
  seedText: string;
  feedbackPath: string | null;
  feedbackText: string;
}

export interface AppState {
  phase: AppPhase;
  task: TaskInputs;
  runDir: string | null;
  runSummary: RunSummaryPayload | null;
  progress: ProgressEventPayload[];
  error: { code: string; message: string; detail?: string | undefined } | null;
}

export const initialState: AppState = {
  phase: "IDLE",
  task: {
    videoPath: null,
    seedPath: null,
    seedText: "",
    feedbackPath: null,
    feedbackText: "",
  },
  runDir: null,
  runSummary: null,
  progress: [],
  error: null,
};

export type AppAction =
  | { type: "TASK_INPUT_CHANGED"; task: Partial<TaskInputs> }
  | { type: "ANALYSIS_STARTED" }
  | { type: "ANALYSIS_PROGRESS"; event: ProgressEventPayload }
  | { type: "ANALYSIS_CANCELLED" }
  | { type: "ANALYSIS_FAILED"; code: string; message: string; detail?: string }
  | { type: "RUN_LOADED"; runDir: string; summary: RunSummaryPayload }
  | { type: "FINALIZE_STARTED" }
  | { type: "READINESS_CHANGED"; readiness: CaptionReadiness; summary?: RunSummaryPayload }
  | { type: "ERROR"; code: string; message: string; detail?: string }
  | { type: "RESET" };

function hasVideo(task: TaskInputs): boolean {
  return task.videoPath !== null && task.videoPath.length > 0;
}

export function phaseForReadiness(readiness: CaptionReadiness): AppPhase {
  switch (readiness) {
    case "BLOCKED":
      return "BLOCKED";
    case "REVIEW_REQUIRED":
      return "REVIEWING";
    case "READY_FOR_FINAL_REVIEW":
      return "READY_FOR_FINAL_REVIEW";
    case "READY_TO_ENTER":
      return "READY_TO_ENTER";
  }
}

export function readinessOf(summary: RunSummaryPayload | null): CaptionReadiness | null {
  const readiness = summary?.caption_final_status?.readiness;
  return readiness ?? null;
}

export function appReducer(state: AppState, action: AppAction): AppState {
  switch (action.type) {
    case "TASK_INPUT_CHANGED": {
      const task = { ...state.task, ...action.task };
      const phase: AppPhase =
        state.phase === "IDLE" || state.phase === "TASK_READY"
          ? hasVideo(task)
            ? "TASK_READY"
            : "IDLE"
          : state.phase;
      return { ...state, task, phase };
    }
    case "ANALYSIS_STARTED":
      return { ...state, phase: "ANALYZING", progress: [], error: null };
    case "ANALYSIS_PROGRESS":
      if (state.phase !== "ANALYZING") return state;
      return { ...state, progress: [...state.progress, action.event] };
    case "ANALYSIS_CANCELLED":
      return { ...state, phase: "ANALYSIS_CANCELLED" };
    case "ANALYSIS_FAILED":
      return {
        ...state,
        phase: "ERROR",
        error: { code: action.code, message: action.message, detail: action.detail },
      };
    case "RUN_LOADED": {
      const readiness = readinessOf(action.summary);
      return {
        ...state,
        runDir: action.runDir,
        runSummary: action.summary,
        phase: readiness ? phaseForReadiness(readiness) : "REVIEWING",
        error: null,
      };
    }
    case "FINALIZE_STARTED":
      return { ...state, phase: "FINALIZING" };
    case "READINESS_CHANGED":
      return {
        ...state,
        runSummary: action.summary ?? state.runSummary,
        phase: phaseForReadiness(action.readiness),
      };
    case "ERROR":
      return {
        ...state,
        phase: "ERROR",
        error: { code: action.code, message: action.message, detail: action.detail },
      };
    case "RESET":
      return { ...initialState };
  }
}
