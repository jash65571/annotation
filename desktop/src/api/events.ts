/** Tauri event subscriptions from the Rust backend. */

import { listen, type UnlistenFn } from "@tauri-apps/api/event";
import type { ProgressEventPayload } from "./types";

export const ANALYSIS_PROGRESS_EVENT = "analysis://progress";
export const ANALYSIS_DONE_EVENT = "analysis://done";
export const ANALYSIS_ERROR_EVENT = "analysis://error";

export interface AnalysisDonePayload {
  run_id: string;
  run_dir: string;
  status: string;
}

export function onAnalysisProgress(
  handler: (payload: ProgressEventPayload) => void,
): Promise<UnlistenFn> {
  return listen<ProgressEventPayload>(ANALYSIS_PROGRESS_EVENT, (event) =>
    handler(event.payload),
  );
}

export function onAnalysisDone(
  handler: (payload: AnalysisDonePayload) => void,
): Promise<UnlistenFn> {
  return listen<AnalysisDonePayload>(ANALYSIS_DONE_EVENT, (event) => handler(event.payload));
}

export function onAnalysisError(
  handler: (payload: { code: string; message: string; detail?: string }) => void,
): Promise<UnlistenFn> {
  return listen<{ code: string; message: string; detail?: string }>(
    ANALYSIS_ERROR_EVENT,
    (event) => handler(event.payload),
  );
}
