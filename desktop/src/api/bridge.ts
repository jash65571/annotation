/**
 * Typed access to the Rust backend, which owns the Python engine worker.
 * React never spawns processes, never touches the filesystem directly, and
 * never parses CLI text — every call below is a typed Tauri command.
 */

import { invoke } from "@tauri-apps/api/core";
import type {
  AuditHistoryEntry,
  BridgeError,
  CaptionStatePayload,
  ExactFramePayload,
  ExportCaptionPayload,
  FinalReviewSignoff,
  HealthPayload,
  HumanCaptionFact,
  HumanReviewDecision,
  Json,
  MediaDimensionsPayload,
  ReviewQueuePayload,
  ReviewResolutionPayload,
  RunSummaryPayload,
  SaveReviewInputsPayload,
  SignoffValidationPayload,
  StartAuditRequest,
  UiStatePayload,
  WaveformMetadataPayload,
} from "./types";

export class BridgeRequestError extends Error {
  readonly bridgeError: BridgeError;

  constructor(error: BridgeError) {
    super(error.message);
    this.name = "BridgeRequestError";
    this.bridgeError = error;
  }
}

function isBridgeError(value: unknown): value is BridgeError {
  return (
    typeof value === "object" &&
    value !== null &&
    typeof (value as BridgeError).code === "string" &&
    typeof (value as BridgeError).message === "string"
  );
}

/** Send one engine command through the Rust-owned query worker. */
async function engineRequest<T>(command: string, payload: Json = {}): Promise<T> {
  try {
    return (await invoke<T>("engine_request", { command, payload })) as T;
  } catch (raw) {
    if (isBridgeError(raw)) throw new BridgeRequestError(raw);
    throw new BridgeRequestError({
      code: "ENGINE_CRASH",
      message: String(raw),
    });
  }
}

// -- health / info ----------------------------------------------------------

export const health = () => engineRequest<HealthPayload>("health");
export const engineInfo = () =>
  engineRequest<{ engine_version: string; rules_version: string; protocol_version: number }>(
    "engine_info",
  );
export const getRules = () =>
  engineRequest<{ rules_version: string; rules: Json }>("get_rules");

// -- analysis ---------------------------------------------------------------

/** Start a full audit as a cancellable background job (dedicated worker
 * process). Progress arrives via the `analysis://progress` Tauri event. */
export async function startAnalysis(request: StartAuditRequest): Promise<string> {
  return invoke<string>("start_analysis", { request });
}

export async function cancelAnalysis(): Promise<void> {
  return invoke<void>("cancel_analysis");
}

// -- run reading ------------------------------------------------------------

export const getRunSummary = (runDir: string) =>
  engineRequest<RunSummaryPayload>("get_run_summary", { run_dir: runDir });

export const getReviewQueue = (runDir: string) =>
  engineRequest<ReviewQueuePayload>("get_review_queue", { run_dir: runDir });

export const getShots = (runDir: string) => engineRequest<Json>("get_shots", { run_dir: runDir });

export const getFrameRecord = (runDir: string, frameIndex: number) =>
  engineRequest<Json>("get_frame_record", { run_dir: runDir, frame_index: frameIndex });

export const getExactFrame = (runDir: string, frameIndex: number) =>
  engineRequest<ExactFramePayload>("get_exact_frame", {
    run_dir: runDir,
    frame_index: frameIndex,
  });

export const getEvidenceBundle = (runDir: string, bundleDir: string) =>
  engineRequest<Json>("get_evidence_bundle", { run_dir: runDir, bundle_dir: bundleDir });

export const getAudioReviewClip = (runDir: string, itemId: string) =>
  engineRequest<Json>("get_audio_review_clip", { run_dir: runDir, item_id: itemId });

export const getWaveformMetadata = (runDir: string) =>
  engineRequest<WaveformMetadataPayload>("get_waveform_metadata", { run_dir: runDir });

export const getCaptionState = (runDir: string) =>
  engineRequest<CaptionStatePayload>("get_caption_state", { run_dir: runDir });

/** Engine truth for decision targets and per-item resolution status — the UI
 * never infers targets from titles or tracks resolution locally. */
export const getReviewResolution = (runDir: string) =>
  engineRequest<ReviewResolutionPayload>("get_review_resolution", { run_dir: runDir });

/** Phase 1 media truth: source pixel dimensions (never hardcoded in the UI). */
export const getMediaDimensions = (runDir: string) =>
  engineRequest<MediaDimensionsPayload>("get_media_dimensions", { run_dir: runDir });

// -- human inputs -----------------------------------------------------------

export const saveReviewDecisions = (runDir: string, decisions: HumanReviewDecision[]) =>
  engineRequest<{ path: string; count: number }>("save_review_decisions", {
    run_dir: runDir,
    decisions: decisions as unknown as Json[],
  });

export const saveHumanFacts = (runDir: string, facts: HumanCaptionFact[]) =>
  engineRequest<{ path: string; count: number }>("save_human_facts", {
    run_dir: runDir,
    facts: facts as unknown as Json[],
  });

/** Transactional save of both decisions and facts in one engine command. */
export const saveReviewInputs = (
  runDir: string,
  decisions: HumanReviewDecision[],
  facts: HumanCaptionFact[],
) =>
  engineRequest<SaveReviewInputsPayload>("save_review_inputs", {
    run_dir: runDir,
    decisions: decisions as unknown as Json[],
    facts: facts as unknown as Json[],
  });

export const appendAuditHistory = (runDir: string, entries: AuditHistoryEntry[]) =>
  engineRequest<{ appended: number }>("append_audit_history", {
    run_dir: runDir,
    entries: entries as unknown as Json[],
  });

export const getAuditHistory = (runDir: string) =>
  engineRequest<{ entries: AuditHistoryEntry[] }>("get_audit_history", { run_dir: runDir });

export const saveUiState = (runDir: string, uiState: UiStatePayload) =>
  engineRequest<{ path: string }>("save_ui_state", {
    run_dir: runDir,
    state: uiState as unknown as Json,
  });

export const getUiState = (runDir: string) =>
  engineRequest<UiStatePayload>("get_ui_state", { run_dir: runDir });

export const exportDraft = (runDir: string) =>
  engineRequest<ExportCaptionPayload>("export_draft", { run_dir: runDir });

export const getReviewInputs = (runDir: string) =>
  engineRequest<{ decisions: HumanReviewDecision[]; facts: HumanCaptionFact[] }>(
    "get_review_inputs",
    { run_dir: runDir },
  );

export const saveVisualAnchors = (
  runDir: string,
  anchors: Array<{
    frame_index: number;
    x: number;
    y: number;
    width: number;
    height: number;
    entity_type: string;
    label: string;
  }>,
) =>
  engineRequest<{ path: string; count: number }>("save_visual_anchors", {
    run_dir: runDir,
    anchors: anchors as unknown as Json[],
  });

export const finalize = (runDir: string) =>
  engineRequest<{ result: Json; caption_state: CaptionStatePayload }>("finalize", {
    run_dir: runDir,
  });

export const createFinalSignoff = (runDir: string, signoff: FinalReviewSignoff) =>
  engineRequest<{ path: string }>("create_final_signoff", {
    run_dir: runDir,
    signoff: signoff as unknown as Json,
  });

export const validateFinalSignoff = (runDir: string) =>
  engineRequest<SignoffValidationPayload>("validate_final_signoff", { run_dir: runDir });

export const exportCaption = (runDir: string) =>
  engineRequest<ExportCaptionPayload>("export_caption", { run_dir: runDir });

// -- local app services (Rust-owned, not engine commands) -------------------

/** Serve a run-local file (evidence image, waveform PNG, review clip) as a
 * data URL through a validated Rust command — no broad fs scope for the UI. */
export const readRunFile = (runDir: string, path: string) =>
  invoke<string>("read_run_file", { runDir, path });

export const listRecentRuns = () => invoke<import("./types").RecentRun[]>("list_recent_runs");

export const rememberRecentRun = (entry: import("./types").RecentRun) =>
  invoke<void>("remember_recent_run", { entry });

export const forgetRecentRun = (runDir: string) =>
  invoke<void>("forget_recent_run", { runDir });

export const openArtifactFolder = (runDir: string) =>
  invoke<void>("open_artifact_folder", { runDir });

/** Register a just-picked intake video with Rust so playback trust exists.
 * Returns the canonical path the app must use from then on. */
export const registerIntakeVideo = (path: string) =>
  invoke<string>("register_intake_video", { path });

/** Start a visual-analysis rerun that consumes the saved anchors. Progress and
 * completion arrive on the same analysis:// events as start_analysis. */
export const startRerunWithAnchors = (runDir: string) =>
  invoke<string>("start_rerun_with_anchors", { runDir });

/** Rust fetches the engine's READY caption bytes itself and writes them to
 * destination — the UI never passes caption text. */
export const saveReadyCaption = (runDir: string, destination: string) =>
  invoke<void>("save_ready_caption", { runDir, destination });

/** Write the engine's draft_review_only.md bytes to destination. */
export const saveReviewDraft = (runDir: string, destination: string) =>
  invoke<void>("save_review_draft", { runDir, destination });
