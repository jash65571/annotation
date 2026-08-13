/**
 * Bridge DTO types, mirroring engine/manuscript_reviewer/ui_bridge/protocol.py
 * and the engine's persisted artifact models. The Python engine remains the
 * factual authority — these types describe what it already writes; they never
 * define parallel truth.
 */

export const UI_BRIDGE_PROTOCOL_VERSION = 1;

// ---------------------------------------------------------------------------
// Protocol envelope
// ---------------------------------------------------------------------------

export type BridgeErrorCode =
  | "ENGINE_NOT_FOUND"
  | "ENGINE_VERSION_MISMATCH"
  | "PROTOCOL_VERSION_MISMATCH"
  | "FFMPEG_UNAVAILABLE"
  | "RUN_NOT_FOUND"
  | "RUN_LOCKED"
  | "MANIFEST_MISMATCH"
  | "INVALID_DECISION"
  | "ASR_UNAVAILABLE"
  | "OCR_UNAVAILABLE"
  | "CANCELLED"
  | "ENGINE_CRASH"
  | "INVALID_INPUT"
  | "INVALID_COMMAND"
  | "NOT_READY"
  | "ARTIFACT_NOT_FOUND";

export interface BridgeError {
  code: BridgeErrorCode;
  message: string;
  detail?: string | null;
}

export interface ProgressEventPayload {
  stage: string;
  status: "STARTED" | "RUNNING" | "COMPLETED" | "FAILED" | "SKIPPED";
  detail?: string | null;
  current?: number | null;
  total?: number | null;
}

// ---------------------------------------------------------------------------
// Engine enums (exact engine vocabulary — never renamed in the UI)
// ---------------------------------------------------------------------------

export type CaptionReadiness =
  | "BLOCKED"
  | "REVIEW_REQUIRED"
  | "READY_FOR_FINAL_REVIEW"
  | "READY_TO_ENTER";

export type ReviewPriority = "CRITICAL" | "HIGH" | "NORMAL" | "LOW";

export type CaptionEligibility =
  | "ELIGIBLE"
  | "REVIEW_REQUIRED"
  | "INELIGIBLE"
  | "REJECTED";

export type DecisionType =
  | "CLAIM_EVIDENCE"
  | "OCR_TEXT"
  | "OCR_TIMING"
  | "IDENTITY_MAPPING"
  | "CAMERA_CLASSIFICATION"
  | "ACTION_SEMANTICS"
  | "ACTION_BOUNDARY"
  | "PLAYBACK_SPEED"
  | "REVIEW_PROPOSAL_OUTCOME"
  | "SPEECH_VERIFICATION"
  | "SPEECH_CORRECTION"
  | "TEXT_VERIFICATION"
  | "TEXT_CORRECTION"
  | "TEXT_TIMING"
  | "TRANSITION_CLASSIFICATION";

// ---------------------------------------------------------------------------
// Evidence + review structures (pass-through of engine artifacts)
// ---------------------------------------------------------------------------

export interface EvidenceReference {
  evidence_id: string;
  evidence_type: string;
  start_frame?: number | null;
  end_frame?: number | null;
  start_pts?: number | null;
  end_pts?: number | null;
  /** Exact rational "num/den" strings — display only, never re-derived. */
  start_time_seconds?: string | null;
  end_time_seconds?: string | null;
  artifact_paths?: string[];
  source?: string | null;
  notes?: string | null;
}

export interface VisualReviewItem {
  item_id: string;
  priority: ReviewPriority;
  title: string;
  reason: string;
  shot_number?: number | null;
  start_frame?: number | null;
  end_frame?: number | null;
  start_exact?: string | null;
  end_exact?: string | null;
  supporting_evidence_refs?: EvidenceReference[];
  contradicting_evidence_refs?: EvidenceReference[];
  recommended_action: string;
  related_claim_ids?: string[];
  evidence_bundle_dir?: string | null;
}

export interface AudioReviewItem {
  item_id: string;
  reasons: string[];
  priority: ReviewPriority;
  start_exact: string;
  end_exact: string;
  playback_start: string;
  playback_end: string;
  asr_text_candidate?: string | null;
  evidence_refs?: string[];
  review_clip?: string | null;
  notes?: string[];
}

export interface ReviewQueuePayload {
  visual_items: VisualReviewItem[];
  audio_items: AudioReviewItem[];
}

export interface HumanReviewDecision {
  decision_id: string;
  subject_id: string;
  decision_type: DecisionType;
  value: string;
  reviewer_note?: string | null;
  evidence_refs?: EvidenceReference[];
  decided_at_utc?: string | null;
  decided_by: string;
  bound_video_sha256: string;
  bound_rules_version: string;
  bound_evidence_id?: string | null;
}

export interface HumanCaptionFact {
  fact_id: string;
  fact_type: string;
  text_value?: string | null;
  semantic_value?: Record<string, string>;
  shot_number?: number | null;
  character_ids?: string[];
  object_ids?: string[];
  start_frame?: number | null;
  end_frame?: number | null;
  start_exact?: string | null;
  end_exact?: string | null;
  evidence_refs: EvidenceReference[];
  decided_by: string;
  decided_at_utc?: string | null;
  bound_video_sha256: string;
  bound_rules_version: string;
  bound_seed_sha256?: string | null;
  bound_evidence_ids?: string[];
}

export interface FinalReviewSignoff {
  video_sha256: string;
  rules_version: string;
  caption_sha256: string;
  reviewer: string;
  reviewed_at_utc?: string | null;
  golden_example_comparison_complete: boolean;
  platform_semantic_pass_complete: boolean;
  final_adversarial_read_complete: boolean;
  no_known_omissions_confirmed: boolean;
  no_known_hallucinations_confirmed: boolean;
  comments?: string | null;
}

// ---------------------------------------------------------------------------
// Command payloads (loosely typed pass-throughs stay `unknown`-shaped JSON)
// ---------------------------------------------------------------------------

export type Json = Record<string, unknown>;

export interface HealthPayload {
  protocol_version: number;
  engine_version: string;
  rules_version: string;
  ffmpeg: { available: boolean; path: string | null; version: string | null };
  ffprobe: { available: boolean; path: string | null; version: string | null };
  asr_worker_envs: { fw_env: boolean; wx_env: boolean };
  ocr: { tesseract_on_path: boolean; tesseract_dir_env: string | null };
}

export interface RunSummaryPayload {
  run_dir: string;
  manifest: Json;
  qc: Json | null;
  caption_final_status: (Json & { readiness?: CaptionReadiness }) | null;
  audio_qc: Json | null;
  visual_qc: Json | null;
  shot_qc: Json | null;
  has_extracted_frames: boolean;
  ui_inputs: {
    review_decisions: boolean;
    human_facts: boolean;
    final_review: boolean;
  };
}

export interface ExactFramePayload {
  frame_index: number;
  record: Json;
  path: string;
}

export interface CaptionStatePayload {
  final_status: (Json & { readiness?: CaptionReadiness; blockers?: string[] }) | null;
  reviewed_caption: Json | null;
  caption_manifest: Json | null;
  /** Signoff-binding hash of the current rendered caption. */
  rendered_caption_sha256: string | null;
  ready_markdown: string | null;
  draft_markdown: string | null;
  m2_validator: Json | null;
  platform_semantic: Json | null;
  golden_gate: Json | null;
  coverage: Json | null;
  assertion_map: Json | null;
  seed_change_log: Json | null;
  caption_facts: Json | null;
  eligibility_report: Json | null;
  final_review_checklist: Json | null;
}

export interface ExportCaptionPayload {
  markdown: string;
  sha256: string;
  path: string;
}

export interface SignoffValidationPayload {
  present: boolean;
  valid: boolean;
  stale: boolean;
  reasons: string[];
}

export interface StartAuditOptions {
  extract_frames?: boolean;
  shot_analysis?: boolean;
  shot_sensitivity?: "high" | "normal" | "low";
  extract_shot_evidence?: boolean;
  scdet?: boolean;
  audio_analysis?: boolean;
  asr?: boolean;
  asr_model?: string;
  asr_device?: string;
  asr_compute_type?: string;
  asr_language?: string;
  asr_bootstrap?: boolean;
  visual_intelligence?: boolean;
  ocr?: boolean;
  caption_brain?: boolean;
}

export interface StartAuditRequest {
  video_path: string;
  artifacts_root?: string;
  seed_path?: string;
  seed_text?: string;
  feedback_path?: string;
  feedback_text?: string;
  options?: StartAuditOptions;
}

export interface StartAuditPayload {
  run_id: string;
  run_dir: string;
  status: string;
  summary: RunSummaryPayload;
}

export interface WaveformMetadataPayload {
  timeline: Json | null;
  waveform_png: string | null;
  spectrogram_png: string | null;
  energy_png: string | null;
  source_wav: string | null;
  audio_qc: Json | null;
}

/** An entry in the app-local recent-runs index (display cache only — the run
 * directory stays the source of truth). */
export interface RecentRun {
  run_dir: string;
  video_name: string;
  last_opened_utc: string;
  display_status: string;
  readiness: CaptionReadiness | "INCOMPLETE" | "UNKNOWN";
}
