/**
 * The single owner of the run's human review inputs on the UI side.
 *
 * - Decisions/facts always use the engine's typed models, bound to the exact
 *   video SHA-256 and rules version (spec §52).
 * - Every valid change auto-saves (atomic write in the engine) and then
 *   re-finalizes the Caption Brain immediately (spec §51, §61).
 * - Undo pops a revision from a previous-value stack, persists the previous
 *   state, and re-finalizes — history is never mutated invisibly (spec §95).
 * - A local structured audit trail records what changed (spec §96).
 */

import {
  appendAuditHistory,
  finalize,
  getAuditHistory,
  getRunSummary,
  saveReviewInputs,
} from "../../api/bridge";
import type {
  CaptionStatePayload,
  HumanCaptionFact,
  HumanReviewDecision,
  Json,
  RunSummaryPayload,
} from "../../api/types";

export interface AuditTrailEntry {
  at_utc: string;
  action:
    | "decision_saved"
    | "decision_replaced"
    | "decision_removed"
    | "fact_saved"
    | "fact_removed"
    | "undo";
  subject: string;
  reviewer: string;
  detail?: string;
}

export interface FinalizeOutcome {
  readiness: string;
  captionState: CaptionStatePayload;
  summary: RunSummaryPayload;
  result: Json;
}

interface Snapshot {
  decisions: HumanReviewDecision[];
  facts: HumanCaptionFact[];
  action: string;
}

export type SaveState = "idle" | "saving" | "saved" | "error";

export class DecisionsStore {
  readonly runDir: string;
  readonly videoSha256: string;
  readonly rulesVersion: string;
  private decisions: HumanReviewDecision[] = [];
  private facts: HumanCaptionFact[] = [];
  private undoStack: Snapshot[] = [];
  /** In-memory cache of the persisted audit history (spec §96). */
  private auditTrail: AuditTrailEntry[] = [];
  /** Entries recorded since the last successful persisted append. */
  private pendingAudit: AuditTrailEntry[] = [];
  /** Last human name seen on a decision/fact — used for remove/undo entries. */
  private lastReviewer = "";
  private listeners = new Set<() => void>();
  saveState: SaveState = "idle";
  lastError: string | null = null;

  constructor(runDir: string, videoSha256: string, rulesVersion: string) {
    this.runDir = runDir;
    this.videoSha256 = videoSha256;
    this.rulesVersion = rulesVersion;
  }

  subscribe(listener: () => void): () => void {
    this.listeners.add(listener);
    return () => this.listeners.delete(listener);
  }

  private notify(): void {
    for (const listener of this.listeners) listener();
  }

  getDecisions(): readonly HumanReviewDecision[] {
    return this.decisions;
  }

  getFacts(): readonly HumanCaptionFact[] {
    return this.facts;
  }

  getAuditTrail(): readonly AuditTrailEntry[] {
    return this.auditTrail;
  }

  canUndo(): boolean {
    return this.undoStack.length > 0;
  }

  seed(decisions: HumanReviewDecision[], facts: HumanCaptionFact[]): void {
    this.decisions = decisions;
    this.facts = facts;
    this.notify();
  }

  decisionFor(subjectId: string, decisionType: string): HumanReviewDecision | undefined {
    return this.decisions.find(
      (d) => d.subject_id === subjectId && d.decision_type === decisionType,
    );
  }

  /** Build a fully-bound decision. `decided_by` must be a human name the
   * reviewer entered — the store never invents one. */
  makeDecision(
    input: Omit<
      HumanReviewDecision,
      "decision_id" | "bound_video_sha256" | "bound_rules_version" | "decided_at_utc"
    >,
  ): HumanReviewDecision {
    return {
      ...input,
      decision_id: `D-${input.decision_type}-${input.subject_id}-${Date.now()}`,
      decided_at_utc: new Date().toISOString(),
      bound_video_sha256: this.videoSha256,
      bound_rules_version: this.rulesVersion,
    };
  }

  /** Apply (add or replace) one decision, persist, re-finalize. */
  async applyDecision(decision: HumanReviewDecision): Promise<FinalizeOutcome> {
    this.pushUndo(`decision ${decision.decision_type} on ${decision.subject_id}`);
    const replaced = this.decisions.some(
      (d) =>
        d.subject_id === decision.subject_id &&
        d.decision_type === decision.decision_type,
    );
    this.decisions = [
      ...this.decisions.filter(
        (d) =>
          !(
            d.subject_id === decision.subject_id &&
            d.decision_type === decision.decision_type
          ),
      ),
      decision,
    ];
    this.lastReviewer = decision.decided_by;
    this.recordAudit(
      replaced ? "decision_replaced" : "decision_saved",
      `${decision.decision_type}:${decision.subject_id}`,
      decision.decided_by,
      decision.value,
    );
    return this.persistAndFinalize();
  }

  async removeDecision(subjectId: string, decisionType: string): Promise<FinalizeOutcome> {
    this.pushUndo(`remove ${decisionType} on ${subjectId}`);
    this.decisions = this.decisions.filter(
      (d) => !(d.subject_id === subjectId && d.decision_type === decisionType),
    );
    this.recordAudit("decision_removed", `${decisionType}:${subjectId}`, this.lastReviewer);
    return this.persistAndFinalize();
  }

  async applyFact(fact: HumanCaptionFact): Promise<FinalizeOutcome> {
    this.pushUndo(`fact ${fact.fact_id}`);
    this.facts = [...this.facts.filter((f) => f.fact_id !== fact.fact_id), fact];
    this.lastReviewer = fact.decided_by;
    this.recordAudit("fact_saved", fact.fact_id, fact.decided_by, fact.fact_type);
    return this.persistAndFinalize();
  }

  async removeFact(factId: string): Promise<FinalizeOutcome> {
    this.pushUndo(`remove fact ${factId}`);
    this.facts = this.facts.filter((f) => f.fact_id !== factId);
    this.recordAudit("fact_removed", factId, this.lastReviewer);
    return this.persistAndFinalize();
  }

  /** Undo the latest human change: restore the previous snapshot, persist it,
   * re-finalize. Machine evidence is never touched. */
  async undo(): Promise<FinalizeOutcome | null> {
    const snapshot = this.undoStack.pop();
    if (!snapshot) return null;
    this.decisions = snapshot.decisions;
    this.facts = snapshot.facts;
    this.recordAudit("undo", snapshot.action, this.lastReviewer);
    return this.persistAndFinalize();
  }

  /** Replace the in-memory audit cache with the engine's persisted history. */
  async loadPersistedAuditTrail(): Promise<readonly AuditTrailEntry[]> {
    const history = await getAuditHistory(this.runDir);
    this.auditTrail = history.entries.map((entry) => ({
      at_utc: entry.at_utc,
      action: entry.operation as AuditTrailEntry["action"],
      subject: entry.subject,
      reviewer: entry.reviewer,
      ...(entry.detail != null ? { detail: entry.detail } : {}),
    }));
    this.notify();
    return this.auditTrail;
  }

  private pushUndo(action: string): void {
    this.undoStack.push({
      decisions: this.decisions,
      facts: this.facts,
      action,
    });
    if (this.undoStack.length > 50) this.undoStack.shift();
  }

  private recordAudit(
    action: AuditTrailEntry["action"],
    subject: string,
    reviewer: string,
    detail?: string,
  ): void {
    const entry: AuditTrailEntry = {
      at_utc: new Date().toISOString(),
      action,
      subject,
      reviewer,
      ...(detail !== undefined ? { detail } : {}),
    };
    this.auditTrail.push(entry);
    this.pendingAudit.push(entry);
  }

  private async persistAndFinalize(): Promise<FinalizeOutcome> {
    this.saveState = "saving";
    this.lastError = null;
    this.notify();
    try {
      const saved = await saveReviewInputs(this.runDir, this.decisions, this.facts);
      await this.flushAuditHistory(saved.review_input_revision_id);
      const finalized = await finalize(this.runDir);
      const summary = await getRunSummary(this.runDir);
      this.saveState = "saved";
      this.notify();
      const readiness =
        (finalized.result as { readiness?: string }).readiness ?? "REVIEW_REQUIRED";
      return {
        readiness,
        captionState: finalized.caption_state,
        summary,
        result: finalized.result,
      };
    } catch (error) {
      this.saveState = "error";
      this.lastError = String(error);
      this.notify();
      throw error;
    }
  }

  /** Append the pending audit entries to the engine's persisted history.
   * A failure never fails the mutation itself: entries stay queued and are
   * re-sent with the next successful save. */
  private async flushAuditHistory(newRevision: string): Promise<void> {
    if (this.pendingAudit.length === 0) return;
    const entries = this.pendingAudit.map((entry) => ({
      at_utc: entry.at_utc,
      reviewer: entry.reviewer,
      operation: entry.action,
      subject: entry.subject,
      ...(entry.detail !== undefined ? { detail: entry.detail } : {}),
      new_revision: newRevision,
    }));
    try {
      await appendAuditHistory(this.runDir, entries);
      this.pendingAudit = [];
    } catch {
      /* keep entries queued for the next save */
    }
  }
}
