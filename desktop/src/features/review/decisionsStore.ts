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
  finalize,
  getRunSummary,
  saveHumanFacts,
  saveReviewDecisions,
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
  private auditTrail: AuditTrailEntry[] = [];
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
    this.recordAudit(
      replaced ? "decision_replaced" : "decision_saved",
      `${decision.decision_type}:${decision.subject_id}`,
      decision.value,
    );
    return this.persistAndFinalize();
  }

  async removeDecision(subjectId: string, decisionType: string): Promise<FinalizeOutcome> {
    this.pushUndo(`remove ${decisionType} on ${subjectId}`);
    this.decisions = this.decisions.filter(
      (d) => !(d.subject_id === subjectId && d.decision_type === decisionType),
    );
    this.recordAudit("decision_removed", `${decisionType}:${subjectId}`);
    return this.persistAndFinalize();
  }

  async applyFact(fact: HumanCaptionFact): Promise<FinalizeOutcome> {
    this.pushUndo(`fact ${fact.fact_id}`);
    this.facts = [...this.facts.filter((f) => f.fact_id !== fact.fact_id), fact];
    this.recordAudit("fact_saved", fact.fact_id, fact.fact_type);
    return this.persistAndFinalize();
  }

  async removeFact(factId: string): Promise<FinalizeOutcome> {
    this.pushUndo(`remove fact ${factId}`);
    this.facts = this.facts.filter((f) => f.fact_id !== factId);
    this.recordAudit("fact_removed", factId);
    return this.persistAndFinalize();
  }

  /** Undo the latest human change: restore the previous snapshot, persist it,
   * re-finalize. Machine evidence is never touched. */
  async undo(): Promise<FinalizeOutcome | null> {
    const snapshot = this.undoStack.pop();
    if (!snapshot) return null;
    this.decisions = snapshot.decisions;
    this.facts = snapshot.facts;
    this.recordAudit("undo", snapshot.action);
    return this.persistAndFinalize();
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
    detail?: string,
  ): void {
    this.auditTrail.push({
      at_utc: new Date().toISOString(),
      action,
      subject,
      ...(detail !== undefined ? { detail } : {}),
    });
  }

  private async persistAndFinalize(): Promise<FinalizeOutcome> {
    this.saveState = "saving";
    this.lastError = null;
    this.notify();
    try {
      await saveReviewDecisions(this.runDir, this.decisions);
      await saveHumanFacts(this.runDir, this.facts);
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
}
