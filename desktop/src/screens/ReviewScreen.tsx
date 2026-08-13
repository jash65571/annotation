/**
 * The review workstation (spec §29+): media viewing (left), timeline + queue
 * (center), evidence + decision editors (right), caption preview + gates
 * (bottom). Every decision persists immediately and re-finalizes the Caption
 * Brain; the engine's readiness vocabulary is displayed verbatim.
 */

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useApp } from "../state/context";
import type { Screen } from "../App";
import {
  getCaptionState,
  getMediaDimensions,
  getReviewInputs,
  getReviewQueue,
  getReviewResolution,
  getShots,
  getUiState,
  saveUiState,
  saveVisualAnchors,
  startRerunWithAnchors,
} from "../api/bridge";
import type {
  AudioReviewItem,
  CaptionStatePayload,
  DecisionTargetRef,
  HumanReviewDecision,
  Json,
  MediaDimensionsPayload,
  ResolutionStatus,
  ReviewQueuePayload,
  ReviewResolutionPayload,
  VisualReviewItem,
} from "../api/types";
import { DecisionsStore, type FinalizeOutcome } from "../features/review/decisionsStore";
import { QueuePanel, buildQueueEntries, type QueueEntry } from "../features/review/QueuePanel";
import { EvidencePanel } from "../features/evidence/EvidencePanel";
import { ExactFrameViewer } from "../features/evidence/ExactFrameViewer";
import { VideoPlayer } from "../features/player/VideoPlayer";
import { WaveformPanel } from "../features/audio/WaveformPanel";
import { Timeline, type TimelineLane } from "../features/timeline/Timeline";
import { AnchorEditor, type AnchorDraft } from "../features/anchors/AnchorEditor";
import { SpeechEditor } from "../features/review/editors/SpeechEditor";
import { TransitionEditor } from "../features/review/editors/TransitionEditor";
import { SpeedEditor } from "../features/review/editors/SpeedEditor";
import { OcrEditor } from "../features/review/editors/OcrEditor";
import { IdentityEditor } from "../features/review/editors/IdentityEditor";
import { ActionEditor } from "../features/review/editors/ActionEditor";
import { CameraEditor } from "../features/review/editors/CameraEditor";
import { ProposalEditor } from "../features/review/editors/ProposalEditor";
import { HumanFactEditor } from "../features/facts/HumanFactEditor";
import { ReviewerNameField } from "../features/review/reviewerName";
import { GateBadge } from "../components/GateBadge";

type MediaMode = "playback" | "frame" | "audio";

interface ShotInfo {
  shot_number: number;
  start_frame: number;
  end_frame: number;
  start_seconds: number;
  end_seconds: number;
  transition?: string | undefined;
}

function parseExact(exact: unknown): number {
  if (typeof exact !== "string") return 0;
  const match = /^(-?\d+)\/(\d+)$/.exec(exact);
  if (match) {
    const den = Number(match[2]);
    return den === 0 ? 0 : Number(match[1]) / den;
  }
  const plain = Number(exact);
  return Number.isFinite(plain) ? plain : 0;
}

function extractShots(shotsPayload: Json | null): ShotInfo[] {
  const proposed = (shotsPayload as { shots_proposed?: { shots?: unknown[] } } | null)
    ?.shots_proposed;
  const list = Array.isArray(proposed?.shots) ? proposed.shots : [];
  const shots: ShotInfo[] = [];
  for (const raw of list) {
    const shot = raw as {
      shot_index?: number;
      start_frame?: number;
      end_frame?: number;
      start_exact?: string;
      end_exact?: string;
      transition_into_shot?: string;
    };
    if (shot.shot_index == null) continue;
    shots.push({
      shot_number: shot.shot_index,
      start_frame: shot.start_frame ?? 0,
      end_frame: shot.end_frame ?? 0,
      start_seconds: parseExact(shot.start_exact),
      end_seconds: parseExact(shot.end_exact),
      transition: shot.transition_into_shot,
    });
  }
  return shots;
}

/** Parse the engine-defined shot index from a SHOT_TRANSITION subject id
 * ("TRANSITION-<n>" — the number after the dash). */
export function transitionShotIndex(subjectId: string): number | null {
  const match = /-(\d+)$/.exec(subjectId);
  const value = match?.[1];
  return value !== undefined ? Number(value) : null;
}

/** A tiny line diff for the caption-change popover (spec §62). */
export function diffLines(before: string, after: string): { removed: string[]; added: string[] } {
  const beforeLines = before.split("\n");
  const afterLines = after.split("\n");
  const beforeSet = new Set(beforeLines);
  const afterSet = new Set(afterLines);
  return {
    removed: beforeLines.filter((line) => line.trim() && !afterSet.has(line)),
    added: afterLines.filter((line) => line.trim() && !beforeSet.has(line)),
  };
}

export function ReviewScreen({ onNavigate }: { onNavigate: (screen: Screen) => void }) {
  const { state, dispatch } = useApp();
  const runDir = state.runDir;
  const manifest = (state.runSummary?.manifest ?? {}) as {
    source_video_sha256?: string;
    rules_version?: string;
    source_video_path?: string;
  };

  const [queue, setQueue] = useState<ReviewQueuePayload | null>(null);
  const [caption, setCaption] = useState<CaptionStatePayload | null>(null);
  const [shots, setShots] = useState<ShotInfo[]>([]);
  const [resolution, setResolution] = useState<ReviewResolutionPayload | null>(null);
  const [mediaDims, setMediaDims] = useState<MediaDimensionsPayload | null>(null);
  const [selected, setSelected] = useState<QueueEntry | null>(null);
  /** The engine-provided target the open editor operates on. */
  const [activeTarget, setActiveTarget] = useState<DecisionTargetRef | null>(null);
  /** Candidate targets awaiting a human choice (chooser panel). */
  const [candidateTargets, setCandidateTargets] = useState<DecisionTargetRef[]>([]);
  const [skippedIds, setSkippedIds] = useState<ReadonlySet<string>>(new Set());
  const [mediaMode, setMediaMode] = useState<MediaMode>("playback");
  const [currentFrame, setCurrentFrame] = useState(0);
  const [seekSeconds, setSeekSeconds] = useState<number | undefined>(undefined);
  const [showFactEditor, setShowFactEditor] = useState(false);
  const [showAnchorEditor, setShowAnchorEditor] = useState(false);
  const [anchors, setAnchors] = useState<AnchorDraft[]>([]);
  const [diff, setDiff] = useState<{ removed: string[]; added: string[] } | null>(null);
  const [saveIndicator, setSaveIndicator] = useState("");
  const [error, setError] = useState<string | null>(null);
  const lastDraftRef = useRef<string>("");

  const store = useMemo(() => {
    if (!runDir || !manifest.source_video_sha256 || !manifest.rules_version) return null;
    return new DecisionsStore(runDir, manifest.source_video_sha256, manifest.rules_version);
  }, [runDir, manifest.source_video_sha256, manifest.rules_version]);

  useEffect(() => {
    if (!store) return undefined;
    return store.subscribe(() => {
      setSaveIndicator(
        store.saveState === "saving"
          ? "Saving…"
          : store.saveState === "saved"
            ? "Saved"
            : store.saveState === "error"
              ? `Save failed: ${store.lastError ?? ""}`
              : "",
      );
    });
  }, [store]);

  /** Resolution status is engine truth — re-fetched after every mutation. */
  const refreshResolution = useCallback(async () => {
    if (!runDir) return;
    try {
      setResolution(await getReviewResolution(runDir));
    } catch (e) {
      setError(String(e));
    }
  }, [runDir]);

  const reload = useCallback(async () => {
    if (!runDir) return;
    const [q, c, s, r] = await Promise.all([
      getReviewQueue(runDir),
      getCaptionState(runDir),
      getShots(runDir),
      getReviewResolution(runDir),
    ]);
    setQueue(q);
    setCaption(c);
    setShots(extractShots(s as Json));
    setResolution(r);
    lastDraftRef.current = c.draft_markdown ?? c.ready_markdown ?? "";
    try {
      setMediaDims(await getMediaDimensions(runDir));
    } catch {
      /* anchors stay disabled without media truth */
    }
    try {
      const uiState = await getUiState(runDir);
      setSkippedIds(new Set(uiState.skipped_item_ids ?? []));
    } catch {
      /* no saved UI state yet */
    }
    if (store) {
      try {
        const inputs = await getReviewInputs(runDir);
        store.seed(inputs.decisions, inputs.facts);
      } catch {
        /* fresh run without saved inputs */
      }
      try {
        await store.loadPersistedAuditTrail();
      } catch {
        /* no persisted audit history yet */
      }
    }
  }, [runDir, store]);

  useEffect(() => {
    reload().catch((e) => setError(String(e)));
  }, [reload]);

  const maxFrameIndex = useMemo(
    () => shots.reduce((max, shot) => Math.max(max, shot.end_frame), 0) || 100000,
    [shots],
  );
  const durationSeconds = useMemo(
    () => shots.reduce((max, shot) => Math.max(max, shot.end_seconds), 0) || 16,
    [shots],
  );

  const handleOutcome = useCallback(
    (outcome: FinalizeOutcome) => {
      setCaption(outcome.captionState);
      const newDraft =
        outcome.captionState.draft_markdown ?? outcome.captionState.ready_markdown ?? "";
      if (lastDraftRef.current && newDraft !== lastDraftRef.current) {
        setDiff(diffLines(lastDraftRef.current, newDraft));
      }
      lastDraftRef.current = newDraft;
      dispatch({
        type: "READINESS_CHANGED",
        readiness: outcome.readiness as never,
        summary: outcome.summary,
      });
      void refreshResolution();
    },
    [dispatch, refreshResolution],
  );

  /** Engine truth: item_id → resolution status for badges and counts. */
  const resolutionById = useMemo(() => {
    const map = new Map<string, ResolutionStatus>();
    for (const item of resolution?.items ?? []) {
      if (item.item_id != null) map.set(item.item_id, item.resolution_status);
    }
    return map;
  }, [resolution]);

  /** Editor routing comes from the resolution payload only — never titles. */
  const selectEntry = useCallback(
    (entry: QueueEntry) => {
      setSelected(entry);
      const resItem = resolution?.items.find((item) => item.item_id === entry.id) ?? null;
      if (resItem?.decision_target) {
        setActiveTarget(resItem.decision_target);
        setCandidateTargets([]);
      } else if (resItem && resItem.candidate_targets.length > 0) {
        setActiveTarget(null);
        setCandidateTargets(resItem.candidate_targets);
      } else {
        setActiveTarget(null);
        setCandidateTargets([]);
      }
      if (entry.source === "audio") {
        setMediaMode("audio");
      } else {
        const item = entry.item as VisualReviewItem;
        if (item.start_frame != null) {
          setCurrentFrame(item.start_frame);
          setMediaMode("frame");
        }
      }
    },
    [resolution],
  );

  /** Open an editor directly on an engine-provided required target. */
  const selectRequiredTarget = useCallback((target: DecisionTargetRef) => {
    setSelected(null);
    setCandidateTargets([]);
    setActiveTarget(target);
  }, []);

  /** Skipped is persisted run-local UI state; it never counts as resolved. */
  const toggleSkip = useCallback(
    (itemId: string) => {
      setSkippedIds((prev) => {
        const next = new Set(prev);
        if (next.has(itemId)) next.delete(itemId);
        else next.add(itemId);
        if (runDir) {
          saveUiState(runDir, { skipped_item_ids: [...next] }).catch((e) =>
            setError(String(e)),
          );
        }
        return next;
      });
    },
    [runDir],
  );

  if (!runDir) {
    return (
      <div className="col" style={{ padding: "var(--gap-lg)" }}>
        <p className="muted">No run loaded. Start a new review from Home.</p>
        <button onClick={() => onNavigate("home")} style={{ alignSelf: "flex-start" }}>
          Home
        </button>
      </div>
    );
  }

  const readiness = caption?.final_status?.readiness ?? "REVIEW_REQUIRED";
  const finalStatus = caption?.final_status;
  const blockers = (finalStatus?.blockers as string[] | undefined) ?? [];
  const videoPath = manifest.source_video_path;

  const timelineLanes: TimelineLane[] = [
    {
      id: "shots",
      label: "Shots",
      blocks: shots.map((shot) => ({
        id: `shot-${shot.shot_number}`,
        startSeconds: shot.start_seconds,
        endSeconds: shot.end_seconds,
        label: `Shot ${shot.shot_number}${shot.transition ? ` (${shot.transition})` : ""}`,
        kind: "shot" as const,
      })),
    },
    {
      id: "flags",
      label: "Review",
      blocks: (queue ? buildQueueEntries(queue) : [])
        .filter((entry) => resolutionById.get(entry.id) !== "RESOLVED")
        .map((entry) => {
          const item = entry.item as { start_exact?: string | null; end_exact?: string | null };
          const start = parseExact(item.start_exact ?? null);
          const end = Math.max(parseExact(item.end_exact ?? null), start + 0.05);
          return {
            id: `flag-${entry.id}`,
            startSeconds: start,
            endSeconds: end,
            label: entry.title,
            kind: "flag" as const,
          };
        }),
    },
    {
      id: "speech",
      label: "Speech",
      blocks: (queue?.audio_items ?? []).map((item) => ({
        id: `speech-${item.item_id}`,
        startSeconds: parseExact(item.start_exact),
        endSeconds: parseExact(item.end_exact),
        label: item.asr_text_candidate ?? "speech",
        kind: "speech" as const,
      })),
    },
  ];

  const selectedDecision =
    store && activeTarget
      ? ((store
          .getDecisions()
          .find((d) => d.subject_id === activeTarget.subject_id) ??
          null) as HumanReviewDecision | null)
      : null;

  return (
    <div className="col" style={{ height: "100%", padding: "var(--gap)", gap: "var(--gap)" }}>
      <div className="row" style={{ justifyContent: "space-between" }}>
        <div className="row">
          <h1 style={{ fontSize: 16, margin: 0 }}>Review</h1>
          <span
            className={`badge ${
              readiness === "READY_TO_ENTER"
                ? "pass"
                : readiness === "BLOCKED"
                  ? "blocked"
                  : "review"
            }`}
          >
            {readiness.replaceAll("_", " ")}
          </span>
          <span className="faint" role="status">
            {saveIndicator}
          </span>
        </div>
        <div className="row">
          <ReviewerNameField />
          <button
            disabled={!store?.canUndo()}
            onClick={() => {
              store
                ?.undo()
                .then((outcome) => outcome && handleOutcome(outcome))
                .catch((e) => setError(String(e)));
            }}
          >
            Undo
          </button>
          <button onClick={() => setShowFactEditor(true)}>Add verified fact</button>
          <button onClick={() => onNavigate("final")}>Final review</button>
        </div>
      </div>

      {error && (
        <div className="panel row">
          <span className="badge fail">ERROR</span>
          <span>{error}</span>
          <button onClick={() => setError(null)}>Dismiss</button>
        </div>
      )}

      <div className="row" style={{ flex: 1, minHeight: 0, alignItems: "stretch" }}>
        {/* LEFT: media viewing */}
        <section className="col" style={{ flex: "1 1 40%", minWidth: 320 }}>
          <div className="row" role="tablist" aria-label="Media mode">
            {(["playback", "frame", "audio"] as MediaMode[]).map((mode) => (
              <button
                key={mode}
                role="tab"
                aria-selected={mediaMode === mode}
                style={mediaMode === mode ? { borderColor: "var(--accent)" } : undefined}
                onClick={() => setMediaMode(mode)}
              >
                {mode === "playback" ? "Playback" : mode === "frame" ? "Exact frame" : "Audio"}
              </button>
            ))}
            <button onClick={() => setShowAnchorEditor((v) => !v)}>
              {showAnchorEditor ? "Close anchors" : "Anchors"}
            </button>
          </div>
          {mediaMode === "playback" && videoPath && <VideoPlayer videoPath={videoPath} {...(seekSeconds !== undefined ? { seekSeconds } : {})} />}
          {mediaMode === "playback" && !videoPath && (
            <p className="faint">Source video path unavailable.</p>
          )}
          {mediaMode === "frame" && (
            <ExactFrameViewer
              runDir={runDir}
              frameIndex={currentFrame}
              {...(shotForFrame(shots, currentFrame) !== undefined
                ? { shotNumber: shotForFrame(shots, currentFrame) as number }
                : {})}
              onFrameChange={setCurrentFrame}
              maxFrameIndex={maxFrameIndex}
            />
          )}
          {mediaMode === "audio" && <WaveformPanel runDir={runDir} />}
          {showAnchorEditor && (
            <div className="panel col">
              {mediaDims ? (
                <AnchorEditor
                  runDir={runDir}
                  frameIndex={currentFrame}
                  sourceWidth={mediaDims.width}
                  sourceHeight={mediaDims.height}
                  onSaveAnchor={(anchor) => {
                    const next = [...anchors, anchor];
                    setAnchors(next);
                    saveVisualAnchors(runDir, next)
                      .then(() => setSaveIndicator("Anchors saved"))
                      .catch((e) => setError(String(e)));
                  }}
                />
              ) : (
                <span className="faint">
                  Source media dimensions unavailable — anchoring is disabled.
                </span>
              )}
              <div className="row">
                <button
                  disabled={anchors.length === 0}
                  onClick={() => {
                    startRerunWithAnchors(runDir)
                      .then(() => {
                        dispatch({ type: "ANALYSIS_STARTED" });
                        onNavigate("analyze");
                      })
                      .catch((e) => setError(String(e)));
                  }}
                >
                  Re-run visual analysis with anchors
                </button>
                <span className="faint">{anchors.length} anchor(s) saved to this run.</span>
              </div>
            </div>
          )}
        </section>

        {/* CENTER: timeline + queue */}
        <section className="col" style={{ flex: "1 1 30%", minWidth: 280 }}>
          <Timeline
            durationExact={`${Math.round(durationSeconds * 1000)}/1000`}
            lanes={timelineLanes}
            onSeekSeconds={(seconds) => {
              setSeekSeconds(seconds);
              const shot = shots.find(
                (s) => seconds >= s.start_seconds && seconds < s.end_seconds,
              );
              if (shot && durationSeconds > 0) {
                const ratio =
                  (seconds - shot.start_seconds) / (shot.end_seconds - shot.start_seconds || 1);
                setCurrentFrame(
                  Math.min(
                    shot.end_frame,
                    shot.start_frame +
                      Math.round(ratio * (shot.end_frame - shot.start_frame)),
                  ),
                );
              }
            }}
          />
          {queue && (
            <QueuePanel
              queue={queue}
              resolutionById={resolutionById}
              selectedId={selected?.id ?? null}
              onSelect={selectEntry}
              skippedIds={skippedIds}
              onToggleSkip={toggleSkip}
            />
          )}
          {resolution && (
            <div className="panel col" aria-label="Required decisions">
              <h2 style={{ fontSize: 13, margin: 0 }}>Required decisions</h2>
              {[...resolution.speed_targets, ...resolution.transition_targets].map(
                (target) => (
                  <div
                    key={target.subject_id}
                    className="row"
                    style={{ justifyContent: "space-between" }}
                  >
                    <span>
                      {target.target_kind === "PLAYBACK_SPEED"
                        ? `Playback speed — Shot ${
                            (target as { shot_number?: number }).shot_number ?? "?"
                          }`
                        : `Transition — Shot ${
                            (target as { shot_index?: number }).shot_index ?? "?"
                          }`}
                    </span>
                    <span className="row">
                      <span
                        className={`badge ${
                          target.resolution_status === "RESOLVED" ? "pass" : "unresolved"
                        }`}
                      >
                        {target.resolution_status}
                      </span>
                      {target.resolution_status === "OPEN" && (
                        <button onClick={() => selectRequiredTarget(target)}>Decide</button>
                      )}
                    </span>
                  </div>
                ),
              )}
            </div>
          )}
        </section>

        {/* RIGHT: evidence + editor */}
        <section className="col" style={{ flex: "1 1 30%", minWidth: 300, overflow: "auto" }}>
          <EvidencePanel
            entry={selected}
            runDir={runDir}
            onShowFrame={(frameIndex) => {
              setCurrentFrame(frameIndex);
              setMediaMode("frame");
            }}
            onShowBundle={() => setMediaMode("frame")}
            onPlayClip={() => setMediaMode("audio")}
            currentDecision={selectedDecision}
          />
          {store && candidateTargets.length > 0 && !activeTarget && (
            <div className="panel col" aria-label="Choose decision target">
              <h2 style={{ fontSize: 13, margin: 0 }}>
                This item touches several engine records — choose one:
              </h2>
              {candidateTargets.map((target) => (
                <button
                  key={`${target.target_kind}:${target.subject_id}`}
                  onClick={() => setActiveTarget(target)}
                >
                  {target.target_kind} — <span className="mono">{target.subject_id}</span>
                </button>
              ))}
            </div>
          )}
          {store && (selected || activeTarget) && (
            <div className="panel col">
              <div className="row" style={{ justifyContent: "space-between" }}>
                <h2 style={{ fontSize: 13, margin: 0 }}>Decision</h2>
                {activeTarget && (
                  <span className="badge machine">
                    {activeTarget.target_kind}{" "}
                    <span className="mono">{activeTarget.subject_id}</span>
                  </span>
                )}
              </div>
              <TargetEditor
                target={activeTarget}
                entry={selected}
                store={store}
                currentFrame={currentFrame}
                onOutcome={handleOutcome}
              />
            </div>
          )}
        </section>
      </div>

      {/* BOTTOM: caption preview + gates */}
      <section className="panel col" style={{ maxHeight: 220, overflow: "auto" }}>
        <div className="row" style={{ flexWrap: "wrap" }}>
          <GateBadge
            label="M2"
            failCount={Number(finalStatus?.m2_fail_count ?? 0)}
            reviewCount={Number(finalStatus?.m2_review_count ?? 0)}
          />
          <GateBadge
            label="Platform"
            status={String(finalStatus?.platform_semantic_status ?? "NOT_RUN")}
          />
          <GateBadge label="Golden" status={String(finalStatus?.golden_gate_status ?? "NOT_RUN")} />
          <GateBadge
            label="Coverage"
            failCount={Number(finalStatus?.coverage_missing_required ?? 0)}
            reviewCount={0}
          />
          {blockers.length > 0 && (
            <span className="muted">{blockers.length} blocker(s) — first: {blockers[0]}</span>
          )}
        </div>
        {diff && (
          <div className="panel col" style={{ borderColor: "var(--accent)" }}>
            <div className="row" style={{ justifyContent: "space-between" }}>
              <strong>Caption changed</strong>
              <button onClick={() => setDiff(null)}>Dismiss</button>
            </div>
            {diff.removed.map((line) => (
              <div key={`r-${line}`} className="mono" style={{ color: "var(--status-fail)" }}>
                − {line}
              </div>
            ))}
            {diff.added.map((line) => (
              <div key={`a-${line}`} className="mono" style={{ color: "var(--status-pass)" }}>
                + {line}
              </div>
            ))}
          </div>
        )}
        <details>
          <summary>
            Caption draft <span className="badge review">REVIEW DRAFT — NOT FINAL</span>
          </summary>
          <pre className="mono" style={{ whiteSpace: "pre-wrap", margin: 0 }}>
            {caption?.draft_markdown ?? caption?.ready_markdown ?? "No caption rendered."}
          </pre>
        </details>
      </section>

      {showFactEditor && store && (
        <div className="panel" style={{ position: "fixed", inset: "10% 20%", overflow: "auto", zIndex: 10 }}>
          <HumanFactEditor
            runDir={runDir}
            store={store}
            shots={shots.map((shot) => ({
              shot_number: shot.shot_number,
              start_frame: shot.start_frame,
              end_frame: shot.end_frame,
            }))}
            currentFrame={currentFrame}
            onSaved={() => {
              setShowFactEditor(false);
              reload().catch((e) => setError(String(e)));
            }}
            onCancel={() => setShowFactEditor(false)}
          />
        </div>
      )}
    </div>
  );
}

function shotForFrame(shots: ShotInfo[], frame: number): number | undefined {
  return shots.find((shot) => frame >= shot.start_frame && frame <= shot.end_frame)
    ?.shot_number;
}

/** Editor routing from the ENGINE-provided decision target only — the UI
 * never infers targets from titles, recommended actions, or evidence ids. */
export function TargetEditor({
  target,
  entry,
  store,
  currentFrame,
  onOutcome,
}: {
  target: DecisionTargetRef | null;
  entry: QueueEntry | null;
  store: DecisionsStore;
  currentFrame: number;
  onOutcome: (outcome: FinalizeOutcome) => void;
}) {
  const onResolved = (outcome: unknown) => onOutcome(outcome as FinalizeOutcome);
  if (!target) {
    return (
      <p className="faint" style={{ margin: 0 }}>
        The engine reports no directly decidable target for this item. If a
        material fact is missing, add it with ADD VERIFIED FACT (evidence
        required).
      </p>
    );
  }
  const visual =
    entry && entry.source === "visual" ? (entry.item as VisualReviewItem) : null;
  switch (target.target_kind) {
    case "SPEECH_REGION": {
      if (!entry || entry.source !== "audio") {
        return (
          <p className="faint" style={{ margin: 0 }}>
            Select the speech review item for region{" "}
            <span className="mono">{target.subject_id}</span> to review it.
          </p>
        );
      }
      return (
        <SpeechEditor
          item={entry.item as AudioReviewItem}
          store={store}
          onResolved={onResolved}
        />
      );
    }
    case "TEXT_TRACK":
      return (
        <OcrEditor
          trackId={target.subject_id}
          currentFrame={currentFrame}
          store={store}
          onResolved={onResolved}
        />
      );
    case "ENTITY_TRACK":
      return (
        <IdentityEditor
          subjectTrackId={target.subject_id}
          candidates={[]}
          store={store}
          onResolved={onResolved}
        />
      );
    case "ACTION_CANDIDATE":
      return (
        <ActionEditor
          candidateId={target.subject_id}
          startFrame={visual?.start_frame ?? currentFrame}
          endFrame={visual?.end_frame ?? currentFrame}
          currentFrame={currentFrame}
          store={store}
          onResolved={onResolved}
        />
      );
    case "CAMERA_EVENT":
      return (
        <CameraEditor
          candidateId={target.subject_id}
          store={store}
          onResolved={onResolved}
        />
      );
    case "SHOT_TRANSITION": {
      const shotIndex = transitionShotIndex(target.subject_id);
      if (shotIndex === null) {
        return (
          <p className="faint" style={{ margin: 0 }}>
            Unrecognized transition subject <span className="mono">{target.subject_id}</span>.
          </p>
        );
      }
      return <TransitionEditor shotIndex={shotIndex} store={store} onResolved={onResolved} />;
    }
    case "PLAYBACK_SPEED": {
      const shotNumber =
        (target as { shot_number?: number }).shot_number ??
        transitionShotIndex(target.subject_id) ??
        1;
      return (
        <SpeedEditor
          shotNumber={shotNumber}
          subjectId={target.subject_id}
          store={store}
          onResolved={onResolved}
        />
      );
    }
    case "REVIEW_PROPOSAL":
    case "SEED_CLAIM":
      return (
        <ProposalEditor
          proposalId={target.subject_id}
          proposalKind={visual?.recommended_action ?? target.target_kind}
          reasonCodes={[]}
          store={store}
          onResolved={onResolved}
        />
      );
    default:
      return (
        <p className="faint" style={{ margin: 0 }}>
          No editor is available for target kind {target.target_kind}.
        </p>
      );
  }
}
