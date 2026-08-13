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
  getReviewInputs,
  getReviewQueue,
  getShots,
  saveVisualAnchors,
} from "../api/bridge";
import type {
  AudioReviewItem,
  CaptionStatePayload,
  HumanReviewDecision,
  Json,
  ReviewQueuePayload,
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
type EditorKind =
  | "speech"
  | "transition"
  | "speed"
  | "ocr"
  | "identity"
  | "action"
  | "camera"
  | "proposal"
  | "none";

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

/** Pick the default editor for a queue entry from engine signals only. */
function editorKindFor(entry: QueueEntry): EditorKind {
  if (entry.source === "audio") return "speech";
  const item = entry.item as VisualReviewItem;
  const title = item.title.toLowerCase();
  if (title.includes("transition")) return "transition";
  if (title.includes("speed")) return "speed";
  if (title.includes("text") || title.includes("ocr")) return "ocr";
  if (title.includes("identity") || item.recommended_action === "CHOOSE_IDENTITY")
    return "identity";
  if (title.includes("camera")) return "camera";
  if (title.includes("action") || item.recommended_action === "CORRECT_BOUNDARY")
    return "action";
  if (
    item.recommended_action === "REBUILD_SECTION" ||
    item.recommended_action === "CONFIRM_CLAIM" ||
    item.recommended_action === "REJECT_CLAIM"
  )
    return "proposal";
  return "none";
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
  const [selected, setSelected] = useState<QueueEntry | null>(null);
  const [editorKind, setEditorKind] = useState<EditorKind>("none");
  const [mediaMode, setMediaMode] = useState<MediaMode>("playback");
  const [currentFrame, setCurrentFrame] = useState(0);
  const [seekSeconds, setSeekSeconds] = useState<number | undefined>(undefined);
  const [resolvedSubjects, setResolvedSubjects] = useState<Set<string>>(new Set());
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
      setResolvedSubjects(new Set(store.getDecisions().map((d) => d.subject_id)));
    });
  }, [store]);

  const reload = useCallback(async () => {
    if (!runDir) return;
    const [q, c, s] = await Promise.all([
      getReviewQueue(runDir),
      getCaptionState(runDir),
      getShots(runDir),
    ]);
    setQueue(q);
    setCaption(c);
    setShots(extractShots(s as Json));
    lastDraftRef.current = c.draft_markdown ?? c.ready_markdown ?? "";
    if (store) {
      try {
        const inputs = await getReviewInputs(runDir);
        store.seed(inputs.decisions, inputs.facts);
      } catch {
        /* fresh run without saved inputs */
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
    },
    [dispatch],
  );

  const selectEntry = useCallback((entry: QueueEntry) => {
    setSelected(entry);
    setEditorKind(editorKindFor(entry));
    if (entry.source === "audio") {
      setMediaMode("audio");
    } else {
      const item = entry.item as VisualReviewItem;
      if (item.start_frame != null) {
        setCurrentFrame(item.start_frame);
        setMediaMode("frame");
      }
    }
  }, []);

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
        .filter((entry) => !resolvedSubjects.has(entry.id))
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
    store && selected
      ? (store
          .getDecisions()
          .find((d) => d.subject_id === subjectIdFor(selected)) as HumanReviewDecision | null)
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
              <AnchorEditor
                runDir={runDir}
                frameIndex={currentFrame}
                sourceWidth={1920}
                sourceHeight={1080}
                onSaveAnchor={(anchor) => {
                  const next = [...anchors, anchor];
                  setAnchors(next);
                  saveVisualAnchors(runDir, next)
                    .then(() => setSaveIndicator("Anchors saved"))
                    .catch((e) => setError(String(e)));
                }}
              />
              <span className="faint">
                {anchors.length} anchor(s) saved to this run. Re-running visual intelligence
                with anchors requires a new analysis pass (the locked engine has no partial
                rerun yet).
              </span>
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
              resolvedSubjects={resolvedSubjects}
              selectedId={selected?.id ?? null}
              onSelect={selectEntry}
            />
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
          {store && selected && (
            <div className="panel col">
              <div className="row" style={{ justifyContent: "space-between" }}>
                <h2 style={{ fontSize: 13, margin: 0 }}>Decision</h2>
                <select
                  aria-label="Editor type"
                  value={editorKind}
                  onChange={(e) => setEditorKind(e.target.value as EditorKind)}
                >
                  {(
                    [
                      "none",
                      "speech",
                      "transition",
                      "speed",
                      "ocr",
                      "identity",
                      "action",
                      "camera",
                      "proposal",
                    ] as EditorKind[]
                  ).map((kind) => (
                    <option key={kind} value={kind}>
                      {kind}
                    </option>
                  ))}
                </select>
              </div>
              <EditorSwitch
                kind={editorKind}
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

function subjectIdFor(entry: QueueEntry): string {
  if (entry.source === "audio") {
    const item = entry.item as AudioReviewItem;
    return (item.evidence_refs ?? []).find((ref) => ref.startsWith("speech_")) ?? entry.id;
  }
  return entry.id;
}

function EditorSwitch({
  kind,
  entry,
  store,
  currentFrame,
  onOutcome,
}: {
  kind: EditorKind;
  entry: QueueEntry;
  store: DecisionsStore;
  currentFrame: number;
  onOutcome: (outcome: FinalizeOutcome) => void;
}) {
  const onResolved = (outcome: unknown) => onOutcome(outcome as FinalizeOutcome);
  const visual = entry.source === "visual" ? (entry.item as VisualReviewItem) : null;
  const shotNumber = visual?.shot_number ?? shotFromTitle(visual?.title) ?? 1;

  switch (kind) {
    case "speech":
      if (entry.source !== "audio") return <NoEditor />;
      return (
        <SpeechEditor
          item={entry.item as AudioReviewItem}
          store={store}
          onResolved={onResolved}
        />
      );
    case "transition":
      return <TransitionEditor shotIndex={shotNumber} store={store} onResolved={onResolved} />;
    case "speed":
      return (
        <SpeedEditor
          shotNumber={shotNumber}
          subjectId={`SPEED-${shotNumber}`}
          store={store}
          onResolved={onResolved}
        />
      );
    case "ocr":
      return (
        <OcrEditor
          trackId={firstRelatedId(visual) ?? ""}
          currentFrame={currentFrame}
          store={store}
          onResolved={onResolved}
        />
      );
    case "identity":
      return (
        <IdentityEditor
          subjectTrackId={firstRelatedId(visual) ?? ""}
          candidates={[]}
          store={store}
          onResolved={onResolved}
        />
      );
    case "action":
      return (
        <ActionEditor
          candidateId={firstRelatedId(visual) ?? ""}
          startFrame={visual?.start_frame ?? currentFrame}
          endFrame={visual?.end_frame ?? currentFrame}
          currentFrame={currentFrame}
          store={store}
          onResolved={onResolved}
        />
      );
    case "camera":
      return (
        <CameraEditor
          candidateId={firstRelatedId(visual) ?? ""}
          store={store}
          onResolved={onResolved}
        />
      );
    case "proposal":
      return (
        <ProposalEditor
          proposalId={firstRelatedId(visual) ?? entry.id}
          proposalKind={visual?.recommended_action ?? ""}
          reasonCodes={[]}
          store={store}
          onResolved={onResolved}
        />
      );
    case "none":
      return <NoEditor />;
  }
}

function NoEditor() {
  return (
    <p className="faint" style={{ margin: 0 }}>
      Select an editor type for this item, or resolve it via a verified fact.
    </p>
  );
}

function shotFromTitle(title: string | undefined): number | null {
  if (!title) return null;
  const match = /shot\s+(\d+)/i.exec(title);
  const value = match?.[1];
  return value !== undefined ? Number(value) : null;
}

function firstRelatedId(item: VisualReviewItem | null): string | null {
  if (!item) return null;
  const related = item.related_claim_ids?.[0];
  if (related) return related;
  const ref = item.supporting_evidence_refs?.[0]?.evidence_id;
  return ref ?? null;
}
