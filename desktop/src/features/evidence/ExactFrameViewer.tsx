/**
 * Exact ledger-frame viewer: shows the engine-extracted frame for a ledger
 * index, annotated straight from the frame record. This viewer IS the frame
 * truth surface — every time shown comes from the engine's exact "num/den"
 * strings, projected for display only (BigInt decimal rendering, never a
 * float round-trip).
 */

import { useCallback, useEffect, useRef, useState } from "react";

import { getExactFrame, readRunFile } from "../../api/bridge";
import type { Json } from "../../api/types";
import { formatExactSeconds } from "../../lib/exactTime";

export interface ExactFrameViewerProps {
  runDir: string;
  frameIndex: number;
  shotNumber?: number;
  onFrameChange: (index: number) => void;
  maxFrameIndex: number;
}

interface FrameEntry {
  dataUrl: string;
  record: Json;
}

const CACHE_MAX = 15;

function lruGet(cache: Map<number, FrameEntry>, index: number): FrameEntry | undefined {
  const entry = cache.get(index);
  if (entry !== undefined) {
    // Touch: move to the end (most recently used).
    cache.delete(index);
    cache.set(index, entry);
  }
  return entry;
}

function lruPut(cache: Map<number, FrameEntry>, index: number, entry: FrameEntry): void {
  cache.delete(index);
  cache.set(index, entry);
  while (cache.size > CACHE_MAX) {
    const oldest = cache.keys().next();
    if (oldest.done === true) break;
    cache.delete(oldest.value);
  }
}

/** Pull a displayable string out of a loosely-typed record field. */
function recordText(record: Json | undefined, key: string): string | null {
  if (record === undefined) return null;
  const value = record[key];
  if (typeof value === "string" && value.length > 0) return value;
  if (typeof value === "number" && Number.isFinite(value)) return String(value);
  return null;
}

export function ExactFrameViewer({
  runDir,
  frameIndex,
  shotNumber,
  onFrameChange,
  maxFrameIndex,
}: ExactFrameViewerProps) {
  const cacheRef = useRef<Map<number, FrameEntry>>(new Map());
  const pendingRef = useRef<Map<number, Promise<FrameEntry>>>(new Map());
  const [entry, setEntry] = useState<FrameEntry | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const clampIndex = useCallback(
    (index: number) => Math.min(Math.max(index, 0), Math.max(maxFrameIndex, 0)),
    [maxFrameIndex],
  );

  const fetchFrame = useCallback(
    (index: number): Promise<FrameEntry> => {
      const cached = lruGet(cacheRef.current, index);
      if (cached !== undefined) return Promise.resolve(cached);
      const pending = pendingRef.current.get(index);
      if (pending !== undefined) return pending;
      const promise = (async () => {
        const payload = await getExactFrame(runDir, index);
        const dataUrl = await readRunFile(runDir, payload.path);
        const fresh: FrameEntry = { dataUrl, record: payload.record };
        lruPut(cacheRef.current, index, fresh);
        return fresh;
      })().finally(() => {
        pendingRef.current.delete(index);
      });
      pendingRef.current.set(index, promise);
      return promise;
    },
    [runDir],
  );

  // Reset the cache when the run changes.
  useEffect(() => {
    cacheRef.current = new Map();
    pendingRef.current = new Map();
  }, [runDir]);

  // Load the current frame, then prefetch neighbours in the background.
  useEffect(() => {
    let cancelled = false;
    const index = clampIndex(frameIndex);
    setError(null);
    setLoading(true);
    fetchFrame(index)
      .then((fresh) => {
        if (cancelled) return;
        setEntry(fresh);
        setLoading(false);
        for (const delta of [1, -1, 2, -2]) {
          const neighbour = index + delta;
          if (neighbour < 0 || neighbour > maxFrameIndex) continue;
          if (cacheRef.current.has(neighbour)) continue;
          void fetchFrame(neighbour).catch(() => undefined);
        }
      })
      .catch((raw: unknown) => {
        if (cancelled) return;
        setError(String(raw));
        setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [frameIndex, maxFrameIndex, clampIndex, fetchFrame]);

  const step = useCallback(
    (delta: number) => {
      onFrameChange(clampIndex(frameIndex + delta));
    },
    [frameIndex, clampIndex, onFrameChange],
  );

  const onKeyDown = useCallback(
    (event: React.KeyboardEvent<HTMLDivElement>) => {
      if (event.key === "ArrowLeft") {
        event.preventDefault();
        step(event.shiftKey ? -10 : -1);
      } else if (event.key === "ArrowRight") {
        event.preventDefault();
        step(event.shiftKey ? 10 : 1);
      }
    },
    [step],
  );

  const record = entry?.record;
  const ptsExact = recordText(record, "pts_time_seconds");
  const sourcePts =
    recordText(record, "pts") ??
    recordText(record, "source_pts") ??
    recordText(record, "pkt_pts");

  const annotation = [
    `F${clampIndex(frameIndex)}`,
    ptsExact !== null ? `t=${formatExactSeconds(ptsExact, 6)}s` : null,
    sourcePts !== null ? `PTS ${sourcePts}` : null,
    shotNumber !== undefined ? `Shot ${shotNumber}` : null,
  ]
    .filter((part): part is string => part !== null)
    .join("  ·  ");

  return (
    <div
      className="col panel"
      tabIndex={0}
      role="group"
      aria-label="Exact ledger frame viewer"
      onKeyDown={onKeyDown}
    >
      <div style={{ position: "relative", background: "var(--bg-inset)" }}>
        {entry !== null ? (
          <img
            src={entry.dataUrl}
            alt={`Exact ledger frame ${clampIndex(frameIndex)}`}
            style={{ display: "block", width: "100%", objectFit: "contain" }}
          />
        ) : (
          <div className="faint" style={{ padding: "24px" }}>
            {error !== null ? `Frame unavailable: ${error}` : "Loading exact frame…"}
          </div>
        )}
        <div
          className="mono"
          style={{
            position: "absolute",
            top: 4,
            left: 4,
            padding: "2px 6px",
            background: "rgba(0, 0, 0, 0.65)",
            color: "#fff",
            borderRadius: "var(--radius)",
            pointerEvents: "none",
          }}
        >
          {annotation}
        </div>
      </div>

      <div className="row">
        <button type="button" onClick={() => step(-10)} aria-label="Back 10 frames">
          −10
        </button>
        <button type="button" onClick={() => step(-1)} aria-label="Previous frame">
          ◀ prev
        </button>
        <button type="button" onClick={() => step(1)} aria-label="Next frame">
          next ▶
        </button>
        <button type="button" onClick={() => step(10)} aria-label="Forward 10 frames">
          +10
        </button>
        <span className="faint">
          Arrows step ±1 (Shift for ±10) when the viewer is focused.
        </span>
        {loading ? <span className="faint">loading…</span> : null}
        <span className="mono muted" style={{ marginLeft: "auto" }}>
          F{clampIndex(frameIndex)} / F{maxFrameIndex}
        </span>
      </div>
    </div>
  );
}
