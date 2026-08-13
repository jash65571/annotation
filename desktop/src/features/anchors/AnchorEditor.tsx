/**
 * Visual-anchor editor: the reviewer drags a rectangle over an exact ledger
 * frame to anchor an entity in SOURCE pixel coordinates. The human draws;
 * only human-drawn geometry is saved. All math goes through the pure coords
 * module so display letterboxing can never leak into stored coordinates.
 */

import { useCallback, useEffect, useRef, useState } from "react";

import { getExactFrame, readRunFile } from "../../api/bridge";
import { contentRect, displayToSource, type Point, type Rect } from "./coords";

export type AnchorEntityType = "character" | "object";

export interface AnchorDraft {
  frame_index: number;
  x: number;
  y: number;
  width: number;
  height: number;
  entity_type: AnchorEntityType;
  label: string;
}

export interface AnchorEditorProps {
  runDir: string;
  frameIndex: number;
  sourceWidth: number;
  sourceHeight: number;
  onSaveAnchor: (anchor: AnchorDraft) => void;
}

interface DragState {
  start: Point;
  current: Point;
}

/** Clamp a display point into the rendered content rect so drags that stray
 * into the letterbox still resolve to a valid source pixel. */
function clampToContent(px: Point, content: Rect): Point {
  return {
    x: Math.min(Math.max(px.x, content.x), content.x + content.width),
    y: Math.min(Math.max(px.y, content.y), content.y + content.height),
  };
}

export function AnchorEditor({
  runDir,
  frameIndex,
  sourceWidth,
  sourceHeight,
  onSaveAnchor,
}: AnchorEditorProps) {
  const surfaceRef = useRef<HTMLDivElement | null>(null);
  const [frameUrl, setFrameUrl] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [drag, setDrag] = useState<DragState | null>(null);
  const [sourceRect, setSourceRect] = useState<Rect | null>(null);
  const [displayRect, setDisplayRect] = useState<Rect | null>(null);
  const [entityType, setEntityType] = useState<AnchorEntityType>("character");
  const [label, setLabel] = useState("");

  useEffect(() => {
    let cancelled = false;
    setFrameUrl(null);
    setError(null);
    setDrag(null);
    setSourceRect(null);
    setDisplayRect(null);
    (async () => {
      try {
        const payload = await getExactFrame(runDir, frameIndex);
        const dataUrl = await readRunFile(runDir, payload.path);
        if (!cancelled) setFrameUrl(dataUrl);
      } catch (raw) {
        if (!cancelled) setError(String(raw));
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [runDir, frameIndex]);

  const localPoint = useCallback((event: MouseEvent | React.MouseEvent): Point | null => {
    const surface = surfaceRef.current;
    if (surface === null) return null;
    const rect = surface.getBoundingClientRect();
    return { x: event.clientX - rect.left, y: event.clientY - rect.top };
  }, []);

  const surfaceSize = useCallback((): { width: number; height: number } | null => {
    const surface = surfaceRef.current;
    if (surface === null) return null;
    const rect = surface.getBoundingClientRect();
    if (rect.width <= 0 || rect.height <= 0) return null;
    return { width: rect.width, height: rect.height };
  }, []);

  const finishDrag = useCallback(
    (state: DragState) => {
      const display = surfaceSize();
      if (display === null) return;
      const source = { width: sourceWidth, height: sourceHeight };
      const content = contentRect(display, source);
      if (content === null) return;

      const a = displayToSource(clampToContent(state.start, content), display, source);
      const b = displayToSource(clampToContent(state.current, content), display, source);
      if (a === null || b === null) return;

      const x = Math.min(a.x, b.x);
      const y = Math.min(a.y, b.y);
      const width = Math.abs(a.x - b.x) + 1;
      const height = Math.abs(a.y - b.y) + 1;
      setSourceRect({ x, y, width, height });
      setDisplayRect({
        x: Math.min(state.start.x, state.current.x),
        y: Math.min(state.start.y, state.current.y),
        width: Math.abs(state.start.x - state.current.x),
        height: Math.abs(state.start.y - state.current.y),
      });
    },
    [surfaceSize, sourceWidth, sourceHeight],
  );

  const onMouseDown = useCallback(
    (event: React.MouseEvent<HTMLDivElement>) => {
      if (event.button !== 0 || frameUrl === null) return;
      const point = localPoint(event);
      if (point === null) return;
      event.preventDefault();
      setSourceRect(null);
      setDisplayRect(null);
      setDrag({ start: point, current: point });
    },
    [frameUrl, localPoint],
  );

  // Track the drag on the window so leaving the surface does not lose it.
  useEffect(() => {
    if (drag === null) return;
    const onMove = (event: MouseEvent) => {
      const point = localPoint(event);
      if (point === null) return;
      setDrag((previous) =>
        previous === null ? null : { start: previous.start, current: point },
      );
    };
    const onUp = (event: MouseEvent) => {
      const point = localPoint(event);
      setDrag((previous) => {
        if (previous !== null) {
          finishDrag({
            start: previous.start,
            current: point ?? previous.current,
          });
        }
        return null;
      });
    };
    window.addEventListener("mousemove", onMove);
    window.addEventListener("mouseup", onUp);
    return () => {
      window.removeEventListener("mousemove", onMove);
      window.removeEventListener("mouseup", onUp);
    };
  }, [drag !== null, localPoint, finishDrag]);

  const liveRect: Rect | null =
    drag !== null
      ? {
          x: Math.min(drag.start.x, drag.current.x),
          y: Math.min(drag.start.y, drag.current.y),
          width: Math.abs(drag.start.x - drag.current.x),
          height: Math.abs(drag.start.y - drag.current.y),
        }
      : displayRect;

  const trimmedLabel = label.trim();
  const canSave = sourceRect !== null && trimmedLabel.length > 0;

  const save = useCallback(() => {
    if (sourceRect === null || trimmedLabel.length === 0) return;
    onSaveAnchor({
      frame_index: frameIndex,
      x: sourceRect.x,
      y: sourceRect.y,
      width: sourceRect.width,
      height: sourceRect.height,
      entity_type: entityType,
      label: trimmedLabel,
    });
  }, [sourceRect, trimmedLabel, entityType, frameIndex, onSaveAnchor]);

  return (
    <div className="col panel" role="group" aria-label="Visual anchor editor">
      <div className="row">
        <span className="muted">Anchor editor — F{frameIndex}</span>
        <span className="faint">
          Drag a box over the entity. Coordinates are saved in SOURCE pixels.
        </span>
      </div>

      <div
        ref={surfaceRef}
        onMouseDown={onMouseDown}
        role="presentation"
        style={{
          position: "relative",
          width: "100%",
          aspectRatio: `${sourceWidth} / ${sourceHeight}`,
          background: "var(--bg-inset)",
          cursor: frameUrl !== null ? "crosshair" : "default",
          userSelect: "none",
          overflow: "hidden",
        }}
      >
        {frameUrl !== null ? (
          <img
            src={frameUrl}
            alt={`Exact ledger frame ${frameIndex} for anchoring`}
            draggable={false}
            style={{
              position: "absolute",
              inset: 0,
              width: "100%",
              height: "100%",
              objectFit: "contain",
              pointerEvents: "none",
            }}
          />
        ) : (
          <div className="faint" style={{ padding: 24 }}>
            {error !== null ? `Frame unavailable: ${error}` : "Loading exact frame…"}
          </div>
        )}
        {liveRect !== null && liveRect.width > 0 && liveRect.height > 0 ? (
          <div
            style={{
              position: "absolute",
              left: liveRect.x,
              top: liveRect.y,
              width: liveRect.width,
              height: liveRect.height,
              border: "2px solid var(--accent-strong)",
              background: "rgba(110, 168, 224, 0.15)",
              pointerEvents: "none",
            }}
          />
        ) : null}
      </div>

      <div className="row">
        <span className="mono muted">
          {sourceRect !== null
            ? `source px: x=${sourceRect.x} y=${sourceRect.y} w=${sourceRect.width} h=${sourceRect.height}`
            : "source px: — (drag to define)"}
        </span>
      </div>

      <div className="row">
        <label className="row" style={{ gap: 4 }}>
          <span className="muted">Type</span>
          <select
            value={entityType}
            onChange={(event) => setEntityType(event.target.value as AnchorEntityType)}
            aria-label="Anchor entity type"
          >
            <option value="character">character</option>
            <option value="object">object</option>
          </select>
        </label>
        <label className="row" style={{ gap: 4, flex: 1 }}>
          <span className="muted">Label</span>
          <input
            type="text"
            value={label}
            onChange={(event) => setLabel(event.target.value)}
            placeholder="temporary label (e.g. red-shirt person)"
            aria-label="Anchor temporary label"
            style={{ flex: 1 }}
          />
        </label>
        <button
          type="button"
          className="primary"
          onClick={save}
          disabled={!canSave}
          aria-label="Save visual anchor"
        >
          Save anchor
        </button>
      </div>
    </div>
  );
}
