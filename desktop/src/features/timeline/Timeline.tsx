/**
 * Compact multi-lane timeline. Positions are CSS-percent projections of the
 * engine's exact values — display geometry only, never re-stored as facts.
 * Clicking is a navigation aid (context seeks / frame jumps), not a decision.
 */

import { useCallback } from "react";

import { manuscriptDisplay, parseExactSeconds } from "../../lib/exactTime";

export type TimelineBlockKind =
  | "shot"
  | "flag"
  | "speech"
  | "text"
  | "camera"
  | "speed"
  | "action";

export interface TimelineBlock {
  id: string;
  startSeconds: number;
  endSeconds: number;
  label?: string;
  kind: TimelineBlockKind;
}

export interface TimelineLane {
  id: string;
  label: string;
  blocks: TimelineBlock[];
}

export interface TimelineProps {
  /** Exact "num/den" seconds string for the media duration. */
  durationExact: string;
  lanes: TimelineLane[];
  onSeekFrame?: (frameIndex: number) => void;
  onSeekSeconds?: (seconds: number) => void;
  frameForSeconds?: (seconds: number) => number;
}

const KIND_COLOR: Record<TimelineBlockKind, string> = {
  shot: "var(--accent)",
  flag: "var(--status-fail)",
  speech: "var(--status-pass)",
  text: "var(--human-corrected)",
  camera: "var(--machine)",
  speed: "var(--status-review)",
  action: "var(--accent-strong)",
};

const LANE_HEIGHT = 20;
const LABEL_WIDTH = 96;

export function Timeline({
  durationExact,
  lanes,
  onSeekFrame,
  onSeekSeconds,
  frameForSeconds,
}: TimelineProps) {
  const duration = parseExactSeconds(durationExact);
  const valid = Number.isFinite(duration) && duration > 0;

  const seekTo = useCallback(
    (seconds: number) => {
      const clamped = Math.min(Math.max(seconds, 0), valid ? duration : 0);
      if (onSeekSeconds !== undefined) onSeekSeconds(clamped);
      if (onSeekFrame !== undefined && frameForSeconds !== undefined) {
        onSeekFrame(frameForSeconds(clamped));
      }
    },
    [duration, valid, onSeekSeconds, onSeekFrame, frameForSeconds],
  );

  const onTrackClick = useCallback(
    (event: React.MouseEvent<HTMLDivElement>) => {
      if (!valid) return;
      const rect = event.currentTarget.getBoundingClientRect();
      if (rect.width <= 0) return;
      const fraction = (event.clientX - rect.left) / rect.width;
      seekTo(fraction * duration);
    },
    [valid, duration, seekTo],
  );

  if (!valid) {
    return (
      <div className="panel faint" role="note">
        Timeline unavailable: duration not readable ({durationExact}).
      </div>
    );
  }

  return (
    <div className="col panel" role="group" aria-label="Run timeline (navigation aid)">
      <div className="row">
        <span className="muted">Timeline</span>
        <span className="faint mono">0s – {manuscriptDisplay(durationExact)}</span>
        <span className="faint">Click to navigate. Positions are display projections.</span>
      </div>
      <div className="col" style={{ gap: 2 }}>
        {lanes.map((lane) => (
          <div key={lane.id} className="row" style={{ gap: 4 }}>
            <span
              className="faint mono"
              style={{
                width: LABEL_WIDTH,
                flex: `0 0 ${LABEL_WIDTH}px`,
                fontSize: 11,
                overflow: "hidden",
                textOverflow: "ellipsis",
                whiteSpace: "nowrap",
              }}
              title={lane.label}
            >
              {lane.label}
            </span>
            <div
              onClick={onTrackClick}
              role="presentation"
              style={{
                position: "relative",
                flex: 1,
                height: LANE_HEIGHT,
                background: "var(--bg-inset)",
                border: "1px solid var(--border)",
                borderRadius: "var(--radius)",
                cursor: "pointer",
                overflow: "hidden",
              }}
            >
              {lane.blocks.map((block) => {
                const start = Math.min(Math.max(block.startSeconds, 0), duration);
                const end = Math.min(Math.max(block.endSeconds, start), duration);
                const left = (start / duration) * 100;
                const width = Math.max(((end - start) / duration) * 100, 0.25);
                const tooltip = `${block.label ?? block.kind} — ${start.toFixed(2)}s to ${end.toFixed(2)}s`;
                return (
                  <button
                    key={block.id}
                    type="button"
                    title={tooltip}
                    aria-label={`Go to ${tooltip}`}
                    onClick={(event) => {
                      event.stopPropagation();
                      seekTo(start);
                    }}
                    style={{
                      position: "absolute",
                      left: `${left}%`,
                      width: `${width}%`,
                      top: 2,
                      bottom: 2,
                      padding: 0,
                      border: "none",
                      borderRadius: 2,
                      background: KIND_COLOR[block.kind],
                      opacity: 0.85,
                      overflow: "hidden",
                      fontSize: 9,
                      lineHeight: 1,
                      color: "#0c0d10",
                      cursor: "pointer",
                    }}
                  >
                    {block.label !== undefined && block.label.length > 0 ? block.label : null}
                  </button>
                );
              })}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
