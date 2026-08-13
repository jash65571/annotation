/**
 * Merged review queue: visual + audio review items in one prioritized list.
 *
 * Sorting: CRITICAL first, then HIGH, then the rest — within each priority
 * group, timeline order (start_exact when present, else start_frame).
 * Skipping is a local visual state only: a skipped item is NOT resolved and
 * keeps the caption in REVIEW REQUIRED.
 */

import { useMemo, useState, type KeyboardEvent, type ReactElement } from "react";
import type {
  AudioReviewItem,
  ReviewPriority,
  ReviewQueuePayload,
  VisualReviewItem,
} from "../../api/types";

export type QueueEntry = {
  id: string;
  source: "visual" | "audio";
  priority: ReviewPriority;
  title: string;
  reason: string;
  item: VisualReviewItem | AudioReviewItem;
};

export interface QueuePanelProps {
  queue: ReviewQueuePayload;
  resolvedSubjects: Set<string>;
  selectedId: string | null;
  onSelect: (item: QueueEntry) => void;
}

const PRIORITY_RANK: Record<ReviewPriority, number> = {
  CRITICAL: 0,
  HIGH: 1,
  NORMAL: 2,
  LOW: 3,
};

const PRIORITY_FILTERS: Array<{ label: string; value: ReviewPriority | "ALL" }> = [
  { label: "All", value: "ALL" },
  { label: "Critical", value: "CRITICAL" },
  { label: "High", value: "HIGH" },
  { label: "Normal", value: "NORMAL" },
  { label: "Low", value: "LOW" },
];

/** Parse an exact "num/den" rational string to seconds for ordering only. */
function parseExactSeconds(exact: string | null | undefined): number | null {
  if (exact == null) return null;
  const match = /^(-?\d+)\/(\d+)$/.exec(exact.trim());
  if (match) {
    const num = Number(match[1]);
    const den = Number(match[2]);
    if (Number.isFinite(num) && Number.isFinite(den) && den !== 0) return num / den;
    return null;
  }
  const plain = Number(exact);
  return Number.isFinite(plain) ? plain : null;
}

function timelineKey(entry: QueueEntry): number {
  const item = entry.item as { start_exact?: string | null; start_frame?: number | null };
  const seconds = parseExactSeconds(item.start_exact);
  if (seconds != null) return seconds;
  if (item.start_frame != null) return item.start_frame;
  return Number.POSITIVE_INFINITY;
}

export function buildQueueEntries(queue: ReviewQueuePayload): QueueEntry[] {
  const visual: QueueEntry[] = queue.visual_items.map((item) => ({
    id: item.item_id,
    source: "visual",
    priority: item.priority,
    title: item.title,
    reason: item.reason,
    item,
  }));
  const audio: QueueEntry[] = queue.audio_items.map((item) => ({
    id: item.item_id,
    source: "audio",
    priority: item.priority,
    title: `Speech review ${item.item_id}`,
    reason: item.reasons.join(", "),
    item,
  }));
  return [...visual, ...audio].sort((a, b) => {
    const rank = PRIORITY_RANK[a.priority] - PRIORITY_RANK[b.priority];
    if (rank !== 0) return rank;
    return timelineKey(a) - timelineKey(b);
  });
}

function statusBadgeClass(status: "OPEN" | "RESOLVED" | "SKIPPED"): string {
  if (status === "RESOLVED") return "badge human";
  if (status === "SKIPPED") return "badge unresolved";
  return "badge neutral";
}

export function QueuePanel({
  queue,
  resolvedSubjects,
  selectedId,
  onSelect,
}: QueuePanelProps): ReactElement {
  const [priorityFilter, setPriorityFilter] = useState<ReviewPriority | "ALL">("ALL");
  const [categoryFilter, setCategoryFilter] = useState("");
  const [skipped, setSkipped] = useState<ReadonlySet<string>>(new Set());

  const entries = useMemo(() => buildQueueEntries(queue), [queue]);

  const visible = useMemo(() => {
    const needle = categoryFilter.trim().toLowerCase();
    return entries.filter((entry) => {
      if (priorityFilter !== "ALL" && entry.priority !== priorityFilter) return false;
      if (needle === "") return true;
      return (
        entry.title.toLowerCase().includes(needle) ||
        entry.reason.toLowerCase().includes(needle)
      );
    });
  }, [entries, priorityFilter, categoryFilter]);

  const completed = entries.filter((entry) => resolvedSubjects.has(entry.id)).length;
  const remaining = entries.length - completed;
  const anySkipped = entries.some(
    (entry) => skipped.has(entry.id) && !resolvedSubjects.has(entry.id),
  );

  function toggleSkip(id: string): void {
    setSkipped((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  }

  function handleKeyDown(event: KeyboardEvent<HTMLDivElement>): void {
    const key = event.key.toLowerCase();
    if (key !== "j" && key !== "k") return;
    if (visible.length === 0) return;
    event.preventDefault();
    const currentIndex = visible.findIndex((entry) => entry.id === selectedId);
    let nextIndex: number;
    if (currentIndex === -1) {
      nextIndex = 0;
    } else if (key === "j") {
      nextIndex = Math.min(currentIndex + 1, visible.length - 1);
    } else {
      nextIndex = Math.max(currentIndex - 1, 0);
    }
    const next = visible[nextIndex];
    if (next) onSelect(next);
  }

  return (
    <div
      className="col panel"
      tabIndex={0}
      role="region"
      aria-label="Review queue (J/K to move selection)"
      onKeyDown={handleKeyDown}
    >
      <div className="row" role="group" aria-label="Priority filter">
        {PRIORITY_FILTERS.map((filter) => (
          <button
            key={filter.value}
            type="button"
            aria-pressed={priorityFilter === filter.value}
            className={priorityFilter === filter.value ? "primary" : ""}
            onClick={() => setPriorityFilter(filter.value)}
          >
            {filter.label}
          </button>
        ))}
      </div>
      <label className="row">
        <span className="muted">Filter</span>
        <input
          type="text"
          value={categoryFilter}
          aria-label="Category text filter"
          placeholder="Filter by title or reason"
          onChange={(event) => setCategoryFilter(event.target.value)}
        />
      </label>
      <div className="muted" aria-live="polite">
        {completed} completed · {remaining} remaining
      </div>
      <ul className="col" style={{ listStyle: "none", margin: 0, padding: 0 }}>
        {visible.map((entry) => {
          const isResolved = resolvedSubjects.has(entry.id);
          const isSkipped = !isResolved && skipped.has(entry.id);
          const status: "OPEN" | "RESOLVED" | "SKIPPED" = isResolved
            ? "RESOLVED"
            : isSkipped
              ? "SKIPPED"
              : "OPEN";
          const isSelected = selectedId === entry.id;
          return (
            <li key={entry.id} className="row" data-status={status}>
              <button
                type="button"
                className="row"
                aria-pressed={isSelected}
                aria-label={`Open ${entry.title}`}
                onClick={() => onSelect(entry)}
                style={{
                  flex: 1,
                  justifyContent: "flex-start",
                  textAlign: "left",
                  opacity: isSkipped ? 0.55 : 1,
                  borderColor: isSelected ? "var(--accent)" : undefined,
                  textDecoration: isSkipped ? "line-through" : undefined,
                }}
              >
                <span
                  className={`badge ${
                    entry.priority === "CRITICAL" || entry.priority === "HIGH"
                      ? "fail"
                      : "neutral"
                  }`}
                >
                  {entry.priority}
                </span>
                <span>{entry.title}</span>
                <span className="muted">{entry.reason}</span>
                <span className={statusBadgeClass(status)}>{status}</span>
              </button>
              <button
                type="button"
                aria-label={isSkipped ? `Unskip ${entry.title}` : `Skip ${entry.title}`}
                disabled={isResolved}
                onClick={() => toggleSkip(entry.id)}
              >
                {isSkipped ? "Unskip" : "Skip"}
              </button>
            </li>
          );
        })}
      </ul>
      {visible.length === 0 && <p className="muted">No queue items match the filters.</p>}
      {anySkipped && (
        <p className="muted">Skipped items keep the caption in REVIEW REQUIRED.</p>
      )}
    </div>
  );
}
