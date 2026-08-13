import { useState } from "react";
import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import type {
  AudioReviewItem,
  ResolutionStatus,
  ReviewQueuePayload,
  VisualReviewItem,
} from "../api/types";
import { QueuePanel } from "../features/review/QueuePanel";

function visualItem(overrides: Partial<VisualReviewItem>): VisualReviewItem {
  return {
    item_id: "V-X",
    priority: "NORMAL",
    title: "Visual item",
    reason: "needs review",
    recommended_action: "VERIFY",
    ...overrides,
  };
}

function audioItem(overrides: Partial<AudioReviewItem>): AudioReviewItem {
  return {
    item_id: "A-X",
    reasons: ["low confidence"],
    priority: "HIGH",
    start_exact: "0/1",
    end_exact: "1/1",
    playback_start: "0/1",
    playback_end: "1/1",
    ...overrides,
  };
}

const queue: ReviewQueuePayload = {
  visual_items: [
    visualItem({
      item_id: "V-NORMAL",
      title: "Normal visual",
      priority: "NORMAL",
      start_exact: "1/1",
      start_frame: 30,
    }),
    visualItem({
      item_id: "V-CRIT",
      title: "Critical visual",
      priority: "CRITICAL",
      start_exact: "10/1",
      start_frame: 300,
    }),
    visualItem({
      item_id: "V-HIGH",
      title: "High visual",
      priority: "HIGH",
      start_exact: "8/1",
      start_frame: 240,
    }),
  ],
  audio_items: [
    audioItem({
      item_id: "A-HIGH",
      priority: "HIGH",
      start_exact: "2/1",
      end_exact: "3/1",
      reasons: ["low confidence", "overlap"],
    }),
  ],
};

const noResolution = new Map<string, ResolutionStatus>();

function entryButtons(): HTMLElement[] {
  return screen
    .getAllByRole("listitem")
    .map((li) => within(li).getAllByRole("button")[0] as HTMLElement);
}

/** Parent-owned skipped state, as the ReviewScreen persists it via ui_state. */
function SkippablePanel({
  resolutionById,
}: {
  resolutionById: ReadonlyMap<string, ResolutionStatus>;
}) {
  const [skipped, setSkipped] = useState<ReadonlySet<string>>(new Set());
  return (
    <QueuePanel
      queue={queue}
      resolutionById={resolutionById}
      selectedId={null}
      onSelect={vi.fn()}
      skippedIds={skipped}
      onToggleSkip={(id) =>
        setSkipped((prev) => {
          const next = new Set(prev);
          if (next.has(id)) next.delete(id);
          else next.add(id);
          return next;
        })
      }
    />
  );
}

describe("QueuePanel", () => {
  it("orders CRITICAL first, then HIGH by timeline, then the rest", () => {
    render(
      <QueuePanel
        queue={queue}
        resolutionById={noResolution}
        selectedId={null}
        onSelect={vi.fn()}
      />,
    );
    const titles = entryButtons().map((button) => button.textContent ?? "");
    expect(titles[0]).toContain("Critical visual");
    // HIGH group in timeline order: audio at 2s before visual at 8s.
    expect(titles[1]).toContain("Speech review A-HIGH");
    expect(titles[2]).toContain("High visual");
    expect(titles[3]).toContain("Normal visual");
  });

  it("builds audio entries with the Speech review title and joined reasons", () => {
    render(
      <QueuePanel
        queue={queue}
        resolutionById={noResolution}
        selectedId={null}
        onSelect={vi.fn()}
      />,
    );
    expect(screen.getByText("Speech review A-HIGH")).toBeInTheDocument();
    expect(screen.getByText("low confidence, overlap")).toBeInTheDocument();
  });

  it("filters by priority chip", async () => {
    const user = userEvent.setup();
    render(
      <QueuePanel
        queue={queue}
        resolutionById={noResolution}
        selectedId={null}
        onSelect={vi.fn()}
      />,
    );
    await user.click(screen.getByRole("button", { name: "Critical" }));
    const titles = entryButtons().map((button) => button.textContent ?? "");
    expect(titles).toHaveLength(1);
    expect(titles[0]).toContain("Critical visual");
    expect(screen.queryByText("Normal visual")).not.toBeInTheDocument();
  });

  it("shows RESOLVED from ENGINE resolution even when the decision subject id differs from the item id", () => {
    // The engine resolved item V-CRIT via a decision on subject SPEED-2 —
    // the map carries item-id → status, never local decision subjects.
    const resolution = new Map<string, ResolutionStatus>([["V-CRIT", "RESOLVED"]]);
    render(
      <QueuePanel
        queue={queue}
        resolutionById={resolution}
        selectedId={null}
        onSelect={vi.fn()}
      />,
    );
    expect(screen.getByText("RESOLVED")).toBeInTheDocument();
    expect(screen.getByText("1 completed · 3 remaining")).toBeInTheDocument();
  });

  it("keeps items OPEN when the engine reports no resolution", () => {
    // A local decision that did NOT apply (stale/invalid) never flips status:
    // the map from engine truth simply has no RESOLVED entry.
    render(
      <QueuePanel
        queue={queue}
        resolutionById={noResolution}
        selectedId={null}
        onSelect={vi.fn()}
      />,
    );
    expect(screen.queryByText("RESOLVED")).not.toBeInTheDocument();
    expect(screen.getByText("0 completed · 4 remaining")).toBeInTheDocument();
  });

  it("marks skipped items distinctly and keeps them not-resolved", async () => {
    const user = userEvent.setup();
    const resolution = new Map<string, ResolutionStatus>([["V-CRIT", "RESOLVED"]]);
    render(<SkippablePanel resolutionById={resolution} />);
    await user.click(screen.getByRole("button", { name: "Skip High visual" }));
    expect(screen.getByText("SKIPPED")).toBeInTheDocument();
    // Skipping never resolves: completed count unchanged, note shown.
    expect(screen.getByText("1 completed · 3 remaining")).toBeInTheDocument();
    expect(
      screen.getByText("Skipped items keep the caption in REVIEW REQUIRED."),
    ).toBeInTheDocument();
    const resolvedBadge = screen.getByText("RESOLVED");
    const skippedBadge = screen.getByText("SKIPPED");
    expect(resolvedBadge.className).not.toEqual(skippedBadge.className);
  });

  it("selects with J/K when the panel has focus", async () => {
    const user = userEvent.setup();
    const onSelect = vi.fn();
    render(
      <QueuePanel
        queue={queue}
        resolutionById={noResolution}
        selectedId={"V-CRIT"}
        onSelect={onSelect}
      />,
    );
    const region = screen.getByRole("region", {
      name: "Review queue (J/K to move selection)",
    });
    region.focus();
    await user.keyboard("j");
    expect(onSelect).toHaveBeenCalledTimes(1);
    expect(onSelect.mock.calls[0]?.[0]?.id).toBe("A-HIGH");
  });
});
