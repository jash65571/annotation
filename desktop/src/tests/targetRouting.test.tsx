/**
 * §Phase 6.1-3/8 regressions: editor routing comes from the ENGINE decision
 * target (never titles), and privileged caption saves never carry renderer
 * text.
 */

import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi, beforeEach } from "vitest";
import { invoke } from "@tauri-apps/api/core";
import { TargetEditor, transitionShotIndex } from "../screens/ReviewScreen";
import { saveReadyCaption } from "../api/bridge";
import { DecisionsStore } from "../features/review/decisionsStore";
import type { DecisionTargetRef, VisualReviewItem } from "../api/types";
import type { QueueEntry } from "../features/review/QueuePanel";

vi.mock("../api/bridge", async () => {
  const actual =
    await vi.importActual<typeof import("../api/bridge")>("../api/bridge");
  return {
    ...actual,
    getRules: vi.fn(async () => ({
      rules_version: "1.3.0",
      rules: {
        shots: {
          allowed_transition_types: ["Opening shot", "Hard cut", "Jump cut"],
          shot_one_transition: "Opening shot",
        },
      },
    })),
  };
});

function target(kind: string, subject: string): DecisionTargetRef {
  return {
    target_kind: kind as DecisionTargetRef["target_kind"],
    subject_id: subject,
    allowed_decision_types: [],
  };
}

/** A visual queue entry whose TITLE lies about its kind on purpose. */
function misleadingEntry(): QueueEntry {
  const item: VisualReviewItem = {
    item_id: "REVIEW-123",
    priority: "HIGH",
    title: "Playback speed something", // title says speed…
    reason: "test",
    recommended_action: "VERIFY",
  };
  return { id: "REVIEW-123", source: "visual", priority: "HIGH", title: item.title, reason: "test", item };
}

const store = new DecisionsStore("C:/runs/x", "a".repeat(64), "1.3.0");

describe("engine-owned editor routing", () => {
  beforeEach(() => vi.clearAllMocks());

  it("parses the engine-defined transition subject id", () => {
    expect(transitionShotIndex("TRANSITION-3")).toBe(3);
    expect(transitionShotIndex("TRANSITION-12")).toBe(12);
    expect(transitionShotIndex("nonsense")).toBeNull();
  });

  it("routes SHOT_TRANSITION targets to the transition editor", async () => {
    render(
      <TargetEditor
        target={target("SHOT_TRANSITION", "TRANSITION-2")}
        entry={misleadingEntry()}
        store={store}
        currentFrame={0}
        onOutcome={vi.fn()}
      />,
    );
    // TransitionEditor loads the rules-file menu — a menu option only it
    // renders proves the routing.
    expect(await screen.findByText("Jump cut")).toBeInTheDocument();
  });

  it("routes by target kind, never by the item title", () => {
    // Title says "Playback speed…" but the ENGINE says TEXT_TRACK: the OCR
    // editor must open, bound to the engine subject id.
    render(
      <TargetEditor
        target={target("TEXT_TRACK", "ocr_track_0007")}
        entry={misleadingEntry()}
        store={store}
        currentFrame={0}
        onOutcome={vi.fn()}
      />,
    );
    expect(screen.getByRole("button", { name: /verify text/i })).toBeInTheDocument();
    expect(screen.queryByText(/slow motion/i)).not.toBeInTheDocument();
  });

  it("shows the honest no-target panel when the engine names none", () => {
    render(
      <TargetEditor
        target={null}
        entry={misleadingEntry()}
        store={store}
        currentFrame={0}
        onOutcome={vi.fn()}
      />,
    );
    expect(
      screen.getByText(/no directly decidable target/i),
    ).toBeInTheDocument();
  });
});

describe("privileged caption saves", () => {
  it("save_ready_caption sends only run + destination — never caption text", async () => {
    const mocked = vi.mocked(invoke);
    mocked.mockResolvedValueOnce(undefined);
    await saveReadyCaption("C:/runs/x", "C:/out/ready_to_enter.md");
    expect(mocked).toHaveBeenCalledWith("save_ready_caption", {
      runDir: "C:/runs/x",
      destination: "C:/out/ready_to_enter.md",
    });
    const args = mocked.mock.calls[0]?.[1] as Record<string, unknown>;
    expect(Object.keys(args)).toEqual(["runDir", "destination"]);
  });
});
