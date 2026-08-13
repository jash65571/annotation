import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import type { AudioReviewItem, HumanReviewDecision } from "../api/types";
import type { DecisionsStore } from "../features/review/decisionsStore";
import { SpeechEditor } from "../features/review/editors/SpeechEditor";
import { SpeedEditor } from "../features/review/editors/SpeedEditor";
import { TransitionEditor } from "../features/review/editors/TransitionEditor";

// The bridge is mocked wholesale: the transition menu below is the ONLY
// source the TransitionEditor may render options from.
vi.mock("../api/bridge", () => ({
  getRules: vi.fn(async () => ({
    rules_version: "1.3.0-test",
    rules: {
      shots: {
        allowed_transition_types: ["Opening shot", "Hard cut", "Cross dissolve"],
        shot_one_transition: "Opening shot",
      },
    },
  })),
  saveReviewDecisions: vi.fn(async () => ({ path: "", count: 0 })),
  saveHumanFacts: vi.fn(async () => ({ path: "", count: 0 })),
  finalize: vi.fn(async () => ({ result: {}, caption_state: {} })),
  getRunSummary: vi.fn(async () => ({})),
}));

// Node's experimental global localStorage can shadow jsdom's with a
// non-functional stub — install a real in-memory Storage for these tests.
const memoryStorage = ((): Storage => {
  const data = new Map<string, string>();
  return {
    getItem: (key: string) => data.get(key) ?? null,
    setItem: (key: string, value: string) => {
      data.set(key, String(value));
    },
    removeItem: (key: string) => {
      data.delete(key);
    },
    clear: () => data.clear(),
    key: (index: number) => [...data.keys()][index] ?? null,
    get length() {
      return data.size;
    },
  } as Storage;
})();
Object.defineProperty(window, "localStorage", {
  value: memoryStorage,
  configurable: true,
});

function mockStore(): {
  store: DecisionsStore;
  makeDecision: ReturnType<typeof vi.fn>;
  applyDecision: ReturnType<typeof vi.fn>;
} {
  const makeDecision = vi.fn(
    (input: Partial<HumanReviewDecision>): HumanReviewDecision =>
      ({
        ...input,
        decision_id: "D-TEST",
        decided_at_utc: "2026-01-01T00:00:00Z",
        bound_video_sha256: "sha-test",
        bound_rules_version: "1.3.0-test",
      }) as HumanReviewDecision,
  );
  const applyDecision = vi.fn(async () => ({ readiness: "REVIEW_REQUIRED" }));
  const store = { makeDecision, applyDecision } as unknown as DecisionsStore;
  return { store, makeDecision, applyDecision };
}

const audioItem: AudioReviewItem = {
  item_id: "SR-0001",
  reasons: ["low confidence"],
  priority: "HIGH",
  start_exact: "0/1",
  end_exact: "1/1",
  playback_start: "0/1",
  playback_end: "1/1",
  asr_text_candidate: "hello there",
};

beforeEach(() => {
  window.localStorage.removeItem("mr.reviewerName");
});

describe("SpeedEditor", () => {
  it("never auto-selects a speed and produces a slow_motion PLAYBACK_SPEED decision", async () => {
    window.localStorage.setItem("mr.reviewerName", "Jash");
    const user = userEvent.setup();
    const { store, makeDecision, applyDecision } = mockStore();
    const onResolved = vi.fn();
    render(
      <SpeedEditor
        shotNumber={3}
        subjectId="SPEED-3"
        machineEvidence="REGULAR_CANDIDATE"
        store={store}
        onResolved={onResolved}
      />,
    );

    // No radio is pre-selected: the machine candidate never becomes a choice.
    for (const radio of screen.getAllByRole("radio")) {
      expect(radio).not.toBeChecked();
    }
    // Save is disabled until the human explicitly chooses.
    expect(screen.getByRole("button", { name: "SAVE PLAYBACK SPEED" })).toBeDisabled();

    await user.click(screen.getByRole("radio", { name: "Slow motion" }));
    await user.click(screen.getByRole("button", { name: "SAVE PLAYBACK SPEED" }));

    await waitFor(() => expect(applyDecision).toHaveBeenCalledTimes(1));
    expect(makeDecision).toHaveBeenCalledWith(
      expect.objectContaining({
        decision_type: "PLAYBACK_SPEED",
        subject_id: "SPEED-3",
        value: "slow_motion",
        decided_by: "Jash",
      }),
    );
    expect(onResolved).toHaveBeenCalledTimes(1);
  });
});

describe("SpeechEditor", () => {
  it("keeps VERIFY WORDING disabled until a reviewer name exists", async () => {
    const user = userEvent.setup();
    const { store } = mockStore();
    render(<SpeechEditor item={audioItem} store={store} onResolved={vi.fn()} />);

    expect(screen.getByRole("button", { name: "VERIFY WORDING" })).toBeDisabled();

    await user.type(screen.getByRole("textbox", { name: "Reviewer name" }), "Jash");
    expect(screen.getByRole("button", { name: "VERIFY WORDING" })).toBeEnabled();
  });

  it("labels the ASR candidate as MACHINE LEAD — unverified", () => {
    const { store } = mockStore();
    render(<SpeechEditor item={audioItem} store={store} onResolved={vi.fn()} />);
    expect(screen.getByText("MACHINE LEAD — unverified")).toBeInTheDocument();
    expect(screen.queryByText(/^Detected/)).not.toBeInTheDocument();
  });
});

describe("TransitionEditor", () => {
  it("renders options only from the rules payload, excluding the opening value", async () => {
    window.localStorage.setItem("mr.reviewerName", "Jash");
    const { store } = mockStore();
    render(<TransitionEditor shotIndex={2} store={store} onResolved={vi.fn()} />);

    await screen.findByRole("option", { name: "Hard cut" });
    const options = screen
      .getAllByRole("option")
      .map((option) => option.textContent ?? "");
    // Placeholder + exactly the rules menu minus the shot-1-only opening value.
    expect(options).toEqual(["— select —", "Hard cut", "Cross dissolve"]);
    expect(
      screen.queryByRole("option", { name: "Opening shot" }),
    ).not.toBeInTheDocument();
  });

  it("fixes shot 1 to the rules' opening transition", async () => {
    window.localStorage.setItem("mr.reviewerName", "Jash");
    const { store } = mockStore();
    render(<TransitionEditor shotIndex={1} store={store} onResolved={vi.fn()} />);

    const select = await screen.findByRole("combobox", { name: "Transition type" });
    expect(select).toBeDisabled();
    expect(select).toHaveValue("Opening shot");
  });
});
