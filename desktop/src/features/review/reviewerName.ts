/**
 * The human reviewer's name, typed once and persisted locally.
 *
 * Every decision's `decided_by` must be this human name — the engine rejects
 * ""/"machine"/"ai"/"system", and the UI never invents a default.
 */

import {
  createElement,
  useState,
  type ChangeEvent,
  type ReactElement,
} from "react";

const STORAGE_KEY = "mr.reviewerName";

const FORBIDDEN = new Set(["", "machine", "ai", "system"]);

/** The persisted reviewer name, or "" when none has been typed yet. */
export function getReviewerName(): string {
  try {
    const raw = (window.localStorage.getItem(STORAGE_KEY) ?? "").trim();
    return FORBIDDEN.has(raw.toLowerCase()) ? "" : raw;
  } catch {
    return "";
  }
}

export function setReviewerName(name: string): void {
  try {
    window.localStorage.setItem(STORAGE_KEY, name.trim());
  } catch {
    // localStorage unavailable — the field still works in-memory.
  }
}

export interface ReviewerNameFieldProps {
  onChange?: (name: string) => void;
}

/** Tiny inline field to capture the reviewer name once. */
export function ReviewerNameField(props: ReviewerNameFieldProps): ReactElement {
  const [name, setName] = useState(getReviewerName());
  return createElement(
    "label",
    { className: "row" },
    createElement("span", { className: "muted" }, "Reviewer name"),
    createElement("input", {
      type: "text",
      value: name,
      placeholder: "Required to save decisions",
      "aria-label": "Reviewer name",
      onChange: (event: ChangeEvent<HTMLInputElement>) => {
        const next = event.target.value;
        setName(next);
        setReviewerName(next);
        props.onChange?.(getReviewerName());
      },
    }),
  );
}
