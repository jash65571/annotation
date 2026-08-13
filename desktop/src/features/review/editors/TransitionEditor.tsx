/**
 * Transition classification editor.
 *
 * The transition menu comes exclusively from the rules file
 * (rules.shots.allowed_transition_types) — never hardcoded. Shot 1 is fixed
 * to rules.shots.shot_one_transition; for later shots the opening value is
 * excluded from the menu. Missing rules ⇒ explicit error state, no fallback.
 */

import { useEffect, useState, type ReactElement } from "react";
import { getRules } from "../../../api/bridge";
import type { DecisionsStore } from "../decisionsStore";
import { getReviewerName, ReviewerNameField } from "../reviewerName";

export interface TransitionEditorProps {
  shotIndex: number;
  currentStatus?: string;
  store: DecisionsStore;
  onResolved: (outcome: unknown) => void;
}

interface TransitionMenu {
  allowed: string[];
  openingValue: string | null;
}

function parseMenu(rules: Record<string, unknown>): TransitionMenu | null {
  const shots = rules["shots"];
  if (typeof shots !== "object" || shots === null) return null;
  const allowed = (shots as Record<string, unknown>)["allowed_transition_types"];
  if (!Array.isArray(allowed) || allowed.length === 0) return null;
  if (!allowed.every((value): value is string => typeof value === "string")) return null;
  const opening = (shots as Record<string, unknown>)["shot_one_transition"];
  return { allowed, openingValue: typeof opening === "string" ? opening : null };
}

export function TransitionEditor({
  shotIndex,
  currentStatus,
  store,
  onResolved,
}: TransitionEditorProps): ReactElement {
  const subjectId = `TRANSITION-${shotIndex}`;

  const [menu, setMenu] = useState<TransitionMenu | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [selected, setSelected] = useState("");
  const [note, setNote] = useState("");
  const [reviewerName, setReviewerNameState] = useState(getReviewerName());
  const nameMissingAtMount = useState(getReviewerName() === "")[0];
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    getRules()
      .then((payload) => {
        if (cancelled) return;
        const parsed = parseMenu(payload.rules);
        if (parsed == null) {
          setLoadError(
            "Rules do not contain shots.allowed_transition_types — the transition menu cannot be shown.",
          );
        } else {
          setMenu(parsed);
        }
      })
      .catch((err: unknown) => {
        if (!cancelled) setLoadError(String(err));
      });
    return () => {
      cancelled = true;
    };
  }, []);

  const isOpeningShot = shotIndex === 1;
  const options =
    menu == null
      ? []
      : isOpeningShot
        ? menu.openingValue != null
          ? [menu.openingValue]
          : []
        : menu.allowed.filter((value) => value !== menu.openingValue);
  const effectiveValue = isOpeningShot ? (menu?.openingValue ?? "") : selected;

  async function save(): Promise<void> {
    if (effectiveValue === "") return;
    setError(null);
    setSaving(true);
    try {
      const outcome = await store.applyDecision(
        store.makeDecision({
          subject_id: subjectId,
          decision_type: "TRANSITION_CLASSIFICATION",
          value: effectiveValue,
          decided_by: reviewerName,
          ...(note.trim() !== "" ? { reviewer_note: note.trim() } : {}),
        }),
      );
      onResolved(outcome);
    } catch (err) {
      setError(String(err));
    } finally {
      setSaving(false);
    }
  }

  if (loadError != null) {
    return (
      <div className="panel col" role="alert" aria-label="Transition editor error">
        <h3 style={{ margin: 0 }}>Transition — shot {shotIndex}</h3>
        <p className="badge fail">{loadError}</p>
      </div>
    );
  }

  return (
    <div className="panel col" aria-label={`Transition editor for shot ${shotIndex}`}>
      <h3 style={{ margin: 0 }}>Transition — shot {shotIndex}</h3>
      {nameMissingAtMount && <ReviewerNameField onChange={setReviewerNameState} />}
      {currentStatus != null && (
        <p style={{ margin: 0 }}>
          <span className="badge machine">MACHINE LEAD</span>{" "}
          <span className="mono">{currentStatus}</span>
        </p>
      )}
      {menu == null ? (
        <p className="muted">Loading transition menu from rules…</p>
      ) : (
        <label className="col">
          <span className="muted">
            {isOpeningShot
              ? "Shot 1 transition is fixed by the rules"
              : "Transition type (from rules)"}
          </span>
          <select
            aria-label="Transition type"
            value={effectiveValue}
            disabled={isOpeningShot}
            onChange={(event) => setSelected(event.target.value)}
          >
            {!isOpeningShot && <option value="">— select —</option>}
            {options.map((value) => (
              <option key={value} value={value}>
                {value}
              </option>
            ))}
          </select>
        </label>
      )}
      <label className="col">
        <span className="muted">Reviewer note (optional)</span>
        <textarea
          aria-label="Reviewer note"
          value={note}
          rows={2}
          onChange={(event) => setNote(event.target.value)}
        />
      </label>
      <button
        type="button"
        className="primary"
        disabled={reviewerName === "" || saving || effectiveValue === ""}
        onClick={() => void save()}
      >
        SAVE TRANSITION
      </button>
      {reviewerName === "" && (
        <p className="muted">Enter your reviewer name to save decisions.</p>
      )}
      {error != null && (
        <p className="badge fail" role="alert">
          {error}
        </p>
      )}
    </div>
  );
}
