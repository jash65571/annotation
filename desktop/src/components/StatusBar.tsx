import { useApp } from "../state/context";
import type { Screen } from "../App";

const PHASE_LABEL: Record<string, string> = {
  IDLE: "Idle",
  TASK_READY: "Task ready",
  ANALYZING: "Analyzing…",
  ANALYSIS_CANCELLED: "Analysis cancelled",
  REVIEWING: "Review required",
  FINALIZING: "Finalizing…",
  READY_FOR_FINAL_REVIEW: "Ready for final review",
  READY_TO_ENTER: "Ready to enter",
  BLOCKED: "Blocked",
  ERROR: "Error",
};

export function StatusBar({
  screen,
  onNavigate,
}: {
  screen: Screen;
  onNavigate: (screen: Screen) => void;
}) {
  const { state } = useApp();
  const tabs: Array<{ id: Screen; label: string }> = [
    { id: "home", label: "Home" },
    { id: "review", label: "Review" },
    { id: "final", label: "Final" },
    { id: "settings", label: "Settings" },
  ];
  return (
    <footer
      className="row"
      style={{
        borderTop: "1px solid var(--border)",
        padding: "4px 12px",
        justifyContent: "space-between",
        background: "var(--bg-raised)",
      }}
    >
      <nav className="row" aria-label="Screens">
        {tabs.map((tab) => (
          <button
            key={tab.id}
            aria-current={screen === tab.id ? "page" : undefined}
            onClick={() => onNavigate(tab.id)}
            style={screen === tab.id ? { borderColor: "var(--accent)" } : undefined}
          >
            {tab.label}
          </button>
        ))}
      </nav>
      <span className="muted" role="status">
        {PHASE_LABEL[state.phase] ?? state.phase}
        {state.runDir ? ` — ${state.runDir}` : ""}
      </span>
    </footer>
  );
}
