import { useEffect, useMemo, useReducer, useState } from "react";
import { appReducer, initialState } from "./state/appState";
import { AppContext } from "./state/context";
import { HomeScreen } from "./screens/HomeScreen";
import { AnalyzeScreen } from "./screens/AnalyzeScreen";
import { ReviewScreen } from "./screens/ReviewScreen";
import { FinalScreen } from "./screens/FinalScreen";
import { SettingsScreen } from "./screens/SettingsScreen";
import { StatusBar } from "./components/StatusBar";
import { onAnalysisDone, onAnalysisError, onAnalysisProgress } from "./api/events";
import { getRunSummary } from "./api/bridge";

export type Screen = "home" | "analyze" | "review" | "final" | "settings";

export function App() {
  const [state, dispatch] = useReducer(appReducer, initialState);
  const [screen, setScreen] = useState<Screen>("home");
  const ctx = useMemo(() => ({ state, dispatch }), [state]);

  useEffect(() => {
    const subs = [
      onAnalysisProgress((event) => dispatch({ type: "ANALYSIS_PROGRESS", event })),
      onAnalysisDone(async (done) => {
        const summary = await getRunSummary(done.run_dir);
        dispatch({ type: "RUN_LOADED", runDir: done.run_dir, summary });
        setScreen("review");
      }),
      onAnalysisError((error) =>
        dispatch({
          type: "ANALYSIS_FAILED",
          code: error.code,
          message: error.message,
          ...(error.detail !== undefined ? { detail: error.detail } : {}),
        }),
      ),
    ];
    return () => {
      for (const sub of subs) void sub.then((unlisten) => unlisten());
    };
  }, []);

  // Blocker-first navigation (spec §120): a loaded run opens on the screen
  // its engine-reported readiness demands.
  useEffect(() => {
    if (state.phase === "READY_FOR_FINAL_REVIEW" || state.phase === "READY_TO_ENTER") {
      setScreen("final");
    }
  }, [state.phase]);

  return (
    <AppContext.Provider value={ctx}>
      <div className="app-shell col" style={{ height: "100%", gap: 0 }}>
        <main style={{ flex: 1, minHeight: 0, overflow: "auto" }}>
          {screen === "home" && <HomeScreen onNavigate={setScreen} />}
          {screen === "analyze" && <AnalyzeScreen onNavigate={setScreen} />}
          {screen === "review" && <ReviewScreen onNavigate={setScreen} />}
          {screen === "final" && <FinalScreen onNavigate={setScreen} />}
          {screen === "settings" && <SettingsScreen onNavigate={setScreen} />}
        </main>
        <StatusBar screen={screen} onNavigate={setScreen} />
      </div>
    </AppContext.Provider>
  );
}
