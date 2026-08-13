import { createContext, useContext, type Dispatch } from "react";
import { initialState, type AppAction, type AppState } from "./appState";

export interface AppContextValue {
  state: AppState;
  dispatch: Dispatch<AppAction>;
}

export const AppContext = createContext<AppContextValue>({
  state: initialState,
  dispatch: () => undefined,
});

export function useApp(): AppContextValue {
  return useContext(AppContext);
}
