"""PyInstaller entry point for the packaged engine sidecar.

Runs the exact same worker module as development mode — the engine is never
forked (spec §13)."""

from manuscript_reviewer.ui_bridge.worker import main

if __name__ == "__main__":
    main()
