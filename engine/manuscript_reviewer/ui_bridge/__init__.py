"""Structured local UI bridge for the Reviewer Cockpit desktop app.

The bridge is transport only: it exposes the existing engine APIs
(`pipeline.run_audit`, `caption_brain.finalize_run`) and run-directory
artifacts as typed JSON over a framed stdin/stdout JSONL protocol. It never
re-implements engine logic, never fabricates human decisions, and never
parses Rich CLI output.
"""

from __future__ import annotations

#: Bumped only on breaking protocol changes; the desktop app performs a
#: handshake and must refuse to talk across incompatible versions.
UI_BRIDGE_PROTOCOL_VERSION = 1
