"""Visual subsystem (Phase 4): a shared bounded frame cache, per-frame machine
observations, and deterministic visual-concern candidates.

All timing derives from exact ledger frame identity (frame_index -> source PTS ->
annotation time); no approximate timestamp seeking is ever used for evidence.
This is a machine observation layer — never final caption prose.
"""
