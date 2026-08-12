"""Review subsystem (Phase 4): machine proposals, review queue, human-decision
persistence, and triage.

Everything here is a *proposal* or a *queue item* — never a human decision, and
never final caption prose. Machine proposals can never be stored as a human
``ReviewDecision`` (validator P4-REVIEW-001).
"""
