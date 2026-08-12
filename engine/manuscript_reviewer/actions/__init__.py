"""Actions subsystem (Phase 4): ownership/contact events, final-state checks, and
atomic action-boundary candidates derived from defensible state changes.

Semantic labels (picks up / throws / raises hand) are NEVER forced from generic
motion — a candidate stays label-less unless the pre/contact/post state sequence
independently supports it. Boundaries use exact frames.
"""
