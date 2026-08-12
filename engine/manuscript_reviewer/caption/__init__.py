"""Caption Brain (Phase 5): eligibility, fact graph, plan, renderer, coverage.

The first phase allowed to create final caption prose — but only from
eligibility-gated :class:`~..models.caption_brain.CaptionFact` records. A
machine candidate never silently becomes a final fact; ONE central policy
(:mod:`.eligibility`) decides caption eligibility.
"""
