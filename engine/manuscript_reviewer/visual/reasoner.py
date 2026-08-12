"""Future visual-reasoner (VLM) interface — DESIGN ONLY (AD).

This module defines a provider-neutral contract for a future visual reasoner. It
is NOT implemented and performs NO network calls; no cloud VLM provider exists.
Its purpose is to fix the contract so that, when a reasoner is added, it can
never own the clock or fabricate evidence:

* Input is ONLY selected evidence: exact frame IDs, exact crops, short frame
  strips, a structured question, and existing entity/deterministic evidence.
* Output is a proposal citing supporting/contradicting FRAME IDs, an uncertainty,
  and a review-required flag. **It can never supply a timestamp** — timing comes
  from frame identity; the engine derives display time from the cited frames.

Any concrete adapter must be tested with mocks; the default runtime stays local.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from ..models.common import StrictModel


class VisualReasonerRequest(StrictModel):
    """Selected evidence handed to a future reasoner. No timestamps are passed —
    only exact frame identities and crops."""

    question: str
    frame_ids: list[int] = []
    crop_ids: list[str] = []
    strip_frame_ids: list[int] = []
    entity_hypothesis_ids: list[str] = []
    deterministic_evidence_ids: list[str] = []


class VisualReasonerResponse(StrictModel):
    """A reasoner proposal. Cites frames, never timestamps; the engine derives
    display time from the cited frames."""

    proposal: str
    claim_type: str | None = None
    supporting_frame_ids: list[int] = []
    contradicting_frame_ids: list[int] = []
    uncertainty: float  # 0.0 (certain) .. 1.0 (unknown)
    review_required: bool = True


@runtime_checkable
class VisualReasonerAdapter(Protocol):
    """Provider-neutral visual-reasoner contract (design only; not implemented).

    A conforming adapter must not perform network I/O in the default runtime and
    must never return a timestamp — only cited frame IDs.
    """

    def reason(self, request: VisualReasonerRequest) -> VisualReasonerResponse: ...
