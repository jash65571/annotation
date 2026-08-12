"""Typed JSONL bridge protocol: requests, responses, progress events, errors.

One JSON object per line. Every request carries ``request_id``, ``command``,
``payload`` and ``protocol_version``; every response echoes the request id.
Long-running jobs additionally emit ``BridgeEvent`` progress lines before the
final response. Errors use typed codes — the frontend never pattern-matches
English exception strings.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal

from ..models.common import StrictModel


class BridgeErrorCode(StrEnum):
    ENGINE_NOT_FOUND = "ENGINE_NOT_FOUND"
    ENGINE_VERSION_MISMATCH = "ENGINE_VERSION_MISMATCH"
    PROTOCOL_VERSION_MISMATCH = "PROTOCOL_VERSION_MISMATCH"
    FFMPEG_UNAVAILABLE = "FFMPEG_UNAVAILABLE"
    RUN_NOT_FOUND = "RUN_NOT_FOUND"
    RUN_LOCKED = "RUN_LOCKED"
    MANIFEST_MISMATCH = "MANIFEST_MISMATCH"
    INVALID_DECISION = "INVALID_DECISION"
    ASR_UNAVAILABLE = "ASR_UNAVAILABLE"
    OCR_UNAVAILABLE = "OCR_UNAVAILABLE"
    CANCELLED = "CANCELLED"
    ENGINE_CRASH = "ENGINE_CRASH"
    INVALID_INPUT = "INVALID_INPUT"
    INVALID_COMMAND = "INVALID_COMMAND"
    NOT_READY = "NOT_READY"
    ARTIFACT_NOT_FOUND = "ARTIFACT_NOT_FOUND"


class BridgeError(StrictModel):
    code: BridgeErrorCode
    message: str
    #: Technical detail for the collapsible log view, never the primary UX.
    detail: str | None = None


class BridgeRequest(StrictModel):
    request_id: str
    command: str
    payload: dict[str, Any] = {}
    protocol_version: int


class BridgeResponse(StrictModel):
    request_id: str
    status: Literal["ok", "error"]
    payload: dict[str, Any] | None = None
    error: BridgeError | None = None


class BridgeEvent(StrictModel):
    """A structured progress event emitted while a request is running."""

    request_id: str
    event: Literal["progress"]
    payload: dict[str, Any] = {}


class BridgeCommandError(Exception):
    """Raised by command handlers to return a typed protocol error."""

    def __init__(
        self, code: BridgeErrorCode, message: str, detail: str | None = None
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.detail = detail

    def to_error(self) -> BridgeError:
        return BridgeError(code=self.code, message=self.message, detail=self.detail)
