"""Pluggable vision backend: describe a keyframe (what/who/action/on-screen text).

Pick ONE backend at run time:
  - CloudVisionBackend  — Anthropic Claude vision. Best quality. Needs ANTHROPIC_API_KEY.
  - OllamaVisionBackend — local VLM (e.g. qwen2.5-vl) via Ollama. Offline, no key.
  - ManualVisionBackend — no model; returns a placeholder (for wiring/tests).

All backends take a PNG path and return a concise factual description string.
"""

from __future__ import annotations

import base64
import json
import os
import urllib.request
from pathlib import Path
from typing import Protocol

PROMPT = (
    "Describe this single video frame precisely and factually for a transcript. "
    "State the setting, the people (by visible role/appearance, not invented names), "
    "what action is happening, and any legible on-screen text. One or two sentences. "
    "Do not speculate beyond what is visible."
)


class VisionBackend(Protocol):
    name: str

    def describe(self, image_path: Path) -> str: ...


def _b64(image_path: Path) -> str:
    return base64.b64encode(image_path.read_bytes()).decode("ascii")


def image_content(image_path: Path) -> dict[str, object]:
    """An OpenAI chat 'content' item wrapping a PNG as a data URI."""
    return {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{_b64(image_path)}"}}


def text_content(text: str) -> dict[str, object]:
    return {"type": "text", "text": text}


class ManualVisionBackend:
    name = "manual"

    def describe(self, image_path: Path) -> str:
        return "[vision backend not configured — no automatic description]"


class OllamaVisionBackend:
    """Local VLM via Ollama (https://ollama.com). Run `ollama pull qwen2.5-vl`."""

    name = "ollama"

    def __init__(self, model: str = "qwen2.5-vl", host: str = "http://localhost:11434") -> None:
        self.model = model
        self.host = host.rstrip("/")

    def describe(self, image_path: Path) -> str:
        payload = json.dumps({
            "model": self.model,
            "prompt": PROMPT,
            "images": [_b64(image_path)],
            "stream": False,
        }).encode()
        req = urllib.request.Request(
            f"{self.host}/api/generate", data=payload,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=180) as resp:
            data = json.loads(resp.read())
        return str(data.get("response", "")).strip()


class CloudVisionBackend:
    """Anthropic Claude vision. Set ANTHROPIC_API_KEY. Best description quality."""

    name = "cloud"

    def __init__(self, model: str = "claude-sonnet-5") -> None:
        self.model = model
        self.api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        if not self.api_key:
            raise RuntimeError("ANTHROPIC_API_KEY is not set for the cloud vision backend.")

    def describe(self, image_path: Path) -> str:
        body = json.dumps({
            "model": self.model,
            "max_tokens": 300,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "image", "source": {
                        "type": "base64", "media_type": "image/png", "data": _b64(image_path)}},
                    {"type": "text", "text": PROMPT},
                ],
            }],
        }).encode()
        req = urllib.request.Request(
            "https://api.anthropic.com/v1/messages", data=body,
            headers={
                "content-type": "application/json",
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
            },
        )
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read())
        parts = [b.get("text", "") for b in data.get("content", []) if b.get("type") == "text"]
        return " ".join(p.strip() for p in parts if p).strip()


class OpenAIVisionBackend:
    """OpenAI vision (Chat Completions). Set OPENAI_API_KEY; model via OPENAI_MODEL.

    Defaults to ``gpt-4o`` (reliable vision). Override with OPENAI_MODEL=gpt-5.6
    (or any vision-capable model your account has).
    """

    name = "openai"

    def __init__(self, model: str | None = None) -> None:
        self.model = model or os.environ.get("OPENAI_MODEL", "gpt-4o")
        self.api_key = os.environ.get("OPENAI_API_KEY", "")
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY is not set for the OpenAI vision backend.")

    def describe(self, image_path: Path) -> str:
        data_uri = f"data:image/png;base64,{_b64(image_path)}"
        body = json.dumps({
            "model": self.model,
            "max_tokens": 300,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": PROMPT},
                    {"type": "image_url", "image_url": {"url": data_uri}},
                ],
            }],
        }).encode()
        req = urllib.request.Request(
            "https://api.openai.com/v1/chat/completions", data=body,
            headers={
                "content-type": "application/json",
                "authorization": f"Bearer {self.api_key}",
            },
        )
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read())
        return str(data["choices"][0]["message"]["content"]).strip()

    def complete(
        self, content: list[dict[str, object]], *, json_mode: bool = False, max_tokens: int = 2000
    ) -> str:
        """Low-level multimodal completion (interleaved text + images)."""
        payload: dict[str, object] = {
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": content}],
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        req = urllib.request.Request(
            "https://api.openai.com/v1/chat/completions", data=json.dumps(payload).encode(),
            headers={"content-type": "application/json", "authorization": f"Bearer {self.api_key}"},
        )
        with urllib.request.urlopen(req, timeout=180) as resp:
            data = json.loads(resp.read())
        content = data["choices"][0]["message"].get("content")
        return str(content).strip() if content else ""


def make_backend(kind: str) -> VisionBackend:
    kind = kind.lower()
    if kind in ("openai", "gpt"):
        return OpenAIVisionBackend()
    if kind == "cloud":
        return CloudVisionBackend()
    if kind == "ollama":
        return OllamaVisionBackend()
    if kind == "manual":
        return ManualVisionBackend()
    raise ValueError(f"unknown vision backend: {kind!r} (cloud|ollama|manual)")
