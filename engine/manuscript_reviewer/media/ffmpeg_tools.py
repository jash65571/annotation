"""The single safe subprocess wrapper for ffmpeg / ffprobe.

All shell-outs in the engine go through :func:`run_tool`. No other module may
call ``subprocess`` directly. stderr is always captured and attached to
structured exceptions. No network access is ever performed.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

#: Optional override so users can point at a specific ffmpeg install.
FFMPEG_DIR_ENV = "MANUSCRIPT_FFMPEG_DIR"


class FFmpegNotFoundError(RuntimeError):
    """ffmpeg/ffprobe could not be located on PATH or via MANUSCRIPT_FFMPEG_DIR."""


class ToolExecutionError(RuntimeError):
    """A media tool exited non-zero; carries the full command and stderr."""

    def __init__(self, command: list[str], returncode: int, stderr: str) -> None:
        self.command = command
        self.returncode = returncode
        self.stderr = stderr
        super().__init__(
            f"Command failed (exit {returncode}): {' '.join(command)}\nstderr:\n{stderr[-4000:]}"
        )


def find_tool(name: str) -> Path:
    """Locate ``ffprobe``/``ffmpeg``, preferring $MANUSCRIPT_FFMPEG_DIR."""
    override = os.environ.get(FFMPEG_DIR_ENV)
    if override:
        candidate = Path(override) / f"{name}.exe"
        if candidate.is_file():
            return candidate
        candidate = Path(override) / name
        if candidate.is_file():
            return candidate
    found = shutil.which(name)
    if found:
        return Path(found)
    raise FFmpegNotFoundError(
        f"{name} not found. Install FFmpeg and add it to PATH, "
        f"or set {FFMPEG_DIR_ENV} to its bin directory."
    )


@dataclass(frozen=True)
class ToolResult:
    """Captured output of one tool invocation."""

    command: list[str]
    stdout: str
    stderr: str


def run_tool(executable: Path, args: list[str], timeout: float | None = 600.0) -> ToolResult:
    """Run a media tool with fully captured output.

    Raises :class:`ToolExecutionError` on non-zero exit. ``stdin`` is closed,
    output is decoded as UTF-8 with replacement so malformed metadata cannot
    crash the run.
    """
    command = [str(executable), *args]
    logger.debug("run_tool: %s", " ".join(command))
    completed = subprocess.run(
        command,
        capture_output=True,
        stdin=subprocess.DEVNULL,
        timeout=timeout,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if completed.returncode != 0:
        raise ToolExecutionError(command, completed.returncode, completed.stderr)
    return ToolResult(command=command, stdout=completed.stdout, stderr=completed.stderr)


@dataclass(frozen=True)
class BinaryToolResult:
    """Captured output of one tool invocation with binary stdout (rawvideo,
    image pipes). stderr is still decoded text for diagnostics."""

    command: list[str]
    stdout: bytes
    stderr: str


def run_tool_binary(
    executable: Path, args: list[str], timeout: float | None = 600.0
) -> BinaryToolResult:
    """Run a media tool capturing stdout as raw bytes.

    Same contract as :func:`run_tool` (closed stdin, captured stderr,
    :class:`ToolExecutionError` on non-zero exit) but without text decoding of
    stdout — for rawvideo/PNG pipe output.
    """
    command = [str(executable), *args]
    logger.debug("run_tool_binary: %s", " ".join(command))
    completed = subprocess.run(
        command,
        capture_output=True,
        stdin=subprocess.DEVNULL,
        timeout=timeout,
        check=False,
    )
    stderr_text = completed.stderr.decode("utf-8", errors="replace")
    if completed.returncode != 0:
        raise ToolExecutionError(command, completed.returncode, stderr_text)
    return BinaryToolResult(command=command, stdout=completed.stdout, stderr=stderr_text)


def tool_version(executable: Path) -> str:
    """First line of ``<tool> -version`` (recorded in the run manifest)."""
    result = run_tool(executable, ["-version"], timeout=30.0)
    return result.stdout.splitlines()[0].strip() if result.stdout else "unknown"
