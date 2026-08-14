"""AutoScribe must never turn a failure to observe into a factual claim.

These tests pin the behaviours that were silently wrong: a swallowed audio
exception reading as "no speech", an unverifiable cut reading as "no cut", and
a failed reviewer pass reading as an approved caption.
"""

from __future__ import annotations

from autoscribe import cuts
from autoscribe.blockers import BLOCKING, WARNING, BlockerLog
from autoscribe.frames import GridFrame


def test_readiness_is_never_true_without_human_signoff() -> None:
    log = BlockerLog()
    ready, reason = log.readiness()
    assert ready is False
    assert "human" in reason.lower()


def test_blocking_entries_are_reported_separately_from_warnings() -> None:
    log = BlockerLog()
    log.add("A", "blocking thing")
    log.add("B", "advisory thing", severity=WARNING)
    assert [b.code for b in log.blocking] == ["A"]
    assert [b.code for b in log.warnings] == ["B"]
    ready, reason = log.readiness()
    assert ready is False
    assert "1 unresolved blocking" in reason


def test_add_exception_records_type_and_message() -> None:
    log = BlockerLog()
    log.add_exception("AUDIO_EXTRACTION_FAILED", RuntimeError("ffmpeg missing"))
    assert log.entries[0].severity == BLOCKING
    assert "RuntimeError: ffmpeg missing" in log.entries[0].detail


class _RaisingBackend:
    """Stands in for a vision backend whose API call fails."""

    def complete(self, *_a: object, **_k: object) -> str:
        raise ConnectionError("network down")


class _UndecidedBackend:
    def complete(self, *_a: object, **_k: object) -> str:
        return '{"is_cut": null}'


def _frames(times: list[float], tmp_path: object) -> list[GridFrame]:
    from pathlib import Path

    base = Path(str(tmp_path))
    out = []
    for i, t in enumerate(times):
        p = base / f"f{i}.png"
        p.write_bytes(b"\x89PNG")
        out.append(GridFrame(index=i, time_seconds=t, path=p, source_index=i))
    return out


def test_cut_verify_failure_is_unresolved_not_false(tmp_path: object) -> None:
    """A network error used to return 'not a cut', silently deleting a shot."""
    log = BlockerLog()
    before = _frames([0.9, 1.0], tmp_path)
    after = _frames([1.1, 1.2], tmp_path)
    is_cut, _ = cuts._verify(
        _RaisingBackend(), before, after, blockers=log, at=1.0,  # type: ignore[arg-type]
    )
    assert is_cut is None, "a failed verification must not be reported as 'no cut'"
    assert any(b.code == "CUT_VERIFY_FAILED" for b in log.blocking)


def test_cut_verify_undecided_is_unresolved(tmp_path: object) -> None:
    log = BlockerLog()
    is_cut, _ = cuts._verify(
        _UndecidedBackend(),  # type: ignore[arg-type]
        _frames([0.9], tmp_path), _frames([1.1], tmp_path),
        blockers=log, at=1.0,
    )
    assert is_cut is None
    assert any(b.code == "CUT_UNDECIDED" for b in log.blocking)


def test_missing_frames_on_one_side_is_unresolved() -> None:
    log = BlockerLog()
    is_cut, _ = cuts._verify(_RaisingBackend(), [], [], blockers=log, at=2.0)  # type: ignore[arg-type]
    assert is_cut is None
    assert any(b.code == "CUT_UNVERIFIABLE" for b in log.blocking)
