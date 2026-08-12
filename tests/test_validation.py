"""Validator unit tests with crafted ledgers/media — no ffmpeg required."""

from __future__ import annotations

from fractions import Fraction

from manuscript_reviewer.models.frame import FrameLedger, FrameRecord
from manuscript_reviewer.models.media import MediaInfo, VideoStreamInfo
from manuscript_reviewer.models.validation import RunStatus, Severity
from manuscript_reviewer.validation.ledger_validator import (
    compute_run_status,
    cross_check_frame_count,
    validate_ledger,
)
from manuscript_reviewer.validation.media_validator import validate_media

TB = Fraction(1, 12288)


def make_frame(index: int, pts: int | None) -> FrameRecord:
    return FrameRecord(
        frame_index=index,
        pts=pts,
        pts_time_seconds=pts * TB if pts is not None else None,
        key_frame=index == 0,
    )


def make_ledger(frames: list[FrameRecord]) -> FrameLedger:
    return FrameLedger(stream_index=0, time_base=TB, frames=frames)


def make_media(
    declared_frames: int | None = None,
    nominal: Fraction | None = Fraction(24),
    average: Fraction | None = Fraction(24),
    video: bool = True,
) -> MediaInfo:
    streams = []
    if video:
        streams.append(
            VideoStreamInfo(
                stream_index=0,
                codec_name="h264",
                width=320,
                height=240,
                nominal_frame_rate=nominal,
                average_frame_rate=average,
                time_base=TB,
                declared_frame_count=declared_frames,
            )
        )
    return MediaInfo(
        file_name="x.mp4",
        file_size_bytes=1,
        container_format="mp4",
        video_streams=streams,
        audio_streams=[],
    )


def test_zero_frames_fails() -> None:
    issues = validate_ledger(make_ledger([]))
    assert any(i.rule_id == "P1-LEDGER-001" and i.severity == Severity.FAIL for i in issues)


def test_non_sequential_indexes_fail() -> None:
    ledger = make_ledger([make_frame(0, 0), make_frame(2, 512)])
    issues = validate_ledger(ledger)
    assert any(i.rule_id == "P1-LEDGER-002" for i in issues)


def test_duplicate_indexes_fail() -> None:
    frames = [make_frame(0, 0), make_frame(0, 512)]
    ledger = make_ledger(frames)
    issues = validate_ledger(ledger)
    assert any(i.rule_id == "P1-LEDGER-003" for i in issues)


def test_backward_pts_fails() -> None:
    ledger = make_ledger([make_frame(0, 0), make_frame(1, 1024), make_frame(2, 512)])
    issues = validate_ledger(ledger)
    assert any(i.rule_id == "P1-LEDGER-005" and i.severity == Severity.FAIL for i in issues)


def test_missing_pts_warns_and_makes_partial() -> None:
    ledger = make_ledger([make_frame(0, 0), make_frame(1, None)])
    issues = validate_ledger(ledger)
    assert any(i.rule_id == "P1-LEDGER-004" and i.severity == Severity.WARN for i in issues)
    assert compute_run_status(issues, ledger) == RunStatus.PARTIAL


def test_clean_ledger_passes() -> None:
    ledger = make_ledger([make_frame(i, i * 512) for i in range(10)])
    issues = validate_ledger(ledger)
    assert issues == []
    assert compute_run_status(issues, ledger) == RunStatus.PASS


def test_decode_count_mismatch_fails() -> None:
    ledger = make_ledger([make_frame(i, i * 512) for i in range(10)])
    _, issues = cross_check_frame_count(ledger, make_media(), decoded_count=11)
    assert any(i.rule_id == "P1-COUNT-002" and i.severity == Severity.FAIL for i in issues)
    assert compute_run_status(issues, ledger) == RunStatus.FAILED


def test_declared_count_mismatch_warns_only() -> None:
    ledger = make_ledger([make_frame(i, i * 512) for i in range(10)])
    signals, issues = cross_check_frame_count(
        ledger, make_media(declared_frames=12), decoded_count=10
    )
    assert any(i.rule_id == "P1-COUNT-003" and i.severity == Severity.WARN for i in issues)
    assert not any(i.severity == Severity.FAIL for i in issues)
    assert len(signals) == 3
    assert compute_run_status(issues, ledger) == RunStatus.PASS


def test_missing_decode_count_warns() -> None:
    ledger = make_ledger([make_frame(0, 0)])
    _, issues = cross_check_frame_count(ledger, make_media(), decoded_count=None)
    assert any(i.rule_id == "P1-COUNT-001" and i.severity == Severity.WARN for i in issues)


def test_no_video_stream_fails() -> None:
    issues = validate_media(make_media(video=False))
    assert any(i.rule_id == "P1-MEDIA-001" and i.severity == Severity.FAIL for i in issues)


def test_fps_mismatch_warns_vfr_signal() -> None:
    issues = validate_media(
        make_media(nominal=Fraction(30), average=Fraction(30000, 1001))
    )
    assert any(i.rule_id == "P1-MEDIA-004" and i.severity == Severity.WARN for i in issues)


def test_no_ledger_means_failed() -> None:
    assert compute_run_status([], None) == RunStatus.FAILED
