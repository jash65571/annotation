"""Exact-timing tests: rational parsing, PTS math, drift, Manuscript 0.1 s display."""

from __future__ import annotations

from decimal import Decimal
from fractions import Fraction

import pytest

from manuscript_reviewer.media.timestamps import (
    cfr_expected_time,
    format_manuscript_display,
    parse_rational,
    pts_to_seconds,
    seconds_to_decimal,
    to_manuscript_display,
)

STANDARD_RATES = [
    Fraction(24000, 1001),  # 23.976
    Fraction(24),
    Fraction(25),
    Fraction(30000, 1001),  # 29.97
    Fraction(30),
    Fraction(50),
    Fraction(60000, 1001),  # 59.94
    Fraction(60),
]


def test_parse_rational_forms() -> None:
    assert parse_rational("30000/1001") == Fraction(30000, 1001)
    assert parse_rational("24") == Fraction(24)
    assert parse_rational("1/15360") == Fraction(1, 15360)


def test_parse_rational_undefined_raises() -> None:
    with pytest.raises(ZeroDivisionError):
        parse_rational("0/0")


def test_pts_to_seconds_exact() -> None:
    # Frame 706 in a 1/60 time base... use realistic mp4 time base 1/15360 at 60fps:
    # frame 706 has pts 706*256 = 180736 → exactly 706/60 s.
    assert pts_to_seconds(180736, Fraction(1, 15360)) == Fraction(706, 60)


@pytest.mark.parametrize("rate", STANDARD_RATES)
def test_no_cumulative_drift_over_ten_minutes(rate: Fraction) -> None:
    """Summing per-frame durations exactly equals N/rate — no float drift ever."""
    n = int(rate * 600) # ~10 minutes of frames
    total = sum([Fraction(1, 1) / rate] * n, Fraction(0))
    assert total == n / rate


@pytest.mark.parametrize("rate", STANDARD_RATES)
def test_cfr_expected_time_matches_pts_grid(rate: Fraction) -> None:
    # A CFR stream whose time_base is 1/(1000*rate.numerator) style grid:
    # simulate pts = index * (time_base_den / rate) exactly.
    time_base = Fraction(1, rate.numerator * 4)
    pts_per_frame = (Fraction(1) / rate) / time_base
    assert pts_per_frame.denominator == 1  # exact grid by construction
    for index in (0, 1, 999):
        pts = index * int(pts_per_frame)
        assert pts_to_seconds(pts, time_base) == cfr_expected_time(index, rate)


@pytest.mark.parametrize(
    ("rate", "frame_index", "expected_display"),
    [
        (Fraction(24000, 1001), 24, "1.0s"),      # 24*1001/24000 = 1.001 → 1.0
        (Fraction(24), 12, "0.5s"),               # 0.5 exactly
        (Fraction(25), 33, "1.3s"),               # 1.32 → 1.3
        (Fraction(30000, 1001), 300, "10.0s"),    # 10.01 → 10.0
        (Fraction(30), 7, "0.2s"),                # 0.2333 → 0.2
        (Fraction(50), 26, "0.5s"),               # 0.52 → 0.5
        (Fraction(60000, 1001), 706, "11.8s"),    # 11.778... → 11.8
        (Fraction(60), 706, "11.8s"),             # 11.7666 → 11.8
    ],
)
def test_manuscript_display_rounding(
    rate: Fraction, frame_index: int, expected_display: str
) -> None:
    exact = cfr_expected_time(frame_index, rate)
    assert format_manuscript_display(exact) == expected_display


def test_manuscript_display_half_up() -> None:
    assert to_manuscript_display(Fraction(1, 20)) == Decimal("0.1")   # 0.05 → 0.1
    assert to_manuscript_display(Fraction(3, 20)) == Decimal("0.2")   # 0.15 → 0.2
    assert to_manuscript_display(Fraction(0)) == Decimal("0.0")


def test_seconds_to_decimal_microsecond_rendering() -> None:
    assert str(seconds_to_decimal(Fraction(706, 60))) == "11.766667"
    assert str(seconds_to_decimal(Fraction(0))) == "0.000000"
