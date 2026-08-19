import pandas as pd
import pytest

from src.trading_calendar import (
    align_to_trading_day,
    shift_to_trading_month_end,
    trading_month_end_map,
    trading_month_ends,
)


CALENDAR = pd.to_datetime(
    [
        "2024-01-05",
        "2024-01-08",
        "2024-01-31",
        "2024-02-01",
        "2024-02-29",
        "2024-03-01",
        "2024-03-29",
    ]
)


def test_align_to_trading_day_uses_inclusive_1457_cutoff():
    timestamps = pd.Series(
        [
            "2024-01-05 14:56:59",
            "2024-01-05 14:57:00",
            "2024-01-05 14:57:00.001",
        ]
    )

    actual = align_to_trading_day(timestamps, CALENDAR)

    assert actual.tolist() == [
        pd.Timestamp("2024-01-05"),
        pd.Timestamp("2024-01-05"),
        pd.Timestamp("2024-01-08"),
    ]


def test_align_to_trading_day_moves_non_trading_days_forward_without_clipping():
    timestamps = pd.Series(
        [
            "2024-01-06 09:00:00",  # Saturday
            "2024-01-07 23:00:00",  # Sunday
            "2024-03-29 15:00:00",  # after cutoff and beyond calendar
            None,
        ]
    )

    actual = align_to_trading_day(timestamps, CALENDAR)

    assert actual.iloc[0] == pd.Timestamp("2024-01-08")
    assert actual.iloc[1] == pd.Timestamp("2024-01-08")
    assert pd.isna(actual.iloc[2])
    assert pd.isna(actual.iloc[3])


def test_align_to_trading_day_converts_aware_timestamps_to_shanghai_time():
    timestamps = pd.Series(
        ["2024-01-05T06:57:00Z", "2024-01-05T06:57:01Z"],
        index=[10, 20],
    )

    actual = align_to_trading_day(timestamps, CALENDAR)

    assert actual.index.tolist() == [10, 20]
    assert actual.tolist() == [pd.Timestamp("2024-01-05"), pd.Timestamp("2024-01-08")]


def test_monthly_dates_are_last_trading_days_not_calendar_month_ends():
    actual = trading_month_ends(CALENDAR, "2024-01", "2024-03")
    assert actual.tolist() == [
        pd.Timestamp("2024-01-31"),
        pd.Timestamp("2024-02-29"),
        pd.Timestamp("2024-03-29"),
    ]

    mapping = trading_month_end_map(CALENDAR, "2024-01", "2024-03")
    assert shift_to_trading_month_end(pd.Timestamp("2024-01-31"), 2, mapping) == pd.Timestamp(
        "2024-03-29"
    )
    assert pd.isna(shift_to_trading_month_end(pd.Timestamp("2024-03-29"), 1, mapping))


def test_monthly_dates_fail_if_calendar_has_a_gap():
    with pytest.raises(ValueError, match="2024-02"):
        trading_month_ends(["2024-01-31", "2024-03-29"], "2024-01", "2024-03")


def test_staleness_is_natural_days_not_number_of_sessions():
    publish_date = pd.Timestamp("2024-01-01")
    asof_date = pd.Timestamp("2024-06-30")
    assert (asof_date - publish_date).days == 181
    assert (asof_date - publish_date).days > 180
