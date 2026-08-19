"""交易日历读取与时间对齐工具。

本模块只消费项目已有的交易日数据，不负责下载或更新交易日历。
"""
from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


DEFAULT_CUTOFF = dt.time(14, 57)


def parse_market_timestamps(values: pd.Series | Iterable[object]) -> pd.Series:
    """解析时间戳；带时区数据统一转换为上海时间后移除时区。"""
    source = values if isinstance(values, pd.Series) else pd.Series(values)
    parsed = pd.to_datetime(source, errors="coerce", format="mixed")
    if isinstance(parsed.dtype, pd.DatetimeTZDtype):
        parsed = parsed.dt.tz_convert("Asia/Shanghai").dt.tz_localize(None)
    if not isinstance(parsed.dtype, np.dtype) or parsed.dtype.kind != "M":
        raise ValueError("时间戳中混有不兼容的时区/格式，无法统一为上海本地时间")
    return parsed


def normalize_trading_dates(values: Iterable[object]) -> pd.DatetimeIndex:
    """清洗为升序、去重、无时区的交易日期索引。"""
    parsed = pd.to_datetime(list(values), errors="coerce", format="mixed")
    dates = pd.DatetimeIndex(parsed).dropna()
    if dates.tz is not None:
        dates = dates.tz_convert("Asia/Shanghai").tz_localize(None)
    dates = dates.normalize().unique().sort_values()
    if dates.empty:
        raise ValueError("交易日历没有有效日期")
    return dates


def load_trading_dates(path: str | Path, date_column: str = "date") -> pd.DatetimeIndex:
    """从现有 CSV/Parquet 文件读取交易日列。

    宽表只读取日期列，避免把全部股票行情载入内存。
    """
    source = Path(path).expanduser()
    if not source.is_file():
        raise FileNotFoundError(
            f"交易日历文件不存在: {source}；请通过 LABEL_TRADING_CALENDAR 指向项目原有文件"
        )

    suffix = source.suffix.lower()
    if suffix in {".parquet", ".pq"}:
        frame = pd.read_parquet(source, columns=[date_column])
    elif suffix in {".csv", ".txt"}:
        frame = pd.read_csv(source, usecols=[date_column])
    else:
        raise ValueError(f"不支持的交易日历格式 {suffix!r}，仅支持 CSV/Parquet: {source}")

    if date_column not in frame.columns:
        raise ValueError(f"交易日历缺少日期列 {date_column!r}: {source}")
    return normalize_trading_dates(frame[date_column])


def align_to_trading_day(
    timestamps: pd.Series | Iterable[object],
    trading_dates: pd.DatetimeIndex | Iterable[object],
    cutoff: dt.time = DEFAULT_CUTOFF,
) -> pd.Series:
    """按 14:57 规则将时间戳映射到信息可用的交易日。

    - 交易日且时间不晚于 cutoff：当日；
    - 交易日且晚于 cutoff：下一交易日；
    - 非交易日：下一交易日。

    超出交易日历右边界或无效的时间戳返回 ``NaT``，绝不裁剪到最后一个交易日。
    """
    source = timestamps if isinstance(timestamps, pd.Series) else pd.Series(timestamps)
    parsed = parse_market_timestamps(source)

    calendar = normalize_trading_dates(trading_dates)
    publish_dates = parsed.dt.normalize()
    cutoff_delta = pd.Timedelta(
        hours=cutoff.hour,
        minutes=cutoff.minute,
        seconds=cutoff.second,
        microseconds=cutoff.microsecond,
    )
    same_day = publish_dates.isin(calendar) & ((parsed - publish_dates) <= cutoff_delta)

    result = pd.Series(pd.NaT, index=source.index, dtype="datetime64[ns]")
    result.loc[same_day] = publish_dates.loc[same_day]

    needs_next = parsed.notna() & ~same_day
    if needs_next.any():
        calendar_values = calendar.to_numpy(dtype="datetime64[ns]")
        dates_values = publish_dates.loc[needs_next].to_numpy(dtype="datetime64[ns]")
        positions = np.asarray(
            np.searchsorted(calendar_values, dates_values, side="right"),
            dtype=np.intp,
        )
        in_range = positions < len(calendar_values)
        target_index = publish_dates.loc[needs_next].index[in_range]
        result.loc[target_index] = calendar_values[positions[in_range]]
    return result


def trading_month_ends(
    trading_dates: pd.DatetimeIndex | Iterable[object],
    start: str | pd.Timestamp | pd.Period,
    end: str | pd.Timestamp | pd.Period,
) -> pd.DatetimeIndex:
    """返回区间内每个月最后一个交易日；任何月份缺失都会报错。"""
    calendar = normalize_trading_dates(trading_dates)
    start_period = pd.Period(start, freq="M")
    end_period = pd.Period(end, freq="M")
    if end_period < start_period:
        raise ValueError(f"月份区间倒置: {start_period} > {end_period}")

    periods = pd.period_range(start_period, end_period, freq="M")
    grouped = pd.Series(calendar, index=calendar.to_period("M")).groupby(level=0).max()
    missing = periods.difference(grouped.index)
    if len(missing):
        raise ValueError("交易日历缺少月份: " + ", ".join(map(str, missing)))
    return pd.DatetimeIndex(grouped.loc[periods].to_numpy())


def trading_month_end_map(
    trading_dates: pd.DatetimeIndex | Iterable[object],
    start: str | pd.Timestamp | pd.Period,
    end: str | pd.Timestamp | pd.Period,
) -> dict[pd.Period, pd.Timestamp]:
    """生成 ``月度 Period -> 当月最后交易日`` 映射。"""
    ends = trading_month_ends(trading_dates, start, end)
    return {date.to_period("M"): date for date in ends}


def shift_to_trading_month_end(
    date: object,
    months: int,
    month_ends: dict[pd.Period, pd.Timestamp],
) -> pd.Timestamp:
    """将日期平移若干月并返回目标月最后交易日，覆盖不足时返回 ``NaT``。"""
    if pd.isna(date):
        return pd.NaT
    target = pd.Timestamp(date).to_period("M") + months
    return month_ends.get(target, pd.NaT)
