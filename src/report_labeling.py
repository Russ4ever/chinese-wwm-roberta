"""单篇研报未来验证 Label 的纯计算核心。

本模块不访问网络或数据库。调用方负责提供原始研报、已有交易日历和可选的
结构化 Actual；这里仅完成规范化、时点一致的同行快照和三类报告级 Label。
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from numba import njit, prange

from .trading_calendar import (
    align_to_trading_day,
    normalize_trading_dates,
    parse_market_timestamps,
)


LABEL_VERSION = "report_future_v1.0"
MAD_SCALE = 1.4826
DISPERSION_EPS = 1e-6


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _stock_code(values: pd.Series) -> pd.Series:
    return (
        values.astype("string")
        .str.replace(r"\..*$", "", regex=True)
        .str.replace(r"\.0$", "", regex=True)
        .str.zfill(6)
    )


def _text(values: pd.Series) -> pd.Series:
    return values.fillna("").astype(str).str.strip()


def _report_ids(frame: pd.DataFrame) -> pd.Series:
    # 使用纳秒整数，避免微秒格式化截断精确发布时间。
    timestamp = frame["publish_timestamp"].astype("int64").astype("string")
    material = (
        frame["stock_code"]
        + "|"
        + frame["org_id"]
        + "|"
        + frame["author_name"]
        + "|"
        + timestamp
        + "|"
        + frame["title"]
    )
    return material.map(_sha256)


def canonicalize_report_rows(
    raw: pd.DataFrame,
    trading_dates: Iterable[object],
    *,
    forecast_horizons: Sequence[int] = (0, 1, 2),
    forecast_multiplier: float = 1.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """把原始预测行规范成报告表和 ``报告×FY`` 表。

    ``raw`` 至少需要 STOCK_CODE/ORGAN_NAME/AUTHOR_NAME/TITLE/CREATE_DATE/
    REPORT_YEAR/FORECAST_NP。CONTENT 可缺省；命令入口会在第二遍流式读取时补齐正文。
    """
    required = {
        "STOCK_CODE",
        "ORGAN_NAME",
        "AUTHOR_NAME",
        "TITLE",
        "CREATE_DATE",
        "REPORT_YEAR",
        "FORECAST_NP",
    }
    missing = sorted(required.difference(raw.columns))
    if missing:
        raise ValueError("研报数据缺少字段: " + ", ".join(missing))
    if forecast_multiplier <= 0:
        raise ValueError("forecast_multiplier 必须为正")

    frame = raw.copy()
    frame["stock_code"] = _stock_code(frame["STOCK_CODE"])
    frame["org_id"] = _text(frame["ORGAN_NAME"])
    frame["author_name"] = _text(frame["AUTHOR_NAME"])
    frame["title"] = _text(frame["TITLE"])
    frame["content"] = _text(frame["CONTENT"]) if "CONTENT" in frame else ""
    frame["publish_timestamp"] = parse_market_timestamps(frame["CREATE_DATE"])
    frame["publish_date"] = frame["publish_timestamp"].dt.normalize()
    frame["available_date"] = align_to_trading_day(
        frame["publish_timestamp"], trading_dates
    )
    frame["fy"] = (
        pd.to_numeric(frame["REPORT_YEAR"], errors="coerce").round().astype("Int64")
    )
    frame["forecast_new"] = pd.to_numeric(frame["FORECAST_NP"], errors="coerce")
    frame["forecast_new"] = frame["forecast_new"] * float(forecast_multiplier)
    valid = (
        frame["stock_code"].str.fullmatch(r"\d{6}", na=False)
        & frame["org_id"].ne("")
        & frame["publish_timestamp"].notna()
        & frame["available_date"].notna()
        & frame["fy"].notna()
        & frame["forecast_new"].notna()
        & np.isfinite(frame["forecast_new"])
    )
    frame = frame.loc[valid].copy()
    if frame.empty:
        empty_reports = pd.DataFrame(
            columns=[
                "report_id",
                "stock_code",
                "org_id",
                "author_name",
                "title",
                "content",
            ]
        )
        return empty_reports, pd.DataFrame()

    frame["fy"] = frame["fy"].astype(int)
    frame["forecast_horizon"] = frame["fy"] - frame["available_date"].dt.year
    frame = frame[
        frame["forecast_horizon"].isin([int(x) for x in forecast_horizons])
    ].copy()
    if frame.empty:
        return pd.DataFrame(), pd.DataFrame()
    frame["report_id"] = _report_ids(frame)
    source_ids = (
        frame.get("ID", pd.Series("", index=frame.index))
        .astype("string")
        .fillna("")
        .str.strip()
    )
    fallback = (
        frame["report_id"]
        + "|"
        + frame["fy"].astype(str)
        + "|"
        + frame["forecast_new"].astype(str)
    ).map(_sha256)
    frame["source_report_id"] = source_ids.mask(source_ids.eq(""), fallback)

    report_source_ids = (
        frame.groupby("report_id", sort=False)["source_report_id"]
        .agg(
            lambda values: json.dumps(sorted(set(map(str, values))), ensure_ascii=False)
        )
        .rename("source_report_ids")
    )
    content_length = frame["content"].str.len()
    report_idx = content_length.groupby(frame["report_id"]).idxmax()
    reports = frame.loc[
        report_idx,
        [
            "report_id",
            "stock_code",
            "org_id",
            "author_name",
            "title",
            "content",
            "publish_timestamp",
            "publish_date",
            "available_date",
        ],
    ].copy()
    reports = reports.merge(report_source_ids, on="report_id", how="left")
    reports["text"] = np.where(
        reports["content"].ne(""),
        reports["title"] + "。" + reports["content"],
        reports["title"],
    )
    reports = reports.sort_values(["available_date", "report_id"]).reset_index(
        drop=True
    )

    import polars as pl

    frame = frame.sort_values(
        ["report_id", "fy", "publish_timestamp", "source_report_id"]
    )
    first_forecast = frame.groupby(["report_id", "fy"], sort=False)[
        "forecast_new"
    ].transform("first")
    frame["_forecast_conflict_piece"] = (
        frame["forecast_new"] - first_forecast
    ).abs() > (1e-6 + 1e-9 * first_forecast.abs())
    summary = (
        pl.from_pandas(frame)
        .group_by(["report_id", "fy"], maintain_order=True)
        .agg(
            pl.col("stock_code").first(),
            pl.col("org_id").first(),
            pl.col("author_name").first(),
            pl.col("title").first(),
            pl.col("publish_timestamp").first(),
            pl.col("publish_date").first(),
            pl.col("available_date").first(),
            pl.col("forecast_horizon").first(),
            pl.col("forecast_new").first(),
            pl.col("source_report_id").unique().sort().alias("_source_ids"),
            pl.col("forecast_new").unique().sort().alias("_forecast_values"),
            pl.col("_forecast_conflict_piece")
            .max()
            .cast(pl.Int8)
            .alias("forecast_conflict"),
            pl.len().alias("n_source_rows"),
        )
        .to_pandas()
    )
    summary["source_report_ids"] = summary.pop("_source_ids").map(
        lambda values: json.dumps(list(values), ensure_ascii=False)
    )
    summary["forecast_values"] = summary.pop("_forecast_values").map(
        lambda values: json.dumps(
            [float(value) for value in values], ensure_ascii=False
        )
    )
    summary.loc[summary["forecast_conflict"].eq(1), "forecast_new"] = np.nan
    rows = summary[
        [
            "report_id",
            "source_report_ids",
            "stock_code",
            "org_id",
            "author_name",
            "title",
            "publish_timestamp",
            "publish_date",
            "available_date",
            "fy",
            "forecast_horizon",
            "forecast_new",
            "forecast_values",
            "forecast_conflict",
            "n_source_rows",
        ]
    ].copy()
    rows["fy"] = rows["fy"].astype(int)
    rows["forecast_horizon"] = rows["forecast_horizon"].astype(int)
    rows = rows.sort_values(
        ["stock_code", "fy", "available_date", "report_id"]
    ).reset_index(drop=True)
    return reports, rows


@dataclass(frozen=True)
class PackedHistories:
    """Numba友好的股票×FY×机构预测历史压缩数组。"""

    group_codes: dict[tuple[str, int], int]
    org_codes: dict[str, int]
    group_org_offsets: np.ndarray
    org_ids: np.ndarray
    org_event_offsets: np.ndarray
    event_dates: np.ndarray
    event_publish_dates: np.ndarray
    event_values: np.ndarray


OrgHistories = PackedHistories | dict[object, object]


def build_org_histories(rows: pd.DataFrame) -> OrgHistories:
    """按股票×FY构造机构日终预测历史的连续数组。"""
    if rows.empty:
        return {}
    usable = rows[
        rows["forecast_conflict"].eq(0)
        & rows["forecast_new"].notna()
        & np.isfinite(rows["forecast_new"])
    ]
    if usable.empty:
        return {}
    import polars as pl

    daily = (
        pl.from_pandas(
            usable[
                [
                    "stock_code",
                    "fy",
                    "org_id",
                    "available_date",
                    "publish_date",
                    "forecast_new",
                ]
            ]
        )
        .group_by(["stock_code", "fy", "org_id", "available_date"])
        .agg(
            pl.col("forecast_new").median().alias("forecast"),
            pl.col("publish_date").max().alias("publish_date"),
        )
        .sort(["stock_code", "fy", "org_id", "available_date"])
        .to_pandas()
    )
    group_values = np.empty(len(daily), dtype=object)
    group_values[:] = list(
        zip(daily["stock_code"].astype(str), daily["fy"].astype(int))
    )
    group_ids, group_uniques = pd.factorize(group_values, sort=True)
    org_ids, org_uniques = pd.factorize(daily["org_id"].astype(str), sort=True)
    order = np.lexsort(
        (
            daily["available_date"].to_numpy(dtype="datetime64[D]").astype(np.int64),
            org_ids,
            group_ids,
        )
    )
    group_ids = np.asarray(group_ids[order], dtype=np.int32)
    org_ids = np.asarray(org_ids[order], dtype=np.int32)
    event_dates = (
        daily["available_date"].to_numpy(dtype="datetime64[D]").astype(np.int64)[order]
    )
    event_publish_dates = (
        daily["publish_date"].to_numpy(dtype="datetime64[D]").astype(np.int64)[order]
    )
    event_values = daily["forecast"].to_numpy(dtype=np.float64)[order]

    pair_starts = np.flatnonzero(
        np.r_[True, (group_ids[1:] != group_ids[:-1]) | (org_ids[1:] != org_ids[:-1])]
    ).astype(np.int64)
    pair_groups = group_ids[pair_starts]
    n_groups = len(group_uniques)
    pair_counts = np.bincount(pair_groups, minlength=n_groups)
    group_org_offsets = np.empty(n_groups + 1, dtype=np.int64)
    group_org_offsets[0] = 0
    np.cumsum(pair_counts, out=group_org_offsets[1:])
    org_event_offsets = np.r_[pair_starts, len(event_dates)].astype(np.int64)

    return PackedHistories(
        group_codes={
            (str(key[0]), int(key[1])): int(code)
            for code, key in enumerate(group_uniques)
        },
        org_codes={str(org): int(code) for code, org in enumerate(org_uniques)},
        group_org_offsets=group_org_offsets,
        org_ids=org_ids[pair_starts],
        org_event_offsets=org_event_offsets,
        event_dates=event_dates,
        event_publish_dates=event_publish_dates,
        event_values=event_values,
    )


@njit(cache=True, inline="always")
def _rightmost_event(
    dates: np.ndarray, begin: int, end: int, target: int, strict: bool
) -> int:
    left = begin
    right = end
    while left < right:
        middle = (left + right) // 2
        if dates[middle] < target or (not strict and dates[middle] == target):
            left = middle + 1
        else:
            right = middle
    return left - 1


@njit(cache=True, inline="always")
def _median_mad_numba(values: np.ndarray, count: int) -> tuple[float, float]:
    selected = values[:count].copy()
    median = np.median(selected)
    deviations = np.empty(count, dtype=np.float64)
    for index in range(count):
        deviations[index] = abs(selected[index] - median)
    return median, np.median(deviations)


@njit(cache=True, parallel=True)
def _pre_consensus_kernel(
    row_groups: np.ndarray,
    row_orgs: np.ndarray,
    row_dates: np.ndarray,
    group_org_offsets: np.ndarray,
    history_orgs: np.ndarray,
    org_event_offsets: np.ndarray,
    event_dates: np.ndarray,
    event_publish_dates: np.ndarray,
    event_values: np.ndarray,
    lookback_days: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_rows = len(row_dates)
    consensus = np.full(n_rows, np.nan)
    mad = np.full(n_rows, np.nan)
    counts = np.zeros(n_rows, dtype=np.int32)
    for row_index in prange(n_rows):
        group = row_groups[row_index]
        if group < 0:
            continue
        pair_begin = group_org_offsets[group]
        pair_end = group_org_offsets[group + 1]
        values = np.empty(pair_end - pair_begin, dtype=np.float64)
        count = 0
        target = row_dates[row_index]
        for pair in range(pair_begin, pair_end):
            if history_orgs[pair] == row_orgs[row_index]:
                continue
            event = _rightmost_event(
                event_dates,
                org_event_offsets[pair],
                org_event_offsets[pair + 1],
                target,
                True,
            )
            if (
                event >= org_event_offsets[pair]
                and target - event_publish_dates[event] <= lookback_days
            ):
                values[count] = event_values[event]
                count += 1
        counts[row_index] = count
        if count:
            consensus[row_index], mad[row_index] = _median_mad_numba(values, count)
    return consensus, mad, counts


def attach_pre_consensus(
    rows: pd.DataFrame,
    histories: OrgHistories,
    *,
    coverage_start: pd.Timestamp,
    lookback_days: int,
    min_peer_orgs: int,
) -> pd.DataFrame:
    """用并行压缩数组为报告×FY附加发布前同行共识。"""
    out = rows.copy()
    if isinstance(histories, dict):
        consensus = np.full(len(out), np.nan)
        mad = np.full(len(out), np.nan)
        counts = np.zeros(len(out), dtype=np.int32)
    else:
        row_groups = np.asarray(
            [
                histories.group_codes.get((str(stock), int(fy)), -1)
                for stock, fy in zip(out["stock_code"], out["fy"])
            ],
            dtype=np.int32,
        )
        row_orgs = np.asarray(
            [histories.org_codes.get(str(org), -1) for org in out["org_id"]],
            dtype=np.int32,
        )
        row_dates = (
            out["available_date"].to_numpy(dtype="datetime64[D]").astype(np.int64)
        )
        consensus, mad, counts = _pre_consensus_kernel(
            row_groups,
            row_orgs,
            row_dates,
            histories.group_org_offsets,
            histories.org_ids,
            histories.org_event_offsets,
            histories.event_dates,
            histories.event_publish_dates,
            histories.event_values,
            int(lookback_days),
        )
    history_ok = (
        out["available_date"] - pd.Timedelta(days=lookback_days) >= coverage_start
    ).to_numpy()
    reasons = np.full(len(out), None, dtype=object)
    reasons[out["forecast_conflict"].to_numpy(dtype=int) == 1] = (
        "conflicting_report_forecast"
    )
    reasons[(out["forecast_conflict"].to_numpy(dtype=int) != 1) & ~history_ok] = (
        "left_censored"
    )
    reasons[
        (out["forecast_conflict"].to_numpy(dtype=int) != 1)
        & history_ok
        & (counts < min_peer_orgs)
    ] = "insufficient_pre_peers"
    out["consensus_pre"] = consensus
    out["mad_pre_abs"] = mad
    out["n_org_pre"] = counts
    out["history_window_complete"] = history_ok.astype(int)
    out["pre_invalid_reason"] = reasons
    return out


def assign_point_in_time_scale(
    rows: pd.DataFrame,
    *,
    history_months: int,
    min_samples: int,
    floor_ratio: float = 0.01,
) -> pd.DataFrame:
    """使用搜索窗口而非全表重复布尔扫描拟合 point-in-time scale floor。"""
    if history_months <= 0 or min_samples <= 0 or floor_ratio <= 0:
        raise ValueError("scale 参数必须为正")
    out = rows.copy()
    months = out["available_date"].dt.to_period("M")
    valid_ref = out["consensus_pre"].notna() & out["pre_invalid_reason"].isna()
    reference = out.loc[
        valid_ref, ["available_date", "forecast_horizon", "consensus_pre"]
    ].sort_values("available_date")
    pooled_dates = reference["available_date"].to_numpy(dtype="datetime64[ns]")
    pooled_values = reference["consensus_pre"].abs().to_numpy(dtype=float)
    by_horizon: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for horizon, group in reference.groupby("forecast_horizon", sort=False):
        by_horizon[int(horizon)] = (
            group["available_date"].to_numpy(dtype="datetime64[ns]"),
            group["consensus_pre"].abs().to_numpy(dtype=float),
        )

    lookup: list[dict[str, object]] = []
    target_pairs = (
        pd.DataFrame(
            {"_scale_month": months, "forecast_horizon": out["forecast_horizon"]}
        )
        .dropna()
        .drop_duplicates()
        .sort_values(["_scale_month", "forecast_horizon"])
    )
    for month, horizon_value in target_pairs.itertuples(index=False, name=None):
        horizon = int(horizon_value)
        month_start = month.start_time
        ref_start = (month - history_months).start_time
        start64 = np.datetime64(ref_start)
        end64 = np.datetime64(month_start)
        horizon_dates, horizon_values = by_horizon.get(
            horizon, (np.empty(0, dtype="datetime64[ns]"), np.empty(0))
        )
        same_left = np.searchsorted(horizon_dates, start64, side="left")
        same_right = np.searchsorted(horizon_dates, end64, side="left")
        pooled_left = np.searchsorted(pooled_dates, start64, side="left")
        pooled_right = np.searchsorted(pooled_dates, end64, side="left")
        same_count = int(same_right - same_left)
        pooled_count = int(pooled_right - pooled_left)
        if same_count >= min_samples:
            values = horizon_values[same_left:same_right]
            reference_name = "same_horizon"
            sample_count = same_count
        elif pooled_count >= min_samples:
            values = pooled_values[pooled_left:pooled_right]
            reference_name = "pooled_horizon"
            sample_count = pooled_count
        else:
            continue
        floor = max(floor_ratio * float(np.median(values)), float(np.finfo(float).eps))
        lookup.append(
            {
                "_scale_month": month,
                "forecast_horizon": horizon,
                "scale_reference": reference_name,
                "scale_sample_count": sample_count,
                "scale_floor": floor,
            }
        )
    out["_scale_month"] = months
    if lookup:
        out = out.merge(
            pd.DataFrame(lookup),
            on=["_scale_month", "forecast_horizon"],
            how="left",
            sort=False,
        )
    else:
        out["scale_reference"] = None
        out["scale_sample_count"] = np.nan
        out["scale_floor"] = np.nan
    out["scale_sample_count"] = out["scale_sample_count"].fillna(0).astype(int)
    out = out.drop(columns="_scale_month")
    out["scale_t"] = np.maximum(out["consensus_pre"].abs(), out["scale_floor"])
    out["dispersion_pre"] = MAD_SCALE * out["mad_pre_abs"] / out["scale_t"]
    values = out["consensus_pre"].to_numpy(dtype=float)
    floors = out["scale_floor"].to_numpy(dtype=float)
    state = np.full(len(out), None, dtype=object)
    finite = np.isfinite(values) & np.isfinite(floors)
    state[finite & (np.abs(values) <= floors)] = "near_zero"
    state[finite & (np.abs(values) > floors) & (values > 0)] = "profit"
    state[finite & (np.abs(values) > floors) & (values < 0)] = "loss"
    out["pre_state"] = state
    out["pre_label_valid"] = (
        out["pre_invalid_reason"].isna() & out["scale_floor"].notna()
    ).astype(int)
    out["pre_label_invalid_reason"] = out["pre_invalid_reason"]
    missing_scale = out["pre_label_invalid_reason"].isna() & out["scale_floor"].isna()
    out.loc[missing_scale, "pre_label_invalid_reason"] = "insufficient_scale_history"
    return out


def normalize_actuals(actuals: pd.DataFrame) -> pd.DataFrame:
    """规范 canonical Actual 表，并选择首次正式年报记录。"""
    columns = [
        "stock_code",
        "fy",
        "actual_np",
        "actual_publish_date",
        "actual_known_date",
        "unit_multiplier",
        "currency",
        "source_id",
        "actual_version",
        "actual_metric",
    ]
    if actuals.empty:
        return pd.DataFrame(columns=columns)
    required = {"stock_code", "fy", "actual_np", "actual_publish_date"}
    missing = sorted(required.difference(actuals.columns))
    if missing:
        raise ValueError("Actual 数据缺少 canonical 字段: " + ", ".join(missing))
    frame = actuals.copy()
    frame["stock_code"] = _stock_code(frame["stock_code"])
    frame["fy"] = pd.to_numeric(frame["fy"], errors="coerce").round().astype("Int64")
    frame["actual_np"] = pd.to_numeric(frame["actual_np"], errors="coerce")
    frame["actual_publish_date"] = pd.to_datetime(
        frame["actual_publish_date"], errors="coerce"
    ).dt.normalize()
    if "actual_known_date" not in frame:
        frame["actual_known_date"] = frame["actual_publish_date"]
    else:
        known = (
            pd.to_datetime(frame["actual_known_date"], errors="coerce")
            .dt.normalize()
            .fillna(frame["actual_publish_date"])
        )
        frame["actual_known_date"] = pd.concat(
            [known, frame["actual_publish_date"]], axis=1
        ).min(axis=1)
    if "unit_multiplier" not in frame:
        frame["unit_multiplier"] = 1.0
    frame["unit_multiplier"] = pd.to_numeric(frame["unit_multiplier"], errors="coerce")
    frame["actual_np"] = frame["actual_np"] * frame["unit_multiplier"]
    if "currency" not in frame:
        frame["currency"] = "CNY"
    frame["currency"] = _text(frame["currency"]).str.upper()
    if "source_id" not in frame:
        frame["source_id"] = ""
    frame["source_id"] = _text(frame["source_id"])
    if "actual_version" not in frame:
        frame["actual_version"] = "annual_consolidated_first_actual_announcement"
    frame["actual_version"] = _text(frame["actual_version"])
    if "actual_metric" not in frame:
        frame["actual_metric"] = "net_profit_incl_min_int_inc"
    frame["actual_metric"] = _text(frame["actual_metric"])
    valid = (
        frame["stock_code"].str.fullmatch(r"\d{6}", na=False)
        & frame["fy"].notna()
        & frame["actual_np"].notna()
        & np.isfinite(frame["actual_np"])
        & frame["unit_multiplier"].gt(0)
        & frame["actual_publish_date"].notna()
        & frame["currency"].eq("CNY")
    )
    frame = frame.loc[valid].copy()
    frame["fy"] = frame["fy"].astype(int)
    frame = frame.sort_values(["stock_code", "fy", "actual_publish_date", "source_id"])
    frame = frame.drop_duplicates(["stock_code", "fy"], keep="first")
    return frame[columns].reset_index(drop=True)


def attach_actual_labels(
    rows: pd.DataFrame,
    actuals: pd.DataFrame,
    *,
    actual_source_available: bool,
) -> pd.DataFrame:
    """计算 Report Residual 与 Report Edge。"""
    out = rows.copy()
    normalized = normalize_actuals(actuals)
    if normalized.empty:
        for column in normalized.columns:
            if column not in {"stock_code", "fy"}:
                out[column] = pd.NaT if column.endswith("_date") else np.nan
        out["residual_signed_raw"] = np.nan
        out["report_abs_error"] = np.nan
        out["consensus_abs_error"] = np.nan
        out["edge_raw"] = np.nan
        out["edge_sign"] = None
        out["days_to_actual"] = np.nan
        out["actual_transition_state"] = None
        out["actual_label_available_date"] = pd.NaT
        out["residual_valid"] = 0
        out["edge_valid"] = 0
        unavailable_reason = (
            "actual_missing" if actual_source_available else "actual_source_unavailable"
        )
        out["actual_invalid_reason"] = unavailable_reason
        return out

    out = out.merge(normalized, on=["stock_code", "fy"], how="left")
    reasons = np.full(len(out), None, dtype=object)
    pre_invalid = out["pre_label_valid"].to_numpy(dtype=int) != 1
    reasons[pre_invalid] = out.loc[pre_invalid, "pre_label_invalid_reason"].astype(str)
    actual_missing = ~pre_invalid & out["actual_np"].isna().to_numpy()
    reasons[actual_missing] = "actual_missing"
    disclosed = (
        ~pre_invalid
        & ~actual_missing
        & (out["available_date"] >= out["actual_known_date"]).to_numpy()
    )
    reasons[disclosed] = "report_not_before_actual_known_date"
    valid = pd.isna(reasons)

    actual = out["actual_np"].to_numpy(dtype=float)
    forecast = out["forecast_new"].to_numpy(dtype=float)
    pre = out["consensus_pre"].to_numpy(dtype=float)
    scale = out["scale_t"].to_numpy(dtype=float)
    residual = (actual - forecast) / scale
    report_error = np.abs(actual - forecast) / scale
    consensus_error = np.abs(actual - pre) / scale
    edge = consensus_error - report_error
    for values in (residual, report_error, consensus_error, edge):
        values[~valid] = np.nan

    signs = np.full(len(out), None, dtype=object)
    signs[valid & (edge > 0)] = "positive"
    signs[valid & (edge < 0)] = "negative"
    signs[valid & (edge == 0)] = "zero"
    days = (out["actual_publish_date"] - out["available_date"]).dt.days.to_numpy(
        dtype=float
    )
    days[~valid] = np.nan

    floor = out["scale_floor"].to_numpy(dtype=float)
    before_states = np.full(len(out), None, dtype=object)
    actual_states = np.full(len(out), None, dtype=object)
    comparable = valid & np.isfinite(floor)
    before_states[comparable & (np.abs(pre) <= floor)] = "near_zero"
    before_states[comparable & (np.abs(pre) > floor) & (pre > 0)] = "profit"
    before_states[comparable & (np.abs(pre) > floor) & (pre < 0)] = "loss"
    actual_states[comparable & (np.abs(actual) <= floor)] = "near_zero"
    actual_states[comparable & (np.abs(actual) > floor) & (actual > 0)] = "profit"
    actual_states[comparable & (np.abs(actual) > floor) & (actual < 0)] = "loss"
    transitions = np.full(len(out), None, dtype=object)
    transition_valid = comparable & pd.notna(before_states) & pd.notna(actual_states)
    transitions[transition_valid] = np.char.add(
        np.char.add(before_states[transition_valid].astype(str), "_to_"),
        actual_states[transition_valid].astype(str),
    )

    out["residual_signed_raw"] = residual
    out["report_abs_error"] = report_error
    out["consensus_abs_error"] = consensus_error
    out["edge_raw"] = edge
    out["edge_sign"] = signs
    out["days_to_actual"] = days
    out["actual_transition_state"] = transitions
    out["actual_label_available_date"] = out["actual_publish_date"]
    out["residual_valid"] = valid.astype(int)
    out["edge_valid"] = valid.astype(int)
    out["actual_invalid_reason"] = reasons
    return out


def validate_actual_scale(
    rows: pd.DataFrame, *, lower: float = 0.2, upper: float = 5.0
) -> float | None:
    """检查 Actual 与发布前共识是否处于合理数量级。"""
    matched = rows[
        rows["actual_np"].notna()
        & rows["consensus_pre"].notna()
        & rows["consensus_pre"].abs().gt(np.finfo(float).eps)
    ]
    if matched.empty:
        return None
    ratio = float(
        np.median(np.abs(matched["actual_np"]) / np.abs(matched["consensus_pre"]))
    )
    if not lower <= ratio <= upper:
        raise ValueError(
            f"Actual/共识数量级中位比 {ratio:.6g} 不在 [{lower}, {upper}]，请检查元/万元映射"
        )
    return ratio


def first_trading_date_on_or_after(
    date: object,
    trading_dates: Iterable[object],
) -> pd.Timestamp:
    calendar = normalize_trading_dates(trading_dates)
    target = pd.Timestamp(date).normalize()
    position = int(
        np.searchsorted(
            calendar.to_numpy(dtype="datetime64[ns]"), np.datetime64(target)
        )
    )
    return pd.Timestamp(calendar[position]) if position < len(calendar) else pd.NaT


@njit(cache=True, parallel=True)
def _confirmation_kernel(
    row_groups: np.ndarray,
    row_orgs: np.ndarray,
    start_dates: np.ndarray,
    target_dates: np.ndarray,
    forecasts: np.ndarray,
    scales: np.ndarray,
    pre_valid: np.ndarray,
    actual_known_dates: np.ndarray,
    group_org_offsets: np.ndarray,
    history_orgs: np.ndarray,
    org_event_offsets: np.ndarray,
    event_dates: np.ndarray,
    event_publish_dates: np.ndarray,
    event_values: np.ndarray,
    coverage_end: int,
    lookback_days: int,
    min_peer_orgs: int,
    min_active_orgs: int,
    min_probe_updates: int,
    nat_value: int,
) -> tuple:
    n_rows, n_horizons = target_dates.shape
    total = n_rows * n_horizons * 3
    consensus_pre = np.full(total, np.nan)
    consensus_future = np.full(total, np.nan)
    mad_pre = np.full(total, np.nan)
    mad_future = np.full(total, np.nan)
    dispersion_pre = np.full(total, np.nan)
    dispersion_future = np.full(total, np.nan)
    progress = np.full(total, np.nan)
    progress_clipped = np.full(total, np.nan)
    delta_dispersion = np.full(total, np.nan)
    n_org_pre = np.zeros(total, dtype=np.int32)
    n_org_future = np.zeros(total, dtype=np.int32)
    n_updates = np.zeros(total, dtype=np.int32)
    n_entries = np.zeros(total, dtype=np.int32)
    n_exits = np.zeros(total, dtype=np.int32)
    valid = np.zeros(total, dtype=np.int8)
    probe_valid = np.zeros(total, dtype=np.int8)
    reason = np.zeros(total, dtype=np.int8)
    probe_reason = np.zeros(total, dtype=np.int8)
    weights = np.zeros(total, dtype=np.float64)

    for task in prange(n_rows * n_horizons):
        row = task // n_horizons
        horizon = task - row * n_horizons
        output_start = task * 3
        target = target_dates[row, horizon]
        if pre_valid[row] != 1:
            reason[output_start : output_start + 3] = 1
            continue
        if target == nat_value or target > coverage_end:
            reason[output_start : output_start + 3] = 2
            continue
        actual_date = actual_known_dates[row]
        if actual_date != nat_value and target >= actual_date:
            reason[output_start : output_start + 3] = 3
            continue
        group = row_groups[row]
        if group < 0:
            reason[output_start : output_start + 3] = 4
            reason[output_start + 2] = 5
            continue

        pair_begin = group_org_offsets[group]
        pair_end = group_org_offsets[group + 1]
        capacity = pair_end - pair_begin
        pre_values = np.empty(capacity, dtype=np.float64)
        fixed_future_values = np.empty(capacity, dtype=np.float64)
        active_pre_values = np.empty(capacity, dtype=np.float64)
        active_future_values = np.empty(capacity, dtype=np.float64)
        market_future_values = np.empty(capacity, dtype=np.float64)
        pre_count = 0
        active_count = 0
        market_count = 0
        entries = 0
        exits = 0
        start = start_dates[row]

        for pair in range(pair_begin, pair_end):
            if history_orgs[pair] == row_orgs[row]:
                continue
            event_begin = org_event_offsets[pair]
            event_end = org_event_offsets[pair + 1]
            pre_event = _rightmost_event(
                event_dates, event_begin, event_end, start, True
            )
            has_pre = (
                pre_event >= event_begin
                and start - event_publish_dates[pre_event] <= lookback_days
            )
            latest_event = _rightmost_event(
                event_dates, event_begin, event_end, target, False
            )
            has_update = (
                has_pre
                and latest_event >= event_begin
                and event_dates[latest_event] > start
            )

            if has_pre:
                old_value = event_values[pre_event]
                pre_values[pre_count] = old_value
                if has_update:
                    fixed_future_values[pre_count] = event_values[latest_event]
                    active_pre_values[active_count] = old_value
                    active_future_values[active_count] = event_values[latest_event]
                    active_count += 1
                else:
                    fixed_future_values[pre_count] = old_value
                pre_count += 1

            market_event = latest_event
            if market_event >= event_begin and event_dates[market_event] == start:
                market_event = pre_event if has_pre else event_begin - 1
            has_market = (
                market_event >= event_begin
                and target - event_publish_dates[market_event] <= lookback_days
            )
            if has_market:
                market_future_values[market_count] = event_values[market_event]
                market_count += 1
                if not has_pre:
                    entries += 1
            elif has_pre:
                exits += 1

        for panel in range(3):
            output = output_start + panel
            n_updates[output] = active_count
            if panel == 0:
                panel_pre = pre_values
                panel_future = fixed_future_values
                count_pre = pre_count
                count_future = pre_count
                required = min_peer_orgs
            elif panel == 1:
                panel_pre = pre_values
                panel_future = market_future_values
                count_pre = pre_count
                count_future = market_count
                required = min_peer_orgs
                n_entries[output] = entries
                n_exits[output] = exits
            else:
                panel_pre = active_pre_values
                panel_future = active_future_values
                count_pre = active_count
                count_future = active_count
                required = min_active_orgs
            n_org_pre[output] = count_pre
            n_org_future[output] = count_future
            if count_pre < required or count_future < required:
                reason[output] = 5 if panel == 2 else 4
                continue

            c_pre, panel_mad_pre = _median_mad_numba(panel_pre, count_pre)
            c_future, panel_mad_future = _median_mad_numba(panel_future, count_future)
            scale = scales[row]
            d_pre = MAD_SCALE * panel_mad_pre / scale
            d_future = MAD_SCALE * panel_mad_future / scale
            gap = forecasts[row] - c_pre
            inline_band = max(0.01, 0.5 * d_pre) * scale
            consensus_pre[output] = c_pre
            consensus_future[output] = c_future
            mad_pre[output] = panel_mad_pre
            mad_future[output] = panel_mad_future
            dispersion_pre[output] = d_pre
            dispersion_future[output] = d_future
            if abs(gap) <= inline_band:
                reason[output] = 6
                continue

            value = (c_future - c_pre) / gap
            progress[output] = value
            progress_clipped[output] = min(2.0, max(-1.0, value))
            delta_dispersion[output] = math.log(
                (d_future + DISPERSION_EPS) / (d_pre + DISPERSION_EPS)
            )
            coverage_weight = min(1.0, math.log1p(count_pre) / math.log(11.0))
            update_weight = (
                min(1.0, math.log1p(active_count) / math.log(4.0))
                if active_count > 0
                else 0.0
            )
            valid[output] = 1
            is_probe_valid = panel == 2 or active_count >= min_probe_updates
            probe_valid[output] = 1 if is_probe_valid else 0
            probe_reason[output] = 0 if is_probe_valid else 1
            weights[output] = math.sqrt(coverage_weight * update_weight)
    return (
        consensus_pre,
        consensus_future,
        mad_pre,
        mad_future,
        dispersion_pre,
        dispersion_future,
        progress,
        progress_clipped,
        delta_dispersion,
        n_org_pre,
        n_org_future,
        n_updates,
        n_entries,
        n_exits,
        valid,
        probe_valid,
        reason,
        probe_reason,
        weights,
    )


def build_confirmation_labels(
    rows: pd.DataFrame,
    histories: OrgHistories,
    trading_dates: Iterable[object],
    *,
    coverage_end: pd.Timestamp,
    confirmation_months: Sequence[int],
    lookback_days: int,
    min_peer_orgs: int,
    min_active_orgs: int,
    min_probe_updates: int,
) -> pd.DataFrame:
    """一次同行扫描并行构造 fixed/market/active 三种未来确认 Label。"""
    if rows.empty:
        return pd.DataFrame()
    calendar = normalize_trading_dates(trading_dates)
    months_array = np.asarray(
        [int(value) for value in confirmation_months], dtype=np.int32
    )
    n_rows = len(rows)
    n_horizons = len(months_array)
    start_series = pd.to_datetime(rows["available_date"])
    calendar_days = calendar.to_numpy(dtype="datetime64[D]").astype(np.int64)
    nat_value = np.datetime64("NaT", "D").astype(np.int64)
    targets = np.full((n_rows, n_horizons), nat_value, dtype=np.int64)
    for horizon_index, months in enumerate(months_array):
        nominal = (
            (start_series + pd.DateOffset(months=int(months)))
            .to_numpy(dtype="datetime64[D]")
            .astype(np.int64)
        )
        positions = np.searchsorted(calendar_days, nominal, side="left")
        valid_positions = positions < len(calendar_days)
        targets[valid_positions, horizon_index] = calendar_days[
            positions[valid_positions]
        ]

    if isinstance(histories, dict):
        packed = PackedHistories(
            {},
            {},
            np.zeros(1, dtype=np.int64),
            np.empty(0, dtype=np.int32),
            np.zeros(1, dtype=np.int64),
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=float),
        )
    else:
        packed = histories
    row_groups = np.asarray(
        [
            packed.group_codes.get((str(stock), int(fy)), -1)
            for stock, fy in zip(rows["stock_code"], rows["fy"])
        ],
        dtype=np.int32,
    )
    row_orgs = np.asarray(
        [packed.org_codes.get(str(org), -1) for org in rows["org_id"]], dtype=np.int32
    )
    start_days = start_series.to_numpy(dtype="datetime64[D]").astype(np.int64)
    if "actual_known_date" in rows:
        actual_days = (
            pd.to_datetime(rows["actual_known_date"], errors="coerce")
            .to_numpy(dtype="datetime64[D]")
            .astype(np.int64)
        )
    else:
        actual_days = np.full(n_rows, nat_value, dtype=np.int64)
    results = _confirmation_kernel(
        row_groups,
        row_orgs,
        start_days,
        targets,
        rows["forecast_new"].to_numpy(dtype=np.float64),
        rows["scale_t"].to_numpy(dtype=np.float64),
        rows["pre_label_valid"].to_numpy(dtype=np.int8),
        actual_days,
        packed.group_org_offsets,
        packed.org_ids,
        packed.org_event_offsets,
        packed.event_dates,
        packed.event_publish_dates,
        packed.event_values,
        np.datetime64(coverage_end.normalize(), "D").astype(np.int64),
        int(lookback_days),
        int(min_peer_orgs),
        int(min_active_orgs),
        int(min_probe_updates),
        int(nat_value),
    )
    (
        consensus_pre,
        consensus_future,
        mad_pre,
        mad_future,
        dispersion_pre,
        dispersion_future,
        progress,
        progress_clipped,
        delta_dispersion,
        n_org_pre,
        n_org_future,
        n_updates,
        n_entries,
        n_exits,
        valid,
        probe_valid,
        reason_codes,
        probe_reason_codes,
        weights,
    ) = results

    row_index = np.repeat(np.arange(n_rows), n_horizons * 3)
    month_values = np.tile(np.repeat(months_array, 3), n_rows)
    panels = np.tile(
        np.asarray(["fixed", "market", "active"], dtype=object), n_rows * n_horizons
    )
    target_flat = (
        np.repeat(targets.reshape(-1), 3)
        .astype("datetime64[D]")
        .astype("datetime64[ns]")
    )
    output = pd.DataFrame(
        {
            "report_id": rows["report_id"].to_numpy()[row_index],
            "stock_code": rows["stock_code"].to_numpy()[row_index],
            "fy": rows["fy"].to_numpy(dtype=int)[row_index],
            "forecast_horizon": rows["forecast_horizon"].to_numpy(dtype=int)[row_index],
            "available_date": start_series.to_numpy()[row_index],
            "confirmation_months": month_values,
            "peer_panel": panels,
            "target_date": target_flat,
            "forecast_new": rows["forecast_new"].to_numpy(dtype=float)[row_index],
            "scale_t": rows["scale_t"].to_numpy(dtype=float)[row_index],
            "consensus_pre": consensus_pre,
            "consensus_future": consensus_future,
            "mad_pre_abs": mad_pre,
            "mad_future_abs": mad_future,
            "dispersion_pre": dispersion_pre,
            "dispersion_future": dispersion_future,
            "progress_raw": progress,
            "progress_clipped": progress_clipped,
            "delta_log_dispersion": delta_dispersion,
            "n_org_pre": n_org_pre,
            "n_org_future": n_org_future,
            "n_peer_updates": n_updates,
            "n_org_entries": n_entries,
            "n_org_exits": n_exits,
            "confirmation_valid": valid,
            "confirmation_probe_valid": probe_valid,
            "sample_weight": weights,
            "label_available_date": target_flat,
            "label_version": LABEL_VERSION,
        }
    )
    invalid_reason = np.full(len(output), None, dtype=object)
    reason_mapping = {
        2: "right_censored",
        3: "crosses_actual_disclosure",
        4: "insufficient_future_peers",
        5: "insufficient_active_peers",
        6: "report_inline_with_consensus",
    }
    for code, text in reason_mapping.items():
        invalid_reason[reason_codes == code] = text
    pre_reasons = rows["pre_label_invalid_reason"].to_numpy(dtype=object)[row_index]
    invalid_reason[reason_codes == 1] = pre_reasons[reason_codes == 1]
    output["invalid_reason"] = invalid_reason
    output["probe_invalid_reason"] = np.where(
        probe_reason_codes == 1, "insufficient_peer_updates", None
    )
    return output


def warm_numba_kernels() -> None:
    """在正式计时前编译两个并行核；磁盘缓存命中时该步骤很快。"""
    group_offsets = np.asarray([0, 1], dtype=np.int64)
    org_ids = np.asarray([0], dtype=np.int32)
    event_offsets = np.asarray([0, 1], dtype=np.int64)
    event_dates = np.asarray([1], dtype=np.int64)
    event_values = np.asarray([1.0], dtype=np.float64)
    _pre_consensus_kernel(
        np.asarray([0], dtype=np.int32),
        np.asarray([1], dtype=np.int32),
        np.asarray([2], dtype=np.int64),
        group_offsets,
        org_ids,
        event_offsets,
        event_dates,
        event_dates,
        event_values,
        180,
    )
    nat_value = int(np.datetime64("NaT", "D").astype(np.int64))
    _confirmation_kernel(
        np.asarray([0], dtype=np.int32),
        np.asarray([1], dtype=np.int32),
        np.asarray([2], dtype=np.int64),
        np.asarray([[3]], dtype=np.int64),
        np.asarray([2.0], dtype=np.float64),
        np.asarray([1.0], dtype=np.float64),
        np.asarray([1], dtype=np.int8),
        np.asarray([nat_value], dtype=np.int64),
        group_offsets,
        org_ids,
        event_offsets,
        event_dates,
        event_dates,
        event_values,
        10,
        180,
        1,
        1,
        1,
        nat_value,
    )


def build_coverage_audit(
    fy_labels: pd.DataFrame,
    confirmation: pd.DataFrame,
) -> pd.DataFrame:
    """生成机器可读的覆盖率与失效原因长表。"""
    records: list[dict[str, object]] = []

    def add(
        common: dict[str, object],
        *,
        status: str,
        count: int,
        total: int,
        reason: object = None,
    ) -> None:
        records.append(
            {
                **common,
                "status": status,
                "reason": reason,
                "count": int(count),
                "total_count": int(total),
                "rate": float(count / total) if total else np.nan,
            }
        )

    base = fy_labels.assign(report_year=fy_labels["available_date"].dt.year)
    for (year, horizon), group in base.groupby(
        ["report_year", "forecast_horizon"], dropna=False
    ):
        common = {
            "report_year": year,
            "forecast_horizon": horizon,
            "confirmation_months": None,
            "peer_panel": None,
        }
        total = len(group)
        add({**common, "label": "report_fy"}, status="total", count=total, total=total)
        matched = int(group["actual_np"].notna().sum())
        add({**common, "label": "actual"}, status="matched", count=matched, total=total)
        if matched < total:
            add(
                {**common, "label": "actual"},
                status="missing",
                reason="actual_missing_or_source_unavailable",
                count=total - matched,
                total=total,
            )
        for label, valid_column in (
            ("residual", "residual_valid"),
            ("edge", "edge_valid"),
        ):
            add(
                {**common, "label": label},
                status="valid",
                count=int(group[valid_column].sum()),
                total=total,
            )
            for reason, count in (
                group.loc[group[valid_column].eq(0), "actual_invalid_reason"]
                .value_counts(dropna=True)
                .items()
            ):
                add(
                    {**common, "label": label},
                    status="invalid",
                    reason=reason,
                    count=int(count),
                    total=total,
                )
    if not confirmation.empty:
        conf = confirmation.assign(report_year=confirmation["available_date"].dt.year)
        keys = ["report_year", "forecast_horizon", "confirmation_months", "peer_panel"]
        for key, group in conf.groupby(keys, dropna=False):
            common = {
                "label": "confirmation",
                "report_year": key[0],
                "forecast_horizon": key[1],
                "confirmation_months": key[2],
                "peer_panel": key[3],
            }
            total = len(group)
            add(common, status="total", count=total, total=total)
            add(
                common,
                status="valid",
                count=int(group["confirmation_valid"].sum()),
                total=total,
            )
            add(
                common,
                status="probe_valid",
                count=int(group["confirmation_probe_valid"].sum()),
                total=total,
            )
            for reason, count in (
                group["invalid_reason"].value_counts(dropna=True).items()
            ):
                add(
                    common,
                    status="invalid",
                    reason=reason,
                    count=int(count),
                    total=total,
                )
    return pd.DataFrame(records)
