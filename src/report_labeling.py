"""单篇研报未来验证 Label 的纯计算核心。

本模块不访问网络或数据库。调用方负责提供原始研报、已有交易日历和可选的
结构化 Actual；这里仅完成规范化、时点一致的同行快照和三类报告级 Label。
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any, Iterable, Literal, Mapping, Sequence, cast

import numpy as np
import pandas as pd

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
        frame.groupby("report_id")["source_report_id"]
        .apply(lambda s: json.dumps(sorted(set(map(str, s))), ensure_ascii=False))
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

    grouped_rows: list[dict[str, object]] = []
    group_cols = ["report_id", "fy"]
    for (_, _), group in frame.groupby(group_cols, sort=False):
        values = group["forecast_new"].to_numpy(dtype=float)
        conflict = not np.allclose(values, values[0], rtol=1e-9, atol=1e-6)
        first = group.sort_values(["publish_timestamp", "source_report_id"]).iloc[0]
        grouped_rows.append(
            {
                "report_id": first["report_id"],
                "source_report_ids": json.dumps(
                    sorted(set(map(str, group["source_report_id"]))), ensure_ascii=False
                ),
                "stock_code": first["stock_code"],
                "org_id": first["org_id"],
                "author_name": first["author_name"],
                "title": first["title"],
                "publish_timestamp": first["publish_timestamp"],
                "publish_date": first["publish_date"],
                "available_date": first["available_date"],
                "fy": int(first["fy"]),
                "forecast_horizon": int(first["forecast_horizon"]),
                # 冲突行只保留证据，不为该报告猜测一个可用预测值。
                "forecast_new": np.nan if conflict else float(values[0]),
                "forecast_values": json.dumps(
                    sorted(set(map(float, values))), ensure_ascii=False
                ),
                "forecast_conflict": int(conflict),
                "n_source_rows": int(len(group)),
                "_line_numbers": json.dumps(
                    sorted(
                        set(
                            map(
                                int, group.get("_line_number", pd.Series([], dtype=int))
                            )
                        )
                    ),
                    ensure_ascii=False,
                ),
            }
        )
    rows = pd.DataFrame(grouped_rows)
    rows = rows.sort_values(
        ["stock_code", "fy", "available_date", "report_id"]
    ).reset_index(drop=True)
    return reports, rows


@dataclass(frozen=True)
class OrgForecastSeries:
    dates: np.ndarray
    publish_dates: np.ndarray
    values: np.ndarray


@dataclass(frozen=True)
class ForecastPoint:
    position: int
    available_date: pd.Timestamp
    publish_date: pd.Timestamp
    value: float


OrgHistories = dict[tuple[str, int], dict[str, OrgForecastSeries]]


def build_org_histories(rows: pd.DataFrame) -> OrgHistories:
    """按股票×FY构造机构日终预测历史。"""
    if rows.empty:
        return {}
    usable = rows[
        rows["forecast_conflict"].eq(0)
        & rows["forecast_new"].notna()
        & np.isfinite(rows["forecast_new"])
    ]
    if usable.empty:
        return {}
    daily = (
        usable.groupby(["stock_code", "fy", "org_id", "available_date"], as_index=False)
        .agg(forecast=("forecast_new", "median"), publish_date=("publish_date", "max"))
        .sort_values(["stock_code", "fy", "org_id", "available_date"])
    )
    result: OrgHistories = {}
    for key, group in daily.groupby(["stock_code", "fy"], sort=False):
        per_org: dict[str, OrgForecastSeries] = {}
        for org, org_group in group.groupby("org_id", sort=False):
            per_org[str(org)] = OrgForecastSeries(
                dates=org_group["available_date"].to_numpy(dtype="datetime64[ns]"),
                publish_dates=org_group["publish_date"].to_numpy(
                    dtype="datetime64[ns]"
                ),
                values=org_group["forecast"].to_numpy(dtype=float),
            )
        result[(str(key[0]), int(key[1]))] = per_org
    return result


def _point_at(
    series: OrgForecastSeries, date: pd.Timestamp, *, strict: bool
) -> ForecastPoint | None:
    side: Literal["left", "right"] = "left" if strict else "right"
    position = int(np.searchsorted(series.dates, np.datetime64(date), side=side)) - 1
    if position < 0:
        return None
    return ForecastPoint(
        position=position,
        available_date=pd.Timestamp(series.dates[position]),
        publish_date=pd.Timestamp(series.publish_dates[position]),
        value=float(series.values[position]),
    )


def _updated_point(
    series: OrgForecastSeries,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> ForecastPoint | None:
    point = _point_at(series, end, strict=False)
    if point is None or point.available_date <= start:
        return None
    return point


def peer_snapshot(
    histories: Mapping[str, OrgForecastSeries],
    *,
    origin_org: str,
    date: pd.Timestamp,
    lookback_days: int,
    strict: bool,
) -> dict[str, ForecastPoint]:
    """返回指定时点的同行最新有效预测，始终排除原机构。"""
    peers: dict[str, ForecastPoint] = {}
    for org, series in histories.items():
        if org == origin_org:
            continue
        point = _point_at(series, date, strict=strict)
        if point is None:
            continue
        if (date - point.publish_date).days <= lookback_days:
            peers[org] = point
    return peers


def _median_mad(values: Sequence[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=float)
    median = float(np.median(array))
    mad = float(np.median(np.abs(array - median)))
    return median, mad


def attach_pre_consensus(
    rows: pd.DataFrame,
    histories: OrgHistories,
    *,
    coverage_start: pd.Timestamp,
    lookback_days: int,
    min_peer_orgs: int,
) -> pd.DataFrame:
    """为每个报告×FY附加发布前、排除本机构的同行共识。"""
    out = rows.copy()
    consensus: list[float] = []
    mad: list[float] = []
    counts: list[int] = []
    complete: list[int] = []
    reasons: list[str | None] = []
    for row in out.itertuples(index=False):
        t = pd.Timestamp(row.available_date)
        history_ok = coverage_start <= t - pd.Timedelta(days=lookback_days)
        peers = peer_snapshot(
            histories.get((str(row.stock_code), int(row.fy)), {}),
            origin_org=str(row.org_id),
            date=t,
            lookback_days=lookback_days,
            strict=True,
        )
        values = [point.value for point in peers.values()]
        if values:
            c, d = _median_mad(values)
        else:
            c = d = np.nan
        consensus.append(c)
        mad.append(d)
        counts.append(len(values))
        complete.append(int(history_ok))
        if int(row.forecast_conflict) == 1:
            reasons.append("conflicting_report_forecast")
        elif not history_ok:
            reasons.append("left_censored")
        elif len(values) < min_peer_orgs:
            reasons.append("insufficient_pre_peers")
        else:
            reasons.append(None)
    out["consensus_pre"] = consensus
    out["mad_pre_abs"] = mad
    out["n_org_pre"] = counts
    out["history_window_complete"] = complete
    out["pre_invalid_reason"] = reasons
    return out


def assign_point_in_time_scale(
    rows: pd.DataFrame,
    *,
    history_months: int,
    min_samples: int,
    floor_ratio: float = 0.01,
) -> pd.DataFrame:
    """使用每月之前的历史共识拟合 point-in-time scale floor。"""
    if history_months <= 0 or min_samples <= 0 or floor_ratio <= 0:
        raise ValueError("scale 参数必须为正")
    out = rows.copy()
    months = out["available_date"].dt.to_period("M")
    out["scale_reference"] = None
    out["scale_sample_count"] = 0
    out["scale_floor"] = np.nan
    valid_ref = out["consensus_pre"].notna() & out["pre_invalid_reason"].isna()
    for month in sorted(months.dropna().unique()):
        month_start = month.start_time
        ref_start = (month - history_months).start_time
        base_ref = (
            valid_ref
            & (out["available_date"] >= ref_start)
            & (out["available_date"] < month_start)
        )
        target_month = months == month
        for horizon in sorted(out.loc[target_month, "forecast_horizon"].unique()):
            target = target_month & (out["forecast_horizon"] == horizon)
            same = base_ref & (out["forecast_horizon"] == horizon)
            if int(same.sum()) >= min_samples:
                reference = same
                reference_name = "same_horizon"
            elif int(base_ref.sum()) >= min_samples:
                reference = base_ref
                reference_name = "pooled_horizon"
            else:
                continue
            floor = floor_ratio * float(
                np.median(np.abs(out.loc[reference, "consensus_pre"]))
            )
            floor = max(floor, float(np.finfo(float).eps))
            out.loc[target, "scale_reference"] = reference_name
            out.loc[target, "scale_sample_count"] = int(reference.sum())
            out.loc[target, "scale_floor"] = floor
    out["scale_t"] = np.maximum(out["consensus_pre"].abs(), out["scale_floor"])
    out["dispersion_pre"] = MAD_SCALE * out["mad_pre_abs"] / out["scale_t"]
    out["pre_state"] = [
        profit_state(value, floor)
        for value, floor in zip(out["consensus_pre"], out["scale_floor"])
    ]
    out["pre_label_valid"] = (
        out["pre_invalid_reason"].isna() & out["scale_floor"].notna()
    ).astype(int)
    out["pre_label_invalid_reason"] = out["pre_invalid_reason"]
    missing_scale = out["pre_label_invalid_reason"].isna() & out["scale_floor"].isna()
    out.loc[missing_scale, "pre_label_invalid_reason"] = "insufficient_scale_history"
    return out


def profit_state(value: object, floor: object) -> str | None:
    if pd.isna(value) or pd.isna(floor):
        return None
    value_f = float(cast(Any, value))
    floor_f = float(cast(Any, floor))
    if abs(value_f) <= floor_f:
        return "near_zero"
    return "profit" if value_f > 0 else "loss"


def transition_state(before: object, actual: object, floor: object) -> str | None:
    before_state = profit_state(before, floor)
    actual_state = profit_state(actual, floor)
    if before_state is None or actual_state is None:
        return None
    return f"{before_state}_to_{actual_state}"


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
        frame["actual_version"] = "annual_report_first"
    frame["actual_version"] = _text(frame["actual_version"])
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
    residual: list[float] = []
    report_error: list[float] = []
    consensus_error: list[float] = []
    edge: list[float] = []
    signs: list[str | None] = []
    days: list[float] = []
    transitions: list[str | None] = []
    valid_values: list[int] = []
    reasons: list[str | None] = []
    for row in out.itertuples(index=False):
        reason: str | None = None
        if int(row.pre_label_valid) != 1:
            reason = str(row.pre_label_invalid_reason)
        elif pd.isna(row.actual_np):
            reason = "actual_missing"
        elif pd.Timestamp(row.available_date) >= pd.Timestamp(row.actual_known_date):
            reason = "report_not_before_actual_known_date"
        if reason is None:
            scale = float(row.scale_t)
            actual = float(row.actual_np)
            forecast = float(row.forecast_new)
            pre = float(row.consensus_pre)
            r = (actual - forecast) / scale
            re = abs(actual - forecast) / scale
            ce = abs(actual - pre) / scale
            e = ce - re
            residual.append(r)
            report_error.append(re)
            consensus_error.append(ce)
            edge.append(e)
            signs.append("positive" if e > 0 else ("negative" if e < 0 else "zero"))
            days.append(
                float(
                    (
                        pd.Timestamp(row.actual_publish_date)
                        - pd.Timestamp(row.available_date)
                    ).days
                )
            )
            transitions.append(transition_state(pre, actual, row.scale_floor))
            valid_values.append(1)
            reasons.append(None)
        else:
            residual.append(np.nan)
            report_error.append(np.nan)
            consensus_error.append(np.nan)
            edge.append(np.nan)
            signs.append(None)
            days.append(np.nan)
            transitions.append(None)
            valid_values.append(0)
            reasons.append(reason)
    out["residual_signed_raw"] = residual
    out["report_abs_error"] = report_error
    out["consensus_abs_error"] = consensus_error
    out["edge_raw"] = edge
    out["edge_sign"] = signs
    out["days_to_actual"] = days
    out["actual_transition_state"] = transitions
    out["actual_label_available_date"] = out["actual_publish_date"]
    out["residual_valid"] = valid_values
    out["edge_valid"] = valid_values
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


def _panel_values(
    panel: str,
    histories: Mapping[str, OrgForecastSeries],
    pre_peers: Mapping[str, ForecastPoint],
    *,
    origin_org: str,
    start: pd.Timestamp,
    target: pd.Timestamp,
    lookback_days: int,
) -> tuple[list[float], list[float], int, int, int]:
    updated: dict[str, ForecastPoint] = {}
    for org, old in pre_peers.items():
        point = _updated_point(histories[org], start, target)
        if point is not None:
            updated[org] = point

    if panel == "fixed":
        pre_values = [point.value for point in pre_peers.values()]
        future_values = [updated.get(org, old).value for org, old in pre_peers.items()]
        return pre_values, future_values, len(updated), 0, 0
    if panel == "active":
        active_orgs = sorted(updated)
        pre_values = [pre_peers[org].value for org in active_orgs]
        future_values = [updated[org].value for org in active_orgs]
        return pre_values, future_values, len(active_orgs), 0, 0
    if panel != "market":
        raise ValueError(f"未知同行面板: {panel}")

    future_peers: dict[str, ForecastPoint] = {}
    for org, series in histories.items():
        if org == origin_org:
            continue
        latest = _point_at(series, target, strict=False)
        if latest is None:
            continue
        # 同一 available_date 的日内顺序不可识别：没有后续更新时退回发布前旧值。
        if latest.available_date == start:
            latest = pre_peers.get(org)
            if latest is None:
                continue
        if (target - latest.publish_date).days <= lookback_days:
            future_peers[org] = latest
    pre_values = [point.value for point in pre_peers.values()]
    future_values = [point.value for point in future_peers.values()]
    entries = len(set(future_peers).difference(pre_peers))
    exits = len(set(pre_peers).difference(future_peers))
    return pre_values, future_values, len(updated), entries, exits


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
    """构造 fixed/market/active 三种报告级未来确认 Label。"""
    calendar = normalize_trading_dates(trading_dates)
    output: list[dict[str, object]] = []
    for row in rows.itertuples(index=False):
        key = (str(row.stock_code), int(row.fy))
        group_histories = histories.get(key, {})
        start = pd.Timestamp(row.available_date)
        pre_peers = peer_snapshot(
            group_histories,
            origin_org=str(row.org_id),
            date=start,
            lookback_days=lookback_days,
            strict=True,
        )
        for months in confirmation_months:
            nominal_target = start + pd.DateOffset(months=int(months))
            target = first_trading_date_on_or_after(nominal_target, calendar)
            for panel in ("fixed", "market", "active"):
                record: dict[str, object] = {
                    "report_id": row.report_id,
                    "stock_code": row.stock_code,
                    "fy": int(row.fy),
                    "forecast_horizon": int(row.forecast_horizon),
                    "available_date": start,
                    "confirmation_months": int(months),
                    "peer_panel": panel,
                    "target_date": target,
                    "forecast_new": float(row.forecast_new),
                    "scale_t": row.scale_t,
                    "consensus_pre": np.nan,
                    "consensus_future": np.nan,
                    "mad_pre_abs": np.nan,
                    "mad_future_abs": np.nan,
                    "dispersion_pre": np.nan,
                    "dispersion_future": np.nan,
                    "progress_raw": np.nan,
                    "progress_clipped": np.nan,
                    "delta_log_dispersion": np.nan,
                    "n_org_pre": 0,
                    "n_org_future": 0,
                    "n_peer_updates": 0,
                    "n_org_entries": 0,
                    "n_org_exits": 0,
                    "confirmation_valid": 0,
                    "confirmation_probe_valid": 0,
                    "invalid_reason": None,
                    "probe_invalid_reason": None,
                    "sample_weight": 0.0,
                    "label_available_date": target,
                    "label_version": LABEL_VERSION,
                }
                reason: str | None = None
                if int(row.pre_label_valid) != 1:
                    reason = str(row.pre_label_invalid_reason)
                elif pd.isna(target) or pd.Timestamp(target) > coverage_end:
                    reason = "right_censored"
                elif (
                    hasattr(row, "actual_known_date")
                    and pd.notna(row.actual_known_date)
                    and target >= row.actual_known_date
                ):
                    reason = "crosses_actual_disclosure"
                if reason is not None:
                    record["invalid_reason"] = reason
                    output.append(record)
                    continue

                pre_values, future_values, n_updates, entries, exits = _panel_values(
                    panel,
                    group_histories,
                    pre_peers,
                    origin_org=str(row.org_id),
                    start=start,
                    target=target,
                    lookback_days=lookback_days,
                )
                required = min_active_orgs if panel == "active" else min_peer_orgs
                record.update(
                    n_org_pre=len(pre_values),
                    n_org_future=len(future_values),
                    n_peer_updates=n_updates,
                    n_org_entries=entries,
                    n_org_exits=exits,
                )
                if len(pre_values) < required or len(future_values) < required:
                    record["invalid_reason"] = (
                        "insufficient_active_peers"
                        if panel == "active"
                        else "insufficient_future_peers"
                    )
                    output.append(record)
                    continue

                c_pre, mad_pre = _median_mad(pre_values)
                c_future, mad_future = _median_mad(future_values)
                scale = float(row.scale_t)
                d_pre = MAD_SCALE * mad_pre / scale
                d_future = MAD_SCALE * mad_future / scale
                gap = float(row.forecast_new) - c_pre
                inline_band = max(0.01, 0.5 * d_pre) * scale
                record.update(
                    consensus_pre=c_pre,
                    consensus_future=c_future,
                    mad_pre_abs=mad_pre,
                    mad_future_abs=mad_future,
                    dispersion_pre=d_pre,
                    dispersion_future=d_future,
                )
                if abs(gap) <= inline_band:
                    record["invalid_reason"] = "report_inline_with_consensus"
                    output.append(record)
                    continue

                progress = (c_future - c_pre) / gap
                delta = math.log((d_future + DISPERSION_EPS) / (d_pre + DISPERSION_EPS))
                coverage_weight = min(1.0, math.log1p(len(pre_values)) / math.log(11.0))
                update_weight = (
                    min(1.0, math.log1p(n_updates) / math.log(4.0))
                    if n_updates
                    else 0.0
                )
                probe_valid = panel == "active" or n_updates >= min_probe_updates
                record.update(
                    progress_raw=progress,
                    progress_clipped=float(np.clip(progress, -1.0, 2.0)),
                    delta_log_dispersion=delta,
                    confirmation_valid=1,
                    confirmation_probe_valid=int(probe_valid),
                    probe_invalid_reason=(
                        None if probe_valid else "insufficient_peer_updates"
                    ),
                    sample_weight=math.sqrt(coverage_weight * update_weight),
                )
                output.append(record)
    return pd.DataFrame(output)


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
