"""单篇研报 Label 经济意义验证的纯计算逻辑。"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


def canonical_stock_code(values: pd.Series | Iterable[object]) -> pd.Series:
    source = values if isinstance(values, pd.Series) else pd.Series(values)
    normalized = source.astype("string").str.strip().str.upper()
    return normalized.str.extract(
        r"^(\d{6})(?:\.(?:SH|SZ|BJ))?$", expand=False
    )


def _maybe_scale_to_decimal(data: pd.DataFrame, *, source_name: str, force_percentile_threshold: float | None = None) -> pd.DataFrame:
    """当收益数据疑似用百分数存储（例如 1.0 = 1%）时，自动除以 100 转回小数。

    启发式规则：如果绝对值的 99th 百分位 > *threshold*，认为整组数据是百分格式。
    threshold 默认取 1.0（即 pctile(abs) > 1 → "不可能是纯小数"）。
    """
    finite_mask = np.isfinite(data.to_numpy(dtype=float))
    if not np.any(finite_mask):
        return data
    p99_abs = float(np.nanpercentile(np.abs(data.to_numpy(dtype=float)[finite_mask]), 99))
    threshold = force_percentile_threshold if force_percentile_threshold is not None else 1.0
    if p99_abs > threshold:
        print("[info] {} percentiles: {:.4f} > {:.2f}, scaling /100".format(source_name, p99_abs, threshold), flush=True)
        return data / 100.0
    return data


def _normalize_return_frame(frame: pd.DataFrame, *, name: str, source_path: str | Path | None = None) -> pd.DataFrame:
    data = frame.copy()
    date_candidates = [c for c in data.columns if str(c).lower() in {"date", "trade_dt", "tradedate"}]
    if date_candidates:
        data.index = pd.to_datetime(data.pop(date_candidates[0]), errors="coerce")
    else:
        data.index = pd.to_datetime(data.index, errors="coerce")
    data = data.loc[data.index.notna()]
    data.index = pd.DatetimeIndex(data.index).normalize()
    if data.index.duplicated().any():
        raise ValueError(f"{name}含重复交易日")
    normalized_columns = canonical_stock_code(pd.Series(data.columns, dtype="string"))
    keep = normalized_columns.notna()
    data = data.loc[:, keep.to_numpy()]
    data.columns = normalized_columns.loc[keep].tolist()
    if data.columns.duplicated().any():
        raise ValueError(f"{name}股票代码规范化后重复")
    data = data.apply(pd.to_numeric, errors="coerce").sort_index()
    finite = data.to_numpy(dtype=float)
    if np.isinf(finite).any():
        raise ValueError(f"{name}含Inf")
    # Auto-detect and convert percentage-formatted returns (e.g. 1.0 = 1%) to decimals
    display_name = source_path.name if isinstance(source_path, Path) and source_path.is_file() else name
    data = _maybe_scale_to_decimal(data, source_name=display_name)
    return data


def read_return_panel(path: str | Path, *, hdf_key: str | None = None) -> pd.DataFrame:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"收益文件不存在: {source}")
    suffix = source.suffix.lower()
    if suffix in {".parquet", ".pq"}:
        frame = pd.read_parquet(source)
    elif suffix in {".h5", ".hdf", ".hdf5"}:
        with pd.HDFStore(source, mode="r") as store:
            keys = store.keys()
            requested = hdf_key if hdf_key and hdf_key.startswith("/") else (f"/{hdf_key}" if hdf_key else None)
            if requested is None:
                if "/df" in keys:
                    requested = "/df"
                elif len(keys) == 1:
                    requested = keys[0]
                else:
                    raise ValueError(f"HDF包含多个key，必须配置specific_return_hdf_key: {keys}")
            if requested not in keys:
                raise KeyError(f"HDF key {requested!r}不存在，可用key={keys}")
            frame = store[requested]
    else:
        raise ValueError(f"不支持的收益格式: {source}")
    return _normalize_return_frame(frame, name=source.name, source_path=source)


def validate_decimal_returns(panel: pd.DataFrame, *, name: str) -> None:
    values = panel.to_numpy(dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        raise ValueError(f"{name}没有有限收益")
    if np.nanpercentile(np.abs(values), 99) > 1.0:
        raise ValueError(f"{name}疑似百分数而非小数收益；禁止静默换算")
    if np.nanmin(values) < -1.0:
        raise ValueError(f"{name}存在小于-100%的收益")


def compound_forward(panel: pd.DataFrame, horizons: Sequence[int]) -> dict[int, pd.DataFrame]:
    results: dict[int, pd.DataFrame] = {}
    for horizon in sorted(set(int(x) for x in horizons)):
        if horizon <= 0:
            raise ValueError("收益窗口必须为正整数")
        product = pd.DataFrame(1.0, index=panel.index, columns=panel.columns)
        complete = pd.DataFrame(True, index=panel.index, columns=panel.columns)
        for offset in range(horizon):
            shifted = panel.shift(-offset)
            complete &= shifted.notna()
            product *= 1.0 + shifted.fillna(0.0)
        results[horizon] = (product - 1.0).where(complete)
    return results


def consistency_audit(
    derived: pd.DataFrame,
    supplied: pd.DataFrame,
    *,
    atol: float,
    rtol: float,
) -> dict[str, float | int]:
    common_dates = derived.index.intersection(supplied.index)
    common_stocks = derived.columns.intersection(supplied.columns)
    left = derived.loc[common_dates, common_stocks].to_numpy(dtype=float)
    right = supplied.loc[common_dates, common_stocks].to_numpy(dtype=float)
    valid = np.isfinite(left) & np.isfinite(right)
    if not valid.any():
        raise ValueError("收益一致性检查没有重合有限值")
    delta = np.abs(left[valid] - right[valid])
    close = np.isclose(left[valid], right[valid], atol=atol, rtol=rtol)
    return {
        "n_compared": int(valid.sum()),
        "fraction_close": float(close.mean()),
        "median_abs_error": float(np.median(delta)),
        "p99_abs_error": float(np.quantile(delta, 0.99)),
        "max_abs_error": float(delta.max()),
    }


def extract_forward_values(
    frame: pd.DataFrame,
    panel: pd.DataFrame,
    *,
    date_column: str,
    stock_column: str = "stock_code",
) -> np.ndarray:
    dates = pd.to_datetime(frame[date_column], errors="coerce").dt.normalize()
    stocks = canonical_stock_code(frame[stock_column])
    date_pos = panel.index.get_indexer(dates)
    stock_pos = panel.columns.get_indexer(stocks)
    output = np.full(len(frame), np.nan)
    valid = (date_pos >= 0) & (stock_pos >= 0)
    output[valid] = panel.to_numpy(dtype=float)[date_pos[valid], stock_pos[valid]]
    return output


def compound_between_events(
    frame: pd.DataFrame,
    daily: pd.DataFrame,
    *,
    start_column: str,
    end_column: str,
    stock_column: str = "stock_code",
) -> np.ndarray:
    dates = daily.index
    starts = pd.to_datetime(frame[start_column], errors="coerce").dt.normalize()
    ends = pd.to_datetime(frame[end_column], errors="coerce").dt.normalize()
    stocks = canonical_stock_code(frame[stock_column])
    start_pos = dates.get_indexer(starts)
    end_pos = dates.get_indexer(ends)
    stock_pos = daily.columns.get_indexer(stocks)
    values = daily.to_numpy(dtype=float)
    output = np.full(len(frame), np.nan)
    for index in np.flatnonzero((start_pos >= 0) & (end_pos >= start_pos) & (stock_pos >= 0)):
        segment = values[start_pos[index] : end_pos[index], stock_pos[index]]
        if len(segment) and np.isfinite(segment).all():
            output[index] = float(np.prod(1.0 + segment) - 1.0)
        elif len(segment) == 0:
            output[index] = 0.0
    return output


def normalize_csi_weights(frame: pd.DataFrame, *, index_code: str) -> pd.DataFrame:
    required = {"S_INFO_WINDCODE", "S_CON_WINDCODE", "TRADE_DT", "I_WEIGHT"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError("沪深300权重缺少字段: " + ", ".join(missing))
    out = frame.copy()
    out = out[out["S_INFO_WINDCODE"].astype(str).str.upper().eq(index_code.upper())]
    out["trade_date"] = pd.to_datetime(out["TRADE_DT"].astype(str).str.replace(r"\.0$", "", regex=True), format="%Y%m%d", errors="coerce")
    out["stock_code"] = canonical_stock_code(out["S_CON_WINDCODE"])
    out["weight"] = pd.to_numeric(out["I_WEIGHT"], errors="coerce")
    out = out.dropna(subset=["trade_date", "stock_code", "weight"])
    out = out[out["weight"] > 0]
    duplicate = out.duplicated(["trade_date", "stock_code"], keep=False)
    if duplicate.any():
        conflicts = out.loc[duplicate].groupby(["trade_date", "stock_code"])["weight"].nunique()
        if (conflicts > 1).any():
            raise ValueError("沪深300同日同股票存在冲突权重")
        out = out.drop_duplicates(["trade_date", "stock_code"])
    if out.empty:
        raise ValueError("沪深300历史权重为空")
    return out[["trade_date", "stock_code", "weight"]].sort_values(["trade_date", "stock_code"]).reset_index(drop=True)


def lock_annual_universes(weights: pd.DataFrame, years: Sequence[int]) -> tuple[pd.DataFrame, dict[int, set[str]]]:
    records: list[pd.DataFrame] = []
    universes: dict[int, set[str]] = {}
    for year in sorted(set(int(y) for y in years)):
        cutoff = pd.Timestamp(year=year - 1, month=12, day=31)
        eligible = weights.loc[weights["trade_date"] <= cutoff]
        if eligible.empty:
            raise ValueError(f"{year}股票池在{cutoff.date()}前没有历史权重")
        snapshot = eligible["trade_date"].max()
        selected = eligible.loc[eligible["trade_date"] == snapshot].copy()
        selected["validation_year"] = year
        selected["snapshot_date"] = snapshot
        selected["normalized_weight"] = selected["weight"] / selected["weight"].sum()
        universes[year] = set(selected["stock_code"])
        records.append(selected)
    return pd.concat(records, ignore_index=True), universes


def index_returns_from_prices(frame: pd.DataFrame, *, index_code: str) -> pd.Series:
    required = {"S_INFO_WINDCODE", "TRADE_DT", "S_DQ_CLOSE"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError("指数行情缺少字段: " + ", ".join(missing))
    data = frame[frame["S_INFO_WINDCODE"].astype(str).str.upper().eq(index_code.upper())].copy()
    data["date"] = pd.to_datetime(data["TRADE_DT"].astype(str).str.replace(r"\.0$", "", regex=True), format="%Y%m%d", errors="coerce")
    data["close"] = pd.to_numeric(data["S_DQ_CLOSE"], errors="coerce")
    data = data.dropna(subset=["date", "close"]).sort_values("date")
    if data["date"].duplicated().any():
        raise ValueError("指数行情含重复日期")
    result = data.set_index("date")["close"]
    return (result.shift(-1) / result - 1.0).rename("csi300_return_1d")


def weighted_index_returns(
    weights: pd.DataFrame,
    stock_returns: pd.DataFrame,
    *,
    min_weight_coverage: float,
) -> pd.DataFrame:
    snapshots = sorted(weights["trade_date"].unique())
    records: list[dict[str, object]] = []
    grouped = {pd.Timestamp(date): group.set_index("stock_code")["weight"] for date, group in weights.groupby("trade_date")}
    for date, row in stock_returns.iterrows():
        position = np.searchsorted(np.asarray(snapshots, dtype="datetime64[ns]"), np.datetime64(date), side="right") - 1
        if position < 0:
            continue
        snapshot = pd.Timestamp(snapshots[position])
        weight = grouped[snapshot]
        common = weight.index.intersection(row.index)
        values = pd.to_numeric(row.loc[common], errors="coerce")
        valid = values.notna()
        total_weight = float(weight.sum())
        available_weight = float(weight.loc[common[valid]].sum())
        coverage = available_weight / total_weight if total_weight > 0 else 0.0
        value = np.nan
        if coverage >= min_weight_coverage and available_weight > 0:
            value = float(np.dot(weight.loc[common[valid]], values.loc[valid]) / available_weight)
        records.append({"date": date, "csi300_return_1d": value, "weight_coverage": coverage, "weight_snapshot_date": snapshot})
    return pd.DataFrame(records).set_index("date") if records else pd.DataFrame(columns=["csi300_return_1d", "weight_coverage", "weight_snapshot_date"])


def _in_locked_universe(frame: pd.DataFrame, universes: Mapping[int, set[str]], *, year_column: str) -> pd.Series:
    stocks = canonical_stock_code(frame["stock_code"])
    years = pd.to_numeric(frame[year_column], errors="coerce")
    return pd.Series([stock in universes.get(int(year), set()) if pd.notna(year) and pd.notna(stock) else False for stock, year in zip(stocks, years)], index=frame.index)


def build_actual_surprise_samples(
    actuals: pd.DataFrame,
    report_history: pd.DataFrame,
    universes: Mapping[int, set[str]],
    *,
    lookback_days: int,
    min_peer_orgs: int,
    floor_ratio: float,
) -> pd.DataFrame:
    actual = actuals.copy()
    actual["stock_code"] = canonical_stock_code(actual["stock_code"])
    actual["fy"] = pd.to_numeric(actual["fy"], errors="coerce").astype("Int64")
    actual["actual_known_date"] = pd.to_datetime(actual["actual_known_date"], errors="coerce").dt.normalize()
    actual["actual_np"] = pd.to_numeric(actual["actual_np"], errors="coerce")
    actual = actual[_in_locked_universe(actual, universes, year_column="fy")]

    history = report_history.copy()
    history["stock_code"] = canonical_stock_code(history["stock_code"])
    history["fy"] = pd.to_numeric(history["fy"], errors="coerce").astype("Int64")
    history["available_date"] = pd.to_datetime(history["available_date"], errors="coerce").dt.normalize()
    history["publish_date"] = pd.to_datetime(history.get("publish_date", history["available_date"]), errors="coerce").dt.normalize()
    history["forecast_new"] = pd.to_numeric(history["forecast_new"], errors="coerce")
    if "forecast_conflict" in history:
        history = history[history["forecast_conflict"].fillna(0).eq(0)]
    history = history.dropna(subset=["stock_code", "fy", "org_id", "available_date", "forecast_new"])
    by_key = {(str(stock), int(fy)): group for (stock, fy), group in history.groupby(["stock_code", "fy"], sort=False)}

    records: list[dict[str, object]] = []
    for row in actual.itertuples(index=False):
        key = (str(row.stock_code), int(row.fy))
        group = by_key.get(key)
        record = row._asdict()
        record.update(
            {
                "consensus_actual_pre": np.nan,
                "mad_actual_pre_abs": np.nan,
                "n_org_actual_pre": 0,
                "actual_scale_floor": np.nan,
                "actual_scale": np.nan,
                "actual_surprise": np.nan,
                "actual_surprise_valid": 0,
                "actual_surprise_invalid_reason": None,
            }
        )
        if pd.isna(row.actual_known_date) or not np.isfinite(row.actual_np):
            record["actual_surprise_invalid_reason"] = "invalid_actual"
            records.append(record)
            continue
        if group is None:
            record["actual_surprise_invalid_reason"] = "no_forecast_history"
            records.append(record)
            continue
        age = (row.actual_known_date - group["available_date"]).dt.days
        pre = group[(age > 0) & (age <= lookback_days)]
        if pre.empty:
            record["actual_surprise_invalid_reason"] = "no_recent_forecast_before_actual"
            records.append(record)
            continue
        daily = pre.groupby(["org_id", "available_date"], as_index=False).agg(forecast_new=("forecast_new", "median"), publish_date=("publish_date", "max"))
        latest = daily.sort_values(["org_id", "available_date"]).groupby("org_id", as_index=False).tail(1)
        record["n_org_actual_pre"] = len(latest)
        if len(latest) < min_peer_orgs:
            record["actual_surprise_invalid_reason"] = "insufficient_actual_pre_peers"
            records.append(record)
            continue
        values = latest["forecast_new"].to_numpy(dtype=float)
        median = float(np.median(values))
        # 尺度下限只由该样本披露前已知的同行预测拟合，不使用
        # 未来年份或全样本统计量。
        floor = max(
            float(floor_ratio * np.median(np.abs(values))),
            float(np.finfo(float).eps),
        )
        scale = max(abs(median), floor)
        record["consensus_actual_pre"] = median
        record["mad_actual_pre_abs"] = float(np.median(np.abs(values - median)))
        record["actual_scale_floor"] = floor
        record["actual_scale"] = scale
        record["actual_surprise"] = (float(row.actual_np) - median) / scale
        record["actual_surprise_valid"] = 1
        records.append(record)
    output = pd.DataFrame(records)
    return output


def build_analyst_information_samples(fy_labels: pd.DataFrame, universes: Mapping[int, set[str]]) -> pd.DataFrame:
    out = fy_labels.copy()
    out["available_date"] = pd.to_datetime(out["available_date"], errors="coerce").dt.normalize()
    out["report_year"] = out["available_date"].dt.year
    out = out[_in_locked_universe(out, universes, year_column="report_year")]
    required = out["pre_label_valid"].eq(1) & out["actual_np"].notna() & (out["available_date"] < pd.to_datetime(out["actual_known_date"], errors="coerce"))
    out = out.loc[required].copy()
    out["report_signal"] = (out["forecast_new"] - out["consensus_pre"]) / out["scale_t"]
    out["realized_fundamental_surprise"] = (out["actual_np"] - out["consensus_pre"]) / out["scale_t"]
    nonzero = out["report_signal"].ne(0) & out["realized_fundamental_surprise"].ne(0)
    out["direction_hit"] = np.where(nonzero, np.sign(out["report_signal"]) == np.sign(out["realized_fundamental_surprise"]), np.nan)
    return out


def build_consensus_revision_samples(
    confirmation: pd.DataFrame,
    fy_labels: pd.DataFrame,
    universes: Mapping[int, set[str]],
) -> pd.DataFrame:
    out = confirmation.copy()
    out["available_date"] = pd.to_datetime(out["available_date"], errors="coerce").dt.normalize()
    out["target_date"] = pd.to_datetime(out["target_date"], errors="coerce").dt.normalize()
    out["report_year"] = out["available_date"].dt.year
    out = out[_in_locked_universe(out, universes, year_column="report_year")]
    out = out[out["confirmation_valid"].eq(1)].copy()
    out["consensus_revision"] = (out["consensus_future"] - out["consensus_pre"]) / out["scale_t"]
    actual = fy_labels[["report_id", "fy", "actual_np"]].drop_duplicates(["report_id", "fy"])
    out = out.merge(actual, on=["report_id", "fy"], how="left")
    out["realized_fundamental_surprise"] = (out["actual_np"] - out["consensus_pre"]) / out["scale_t"]
    out["report_direction"] = np.sign(out["forecast_new"] - out["consensus_pre"])
    return out


def clustered_ols(x: pd.Series, y: pd.Series, clusters: pd.Series) -> dict[str, float | int]:
    data = pd.DataFrame({"x": x, "y": y, "cluster": clusters}).replace([np.inf, -np.inf], np.nan).dropna()
    if len(data) < 3 or data["x"].nunique() < 2 or data["cluster"].nunique() < 2:
        return {"beta": np.nan, "se_cluster": np.nan, "t_cluster": np.nan, "n": len(data), "n_clusters": data["cluster"].nunique()}
    design = np.column_stack([np.ones(len(data)), data["x"].to_numpy(dtype=float)])
    outcome = data["y"].to_numpy(dtype=float)
    bread = np.linalg.inv(design.T @ design)
    coef = bread @ design.T @ outcome
    resid = outcome - design @ coef
    meat = np.zeros((2, 2))
    for _, positions in data.groupby("cluster").indices.items():
        score = design[positions].T @ resid[positions]
        meat += np.outer(score, score)
    n, k = design.shape
    groups = data["cluster"].nunique()
    correction = groups / (groups - 1) * (n - 1) / (n - k)
    covariance = correction * bread @ meat @ bread
    se = float(math.sqrt(max(covariance[1, 1], 0.0)))
    beta = float(coef[1])
    return {"beta": beta, "se_cluster": se, "t_cluster": beta / se if se > 0 else np.nan, "n": n, "n_clusters": groups}


def association_summary(
    frame: pd.DataFrame,
    *,
    analysis: str,
    x_column: str,
    y_column: str,
    cluster_column: str,
) -> dict[str, object]:
    from scipy.stats import spearmanr

    data = frame[[x_column, y_column, cluster_column]].replace([np.inf, -np.inf], np.nan).dropna()
    rho, pvalue = (np.nan, np.nan)
    if len(data) >= 3 and data[x_column].nunique() > 1 and data[y_column].nunique() > 1:
        rho, pvalue = spearmanr(data[x_column], data[y_column])
    ols = clustered_ols(data[x_column], data[y_column], data[cluster_column])
    sign_mask = data[x_column].ne(0) & data[y_column].ne(0)
    sign_hit = float((np.sign(data.loc[sign_mask, x_column]) == np.sign(data.loc[sign_mask, y_column])).mean()) if sign_mask.any() else np.nan
    return {"analysis": analysis, "x": x_column, "y": y_column, "n": len(data), "spearman": float(rho), "spearman_pvalue": float(pvalue), "sign_hit_rate": sign_hit, **ols}


def quantile_summary(
    frame: pd.DataFrame,
    *,
    analysis: str,
    x_column: str,
    y_column: str,
    quantiles: int,
) -> pd.DataFrame:
    data = frame[[x_column, y_column]].replace([np.inf, -np.inf], np.nan).dropna().copy()
    if len(data) < quantiles or data[x_column].nunique() < quantiles:
        return pd.DataFrame(columns=["analysis", "x", "y", "quantile", "n", "x_mean", "y_mean", "y_median"])
    data["quantile"] = pd.qcut(data[x_column].rank(method="first"), quantiles, labels=False) + 1
    result = data.groupby("quantile", as_index=False).agg(n=(y_column, "size"), x_mean=(x_column, "mean"), y_mean=(y_column, "mean"), y_median=(y_column, "median"))
    result.insert(0, "y", y_column)
    result.insert(0, "x", x_column)
    result.insert(0, "analysis", analysis)
    return result
