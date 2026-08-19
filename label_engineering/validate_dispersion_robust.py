# -*- coding: utf-8 -*-
"""验证 ``dispersion_robust`` 是否是可靠的一致预期分歧 Label。

验证分为两层：

1. 硬性质量检查：公式可复现、无未来数据、180 天窗口正确、正式样本数值有限；
2. 证据性检查：与 IQR/两两分歧的一致性、单机构剔除稳定性、极端值稳健性、
   数据质量变量依赖，以及（可选）对未来共识修正绝对幅度的区分能力。

默认读取 ``build_labels.py`` 在仓库 ``artifacts/label_engineering/v2`` 下生成的
三个 parquet 文件，并把表格、图形和 PASS/CONDITIONAL_PASS/FAIL 总结写入
``artifacts/label_engineering/v2/dispersion_validation``。

示例：

    python label_engineering/validate_dispersion_robust.py

指定远程产物：

    python label_engineering/validate_dispersion_robust.py \
      --snapshot /path/to/consensus_snapshot_monthly_v2.parquet \
      --org-daily /path/to/org_daily_forecast_v2.parquet \
      --revision /path/to/revision_label_monthly_v2.parquet \
      --output-dir /path/to/dispersion_validation

只运行不依赖真实数据的合成测试：

    python label_engineering/validate_dispersion_robust.py --self-test
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd


MAD_SCALE = 1.4826
VALID_PROFIT_STATES = ("profit", "loss")


def log(message: str) -> None:
    print(f"[dispersion-validation] {message}", flush=True)


def robust_dispersion(values: np.ndarray, scale_floor: float) -> float:
    """按 Label 构造口径计算 1.4826 * MAD / max(|median|, scale_floor)。"""
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0 or not np.isfinite(scale_floor) or scale_floor <= 0:
        return np.nan
    median = float(np.median(x))
    mad = float(np.median(np.abs(x - median)))
    return float(MAD_SCALE * mad / max(abs(median), float(scale_floor)))


def normalized_iqr(values: np.ndarray, scale_floor: float) -> float:
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.nan
    median = float(np.median(x))
    q25 = float(np.quantile(x, 0.25))
    q75 = float(np.quantile(x, 0.75))
    return float((q75 - q25) / max(abs(median), float(scale_floor)))


def normalized_std(values: np.ndarray, scale_floor: float) -> float:
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return np.nan
    mean = float(np.mean(x))
    return float(np.std(x, ddof=1) / max(abs(mean), float(scale_floor)))


def normalized_pairwise_median(values: np.ndarray, scale_floor: float) -> float:
    """所有机构两两预测差绝对值的中位数，经一致预期尺度归一化。"""
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return np.nan
    diff = np.abs(x[:, None] - x[None, :])
    pairwise = diff[np.triu_indices(len(x), k=1)]
    median = float(np.median(x))
    return float(np.median(pairwise) / max(abs(median), float(scale_floor)))


def safe_relative_change(new_value: float, old_value: float) -> float:
    if not np.isfinite(new_value) or not np.isfinite(old_value):
        return np.nan
    return float(abs(new_value - old_value) / max(abs(old_value), 1e-12))


def spearman_correlation(left: pd.Series, right: pd.Series) -> float:
    """不依赖 SciPy 的 Spearman：先取平均秩，再计算 Pearson 相关。"""
    pair = pd.DataFrame({"left": left, "right": right}).dropna()
    if len(pair) < 2:
        return np.nan
    ranked = pair.rank(method="average")
    if ranked["left"].nunique() < 2 or ranked["right"].nunique() < 2:
        return np.nan
    return float(np.corrcoef(ranked["left"], ranked["right"])[0, 1])


def spearman_matrix(frame: pd.DataFrame) -> pd.DataFrame:
    """返回不依赖 SciPy 的 Spearman 相关矩阵。"""
    ranked = frame.rank(method="average")
    return ranked.corr(method="pearson")


def normalize_org_name(value: object) -> str:
    """生成仅用于发现疑似机构别名的宽松名称键，不自动改写正式 org_id。"""
    text = unicodedata.normalize("NFKC", str(value)).lower().strip()
    text = re.sub(r"[\s·•・,，。._\-—（）()]+", "", text)
    suffixes = (
        "证券研究院", "证券研究所", "股份有限公司", "有限责任公司",
        "有限公司", "研究院", "研究所",
    )
    changed = True
    while changed and text:
        changed = False
        for suffix in suffixes:
            if text.endswith(suffix) and len(text) > len(suffix):
                text = text[:-len(suffix)]
                changed = True
                break
    return text


def organization_alias_candidates(org_daily: pd.DataFrame) -> pd.DataFrame:
    """列出经宽松规范化后落入同一键的多个原始机构名称。"""
    names = org_daily[["org_id"]].copy()
    names["org_id"] = names["org_id"].astype(str).str.strip()
    names["normalized_org_key"] = names["org_id"].map(normalize_org_name)
    counts = names.groupby(
        ["normalized_org_key", "org_id"], dropna=False
    ).size().rename("n_rows").reset_index()
    summary = counts.groupby("normalized_org_key").agg(
        n_distinct_names=("org_id", "nunique"),
        candidate_names=("org_id", lambda x: " | ".join(sorted(set(x)))),
        n_rows=("n_rows", "sum"),
    ).reset_index()
    return summary[summary["n_distinct_names"] > 1].sort_values(
        ["n_distinct_names", "n_rows"], ascending=False
    )


def read_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    suffix = path.suffix.lower()
    if suffix in (".parquet", ".pq"):
        frame = pd.read_parquet(path)
    elif suffix in (".csv", ".csv.gz") or path.name.endswith(".csv.gz"):
        frame = pd.read_csv(path, dtype={"stock_code": str})
    else:
        raise ValueError(f"不支持的文件类型: {path}")
    if "stock_code" in frame:
        frame["stock_code"] = frame["stock_code"].astype(str).str.replace(
            r"\.0$", "", regex=True
        ).str.zfill(6)
    return frame


def require_columns(frame: pd.DataFrame, required: set[str], name: str) -> None:
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{name} 缺少必要字段: {missing}")


def stratified_sample(
    frame: pd.DataFrame,
    n: int,
    strata: str,
    seed: int,
) -> pd.DataFrame:
    if n <= 0 or len(frame) <= n:
        return frame.copy()
    groups = list(frame.groupby(strata, dropna=False, sort=True))
    if not groups:
        return frame.sample(n=n, random_state=seed)
    per_group = max(1, n // len(groups))
    sampled = []
    used_indices: set[int] = set()
    for offset, (_, group) in enumerate(groups):
        take = min(len(group), per_group)
        part = group.sample(n=take, random_state=seed + offset)
        sampled.append(part)
        used_indices.update(part.index.tolist())
    result = pd.concat(sampled)
    remaining = n - len(result)
    if remaining > 0:
        pool = frame.loc[~frame.index.isin(used_indices)]
        if len(pool):
            result = pd.concat([
                result,
                pool.sample(n=min(remaining, len(pool)), random_state=seed + 999),
            ])
    return result.sample(frac=1, random_state=seed).head(n).copy()


def make_org_groups(org_daily: pd.DataFrame) -> dict[tuple[str, int], pd.DataFrame]:
    groups: dict[tuple[str, int], pd.DataFrame] = {}
    for (stock_code, fy), group in org_daily.groupby(["stock_code", "fy"], sort=False):
        groups[(str(stock_code), int(fy))] = group.sort_values("publish_date").copy()
    return groups


def forecasts_at_snapshot(
    org_groups: dict[tuple[str, int], pd.DataFrame],
    stock_code: str,
    fy: int,
    asof_month: pd.Timestamp,
    lookback_days: int,
) -> pd.DataFrame:
    """重建某个时点每家机构的一条最新有效预测。"""
    group = org_groups.get((str(stock_code), int(fy)))
    if group is None or len(group) == 0:
        return pd.DataFrame(columns=["org_id", "forecast", "publish_date", "age"])
    asof_month = pd.Timestamp(asof_month)
    history = group[group["publish_date"] <= asof_month].copy()
    if len(history) == 0:
        return pd.DataFrame(columns=["org_id", "forecast", "publish_date", "age"])
    history = history.sort_values("publish_date")
    latest = history.groupby("org_id", sort=False).tail(1).copy()
    latest["age"] = (asof_month - latest["publish_date"]).dt.days
    latest = latest[latest["age"].between(0, lookback_days, inclusive="both")]
    return latest.sort_values("org_id").reset_index(drop=True)


class CheckBook:
    def __init__(self) -> None:
        self.rows: list[dict[str, object]] = []

    def add(
        self,
        name: str,
        passed: bool | None,
        level: str,
        value: object = None,
        threshold: object = None,
        detail: str = "",
    ) -> None:
        if passed is None:
            status = "SKIP"
        elif passed:
            status = "PASS"
        else:
            status = "FAIL" if level == "hard" else "WARN"
        self.rows.append({
            "check": name,
            "status": status,
            "level": level,
            "value": value,
            "threshold": threshold,
            "detail": detail,
        })
        log(f"{status:4s} | {name}" + (f" | {detail}" if detail else ""))

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows)

    def overall_status(self) -> str:
        statuses = {str(row["status"]) for row in self.rows}
        if "FAIL" in statuses:
            return "FAIL"
        if "WARN" in statuses or "SKIP" in statuses:
            return "CONDITIONAL_PASS"
        return "PASS"


def prepare_inputs(
    snapshot_path: Path,
    org_daily_path: Path,
    revision_path: Path | None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame | None]:
    log(f"读取快照: {snapshot_path}")
    snapshot = read_table(snapshot_path)
    log(f"读取机构日终预测: {org_daily_path}")
    org_daily = read_table(org_daily_path)

    require_columns(snapshot, {
        "stock_code", "fy", "forecast_horizon", "asof_month", "consensus_median",
        "mad", "dispersion_robust", "scale_floor", "n_org", "snapshot_valid",
        "profit_state", "forecast_age_median", "forecast_age_p90", "stale_ratio",
    }, "consensus snapshot")
    require_columns(org_daily, {
        "stock_code", "fy", "org_id", "publish_date", "forecast",
    }, "org daily")

    snapshot["asof_month"] = pd.to_datetime(snapshot["asof_month"], errors="coerce")
    org_daily["publish_date"] = pd.to_datetime(org_daily["publish_date"], errors="coerce")
    snapshot["fy"] = pd.to_numeric(snapshot["fy"], errors="coerce").astype("Int64")
    org_daily["fy"] = pd.to_numeric(org_daily["fy"], errors="coerce").astype("Int64")
    org_daily["forecast"] = pd.to_numeric(org_daily["forecast"], errors="coerce")
    snapshot = snapshot.dropna(subset=["stock_code", "fy", "asof_month"])
    org_daily = org_daily.dropna(
        subset=["stock_code", "fy", "org_id", "publish_date", "forecast"]
    )
    snapshot["fy"] = snapshot["fy"].astype(int)
    org_daily["fy"] = org_daily["fy"].astype(int)

    revision = None
    if revision_path is not None and revision_path.exists():
        log(f"读取未来修正: {revision_path}")
        revision = read_table(revision_path)
        require_columns(revision, {
            "stock_code", "fy", "forecast_horizon", "asof_month",
            "target_horizon_months", "revision_consensus_symmetric",
            "revision_consensus_valid",
        }, "revision label")
        revision["asof_month"] = pd.to_datetime(revision["asof_month"], errors="coerce")
        revision["fy"] = pd.to_numeric(revision["fy"], errors="coerce").astype("Int64")
        revision = revision.dropna(subset=["stock_code", "fy", "asof_month"])
        revision["fy"] = revision["fy"].astype(int)
    elif revision_path is not None:
        log(f"未来修正文件不存在，将跳过经济意义检验: {revision_path}")

    return snapshot, org_daily, revision


def build_valid_sample(
    snapshot: pd.DataFrame,
    min_org: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    finite = np.isfinite(pd.to_numeric(snapshot["dispersion_robust"], errors="coerce"))
    valid_mask = (
        snapshot["snapshot_valid"].eq(1)
        & snapshot["n_org"].ge(min_org)
        & snapshot["profit_state"].isin(VALID_PROFIT_STATES)
        & finite
    )
    valid = snapshot.loc[valid_mask].copy()
    flow = pd.DataFrame([
        {"stage": "all_snapshots", "n_rows": len(snapshot)},
        {"stage": "snapshot_valid", "n_rows": int(snapshot["snapshot_valid"].eq(1).sum())},
        {"stage": f"n_org_at_least_{min_org}", "n_rows": int(snapshot["n_org"].ge(min_org).sum())},
        {
            "stage": "profit_or_loss",
            "n_rows": int(snapshot["profit_state"].isin(VALID_PROFIT_STATES).sum()),
        },
        {"stage": "finite_dispersion", "n_rows": int(finite.sum())},
        {"stage": "formal_validation_sample", "n_rows": len(valid)},
        {
            "stage": "near_zero_observation_group",
            "n_rows": int(snapshot["profit_state"].eq("near_zero").sum()),
        },
        {
            "stage": "mixed_sign_observation_group",
            "n_rows": int(snapshot["profit_state"].eq("mixed_sign").sum()),
        },
    ])
    return valid, flow


def recompute_sample(
    sample: pd.DataFrame,
    org_groups: dict[tuple[str, int], pd.DataFrame],
    lookback_days: int,
    tolerance: float,
) -> pd.DataFrame:
    rows = []
    for _, row in sample.iterrows():
        active = forecasts_at_snapshot(
            org_groups,
            row["stock_code"],
            int(row["fy"]),
            row["asof_month"],
            lookback_days,
        )
        values = active["forecast"].to_numpy(float)
        if len(values):
            median = float(np.median(values))
            mad = float(np.median(np.abs(values - median)))
            dispersion = robust_dispersion(values, float(row["scale_floor"]))
            max_date = active["publish_date"].max()
            max_age = float(active["age"].max())
        else:
            median = mad = dispersion = np.nan
            max_date = pd.NaT
            max_age = np.nan
        rows.append({
            "stock_code": row["stock_code"],
            "fy": int(row["fy"]),
            "forecast_horizon": row["forecast_horizon"],
            "asof_month": row["asof_month"],
            "n_org_saved": int(row["n_org"]),
            "n_org_recomputed": len(values),
            "consensus_saved": row["consensus_median"],
            "consensus_recomputed": median,
            "mad_saved": row["mad"],
            "mad_recomputed": mad,
            "dispersion_saved": row["dispersion_robust"],
            "dispersion_recomputed": dispersion,
            "n_org_match": int(len(values) == int(row["n_org"])),
            "consensus_match": int(np.isclose(
                median, row["consensus_median"], rtol=tolerance, atol=tolerance,
                equal_nan=False,
            )),
            "mad_match": int(np.isclose(
                mad, row["mad"], rtol=tolerance, atol=tolerance, equal_nan=False,
            )),
            "dispersion_match": int(np.isclose(
                dispersion, row["dispersion_robust"], rtol=tolerance,
                atol=tolerance, equal_nan=False,
            )),
            "no_future_data": int(pd.notna(max_date) and max_date <= row["asof_month"]),
            "lookback_respected": int(pd.notna(max_age) and max_age <= lookback_days),
            "max_publish_date": max_date,
            "max_age": max_age,
        })
    return pd.DataFrame(rows)


def metric_evidence_sample(
    sample: pd.DataFrame,
    org_groups: dict[tuple[str, int], pd.DataFrame],
    lookback_days: int,
) -> pd.DataFrame:
    rows = []
    for _, row in sample.iterrows():
        active = forecasts_at_snapshot(
            org_groups,
            row["stock_code"],
            int(row["fy"]),
            row["asof_month"],
            lookback_days,
        )
        values = active["forecast"].to_numpy(float)
        floor = float(row["scale_floor"])
        if len(values) < 3:
            continue

        d_robust = robust_dispersion(values, floor)
        d_iqr = normalized_iqr(values, floor)
        d_std = normalized_std(values, floor)
        d_pairwise = normalized_pairwise_median(values, floor)

        loo = np.array([
            robust_dispersion(np.delete(values, i), floor)
            for i in range(len(values))
        ], dtype=float)
        loo_abs = float(np.nanmax(np.abs(loo - d_robust)))
        loo_relative = loo_abs / max(abs(d_robust), 1e-12)

        median = float(np.median(values))
        outlier_values = values.copy()
        outlier_index = int(np.argmax(np.abs(values - median)))
        direction = float(np.sign(values[outlier_index] - median))
        if direction == 0:
            direction = 1.0
        shock = max(abs(median), floor)
        outlier_values[outlier_index] = median + direction * 5.0 * shock
        robust_outlier = robust_dispersion(outlier_values, floor)
        std_outlier = normalized_std(outlier_values, floor)

        scaled = robust_dispersion(values * 10.0, floor * 10.0)
        rows.append({
            "stock_code": row["stock_code"],
            "fy": int(row["fy"]),
            "forecast_horizon": row["forecast_horizon"],
            "asof_month": row["asof_month"],
            "n_org": len(values),
            "dispersion_robust": d_robust,
            "dispersion_iqr": d_iqr,
            "dispersion_std": d_std,
            "dispersion_pairwise": d_pairwise,
            "loo_max_abs_change": loo_abs,
            "loo_max_relative_change": loo_relative,
            "robust_outlier_relative_change": safe_relative_change(
                robust_outlier, d_robust
            ),
            "std_outlier_relative_change": safe_relative_change(std_outlier, d_std),
            "scale_invariance_abs_error": abs(scaled - d_robust),
        })
    return pd.DataFrame(rows)


def correlation_outputs(metric_sample: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    cols = [
        "dispersion_robust", "dispersion_iqr", "dispersion_std",
        "dispersion_pairwise",
    ]
    overall = spearman_matrix(metric_sample[cols])
    by_horizon = []
    for horizon, group in metric_sample.groupby("forecast_horizon"):
        corr = spearman_matrix(group[cols])
        for other in cols[1:]:
            by_horizon.append({
                "forecast_horizon": horizon,
                "comparison_metric": other,
                "spearman": corr.loc["dispersion_robust", other],
                "n": len(group),
            })
    return overall, pd.DataFrame(by_horizon)


def quality_outputs(valid: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    quality_columns = [
        "n_org", "forecast_age_median", "forecast_age_p90", "stale_ratio",
    ]
    rows = []
    for horizon, group in [("all", valid), *list(valid.groupby("forecast_horizon"))]:
        for column in quality_columns:
            pair = group[["dispersion_robust", column]].dropna()
            corr = spearman_correlation(pair["dispersion_robust"], pair[column])
            rows.append({
                "forecast_horizon": horizon,
                "quality_variable": column,
                "spearman": corr,
                "n": len(pair),
            })
    correlations = pd.DataFrame(rows)

    work = valid.copy()
    work["asof_month"] = pd.to_datetime(work["asof_month"])
    distributions = (
        work.groupby(["asof_month", "forecast_horizon"])["dispersion_robust"]
        .agg(
            n="size",
            mean="mean",
            median="median",
            std="std",
            q10=lambda x: x.quantile(0.10),
            q90=lambda x: x.quantile(0.90),
            q99=lambda x: x.quantile(0.99),
        )
        .reset_index()
    )
    return correlations, distributions


def manual_review_outputs(
    valid: pd.DataFrame,
    org_groups: dict[tuple[str, int], pd.DataFrame],
    lookback_days: int,
    per_bucket: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    work = valid.copy()
    work["dispersion_percentile"] = work.groupby(
        ["asof_month", "forecast_horizon"]
    )["dispersion_robust"].rank(method="average", pct=True)
    bucket_masks = {
        "low_0_10pct": work["dispersion_percentile"] <= 0.10,
        "middle_45_55pct": work["dispersion_percentile"].between(0.45, 0.55),
        "high_90_100pct": work["dispersion_percentile"] >= 0.90,
    }
    selected_parts = []
    for offset, (bucket, mask) in enumerate(bucket_masks.items()):
        candidates = work.loc[mask]
        if len(candidates) == 0:
            continue
        part = stratified_sample(
            candidates,
            min(per_bucket, len(candidates)),
            "forecast_horizon",
            seed + offset * 100,
        )
        part["review_bucket"] = bucket
        selected_parts.append(part)
    if not selected_parts:
        return pd.DataFrame(), pd.DataFrame()

    selected = pd.concat(selected_parts, ignore_index=True)
    keep = [
        "stock_code", "fy", "forecast_horizon", "asof_month", "review_bucket",
        "dispersion_percentile", "dispersion_robust", "consensus_median", "mad",
        "n_org", "forecast_age_median", "forecast_age_p90", "stale_ratio",
        "profit_state",
    ]
    if "stock_name" in selected.columns:
        keep.insert(1, "stock_name")
    summary = selected[keep].copy()

    forecasts = []
    for _, row in selected.iterrows():
        active = forecasts_at_snapshot(
            org_groups,
            row["stock_code"],
            int(row["fy"]),
            row["asof_month"],
            lookback_days,
        )
        for _, forecast_row in active.iterrows():
            forecasts.append({
                "stock_code": row["stock_code"],
                "fy": int(row["fy"]),
                "forecast_horizon": row["forecast_horizon"],
                "asof_month": row["asof_month"],
                "review_bucket": row["review_bucket"],
                "dispersion_robust": row["dispersion_robust"],
                "org_id": forecast_row["org_id"],
                "forecast": forecast_row["forecast"],
                "publish_date": forecast_row["publish_date"],
                "forecast_age": forecast_row["age"],
            })
    return summary, pd.DataFrame(forecasts)


def future_revision_outputs(
    valid: pd.DataFrame,
    revision: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    keys = ["stock_code", "fy", "forecast_horizon", "asof_month"]
    left = valid[keys + ["dispersion_robust"]].copy()
    future = revision[
        revision["revision_consensus_valid"].eq(1)
        & np.isfinite(pd.to_numeric(
            revision["revision_consensus_symmetric"], errors="coerce"
        ))
    ].copy()
    merged = future.merge(left, on=keys, how="inner", validate="many_to_one")
    if len(merged) == 0:
        return pd.DataFrame(), pd.DataFrame()
    merged["abs_future_revision"] = merged["revision_consensus_symmetric"].abs()
    group_keys = ["asof_month", "target_horizon_months", "forecast_horizon"]
    merged["dispersion_rank"] = merged.groupby(group_keys)["dispersion_robust"].rank(
        method="average", pct=True
    )
    merged["dispersion_quintile"] = np.ceil(
        merged["dispersion_rank"] * 5
    ).clip(1, 5).astype(int)
    quintiles = (
        merged.groupby([
            "target_horizon_months", "forecast_horizon", "dispersion_quintile"
        ])["abs_future_revision"]
        .agg(n="size", mean_abs_revision="mean", median_abs_revision="median")
        .reset_index()
    )

    summaries = []
    for (target_horizon, forecast_horizon), group in quintiles.groupby(
        ["target_horizon_months", "forecast_horizon"]
    ):
        ordered = group.sort_values("dispersion_quintile")
        rho = spearman_correlation(
            ordered["dispersion_quintile"], ordered["median_abs_revision"]
        )
        q1 = ordered.loc[
            ordered["dispersion_quintile"].eq(1), "median_abs_revision"
        ]
        q5 = ordered.loc[
            ordered["dispersion_quintile"].eq(5), "median_abs_revision"
        ]
        ratio = (
            float(q5.iloc[0] / q1.iloc[0])
            if len(q1) and len(q5) and q1.iloc[0] > 0
            else np.nan
        )
        summaries.append({
            "target_horizon_months": target_horizon,
            "forecast_horizon": forecast_horizon,
            "quintile_spearman": rho,
            "q5_q1_median_ratio": ratio,
            "n": int(ordered["n"].sum()),
        })
    return quintiles, pd.DataFrame(summaries)


def save_plots(
    valid: pd.DataFrame,
    metric_correlation: pd.DataFrame,
    metric_sample: pd.DataFrame,
    future_quintiles: pd.DataFrame,
    output_dir: Path,
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - 环境可选依赖
        log(f"matplotlib 不可用，跳过图形输出: {exc}")
        return

    horizons = sorted(valid["forecast_horizon"].dropna().unique().tolist())
    if horizons:
        fig, axes = plt.subplots(1, len(horizons), figsize=(5 * len(horizons), 4))
        axes = np.atleast_1d(axes)
        for axis, horizon in zip(axes, horizons):
            values = valid.loc[
                valid["forecast_horizon"].eq(horizon), "dispersion_robust"
            ].dropna()
            if len(values):
                axis.hist(values.clip(upper=values.quantile(0.99)), bins=50)
            axis.set_title(f"FY{horizon} dispersion (clip p99)")
            axis.set_xlabel("dispersion_robust")
        fig.tight_layout()
        fig.savefig(output_dir / "01_dispersion_by_horizon.png", dpi=160)
        plt.close(fig)

    if len(metric_correlation):
        fig, axis = plt.subplots(figsize=(7, 6))
        image = axis.imshow(metric_correlation.to_numpy(float), vmin=-1, vmax=1, cmap="coolwarm")
        axis.set_xticks(range(len(metric_correlation.columns)))
        axis.set_yticks(range(len(metric_correlation.index)))
        axis.set_xticklabels(metric_correlation.columns, rotation=40, ha="right")
        axis.set_yticklabels(metric_correlation.index)
        for i in range(len(metric_correlation.index)):
            for j in range(len(metric_correlation.columns)):
                axis.text(j, i, f"{metric_correlation.iloc[i, j]:.2f}", ha="center", va="center")
        fig.colorbar(image, ax=axis)
        axis.set_title("Spearman correlation of disagreement metrics")
        fig.tight_layout()
        fig.savefig(output_dir / "02_metric_correlation.png", dpi=160)
        plt.close(fig)

    if len(metric_sample):
        fig, axis = plt.subplots(figsize=(7, 4))
        values = metric_sample["loo_max_relative_change"].replace([np.inf, -np.inf], np.nan).dropna()
        if len(values):
            axis.hist(values.clip(upper=values.quantile(0.99)), bins=50)
        axis.set_title("Leave-one-organization-out sensitivity (clip p99)")
        axis.set_xlabel("maximum relative change")
        fig.tight_layout()
        fig.savefig(output_dir / "03_leave_one_out_sensitivity.png", dpi=160)
        plt.close(fig)

    if len(future_quintiles):
        fig, axis = plt.subplots(figsize=(8, 5))
        for (target, forecast_horizon), group in future_quintiles.groupby(
            ["target_horizon_months", "forecast_horizon"]
        ):
            ordered = group.sort_values("dispersion_quintile")
            axis.plot(
                ordered["dispersion_quintile"],
                ordered["median_abs_revision"],
                marker="o",
                label=f"{target}m/FY{forecast_horizon}",
            )
        axis.set_xticks([1, 2, 3, 4, 5])
        axis.set_xlabel("dispersion quintile")
        axis.set_ylabel("median |future symmetric revision|")
        axis.set_title("Current disagreement vs future revision magnitude")
        axis.legend()
        fig.tight_layout()
        fig.savefig(output_dir / "04_future_revision_quintiles.png", dpi=160)
        plt.close(fig)


def write_readme(
    output_dir: Path,
    status: str,
    checks: pd.DataFrame,
    paths: dict[str, str],
) -> None:
    counts = checks["status"].value_counts().to_dict()
    lines = [
        "# dispersion_robust 验证结果",
        "",
        f"总体结论：**{status}**",
        "",
        f"- PASS: {counts.get('PASS', 0)}",
        f"- WARN: {counts.get('WARN', 0)}",
        f"- FAIL: {counts.get('FAIL', 0)}",
        f"- SKIP: {counts.get('SKIP', 0)}",
        "",
        "结论规则：任何硬性检查失败为 `FAIL`；硬性检查通过但证据性检查警告或跳过为 "
        "`CONDITIONAL_PASS`；全部通过为 `PASS`。",
        "",
        "## 输入",
        "",
    ]
    lines.extend(f"- {key}: `{value}`" for key, value in paths.items())
    lines.extend([
        "",
        "## 主要文件",
        "",
        "- `validation_checks.csv`：逐项验收结论",
        "- `recomputation_check.csv`：随机快照公式重算",
        "- `alternative_metrics_sample.csv`：替代分歧指标与稳健性实验",
        "- `metric_spearman.csv`：指标相关性",
        "- `quality_correlations.csv`：分歧度与覆盖/新鲜度关系",
        "- `organization_alias_candidates.csv`：疑似机构别名，需人工确认",
        "- `manual_review_samples.csv`：人工复核样本",
        "- `manual_review_forecasts.csv`：人工复核样本的机构预测明细",
        "- `future_revision_quintiles.csv`：当前分歧五分组与未来修正",
        "",
        "注意：统计检查不能替代人工复核。请重点查看高/中/低分歧样本以及 WARN 项。",
        "",
    ])
    (output_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


def run_validation(args: argparse.Namespace) -> str:
    snapshot, org_daily, revision = prepare_inputs(
        args.snapshot, args.org_daily, args.revision
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    valid, sample_flow = build_valid_sample(snapshot, args.min_org)
    sample_flow.to_csv(args.output_dir / "sample_flow.csv", index=False)
    if len(valid) == 0:
        raise ValueError("筛选后没有正式验证样本，请检查 snapshot_valid/profit_state/数值字段")

    checks = CheckBook()
    checks.add("正式验证样本非空", len(valid) > 0, "hard", len(valid), "> 0")
    finite_all = np.isfinite(valid["dispersion_robust"]).all()
    checks.add("正式样本 dispersion 全部为有限值", bool(finite_all), "hard")
    nonnegative = valid["dispersion_robust"].ge(0).all()
    checks.add("正式样本 dispersion 全部非负", bool(nonnegative), "hard")
    enough_org = valid["n_org"].ge(args.min_org).all()
    checks.add(
        "正式样本机构数达到下限", bool(enough_org), "hard",
        int(valid["n_org"].min()), f">= {args.min_org}",
    )

    alias_candidates = organization_alias_candidates(org_daily)
    alias_candidates.to_csv(
        args.output_dir / "organization_alias_candidates.csv", index=False
    )
    checks.add(
        "未发现疑似机构名称别名",
        len(alias_candidates) == 0,
        "evidence",
        len(alias_candidates),
        "= 0",
        "使用宽松名称规则筛查；出现 WARN 时必须人工确认，脚本不会自动合并机构",
    )

    log("建立 stock/FY 机构预测索引 ...")
    org_groups = make_org_groups(org_daily)

    recompute_input = stratified_sample(
        valid, args.recompute_sample_size, "forecast_horizon", args.seed
    )
    recomputed = recompute_sample(
        recompute_input, org_groups, args.lookback_days, args.tolerance
    )
    recomputed.to_csv(args.output_dir / "recomputation_check.csv", index=False)
    for column, label in [
        ("n_org_match", "随机快照 n_org 可复现"),
        ("consensus_match", "随机快照 consensus_median 可复现"),
        ("mad_match", "随机快照 MAD 可复现"),
        ("dispersion_match", "随机快照 dispersion_robust 可复现"),
        ("no_future_data", "随机快照不使用未来预测"),
        ("lookback_respected", "随机快照遵守回看期限"),
    ]:
        passed = bool(len(recomputed) and recomputed[column].eq(1).all())
        checks.add(label, passed, "hard", int(recomputed[column].sum()), len(recomputed))

    metric_input = stratified_sample(
        valid, args.metric_sample_size, "forecast_horizon", args.seed + 10
    )
    metric_sample = metric_evidence_sample(
        metric_input, org_groups, args.lookback_days
    )
    metric_sample.to_csv(args.output_dir / "alternative_metrics_sample.csv", index=False)
    metric_correlation, metric_by_horizon = correlation_outputs(metric_sample)
    metric_correlation.to_csv(args.output_dir / "metric_spearman.csv")
    metric_by_horizon.to_csv(
        args.output_dir / "metric_spearman_by_horizon.csv", index=False
    )

    rho_iqr = float(metric_correlation.loc["dispersion_robust", "dispersion_iqr"])
    rho_pairwise = float(
        metric_correlation.loc["dispersion_robust", "dispersion_pairwise"]
    )
    checks.add(
        "与 normalized IQR 排名一致", rho_iqr >= args.min_alt_spearman,
        "evidence", round(rho_iqr, 4), f">= {args.min_alt_spearman}",
    )
    checks.add(
        "与两两机构分歧排名一致", rho_pairwise >= args.min_alt_spearman,
        "evidence", round(rho_pairwise, 4), f">= {args.min_alt_spearman}",
    )
    scale_error = float(metric_sample["scale_invariance_abs_error"].max())
    checks.add(
        "同比例缩放不改变 dispersion", scale_error <= args.tolerance,
        "hard", scale_error, f"<= {args.tolerance}",
    )

    robust_outlier = float(metric_sample["robust_outlier_relative_change"].median())
    std_outlier = float(metric_sample["std_outlier_relative_change"].median())
    checks.add(
        "极端值扰动下 MAD 指标比标准差稳定",
        robust_outlier < std_outlier,
        "evidence",
        round(robust_outlier, 4),
        f"< std median {std_outlier:.4f}",
    )
    loo_p95 = float(metric_sample["loo_max_relative_change"].replace(
        [np.inf, -np.inf], np.nan
    ).quantile(0.95))
    checks.add(
        "单机构剔除敏感度 P95 在参考范围内",
        loo_p95 <= args.max_loo_p95,
        "evidence",
        round(loo_p95, 4),
        f"<= {args.max_loo_p95}",
        "该阈值为诊断参考，低覆盖样本通常更敏感",
    )

    quality_corr, distributions = quality_outputs(valid)
    quality_corr.to_csv(args.output_dir / "quality_correlations.csv", index=False)
    distributions.to_csv(
        args.output_dir / "distribution_by_month_horizon.csv", index=False
    )
    all_quality = quality_corr[quality_corr["forecast_horizon"].astype(str).eq("all")]
    max_quality = float(all_quality["spearman"].abs().max())
    checks.add(
        "与单一数据质量变量不存在过强机械相关",
        max_quality <= args.max_quality_spearman,
        "evidence",
        round(max_quality, 4),
        f"<= {args.max_quality_spearman}",
        "相关不等于错误；WARN 时应结合分组图和业务原因复核",
    )

    manual_summary, manual_forecasts = manual_review_outputs(
        valid,
        org_groups,
        args.lookback_days,
        args.manual_sample_per_bucket,
        args.seed + 20,
    )
    manual_summary.to_csv(args.output_dir / "manual_review_samples.csv", index=False)
    manual_forecasts.to_csv(
        args.output_dir / "manual_review_forecasts.csv", index=False
    )
    checks.add(
        "已生成高/中/低分歧人工复核样本",
        len(manual_summary) > 0,
        "hard",
        len(manual_summary),
        "> 0",
    )

    future_quintiles = pd.DataFrame()
    future_summary = pd.DataFrame()
    if revision is not None:
        future_quintiles, future_summary = future_revision_outputs(valid, revision)
        future_quintiles.to_csv(
            args.output_dir / "future_revision_quintiles.csv", index=False
        )
        future_summary.to_csv(
            args.output_dir / "future_revision_summary.csv", index=False
        )
        if len(future_summary):
            median_rho = float(future_summary["quintile_spearman"].median())
            median_ratio = float(future_summary["q5_q1_median_ratio"].median())
            checks.add(
                "分歧五分组与未来绝对修正基本单调",
                median_rho >= args.min_future_spearman,
                "evidence",
                round(median_rho, 4),
                f">= {args.min_future_spearman}",
            )
            checks.add(
                "高分歧组未来修正大于低分歧组",
                median_ratio >= args.min_future_top_bottom_ratio,
                "evidence",
                round(median_ratio, 4),
                f">= {args.min_future_top_bottom_ratio}",
            )
        else:
            checks.add("未来修正经济意义检验", None, "evidence", detail="无可合并样本")
    else:
        checks.add("未来修正经济意义检验", None, "evidence", detail="未提供 revision 文件")

    save_plots(
        valid, metric_correlation, metric_sample, future_quintiles, args.output_dir
    )
    check_frame = checks.to_frame()
    check_frame.to_csv(args.output_dir / "validation_checks.csv", index=False)
    status = checks.overall_status()
    summary = {
        "overall_status": status,
        "n_snapshot_rows": int(len(snapshot)),
        "n_formal_validation_rows": int(len(valid)),
        "n_recomputed_rows": int(len(recomputed)),
        "n_metric_evidence_rows": int(len(metric_sample)),
        "n_manual_review_rows": int(len(manual_summary)),
        "check_status_counts": check_frame["status"].value_counts().to_dict(),
        "parameters": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
            if key != "self_test"
        },
    }
    (args.output_dir / "validation_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    paths = {
        "snapshot": str(args.snapshot),
        "org_daily": str(args.org_daily),
        "revision": str(args.revision) if args.revision is not None else "未提供",
    }
    write_readme(args.output_dir, status, check_frame, paths)
    log(f"总体结论: {status}")
    log(f"结果目录: {args.output_dir}")
    return status


def run_self_test() -> None:
    values = np.array([90.0, 95.0, 100.0, 105.0, 110.0])
    expected = MAD_SCALE * 5.0 / 100.0
    actual = robust_dispersion(values, 1.0)
    assert math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12)
    assert math.isclose(
        robust_dispersion(values * 10.0, 10.0), actual,
        rel_tol=1e-12, abs_tol=1e-12,
    )
    outlier = values.copy()
    outlier[-1] = 1000.0
    assert math.isclose(
        robust_dispersion(outlier, 1.0), actual,
        rel_tol=1e-12, abs_tol=1e-12,
    )
    assert normalized_iqr(values, 1.0) > 0
    assert normalized_std(values, 1.0) > 0
    assert normalized_pairwise_median(values, 1.0) > 0
    assert math.isclose(
        spearman_correlation(pd.Series([1, 2, 3]), pd.Series([10, 20, 30])),
        1.0,
    )

    aliases = organization_alias_candidates(pd.DataFrame({
        "org_id": ["中信证券", "中信证券股份有限公司", "其他证券"]
    }))
    assert len(aliases) == 1

    asof = pd.Timestamp("2024-06-30")
    group = pd.DataFrame({
        "org_id": ["A", "A", "B", "C"],
        "publish_date": pd.to_datetime([
            "2024-05-01", "2024-06-20", "2023-01-01", "2024-07-01"
        ]),
        "forecast": [90.0, 100.0, 80.0, 120.0],
    })
    groups = {("000001", 2024): group}
    active = forecasts_at_snapshot(groups, "000001", 2024, asof, 180)
    assert active["org_id"].tolist() == ["A"]
    assert active["forecast"].tolist() == [100.0]
    log("PASS | 合成数据自检通过")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    default_v2 = repo_root / "artifacts" / "label_engineering" / "v2"
    parser = argparse.ArgumentParser(
        description="验证 dispersion_robust 的构造正确性、稳健性和经济有效性",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--snapshot", type=Path,
        default=default_v2 / "consensus_snapshot_monthly_v2.parquet",
        help="月末一致预期快照 parquet/csv",
    )
    parser.add_argument(
        "--org-daily", type=Path,
        default=default_v2 / "org_daily_forecast_v2.parquet",
        help="机构日终预测 parquet/csv",
    )
    parser.add_argument(
        "--revision", type=Path,
        default=default_v2 / "revision_label_monthly_v2.parquet",
        help="未来修正 parquet/csv；不存在时自动跳过经济意义检验",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=default_v2 / "dispersion_validation",
        help="验证结果输出目录",
    )
    parser.add_argument("--lookback-days", type=int, default=180)
    parser.add_argument("--min-org", type=int, default=5)
    parser.add_argument("--recompute-sample-size", type=int, default=100)
    parser.add_argument("--metric-sample-size", type=int, default=2000)
    parser.add_argument("--manual-sample-per-bucket", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tolerance", type=float, default=1e-8)
    parser.add_argument("--min-alt-spearman", type=float, default=0.80)
    parser.add_argument("--max-loo-p95", type=float, default=0.50)
    parser.add_argument("--max-quality-spearman", type=float, default=0.50)
    parser.add_argument("--min-future-spearman", type=float, default=0.50)
    parser.add_argument("--min-future-top-bottom-ratio", type=float, default=1.10)
    parser.add_argument(
        "--self-test", action="store_true",
        help="只运行合成数据测试，不读取真实产物",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.self_test:
        run_self_test()
        return 0
    try:
        status = run_validation(args)
    except Exception as exc:
        log(f"FAIL | {type(exc).__name__}: {exc}")
        return 2
    return 1 if status == "FAIL" else 0


if __name__ == "__main__":
    sys.exit(main())
