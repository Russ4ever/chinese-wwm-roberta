"""Stock-day return diagnostics for continuous-label OOS factors.

The tradable object in this module is the strictly OOS probe prediction, never
the realized future label.  Realized-label/return associations are emitted in a
separate, explicitly non-tradable diagnostic table.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .layer_probe_continuous import _atomic_write_once, _reuse_exact_output, _run_directory
from .layer_probe_factors import benjamini_hochberg, newey_west_mean_test
from .layer_probe_models import daily_rank_ic, summarize_rank_ic
from .layer_probe_panel import attach_forward_returns
from .layer_probe_representations import protocol_config_hash, sha256_file
from .report_label_economics import canonical_stock_code, read_return_panel


LABEL_FACTOR_SCHEMA = "continuous_label_stock_day_factor_v1.0"
LAYER_CORRELATION_SCHEMA = "layer_factor_correlation_v1.0"


def aggregate_label_predictions_to_stock_day(
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    """Unweighted report aggregation; future-derived target weights are not used."""

    required = {
        "sample_id",
        "report_id",
        "task_id",
        "label_name",
        "feature_available_date",
        "layer",
        "prediction",
        "label_value",
        "prediction_role",
    }
    missing = sorted(required.difference(predictions.columns))
    if missing:
        raise ValueError("连续Label OOS预测缺少字段: " + ", ".join(missing))
    if not predictions["prediction_role"].eq("oos").all():
        raise ValueError("股票日Label因子只能使用prediction_role=oos")
    symbol_column = "stock_code" if "stock_code" in predictions else "symbol"
    if symbol_column not in predictions:
        raise ValueError("连续Label OOS预测缺少股票代码")
    work = predictions.copy()
    work["symbol"] = canonical_stock_code(work[symbol_column])
    work["trading_date"] = pd.to_datetime(
        work["feature_available_date"], errors="coerce"
    ).dt.normalize()
    work["prediction"] = pd.to_numeric(work["prediction"], errors="coerce")
    work["label_value"] = pd.to_numeric(work["label_value"], errors="coerce")
    if work[["symbol", "trading_date"]].isna().any().any():
        raise ValueError("股票日Label因子聚合键含空值")
    if not np.isfinite(work[["prediction", "label_value"]].to_numpy(dtype=float)).all():
        raise ValueError("连续Label OOS预测或Label含NaN/Inf")
    key = ["task_id", "label_name", "layer", "trading_date", "symbol"]
    grouped = (
        work.groupby(key, sort=True, observed=True)
        .agg(
            factor_value=("prediction", "mean"),
            realized_label_mean=("label_value", "mean"),
            n_reports=("report_id", "nunique"),
            n_target_rows=("sample_id", "size"),
            label_available_date_max=("label_available_date", "max"),
        )
        .reset_index()
    )
    if grouped.duplicated(["task_id", "layer", "trading_date", "symbol"]).any():
        raise RuntimeError("股票日Label因子键不唯一")
    return grouped


def _quantile_return_tables(
    panel: pd.DataFrame,
    *,
    horizons: Sequence[int],
    quantiles: int,
    minimum_observations: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    daily_records: list[dict[str, object]] = []
    summary_records: list[dict[str, object]] = []
    for (task_id, layer), factor in panel.groupby(["task_id", "layer"], sort=True):
        for horizon in horizons:
            return_column = f"industry_adjusted_return_fut{int(horizon)}d"
            for date, group in factor.groupby("trading_date", sort=True):
                data = (
                    group[["factor_value", return_column]]
                    .replace([np.inf, -np.inf], np.nan)
                    .dropna()
                )
                if (
                    len(data) < max(minimum_observations, quantiles)
                    or data["factor_value"].nunique() < quantiles
                ):
                    continue
                data["quantile"] = (
                    pd.qcut(
                        data["factor_value"].rank(method="first"),
                        q=quantiles,
                        labels=False,
                    )
                    + 1
                )
                for quantile, values in data.groupby("quantile", sort=True):
                    daily_records.append(
                        {
                            "task_id": task_id,
                            "layer": int(layer),
                            "horizon": int(horizon),
                            "trading_date": date,
                            "quantile": int(quantile),
                            "n": len(values),
                            "mean_return": float(values[return_column].mean()),
                        }
                    )
    daily = pd.DataFrame(daily_records)
    if daily.empty:
        return daily, pd.DataFrame()
    for (task_id, layer, horizon), group in daily.groupby(
        ["task_id", "layer", "horizon"], sort=True
    ):
        means = group.groupby("quantile")["mean_return"].mean()
        monotonic = (
            float(spearmanr(means.index.to_numpy(), means.to_numpy()).correlation)
            if len(means) >= 2
            else np.nan
        )
        wide = group.pivot(
            index="trading_date", columns="quantile", values="mean_return"
        )
        spread = (
            wide[quantiles] - wide[1]
            if 1 in wide.columns and quantiles in wide.columns
            else pd.Series(dtype=float)
        )
        hac = newey_west_mean_test(spread, lags=max(0, int(horizon) - 1))
        summary_records.append(
            {
                "task_id": task_id,
                "layer": int(layer),
                "horizon": int(horizon),
                "quantiles": quantiles,
                "monotonic_spearman": monotonic,
                "top_minus_bottom_mean": (
                    float(spread.mean()) if len(spread) else np.nan
                ),
                "top_minus_bottom_hac_t": hac["hac_t_stat"],
                "top_minus_bottom_hac_pvalue": hac["hac_pvalue"],
                "n_days": int(spread.notna().sum()),
            }
        )
    summary = pd.DataFrame(summary_records)
    summary["qvalue_bh"] = benjamini_hochberg(
        summary["top_minus_bottom_hac_pvalue"]
    )
    return daily, summary


def _factor_ic_tables(
    panel: pd.DataFrame,
    *,
    horizons: Sequence[int],
    minimum_observations: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    daily_records: list[pd.DataFrame] = []
    summaries: list[dict[str, object]] = []
    for (task_id, label_name, layer), factor in panel.groupby(
        ["task_id", "label_name", "layer"], sort=True
    ):
        for horizon in horizons:
            target = f"industry_adjusted_return_fut{int(horizon)}d"
            daily = daily_rank_ic(
                factor,
                prediction_column="factor_value",
                target_column=target,
                min_observations=minimum_observations,
            )
            daily.insert(0, "horizon", int(horizon))
            daily.insert(0, "layer", int(layer))
            daily.insert(0, "label_name", label_name)
            daily.insert(0, "task_id", task_id)
            daily_records.append(daily)
            valid = daily["ic"].dropna()
            summary = summarize_rank_ic(valid)
            hac = newey_west_mean_test(valid, lags=max(0, int(horizon) - 1))
            summaries.append(
                {
                    "task_id": task_id,
                    "label_name": label_name,
                    "layer": int(layer),
                    "horizon": int(horizon),
                    **summary,
                    "icir_annualized": (
                        float(summary["icir"] * np.sqrt(252))
                        if np.isfinite(summary["icir"])
                        else np.nan
                    ),
                    **hac,
                }
            )
    summary_frame = pd.DataFrame(summaries)
    summary_frame["qvalue_bh"] = benjamini_hochberg(
        summary_frame["hac_pvalue"]
    )
    return pd.concat(daily_records, ignore_index=True), summary_frame


def _realized_label_diagnostic(
    panel: pd.DataFrame,
    *,
    horizons: Sequence[int],
    minimum_observations: int,
) -> pd.DataFrame:
    # Layer 0 is used only to deduplicate the identical realized label aggregation.
    realized = panel[panel["layer"].eq(0)].copy()
    records: list[dict[str, object]] = []
    for (task_id, label_name), group in realized.groupby(
        ["task_id", "label_name"], sort=True
    ):
        for horizon in horizons:
            target = f"industry_adjusted_return_fut{int(horizon)}d"
            daily = daily_rank_ic(
                group,
                prediction_column="realized_label_mean",
                target_column=target,
                min_observations=minimum_observations,
            )
            summary = summarize_rank_ic(daily["ic"])
            records.append(
                {
                    "task_id": task_id,
                    "label_name": label_name,
                    "horizon": int(horizon),
                    "diagnostic_only": True,
                    "tradable_factor": False,
                    "lookahead_warning": (
                        "realized label is known after feature time; never use as a "
                        "contemporaneous trading signal"
                    ),
                    **summary,
                }
            )
    return pd.DataFrame(records)


def run_label_factor_return_stage(config: Mapping[str, object]) -> Path:
    """Evaluate validation OOS label-prediction factors on future returns."""

    output_cfg = config.get("output", {})
    returns_cfg = config.get("returns", {})
    factor_cfg = config.get("label_factor_returns", {})
    if not all(
        isinstance(value, Mapping)
        for value in (output_cfg, returns_cfg, factor_cfg)
    ):
        raise ValueError("output/returns/label_factor_returns配置必须是对象")
    run = _run_directory(config)
    source = (
        run
        / "walk_forward_probe"
        / "validation"
        / "walk_forward_oos_predictions.parquet"
    )
    source_manifest = run / "walk_forward_probe" / "validation" / "manifest.json"
    if not source.is_file() or not source_manifest.is_file():
        raise FileNotFoundError("Label股票日因子缺少walk-forward validation预测")
    predictions = pd.read_parquet(source)
    stock_day = aggregate_label_predictions_to_stock_day(predictions)
    returns_path = Path(
        str(returns_cfg.get("industry_adjusted_daily_path", ""))
    ).expanduser().resolve()
    daily_returns = read_return_panel(
        returns_path,
        hdf_key=(str(returns_cfg.get("hdf_key")) if returns_cfg.get("hdf_key") else None),
    )
    horizons = sorted(
        {int(value) for value in factor_cfg.get("horizons", returns_cfg.get("horizons", [1, 5, 20]))}
    )
    primary_horizon = int(
        factor_cfg.get("primary_horizon", returns_cfg.get("primary_horizon", 5))
    )
    stock_day = attach_forward_returns(
        stock_day,
        daily_returns,
        horizons=horizons,
        primary_horizon=primary_horizon,
    )
    minimum_observations = int(factor_cfg.get("min_daily_observations", 20))
    quantiles = int(factor_cfg.get("quantiles", 5))
    daily_ic, summary = _factor_ic_tables(
        stock_day,
        horizons=horizons,
        minimum_observations=minimum_observations,
    )
    quantile_daily, quantile_summary = _quantile_return_tables(
        stock_day,
        horizons=horizons,
        quantiles=quantiles,
        minimum_observations=minimum_observations,
    )
    realized_diagnostic = _realized_label_diagnostic(
        stock_day,
        horizons=horizons,
        minimum_observations=minimum_observations,
    )
    output = run / "label_factor_returns" / "validation"
    expected = {
        "schema_version": LABEL_FACTOR_SCHEMA,
        "config_sha256": protocol_config_hash(config),
        "walk_forward_probe_manifest_sha256": sha256_file(source_manifest),
    }
    if _reuse_exact_output(
        output,
        expected_manifest=expected,
        validator=validate_label_factor_return_outputs,
    ):
        return output
    return _atomic_write_once(
        output,
        tables={
            "label_factor_stock_day.parquet": stock_day,
            "label_factor_daily_ic.parquet": daily_ic,
            "label_factor_summary.csv": summary,
            "label_factor_quantile_returns.parquet": quantile_daily,
            "label_factor_quantile_summary.csv": quantile_summary,
            "realized_label_return_diagnostic.csv": realized_diagnostic,
        },
        manifest={
            **expected,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "evaluation_split": "validation",
            "prediction_role": "oos",
            "aggregation": "unweighted mean report prediction by task/layer/symbol/date",
            "target_weight_used_for_factor_aggregation": False,
            "horizons": horizons,
            "primary_horizon": primary_horizon,
            "minimum_daily_observations": minimum_observations,
            "realized_label_diagnostic": "non-tradable look-ahead diagnostic only",
            "returns_source": str(returns_path),
        },
    )


def validate_label_factor_return_outputs(directory: str | Path) -> dict[str, object]:
    root = Path(directory).expanduser().resolve()
    required = [
        root / "label_factor_stock_day.parquet",
        root / "label_factor_daily_ic.parquet",
        root / "label_factor_summary.csv",
        root / "label_factor_quantile_returns.parquet",
        root / "label_factor_quantile_summary.csv",
        root / "realized_label_return_diagnostic.csv",
        root / "manifest.json",
    ]
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(f"Label股票日因子产物缺失: {path}")
    manifest = json.loads(required[-1].read_text(encoding="utf-8"))
    if manifest.get("schema_version") != LABEL_FACTOR_SCHEMA:
        raise ValueError("Label股票日因子协议版本不匹配")
    panel = pd.read_parquet(
        required[0], columns=["task_id", "layer", "trading_date", "symbol"]
    )
    if panel.duplicated(["task_id", "layer", "trading_date", "symbol"]).any():
        raise ValueError("Label股票日因子键重复")
    summary = pd.read_csv(required[2])
    layer_sets = summary.groupby(["task_id", "horizon"])["layer"].agg(set)
    if not layer_sets.map(lambda values: values == set(range(13))).all():
        raise ValueError("Label股票日收益部分任务没有完整13层")
    diagnostic = pd.read_csv(required[5])
    if not diagnostic["tradable_factor"].astype(str).str.lower().eq("false").all():
        raise ValueError("真实Label收益诊断被错误标记为可交易")
    return {
        "stock_day_rows": len(panel),
        "summary_rows": len(summary),
        "tasks": int(panel["task_id"].nunique()),
    }


def layer_correlation_tables(
    factors: pd.DataFrame,
    *,
    value_column: str = "factor_value",
    minimum_observations: int = 20,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Daily cross-sectional Spearman correlations for all layer pairs."""

    required = {"factor_source", "task_id", "trading_date", "symbol", "layer", value_column}
    missing = sorted(required.difference(factors.columns))
    if missing:
        raise ValueError("层相关性输入缺少字段: " + ", ".join(missing))
    records: list[dict[str, object]] = []
    for (source, task_id, date), group in factors.groupby(
        ["factor_source", "task_id", "trading_date"], sort=True
    ):
        if group.duplicated(["symbol", "layer"]).any():
            raise ValueError("同一日期股票层因子重复")
        wide = group.pivot(index="symbol", columns="layer", values=value_column)
        for left in range(13):
            for right in range(left, 13):
                if left not in wide or right not in wide:
                    value = np.nan
                    n_obs = 0
                elif left == right:
                    diagonal = wide[left].replace([np.inf, -np.inf], np.nan).dropna()
                    n_obs = len(diagonal)
                    value = (
                        1.0
                        if n_obs >= minimum_observations and diagonal.nunique() > 1
                        else np.nan
                    )
                else:
                    pair = wide[[left, right]].replace([np.inf, -np.inf], np.nan).dropna()
                    n_obs = len(pair)
                    if n_obs < minimum_observations:
                        value = np.nan
                    elif pair[left].nunique() < 2 or pair[right].nunique() < 2:
                        value = np.nan
                    else:
                        result = spearmanr(pair[left], pair[right]).correlation
                        value = float(result) if np.isfinite(result) else np.nan
                records.append(
                    {
                        "factor_source": source,
                        "task_id": task_id,
                        "trading_date": date,
                        "layer_left": left,
                        "layer_right": right,
                        "spearman": value,
                        "n_obs": n_obs,
                    }
                )
    daily = pd.DataFrame(records)
    summary = (
        daily.groupby(
            ["factor_source", "task_id", "layer_left", "layer_right"],
            sort=True,
        )
        .agg(
            mean_spearman=("spearman", "mean"),
            median_spearman=("spearman", "median"),
            std_spearman=("spearman", "std"),
            n_days=("spearman", "count"),
            mean_n_obs=("n_obs", "mean"),
        )
        .reset_index()
    )
    return daily, summary


def run_layer_factor_correlation_stage(config: Mapping[str, object]) -> Path:
    run = _run_directory(config)
    label_directory = run / "label_factor_returns" / "validation"
    label_path = label_directory / "label_factor_stock_day.parquet"
    label_manifest = label_directory / "manifest.json"
    return_directory = run / "return_probe" / "validation"
    return_path = return_directory / "return_oos_predictions.parquet"
    return_manifest = return_directory / "manifest.json"
    for path in (label_path, label_manifest, return_path, return_manifest):
        if not path.is_file():
            raise FileNotFoundError(f"层相关性缺少上游产物: {path}")
    label = pd.read_parquet(
        label_path,
        columns=["task_id", "trading_date", "symbol", "layer", "factor_value"],
    )
    label["factor_source"] = "continuous_label_probe"
    direct = pd.read_parquet(return_path)
    direct = direct[direct["prediction_role"].eq("oos")].copy()
    direct["factor_source"] = "direct_return_probe"
    direct["task_id"] = "direct_return_target"
    direct = direct.rename(columns={"prediction": "factor_value"})[
        ["factor_source", "task_id", "trading_date", "symbol", "layer", "factor_value"]
    ]
    combined = pd.concat(
        [
            label[
                ["factor_source", "task_id", "trading_date", "symbol", "layer", "factor_value"]
            ],
            direct,
        ],
        ignore_index=True,
    )
    cfg = config.get("layer_factor_correlations", {})
    if not isinstance(cfg, Mapping):
        raise ValueError("layer_factor_correlations配置必须是对象")
    minimum = int(cfg.get("min_daily_observations", 20))
    daily, summary = layer_correlation_tables(
        combined,
        minimum_observations=minimum,
    )
    output = run / "layer_factor_correlations" / "validation"
    expected = {
        "schema_version": LAYER_CORRELATION_SCHEMA,
        "config_sha256": protocol_config_hash(config),
        "label_factor_manifest_sha256": sha256_file(label_manifest),
        "return_probe_manifest_sha256": sha256_file(return_manifest),
    }
    if _reuse_exact_output(
        output,
        expected_manifest=expected,
        validator=validate_layer_factor_correlation_outputs,
    ):
        return output
    return _atomic_write_once(
        output,
        tables={
            "daily_layer_correlations.parquet": daily,
            "layer_correlation_summary.csv": summary,
        },
        manifest={
            **expected,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "evaluation_split": "validation",
            "method": "daily cross-sectional Spearman",
            "minimum_daily_observations": minimum,
            "sources": ["continuous_label_probe", "direct_return_probe"],
        },
    )


def validate_layer_factor_correlation_outputs(
    directory: str | Path,
) -> dict[str, object]:
    root = Path(directory).expanduser().resolve()
    daily_path = root / "daily_layer_correlations.parquet"
    summary_path = root / "layer_correlation_summary.csv"
    manifest_path = root / "manifest.json"
    for path in (daily_path, summary_path, manifest_path):
        if not path.is_file():
            raise FileNotFoundError(f"层相关性产物缺失: {path}")
    daily = pd.read_parquet(daily_path)
    summary = pd.read_csv(summary_path)
    if (daily["layer_left"] > daily["layer_right"]).any():
        raise ValueError("层相关性只应保存上三角")
    duplicates = daily.duplicated(
        ["factor_source", "task_id", "trading_date", "layer_left", "layer_right"]
    )
    if duplicates.any():
        raise ValueError("日度层相关性键重复")
    return {
        "daily_rows": len(daily),
        "summary_rows": len(summary),
        "tasks": int(summary[["factor_source", "task_id"]].drop_duplicates().shape[0]),
    }


def plot_layer_correlation_heatmap(
    directory: str | Path,
    *,
    factor_source: str,
    task_id: str,
):
    import matplotlib.pyplot as plt

    summary = pd.read_csv(Path(directory) / "layer_correlation_summary.csv")
    selected = summary[
        summary["factor_source"].eq(factor_source)
        & summary["task_id"].eq(task_id)
    ]
    if selected.empty:
        raise ValueError(f"没有层相关性: {factor_source}/{task_id}")
    matrix = np.full((13, 13), np.nan, dtype=float)
    for row in selected.itertuples():
        left = int(row.layer_left)
        right = int(row.layer_right)
        matrix[left, right] = matrix[right, left] = float(row.mean_spearman)
    fig, ax = plt.subplots(figsize=(7, 6))
    image = ax.imshow(matrix, vmin=-1, vmax=1, cmap="coolwarm")
    ax.set_xticks(range(13))
    ax.set_yticks(range(13))
    ax.set_xlabel("Layer")
    ax.set_ylabel("Layer")
    ax.set_title(f"{factor_source}: {task_id}")
    fig.colorbar(image, ax=ax, label="Mean daily Spearman")
    fig.tight_layout()
    return fig


def plot_label_factor_ic_curves(
    directory: str | Path, *, horizon: int = 5
):
    import matplotlib.pyplot as plt

    summary = pd.read_csv(Path(directory) / "label_factor_summary.csv")
    summary = summary[summary["horizon"].eq(int(horizon))]
    tasks = sorted(summary["task_id"].unique())
    fig, axes = plt.subplots(
        len(tasks),
        1,
        figsize=(9, max(4, 2.6 * len(tasks))),
        squeeze=False,
    )
    for axis, task in zip(axes[:, 0], tasks):
        selected = summary[summary["task_id"].eq(task)].sort_values("layer")
        axis.plot(selected["layer"], selected["rank_ic"], marker="o")
        axis.axhline(0, color="black", linewidth=0.8)
        axis.set(title=task, xlabel="Layer", ylabel=f"{horizon}d Rank IC")
        axis.grid(alpha=0.2)
    fig.tight_layout()
    return fig
