"""跨层因子构造与严格的OOS截面检验。"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import uuid
from datetime import datetime
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.stats import norm, spearmanr
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

from .layer_probe_models import daily_rank_ic, summarize_rank_ic


LAYER_COLUMNS = [f"layer_{layer}" for layer in range(13)]


def _prediction_wide(
    predictions: pd.DataFrame,
    *,
    evaluation_split: str,
    prediction_role: str,
) -> pd.DataFrame:
    selected = predictions[
        predictions["evaluation_split"].eq(evaluation_split)
        & predictions["prediction_role"].eq(prediction_role)
    ].copy()
    if selected.empty:
        raise ValueError(
            f"没有evaluation_split={evaluation_split}, role={prediction_role}预测"
        )
    key = "representation_row"
    if selected.duplicated([key, "layer"]).any():
        raise ValueError("同一股票日同一层存在重复预测")
    layer_sets = selected.groupby(key)["layer"].agg(lambda values: set(values))
    expected = set(range(13))
    if not layer_sets.map(lambda values: values == expected).all():
        raise ValueError("每个股票日必须具有完整的13层预测")
    wide = selected.pivot(index=key, columns="layer", values="prediction")
    wide = wide.rename(columns={layer: f"layer_{layer}" for layer in range(13)})
    metadata_columns = [
        column for column in selected.columns if column not in {"layer", "prediction"}
    ]
    for column in metadata_columns:
        if column == key:
            continue
        conflicts = selected.groupby(key)[column].nunique(dropna=False)
        if conflicts.gt(1).any():
            raise ValueError(f"逐层预测的{column}在同一股票日不一致")
    metadata = selected[metadata_columns].drop_duplicates(key)
    if metadata[key].duplicated().any():
        raise ValueError("逐层预测携带的股票日metadata不一致")
    return metadata.merge(wide.reset_index(), on=key, validate="one_to_one")


def build_cross_layer_factors(
    reference_predictions: pd.DataFrame,
    evaluation_predictions: pd.DataFrame,
    *,
    pca_components: int = 3,
    middle_layers: Sequence[int] = (4, 5, 6, 7, 8),
    deep_layers: Sequence[int] = (9, 10, 11, 12),
) -> tuple[pd.DataFrame, dict[str, object]]:
    """用历史reference拟合deep residual/PCA，再变换OOS evaluation预测。"""

    for name, frame in (
        ("reference", reference_predictions),
        ("evaluation", evaluation_predictions),
    ):
        missing = sorted(set(LAYER_COLUMNS).difference(frame.columns))
        if missing:
            raise ValueError(f"{name}缺少逐层预测: {missing}")
    if not 1 <= pca_components <= 13:
        raise ValueError("pca_components必须在1~13之间")
    middle = [f"layer_{int(layer)}" for layer in middle_layers]
    deep = [f"layer_{int(layer)}" for layer in deep_layers]
    if not middle or not deep:
        raise ValueError("middle_layers和deep_layers不能为空")

    reference_values = reference_predictions[LAYER_COLUMNS].to_numpy(dtype=float)
    evaluation_values = evaluation_predictions[LAYER_COLUMNS].to_numpy(dtype=float)
    if (
        not np.isfinite(reference_values).all()
        or not np.isfinite(evaluation_values).all()
    ):
        raise ValueError("逐层预测含NaN或Inf")
    out = evaluation_predictions.copy()
    for column in LAYER_COLUMNS:
        out[f"single_{column}"] = out[column]
    out["layer_consensus"] = evaluation_values.mean(axis=1)
    out["layer_disagreement"] = evaluation_values.std(axis=1, ddof=0)
    reference_middle = reference_predictions[middle].mean(axis=1).to_numpy(dtype=float)
    evaluation_middle = (
        evaluation_predictions[middle].mean(axis=1).to_numpy(dtype=float)
    )
    out["deep_minus_middle"] = (
        evaluation_predictions[deep].mean(axis=1).to_numpy(dtype=float)
        - evaluation_middle
    )
    deep_residual_model = LinearRegression().fit(
        reference_middle.reshape(-1, 1),
        reference_predictions["layer_12"].to_numpy(dtype=float),
    )
    out["deep_residual"] = evaluation_predictions["layer_12"].to_numpy(
        dtype=float
    ) - deep_residual_model.predict(evaluation_middle.reshape(-1, 1))

    scaler = StandardScaler().fit(reference_values)
    pca = PCA(n_components=pca_components, svd_solver="full").fit(
        scaler.transform(reference_values)
    )
    components = pca.transform(scaler.transform(evaluation_values))
    oriented_loadings = pca.components_.copy()
    for component in range(pca_components):
        if oriented_loadings[component, 12] < 0:
            oriented_loadings[component] *= -1
            components[:, component] *= -1
        out[f"layer_pca_{component + 1}"] = components[:, component]
    model = {
        "reference_rows": len(reference_predictions),
        "middle_layers": [int(value) for value in middle_layers],
        "deep_layers": [int(value) for value in deep_layers],
        "deep_residual_intercept": float(deep_residual_model.intercept_),
        "deep_residual_coefficient": float(deep_residual_model.coef_[0]),
        "pca_explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        "pca_loadings": oriented_loadings.tolist(),
        "pca_scaler_mean": scaler.mean_.tolist(),
        "pca_scaler_scale": scaler.scale_.tolist(),
    }
    return out, model


def factor_columns(frame: pd.DataFrame) -> list[str]:
    return [
        column
        for column in frame.columns
        if column.startswith("single_layer_")
        or column
        in {
            "layer_consensus",
            "layer_disagreement",
            "deep_minus_middle",
            "deep_residual",
        }
        or column.startswith("layer_pca_")
    ]


def benjamini_hochberg(pvalues: pd.Series | np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg FDR校正，保留NaN位置。"""

    values = np.asarray(pvalues, dtype=float)
    output = np.full(len(values), np.nan)
    valid = np.isfinite(values)
    valid_values = values[valid]
    if not len(valid_values):
        return output
    order = np.argsort(valid_values)
    ranked = valid_values[order]
    adjusted = ranked * len(ranked) / np.arange(1, len(ranked) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0.0, 1.0)
    restored = np.empty_like(adjusted)
    restored[order] = adjusted
    output[valid] = restored
    return output


def newey_west_mean_test(
    values: pd.Series | np.ndarray, *, lags: int
) -> dict[str, float | int]:
    """对均值为零假设计算Bartlett权重Newey-West t/p值。"""

    data = np.asarray(values, dtype=float)
    data = data[np.isfinite(data)]
    n = len(data)
    if n < 2:
        return {"hac_t_stat": np.nan, "hac_pvalue": np.nan, "hac_lags": int(lags)}
    lags = max(0, min(int(lags), n - 1))
    centered = data - data.mean()
    long_run_variance = float(np.dot(centered, centered) / n)
    for lag in range(1, lags + 1):
        covariance = float(np.dot(centered[lag:], centered[:-lag]) / n)
        weight = 1.0 - lag / (lags + 1.0)
        long_run_variance += 2.0 * weight * covariance
    variance_of_mean = max(long_run_variance / n, 0.0)
    if variance_of_mean <= 0:
        return {"hac_t_stat": np.nan, "hac_pvalue": np.nan, "hac_lags": lags}
    statistic = float(data.mean() / np.sqrt(variance_of_mean))
    return {
        "hac_t_stat": statistic,
        "hac_pvalue": float(2.0 * norm.sf(abs(statistic))),
        "hac_lags": lags,
    }


def evaluate_factor_ic(
    matrix: pd.DataFrame,
    *,
    factors: Sequence[str],
    target_column: str,
    min_daily_observations: int,
    hac_lags: int = 4,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """计算所有候选因子的日度Rank IC、ICIR和p/q值。"""

    daily_records: list[pd.DataFrame] = []
    summaries: list[dict[str, object]] = []
    for factor in factors:
        daily = daily_rank_ic(
            matrix,
            prediction_column=factor,
            target_column=target_column,
            min_observations=min_daily_observations,
        )
        daily.insert(0, "factor", factor)
        daily_records.append(daily)
        valid = daily["ic"].dropna().to_numpy(dtype=float)
        hac = newey_west_mean_test(valid, lags=hac_lags)
        summary = summarize_rank_ic(valid)
        summary["icir_annualized"] = (
            float(summary["icir"] * np.sqrt(252))
            if np.isfinite(summary["icir"])
            else np.nan
        )
        summaries.append({"factor": factor, **hac, **summary})
    summary_frame = pd.DataFrame(summaries)
    summary_frame["qvalue_bh"] = benjamini_hochberg(summary_frame["hac_pvalue"])
    return summary_frame, pd.concat(daily_records, ignore_index=True)


def evaluate_quantile_monotonicity(
    matrix: pd.DataFrame,
    *,
    factors: Sequence[str],
    return_column: str,
    quantiles: int,
    hac_lags: int = 4,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """逐日分组后汇总各组收益、顶底收益差与单调性。"""

    group_records: list[dict[str, object]] = []
    for factor in factors:
        for date, group in matrix.groupby("trading_date", sort=True):
            data = (
                group[[factor, return_column]]
                .replace([np.inf, -np.inf], np.nan)
                .dropna()
            )
            if len(data) < quantiles or data[factor].nunique() < quantiles:
                continue
            data["quantile"] = (
                pd.qcut(
                    data[factor].rank(method="first"),
                    q=quantiles,
                    labels=False,
                )
                + 1
            )
            for quantile, values in data.groupby("quantile"):
                group_records.append(
                    {
                        "factor": factor,
                        "trading_date": date,
                        "quantile": int(quantile),
                        "n": len(values),
                        "mean_return": float(values[return_column].mean()),
                    }
                )
    groups = pd.DataFrame(group_records)
    summaries: list[dict[str, object]] = []
    if groups.empty:
        return groups, pd.DataFrame()
    for factor, factor_groups in groups.groupby("factor"):
        average = factor_groups.groupby("quantile")["mean_return"].mean()
        monotonic = (
            spearmanr(average.index.to_numpy(), average.to_numpy()).correlation
            if len(average) >= 2
            else np.nan
        )
        daily_wide = factor_groups.pivot(
            index="trading_date", columns="quantile", values="mean_return"
        )
        spread = (
            daily_wide[quantiles] - daily_wide[1]
            if 1 in daily_wide and quantiles in daily_wide
            else pd.Series(dtype=float)
        )
        hac = newey_west_mean_test(spread, lags=hac_lags)
        summaries.append(
            {
                "factor": factor,
                "quantiles": quantiles,
                "monotonic_spearman": float(monotonic),
                "top_minus_bottom_mean": (
                    float(spread.mean()) if len(spread) else np.nan
                ),
                "top_minus_bottom_hac_t": hac["hac_t_stat"],
                "top_minus_bottom_hac_pvalue": hac["hac_pvalue"],
                "hac_lags": hac["hac_lags"],
                "n_days": int(spread.notna().sum()),
            }
        )
    return groups, pd.DataFrame(summaries)


def _n_texts_group(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    return pd.Series(
        np.select(
            [numeric.eq(1), numeric.eq(2), numeric.ge(3)],
            ["1", "2", "3+"],
            default="missing",
        ),
        index=values.index,
        dtype="string",
    )


def _size_group(frame: pd.DataFrame) -> pd.Series:
    output = pd.Series(pd.NA, index=frame.index, dtype="string")
    for _, group in frame.groupby("trading_date"):
        valid = pd.to_numeric(group["size"], errors="coerce").dropna()
        if len(valid) < 3 or valid.nunique() < 3:
            continue
        output.loc[valid.index] = pd.qcut(
            valid.rank(method="first"),
            q=3,
            labels=["small", "middle", "large"],
        ).astype("string")
    return output


def evaluate_stratified_ic(
    matrix: pd.DataFrame,
    *,
    factors: Sequence[str],
    target_column: str,
    min_daily_observations: int,
) -> pd.DataFrame:
    """按n_texts、行业、规模和年份分别计算Rank IC。"""

    work = matrix.copy()
    work["n_texts_group"] = _n_texts_group(work["n_texts"])
    work["year_group"] = pd.to_datetime(work["trading_date"]).dt.year.astype(str)
    group_columns = ["n_texts_group", "year_group"]
    if "industry" in work.columns and work["industry"].notna().any():
        work["industry_group"] = work["industry"].astype("string")
        group_columns.append("industry_group")
    if "size" in work.columns and work["size"].notna().any():
        work["size_group"] = _size_group(work)
        group_columns.append("size_group")
    records: list[dict[str, object]] = []
    for dimension in group_columns:
        for group_value, selected in work.groupby(dimension, dropna=False):
            for factor in factors:
                daily = daily_rank_ic(
                    selected,
                    prediction_column=factor,
                    target_column=target_column,
                    min_observations=min_daily_observations,
                )
                records.append(
                    {
                        "dimension": dimension,
                        "group": str(group_value),
                        "factor": factor,
                        **summarize_rank_ic(daily["ic"]),
                    }
                )
    return pd.DataFrame(records)


def _partial_rank_correlation(
    group: pd.DataFrame,
    *,
    factor: str,
    target: str,
    controls: Sequence[str],
    min_observations: int,
) -> tuple[float, int]:
    columns = [factor, target, *controls]
    data = group[columns].replace([np.inf, -np.inf], np.nan).dropna()
    if len(data) < min_observations:
        return np.nan, len(data)
    ranked = data.rank(method="average", pct=True)
    design = np.column_stack(
        [np.ones(len(ranked)), ranked[list(controls)].to_numpy(dtype=float)]
    )
    x = ranked[factor].to_numpy(dtype=float)
    y = ranked[target].to_numpy(dtype=float)
    x_residual = x - design @ np.linalg.lstsq(design, x, rcond=None)[0]
    y_residual = y - design @ np.linalg.lstsq(design, y, rcond=None)[0]
    if np.std(x_residual) == 0 or np.std(y_residual) == 0:
        return np.nan, len(data)
    return float(np.corrcoef(x_residual, y_residual)[0, 1]), len(data)


def evaluate_incremental_ic(
    matrix: pd.DataFrame,
    *,
    factors: Sequence[str],
    target_column: str,
    controls: Sequence[str] = ("final_sentiment_logit", "log_n_texts"),
    min_daily_observations: int = 20,
    hac_lags: int = 4,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """控制最终层情绪和文本数后的日度partial Rank IC。"""

    work = matrix.copy()
    work["log_n_texts"] = np.log1p(pd.to_numeric(work["n_texts"], errors="coerce"))
    missing = sorted(set(controls).difference(work.columns))
    if missing:
        raise ValueError("增量IC缺少控制变量: " + ", ".join(missing))
    daily_records: list[dict[str, object]] = []
    summaries: list[dict[str, object]] = []
    for factor in factors:
        for date, group in work.groupby("trading_date", sort=True):
            ic, n_obs = _partial_rank_correlation(
                group,
                factor=factor,
                target=target_column,
                controls=controls,
                min_observations=min_daily_observations,
            )
            daily_records.append(
                {"factor": factor, "trading_date": date, "ic": ic, "n_obs": n_obs}
            )
        factor_daily = pd.DataFrame(daily_records)
        valid = factor_daily.loc[factor_daily["factor"].eq(factor), "ic"].dropna()
        hac = newey_west_mean_test(valid, lags=hac_lags)
        summaries.append({"factor": factor, **hac, **summarize_rank_ic(valid)})
    summary = pd.DataFrame(summaries)
    summary["qvalue_bh"] = benjamini_hochberg(summary["hac_pvalue"])
    return summary, pd.DataFrame(daily_records)


def _fingerprint_frame(frame: pd.DataFrame, columns: Sequence[str]) -> str:
    hashed = pd.util.hash_pandas_object(frame[list(columns)], index=False).to_numpy()
    return hashlib.sha256(hashed.tobytes()).hexdigest()


def _atomic_write_factor_outputs(
    output: Path,
    *,
    tables: Mapping[str, pd.DataFrame],
    manifest: Mapping[str, object],
    final_test_marker: Mapping[str, object] | None,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent))
    backup = output.parent / f".{output.name}.backup-{uuid.uuid4().hex}"
    moved = False
    try:
        for filename, frame in tables.items():
            path = temporary / filename
            if path.suffix == ".csv":
                frame.to_csv(path, index=False)
            else:
                frame.to_parquet(path, index=False, compression="zstd")
        (temporary / "manifest.json").write_text(
            json.dumps(dict(manifest), ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        if final_test_marker is not None:
            (temporary / "FINAL_TEST_OPENED.json").write_text(
                json.dumps(
                    dict(final_test_marker), ensure_ascii=False, indent=2, default=str
                ),
                encoding="utf-8",
            )
        if output.exists():
            os.replace(output, backup)
            moved = True
        try:
            os.replace(temporary, output)
        except BaseException:
            if moved and backup.exists() and not output.exists():
                os.replace(backup, output)
                moved = False
            raise
        if moved:
            shutil.rmtree(backup)
            moved = False
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
        if moved and backup.exists() and not output.exists():
            os.replace(backup, output)


def run_factor_validation_stage(config: Mapping[str, object]) -> Path:
    """执行阶段5/6；默认只看validation，test必须显式解锁且只能写一次。"""

    output_cfg = config.get("output", {})
    factor_cfg = config.get("cross_layer_factors", {})
    strict_cfg = config.get("strict_test", {})
    returns_cfg = config.get("returns", {})
    probe_cfg = config.get("return_probe", {})
    if not all(
        isinstance(value, Mapping)
        for value in (output_cfg, factor_cfg, strict_cfg, returns_cfg, probe_cfg)
    ):
        raise ValueError("因子检验相关配置必须是对象")
    return_probe_directory = (
        Path(str(output_cfg.get("return_probe", "artifacts/layer_probe/return_probe")))
        .expanduser()
        .resolve()
    )
    open_test = bool(strict_cfg.get("open_final_test", False))
    return_probe_directory = return_probe_directory / (
        "final_test" if open_test else "validation"
    )
    predictions_path = return_probe_directory / "return_oos_predictions.parquet"
    reference_path = return_probe_directory / "return_fit_reference_predictions.parquet"
    for path in (predictions_path, reference_path):
        if not path.is_file():
            raise FileNotFoundError(f"收益Probe预测不存在: {path}")
    predictions = pd.concat(
        [pd.read_parquet(predictions_path), pd.read_parquet(reference_path)],
        ignore_index=True,
    )
    evaluation_split = "test" if open_test else "validation"
    output = (
        Path(
            str(
                output_cfg.get(
                    "factor_validation", "artifacts/layer_probe/factor_validation"
                )
            )
        )
        .expanduser()
        .resolve()
    )
    output = output / ("final_test" if open_test else "validation")
    marker_path = output / "FINAL_TEST_OPENED.json"
    if open_test and marker_path.exists():
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        raise RuntimeError(
            "最终测试集已经打开过，拒绝重复运行。首次记录: "
            + json.dumps(marker, ensure_ascii=False)
        )
    reference = _prediction_wide(
        predictions,
        evaluation_split=evaluation_split,
        prediction_role="fit_reference",
    )
    evaluation = _prediction_wide(
        predictions,
        evaluation_split=evaluation_split,
        prediction_role="oos",
    )
    matrix, transform_manifest = build_cross_layer_factors(
        reference,
        evaluation,
        pca_components=int(factor_cfg.get("pca_components", 3)),
        middle_layers=factor_cfg.get("middle_layers", [4, 5, 6, 7, 8]),
        deep_layers=factor_cfg.get("deep_layers", [9, 10, 11, 12]),
    )
    available_factors = factor_columns(matrix)
    if open_test:
        selected = [str(value) for value in strict_cfg.get("selected_factors", [])]
        if not selected:
            raise ValueError(
                "打开最终测试前必须在strict_test.selected_factors预注册因子"
            )
        unknown = sorted(set(selected).difference(available_factors))
        if unknown:
            raise ValueError("预注册因子不存在: " + ", ".join(unknown))
        return_marker_path = return_probe_directory / "FINAL_RETURN_TEST_OPENED.json"
        if not return_marker_path.is_file():
            raise RuntimeError("收益最终测试没有受控开启记录，拒绝继续因子检验")
        return_marker = json.loads(return_marker_path.read_text(encoding="utf-8"))
        if return_marker.get("selected_factors_preregistered") != selected:
            raise RuntimeError("收益测试开启后selected_factors发生变化，拒绝检验")
        factors = selected
    else:
        factors = available_factors
    primary_horizon = int(returns_cfg.get("primary_horizon", 5))
    target_column = str(
        probe_cfg.get("target_column", f"target_return_rank_{primary_horizon}d")
    )
    return_column = f"industry_adjusted_return_fut{primary_horizon}d"
    min_observations = int(probe_cfg.get("min_daily_observations", 20))
    summary, daily = evaluate_factor_ic(
        matrix,
        factors=factors,
        target_column=target_column,
        min_daily_observations=min_observations,
        hac_lags=max(0, primary_horizon - 1),
    )
    quantile_groups, monotonicity = evaluate_quantile_monotonicity(
        matrix,
        factors=factors,
        return_column=return_column,
        quantiles=int(strict_cfg.get("quantiles", 5)),
        hac_lags=max(0, primary_horizon - 1),
    )
    stratified = evaluate_stratified_ic(
        matrix,
        factors=factors,
        target_column=target_column,
        min_daily_observations=min_observations,
    )
    incremental, incremental_daily = evaluate_incremental_ic(
        matrix,
        factors=factors,
        target_column=target_column,
        min_daily_observations=min_observations,
        hac_lags=max(0, primary_horizon - 1),
    )
    fingerprint = _fingerprint_frame(
        matrix,
        columns=["representation_row", "symbol", "trading_date", target_column],
    )
    final_marker = (
        {
            "opened_at": datetime.now().isoformat(timespec="seconds"),
            "evaluation_split": "test",
            "selected_factors": factors,
            "test_panel_fingerprint": fingerprint,
        }
        if open_test
        else None
    )
    _atomic_write_factor_outputs(
        output,
        tables={
            "candidate_factor_matrix.parquet": matrix,
            "daily_rank_ic.csv": daily,
            "factor_summary.csv": summary,
            "quantile_group_returns.csv": quantile_groups,
            "quantile_monotonicity.csv": monotonicity,
            "stratified_ic.csv": stratified,
            "incremental_ic.csv": incremental,
            "incremental_daily_ic.csv": incremental_daily,
        },
        manifest={
            "schema_version": "cross_layer_factor_validation_v1.0",
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "evaluation_split": evaluation_split,
            "factors": factors,
            "target_column": target_column,
            "return_column": return_column,
            "inference": (
                f"Newey-West HAC lag={max(0, primary_horizon - 1)}; "
                "Benjamini-Hochberg FDR"
            ),
            "factor_transform": transform_manifest,
            "panel_fingerprint": fingerprint,
        },
        final_test_marker=final_marker,
    )
    return output
