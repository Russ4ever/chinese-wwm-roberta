"""六阶段Layer Probe Notebook的预检、验收与绘图入口。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

from .config import load_yaml_config
from .layer_probe_factors import factor_columns
from .layer_probe_models import parse_time_windows
from .layer_probe_panel import validate_stock_day_artifacts
from .layer_probe_representations import validate_representation_artifacts


def load_layer_probe_config(path: str | Path) -> dict[str, object]:
    config = load_yaml_config(path)
    if not isinstance(config, dict):
        raise TypeError("Layer Probe配置顶层必须是对象")
    return config


def evaluation_output_directory(
    directory: str | Path, *, final_test: bool = False
) -> Path:
    """解析validation/final_test子目录，同时兼容直接传入具体运行目录。"""

    root = Path(directory).expanduser().resolve()
    candidate = root / ("final_test" if final_test else "validation")
    return candidate if candidate.exists() else root


def _path_status(
    name: str, value: object, *, directory: bool = False
) -> dict[str, object]:
    path = Path(str(value or "")).expanduser().resolve() if value else None
    exists = bool(path and (path.is_dir() if directory else path.is_file()))
    return {
        "check": name,
        "status": "ok" if exists else "missing",
        "blocking": True,
        "detail": str(path) if path else "未配置",
    }


def _table_columns(path: str | Path) -> set[str]:
    source = Path(path).expanduser().resolve()
    if source.suffix.lower() in {".parquet", ".pq"}:
        import pyarrow.parquet as pq

        return set(pq.ParquetFile(source).schema.names)
    suffixes = "".join(source.suffixes).lower()
    if suffixes.endswith(".csv.gz") or source.suffix.lower() == ".csv":
        return set(pd.read_csv(source, nrows=0).columns)
    return set()


def preflight_report(config: Mapping[str, object]) -> pd.DataFrame:
    """在任何昂贵计算前检查路径、切分、标签映射和最终测试开关。"""

    text = config.get("text", {})
    model = config.get("model", {})
    returns = config.get("returns", {})
    exposures = config.get("exposures", {})
    sentiment = config.get("sentiment_probe", {})
    return_probe = config.get("return_probe", {})
    strict = config.get("strict_test", {})
    splits = config.get("time_splits", {})
    mappings = (
        text,
        model,
        returns,
        exposures,
        sentiment,
        return_probe,
        strict,
        splits,
    )
    if not all(isinstance(value, Mapping) for value in mappings):
        raise ValueError("Layer Probe配置中的阶段配置必须是对象")
    records = [
        _path_status("text.path", text.get("path")),
        _path_status("model.checkpoint", model.get("checkpoint")),
        _path_status(
            "model.base_model_dir", model.get("base_model_dir"), directory=True
        ),
        _path_status(
            "returns.industry_adjusted_daily_path",
            returns.get("industry_adjusted_daily_path"),
        ),
    ]
    if text.get("sentiment_labels_path"):
        records.append(
            _path_status(
                "text.sentiment_labels_path", text.get("sentiment_labels_path")
            )
        )
        label_source = text.get("sentiment_labels_path")
    else:
        label_source = text.get("path")
    label_column = str(text.get("sentiment_label_column", "sentiment_label"))
    label_source_path = Path(str(label_source or "")).expanduser().resolve()
    if label_source_path.is_file():
        has_label = label_column in _table_columns(label_source_path)
        records.append(
            {
                "check": "sentiment_label_source",
                "status": "ok" if has_label else "invalid",
                "blocking": True,
                "detail": (
                    f"{label_column} found in {label_source_path}"
                    if has_label
                    else f"{label_source_path}没有{label_column}"
                ),
            }
        )
    for name in ("industry_path", "size_path"):
        if exposures.get(name):
            record = _path_status(f"exposures.{name}", exposures.get(name))
            record["blocking"] = False
            records.append(record)
    try:
        windows = parse_time_windows(splits)
        records.append(
            {
                "check": "time_splits",
                "status": "ok",
                "blocking": True,
                "detail": "; ".join(
                    f"{window.name}:{window.start.date()}~{window.end.date()}"
                    for window in windows
                ),
            }
        )
    except (TypeError, ValueError) as exc:
        records.append(
            {
                "check": "time_splits",
                "status": "invalid",
                "blocking": True,
                "detail": str(exc),
            }
        )
    mapping_confirmed = bool(sentiment.get("label_mapping_confirmed", False))
    records.append(
        {
            "check": "sentiment_label_mapping",
            "status": "ok" if mapping_confirmed else "warning",
            "blocking": False,
            "detail": (
                "class语义已确认"
                if mapping_confirmed
                else "仍可运行数值诊断，但不得把class_1命名为正/负情绪"
            ),
        }
    )
    return_test = bool(return_probe.get("open_final_test", False))
    factor_test = bool(strict.get("open_final_test", False))
    selected = [str(value) for value in strict.get("selected_factors", [])]
    strict_valid = (not return_test and not factor_test) or (
        return_test and factor_test and bool(selected)
    )
    records.append(
        {
            "check": "final_test_gate",
            "status": "ok" if strict_valid else "invalid",
            "blocking": True,
            "detail": (
                "test保持关闭"
                if not return_test and not factor_test
                else f"return_test={return_test}, factor_test={factor_test}, selected={selected}"
            ),
        }
    )
    return pd.DataFrame(records)


def assert_preflight(
    report: pd.DataFrame, *, allow_missing_optional: bool = True
) -> None:
    invalid = report[report["blocking"].eq(True) & ~report["status"].eq("ok")]
    if not invalid.empty:
        raise RuntimeError(
            "预检未通过:\n" + invalid[["check", "detail"]].to_string(index=False)
        )
    if not allow_missing_optional:
        missing = report[report["status"].eq("missing")]
        if not missing.empty:
            raise RuntimeError(
                "可选输入缺失:\n" + missing[["check", "detail"]].to_string(index=False)
            )


def validate_sentiment_probe_outputs(directory: str | Path) -> dict[str, object]:
    root = evaluation_output_directory(directory)
    paths = {
        "metrics": root / "sentiment_metrics.csv",
        "predictions": root / "sentiment_oos_predictions.parquet",
        "distributions": root / "sentiment_logit_distributions.csv",
        "diagnostic": root / "probability_compression_diagnostic.csv",
        "manifest": root / "manifest.json",
    }
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(f"情绪Probe产物缺失: {path}")
    metrics = pd.read_csv(paths["metrics"])
    predictions = pd.read_parquet(paths["predictions"])
    layer_metrics = metrics[metrics["model_kind"].eq("layer_logistic")]
    for split in layer_metrics["split"].unique():
        if set(layer_metrics.loc[layer_metrics["split"].eq(split), "layer"]) != set(
            range(13)
        ):
            raise ValueError(f"情绪{split}没有完整13层指标")
    if predictions[["logit", "probability"]].isna().any().any():
        raise ValueError("情绪OOS预测含NaN")
    return {
        "metric_rows": len(metrics),
        "prediction_rows": len(predictions),
        "splits": sorted(metrics["split"].unique()),
    }


def validate_return_probe_outputs(directory: str | Path) -> dict[str, object]:
    root = evaluation_output_directory(directory)
    metrics_path = root / "return_probe_metrics.csv"
    predictions_path = root / "return_oos_predictions.parquet"
    tuning_path = root / "return_probe_tuning.csv"
    reference_path = root / "return_fit_reference_predictions.parquet"
    for path in (
        metrics_path,
        predictions_path,
        reference_path,
        tuning_path,
        root / "manifest.json",
    ):
        if not path.is_file():
            raise FileNotFoundError(f"收益Probe产物缺失: {path}")
    metrics = pd.read_csv(metrics_path)
    predictions = pd.read_parquet(predictions_path)
    reference = pd.read_parquet(reference_path)
    for split in metrics["split"].unique():
        if set(metrics.loc[metrics["split"].eq(split), "layer"]) != set(range(13)):
            raise ValueError(f"收益{split}没有完整13层指标")
    if not predictions["prediction_role"].eq("oos").all():
        raise ValueError("return_oos_predictions混入非OOS行")
    if not reference["prediction_role"].eq("fit_reference").all():
        raise ValueError("return_fit_reference_predictions混入OOS行")
    if predictions["prediction"].isna().any():
        raise ValueError("收益OOS预测含NaN")
    return {
        "metric_rows": len(metrics),
        "prediction_rows": len(predictions) + len(reference),
        "oos_rows": len(predictions),
        "fit_reference_rows": len(reference),
        "splits": sorted(metrics["split"].unique()),
    }


def validate_factor_outputs(directory: str | Path) -> dict[str, object]:
    root = evaluation_output_directory(directory)
    required = [
        root / "candidate_factor_matrix.parquet",
        root / "daily_rank_ic.csv",
        root / "factor_summary.csv",
        root / "quantile_monotonicity.csv",
        root / "stratified_ic.csv",
        root / "incremental_ic.csv",
        root / "manifest.json",
    ]
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(f"跨层因子产物缺失: {path}")
    matrix = pd.read_parquet(root / "candidate_factor_matrix.parquet")
    summary = pd.read_csv(root / "factor_summary.csv")
    factors = factor_columns(matrix)
    if not factors:
        raise ValueError("候选因子矩阵没有跨层因子")
    if summary["factor"].duplicated().any():
        raise ValueError("因子汇总存在重复因子")
    return {
        "rows": len(matrix),
        "candidate_factors": len(factors),
        "evaluated_factors": len(summary),
    }


def validate_pipeline_outputs(config: Mapping[str, object]) -> pd.DataFrame:
    """集中执行Notebook末尾验收；只验证已经存在的阶段目录。"""

    output = config.get("output", {})
    if not isinstance(output, Mapping):
        raise ValueError("output配置必须是对象")
    sentiment_config = config.get("sentiment_probe", {})
    return_config = config.get("return_probe", {})
    strict_config = config.get("strict_test", {})
    validators = [
        ("representations", validate_representation_artifacts, False, False),
        (
            "sentiment_probe",
            validate_sentiment_probe_outputs,
            True,
            bool(
                sentiment_config.get("open_final_test", False)
                if isinstance(sentiment_config, Mapping)
                else False
            ),
        ),
        ("stock_day_panel", validate_stock_day_artifacts, False, False),
        (
            "return_probe",
            validate_return_probe_outputs,
            True,
            bool(
                return_config.get("open_final_test", False)
                if isinstance(return_config, Mapping)
                else False
            ),
        ),
        (
            "factor_validation",
            validate_factor_outputs,
            True,
            bool(
                strict_config.get("open_final_test", False)
                if isinstance(strict_config, Mapping)
                else False
            ),
        ),
    ]
    records: list[dict[str, object]] = []
    for key, validator, has_runs, final_test in validators:
        path = Path(str(output.get(key, ""))).expanduser().resolve()
        if has_runs:
            path = path / ("final_test" if final_test else "validation")
        if not path.exists():
            records.append({"stage": key, "status": "not_run", "detail": str(path)})
            continue
        try:
            detail = validator(path)
            records.append(
                {
                    "stage": key,
                    "status": "ok",
                    "detail": json.dumps(detail, ensure_ascii=False, default=str),
                }
            )
        except (OSError, TypeError, ValueError, KeyError, json.JSONDecodeError) as exc:
            records.append({"stage": key, "status": "failed", "detail": str(exc)})
    return pd.DataFrame(records)


def plot_sentiment_curve(directory: str | Path, *, split: str = "validation"):
    import matplotlib.pyplot as plt

    root = evaluation_output_directory(directory, final_test=split == "test")
    metrics = pd.read_csv(root / "sentiment_metrics.csv")
    layers = metrics[
        metrics["model_kind"].eq("layer_logistic") & metrics["split"].eq(split)
    ].sort_values("layer")
    head = metrics[
        metrics["model_kind"].eq("original_fc_head") & metrics["split"].eq(split)
    ]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(layers["layer"], layers["auc"], marker="o", label="Logistic probe")
    axes[1].plot(layers["layer"], layers["pr_auc"], marker="o", label="Logistic probe")
    if not head.empty:
        axes[0].axhline(head.iloc[0]["auc"], linestyle="--", label="original fc")
        axes[1].axhline(head.iloc[0]["pr_auc"], linestyle="--", label="original fc")
    axes[0].set(title=f"Sentiment AUC ({split})", xlabel="Layer", ylabel="AUC")
    axes[1].set(title=f"Sentiment PR-AUC ({split})", xlabel="Layer", ylabel="PR-AUC")
    for axis in axes:
        axis.grid(alpha=0.2)
        axis.legend()
    fig.tight_layout()
    return fig


def plot_logit_comparison(directory: str | Path, *, split: str = "validation"):
    import matplotlib.pyplot as plt

    root = evaluation_output_directory(directory, final_test=split == "test")
    predictions = pd.read_parquet(root / "sentiment_oos_predictions.parquet")
    selected = predictions[
        predictions["split"].eq(split)
        & (
            predictions["model_kind"].eq("original_fc_head")
            | (
                predictions["model_kind"].eq("layer_logistic")
                & predictions["layer"].eq(12)
            )
        )
    ]
    fig, ax = plt.subplots(figsize=(7, 4))
    for name, group in selected.groupby("model_kind"):
        ax.hist(group["logit"], bins=50, density=True, alpha=0.45, label=name)
    ax.set(title=f"OOS logit distribution ({split})", xlabel="logit", ylabel="density")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_return_curve(directory: str | Path, *, split: str = "validation"):
    import matplotlib.pyplot as plt

    root = evaluation_output_directory(directory, final_test=split == "test")
    metrics = pd.read_csv(root / "return_probe_metrics.csv")
    selected = metrics[metrics["split"].eq(split)].sort_values("layer")
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(selected["layer"], selected["rank_ic"], marker="o")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set(
        title=f"Return decodability ({split})", xlabel="Layer", ylabel="Mean Rank IC"
    )
    ax.grid(alpha=0.2)
    fig.tight_layout()
    return fig


def plot_factor_summary(directory: str | Path, *, final_test: bool = False):
    import matplotlib.pyplot as plt

    root = evaluation_output_directory(directory, final_test=final_test)
    summary = pd.read_csv(root / "factor_summary.csv").sort_values("rank_ic")
    fig, ax = plt.subplots(figsize=(9, max(4, 0.3 * len(summary))))
    colors = np.where(summary["qvalue_bh"] <= 0.05, "tab:blue", "tab:gray")
    ax.barh(summary["factor"], summary["rank_ic"], color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set(title="Cross-layer factor Rank IC", xlabel="Mean daily Rank IC")
    fig.tight_layout()
    return fig
