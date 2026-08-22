"""逐层 Logistic/Ridge Probe、时间切分与完全 OOS 预测。"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler

from .layer_probe_representations import (
    protocol_config_hash,
    representation_artifacts,
    validate_representation_artifacts,
)


SPLIT_NAMES = ("train", "validation", "test")


@dataclass(frozen=True)
class TimeWindow:
    name: str
    start: pd.Timestamp
    end: pd.Timestamp


def _parse_date(value: object, *, name: str) -> pd.Timestamp:
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        raise ValueError(f"{name}不是有效日期: {value!r}")
    return pd.Timestamp(parsed).normalize()


def parse_time_windows(config: Mapping[str, object]) -> list[TimeWindow]:
    """解析显式、互不重叠的train/validation/test窗口。"""

    windows: list[TimeWindow] = []
    for name in SPLIT_NAMES:
        raw = config.get(name)
        if not isinstance(raw, Mapping):
            raise ValueError(f"time_splits.{name}必须是对象")
        start = _parse_date(raw.get("start"), name=f"time_splits.{name}.start")
        end = _parse_date(raw.get("end"), name=f"time_splits.{name}.end")
        if end < start:
            raise ValueError(f"time_splits.{name}日期倒置")
        windows.append(TimeWindow(name, start, end))
    for left, right in zip(windows, windows[1:]):
        if left.end >= right.start:
            raise ValueError(f"{left.name}与{right.name}重叠或没有隔离区间")
    return windows


def assign_purged_time_splits(
    frame: pd.DataFrame,
    *,
    config: Mapping[str, object],
    date_column: str,
    label_end_column: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """按特征时间划分，并清除跨越下一分区起点的未来Label。

    ``label_end_column`` 为空时适用于同期情绪标签；收益任务必须传入最长收益窗口的
    结束日，使训练/验证样本的未来收益区间严格早于下一分区开始日。
    """

    windows = parse_time_windows(config)
    out = frame.copy()
    dates = pd.to_datetime(out[date_column], errors="coerce").dt.normalize()
    if dates.isna().any():
        raise ValueError(f"{date_column}含无效日期")
    label_end = None
    if label_end_column:
        if label_end_column not in out.columns:
            raise ValueError(f"缺少隔离期字段{label_end_column}")
        label_end = pd.to_datetime(
            out[label_end_column], errors="coerce"
        ).dt.normalize()
    split = pd.Series(pd.NA, index=out.index, dtype="string")
    audit: list[dict[str, object]] = []
    in_any_window = pd.Series(False, index=out.index)
    purged = pd.Series(False, index=out.index)
    for index, window in enumerate(windows):
        feature_mask = dates.between(window.start, window.end, inclusive="both")
        in_any_window |= feature_mask
        eligible = feature_mask.copy()
        if label_end is not None and index < len(windows) - 1:
            before_next = label_end < windows[index + 1].start
            purged |= feature_mask & ~before_next.fillna(False)
            eligible &= before_next.fillna(False)
        elif label_end is not None:
            eligible &= label_end.notna()
        split.loc[eligible] = window.name
        audit.append(
            {
                "stage": "split",
                "reason": f"included_{window.name}",
                "count": int(eligible.sum()),
            }
        )
    audit.extend(
        [
            {
                "stage": "split_excluded",
                "reason": "outside_feature_windows",
                "count": int((~in_any_window).sum()),
            },
            {
                "stage": "split_excluded",
                "reason": "purged_future_label_crosses_next_split",
                "count": int(purged.sum()),
            },
        ]
    )
    out["split"] = split
    out = out[out["split"].notna()].reset_index(drop=True)
    if out.empty:
        raise ValueError("时间切分和隔离期过滤后没有样本")
    return out, pd.DataFrame(audit)


def load_text_representations(
    directory: str | Path,
) -> tuple[np.memmap, pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """加载阶段1产物并验证逐行对齐。"""

    validate_representation_artifacts(directory)
    artifacts = representation_artifacts(directory)
    array = np.load(artifacts.representations, mmap_mode="r")
    metadata = pd.read_parquet(artifacts.metadata)
    head = pd.read_parquet(artifacts.fixed_head_outputs)
    manifest = json.loads(artifacts.manifest.read_text(encoding="utf-8"))
    return array, metadata, head, manifest


def _binary_metrics(y: np.ndarray, probability: np.ndarray) -> dict[str, float | int]:
    if len(np.unique(y)) != 2:
        raise ValueError("AUC/PR-AUC要求评价集同时包含0和1")
    clipped = np.clip(probability, 1e-7, 1 - 1e-7)
    return {
        "n": int(len(y)),
        "positive_rate": float(np.mean(y)),
        "auc": float(roc_auc_score(y, probability)),
        "pr_auc": float(average_precision_score(y, probability)),
        "log_loss": float(log_loss(y, clipped, labels=[0, 1])),
    }


def _distribution_metrics(
    logit: np.ndarray, probability: np.ndarray
) -> dict[str, float]:
    quantiles = np.quantile(logit, [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
    return {
        "logit_mean": float(np.mean(logit)),
        "logit_std": float(np.std(logit, ddof=1)) if len(logit) > 1 else np.nan,
        "logit_q01": float(quantiles[0]),
        "logit_q05": float(quantiles[1]),
        "logit_q25": float(quantiles[2]),
        "logit_q50": float(quantiles[3]),
        "logit_q75": float(quantiles[4]),
        "logit_q95": float(quantiles[5]),
        "logit_q99": float(quantiles[6]),
        "probability_mean": float(np.mean(probability)),
        "probability_std": (
            float(np.std(probability, ddof=1)) if len(probability) > 1 else np.nan
        ),
        "probability_near_half_fraction": float(
            np.mean((probability >= 0.45) & (probability <= 0.55))
        ),
    }


def fit_sentiment_layer_probes(
    representations: np.ndarray,
    metadata: pd.DataFrame,
    head_outputs: pd.DataFrame,
    *,
    split_config: Mapping[str, object],
    c_grid: Sequence[float],
    positive_class_index: int,
    label_mapping_confirmed: bool,
    include_test: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """拟合13个Logistic Probe并生成完全OOS预测与原头对照。"""

    if representations.ndim != 3 or representations.shape[1] != 13:
        raise ValueError("情绪Probe要求representation形状为[N,13,H]")
    if len(metadata) != representations.shape[0] or len(head_outputs) != len(metadata):
        raise ValueError("representation、metadata与head outputs行数不一致")
    if "sentiment_label" not in metadata.columns:
        raise ValueError("文本元数据没有sentiment_label，无法运行情绪Probe")
    if positive_class_index not in {0, 1}:
        raise ValueError("positive_class_index必须是0或1")
    c_values = sorted({float(value) for value in c_grid})
    if not c_values or min(c_values) <= 0:
        raise ValueError("logistic C grid必须为正数")

    labeled = metadata[metadata["sentiment_label"].notna()].copy()
    labeled, split_audit = assign_purged_time_splits(
        labeled,
        config=split_config,
        date_column="trading_date",
    )
    labeled["sentiment_label"] = labeled["sentiment_label"].astype(int)
    positions = {
        split: labeled.loc[labeled["split"].eq(split), "representation_row"].to_numpy(
            dtype=int
        )
        for split in SPLIT_NAMES
    }
    y_by_split = {
        split: labeled.loc[labeled["split"].eq(split), "sentiment_label"].to_numpy(
            dtype=int
        )
        for split in SPLIT_NAMES
    }
    required_splits = ["train", "validation"] + (["test"] if include_test else [])
    for split in required_splits:
        if len(positions[split]) == 0 or len(np.unique(y_by_split[split])) != 2:
            raise ValueError(f"情绪{split}分区为空或不同时包含0/1")

    prediction_records: list[pd.DataFrame] = []
    metric_records: list[dict[str, object]] = []
    distribution_records: list[dict[str, object]] = []
    tuning_records: list[dict[str, object]] = []
    evaluation_splits = ["validation"] + (["test"] if include_test else [])
    for layer in range(13):
        x_train = np.asarray(
            representations[positions["train"], layer, :], dtype=np.float32
        )
        scaler = StandardScaler().fit(x_train)
        x_train_scaled = scaler.transform(x_train)
        best_c: float | None = None
        best_auc = -np.inf
        for c_value in c_values:
            candidate = LogisticRegression(
                C=c_value,
                penalty="l2",
                solver="lbfgs",
                max_iter=2_000,
                random_state=0,
            ).fit(x_train_scaled, y_by_split["train"])
            val_probability = candidate.predict_proba(
                scaler.transform(
                    np.asarray(
                        representations[positions["validation"], layer, :],
                        dtype=np.float32,
                    )
                )
            )[:, 1]
            val_auc = float(roc_auc_score(y_by_split["validation"], val_probability))
            tuning_records.append(
                {"layer": layer, "C": c_value, "validation_auc": val_auc}
            )
            if val_auc > best_auc:
                best_auc = val_auc
                best_c = c_value
        assert best_c is not None
        model = LogisticRegression(
            C=best_c,
            penalty="l2",
            solver="lbfgs",
            max_iter=2_000,
            random_state=0,
        ).fit(x_train_scaled, y_by_split["train"])
        for split in evaluation_splits:
            x_eval = scaler.transform(
                np.asarray(
                    representations[positions[split], layer, :], dtype=np.float32
                )
            )
            logit = model.decision_function(x_eval)
            probability = model.predict_proba(x_eval)[:, 1]
            rows = labeled[labeled["split"].eq(split)].copy()
            prediction_records.append(
                pd.DataFrame(
                    {
                        "representation_row": rows["representation_row"].to_numpy(),
                        "text_id": rows["text_id"].astype(str).to_numpy(),
                        "trading_date": rows["trading_date"].to_numpy(),
                        "split": split,
                        "model_kind": "layer_logistic",
                        "layer": layer,
                        "label": y_by_split[split],
                        "logit": logit,
                        "probability": probability,
                    }
                )
            )
            metric_records.append(
                {
                    "model_kind": "layer_logistic",
                    "layer": layer,
                    "split": split,
                    "selected_C": best_c,
                    **_binary_metrics(y_by_split[split], probability),
                }
            )
            distribution_records.append(
                {
                    "model_kind": "layer_logistic",
                    "layer": layer,
                    "split": split,
                    **_distribution_metrics(logit, probability),
                }
            )

    head = metadata[["representation_row", "text_id", "trading_date"]].merge(
        head_outputs,
        on=["representation_row", "text_id"],
        how="left",
        validate="one_to_one",
    )
    if positive_class_index == 1:
        head_logit_column = "logit_margin_1_minus_0"
        head_probability_column = "class_1_prob"
    else:
        head["logit_margin_0_minus_1"] = -head["logit_margin_1_minus_0"]
        head_logit_column = "logit_margin_0_minus_1"
        head_probability_column = "class_0_prob"
    for split in evaluation_splits:
        rows = labeled[labeled["split"].eq(split)]
        selected = head.set_index("representation_row").loc[
            rows["representation_row"].to_numpy(dtype=int)
        ]
        logit = selected[head_logit_column].to_numpy(dtype=float)
        probability = selected[head_probability_column].to_numpy(dtype=float)
        prediction_records.append(
            pd.DataFrame(
                {
                    "representation_row": rows["representation_row"].to_numpy(),
                    "text_id": rows["text_id"].astype(str).to_numpy(),
                    "trading_date": rows["trading_date"].to_numpy(),
                    "split": split,
                    "model_kind": "original_fc_head",
                    "layer": 12,
                    "label": y_by_split[split],
                    "logit": logit,
                    "probability": probability,
                }
            )
        )
        metric_records.append(
            {
                "model_kind": "original_fc_head",
                "layer": 12,
                "split": split,
                "selected_C": np.nan,
                "label_mapping_confirmed": bool(label_mapping_confirmed),
                **_binary_metrics(y_by_split[split], probability),
            }
        )
        distribution_records.append(
            {
                "model_kind": "original_fc_head",
                "layer": 12,
                "split": split,
                **_distribution_metrics(logit, probability),
            }
        )
    predictions = pd.concat(prediction_records, ignore_index=True)
    return (
        pd.DataFrame(metric_records),
        predictions,
        pd.DataFrame(distribution_records),
        pd.concat([split_audit, pd.DataFrame(tuning_records)], ignore_index=True),
    )


def diagnose_probability_compression(
    metrics: pd.DataFrame,
    distributions: pd.DataFrame,
    *,
    split: str = "validation",
) -> pd.DataFrame:
    """并排输出Layer12 Logistic与原fc头，供Notebook判断概率压扁来源。"""

    left = metrics[
        metrics["split"].eq(split)
        & metrics["model_kind"].eq("layer_logistic")
        & metrics["layer"].eq(12)
    ]
    right = metrics[
        metrics["split"].eq(split) & metrics["model_kind"].eq("original_fc_head")
    ]
    left_dist = distributions[
        distributions["split"].eq(split)
        & distributions["model_kind"].eq("layer_logistic")
        & distributions["layer"].eq(12)
    ]
    right_dist = distributions[
        distributions["split"].eq(split)
        & distributions["model_kind"].eq("original_fc_head")
    ]
    if any(frame.empty for frame in (left, right, left_dist, right_dist)):
        raise ValueError("缺少Layer12或原分类头的对照结果")
    columns = ["auc", "pr_auc", "log_loss"]
    distribution_columns = [
        "logit_std",
        "probability_std",
        "probability_near_half_fraction",
    ]
    return pd.DataFrame(
        [
            {
                "model": "layer12_logistic_probe",
                **left.iloc[0][columns].to_dict(),
                **left_dist.iloc[0][distribution_columns].to_dict(),
            },
            {
                "model": "original_fc_head",
                **right.iloc[0][columns].to_dict(),
                **right_dist.iloc[0][distribution_columns].to_dict(),
            },
        ]
    )


def daily_rank_ic(
    frame: pd.DataFrame,
    *,
    prediction_column: str,
    target_column: str,
    date_column: str = "trading_date",
    min_observations: int = 20,
) -> pd.DataFrame:
    """计算日度截面Spearman Rank IC。"""

    records: list[dict[str, object]] = []
    for date, group in frame.groupby(date_column, sort=True):
        data = (
            group[[prediction_column, target_column]]
            .replace([np.inf, -np.inf], np.nan)
            .dropna()
        )
        ic = np.nan
        if (
            len(data) >= min_observations
            and data[prediction_column].nunique() > 1
            and data[target_column].nunique() > 1
        ):
            value = spearmanr(data[prediction_column], data[target_column]).correlation
            ic = float(value) if np.isfinite(value) else np.nan
        records.append({"trading_date": date, "ic": ic, "n_obs": len(data)})
    return pd.DataFrame(records)


def summarize_rank_ic(ic: pd.Series | np.ndarray) -> dict[str, float | int]:
    values = np.asarray(ic, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return {
            "rank_ic": np.nan,
            "ic_std": np.nan,
            "icir": np.nan,
            "t_stat": np.nan,
            "positive_fraction": np.nan,
            "n_days": 0,
        }
    mean = float(np.mean(values))
    std = float(np.std(values, ddof=1)) if len(values) > 1 else np.nan
    return {
        "rank_ic": mean,
        "ic_std": std,
        "icir": mean / std if np.isfinite(std) and std > 0 else np.nan,
        "t_stat": (
            mean / std * np.sqrt(len(values))
            if np.isfinite(std) and std > 0
            else np.nan
        ),
        "positive_fraction": float(np.mean(values > 0)),
        "n_days": int(len(values)),
    }


def fit_return_layer_probes(
    representations: np.ndarray,
    panel: pd.DataFrame,
    *,
    target_column: str,
    alpha_grid: Sequence[float],
    min_daily_observations: int,
    include_test: bool,
    selected_alpha_by_layer: Mapping[int, float] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """逐层拟合Ridge；validation用train拟合，test用train+validation重拟合。"""

    if representations.ndim != 3 or representations.shape[1] != 13:
        raise ValueError("收益Probe要求representation形状为[N,13,H]")
    required = {
        "representation_row",
        "symbol",
        "trading_date",
        "split",
        target_column,
    }
    missing = sorted(required.difference(panel.columns))
    if missing:
        raise ValueError("股票日panel缺少字段: " + ", ".join(missing))
    representation_rows = pd.to_numeric(
        panel["representation_row"], errors="coerce"
    )
    if representation_rows.isna().any():
        raise ValueError("股票日panel含无效representation_row")
    integer_rows = representation_rows.to_numpy(dtype=np.int64)
    if not np.array_equal(representation_rows.to_numpy(dtype=float), integer_rows):
        raise ValueError("股票日panel的representation_row必须是整数")
    if (integer_rows < 0).any() or (integer_rows >= representations.shape[0]).any():
        raise ValueError("股票日panel的representation_row越界")
    alphas = sorted({float(value) for value in alpha_grid})
    if not alphas or min(alphas) < 0:
        raise ValueError("Ridge alpha必须非负")
    usable = panel[panel[target_column].notna()].copy()
    positions = {
        split: usable.loc[usable["split"].eq(split), "representation_row"].to_numpy(
            dtype=int
        )
        for split in SPLIT_NAMES
    }
    y = {
        split: usable.loc[usable["split"].eq(split), target_column].to_numpy(
            dtype=float
        )
        for split in SPLIT_NAMES
    }
    for split in ["train", "validation"] + (["test"] if include_test else []):
        if len(positions[split]) == 0:
            raise ValueError(f"收益{split}分区没有有效{target_column}")

    predictions: list[pd.DataFrame] = []
    metrics: list[dict[str, object]] = []
    tuning: list[dict[str, object]] = []
    prediction_columns = [
        column
        for column in (
            "representation_row",
            "symbol",
            "trading_date",
            "split",
            "n_texts",
            "fixed_head_margin",
            "fixed_head_class_1_probability",
            "industry",
            "size",
            target_column,
        )
        if column in usable.columns
    ]
    prediction_columns.extend(
        column
        for column in usable.columns
        if column.startswith("industry_adjusted_return_fut")
        and column not in prediction_columns
    )
    for layer in range(13):
        x_train = np.asarray(
            representations[positions["train"], layer, :], dtype=np.float32
        )
        scaler = StandardScaler().fit(x_train)
        x_train_scaled = scaler.transform(x_train)
        x_validation = scaler.transform(
            np.asarray(
                representations[positions["validation"], layer, :], dtype=np.float32
            )
        )
        best_alpha: float | None = None
        best_ic = -np.inf
        if selected_alpha_by_layer is not None:
            if layer not in selected_alpha_by_layer:
                raise ValueError(f"冻结的收益alpha缺少Layer {layer}")
            best_alpha = float(selected_alpha_by_layer[layer])
            if best_alpha not in alphas:
                raise ValueError(f"冻结的收益alpha不在当前网格: layer={layer}, alpha={best_alpha}")
            tuning.append(
                {
                    "layer": layer,
                    "alpha": best_alpha,
                    "validation_rank_ic": np.nan,
                    "selection_source": "frozen_validation_manifest",
                }
            )
        else:
            for alpha in alphas:
                candidate = Ridge(alpha=alpha).fit(x_train_scaled, y["train"])
                validation_prediction = candidate.predict(x_validation)
                validation_rows = usable[usable["split"].eq("validation")].copy()
                validation_rows["prediction"] = validation_prediction
                ic_table = daily_rank_ic(
                    validation_rows,
                    prediction_column="prediction",
                    target_column=target_column,
                    min_observations=min_daily_observations,
                )
                score = summarize_rank_ic(ic_table["ic"])["rank_ic"]
                tuning.append(
                    {
                        "layer": layer,
                        "alpha": alpha,
                        "validation_rank_ic": score,
                    }
                )
                comparable = -np.inf if not np.isfinite(score) else float(score)
                if (
                    best_alpha is None
                    or comparable > best_ic
                    or (comparable == best_ic and alpha > best_alpha)
                ):
                    best_ic = comparable
                    best_alpha = alpha
        if best_alpha is None:
            raise ValueError(f"Layer {layer}无法在validation计算有效Rank IC")
        validation_model = Ridge(alpha=best_alpha).fit(x_train_scaled, y["train"])
        train_reference = usable.loc[
            usable["split"].eq("train"), prediction_columns
        ].copy()
        train_reference["prediction"] = validation_model.predict(x_train_scaled)
        train_reference["layer"] = layer
        train_reference["prediction_role"] = "fit_reference"
        train_reference["evaluation_split"] = "validation"
        predictions.append(train_reference)
        validation_prediction = validation_model.predict(x_validation)
        validation_rows = usable.loc[
            usable["split"].eq("validation"), prediction_columns
        ].copy()
        validation_rows["prediction"] = validation_prediction
        validation_rows["layer"] = layer
        validation_rows["prediction_role"] = "oos"
        validation_rows["evaluation_split"] = "validation"
        predictions.append(validation_rows)
        validation_ic = daily_rank_ic(
            validation_rows,
            prediction_column="prediction",
            target_column=target_column,
            min_observations=min_daily_observations,
        )
        validation_summary = summarize_rank_ic(validation_ic["ic"])
        validation_summary["icir_annualized"] = (
            float(validation_summary["icir"] * np.sqrt(252))
            if np.isfinite(validation_summary["icir"])
            else np.nan
        )
        metrics.append(
            {
                "layer": layer,
                "split": "validation",
                "selected_alpha": best_alpha,
                **validation_summary,
            }
        )
        if include_test:
            train_validation_mask = usable["split"].isin(["train", "validation"])
            train_validation_rows = usable.loc[train_validation_mask]
            train_validation_positions = train_validation_rows[
                "representation_row"
            ].to_numpy(dtype=int)
            final_scaler = StandardScaler().fit(
                np.asarray(
                    representations[train_validation_positions, layer, :],
                    dtype=np.float32,
                )
            )
            final_model = Ridge(alpha=best_alpha).fit(
                final_scaler.transform(
                    np.asarray(
                        representations[train_validation_positions, layer, :],
                        dtype=np.float32,
                    )
                ),
                train_validation_rows[target_column].to_numpy(dtype=float),
            )
            train_validation_reference = train_validation_rows[
                prediction_columns
            ].copy()
            train_validation_reference["prediction"] = final_model.predict(
                final_scaler.transform(
                    np.asarray(
                        representations[train_validation_positions, layer, :],
                        dtype=np.float32,
                    )
                )
            )
            train_validation_reference["layer"] = layer
            train_validation_reference["prediction_role"] = "fit_reference"
            train_validation_reference["evaluation_split"] = "test"
            predictions.append(train_validation_reference)
            test_prediction = final_model.predict(
                final_scaler.transform(
                    np.asarray(
                        representations[positions["test"], layer, :], dtype=np.float32
                    )
                )
            )
            test_rows = usable.loc[
                usable["split"].eq("test"), prediction_columns
            ].copy()
            test_rows["prediction"] = test_prediction
            test_rows["layer"] = layer
            test_rows["prediction_role"] = "oos"
            test_rows["evaluation_split"] = "test"
            predictions.append(test_rows)
            test_ic = daily_rank_ic(
                test_rows,
                prediction_column="prediction",
                target_column=target_column,
                min_observations=min_daily_observations,
            )
            test_summary = summarize_rank_ic(test_ic["ic"])
            test_summary["icir_annualized"] = (
                float(test_summary["icir"] * np.sqrt(252))
                if np.isfinite(test_summary["icir"])
                else np.nan
            )
            metrics.append(
                {
                    "layer": layer,
                    "split": "test",
                    "selected_alpha": best_alpha,
                    **test_summary,
                }
            )
    return (
        pd.DataFrame(metrics),
        pd.concat(predictions, ignore_index=True),
        pd.DataFrame(tuning),
    )


def _atomic_write_probe_outputs(
    output_directory: str | Path,
    *,
    tables: Mapping[str, pd.DataFrame],
    manifest: Mapping[str, object],
    final_test_marker_name: str | None = None,
    final_test_marker: Mapping[str, object] | None = None,
) -> Path:
    output = Path(output_directory).expanduser().resolve()
    if output == Path(output.anchor) or len(output.parts) < 3:
        raise ValueError(f"拒绝写入过宽目录: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent))
    if output.exists():
        raise FileExistsError(f"Probe产物已存在，拒绝覆盖: {output}")
    try:
        for filename, table in tables.items():
            path = temporary / filename
            if path.suffix == ".csv":
                table.to_csv(path, index=False)
            else:
                table.to_parquet(path, index=False, compression="zstd")
        (temporary / "manifest.json").write_text(
            json.dumps(dict(manifest), ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        if final_test_marker_name is not None:
            if final_test_marker is None:
                raise ValueError("配置marker文件名时必须提供marker内容")
            (temporary / final_test_marker_name).write_text(
                json.dumps(
                    dict(final_test_marker), ensure_ascii=False, indent=2, default=str
                ),
                encoding="utf-8",
            )
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return output


def run_optional_sentiment_probe_stage(config: Mapping[str, object]) -> Path:
    """运行默认关闭、与主流程解耦的二元情绪附录。"""

    from .layer_probe_representations import resolve_representation_directory

    output_cfg = config.get("output", {})
    probe_cfg = config.get("sentiment_appendix", {})
    split_cfg = config.get("return_time_splits", {})
    if not all(isinstance(value, Mapping) for value in (output_cfg, probe_cfg, split_cfg)):
        raise ValueError("output/sentiment_appendix/return_time_splits配置必须是对象")
    if not bool(probe_cfg.get("enabled", False)):
        raise RuntimeError("sentiment_appendix.enabled=false，附录未启用")
    if not bool(probe_cfg.get("label_mapping_confirmed", False)):
        raise RuntimeError("情绪附录要求先确认0/1类别映射")
    label_path = Path(str(probe_cfg.get("labels_path", ""))).expanduser().resolve()
    if not label_path.is_file():
        raise FileNotFoundError(f"情绪附录标签表不存在: {label_path}")
    labels = pd.read_parquet(label_path) if label_path.suffix.lower() in {".parquet", ".pq"} else pd.read_csv(label_path)
    id_column = str(probe_cfg.get("label_id_column", "report_id"))
    label_column = str(probe_cfg.get("label_column", "sentiment_label"))
    if id_column not in labels or label_column not in labels:
        raise ValueError("情绪附录标签表缺少ID或Label字段")
    representations, metadata, head, rep_manifest = load_text_representations(
        resolve_representation_directory(config)
    )
    metadata = metadata.merge(
        labels[[id_column, label_column]].rename(columns={id_column: "report_id", label_column: "sentiment_label"}),
        on="report_id",
        how="left",
        validate="one_to_one",
    ).rename(columns={"report_id": "text_id", "feature_available_date": "trading_date"})
    final_head = head[head["layer"].eq(12)].merge(
        metadata[["representation_row", "text_id"]],
        on="representation_row",
        validate="one_to_one",
    )
    include_test = bool(probe_cfg.get("open_final_test", False))
    run_directory = Path(str(output_cfg.get("run_directory", ""))).expanduser().resolve()
    sentiment_output = run_directory / "sentiment_appendix" / (
        "final_test" if include_test else "validation"
    )
    if sentiment_output.exists():
        raise FileExistsError(f"情绪附录产物已存在，拒绝覆盖: {sentiment_output}")
    metrics, predictions, distributions, audit = fit_sentiment_layer_probes(
        representations,
        metadata,
        final_head,
        split_config=split_cfg,
        c_grid=probe_cfg.get("c_grid", [0.01, 0.1, 1.0, 10.0]),
        positive_class_index=int(probe_cfg.get("positive_class_index", 1)),
        label_mapping_confirmed=bool(probe_cfg.get("label_mapping_confirmed", False)),
        include_test=include_test,
    )
    diagnostic = diagnose_probability_compression(
        metrics, distributions, split="validation"
    )
    return _atomic_write_probe_outputs(
        sentiment_output,
        tables={
            "sentiment_metrics.csv": metrics,
            "sentiment_oos_predictions.parquet": predictions,
            "sentiment_logit_distributions.csv": distributions,
            "probability_compression_diagnostic.csv": diagnostic,
            "sentiment_probe_audit.csv": audit,
        },
        manifest={
            "schema_version": "optional_sentiment_appendix_v2.0",
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "representation_manifest": rep_manifest,
            "include_test": include_test,
            "positive_class_index": int(probe_cfg.get("positive_class_index", 1)),
            "label_mapping_confirmed": bool(
                probe_cfg.get("label_mapping_confirmed", False)
            ),
            "selection_metric": "validation_auc",
        },
        final_test_marker_name=("FINAL_SENTIMENT_TEST_OPENED.json" if include_test else None),
        final_test_marker=(
            {
                "opened_at": datetime.now().isoformat(timespec="seconds"),
                "evaluation_split": "test",
            }
            if include_test
            else None
        ),
    )


def validate_optional_sentiment_outputs(directory: str | Path) -> dict[str, object]:
    root = Path(directory).expanduser().resolve()
    required = [
        root / "sentiment_metrics.csv",
        root / "sentiment_oos_predictions.parquet",
        root / "sentiment_logit_distributions.csv",
        root / "probability_compression_diagnostic.csv",
        root / "sentiment_probe_audit.csv",
        root / "manifest.json",
    ]
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(f"情绪附录产物缺失: {path}")
    metrics = pd.read_csv(required[0])
    predictions = pd.read_parquet(required[1])
    layer_metrics = metrics[metrics["model_kind"].eq("layer_logistic")]
    for split in layer_metrics["split"].unique():
        if set(layer_metrics.loc[layer_metrics["split"].eq(split), "layer"]) != set(
            range(13)
        ):
            raise ValueError(f"情绪附录{split}没有完整13层")
    return {"metric_rows": len(metrics), "prediction_rows": len(predictions)}


def plot_optional_sentiment_curve(
    directory: str | Path, *, split: str = "validation", metric: str = "auc"
):
    import matplotlib.pyplot as plt

    data = pd.read_csv(Path(directory) / "sentiment_metrics.csv")
    data = data[
        data["model_kind"].eq("layer_logistic") & data["split"].eq(split)
    ].sort_values("layer")
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(data["layer"], data[metric], marker="o")
    ax.set(title=f"Optional sentiment appendix ({split})", xlabel="Layer", ylabel=metric)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    return fig


def run_return_probe_stage(
    config: Mapping[str, object], evaluation_split: str = "validation"
) -> Path:
    """加载股票日representation，执行阶段4 Ridge Probe。"""

    from .layer_probe_panel import (
        STOCK_DAY_MANIFEST_FILE,
        STOCK_DAY_PANEL_FILE,
        STOCK_DAY_REPRESENTATION_FILE,
        validate_stock_day_artifacts,
    )

    output_cfg = config.get("output", {})
    probe_cfg = config.get("return_probe", {})
    returns_cfg = config.get("returns", {})
    strict_cfg = config.get("strict_test", {})
    if not all(
        isinstance(value, Mapping)
        for value in (output_cfg, probe_cfg, returns_cfg, strict_cfg)
    ):
        raise ValueError("output/return_probe/returns配置必须是对象")
    if evaluation_split not in {"validation", "test"}:
        raise ValueError("evaluation_split必须是validation或test")
    run_directory = Path(str(output_cfg.get("run_directory", ""))).expanduser().resolve()
    stock_day_directory = run_directory / "stock_day_panel"
    validate_stock_day_artifacts(
        stock_day_directory, evaluation_split=evaluation_split
    )
    representations = np.load(
        stock_day_directory / STOCK_DAY_REPRESENTATION_FILE, mmap_mode="r"
    )
    stock_day_evaluation_directory = stock_day_directory / (
        "final_test" if evaluation_split == "test" else "validation"
    )
    panel = pd.read_parquet(stock_day_evaluation_directory / STOCK_DAY_PANEL_FILE)
    canonical_manifest = json.loads(
        (stock_day_directory / STOCK_DAY_MANIFEST_FILE).read_text(encoding="utf-8")
    )
    evaluation_manifest = json.loads(
        (stock_day_evaluation_directory / STOCK_DAY_MANIFEST_FILE).read_text(
            encoding="utf-8"
        )
    )
    target_column = str(
        probe_cfg.get(
            "target_column",
            f"target_return_rank_{int(returns_cfg.get('primary_horizon', 5))}d",
        )
    )
    include_test = evaluation_split == "test"
    if include_test:
        if not bool(strict_cfg.get("open_final_test", False)):
            raise RuntimeError("收益test未通过strict_test授权")
        if not (run_directory / "FINAL_TEST_OPENED.json").is_file():
            raise RuntimeError("收益test缺少全局一次性marker")
    return_output = run_directory / "return_probe" / (
        "final_test" if include_test else "validation"
    )
    selected_factors = [str(value) for value in strict_cfg.get("selected_factors", [])]
    if include_test and not selected_factors:
        raise ValueError(
            "打开收益最终测试前必须在strict_test.selected_factors预注册因子"
        )
    frozen_alpha = None
    validation_manifest_sha = None
    if include_test:
        validation_directory = run_directory / "return_probe" / "validation"
        validation_metrics_path = validation_directory / "return_probe_metrics.csv"
        validation_manifest_path = validation_directory / "manifest.json"
        if not validation_metrics_path.is_file() or not validation_manifest_path.is_file():
            raise FileNotFoundError("收益test缺少已冻结的validation结果")
        validation_metrics = pd.read_csv(validation_metrics_path)
        validation_metrics = validation_metrics[validation_metrics["split"].eq("validation")]
        frozen_alpha = {
            int(row.layer): float(row.selected_alpha)
            for row in validation_metrics.itertuples()
        }
        from .layer_probe_representations import sha256_file

        validation_manifest_sha = sha256_file(validation_manifest_path)
    if return_output.exists():
        manifest_path = return_output / "manifest.json"
        required = [
            return_output / "return_probe_metrics.csv",
            return_output / "return_oos_predictions.parquet",
            return_output / "return_fit_reference_predictions.parquet",
            return_output / "return_probe_tuning.csv",
            manifest_path,
        ]
        if include_test:
            required.append(return_output / "FINAL_RETURN_TEST_OPENED.json")
        for path in required:
            if not path.is_file():
                raise RuntimeError(f"已存在收益Probe目录缺少文件，拒绝混读: {path}")
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected = {
            "schema_version": "return_layer_probe_v2.0",
            "config_sha256": protocol_config_hash(config),
            "target_column": target_column,
            "include_test": include_test,
            "validation_manifest_sha256": validation_manifest_sha,
        }
        mismatches = {
            key: {"actual": existing.get(key), "expected": value}
            for key, value in expected.items()
            if existing.get(key) != value
        }
        if (
            existing.get("stock_day_evaluation_manifest")
            != evaluation_manifest
        ):
            mismatches["stock_day_evaluation_manifest"] = "changed"
        if include_test:
            opened = json.loads(
                (return_output / "FINAL_RETURN_TEST_OPENED.json").read_text(
                    encoding="utf-8"
                )
            )
            if opened.get("selected_factors_preregistered") != selected_factors:
                mismatches["selected_factors_preregistered"] = "changed"
        if mismatches:
            raise RuntimeError(
                "已存在收益Probe与当前协议不匹配，拒绝覆盖或混读: "
                + json.dumps(mismatches, ensure_ascii=False, default=str)
            )
        return return_output
    metrics, predictions, tuning = fit_return_layer_probes(
        representations,
        panel,
        target_column=target_column,
        alpha_grid=probe_cfg.get("alpha_grid", [0.1, 1.0, 10.0, 100.0]),
        min_daily_observations=int(probe_cfg.get("min_daily_observations", 20)),
        include_test=include_test,
        selected_alpha_by_layer=frozen_alpha,
    )
    oos_predictions = predictions[predictions["prediction_role"].eq("oos")].copy()
    reference_predictions = predictions[
        predictions["prediction_role"].eq("fit_reference")
    ].copy()
    return _atomic_write_probe_outputs(
        return_output,
        tables={
            "return_probe_metrics.csv": metrics,
            "return_oos_predictions.parquet": oos_predictions,
            "return_fit_reference_predictions.parquet": reference_predictions,
            "return_probe_tuning.csv": tuning,
        },
        manifest={
            "schema_version": "return_layer_probe_v2.0",
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "config_sha256": protocol_config_hash(config),
            "stock_day_canonical_manifest": canonical_manifest,
            "stock_day_evaluation_manifest": evaluation_manifest,
            "target_column": target_column,
            "include_test": include_test,
            "selection_metric": "validation_daily_mean_rank_ic",
            "test_fit": "selected alpha; scaler and Ridge refit on train+validation",
            "validation_manifest_sha256": validation_manifest_sha,
        },
        final_test_marker_name=(
            "FINAL_RETURN_TEST_OPENED.json" if include_test else None
        ),
        final_test_marker=(
            {
                "opened_at": datetime.now().isoformat(timespec="seconds"),
                "evaluation_split": "test",
                "selected_factors_preregistered": selected_factors,
            }
            if include_test
            else None
        ),
    )
