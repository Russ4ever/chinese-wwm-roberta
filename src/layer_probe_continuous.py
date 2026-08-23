"""Scheme A/A+ diagnostics and continuous-label Layer Ridge probes."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

from .layer_probe_representations import (
    fingerprint_frame,
    protocol_config_hash,
    representation_artifacts,
    resolve_representation_directory,
    sha256_file,
    validate_representation_artifacts,
)


RESIDUAL_LABEL = "residual_signed_raw"
DISPERSION_LABEL = "delta_log_dispersion"
SPLITS = ("train", "validation", "test")
RESIDUAL_TASK = re.compile(r"^residual_signed_raw__fh(?P<horizon>\d+)$")
DISPERSION_TASK = re.compile(
    r"^delta_log_dispersion__(?P<months>1|3)m__"
    r"(?P<panel>fixed|market|active)__fh(?P<horizon>\d+)$"
)


def _config_hash(config: Mapping[str, object]) -> str:
    return protocol_config_hash(config)


def _run_directory(config: Mapping[str, object]) -> Path:
    output = config.get("output", {})
    if not isinstance(output, Mapping):
        raise ValueError("output配置必须是对象")
    value = str(output.get("run_directory", "")).strip()
    if not value:
        raise ValueError("output.run_directory不能为空")
    return Path(value).expanduser().resolve()


def _safe_output(path: Path) -> None:
    if path == Path(path.anchor) or len(path.parts) < 3:
        raise ValueError(f"拒绝写入过宽目录: {path}")


def _atomic_write_once(
    output: Path,
    *,
    tables: Mapping[str, pd.DataFrame],
    manifest: Mapping[str, object],
) -> Path:
    _safe_output(output)
    if output.exists():
        raise FileExistsError(f"阶段产物已存在，拒绝覆盖: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent))
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
        os.replace(temporary, output)
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    return output


def _reuse_exact_output(
    output: Path,
    *,
    expected_manifest: Mapping[str, object],
    validator,
) -> bool:
    if not output.exists():
        return False
    manifest_path = output / "manifest.json"
    if not manifest_path.is_file():
        raise RuntimeError(f"已存在阶段目录缺少manifest，拒绝覆盖或混读: {output}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    mismatches = {
        key: {"actual": manifest.get(key), "expected": expected}
        for key, expected in expected_manifest.items()
        if manifest.get(key) != expected
    }
    if mismatches:
        raise RuntimeError(
            "已存在阶段产物与当前协议不匹配，拒绝覆盖或混读: "
            + json.dumps(mismatches, ensure_ascii=False, default=str)
        )
    validator(output)
    return True


def _load_representation_bundle(
    config: Mapping[str, object],
) -> tuple[np.memmap, pd.DataFrame, pd.DataFrame, dict[str, object], Path]:
    directory = resolve_representation_directory(config)
    validate_representation_artifacts(directory)
    artifacts = representation_artifacts(directory)
    return (
        np.load(artifacts.representations, mmap_mode="r"),
        pd.read_parquet(artifacts.metadata),
        pd.read_parquet(artifacts.fixed_head_outputs),
        json.loads(artifacts.manifest.read_text(encoding="utf-8")),
        directory,
    )


def _safe_correlation(x: np.ndarray, y: np.ndarray, *, kind: str) -> float:
    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]
    if len(x) < 2 or np.unique(x).size < 2 or np.unique(y).size < 2:
        return np.nan
    result = spearmanr(x, y) if kind == "spearman" else pearsonr(x, y)
    value = result.correlation if kind == "spearman" else result.statistic
    return float(value) if np.isfinite(value) else np.nan


def _distribution_record(
    group: pd.DataFrame, layer: int, saturation_thresholds: Sequence[float]
) -> dict[str, object]:
    margin = group["logit_margin_1_minus_0"].to_numpy(dtype=float)
    p0 = group["class_0_prob"].to_numpy(dtype=float)
    p1 = group["class_1_prob"].to_numpy(dtype=float)
    entropy = -(p0 * np.log(np.clip(p0, 1e-12, 1.0)) + p1 * np.log(np.clip(p1, 1e-12, 1.0)))
    record: dict[str, object] = {
        "layer": int(layer),
        "n": int(len(group)),
        "class_0_logit_mean": float(group["class_0_logit"].mean()),
        "class_0_logit_std": float(group["class_0_logit"].std(ddof=1)),
        "class_1_logit_mean": float(group["class_1_logit"].mean()),
        "class_1_logit_std": float(group["class_1_logit"].std(ddof=1)),
        "margin_mean": float(np.mean(margin)),
        "margin_std": float(np.std(margin, ddof=1)) if len(margin) > 1 else np.nan,
        "class_0_probability_mean": float(np.mean(p0)),
        "class_0_probability_std": float(np.std(p0, ddof=1)) if len(p0) > 1 else np.nan,
        "class_1_probability_mean": float(np.mean(p1)),
        "class_1_probability_std": float(np.std(p1, ddof=1)) if len(p1) > 1 else np.nan,
        "probability_entropy_mean": float(np.mean(entropy)),
    }
    distributions = {
        "class_0_logit": group["class_0_logit"].to_numpy(dtype=float),
        "class_1_logit": group["class_1_logit"].to_numpy(dtype=float),
        "margin": margin,
        "class_0_probability": p0,
        "class_1_probability": p1,
    }
    for name, values in distributions.items():
        for quantile in (0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99):
            record[f"{name}_q{int(quantile * 100):02d}"] = float(
                np.quantile(values, quantile)
            )
    maximum = np.maximum(p0, p1)
    for threshold in saturation_thresholds:
        if not 0 < float(threshold) < 1:
            raise ValueError("Scheme A饱和阈值必须在(0,1)内")
        record[f"max_probability_ge_{int(round(float(threshold) * 100))}"] = float(
            np.mean(maximum >= threshold)
        )
    return record


def run_fixed_head_analysis_stage(config: Mapping[str, object]) -> Path:
    """Run label-free Scheme A distribution, flip and convergence diagnostics."""

    _, metadata, head, rep_manifest, _ = _load_representation_bundle(config)
    analysis_cfg = config.get("fixed_head_analysis", {})
    if not isinstance(analysis_cfg, Mapping):
        raise ValueError("fixed_head_analysis配置必须是对象")
    saturation_thresholds = [
        float(value)
        for value in analysis_cfg.get("saturation_thresholds", [0.90, 0.95, 0.99])
    ]
    output = _run_directory(config) / "fixed_head_analysis"
    if _reuse_exact_output(
        output,
        expected_manifest={
            "schema_version": "frozen_cls_fc_analysis_v2.0",
            "representation_fingerprint": rep_manifest["representation_fingerprint"],
            "config_sha256": _config_hash(config),
        },
        validator=validate_fixed_head_analysis_outputs,
    ):
        return output
    summaries = pd.DataFrame(
        [
            _distribution_record(group, int(layer), saturation_thresholds)
            for layer, group in head.groupby("layer")
        ]
    ).sort_values("layer")
    margin = head.pivot(
        index="representation_row", columns="layer", values="logit_margin_1_minus_0"
    ).sort_index()
    probability = head.pivot(
        index="representation_row", columns="layer", values="class_1_prob"
    ).sort_index()
    predicted = head.pivot(
        index="representation_row", columns="layer", values="predicted_class"
    ).sort_index()
    class_0_logit = head.pivot(
        index="representation_row", columns="layer", values="class_0_logit"
    ).sort_index()
    class_1_logit = head.pivot(
        index="representation_row", columns="layer", values="class_1_logit"
    ).sort_index()
    transitions: list[dict[str, object]] = []
    for layer in range(1, 13):
        margin_delta = margin[layer].to_numpy() - margin[layer - 1].to_numpy()
        probability_delta = probability[layer].to_numpy() - probability[layer - 1].to_numpy()
        transitions.append(
            {
                "from_layer": layer - 1,
                "to_layer": layer,
                "class_0_logit_delta_mean": float(
                    np.mean(class_0_logit[layer] - class_0_logit[layer - 1])
                ),
                "class_0_logit_delta_abs_mean": float(
                    np.mean(np.abs(class_0_logit[layer] - class_0_logit[layer - 1]))
                ),
                "class_1_logit_delta_mean": float(
                    np.mean(class_1_logit[layer] - class_1_logit[layer - 1])
                ),
                "class_1_logit_delta_abs_mean": float(
                    np.mean(np.abs(class_1_logit[layer] - class_1_logit[layer - 1]))
                ),
                "margin_delta_mean": float(np.mean(margin_delta)),
                "margin_delta_abs_mean": float(np.mean(np.abs(margin_delta))),
                "probability_delta_mean": float(np.mean(probability_delta)),
                "probability_delta_abs_mean": float(np.mean(np.abs(probability_delta))),
                "class_flip_rate": float(np.mean(predicted[layer] != predicted[layer - 1])),
            }
        )
    convergence: list[dict[str, object]] = []
    final_margin = margin[12].to_numpy(dtype=float)
    final_class = predicted[12].to_numpy(dtype=int)
    for layer in range(13):
        values = margin[layer].to_numpy(dtype=float)
        classes = predicted[layer].to_numpy(dtype=int)
        convergence.append(
            {
                "layer": layer,
                "margin_to_layer12_spearman": _safe_correlation(
                    values, final_margin, kind="spearman"
                ),
                "margin_to_layer12_pearson": _safe_correlation(
                    values, final_margin, kind="pearson"
                ),
                "margin_to_layer12_mae": float(np.mean(np.abs(values - final_margin))),
                "margin_sign_agreement_with_layer12": float(
                    np.mean(np.sign(values) == np.sign(final_margin))
                ),
                "class_agreement_with_layer12": float(np.mean(classes == final_class)),
            }
        )
    flip_matrix = predicted.to_numpy(dtype=int)[:, 1:] != predicted.to_numpy(dtype=int)[:, :-1]
    flip_count = flip_matrix.sum(axis=1)
    first_flip = np.where(flip_matrix.any(axis=1), flip_matrix.argmax(axis=1) + 1, -1)
    trajectories = metadata[["representation_row", "report_id"]].copy()
    trajectories["total_adjacent_flips"] = flip_count.astype(np.int8)
    trajectories["first_flip_to_layer"] = first_flip.astype(np.int8)
    trajectories["layer_0_class"] = predicted[0].to_numpy(dtype=np.int8)
    trajectories["layer_12_class"] = final_class.astype(np.int8)
    return _atomic_write_once(
        output,
        tables={
            "fixed_head_layer_distributions.csv": summaries,
            "fixed_head_adjacent_transitions.csv": pd.DataFrame(transitions),
            "fixed_head_convergence.csv": pd.DataFrame(convergence),
            "report_head_trajectories.parquet": trajectories,
        },
        manifest={
            "schema_version": "frozen_cls_fc_analysis_v2.0",
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "representation_fingerprint": rep_manifest["representation_fingerprint"],
            "config_sha256": _config_hash(config),
            "head_contract": rep_manifest["head_contract"],
            "interpretation_constraint": (
                "Intermediate outputs describe trajectory and distribution mismatch; "
                "they do not identify a best or most accurate layer."
            ),
        },
    )


def validate_fixed_head_analysis_outputs(directory: str | Path) -> dict[str, object]:
    root = Path(directory).expanduser().resolve()
    required = [
        root / "fixed_head_layer_distributions.csv",
        root / "fixed_head_adjacent_transitions.csv",
        root / "fixed_head_convergence.csv",
        root / "report_head_trajectories.parquet",
        root / "manifest.json",
    ]
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(f"Scheme A产物缺失: {path}")
    layers = pd.read_csv(required[0])
    convergence = pd.read_csv(required[2])
    transitions = pd.read_csv(required[1])
    if set(layers["layer"]) != set(range(13)) or set(convergence["layer"]) != set(range(13)):
        raise ValueError("Scheme A没有完整Layer 0~12")
    if len(transitions) != 12:
        raise ValueError("Scheme A相邻层变化应有12行")
    return {"layers": 13, "transitions": 12}


def _target_paths(config: Mapping[str, object]) -> tuple[Path, Path, Path]:
    target_cfg = config.get("continuous_targets", {})
    if not isinstance(target_cfg, Mapping):
        raise ValueError("continuous_targets配置必须是对象")
    directory = Path(str(target_cfg.get("bundle_directory", ""))).expanduser().resolve()
    return (
        directory / "probe_targets.parquet",
        directory / "probe_dataset_metadata.json",
        directory / "probe_merge_audit.csv",
    )


def validate_target_semantics(
    targets: pd.DataFrame, metadata: Mapping[str, object]
) -> None:
    required = {
        "report_id",
        "task_id",
        "split",
        "label_name",
        "label_value",
        "target_weight",
        "feature_available_date",
        "label_available_date",
        "forecast_horizon",
        "label_version",
    }
    missing = sorted(required.difference(targets.columns))
    if missing:
        raise ValueError("连续Target表缺少字段: " + ", ".join(missing))
    if targets.duplicated(["report_id", "task_id"]).any():
        examples = targets.loc[
            targets.duplicated(["report_id", "task_id"], keep=False),
            ["report_id", "task_id"],
        ].head()
        raise ValueError("(report_id,task_id)不唯一:\n" + examples.to_string(index=False))
    if not set(targets["split"].astype(str)).issubset(set(SPLITS)):
        raise ValueError("连续Target必须显式配置train/validation/test，不能含unassigned")
    label = pd.to_numeric(targets["label_value"], errors="coerce").to_numpy(dtype=float)
    weight = pd.to_numeric(targets["target_weight"], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(label).all() or not np.isfinite(weight).all() or (weight < 0).any():
        raise ValueError("连续Target含非有限Label或无效权重")
    feature = pd.to_datetime(targets["feature_available_date"], errors="coerce")
    available = pd.to_datetime(targets["label_available_date"], errors="coerce")
    if feature.isna().any() or available.isna().any() or (available <= feature).any():
        raise ValueError("连续Target违反label_available_date严格晚于feature的PIT约束")
    report_splits = targets.groupby("report_id")["split"].nunique()
    if report_splits.gt(1).any():
        raise ValueError("同一report_id跨越多个split")
    unknown_labels = sorted(
        set(targets["label_name"].astype(str)).difference(
            {RESIDUAL_LABEL, DISPERSION_LABEL}
        )
    )
    if unknown_labels:
        raise ValueError("连续Target含未知核心Label: " + ", ".join(unknown_labels))
    residual = targets["label_name"].eq(RESIDUAL_LABEL)
    if residual.any() and "residual_valid" not in targets:
        raise ValueError("residual_signed_raw Target缺少residual_valid")
    if residual.any() and not targets.loc[residual, "residual_valid"].eq(1).all():
        raise ValueError("residual_signed_raw主样本必须residual_valid=1")
    dispersion = targets["label_name"].eq(DISPERSION_LABEL)
    if dispersion.any():
        dispersion_missing = sorted(
            {"confirmation_months", "peer_panel"}.difference(targets.columns)
        )
        if dispersion_missing:
            raise ValueError(
                "delta_log_dispersion Target缺少任务维度: "
                + ", ".join(dispersion_missing)
            )
    if dispersion.any() and "confirmation_valid" not in targets:
        raise ValueError("delta_log_dispersion Target缺少confirmation_valid")
    if dispersion.any() and not targets.loc[dispersion, "confirmation_valid"].eq(1).all():
        raise ValueError("delta_log_dispersion主样本必须confirmation_valid=1")
    for row in targets.itertuples():
        task_id = str(row.task_id)
        horizon = int(row.forecast_horizon)
        if row.label_name == RESIDUAL_LABEL:
            match = RESIDUAL_TASK.fullmatch(task_id)
            if match is None or int(match.group("horizon")) != horizon:
                raise ValueError(f"Residual task_id与forecast_horizon不一致: {task_id}")
        else:
            match = DISPERSION_TASK.fullmatch(task_id)
            if match is None:
                raise ValueError(f"Dispersion task_id格式无效: {task_id}")
            months = int(getattr(row, "confirmation_months"))
            panel = str(getattr(row, "peer_panel"))
            if (
                int(match.group("horizon")) != horizon
                or int(match.group("months")) != months
                or match.group("panel") != panel
            ):
                raise ValueError(f"Dispersion task_id与任务维度不一致: {task_id}")
    windows = metadata.get("splits", [])
    if not isinstance(windows, list) or len(windows) != 3:
        raise ValueError("Probe bundle未记录完整的三段时间切分")
    by_name = {str(item.get("name")): item for item in windows if isinstance(item, Mapping)}
    previous_end: pd.Timestamp | None = None
    for split in SPLITS:
        item = by_name.get(split)
        if not item or not item.get("feature_start") or not item.get("feature_end"):
            raise ValueError(f"{split}缺少显式feature时间窗口")
        start = pd.Timestamp(item["feature_start"])
        end = pd.Timestamp(item["feature_end"])
        if end < start or (previous_end is not None and start <= previous_end):
            raise ValueError("连续Target时间窗口倒置或重叠")
        rows = targets["split"].eq(split)
        if not feature.loc[rows].between(start, end, inclusive="both").all():
            raise ValueError(f"{split}含不属于其feature时间窗口的样本")
        previous_end = end
    for index, split in enumerate(("train", "validation")):
        item = by_name.get(split)
        if not item or not item.get("label_cutoff"):
            raise ValueError(f"{split}缺少label_cutoff")
        cutoff = pd.Timestamp(item["label_cutoff"])
        rows = targets["split"].eq(split)
        if (available.loc[rows] > cutoff).any():
            raise ValueError(f"{split}含晚于label_cutoff才可知的Label")
        next_start = pd.Timestamp(by_name[SPLITS[index + 1]]["feature_start"])
        if cutoff >= next_start:
            raise ValueError(f"{split}.label_cutoff必须早于下一阶段开始日")


def validate_target_provenance(metadata: Mapping[str, object]) -> None:
    sources = metadata.get("sources")
    if not isinstance(sources, list):
        raise ValueError("Probe bundle metadata缺少带哈希的sources来源清单")
    indexed = {
        Path(str(item.get("path", ""))).name: item
        for item in sources
        if isinstance(item, Mapping)
    }
    required = {"report_fy_labels.parquet", "report_confirmation_labels.parquet"}
    missing = sorted(required.difference(indexed))
    if missing:
        raise ValueError("Probe bundle未证明核心Label上游来源: " + ", ".join(missing))
    try:
        from blake3 import blake3
    except ImportError as exc:
        raise RuntimeError("验证Probe bundle来源哈希需要blake3") from exc
    for filename in sorted(required):
        record = indexed[filename]
        source = Path(str(record.get("path", ""))).expanduser().resolve()
        declared = str(record.get("blake3", ""))
        if not source.is_file() or not declared:
            raise ValueError(f"Probe bundle上游来源不存在或缺少哈希: {source}")
        if int(record.get("size_bytes", -1)) != source.stat().st_size:
            raise ValueError(f"Probe bundle上游来源大小已变化: {source}")
        digest = blake3()
        digest.update_mmap(str(source))
        if digest.hexdigest() != declared:
            raise ValueError(f"Probe bundle上游来源哈希已变化: {source}")


def align_continuous_targets(
    config: Mapping[str, object], evaluation_split: str = "validation"
) -> Path:
    """Join permitted target rows without opening test labels during development."""

    if evaluation_split not in {"validation", "test"}:
        raise ValueError("evaluation_split必须是validation或test")
    if evaluation_split == "test":
        _assert_test_authorized(config)

    _, report_metadata, _, rep_manifest, _ = _load_representation_bundle(config)
    targets_path, target_manifest_path, audit_path = _target_paths(config)
    for path in (targets_path, target_manifest_path, audit_path):
        if not path.is_file():
            raise FileNotFoundError(f"连续Target bundle缺少文件: {path}")
    target_manifest = json.loads(target_manifest_path.read_text(encoding="utf-8"))
    validate_target_provenance(target_manifest)
    target_keys = pd.read_parquet(
        targets_path, columns=["report_id", "task_id", "split"]
    )
    if target_keys.duplicated(["report_id", "task_id"]).any():
        raise ValueError("完整Target bundle的(report_id,task_id)不唯一")
    if target_keys.groupby("report_id")["split"].nunique().gt(1).any():
        raise ValueError("完整Target bundle中同一report_id跨越多个split")
    permitted_splits = ["train", "validation"] if evaluation_split == "validation" else list(SPLITS)
    targets = pd.read_parquet(targets_path, filters=[("split", "in", permitted_splits)])
    validate_target_semantics(targets, target_manifest)
    aligned = targets.merge(
        report_metadata[["report_id", "representation_row", "text_sha256"]],
        on="report_id",
        how="left",
        validate="many_to_one",
        indicator=True,
    )
    if not aligned["_merge"].eq("both").all() or aligned["representation_row"].isna().any():
        missing = aligned.loc[~aligned["_merge"].eq("both"), "report_id"].head().tolist()
        raise RuntimeError(f"连续Target无法覆盖Representation report_id: {missing}")
    aligned = aligned.drop(columns="_merge")
    aligned["representation_row"] = aligned["representation_row"].astype(np.int64)
    aligned = aligned.sort_values(
        ["task_id", "split", "feature_available_date", "report_id"]
    ).reset_index(drop=True)
    output = _run_directory(config) / "continuous_targets" / (
        "validation" if evaluation_split == "validation" else "final_test"
    )
    source_fingerprint = hashlib.sha256(
        (
            sha256_file(targets_path)
            + sha256_file(target_manifest_path)
            + rep_manifest["representation_fingerprint"]
        ).encode("utf-8")
    ).hexdigest()
    if output.exists():
        manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
        if manifest.get("alignment_fingerprint") != source_fingerprint:
            raise RuntimeError("已有连续Target对齐产物与当前输入不兼容")
        validate_aligned_targets(output)
        return output
    counts = (
        aligned.groupby(["task_id", "split"], dropna=False)
        .agg(
            rows=("report_id", "size"),
            effective_rows=("target_weight", lambda x: int((x > 0).sum())),
            weight_sum=("target_weight", "sum"),
        )
        .reset_index()
    )
    source_audit = pd.read_csv(audit_path)
    source_audit.insert(0, "audit_source", "probe_bundle")
    return _atomic_write_once(
        output,
        tables={
            "aligned_probe_targets.parquet": aligned,
            "target_task_counts.csv": counts,
            "target_source_audit.csv": source_audit,
        },
        manifest={
            "schema_version": "aligned_continuous_targets_v2.0",
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "alignment_fingerprint": source_fingerprint,
            "target_source_sha256": sha256_file(targets_path),
            "target_manifest_sha256": sha256_file(target_manifest_path),
            "representation_fingerprint": rep_manifest["representation_fingerprint"],
            "config_sha256": _config_hash(config),
            "label_version": target_manifest.get("label_version"),
            "evaluation_split": evaluation_split,
            "permitted_splits": permitted_splits,
            "splits": target_manifest.get("splits"),
            "target_selection": target_manifest.get("target_selection"),
            "join_contract": "targets many-to-one report_metadata on report_id",
        },
    )


def validate_aligned_targets(directory: str | Path) -> dict[str, object]:
    root = Path(directory).expanduser().resolve()
    required = [
        root / "aligned_probe_targets.parquet",
        root / "target_task_counts.csv",
        root / "target_source_audit.csv",
        root / "manifest.json",
    ]
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(f"连续Target对齐产物缺失: {path}")
    targets = pd.read_parquet(required[0])
    if targets.duplicated(["report_id", "task_id"]).any():
        raise ValueError("对齐Target的(report_id,task_id)不唯一")
    if targets["representation_row"].isna().any():
        raise ValueError("对齐Target含空representation_row")
    return {
        "rows": len(targets),
        "tasks": int(targets["task_id"].nunique()),
        "splits": sorted(targets["split"].astype(str).unique()),
    }


def _aligned_targets(
    config: Mapping[str, object], evaluation_split: str
) -> tuple[pd.DataFrame, dict[str, object]]:
    root = align_continuous_targets(config, evaluation_split=evaluation_split)
    return (
        pd.read_parquet(root / "aligned_probe_targets.parquet"),
        json.loads((root / "manifest.json").read_text(encoding="utf-8")),
    )


def _assert_test_authorized(config: Mapping[str, object]) -> None:
    strict = config.get("strict_test", {})
    if not isinstance(strict, Mapping) or not bool(strict.get("open_final_test", False)):
        raise RuntimeError("最终test未通过strict_test.open_final_test授权")
    marker = _run_directory(config) / "FINAL_TEST_OPENED.json"
    if not marker.is_file():
        raise RuntimeError("最终test缺少一次性全局marker")


def _association_metrics(group: pd.DataFrame) -> dict[str, object]:
    x = group["logit_margin_1_minus_0"].to_numpy(dtype=float)
    y = group["label_value"].to_numpy(dtype=float)
    weights = group["target_weight"].to_numpy(dtype=float)
    direction = (x != 0) & (y != 0)
    return {
        "n": int(len(group)),
        "effective_n": int(np.sum(weights > 0)),
        "weight_sum": float(np.sum(weights)),
        "spearman": _safe_correlation(x, y, kind="spearman"),
        "pearson": _safe_correlation(x, y, kind="pearson"),
        "direction_n": int(direction.sum()),
        "zero_or_tie_n": int((~direction).sum()),
        "class_1_margin_direction_accuracy": (
            float(np.mean(np.sign(x[direction]) == np.sign(y[direction])))
            if direction.any()
            else np.nan
        ),
    }


def _bin_edges(values: np.ndarray, quantiles: int) -> np.ndarray:
    edges = np.unique(np.quantile(values, np.linspace(0, 1, quantiles + 1)))
    if len(edges) < 3:
        return np.array([], dtype=float)
    edges[0] = -np.inf
    edges[-1] = np.inf
    return edges


def run_fixed_head_label_stage(
    config: Mapping[str, object], evaluation_split: str = "validation"
) -> Path:
    """Run no-training Scheme A+ against each continuous task independently."""

    if evaluation_split not in {"validation", "test"}:
        raise ValueError("evaluation_split必须是validation或test")
    if evaluation_split == "test":
        _assert_test_authorized(config)
    _, _, head, rep_manifest, _ = _load_representation_bundle(config)
    targets, target_manifest = _aligned_targets(config, evaluation_split)
    output = _run_directory(config) / "fixed_head_label" / (
        "validation" if evaluation_split == "validation" else "final_test"
    )
    if _reuse_exact_output(
        output,
        expected_manifest={
            "schema_version": "fixed_head_continuous_association_v2.0",
            "evaluation_split": evaluation_split,
            "representation_fingerprint": rep_manifest["representation_fingerprint"],
            "config_sha256": _config_hash(config),
            "target_alignment_fingerprint": target_manifest["alignment_fingerprint"],
        },
        validator=validate_fixed_head_label_outputs,
    ):
        return output
    reference_splits = ["train"] if evaluation_split == "validation" else ["train", "validation"]
    in_scope = targets["split"].isin([*reference_splits, evaluation_split])
    positive_weight = pd.to_numeric(targets["target_weight"], errors="coerce").gt(0)
    zero_weight_count = int((in_scope & ~positive_weight).sum())
    selected = targets[in_scope].copy()
    merged = selected.merge(head, on="representation_row", how="inner", validate="many_to_many")
    if len(merged) != len(selected) * 13:
        raise RuntimeError("Scheme A+ Target与13层固定头连接不完整")
    metrics: list[dict[str, object]] = []
    groups: list[dict[str, object]] = []
    edges_records: list[dict[str, object]] = []
    audit: list[dict[str, object]] = [
        {
            "task_id": "all",
            "layer": pd.NA,
            "reason": "zero_weight_retained_for_unweighted_association",
            "count": zero_weight_count,
        }
    ]
    quantiles = int(
        (config.get("fixed_head_label_analysis", {}) or {}).get("quantiles", 5)
    )
    for (task_id, layer), task_layer in merged.groupby(["task_id", "layer"], sort=True):
        reference = task_layer[task_layer["split"].isin(reference_splits)]
        evaluation = task_layer[task_layer["split"].eq(evaluation_split)]
        if reference.empty or evaluation.empty:
            audit.append(
                {"task_id": task_id, "layer": layer, "reason": "missing_reference_or_evaluation"}
            )
            continue
        for role, frame in (("fit_reference", reference), ("oos", evaluation)):
            dimensions = {
                column: frame[column].iloc[0] if column in frame else pd.NA
                for column in (
                    "forecast_horizon",
                    "confirmation_months",
                    "peer_panel",
                )
            }
            metrics.append(
                {
                    "task_id": task_id,
                    "label_name": frame["label_name"].iloc[0],
                    "layer": int(layer),
                    "split": "+".join(reference_splits) if role == "fit_reference" else evaluation_split,
                    "prediction_role": role,
                    **dimensions,
                    **_association_metrics(frame),
                }
            )
        edges = _bin_edges(reference["logit_margin_1_minus_0"].to_numpy(dtype=float), quantiles)
        if not len(edges):
            audit.append({"task_id": task_id, "layer": layer, "reason": "insufficient_margin_bins"})
            continue
        for edge_index, edge in enumerate(edges):
            edges_records.append(
                {"task_id": task_id, "layer": int(layer), "edge_index": edge_index, "edge": edge}
            )
        for role, frame in (("fit_reference", reference.copy()), ("oos", evaluation.copy())):
            frame["margin_group"] = np.digitize(
                frame["logit_margin_1_minus_0"].to_numpy(dtype=float), edges[1:-1], right=True
            ) + 1
            for group_id, values in frame.groupby("margin_group"):
                weights = values["target_weight"].to_numpy(dtype=float)
                labels = values["label_value"].to_numpy(dtype=float)
                groups.append(
                    {
                        "task_id": task_id,
                        "label_name": values["label_name"].iloc[0],
                        "layer": int(layer),
                        "split": "+".join(reference_splits) if role == "fit_reference" else evaluation_split,
                        "prediction_role": role,
                        "margin_group": int(group_id),
                        "n": int(len(values)),
                        "label_mean": float(np.mean(labels)),
                        "weighted_label_mean": (
                            float(np.average(labels, weights=weights))
                            if np.sum(weights) > 0
                            else np.nan
                        ),
                    }
                )
    metric_frame = pd.DataFrame(metrics)
    if metric_frame.empty:
        raise ValueError(f"Scheme A+没有可评价的{evaluation_split}任务")
    stability = (
        metric_frame[metric_frame["prediction_role"].eq("oos")]
        .groupby(["label_name", "layer"])
        .agg(
            tasks=("task_id", "nunique"),
            mean_task_spearman=("spearman", "mean"),
            std_task_spearman=("spearman", "std"),
            mean_task_pearson=("pearson", "mean"),
            mean_direction_accuracy=("class_1_margin_direction_accuracy", "mean"),
        )
        .reset_index()
    )
    return _atomic_write_once(
        output,
        tables={
            "fixed_head_label_metrics.csv": metric_frame,
            "fixed_head_label_groups.parquet": pd.DataFrame(groups),
            "fixed_head_label_bin_edges.csv": pd.DataFrame(edges_records),
            "fixed_head_label_stability.csv": stability,
            "fixed_head_label_audit.csv": pd.DataFrame(audit),
        },
        manifest={
            "schema_version": "fixed_head_continuous_association_v2.0",
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "evaluation_split": evaluation_split,
            "reference_splits": reference_splits,
            "representation_fingerprint": rep_manifest["representation_fingerprint"],
            "config_sha256": _config_hash(config),
            "target_alignment_fingerprint": target_manifest["alignment_fingerprint"],
            "head_contract": rep_manifest["head_contract"],
            "training": "none",
            "group_boundaries": f"{quantiles} quantiles fitted on reference head outputs only",
            "weight_contract": (
                "unweighted correlations retain zero-weight rows; effective_n and "
                "weight_sum are reported; weighted group means use target_weight"
            ),
            "label_semantics": {
                "residual_signed_raw": "signed forecast residual",
                "delta_log_dispersion": {
                    "positive": "dispersion expansion",
                    "negative": "dispersion contraction",
                },
            },
        },
    )


def validate_fixed_head_label_outputs(directory: str | Path) -> dict[str, object]:
    root = Path(directory).expanduser().resolve()
    required = [
        root / "fixed_head_label_metrics.csv",
        root / "fixed_head_label_groups.parquet",
        root / "fixed_head_label_bin_edges.csv",
        root / "fixed_head_label_stability.csv",
        root / "manifest.json",
    ]
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(f"Scheme A+产物缺失: {path}")
    metrics = pd.read_csv(required[0])
    oos = metrics[metrics["prediction_role"].eq("oos")]
    incomplete = oos.groupby("task_id")["layer"].agg(lambda x: set(x) != set(range(13)))
    if incomplete.any():
        raise ValueError("Scheme A+部分任务没有完整13层")
    return {"metric_rows": len(metrics), "tasks": int(oos["task_id"].nunique())}


def _regression_metrics(
    y: np.ndarray, prediction: np.ndarray, weights: np.ndarray
) -> dict[str, object]:
    direction = (y != 0) & (prediction != 0)
    return {
        "n": int(len(y)),
        "weight_sum": float(np.sum(weights)),
        "spearman": _safe_correlation(prediction, y, kind="spearman"),
        "pearson": _safe_correlation(prediction, y, kind="pearson"),
        "r2": float(r2_score(y, prediction)),
        "mae": float(mean_absolute_error(y, prediction)),
        "rmse": float(np.sqrt(mean_squared_error(y, prediction))),
        "weighted_r2": float(r2_score(y, prediction, sample_weight=weights)),
        "weighted_mae": float(mean_absolute_error(y, prediction, sample_weight=weights)),
        "weighted_rmse": float(
            np.sqrt(mean_squared_error(y, prediction, sample_weight=weights))
        ),
        "sign_n": int(direction.sum()),
        "sign_accuracy": (
            float(np.mean(np.sign(y[direction]) == np.sign(prediction[direction])))
            if direction.any()
            else np.nan
        ),
    }


def _task_rows(targets: pd.DataFrame, task_id: str, splits: Sequence[str]) -> pd.DataFrame:
    rows = targets[targets["task_id"].eq(task_id) & targets["split"].isin(splits)].copy()
    rows["label_value"] = pd.to_numeric(rows["label_value"], errors="coerce")
    rows["target_weight"] = pd.to_numeric(rows["target_weight"], errors="coerce")
    return rows[
        np.isfinite(rows["label_value"])
        & np.isfinite(rows["target_weight"])
        & rows["target_weight"].gt(0)
    ]


def run_continuous_probe_stage(
    config: Mapping[str, object], evaluation_split: str = "validation"
) -> Path:
    """Fit one weighted StandardScaler+Ridge per task and layer."""

    if evaluation_split not in {"validation", "test"}:
        raise ValueError("evaluation_split必须是validation或test")
    if evaluation_split == "test":
        _assert_test_authorized(config)
    representations, _, _, rep_manifest, _ = _load_representation_bundle(config)
    targets, target_manifest = _aligned_targets(config, evaluation_split)
    probe_cfg = config.get("continuous_probe", {})
    if not isinstance(probe_cfg, Mapping):
        raise ValueError("continuous_probe配置必须是对象")
    alphas = sorted({float(value) for value in probe_cfg.get("alpha_grid", [0.1, 1, 10, 100, 1000])})
    if not alphas or min(alphas) < 0:
        raise ValueError("Ridge alpha网格必须是非负数")
    output = _run_directory(config) / "continuous_probe" / (
        "validation" if evaluation_split == "validation" else "final_test"
    )
    if _reuse_exact_output(
        output,
        expected_manifest={
            "schema_version": "continuous_label_ridge_probe_v2.0",
            "evaluation_split": evaluation_split,
            "representation_fingerprint": rep_manifest["representation_fingerprint"],
            "config_sha256": _config_hash(config),
            "target_alignment_fingerprint": target_manifest["alignment_fingerprint"],
        },
        validator=validate_continuous_probe_outputs,
    ):
        return output

    selected_alpha: dict[tuple[str, int], float] = {}
    validation_manifest_sha: str | None = None
    if evaluation_split == "test":
        validation_dir = _run_directory(config) / "continuous_probe" / "validation"
        validate_continuous_probe_outputs(validation_dir)
        validation_manifest_sha = sha256_file(validation_dir / "manifest.json")
        validation_metrics = pd.read_csv(validation_dir / "continuous_probe_metrics.csv")
        validation_metrics = validation_metrics[
            validation_metrics["prediction_role"].eq("oos")
        ]
        selected_alpha = {
            (str(row.task_id), int(row.layer)): float(row.selected_alpha)
            for row in validation_metrics.itertuples()
        }

    metrics: list[dict[str, object]] = []
    oos_predictions: list[pd.DataFrame] = []
    reference_predictions: list[pd.DataFrame] = []
    tuning: list[dict[str, object]] = []
    eligibility: list[dict[str, object]] = []
    for task_id in sorted(targets["task_id"].astype(str).unique()):
        train_splits = ["train"] if evaluation_split == "validation" else ["train", "validation"]
        for scope, scope_splits in (
            ("training", train_splits),
            ("evaluation", [evaluation_split]),
        ):
            raw = targets[
                targets["task_id"].eq(task_id)
                & targets["split"].isin(scope_splits)
            ]
            raw_label = pd.to_numeric(raw["label_value"], errors="coerce")
            raw_weight = pd.to_numeric(raw["target_weight"], errors="coerce")
            exclusions = {
                "nonfinite_label": int((~np.isfinite(raw_label)).sum()),
                "nonfinite_weight": int((~np.isfinite(raw_weight)).sum()),
                "nonpositive_weight": int(
                    (np.isfinite(raw_weight) & raw_weight.le(0)).sum()
                ),
            }
            for reason_name, count in exclusions.items():
                eligibility.append(
                    {
                        "record_type": "exclusion_audit",
                        "task_id": task_id,
                        "evaluation_split": evaluation_split,
                        "scope": scope,
                        "reason": reason_name,
                        "count": count,
                    }
                )
        train = _task_rows(targets, task_id, train_splits)
        evaluation = _task_rows(targets, task_id, [evaluation_split])
        reason = None
        if train.empty or evaluation.empty:
            reason = "missing_positive_weight_train_or_evaluation"
        elif train["label_value"].nunique() < 2:
            reason = "constant_training_target"
        elif evaluation["label_value"].nunique() < 2:
            reason = "constant_evaluation_target"
        eligibility.append(
            {
                "record_type": "task_eligibility",
                "task_id": task_id,
                "evaluation_split": evaluation_split,
                "training_rows": len(train),
                "evaluation_rows": len(evaluation),
                "eligible": reason is None,
                "reason": reason,
            }
        )
        if reason is not None:
            continue
        train_positions = train["representation_row"].to_numpy(dtype=int)
        evaluation_positions = evaluation["representation_row"].to_numpy(dtype=int)
        y_train = train["label_value"].to_numpy(dtype=float)
        y_evaluation = evaluation["label_value"].to_numpy(dtype=float)
        w_train = train["target_weight"].to_numpy(dtype=float)
        w_evaluation = evaluation["target_weight"].to_numpy(dtype=float)
        for layer in range(13):
            x_train = np.asarray(representations[train_positions, layer, :], dtype=np.float32)
            x_evaluation = np.asarray(
                representations[evaluation_positions, layer, :], dtype=np.float32
            )
            scaler = StandardScaler().fit(x_train, sample_weight=w_train)
            x_train_scaled = scaler.transform(x_train)
            x_evaluation_scaled = scaler.transform(x_evaluation)
            if evaluation_split == "validation":
                best_score = -np.inf
                best_alpha: float | None = None
                for alpha in alphas:
                    candidate = Ridge(alpha=alpha).fit(
                        x_train_scaled, y_train, sample_weight=w_train
                    )
                    candidate_prediction = candidate.predict(x_evaluation_scaled)
                    score = _safe_correlation(
                        candidate_prediction, y_evaluation, kind="spearman"
                    )
                    tuning.append(
                        {
                            "task_id": task_id,
                            "layer": layer,
                            "alpha": alpha,
                            "validation_spearman": score,
                        }
                    )
                    comparable = -np.inf if not np.isfinite(score) else float(score)
                    if comparable > best_score or (
                        np.isclose(comparable, best_score) and (best_alpha is None or alpha > best_alpha)
                    ):
                        best_score = comparable
                        best_alpha = alpha
                if best_alpha is None or not np.isfinite(best_score):
                    eligibility.append(
                        {
                            "record_type": "layer_eligibility",
                            "task_id": task_id,
                            "evaluation_split": evaluation_split,
                            "layer": layer,
                            "eligible": False,
                            "reason": "undefined_validation_spearman",
                        }
                    )
                    continue
            else:
                best_alpha = selected_alpha.get((task_id, layer))
                if best_alpha is None:
                    raise RuntimeError(f"test缺少validation选定alpha: {task_id}, layer={layer}")
            model = Ridge(alpha=float(best_alpha)).fit(
                x_train_scaled, y_train, sample_weight=w_train
            )
            train_prediction = model.predict(x_train_scaled)
            evaluation_prediction = model.predict(x_evaluation_scaled)
            metrics.append(
                {
                    "task_id": task_id,
                    "label_name": train["label_name"].iloc[0],
                    "layer": layer,
                    "split": "+".join(train_splits),
                    "prediction_role": "fit_reference",
                    "selected_alpha": best_alpha,
                    **_regression_metrics(y_train, train_prediction, w_train),
                }
            )
            metrics.append(
                {
                    "task_id": task_id,
                    "label_name": evaluation["label_name"].iloc[0],
                    "layer": layer,
                    "split": evaluation_split,
                    "prediction_role": "oos",
                    "selected_alpha": best_alpha,
                    **_regression_metrics(y_evaluation, evaluation_prediction, w_evaluation),
                }
            )
            ref = train[
                ["report_id", "representation_row", "task_id", "split", "label_value", "target_weight"]
            ].copy()
            ref["layer"] = layer
            ref["prediction"] = train_prediction
            ref["prediction_role"] = "fit_reference"
            ref["evaluation_split"] = evaluation_split
            reference_predictions.append(ref)
            pred = evaluation[
                ["report_id", "representation_row", "task_id", "split", "label_value", "target_weight"]
            ].copy()
            pred["layer"] = layer
            pred["prediction"] = evaluation_prediction
            pred["prediction_role"] = "oos"
            pred["evaluation_split"] = evaluation_split
            oos_predictions.append(pred)
    if not oos_predictions:
        raise ValueError(f"没有任何任务可生成{evaluation_split}连续Label OOS预测")
    metric_frame = pd.DataFrame(metrics)
    return _atomic_write_once(
        output,
        tables={
            "continuous_probe_metrics.csv": metric_frame,
            "continuous_probe_oos_predictions.parquet": pd.concat(oos_predictions, ignore_index=True),
            "continuous_probe_fit_reference.parquet": pd.concat(reference_predictions, ignore_index=True),
            "continuous_probe_tuning.csv": pd.DataFrame(tuning),
            "continuous_probe_eligibility.csv": pd.DataFrame(eligibility),
        },
        manifest={
            "schema_version": "continuous_label_ridge_probe_v2.0",
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "evaluation_split": evaluation_split,
            "training_splits": ["train"] if evaluation_split == "validation" else ["train", "validation"],
            "representation_fingerprint": rep_manifest["representation_fingerprint"],
            "config_sha256": _config_hash(config),
            "target_alignment_fingerprint": target_manifest["alignment_fingerprint"],
            "alpha_grid": alphas,
            "selection_metric": "validation_spearman",
            "alpha_tie_break": "larger_alpha",
            "sample_weight": "used by StandardScaler and Ridge; zero-weight excluded",
            "target_transform": "none",
            "validation_manifest_sha256": validation_manifest_sha,
            "oos_contract": (
                "validation=train fit; test=train+validation refit using frozen validation alpha"
            ),
        },
    )


def validate_continuous_probe_outputs(directory: str | Path) -> dict[str, object]:
    root = Path(directory).expanduser().resolve()
    required = [
        root / "continuous_probe_metrics.csv",
        root / "continuous_probe_oos_predictions.parquet",
        root / "continuous_probe_fit_reference.parquet",
        root / "continuous_probe_tuning.csv",
        root / "continuous_probe_eligibility.csv",
        root / "manifest.json",
    ]
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(f"连续Label Probe产物缺失: {path}")
    metrics = pd.read_csv(required[0])
    oos = pd.read_parquet(required[1])
    reference = pd.read_parquet(required[2])
    if not oos["prediction_role"].eq("oos").all():
        raise ValueError("continuous_probe_oos_predictions混入fit-reference")
    if not reference["prediction_role"].eq("fit_reference").all():
        raise ValueError("continuous_probe_fit_reference混入OOS")
    oos_metrics = metrics[metrics["prediction_role"].eq("oos")]
    incomplete = oos_metrics.groupby("task_id")["layer"].agg(
        lambda values: set(values) != set(range(13))
    )
    if incomplete.any():
        raise ValueError("连续Label Probe部分已评价任务没有完整13层")
    if oos["prediction"].isna().any():
        raise ValueError("连续Label OOS预测含NaN")
    return {
        "metric_rows": len(metrics),
        "oos_rows": len(oos),
        "reference_rows": len(reference),
        "tasks": int(oos["task_id"].nunique()),
    }


def plot_fixed_head_convergence(directory: str | Path):
    import matplotlib.pyplot as plt

    data = pd.read_csv(Path(directory) / "fixed_head_convergence.csv")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(data["layer"], data["margin_to_layer12_spearman"], marker="o")
    axes[0].set(title="Margin convergence to Layer 12", xlabel="Layer", ylabel="Spearman")
    axes[1].plot(data["layer"], data["class_agreement_with_layer12"], marker="o")
    axes[1].set(title="Class convergence to Layer 12", xlabel="Layer", ylabel="Agreement")
    for axis in axes:
        axis.grid(alpha=0.2)
    fig.tight_layout()
    return fig


def plot_aligned_target_counts(directory: str | Path):
    import matplotlib.pyplot as plt

    data = pd.read_csv(Path(directory) / "target_task_counts.csv")
    pivot = data.pivot(index="task_id", columns="split", values="effective_rows").fillna(0)
    fig, ax = plt.subplots(figsize=(9, max(4, 0.35 * len(pivot))))
    pivot.plot.barh(ax=ax)
    ax.set(title="Continuous target eligibility", xlabel="Positive-weight rows")
    fig.tight_layout()
    return fig


def plot_fixed_head_label_associations(
    directory: str | Path, *, metric: str = "spearman"
):
    import matplotlib.pyplot as plt

    data = pd.read_csv(Path(directory) / "fixed_head_label_metrics.csv")
    data = data[data["prediction_role"].eq("oos")]
    tasks = sorted(data["task_id"].unique())
    fig, axes = plt.subplots(
        len(tasks), 1, figsize=(9, max(4, 2.6 * len(tasks))), squeeze=False
    )
    for axis, task in zip(axes[:, 0], tasks):
        selected = data[data["task_id"].eq(task)].sort_values("layer")
        axis.plot(selected["layer"], selected[metric], marker="o")
        axis.axhline(0, color="black", linewidth=0.8)
        axis.set(title=task, xlabel="Layer", ylabel=metric)
        axis.grid(alpha=0.2)
    fig.tight_layout()
    return fig


def plot_continuous_probe_curves(directory: str | Path, *, metric: str = "spearman"):
    import matplotlib.pyplot as plt

    data = pd.read_csv(Path(directory) / "continuous_probe_metrics.csv")
    data = data[data["prediction_role"].eq("oos")]
    tasks = sorted(data["task_id"].unique())
    fig, axes = plt.subplots(len(tasks), 1, figsize=(9, max(4, 2.6 * len(tasks))), squeeze=False)
    for axis, task in zip(axes[:, 0], tasks):
        selected = data[data["task_id"].eq(task)].sort_values("layer")
        axis.plot(selected["layer"], selected[metric], marker="o")
        axis.axhline(0, color="black", linewidth=0.8)
        axis.set(title=task, xlabel="Layer", ylabel=metric)
        axis.grid(alpha=0.2)
    fig.tight_layout()
    return fig
