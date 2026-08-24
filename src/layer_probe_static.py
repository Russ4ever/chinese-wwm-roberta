"""Static FY0 report-level probes and CSI300-only return evaluation.

This module is an explicitly conflicting alternative to the strict PIT
walk-forward protocol.  It keeps the canonical continuous-label semantics and
chronological feature splits, but deliberately does not use
``label_available_date`` to filter split membership.  All trainable probes use
Layer 1 through Layer 12; Layer 0 remains available only to the frozen-head
descriptive stage.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from .layer_probe_continuous import (
    _atomic_write_once,
    _load_representation_bundle,
    _reuse_exact_output,
    _run_directory,
    validate_target_provenance,
)
from .layer_probe_representations import (
    configure_layer_probe_runtime,
    gpu_runtime_audit,
    protocol_config_hash,
    sha256_file,
)
from .layer_probe_walk_forward import (
    _target_paths,
    _validate_unsplit_targets,
)
from .report_label_economics import (
    canonical_stock_code,
    compound_forward,
    extract_forward_values,
    normalize_csi_weights,
    read_return_panel,
    validate_decimal_returns,
)


STATIC_SCHEMA = "static_fy0_report_probe_v1.0"
STATIC_TARGET_SCHEMA = "static_fy0_continuous_targets_v1.0"
DIRECT_TARGET_SCHEMA = "static_report_direct_return_target_v1.0"
REPORT_PROBE_SCHEMA = "static_report_ridge_probe_v1.0"
CSI_EVALUATION_SCHEMA = "static_csi300_rank_ic_v1.0"

STATIC_TASKS = (
    "residual_signed_raw__fh0",
    "delta_log_dispersion__1m__fixed__fh0",
    "delta_log_dispersion__1m__market__fh0",
    "delta_log_dispersion__3m__fixed__fh0",
    "delta_log_dispersion__3m__market__fh0",
)
DIRECT_TASK_ID = "direct_return_rank_5d"
MODELED_LAYERS = tuple(range(1, 13))
DESCRIPTIVE_LAYERS = tuple(range(13))
SPLIT_NAMES = ("train", "validation", "test")


@dataclass(frozen=True)
class StaticWindow:
    name: str
    start: pd.Timestamp
    end: pd.Timestamp

    def to_dict(self) -> dict[str, str]:
        return {
            "name": self.name,
            "start": str(self.start.date()),
            "end": str(self.end.date()),
        }


@dataclass(frozen=True)
class StaticProtocol:
    train: StaticWindow
    validation: StaticWindow
    test: StaticWindow
    task_ids: tuple[str, ...]
    modeled_layers: tuple[int, ...]
    descriptive_layers: tuple[int, ...]

    @property
    def windows(self) -> tuple[StaticWindow, ...]:
        return (self.train, self.validation, self.test)

    def to_dict(self) -> dict[str, object]:
        return {
            "train": self.train.to_dict(),
            "validation": self.validation.to_dict(),
            "test": self.test.to_dict(),
            "task_ids": list(self.task_ids),
            "modeled_layers": list(self.modeled_layers),
            "descriptive_layers": list(self.descriptive_layers),
            "pit_label_availability_enforced": False,
            "continuous_label_embargo_days": 0,
            "test_opened_in_same_run": True,
            "protocol_class": "conflicting_alternative",
        }


def _date(value: object, *, name: str) -> pd.Timestamp:
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        raise ValueError(f"{name}不是有效日期: {value!r}")
    return pd.Timestamp(parsed).normalize()


def _parse_window(raw: object, *, name: str) -> StaticWindow:
    if not isinstance(raw, Mapping):
        raise ValueError(f"static_protocol.{name}必须是对象")
    start = _date(raw.get("start"), name=f"static_protocol.{name}.start")
    end = _date(raw.get("end"), name=f"static_protocol.{name}.end")
    if end < start:
        raise ValueError(f"static_protocol.{name}日期倒置")
    return StaticWindow(name, start, end)


def parse_static_protocol(config: Mapping[str, object]) -> StaticProtocol:
    raw = config.get("static_protocol", {})
    targets = config.get("continuous_targets", {})
    if not isinstance(raw, Mapping) or not bool(raw.get("enabled", False)):
        raise ValueError("static_protocol.enabled必须为true")
    if not isinstance(targets, Mapping):
        raise ValueError("continuous_targets配置必须是对象")
    if bool(raw.get("pit_label_availability_enforced", True)):
        raise ValueError("本替代协议必须显式配置pit_label_availability_enforced=false")
    if int(raw.get("continuous_label_embargo_days", -1)) != 0:
        raise ValueError("本替代协议的连续Label embargo必须为0")
    if not bool(raw.get("test_opened_in_same_run", False)):
        raise ValueError("本替代协议必须显式记录test_opened_in_same_run=true")
    train = _parse_window(raw.get("train"), name="train")
    validation = _parse_window(raw.get("validation"), name="validation")
    test = _parse_window(raw.get("test"), name="test")
    if not train.end < validation.start <= validation.end < test.start:
        raise ValueError("静态特征窗口必须满足train < validation < test")
    task_ids = tuple(str(value) for value in targets.get("task_ids", []))
    if set(task_ids) != set(STATIC_TASKS) or len(task_ids) != len(STATIC_TASKS):
        raise ValueError("静态协议任务必须严格等于5个FY0 fixed/market任务")
    modeled = tuple(int(value) for value in raw.get("modeled_layers", []))
    descriptive = tuple(int(value) for value in raw.get("descriptive_layers", []))
    if modeled != MODELED_LAYERS:
        raise ValueError("静态Ridge层必须严格为Layer 1~12")
    if descriptive != DESCRIPTIVE_LAYERS:
        raise ValueError("描述性层必须严格为Layer 0~12")
    return StaticProtocol(
        train=train,
        validation=validation,
        test=test,
        task_ids=task_ids,
        modeled_layers=modeled,
        descriptive_layers=descriptive,
    )


def assign_static_splits(
    dates: pd.Series | Iterable[object], protocol: StaticProtocol
) -> pd.Series:
    values = pd.to_datetime(pd.Series(dates), errors="coerce").dt.normalize()
    if values.isna().any():
        raise ValueError("静态切分日期含无效值")
    result = pd.Series(pd.NA, index=values.index, dtype="string")
    for window in protocol.windows:
        mask = values.between(window.start, window.end, inclusive="both")
        result.loc[mask] = window.name
    return result


def _safe_stage_directory(output: Path) -> None:
    if output == Path(output.anchor) or len(output.parts) < 3:
        raise ValueError(f"拒绝写入过宽目录: {output}")


def _atomic_directory(output: Path, writer) -> Path:
    _safe_stage_directory(output)
    if output.exists():
        raise FileExistsError(f"阶段产物已存在，拒绝覆盖: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent))
    try:
        writer(temporary)
        os.replace(temporary, output)
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    return output


def _stat_identity(path: Path) -> dict[str, object]:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def _output_disk_budget(
    config: Mapping[str, object], *, estimated_peak_bytes: int
) -> dict[str, int]:
    performance = config.get("performance", {})
    if not isinstance(performance, Mapping):
        raise ValueError("performance配置必须是对象")
    run = _run_directory(config)
    parent = run.parent
    parent.mkdir(parents=True, exist_ok=True)
    usage = shutil.disk_usage(parent)
    minimum_gb = float(performance.get("minimum_free_disk_gb", 15))
    minimum_fraction = float(performance.get("minimum_free_disk_fraction", 0.20))
    if minimum_gb <= 0 or not 0 < minimum_fraction < 1:
        raise ValueError("磁盘保留阈值配置无效")
    reserve = max(int(minimum_gb * (1 << 30)), int(usage.free * minimum_fraction))
    if int(estimated_peak_bytes) > usage.free - reserve:
        raise RuntimeError(
            "静态Pipeline磁盘预算不足: "
            f"estimated_peak={estimated_peak_bytes}, free={usage.free}, reserve={reserve}"
        )
    return {
        "estimated_peak_bytes": int(estimated_peak_bytes),
        "free_bytes_at_preflight": int(usage.free),
        "required_reserve_bytes": int(reserve),
    }


def _static_target_output(config: Mapping[str, object]) -> Path:
    return _run_directory(config) / "static_targets"


def run_static_target_stage(config: Mapping[str, object]) -> Path:
    """Select five FY0 tasks and split only by feature date.

    ``label_available_date`` remains in every output row but is deliberately
    absent from the membership masks.
    """

    protocol = parse_static_protocol(config)
    _, report_metadata, _, rep_manifest, _ = _load_representation_bundle(config)
    targets_path, metadata_path, audit_path = _target_paths(config)
    for path in (targets_path, metadata_path, audit_path):
        if not path.is_file():
            raise FileNotFoundError(f"静态Target bundle缺少文件: {path}")
    target_manifest = json.loads(metadata_path.read_text(encoding="utf-8"))
    output = _static_target_output(config)
    source_stat = _stat_identity(targets_path)
    metadata_stat = _stat_identity(metadata_path)
    if output.exists():
        existing = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
        fast_expected = {
            "schema_version": STATIC_TARGET_SCHEMA,
            "representation_fingerprint": rep_manifest["representation_fingerprint"],
            "config_sha256": protocol_config_hash(config),
            "target_source_stat": source_stat,
            "target_manifest_stat": metadata_stat,
        }
        if all(existing.get(key) == value for key, value in fast_expected.items()):
            validate_static_target_outputs(output)
            return output
    target_sha256 = sha256_file(targets_path)
    target_manifest_sha256 = sha256_file(metadata_path)
    source_fingerprint = hashlib.sha256(
        (
            target_sha256
            + target_manifest_sha256
            + rep_manifest["representation_fingerprint"]
            + json.dumps(protocol.to_dict(), sort_keys=True)
        ).encode("utf-8")
    ).hexdigest()
    expected = {
        "schema_version": STATIC_TARGET_SCHEMA,
        "alignment_fingerprint": source_fingerprint,
        "representation_fingerprint": rep_manifest["representation_fingerprint"],
        "config_sha256": protocol_config_hash(config),
    }
    if _reuse_exact_output(
        output,
        expected_manifest=expected,
        validator=validate_static_target_outputs,
    ):
        return output
    validate_target_provenance(target_manifest)
    targets = pd.read_parquet(
        targets_path,
        filters=[
            ("task_id", "in", list(protocol.task_ids)),
            ("feature_available_date", ">=", protocol.train.start),
            ("feature_available_date", "<=", protocol.test.end),
        ],
    )
    _validate_unsplit_targets(targets, target_manifest)
    targets = targets[targets["task_id"].astype(str).isin(protocol.task_ids)].copy()
    weights = pd.to_numeric(targets["target_weight"], errors="coerce")
    labels = pd.to_numeric(targets["label_value"], errors="coerce")
    targets = targets[
        weights.gt(0) & np.isfinite(weights) & np.isfinite(labels)
    ].copy()
    targets["split"] = assign_static_splits(
        targets["feature_available_date"], protocol
    ).to_numpy()
    targets = targets[targets["split"].notna()].copy()
    actual_tasks = set(targets["task_id"].astype(str).unique())
    if actual_tasks != set(protocol.task_ids):
        raise ValueError(
            "静态Target没有覆盖全部5任务: "
            f"missing={sorted(set(protocol.task_ids).difference(actual_tasks))}"
        )
    aligned = targets.merge(
        report_metadata[
            [
                "report_id",
                "representation_row",
                "symbol",
                "feature_available_date",
                "text_sha256",
            ]
        ],
        on=["report_id", "feature_available_date"],
        how="left",
        validate="many_to_one",
        indicator=True,
        suffixes=("", "_representation"),
    )
    if not aligned["_merge"].eq("both").all() or aligned[
        "representation_row"
    ].isna().any():
        missing = aligned.loc[~aligned["_merge"].eq("both"), "report_id"].head()
        raise RuntimeError(
            "静态Target无法覆盖representation: " + ",".join(missing.astype(str))
        )
    aligned = aligned.drop(columns="_merge")
    aligned["representation_row"] = aligned["representation_row"].astype(np.int64)
    aligned = aligned.sort_values(
        ["task_id", "split", "feature_available_date", "report_id"]
    ).reset_index(drop=True)
    counts = (
        aligned.groupby(["task_id", "split"], sort=True)
        .agg(
            rows=("report_id", "size"),
            reports=("report_id", "nunique"),
            weight_sum=("target_weight", "sum"),
            label_available_date_max=("label_available_date", "max"),
        )
        .reset_index()
    )
    if set(counts["split"].astype(str)) != set(SPLIT_NAMES):
        raise ValueError("静态Target缺少train/validation/test分区")
    source_audit = pd.read_csv(audit_path)
    source_audit.insert(0, "audit_source", "probe_bundle")
    return _atomic_write_once(
        output,
        tables={
            "aligned_static_targets.parquet": aligned,
            "static_target_counts.csv": counts,
            "target_source_audit.csv": source_audit,
        },
        manifest={
            **expected,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "protocol": protocol.to_dict(),
            "target_source_sha256": target_sha256,
            "target_manifest_sha256": target_manifest_sha256,
            "target_source_stat": source_stat,
            "target_manifest_stat": metadata_stat,
            "target_selection": list(protocol.task_ids),
            "label_available_date_filter": "disabled_by_user",
            "continuous_label_embargo": "none",
            "test_values_opened": True,
            "join_contract": "target many-to-one representation on report_id+feature_available_date",
        },
    )


def validate_static_target_outputs(directory: str | Path) -> dict[str, object]:
    root = Path(directory).expanduser().resolve()
    target_path = root / "aligned_static_targets.parquet"
    count_path = root / "static_target_counts.csv"
    manifest_path = root / "manifest.json"
    for path in (target_path, count_path, manifest_path):
        if not path.is_file():
            raise FileNotFoundError(f"静态Target产物缺失: {path}")
    frame = pd.read_parquet(
        target_path,
        columns=[
            "sample_id",
            "report_id",
            "task_id",
            "split",
            "representation_row",
            "feature_available_date",
            "label_available_date",
        ],
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != STATIC_TARGET_SCHEMA:
        raise ValueError("静态Target schema不匹配")
    if frame["sample_id"].duplicated().any():
        raise ValueError("静态Target sample_id重复")
    if frame.duplicated(["report_id", "task_id"]).any():
        raise ValueError("静态Target (report_id,task_id)重复")
    if set(frame["task_id"].astype(str)) != set(STATIC_TASKS):
        raise ValueError("静态Target任务集合不是约定的5项")
    if set(frame["split"].astype(str)) != set(SPLIT_NAMES):
        raise ValueError("静态Target分区不完整")
    if manifest.get("label_available_date_filter") != "disabled_by_user":
        raise ValueError("静态Target没有记录PIT覆盖")
    return {
        "rows": len(frame),
        "tasks": int(frame["task_id"].nunique()),
        "splits": sorted(frame["split"].astype(str).unique()),
    }


def _label_end_dates(
    feature_dates: pd.Series, return_index: pd.DatetimeIndex, horizon: int
) -> pd.Series:
    positions = return_index.get_indexer(
        pd.to_datetime(feature_dates, errors="coerce").dt.normalize()
    )
    end_positions = positions + int(horizon)
    valid = (positions >= 0) & (end_positions < len(return_index))
    output = pd.Series(pd.NaT, index=feature_dates.index, dtype="datetime64[ns]")
    output.loc[valid] = return_index.to_numpy()[end_positions[valid]]
    return output


def assign_direct_return_partitions(
    frame: pd.DataFrame, protocol: StaticProtocol
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply the separately authorized five-trading-day boundary purge."""

    required = {"split", "label_end_date_5d"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError("直接收益分区缺少字段: " + ", ".join(missing))
    out = frame.copy()
    label_end = pd.to_datetime(out["label_end_date_5d"], errors="coerce").dt.normalize()
    split = out["split"].astype("string")
    partition = pd.Series(pd.NA, index=out.index, dtype="string")
    train = split.eq("train") & label_end.notna()
    partition.loc[train & label_end.lt(protocol.validation.start)] = "train_tuning"
    partition.loc[
        train
        & label_end.ge(protocol.validation.start)
        & label_end.lt(protocol.test.start)
    ] = "train_boundary"
    validation = split.eq("validation") & label_end.lt(protocol.test.start)
    partition.loc[validation] = "validation"
    partition.loc[split.eq("test") & label_end.notna()] = "test"
    audit = pd.DataFrame(
        [
            {
                "stage": "direct_return_split",
                "reason": "train_tuning",
                "count": int(partition.eq("train_tuning").sum()),
            },
            {
                "stage": "direct_return_split",
                "reason": "train_boundary_saved_for_final_refit",
                "count": int(partition.eq("train_boundary").sum()),
            },
            {
                "stage": "direct_return_split",
                "reason": "validation",
                "count": int(partition.eq("validation").sum()),
            },
            {
                "stage": "direct_return_split",
                "reason": "test",
                "count": int(partition.eq("test").sum()),
            },
            {
                "stage": "direct_return_split",
                "reason": "purged_cross_boundary_or_missing_target",
                "count": int(partition.isna().sum()),
            },
        ]
    )
    out["partition"] = partition
    out = out[out["partition"].notna()].reset_index(drop=True)
    return out, audit


def _direct_target_output(config: Mapping[str, object]) -> Path:
    return _run_directory(config) / "direct_return_targets"


def run_static_direct_return_target_stage(config: Mapping[str, object]) -> Path:
    """Attach a full-market five-day return-rank target to every usable report."""

    protocol = parse_static_protocol(config)
    direct_cfg = config.get("direct_return", {})
    if not isinstance(direct_cfg, Mapping) or not bool(direct_cfg.get("enabled", False)):
        raise ValueError("direct_return.enabled必须为true")
    horizon = int(direct_cfg.get("horizon", 5))
    if horizon != 5:
        raise ValueError("本实验直接收益窗口必须固定为5")
    _, metadata, _, rep_manifest, _ = _load_representation_bundle(config)
    reports = metadata[
        ["report_id", "representation_row", "symbol", "feature_available_date"]
    ].copy()
    reports["trading_date"] = pd.to_datetime(
        reports["feature_available_date"], errors="coerce"
    ).dt.normalize()
    reports["split"] = assign_static_splits(
        reports["feature_available_date"], protocol
    ).to_numpy()
    reports = reports[reports["split"].notna()].copy()
    reports["symbol"] = canonical_stock_code(reports["symbol"])
    returns_path = Path(
        str(direct_cfg.get("industry_adjusted_daily_path", ""))
    ).expanduser().resolve()
    if not returns_path.is_file():
        raise FileNotFoundError(f"直接收益数据不存在: {returns_path}")
    output = _direct_target_output(config)
    return_stat = _stat_identity(returns_path)
    if output.exists():
        existing = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
        fast_expected = {
            "schema_version": DIRECT_TARGET_SCHEMA,
            "representation_fingerprint": rep_manifest["representation_fingerprint"],
            "config_sha256": protocol_config_hash(config),
            "return_source_stat": return_stat,
        }
        if all(existing.get(key) == value for key, value in fast_expected.items()):
            validate_direct_return_target_outputs(output)
            return output
    return_sha256 = sha256_file(returns_path)
    source_fingerprint = hashlib.sha256(
        (
            rep_manifest["representation_fingerprint"]
            + return_sha256
            + json.dumps(protocol.to_dict(), sort_keys=True)
        ).encode("utf-8")
    ).hexdigest()
    expected = {
        "schema_version": DIRECT_TARGET_SCHEMA,
        "target_fingerprint": source_fingerprint,
        "representation_fingerprint": rep_manifest["representation_fingerprint"],
        "config_sha256": protocol_config_hash(config),
    }
    if _reuse_exact_output(
        output,
        expected_manifest=expected,
        validator=validate_direct_return_target_outputs,
    ):
        return output
    daily = read_return_panel(
        returns_path,
        hdf_key=(
            str(direct_cfg["hdf_key"])
            if direct_cfg.get("hdf_key") not in (None, "")
            else None
        ),
    )
    validate_decimal_returns(daily, name="industry_adjusted_return_daily")
    forward = compound_forward(daily, [horizon])[horizon]
    full_market_rank = forward.rank(axis=1, method="average", pct=True) - 0.5
    reports["industry_adjusted_return_fut5d"] = extract_forward_values(
        reports,
        forward,
        date_column="trading_date",
        stock_column="symbol",
    )
    reports["target_return_rank_5d"] = extract_forward_values(
        reports,
        full_market_rank,
        date_column="trading_date",
        stock_column="symbol",
    )
    reports["label_end_date_5d"] = _label_end_dates(
        reports["trading_date"], daily.index, horizon
    )
    valid = (
        np.isfinite(reports["industry_adjusted_return_fut5d"].to_numpy(dtype=float))
        & np.isfinite(reports["target_return_rank_5d"].to_numpy(dtype=float))
        & reports["label_end_date_5d"].notna().to_numpy()
    )
    missing_target_rows = int((~valid).sum())
    reports = reports.loc[valid].copy()
    reports, split_audit = assign_direct_return_partitions(reports, protocol)
    counts = reports.groupby(["symbol", "trading_date"])["report_id"].transform("size")
    reports["sample_weight"] = 1.0 / counts.to_numpy(dtype=float)
    weight_sums = reports.groupby(["symbol", "trading_date"])["sample_weight"].sum()
    if not np.allclose(weight_sums.to_numpy(dtype=float), 1.0, atol=1e-12, rtol=0):
        raise RuntimeError("直接收益同股票日样本权重之和不为1")
    reports["task_id"] = DIRECT_TASK_ID
    reports = reports.sort_values(
        ["partition", "trading_date", "symbol", "report_id"]
    ).reset_index(drop=True)
    audit = pd.concat(
        [
            pd.DataFrame(
                [
                    {
                        "stage": "direct_return_target",
                        "reason": "feature_window_reports",
                        "count": int(len(valid)),
                    },
                    {
                        "stage": "direct_return_target",
                        "reason": "missing_return_or_nontrading_date",
                        "count": missing_target_rows,
                    },
                    {
                        "stage": "direct_return_target",
                        "reason": "usable_after_purge",
                        "count": len(reports),
                    },
                ]
            ),
            split_audit,
        ],
        ignore_index=True,
    )
    return _atomic_write_once(
        output,
        tables={
            "direct_return_report_targets.parquet": reports,
            "direct_return_target_audit.csv": audit,
        },
        manifest={
            **expected,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "protocol": protocol.to_dict(),
            "horizon": 5,
            "target": "full-market daily percentile rank of future 5-day industry-adjusted return",
            "sample_weight": "inverse_reports_per_stock_day",
            "purge": "five-trading-day target must end before next split",
            "aggregation_order": "fit and predict at report level before stock-day aggregation",
            "return_source": str(returns_path),
            "return_source_sha256": return_sha256,
            "return_source_stat": return_stat,
        },
    )


def validate_direct_return_target_outputs(directory: str | Path) -> dict[str, object]:
    root = Path(directory).expanduser().resolve()
    target_path = root / "direct_return_report_targets.parquet"
    audit_path = root / "direct_return_target_audit.csv"
    manifest_path = root / "manifest.json"
    for path in (target_path, audit_path, manifest_path):
        if not path.is_file():
            raise FileNotFoundError(f"直接收益Target产物缺失: {path}")
    frame = pd.read_parquet(target_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != DIRECT_TARGET_SCHEMA:
        raise ValueError("直接收益Target schema不匹配")
    if frame["report_id"].duplicated().any():
        raise ValueError("直接收益Target report_id重复")
    if not set(frame["partition"].astype(str)).issubset(
        {"train_tuning", "train_boundary", "validation", "test"}
    ):
        raise ValueError("直接收益Target含未知partition")
    sums = frame.groupby(["symbol", "trading_date"])["sample_weight"].sum()
    if not np.allclose(sums.to_numpy(dtype=float), 1.0, atol=1e-12, rtol=0):
        raise ValueError("直接收益Target股票日权重和不为1")
    return {
        "rows": len(frame),
        "reports": int(frame["report_id"].nunique()),
        "partitions": frame["partition"].value_counts().to_dict(),
    }


@dataclass(frozen=True)
class WeightedRidgeStats:
    n_rows: int
    sum_weight: float
    mean_x: np.ndarray
    mean_y: float
    centered_sum_x2: np.ndarray
    centered_gram: np.ndarray
    centered_x_y: np.ndarray

    @property
    def hidden_size(self) -> int:
        return int(self.mean_x.shape[0])


@dataclass(frozen=True)
class RidgeParameters:
    alpha: float
    mean: np.ndarray
    scale: np.ndarray
    coef: np.ndarray
    intercept: float


def merge_weighted_ridge_stats(
    values: Sequence[WeightedRidgeStats],
) -> WeightedRidgeStats:
    if not values:
        raise ValueError("合并Ridge统计量时输入为空")
    hidden = values[0].hidden_size
    if any(item.hidden_size != hidden for item in values):
        raise ValueError("Ridge统计量hidden_size不一致")
    if any(item.sum_weight <= 0 for item in values):
        raise ValueError("合并Ridge统计量时权重和必须为正")
    merged = values[0]
    for item in values[1:]:
        left_weight = float(merged.sum_weight)
        right_weight = float(item.sum_weight)
        total_weight = left_weight + right_weight
        mean_delta = item.mean_x - merged.mean_x
        target_mean_delta = float(item.mean_y - merged.mean_y)
        correction = left_weight * right_weight / total_weight
        merged = WeightedRidgeStats(
            n_rows=merged.n_rows + item.n_rows,
            sum_weight=total_weight,
            mean_x=(
                merged.mean_x + mean_delta * (right_weight / total_weight)
            ),
            mean_y=(
                merged.mean_y
                + target_mean_delta * (right_weight / total_weight)
            ),
            centered_sum_x2=(
                merged.centered_sum_x2
                + item.centered_sum_x2
                + np.square(mean_delta) * correction
            ),
            centered_gram=(
                merged.centered_gram
                + item.centered_gram
                + np.outer(mean_delta, mean_delta) * correction
            ),
            centered_x_y=(
                merged.centered_x_y
                + item.centered_x_y
                + mean_delta * target_mean_delta * correction
            ),
        )
    return merged


class _TorchMoments:
    def __init__(self, hidden_size: int, *, device: str):
        import torch

        self.n_rows = 0
        self.sum_weight = torch.zeros((), dtype=torch.float64, device=device)
        self.mean_x = torch.zeros(hidden_size, dtype=torch.float64, device=device)
        self.mean_y = torch.zeros((), dtype=torch.float64, device=device)
        self.centered_sum_x2 = torch.zeros(
            hidden_size, dtype=torch.float64, device=device
        )
        self.centered_gram = torch.zeros(
            (hidden_size, hidden_size), dtype=torch.float64, device=device
        )
        self.centered_x_y = torch.zeros(
            hidden_size, dtype=torch.float64, device=device
        )

    def update(self, x, y, weight) -> None:
        import torch

        if x.ndim != 2 or y.ndim != 1 or weight.ndim != 1:
            raise ValueError("Ridge统计量批次shape无效")
        if len(x) != len(y) or len(x) != len(weight):
            raise ValueError("Ridge统计量批次长度不一致")
        if not len(x):
            return
        if not torch.isfinite(x).all() or not torch.isfinite(y).all():
            raise ValueError("Ridge统计量输入含NaN/Inf")
        if not torch.isfinite(weight).all() or torch.any(weight <= 0):
            raise ValueError("Ridge统计量权重必须有限且为正")
        batch_weight = weight.sum(dtype=torch.float64)
        batch_mean_x = (
            (x * weight[:, None]).sum(dim=0, dtype=torch.float64) / batch_weight
        )
        batch_mean_y = (weight * y).sum(dtype=torch.float64) / batch_weight
        centered_x = x - batch_mean_x.to(dtype=x.dtype)
        centered_y = y - batch_mean_y.to(dtype=y.dtype)
        root_weighted_x = centered_x * torch.sqrt(weight)[:, None]
        batch_centered_sum_x2 = (
            centered_x.square() * weight[:, None]
        ).sum(dim=0, dtype=torch.float64)
        batch_centered_gram = (
            root_weighted_x.T @ root_weighted_x
        ).to(torch.float64)
        batch_centered_x_y = (
            centered_x.T @ (weight * centered_y)
        ).to(torch.float64)
        if self.n_rows == 0:
            self.mean_x.copy_(batch_mean_x)
            self.mean_y.copy_(batch_mean_y)
            self.centered_sum_x2.copy_(batch_centered_sum_x2)
            self.centered_gram.copy_(batch_centered_gram)
            self.centered_x_y.copy_(batch_centered_x_y)
        else:
            total_weight = self.sum_weight + batch_weight
            mean_delta = batch_mean_x - self.mean_x
            target_mean_delta = batch_mean_y - self.mean_y
            correction = self.sum_weight * batch_weight / total_weight
            self.centered_sum_x2 += (
                batch_centered_sum_x2 + mean_delta.square() * correction
            )
            self.centered_gram += (
                batch_centered_gram
                + torch.outer(mean_delta, mean_delta) * correction
            )
            self.centered_x_y += (
                batch_centered_x_y
                + mean_delta * target_mean_delta * correction
            )
            self.mean_x += mean_delta * (batch_weight / total_weight)
            self.mean_y += target_mean_delta * (batch_weight / total_weight)
        self.n_rows += int(len(x))
        self.sum_weight += batch_weight

    def freeze(self) -> WeightedRidgeStats:
        return WeightedRidgeStats(
            n_rows=self.n_rows,
            sum_weight=float(self.sum_weight.detach().cpu()),
            mean_x=self.mean_x.detach().cpu().numpy(),
            mean_y=float(self.mean_y.detach().cpu()),
            centered_sum_x2=self.centered_sum_x2.detach().cpu().numpy(),
            centered_gram=self.centered_gram.detach().cpu().numpy(),
            centered_x_y=self.centered_x_y.detach().cpu().numpy(),
        )


def _stats_from_arrays(
    x: np.ndarray,
    y: np.ndarray,
    weight: np.ndarray,
    *,
    device: str,
) -> WeightedRidgeStats:
    import torch

    values = np.asarray(x, dtype=np.float32)
    target = np.asarray(y, dtype=np.float32)
    weights = np.asarray(weight, dtype=np.float32)
    accumulator = _TorchMoments(values.shape[1], device=device)
    accumulator.update(
        torch.from_numpy(values).to(device),
        torch.from_numpy(target).to(device),
        torch.from_numpy(weights).to(device),
    )
    return accumulator.freeze()


def accumulate_layer_statistics(
    representations: np.ndarray,
    *,
    layer: int,
    partitions: Mapping[
        tuple[str, str], tuple[np.ndarray, np.ndarray, np.ndarray]
    ],
    device: str,
    row_chunk_size: int,
) -> dict[tuple[str, str], WeightedRidgeStats]:
    """Read one complete layer once and update every task/partition statistic."""

    import torch

    if layer not in MODELED_LAYERS:
        raise ValueError("Ridge统计只允许Layer 1~12")
    if representations.ndim != 3 or representations.shape[1] != 13:
        raise ValueError("Representation必须是[N,13,H]")
    if row_chunk_size <= 0:
        raise ValueError("row_chunk_size必须为正")
    prepared: dict[
        tuple[str, str], tuple[np.ndarray, np.ndarray, np.ndarray]
    ] = {}
    accumulators: dict[tuple[str, str], _TorchMoments] = {}
    for key, payload in partitions.items():
        rows, y, weight = payload
        rows = np.asarray(rows, dtype=np.int64)
        order = np.argsort(rows, kind="stable")
        rows = rows[order]
        y = np.asarray(y, dtype=np.float32)[order]
        weight = np.asarray(weight, dtype=np.float32)[order]
        if len(rows) == 0:
            continue
        if (rows < 0).any() or (rows >= len(representations)).any():
            raise ValueError(f"{key}含越界representation_row")
        if np.unique(rows).size != len(rows):
            raise ValueError(f"{key}在同一任务分区内含重复representation_row")
        prepared[key] = (rows, y, weight)
        accumulators[key] = _TorchMoments(
            int(representations.shape[2]), device=device
        )
    for start in range(0, len(representations), int(row_chunk_size)):
        end = min(start + int(row_chunk_size), len(representations))
        x_numpy = np.array(
            representations[start:end, layer, :], dtype=np.float32, copy=True
        )
        x_chunk = torch.from_numpy(x_numpy).to(
            device, non_blocking=device.startswith("cuda")
        )
        for key, (rows, y, weight) in prepared.items():
            left = int(np.searchsorted(rows, start, side="left"))
            right = int(np.searchsorted(rows, end, side="left"))
            if right <= left:
                continue
            local = torch.from_numpy(rows[left:right] - start).to(
                device=device, dtype=torch.long
            )
            accumulators[key].update(
                x_chunk.index_select(0, local),
                torch.from_numpy(y[left:right]).to(device),
                torch.from_numpy(weight[left:right]).to(device),
            )
        del x_chunk
    return {key: value.freeze() for key, value in accumulators.items()}


def ridge_path_from_stats(
    stats: WeightedRidgeStats,
    alphas: Sequence[float],
    *,
    device: str,
) -> dict[float, RidgeParameters]:
    """Solve standardized weighted primal Ridge from reusable moments."""

    import torch

    alpha_values = sorted({float(value) for value in alphas})
    if not alpha_values or min(alpha_values) <= 0:
        raise ValueError("Ridge alpha必须为正")
    if stats.n_rows < 2 or stats.sum_weight <= 0:
        raise ValueError("Ridge统计量样本不足")
    mean = stats.mean_x
    variance = stats.centered_sum_x2 / stats.sum_weight
    variance = np.maximum(variance, 0.0)
    scale = np.sqrt(variance)
    scale[~np.isfinite(scale) | (scale <= np.finfo(float).eps)] = 1.0
    y_mean = stats.mean_y
    standardized_gram = stats.centered_gram / np.outer(scale, scale)
    standardized_gram = (standardized_gram + standardized_gram.T) / 2.0
    standardized_xy = stats.centered_x_y / scale
    if not np.isfinite(standardized_gram).all() or not np.isfinite(
        standardized_xy
    ).all():
        raise FloatingPointError("标准化Ridge统计量含NaN/Inf")
    gram = torch.from_numpy(standardized_gram).to(device=device, dtype=torch.float64)
    rhs = torch.from_numpy(standardized_xy).to(device=device, dtype=torch.float64)
    identity = torch.eye(stats.hidden_size, dtype=torch.float64, device=device)
    result: dict[float, RidgeParameters] = {}
    for alpha in alpha_values:
        matrix = gram + alpha * identity
        factor, info = torch.linalg.cholesky_ex(matrix)
        if int(info.max().detach().cpu()) == 0:
            coef = torch.cholesky_solve(rhs[:, None], factor).squeeze(1)
        else:
            coef = torch.linalg.solve(matrix, rhs)
        if not torch.isfinite(coef).all():
            raise FloatingPointError(f"Ridge求解结果含NaN/Inf: alpha={alpha}")
        result[alpha] = RidgeParameters(
            alpha=alpha,
            mean=mean.astype(np.float64, copy=True),
            scale=scale.astype(np.float64, copy=True),
            coef=coef.detach().cpu().numpy(),
            intercept=float(y_mean),
        )
    return result


def predict_representation_rows(
    representations: np.ndarray,
    rows: np.ndarray,
    *,
    layer: int,
    models: Sequence[RidgeParameters],
    device: str,
    row_chunk_size: int,
) -> np.ndarray:
    import torch

    positions = np.asarray(rows, dtype=np.int64)
    if not models:
        raise ValueError("预测模型列表为空")
    if (positions < 0).any() or (positions >= len(representations)).any():
        raise ValueError("预测representation_row越界")
    reference = models[0]
    if any(
        not np.array_equal(model.mean, reference.mean)
        or not np.array_equal(model.scale, reference.scale)
        for model in models[1:]
    ):
        raise ValueError("同一Ridge path模型的scaler不一致")
    mean = torch.from_numpy(reference.mean.astype(np.float32)).to(device)
    scale = torch.from_numpy(reference.scale.astype(np.float32)).to(device)
    coefficients = torch.from_numpy(
        np.column_stack([model.coef for model in models]).astype(np.float32)
    ).to(device)
    intercepts = torch.tensor(
        [model.intercept for model in models], dtype=torch.float32, device=device
    )
    output = np.empty((len(positions), len(models)), dtype=np.float32)
    for start in range(0, len(positions), int(row_chunk_size)):
        end = min(start + int(row_chunk_size), len(positions))
        x = torch.from_numpy(
            np.asarray(
                representations[positions[start:end], layer, :], dtype=np.float32
            )
        ).to(device, non_blocking=device.startswith("cuda"))
        prediction = ((x - mean) / scale) @ coefficients + intercepts
        output[start:end] = prediction.detach().cpu().numpy()
    return output


def _safe_spearman(left: np.ndarray, right: np.ndarray) -> float:
    valid = np.isfinite(left) & np.isfinite(right)
    if valid.sum() < 2:
        return np.nan
    x = left[valid]
    y = right[valid]
    if np.unique(x).size < 2 or np.unique(y).size < 2:
        return np.nan
    value = spearmanr(x, y).correlation
    return float(value) if np.isfinite(value) else np.nan


def select_alpha_by_report_spearman(
    predictions: np.ndarray,
    y_validation: np.ndarray,
    alphas: Sequence[float],
) -> tuple[float, list[dict[str, float]]]:
    values = [float(value) for value in alphas]
    if predictions.shape != (len(y_validation), len(values)):
        raise ValueError("Validation预测矩阵shape与alpha不一致")
    records = [
        {
            "alpha": alpha,
            "validation_report_spearman": _safe_spearman(
                predictions[:, index], np.asarray(y_validation, dtype=float)
            ),
        }
        for index, alpha in enumerate(values)
    ]
    finite = [item for item in records if np.isfinite(item["validation_report_spearman"])]
    if not finite:
        raise ValueError("所有alpha的Validation研报Spearman均无效")
    selected = max(
        finite,
        key=lambda item: (item["validation_report_spearman"], item["alpha"]),
    )
    return float(selected["alpha"]), records


def _ridge_equivalence_audit(
    representations: np.ndarray,
    frame: pd.DataFrame,
    *,
    layer: int,
    alphas: Sequence[float],
    device: str,
    sample_rows: int,
    minimum_pearson: float,
    maximum_spearman_difference: float,
) -> dict[str, object]:
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler

    train = frame[frame["partition"].eq("train_tuning")].head(sample_rows)
    validation = frame[frame["partition"].eq("validation")].head(
        max(256, sample_rows // 4)
    )
    if len(train) < 20 or len(validation) < 20:
        raise ValueError("GPU Ridge等价性测试样本不足")
    x_train = np.asarray(
        representations[train["representation_row"].to_numpy(dtype=int), layer, :],
        dtype=np.float32,
    )
    x_validation = np.asarray(
        representations[
            validation["representation_row"].to_numpy(dtype=int), layer, :
        ],
        dtype=np.float32,
    )
    y_train = train["label_value"].to_numpy(dtype=float)
    y_validation = validation["label_value"].to_numpy(dtype=float)
    weights = train["sample_weight"].to_numpy(dtype=float)
    nonfinite_inputs = {
        "x_train": int(x_train.size - np.isfinite(x_train).sum()),
        "x_validation": int(
            x_validation.size - np.isfinite(x_validation).sum()
        ),
        "y_train": int(y_train.size - np.isfinite(y_train).sum()),
        "y_validation": int(
            y_validation.size - np.isfinite(y_validation).sum()
        ),
        "sample_weight": int(weights.size - np.isfinite(weights).sum()),
    }
    if any(nonfinite_inputs.values()) or np.any(weights <= 0):
        raise ValueError(
            "Ridge等价性审计原始输入无效: "
            + json.dumps(
                {
                    "nonfinite_counts": nonfinite_inputs,
                    "nonpositive_weight_rows": int(np.sum(weights <= 0)),
                },
                ensure_ascii=False,
            )
        )
    accelerated_stats = _stats_from_arrays(
        x_train, y_train, weights, device=device
    )
    accelerated_path = ridge_path_from_stats(
        accelerated_stats, alphas, device=device
    )
    accelerated = np.column_stack(
        [
            ((x_validation - model.mean) / model.scale) @ model.coef
            + model.intercept
            for model in accelerated_path.values()
        ]
    )
    scaler = StandardScaler().fit(x_train, sample_weight=weights)
    x_train_scaled = scaler.transform(x_train)
    x_validation_scaled = scaler.transform(x_validation)
    reference_columns = []
    for alpha in accelerated_path:
        model = Ridge(alpha=alpha, solver="cholesky").fit(
            x_train_scaled, y_train, sample_weight=weights
        )
        reference_columns.append(model.predict(x_validation_scaled))
    reference = np.column_stack(reference_columns)
    nonfinite_predictions = {
        "accelerated": int(accelerated.size - np.isfinite(accelerated).sum()),
        "reference": int(reference.size - np.isfinite(reference).sum()),
    }
    base_result = {
        "rows_train": len(train),
        "rows_validation": len(validation),
        "nonfinite_input_counts": nonfinite_inputs,
        "nonfinite_prediction_counts": nonfinite_predictions,
        "required_minimum_pearson": minimum_pearson,
        "allowed_maximum_spearman_difference": maximum_spearman_difference,
    }
    if any(nonfinite_predictions.values()):
        return {
            "passed": False,
            **base_result,
            "failure_reason": "equivalence_predictions_contain_nan_or_inf",
            "minimum_prediction_pearson": None,
            "maximum_validation_spearman_difference": None,
            "accelerated_selected_alpha": None,
            "reference_selected_alpha": None,
        }
    pearsons = []
    differences = []
    for index in range(reference.shape[1]):
        value = pearsonr(accelerated[:, index], reference[:, index]).statistic
        pearsons.append(float(value))
        differences.append(
            abs(
                _safe_spearman(accelerated[:, index], y_validation)
                - _safe_spearman(reference[:, index], y_validation)
            )
        )
    if not np.isfinite(pearsons).all() or not np.isfinite(differences).all():
        return {
            "passed": False,
            **base_result,
            "failure_reason": "equivalence_metrics_are_nonfinite",
            "minimum_prediction_pearson": (
                float(np.nanmin(pearsons))
                if np.isfinite(pearsons).any()
                else None
            ),
            "maximum_validation_spearman_difference": (
                float(np.nanmax(differences))
                if np.isfinite(differences).any()
                else None
            ),
            "accelerated_selected_alpha": None,
            "reference_selected_alpha": None,
        }
    try:
        accelerated_alpha, _ = select_alpha_by_report_spearman(
            accelerated, y_validation, list(accelerated_path)
        )
        reference_alpha, _ = select_alpha_by_report_spearman(
            reference, y_validation, list(accelerated_path)
        )
    except ValueError as error:
        return {
            "passed": False,
            **base_result,
            "failure_reason": f"alpha_selection_failed: {error}",
            "minimum_prediction_pearson": min(pearsons),
            "maximum_validation_spearman_difference": max(differences),
            "accelerated_selected_alpha": None,
            "reference_selected_alpha": None,
        }
    passed = (
        min(pearsons) >= minimum_pearson
        and max(differences) <= maximum_spearman_difference
        and accelerated_alpha == reference_alpha
    )
    return {
        "passed": bool(passed),
        **base_result,
        "failure_reason": None if passed else "equivalence_threshold_not_met",
        "minimum_prediction_pearson": min(pearsons),
        "maximum_validation_spearman_difference": max(differences),
        "accelerated_selected_alpha": accelerated_alpha,
        "reference_selected_alpha": reference_alpha,
    }


def _benchmark_row_chunk_size(
    representations: np.ndarray,
    *,
    layer: int,
    candidates: Sequence[int],
    device: str,
) -> tuple[int, list[dict[str, object]]]:
    import torch

    values = sorted({int(value) for value in candidates if int(value) > 0})
    if not values:
        raise ValueError("row_chunk_candidates必须含正整数")
    if not device.startswith("cuda"):
        return values[0], [{"row_chunk_size": values[0], "status": "cpu_configured"}]
    records: list[dict[str, object]] = []
    successful: list[tuple[int, float]] = []
    for candidate in values:
        rows = min(candidate, len(representations))
        try:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
            started = time.perf_counter()
            x = torch.from_numpy(
                np.array(
                    representations[:rows, layer, :],
                    dtype=np.float32,
                    copy=True,
                )
            ).to(device)
            _ = x.T @ x
            torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - started
            throughput = rows / max(elapsed, 1e-9)
            peak = int(torch.cuda.max_memory_allocated(device))
            records.append(
                {
                    "row_chunk_size": candidate,
                    "benchmark_rows": rows,
                    "status": "ok",
                    "seconds": elapsed,
                    "rows_per_second": throughput,
                    "peak_memory_bytes": peak,
                }
            )
            successful.append((candidate, throughput))
            del x
        except torch.cuda.OutOfMemoryError:
            records.append({"row_chunk_size": candidate, "status": "oom"})
            torch.cuda.empty_cache()
    if not successful:
        raise RuntimeError("所有Ridge row chunk候选均OOM")
    selected = max(successful, key=lambda item: (item[1], item[0]))[0]
    return selected, records


def _prepare_probe_frames(
    config: Mapping[str, object],
) -> dict[str, pd.DataFrame]:
    static_root = run_static_target_stage(config)
    direct_root = run_static_direct_return_target_stage(config)
    continuous = pd.read_parquet(static_root / "aligned_static_targets.parquet")
    frames: dict[str, pd.DataFrame] = {}
    for task_id, raw in continuous.groupby("task_id", sort=True):
        frame = raw.copy()
        frame["symbol"] = canonical_stock_code(
            frame["symbol"] if "symbol" in frame else frame["stock_code"]
        )
        frame["trading_date"] = pd.to_datetime(
            frame["feature_available_date"], errors="coerce"
        ).dt.normalize()
        frame["label_value"] = pd.to_numeric(frame["label_value"], errors="coerce")
        frame["sample_weight"] = pd.to_numeric(
            frame["target_weight"], errors="coerce"
        )
        frame["partition"] = frame["split"].map(
            {
                "train": "train_tuning",
                "validation": "validation",
                "test": "test",
            }
        ).astype("string")
        frames[str(task_id)] = frame
    direct = pd.read_parquet(
        direct_root / "direct_return_report_targets.parquet"
    ).copy()
    direct["label_value"] = pd.to_numeric(
        direct["target_return_rank_5d"], errors="coerce"
    )
    frames[DIRECT_TASK_ID] = direct
    if set(frames) != set(STATIC_TASKS).union({DIRECT_TASK_ID}):
        raise ValueError("研报Ridge输入没有覆盖5个连续任务和直接收益任务")
    return frames


def _partition_arrays(
    frames: Mapping[str, pd.DataFrame],
) -> dict[tuple[str, str], tuple[np.ndarray, np.ndarray, np.ndarray]]:
    output = {}
    for task_id, frame in frames.items():
        for partition in ("train_tuning", "train_boundary", "validation"):
            selected = frame[frame["partition"].eq(partition)]
            if selected.empty:
                continue
            output[(task_id, partition)] = (
                selected["representation_row"].to_numpy(dtype=np.int64),
                selected["label_value"].to_numpy(dtype=np.float32),
                selected["sample_weight"].to_numpy(dtype=np.float32),
            )
    return output


def _report_probe_output(config: Mapping[str, object]) -> Path:
    return _run_directory(config) / "report_ridge_probes"


def run_static_report_probe_stage(config: Mapping[str, object]) -> Path:
    """Fit all six report-level factor sources with one representation scan/layer."""

    import torch

    protocol = parse_static_protocol(config)
    ridge_cfg = config.get("report_ridge", {})
    performance_cfg = config.get("performance", {})
    if not isinstance(ridge_cfg, Mapping) or not bool(ridge_cfg.get("enabled", False)):
        raise ValueError("report_ridge.enabled必须为true")
    if not isinstance(performance_cfg, Mapping):
        raise ValueError("performance配置必须是对象")
    if str(ridge_cfg.get("backend")) != "torch_primal_gram":
        raise ValueError("本实验只允许torch_primal_gram后端")
    alphas = sorted({float(value) for value in ridge_cfg.get("alpha_grid", [])})
    if not alphas or min(alphas) <= 0:
        raise ValueError("report_ridge.alpha_grid必须含正数")
    device = str(ridge_cfg.get("device", "cuda:1"))
    representations, _, _, rep_manifest, _ = _load_representation_bundle(config)
    frames = _prepare_probe_frames(config)
    static_manifest = _static_target_output(config) / "manifest.json"
    direct_manifest = _direct_target_output(config) / "manifest.json"
    output = _report_probe_output(config)
    expected = {
        "schema_version": REPORT_PROBE_SCHEMA,
        "config_sha256": protocol_config_hash(config),
        "representation_fingerprint": rep_manifest["representation_fingerprint"],
        "static_target_manifest_sha256": sha256_file(static_manifest),
        "direct_target_manifest_sha256": sha256_file(direct_manifest),
    }
    if output.exists():
        manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
        mismatches = {
            key: {"actual": manifest.get(key), "expected": value}
            for key, value in expected.items()
            if manifest.get(key) != value
        }
        if mismatches:
            raise RuntimeError(
                "已有静态Ridge产物与当前协议不匹配: "
                + json.dumps(mismatches, ensure_ascii=False)
            )
        validate_static_report_probe_outputs(output)
        return output
    predicted_rows = sum(
        int(frame["partition"].isin(["validation", "test"]).sum())
        for frame in frames.values()
    )
    disk_audit = _output_disk_budget(
        config,
        estimated_peak_bytes=int(predicted_rows * (12 * 4 + 384) * 2.0),
    )
    if device.startswith("cuda"):
        import gc

        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize(device)
    runtime_audit = configure_layer_probe_runtime(performance_cfg, device=device)
    gpu_audit = gpu_runtime_audit(device)
    minimum_train = int(ridge_cfg.get("minimum_train_rows", 100))
    minimum_validation = int(ridge_cfg.get("minimum_validation_rows", 100))
    for task_id, frame in frames.items():
        if int(frame["partition"].eq("train_tuning").sum()) < minimum_train:
            raise ValueError(f"{task_id}训练样本不足")
        if int(frame["partition"].eq("validation").sum()) < minimum_validation:
            raise ValueError(f"{task_id}验证样本不足")
        if not frame["partition"].eq("test").any():
            raise ValueError(f"{task_id}测试样本为空")
    candidates = [int(value) for value in ridge_cfg.get("row_chunk_candidates", [])]
    row_chunk_size, chunk_benchmark = _benchmark_row_chunk_size(
        representations,
        layer=1,
        candidates=candidates,
        device=device,
    )
    numeric_cfg = ridge_cfg.get("numerical_validation", {})
    if not isinstance(numeric_cfg, Mapping):
        raise ValueError("report_ridge.numerical_validation必须是对象")
    minimum_pearson = float(numeric_cfg.get("minimum_prediction_pearson", 0.99999))
    maximum_spearman_difference = float(
        numeric_cfg.get("maximum_spearman_difference", 1e-4)
    )
    representative = frames[STATIC_TASKS[0]]
    tf32_modes = [True, False] if device.startswith("cuda") else [False]
    numerical_audit = None
    numerical_audit_attempts: list[dict[str, object]] = []
    selected_tf32 = False
    for allow_tf32 in tf32_modes:
        if device.startswith("cuda"):
            torch.backends.cuda.matmul.allow_tf32 = allow_tf32
        try:
            audit = _ridge_equivalence_audit(
                representations,
                representative,
                layer=1,
                alphas=alphas,
                device=device,
                sample_rows=int(ridge_cfg.get("validation_sample_rows", 4096)),
                minimum_pearson=minimum_pearson,
                maximum_spearman_difference=maximum_spearman_difference,
            )
        except (FloatingPointError, RuntimeError, ValueError) as error:
            audit = {
                "passed": False,
                "failure_reason": (
                    f"{type(error).__name__}: {error}"
                ),
            }
        audit["allow_tf32"] = allow_tf32
        numerical_audit_attempts.append(audit)
        numerical_audit = audit
        if audit["passed"]:
            selected_tf32 = allow_tf32
            break
    if numerical_audit is None or not numerical_audit["passed"]:
        raise RuntimeError(
            "GPU Ridge未通过sklearn等价性检查: "
            + json.dumps(
                numerical_audit_attempts, ensure_ascii=False, default=str
            )
        )
    if device.startswith("cuda"):
        torch.backends.cuda.matmul.allow_tf32 = selected_tf32
    partitions = _partition_arrays(frames)
    prediction_frames: dict[str, pd.DataFrame] = {}
    for task_id, frame in frames.items():
        selected = frame[frame["partition"].isin(["validation", "test"])].copy()
        selected = selected.reset_index(drop=True)
        selected["prediction_role"] = "oos"
        selected["model_scope"] = selected["partition"].map(
            {"validation": "train_only", "test": "train_plus_validation"}
        )
        for layer in protocol.modeled_layers:
            selected[f"prediction_layer_{layer}"] = np.nan
        prediction_frames[task_id] = selected
    tuning_records: list[dict[str, object]] = []
    selection_records: list[dict[str, object]] = []
    model_records: list[dict[str, object]] = []
    model_arrays: dict[str, np.ndarray] = {}
    for layer in protocol.modeled_layers:
        layer_stats = accumulate_layer_statistics(
            representations,
            layer=layer,
            partitions=partitions,
            device=device,
            row_chunk_size=row_chunk_size,
        )
        for task_id, frame in frames.items():
            train_key = (task_id, "train_tuning")
            validation_key = (task_id, "validation")
            if train_key not in layer_stats or validation_key not in layer_stats:
                raise RuntimeError(f"{task_id} Layer {layer}缺少训练或验证统计量")
            validation_path = ridge_path_from_stats(
                layer_stats[train_key], alphas, device=device
            )
            validation_rows = frame[frame["partition"].eq("validation")]
            ordered_models = [validation_path[alpha] for alpha in alphas]
            validation_candidates = predict_representation_rows(
                representations,
                validation_rows["representation_row"].to_numpy(dtype=int),
                layer=layer,
                models=ordered_models,
                device=device,
                row_chunk_size=row_chunk_size,
            )
            selected_alpha, tuning = select_alpha_by_report_spearman(
                validation_candidates,
                validation_rows["label_value"].to_numpy(dtype=float),
                alphas,
            )
            for record in tuning:
                tuning_records.append(
                    {"task_id": task_id, "layer": layer, **record}
                )
            selection_records.append(
                {
                    "task_id": task_id,
                    "layer": layer,
                    "selected_alpha": selected_alpha,
                    "validation_report_spearman": next(
                        item["validation_report_spearman"]
                        for item in tuning
                        if item["alpha"] == selected_alpha
                    ),
                }
            )
            validation_model = validation_path[selected_alpha]
            final_parts = [layer_stats[train_key], layer_stats[validation_key]]
            boundary_key = (task_id, "train_boundary")
            if boundary_key in layer_stats:
                final_parts.append(layer_stats[boundary_key])
            final_stats = merge_weighted_ridge_stats(final_parts)
            final_model = ridge_path_from_stats(
                final_stats, [selected_alpha], device=device
            )[selected_alpha]
            prediction_output = prediction_frames[task_id]
            validation_output = prediction_output["partition"].eq("validation")
            test_output = prediction_output["partition"].eq("test")
            prediction_output.loc[
                validation_output, f"prediction_layer_{layer}"
            ] = (
                validation_candidates[:, alphas.index(selected_alpha)]
            )
            test_rows = frame[frame["partition"].eq("test")]
            prediction_output.loc[test_output, f"prediction_layer_{layer}"] = (
                predict_representation_rows(
                    representations,
                    test_rows["representation_row"].to_numpy(dtype=int),
                    layer=layer,
                    models=[final_model],
                    device=device,
                    row_chunk_size=row_chunk_size,
                )[:, 0]
            )
            for scope, model, stats in (
                ("validation", validation_model, layer_stats[train_key]),
                ("test", final_model, final_stats),
            ):
                prefix = f"{scope}__{task_id}__layer_{layer}"
                model_arrays[f"{prefix}__mean"] = model.mean.astype(np.float64)
                model_arrays[f"{prefix}__scale"] = model.scale.astype(np.float64)
                model_arrays[f"{prefix}__coef"] = model.coef.astype(np.float64)
                model_records.append(
                    {
                        "task_id": task_id,
                        "layer": layer,
                        "evaluation_split": scope,
                        "selected_alpha": selected_alpha,
                        "intercept": model.intercept,
                        "training_rows": stats.n_rows,
                        "training_weight_sum": stats.sum_weight,
                        "array_prefix": prefix,
                    }
                )
    for task_id, frame in prediction_frames.items():
        columns = [f"prediction_layer_{layer}" for layer in protocol.modeled_layers]
        values = frame[columns].to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise RuntimeError(f"{task_id} OOS预测含NaN/Inf")
    def writer(temporary: Path) -> None:
        prediction_root = temporary / "predictions"
        prediction_root.mkdir()
        for task_id, frame in prediction_frames.items():
            frame.to_parquet(
                prediction_root / f"{task_id}.parquet",
                index=False,
                compression="zstd",
            )
        pd.DataFrame(tuning_records).to_csv(
            temporary / "ridge_tuning.csv", index=False
        )
        pd.DataFrame(selection_records).to_csv(
            temporary / "selected_alphas.csv", index=False
        )
        pd.DataFrame(model_records).to_csv(
            temporary / "model_index.csv", index=False
        )
        np.savez_compressed(temporary / "ridge_models.npz", **model_arrays)
        (temporary / "manifest.json").write_text(
            json.dumps(
                {
                    **expected,
                    "created_at": datetime.now().isoformat(timespec="seconds"),
                    "protocol": protocol.to_dict(),
                    "backend": "torch_primal_gram",
                    "device": device,
                    "modeled_layers": list(protocol.modeled_layers),
                    "task_ids": sorted(frames),
                    "alpha_grid": alphas,
                    "alpha_selection": "pooled validation report Spearman",
                    "alpha_tie_break": "larger_alpha",
                    "row_chunk_size": row_chunk_size,
                    "row_chunk_benchmark": chunk_benchmark,
                    "allow_tf32": selected_tf32,
                    "numerical_equivalence": numerical_audit,
                    "numerical_equivalence_attempts": numerical_audit_attempts,
                    "gpu_runtime": gpu_audit,
                    "cpu_runtime": runtime_audit,
                    "disk_preflight": disk_audit,
                    "test_used_for_selection": False,
                    "test_opened_in_same_run": True,
                    "aggregation_order": "report Ridge and prediction before stock-day mean",
                },
                ensure_ascii=False,
                indent=2,
                default=str,
            ),
            encoding="utf-8",
        )

    _atomic_directory(output, writer)
    validate_static_report_probe_outputs(output)
    return output


def validate_static_report_probe_outputs(directory: str | Path) -> dict[str, object]:
    root = Path(directory).expanduser().resolve()
    required = [
        root / "ridge_tuning.csv",
        root / "selected_alphas.csv",
        root / "model_index.csv",
        root / "ridge_models.npz",
        root / "manifest.json",
    ]
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(f"静态研报Ridge产物缺失: {path}")
    manifest = json.loads(required[-1].read_text(encoding="utf-8"))
    if manifest.get("schema_version") != REPORT_PROBE_SCHEMA:
        raise ValueError("静态研报Ridge schema不匹配")
    if tuple(manifest.get("modeled_layers", [])) != MODELED_LAYERS:
        raise ValueError("静态研报Ridge建模层不是Layer 1~12")
    if 0 in manifest.get("modeled_layers", []):
        raise ValueError("Layer 0错误进入Ridge")
    expected_tasks = set(STATIC_TASKS).union({DIRECT_TASK_ID})
    if set(manifest.get("task_ids", [])) != expected_tasks:
        raise ValueError("静态研报Ridge任务集合不完整")
    selections = pd.read_csv(root / "selected_alphas.csv")
    layer_sets = selections.groupby("task_id")["layer"].agg(set)
    if set(layer_sets.index.astype(str)) != expected_tasks or not layer_sets.map(
        lambda value: value == set(MODELED_LAYERS)
    ).all():
        raise ValueError("静态研报Ridge没有覆盖全部任务×Layer 1~12")
    prediction_columns = [
        f"prediction_layer_{layer}" for layer in MODELED_LAYERS
    ]
    rows = 0
    for task_id in expected_tasks:
        path = root / "predictions" / f"{task_id}.parquet"
        if not path.is_file():
            raise FileNotFoundError(f"静态研报Ridge缺少预测: {path}")
        frame = pd.read_parquet(path)
        missing = sorted(set(prediction_columns).difference(frame.columns))
        if missing:
            raise ValueError(f"{task_id}缺少Layer预测: {missing}")
        if "prediction_layer_0" in frame:
            raise ValueError("Layer 0错误进入预测产物")
        if set(frame["partition"].astype(str)) != {"validation", "test"}:
            raise ValueError(f"{task_id} OOS预测分区不完整")
        if not np.isfinite(frame[prediction_columns].to_numpy(dtype=float)).all():
            raise ValueError(f"{task_id}预测含NaN/Inf")
        rows += len(frame)
    return {
        "prediction_rows": rows,
        "tasks": len(expected_tasks),
        "layers": len(MODELED_LAYERS),
    }


def aggregate_report_predictions_to_stock_day(
    predictions: pd.DataFrame,
    *,
    task_id: str,
) -> pd.DataFrame:
    """Aggregate only after report-level model weights and predictions are fixed."""

    layer_columns = [f"prediction_layer_{layer}" for layer in MODELED_LAYERS]
    required = {
        "report_id",
        "symbol",
        "trading_date",
        "partition",
        *layer_columns,
    }
    missing = sorted(required.difference(predictions.columns))
    if missing:
        raise ValueError("股票日聚合缺少研报预测字段: " + ", ".join(missing))
    if not predictions["partition"].isin(["validation", "test"]).all():
        raise ValueError("股票日因子只能使用validation/test OOS预测")
    work = predictions.copy()
    work["symbol"] = canonical_stock_code(work["symbol"])
    work["trading_date"] = pd.to_datetime(
        work["trading_date"], errors="coerce"
    ).dt.normalize()
    if work[["symbol", "trading_date"]].isna().any().any():
        raise ValueError("股票日因子聚合键含空值")
    grouped = (
        work.groupby(
            ["partition", "trading_date", "symbol"],
            sort=True,
            observed=True,
        )
        .agg(
            **{column: (column, "mean") for column in layer_columns},
            n_reports=("report_id", "nunique"),
        )
        .reset_index()
        .rename(columns={"partition": "split"})
    )
    grouped.insert(0, "task_id", task_id)
    grouped.insert(
        0,
        "factor_source",
        "direct_return_probe"
        if task_id == DIRECT_TASK_ID
        else "continuous_label_probe",
    )
    if grouped.duplicated(["task_id", "split", "trading_date", "symbol"]).any():
        raise RuntimeError("股票日因子键重复")
    return grouped


def apply_csi300_asof_membership(
    factors: pd.DataFrame,
    weights: pd.DataFrame,
) -> pd.DataFrame:
    """Use the latest complete CSI300 snapshot on or before each factor date."""

    required_factors = {"trading_date", "symbol"}
    required_weights = {"trade_date", "stock_code", "weight"}
    if missing := sorted(required_factors.difference(factors.columns)):
        raise ValueError("CSI300因子输入缺少字段: " + ", ".join(missing))
    if missing := sorted(required_weights.difference(weights.columns)):
        raise ValueError("CSI300权重输入缺少字段: " + ", ".join(missing))
    work = factors.copy()
    work["trading_date"] = pd.to_datetime(
        work["trading_date"], errors="coerce"
    ).dt.normalize()
    work["symbol"] = canonical_stock_code(work["symbol"])
    normalized = weights.copy()
    normalized["trade_date"] = pd.to_datetime(
        normalized["trade_date"], errors="coerce"
    ).dt.normalize()
    normalized["stock_code"] = canonical_stock_code(normalized["stock_code"])
    normalized["weight"] = pd.to_numeric(normalized["weight"], errors="coerce")
    normalized = normalized.dropna(subset=["trade_date", "stock_code", "weight"])
    normalized = normalized[normalized["weight"].gt(0)]
    if normalized.duplicated(["trade_date", "stock_code"]).any():
        raise ValueError("CSI300权重快照键重复")
    snapshots = np.sort(normalized["trade_date"].unique())
    if not len(snapshots):
        raise ValueError("CSI300权重没有有效快照")
    dates = work["trading_date"].to_numpy(dtype="datetime64[ns]")
    positions = np.searchsorted(snapshots, dates, side="right") - 1
    snapshot_date = np.full(len(work), np.datetime64("NaT"), dtype="datetime64[ns]")
    valid = positions >= 0
    snapshot_date[valid] = snapshots[positions[valid]]
    work["csi300_snapshot_date"] = snapshot_date
    lookup = normalized.rename(
        columns={
            "trade_date": "csi300_snapshot_date",
            "stock_code": "symbol",
            "weight": "csi300_weight",
        }
    )[["csi300_snapshot_date", "symbol", "csi300_weight"]]
    out = work.merge(
        lookup,
        on=["csi300_snapshot_date", "symbol"],
        how="left",
        validate="many_to_one",
    )
    out["csi300_member"] = out["csi300_weight"].notna()
    future_snapshot = out["csi300_snapshot_date"].gt(out["trading_date"])
    if future_snapshot.any():
        raise RuntimeError("CSI300评价使用了未来权重快照")
    return out


def daily_rank_ic_tables(
    factors: pd.DataFrame,
    *,
    target_column: str,
    minimum_observations: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    layer_columns = [f"prediction_layer_{layer}" for layer in MODELED_LAYERS]
    required = {
        "factor_source",
        "task_id",
        "split",
        "trading_date",
        "symbol",
        target_column,
        *layer_columns,
    }
    if missing := sorted(required.difference(factors.columns)):
        raise ValueError("RankIC输入缺少字段: " + ", ".join(missing))
    records: list[dict[str, object]] = []
    for (source, task_id, split, date), group in factors.groupby(
        ["factor_source", "task_id", "split", "trading_date"], sort=True
    ):
        target = pd.to_numeric(group[target_column], errors="coerce")
        for layer, column in zip(MODELED_LAYERS, layer_columns):
            prediction = pd.to_numeric(group[column], errors="coerce")
            valid = np.isfinite(prediction.to_numpy(dtype=float)) & np.isfinite(
                target.to_numpy(dtype=float)
            )
            x = prediction.to_numpy(dtype=float)[valid]
            y = target.to_numpy(dtype=float)[valid]
            rank_ic = np.nan
            if (
                len(x) >= minimum_observations
                and np.unique(x).size > 1
                and np.unique(y).size > 1
            ):
                value = spearmanr(x, y).correlation
                rank_ic = float(value) if np.isfinite(value) else np.nan
            records.append(
                {
                    "factor_source": source,
                    "task_id": task_id,
                    "split": split,
                    "trading_date": date,
                    "layer": layer,
                    "rank_ic": rank_ic,
                    "n_stocks": int(len(x)),
                }
            )
    daily = pd.DataFrame(records)
    summary = (
        daily.groupby(
            ["factor_source", "task_id", "split", "layer"], sort=True
        )
        .agg(
            mean_rank_ic=("rank_ic", "mean"),
            n_days=("rank_ic", "count"),
            mean_n_stocks=("n_stocks", "mean"),
        )
        .reset_index()
    )
    return daily, summary


def daily_layer_correlation_tables_wide(
    factors: pd.DataFrame,
    *,
    minimum_observations: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    layer_columns = [f"prediction_layer_{layer}" for layer in MODELED_LAYERS]
    required = {
        "factor_source",
        "task_id",
        "split",
        "trading_date",
        "symbol",
        *layer_columns,
    }
    if missing := sorted(required.difference(factors.columns)):
        raise ValueError("层相关性输入缺少字段: " + ", ".join(missing))
    records: list[dict[str, object]] = []
    for (source, task_id, split, date), group in factors.groupby(
        ["factor_source", "task_id", "split", "trading_date"], sort=True
    ):
        values = group[layer_columns].replace([np.inf, -np.inf], np.nan)
        for left_index, left in enumerate(MODELED_LAYERS):
            for right in MODELED_LAYERS[left_index:]:
                pair = values[
                    [f"prediction_layer_{left}", f"prediction_layer_{right}"]
                ].dropna()
                n_obs = len(pair)
                correlation = np.nan
                if n_obs >= minimum_observations:
                    if left == right:
                        if pair.iloc[:, 0].nunique() > 1:
                            correlation = 1.0
                    elif pair.iloc[:, 0].nunique() > 1 and pair.iloc[:, 1].nunique() > 1:
                        value = spearmanr(pair.iloc[:, 0], pair.iloc[:, 1]).correlation
                        correlation = float(value) if np.isfinite(value) else np.nan
                records.append(
                    {
                        "factor_source": source,
                        "task_id": task_id,
                        "split": split,
                        "trading_date": date,
                        "layer_left": left,
                        "layer_right": right,
                        "spearman": correlation,
                        "n_stocks": n_obs,
                    }
                )
    daily = pd.DataFrame(records)
    summary = (
        daily.groupby(
            [
                "factor_source",
                "task_id",
                "split",
                "layer_left",
                "layer_right",
            ],
            sort=True,
        )
        .agg(
            mean_spearman=("spearman", "mean"),
            median_spearman=("spearman", "median"),
            n_days=("spearman", "count"),
            mean_n_stocks=("n_stocks", "mean"),
        )
        .reset_index()
    )
    return daily, summary


def _read_csi_weights(
    path: Path,
    *,
    index_code: str,
    minimum_snapshot_constituents: int,
) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"沪深300权重不存在: {path}")
    if path.suffix.lower() in {".parquet", ".pq"}:
        raw = pd.read_parquet(path)
    elif path.suffix.lower() in {".csv", ".txt"}:
        raw = pd.read_csv(path)
    else:
        raise ValueError("沪深300权重仅支持Parquet/CSV")
    normalized = normalize_csi_weights(raw, index_code=index_code)
    if minimum_snapshot_constituents <= 0:
        raise ValueError("minimum_snapshot_constituents必须为正")
    counts = normalized.groupby("trade_date")["stock_code"].nunique()
    complete_dates = counts[counts.ge(minimum_snapshot_constituents)].index
    normalized = normalized[normalized["trade_date"].isin(complete_dates)].copy()
    if normalized.empty:
        raise ValueError(
            "沪深300权重没有达到最低成分数的完整快照: "
            f"minimum={minimum_snapshot_constituents}"
        )
    return normalized


def _csi_evaluation_output(config: Mapping[str, object]) -> Path:
    return _run_directory(config) / "csi300_rank_ic"


def run_static_csi300_evaluation_stage(config: Mapping[str, object]) -> Path:
    """Aggregate fixed report predictions and evaluate only in historical CSI300."""

    protocol = parse_static_protocol(config)
    evaluation_cfg = config.get("csi300_evaluation", {})
    direct_cfg = config.get("direct_return", {})
    correlation_cfg = config.get("layer_factor_correlations", {})
    if not all(
        isinstance(value, Mapping)
        for value in (evaluation_cfg, direct_cfg, correlation_cfg)
    ):
        raise ValueError("CSI300/direct_return/layer_factor_correlations配置必须是对象")
    if not bool(evaluation_cfg.get("enabled", False)):
        raise ValueError("csi300_evaluation.enabled必须为true")
    if str(evaluation_cfg.get("membership")) != "latest_snapshot_on_or_before_date":
        raise ValueError("沪深300评价必须使用逐日历史as-of成分")
    probe_root = run_static_report_probe_stage(config)
    probe_manifest = probe_root / "manifest.json"
    direct_target_root = _direct_target_output(config)
    direct_target_manifest = direct_target_root / "manifest.json"
    weights_path = Path(str(evaluation_cfg.get("weights_path", ""))).expanduser().resolve()
    probe_manifest_sha256 = sha256_file(probe_manifest)
    direct_target_manifest_sha256 = sha256_file(direct_target_manifest)
    weights_stat = _stat_identity(weights_path)
    output = _csi_evaluation_output(config)
    if output.exists():
        existing = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
        fast_expected = {
            "schema_version": CSI_EVALUATION_SCHEMA,
            "config_sha256": protocol_config_hash(config),
            "report_probe_manifest_sha256": probe_manifest_sha256,
            "direct_target_manifest_sha256": direct_target_manifest_sha256,
            "csi300_weights_stat": weights_stat,
        }
        if all(existing.get(key) == value for key, value in fast_expected.items()):
            validate_static_csi300_evaluation_outputs(output)
            return output
    expected = {
        "schema_version": CSI_EVALUATION_SCHEMA,
        "config_sha256": protocol_config_hash(config),
        "report_probe_manifest_sha256": probe_manifest_sha256,
        "direct_target_manifest_sha256": direct_target_manifest_sha256,
        "csi300_weights_sha256": sha256_file(weights_path),
    }
    if _reuse_exact_output(
        output,
        expected_manifest=expected,
        validator=validate_static_csi300_evaluation_outputs,
    ):
        return output
    factors: list[pd.DataFrame] = []
    for task_id in list(STATIC_TASKS) + [DIRECT_TASK_ID]:
        prediction = pd.read_parquet(
            probe_root / "predictions" / f"{task_id}.parquet"
        )
        factors.append(
            aggregate_report_predictions_to_stock_day(
                prediction,
                task_id=task_id,
            )
        )
    stock_day = pd.concat(factors, ignore_index=True)
    return_rows = pd.read_parquet(
        direct_target_root / "direct_return_report_targets.parquet",
        columns=["trading_date", "symbol", "industry_adjusted_return_fut5d"],
    )
    return_rows["trading_date"] = pd.to_datetime(
        return_rows["trading_date"], errors="coerce"
    ).dt.normalize()
    return_rows["symbol"] = canonical_stock_code(return_rows["symbol"])
    conflicts = return_rows.groupby(["trading_date", "symbol"])[
        "industry_adjusted_return_fut5d"
    ].nunique(dropna=False)
    if conflicts.gt(1).any():
        raise RuntimeError("直接收益Target同股票日存在冲突的未来5日收益")
    return_lookup = return_rows.drop_duplicates(["trading_date", "symbol"])
    stock_day = stock_day.merge(
        return_lookup,
        on=["trading_date", "symbol"],
        how="left",
        validate="many_to_one",
    )
    weights = _read_csi_weights(
        weights_path,
        index_code=str(evaluation_cfg.get("index_code", "000300.SH")),
        minimum_snapshot_constituents=int(
            evaluation_cfg.get("minimum_snapshot_constituents", 250)
        ),
    )
    with_membership = apply_csi300_asof_membership(stock_day, weights)
    csi300 = with_membership[
        with_membership["csi300_member"]
        & np.isfinite(
            pd.to_numeric(
                with_membership["industry_adjusted_return_fut5d"], errors="coerce"
            ).to_numpy(dtype=float)
        )
    ].copy()
    if csi300.empty:
        raise ValueError("沪深300评价没有有效股票日因子")
    minimum = int(evaluation_cfg.get("minimum_daily_observations", 20))
    daily_ic, summary_ic = daily_rank_ic_tables(
        csi300,
        target_column="industry_adjusted_return_fut5d",
        minimum_observations=minimum,
    )
    correlation_minimum = int(
        correlation_cfg.get("minimum_daily_observations", minimum)
    )
    daily_correlation, summary_correlation = daily_layer_correlation_tables_wide(
        csi300,
        minimum_observations=correlation_minimum,
    )
    expected_tasks = set(STATIC_TASKS).union({DIRECT_TASK_ID})
    for table, name in (
        (summary_ic, "RankIC"),
        (summary_correlation, "层相关性"),
    ):
        if set(table["task_id"].astype(str)) != expected_tasks:
            raise RuntimeError(f"{name}没有覆盖全部6个因子来源")
        if set(table["split"].astype(str)) != {"validation", "test"}:
            raise RuntimeError(f"{name}没有分开覆盖validation/test")
    universe_audit = (
        with_membership.groupby(["split", "trading_date"], sort=True)
        .agg(
            factor_rows=("symbol", "size"),
            csi300_rows=("csi300_member", "sum"),
            csi300_snapshot_date=("csi300_snapshot_date", "max"),
        )
        .reset_index()
    )
    estimated_bytes = sum(
        int(frame.memory_usage(index=True, deep=True).sum())
        for frame in (
            stock_day,
            csi300,
            daily_ic,
            summary_ic,
            daily_correlation,
            summary_correlation,
            universe_audit,
        )
    )
    disk_audit = _output_disk_budget(
        config, estimated_peak_bytes=int(estimated_bytes * 2.5)
    )
    return _atomic_write_once(
        output,
        tables={
            "stock_day_factors.parquet": stock_day,
            "csi300_stock_day_factors.parquet": csi300,
            "daily_rank_ic.parquet": daily_ic,
            "rank_ic_summary.csv": summary_ic,
            "daily_layer_correlations.parquet": daily_correlation,
            "layer_correlation_summary.csv": summary_correlation,
            "csi300_universe_audit.csv": universe_audit,
        },
        manifest={
            **expected,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "protocol": protocol.to_dict(),
            "evaluation_splits": ["validation", "test"],
            "modeled_layers": list(MODELED_LAYERS),
            "factor_tasks": sorted(expected_tasks),
            "aggregation": "unweighted mean report prediction within stock x trading_date",
            "csi300_role": "evaluation mask only after model selection and report prediction",
            "csi300_membership": "latest complete snapshot on or before factor date",
            "metrics": ["daily_rank_ic", "mean_rank_ic"],
            "minimum_daily_observations": minimum,
            "layer_correlation": "daily cross-sectional Spearman within task",
            "test_used_for_selection": False,
            "csi300_weights_stat": weights_stat,
            "disk_preflight": disk_audit,
        },
    )


def validate_static_csi300_evaluation_outputs(
    directory: str | Path,
) -> dict[str, object]:
    root = Path(directory).expanduser().resolve()
    required = [
        root / "stock_day_factors.parquet",
        root / "csi300_stock_day_factors.parquet",
        root / "daily_rank_ic.parquet",
        root / "rank_ic_summary.csv",
        root / "daily_layer_correlations.parquet",
        root / "layer_correlation_summary.csv",
        root / "csi300_universe_audit.csv",
        root / "manifest.json",
    ]
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(f"沪深300评价产物缺失: {path}")
    manifest = json.loads(required[-1].read_text(encoding="utf-8"))
    if manifest.get("schema_version") != CSI_EVALUATION_SCHEMA:
        raise ValueError("沪深300评价schema不匹配")
    if tuple(manifest.get("modeled_layers", [])) != MODELED_LAYERS:
        raise ValueError("沪深300评价层集合不是Layer 1~12")
    factors = pd.read_parquet(
        root / "csi300_stock_day_factors.parquet",
        columns=[
            "factor_source",
            "task_id",
            "split",
            "trading_date",
            "symbol",
            "csi300_snapshot_date",
            "csi300_member",
        ],
    )
    if not factors["csi300_member"].all():
        raise ValueError("沪深300评价表混入非成分股")
    if (factors["csi300_snapshot_date"] > factors["trading_date"]).any():
        raise ValueError("沪深300评价表使用未来快照")
    rank_ic = pd.read_csv(root / "rank_ic_summary.csv")
    expected_tasks = set(STATIC_TASKS).union({DIRECT_TASK_ID})
    if set(rank_ic["task_id"].astype(str)) != expected_tasks:
        raise ValueError("RankIC任务集合不完整")
    layer_sets = rank_ic.groupby(["task_id", "split"])["layer"].agg(set)
    if not layer_sets.map(lambda value: value == set(MODELED_LAYERS)).all():
        raise ValueError("RankIC未覆盖全部Layer 1~12")
    correlation = pd.read_csv(root / "layer_correlation_summary.csv")
    if (correlation["layer_left"] < 1).any() or (
        correlation["layer_right"] > 12
    ).any():
        raise ValueError("层相关性混入Layer 0或越界层")
    return {
        "csi300_factor_rows": len(factors),
        "rank_ic_rows": len(rank_ic),
        "correlation_rows": len(correlation),
        "tasks": len(expected_tasks),
    }


def plot_static_rank_ic(directory: str | Path, *, split: str = "test"):
    import matplotlib.pyplot as plt

    if split not in {"validation", "test"}:
        raise ValueError("split必须是validation或test")
    root = Path(directory).expanduser().resolve()
    validate_static_csi300_evaluation_outputs(root)
    summary = pd.read_csv(root / "rank_ic_summary.csv")
    selected = summary[summary["split"].eq(split)]
    tasks = sorted(selected["task_id"].astype(str).unique())
    fig, axes = plt.subplots(
        len(tasks), 1, figsize=(9, max(3, 2.6 * len(tasks))), squeeze=False
    )
    for axis, task_id in zip(axes[:, 0], tasks):
        frame = selected[selected["task_id"].eq(task_id)].sort_values("layer")
        axis.plot(frame["layer"], frame["mean_rank_ic"], marker="o")
        axis.axhline(0, color="black", linewidth=0.8, alpha=0.5)
        axis.set(title=task_id, xlabel="Layer", ylabel="Mean RankIC")
        axis.grid(alpha=0.2)
    fig.tight_layout()
    plt.close(fig)
    return fig


def plot_static_layer_correlation(
    directory: str | Path,
    *,
    task_id: str,
    split: str = "test",
):
    import matplotlib.pyplot as plt

    root = Path(directory).expanduser().resolve()
    validate_static_csi300_evaluation_outputs(root)
    summary = pd.read_csv(root / "layer_correlation_summary.csv")
    selected = summary[
        summary["task_id"].eq(task_id) & summary["split"].eq(split)
    ]
    if selected.empty:
        raise ValueError(f"找不到层相关性: task={task_id}, split={split}")
    matrix = np.full((12, 12), np.nan)
    for row in selected.itertuples(index=False):
        left = int(row.layer_left) - 1
        right = int(row.layer_right) - 1
        matrix[left, right] = row.mean_spearman
        matrix[right, left] = row.mean_spearman
    fig, ax = plt.subplots(figsize=(7, 6))
    image = ax.imshow(matrix, vmin=-1, vmax=1, cmap="coolwarm")
    ax.set_xticks(range(12), labels=range(1, 13))
    ax.set_yticks(range(12), labels=range(1, 13))
    ax.set(title=f"{task_id} — {split}", xlabel="Layer", ylabel="Layer")
    fig.colorbar(image, ax=ax, label="Mean daily Spearman")
    fig.tight_layout()
    plt.close(fig)
    return fig


def run_static_pipeline(config: Mapping[str, object]) -> dict[str, Path]:
    """One-call orchestration used by the Run-All notebook."""

    from .layer_probe_continuous import run_fixed_head_analysis_stage
    from .layer_probe_representations import run_representation_stage

    parse_static_protocol(config)
    stages = config.get("stages", {})
    if not isinstance(stages, Mapping):
        raise ValueError("stages配置必须是对象")
    required_stages = {
        "representations",
        "fixed_head",
        "static_targets",
        "direct_return_targets",
        "report_ridge",
        "csi300_rank_ic",
        "layer_correlations",
    }
    if any(not bool(stages.get(name, False)) for name in required_stages):
        disabled = sorted(name for name in required_stages if not stages.get(name, False))
        raise ValueError("完整Run All要求所有阶段开启: " + ", ".join(disabled))
    representation = run_representation_stage(config)
    fixed_head = run_fixed_head_analysis_stage(config)
    targets = run_static_target_stage(config)
    direct_targets = run_static_direct_return_target_stage(config)
    probes = run_static_report_probe_stage(config)
    evaluation = run_static_csi300_evaluation_stage(config)
    return {
        "representations": representation.directory,
        "fixed_head": fixed_head,
        "static_targets": targets,
        "direct_return_targets": direct_targets,
        "report_ridge": probes,
        "csi300_rank_ic": evaluation,
    }
