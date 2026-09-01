"""Strict point-in-time expanding-window probes for continuous report labels.

The legacy/static probe pipeline assigns each target to one train/validation/test
bucket.  That is intentionally simple, but wastes long-maturity FY targets.  This
module keeps the canonical long-form target table intact and constructs annual
validation folds dynamically:

* a fold may train only on reports before the fold start;
* every training label must be known before the fold start;
* validation outcomes used for model selection must be known before the closed
  final-test feature window starts;
* scalers and Ridge models are fitted independently for every task, layer and
  fold using training rows only.

No final-test label is loaded by the validation functions in this module.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from .layer_probe_continuous import (
    DISPERSION_LABEL,
    RESIDUAL_LABEL,
    _atomic_write_once,
    _load_representation_bundle,
    _regression_metrics,
    _reuse_exact_output,
    _run_directory,
    _safe_correlation,
    validate_target_provenance,
)
from .layer_probe_representations import protocol_config_hash, sha256_file


WALK_FORWARD_SCHEMA = "walk_forward_continuous_targets_v1.0"
WALK_FORWARD_PROBE_SCHEMA = "walk_forward_continuous_probe_v1.0"
WALK_FORWARD_ASSOCIATION_SCHEMA = "walk_forward_fixed_head_association_v1.0"


@dataclass(frozen=True)
class WalkForwardFold:
    name: str
    feature_start: pd.Timestamp
    feature_end: pd.Timestamp

    def to_dict(self) -> dict[str, str]:
        return {
            "name": self.name,
            "feature_start": str(self.feature_start.date()),
            "feature_end": str(self.feature_end.date()),
            "training_label_cutoff": str(
                (self.feature_start - pd.Timedelta(days=1)).date()
            ),
        }


@dataclass(frozen=True)
class WalkForwardProtocol:
    history_start: pd.Timestamp
    validation_start: pd.Timestamp
    validation_end: pd.Timestamp
    selection_label_cutoff: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    test_label_cutoff: pd.Timestamp
    folds: tuple[WalkForwardFold, ...]
    minimum_train_rows: int
    minimum_evaluation_rows: int
    minimum_validation_folds: int

    def to_dict(self) -> dict[str, object]:
        return {
            "history_start": str(self.history_start.date()),
            "validation_start": str(self.validation_start.date()),
            "validation_end": str(self.validation_end.date()),
            "selection_label_cutoff": str(self.selection_label_cutoff.date()),
            "test_start": str(self.test_start.date()),
            "test_end": str(self.test_end.date()),
            "test_label_cutoff": str(self.test_label_cutoff.date()),
            "folds": [fold.to_dict() for fold in self.folds],
            "minimum_train_rows": self.minimum_train_rows,
            "minimum_evaluation_rows": self.minimum_evaluation_rows,
            "minimum_validation_folds": self.minimum_validation_folds,
        }


def _date(value: object, *, name: str) -> pd.Timestamp:
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        raise ValueError(f"{name}不是有效日期: {value!r}")
    return pd.Timestamp(parsed).normalize()


def parse_walk_forward_protocol(config: Mapping[str, object]) -> WalkForwardProtocol:
    raw = config.get("walk_forward", {})
    if not isinstance(raw, Mapping) or not bool(raw.get("enabled", False)):
        raise ValueError("walk_forward.enabled必须为true")
    validation = raw.get("validation", {})
    final_test = raw.get("final_test", {})
    if not isinstance(validation, Mapping) or not isinstance(final_test, Mapping):
        raise ValueError("walk_forward.validation/final_test必须是对象")

    history_start = _date(raw.get("history_start"), name="walk_forward.history_start")
    validation_start = _date(
        validation.get("feature_start"), name="walk_forward.validation.feature_start"
    )
    validation_end = _date(
        validation.get("feature_end"), name="walk_forward.validation.feature_end"
    )
    selection_cutoff = _date(
        validation.get("selection_label_cutoff"),
        name="walk_forward.validation.selection_label_cutoff",
    )
    test_start = _date(
        final_test.get("feature_start"), name="walk_forward.final_test.feature_start"
    )
    test_end = _date(
        final_test.get("feature_end"), name="walk_forward.final_test.feature_end"
    )
    test_cutoff = _date(
        final_test.get("label_cutoff"),
        name="walk_forward.final_test.label_cutoff",
    )
    if not history_start < validation_start <= validation_end < test_start <= test_end:
        raise ValueError("walk-forward特征日期必须满足history < validation < test")
    if selection_cutoff >= test_start:
        raise ValueError("validation选择所用Label必须在final test开始前可知")
    if selection_cutoff < validation_start:
        raise ValueError("selection_label_cutoff早于validation开始，无法评价")
    if test_cutoff < test_end:
        raise ValueError("final test label_cutoff不能早于test特征结束日")
    if str(validation.get("fold_frequency", "year")) != "year":
        raise ValueError("当前walk-forward协议只支持年度fold")

    folds: list[WalkForwardFold] = []
    for year in range(validation_start.year, validation_end.year + 1):
        start = max(validation_start, pd.Timestamp(year=year, month=1, day=1))
        end = min(validation_end, pd.Timestamp(year=year, month=12, day=31))
        if start <= end:
            folds.append(WalkForwardFold(f"validation_{year}", start, end))
    minimum_validation_folds = int(validation.get("minimum_folds", 2))
    if minimum_validation_folds <= 0 or len(folds) < minimum_validation_folds:
        raise ValueError("walk-forward validation fold数量不足")
    minimum_train_rows = int(raw.get("minimum_train_rows", 100))
    minimum_evaluation_rows = int(raw.get("minimum_evaluation_rows", 100))
    if minimum_train_rows < 2 or minimum_evaluation_rows < 2:
        raise ValueError("walk-forward最小训练/评价样本数必须至少为2")
    return WalkForwardProtocol(
        history_start=history_start,
        validation_start=validation_start,
        validation_end=validation_end,
        selection_label_cutoff=selection_cutoff,
        test_start=test_start,
        test_end=test_end,
        test_label_cutoff=test_cutoff,
        folds=tuple(folds),
        minimum_train_rows=minimum_train_rows,
        minimum_evaluation_rows=minimum_evaluation_rows,
        minimum_validation_folds=minimum_validation_folds,
    )


def _target_paths(config: Mapping[str, object]) -> tuple[Path, Path, Path]:
    target_cfg = config.get("continuous_targets", {})
    if not isinstance(target_cfg, Mapping):
        raise ValueError("continuous_targets配置必须是对象")
    root = Path(str(target_cfg.get("bundle_directory", ""))).expanduser().resolve()
    return (
        root / "probe_targets.parquet",
        root / "probe_dataset_metadata.json",
        root / "probe_merge_audit.csv",
    )


def _validate_unsplit_targets(
    targets: pd.DataFrame, metadata: Mapping[str, object]
) -> None:
    required = {
        "sample_id",
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
        raise ValueError("walk-forward Target缺少字段: " + ", ".join(missing))
    if targets["sample_id"].duplicated().any():
        raise ValueError("walk-forward Target的sample_id不唯一")
    if targets.duplicated(["report_id", "task_id"]).any():
        raise ValueError("walk-forward Target的(report_id,task_id)不唯一")
    if not targets["split"].astype(str).eq("unassigned").all():
        raise ValueError("walk-forward bundle必须关闭静态split并保留全部有效Target")
    if metadata.get("splits") not in ([], None):
        raise ValueError("walk-forward bundle metadata不得含静态时间切分")
    labels = pd.to_numeric(targets["label_value"], errors="coerce")
    weights = pd.to_numeric(targets["target_weight"], errors="coerce")
    if (
        not np.isfinite(labels.to_numpy(dtype=float)).all()
        or not np.isfinite(weights.to_numpy(dtype=float)).all()
        or weights.lt(0).any()
    ):
        raise ValueError("walk-forward Target含非有限Label或无效权重")
    feature = pd.to_datetime(targets["feature_available_date"], errors="coerce")
    available = pd.to_datetime(targets["label_available_date"], errors="coerce")
    if feature.isna().any() or available.isna().any() or available.le(feature).any():
        raise ValueError("walk-forward Target违反Label严格晚于feature的PIT约束")
    unknown = sorted(
        set(targets["label_name"].astype(str)).difference(
            {RESIDUAL_LABEL, DISPERSION_LABEL}
        )
    )
    if unknown:
        raise ValueError("walk-forward Target含未知Label: " + ", ".join(unknown))
    versions = set(targets["label_version"].dropna().astype(str))
    if versions != {str(metadata.get("label_version"))}:
        raise ValueError("walk-forward Target与bundle label_version不一致")


def fold_masks(
    task: pd.DataFrame,
    fold: WalkForwardFold,
    protocol: WalkForwardProtocol,
) -> tuple[pd.Series, pd.Series]:
    """Return strictly PIT training and outcome-blind evaluation membership."""

    feature = pd.to_datetime(task["feature_available_date"], errors="coerce")
    available = pd.to_datetime(task["label_available_date"], errors="coerce")
    weight = pd.to_numeric(task["target_weight"], errors="coerce")
    positive = np.isfinite(weight) & weight.gt(0)
    train = (
        feature.ge(protocol.history_start)
        & feature.lt(fold.feature_start)
        & available.lt(fold.feature_start)
        & positive
    )
    evaluation = (
        feature.between(fold.feature_start, fold.feature_end, inclusive="both")
        & available.le(protocol.selection_label_cutoff)
        & positive
    )
    return train, evaluation


def align_walk_forward_targets(config: Mapping[str, object]) -> Path:
    """Align validation-era targets without loading final-test feature rows."""

    protocol = parse_walk_forward_protocol(config)
    _, report_metadata, _, rep_manifest, _ = _load_representation_bundle(config)
    targets_path, metadata_path, audit_path = _target_paths(config)
    for path in (targets_path, metadata_path, audit_path):
        if not path.is_file():
            raise FileNotFoundError(f"walk-forward bundle缺少文件: {path}")
    target_manifest = json.loads(metadata_path.read_text(encoding="utf-8"))
    validate_target_provenance(target_manifest)
    targets = pd.read_parquet(
        targets_path,
        filters=[
            ("feature_available_date", ">=", protocol.history_start),
            ("feature_available_date", "<=", protocol.validation_end),
        ],
    )
    _validate_unsplit_targets(targets, target_manifest)
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
            "walk-forward Target无法覆盖representation: " + ",".join(missing.astype(str))
        )
    aligned = aligned.drop(columns="_merge")
    aligned["representation_row"] = aligned["representation_row"].astype(np.int64)
    aligned["fold"] = pd.Series(pd.NA, index=aligned.index, dtype="string")
    for fold in protocol.folds:
        membership = pd.to_datetime(aligned["feature_available_date"]).between(
            fold.feature_start, fold.feature_end, inclusive="both"
        )
        aligned.loc[membership, "fold"] = fold.name
    aligned = aligned.sort_values(
        ["task_id", "feature_available_date", "report_id"]
    ).reset_index(drop=True)
    counts: list[dict[str, object]] = []
    for task_id, task in aligned.groupby("task_id", sort=True):
        for fold in protocol.folds:
            train, evaluation = fold_masks(task, fold, protocol)
            counts.append(
                {
                    "task_id": task_id,
                    "fold": fold.name,
                    "training_rows": int(train.sum()),
                    "evaluation_rows": int(evaluation.sum()),
                    "training_label_cutoff": str(
                        (fold.feature_start - pd.Timedelta(days=1)).date()
                    ),
                    "selection_label_cutoff": str(
                        protocol.selection_label_cutoff.date()
                    ),
                }
            )
    output = _run_directory(config) / "walk_forward_targets" / "validation"
    source_fingerprint = hashlib.sha256(
        (
            sha256_file(targets_path)
            + sha256_file(metadata_path)
            + rep_manifest["representation_fingerprint"]
            + json.dumps(protocol.to_dict(), sort_keys=True)
        ).encode("utf-8")
    ).hexdigest()
    expected = {
        "schema_version": WALK_FORWARD_SCHEMA,
        "alignment_fingerprint": source_fingerprint,
        "representation_fingerprint": rep_manifest["representation_fingerprint"],
        "config_sha256": protocol_config_hash(config),
    }
    if _reuse_exact_output(
        output,
        expected_manifest=expected,
        validator=validate_walk_forward_targets,
    ):
        return output
    source_audit = pd.read_csv(audit_path)
    source_audit.insert(0, "audit_source", "probe_bundle")
    return _atomic_write_once(
        output,
        tables={
            "aligned_walk_forward_targets.parquet": aligned,
            "walk_forward_fold_counts.csv": pd.DataFrame(counts),
            "target_source_audit.csv": source_audit,
        },
        manifest={
            **expected,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "target_source_sha256": sha256_file(targets_path),
            "target_manifest_sha256": sha256_file(metadata_path),
            "label_version": target_manifest.get("label_version"),
            "protocol": protocol.to_dict(),
            "target_selection": target_manifest.get("target_selection"),
            "final_test_rows_loaded": False,
            "join_contract": (
                "targets many-to-one representation metadata on "
                "report_id+feature_available_date"
            ),
        },
    )


def validate_walk_forward_targets(directory: str | Path) -> dict[str, object]:
    root = Path(directory).expanduser().resolve()
    required = [
        root / "aligned_walk_forward_targets.parquet",
        root / "walk_forward_fold_counts.csv",
        root / "target_source_audit.csv",
        root / "manifest.json",
    ]
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(f"walk-forward对齐产物缺失: {path}")
    manifest = json.loads(required[-1].read_text(encoding="utf-8"))
    if manifest.get("schema_version") != WALK_FORWARD_SCHEMA:
        raise ValueError("walk-forward对齐协议版本不匹配")
    if manifest.get("final_test_rows_loaded") is not False:
        raise ValueError("validation对齐产物不得加载final-test行")
    targets = pd.read_parquet(
        required[0],
        columns=[
            "sample_id",
            "report_id",
            "task_id",
            "representation_row",
            "feature_available_date",
        ],
    )
    if targets["sample_id"].duplicated().any():
        raise ValueError("walk-forward对齐后sample_id不唯一")
    if targets["representation_row"].isna().any():
        raise ValueError("walk-forward对齐后representation_row为空")
    protocol = manifest["protocol"]
    if pd.to_datetime(targets["feature_available_date"]).max() > pd.Timestamp(
        protocol["validation_end"]
    ):
        raise ValueError("validation对齐产物混入final-test feature")
    return {
        "rows": len(targets),
        "tasks": int(targets["task_id"].nunique()),
        "folds": len(protocol["folds"]),
    }


def _aligned_validation_targets(
    config: Mapping[str, object],
) -> tuple[pd.DataFrame, dict[str, object]]:
    root = align_walk_forward_targets(config)
    return (
        pd.read_parquet(root / "aligned_walk_forward_targets.parquet"),
        json.loads((root / "manifest.json").read_text(encoding="utf-8")),
    )


def _association_metrics(frame: pd.DataFrame) -> dict[str, object]:
    x = frame["logit_margin_1_minus_0"].to_numpy(dtype=float)
    y = frame["label_value"].to_numpy(dtype=float)
    weights = frame["target_weight"].to_numpy(dtype=float)
    direction = (x != 0) & (y != 0)
    return {
        "n": int(len(frame)),
        "effective_n": int(np.sum(weights > 0)),
        "weight_sum": float(np.sum(weights)),
        "spearman": _safe_correlation(x, y, kind="spearman"),
        "pearson": _safe_correlation(x, y, kind="pearson"),
        "direction_n": int(direction.sum()),
        "direction_accuracy": (
            float(np.mean(np.sign(x[direction]) == np.sign(y[direction])))
            if direction.any()
            else np.nan
        ),
    }


def run_walk_forward_fixed_head_stage(config: Mapping[str, object]) -> Path:
    """Scheme A+ association on selection-known validation outcomes; no fit."""

    protocol = parse_walk_forward_protocol(config)
    targets, target_manifest = _aligned_validation_targets(config)
    _, _, head, rep_manifest, _ = _load_representation_bundle(config)
    evaluation = targets[
        pd.to_datetime(targets["feature_available_date"]).between(
            protocol.validation_start, protocol.validation_end, inclusive="both"
        )
        & pd.to_datetime(targets["label_available_date"]).le(
            protocol.selection_label_cutoff
        )
    ].copy()
    merged = evaluation.merge(
        head,
        on="representation_row",
        how="inner",
        validate="many_to_many",
    )
    if len(merged) != len(evaluation) * 13:
        raise RuntimeError("walk-forward Scheme A+未完整连接13层固定头")
    records: list[dict[str, object]] = []
    for (task_id, fold, layer), frame in merged.groupby(
        ["task_id", "fold", "layer"], sort=True
    ):
        records.append(
            {
                "task_id": task_id,
                "label_name": frame["label_name"].iloc[0],
                "fold": fold,
                "layer": int(layer),
                "scope": "fold",
                **_association_metrics(frame),
            }
        )
    for (task_id, layer), frame in merged.groupby(["task_id", "layer"], sort=True):
        records.append(
            {
                "task_id": task_id,
                "label_name": frame["label_name"].iloc[0],
                "fold": "pooled_validation",
                "layer": int(layer),
                "scope": "pooled",
                **_association_metrics(frame),
            }
        )
    metrics = pd.DataFrame(records)
    expected_tasks = set(targets["task_id"].astype(str).unique())
    pooled = metrics[metrics["scope"].eq("pooled")]
    actual_tasks = set(pooled["task_id"].astype(str).unique())
    layer_sets = pooled.groupby("task_id")["layer"].agg(set)
    if actual_tasks != expected_tasks or not layer_sets.map(
        lambda values: values == set(range(13))
    ).all():
        missing = sorted(expected_tasks.difference(actual_tasks))
        raise ValueError(
            "walk-forward Scheme A+未覆盖全部任务或13层: "
            f"missing_tasks={missing}"
        )
    output = _run_directory(config) / "walk_forward_fixed_head_label" / "validation"
    expected = {
        "schema_version": WALK_FORWARD_ASSOCIATION_SCHEMA,
        "representation_fingerprint": rep_manifest["representation_fingerprint"],
        "config_sha256": protocol_config_hash(config),
        "target_alignment_fingerprint": target_manifest["alignment_fingerprint"],
    }
    if _reuse_exact_output(
        output,
        expected_manifest=expected,
        validator=validate_walk_forward_fixed_head_outputs,
    ):
        return output
    return _atomic_write_once(
        output,
        tables={"walk_forward_fixed_head_metrics.csv": metrics},
        manifest={
            **expected,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "training": "none",
            "evaluation_label_cutoff": str(protocol.selection_label_cutoff.date()),
            "final_test_rows_loaded": False,
        },
    )


def validate_walk_forward_fixed_head_outputs(
    directory: str | Path,
) -> dict[str, object]:
    root = Path(directory).expanduser().resolve()
    metrics_path = root / "walk_forward_fixed_head_metrics.csv"
    manifest_path = root / "manifest.json"
    for path in (metrics_path, manifest_path):
        if not path.is_file():
            raise FileNotFoundError(f"walk-forward Scheme A+产物缺失: {path}")
    metrics = pd.read_csv(metrics_path)
    pooled = metrics[metrics["scope"].eq("pooled")]
    layer_sets = pooled.groupby("task_id")["layer"].agg(set)
    if not layer_sets.map(lambda values: values == set(range(13))).all():
        raise ValueError("walk-forward Scheme A+部分任务没有完整13层")
    return {
        "metric_rows": len(metrics),
        "tasks": int(pooled["task_id"].nunique()),
    }


def _positive_rows(frame: pd.DataFrame) -> pd.DataFrame:
    label = pd.to_numeric(frame["label_value"], errors="coerce")
    weight = pd.to_numeric(frame["target_weight"], errors="coerce")
    return frame[
        np.isfinite(label)
        & np.isfinite(weight)
        & weight.gt(0)
    ].copy()


def run_walk_forward_probe_stage(
    config: Mapping[str, object],
    *,
    task_ids: Sequence[str] | None = None,
    shard_tag: str | None = None,
) -> Path:
    """Fit annual expanding StandardScaler+Ridge probes on validation folds.

    When ``task_ids``/``shard_tag`` are set the stage processes only those tasks
    and writes a partial shard directory (``validation_shard_<shard_tag>``)
    without the full coverage check; ``merge_walk_forward_probe_shards``
    reassembles shards into the canonical ``validation`` directory so the
    fingerprint matches a single-process run.
    """

    protocol = parse_walk_forward_protocol(config)
    representations, _, _, rep_manifest, _ = _load_representation_bundle(config)
    targets, target_manifest = _aligned_validation_targets(config)
    probe_cfg = config.get("continuous_probe", {})
    if not isinstance(probe_cfg, Mapping):
        raise ValueError("continuous_probe配置必须是对象")
    alphas = sorted(
        {float(value) for value in probe_cfg.get("alpha_grid", [0.1, 1, 10, 100, 1000])}
    )
    if not alphas or min(alphas) < 0:
        raise ValueError("Ridge alpha网格必须非负")
    solver = str(probe_cfg.get("solver", "lsqr"))
    tolerance = float(probe_cfg.get("tolerance", 1e-6))
    maximum_iterations = int(probe_cfg.get("maximum_iterations", 10000))
    if tolerance <= 0 or maximum_iterations <= 0:
        raise ValueError("Ridge tolerance/maximum_iterations必须为正")

    sharded = shard_tag is not None
    probe_root = _run_directory(config) / "walk_forward_probe"
    output = probe_root / (f"validation_shard_{shard_tag}" if sharded else "validation")
    expected = {
        "schema_version": WALK_FORWARD_PROBE_SCHEMA,
        "representation_fingerprint": rep_manifest["representation_fingerprint"],
        "config_sha256": protocol_config_hash(config),
        "target_alignment_fingerprint": target_manifest["alignment_fingerprint"],
    }
    # Reuse only applies to the canonical (full) output; shards are always
    # recomputed so a partial shard is never mistaken for a complete one.
    if not sharded and _reuse_exact_output(
        output,
        expected_manifest=expected,
        validator=validate_walk_forward_probe_outputs,
    ):
        return output

    grouped = list(targets.groupby("task_id", sort=True))
    if task_ids is not None:
        wanted = {str(value) for value in task_ids}
        grouped = [
            (task_id, frame) for task_id, frame in grouped if str(task_id) in wanted
        ]
        missing = sorted(wanted.difference(str(task_id) for task_id, _ in grouped))
        if missing:
            raise ValueError("shard引用了不存在的task_id: " + ", ".join(missing))
    tuning_records: list[dict[str, object]] = []
    selection_records: list[dict[str, object]] = []
    metric_records: list[dict[str, object]] = []
    prediction_frames: list[pd.DataFrame] = []
    eligibility: list[dict[str, object]] = []
    n_tasks = len(grouped)
    for task_index, (task_id, raw_task) in enumerate(grouped, start=1):
        print(
            f"[stage6{f'/{shard_tag}' if sharded else ''}] "
            f"task {task_index}/{n_tasks} {task_id}",
            flush=True,
        )
        task = _positive_rows(raw_task)
        if task.empty:
            eligibility.append(
                {"task_id": task_id, "eligible": False, "reason": "no_positive_weight_rows"}
            )
            continue
        for layer in range(13):
            print(f"[stage6]   {task_id} layer {layer}", flush=True)
            fold_payloads: list[dict[str, object]] = []
            for fold in protocol.folds:
                train_mask, evaluation_mask = fold_masks(task, fold, protocol)
                train = task.loc[train_mask]
                evaluation = task.loc[evaluation_mask]
                reason = None
                if len(train) < protocol.minimum_train_rows:
                    reason = "insufficient_training_rows"
                elif len(evaluation) < protocol.minimum_evaluation_rows:
                    reason = "insufficient_evaluation_rows"
                elif train["label_value"].nunique() < 2:
                    reason = "constant_training_target"
                elif evaluation["label_value"].nunique() < 2:
                    reason = "constant_evaluation_target"
                eligibility.append(
                    {
                        "task_id": task_id,
                        "layer": layer,
                        "fold": fold.name,
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
                x_train = np.asarray(
                    representations[train_positions, layer, :], dtype=np.float32
                )
                x_evaluation = np.asarray(
                    representations[evaluation_positions, layer, :], dtype=np.float32
                )
                y_train = train["label_value"].to_numpy(dtype=float)
                y_evaluation = evaluation["label_value"].to_numpy(dtype=float)
                w_train = train["target_weight"].to_numpy(dtype=float)
                scaler = StandardScaler().fit(x_train, sample_weight=w_train)
                x_train_scaled = scaler.transform(x_train)
                x_evaluation_scaled = scaler.transform(x_evaluation)
                predictions: dict[float, np.ndarray] = {}
                for alpha in alphas:
                    model = Ridge(
                        alpha=alpha,
                        solver=solver,
                        tol=tolerance,
                        max_iter=maximum_iterations,
                    ).fit(x_train_scaled, y_train, sample_weight=w_train)
                    candidate = model.predict(x_evaluation_scaled)
                    predictions[alpha] = candidate
                    score = _safe_correlation(
                        candidate, y_evaluation, kind="spearman"
                    )
                    tuning_records.append(
                        {
                            "task_id": task_id,
                            "label_name": task["label_name"].iloc[0],
                            "layer": layer,
                            "fold": fold.name,
                            "alpha": alpha,
                            "fold_spearman": score,
                            "training_rows": len(train),
                            "evaluation_rows": len(evaluation),
                        }
                    )
                fold_payloads.append(
                    {
                        "fold": fold,
                        "train": train,
                        "evaluation": evaluation,
                        "predictions": predictions,
                    }
                )
            layer_tuning = [
                record
                for record in tuning_records
                if record["task_id"] == task_id and record["layer"] == layer
            ]
            alpha_scores: list[tuple[float, float, int]] = []
            for alpha in alphas:
                scores = np.asarray(
                    [
                        record["fold_spearman"]
                        for record in layer_tuning
                        if record["alpha"] == alpha
                    ],
                    dtype=float,
                )
                scores = scores[np.isfinite(scores)]
                if len(scores) >= protocol.minimum_validation_folds:
                    alpha_scores.append((alpha, float(np.mean(scores)), len(scores)))
            if not alpha_scores:
                eligibility.append(
                    {
                        "task_id": task_id,
                        "layer": layer,
                        "eligible": False,
                        "reason": "insufficient_valid_tuning_folds",
                    }
                )
                continue
            selected_alpha, selected_score, selected_folds = max(
                alpha_scores, key=lambda item: (item[1], item[0])
            )
            selection_records.append(
                {
                    "task_id": task_id,
                    "label_name": task["label_name"].iloc[0],
                    "layer": layer,
                    "selected_alpha": selected_alpha,
                    "mean_fold_spearman": selected_score,
                    "valid_folds": selected_folds,
                }
            )
            selected_layer_predictions: list[pd.DataFrame] = []
            for payload in fold_payloads:
                fold = payload["fold"]
                evaluation = payload["evaluation"].copy()
                prediction = payload["predictions"][selected_alpha]
                y = evaluation["label_value"].to_numpy(dtype=float)
                weights = evaluation["target_weight"].to_numpy(dtype=float)
                metric_records.append(
                    {
                        "task_id": task_id,
                        "label_name": evaluation["label_name"].iloc[0],
                        "layer": layer,
                        "scope": "fold",
                        "fold": fold.name,
                        "selected_alpha": selected_alpha,
                        "training_rows": len(payload["train"]),
                        **_regression_metrics(y, prediction, weights),
                    }
                )
                keep = [
                    column
                    for column in (
                        "sample_id",
                        "report_id",
                        "representation_row",
                        "stock_code",
                        "symbol",
                        "feature_available_date",
                        "label_available_date",
                        "task_id",
                        "label_name",
                        "label_value",
                        "target_weight",
                        "forecast_horizon",
                        "confirmation_months",
                        "peer_panel",
                    )
                    if column in evaluation.columns
                ]
                frame = evaluation[keep].copy()
                frame["fold"] = fold.name
                frame["layer"] = layer
                frame["prediction"] = prediction.astype(np.float32, copy=False)
                frame["selected_alpha"] = selected_alpha
                frame["prediction_role"] = "oos"
                selected_layer_predictions.append(frame)
            pooled = pd.concat(selected_layer_predictions, ignore_index=True)
            metric_records.append(
                {
                    "task_id": task_id,
                    "label_name": pooled["label_name"].iloc[0],
                    "layer": layer,
                    "scope": "pooled",
                    "fold": "pooled_validation",
                    "selected_alpha": selected_alpha,
                    "training_rows": pd.NA,
                    **_regression_metrics(
                        pooled["label_value"].to_numpy(dtype=float),
                        pooled["prediction"].to_numpy(dtype=float),
                        pooled["target_weight"].to_numpy(dtype=float),
                    ),
                }
            )
            prediction_frames.extend(selected_layer_predictions)
    if not prediction_frames:
        if sharded:
            raise ValueError(f"shard {shard_tag} 没有生成walk-forward OOS预测")
        raise ValueError("没有任务生成walk-forward OOS预测")
    predictions = pd.concat(prediction_frames, ignore_index=True)
    metrics = pd.DataFrame(metric_records)
    selections = pd.DataFrame(selection_records)
    if not sharded:
        expected_tasks = set(targets["task_id"].astype(str).unique())
        actual_tasks = set(selections["task_id"].astype(str).unique())
        layer_sets = selections.groupby("task_id")["layer"].agg(set)
        if actual_tasks != expected_tasks or not layer_sets.map(
            lambda values: values == set(range(13))
        ).all():
            missing = sorted(expected_tasks.difference(actual_tasks))
            raise ValueError(
                "walk-forward probe未覆盖全部任务或13层: "
                f"missing_tasks={missing}"
            )
    print(
        f"[stage6{f'/{shard_tag}' if sharded else ''}] writing {output.name} outputs",
        flush=True,
    )
    return _atomic_write_once(
        output,
        tables={
            "walk_forward_probe_metrics.csv": metrics,
            "walk_forward_oos_predictions.parquet": predictions,
            "walk_forward_tuning.csv": pd.DataFrame(tuning_records),
            "walk_forward_selected_alphas.csv": selections,
            "walk_forward_eligibility.csv": pd.DataFrame(eligibility),
        },
        manifest={
            **expected,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "protocol": protocol.to_dict(),
            "alpha_grid": alphas,
            "alpha_selection": "equal-weight mean annual fold Spearman",
            "alpha_tie_break": "larger_alpha",
            "solver": solver,
            "solver_tolerance": tolerance,
            "solver_maximum_iterations": maximum_iterations,
            "sample_weight": "positive weights; used by StandardScaler and Ridge",
            "target_transform": "none",
            "final_test_rows_loaded": False,
            "partial": sharded,
            "shard_tag": shard_tag,
            "shard_task_ids": (
                [str(task_id) for task_id, _ in grouped] if sharded else None
            ),
            "oos_contract": (
                "for fold Y train feature<Y and label_available<Y; "
                "evaluation outcomes known by selection cutoff"
            ),
        },
    )


def merge_walk_forward_probe_shards(
    config: Mapping[str, object], shard_tags: Sequence[str]
) -> Path:
    """Reassemble per-task shards into the canonical validation directory.

    The merged artifact carries the same reuse fingerprint as a single-process
    run, so downstream stages cannot tell a sharded run from a sequential one.
    """

    parse_walk_forward_protocol(config)
    _, _, _, rep_manifest, _ = _load_representation_bundle(config)
    targets, target_manifest = _aligned_validation_targets(config)
    probe_root = _run_directory(config) / "walk_forward_probe"
    output = probe_root / "validation"
    expected = {
        "schema_version": WALK_FORWARD_PROBE_SCHEMA,
        "representation_fingerprint": rep_manifest["representation_fingerprint"],
        "config_sha256": protocol_config_hash(config),
        "target_alignment_fingerprint": target_manifest["alignment_fingerprint"],
    }
    if _reuse_exact_output(
        output, expected_manifest=expected, validator=validate_walk_forward_probe_outputs
    ):
        return output
    file_names = (
        "walk_forward_probe_metrics.csv",
        "walk_forward_oos_predictions.parquet",
        "walk_forward_tuning.csv",
        "walk_forward_selected_alphas.csv",
        "walk_forward_eligibility.csv",
    )
    parts: dict[str, list[pd.DataFrame]] = {}
    shard_manifests: list[dict[str, object]] = []
    for tag in shard_tags:
        shard_dir = probe_root / f"validation_shard_{tag}"
        for name in file_names:
            path = shard_dir / name
            if not path.is_file():
                raise FileNotFoundError(f"shard {tag} 缺少产物: {path}")
            parts.setdefault(name, []).append(
                pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
            )
        shard_manifests.append(
            json.loads((shard_dir / "manifest.json").read_text(encoding="utf-8"))
        )
    tables = {name: pd.concat(frames, ignore_index=True) for name, frames in parts.items()}
    selections = tables["walk_forward_selected_alphas.csv"]
    expected_tasks = set(targets["task_id"].astype(str).unique())
    actual_tasks = set(selections["task_id"].astype(str).unique())
    layer_sets = selections.groupby("task_id")["layer"].agg(set)
    if actual_tasks != expected_tasks or not layer_sets.map(
        lambda values: values == set(range(13))
    ).all():
        missing = sorted(expected_tasks.difference(actual_tasks))
        raise ValueError(
            "合并后的walk-forward probe未覆盖全部任务或13层: "
            f"missing_tasks={missing}"
        )
    base_manifest = dict(shard_manifests[0])
    for key in ("partial", "shard_tag", "shard_task_ids", "created_at"):
        base_manifest.pop(key, None)
    return _atomic_write_once(
        output,
        tables=tables,
        manifest={
            **base_manifest,
            **expected,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "partial": False,
            "merged_from_shards": list(shard_tags),
            "final_test_rows_loaded": False,
        },
    )


def validate_walk_forward_probe_outputs(directory: str | Path) -> dict[str, object]:
    root = Path(directory).expanduser().resolve()
    required = [
        root / "walk_forward_probe_metrics.csv",
        root / "walk_forward_oos_predictions.parquet",
        root / "walk_forward_tuning.csv",
        root / "walk_forward_selected_alphas.csv",
        root / "walk_forward_eligibility.csv",
        root / "manifest.json",
    ]
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(f"walk-forward probe产物缺失: {path}")
    manifest = json.loads(required[-1].read_text(encoding="utf-8"))
    if manifest.get("schema_version") != WALK_FORWARD_PROBE_SCHEMA:
        raise ValueError("walk-forward probe协议版本不匹配")
    if manifest.get("final_test_rows_loaded") is not False:
        raise ValueError("validation probe不得加载final-test rows")
    metrics = pd.read_csv(required[0])
    predictions = pd.read_parquet(
        required[1],
        columns=["sample_id", "task_id", "fold", "layer", "prediction_role", "prediction"],
    )
    if not predictions["prediction_role"].eq("oos").all():
        raise ValueError("walk-forward预测混入非OOS记录")
    if predictions.duplicated(["sample_id", "layer"]).any():
        raise ValueError("walk-forward同一样本同一层重复预测")
    if not np.isfinite(predictions["prediction"].to_numpy(dtype=float)).all():
        raise ValueError("walk-forward预测含NaN或Inf")
    pooled = metrics[metrics["scope"].eq("pooled")]
    layer_sets = pooled.groupby("task_id")["layer"].agg(set)
    if not layer_sets.map(lambda values: values == set(range(13))).all():
        raise ValueError("walk-forward部分任务没有完整13层")
    return {
        "metric_rows": len(metrics),
        "oos_rows": len(predictions),
        "tasks": int(predictions["task_id"].nunique()),
        "folds": int(predictions["fold"].nunique()),
    }


def plot_walk_forward_probe_curves(
    directory: str | Path, *, metric: str = "spearman"
):
    import matplotlib.pyplot as plt

    data = pd.read_csv(Path(directory) / "walk_forward_probe_metrics.csv")
    data = data[data["scope"].eq("pooled")]
    tasks = sorted(data["task_id"].unique())
    fig, axes = plt.subplots(
        len(tasks),
        1,
        figsize=(9, max(4, 2.6 * len(tasks))),
        squeeze=False,
    )
    for axis, task in zip(axes[:, 0], tasks):
        selected = data[data["task_id"].eq(task)].sort_values("layer")
        axis.plot(selected["layer"], selected[metric], marker="o")
        axis.axhline(0, color="black", linewidth=0.8)
        axis.set(title=task, xlabel="Layer", ylabel=metric)
        axis.grid(alpha=0.2)
    fig.tight_layout()
    return fig


def plot_walk_forward_fixed_head_curves(
    directory: str | Path, *, metric: str = "spearman"
):
    import matplotlib.pyplot as plt

    data = pd.read_csv(Path(directory) / "walk_forward_fixed_head_metrics.csv")
    data = data[data["scope"].eq("pooled")]
    tasks = sorted(data["task_id"].unique())
    fig, axes = plt.subplots(
        len(tasks),
        1,
        figsize=(9, max(4, 2.6 * len(tasks))),
        squeeze=False,
    )
    for axis, task in zip(axes[:, 0], tasks):
        selected = data[data["task_id"].eq(task)].sort_values("layer")
        axis.plot(selected["layer"], selected[metric], marker="o")
        axis.axhline(0, color="black", linewidth=0.8)
        axis.set(title=task, xlabel="Layer", ylabel=metric)
        axis.grid(alpha=0.2)
    fig.tight_layout()
    return fig
