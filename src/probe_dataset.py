"""逐层 Linear Probe 的无前视数据装配。

本模块只负责把唯一研报文本与已经验证的未来 Label 建立可审计关联。
文本按 ``report_id`` 只保存一次；目标表按 ``task_id`` 区分 FY、确认窗口和
同行面板。任何裁剪、标准化或缺失值填充都必须留到训练阶段，并且只能用训练集
拟合参数。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


RESIDUAL_LABEL = "residual_signed_raw"
DISPERSION_LABEL = "delta_log_dispersion"
SPLIT_NAMES = ("train", "validation", "test")


@dataclass(frozen=True)
class SplitWindow:
    """一个按研报可用日定义的时间分区。

    ``label_cutoff`` 是该分区允许使用的最晚 Label 可知日。训练集与验证集必须
    分别早于下一分区开始日，确保模型训练或选择时目标已经真实可知。
    """

    name: str
    feature_start: pd.Timestamp
    feature_end: pd.Timestamp
    label_cutoff: pd.Timestamp | None

    def to_dict(self) -> dict[str, str | None]:
        return {
            "name": self.name,
            "feature_start": str(self.feature_start.date()),
            "feature_end": str(self.feature_end.date()),
            "label_cutoff": (
                str(self.label_cutoff.date()) if self.label_cutoff is not None else None
            ),
        }


@dataclass
class ProbeBundle:
    texts: pd.DataFrame
    targets: pd.DataFrame
    audit: pd.DataFrame
    metadata: dict[str, Any]


def _require_columns(frame: pd.DataFrame, required: set[str], name: str) -> None:
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{name}缺少字段: {', '.join(missing)}")


def _date_series(values: pd.Series, *, name: str) -> pd.Series:
    parsed = pd.to_datetime(values, errors="coerce").dt.normalize()
    if parsed.isna().any():
        raise ValueError(f"{name}含{int(parsed.isna().sum())}个无效日期")
    return parsed


def _parse_date(value: object, *, name: str) -> pd.Timestamp:
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        raise ValueError(f"{name}不是有效日期: {value!r}")
    return pd.Timestamp(parsed).normalize()


def parse_split_windows(config: Mapping[str, object]) -> list[SplitWindow]:
    """解析并验证显式时间分区；关闭时返回空列表，绝不随机切分。"""

    if not bool(config.get("enabled", False)):
        return []

    windows: list[SplitWindow] = []
    for name in SPLIT_NAMES:
        raw = config.get(name)
        if not isinstance(raw, Mapping):
            raise ValueError(f"splits.enabled=true时必须配置splits.{name}")
        start = _parse_date(
            raw.get("feature_start"), name=f"splits.{name}.feature_start"
        )
        end = _parse_date(raw.get("feature_end"), name=f"splits.{name}.feature_end")
        if end < start:
            raise ValueError(f"splits.{name}日期区间倒置")
        cutoff_value = raw.get("label_cutoff")
        cutoff = (
            None
            if cutoff_value is None or str(cutoff_value).strip() == ""
            else _parse_date(cutoff_value, name=f"splits.{name}.label_cutoff")
        )
        if name != "test" and cutoff is None:
            raise ValueError(f"splits.{name}.label_cutoff不能为空")
        windows.append(SplitWindow(name, start, end, cutoff))

    for left, right in zip(windows, windows[1:]):
        if left.feature_end >= right.feature_start:
            raise ValueError(f"splits.{left.name}与splits.{right.name}重叠或顺序错误")
        if left.label_cutoff is None or left.label_cutoff >= right.feature_start:
            raise ValueError(
                f"splits.{left.name}.label_cutoff必须严格早于"
                f"splits.{right.name}.feature_start"
            )
    return windows


def _prepare_reports(reports: pd.DataFrame) -> pd.DataFrame:
    required = {"report_id", "stock_code", "available_date", "text"}
    _require_columns(reports, required, "reports.parquet")
    out = reports.copy()
    out["report_id"] = out["report_id"].astype("string").str.strip()
    if out["report_id"].isna().any() or out["report_id"].eq("").any():
        raise ValueError("reports.parquet含空report_id")
    if out["report_id"].duplicated().any():
        raise ValueError("reports.parquet的report_id不唯一")
    out["stock_code"] = out["stock_code"].astype("string").str.strip()
    if not out["stock_code"].str.fullmatch(r"\d{6}", na=False).all():
        raise ValueError("reports.parquet含非六位股票代码")
    out["feature_available_date"] = _date_series(
        out["available_date"], name="reports.available_date"
    )
    out["text"] = out["text"].fillna("").astype(str).str.strip()
    if out["text"].eq("").any():
        raise ValueError(f"reports.parquet含{int(out['text'].eq('').sum())}篇空文本")
    keep = [
        column
        for column in (
            "report_id",
            "stock_code",
            "org_id",
            "author_name",
            "title",
            "publish_timestamp",
            "publish_date",
            "feature_available_date",
            "text",
        )
        if column in out.columns
    ]
    return (
        out[keep]
        .sort_values(["feature_available_date", "report_id"])
        .reset_index(drop=True)
    )


def _validate_binary_flag(frame: pd.DataFrame, column: str, name: str) -> None:
    values = pd.to_numeric(frame[column], errors="coerce")
    if values.isna().any() or not values.isin([0, 1]).all():
        raise ValueError(f"{name}.{column}必须只包含0/1")


def _assert_invalid_targets_are_empty(
    frame: pd.DataFrame,
    *,
    target: str,
    valid_flag: str,
    name: str,
) -> None:
    invalid = pd.to_numeric(frame[valid_flag], errors="coerce").eq(0)
    leaked = invalid & frame[target].notna()
    if leaked.any():
        raise ValueError(
            f"{name}有{int(leaked.sum())}条无效样本仍保留{target}，拒绝合并"
        )


def _validate_source_keys(
    fy_labels: pd.DataFrame, confirmation_labels: pd.DataFrame
) -> None:
    _require_columns(
        fy_labels,
        {
            "report_id",
            "stock_code",
            "available_date",
            "fy",
            "forecast_horizon",
            "residual_signed_raw",
            "residual_valid",
            "actual_label_available_date",
            "actual_invalid_reason",
            "label_version",
        },
        "report_fy_labels.parquet",
    )
    _require_columns(
        confirmation_labels,
        {
            "report_id",
            "stock_code",
            "available_date",
            "fy",
            "forecast_horizon",
            "confirmation_months",
            "peer_panel",
            "target_date",
            "delta_log_dispersion",
            "confirmation_valid",
            "confirmation_probe_valid",
            "sample_weight",
            "label_available_date",
            "invalid_reason",
            "probe_invalid_reason",
            "label_version",
        },
        "report_confirmation_labels.parquet",
    )
    if fy_labels.duplicated(["report_id", "fy"]).any():
        raise ValueError("report_fy_labels.parquet的(report_id, fy)不唯一")
    confirmation_key = ["report_id", "fy", "confirmation_months", "peer_panel"]
    if confirmation_labels.duplicated(confirmation_key).any():
        raise ValueError(
            "report_confirmation_labels.parquet的"
            "(report_id, fy, confirmation_months, peer_panel)不唯一"
        )
    _validate_binary_flag(fy_labels, "residual_valid", "report_fy_labels")
    _validate_binary_flag(
        confirmation_labels, "confirmation_valid", "report_confirmation_labels"
    )
    _validate_binary_flag(
        confirmation_labels,
        "confirmation_probe_valid",
        "report_confirmation_labels",
    )
    probe_without_valid = confirmation_labels["confirmation_probe_valid"].eq(
        1
    ) & ~confirmation_labels["confirmation_valid"].eq(1)
    if probe_without_valid.any():
        raise ValueError("confirmation_probe_valid=1不能出现在confirmation_valid=0样本")
    _assert_invalid_targets_are_empty(
        fy_labels,
        target=RESIDUAL_LABEL,
        valid_flag="residual_valid",
        name="report_fy_labels.parquet",
    )
    _assert_invalid_targets_are_empty(
        confirmation_labels,
        target=DISPERSION_LABEL,
        valid_flag="confirmation_valid",
        name="report_confirmation_labels.parquet",
    )


def _versions(frame: pd.DataFrame) -> set[str]:
    return set(frame["label_version"].dropna().astype(str).str.strip())


def _attach_report_identity(
    targets: pd.DataFrame, reports: pd.DataFrame, *, source_name: str
) -> pd.DataFrame:
    identity = reports[["report_id", "stock_code", "feature_available_date"]]
    merged = targets.merge(
        identity,
        on="report_id",
        how="left",
        validate="many_to_one",
        indicator=True,
        suffixes=("_label", "_report"),
    )
    missing = merged["_merge"].ne("both")
    if missing.any():
        raise ValueError(
            f"{source_name}有{int(missing.sum())}条report_id无法连接reports.parquet"
        )
    label_stock = merged["stock_code_label"].astype("string").str.strip()
    report_stock = merged["stock_code_report"].astype("string").str.strip()
    if not label_stock.eq(report_stock).all():
        raise ValueError(f"{source_name}与reports.parquet的stock_code不一致")
    label_date = _date_series(
        merged["available_date"], name=f"{source_name}.available_date"
    )
    if not label_date.eq(merged["feature_available_date"]).all():
        raise ValueError(f"{source_name}与reports.parquet的available_date不一致")
    merged["stock_code"] = report_stock
    return merged.drop(
        columns=[
            "stock_code_label",
            "stock_code_report",
            "available_date",
            "_merge",
        ]
    )


def _filter_values(values: pd.Series, allowed: Sequence[object]) -> pd.Series:
    return (
        values.isin(list(allowed)) if allowed else pd.Series(True, index=values.index)
    )


def _reason_audit(
    frame: pd.DataFrame,
    *,
    mask: pd.Series,
    column: str,
    label_name: str,
    stage: str,
) -> list[dict[str, object]]:
    reasons = frame.loc[mask, column].astype("string").fillna("<missing_reason>")
    return [
        {
            "label_name": label_name,
            "stage": stage,
            "reason": str(reason),
            "count": int(count),
        }
        for reason, count in reasons.value_counts(dropna=False).items()
    ]


def _residual_targets(
    fy_labels: pd.DataFrame,
    reports: pd.DataFrame,
    *,
    forecast_horizons: Sequence[int],
) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    audit = [
        {
            "label_name": RESIDUAL_LABEL,
            "stage": "source_rows",
            "reason": "all",
            "count": len(fy_labels),
        },
        {
            "label_name": RESIDUAL_LABEL,
            "stage": "source_valid",
            "reason": "residual_valid",
            "count": int(fy_labels["residual_valid"].eq(1).sum()),
        },
    ]
    audit.extend(
        _reason_audit(
            fy_labels,
            mask=fy_labels["residual_valid"].eq(0),
            column="actual_invalid_reason",
            label_name=RESIDUAL_LABEL,
            stage="source_invalid_reason",
        )
    )
    selected = fy_labels[
        fy_labels["residual_valid"].eq(1)
        & _filter_values(fy_labels["forecast_horizon"], forecast_horizons)
    ].copy()
    selected["label_value"] = pd.to_numeric(selected[RESIDUAL_LABEL], errors="coerce")
    if not np.isfinite(selected["label_value"].to_numpy(dtype=float)).all():
        raise ValueError("residual_valid=1样本含非有限residual_signed_raw")
    selected["label_available_date"] = _date_series(
        selected["actual_label_available_date"],
        name="report_fy_labels.actual_label_available_date",
    )
    selected = _attach_report_identity(
        selected, reports, source_name="report_fy_labels.parquet"
    )
    selected["label_name"] = RESIDUAL_LABEL
    selected["target_weight"] = 1.0
    selected["confirmation_months"] = pd.Series(
        pd.NA, index=selected.index, dtype="Int64"
    )
    selected["peer_panel"] = pd.Series(pd.NA, index=selected.index, dtype="string")
    selected["confirmation_probe_valid"] = pd.Series(
        pd.NA, index=selected.index, dtype="Int64"
    )
    selected["task_id"] = (
        RESIDUAL_LABEL
        + "__fh"
        + pd.to_numeric(selected["forecast_horizon"]).astype(int).astype(str)
    )
    selected["sample_id"] = (
        RESIDUAL_LABEL
        + "|"
        + selected["report_id"].astype(str)
        + "|fy="
        + pd.to_numeric(selected["fy"]).astype(int).astype(str)
    )
    audit.append(
        {
            "label_name": RESIDUAL_LABEL,
            "stage": "configured_selection",
            "reason": "valid_and_selected_horizon",
            "count": len(selected),
        }
    )
    return selected, audit


def _dispersion_targets(
    confirmation: pd.DataFrame,
    reports: pd.DataFrame,
    *,
    forecast_horizons: Sequence[int],
    confirmation_months: Sequence[int],
    peer_panels: Sequence[str],
    require_probe_valid: bool,
) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    validity_column = (
        "confirmation_probe_valid" if require_probe_valid else "confirmation_valid"
    )
    audit = [
        {
            "label_name": DISPERSION_LABEL,
            "stage": "source_rows",
            "reason": "all",
            "count": len(confirmation),
        },
        {
            "label_name": DISPERSION_LABEL,
            "stage": "source_valid",
            "reason": "confirmation_valid",
            "count": int(confirmation["confirmation_valid"].eq(1).sum()),
        },
        {
            "label_name": DISPERSION_LABEL,
            "stage": "source_probe_valid",
            "reason": "confirmation_probe_valid",
            "count": int(confirmation["confirmation_probe_valid"].eq(1).sum()),
        },
    ]
    audit.extend(
        _reason_audit(
            confirmation,
            mask=confirmation["confirmation_valid"].eq(0),
            column="invalid_reason",
            label_name=DISPERSION_LABEL,
            stage="source_invalid_reason",
        )
    )
    audit.extend(
        _reason_audit(
            confirmation,
            mask=confirmation["confirmation_valid"].eq(1)
            & confirmation["confirmation_probe_valid"].eq(0),
            column="probe_invalid_reason",
            label_name=DISPERSION_LABEL,
            stage="source_probe_invalid_reason",
        )
    )
    selected = confirmation[
        confirmation[validity_column].eq(1)
        & _filter_values(confirmation["forecast_horizon"], forecast_horizons)
        & _filter_values(confirmation["confirmation_months"], confirmation_months)
        & _filter_values(confirmation["peer_panel"], peer_panels)
    ].copy()
    selected["label_value"] = pd.to_numeric(selected[DISPERSION_LABEL], errors="coerce")
    if not np.isfinite(selected["label_value"].to_numpy(dtype=float)).all():
        raise ValueError(f"{validity_column}=1样本含非有限delta_log_dispersion")
    selected["label_available_date"] = _date_series(
        selected["label_available_date"],
        name="report_confirmation_labels.label_available_date",
    )
    target_date = _date_series(
        selected["target_date"], name="report_confirmation_labels.target_date"
    )
    if not target_date.eq(selected["label_available_date"]).all():
        raise ValueError("Confirmation的label_available_date必须等于target_date")
    selected["target_weight"] = pd.to_numeric(
        selected["sample_weight"], errors="coerce"
    )
    weights = selected["target_weight"].to_numpy(dtype=float)
    if not np.isfinite(weights).all() or (weights < 0).any():
        raise ValueError("Confirmation有效样本含无效sample_weight")
    selected = _attach_report_identity(
        selected, reports, source_name="report_confirmation_labels.parquet"
    )
    selected["label_name"] = DISPERSION_LABEL
    selected["task_id"] = (
        DISPERSION_LABEL
        + "__"
        + pd.to_numeric(selected["confirmation_months"]).astype(int).astype(str)
        + "m__"
        + selected["peer_panel"].astype(str)
        + "__fh"
        + pd.to_numeric(selected["forecast_horizon"]).astype(int).astype(str)
    )
    selected["sample_id"] = (
        DISPERSION_LABEL
        + "|"
        + selected["report_id"].astype(str)
        + "|fy="
        + pd.to_numeric(selected["fy"]).astype(int).astype(str)
        + "|m="
        + pd.to_numeric(selected["confirmation_months"]).astype(int).astype(str)
        + "|panel="
        + selected["peer_panel"].astype(str)
    )
    audit.append(
        {
            "label_name": DISPERSION_LABEL,
            "stage": "configured_selection",
            "reason": validity_column,
            "count": len(selected),
        }
    )
    return selected, audit


def _enforce_label_after_feature(targets: pd.DataFrame) -> None:
    invalid = targets["label_available_date"] <= targets["feature_available_date"]
    if invalid.any():
        examples = targets.loc[
            invalid,
            ["sample_id", "feature_available_date", "label_available_date"],
        ].head(3)
        raise ValueError(
            "存在Label在研报可用日当日或之前可知的样本，拒绝潜在PIT:\n"
            + examples.to_string(index=False)
        )


def _assign_splits(
    targets: pd.DataFrame, windows: Sequence[SplitWindow]
) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    out = targets.copy()
    audit: list[dict[str, object]] = []
    if not windows:
        out["split"] = "unassigned"
        audit.append(
            {
                "label_name": "all",
                "stage": "split",
                "reason": "unassigned_split_disabled",
                "count": len(out),
            }
        )
        return out, audit

    split = pd.Series(pd.NA, index=out.index, dtype="string")
    feature_in_any = pd.Series(False, index=out.index)
    label_too_late = pd.Series(False, index=out.index)
    for window in windows:
        feature_mask = out["feature_available_date"].between(
            window.feature_start, window.feature_end, inclusive="both"
        )
        feature_in_any |= feature_mask
        eligible = feature_mask
        if window.label_cutoff is not None:
            known = out["label_available_date"] <= window.label_cutoff
            label_too_late |= feature_mask & ~known
            eligible &= known
        split.loc[eligible] = window.name
        audit.append(
            {
                "label_name": "all",
                "stage": "split",
                "reason": f"included_{window.name}",
                "count": int(eligible.sum()),
            }
        )
    audit.extend(
        [
            {
                "label_name": "all",
                "stage": "split_excluded",
                "reason": "outside_feature_windows",
                "count": int((~feature_in_any).sum()),
            },
            {
                "label_name": "all",
                "stage": "split_excluded",
                "reason": "label_not_known_by_split_cutoff",
                "count": int(label_too_late.sum()),
            },
        ]
    )
    out["split"] = split
    return out[out["split"].notna()].reset_index(drop=True), audit


def _target_columns(targets: pd.DataFrame) -> list[str]:
    preferred = [
        "sample_id",
        "task_id",
        "report_id",
        "stock_code",
        "feature_available_date",
        "label_available_date",
        "split",
        "fy",
        "forecast_horizon",
        "label_name",
        "label_value",
        "target_weight",
        "confirmation_months",
        "peer_panel",
        "residual_valid",
        "confirmation_valid",
        "confirmation_probe_valid",
        "actual_known_date",
        "actual_publish_date",
        "target_date",
        "n_org_pre",
        "n_org_future",
        "n_peer_updates",
        "dispersion_pre",
        "dispersion_future",
        "scale_t",
        "actual_invalid_reason",
        "invalid_reason",
        "probe_invalid_reason",
        "label_version",
    ]
    return [column for column in preferred if column in targets.columns]


def _target_enabled(value: object, *, name: str) -> bool:
    if isinstance(value, Mapping):
        return bool(value.get("enabled", True))
    if isinstance(value, bool):
        return value
    if value is None:
        return True
    raise ValueError(f"targets.{name}必须是布尔值或配置对象")


def _task_split_metadata(
    targets: pd.DataFrame, *, windows_enabled: bool
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[str]]:
    grouped = targets.groupby(["task_id", "split"], sort=True, dropna=False)
    task_counts: list[dict[str, object]] = []
    for (task_id, split), group in grouped:
        weights = pd.to_numeric(group["target_weight"], errors="raise")
        task_counts.append(
            {
                "task_id": str(task_id),
                "split": str(split),
                "rows": int(len(group)),
                "effective_rows": int(weights.gt(0).sum()),
                "weight_sum": float(weights.sum()),
            }
        )

    readiness: list[dict[str, object]] = []
    ready_tasks: list[str] = []
    for task_id in sorted(targets["task_id"].astype(str).unique()):
        records = [record for record in task_counts if record["task_id"] == task_id]
        effective_by_split = {
            str(record["split"]): int(record["effective_rows"]) for record in records
        }
        missing = [
            split for split in SPLIT_NAMES if effective_by_split.get(split, 0) == 0
        ]
        is_ready = windows_enabled and not missing
        readiness.append(
            {
                "task_id": task_id,
                "training_ready": is_ready,
                "missing_or_zero_weight_splits": missing,
            }
        )
        if is_ready:
            ready_tasks.append(task_id)
    return task_counts, readiness, ready_tasks


def build_probe_bundle(
    reports: pd.DataFrame,
    fy_labels: pd.DataFrame,
    confirmation_labels: pd.DataFrame,
    *,
    target_config: Mapping[str, object] | None = None,
    split_config: Mapping[str, object] | None = None,
) -> ProbeBundle:
    """连接两类目标并返回无前视的规范化 Probe bundle。"""

    if target_config is None:
        target_config = {}
    elif not isinstance(target_config, Mapping):
        raise ValueError("targets配置必须是对象")
    if split_config is None:
        split_config = {}
    elif not isinstance(split_config, Mapping):
        raise ValueError("splits配置必须是对象")
    normalized_reports = _prepare_reports(reports)
    _validate_source_keys(fy_labels, confirmation_labels)

    fy_versions = _versions(fy_labels)
    confirmation_versions = _versions(confirmation_labels)
    if len(fy_versions) != 1 or fy_versions != confirmation_versions:
        raise ValueError(
            "Residual与Confirmation的label_version不唯一或不一致: "
            f"residual={sorted(fy_versions)}, confirmation={sorted(confirmation_versions)}"
        )

    horizons = [int(x) for x in target_config.get("forecast_horizons", [0, 1, 2])]
    months = [int(x) for x in target_config.get("confirmation_months", [1, 3])]
    panels = [
        str(x) for x in target_config.get("peer_panels", ["fixed", "market", "active"])
    ]
    if not horizons or not months or not panels:
        raise ValueError(
            "目标筛选的forecast_horizons/confirmation_months/peer_panels不能为空"
        )
    unknown_panels = sorted(set(panels).difference({"fixed", "market", "active"}))
    if unknown_panels:
        raise ValueError("未知peer_panel: " + ", ".join(unknown_panels))

    audit_records: list[dict[str, object]] = []
    target_frames: list[pd.DataFrame] = []
    residual_cfg = target_config.get("residual", {})
    residual_enabled = _target_enabled(residual_cfg, name="residual")
    if residual_enabled:
        residual, audit = _residual_targets(
            fy_labels,
            normalized_reports,
            forecast_horizons=horizons,
        )
        target_frames.append(residual)
        audit_records.extend(audit)

    dispersion_cfg = target_config.get("dispersion", {})
    dispersion_enabled = _target_enabled(dispersion_cfg, name="dispersion")
    require_probe_valid = (
        bool(dispersion_cfg.get("require_probe_valid", False))
        if isinstance(dispersion_cfg, Mapping)
        else False
    )
    if dispersion_enabled:
        dispersion, audit = _dispersion_targets(
            confirmation_labels,
            normalized_reports,
            forecast_horizons=horizons,
            confirmation_months=months,
            peer_panels=panels,
            require_probe_valid=require_probe_valid,
        )
        target_frames.append(dispersion)
        audit_records.extend(audit)
    if not target_frames:
        raise ValueError("至少启用一个Probe目标")

    targets = pd.concat(target_frames, ignore_index=True, sort=False)
    if targets["sample_id"].duplicated().any():
        raise ValueError("生成的sample_id不唯一")
    _enforce_label_after_feature(targets)
    windows = parse_split_windows(split_config)
    targets, split_audit = _assign_splits(targets, windows)
    audit_records.extend(split_audit)
    if targets.empty:
        raise ValueError("时间分区与Label可知日过滤后没有任何Probe样本")
    report_split_counts = targets.groupby("report_id")["split"].nunique()
    if report_split_counts.gt(1).any():
        raise RuntimeError("同一report_id被分配到多个时间分区，拒绝潜在PIT")

    used_report_ids = pd.Index(targets["report_id"].unique())
    texts = normalized_reports[
        normalized_reports["report_id"].isin(used_report_ids)
    ].copy()
    if len(texts) != len(used_report_ids):
        raise RuntimeError("最终文本表与目标表report_id覆盖不一致")
    targets = (
        targets[_target_columns(targets)]
        .sort_values(["task_id", "feature_available_date", "report_id"])
        .reset_index(drop=True)
    )
    audit = pd.DataFrame(audit_records)

    task_counts, task_readiness, ready_tasks = _task_split_metadata(
        targets, windows_enabled=bool(windows)
    )
    metadata: dict[str, Any] = {
        "schema_version": "probe_bundle_v1.0",
        "label_version": next(iter(fy_versions)),
        "training_ready": bool(ready_tasks),
        "training_ready_tasks": ready_tasks,
        "counts": {
            "texts": len(texts),
            "targets": len(targets),
            "tasks": int(targets["task_id"].nunique()),
            "residual_targets": int(targets["label_name"].eq(RESIDUAL_LABEL).sum()),
            "dispersion_targets": int(targets["label_name"].eq(DISPERSION_LABEL).sum()),
        },
        "task_counts": task_counts,
        "task_readiness": task_readiness,
        "target_selection": {
            "forecast_horizons": horizons,
            "confirmation_months": months,
            "peer_panels": panels,
            "dispersion_require_probe_valid": require_probe_valid,
        },
        "splits": [window.to_dict() for window in windows],
        "pit_contract": {
            "feature_time": "reports.available_date",
            "residual_label_time": "actual_label_available_date",
            "dispersion_label_time": "label_available_date == target_date",
            "label_strictly_after_feature": True,
            "random_split_forbidden": True,
            "same_report_cross_split_forbidden": True,
            "train_and_validation_labels_known_before_next_split": bool(windows),
            "global_target_transform_applied": False,
            "training_transform_rule": (
                "fit clipping, centering and scaling on train split only"
            ),
        },
        "storage_contract": {
            "texts": "probe_texts.parquet; one row per report_id",
            "targets": "probe_targets.parquet; join many-to-one on report_id",
            "training_loader": "load_probe_task requires one task_id and one split",
        },
    }
    return ProbeBundle(texts=texts, targets=targets, audit=audit, metadata=metadata)


def load_probe_task(
    bundle_directory: str | Path,
    *,
    task_id: str,
    split: str,
    require_training_ready: bool = True,
) -> pd.DataFrame:
    """按单一任务与时间分区即时连接文本，供 Probe 训练器直接读取。"""

    directory = Path(bundle_directory).expanduser().resolve()
    metadata_path = directory / "probe_dataset_metadata.json"
    texts_path = directory / "probe_texts.parquet"
    targets_path = directory / "probe_targets.parquet"
    for path in (metadata_path, texts_path, targets_path):
        if not path.is_file():
            raise FileNotFoundError(f"Probe bundle缺少文件: {path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if split not in SPLIT_NAMES:
        raise ValueError(f"split必须是{SPLIT_NAMES}之一")
    if require_training_ready:
        ready_tasks = set(metadata.get("training_ready_tasks", []))
        if task_id not in ready_tasks:
            raise ValueError(
                f"task_id={task_id!r}尚未具备完整且有效的train/validation/test，"
                "禁止直接训练"
            )
    targets = pd.read_parquet(
        targets_path, filters=[("task_id", "==", task_id), ("split", "==", split)]
    )
    if targets.empty:
        raise ValueError(f"task_id={task_id!r}, split={split!r}没有样本")
    texts = pd.read_parquet(texts_path)
    merged = targets.merge(
        texts,
        on=["report_id", "stock_code", "feature_available_date"],
        how="left",
        validate="many_to_one",
        indicator=True,
    )
    if not merged["_merge"].eq("both").all() or merged["text"].isna().any():
        raise RuntimeError("Probe目标与文本表连接不完整")
    merged = merged.drop(columns="_merge")
    if merged["task_id"].nunique() != 1 or merged["split"].nunique() != 1:
        raise RuntimeError("训练加载器混入多个任务或时间分区")
    return merged.sort_values(["feature_available_date", "report_id"]).reset_index(
        drop=True
    )
