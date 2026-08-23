"""Read-only coverage and maturity audit for continuous report labels.

The audit deliberately does not load ``residual_signed_raw`` or
``delta_log_dispersion`` values.  It only examines identifiers, task dimensions,
validity flags, availability dates, weights, and invalid reasons.  This makes it
safe to use before fixing chronological probe windows.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .probe_dataset import parse_split_windows


RESIDUAL_LABEL = "residual_signed_raw"
DISPERSION_LABEL = "delta_log_dispersion"
DEFAULT_MATURITY_MONTHS = (1, 3, 6, 9, 12, 18, 24, 30, 36, 42, 48)

REPORT_COLUMNS = ("report_id", "stock_code", "available_date")
RESIDUAL_AUDIT_COLUMNS = (
    "report_id",
    "stock_code",
    "available_date",
    "fy",
    "forecast_horizon",
    "residual_valid",
    "actual_label_available_date",
    "actual_invalid_reason",
    "label_version",
)
DISPERSION_AUDIT_COLUMNS = (
    "report_id",
    "stock_code",
    "available_date",
    "fy",
    "forecast_horizon",
    "confirmation_months",
    "peer_panel",
    "confirmation_valid",
    "confirmation_probe_valid",
    "sample_weight",
    "label_available_date",
    "invalid_reason",
    "probe_invalid_reason",
    "label_version",
)


@dataclass(frozen=True)
class AuditInputs:
    config_path: Path
    label_directory: Path
    reports_path: Path
    residual_path: Path
    dispersion_path: Path
    label_metadata_path: Path


def _require_columns(
    frame: pd.DataFrame, required: Sequence[str], *, source: str
) -> None:
    missing = sorted(set(required).difference(frame.columns))
    if missing:
        raise ValueError(f"{source}缺少审计字段: {', '.join(missing)}")


def _as_date(values: pd.Series) -> pd.Series:
    return pd.to_datetime(values, errors="coerce").dt.normalize()


def _as_binary(values: pd.Series, *, name: str) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    invalid = numeric.isna() | ~numeric.isin([0, 1])
    if invalid.any():
        raise ValueError(f"{name}含{int(invalid.sum())}个非0/1值")
    return numeric.astype(np.int8)


def _configured_target_settings(
    config: Mapping[str, object],
) -> dict[str, object]:
    raw = config.get("targets", {})
    if not isinstance(raw, Mapping):
        raise ValueError("targets配置必须是对象")
    residual = raw.get("residual", {})
    dispersion = raw.get("dispersion", {})
    if not isinstance(residual, Mapping) or not isinstance(dispersion, Mapping):
        raise ValueError("targets.residual/dispersion必须是对象")
    return {
        "forecast_horizons": tuple(
            int(value) for value in raw.get("forecast_horizons", [0, 1, 2])
        ),
        "confirmation_months": tuple(
            int(value) for value in raw.get("confirmation_months", [1, 3])
        ),
        "peer_panels": tuple(
            str(value)
            for value in raw.get("peer_panels", ["fixed", "market", "active"])
        ),
        "residual_enabled": bool(residual.get("enabled", True)),
        "dispersion_enabled": bool(dispersion.get("enabled", True)),
        "dispersion_require_probe_valid": bool(
            dispersion.get("require_probe_valid", False)
        ),
    }


def _normalize_reports(reports: pd.DataFrame) -> pd.DataFrame:
    _require_columns(reports, REPORT_COLUMNS, source="reports.parquet")
    out = reports[list(REPORT_COLUMNS)].copy()
    out["report_id"] = out["report_id"].astype("string").str.strip()
    out["stock_code"] = out["stock_code"].astype("string").str.strip()
    out["feature_available_date"] = _as_date(out["available_date"])
    if out["report_id"].isna().any() or out["report_id"].eq("").any():
        raise ValueError("reports.parquet含空report_id")
    if out["report_id"].duplicated().any():
        raise ValueError("reports.parquet的report_id不唯一")
    if out["feature_available_date"].isna().any():
        raise ValueError("reports.parquet含无效available_date")
    return out.drop(columns="available_date")


def _attach_report_identity(
    frame: pd.DataFrame, reports: pd.DataFrame, *, source: str
) -> pd.DataFrame:
    out = frame.merge(
        reports,
        on="report_id",
        how="left",
        validate="many_to_one",
        indicator=True,
        suffixes=("_label", "_report"),
    )
    missing = out["_merge"].ne("both")
    if missing.any():
        raise ValueError(
            f"{source}有{int(missing.sum())}个report_id不在reports.parquet"
        )
    label_stock = out["stock_code_label"].astype("string").str.strip()
    report_stock = out["stock_code_report"].astype("string").str.strip()
    if not label_stock.eq(report_stock).all():
        raise ValueError(f"{source}与reports.parquet的stock_code不一致")
    label_feature = _as_date(out["available_date"])
    if not label_feature.eq(out["feature_available_date"]).all():
        raise ValueError(f"{source}与reports.parquet的available_date不一致")
    out["stock_code"] = report_stock
    return out.drop(
        columns=[
            "stock_code_label",
            "stock_code_report",
            "available_date",
            "_merge",
        ]
    )


def normalize_audit_rows(
    reports: pd.DataFrame,
    residual: pd.DataFrame,
    dispersion: pd.DataFrame,
    *,
    target_config: Mapping[str, object] | None = None,
) -> pd.DataFrame:
    """Return one audit row per raw target row, without target values."""

    settings = _configured_target_settings({"targets": target_config or {}})
    reports = _normalize_reports(reports)
    _require_columns(
        residual, RESIDUAL_AUDIT_COLUMNS, source="report_fy_labels.parquet"
    )
    _require_columns(
        dispersion,
        DISPERSION_AUDIT_COLUMNS,
        source="report_confirmation_labels.parquet",
    )

    residual_work = _attach_report_identity(
        residual[list(RESIDUAL_AUDIT_COLUMNS)].copy(),
        reports,
        source="report_fy_labels.parquet",
    )
    residual_work["forecast_horizon"] = pd.to_numeric(
        residual_work["forecast_horizon"], errors="raise"
    ).astype(int)
    residual_work["source_valid"] = _as_binary(
        residual_work["residual_valid"], name="residual_valid"
    ).astype(bool)
    residual_work["probe_valid"] = pd.Series(
        pd.NA, index=residual_work.index, dtype="boolean"
    )
    residual_work["label_available_date"] = _as_date(
        residual_work["actual_label_available_date"]
    )
    residual_work["target_weight"] = np.where(residual_work["source_valid"], 1.0, 0.0)
    residual_work["label_name"] = RESIDUAL_LABEL
    residual_work["task_id"] = (
        RESIDUAL_LABEL + "__fh" + residual_work["forecast_horizon"].astype(str)
    )
    residual_work["invalid_reason_main"] = residual_work[
        "actual_invalid_reason"
    ].astype("string")
    residual_work["invalid_reason_probe"] = pd.Series(
        pd.NA, index=residual_work.index, dtype="string"
    )
    residual_work["configured_task"] = bool(settings["residual_enabled"]) & (
        residual_work["forecast_horizon"].isin(settings["forecast_horizons"])
    )
    residual_work["configured_valid"] = (
        residual_work["configured_task"] & residual_work["source_valid"]
    )

    dispersion_work = _attach_report_identity(
        dispersion[list(DISPERSION_AUDIT_COLUMNS)].copy(),
        reports,
        source="report_confirmation_labels.parquet",
    )
    for column in ("forecast_horizon", "confirmation_months"):
        dispersion_work[column] = pd.to_numeric(
            dispersion_work[column], errors="raise"
        ).astype(int)
    dispersion_work["peer_panel"] = (
        dispersion_work["peer_panel"].astype("string").str.strip()
    )
    dispersion_work["source_valid"] = _as_binary(
        dispersion_work["confirmation_valid"], name="confirmation_valid"
    ).astype(bool)
    dispersion_work["probe_valid"] = _as_binary(
        dispersion_work["confirmation_probe_valid"],
        name="confirmation_probe_valid",
    ).astype(bool)
    if (dispersion_work["probe_valid"] & ~dispersion_work["source_valid"]).any():
        raise ValueError("confirmation_probe_valid=1不能出现在confirmation_valid=0")
    dispersion_work["label_available_date"] = _as_date(
        dispersion_work["label_available_date"]
    )
    dispersion_work["target_weight"] = pd.to_numeric(
        dispersion_work["sample_weight"], errors="coerce"
    )
    dispersion_work["label_name"] = DISPERSION_LABEL
    dispersion_work["task_id"] = (
        DISPERSION_LABEL
        + "__"
        + dispersion_work["confirmation_months"].astype(str)
        + "m__"
        + dispersion_work["peer_panel"].astype(str)
        + "__fh"
        + dispersion_work["forecast_horizon"].astype(str)
    )
    dispersion_work["invalid_reason_main"] = dispersion_work["invalid_reason"].astype(
        "string"
    )
    dispersion_work["invalid_reason_probe"] = dispersion_work[
        "probe_invalid_reason"
    ].astype("string")
    dispersion_work["configured_task"] = (
        bool(settings["dispersion_enabled"])
        & dispersion_work["forecast_horizon"].isin(settings["forecast_horizons"])
        & dispersion_work["confirmation_months"].isin(settings["confirmation_months"])
        & dispersion_work["peer_panel"].isin(settings["peer_panels"])
    )
    configured_dispersion_valid = dispersion_work["source_valid"]
    if settings["dispersion_require_probe_valid"]:
        configured_dispersion_valid &= dispersion_work["probe_valid"]
    dispersion_work["configured_valid"] = (
        dispersion_work["configured_task"] & configured_dispersion_valid
    )

    keep = [
        "label_name",
        "task_id",
        "report_id",
        "stock_code",
        "fy",
        "forecast_horizon",
        "confirmation_months",
        "peer_panel",
        "feature_available_date",
        "label_available_date",
        "source_valid",
        "probe_valid",
        "configured_task",
        "configured_valid",
        "target_weight",
        "invalid_reason_main",
        "invalid_reason_probe",
        "label_version",
    ]
    residual_keep = [
        column
        for column in keep
        if column
        not in {
            "confirmation_months",
            "peer_panel",
            "probe_valid",
            "invalid_reason_probe",
        }
    ]
    rows = pd.concat(
        [residual_work[residual_keep], dispersion_work[keep]],
        ignore_index=True,
        sort=False,
    )[keep]
    rows["probe_valid"] = rows["probe_valid"].astype("boolean")
    for column in ("source_valid", "configured_task", "configured_valid"):
        rows[column] = rows[column].astype(bool)
    rows["maturity_days"] = (
        rows["label_available_date"] - rows["feature_available_date"]
    ).dt.days
    rows["positive_weight"] = (
        pd.to_numeric(rows["target_weight"], errors="coerce").fillna(0) > 0
    )
    numeric_weight = pd.to_numeric(rows["target_weight"], errors="coerce")
    rows["weight_valid"] = np.isfinite(numeric_weight) & numeric_weight.ge(0)
    rows["pit_date_valid"] = (
        rows["label_available_date"].notna()
        & rows["feature_available_date"].notna()
        & (rows["label_available_date"] > rows["feature_available_date"])
    )
    return rows.sort_values(
        ["label_name", "task_id", "feature_available_date", "report_id"]
    ).reset_index(drop=True)


def _iso(value: object) -> str | None:
    if value is None or pd.isna(value):
        return None
    return str(pd.Timestamp(value).date())


def _finite_or_none(value: object) -> float | int | None:
    if value is None or pd.isna(value):
        return None
    numeric = float(value)
    return int(numeric) if numeric.is_integer() else numeric


def summarize_task_coverage(rows: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for (label_name, task_id), group in rows.groupby(
        ["label_name", "task_id"], sort=True
    ):
        valid = group["configured_valid"]
        valid_dates = group.loc[valid & group["pit_date_valid"]]
        maturity = valid_dates["maturity_days"].to_numpy(dtype=float)
        records.append(
            {
                "label_name": label_name,
                "task_id": task_id,
                "configured_task": bool(group["configured_task"].any()),
                "source_rows": int(len(group)),
                "source_valid_rows": int(group["source_valid"].sum()),
                "probe_valid_rows": (
                    int(group["probe_valid"].fillna(False).sum())
                    if label_name == DISPERSION_LABEL
                    else None
                ),
                "configured_valid_rows": int(valid.sum()),
                "positive_weight_valid_rows": int(
                    (valid & group["positive_weight"]).sum()
                ),
                "invalid_weight_valid_rows": int(
                    (valid & ~group["weight_valid"]).sum()
                ),
                "valid_missing_label_date": int(
                    (valid & group["label_available_date"].isna()).sum()
                ),
                "valid_pit_violations": int(
                    (
                        valid
                        & group["label_available_date"].notna()
                        & ~group["pit_date_valid"]
                    ).sum()
                ),
                "unique_reports": int(group.loc[valid, "report_id"].nunique()),
                "unique_symbols": int(group.loc[valid, "stock_code"].nunique()),
                "feature_date_min": _iso(group["feature_available_date"].min()),
                "feature_date_max": _iso(group["feature_available_date"].max()),
                "label_date_min": _iso(valid_dates["label_available_date"].min()),
                "label_date_max": _iso(valid_dates["label_available_date"].max()),
                "maturity_days_p50": (
                    _finite_or_none(np.quantile(maturity, 0.5))
                    if len(maturity)
                    else None
                ),
                "maturity_days_p90": (
                    _finite_or_none(np.quantile(maturity, 0.9))
                    if len(maturity)
                    else None
                ),
                "maturity_days_p99": (
                    _finite_or_none(np.quantile(maturity, 0.99))
                    if len(maturity)
                    else None
                ),
            }
        )
    return pd.DataFrame(records)


def configured_task_catalog(config: Mapping[str, object]) -> pd.DataFrame:
    settings = _configured_target_settings(config)
    records: list[dict[str, object]] = []
    if settings["residual_enabled"]:
        records.extend(
            {
                "label_name": RESIDUAL_LABEL,
                "task_id": f"{RESIDUAL_LABEL}__fh{horizon}",
            }
            for horizon in settings["forecast_horizons"]
        )
    if settings["dispersion_enabled"]:
        records.extend(
            {
                "label_name": DISPERSION_LABEL,
                "task_id": (f"{DISPERSION_LABEL}__{months}m__{panel}__fh{horizon}"),
            }
            for months in settings["confirmation_months"]
            for panel in settings["peer_panels"]
            for horizon in settings["forecast_horizons"]
        )
    frame = pd.DataFrame(records, columns=["label_name", "task_id"])
    if frame.empty:
        return frame
    return (
        frame.drop_duplicates()
        .sort_values(["label_name", "task_id"])
        .reset_index(drop=True)
    )


def _complete_task_coverage(
    coverage: pd.DataFrame, catalog: pd.DataFrame
) -> pd.DataFrame:
    if catalog.empty:
        return coverage
    if coverage.empty:
        coverage = pd.DataFrame(columns=["label_name", "task_id"])
    catalog = catalog.assign(configured_task_expected=True)
    output = coverage.merge(
        catalog, on=["label_name", "task_id"], how="outer", validate="one_to_one"
    )
    count_columns = [
        "source_rows",
        "source_valid_rows",
        "probe_valid_rows",
        "configured_valid_rows",
        "positive_weight_valid_rows",
        "invalid_weight_valid_rows",
        "valid_missing_label_date",
        "valid_pit_violations",
        "unique_reports",
        "unique_symbols",
    ]
    for column in count_columns:
        if column not in output:
            output[column] = np.nan
    if "configured_task" not in output:
        output["configured_task"] = False
    for column in (
        "feature_date_min",
        "feature_date_max",
        "label_date_min",
        "label_date_max",
        "maturity_days_p50",
        "maturity_days_p90",
        "maturity_days_p99",
    ):
        if column not in output:
            output[column] = None
    missing_source = output["source_rows"].isna()
    output.loc[missing_source, count_columns] = 0
    output["configured_task"] = output["configured_task"].astype("boolean").fillna(
        False
    ) | output["configured_task_expected"].astype("boolean").fillna(False)
    for column in count_columns:
        output[column] = pd.to_numeric(output[column], errors="coerce").astype("Int64")
    return output.sort_values(["label_name", "task_id"]).reset_index(drop=True)


def _complete_split_counts(
    counts: pd.DataFrame,
    *,
    catalog: pd.DataFrame,
    split_config: Mapping[str, object],
) -> pd.DataFrame:
    windows = parse_split_windows(split_config)
    if not windows or catalog.empty:
        return counts
    expected = pd.DataFrame(
        [
            {
                **task,
                "split": window.name,
                "feature_start": _iso(window.feature_start),
                "feature_end": _iso(window.feature_end),
                "label_cutoff": _iso(window.label_cutoff),
            }
            for task in catalog.to_dict(orient="records")
            for window in windows
        ]
    )
    keys = ["label_name", "task_id", "split"]
    if counts.empty:
        output = expected.copy()
    else:
        output = expected.merge(
            counts, on=keys, how="outer", suffixes=("_expected", "")
        )
        for column in ("feature_start", "feature_end", "label_cutoff"):
            expected_column = f"{column}_expected"
            output[column] = output[column].fillna(output[expected_column])
            output = output.drop(columns=expected_column)
    count_columns = [
        "source_rows_in_feature_window",
        "configured_valid_rows_in_feature_window",
        "eligible_rows",
        "positive_weight_eligible_rows",
        "unique_reports_eligible",
        "unique_symbols_eligible",
        "excluded_label_too_late",
        "excluded_missing_or_invalid_label_date",
    ]
    for column in count_columns:
        output[column] = (
            pd.to_numeric(output[column], errors="coerce").fillna(0).astype(int)
        )
    return output.sort_values(["label_name", "task_id", "split"]).reset_index(drop=True)


def summarize_maturity(rows: pd.DataFrame) -> pd.DataFrame:
    quantiles = (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0)
    records: list[dict[str, object]] = []
    selected = rows[
        rows["configured_valid"]
        & rows["pit_date_valid"]
        & rows["maturity_days"].notna()
    ]
    for (label_name, task_id), group in selected.groupby(
        ["label_name", "task_id"], sort=True
    ):
        values = group["maturity_days"].to_numpy(dtype=float)
        record: dict[str, object] = {
            "label_name": label_name,
            "task_id": task_id,
            "rows": int(len(values)),
        }
        for quantile in quantiles:
            name = f"p{int(round(100 * quantile)):02d}"
            record[name] = _finite_or_none(np.quantile(values, quantile))
        records.append(record)
    return pd.DataFrame(records)


def summarize_invalid_reasons(rows: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for (label_name, task_id), group in rows.groupby(
        ["label_name", "task_id"], sort=True
    ):
        main_invalid = group[~group["source_valid"]]
        reasons = main_invalid["invalid_reason_main"].fillna("<missing_reason>")
        for reason, count in reasons.value_counts(dropna=False).items():
            records.append(
                {
                    "label_name": label_name,
                    "task_id": task_id,
                    "scope": "main_validity",
                    "reason": str(reason),
                    "count": int(count),
                }
            )
        if label_name == DISPERSION_LABEL:
            probe_invalid = group[
                group["source_valid"] & ~group["probe_valid"].fillna(False)
            ]
            probe_reasons = probe_invalid["invalid_reason_probe"].fillna(
                "<missing_reason>"
            )
            for reason, count in probe_reasons.value_counts(dropna=False).items():
                records.append(
                    {
                        "label_name": label_name,
                        "task_id": task_id,
                        "scope": "probe_validity",
                        "reason": str(reason),
                        "count": int(count),
                    }
                )
    return pd.DataFrame(records)


def summarize_configured_splits(
    rows: pd.DataFrame, split_config: Mapping[str, object]
) -> pd.DataFrame:
    windows = parse_split_windows(split_config)
    if not windows:
        return pd.DataFrame()
    configured = rows[rows["configured_task"]].copy()
    records: list[dict[str, object]] = []
    for (label_name, task_id), task in configured.groupby(
        ["label_name", "task_id"], sort=True
    ):
        for window in windows:
            in_window = task["feature_available_date"].between(
                window.feature_start, window.feature_end, inclusive="both"
            )
            source = task[in_window]
            valid = source[source["configured_valid"]]
            if window.label_cutoff is None:
                known = valid["pit_date_valid"]
            else:
                known = valid["pit_date_valid"] & (
                    valid["label_available_date"] <= window.label_cutoff
                )
            eligible = valid[known]
            records.append(
                {
                    "label_name": label_name,
                    "task_id": task_id,
                    "split": window.name,
                    "feature_start": _iso(window.feature_start),
                    "feature_end": _iso(window.feature_end),
                    "label_cutoff": _iso(window.label_cutoff),
                    "source_rows_in_feature_window": int(len(source)),
                    "configured_valid_rows_in_feature_window": int(len(valid)),
                    "eligible_rows": int(len(eligible)),
                    "positive_weight_eligible_rows": int(
                        eligible["positive_weight"].sum()
                    ),
                    "unique_reports_eligible": int(eligible["report_id"].nunique()),
                    "unique_symbols_eligible": int(eligible["stock_code"].nunique()),
                    "excluded_label_too_late": int(
                        (
                            valid["pit_date_valid"]
                            & ~known
                            & valid["label_available_date"].notna()
                        ).sum()
                    ),
                    "excluded_missing_or_invalid_label_date": int(
                        (~valid["pit_date_valid"]).sum()
                    ),
                }
            )
    return pd.DataFrame(records)


def _calendar_windows(
    minimum: pd.Timestamp, maximum: pd.Timestamp
) -> list[tuple[str, str, pd.Timestamp, pd.Timestamp]]:
    windows: list[tuple[str, str, pd.Timestamp, pd.Timestamp]] = []
    for year in range(minimum.year, maximum.year + 1):
        year_start = pd.Timestamp(year=year, month=1, day=1)
        year_end = pd.Timestamp(year=year, month=12, day=31)
        windows.append(("year", str(year), year_start, year_end))
        windows.append(
            (
                "half_year",
                f"{year}H1",
                year_start,
                pd.Timestamp(year=year, month=6, day=30),
            )
        )
        windows.append(
            (
                "half_year",
                f"{year}H2",
                pd.Timestamp(year=year, month=7, day=1),
                year_end,
            )
        )
    return windows


def summarize_candidate_windows(
    rows: pd.DataFrame,
    *,
    maturity_months: Sequence[int] = DEFAULT_MATURITY_MONTHS,
) -> pd.DataFrame:
    months = sorted({int(value) for value in maturity_months})
    if not months or months[0] <= 0:
        raise ValueError("候选成熟月数必须为正整数")
    configured = rows[rows["configured_valid"]].copy()
    if configured.empty:
        return pd.DataFrame()
    minimum = configured["feature_available_date"].min()
    maximum = configured["feature_available_date"].max()
    if pd.isna(minimum) or pd.isna(maximum):
        return pd.DataFrame()
    windows = _calendar_windows(pd.Timestamp(minimum), pd.Timestamp(maximum))
    records: list[dict[str, object]] = []
    for (label_name, task_id), task in configured.groupby(
        ["label_name", "task_id"], sort=True
    ):
        for frequency, cohort, start, end in windows:
            in_window = task["feature_available_date"].between(
                start, end, inclusive="both"
            )
            feature_rows = task[in_window]
            if feature_rows.empty:
                continue
            for month_count in months:
                cutoff = end + pd.DateOffset(months=month_count)
                known = feature_rows[
                    feature_rows["pit_date_valid"]
                    & (feature_rows["label_available_date"] <= cutoff)
                ]
                records.append(
                    {
                        "label_name": label_name,
                        "task_id": task_id,
                        "frequency": frequency,
                        "cohort": cohort,
                        "feature_start": _iso(start),
                        "feature_end": _iso(end),
                        "maturity_months": month_count,
                        "label_cutoff": _iso(cutoff),
                        "configured_valid_rows": int(len(feature_rows)),
                        "known_rows": int(len(known)),
                        "positive_weight_known_rows": int(
                            known["positive_weight"].sum()
                        ),
                        "known_fraction": float(len(known) / len(feature_rows)),
                        "unique_reports_known": int(known["report_id"].nunique()),
                        "unique_symbols_known": int(known["stock_code"].nunique()),
                    }
                )
    return pd.DataFrame(records)


def build_audit_tables(
    reports: pd.DataFrame,
    residual: pd.DataFrame,
    dispersion: pd.DataFrame,
    *,
    config: Mapping[str, object],
    maturity_months: Sequence[int] = DEFAULT_MATURITY_MONTHS,
) -> dict[str, pd.DataFrame]:
    rows = normalize_audit_rows(
        reports,
        residual,
        dispersion,
        target_config=config.get("targets", {}),
    )
    split_config = config.get("splits", {})
    if not isinstance(split_config, Mapping):
        raise ValueError("splits配置必须是对象")
    catalog = configured_task_catalog(config)
    coverage = _complete_task_coverage(summarize_task_coverage(rows), catalog)
    split_counts = _complete_split_counts(
        summarize_configured_splits(rows, split_config),
        catalog=catalog,
        split_config=split_config,
    )
    return {
        "task_coverage": coverage,
        "maturity_distribution": summarize_maturity(rows),
        "invalid_reason_counts": summarize_invalid_reasons(rows),
        "configured_split_counts": split_counts,
        "candidate_window_counts": summarize_candidate_windows(
            rows, maturity_months=maturity_months
        ),
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _file_record(path: Path) -> dict[str, object]:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "sha256": _sha256_file(path),
    }


def summarize_source_integrity(
    reports: pd.DataFrame,
    residual: pd.DataFrame,
    dispersion: pd.DataFrame,
    *,
    label_metadata: Mapping[str, object],
) -> dict[str, object]:
    residual_versions = sorted(
        residual["label_version"].dropna().astype(str).str.strip().unique().tolist()
    )
    dispersion_versions = sorted(
        dispersion["label_version"].dropna().astype(str).str.strip().unique().tolist()
    )
    metadata_version = str(label_metadata.get("label_version", "")).strip()
    return {
        "reports_duplicate_report_id_rows": int(
            reports.duplicated(["report_id"], keep=False).sum()
        ),
        "residual_duplicate_key_rows": int(
            residual.duplicated(["report_id", "fy"], keep=False).sum()
        ),
        "dispersion_duplicate_key_rows": int(
            dispersion.duplicated(
                ["report_id", "fy", "confirmation_months", "peer_panel"],
                keep=False,
            ).sum()
        ),
        "residual_label_versions": residual_versions,
        "dispersion_label_versions": dispersion_versions,
        "metadata_label_version": metadata_version,
        "label_version_consistent": bool(
            len(residual_versions) == 1
            and residual_versions == dispersion_versions
            and residual_versions[0] == metadata_version
        ),
    }


def _records(frame: pd.DataFrame) -> list[dict[str, object]]:
    clean = frame.astype(object).where(pd.notna(frame), None)
    return clean.to_dict(orient="records")


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False, default=str),
        encoding="utf-8",
    )


def _markdown_table(frame: pd.DataFrame, columns: Sequence[str]) -> str:
    if frame.empty:
        return "_无记录_"
    selected = (
        frame[list(columns)]
        .copy()
        .astype(object)
        .where(pd.notna(frame[list(columns)]), "")
    )
    headers = list(selected.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in selected.itertuples(index=False, name=None):
        values = [str(value).replace("|", "\\|").replace("\n", " ") for value in row]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def render_summary(
    tables: Mapping[str, pd.DataFrame], *, source_counts: Mapping[str, int]
) -> str:
    coverage = tables["task_coverage"].copy()
    splits = tables["configured_split_counts"].copy()
    invalid = tables["invalid_reason_counts"].copy()
    zero_validation = (
        splits[splits["split"].eq("validation") & splits["eligible_rows"].eq(0)]
        if not splits.empty
        else splits
    )
    lines = [
        "# Continuous-label coverage audit",
        "",
        "This report is outcome-blind: label value columns were not loaded.",
        "",
        "## Source row counts",
        "",
        *[f"- `{name}`: {count:,}" for name, count in source_counts.items()],
        "",
        "## Task coverage and maturity",
        "",
        _markdown_table(
            coverage,
            [
                "task_id",
                "source_rows",
                "source_valid_rows",
                "configured_valid_rows",
                "positive_weight_valid_rows",
                "feature_date_min",
                "feature_date_max",
                "label_date_min",
                "label_date_max",
                "maturity_days_p50",
                "maturity_days_p90",
                "maturity_days_p99",
            ],
        ),
        "",
        "## Configured split counts",
        "",
        _markdown_table(
            splits,
            [
                "task_id",
                "split",
                "configured_valid_rows_in_feature_window",
                "eligible_rows",
                "positive_weight_eligible_rows",
                "excluded_label_too_late",
                "excluded_missing_or_invalid_label_date",
            ],
        ),
        "",
        "## Tasks with zero configured validation rows",
        "",
        _markdown_table(
            zero_validation,
            [
                "task_id",
                "configured_valid_rows_in_feature_window",
                "excluded_label_too_late",
                "excluded_missing_or_invalid_label_date",
            ],
        ),
        "",
        "## Invalid reasons",
        "",
        _markdown_table(
            (
                invalid.sort_values("count", ascending=False).head(100)
                if not invalid.empty
                else invalid
            ),
            ["task_id", "scope", "reason", "count"],
        ),
        "",
        "## Candidate-window interpretation",
        "",
        "`candidate_window_counts.json` enumerates calendar-year and half-year feature cohorts with fixed 1–48 month maturity cutoffs. It is descriptive only and does not select a split automatically.",
        "",
    ]
    return "\n".join(lines)


def resolve_audit_inputs(
    config_path: str | Path,
    *,
    repository_root: str | Path,
    label_directory: str | Path | None = None,
) -> tuple[AuditInputs, dict[str, object]]:
    from .config import load_yaml_config

    root = Path(repository_root).expanduser().resolve()
    config_file = Path(config_path).expanduser()
    if not config_file.is_absolute():
        config_file = root / config_file
    config_file = config_file.resolve()
    if not config_file.is_file():
        raise FileNotFoundError(f"审计配置不存在: {config_file}")
    config = load_yaml_config(config_file)
    if label_directory is None:
        paths = config.get("paths", {})
        if not isinstance(paths, Mapping):
            raise ValueError("paths配置必须是对象")
        label_directory = paths.get("report_labels")
    label_dir = Path(str(label_directory or "")).expanduser()
    if not label_dir.is_absolute():
        label_dir = root / label_dir
    label_dir = label_dir.resolve()
    if not label_dir.is_dir():
        raise FileNotFoundError(f"report_labels目录不存在: {label_dir}")
    inputs = AuditInputs(
        config_path=config_file,
        label_directory=label_dir,
        reports_path=label_dir / "reports.parquet",
        residual_path=label_dir / "report_fy_labels.parquet",
        dispersion_path=label_dir / "report_confirmation_labels.parquet",
        label_metadata_path=label_dir / "label_metadata.json",
    )
    for path in (
        inputs.reports_path,
        inputs.residual_path,
        inputs.dispersion_path,
        inputs.label_metadata_path,
    ):
        if not path.is_file():
            raise FileNotFoundError(f"审计输入不存在: {path}")
    return inputs, config


def _safe_output(output: Path, *, repository_root: Path) -> None:
    if output == Path(output.anchor) or len(output.parts) < 4:
        raise ValueError(f"拒绝使用过宽审计输出目录: {output}")
    default_root = repository_root / "audit_reports" / "continuous_label_audit"
    if output == default_root:
        raise ValueError("审计输出必须包含独立run_id，不能直接写入输出根目录")


def run_continuous_label_audit(
    *,
    config_path: str | Path,
    repository_root: str | Path,
    output_directory: str | Path,
    label_directory: str | Path | None = None,
    maturity_months: Sequence[int] = DEFAULT_MATURITY_MONTHS,
) -> dict[str, object]:
    """Read source metadata, write aggregate JSON/Markdown, and never overwrite."""

    root = Path(repository_root).expanduser().resolve()
    output = Path(output_directory).expanduser()
    if not output.is_absolute():
        output = root / output
    output = output.resolve()
    _safe_output(output, repository_root=root)
    if output.exists():
        raise FileExistsError(f"审计输出已存在，拒绝覆盖: {output}")
    inputs, config = resolve_audit_inputs(
        config_path,
        repository_root=root,
        label_directory=label_directory,
    )
    reports = pd.read_parquet(inputs.reports_path, columns=list(REPORT_COLUMNS))
    residual = pd.read_parquet(
        inputs.residual_path, columns=list(RESIDUAL_AUDIT_COLUMNS)
    )
    dispersion = pd.read_parquet(
        inputs.dispersion_path, columns=list(DISPERSION_AUDIT_COLUMNS)
    )
    label_metadata = json.loads(inputs.label_metadata_path.read_text(encoding="utf-8"))
    if not isinstance(label_metadata, Mapping):
        raise ValueError("label_metadata.json必须是对象")
    tables = build_audit_tables(
        reports,
        residual,
        dispersion,
        config=config,
        maturity_months=maturity_months,
    )
    source_counts = {
        "reports": len(reports),
        "report_fy_labels": len(residual),
        "report_confirmation_labels": len(dispersion),
    }
    sources = [
        _file_record(path)
        for path in (
            inputs.config_path,
            inputs.reports_path,
            inputs.residual_path,
            inputs.dispersion_path,
            inputs.label_metadata_path,
        )
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent))
    try:
        table_files: dict[str, str] = {}
        for name, frame in tables.items():
            filename = f"{name}.json"
            _write_json(temporary / filename, _records(frame))
            table_files[name] = filename
        summary_path = temporary / "SUMMARY.md"
        summary_path.write_text(
            render_summary(tables, source_counts=source_counts), encoding="utf-8"
        )
        generated_files = [
            temporary / filename for filename in table_files.values()
        ] + [summary_path]
        manifest = {
            "schema_version": "continuous_label_coverage_audit_v1.0",
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "outcome_blind": True,
            "label_value_columns_loaded": [],
            "repository_root": str(root),
            "output_directory": str(output),
            "maturity_months": sorted({int(value) for value in maturity_months}),
            "target_settings": _configured_target_settings(config),
            "source_counts": source_counts,
            "source_integrity": summarize_source_integrity(
                reports,
                residual,
                dispersion,
                label_metadata=label_metadata,
            ),
            "upstream_label_metadata": {
                "label_version": label_metadata.get("label_version"),
                "created_at": label_metadata.get("created_at"),
                "counts": label_metadata.get("counts", {}),
            },
            "sources": sources,
            "tables": {
                name: {
                    "file": filename,
                    "rows": int(len(tables[name])),
                }
                for name, filename in table_files.items()
            },
            "generated_files": [
                {
                    "file": path.name,
                    "size_bytes": int(path.stat().st_size),
                    "sha256": _sha256_file(path),
                }
                for path in generated_files
            ],
            "write_contract": {
                "overwrite": False,
                "extensions": [".json", ".md"],
                "default_root": "audit_reports/continuous_label_audit",
            },
        }
        _write_json(temporary / "manifest.json", manifest)
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return manifest
