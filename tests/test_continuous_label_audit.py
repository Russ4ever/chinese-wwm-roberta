import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from label_engineering.audit_continuous_labels import run
from src.continuous_label_audit import build_audit_tables


def _reports() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"report_id": "r1", "stock_code": "000001", "available_date": "2022-02-01"},
            {"report_id": "r2", "stock_code": "000002", "available_date": "2022-08-01"},
            {"report_id": "r3", "stock_code": "000003", "available_date": "2022-08-15"},
            {"report_id": "r4", "stock_code": "000004", "available_date": "2022-08-20"},
        ]
    )


def _residual() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "report_id": "r1",
                "stock_code": "000001",
                "available_date": "2022-02-01",
                "fy": 2022,
                "forecast_horizon": 0,
                "residual_valid": 1,
                "actual_label_available_date": "2023-03-01",
                "actual_invalid_reason": None,
                "label_version": "v1",
            },
            {
                "report_id": "r2",
                "stock_code": "000002",
                "available_date": "2022-08-01",
                "fy": 2022,
                "forecast_horizon": 0,
                "residual_valid": 1,
                "actual_label_available_date": "2023-03-15",
                "actual_invalid_reason": None,
                "label_version": "v1",
            },
            {
                "report_id": "r3",
                "stock_code": "000003",
                "available_date": "2022-08-15",
                "fy": 2023,
                "forecast_horizon": 1,
                "residual_valid": 0,
                "actual_label_available_date": pd.NaT,
                "actual_invalid_reason": "actual_missing",
                "label_version": "v1",
            },
        ]
    )


def _dispersion() -> pd.DataFrame:
    rows = []
    for report_id, available, months, label_date, valid, reason in (
        ("r1", "2022-02-01", 1, "2022-03-01", 1, None),
        ("r2", "2022-08-01", 3, "2022-11-01", 1, None),
        ("r3", "2022-08-15", 3, "2022-11-15", 1, None),
        ("r4", "2022-08-20", 3, "2022-11-20", 0, "insufficient_pre_peers"),
    ):
        rows.append(
            {
                "report_id": report_id,
                "stock_code": _reports()
                .set_index("report_id")
                .loc[report_id, "stock_code"],
                "available_date": available,
                "fy": 2022,
                "forecast_horizon": 0,
                "confirmation_months": months,
                "peer_panel": "fixed",
                "confirmation_valid": valid,
                "confirmation_probe_valid": valid,
                "sample_weight": 1.0 if valid else 0.0,
                "label_available_date": label_date,
                "invalid_reason": reason,
                "probe_invalid_reason": None,
                "label_version": "v1",
            }
        )
    return pd.DataFrame(rows)


def _config(label_dir: Path | None = None) -> dict:
    return {
        "paths": {"report_labels": str(label_dir) if label_dir else "unused"},
        "targets": {
            "forecast_horizons": [0, 1, 2],
            "confirmation_months": [1, 3],
            "peer_panels": ["fixed", "market", "active"],
            "residual": {"enabled": True},
            "dispersion": {"enabled": True, "require_probe_valid": False},
        },
        "splits": {
            "enabled": True,
            "train": {
                "feature_start": "2022-01-01",
                "feature_end": "2022-06-30",
                "label_cutoff": "2022-06-30",
            },
            "validation": {
                "feature_start": "2022-07-01",
                "feature_end": "2022-09-30",
                "label_cutoff": "2022-09-30",
            },
            "test": {
                "feature_start": "2022-10-01",
                "feature_end": "2022-12-31",
                "label_cutoff": "2023-12-31",
            },
        },
    }


def test_audit_counts_late_3m_and_residual_without_label_values():
    tables = build_audit_tables(
        _reports(),
        _residual(),
        _dispersion(),
        config=_config(),
        maturity_months=[1, 3, 6, 12],
    )
    splits = tables["configured_split_counts"]
    three_month = splits[
        splits["task_id"].eq("delta_log_dispersion__3m__fixed__fh0")
        & splits["split"].eq("validation")
    ].iloc[0]
    assert three_month["configured_valid_rows_in_feature_window"] == 2
    assert three_month["eligible_rows"] == 0
    assert three_month["excluded_label_too_late"] == 2

    residual = splits[
        splits["task_id"].eq("residual_signed_raw__fh0")
        & splits["split"].eq("validation")
    ].iloc[0]
    assert residual["eligible_rows"] == 0
    assert residual["excluded_label_too_late"] == 1

    reasons = tables["invalid_reason_counts"]
    assert (reasons["reason"].eq("actual_missing") & reasons["count"].eq(1)).any()
    assert (
        reasons["reason"].eq("insufficient_pre_peers") & reasons["count"].eq(1)
    ).any()
    candidates = tables["candidate_window_counts"]
    assert not candidates.empty
    assert set(candidates["maturity_months"]) == {1, 3, 6, 12}
    absent_task = (
        tables["task_coverage"]
        .loc[lambda frame: frame["task_id"].eq("delta_log_dispersion__3m__market__fh2")]
        .iloc[0]
    )
    assert absent_task["configured_task"]
    assert absent_task["source_rows"] == 0


def test_cli_writes_trackable_json_and_markdown_without_overwrite(tmp_path: Path):
    label_dir = tmp_path / "labels"
    label_dir.mkdir()
    _reports().to_parquet(label_dir / "reports.parquet", index=False)
    _residual().assign(residual_signed_raw=np.nan).to_parquet(
        label_dir / "report_fy_labels.parquet", index=False
    )
    _dispersion().assign(delta_log_dispersion=np.nan).to_parquet(
        label_dir / "report_confirmation_labels.parquet", index=False
    )
    (label_dir / "label_metadata.json").write_text(
        json.dumps({"label_version": "v1"}), encoding="utf-8"
    )
    config = _config(label_dir)
    config_path = tmp_path / "probe.yaml"
    import yaml

    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    output_root = tmp_path / "audit_reports" / "continuous_label_audit"
    args = argparse.Namespace(
        config=str(config_path),
        label_dir=None,
        output_root=str(output_root),
        run_id="server_audit",
        maturity_months=(1, 3, 6, 12),
    )
    manifest = run(args)
    output = output_root / "server_audit"
    assert manifest["outcome_blind"] is True
    assert manifest["label_value_columns_loaded"] == []
    assert manifest["source_integrity"]["label_version_consistent"] is True
    assert (output / "manifest.json").is_file()
    assert (output / "SUMMARY.md").is_file()
    assert (output / "candidate_window_counts.json").is_file()
    assert {path.suffix for path in output.iterdir()} <= {".json", ".md"}

    import pytest

    with pytest.raises(FileExistsError, match="拒绝覆盖"):
        run(args)
