import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from label_engineering.build_probe_dataset import run
from src.probe_dataset import build_probe_bundle, load_probe_task


LABEL_VERSION = "report_future_v1.0"


def _reports() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "report_id": "r1",
                "stock_code": "000001",
                "org_id": "A",
                "title": "训练报告",
                "available_date": "2021-01-10",
                "text": "训练文本",
            },
            {
                "report_id": "r2",
                "stock_code": "000002",
                "org_id": "B",
                "title": "验证报告",
                "available_date": "2022-05-10",
                "text": "验证文本",
            },
            {
                "report_id": "r3",
                "stock_code": "000003",
                "org_id": "C",
                "title": "测试报告",
                "available_date": "2023-09-10",
                "text": "测试文本",
            },
        ]
    )


def _fy_labels() -> pd.DataFrame:
    rows = []
    for report_id, stock, available, fy, label_date, value in (
        ("r1", "000001", "2021-01-10", 2021, "2022-03-01", -0.2),
        ("r2", "000002", "2022-05-10", 2022, "2023-08-01", 0.1),
        ("r3", "000003", "2023-09-10", 2023, "2024-03-01", 0.3),
    ):
        rows.append(
            {
                "report_id": report_id,
                "stock_code": stock,
                "available_date": available,
                "fy": fy,
                "forecast_horizon": 0,
                "residual_signed_raw": value,
                "residual_valid": 1,
                "actual_label_available_date": label_date,
                "actual_known_date": label_date,
                "actual_publish_date": label_date,
                "actual_invalid_reason": None,
                "n_org_pre": 5,
                "scale_t": 100.0,
                "label_version": LABEL_VERSION,
            }
        )
    rows.append(
        {
            "report_id": "r1",
            "stock_code": "000001",
            "available_date": "2021-01-10",
            "fy": 2022,
            "forecast_horizon": 1,
            "residual_signed_raw": np.nan,
            "residual_valid": 0,
            "actual_label_available_date": pd.NaT,
            "actual_known_date": pd.NaT,
            "actual_publish_date": pd.NaT,
            "actual_invalid_reason": "actual_missing",
            "n_org_pre": 5,
            "scale_t": 100.0,
            "label_version": LABEL_VERSION,
        }
    )
    return pd.DataFrame(rows)


def _confirmation() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "report_id": "r1",
                "stock_code": "000001",
                "available_date": "2021-01-10",
                "fy": 2021,
                "forecast_horizon": 0,
                "confirmation_months": 3,
                "peer_panel": "fixed",
                "target_date": "2022-05-01",
                "delta_log_dispersion": -0.3,
                "confirmation_valid": 1,
                "confirmation_probe_valid": 0,
                "sample_weight": 0.8,
                "label_available_date": "2022-05-01",
                "n_org_pre": 5,
                "n_org_future": 5,
                "n_peer_updates": 1,
                "dispersion_pre": 0.2,
                "dispersion_future": 0.1,
                "invalid_reason": None,
                "probe_invalid_reason": "insufficient_peer_updates",
                "label_version": LABEL_VERSION,
            },
            {
                "report_id": "r2",
                "stock_code": "000002",
                "available_date": "2022-05-10",
                "fy": 2022,
                "forecast_horizon": 0,
                "confirmation_months": 1,
                "peer_panel": "active",
                "target_date": "2023-07-01",
                "delta_log_dispersion": 0.4,
                "confirmation_valid": 1,
                "confirmation_probe_valid": 1,
                "sample_weight": 1.0,
                "label_available_date": "2023-07-01",
                "n_org_pre": 4,
                "n_org_future": 4,
                "n_peer_updates": 4,
                "dispersion_pre": 0.1,
                "dispersion_future": 0.15,
                "invalid_reason": None,
                "probe_invalid_reason": None,
                "label_version": LABEL_VERSION,
            },
            {
                "report_id": "r3",
                "stock_code": "000003",
                "available_date": "2023-09-10",
                "fy": 2023,
                "forecast_horizon": 0,
                "confirmation_months": 1,
                "peer_panel": "active",
                "target_date": "2023-10-10",
                "delta_log_dispersion": np.nan,
                "confirmation_valid": 0,
                "confirmation_probe_valid": 0,
                "sample_weight": 0.0,
                "label_available_date": "2023-10-10",
                "n_org_pre": 2,
                "n_org_future": 2,
                "n_peer_updates": 0,
                "dispersion_pre": np.nan,
                "dispersion_future": np.nan,
                "invalid_reason": "insufficient_pre_peers",
                "probe_invalid_reason": None,
                "label_version": LABEL_VERSION,
            },
        ]
    )


def _safe_splits() -> dict:
    return {
        "enabled": True,
        "train": {
            "feature_start": "2021-01-01",
            "feature_end": "2021-12-31",
            "label_cutoff": "2022-04-30",
        },
        "validation": {
            "feature_start": "2022-05-01",
            "feature_end": "2022-12-31",
            "label_cutoff": "2023-08-31",
        },
        "test": {
            "feature_start": "2023-09-01",
            "feature_end": "2023-12-31",
            "label_cutoff": None,
        },
    }


def test_bundle_keeps_text_once_and_builds_separate_tasks():
    bundle = build_probe_bundle(_reports(), _fy_labels(), _confirmation())
    assert bundle.metadata["training_ready"] is False
    assert set(bundle.targets["split"]) == {"unassigned"}
    assert len(bundle.targets) == 5  # 3个Residual + 2个正式有效Dispersion
    assert len(bundle.texts) == 3
    assert bundle.texts["report_id"].is_unique
    assert bundle.targets["sample_id"].is_unique
    assert set(bundle.targets["label_name"]) == {
        "residual_signed_raw",
        "delta_log_dispersion",
    }
    assert bundle.targets["task_id"].nunique() == 3
    residual = bundle.targets[bundle.targets["label_name"] == "residual_signed_raw"]
    assert residual["target_weight"].eq(1.0).all()
    dispersion = bundle.targets[bundle.targets["label_name"] == "delta_log_dispersion"]
    assert set(dispersion["target_weight"]) == {0.8, 1.0}


def test_probe_valid_filter_is_optional_but_never_ignores_main_validity():
    bundle = build_probe_bundle(
        _reports(),
        _fy_labels(),
        _confirmation(),
        target_config={"dispersion": {"enabled": True, "require_probe_valid": True}},
    )
    dispersion = bundle.targets[bundle.targets["label_name"] == "delta_log_dispersion"]
    assert dispersion["report_id"].tolist() == ["r2"]
    assert dispersion["confirmation_valid"].eq(1).all()
    assert dispersion["confirmation_probe_valid"].eq(1).all()


def test_same_day_or_past_label_is_rejected_as_pit():
    fy = _fy_labels()
    fy.loc[
        fy["report_id"].eq("r1") & fy["residual_valid"].eq(1),
        "actual_label_available_date",
    ] = "2021-01-10"
    with pytest.raises(ValueError, match="潜在PIT"):
        build_probe_bundle(_reports(), fy, _confirmation())


def test_time_splits_drop_labels_not_known_by_next_split():
    bundle = build_probe_bundle(
        _reports(),
        _fy_labels(),
        _confirmation(),
        split_config=_safe_splits(),
    )
    assert bundle.metadata["training_ready"] is True
    assert bundle.metadata["training_ready_tasks"] == ["residual_signed_raw__fh0"]
    # r1的3个月Dispersion到2022-05-01才可知，晚于训练cutoff 2022-04-30。
    assert not (
        bundle.targets["report_id"].eq("r1")
        & bundle.targets["label_name"].eq("delta_log_dispersion")
    ).any()
    split_by_report = (
        bundle.targets.groupby("report_id")["split"]
        .agg(lambda values: tuple(values.unique()))
        .to_dict()
    )
    assert split_by_report == {
        "r1": ("train",),
        "r2": ("validation",),
        "r3": ("test",),
    }
    late = bundle.audit[bundle.audit["reason"].eq("label_not_known_by_split_cutoff")]
    assert late["count"].sum() == 1


def test_cli_writes_bundle_and_loader_joins_one_safe_task(tmp_path: Path):
    label_dir = tmp_path / "labels"
    label_dir.mkdir()
    _reports().to_parquet(label_dir / "reports.parquet", index=False)
    _fy_labels().to_parquet(label_dir / "report_fy_labels.parquet", index=False)
    _confirmation().to_parquet(
        label_dir / "report_confirmation_labels.parquet", index=False
    )
    (label_dir / "label_metadata.json").write_text(
        json.dumps({"label_version": LABEL_VERSION}), encoding="utf-8"
    )
    output = tmp_path / "bundle"
    config = tmp_path / "probe.yaml"
    config.write_text(
        f"""
paths:
  report_labels: {label_dir}
targets:
  forecast_horizons: [0, 1, 2]
  confirmation_months: [1, 3]
  peer_panels: [fixed, market, active]
  residual:
    enabled: true
  dispersion:
    enabled: true
    require_probe_valid: false
splits:
  enabled: true
  train:
    feature_start: 2021-01-01
    feature_end: 2021-12-31
    label_cutoff: 2022-04-30
  validation:
    feature_start: 2022-05-01
    feature_end: 2022-12-31
    label_cutoff: 2023-08-31
  test:
    feature_start: 2023-09-01
    feature_end: 2023-12-31
    label_cutoff: null
performance:
  shared_server: true
  max_threads: 2
  nice_increment: 0
  blas_threads: 1
output:
  directory: {output}
""",
        encoding="utf-8",
    )
    metadata = run(argparse.Namespace(config=str(config), output_dir=None))
    assert metadata["training_ready"] is True
    assert (output / "probe_texts.parquet").is_file()
    assert (output / "probe_targets.parquet").is_file()
    task = load_probe_task(
        output,
        task_id="residual_signed_raw__fh0",
        split="train",
    )
    assert task[["report_id", "text", "label_value"]].to_dict("records") == [
        {"report_id": "r1", "text": "训练文本", "label_value": -0.2}
    ]
    with pytest.raises(ValueError, match="完整且有效"):
        load_probe_task(
            output,
            task_id="delta_log_dispersion__1m__active__fh0",
            split="validation",
        )


def test_cli_rejects_mismatched_upstream_manifest(tmp_path: Path):
    label_dir = tmp_path / "labels"
    label_dir.mkdir()
    _reports().to_parquet(label_dir / "reports.parquet", index=False)
    _fy_labels().to_parquet(label_dir / "report_fy_labels.parquet", index=False)
    _confirmation().to_parquet(
        label_dir / "report_confirmation_labels.parquet", index=False
    )
    (label_dir / "label_metadata.json").write_text(
        json.dumps({"label_version": "wrong_version"}), encoding="utf-8"
    )
    config = tmp_path / "probe.yaml"
    config.write_text(
        f"""
paths:
  report_labels: {label_dir}
splits:
  enabled: false
performance:
  shared_server: true
  max_threads: 1
  nice_increment: 0
  blas_threads: 1
output:
  directory: {tmp_path / 'bundle'}
""",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="label_version不一致"):
        run(argparse.Namespace(config=str(config), output_dir=None))


def test_mismatched_report_identity_is_rejected():
    confirmation = _confirmation()
    confirmation.loc[0, "stock_code"] = "999999"
    with pytest.raises(ValueError, match="stock_code不一致"):
        build_probe_bundle(_reports(), _fy_labels(), confirmation)
