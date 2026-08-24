import ast
import json
from pathlib import Path

import nbformat
import numpy as np
import pandas as pd
import pytest
import torch
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from src.config import load_yaml_config
from src.layer_probe_static import (
    DIRECT_TASK_ID,
    MODELED_LAYERS,
    STATIC_TASKS,
    _stats_from_arrays,
    accumulate_layer_statistics,
    aggregate_report_predictions_to_stock_day,
    apply_csi300_asof_membership,
    assign_direct_return_partitions,
    assign_static_splits,
    daily_layer_correlation_tables_wide,
    daily_rank_ic_tables,
    parse_static_protocol,
    ridge_path_from_stats,
    run_static_csi300_evaluation_stage,
    validate_static_csi300_evaluation_outputs,
)
from src.layer_probe_representations import (
    extract_cls_representations,
    write_representation_pointer,
)


ROOT = Path(__file__).resolve().parents[1]


def _config():
    return load_yaml_config(ROOT / "configs" / "layer_probe_static_fy0_csi300.yaml")


def test_static_protocol_is_exact_and_label_availability_is_not_a_split_input():
    protocol = parse_static_protocol(_config())
    assert protocol.task_ids == STATIC_TASKS
    assert protocol.modeled_layers == tuple(range(1, 13))
    assert 0 not in protocol.modeled_layers
    dates = pd.Series(pd.to_datetime(["2022-12-31", "2023-01-01", "2025-08-01"]))
    splits = assign_static_splits(dates, protocol)
    assert splits.tolist() == ["train", "validation", "test"]
    # A label date in 2026 is deliberately irrelevant to feature-date membership.
    label_available_date = pd.Timestamp("2026-06-18")
    assert label_available_date > protocol.test.end
    assert splits.iloc[2] == "test"


def test_static_protocol_rejects_layer_zero_ridge():
    config = _config()
    config["static_protocol"]["modeled_layers"] = list(range(13))
    with pytest.raises(ValueError, match="Layer 1~12"):
        parse_static_protocol(config)


def test_direct_return_partition_purges_only_cross_boundary_rows():
    protocol = parse_static_protocol(_config())
    frame = pd.DataFrame(
        {
            "report_id": ["a", "b", "c", "d", "e"],
            "split": ["train", "train", "validation", "validation", "test"],
            "label_end_date_5d": pd.to_datetime(
                [
                    "2022-12-30",
                    "2023-01-04",
                    "2024-12-30",
                    "2025-01-03",
                    "2025-08-08",
                ]
            ),
        }
    )
    selected, audit = assign_direct_return_partitions(frame, protocol)
    assert dict(zip(selected["report_id"], selected["partition"])) == {
        "a": "train_tuning",
        "b": "train_boundary",
        "c": "validation",
        "e": "test",
    }
    purged = audit.loc[
        audit["reason"].eq("purged_cross_boundary_or_missing_target"), "count"
    ].iloc[0]
    assert purged == 1


def test_weighted_primal_ridge_matches_sklearn_on_cpu():
    rng = np.random.default_rng(7)
    x_train = rng.normal(size=(240, 11)).astype(np.float32)
    y_train = x_train @ rng.normal(size=11) + rng.normal(scale=0.05, size=240)
    weights = rng.uniform(0.2, 2.0, size=240)
    x_validation = rng.normal(size=(50, 11)).astype(np.float32)
    stats = _stats_from_arrays(x_train, y_train, weights, device="cpu")
    accelerated = ridge_path_from_stats(stats, [0.1, 10.0], device="cpu")
    scaler = StandardScaler().fit(x_train, sample_weight=weights)
    for alpha, parameters in accelerated.items():
        reference = Ridge(alpha=alpha, solver="cholesky").fit(
            scaler.transform(x_train), y_train, sample_weight=weights
        )
        actual = (
            (x_validation - parameters.mean) / parameters.scale
        ) @ parameters.coef + parameters.intercept
        expected = reference.predict(scaler.transform(x_validation))
        assert np.allclose(actual, expected, atol=2e-4, rtol=2e-4)


def test_one_layer_scan_accumulates_multiple_tasks_and_rejects_layer_zero():
    rng = np.random.default_rng(11)
    representations = rng.normal(size=(40, 13, 6)).astype(np.float32)
    partitions = {
        ("task_a", "train_tuning"): (
            np.arange(0, 30, 2),
            rng.normal(size=15),
            np.ones(15),
        ),
        ("task_b", "validation"): (
            np.arange(1, 40, 2),
            rng.normal(size=20),
            np.full(20, 0.5),
        ),
    }
    result = accumulate_layer_statistics(
        representations,
        layer=1,
        partitions=partitions,
        device="cpu",
        row_chunk_size=9,
    )
    assert set(result) == set(partitions)
    assert result[("task_a", "train_tuning")].n_rows == 15
    assert result[("task_b", "validation")].n_rows == 20
    with pytest.raises(ValueError, match="Layer 1~12"):
        accumulate_layer_statistics(
            representations,
            layer=0,
            partitions=partitions,
            device="cpu",
            row_chunk_size=9,
        )


def test_report_predictions_are_aggregated_after_prediction():
    rows = []
    for report_id, base in (("r1", 1.0), ("r2", 3.0)):
        row = {
            "report_id": report_id,
            "symbol": "000001",
            "trading_date": pd.Timestamp("2024-01-02"),
            "partition": "validation",
        }
        for layer in MODELED_LAYERS:
            row[f"prediction_layer_{layer}"] = base + layer
        rows.append(row)
    stock_day = aggregate_report_predictions_to_stock_day(
        pd.DataFrame(rows), task_id=STATIC_TASKS[0]
    )
    assert len(stock_day) == 1
    assert stock_day.loc[0, "n_reports"] == 2
    assert stock_day.loc[0, "prediction_layer_1"] == 3.0
    assert "prediction_layer_0" not in stock_day


def test_csi300_membership_uses_complete_latest_snapshot_not_per_stock_carry():
    factors = pd.DataFrame(
        {
            "trading_date": pd.to_datetime(
                ["2024-01-15"] * 3 + ["2024-02-15"] * 3
            ),
            "symbol": ["000001", "000002", "000003"] * 2,
        }
    )
    weights = pd.DataFrame(
        {
            "trade_date": pd.to_datetime(
                ["2024-01-01", "2024-01-01", "2024-02-01", "2024-02-01"]
            ),
            "stock_code": ["000001", "000002", "000002", "000003"],
            "weight": [0.5, 0.5, 0.4, 0.6],
        }
    )
    result = apply_csi300_asof_membership(factors, weights)
    first = result[result["trading_date"].eq(pd.Timestamp("2024-01-15"))]
    second = result[result["trading_date"].eq(pd.Timestamp("2024-02-15"))]
    assert set(first.loc[first["csi300_member"], "symbol"]) == {"000001", "000002"}
    assert set(second.loc[second["csi300_member"], "symbol"]) == {"000002", "000003"}
    assert not second.loc[second["symbol"].eq("000001"), "csi300_member"].iloc[0]
    assert (result["csi300_snapshot_date"] <= result["trading_date"]).all()


def test_rank_ic_and_layer_correlations_cover_layers_one_to_twelve_only():
    rows = []
    for split, date in (
        ("validation", pd.Timestamp("2024-01-02")),
        ("test", pd.Timestamp("2025-01-02")),
    ):
        for value in range(1, 8):
            row = {
                "factor_source": "direct_return_probe",
                "task_id": DIRECT_TASK_ID,
                "split": split,
                "trading_date": date,
                "symbol": f"{value:06d}",
                "industry_adjusted_return_fut5d": float(value),
            }
            for layer in MODELED_LAYERS:
                row[f"prediction_layer_{layer}"] = float(value * layer)
            rows.append(row)
    factors = pd.DataFrame(rows)
    daily_ic, summary_ic = daily_rank_ic_tables(
        factors,
        target_column="industry_adjusted_return_fut5d",
        minimum_observations=3,
    )
    assert set(daily_ic["layer"]) == set(MODELED_LAYERS)
    assert np.allclose(summary_ic["mean_rank_ic"], 1.0)
    daily_corr, summary_corr = daily_layer_correlation_tables_wide(
        factors, minimum_observations=3
    )
    assert daily_corr["layer_left"].min() == 1
    assert daily_corr["layer_right"].max() == 12
    assert len(summary_corr) == 2 * (12 * 13 // 2)
    assert np.allclose(summary_corr["mean_spearman"], 1.0)


def test_static_notebook_is_run_all_orchestration_only():
    path = ROOT / "notebooks" / "layer_probe_static_fy0_csi300_pipeline.ipynb"
    notebook = nbformat.read(path, as_version=4)
    source = "\n".join(
        "".join(cell.source)
        for cell in notebook.cells
        if cell.cell_type == "code"
    )
    ast.parse(source)
    assert "run_static_report_probe_stage" in source
    assert "run_static_csi300_evaluation_stage" in source
    assert "RUN_FINAL_TEST" not in source
    assert "range(1, 13)" in source
    assert all(cell.get("execution_count") is None for cell in notebook.cells if cell.cell_type == "code")


class _ToyBert(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(128, 4)
        self.dropout = torch.nn.Dropout(0.25)
        self.config = type("Config", (), {"num_hidden_layers": 12})()


class _ToyCandidate(torch.nn.Module):
    pooling = "cls"

    def __init__(self):
        super().__init__()
        torch.manual_seed(3)
        self.bert = _ToyBert()
        self.fc = torch.nn.Linear(4, 2)

    def forward(
        self,
        input_ids,
        attention_mask=None,
        token_type_ids=None,
        output_hidden_states=False,
    ):
        base = self.bert.dropout(self.bert.embedding(input_ids))
        hidden = tuple(base + layer / 20 for layer in range(13))
        logits = self.fc(hidden[-1][:, 0])
        return type(
            "Output",
            (),
            {
                "hidden_states": hidden if output_hidden_states else None,
                "logits": logits,
                "probabilities": torch.softmax(logits, dim=1),
            },
        )()


class _ToyTokenizer:
    def __call__(self, texts, **kwargs):
        ids = torch.zeros((len(texts), 3), dtype=torch.long)
        for row, value in enumerate(texts):
            ids[row, 0] = min(127, len(value) + 1)
            ids[row, 1:] = torch.tensor([2, 3])
        return {
            "input_ids": ids,
            "attention_mask": torch.ones_like(ids),
            "token_type_ids": torch.zeros_like(ids),
        }


def _write_static_integration_inputs(tmp_path: Path):
    from blake3 import blake3

    stocks = ["000001", "000002", "000003"]
    dates = []
    for start in ("2022-10-10", "2023-10-09", "2025-01-02"):
        for date in pd.bdate_range(start, periods=10):
            dates.extend([date] * len(stocks))
    n = len(dates)
    reports = pd.DataFrame(
        {
            "representation_row": np.arange(n),
            "report_id": [f"r{index:03d}" for index in range(n)],
            "text": ["甲" * (index + 1) for index in range(n)],
            "symbol": stocks * (n // len(stocks)),
            "feature_available_date": pd.to_datetime(dates),
            "text_sha256": [f"text-{index}" for index in range(n)],
        }
    )
    artifacts = extract_cls_representations(
        reports,
        model=_ToyCandidate(),
        tokenizer=_ToyTokenizer(),
        output_directory=tmp_path / "representation_store" / "toy",
        batch_size=15,
        max_length=8,
        storage_dtype="float32",
        device="cpu",
        model_identity={
            "checkpoint_sha256": "toy",
            "representation_fingerprint": "toy-static-fy0",
        },
    )
    run = tmp_path / "runs" / "static"
    write_representation_pointer(run, artifacts)

    bundle = tmp_path / "bundle"
    bundle.mkdir()
    source_records = []
    for filename in ("report_fy_labels.parquet", "report_confirmation_labels.parquet"):
        path = bundle / filename
        pd.DataFrame({"source": [filename]}).to_parquet(path, index=False)
        digest = blake3()
        digest.update_mmap(str(path))
        source_records.append(
            {
                "path": str(path.resolve()),
                "size_bytes": path.stat().st_size,
                "blake3": digest.hexdigest(),
            }
        )
    target_rows = []
    for task_index, task_id in enumerate(STATIC_TASKS):
        label_name = (
            "residual_signed_raw"
            if task_id.startswith("residual_signed_raw")
            else "delta_log_dispersion"
        )
        for index, report in reports.iterrows():
            feature_date = report["feature_available_date"]
            label_date = feature_date + pd.Timedelta(days=45)
            if label_name == "residual_signed_raw" and feature_date.year == 2025:
                label_date = pd.Timestamp("2026-06-18")
            target_rows.append(
                {
                    "sample_id": f"{task_index}-{report.report_id}",
                    "report_id": report.report_id,
                    "task_id": task_id,
                    "split": "unassigned",
                    "label_name": label_name,
                    "label_value": float(np.sin(index / 5 + task_index) + index / 100),
                    "target_weight": 1.0,
                    "feature_available_date": feature_date,
                    "label_available_date": label_date,
                    "forecast_horizon": 0,
                    "label_version": "toy-v1",
                    "stock_code": report.symbol,
                }
            )
    pd.DataFrame(target_rows).to_parquet(bundle / "probe_targets.parquet", index=False)
    (bundle / "probe_dataset_metadata.json").write_text(
        json.dumps(
            {
                "schema_version": "probe_bundle_v1.0",
                "label_version": "toy-v1",
                "splits": [],
                "target_selection": {"forecast_horizons": [0]},
                "sources": source_records,
            }
        ),
        encoding="utf-8",
    )
    pd.DataFrame([{"stage": "toy", "reason": "valid", "count": len(target_rows)}]).to_csv(
        bundle / "probe_merge_audit.csv", index=False
    )

    return_dates = pd.bdate_range("2022-10-10", "2025-02-28")
    returns = pd.DataFrame(
        {
            stock: 0.0005 * (stock_index + 1)
            + 0.0001 * np.sin(np.arange(len(return_dates)) / (stock_index + 2))
            for stock_index, stock in enumerate(stocks)
        },
        index=return_dates,
    )
    return_path = tmp_path / "specific_returns.parquet"
    returns.to_parquet(return_path)
    csi_path = tmp_path / "csi300_weights.parquet"
    pd.DataFrame(
        [
            {
                "S_INFO_WINDCODE": "000300.SH",
                "S_CON_WINDCODE": f"{stock}.SZ",
                "TRADE_DT": date,
                "I_WEIGHT": 100 / len(stocks),
            }
            for date in (20221010, 20231009, 20250102)
            for stock in stocks
        ]
    ).to_parquet(csi_path, index=False)

    config = _config()
    config["continuous_targets"]["bundle_directory"] = str(bundle)
    config["output"]["run_directory"] = str(run)
    config["direct_return"]["industry_adjusted_daily_path"] = str(return_path)
    config["csi300_evaluation"]["weights_path"] = str(csi_path)
    config["csi300_evaluation"]["minimum_snapshot_constituents"] = 3
    config["csi300_evaluation"]["minimum_daily_observations"] = 3
    config["layer_factor_correlations"]["minimum_daily_observations"] = 3
    config["report_ridge"]["device"] = "cpu"
    config["report_ridge"]["row_chunk_candidates"] = [16]
    config["report_ridge"]["validation_sample_rows"] = 64
    config["report_ridge"]["minimum_train_rows"] = 20
    config["report_ridge"]["minimum_validation_rows"] = 20
    return config


def test_static_pipeline_toy_end_to_end(tmp_path: Path):
    config = _write_static_integration_inputs(tmp_path)
    output = run_static_csi300_evaluation_stage(config)
    check = validate_static_csi300_evaluation_outputs(output)
    assert check["tasks"] == 6
    selections = pd.read_csv(
        Path(config["output"]["run_directory"])
        / "report_ridge_probes"
        / "selected_alphas.csv"
    )
    assert set(selections["task_id"]) == set(STATIC_TASKS).union({DIRECT_TASK_ID})
    assert set(selections["layer"]) == set(MODELED_LAYERS)
    assert 0 not in set(selections["layer"])
    rank_ic = pd.read_csv(output / "rank_ic_summary.csv")
    assert set(rank_ic["split"]) == {"validation", "test"}
