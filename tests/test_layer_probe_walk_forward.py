import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.layer_probe_label_factors import (
    aggregate_label_predictions_to_stock_day,
    layer_correlation_tables,
)
from src.layer_probe_representations import (
    extract_cls_representations,
    write_representation_pointer,
)
from src.layer_probe_walk_forward import (
    align_walk_forward_targets,
    fold_masks,
    merge_walk_forward_probe_shards,
    parse_walk_forward_protocol,
    run_walk_forward_fixed_head_stage,
    run_walk_forward_probe_stage,
    validate_walk_forward_fixed_head_outputs,
    validate_walk_forward_probe_outputs,
    validate_walk_forward_targets,
)


def _protocol_config():
    return {
        "walk_forward": {
            "enabled": True,
            "history_start": "2020-01-01",
            "validation": {
                "feature_start": "2021-01-01",
                "feature_end": "2022-12-31",
                "selection_label_cutoff": "2022-12-31",
                "fold_frequency": "year",
                "minimum_folds": 2,
            },
            "final_test": {
                "feature_start": "2023-01-01",
                "feature_end": "2023-12-31",
                "label_cutoff": "2024-12-31",
            },
            "minimum_train_rows": 4,
            "minimum_evaluation_rows": 4,
        }
    }


def test_walk_forward_fold_masks_enforce_feature_and_label_cutoffs():
    protocol = parse_walk_forward_protocol(_protocol_config())
    fold = protocol.folds[0]
    task = pd.DataFrame(
        {
            "feature_available_date": pd.to_datetime(
                ["2020-01-01", "2020-02-01", "2021-01-01", "2021-02-01"]
            ),
            "label_available_date": pd.to_datetime(
                ["2020-06-01", "2021-03-01", "2021-06-01", "2023-01-01"]
            ),
            "target_weight": [1.0, 1.0, 1.0, 1.0],
        }
    )
    train, evaluation = fold_masks(task, fold, protocol)
    assert train.tolist() == [True, False, False, False]
    assert evaluation.tolist() == [False, False, True, False]


def test_stock_day_label_factor_aggregation_is_unweighted():
    predictions = pd.DataFrame(
        {
            "sample_id": ["s1", "s2"],
            "report_id": ["r1", "r2"],
            "task_id": ["residual_signed_raw__fh0"] * 2,
            "label_name": ["residual_signed_raw"] * 2,
            "stock_code": ["000001", "000001"],
            "feature_available_date": pd.to_datetime(["2021-01-04"] * 2),
            "label_available_date": pd.to_datetime(["2021-06-01"] * 2),
            "layer": [0, 0],
            "prediction": [1.0, 3.0],
            "label_value": [2.0, 6.0],
            "target_weight": [1000.0, 1.0],
            "prediction_role": ["oos", "oos"],
        }
    )
    stock_day = aggregate_label_predictions_to_stock_day(predictions)
    assert len(stock_day) == 1
    assert stock_day.loc[0, "factor_value"] == 2.0
    assert stock_day.loc[0, "realized_label_mean"] == 4.0
    assert stock_day.loc[0, "n_reports"] == 2


def test_daily_layer_correlations_are_symmetric_upper_triangle():
    rows = []
    for symbol in range(5):
        for layer, value in ((0, float(symbol)), (1, float(-symbol))):
            rows.append(
                {
                    "factor_source": "toy",
                    "task_id": "task",
                    "trading_date": pd.Timestamp("2021-01-04"),
                    "symbol": f"{symbol + 1:06d}",
                    "layer": layer,
                    "factor_value": value,
                }
            )
    daily, summary = layer_correlation_tables(
        pd.DataFrame(rows), minimum_observations=3
    )
    pair = daily[(daily["layer_left"].eq(0)) & (daily["layer_right"].eq(1))]
    assert np.isclose(pair["spearman"].iloc[0], -1.0)
    assert (daily["layer_left"] <= daily["layer_right"]).all()
    diagonal = summary[
        summary["layer_left"].eq(0) & summary["layer_right"].eq(0)
    ]
    assert np.isclose(diagonal["mean_spearman"].iloc[0], 1.0)


class _ToyBert(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(128, 4)
        self.dropout = torch.nn.Dropout(0.5)
        self.config = type("Config", (), {"num_hidden_layers": 12})()


class _ToyCandidate(torch.nn.Module):
    pooling = "cls"

    def __init__(self):
        super().__init__()
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
        hidden = tuple(base + layer / 10 for layer in range(13))
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


def _walk_forward_artifacts(tmp_path: Path, task_ids=("residual_signed_raw__fh0",)):
    from blake3 import blake3

    rows_per_year = 8
    dates = pd.DatetimeIndex(
        np.concatenate(
            [
                pd.date_range(f"{year}-01-02", periods=rows_per_year, freq="D")
                for year in (2020, 2021, 2022)
            ]
        )
    )
    n = len(dates)
    reports = pd.DataFrame(
        {
            "representation_row": np.arange(n),
            "report_id": [f"r{index:03d}" for index in range(n)],
            "text": ["甲" * (index + 1) for index in range(n)],
            "symbol": [f"{index + 1:06d}" for index in range(n)],
            "feature_available_date": dates,
            "text_sha256": [f"hash-{index}" for index in range(n)],
        }
    )
    artifacts = extract_cls_representations(
        reports,
        model=_ToyCandidate(),
        tokenizer=_ToyTokenizer(),
        output_directory=tmp_path / "representation_store" / "toy",
        batch_size=6,
        max_length=8,
        storage_dtype="float32",
        device="cpu",
        model_identity={
            "checkpoint_sha256": "toy",
            "representation_fingerprint": "toy-walk-forward",
        },
    )
    run = tmp_path / "runs" / "walk-forward"
    write_representation_pointer(run, artifacts)

    bundle = tmp_path / "targets"
    bundle.mkdir()
    source_records = []
    for filename in (
        "report_fy_labels.parquet",
        "report_confirmation_labels.parquet",
    ):
        source = bundle / filename
        pd.DataFrame({"source_marker": [filename]}).to_parquet(source, index=False)
        digest = blake3()
        digest.update_mmap(str(source))
        source_records.append(
            {
                "path": str(source.resolve()),
                "size_bytes": source.stat().st_size,
                "blake3": digest.hexdigest(),
            }
        )
    target_rows = []
    for task_index, task_id in enumerate(task_ids):
        for index in range(n):
            target_rows.append(
                {
                    "sample_id": f"s{task_index}_{index:03d}",
                    "report_id": reports["report_id"].iloc[index],
                    "task_id": task_id,
                    "split": "unassigned",
                    "label_name": task_id.rsplit("__", 1)[0],
                    "label_value": float(np.sin(index) + index / 10 + task_index),
                    "target_weight": 1.0,
                    "feature_available_date": dates[index],
                    "label_available_date": dates[index] + pd.Timedelta(days=120),
                    "forecast_horizon": 0,
                    "label_version": "toy-v1",
                    "stock_code": reports["symbol"].iloc[index],
                }
            )
    targets = pd.DataFrame(target_rows)
    targets.to_parquet(bundle / "probe_targets.parquet", index=False)
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
    pd.DataFrame(
        [{"stage": "source_valid", "reason": "toy", "count": n}]
    ).to_csv(bundle / "probe_merge_audit.csv", index=False)
    config = {
        **_protocol_config(),
        "output": {"run_directory": str(run)},
        "continuous_targets": {"bundle_directory": str(bundle)},
        "continuous_probe": {
            "alpha_grid": [0.1, 1.0],
            "solver": "lsqr",
            "tolerance": 1e-6,
            "maximum_iterations": 1000,
        },
        "strict_test": {"open_final_test": False, "selected_factors": []},
    }
    return config


def test_walk_forward_artifact_pipeline_never_loads_final_test(tmp_path: Path):
    config = _walk_forward_artifacts(tmp_path)
    aligned = align_walk_forward_targets(config)
    check = validate_walk_forward_targets(aligned)
    assert check == {"rows": 24, "tasks": 1, "folds": 2}

    association = run_walk_forward_fixed_head_stage(config)
    assert validate_walk_forward_fixed_head_outputs(association)["tasks"] == 1

    probe = run_walk_forward_probe_stage(config)
    result = validate_walk_forward_probe_outputs(probe)
    assert result["tasks"] == 1
    assert result["folds"] == 2
    predictions = pd.read_parquet(probe / "walk_forward_oos_predictions.parquet")
    assert pd.to_datetime(predictions["feature_available_date"]).max().year == 2022
    assert set(predictions["layer"]) == set(range(13))


def test_walk_forward_probe_shard_merge_matches_sequential(tmp_path: Path):
    import shutil

    task_ids = ("residual_signed_raw__fh0", "residual_signed_raw__fh1")
    config = _walk_forward_artifacts(tmp_path, task_ids=task_ids)
    align_walk_forward_targets(config)

    sequential = run_walk_forward_probe_stage(config)
    seq_pred = pd.read_parquet(
        sequential / "walk_forward_oos_predictions.parquet"
    )
    seq_sel = pd.read_csv(sequential / "walk_forward_selected_alphas.csv")

    # Remove the canonical output so the sharded path writes a fresh merge.
    shutil.rmtree(sequential)
    run_walk_forward_probe_stage(config, task_ids=[task_ids[0]], shard_tag="a")
    run_walk_forward_probe_stage(config, task_ids=[task_ids[1]], shard_tag="b")
    merged = merge_walk_forward_probe_shards(config, ["a", "b"])
    assert validate_walk_forward_probe_outputs(merged)["tasks"] == 2

    merged_pred = pd.read_parquet(
        merged / "walk_forward_oos_predictions.parquet"
    )
    merged_sel = pd.read_csv(merged / "walk_forward_selected_alphas.csv")

    sort_cols = ["task_id", "layer", "sample_id"]
    pd.testing.assert_frame_equal(
        merged_pred.sort_values(sort_cols).reset_index(drop=True)[
            ["task_id", "layer", "sample_id", "prediction"]
        ].astype({"prediction": "float64"}),
        seq_pred.sort_values(sort_cols).reset_index(drop=True)[
            ["task_id", "layer", "sample_id", "prediction"]
        ].astype({"prediction": "float64"}),
        check_exact=False,
        rtol=1e-5,
    )
    sel_cols = ["task_id", "layer", "selected_alpha", "mean_fold_spearman"]
    pd.testing.assert_frame_equal(
        merged_sel.sort_values(sel_cols).reset_index(drop=True)[sel_cols],
        seq_sel.sort_values(sel_cols).reset_index(drop=True)[sel_cols],
        check_exact=False,
        rtol=1e-5,
    )
