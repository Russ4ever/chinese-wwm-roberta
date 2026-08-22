import ast
from pathlib import Path

import nbformat
import numpy as np
import pandas as pd
import torch

from src.layer_probe_factors import (
    benjamini_hochberg,
    build_cross_layer_factors,
    evaluate_factor_ic,
    evaluate_incremental_ic,
    factor_columns,
)
from src.layer_probe_models import (
    assign_purged_time_splits,
    fit_return_layer_probes,
    fit_sentiment_layer_probes,
)
from src.layer_probe_panel import (
    aggregate_stock_day_array,
    attach_forward_returns,
)
from src.layer_probe_representations import (
    extract_cls_representations,
    freeze_and_validate_inference_model,
    load_text_dataset,
    stack_layer_cls,
    validate_representation_artifacts,
)


def _split_config():
    return {
        "train": {"start": "2021-01-01", "end": "2021-01-10"},
        "validation": {"start": "2021-01-15", "end": "2021-01-20"},
        "test": {"start": "2021-01-25", "end": "2021-01-31"},
    }


def test_text_loading_layer_stack_and_strict_inference_state(tmp_path: Path):
    texts = pd.DataFrame(
        {
            "id": ["a", "b"],
            "body": ["甲", "乙"],
            "stock": ["000001.SZ", "600000.SH"],
            "date": ["2021-01-01", "2021-01-02"],
        }
    )
    text_path = tmp_path / "texts.parquet"
    texts.to_parquet(text_path, index=False)
    loaded = load_text_dataset(
        text_path,
        id_column="id",
        text_column="body",
        symbol_column="stock",
        date_column="date",
    )
    assert loaded["symbol"].tolist() == ["000001", "600000"]
    assert "sentiment_label" not in loaded
    assert loaded["report_id"].tolist() == ["a", "b"]
    assert loaded["representation_row"].tolist() == [0, 1]

    hidden = [torch.full((2, 3, 4), float(layer)) for layer in range(13)]
    stacked = stack_layer_cls(hidden, expected_layers=13)
    assert stacked.shape == (2, 13, 4)
    assert torch.equal(stacked[:, 12], torch.full((2, 4), 12.0))

    model = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.Dropout(0.5))
    model.train()
    audit = freeze_and_validate_inference_model(model)
    assert audit["trainable_parameter_count"] == 0
    assert audit["dropout_modules_in_training_mode"] == []
    assert model.training is False


def test_representation_extraction_uses_one_forward_for_all_layers(tmp_path: Path):
    class ToyBert(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = torch.nn.Embedding(32, 4)
            self.dropout = torch.nn.Dropout(0.5)
            self.config = type("Config", (), {"num_hidden_layers": 12})()

    class ToyCandidate(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.bert = ToyBert()
            self.fc = torch.nn.Linear(4, 2)
            self.forward_calls = 0

        def forward(
            self,
            input_ids,
            attention_mask=None,
            token_type_ids=None,
            output_hidden_states=False,
        ):
            self.forward_calls += 1
            base = self.bert.dropout(self.bert.embedding(input_ids))
            hidden = tuple(base + layer for layer in range(13))
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

    class ToyTokenizer:
        def __call__(self, texts, **kwargs):
            width = max(len(text) for text in texts) + 1
            ids = torch.zeros((len(texts), width), dtype=torch.long)
            mask = torch.zeros_like(ids)
            for row, text in enumerate(texts):
                length = len(text) + 1
                ids[row, :length] = torch.arange(1, length + 1)
                mask[row, :length] = 1
            return {
                "input_ids": ids,
                "attention_mask": mask,
                "token_type_ids": torch.zeros_like(ids),
            }

    texts = pd.DataFrame(
        {
            "representation_row": [0, 1, 2],
            "report_id": ["a", "b", "c"],
            "text": ["甲", "乙乙", "丙丙丙"],
            "symbol": ["000001", "000002", "000003"],
            "feature_available_date": pd.to_datetime(["2021-01-01", "2021-01-02", "2021-01-03"]),
            "text_sha256": ["a", "b", "c"],
        }
    )
    model = ToyCandidate()
    artifacts = extract_cls_representations(
        texts,
        model=model,
        tokenizer=ToyTokenizer(),
        output_directory=tmp_path / "representations",
        batch_size=2,
        max_length=8,
        device="cpu",
        storage_dtype="float32",
        model_identity={"checkpoint_sha256": "toy"},
    )
    assert model.forward_calls == 2  # 两个batch；不是13层各跑一次。
    check = validate_representation_artifacts(artifacts.directory)
    assert check["rows"] == 3
    assert check["layers"] == 13
    assert check["hidden_size"] == 4
    assert check["dtype"] == "float32"
    assert check["fixed_head_rows"] == 39
    assert check["head_recipe"] == "cls_fc"


def test_purged_split_drops_future_window_crossing_boundary():
    frame = pd.DataFrame(
        {
            "trading_date": ["2021-01-05", "2021-01-09", "2021-01-18", "2021-01-26"],
            "label_end": ["2021-01-12", "2021-01-16", "2021-01-24", "2021-01-29"],
        }
    )
    split, audit = assign_purged_time_splits(
        frame,
        config=_split_config(),
        date_column="trading_date",
        label_end_column="label_end",
    )
    assert split["split"].tolist() == ["train", "validation", "test"]
    assert pd.Timestamp("2021-01-09") not in set(split["trading_date"])
    purged = audit[audit["reason"].eq("purged_future_label_crosses_next_split")]
    assert purged["count"].iloc[0] == 1


def _sentiment_data(seed: int = 7):
    rng = np.random.default_rng(seed)
    n_per_split = 40
    dates = list(pd.date_range("2021-01-01", periods=10))
    dates += list(pd.date_range("2021-01-15", periods=6))
    dates += list(pd.date_range("2021-01-25", periods=7))
    dates = np.repeat(dates, np.ceil((3 * n_per_split) / len(dates)).astype(int))[
        : 3 * n_per_split
    ]
    representations = rng.normal(size=(3 * n_per_split, 13, 5)).astype("float32")
    label = (representations[:, 12, 0] > 0).astype(int)
    metadata = pd.DataFrame(
        {
            "representation_row": np.arange(len(label)),
            "text_id": [f"t{i}" for i in range(len(label))],
            "trading_date": dates,
            "sentiment_label": label,
        }
    )
    logit = representations[:, 12, 0] * 0.1
    probability = 1 / (1 + np.exp(-logit))
    head = pd.DataFrame(
        {
            "representation_row": np.arange(len(label)),
            "text_id": metadata["text_id"],
            "logit_margin_1_minus_0": logit,
            "class_0_prob": 1 - probability,
            "class_1_prob": probability,
        }
    )
    return representations, metadata, head


def test_sentiment_probes_create_13_oos_models_and_head_comparison():
    representations, metadata, head = _sentiment_data()
    metrics, predictions, distributions, _ = fit_sentiment_layer_probes(
        representations,
        metadata,
        head,
        split_config=_split_config(),
        c_grid=[1.0],
        positive_class_index=1,
        label_mapping_confirmed=False,
        include_test=True,
    )
    layer_metrics = metrics[metrics["model_kind"].eq("layer_logistic")]
    assert len(layer_metrics) == 26
    assert set(layer_metrics["layer"]) == set(range(13))
    assert set(predictions["split"]) == {"validation", "test"}
    assert set(predictions["model_kind"]) == {
        "layer_logistic",
        "original_fc_head",
    }
    assert distributions["probability_near_half_fraction"].between(0, 1).all()


def test_stock_day_aggregation_and_forward_return_orientation(tmp_path: Path):
    representations = np.zeros((4, 13, 2), dtype="float32")
    representations[0] = 1
    representations[1] = 3
    representations[2] = 5
    representations[3] = 7
    metadata = pd.DataFrame(
        {
            "representation_row": np.arange(4),
            "symbol": ["000001", "000001", "000002", "000002"],
            "trading_date": ["2021-01-04", "2021-01-04", "2021-01-04", "2021-01-05"],
        }
    )
    groups, array = aggregate_stock_day_array(
        representations,
        metadata,
        output_path=tmp_path / "stock.npy",
        storage_dtype="float32",
        chunk_size=2,
    )
    assert groups["n_texts"].tolist() == [2, 1, 1]
    assert np.allclose(array[0], 2.0)
    del array

    dates = pd.date_range("2021-01-04", periods=8, freq="B")
    daily = pd.DataFrame(
        {"000001": np.full(8, 0.01), "000002": np.full(8, 0.02)},
        index=dates,
    )
    panel = attach_forward_returns(
        groups,
        daily,
        horizons=[1, 5],
        primary_horizon=5,
    )
    first = panel.iloc[0]
    assert np.isclose(first["industry_adjusted_return_fut1d"], 0.01)
    assert np.isclose(first["industry_adjusted_return_fut5d"], 1.01**5 - 1)
    assert first["label_end_date_5d"] == dates[5]
    assert panel["target_return_rank_5d"].dropna().between(-0.5, 0.5).all()


def _return_probe_data(seed: int = 11):
    rng = np.random.default_rng(seed)
    dates = list(pd.date_range("2021-01-01", periods=10))
    dates += list(pd.date_range("2021-01-15", periods=6))
    dates += list(pd.date_range("2021-01-25", periods=7))
    rows = [(date, stock) for date in dates for stock in range(4)]
    n = len(rows)
    representations = rng.normal(size=(n, 13, 4)).astype("float32")
    target = representations[:, 8, 0] + rng.normal(scale=0.05, size=n)
    panel = pd.DataFrame(
        {
            "representation_row": np.arange(n),
            "trading_date": [row[0] for row in rows],
            "symbol": [f"{row[1] + 1:06d}" for row in rows],
            "split": [
                (
                    "train"
                    if row[0] <= pd.Timestamp("2021-01-10")
                    else (
                        "validation" if row[0] <= pd.Timestamp("2021-01-20") else "test"
                    )
                )
                for row in rows
            ],
            "n_texts": 1,
            "fixed_head_margin": rng.normal(size=n),
            "target_return_rank_5d": target,
            "industry_adjusted_return_fut5d": target / 100,
            "industry": [row[1] % 2 for row in rows],
            "size": rng.normal(size=n),
        }
    )
    return representations, panel


def test_return_probes_and_cross_layer_factor_validation():
    representations, panel = _return_probe_data()
    metrics, predictions, _ = fit_return_layer_probes(
        representations,
        panel,
        target_column="target_return_rank_5d",
        alpha_grid=[1.0],
        min_daily_observations=3,
        include_test=True,
    )
    assert len(metrics) == 26
    assert set(predictions["prediction_role"]) == {"fit_reference", "oos"}
    assert set(predictions["evaluation_split"]) == {"validation", "test"}

    reference_long = predictions[
        predictions["evaluation_split"].eq("validation")
        & predictions["prediction_role"].eq("fit_reference")
    ]
    evaluation_long = predictions[
        predictions["evaluation_split"].eq("validation")
        & predictions["prediction_role"].eq("oos")
    ]

    def wide(frame):
        values = frame.pivot(
            index="representation_row", columns="layer", values="prediction"
        ).rename(columns=lambda value: f"layer_{value}")
        metadata = frame.drop(columns=["layer", "prediction"]).drop_duplicates(
            "representation_row"
        )
        return metadata.merge(values.reset_index(), on="representation_row")

    factors, transform = build_cross_layer_factors(
        wide(reference_long), wide(evaluation_long), pca_components=2
    )
    candidates = factor_columns(factors)
    assert len(candidates) == 19
    assert len(transform["pca_loadings"]) == 2
    summary, daily = evaluate_factor_ic(
        factors,
        factors=candidates[:3],
        target_column="target_return_rank_5d",
        min_daily_observations=3,
    )
    assert len(summary) == 3
    assert not daily.empty
    incremental, _ = evaluate_incremental_ic(
        factors,
        factors=candidates[:2],
        target_column="target_return_rank_5d",
        min_daily_observations=3,
    )
    assert len(incremental) == 2
    adjusted = benjamini_hochberg(pd.Series([0.01, 0.04, 0.03, np.nan]))
    assert np.allclose(adjusted[:3], [0.03, 0.04, 0.04])
    assert np.isnan(adjusted[3])


def test_notebook_only_executes_python_tools_and_has_final_test_gate():
    path = (
        Path(__file__).resolve().parents[1] / "notebooks" / "layer_probe_pipeline.ipynb"
    )
    notebook = nbformat.read(path, as_version=4)
    nbformat.validate(notebook)
    assert len({cell.id for cell in notebook.cells}) == len(notebook.cells)
    definition_cells = []
    for index, cell in enumerate(notebook.cells):
        if cell.cell_type != "code":
            continue
        tree = ast.parse(cell.source)
        if any(
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
            for node in ast.walk(tree)
        ):
            definition_cells.append(index)
    assert definition_cells == []
    final_cells = [
        cell
        for cell in notebook.cells
        if "final-test-once" in cell.metadata.get("tags", [])
    ]
    assert len(final_cells) == 1
    assert "RUN_FINAL_TEST_CELL" in final_cells[0].source
