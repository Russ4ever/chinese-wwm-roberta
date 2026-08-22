import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.layer_probe_continuous import (
    align_continuous_targets,
    run_continuous_probe_stage,
    run_fixed_head_analysis_stage,
    run_fixed_head_label_stage,
    validate_aligned_targets,
    validate_continuous_probe_outputs,
    validate_fixed_head_analysis_outputs,
    validate_fixed_head_label_outputs,
)
from src.layer_probe_pipeline import preflight_optional_sentiment, preflight_strict_test
from src.layer_probe_panel import (
    run_stock_day_panel_stage,
    validate_stock_day_artifacts,
)
from src.layer_probe_representations import (
    HEAD_PROVENANCE,
    HEAD_RECIPE,
    HEAD_OUTPUT_FILE,
    MANIFEST_FILE,
    METADATA_FILE,
    REPRESENTATION_FILE,
    SCHEMA_VERSION,
    disk_budget,
    extract_cls_representations,
    fingerprint_frame,
    gpu_runtime_audit,
    representation_artifacts,
    protocol_config_hash,
    sha256_file,
    write_representation_pointer,
)


class ToyBert(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(128, 4)
        self.dropout = torch.nn.Dropout(0.5)
        self.config = type("Config", (), {"num_hidden_layers": 12})()


class ToyCandidate(torch.nn.Module):
    pooling = "cls"

    def __init__(self, *, corrupt_final: bool = False):
        super().__init__()
        self.bert = ToyBert()
        self.fc = torch.nn.Linear(4, 2)
        self.forward_calls = 0
        self.corrupt_final = corrupt_final

    def forward(
        self,
        input_ids,
        attention_mask=None,
        token_type_ids=None,
        output_hidden_states=False,
    ):
        self.forward_calls += 1
        base = self.bert.dropout(self.bert.embedding(input_ids))
        hidden = tuple(base + layer / 10 for layer in range(13))
        logits = self.fc(hidden[-1][:, 0])
        if self.corrupt_final:
            logits = logits + 1
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
        ids = torch.zeros((len(texts), 3), dtype=torch.long)
        mask = torch.ones_like(ids)
        for row, value in enumerate(texts):
            ids[row, 0] = min(127, len(value) + 1)
            ids[row, 1:] = torch.tensor([2, 3])
        return {
            "input_ids": ids,
            "attention_mask": mask,
            "token_type_ids": torch.zeros_like(ids),
        }


def _build_representation(tmp_path: Path, n: int = 24):
    reports = pd.DataFrame(
        {
            "representation_row": np.arange(n),
            "report_id": [f"r{i:02d}" for i in range(n)],
            "text": ["甲" * (i + 1) for i in range(n)],
            "symbol": [f"{i + 1:06d}" for i in range(n)],
            "feature_available_date": pd.to_datetime(
                [
                    *(f"2020-01-{i + 1:02d}" for i in range(8)),
                    *(f"2021-01-{i + 1:02d}" for i in range(8)),
                    *(f"2022-01-{i + 1:02d}" for i in range(8)),
                ]
            ),
            "text_sha256": [f"hash-{i}" for i in range(n)],
        }
    )
    model = ToyCandidate()
    artifacts = extract_cls_representations(
        reports,
        model=model,
        tokenizer=ToyTokenizer(),
        output_directory=tmp_path / "representation_store" / "toy",
        batch_size=6,
        max_length=8,
        storage_dtype="float32",
        device="cpu",
        model_identity={"checkpoint_sha256": "toy", "representation_fingerprint": "toy"},
    )
    run = tmp_path / "runs" / "test"
    write_representation_pointer(run, artifacts)
    assert model.forward_calls == 4
    return reports, run


def _build_targets(tmp_path: Path, reports: pd.DataFrame) -> Path:
    from blake3 import blake3

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
    rows = []
    for index, report in reports.iterrows():
        split = "train" if index < 8 else ("validation" if index < 16 else "test")
        label_date = f"{2020 + index // 8}-06-01"
        rows.append(
            {
                "sample_id": f"s{index}",
                "task_id": "residual_signed_raw__fh0",
                "report_id": report.report_id,
                "stock_code": report.symbol,
                "feature_available_date": report.feature_available_date,
                "label_available_date": label_date,
                "split": split,
                "fy": 2020 + index // 8,
                "forecast_horizon": 0,
                "label_name": "residual_signed_raw",
                "label_value": float((index % 8) - 3.5),
                "target_weight": 0.5 if index % 3 == 0 else 1.0,
                "residual_valid": 1,
                "label_version": "test-v1",
            }
        )
    pd.DataFrame(rows).to_parquet(bundle / "probe_targets.parquet", index=False)
    metadata = {
        "label_version": "test-v1",
        "splits": [
            {
                "name": "train",
                "feature_start": "2020-01-01",
                "feature_end": "2020-12-31",
                "label_cutoff": "2020-12-31",
            },
            {
                "name": "validation",
                "feature_start": "2021-01-01",
                "feature_end": "2021-12-31",
                "label_cutoff": "2021-12-31",
            },
            {
                "name": "test",
                "feature_start": "2022-01-01",
                "feature_end": "2022-12-31",
                "label_cutoff": None,
            },
        ],
        "target_selection": {"forecast_horizons": [0]},
        "sources": source_records,
    }
    (bundle / "probe_dataset_metadata.json").write_text(
        json.dumps(metadata), encoding="utf-8"
    )
    pd.DataFrame(
        [{"label_name": "residual_signed_raw", "stage": "source_valid", "reason": "residual_valid", "count": 24}]
    ).to_csv(bundle / "probe_merge_audit.csv", index=False)
    return bundle


def _config(run: Path, bundle: Path):
    return {
        "output": {"run_directory": str(run)},
        "continuous_targets": {"bundle_directory": str(bundle)},
        "fixed_head_label_analysis": {"quantiles": 5},
        "continuous_probe": {"alpha_grid": [0.1, 1.0]},
        "strict_test": {"open_final_test": False, "selected_factors": []},
        "sentiment_appendix": {"enabled": False},
    }


def test_scheme_a_alignment_a_plus_and_continuous_probe(tmp_path: Path):
    reports, run = _build_representation(tmp_path)
    bundle = _build_targets(tmp_path, reports)
    config = _config(run, bundle)

    scheme_a = run_fixed_head_analysis_stage(config)
    assert validate_fixed_head_analysis_outputs(scheme_a)["layers"] == 13

    aligned = align_continuous_targets(config)
    assert validate_aligned_targets(aligned) == {
        "rows": 16,
        "tasks": 1,
        "splits": ["train", "validation"],
    }
    stored = pd.read_parquet(aligned / "aligned_probe_targets.parquet")
    assert stored["representation_row"].nunique() == 16
    assert stored.duplicated(["report_id", "task_id"]).sum() == 0

    association = run_fixed_head_label_stage(config, "validation")
    assert validate_fixed_head_label_outputs(association)["tasks"] == 1

    probe = run_continuous_probe_stage(config, "validation")
    result = validate_continuous_probe_outputs(probe)
    assert result["tasks"] == 1
    metrics = pd.read_csv(probe / "continuous_probe_metrics.csv")
    assert set(metrics.loc[metrics["prediction_role"].eq("oos"), "layer"]) == set(range(13))
    oos = pd.read_parquet(probe / "continuous_probe_oos_predictions.parquet")
    reference = pd.read_parquet(probe / "continuous_probe_fit_reference.parquet")
    assert oos["prediction_role"].eq("oos").all()
    assert reference["prediction_role"].eq("fit_reference").all()


def test_continuous_probe_finds_task_specific_signal_layers(tmp_path: Path):
    from blake3 import blake3

    rng = np.random.default_rng(2026)
    rows_per_split = 60
    n = rows_per_split * 3
    representations = rng.normal(size=(n, 13, 4)).astype("float32")
    report_ids = [f"signal-{index:03d}" for index in range(n)]
    dates = pd.DatetimeIndex(
        np.concatenate(
            [
                pd.date_range(f"{year}-01-01", periods=rows_per_split, freq="D")
                for year in (2020, 2021, 2022)
            ]
        )
    )
    representation_dir = tmp_path / "representation_store" / "signal"
    representation_dir.mkdir(parents=True)
    np.save(representation_dir / REPRESENTATION_FILE, representations)
    metadata = pd.DataFrame(
        {
            "representation_row": np.arange(n, dtype=np.int64),
            "report_id": report_ids,
            "symbol": [f"{index + 1:06d}" for index in range(n)],
            "feature_available_date": dates,
            "text_sha256": [f"text-{index}" for index in range(n)],
        }
    )
    metadata.to_parquet(representation_dir / METADATA_FILE, index=False)
    logits = representations[:, :, :2]
    probabilities = np.exp(logits - logits.max(axis=2, keepdims=True))
    probabilities /= probabilities.sum(axis=2, keepdims=True)
    head = pd.DataFrame(
        {
            "representation_row": np.repeat(np.arange(n), 13),
            "layer": np.tile(np.arange(13, dtype=np.int8), n),
            "class_0_logit": logits[:, :, 0].reshape(-1),
            "class_1_logit": logits[:, :, 1].reshape(-1),
            "logit_margin_1_minus_0": (logits[:, :, 1] - logits[:, :, 0]).reshape(-1),
            "class_0_prob": probabilities[:, :, 0].reshape(-1),
            "class_1_prob": probabilities[:, :, 1].reshape(-1),
            "predicted_class": np.argmax(logits, axis=2).astype("int8").reshape(-1),
        }
    )
    head.to_parquet(representation_dir / HEAD_OUTPUT_FILE, index=False)
    artifact_files = {}
    for filename in (REPRESENTATION_FILE, METADATA_FILE, HEAD_OUTPUT_FILE):
        path = representation_dir / filename
        stat = path.stat()
        artifact_files[filename] = {
            "size_bytes": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "sha256": sha256_file(path),
        }
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "representation_fingerprint": "signal-v1",
        "shape": list(representations.shape),
        "text_dataset_fingerprint": fingerprint_frame(
            metadata,
            ["report_id", "text_sha256", "symbol", "feature_available_date"],
        ),
        "layer12_equivalence": {"passed": True},
        "head_contract": {
            "recipe": HEAD_RECIPE,
            "provenance": HEAD_PROVENANCE,
            "historical_training_path_confirmed": False,
        },
        "artifact_files": artifact_files,
    }
    (representation_dir / MANIFEST_FILE).write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    run = tmp_path / "runs" / "signal"
    write_representation_pointer(run, representation_artifacts(representation_dir))

    bundle = tmp_path / "signal_targets"
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
    targets = []
    splits = np.repeat(["train", "validation", "test"], rows_per_split)
    for index in range(n):
        for horizon, signal_layer in ((0, 7), (1, 3)):
            targets.append(
                {
                    "report_id": report_ids[index],
                    "task_id": f"residual_signed_raw__fh{horizon}",
                    "split": splits[index],
                    "label_name": "residual_signed_raw",
                    "label_value": float(
                        representations[index, signal_layer, 0]
                        + rng.normal(scale=0.01)
                    ),
                    "target_weight": 1.0,
                    "feature_available_date": dates[index],
                    "label_available_date": dates[index] + pd.Timedelta(days=120),
                    "forecast_horizon": horizon,
                    "fy": dates[index].year + horizon,
                    "residual_valid": 1,
                    "label_version": "signal-v1",
                }
            )
    pd.DataFrame(targets).to_parquet(bundle / "probe_targets.parquet", index=False)
    target_metadata = {
        "label_version": "signal-v1",
        "sources": source_records,
        "splits": [
            {
                "name": split,
                "feature_start": f"{year}-01-01",
                "feature_end": f"{year}-12-31",
                "label_cutoff": f"{year}-12-31" if split != "test" else None,
            }
            for split, year in (("train", 2020), ("validation", 2021), ("test", 2022))
        ],
        "target_selection": {"forecast_horizons": [0, 1]},
    }
    (bundle / "probe_dataset_metadata.json").write_text(
        json.dumps(target_metadata), encoding="utf-8"
    )
    pd.DataFrame(
        [{"stage": "source_valid", "reason": "residual_valid", "count": len(targets)}]
    ).to_csv(bundle / "probe_merge_audit.csv", index=False)
    config = {
        "output": {"run_directory": str(run)},
        "continuous_targets": {"bundle_directory": str(bundle)},
        "continuous_probe": {"alpha_grid": [0.1, 1.0, 10.0]},
        "strict_test": {"open_final_test": False, "selected_factors": []},
    }
    output = run_continuous_probe_stage(config, "validation")
    metrics = pd.read_csv(output / "continuous_probe_metrics.csv")
    oos = metrics[metrics["prediction_role"].eq("oos")]
    best = oos.loc[oos.groupby("task_id")["spearman"].idxmax()]
    best_layers = dict(zip(best["task_id"], best["layer"]))
    assert best_layers == {
        "residual_signed_raw__fh0": 7,
        "residual_signed_raw__fh1": 3,
    }


def test_layer12_mismatch_aborts_without_artifact(tmp_path: Path):
    reports, _ = _build_representation(tmp_path / "good", n=24)
    output = tmp_path / "bad" / "representation"
    try:
        extract_cls_representations(
            reports,
            model=ToyCandidate(corrupt_final=True),
            tokenizer=ToyTokenizer(),
            output_directory=output,
            batch_size=6,
            max_length=8,
            storage_dtype="float32",
            device="cpu",
        )
    except RuntimeError as exc:
        assert "Layer 12" in str(exc)
    else:
        raise AssertionError("错误Layer 12映射必须失败")
    assert not output.exists()


def test_sentiment_is_nonblocking_when_appendix_disabled():
    report = preflight_optional_sentiment({"sentiment_appendix": {"enabled": False}})
    assert report.iloc[0]["status"] == "ok"
    assert report.iloc[0]["blocking"] == False


def test_disk_budget_enforces_reserve(monkeypatch, tmp_path: Path):
    usage = shutil_usage = type("Usage", (), {"total": 20 << 30, "used": 1 << 30, "free": 19 << 30})()
    monkeypatch.setattr("src.layer_probe_representations.shutil.disk_usage", lambda _: usage)
    try:
        disk_budget(tmp_path, rows=10_000_000, layers=13, hidden=768, dtype="float16")
    except RuntimeError as exc:
        assert "磁盘空间不足" in str(exc)
    else:
        raise AssertionError("超出安全磁盘预算必须失败")


def test_stock_day_validation_is_separate_from_final_test(tmp_path: Path):
    reports, run = _build_representation(tmp_path)
    dates = pd.date_range("2019-12-01", "2022-03-31", freq="D")
    rng = np.random.default_rng(21)
    returns = pd.DataFrame(
        rng.normal(0, 0.005, size=(len(dates), len(reports))),
        index=dates,
        columns=reports["symbol"],
    )
    return_path = tmp_path / "daily_returns.parquet"
    returns.to_parquet(return_path)
    config = {
        "output": {"run_directory": str(run)},
        "returns": {
            "industry_adjusted_daily_path": str(return_path),
            "horizons": [1, 5, 20],
            "primary_horizon": 5,
            "purge_horizon": 20,
            "storage_dtype": "float32",
            "aggregation_chunk_size": 8,
        },
        "return_time_splits": {
            "train": {"start": "2020-01-01", "end": "2020-01-31"},
            "validation": {"start": "2021-01-01", "end": "2021-01-31"},
            "test": {"start": "2022-01-01", "end": "2022-01-31"},
        },
        "exposures": {},
        "strict_test": {"open_final_test": False, "selected_factors": []},
    }
    validation = run_stock_day_panel_stage(config, "validation")
    assert validation.name == "validation"
    check = validate_stock_day_artifacts(validation)
    assert check["evaluation_split"] == "validation"
    panel = pd.read_parquet(validation / "stock_day_panel.parquet")
    assert set(panel["split"]) == {"train", "validation"}
    assert not (validation.parent / "final_test").exists()
    try:
        run_stock_day_panel_stage(config, "test")
    except RuntimeError as exc:
        assert "strict_test" in str(exc)
    else:
        raise AssertionError("最终test收益在全局门禁前必须保持不可读")
    assert not (validation.parent / "final_test").exists()


def test_gpu_physical_and_process_local_mapping(monkeypatch):
    properties = type("Properties", (), {"name": "A100", "uuid": "GPU-test"})()
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda _: properties)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda _: (10_000, 20_000))
    status_result = type("Result", (), {"stdout": "GPU-test, 40000, 100, 0"})()
    process_result = type("Result", (), {"stdout": ""})()
    monkeypatch.setattr(
        "src.layer_probe_representations.subprocess.run",
        lambda command, **kwargs: (
            status_result if any("--query-gpu" in value for value in command) else process_result
        ),
    )
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
    audit = gpu_runtime_audit("cuda:0")
    assert audit["physical_index"] == 1
    assert audit["process_local_index"] == 0

    try:
        gpu_runtime_audit("cuda:1")
    except RuntimeError as exc:
        assert "本地编号" in str(exc)
    else:
        raise AssertionError("CUDA_VISIBLE_DEVICES=1时cuda:1必须被拒绝")

    monkeypatch.delenv("CUDA_VISIBLE_DEVICES")
    try:
        gpu_runtime_audit("cuda:0")
    except RuntimeError as exc:
        assert "物理GPU 1" in str(exc)
    else:
        raise AssertionError("GPU 0不得被自动使用")


def test_strict_preflight_detects_protocol_change(tmp_path: Path):
    run = tmp_path / "runs" / "strict"
    config = {
        "output": {"run_directory": str(run)},
        "continuous_probe": {"alpha_grid": [0.1, 1.0]},
        "strict_test": {
            "open_final_test": True,
            "selected_factors": ["layer_consensus"],
        },
        "sentiment_appendix": {"enabled": False},
    }
    manifest_paths = [
        run / "continuous_targets" / "validation" / "manifest.json",
        run / "fixed_head_label" / "validation" / "manifest.json",
        run / "continuous_probe" / "validation" / "manifest.json",
        run / "stock_day_panel" / "validation" / "stock_day_manifest.json",
        run / "return_probe" / "validation" / "manifest.json",
        run / "factor_validation" / "validation" / "manifest.json",
    ]
    for path in manifest_paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps({"config_sha256": protocol_config_hash(config)}),
            encoding="utf-8",
        )
    report = preflight_strict_test(config)
    assert not report[report["blocking"] & ~report["status"].eq("ok")].any().any()

    changed = {**config, "continuous_probe": {"alpha_grid": [100.0]}}
    changed_report = preflight_strict_test(changed)
    mismatch = changed_report[
        changed_report["check"].str.endswith("_protocol_hash")
        & changed_report["status"].eq("invalid")
    ]
    assert len(mismatch) == len(manifest_paths)
