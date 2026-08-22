"""Stage-aware preflight, final-test orchestration, validation and plots."""

from __future__ import annotations

import json
import hashlib
import os
import shutil
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from .config import load_yaml_config
from .layer_probe_continuous import (
    run_continuous_probe_stage,
    run_fixed_head_label_stage,
    validate_target_provenance,
    validate_target_semantics,
    validate_aligned_targets,
    validate_continuous_probe_outputs,
    validate_fixed_head_analysis_outputs,
    validate_fixed_head_label_outputs,
)
from .layer_probe_factors import factor_columns, run_factor_validation_stage
from .layer_probe_models import parse_time_windows, run_return_probe_stage
from .layer_probe_panel import run_stock_day_panel_stage, validate_stock_day_artifacts
from .layer_probe_representations import (
    HEAD_RECIPE,
    gpu_runtime_audit,
    protocol_config_hash,
    resolve_representation_directory,
    sha256_file,
    validate_representation_artifacts,
)
from .report_label_runtime import resolve_runtime_resources


def load_layer_probe_config(path: str | Path) -> dict[str, object]:
    config = load_yaml_config(path)
    if not isinstance(config, dict):
        raise TypeError("Layer Probe配置顶层必须是对象")
    return config


def _record(
    stage: str,
    check: str,
    *,
    ok: bool,
    detail: object,
    blocking: bool = True,
    warning: bool = False,
) -> dict[str, object]:
    return {
        "stage": stage,
        "check": check,
        "status": "warning" if warning else ("ok" if ok else "invalid"),
        "blocking": bool(blocking),
        "detail": str(detail),
    }


def _path_record(
    stage: str, check: str, value: object, *, directory: bool = False, blocking: bool = True
) -> dict[str, object]:
    path = Path(str(value or "")).expanduser().resolve() if value else None
    exists = bool(path and (path.is_dir() if directory else path.is_file()))
    return _record(
        stage,
        check,
        ok=exists,
        blocking=blocking,
        detail=path if path else "未配置",
    )


def _nearest_existing(path: Path) -> Path:
    candidate = path
    while not candidate.exists() and candidate != candidate.parent:
        candidate = candidate.parent
    return candidate


def _path_size(path: Path) -> int:
    if not path.exists():
        return 0
    if path.is_file():
        return int(path.stat().st_size)
    total = 0
    for root, _, files in os.walk(path):
        for filename in files:
            try:
                total += (Path(root) / filename).stat().st_size
            except OSError:
                continue
    return int(total)


def _estimated_text_rows(path: Path, limit: object) -> int:
    if path.suffix.lower() in {".parquet", ".pq"}:
        import pyarrow.parquet as pq

        rows = int(pq.ParquetFile(path).metadata.num_rows)
    else:
        with path.open("rb") as handle:
            rows = max(0, sum(1 for _ in handle) - 1)
    return min(rows, int(limit)) if limit else rows


def preflight_representation(config: Mapping[str, object]) -> pd.DataFrame:
    """Representation checks only text/model/device/disk; never labels or splits."""

    text = config.get("text", {})
    model = config.get("model", {})
    output = config.get("output", {})
    if not all(isinstance(value, Mapping) for value in (text, model, output)):
        raise ValueError("text/model/output配置必须是对象")
    records = [
        _path_record("representation", "text.path", text.get("path")),
        _path_record("representation", "model.checkpoint", model.get("checkpoint")),
        _path_record(
            "representation", "model.base_model_dir", model.get("base_model_dir"), directory=True
        ),
        _record(
            "representation",
            "model.head_recipe",
            ok=str(model.get("head_recipe", "")) == HEAD_RECIPE,
            detail=(
                f"{model.get('head_recipe')} (provenance={model.get('head_provenance')}; "
                "historical path not confirmed)"
            ),
        ),
    ]
    run_directory = Path(str(output.get("run_directory", ""))).expanduser().resolve()
    store = Path(str(output.get("representation_store", ""))).expanduser().resolve()
    old_root = Path("artifacts/layer_probe").resolve()
    isolated = old_root not in run_directory.parents and old_root not in store.parents
    records.append(
        _record(
            "representation",
            "legacy_artifact_isolation",
            ok=isolated,
            detail=f"run={run_directory}; store={store}; legacy={old_root}",
        )
    )
    try:
        audit = gpu_runtime_audit(str(model.get("device", "cuda:1")))
        records.append(
            _record(
                "representation", "gpu_mapping", ok=True, detail=json.dumps(audit, default=str)
            )
        )
    except (OSError, RuntimeError, ValueError) as exc:
        records.append(_record("representation", "gpu_mapping", ok=False, detail=exc))
    performance = config.get("performance", {})
    if not isinstance(performance, Mapping):
        records.append(
            _record(
                "representation",
                "cpu_thread_budget",
                ok=False,
                detail="performance配置必须是对象",
            )
        )
    else:
        try:
            resources = resolve_runtime_resources(performance)
            records.append(
                _record(
                    "representation",
                    "cpu_thread_budget",
                    ok=resources.effective_threads <= 32,
                    detail=json.dumps(resources.to_dict(), default=str),
                )
            )
        except (TypeError, ValueError) as exc:
            records.append(
                _record(
                    "representation", "cpu_thread_budget", ok=False, detail=exc
                )
            )
    filesystem = _nearest_existing(store)
    usage = shutil.disk_usage(filesystem)
    reserve = max(15 * (1 << 30), int(usage.free * 0.20))
    text_path = Path(str(text.get("path", ""))).expanduser().resolve()
    base_model = Path(str(model.get("base_model_dir", ""))).expanduser().resolve()
    projected_peak = None
    estimate_error = None
    try:
        rows = _estimated_text_rows(text_path, text.get("limit"))
        model_config = json.loads((base_model / "config.json").read_text(encoding="utf-8"))
        hidden = int(model_config["hidden_size"])
        dtype_bytes = np.dtype(str(model.get("storage_dtype", "float16"))).itemsize
        representation_bytes = rows * 13 * hidden * dtype_bytes
        fixed_head_bytes = rows * 13 * 48
        metadata_bytes = max(rows * 256, 1 << 20)
        projected_peak = int(
            (representation_bytes + fixed_head_bytes + metadata_bytes) * 1.25
        )
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        estimate_error = exc
    records.append(
        _record(
            "representation",
            "disk_peak_estimate",
            ok=projected_peak is not None and projected_peak < usage.free - reserve,
            detail=(
                f"filesystem={filesystem}, projected_peak={projected_peak}, "
                f"free={usage.free}, reserve={reserve}"
                if estimate_error is None
                else f"无法估算: {estimate_error}"
            ),
        )
    )
    records.append(
        _record(
            "representation",
            "disk_inventory",
            ok=True,
            blocking=False,
            detail=json.dumps(
                {
                    "text_input_bytes": _path_size(text_path),
                    "model_cache_bytes": _path_size(base_model),
                    "legacy_artifacts_bytes": _path_size(old_root),
                    "new_store_bytes": _path_size(store),
                    "temporary_directory": str(Path(tempfile.gettempdir()).resolve()),
                    "target_filesystem": str(filesystem),
                }
            ),
        )
    )
    return pd.DataFrame(records)


def preflight_fixed_head(config: Mapping[str, object]) -> pd.DataFrame:
    try:
        directory = resolve_representation_directory(config)
        check = validate_representation_artifacts(directory)
        ok = check["head_recipe"] == HEAD_RECIPE
        return pd.DataFrame(
            [
                _record("fixed_head", "representation_and_layer12", ok=ok, detail=check),
                _record(
                    "fixed_head",
                    "historical_head_provenance",
                    ok=True,
                    blocking=False,
                    warning=True,
                    detail="CLS->fc is a user-locked assumption, not historical proof",
                ),
            ]
        )
    except (OSError, RuntimeError, TypeError, ValueError, KeyError, json.JSONDecodeError) as exc:
        return pd.DataFrame(
            [_record("fixed_head", "representation_and_layer12", ok=False, detail=exc)]
        )


def preflight_continuous_labels(config: Mapping[str, object]) -> pd.DataFrame:
    target = config.get("continuous_targets", {})
    if not isinstance(target, Mapping):
        raise ValueError("continuous_targets配置必须是对象")
    directory = Path(str(target.get("bundle_directory", ""))).expanduser().resolve()
    records = [
        _path_record("continuous_labels", "probe_targets", directory / "probe_targets.parquet"),
        _path_record(
            "continuous_labels",
            "probe_dataset_metadata",
            directory / "probe_dataset_metadata.json",
        ),
        _path_record("continuous_labels", "probe_merge_audit", directory / "probe_merge_audit.csv"),
    ]
    targets_path = directory / "probe_targets.parquet"
    metadata_path = directory / "probe_dataset_metadata.json"
    if targets_path.is_file() and metadata_path.is_file():
        try:
            targets = pd.read_parquet(
                targets_path, filters=[("split", "in", ["train", "validation"])]
            )
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            validate_target_provenance(metadata)
            validate_target_semantics(targets, metadata)
            unique = not targets.duplicated(["report_id", "task_id"]).any()
            split_ok = set(targets["split"].astype(str)).issubset({"train", "validation", "test"})
            pit = pd.to_datetime(targets["label_available_date"]) > pd.to_datetime(
                targets["feature_available_date"]
            )
            windows = metadata.get("splits", [])
            records.extend(
                [
                    _record("continuous_labels", "report_task_unique", ok=unique, detail=unique),
                    _record("continuous_labels", "chronological_splits", ok=split_ok and len(windows) == 3, detail=windows),
                    _record("continuous_labels", "label_available_date", ok=bool(pit.all()), detail=f"valid={int(pit.sum())}/{len(pit)}"),
                    _record(
                        "continuous_labels",
                        "upstream_label_provenance",
                        ok=True,
                        detail="report_fy_labels.parquet + report_confirmation_labels.parquet hashes verified",
                    ),
                ]
            )
        except (OSError, RuntimeError, ValueError, KeyError, json.JSONDecodeError) as exc:
            records.append(_record("continuous_labels", "target_schema", ok=False, detail=exc))
    return pd.DataFrame(records)


def preflight_returns(config: Mapping[str, object]) -> pd.DataFrame:
    returns = config.get("returns", {})
    splits = config.get("return_time_splits", {})
    exposures = config.get("exposures", {})
    if not all(isinstance(value, Mapping) for value in (returns, splits, exposures)):
        raise ValueError("returns/return_time_splits/exposures配置必须是对象")
    records = [
        _path_record(
            "returns",
            "industry_adjusted_daily_path",
            returns.get("industry_adjusted_daily_path"),
        )
    ]
    try:
        windows = parse_time_windows(splits)
        records.append(
            _record(
                "returns",
                "return_time_splits",
                ok=True,
                detail="; ".join(
                    f"{window.name}:{window.start.date()}~{window.end.date()}" for window in windows
                ),
            )
        )
    except (TypeError, ValueError) as exc:
        records.append(_record("returns", "return_time_splits", ok=False, detail=exc))
    for name in ("industry_path", "size_path"):
        if exposures.get(name):
            records.append(
                _path_record("returns", f"exposures.{name}", exposures.get(name), blocking=False)
            )
    return pd.DataFrame(records)


def preflight_optional_sentiment(config: Mapping[str, object]) -> pd.DataFrame:
    appendix = config.get("sentiment_appendix", {})
    if not isinstance(appendix, Mapping):
        raise ValueError("sentiment_appendix配置必须是对象")
    if not bool(appendix.get("enabled", False)):
        return pd.DataFrame(
            [_record("sentiment_appendix", "enabled", ok=True, blocking=False, detail="disabled")]
        )
    return pd.DataFrame(
        [
            _path_record("sentiment_appendix", "labels_path", appendix.get("labels_path")),
            _record(
                "sentiment_appendix",
                "label_mapping_confirmed",
                ok=bool(appendix.get("label_mapping_confirmed", False)),
                detail=appendix.get("label_mapping_confirmed", False),
            ),
        ]
    )


def preflight_strict_test(config: Mapping[str, object]) -> pd.DataFrame:
    strict = config.get("strict_test", {})
    if not isinstance(strict, Mapping):
        raise ValueError("strict_test配置必须是对象")
    run = _run_path(config)
    selected = [str(value) for value in strict.get("selected_factors", [])]
    records = [
        _record(
            "strict_test",
            "open_final_test",
            ok=bool(strict.get("open_final_test", False)),
            detail=strict.get("open_final_test", False),
        ),
        _record("strict_test", "selected_factors", ok=bool(selected), detail=selected),
    ]
    validation_paths = (
        ("continuous_targets_validation", run / "continuous_targets" / "validation" / "manifest.json"),
        ("fixed_head_label_validation", run / "fixed_head_label" / "validation" / "manifest.json"),
        ("continuous_probe_validation", run / "continuous_probe" / "validation" / "manifest.json"),
        ("stock_day_validation", run / "stock_day_panel" / "validation" / "stock_day_manifest.json"),
        ("return_probe_validation", run / "return_probe" / "validation" / "manifest.json"),
        ("factor_validation", run / "factor_validation" / "validation" / "manifest.json"),
    )
    expected_protocol_hash = protocol_config_hash(config)
    for name, path in validation_paths:
        records.append(_path_record("strict_test", name, path))
        if path.is_file():
            try:
                manifest = json.loads(path.read_text(encoding="utf-8"))
                actual_hash = manifest.get("config_sha256")
                records.append(
                    _record(
                        "strict_test",
                        f"{name}_protocol_hash",
                        ok=actual_hash == expected_protocol_hash,
                        detail=f"actual={actual_hash}; expected={expected_protocol_hash}",
                    )
                )
            except (OSError, json.JSONDecodeError) as exc:
                records.append(
                    _record(
                        "strict_test",
                        f"{name}_protocol_hash",
                        ok=False,
                        detail=exc,
                    )
                )
    marker = run / "FINAL_TEST_OPENED.json"
    records.append(
        _record(
            "strict_test",
            "not_opened_before",
            ok=not marker.exists(),
            detail=marker,
        )
    )
    return pd.DataFrame(records)


def preflight_fixed_head_analysis(config: Mapping[str, object]) -> pd.DataFrame:
    return preflight_fixed_head(config)


def preflight_target_alignment(config: Mapping[str, object]) -> pd.DataFrame:
    return pd.concat(
        [preflight_fixed_head(config), preflight_continuous_labels(config)],
        ignore_index=True,
    )


def preflight_fixed_head_label(config: Mapping[str, object]) -> pd.DataFrame:
    return preflight_target_alignment(config)


def preflight_continuous_probe(config: Mapping[str, object]) -> pd.DataFrame:
    return preflight_target_alignment(config)


def preflight_stock_day_panel(config: Mapping[str, object]) -> pd.DataFrame:
    return pd.concat(
        [preflight_fixed_head(config), preflight_returns(config)], ignore_index=True
    )


def preflight_return_probe(config: Mapping[str, object]) -> pd.DataFrame:
    report = preflight_returns(config)
    stock_manifest = (
        _run_path(config)
        / "stock_day_panel"
        / "validation"
        / "stock_day_manifest.json"
    )
    return pd.concat(
        [
            report,
            pd.DataFrame(
                [
                    _path_record(
                        "return_probe", "stock_day_validation", stock_manifest
                    )
                ]
            ),
        ],
        ignore_index=True,
    )


def preflight_factor_validation(config: Mapping[str, object]) -> pd.DataFrame:
    return_directory = _run_path(config) / "return_probe" / "validation"
    return pd.DataFrame(
        [
            _path_record(
                "factor_validation",
                "return_oos_predictions",
                return_directory / "return_oos_predictions.parquet",
            ),
            _path_record(
                "factor_validation",
                "return_fit_reference_predictions",
                return_directory / "return_fit_reference_predictions.parquet",
            ),
            _path_record(
                "factor_validation",
                "return_manifest",
                return_directory / "manifest.json",
            ),
        ]
    )


def _run_path(config: Mapping[str, object]) -> Path:
    output = config.get("output", {})
    if not isinstance(output, Mapping):
        raise ValueError("output配置必须是对象")
    return Path(str(output.get("run_directory", ""))).expanduser().resolve()


PREFLIGHTS = {
    "representation": preflight_representation,
    "fixed_head": preflight_fixed_head,
    "continuous_labels": preflight_continuous_labels,
    "returns": preflight_returns,
    "sentiment_appendix": preflight_optional_sentiment,
    "strict_test": preflight_strict_test,
}


def preflight_report(
    config: Mapping[str, object], stages: Sequence[str] | None = None
) -> pd.DataFrame:
    selected = list(stages or PREFLIGHTS.keys())
    unknown = sorted(set(selected).difference(PREFLIGHTS))
    if unknown:
        raise ValueError("未知预检阶段: " + ", ".join(unknown))
    return pd.concat([PREFLIGHTS[name](config) for name in selected], ignore_index=True)


def assert_preflight(report: pd.DataFrame) -> None:
    invalid = report[report["blocking"].eq(True) & ~report["status"].eq("ok")]
    if not invalid.empty:
        raise RuntimeError(
            "预检未通过:\n" + invalid[["stage", "check", "detail"]].to_string(index=False)
        )


def validate_return_probe_outputs(directory: str | Path) -> dict[str, object]:
    root = Path(directory).expanduser().resolve()
    required = [
        root / "return_probe_metrics.csv",
        root / "return_oos_predictions.parquet",
        root / "return_fit_reference_predictions.parquet",
        root / "return_probe_tuning.csv",
        root / "manifest.json",
    ]
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(f"收益Probe产物缺失: {path}")
    metrics = pd.read_csv(required[0])
    oos = pd.read_parquet(required[1])
    reference = pd.read_parquet(required[2])
    for split in metrics["split"].unique():
        if set(metrics.loc[metrics["split"].eq(split), "layer"]) != set(range(13)):
            raise ValueError(f"收益{split}没有完整13层指标")
    if not oos["prediction_role"].eq("oos").all():
        raise ValueError("收益OOS文件混入fit-reference")
    if not reference["prediction_role"].eq("fit_reference").all():
        raise ValueError("收益fit-reference文件混入OOS")
    return {"metric_rows": len(metrics), "oos_rows": len(oos), "reference_rows": len(reference)}


def validate_factor_outputs(directory: str | Path) -> dict[str, object]:
    root = Path(directory).expanduser().resolve()
    required = [
        root / "candidate_factor_matrix.parquet",
        root / "daily_rank_ic.csv",
        root / "factor_summary.csv",
        root / "quantile_monotonicity.csv",
        root / "stratified_ic.csv",
        root / "incremental_ic.csv",
        root / "manifest.json",
    ]
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(f"跨层因子产物缺失: {path}")
    matrix = pd.read_parquet(required[0])
    summary = pd.read_csv(required[2])
    factors = factor_columns(matrix)
    if not factors or summary["factor"].duplicated().any():
        raise ValueError("跨层因子矩阵或汇总无效")
    return {"rows": len(matrix), "candidate_factors": len(factors), "evaluated": len(summary)}


def validate_pipeline_outputs(config: Mapping[str, object]) -> pd.DataFrame:
    run = _run_path(config)
    validators: list[tuple[str, Path, object]] = []
    try:
        rep = resolve_representation_directory(config)
        validators.append(("representations", rep, validate_representation_artifacts))
    except (OSError, RuntimeError, ValueError, KeyError):
        pass
    validators.extend(
        [
            ("fixed_head_analysis", run / "fixed_head_analysis", validate_fixed_head_analysis_outputs),
            ("continuous_targets", run / "continuous_targets" / "validation", validate_aligned_targets),
            ("fixed_head_label_validation", run / "fixed_head_label" / "validation", validate_fixed_head_label_outputs),
            ("continuous_probe_validation", run / "continuous_probe" / "validation", validate_continuous_probe_outputs),
            ("stock_day_panel", run / "stock_day_panel", validate_stock_day_artifacts),
            ("return_probe_validation", run / "return_probe" / "validation", validate_return_probe_outputs),
            ("factor_validation", run / "factor_validation" / "validation", validate_factor_outputs),
        ]
    )
    records: list[dict[str, object]] = []
    for name, path, validator in validators:
        if not path.exists():
            records.append({"stage": name, "status": "not_run", "detail": str(path)})
            continue
        try:
            result = validator(path)  # type: ignore[operator]
            records.append({"stage": name, "status": "ok", "detail": json.dumps(result, default=str)})
        except (OSError, RuntimeError, TypeError, ValueError, KeyError, json.JSONDecodeError) as exc:
            records.append({"stage": name, "status": "failed", "detail": str(exc)})
    return pd.DataFrame(records)


def run_final_test_once(config: Mapping[str, object]) -> dict[str, str]:
    """Open all confirmatory test labels once and never retune afterward."""

    report = preflight_strict_test(config)
    assert_preflight(report)
    run = _run_path(config)
    run.mkdir(parents=True, exist_ok=True)
    marker = run / "FINAL_TEST_OPENED.json"
    validation_manifests = {
        name: {"path": str(path), "sha256": sha256_file(path)}
        for name, path in (
            ("continuous_targets", run / "continuous_targets" / "validation" / "manifest.json"),
            ("fixed_head_label", run / "fixed_head_label" / "validation" / "manifest.json"),
            ("continuous_probe", run / "continuous_probe" / "validation" / "manifest.json"),
            ("stock_day", run / "stock_day_panel" / "validation" / "stock_day_manifest.json"),
            ("return_probe", run / "return_probe" / "validation" / "manifest.json"),
            ("factor_validation", run / "factor_validation" / "validation" / "manifest.json"),
        )
    }
    representation_manifest_path = (
        resolve_representation_directory(config) / "representation_manifest.json"
    )
    continuous_metrics = pd.read_csv(
        run / "continuous_probe" / "validation" / "continuous_probe_metrics.csv"
    )
    continuous_metrics = continuous_metrics[
        continuous_metrics["prediction_role"].eq("oos")
    ]
    continuous_alphas = {
        f"{row.task_id}::layer_{int(row.layer)}": float(row.selected_alpha)
        for row in continuous_metrics.itertuples()
    }
    return_metrics = pd.read_csv(
        run / "return_probe" / "validation" / "return_probe_metrics.csv"
    )
    return_metrics = return_metrics[return_metrics["split"].eq("validation")]
    return_alphas = {
        f"layer_{int(row.layer)}": float(row.selected_alpha)
        for row in return_metrics.itertuples()
    }
    marker_payload = {
        "schema_version": "final_test_gate_v2.0",
        "opened_at": datetime.now().isoformat(timespec="seconds"),
        "pid": os.getpid(),
        "config_sha256": hashlib.sha256(
            json.dumps(config, sort_keys=True, default=str).encode("utf-8")
        ).hexdigest(),
        "selected_factors": list((config.get("strict_test", {}) or {}).get("selected_factors", [])),
        "selected_alphas": {
            "continuous_probe": continuous_alphas,
            "return_probe": return_alphas,
        },
        "representation_manifest": {
            "path": str(representation_manifest_path),
            "sha256": sha256_file(representation_manifest_path),
        },
        "validation_manifests": validation_manifests,
    }
    with marker.open("x", encoding="utf-8") as handle:
        json.dump(marker_payload, handle, ensure_ascii=False, indent=2)
    marker.chmod(0o444)

    def verify_frozen_inputs() -> None:
        frozen = [marker_payload["representation_manifest"]]
        frozen.extend(marker_payload["validation_manifests"].values())
        for item in frozen:
            path = Path(str(item["path"]))
            if not path.is_file() or sha256_file(path) != item["sha256"]:
                raise RuntimeError(f"最终test开启后冻结输入发生变化: {path}")

    try:
        verify_frozen_inputs()
        outputs = {
            "fixed_head_label": str(run_fixed_head_label_stage(config, "test"))
        }
        verify_frozen_inputs()
        outputs["continuous_probe"] = str(run_continuous_probe_stage(config, "test"))
        verify_frozen_inputs()
        outputs["stock_day_panel"] = str(run_stock_day_panel_stage(config, "test"))
        verify_frozen_inputs()
        outputs["return_probe"] = str(run_return_probe_stage(config, "test"))
        verify_frozen_inputs()
        outputs["factor_validation"] = str(
            run_factor_validation_stage(config, "test")
        )
        with (run / "FINAL_TEST_COMPLETED.json").open("x", encoding="utf-8") as handle:
            json.dump(
                {"completed_at": datetime.now().isoformat(timespec="seconds"), "outputs": outputs},
                handle,
                ensure_ascii=False,
                indent=2,
            )
        return outputs
    except BaseException as exc:
        with (run / "FINAL_TEST_FAILED.json").open("x", encoding="utf-8") as handle:
            json.dump(
                {"failed_at": datetime.now().isoformat(timespec="seconds"), "error": repr(exc)},
                handle,
                ensure_ascii=False,
                indent=2,
            )
        raise


def plot_return_curve(directory: str | Path, *, split: str = "validation"):
    import matplotlib.pyplot as plt

    metrics = pd.read_csv(Path(directory) / "return_probe_metrics.csv")
    selected = metrics[metrics["split"].eq(split)].sort_values("layer")
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(selected["layer"], selected["rank_ic"], marker="o")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set(title=f"Return decodability ({split})", xlabel="Layer", ylabel="Mean Rank IC")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    return fig


def plot_factor_summary(directory: str | Path):
    import matplotlib.pyplot as plt

    summary = pd.read_csv(Path(directory) / "factor_summary.csv").sort_values("rank_ic")
    fig, ax = plt.subplots(figsize=(9, max(4, 0.3 * len(summary))))
    colors = np.where(summary["qvalue_bh"] <= 0.05, "tab:blue", "tab:gray")
    ax.barh(summary["factor"], summary["rank_ic"], color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set(title="Cross-layer factor Rank IC", xlabel="Mean daily Rank IC")
    fig.tight_layout()
    return fig
