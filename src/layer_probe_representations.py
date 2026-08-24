"""Canonical Layer 0~12 representations and frozen CLS->fc projections.

The content-addressed artifact is label independent. Every report is stored once
in a mmap-friendly ``[report, layer, hidden]`` NPY array. Small per-layer
frozen-head outputs are stored separately and keyed by ``representation_row``.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .inference_cache import tokenizer_signature
from .models.modeling import build_candidate, checkpoint_hash
from .report_label_economics import canonical_stock_code
from .report_label_runtime import resolve_runtime_resources


REPRESENTATION_FILE = "representations.npy"
METADATA_FILE = "report_metadata.parquet"
HEAD_OUTPUT_FILE = "fixed_head_layer_outputs.parquet"
MANIFEST_FILE = "representation_manifest.json"
POINTER_FILE = "representation_pointer.json"
SCHEMA_VERSION = "continuous_label_layer_representation_v2.0"
HEAD_RECIPE = "cls_fc"
HEAD_PROVENANCE = "user_locked_assumption"
_LAYER_PROBE_THREADPOOL_GUARD: object | None = None


@dataclass(frozen=True)
class RepresentationArtifacts:
    directory: Path
    representations: Path
    metadata: Path
    fixed_head_outputs: Path
    manifest: Path


def representation_artifacts(directory: str | Path) -> RepresentationArtifacts:
    root = Path(directory).expanduser().resolve()
    return RepresentationArtifacts(
        directory=root,
        representations=root / REPRESENTATION_FILE,
        metadata=root / METADATA_FILE,
        fixed_head_outputs=root / HEAD_OUTPUT_FILE,
        manifest=root / MANIFEST_FILE,
    )


def _require_columns(frame: pd.DataFrame, columns: Sequence[str], name: str) -> None:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise ValueError(f"{name}缺少字段: {', '.join(missing)}")


def _read_table(path: str | Path) -> pd.DataFrame:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"研报文本数据不存在: {source}")
    suffixes = "".join(source.suffixes).lower()
    if source.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(source)
    if suffixes.endswith(".csv.gz") or source.suffix.lower() == ".csv":
        return pd.read_csv(source)
    raise ValueError(f"仅支持CSV/CSV.GZ/Parquet研报表: {source}")


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_file(path: str | Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def protocol_config_hash(config: Mapping[str, object]) -> str:
    """Hash scientific settings while excluding the one-time gate toggles."""

    normalized = json.loads(json.dumps(config, sort_keys=True, default=str))
    strict = normalized.get("strict_test")
    if isinstance(strict, dict):
        strict.pop("open_final_test", None)
        strict.pop("selected_factors", None)
    normalized.pop("sentiment_appendix", None)
    return hashlib.sha256(
        json.dumps(normalized, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
        return result.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def configure_layer_probe_runtime(
    performance: Mapping[str, object], *, device: str
) -> dict[str, object]:
    """Apply one CPU/thread budget and record scheduler/cgroup/NUMA topology."""

    global _LAYER_PROBE_THREADPOOL_GUARD
    resources = resolve_runtime_resources(performance)
    threads = str(resources.effective_threads)
    blas_threads = str(resources.blas_threads)
    for name in ("TOKENIZERS_PARALLELISM",):
        os.environ[name] = "false"
    for name in ("OMP_NUM_THREADS", "POLARS_MAX_THREADS"):
        os.environ[name] = threads
    for name in (
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "BLIS_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[name] = blas_threads
    from threadpoolctl import threadpool_limits

    _LAYER_PROBE_THREADPOOL_GUARD = threadpool_limits(
        limits=resources.blas_threads, user_api="blas"
    )
    try:
        import torch

        torch.set_num_threads(resources.effective_threads)
        try:
            torch.set_num_interop_threads(
                max(1, min(4, resources.effective_threads))
            )
        except RuntimeError:
            pass
    except ImportError:
        pass
    audit: dict[str, object] = resources.to_dict()
    audit["process_affinity"] = (
        sorted(os.sched_getaffinity(0))
        if hasattr(os, "sched_getaffinity")
        else None
    )
    for name, command in (
        ("lscpu", ["lscpu", "--json"]),
        ("gpu_numa_topology", ["nvidia-smi", "topo", "-m"]),
    ):
        if name == "gpu_numa_topology" and not device.startswith("cuda"):
            audit[name] = "not_applicable"
            continue
        try:
            result = subprocess.run(
                command,
                check=True,
                capture_output=True,
                text=True,
                timeout=10,
            )
            audit[name] = result.stdout.strip()
        except (OSError, subprocess.SubprocessError):
            audit[name] = "unavailable"
    return audit


def fingerprint_frame(frame: pd.DataFrame, columns: Sequence[str]) -> str:
    values = pd.util.hash_pandas_object(frame[list(columns)], index=False).to_numpy()
    return hashlib.sha256(values.tobytes()).hexdigest()


def load_text_dataset(
    path: str | Path,
    *,
    id_column: str,
    text_column: str,
    symbol_column: str,
    date_column: str,
    limit: int | None = None,
) -> pd.DataFrame:
    """Load the label-independent canonical report table."""

    frame = _read_table(path)
    _require_columns(
        frame, [id_column, text_column, symbol_column, date_column], "研报文本输入"
    )
    out = frame.rename(
        columns={
            id_column: "report_id",
            text_column: "text",
            symbol_column: "symbol",
            date_column: "feature_available_date",
        }
    ).copy()
    out.insert(0, "source_row", np.arange(len(out), dtype=np.int64))
    out["report_id"] = out["report_id"].astype("string").str.strip()
    if out["report_id"].isna().any() or out["report_id"].eq("").any():
        raise ValueError("研报文本输入含空report_id")
    if out["report_id"].duplicated().any():
        raise ValueError("研报文本输入的report_id不唯一")
    out["text"] = out["text"].fillna("").astype(str).str.strip()
    if out["text"].eq("").any():
        raise ValueError(f"研报文本输入含{int(out['text'].eq('').sum())}篇空文本")
    out["symbol"] = canonical_stock_code(out["symbol"])
    if out["symbol"].isna().any():
        raise ValueError("研报文本输入含无效股票代码")
    out["feature_available_date"] = pd.to_datetime(
        out["feature_available_date"], errors="coerce"
    ).dt.normalize()
    if out["feature_available_date"].isna().any():
        raise ValueError("研报文本输入含无效feature_available_date")
    if limit is not None:
        if int(limit) <= 0:
            raise ValueError("limit必须为正整数")
        out = out.iloc[: int(limit)].copy()
    if out.empty:
        raise ValueError("Representation输入为空")
    out = out.reset_index(drop=True)
    out.insert(0, "representation_row", np.arange(len(out), dtype=np.int64))
    out["text_sha256"] = out["text"].map(_sha256_text)
    return out


def freeze_and_validate_inference_model(
    model: Any, *, expected_hidden_layers: int | None = None
) -> dict[str, object]:
    """Freeze backbone, pooler and head and verify strict inference state."""

    import torch

    model.eval()
    model.requires_grad_(False)
    parameters = list(model.parameters())
    trainable = sum(p.numel() for p in parameters if p.requires_grad)
    dropout_training = [
        name
        for name, module in model.named_modules()
        if isinstance(module, torch.nn.Dropout) and module.training
    ]
    if model.training or trainable or dropout_training:
        raise RuntimeError(
            "模型未处于严格推理状态: "
            f"model.training={model.training}, trainable={trainable}, "
            f"dropout_training={dropout_training[:5]}"
        )
    configured = getattr(getattr(model, "bert", None), "config", None)
    configured = getattr(configured, "num_hidden_layers", None)
    if expected_hidden_layers is not None and configured is not None:
        if int(configured) + 1 != int(expected_hidden_layers):
            raise ValueError(
                f"模型应输出{expected_hidden_layers}层（含embedding），"
                f"实际为{int(configured) + 1}层"
            )
    pooler = getattr(getattr(model, "bert", None), "pooler", None)
    head = getattr(model, "fc", None)
    return {
        "model_training": bool(model.training),
        "parameter_count": int(sum(p.numel() for p in parameters)),
        "trainable_parameter_count": int(trainable),
        "dropout_modules_in_training_mode": dropout_training,
        "backbone_frozen": not any(
            p.requires_grad for p in getattr(model, "bert", model).parameters()
        ),
        "pooler_frozen": (
            not any(p.requires_grad for p in pooler.parameters())
            if pooler is not None
            else True
        ),
        "head_frozen": (
            not any(p.requires_grad for p in head.parameters())
            if head is not None
            else True
        ),
        "forward_context": "torch.inference_mode",
    }


def stack_layer_cls(hidden_states: Sequence[Any], *, expected_layers: int) -> Any:
    """Stack CLS vectors from one backbone forward as ``[B,L,H]``."""

    import torch

    if hidden_states is None or len(hidden_states) != expected_layers:
        actual = None if hidden_states is None else len(hidden_states)
        raise ValueError(f"hidden_states层数错误: {actual} != {expected_layers}")
    if any(state.ndim != 3 for state in hidden_states):
        raise ValueError("每层hidden state必须是[batch,tokens,hidden]")
    shapes = {tuple(state.shape) for state in hidden_states}
    if len(shapes) != 1:
        raise ValueError(f"各层hidden state形状不一致: {sorted(shapes)}")
    return torch.stack([state[:, 0, :] for state in hidden_states], dim=1)


def apply_cls_fc_to_layers(
    model: Any,
    hidden_states: Sequence[Any],
    *,
    attention_mask: Any | None = None,
) -> Any:
    """Apply the same frozen fc to all 13 raw CLS vectors."""

    import torch

    if getattr(model, "fc", None) is None:
        raise ValueError("固定CLS-fc投影要求模型具有fc头")
    logits = []
    for state in hidden_states:
        if hasattr(model, "apply_frozen_head"):
            _, layer_logits = model.apply_frozen_head(
                state, attention_mask=attention_mask
            )
        else:
            layer_logits = model.fc(state[:, 0, :])
        logits.append(layer_logits)
    return torch.stack(logits, dim=1)


def _safe_output(path: Path) -> None:
    if path == Path(path.anchor) or len(path.parts) < 3:
        raise ValueError(f"拒绝写入过宽目录: {path}")


def _largest_children(directory: Path, limit: int = 8) -> list[tuple[str, int]]:
    if not directory.is_dir():
        return []
    records: list[tuple[str, int]] = []
    for child in directory.iterdir():
        try:
            if child.is_file():
                size = child.stat().st_size
            else:
                size = sum(
                    (Path(root) / filename).stat().st_size
                    for root, _, files in os.walk(child)
                    for filename in files
                )
            records.append((str(child), int(size)))
        except OSError:
            continue
    return sorted(records, key=lambda item: item[1], reverse=True)[:limit]


def disk_budget(
    output_parent: Path, *, rows: int, layers: int, hidden: int, dtype: str
) -> dict[str, int]:
    bytes_per = np.dtype(dtype).itemsize
    representation = int(rows * layers * hidden * bytes_per)
    fixed_head = int(rows * layers * 48)
    metadata = int(max(rows * 256, 1 << 20))
    projected_peak = int((representation + fixed_head + metadata) * 1.25)
    usage = shutil.disk_usage(output_parent)
    reserve = max(15 * (1 << 30), int(usage.free * 0.20))
    if projected_peak > usage.free - reserve:
        suggestions = _largest_children(output_parent)
        raise RuntimeError(
            "磁盘空间不足以安全生成Representation: "
            f"projected_peak={projected_peak}, free={usage.free}, reserve={reserve}; "
            "仅建议迁移/清理（程序不会自动删除），按大小排序="
            + json.dumps(suggestions, ensure_ascii=False)
        )
    return {
        "representation_bytes": representation,
        "fixed_head_estimated_bytes": fixed_head,
        "metadata_estimated_bytes": metadata,
        "projected_peak_bytes": projected_peak,
        "free_bytes_at_preflight": int(usage.free),
        "required_reserve_bytes": reserve,
    }


def gpu_runtime_audit(device: str) -> dict[str, object]:
    """Validate physical GPU 1 policy and record local/physical mapping."""

    import torch

    if not device.startswith("cuda"):
        return {"requested_device": device, "cuda": False}
    if not torch.cuda.is_available():
        raise RuntimeError("配置要求CUDA，但当前没有可用GPU")
    local_index = int(device.split(":", 1)[1]) if ":" in device else 0
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible:
        entries = [value.strip() for value in visible.split(",") if value.strip()]
        if local_index >= len(entries):
            raise RuntimeError(
                f"device={device}超出CUDA_VISIBLE_DEVICES={visible}的本地编号"
            )
        physical = entries[local_index]
    else:
        physical = str(local_index)
    if physical != "1":
        try:
            uuid_query = subprocess.run(
                [
                    "nvidia-smi",
                    "--id=1",
                    "--query-gpu=uuid",
                    "--format=csv,noheader,nounits",
                ],
                check=True,
                capture_output=True,
                text=True,
                timeout=10,
            )
            physical_one_uuid = uuid_query.stdout.strip()
        except (OSError, subprocess.SubprocessError):
            physical_one_uuid = ""
        if not physical_one_uuid or physical != physical_one_uuid:
            raise RuntimeError(
                "Layer Probe只允许自动使用物理GPU 1；"
                f"当前device={device}, CUDA_VISIBLE_DEVICES={visible!r}, "
                f"physical={physical}"
            )
    props = torch.cuda.get_device_properties(local_index)
    free, total = torch.cuda.mem_get_info(local_index)
    audit: dict[str, object] = {
        "requested_device": device,
        "cuda": True,
        "cuda_visible_devices": visible,
        "process_local_index": local_index,
        "physical_index": 1,
        "physical_selector": physical,
        "device_name": props.name,
        "device_uuid": getattr(props, "uuid", None),
        "free_memory_bytes": int(free),
        "total_memory_bytes": int(total),
    }
    try:
        status = subprocess.run(
            [
                "nvidia-smi",
                "--id=1",
                "--query-gpu=uuid,memory.total,memory.used,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
        audit["physical_gpu_status"] = status.stdout.strip()
        status_fields = [value.strip() for value in status.stdout.strip().split(",")]
        if len(status_fields) < 4:
            raise RuntimeError("无法解析物理GPU 1状态，按共享服务器策略拒绝启动")
        status_total = float(status_fields[1])
        status_used = float(status_fields[2])
        status_utilization = float(status_fields[3])
        if status_utilization > 10 or (
            status_total > 0 and status_used / status_total > 0.10
        ):
            raise RuntimeError(
                "物理GPU 1当前忙碌，按策略停止而不切换GPU 0: "
                f"used={status_used}/{status_total} MiB, utilization={status_utilization}%"
            )
        result = subprocess.run(
            [
                "nvidia-smi",
                "--id=1",
                "--query-compute-apps=gpu_uuid,pid,used_memory",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
        active = [
            line.strip() for line in result.stdout.splitlines() if line.strip()
        ]
        audit["active_compute_processes"] = active
        foreign = []
        for line in active:
            fields = [value.strip() for value in line.split(",")]
            if len(fields) >= 2 and fields[1].isdigit() and int(fields[1]) != os.getpid():
                foreign.append(line)
        if foreign:
            raise RuntimeError(
                "物理GPU 1已有其他计算进程，按共享服务器策略拒绝启动: "
                + "; ".join(foreign)
            )
    except (OSError, subprocess.SubprocessError) as exc:
        raise RuntimeError(
            "无法确认物理GPU 1的活动计算进程，按共享服务器策略拒绝启动"
        ) from exc
    return audit


def benchmark_batch_size(
    *,
    model: Any,
    tokenizer: Any,
    texts: pd.Series,
    candidates: Sequence[int],
    max_length: int,
    device: str,
) -> tuple[int, list[dict[str, object]]]:
    """Benchmark representative CUDA batches and select the largest safe size."""

    import time
    import torch

    values = sorted({int(value) for value in candidates if int(value) > 0})
    if not values:
        raise ValueError("batch_size_candidates必须含正整数")
    if not device.startswith("cuda"):
        return values[0], [{"batch_size": values[0], "status": "cpu_configured"}]
    results: list[dict[str, object]] = []
    successful: list[int] = []
    model.eval()
    for candidate in values:
        sample = texts.iloc[np.arange(candidate) % len(texts)].tolist()
        try:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
            encoded = tokenizer(
                sample,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
                return_attention_mask=True,
                return_token_type_ids=True,
            )
            inputs = {
                key: value.pin_memory().to(device, non_blocking=True)
                for key, value in encoded.items()
                if key in {"input_ids", "attention_mask", "token_type_ids"}
            }
            started = time.perf_counter()
            with torch.inference_mode():
                output = model(**inputs, output_hidden_states=True)
                _ = apply_cls_fc_to_layers(
                    model,
                    output.hidden_states,
                    attention_mask=inputs.get("attention_mask"),
                )
            torch.cuda.synchronize(device)
            seconds = time.perf_counter() - started
            peak = int(torch.cuda.max_memory_allocated(device))
            total_memory = int(torch.cuda.get_device_properties(device).total_memory)
            headroom_fraction = max(0.0, 1.0 - peak / total_memory)
            safe = headroom_fraction >= 0.15
            results.append(
                {
                    "batch_size": candidate,
                    "status": "ok" if safe else "insufficient_headroom",
                    "seconds": seconds,
                    "rows_per_second": candidate / max(seconds, 1e-9),
                    "peak_memory_bytes": peak,
                    "total_memory_bytes": total_memory,
                    "headroom_fraction": headroom_fraction,
                }
            )
            if safe:
                successful.append(candidate)
            del output, inputs, encoded
            if not safe:
                break
        except torch.cuda.OutOfMemoryError:
            results.append({"batch_size": candidate, "status": "oom"})
            torch.cuda.empty_cache()
            break
    if not successful:
        raise RuntimeError("所有候选batch size均OOM或无法保留15%显存余量")
    return max(successful), results


def write_representation_pointer(
    run_directory: Path, artifacts: RepresentationArtifacts
) -> Path:
    run_directory.mkdir(parents=True, exist_ok=True)
    pointer = run_directory / POINTER_FILE
    manifest = json.loads(artifacts.manifest.read_text(encoding="utf-8"))
    payload = {
        "schema_version": "continuous_label_representation_pointer_v2.0",
        "representation_directory": str(artifacts.directory),
        "representation_fingerprint": manifest["representation_fingerprint"],
        "representation_manifest_sha256": sha256_file(artifacts.manifest),
    }
    if pointer.exists():
        existing = json.loads(pointer.read_text(encoding="utf-8"))
        if existing != payload:
            raise RuntimeError(f"run目录已有不兼容Representation pointer: {pointer}")
    else:
        pointer.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    return pointer


def resolve_representation_directory(config: Mapping[str, object]) -> Path:
    output = config.get("output", {})
    if not isinstance(output, Mapping):
        raise ValueError("output配置必须是对象")
    run_directory = Path(str(output.get("run_directory", ""))).expanduser().resolve()
    pointer = run_directory / POINTER_FILE
    if not pointer.is_file():
        raise FileNotFoundError(f"Representation pointer不存在: {pointer}")
    payload = json.loads(pointer.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "continuous_label_representation_pointer_v2.0":
        raise ValueError("Representation pointer协议不匹配，拒绝读取旧主线产物")
    directory = Path(payload["representation_directory"]).expanduser().resolve()
    check = validate_representation_artifacts(directory)
    if check["representation_fingerprint"] != payload["representation_fingerprint"]:
        raise RuntimeError("Representation pointer与manifest指纹不一致")
    return directory


def extract_cls_representations(
    texts: pd.DataFrame,
    *,
    model: Any,
    tokenizer: Any,
    output_directory: str | Path,
    batch_size: int,
    max_length: int,
    expected_layers: int = 13,
    storage_dtype: str = "float16",
    device: str = "cpu",
    model_identity: Mapping[str, object] | None = None,
    layer12_rtol: float | None = None,
    layer12_atol: float | None = None,
) -> RepresentationArtifacts:
    """Extract one canonical representation artifact without overwriting output."""

    import pyarrow as pa
    import pyarrow.parquet as pq
    import torch

    _require_columns(
        texts,
        [
            "representation_row",
            "report_id",
            "text",
            "symbol",
            "feature_available_date",
            "text_sha256",
        ],
        "Representation研报表",
    )
    if not texts["representation_row"].equals(pd.Series(np.arange(len(texts)))):
        raise ValueError("representation_row必须从0连续递增")
    if batch_size <= 0 or max_length <= 0:
        raise ValueError("batch_size和max_length必须为正整数")
    if storage_dtype not in {"float16", "float32"}:
        raise ValueError("storage_dtype必须是float16或float32")
    output = Path(output_directory).expanduser().resolve()
    _safe_output(output)
    if output.exists():
        raise FileExistsError(f"Representation目录已存在，拒绝覆盖: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)

    validation = freeze_and_validate_inference_model(
        model, expected_hidden_layers=expected_layers
    )
    model = model.to(device).eval()
    configured_pooling = str(getattr(model, "pooling", "cls"))
    if configured_pooling != "cls":
        raise ValueError(
            "主协议锁定head_recipe=cls_fc，模型pooling必须为cls；"
            f"实际为{configured_pooling!r}"
        )
    compute_is_half = any(p.dtype == torch.float16 for p in model.parameters())
    rtol = float(
        layer12_rtol
        if layer12_rtol is not None
        else (1e-3 if compute_is_half else 1e-5)
    )
    atol = float(
        layer12_atol
        if layer12_atol is not None
        else (1e-3 if compute_is_half else 1e-6)
    )

    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent))
    n_rows = len(texts)
    representation_map: np.memmap | None = None
    head_writer: pq.ParquetWriter | None = None
    hidden_size: int | None = None
    max_absolute_error = 0.0
    max_relative_error = 0.0
    forward_calls = 0
    disk_preflight: dict[str, int] | None = None
    try:
        with torch.inference_mode():
            for start in range(0, n_rows, batch_size):
                end = min(n_rows, start + batch_size)
                encoded = tokenizer(
                    texts.iloc[start:end]["text"].tolist(),
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                    return_tensors="pt",
                    return_attention_mask=True,
                    return_token_type_ids=True,
                )
                inputs = {}
                for key, value in encoded.items():
                    if key not in {"input_ids", "attention_mask", "token_type_ids"}:
                        continue
                    if device.startswith("cuda"):
                        value = value.pin_memory()
                    inputs[key] = value.to(
                        device, non_blocking=device.startswith("cuda")
                    )
                candidate = model(**inputs, output_hidden_states=True)
                forward_calls += 1
                layer_cls = stack_layer_cls(
                    candidate.hidden_states, expected_layers=expected_layers
                )
                layer_logits = apply_cls_fc_to_layers(
                    model,
                    candidate.hidden_states,
                    attention_mask=inputs.get("attention_mask"),
                )
                if layer_cls.requires_grad or layer_logits.requires_grad:
                    raise RuntimeError("inference_mode下输出仍携带梯度")
                if not torch.isfinite(layer_cls).all() or not torch.isfinite(
                    layer_logits
                ).all():
                    raise RuntimeError("模型输出含NaN或Inf，拒绝写入Representation")
                reference = candidate.logits
                projected = layer_logits[:, -1, :]
                absolute = (reference - projected).abs()
                relative = absolute / reference.abs().clamp_min(atol)
                max_absolute_error = max(max_absolute_error, float(absolute.max()))
                max_relative_error = max(max_relative_error, float(relative.max()))
                if not torch.allclose(reference, projected, rtol=rtol, atol=atol):
                    raise RuntimeError(
                        "Layer 12固定CLS-fc投影与normal forward不一致: "
                        f"max_abs={float(absolute.max())}, max_rel={float(relative.max())}, "
                        f"rtol={rtol}, atol={atol}"
                    )
                if hidden_size is None:
                    hidden_size = int(layer_cls.shape[-1])
                    disk_preflight = disk_budget(
                        output.parent,
                        rows=n_rows,
                        layers=expected_layers,
                        hidden=hidden_size,
                        dtype=storage_dtype,
                    )
                    representation_map = np.lib.format.open_memmap(
                        temporary / REPRESENTATION_FILE,
                        mode="w+",
                        dtype=np.dtype(storage_dtype),
                        shape=(n_rows, expected_layers, hidden_size),
                    )
                assert representation_map is not None
                representation_map[start:end] = (
                    layer_cls.float().cpu().numpy().astype(storage_dtype, copy=False)
                )
                logits = layer_logits.float().cpu().numpy()
                probabilities = torch.softmax(layer_logits, dim=-1).float().cpu().numpy()
                rows = end - start
                frame = pd.DataFrame(
                    {
                        "representation_row": np.repeat(
                            np.arange(start, end, dtype=np.int64), expected_layers
                        ),
                        "layer": np.tile(
                            np.arange(expected_layers, dtype=np.int8), rows
                        ),
                        "class_0_logit": logits[:, :, 0].reshape(-1),
                        "class_1_logit": logits[:, :, 1].reshape(-1),
                        "logit_margin_1_minus_0": (
                            logits[:, :, 1] - logits[:, :, 0]
                        ).reshape(-1),
                        "class_0_prob": probabilities[:, :, 0].reshape(-1),
                        "class_1_prob": probabilities[:, :, 1].reshape(-1),
                        "predicted_class": np.argmax(logits, axis=-1)
                        .astype(np.int8)
                        .reshape(-1),
                    }
                )
                table = pa.Table.from_pandas(frame, preserve_index=False)
                if head_writer is None:
                    head_writer = pq.ParquetWriter(
                        temporary / HEAD_OUTPUT_FILE,
                        table.schema,
                        compression="zstd",
                    )
                head_writer.write_table(table)
                if disk_preflight is not None and (
                    start == 0 or (start // batch_size) % 8 == 0
                ):
                    free_now = shutil.disk_usage(output.parent).free
                    if free_now <= disk_preflight["required_reserve_bytes"]:
                        raise RuntimeError(
                            "Representation生成期间剩余空间跌破安全阈值: "
                            f"free={free_now}, reserve="
                            f"{disk_preflight['required_reserve_bytes']}"
                        )
        if representation_map is None or hidden_size is None or head_writer is None:
            raise RuntimeError("没有生成任何Representation")
        representation_map.flush()
        del representation_map
        head_writer.close()
        head_writer = None

        metadata = texts.drop(columns=["text"], errors="ignore").copy()
        metadata.to_parquet(temporary / METADATA_FILE, index=False, compression="zstd")
        artifact_file_metadata = {}
        for filename in (REPRESENTATION_FILE, METADATA_FILE, HEAD_OUTPUT_FILE):
            path = temporary / filename
            stat = path.stat()
            artifact_file_metadata[filename] = {
                "size_bytes": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
                "sha256": sha256_file(path),
            }
        fingerprint_columns = [
            "report_id",
            "text_sha256",
            "symbol",
            "feature_available_date",
        ]
        text_fingerprint = fingerprint_frame(metadata, fingerprint_columns)
        identity = dict(model_identity or {})
        representation_fingerprint = str(
            identity.get("representation_fingerprint")
            or hashlib.sha256(
                json.dumps(
                    {
                        "text": text_fingerprint,
                        "checkpoint": identity.get("checkpoint_sha256"),
                        "tokenizer": identity.get("tokenizer_sha256"),
                        "max_length": int(max_length),
                        "storage_dtype": storage_dtype,
                        "head_recipe": HEAD_RECIPE,
                    },
                    sort_keys=True,
                ).encode("utf-8")
            ).hexdigest()
        )
        if disk_preflight is None:
            raise RuntimeError("Representation缺少磁盘预检记录")
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "protocol_name": "continuous_label_layer_probe",
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "representation_fingerprint": representation_fingerprint,
            "shape": [n_rows, expected_layers, hidden_size],
            "layer_indices": list(range(expected_layers)),
            "representation": "raw_cls_token",
            "storage_dtype": storage_dtype,
            "compute_dtype": "float16" if compute_is_half else "float32",
            "batch_size": int(batch_size),
            "backbone_forward_calls": int(forward_calls),
            "max_length": int(max_length),
            "truncation": True,
            "text_dataset_fingerprint": text_fingerprint,
            "row_alignment": "representation_row is zero-based and stable",
            "head_contract": {
                "analysis_name": "frozen_cls_fc_projection",
                "recipe": HEAD_RECIPE,
                "provenance": HEAD_PROVENANCE,
                "historical_training_path_confirmed": False,
                "interpretation": (
                    "Layer 12 equivalence validates implementation consistency only; "
                    "it does not establish the historical training architecture."
                ),
            },
            "layer12_equivalence": {
                "passed": True,
                "rtol": rtol,
                "atol": atol,
                "max_absolute_error": max_absolute_error,
                "max_relative_error": max_relative_error,
            },
            "model_validation": validation,
            "model_identity": identity,
            "disk_preflight": disk_preflight,
            "files": {
                "representations": REPRESENTATION_FILE,
                "metadata": METADATA_FILE,
                "fixed_head_layer_outputs": HEAD_OUTPUT_FILE,
            },
            "artifact_files": artifact_file_metadata,
        }
        (temporary / MANIFEST_FILE).write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        os.replace(temporary, output)
    except BaseException:
        if head_writer is not None:
            head_writer.close()
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    artifacts = representation_artifacts(output)
    validate_representation_artifacts(artifacts.directory)
    return artifacts


def validate_representation_artifacts(directory: str | Path) -> dict[str, object]:
    artifacts = representation_artifacts(directory)
    for path in (
        artifacts.representations,
        artifacts.metadata,
        artifacts.fixed_head_outputs,
        artifacts.manifest,
    ):
        if not path.is_file():
            raise FileNotFoundError(f"Representation产物缺失: {path}")
    manifest = json.loads(artifacts.manifest.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("拒绝读取旧协议Representation产物")
    shape = tuple(int(value) for value in manifest.get("shape", []))
    if len(shape) != 3 or shape[1] != 13:
        raise ValueError(f"Representation shape manifest无效: {shape}")
    array = np.load(artifacts.representations, mmap_mode="r")
    if array.shape != shape:
        raise ValueError(f"Representation shape错误: {array.shape} != {shape}")
    artifact_file_metadata = manifest.get("artifact_files", {})
    if not isinstance(artifact_file_metadata, Mapping):
        raise ValueError("Representation manifest缺少artifact_files")
    for path in (
        artifacts.representations,
        artifacts.metadata,
        artifacts.fixed_head_outputs,
    ):
        record = artifact_file_metadata.get(path.name)
        if not isinstance(record, Mapping):
            raise ValueError(f"Representation manifest缺少文件哈希: {path.name}")
        stat = path.stat()
        if stat.st_size != int(record.get("size_bytes", -1)):
            raise ValueError(f"Representation文件大小已变化: {path.name}")
        if stat.st_mtime_ns != int(record.get("mtime_ns", -1)) and sha256_file(
            path
        ) != record.get("sha256"):
            raise ValueError(f"Representation文件内容已变化: {path.name}")
    metadata = pd.read_parquet(artifacts.metadata)
    _require_columns(
        metadata,
        [
            "representation_row",
            "report_id",
            "symbol",
            "feature_available_date",
            "text_sha256",
        ],
        "report_metadata",
    )
    expected = pd.Series(np.arange(shape[0]), name="representation_row")
    if not metadata["representation_row"].reset_index(drop=True).equals(expected):
        raise ValueError("report_metadata的representation_row不连续")
    if metadata["report_id"].duplicated().any():
        raise ValueError("report_metadata的report_id不唯一")
    text_fingerprint = fingerprint_frame(
        metadata,
        ["report_id", "text_sha256", "symbol", "feature_available_date"],
    )
    if text_fingerprint != manifest.get("text_dataset_fingerprint"):
        raise ValueError("report_metadata指纹与manifest不一致")
    head = pd.read_parquet(
        artifacts.fixed_head_outputs,
        columns=["representation_row", "layer"],
    )
    if len(head) != shape[0] * 13:
        raise ValueError("固定头输出行数不是N×13")
    expected_rows = np.repeat(np.arange(shape[0], dtype=np.int64), 13)
    expected_layers = np.tile(np.arange(13, dtype=np.int8), shape[0])
    if not np.array_equal(head["representation_row"].to_numpy(), expected_rows):
        raise ValueError("固定头输出representation_row顺序错误")
    if not np.array_equal(head["layer"].to_numpy(), expected_layers):
        raise ValueError("固定头输出Layer 0~12不完整")
    equivalence = manifest.get("layer12_equivalence", {})
    if not equivalence.get("passed"):
        raise ValueError("Representation manifest未通过Layer 12一致性")
    return {
        "rows": shape[0],
        "layers": shape[1],
        "hidden_size": shape[2],
        "dtype": str(array.dtype),
        "fixed_head_rows": len(head),
        "representation_fingerprint": manifest["representation_fingerprint"],
        "head_recipe": manifest["head_contract"]["recipe"],
        "head_provenance": manifest["head_contract"]["provenance"],
    }


def plot_representation_layer_norms(
    directory: str | Path, *, max_reports: int = 10_000
):
    """Plot a bounded diagnostic sample of raw CLS-vector norms by layer."""

    import matplotlib.pyplot as plt

    artifacts = representation_artifacts(directory)
    validate_representation_artifacts(directory)
    array = np.load(artifacts.representations, mmap_mode="r")
    rows = np.linspace(
        0, len(array) - 1, min(len(array), int(max_reports)), dtype=int
    )
    values = np.asarray(array[rows], dtype=np.float32)
    norms = np.linalg.norm(values, axis=2)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(range(13), norms.mean(axis=0), marker="o")
    ax.fill_between(
        range(13),
        np.quantile(norms, 0.25, axis=0),
        np.quantile(norms, 0.75, axis=0),
        alpha=0.2,
    )
    ax.set(
        title="Raw CLS representation norm",
        xlabel="Layer",
        ylabel="L2 norm (mean and IQR)",
    )
    ax.grid(alpha=0.2)
    fig.tight_layout()
    plt.close(fig)
    return fig


def run_representation_stage(config: Mapping[str, object]) -> RepresentationArtifacts:
    """Build or hash-validate the canonical label-independent artifact."""

    import torch
    from transformers import BertTokenizerFast

    text_cfg = config.get("text", {})
    model_cfg = config.get("model", {})
    output_cfg = config.get("output", {})
    experiment_cfg = config.get("experiment", {})
    performance_cfg = config.get("performance", {})
    if not all(
        isinstance(value, Mapping)
        for value in (
            text_cfg,
            model_cfg,
            output_cfg,
            experiment_cfg,
            performance_cfg,
        )
    ):
        raise ValueError("text/model/output/experiment/performance配置必须是对象")
    if str(model_cfg.get("head_recipe", HEAD_RECIPE)) != HEAD_RECIPE:
        raise ValueError("当前主协议只允许head_recipe=cls_fc")
    texts = load_text_dataset(
        text_cfg.get("path", ""),
        id_column=str(text_cfg.get("id_column", "report_id")),
        text_column=str(text_cfg.get("text_column", "text")),
        symbol_column=str(text_cfg.get("symbol_column", "stock_code")),
        date_column=str(text_cfg.get("date_column", "available_date")),
        limit=int(text_cfg["limit"]) if text_cfg.get("limit") else None,
    )
    base_model = Path(str(model_cfg.get("base_model_dir", ""))).expanduser().resolve()
    checkpoint = Path(str(model_cfg.get("checkpoint", ""))).expanduser().resolve()
    if not base_model.is_dir():
        raise FileNotFoundError(f"tokenizer/base model目录不存在: {base_model}")
    if not checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint不存在: {checkpoint}")
    device = str(model_cfg.get("device", "cuda:1"))
    model_hash = checkpoint_hash(str(checkpoint))
    tokenizer_hash = tokenizer_signature(base_model)
    text_fingerprint = fingerprint_frame(
        texts,
        ["report_id", "text_sha256", "symbol", "feature_available_date"],
    )
    fingerprint_payload = {
        "protocol": SCHEMA_VERSION,
        "text": text_fingerprint,
        "checkpoint": model_hash,
        "tokenizer": tokenizer_hash,
        "max_length": int(model_cfg.get("max_length", 512)),
        "storage_dtype": str(model_cfg.get("storage_dtype", "float16")),
        "head_recipe": HEAD_RECIPE,
    }
    fingerprint = hashlib.sha256(
        json.dumps(fingerprint_payload, sort_keys=True).encode("utf-8")
    ).hexdigest()
    store_value = str(output_cfg.get("representation_store", "")).strip()
    run_value = str(output_cfg.get("run_directory", "")).strip()
    if not store_value or not run_value:
        raise ValueError("output.representation_store和run_directory不能为空")
    store = Path(store_value).expanduser().resolve()
    run_directory = Path(run_value).expanduser().resolve()
    output = store / fingerprint
    if output.exists():
        check = validate_representation_artifacts(output)
        if check["representation_fingerprint"] != fingerprint:
            raise RuntimeError("content-addressed Representation目录指纹不一致")
        artifacts = representation_artifacts(output)
        write_representation_pointer(run_directory, artifacts)
        return artifacts
    resource_audit = configure_layer_probe_runtime(
        performance_cfg, device=device
    )
    gpu_audit = gpu_runtime_audit(device)
    compute_dtype = str(model_cfg.get("compute_dtype", "float16"))
    dtype = (
        torch.float16
        if compute_dtype == "float16" and device.startswith("cuda")
        else torch.float32
    )
    model = build_candidate(
        str(base_model),
        str(checkpoint),
        pooling="cls",
        pooling_confirmed=False,
        model_hash=model_hash,
        device=device,
        dtype=dtype,
    )
    tokenizer = BertTokenizerFast.from_pretrained(base_model, local_files_only=True)
    configured_batch = int(model_cfg.get("batch_size", 128))
    candidates = model_cfg.get("batch_size_candidates", [])
    if device.startswith("cuda") and isinstance(candidates, Sequence) and candidates:
        selected_batch, batch_benchmark = benchmark_batch_size(
            model=model,
            tokenizer=tokenizer,
            texts=texts["text"],
            candidates=[int(value) for value in candidates],
            max_length=int(model_cfg.get("max_length", 512)),
            device=device,
        )
    else:
        selected_batch = configured_batch
        batch_benchmark = [
            {"batch_size": configured_batch, "status": "configured_without_cuda_benchmark"}
        ]
    identity = {
        "experiment_name": str(
            experiment_cfg.get("name", "continuous_label_layer_probe")
        ),
        "text_source_path": str(
            Path(str(text_cfg.get("path", ""))).expanduser().resolve()
        ),
        "text_source_sha256": sha256_file(text_cfg.get("path", "")),
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": model_hash,
        "tokenizer_path": str(base_model),
        "tokenizer_sha256": tokenizer_hash,
        "representation_fingerprint": fingerprint,
        "head_recipe": HEAD_RECIPE,
        "head_provenance": HEAD_PROVENANCE,
        "gpu_runtime": gpu_audit,
        "cpu_runtime": resource_audit,
        "batch_size_benchmark": batch_benchmark,
        "config_sha256": protocol_config_hash(config),
        "git_commit": git_commit(),
    }
    artifacts = extract_cls_representations(
        texts,
        model=model,
        tokenizer=tokenizer,
        output_directory=output,
        batch_size=selected_batch,
        max_length=int(model_cfg.get("max_length", 512)),
        expected_layers=13,
        storage_dtype=str(model_cfg.get("storage_dtype", "float16")),
        device=device,
        model_identity=identity,
        layer12_rtol=(
            float(model_cfg["layer12_rtol"])
            if model_cfg.get("layer12_rtol") is not None
            else None
        ),
        layer12_atol=(
            float(model_cfg["layer12_atol"])
            if model_cfg.get("layer12_atol") is not None
            else None
        ),
    )
    write_representation_pointer(run_directory, artifacts)
    return artifacts
