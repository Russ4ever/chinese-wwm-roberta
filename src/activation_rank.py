"""Label-free activation-rank audit for the frozen Chinese BERT checkpoint.

The module deliberately never imports or reads label/return artifacts.  It
collects centered sufficient statistics for 49 within-block activation sites,
derives singular spectra from covariance matrices, and attributes attention
output compression to the learned output projection.  Notebook code should
only call the stage functions defined here.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import tempfile
import time
from contextlib import AbstractContextManager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, MutableMapping, Sequence

import numpy as np
import pandas as pd

from .config import load_yaml_config
from .inference_cache import tokenizer_signature
from .layer_probe_representations import (
    checkpoint_hash,
    configure_layer_probe_runtime,
    freeze_and_validate_inference_model,
    git_commit,
    gpu_runtime_audit,
    sha256_file,
)
from .models.modeling import build_candidate


SCHEMA_VERSION = "checkpoint_activation_rank_v2.0"
SAMPLE_SCHEMA = "activation_rank_sample_v2.0"
PILOT_SCHEMA = "activation_rank_pilot_v2.0"
SHARD_SCHEMA = "activation_rank_shard_v2.0"
AUXILIARY_SCHEMA = "activation_rank_auxiliary_v2.0"
ANALYSIS_SCHEMA = "activation_rank_analysis_v2.0"
MECHANISM_SCHEMA = "activation_rank_wo_mechanism_v2.0"
FINAL_SCHEMA = "activation_rank_run_v2.0"

PRIMARY_STREAM = "token_natural_filtered"
AUXILIARY_STREAMS = (
    "token_natural_unfiltered",
    "token_unique_filtered",
    "cls_natural_filtered",
    "cls_natural_unfiltered",
)
ALL_STREAMS = (PRIMARY_STREAM,) + AUXILIARY_STREAMS


def activation_sites(hidden_layers: int = 12) -> tuple[str, ...]:
    if hidden_layers <= 0:
        raise ValueError("hidden_layers必须为正整数")
    residual = tuple(f"residual_{layer:02d}" for layer in range(hidden_layers + 1))
    z_sites = tuple(f"z_{layer:02d}" for layer in range(1, hidden_layers + 1))
    attention = tuple(
        f"attention_output_{layer:02d}" for layer in range(1, hidden_layers + 1)
    )
    mlp = tuple(f"mlp_output_{layer:02d}" for layer in range(1, hidden_layers + 1))
    return residual + z_sites + attention + mlp


CORE_SITES = activation_sites(12)


def load_activation_rank_config(path: str | Path) -> dict[str, object]:
    config = load_yaml_config(path)
    required = {
        "experiment",
        "text",
        "model",
        "sampling",
        "numerics",
        "norm_audit",
        "performance",
        "output",
    }
    missing = sorted(required.difference(config))
    if missing:
        raise ValueError(f"Activation Rank配置缺少顶层字段: {missing}")
    forbidden = {"continuous_targets", "returns", "return_probe", "strict_test"}
    present = sorted(forbidden.intersection(config))
    if present:
        raise ValueError(f"无标签Activation Rank配置禁止包含标签/收益字段: {present}")
    forbidden_key_tokens = (
        "label",
        "target",
        "return",
        "split",
        "sae",
        "intervention",
    )

    def walk_keys(value: object, prefix: str = "") -> list[str]:
        hits: list[str] = []
        if isinstance(value, Mapping):
            for key, child in value.items():
                name = str(key).lower()
                dotted = f"{prefix}.{key}" if prefix else str(key)
                if any(token in name for token in forbidden_key_tokens):
                    hits.append(dotted)
                hits.extend(walk_keys(child, dotted))
        elif isinstance(value, list):
            for index, child in enumerate(value):
                hits.extend(walk_keys(child, f"{prefix}[{index}]"))
        return hits

    nested_forbidden = walk_keys(config)
    if nested_forbidden:
        raise ValueError(
            "无标签Activation Rank配置含越界字段: " + ", ".join(nested_forbidden)
        )
    model = _mapping(config, "model")
    if (
        int(model.get("hidden_size", 0)) != 768
        or int(model.get("hidden_layers", 0)) != 12
        or int(model.get("max_length", 0)) != 512
    ):
        raise ValueError("主协议固定12层、hidden size 768及max_length=512")
    return config


def _mapping(config: Mapping[str, object], key: str) -> Mapping[str, object]:
    value = config.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"{key}配置必须是对象")
    return value


def _run_directory(config: Mapping[str, object]) -> Path:
    value = str(_mapping(config, "output").get("run_directory", "")).strip()
    if not value:
        raise ValueError("output.run_directory不能为空")
    path = Path(value).expanduser().resolve()
    if path == Path(path.anchor) or len(path.parts) < 3:
        raise ValueError(f"拒绝使用过宽输出目录: {path}")
    return path


def _json_hash(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()


def _atomic_json(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(dict(value), ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _atomic_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        np.savez(handle, **arrays)
    os.replace(temporary, path)


def _atomic_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    frame.to_parquet(temporary, index=False, compression="zstd")
    os.replace(temporary, path)


def _atomic_hardlink(source: Path, target: Path) -> None:
    """Expose a canonical artifact at a second path without duplicating bytes."""

    if not source.is_file():
        raise FileNotFoundError(source)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + f".{os.getpid()}.tmp")
    if temporary.exists():
        temporary.unlink()
    os.link(source, temporary)
    os.replace(temporary, target)


def _require_exact_manifest(
    path: Path, expected: Mapping[str, object]
) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(f"manifest不存在: {path}")
    actual = json.loads(path.read_text(encoding="utf-8"))
    mismatches = {
        key: {"actual": actual.get(key), "expected": value}
        for key, value in expected.items()
        if actual.get(key) != value
    }
    if mismatches:
        raise RuntimeError(
            "阶段产物指纹不匹配，拒绝复用: "
            + json.dumps(mismatches, ensure_ascii=False, default=str)
        )
    return actual


@dataclass
class OnlineMoments:
    """Centered vector moments with numerically stable Chan merging."""

    count: int
    mean: np.ndarray
    m2: np.ndarray

    @classmethod
    def empty(cls, dimension: int) -> "OnlineMoments":
        if dimension <= 0:
            raise ValueError("dimension必须为正整数")
        return cls(
            count=0,
            mean=np.zeros(dimension, dtype=np.float64),
            m2=np.zeros((dimension, dimension), dtype=np.float64),
        )

    @classmethod
    def from_array(cls, values: np.ndarray) -> "OnlineMoments":
        x = np.asarray(values, dtype=np.float64)
        if x.ndim != 2:
            raise ValueError("moments输入必须是二维数组")
        out = cls.empty(x.shape[1])
        if len(x):
            mean = x.mean(axis=0, dtype=np.float64)
            centered = x - mean
            out.count = int(len(x))
            out.mean = mean
            out.m2 = centered.T @ centered
        return out

    def copy(self) -> "OnlineMoments":
        return OnlineMoments(self.count, self.mean.copy(), self.m2.copy())

    @property
    def dimension(self) -> int:
        return int(self.mean.shape[0])

    def validate(self) -> None:
        if self.count < 0:
            raise ValueError("moments count不能为负")
        if self.mean.ndim != 1 or self.m2.shape != (self.dimension, self.dimension):
            raise ValueError("moments shape不一致")
        if not np.isfinite(self.mean).all() or not np.isfinite(self.m2).all():
            raise ValueError("moments含NaN或Inf")
        if not np.allclose(self.m2, self.m2.T, rtol=1e-10, atol=1e-8):
            raise ValueError("moments M2不对称")

    def merge(self, other: "OnlineMoments") -> "OnlineMoments":
        if self.dimension != other.dimension:
            raise ValueError("合并moments维度不一致")
        if other.count == 0:
            return self
        if self.count == 0:
            self.count = int(other.count)
            self.mean = other.mean.copy()
            self.m2 = other.m2.copy()
            return self
        total = self.count + other.count
        delta = other.mean - self.mean
        self.m2 = (
            self.m2
            + other.m2
            + np.outer(delta, delta) * (self.count * other.count / total)
        )
        self.mean = self.mean + delta * (other.count / total)
        self.count = int(total)
        return self

    def update(self, values: np.ndarray) -> "OnlineMoments":
        return self.merge(OnlineMoments.from_array(values))

    def covariance(self) -> np.ndarray:
        self.validate()
        if self.count < 2:
            raise ValueError("至少需要两个样本计算协方差")
        covariance = self.m2 / (self.count - 1)
        return (covariance + covariance.T) * 0.5


@dataclass
class ScalarMoments:
    count: int = 0
    mean: float = 0.0
    m2: float = 0.0
    minimum: float = math.inf
    maximum: float = -math.inf

    @classmethod
    def from_array(cls, values: np.ndarray) -> "ScalarMoments":
        x = np.asarray(values, dtype=np.float64).reshape(-1)
        if len(x) == 0:
            return cls()
        mean = float(x.mean())
        return cls(
            count=int(len(x)),
            mean=mean,
            m2=float(np.square(x - mean).sum()),
            minimum=float(x.min()),
            maximum=float(x.max()),
        )

    def merge(self, other: "ScalarMoments") -> "ScalarMoments":
        if other.count == 0:
            return self
        if self.count == 0:
            self.count = other.count
            self.mean = other.mean
            self.m2 = other.m2
            self.minimum = other.minimum
            self.maximum = other.maximum
            return self
        total = self.count + other.count
        delta = other.mean - self.mean
        self.m2 += other.m2 + delta * delta * self.count * other.count / total
        self.mean += delta * other.count / total
        self.count = int(total)
        self.minimum = min(self.minimum, other.minimum)
        self.maximum = max(self.maximum, other.maximum)
        return self

    @property
    def std(self) -> float:
        return math.sqrt(self.m2 / (self.count - 1)) if self.count > 1 else 0.0

    def to_dict(self) -> dict[str, object]:
        return {
            "count": int(self.count),
            "mean": float(self.mean),
            "std": float(self.std),
            "minimum": float(self.minimum) if self.count else None,
            "maximum": float(self.maximum) if self.count else None,
        }

    def to_state_dict(self) -> dict[str, object]:
        """Serialize the mergeable state, not merely descriptive statistics."""

        return {
            "count": int(self.count),
            "mean": float(self.mean),
            "m2": float(self.m2),
            "minimum": float(self.minimum) if self.count else None,
            "maximum": float(self.maximum) if self.count else None,
        }

    @classmethod
    def from_state_dict(cls, value: Mapping[str, object]) -> "ScalarMoments":
        count = int(value.get("count", 0))
        if count == 0:
            return cls()
        out = cls(
            count=count,
            mean=float(value["mean"]),
            m2=float(value["m2"]),
            minimum=float(value["minimum"]),
            maximum=float(value["maximum"]),
        )
        if out.m2 < -1e-12 or not all(
            np.isfinite(item) for item in (out.mean, out.m2, out.minimum, out.maximum)
        ):
            raise ValueError("scalar moments状态无效")
        out.m2 = max(out.m2, 0.0)
        return out


def scalar_moment_maps_to_state(
    values: Mapping[str, Mapping[str, ScalarMoments]],
) -> dict[str, dict[str, object]]:
    return {
        f"{population}__{site}": moment.to_state_dict()
        for population, site_map in values.items()
        for site, moment in site_map.items()
    }


def scalar_moment_maps_from_state(
    values: Mapping[str, Mapping[str, object]],
) -> dict[str, dict[str, ScalarMoments]]:
    output = {
        population: {site: ScalarMoments() for site in CORE_SITES}
        for population in ("token", "cls")
    }
    expected = {
        f"{population}__{site}"
        for population in ("token", "cls")
        for site in CORE_SITES
    }
    if set(values) != expected:
        raise ValueError("norm scalar moments状态字段不完整")
    for key, state in values.items():
        population, site = key.split("__", 1)
        output[population][site] = ScalarMoments.from_state_dict(state)
    return output


def merge_scalar_moment_maps(
    left: MutableMapping[str, MutableMapping[str, ScalarMoments]],
    right: Mapping[str, Mapping[str, ScalarMoments]],
) -> MutableMapping[str, MutableMapping[str, ScalarMoments]]:
    for population in ("token", "cls"):
        for site in CORE_SITES:
            left[population][site].merge(right[population][site])
    return left


def merge_moment_maps(
    left: MutableMapping[str, OnlineMoments],
    right: Mapping[str, OnlineMoments],
) -> MutableMapping[str, OnlineMoments]:
    for key, value in right.items():
        if key not in left:
            left[key] = value.copy()
        else:
            left[key].merge(value)
    return left


def moments_to_arrays(
    streams: Mapping[str, Mapping[str, OnlineMoments]],
) -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {}
    for stream, site_map in sorted(streams.items()):
        for site, moment in sorted(site_map.items()):
            moment.validate()
            prefix = f"{stream}__{site}"
            arrays[f"{prefix}__count"] = np.asarray([moment.count], dtype=np.int64)
            arrays[f"{prefix}__mean"] = moment.mean.astype(np.float64, copy=False)
            arrays[f"{prefix}__m2"] = moment.m2.astype(np.float64, copy=False)
    return arrays


def arrays_to_moments(
    arrays: Mapping[str, np.ndarray]
) -> dict[str, dict[str, OnlineMoments]]:
    prefixes = sorted(
        key[: -len("__count")] for key in arrays if key.endswith("__count")
    )
    streams: dict[str, dict[str, OnlineMoments]] = {}
    for prefix in prefixes:
        stream, site = prefix.split("__", 1)
        moment = OnlineMoments(
            count=int(np.asarray(arrays[f"{prefix}__count"]).reshape(-1)[0]),
            mean=np.asarray(arrays[f"{prefix}__mean"], dtype=np.float64),
            m2=np.asarray(arrays[f"{prefix}__m2"], dtype=np.float64),
        )
        moment.validate()
        streams.setdefault(stream, {})[site] = moment
    return streams


def load_moment_file(path: str | Path) -> dict[str, dict[str, OnlineMoments]]:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(source)
    with np.load(source, allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    return arrays_to_moments(arrays)


def covariance_eigendecomposition(
    moment: OnlineMoments,
) -> tuple[np.ndarray, np.ndarray]:
    covariance = moment.covariance()
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    scale = max(float(np.max(np.abs(eigenvalues))), 1.0)
    if float(eigenvalues.min()) < -1e-9 * scale:
        raise ValueError(
            f"协方差存在超出数值容差的负特征值: min={float(eigenvalues.min())}"
        )
    eigenvalues = np.clip(eigenvalues, 0.0, None)
    # Recovering singular values via sqrt magnifies covariance eigensolver noise
    # near exact null directions.  Remove only the FP64 numerical floor.
    numerical_floor = (
        np.finfo(np.float64).eps
        * max(moment.dimension, 1)
        * max(float(eigenvalues[0]), np.finfo(np.float64).tiny)
        * 10.0
    )
    eigenvalues[eigenvalues < numerical_floor] = 0.0
    return eigenvalues, eigenvectors[:, order]


def rank_metrics_from_eigenvalues(
    eigenvalues: Sequence[float], *, dimension: int | None = None
) -> dict[str, float | int]:
    values = np.clip(np.asarray(eigenvalues, dtype=np.float64), 0.0, None)
    if values.ndim != 1 or not len(values):
        raise ValueError("eigenvalues必须是一维非空数组")
    values = np.sort(values)[::-1]
    singular = np.sqrt(values)
    total_singular = float(singular.sum())
    total_variance = float(values.sum())
    if total_singular <= 0 or total_variance <= 0:
        raise ValueError("零方差激活没有有效秩")
    probabilities = singular / total_singular
    positive = probabilities > 0
    effective_rank = float(
        np.exp(-(probabilities[positive] * np.log(probabilities[positive])).sum())
    )
    dim = int(dimension or len(values))
    cumulative = np.cumsum(values) / total_variance

    def k_for(fraction: float) -> int:
        return int(np.searchsorted(cumulative, fraction, side="left") + 1)

    maximum = float(singular[0])
    return {
        "effective_rank": effective_rank,
        "normalized_effective_rank": effective_rank / dim,
        "stable_rank": total_variance / max(float(values[0]), np.finfo(float).tiny),
        "k90_variance": k_for(0.90),
        "k95_variance": k_for(0.95),
        "k99_variance": k_for(0.99),
        "directions_above_1pct_max": int(np.sum(singular > 0.01 * maximum)),
        "dimension": dim,
    }


def rank_metrics_for_moment(moment: OnlineMoments) -> dict[str, float | int]:
    eigenvalues, _ = covariance_eigendecomposition(moment)
    metrics = rank_metrics_from_eigenvalues(eigenvalues, dimension=moment.dimension)
    metrics["count"] = int(moment.count)
    return metrics


def paired_bootstrap_log_ratio_ci(
    numerator: Sequence[float],
    denominator: Sequence[float],
    *,
    samples: int = 5000,
    seed: int = 20260825,
) -> tuple[float, float, float]:
    x = np.asarray(numerator, dtype=np.float64)
    y = np.asarray(denominator, dtype=np.float64)
    if x.shape != y.shape or x.ndim != 1 or len(x) < 2:
        raise ValueError("配对bootstrap需要等长且至少两个shard")
    if np.any(x <= 0) or np.any(y <= 0):
        raise ValueError("有效秩比率必须为正")
    log_ratio = np.log(x) - np.log(y)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(x), size=(int(samples), len(x)))
    draws = np.exp(log_ratio[indices].mean(axis=1))
    point = float(np.exp(log_ratio.mean()))
    lower, upper = np.quantile(draws, [0.025, 0.975])
    return point, float(lower), float(upper)


def collapse_evidence(
    *,
    ratio_ci_upper_z: float,
    ratio_ci_upper_mlp: float,
    ratio_ci_upper_residual: float,
    attention_k99: int,
    z_k99: int,
    mlp_k99: int,
    residual_k99: int,
) -> str:
    uppers = (ratio_ci_upper_z, ratio_ci_upper_mlp, ratio_ci_upper_residual)
    k_reference = min(z_k99, mlp_k99, residual_k99)
    if all(value < 0.90 for value in uppers) and attention_k99 <= 0.90 * k_reference:
        return "clear_collapse"
    if all(value < 1.0 for value in uppers):
        return "mild_compression"
    return "not_found_or_inconclusive"


def _site_layer(site: str) -> tuple[str, int]:
    if site.startswith("residual_"):
        return "residual", int(site.rsplit("_", 1)[1])
    if site.startswith("z_"):
        return "z", int(site.rsplit("_", 1)[1])
    if site.startswith("attention_output_"):
        return "attention_output", int(site.rsplit("_", 1)[1])
    if site.startswith("mlp_output_"):
        return "mlp_output", int(site.rsplit("_", 1)[1])
    raise ValueError(f"未知activation site: {site}")


class BertActivationHooks(AbstractContextManager["BertActivationHooks"]):
    """Register exact BERT activation hooks without retaining full activations."""

    def __init__(
        self,
        model: Any,
        consumer: Callable[[str, Any], None],
        *,
        expected_layers: int = 12,
    ):
        self.model = model
        self.consumer = consumer
        self.expected_layers = int(expected_layers)
        self.expected_sites = activation_sites(self.expected_layers)
        self.handles: list[Any] = []
        self.seen: list[str] = []

    @staticmethod
    def _tensor(output: Any) -> Any:
        return output[0] if isinstance(output, (tuple, list)) else output

    def _hook(self, site: str):
        def callback(_module: Any, _inputs: Any, output: Any) -> None:
            tensor = self._tensor(output)
            if getattr(tensor, "ndim", None) != 3:
                raise RuntimeError(f"{site}输出必须为[batch,tokens,hidden]")
            self.seen.append(site)
            self.consumer(site, tensor)

        return callback

    def __enter__(self) -> "BertActivationHooks":
        bert = getattr(self.model, "bert", self.model)
        layers = list(bert.encoder.layer)
        if len(layers) != self.expected_layers:
            raise ValueError(f"BERT层数不匹配: {len(layers)} != {self.expected_layers}")
        self.handles.append(
            bert.embeddings.register_forward_hook(self._hook("residual_00"))
        )
        for index, layer in enumerate(layers, start=1):
            self.handles.append(
                layer.attention.self.register_forward_hook(self._hook(f"z_{index:02d}"))
            )
            self.handles.append(
                layer.attention.output.dense.register_forward_hook(
                    self._hook(f"attention_output_{index:02d}")
                )
            )
            self.handles.append(
                layer.output.dense.register_forward_hook(
                    self._hook(f"mlp_output_{index:02d}")
                )
            )
            self.handles.append(
                layer.register_forward_hook(self._hook(f"residual_{index:02d}"))
            )
        return self

    def assert_complete_forward(self) -> None:
        missing = sorted(set(self.expected_sites).difference(self.seen))
        unexpected = sorted(set(self.seen).difference(self.expected_sites))
        duplicates = sorted({site for site in self.seen if self.seen.count(site) > 1})
        if (
            len(self.seen) != len(self.expected_sites)
            or missing
            or unexpected
            or duplicates
        ):
            raise RuntimeError(
                "单次前向hook不完整: "
                f"seen={len(self.seen)}, missing={missing}, unexpected={unexpected}, "
                f"duplicates={duplicates}"
            )
        self.seen.clear()

    def __exit__(self, exc_type, exc, traceback) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


class TorchOnlineMoments:
    def __init__(
        self, dimension: int, *, device: str, initial: OnlineMoments | None = None
    ):
        import torch

        self.count = int(initial.count) if initial is not None else 0
        self.mean = torch.as_tensor(
            initial.mean if initial is not None else np.zeros(dimension),
            device=device,
            dtype=torch.float64,
        ).clone()
        self.m2 = torch.as_tensor(
            initial.m2 if initial is not None else np.zeros((dimension, dimension)),
            device=device,
            dtype=torch.float64,
        ).clone()

    @staticmethod
    def batch_stats(values: Any) -> tuple[int, Any, Any]:
        x = values.to(dtype=__import__("torch").float64)
        count = int(x.shape[0])
        if count == 0:
            raise ValueError("空batch不能计算moments")
        mean = x.mean(dim=0)
        centered = x - mean
        m2 = centered.transpose(0, 1).matmul(centered)
        return count, mean, m2

    def merge_stats(self, count: int, mean: Any, m2: Any) -> None:
        if count == 0:
            return
        if self.count == 0:
            self.count = int(count)
            self.mean.copy_(mean)
            self.m2.copy_(m2)
            return
        total = self.count + count
        delta = mean - self.mean
        self.m2.add_(m2).add_(
            delta[:, None] * delta[None, :], alpha=self.count * count / total
        )
        self.mean.add_(delta, alpha=count / total)
        self.count = int(total)

    def update(self, values: Any) -> tuple[int, Any, Any] | None:
        if int(values.shape[0]) == 0:
            return None
        stats = self.batch_stats(values)
        self.merge_stats(*stats)
        return stats

    def to_numpy(self) -> OnlineMoments:
        moment = OnlineMoments(
            count=int(self.count),
            mean=self.mean.detach().cpu().numpy(),
            m2=self.m2.detach().cpu().numpy(),
        )
        moment.validate()
        return moment


class TorchScalarMoments:
    """Device-resident scalar Chan moments; transfers only once at export."""

    def __init__(self, *, device: str):
        import torch

        self.count = 0
        self.mean = torch.zeros((), dtype=torch.float64, device=device)
        self.m2 = torch.zeros((), dtype=torch.float64, device=device)
        self.minimum = torch.full((), math.inf, dtype=torch.float64, device=device)
        self.maximum = torch.full((), -math.inf, dtype=torch.float64, device=device)

    def update(self, values: Any) -> None:
        import torch

        x = values.reshape(-1).to(dtype=torch.float64)
        count = int(x.numel())
        if count == 0:
            return
        mean = x.mean()
        m2 = torch.square(x - mean).sum()
        minimum = x.min()
        maximum = x.max()
        if self.count == 0:
            self.count = count
            self.mean.copy_(mean)
            self.m2.copy_(m2)
            self.minimum.copy_(minimum)
            self.maximum.copy_(maximum)
            return
        total = self.count + count
        delta = mean - self.mean
        self.m2.add_(m2).add_(delta * delta, alpha=self.count * count / total)
        self.mean.add_(delta, alpha=count / total)
        self.minimum.copy_(torch.minimum(self.minimum, minimum))
        self.maximum.copy_(torch.maximum(self.maximum, maximum))
        self.count = int(total)

    def to_numpy(self) -> ScalarMoments:
        if self.count == 0:
            return ScalarMoments()
        return ScalarMoments(
            count=int(self.count),
            mean=float(self.mean.detach().cpu()),
            m2=float(self.m2.detach().cpu()),
            minimum=float(self.minimum.detach().cpu()),
            maximum=float(self.maximum.detach().cpu()),
        )


class ActivationMomentConsumer:
    """Consume hooks immediately and retain only centered moments on device."""

    def __init__(
        self,
        *,
        hidden_size: int,
        device: str,
        thresholds: Mapping[str, float],
        initial_primary: Mapping[str, OnlineMoments] | None = None,
        include_primary: bool = True,
        include_auxiliary: bool = True,
    ):
        import torch

        self.hidden_size = int(hidden_size)
        self.device = device
        self.thresholds = {str(key): float(value) for key, value in thresholds.items()}
        self.include_primary = bool(include_primary)
        self.include_auxiliary = bool(include_auxiliary)
        self.streams: dict[str, dict[str, TorchOnlineMoments]] = {}
        if include_primary:
            self.streams[PRIMARY_STREAM] = {
                site: TorchOnlineMoments(
                    hidden_size,
                    device=device,
                    initial=(initial_primary or {}).get(site),
                )
                for site in CORE_SITES
            }
        if include_auxiliary:
            for stream in AUXILIARY_STREAMS:
                self.streams[stream] = {
                    site: TorchOnlineMoments(hidden_size, device=device)
                    for site in CORE_SITES
                }
        self.norms = {
            population: {site: TorchScalarMoments(device=device) for site in CORE_SITES}
            for population in ("token", "cls")
        }
        self.filtered_counts = {
            population: {
                site: torch.zeros((), dtype=torch.int64, device=device)
                for site in CORE_SITES
            }
            for population in ("token", "cls")
        }
        self.token_mask: Any | None = None
        self.unique_rows: Any | None = None
        self.batch_finite: list[Any] = []

    def begin_batch(self, *, token_mask: Any, unique_rows: Any) -> None:
        if token_mask.ndim != 2 or unique_rows.ndim != 1:
            raise ValueError("token_mask/unique_rows shape错误")
        if token_mask.shape[0] != unique_rows.shape[0]:
            raise ValueError("token_mask与unique_rows batch不一致")
        self.token_mask = token_mask.bool()
        self.unique_rows = unique_rows.bool()
        self.batch_finite = []

    def _threshold(self, population: str, site: str) -> float:
        key = f"{population}__{site}"
        if key not in self.thresholds:
            raise KeyError(f"缺少norm threshold: {key}")
        return self.thresholds[key]

    def __call__(self, site: str, tensor: Any) -> None:
        import torch

        if self.token_mask is None or self.unique_rows is None:
            raise RuntimeError("每个batch前必须调用begin_batch")
        if tuple(tensor.shape[:2]) != tuple(self.token_mask.shape):
            raise RuntimeError(f"{site}激活与token mask shape不一致")
        if int(tensor.shape[-1]) != self.hidden_size:
            raise RuntimeError(f"{site} hidden size不一致")
        self.batch_finite.append(torch.isfinite(tensor).all())
        token_values = tensor[self.token_mask]
        token_norm = torch.linalg.vector_norm(token_values.float(), dim=1)
        cls_values = tensor[:, 0, :]
        cls_norm = torch.linalg.vector_norm(cls_values.float(), dim=1)
        self.norms["token"][site].update(token_norm)
        self.norms["cls"][site].update(cls_norm)
        token_keep = token_norm <= self._threshold("token", site)
        cls_keep = cls_norm <= self._threshold("cls", site)
        self.filtered_counts["token"][site].add_((~token_keep).sum())
        self.filtered_counts["cls"][site].add_((~cls_keep).sum())
        if self.include_primary:
            self.streams[PRIMARY_STREAM][site].update(token_values[token_keep])
        if not self.include_auxiliary:
            return
        self.streams["token_natural_unfiltered"][site].update(token_values)
        unique_token_mask = self.token_mask & self.unique_rows[:, None]
        unique_values = tensor[unique_token_mask]
        unique_norm = torch.linalg.vector_norm(unique_values.float(), dim=1)
        unique_keep = unique_norm <= self._threshold("token", site)
        unique_stream = self.streams["token_unique_filtered"][site]
        unique_stream.update(unique_values[unique_keep])
        self.streams["cls_natural_filtered"][site].update(cls_values[cls_keep])
        self.streams["cls_natural_unfiltered"][site].update(cls_values)

    def assert_batch_valid(self) -> None:
        import torch

        if len(self.batch_finite) != len(CORE_SITES):
            raise RuntimeError("batch非有限值检查未覆盖全部activation sites")
        if not bool(torch.stack(self.batch_finite).all().item()):
            raise RuntimeError("batch激活含NaN或Inf")

    def export_norms(self) -> dict[str, dict[str, ScalarMoments]]:
        return {
            population: {site: moment.to_numpy() for site, moment in site_map.items()}
            for population, site_map in self.norms.items()
        }

    def export_filtered_counts(self) -> dict[str, dict[str, int]]:
        return {
            population: {
                site: int(value.detach().cpu()) for site, value in site_map.items()
            }
            for population, site_map in self.filtered_counts.items()
        }

    def export(self) -> dict[str, dict[str, OnlineMoments]]:
        return {
            stream: {site: moment.to_numpy() for site, moment in site_map.items()}
            for stream, site_map in self.streams.items()
        }


class PilotMomentConsumer:
    """Unfiltered token/CLS moments and norm statistics for precision pilot."""

    def __init__(self, *, hidden_size: int, device: str):
        self.hidden_size = int(hidden_size)
        self.device = device
        self.token = {
            site: TorchOnlineMoments(hidden_size, device=device) for site in CORE_SITES
        }
        self.cls = {
            site: TorchOnlineMoments(hidden_size, device=device) for site in CORE_SITES
        }
        self.norms = {
            population: {site: TorchScalarMoments(device=device) for site in CORE_SITES}
            for population in ("token", "cls")
        }
        self.token_mask: Any | None = None
        self.batch_finite: list[Any] = []

    def begin_batch(self, *, token_mask: Any, unique_rows: Any | None = None) -> None:
        self.token_mask = token_mask.bool()
        self.batch_finite = []

    def __call__(self, site: str, tensor: Any) -> None:
        import torch

        if self.token_mask is None or tuple(tensor.shape[:2]) != tuple(
            self.token_mask.shape
        ):
            raise RuntimeError("pilot mask尚未设置或shape不匹配")
        self.batch_finite.append(torch.isfinite(tensor).all())
        token = tensor[self.token_mask]
        cls = tensor[:, 0, :]
        token_norm = torch.linalg.vector_norm(token.float(), dim=1)
        cls_norm = torch.linalg.vector_norm(cls.float(), dim=1)
        self.norms["token"][site].update(token_norm)
        self.norms["cls"][site].update(cls_norm)
        self.token[site].update(token)
        self.cls[site].update(cls)

    def export(self) -> dict[str, dict[str, OnlineMoments]]:
        return {
            "token": {site: value.to_numpy() for site, value in self.token.items()},
            "cls": {site: value.to_numpy() for site, value in self.cls.items()},
        }

    def export_norms(self) -> dict[str, dict[str, ScalarMoments]]:
        return {
            population: {site: moment.to_numpy() for site, moment in site_map.items()}
            for population, site_map in self.norms.items()
        }

    def assert_batch_valid(self) -> None:
        import torch

        if len(self.batch_finite) != len(CORE_SITES):
            raise RuntimeError("pilot非有限值检查未覆盖全部activation sites")
        if not bool(torch.stack(self.batch_finite).all().item()):
            raise RuntimeError("pilot batch激活含NaN或Inf")


class BenchmarkMomentConsumer:
    """Exercise the same FP64 Gram-matrix kernel without retaining moments."""

    def __init__(self):
        self.token_mask: Any | None = None
        self.checksum: Any | None = None
        self.batch_finite: list[Any] = []

    def begin_batch(self, *, token_mask: Any, unique_rows: Any | None = None) -> None:
        self.token_mask = token_mask.bool()
        self.batch_finite = []

    def __call__(self, site: str, tensor: Any) -> None:
        import torch

        if self.token_mask is None or tuple(tensor.shape[:2]) != tuple(
            self.token_mask.shape
        ):
            raise RuntimeError("benchmark mask尚未设置或shape不匹配")
        values = tensor[self.token_mask].to(dtype=torch.float64)
        self.batch_finite.append(torch.isfinite(tensor).all())
        centered = values - values.mean(dim=0)
        gram = centered.transpose(0, 1).matmul(centered)
        self.checksum = (
            gram[0, 0] if self.checksum is None else self.checksum + gram[0, 0]
        )

    def assert_batch_valid(self) -> None:
        import torch

        if len(self.batch_finite) != len(CORE_SITES) or not bool(
            torch.stack(self.batch_finite).all().item()
        ):
            raise RuntimeError("benchmark batch hook或finite校验失败")

    def checksum_value(self) -> float:
        return float(self.checksum.detach().cpu()) if self.checksum is not None else 0.0


def _stable_hash(value: str, seed: str) -> int:
    digest = hashlib.sha256(f"{seed}|{value}".encode()).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


def _read_report_texts(config: Mapping[str, object]) -> pd.DataFrame:
    text_cfg = _mapping(config, "text")
    path = Path(str(text_cfg.get("path", ""))).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"研报文本不存在: {path}")
    id_column = str(text_cfg.get("id_column", "report_id"))
    text_column = str(text_cfg.get("text_column", "text"))
    if path.suffix.lower() not in {".parquet", ".pq"}:
        raise ValueError("Activation Rank主协议只接受canonical Parquet研报表")
    frame = pd.read_parquet(path, columns=[id_column, text_column]).rename(
        columns={id_column: "report_id", text_column: "text"}
    )
    frame.insert(0, "source_row", np.arange(len(frame), dtype=np.int64))
    frame["report_id"] = frame["report_id"].astype("string").str.strip()
    frame["text"] = frame["text"].fillna("").astype(str).str.strip()
    if frame["report_id"].isna().any() or frame["report_id"].eq("").any():
        raise ValueError("研报含空report_id")
    if frame["report_id"].duplicated().any():
        raise ValueError("研报report_id不唯一")
    if frame["text"].eq("").any():
        raise ValueError("研报含空文本")
    frame["text_sha256"] = frame["text"].map(
        lambda value: hashlib.sha256(value.encode()).hexdigest()
    )
    return frame


def _model_paths(config: Mapping[str, object]) -> tuple[Path, Path]:
    model_cfg = _mapping(config, "model")
    base = Path(str(model_cfg.get("base_model_dir", ""))).expanduser().resolve()
    checkpoint = Path(str(model_cfg.get("checkpoint", ""))).expanduser().resolve()
    if not base.is_dir():
        raise FileNotFoundError(f"base model/tokenizer目录不存在: {base}")
    if not checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint不存在: {checkpoint}")
    return base, checkpoint


def _identity(config: Mapping[str, object]) -> dict[str, object]:
    text_path = (
        Path(str(_mapping(config, "text").get("path", ""))).expanduser().resolve()
    )
    base, checkpoint = _model_paths(config)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "checkpoint_sha256": checkpoint_hash(str(checkpoint)),
        "tokenizer_sha256": tokenizer_signature(base),
        "text_source_sha256": sha256_file(text_path),
        "config_sha256": _json_hash(config),
        "max_length": int(_mapping(config, "model").get("max_length", 512)),
        "git_commit": git_commit(),
        "activation_rank_code_sha256": sha256_file(Path(__file__)),
    }
    payload["run_fingerprint"] = _json_hash(payload)
    return payload


def preflight_activation_rank(config: Mapping[str, object]) -> pd.DataFrame:
    model_cfg = _mapping(config, "model")
    performance = _mapping(config, "performance")
    output = _run_directory(config)
    output.parent.mkdir(parents=True, exist_ok=True)
    base, checkpoint = _model_paths(config)
    text_path = (
        Path(str(_mapping(config, "text").get("path", ""))).expanduser().resolve()
    )
    if not text_path.is_file():
        raise FileNotFoundError(text_path)
    estimated_peak = int(float(performance.get("estimated_peak_gib", 6)) * (1 << 30))
    usage = shutil.disk_usage(output.parent)
    reserve = max(
        int(float(performance.get("minimum_free_disk_gb", 15)) * (1 << 30)),
        int(usage.free * float(performance.get("minimum_free_disk_fraction", 0.20))),
    )
    disk_ok = estimated_peak <= usage.free - reserve
    records: list[dict[str, object]] = [
        {"check": "text", "status": "ok", "detail": str(text_path)},
        {"check": "checkpoint", "status": "ok", "detail": str(checkpoint)},
        {"check": "tokenizer", "status": "ok", "detail": str(base)},
        {
            "check": "disk_budget",
            "status": "ok" if disk_ok else "invalid",
            "detail": json.dumps(
                {
                    "estimated_peak": estimated_peak,
                    "free": usage.free,
                    "reserve": reserve,
                }
            ),
        },
    ]
    if not disk_ok:
        raise RuntimeError("Activation Rank磁盘预算不足")
    runtime = configure_layer_probe_runtime(
        performance, device=str(model_cfg.get("device", "cuda:1"))
    )
    gpu = gpu_runtime_audit(str(model_cfg.get("device", "cuda:1")))
    records.extend(
        [
            {
                "check": "cpu_runtime",
                "status": "ok",
                "detail": json.dumps(runtime, default=str),
            },
            {
                "check": "gpu_runtime",
                "status": "ok",
                "detail": json.dumps(gpu, default=str),
            },
        ]
    )
    return pd.DataFrame(records)


def _token_counts(
    tokenizer: Any, texts: Sequence[str], *, max_length: int
) -> list[int]:
    encoded = tokenizer(
        list(texts),
        add_special_tokens=True,
        truncation=True,
        max_length=int(max_length),
        padding=False,
        return_special_tokens_mask=True,
    )
    masks = encoded["special_tokens_mask"]
    return [int(sum(1 - int(value) for value in mask)) for mask in masks]


def validate_sample_manifest(
    directory: str | Path, expected: Mapping[str, object] | None = None
) -> dict[str, object]:
    root = Path(directory)
    manifest = _require_exact_manifest(
        root / "manifest.json",
        {"schema_version": SAMPLE_SCHEMA, **dict(expected or {})},
    )
    table = root / "sample_manifest.parquet"
    if not table.is_file():
        raise FileNotFoundError(table)
    frame = pd.read_parquet(table)
    required = {
        "source_row",
        "report_id",
        "text_sha256",
        "stable_order",
        "shard",
        "token_count",
        "is_unique_text",
    }
    if not required.issubset(frame.columns):
        raise ValueError(
            f"sample manifest缺字段: {sorted(required.difference(frame.columns))}"
        )
    if frame["report_id"].duplicated().any() or (frame["token_count"] <= 0).any():
        raise ValueError("sample manifest含重复report_id或非正token_count")
    if sha256_file(table) != manifest.get("sample_manifest_sha256"):
        raise RuntimeError("sample manifest文件hash不匹配")
    public_table = root.parent / "sample_manifest.parquet"
    if not public_table.is_file() or not os.path.samefile(table, public_table):
        raise RuntimeError(
            "sample_manifest.parquet公开路径缺失或不是canonical hardlink"
        )
    return manifest


def run_sample_stage(config: Mapping[str, object]) -> Path:
    from transformers import BertTokenizerFast

    identity = _identity(config)
    output = _run_directory(config) / "sample"
    expected = {"run_fingerprint": identity["run_fingerprint"]}
    if output.exists():
        validate_sample_manifest(output, expected)
        return output
    sampling = _mapping(config, "sampling")
    model_cfg = _mapping(config, "model")
    shards = int(sampling.get("shards", 8))
    checkpoints = [int(value) for value in sampling.get("token_checkpoints", [])]
    if shards != 8 or checkpoints != [500000, 1000000, 2000000, 5000000, 10000000]:
        raise ValueError("主协议固定8 shards及50万/100万/200万/500万/1000万token检查点")
    seed = str(sampling.get("hash_seed", "checkpoint_activation_rank_v1"))
    frame = _read_report_texts(config)
    frame["stable_order"] = frame["report_id"].map(
        lambda value: _stable_hash(str(value), seed)
    )
    frame["shard"] = (frame["stable_order"] % shards).astype(np.int8)
    frame = frame.sort_values(
        ["stable_order", "report_id"], kind="mergesort"
    ).reset_index(drop=True)
    base, _ = _model_paths(config)
    tokenizer = BertTokenizerFast.from_pretrained(base, local_files_only=True)
    max_length = int(model_cfg.get("max_length", 512))
    local_target = int(math.ceil(max(checkpoints) / shards))
    counts = np.zeros(shards, dtype=np.int64)
    selected: list[pd.DataFrame] = []
    chunk_size = int(sampling.get("tokenization_batch_size", 2048))
    for start in range(0, len(frame), chunk_size):
        chunk = frame.iloc[start : start + chunk_size].copy()
        chunk["token_count"] = _token_counts(
            tokenizer, chunk["text"].tolist(), max_length=max_length
        )
        keep_rows = []
        for row_index, row in chunk.iterrows():
            shard = int(row["shard"])
            if counts[shard] >= local_target:
                continue
            keep_rows.append(row_index)
            counts[shard] += int(row["token_count"])
        if keep_rows:
            selected.append(chunk.loc[keep_rows])
        if bool(np.all(counts >= local_target)):
            break
    if not bool(np.all(counts >= local_target)):
        raise RuntimeError(f"研报语料不足以达到每shard token预算: {counts.tolist()}")
    sample = pd.concat(selected, ignore_index=True)
    sample = sample.sort_values(["stable_order", "report_id"], kind="mergesort")
    sample["is_unique_text"] = ~sample["text_sha256"].duplicated(keep="first")
    sample = sample.drop(columns=["text"])
    sample["sample_row"] = np.arange(len(sample), dtype=np.int64)
    sample = sample[
        [
            "sample_row",
            "source_row",
            "report_id",
            "text_sha256",
            "stable_order",
            "shard",
            "token_count",
            "is_unique_text",
        ]
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent))
    try:
        table = temporary / "sample_manifest.parquet"
        sample.to_parquet(table, index=False, compression="zstd")
        manifest = {
            **identity,
            "schema_version": SAMPLE_SCHEMA,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "rows": int(len(sample)),
            "token_counts_by_shard": {
                str(i): int(sample.loc[sample["shard"].eq(i), "token_count"].sum())
                for i in range(shards)
            },
            "sample_manifest_sha256": sha256_file(table),
            "sampling_policy": "stable_hash_natural_distribution",
            "labels_or_returns_loaded": False,
        }
        (temporary / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        os.replace(temporary, output)
        _atomic_hardlink(
            output / "sample_manifest.parquet",
            output.parent / "sample_manifest.parquet",
        )
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    validate_sample_manifest(output, expected)
    return output


def _sample_with_text(
    config: Mapping[str, object], sample_directory: Path
) -> pd.DataFrame:
    sample = pd.read_parquet(sample_directory / "sample_manifest.parquet")
    reports = _read_report_texts(config)[
        ["source_row", "report_id", "text", "text_sha256"]
    ]
    merged = sample.merge(
        reports,
        on=["source_row", "report_id", "text_sha256"],
        how="left",
        validate="one_to_one",
    )
    if merged["text"].isna().any():
        raise RuntimeError("sample manifest无法严格回连canonical文本")
    return merged.sort_values(["shard", "stable_order"], kind="mergesort").reset_index(
        drop=True
    )


def _torch_dtype(name: str, *, device: str):
    import torch

    normalized = str(name).lower()
    if not device.startswith("cuda") and normalized in {"float16", "bfloat16"}:
        return torch.float32
    choices = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    if normalized not in choices:
        raise ValueError(f"不支持compute dtype: {name}")
    return choices[normalized]


def _tokenize_batch_cpu(
    tokenizer: Any, rows: pd.DataFrame, *, max_length: int
) -> tuple[dict[str, Any], Any, Any, int]:
    import torch

    encoded = tokenizer(
        rows["text"].tolist(),
        padding=True,
        truncation=True,
        max_length=int(max_length),
        return_tensors="pt",
        return_attention_mask=True,
        return_token_type_ids=True,
        return_special_tokens_mask=True,
    )
    special = encoded.pop("special_tokens_mask").bool()
    attention = encoded["attention_mask"].bool()
    token_mask = attention & ~special
    unique = torch.as_tensor(rows["is_unique_text"].to_numpy(dtype=bool))
    valid_tokens = int(token_mask.sum().item())
    return encoded, token_mask, unique, valid_tokens


def _move_tokenized_batch(
    payload: tuple[dict[str, Any], Any, Any, int], *, device: str
) -> tuple[dict[str, Any], Any, Any, int]:
    encoded, token_mask, unique, valid_tokens = payload
    if device.startswith("cuda"):
        encoded = {
            key: value.pin_memory().to(device, non_blocking=True)
            for key, value in encoded.items()
        }
        token_mask = token_mask.pin_memory().to(device, non_blocking=True)
        unique = unique.pin_memory().to(device, non_blocking=True)
    return encoded, token_mask, unique, valid_tokens


def _tokenize_batch(
    tokenizer: Any, rows: pd.DataFrame, *, max_length: int, device: str
) -> tuple[dict[str, Any], Any, Any, int]:
    return _move_tokenized_batch(
        _tokenize_batch_cpu(tokenizer, rows, max_length=max_length), device=device
    )


def _length_bucketed_batches(
    rows: pd.DataFrame, *, batch_size: int, max_length: int
) -> Iterable[pd.DataFrame]:
    if batch_size <= 0:
        raise ValueError("batch_size必须为正整数")
    boundaries = [value for value in (64, 128, 256, 512) if value < int(max_length)]
    working = rows.copy()
    lengths = working["token_count"].to_numpy(dtype=np.int64)
    working["_length_bucket"] = np.searchsorted(
        np.asarray(boundaries, dtype=np.int64), lengths, side="left"
    )
    working = working.sort_values(
        ["_length_bucket", "token_count", "stable_order"], kind="mergesort"
    )
    for _, group in working.groupby("_length_bucket", sort=True):
        clean = group.drop(columns=["_length_bucket"])
        for start in range(0, len(clean), int(batch_size)):
            yield clean.iloc[start : start + int(batch_size)]


def _run_rows(
    *,
    model: Any,
    tokenizer: Any,
    rows: pd.DataFrame,
    consumer: Any,
    batch_size: int,
    max_length: int,
    device: str,
) -> dict[str, float | int]:
    import torch
    from concurrent.futures import ThreadPoolExecutor

    started = time.perf_counter()
    tokens = 0
    batches = 0
    padding_slots = 0
    actual_slots = 0
    batch_iterator = iter(
        _length_bucketed_batches(
            rows, batch_size=int(batch_size), max_length=max_length
        )
    )
    first_batch = next(batch_iterator, None)
    if first_batch is None:
        raise ValueError("不能运行空rows")
    with (
        BertActivationHooks(model, consumer) as hooks,
        torch.inference_mode(),
        ThreadPoolExecutor(max_workers=1, thread_name_prefix="rank-tokenizer") as pool,
    ):
        future = pool.submit(
            _tokenize_batch_cpu, tokenizer, first_batch, max_length=max_length
        )
        while future is not None:
            cpu_payload = future.result()
            next_batch = next(batch_iterator, None)
            next_future = (
                pool.submit(
                    _tokenize_batch_cpu, tokenizer, next_batch, max_length=max_length
                )
                if next_batch is not None
                else None
            )
            encoded, token_mask, unique, valid = _move_tokenized_batch(
                cpu_payload, device=device
            )
            consumer.begin_batch(token_mask=token_mask, unique_rows=unique)
            _ = model(
                **encoded,
                output_hidden_states=False,
            )
            hooks.assert_complete_forward()
            if hasattr(consumer, "assert_batch_valid"):
                consumer.assert_batch_valid()
            tokens += valid
            batches += 1
            padding_slots += int(encoded["attention_mask"].numel())
            actual_slots += int(encoded["attention_mask"].sum().item())
            future = next_future
    if device.startswith("cuda"):
        torch.cuda.synchronize(device)
    seconds = time.perf_counter() - started
    return {
        "rows": int(len(rows)),
        "valid_tokens": int(tokens),
        "batches": int(batches),
        "seconds": float(seconds),
        "valid_tokens_per_second": float(tokens / max(seconds, 1e-9)),
        "padding_ratio": float(1.0 - actual_slots / max(padding_slots, 1)),
    }


def select_activation_batch_size(
    records: Sequence[Mapping[str, object]], *, minimum_headroom: float = 0.15
) -> int:
    safe = [
        record
        for record in records
        if record.get("status") == "ok"
        and float(record.get("headroom_fraction", 1.0)) >= minimum_headroom
    ]
    if not safe:
        raise RuntimeError("所有候选batch均OOM或无法保留要求的显存余量")
    best = max(
        safe,
        key=lambda record: (
            float(record.get("valid_tokens_per_second", 0.0)),
            int(record.get("batch_size", 0)),
        ),
    )
    return int(best["batch_size"])


def benchmark_activation_batch_sizes(
    *,
    model: Any,
    tokenizer: Any,
    rows: pd.DataFrame,
    candidates: Sequence[int],
    max_length: int,
    device: str,
) -> tuple[int, list[dict[str, object]]]:
    """Benchmark hook plus FP64 moment kernels on representative text batches."""

    import torch

    values = sorted({int(value) for value in candidates if int(value) > 0})
    if not values:
        raise ValueError("batch_size_candidates必须含正整数")
    if not device.startswith("cuda"):
        configured = values[0]
        return configured, [
            {
                "batch_size": configured,
                "status": "ok",
                "headroom_fraction": 1.0,
                "valid_tokens_per_second": 0.0,
                "device": "cpu",
            }
        ]
    records: list[dict[str, object]] = []
    total_memory = int(torch.cuda.get_device_properties(device).total_memory)
    representative = rows.sort_values("stable_order", kind="mergesort")
    for candidate in values:
        repeats = int(math.ceil((2 * candidate) / max(len(representative), 1)))
        candidate_rows = pd.concat([representative] * repeats, ignore_index=True).iloc[
            : 2 * candidate
        ]
        try:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
            consumer = BenchmarkMomentConsumer()
            throughput = _run_rows(
                model=model,
                tokenizer=tokenizer,
                rows=candidate_rows,
                consumer=consumer,
                batch_size=candidate,
                max_length=max_length,
                device=device,
            )
            peak = int(torch.cuda.max_memory_allocated(device))
            headroom = max(0.0, 1.0 - peak / max(total_memory, 1))
            records.append(
                {
                    "batch_size": candidate,
                    "status": "ok" if headroom >= 0.15 else "insufficient_headroom",
                    "peak_memory_bytes": peak,
                    "total_memory_bytes": total_memory,
                    "headroom_fraction": headroom,
                    "checksum": consumer.checksum_value(),
                    **throughput,
                }
            )
            del consumer
            if headroom < 0.15:
                break
        except torch.cuda.OutOfMemoryError:
            records.append({"batch_size": candidate, "status": "oom"})
            torch.cuda.empty_cache()
            break
    return select_activation_batch_size(records), records


def _pilot_rows(sample: pd.DataFrame, token_budget: int) -> pd.DataFrame:
    pieces = []
    total = 0
    for shard in sorted(sample["shard"].unique()):
        group = sample[sample["shard"].eq(shard)].copy()
        target = int(math.ceil(token_budget / sample["shard"].nunique()))
        cumulative = group["token_count"].cumsum()
        take = int(np.searchsorted(cumulative.to_numpy(), target, side="left") + 1)
        pieces.append(group.iloc[:take])
        total += int(group.iloc[:take]["token_count"].sum())
    if total < token_budget:
        raise RuntimeError("pilot样本token不足")
    return pd.concat(pieces, ignore_index=True)


def _precision_comparison(
    reference: Mapping[str, OnlineMoments], candidate: Mapping[str, OnlineMoments]
) -> dict[str, float | int | bool]:
    max_erank = 0.0
    max_k99 = 0
    max_spectrum = 0.0
    for site in CORE_SITES:
        ref_values, _ = covariance_eigendecomposition(reference[site])
        cand_values, _ = covariance_eigendecomposition(candidate[site])
        ref_metrics = rank_metrics_from_eigenvalues(ref_values)
        cand_metrics = rank_metrics_from_eigenvalues(cand_values)
        max_erank = max(
            max_erank,
            abs(
                float(ref_metrics["normalized_effective_rank"])
                - float(cand_metrics["normalized_effective_rank"])
            ),
        )
        max_k99 = max(
            max_k99,
            abs(int(ref_metrics["k99_variance"]) - int(cand_metrics["k99_variance"])),
        )
        ref_singular = np.sqrt(ref_values)
        cand_singular = np.sqrt(cand_values)
        mask = ref_singular > 0.01 * ref_singular[0]
        relative = np.abs(cand_singular[mask] - ref_singular[mask]) / np.maximum(
            ref_singular[mask], 1e-12
        )
        if len(relative):
            max_spectrum = max(max_spectrum, float(relative.max()))
    passed = max_erank <= 0.005 and max_k99 <= 5 and max_spectrum <= 0.01
    return {
        "maximum_normalized_erank_difference": max_erank,
        "maximum_k99_difference": int(max_k99),
        "maximum_relative_spectrum_difference_above_1pct": max_spectrum,
        "passed": bool(passed),
    }


def select_compute_dtype(records: Sequence[Mapping[str, object]]) -> str:
    passed = [record for record in records if bool(record.get("passed"))]
    if not passed:
        return "float32"
    return str(
        max(
            passed, key=lambda record: float(record.get("valid_tokens_per_second", 0.0))
        )["dtype"]
    )


def compute_dtype_epsilon(dtype_name: str) -> float:
    normalized = str(dtype_name).lower()
    values = {
        "float32": float(np.finfo(np.float32).eps),
        "float16": float(np.finfo(np.float16).eps),
        "bfloat16": float(2.0**-7),
    }
    if normalized not in values:
        raise ValueError(f"不支持的compute dtype epsilon: {dtype_name}")
    return values[normalized]


def select_norm_calibration(
    norms_by_dtype: Mapping[str, Mapping[str, Mapping[str, ScalarMoments]]],
    selected_dtype: str,
) -> Mapping[str, Mapping[str, ScalarMoments]]:
    if selected_dtype not in norms_by_dtype:
        raise KeyError(f"缺少selected dtype的norm calibration: {selected_dtype}")
    return norms_by_dtype[selected_dtype]


def validate_pilot_outputs(
    directory: str | Path, expected: Mapping[str, object] | None = None
) -> dict[str, object]:
    root = Path(directory)
    manifest = _require_exact_manifest(
        root / "manifest.json", {"schema_version": PILOT_SCHEMA, **dict(expected or {})}
    )
    thresholds = root / "norm_thresholds.json"
    norm_stats = root / "norm_stats.json"
    metrics = root / "precision_metrics.parquet"
    batch_metrics = root / "batch_benchmark.parquet"
    if not all(
        path.is_file() for path in (thresholds, norm_stats, metrics, batch_metrics)
    ):
        raise FileNotFoundError("pilot缺少norm thresholds/stats或benchmark metrics")
    values = json.loads(thresholds.read_text(encoding="utf-8"))
    expected_keys = {
        f"{population}__{site}"
        for population in ("token", "cls")
        for site in CORE_SITES
    }
    if set(values) != expected_keys or not all(
        np.isfinite(float(v)) and float(v) > 0 for v in values.values()
    ):
        raise ValueError("pilot norm thresholds不完整或无效")
    if sha256_file(thresholds) != manifest.get("norm_thresholds_sha256"):
        raise RuntimeError("pilot norm thresholds hash不匹配")
    if sha256_file(norm_stats) != manifest.get("norm_stats_sha256"):
        raise RuntimeError("pilot norm stats hash不匹配")
    if sha256_file(metrics) != manifest.get("precision_metrics_sha256"):
        raise RuntimeError("pilot precision metrics hash不匹配")
    if sha256_file(batch_metrics) != manifest.get("batch_benchmark_sha256"):
        raise RuntimeError("pilot batch benchmark hash不匹配")
    stats_values = json.loads(norm_stats.read_text(encoding="utf-8"))
    if set(stats_values) != expected_keys:
        raise ValueError("pilot norm stats不完整")
    precision = pd.read_parquet(metrics)
    if "float32" not in set(precision["dtype"]) or not precision["passed"].any():
        raise ValueError("pilot缺少通过的float32 reference")
    if manifest.get("norm_calibration_dtype") != manifest.get("selected_compute_dtype"):
        raise RuntimeError("pilot norm calibration dtype与主扫描dtype不一致")
    if not isinstance(manifest.get("norm_audit_policy_sha256"), str):
        raise RuntimeError("pilot缺少norm audit policy fingerprint")
    return manifest


def _execution_identity(pilot_directory: Path) -> dict[str, object]:
    manifest = validate_pilot_outputs(pilot_directory)
    payload = {
        "selected_compute_dtype": manifest["selected_compute_dtype"],
        "norm_calibration_dtype": manifest["norm_calibration_dtype"],
        "selected_batch_size": int(manifest["batch_size"]),
        "norm_thresholds_sha256": manifest["norm_thresholds_sha256"],
        "norm_stats_sha256": manifest["norm_stats_sha256"],
        "precision_metrics_sha256": manifest["precision_metrics_sha256"],
        "batch_benchmark_sha256": manifest["batch_benchmark_sha256"],
        "norm_audit_policy_sha256": manifest["norm_audit_policy_sha256"],
    }
    payload["execution_fingerprint"] = _json_hash(payload)
    return payload


def run_pilot_stage(config: Mapping[str, object]) -> Path:
    import torch
    from transformers import BertTokenizerFast

    identity = _identity(config)
    output = _run_directory(config) / "pilot"
    expected = {"run_fingerprint": identity["run_fingerprint"]}
    if output.exists():
        validate_pilot_outputs(output, expected)
        return output
    preflight_activation_rank(config)
    sample_directory = run_sample_stage(config)
    sample = _sample_with_text(config, sample_directory)
    sampling = _mapping(config, "sampling")
    numerics = _mapping(config, "numerics")
    model_cfg = _mapping(config, "model")
    rows = _pilot_rows(sample, int(sampling.get("pilot_token_budget", 500000)))
    base, checkpoint = _model_paths(config)
    tokenizer = BertTokenizerFast.from_pretrained(base, local_files_only=True)
    device = str(model_cfg.get("device", "cuda:1"))
    hidden_size = int(model_cfg.get("hidden_size", 768))
    max_length = int(model_cfg.get("max_length", 512))
    benchmark_model = build_candidate(
        str(base),
        str(checkpoint),
        pooling="cls",
        pooling_confirmed=False,
        model_hash=identity["checkpoint_sha256"],
        device=device,
        dtype=_torch_dtype("float32", device=device),
    )
    freeze_and_validate_inference_model(benchmark_model, expected_hidden_layers=13)
    batch_size, batch_records = benchmark_activation_batch_sizes(
        model=benchmark_model,
        tokenizer=tokenizer,
        rows=rows,
        candidates=[
            int(value)
            for value in model_cfg.get("batch_size_candidates", [64, 128, 256, 512])
        ],
        max_length=max_length,
        device=device,
    )
    del benchmark_model
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    dtype_records: list[dict[str, object]] = []
    moment_results: dict[str, dict[str, OnlineMoments]] = {}
    norm_results_by_dtype: dict[str, dict[str, dict[str, ScalarMoments]]] = {}
    dtype_candidates = [
        str(value)
        for value in numerics.get(
            "dtype_candidates", ["float32", "float16", "bfloat16"]
        )
    ]
    if not device.startswith("cuda"):
        dtype_candidates = ["float32"]
    for dtype_name in dtype_candidates:
        dtype = _torch_dtype(dtype_name, device=device)
        model = build_candidate(
            str(base),
            str(checkpoint),
            pooling="cls",
            pooling_confirmed=False,
            model_hash=identity["checkpoint_sha256"],
            device=device,
            dtype=dtype,
        )
        freeze_and_validate_inference_model(model, expected_hidden_layers=13)
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
        consumer = PilotMomentConsumer(hidden_size=hidden_size, device=device)
        throughput = _run_rows(
            model=model,
            tokenizer=tokenizer,
            rows=rows,
            consumer=consumer,
            batch_size=batch_size,
            max_length=max_length,
            device=device,
        )
        exported = consumer.export()
        moment_results[dtype_name] = exported["token"]
        norm_results_by_dtype[dtype_name] = consumer.export_norms()
        peak = (
            int(torch.cuda.max_memory_allocated(device))
            if device.startswith("cuda")
            else 0
        )
        dtype_records.append(
            {"dtype": dtype_name, **throughput, "peak_memory_bytes": peak}
        )
        del consumer, model
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
    if "float32" not in moment_results or "float32" not in norm_results_by_dtype:
        raise RuntimeError("pilot必须包含严格float32 reference")
    reference = moment_results["float32"]
    for record in dtype_records:
        name = str(record["dtype"])
        comparison = (
            {
                "maximum_normalized_erank_difference": 0.0,
                "maximum_k99_difference": 0,
                "maximum_relative_spectrum_difference_above_1pct": 0.0,
                "passed": True,
            }
            if name == "float32"
            else _precision_comparison(reference, moment_results[name])
        )
        record.update(comparison)
    selected_dtype = select_compute_dtype(dtype_records)
    norm_results = select_norm_calibration(norm_results_by_dtype, selected_dtype)
    thresholds = {
        f"{population}__{site}": float(
            stats.mean + float(numerics.get("outlier_sigma", 5.0)) * stats.std
        )
        for population, site_map in norm_results.items()
        for site, stats in site_map.items()
    }
    temporary = Path(tempfile.mkdtemp(prefix=".pilot-", dir=_run_directory(config)))
    try:
        threshold_path = temporary / "norm_thresholds.json"
        threshold_path.write_text(
            json.dumps(thresholds, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        norm_stats_path = temporary / "norm_stats.json"
        norm_stats_path.write_text(
            json.dumps(
                {
                    f"{population}__{site}": stats.to_dict()
                    for population, site_map in norm_results.items()
                    for site, stats in site_map.items()
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        precision = pd.DataFrame(dtype_records)
        precision_path = temporary / "precision_metrics.parquet"
        precision.to_parquet(precision_path, index=False, compression="zstd")
        batch_path = temporary / "batch_benchmark.parquet"
        pd.DataFrame(batch_records).to_parquet(
            batch_path, index=False, compression="zstd"
        )
        best_rate = float(
            precision.loc[
                precision["dtype"].eq(selected_dtype), "valid_tokens_per_second"
            ].iloc[0]
        )
        manifest = {
            **identity,
            "schema_version": PILOT_SCHEMA,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "selected_compute_dtype": selected_dtype,
            "norm_calibration_dtype": selected_dtype,
            "pilot_valid_tokens": int(
                max(record["valid_tokens"] for record in dtype_records)
            ),
            "batch_size": batch_size,
            "norm_policy": f"population_site_mean_plus_{float(numerics.get('outlier_sigma', 5.0)):g}sigma",
            "norm_audit_policy_sha256": _json_hash(_mapping(config, "norm_audit")),
            "norm_thresholds_sha256": sha256_file(threshold_path),
            "norm_stats_sha256": sha256_file(norm_stats_path),
            "precision_metrics_sha256": sha256_file(precision_path),
            "batch_benchmark_sha256": sha256_file(batch_path),
            "eta_5m_seconds": 5000000 / max(best_rate, 1e-9),
            "eta_10m_seconds": 10000000 / max(best_rate, 1e-9),
            "labels_or_returns_loaded": False,
        }
        (temporary / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        os.replace(temporary, output)
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    validate_pilot_outputs(output, expected)
    return output


def _local_checkpoint_targets(config: Mapping[str, object]) -> list[int]:
    sampling = _mapping(config, "sampling")
    shards = int(sampling.get("shards", 8))
    return [
        int(math.ceil(int(value) / shards))
        for value in sampling.get("token_checkpoints", [])
    ]


def _rows_to_local_budget(rows: pd.DataFrame, target: int) -> pd.DataFrame:
    cumulative = rows["token_count"].cumsum().to_numpy()
    end = int(np.searchsorted(cumulative, int(target), side="left") + 1)
    if end > len(rows):
        raise RuntimeError(f"shard样本不足以达到local token target={target}")
    return rows.iloc[:end].copy()


def _moment_metrics_snapshot(
    stream: Mapping[str, OnlineMoments], checkpoint: int
) -> list[dict[str, object]]:
    records = []
    for site in CORE_SITES:
        metrics = rank_metrics_for_moment(stream[site])
        kind, layer = _site_layer(site)
        records.append(
            {
                "checkpoint_tokens": int(checkpoint),
                "site": site,
                "kind": kind,
                "layer": layer,
                **metrics,
            }
        )
    return records


def _validate_primary_shard(
    root: Path,
    identity: Mapping[str, object],
    shard: int,
    execution: Mapping[str, object] | None = None,
) -> dict[str, object] | None:
    directory = root / "moments" / f"shard_{shard:02d}"
    if not directory.exists():
        return None
    manifest = _require_exact_manifest(
        directory / "manifest.json",
        {
            "schema_version": SHARD_SCHEMA,
            "run_fingerprint": identity["run_fingerprint"],
            "shard": int(shard),
            **(
                {"execution_fingerprint": execution["execution_fingerprint"]}
                if execution is not None
                else {}
            ),
        },
    )
    moments = directory / "primary_moments.npz"
    if not moments.is_file() or sha256_file(moments) != manifest.get("moments_sha256"):
        raise RuntimeError(f"shard {shard} moments缺失或hash不匹配")
    loaded = load_moment_file(moments)
    if set(loaded) != {PRIMARY_STREAM} or set(loaded[PRIMARY_STREAM]) != set(
        CORE_SITES
    ):
        raise ValueError(f"shard {shard} primary moments不完整")
    public_moments = root / "moments" / f"shard_{shard:02d}.npz"
    if not public_moments.is_file() or not os.path.samefile(moments, public_moments):
        raise RuntimeError(f"shard {shard}公开moments路径缺失或不是canonical hardlink")
    return manifest


def _load_auxiliary_state(
    root: Path,
    identity: Mapping[str, object],
    execution: Mapping[str, object],
) -> tuple[dict[str, dict[str, OnlineMoments]], dict[str, object]]:
    path = root / "moments" / "auxiliary_state.npz"
    manifest_path = root / "moments" / "auxiliary_manifest.json"
    if not path.exists() and not manifest_path.exists():
        return {}, {
            "schema_version": AUXILIARY_SCHEMA,
            "run_fingerprint": identity["run_fingerprint"],
            "execution_fingerprint": execution["execution_fingerprint"],
            "included_segments": [],
        }
    manifest = _require_exact_manifest(
        manifest_path,
        {
            "schema_version": AUXILIARY_SCHEMA,
            "run_fingerprint": identity["run_fingerprint"],
            "execution_fingerprint": execution["execution_fingerprint"],
        },
    )
    if not path.is_file() or sha256_file(path) != manifest.get("moments_sha256"):
        raise RuntimeError("auxiliary moments缺失或hash不匹配")
    streams = load_moment_file(path)
    if set(streams) != set(AUXILIARY_STREAMS):
        raise ValueError("auxiliary streams不完整")
    return streams, manifest


def _publish_auxiliary_state(
    root: Path,
    identity: Mapping[str, object],
    execution: Mapping[str, object],
    existing: dict[str, dict[str, OnlineMoments]],
    delta: Mapping[str, Mapping[str, OnlineMoments]],
    *,
    segment: str,
) -> None:
    _, manifest = _load_auxiliary_state(root, identity, execution)
    included = list(manifest.get("included_segments", []))
    if segment in included:
        return
    for stream in AUXILIARY_STREAMS:
        merge_moment_maps(existing.setdefault(stream, {}), delta[stream])
    path = root / "moments" / "auxiliary_state.npz"
    _atomic_npz(path, moments_to_arrays(existing))
    included.append(segment)
    payload = {
        "schema_version": AUXILIARY_SCHEMA,
        "run_fingerprint": identity["run_fingerprint"],
        **dict(execution),
        "included_segments": included,
        "moments_sha256": sha256_file(path),
        "updated_at": datetime.now().isoformat(timespec="seconds"),
    }
    _atomic_json(root / "moments" / "auxiliary_manifest.json", payload)


def _publish_primary_shard(
    root: Path,
    identity: Mapping[str, object],
    execution: Mapping[str, object],
    *,
    shard: int,
    primary: Mapping[str, OnlineMoments],
    processed_rows: int,
    processed_tokens: int,
    checkpoints: Sequence[Mapping[str, object]],
    norm_audit: Mapping[str, object],
    norm_states: Mapping[str, Mapping[str, object]],
    filtered_counts: Mapping[str, int],
) -> Path:
    output = root / "moments" / f"shard_{shard:02d}"
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".shard_{shard:02d}-", dir=output.parent))
    try:
        moments_path = temporary / "primary_moments.npz"
        with moments_path.open("wb") as handle:
            np.savez(handle, **moments_to_arrays({PRIMARY_STREAM: primary}))
        pd.DataFrame(checkpoints).to_parquet(
            temporary / "checkpoint_metrics.parquet", index=False, compression="zstd"
        )
        manifest = {
            **identity,
            "schema_version": SHARD_SCHEMA,
            **dict(execution),
            "shard": int(shard),
            "processed_rows": int(processed_rows),
            "processed_tokens": int(processed_tokens),
            "moments_sha256": sha256_file(moments_path),
            "norm_audit": dict(norm_audit),
            "norm_states": dict(norm_states),
            "filtered_counts": {
                str(key): int(value) for key, value in filtered_counts.items()
            },
            "completed_at": datetime.now().isoformat(timespec="seconds"),
        }
        (temporary / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        if output.exists():
            backup = output.with_name(output.name + f".{os.getpid()}.old")
            os.replace(output, backup)
            os.replace(temporary, output)
            shutil.rmtree(backup)
        else:
            os.replace(temporary, output)
        _atomic_hardlink(
            output / "primary_moments.npz",
            output.parent / f"shard_{shard:02d}.npz",
        )
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    return output


def _norm_audit_table(
    norms: Mapping[str, Mapping[str, ScalarMoments]],
    filtered_counts: Mapping[str, int],
    pilot_norms: Mapping[str, Mapping[str, object]],
    *,
    compute_dtype: str,
    policy: Mapping[str, object],
) -> pd.DataFrame:
    blocking_populations = {
        str(value) for value in policy.get("blocking_populations", ["token"])
    }
    if not blocking_populations or not blocking_populations.issubset({"token", "cls"}):
        raise ValueError("norm_audit.blocking_populations只能包含token/cls且不能为空")
    floor_policy = str(
        policy.get("std_denominator_floor", "compute_dtype_epsilon_times_abs_mean")
    )
    if floor_policy != "compute_dtype_epsilon_times_abs_mean":
        raise ValueError(f"不支持的norm std denominator floor: {floor_policy}")
    mean_tolerance = float(policy.get("mean_relative_shift_tolerance", 0.01))
    std_tolerance = float(policy.get("std_relative_shift_tolerance", 0.01))
    filtered_tolerance = float(policy.get("maximum_filtered_fraction", 0.001))
    if min(mean_tolerance, std_tolerance, filtered_tolerance) < 0:
        raise ValueError("norm audit容差不能为负")
    dtype_epsilon = compute_dtype_epsilon(compute_dtype)
    records: list[dict[str, object]] = []
    for population in ("token", "cls"):
        for site in CORE_SITES:
            current = norms[population][site]
            reference = pilot_norms[f"{population}__{site}"]
            reference_mean = max(abs(float(reference["mean"])), 1e-12)
            reference_std = abs(float(reference["std"]))
            std_denominator_floor = dtype_epsilon * reference_mean
            std_denominator = max(reference_std, std_denominator_floor, 1e-12)
            mean_shift = abs(current.mean - float(reference["mean"])) / reference_mean
            std_shift = abs(current.std - reference_std) / std_denominator
            fraction = int(filtered_counts[f"{population}__{site}"]) / max(
                current.count, 1
            )
            mean_passed = mean_shift <= mean_tolerance
            std_passed = std_shift <= std_tolerance
            filtered_passed = fraction <= filtered_tolerance
            position_passed = mean_passed and std_passed and filtered_passed
            records.append(
                {
                    "population": population,
                    "site": site,
                    "blocking": population in blocking_populations,
                    "compute_dtype": compute_dtype,
                    "dtype_epsilon": dtype_epsilon,
                    "pilot_count": int(reference["count"]),
                    "main_count": int(current.count),
                    "pilot_mean": float(reference["mean"]),
                    "main_mean": float(current.mean),
                    "pilot_std": reference_std,
                    "main_std": float(current.std),
                    "std_denominator_floor": std_denominator_floor,
                    "std_denominator": std_denominator,
                    "relative_norm_mean_shift": mean_shift,
                    "relative_norm_std_shift": std_shift,
                    "filtered_count": int(filtered_counts[f"{population}__{site}"]),
                    "filtered_fraction": fraction,
                    "mean_tolerance": mean_tolerance,
                    "std_tolerance": std_tolerance,
                    "filtered_fraction_tolerance": filtered_tolerance,
                    "mean_passed": mean_passed,
                    "std_passed": std_passed,
                    "filtered_passed": filtered_passed,
                    "position_passed": position_passed,
                }
            )
    return pd.DataFrame(records)


def _summarize_norm_audit(
    table: pd.DataFrame, *, compute_dtype: str
) -> dict[str, object]:
    def maximum_record(frame: pd.DataFrame, metric: str) -> dict[str, object]:
        row = frame.loc[frame[metric].idxmax()]
        return {
            "value": float(row[metric]),
            "population": str(row["population"]),
            "site": str(row["site"]),
        }

    blocking = table[table["blocking"]]
    token = table[table["population"].eq("token")]
    cls = table[table["population"].eq("cls")]
    failed = blocking[~blocking["position_passed"]]
    cls_failed = cls[~cls["position_passed"]]
    metrics = (
        "relative_norm_mean_shift",
        "relative_norm_std_shift",
        "filtered_fraction",
    )
    return {
        "compute_dtype": compute_dtype,
        "std_denominator_floor_policy": "compute_dtype_epsilon_times_abs_mean",
        "blocking_populations": sorted(set(blocking["population"].astype(str))),
        "passed": bool(blocking["position_passed"].all()),
        "token_passed": bool(token["position_passed"].all()),
        "cls_passed": bool(cls["position_passed"].all()),
        "failed_blocking_position_count": int(len(failed)),
        "failed_blocking_positions": failed[
            [
                "population",
                "site",
                "relative_norm_mean_shift",
                "relative_norm_std_shift",
                "filtered_fraction",
            ]
        ].to_dict("records"),
        "cls_warning_position_count": int(len(cls_failed)),
        "maxima": {
            "all": {metric: maximum_record(table, metric) for metric in metrics},
            "token": {metric: maximum_record(token, metric) for metric in metrics},
            "cls": {metric: maximum_record(cls, metric) for metric in metrics},
        },
    }


def _norm_audit(
    norms: Mapping[str, Mapping[str, ScalarMoments]],
    filtered_counts: Mapping[str, int],
    pilot_norms: Mapping[str, Mapping[str, object]],
    *,
    compute_dtype: str,
    policy: Mapping[str, object],
) -> tuple[dict[str, object], pd.DataFrame]:
    table = _norm_audit_table(
        norms,
        filtered_counts,
        pilot_norms,
        compute_dtype=compute_dtype,
        policy=policy,
    )
    return _summarize_norm_audit(table, compute_dtype=compute_dtype), table


def _pilot_norm_reference(
    pilot_directory: Path, *, outlier_sigma: float
) -> dict[str, dict[str, object]]:
    validate_pilot_outputs(pilot_directory)
    thresholds = json.loads(
        (pilot_directory / "norm_thresholds.json").read_text(encoding="utf-8")
    )
    sigma = float(outlier_sigma)
    reference: dict[str, dict[str, object]] = {}
    # The threshold alone cannot recover mean/std.  Store conservative values in
    # old manifests only as an explicit failure; current pilot writes norm_stats.
    stats_path = pilot_directory / "norm_stats.json"
    if not stats_path.is_file():
        raise RuntimeError("pilot缺少norm_stats.json，无法审计主样本分布漂移")
    values = json.loads(stats_path.read_text(encoding="utf-8"))
    for key, record in values.items():
        if not math.isclose(
            float(thresholds[key]),
            float(record["mean"]) + sigma * float(record["std"]),
            rel_tol=1e-6,
            abs_tol=1e-6,
        ):
            raise RuntimeError(f"pilot threshold与norm stats不一致: {key}")
        reference[key] = record
    return reference


def _run_one_shard_to_target(
    config: Mapping[str, object],
    sample: pd.DataFrame,
    *,
    shard: int,
    target: int,
    identity: Mapping[str, object],
    model: Any | None = None,
    tokenizer: Any | None = None,
) -> dict[str, object]:
    import torch
    from transformers import BertTokenizerFast

    root = _run_directory(config)
    pilot = root / "pilot"
    pilot_manifest = validate_pilot_outputs(
        pilot, {"run_fingerprint": identity["run_fingerprint"]}
    )
    execution = _execution_identity(pilot)
    existing_manifest = _validate_primary_shard(root, identity, shard, execution)
    processed_rows = (
        int(existing_manifest.get("processed_rows", 0)) if existing_manifest else 0
    )
    processed_tokens = (
        int(existing_manifest.get("processed_tokens", 0)) if existing_manifest else 0
    )
    auxiliary, auxiliary_manifest = _load_auxiliary_state(root, identity, execution)
    segment = f"shard_{shard:02d}_through_{int(target)}"
    primary_complete = processed_tokens >= target
    auxiliary_complete = segment in set(auxiliary_manifest.get("included_segments", []))
    if primary_complete and auxiliary_complete:
        return dict(existing_manifest)
    if primary_complete and not auxiliary_complete:
        raise RuntimeError(
            "发现主分片已发布但对应辅助segment缺失；该不完整状态拒绝自动复用"
        )
    shard_rows = sample[sample["shard"].eq(shard)].sort_values(
        "stable_order", kind="mergesort"
    )
    target_rows = _rows_to_local_budget(shard_rows, target)
    segment_rows = target_rows.iloc[processed_rows:].copy()
    if segment_rows.empty:
        raise RuntimeError("shard需要扩展但没有新增报告")
    initial_primary = None
    if existing_manifest:
        loaded = load_moment_file(
            root / "moments" / f"shard_{shard:02d}" / "primary_moments.npz"
        )
        initial_primary = loaded[PRIMARY_STREAM]
    thresholds = json.loads(
        (pilot / "norm_thresholds.json").read_text(encoding="utf-8")
    )
    numerics = _mapping(config, "numerics")
    norm_reference = _pilot_norm_reference(
        pilot, outlier_sigma=float(numerics.get("outlier_sigma", 5.0))
    )
    model_cfg = _mapping(config, "model")
    base, checkpoint = _model_paths(config)
    device = str(model_cfg.get("device", "cuda:1"))
    dtype_name = str(pilot_manifest["selected_compute_dtype"])
    if model is None:
        model = build_candidate(
            str(base),
            str(checkpoint),
            pooling="cls",
            pooling_confirmed=False,
            model_hash=identity["checkpoint_sha256"],
            device=device,
            dtype=_torch_dtype(dtype_name, device=device),
        )
    freeze_and_validate_inference_model(model, expected_hidden_layers=13)
    if tokenizer is None:
        tokenizer = BertTokenizerFast.from_pretrained(base, local_files_only=True)
    consumer = ActivationMomentConsumer(
        hidden_size=int(model_cfg.get("hidden_size", 768)),
        device=device,
        thresholds=thresholds,
        initial_primary=initial_primary,
        include_primary=True,
        include_auxiliary=True,
    )
    throughput = _run_rows(
        model=model,
        tokenizer=tokenizer,
        rows=segment_rows,
        consumer=consumer,
        batch_size=int(pilot_manifest["batch_size"]),
        max_length=int(model_cfg.get("max_length", 512)),
        device=device,
    )
    exported = consumer.export()
    cumulative_norms = (
        scalar_moment_maps_from_state(existing_manifest["norm_states"])
        if existing_manifest
        else {
            population: {site: ScalarMoments() for site in CORE_SITES}
            for population in ("token", "cls")
        }
    )
    segment_norms = consumer.export_norms()
    segment_filtered = consumer.export_filtered_counts()
    merge_scalar_moment_maps(cumulative_norms, segment_norms)
    cumulative_filtered = {
        f"{population}__{site}": (
            int(
                (existing_manifest or {})
                .get("filtered_counts", {})
                .get(f"{population}__{site}", 0)
            )
            + int(segment_filtered[population][site])
        )
        for population in ("token", "cls")
        for site in CORE_SITES
    }
    audit, _ = _norm_audit(
        cumulative_norms,
        cumulative_filtered,
        norm_reference,
        compute_dtype=dtype_name,
        policy=_mapping(config, "norm_audit"),
    )
    new_processed_rows = len(target_rows)
    new_processed_tokens = int(target_rows["token_count"].sum())
    previous_checkpoints: list[dict[str, object]] = []
    if existing_manifest:
        checkpoint_path = (
            root / "moments" / f"shard_{shard:02d}" / "checkpoint_metrics.parquet"
        )
        if checkpoint_path.is_file():
            previous_checkpoints = pd.read_parquet(checkpoint_path).to_dict("records")
    snapshot = _moment_metrics_snapshot(exported[PRIMARY_STREAM], int(target))
    checkpoints = previous_checkpoints + snapshot
    # Publish auxiliary first.  If interruption occurs before the primary
    # publication, a retry recomputes this segment but the segment ledger makes
    # the auxiliary merge a no-op.
    _publish_auxiliary_state(
        root, identity, execution, auxiliary, exported, segment=segment
    )
    _publish_primary_shard(
        root,
        identity,
        execution,
        shard=shard,
        primary=exported[PRIMARY_STREAM],
        processed_rows=new_processed_rows,
        processed_tokens=new_processed_tokens,
        checkpoints=checkpoints,
        norm_audit={**audit, **throughput},
        norm_states=scalar_moment_maps_to_state(cumulative_norms),
        filtered_counts=cumulative_filtered,
    )
    if device.startswith("cuda"):
        del consumer
        torch.cuda.empty_cache()
    return dict(_validate_primary_shard(root, identity, shard, execution) or {})


def evaluate_checkpoint_stability(
    root: str | Path,
    *,
    shards: int = 8,
    earlier_checkpoint: int | None = None,
    later_checkpoint: int | None = None,
) -> dict[str, object]:
    directory = Path(root)
    records = []
    for shard in range(shards):
        table = (
            directory / "moments" / f"shard_{shard:02d}" / "checkpoint_metrics.parquet"
        )
        if not table.is_file():
            raise FileNotFoundError(table)
        frame = pd.read_parquet(table)
        frame["shard"] = shard
        records.append(frame)
    metrics = pd.concat(records, ignore_index=True)
    targets = sorted(metrics["checkpoint_tokens"].unique())
    if len(targets) < 2:
        return {"passed": False, "reason": "insufficient_checkpoints"}
    earlier = (
        int(earlier_checkpoint) if earlier_checkpoint is not None else int(targets[-2])
    )
    later = int(later_checkpoint) if later_checkpoint is not None else int(targets[-1])
    if earlier not in targets or later not in targets:
        raise RuntimeError(
            f"稳定性所需检查点不存在: earlier={earlier}, later={later}, available={targets}"
        )
    left = metrics[metrics["checkpoint_tokens"].eq(earlier)]
    right = metrics[metrics["checkpoint_tokens"].eq(later)]
    joined = left.merge(
        right,
        on=["shard", "site"],
        suffixes=("_earlier", "_later"),
        validate="one_to_one",
    )
    erank_change = np.abs(
        joined["normalized_effective_rank_later"]
        - joined["normalized_effective_rank_earlier"]
    )
    k99_change = np.abs(joined["k99_variance_later"] - joined["k99_variance_earlier"])
    cv_values = []
    for _, group in right.groupby("site"):
        values = group["effective_rank"].to_numpy(dtype=float)
        cv_values.append(
            float(np.std(values, ddof=1) / max(abs(np.mean(values)), 1e-12))
        )
    result = {
        "earlier_checkpoint_tokens_per_shard": int(earlier),
        "later_checkpoint_tokens_per_shard": int(later),
        "maximum_normalized_erank_change": float(erank_change.max()),
        "maximum_k99_change": int(k99_change.max()),
        "maximum_shard_erank_cv": float(max(cv_values)),
    }
    result["passed"] = bool(
        result["maximum_normalized_erank_change"] <= 0.005
        and result["maximum_k99_change"] <= 5
        and result["maximum_shard_erank_cv"] <= 0.005
    )
    return result


def _pooled_norm_audit(
    manifests: Sequence[Mapping[str, object]],
    pilot_directory: Path,
    *,
    outlier_sigma: float,
    compute_dtype: str,
    policy: Mapping[str, object],
) -> tuple[dict[str, object], pd.DataFrame]:
    pooled = {
        population: {site: ScalarMoments() for site in CORE_SITES}
        for population in ("token", "cls")
    }
    filtered = {
        f"{population}__{site}": 0
        for population in ("token", "cls")
        for site in CORE_SITES
    }
    for manifest in manifests:
        merge_scalar_moment_maps(
            pooled, scalar_moment_maps_from_state(manifest["norm_states"])
        )
        for key in filtered:
            filtered[key] += int(manifest["filtered_counts"][key])
    reference = _pilot_norm_reference(pilot_directory, outlier_sigma=outlier_sigma)
    return _norm_audit(
        pooled,
        filtered,
        reference,
        compute_dtype=compute_dtype,
        policy=policy,
    )


def validate_rank_shards_outputs(
    directory: str | Path,
    *,
    identity: Mapping[str, object],
    execution: Mapping[str, object],
    shards: int,
) -> dict[str, object]:
    root = Path(directory)
    manifest = _require_exact_manifest(
        root / "manifest.json",
        {
            "schema_version": SHARD_SCHEMA,
            "run_fingerprint": identity["run_fingerprint"],
            "execution_fingerprint": execution["execution_fingerprint"],
            "shards": int(shards),
        },
    )
    if not bool(manifest.get("norm_audit", {}).get("passed")):
        raise RuntimeError("主运行全样本norm审计未通过")
    audit_path = root / "norm_audit.parquet"
    if not audit_path.is_file() or sha256_file(audit_path) != manifest.get(
        "norm_audit_sha256"
    ):
        raise RuntimeError("主运行逐位置norm audit缺失或hash不匹配")
    for shard in range(shards):
        _validate_primary_shard(root.parent, identity, shard, execution)
    _load_auxiliary_state(root.parent, identity, execution)
    return manifest


def run_rank_shards_stage(config: Mapping[str, object]) -> Path:
    import torch
    from transformers import BertTokenizerFast

    identity = _identity(config)
    root = _run_directory(config)
    pilot = run_pilot_stage(config)
    execution = _execution_identity(pilot)
    checkpoints = _local_checkpoint_targets(config)
    shards = int(_mapping(config, "sampling").get("shards", 8))
    stage_directory = root / "moments"
    if (stage_directory / "manifest.json").exists():
        validate_rank_shards_outputs(
            stage_directory,
            identity=identity,
            execution=execution,
            shards=shards,
        )
        return stage_directory
    preflight_activation_rank(config)
    sample_directory = run_sample_stage(config)
    sample = _sample_with_text(config, sample_directory)
    model_cfg = _mapping(config, "model")
    base, checkpoint = _model_paths(config)
    device = str(model_cfg.get("device", "cuda:1"))
    pilot_manifest = validate_pilot_outputs(pilot)
    compute_dtype = str(pilot_manifest["selected_compute_dtype"])
    norm_policy = _mapping(config, "norm_audit")
    model = build_candidate(
        str(base),
        str(checkpoint),
        pooling="cls",
        pooling_confirmed=False,
        model_hash=identity["checkpoint_sha256"],
        device=device,
        dtype=_torch_dtype(
            str(pilot_manifest["selected_compute_dtype"]), device=device
        ),
    )
    freeze_and_validate_inference_model(model, expected_hidden_layers=13)
    tokenizer = BertTokenizerFast.from_pretrained(base, local_files_only=True)
    five_million_target = checkpoints[-2]
    try:
        for shard in range(shards):
            for target in checkpoints[:-1]:
                _run_one_shard_to_target(
                    config,
                    sample,
                    shard=shard,
                    target=target,
                    identity=identity,
                    model=model,
                    tokenizer=tokenizer,
                )
        manifests = [
            _validate_primary_shard(root, identity, shard, execution)
            for shard in range(shards)
        ]
        norm_audit, norm_audit_table = _pooled_norm_audit(
            [value for value in manifests if value],
            pilot,
            outlier_sigma=float(_mapping(config, "numerics").get("outlier_sigma", 5.0)),
            compute_dtype=compute_dtype,
            policy=norm_policy,
        )
        _atomic_parquet(stage_directory / "norm_audit_5m.parquet", norm_audit_table)
        _atomic_json(stage_directory / "norm_audit_5m.json", norm_audit)
        if not norm_audit["passed"]:
            raise RuntimeError(
                "普通token norm/过滤审计失败；逐位置明细已写入"
                "moments/norm_audit_5m.parquet: "
                + json.dumps(norm_audit, ensure_ascii=False)
            )
        stability = evaluate_checkpoint_stability(
            root,
            shards=shards,
            earlier_checkpoint=checkpoints[-3],
            later_checkpoint=checkpoints[-2],
        )
        already_at_10m = (
            min(int(value["processed_tokens"]) for value in manifests if value)
            >= checkpoints[-1]
        )
        if not stability["passed"] or already_at_10m:
            for shard in range(shards):
                _run_one_shard_to_target(
                    config,
                    sample,
                    shard=shard,
                    target=checkpoints[-1],
                    identity=identity,
                    model=model,
                    tokenizer=tokenizer,
                )
            stopping = "continued_to_10m"
        else:
            stopping = "stopped_at_5m"
    finally:
        del model
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
    manifests = [
        _validate_primary_shard(root, identity, shard, execution)
        for shard in range(shards)
    ]
    norm_audit, norm_audit_table = _pooled_norm_audit(
        [value for value in manifests if value],
        pilot,
        outlier_sigma=float(_mapping(config, "numerics").get("outlier_sigma", 5.0)),
        compute_dtype=compute_dtype,
        policy=norm_policy,
    )
    norm_audit_path = stage_directory / "norm_audit.parquet"
    _atomic_parquet(norm_audit_path, norm_audit_table)
    if not norm_audit["passed"]:
        _atomic_json(stage_directory / "norm_audit_failure.json", norm_audit)
        raise RuntimeError(
            "最终普通token norm/过滤审计失败；请检查moments/norm_audit.parquet"
        )
    final_target = min(int(value["processed_tokens"]) for value in manifests if value)
    stage_manifest = {
        **identity,
        "schema_version": SHARD_SCHEMA,
        **execution,
        "completed_at": datetime.now().isoformat(timespec="seconds"),
        "shards": shards,
        "minimum_tokens_per_shard": final_target,
        "five_million_local_target": five_million_target,
        "stopping_decision": stopping,
        "stability": stability,
        "norm_audit": norm_audit,
        "norm_audit_sha256": sha256_file(norm_audit_path),
        "labels_or_returns_loaded": False,
    }
    _atomic_json(root / "moments" / "manifest.json", stage_manifest)
    validate_rank_shards_outputs(
        stage_directory,
        identity=identity,
        execution=execution,
        shards=shards,
    )
    return stage_directory


def _merged_primary(
    root: Path,
    identity: Mapping[str, object],
    execution: Mapping[str, object],
    shards: int,
) -> dict[str, OnlineMoments]:
    merged: dict[str, OnlineMoments] = {}
    for shard in range(shards):
        _validate_primary_shard(root, identity, shard, execution)
        streams = load_moment_file(
            root / "moments" / f"shard_{shard:02d}" / "primary_moments.npz"
        )
        merge_moment_maps(merged, streams[PRIMARY_STREAM])
    return merged


def validate_rank_analysis_outputs(
    directory: str | Path, expected: Mapping[str, object] | None = None
) -> dict[str, object]:
    root = Path(directory)
    manifest = _require_exact_manifest(
        root / "manifest.json",
        {"schema_version": ANALYSIS_SCHEMA, **dict(expected or {})},
    )
    for filename in (
        "rank_metrics.parquet",
        "compression_metrics.parquet",
        "subspaces.npz",
    ):
        path = root / filename
        if not path.is_file() or sha256_file(path) != manifest.get("files", {}).get(
            filename
        ):
            raise RuntimeError(f"rank analysis文件缺失或hash不匹配: {filename}")
        public_path = root.parent / filename
        if not public_path.is_file() or not os.path.samefile(path, public_path):
            raise RuntimeError(
                f"rank analysis公开路径缺失或不是canonical hardlink: {filename}"
            )
    metrics = pd.read_parquet(root / "rank_metrics.parquet")
    if not set(ALL_STREAMS).issubset(metrics["stream"].unique()):
        raise ValueError("rank metrics缺少主/辅助stream")
    return manifest


def _rank_record(
    stream: str, site: str, moment: OnlineMoments
) -> tuple[dict[str, object], np.ndarray, np.ndarray]:
    eigenvalues, eigenvectors = covariance_eigendecomposition(moment)
    kind, layer = _site_layer(site)
    record = {
        "stream": stream,
        "site": site,
        "kind": kind,
        "layer": layer,
        "count": int(moment.count),
        **rank_metrics_from_eigenvalues(eigenvalues, dimension=moment.dimension),
    }
    return record, eigenvalues, eigenvectors


def run_rank_analysis_stage(config: Mapping[str, object]) -> Path:
    identity = _identity(config)
    root = _run_directory(config)
    pilot = run_pilot_stage(config)
    execution = _execution_identity(pilot)
    output = root / "analysis"
    expected = {
        "run_fingerprint": identity["run_fingerprint"],
        "execution_fingerprint": execution["execution_fingerprint"],
    }
    if output.exists():
        validate_rank_analysis_outputs(output, expected)
        return output
    run_rank_shards_stage(config)
    shards = int(_mapping(config, "sampling").get("shards", 8))
    primary = _merged_primary(root, identity, execution, shards)
    auxiliary, _ = _load_auxiliary_state(root, identity, execution)
    streams = {PRIMARY_STREAM: primary, **auxiliary}
    records: list[dict[str, object]] = []
    subspaces: dict[str, np.ndarray] = {}
    for stream, site_map in streams.items():
        for site in CORE_SITES:
            record, eigenvalues, eigenvectors = _rank_record(
                stream, site, site_map[site]
            )
            records.append(record)
            prefix = f"{stream}__{site}"
            subspaces[f"{prefix}__mean"] = site_map[site].mean.astype(np.float32)
            subspaces[f"{prefix}__eigenvalues"] = eigenvalues.astype(np.float64)
            subspaces[f"{prefix}__eigenvectors"] = eigenvectors.astype(np.float32)
    metrics = pd.DataFrame(records)
    shard_records = []
    for shard in range(shards):
        site_map = load_moment_file(
            root / "moments" / f"shard_{shard:02d}" / "primary_moments.npz"
        )[PRIMARY_STREAM]
        for site in CORE_SITES:
            record, _, _ = _rank_record(PRIMARY_STREAM, site, site_map[site])
            record["shard"] = shard
            shard_records.append(record)
    shard_metrics = pd.DataFrame(shard_records)
    compression = []
    primary_metrics = metrics[metrics["stream"].eq(PRIMARY_STREAM)].set_index("site")
    for layer in range(1, 13):
        sites = {
            "attention": f"attention_output_{layer:02d}",
            "z": f"z_{layer:02d}",
            "mlp": f"mlp_output_{layer:02d}",
            "residual": f"residual_{layer:02d}",
        }
        cis = {}
        for denominator in ("z", "mlp", "residual"):
            numerator_values = shard_metrics.loc[
                shard_metrics["site"].eq(sites["attention"])
            ].sort_values("shard")["effective_rank"]
            denominator_values = shard_metrics.loc[
                shard_metrics["site"].eq(sites[denominator])
            ].sort_values("shard")["effective_rank"]
            point, lower, upper = paired_bootstrap_log_ratio_ci(
                numerator_values,
                denominator_values,
                samples=int(
                    _mapping(config, "numerics").get("bootstrap_samples", 5000)
                ),
                seed=int(_mapping(config, "numerics").get("bootstrap_seed", 20260825))
                + layer,
            )
            cis[denominator] = (point, lower, upper)
        evidence = collapse_evidence(
            ratio_ci_upper_z=cis["z"][2],
            ratio_ci_upper_mlp=cis["mlp"][2],
            ratio_ci_upper_residual=cis["residual"][2],
            attention_k99=int(primary_metrics.loc[sites["attention"], "k99_variance"]),
            z_k99=int(primary_metrics.loc[sites["z"], "k99_variance"]),
            mlp_k99=int(primary_metrics.loc[sites["mlp"], "k99_variance"]),
            residual_k99=int(primary_metrics.loc[sites["residual"], "k99_variance"]),
        )
        compression.append(
            {
                "layer": layer,
                "ratio_erank_o_to_z": cis["z"][0],
                "ratio_erank_o_to_z_ci_lower": cis["z"][1],
                "ratio_erank_o_to_z_ci_upper": cis["z"][2],
                "ratio_erank_o_to_mlp": cis["mlp"][0],
                "ratio_erank_o_to_mlp_ci_lower": cis["mlp"][1],
                "ratio_erank_o_to_mlp_ci_upper": cis["mlp"][2],
                "ratio_erank_o_to_residual": cis["residual"][0],
                "ratio_erank_o_to_residual_ci_lower": cis["residual"][1],
                "ratio_erank_o_to_residual_ci_upper": cis["residual"][2],
                "evidence": evidence,
            }
        )
    temporary = Path(tempfile.mkdtemp(prefix=".analysis-", dir=root))
    try:
        metrics.to_parquet(
            temporary / "rank_metrics.parquet", index=False, compression="zstd"
        )
        pd.DataFrame(compression).to_parquet(
            temporary / "compression_metrics.parquet", index=False, compression="zstd"
        )
        with (temporary / "subspaces.npz").open("wb") as handle:
            np.savez(handle, **subspaces)
        files = {
            filename: sha256_file(temporary / filename)
            for filename in (
                "rank_metrics.parquet",
                "compression_metrics.parquet",
                "subspaces.npz",
            )
        }
        manifest = {
            **identity,
            "schema_version": ANALYSIS_SCHEMA,
            **execution,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "streams": list(streams),
            "sites": list(CORE_SITES),
            "files": files,
            "labels_or_returns_loaded": False,
        }
        (temporary / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        os.replace(temporary, output)
        for filename in (
            "rank_metrics.parquet",
            "compression_metrics.parquet",
            "subspaces.npz",
        ):
            _atomic_hardlink(output / filename, root / filename)
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    validate_rank_analysis_outputs(output, expected)
    return output


def output_projection_covariance(
    weight: np.ndarray, covariance_z: np.ndarray
) -> np.ndarray:
    w = np.asarray(weight, dtype=np.float64)
    cov = np.asarray(covariance_z, dtype=np.float64)
    if w.ndim != 2 or cov.shape != (w.shape[1], w.shape[1]):
        raise ValueError("W^O或Z covariance shape不匹配")
    result = w @ cov @ w.T
    return (result + result.T) * 0.5


def relative_frobenius_error(actual: np.ndarray, expected: np.ndarray) -> float:
    return float(
        np.linalg.norm(np.asarray(actual) - np.asarray(expected), ord="fro")
        / max(np.linalg.norm(np.asarray(actual), ord="fro"), np.finfo(float).tiny)
    )


def _matrix_rank_metrics(matrix: np.ndarray) -> dict[str, float | int]:
    eigenvalues = np.linalg.eigvalsh((matrix + matrix.T) * 0.5)[::-1]
    return rank_metrics_from_eigenvalues(np.clip(eigenvalues, 0.0, None))


def wo_mechanism_records(
    *,
    layer: int,
    weight: np.ndarray,
    z_moment: OnlineMoments,
    o_moment: OnlineMoments,
    seed: int,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    cov_z = z_moment.covariance()
    cov_o = o_moment.covariance()
    predicted = output_projection_covariance(weight, cov_z)
    error = relative_frobenius_error(cov_o, predicted)
    eigenvalues, eigenvectors = np.linalg.eigh(cov_o)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.clip(eigenvalues[order], 0.0, None)
    eigenvectors = eigenvectors[:, order]
    w_math = np.asarray(weight, dtype=np.float64).T
    records = []
    for component, direction in enumerate(eigenvectors.T, start=1):
        v = w_math @ direction
        scale = float(v @ v)
        z_variance = float(v @ cov_z @ v / max(scale, np.finfo(float).tiny))
        records.append(
            {
                "layer": int(layer),
                "component": int(component),
                "attention_output_eigenvalue": float(eigenvalues[component - 1]),
                "z_directional_variance": z_variance,
                "wo_scale_squared": scale,
                "factor_product": z_variance * scale,
            }
        )
    u, _, vt = np.linalg.svd(w_math, full_matrices=False)
    polar = u @ vt
    norm_scale = np.linalg.norm(w_math, ord="fro") / math.sqrt(w_math.shape[0])
    polar_scaled = polar * norm_scale
    rng = np.random.default_rng(seed)
    random_q, _ = np.linalg.qr(rng.normal(size=w_math.shape))
    random_scaled = random_q * norm_scale
    counterfactuals = {
        "observed_learned_wo": predicted,
        "polar_orthogonal_scaled": polar_scaled.T @ cov_z @ polar_scaled,
        "random_orthogonal_scaled": random_scaled.T @ cov_z @ random_scaled,
        "wo_only_isotropic_input": w_math.T @ w_math,
    }
    summary: dict[str, object] = {
        "layer": int(layer),
        "covariance_relative_frobenius_error": error,
    }
    for name, covariance in counterfactuals.items():
        metrics = _matrix_rank_metrics(covariance)
        summary[f"{name}_effective_rank"] = metrics["effective_rank"]
        summary[f"{name}_normalized_effective_rank"] = metrics[
            "normalized_effective_rank"
        ]
        summary[f"{name}_k99_variance"] = metrics["k99_variance"]
    return records, summary


def validate_wo_mechanism_outputs(
    directory: str | Path, expected: Mapping[str, object] | None = None
) -> dict[str, object]:
    root = Path(directory)
    manifest = _require_exact_manifest(
        root / "manifest.json",
        {"schema_version": MECHANISM_SCHEMA, **dict(expected or {})},
    )
    for filename in ("wo_mechanism.parquet", "wo_summary.parquet"):
        path = root / filename
        if not path.is_file() or sha256_file(path) != manifest.get("files", {}).get(
            filename
        ):
            raise RuntimeError(f"W^O机制文件缺失或hash不匹配: {filename}")
        public_path = root.parent / filename
        if not public_path.is_file() or not os.path.samefile(path, public_path):
            raise RuntimeError(
                f"W^O机制公开路径缺失或不是canonical hardlink: {filename}"
            )
    summary = pd.read_parquet(root / "wo_summary.parquet")
    if len(summary) != 12 or not summary["covariance_identity_passed"].all():
        raise RuntimeError("W^O covariance identity未全部通过")
    return manifest


def run_wo_mechanism_stage(config: Mapping[str, object]) -> Path:
    import torch

    identity = _identity(config)
    root = _run_directory(config)
    pilot = run_pilot_stage(config)
    execution = _execution_identity(pilot)
    output = root / "wo_mechanism"
    expected = {
        "run_fingerprint": identity["run_fingerprint"],
        "execution_fingerprint": execution["execution_fingerprint"],
    }
    if output.exists():
        validate_wo_mechanism_outputs(output, expected)
        return output
    run_rank_analysis_stage(config)
    auxiliary, _ = _load_auxiliary_state(root, identity, execution)
    unfiltered = auxiliary["token_natural_unfiltered"]
    base, checkpoint = _model_paths(config)
    model = build_candidate(
        str(base),
        str(checkpoint),
        pooling="cls",
        pooling_confirmed=False,
        model_hash=identity["checkpoint_sha256"],
        device="cpu",
        dtype=torch.float32,
    )
    freeze_and_validate_inference_model(model, expected_hidden_layers=13)
    records: list[dict[str, object]] = []
    summaries: list[dict[str, object]] = []
    selected_dtype = validate_pilot_outputs(root / "pilot")["selected_compute_dtype"]
    tolerance = 1e-5 if selected_dtype == "float32" else 0.005
    for layer in range(1, 13):
        weight = (
            model.bert.encoder.layer[layer - 1]
            .attention.output.dense.weight.detach()
            .cpu()
            .numpy()
            .astype(np.float64)
        )
        component, summary = wo_mechanism_records(
            layer=layer,
            weight=weight,
            z_moment=unfiltered[f"z_{layer:02d}"],
            o_moment=unfiltered[f"attention_output_{layer:02d}"],
            seed=int(_mapping(config, "numerics").get("bootstrap_seed", 20260825))
            + layer,
        )
        summary["covariance_identity_tolerance"] = tolerance
        summary["covariance_identity_passed"] = bool(
            float(summary["covariance_relative_frobenius_error"]) <= tolerance
        )
        records.extend(component)
        summaries.append(summary)
    summary_frame = pd.DataFrame(summaries)
    if not summary_frame["covariance_identity_passed"].all():
        failures = summary_frame.loc[
            ~summary_frame["covariance_identity_passed"],
            ["layer", "covariance_relative_frobenius_error"],
        ]
        raise RuntimeError(
            "W^O covariance identity失败，拒绝发布机制结果: "
            + failures.to_json(orient="records")
        )
    temporary = Path(tempfile.mkdtemp(prefix=".wo_mechanism-", dir=root))
    try:
        pd.DataFrame(records).to_parquet(
            temporary / "wo_mechanism.parquet", index=False, compression="zstd"
        )
        summary_frame.to_parquet(
            temporary / "wo_summary.parquet", index=False, compression="zstd"
        )
        files = {
            filename: sha256_file(temporary / filename)
            for filename in ("wo_mechanism.parquet", "wo_summary.parquet")
        }
        manifest = {
            **identity,
            "schema_version": MECHANISM_SCHEMA,
            **execution,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "files": files,
            "labels_or_returns_loaded": False,
        }
        (temporary / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        os.replace(temporary, output)
        for filename in ("wo_mechanism.parquet", "wo_summary.parquet"):
            _atomic_hardlink(output / filename, root / filename)
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    validate_wo_mechanism_outputs(output, expected)
    return output


def plot_rank_heatmap(analysis_directory: str | Path):
    import matplotlib.pyplot as plt

    validate_rank_analysis_outputs(analysis_directory)
    metrics = pd.read_parquet(Path(analysis_directory) / "rank_metrics.parquet")
    primary = metrics[
        metrics["stream"].eq(PRIMARY_STREAM) & metrics["layer"].between(1, 12)
    ]
    table = primary.pivot(
        index="kind", columns="layer", values="normalized_effective_rank"
    )
    order = [
        value
        for value in ("z", "attention_output", "mlp_output", "residual")
        if value in table.index
    ]
    table = table.loc[order]
    fig, ax = plt.subplots(figsize=(11, 3.8))
    image = ax.imshow(table.to_numpy(), aspect="auto", vmin=0, vmax=1, cmap="viridis")
    ax.set_xticks(range(len(table.columns)), table.columns)
    ax.set_yticks(range(len(table.index)), table.index)
    ax.set(
        xlabel="Transformer layer", title="Normalized effective rank (ordinary tokens)"
    )
    fig.colorbar(image, ax=ax, label="effective rank / 768")
    fig.tight_layout()
    plt.close(fig)
    return fig


def plot_rank_curves(analysis_directory: str | Path):
    import matplotlib.pyplot as plt

    validate_rank_analysis_outputs(analysis_directory)
    metrics = pd.read_parquet(Path(analysis_directory) / "rank_metrics.parquet")
    primary = metrics[
        metrics["stream"].eq(PRIMARY_STREAM) & metrics["layer"].between(1, 12)
    ]
    fig, ax = plt.subplots(figsize=(9, 5))
    for kind, group in primary.groupby("kind", sort=False):
        ax.plot(
            group["layer"], group["normalized_effective_rank"], marker="o", label=kind
        )
    ax.set(
        xlabel="Transformer layer",
        ylabel="effective rank / 768",
        title="Activation dimensionality across layers",
        xticks=range(1, 13),
        ylim=(0, 1.02),
    )
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    plt.close(fig)
    return fig


def validate_final_outputs(
    directory: str | Path, expected: Mapping[str, object] | None = None
) -> dict[str, object]:
    root = Path(directory)
    manifest = _require_exact_manifest(
        root / "manifest.json",
        {"schema_version": FINAL_SCHEMA, **dict(expected or {})},
    )
    for filename, expected_hash in manifest.get("figures", {}).items():
        path = root / "figures" / filename
        if not path.is_file() or sha256_file(path) != expected_hash:
            raise RuntimeError(f"最终图表缺失或hash不匹配: {filename}")
    analysis = validate_rank_analysis_outputs(
        root / "analysis",
        {"run_fingerprint": manifest["run_fingerprint"]},
    )
    mechanism = validate_wo_mechanism_outputs(
        root / "wo_mechanism",
        {"run_fingerprint": manifest["run_fingerprint"]},
    )
    if sha256_file(root / "analysis" / "manifest.json") != manifest.get(
        "analysis_manifest_sha256"
    ):
        raise RuntimeError("最终manifest引用的analysis manifest hash不匹配")
    if sha256_file(root / "wo_mechanism" / "manifest.json") != manifest.get(
        "mechanism_manifest_sha256"
    ):
        raise RuntimeError("最终manifest引用的mechanism manifest hash不匹配")
    if analysis.get("execution_fingerprint") != mechanism.get("execution_fingerprint"):
        raise RuntimeError("analysis与W^O机制execution fingerprint不一致")
    return manifest


def finalize_activation_rank_run(config: Mapping[str, object]) -> Path:
    identity = _identity(config)
    root = _run_directory(config)
    pilot = run_pilot_stage(config)
    execution = _execution_identity(pilot)
    expected = {
        "run_fingerprint": identity["run_fingerprint"],
        "execution_fingerprint": execution["execution_fingerprint"],
    }
    if (root / "manifest.json").exists():
        validate_final_outputs(root, expected)
        return root
    analysis = run_rank_analysis_stage(config)
    mechanism = run_wo_mechanism_stage(config)
    figures = root / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    heatmap = plot_rank_heatmap(analysis)
    curves = plot_rank_curves(analysis)
    heatmap.savefig(figures / "normalized_effective_rank_heatmap.png", dpi=180)
    curves.savefig(figures / "normalized_effective_rank_curves.png", dpi=180)
    manifest = {
        **identity,
        "schema_version": FINAL_SCHEMA,
        **execution,
        "completed_at": datetime.now().isoformat(timespec="seconds"),
        "analysis_manifest_sha256": sha256_file(analysis / "manifest.json"),
        "mechanism_manifest_sha256": sha256_file(mechanism / "manifest.json"),
        "figures": {
            path.name: sha256_file(path) for path in sorted(figures.glob("*.png"))
        },
        "scope": "descriptive_global_geometry_financial_reports_only",
        "oos_reuse_forbidden": True,
        "labels_or_returns_loaded": False,
        "status": "completed",
    }
    _atomic_json(root / "manifest.json", manifest)
    validate_final_outputs(root, expected)
    return root
