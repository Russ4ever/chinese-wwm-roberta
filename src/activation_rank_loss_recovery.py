"""Held-out, per-layer CLS representation recovery for activation rank.

This is a label-free extension of :mod:`src.activation_rank`.  It measures how
much each layer's attention output contributes to the final CLS pooled
representation, and how compressible that contribution is.  The metric is the
squared L2 distance between the modified and baseline CLS vectors -- always
non-negative, continuous, and naturally bounded in [0, 1] for recovery fraction.

The model is the fine-tuned ``BinaryClassificationCandidate`` (backbone from
checkpoint, classification head from checkpoint).  The CLS pooled feature is
the same 768-dim vector the ``fc`` head reads.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import tempfile
from contextlib import AbstractContextManager, nullcontext
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .activation_rank import (
    PRIMARY_STREAM,
    _json_hash,
    _mapping,
    _model_paths,
    _read_report_texts,
    _run_directory,
    preflight_activation_rank,
    run_pilot_stage,
    validate_pilot_outputs,
)
from .config import load_yaml_config
from .layer_probe_representations import (
    freeze_and_validate_inference_model,
    git_commit,
    sha256_file,
)
from .models.modeling import build_candidate


LOSS_RECOVERY_SCHEMA = "activation_rank_loss_recovery_v1.1"
LOSS_SEMANTICS = "cls_representation_distance__per_layer"
SITE_KINDS = ("attention_output", "mlp_output", "residual")


# --------------------------------------------------------------------------- #
# Policy
# --------------------------------------------------------------------------- #
def load_loss_recovery_policy(path: str | Path) -> dict[str, object]:
    """Load and strictly validate the label-free representation-recovery policy."""

    policy = load_yaml_config(path)
    required = {
        "schema_version",
        "experiment",
        "evaluation",
        "projection",
        "inference",
        "identifiability",
        "output",
    }
    missing = sorted(required.difference(policy))
    if missing:
        raise ValueError(f"loss-recovery配置缺少顶层字段: {missing}")
    if policy["schema_version"] != "activation_rank_loss_recovery_policy_v1.1":
        raise ValueError("不支持的loss-recovery配置版本")

    serialized = json.dumps(policy, ensure_ascii=False).lower()
    for forbidden in ("sentiment_label", "continuous_target", "return_probe"):
        if forbidden in serialized:
            raise ValueError(f"label-free loss recovery禁止字段: {forbidden}")

    evaluation = _mapping(policy, "evaluation")
    projection = _mapping(policy, "projection")
    identifiability = _mapping(policy, "identifiability")

    layers = [int(v) for v in evaluation.get("layers", [])]
    if not layers or any(l < 1 or l > 12 for l in layers):
        raise ValueError("evaluation.layers必须为1-12的非空列表")
    if len(set(layers)) != len(layers):
        raise ValueError("evaluation.layers不得有重复")

    kinds = [str(v) for v in evaluation.get("kinds", [])]
    if not kinds or any(k not in SITE_KINDS for k in kinds):
        raise ValueError(f"evaluation.kinds必须是{SITE_KINDS}的非空子集")

    if int(evaluation.get("report_count", 0)) <= 0:
        raise ValueError("evaluation.report_count必须为正整数")
    if int(evaluation.get("batch_size", 0)) <= 0:
        raise ValueError("evaluation.batch_size必须为正整数")

    components = [int(value) for value in projection.get("components", [])]
    if components != sorted(set(components)) or not components:
        raise ValueError("projection.components必须严格递增且无重复")
    if components[0] != 0 or components[-1] != 768:
        raise ValueError("projection.components必须同时包含0和768")
    if any(value < 0 or value > 768 for value in components):
        raise ValueError("projection.components越界")

    if float(identifiability.get("minimum_ablation_delta", -1)) < 0:
        raise ValueError("minimum_ablation_delta必须为非负数")
    if float(identifiability.get("minimum_signal_to_noise", 0)) <= 0:
        raise ValueError("minimum_signal_to_noise必须为正数")
    if int(identifiability.get("bootstrap_samples", 0)) < 200:
        raise ValueError("bootstrap_samples至少为200")
    confidence = float(identifiability.get("confidence_level", 0))
    if not 0.5 < confidence < 1.0:
        raise ValueError("confidence_level必须位于(0.5, 1)")
    threshold = float(identifiability.get("sustained_recovery_threshold", 0))
    if not 0.0 < threshold <= 1.0:
        raise ValueError("sustained_recovery_threshold必须位于(0, 1]")
    return policy


# --------------------------------------------------------------------------- #
# Data selection (unchanged from collaborator)
# --------------------------------------------------------------------------- #
def _stable_order(report_id: str, text_sha256: str, seed: str) -> int:
    digest = hashlib.sha256(
        f"{seed}|{report_id}|{text_sha256}".encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


def select_disjoint_evaluation_reports(
    reports: pd.DataFrame,
    rank_sample: pd.DataFrame,
    *,
    report_count: int,
    order_seed: str,
) -> pd.DataFrame:
    """Choose a deterministic evaluation set disjoint by ID and text hash."""

    report_required = {"report_id", "text", "text_sha256"}
    sample_required = {"report_id", "text_sha256"}
    if not report_required.issubset(reports.columns):
        raise ValueError(
            f"研报表缺字段: {sorted(report_required.difference(reports.columns))}"
        )
    if not sample_required.issubset(rank_sample.columns):
        raise ValueError(
            f"rank sample缺字段: {sorted(sample_required.difference(rank_sample.columns))}"
        )
    if int(report_count) <= 0:
        raise ValueError("report_count必须为正整数")

    excluded_ids = set(rank_sample["report_id"].astype(str))
    excluded_hashes = set(rank_sample["text_sha256"].astype(str))
    candidates = reports.loc[
        ~reports["report_id"].astype(str).isin(excluded_ids)
        & ~reports["text_sha256"].astype(str).isin(excluded_hashes)
    ].copy()
    if len(candidates) < int(report_count):
        raise ValueError(
            f"PCA样本外可用研报不足: need={report_count}, available={len(candidates)}"
        )
    candidates["evaluation_order"] = [
        _stable_order(str(report_id), str(text_hash), str(order_seed))
        for report_id, text_hash in zip(
            candidates["report_id"], candidates["text_sha256"], strict=True
        )
    ]
    selected = (
        candidates.sort_values(["evaluation_order", "report_id"], kind="mergesort")
        .iloc[: int(report_count)]
        .reset_index(drop=True)
    )
    selected.insert(0, "evaluation_row", np.arange(len(selected), dtype=np.int64))
    if set(selected["report_id"].astype(str)).intersection(excluded_ids):
        raise RuntimeError("evaluation report_id与PCA样本重叠")
    if set(selected["text_sha256"].astype(str)).intersection(excluded_hashes):
        raise RuntimeError("evaluation text hash与PCA样本重叠")
    return selected


# --------------------------------------------------------------------------- #
# Module resolution (modified: allow layers 1-12)
# --------------------------------------------------------------------------- #
def resolve_projection_module(bert: Any, site: str) -> Any:
    """Resolve one activation site on a BERT backbone (layers 1-12)."""

    kind, layer_text = site.rsplit("_", 1)
    layer = int(layer_text)
    if layer < 1 or layer > 12 or kind not in SITE_KINDS:
        raise ValueError(f"只允许Layer 1-12三类位置，收到: {site}")
    block = bert.encoder.layer[layer - 1]
    if kind == "attention_output":
        return block.attention.output.dense
    if kind == "mlp_output":
        return block.output.dense
    return block


# --------------------------------------------------------------------------- #
# Projection hook (unchanged from collaborator)
# --------------------------------------------------------------------------- #
class SingleSiteProjection(AbstractContextManager["SingleSiteProjection"]):
    """Project exactly one module output and always remove the hook."""

    def __init__(self, module: Any, eigenvectors: Any, n_components: int):
        import torch

        if not 0 <= int(n_components) <= int(eigenvectors.shape[1]):
            raise ValueError("n_components越界")
        self.module = module
        self.n_components = int(n_components)
        self.dimension = int(eigenvectors.shape[0])
        self.basis = torch.as_tensor(
            eigenvectors[:, : self.n_components], dtype=torch.float32
        )
        self.handle: Any | None = None

    def __enter__(self) -> "SingleSiteProjection":
        try:
            parameter = next(self.module.parameters())
        except (AttributeError, StopIteration):
            parameter = None
        if parameter is not None:
            self.basis = self.basis.to(device=parameter.device)

        def hook(_module: Any, _inputs: Any, output: Any) -> Any:
            import torch

            tensor = output[0] if isinstance(output, (tuple, list)) else output
            if tensor.shape[-1] != self.dimension:
                raise ValueError("hook输出hidden size与PCA基不一致")
            if self.n_components == 0:
                projected = torch.zeros_like(tensor)
            else:
                if self.basis.device != tensor.device:
                    raise RuntimeError("PCA基与hook输出不在同一device")
                projected = ((tensor.float() @ self.basis) @ self.basis.T).to(
                    tensor.dtype
                )
            if isinstance(output, tuple):
                return (projected, *output[1:])
            if isinstance(output, list):
                return [projected, *output[1:]]
            return projected

        self.handle = self.module.register_forward_hook(hook)
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        if self.handle is not None:
            self.handle.remove()
            self.handle = None


# --------------------------------------------------------------------------- #
# Representation distance statistics
# --------------------------------------------------------------------------- #
def _mean_distance(frame: pd.DataFrame) -> float:
    return float(frame["representation_distance"].mean())


def _bootstrap_distances(
    frame: pd.DataFrame, indices: np.ndarray, expected_ids: Sequence[str]
) -> np.ndarray:
    """Bootstrap-resample mean representation distance by report_id."""
    per_report = frame.set_index("report_id").loc[list(expected_ids)]
    distances = per_report["representation_distance"].to_numpy(dtype=np.float64)
    return distances[indices].mean(axis=1)


def summarize_loss_recovery(
    report_distances: pd.DataFrame, policy: Mapping[str, object]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Summarize raw CLS distances into recovery fractions + CIs.

    Unlike the MLM-CE version, ``distance_original = 0`` by construction
    (baseline vs itself), so the denominator is simply ``distance_zero``.
    """

    required = {
        "condition",
        "site",
        "layer",
        "kind",
        "n_components",
        "report_id",
        "representation_distance",
    }
    if not required.issubset(report_distances.columns):
        raise ValueError(
            f"distance表缺字段: {sorted(required.difference(report_distances.columns))}"
        )

    evaluation = _mapping(policy, "evaluation")
    projection = _mapping(policy, "projection")
    ident = _mapping(policy, "identifiability")

    layers = [int(v) for v in evaluation["layers"]]
    kinds = [str(v) for v in evaluation["kinds"]]
    components = [int(value) for value in projection["components"]]

    # All report_ids that appear in any condition
    report_ids = sorted(report_distances["report_id"].astype(str).unique())
    draws = int(ident["bootstrap_samples"])
    rng = np.random.default_rng(int(ident["bootstrap_seed"]))
    indices = rng.integers(0, len(report_ids), size=(draws, len(report_ids)))
    alpha = (1.0 - float(ident["confidence_level"])) / 2.0

    # distance_original = 0 by construction
    distance_original = 0.0
    original_boot = np.zeros(draws)

    metric_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []

    for kind in kinds:
        for layer in layers:
            site = f"{kind}_{layer:02d}"
            projected = report_distances.loc[
                report_distances["site"].eq(site)
            ].copy()

            if set(projected["n_components"].astype(int)) != set(components):
                raise ValueError(f"{site}缺少projection components")
            if projected.duplicated(["n_components", "report_id"]).any():
                raise ValueError(f"{site}含重复component/report")

            zero = projected.loc[projected["n_components"].eq(0)]
            distance_zero = _mean_distance(zero)
            zero_boot = _bootstrap_distances(zero, indices, report_ids)

            denominator = float(distance_zero - distance_original)
            denominator_boot = zero_boot - original_boot
            denominator_se = float(np.std(denominator_boot, ddof=1))
            signal_to_noise = (
                math.inf
                if denominator_se == 0.0 and denominator > 0.0
                else denominator / denominator_se if denominator_se > 0.0 else -math.inf
            )
            identifiable = bool(
                denominator >= float(ident["minimum_ablation_delta"])
                and signal_to_noise >= float(ident["minimum_signal_to_noise"])
            )
            status = "identifiable" if identifiable else "non_identifiable"

            for n_components in components:
                current = projected.loc[
                    projected["n_components"].eq(int(n_components))
                ]
                distance_n = _mean_distance(current)
                current_boot = _bootstrap_distances(current, indices, report_ids)

                if identifiable:
                    valid = denominator_boot > 0.0
                    if int(valid.sum()) < max(100, draws // 2):
                        raise RuntimeError(f"{site} bootstrap中正分母过少")
                    recovery = float(
                        (distance_zero - distance_n) / denominator
                    )
                    recovery_boot = (
                        zero_boot[valid] - current_boot[valid]
                    ) / denominator_boot[valid]
                    lower, upper = np.quantile(recovery_boot, [alpha, 1.0 - alpha])
                else:
                    recovery = math.nan
                    lower = math.nan
                    upper = math.nan

                metric_rows.append(
                    {
                        "site": site,
                        "kind": kind,
                        "layer": int(layer),
                        "n_components": int(n_components),
                        "distance_original": distance_original,
                        "distance_zero": distance_zero,
                        "distance_projected": distance_n,
                        "ablation_delta": denominator,
                        "denominator_bootstrap_se": denominator_se,
                        "denominator_signal_to_noise": signal_to_noise,
                        "status": status,
                        "recovery_fraction": recovery,
                        "recovery_ci_lower": float(lower),
                        "recovery_ci_upper": float(upper),
                        "evaluation_reports": len(report_ids),
                    }
                )

            site_metrics = metric_rows[-len(components):]
            full = next(row for row in site_metrics if row["n_components"] == 768)
            full_difference = abs(float(full["distance_projected"]) - distance_original)
            tolerance = float(ident["full_projection_absolute_tolerance"])
            full_passed = bool(full_difference <= tolerance)
            threshold = float(ident["sustained_recovery_threshold"])
            n_sustained: int | None = None
            if identifiable:
                for index, row in enumerate(site_metrics):
                    remaining = site_metrics[index:]
                    if all(
                        float(c["recovery_ci_lower"]) >= threshold
                        for c in remaining
                    ):
                        n_sustained = int(row["n_components"])
                        break
            summary_rows.append(
                {
                    "site": site,
                    "kind": kind,
                    "layer": int(layer),
                    "status": status,
                    "distance_original": distance_original,
                    "distance_zero": distance_zero,
                    "ablation_delta": denominator,
                    "denominator_bootstrap_se": denominator_se,
                    "denominator_signal_to_noise": signal_to_noise,
                    "sustained_recovery_threshold": threshold,
                    "n_sustained_recovery": n_sustained,
                    "full_projection_absolute_difference": full_difference,
                    "full_projection_tolerance": tolerance,
                    "full_projection_passed": full_passed,
                }
            )

    metrics = pd.DataFrame(metric_rows)
    summary = pd.DataFrame(summary_rows)
    summary["n_sustained_recovery"] = summary["n_sustained_recovery"].astype("Int64")
    if not summary["full_projection_passed"].all():
        failed = summary.loc[~summary["full_projection_passed"], "site"].tolist()
        raise RuntimeError(f"N=768全空间投影未通过identity audit: {failed}")
    return metrics, summary


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #
def _tokenize_evaluation(
    tokenizer: Any, rows: pd.DataFrame, max_length: int
) -> dict[str, Any]:
    encoded = tokenizer(
        rows["text"].tolist(),
        padding="max_length",
        truncation=True,
        max_length=int(max_length),
        return_tensors="pt",
        return_attention_mask=True,
        return_token_type_ids=True,
    )
    return dict(encoded)


def _evaluate_representation_condition(
    *,
    candidate: Any,
    encoded: Mapping[str, Any],
    rows: pd.DataFrame,
    device: str,
    batch_size: int,
    condition: str,
    site: str,
    layer: int,
    n_components: int,
    hook_context: Any,
    baseline_cls: Any,
) -> list[dict[str, object]]:
    """Run candidate with projection hook and compute per-report CLS L2 distance."""

    import torch

    records: list[dict[str, object]] = []
    with hook_context, torch.inference_mode():
        for start in range(0, len(rows), int(batch_size)):
            stop = min(start + int(batch_size), len(rows))
            inputs = {
                "input_ids": encoded["input_ids"][start:stop],
                "attention_mask": encoded["attention_mask"][start:stop],
                "token_type_ids": encoded["token_type_ids"][start:stop],
            }
            if str(device).startswith("cuda"):
                inputs = {
                    key: value.pin_memory().to(device, non_blocking=True)
                    for key, value in inputs.items()
                }
            else:
                inputs = {key: value.to(device) for key, value in inputs.items()}
            output = candidate(**inputs)
            modified_cls = output.pooled_feature.float().cpu()
            batch_baseline = baseline_cls[start:stop]
            distances = ((modified_cls - batch_baseline) ** 2).sum(dim=1)
            for offset, dist in enumerate(distances.tolist()):
                records.append(
                    {
                        "condition": condition,
                        "site": site,
                        "layer": int(layer),
                        "kind": site.rsplit("_", 1)[0],
                        "n_components": int(n_components),
                        "report_id": str(rows.iloc[start + offset]["report_id"]),
                        "representation_distance": float(dist),
                    }
                )
            del output, modified_cls
    return records


# --------------------------------------------------------------------------- #
# Model loading (replaces _load_mlm_proxy)
# --------------------------------------------------------------------------- #
def _load_candidate(
    *, base: Path, checkpoint: Path, device: str, dtype_name: str
) -> tuple[Any, Any]:
    """Load the fine-tuned BinaryClassificationCandidate + tokenizer."""

    import torch
    from transformers import BertTokenizerFast

    candidate = build_candidate(
        base_model_dir=str(base),
        checkpoint_path=str(checkpoint),
        device=device,
        dtype=_torch_dtype(dtype_name),
    )
    freeze_and_validate_inference_model(candidate, expected_hidden_layers=13)
    tokenizer = BertTokenizerFast.from_pretrained(base, local_files_only=True)
    return candidate, tokenizer


def _torch_dtype(name: str):
    import torch

    choices = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    normalized = str(name).lower()
    if normalized not in choices:
        raise ValueError(f"不支持的pilot dtype: {name}")
    return choices[normalized]


# --------------------------------------------------------------------------- #
# Fingerprint (unchanged)
# --------------------------------------------------------------------------- #
def _model_source_fingerprint(base: Path) -> str:
    files = sorted(
        {
            path
            for pattern in (
                "config.json",
                "pytorch_model*.bin",
                "model*.safetensors",
                "*.index.json",
            )
            for path in base.glob(pattern)
            if path.is_file()
        }
    )
    if not files:
        raise FileNotFoundError("base model目录缺少可识别的config/weight文件")
    return _json_hash({path.name: sha256_file(path) for path in files})


# --------------------------------------------------------------------------- #
# Plotting (modified: 12 per-layer curves + contribution bar chart)
# --------------------------------------------------------------------------- #
def _plot_metrics(metrics: pd.DataFrame, summary: pd.DataFrame, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax_curves, ax_bar) = plt.subplots(
        1, 2, figsize=(16, 6), gridspec_kw={"width_ratios": [3, 1]}
    )

    layers = sorted(metrics["layer"].unique())
    n_layers = len(layers)
    colors = plt.cm.coolwarm(np.linspace(0, 1, max(n_layers, 1)))

    # Left: recovery curves
    for i, layer in enumerate(layers):
        sub = metrics.loc[
            metrics["layer"].eq(int(layer))
        ].sort_values("n_components")
        if sub.empty:
            continue
        status = (
            summary.loc[summary["layer"].eq(int(layer)), "status"].values[0]
            if not summary.loc[summary["layer"].eq(int(layer))].empty
            else "unknown"
        )
        label = f"L{layer}" if status == "identifiable" else f"L{layer} (non-id.)"
        if status == "identifiable":
            ax_curves.plot(
                sub["n_components"],
                sub["recovery_fraction"],
                marker="o",
                color=colors[i],
                label=label,
                linewidth=1.5,
                markersize=4,
            )
            ax_curves.fill_between(
                sub["n_components"].to_numpy(),
                sub["recovery_ci_lower"].to_numpy(),
                sub["recovery_ci_upper"].to_numpy(),
                color=colors[i],
                alpha=0.1,
            )
        else:
            ax_curves.plot([], [], color=colors[i], label=label)

    ax_curves.axhline(0.99, color="gray", linestyle="--", linewidth=1)
    ax_curves.set_xlabel("Retained PCA components (per layer)")
    ax_curves.set_ylabel("CLS representation recovered")
    ax_curves.set_title("Per-layer attention output recovery")
    ax_curves.legend(fontsize=7, ncol=3, loc="lower right")
    ax_curves.set_ylim(-0.05, 1.05)
    ax_curves.set_xlim(-15, 785)

    # Right: zero-ablation contribution per layer
    bar_data = summary.sort_values("layer")
    ax_bar.barh(
        bar_data["layer"],
        bar_data["ablation_delta"],
        color="steelblue",
        height=0.6,
    )
    ax_bar.set_xlabel("Zero-ablation CLS distance")
    ax_bar.set_ylabel("Layer")
    ax_bar.set_title("Per-layer attention contribution")
    ax_bar.invert_yaxis()

    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Validation (unchanged)
# --------------------------------------------------------------------------- #
def validate_loss_recovery_outputs(
    directory: str | Path, expected: Mapping[str, object]
) -> dict[str, object]:
    root = Path(directory)
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"loss-recovery manifest不存在: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    mismatches = {
        key: {"actual": manifest.get(key), "expected": value}
        for key, value in expected.items()
        if manifest.get(key) != value
    }
    if mismatches:
        raise RuntimeError(
            "loss-recovery产物指纹不匹配: " + json.dumps(mismatches, ensure_ascii=False)
        )
    for filename, expected_hash in manifest.get("file_sha256", {}).items():
        path = root / filename
        if not path.is_file() or sha256_file(path) != expected_hash:
            raise RuntimeError(f"loss-recovery文件缺失或hash不匹配: {filename}")
    return manifest


# --------------------------------------------------------------------------- #
# Main stage (modified: per-layer + CLS distance)
# --------------------------------------------------------------------------- #
def run_loss_recovery_stage(
    config: Mapping[str, object], policy_path: str | Path
) -> Path:
    """Run or strictly reuse the per-layer CLS representation recovery extension."""

    import torch

    policy_path = Path(policy_path).expanduser().resolve()
    policy = load_loss_recovery_policy(policy_path)
    run_dir = _run_directory(config)
    analysis = run_dir / "analysis"
    sample_directory = run_dir / "sample"
    analysis_manifest_path = analysis / "manifest.json"
    if not analysis_manifest_path.is_file():
        raise RuntimeError(
            "上游 analysis 产物不存在, 请先执行 notebook cells 1-6 生成 rank analysis"
        )
    if not (sample_directory / "sample_manifest.parquet").is_file():
        raise RuntimeError(
            "上游 sample 产物不存在, 请先执行 notebook cells 1-3 生成 sample manifest"
        )
    analysis_manifest = json.loads(analysis_manifest_path.read_text(encoding="utf-8"))
    expected = {
        "schema_version": LOSS_RECOVERY_SCHEMA,
        "upstream_run_fingerprint": analysis_manifest.get("run_fingerprint", ""),
        "upstream_execution_fingerprint": analysis_manifest.get("execution_fingerprint", ""),
        "upstream_analysis_manifest_sha256": sha256_file(analysis_manifest_path),
        "policy_sha256": sha256_file(policy_path),
        "loss_recovery_code_sha256": sha256_file(Path(__file__)),
    }
    output_cfg = _mapping(policy, "output")
    relative = Path(str(output_cfg.get("subdirectory", "")))
    if not str(relative) or relative.is_absolute() or ".." in relative.parts:
        raise ValueError("output.subdirectory必须是安全的相对路径")
    output = run_dir / relative
    if output.exists():
        validate_loss_recovery_outputs(output, expected)
        return output

    preflight = preflight_activation_rank(config)
    pilot = validate_pilot_outputs(run_pilot_stage(config))
    evaluation = _mapping(policy, "evaluation")
    projection = _mapping(policy, "projection")
    layers = [int(v) for v in evaluation["layers"]]
    kinds = [str(v) for v in evaluation["kinds"]]
    sites = tuple(f"{kind}_{layer:02d}" for kind in kinds for layer in layers)

    reports = _read_report_texts(config)
    rank_sample = pd.read_parquet(sample_directory / "sample_manifest.parquet")
    selected = select_disjoint_evaluation_reports(
        reports,
        rank_sample,
        report_count=int(evaluation["report_count"]),
        order_seed=str(evaluation["order_seed"]),
    )

    base, checkpoint = _model_paths(config)
    device = str(_mapping(config, "model").get("device", "cuda:1"))
    dtype_name = str(pilot["selected_compute_dtype"])
    candidate, tokenizer = _load_candidate(
        base=base,
        checkpoint=checkpoint,
        device=device,
        dtype_name=dtype_name,
    )
    encoded = _tokenize_evaluation(
        tokenizer, selected, int(_mapping(config, "model").get("max_length", 512))
    )

    with np.load(analysis / "subspaces.npz", allow_pickle=False) as archive:
        bases = {
            site: np.asarray(
                archive[f"{PRIMARY_STREAM}__{site}__eigenvectors"], dtype=np.float32
            )
            for site in sites
        }

    batch_size = int(evaluation["batch_size"])
    components = [int(value) for value in projection["components"]]

    # --- Compute baseline CLS (no hook) ---
    baseline_parts: list[Any] = []
    with torch.inference_mode():
        for start in range(0, len(selected), batch_size):
            stop = min(start + batch_size, len(selected))
            inputs = {
                "input_ids": encoded["input_ids"][start:stop],
                "attention_mask": encoded["attention_mask"][start:stop],
                "token_type_ids": encoded["token_type_ids"][start:stop],
            }
            if str(device).startswith("cuda"):
                inputs = {
                    key: value.pin_memory().to(device, non_blocking=True)
                    for key, value in inputs.items()
                }
            output = candidate(**inputs)
            baseline_parts.append(output.pooled_feature.float().cpu())
            del output
    baseline_cls = torch.cat(baseline_parts, dim=0)

    # --- Per-layer per-kind per-N evaluation ---
    distance_rows: list[dict[str, object]] = []
    for kind in kinds:
        for layer in layers:
            site = f"{kind}_{layer:02d}"
            module = resolve_projection_module(candidate.bert, site)
            for n_components in components:
                hook = (
                    SingleSiteProjection(module, bases[site], n_components)
                    if n_components < 768
                    else nullcontext()
                )
                distance_rows.extend(
                    _evaluate_representation_condition(
                        candidate=candidate,
                        encoded=encoded,
                        rows=selected,
                        device=device,
                        batch_size=batch_size,
                        condition="projection",
                        site=site,
                        layer=layer,
                        n_components=n_components,
                        hook_context=hook,
                        baseline_cls=baseline_cls,
                    )
                )

    report_distances = pd.DataFrame(distance_rows)
    metrics, site_summary = summarize_loss_recovery(report_distances, policy)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".loss-recovery-", dir=output.parent))
    try:
        evaluation_manifest = selected[
            ["evaluation_row", "report_id", "text_sha256", "evaluation_order"]
        ].copy()
        evaluation_manifest.to_parquet(
            temporary / "evaluation_manifest.parquet", index=False, compression="zstd"
        )
        report_distances.to_parquet(
            temporary / "report_distances.parquet", index=False, compression="zstd"
        )
        metrics.to_parquet(
            temporary / "loss_recovery_metrics.parquet", index=False, compression="zstd"
        )
        site_summary.to_parquet(
            temporary / "site_summary.parquet", index=False, compression="zstd"
        )
        preflight.to_parquet(
            temporary / "preflight.parquet", index=False, compression="zstd"
        )
        _plot_metrics(metrics, site_summary, temporary / "loss_recovery.png")
        filenames = (
            "evaluation_manifest.parquet",
            "report_distances.parquet",
            "loss_recovery_metrics.parquet",
            "site_summary.parquet",
            "preflight.parquet",
            "loss_recovery.png",
        )
        manifest = {
            **expected,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "git_commit": git_commit(),
            "loss_semantics": LOSS_SEMANTICS,
            "labels_or_returns_loaded": False,
            "intervention_scope": "per_layer_single_site",
            "layers": layers,
            "kinds": kinds,
            "sites": list(sites),
            "components": components,
            "device_requested": device,
            "selected_compute_dtype": dtype_name,
            "base_model_source_sha256": _model_source_fingerprint(base),
            "evaluation_reports": len(selected),
            "pca_sample_report_overlap": 0,
            "pca_sample_text_hash_overlap": 0,
            "batch_size": batch_size,
            "site_status": dict(zip(site_summary["site"], site_summary["status"])),
            "file_sha256": {
                filename: sha256_file(temporary / filename) for filename in filenames
            },
        }
        (temporary / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        os.replace(temporary, output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return output
