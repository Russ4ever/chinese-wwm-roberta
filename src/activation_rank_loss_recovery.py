"""Held-out, single-position MLM-proxy loss recovery for activation rank.

This is a label-free extension of :mod:`src.activation_rank`.  It deliberately
keeps the intervention site fixed (Layer 6) while comparing attention output,
MLP output, and the residual stream.  The MLM head comes from the base model,
whereas the backbone comes from the fine-tuned checkpoint, so the resulting
cross-entropy is an explicitly named proxy loss rather than the checkpoint's
original supervised objective.
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
    run_rank_analysis_stage,
    run_sample_stage,
    validate_pilot_outputs,
)
from .config import load_yaml_config
from .layer_probe_representations import (
    freeze_and_validate_inference_model,
    git_commit,
    sha256_file,
)
from .models.checkpoint import load_state_dict_safe, strip_prefix


LOSS_RECOVERY_SCHEMA = "activation_rank_loss_recovery_v1.0"
LOSS_SEMANTICS = "mlm_proxy__finetuned_backbone__base_model_mlm_head"
SITE_KINDS = ("attention_output", "mlp_output", "residual")


def load_loss_recovery_policy(path: str | Path) -> dict[str, object]:
    """Load and strictly validate the label-free loss-recovery policy."""

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
    if policy["schema_version"] != "activation_rank_loss_recovery_policy_v1.0":
        raise ValueError("不支持的loss-recovery配置版本")

    serialized = json.dumps(policy, ensure_ascii=False).lower()
    for forbidden in ("sentiment_label", "continuous_target", "return_probe"):
        if forbidden in serialized:
            raise ValueError(f"label-free loss recovery禁止字段: {forbidden}")

    evaluation = _mapping(policy, "evaluation")
    projection = _mapping(policy, "projection")
    identifiability = _mapping(policy, "identifiability")
    if int(evaluation.get("layer", -1)) != 6:
        raise ValueError("主协议固定在Layer 6做单位置干预")
    if int(evaluation.get("report_count", 0)) <= 0:
        raise ValueError("evaluation.report_count必须为正整数")
    seeds = [int(value) for value in evaluation.get("mask_seeds", [])]
    if len(seeds) < 2 or len(set(seeds)) != len(seeds):
        raise ValueError("至少需要两个互异mask seed")
    probability = float(evaluation.get("mask_probability", 0.0))
    if not 0.0 < probability < 1.0:
        raise ValueError("mask_probability必须位于(0, 1)")
    if int(evaluation.get("batch_size", 0)) <= 0:
        raise ValueError("evaluation.batch_size必须为正整数")

    components = [int(value) for value in projection.get("components", [])]
    if components != sorted(set(components)) or not components:
        raise ValueError("projection.components必须严格递增且无重复")
    if components[0] != 0 or components[-1] != 768:
        raise ValueError("projection.components必须同时包含0和768")
    if any(value < 0 or value > 768 for value in components):
        raise ValueError("projection.components越界")

    if float(identifiability.get("minimum_ablation_delta", 0.0)) <= 0:
        raise ValueError("minimum_ablation_delta必须为正数")
    if float(identifiability.get("minimum_signal_to_noise", 0.0)) <= 0:
        raise ValueError("minimum_signal_to_noise必须为正数")
    if int(identifiability.get("bootstrap_samples", 0)) < 200:
        raise ValueError("bootstrap_samples至少为200")
    confidence = float(identifiability.get("confidence_level", 0.0))
    if not 0.5 < confidence < 1.0:
        raise ValueError("confidence_level必须位于(0.5, 1)")
    threshold = float(identifiability.get("sustained_recovery_threshold", 0.0))
    if not 0.0 < threshold <= 1.0:
        raise ValueError("sustained_recovery_threshold必须位于(0, 1]")
    return policy


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


def resolve_projection_module(bert: Any, site: str) -> Any:
    """Resolve one canonical Layer-6 activation site on a BERT backbone."""

    kind, layer_text = site.rsplit("_", 1)
    layer = int(layer_text)
    if layer != 6 or kind not in SITE_KINDS:
        raise ValueError(f"主协议只允许Layer 6三类位置，收到: {site}")
    block = bert.encoder.layer[layer - 1]
    if kind == "attention_output":
        return block.attention.output.dense
    if kind == "mlp_output":
        return block.output.dense
    return block


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


def _pooled_loss(frame: pd.DataFrame) -> float:
    count = float(frame["mask_count"].sum())
    if count <= 0:
        raise ValueError("masked token count必须为正")
    return float(frame["ce_sum"].sum() / count)


def _aggregate_reports(frame: pd.DataFrame) -> pd.DataFrame:
    out = (
        frame.groupby("report_id", sort=False, as_index=False)[["ce_sum", "mask_count"]]
        .sum()
        .sort_values("report_id", kind="mergesort")
        .reset_index(drop=True)
    )
    if (out["mask_count"] <= 0).any():
        raise ValueError("每份研报至少应有一个masked token")
    return out


def _bootstrap_losses(
    frame: pd.DataFrame, indices: np.ndarray, expected_ids: Sequence[str]
) -> np.ndarray:
    aggregated = (
        _aggregate_reports(frame).set_index("report_id").loc[list(expected_ids)]
    )
    sums = aggregated["ce_sum"].to_numpy(dtype=np.float64)
    counts = aggregated["mask_count"].to_numpy(dtype=np.float64)
    return sums[indices].sum(axis=1) / counts[indices].sum(axis=1)


def summarize_loss_recovery(
    report_losses: pd.DataFrame, policy: Mapping[str, object]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Summarize raw CE values without rounding or denominator clipping."""

    required = {
        "condition",
        "site",
        "n_components",
        "mask_seed",
        "report_id",
        "ce_sum",
        "mask_count",
    }
    if not required.issubset(report_losses.columns):
        raise ValueError(
            f"loss表缺字段: {sorted(required.difference(report_losses.columns))}"
        )
    evaluation = _mapping(policy, "evaluation")
    projection = _mapping(policy, "projection")
    ident = _mapping(policy, "identifiability")
    seeds = [int(value) for value in evaluation["mask_seeds"]]
    components = [int(value) for value in projection["components"]]
    baseline = report_losses.loc[report_losses["condition"].eq("original")].copy()
    expected_pairs = {
        (int(seed), str(report_id))
        for seed, report_id in zip(
            baseline["mask_seed"], baseline["report_id"], strict=True
        )
    }
    if not expected_pairs or set(baseline["mask_seed"].astype(int)) != set(seeds):
        raise ValueError("original baseline缺少mask seed")
    if baseline.duplicated(["mask_seed", "report_id"]).any():
        raise ValueError("original baseline含重复seed/report")

    report_ids = sorted(baseline["report_id"].astype(str).unique())
    draws = int(ident["bootstrap_samples"])
    rng = np.random.default_rng(int(ident["bootstrap_seed"]))
    indices = rng.integers(0, len(report_ids), size=(draws, len(report_ids)))
    baseline_loss = _pooled_loss(baseline)
    baseline_boot = _bootstrap_losses(baseline, indices, report_ids)
    alpha = (1.0 - float(ident["confidence_level"])) / 2.0

    metric_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    for kind in SITE_KINDS:
        site = f"{kind}_{int(evaluation['layer']):02d}"
        projected = report_losses.loc[
            report_losses["condition"].eq("projection") & report_losses["site"].eq(site)
        ].copy()
        actual_pairs = {
            (int(seed), str(report_id))
            for seed, report_id in zip(
                projected["mask_seed"], projected["report_id"], strict=True
            )
        }
        if actual_pairs != expected_pairs:
            raise ValueError(f"{site}与original没有使用完全相同的seed/report集合")
        if set(projected["n_components"].astype(int)) != set(components):
            raise ValueError(f"{site}缺少projection components")
        if projected.duplicated(["n_components", "mask_seed", "report_id"]).any():
            raise ValueError(f"{site}含重复component/seed/report")
        baseline_counts = baseline.set_index(["mask_seed", "report_id"])["mask_count"]
        for n_components in components:
            current = projected.loc[projected["n_components"].eq(int(n_components))]
            component_pairs = {
                (int(seed), str(report_id))
                for seed, report_id in zip(
                    current["mask_seed"], current["report_id"], strict=True
                )
            }
            if component_pairs != expected_pairs:
                raise ValueError(
                    f"{site} N={n_components}没有使用完整的seed/report集合"
                )
            current_counts = current.set_index(["mask_seed", "report_id"])[
                "mask_count"
            ].loc[baseline_counts.index]
            if not np.array_equal(
                current_counts.to_numpy(dtype=np.int64),
                baseline_counts.to_numpy(dtype=np.int64),
            ):
                raise ValueError(f"{site} N={n_components}与original的mask不一致")

        zero = projected.loc[projected["n_components"].eq(0)]
        zero_loss = _pooled_loss(zero)
        zero_boot = _bootstrap_losses(zero, indices, report_ids)
        denominator = float(zero_loss - baseline_loss)
        denominator_boot = zero_boot - baseline_boot
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
            current = projected.loc[projected["n_components"].eq(int(n_components))]
            current_loss = _pooled_loss(current)
            current_boot = _bootstrap_losses(current, indices, report_ids)
            if identifiable:
                recovery = float((zero_loss - current_loss) / denominator)
                valid = denominator_boot > 0.0
                if int(valid.sum()) < max(100, draws // 2):
                    raise RuntimeError(f"{site} bootstrap中正分母过少")
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
                    "layer": int(evaluation["layer"]),
                    "n_components": int(n_components),
                    "loss_original": baseline_loss,
                    "loss_zero": zero_loss,
                    "loss_projected": current_loss,
                    "ablation_delta": denominator,
                    "denominator_bootstrap_se": denominator_se,
                    "denominator_signal_to_noise": signal_to_noise,
                    "status": status,
                    "recovery_fraction": recovery,
                    "recovery_ci_lower": float(lower),
                    "recovery_ci_upper": float(upper),
                    "masked_tokens": int(current["mask_count"].sum()),
                    "mask_seed_count": len(seeds),
                    "evaluation_reports": len(report_ids),
                }
            )

        site_metrics = metric_rows[-len(components) :]
        full = next(row for row in site_metrics if row["n_components"] == 768)
        full_difference = abs(float(full["loss_projected"]) - baseline_loss)
        tolerance = float(ident["full_projection_absolute_tolerance"])
        full_passed = bool(full_difference <= tolerance)
        threshold = float(ident["sustained_recovery_threshold"])
        n_sustained: int | None = None
        if identifiable:
            for index, row in enumerate(site_metrics):
                remaining = site_metrics[index:]
                if all(
                    float(candidate["recovery_ci_lower"]) >= threshold
                    for candidate in remaining
                ):
                    n_sustained = int(row["n_components"])
                    break
        summary_rows.append(
            {
                "site": site,
                "kind": kind,
                "layer": int(evaluation["layer"]),
                "status": status,
                "loss_original": baseline_loss,
                "loss_zero": zero_loss,
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
        return_special_tokens_mask=True,
    )
    return dict(encoded)


def _masked_inputs(
    encoded: Mapping[str, Any], tokenizer: Any, *, seed: int, probability: float
) -> tuple[Any, Any, int]:
    import torch

    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    ordinary = encoded["attention_mask"].bool() & ~encoded["special_tokens_mask"].bool()
    mask = (
        torch.rand(ordinary.shape, generator=generator) < float(probability)
    ) & ordinary
    forced = 0
    for row in range(mask.shape[0]):
        if not bool(mask[row].any()):
            candidates = torch.nonzero(ordinary[row], as_tuple=False).flatten()
            if candidates.numel() == 0:
                raise ValueError("研报截断后没有可mask的普通token")
            mask[row, int(candidates[0])] = True
            forced += 1
    masked = encoded["input_ids"].clone()
    masked[mask] = int(tokenizer.mask_token_id)
    return masked, mask, forced


def _evaluate_condition(
    *,
    model: Any,
    encoded: Mapping[str, Any],
    masked_input_ids: Any,
    mask_positions: Any,
    rows: pd.DataFrame,
    device: str,
    batch_size: int,
    mask_seed: int,
    condition: str,
    site: str,
    n_components: int,
    hook_context: Any,
) -> list[dict[str, object]]:
    import torch
    import torch.nn.functional as functional

    records: list[dict[str, object]] = []
    with hook_context, torch.inference_mode():
        for start in range(0, len(rows), int(batch_size)):
            stop = min(start + int(batch_size), len(rows))
            batch_mask = mask_positions[start:stop]
            labels = encoded["input_ids"][start:stop].clone()
            labels[~batch_mask] = -100
            inputs = {
                "input_ids": masked_input_ids[start:stop],
                "attention_mask": encoded["attention_mask"][start:stop],
                "token_type_ids": encoded["token_type_ids"][start:stop],
            }
            if str(device).startswith("cuda"):
                inputs = {
                    key: value.pin_memory().to(device, non_blocking=True)
                    for key, value in inputs.items()
                }
                labels = labels.pin_memory().to(device, non_blocking=True)
            else:
                inputs = {key: value.to(device) for key, value in inputs.items()}
                labels = labels.to(device)
            logits = model(**inputs).logits.float()
            token_losses = functional.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                labels.reshape(-1),
                reduction="none",
                ignore_index=-100,
            ).reshape(labels.shape)
            valid = labels.ne(-100)
            ce_sums = (token_losses * valid).sum(dim=1).detach().cpu().numpy()
            counts = valid.sum(dim=1).detach().cpu().numpy()
            for offset, (ce_sum, count) in enumerate(zip(ce_sums, counts, strict=True)):
                records.append(
                    {
                        "condition": condition,
                        "site": site,
                        "n_components": int(n_components),
                        "mask_seed": int(mask_seed),
                        "report_id": str(rows.iloc[start + offset]["report_id"]),
                        "ce_sum": float(ce_sum),
                        "mask_count": int(count),
                        "ce_loss": float(ce_sum / count),
                    }
                )
            del logits, token_losses
    return records


def _load_mlm_proxy(
    *, base: Path, checkpoint: Path, device: str, dtype_name: str
) -> tuple[Any, Any]:
    import torch
    from transformers import BertForMaskedLM, BertTokenizerFast

    model = BertForMaskedLM.from_pretrained(base, local_files_only=True)
    state = strip_prefix(load_state_dict_safe(str(checkpoint), map_location="cpu"))
    bert_state = {
        key[len("bert.") :]: value
        for key, value in state.items()
        if key.startswith("bert.") and not key.startswith("bert.pooler.")
    }
    model.bert.load_state_dict(bert_state, strict=True)
    dtypes = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    if dtype_name not in dtypes:
        raise ValueError(f"不支持的pilot dtype: {dtype_name}")
    model = model.to(device=device, dtype=dtypes[dtype_name])
    freeze_and_validate_inference_model(model, expected_hidden_layers=13)
    tokenizer = BertTokenizerFast.from_pretrained(base, local_files_only=True)
    return model, tokenizer


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


def _plot_metrics(metrics: pd.DataFrame, summary: pd.DataFrame, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axis = plt.subplots(figsize=(8, 5))
    labels = {
        "attention_output": "Attention output",
        "mlp_output": "MLP output",
        "residual": "Residual stream",
    }
    for row in summary.itertuples(index=False):
        current = metrics.loc[metrics["site"].eq(row.site)].sort_values("n_components")
        if row.status == "identifiable":
            axis.plot(
                current["n_components"],
                current["recovery_fraction"],
                marker="o",
                label=labels[row.kind],
            )
            axis.fill_between(
                current["n_components"].to_numpy(),
                current["recovery_ci_lower"].to_numpy(),
                current["recovery_ci_upper"].to_numpy(),
                alpha=0.15,
            )
        else:
            axis.plot([], [], label=f"{labels[row.kind]} (non-identifiable)")
    axis.axhline(0.99, color="gray", linestyle="--", linewidth=1)
    axis.set_xlabel("Retained PCA components at Layer 6")
    axis.set_ylabel("MLM-proxy loss recovered")
    axis.set_title("Single-position activation loss recovery")
    axis.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


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


def run_loss_recovery_stage(
    config: Mapping[str, object], policy_path: str | Path
) -> Path:
    """Run or strictly reuse the Layer-6 single-position recovery extension."""

    policy_path = Path(policy_path).expanduser().resolve()
    policy = load_loss_recovery_policy(policy_path)
    analysis = run_rank_analysis_stage(config)
    sample_directory = run_sample_stage(config)
    analysis_manifest_path = analysis / "manifest.json"
    analysis_manifest = json.loads(analysis_manifest_path.read_text(encoding="utf-8"))
    expected = {
        "schema_version": LOSS_RECOVERY_SCHEMA,
        "upstream_run_fingerprint": analysis_manifest["run_fingerprint"],
        "upstream_execution_fingerprint": analysis_manifest["execution_fingerprint"],
        "upstream_analysis_manifest_sha256": sha256_file(analysis_manifest_path),
        "policy_sha256": sha256_file(policy_path),
        "loss_recovery_code_sha256": sha256_file(Path(__file__)),
    }
    output_cfg = _mapping(policy, "output")
    relative = Path(str(output_cfg.get("subdirectory", "")))
    if not str(relative) or relative.is_absolute() or ".." in relative.parts:
        raise ValueError("output.subdirectory必须是安全的相对路径")
    output = _run_directory(config) / relative
    if output.exists():
        validate_loss_recovery_outputs(output, expected)
        return output

    preflight = preflight_activation_rank(config)
    pilot = validate_pilot_outputs(run_pilot_stage(config))
    evaluation = _mapping(policy, "evaluation")
    projection = _mapping(policy, "projection")
    layer = int(evaluation["layer"])
    sites = tuple(f"{kind}_{layer:02d}" for kind in SITE_KINDS)
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
    model, tokenizer = _load_mlm_proxy(
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

    loss_rows: list[dict[str, object]] = []
    mask_audit: list[dict[str, object]] = []
    probability = float(evaluation["mask_probability"])
    batch_size = int(evaluation["batch_size"])
    components = [int(value) for value in projection["components"]]
    for seed in [int(value) for value in evaluation["mask_seeds"]]:
        masked, mask_positions, forced = _masked_inputs(
            encoded, tokenizer, seed=seed, probability=probability
        )
        mask_audit.append(
            {
                "mask_seed": seed,
                "masked_tokens": int(mask_positions.sum().item()),
                "forced_mask_reports": int(forced),
                "evaluation_reports": len(selected),
            }
        )
        loss_rows.extend(
            _evaluate_condition(
                model=model,
                encoded=encoded,
                masked_input_ids=masked,
                mask_positions=mask_positions,
                rows=selected,
                device=device,
                batch_size=batch_size,
                mask_seed=seed,
                condition="original",
                site="original",
                n_components=-1,
                hook_context=nullcontext(),
            )
        )
        for site in sites:
            module = resolve_projection_module(model.bert, site)
            for n_components in components:
                loss_rows.extend(
                    _evaluate_condition(
                        model=model,
                        encoded=encoded,
                        masked_input_ids=masked,
                        mask_positions=mask_positions,
                        rows=selected,
                        device=device,
                        batch_size=batch_size,
                        mask_seed=seed,
                        condition="projection",
                        site=site,
                        n_components=n_components,
                        hook_context=SingleSiteProjection(
                            module, bases[site], n_components
                        ),
                    )
                )

    report_losses = pd.DataFrame(loss_rows)
    metrics, site_summary = summarize_loss_recovery(report_losses, policy)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".loss-recovery-", dir=output.parent))
    try:
        evaluation_manifest = selected[
            ["evaluation_row", "report_id", "text_sha256", "evaluation_order"]
        ].copy()
        evaluation_manifest.to_parquet(
            temporary / "evaluation_manifest.parquet", index=False, compression="zstd"
        )
        pd.DataFrame(mask_audit).to_parquet(
            temporary / "mask_audit.parquet", index=False, compression="zstd"
        )
        report_losses.to_parquet(
            temporary / "report_losses.parquet", index=False, compression="zstd"
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
            "mask_audit.parquet",
            "report_losses.parquet",
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
            "intervention_scope": "one_site_at_a_time",
            "layer": layer,
            "sites": list(sites),
            "components": components,
            "device_requested": device,
            "selected_compute_dtype": dtype_name,
            "base_model_source_sha256": _model_source_fingerprint(base),
            "evaluation_reports": len(selected),
            "pca_sample_report_overlap": 0,
            "pca_sample_text_hash_overlap": 0,
            "mask_seeds": [int(value) for value in evaluation["mask_seeds"]],
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
