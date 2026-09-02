"""Attention output top-K write-space coverage analysis.

Label-free, GPU-free geometric extension of activation-rank.  Computes
pairwise subspace overlap, cumulative coverage spectrum, and identifies
low-coverage directions across Layer 1-12 attention outputs.

All inputs are canonical SVD results from the upstream analysis stage;
no model loading, no labels, no GPU.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from .activation_rank import PRIMARY_STREAM, _json_hash, _mapping, _run_directory
from .config import load_yaml_config
from .layer_probe_representations import git_commit, sha256_file


COVERAGE_SCHEMA = "activation_subspace_coverage_v1.0"
HIDDEN_DIM = 768


# --------------------------------------------------------------------------- #
# Policy
# --------------------------------------------------------------------------- #
def load_coverage_policy(path: str | Path) -> dict[str, object]:
    """Load and validate the coverage policy YAML."""

    policy = load_yaml_config(path)
    required = {"schema_version", "experiment", "input", "parameters", "output"}
    missing = sorted(required.difference(policy))
    if missing:
        raise ValueError(f"coverage配置缺少顶层字段: {missing}")
    if policy["schema_version"] != "activation_subspace_coverage_policy_v1.0":
        raise ValueError("不支持的coverage配置版本")

    serialized = json.dumps(policy, ensure_ascii=False).lower()
    for forbidden in ("label", "return", "split", "exposure"):
        if forbidden in serialized:
            raise ValueError(f"label-free coverage禁止字段: {forbidden}")

    params = _mapping(policy, "parameters")
    k_grid = [int(v) for v in params.get("k_grid", [])]
    if not k_grid or any(k < 1 or k > HIDDEN_DIM for k in k_grid):
        raise ValueError(f"k_grid必须为1-{HIDDEN_DIM}的非空列表")
    if len(set(k_grid)) != len(k_grid):
        raise ValueError("k_grid不得有重复")
    if int(params.get("primary_k", 0)) not in k_grid:
        raise ValueError("primary_k必须在k_grid中")
    if int(params.get("n_layers", 0)) < 1:
        raise ValueError("n_layers必须为正整数")

    tol = _mapping(params, "tolerances")
    for key in ("orthogonal", "symmetry", "projector", "trace", "psd"):
        if float(tol.get(key, -1)) < 0:
            raise ValueError(f"tolerance {key}必须为非负数")

    return policy


# --------------------------------------------------------------------------- #
# Core linear algebra
# --------------------------------------------------------------------------- #
def load_layer_bases(
    archive: np.lib.npyio.NpzFile, layers: list[int], k: int
) -> list[np.ndarray]:
    """Load top-k eigenvector bases for each layer's attention_output site."""
    bases = []
    for layer in layers:
        key = f"{PRIMARY_STREAM}__attention_output_{layer:02d}__eigenvectors"
        if key not in archive:
            raise KeyError(f"subspaces.npz缺少: {key}")
        V = np.asarray(archive[key], dtype=np.float64)[:, :k]
        # numerical orthonormality check
        err = float(np.max(np.abs(V.T @ V - np.eye(k))))
        bases.append(V)
    return bases


def compute_projector(V: np.ndarray) -> np.ndarray:
    """P = V V^T  (768 x 768)."""
    return V @ V.T


def pairwise_overlap(V_i: np.ndarray, V_j: np.ndarray, k: int) -> dict[str, float]:
    """Overlap and principal-angle statistics for two k-dim subspaces."""
    cross = V_i.T @ V_j                                   # [k, k]
    singular = np.linalg.svd(cross, compute_uv=False)     # [k]
    cos2 = singular ** 2                                   # [k]
    overlap = float(cos2.sum()) / k
    random_baseline = k / HIDDEN_DIM
    return {
        "overlap": overlap,
        "random_overlap_baseline": random_baseline,
        "excess_overlap": overlap - random_baseline,
        "principal_cos2_mean": float(cos2.mean()),
        "principal_cos2_min": float(cos2.min()),
        "principal_cos2_max": float(cos2.max()),
    }


def coverage_spectrum(projectors: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Eigen-decompose average projector C = mean(P_l).

    Returns (eigenvalues_desc, eigenvectors_desc) sorted descending.
    """
    C = np.mean(projectors, axis=0)                        # [768, 768]
    # symmetrise to kill numerical asymmetry
    C = (C + C.T) / 2.0
    eigvals, eigvecs = np.linalg.eigh(C)
    order = np.argsort(eigvals)[::-1]
    return eigvals[order], eigvecs[:, order]


def effective_rank(eigvals: np.ndarray) -> float:
    """Entropy effective rank: exp(-sum p log p)."""
    total = float(eigvals.sum())
    if total <= 0:
        return 0.0
    p = eigvals / total
    p = p[p > 0]
    return float(np.exp(-(p * np.log(p)).sum()))


def coverage_summary_row(eigvals: np.ndarray) -> dict[str, float | int]:
    """Summarise a coverage eigenvalue spectrum."""
    return {
        "effective_coverage_rank": round(effective_rank(eigvals), 2),
        "directions_coverage_below_005": int((eigvals < 0.05).sum()),
        "directions_coverage_below_010": int((eigvals < 0.10).sum()),
        "directions_coverage_above_050": int((eigvals >= 0.50).sum()),
        "directions_coverage_above_090": int((eigvals >= 0.90).sum()),
        "coverage_eigenvalue_min": round(float(eigvals.min()), 6),
        "coverage_eigenvalue_max": round(float(eigvals.max()), 6),
        "coverage_eigenvalue_mean": round(float(eigvals.mean()), 6),
    }


def numerical_invariants(
    V: np.ndarray, P: np.ndarray, k: int, tol: Mapping[str, float]
) -> dict[str, float | bool]:
    """Verify projector properties."""
    ortho_err = float(np.max(np.abs(V.T @ V - np.eye(k))))
    symm_err = float(np.max(np.abs(P - P.T)))
    idem_err = float(np.max(np.abs(P @ P - P)))
    trace_err = float(abs(np.trace(P) - k))
    return {
        "orthogonal_error": ortho_err,
        "symmetry_error": symm_err,
        "projector_idempotency_error": idem_err,
        "trace_error": trace_err,
        "orthogonal_passed": ortho_err <= tol["orthogonal"],
        "symmetry_passed": symm_err <= tol["symmetry"],
        "projector_passed": idem_err <= tol["projector"],
        "trace_passed": trace_err <= tol["trace"],
    }


# --------------------------------------------------------------------------- #
# Plotting
# --------------------------------------------------------------------------- #
def plot_results(
    pairwise_df: pd.DataFrame,
    cumulative_df: pd.DataFrame,
    coverage_eigvals_df: pd.DataFrame,
    primary_k: int,
    output_dir: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    k_values = sorted(pairwise_df["k"].unique())

    # --- Figure 1: pairwise excess overlap heatmaps ---
    n_k = len(k_values)
    n_cols = min(3, n_k)
    n_rows = (n_k + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4.5 * n_rows))
    if n_k == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)

    for idx, k in enumerate(k_values):
        row, col = divmod(idx, n_cols)
        ax = axes[row][col]
        sub = pairwise_df[pairwise_df["k"] == k].pivot(
            index="layer_i", columns="layer_j", values="excess_overlap"
        )
        im = ax.imshow(sub.values, cmap="RdBu_r", vmin=-0.3, vmax=0.3, aspect="auto")
        ax.set_title(f"K={k} (random={k / HIDDEN_DIM:.3f})", fontsize=10)
        ax.set_xlabel("Layer j")
        ax.set_ylabel("Layer i")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    for idx in range(len(k_values), n_rows * n_cols):
        row, col = divmod(idx, n_cols)
        axes[row][col].set_visible(False)

    fig.tight_layout()
    fig.savefig(output_dir / "pairwise_overlap_heatmap.png", dpi=180)
    plt.close(fig)

    # --- Figure 2: cumulative coverage ---
    fig, ax = plt.subplots(figsize=(8, 5))
    for k in k_values:
        sub = cumulative_df[cumulative_df["k"] == k].sort_values("prefix_layers")
        ax.plot(sub["prefix_layers"], sub["effective_coverage_rank"] / HIDDEN_DIM,
                "o-", label=f"K={k}", linewidth=1.5, markersize=4)
    ax.set_xlabel("Cumulative layers (prefix)")
    ax.set_ylabel("Effective coverage rank / 768")
    ax.set_title("Cumulative coverage growth")
    ax.legend(fontsize=8)
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    fig.savefig(output_dir / "cumulative_coverage.png", dpi=180)
    plt.close(fig)

    # --- Figure 3: coverage spectrum ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    # left: primary K
    ax = axes[0]
    sub = coverage_eigvals_df[coverage_eigvals_df["k"] == primary_k].sort_values("component")
    ax.plot(sub["component"], sub["coverage_eigenvalue"], linewidth=1.5)
    ax.axhline(0.05, color="red", linestyle="--", alpha=0.3, label="0.05")
    ax.axhline(0.50, color="orange", linestyle="--", alpha=0.3, label="0.50")
    ax.axhline(0.90, color="green", linestyle="--", alpha=0.3, label="0.90")
    ax.axhline(primary_k / HIDDEN_DIM, color="gray", linestyle=":", alpha=0.5, label=f"random={primary_k / HIDDEN_DIM:.3f}")
    ax.set_xlabel("Coverage component")
    ax.set_ylabel("Coverage eigenvalue")
    ax.set_title(f"Coverage spectrum (K={primary_k})")
    ax.legend(fontsize=8)

    # right: all K overlaid
    ax = axes[1]
    for k in k_values:
        sub = coverage_eigvals_df[coverage_eigvals_df["k"] == k].sort_values("component")
        ax.plot(sub["component"], sub["coverage_eigenvalue"], linewidth=0.8, alpha=0.7, label=f"K={k}")
    ax.set_xlabel("Coverage component")
    ax.set_ylabel("Coverage eigenvalue")
    ax.set_title("Coverage spectrum (all K)")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(output_dir / "coverage_spectrum.png", dpi=180)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #
def validate_coverage_outputs(
    directory: str | Path, expected: Mapping[str, object]
) -> dict[str, object]:
    root = Path(directory)
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"coverage manifest不存在: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    mismatches = {
        key: {"actual": manifest.get(key), "expected": value}
        for key, value in expected.items()
        if manifest.get(key) != value
    }
    if mismatches:
        raise RuntimeError(
            "coverage产物指纹不匹配: " + json.dumps(mismatches, ensure_ascii=False)
        )
    for filename, expected_hash in manifest.get("file_sha256", {}).items():
        path = root / filename
        if not path.is_file() or sha256_file(path) != expected_hash:
            raise RuntimeError(f"coverage文件缺失或hash不匹配: {filename}")
    return manifest


# --------------------------------------------------------------------------- #
# Main stage
# --------------------------------------------------------------------------- #
def run_coverage_stage(
    config: Mapping[str, object], policy_path: str | Path
) -> Path:
    """Run or strictly reuse the attention subspace coverage extension."""

    policy_path = Path(policy_path).expanduser().resolve()
    policy = load_coverage_policy(policy_path)
    run_dir = _run_directory(config)
    analysis_dir = run_dir / "analysis"
    analysis_manifest_path = analysis_dir / "manifest.json"
    if not analysis_manifest_path.is_file():
        raise RuntimeError("上游 analysis 产物不存在")
    analysis_manifest = json.loads(analysis_manifest_path.read_text(encoding="utf-8"))

    params = _mapping(policy, "parameters")
    k_grid = [int(v) for v in params["k_grid"]]
    primary_k = int(params["primary_k"])
    n_layers = int(params["n_layers"])
    layers = list(range(1, n_layers + 1))
    tol = _mapping(params, "tolerances")
    tol = {k: float(v) for k, v in tol.items()}

    expected = {
        "schema_version": COVERAGE_SCHEMA,
        "upstream_run_fingerprint": analysis_manifest.get("run_fingerprint", ""),
        "upstream_analysis_manifest_sha256": sha256_file(analysis_manifest_path),
        "policy_sha256": sha256_file(policy_path),
        "coverage_code_sha256": sha256_file(Path(__file__)),
    }

    output_cfg = _mapping(policy, "output")
    relative = Path(str(output_cfg.get("subdirectory", "")))
    if not str(relative) or relative.is_absolute() or ".." in relative.parts:
        raise ValueError("output.subdirectory必须是安全的相对路径")
    output = run_dir / relative
    if output.exists():
        validate_coverage_outputs(output, expected)
        return output

    # --- Load eigenvectors ---
    with np.load(analysis_dir / "subspaces.npz", allow_pickle=False) as archive:
        all_bases: dict[int, list[np.ndarray]] = {}
        for k in k_grid:
            all_bases[k] = load_layer_bases(archive, layers, k)

    # --- Per-K computation ---
    all_pairwise: list[dict[str, object]] = []
    all_cumulative: list[dict[str, object]] = []
    all_eigvals: list[dict[str, object]] = []
    all_summary: list[dict[str, object]] = []
    all_invariants: list[dict[str, object]] = []
    low_coverage_store: dict[str, np.ndarray] = {}

    for k in k_grid:
        bases = all_bases[k]
        projectors = [compute_projector(V) for V in bases]

        # numerical invariants (layer 1 as representative)
        inv = numerical_invariants(bases[0], projectors[0], k, tol)
        inv["k"] = k
        all_invariants.append(inv)

        # pairwise overlap
        for i in range(n_layers):
            for j in range(n_layers):
                stats = pairwise_overlap(bases[i], bases[j], k)
                stats["k"] = k
                stats["layer_i"] = layers[i]
                stats["layer_j"] = layers[j]
                stats["layer_distance"] = abs(layers[i] - layers[j])
                all_pairwise.append(stats)

        # cumulative coverage
        for m in range(1, n_layers + 1):
            C_m = np.mean(projectors[:m], axis=0)
            C_m = (C_m + C_m.T) / 2.0
            eigvals_m = np.linalg.eigvalsh(C_m)[::-1]
            summary = coverage_summary_row(eigvals_m)
            summary["k"] = k
            summary["prefix_layers"] = m
            all_cumulative.append(summary)

        # all-layer coverage spectrum
        eigvals_all, eigvecs_all = coverage_spectrum(projectors)
        for r in range(HIDDEN_DIM):
            all_eigvals.append({
                "k": k,
                "component": r + 1,
                "coverage_eigenvalue": round(float(eigvals_all[r]), 6),
                "cumulative_share": round(float(eigvals_all[:r + 1].sum() / eigvals_all.sum()), 6),
            })
        summary_all = coverage_summary_row(eigvals_all)
        summary_all["k"] = k
        summary_all["prefix_layers"] = n_layers
        all_summary.append(summary_all)

        # save low/high coverage directions for primary K
        if k == primary_k:
            low_idx = np.argsort(eigvals_all)[:128]
            low_coverage_store["low_coverage_vectors"] = eigvecs_all[:, low_idx].astype(np.float32)
            low_coverage_store["low_coverage_eigenvalues"] = eigvals_all[low_idx].astype(np.float64)
            high_idx = np.argsort(eigvals_all)[-128:]
            low_coverage_store["high_coverage_vectors"] = eigvecs_all[:, high_idx].astype(np.float32)
            low_coverage_store["high_coverage_eigenvalues"] = eigvals_all[high_idx].astype(np.float64)

    pairwise_df = pd.DataFrame(all_pairwise)
    cumulative_df = pd.DataFrame(all_cumulative)
    eigvals_df = pd.DataFrame(all_eigvals)
    summary_df = pd.DataFrame(all_summary)
    invariants_df = pd.DataFrame(all_invariants)

    # --- Save outputs ---
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".coverage-", dir=output.parent))
    try:
        pairwise_df.to_parquet(temporary / "pairwise_overlap.parquet", index=False, compression="zstd")
        cumulative_df.to_parquet(temporary / "cumulative_coverage.parquet", index=False, compression="zstd")
        eigvals_df.to_parquet(temporary / "coverage_eigenvalues.parquet", index=False, compression="zstd")
        summary_df.to_parquet(temporary / "coverage_summary.parquet", index=False, compression="zstd")
        invariants_df.to_parquet(temporary / "numerical_invariants.parquet", index=False, compression="zstd")
        np.savez(temporary / "low_coverage_subspace.npz", **low_coverage_store)
        plot_results(pairwise_df, cumulative_df, eigvals_df, primary_k, temporary)

        filenames = (
            "pairwise_overlap.parquet",
            "cumulative_coverage.parquet",
            "coverage_eigenvalues.parquet",
            "coverage_summary.parquet",
            "numerical_invariants.parquet",
            "low_coverage_subspace.npz",
            "pairwise_overlap_heatmap.png",
            "cumulative_coverage.png",
            "coverage_spectrum.png",
        )
        manifest = {
            **expected,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "git_commit": git_commit(),
            "labels_or_returns_loaded": False,
            "scope": "descriptive_attention_write_space_geometry",
            "stream": PRIMARY_STREAM,
            "site_kind": "attention_output",
            "layers": layers,
            "k_grid": k_grid,
            "primary_k": primary_k,
            "hidden_dimension": HIDDEN_DIM,
            "random_overlap_formula": "K / 768",
            "tolerances": tol,
            "file_sha256": {
                fn: sha256_file(temporary / fn) for fn in filenames
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
