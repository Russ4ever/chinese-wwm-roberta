"""谱衰减分析 (Figure 3a) + 激活消融损失恢复 (Figure 3b)。

复现论文 2508.16929v4 Section 4.3 的两条证据线：
  - Path C: 奇异值谱衰减 + 有效维度对比 (已有数据, 纯分析)
  - Path A: 激活消融 → SVD 分量逐步恢复 → 下游损失恢复曲线

用法:
  python scripts/spectral_loss_recovery.py [--device cuda:0] [--n-texts 500]

输出:
  artifacts/checkpoint_activation_rank/runs/financial_reports_v2/figures/
    spectral_analysis.png        — Figure 3a 复现
    loss_recovery.png             — Figure 3b 复现
  artifacts/checkpoint_activation_rank/runs/financial_reports_v2/analysis/
    loss_recovery_metrics.parquet  — 数值结果
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# ---------------------------------------------------------------------------
# 路径与配置
# ---------------------------------------------------------------------------
ROOT = Path("/home/intern_fjq_2026/Projects/chinese-wwm-roberta")
sys.path.insert(0, str(ROOT))

from src.activation_rank import load_activation_rank_config  # noqa: E402

CONFIG = load_activation_rank_config(ROOT / "configs" / "activation_rank.yaml")
RUN_DIR = Path(CONFIG["output"]["run_directory"]).expanduser().resolve()
ANALYSIS_DIR = RUN_DIR / "analysis"
FIGURES_DIR = RUN_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

STREAM = "token_natural_filtered"
TARGET_LAYER = 6          # 中间层 (0-indexed: layer 5)
N_TEXTS_DEFAULT = 500


# ---------------------------------------------------------------------------
# Path C: 谱衰减与有效维度对比 (Figure 3a)
# ---------------------------------------------------------------------------
def path_c_spectral_analysis() -> pd.DataFrame:
    """从已有 rank_metrics + subspaces 提取指标并画图。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    print("=" * 60)
    print("Path C: 谱衰减与有效维度对比")
    print("=" * 60)

    # --- 汇总表 ---
    rm = pd.read_parquet(ANALYSIS_DIR / "rank_metrics.parquet")
    rm = rm[rm["stream"] == STREAM].copy()
    # 排除 residual_00 (embedding 层, layer=0)
    rm_layers = rm[rm["layer"] > 0]

    summary = rm_layers.groupby("kind").agg(
        effective_rank=("effective_rank", "mean"),
        normalized_erank=("normalized_effective_rank", "mean"),
        k90=("k90_variance", "mean"),
        k95=("k95_variance", "mean"),
        k99=("k99_variance", "mean"),
        dirs_above_1pct=("directions_above_1pct_max", "mean"),
        dimension=("dimension", "first"),
    ).round(1)
    print("\n有效维度对比 (layer 1-12 均值):")
    print(summary.to_string())
    summary.to_parquet(ANALYSIS_DIR / "spectral_summary.parquet")

    # --- 加载 subspaces ---
    sub = np.load(ANALYSIS_DIR / "subspaces.npz", allow_pickle=False)

    sites = {
        "Attention Output": f"{STREAM}__attention_output_{TARGET_LAYER:02d}",
        "MLP Output": f"{STREAM}__mlp_output_{TARGET_LAYER:02d}",
        "Residual Stream": f"{STREAM}__residual_{TARGET_LAYER:02d}",
    }

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 左: 奇异值谱 (Figure 3a)
    ax = axes[0]
    for label, prefix in sites.items():
        eigvals = sub[f"{prefix}__eigenvalues"]
        sv = np.sqrt(np.maximum(eigvals, 0))
        rel_sv = sv / sv.max()
        ax.semilogy(range(len(rel_sv)), rel_sv, label=label, linewidth=2)
    ax.set_xlabel("Singular Value Rank Index")
    ax.set_ylabel("Relative Singular Value")
    ax.set_title(f"Singular Value Spectra (Layer {TARGET_LAYER})")
    ax.legend()
    ax.set_ylim(1e-5, 2)

    # 中: 各层有效秩
    ax = axes[1]
    colors = {"attention_output": "tab:blue", "mlp_output": "tab:orange", "residual": "tab:green"}
    for kind in ["attention_output", "mlp_output", "residual"]:
        sub_df = rm_layers[rm_layers["kind"] == kind].sort_values("layer")
        ax.plot(sub_df["layer"], sub_df["normalized_effective_rank"], "o-",
                label=kind.replace("_", " ").title(), color=colors[kind], linewidth=2)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Fraction of Effective Rank")
    ax.set_title("Effective Rank by Activation Type")
    ax.legend()
    ax.set_ylim(0, 1)

    # 右: 累积方差恢复
    ax = axes[2]
    for label, prefix in sites.items():
        eigvals = np.maximum(sub[f"{prefix}__eigenvalues"], 0)
        cum = np.cumsum(eigvals) / eigvals.sum()
        ax.plot(range(1, len(cum) + 1), cum, label=label, linewidth=2)
    ax.axhline(y=0.99, color="gray", linestyle="--", alpha=0.5, label="99% variance")
    ax.set_xlabel("Number of Components")
    ax.set_ylabel("Cumulative Variance Captured")
    ax.set_title(f"Variance Recovery (Layer {TARGET_LAYER})")
    ax.legend()
    ax.set_ylim(0, 1.05)
    ax.set_xlim(0, 768)

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "spectral_analysis.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n图已保存: {FIGURES_DIR / 'spectral_analysis.png'}")

    # --- 关键数值 ---
    print(f"\n关键数值 (Layer {TARGET_LAYER}):")
    records = []
    for label, prefix in sites.items():
        eigvals = np.maximum(sub[f"{prefix}__eigenvalues"], 0)
        sv = np.sqrt(eigvals)
        n_above_1pct = int((sv > 0.01 * sv.max()).sum())
        cum = np.cumsum(eigvals) / eigvals.sum()
        n_99 = int(np.searchsorted(cum, 0.99) + 1)
        dim = len(eigvals)
        print(f"  {label:20s}: dirs>1%={n_above_1pct}/{dim} ({100*n_above_1pct/dim:.1f}%), "
              f"99% var in {n_99}/{dim} ({100*n_99/dim:.1f}%)")
        records.append({
            "activation_type": label,
            "layer": TARGET_LAYER,
            "directions_above_1pct": n_above_1pct,
            "dimension": dim,
            "pct_above_1pct": round(100 * n_above_1pct / dim, 1),
            "n_for_99pct_variance": n_99,
            "pct_for_99pct": round(100 * n_99 / dim, 1),
        })
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Path A: 激活消融与损失恢复 (Figure 3b)
# ---------------------------------------------------------------------------
def make_zero_hook():
    """零消融 hook: 把模块输出替换为全零。"""
    def hook(_module, _inputs, output):
        t = output[0] if isinstance(output, (tuple, list)) else output
        result = torch.zeros_like(t)
        if isinstance(output, (tuple, list)):
            return (result,) + tuple(output[1:])
        return result
    return hook


def make_projection_hook(eigenvectors: np.ndarray, mean: np.ndarray,
                         n_components: int, device: str):
    """SVD 投影 hook: 把输出投影到前 n 个主成分子空间。"""
    V_n = torch.from_numpy(eigenvectors[:, :n_components]).to(device).float()   # [H, n]
    mean_t = torch.from_numpy(mean).to(device).float()                          # [H]
    P = V_n @ V_n.T                                                            # [H, H]

    def hook(_module, _inputs, output):
        t = output[0] if isinstance(output, (tuple, list)) else output
        result = ((t.float() - mean_t) @ P + mean_t).to(t.dtype)
        if isinstance(output, (tuple, list)):
            return (result,) + tuple(output[1:])
        return result
    return hook


def path_a_loss_recovery(device: str = "cuda:0", n_texts: int = N_TEXTS_DEFAULT) -> pd.DataFrame:
    """激活消融 → SVD 分量逐步恢复 → 下游损失恢复曲线。"""
    import torch.nn.functional as F
    from transformers import BertTokenizerFast
    from src.models.modeling import build_candidate
    from src.layer_probe_representations import freeze_and_validate_inference_model

    print("\n" + "=" * 60)
    print("Path A: 激活消融与损失恢复")
    print("=" * 60)

    # --- 加载模型 ---
    print("加载模型...")
    candidate = build_candidate(
        base_model_dir=str(ROOT / CONFIG["model"]["base_model_dir"]),
        checkpoint_path=str(ROOT / CONFIG["model"]["checkpoint"]),
        device=device,
        dtype=torch.float16,
    )
    freeze_and_validate_inference_model(candidate, expected_hidden_layers=13)
    candidate.eval()
    bert = candidate.bert

    # --- 加载 tokenizer ---
    tokenizer = BertTokenizerFast.from_pretrained(
        str(ROOT / CONFIG["model"]["base_model_dir"]), local_files_only=True
    )

    # --- 加载文本 ---
    print("加载文本...")
    sample = pd.read_parquet(RUN_DIR / "sample" / "sample_manifest.parquet")
    reports_path = Path(str(CONFIG["text"]["path"]))
    if not reports_path.is_absolute():
        reports_path = ROOT / reports_path
    id_col = CONFIG["text"].get("id_column", "report_id")
    text_col = CONFIG["text"].get("text_column", "text")
    reports = pd.read_parquet(reports_path, columns=[id_col, text_col]).rename(
        columns={id_col: "report_id", text_col: "text"})
    merged = sample.merge(reports[["report_id", "text"]], on="report_id", how="left")
    texts = merged["text"].dropna().tolist()[:n_texts]
    print(f"  使用 {len(texts)} 条文本")

    # --- 分词 ---
    encoded = tokenizer(
        texts, padding=True, truncation=True, max_length=512,
        return_tensors="pt", return_token_type_ids=True,
    )
    encoded = {k: v.to(device) for k, v in encoded.items()}

    # --- 基线: 正常前向 ---
    print("基线前向...")
    with torch.inference_mode():
        out = candidate(**encoded)
        baseline_logits = out.logits.float().cpu()
    baseline_probs = F.softmax(baseline_logits, dim=-1)

    # --- 消融实验 ---
    sub = np.load(ANALYSIS_DIR / "subspaces.npz", allow_pickle=False)
    experiment_sites = {
        "Attention Output": "attention_output_06",
        "Residual Stream": "residual_06",
    }
    n_list = [0, 1, 2, 4, 8, 16, 32, 64, 128, 192, 256, 384, 512, 640, 768]

    all_results = []
    for label, site in experiment_sites.items():
        print(f"\n  [{label}] site={site}")
        prefix = f"{STREAM}__{site}"
        eigvecs = sub[f"{prefix}__eigenvectors"]   # [768, 768]
        mean = sub[f"{prefix}__mean"]              # [768]

        # 根据 site 类型确定 hook 目标模块 (与 BertActivationHooks 一致)
        site_type = site.rsplit("_", 1)[0]   # attention_output / mlp_output / residual / z
        layer_num = int(site.rsplit("_", 1)[1])
        layer_idx = layer_num - 1             # 0-indexed
        if site_type == "attention_output":
            target_module = bert.encoder.layer[layer_idx].attention.output.dense
        elif site_type == "mlp_output":
            target_module = bert.encoder.layer[layer_idx].output.dense
        elif site_type == "residual" and layer_num > 0:
            target_module = bert.encoder.layer[layer_idx]
        elif site_type == "residual" and layer_num == 0:
            target_module = bert.embeddings
        elif site_type == "z":
            target_module = bert.encoder.layer[layer_idx].attention.self
        else:
            raise ValueError(f"未知 site 类型: {site_type}")

        for n in n_list:
            if n == 0:
                hook_fn = make_zero_hook()
                desc = "zero-ablation"
            elif n >= 768:
                hook_fn = None
                desc = "full (no modification)"
            else:
                hook_fn = make_projection_hook(eigvecs, mean, n, device)
                desc = f"top-{n} SVD projection"

            handle = None
            if hook_fn is not None:
                handle = target_module.register_forward_hook(hook_fn)

            with torch.inference_mode():
                out = candidate(**encoded)
                mod_logits = out.logits.float().cpu()

            if handle is not None:
                handle.remove()

            mod_probs = F.softmax(mod_logits, dim=-1)
            kl = F.kl_div(
                torch.log(mod_probs + 1e-8), baseline_probs, reduction="sum"
            ).item() / len(texts)

            all_results.append({
                "activation_type": label,
                "site": site,
                "layer": TARGET_LAYER,
                "n_components": n,
                "kl_divergence": round(kl, 6),
                "description": desc,
            })
            if n in (0, 64, 128, 256, 384, 768):
                print(f"    N={n:3d} ({desc}): KL={kl:.6f}")

    results_df = pd.DataFrame(all_results)

    # --- 计算恢复比例 ---
    recovery = []
    for label in experiment_sites:
        sub_df = results_df[results_df["activation_type"] == label].set_index("n_components")
        kl_zero = sub_df.loc[0, "kl_divergence"]
        for n in n_list:
            kl_n = sub_df.loc[n, "kl_divergence"]
            frac = 1.0 - kl_n / max(kl_zero, 1e-8)
            recovery.append({
                "activation_type": label,
                "n_components": n,
                "kl_divergence": kl_n,
                "kl_zero": kl_zero,
                "fraction_recovered": round(frac, 6),
            })
    recovery_df = pd.DataFrame(recovery)
    recovery_df.to_parquet(ANALYSIS_DIR / "loss_recovery_metrics.parquet", index=False)

    # --- 画图 ---
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 5))
    for label in experiment_sites:
        sub_r = recovery_df[recovery_df["activation_type"] == label].sort_values("n_components")
        ax.plot(sub_r["n_components"], sub_r["fraction_recovered"], "o-",
                label=label, linewidth=2, markersize=6)
    ax.axhline(y=0.99, color="gray", linestyle="--", alpha=0.5, label="99% recovered")
    ax.set_xlabel("Number of Components Retained", fontsize=12)
    ax.set_ylabel("Fraction of Loss Recovered", fontsize=12)
    ax.set_title(f"Downstream Loss Recovery (Layer {TARGET_LAYER})", fontsize=13)
    ax.legend(fontsize=11)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlim(-10, 780)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "loss_recovery.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n图已保存: {FIGURES_DIR / 'loss_recovery.png'}")

    # --- 关键数值 ---
    print(f"\n损失恢复关键数值 (Layer {TARGET_LAYER}):")
    for label in experiment_sites:
        sub_r = recovery_df[recovery_df["activation_type"] == label].sort_values("n_components")
        for _, row in sub_r[sub_r["n_components"].isin((0, 64, 128, 256, 384, 768))].iterrows():
            print(f"  {label:20s} N={int(row['n_components']):3d}: KL={row['kl_divergence']:.6f}  recovered={row['fraction_recovered']:.1%}")
        n_99_rows = sub_r[sub_r["fraction_recovered"] >= 0.99]
        n_99 = int(n_99_rows["n_components"].min()) if len(n_99_rows) > 0 else 768
        print(f"  -> 99% 恢复需要 {n_99}/768 维 ({100*n_99/768:.1f}%)")

    return recovery_df


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--n-texts", type=int, default=N_TEXTS_DEFAULT)
    parser.add_argument("--skip-path-a", action="store_true", help="只跑 Path C")
    args = parser.parse_args()

    df_c = path_c_spectral_analysis()
    print("\nPath C 完成。\n")

    if not args.skip_path_a:
        df_a = path_a_loss_recovery(device=args.device, n_texts=args.n_texts)
        print("\nPath A 完成。")

    print("\n全部完成。结果保存在:")
    print(f"  {FIGURES_DIR / 'spectral_analysis.png'}")
    print(f"  {FIGURES_DIR / 'loss_recovery.png'}")
    print(f"  {ANALYSIS_DIR / 'loss_recovery_metrics.parquet'}")
    print(f"  {ANALYSIS_DIR / 'spectral_summary.parquet'}")
