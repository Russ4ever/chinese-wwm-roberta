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
# 与论文公式严格一致:
#   - 下游损失 = MLM 交叉熵 (微调backbone + base MLM head), 对应论文的 language model loss
#   - 投影作用于原始激活 (不做均值中心化), N=0 严格等价于零消融, 曲线起点必为 0%
#   - fraction = (loss_zero - loss_N) / (loss_zero - loss_original)
# ---------------------------------------------------------------------------
def make_projection_hook(eigenvectors: np.ndarray, n_components: int, device: str):
    """SVD 投影 hook: 把输出替换为其在前 n 个主成分子空间上的投影 (原始激活, 不中心化)。

    n_components=0 时投影矩阵为零矩阵, 输出被替换为全零 —— 与零消融严格等价,
    因此恢复曲线从 N=0 的 0% 出发, 单调趋向 N=768 的 100%。
    """
    V_n = torch.from_numpy(eigenvectors[:, :n_components]).to(device).float()   # [H, n]
    P = V_n @ V_n.T                                                            # [H, H]

    def hook(_module, _inputs, output):
        t = output[0] if isinstance(output, (tuple, list)) else output
        result = (t.float() @ P).to(t.dtype)
        if isinstance(output, (tuple, list)):
            return (result,) + tuple(output[1:])
        return result
    return hook


def _resolve_target_module(bert, site: str):
    """site 名 -> hook 目标模块 (与 BertActivationHooks 的注册点一致)。"""
    site_type = site.rsplit("_", 1)[0]
    layer_num = int(site.rsplit("_", 1)[1])
    layer_idx = layer_num - 1  # 0-indexed
    if site_type == "attention_output":
        return bert.encoder.layer[layer_idx].attention.output.dense
    if site_type == "mlp_output":
        return bert.encoder.layer[layer_idx].output.dense
    if site_type == "residual" and layer_num > 0:
        return bert.encoder.layer[layer_idx]
    if site_type == "residual" and layer_num == 0:
        return bert.embeddings
    if site_type == "z":
        return bert.encoder.layer[layer_idx].attention.self
    raise ValueError(f"未知 site 类型: {site_type}")


def path_a_loss_recovery(device: str = "cuda:0", n_texts: int = N_TEXTS_DEFAULT) -> pd.DataFrame:
    """激活消融 → SVD 分量逐步恢复 → 下游 MLM 损失恢复曲线。"""
    import torch.nn.functional as F
    from transformers import BertTokenizerFast, BertForMaskedLM
    from src.models.checkpoint import load_state_dict_safe, strip_prefix

    print("\n" + "=" * 60)
    print("Path A: 激活消融与损失恢复 (MLM CE, 论文公式)")
    print("=" * 60)

    # --- 加载模型: 微调 backbone + base MLM head ---
    print("加载模型 (微调backbone + base MLM head)...")
    base_dir = str(ROOT / CONFIG["model"]["base_model_dir"])
    mlm = BertForMaskedLM.from_pretrained(base_dir)
    state = strip_prefix(load_state_dict_safe(str(ROOT / CONFIG["model"]["checkpoint"]), map_location="cpu"))
    # BertForMaskedLM 的 bert 默认不含 pooler, 过滤掉 checkpoint 中的 pooler 权重
    bert_state = {k[len("bert."):]: v for k, v in state.items()
                  if k.startswith("bert.") and not k.startswith("bert.pooler.")}
    mlm.bert.load_state_dict(bert_state, strict=True)
    mlm = mlm.to(device=device, dtype=torch.float16).eval()
    bert = mlm.bert

    tokenizer = BertTokenizerFast.from_pretrained(base_dir, local_files_only=True)

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

    # --- 分词 + 固定 mask (所有条件共用同一mask, 保证可比) ---
    encoded = tokenizer(
        texts, padding=True, truncation=True, max_length=512,
        return_tensors="pt", return_token_type_ids=True,
    )
    encoded = {k: v.to(device) for k, v in encoded.items()}

    generator = torch.Generator(device=device).manual_seed(20260901)
    special = (
        (encoded["input_ids"] == tokenizer.cls_token_id)
        | (encoded["input_ids"] == tokenizer.sep_token_id)
        | (encoded["input_ids"] == tokenizer.pad_token_id)
    )
    prob = torch.rand(encoded["input_ids"].shape, device=device, generator=generator)
    mask_positions = (prob < 0.15) & ~special
    masked_ids = encoded["input_ids"].clone()
    masked_ids[mask_positions] = tokenizer.mask_token_id
    targets = encoded["input_ids"][mask_positions]
    print(f"  mask token 数: {int(mask_positions.sum())}")

    def masked_lm_loss() -> float:
        """当前模型状态 (受hook影响) 下的 MLM 交叉熵。"""
        out = mlm(input_ids=masked_ids,
                  attention_mask=encoded["attention_mask"],
                  token_type_ids=encoded["token_type_ids"])
        logits = out.logits[mask_positions].float()
        return F.cross_entropy(logits, targets).item()

    # --- 基线损失 ---
    print("基线前向...")
    with torch.inference_mode():
        loss_original = masked_lm_loss()

    # --- 消融实验 ---
    # BERT 残差连接冗余度高, 单消融中间层 attention output 对 MLM 损失影响≈0
    # (CE 1.0542 -> 1.0558, 分母为噪声级), 因此 attention/MLP 采用全部 12 层联合消融;
    # residual stream 保持论文的单位置消融 (layer 6), 其分母天然健康。
    sub = np.load(ANALYSIS_DIR / "subspaces.npz", allow_pickle=False)
    experiment_sites = {
        "Attention Output": [f"attention_output_{i:02d}" for i in range(1, 13)],
        "MLP Output": [f"mlp_output_{i:02d}" for i in range(1, 13)],
        "Residual Stream": ["residual_06"],
    }
    n_list = [0, 8, 16, 32, 64, 128, 192, 256, 320, 384, 448, 512, 576, 640, 704, 736, 768]

    all_results = []
    for label, site_list in experiment_sites.items():
        print(f"\n  [{label}] sites={site_list[0]}..{site_list[-1]} ({len(site_list)}个)")

        # 每个 site 用各自层的主成分矩阵
        module_eigvecs = []
        for site in site_list:
            prefix = f"{STREAM}__{site}"
            eigvecs = sub[f"{prefix}__eigenvectors"]   # [768, 768]
            module_eigvecs.append((_resolve_target_module(bert, site), eigvecs))

        for n in n_list:
            desc = "zero-ablation (=0 components)" if n == 0 else (
                "full (no modification)" if n >= 768 else f"top-{n} SVD projection/layer")

            handles = []
            if n < 768:
                # n=0 时 P=0, 输出全零 —— 与零消融严格一致
                for target_module, eigvecs in module_eigvecs:
                    handles.append(target_module.register_forward_hook(
                        make_projection_hook(eigvecs, n, device)))

            with torch.inference_mode():
                loss_n = masked_lm_loss()

            for handle in handles:
                handle.remove()

            all_results.append({
                "activation_type": label,
                "n_sites": len(site_list),
                "n_components": n,
                "ce_loss": round(loss_n, 6),
                "description": desc,
            })
            if n in (0, 64, 128, 256, 384, 512, 768):
                print(f"    N={n:3d}: CE={loss_n:.6f}")

    results_df = pd.DataFrame(all_results)

    # --- 恢复比例: (loss_zero - loss_N) / (loss_zero - loss_original) ---
    recovery = []
    for label in experiment_sites:
        sub_df = results_df[results_df["activation_type"] == label].set_index("n_components")
        loss_zero = sub_df.loc[0, "ce_loss"]
        denom = max(loss_zero - loss_original, 1e-8)
        for n in n_list:
            loss_n = sub_df.loc[n, "ce_loss"]
            recovery.append({
                "activation_type": label,
                "n_components": n,
                "ce_loss": loss_n,
                "ce_zero": loss_zero,
                "ce_original": round(loss_original, 6),
                "fraction_recovered": round((loss_zero - loss_n) / denom, 6),
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
    ax.set_xlabel("Number of Components Retained (per site)", fontsize=12)
    ax.set_ylabel("Fraction of Loss Recovered", fontsize=12)
    ax.set_title("Downstream MLM Loss Recovery (Attention/MLP: L1-12 joint; Residual: L6)", fontsize=12)
    ax.legend(fontsize=11)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlim(-10, 780)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "loss_recovery.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n图已保存: {FIGURES_DIR / 'loss_recovery.png'}")

    # --- 关键数值 ---
    print(f"\n损失恢复关键数值 (Layer {TARGET_LAYER}, loss_original={loss_original:.4f}):")
    for label in experiment_sites:
        sub_r = recovery_df[recovery_df["activation_type"] == label].sort_values("n_components")
        for _, row in sub_r[sub_r["n_components"].isin((0, 64, 128, 256, 384, 768))].iterrows():
            print(f"  {label:20s} N={int(row['n_components']):3d}: CE={row['ce_loss']:.4f}  recovered={row['fraction_recovered']:.1%}")
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
