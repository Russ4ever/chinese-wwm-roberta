"""Generate subspace_factor_output.ipynb — project CLS to H/L subspaces, fc head readout, wide table output."""
import json
from pathlib import Path

NB_PATH = Path("notebooks/subspace_factor_output.ipynb")
CELLS = []

def add_code(source, cell_id):
    lines = source.strip().split("\n")
    src = [line + "\n" for line in lines[:-1]] + [lines[-1]]
    CELLS.append({"cell_type": "code", "id": cell_id, "metadata": {}, "execution_count": None, "outputs": [], "source": src})

CELLS.append({
    "cell_type": "markdown", "id": "title", "metadata": {},
    "source": [
        "# 子空间因子输出\n",
        "\n",
        "将 CLS 表征投影到高覆盖 (H) / 低覆盖 (L) 子空间，映射回 768 维，\n",
        "用冻结的 fc head (Linear + softmax) 读出 prob，sum 聚合到 stock-day。\n",
        "\n",
        "输出两张宽表 (date × symbol)：\n",
        "- `factor_prob_H.parquet` — 高覆盖子空间 prob\n",
        "- `factor_prob_L.parquet` — 低覆盖子空间 prob\n",
        "\n",
        "不训练、不跑模型、纯 numpy。",
    ],
})

add_code(r'''# 1. 加载 fc head 权重 + coverage 方向
import os, sys, glob, numpy as np, pandas as pd, torch
from scipy.special import softmax

ROOT = "/home/intern_fjq_2026/Projects/chinese-wwm-roberta"
os.chdir(ROOT); sys.path.insert(0, ROOT)

# fc head 权重 (从 checkpoint 直接读, 不加载 BERT)
CKPT = os.path.join(ROOT, "chinese-wwm-roberta.ckpt")
state = torch.load(CKPT, map_location="cpu", weights_only=True)
sd = state.get("state_dict", state)
W_fc = {k[len("fc."):]: v for k, v in sd.items() if k.startswith("fc.")}
W = W_fc["weight"].numpy().astype(np.float64)   # [2, 768]
b = W_fc["bias"].numpy().astype(np.float64)      # [2]
print(f"fc head: W {W.shape}, b {b.shape}")

# coverage 方向
DIRS = os.path.join(ROOT, "artifacts", "checkpoint_activation_rank", "runs", "gubapost_v1",
                    "extensions", "coverage_ablation_v1", "direction_sets.npz")
dirs = np.load(DIRS)
Q_H = dirs["keep_317_complement_K64"].astype(np.float64)   # [768, 317]
Q_L = dirs["keep_451_lowcov_K64"].astype(np.float64)      # [768, 451]
PH = Q_H @ Q_H.T   # [768, 768] H 投影矩阵
PL = Q_L @ Q_L.T   # [768, 768] L 投影矩阵
print(f"Q_H {Q_H.shape} Q_L {Q_L.shape}")

# 预计算: fc head 对投影矩阵的组合 (只算一次)
# prob = softmax(cls_projected @ W.T + b)[:, 1]
# cls_projected = mean_cls @ P.T  (P is 768x768)
# = mean_cls @ P.T @ W.T + b
# 预算 W_H = W @ PH  [2, 768], W_L = W @ PL  [2, 768]
W_H = W @ PH    # [2, 768] — fc head 权重作用在 H 投影后的 CLS 上
W_L = W @ PL    # [2, 768]
print(f"预计算: W_H {W_H.shape}, W_L {W_L.shape}")
print("加载完成")''', "load-weights")

add_code(r'''# 2. 逐文件处理: project → fc head → softmax → (date, symbol, prob_H, prob_L)
PER_DIR = os.path.join(ROOT, "artifacts", "gubapost_cls", "per_file")
pf_files = sorted(glob.glob(os.path.join(PER_DIR, "*.parquet")))
print(f"per_file: {len(pf_files)} 个文件")

results = []
for fi, fp in enumerate(pf_files):
    # 只读需要的列, 跳过 sum_attn (省内存)
    df = pd.read_parquet(fp, columns=["available_date", "symbol", "n_posts", "sum_cls"])
    n = len(df)
    if n == 0:
        continue

    nposts = df["n_posts"].values.astype(np.float64)
    sum_cls = np.stack(df["sum_cls"].values).astype(np.float64)     # [n, 768]
    # mean_cls: 回到单帖尺度 (fc head 训练时的输入尺度)
    mean_cls = sum_cls / nposts[:, None]                              # [n, 768]

    # fc head 读出 (用预算的 W_H, W_L)
    logits_H = mean_cls @ W_H.T + b    # [n, 2]
    logits_L = mean_cls @ W_L.T + b    # [n, 2]
    prob_H = softmax(logits_H, axis=1)[:, 1]   # [n]
    prob_L = softmax(logits_L, axis=1)[:, 1]   # [n]

    # sum 聚合: prob * n_posts (恢复到 sum 尺度)
    sum_prob_H = prob_H * nposts
    sum_prob_L = prob_L * nposts

    results.append(pd.DataFrame({
        "date": df["available_date"].values,
        "symbol": df["symbol"].values,
        "sum_prob_H": sum_prob_H,
        "sum_prob_L": sum_prob_L,
        "n_posts": nposts.astype(np.int32),
        "mean_prob_H": prob_H.astype(np.float32),
        "mean_prob_L": prob_L.astype(np.float32),
    }))

    if (fi + 1) % 100 == 0 or fi == 0 or fi == len(pf_files) - 1:
        print(f"  {fi+1}/{len(pf_files)}: {os.path.basename(fp)} -> {n:,} rows")

long_df = pd.concat(results, ignore_index=True)
del results
print(f"\n总计: {len(long_df):,} stock-days | {long_df['date'].nunique()} dates | {long_df['symbol'].nunique()} symbols")
print(f"日期范围: {long_df['date'].min()} ~ {long_df['date'].max()}")''', "process-files")

add_code(r'''# 3. 生成宽表 (date × symbol) 并保存
OUT_DIR = os.path.join(ROOT, "artifacts", "gubapost_cls", "subspace_factors")
os.makedirs(OUT_DIR, exist_ok=True)

# 宽表 1: sum_prob_H (sum 聚合的高覆盖 prob)
wide_H = long_df.pivot_table(index="date", columns="symbol", values="sum_prob_H", aggfunc="sum")
wide_H = wide_H.sort_index()
wide_H.to_parquet(os.path.join(OUT_DIR, "factor_sum_prob_H.parquet"))
print(f"factor_sum_prob_H: {wide_H.shape} (date × symbol)")

# 宽表 2: sum_prob_L (sum 聚合的低覆盖 prob)
wide_L = long_df.pivot_table(index="date", columns="symbol", values="sum_prob_L", aggfunc="sum")
wide_L = wide_L.sort_index()
wide_L.to_parquet(os.path.join(OUT_DIR, "factor_sum_prob_L.parquet"))
print(f"factor_sum_prob_L: {wide_L.shape} (date × symbol)")

# 额外: mean 版本 (每股每天帖子的平均 prob, 可能更稳定)
wide_mH = long_df.pivot_table(index="date", columns="symbol", values="mean_prob_H", aggfunc="mean")
wide_mH = wide_mH.sort_index()
wide_mH.to_parquet(os.path.join(OUT_DIR, "factor_mean_prob_H.parquet"))
print(f"factor_mean_prob_H: {wide_mH.shape}")

wide_mL = long_df.pivot_table(index="date", columns="symbol", values="mean_prob_L", aggfunc="mean")
wide_mL = wide_mL.sort_index()
wide_mL.to_parquet(os.path.join(OUT_DIR, "factor_mean_prob_L.parquet"))
print(f"factor_mean_prob_L: {wide_mL.shape}")

# 也存 long format (方便对齐)
long_df.to_parquet(os.path.join(OUT_DIR, "subspace_factors_long.parquet"), index=False)
print(f"\nlong format: {long_df.shape}")
print(f"\n所有产物保存在: {OUT_DIR}")
print("文件列表:")
for f in sorted(os.listdir(OUT_DIR)):
    sz = os.path.getsize(os.path.join(OUT_DIR, f))
    print(f"  {f}: {sz/1e6:.1f}MB")''', "build-wide-tables")

add_code(r'''# 4. 验证 + 预览
print("=== 预览: sum_prob_H 宽表 ===")
print(wide_H.iloc[:5, :5])
print(f"  shape: {wide_H.shape}")
print(f"  NaN 比例: {wide_H.isna().sum().sum() / wide_H.size:.1%}")
print()
print("=== 预览: sum_prob_L 宽表 ===")
print(wide_L.iloc[:5, :5])
print(f"  shape: {wide_L.shape}")
print(f"  NaN 比例: {wide_L.isna().sum().sum() / wide_L.size:.1%}")
print()
print("=== 统计 ===")
for name, w in [("sum_prob_H", wide_H), ("sum_prob_L", wide_L), ("mean_prob_H", wide_mH), ("mean_prob_L", wide_mL)]:
    vals = w.values.flatten()
    vals = vals[~np.isnan(vals)]
    print(f"  {name}: min={vals.min():.4f} max={vals.max():.4f} mean={vals.mean():.4f} std={vals.std():.4f}")''', "verify")

nb = {
    "cells": CELLS,
    "metadata": {
        "kernelspec": {"display_name": "nlp_fjq", "language": "python", "name": "nlp_fjq"},
        "language_info": {"name": "python", "version": "3.10.18"},
    },
    "nbformat": 4, "nbformat_minor": 5,
}
with open(NB_PATH, "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
print(f"Created {NB_PATH} with {len(CELLS)} cells")
