"""Generate subspace_factor_full.ipynb — coverage H/L + 12-layer dead-128 via fc head, wide table output."""
import json
from pathlib import Path

NB_PATH = Path("notebooks/subspace_factor_full.ipynb")
CELLS = []

def add_code(source, cell_id):
    lines = source.strip().split("\n")
    src = [line + "\n" for line in lines[:-1]] + [lines[-1]]
    CELLS.append({"cell_type": "code", "id": cell_id, "metadata": {}, "execution_count": None, "outputs": [], "source": src})

def add_md(source, cell_id):
    lines = source.strip().split("\n")
    src = [line + "\n" for line in lines]
    CELLS.append({"cell_type": "markdown", "id": cell_id, "metadata": {}, "source": src})

add_md("""# 子空间因子输出 (完整版)

将 CLS 表征投影到多个子空间, 映射回 768 维, 用冻结 fc head 读出 prob, sum 聚合到 stock-day。

**14 组因子**:
- H: 高覆盖子空间 (317 维, coverage ≥ 0.05)
- L: 低覆盖子空间 (451 维, coverage < 0.05)
- Dead-01 ~ Dead-12: 每层 attention output 的 bottom-128 dead 方向

**不训练、不跑模型、纯 numpy。需要 per_file/ 下有 sum_cls (extract_gubapost_cls 跑完后可用)。**""", "title")

add_code(r'''# 1. 加载 fc head 权重 + 所有投影基 + 预计算
import os, sys, glob, numpy as np, pandas as pd, torch
from scipy.special import softmax

ROOT = "/home/intern_fjq_2026/Projects/chinese-wwm-roberta"
os.chdir(ROOT); sys.path.insert(0, ROOT)

# --- fc head 权重 (从 checkpoint, 不加载 BERT) ---
CKPT = os.path.join(ROOT, "chinese-wwm-roberta.ckpt")
state = torch.load(CKPT, map_location="cpu", weights_only=True)
sd = state.get("state_dict", state)
fc_keys = {k: v for k, v in sd.items() if k.startswith("fc.")}
W = fc_keys["fc.weight"].numpy().astype(np.float64)    # [2, 768]
b = fc_keys["fc.bias"].numpy().astype(np.float64)        # [2]
print(f"fc head: W {W.shape}, b {b.shape}")

# --- run 目录 ---
RUN_DIR = os.path.join(ROOT, "artifacts", "checkpoint_activation_rank", "runs", "gubapost_v1")

# --- coverage H/L 方向 (来自 coverage_ablation) ---
COV_DIR = os.path.join(RUN_DIR, "extensions", "coverage_ablation_v1", "direction_sets.npz")
cov = np.load(COV_DIR)
Q_H = cov["keep_317_complement_K64"].astype(np.float64)   # [768, 317]
Q_L = cov["keep_451_lowcov_K64"].astype(np.float64)       # [768, 451]
print(f"coverage: Q_H {Q_H.shape}, Q_L {Q_L.shape}")

# --- 12 层 dead-128 方向 (来自 subspaces.npz) ---
SUB_PATH = os.path.join(RUN_DIR, "analysis", "subspaces.npz")
sub = np.load(SUB_PATH, allow_pickle=False)
STREAM = "token_natural_filtered"
N_LAYERS = 12
DEAD_DIM = 128
HIDDEN_DIM = 768
K_SPLIT = HIDDEN_DIM - DEAD_DIM  # 640, 前 640 是 top, 后 128 是 dead

Q_dead = {}  # {layer: [768, 128]}
for L in range(1, N_LAYERS + 1):
    key = f"{STREAM}__attention_output_{L:02d}__eigenvectors"
    assert key in sub, f"subspaces.npz 缺少: {key}"
    full_V = sub[key].astype(np.float64)  # [768, 768], 列按特征值降序
    Q_dead[L] = full_V[:, K_SPLIT:]        # [768, 128] bottom-128
print(f"dead-128: {N_LAYERS} layers, each {Q_dead[1].shape}")

# --- 验证正交性 ---
for L in range(1, N_LAYERS + 1):
    err = float(np.max(np.abs(Q_dead[L].T @ Q_dead[L] - np.eye(DEAD_DIM))))
    assert err < 1e-5, f"Layer {L} dead-128 正交性失败: {err}"
print("dead-128 正交性全部通过")

# --- 预计算: W @ P (把投影+fc head 合并为一次矩阵乘) ---
# logits = mean_cls @ P @ W.T + b = mean_cls @ (W @ P).T + b
# 预算 W_sub = W @ P, 则 logits = mean_cls @ W_sub.T + b
W_H = W @ (Q_H @ Q_H.T)          # [2, 768]
W_L = W @ (Q_L @ Q_L.T)          # [2, 768]
W_dead = {}
for L in range(1, N_LAYERS + 1):
    P_dead = Q_dead[L] @ Q_dead[L].T   # [768, 768]
    W_dead[L] = W @ P_dead             # [2, 768]

# 所有子空间的预计算权重
SUBSPACE_NAMES = ["H", "L"] + [f"dead_{L:02d}" for L in range(1, N_LAYERS + 1)]
W_SUB = {"H": W_H, "L": W_L}
W_SUB.update({f"dead_{L:02d}": W_dead[L] for L in range(1, N_LAYERS + 1)})
print(f"预计算完成: {len(W_SUB)} 个子空间权重矩阵")
for name in SUBSPACE_NAMES:
    print(f"  {name}: {W_SUB[name].shape}")''', "load-and-precompute")

add_code(r'''# 2. 逐文件处理: mean_cls → 14 组投影+fc head → prob → sum 聚合
PER_DIR = os.path.join(ROOT, "artifacts", "gubapost_cls", "per_file")
pf_files = sorted(glob.glob(os.path.join(PER_DIR, "*.parquet")))
assert pf_files, f"per_file/ 下没有文件 -> 先跑 extract_gubapost_cls.ipynb"
print(f"per_file: {len(pf_files)} 个文件")

results = []
for fi, fp in enumerate(pf_files):
    df = pd.read_parquet(fp, columns=["available_date", "symbol", "n_posts", "sum_cls"])
    n = len(df)
    if n == 0:
        continue

    nposts = df["n_posts"].values.astype(np.float64)
    # 防除零
    nposts_safe = np.where(nposts > 0, nposts, 1.0)
    sum_cls = np.stack(df["sum_cls"].values).astype(np.float64)
    mean_cls = sum_cls / nposts_safe[:, None]    # [n, 768] 回到单帖尺度

    # 对每个子空间: logits = mean_cls @ W_sub.T + b, prob = softmax[:, 1]
    row_data = {
        "date": df["available_date"].values,
        "symbol": df["symbol"].values,
        "n_posts": nposts.astype(np.int32),
    }
    for name in SUBSPACE_NAMES:
        logits = mean_cls @ W_SUB[name].T + b    # [n, 2]
        prob = softmax(logits, axis=1)[:, 1]     # [n]
        row_data[f"mean_prob_{name}"] = prob.astype(np.float32)
        row_data[f"sum_prob_{name}"] = (prob * nposts_safe).astype(np.float32)

    results.append(pd.DataFrame(row_data))

    if (fi + 1) % 100 == 0 or fi == 0 or fi == len(pf_files) - 1:
        print(f"  {fi+1}/{len(pf_files)}: {os.path.basename(fp)} -> {n:,} rows")

long_df = pd.concat(results, ignore_index=True)
del results
n_dates = long_df["date"].nunique()
n_syms = long_df["symbol"].nunique()
print(f"\n总计: {len(long_df):,} stock-days | {n_dates} dates | {n_syms} symbols")
print(f"日期范围: {long_df['date'].min()} ~ {long_df['date'].max()}")''', "process-files")

add_code(r'''# 3. 生成宽表并保存
OUT_DIR = os.path.join(ROOT, "artifacts", "gubapost_cls", "subspace_factors_full")
os.makedirs(OUT_DIR, exist_ok=True)

# 每个子空间生成两张宽表: sum 版 + mean 版
saved_files = []
for name in SUBSPACE_NAMES:
    for agg in ["sum", "mean"]:
        col = f"{agg}_prob_{name}"
        wide = long_df.pivot_table(index="date", columns="symbol", values=col, aggfunc="sum")
        wide = wide.sort_index()
        fname = f"factor_{agg}_{name}.parquet"
        fpath = os.path.join(OUT_DIR, fname)
        wide.to_parquet(fpath)
        saved_files.append(fname)

# long format (全部列, 方便对齐)
long_df.to_parquet(os.path.join(OUT_DIR, "subspace_factors_long.parquet"), index=False)
saved_files.append("subspace_factors_long.parquet")

print(f"保存 {len(saved_files)} 个文件到 {OUT_DIR}:")
for f in sorted(saved_files):
    sz = os.path.getsize(os.path.join(OUT_DIR, f))
    print(f"  {f}: {sz/1e6:.1f}MB")''', "build-wide-tables")

add_code(r'''# 4. 验证 + 预览
print("=== 预览: sum_prob_H ===")
w = pd.read_parquet(os.path.join(OUT_DIR, "factor_sum_H.parquet"))
print(f"  shape: {w.shape}")
print(w.iloc[:3, :4])
print(f"  NaN: {w.isna().sum().sum() / w.size:.1%}")

print("\n=== 预览: sum_prob_dead_06 ===")
w = pd.read_parquet(os.path.join(OUT_DIR, "factor_sum_dead_06.parquet"))
print(f"  shape: {w.shape}")
print(w.iloc[:3, :4])

print("\n=== 各子空间统计 ===")
for name in SUBSPACE_NAMES:
    vals = long_df[f"sum_prob_{name}"].values
    vals = vals[~np.isnan(vals)]
    print(f"  {name:10s}: min={vals.min():.2f} max={vals.max():.2f} "
          f"mean={vals.mean():.4f} std={vals.std():.4f}")

print("\n=== logits 分解验证 (H + L = full - b) ===")
# 重新加载一个样本验证
sample = pd.read_parquet(sorted(glob.glob(os.path.join(PER_DIR, "*.parquet")))[0],
                         columns=["n_posts", "sum_cls"]).head(100)
np_s = sample["n_posts"].values.astype(np.float64)
np_s = np.where(np_s > 0, np_s, 1.0)
mc = np.stack(sample["sum_cls"].values).astype(np.float64) / np_s[:, None]
logits_H = mc @ W_H.T + b
logits_L = mc @ W_L.T + b
logits_full = mc @ W.T + b
err = np.max(np.abs((logits_H + logits_L - b) - logits_full))
print(f"  H+L 分解误差: {err:.2e} (应为 ~0)")
print("\n验证完成, 可以用宽表做回测。")''', "verify")

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
