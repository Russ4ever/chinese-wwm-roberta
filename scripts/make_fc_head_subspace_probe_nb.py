"""Generate fc_head_subspace_probe.ipynb — pure-numpy fc head subspace probe."""
import json
from pathlib import Path

NB_PATH = Path("notebooks/fc_head_subspace_probe.ipynb")

CELLS = []

def add_cell(source, cell_id):
    lines = source.strip().split("\n")
    src = [line + "\n" for line in lines[:-1]] + [lines[-1]]
    CELLS.append({
        "cell_type": "code", "id": cell_id,
        "metadata": {}, "execution_count": None, "outputs": [], "source": src,
    })

CELLS.append({
    "cell_type": "markdown", "id": "title", "metadata": {},
    "source": [
        "# FC-Head Subspace Probe\n",
        "\n",
        "用模型自己的 fc head (Linear + softmax) 对 CLS 的高/低覆盖子空间分别读出,\n",
        "测各自的收益预测力。与 Ridge probe 互补: fc head 含 softmax 非线性, 权重来自端到端训练。\n",
        "\n",
        "**全程不跑模型、不训练、不用 GPU、不用 Ridge** — 纯 numpy。",
    ],
})

add_cell(r'''# 1. 加载 sum_cls + 投影 + fc head 读出
import os, sys, glob, numpy as np, pandas as pd
from scipy.special import softmax
from scipy.stats import spearmanr

ROOT = "/home/intern_fjq_2026/Projects/chinese-wwm-roberta"
os.chdir(ROOT); sys.path.insert(0, ROOT)

# --- 加载 per_file sum_cls ---
per_dir = os.path.join(ROOT, "artifacts", "gubapost_cls", "per_file")
pf_files = sorted(glob.glob(os.path.join(per_dir, "*.parquet")))
assert pf_files, "per_file/ 下没有文件"
print(f"加载 {len(pf_files)} 个 per-file parquet...")
dfs = [pd.read_parquet(f) for f in pf_files]
pf = pd.concat(dfs, ignore_index=True); del dfs
nposts = pf["n_posts"].values.astype(np.float64)
mean_cls = (np.stack(pf["sum_cls"].values).astype(np.float64) / nposts[:, None]).astype(np.float32)
mean_prob = (pf["sum_prob"].values / nposts).astype(np.float32)
meta = pf[["available_date", "symbol", "n_posts"]].copy()
del pf
print(f"mean_cls: {mean_cls.shape} | dates: {meta.available_date.min()}~{meta.available_date.max()}")

# --- 加载 coverage 方向 ---
DIRS = os.path.join(ROOT, "artifacts", "checkpoint_activation_rank", "runs", "gubapost_v1",
                    "extensions", "coverage_ablation_v1", "direction_sets.npz")
dirs = np.load(DIRS)
Q_H = dirs["keep_317_complement_K64"].astype(np.float32)   # [768, 317]
Q_L = dirs["keep_451_lowcov_K64"].astype(np.float32)       # [768, 451]
print(f"Q_H {Q_H.shape} Q_L {Q_L.shape}")

# --- 加载 fc head 权重 ---
import torch
CKPT = os.path.join(ROOT, "chinese-wwm-roberta.ckpt")
state = torch.load(CKPT, map_location="cpu", weights_only=True)
sd = state.get("state_dict", state)
W_fc = {k[len("fc."):]: v for k, v in sd.items() if k.startswith("fc.")}
W = W_fc["weight"].numpy().astype(np.float64)   # [2, 768]
b = W_fc["bias"].numpy().astype(np.float64)      # [2]
print(f"fc head: W {W.shape}, b {b.shape}")

# --- 投影 + fc head 读出 ---
PH = Q_H @ Q_H.T   # [768, 768] H 投影矩阵
PL = Q_L @ Q_L.T   # [768, 768] L 投影矩阵
cls_H = mean_cls @ PH.T   # [N, 768] 只保留 H 分量
cls_L = mean_cls @ PL.T   # [N, 768] 只保留 L 分量

logits_H = cls_H @ W.T + b    # [N, 2]
logits_L = cls_L @ W.T + b    # [N, 2]
logits_full = mean_cls @ W.T + b   # [N, 2]

prob_H = softmax(logits_H, axis=1)[:, 1]      # [N]
prob_L = softmax(logits_L, axis=1)[:, 1]      # [N]
prob_full = softmax(logits_full, axis=1)[:, 1]  # [N]
print(f"prob_H: [{prob_H.min():.4f}, {prob_H.max():.4f}]")
print(f"prob_L: [{prob_L.min():.4f}, {prob_L.max():.4f}]")
print(f"prob_full: [{prob_full.min():.4f}, {prob_full.max():.4f}]")
print(f"mean_prob (stored): [{mean_prob.min():.4f}, {mean_prob.max():.4f}]")

# 验证 logits 分解
err = np.max(np.abs((logits_H + logits_L - b) - logits_full))
print(f"logits 分解误差: {err:.2e} (应为 ~0)")''', "load-and-project")

add_cell(r'''# 2. 收益 + walk-forward + 直接 IC
# --- 收益 (winsorize, 前移1天) ---
rtn = pd.read_parquet("/home/intern_fjq_2026/data/RTN_daily/rtn_1d.parquet")
rtn_long = rtn.melt(id_vars="date", var_name="sym", value_name="r")
rtn_long["symbol"] = rtn_long["sym"].str.split(".").str[0]
rtn_long["date"] = pd.to_datetime(rtn_long["date"]).dt.strftime("%Y-%m-%d")
rtn_long = rtn_long.sort_values(["symbol", "date"])
rtn_long["y"] = rtn_long.groupby("symbol")["r"].shift(-1)
rtn_long = rtn_long[["date", "symbol", "y"]].dropna(subset=["y"])
rtn_long["y"] = rtn_long["y"].clip(-0.2, 0.2)

merged = meta[["available_date", "symbol"]].merge(
    rtn_long, left_on=["available_date", "symbol"], right_on=["date", "symbol"], how="inner")
merged = merged[["available_date", "symbol", "y"]].reset_index(drop=True)
merged["year"] = merged["available_date"].str[:4]

# 只用已完成的年份
PROBE_YEARS = {"2020", "2021"}
_keep = meta["available_date"].str[:4].isin(PROBE_YEARS).values
mean_cls = mean_cls[_keep]; meta = meta[_keep].reset_index(drop=True)
prob_H = prob_H[_keep]; prob_L = prob_L[_keep]; prob_full = prob_full[_keep]; mean_prob = mean_prob[_keep]
_keep2 = merged["available_date"].str[:4].isin(PROBE_YEARS).values
merged = merged[_keep2].reset_index(drop=True)
# 重新对齐: merged 和 meta/prob_* 应该有相同的行
# 但 merged 是 inner join 后的, 可能比 meta 短。用 merged 的 (date, symbol) 做 mask
key = merged["available_date"] + "_" + merged["symbol"]
meta_key = meta["available_date"] + "_" + meta["symbol"]
mask = meta_key.isin(set(key)).values
mean_cls = mean_cls[mask]; prob_H = prob_H[mask]; prob_L = prob_L[mask]
prob_full = prob_full[mask]; mean_prob = mean_prob[mask]
meta = meta[mask].reset_index(drop=True)
# 现在用 merged 的 y 和 meta 的行做对齐
y_df = merged[["available_date", "symbol", "y"]].copy()
assert len(y_df) == len(meta), f"length mismatch: {len(y_df)} vs {len(meta)}"
y_all = y_df["y"].values.astype(np.float64)
dates_all = y_df["available_date"].values

years = sorted(y_df["available_date"].str[:4].unique())
MIN_TEST_DAYS = 20
folds = [(years[:i], years[i]) for i in range(1, len(years))
         if (y_df["available_date"].str[:4] == years[i]).sum() >= MIN_TEST_DAYS]
idx_all = np.arange(len(y_df))
print(f"walk-forward: {len(folds)} folds, years={years}")

def ric(yp, yt, d):
    ics = []
    for dt in np.unique(d):
        m = d == dt
        if m.sum() >= 10:
            ics.append(spearmanr(yp[m], yt[m])[0])
    return np.array(ics)
def safe_icir(a):
    s = a.std()
    return a.mean()/s if s > 1e-6 else 0

# --- 直接 IC (不需要 Ridge!) ---
print("\n=== 直接 IC (fc head 子空间读出) ===")
for name, sig in [("prob_H", prob_H), ("prob_L", prob_L), ("prob_full", prob_full), ("mean_prob", mean_prob)]:
    ics = []
    for tr, te in folds:
        ei = idx_all[y_df["available_date"].str[:4].isin([te]).values]
        ics.append(ric(sig[ei], y_all[ei], dates_all[ei]))
    all_ics = np.concatenate(ics)
    print(f"  {name:12s}: IC={all_ics.mean():.4f} ICIR={safe_icir(all_ics):.2f} n={len(all_ics)}")''', "returns-and-ic")

add_cell(r'''# 3. 残差 IC: L 在 H 之外的增量
print("=== 残差 IC (L|H) ===")
# H 信号去除: 用 OLS beta (train 拟合, test 预测)
res_ics = []
for tr, te in folds:
    ti = idx_all[y_df["available_date"].str[:4].isin(tr).values]
    ei = idx_all[y_df["available_date"].str[:4].isin([te]).values]
    # OLS: y = beta * prob_H (train)
    beta = np.dot(prob_H[ti], y_all[ti]) / np.dot(prob_H[ti], prob_H[ti])
    # 残差
    res_train = y_all[ti] - beta * prob_H[ti]
    res_test = y_all[ei] - beta * prob_H[ei]
    # L 能预测残差吗? (直接 RankIC, 不需要 Ridge)
    ics = ric(prob_L[ei], res_test, dates_all[ei])
    res_ics.append(ics)
    print(f"  fold {tr}->{te}: beta={beta:.3f} ResIC={ics.mean():.4f}")
all_res = np.concatenate(res_ics)
res_mean = all_res.mean()
res_icir = safe_icir(all_res)
print(f"\nResIC_{{L|H}} = {res_mean:.4f}  ICIR={res_icir:.2f}  n={len(all_res)}")''', "residual-ic")

add_cell(r'''# 4. 随机基线 + 显著性
print("=== 随机基线 (20 个随机 451 维子空间) ===")
rand_ics = []
HIDDEN = 768
for seed in range(20):
    rng = np.random.default_rng(seed)
    Qr = np.linalg.qr(rng.standard_normal((HIDDEN, 451)).astype(np.float64))[0]
    cls_rand = mean_cls @ Qr @ Qr.T   # [N, 768]
    logits_rand = cls_rand @ W.T + b
    prob_rand = softmax(logits_rand, axis=1)[:, 1]
    # 残差 IC: random subspace vs H 残差
    fm = []
    for tr, te in folds:
        ti = idx_all[y_df["available_date"].str[:4].isin(tr).values]
        ei = idx_all[y_df["available_date"].str[:4].isin([te]).values]
        beta = np.dot(prob_H[ti], y_all[ti]) / np.dot(prob_H[ti], prob_H[ti])
        res_test = y_all[ei] - beta * prob_H[ei]
        fm.append(ric(prob_rand[ei], res_test, dates_all[ei]).mean())
    rand_ics.append(np.mean(fm))
    if (seed+1) % 5 == 0:
        print(f"  random {seed+1}/20: {np.mean(fm):.4f}")
rand_m, rand_s = np.mean(rand_ics), np.std(rand_ics)
sig = "YES" if res_mean > rand_m + 2*rand_s else ("~" if res_mean > rand_m else "no")
print(f"\nRandom: {rand_m:.4f} +/- {rand_s:.4f} (2sigma={rand_m+2*rand_s:.4f})")
print(f"ResIC_L|H = {res_mean:.4f}  -> {sig}")''', "random-baseline")

add_cell(r'''# 5. 汇总 + 保存 + 图
import matplotlib.pyplot as plt

# 汇总表
rows = []
for name, sig in [("prob_H", prob_H), ("prob_L", prob_L), ("prob_full", prob_full), ("mean_prob", mean_prob)]:
    ics = []
    for tr, te in folds:
        ei = idx_all[y_df["available_date"].str[:4].isin([te]).values]
        ics.append(ric(sig[ei], y_all[ei], dates_all[ei]))
    a = np.concatenate(ics)
    rows.append({"signal": name, "IC": a.mean(), "ICIR": safe_icir(a), "n_days": len(a)})
rows.append({"signal": "ResIC_{L|H}", "IC": res_mean, "ICIR": res_icir, "n_days": len(all_res)})
rows.append({"signal": "Random(mean)", "IC": rand_m, "ICIR": np.nan, "n_days": np.nan})
rows.append({"signal": "Random(std)", "IC": rand_s, "ICIR": np.nan, "n_days": np.nan})
rows.append({"signal": "significant", "IC": 1 if sig=="YES" else 0, "ICIR": np.nan, "n_days": np.nan})
summary = pd.DataFrame(rows)
print("=== FC-Head Subspace Probe Summary ===")
print(summary.to_string(index=False))

OUT = os.path.join(ROOT, "artifacts", "gubapost_cls", "fc_head_subspace_probe_results.parquet")
summary.to_parquet(OUT, index=False)
print(f"\n已保存: {OUT}")

# 图
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 左: 直接 IC 对比
ax = axes[0]
names = ["prob_H\n(high-cov)", "prob_L\n(low-cov)", "prob_full\n(all)", "mean_prob\n(per-post)"]
vals = [summary.loc[summary.signal=="prob_H","IC"].values[0],
        summary.loc[summary.signal=="prob_L","IC"].values[0],
        summary.loc[summary.signal=="prob_full","IC"].values[0],
        summary.loc[summary.signal=="mean_prob","IC"].values[0]]
colors = ["blue", "orange", "green", "gray"]
ax.bar(names, vals, color=colors)
ax.axhline(0, color="black", lw=0.5)
ax.set_ylabel("mean daily Rank IC")
ax.set_title("FC-Head Subspace Direct IC")

# 右: 残差 IC vs 随机
ax = axes[1]
ax.bar(["ResIC\n(L|H)", "Random"], [res_mean, rand_m],
       yerr=[0, rand_s], color=["red", "gray"], capsize=5)
ax.axhline(rand_m + 2*rand_s, color="blue", ls="--", lw=1, label=f"2sigma={rand_m+2*rand_s:.4f}")
ax.axhline(0, color="black", lw=0.5)
ax.set_ylabel("mean daily Rank IC")
ax.set_title(f"Residual IC: L vs Random (significant={sig})")
ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig(os.path.join(ROOT, "artifacts", "gubapost_cls", "fc_head_subspace_probe.png"), dpi=150)
plt.show()''', "summary-and-plot")

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
