"""Regenerate coverage_alpha_probe and dead_residual_alpha_probe with rolling folds + OOS split."""
import json
from pathlib import Path

def make_nb(cells):
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "nlp_fjq", "language": "python", "name": "nlp_fjq"},
            "language_info": {"name": "python", "version": "3.10.18"},
        },
        "nbformat": 4, "nbformat_minor": 5,
    }

def md(source, cid):
    lines = source.strip().split("\n")
    return {"cell_type": "markdown", "id": cid, "metadata": {}, "source": [l + "\n" for l in lines]}

def code(source, cid):
    lines = source.strip().split("\n")
    src = [l + "\n" for l in lines[:-1]] + [lines[-1]]
    return {"cell_type": "code", "id": cid, "metadata": {}, "execution_count": None, "outputs": [], "source": src}

# =========================================================================
# Shared code: load + split + folds + h_cache
# =========================================================================
LOAD_SPLIT = r'''# 1. 加载 + 投影 + 收益 + 滚动切分 + h_cache
import os, sys, time, glob, numpy as np, pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold

ROOT = "/home/intern_fjq_2026/Projects/chinese-wwm-roberta"
os.chdir(ROOT); sys.path.insert(0, ROOT)

# --- 加载 per_file (全量, 不按年份过滤) ---
per_dir = os.path.join(ROOT, "artifacts", "gubapost_cls", "per_file")
pf_files = sorted(glob.glob(os.path.join(per_dir, "*.parquet")))
assert pf_files, "per_file/ 下没有文件"
print(f"加载 {len(pf_files)} 个 per-file parquet...")
NEED_COLS = ["available_date", "symbol", "n_posts", "sum_cls"]
dfs = [pd.read_parquet(f, columns=NEED_COLS) for f in pf_files]
pf = pd.concat(dfs, ignore_index=True); del dfs
nposts = pf["n_posts"].values.astype(np.float64)
nposts_safe = np.where(nposts > 0, nposts, 1.0)
cls = np.stack(pf["sum_cls"].values).astype(np.float32)
cls_df = pf[["available_date", "symbol", "n_posts"]].copy()
del pf
print(f"CLS: {cls.shape} | dates: {cls_df.available_date.min()}~{cls_df.available_date.max()}")

# --- 收益 (前移1天, winsorize) ---
rtn = pd.read_parquet("/home/intern_fjq_2026/data/RTN_daily/rtn_1d.parquet")
rtn_long = rtn.melt(id_vars="date", var_name="sym", value_name="r")
rtn_long["symbol"] = rtn_long["sym"].str.split(".").str[0]
rtn_long["date"] = pd.to_datetime(rtn_long["date"]).dt.strftime("%Y-%m-%d")
rtn_long = rtn_long.sort_values(["symbol", "date"])
rtn_long["y"] = rtn_long.groupby("symbol")["r"].shift(-1)
rtn_long = rtn_long[["date", "symbol", "y"]].dropna(subset=["y"])
rtn_long["y"] = rtn_long["y"].clip(-0.2, 0.2)

merged = cls_df[["available_date", "symbol"]].merge(
    rtn_long, left_on=["available_date", "symbol"], right_on=["date", "symbol"], how="inner")
merged = merged[["available_date", "symbol", "y"]].reset_index(drop=True)
merged["ym"] = merged["available_date"].str[:7]

# 对齐 features
meta_key = cls_df["available_date"] + "_" + cls_df["symbol"]
merged_key = merged["available_date"] + "_" + merged["symbol"]
mask = meta_key.isin(set(merged_key)).values
cls = cls[mask]
nposts_aligned = nposts_safe[mask]
assert len(cls) == len(merged), f"{len(cls)} != {len(merged)}"

y_all = merged["y"].values.astype(np.float64)
dates_all = merged["available_date"].values
idx_all = np.arange(len(merged))

# --- 滚动切分: in-sample (< 2024-06) + OOS (>= 2024-06) ---
OOS_START = "2024-06"
is_oos = merged["ym"] >= OOS_START
is_insample = ~is_oos
insample_idx = idx_all[is_insample.values]
oos_idx = idx_all[is_oos.values]

def ym_to_half(ym):
    y, m = ym.split("-")
    return y + ("H1" if int(m) <= 6 else "H2")

insample_ym = merged.loc[is_insample, "ym"].values
half_periods = sorted(set(ym_to_half(ym) for ym in insample_ym))
folds = []
for i in range(1, len(half_periods)):
    train_periods = set(half_periods[:i])
    test_period = half_periods[i]
    train_mask = np.array([ym_to_half(ym) in train_periods for ym in insample_ym])
    test_mask = np.array([ym_to_half(ym) == test_period for ym in insample_ym])
    ti = insample_idx[train_mask]
    ei = insample_idx[test_mask]
    if len(ei) >= 20:
        folds.append((ti, ei, test_period))
        print(f"  fold {test_period}: train={len(ti):,} test={len(ei):,}")
print(f"\nIn-sample folds: {len(folds)} | OOS: {len(oos_idx):,} samples (>= {OOS_START})")

# --- helpers ---
ALPHA = 100
def ric(yp, yt, d):
    ics = []
    for dt in np.unique(d):
        m = d == dt
        if m.sum() >= 10:
            ics.append(spearmanr(yp[m], yt[m])[0])
    return np.array(ics)
def rfp(Xtr, ytr, Xte):
    sc = StandardScaler()
    m = Ridge(alpha=ALPHA)
    m.fit(sc.fit_transform(Xtr), ytr)
    return m.predict(sc.transform(Xte))
def safe_icir(a):
    s = a.std()
    return a.mean() / s if s > 1e-6 else 0

# --- H 方向 + h_cache ---
DIRS_PATH = os.path.join(ROOT, "artifacts", "checkpoint_activation_rank", "runs", "gubapost_v1",
                         "extensions", "coverage_ablation_v1", "direction_sets.npz")
dirs_npz = np.load(DIRS_PATH)
H = cls @ dirs_npz["keep_317_complement_K64"].astype(np.float32)
L_cov = cls @ dirs_npz["keep_451_lowcov_K64"].astype(np.float32)
print(f"H {H.shape} L {L_cov.shape}")

def build_h_cache(train_idx, test_idx):
    """5-fold CV 残差 (train) + Ridge 预测残差 (test)."""
    kf = KFold(5, shuffle=False)
    oos = np.full(len(train_idx), np.nan)
    for a, b in kf.split(H[train_idx]):
        sc = StandardScaler(); m = Ridge(alpha=ALPHA)
        m.fit(sc.fit_transform(H[train_idx][a]), y_all[train_idx][a])
        oos[b] = m.predict(sc.transform(H[train_idx][b]))
    train_res = y_all[train_idx] - oos
    ypH = rfp(H[train_idx], y_all[train_idx], H[test_idx])
    test_res = y_all[test_idx] - ypH
    return train_res, test_res, dates_all[test_idx]

# in-sample folds h_cache
h_cache = {}
for fi, (ti, ei, period) in enumerate(folds):
    h_cache[fi] = (*build_h_cache(ti, ei), ti, ei)
    print(f"  h_cache fold {period} done")
# OOS h_cache (train = 全 in-sample, test = OOS)
h_cache_oos = (*build_h_cache(insample_idx, oos_idx), insample_idx, oos_idx)
print("h_cache (in-sample + OOS) 完成")'''

# =========================================================================
# coverage_alpha_probe: Cell 3 + Cell 4
# =========================================================================
COV_CELL3 = r'''# 2. Ridge probe: in-sample folds + OOS
print("=== In-sample folds ===")
probe = {}
for name, feat in [("H_317", H), ("L_451", L_cov)]:
    probe[name] = []
    for fi, (ti, ei, period) in enumerate(folds):
        yp = rfp(feat[ti], y_all[ti], feat[ei])
        ics = ric(yp, y_all[ei], dates_all[ei])
        probe[name].append((period, ics))

for name in ["H_317", "L_451"]:
    a = np.concatenate([r[1] for r in probe[name]])
    print(f"  {name}: IC={a.mean():.4f} ICIR={safe_icir(a):.2f} n={len(a)}")

# In-sample residual IC
res_ics = []
for fi, (ti, ei, period) in enumerate(folds):
    train_res, test_res, test_dates, _, _ = h_cache[fi]
    rp = rfp(L_cov[ti], train_res, L_cov[ei])
    ics = ric(rp, test_res, test_dates)
    res_ics.append((period, ics))
all_res = np.concatenate([r[1] for r in res_ics])
print(f"  ResIC_{{L|H}} (in-sample): {all_res.mean():.4f} ICIR={safe_icir(all_res):.2f}")

print("\n=== OOS (>= 2024-06) ===")
# Direct IC
for name, feat in [("H", H), ("L", L_cov)]:
    yp = rfp(feat[insample_idx], y_all[insample_idx], feat[oos_idx])
    ics = ric(yp, y_all[oos_idx], dates_all[oos_idx])
    print(f"  IC_{name} (OOS): {ics.mean():.4f} ICIR={safe_icir(ics):.2f} n={len(ics)}")

# OOS residual IC
train_res_oos, test_res_oos, oos_dates, _, _ = h_cache_oos
rp_oos = rfp(L_cov[insample_idx], train_res_oos, L_cov[oos_idx])
oos_res_ics = ric(rp_oos, test_res_oos, oos_dates)
oos_res_mean = oos_res_ics.mean()
print(f"  ResIC_{{L|H}} (OOS): {oos_res_mean:.4f} ICIR={safe_icir(oos_res_ics):.2f}")

# Random baseline (OOS only, 50 random 451-dim)
print("\n=== Random baseline (OOS, 50 random) ===")
rand_ics = []
for seed in range(50):
    rng = np.random.default_rng(seed)
    Qr = np.linalg.qr(rng.standard_normal((768, 451)))[0].astype(np.float32)
    R = cls @ Qr
    rp = rfp(R[insample_idx], train_res_oos, R[oos_idx])
    rand_ics.append(ric(rp, test_res_oos, oos_dates).mean())
    if (seed+1) % 10 == 0:
        print(f"  random {seed+1}/50: {np.mean(rand_ics):.4f}")
rand_m, rand_s = np.mean(rand_ics), np.std(rand_ics)
sig = "YES" if oos_res_mean > rand_m + 2*rand_s else ("~" if oos_res_mean > rand_m else "no")
print(f"\nRandom: {rand_m:.4f} +/- {rand_s:.4f} (2sigma={rand_m+2*rand_s:.4f})")
print(f"ResIC_{{L|H}} OOS = {oos_res_mean:.4f} -> {sig}")'''

COV_CELL4 = r'''# 3. 汇总 + 保存 + 图
import matplotlib.pyplot as plt

rows = []
for name in ["H_317", "L_451"]:
    a = np.concatenate([r[1] for r in probe[name]])
    rows.append({"statistic": f"IC_{name}_insample", "IC": a.mean(), "ICIR": safe_icir(a), "n": len(a)})
rows.append({"statistic": "ResIC_{L|H}_insample", "IC": all_res.mean(), "ICIR": safe_icir(all_res), "n": len(all_res)})
rows.append({"statistic": "ResIC_{L|H}_OOS", "IC": oos_res_mean, "ICIR": safe_icir(oos_res_ics), "n": len(oos_res_ics)})
rows.append({"statistic": "Random_OOS_mean", "IC": rand_m, "ICIR": np.nan, "n": np.nan})
rows.append({"statistic": "Random_OOS_std", "IC": rand_s, "ICIR": np.nan, "n": np.nan})
rows.append({"statistic": "significant", "IC": 1 if sig=="YES" else 0, "ICIR": np.nan, "n": np.nan})
summary = pd.DataFrame(rows)
print(summary.to_string(index=False))
summary.to_parquet(os.path.join(ROOT, "artifacts", "gubapost_cls", "coverage_alpha_probe_results.parquet"), index=False)

fig, ax = plt.subplots(figsize=(8, 4))
ax.bar(["ResIC\n(in-sample)", "ResIC\n(OOS)", "Random\n(OOS)"],
       [all_res.mean(), oos_res_mean, rand_m],
       yerr=[0, 0, rand_s], color=["steelblue", "red", "gray"], capsize=5)
ax.axhline(rand_m + 2*rand_s, color="blue", ls="--", lw=1, label=f"2sigma={rand_m+2*rand_s:.4f}")
ax.axhline(0, color="black", lw=0.5)
ax.set_ylabel("mean daily Rank IC")
ax.set_title(f"Coverage Alpha Probe (OOS significant={sig})")
ax.legend(fontsize=8)
plt.tight_layout(); plt.show()'''

# =========================================================================
# dead_residual_alpha_probe: Cell 3 + Cell 4
# =========================================================================
DEAD_LOAD_EXTRA = r'''
# --- dead-128 features (per-layer) ---
N_LAYERS = 12; DEAD_DIM = 128
dead = {}
for L in range(1, N_LAYERS+1):
    dead[L] = np.stack(
        [pd.read_parquet(f, columns=[f"sum_attn_{L:02d}"])[f"sum_attn_{L:02d}"].iloc[0]
         for f in pf_files]
    ).astype(np.float32) if False else None  # placeholder, real load below
del pf'''

# Actually, for dead_residual, we need to load sum_attn columns too.
# Let me rewrite the load part for dead_residual to include sum_attn loading.
DEAD_LOAD_SPLIT = LOAD_SPLIT.replace(
    'NEED_COLS = ["available_date", "symbol", "n_posts", "sum_cls"]',
    'NEED_COLS = ["available_date", "symbol", "n_posts", "sum_cls"] + [f"sum_attn_{L:02d}" for L in range(1, 13)]'
).replace(
    '''cls = np.stack(pf["sum_cls"].values).astype(np.float32)
cls_df = pf[["available_date", "symbol", "n_posts"]].copy()
del pf''',
    '''cls = np.stack(pf["sum_cls"].values).astype(np.float32)
cls_df = pf[["available_date", "symbol", "n_posts"]].copy()
N_LAYERS = 12; DEAD_DIM = 128
dead = {}
for L in range(1, N_LAYERS+1):
    dead[L] = np.stack(pf[f"sum_attn_{L:02d}"].values).astype(np.float32)
print(f"dead-128: {N_LAYERS} layers, shape={dead[1].shape}")
del pf'''
).replace(
    '''cls = cls[mask]
nposts_aligned = nposts_safe[mask]''',
    '''cls = cls[mask]
nposts_aligned = nposts_safe[mask]
for L in range(1, N_LAYERS+1):
    dead[L] = dead[L][mask]'''
)

DEAD_CELL3 = r'''# 2. Per-layer probe: in-sample folds + OOS
layer_results = []
for L in range(1, N_LAYERS+1):
    t0 = time.time()
    r_L = dead[L]

    # In-sample folds IC
    ic_f = []
    for fi, (ti, ei, period) in enumerate(folds):
        _, _, _, _, _ = h_cache[fi]
        yp = rfp(r_L[ti], y_all[ti], r_L[ei])
        ic_f.append(ric(yp, y_all[ei], dates_all[ei]))
    ic_ins = np.concatenate(ic_f).mean()

    # In-sample residual IC
    res_f = []
    for fi, (ti, ei, period) in enumerate(folds):
        train_res, test_res, test_dates, _, _ = h_cache[fi]
        rp = rfp(r_L[ti], train_res, r_L[ei])
        res_f.append(ric(rp, test_res, test_dates))
    res_ic_ins = np.concatenate(res_f).mean()

    # OOS residual IC
    train_res_oos, test_res_oos, oos_dates, _, _ = h_cache_oos
    rp_oos = rfp(r_L[insample_idx], train_res_oos, r_L[oos_idx])
    oos_ics = ric(rp_oos, test_res_oos, oos_dates)
    res_ic_oos = oos_ics.mean()

    # Random baseline (OOS, 20 random 128-dim)
    rng = np.random.default_rng(L)
    rand_means = []
    for s in range(20):
        Qr = np.linalg.qr(rng.standard_normal((768, DEAD_DIM)))[0].astype(np.float32)
        R = cls @ Qr
        rp = rfp(R[insample_idx], train_res_oos, R[oos_idx])
        rand_means.append(ric(rp, test_res_oos, oos_dates).mean())
    rand_m, rand_s = np.mean(rand_means), np.std(rand_means)
    sig = "YES" if res_ic_oos > rand_m + 2*rand_s else ("~" if res_ic_oos > rand_m else "no")

    layer_results.append({
        "layer": f"L{L:02d}", "IC_ins": ic_ins, "ResIC_ins": res_ic_ins,
        "ResIC_oos": res_ic_oos, "Rand_m": rand_m, "Rand_s": rand_s, "sig": sig,
    })
    print(f"L{L:02d}: IC_ins={ic_ins:.4f} ResIC_ins={res_ic_ins:.4f} "
          f"ResIC_oos={res_ic_oos:.4f} Rand={rand_m:.4f}+/-{rand_s:.4f} {sig} ({time.time()-t0:.0f}s)")
    del r_L'''

DEAD_CELL4 = r'''# 3. 汇总 + 保存 + 图
import matplotlib.pyplot as plt
res_df = pd.DataFrame(layer_results)
print("=== Per-Layer Dead Residual Alpha Probe (rolling + OOS) ===")
print(res_df.to_string(index=False))
n_sig = (res_df.sig == "YES").sum()
print(f"\nOOS 显著层数 (ResIC > Rand+2sigma): {n_sig}/{N_LAYERS}")
res_df.to_parquet(os.path.join(ROOT, "artifacts", "gubapost_cls", "dead_residual_alpha_results.parquet"), index=False)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
x = np.arange(N_LAYERS)
colors = ["red" if s=="YES" else ("orange" if s=="~" else "gray") for s in res_df.sig]
axes[0].bar(x, res_df.ResIC_oos, color=colors, label="ResIC OOS")
axes[0].errorbar(x, res_df.Rand_m, yerr=res_df.Rand_s, fmt="none", ecolor="blue", capsize=3, label="Random")
axes[0].set_xticks(x); axes[0].set_xticklabels(res_df.layer, rotation=45)
axes[0].set_ylabel("mean daily Rank IC"); axes[0].set_title("Per-Layer ResIC OOS: Dead 128 vs Random")
axes[0].axhline(0, color="black", lw=0.5); axes[0].legend(fontsize=8)

axes[1].bar(x, res_df.ResIC_ins, color="steelblue", label="ResIC in-sample")
axes[1].bar(x, res_df.ResIC_oos, color="red", alpha=0.5, label="ResIC OOS")
axes[1].set_xticks(x); axes[1].set_xticklabels(res_df.layer, rotation=45)
axes[1].set_title("In-sample vs OOS ResIC"); axes[1].axhline(0, color="black", lw=0.5)
axes[1].legend(fontsize=8)
plt.tight_layout(); plt.show()'''

# =========================================================================
# Write notebooks
# =========================================================================
cov_nb = make_nb([
    md("# Coverage Alpha Probe (滚动训练 + OOS)\n\n全量数据 2020-2026, 2024-06 起为 OOS, 之前滚动半年 fold.", "title"),
    code(LOAD_SPLIT, "load"),
    code(COV_CELL3, "probe"),
    code(COV_CELL4, "summary"),
])

dead_nb = make_nb([
    md("# Per-Layer Dead Residual Alpha Probe (滚动训练 + OOS)\n\n全量数据 2020-2026, 2024-06 起为 OOS, 之前滚动半年 fold.", "title"),
    code(DEAD_LOAD_SPLIT, "load"),
    code(DEAD_CELL3, "probe"),
    code(DEAD_CELL4, "summary"),
])

for path, nb in [("notebooks/coverage_alpha_probe.ipynb", cov_nb),
                  ("notebooks/dead_residual_alpha_probe.ipynb", dead_nb)]:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)
    print(f"Written {path}")
