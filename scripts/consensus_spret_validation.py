# %% [markdown]
# # 一致预期因子与个股特质收益率关联验证
#
# **目标**：验证 `con_sec_fy12`（个股动态一致预期表）中**全部数值列**与个股特质收益率
# `dy1d_specific_ret_cne6_sw21.SPRET` 之间的关联性（因子-收益检验）。
#
# **数据源**：两表同在通联 `datayes` MySQL（`10.80.139.20:3306`，库 `datayes`），
# 与项目 `src/data/fetch/fetch_news.py` / `src/data/fetch/config.py` 的连接方式一致（pymysql）。
#
# **⚠ 重要坑**：该 MySQL 实例上「字符串日期字面量 / 隐式转换」会得到错误结果（如
# `WHERE REP_FORE_DATE >= '2020-02-03'` 返回 0 行），**日期过滤一律用参数化 `%s` 传
# Python `datetime.date`**。

# %% [markdown]
# ## 1. 表结构与对齐
#
# - `con_sec_fy12`：23 列，键 `(SEC_CODE, REP_FORE_DATE)`，含 18 个数值列 `CON_*`；
#   覆盖 2010-01-04 ~ 2026-08-26，约 1463 万行；`REP_FORE_DATE` 为 `DATE` 类型。
# - `dy1d_specific_ret_cne6_sw21`：5 列，键 `(TICKER_SYMBOL, TRADE_DATE)`，`SPRET`；
#   `TRADE_DATE` 为 `'YYYYMMDD'` 字符串；覆盖 2020-02-03 ~ 2025-03-24，约 597 万行。
# - **重叠窗口** = 2020-02-03 ~ 2025-03-24 = **1248 个交易日**，两表代码均为 6 位数字，可直接对齐。
# - `SPRET` 为**百分数**量纲（= 小数收益 ×100），~86.5% 落在 ±3% 内，min≈-96 / max≈488 为极端值。

# %%
import datetime as dt
import os

import numpy as np
import pandas as pd
import pymysql
from scipy import stats as sps

# 与 fetch_news.py / config.py 一致的 datayes MySQL 连接
DB = dict(
    host="10.80.139.20", port=3306, user="chenrongjun",
    password="40!VEz6QX", database="datayes", charset="utf8mb4",
    connect_timeout=30,
)
conn = pymysql.connect(**DB)


def query(sql, args=None):
    """流式游标执行并返回 DataFrame；日期过滤统一走参数化。"""
    with conn.cursor(pymysql.cursors.SSCursor) as cur:
        cur.execute(sql, args)
        cols = [d[0] for d in cur.description]
        rows = cur.fetchall()
    return pd.DataFrame(rows, columns=cols)


# %%
START = dt.date(2020, 2, 3)   # dy1d SPRET 覆盖起点
END = dt.date(2025, 3, 24)     # dy1d SPRET 覆盖终点

# con_sec_fy12 的 18 个数值列（按 information_schema 顺序）
NUM_COLS = [
    "CON_EPS_FY12", "CON_PROFIT_FY12", "CON_INCOME_FY12", "CON_PE_FY12",
    "CON_PB_FY12", "CON_PS_FY12", "CON_NA_FY12", "CON_ROE_FY12",
    "CON_PROFIT_FY12_YOY", "CON_INCOME_FY12_YOY",
    "CON_PROFIT_FY12_CGR2Y", "CON_INCOME_FY12_CGR2Y",
    "CON_PEG1_FY12", "CON_PSG1_FY12", "CON_PEG2_FY12", "CON_PSG2_FY12",
    "CON_PROFIT_FY12_CHGPCT1Y", "CON_INCOME_FY12_CHGPCT1Y",
]

# DECIMAL -> DOUBLE（+0e0 强制浮点，绕开 MySQL 版本 CAST 差异），服务端完成类型转换
cast_cols = ", ".join(f"({c} + 0e0) AS {c}" for c in NUM_COLS)

cons = query(
    "SELECT SEC_CODE, REP_FORE_DATE, " + cast_cols +
    " FROM con_sec_fy12 WHERE REP_FORE_DATE >= %s AND REP_FORE_DATE <= %s",
    (START, END),
)
print("con_sec_fy12 rows in window:", len(cons))

spret = query(
    "SELECT TICKER_SYMBOL AS SEC_CODE, TRADE_DATE, (SPRET + 0e0) AS SPRET "
    "FROM dy1d_specific_ret_cne6_sw21"
)
print("dy1d_specific_ret_cne6_sw21 rows:", len(spret))


# %% [markdown]
# ## 2. 对齐与工具函数
#
# - `TRADE_DATE` 由 `'YYYYMMDD'` 字符串转 `date`；`REP_FORE_DATE` 由 datetime 转 `date`。
# - `SPRET`（百分数）除以 100 转小数收益，再透视成 `(date × stock)` 宽表。
# - **截面 Rank IC**（Spearman）：先逐日对因子/收益横截面求秩，再做去均值 Pearson；
#   秩相关对量纲与极端值天然稳健（因此 SPRET 百分数 vs 小数并不改变结果，仅 Pearson 需缩尾）。

# %%
# ---- 对齐 ----
cons["REP_FORE_DATE"] = pd.to_datetime(cons["REP_FORE_DATE"]).dt.date
spret["TRADE_DATE"] = pd.to_datetime(spret["TRADE_DATE"], format="%Y%m%d").dt.date

spret_w = (
    spret.pivot_table(index="TRADE_DATE", columns="SEC_CODE",
                      values="SPRET", aggfunc="mean")
    .sort_index() / 100.0
)
print("spret panel:", spret_w.shape,
      "| dates:", spret_w.index.min(), "->", spret_w.index.max())

MIN_OBS = 30  # 每日截面至少 30 只有效样本才计入当日 IC


def cross_sectional_ic(F, S, method="spearman"):
    """逐日截面相关系数；F/S 为 (date x stock)。spearman = 先秩后 Pearson。"""
    F = F.astype("float64")
    S = S.astype("float64")
    if method == "spearman":
        F = F.rank(axis=1)
        S = S.rank(axis=1)
    Fc = F.sub(F.mean(axis=1), axis=0)
    Sc = S.sub(S.mean(axis=1), axis=0)
    num = (Fc * Sc).sum(axis=1, min_count=1)
    den = np.sqrt((Fc ** 2).sum(axis=1) * (Sc ** 2).sum(axis=1))
    return (num / den).replace([np.inf, -np.inf], np.nan)


def winsorize_cs(df, lo=0.01, hi=0.99):
    """逐日截面 1%/99% 缩尾（仅 Pearson IC 使用；秩相关本身稳健）。"""
    return df.apply(lambda r: r.clip(r.quantile(lo), r.quantile(hi)), axis=1)


def ic_summary(ic):
    """IC 时序 → 汇总指标。t = mean/(std/sqrt(N))。"""
    s = ic.dropna()
    mean, std = s.mean(), s.std(ddof=1)
    return {
        "mean_ic": mean,
        "std_ic": std,
        "icir": mean / std if std > 0 else np.nan,
        "t_stat": mean / (std / np.sqrt(len(s))) if (std > 0 and len(s)) else np.nan,
        "ic_pos_ratio": float((s > 0).mean()),
        "n_days": len(s),
    }


def factor_panel(col):
    """单个数值列透视成 (date x stock)，并与 spret 内联结对齐。"""
    f = cons.pivot_table(index="REP_FORE_DATE", columns="SEC_CODE",
                         values=col, aggfunc="mean").sort_index()
    f, s = f.align(spret_w, join="inner")
    return f, s


def daily_ic(F, R, method="spearman", min_obs=MIN_OBS):
    """计算逐日截面 IC 并对样本不足的日期置 NaN。"""
    mask = F.notna() & R.notna()
    F = F.where(mask)
    R = R.where(mask)
    ok = mask.sum(axis=1) >= min_obs
    return cross_sectional_ic(F, R, method).where(ok)


def forward_return(r, h):
    """未来 h 日累计特质收益（小数、复合），区间 t+1..t+h，用于预测性检验。"""
    logr = np.log1p(r.clip(lower=-1 + 1e-9))
    cum = logr.cumsum()
    return np.expm1(cum.shift(-h) - cum)


# %% [markdown]
# ## 3. 验证一：当期截面关联（concurrent）
#
# 对每个数值列：`corr(因子_t, SPRET_t)`，逐日截面，均值/ICIR/t/IC>0 占比。
# 同时给 Pearson IC（横截面 1%/99% 缩尾后）作为对照。

# %%
concurrent = []
for col in NUM_COLS:
    f, s = factor_panel(col)
    ic_sp = daily_ic(f, s, "spearman")
    ic_pe = daily_ic(winsorize_cs(f), winsorize_cs(s), "pearson")
    rec = {"factor": col}
    rec.update({f"rc_{k}": v for k, v in ic_summary(ic_sp).items()})
    rec.update({f"pc_{k}": v for k, v in ic_summary(ic_pe).items()})
    rec["avg_n_stocks"] = float((f.notna() & s.notna()).sum(axis=1).mean())
    concurrent.append(rec)

concurrent_df = pd.DataFrame(concurrent).set_index("factor")
concurrent_df = concurrent_df.sort_values("rc_t_stat", ascending=False)
print(concurrent_df.round(4).to_string())


# %% [markdown]
# ## 4. 验证二：前瞻收益关联（预测性）
#
# `corr(因子_t, 未来 h 日累计特质收益)`，h = 1 / 5 / 20。
# 这是更干净的点-in-time 检验：用 t 日已有的预期值预测 t+1..t+h 的特质收益。

# %%
HORIZONS = [1, 5, 20]
fwd = {h: {} for h in HORIZONS}
for h in HORIZONS:
    fr = forward_return(spret_w, h)
    for col in NUM_COLS:
        f, s = factor_panel(col)
        s_fwd = fr.reindex(index=s.index, columns=s.columns)
        ic = daily_ic(f, s_fwd, "spearman")
        fwd[h][col] = ic_summary(ic)

fwd_records = [
    {"factor": col, "horizon": h, "metric": k, "value": fwd[h][col][k]}
    for h in HORIZONS for col in NUM_COLS for k in ("mean_ic", "t_stat", "icir")
]
fwd_df = pd.DataFrame(fwd_records)
fwd_wide = fwd_df.pivot_table(index="factor", columns=["horizon", "metric"],
                              values="value").sort_index(axis=1)
print(fwd_wide.round(4).to_string())


# %% [markdown]
# ## 5. 验证三：分组单调性
#
# 逐日截面按因子值分 10 组，考察未来 1 日特质收益的组均值是否随分组单调：
# 输出 第1组 / 第10组 / 价差 / 组序与收益的 Spearman（单调性）。

# %%
def grouped_return(col, h=1, q=10):
    f, s = factor_panel(col)
    fr = forward_return(spret_w, h).reindex(index=s.index, columns=s.columns)
    mask = f.notna() & fr.notna()
    f = f.where(mask)
    fr = fr.where(mask)
    g = np.ceil(f.rank(axis=1, pct=True) * q)
    df = pd.DataFrame({"g": g.stack(dropna=False),
                       "r": fr.stack(dropna=False)}).dropna()
    means = df.groupby("g")["r"].mean()
    means.index = means.index.astype(int)
    rho = float(sps.spearmanr(df["g"], df["r"])[0])
    return means, rho


mono = []
for col in NUM_COLS:
    means, rho = grouped_return(col, h=1, q=10)
    mono.append({
        "factor": col,
        "g1_ret": means.get(1, np.nan),
        "g10_ret": means.get(10, np.nan),
        "spread_g10_g1": means.get(10, np.nan) - means.get(1, np.nan),
        "monotonic_rho": rho,
    })
mono_df = pd.DataFrame(mono).set_index("factor")
print(mono_df.round(6).to_string())


# %% [markdown]
# ## 6. 汇总与解读
#
# - **研判标准**：`|mean_ic| >= 0.03` 具经济意义；`|t_stat| > 2` 统计显著。
# - **分组解读**：
#   - 估值类（PE/PB/PS/NA/PEG/PSG）：低估值 → 高未来收益，通常与收益**负**相关。
#   - 成长类（YOY/CGR2Y/CHGPCT1Y）：高成长 → 高未来收益，通常**正**相关。
#   - 盈利类（EPS/PROFIT/INCOME/ROE）：盈利越强越优，通常**正**相关。
# - 特质收益已被 CNE6 + SW21 中性化（行业/风格/市值等），无需再中性化。

# %%
out_dir = "/home/intern_fjq_2026/Projects/chinese-wwm-roberta/artifacts/consensus_spret_validation"
os.makedirs(out_dir, exist_ok=True)
concurrent_df.to_csv(f"{out_dir}/concurrent_ic.csv")
fwd_wide.to_csv(f"{out_dir}/forward_ic.csv")
mono_df.to_csv(f"{out_dir}/quantile_monotonicity.csv")
print("saved to", out_dir)

fwd_flat = fwd_wide.copy()
fwd_flat.columns = [f"fwd{h}_{m}" for h, m in fwd_wide.columns]
master = concurrent_df.join(fwd_flat).join(mono_df)
master.to_csv(f"{out_dir}/master_summary.csv")
print("\n==== MASTER SUMMARY (排名按 当期 Rank IC t 值) ====")
print(master.round(4).to_string())

conn.close()
print("\nDONE")