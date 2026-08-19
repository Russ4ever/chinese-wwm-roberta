# -*- coding: utf-8 -*-
"""
研报盈利预测因子 中性化 + RankIC 评测
=======================================

对 forecast_stk_20240101_20241231.jsonl 中的
    FORECAST_ROE / FORECAST_OR / FORECAST_NP / FORECAST_EPS
做「MAD 去极值 -> 行业+市值中性化 -> MAD 标准化」，
再用 RTN_daily 的 rtn_{1,5,10,20}d 分别计算截面 rank_ic（逐日 Spearman）。

对 GG_RATING_CODE（有序评级 1卖出 < 3中性 < 5收集 < 7买入，0=无）：
    按 (date,stock) 取多数评级（众数，平票取更高档），0 视为无信号剔除，
    同样计算 rank_ic，并额外输出评级分组收益 + 多空价差。

运行环境：/home/intern_fjq_2026/miniconda3/envs/intern_fjq/bin/python
运行命令：
    cd /home/intern_fjq_2026/Projects/chinese-wwm-roberta/artifacts/label_engineering
    /home/intern_fjq_2026/miniconda3/envs/intern_fjq/bin/python eval_forecast_factor.py
"""

import json
import os

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

# ----------------------------------------------------------------------------
# 配置
# ----------------------------------------------------------------------------
DATA = "/home/intern_fjq_2026/data"
BARRA = "/home/intern_fjq_2026/barra"
OUT = "/home/intern_fjq_2026/Projects/chinese-wwm-roberta/artifacts/label_engineering"

JSONL = os.path.join(DATA, "NLP/research_report/forecast_stk_20240101_20241231.jsonl")
RTN_DIR = os.path.join(DATA, "RTN_daily")
IND_H5 = os.path.join(BARRA, "Barra_Industry.h5")
SIZE_H5 = os.path.join(BARRA, "Barra_SIZE.h5")

RTNS = ["rtn_1d", "rtn_5d", "rtn_10d", "rtn_20d"]
FACTORS = ["FORECAST_ROE", "FORECAST_OR", "FORECAST_NP", "FORECAST_EPS"]
FACTOR_OUT = {
    "FORECAST_ROE": "roe",
    "FORECAST_OR": "or_",
    "FORECAST_NP": "np",
    "FORECAST_EPS": "eps",
}

# year_mode: all | fy0 | fy1 | fy2
#   all = 不筛预测年份，按 (date,stock) 中位数聚合（分析师共识口径，默认）
#   fy0 = 只保留 REPORT_YEAR == CREATE_DATE 年份
#   fy1 = 只保留 REPORT_YEAR == CREATE_DATE 年份 + 1（下一财年）
#   fy2 = 只保留 REPORT_YEAR == CREATE_DATE 年份 + 2
YEAR_MODE = "all"

MAD_N_SIGMA = 3.0       # 去极值：median ± 3 * 1.4826 * MAD
MIN_NEUT_OBS = 50       # 当日截面样本少于该值不做中性化（该日记 NaN）
MIN_IC_OBS = 20         # 当日配对样本少于该值不算 IC
MIN_IND_MEMBERS = 5     # 行业当日成员数少于该值并入 OTHER
START, END = "2024-01-01", "2024-12-31"

RATING_NONE = 0         # GG_RATING_CODE == 0 表示「无」，视为无信号
RATING_NAME = {1: "卖出", 3: "中性", 5: "收集", 7: "买入"}

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 50)


# ----------------------------------------------------------------------------
# 工具
# ----------------------------------------------------------------------------
def log(msg):
    print(f"[eval] {msg}", flush=True)


def read_jsonl(path, usecols):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            rows.append({k: obj.get(k) for k in usecols})
    return pd.DataFrame(rows)


def mad_winsor(x):
    """MAD 去极值：median ± N_SIGMA * 1.4826 * MAD 截断。mad==0 返回原值。"""
    x = np.asarray(x, dtype=float)
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med))
    if (not np.isfinite(mad)) or mad == 0:
        return x
    sigma = 1.4826 * mad
    return np.clip(x, med - MAD_N_SIGMA * sigma, med + MAD_N_SIGMA * sigma)


def neutralize_group(y, ind, size):
    """截面 OLS 残差：y ~ [1, 行业 one-hot, SIZE]，返回残差。

    行业 dummy 与截距共线不影响最小二乘投影残差，直接用 lstsq 残差。
    成员数 < MIN_IND_MEMBERS 的行业并入 OTHER（-1）。
    """
    y = np.asarray(y, dtype=float)
    ind = np.asarray(ind)
    size = np.asarray(size, dtype=float)
    n = len(y)

    codes, cnt = np.unique(ind, return_counts=True)
    small = set(codes[cnt < MIN_IND_MEMBERS].tolist())
    ind_work = np.array([-1 if c in small else c for c in ind])

    cols = [np.ones(n)]
    for c in np.unique(ind_work):
        cols.append((ind_work == c).astype(float))
    cols.append(size)
    X = np.column_stack(cols)

    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    return y - X @ coef


def mad_standardize(resid):
    """MAD 标准化：z = (resid - median) / (1.4826 * MAD)。mad==0 退化为去中心。"""
    resid = np.asarray(resid, dtype=float)
    med = np.nanmedian(resid)
    mad = np.nanmedian(np.abs(resid - med))
    if (not np.isfinite(mad)) or mad == 0:
        return resid - med
    return (resid - med) / (1.4826 * mad)


def neutralize_factor(df, col):
    """对 df 的 col 列做 去极值->中性化->标准化，返回与 df 索引对齐的 Series。"""
    out = pd.Series(np.nan, index=df.index, dtype=float)
    for date, g in df.groupby("date"):
        sub = g[[col, "industry", "size"]]
        valid = sub[col].notna() & sub["industry"].notna() & sub["size"].notna()
        idx = sub.index[valid]
        if valid.sum() < MIN_NEUT_OBS:
            continue
        y = mad_winsor(sub.loc[idx, col].to_numpy(float))
        ind = sub.loc[idx, "industry"].to_numpy()
        size = sub.loc[idx, "size"].to_numpy(float)
        resid = neutralize_group(y, ind, size)
        out.loc[idx] = mad_standardize(resid)
    return out


def summarize_ic(ics):
    """由逐日 IC 数组（含 NaN）聚合成统计量。"""
    m = ics[np.isfinite(ics)]
    n = len(m)
    if n == 0:
        return dict(ic_mean=np.nan, ic_std=np.nan, icir=np.nan,
                    t_stat=np.nan, ic_gt0_pct=np.nan, n_days=0)
    mean = float(np.mean(m))
    std = float(np.std(m, ddof=1))
    return dict(
        ic_mean=mean,
        ic_std=std,
        icir=(mean / std) if std > 0 else np.nan,
        t_stat=(mean / std * np.sqrt(n)) if std > 0 else np.nan,
        ic_gt0_pct=float((m > 0).mean()),
        n_days=n,
    )


def compute_ic_table(panel, eval_cols, rtn_cols):
    """单遍逐日计算所有 (因子 × 收益档) 的 rank_ic，返回 (summary, timeseries)。"""
    ts_rows = []
    for date, g in panel.groupby("date"):
        for col in eval_cols:
            for r in rtn_cols:
                sub = g[[col, r]].dropna()
                n = len(sub)
                if n < MIN_IC_OBS:
                    ic = np.nan
                else:
                    ic = spearmanr(sub[col], sub[r]).correlation
                    ic = float(ic) if np.isfinite(ic) else np.nan
                ts_rows.append(dict(date=date, factor=col, horizon=r, ic=ic, n_obs=n))
    ts = pd.DataFrame(ts_rows)

    rows = []
    for (col, r), grp in ts.groupby(["factor", "horizon"]):
        s = summarize_ic(grp["ic"].to_numpy(float))
        s["factor"] = col
        s["horizon"] = r
        valid_obs = grp.loc[grp["n_obs"] >= MIN_IC_OBS, "n_obs"]
        s["mean_n_obs"] = float(valid_obs.mean()) if len(valid_obs) else np.nan
        rows.append(s)
    summary = pd.DataFrame(rows)
    summary = summary[["factor", "horizon", "ic_mean", "ic_std", "icir",
                       "t_stat", "ic_gt0_pct", "n_days", "mean_n_obs"]]
    return summary, ts


def rating_consensus(s):
    """众数评级，平票取更高档；0(无)/空 剔除。"""
    s = s.dropna()
    s = s[s != RATING_NONE]
    if len(s) == 0:
        return np.nan
    vc = s.value_counts()
    top = vc.values[0]
    return int(vc[vc.values == top].index.max())


# ----------------------------------------------------------------------------
# 主流程
# ----------------------------------------------------------------------------
def main():
    os.makedirs(OUT, exist_ok=True)

    # 1. 读研报预测 jsonl
    log("读取研报预测 jsonl ...")
    usecols = ["CREATE_DATE", "STOCK_CODE", "REPORT_YEAR"] + FACTORS + ["GG_RATING_CODE"]
    fc = read_jsonl(JSONL, usecols)
    log(f"  rows = {len(fc)}")

    fc["CREATE_DATE"] = pd.to_datetime(fc["CREATE_DATE"], errors="coerce")
    fc["STOCK_CODE"] = fc["STOCK_CODE"].astype(str).str.zfill(6)
    fc["REPORT_YEAR"] = pd.to_numeric(fc["REPORT_YEAR"], errors="coerce")
    fc["GG_RATING_CODE"] = pd.to_numeric(fc["GG_RATING_CODE"], errors="coerce")
    for c in FACTORS:
        fc[c] = pd.to_numeric(fc[c], errors="coerce")
    fc = fc[fc["CREATE_DATE"].notna()]

    # 2. year_mode 预测年份筛选
    if YEAR_MODE != "all":
        delta = {"fy0": 0, "fy1": 1, "fy2": 2}[YEAR_MODE]
        target_year = fc["CREATE_DATE"].dt.year + delta
        fc = fc[fc["REPORT_YEAR"] == target_year]
        log(f"  year_mode={YEAR_MODE} -> rows = {len(fc)}")

    # 3. 读 rtn 宽表，取股票列名集合 / 交易日序列
    log("读取 RTN 宽表 ...")
    rtn_frames = {}
    trade_dates = None
    for f in RTNS:
        d = pd.read_parquet(os.path.join(RTN_DIR, f + ".parquet"), engine="pyarrow")
        if trade_dates is None:
            trade_dates = pd.to_datetime(d["date"]).sort_values().values
        rtn_frames[f] = d.set_index("date")
    stock_cols = [c for c in rtn_frames[RTNS[0]].columns if c != "date"]

    # 4. STOCK_CODE -> .SZ/.SH（以 RTN 列名集合为准）
    valid = set(stock_cols)

    def to_code(c):
        if c + ".SZ" in valid:
            return c + ".SZ"
        if c + ".SH" in valid:
            return c + ".SH"
        return None

    fc["stock"] = fc["STOCK_CODE"].map(to_code)
    log(f"  映射失败(丢弃): {int(fc['stock'].isna().sum())} / {len(fc)}")
    fc = fc[fc["stock"].notna()]

    # 5. CREATE_DATE asof 对齐到 >= 的最近交易日
    pos = np.searchsorted(trade_dates, fc["CREATE_DATE"].values, side="left")
    pos = np.clip(pos, 0, len(trade_dates) - 1)
    fc["date"] = pd.to_datetime(trade_dates[pos])
    fc = fc[(fc["date"] >= START) & (fc["date"] <= END)]
    log(f"  对齐后 rows = {len(fc)}")

    # 6. (date, stock) 聚合：数值因子中位数，评级众数
    log("聚合到 (date, stock) ...")
    g = fc.groupby(["date", "stock"])
    panel = g[FACTORS].median().rename(columns=FACTOR_OUT)
    panel["rating"] = g["GG_RATING_CODE"].agg(rating_consensus)
    panel = panel.reset_index()
    log(f"  panel rows = {len(panel)}")

    # 7. 并入 4 档未来收益（stack 成长表）
    log("并入未来收益 ...")
    rtn_long = None
    for f in RTNS:
        d = rtn_frames[f]
        d = d.loc[(d.index >= pd.Timestamp(START)) & (d.index <= pd.Timestamp(END))]
        s = d.stack(dropna=False).rename(f)
        s.index.names = ["date", "stock"]
        s = s.to_frame()
        rtn_long = s if rtn_long is None else rtn_long.join(s, how="outer")
    rtn_long = rtn_long.reset_index()
    panel = panel.merge(rtn_long, on=["date", "stock"], how="inner")
    log(f"  合并收益后 rows = {len(panel)}")

    # 8. 并入行业 / 市值
    log("读取 Barra 行业 / 市值 ...")

    def read_barra_long(path, name):
        d = pd.read_hdf(path, key="df")
        d = d.loc[(d.index >= pd.Timestamp(START)) & (d.index <= pd.Timestamp(END))]
        s = d.stack(dropna=False).rename(name)
        s.index.names = ["date", "stock"]
        return s.to_frame()

    ind = read_barra_long(IND_H5, "industry").reset_index()
    size = read_barra_long(SIZE_H5, "size").reset_index()
    panel = panel.merge(ind, on=["date", "stock"], how="left")
    panel = panel.merge(size, on=["date", "stock"], how="left")
    log(f"  并入行业/市值后 rows = {len(panel)}")

    # 9. 中性化 4 个数值因子
    for c in FACTORS:
        out = FACTOR_OUT[c] + "_neu"
        log(f"中性化 {c} -> {out} ...")
        panel[out] = neutralize_factor(panel, FACTOR_OUT[c])

    # 10. rank IC：4 个中性化因子 + 评级因子 × 4 档收益
    log("计算 rank_ic ...")
    eval_cols = [FACTOR_OUT[c] + "_neu" for c in FACTORS] + ["rating"]
    summary, ts = compute_ic_table(panel, eval_cols, RTNS)

    # 11. 评级分组收益 + 多空价差
    log("计算评级分组收益 ...")
    grp_rows = []
    for r in RTNS:
        g = panel.dropna(subset=["rating", r])
        vc = g["rating"].value_counts()
        for code, name in RATING_NAME.items():
            sub = g.loc[g["rating"] == code, r]
            grp_rows.append(dict(horizon=r, rating=name, code=code,
                                 n=int(vc.get(code, 0)),
                                 mean_rtn=float(sub.mean()) if len(sub) else np.nan,
                                 std_rtn=float(sub.std()) if len(sub) else np.nan))
    grp = pd.DataFrame(grp_rows)

    ls_rows = []
    for r in RTNS:
        spreads = []
        for date, g in panel.groupby("date"):
            gg = g.dropna(subset=["rating", r])
            l = gg.loc[gg["rating"] == 7, r]
            s_ = gg.loc[gg["rating"] == 5, r]
            if len(l) < 5 or len(s_) < 5:
                spreads.append(np.nan)
            else:
                spreads.append(l.mean() - s_.mean())
        spreads = np.asarray(spreads, dtype=float)
        m = spreads[np.isfinite(spreads)]
        if len(m) == 0:
            ls_rows.append(dict(horizon=r, long="买入(7)", short="收集(5)",
                                n_days=0, mean_spread=np.nan,
                                std_spread=np.nan, t_stat=np.nan))
        else:
            std = float(m.std(ddof=1))
            ls_rows.append(dict(horizon=r, long="买入(7)", short="收集(5)",
                                n_days=len(m), mean_spread=float(m.mean()),
                                std_spread=std,
                                t_stat=(float(m.mean() / std * np.sqrt(len(m))) if std > 0 else np.nan)))
    ls = pd.DataFrame(ls_rows)

    # 12. 落盘
    log("写入输出 ...")
    panel.to_parquet(os.path.join(OUT, "neutralized_factors.parquet"), index=False)
    summary.to_csv(os.path.join(OUT, "rank_ic_summary.csv"), index=False)
    ts.to_csv(os.path.join(OUT, "rank_ic_timeseries.csv"), index=False)
    grp.to_csv(os.path.join(OUT, "rating_group_returns.csv"), index=False)
    ls.to_csv(os.path.join(OUT, "rating_longshort.csv"), index=False)

    print("\n" + "=" * 100)
    print("RankIC 汇总（ROE/OR/NP/EPS 已 MAD 中性化；rating 为有序评级）")
    print("=" * 100)
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\n评级分组收益（pooled）")
    print(grp.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\n评级多空（买入 - 收集，逐日价差）")
    print(ls.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\n输出目录:", OUT)


if __name__ == "__main__":
    main()