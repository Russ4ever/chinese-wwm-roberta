# -*- coding: utf-8 -*-
"""
研报"一致预期—预期修正" Label 构造（V2）
依据: consensus_label_V2_完整修改方案.md

V2 核心修复（对应方案 P01~P14）:
  - 分离"数据读取窗口"与"Label 生成窗口"，新增 left/right_censored、future_snapshot_missing
  - 股票代码 zfill(6) 字符串；report_signature 用 sha256；保留 source_report_id
  - 报告级(逐报告) 与 机构日终面板(org_daily) 两条路径，不再把同日多报告取中位数挂第一篇文本
  - 报告发布前共识执行 180 天过期、排除本机构、严格早于发布日
  - 全部正式目标无效时置 NaN/None；_raw 保留诊断值；每个目标独立 valid 标志
  - 近零尺度/共识强度/strong 阈值只用训练期 & 按 forecast_horizon 拟合
  - 普通正盈利五分类 与 盈亏状态迁移 分离
"""
import os
import json
import hashlib
import datetime as dt
import numpy as np
import pandas as pd

# ===========================================================================
# 0. 配置
# ===========================================================================
SRC = "/home/intern_fjq_2026/data/NLP/research_report/forecast_stk_20100101_20250831.jsonl"
BASE = "/home/intern_fjq_2026/Projects/chinese-wwm-roberta/artifacts/label_engineering"
OUT = os.path.join(BASE, "v2")
os.makedirs(OUT, exist_ok=True)

METRIC = "FORECAST_NP"
LOOKBACK_DAYS = 180
TARGET_HORIZONS = [1, 3]

LABEL_START = pd.Timestamp("2024-01-31")
LABEL_END = pd.Timestamp("2024-12-31")
FY_MIN, FY_MAX = 2024, 2026          # FY0/FY1/FY2
SOURCE_START = "2023-07-01"          # 读取下限（180 天回看 + 缓冲）
SOURCE_END = "2025-03-31"            # 读取上限（3m 未来 + 缓冲）
SNAPSHOT_END = pd.Timestamp("2025-03-31")   # 快照需覆盖到未来窗口

TRAIN_END = pd.Timestamp("2024-06-30")      # 阈值只在 asof <= TRAIN_END 上拟合

MIN_ORG = 5
MIN_ORG_PRE = 4
MIN_ORG_FIXED = 3
MIN_ORG_ACTIVE = 3
FLAT = {1: 0.01, 3: 0.02}            # flat 经济阈值
STRONG_FIXED = {1: 0.05, 3: 0.10}    # strong 固定阈值(敏感性对照)
MAD_SCALE = 1.4826
LABEL_VERSION = "v2.0"

KEEP_COLS = ["ID", "STOCK_CODE", "STOCK_NAME", "ORGAN_NAME", "AUTHOR_NAME",
             "TITLE", "CREATE_DATE", "REPORT_YEAR", "REPORT_QUARTER", "FORECAST_NP"]


# ===========================================================================
# 1. 工具函数
# ===========================================================================
def trimmed_mean(x, frac=0.1):
    x = np.sort(np.asarray(x, dtype=float))
    n = len(x)
    k = int(round(n * frac))
    if k == 0 or n - 2 * k <= 0:
        return float(np.mean(x))
    return float(np.mean(x[k:n - k]))


def classify_dir(v, flat_thr):
    if pd.isna(v):
        return None
    if abs(v) <= flat_thr:
        return "flat"
    return "up" if v > 0 else "down"


def classify5(v, flat_thr, strong_thr):
    if pd.isna(v):
        return None
    a = abs(v)
    if a <= flat_thr:
        return "flat"
    if a < strong_thr:
        return "up" if v > 0 else "down"
    return "strong_up" if v > 0 else "strong_down"


def profit_state_of(median, mn, mx, nz_thr):
    if mn < -nz_thr and mx > nz_thr:
        return "mixed_sign"
    if abs(median) <= nz_thr:
        return "near_zero"
    return "profit" if median > 0 else "loss"


def transition_class(a, b, st_t, st_f, cls5):
    if pd.isna(a) or pd.isna(b):
        return None
    if st_t == "profit" and st_f == "profit":
        m = {"flat": "profit_flat", "up": "profit_up", "down": "profit_down",
             "strong_up": "profit_strong_up", "strong_down": "profit_strong_down"}
        return m.get(cls5)
    if st_t == "loss" and st_f == "loss":
        return "loss_narrowing" if b > a else ("loss_widening" if b < a else "loss_unchanged")
    if st_t == "profit" and st_f == "loss":
        return "turn_to_loss"
    if st_t in ("loss", "near_zero", "mixed_sign") and st_f == "profit":
        return "turn_to_profit"
    if "mixed_sign" in (st_t, st_f):
        return "mixed_sign_transition"
    if "near_zero" in (st_t, st_f):
        return "near_zero_transition"
    return "other"


def sha256(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def future_month(t, h):
    return (pd.Timestamp(t).to_period("M") + h).to_timestamp("M")


def _q(series, q):
    return float(series.quantile(q)) if len(series) else np.nan


# ===========================================================================
# 2. 读取（流式，只保留窗口内 & 指定财年 & NP 非空，不加载 CONTENT）
# ===========================================================================
print("[1/6] 流式读取数据 ...")
_rows = []
with open(SRC, "r", encoding="utf-8", errors="replace") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        try:
            o = json.loads(line)
        except Exception:
            continue
        cd = o.get("CREATE_DATE")
        if not isinstance(cd, str):
            continue
        if cd[:10] < SOURCE_START or cd[:10] > SOURCE_END:
            continue
        ry = o.get("REPORT_YEAR")
        if ry is None:
            continue
        ry = float(ry)
        if ry < FY_MIN - 0.5 or ry > FY_MAX + 0.5:
            continue
        if o.get("FORECAST_NP") is None:
            continue
        _rows.append({k: o.get(k) for k in KEEP_COLS})

raw = pd.DataFrame(_rows)
print("    窗口内 NP 记录数:", len(raw))

raw["source_report_id"] = raw["ID"].astype(str)
raw["stock_code"] = raw["STOCK_CODE"].astype(str).str.replace(r"\..*$", "", regex=True).str.zfill(6)
raw["stock_name"] = raw["STOCK_NAME"].astype(str)
raw["org_id"] = raw["ORGAN_NAME"].astype(str).str.strip()
raw["org_name"] = raw["ORGAN_NAME"].astype(str)
raw["author_name"] = raw["AUTHOR_NAME"].fillna("").astype(str).str.strip()
raw["title"] = raw["TITLE"].astype(str).str.strip()
raw["publish_date"] = pd.to_datetime(raw["CREATE_DATE"])
raw["publish_time"] = raw["CREATE_DATE"].str[11:16]
raw["fy"] = raw["REPORT_YEAR"].astype(int)
raw["forecast"] = raw["FORECAST_NP"].astype(float)

_sig = (raw["stock_code"] + "|" + raw["org_id"] + "|" + raw["author_name"] + "|" +
        raw["publish_date"].dt.strftime("%Y-%m-%d") + "|" + raw["title"])
raw["report_signature"] = _sig.map(sha256)
raw = raw.drop_duplicates(subset=["source_report_id"]).reset_index(drop=True)

# ---- 报告级数据（不聚合，一篇报告=一条报告级记录 × FY）----
report = raw.copy()

# ---- 机构日终面板（同机构同日同股票同FY 聚合为日终值）----
org_daily = (raw.groupby(["stock_code", "fy", "org_id", "publish_date"], as_index=False)
             .agg(forecast=("forecast", "median"),
                  n_reports_same_org_day=("forecast", "size"),
                  same_day_value_range=("forecast", lambda x: float(np.max(x) - np.min(x)))))
org_daily["org_day_aggregate"] = (org_daily["n_reports_same_org_day"] > 1).astype(int)
org_daily = org_daily.sort_values(["stock_code", "fy", "org_id", "publish_date"]).reset_index(drop=True)
print("    报告级行数:", len(report), "| 机构日终行数:", len(org_daily))

source_min_date = raw["publish_date"].min()
source_max_date = raw["publish_date"].max()
source_max_month = pd.Timestamp(source_max_date).to_period("M").to_timestamp("M")
print("    数据实际范围:", source_min_date.date(), "->", source_max_date.date())

# ===========================================================================
# 3. 月末机构面板 + 一致性快照（快照覆盖到 2024-01 ~ 2025-03）
# ===========================================================================
print("[2/6] 构造月末快照 ...")
snapshot_months = list(pd.date_range(start="2024-01-01", end=SNAPSHOT_END, freq="M"))
label_months = [t for t in snapshot_months if t <= LABEL_END]
me_idx = {t: i for i, t in enumerate(snapshot_months)}

panel_parts = []
snap_parts = []
for t in snapshot_months:
    sub = org_daily[org_daily["publish_date"] <= t].copy()
    sub["age"] = (t - sub["publish_date"]).dt.days
    sub = sub.sort_values("publish_date")
    cand = sub.groupby(["stock_code", "fy", "org_id"]).tail(1).copy()     # 每机构最新
    cand["is_stale"] = (cand["age"] > LOOKBACK_DAYS).astype(int)
    stale_ratio = cand.groupby(["stock_code", "fy"])["is_stale"].mean().rename("stale_ratio")
    valid = cand[cand["age"] <= LOOKBACK_DAYS].copy()
    valid["asof_month"] = t
    valid["mi"] = me_idx[t]
    panel_parts.append(valid[["stock_code", "fy", "org_id", "forecast", "age", "asof_month", "mi"]])

    g = valid.groupby(["stock_code", "fy"])
    s = pd.DataFrame({
        "consensus_median": g["forecast"].median(),
        "consensus_mean": g["forecast"].mean(),
        "consensus_trimmed_mean": g["forecast"].apply(trimmed_mean),
        "forecast_min": g["forecast"].min(),
        "forecast_max": g["forecast"].max(),
        "n_org": g["forecast"].size(),
        "forecast_age_median": g["age"].median(),
        "forecast_age_p90": g["age"].quantile(0.9),
    })
    s["mad"] = g["forecast"].apply(lambda x: float(np.median(np.abs(x - np.median(x)))))
    s["iqr"] = g["forecast"].apply(lambda x: float(x.quantile(0.75) - x.quantile(0.25)))
    s["stale_ratio"] = stale_ratio
    s["asof_month"] = t
    snap_parts.append(s)

panel = pd.concat(panel_parts, ignore_index=True)
snapshots = pd.concat(snap_parts).reset_index()
snapshots["asof_month"] = pd.to_datetime(snapshots["asof_month"])
snapshots["forecast_horizon"] = snapshots["fy"] - LABEL_END.year  # 相对标签年 2024 的期限 FY0/1/2
_stock_name = raw.drop_duplicates("stock_code").set_index("stock_code")["stock_name"].to_dict()
snapshots["stock_name"] = snapshots["stock_code"].map(_stock_name)
snapshots["metric"] = METRIC

snapshots["left_censored"] = (source_min_date > (snapshots["asof_month"] - pd.Timedelta(days=LOOKBACK_DAYS))).astype(int)
snapshots["history_window_complete"] = 1 - snapshots["left_censored"]

# ---- 近零尺度/scale_floor：训练期按 forecast_horizon 拟合 ----
scale_ref = (snapshots[snapshots["asof_month"] <= TRAIN_END]
             .groupby("forecast_horizon")["consensus_median"]
             .apply(lambda x: float(np.median(np.abs(x)))))
nz_thr = {h: 0.01 * float(scale_ref.get(h, 0.0)) for h in [0, 1, 2]}
scale_floor = {h: (nz_thr[h] if nz_thr[h] > 0 else 1e-6) for h in [0, 1, 2]}
print("    近零阈值(按 horizon):", {h: round(v, 2) for h, v in nz_thr.items()})

snapshots["near_zero_threshold"] = snapshots["forecast_horizon"].map(nz_thr)
snapshots["scale_floor"] = snapshots["forecast_horizon"].map(scale_floor)
snapshots["dispersion_robust"] = (MAD_SCALE * snapshots["mad"] /
                                  np.maximum(snapshots["consensus_median"].abs(), snapshots["scale_floor"]))
snapshots["profit_state"] = snapshots.apply(
    lambda r: profit_state_of(r["consensus_median"], r["forecast_min"], r["forecast_max"],
                              r["near_zero_threshold"]), axis=1)

train_snap = snapshots[snapshots["asof_month"] <= TRAIN_END].copy()

# ---- 共识强度：训练期按 horizon 拟合 q30/q70 ----
strength_q = {}
for h in [0, 1, 2]:
    d = train_snap[(train_snap["forecast_horizon"] == h) &
                   (train_snap["n_org"] >= MIN_ORG) &
                   (train_snap["profit_state"].isin(["profit", "loss"]))]["dispersion_robust"].dropna()
    strength_q[h] = (_q(d, 0.30), _q(d, 0.70))
print("    共识强度 q30/q70:", {h: tuple(round(x, 4) if x == x else None for x in v) for h, v in strength_q.items()})


def strength_of(row):
    q30, q70 = strength_q.get(row["forecast_horizon"], (np.nan, np.nan))
    if row["n_org"] < MIN_ORG:
        return "insufficient_coverage"
    if row["profit_state"] in ("near_zero", "mixed_sign"):
        return "unclassifiable"
    if pd.isna(q30) or pd.isna(q70):
        return "unclassifiable"
    d = row["dispersion_robust"]
    if d <= q30:
        return "strong_consensus"
    if d <= q70:
        return "moderate_consensus"
    return "weak_consensus"


snapshots["consensus_strength"] = snapshots.apply(strength_of, axis=1)
snapshots["consensus_strength_classifiable"] = (
    (snapshots["n_org"] >= MIN_ORG) &
    (snapshots["profit_state"].isin(["profit", "loss"]))).astype(int)

snapshots["snapshot_valid"] = ((snapshots["n_org"] >= MIN_ORG) &
                               (snapshots["history_window_complete"] == 1)).astype(int)
snapshots["snapshot_invalid_reason"] = snapshots.apply(
    lambda r: None if r["snapshot_valid"] == 1 else (
        "left_censored" if r["left_censored"] == 1 else "insufficient_current_coverage"), axis=1)
snapshots["strength_threshold_q30"] = snapshots["forecast_horizon"].map(lambda h: strength_q[h][0])
snapshots["strength_threshold_q70"] = snapshots["forecast_horizon"].map(lambda h: strength_q[h][1])
snapshots["strength_threshold_version"] = "train_2024H1_per_horizon"
snapshots["label_version"] = LABEL_VERSION

snap_cols = ["stock_code", "stock_name", "fy", "forecast_horizon", "asof_month", "metric",
             "consensus_median", "consensus_mean", "consensus_trimmed_mean",
             "forecast_min", "forecast_max", "mad", "iqr", "dispersion_robust",
             "consensus_strength", "consensus_strength_classifiable",
             "n_org", "forecast_age_median", "forecast_age_p90", "stale_ratio",
             "profit_state", "near_zero_threshold", "scale_floor",
             "history_window_complete", "left_censored", "snapshot_valid",
             "snapshot_invalid_reason", "strength_threshold_q30", "strength_threshold_q70",
             "strength_threshold_version", "label_version"]
snapshots = snapshots[snap_cols]
print("    快照行数(含未来窗口):", len(snapshots))

snap_pivot = snapshots[["stock_code", "fy", "asof_month", "consensus_median", "n_org",
                        "profit_state", "forecast_age_median", "consensus_strength"]].copy()

# ===========================================================================
# 4. strong 修正阈值：训练期按 (target_horizon × forecast_horizon) 拟合
# ===========================================================================
print("[3/6] 拟合 strong 修正阈值(训练期) ...")
_strong_thr = {}
for h in TARGET_HORIZONS:
    tp = snap_pivot.copy()
    tp["target_month"] = tp["asof_month"].map(lambda t: future_month(t, h))
    fut = snap_pivot.rename(columns={"asof_month": "target_month",
                                     "consensus_median": "consensus_future",
                                     "n_org": "n_org_future",
                                     "profit_state": "profit_state_future"})
    m = tp.merge(fut[["stock_code", "fy", "target_month", "consensus_future",
                      "n_org_future", "profit_state_future"]],
                 on=["stock_code", "fy", "target_month"], how="left")
    m = m[m["asof_month"] <= TRAIN_END]
    m = m[(m["n_org"] >= MIN_ORG) & (m["n_org_future"] >= MIN_ORG)]
    m = m[(m["profit_state"] == "profit") & (m["profit_state_future"] == "profit")]
    m["rev_pct"] = (m["consensus_future"] - m["consensus_median"]) / m["consensus_median"].abs()
    m = m.dropna(subset=["rev_pct"])
    m["nonflat"] = m["rev_pct"].abs() > FLAT[h]
    for fh in [0, 1, 2]:
        sub = m[(m["fy"] - m["asof_month"].dt.year == fh) & m["nonflat"]]
        _strong_thr[(h, fh)] = _q(sub["rev_pct"].abs(), 0.75)
print("    strong 阈值(训练期):", {k: round(v, 5) for k, v in _strong_thr.items()})

# ===========================================================================
# 5. 未来修正 Label（只对 2024 年月末 asof）
# ===========================================================================
print("[4/6] 生成未来修正 Label ...")
revision_parts = []
for h in TARGET_HORIZONS:
    base = snapshots[snapshots["asof_month"] <= LABEL_END].copy()
    base["target_month"] = base["asof_month"].map(lambda t: future_month(t, h))
    fut = snap_pivot.rename(columns={"asof_month": "target_month",
                                     "consensus_median": "consensus_future",
                                     "n_org": "n_org_future",
                                     "profit_state": "profit_state_future",
                                     "forecast_age_median": "forecast_age_median_future"})
    m = base.merge(fut[["stock_code", "fy", "target_month", "consensus_future",
                        "n_org_future", "profit_state_future", "forecast_age_median_future"]],
                   on=["stock_code", "fy", "target_month"], how="left")
    m["target_horizon_months"] = h
    m["metric"] = METRIC

    Ct = m["consensus_median"].astype(float)
    Cf = m["consensus_future"].astype(float)
    m["consensus_t"] = Ct
    m["revision_consensus_pct_raw"] = (Cf - Ct) / Ct.abs()
    m["revision_consensus_symmetric_raw"] = 2 * (Cf - Ct) / (Cf.abs() + Ct.abs() + m["scale_floor"])

    target_present = m["target_month"].notna()
    right_censored = (~target_present) | (m["target_month"] > source_max_month)
    future_snapshot_missing = (~right_censored) & Cf.isna()
    n_org_future_eff = m["n_org_future"].fillna(0)
    history_complete = (m["history_window_complete"] == 1)
    revision_consensus_valid = (history_complete & ~right_censored & ~future_snapshot_missing &
                                (m["n_org"] >= MIN_ORG) & (n_org_future_eff >= MIN_ORG)).astype(int)

    m["revision_consensus_pct"] = np.where(revision_consensus_valid == 1, m["revision_consensus_pct_raw"], np.nan)
    m["revision_consensus_symmetric"] = np.where(revision_consensus_valid == 1, m["revision_consensus_symmetric_raw"], np.nan)
    m["revision_consensus_valid"] = revision_consensus_valid

    profit_pair = (m["profit_state"] == "profit") & (m["profit_state_future"] == "profit")


    def _cls5(row):
        if not profit_pair.loc[row.name]:
            return None
        v = row["revision_consensus_pct"]
        if pd.isna(v):
            return None
        strong = _strong_thr.get((row["target_horizon_months"], row["forecast_horizon"]),
                                 STRONG_FIXED[row["target_horizon_months"]])
        return classify5(v, FLAT[row["target_horizon_months"]], strong)


    m["revision_consensus_class"] = m.apply(_cls5, axis=1)

    m["profit_transition_class"] = m.apply(
        lambda r: transition_class(r["consensus_t"], r["consensus_future"],
                                   r["profit_state"], r["profit_state_future"],
                                   r["revision_consensus_class"] if r["revision_consensus_valid"] == 1 else None),
        axis=1)
    m["profit_transition_valid"] = ((revision_consensus_valid == 1) &
                                    m["profit_transition_class"].notna()).astype(int)
    m.loc[m["profit_transition_valid"] == 0, "profit_transition_class"] = None

    # ---- 固定面板 ----
    pfut = panel.rename(columns={"forecast": "np_future", "mi": "mi_f"})
    pn = panel.merge(pfut[["stock_code", "fy", "org_id", "np_future", "mi_f"]],
                     on=["stock_code", "fy", "org_id"], how="inner")
    pn = pn[pn["mi_f"] == pn["mi"] + h]
    fixed = pn.groupby(["stock_code", "fy", "mi"]).agg(
        n_org_fixed=("org_id", "size"),
        fixed_Ct=("forecast", "median"),
        fixed_Cf=("np_future", "median")).reset_index()
    fixed["revision_fixed_pct_raw"] = (fixed["fixed_Cf"] - fixed["fixed_Ct"]) / fixed["fixed_Ct"].abs()
    fixed["asof_month"] = fixed["mi"].map({i: t for t, i in me_idx.items()})
    m = m.merge(fixed[["stock_code", "fy", "asof_month", "n_org_fixed", "revision_fixed_pct_raw"]],
                on=["stock_code", "fy", "asof_month"], how="left")
    m["revision_fixed_valid"] = ((revision_consensus_valid == 1) &
                                 (m["n_org_fixed"] >= MIN_ORG_FIXED)).astype(int)
    m["revision_fixed_pct"] = np.where(m["revision_fixed_valid"] == 1, m["revision_fixed_pct_raw"], np.nan)

    # ---- 主动更新 ----
    act_rows = []
    for t in label_months:
        fut = future_month(t, h)
        win = org_daily[(org_daily["publish_date"] > t) & (org_daily["publish_date"] <= fut)]
        if len(win) == 0:
            continue
        win = win.sort_values("publish_date")
        wlatest = win.groupby(["stock_code", "fy", "org_id"]).tail(1)
        pt = panel[(panel["asof_month"] == t)][["stock_code", "fy", "org_id", "forecast"]].rename(
            columns={"forecast": "forecast_old"})
        a = wlatest.merge(pt, on=["stock_code", "fy", "org_id"], how="inner")
        if len(a) == 0:
            continue
        a["chg"] = a["forecast"] - a["forecast_old"]
        a["chg_pct"] = a["chg"] / a["forecast_old"].abs()
        a["dir"] = [classify_dir(v, FLAT[h]) for v in a["chg_pct"]]
        g = a.groupby(["stock_code", "fy"])
        agg = pd.DataFrame({
            "n_active_update": g["org_id"].size(),
            "revision_active_pct_raw": g["chg_pct"].median(),
            "n_up": g["dir"].apply(lambda s: (s == "up").sum()),
            "n_down": g["dir"].apply(lambda s: (s == "down").sum()),
            "n_flat": g["dir"].apply(lambda s: (s == "flat").sum()),
        }).reset_index()
        agg["asof_month"] = t
        act_rows.append(agg)
    if act_rows:
        act_all = pd.concat(act_rows, ignore_index=True)
        m = m.merge(act_all, on=["stock_code", "fy", "asof_month"], how="left")
    else:
        for c in ["n_active_update", "revision_active_pct_raw", "n_up", "n_down", "n_flat"]:
            m[c] = np.nan

    m["n_active_update"] = m["n_active_update"].fillna(0).astype(int)
    m["revision_active_valid"] = (m["n_active_update"] >= MIN_ORG_ACTIVE).astype(int)
    m["revision_active_pct"] = np.where(m["revision_active_valid"] == 1, m["revision_active_pct_raw"], np.nan)
    m["revision_active_class"] = m.apply(
        lambda r: classify_dir(r["revision_active_pct"], FLAT[r["target_horizon_months"]])
        if r["revision_active_valid"] == 1 else None, axis=1)

    m["revision_breadth_raw"] = (m["n_up"] - m["n_down"]) / (m["n_up"] + m["n_down"] + m["n_flat"]).replace(0, np.nan)
    m["breadth_valid"] = m["revision_active_valid"]
    m["revision_breadth"] = np.where(m["breadth_valid"] == 1, m["revision_breadth_raw"], np.nan)

    m["composition_effect_approx"] = m["revision_consensus_pct_raw"] - m["revision_fixed_pct_raw"]
    m["right_censored"] = right_censored.astype(int)
    m["future_snapshot_missing"] = future_snapshot_missing.astype(int)

    def inv_reason(r):
        if r["revision_consensus_valid"] == 1:
            return None
        if r["right_censored"] == 1:
            return "right_censored"
        if r["left_censored"] == 1:
            return "left_censored"
        if r["future_snapshot_missing"] == 1:
            return "future_snapshot_missing"
        if r["n_org"] < MIN_ORG:
            return "insufficient_current_coverage"
        return "insufficient_future_coverage"

    m["invalid_reason"] = m.apply(inv_reason, axis=1)

    w_cov = np.minimum(1.0, np.log(1 + m["n_org"]) / np.log(11))
    w_fresh = np.exp(-m["forecast_age_median"].fillna(LOOKBACK_DAYS) / LOOKBACK_DAYS)
    m["consensus_revision_weight"] = np.where(revision_consensus_valid == 1, (w_cov * w_fresh) ** 0.5, 0.0)
    m["fixed_revision_weight"] = np.where(m["revision_fixed_valid"] == 1, (w_cov * w_fresh) ** 0.5, 0.0)
    m["active_revision_weight"] = np.where(m["revision_active_valid"] == 1, w_cov, 0.0)
    m["breadth_weight"] = np.where(m["breadth_valid"] == 1, w_cov, 0.0)

    revision_parts.append(m)

revision = pd.concat(revision_parts, ignore_index=True)
revision["n_org_t"] = revision["n_org"]
revision["n_org_future"] = revision["n_org_future"].fillna(0).astype(int)
revision["n_org_fixed"] = revision["n_org_fixed"].fillna(0).astype(int)
revision["label_version"] = LABEL_VERSION

rev_cols = ["stock_code", "fy", "forecast_horizon", "asof_month", "target_month",
            "target_horizon_months", "metric", "consensus_t", "consensus_future",
            "revision_consensus_pct_raw", "revision_consensus_symmetric_raw",
            "revision_consensus_pct", "revision_consensus_symmetric", "revision_consensus_class",
            "revision_consensus_valid", "revision_fixed_pct_raw", "revision_fixed_pct",
            "revision_fixed_valid", "revision_active_pct_raw", "revision_active_pct",
            "revision_active_class", "revision_active_valid", "revision_breadth_raw",
            "revision_breadth", "breadth_valid", "profit_transition_class", "profit_transition_valid",
            "n_up", "n_down", "n_flat", "n_active_update", "n_org_t", "n_org_future", "n_org_fixed",
            "left_censored", "right_censored", "future_snapshot_missing", "invalid_reason",
            "composition_effect_approx", "consensus_revision_weight", "fixed_revision_weight",
            "active_revision_weight", "breadth_weight", "label_version"]
revision = revision[rev_cols]
print("    修正 Label 行数:", len(revision),
      "| 主目标有效:", int(revision["revision_consensus_valid"].sum()),
      "| 右删失:", int(revision["right_censored"].sum()),
      "| future_snapshot_missing:", int(revision["future_snapshot_missing"].sum()))

# ===========================================================================
# 6. 报告相对立场 Label（仅 2024 年发布的报告）
# ===========================================================================
print("[5/6] 生成报告级相对立场 Label ...")
report_label_window = report[(report["publish_date"] >= "2024-01-01") &
                             (report["publish_date"] <= "2024-12-31")].copy()

org_day_cnt = org_daily[org_daily["n_reports_same_org_day"] > 1][
    ["stock_code", "fy", "org_id", "publish_date"]].copy()
org_day_cnt["same_day_has_multiple"] = 1
report_label_window = report_label_window.merge(org_day_cnt,
                                                on=["stock_code", "fy", "org_id", "publish_date"], how="left")
report_label_window["same_org_day_multiple_reports"] = report_label_window["same_day_has_multiple"].fillna(0).astype(int)

od_map = {k: v.sort_values("publish_date") for k, v in org_daily.groupby(["stock_code", "fy"])}


def report_labels_for_group(key, g):
    stock, fy = key
    od = od_map.get((stock, fy))
    if od is None or len(od) == 0:
        return None
    od = od.reset_index(drop=True)
    org_dates, org_vals = {}, {}
    for o, grp in od.groupby("org_id"):
        org_dates[o] = grp["publish_date"].astype("int64").values   # ns 时间戳
        org_vals[o] = grp["forecast"].values
    g = g.sort_values("publish_date").reset_index(drop=True)
    nz = nz_thr.get(fy - 2024, nz_thr.get(0, 0.0))
    sf = scale_floor.get(fy - 2024, 1e-6)
    out = []
    for _, r in g.iterrows():
        t = r["publish_date"]
        t_ns = int(t.value)
        i_org = r["org_id"]
        F = float(r["forecast"])
        others = []
        for o in org_dates:
            if o == i_org:
                continue
            ds, vs = org_dates[o], org_vals[o]
            pos = int(np.searchsorted(ds, t_ns, side="left")) - 1
            if pos >= 0 and (t - pd.Timestamp(ds[pos])).days <= LOOKBACK_DAYS:
                others.append(float(vs[pos]))
        n_pre = len(others)
        history_ok = int(source_min_date <= t - pd.Timedelta(days=LOOKBACK_DAYS))
        C = disp = pct = sym = np.nan
        classifiable = False
        pos_cls = None
        if n_pre >= MIN_ORG_PRE:
            arr = np.array(others)
            C = float(np.median(arr))
            MAD = float(np.median(np.abs(arr - C)))
            disp = MAD_SCALE * MAD / max(abs(C), sf)
            pct = (F - C) / max(abs(C), sf)
            sym = 2 * (F - C) / (abs(F) + abs(C) + sf)
            classifiable = abs(C) > nz
            if classifiable:
                ib = max(FLAT[1], 0.5 * disp)
                sb = max(STRONG_FIXED[1], 1.5 * disp)
                a = abs(pct)
                if a <= ib:
                    pos_cls = "inline"
                elif a < sb:
                    pos_cls = "above" if pct > 0 else "below"
                else:
                    pos_cls = "strong_above" if pct > 0 else "strong_below"

        if i_org in org_dates:
            ds, vs = org_dates[i_org], org_vals[i_org]
            pos = int(np.searchsorted(ds, t_ns, side="left")) - 1
            if pos >= 0:
                old_val = float(vs[pos])
                old_age = (t - pd.Timestamp(ds[pos])).days
            else:
                old_val = old_age = np.nan
        else:
            old_val = old_age = np.nan
        self_revision_valid = int((not pd.isna(old_val)) and old_age <= LOOKBACK_DAYS)
        if pd.notna(old_val) and old_val != 0:
            self_pct_raw = (F - old_val) / abs(old_val)
            self_pct = self_pct_raw if self_revision_valid else np.nan
            self_cls = classify_dir(self_pct_raw, FLAT[1]) if self_revision_valid else None
        else:
            self_pct_raw = self_pct = np.nan
            self_cls = None

        report_position_valid = int(history_ok and n_pre >= MIN_ORG_PRE and classifiable)
        pos_cls_final = pos_cls if report_position_valid else None
        pct_final = pct if report_position_valid else np.nan
        sym_final = sym if report_position_valid else np.nan
        rpw = min(1.0, np.log(1 + n_pre) / np.log(11)) if report_position_valid else 0.0
        if report_position_valid:
            invalid = None
        elif not history_ok:
            invalid = "left_censored"
        elif n_pre < MIN_ORG_PRE:
            invalid = "insufficient_pre_consensus"
        else:
            invalid = "near_zero_unclassifiable"

        out.append(dict(
            source_report_id=r["source_report_id"],
            report_signature=r["report_signature"],
            publish_date=t, publish_time=r["publish_time"],
            stock_code=stock, fy=int(fy), forecast_horizon=int(fy - 2024),
            org_id=i_org, org_name=r["org_name"], author_name=r["author_name"], title=r["title"],
            metric=METRIC,
            forecast_new=F,
            forecast_old_self=(old_val if pd.notna(old_val) else np.nan),
            self_old_forecast_age=(old_age if pd.notna(old_age) else np.nan),
            consensus_pre_ex_org=(C if n_pre >= MIN_ORG_PRE else np.nan),
            dispersion_pre_ex_org=(disp if n_pre >= MIN_ORG_PRE else np.nan),
            n_org_pre_ex_org=n_pre,
            report_distance_pct_raw=pct,
            report_distance_symmetric_raw=sym,
            report_distance_pct=pct_final,
            report_distance_symmetric=sym_final,
            report_position_class=pos_cls_final,
            report_position_valid=report_position_valid,
            self_revision_pct_raw=self_pct_raw,
            self_revision_pct=self_pct,
            self_revision_class=self_cls,
            self_revision_valid=self_revision_valid,
            same_org_day_multiple_reports=r["same_org_day_multiple_reports"],
            history_window_complete=history_ok,
            invalid_reason=invalid,
            report_position_weight=rpw,
            label_version=LABEL_VERSION,
        ))
    return pd.DataFrame(out)


rep_parts = []
for key, g in report_label_window.groupby(["stock_code", "fy"]):
    rdf = report_labels_for_group(key, g)
    if rdf is not None and len(rdf):
        rep_parts.append(rdf)
rep_detail = pd.concat(rep_parts, ignore_index=True)
print("    报告级 Label 行数:", len(rep_detail), "| 位置有效:", int(rep_detail["report_position_valid"].sum()))

# ===========================================================================
# 7. 保存 + 元数据
# ===========================================================================
print("[6/6] 保存输出 ...")
report_out = report_label_window[["source_report_id", "report_signature", "publish_date", "publish_time",
                                  "stock_code", "org_id", "org_name", "author_name", "title", "fy",
                                  "forecast", "same_org_day_multiple_reports"]].copy()
report_out.to_parquet(os.path.join(OUT, "clean_report_v2.parquet"), index=False)
org_daily.to_parquet(os.path.join(OUT, "org_daily_forecast_v2.parquet"), index=False)

# 只输出 label 窗口（2024）的快照
snap_out = snapshots[snapshots["asof_month"] <= LABEL_END].copy()
for name, obj in [("consensus_snapshot_monthly_v2", snap_out),
                  ("revision_label_monthly_v2", revision),
                  ("report_label_detail_v2", rep_detail)]:
    obj.to_parquet(os.path.join(OUT, name + ".parquet"), index=False)
    obj.to_csv(os.path.join(OUT, name + ".csv"), index=False)
    print("    已写:", name, obj.shape)

meta = {
    "label_version": LABEL_VERSION,
    "created_at": dt.datetime.now().isoformat(timespec="seconds"),
    "source_data": SRC,
    "metric": METRIC,
    "label_window": [str(LABEL_START.date()), str(LABEL_END.date())],
    "source_window": [SOURCE_START, SOURCE_END],
    "fy_range": [FY_MIN, FY_MAX],
    "lookback_days": LOOKBACK_DAYS,
    "target_horizon_months": TARGET_HORIZONS,
    "train_period": "asof_month <= " + str(TRAIN_END.date()),
    "min_org_snapshot": MIN_ORG,
    "min_org_pre_report": MIN_ORG_PRE,
    "min_org_fixed": MIN_ORG_FIXED,
    "min_org_active": MIN_ORG_ACTIVE,
    "flat_threshold": FLAT,
    "strong_threshold_fixed_sensitivity": STRONG_FIXED,
    "strong_threshold_fitted": {f"{k[0]}m_fy{k[1]}": (round(v, 6) if v == v else None) for k, v in _strong_thr.items()},
    "near_zero_threshold": {str(k): v for k, v in nz_thr.items()},
    "scale_floor": {str(k): v for k, v in scale_floor.items()},
    "consensus_strength_q30_q70": {str(k): [round(x, 6) if x == x else None for x in v] for k, v in strength_q.items()},
    "consensus": "org equal-weight, latest valid forecast per org, median",
    "report_position_metric": "normal profit: pct; classified only when |consensus|>near_zero_threshold",
    "notes": [
        "数据已补齐到 2010-2025，180 天历史与 3m 未来窗口完整，left/right_censored 应基本为 0",
        "正式目标无效时置空；_raw 仅诊断，勿入模型目标",
        "阈值均在 asof<=%s 训练期拟合，按 forecast_horizon 分组" % TRAIN_END.date(),
        "revision_consensus_valid 仅代表市场共识主目标；fixed/active/breadth 各有独立 valid",
    ],
}
with open(os.path.join(OUT, "label_metadata_v2.json"), "w", encoding="utf-8") as f:
    json.dump(meta, f, ensure_ascii=False, indent=2)

print("\n完成。输出目录:", OUT)
print(json.dumps(meta, ensure_ascii=False, indent=2)[:1000])