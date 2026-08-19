# -*- coding: utf-8 -*-
"""
研报"一致预期—预期修正" Label 构造（V1）
依据: consensus_revision_label_design_v2.md

落地范围（文档 §1.3 / §15 V1）:
  - 主指标: FORECAST_NP
  - 样本单元: (STOCK_CODE, FY, ASOF_MONTH_END)
  - 财年: FY0/FY1/FY2 (2024/2025/2026)，排除 FY-1 与 FY3+
  - 共识口径: 机构等权 + 每机构最新有效预测 + 取中位数
  - 有效期: L=180 天
  - 主目标: 未来 1m / 3m 市场一致预期修正
  - 辅助: 固定面板修正、主动更新修正、修正广度
  - 报告级: report_position_class（相对发布前共识）+ self_revision_class
  - 有效条件: 当前/未来机构数>=5，固定面板>=3，主动修正>=3
  - 连续值 + 五分类 + 有效标志 + 样本权重 + 版本元数据
"""
import os
import json
import datetime as dt
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------
SRC = "/home/intern_fjq_2026/data/NLP/research_report/forecast_stk_20240101_20241231.jsonl"
OUT = "/home/intern_fjq_2026/Projects/chinese-wwm-roberta/artifacts/label_engineering"
os.makedirs(OUT, exist_ok=True)

METRIC = "FORECAST_NP"
L = 180                 # 预测有效期(天)
FY_MIN, FY_MAX = 2024, 2026   # V1 只做 FY0/FY1/FY2
MAD_SCALE = 1.4826      # MAD -> 标准差近似
EPS = 1e-9
SCALE_FLOOR = 1e-9
MIN_ORG = 5            # 快照/修正 硬性有效机构数
MIN_ORG_PRE = 4        # 报告相对立场: 排除本机构后至少 4 家
MIN_ORG_FIXED = 3      # 固定面板
MIN_ORG_ACTIVE = 3     # 修正广度
FLAT1, STRONG1 = 0.01, 0.05   # 1m 五分类阈值
FLAT3, STRONG3 = 0.02, 0.10   # 3m 五分类阈值
LABEL_VERSION = "v1.0"

# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------
def trimmed_mean(x, frac=0.1):
    x = np.sort(np.asarray(x, dtype=float))
    n = len(x)
    k = int(round(n * frac))
    if k == 0 or n - 2 * k <= 0:
        return float(np.mean(x))
    return float(np.mean(x[k:n - k]))


def classify_dir(v, flat_thr):
    """三分类 up/down/flat, v 为相对变化"""
    if pd.isna(v):
        return None
    if abs(v) <= flat_thr:
        return "flat"
    return "up" if v > 0 else "down"


def classify5(v, flat_thr, strong_thr):
    """五分类 flat/up/down/strong_up/strong_down"""
    if pd.isna(v):
        return None
    a = abs(v)
    if a <= flat_thr:
        return "flat"
    if a < strong_thr:
        return "up" if v > 0 else "down"
    return "strong_up" if v > 0 else "strong_down"


def profit_state(median, mn, mx):
    if abs(median) < 1e-6:
        return "near_zero"
    if mn < -1e-6 and mx > 1e-6:
        return "mixed_sign"
    return "profit" if median > 0 else "loss"


def profit_transition(a, b):
    if pd.isna(a) or pd.isna(b):
        return None
    e = 1e-6
    if a > e and b <= -e:
        return "turn_to_loss"
    if a <= -e and b > e:
        return "turn_to_profit"
    if a < -e and b < -e:
        return "loss_narrowing" if b > a else ("loss_widening" if b < a else "loss_unchanged")
    if abs(a) <= e and abs(b) <= e:
        return "zero_unchanged"
    if a > e and b > e:
        return "profit"
    return "mixed"


# ---------------------------------------------------------------------------
# 阶段 1: 清洗 + 机构级同日去重
# ---------------------------------------------------------------------------
print("[1/5] 读取数据并清洗 ...")
df = pd.read_json(SRC, lines=True)
use = df[["ID", "STOCK_CODE", "STOCK_NAME", "ORGAN_NAME", "AUTHOR_NAME",
          "TITLE", "CREATE_DATE", "REPORT_YEAR"] + [METRIC]].copy()
use = use.dropna(subset=[METRIC, "REPORT_YEAR"])
use["CREATE_DATE"] = pd.to_datetime(use["CREATE_DATE"])
use = use[(use["CREATE_DATE"] >= pd.Timestamp("2024-01-01")) &
          (use["CREATE_DATE"] <= pd.Timestamp("2024-12-31"))]
use["FY"] = use["REPORT_YEAR"].astype(int)
use = use[(use["FY"] >= FY_MIN) & (use["FY"] <= FY_MAX)]

# 报告签名（股票+机构+作者+日期+标题），同一报告拆出的多财年共享同一签名
use["_sig_auth"] = use["AUTHOR_NAME"].fillna("")
_sig = (use["STOCK_CODE"].astype(str) + "|" + use["ORGAN_NAME"].astype(str) + "|" +
        use["_sig_auth"] + "|" + use["CREATE_DATE"].dt.strftime("%Y%m%d") + "|" +
        use["TITLE"].astype(str))
use["report_signature"] = _sig.astype("category").cat.codes

# 同机构同日同股同FY: 取中位数，标记冲突
key = ["STOCK_CODE", "FY", "ORGAN_NAME", "CREATE_DATE"]
dd = use.groupby(key, as_index=False).agg(
    FORECAST_NP=(METRIC, "median"),
    STOCK_NAME=("STOCK_NAME", "first"),
    AUTHOR_NAME=("AUTHOR_NAME", "first"),
    TITLE=("TITLE", "first"),
    report_signature=("report_signature", "first"),
    ID=("ID", "first"),
    n_same_day=(METRIC, "size"),
)
dd["same_day_conflict"] = (dd["n_same_day"] > 1).astype(int)
dd = dd.sort_values(["STOCK_CODE", "FY", "ORGAN_NAME", "CREATE_DATE"]).reset_index(drop=True)
print("    清洗后机构级记录数:", len(dd), "| 报告签名数:", dd["report_signature"].nunique())

# ---------------------------------------------------------------------------
# 阶段 2: 月末 As-of 机构面板 + 共识快照
# ---------------------------------------------------------------------------
print("[2/5] 构造月度 as-of 快照 ...")
month_ends = list(pd.date_range(start="2024-01-01", end="2024-12-31", freq="M"))
me_idx = {t: i for i, t in enumerate(month_ends)}

panel_parts = []
snap_parts = []
for t in month_ends:
    sub = dd[dd["CREATE_DATE"] <= t].copy()
    sub["age"] = (t - sub["CREATE_DATE"]).dt.days
    sub = sub.sort_values("CREATE_DATE")
    cand = sub.groupby(["STOCK_CODE", "FY", "ORGAN_NAME"]).tail(1)   # 每机构最新
    cand["is_stale"] = (cand["age"] > L).astype(int)
    stale_ratio = cand.groupby(["STOCK_CODE", "FY"])["is_stale"].mean().rename("stale_ratio")
    valid = cand[cand["age"] <= L].copy()
    valid["asof_month"] = t
    valid["mi"] = me_idx[t]
    panel_parts.append(valid[["STOCK_CODE", "FY", "ORGAN_NAME", "FORECAST_NP",
                              "age", "asof_month", "mi"]])

    g = valid.groupby(["STOCK_CODE", "FY"])
    s = pd.DataFrame({
        "consensus_median": g["FORECAST_NP"].median(),
        "consensus_mean": g["FORECAST_NP"].mean(),
        "consensus_trimmed_mean": g["FORECAST_NP"].apply(trimmed_mean),
        "n_org": g["FORECAST_NP"].size(),
        "min": g["FORECAST_NP"].min(),
        "max": g["FORECAST_NP"].max(),
        "forecast_age_median": g["age"].median(),
        "forecast_age_p90": g["age"].quantile(0.9),
    })
    s["mad"] = g["FORECAST_NP"].apply(lambda x: float(np.median(np.abs(x - np.median(x)))))
    s["iqr"] = g["FORECAST_NP"].apply(lambda x: float(x.quantile(0.75) - x.quantile(0.25)))
    s["stale_ratio"] = stale_ratio
    s["asof_month"] = t
    snap_parts.append(s)

panel = pd.concat(panel_parts, ignore_index=True)
snapshots = pd.concat(snap_parts).reset_index()

snapshots["dispersion_robust"] = (MAD_SCALE * snapshots["mad"] /
                                  (snapshots["consensus_median"].abs() + EPS))
snapshots["profit_state"] = snapshots.apply(
    lambda r: profit_state(r["consensus_median"], r["min"], r["max"]), axis=1)
snapshots["snapshot_valid"] = (snapshots["n_org"] >= MIN_ORG).astype(int)

# 共识强度: 有效样本的 dispersion_robust 30/70 分位 -> strong/moderate/weak
_v = snapshots.loc[snapshots["snapshot_valid"] == 1, "dispersion_robust"].dropna()
q30, q70 = float(_v.quantile(0.30)), float(_v.quantile(0.70))


def strength(row):
    if row["n_org"] < MIN_ORG:
        return "insufficient_coverage"
    d = row["dispersion_robust"]
    if d <= q30:
        return "strong_consensus"
    if d <= q70:
        return "moderate_consensus"
    return "weak_consensus"


snapshots["consensus_strength"] = snapshots.apply(strength, axis=1)
_stock_name = dd.drop_duplicates("STOCK_CODE").set_index("STOCK_CODE")["STOCK_NAME"].to_dict()
snapshots["stock_name"] = snapshots["STOCK_CODE"].map(_stock_name)
snapshots["forecast_horizon"] = snapshots["FY"] - snapshots["asof_month"].dt.year
snapshots["metric"] = METRIC

snap_cols = ["STOCK_CODE", "stock_name", "FY", "forecast_horizon", "asof_month", "metric",
             "consensus_median", "consensus_mean", "consensus_trimmed_mean", "mad", "iqr",
             "dispersion_robust", "consensus_strength", "n_org", "forecast_age_median",
             "forecast_age_p90", "stale_ratio", "profit_state", "snapshot_valid"]
snapshots = snapshots[snap_cols].rename(columns={"STOCK_CODE": "stock_code",
                                                 "FY": "fy",
                                                 "asof_month": "asof_month"}, inplace=False)
# 快照列名与文档 12.1 对齐
snapshots = snapshots.rename(columns={"stock_code": "stock_code"})
print("    快照行数:", len(snapshots), "| 有效快照(snapshot_valid=1):", int(snapshots["snapshot_valid"].sum()))

# ---------------------------------------------------------------------------
# 阶段 3: 未来修正 Label (1m / 3m)
# ---------------------------------------------------------------------------
print("[3/5] 生成未来修正 Label (1m/3m) ...")
snap = snapshots.copy()
snap["asof_month"] = pd.to_datetime(snap["asof_month"])


def future_month(t, h):
    i = me_idx[t] + h
    return month_ends[i] if i < len(month_ends) else None


revision_parts = []
for h, (flat, strong) in [(1, (FLAT1, STRONG1)), (3, (FLAT3, STRONG3))]:
    fut_map = {t: future_month(t, h) for t in month_ends}
    base = snap.copy()
    base["asof_future"] = base["asof_month"].map(fut_map)

    fut = snap[["stock_code", "fy", "asof_month", "consensus_median", "n_org",
                "forecast_age_median"]].rename(columns={
        "asof_month": "asof_future", "consensus_median": "consensus_future",
        "n_org": "n_org_future", "forecast_age_median": "forecast_age_median_future"})
    m = base.merge(fut, on=["stock_code", "fy", "asof_future"], how="left",
                   suffixes=("", "_f"))
    # 去掉重复列残留
    m = m.loc[:, ~m.columns.duplicated()]

    Ct = m["consensus_median"].astype(float)
    Cf = m["consensus_future"].astype(float)
    m["target_horizon_months"] = h
    m["consensus_t"] = Ct
    m["revision_consensus_pct"] = (Cf - Ct) / Ct.abs()
    m["revision_consensus_symmetric"] = 2 * (Cf - Ct) / (Cf.abs() + Ct.abs() + EPS)
    _log = np.log(Cf / Ct)
    _log[~((Ct > 0) & (Cf > 0))] = np.nan
    m["revision_consensus_log"] = _log
    _use_pct = (Ct > 0) & (Cf > 0)
    _cls_val = np.where(_use_pct, m["revision_consensus_pct"], m["revision_consensus_symmetric"])
    m["revision_consensus_class"] = [classify5(v, flat, strong) for v in _cls_val]
    m["profit_transition"] = [profit_transition(a, b) for a, b in zip(Ct, Cf)]

    # ---- 固定面板修正 ----
    pfut = panel.rename(columns={"FORECAST_NP": "np_future", "mi": "mi_f"})
    pfut = pfut[["STOCK_CODE", "FY", "ORGAN_NAME", "np_future", "mi_f"]]
    pn = panel.merge(pfut, on=["STOCK_CODE", "FY", "ORGAN_NAME"], how="inner")
    pn = pn[pn["mi_f"] == pn["mi"] + h]
    fixed = pn.groupby(["STOCK_CODE", "FY", "mi"]).agg(
        n_org_fixed=("ORGAN_NAME", "size"),
        fixed_Ct=("FORECAST_NP", "median"),
        fixed_Cf=("np_future", "median"),
    ).reset_index()
    fixed["revision_fixed_pct"] = (fixed["fixed_Cf"] - fixed["fixed_Ct"]) / fixed["fixed_Ct"].abs()
    fixed["asof_month"] = fixed["mi"].map({i: t for t, i in me_idx.items()})
    fixed = fixed.rename(columns={"STOCK_CODE": "stock_code", "FY": "fy"})
    m = m.merge(fixed[["stock_code", "fy", "asof_month", "n_org_fixed",
                       "revision_fixed_pct"]], on=["stock_code", "fy", "asof_month"], how="left")

    # ---- 主动更新修正 (窗口 (t, t+h] 内实际更新预测的机构) ----
    act_rows = []
    for t in month_ends:
        fut = future_month(t, h)
        if fut is None:
            continue
        win = dd[(dd["CREATE_DATE"] > t) & (dd["CREATE_DATE"] <= fut)]
        if len(win) == 0:
            continue
        win = win.sort_values("CREATE_DATE")
        wlatest = win.groupby(["STOCK_CODE", "FY", "ORGAN_NAME"]).tail(1)
        pt = panel[(panel["asof_month"] == t)][["STOCK_CODE", "FY", "ORGAN_NAME",
                                                "FORECAST_NP"]].rename(columns={"FORECAST_NP": "np_old"})
        a = wlatest.merge(pt, on=["STOCK_CODE", "FY", "ORGAN_NAME"], how="inner")
        if len(a) == 0:
            continue
        a["chg"] = a["FORECAST_NP"] - a["np_old"]
        a["chg_pct"] = a["chg"] / a["np_old"].abs()
        a["dir"] = [classify_dir(v, flat) for v in a["chg_pct"]]
        g = a.groupby(["STOCK_CODE", "FY"])
        act = pd.DataFrame({
            "n_active_update": g["ORGAN_NAME"].size(),
            "revision_active_pct": g["chg_pct"].median(),
            "n_up": g["dir"].apply(lambda s: (s == "up").sum()),
            "n_down": g["dir"].apply(lambda s: (s == "down").sum()),
            "n_flat": g["dir"].apply(lambda s: (s == "flat").sum()),
        }).reset_index()
        act["revision_breadth"] = (act["n_up"] - act["n_down"]) / (
            act["n_up"] + act["n_down"] + act["n_flat"])
        act["asof_month"] = t
        act_rows.append(act)
    if act_rows:
        act_all = pd.concat(act_rows, ignore_index=True).rename(
            columns={"STOCK_CODE": "stock_code", "FY": "fy"})
        m = m.merge(act_all, on=["stock_code", "fy", "asof_month"], how="left")
    else:
        for c in ["n_active_update", "revision_active_pct", "n_up", "n_down",
                  "n_flat", "revision_breadth"]:
            m[c] = np.nan

    # 主动修正类别 / 广度有效性
    m["revision_active_class"] = [classify5(v, flat, strong) for v in m["revision_active_pct"]]
    m.loc[m["n_active_update"] < MIN_ORG_ACTIVE, "revision_breadth"] = np.nan
    m.loc[m["n_active_update"] < MIN_ORG_ACTIVE, "revision_active_class"] = None
    m["composition_effect"] = m["revision_consensus_pct"] - m["revision_fixed_pct"]

    revision_parts.append(m)

revision = pd.concat(revision_parts, ignore_index=True)

# ---- 有效标志 / 理由 / 权重 ----
revision["right_censored"] = revision["asof_future"].isna() | revision["consensus_future"].isna()
revision["n_org_t"] = revision["n_org"]
revision["n_org_future"] = revision["n_org_future"].fillna(0).astype(int)
revision["n_org_fixed"] = revision["n_org_fixed"].fillna(0).astype(int)
revision["n_active_update"] = revision["n_active_update"].fillna(0).astype(int)


def inv_reason(r):
    if r["right_censored"]:
        return "right_censored"
    if r["n_org_t"] < MIN_ORG or r["n_org_future"] < MIN_ORG:
        return "insufficient_coverage"
    return ""


revision["label_valid"] = ((~revision["right_censored"]) &
                           (revision["n_org_t"] >= MIN_ORG) &
                           (revision["n_org_future"] >= MIN_ORG)).astype(int)
revision["invalid_reason"] = revision.apply(inv_reason, axis=1)
revision.loc[revision["label_valid"] == 0, "revision_consensus_class"] = None

# 软权重 (§10.3)
_w_cov = np.minimum(1.0, np.log(1 + revision["n_org_t"]) / np.log(11))
_w_fresh = np.exp(-revision["forecast_age_median"].fillna(L) / L)
_w_panel = np.minimum(1.0, revision["n_org_fixed"] / 5.0)
revision["sample_weight"] = (_w_cov * _w_fresh * _w_panel) ** (1.0 / 3.0)
revision.loc[revision["label_valid"] == 0, "sample_weight"] = 0.0
revision["label_version"] = LABEL_VERSION

rev_cols = ["stock_code", "fy", "forecast_horizon", "asof_month", "metric",
            "target_horizon_months", "consensus_t", "consensus_future",
            "revision_consensus_pct", "revision_consensus_log", "revision_consensus_symmetric",
            "revision_consensus_class", "revision_fixed_pct", "revision_active_pct",
            "revision_active_class", "composition_effect", "revision_breadth",
            "n_up", "n_down", "n_flat", "n_active_update", "profit_transition",
            "n_org_t", "n_org_future", "n_org_fixed", "label_valid",
            "invalid_reason", "sample_weight", "label_version"]
revision = revision[rev_cols]
print("    修正 Label 行数:", len(revision),
      "| 有效(label_valid=1):", int(revision["label_valid"].sum()),
      "| 右删失:", int(revision["right_censored"].sum() if "right_censored" in revision else 0))

# ---------------------------------------------------------------------------
# 阶段 4: 报告相对立场 Label
# ---------------------------------------------------------------------------
print("[4/5] 生成报告级相对立场 Label ...")


def report_labels_for_group(g):
    """对 (stock,fy) 内按日期顺序，用严格早于发布日的其他机构最新预测作为基准"""
    g = g.sort_values("CREATE_DATE")
    org_last = {}
    out = []
    for date, grp in g.groupby("CREATE_DATE"):
        for _, r in grp.iterrows():
            i_org = r["ORGAN_NAME"]
            F = float(r["FORECAST_NP"])
            others = np.array([v for o, v in org_last.items() if o != i_org], dtype=float)
            n_pre = len(others)
            if n_pre >= MIN_ORG_PRE:
                C = float(np.median(others))
                MAD = float(np.median(np.abs(others - C)))
                disp = MAD_SCALE * MAD / (abs(C) + EPS)
                d_pct = (F - C) / abs(C)
                d_sym = 2 * (F - C) / (abs(F) + abs(C) + EPS)
                zden = MAD_SCALE * MAD + SCALE_FLOOR
                z = (F - C) / zden if zden > 0 else np.nan
                ib = max(0.01, 0.5 * disp)
                sb = max(0.05, 1.5 * disp)
                a = abs(d_pct)
                if a <= ib:
                    pos = "inline"
                elif a < sb:
                    pos = "above" if d_pct > 0 else "below"
                else:
                    pos = "strong_above" if d_pct > 0 else "strong_below"
            else:
                C = MAD = disp = d_pct = d_sym = z = np.nan
                pos = None
            old_self = org_last.get(i_org, np.nan)
            if pd.notna(old_self) and old_self != 0:
                self_abs = F - old_self
                self_pct = self_abs / abs(old_self)
                self_cls = classify_dir(self_pct, 0.01)
            else:
                self_abs = self_pct = self_cls = np.nan
            out.append(dict(
                publish_date=date, stock_code=r["STOCK_CODE"], fy=r["FY"],
                org_id=i_org, org_name=i_org,
                report_id=r["report_signature"], report_signature=r["report_signature"],
                author_name=r["AUTHOR_NAME"], title=r["TITLE"],
                metric=METRIC,
                forecast_new=F,
                forecast_old_self=(old_self if pd.notna(old_self) else np.nan),
                consensus_pre_ex_org=C,
                dispersion_pre_ex_org=disp,
                report_distance_pct=d_pct,
                report_distance_symmetric=d_sym,
                report_distance_robust_z=z,
                report_position_class=pos,
                self_revision_abs=self_abs,
                self_revision_pct=self_pct,
                self_revision_class=self_cls,
                n_org_pre_ex_org=n_pre,
            ))
        for _, r in grp.iterrows():
            org_last[r["ORGAN_NAME"]] = float(r["FORECAST_NP"])
    return pd.DataFrame(out)


rep = dd.groupby(["STOCK_CODE", "FY"], group_keys=False).apply(report_labels_for_group)
rep = rep.reset_index(drop=True)
rep["forecast_horizon"] = rep["fy"] - rep["publish_date"].dt.year
rep["label_valid"] = (rep["n_org_pre_ex_org"] >= MIN_ORG_PRE).astype(int)
rep["invalid_reason"] = rep["label_valid"].apply(
    lambda v: "" if v == 1 else "insufficient_pre_consensus")
rep["sample_weight"] = np.minimum(1.0, np.log(1 + rep["n_org_pre_ex_org"].clip(lower=0)) / np.log(11))
rep.loc[rep["label_valid"] == 0, "sample_weight"] = 0.0
rep.loc[rep["label_valid"] == 0, "report_position_class"] = None
rep["label_version"] = LABEL_VERSION

rep_cols = ["report_id", "report_signature", "publish_date", "stock_code", "org_id",
            "fy", "forecast_horizon", "metric", "forecast_new", "forecast_old_self",
            "consensus_pre_ex_org", "dispersion_pre_ex_org", "report_distance_pct",
            "report_distance_symmetric", "report_distance_robust_z", "report_position_class",
            "self_revision_abs", "self_revision_pct", "self_revision_class",
            "n_org_pre_ex_org", "label_valid", "invalid_reason", "sample_weight",
            "label_version", "org_name", "author_name", "title"]
rep = rep[rep_cols]
print("    报告级 Label 行数:", len(rep), "| 有效:", int(rep["label_valid"].sum()))

# ---------------------------------------------------------------------------
# 阶段 5: 保存 + 元数据
# ---------------------------------------------------------------------------
print("[5/5] 保存输出 ...")
meta = {
    "label_version": LABEL_VERSION,
    "created_at": dt.datetime.now().isoformat(timespec="seconds"),
    "source_data": SRC,
    "metric": METRIC,
    "fy_range": [FY_MIN, FY_MAX],
    "forecast_horizon": {"-1": "FY-1_pending", "0": "FY0", "1": "FY1", "2": "FY2"},
    "target_horizon_months": [1, 3],
    "staleness_days": L,
    "min_org_snapshot": MIN_ORG,
    "min_org_pre_report": MIN_ORG_PRE,
    "min_org_fixed": MIN_ORG_FIXED,
    "min_org_active_breadth": MIN_ORG_ACTIVE,
    "flat_threshold_1m": FLAT1,
    "strong_threshold_1m": STRONG1,
    "flat_threshold_3m": FLAT3,
    "strong_threshold_3m": STRONG3,
    "consensus": "机构等权 + 每机构最新有效预测 取中位数",
    "revision_class_metric": "normal positive: pct; negative/sign-change: symmetric",
    "dispersion": {"mad_scale": MAD_SCALE, "eps": EPS, "scale_floor": SCALE_FLOOR},
    "consensus_strength_dispersion_quantiles": [round(q30, 6), round(q70, 6)],
    "notes": [
        "V1 仅覆盖 FY0/FY1/FY2 (2024/2025/2026)，排除了 FY-1(2023) 与 FY3+",
        "报告相对立场基准：严格早于发布日的其他机构最新预测，排除本机构",
        "分类阈值基于相对幅度；正式上线前需在训练集内校准",
        "consensus_strength 分位数在全部有效快照上拟合(30/70)，训练/验证划分时应重新拟合",
    ],
}

for name, obj in [("consensus_snapshot_monthly", snapshots),
                  ("revision_label_monthly", revision),
                  ("report_label_detail", rep)]:
    p = os.path.join(OUT, name + ".parquet")
    obj.to_parquet(p, index=False)
    obj.to_csv(os.path.join(OUT, name + ".csv"), index=False)
    print("    已写:", p, "形状", obj.shape)

with open(os.path.join(OUT, "label_metadata.json"), "w", encoding="utf-8") as f:
    json.dump(meta, f, ensure_ascii=False, indent=2)

print("\n全部完成。输出目录:", OUT)
print("元数据:", json.dumps(meta, ensure_ascii=False, indent=2)[:1200])