# -*- coding: utf-8 -*-
"""取数脚本共享工具:数据库连接、流式 JSONL 写入、清单、窗口日期。"""
import os
import json
import sys
from datetime import datetime, date, timedelta

import config as cfg
from progress import ProgressBar


# ---------------- 窗口日期 ----------------
def window_dates():
    """返回 (start, end_inclusive, end_plus1),end 为闭区间。"""
    s = datetime.strptime(cfg.START_DATE, "%Y-%m-%d").date()
    e = datetime.strptime(cfg.END_DATE, "%Y-%m-%d").date()
    return s, e, e + timedelta(days=1)


# ---------------- Oracle ----------------
def oracle_connect(which):
    """which: 'wind'(thin 直连) 或 'zyyx2'(厚模式,库版本 11.2)。"""
    import oracledb
    oracledb.defaults.fetch_lobs = False   # CLOB 直接返回 str
    if which == "wind":
        return oracledb.connect(user=cfg.WIND_USER, password=cfg.WIND_PWD, dsn=cfg.WIND_DSN)
    if which == "zyyx2":
        oracledb.init_oracle_client(lib_dir=cfg.ORACLE_LIB_DIR)
        return oracledb.connect(user=cfg.ZYYX2_USER, password=cfg.ZYYX2_PWD, dsn=cfg.ZYYX2_DSN)
    raise ValueError(which)


def _norm(v):
    """把取到的值规整为可 JSON 序列化的类型。"""
    if v is None:
        return None
    if isinstance(v, (datetime, date)):
        return v.isoformat(sep=" ") if isinstance(v, datetime) else v.isoformat()
    if hasattr(v, "read"):          # LOB 兜底(fetch_lobs 未关时)
        return v.read()
    return v


# ---------------- 游标行拍平 ----------------
def rows_from_cursor(cur, batch=1000):
    """把 fetchmany 返回的批次拍平成单行序列(避免整表载入内存)。"""
    while True:
        rows = cur.fetchmany(batch)
        if not rows:
            break
        for r in rows:
            yield r


# ---------------- 流式写 JSONL ----------------
def write_jsonl(path, col_names, row_iter, total=None, desc=None):
    """流式写 JSONL,col_names 与每行顺序对应。返回写入行数。total 已知时显示百分比进度。"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    n = 0
    pb = ProgressBar(total=total, desc=desc or os.path.basename(path))
    with open(path, "w", encoding="utf-8") as f:
        for row in row_iter:
            rec = {c: _norm(v) for c, v in zip(col_names, row)}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1
            pb.update()
    pb.close()
    return n


# ---------------- 清单 ----------------
def write_manifest(out_dir, meta):
    with open(os.path.join(out_dir, "_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"manifest -> {os.path.join(out_dir, '_manifest.json')}")


def banner(script, rows, out_file, secs, src):
    print(f"[{src}] {script}: {rows} rows -> {out_file} ({secs:.1f}s)")
