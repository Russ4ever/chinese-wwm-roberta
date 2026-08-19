# -*- coding: utf-8 -*-
"""抓取公司公告(Wind 镜像 Oracle):ASHAREANNINF + ASHAREANNTEXT。

正文在 ASHAREANNTEXT.ANN_TEXT(CLOB),按 OBJECT_ID(GUID) 与
ASHAREANNINF.OBJECT_ID 关联(两端均有主键索引,LEFT JOIN 保留无正文的公告)。
按 ANN_DT(有索引)过滤日期窗口。

用法:
    /home/intern_fjq_2026/miniconda3/envs/intern_fjq/bin/python \
        data_fetch/fetch_company_filings.py
输出:
    <OUTPUT_ROOT>/company_filings/announcements_YYYYMMDD_YYYYMMDD.jsonl
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as cfg
from common import oracle_connect, rows_from_cursor, write_jsonl, write_manifest, window_dates, banner

SQL = """
SELECT a.S_INFO_WINDCODE,
       a.ANN_DT,
       a.N_INFO_TITLE,
       a.N_INFO_ANNLINK,
       t.ANN_TEXT
FROM wind.ASHAREANNINF a
LEFT JOIN wind.ASHAREANNTEXT t ON a.OBJECT_ID = t.OBJECT_ID
WHERE a.ANN_DT >= :s AND a.ANN_DT < :e
ORDER BY a.ANN_DT, a.S_INFO_WINDCODE
"""

COUNT_SQL = """
SELECT COUNT(*)
FROM wind.ASHAREANNINF a
LEFT JOIN wind.ASHAREANNTEXT t ON a.OBJECT_ID = t.OBJECT_ID
WHERE a.ANN_DT >= :s AND a.ANN_DT < :e
"""

COLS = ["S_INFO_WINDCODE", "ANN_DT", "N_INFO_TITLE", "N_INFO_ANNLINK", "ANN_TEXT"]


def main():
    s, e, e_plus = window_dates()
    # out_dir = os.path.join(cfg.OUTPUT_ROOT, "company_filings")
    out_dir = os.path.join("/home/intern_fjq_2026/Projects/chinese-wwm-roberta/src/data/out_test", "company_filings")
    out_file = os.path.join(out_dir, f"announcements_{s:%Y%m%d}_{e:%Y%m%d}.jsonl")

    t0 = time.time()
    conn = oracle_connect("wind")
    cur = conn.cursor()
    cur.arraysize = 1000
    cur.execute(COUNT_SQL, s=s, e=e_plus)
    total = cur.fetchone()[0]
    cur.execute(SQL, s=s, e=e_plus)
    n = write_jsonl(out_file, COLS, rows_from_cursor(cur), total=total, desc="company_filings")
    conn.close()

    write_manifest(out_dir, {
        "source": "company_filings", "script": "fetch_company_filings.py",
        "window": {"start": str(s), "end": str(e)},
        "rows": n, "file": os.path.basename(out_file), "columns": COLS,
        "note": "ANN_TEXT 为公告正文(CLOB);部分公告无正文(LEFT JOIN 保留)。",
    })
    banner("fetch_company_filings", n, out_file, time.time() - t0, "Oracle wind")


if __name__ == "__main__":
    main()
