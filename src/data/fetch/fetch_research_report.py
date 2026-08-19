# -*- coding: utf-8 -*-
"""抓取研报盈利预测(Oracle zyyx2):ZYYQ.RPT_FORECAST_STK。

zyyx2 为 Oracle 11.2,thin 模式不支持,须用厚模式(oracledb.init_oracle_client)。
表结构:既有研报正文 CONTENT(CLOB),也有结构化预测数值(FORECAST_*)、评级、目标价。
按 CREATE_DATE(有索引)过滤窗口。

用法:
    /home/intern_fjq_2026/miniconda3/envs/intern_fjq/bin/python \
        data_fetch/fetch_research_report.py
输出:
    <OUTPUT_ROOT>/research_report/forecast_stk_YYYYMMDD_YYYYMMDD.jsonl
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as cfg
from common import oracle_connect, rows_from_cursor, write_jsonl, write_manifest, window_dates, banner

SQL = """
SELECT ID, STOCK_CODE, STOCK_NAME, TITLE, CONTENT,
       REPORT_TYPE, RELIABILITY,
       ORGAN_NAME, AUTHOR_NAME,
       CREATE_DATE, REPORT_YEAR, REPORT_QUARTER,
       FORECAST_OR, FORECAST_OP, FORECAST_TP, FORECAST_NP, FORECAST_EPS,
       FORECAST_DPS, FORECAST_RD, FORECAST_PE, FORECAST_ROE, FORECAST_EV_EBITDA,
       ORGAN_RATING_CODE, ORGAN_RATING_CONTENT,
       GG_RATING_CODE, GG_RATING_CONTENT,
       TARGET_PRICE_CEILING, TARGET_PRICE_FLOOR, CURRENT_PRICE,
       CURRENCY, LANGUAGE, ATTENTION
FROM ZYYQ.RPT_FORECAST_STK
WHERE CREATE_DATE >= :s AND CREATE_DATE < :e
ORDER BY CREATE_DATE, STOCK_CODE
"""

COUNT_SQL = "SELECT COUNT(*) FROM ZYYQ.RPT_FORECAST_STK WHERE CREATE_DATE >= :s AND CREATE_DATE < :e"

COLS = ["ID", "STOCK_CODE", "STOCK_NAME", "TITLE", "CONTENT",
        "REPORT_TYPE", "RELIABILITY", "ORGAN_NAME", "AUTHOR_NAME",
        "CREATE_DATE", "REPORT_YEAR", "REPORT_QUARTER",
        "FORECAST_OR", "FORECAST_OP", "FORECAST_TP", "FORECAST_NP", "FORECAST_EPS",
        "FORECAST_DPS", "FORECAST_RD", "FORECAST_PE", "FORECAST_ROE", "FORECAST_EV_EBITDA",
        "ORGAN_RATING_CODE", "ORGAN_RATING_CONTENT",
        "GG_RATING_CODE", "GG_RATING_CONTENT",
        "TARGET_PRICE_CEILING", "TARGET_PRICE_FLOOR", "CURRENT_PRICE",
        "CURRENCY", "LANGUAGE", "ATTENTION"]


def main():
    s, e, e_plus = window_dates()
    out_dir = os.path.join(cfg.OUTPUT_ROOT, "research_report")
    out_file = os.path.join(out_dir, f"forecast_stk_{s:%Y%m%d}_{e:%Y%m%d}.jsonl")

    t0 = time.time()
    conn = oracle_connect("zyyx2")
    cur = conn.cursor()
    cur.arraysize = 1000
    cur.execute(COUNT_SQL, s=s, e=e_plus)
    total = cur.fetchone()[0]
    cur.execute(SQL, s=s, e=e_plus)
    n = write_jsonl(out_file, COLS, rows_from_cursor(cur), total=total, desc="research_report")
    conn.close()

    write_manifest(out_dir, {
        "source": "research_report", "script": "fetch_research_report.py",
        "window": {"start": str(s), "end": str(e)},
        "rows": n, "file": os.path.basename(out_file), "columns": COLS,
        "note": "CONTENT 为研报正文(CLOB);FORECAST_* 为盈利预测数值。",
    })
    banner("fetch_research_report", n, out_file, time.time() - t0, "Oracle zyyx2")


if __name__ == "__main__":
    main()
