# -*- coding: utf-8 -*-
"""抓取调研问答(Wind 镜像 Oracle):ASHAREISQA。

表自带 S_INFO_WINDCODE(股票)与 S_ASKDATE('YYYYMMDD' 字符串)。
按 S_ASKDATE 过滤窗口(纯字符串区间,无需转日期)。

用法:
    /home/intern_fjq_2026/miniconda3/envs/intern_fjq/bin/python \
        data_fetch/fetch_field_research.py
输出:
    <OUTPUT_ROOT>/field_research/isqa_YYYYMMDD_YYYYMMDD.jsonl
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as cfg
from common import oracle_connect, rows_from_cursor, write_jsonl, write_manifest, window_dates, banner

SQL = """
SELECT S_INFO_WINDCODE,
       S_ASKDATE,
       S_ANSWERDATE,
       EVENT_ID,
       S_QUESTIONTYPE,
       S_QUESTIONCONTENT,
       S_ANSWERCONTENT,
       S_QUESTIONSOURCETYPE,
       QUESTION_PERSON,
       ANSWER_PERSON,
       ORIGINALWEBSITE
FROM wind.ASHAREISQA
WHERE S_ASKDATE >= :s AND S_ASKDATE <= :e
ORDER BY S_ASKDATE, S_INFO_WINDCODE
"""

COUNT_SQL = "SELECT COUNT(*) FROM wind.ASHAREISQA WHERE S_ASKDATE >= :s AND S_ASKDATE <= :e"

COLS = ["S_INFO_WINDCODE", "S_ASKDATE", "S_ANSWERDATE", "EVENT_ID", "S_QUESTIONTYPE",
        "S_QUESTIONCONTENT", "S_ANSWERCONTENT", "S_QUESTIONSOURCETYPE",
        "QUESTION_PERSON", "ANSWER_PERSON", "ORIGINALWEBSITE"]


def main():
    s, e, _ = window_dates()
    s_str, e_str = s.strftime("%Y%m%d"), e.strftime("%Y%m%d")
    out_dir = os.path.join(cfg.OUTPUT_ROOT, "field_research")
    out_file = os.path.join(out_dir, f"isqa_{s_str}_{e_str}.jsonl")

    t0 = time.time()
    conn = oracle_connect("wind")
    cur = conn.cursor()
    cur.arraysize = 1000
    cur.execute(COUNT_SQL, s=s_str, e=e_str)
    total = cur.fetchone()[0]
    cur.execute(SQL, s=s_str, e=e_str)
    n = write_jsonl(out_file, COLS, rows_from_cursor(cur), total=total, desc="field_research")
    conn.close()

    write_manifest(out_dir, {
        "source": "field_research", "script": "fetch_field_research.py",
        "window": {"start": str(s), "end": str(e)},
        "rows": n, "file": os.path.basename(out_file), "columns": COLS,
    })
    banner("fetch_field_research", n, out_file, time.time() - t0, "Oracle wind")


if __name__ == "__main__":
    main()
