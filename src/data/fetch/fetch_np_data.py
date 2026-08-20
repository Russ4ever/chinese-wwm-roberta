# -*- coding: utf-8 -*-
"""从 wind 镜像 Oracle 取 AShareIncome 利润表字段。

字段: S_INFO_WINDCODE, ANN_DT, REPORT_PERIOD, NET_PROFIT_INCL_MIN_INT_INC, ACTUAL_ANN_DT
条件: ANN_DT 在 [20050101, 20260801] 闭区间(VARCHAR2 'YYYYMMDD', 字符串比较=日期序)
输出: 与本脚本同目录 ashare_income_20050101_20260801.csv
运行: /home/fanjingqi/software/miniconda3/envs/factors/bin/python fetch_ashare_income.py
"""
import os
import csv
import time

import oracledb

WIND_USER = "db"
WIND_PWD = "2021"
WIND_DSN = ":21010/wind"

START = "20050101"
END = "20260801"

COLUMNS = [
    "S_INFO_WINDCODE",
    "ANN_DT",
    "REPORT_PERIOD",
    "NET_PROFIT_INCL_MIN_INT_INC",
    "ACTUAL_ANN_DT",
]

OUT_DIR = "/home/intern_fjq_2026/data/NLP/market"
OUT_FILE = os.path.join(OUT_DIR, "ashare_np_20050101_20260801.csv")

SQL = (
    "SELECT " + ", ".join(COLUMNS)
    + " FROM WIND.ASHAREINCOME"
    + " WHERE ANN_DT >= :s AND ANN_DT <= :e"
)


def main():
    oracledb.defaults.fetch_lobs = False
    conn = oracledb.connect(user=WIND_USER, password=WIND_PWD, dsn=WIND_DSN)
    cur = conn.cursor()
    cur.arraysize = 1000

    t0 = time.time()
    cur.execute(SQL, s=START, e=END)

    n = 0
    with open(OUT_FILE, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(COLUMNS)
        while True:
            rows = cur.fetchmany(1000)
            if not rows:
                break
            w.writerows(rows)
            n += len(rows)
            if n % 500000 == 0:
                print("written %d rows (%.1fs)" % (n, time.time() - t0))

    conn.close()
    print("DONE: %d rows -> %s (%.1fs)" % (n, OUT_FILE, time.time() - t0))


if __name__ == "__main__":
    main()