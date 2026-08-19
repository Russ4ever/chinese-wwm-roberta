# -*- coding: utf-8 -*-
"""取 2023-07 行情数据(M8-1)：日行情 + 证券主表 + 交易日历。

复用 common.py 的 oracle_connect / rows_from_cursor / _norm。
"""
import sys, os, json, datetime, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as cfg
from common import oracle_connect, rows_from_cursor, _norm

SQL_PRICE = """
SELECT S_INFO_WINDCODE, TRADE_DT, S_DQ_PRECLOSE, S_DQ_CLOSE, S_DQ_ADJCLOSE,
       S_DQ_VOLUME, S_DQ_AMOUNT, S_DQ_TRADESTATUS
FROM wind.ASHAREEODPRICES
WHERE TRADE_DT >= :s AND TRADE_DT < :e
ORDER BY TRADE_DT, S_INFO_WINDCODE
"""
PRICE_COLS = ["S_INFO_WINDCODE", "TRADE_DT", "S_DQ_PRECLOSE", "S_DQ_CLOSE",
              "S_DQ_ADJCLOSE", "S_DQ_VOLUME", "S_DQ_AMOUNT", "S_DQ_TRADESTATUS"]

SQL_DESC = """
SELECT S_INFO_WINDCODE, S_INFO_NAME, S_INFO_EXCHMARKET, S_INFO_LISTBOARD,
       S_INFO_LISTDATE, S_INFO_DELISTDATE, IS_DELISTED
FROM wind.ASHAREDESCRIPTION
"""
DESC_COLS = ["S_INFO_WINDCODE", "S_INFO_NAME", "S_INFO_EXCHMARKET",
             "S_INFO_LISTBOARD", "S_INFO_LISTDATE", "S_INFO_DELISTDATE", "IS_DELISTED"]


def write_jsonl(path, cols, row_iter):
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for row in row_iter:
            rec = {c: _norm(v) for c, v in zip(cols, row)}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1
    return n


def main():
    # TRADE_DT 是 VARCHAR2 'YYYYMMDD'，用字符串比较(字典序 = 日期序)
    s = "20230701"
    e = "20230801"
    out_dir = os.path.join(cfg.OUTPUT_ROOT, "market")
    os.makedirs(out_dir, exist_ok=True)

    conn = oracle_connect("wind")
    cur = conn.cursor()
    cur.arraysize = 1000
    t0 = time.time()

    # 1. 日行情
    cur.execute(SQL_PRICE, s=s, e=e)
    n_price = write_jsonl(os.path.join(out_dir, "eod_prices_202307.jsonl"),
                          PRICE_COLS, rows_from_cursor(cur))
    print("eod_prices: %d rows (%.1fs)" % (n_price, time.time() - t0))

    # 2. 证券主表
    t1 = time.time()
    cur.execute(SQL_DESC)
    n_desc = write_jsonl(os.path.join(out_dir, "stock_description.jsonl"),
                         DESC_COLS, rows_from_cursor(cur))
    print("stock_description: %d rows (%.1fs)" % (n_desc, time.time() - t1))

    conn.close()
    print("DONE")


if __name__ == "__main__":
    main()
