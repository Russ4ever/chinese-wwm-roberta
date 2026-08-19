# -*- coding: utf-8 -*-
"""数据库工具:列结构 / 表清单 / 按列名搜表。
用法:
  python db_schema.py                        # 列出 WIND schema 相关表
  python db_schema.py ASHAREANNINF           # 查看某表全部列
  python db_schema.py --all WIND             # 该 schema 所有表
  python db_schema.py --searchcol EPS WIND   # 按字段名搜表(指定 schema)
  python db_schema.py --searchcol EPS        # 全 schema 搜
"""
import sys, os, re
from collections import defaultdict
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as cfg


def get_conn(which="wind"):
    import cx_Oracle
    cx_Oracle.init_oracle_client(lib_dir=cfg.ORACLE_LIB_DIR)
    if which == "wind":
        return cx_Oracle.connect(user=cfg.WIND_USER, password=cfg.WIND_PWD, dsn=cfg.WIND_DSN)
    if which == "zyyx2":
        return cx_Oracle.connect(user=cfg.ZYYX2_USER, password=cfg.ZYYX2_PWD, dsn=cfg.ZYYX2_DSN)
    raise ValueError(which)


def show_columns(conn, owner, table):
    cur = conn.cursor()
    cur.execute(
        "SELECT column_name, data_type, data_length, nullable "
        "FROM all_tab_columns WHERE owner=:o AND table_name=:t ORDER BY column_id",
        o=owner, t=table.upper())
    rows = cur.fetchall()
    print("\n== %s.%s (%d cols) ==" % (owner, table.upper(), len(rows)))
    for r in rows:
        print("  %-34s %-14s len=%s  null=%s" % (r[0], r[1], r[2], r[3]))
    cur.close()


def list_tables(conn, owner):
    cur = conn.cursor()
    cur.execute("SELECT table_name FROM all_tables WHERE owner=:o ORDER BY table_name", o=owner)
    tables = [r[0] for r in cur.fetchall()]
    cur.close()
    print("%s schema 共 %d 张表" % (owner, len(tables)))
    return tables


def search_columns(conn, owner, keyword):
    cur = conn.cursor()
    sql = ("SELECT owner, table_name, column_name, data_type "
           "FROM all_tab_columns WHERE column_name LIKE :k")
    params = {"k": "%" + keyword.upper() + "%"}
    if owner:
        sql += " AND owner=:o"
        params["o"] = owner
    sql += " ORDER BY owner, table_name, column_id"
    cur.execute(sql, params)
    rows = cur.fetchall()
    cur.close()
    if not rows:
        print("未找到含 %r 的列" % keyword)
        return
    bytab = defaultdict(list)
    for own, tab, col, dt in rows:
        bytab[(own, tab)].append("%s(%s)" % (col, dt))
    print("含 %r 的列: %d 个, 分布在 %d 张表:" % (keyword, len(rows), len(bytab)))
    for (own, tab), cols in bytab.items():
        print("  %s.%s: %s" % (own, tab, ", ".join(cols)))


if __name__ == "__main__":
    conn = get_conn("wind")
    args = sys.argv[1:]
    if not args:
        list_tables(conn, "WIND")
    elif args[0] == "--all":
        tables = list_tables(conn, args[1].upper())
        for t in tables:
            show_columns(conn, args[1].upper(), t)
    elif args[0] == "--searchcol":
        kw = args[1]
        owner = args[2].upper() if len(args) > 2 else None
        search_columns(conn, owner, kw)
    else:
        for t in args:
            show_columns(conn, "WIND", t)
    conn.close()
