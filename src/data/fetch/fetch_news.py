# -*- coding: utf-8 -*-
"""抓取新闻(datayes MySQL),三个渠道:新闻 / 微信 / 快讯。

表结构发现(2026-08 验证):
- 正文表 vnews_body_*_s3 只有 NEWS_ID + S3_URL(对象键 /newsbody/...),正文存于 S3;
- 元数据在 vnews_content_v1 / vnews_content_wechat / news_content_flash
  (NEWS_TITLE / NEWS_PUBLISH_TIME / NEWS_ORIGIN_SOURCE / NEWS_URL 等);
- vnews_summary_v1 有 NEWS_SUMMARY 文本,可作正文替代;
- 日期过滤必须走 EFFECTIVE_TIME(有索引,≈NEWS_PUBLISH_TIME),NEWS_PUBLISH_TIME 无索引。

S3 访问方式未知(config.S3_ENDPOINT 为空),脚本会落 S3_URL 键,待有 S3 凭据后
用 fetch_s3_body 补抓正文。数据量大(一个月 v1 约 230 万条),全程服务端游标流式写。

用法:
    /home/intern_fjq_2026/miniconda3/envs/intern_fjq/bin/python \
        data_fetch/fetch_news.py [--limit N]
输出:
    <OUTPUT_ROOT>/news/{news,wechat,flash}_YYYYMMDD_YYYYMMDD.jsonl
"""
import os
import sys
import time
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as cfg
from common import write_jsonl, write_manifest, window_dates, banner

# 渠道: (标签, 内容表, 正文本表)
CHANNELS = [
    ("news",   "vnews_content_v1",    "vnews_body_v1_s3"),
    ("wechat", "vnews_content_wechat", "vnews_body_wechat_s3"),
    ("flash",  "news_content_flash",   "vnews_body_flash_s3"),
]

SQL = """
SELECT c.NEWS_ID, c.NEWS_TITLE, c.NEWS_PUBLISH_TIME, c.EFFECTIVE_TIME,
       c.NEWS_ORIGIN_SOURCE, c.NEWS_AUTHOR, c.NEWS_PUBLISH_SITE, c.NEWS_URL,
       b.S3_URL, s.NEWS_SUMMARY
FROM `{ct}` c
LEFT JOIN `{bt}` b ON c.NEWS_ID = b.NEWS_ID
LEFT JOIN vnews_summary_v1 s ON c.NEWS_ID = s.NEWS_ID
WHERE c.EFFECTIVE_TIME >= %s AND c.EFFECTIVE_TIME < %s
ORDER BY c.EFFECTIVE_TIME
"""

COUNT_SQL = """
SELECT COUNT(*)
FROM `{ct}` c
LEFT JOIN `{bt}` b ON c.NEWS_ID = b.NEWS_ID
LEFT JOIN vnews_summary_v1 s ON c.NEWS_ID = s.NEWS_ID
WHERE c.EFFECTIVE_TIME >= %s AND c.EFFECTIVE_TIME < %s
"""

COLS = ["NEWS_ID", "NEWS_TITLE", "NEWS_PUBLISH_TIME", "EFFECTIVE_TIME",
        "NEWS_ORIGIN_SOURCE", "NEWS_AUTHOR", "NEWS_PUBLISH_SITE", "NEWS_URL",
        "S3_URL", "NEWS_SUMMARY"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="每渠道最多抓取条数(0=不限)")
    args = ap.parse_args()

    s, e, e_plus = window_dates()
    start_dt = f"{s} 00:00:00"
    end_dt = f"{e_plus} 00:00:00"   # 闭区间 [s, e],用 < e+1
    out_dir = os.path.join(cfg.OUTPUT_ROOT, "news")
    os.makedirs(out_dir, exist_ok=True)

    import pymysql

    conn = pymysql.connect(
        host=cfg.MYSQL_HOST, port=cfg.MYSQL_PORT, user=cfg.MYSQL_USER,
        password=cfg.MYSQL_PWD, database=cfg.MYSQL_DB, charset="utf8mb4",
        connect_timeout=15, cursorclass=pymysql.cursors.SSCursor,  # 服务端游标,流式
    )

    t0 = time.time()
    total = 0
    for tag, ct, bt in CHANNELS:
        out_file = os.path.join(out_dir, f"{tag}_{s:%Y%m%d}_{e:%Y%m%d}.jsonl")
        with conn.cursor(pymysql.cursors.Cursor) as ccur:
            ccur.execute(COUNT_SQL.format(ct=ct, bt=bt), (start_dt, end_dt))
            ch_total = ccur.fetchone()[0]
        if args.limit:
            ch_total = min(ch_total, int(args.limit))

        cur = conn.cursor()
        sql = SQL.format(ct=ct, bt=bt)
        if args.limit:
            sql += f" LIMIT {int(args.limit)}"
        cur.execute(sql, (start_dt, end_dt))

        def rows():
            while True:
                batch = cur.fetchmany(5000)
                if not batch:
                    break
                for r in batch:
                    yield r
        n = write_jsonl(out_file, COLS, rows(), total=ch_total, desc=f"[{tag}]")
        cur.close()
        total += n
        print(f"  [{tag:6s}] {n:>9,} rows -> {os.path.basename(out_file)}")

    conn.close()

    write_manifest(out_dir, {
        "source": "news", "script": "fetch_news.py",
        "window": {"start": str(s), "end": str(e)},
        "channels": {t: f"{t}_{s:%Y%m%d}_{e:%Y%m%d}.jsonl" for t, _, _ in CHANNELS},
        "rows": total, "columns": COLS,
        "note": "正文在 S3(S3_URL 为对象键),config.S3_ENDPOINT 配好后可补抓;"
                "NEWS_SUMMARY 为摘要文本;日期过滤用 EFFECTIVE_TIME(≈发布时刻,有索引)。",
    })
    banner("fetch_news", total, out_dir, time.time() - t0, "MySQL datayes")


if __name__ == "__main__":
    main()
