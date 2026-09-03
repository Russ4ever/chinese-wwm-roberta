#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""从 gubapost 日文件采样文本, 生成 activation-rank 输入 parquet。

activation_rank pipeline 按 token budget(最多 10M token) + hash 采样处理输入,
因此只需一个有代表性的子集即可——不必全量 3.35 亿帖。
本脚本逐日文件随机采样 N 帖, 合并 title+content 为 text 列。

用法:
  python scripts/sample_gubapost_text.py
  python scripts/sample_gubapost_text.py --n-per-file 500 --years 2020,2021,2022,2023
"""
from __future__ import annotations

import argparse
import glob
import os
import random
import re
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_INPUT = "/home/intern_fjq_2026/data/NLP/gubapost"
DEFAULT_OUT = os.path.join(ROOT, "artifacts", "gubapost_baseline", "gubapost_text_sample.parquet")
DEFAULT_YEARS = ["2020", "2021", "2022", "2023"]


def merge_text(title: str, content: str) -> str:
    """title+content 合并; content 已包含 title 或 title 为空时不重复。"""
    ti = (title or "").strip()
    ci = (content or "").strip()
    if not ci:
        return ti
    if not ti or ci.startswith(ti):
        return ci
    return ti + " " + ci


def process_file(args: tuple) -> pd.DataFrame:
    path, n_per_file, seed = args
    df = pd.read_parquet(path, columns=["postId", "title", "content"])
    if len(df) > n_per_file:
        df = df.sample(n=n_per_file, random_state=seed)
    texts = [merge_text(t, c) for t, c in zip(df["title"].values, df["content"].values)]
    return pd.DataFrame({"post_id": df["postId"].astype(str).values, "text": texts})


def main():
    ap = argparse.ArgumentParser(description="采样 gubapost 文本, 生成 activation-rank 输入")
    ap.add_argument("--input-dir", default=DEFAULT_INPUT)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--years", default=",".join(DEFAULT_YEARS))
    ap.add_argument("--n-per-file", type=int, default=300)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    years = args.years.split(",")
    files = []
    for y in years:
        files += sorted(glob.glob(os.path.join(args.input_dir, y, "gubapost*.parquet")))
    files = [f for f in files if re.fullmatch(r"gubapost\d{8}\.parquet", os.path.basename(f))]
    assert files, f"没有找到文件, 检查 {args.input_dir}/{years}"
    print(f"共 {len(files)} 个日文件, 每文件采样 {args.n_per_file} 帖, workers={args.workers}")

    rng = random.Random(args.seed)
    file_seeds = [rng.randint(0, 2**31) for _ in files]

    from concurrent.futures import ProcessPoolExecutor

    parts = []
    with ProcessPoolExecutor(args.workers) as ex:
        for i, result in enumerate(ex.map(
            process_file,
            [(f, args.n_per_file, s) for f, s in zip(files, file_seeds)],
            chunksize=4,
        )):
            parts.append(result)
            if (i + 1) % 200 == 0:
                print(f"  {i+1}/{len(files)}", flush=True)

    df = pd.concat(parts, ignore_index=True)
    df = df[df["text"].str.len() > 0]  # 过滤空文本
    df = df.drop_duplicates(subset="post_id").reset_index(drop=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df.to_parquet(args.out, index=False)
    lens = df["text"].str.len()
    print(f"采样完成: {len(df)} 行 -> {args.out}")
    print(f"text 长度: mean={lens.mean():.0f}, median={lens.median():.0f}, "
          f"p95={lens.quantile(0.95):.0f}, max={lens.max()}")
    print(f"估算 token: ~{int(lens.sum() / 1.5):,} (按 1.5 char/token 粗估)")


if __name__ == "__main__":
    main()
