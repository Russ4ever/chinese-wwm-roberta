#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""social_text 批量推理（预缓存版）——取代旧版(边 tokenize 边前向)。

架构:
  1) 预缓存: 一次性多进程(默认 ≤32 核)把整月文本 tokenize 成 input_ids(int32) 存 .npy
  2) 纯前向: 推理阶段只从缓存取 tensor 前向, GPU 不再等 CPU -> 吞吐逼近显卡上限
  缓存按 (month, max_length) 复用, 换 pooling / dtype / 单卡双卡 都不用重新 tokenize。

用法:
  python scripts/infer_social_text.py --month 202403 --gpus all
  python scripts/infer_social_text.py --month 202403 --gpus 0
  python scripts/infer_social_text.py --month 202403 --gpus 1 --dtype fp32
  python scripts/infer_social_text.py --month 202403 --gpus 0 --limit 200000   # 试水

可选:
  --batch-size 1024  --max-length 128  --pooling cls|pooler|masked_mean
  --dtype fp16|fp32  --token-workers N  --re-tokenize(强制重建缓存)

产物:
  缓存: artifacts/token_cache/social_text_<month>_len<maxlen>_ids.npy (+ .meta.json)
  结果: artifacts/social_text_probs_<month>.csv
  列: id, class_0_prob, class_1_prob, symbol, published_at, date, available_date, source
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.inference_cache import save_csv  # noqa: E402

DEFAULT_TRADING_CALENDAR = os.environ.get(
    "SOCIAL_TEXT_TRADING_CALENDAR",
    os.path.join(ROOT, "data", "RTN_daily", "rtn_1d.parquet"),
)


def parse_args():
    ap = argparse.ArgumentParser(description="social_text 批量推理(预缓存版)")
    ap.add_argument("--month", required=True, help="月份, 如 202403")
    ap.add_argument(
        "--gpus",
        default=os.environ.get("SOCIAL_TEXT_GPUS", "1"),
        help="0 | 1 | 0,1 | all | cpu（默认仅GPU 1；GPU 0须先确认空闲）",
    )
    ap.add_argument("--batch-size", type=int, default=1024)
    ap.add_argument("--max-length", type=int, default=128)
    ap.add_argument("--pooling", default="cls", choices=["cls", "pooler", "masked_mean"])
    ap.add_argument("--dtype", default="fp16", choices=["fp16", "fp32"])
    ap.add_argument("--limit", type=int, default=None, help="只处理前 N 行(None=全部)")
    ap.add_argument("--token-workers", type=int, default=None, help="tokenize 进程数(默认 min(32, 核数))")
    ap.add_argument("--re-tokenize", action="store_true", help="忽略已有缓存强制重建")
    ap.add_argument("--data-dir", default=os.path.join(ROOT, "filtered_data"))
    ap.add_argument("--out-dir", default=os.path.join(ROOT, "artifacts"))
    ap.add_argument(
        "--trading-calendar",
        default=DEFAULT_TRADING_CALENDAR,
        help="项目已有交易日文件（CSV/Parquet，含 date 列）",
    )
    ap.add_argument("--part", type=int, default=None, help="[内部] 分片序号")
    ap.add_argument("--num-parts", type=int, default=1, help="[内部] 分片数")
    args = ap.parse_args()
    if not re.fullmatch(r"\d{6}", args.month):
        ap.error("--month 必须是 YYYYMM，例如 202403")
    month_number = int(args.month[4:])
    if not 1 <= month_number <= 12:
        ap.error("--month 的月份必须在 01..12")
    for name in ("batch_size", "max_length", "num_parts"):
        if getattr(args, name) <= 0:
            ap.error(f"--{name.replace('_', '-')} 必须大于 0")
    if args.limit is not None and args.limit <= 0:
        ap.error("--limit 必须大于 0")
    if args.token_workers is not None and args.token_workers <= 0:
        ap.error("--token-workers 必须大于 0")
    if args.part is not None and not 0 <= args.part < args.num_parts:
        ap.error("--part 必须满足 0 <= part < num-parts")
    return args


# --------------------------------------------------------------------------- #
# 预缓存
# --------------------------------------------------------------------------- #
def ensure_cache(args):
    """返回 (cache_npy, meta)。缓存失效(源文件变动或强制)则多进程 tokenize 重建。"""
    cache_dir = os.path.join(args.out_dir, "token_cache")
    os.makedirs(cache_dir, exist_ok=True)
    tag = args.month if args.limit is None else f"{args.month}_lim{args.limit}"
    cache_npy = os.path.join(cache_dir, f"social_text_{tag}_len{args.max_length}_ids.npy")
    cache_meta = cache_npy + ".meta.json"
    data_path = os.path.join(args.data_dir, f"social_text_{args.month}.csv.gz")

    from src.config import load_yaml_config
    from src.inference_cache import (
        file_signature,
        load_json_object,
        save_json_object,
        save_npy,
        tokenize_texts_parallel,
        tokenizer_signature,
        validate_cached_ids,
    )

    cfg = load_yaml_config(os.path.join(ROOT, "configs", "model.yaml"))
    base = os.path.join(ROOT, cfg["paths"]["base_model_dir"])
    expected = {
        "source": file_signature(data_path),
        "max_length": args.max_length,
        "tokenizer_sha256": tokenizer_signature(base),
    }

    if os.path.exists(cache_npy) and os.path.exists(cache_meta) and not args.re_tokenize:
        try:
            meta = load_json_object(cache_meta)
            if all(meta.get(k) == v for k, v in expected.items()):
                validate_cached_ids(
                    cache_npy,
                    n_rows=int(meta["n"]),
                    max_length=args.max_length,
                )
                print(f"缓存复用: {cache_npy} (n={meta['n']})", flush=True)
                return cache_npy, meta
        except (OSError, TypeError, ValueError, KeyError):
            pass
        print("缓存失效, 重建", flush=True)

    import pandas as pd
    df = pd.read_csv(data_path)
    required = {"id", "text", "symbol", "published_at", "source"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"输入 CSV 缺少必需列: {missing}")
    if args.limit is not None:
        df = df.iloc[: args.limit]
    n = len(df)
    if n == 0:
        raise ValueError(f"输入 CSV 没有可处理的行: {data_path}")
    texts = df["text"].fillna("").tolist()

    nw = args.token_workers if args.token_workers is not None else min(32, os.cpu_count() or 1)
    print(f"预 tokenize: {n} 条, max_length={args.max_length}, workers={nw}, "
          "开始构建缓存", flush=True)
    t0 = time.time()
    ids, pad = tokenize_texts_parallel(
        texts,
        base_dir=base,
        max_length=args.max_length,
        workers=nw,
    )
    save_npy(cache_npy, ids)
    meta = {"n": int(n), "pad_token_id": pad, **expected}
    save_json_object(cache_meta, meta)
    print(f"预 tokenize 完成: {cache_npy} 形状 {ids.shape}, "
          f"用时 {time.time()-t0:.1f}s ({n/max(time.time()-t0,1e-6):.0f} 条/s)", flush=True)
    return cache_npy, meta


# --------------------------------------------------------------------------- #
# 纯前向(读缓存, 不经 tokenizer)
# --------------------------------------------------------------------------- #
def run_inference(args, cache_npy, meta, lo, hi):
    import torch
    import pandas as pd

    from src.config import load_yaml_config
    from src.models.modeling import build_candidate

    cfg = load_yaml_config(os.path.join(ROOT, "configs", "model.yaml"))
    base = os.path.join(ROOT, cfg["paths"]["base_model_dir"])
    ckpt = os.path.join(ROOT, cfg["paths"]["checkpoint"])
    wants_cpu = str(args.gpus).lower() == "cpu"
    if not wants_cpu and not torch.cuda.is_available():
        raise RuntimeError(f"请求 GPU {args.gpus}，但当前进程没有可用 CUDA 设备")
    device = "cpu" if wants_cpu else "cuda"
    if device == "cpu" and args.dtype == "fp16":
        print("[warn] CPU 推理不使用 fp16，已回退到 fp32", flush=True)
    dtype = torch.float16 if args.dtype == "fp16" and device == "cuda" else torch.float32
    gpu_name = torch.cuda.get_device_name(0) if device == "cuda" else "CPU"
    print(f"[part {args.part}/{args.num_parts}] device={device} ({gpu_name}) "
          f"dtype={args.dtype} pooling={args.pooling} batch={args.batch_size}", flush=True)

    model = build_candidate(base, ckpt, pooling=args.pooling, pooling_confirmed=False,
                            device=device, dtype=dtype).eval()
    pad = int(meta["pad_token_id"])

    df = pd.read_csv(os.path.join(args.data_dir, f"social_text_{args.month}.csv.gz"))
    if args.limit is not None:
        df = df.iloc[: args.limit]
    df = df.iloc[lo:hi].reset_index(drop=True)
    k = len(df)
    if k != hi - lo:
        raise ValueError(f"数据分片行数 {k} 与缓存范围 {hi - lo} 不一致")
    ids = df["id"].astype(str).tolist()
    from src.trading_calendar import align_to_trading_day, load_trading_dates, parse_market_timestamps

    timestamps = parse_market_timestamps(df["published_at"])
    aligned_dates = align_to_trading_day(timestamps, load_trading_dates(args.trading_calendar))

    mm = np.load(cache_npy, mmap_mode="r")            # [N,128] int32, mmap 懒加载
    if mm.shape != (int(meta["n"]), int(meta["max_length"])):
        raise ValueError(f"缓存 shape 与元数据不一致: {mm.shape}")
    out0 = torch.empty(k, dtype=torch.float32)
    out1 = torch.empty(k, dtype=torch.float32)

    SB = 65536                                          # 超级批: 每次从 mmap 取这么多行
    t0 = time.time()
    with torch.inference_mode():
        for s in range(0, k, SB):
            e = min(k, s + SB)
            ids_t = torch.from_numpy(mm[lo + s: lo + e].astype("int64"))
            for i in range(0, ids_t.shape[0], args.batch_size):
                j = min(ids_t.shape[0], i + args.batch_size)
                b = ids_t[i:j].to(device)
                mask = (b != pad)
                probs = model(b, attention_mask=mask).probabilities.float()
                out0[s + i:s + j] = probs[:, 0].cpu()
                out1[s + i:s + j] = probs[:, 1].cpu()
            done = e
            el = time.time() - t0
            rate = done / el if el > 0 else 0
            eta = (k - done) / rate if rate > 0 else 0
            print(f"  {done}/{k}  {rate:.0f}条/s  ETA {eta/60:.1f}min", flush=True)
    dt = time.time() - t0
    print(f"[part {args.part}] 完成 {k} 行, 前向耗时 {dt:.1f}s, 吞吐 {k/dt:.0f} 条/s", flush=True)

    return pd.DataFrame({
        "id": ids,
        "class_0_prob": out0.numpy(),
        "class_1_prob": out1.numpy(),
        "symbol": df["symbol"].values,
        "published_at": df["published_at"].values,
        "date": timestamps.dt.normalize().values,
        "available_date": aligned_dates.values,
        "source": df["source"].values,
    })


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #
def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    requested = str(args.gpus).strip().lower()
    if requested == "all":
        import torch

        gpus = [str(i) for i in range(torch.cuda.device_count())] or ["cpu"]
    elif requested == "cpu":
        gpus = ["cpu"]
    else:
        gpus = [g.strip() for g in requested.split(",") if g.strip()]
        if not gpus or any(not g.isdigit() for g in gpus):
            raise ValueError("--gpus 只能是整数列表、all 或 cpu")
        if len(set(gpus)) != len(gpus):
            raise ValueError("--gpus 不能重复指定同一个设备")

    out_tag = args.month if args.limit is None else f"{args.month}_lim{args.limit}"
    final = os.path.join(args.out_dir, f"social_text_probs_{out_tag}.csv")

    # 单卡 & 非分片子进程: 本进程缓存 + 前向
    if args.part is None and len(gpus) == 1:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1" if gpus[0] == "cpu" else gpus[0]
        args.gpus = gpus[0]
        cache, meta = ensure_cache(args)
        res = run_inference(args, cache, meta, 0, meta["n"])
        save_csv(final, res)
        print("已保存:", final, "| 行数:", len(res), flush=True)
        print("NaN 校验:", int(res[["class_0_prob", "class_1_prob"]].isna().sum().sum()), flush=True)
        return 0

    # 需要分片: 先统一建缓存(双卡时由父进程建, 子进程复用)
    cache, meta = ensure_cache(args)
    N = meta["n"]

    # 子进程模式: 处理自己的 [lo,hi) 分片
    if args.part is not None:
        step = (N + args.num_parts - 1) // args.num_parts
        lo = args.part * step
        hi = min(N, lo + step)
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1" if gpus[0] == "cpu" else gpus[0]
        args.gpus = gpus[0]
        res = run_inference(args, cache, meta, lo, hi)
        shard = final.replace(".csv", f".part{args.part}.csv")
        save_csv(shard, res)
        print("分片已存:", shard, "| 行数:", len(res), flush=True)
        return 0

    # 双卡: 每卡一个子进程分片前向, 父进程合并
    procs, parts = [], []
    gpus = gpus[:N]
    for part, gpu in enumerate(gpus):
        shard = final.replace(".csv", f".part{part}.csv")
        parts.append(shard)
        cmd = [sys.executable, os.path.abspath(__file__)]
        for k, v in vars(args).items():
            if k in ("gpus", "part", "num_parts", "re_tokenize") or v is None:
                continue
            cmd += ["--" + k.replace("_", "-"), str(v)]
        cmd += ["--gpus", gpu, "--num-parts", str(len(gpus)), "--part", str(part)]
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = "-1" if gpu == "cpu" else gpu
        print("启动子进程(GPU %s)" % gpu, flush=True)
        procs.append(subprocess.Popen(cmd, env=env))

    rc = [p.wait() for p in procs]
    if any(rc):
        print("子进程失败 rc=", rc, file=sys.stderr, flush=True)
        return 1

    import pandas as pd
    merged = pd.concat([pd.read_csv(p) for p in parts], ignore_index=True)
    save_csv(final, merged)
    for p in parts:
        os.remove(p)
    print("已合并保存:", final, "| 行数:", len(merged), flush=True)
    print("抽样:", flush=True)
    print(merged.head(3).to_string(), flush=True)
    print("NaN 校验:", int(merged[["class_0_prob", "class_1_prob"]].isna().sum().sum()), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
