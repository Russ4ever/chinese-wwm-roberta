#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""social_text 批量推理（预缓存版）——取代旧版(边 tokenize 边前向)。

架构:
  1) 预缓存: 一次性多进程(默认 ≤32 核)把整月文本 tokenize 成 input_ids(int32) 存 .npy
  2) 纯前向: 推理阶段只从缓存取 tensor 前向, GPU 不再等 CPU -> 吞吐逼近显卡上限
  缓存按 (month, max_length) 复用, 换 pooling / dtype / 单卡双卡 都不用重新 tokenize。

用法:
  conda activate nlp_fjq
  cd /home/intern_fjq_2026/Projects/chinese-wwm-roberta
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
import json
import multiprocessing
import os
import subprocess
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def parse_args():
    ap = argparse.ArgumentParser(description="social_text 批量推理(预缓存版)")
    ap.add_argument("--month", required=True, help="月份, 如 202403")
    ap.add_argument("--gpus", default="all", help="0 | 1 | all(双卡分片)")
    ap.add_argument("--batch-size", type=int, default=1024)
    ap.add_argument("--max-length", type=int, default=128)
    ap.add_argument("--pooling", default="cls", choices=["cls", "pooler", "masked_mean"])
    ap.add_argument("--dtype", default="fp16", choices=["fp16", "fp32"])
    ap.add_argument("--limit", type=int, default=None, help="只处理前 N 行(None=全部)")
    ap.add_argument("--token-workers", type=int, default=None, help="tokenize 进程数(默认 min(32, 核数))")
    ap.add_argument("--re-tokenize", action="store_true", help="忽略已有缓存强制重建")
    ap.add_argument("--data-dir", default=os.path.join(ROOT, "filtered_data"))
    ap.add_argument("--out-dir", default=os.path.join(ROOT, "artifacts"))
    ap.add_argument("--part", type=int, default=None, help="[内部] 分片序号")
    ap.add_argument("--num-parts", type=int, default=1, help="[内部] 分片数")
    return ap.parse_args()


# --------------------------------------------------------------------------- #
# 预缓存
# --------------------------------------------------------------------------- #
def _tok_init(base_dir, max_length):
    global _TOK, _MAXL
    from transformers import BertTokenizerFast
    _TOK = BertTokenizerFast.from_pretrained(base_dir, local_files_only=True)
    _MAXL = max_length


def _tok_worker(texts):
    enc = _TOK(texts, truncation=True, padding="max_length", max_length=_MAXL,
               return_tensors="np", return_attention_mask=False, return_token_type_ids=False)
    return enc["input_ids"].astype("int32")


def ensure_cache(args):
    """返回 (cache_npy, meta)。缓存失效(源文件变动或强制)则多进程 tokenize 重建。"""
    cache_dir = os.path.join(args.out_dir, "token_cache")
    os.makedirs(cache_dir, exist_ok=True)
    tag = args.month if args.limit is None else f"{args.month}_lim{args.limit}"
    cache_npy = os.path.join(cache_dir, f"social_text_{tag}_len{args.max_length}_ids.npy")
    cache_meta = cache_npy + ".meta.json"
    data_path = os.path.join(args.data_dir, f"social_text_{args.month}.csv.gz")

    st = os.stat(data_path)
    src_sig = (int(st.st_mtime), int(st.st_size))

    if os.path.exists(cache_npy) and os.path.exists(cache_meta) and not args.re_tokenize:
        meta = json.load(open(cache_meta, encoding="utf-8"))
        if meta.get("src") == list(src_sig) and meta.get("max_length") == args.max_length:
            print(f"缓存复用: {cache_npy} (n={meta['n']})", flush=True)
            return cache_npy, meta
        print("缓存失效, 重建", flush=True)

    from src.config import load_yaml_config
    cfg = load_yaml_config(os.path.join(ROOT, "configs", "model.yaml"))
    base = os.path.join(ROOT, cfg["paths"]["base_model_dir"])

    from transformers import BertTokenizerFast
    tok = BertTokenizerFast.from_pretrained(base, local_files_only=True)
    pad = int(tok.pad_token_id)
    del tok

    import pandas as pd
    df = pd.read_csv(data_path)
    if args.limit is not None:
        df = df.iloc[: args.limit]
    n = len(df)
    texts = df["text"].fillna("").tolist()

    nw = args.token_workers or min(32, (os.cpu_count() or 1))
    slots = max(1, nw * 4)
    step = max(1, (n + slots - 1) // slots)
    chunks = [texts[i:i + step] for i in range(0, n, step)]
    print(f"预 tokenize: {n} 条, max_length={args.max_length}, workers={nw}, "
          f"{len(chunks)} 个分片", flush=True)
    t0 = time.time()
    ctx = multiprocessing.get_context("fork")
    with ctx.Pool(nw, initializer=_tok_init, initargs=(base, args.max_length)) as pool:
        results = pool.map(_tok_worker, chunks)
    ids = np.concatenate(results, axis=0)
    np.save(cache_npy, ids)
    meta = {"n": int(n), "max_length": args.max_length,
            "pad_token_id": pad, "src": list(src_sig)}
    json.dump(meta, open(cache_meta, "w", encoding="utf-8"))
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
    dtype = torch.float16 if args.dtype == "fp16" else torch.float32
    device = "cuda" if torch.cuda.is_available() else "cpu"
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
    ids = df["id"].astype(str).tolist()

    mm = np.load(cache_npy, mmap_mode="r")            # [N,128] int32, mmap 懒加载
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
        "date": df["date"].values,
        "available_date": df["available_date"].values,
        "source": df["source"].values,
    })


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #
def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    gpus = [g.strip() for g in str(args.gpus).split(",") if g.strip()]
    if args.gpus == "all":
        gpus = ["0", "1"]

    out_tag = args.month if args.limit is None else f"{args.month}_lim{args.limit}"
    final = os.path.join(args.out_dir, f"social_text_probs_{out_tag}.csv")

    # 单卡 & 非分片子进程: 本进程缓存 + 前向
    if args.part is None and len(gpus) == 1:
        cache, meta = ensure_cache(args)
        os.environ["CUDA_VISIBLE_DEVICES"] = gpus[0]
        res = run_inference(args, cache, meta, 0, meta["n"])
        res.to_csv(final, index=False)
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
        os.environ["CUDA_VISIBLE_DEVICES"] = gpus[0]
        res = run_inference(args, cache, meta, lo, hi)
        shard = final.replace(".csv", f".part{args.part}.csv")
        res.to_csv(shard, index=False)
        print("分片已存:", shard, "| 行数:", len(res), flush=True)
        return 0

    # 双卡: 每卡一个子进程分片前向, 父进程合并
    procs, parts = [], []
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
        env["CUDA_VISIBLE_DEVICES"] = gpu
        print("启动子进程(GPU %s)" % gpu, flush=True)
        procs.append(subprocess.Popen(cmd, env=env))

    rc = [p.wait() for p in procs]
    if any(rc):
        print("子进程失败 rc=", rc, file=sys.stderr, flush=True)
        return 1

    import pandas as pd
    merged = pd.concat([pd.read_csv(p) for p in parts], ignore_index=True)
    merged.to_csv(final, index=False)
    for p in parts:
        os.remove(p)
    print("已合并保存:", final, "| 行数:", len(merged), flush=True)
    print("抽样:", flush=True)
    print(merged.head(3).to_string(), flush=True)
    print("NaN 校验:", int(merged[["class_0_prob", "class_1_prob"]].isna().sum().sum()), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())