#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""研报 JSONL 二分类批量推理（预缓存版）。

复用 social_text 推理脚本"多进程预 tokenize + mmap 纯前向"的架构，面向
research_report 的 forecast_stk JSONL（TITLE 在前、CONTENT 在后拼接为输入，
正文为空时退化为仅 TITLE）。

用法:
  conda activate nlp_fjq
  cd /home/intern_fjq_2026/Projects/chinese-wwm-roberta
  python scripts/infer_research_report.py --gpus 1 --max-length 512                # 全量
  python scripts/infer_research_report.py --gpus 1 --max-length 512 --limit 2000    # 试水

可选:
  --batch-size 1024  --max-length 512  --pooling cls|pooler|masked_mean
  --dtype fp16|fp32  --token-workers N  --re-tokenize(强制重建缓存)

产物:
  缓存: artifacts/token_cache/research_report_<tag>_len<maxlen>_ids.npy (+ .meta.json)
  结果: artifacts/research_report_probs_<tag>.csv
  列: id, class_0_prob, class_1_prob, symbol, stock_name, title,
      create_date, date, organ_name, report_type, reliability, source
"""
from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

DEFAULT_INPUT = "/home/intern_fjq_2026/data/NLP/research_report/forecast_stk_20240101_20241231.jsonl"

# 研报需要保留到结果里的列（CONTENT 不落盘，避免 CSV 过大）
OUT_COLUMNS = ["id", "symbol", "stock_name", "title", "create_date", "date",
               "organ_name", "report_type", "reliability", "source"]


def parse_args():
    ap = argparse.ArgumentParser(description="研报 JSONL 二分类批量推理(预缓存版)")
    ap.add_argument("--input", default=DEFAULT_INPUT, help="研报 JSONL 路径")
    ap.add_argument("--out-dir", default=os.path.join(ROOT, "artifacts"))
    ap.add_argument("--gpus", default="1", help="只支持单个 GPU，如 1（映射为 CUDA_VISIBLE_DEVICES）")
    ap.add_argument("--batch-size", type=int, default=1024)
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--pooling", default="cls", choices=["cls", "pooler", "masked_mean"])
    ap.add_argument("--dtype", default="fp16", choices=["fp16", "fp32"])
    ap.add_argument("--limit", type=int, default=None, help="只处理前 N 行(None=全部)")
    ap.add_argument("--token-workers", type=int, default=None, help="tokenize 进程数(默认 min(32, 核数))")
    ap.add_argument("--re-tokenize", action="store_true", help="忽略已有缓存强制重建")
    return ap.parse_args()


def _tag(args):
    base = os.path.splitext(os.path.basename(args.input))[0]
    return base if args.limit is None else f"{base}_lim{args.limit}"


# --------------------------------------------------------------------------- #
# 读取 JSONL -> (texts, meta_df)
# --------------------------------------------------------------------------- #
def read_jsonl(input_path, limit=None):
    """流式读研报 JSONL，构造 texts 与轻量元数据表。

    text = TITLE + "。" + CONTENT（CONTENT 为空时仅 TITLE）。
    """
    import pandas as pd

    texts = []
    data = {c: [] for c in OUT_COLUMNS}
    bad = 0
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except Exception:  # noqa: BLE001
                bad += 1
                continue
            if limit is not None and len(texts) >= limit:
                break

            title = d.get("TITLE") or ""
            content = d.get("CONTENT") or ""
            text = f"{title}。{content}" if content else title

            create = d.get("CREATE_DATE") or ""
            create = str(create)

            texts.append(text)
            data["id"].append(str(d.get("ID")))
            data["symbol"].append(d.get("STOCK_CODE"))
            data["stock_name"].append(d.get("STOCK_NAME"))
            data["title"].append(title)
            data["create_date"].append(create)
            data["date"].append(create[:10])
            data["organ_name"].append(d.get("ORGAN_NAME"))
            data["report_type"].append(d.get("REPORT_TYPE"))
            data["reliability"].append(d.get("RELIABILITY"))
            data["source"].append("research_report")

    if bad:
        print(f"[warn] 跳过无法解析的行: {bad}", flush=True)
    print(f"读取完成: {len(texts)} 行", flush=True)
    return texts, pd.DataFrame(data)


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
    """返回 (cache_npy, meta)。缓存失效则多进程 tokenize 重建。"""
    cache_dir = os.path.join(args.out_dir, "token_cache")
    os.makedirs(cache_dir, exist_ok=True)
    tag = _tag(args)
    cache_npy = os.path.join(cache_dir, f"research_report_{tag}_len{args.max_length}_ids.npy")
    cache_meta = cache_npy + ".meta.json"
    meta_csv = os.path.join(cache_dir, f"research_report_{tag}_meta.csv.gz")

    st = os.stat(args.input)
    src_sig = (int(st.st_mtime), int(st.st_size))

    if (os.path.exists(cache_npy) and os.path.exists(cache_meta) and os.path.exists(meta_csv)
            and not args.re_tokenize):
        meta = json.load(open(cache_meta, encoding="utf-8"))
        if meta.get("src") == list(src_sig) and meta.get("max_length") == args.max_length:
            print(f"缓存复用: {cache_npy} (n={meta['n']})", flush=True)
            return cache_npy, meta, meta_csv
        print("缓存失效, 重建", flush=True)

    from src.config import load_yaml_config
    cfg = load_yaml_config(os.path.join(ROOT, "configs", "model.yaml"))
    base = os.path.join(ROOT, cfg["paths"]["base_model_dir"])

    from transformers import BertTokenizerFast
    tok = BertTokenizerFast.from_pretrained(base, local_files_only=True)
    pad = int(tok.pad_token_id)
    del tok

    texts, meta_df = read_jsonl(args.input, args.limit)
    n = len(texts)

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
    meta_df.to_csv(meta_csv, index=False)
    print(f"预 tokenize 完成: {cache_npy} 形状 {ids.shape}, "
          f"用时 {time.time()-t0:.1f}s ({n/max(time.time()-t0,1e-6):.0f} 条/s)", flush=True)
    return cache_npy, meta, meta_csv


# --------------------------------------------------------------------------- #
# 纯前向(读缓存, 不经 tokenizer)
# --------------------------------------------------------------------------- #
def run_inference(args, cache_npy, meta, meta_csv):
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
    print(f"device={device} ({gpu_name}) visible_devices={os.environ.get('CUDA_VISIBLE_DEVICES')} "
          f"dtype={args.dtype} pooling={args.pooling} batch={args.batch_size}", flush=True)

    model = build_candidate(base, ckpt, pooling=args.pooling, pooling_confirmed=False,
                            device=device, dtype=dtype).eval()
    pad = int(meta["pad_token_id"])

    df = pd.read_csv(meta_csv, dtype={"id": str, "symbol": str})
    if args.limit is not None:
        df = df.iloc[: args.limit]
    k = len(df)
    if k != meta["n"]:
        print(f"[warn] 元数据 {k} 行 != 缓存 {meta['n']} 行", flush=True)
    ids = df["id"].astype(str).tolist()

    mm = np.load(cache_npy, mmap_mode="r")            # [N,max_length] int32
    out0 = torch.empty(k, dtype=torch.float32)
    out1 = torch.empty(k, dtype=torch.float32)

    SB = 65536                                          # 超级批: 每次从 mmap 取这么多行
    t0 = time.time()
    with torch.inference_mode():
        for s in range(0, k, SB):
            e = min(k, s + SB)
            ids_t = torch.from_numpy(mm[s: e].astype("int64"))
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
    print(f"完成 {k} 行, 前向耗时 {dt:.1f}s, 吞吐 {k/dt:.0f} 条/s", flush=True)

    res = df.copy()
    res.insert(0, "class_0_prob", out0.numpy())
    res.insert(1, "class_1_prob", out1.numpy())
    return res


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #
def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    gpus = [g.strip() for g in str(args.gpus).split(",") if g.strip()]
    if len(gpus) != 1:
        print("本脚本仅支持单卡(只用一个 GPU)。请用 --gpus 1", file=sys.stderr, flush=True)
        return 2
    os.environ["CUDA_VISIBLE_DEVICES"] = gpus[0]

    tag = _tag(args)
    final = os.path.join(args.out_dir, f"research_report_probs_{tag}.csv")

    cache_npy, meta, meta_csv = ensure_cache(args)
    res = run_inference(args, cache_npy, meta, meta_csv)
    res.to_csv(final, index=False)
    print("已保存:", final, "| 行数:", len(res), flush=True)
    print("NaN 校验:", int(res[["class_0_prob", "class_1_prob"]].isna().sum().sum()), flush=True)
    print("抽样:", flush=True)
    print(res.head(3).to_string(), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())