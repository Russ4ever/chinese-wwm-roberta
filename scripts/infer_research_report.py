#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""研报 JSONL 二分类批量推理（预缓存版）。

复用 social_text 推理脚本"多进程预 tokenize + mmap 纯前向"的架构，面向
research_report 的 forecast_stk JSONL（TITLE 在前、CONTENT 在后拼接为输入，
正文为空时退化为仅 TITLE）。

用法:
  python scripts/infer_research_report.py --gpus 0 --max-length 512                # 全量
  python scripts/infer_research_report.py --gpus cpu --max-length 512 --limit 2000  # 试水

可选:
  --batch-size 1024  --max-length 512  --pooling cls|pooler|masked_mean
  --dtype fp16|fp32  --token-workers N  --re-tokenize(强制重建缓存)

产物:
  缓存: artifacts/token_cache/research_report_<tag>_len<maxlen>_ids.npy (+ .meta.json)
  结果: artifacts/research_report_probs_<tag>.csv
  列: id, class_0_prob, class_1_prob, symbol, stock_name, title,
      create_date, date, available_date, organ_name, report_type, reliability, source
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.trading_calendar import (  # noqa: E402
    align_to_trading_day,
    load_trading_dates,
    parse_market_timestamps,
)
from src.inference_cache import save_csv  # noqa: E402

DEFAULT_INPUT = os.environ.get(
    "RESEARCH_REPORT_INPUT",
    os.path.join(ROOT, "data", "research_report", "forecast_stk_20240101_20241231.jsonl"),
)
DEFAULT_TRADING_CALENDAR = os.environ.get(
    "RESEARCH_REPORT_TRADING_CALENDAR",
    os.path.join(ROOT, "data", "RTN_daily", "rtn_1d.parquet"),
)

# 研报需要保留到结果里的列（CONTENT 不落盘，避免 CSV 过大）
OUT_COLUMNS = ["id", "symbol", "stock_name", "title", "create_date", "date",
               "available_date", "organ_name", "report_type", "reliability", "source"]


def parse_args():
    ap = argparse.ArgumentParser(description="研报 JSONL 二分类批量推理(预缓存版)")
    ap.add_argument("--input", default=DEFAULT_INPUT, help="研报 JSONL 路径")
    ap.add_argument(
        "--trading-calendar",
        default=DEFAULT_TRADING_CALENDAR,
        help="项目已有交易日文件（CSV/Parquet，含 date 列）",
    )
    ap.add_argument("--out-dir", default=os.path.join(ROOT, "artifacts"))
    ap.add_argument("--gpus", default="0", help="单个 GPU 编号，或 cpu")
    ap.add_argument("--batch-size", type=int, default=1024)
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--pooling", default="cls", choices=["cls", "pooler", "masked_mean"])
    ap.add_argument("--dtype", default="fp16", choices=["fp16", "fp32"])
    ap.add_argument("--limit", type=int, default=None, help="只处理前 N 行(None=全部)")
    ap.add_argument("--token-workers", type=int, default=None, help="tokenize 进程数(默认 min(32, 核数))")
    ap.add_argument("--re-tokenize", action="store_true", help="忽略已有缓存强制重建")
    args = ap.parse_args()
    for name in ("batch_size", "max_length"):
        if getattr(args, name) <= 0:
            ap.error(f"--{name.replace('_', '-')} 必须大于 0")
    if args.limit is not None and args.limit <= 0:
        ap.error("--limit 必须大于 0")
    if args.token_workers is not None and args.token_workers <= 0:
        ap.error("--token-workers 必须大于 0")
    return args


def _tag(args):
    base = os.path.splitext(os.path.basename(args.input))[0]
    return base if args.limit is None else f"{base}_lim{args.limit}"


# --------------------------------------------------------------------------- #
# 读取 JSONL -> (texts, meta_df)
# --------------------------------------------------------------------------- #
def read_jsonl(input_path, trading_dates, limit=None):
    """流式读研报 JSONL，构造 texts 与轻量元数据表。

    text = TITLE + "。" + CONTENT（CONTENT 为空时仅 TITLE）。
    """
    import pandas as pd

    texts = []
    data = {c: [] for c in OUT_COLUMNS}
    bad = 0
    with open(input_path, "r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                bad += 1
                continue
            if not isinstance(d, dict):
                bad += 1
                continue
            if limit is not None and len(texts) >= limit:
                break

            title = str(d.get("TITLE") or "")
            content = str(d.get("CONTENT") or "")
            text = f"{title}。{content}" if content else title

            create = d.get("CREATE_DATE") or ""
            create = str(create)

            texts.append(text)
            source_id = d.get("ID")
            source_id = str(source_id).strip() if source_id is not None else ""
            data["id"].append(source_id or f"line:{line_number}")
            data["symbol"].append(d.get("STOCK_CODE"))
            data["stock_name"].append(d.get("STOCK_NAME"))
            data["title"].append(title)
            data["create_date"].append(create)
            data["date"].append(None)
            data["available_date"].append(None)
            data["organ_name"].append(d.get("ORGAN_NAME"))
            data["report_type"].append(d.get("REPORT_TYPE"))
            data["reliability"].append(d.get("RELIABILITY"))
            data["source"].append("research_report")

    if bad:
        print(f"[warn] 跳过无法解析的行: {bad}", flush=True)
    frame = pd.DataFrame(data)
    timestamps = parse_market_timestamps(frame["create_date"])
    frame["date"] = timestamps.dt.normalize()
    frame["available_date"] = align_to_trading_day(timestamps, trading_dates)
    print(f"读取完成: {len(texts)} 行", flush=True)
    return texts, frame


# --------------------------------------------------------------------------- #
# 预缓存
# --------------------------------------------------------------------------- #
def ensure_cache(args):
    """返回 (cache_npy, meta)。缓存失效则多进程 tokenize 重建。"""
    cache_dir = os.path.join(args.out_dir, "token_cache")
    os.makedirs(cache_dir, exist_ok=True)
    tag = _tag(args)
    cache_npy = os.path.join(cache_dir, f"research_report_{tag}_len{args.max_length}_ids.npy")
    cache_meta = cache_npy + ".meta.json"
    meta_csv = os.path.join(cache_dir, f"research_report_{tag}_meta.csv.gz")

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
    # 交易日历参与元数据输出，变动后必须同步重建 metadata 缓存。
    expected = {
        "source": file_signature(args.input),
        "trading_calendar_source": file_signature(args.trading_calendar),
        "max_length": args.max_length,
        "tokenizer_sha256": tokenizer_signature(base),
    }

    if (os.path.exists(cache_npy) and os.path.exists(cache_meta) and os.path.exists(meta_csv)
            and not args.re_tokenize):
        try:
            meta = load_json_object(cache_meta)
            if all(meta.get(k) == v for k, v in expected.items()):
                validate_cached_ids(
                    cache_npy,
                    n_rows=int(meta["n"]),
                    max_length=args.max_length,
                )
                import pandas as pd

                meta_rows = len(pd.read_csv(meta_csv, usecols=["id"]))
                if meta_rows != int(meta["n"]):
                    raise ValueError("元数据 CSV 行数与 token 缓存不一致")
                print(f"缓存复用: {cache_npy} (n={meta['n']})", flush=True)
                return cache_npy, meta, meta_csv
        except (OSError, TypeError, ValueError, KeyError):
            pass
        print("缓存失效, 重建", flush=True)

    trading_dates = load_trading_dates(args.trading_calendar)
    texts, meta_df = read_jsonl(args.input, trading_dates, args.limit)
    n = len(texts)
    if n == 0:
        raise ValueError(f"输入 JSONL 没有可处理的记录: {args.input}")

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
    meta_tmp = meta_csv + ".tmp"
    meta_df.to_csv(meta_tmp, index=False, compression="gzip")
    os.replace(meta_tmp, meta_csv)
    save_json_object(cache_meta, meta)
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
    wants_cpu = str(args.gpus).lower() == "cpu"
    if not wants_cpu and not torch.cuda.is_available():
        raise RuntimeError(f"请求 GPU {args.gpus}，但当前进程没有可用 CUDA 设备")
    device = "cpu" if wants_cpu else "cuda"
    if device == "cpu" and args.dtype == "fp16":
        print("[warn] CPU 推理不使用 fp16，已回退到 fp32", flush=True)
    dtype = torch.float16 if args.dtype == "fp16" and device == "cuda" else torch.float32
    gpu_name = torch.cuda.get_device_name(0) if device == "cuda" else "CPU"
    print(f"device={device} ({gpu_name}) visible_devices={os.environ.get('CUDA_VISIBLE_DEVICES')} "
          f"dtype={args.dtype} pooling={args.pooling} batch={args.batch_size}", flush=True)

    model = build_candidate(base, ckpt, pooling=args.pooling, pooling_confirmed=False,
                            device=device, dtype=dtype).eval()
    pad = int(meta["pad_token_id"])

    df = pd.read_csv(meta_csv, dtype={"id": str, "symbol": str})
    k = len(df)
    if k != meta["n"]:
        raise ValueError(f"元数据 {k} 行 != 缓存 {meta['n']} 行")

    mm = np.load(cache_npy, mmap_mode="r")            # [N,max_length] int32
    if mm.shape != (int(meta["n"]), int(meta["max_length"])):
        raise ValueError(f"缓存 shape 与元数据不一致: {mm.shape}")
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
    res.insert(1, "class_0_prob", out0.numpy())
    res.insert(2, "class_1_prob", out1.numpy())
    return res


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #
def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    requested = str(args.gpus).strip().lower()
    gpus = [g.strip() for g in requested.split(",") if g.strip()]
    if len(gpus) != 1 or (gpus[0] != "cpu" and not gpus[0].isdigit()):
        print("本脚本仅支持单个 GPU 编号或 cpu", file=sys.stderr, flush=True)
        return 2
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1" if gpus[0] == "cpu" else gpus[0]
    args.gpus = gpus[0]

    tag = _tag(args)
    final = os.path.join(args.out_dir, f"research_report_probs_{tag}.csv")

    cache_npy, meta, meta_csv = ensure_cache(args)
    res = run_inference(args, cache_npy, meta, meta_csv)
    save_csv(final, res)
    print("已保存:", final, "| 行数:", len(res), flush=True)
    print("NaN 校验:", int(res[["class_0_prob", "class_1_prob"]].isna().sum().sum()), flush=True)
    print("抽样:", flush=True)
    print(res.head(3).to_string(), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
