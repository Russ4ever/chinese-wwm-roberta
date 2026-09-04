#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""gubapost 特征提取 → stock-day mean-pooled 特征 (per-file 输出, 可断点续跑)。

一次前向同时产出:
  1. CLS [768] + class_1_prob  → gubapost_stockday_cls.parquet
  2. 12 层 dead 128 投影 → gubapost_stockday_attn.parquet

**per-file 模式**: 每个日文件处理完后立即存一个小 parquet (per_file/gubapost_YYYYMMDD.parquet),
已有则跳过 → 可随时 kill + 换卡 + 续跑。内存峰值 = 一个日文件的数据 (~2GB)。

用法:
  python scripts/extract_gubapost_cls.py --years 2023,2024,2025,2026 --gpus 0,1 --workers 8
  python scripts/extract_gubapost_cls.py --only 20230103 --gpus 1 --workers 1  # pilot
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

SOURCE_COLUMNS = ["baName", "title", "content", "available_date"]
SHSZ_PREFIXES = ("0", "3", "6")
N_LAYERS = 12
DEAD_DIM = 128


def parse_args():
    ap = argparse.ArgumentParser(description="gubapost 特征提取 (per-file, 可续跑)")
    ap.add_argument("--input-dir", default="/home/intern_fjq_2026/data/NLP/gubapost")
    ap.add_argument("--out-dir", default=os.path.join(ROOT, "artifacts", "gubapost_cls"))
    ap.add_argument("--gpus", default="0,1")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--batch-tokens", type=int, default=131072)
    ap.add_argument("--batch-rows", type=int, default=4096)
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--dtype", default="fp16", choices=["fp16", "fp32"])
    ap.add_argument("--token-workers", type=int, default=4)
    ap.add_argument("--only", default=None)
    ap.add_argument("--years", default=None, help="逗号分隔年份, 如 2023,2024,2025,2026")
    args = ap.parse_args()
    gpus = [g.strip() for g in args.gpus.split(",") if g.strip()]
    if not gpus or any(not g.isdigit() for g in gpus):
        ap.error("--gpus 只能是逗号分隔 GPU 序号")
    args.gpus = gpus
    if args.years:
        args.years = {y.strip() for y in args.years.split(",") if y.strip()}
    for name in ("workers", "token_workers", "batch_tokens", "batch_rows", "max_length"):
        if getattr(args, name) <= 0:
            ap.error(f"--{name.replace('_', '-')} 必须大于 0")
    return args


def discover_files(input_dir, years=None):
    files = []
    for year in sorted(os.listdir(input_dir)):
        if years and year not in years:
            continue
        ydir = os.path.join(input_dir, year)
        if not os.path.isdir(ydir):
            continue
        for name in sorted(os.listdir(ydir)):
            if re.fullmatch(r"gubapost\d{8}\.parquet", name):
                files.append(os.path.join(ydir, name))
    if not files:
        raise FileNotFoundError(f"{input_dir} 下没有找到文件")
    return files


def file_date(path):
    return re.search(r"gubapost(\d{8})\.parquet$", os.path.basename(path)).group(1)


# --------------------------------------------------------------------------- #
# tokenize
# --------------------------------------------------------------------------- #
_TOKENIZER = None
_MAX_LENGTH = None


def _tokenizer_init(base_dir, max_length):
    global _TOKENIZER, _MAX_LENGTH
    from transformers import BertTokenizerFast
    _TOKENIZER = BertTokenizerFast.from_pretrained(base_dir, local_files_only=True)
    _MAX_LENGTH = max_length


def _tokenize_chunk(texts):
    return _TOKENIZER(texts, truncation=True, padding=False, max_length=_MAX_LENGTH)["input_ids"]


def build_texts(df):
    t = df["title"].fillna("").astype(str)
    c = df["content"].fillna("").astype(str)
    t_arr, c_arr = t.str.strip().values, c.str.strip().values
    n = len(t_arr)
    out = np.empty(n, dtype=object)
    for i in range(n):
        ti, ci = t_arr[i], c_arr[i]
        if not ci:
            out[i] = ti
        elif not ti or ci.startswith(ti):
            out[i] = ci
        else:
            out[i] = ti + " " + ci
    return out


def make_batches(lengths, order, batch_tokens, batch_rows):
    start = 0; n = len(order)
    while start < n:
        end = start + 1
        longest = int(lengths[order[end - 1]])
        while end < n:
            cl = max(longest, int(lengths[order[end]]))
            cr = end - start + 1
            if cr > batch_rows or cr * cl > batch_tokens:
                break
            longest = cl; end += 1
        yield order[start:end]; start = end


def load_q_dead(root):
    sub_path = os.path.join(root, "artifacts", "checkpoint_activation_rank", "runs",
                            "gubapost_v1", "analysis", "subspaces.npz")
    sub = np.load(sub_path, allow_pickle=False)
    return {L: np.ascontiguousarray(
        sub[f"token_natural_filtered__attention_output_{L:02d}__eigenvectors"][:, -DEAD_DIM:].astype(np.float32))
        for L in range(1, N_LAYERS + 1)}


# --------------------------------------------------------------------------- #
# 前向 + per-file 输出
# --------------------------------------------------------------------------- #
def forward_file(in_path, out_path, *, model, hook_state, pad_id, batch_tokens,
                 batch_rows, token_workers, device, pool):
    import pandas as pd
    import torch

    t0 = time.time()
    df = pd.read_parquet(in_path, columns=SOURCE_COLUMNS)
    mask = df["baName"].astype(str).str[0].isin(SHSZ_PREFIXES)
    df = df[mask].reset_index(drop=True)
    n = len(df)
    if n == 0:
        return {"file": os.path.basename(in_path), "rows": 0, "total_s": 0.0}

    texts = build_texts(df)
    dates = df["available_date"].astype(str).values
    symbols = df["baName"].astype(str).values

    # tokenize
    chunk_size = max(1, n // (token_workers * 4) if n > token_workers * 4 else 1)
    futures = [pool.submit(_tokenize_chunk, texts[i:i+chunk_size].tolist())
               for i in range(0, n, chunk_size)]
    tok_lists = []
    for fut in futures:
        tok_lists.extend(fut.result())
    lengths = np.fromiter((len(x) for x in tok_lists), dtype=np.int32, count=n)

    # 前向
    order = np.argsort(lengths, kind="stable")
    cls_all = np.empty((n, 768), dtype=np.float32)
    prob_all = np.empty(n, dtype=np.float32)
    dead_all = np.empty((n, N_LAYERS, DEAD_DIM), dtype=np.float32)

    with torch.inference_mode():
        for idx in make_batches(lengths, order, batch_tokens, batch_rows):
            chunk = [tok_lists[i] for i in idx]
            maxlen = max(len(x) for x in chunk)
            ids = np.full((len(chunk), maxlen), pad_id, dtype=np.int64)
            for r, seq in enumerate(chunk):
                ids[r, :len(seq)] = seq
            b = torch.from_numpy(ids).to(device, non_blocking=True)
            attn_mask = (b != pad_id)
            out = model(input_ids=b, attention_mask=attn_mask)
            cls_all[idx] = out.pooled_feature.float().cpu().numpy()
            prob_all[idx] = out.probabilities[:, 1].float().cpu().numpy()
            for li, L in enumerate(range(1, N_LAYERS + 1)):
                dead_all[idx, li, :] = hook_state["batch_attn"][L]

    # per-file: group by (available_date, symbol) → sum
    import pyarrow as pa
    import pyarrow.parquet as pq

    post_df = pd.DataFrame({"available_date": dates, "symbol": symbols})
    groups = post_df.groupby(["available_date", "symbol"]).ngroup().values
    n_grp = int(groups.max()) + 1
    unique = post_df.drop_duplicates(["available_date", "symbol"]).reset_index(drop=True)

    sum_cls = np.zeros((n_grp, 768), dtype=np.float32)
    np.add.at(sum_cls, groups, cls_all)
    sum_prob = np.zeros(n_grp, dtype=np.float32)
    np.add.at(sum_prob, groups, prob_all)
    n_posts = np.bincount(groups, minlength=n_grp).astype(np.int32)
    sum_attn = np.zeros((n_grp, N_LAYERS, DEAD_DIM), dtype=np.float32)
    np.add.at(sum_attn, groups, dead_all)

    # 存 per-file parquet (float16 省磁盘)
    data = {
        "available_date": pa.array(unique["available_date"].values, type=pa.string()),
        "symbol": pa.array(unique["symbol"].values, type=pa.string()),
        "n_posts": pa.array(n_posts, type=pa.int32()),
        "sum_cls": pa.FixedSizeListArray.from_arrays(pa.array(sum_cls.ravel(), type=pa.float16()), 768),
        "sum_prob": pa.array(sum_prob, type=pa.float32()),
    }
    for li in range(N_LAYERS):
        data[f"sum_attn_{li+1:02d}"] = pa.FixedSizeListArray.from_arrays(
            pa.array(sum_attn[:, li, :].ravel(), type=pa.float16()), DEAD_DIM)
    table = pa.table(data)
    tmp = out_path + ".tmp"
    pq.write_table(table, tmp, compression="snappy")
    os.replace(tmp, out_path)

    return {"file": os.path.basename(in_path), "rows": int(n), "n_stocks": n_grp,
            "total_s": round(time.time() - t0, 2)}


def per_file_valid(out_path):
    if not os.path.exists(out_path):
        return False
    try:
        import pyarrow.parquet as pq
        return pq.ParquetFile(out_path).metadata.num_rows > 0
    except Exception:
        return False


# --------------------------------------------------------------------------- #
# worker
# --------------------------------------------------------------------------- #
def worker_main(args, shard_files, worker_id):
    import torch
    from multiprocessing import get_context

    from src.config import load_yaml_config
    from src.models.modeling import build_candidate
    from src.activation_rank_loss_recovery import resolve_projection_module

    log = open(os.path.join(args.out_dir, "logs", f"worker{worker_id}.log"), "a", buffering=1)
    def echo(msg):
        log.write(f"[{time.strftime('%m-%d %H:%M:%S')}] {msg}\n")

    cfg = load_yaml_config(os.path.join(ROOT, "configs", "model.yaml"))
    base = os.path.join(ROOT, cfg["paths"]["base_model_dir"])
    ckpt = os.path.join(ROOT, cfg["paths"]["checkpoint"])
    device = "cuda"
    dtype = torch.float16 if args.dtype == "fp16" else torch.float32

    echo(f"worker{worker_id}(GPU{os.environ.get('GUBAPOST_GPU','?')}) 启动: "
         f"{len(shard_files)} 文件, attn=sdpa, hooks={N_LAYERS}层 dead{DEAD_DIM}")

    model = build_candidate(base, ckpt, pooling="cls", pooling_confirmed=False,
                            device=device, dtype=dtype,
                            attn_implementation="sdpa").eval()

    # hooks (dead 128 投影在 hook 内)
    Q_dead = load_q_dead(ROOT)
    Q_dead_t = {L: torch.from_numpy(Q_dead[L]).to(device) for L in range(1, N_LAYERS + 1)}
    hook_state = {"batch_attn": {}}

    def make_hook(layer_idx):
        Q = Q_dead_t[layer_idx]
        def hook(_module, _input, output):
            tensor = output[0] if isinstance(output, (tuple, list)) else output
            hook_state["batch_attn"][layer_idx] = (tensor[:, 0, :].float() @ Q).cpu().numpy()
        return hook

    handles = []
    for L in range(1, N_LAYERS + 1):
        module = resolve_projection_module(model.bert, f"attention_output_{L:02d}")
        handles.append(module.register_forward_hook(make_hook(L)))

    from transformers import BertTokenizerFast
    pad_id = int(BertTokenizerFast.from_pretrained(base, local_files_only=True).pad_token_id)

    pool = ProcessPoolExecutor(
        max_workers=args.token_workers, mp_context=get_context("spawn"),
        initializer=_tokenizer_init, initargs=(base, args.max_length))

    per_file_dir = os.path.join(args.out_dir, "per_file")
    os.makedirs(per_file_dir, exist_ok=True)

    n_done = n_skip = n_fail = 0
    for k, in_path in enumerate(shard_files):
        d = file_date(in_path)
        out_path = os.path.join(per_file_dir, f"gubapost_{d}.parquet")
        if per_file_valid(out_path):
            n_skip += 1
            continue
        try:
            stat = forward_file(
                in_path, out_path, model=model, hook_state=hook_state, pad_id=pad_id,
                batch_tokens=args.batch_tokens, batch_rows=args.batch_rows,
                token_workers=args.token_workers, device=device, pool=pool)
            n_done += 1
            echo(f"({k+1}/{len(shard_files)}) {stat['file']} rows={stat['rows']} "
                 f"stocks={stat['n_stocks']} {stat['total_s']}s")
        except Exception:
            n_fail += 1
            echo(f"({k+1}/{len(shard_files)}) FAIL {in_path}\n{traceback.format_exc()[-1500:]}")

    pool.shutdown()
    for h in handles:
        h.remove()
    echo(f"worker{worker_id} 结束: done={n_done} skip={n_skip} fail={n_fail}")
    log.close()
    del model, Q_dead_t
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return 0 if n_fail == 0 else 3


# --------------------------------------------------------------------------- #
# 父进程
# --------------------------------------------------------------------------- #
def merge_per_file(out_dir):
    """合并 per_file/ 下所有 parquet → 两个最终 parquet。"""
    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq

    per_dir = os.path.join(out_dir, "per_file")
    files = sorted(glob.glob(os.path.join(per_dir, "gubapost_*.parquet")))
    if not files:
        print("[error] per_file/ 下没有文件", flush=True)
        return None, None

    print(f"合并 {len(files)} 个 per-file parquet...", flush=True)
    dfs = []
    for i, f in enumerate(files):
        dfs.append(pd.read_parquet(f))
        if (i + 1) % 200 == 0:
            print(f"  读取 {i+1}/{len(files)}", flush=True)
    df = pd.concat(dfs, ignore_index=True)
    del dfs
    print(f"  concat: {len(df):,} 行", flush=True)

    counts = df["n_posts"].values.astype(np.float64)
    groups = df.groupby(["available_date", "symbol"]).ngroup().values
    n_groups = int(groups.max()) + 1
    count_per = np.zeros(n_groups, dtype=np.float64)
    np.add.at(count_per, groups, counts)
    meta = df.groupby(["available_date", "symbol"], as_index=False)["n_posts"].sum()

    # CLS
    cls_data = {
        "available_date": pa.array(meta["available_date"].values, type=pa.string()),
        "symbol": pa.array(meta["symbol"].values, type=pa.string()),
        "n_posts": pa.array(meta["n_posts"].values.astype(np.int32), type=pa.int32()),
    }
    vals = np.stack(df["sum_cls"].tolist()).astype(np.float64)
    sum_p = np.zeros((n_groups, 768), dtype=np.float64)
    np.add.at(sum_p, groups, vals)
    cls_data["cls"] = pa.FixedSizeListArray.from_arrays(
        pa.array((sum_p / count_per[:, None]).astype(np.float16).ravel(), type=pa.float16()), 768)
    del vals, sum_p
    vals_p = df["sum_prob"].values.astype(np.float64)
    sum_p2 = np.zeros(n_groups, dtype=np.float64)
    np.add.at(sum_p2, groups, vals_p)
    cls_data["prob"] = pa.array((sum_p2 / count_per).astype(np.float32), type=pa.float32())
    del vals_p, sum_p2
    cls_path = os.path.join(out_dir, "gubapost_stockday_cls.parquet")
    pq.write_table(pa.table(cls_data), cls_path, compression="snappy")
    print(f"  CLS: {n_groups:,} stock-days, {os.path.getsize(cls_path)/1e6:.1f} MB", flush=True)
    del cls_data

    # Attn (逐层, 避免大数组)
    attn_data = {
        "available_date": pa.array(meta["available_date"].values, type=pa.string()),
        "symbol": pa.array(meta["symbol"].values, type=pa.string()),
        "n_posts": pa.array(meta["n_posts"].values.astype(np.int32), type=pa.int32()),
    }
    for li in range(N_LAYERS):
        col = f"sum_attn_{li+1:02d}"
        vals = np.stack(df[col].tolist()).astype(np.float64)
        sum_p = np.zeros((n_groups, DEAD_DIM), dtype=np.float64)
        np.add.at(sum_p, groups, vals)
        attn_data[f"attn_{li+1:02d}"] = pa.FixedSizeListArray.from_arrays(
            pa.array((sum_p / count_per[:, None]).astype(np.float16).ravel(), type=pa.float16()), DEAD_DIM)
        del vals, sum_p
    attn_path = os.path.join(out_dir, "gubapost_stockday_attn.parquet")
    pq.write_table(pa.table(attn_data), attn_path, compression="snappy")
    print(f"  Attn: {n_groups:,} stock-days, {os.path.getsize(attn_path)/1e6:.1f} MB", flush=True)

    return n_groups, len(files)


def main():
    args = parse_args()
    os.makedirs(os.path.join(args.out_dir, "logs"), exist_ok=True)
    os.makedirs(os.path.join(args.out_dir, "per_file"), exist_ok=True)

    files = discover_files(args.input_dir, args.years)
    if args.only:
        keep = {s.strip() for s in args.only.split(",") if s.strip()}
        files = [f for f in files if file_date(f) in keep]
    print(f"共 {len(files)} 个日文件 (gpus={args.gpus}, workers={args.workers})", flush=True)

    import subprocess

    n_w = args.workers
    # contiguous sharding
    chunk = (len(files) + n_w - 1) // n_w
    shards = [files[i*chunk:(i+1)*chunk] for i in range(n_w)]
    procs = []
    try:
        for w in range(n_w):
            if not shards[w]:
                continue
            shard_list = os.path.join(args.out_dir, "logs", f"shard{w}.txt")
            with open(shard_list, "w") as f:
                f.write("\n".join(shards[w]))
            cmd = [sys.executable, os.path.abspath(__file__)]
            cmd += ["--input-dir", args.input_dir, "--out-dir", args.out_dir,
                    "--gpus", args.gpus[0], "--workers", str(n_w),
                    "--batch-tokens", str(args.batch_tokens),
                    "--batch-rows", str(args.batch_rows),
                    "--max-length", str(args.max_length), "--dtype", args.dtype,
                    "--token-workers", str(args.token_workers)]
            if args.years:
                cmd += ["--years", ",".join(sorted(args.years))]
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = args.gpus[w % len(args.gpus)]
            env["GUBAPOST_SHARD_LIST"] = shard_list
            env["GUBAPOST_WORKER_ID"] = str(w)
            env["GUBAPOST_GPU"] = args.gpus[w % len(args.gpus)]
            logf = open(os.path.join(args.out_dir, "logs", f"worker{w}.log"), "a")
            procs.append((subprocess.Popen(cmd, env=env, stdout=logf, stderr=subprocess.STDOUT),
                          len(shards[w]), args.gpus[w % len(args.gpus)]))
            print(f"启动 worker{w}: GPU{args.gpus[w % len(args.gpus)]}, {len(shards[w])} 文件", flush=True)

        try:
            from tqdm import tqdm
            bar = tqdm(total=len(files), desc="特征提取", unit="file", mininterval=5, file=sys.stderr)
            done = set()
            while any(p.poll() is None for p, _, _ in procs):
                time.sleep(5)
                cur = set()
                for w in range(n_w):
                    lf = os.path.join(args.out_dir, "logs", f"worker{w}.log")
                    if os.path.exists(lf):
                        for line in open(lf):
                            m = re.search(r"gubapost\d{8}\.parquet", line)
                            if m and "rows=" in line:
                                cur.add(m.group())
                if len(cur) > len(done):
                    bar.update(len(cur) - len(done))
                    done = cur
            bar.close()
        except ImportError:
            while any(p.poll() is None for p, _, _ in procs):
                time.sleep(60)

        rc = [p.wait() for p, _, _ in procs]
    except BaseException:
        print("[abort] 终止 worker ...", flush=True)
        for p, _, _ in procs:
            p.terminate()
        for p, _, _ in procs:
            try:
                p.wait(10)
            except subprocess.TimeoutExpired:
                p.kill()
        raise

    for (p, n, g), c in zip(procs, rc):
        print(f"GPU{g} worker 退出码 {c} (分配 {n})", flush=True)

    # 合并 per-file → 最终 parquet
    n_stockdays, n_files = merge_per_file(args.out_dir)

    manifest = {
        "schema": "gubapost_stockday_features_v1",
        "n_stock_days": int(n_stockdays or 0),
        "n_per_files": int(n_files or 0),
        "products": ["gubapost_stockday_cls.parquet", "gubapost_stockday_attn.parquet"],
        "attn_dim": DEAD_DIM,
        "max_length": args.max_length, "dtype": args.dtype, "attn": "sdpa",
        "hooks": f"attention.output.dense L1-{N_LAYERS} (dead {DEAD_DIM} projection, per-file)",
        "gpus": args.gpus, "workers": args.workers,
        "years": sorted(args.years) if args.years else "all",
        "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(os.path.join(args.out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(f"\nmanifest: {json.dumps(manifest, ensure_ascii=False)}", flush=True)
    return 0


if __name__ == "__main__":
    if os.environ.get("GUBAPOST_SHARD_LIST"):
        _args = parse_args()
        with open(os.environ["GUBAPOST_SHARD_LIST"]) as _f:
            _shard = [line.strip() for line in _f if line.strip()]
        sys.exit(worker_main(_args, _shard, int(os.environ.get("GUBAPOST_WORKER_ID", "0"))))
    sys.exit(main())
