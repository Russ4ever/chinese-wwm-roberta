#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""gubapost(股吧帖子) 全量批量推理 —— baseline 语料过原始二分类模型。

数据: /home/intern_fjq_2026/data/NLP/gubapost/<year>/gubapostYYYYMMDD.parquet
      列: baName(股票代码), postId, title, content, publishTime, date, available_date
      available_date 已由数据方对齐到下一交易日, 本脚本直接沿用, 不再做交易日对齐。

架构(约 3.35 亿帖 / 50G; --gpus 1 单卡或 0,1 双卡开关):
  - 不建整库 token 缓存(335M x 128 int32 ≈ 171GB, 磁盘放不下):
    每个 worker 进程负责一组日文件, 进程内 spawn 池在线 tokenize(变长, 截断 max_length),
    主线程按 token 预算组 batch(长度分桶, 短帖不 pad 到 128)喂 GPU。
  - 多个 worker 进程共享同一块 GPU, 互相填补 tokenize/IO 空隙。
  - 产物按日分片 parquet, 已存在且行数一致则跳过 → 可断点续跑。
  - stock-day 聚合(mean)由 notebook 流式完成, 本脚本只负责推理。

用法:
  # 全量(默认仅 GPU 1; 4 个 worker 共享)
  python scripts/infer_gubapost.py \
      --input-dir /home/intern_fjq_2026/data/NLP/gubapost \
      --out-dir artifacts/gubapost_baseline --years 2020,2021,2022,2023
  # 双卡(0,1 各分一半 worker, 约快一倍)
  python scripts/infer_gubapost.py ... --gpus 0,1 --workers 8
  # 试水: 指定日期
  python scripts/infer_gubapost.py ... --only 20240426,20200614
  # 试水: 只跑前 N 个文件
  python scripts/infer_gubapost.py ... --max-files 5

产物(相对 --out-dir):
  probs_daily/gubapost_probs_YYYYMMDD.parquet   逐帖概率
      列: post_id, symbol, date, available_date, class_0_prob, class_1_prob
  completed.jsonl / failed.jsonl                逐文件完成/失败记录(断点续跑依据)
  manifest.json                                 全量完成后由父进程写出
"""
from __future__ import annotations

import argparse
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

OUT_COLUMNS = ["post_id", "symbol", "date", "available_date", "class_0_prob", "class_1_prob"]
SOURCE_COLUMNS = ["baName", "postId", "title", "content", "date", "available_date"]


def parse_args():
    ap = argparse.ArgumentParser(description="gubapost 全量批量推理(逐日分片, 可续跑)")
    ap.add_argument("--input-dir", default="/home/intern_fjq_2026/data/NLP/gubapost")
    ap.add_argument("--out-dir", default=os.path.join(ROOT, "artifacts", "gubapost_baseline"))
    ap.add_argument("--gpus", default="1",
                    help="逗号分隔的物理 GPU 序号: 1(默认, 仅 cuda:1) | 0,1(双卡)")
    ap.add_argument("--workers", type=int, default=4,
                    help="worker 进程总数(在 --gpus 间轮转均分)")
    ap.add_argument("--years", default=None,
                    help="逗号分隔年份过滤, 如 2020,2021,2022,2023(None=全部)")
    ap.add_argument("--batch-tokens", type=int, default=131072, help="单 batch padding 后 token 上限")
    ap.add_argument("--batch-rows", type=int, default=8192, help="单 batch 行数上限")
    ap.add_argument("--max-length", type=int, default=128)
    ap.add_argument("--dtype", default="fp16", choices=["fp16", "fp32"])
    ap.add_argument("--token-workers", type=int, default=4, help="每个 worker 内部 tokenize 进程数")
    ap.add_argument("--only", default=None, help="逗号分隔 YYYYMMDD, 只跑这些日期(试水)")
    ap.add_argument("--max-files", type=int, default=None, help="只跑前 N 个文件(试水)")
    args = ap.parse_args()
    for name in ("batch_tokens", "batch_rows", "max_length", "workers", "token_workers"):
        if getattr(args, name) <= 0:
            ap.error(f"--{name.replace('_', '-')} 必须大于 0")
    gpus = [g.strip() for g in args.gpus.split(",") if g.strip()]
    if not gpus or any(not g.isdigit() for g in gpus):
        ap.error("--gpus 只能是逗号分隔的 GPU 序号, 如 1 或 0,1")
    if len(set(gpus)) != len(gpus):
        ap.error("--gpus 不能重复指定同一个设备")
    args.gpus = gpus
    if args.years:
        args.years = {y.strip() for y in args.years.split(",") if y.strip()}
        if any(not re.fullmatch(r"\d{4}", y) for y in args.years):
            ap.error("--years 必须是 4 位年份, 如 2020,2021,2022,2023")
    return args


def discover_files(input_dir: str) -> list[str]:
    files, skipped = [], []
    for year in sorted(os.listdir(input_dir)):
        ydir = os.path.join(input_dir, year)
        if not os.path.isdir(ydir):
            continue
        for name in sorted(os.listdir(ydir)):
            if not name.endswith(".parquet"):
                continue
            path = os.path.join(ydir, name)
            if re.fullmatch(r"gubapost\d{8}\.parquet", name):
                files.append(path)
            else:
                skipped.append(path)
    if not files:
        raise FileNotFoundError(f"{input_dir} 下没有找到 parquet 日文件")
    if skipped:
        print(f"[warn] 跳过 {len(skipped)} 个命名不符合 gubapostYYYYMMDD.parquet 的文件: "
              f"{[os.path.basename(p) for p in skipped]}", flush=True)
    return files


def file_date(path: str) -> str:
    m = re.search(r"gubapost(\d{8})\.parquet$", os.path.basename(path))
    if not m:
        raise ValueError(f"文件名不符合 gubapostYYYYMMDD.parquet: {path}")
    return m.group(1)


# --------------------------------------------------------------------------- #
# tokenize(spawn 池, 变长输出, 不 pad)
# --------------------------------------------------------------------------- #
_TOKENIZER = None
_MAX_LENGTH = None


def _tokenizer_init(base_dir: str, max_length: int) -> None:
    global _TOKENIZER, _MAX_LENGTH
    from transformers import BertTokenizerFast

    _TOKENIZER = BertTokenizerFast.from_pretrained(base_dir, local_files_only=True)
    _MAX_LENGTH = max_length


def _tokenize_chunk(texts: list[str]) -> list[list[int]]:
    encoded = _TOKENIZER(
        texts,
        truncation=True,
        padding=False,
        max_length=_MAX_LENGTH,
    )
    return encoded["input_ids"]


def build_texts(df) -> np.ndarray:
    """title+content 拼接; content 已包含 title 或 title 为空时不再重复。"""
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


def make_batches(lengths: np.ndarray, order: np.ndarray, batch_tokens: int, batch_rows: int):
    """按长度升序扫描, token 预算 + 行数上限双约束切 batch。yield 排序后的索引片段。"""
    start = 0
    n = len(order)
    while start < n:
        end = start + 1
        longest = int(lengths[order[end - 1]])
        while end < n:
            cand_longest = max(longest, int(lengths[order[end]]))
            cand_rows = end - start + 1
            if cand_rows > batch_rows or cand_rows * cand_longest > batch_tokens:
                break
            longest = cand_longest
            end += 1
        yield order[start:end]
        start = end


def forward_file(
    in_path: str,
    out_path: str,
    *,
    model,
    pad_id: int,
    batch_tokens: int,
    batch_rows: int,
    token_workers: int,
    device: str,
    pool: ProcessPoolExecutor,
) -> dict:
    """读一个日文件 → tokenize → 分桶前向 → 写 parquet。返回统计信息。"""
    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq
    import torch

    t0 = time.time()
    df = pd.read_parquet(in_path, columns=SOURCE_COLUMNS)
    n = len(df)
    texts = build_texts(df)

    # tokenize: 分 chunk 提交 spawn 池, 按序收回(变长 id 列表)
    chunk_size = max(1, n // (token_workers * 4) if n > token_workers * 4 else 1)
    futures = []
    for i in range(0, n, chunk_size):
        futures.append(pool.submit(_tokenize_chunk, texts[i:i + chunk_size].tolist()))
    tok_lists: list[list[int]] = []
    for fut in futures:
        tok_lists.extend(fut.result())
    lengths = np.fromiter((len(x) for x in tok_lists), dtype=np.int32, count=n)
    t_tok = time.time()

    # 长度分桶(升序), 组 batch 前向, 结果按原顺序写回
    order = np.argsort(lengths, kind="stable")
    out0 = np.empty(n, dtype=np.float32)
    out1 = np.empty(n, dtype=np.float32)
    with torch.inference_mode():
        for idx in make_batches(lengths, order, batch_tokens, batch_rows):
            chunk = [tok_lists[i] for i in idx]
            maxlen = max(len(x) for x in chunk)
            ids = np.full((len(chunk), maxlen), pad_id, dtype=np.int64)
            for r, seq in enumerate(chunk):
                ids[r, : len(seq)] = seq
            b = torch.from_numpy(ids).to(device, non_blocking=True)
            mask = (b != pad_id)
            probs = model(b, attention_mask=mask).probabilities.float().cpu().numpy()
            out0[idx] = probs[:, 0]
            out1[idx] = probs[:, 1]
    t_fwd = time.time()

    out = pd.DataFrame({
        "post_id": df["postId"].astype(str).values,
        "symbol": df["baName"].astype(str).values,
        "date": df["date"].astype(str).values,
        "available_date": df["available_date"].astype(str).values,
        "class_0_prob": out0,
        "class_1_prob": out1,
    })
    table = pa.Table.from_pandas(out[OUT_COLUMNS], preserve_index=False)
    tmp = out_path + ".tmp.parquet"
    pq.write_table(table, tmp, compression="snappy")
    os.replace(tmp, out_path)

    return {
        "file": os.path.basename(in_path),
        "rows": int(n),
        "tokenize_s": round(t_tok - t0, 2),
        "forward_s": round(t_fwd - t_tok, 2),
        "total_s": round(time.time() - t0, 2),
        "rows_per_s": round(n / max(time.time() - t0, 1e-6), 1),
    }


def output_valid(out_path: str, n_src_rows: int | None) -> bool:
    """产物存在且(能取到行数时)行数与源文件一致才算完成。"""
    if not os.path.exists(out_path):
        return False
    try:
        import pyarrow.parquet as pq

        if n_src_rows is not None and pq.ParquetFile(out_path).metadata.num_rows != n_src_rows:
            return False
    except Exception:
        return False
    return True


def worker_main(args, shard_files: list[str], worker_id: int) -> int:
    import pandas as pd
    import pyarrow.parquet as pq
    import torch
    from multiprocessing import get_context

    from src.config import load_yaml_config
    from src.models.modeling import build_candidate

    log = open(os.path.join(args.out_dir, "logs", f"worker{worker_id}.log"), "a", buffering=1)
    def echo(msg):
        log.write(f"[{time.strftime('%m-%d %H:%M:%S')}] {msg}\n")

    cfg = load_yaml_config(os.path.join(ROOT, "configs", "model.yaml"))
    base = os.path.join(ROOT, cfg["paths"]["base_model_dir"])
    ckpt = os.path.join(ROOT, cfg["paths"]["checkpoint"])
    device = "cuda"
    dtype = torch.float16 if args.dtype == "fp16" else torch.float32

    model = build_candidate(base, ckpt, pooling="cls", pooling_confirmed=False,
                            device=device, dtype=dtype).eval()
    from transformers import BertTokenizerFast

    tok = BertTokenizerFast.from_pretrained(base, local_files_only=True)
    pad_id = int(tok.pad_token_id)
    del tok

    pool = ProcessPoolExecutor(
        max_workers=args.token_workers, mp_context=get_context("spawn"),
        initializer=_tokenizer_init, initargs=(base, args.max_length))

    done_log = os.path.join(args.out_dir, "completed.jsonl")
    fail_log = os.path.join(args.out_dir, "failed.jsonl")
    echo(f"worker{worker_id}(GPU{os.environ.get('GUBAPOST_GPU', '?')}) 启动: "
         f"{len(shard_files)} 个日文件, device={torch.cuda.get_device_name(0)}, dtype={args.dtype}")

    n_done = n_skip = n_fail = 0
    for k, in_path in enumerate(shard_files):
        d = file_date(in_path)
        out_path = os.path.join(args.out_dir, "probs_daily", f"gubapost_probs_{d}.parquet")
        try:
            n_src = pq.ParquetFile(in_path).metadata.num_rows
        except Exception:
            n_src = None
        if output_valid(out_path, n_src):
            n_skip += 1
            continue
        try:
            stat = forward_file(
                in_path, out_path, model=model, pad_id=pad_id,
                batch_tokens=args.batch_tokens,
                batch_rows=args.batch_rows, token_workers=args.token_workers,
                device=device, pool=pool)
            with open(done_log, "a") as f:
                f.write(json.dumps({"worker": worker_id, **stat}) + "\n")
            n_done += 1
            echo(f"({k+1}/{len(shard_files)}) {stat['file']} rows={stat['rows']} "
                 f"tok={stat['tokenize_s']}s fwd={stat['forward_s']}s "
                 f"{stat['rows_per_s']}行/s")
        except Exception:
            err = traceback.format_exc()
            with open(fail_log, "a") as f:
                f.write(json.dumps({"worker": worker_id, "file": os.path.basename(in_path),
                                    "error": err[-2000:]}) + "\n")
            n_fail += 1
            echo(f"({k+1}/{len(shard_files)}) FAIL {in_path}\n{err}")
    pool.shutdown()
    echo(f"worker{worker_id} 结束: done={n_done} skip={n_skip} fail={n_fail}")
    log.close()
    return 0 if n_fail == 0 else 3


# --------------------------------------------------------------------------- #
# 父进程: 分片 + 拉起 worker + manifest
# --------------------------------------------------------------------------- #
def main():
    args = parse_args()
    os.makedirs(os.path.join(args.out_dir, "probs_daily"), exist_ok=True)
    os.makedirs(os.path.join(args.out_dir, "logs"), exist_ok=True)

    files = discover_files(args.input_dir)
    if args.years:
        files = [f for f in files if os.path.basename(os.path.dirname(f)) in args.years]
        if not files:
            raise FileNotFoundError(f"--years {sorted(args.years)} 过滤后没有剩余文件")
    if args.only:
        keep = {s.strip() for s in args.only.split(",") if s.strip()}
        files = [f for f in files if file_date(f) in keep]
    if args.max_files:
        files = files[: args.max_files]
    print(f"共 {len(files)} 个日文件待处理 (gpus={args.gpus}, workers={args.workers})",
          flush=True)

    import subprocess

    n_w = args.workers
    shards = [files[i::n_w] for i in range(n_w)]
    procs = []
    try:
        for w in range(n_w):
            shard_list = os.path.join(args.out_dir, "logs", f"shard{w}.txt")
            with open(shard_list, "w") as f:
                f.write("\n".join(shards[w]))
            cmd = [sys.executable, os.path.abspath(__file__)]
            # 子进程复用父进程参数, 但只处理自己的分片(经 GUBAPOST_SHARD_LIST 传入)
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
            print(f"启动 worker{w}: GPU{args.gpus[w % len(args.gpus)]}, "
                  f"{len(shards[w])} 个文件 (日志 logs/worker{w}.log)", flush=True)

        # 驱动侧 tqdm 进度条(写 stderr → run.log; 终端 tail -f 可见)
        # 无 tqdm 时退化为每 60s 打一行。
        def _done_names() -> set:
            path = os.path.join(args.out_dir, "completed.jsonl")
            if not os.path.exists(path):
                return set()
            with open(path) as f:
                return {json.loads(line)["file"] for line in f if line.strip()}

        try:
            from tqdm import tqdm
        except ImportError:
            tqdm = None
        if tqdm is not None:
            bar = tqdm(total=len(files), desc="gubapost inference", unit="file",
                       mininterval=5, file=sys.stderr)
            seen = _done_names()
            bar.update(len(seen))
            while any(p.poll() is None for p, _, _ in procs):
                time.sleep(5)
                cur = _done_names()
                if len(cur) > len(seen):
                    bar.update(len(cur) - len(seen))
                    seen = cur
            cur = _done_names()
            if len(cur) > len(seen):
                bar.update(len(cur) - len(seen))
            bar.close()
        else:
            while any(p.poll() is None for p, _, _ in procs):
                time.sleep(60)
                print(f"[poll] 完成 {len(_done_names())}/{len(files)} 个文件", flush=True)
        rc = [p.wait() for p, _, _ in procs]
    except BaseException:
        # notebook 中断内核 / Ctrl-C: 确保 GPU worker 不变孤儿进程
        print("[abort] 父进程被中断, 终止所有 worker ...", flush=True)
        for p, _, _ in procs:
            p.terminate()
        for p, _, _ in procs:
            try:
                p.wait(10)
            except subprocess.TimeoutExpired:
                p.kill()
        raise

    for (p, n, g), c in zip(procs, rc):
        print(f"GPU{g} worker 退出码 {c} (分配 {n} 文件)", flush=True)

    # manifest(只统计本轮文件集并按文件去重, 历史残留记录不计入)
    expect = {os.path.basename(f) for f in files}
    done_by_file: dict = {}
    failed_by_file: dict = {}
    for name, bucket in (("completed.jsonl", done_by_file), ("failed.jsonl", failed_by_file)):
        path = os.path.join(args.out_dir, name)
        if os.path.exists(path):
            with open(path) as f:
                for line in f:
                    d = json.loads(line)
                    if d.get("file") in expect:
                        bucket[d["file"]] = d
    done, failed = list(done_by_file.values()), list(failed_by_file.values())
    total_rows = sum(d["rows"] for d in done)
    manifest = {
        "status": "completed" if not failed and len(done) == len(files) else "partial",
        "input_dir": args.input_dir,
        "years": sorted(args.years) if args.years else "all",
        "n_files": len(files),
        "n_done_files": len(done),
        "n_failed_files": len(failed),
        "n_rows_inferred": total_rows,
        "max_length": args.max_length,
        "dtype": args.dtype,
        "batch_tokens": args.batch_tokens,
        "batch_rows": args.batch_rows,
        "gpus": args.gpus,
        "workers": args.workers,
        "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(os.path.join(args.out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print("manifest:", json.dumps(manifest, ensure_ascii=False), flush=True)
    return 0 if manifest["status"] == "completed" else 3


if __name__ == "__main__":
    # 子进程 worker 模式: 父进程通过环境变量传入分片清单
    if os.environ.get("GUBAPOST_SHARD_LIST"):
        _args = parse_args()
        with open(os.environ["GUBAPOST_SHARD_LIST"]) as _f:
            _shard = [line.strip() for line in _f if line.strip()]
        _wid = int(os.environ.get("GUBAPOST_WORKER_ID", "0"))
        sys.exit(worker_main(_args, _shard, _wid))
    sys.exit(main())
