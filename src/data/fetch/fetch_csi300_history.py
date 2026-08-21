#!/usr/bin/env python
"""从 Wind Oracle 提取沪深300历史权重及可选指数收盘价。

数据库凭据复用同目录 ``config.py``。历史权重是必需输出；指数行情表不存在或
不可访问时仍保留权重输出，并在审计中记录失败，验证入口随后用权重和个股收益
构造基准，不会退回当前成分股。
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
import time
from pathlib import Path

import pandas as pd

DEFAULT_OUTPUT_DIR = Path("/home/intern_fjq_2026/data/NLP/market")
WEIGHT_COLUMNS = ["S_INFO_WINDCODE", "S_CON_WINDCODE", "TRADE_DT", "I_WEIGHT"]
PRICE_COLUMNS = ["S_INFO_WINDCODE", "TRADE_DT", "S_DQ_CLOSE"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="提取沪深300历史权重和指数收盘价")
    parser.add_argument("--index-code", default="000300.SH")
    parser.add_argument("--start-date", default="20050101")
    parser.add_argument("--end-date", default="20260820")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--batch-size", type=int, default=50_000)
    return parser.parse_args()


def _credentials() -> tuple[str, str, str]:
    fetch_dir = os.path.dirname(os.path.abspath(__file__))
    if fetch_dir not in sys.path:
        sys.path.insert(0, fetch_dir)
    try:
        cfg = importlib.import_module("config")
    except ModuleNotFoundError as exc:
        raise ValueError(
            "缺少远端私有配置 src/data/fetch/config.py，无法连接 Wind"
        ) from exc
    names = ("WIND_USER", "WIND_PWD", "WIND_DSN")
    values = tuple(str(getattr(cfg, name, "") or "").strip() for name in names)
    missing = [name for name, value in zip(names, values) if not value]
    if missing:
        raise ValueError("config.py 缺少 Wind 凭据: " + ", ".join(missing))
    return values


def _fetch_frame(cursor, sql: str, columns: list[str], params: dict, batch: int) -> pd.DataFrame:
    cursor.arraysize = batch
    cursor.prefetchrows = batch
    cursor.execute(sql, params)
    parts: list[pd.DataFrame] = []
    while rows := cursor.fetchmany(batch):
        parts.append(pd.DataFrame.from_records(rows, columns=columns))
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=columns)


def _atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    frame.to_parquet(temporary, index=False, compression="zstd")
    os.replace(temporary, path)


def _atomic_json(value: dict, path: Path) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, default=str)
    os.replace(temporary, path)


def export(args: argparse.Namespace) -> dict[str, object]:
    for name in ("start_date", "end_date"):
        value = str(getattr(args, name))
        if len(value) != 8 or not value.isdigit():
            raise ValueError(f"--{name.replace('_', '-')} 必须为YYYYMMDD")
    if args.end_date < args.start_date:
        raise ValueError("日期区间倒置")
    if args.batch_size < 1_000:
        raise ValueError("batch-size不得低于1000")

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"{args.start_date}_{args.end_date}"
    weights_path = output_dir / f"csi300_weights_{suffix}.parquet"
    prices_path = output_dir / f"csi300_index_prices_{suffix}.parquet"
    audit_path = output_dir / f"csi300_history_{suffix}_audit.json"

    try:
        import oracledb
    except ImportError as exc:
        raise RuntimeError("取数要求安装oracledb") from exc

    user, password, dsn = _credentials()
    started = time.perf_counter()
    connection = oracledb.connect(user=user, password=password, dsn=dsn)
    index_error = None
    try:
        cursor = connection.cursor()
        weights = _fetch_frame(
            cursor,
            """
            SELECT S_INFO_WINDCODE, S_CON_WINDCODE, TRADE_DT, I_WEIGHT
            FROM WIND.AINDEXHS300WEIGHT
            WHERE S_INFO_WINDCODE = :index_code
              AND TRADE_DT >= :start_date AND TRADE_DT <= :end_date
            ORDER BY TRADE_DT, S_CON_WINDCODE
            """,
            WEIGHT_COLUMNS,
            {
                "index_code": args.index_code,
                "start_date": args.start_date,
                "end_date": args.end_date,
            },
            args.batch_size,
        )
        if weights.empty:
            raise ValueError("AINDEXHS300WEIGHT未返回任何历史权重")
        _atomic_parquet(weights, weights_path)

        try:
            prices = _fetch_frame(
                cursor,
                """
                SELECT S_INFO_WINDCODE, TRADE_DT, S_DQ_CLOSE
                FROM WIND.AINDEXEODPRICES
                WHERE S_INFO_WINDCODE = :index_code
                  AND TRADE_DT >= :start_date AND TRADE_DT <= :end_date
                ORDER BY TRADE_DT
                """,
                PRICE_COLUMNS,
                {
                    "index_code": args.index_code,
                    "start_date": args.start_date,
                    "end_date": args.end_date,
                },
                args.batch_size,
            )
            if prices.empty:
                index_error = "AINDEXEODPRICES returned no rows"
            else:
                _atomic_parquet(prices, prices_path)
        except oracledb.Error as exc:
            index_error = str(exc)
            prices = pd.DataFrame(columns=PRICE_COLUMNS)
        if prices.empty and prices_path.exists():
            # 同一区间重跑失败时不保留上次的指数行情，避免验证
            # 入口将旧文件误当成本次成功产物。
            prices_path.unlink()
    finally:
        connection.close()

    audit = {
        "index_code": args.index_code,
        "start_date": args.start_date,
        "end_date": args.end_date,
        "weights": {"path": str(weights_path), "rows": len(weights)},
        "index_prices": {
            "path": str(prices_path) if not prices.empty else "unavailable",
            "rows": len(prices),
            "error": index_error,
        },
        "index_return_fallback": "historical_weighted_rtn_1d",
        "seconds": time.perf_counter() - started,
    }
    _atomic_json(audit, audit_path)
    return audit


def main() -> int:
    try:
        result = export(parse_args())
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        print(f"[error] {exc}", file=sys.stderr, flush=True)
        return 2
    print(json.dumps(result, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
