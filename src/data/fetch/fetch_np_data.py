#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""从 Wind Oracle 增量导出年度合并利润表净利润。

数据库连接凭据从同目录 config.py 读取（WIND_USER/WIND_PWD/WIND_DSN）。原始五列历史
CSV不覆盖；默认另写年度合并口径五列 CSV 和用于 point-in-time 清洗的增强 Parquet。
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as cfg

BASE_COLUMNS = [
    "S_INFO_WINDCODE",
    "ANN_DT",
    "REPORT_PERIOD",
    "NET_PROFIT_INCL_MIN_INT_INC",
    "ACTUAL_ANN_DT",
]
ENRICHED_COLUMNS = BASE_COLUMNS + [
    "STATEMENT_TYPE",
    "NET_PROFIT_EXCL_MIN_INT_INC",
    "OBJECT_ID",
    "OPDATE",
]
DEFAULT_OUTPUT_DIR = Path("/home/intern_fjq_2026/data/NLP/market")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="导出Wind年度合并净利润")
    parser.add_argument("--start-date", default="20050101")
    parser.add_argument("--end-date", default="20260801")
    parser.add_argument("--batch-size", type=int, default=50_000)
    parser.add_argument("--csv-output", default=None)
    parser.add_argument("--parquet-output", default=None)
    return parser.parse_args()


def _credentials() -> tuple[str, str, str]:
    """从同目录 config.py 读取 Wind 镜像库凭据。"""
    names = ("WIND_USER", "WIND_PWD", "WIND_DSN")
    values = tuple(str(getattr(cfg, name, "") or "").strip() for name in names)
    missing = [name for name, value in zip(names, values) if not value]
    if missing:
        raise ValueError("config.py 缺少 Wind 凭据: " + ", ".join(missing))
    return values


def _sql() -> str:
    return (
        "SELECT "
        + ", ".join(ENRICHED_COLUMNS)
        + " FROM WIND.ASHAREINCOME"
        + " WHERE ACTUAL_ANN_DT >= :start_date AND ACTUAL_ANN_DT <= :end_date"
        + " AND SUBSTR(REPORT_PERIOD, 5, 4) = '1231'"
        + " AND STATEMENT_TYPE = '408001000'"
        + " ORDER BY ACTUAL_ANN_DT, S_INFO_WINDCODE, REPORT_PERIOD, OBJECT_ID"
    )


def export(args: argparse.Namespace) -> dict[str, object]:
    if len(args.start_date) != 8 or len(args.end_date) != 8:
        raise ValueError("start-date/end-date 必须为YYYYMMDD")
    if args.end_date < args.start_date:
        raise ValueError("日期区间倒置")
    if args.batch_size < 50_000:
        raise ValueError("batch-size 不得低于50000")

    csv_output = Path(
        args.csv_output
        or DEFAULT_OUTPUT_DIR
        / f"ashare_np_annual_consolidated_{args.start_date}_{args.end_date}.csv"
    ).expanduser()
    parquet_output = Path(
        args.parquet_output
        or DEFAULT_OUTPUT_DIR
        / f"ashare_np_enriched_{args.start_date}_{args.end_date}.parquet"
    ).expanduser()
    csv_output.parent.mkdir(parents=True, exist_ok=True)
    parquet_output.parent.mkdir(parents=True, exist_ok=True)
    csv_tmp = csv_output.with_name(csv_output.name + ".tmp")
    parquet_tmp = parquet_output.with_name(parquet_output.name + ".tmp")

    try:
        import oracledb
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("导出要求安装 oracledb 与 pyarrow") from exc

    user, password, dsn = _credentials()
    started = time.perf_counter()
    count = 0
    writer = None
    succeeded = False
    try:
        connection = oracledb.connect(user=user, password=password, dsn=dsn)
    except oracledb.Error as exc:
        raise RuntimeError(f"连接 Wind 数据库失败: {exc}") from exc
    try:
        cursor = connection.cursor()
        cursor.arraysize = args.batch_size
        cursor.prefetchrows = args.batch_size
        cursor.execute(_sql(), start_date=args.start_date, end_date=args.end_date)
        with csv_tmp.open("w", newline="", encoding="utf-8") as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(BASE_COLUMNS)
            while rows := cursor.fetchmany(args.batch_size):
                columns = list(zip(*rows))
                arrays = []
                for name, values in zip(ENRICHED_COLUMNS, columns):
                    if name in {
                        "NET_PROFIT_INCL_MIN_INT_INC",
                        "NET_PROFIT_EXCL_MIN_INT_INC",
                    }:
                        arrays.append(
                            pa.array(values, type=pa.float64(), from_pandas=True)
                        )
                    else:
                        arrays.append(
                            pa.array(
                                [
                                    None if value is None else str(value)
                                    for value in values
                                ],
                                type=pa.string(),
                            )
                        )
                batch = pa.RecordBatch.from_arrays(arrays, ENRICHED_COLUMNS)
                if writer is None:
                    metadata = {
                        b"actual_metric": b"net_profit_incl_min_int_inc",
                        b"statement_type": b"408001000",
                        b"date_filter": b"ACTUAL_ANN_DT",
                    }
                    schema = batch.schema.with_metadata(metadata)
                    writer = pq.ParquetWriter(
                        parquet_tmp, schema, compression="zstd", use_dictionary=True
                    )
                writer.write_batch(batch)
                csv_writer.writerows(tuple(row[: len(BASE_COLUMNS)]) for row in rows)
                count += len(rows)
                if count % 500_000 < len(rows):
                    elapsed = max(time.perf_counter() - started, 1e-9)
                    print(
                        f"written {count:,} rows ({count / elapsed:,.0f} rows/s)",
                        flush=True,
                    )
        if writer is None:
            empty_schema = pa.schema(
                [
                    (
                        name,
                        (
                            pa.float64()
                            if name
                            in {
                                "NET_PROFIT_INCL_MIN_INT_INC",
                                "NET_PROFIT_EXCL_MIN_INT_INC",
                            }
                            else pa.string()
                        ),
                    )
                    for name in ENRICHED_COLUMNS
                ],
                metadata={
                    b"actual_metric": b"net_profit_incl_min_int_inc",
                    b"statement_type": b"408001000",
                    b"date_filter": b"ACTUAL_ANN_DT",
                },
            )
            writer = pq.ParquetWriter(parquet_tmp, empty_schema, compression="zstd")
        writer.close()
        writer = None
        os.replace(csv_tmp, csv_output)
        os.replace(parquet_tmp, parquet_output)
        succeeded = True
    except oracledb.Error as exc:
        raise RuntimeError(f"Wind 数据库查询失败: {exc}") from exc
    finally:
        if writer is not None:
            writer.close()
        connection.close()
        if not succeeded:
            for temporary in (csv_tmp, parquet_tmp):
                if temporary.is_file():
                    temporary.unlink()
    elapsed = time.perf_counter() - started
    return {
        "rows": count,
        "seconds": elapsed,
        "rows_per_second": count / elapsed if elapsed else None,
        "csv": str(csv_output),
        "parquet": str(parquet_output),
    }


def main() -> int:
    try:
        result = export(parse_args())
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        print(f"[error] {exc}", file=sys.stderr, flush=True)
        return 2
    print(f"DONE: {result}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())