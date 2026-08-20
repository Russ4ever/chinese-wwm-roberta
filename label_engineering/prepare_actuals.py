#!/usr/bin/env python
"""把 Wind AShareIncome 增强导出整理成 point-in-time 年度 Actual。"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

ACTUAL_METRIC = "net_profit_incl_min_int_inc"
ACTUAL_VERSION = "annual_consolidated_first_actual_announcement"
CONSOLIDATED_STATEMENT_TYPE = 408001000
REQUIRED_COLUMNS = {
    "S_INFO_WINDCODE",
    "ANN_DT",
    "REPORT_PERIOD",
    "NET_PROFIT_INCL_MIN_INT_INC",
    "ACTUAL_ANN_DT",
    "STATEMENT_TYPE",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="整理年度首次披露净利润 Actual")
    parser.add_argument("--input", required=True, help="Wind增强CSV/Parquet")
    parser.add_argument("--output", default=None, help="canonical Actual Parquet")
    parser.add_argument("--conflicts-output", default=None)
    parser.add_argument("--audit-output", default=None)
    return parser.parse_args()


def _default_outputs(source: Path) -> tuple[Path, Path, Path]:
    parent = source.parent
    stem = source.stem.replace("_enriched", "")
    date_part = (
        stem.removeprefix("ashare_np_") if stem.startswith("ashare_np_") else stem
    )
    return (
        parent / f"ashare_np_annual_first_{date_part}.parquet",
        parent / "ashare_np_annual_first_conflicts.parquet",
        parent / "ashare_np_annual_first_audit.json",
    )


def _read_source(path: Path):
    try:
        import polars as pl
    except ImportError as exc:
        raise RuntimeError("Actual整理要求安装 polars") from exc
    suffix = path.suffix.lower()
    if suffix in {".parquet", ".pq"}:
        return pl.read_parquet(path)
    if suffix in {".csv", ".txt"}:
        return pl.read_csv(
            path,
            infer_schema=False,
            ignore_errors=False,
            n_threads=min(8, os.cpu_count() or 1),
        )
    raise ValueError(f"不支持的数据格式 {suffix!r}: {path}")


def _blake3_file(path: Path) -> str:
    try:
        from blake3 import blake3
    except ImportError as exc:
        raise RuntimeError("Actual整理要求安装 blake3") from exc
    digest = blake3(max_threads=min(8, os.cpu_count() or 1))
    digest.update_mmap(str(path))
    return digest.hexdigest()


def _write_parquet(frame: Any, path: Path, metadata: dict[str, str]) -> None:
    import pyarrow.parquet as pq

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    table = frame.to_arrow()
    encoded = {str(k).encode(): str(v).encode() for k, v in metadata.items()}
    table = table.replace_schema_metadata({**(table.schema.metadata or {}), **encoded})
    pq.write_table(table, temporary, compression="zstd", use_dictionary=True)
    os.replace(temporary, path)


def _write_json(value: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, default=str)
    os.replace(temporary, path)


def prepare_actuals(
    input_path: str | Path,
    output_path: str | Path,
    conflicts_path: str | Path,
    audit_path: str | Path,
) -> dict[str, Any]:
    import polars as pl

    source = Path(input_path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Actual源文件不存在: {source}")
    output = Path(output_path).expanduser().resolve()
    conflicts_output = Path(conflicts_path).expanduser().resolve()
    audit_output = Path(audit_path).expanduser().resolve()

    raw = _read_source(source).with_row_index("_source_row")
    missing = sorted(REQUIRED_COLUMNS.difference(raw.columns))
    if missing:
        raise ValueError(
            "Actual增强源缺少字段: "
            + ", ".join(missing)
            + "；请先用更新后的 fetch_np_data.py 重新导出"
        )
    for optional in ("NET_PROFIT_EXCL_MIN_INT_INC", "OBJECT_ID", "OPDATE"):
        if optional not in raw.columns:
            raw = raw.with_columns(pl.lit(None).alias(optional))

    normalized = raw.with_columns(
        pl.col("S_INFO_WINDCODE")
        .cast(pl.String, strict=False)
        .str.strip_chars()
        .str.to_uppercase()
        .alias("source_stock_code"),
        pl.col("REPORT_PERIOD")
        .cast(pl.String, strict=False)
        .str.replace(r"\.0$", "")
        .alias("report_period_text"),
        pl.col("ANN_DT")
        .cast(pl.String, strict=False)
        .str.replace(r"\.0$", "")
        .alias("ann_dt_text"),
        pl.col("ACTUAL_ANN_DT")
        .cast(pl.String, strict=False)
        .str.replace(r"\.0$", "")
        .alias("actual_ann_dt_text"),
        pl.col("STATEMENT_TYPE")
        .cast(pl.String, strict=False)
        .str.replace(r"\.0$", "")
        .cast(pl.Int64, strict=False)
        .alias("statement_type_int"),
        pl.col("NET_PROFIT_INCL_MIN_INT_INC")
        .cast(pl.Float64, strict=False)
        .alias("actual_np_value"),
        pl.col("NET_PROFIT_EXCL_MIN_INT_INC")
        .cast(pl.Float64, strict=False)
        .alias("parent_np_audit"),
        pl.col("OBJECT_ID")
        .cast(pl.String, strict=False)
        .fill_null("")
        .alias("object_id"),
        pl.col("OPDATE").cast(pl.String, strict=False).fill_null("").alias("opdate"),
    ).with_columns(
        pl.col("report_period_text")
        .str.strptime(pl.Date, "%Y%m%d", strict=False)
        .alias("report_period_date"),
        pl.col("ann_dt_text")
        .str.strptime(pl.Date, "%Y%m%d", strict=False)
        .alias("ann_date"),
        pl.col("actual_ann_dt_text")
        .str.strptime(pl.Date, "%Y%m%d", strict=False)
        .alias("actual_known_date"),
    )

    normalized = normalized.with_columns(
        pl.when(~pl.col("source_stock_code").str.contains(r"^\d{6}\.(SH|SZ|BJ)$"))
        .then(pl.lit("invalid_stock_code"))
        .when(
            pl.col("report_period_date").is_null()
            | ~pl.col("report_period_text").str.ends_with("1231")
        )
        .then(pl.lit("not_valid_annual_period"))
        .when(pl.col("statement_type_int") != CONSOLIDATED_STATEMENT_TYPE)
        .then(pl.lit("not_consolidated_statement"))
        .when(
            pl.col("actual_np_value").is_null() | ~pl.col("actual_np_value").is_finite()
        )
        .then(pl.lit("invalid_actual_np"))
        .when(pl.col("ann_date").is_null())
        .then(pl.lit("invalid_ann_dt"))
        .when(pl.col("actual_known_date").is_null())
        .then(pl.lit("invalid_actual_ann_dt"))
        .when(pl.col("actual_known_date") < pl.col("ann_date"))
        .then(pl.lit("actual_ann_before_ann_dt"))
        .otherwise(pl.lit(None, dtype=pl.String))
        .alias("invalid_reason")
    )
    invalid = normalized.filter(pl.col("invalid_reason").is_not_null())
    valid = (
        normalized.filter(pl.col("invalid_reason").is_null())
        .with_columns(
            pl.col("source_stock_code").str.slice(0, 6).alias("stock_code"),
            pl.col("report_period_date").dt.year().alias("fy"),
        )
        .sort(
            [
                "stock_code",
                "fy",
                "actual_known_date",
                "object_id",
                "opdate",
                "_source_row",
            ]
        )
    )

    keys = ["stock_code", "fy"]
    valid = valid.with_columns(
        pl.col("actual_known_date").min().over(keys).alias("first_actual_known_date"),
        pl.len().over(keys).alias("n_source_rows"),
    )
    first_rows = valid.filter(
        pl.col("actual_known_date") == pl.col("first_actual_known_date")
    )
    grouped = (
        first_rows.group_by(keys, maintain_order=True)
        .agg(
            pl.col("actual_np_value").first().alias("actual_np"),
            pl.col("actual_np_value").min().alias("actual_np_min"),
            pl.col("actual_np_value").max().alias("actual_np_max"),
            pl.col("actual_known_date").first().alias("actual_known_date"),
            pl.col("ann_date").first().alias("ann_dt"),
            pl.col("report_period_date").first().alias("report_period"),
            pl.col("source_stock_code").first().alias("source_stock_code"),
            pl.col("statement_type_int").first().alias("statement_type"),
            pl.col("object_id").first().alias("source_id"),
            pl.col("opdate").first().alias("opdate"),
            pl.col("parent_np_audit").first().alias("parent_np_audit"),
            pl.len().alias("n_first_rows"),
            pl.col("n_source_rows").first().alias("n_source_rows"),
        )
        .with_columns(
            (
                (pl.col("actual_np_max") - pl.col("actual_np_min")).abs()
                > (pl.lit(0.01) + pl.lit(1e-10) * pl.col("actual_np").abs())
            ).alias("first_disclosure_conflict")
        )
        .with_columns(
            (pl.col("n_source_rows") - pl.col("n_first_rows")).alias(
                "n_later_revisions"
            )
        )
    )

    conflict_keys = grouped.filter(pl.col("first_disclosure_conflict")).select(keys)
    conflicts = (
        first_rows.join(conflict_keys, on=keys, how="inner")
        .with_columns(pl.lit("conflicting_first_disclosure").alias("conflict_reason"))
        .select(
            "stock_code",
            "fy",
            "source_stock_code",
            "ann_date",
            "report_period_date",
            "actual_known_date",
            "actual_np_value",
            "statement_type_int",
            "object_id",
            "opdate",
            "_source_row",
            "conflict_reason",
        )
    )

    canonical = (
        grouped.filter(~pl.col("first_disclosure_conflict"))
        .with_columns(
            pl.col("actual_known_date").alias("actual_publish_date"),
            pl.lit(1.0).alias("unit_multiplier"),
            pl.lit("CNY").alias("currency"),
            pl.when(pl.col("source_id") == "")
            .then(
                pl.concat_str(
                    [
                        "source_stock_code",
                        pl.col("report_period").cast(pl.String),
                        pl.col("actual_known_date").cast(pl.String),
                    ],
                    separator="|",
                )
            )
            .otherwise(pl.col("source_id"))
            .alias("source_id"),
            pl.lit(ACTUAL_VERSION).alias("actual_version"),
            pl.lit(ACTUAL_METRIC).alias("actual_metric"),
        )
        .select(
            "stock_code",
            "fy",
            "actual_np",
            "actual_publish_date",
            "actual_known_date",
            "unit_multiplier",
            "currency",
            "source_id",
            "actual_version",
            "actual_metric",
            "source_stock_code",
            "ann_dt",
            "report_period",
            "statement_type",
            "opdate",
            "parent_np_audit",
            "n_source_rows",
            "n_first_rows",
            "n_later_revisions",
        )
        .sort(["stock_code", "fy"])
    )

    source_stat = source.stat()
    source_hash = _blake3_file(source)
    metadata = {
        "actual_metric": ACTUAL_METRIC,
        "actual_version": ACTUAL_VERSION,
        "currency": "CNY",
        "unit_multiplier": "1.0",
        "known_date_field": "ACTUAL_ANN_DT",
        "source_blake3": source_hash,
    }
    _write_parquet(canonical, output, metadata)
    _write_parquet(conflicts, conflicts_output, metadata)

    reason_counts = {
        str(row["invalid_reason"]): int(row["len"])
        for row in invalid.group_by("invalid_reason").len().to_dicts()
    }
    audit: dict[str, Any] = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "source": {
            "path": str(source),
            "size_bytes": source_stat.st_size,
            "mtime_ns": source_stat.st_mtime_ns,
            "blake3": source_hash,
        },
        "outputs": {"canonical": str(output), "conflicts": str(conflicts_output)},
        "rules": {
            "statement_type": CONSOLIDATED_STATEMENT_TYPE,
            "annual_period_suffix": "1231",
            "known_date_field": "ACTUAL_ANN_DT",
            "same_day_rtol": 1e-10,
            "same_day_atol_yuan": 0.01,
            "actual_metric": ACTUAL_METRIC,
            "actual_version": ACTUAL_VERSION,
        },
        "counts": {
            "input_rows": raw.height,
            "valid_source_rows": valid.height,
            "invalid_source_rows": invalid.height,
            "canonical_rows": canonical.height,
            "conflicting_groups": conflict_keys.height,
            "conflicting_rows": conflicts.height,
            "invalid_reasons": reason_counts,
        },
        "invalid_examples": invalid.select(
            "_source_row",
            "source_stock_code",
            "report_period_text",
            "ann_dt_text",
            "actual_ann_dt_text",
            "invalid_reason",
        )
        .head(20)
        .to_dicts(),
    }
    _write_json(audit, audit_output)
    return audit


def main() -> int:
    args = parse_args()
    source = Path(args.input).expanduser()
    default_output, default_conflicts, default_audit = _default_outputs(source)
    try:
        audit = prepare_actuals(
            source,
            args.output or default_output,
            args.conflicts_output or default_conflicts,
            args.audit_output or default_audit,
        )
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        print(f"[error] {exc}", file=sys.stderr, flush=True)
        return 2
    print(json.dumps(audit["counts"], ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
