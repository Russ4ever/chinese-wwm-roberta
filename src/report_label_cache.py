"""单篇研报规范化缓存：源JSONL只在缓存失效时解析一次。"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Sequence

import pandas as pd

from .report_labeling import canonicalize_report_rows


CACHE_SCHEMA_VERSION = 2
REPORT_FIELDS = [
    "ID",
    "STOCK_CODE",
    "ORGAN_NAME",
    "AUTHOR_NAME",
    "TITLE",
    "CONTENT",
    "CREATE_DATE",
    "REPORT_YEAR",
    "FORECAST_NP",
]


def _blake3_file(path: Path, threads: int) -> str:
    from blake3 import blake3

    digest = blake3(max_threads=max(1, int(threads)))
    digest.update_mmap(str(path))
    return digest.hexdigest()


def _cache_identity(
    source: Path,
    calendar_identity: dict[str, object],
    horizons: Sequence[int],
    multiplier: float,
) -> tuple[str, dict[str, object]]:
    stat = source.stat()
    material: dict[str, object] = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "source_path": str(source.resolve()),
        "source_size": stat.st_size,
        "source_mtime_ns": stat.st_mtime_ns,
        "calendar": calendar_identity,
        "forecast_horizons": [int(x) for x in horizons],
        "forecast_multiplier": float(multiplier),
    }
    key = hashlib.sha256(
        json.dumps(material, sort_keys=True, ensure_ascii=True).encode("utf-8")
    ).hexdigest()[:24]
    return key, material


def _read_reports_source(path: Path) -> pd.DataFrame:
    import polars as pl

    suffix = path.suffix.lower()
    if suffix in {".jsonl", ".ndjson", ".json"}:
        frame = pl.read_ndjson(
            path,
            infer_schema_length=10_000,
            batch_size=16_384,
            low_memory=False,
            ignore_errors=True,
        )
    elif suffix in {".parquet", ".pq"}:
        frame = pl.read_parquet(path)
    else:
        raise ValueError(f"研报缓存仅支持JSONL/Parquet，实际为: {path}")
    missing = sorted(set(REPORT_FIELDS).difference(frame.columns))
    if missing:
        raise ValueError("研报源缺少字段: " + ", ".join(missing))
    return frame.select(REPORT_FIELDS).to_pandas()


def _write_partitioned(
    reports: pd.DataFrame, rows: pd.DataFrame, destination: Path
) -> None:
    import polars as pl

    destination.mkdir(parents=True, exist_ok=True)
    years = sorted(
        set(reports["available_date"].dt.year.dropna().astype(int)).union(
            rows["available_date"].dt.year.dropna().astype(int)
        )
    )
    for year in years:
        partition = destination / f"year={year}"
        partition.mkdir(parents=True, exist_ok=True)
        report_part = reports[reports["available_date"].dt.year == year]
        row_part = rows[rows["available_date"].dt.year == year]
        pl.from_pandas(report_part).write_parquet(
            partition / "reports.parquet", compression="zstd", statistics=True
        )
        pl.from_pandas(row_part).write_parquet(
            partition / "report_fy.parquet", compression="zstd", statistics=True
        )


def _read_partitions(
    cache_version: Path, read_start: pd.Timestamp, read_end: pd.Timestamp
) -> tuple[pd.DataFrame, pd.DataFrame]:
    import polars as pl

    report_parts: list[Path] = []
    row_parts: list[Path] = []
    for year in range(read_start.year, read_end.year + 1):
        partition = cache_version / f"year={year}"
        if (partition / "reports.parquet").is_file():
            report_parts.append(partition / "reports.parquet")
        if (partition / "report_fy.parquet").is_file():
            row_parts.append(partition / "report_fy.parquet")
    if not row_parts:
        return pd.DataFrame(), pd.DataFrame()
    reports = (
        pl.read_parquet(report_parts)
        .filter(
            pl.col("available_date").is_between(read_start, read_end, closed="both")
        )
        .to_pandas()
        if report_parts
        else pd.DataFrame()
    )
    rows = (
        pl.read_parquet(row_parts)
        .filter(
            pl.col("available_date").is_between(read_start, read_end, closed="both")
        )
        .to_pandas()
    )
    return reports, rows


def _cleanup_stale_versions(cache_root: Path, keep: Path) -> None:
    for candidate in cache_root.iterdir():
        if (
            candidate == keep
            or not candidate.is_dir()
            or candidate.name.startswith(".")
        ):
            continue
        marker = candidate / "manifest.json"
        if marker.is_file():
            shutil.rmtree(candidate)


def load_or_build_report_cache(
    reports_path: str | Path,
    cache_directory: str | Path,
    trading_dates: Iterable[object],
    *,
    calendar_identity: dict[str, object],
    forecast_horizons: Sequence[int],
    forecast_multiplier: float,
    read_start: pd.Timestamp,
    read_end: pd.Timestamp,
    enabled: bool,
    hash_threads: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    source = Path(reports_path).expanduser().resolve()
    cache_root = Path(cache_directory).expanduser().resolve()
    key, identity = _cache_identity(
        source, calendar_identity, forecast_horizons, forecast_multiplier
    )
    version = cache_root / key
    manifest_path = version / "manifest.json"
    if enabled and manifest_path.is_file():
        with manifest_path.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        reports, rows = _read_partitions(version, read_start, read_end)
        return (
            reports,
            rows,
            {
                "enabled": True,
                "hit": True,
                "key": key,
                "directory": str(version),
                "source_blake3": manifest.get("source_blake3"),
                "source_rows": manifest.get("source_rows"),
            },
        )

    raw = _read_reports_source(source)
    reports, rows = canonicalize_report_rows(
        raw,
        trading_dates,
        forecast_horizons=forecast_horizons,
        forecast_multiplier=forecast_multiplier,
    )
    source_hash = _blake3_file(source, hash_threads)
    if enabled:
        cache_root.mkdir(parents=True, exist_ok=True)
        temporary = Path(tempfile.mkdtemp(prefix=f".{key}-", dir=cache_root))
        try:
            _write_partitioned(reports, rows, temporary)
            manifest = {
                **identity,
                "created_at": datetime.now().isoformat(timespec="seconds"),
                "source_blake3": source_hash,
                "source_rows": len(raw),
                "canonical_reports": len(reports),
                "canonical_report_fy_rows": len(rows),
            }
            with (temporary / "manifest.json").open("w", encoding="utf-8") as handle:
                json.dump(manifest, handle, ensure_ascii=False, indent=2)
            if version.exists():
                shutil.rmtree(version)
            os.replace(temporary, version)
            _cleanup_stale_versions(cache_root, version)
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)
    target_reports = reports[
        reports["available_date"].between(read_start, read_end, inclusive="both")
    ].reset_index(drop=True)
    target_rows = rows[
        rows["available_date"].between(read_start, read_end, inclusive="both")
    ].reset_index(drop=True)
    return (
        target_reports,
        target_rows,
        {
            "enabled": bool(enabled),
            "hit": False,
            "key": key,
            "directory": str(version) if enabled else None,
            "source_blake3": source_hash,
            "source_rows": len(raw),
        },
    )
