#!/usr/bin/env python
"""构建可配置日期区间的单篇研报未来验证 Label。

本入口只读取配置指向的已有研报、交易日历和可选 Actual 文件，不访问网络或数据库。
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import load_yaml_config  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="单篇研报未来验证 Label 构建")
    parser.add_argument(
        "--config", default=str(ROOT / "configs" / "report_labels.yaml")
    )
    parser.add_argument(
        "--start-date", default=None, help="覆盖配置中的 Label 开始日期"
    )
    parser.add_argument("--end-date", default=None, help="覆盖配置中的 Label 结束日期")
    parser.add_argument("--output-dir", default=None, help="覆盖配置中的输出目录")
    return parser.parse_args()


def _path(value: object) -> Path | None:
    text = str(value or "").strip()
    if not text:
        return None
    path = Path(text).expanduser()
    return path if path.is_absolute() else ROOT / path


def _date(value: object, name: str) -> pd.Timestamp | None:
    if value is None or str(value).strip() == "":
        return None
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        raise ValueError(f"{name} 不是有效日期: {value!r}")
    return pd.Timestamp(parsed).normalize()


def _blake3_file(path: Path, threads: int) -> str:
    from blake3 import blake3

    digest = blake3(max_threads=max(1, int(threads)))
    digest.update_mmap(str(path))
    return digest.hexdigest()


def _file_meta(
    path: Path | None, *, threads: int, known_blake3: str | None = None
) -> dict[str, object] | str:
    if path is None:
        return "unavailable"
    stat = path.stat()
    return {
        "path": str(path),
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "blake3": known_blake3 or _blake3_file(path, threads),
    }


def resolve_source_coverage(
    reports_path: Path,
    coverage_config: dict[str, Any],
) -> tuple[pd.Timestamp, pd.Timestamp, str]:
    explicit_start = _date(coverage_config.get("start_date"), "coverage.start_date")
    explicit_end = _date(coverage_config.get("end_date"), "coverage.end_date")
    if (explicit_start is None) != (explicit_end is None):
        raise ValueError(
            "coverage.start_date 与 coverage.end_date 必须同时填写或同时留空"
        )
    if explicit_start is not None and explicit_end is not None:
        if explicit_end < explicit_start:
            raise ValueError("数据覆盖区间倒置")
        return explicit_start, explicit_end, "config"

    manifest_candidates = [
        reports_path.parent / "_manifest.json",
        reports_path.with_suffix(reports_path.suffix + ".manifest.json"),
    ]
    for manifest in manifest_candidates:
        if not manifest.is_file():
            continue
        try:
            with manifest.open("r", encoding="utf-8") as handle:
                metadata = json.load(handle)
            window = metadata.get("window", {})
            start = _date(window.get("start"), "manifest.window.start")
            end = _date(window.get("end"), "manifest.window.end")
            if start is not None and end is not None:
                return start, end, f"manifest:{manifest}"
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            pass

    match = re.search(r"(\d{8})_(\d{8})", reports_path.stem)
    if match:
        start = pd.Timestamp(datetime.strptime(match.group(1), "%Y%m%d")).normalize()
        end = pd.Timestamp(datetime.strptime(match.group(2), "%Y%m%d")).normalize()
        return start, end, "filename"
    raise ValueError(
        "无法确定研报数据真实覆盖边界；请在配置 coverage.start_date/end_date 中明确填写"
    )


def _read_tabular(path: Path) -> pd.DataFrame:
    import polars as pl

    suffix = path.suffix.lower()
    if suffix in {".parquet", ".pq"}:
        return pl.read_parquet(path).to_pandas()
    if suffix in {".csv", ".txt"}:
        return pl.read_csv(path, infer_schema_length=10_000).to_pandas()
    raise ValueError(f"不支持的数据格式 {suffix!r}: {path}；仅支持 CSV/Parquet")


def load_actuals(
    path: Path | None, schema_path: Path | None, *, expected_metric: str
) -> tuple[pd.DataFrame, bool, dict[str, Any]]:
    if path is None:
        return pd.DataFrame(), False, {}
    if not path.is_file():
        raise FileNotFoundError(f"Actual 文件不存在: {path}")
    frame = _read_tabular(path)
    parquet_metadata: dict[str, str] = {}
    if path.suffix.lower() in {".parquet", ".pq"}:
        import pyarrow.parquet as pq

        raw_metadata = pq.read_metadata(path).metadata or {}
        parquet_metadata = {
            key.decode("utf-8", errors="replace"): value.decode(
                "utf-8", errors="replace"
            )
            for key, value in raw_metadata.items()
        }
    schema: dict[str, Any] = (
        load_yaml_config(schema_path) if schema_path is not None else {}
    )
    columns = schema.get("columns", {})
    if columns:
        rename = {
            source: canonical
            for canonical, source in columns.items()
            if source in frame.columns
        }
        frame = frame.rename(columns=rename)
    for column, value in schema.get("defaults", {}).items():
        if column not in frame:
            frame[column] = value
    for column, allowed in schema.get("filters", {}).items():
        if column not in frame:
            raise ValueError(f"Actual filter 字段不存在: {column}")
        values = allowed if isinstance(allowed, list) else [allowed]
        frame = frame[frame[column].isin(values)]
    if "actual_metric" not in frame:
        raise ValueError(
            "Actual缺少 actual_metric 声明；必须使用prepare_actuals.py产物或schema defaults显式声明"
        )
    actual_metrics = set(frame["actual_metric"].dropna().astype(str).str.strip())
    if not actual_metrics and parquet_metadata.get("actual_metric"):
        actual_metrics = {parquet_metadata["actual_metric"]}
    if actual_metrics != {expected_metric}:
        raise ValueError(
            f"Actual指标{sorted(actual_metrics)}与预测指标{expected_metric!r}不一致"
        )
    if "actual_version" not in frame:
        raise ValueError("Actual缺少 actual_version 声明")
    expected_version = "annual_consolidated_first_actual_announcement"
    versions = set(frame["actual_version"].dropna().astype(str).str.strip())
    if not versions and parquet_metadata.get("actual_version"):
        versions = {parquet_metadata["actual_version"]}
    if versions != {expected_version}:
        raise ValueError(
            f"Actual版本必须为{expected_version!r}，实际为{sorted(versions)}"
        )
    if not frame.empty:
        known = pd.to_datetime(
            frame["actual_known_date"], errors="coerce"
        ).dt.normalize()
        published = pd.to_datetime(
            frame["actual_publish_date"], errors="coerce"
        ).dt.normalize()
        if known.isna().any() or published.isna().any() or (known != published).any():
            raise ValueError(
                "首次披露Actual要求actual_known_date与actual_publish_date均等于ACTUAL_ANN_DT"
            )
        multipliers = (
            pd.to_numeric(frame["unit_multiplier"], errors="coerce").dropna().unique()
        )
        if len(multipliers) != 1 or float(multipliers[0]) != 1.0:
            raise ValueError(
                "首次披露Actual必须已换算为人民币元，unit_multiplier必须为1"
            )
        currencies = set(frame["currency"].dropna().astype(str).str.upper())
        if currencies != {"CNY"}:
            raise ValueError(f"Actual币种必须为CNY，实际为{sorted(currencies)}")
    return frame, True, schema


def _atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    import polars as pl

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    pl.from_pandas(frame).write_parquet(temporary, compression="zstd", statistics=True)
    os.replace(temporary, path)


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def _atomic_json(value: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, default=str)
    os.replace(temporary, path)


def run(args: argparse.Namespace) -> dict[str, Any]:
    total_started = time.perf_counter()
    stage_metrics: dict[str, dict[str, float | int | None]] = {}

    def record_stage(name: str, started: float, rows: int | None = None) -> None:
        seconds = time.perf_counter() - started
        stage_metrics[name] = {
            "seconds": seconds,
            "rows": rows,
            "rows_per_second": (
                float(rows / seconds) if rows is not None and seconds > 0 else None
            ),
        }

    config_path = _path(args.config)
    if config_path is None or not config_path.is_file():
        raise FileNotFoundError(f"Label 配置不存在: {args.config}")
    config = load_yaml_config(config_path)
    from src.report_label_runtime import configure_runtime, peak_rss_mb

    resources = configure_runtime(config.get("performance", {}))
    # 以下导入必须晚于线程配置，确保Polars和Numba不会各自占满整台服务器。
    from src.report_label_cache import load_or_build_report_cache
    from src.report_labeling import (
        LABEL_VERSION,
        assign_point_in_time_scale,
        attach_actual_labels,
        attach_pre_consensus,
        build_confirmation_labels,
        build_coverage_audit,
        build_org_histories,
        validate_actual_scale,
        warm_numba_kernels,
    )
    from src.trading_calendar import load_trading_dates

    paths = config.get("paths", {})
    reports_path = _path(paths.get("reports"))
    calendar_path = _path(paths.get("trading_calendar"))
    actual_path = _path(paths.get("actuals"))
    actual_schema_path = _path(paths.get("actual_schema"))
    if reports_path is None:
        raise ValueError(
            "研报路径未配置；请填写 paths.reports 或 REPORT_LABEL_REPORTS_PATH"
        )
    if calendar_path is None:
        raise ValueError(
            "交易日历路径未配置；请填写 paths.trading_calendar 或 REPORT_LABEL_CALENDAR_PATH"
        )
    if not reports_path.is_file():
        raise FileNotFoundError(f"研报文件不存在: {reports_path}")
    if not calendar_path.is_file():
        raise FileNotFoundError(f"交易日历文件不存在: {calendar_path}")

    build = config.get("build", {})
    forecast_metric = str(
        build.get("forecast_metric", "net_profit_incl_min_int_inc")
    ).strip()
    if forecast_metric != "net_profit_incl_min_int_inc":
        raise ValueError("本轮FORECAST_NP只支持已确认的net_profit_incl_min_int_inc口径")
    horizons = tuple(int(value) for value in build.get("forecast_horizons", [0, 1, 2]))
    confirmation_months = tuple(
        int(value) for value in build.get("confirmation_months", [1, 3])
    )
    if not horizons or not confirmation_months or min(confirmation_months) <= 0:
        raise ValueError("forecast_horizons/confirmation_months 配置无效")
    lookback_days = int(build.get("lookback_days", 180))
    coverage_start, coverage_end, coverage_source = resolve_source_coverage(
        reports_path, config.get("coverage", {})
    )
    configured_start = (
        args.start_date if args.start_date is not None else build.get("start_date")
    )
    configured_end = (
        args.end_date if args.end_date is not None else build.get("end_date")
    )
    label_start = _date(configured_start, "build.start_date") or coverage_start
    label_end = _date(configured_end, "build.end_date") or coverage_end
    if label_start < coverage_start or label_end > coverage_end:
        raise ValueError(
            f"Label 区间 {label_start.date()}~{label_end.date()} 超出数据覆盖 "
            f"{coverage_start.date()}~{coverage_end.date()}"
        )
    if label_end < label_start:
        raise ValueError("Label 构建区间倒置")

    history_months = int(build.get("scale_history_months", 12))
    read_start = max(
        coverage_start,
        label_start
        - pd.DateOffset(months=history_months)
        - pd.Timedelta(days=lookback_days),
    )
    read_end = min(
        coverage_end,
        label_end
        + pd.DateOffset(months=max(confirmation_months))
        + pd.Timedelta(days=7),
    )
    date_column = str(config.get("calendar", {}).get("date_column", "date"))
    calendar_started = time.perf_counter()
    trading_dates = load_trading_dates(calendar_path, date_column=date_column)
    record_stage("calendar_load", calendar_started, len(trading_dates))
    # 允许日历只覆盖区间的一部分：逐条越界由 available_date/target_date 的
    # NaT 与删失标志处理；仅在二者完全不相交时阻止误配文件。
    if trading_dates.max() < read_start or trading_dates.min() > read_end:
        raise ValueError(
            f"交易日历覆盖 {trading_dates.min().date()}~{trading_dates.max().date()} "
            f"与读取区间 {read_start.date()}~{read_end.date()} 完全不相交"
        )

    cache_config = config.get("cache", {})
    cache_directory = _path(
        cache_config.get("directory", "artifacts/report_label_cache")
    )
    if cache_directory is None:
        raise ValueError("cache.directory不能为空")
    print(
        f"[1/6] 读取/构建单遍研报缓存 {read_start.date()}~{read_end.date()}",
        flush=True,
    )
    stage_started = time.perf_counter()
    calendar_stat = calendar_path.stat()
    reports, all_rows, cache_info = load_or_build_report_cache(
        reports_path,
        cache_directory,
        trading_dates,
        calendar_identity={
            "path": str(calendar_path.resolve()),
            "size": calendar_stat.st_size,
            "mtime_ns": calendar_stat.st_mtime_ns,
            "date_column": date_column,
        },
        forecast_horizons=horizons,
        forecast_multiplier=float(build.get("forecast_np_unit_multiplier", 10000.0)),
        read_start=read_start,
        read_end=read_end,
        enabled=bool(cache_config.get("enabled", False)),
        hash_threads=resources.effective_threads,
    )
    record_stage("report_cache_load_or_build", stage_started, len(all_rows))
    if all_rows.empty:
        raise ValueError("规范化后没有动态 FY0/FY1/FY2 预测记录")

    warmup_started = time.perf_counter()
    warm_numba_kernels()
    numba_compile_seconds = time.perf_counter() - warmup_started
    print("[2/6] 构造发布前同行共识与 point-in-time scale", flush=True)
    stage_started = time.perf_counter()
    histories = build_org_histories(all_rows)
    all_rows = attach_pre_consensus(
        all_rows,
        histories,
        coverage_start=coverage_start,
        lookback_days=lookback_days,
        min_peer_orgs=int(build.get("min_peer_orgs", 4)),
    )
    all_rows = assign_point_in_time_scale(
        all_rows,
        history_months=history_months,
        min_samples=int(build.get("min_scale_samples", 500)),
    )
    target_mask = (all_rows["available_date"] >= label_start) & (
        all_rows["available_date"] <= label_end
    )
    target_rows = all_rows.loc[target_mask].copy().reset_index(drop=True)
    if target_rows.empty:
        raise ValueError("Label 区间内没有可输出的报告×FY记录")
    target_ids = set(target_rows["report_id"])
    reports = (
        reports[reports["report_id"].isin(target_ids)]
        .sort_values(["available_date", "report_id"])
        .reset_index(drop=True)
    )
    record_stage("pre_consensus_and_scale", stage_started, len(all_rows))

    print("[3/6] 读取可选 Actual 并计算 Residual/Edge", flush=True)
    stage_started = time.perf_counter()
    actuals, actual_available, actual_schema = load_actuals(
        actual_path, actual_schema_path, expected_metric=forecast_metric
    )
    fy_labels = attach_actual_labels(
        target_rows, actuals, actual_source_available=actual_available
    )
    actual_scale_ratio = validate_actual_scale(fy_labels) if actual_available else None
    fy_labels["label_version"] = LABEL_VERSION
    record_stage("actual_residual_edge", stage_started, len(fy_labels))

    print("[4/6] 构造1/3个月三种同行面板 Future Confirmation", flush=True)
    stage_started = time.perf_counter()
    confirmation = build_confirmation_labels(
        fy_labels,
        histories,
        trading_dates,
        coverage_end=coverage_end,
        confirmation_months=confirmation_months,
        lookback_days=lookback_days,
        min_peer_orgs=int(build.get("min_peer_orgs", 4)),
        min_active_orgs=int(build.get("min_active_orgs", 3)),
        min_probe_updates=int(build.get("min_probe_updates", 2)),
    )
    audit = build_coverage_audit(fy_labels, confirmation)
    record_stage("future_confirmation_and_audit", stage_started, len(confirmation))

    output_value = (
        args.output_dir
        if args.output_dir is not None
        else config.get("output", {}).get("directory", "artifacts/report_labels")
    )
    output_dir = _path(output_value)
    if output_dir is None:
        raise ValueError("输出目录不能为空")
    print(f"[5/6] 写入 {output_dir}", flush=True)
    stage_started = time.perf_counter()
    report_columns = [
        "report_id",
        "source_report_ids",
        "stock_code",
        "org_id",
        "author_name",
        "title",
        "content",
        "text",
        "publish_timestamp",
        "publish_date",
        "available_date",
    ]
    _atomic_parquet(reports[report_columns], output_dir / "reports.parquet")
    fy_output = fy_labels
    _atomic_parquet(fy_output, output_dir / "report_fy_labels.parquet")
    _atomic_parquet(confirmation, output_dir / "report_confirmation_labels.parquet")
    _atomic_csv(audit, output_dir / "label_coverage_audit.csv")
    record_stage(
        "output_write", stage_started, len(reports) + len(fy_output) + len(confirmation)
    )

    stage_started = time.perf_counter()
    source_metadata = {
        "reports": _file_meta(
            reports_path,
            threads=resources.effective_threads,
            known_blake3=cache_info.get("source_blake3"),
        ),
        "trading_calendar": _file_meta(
            calendar_path, threads=resources.effective_threads
        ),
        "actuals": (
            _file_meta(actual_path, threads=resources.effective_threads)
            if actual_available
            else "unavailable"
        ),
        "actual_schema": (
            _file_meta(actual_schema_path, threads=resources.effective_threads)
            if actual_schema_path
            else "unavailable"
        ),
    }
    record_stage("source_fingerprints", stage_started)
    import numba
    import polars as pl

    thread_pools = {
        "numba_threads": numba.get_num_threads(),
        "polars_threads": pl.thread_pool_size(),
    }

    metadata: dict[str, Any] = {
        "label_version": LABEL_VERSION,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "configuration": {
            "label_start": str(label_start.date()),
            "label_end": str(label_end.date()),
            "read_start": str(read_start.date()),
            "read_end": str(read_end.date()),
            "coverage_start": str(coverage_start.date()),
            "coverage_end": str(coverage_end.date()),
            "coverage_source": coverage_source,
            "forecast_horizons": list(horizons),
            "confirmation_months": list(confirmation_months),
            "lookback_days": lookback_days,
            "scale_history_months": history_months,
            "min_scale_samples": int(build.get("min_scale_samples", 500)),
            "forecast_np_unit_multiplier": float(
                build.get("forecast_np_unit_multiplier", 10000.0)
            ),
            "forecast_metric": forecast_metric,
            "actual_metric_alignment": "user_confirmed_forecast_np_total_net_profit",
            "forecast_horizon_formula": "fy - year(available_date)",
            "scale_formula": "max(abs(consensus_pre), point_in_time_floor_by_horizon)",
            "currency": "CNY",
            "calendar_date_column": date_column,
            "calendar_start": str(trading_dates.min().date()),
            "calendar_end": str(trading_dates.max().date()),
            "actual_schema": actual_schema,
        },
        "sources": source_metadata,
        "runtime": {
            **resources.to_dict(),
            **thread_pools,
            "numba_compile_or_cache_load_seconds": numba_compile_seconds,
            "stage_metrics": stage_metrics,
            "peak_rss_mb": peak_rss_mb(),
            "total_seconds_before_metadata_write": time.perf_counter() - total_started,
        },
        "cache": cache_info,
        "counts": {
            "source_rows": cache_info.get("source_rows"),
            "canonical_report_fy_rows_read": len(all_rows),
            "reports_output": len(reports),
            "report_fy_rows_output": len(fy_output),
            "confirmation_rows_output": len(confirmation),
            "residual_valid": int(fy_output["residual_valid"].sum()),
            "edge_valid": int(fy_output["edge_valid"].sum()),
            "confirmation_valid": int(confirmation["confirmation_valid"].sum()),
            "confirmation_probe_valid": int(
                confirmation["confirmation_probe_valid"].sum()
            ),
        },
        "actual_scale_median_ratio": actual_scale_ratio,
        "notes": [
            "仅读取服务器已有文件，不联网、不访问数据库、不生成交易日历或 Actual",
            "FORECAST_NP按用户确认的总净利润口径匹配NET_PROFIT_INCL_MIN_INT_INC",
            "同一 available_date 的其他报告从发布前基准与未来更新证据中保守排除",
            "正式无效 Label 为空；invalid_reason 记录原因",
            "1%/99% winsorization 留到未来 probe 的训练集阶段执行",
        ],
    }
    _atomic_json(metadata, output_dir / "label_metadata.json")
    print("[6/6] 完成", json.dumps(metadata["counts"], ensure_ascii=False), flush=True)
    return metadata


def main() -> int:
    try:
        run(parse_args())
    except (
        ImportError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
        KeyError,
        json.JSONDecodeError,
    ) as exc:
        print(f"[error] {exc}", file=sys.stderr, flush=True)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
