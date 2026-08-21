#!/usr/bin/env python
"""在历史时点沪深300股票池内验证单篇研报 Label 的经济意义。"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import load_yaml_config  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="单篇研报Label经济意义验证")
    parser.add_argument("--config", default=str(ROOT / "configs" / "report_label_economic_validation.yaml"))
    parser.add_argument("--years", default=None, help="覆盖配置，例如2023,2024")
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def _path(value: object) -> Path | None:
    text = str(value or "").strip()
    if not text:
        return None
    result = Path(text).expanduser()
    return result.resolve() if result.is_absolute() else (ROOT / result).resolve()


def _required_path(value: object, name: str) -> Path:
    path = _path(value)
    if path is None:
        raise ValueError(f"{name}未配置")
    if not path.exists():
        raise FileNotFoundError(f"{name}不存在: {path}")
    return path


def _read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if path.suffix.lower() in {".csv", ".txt"}:
        return pd.read_csv(path)
    raise ValueError(f"不支持的表格式: {path}")


def _history_files(path: Path, start_year: int, end_year: int) -> list[Path]:
    if path.is_file():
        return [path]
    parts: list[Path] = []
    missing: list[int] = []
    for year in range(start_year, end_year + 1):
        candidate = path / f"year={year}" / "report_fy.parquet"
        if candidate.is_file():
            parts.append(candidate)
        else:
            missing.append(year)
    if missing:
        raise FileNotFoundError(
            f"报告历史缓存缺少年份分区{missing}: {path}"
        )
    return parts


def _history_from_cache(files: list[Path]) -> pd.DataFrame:
    return pd.concat([pd.read_parquet(part) for part in files], ignore_index=True)


def _resolve_history_cache(configured: Path | None, label_dir: Path) -> Path:
    if configured is not None:
        return configured
    metadata_path = label_dir / "label_metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError("未配置report_history_cache，且Label目录没有label_metadata.json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    candidate = _path(metadata.get("cache", {}).get("directory"))
    if candidate is None or not candidate.exists():
        raise FileNotFoundError("Label元数据没有可用的精确报告历史缓存；请配置paths.report_history_cache")
    return candidate


def _truncate_by_coverage_end(
    relevant_actuals: pd.DataFrame,
    coverage_end: pd.Timestamp,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """将 actuals 截断到 coverage_end 之前，以便只验证具有完整研报窗口的记录。

    当部分 Actual 的披露日晚于研报历史覆盖截止日时（例如 2023/2024 财年年报延至
    次年中才披露），我们无法在其披露前构建完整的共识窗口。该函数剔除这些"超期"记录，
    并返回被丢弃的行数供日志审计。
    """
    known_dates = pd.to_datetime(relevant_actuals["actual_known_date"], errors="coerce")
    valid = known_dates.le(coverage_end.normalize())
    total_rows = len(relevant_actuals)
    kept_rows = int(valid.sum())
    dropped_rows = total_rows - kept_rows
    # Build a per-FY breakdown for the log
    fy_col = "fy"
    by_fy: dict[str, int] = {}
    if fy_col in relevant_actuals.columns:
        grouped = relevant_actuals[fy_col].dropna().astype(str)
        valid_grouped = grouped[valid.to_numpy()]
        invalid_grouped = grouped[~valid.to_numpy()]
        for fy_val in grouped.unique():
            total_fy = int((grouped == fy_val).sum())
            kept_fy = int((valid_grouped == fy_val).sum())
            by_fy[str(fy_val)] = {
                "total": total_fy,
                "kept": kept_fy,
                "dropped": total_fy - kept_fy,
            }
    result = relevant_actuals.loc[valid.to_numpy()].copy()
    return result, {"total": total_rows, "kept": kept_rows, "dropped": dropped_rows, "by_fy": by_fy}


def _validate_history_coverage(
    label_metadata: dict[str, Any],
    relevant_actuals: pd.DataFrame,
    *,
    lookback_days: int,
) -> tuple[pd.Timestamp, pd.Timestamp]:
    configuration = label_metadata.get("configuration", {})
    coverage_start = pd.to_datetime(
        configuration.get("coverage_start"), errors="coerce"
    )
    coverage_end = pd.to_datetime(
        configuration.get("coverage_end"), errors="coerce"
    )
    if pd.isna(coverage_start) or pd.isna(coverage_end):
        raise ValueError(
            "label_metadata.json缺少可验证的研报源覆盖边界"
        )
    known = pd.to_datetime(
        relevant_actuals["actual_known_date"], errors="coerce"
    ).dropna()
    if known.empty:
        raise ValueError("所选年份没有有效Actual披露日")
    required_start = known.min().normalize() - pd.Timedelta(days=lookback_days)
    required_end = known.max().normalize()
    if coverage_start.normalize() > required_start or coverage_end.normalize() < required_end:
        raise ValueError(
            "研报历史覆盖不足以构造Actual披露前完整共识窗口: "
            f"要求{required_start.date()}~{required_end.date()}，"
            f"实际{coverage_start.date()}~{coverage_end.date()}"
        )
    return coverage_start.normalize(), coverage_end.normalize()


def _map_on_or_after(values: pd.Series, calendar: pd.DatetimeIndex) -> pd.Series:
    dates = pd.to_datetime(values, errors="coerce").dt.normalize()
    positions = np.searchsorted(calendar.to_numpy(dtype="datetime64[ns]"), dates.to_numpy(dtype="datetime64[ns]"), side="left")
    result = pd.Series(pd.NaT, index=values.index, dtype="datetime64[ns]")
    valid = dates.notna().to_numpy() & (positions < len(calendar))
    result.loc[valid] = calendar.to_numpy()[positions[valid]]
    return result


def _extract_series(frame: pd.DataFrame, series: pd.Series, date_column: str) -> np.ndarray:
    dates = pd.to_datetime(frame[date_column], errors="coerce").dt.normalize()
    return series.reindex(dates).to_numpy(dtype=float)


def _compound_index_between(frame: pd.DataFrame, index_daily: pd.Series, start: str, end: str) -> np.ndarray:
    dates = index_daily.index
    starts = pd.to_datetime(frame[start], errors="coerce").dt.normalize()
    ends = pd.to_datetime(frame[end], errors="coerce").dt.normalize()
    start_pos = dates.get_indexer(starts)
    end_pos = dates.get_indexer(ends)
    values = index_daily.to_numpy(dtype=float)
    output = np.full(len(frame), np.nan)
    for i in np.flatnonzero((start_pos >= 0) & (end_pos >= start_pos)):
        segment = values[start_pos[i] : end_pos[i]]
        if len(segment) and np.isfinite(segment).all():
            output[i] = float(np.prod(1.0 + segment) - 1.0)
        elif len(segment) == 0:
            output[i] = 0.0
    return output


def _file_meta(path: Path, *, threads: int) -> dict[str, object]:
    from blake3 import blake3

    stat = path.stat()
    digest = blake3(max_threads=threads)
    digest.update_mmap(str(path))
    return {"path": str(path), "size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns, "blake3": digest.hexdigest()}


def _write_outputs(frames: dict[str, pd.DataFrame], metadata: dict[str, Any], output_dir: Path) -> None:
    if output_dir == Path(output_dir.anchor) or len(output_dir.parts) < 3:
        raise ValueError(f"拒绝覆盖过宽输出目录: {output_dir}")
    parent = output_dir.parent
    parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}-", dir=parent))
    try:
        for filename, frame in frames.items():
            path = temporary / filename
            if filename.endswith(".parquet"):
                frame.to_parquet(path, index=False, compression="zstd")
            else:
                frame.to_csv(path, index=False)
        (temporary / "validation_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        figures = temporary / "figures"
        figures.mkdir()
        groups = frames.get("group_monotonicity.csv", pd.DataFrame())
        if not groups.empty:
            import matplotlib.pyplot as plt

            for analysis, group in groups.groupby("analysis"):
                fig, ax = plt.subplots(figsize=(6, 4))
                ax.plot(group["quantile"], group["y_mean"], marker="o")
                ax.axhline(0, color="gray", linewidth=0.8)
                ax.set_title(str(analysis))
                ax.set_xlabel("label quantile")
                ax.set_ylabel("mean outcome")
                fig.tight_layout()
                safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in str(analysis))
                fig.savefig(figures / f"{safe}.png", dpi=140)
                plt.close(fig)
        if output_dir.exists():
            shutil.rmtree(output_dir)
        os.replace(temporary, output_dir)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    config_path = _required_path(args.config, "配置")
    config = load_yaml_config(config_path)
    from src.report_label_runtime import configure_runtime, peak_rss_mb

    resources = configure_runtime(config.get("performance", {}))
    from label_engineering.build_report_labels import load_actuals
    from src.report_label_economics import (
        association_summary,
        build_actual_surprise_samples,
        build_analyst_information_samples,
        build_consensus_revision_samples,
        compound_between_events,
        compound_forward,
        consistency_audit,
        extract_forward_values,
        index_returns_from_prices,
        lock_annual_universes,
        normalize_csi_weights,
        quantile_summary,
        read_return_panel,
        validate_decimal_returns,
        weighted_index_returns,
    )
    from src.trading_calendar import load_trading_dates

    paths = config.get("paths", {})
    validation = config.get("validation", {})
    years = [int(x) for x in (args.years.split(",") if args.years else validation.get("years", [2024]))]
    if not years:
        raise ValueError("validation.years不能为空")
    index_code = str(validation.get("index_code", "000300.SH"))
    windows = [int(x) for x in validation.get("return_windows", [1, 5, 10, 20])]
    event_window = int(validation.get("actual_event_window", 10))
    if event_window not in windows:
        windows.append(event_window)

    label_dir = _required_path(paths.get("report_labels"), "report_labels")
    actual_path = _required_path(paths.get("actuals"), "actuals")
    calendar_path = _required_path(paths.get("trading_calendar"), "trading_calendar")
    rtn1_path = _required_path(paths.get("rtn_1d"), "rtn_1d")
    rtn20_path = _required_path(paths.get("rtn_20d"), "rtn_20d")
    spret_daily_path = _required_path(paths.get("specific_return_daily"), "specific_return_daily")
    spret10_path = _required_path(paths.get("specific_return_fut10"), "specific_return_fut10")
    weights_path = _required_path(paths.get("csi300_weights"), "csi300_weights")
    prices_path = _path(paths.get("csi300_index_prices"))

    fy_path = label_dir / "report_fy_labels.parquet"
    confirmation_path = label_dir / "report_confirmation_labels.parquet"
    label_metadata_path = label_dir / "label_metadata.json"
    if (
        not fy_path.is_file()
        or not confirmation_path.is_file()
        or not label_metadata_path.is_file()
    ):
        raise FileNotFoundError(
            "report_labels目录缺少FY、Confirmation或metadata产物"
        )
    fy_labels = pd.read_parquet(fy_path)
    confirmation = pd.read_parquet(confirmation_path)
    label_metadata = json.loads(label_metadata_path.read_text(encoding="utf-8"))
    actual_schema_path = _path(paths.get("actual_schema"))
    actuals, _, _ = load_actuals(actual_path, actual_schema_path, expected_metric="net_profit_incl_min_int_inc")
    relevant_actuals = actuals[pd.to_numeric(actuals["fy"], errors="coerce").isin(years)]

    # Truncate actuals to coverage_end so that we only validate records with a
    # complete pre-disclosure consensus window (lookback_days of reports before each
    # actual_known_date). Recent fiscal years often have disclosures after the report
    # source cutoff, making their labels unverifiable.
    metadata_cfg = label_metadata.get("configuration", {})
    _coverage_end_raw = pd.to_datetime(
        metadata_cfg.get("coverage_end"), errors="coerce"
    )
    if pd.isna(_coverage_end_raw):
        raise ValueError("metadata 中缺少可解析的 coverage_end")
    _coverage_end = _coverage_end_raw.normalize()
    truncated_actuals, truncation_stats = _truncate_by_coverage_end(relevant_actuals, _coverage_end)
    relevant_actuals = truncated_actuals
    print("[info] actuals truncated by coverage_end={} stats={}".format(
        _coverage_end.date(), truncation_stats), flush=True)

    lookback_days = int(validation.get("lookback_days", 180))
    coverage_start, coverage_end = _validate_history_coverage(
        label_metadata,
        relevant_actuals,
        lookback_days=lookback_days,
    )
    known_years = (
        pd.to_datetime(relevant_actuals["actual_known_date"], errors="coerce")
        .dt.year.dropna().astype(int)
    )
    earliest_known = pd.to_datetime(
        relevant_actuals["actual_known_date"], errors="coerce"
    ).dropna().min().normalize()
    min_history_year = int(
        (earliest_known - pd.Timedelta(days=lookback_days)).year
    )
    max_history_year = max(int(known_years.max()) if not known_years.empty else max(years), max(years))
    history_path = _resolve_history_cache(_path(paths.get("report_history_cache")), label_dir)
    history_files = _history_files(history_path, min_history_year, int(max_history_year))
    history = _history_from_cache(history_files)

    calendar_cfg = config.get("calendar", {})
    trading_dates = load_trading_dates(calendar_path, date_column=str(calendar_cfg.get("date_column", "date")), date_format=calendar_cfg.get("date_format"))
    rtn1 = read_return_panel(rtn1_path)
    rtn20 = read_return_panel(rtn20_path)
    spret_daily = read_return_panel(spret_daily_path, hdf_key=validation.get("specific_return_hdf_key"))
    spret10 = read_return_panel(spret10_path)
    for name, panel in (("rtn_1d", rtn1), ("rtn_20d", rtn20), ("specific_return_daily", spret_daily), ("specific_return_fut10", spret10)):
        validate_decimal_returns(panel, name=name)
    raw_forward = compound_forward(rtn1, windows)
    spret_forward = compound_forward(spret_daily, [10])
    atol = float(validation.get("consistency_atol", 1e-8))
    rtol = float(validation.get("consistency_rtol", 1e-6))
    raw_audit = consistency_audit(raw_forward[20], rtn20, atol=atol, rtol=rtol)
    spret_audit = consistency_audit(spret_forward[10], spret10, atol=atol, rtol=rtol)
    min_fraction = float(validation.get("consistency_min_fraction", 0.99))
    if raw_audit["fraction_close"] < min_fraction or spret_audit["fraction_close"] < min_fraction:
        raise ValueError(f"收益复合口径不一致: raw={raw_audit}, specific={spret_audit}")

    weights = normalize_csi_weights(_read_table(weights_path), index_code=index_code)
    locked_frame, universes = lock_annual_universes(weights, years)
    if prices_path is not None and prices_path.is_file():
        index_daily = index_returns_from_prices(_read_table(prices_path), index_code=index_code)
        index_source = "wind_index_close"
        index_coverage = pd.DataFrame(index_daily)
    else:
        index_coverage = weighted_index_returns(weights, rtn1, min_weight_coverage=float(validation.get("weight_coverage_min", 0.95)))
        index_daily = index_coverage["csi300_return_1d"]
        index_source = "historical_weighted_rtn_1d"
    index_forward = {h: compound_forward(index_daily.to_frame("000300"), [h])[h]["000300"] for h in windows}

    actual_samples = build_actual_surprise_samples(actuals, history, universes, lookback_days=lookback_days, min_peer_orgs=int(validation.get("min_peer_orgs", 4)), floor_ratio=float(validation.get("floor_ratio", 0.01)))
    actual_samples["event_date"] = _map_on_or_after(actual_samples["actual_known_date"], trading_dates)
    for horizon in windows:
        actual_samples[f"return_fut{horizon}"] = extract_forward_values(actual_samples, raw_forward[horizon], date_column="event_date")
        actual_samples[f"index_return_fut{horizon}"] = _extract_series(actual_samples, index_forward[horizon], "event_date")
        actual_samples[f"excess_return_fut{horizon}"] = actual_samples[f"return_fut{horizon}"] - actual_samples[f"index_return_fut{horizon}"]
    actual_samples["specific_return_fut10"] = extract_forward_values(actual_samples, spret10, date_column="event_date")

    analyst = build_analyst_information_samples(fy_labels, universes)
    analyst["cluster_stock_date"] = analyst["stock_code"].astype(str) + "|" + analyst["available_date"].astype(str)

    consensus = build_consensus_revision_samples(confirmation, fy_labels, universes)
    consensus["cluster_stock_date"] = consensus["stock_code"].astype(str) + "|" + consensus["available_date"].astype(str)
    consensus["return_event_window"] = compound_between_events(consensus, rtn1, start_column="available_date", end_column="target_date")
    consensus["specific_return_event_window"] = compound_between_events(consensus, spret_daily, start_column="available_date", end_column="target_date")
    consensus["index_return_event_window"] = _compound_index_between(consensus, index_daily, "available_date", "target_date")
    consensus["excess_return_event_window"] = consensus["return_event_window"] - consensus["index_return_event_window"]
    consensus["aligned_specific_return"] = consensus["report_direction"] * consensus["specific_return_event_window"]
    consensus["zero_type"] = np.select(
        [consensus["progress_raw"].eq(0) & consensus["n_peer_updates"].eq(0), consensus["progress_raw"].eq(0) & consensus["n_peer_updates"].eq(1), consensus["progress_raw"].eq(0) & consensus["n_peer_updates"].ge(2)],
        ["no_update", "one_update_no_move", "sufficient_updates_no_move"],
        default="nonzero",
    )

    summary_records: list[dict[str, object]] = []
    group_frames: list[pd.DataFrame] = []
    actual_valid = actual_samples[actual_samples["actual_surprise_valid"].eq(1)].copy()
    for outcome in ("specific_return_fut10", f"excess_return_fut{event_window}", f"return_fut{event_window}"):
        summary_records.append(association_summary(actual_valid, analysis=f"fundamental_pricing_{outcome}", x_column="actual_surprise", y_column=outcome, cluster_column="stock_code"))
    group_frames.append(quantile_summary(actual_valid, analysis="fundamental_pricing_specific_fut10", x_column="actual_surprise", y_column="specific_return_fut10", quantiles=int(validation.get("quantiles", 5))))

    for horizon, group in analyst.groupby("forecast_horizon"):
        analysis = f"analyst_information_fy{int(horizon)}"
        summary_records.append(association_summary(group, analysis=analysis, x_column="report_signal", y_column="realized_fundamental_surprise", cluster_column="cluster_stock_date"))
        group_frames.append(quantile_summary(group, analysis=analysis, x_column="report_signal", y_column="realized_fundamental_surprise", quantiles=int(validation.get("quantiles", 5))))

    for (months, panel, horizon), group in consensus.groupby(["confirmation_months", "peer_panel", "forecast_horizon"]):
        prefix = f"consensus_{int(months)}m_{panel}_fy{int(horizon)}"
        summary_records.append(association_summary(group, analysis=prefix + "_price", x_column="consensus_revision", y_column="specific_return_event_window", cluster_column="cluster_stock_date"))
        actual_group = group.dropna(subset=["realized_fundamental_surprise"])
        summary_records.append(association_summary(actual_group, analysis=prefix + "_actual", x_column="consensus_revision", y_column="realized_fundamental_surprise", cluster_column="cluster_stock_date"))
        summary_records.append(association_summary(group, analysis=prefix + "_progress", x_column="progress_clipped", y_column="aligned_specific_return", cluster_column="cluster_stock_date"))
        if panel == "fixed":
            group_frames.append(quantile_summary(group, analysis=prefix + "_price", x_column="consensus_revision", y_column="specific_return_event_window", quantiles=int(validation.get("quantiles", 5))))

    summary = pd.DataFrame(summary_records)
    groups = pd.concat(group_frames, ignore_index=True) if group_frames else pd.DataFrame()
    analyst_metrics = analyst.groupby("forecast_horizon", as_index=False).agg(
        rows=("report_id", "size"),
        direction_hit_rate=("direction_hit", "mean"),
        edge_positive_rate=("edge_raw", lambda x: float((x > 0).mean())),
        mean_report_abs_error=("report_abs_error", "mean"),
        mean_consensus_abs_error=("consensus_abs_error", "mean"),
    )
    zero_audit = consensus.groupby(["confirmation_months", "peer_panel", "forecast_horizon", "zero_type"], as_index=False).agg(rows=("report_id", "size"), mean_specific_return=("specific_return_event_window", "mean"), mean_aligned_specific_return=("aligned_specific_return", "mean"))
    coverage = pd.DataFrame([
        {"stage": "locked_universe", "rows": len(locked_frame), "valid": len(locked_frame), "rate": 1.0},
        {"stage": "actual_surprise", "rows": len(actual_samples), "valid": int(actual_samples["actual_surprise_valid"].sum()), "rate": float(actual_samples["actual_surprise_valid"].mean()) if len(actual_samples) else np.nan},
        {"stage": "analyst_information", "rows": len(fy_labels), "valid": len(analyst), "rate": len(analyst) / len(fy_labels) if len(fy_labels) else np.nan},
        {"stage": "consensus_revision", "rows": len(confirmation), "valid": len(consensus), "rate": len(consensus) / len(confirmation) if len(confirmation) else np.nan},
    ])

    output_dir = _path(args.output_dir or config.get("output", {}).get("directory"))
    if output_dir is None:
        raise ValueError("output.directory不能为空")
    source_paths = [config_path, actual_path, calendar_path, rtn1_path, rtn20_path, spret_daily_path, spret10_path, weights_path, fy_path, confirmation_path, label_metadata_path, *history_files]
    if actual_schema_path is not None and actual_schema_path.is_file():
        source_paths.append(actual_schema_path)
    if prices_path is not None and prices_path.is_file():
        source_paths.append(prices_path)
    metadata = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "validation_years": years,
        "point_in_time_rules": {
            "universe": "latest CSI300 weight snapshot on or before prior calendar year end, frozen for validation year",
            "actual_consensus": "report available_date strictly before actual_known_date",
            "actual_scale": "per-event floor from only the latest peer forecasts known before actual disclosure",
            "report_history_coverage": {
                "start": str(coverage_start.date()),
                "end": str(coverage_end.date()),
                "required_through_actual_known_date": True,
            },
            "return": "close(t) to close(t+1); event returns begin at event date close",
            "same_day_forecasts_excluded": True,
        },
        "index_return_source": index_source,
        "return_consistency": {"raw_20d": raw_audit, "specific_10d": spret_audit},
        "runtime": {**resources.to_dict(), "peak_rss_mb": peak_rss_mb(), "seconds": time.perf_counter() - started},
        "sources": [
            _file_meta(path, threads=resources.effective_threads)
            for path in dict.fromkeys(source_paths)
        ],
        "counts": {"locked_universe_rows": len(locked_frame), "actual_samples": len(actual_samples), "analyst_samples": len(analyst), "consensus_samples": len(consensus)},
    }
    _write_outputs(
        {
            "csi300_locked_universe.parquet": locked_frame,
            "fundamental_pricing_samples.parquet": actual_samples,
            "analyst_information_samples.parquet": analyst,
            "consensus_revision_samples.parquet": consensus,
            "validation_summary.csv": summary,
            "group_monotonicity.csv": groups,
            "analyst_information_summary.csv": analyst_metrics,
            "zero_value_audit.csv": zero_audit,
            "coverage_audit.csv": coverage,
            "index_return_audit.parquet": index_coverage.reset_index(),
        },
        metadata,
        output_dir,
    )
    print(json.dumps(metadata["counts"], ensure_ascii=False), flush=True)
    return metadata


def main() -> int:
    try:
        run(parse_args())
    except (ImportError, KeyError, OSError, RuntimeError, TypeError, ValueError, json.JSONDecodeError) as exc:
        print(f"[error] {exc}", file=sys.stderr, flush=True)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
