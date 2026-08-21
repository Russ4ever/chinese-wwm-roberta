"""文本representation聚合为股票日面板并连接未来行业调整收益。"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import uuid
from datetime import datetime
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from .layer_probe_models import assign_purged_time_splits, load_text_representations
from .report_label_economics import (
    canonical_stock_code,
    compound_forward,
    extract_forward_values,
    read_return_panel,
    validate_decimal_returns,
)


STOCK_DAY_REPRESENTATION_FILE = "stock_day_representations.npy"
STOCK_DAY_PANEL_FILE = "stock_day_panel.parquet"
STOCK_DAY_AUDIT_FILE = "stock_day_audit.csv"
STOCK_DAY_MANIFEST_FILE = "stock_day_manifest.json"


def _atomic_replace_directory(temporary: Path, output: Path) -> None:
    backup = output.parent / f".{output.name}.backup-{uuid.uuid4().hex}"
    moved = False
    try:
        if output.exists():
            os.replace(output, backup)
            moved = True
        try:
            os.replace(temporary, output)
        except BaseException:
            if moved and backup.exists() and not output.exists():
                os.replace(backup, output)
                moved = False
            raise
        if moved:
            shutil.rmtree(backup)
            moved = False
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
        if moved and backup.exists() and not output.exists():
            os.replace(backup, output)


def _group_codes(metadata: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    required = {"symbol", "trading_date", "representation_row"}
    missing = sorted(required.difference(metadata.columns))
    if missing:
        raise ValueError("文本metadata缺少字段: " + ", ".join(missing))
    source = metadata[["representation_row", "symbol", "trading_date"]].copy()
    source["symbol"] = canonical_stock_code(source["symbol"])
    source["trading_date"] = pd.to_datetime(
        source["trading_date"], errors="coerce"
    ).dt.normalize()
    if source[["symbol", "trading_date"]].isna().any().any():
        raise ValueError("股票日聚合键含空值")
    groups = (
        source[["trading_date", "symbol"]]
        .drop_duplicates()
        .sort_values(["trading_date", "symbol"])
        .reset_index(drop=True)
    )
    group_index = pd.MultiIndex.from_frame(groups[["trading_date", "symbol"]])
    row_index = pd.MultiIndex.from_frame(source[["trading_date", "symbol"]])
    codes = group_index.get_indexer(row_index)
    if (codes < 0).any():
        raise RuntimeError("无法建立股票日聚合编码")
    groups.insert(0, "representation_row", np.arange(len(groups), dtype=np.int64))
    return groups, codes


def aggregate_stock_day_array(
    representations: np.ndarray,
    metadata: pd.DataFrame,
    *,
    output_path: str | Path,
    storage_dtype: str = "float16",
    chunk_size: int = 16_384,
) -> tuple[pd.DataFrame, np.memmap]:
    """按symbol×trading_date均值聚合所有层，逐层写入mmap避免峰值内存。"""

    if representations.ndim != 3:
        raise ValueError("representation必须是[N,L,H]")
    if len(metadata) != representations.shape[0]:
        raise ValueError("representation与metadata行数不一致")
    if storage_dtype not in {"float16", "float32"}:
        raise ValueError("storage_dtype必须是float16或float32")
    if chunk_size <= 0:
        raise ValueError("chunk_size必须为正整数")
    groups, codes = _group_codes(metadata)
    counts = np.bincount(codes, minlength=len(groups)).astype(np.int64)
    if (counts <= 0).any():
        raise RuntimeError("生成了没有文本的股票日组")
    groups["n_texts"] = counts
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    output = np.lib.format.open_memmap(
        target,
        mode="w+",
        dtype=np.dtype(storage_dtype),
        shape=(len(groups), representations.shape[1], representations.shape[2]),
    )
    for layer in range(representations.shape[1]):
        sums = np.zeros((len(groups), representations.shape[2]), dtype=np.float32)
        for start in range(0, len(metadata), chunk_size):
            end = min(len(metadata), start + chunk_size)
            np.add.at(
                sums,
                codes[start:end],
                np.asarray(representations[start:end, layer, :], dtype=np.float32),
            )
        sums /= counts[:, None]
        output[:, layer, :] = sums.astype(storage_dtype, copy=False)
    output.flush()
    return groups, output


def aggregate_original_head(
    groups: pd.DataFrame,
    metadata: pd.DataFrame,
    head_outputs: pd.DataFrame,
) -> pd.DataFrame:
    """把原分类头logit/probability同步聚合到股票日。"""

    merged = metadata[["representation_row", "symbol", "trading_date"]].merge(
        head_outputs,
        on="representation_row",
        how="left",
        validate="one_to_one",
        suffixes=("", "_head"),
    )
    keys = groups[["trading_date", "symbol"]].copy()
    key_index = pd.MultiIndex.from_frame(keys)
    row_index = pd.MultiIndex.from_frame(merged[["trading_date", "symbol"]])
    codes = key_index.get_indexer(row_index)
    if (codes < 0).any():
        raise RuntimeError("原分类头输出无法对齐股票日组")
    output = groups.copy()
    for source, target in (
        ("logit_margin_1_minus_0", "final_sentiment_logit"),
        ("class_1_prob", "final_sentiment_probability"),
    ):
        values = pd.to_numeric(merged[source], errors="coerce").to_numpy(dtype=float)
        valid = np.isfinite(values)
        sums = np.bincount(codes[valid], weights=values[valid], minlength=len(groups))
        counts = np.bincount(codes[valid], minlength=len(groups))
        output[target] = np.divide(
            sums,
            counts,
            out=np.full(len(groups), np.nan),
            where=counts > 0,
        )
    return output


def _label_end_dates(
    feature_dates: pd.Series, return_index: pd.DatetimeIndex, horizon: int
) -> pd.Series:
    positions = return_index.get_indexer(
        pd.to_datetime(feature_dates, errors="coerce").dt.normalize()
    )
    end_positions = positions + int(horizon)
    valid = (positions >= 0) & (end_positions < len(return_index))
    output = pd.Series(pd.NaT, index=feature_dates.index, dtype="datetime64[ns]")
    output.loc[valid] = return_index.to_numpy()[end_positions[valid]]
    return output


def cross_sectional_rank(
    frame: pd.DataFrame, *, value_column: str, date_column: str = "trading_date"
) -> pd.Series:
    """逐日把未来收益变为[-0.5,0.5]百分位排名。"""

    output = pd.Series(np.nan, index=frame.index, dtype=float)
    valid = np.isfinite(pd.to_numeric(frame[value_column], errors="coerce"))
    output.loc[valid] = (
        frame.loc[valid]
        .groupby(date_column)[value_column]
        .rank(method="average", pct=True)
        - 0.5
    )
    return output


def attach_forward_returns(
    stock_day: pd.DataFrame,
    daily_industry_adjusted_returns: pd.DataFrame,
    *,
    horizons: Sequence[int] = (1, 5, 20),
    primary_horizon: int = 5,
) -> pd.DataFrame:
    """连接未来1/5/20日行业调整收益并生成5日截面排名目标。"""

    unique_horizons = sorted({int(value) for value in horizons})
    if primary_horizon not in unique_horizons:
        raise ValueError("primary_horizon必须包含在horizons中")
    validate_decimal_returns(
        daily_industry_adjusted_returns, name="industry_adjusted_return_daily"
    )
    forward = compound_forward(daily_industry_adjusted_returns, unique_horizons)
    out = stock_day.copy()
    out["symbol"] = canonical_stock_code(out["symbol"])
    out["trading_date"] = pd.to_datetime(
        out["trading_date"], errors="coerce"
    ).dt.normalize()
    for horizon in unique_horizons:
        return_column = f"industry_adjusted_return_fut{horizon}d"
        out[return_column] = extract_forward_values(
            out,
            forward[horizon],
            date_column="trading_date",
            stock_column="symbol",
        )
        out[f"label_end_date_{horizon}d"] = _label_end_dates(
            out["trading_date"], daily_industry_adjusted_returns.index, horizon
        )
    primary_return = f"industry_adjusted_return_fut{primary_horizon}d"
    out[f"target_return_rank_{primary_horizon}d"] = cross_sectional_rank(
        out, value_column=primary_return
    )
    return out


def _read_exposure_panel(
    path: str | Path, *, hdf_key: str | None = None
) -> pd.DataFrame:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"风险暴露文件不存在: {source}")
    if source.suffix.lower() in {".parquet", ".pq"}:
        frame = pd.read_parquet(source)
    elif source.suffix.lower() in {".h5", ".hdf", ".hdf5"}:
        with pd.HDFStore(source, mode="r") as store:
            keys = store.keys()
            requested = hdf_key
            if requested and not requested.startswith("/"):
                requested = "/" + requested
            if requested is None:
                if len(keys) != 1:
                    raise ValueError(f"风险暴露HDF有多个key，必须显式配置: {keys}")
                requested = keys[0]
            frame = store[requested]
    else:
        raise ValueError(f"风险暴露仅支持Parquet/HDF: {source}")
    data = frame.copy()
    date_candidates = [
        column
        for column in data.columns
        if str(column).lower() in {"date", "trade_dt", "tradedate"}
    ]
    if date_candidates:
        data.index = pd.to_datetime(data.pop(date_candidates[0]), errors="coerce")
    else:
        data.index = pd.to_datetime(data.index, errors="coerce")
    data = data.loc[data.index.notna()]
    data.index = pd.DatetimeIndex(data.index).normalize()
    normalized = canonical_stock_code(pd.Series(data.columns, dtype="string"))
    keep = normalized.notna()
    data = data.loc[:, keep.to_numpy()]
    data.columns = normalized.loc[keep].tolist()
    if data.index.duplicated().any() or data.columns.duplicated().any():
        raise ValueError("风险暴露日期或股票代码重复")
    return data.sort_index()


def _extract_exposure(
    frame: pd.DataFrame, panel: pd.DataFrame, *, output_column: str
) -> pd.Series:
    dates = pd.to_datetime(frame["trading_date"], errors="coerce").dt.normalize()
    stocks = canonical_stock_code(frame["symbol"])
    date_positions = panel.index.get_indexer(dates)
    stock_positions = panel.columns.get_indexer(stocks)
    output = np.full(len(frame), np.nan, dtype=object)
    valid = (date_positions >= 0) & (stock_positions >= 0)
    values = panel.to_numpy()
    output[valid] = values[date_positions[valid], stock_positions[valid]]
    return pd.Series(output, index=frame.index, name=output_column)


def attach_optional_exposures(
    panel: pd.DataFrame,
    *,
    industry_path: str | Path | None = None,
    size_path: str | Path | None = None,
    industry_hdf_key: str | None = None,
    size_hdf_key: str | None = None,
) -> pd.DataFrame:
    out = panel.copy()
    if industry_path:
        industry = _read_exposure_panel(industry_path, hdf_key=industry_hdf_key)
        out["industry"] = _extract_exposure(out, industry, output_column="industry")
    if size_path:
        size = _read_exposure_panel(size_path, hdf_key=size_hdf_key)
        out["size"] = pd.to_numeric(
            _extract_exposure(out, size, output_column="size"), errors="coerce"
        )
    return out


def run_stock_day_panel_stage(config: Mapping[str, object]) -> Path:
    """执行阶段3并写出股票日13层representation panel。"""

    output_cfg = config.get("output", {})
    returns_cfg = config.get("returns", {})
    split_cfg = config.get("time_splits", {})
    exposure_cfg = config.get("exposures", {})
    if not all(
        isinstance(value, Mapping)
        for value in (output_cfg, returns_cfg, split_cfg, exposure_cfg)
    ):
        raise ValueError("output/returns/time_splits/exposures配置必须是对象")
    representations, metadata, head, rep_manifest = load_text_representations(
        output_cfg.get("representations", "artifacts/layer_probe/representations")
    )
    output = (
        Path(
            str(
                output_cfg.get(
                    "stock_day_panel", "artifacts/layer_probe/stock_day_panel"
                )
            )
        )
        .expanduser()
        .resolve()
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent))
    try:
        groups, aggregate = aggregate_stock_day_array(
            representations,
            metadata,
            output_path=temporary / STOCK_DAY_REPRESENTATION_FILE,
            storage_dtype=str(returns_cfg.get("storage_dtype", "float16")),
            chunk_size=int(returns_cfg.get("aggregation_chunk_size", 16_384)),
        )
        aggregate.flush()
        del aggregate
        panel = aggregate_original_head(groups, metadata, head)
        daily_path = returns_cfg.get("industry_adjusted_daily_path")
        if not daily_path:
            raise ValueError("returns.industry_adjusted_daily_path不能为空")
        daily = read_return_panel(
            daily_path,
            hdf_key=(
                str(returns_cfg["hdf_key"]) if returns_cfg.get("hdf_key") else None
            ),
        )
        horizons = [int(value) for value in returns_cfg.get("horizons", [1, 5, 20])]
        primary_horizon = int(returns_cfg.get("primary_horizon", 5))
        panel = attach_forward_returns(
            panel,
            daily,
            horizons=horizons,
            primary_horizon=primary_horizon,
        )
        configured_industry = exposure_cfg.get("industry_path") or None
        configured_size = exposure_cfg.get("size_path") or None
        industry_path = (
            configured_industry
            if configured_industry
            and Path(str(configured_industry)).expanduser().is_file()
            else None
        )
        size_path = (
            configured_size
            if configured_size and Path(str(configured_size)).expanduser().is_file()
            else None
        )
        panel = attach_optional_exposures(
            panel,
            industry_path=industry_path,
            size_path=size_path,
            industry_hdf_key=exposure_cfg.get("industry_hdf_key") or None,
            size_hdf_key=exposure_cfg.get("size_hdf_key") or None,
        )
        purge_horizon = int(returns_cfg.get("purge_horizon", max(horizons)))
        if purge_horizon not in horizons:
            raise ValueError("purge_horizon必须包含在returns.horizons中")
        panel, split_audit = assign_purged_time_splits(
            panel,
            config=split_cfg,
            date_column="trading_date",
            label_end_column=f"label_end_date_{purge_horizon}d",
        )
        # 时间过滤会删除股票日，因此同步压缩representation并重建连续行号。
        original_rows = panel["representation_row"].to_numpy(dtype=int)
        full_array = np.load(temporary / STOCK_DAY_REPRESENTATION_FILE, mmap_mode="r")
        filtered_path = temporary / "stock_day_representations.filtered.npy"
        filtered = np.lib.format.open_memmap(
            filtered_path,
            mode="w+",
            dtype=full_array.dtype,
            shape=(len(panel), full_array.shape[1], full_array.shape[2]),
        )
        for start in range(0, len(panel), 8_192):
            end = min(len(panel), start + 8_192)
            filtered[start:end] = full_array[original_rows[start:end]]
        filtered.flush()
        del filtered, full_array
        os.replace(filtered_path, temporary / STOCK_DAY_REPRESENTATION_FILE)
        panel["representation_row"] = np.arange(len(panel), dtype=np.int64)
        panel.to_parquet(
            temporary / STOCK_DAY_PANEL_FILE, index=False, compression="zstd"
        )
        audit = pd.concat(
            [
                pd.DataFrame(
                    [
                        {
                            "stage": "aggregation",
                            "reason": "source_texts",
                            "count": len(metadata),
                        },
                        {
                            "stage": "aggregation",
                            "reason": "stock_days_before_split",
                            "count": len(groups),
                        },
                        {
                            "stage": "aggregation",
                            "reason": "stock_days_after_split",
                            "count": len(panel),
                        },
                        {
                            "stage": "optional_exposure",
                            "reason": "industry_attached",
                            "count": int(industry_path is not None),
                        },
                        {
                            "stage": "optional_exposure",
                            "reason": "size_attached",
                            "count": int(size_path is not None),
                        },
                    ]
                ),
                split_audit,
            ],
            ignore_index=True,
        )
        audit.to_csv(temporary / STOCK_DAY_AUDIT_FILE, index=False)
        manifest = {
            "schema_version": "stock_day_layer_representation_v1.0",
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "shape": [len(panel), 13, int(representations.shape[2])],
            "aggregation": "arithmetic mean within symbol x trading_date",
            "return_orientation": "feature date t uses close(t) to close(t+h) future return",
            "return_horizons": horizons,
            "primary_target": f"target_return_rank_{primary_horizon}d",
            "purge_label_end": f"label_end_date_{purge_horizon}d",
            "optional_exposures": {
                "industry_attached": industry_path is not None,
                "size_attached": size_path is not None,
            },
            "representation_manifest": rep_manifest,
        }
        (temporary / STOCK_DAY_MANIFEST_FILE).write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        _atomic_replace_directory(temporary, output)
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    validate_stock_day_artifacts(output)
    return output


def validate_stock_day_artifacts(directory: str | Path) -> dict[str, object]:
    root = Path(directory).expanduser().resolve()
    required = [
        root / STOCK_DAY_REPRESENTATION_FILE,
        root / STOCK_DAY_PANEL_FILE,
        root / STOCK_DAY_AUDIT_FILE,
        root / STOCK_DAY_MANIFEST_FILE,
    ]
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(f"股票日representation产物缺失: {path}")
    array = np.load(root / STOCK_DAY_REPRESENTATION_FILE, mmap_mode="r")
    panel = pd.read_parquet(root / STOCK_DAY_PANEL_FILE)
    manifest = json.loads((root / STOCK_DAY_MANIFEST_FILE).read_text(encoding="utf-8"))
    if tuple(array.shape) != tuple(manifest["shape"]):
        raise ValueError("股票日representation shape与manifest不一致")
    if len(panel) != array.shape[0]:
        raise ValueError("股票日panel与representation行数不一致")
    expected = pd.Series(np.arange(len(panel)), name="representation_row")
    if not panel["representation_row"].reset_index(drop=True).equals(expected):
        raise ValueError("股票日representation_row不连续")
    if not np.isfinite(array).all():
        raise ValueError("股票日representation含NaN或Inf")
    if panel.duplicated(["symbol", "trading_date"]).any():
        raise ValueError("股票日panel的symbol×trading_date不唯一")
    return {
        "rows": len(panel),
        "layers": array.shape[1],
        "hidden_size": array.shape[2],
        "split_counts": panel["split"].value_counts().to_dict(),
    }
