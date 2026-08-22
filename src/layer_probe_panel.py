"""文本representation聚合为股票日面板并连接未来行业调整收益。"""

from __future__ import annotations

import json
import hashlib
import os
import shutil
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from .layer_probe_models import (
    assign_purged_time_splits,
    load_text_representations,
    parse_time_windows,
)
from .layer_probe_representations import (
    protocol_config_hash,
    resolve_representation_directory,
    sha256_file,
)
from .report_label_economics import (
    canonical_stock_code,
    compound_forward,
    extract_forward_values,
    read_return_panel,
    validate_decimal_returns,
)


STOCK_DAY_REPRESENTATION_FILE = "stock_day_representations.npy"
STOCK_DAY_GROUPS_FILE = "stock_day_groups.parquet"
STOCK_DAY_PANEL_FILE = "stock_day_panel.parquet"
STOCK_DAY_AUDIT_FILE = "stock_day_audit.csv"
STOCK_DAY_MANIFEST_FILE = "stock_day_manifest.json"


def _atomic_replace_directory(temporary: Path, output: Path) -> None:
    if output.exists():
        raise FileExistsError(f"股票日产物已存在，拒绝覆盖: {output}")
    try:
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


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


def aggregate_fixed_head(
    groups: pd.DataFrame,
    metadata: pd.DataFrame,
    head_outputs: pd.DataFrame,
) -> pd.DataFrame:
    """把最终层固定CLS-fc输出同步聚合到股票日。"""

    head_outputs = head_outputs[head_outputs["layer"].eq(12)].copy()

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
        ("logit_margin_1_minus_0", "fixed_head_margin"),
        ("class_1_prob", "fixed_head_class_1_probability"),
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


def _json_hash(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _evaluation_name(evaluation_split: str) -> str:
    if evaluation_split == "validation":
        return "validation"
    if evaluation_split == "test":
        return "final_test"
    raise ValueError("evaluation_split必须是validation或test")


def _validate_canonical_stock_day(root: Path) -> dict[str, object]:
    required = [
        root / STOCK_DAY_REPRESENTATION_FILE,
        root / STOCK_DAY_GROUPS_FILE,
        root / STOCK_DAY_AUDIT_FILE,
        root / STOCK_DAY_MANIFEST_FILE,
    ]
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(f"股票日canonical产物缺失: {path}")
    array = np.load(root / STOCK_DAY_REPRESENTATION_FILE, mmap_mode="r")
    groups = pd.read_parquet(root / STOCK_DAY_GROUPS_FILE)
    manifest = json.loads((root / STOCK_DAY_MANIFEST_FILE).read_text(encoding="utf-8"))
    if manifest.get("schema_version") != "stock_day_feature_store_v2.0":
        raise ValueError("股票日canonical manifest协议不匹配，拒绝混读")
    artifact_files = manifest.get("artifact_files", {})
    if not isinstance(artifact_files, Mapping):
        raise ValueError("股票日canonical manifest缺少文件哈希")
    for path in required[:-1]:
        record = artifact_files.get(path.name)
        if not isinstance(record, Mapping):
            raise ValueError(f"股票日canonical manifest缺少文件哈希: {path.name}")
        stat = path.stat()
        if stat.st_size != int(record.get("size_bytes", -1)):
            raise ValueError(f"股票日canonical文件大小已变化: {path.name}")
        if stat.st_mtime_ns != int(record.get("mtime_ns", -1)) and sha256_file(
            path
        ) != record.get("sha256"):
            raise ValueError(f"股票日canonical文件内容已变化: {path.name}")
    if tuple(array.shape) != tuple(manifest.get("shape", [])):
        raise ValueError("股票日canonical shape与manifest不一致")
    if len(groups) != array.shape[0]:
        raise ValueError("股票日groups与representation行数不一致")
    rows = pd.to_numeric(groups.get("representation_row"), errors="coerce")
    if not np.array_equal(rows.to_numpy(), np.arange(len(groups))):
        raise ValueError("股票日canonical representation_row不连续")
    if groups.duplicated(["symbol", "trading_date"]).any():
        raise ValueError("股票日groups的symbol×trading_date不唯一")
    return {
        "rows": len(groups),
        "layers": int(array.shape[1]),
        "hidden_size": int(array.shape[2]),
        "manifest": manifest,
    }


def _build_canonical_stock_day(
    config: Mapping[str, object], root: Path
) -> dict[str, object]:
    output_cfg = config["output"]
    returns_cfg = config["returns"]
    assert isinstance(output_cfg, Mapping) and isinstance(returns_cfg, Mapping)
    representations, metadata, head, rep_manifest = load_text_representations(
        resolve_representation_directory(config)
    )
    metadata = metadata.rename(columns={"feature_available_date": "trading_date"})
    canonical_contract = {
        "representation_fingerprint": rep_manifest["representation_fingerprint"],
        "storage_dtype": str(returns_cfg.get("storage_dtype", "float16")),
        "aggregation": "arithmetic mean within symbol x trading_date",
        "fixed_head_layer": 12,
    }
    if root.exists():
        check = _validate_canonical_stock_day(root)
        if check["manifest"].get("canonical_contract_sha256") != _json_hash(
            canonical_contract
        ):
            raise ValueError("已存在股票日canonical产物与当前配置不匹配，拒绝覆盖")
        return check
    root.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{root.name}-", dir=root.parent))
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
        groups = aggregate_fixed_head(groups, metadata, head)
        groups.to_parquet(
            temporary / STOCK_DAY_GROUPS_FILE, index=False, compression="zstd"
        )
        pd.DataFrame(
            [
                {"stage": "aggregation", "reason": "source_texts", "count": len(metadata)},
                {"stage": "aggregation", "reason": "canonical_stock_days", "count": len(groups)},
            ]
        ).to_csv(temporary / STOCK_DAY_AUDIT_FILE, index=False)
        artifact_files = {}
        for filename in (
            STOCK_DAY_REPRESENTATION_FILE,
            STOCK_DAY_GROUPS_FILE,
            STOCK_DAY_AUDIT_FILE,
        ):
            path = temporary / filename
            stat = path.stat()
            artifact_files[filename] = {
                "size_bytes": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
                "sha256": sha256_file(path),
            }
        manifest = {
            "schema_version": "stock_day_feature_store_v2.0",
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "shape": [len(groups), 13, int(representations.shape[2])],
            "label_free": True,
            "canonical_contract": canonical_contract,
            "canonical_contract_sha256": _json_hash(canonical_contract),
            "representation_fingerprint": rep_manifest["representation_fingerprint"],
            "representation_manifest_schema": rep_manifest["schema_version"],
            "fixed_head_contract": rep_manifest["head_contract"],
            "artifact_files": artifact_files,
        }
        (temporary / STOCK_DAY_MANIFEST_FILE).write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        _atomic_replace_directory(temporary, root)
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    return _validate_canonical_stock_day(root)


def _configured_optional_path(value: object) -> str | None:
    if not value:
        return None
    path = Path(str(value)).expanduser().resolve()
    return str(path) if path.is_file() else None


def _build_stock_day_evaluation(
    config: Mapping[str, object],
    root: Path,
    *,
    evaluation_split: str,
) -> Path:
    returns_cfg = config["returns"]
    split_cfg = config["return_time_splits"]
    exposure_cfg = config["exposures"]
    output_cfg = config["output"]
    strict_cfg = config.get("strict_test", {})
    assert all(
        isinstance(value, Mapping)
        for value in (returns_cfg, split_cfg, exposure_cfg, output_cfg, strict_cfg)
    )
    evaluation_name = _evaluation_name(evaluation_split)
    output = root / evaluation_name
    run_directory = Path(str(output_cfg.get("run_directory", ""))).expanduser().resolve()
    if evaluation_split == "test":
        if not bool(strict_cfg.get("open_final_test", False)):
            raise RuntimeError("股票日test收益未通过strict_test授权")
        if not (run_directory / "FINAL_TEST_OPENED.json").is_file():
            raise RuntimeError("股票日test收益缺少全局一次性marker")
    evaluation_contract = {
        "evaluation_split": evaluation_split,
        "returns": dict(returns_cfg),
        "return_time_splits": dict(split_cfg),
        "exposures": dict(exposure_cfg),
        "canonical_manifest_sha256": sha256_file(root / STOCK_DAY_MANIFEST_FILE),
    }
    if output.exists():
        check = validate_stock_day_artifacts(root, evaluation_split=evaluation_split)
        manifest = json.loads((output / STOCK_DAY_MANIFEST_FILE).read_text(encoding="utf-8"))
        if manifest.get("evaluation_contract_sha256") != _json_hash(evaluation_contract):
            raise ValueError("已存在股票日评价面板与当前配置不匹配，拒绝覆盖")
        if check["evaluation_split"] != evaluation_split:
            raise ValueError("股票日评价分区不匹配")
        return output

    groups = pd.read_parquet(root / STOCK_DAY_GROUPS_FILE)
    windows = parse_time_windows(split_cfg)
    allowed_windows = windows[:2] if evaluation_split == "validation" else windows
    trading_dates = pd.to_datetime(groups["trading_date"], errors="coerce").dt.normalize()
    allowed = pd.Series(False, index=groups.index)
    for window in allowed_windows:
        allowed |= trading_dates.between(window.start, window.end, inclusive="both")
    panel = groups.loc[allowed].copy()
    if panel.empty:
        raise ValueError(f"股票日{evaluation_split}允许窗口内没有样本")

    daily_path = returns_cfg.get("industry_adjusted_daily_path")
    if not daily_path:
        raise ValueError("returns.industry_adjusted_daily_path不能为空")
    daily = read_return_panel(
        daily_path,
        hdf_key=str(returns_cfg["hdf_key"]) if returns_cfg.get("hdf_key") else None,
    )
    # validation从计算图中排除test窗口及其后的收益；最终test只有在marker落盘后才走到这里。
    if evaluation_split == "validation":
        daily = daily.loc[daily.index < windows[2].start]
    horizons = [int(value) for value in returns_cfg.get("horizons", [1, 5, 20])]
    primary_horizon = int(returns_cfg.get("primary_horizon", 5))
    panel = attach_forward_returns(
        panel, daily, horizons=horizons, primary_horizon=primary_horizon
    )
    industry_path = _configured_optional_path(exposure_cfg.get("industry_path"))
    size_path = _configured_optional_path(exposure_cfg.get("size_path"))
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
    expected_splits = {"train", "validation"} if evaluation_split == "validation" else set(
        ("train", "validation", "test")
    )
    actual_splits = set(panel["split"].astype(str))
    if actual_splits != expected_splits:
        raise RuntimeError(
            "股票日评价面板缺少必需split或泄漏未授权split: "
            f"actual={sorted(actual_splits)}, expected={sorted(expected_splits)}"
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent))
    try:
        panel.to_parquet(
            temporary / STOCK_DAY_PANEL_FILE, index=False, compression="zstd"
        )
        audit = pd.concat(
            [
                pd.DataFrame(
                    [
                        {"stage": "evaluation", "reason": "canonical_stock_days", "count": len(groups)},
                        {"stage": "evaluation", "reason": "allowed_feature_windows", "count": int(allowed.sum())},
                        {"stage": "evaluation", "reason": "eligible_after_purge", "count": len(panel)},
                        {"stage": "optional_exposure", "reason": "industry_attached", "count": int(industry_path is not None)},
                        {"stage": "optional_exposure", "reason": "size_attached", "count": int(size_path is not None)},
                    ]
                ),
                split_audit,
            ],
            ignore_index=True,
        )
        audit.to_csv(temporary / STOCK_DAY_AUDIT_FILE, index=False)
        manifest = {
            "schema_version": "stock_day_return_evaluation_v2.0",
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "evaluation_split": evaluation_split,
            "config_sha256": protocol_config_hash(config),
            "included_splits": sorted(panel["split"].astype(str).unique()),
            "rows": len(panel),
            "canonical_shape": _validate_canonical_stock_day(root)["manifest"]["shape"],
            "evaluation_contract_sha256": _json_hash(evaluation_contract),
            "canonical_manifest_sha256": evaluation_contract["canonical_manifest_sha256"],
            "return_source": str(Path(str(daily_path)).expanduser().resolve()),
            "return_source_sha256": sha256_file(daily_path),
            "return_orientation": "feature date t uses close(t) to close(t+h) future return",
            "return_horizons": horizons,
            "primary_target": f"target_return_rank_{primary_horizon}d",
            "purge_label_end": f"label_end_date_{purge_horizon}d",
            "test_values_opened": evaluation_split == "test",
            "optional_exposures": {
                "industry_attached": industry_path is not None,
                "size_attached": size_path is not None,
            },
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
    validate_stock_day_artifacts(root, evaluation_split=evaluation_split)
    return output


def run_stock_day_panel_stage(
    config: Mapping[str, object], evaluation_split: str = "validation"
) -> Path:
    """生成共享股票日表示，再建立隔离的validation或final-test收益面板。"""

    output_cfg = config.get("output", {})
    returns_cfg = config.get("returns", {})
    split_cfg = config.get("return_time_splits", {})
    exposure_cfg = config.get("exposures", {})
    if not all(
        isinstance(value, Mapping)
        for value in (output_cfg, returns_cfg, split_cfg, exposure_cfg)
    ):
        raise ValueError("output/returns/return_time_splits/exposures配置必须是对象")
    run_directory = Path(str(output_cfg.get("run_directory", ""))).expanduser().resolve()
    root = run_directory / "stock_day_panel"
    _build_canonical_stock_day(config, root)
    return _build_stock_day_evaluation(
        config, root, evaluation_split=evaluation_split
    )


def validate_stock_day_artifacts(
    directory: str | Path, evaluation_split: str = "validation"
) -> dict[str, object]:
    supplied = Path(directory).expanduser().resolve()
    if supplied.name in {"validation", "final_test"}:
        inferred = "validation" if supplied.name == "validation" else "test"
        if evaluation_split != inferred and evaluation_split != "validation":
            raise ValueError("目录名与evaluation_split不一致")
        evaluation_split = inferred
        root = supplied.parent
    else:
        root = supplied
    canonical = _validate_canonical_stock_day(root)
    evaluation = root / _evaluation_name(evaluation_split)
    required = [
        evaluation / STOCK_DAY_PANEL_FILE,
        evaluation / STOCK_DAY_AUDIT_FILE,
        evaluation / STOCK_DAY_MANIFEST_FILE,
    ]
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(f"股票日评价产物缺失: {path}")
    panel = pd.read_parquet(evaluation / STOCK_DAY_PANEL_FILE)
    manifest = json.loads(
        (evaluation / STOCK_DAY_MANIFEST_FILE).read_text(encoding="utf-8")
    )
    if manifest.get("schema_version") != "stock_day_return_evaluation_v2.0":
        raise ValueError("股票日评价manifest协议不匹配，拒绝混读")
    if manifest.get("evaluation_split") != evaluation_split:
        raise ValueError("股票日评价manifest分区不匹配")
    if len(panel) != int(manifest.get("rows", -1)):
        raise ValueError("股票日评价panel行数与manifest不一致")
    positions = pd.to_numeric(panel.get("representation_row"), errors="coerce")
    if positions.isna().any() or (positions < 0).any() or (
        positions >= canonical["rows"]
    ).any():
        raise ValueError("股票日评价panel含越界representation_row")
    if panel.duplicated(["symbol", "trading_date"]).any():
        raise ValueError("股票日评价panel的symbol×trading_date不唯一")
    allowed = {"train", "validation"} if evaluation_split == "validation" else {
        "train",
        "validation",
        "test",
    }
    actual = set(panel["split"].astype(str))
    if actual != allowed:
        raise ValueError(
            "股票日评价panel缺少必需split或含未授权split: "
            f"actual={sorted(actual)}, expected={sorted(allowed)}"
        )
    return {
        "canonical_rows": canonical["rows"],
        "evaluation_rows": len(panel),
        "layers": canonical["layers"],
        "hidden_size": canonical["hidden_size"],
        "evaluation_split": evaluation_split,
        "split_counts": panel["split"].value_counts().to_dict(),
    }


def plot_stock_day_summary(
    directory: str | Path, *, evaluation_split: str = "validation"
):
    import matplotlib.pyplot as plt

    supplied = Path(directory).expanduser().resolve()
    root = supplied.parent if supplied.name in {"validation", "final_test"} else supplied
    validate_stock_day_artifacts(root, evaluation_split=evaluation_split)
    panel = pd.read_parquet(
        root / _evaluation_name(evaluation_split) / STOCK_DAY_PANEL_FILE
    )
    counts = (
        panel.groupby(["split", "trading_date"])["representation_row"]
        .size()
        .rename("stock_days")
        .reset_index()
    )
    fig, ax = plt.subplots(figsize=(10, 4))
    for split, frame in counts.groupby("split"):
        ax.plot(frame["trading_date"], frame["stock_days"], label=split)
    ax.set(title="Stock-day panel coverage", xlabel="Feature date", ylabel="Stocks")
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    return fig
