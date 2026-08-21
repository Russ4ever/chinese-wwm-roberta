#!/usr/bin/env python
"""把研报文本与 Residual/Dispersion Label 装配为 Linear Probe bundle。"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import load_yaml_config  # noqa: E402
from src.probe_dataset import build_probe_bundle  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="构建无前视的逐层Probe数据")
    parser.add_argument(
        "--config", default=str(ROOT / "configs" / "probe_dataset.yaml")
    )
    parser.add_argument("--output-dir", default=None, help="覆盖配置中的输出目录")
    return parser.parse_args()


def _path(value: object) -> Path | None:
    text = str(value or "").strip()
    if not text:
        return None
    path = Path(text).expanduser()
    return path if path.is_absolute() else ROOT / path


def _required_path(value: object, name: str) -> Path:
    path = _path(value)
    if path is None:
        raise ValueError(f"{name}路径不能为空")
    path = path.resolve()
    if not path.exists():
        raise FileNotFoundError(f"{name}不存在: {path}")
    return path


def _blake3_file(path: Path, *, threads: int) -> str:
    from blake3 import blake3

    digest = blake3(max_threads=max(1, int(threads)))
    digest.update_mmap(str(path))
    return digest.hexdigest()


def _file_meta(path: Path, *, threads: int) -> dict[str, object]:
    stat = path.stat()
    return {
        "path": str(path),
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "blake3": _blake3_file(path, threads=threads),
    }


def _safe_output(path: Path) -> None:
    if path == Path(path.anchor) or len(path.parts) < 3:
        raise ValueError(f"拒绝覆盖过宽输出目录: {path}")


def _validate_label_metadata(
    path: Path,
    *,
    reports: pd.DataFrame,
    fy_labels: pd.DataFrame,
    confirmation: pd.DataFrame,
    label_version: str,
) -> dict[str, object]:
    metadata = json.loads(path.read_text(encoding="utf-8"))
    source_version = str(metadata.get("label_version", "")).strip()
    if source_version != label_version:
        raise ValueError(
            "label_metadata.json与Parquet的label_version不一致: "
            f"metadata={source_version!r}, parquet={label_version!r}"
        )
    counts = metadata.get("counts", {})
    if not isinstance(counts, dict):
        raise ValueError("label_metadata.json.counts必须是对象")
    expected = {
        "reports_output": len(reports),
        "report_fy_rows_output": len(fy_labels),
        "confirmation_rows_output": len(confirmation),
    }
    for key, actual in expected.items():
        if key in counts and int(counts[key]) != actual:
            raise ValueError(
                f"label_metadata.json记录的{key}={counts[key]}，"
                f"但实际Parquet为{actual}行"
            )
    return metadata


def _write_bundle(
    *,
    texts: pd.DataFrame,
    targets: pd.DataFrame,
    audit: pd.DataFrame,
    metadata: dict[str, Any],
    output_dir: Path,
) -> None:
    _safe_output(output_dir)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}-", dir=output_dir.parent)
    )
    backup = output_dir.parent / f".{output_dir.name}.backup-{uuid.uuid4().hex}"
    moved_existing = False
    try:
        texts.to_parquet(
            temporary / "probe_texts.parquet",
            index=False,
            compression="zstd",
        )
        targets.to_parquet(
            temporary / "probe_targets.parquet",
            index=False,
            compression="zstd",
        )
        audit.to_csv(temporary / "probe_merge_audit.csv", index=False)
        (temporary / "probe_dataset_metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        if output_dir.exists():
            os.replace(output_dir, backup)
            moved_existing = True
        try:
            os.replace(temporary, output_dir)
        except BaseException:
            if moved_existing and backup.exists() and not output_dir.exists():
                os.replace(backup, output_dir)
                moved_existing = False
            raise
        if moved_existing:
            shutil.rmtree(backup)
            moved_existing = False
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
        if moved_existing and backup.exists() and not output_dir.exists():
            os.replace(backup, output_dir)


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    config_path = _required_path(args.config, "配置")
    config = load_yaml_config(config_path)

    from src.report_label_runtime import peak_rss_mb, resolve_runtime_resources

    # 这里只做 Parquet 连接与校验，不启动 Label 构造所需的 Numba/Polars 内核。
    # 因此只解析资源上限，不改写调用方进程的线程池或环境变量。
    resources = resolve_runtime_resources(config.get("performance", {}))
    paths = config.get("paths", {})
    label_dir = _required_path(paths.get("report_labels"), "report_labels")
    if not label_dir.is_dir():
        raise ValueError(f"report_labels必须是目录: {label_dir}")
    reports_path = label_dir / "reports.parquet"
    fy_path = label_dir / "report_fy_labels.parquet"
    confirmation_path = label_dir / "report_confirmation_labels.parquet"
    label_metadata_path = label_dir / "label_metadata.json"
    for path in (reports_path, fy_path, confirmation_path, label_metadata_path):
        if not path.is_file():
            raise FileNotFoundError(f"report_labels缺少产物: {path}")

    reports = pd.read_parquet(reports_path)
    fy_labels = pd.read_parquet(fy_path)
    confirmation = pd.read_parquet(confirmation_path)
    bundle = build_probe_bundle(
        reports,
        fy_labels,
        confirmation,
        target_config=config.get("targets", {}),
        split_config=config.get("splits", {}),
    )
    source_label_metadata = _validate_label_metadata(
        label_metadata_path,
        reports=reports,
        fy_labels=fy_labels,
        confirmation=confirmation,
        label_version=bundle.metadata["label_version"],
    )

    output_dir = _path(args.output_dir or config.get("output", {}).get("directory"))
    if output_dir is None:
        raise ValueError("output.directory不能为空")
    output_dir = output_dir.resolve()
    source_paths = [
        config_path,
        reports_path,
        fy_path,
        confirmation_path,
        label_metadata_path,
    ]
    bundle.metadata.update(
        {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "runtime": {
                **resources.to_dict(),
                "peak_rss_mb": peak_rss_mb(),
                "seconds_before_write": time.perf_counter() - started,
            },
            "sources": [
                _file_meta(path, threads=resources.effective_threads)
                for path in source_paths
            ],
            "output_directory": str(output_dir),
            "upstream_label_manifest": {
                "label_version": source_label_metadata["label_version"],
                "created_at": source_label_metadata.get("created_at"),
                "counts": source_label_metadata.get("counts", {}),
            },
        }
    )
    _write_bundle(
        texts=bundle.texts,
        targets=bundle.targets,
        audit=bundle.audit,
        metadata=bundle.metadata,
        output_dir=output_dir,
    )
    print(json.dumps(bundle.metadata["counts"], ensure_ascii=False), flush=True)
    if not bundle.metadata["training_ready"]:
        print(
            "[warn] 当前没有任务具备完整且有效的train/validation/test；"
            "已完成连接审计，但加载器会禁止直接训练",
            file=sys.stderr,
            flush=True,
        )
    return bundle.metadata


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
