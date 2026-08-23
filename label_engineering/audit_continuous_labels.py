#!/usr/bin/env python
"""CLI for the outcome-blind continuous-label coverage audit."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.continuous_label_audit import (  # noqa: E402
    DEFAULT_MATURITY_MONTHS,
    run_continuous_label_audit,
)


def _maturity_months(value: str) -> tuple[int, ...]:
    try:
        months = tuple(sorted({int(item.strip()) for item in value.split(",")}))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("成熟月数必须是逗号分隔的整数") from exc
    if not months or months[0] <= 0:
        raise argparse.ArgumentTypeError("成熟月数必须是正整数")
    return months


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="只读审计Residual/Dispersion覆盖、成熟期和候选窗口"
    )
    parser.add_argument(
        "--config",
        default=str(ROOT / "configs" / "probe_dataset.yaml"),
        help="现有probe dataset配置；不会修改该文件",
    )
    parser.add_argument(
        "--label-dir",
        default=None,
        help="可选：覆盖config.paths.report_labels，仅用于读取",
    )
    parser.add_argument(
        "--output-root",
        default=str(ROOT / "audit_reports" / "continuous_label_audit"),
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="独立输出子目录；默认使用UTC时间戳",
    )
    parser.add_argument(
        "--maturity-months",
        type=_maturity_months,
        default=DEFAULT_MATURITY_MONTHS,
        help="候选成熟期，默认1,3,6,9,12,18,24,30,36,42,48",
    )
    return parser.parse_args()


def run(args: argparse.Namespace) -> dict[str, object]:
    run_id = str(args.run_id or datetime.utcnow().strftime("%Y%m%dT%H%M%SZ"))
    if not run_id.strip() or run_id in {".", ".."} or "/" in run_id:
        raise ValueError("run-id必须是单个非空目录名")
    output = Path(args.output_root).expanduser() / run_id
    return run_continuous_label_audit(
        config_path=args.config,
        repository_root=ROOT,
        output_directory=output,
        label_directory=args.label_dir,
        maturity_months=args.maturity_months,
    )


def main() -> int:
    manifest = run(parse_args())
    print(
        json.dumps(
            {
                "status": "ok",
                "output_directory": manifest["output_directory"],
                "source_counts": manifest["source_counts"],
                "outcome_blind": manifest["outcome_blind"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
