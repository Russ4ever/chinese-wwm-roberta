import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def test_build_labels_uses_trading_month_end_and_cutoff(tmp_path: Path):
    calendar = pd.bdate_range("2023-07-03", "2025-04-02")
    calendar_path = tmp_path / "rtn_1d.parquet"
    pd.DataFrame({"date": calendar}).to_parquet(calendar_path, index=False)

    records = []
    first_period = pd.Period("2023-07", freq="M")
    for period in pd.period_range(first_period, "2025-03", freq="M"):
        days = calendar[calendar.to_period("M") == period]
        timestamp = days[min(9, len(days) - 1)] + pd.Timedelta(hours=14)
        for org_index in range(6):
            for fy in (2024, 2025, 2026):
                records.append(
                    {
                        "ID": f"{period}-{org_index}-{fy}",
                        "STOCK_CODE": "1",
                        "STOCK_NAME": "测试股",
                        "ORGAN_NAME": f"机构{org_index}",
                        "AUTHOR_NAME": f"作者{org_index}",
                        "TITLE": "预测更新",
                        "CREATE_DATE": str(timestamp),
                        "REPORT_YEAR": fy,
                        "REPORT_QUARTER": 4,
                        "FORECAST_NP": 1000
                        + 50 * (fy - 2024)
                        + 10 * org_index
                        + 3 * (period.ordinal - first_period.ordinal),
                    }
                )
    records.append(
        {
            "ID": "late",
            "STOCK_CODE": "1",
            "STOCK_NAME": "测试股",
            "ORGAN_NAME": "机构0",
            "AUTHOR_NAME": "作者0",
            "TITLE": "盘后更新",
            "CREATE_DATE": "2024-01-31 15:00:00",
            "REPORT_YEAR": 2024,
            "REPORT_QUARTER": 4,
            "FORECAST_NP": 9999,
        }
    )
    for hour in (10, 11):
        records.append(
            {
                "STOCK_CODE": "1",
                "STOCK_NAME": "测试股",
                "ORGAN_NAME": "机构0",
                "AUTHOR_NAME": "作者0",
                "TITLE": "无编号更新",
                "CREATE_DATE": f"2024-02-15 {hour}:00:00",
                "REPORT_YEAR": 2024,
                "REPORT_QUARTER": 4,
                "FORECAST_NP": 1200,
            }
        )
    source = tmp_path / "reports.jsonl"
    source.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )

    output = tmp_path / "output"
    env = os.environ.copy()
    env.update(
        LABEL_SOURCE=str(source),
        LABEL_TRADING_CALENDAR=str(calendar_path),
        LABEL_OUTPUT_DIR=str(output),
    )
    subprocess.run(
        [sys.executable, str(ROOT / "label_engineering" / "build_labels.py")],
        cwd=ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )

    snapshot = pd.read_parquet(output / "v2" / "consensus_snapshot_monthly_v2.parquet")
    actual_month_ends = pd.DatetimeIndex(snapshot["asof_month"].drop_duplicates().sort_values())
    assert pd.Timestamp("2024-03-29") in actual_month_ends
    assert pd.Timestamp("2024-03-31") not in actual_month_ends

    report = pd.read_parquet(output / "v2" / "report_label_detail_v2.parquet")
    late = report.loc[report["source_report_id"] == "late"].iloc[0]
    assert late["publish_date"] == pd.Timestamp("2024-01-31")
    assert late["available_date"] == pd.Timestamp("2024-02-01")

    clean = pd.read_parquet(output / "v2" / "clean_report_v2.parquet")
    fallback_ids = clean.loc[clean["title"] == "无编号更新", "source_report_id"]
    assert len(fallback_ids) == 2
    assert fallback_ids.nunique() == 2
