import json
from pathlib import Path

import pandas as pd

from scripts.infer_research_report import read_jsonl


def test_research_metadata_applies_cutoff_and_nontrading_alignment(tmp_path: Path):
    source = tmp_path / "reports.jsonl"
    records = [
        {"ID": "", "TITLE": "A", "CONTENT": "", "CREATE_DATE": "2024-01-05 14:57:00"},
        {"ID": 2, "TITLE": "B", "CONTENT": "正文", "CREATE_DATE": "2024-01-05 14:57:01"},
        {"ID": 3, "TITLE": "C", "CONTENT": "", "CREATE_DATE": "2024-01-06 09:00:00"},
    ]
    source.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records) + "[]\n",
        encoding="utf-8",
    )
    calendar = pd.to_datetime(["2024-01-05", "2024-01-08"])

    texts, metadata = read_jsonl(source, calendar)

    assert texts == ["A", "B。正文", "C"]
    assert metadata["id"].tolist() == ["line:1", "2", "3"]
    assert metadata["available_date"].tolist() == [
        pd.Timestamp("2024-01-05"),
        pd.Timestamp("2024-01-08"),
        pd.Timestamp("2024-01-08"),
    ]
