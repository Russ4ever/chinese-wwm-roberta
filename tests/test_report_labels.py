import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from label_engineering.prepare_actuals import (
    ACTUAL_METRIC,
    ACTUAL_VERSION,
    prepare_actuals,
)
from label_engineering.build_report_labels import load_actuals, run
from src.report_labeling import (
    assign_point_in_time_scale,
    attach_actual_labels,
    attach_pre_consensus,
    build_confirmation_labels,
    build_org_histories,
    canonicalize_report_rows,
    first_trading_date_on_or_after,
    validate_actual_scale,
)
from src.report_label_runtime import resolve_runtime_resources


CALENDAR = pd.bdate_range("2023-01-02", "2025-12-31")


def _record(org, date, forecast, *, title=None, fy=2024, content="正文"):
    return {
        "ID": f"{org}-{date}-{fy}",
        "STOCK_CODE": "000001",
        "STOCK_NAME": "样例公司",
        "ORGAN_NAME": org,
        "AUTHOR_NAME": "分析师",
        "TITLE": title or f"{org}预测",
        "CONTENT": content,
        "CREATE_DATE": f"{date} 09:00:00",
        "REPORT_YEAR": fy,
        "FORECAST_NP": forecast,
    }


def _scenario():
    records = [
        _record("A", "2023-12-01", 90),
        _record("B", "2023-12-01", 100),
        _record("C", "2023-12-01", 110),
        _record("D", "2023-12-01", 120),
        _record("Seed", "2024-01-10", 105, title="历史校准报告"),
        _record("Origin", "2024-02-02", 125, title="目标报告"),
        _record("A", "2024-02-15", 110),
        _record("B", "2024-02-15", 111),
        _record("C", "2024-02-15", 112),
        _record("Entrant", "2024-02-15", 200),
        _record("Origin", "2024-02-16", 300, title="原机构后续报告"),
    ]
    raw = pd.DataFrame(records)
    reports, rows = canonicalize_report_rows(raw, CALENDAR, forecast_multiplier=1.0)
    histories = build_org_histories(rows)
    rows = attach_pre_consensus(
        rows,
        histories,
        coverage_start=pd.Timestamp("2023-01-01"),
        lookback_days=180,
        min_peer_orgs=4,
    )
    rows = assign_point_in_time_scale(rows, history_months=12, min_samples=1)
    target = rows[rows["title"] == "目标报告"].copy().reset_index(drop=True)
    return reports, rows, histories, target


def test_dynamic_fy_uses_report_year_and_text_is_deduplicated():
    raw = pd.DataFrame(
        [
            _record("Origin", "2024-02-02", 125, title="同一报告", fy=2024),
            _record("Origin", "2024-02-02", 150, title="同一报告", fy=2025),
        ]
    )
    reports, rows = canonicalize_report_rows(raw, CALENDAR, forecast_multiplier=1.0)
    assert len(reports) == 1
    assert sorted(rows["forecast_horizon"].tolist()) == [0, 1]

    conflict_raw = pd.DataFrame(
        [
            _record("Origin", "2024-02-02", 125, title="冲突报告", fy=2024),
            _record("Origin", "2024-02-02", 130, title="冲突报告", fy=2024),
        ]
    )
    _, conflict_rows = canonicalize_report_rows(
        conflict_raw, CALENDAR, forecast_multiplier=1.0
    )
    assert conflict_rows.loc[0, "forecast_conflict"] == 1
    assert pd.isna(conflict_rows.loc[0, "forecast_new"])
    assert build_org_histories(conflict_rows) == {}


def test_confirmation_panels_exclude_origin_and_measure_convergence():
    _, _, histories, target = _scenario()
    confirmation = build_confirmation_labels(
        target,
        histories,
        CALENDAR,
        coverage_end=pd.Timestamp("2025-12-31"),
        confirmation_months=[1],
        lookback_days=180,
        min_peer_orgs=4,
        min_active_orgs=3,
        min_probe_updates=2,
    )
    assert set(confirmation["peer_panel"]) == {"fixed", "market", "active"}
    assert confirmation["confirmation_valid"].eq(1).all()
    assert confirmation["confirmation_probe_valid"].eq(1).all()
    assert confirmation["progress_raw"].gt(0).all()
    assert confirmation["delta_log_dispersion"].lt(0).all()
    fixed = confirmation[confirmation["peer_panel"] == "fixed"].iloc[0]
    market = confirmation[confirmation["peer_panel"] == "market"].iloc[0]
    active = confirmation[confirmation["peer_panel"] == "active"].iloc[0]
    assert fixed["n_org_entries"] == 0
    assert fixed["n_org_future"] == 5
    assert fixed["n_peer_updates"] == 3  # D与Seed未更新，fixed沿用发布前旧值
    assert market["n_org_entries"] == 1
    assert active["n_org_pre"] == 3
    assert confirmation["consensus_future"].lt(200).all()  # 原机构的300从未进入同行共识

    inline = target.copy()
    inline["forecast_new"] = inline["consensus_pre"]
    inline_confirmation = build_confirmation_labels(
        inline,
        histories,
        CALENDAR,
        coverage_end=pd.Timestamp("2025-12-31"),
        confirmation_months=[1],
        lookback_days=180,
        min_peer_orgs=4,
        min_active_orgs=3,
        min_probe_updates=2,
    )
    assert inline_confirmation["confirmation_valid"].eq(0).all()
    assert inline_confirmation["progress_raw"].isna().all()
    assert set(inline_confirmation["invalid_reason"]) == {
        "report_inline_with_consensus"
    }

    down_records = [
        _record("A", "2023-12-01", 90),
        _record("B", "2023-12-01", 100),
        _record("C", "2023-12-01", 110),
        _record("D", "2023-12-01", 120),
        _record("Seed", "2024-01-10", 105, title="下看历史校准"),
        _record("Origin", "2024-02-02", 85, title="下看目标报告"),
        _record("A", "2024-02-15", 80),
        _record("B", "2024-02-15", 90),
        _record("C", "2024-02-15", 100),
        _record("D", "2024-02-15", 110),
    ]
    _, down_rows = canonicalize_report_rows(
        pd.DataFrame(down_records), CALENDAR, forecast_multiplier=1.0
    )
    down_histories = build_org_histories(down_rows)
    down_rows = attach_pre_consensus(
        down_rows,
        down_histories,
        coverage_start=pd.Timestamp("2023-01-01"),
        lookback_days=180,
        min_peer_orgs=4,
    )
    down_rows = assign_point_in_time_scale(down_rows, history_months=12, min_samples=1)
    down_target = down_rows[down_rows["title"] == "下看目标报告"]
    down_confirmation = build_confirmation_labels(
        down_target,
        down_histories,
        CALENDAR,
        coverage_end=pd.Timestamp("2025-12-31"),
        confirmation_months=[1],
        lookback_days=180,
        min_peer_orgs=4,
        min_active_orgs=3,
        min_probe_updates=2,
    )
    assert down_confirmation["progress_raw"].gt(0).all()


def test_actual_residual_edge_and_disclosure_censoring():
    _, _, histories, target = _scenario()
    actuals = pd.DataFrame(
        [
            {
                "stock_code": "000001",
                "fy": 2024,
                "actual_np": 120.0,
                "actual_publish_date": "2025-03-01",
                "actual_known_date": "2025-02-01",
                "unit_multiplier": 1.0,
                "currency": "CNY",
                "source_id": "annual-2024",
                "actual_version": "annual_report_first",
            }
        ]
    )
    labeled = attach_actual_labels(target, actuals, actual_source_available=True)
    row = labeled.iloc[0]
    assert row["residual_valid"] == 1
    assert row["residual_signed_raw"] < 0  # Actual 120 < 报告预测125，报告高估
    assert row["edge_raw"] > 0  # 报告125比发布前共识105更接近Actual120
    assert row["edge_sign"] == "positive"
    assert row["actual_label_available_date"] == pd.Timestamp("2025-03-01")

    crossed = labeled.copy()
    crossed["actual_known_date"] = pd.Timestamp("2024-02-20")
    confirmation = build_confirmation_labels(
        crossed,
        histories,
        CALENDAR,
        coverage_end=pd.Timestamp("2025-12-31"),
        confirmation_months=[1],
        lookback_days=180,
        min_peer_orgs=4,
        min_active_orgs=3,
        min_probe_updates=2,
    )
    assert confirmation["confirmation_valid"].eq(0).all()
    assert set(confirmation["invalid_reason"]) == {"crosses_actual_disclosure"}

    unavailable = attach_actual_labels(
        target, pd.DataFrame(), actual_source_available=False
    )
    assert unavailable["residual_valid"].eq(0).all()
    assert unavailable["edge_valid"].eq(0).all()
    assert set(unavailable["actual_invalid_reason"]) == {"actual_source_unavailable"}

    too_late = target.copy()
    too_late["available_date"] = pd.Timestamp("2025-02-01")
    leaked = attach_actual_labels(too_late, actuals, actual_source_available=True)
    assert leaked.loc[0, "residual_valid"] == 0
    assert (
        leaked.loc[0, "actual_invalid_reason"] == "report_not_before_actual_known_date"
    )
    assert pd.isna(leaked.loc[0, "residual_signed_raw"])

    loss_actuals = actuals.copy()
    loss_actuals["actual_np"] = -10.0
    loss = attach_actual_labels(target, loss_actuals, actual_source_available=True)
    assert np.isfinite(loss.loc[0, "residual_signed_raw"])
    assert loss.loc[0, "actual_transition_state"] == "profit_to_loss"

    near_zero_actuals = actuals.copy()
    near_zero_actuals["actual_np"] = 0.5
    near_zero = attach_actual_labels(
        target, near_zero_actuals, actual_source_available=True
    )
    assert np.isfinite(near_zero.loc[0, "edge_raw"])
    assert near_zero.loc[0, "actual_transition_state"] == "profit_to_near_zero"

    unit_mismatch = labeled.copy()
    unit_mismatch["actual_np"] = 120_000.0
    with np.testing.assert_raises_regex(ValueError, "数量级"):
        validate_actual_scale(unit_mismatch)


def test_target_dates_are_report_anchored_not_month_end():
    first_report = pd.Timestamp("2024-01-02")
    second_report = pd.Timestamp("2024-01-30")
    first = first_trading_date_on_or_after(
        first_report + pd.DateOffset(months=1), CALENDAR
    )
    second = first_trading_date_on_or_after(
        second_report + pd.DateOffset(months=1), CALENDAR
    )
    assert first == pd.Timestamp("2024-02-02")
    assert second == pd.Timestamp("2024-02-29")
    assert first != second
    assert pd.isna(first_trading_date_on_or_after(pd.Timestamp("2026-01-01"), CALENDAR))


def test_configurable_cli_writes_only_requested_report_window(tmp_path: Path):
    records = [
        _record("A", "2023-12-01", 90),
        _record("B", "2023-12-01", 100),
        _record("C", "2023-12-01", 110),
        _record("D", "2023-12-01", 120),
        _record("Seed", "2024-01-10", 105, title="历史校准报告"),
        _record("Origin", "2024-02-02", 125, title="目标报告"),
        _record("A", "2024-02-15", 110),
        _record("B", "2024-02-15", 111),
        _record("C", "2024-02-15", 112),
    ]
    reports_path = tmp_path / "forecast_stk_20230101_20251231.jsonl"
    reports_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records),
        encoding="utf-8",
    )
    calendar_path = tmp_path / "calendar.parquet"
    pd.DataFrame({"date": CALENDAR}).to_parquet(calendar_path, index=False)
    output = tmp_path / "out"
    config = tmp_path / "config.yaml"
    config.write_text(
        f"""
paths:
  reports: {reports_path}
  trading_calendar: {calendar_path}
  actuals: ""
  actual_schema: ""
calendar:
  date_column: date
coverage:
  start_date: 2023-01-01
  end_date: 2025-12-31
build:
  start_date: 2024-02-01
  end_date: 2024-02-10
  forecast_horizons: [0, 1, 2]
  confirmation_months: [1, 3]
  lookback_days: 180
  scale_history_months: 12
  min_scale_samples: 1
  min_peer_orgs: 4
  min_active_orgs: 3
  min_probe_updates: 2
  forecast_np_unit_multiplier: 1.0
  forecast_metric: net_profit_incl_min_int_inc
performance:
  shared_server: true
  max_threads: 2
  nice_increment: 0
  blas_threads: 1
cache:
  enabled: true
  directory: {tmp_path / "cache"}
output:
  directory: {output}
""",
        encoding="utf-8",
    )
    first_metadata = run(
        argparse.Namespace(
            config=str(config), start_date=None, end_date=None, output_dir=None
        )
    )
    second_metadata = run(
        argparse.Namespace(
            config=str(config), start_date=None, end_date=None, output_dir=None
        )
    )
    reports = pd.read_parquet(output / "reports.parquet")
    fy = pd.read_parquet(output / "report_fy_labels.parquet")
    confirmation = pd.read_parquet(output / "report_confirmation_labels.parquet")
    assert reports["title"].tolist() == ["目标报告"]
    assert reports["text"].tolist() == ["目标报告。正文"]
    assert fy["title"].tolist() == ["目标报告"]
    assert len(confirmation) == 6  # 1/3个月 × 三种面板
    assert set(fy["actual_invalid_reason"]) == {"actual_source_unavailable"}
    assert np.isfinite(
        confirmation.loc[confirmation["confirmation_valid"] == 1, "progress_raw"]
    ).all()
    audit = pd.read_csv(output / "label_coverage_audit.csv")
    assert {"count", "total_count", "rate"}.issubset(audit.columns)
    assert audit["rate"].between(0, 1).all()
    assert first_metadata["cache"]["hit"] is False
    assert second_metadata["cache"]["hit"] is True
    assert second_metadata["runtime"]["effective_threads"] <= 2
    assert {
        "report_cache_load_or_build",
        "pre_consensus_and_scale",
        "future_confirmation_and_audit",
    }.issubset(second_metadata["runtime"]["stage_metrics"])
    reports_path.write_text(
        reports_path.read_text(encoding="utf-8")
        + json.dumps(_record("Late", "2024-03-01", 130), ensure_ascii=False)
        + "\n",
        encoding="utf-8",
    )
    invalidated_metadata = run(
        argparse.Namespace(
            config=str(config), start_date=None, end_date=None, output_dir=None
        )
    )
    assert invalidated_metadata["cache"]["hit"] is False
    assert invalidated_metadata["cache"]["key"] != first_metadata["cache"]["key"]


def test_numba_labels_are_deterministic_between_one_and_eight_threads():
    import numba

    _, _, histories, target = _scenario()
    previous = numba.get_num_threads()
    try:
        numba.set_num_threads(1)
        one = build_confirmation_labels(
            target,
            histories,
            CALENDAR,
            coverage_end=pd.Timestamp("2025-12-31"),
            confirmation_months=[1, 3],
            lookback_days=180,
            min_peer_orgs=4,
            min_active_orgs=3,
            min_probe_updates=2,
        )
        numba.set_num_threads(min(8, numba.config.NUMBA_NUM_THREADS))
        many = build_confirmation_labels(
            target,
            histories,
            CALENDAR,
            coverage_end=pd.Timestamp("2025-12-31"),
            confirmation_months=[1, 3],
            lookback_days=180,
            min_peer_orgs=4,
            min_active_orgs=3,
            min_probe_updates=2,
        )
    finally:
        numba.set_num_threads(previous)
    pd.testing.assert_frame_equal(one, many, check_exact=True)


def test_shared_server_runtime_never_exceeds_machine_or_scheduler_limit(monkeypatch):
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "4")
    resources = resolve_runtime_resources(
        {
            "shared_server": True,
            "max_threads": 32,
            "nice_increment": 5,
            "blas_threads": 1,
        }
    )
    assert resources.configured_threads == 32
    assert 1 <= resources.effective_threads <= 4


def test_prepare_actuals_uses_first_actual_date_and_quarantines_conflicts(
    tmp_path: Path,
):
    columns = {
        "STATEMENT_TYPE": "408001000",
        "NET_PROFIT_EXCL_MIN_INT_INC": 90.0,
        "OPDATE": "20260101",
    }
    source = pd.DataFrame(
        [
            {
                **columns,
                "S_INFO_WINDCODE": "000001.SZ",
                "ANN_DT": "20250401",
                "REPORT_PERIOD": "20241231",
                "NET_PROFIT_INCL_MIN_INT_INC": 100.0,
                "ACTUAL_ANN_DT": "20250401",
                "OBJECT_ID": "first-a",
            },
            {
                **columns,
                "S_INFO_WINDCODE": "000001.SZ",
                "ANN_DT": "20250401",
                "REPORT_PERIOD": "20241231",
                "NET_PROFIT_INCL_MIN_INT_INC": 100.005,
                "ACTUAL_ANN_DT": "20250401",
                "OBJECT_ID": "first-b",
            },
            {
                **columns,
                "S_INFO_WINDCODE": "000001.SZ",
                "ANN_DT": "20250501",
                "REPORT_PERIOD": "20241231",
                "NET_PROFIT_INCL_MIN_INT_INC": 110.0,
                "ACTUAL_ANN_DT": "20250501",
                "OBJECT_ID": "revision",
            },
            # 截图同类问题：同一股票、报告期、首次实际公告日却有明显不同数值。
            {
                **columns,
                "S_INFO_WINDCODE": "301699.SZ",
                "ANN_DT": "20251128",
                "REPORT_PERIOD": "20251231",
                "NET_PROFIT_INCL_MIN_INT_INC": 42_309_240.0,
                "ACTUAL_ANN_DT": "20251128",
                "OBJECT_ID": "conflict-a",
            },
            {
                **columns,
                "S_INFO_WINDCODE": "301699.SZ",
                "ANN_DT": "20251128",
                "REPORT_PERIOD": "20251231",
                "NET_PROFIT_INCL_MIN_INT_INC": 262_825_000.0,
                "ACTUAL_ANN_DT": "20251128",
                "OBJECT_ID": "conflict-b",
            },
            {
                **columns,
                "S_INFO_WINDCODE": "000002.SZ",
                "ANN_DT": "20250801",
                "REPORT_PERIOD": "20250630",
                "NET_PROFIT_INCL_MIN_INT_INC": -20.0,
                "ACTUAL_ANN_DT": "20250801",
                "OBJECT_ID": "quarterly",
            },
            {
                **columns,
                "S_INFO_WINDCODE": "A25082.SH",
                "ANN_DT": "20251223",
                "REPORT_PERIOD": "20241231",
                "NET_PROFIT_INCL_MIN_INT_INC": -30.0,
                "ACTUAL_ANN_DT": "20251223",
                "OBJECT_ID": "bad-code",
            },
        ]
    )
    input_path = tmp_path / "ashare_np_enriched_20050101_20260801.parquet"
    output_path = tmp_path / "clean.parquet"
    conflicts_path = tmp_path / "conflicts.parquet"
    audit_path = tmp_path / "audit.json"
    source.to_parquet(input_path, index=False)

    audit = prepare_actuals(input_path, output_path, conflicts_path, audit_path)
    clean = pd.read_parquet(output_path)
    conflicts = pd.read_parquet(conflicts_path)
    assert clean[["stock_code", "fy"]].to_records(index=False).tolist() == [
        ("000001", 2024)
    ]
    assert clean.loc[0, "actual_np"] == 100.0
    assert str(clean.loc[0, "actual_known_date"]) == "2025-04-01"
    assert clean.loc[0, "n_later_revisions"] == 1
    assert clean.loc[0, "actual_metric"] == ACTUAL_METRIC
    assert clean.loc[0, "actual_version"] == ACTUAL_VERSION
    assert set(conflicts["stock_code"]) == {"301699"}
    assert audit["counts"]["conflicting_groups"] == 1
    assert audit["counts"]["invalid_reasons"] == {
        "invalid_stock_code": 1,
        "not_valid_annual_period": 1,
    }
    parquet_metadata = pq.read_metadata(output_path).metadata
    assert parquet_metadata[b"actual_metric"].decode() == ACTUAL_METRIC
    loaded, available, _ = load_actuals(
        output_path, None, expected_metric=ACTUAL_METRIC
    )
    assert available and len(loaded) == 1
    with np.testing.assert_raises_regex(ValueError, "预测指标"):
        load_actuals(output_path, None, expected_metric="net_profit_excl_min_int_inc")
