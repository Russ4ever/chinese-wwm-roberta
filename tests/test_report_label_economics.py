import numpy as np
import pandas as pd
import pytest

from label_engineering.validate_report_label_economics import (
    _validate_history_coverage,
)
from src.report_label_economics import (
    build_actual_surprise_samples,
    build_analyst_information_samples,
    build_consensus_revision_samples,
    canonical_stock_code,
    compound_between_events,
    compound_forward,
    consistency_audit,
    index_returns_from_prices,
    lock_annual_universes,
    normalize_csi_weights,
    weighted_index_returns,
)


def test_stock_codes_are_strict_and_do_not_guess_malformed_identifiers():
    result = canonical_stock_code(
        pd.Series(["000001.SZ", "600000", "A25082.SH", "Unnamed: 0"])
    )
    assert result.iloc[:2].tolist() == ["000001", "600000"]
    assert result.iloc[2:].isna().all()


def test_actual_consensus_requires_full_declared_report_history_coverage():
    actuals = pd.DataFrame({"actual_known_date": ["2025-03-01"]})
    metadata = {
        "configuration": {
            "coverage_start": "2024-09-03",
            "coverage_end": "2025-03-01",
        }
    }
    with pytest.raises(ValueError, match="覆盖不足"):
        _validate_history_coverage(metadata, actuals, lookback_days=180)

    metadata["configuration"]["coverage_start"] = "2024-09-02"
    start, end = _validate_history_coverage(metadata, actuals, lookback_days=180)
    assert start == pd.Timestamp("2024-09-02")
    assert end == pd.Timestamp("2025-03-01")


def test_forward_returns_use_close_t_to_close_t_plus_one_without_filling_missing():
    dates = pd.bdate_range("2024-01-02", periods=4)
    panel = pd.DataFrame({"000001": [0.10, -0.05, 0.02, 0.01], "000002": [0.03, np.nan, 0.01, 0.02]}, index=dates)
    result = compound_forward(panel, [1, 2])
    assert np.isclose(result[1].loc[dates[0], "000001"], 0.10)
    assert np.isclose(result[2].loc[dates[0], "000001"], 1.10 * 0.95 - 1.0)
    assert pd.isna(result[2].loc[dates[0], "000002"])
    audit = consistency_audit(result[2], result[2].copy(), atol=1e-12, rtol=1e-12)
    assert audit["fraction_close"] == 1.0


def test_event_compounding_stops_before_target_close():
    dates = pd.bdate_range("2024-01-02", periods=4)
    daily = pd.DataFrame({"000001": [0.10, 0.20, 0.30, 0.40]}, index=dates)
    events = pd.DataFrame({"stock_code": ["000001.SZ"], "start": [dates[0]], "end": [dates[2]]})
    value = compound_between_events(events, daily, start_column="start", end_column="end")[0]
    assert np.isclose(value, 1.10 * 1.20 - 1.0)


def _weights():
    return normalize_csi_weights(
        pd.DataFrame(
            {
                "S_INFO_WINDCODE": ["000300.SH"] * 5,
                "S_CON_WINDCODE": ["000001.SZ", "000002.SZ", "000001.SZ", "000003.SZ", "000004.SZ"],
                "TRADE_DT": ["20221230", "20221230", "20231229", "20231229", "20240115"],
                "I_WEIGHT": [60.0, 40.0, 70.0, 30.0, 100.0],
            }
        ),
        index_code="000300.SH",
    )


def test_locked_universe_uses_only_prior_year_snapshot():
    locked, universes = lock_annual_universes(_weights(), [2023, 2024])
    assert universes[2023] == {"000001", "000002"}
    assert universes[2024] == {"000001", "000003"}
    assert "000004" not in universes[2024]
    assert set(locked.loc[locked["validation_year"].eq(2024), "snapshot_date"]) == {pd.Timestamp("2023-12-29")}


def test_index_return_paths_have_same_close_t_orientation_and_no_future_weights():
    prices = pd.DataFrame(
        {
            "S_INFO_WINDCODE": ["000300.SH"] * 3,
            "TRADE_DT": ["20240102", "20240103", "20240104"],
            "S_DQ_CLOSE": [100.0, 110.0, 99.0],
        }
    )
    direct = index_returns_from_prices(prices, index_code="000300.SH")
    assert np.isclose(direct.loc[pd.Timestamp("2024-01-02")], 0.10)

    weights = _weights()
    dates = pd.to_datetime(["2024-01-02", "2024-01-16"])
    returns = pd.DataFrame({"000001": [0.10, 0.50], "000003": [0.20, 0.50], "000004": [0.90, 0.10]}, index=dates)
    weighted = weighted_index_returns(weights, returns, min_weight_coverage=0.95)
    assert np.isclose(weighted.loc[dates[0], "csi300_return_1d"], 0.70 * 0.10 + 0.30 * 0.20)
    assert np.isclose(weighted.loc[dates[1], "csi300_return_1d"], 0.10)


def test_actual_consensus_is_strictly_before_disclosure_and_latest_per_org():
    actuals = pd.DataFrame(
        {
            "stock_code": ["000001"],
            "fy": [2024],
            "actual_np": [130.0],
            "actual_known_date": ["2025-03-01"],
        }
    )
    history = pd.DataFrame(
        {
            "stock_code": ["000001"] * 7,
            "fy": [2024] * 7,
            "org_id": ["A", "A", "B", "C", "D", "E", "SAME_DAY"],
            "available_date": pd.to_datetime(["2024-12-01", "2025-02-01", "2025-01-10", "2025-01-11", "2025-01-12", "2024-01-01", "2025-03-01"]),
            "publish_date": pd.to_datetime(["2024-12-01", "2025-02-01", "2025-01-10", "2025-01-11", "2025-01-12", "2024-01-01", "2025-03-01"]),
            "forecast_new": [80.0, 100.0, 110.0, 120.0, 130.0, 999.0, 1000.0],
            "forecast_conflict": [0] * 7,
        }
    )
    result = build_actual_surprise_samples(actuals, history, {2024: {"000001"}}, lookback_days=180, min_peer_orgs=4, floor_ratio=0.01)
    row = result.iloc[0]
    assert row["n_org_actual_pre"] == 4
    assert row["consensus_actual_pre"] == 115.0
    assert row["actual_surprise_valid"] == 1
    assert np.isclose(row["actual_surprise"], 15.0 / 115.0)


def test_actual_surprise_scale_uses_only_that_events_pre_disclosure_peers():
    actuals = pd.DataFrame(
        {
            "stock_code": ["000001", "000002"],
            "fy": [2024, 2024],
            "actual_np": [1.0, 1_000_000.0],
            "actual_known_date": ["2025-03-01", "2025-03-01"],
        }
    )
    rows = []
    for stock, base in (("000001", 0.0), ("000002", 1_000_000.0)):
        for offset, org in enumerate(("A", "B", "C", "D")):
            rows.append(
                {
                    "stock_code": stock,
                    "fy": 2024,
                    "org_id": org,
                    "available_date": pd.Timestamp("2025-02-01"),
                    "publish_date": pd.Timestamp("2025-02-01"),
                    "forecast_new": base + offset,
                    "forecast_conflict": 0,
                }
            )
    result = build_actual_surprise_samples(
        actuals,
        pd.DataFrame(rows),
        {2024: {"000001", "000002"}},
        lookback_days=180,
        min_peer_orgs=4,
        floor_ratio=0.01,
    ).set_index("stock_code")
    assert np.isclose(result.loc["000001", "actual_scale_floor"], 0.015)
    assert np.isclose(result.loc["000002", "actual_scale_floor"], 10_000.015)


def test_analyst_information_filters_by_report_year_point_in_time_pool():
    labels = pd.DataFrame(
        {
            "report_id": ["in", "out"],
            "stock_code": ["000001", "000002"],
            "available_date": pd.to_datetime(["2024-06-01", "2024-06-01"]),
            "actual_known_date": pd.to_datetime(["2025-03-01", "2025-03-01"]),
            "pre_label_valid": [1, 1],
            "actual_np": [120.0, 120.0],
            "forecast_new": [110.0, 110.0],
            "consensus_pre": [100.0, 100.0],
            "scale_t": [100.0, 100.0],
            "edge_raw": [0.1, 0.1],
            "report_abs_error": [0.1, 0.1],
            "consensus_abs_error": [0.2, 0.2],
        }
    )
    result = build_analyst_information_samples(labels, {2024: {"000001"}})
    assert result["report_id"].tolist() == ["in"]
    assert result.iloc[0]["direction_hit"]


def test_valid_zero_progress_is_retained_even_when_not_probe_valid():
    confirmation = pd.DataFrame(
        {
            "report_id": ["zero", "invalid"],
            "stock_code": ["000001", "000001"],
            "fy": [2024, 2024],
            "available_date": pd.to_datetime(["2024-06-01", "2024-06-01"]),
            "target_date": pd.to_datetime(["2024-07-01", "2024-07-01"]),
            "confirmation_valid": [1, 0],
            "confirmation_probe_valid": [0, 0],
            "consensus_pre": [100.0, 100.0],
            "consensus_future": [100.0, 100.0],
            "scale_t": [100.0, 100.0],
            "forecast_new": [120.0, 120.0],
            "progress_raw": [0.0, 0.0],
        }
    )
    labels = pd.DataFrame(
        {
            "report_id": ["zero", "invalid"],
            "fy": [2024, 2024],
            "actual_np": [110.0, 110.0],
        }
    )
    result = build_consensus_revision_samples(
        confirmation, labels, {2024: {"000001"}}
    )
    assert result["report_id"].tolist() == ["zero"]
    assert result.iloc[0]["progress_raw"] == 0.0
    assert result.iloc[0]["confirmation_probe_valid"] == 0
