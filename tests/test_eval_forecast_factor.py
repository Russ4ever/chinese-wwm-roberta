import numpy as np
import pandas as pd

from label_engineering.eval_forecast_factor import (
    neutralize_group,
    rating_consensus,
    summarize_ic,
)


def test_rating_consensus_ignores_unknown_codes_and_breaks_ties_upward():
    values = pd.Series([0, 1, 1, 7, 7, 2, np.nan])
    assert rating_consensus(values) == 7
    assert pd.isna(rating_consensus(pd.Series([0, 2, np.nan])))


def test_neutralize_group_returns_finite_residuals_without_dummy_trap():
    y = np.array([1.0, 2.0, 4.0, 5.0])
    industry = np.array([1, 1, 2, 2])
    size = np.array([1.0, 2.0, 1.0, 2.0])
    residual = neutralize_group(y, industry, size)
    assert np.isfinite(residual).all()
    assert abs(residual.mean()) < 1e-12


def test_single_ic_day_does_not_report_invalid_sample_std():
    summary = summarize_ic(np.array([0.2, np.nan]))
    assert summary["n_days"] == 1
    assert np.isnan(summary["ic_std"])
    assert np.isnan(summary["t_stat"])
