"""
Deep learning models are auto-skipped below a frequency-aware row-count
threshold rather than trained anyway and flagged as unreliable after the
fact (wasting real compute time — DL models took 40-130s each in manual
testing). These tests use the fake fast registry so they exercise the real
gating logic in run_all_models without paying for real model training.
"""
import numpy as np
import pandas as pd

from services.model_runner import run_all_models, MIN_ROWS_FOR_DL_BY_TIER


def test_dl_skipped_below_daily_threshold(below_dl_threshold_series, fake_fast_registry):
    results, run_info = run_all_models(
        below_dl_threshold_series, forecast_horizon=10,
        include_dl=True, models=fake_fast_registry,
    )
    assert run_info["dl_skipped"] is True
    assert run_info["min_rows_for_dl"] == 500
    assert run_info["frequency_tier"] == "daily"
    assert "FakeDL" not in [r["model_name"] for r in results]
    assert "FakeStat" in [r["model_name"] for r in results]


def test_dl_runs_above_daily_threshold(above_dl_threshold_series, fake_fast_registry):
    results, run_info = run_all_models(
        above_dl_threshold_series, forecast_horizon=10,
        include_dl=True, models=fake_fast_registry,
    )
    assert run_info["dl_skipped"] is False
    assert "FakeDL" in [r["model_name"] for r in results]


def test_explicit_skip_dl_flag_still_works(above_dl_threshold_series, fake_fast_registry):
    """A caller explicitly requesting include_dl=False should still get no DL
    models even when the dataset is large enough to otherwise run them."""
    results, run_info = run_all_models(
        above_dl_threshold_series, forecast_horizon=10,
        include_dl=False, models=fake_fast_registry,
    )
    assert "FakeDL" not in [r["model_name"] for r in results]
    # dl_skipped only describes the *automatic* row-count gate, not an
    # explicit caller request — assert it reflects that the row count itself
    # was sufficient (the gate never had to intervene).
    assert run_info["dl_skipped"] is False


def test_weekly_frequency_uses_weekly_threshold(fake_fast_registry):
    series = pd.Series(
        np.random.RandomState(0).randn(150) + 100,
        index=pd.date_range("2020-01-01", periods=150, freq="W"),
    )
    results, run_info = run_all_models(
        series, forecast_horizon=4, include_dl=True, models=fake_fast_registry,
    )
    assert run_info["frequency_tier"] == "weekly"
    assert run_info["min_rows_for_dl"] == MIN_ROWS_FOR_DL_BY_TIER["weekly"]
    assert run_info["dl_skipped"] is False  # 150 > 104-row weekly threshold


def test_monthly_frequency_below_its_own_threshold(fake_fast_registry):
    series = pd.Series(
        np.random.RandomState(0).randn(20) + 100,
        index=pd.date_range("2020-01-01", periods=20, freq="MS"),
    )
    results, run_info = run_all_models(
        series, forecast_horizon=3, include_dl=True, models=fake_fast_registry,
    )
    assert run_info["frequency_tier"] == "monthly"
    assert run_info["dl_skipped"] is True  # 20 < 36-row monthly threshold
