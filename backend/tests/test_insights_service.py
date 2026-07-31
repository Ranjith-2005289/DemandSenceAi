"""Regression tests for the deterministic (non-LLM) parts of the AI Insights
Report: stockout-pattern detection, verified manually earlier this session."""
import numpy as np
import pandas as pd

from services.insights_service import detect_stockout_alerts, compute_dataset_overview


def test_detects_injected_flat_then_recovery_pattern():
    rng = np.random.RandomState(0)
    n = 100
    values = 100 + rng.randn(n) * 10
    values[40:50] = 5.0 + rng.randn(10) * 0.1   # flat, near-zero plateau
    values[50] = 110.0                           # sharp recovery jump right after
    series = pd.Series(values, index=pd.date_range("2023-01-01", periods=n, freq="D"))

    alerts = detect_stockout_alerts(series)
    assert len(alerts) == 1
    assert alerts[0]["start_date"] == "2023-02-10"
    assert alerts[0]["recovery_jump_pct"] > 50


def test_no_false_positives_on_clean_series():
    rng = np.random.RandomState(0)
    series = pd.Series(100 + rng.randn(100) * 10, index=pd.date_range("2023-01-01", periods=100, freq="D"))
    assert detect_stockout_alerts(series) == []


def test_dataset_overview_trend_direction():
    n = 100
    growing = pd.Series(np.linspace(100, 200, n), index=pd.date_range("2023-01-01", periods=n, freq="D"))
    overview = compute_dataset_overview(growing)
    assert overview["trend"] == "growing"
    assert overview["row_count"] == n
    assert overview["growth_rate_pct"] > 0
