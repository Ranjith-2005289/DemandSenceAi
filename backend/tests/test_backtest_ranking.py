"""
Common-backtest ranking (services/model_runner.py).

Models used to be ranked on their OWN internal CV metrics, which were not
comparable: some scored one-step-ahead errors (true lags fed in at every test
step), others genuine multi-step forecasts. These tests pin the fix — every
model is re-run from the same forecast origins and ranked on those windows —
plus the ensemble and the backtest-calibrated intervals.
"""
import numpy as np
import pandas as pd
import pytest

from services.model_runner import _backtest_plan, run_all_models


def _truth(t):
    return 100 + 10 * np.sin(2 * np.pi * t / 7) + 0.05 * t


@pytest.fixture
def series():
    n = 300
    return pd.Series(_truth(np.arange(n)), index=pd.date_range("2023-01-01", periods=n, freq="D"))


def _oracle(internal_rmse):
    """Forecasts the true continuation — genuinely the best model."""
    def fn(s, h):
        t = np.arange(len(s), len(s) + h)
        return {"rmse": internal_rmse, "mae": internal_rmse, "mape": 0.1, "r2": 0.5,
                "forecast": list(_truth(t))}
    return fn


def _flatliner(internal_rmse):
    """Repeats the last value — poor multi-step forecaster, but claims a tiny
    internal RMSE (what a leaky one-step CV produces)."""
    def fn(s, h):
        return {"rmse": internal_rmse, "mae": internal_rmse, "mape": 0.01, "r2": 0.99,
                "forecast": [float(s.iloc[-1])] * h}
    return fn


def _noisy(scale, seed):
    def fn(s, h):
        rng = np.random.default_rng(seed + len(s))
        t = np.arange(len(s), len(s) + h)
        return {"rmse": 5.0, "mae": 4.0, "mape": 0.05, "r2": 0.7,
                "forecast": list(_truth(t) + rng.normal(0, scale, h))}
    return fn


def test_backtest_beats_misleading_internal_metrics(series):
    registry = {
        "Leaky":  (_flatliner(0.001), 30, "ml"),
        "Honest": (_oracle(50.0), 30, "statistical"),
    }
    # Old behaviour: the leaky internal metric wins.
    old, _ = run_all_models(series, 14, models=registry, backtest=False)
    assert old[0]["model_name"] == "Leaky"

    # New behaviour: on identical held-out windows the genuinely better model wins.
    new, info = run_all_models(series, 14, models=registry, ensemble=False)
    assert info["evaluation"] == "holdout_backtest"
    assert new[0]["model_name"] == "Honest"
    assert new[0]["mase"] < new[1]["mase"]
    # Internal metrics are preserved for transparency.
    assert new[0]["internal_cv"]["rmse"] == 50.0


def test_backtest_plan_uses_non_overlapping_windows_and_respects_min_train():
    h, origins = _backtest_plan(300, 30)
    assert h == 30 and origins == [270, 240]
    # Short series: horizon is capped at n // 5 and origins need ≥ 30 train points.
    h, origins = _backtest_plan(40, 30)
    assert h == 8 and all(c >= 30 for c in origins)
    assert _backtest_plan(20, 5)[1] == []  # too short to backtest at all


def test_too_short_series_falls_back_to_internal_cv():
    s = pd.Series(np.arange(20, dtype=float) + 1, index=pd.date_range("2024-01-01", periods=20))
    registry = {"A": (_oracle(3.0), 30, "statistical"), "B": (_flatliner(1.0), 30, "ml")}
    results, info = run_all_models(s, 5, models=registry)
    assert info["evaluation"] == "internal_cv"
    assert results[0]["model_name"] == "B"  # ranked by internal RMSE
    assert all(r.get("evaluation") == "internal_cv" for r in results)


def test_ensemble_averages_top_three_and_is_ranked(series):
    registry = {
        "N1": (_noisy(3.0, 1), 30, "statistical"),
        "N2": (_noisy(3.0, 2), 30, "statistical"),
        "N3": (_noisy(3.0, 3), 30, "statistical"),
        "Bad": (_flatliner(1.0), 30, "ml"),
    }
    results, _ = run_all_models(series, 14, models=registry)
    by = {r["model_name"]: r for r in results}
    ens = by["Ensemble"]
    assert sorted(ens["ensemble_members"]) == ["N1", "N2", "N3"]
    expected = np.mean([by[m]["forecast"] for m in ("N1", "N2", "N3")], axis=0)
    assert np.allclose(ens["forecast"], expected, atol=1e-3)
    # Averaging independent errors helps, so the ensemble should top the board.
    assert results[0]["model_name"] == "Ensemble"
    assert ens["mase"] < min(by[m]["mase"] for m in ("N1", "N2", "N3"))


def test_intervals_are_calibrated_on_backtest_errors_and_widen(series):
    registry = {"N1": (_noisy(3.0, 1), 30, "statistical"), "N2": (_noisy(6.0, 2), 30, "statistical")}
    results, _ = run_all_models(series, 14, models=registry, ensemble=False)
    by = {r["model_name"]: r for r in results}
    for r in by.values():
        assert r["interval_method"] == "backtest_calibrated"
        width = np.array(r["forecast_upper"]) - np.array(r["forecast_lower"])
        assert np.all(np.diff(width) >= -1e-9)  # never narrows with horizon
    # The noisier model must get the wider band.
    w1 = np.mean(np.array(by["N1"]["forecast_upper"]) - np.array(by["N1"]["forecast_lower"]))
    w2 = np.mean(np.array(by["N2"]["forecast_upper"]) - np.array(by["N2"]["forecast_lower"]))
    assert w2 > w1


def test_backtest_failure_ranks_after_backtested_models(series):
    calls = {"n": 0}

    def flaky(s, h):
        # Succeeds only on the full series, fails on every backtest refit.
        if len(s) < len(series):
            raise RuntimeError("cannot refit")
        return {"rmse": 0.0001, "mae": 0.0001, "mape": 0.0, "r2": 1.0, "forecast": [1.0] * h}

    registry = {"Flaky": (flaky, 30, "ml"), "Honest": (_oracle(50.0), 30, "statistical")}
    results, _ = run_all_models(series, 14, models=registry, ensemble=False)
    assert [r["model_name"] for r in results] == ["Honest", "Flaky"]
    assert results[1]["evaluation"] == "internal_cv"
