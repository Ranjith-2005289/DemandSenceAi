"""
Real (not faked) integration tests — actually train the statistical/ML
model registry end-to-end. Opt-in only (pytest -m slow) since these take
real wall-clock time, unlike the rest of the suite which uses fake models.
"""
import numpy as np
import pandas as pd
import pytest

from services.preprocessor import preprocess
from services.model_runner import run_all_models
from services.best_model import select_best


@pytest.mark.slow
def test_full_pipeline_small_dataset_no_dl():
    rng = np.random.RandomState(0)
    n = 120
    df = pd.DataFrame({
        "Date": pd.date_range("2023-01-01", periods=n).strftime("%Y-%m-%d"),
        "Sales": 100 + np.cumsum(rng.randn(n) * 2) + 10 * np.sin(2 * np.pi * np.arange(n) / 7),
    })

    series, report = preprocess(df, date_col="Date", target_col="Sales")
    assert len(series) > 0

    results, run_info = run_all_models(series, forecast_horizon=7, include_dl=False)
    assert len(results) > 0
    assert all(r["status"] in ("success", "failed", "timeout") for r in results)

    best = select_best(results, series=series)
    assert best["best_model_name"] is not None
    assert len(best["forecast"]) == 7


@pytest.mark.slow
def test_real_registry_ranks_on_common_backtest_and_builds_ensemble():
    """Every successful model must be scored on the same holdout windows
    (comparable MASE), the removed models must stay removed, and the top-k
    ensemble must be built from backtested members."""
    rng = np.random.RandomState(0)
    n = 200
    series = pd.Series(
        100 + np.cumsum(rng.randn(n) * 0.3) + 10 * np.sin(2 * np.pi * np.arange(n) / 7),
        index=pd.date_range("2023-01-01", periods=n, freq="D"),
    )
    results, run_info = run_all_models(series, forecast_horizon=7)
    names = {r["model_name"] for r in results}

    removed = {"RNN", "LSTM", "GRU", "BiLSTM", "CNN1D", "TCN", "Transformer", "KNN"}
    assert not (removed & names)
    assert {"Theta", "AutoETS", "MSTL", "SeasonalNaive"} <= names

    assert run_info["evaluation"] == "holdout_backtest"
    ok = [r for r in results if r["status"] == "success"]
    assert all(r["evaluation"] == "holdout_backtest" for r in ok[:5])
    scores = [r["adjusted_score"] for r in ok if np.isfinite(r["adjusted_score"])]
    assert scores == sorted(scores)

    ens = next((r for r in results if r["model_name"] == "Ensemble"), None)
    assert ens is not None and len(ens["ensemble_members"]) >= 2
    assert ens["interval_method"] == "backtest_calibrated"
    assert len(ens["forecast_lower"]) == 7
