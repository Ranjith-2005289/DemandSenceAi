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
def test_dl_gate_with_real_models_below_threshold():
    """Confirms the real (non-fake) registry actually excludes DL models
    below the row threshold, not just the fake-registry unit tests."""
    rng = np.random.RandomState(0)
    n = 200  # below the 500-row daily DL threshold
    series = pd.Series(
        100 + np.cumsum(rng.randn(n) * 0.3) + 10 * np.sin(2 * np.pi * np.arange(n) / 7),
        index=pd.date_range("2023-01-01", periods=n, freq="D"),
    )
    results, run_info = run_all_models(series, forecast_horizon=7, include_dl=True)
    assert run_info["dl_skipped"] is True
    dl_names = {"RNN", "LSTM", "GRU", "BiLSTM", "CNN1D", "TCN", "Transformer"}
    assert not (dl_names & {r["model_name"] for r in results})
