"""Shared fixtures for the backend test suite."""
import numpy as np
import pandas as pd
import pytest


def _make_trending_seasonal_series(n: int, freq: str = "D", seed: int = 0) -> pd.Series:
    rng = np.random.RandomState(seed)
    trend = np.linspace(100, 250, n)
    weekly = 20 * np.sin(2 * np.pi * np.arange(n) / 7)
    noise = rng.randn(n) * 5
    values = trend + weekly + noise
    return pd.Series(values, index=pd.date_range("2023-01-01", periods=n, freq=freq))


@pytest.fixture
def small_trending_series() -> pd.Series:
    """~180 rows — mirrors the benchmark series used throughout manual verification."""
    return _make_trending_seasonal_series(180)


@pytest.fixture
def below_dl_threshold_series() -> pd.Series:
    """366 daily rows — the user's own real-world example; must skip DL (threshold 500)."""
    return _make_trending_seasonal_series(366)


@pytest.fixture
def above_dl_threshold_series() -> pd.Series:
    """600 daily rows — above the 500-row daily DL threshold; DL should run."""
    return _make_trending_seasonal_series(600)


@pytest.fixture
def fake_fast_registry():
    """
    A tiny model registry with near-instant dummy models, for tests that need
    to exercise run_all_models' orchestration/gating logic without paying for
    real model training. Mirrors run_all_models' own models= override, which
    is built for exactly this purpose.
    """
    def _fake_model(rmse=1.0, mae=1.0, mape=0.01, r2=0.9, train_time=0.01, pred_time=0.01, name="Fake"):
        def _fn(series, horizon):
            return {
                "model_name": name,
                "rmse": rmse, "mae": mae, "mape": mape, "r2": r2,
                "forecast": [float(series.iloc[-1])] * horizon,
                "train_time_sec": train_time, "pred_time_sec": pred_time,
            }
        return _fn

    return {
        "FakeStat": (_fake_model(name="FakeStat"), 5, "statistical"),
        "FakeML": (_fake_model(name="FakeML", rmse=2.0, mae=1.5, mape=0.02), 5, "ml"),
        "FakeDL": (_fake_model(name="FakeDL", rmse=3.0, mae=2.0, mape=0.03), 5, "dl"),
    }
