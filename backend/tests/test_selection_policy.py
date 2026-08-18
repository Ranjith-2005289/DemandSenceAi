"""
Model ranking policy: cross-validated RMSE, optionally restricted to models
that fit a training-time budget (services/model_runner.py).

This replaced a six-term weighted score that lost to cross-validated RMSE alone
on every one of four M5 aggregation levels. The failure was structural: each
term was divided by its own Tukey fence, which bounds the denominator but not
the ratio, so training time (spanning six orders of magnitude) produced
normalized values reaching 43 while RMSE (spanning ~1.2x) stayed inside 0.5-1.0.
A nominally 0.10-weighted term therefore drove 63-88% of selections.

These tests pin the replacement's behaviour so that regression is detectable.
"""
import numpy as np
import pandas as pd
import pytest

from services import model_runner
from services.model_runner import run_all_models


def _model(name, rmse, train_time=0.01, pred_time=0.001, mae=None, mape=0.05, r2=0.8):
    """A model returning fixed metrics, so ranking is fully determined."""
    def _fn(series, horizon):
        return {
            "model_name": name, "rmse": rmse,
            "mae": mae if mae is not None else rmse * 0.8,
            "mape": mape, "r2": r2,
            "forecast": [float(series.iloc[-1])] * horizon,
            "train_time_sec": train_time, "pred_time_sec": pred_time,
        }
    return _fn


@pytest.fixture
def series():
    # 600 rows keeps us above the 500-row daily DL gate, so a registry tagged
    # "dl" is not filtered out before ranking is exercised.
    return pd.Series(
        np.random.RandomState(0).randn(600) + 100,
        index=pd.date_range("2023-01-01", periods=600, freq="D"),
    )


@pytest.fixture
def registry():
    """Mirrors the real cost profile: a few slow, accurate statistical models
    and many fast, less accurate ones. This is the composition under which the
    old speed term inverted the ranking."""
    return {
        "SARIMA":        (_model("SARIMA", 1100.0, train_time=118.0, pred_time=1.10), 200, "statistical"),
        "SARIMAX":       (_model("SARIMAX", 1140.0, train_time=112.0, pred_time=1.09), 200, "statistical"),
        "XGBoost":       (_model("XGBoost", 1250.0, train_time=0.42), 200, "ml"),
        "KNN":           (_model("KNN", 1300.0, train_time=0.0001, pred_time=0.20), 200, "ml"),
        "MovingAverage": (_model("MovingAverage", 1290.0, train_time=0.00005), 200, "statistical"),
    }


@pytest.fixture(autouse=True)
def _restore_budget():
    """Every test sets the budget explicitly; restore the shipped default after,
    so ordering between tests cannot leak."""
    original = model_runner.SELECTION_TIME_BUDGET_SEC
    yield
    model_runner.SELECTION_TIME_BUDGET_SEC = original


# ── Default policy: accuracy only ────────────────────────────────────────────

def test_default_ranks_by_cross_validated_rmse(series, registry):
    """The regression this whole change exists to prevent: the most accurate
    model must win, even when it is by far the slowest to train."""
    model_runner.SELECTION_TIME_BUDGET_SEC = None
    results, _ = run_all_models(series, forecast_horizon=10, models=registry)

    assert [r["model_name"] for r in results] == [
        "SARIMA", "SARIMAX", "XGBoost", "MovingAverage", "KNN",
    ]
    # And that order is exactly RMSE order.
    assert [r["rmse"] for r in results] == sorted(r["rmse"] for r in results)


def test_slow_accurate_model_is_not_penalised_for_being_slow(series, registry):
    """SARIMA trains ~1.2 million times slower than MovingAverage. Under the old
    score that alone demoted it; it must not now."""
    model_runner.SELECTION_TIME_BUDGET_SEC = None
    results, _ = run_all_models(series, forecast_horizon=10, models=registry)
    by_name = {r["model_name"]: r for r in results}

    assert by_name["SARIMA"]["train_time_sec"] > by_name["MovingAverage"]["train_time_sec"] * 1e6
    assert by_name["SARIMA"]["adjusted_score"] < by_name["MovingAverage"]["adjusted_score"]


def test_default_budget_is_none(series, registry):
    """The useful budget is dataset-dependent and cannot be inferred at request
    time, so the shipped default must impose none."""
    assert model_runner.SELECTION_TIME_BUDGET_SEC is None


# ── Budget constraint ────────────────────────────────────────────────────────

def test_budget_excludes_models_that_exceed_it(series, registry):
    model_runner.SELECTION_TIME_BUDGET_SEC = 1.5
    results, _ = run_all_models(series, forecast_horizon=10, models=registry)
    by_name = {r["model_name"]: r for r in results}

    assert by_name["SARIMA"]["within_time_budget"] is False
    assert by_name["SARIMAX"]["within_time_budget"] is False
    assert by_name["XGBoost"]["within_time_budget"] is True
    # The winner must be the most accurate AFFORDABLE model, not the most
    # accurate overall.
    assert results[0]["model_name"] == "XGBoost"


def test_over_budget_models_rank_below_every_affordable_one(series, registry):
    model_runner.SELECTION_TIME_BUDGET_SEC = 1.5
    results, _ = run_all_models(series, forecast_horizon=10, models=registry)

    affordable = [r["adjusted_score"] for r in results if r["within_time_budget"]]
    excluded = [r["adjusted_score"] for r in results if not r["within_time_budget"]]
    assert affordable and excluded
    assert max(affordable) < min(excluded)


def test_over_budget_offset_survives_a_catastrophic_outlier(series):
    """A fixed offset would fail here: the outlier's normalized RMSE alone
    exceeds any constant, so an affordable-but-terrible model could outrank a
    cheap good one only by accident. The offset must scale with the observed
    spread."""
    model_runner.SELECTION_TIME_BUDGET_SEC = 1.0
    registry = {
        "Catastrophic": (_model("Catastrophic", 823_348_720.0, train_time=0.01), 200, "statistical"),
        "SlowGood":     (_model("SlowGood", 1000.0, train_time=90.0), 200, "statistical"),
        "FastOkay":     (_model("FastOkay", 1200.0, train_time=0.02), 200, "ml"),
    }
    results, _ = run_all_models(series, forecast_horizon=10, models=registry)

    affordable = [r["adjusted_score"] for r in results if r["within_time_budget"]]
    excluded = [r["adjusted_score"] for r in results if not r["within_time_budget"]]
    assert max(affordable) < min(excluded), (
        "an over-budget model outranked an affordable one")
    # Within the affordable set, accuracy still decides.
    assert results[0]["model_name"] == "FastOkay"


def test_budget_that_admits_everything_matches_no_budget(series, registry):
    model_runner.SELECTION_TIME_BUDGET_SEC = None
    unconstrained, _ = run_all_models(series, forecast_horizon=10, models=registry)

    model_runner.SELECTION_TIME_BUDGET_SEC = 1000.0
    generous, _ = run_all_models(series, forecast_horizon=10, models=registry)

    assert ([r["model_name"] for r in unconstrained]
            == [r["model_name"] for r in generous])
    assert all(r["within_time_budget"] for r in generous)


def test_impossible_budget_falls_back_to_the_cheapest_model(series, registry):
    """Nothing fits. The caller asked for a latency bound, so the closest
    honouring of it is the cheapest model — matching the benchmarked policy
    exactly rather than silently ignoring the budget."""
    model_runner.SELECTION_TIME_BUDGET_SEC = 1e-9
    results, _ = run_all_models(series, forecast_horizon=10, models=registry)

    assert results[0]["model_name"] == "MovingAverage"   # cheapest at 5e-05s
    assert sum(r["within_time_budget"] for r in results) == 1


# ── Preserved behaviour ──────────────────────────────────────────────────────

def test_stability_is_still_reported_but_no_longer_scored(series, registry):
    """Stability remains useful diagnostic output. It must not affect ranking:
    its discontinuity term penalises models for not anchoring to the last
    observation, which rewards persistence over genuine forecasting."""
    model_runner.SELECTION_TIME_BUDGET_SEC = None
    results, _ = run_all_models(series, forecast_horizon=10, models=registry)

    assert all("stability_penalty" in r for r in results)
    # Ranking is pure RMSE order, so it cannot have been perturbed by stability.
    assert [r["rmse"] for r in results] == sorted(r["rmse"] for r in results)


def test_failed_models_still_sink_to_the_bottom(series):
    def _broken(s, h):
        raise RuntimeError("model exploded")

    model_runner.SELECTION_TIME_BUDGET_SEC = None
    registry = {
        "Good":   (_model("Good", 100.0), 200, "ml"),
        "Broken": (_broken, 200, "ml"),
    }
    results, _ = run_all_models(series, forecast_horizon=10, models=registry)

    assert results[0]["model_name"] == "Good"
    assert results[-1]["model_name"] == "Broken"
    assert results[-1]["status"] == "failed"
    assert not np.isfinite(results[-1]["adjusted_score"])


def test_select_best_still_consumes_adjusted_score(series, registry):
    """best_model.select_best picks the lowest adjusted_score. The ranking key
    changed meaning, so confirm the downstream contract still holds."""
    from services.best_model import select_best

    model_runner.SELECTION_TIME_BUDGET_SEC = None
    results, _ = run_all_models(series, forecast_horizon=10, models=registry)
    best = select_best(results, series=series)

    assert best["best_model_name"] == "SARIMA"
    assert best["rank"] == 1
    assert best["error"] is None
