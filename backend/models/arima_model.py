"""
arima_model.py — Auto ARIMA forecasting model
Tries all (p,1,q) orders from 1–3, picks best AIC, forecasts future horizon.
"""

import warnings
import itertools
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
import time
from services.metrics import calculate_metrics, get_cv_splits

warnings.filterwarnings("ignore")

# Cap evaluation (order-search + CV) cost so it stays bounded regardless of
# how large the input series is. The final production forecast still refits
# on the complete series — this only bounds the cost of deciding which order
# is best and computing the reported CV metric.
MAX_EVAL_ROWS = 2000


# ── Helpers ───────────────────────────────────────────────────────────────────

def _check_stationarity(series: pd.Series) -> int:
    """
    Run ADF test. Return differencing order d.
    d=0 if already stationary, d=1 otherwise (covers most real-world retail data).
    """
    try:
        result = adfuller(series.dropna(), autolag="AIC")
        p_value = result[1]
        return 0 if p_value < 0.05 else 1
    except Exception:
        return 1


def _select_best_order(
    train: pd.Series,
    d: int,
    p_range: range,
    q_range: range,
) -> tuple[int, int, int]:
    """
    Grid-search (p, d, q) orders and return the one with the lowest AIC.
    Falls back to (1,1,1) if all fits fail.
    """
    best_aic = np.inf
    best_order = (1, d, 1)

    candidates = list(itertools.product(p_range, [d], q_range))

    for order in candidates:
        try:
            model = ARIMA(train, order=order)
            fit = model.fit(method_kwargs={"warn_convergence": False})
            if fit.aic < best_aic:
                best_aic = fit.aic
                best_order = order
        except Exception:
            continue

    return best_order


def _add_seasonal_differencing(series: pd.Series, period: int = 7) -> pd.Series:
    """Apply seasonal differencing to remove weekly patterns if present."""
    return series.diff(period).dropna()


def _make_stationary(series: pd.Series) -> tuple[pd.Series, int, bool]:
    """
    Attempt to make series stationary.
    Returns (processed_series, d, applied_seasonal_diff).
    """
    d = _check_stationarity(series)

    # Check for weekly seasonality in daily data
    seasonal_diff = False
    if len(series) >= 14:
        try:
            seasonal_check = adfuller(series.diff(7).dropna(), autolag="AIC")
            if seasonal_check[1] < 0.05 and d == 1:
                seasonal_diff = True
        except Exception:
            pass

    return series, d, seasonal_diff


# ── Main function ─────────────────────────────────────────────────────────────

def train_and_forecast(
    train_series: pd.Series,
    forecast_horizon: int,
) -> dict:
    """
    Fit the best ARIMA model on train_series and forecast `forecast_horizon` steps ahead.

    Args:
        train_series : pd.Series with a DatetimeIndex, already cleaned (no NaNs).
        forecast_horizon : int — number of future periods to forecast.

    Returns:
        {
            "model_name": "ARIMA",
            "rmse": float,
            "forecast": list[float],
            "best_order": str,       e.g. "(2, 1, 1)"
            "aic": float,
        }
    """
    series = train_series.copy().astype(float)

    # ── 1. Enforce minimum length ─────────────────────────────────────────────
    if len(series) < 15:
        raise ValueError(
            f"ARIMA needs at least 15 observations, got {len(series)}."
        )

    # Bound evaluation cost to a fixed recent window regardless of total
    # series length (see MAX_EVAL_ROWS above).
    eval_series = series.iloc[-MAX_EVAL_ROWS:] if len(series) > MAX_EVAL_ROWS else series

    tscv = TimeSeriesSplit(n_splits=get_cv_splits(len(eval_series)))
    cv_metrics = []
    train_times = []
    pred_times = []

    # ── 3. Find global best order on initial 80% to save time ────────────────
    split_idx = int(len(eval_series) * 0.8)
    split_idx = max(split_idx, 10)
    train_init = eval_series.iloc[:split_idx]
    _, d, _ = _make_stationary(train_init)
    best_order = _select_best_order(train_init, d, range(1, 4), range(1, 4))

    # ── 4. Cross-validate ────────────────────────────────────────────────────
    # Walk-forward validation: fit once per fold, then cheaply extend the
    # fitted state with each new true observation via .append(refit=False)
    # instead of re-estimating the model from scratch at every test point —
    # mathematically near-identical one-step-ahead predictions, ~10x+ faster,
    # and the gap widens further as the test fold grows.
    for train_idx, test_idx in tscv.split(eval_series):
        train, test = eval_series.iloc[train_idx], eval_series.iloc[test_idx]
        predictions = []

        t0 = time.perf_counter()
        try:
            m = ARIMA(list(train), order=best_order).fit(method_kwargs={"warn_convergence": False})
        except Exception:
            m = None
        train_times.append(time.perf_counter() - t0)

        t1 = time.perf_counter()
        for i in range(len(test)):
            try:
                if m is None:
                    raise RuntimeError("no fitted model")
                yhat = float(np.asarray(m.forecast(steps=1))[0])
                m = m.append([test.iloc[i]], refit=False)
            except Exception:
                yhat = float(train.iloc[-1]) if i == 0 else predictions[-1]
            predictions.append(yhat)
        pred_times.append(time.perf_counter() - t1)

        metrics = calculate_metrics(test.values, predictions)
        cv_metrics.append(metrics)

    avg_metrics = {k: sum(m[k] for m in cv_metrics)/len(cv_metrics) for k in cv_metrics[0].keys()}
    avg_train_time = sum(train_times)/len(train_times)
    avg_pred_time = sum(pred_times)/len(pred_times)

    # ── 6. Refit on FULL series for final forecast ────────────────────────────
    final_model = ARIMA(series, order=best_order)
    final_fit   = final_model.fit(method_kwargs={"warn_convergence": False})
    forecast_raw = final_fit.forecast(steps=forecast_horizon)

    # Clip negatives for demand/sales data (values can't be negative)
    series_min = float(series.min())
    floor = 0.0 if series_min >= 0 else series_min
    forecast_values = [max(floor, float(v)) for v in forecast_raw]

    return {
        "model_name": "ARIMA",
        **avg_metrics,
        "train_time_sec": round(avg_train_time, 4),
        "pred_time_sec": round(avg_pred_time, 4),
        "forecast": [round(v, 4) for v in forecast_values],
        "best_order": str(best_order),
        "aic": round(final_fit.aic, 2),
    }