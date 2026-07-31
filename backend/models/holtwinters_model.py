"""
holtwinters_model.py — Holt-Winters & Exponential Smoothing models
Auto-selects additive vs multiplicative seasonality.
"""

import warnings
import numpy as np
import pandas as pd
from math import sqrt
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
import time
from services.metrics import calculate_metrics, get_cv_splits
from statsmodels.tsa.holtwinters import ExponentialSmoothing, SimpleExpSmoothing

warnings.filterwarnings("ignore")


def _infer_period(series: pd.Series) -> int:
    if not isinstance(series.index, pd.DatetimeIndex):
        return 7
    try:
        freq = (pd.infer_freq(series.index) or "D").upper()
        if "D" in freq: return 7
        if "W" in freq: return 52
        if "M" in freq: return 12
        if "Q" in freq: return 4
        if "H" in freq: return 24
    except Exception:
        pass
    return 7


# ── Holt-Winters ──────────────────────────────────────────────────────────────

def train_and_forecast(series: pd.Series, forecast_horizon: int) -> dict:
    """
    Holt-Winters Triple Exponential Smoothing.
    Auto trend + seasonality mode selection.
    """
    series = series.dropna().astype(float)
    if len(series) < 10:
        raise ValueError(f"Holt-Winters needs ≥ 10 observations, got {len(series)}.")

    m = _infer_period(series)

    # Need at least 2 full seasonal cycles for seasonal model
    use_seasonal = len(series) >= m * 2

    tscv = TimeSeriesSplit(n_splits=get_cv_splits(len(series)))
    cv_metrics = []
    train_times = []
    pred_times = []

    best_rmse   = np.inf
    best_params = {}
    
    trend_opts  = ["add", "mul"] if series.min() > 0 else ["add"]
    season_opts = ["add", "mul"] if (use_seasonal and series.min() > 0) else (["add"] if use_seasonal else [None])

    for trend in trend_opts:
        for seasonal in season_opts:
            fold_rmses = []
            for train_idx, test_idx in tscv.split(series):
                train, test = series.iloc[train_idx], series.iloc[test_idx]
                try:
                    hw = ExponentialSmoothing(
                        train,
                        trend=trend,
                        seasonal=seasonal,
                        seasonal_periods=m if seasonal else None,
                        damped_trend=True,
                        initialization_method="estimated",
                    ).fit(optimized=True, remove_bias=True)

                    pred   = hw.forecast(len(test))
                    floor  = 0.0 if series.min() >= 0 else None
                    if floor is not None:
                        pred = np.clip(pred, floor, None)
                    fold_rmses.append(sqrt(mean_squared_error(test.values, pred.values)))
                except Exception:
                    continue
            if fold_rmses:
                avg_rmse = np.mean(fold_rmses)
                if avg_rmse < best_rmse:
                    best_rmse   = avg_rmse
                    best_params = {"trend": trend, "seasonal": seasonal}

    if best_params == {}:
        best_params = {"trend": "add", "seasonal": None}

    for train_idx, test_idx in tscv.split(series):
        train, test = series.iloc[train_idx], series.iloc[test_idx]
        try:
            t0 = time.perf_counter()
            hw = ExponentialSmoothing(
                train,
                trend=best_params["trend"],
                seasonal=best_params["seasonal"],
                seasonal_periods=m if best_params["seasonal"] else None,
                damped_trend=True,
                initialization_method="estimated",
            ).fit(optimized=True, remove_bias=True)
            train_times.append(time.perf_counter() - t0)

            t1 = time.perf_counter()
            pred   = hw.forecast(len(test))
            pred_times.append(time.perf_counter() - t1)
            
            floor  = 0.0 if series.min() >= 0 else None
            if floor is not None:
                pred = np.clip(pred, floor, None)
            
            metrics = calculate_metrics(test.values, pred.values)
            cv_metrics.append(metrics)
        except Exception:
            continue

    if cv_metrics:
        avg_metrics = {k: sum(m[k] for m in cv_metrics)/len(cv_metrics) for k in cv_metrics[0].keys()}
        avg_train_time = sum(train_times)/len(train_times)
        avg_pred_time = sum(pred_times)/len(pred_times)
    else:
        avg_metrics = {"rmse": best_rmse, "mae": 0.0, "mape": 0.0, "r2": 0.0}
        avg_train_time = 0.0
        avg_pred_time = 0.0

    if best_params == {}:
        best_params = {"trend": "add", "seasonal": None}

    # Refit on full series
    final = ExponentialSmoothing(
        series,
        trend=best_params["trend"],
        seasonal=best_params["seasonal"],
        seasonal_periods=m if best_params["seasonal"] else None,
        damped_trend=True,
        initialization_method="estimated",
    ).fit(optimized=True, remove_bias=True)

    floor = 0.0 if series.min() >= 0 else None
    fc    = final.forecast(forecast_horizon)
    if floor is not None:
        fc = np.clip(fc, floor, None)

    return {
        "model_name":      "HoltWinters",
        **avg_metrics,
        "train_time_sec": round(avg_train_time, 4),
        "pred_time_sec": round(avg_pred_time, 4),
        "forecast":        [round(float(v), 4) for v in fc],
        "trend_mode":      best_params["trend"],
        "seasonal_mode":   str(best_params["seasonal"]),
        "seasonal_period": m,
    }


# ── Simple Exponential Smoothing ──────────────────────────────────────────────

def train_and_forecast_exp(series: pd.Series, forecast_horizon: int) -> dict:
    """
    Exponential Smoothing — optimised alpha, with Holt trend if data is trending.
    """
    series = series.dropna().astype(float)
    if len(series) < 5:
        raise ValueError(f"Exponential Smoothing needs ≥ 5 observations, got {len(series)}.")

    tscv = TimeSeriesSplit(n_splits=get_cv_splits(len(series)))
    cv_metrics = []
    train_times = []
    pred_times = []

    best_rmse, best_fc_full = np.inf, None
    best_name = "ExponentialSmoothing"
    best_use_trend = False

    for use_trend in [True, False]:
        fold_rmses = []
        for train_idx, test_idx in tscv.split(series):
            train, test = series.iloc[train_idx], series.iloc[test_idx]
            try:
                if use_trend:
                    model = ExponentialSmoothing(
                        train, trend="add", damped_trend=True,
                        initialization_method="estimated"
                    ).fit(optimized=True)
                else:
                    model = SimpleExpSmoothing(train, initialization_method="estimated").fit(optimized=True)

                pred  = model.forecast(len(test))
                floor = 0.0 if series.min() >= 0 else None
                if floor is not None:
                    pred = np.clip(pred, floor, None)
                fold_rmses.append(sqrt(mean_squared_error(test.values[:len(pred)], pred[:len(test)])))
            except Exception:
                continue
        if fold_rmses:
            avg_rmse = np.mean(fold_rmses)
            if avg_rmse < best_rmse:
                best_rmse    = avg_rmse
                best_name    = "HoltExponentialSmoothing" if use_trend else "ExponentialSmoothing"
                best_use_trend = use_trend

    for train_idx, test_idx in tscv.split(series):
        train, test = series.iloc[train_idx], series.iloc[test_idx]
        try:
            t0 = time.perf_counter()
            if best_use_trend:
                model = ExponentialSmoothing(
                    train, trend="add", damped_trend=True,
                    initialization_method="estimated"
                ).fit(optimized=True)
            else:
                model = SimpleExpSmoothing(train, initialization_method="estimated").fit(optimized=True)
            train_times.append(time.perf_counter() - t0)

            t1 = time.perf_counter()
            pred  = model.forecast(len(test))
            pred_times.append(time.perf_counter() - t1)
            
            floor = 0.0 if series.min() >= 0 else None
            if floor is not None:
                pred = np.clip(pred, floor, None)
            
            metrics = calculate_metrics(test.values[:len(pred)], pred[:len(test)])
            cv_metrics.append(metrics)
            best_fc_full = model
        except Exception:
            continue

    if cv_metrics:
        avg_metrics = {k: sum(m[k] for m in cv_metrics)/len(cv_metrics) for k in cv_metrics[0].keys()}
        avg_train_time = sum(train_times)/len(train_times)
        avg_pred_time = sum(pred_times)/len(pred_times)
    else:
        avg_metrics = {"rmse": best_rmse, "mae": 0.0, "mape": 0.0, "r2": 0.0}
        avg_train_time = 0.0
        avg_pred_time = 0.0

    if best_fc_full is None:
        raise RuntimeError("All Exponential Smoothing variants failed.")

    # Refit winner on full series
    if best_name == "HoltExponentialSmoothing":
        final = ExponentialSmoothing(
            series, trend="add", damped_trend=True,
            initialization_method="estimated"
        ).fit(optimized=True)
    else:
        final = SimpleExpSmoothing(series, initialization_method="estimated").fit(optimized=True)

    fc    = final.forecast(forecast_horizon)
    floor = 0.0 if series.min() >= 0 else None
    if floor is not None:
        fc = np.clip(fc, floor, None)

    return {
        "model_name": best_name,
        **avg_metrics,
        "train_time_sec": round(avg_train_time, 4),
        "pred_time_sec": round(avg_pred_time, 4),
        "forecast":   [round(float(v), 4) for v in fc],
        "alpha":      round(float(final.params.get("smoothing_level", 0)), 4),
    }
