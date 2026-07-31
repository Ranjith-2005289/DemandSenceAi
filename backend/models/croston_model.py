"""
croston_model.py — Croston's method & SBA (Syntetos-Boylan Approximation) for
intermittent/sparse demand: many zero-demand periods with occasional spikes
(slow-moving SKUs, spare parts, low-frequency items). None of the other 17
models in this registry are built for this pattern — ARIMA/ETS-family models
assume roughly continuous demand, and tree/DL models trained on zero-inflated
data tend to just predict near-zero everywhere, which technically minimizes
RMSE on a mostly-zero series but is useless for reorder-point planning.

Croston's method decomposes the series into two independently-smoothed
components — the SIZE of a demand event when one occurs, and the INTERVAL (in
periods) between demand events — and forecasts a constant *rate*
(size / interval) rather than a per-period curve, since there's no seasonal
or trend signal to extract from mostly-zero data.

SBA (Syntetos-Boylan Approximation) is Croston's method with a small bias
correction — Croston's original formula is known to systematically
over-forecast; SBA is the standard, more accurate fix and is the default here.

This model naturally ranks low on dense/continuous datasets (as it should —
a constant-rate forecast is a poor fit when there's real trend/seasonality
to capture) and naturally ranks well on genuinely sparse ones, via the same
leaderboard RMSE comparison every other model goes through — no special-
casing needed to decide when it "applies."
"""
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

from services.metrics import calculate_metrics, get_cv_splits

MIN_OBS = 10


def _croston_rate(values: np.ndarray, alpha: float = 0.1, variant: str = "sba") -> tuple[float, float]:
    """
    Fit Croston/SBA on a 1D array of non-negative demand values.

    Returns (forecast_rate, smoothed_interval) — forecast_rate is the constant
    per-period forecast Croston produces (the same value repeats for every
    future period, since there's no other structure to extract).
    """
    values = np.asarray(values, dtype=float)
    nonzero_idx = np.flatnonzero(values > 0)

    if len(nonzero_idx) == 0:
        return 0.0, 0.0
    if len(nonzero_idx) == 1:
        # Only one non-zero observation ever seen — spread it over however
        # many periods have passed since, rather than fabricating a smoothed
        # trajectory out of a single point.
        size = values[nonzero_idx[0]]
        interval = len(values) - nonzero_idx[0]
        return float(size / max(interval, 1)), 0.0

    sizes = values[nonzero_idx]
    intervals = np.diff(nonzero_idx, prepend=-1)  # gap since the previous demand (or series start)

    z_hat = sizes[0]
    q_hat = intervals[0]
    for i in range(1, len(sizes)):
        z_hat = alpha * sizes[i] + (1 - alpha) * z_hat
        q_hat = alpha * intervals[i] + (1 - alpha) * q_hat

    correction = (1 - alpha / 2) if variant == "sba" else 1.0
    rate = correction * (z_hat / q_hat) if q_hat > 0 else 0.0
    return float(rate), float(q_hat)


def train_and_forecast(series: pd.Series, forecast_horizon: int, variant: str = "sba") -> dict:
    """
    Croston/SBA intermittent-demand forecast — a constant rate, not a curve.

    Args:
        series           : pd.Series, non-negative, already cleaned (no NaNs).
        forecast_horizon : int — number of future periods to forecast.
        variant           : "sba" (default, bias-corrected) or "croston" (original).

    Returns:
        {
            "model_name": "CrostonSBA" | "Croston",
            "rmse"/"mae"/"mape"/"r2": float,
            "forecast": list[float],           constant rate, repeated
            "avg_demand_interval": float,        smoothed periods-between-demand
            "zero_period_fraction": float,       context for how sparse this series is
        }
    """
    series = series.dropna().astype(float)
    if len(series) < MIN_OBS:
        raise ValueError(f"Croston/SBA needs at least {MIN_OBS} observations, got {len(series)}.")
    if float(series.min()) < 0:
        raise ValueError("Croston/SBA is defined for non-negative demand only.")

    alpha = 0.1
    values = series.values

    # ── Walk-forward CV: a fold's forecast is one constant rate, evaluated
    #    against every point in that fold's held-out window ─────────────────
    n_splits = get_cv_splits(len(values))
    tscv = TimeSeriesSplit(n_splits=n_splits)
    cv_metrics, train_times, pred_times = [], [], []

    for train_idx, test_idx in tscv.split(values):
        train_vals = values[train_idx]
        test_vals = values[test_idx]

        t0 = time.perf_counter()
        rate, _ = _croston_rate(train_vals, alpha=alpha, variant=variant)
        train_times.append(time.perf_counter() - t0)

        t1 = time.perf_counter()
        preds = [rate] * len(test_vals)
        pred_times.append(time.perf_counter() - t1)

        cv_metrics.append(calculate_metrics(test_vals, preds))

    avg_metrics = {
        k: sum(m[k] for m in cv_metrics) / len(cv_metrics)
        for k in cv_metrics[0].keys()
    }
    avg_train_time = sum(train_times) / len(train_times)
    avg_pred_time = sum(pred_times) / len(pred_times)

    # ── Refit on the full series for the final forecast ───────────────────
    final_rate, q_hat = _croston_rate(values, alpha=alpha, variant=variant)
    forecast_values = [final_rate] * forecast_horizon
    zero_fraction = float((values == 0).mean())

    return {
        "model_name": "CrostonSBA" if variant == "sba" else "Croston",
        **avg_metrics,
        "train_time_sec": round(avg_train_time, 4),
        "pred_time_sec": round(avg_pred_time, 4),
        "forecast": [round(float(v), 4) for v in forecast_values],
        "avg_demand_interval": round(q_hat, 4),
        "zero_period_fraction": round(zero_fraction, 4),
    }
