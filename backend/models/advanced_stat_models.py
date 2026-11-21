"""
advanced_stat_models.py — Theta, AutoETS, MSTL and SeasonalNaive.

These were added after a held-out benchmark (every model trained without the
last 28 points of 10 series and scored on them by MASE) showed that the
seven deep-learning models and KNN were, on average, *worse than a
seasonal-naive forecast*, while the exponential-smoothing / state-space
family consistently led. The models here are the strongest well-established
members of that family:

  Theta         Assimakopoulos & Nikolopoulos (2000); winner of the M3
                competition and still a standard strong benchmark.
  AutoETS       Hyndman et al. state-space ETS: error/trend/damping/seasonal
                form chosen by AICc instead of by a fixed grid on CV RMSE.
  MSTL          Multiple-seasonal STL decomposition (e.g. weekly AND yearly
                in daily retail data — something a single-period
                Holt-Winters cannot represent), with a damped-trend ETS on
                the seasonally adjusted series.
  SeasonalNaive "Same as last season". Not expected to win often, but it
                is the yardstick MASE is defined against, so keeping it on
                the leaderboard makes "did we beat the naive?" visible.

Every function follows the registry contract: fn(series, horizon) -> dict
with "forecast" plus internal multi-step CV metrics. model_runner replaces
those metrics with a common holdout backtest whenever the series is long
enough for one, so all models are ranked on identical windows.
"""

from __future__ import annotations

import time
import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

from services.metrics import calculate_metrics, get_cv_splits
from models.sarima_model import _infer_seasonal_period

warnings.filterwarnings("ignore")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _usable_period(series: pd.Series, min_cycles: int = 2) -> int:
    """Seasonal period, or 1 (non-seasonal) if there aren't enough cycles."""
    m = _infer_seasonal_period(series)
    return m if m > 1 and len(series) >= min_cycles * m else 1


def _floor(series: pd.Series, values) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if float(series.min()) >= 0:
        arr = np.clip(arr, 0, None)
    return arr


def _multistep_cv(series: pd.Series, forecaster) -> tuple[dict, float, float]:
    """
    Walk-forward CV where each fold forecasts its whole test window from the
    end of its training window — the same multi-step setting the model is
    used in. Returns (avg_metrics, avg_train_time, avg_pred_time).
    """
    tscv = TimeSeriesSplit(n_splits=get_cv_splits(len(series)))
    metrics, t_fit = [], []
    for tr, te in tscv.split(series):
        train, test = series.iloc[tr], series.iloc[te]
        if len(train) < 10:
            continue
        t0 = time.perf_counter()
        try:
            pred = forecaster(train, len(test))
        except Exception:
            continue
        t_fit.append(time.perf_counter() - t0)
        metrics.append(calculate_metrics(test.values, _floor(series, pred)))
    if not metrics:
        return {"rmse": float("inf"), "mae": float("inf"), "mape": float("inf"), "r2": float("-inf")}, 0.0, 0.0
    avg = {k: float(np.mean([m[k] for m in metrics])) for k in metrics[0]}
    return avg, float(np.mean(t_fit)), 0.0


def _wrap(name: str, series: pd.Series, horizon: int, forecaster, **extra) -> dict:
    series = series.dropna().astype(float)
    if len(series) < 10:
        raise ValueError(f"{name} needs at least 10 observations, got {len(series)}.")
    avg, t_train, t_pred = _multistep_cv(series, forecaster)
    t0 = time.perf_counter()
    fc = _floor(series, forecaster(series, horizon))
    t_final = time.perf_counter() - t0
    if not np.all(np.isfinite(fc)):
        raise ValueError(f"{name} produced a non-finite forecast.")
    return {
        "model_name": name,
        **avg,
        "train_time_sec": round(t_train or t_final, 4),
        "pred_time_sec": round(t_pred, 4),
        "forecast": [round(float(v), 4) for v in fc],
        **extra,
    }


# ── Theta ─────────────────────────────────────────────────────────────────────

def _theta_forecast(series: pd.Series, horizon: int) -> np.ndarray:
    from statsmodels.tsa.forecasting.theta import ThetaModel

    m = _usable_period(series)
    y = series.reset_index(drop=True)
    # Multiplicative deseasonalisation needs strictly positive data.
    method = "multiplicative" if float(series.min()) > 0 else "additive"
    model = ThetaModel(
        y, period=m if m > 1 else None, deseasonalize=m > 1, method=method,
    ).fit()
    return np.asarray(model.forecast(horizon), dtype=float)


def train_and_forecast_theta(series: pd.Series, forecast_horizon: int) -> dict:
    return _wrap("Theta", series, forecast_horizon, _theta_forecast,
                 seasonal_period=_usable_period(series))


# ── AutoETS ───────────────────────────────────────────────────────────────────

def _ets_candidates(series: pd.Series, m: int) -> list[dict]:
    positive = float(series.min()) > 0
    trends = [(None, False), ("add", False), ("add", True)]
    seasonals = [None]
    if m > 1:
        seasonals.append("add")
        if positive:
            seasonals.append("mul")
    errors = ["add", "mul"] if positive else ["add"]
    out = []
    for err in errors:
        for trend, damped in trends:
            for seas in seasonals:
                out.append({"error": err, "trend": trend, "damped_trend": damped,
                            "seasonal": seas})
    return out


def _fit_best_ets(series: pd.Series):
    from statsmodels.tsa.exponential_smoothing.ets import ETSModel

    m = _usable_period(series)
    # A 365-long seasonal state vector makes ETS slow and unstable; MSTL is
    # the model meant for long periods.
    if m > 60:
        m = 1
    y = series.reset_index(drop=True)
    best, best_cfg, best_aicc = None, None, np.inf
    for cfg in _ets_candidates(series, m):
        try:
            fit = ETSModel(
                y, error=cfg["error"], trend=cfg["trend"],
                damped_trend=cfg["damped_trend"], seasonal=cfg["seasonal"],
                seasonal_periods=m if cfg["seasonal"] else None,
                initialization_method="estimated",
            ).fit(disp=False, maxiter=200)
            aicc = float(fit.aicc)
            if np.isfinite(aicc) and aicc < best_aicc:
                best, best_cfg, best_aicc = fit, cfg, aicc
        except Exception:
            continue
    if best is None:
        raise RuntimeError("No ETS configuration could be fitted.")
    return best, best_cfg, m


def train_and_forecast_autoets(series: pd.Series, forecast_horizon: int) -> dict:
    series = series.dropna().astype(float)
    chosen: dict = {}

    def _ets_forecast(s: pd.Series, horizon: int) -> np.ndarray:
        fit, cfg, m = _fit_best_ets(s)
        chosen.update(cfg=cfg, m=m)  # last call is the full-series fit
        return np.asarray(fit.forecast(horizon), dtype=float)

    result = _wrap("AutoETS", series, forecast_horizon, _ets_forecast)
    cfg = chosen.get("cfg")
    if cfg:
        e = "A" if cfg["error"] == "add" else "M"
        t = "N" if cfg["trend"] is None else ("Ad" if cfg["damped_trend"] else "A")
        s = "N" if cfg["seasonal"] is None else ("A" if cfg["seasonal"] == "add" else "M")
        result["ets_form"] = f"ETS({e},{t},{s})"
        result["seasonal_period"] = chosen["m"]
    return result


# ── MSTL ──────────────────────────────────────────────────────────────────────

def _mstl_periods(series: pd.Series) -> list[int]:
    """Seasonal periods worth modelling given the frequency and history."""
    base = _infer_seasonal_period(series)
    n = len(series)
    periods = []
    if base > 1 and n >= 2 * base:
        periods.append(base)
    if base == 7 and n >= 2 * 365:      # daily → also yearly
        periods.append(365)
    if base == 24 and n >= 2 * 168:     # hourly → also weekly
        periods.append(168)
    return periods


def _mstl_forecast(series: pd.Series, horizon: int) -> np.ndarray:
    from statsmodels.tsa.seasonal import MSTL
    from statsmodels.tsa.holtwinters import ExponentialSmoothing

    periods = _mstl_periods(series)
    y = series.reset_index(drop=True).astype(float)
    if not periods:
        # Nothing seasonal to decompose — damped-trend smoothing alone.
        fit = ExponentialSmoothing(y, trend="add", damped_trend=True,
                                   initialization_method="estimated").fit()
        return np.asarray(fit.forecast(horizon), dtype=float)

    res = MSTL(y, periods=periods).fit()
    seasonal = res.seasonal
    if isinstance(seasonal, pd.Series):
        seasonal = seasonal.to_frame()
    adjusted = y - seasonal.sum(axis=1)

    fit = ExponentialSmoothing(adjusted, trend="add", damped_trend=True,
                               initialization_method="estimated").fit()
    fc = np.asarray(fit.forecast(horizon), dtype=float)

    # Each seasonal component continues by repeating its last full cycle.
    for col, p in zip(seasonal.columns, periods):
        last_cycle = seasonal[col].values[-p:]
        reps = int(np.ceil(horizon / p)) + 1
        fc = fc + np.tile(last_cycle, reps)[:horizon]
    return fc


def train_and_forecast_mstl(series: pd.Series, forecast_horizon: int) -> dict:
    return _wrap("MSTL", series, forecast_horizon, _mstl_forecast,
                 seasonal_periods=_mstl_periods(series.dropna()))


# ── Seasonal naive ────────────────────────────────────────────────────────────

def _snaive_forecast(series: pd.Series, horizon: int) -> np.ndarray:
    m = _usable_period(series, min_cycles=1)
    last = series.values[-m:]
    return np.tile(last, int(np.ceil(horizon / m)) + 1)[:horizon].astype(float)


def train_and_forecast_snaive(series: pd.Series, forecast_horizon: int) -> dict:
    return _wrap("SeasonalNaive", series, forecast_horizon, _snaive_forecast,
                 seasonal_period=_usable_period(series.dropna(), min_cycles=1))
