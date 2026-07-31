"""
prophet_model.py — Facebook Prophet forecasting model
Yearly + weekly seasonality, robust to outliers, handles missing dates automatically.
"""

import logging
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
import time
from services.metrics import calculate_metrics, get_cv_splits

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

try:
    from prophet import Prophet
except ImportError:
    from fbprophet import Prophet  # older installs


# ── Helpers ───────────────────────────────────────────────────────────────────

def _infer_frequency(series: pd.Series) -> str:
    """Infer pandas frequency string from a DatetimeIndex."""
    if not isinstance(series.index, pd.DatetimeIndex):
        return "D"
    try:
        freq = pd.infer_freq(series.index)
        if freq is None:
            gaps = series.index.to_series().diff().dropna()
            median_gap = gaps.median()
            if median_gap.days <= 1:
                return "D"
            elif median_gap.days <= 8:
                return "W"
            elif median_gap.days <= 35:
                return "MS"
            else:
                return "QS"
        return freq
    except Exception:
        return "D"


def _remove_outliers_iqr(df: pd.DataFrame, col: str = "y", factor: float = 3.0) -> pd.DataFrame:
    """Cap extreme outliers using IQR so Prophet doesn't overfit to spikes."""
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - factor * IQR
    upper = Q3 + factor * IQR
    df = df.copy()
    df[col] = df[col].clip(lower=lower, upper=upper)
    return df


def _build_prophet_df(series: pd.Series) -> pd.DataFrame:
    """Convert a pd.Series with DatetimeIndex to Prophet's ds/y format."""
    df = pd.DataFrame({"ds": series.index, "y": series.values})
    df["ds"] = pd.to_datetime(df["ds"])
    df = df.dropna(subset=["y"])
    df = df.sort_values("ds").reset_index(drop=True)
    return df


def _get_seasonality_config(series: pd.Series) -> dict:
    """
    Decide which seasonality modes and components to enable
    based on data length and inferred frequency.
    """
    n = len(series)
    freq = _infer_frequency(series)
    freq_upper = freq.upper()

    config = {
        "yearly_seasonality": False,
        "weekly_seasonality": False,
        "daily_seasonality": False,
        "seasonality_mode": "additive",
    }

    # Yearly: need at least ~2 years of data
    if n >= 365 * 2 or (freq_upper.startswith("W") and n >= 104):
        config["yearly_seasonality"] = True

    # Weekly: only meaningful for daily data
    if "D" in freq_upper or "T" in freq_upper or "H" in freq_upper:
        if n >= 14:
            config["weekly_seasonality"] = True

    # Daily: only for sub-daily data
    if "H" in freq_upper or "T" in freq_upper:
        config["daily_seasonality"] = True

    # Multiplicative mode works better when the variance grows with the level
    if n >= 30:
        cv = series.std() / (series.mean() + 1e-8)
        if cv > 0.3:
            config["seasonality_mode"] = "multiplicative"

    return config


# ── Main function ─────────────────────────────────────────────────────────────

def _add_holidays_if_requested(model: "Prophet", country: str | None) -> bool:
    """
    Add country-specific holiday effects to a Prophet model, if requested.
    Must be called before model.fit(). Degrades gracefully (no holidays,
    forecast proceeds normally) if the country code isn't recognized by the
    underlying `holidays` package, rather than crashing the whole forecast.
    Returns True if holidays were actually added.
    """
    if not country:
        return False
    try:
        model.add_country_holidays(country_name=country)
        return True
    except Exception as e:
        logger.warning(f"Could not add holidays for country={country!r}: {e}")
        return False


def train_and_forecast(
    train_series: pd.Series,
    forecast_horizon: int,
    country: str | None = None,
) -> dict:
    """
    Fit a Prophet model on train_series and forecast `forecast_horizon` steps ahead.

    Args:
        train_series : pd.Series with a DatetimeIndex, already cleaned (no NaNs).
        forecast_horizon : int — number of future periods to forecast.
        country : Optional ISO country code (e.g. "US", "IN", "GB") to add
                   that country's public holidays as an extra Prophet
                   regressor. None (default) preserves prior behavior —
                   no holiday effects.

    Returns:
        {
            "model_name": "Prophet",
            "rmse": float,
            "forecast": list[float],
            "seasonality_mode": str,
            "components": list[str],
            "holidays_country": str | None,
        }
    """
    series = train_series.copy().astype(float)

    if len(series) < 10:
        raise ValueError(
            f"Prophet needs at least 10 observations, got {len(series)}."
        )

    # ── 1. Build Prophet dataframe ────────────────────────────────────────────
    full_df = _build_prophet_df(series)
    full_df = _remove_outliers_iqr(full_df, col="y", factor=3.0)

    # ── 2. Time Series Cross Validation ──────────────────────────────────────
    tscv = TimeSeriesSplit(n_splits=get_cv_splits(len(full_df)))
    cv_metrics = []
    train_times = []
    pred_times = []
    
    freq   = _infer_frequency(series)
    s_cfg  = _get_seasonality_config(series)

    for train_idx, test_idx in tscv.split(full_df):
        train_df = full_df.iloc[train_idx].copy()
        test_df  = full_df.iloc[test_idx].copy()
        
        model = Prophet(
            yearly_seasonality  = s_cfg["yearly_seasonality"],
            weekly_seasonality  = s_cfg["weekly_seasonality"],
            daily_seasonality   = s_cfg["daily_seasonality"],
            seasonality_mode    = s_cfg["seasonality_mode"],
            changepoint_prior_scale    = 0.1,
            seasonality_prior_scale    = 10.0,
            changepoint_range          = 0.9,
            interval_width             = 0.95,
            uncertainty_samples        = 0,
        )

        if "D" in freq.upper() and len(train_df) >= 60:
            model.add_seasonality(name="monthly", period=30.5, fourier_order=5)
        _add_holidays_if_requested(model, country)

        t0 = time.perf_counter()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(train_df)
        train_times.append(time.perf_counter() - t0)

        # ── Predict on test period ─────────────────────────────────────────────
        future_test = model.make_future_dataframe(periods=len(test_df), freq=freq, include_history=False)
        future_test["ds"] = test_df["ds"].values[: len(future_test)]

        t1 = time.perf_counter()
        forecast_test = model.predict(future_test)
        pred_times.append(time.perf_counter() - t1)

        y_pred = forecast_test["yhat"].values[: len(test_df)]
        y_true = test_df["y"].values[: len(y_pred)]

        series_min = float(series.min())
        floor = 0.0 if series_min >= 0 else None
        if floor is not None:
            y_pred = np.clip(y_pred, floor, None)

        metrics = calculate_metrics(y_true, y_pred)
        cv_metrics.append(metrics)

    avg_metrics = {k: sum(m[k] for m in cv_metrics)/len(cv_metrics) for k in cv_metrics[0].keys()}
    avg_train_time = sum(train_times)/len(train_times)
    avg_pred_time = sum(pred_times)/len(pred_times)

    # ── 6. Refit on FULL series for final forecast ────────────────────────────
    final_model = Prophet(
        yearly_seasonality       = s_cfg["yearly_seasonality"],
        weekly_seasonality       = s_cfg["weekly_seasonality"],
        daily_seasonality        = s_cfg["daily_seasonality"],
        seasonality_mode         = s_cfg["seasonality_mode"],
        changepoint_prior_scale  = 0.1,
        seasonality_prior_scale  = 10.0,
        changepoint_range        = 0.9,
        interval_width           = 0.95,
        uncertainty_samples      = 0,
    )

    if "D" in freq.upper() and len(full_df) >= 60:
        final_model.add_seasonality(name="monthly", period=30.5, fourier_order=5)
    holidays_added = _add_holidays_if_requested(final_model, country)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        final_model.fit(full_df)

    future_df       = final_model.make_future_dataframe(periods=forecast_horizon, freq=freq)
    forecast_df     = final_model.predict(future_df)
    future_forecast = forecast_df.tail(forecast_horizon)["yhat"].values

    if floor is not None:
        future_forecast = np.clip(future_forecast, floor, None)

    # ── 7. Detect which components were active ────────────────────────────────
    components = ["trend"]
    if s_cfg["weekly_seasonality"]:
        components.append("weekly")
    if s_cfg["yearly_seasonality"]:
        components.append("yearly")
    if "D" in freq.upper() and len(full_df) >= 60:
        components.append("monthly")
    if s_cfg["daily_seasonality"]:
        components.append("daily")
    if holidays_added:
        components.append("holidays")

    return {
        "model_name": "Prophet",
        **avg_metrics,
        "train_time_sec": round(avg_train_time, 4),
        "pred_time_sec": round(avg_pred_time, 4),
        "forecast": [round(float(v), 4) for v in future_forecast],
        "seasonality_mode": s_cfg["seasonality_mode"],
        "components": components,
        "holidays_country": country if holidays_added else None,
    }