"""
xgboost_model.py — XGBoost time series forecasting model
Rich feature engineering: lags, rolling stats, EWM, calendar features.
Recursive multi-step forecast for any horizon.
"""

import warnings
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
import time

from services.metrics import calculate_metrics, get_cv_splits
from services.preprocessor import build_future_exog

warnings.filterwarnings("ignore")


# ── Feature engineering ───────────────────────────────────────────────────────

LAG_DAYS = [1, 2, 3, 7, 14, 21, 28, 30]

def _create_features(
    series: pd.Series,
    lag_days: list[int] = LAG_DAYS,
    exog_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Build a feature matrix from a time series.
    Features:
      - Lag values (t-1, t-2, t-3, t-7, t-14, t-21, t-28, t-30)
      - Rolling mean / std (7, 14, 30 day windows)
      - Exponentially weighted mean (span 7 and 14)
      - Calendar: dayofweek, month, quarter, dayofyear, is_month_start/end
      - Trend: integer time index
      - Real exogenous features (price/promo), if exog_df is given — joined
        by date, so exog_df must cover every date in `series` (including
        future dates during recursive forecasting — see build_future_exog).
    """
    df = pd.DataFrame({"y": series.values}, index=series.index)

    # ── Lag features ─────────────────────────────────────────────────────────
    available_lags = [l for l in lag_days if l < len(series)]
    for lag in available_lags:
        df[f"lag_{lag}"] = df["y"].shift(lag)

    # ── Rolling statistics ────────────────────────────────────────────────────
    for window in [7, 14, 30]:
        if window < len(series):
            df[f"roll_mean_{window}"] = df["y"].shift(1).rolling(window, min_periods=2).mean()
            df[f"roll_std_{window}"]  = df["y"].shift(1).rolling(window, min_periods=2).std()
            df[f"roll_min_{window}"]  = df["y"].shift(1).rolling(window, min_periods=2).min()
            df[f"roll_max_{window}"]  = df["y"].shift(1).rolling(window, min_periods=2).max()

    # ── Exponentially weighted mean ───────────────────────────────────────────
    for span in [7, 14]:
        if span < len(series):
            df[f"ewm_{span}"] = df["y"].shift(1).ewm(span=span, min_periods=2).mean()

    # ── Calendar features (only if DatetimeIndex) ────────────────────────────
    if isinstance(series.index, pd.DatetimeIndex):
        df["dayofweek"]      = series.index.dayofweek
        df["month"]          = series.index.month
        df["quarter"]        = series.index.quarter
        df["dayofyear"]      = series.index.dayofyear
        df["weekofyear"]     = series.index.isocalendar().week.astype(int)
        df["is_month_start"] = series.index.is_month_start.astype(int)
        df["is_month_end"]   = series.index.is_month_end.astype(int)
        df["is_quarter_end"] = series.index.is_quarter_end.astype(int)
        # Cyclical encoding for periodicity
        df["sin_dow"]  = np.sin(2 * np.pi * df["dayofweek"] / 7)
        df["cos_dow"]  = np.cos(2 * np.pi * df["dayofweek"] / 7)
        df["sin_month"] = np.sin(2 * np.pi * df["month"] / 12)
        df["cos_month"] = np.cos(2 * np.pi * df["month"] / 12)

    # ── Trend ─────────────────────────────────────────────────────────────────
    df["trend"] = np.arange(len(df))

    # ── Real exogenous features (price/promo), if provided ───────────────────
    if exog_df is not None and not exog_df.empty:
        df = df.join(exog_df, how="left")

    return df


def _build_xy(feature_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """
    Drop rows with NaNs introduced by lags/rolling, split X and y.

    Also returns t_idx — each surviving row's integer position in the
    original series (taken from the "trend" feature column, which is exactly
    that) — used by callers to reconstruct a detrended target back to a level.
    """
    df = feature_df.dropna()
    feature_cols = [c for c in df.columns if c != "y"]
    X = df[feature_cols].values
    y = df["y"].values
    t_idx = df["trend"].values.astype(int)
    return X, y, t_idx, feature_cols


# ── Trend extrapolation ───────────────────────────────────────────────────────
# Trees can't predict values outside the y-range seen in training, so on
# trending data they plateau instead of continuing the trend. Fitting a
# simple global linear trend (which *can* extrapolate correctly) and having
# the tree model predict only the residual sidesteps this: the residual stays
# roughly stationary regardless of how far the level has trended.

def _fit_trend(series: pd.Series) -> LinearRegression:
    t_idx = np.arange(len(series)).reshape(-1, 1)
    return LinearRegression().fit(t_idx, series.values)


def _detrend(series: pd.Series, trend_model: LinearRegression) -> pd.Series:
    t_idx = np.arange(len(series)).reshape(-1, 1)
    return pd.Series(series.values - trend_model.predict(t_idx), index=series.index)


# ── Recursive multi-step forecasting ─────────────────────────────────────────

def _recursive_forecast(
    model: XGBRegressor,
    scaler_X: RobustScaler,
    history: pd.Series,
    forecast_horizon: int,
    feature_cols: list[str],
    exog_df: pd.DataFrame | None = None,
) -> list[float]:
    """
    Forecast step-by-step: feed each prediction back as a lag for the next step.

    `history` is expected to already be detrended (residual) — the caller adds
    the extrapolated trend back on afterward. Working in residual space here
    means each step's lag/rolling features stay consistent with what the
    model was trained on.

    `exog_df`, if given, must already cover BOTH the historical dates in
    `history` AND every future date this call will step through (see
    build_future_exog) — each step just joins against whatever rows exist.
    """
    predictions: list[float] = []
    current_series = history.copy()

    # Infer step size from index
    if isinstance(history.index, pd.DatetimeIndex) and len(history) >= 2:
        step = history.index[-1] - history.index[-2]
    else:
        step = pd.Timedelta(days=1)

    last_date = history.index[-1] if isinstance(history.index, pd.DatetimeIndex) else None

    for i in range(forecast_horizon):
        # Build features for the next step
        feature_df  = _create_features(current_series, exog_df=exog_df)
        feature_df  = feature_df[feature_cols].dropna()

        if len(feature_df) == 0:
            # Fall back to last known value if features can't be built
            predictions.append(float(current_series.iloc[-1]))
        else:
            row = feature_df.iloc[[-1]]
            row_scaled = scaler_X.transform(row)
            yhat = float(model.predict(row_scaled)[0])
            predictions.append(yhat)

        # Append prediction to history for next iteration
        if last_date is not None:
            next_date = last_date + step * (i + 1)
            new_row   = pd.Series([predictions[-1]], index=[next_date])
        else:
            new_row = pd.Series([predictions[-1]])

        current_series = pd.concat([current_series, new_row])

    return predictions


# ── Main function ─────────────────────────────────────────────────────────────

def train_and_forecast(
    train_series: pd.Series,
    forecast_horizon: int,
    exog_df: pd.DataFrame | None = None,
    exog_meta: dict | None = None,
) -> dict:
    """
    Fit an XGBoost model on train_series using lag/rolling/calendar features
    and forecast `forecast_horizon` steps ahead recursively.

    Args:
        train_series : pd.Series with a DatetimeIndex, already cleaned (no NaNs).
        forecast_horizon : int — number of future periods to forecast.
        exog_df       : Optional real exogenous features (price/promo), aligned
                        to train_series.index (see services/preprocessor.py).
        exog_meta     : {"price_cols": [...], "promo_cols": [...]} — required
                        alongside exog_df, to know how to fill in *future*
                        values for the forecast horizon (see build_future_exog).

    Returns:
        {
            "model_name": "XGBoost",
            "rmse": float,
            "forecast": list[float],
            "feature_importance": dict,
            "n_features": int,
            "exog_features_used": list[str],
        }
    """
    series = train_series.copy().astype(float)
    min_required = max(LAG_DAYS) + 5
    if len(series) < min_required:
        raise ValueError(
            f"XGBoost needs at least {min_required} observations, got {len(series)}."
        )

    has_exog = exog_df is not None and not exog_df.empty
    exog_features_used = list(exog_df.columns) if has_exog else []

    # ── 1. Detrend, then build feature matrix from the residual ──────────────
    trend_model = _fit_trend(series)
    residual = _detrend(series, trend_model)

    feature_df = _create_features(residual, exog_df=exog_df)
    X, y, t_idx, feature_cols = _build_xy(feature_df)

    # ── 2. Time Series Cross Validation (adaptive fold count) ────────────────
    n_splits = get_cv_splits(len(X))
    tscv = TimeSeriesSplit(n_splits=n_splits)

    cv_metrics = []
    train_times = []
    pred_times = []
    best_iteration = 500

    for train_idx, test_idx in tscv.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        t_test = t_idx[test_idx]

        # ── 3. Scale features ───────────────────────────────────────────────────
        scaler_X = RobustScaler()
        X_train_s = scaler_X.fit_transform(X_train)
        X_test_s  = scaler_X.transform(X_test)

        # ── 4. Train XGBoost ────────────────────────────────────────────────────
        model = XGBRegressor(
            n_estimators          = 500,
            learning_rate         = 0.05,
            max_depth             = 5,
            min_child_weight      = 3,
            subsample             = 0.8,
            colsample_bytree      = 0.8,
            reg_alpha             = 0.1,
            reg_lambda            = 1.0,
            gamma                 = 0.05,
            early_stopping_rounds = 30,
            eval_metric           = "rmse",
            random_state          = 42,
            n_jobs                = -1,
            verbosity             = 0,
        )

        t0 = time.perf_counter()
        model.fit(
            X_train_s, y_train,
            eval_set=[(X_test_s, y_test)],
            verbose=False,
        )
        train_times.append(time.perf_counter() - t0)

        best_iteration = model.best_iteration + 1 if hasattr(model, "best_iteration") else 500

        # ── 5. Evaluate on test fold (add trend back to compare in level units) ─
        t1 = time.perf_counter()
        y_pred_resid = model.predict(X_test_s)
        pred_times.append(time.perf_counter() - t1)

        y_pred = y_pred_resid + trend_model.predict(t_test.reshape(-1, 1))
        y_level_test = series.values[t_test]
        if float(series.min()) >= 0:
            y_pred = np.clip(y_pred, 0, None)

        metrics = calculate_metrics(y_level_test, y_pred)
        cv_metrics.append(metrics)

    # Average metrics across folds
    avg_metrics = {
        k: sum(m[k] for m in cv_metrics) / len(cv_metrics)
        for k in cv_metrics[0].keys()
    }
    avg_train_time = sum(train_times) / len(train_times)
    avg_pred_time = sum(pred_times) / len(pred_times)

    # ── 6. Refit on full series for final forecast ────────────────────────────
    full_feature_df = _create_features(residual, exog_df=exog_df)
    X_full, y_full, t_idx_full, feature_cols_full = _build_xy(full_feature_df)

    scaler_full = RobustScaler()
    X_full_s    = scaler_full.fit_transform(X_full)

    final_model = XGBRegressor(
        n_estimators     = best_iteration,
        learning_rate    = 0.05,
        max_depth        = 5,
        min_child_weight = 3,
        subsample        = 0.8,
        colsample_bytree = 0.8,
        reg_alpha        = 0.1,
        reg_lambda       = 1.0,
        gamma            = 0.05,
        random_state     = 42,
        n_jobs           = -1,
        verbosity        = 0,
    )
    final_model.fit(X_full_s, y_full, verbose=False)

    # ── 7. Recursive multi-step forecast (in residual space), then add the
    #        extrapolated trend back on to get the final level forecast ──────
    # If real exog features were used in training, the recursive loop needs
    # them for the future dates too — extend historical exog into the horizon
    # (see build_future_exog: price carries forward, promo assumes none).
    combined_exog = None
    if has_exog and isinstance(residual.index, pd.DatetimeIndex) and len(residual) >= 2:
        step = residual.index[-1] - residual.index[-2]
        future_index = pd.DatetimeIndex(
            [residual.index[-1] + step * (i + 1) for i in range(forecast_horizon)]
        )
        future_exog = build_future_exog(exog_df, exog_meta or {}, future_index)
        combined_exog = pd.concat([exog_df, future_exog])

    forecast_resid = _recursive_forecast(
        final_model, scaler_full, residual, forecast_horizon, feature_cols_full,
        exog_df=combined_exog,
    )
    future_t = np.arange(len(series), len(series) + forecast_horizon).reshape(-1, 1)
    forecast_values = np.array(forecast_resid) + trend_model.predict(future_t)
    if float(series.min()) >= 0:
        forecast_values = np.clip(forecast_values, 0, None)

    # ── 8. Feature importance (top 10) ────────────────────────────────────────
    importance_raw  = final_model.feature_importances_
    importance_dict = dict(zip(feature_cols_full, importance_raw.tolist()))
    top_features    = dict(
        sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)[:10]
    )

    return {
        "model_name": "XGBoost",
        **avg_metrics,
        "train_time_sec": round(avg_train_time, 4),
        "pred_time_sec": round(avg_pred_time, 4),
        "forecast": [round(float(v), 4) for v in forecast_values],
        "feature_importance": {k: round(v, 4) for k, v in top_features.items()},
        "n_features": len(feature_cols_full),
        "exog_features_used": exog_features_used,
    }