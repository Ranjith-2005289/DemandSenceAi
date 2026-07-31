"""
rf_model.py — Random Forest time series forecasting model
Same feature engineering as XGBoost (lags, rolling, calendar).
Uses OOB scoring + recursive multi-step forecast.
"""

import warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import LinearRegression
import time

from services.metrics import calculate_metrics, get_cv_splits
from services.preprocessor import build_future_exog

warnings.filterwarnings("ignore")


# ── Feature engineering (mirrors xgboost_model.py) ───────────────────────────

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
      - Rolling mean / std / min / max (7, 14, 30 day windows)
      - Exponentially weighted mean (span 7 and 14)
      - Calendar: dayofweek, month, quarter, dayofyear, is_month_start/end
      - Trend: integer time index
      - Real exogenous features (price/promo), if exog_df is given — see
        xgboost_model.py's _create_features for the same pattern.
    Identical feature set to XGBoost for fair RMSE comparison.
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
        # Cyclical encoding
        df["sin_dow"]   = np.sin(2 * np.pi * df["dayofweek"] / 7)
        df["cos_dow"]   = np.cos(2 * np.pi * df["dayofweek"] / 7)
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
    Drop NaN rows from lags/rolling, split X and y.

    Also returns t_idx — each surviving row's integer position in the
    original series (from the "trend" feature column) — used to reconstruct
    a detrended target back to a level.
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
# the forest predict only the residual sidesteps this.

def _fit_trend(series: pd.Series) -> LinearRegression:
    t_idx = np.arange(len(series)).reshape(-1, 1)
    return LinearRegression().fit(t_idx, series.values)


def _detrend(series: pd.Series, trend_model: LinearRegression) -> pd.Series:
    t_idx = np.arange(len(series)).reshape(-1, 1)
    return pd.Series(series.values - trend_model.predict(t_idx), index=series.index)


# ── Recursive multi-step forecasting ─────────────────────────────────────────

def _recursive_forecast(
    model: RandomForestRegressor,
    scaler_X: RobustScaler,
    history: pd.Series,
    forecast_horizon: int,
    feature_cols: list[str],
    exog_df: pd.DataFrame | None = None,
) -> list[float]:
    """
    Forecast step-by-step: append each prediction to the history series
    so lags stay up to date for the next step.

    `exog_df`, if given, must already cover both the historical dates in
    `history` and every future date this call steps through (see
    build_future_exog in services/preprocessor.py).
    """
    predictions: list[float] = []
    current_series = history.copy()

    if isinstance(history.index, pd.DatetimeIndex) and len(history) >= 2:
        step = history.index[-1] - history.index[-2]
    else:
        step = pd.Timedelta(days=1)

    last_date = history.index[-1] if isinstance(history.index, pd.DatetimeIndex) else None
    floor = 0.0 if float(history.min()) >= 0 else None

    for i in range(forecast_horizon):
        feature_df = _create_features(current_series, exog_df=exog_df)

        if feature_cols[0] not in feature_df.columns:
            predictions.append(float(current_series.iloc[-1]))
        else:
            row = feature_df[feature_cols].dropna()
            if len(row) == 0:
                predictions.append(float(current_series.iloc[-1]))
            else:
                row_scaled = scaler_X.transform(row.iloc[[-1]])
                yhat = float(model.predict(row_scaled)[0])
                if floor is not None:
                    yhat = max(floor, yhat)
                predictions.append(yhat)

        if last_date is not None:
            next_date = last_date + step * (i + 1)
            new_row   = pd.Series([predictions[-1]], index=[next_date])
        else:
            new_row = pd.Series([predictions[-1]])

        current_series = pd.concat([current_series, new_row])

    return predictions


# ── Quantile prediction (uncertainty bands) ──────────────────────────────────

def _tree_predictions(model: RandomForestRegressor, X_row: np.ndarray) -> np.ndarray:
    """Get per-tree predictions for uncertainty estimation."""
    return np.array([tree.predict(X_row)[0] for tree in model.estimators_])


# ── Main function ─────────────────────────────────────────────────────────────

def train_and_forecast(
    train_series: pd.Series,
    forecast_horizon: int,
    exog_df: pd.DataFrame | None = None,
    exog_meta: dict | None = None,
) -> dict:
    """
    Fit a Random Forest model on train_series using lag/rolling/calendar features
    and forecast `forecast_horizon` steps ahead recursively.

    Args:
        train_series : pd.Series with a DatetimeIndex, already cleaned (no NaNs).
        forecast_horizon : int — number of future periods to forecast.
        exog_df       : Optional real exogenous features (price/promo), aligned
                        to train_series.index (see services/preprocessor.py).
        exog_meta     : {"price_cols": [...], "promo_cols": [...]} — required
                        alongside exog_df to fill in future horizon values.

    Returns:
        {
            "model_name": "RandomForest",
            "rmse": float,
            "forecast": list[float],
            "forecast_lower": list[float],   95% lower bound (tree std)
            "forecast_upper": list[float],   95% upper bound (tree std)
            "feature_importance": dict,
            "oob_score": float | None,
            "exog_features_used": list[str],
        }
    """
    series = train_series.copy().astype(float)
    min_required = max(LAG_DAYS) + 5
    if len(series) < min_required:
        raise ValueError(
            f"RandomForest needs at least {min_required} observations, got {len(series)}."
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

    for train_idx, test_idx in tscv.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train = y[train_idx]
        t_test = t_idx[test_idx]

        # ── 3. Scale features ─────────────────────────────────────────────────────
        scaler_X  = RobustScaler()
        X_train_s = scaler_X.fit_transform(X_train)
        X_test_s  = scaler_X.transform(X_test)

        # ── 4. Train Random Forest ────────────────────────────────────────────────
        model = RandomForestRegressor(
            n_estimators      = 100,
            max_depth         = None,
            min_samples_split = 5,
            min_samples_leaf  = 2,
            max_features      = "sqrt",
            bootstrap         = True,
            oob_score         = True,
            n_jobs            = -1,
            random_state      = 42,
        )

        t0 = time.perf_counter()
        model.fit(X_train_s, y_train)
        train_times.append(time.perf_counter() - t0)

        # ── 5. Evaluate on test fold (add trend back to compare in level units) ─
        t1 = time.perf_counter()
        y_pred_resid = model.predict(X_test_s)
        pred_times.append(time.perf_counter() - t1)

        y_pred = y_pred_resid + trend_model.predict(t_test.reshape(-1, 1))
        y_level_test = series.values[t_test]
        floor  = 0.0 if float(series.min()) >= 0 else None
        if floor is not None:
            y_pred = np.clip(y_pred, floor, None)

        metrics = calculate_metrics(y_level_test, y_pred)
        cv_metrics.append(metrics)
        
    avg_metrics = {
        k: sum(m[k] for m in cv_metrics) / len(cv_metrics)
        for k in cv_metrics[0].keys()
    }
    avg_train_time = sum(train_times) / len(train_times)
    avg_pred_time = sum(pred_times) / len(pred_times)

    # OOB R² (free internal validation from Random Forest bootstrap)
    oob_score = round(float(model.oob_score_), 4) if hasattr(model, "oob_score_") else None

    # ── 6. Refit on full series ───────────────────────────────────────────────
    full_feature_df = _create_features(residual, exog_df=exog_df)
    X_full, y_full, t_idx_full, feature_cols_full = _build_xy(full_feature_df)

    scaler_full = RobustScaler()
    X_full_s    = scaler_full.fit_transform(X_full)

    final_model = RandomForestRegressor(
        n_estimators      = 100,
        min_samples_split = 5,
        min_samples_leaf  = 2,
        max_features      = "sqrt",
        bootstrap         = True,
        oob_score         = True,
        n_jobs            = -1,
        random_state      = 42,
    )
    final_model.fit(X_full_s, y_full)

    # ── 7. Recursive multi-step forecast (residual space), then add the
    #        extrapolated trend back on to get the final level forecast ──────
    # If real exog features were used in training, extend them into the
    # horizon for the recursive loop (see build_future_exog).
    combined_exog = None
    if has_exog and isinstance(residual.index, pd.DatetimeIndex) and len(residual) >= 2:
        step_size = residual.index[-1] - residual.index[-2]
        future_index = pd.DatetimeIndex(
            [residual.index[-1] + step_size * (i + 1) for i in range(forecast_horizon)]
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

    # ── 8. Uncertainty bands via per-tree std deviation ───────────────────────
    # Rebuild feature row for the first forecast step to estimate spread
    try:
        feat_df_last = _create_features(residual, exog_df=exog_df)
        last_row = feat_df_last[feature_cols_full].dropna().iloc[[-1]]
        last_row_s  = scaler_full.transform(last_row)
        tree_preds  = _tree_predictions(final_model, last_row_s)
        std_est     = float(np.std(tree_preds))

        floor = 0.0 if float(series.min()) >= 0 else None
        z = 1.96  # 95% interval
        forecast_lower = [max(0.0 if floor == 0.0 else -np.inf, v - z * std_est) for v in forecast_values]
        forecast_upper = [v + z * std_est for v in forecast_values]
    except Exception:
        forecast_lower = forecast_values
        forecast_upper = forecast_values

    # ── 9. Feature importance (top 10) ────────────────────────────────────────
    importance_raw  = final_model.feature_importances_
    importance_dict = dict(zip(feature_cols_full, importance_raw.tolist()))
    top_features    = dict(
        sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)[:10]
    )

    return {
        "model_name": "RandomForest",
        **avg_metrics,
        "train_time_sec": round(avg_train_time, 4),
        "pred_time_sec": round(avg_pred_time, 4),
        "forecast": [round(float(v), 4) for v in forecast_values],
        "forecast_lower": [round(float(v), 4) for v in forecast_lower],
        "forecast_upper": [round(float(v), 4) for v in forecast_upper],
        "feature_importance": {k: round(v, 4) for k, v in top_features.items()},
        "oob_score": oob_score,
        "exog_features_used": exog_features_used,
    }
