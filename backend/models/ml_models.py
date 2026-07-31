"""ml_models.py — Moving Average, KNN, SVM forecasting models
All use the same lag/rolling feature engineering for fair comparison.
"""

import warnings
import numpy as np
import pandas as pd
from math import sqrt
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import LinearRegression
import time
from services.metrics import calculate_metrics, get_cv_splits

warnings.filterwarnings("ignore")

LAG_DAYS = [1, 2, 3, 7, 14, 21, 28, 30]


# ── Shared feature engineering ────────────────────────────────────────────────

def _create_features(series: pd.Series) -> pd.DataFrame:
    df = pd.DataFrame({"y": series.values}, index=series.index)

    for lag in [l for l in LAG_DAYS if l < len(series)]:
        df[f"lag_{lag}"] = df["y"].shift(lag)

    for w in [7, 14, 30]:
        if w < len(series):
            df[f"roll_mean_{w}"] = df["y"].shift(1).rolling(w, min_periods=2).mean()
            df[f"roll_std_{w}"]  = df["y"].shift(1).rolling(w, min_periods=2).std()

    for span in [7, 14]:
        if span < len(series):
            df[f"ewm_{span}"] = df["y"].shift(1).ewm(span=span, min_periods=2).mean()

    if isinstance(series.index, pd.DatetimeIndex):
        df["dayofweek"] = series.index.dayofweek
        df["month"]     = series.index.month
        df["quarter"]   = series.index.quarter
        df["sin_dow"]   = np.sin(2 * np.pi * df["dayofweek"] / 7)
        df["cos_dow"]   = np.cos(2 * np.pi * df["dayofweek"] / 7)
        df["sin_month"] = np.sin(2 * np.pi * df["month"] / 12)
        df["cos_month"] = np.cos(2 * np.pi * df["month"] / 12)

    df["trend"] = np.arange(len(df))
    return df


def _build_xy(df: pd.DataFrame):
    """
    Returns (X, y, t_idx, cols). t_idx is each surviving row's integer
    position in the original series (from the "trend" feature column) — used
    to reconstruct a detrended target back to a level.
    """
    df = df.dropna()
    cols = [c for c in df.columns if c != "y"]
    t_idx = df["trend"].values.astype(int)
    return df[cols].values, df["y"].values, t_idx, cols


# ── Trend extrapolation ───────────────────────────────────────────────────────
# KNN/SVR predictions are bounded by the y-values seen in training, so on
# trending data they lag behind instead of continuing the trend. Fitting a
# simple global linear trend (which *can* extrapolate correctly) and having
# the model predict only the residual sidesteps this. Not applied to Moving
# Average, which doesn't use lag-feature regression at all.

def _fit_trend(series: pd.Series) -> LinearRegression:
    t_idx = np.arange(len(series)).reshape(-1, 1)
    return LinearRegression().fit(t_idx, series.values)


def _detrend(series: pd.Series, trend_model: LinearRegression) -> pd.Series:
    t_idx = np.arange(len(series)).reshape(-1, 1)
    return pd.Series(series.values - trend_model.predict(t_idx), index=series.index)


def _recursive_forecast(model, scaler, history, horizon, feature_cols):
    preds  = []
    series = history.copy()
    step   = (history.index[-1] - history.index[-2]) if len(history) >= 2 and isinstance(history.index, pd.DatetimeIndex) else pd.Timedelta(days=1)
    last   = history.index[-1] if isinstance(history.index, pd.DatetimeIndex) else None
    floor  = 0.0 if float(history.min()) >= 0 else None

    for i in range(horizon):
        feat = _create_features(series)
        row  = feat[feature_cols].dropna()
        if len(row) == 0:
            preds.append(float(series.iloc[-1]))
        else:
            x    = scaler.transform(row.iloc[[-1]])
            yhat = float(model.predict(x)[0])
            if floor is not None:
                yhat = max(floor, yhat)
            preds.append(yhat)

        if last is not None:
            new_idx = pd.DatetimeIndex([last + step * (i + 1)])
        else:
            new_idx = pd.RangeIndex(len(series), len(series) + 1)

        series = pd.concat([series, pd.Series([preds[-1]], index=new_idx)])

    return preds


# ── Moving Average ────────────────────────────────────────────────────────────

def train_and_forecast_ma(series: pd.Series, forecast_horizon: int) -> dict:
    """
    Moving Average — tries windows 3,7,14,21,30 and picks best RMSE on test.
    Uses expanding-window MA for multi-step forecast.
    """
    series = series.dropna().astype(float)
    if len(series) < 5:
        raise ValueError(f"Moving Average needs ≥ 5 observations, got {len(series)}.")

    tscv = TimeSeriesSplit(n_splits=get_cv_splits(len(series)))
    cv_metrics = []
    train_times = []
    pred_times = []
    
    windows     = [w for w in [3, 7, 14, 21, 30] if w < len(series)]
    best_rmse   = np.inf
    best_window = windows[0]

    for w in windows:
        fold_rmses = []
        for train_idx, test_idx in tscv.split(series):
            tr, te = series.iloc[train_idx], series.iloc[test_idx]
            preds = []
            hist = list(tr)
            for _ in range(len(te)):
                preds.append(np.mean(hist[-w:]))
                hist.append(te.iloc[len(preds) - 1])
            fold_rmses.append(sqrt(mean_squared_error(te.values, preds)))
        avg_fold_rmse = np.mean(fold_rmses)
        if avg_fold_rmse < best_rmse:
            best_rmse = avg_fold_rmse
            best_window = w

    # For reporting metrics on best window
    for train_idx, test_idx in tscv.split(series):
        tr, te = series.iloc[train_idx], series.iloc[test_idx]
        preds = []
        hist = list(tr)
        t0 = time.perf_counter()
        # MA "trains" instantly
        train_times.append(time.perf_counter() - t0)
        
        t1 = time.perf_counter()
        for _ in range(len(te)):
            preds.append(np.mean(hist[-best_window:]))
            hist.append(te.iloc[len(preds) - 1])
        pred_times.append(time.perf_counter() - t1)
        
        metrics = calculate_metrics(te.values, preds)
        cv_metrics.append(metrics)
        
    avg_metrics = {k: sum(m[k] for m in cv_metrics)/len(cv_metrics) for k in cv_metrics[0].keys()}
    avg_train_time = sum(train_times)/len(train_times)
    avg_pred_time = sum(pred_times)/len(pred_times)

    # Forecast: use the best window recursively
    history = list(series)
    fc      = []
    floor   = 0.0 if series.min() >= 0 else None
    for _ in range(forecast_horizon):
        val = max(floor, np.mean(history[-best_window:])) if floor is not None else np.mean(history[-best_window:])
        fc.append(val)
        history.append(val)

    return {
        "model_name":  "MovingAverage",
        **avg_metrics,
        "train_time_sec": round(avg_train_time, 4),
        "pred_time_sec": round(avg_pred_time, 4),
        "forecast":    [round(float(v), 4) for v in fc],
        "best_window": best_window,
    }


# ── KNN ───────────────────────────────────────────────────────────────────────

def train_and_forecast_knn(series: pd.Series, forecast_horizon: int) -> dict:
    """KNN Regressor with lag features. Auto-tunes k via cross-val on test."""
    series = series.dropna().astype(float)
    min_req = max(LAG_DAYS) + 5
    if len(series) < min_req:
        raise ValueError(f"KNN needs ≥ {min_req} observations, got {len(series)}.")

    trend_model = _fit_trend(series)
    residual = _detrend(series, trend_model)

    feat_df = _create_features(residual)
    X, y, t_idx, cols = _build_xy(feat_df)

    tscv = TimeSeriesSplit(n_splits=get_cv_splits(len(X)))

    cv_metrics = []
    train_times = []
    pred_times = []

    # Tune k (scored on the reconstructed level, not the raw residual)
    best_rmse, best_k = np.inf, 5
    for k in [3, 5, 7, 10, 15]:
        if k >= len(X) * 0.5:
            continue
        fold_rmses = []
        for train_idx, test_idx in tscv.split(X):
            X_tr, X_te = X[train_idx], X[test_idx]
            y_tr = y[train_idx]
            t_te = t_idx[test_idx]
            scaler = RobustScaler()
            X_tr_s = scaler.fit_transform(X_tr)
            X_te_s = scaler.transform(X_te)
            try:
                m = KNeighborsRegressor(n_neighbors=k, weights="distance", metric="euclidean")
                m.fit(X_tr_s, y_tr)
                pred = m.predict(X_te_s) + trend_model.predict(t_te.reshape(-1, 1))
                level_te = series.values[t_te]
                if series.min() >= 0:
                    pred = np.clip(pred, 0, None)
                fold_rmses.append(sqrt(mean_squared_error(level_te, pred)))
            except Exception:
                continue
        if fold_rmses:
            avg_rmse = np.mean(fold_rmses)
            if avg_rmse < best_rmse:
                best_rmse, best_k = avg_rmse, k

    # Get metrics on best k
    for train_idx, test_idx in tscv.split(X):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr = y[train_idx]
        t_te = t_idx[test_idx]
        scaler = RobustScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_te_s = scaler.transform(X_te)

        m = KNeighborsRegressor(n_neighbors=best_k, weights="distance", metric="euclidean")
        t0 = time.perf_counter()
        m.fit(X_tr_s, y_tr)
        train_times.append(time.perf_counter() - t0)

        t1 = time.perf_counter()
        pred = m.predict(X_te_s) + trend_model.predict(t_te.reshape(-1, 1))
        pred_times.append(time.perf_counter() - t1)

        level_te = series.values[t_te]
        if series.min() >= 0:
            pred = np.clip(pred, 0, None)

        metrics = calculate_metrics(level_te, pred)
        cv_metrics.append(metrics)

    avg_metrics = {k: sum(m[k] for m in cv_metrics)/len(cv_metrics) for k in cv_metrics[0].keys()}
    avg_train_time = sum(train_times)/len(train_times)
    avg_pred_time = sum(pred_times)/len(pred_times)

    # Refit on full data
    feat_full = _create_features(residual)
    X_f, y_f, t_idx_f, cols_f = _build_xy(feat_full)
    sc_f = RobustScaler()
    X_fs = sc_f.fit_transform(X_f)
    final = KNeighborsRegressor(n_neighbors=best_k, weights="distance", metric="euclidean")
    final.fit(X_fs, y_f)

    fc_resid = _recursive_forecast(final, sc_f, residual, forecast_horizon, cols_f)
    future_t = np.arange(len(series), len(series) + forecast_horizon).reshape(-1, 1)
    fc = np.array(fc_resid) + trend_model.predict(future_t)
    if series.min() >= 0:
        fc = np.clip(fc, 0, None)

    return {
        "model_name": "KNN",
        **avg_metrics,
        "train_time_sec": round(avg_train_time, 4),
        "pred_time_sec": round(avg_pred_time, 4),
        "forecast":   [round(float(v), 4) for v in fc],
        "best_k":     best_k,
    }


# ── SVM ───────────────────────────────────────────────────────────────────────

def train_and_forecast_svm(series: pd.Series, forecast_horizon: int) -> dict:
    """SVR with RBF kernel. Auto-tunes C and epsilon on test RMSE."""
    series = series.dropna().astype(float)
    min_req = max(LAG_DAYS) + 5
    if len(series) < min_req:
        raise ValueError(f"SVM needs ≥ {min_req} observations, got {len(series)}.")

    trend_model = _fit_trend(series)
    residual = _detrend(series, trend_model)

    feat_df = _create_features(residual)
    X, y, t_idx, cols = _build_xy(feat_df)

    tscv = TimeSeriesSplit(n_splits=get_cv_splits(len(X)))
    cv_metrics = []
    train_times = []
    pred_times = []

    best_rmse, best_params = np.inf, {"C": 1.0, "epsilon": 0.1, "gamma": "scale"}
    param_grid = [
        {"C": c, "epsilon": eps, "gamma": g}
        for c   in [1.0, 10.0]
        for eps in [0.1, 0.5]
        for g   in ["scale"]
    ]

    for params in param_grid:
        fold_rmses = []
        for train_idx, test_idx in tscv.split(X):
            X_tr, X_te = X[train_idx], X[test_idx]
            y_tr = y[train_idx]
            t_te = t_idx[test_idx]

            scaler = RobustScaler()
            X_tr_s = scaler.fit_transform(X_tr)
            X_te_s = scaler.transform(X_te)

            y_scaler = RobustScaler()
            y_tr_s = y_scaler.fit_transform(y_tr.reshape(-1, 1)).ravel()

            try:
                m = SVR(kernel="rbf", **params)
                m.fit(X_tr_s, y_tr_s)
                pred_resid = y_scaler.inverse_transform(m.predict(X_te_s).reshape(-1, 1)).ravel()
                pred = pred_resid + trend_model.predict(t_te.reshape(-1, 1))
                level_te = series.values[t_te]
                if series.min() >= 0:
                    pred = np.clip(pred, 0, None)
                fold_rmses.append(sqrt(mean_squared_error(level_te, pred)))
            except Exception:
                continue
        if fold_rmses:
            avg_rmse = np.mean(fold_rmses)
            if avg_rmse < best_rmse:
                best_rmse, best_params = avg_rmse, params

    # Get metrics on best params
    for train_idx, test_idx in tscv.split(X):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr = y[train_idx]
        t_te = t_idx[test_idx]

        scaler = RobustScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_te_s = scaler.transform(X_te)

        y_scaler = RobustScaler()
        y_tr_s = y_scaler.fit_transform(y_tr.reshape(-1, 1)).ravel()

        m = SVR(kernel="rbf", **best_params)
        t0 = time.perf_counter()
        m.fit(X_tr_s, y_tr_s)
        train_times.append(time.perf_counter() - t0)

        t1 = time.perf_counter()
        pred_resid = y_scaler.inverse_transform(m.predict(X_te_s).reshape(-1, 1)).ravel()
        pred_times.append(time.perf_counter() - t1)

        pred = pred_resid + trend_model.predict(t_te.reshape(-1, 1))
        level_te = series.values[t_te]
        if series.min() >= 0:
            pred = np.clip(pred, 0, None)

        metrics = calculate_metrics(level_te, pred)
        cv_metrics.append(metrics)

    avg_metrics = {k: sum(m[k] for m in cv_metrics)/len(cv_metrics) for k in cv_metrics[0].keys()}
    avg_train_time = sum(train_times)/len(train_times)
    avg_pred_time = sum(pred_times)/len(pred_times)

    # Refit on full data
    feat_full = _create_features(residual)
    X_f, y_f, t_idx_f, cols_f = _build_xy(feat_full)
    sc_f  = RobustScaler()
    X_fs  = sc_f.fit_transform(X_f)
    ysc_f = RobustScaler()
    y_fs  = ysc_f.fit_transform(y_f.reshape(-1, 1)).ravel()

    final = SVR(kernel="rbf", **best_params)
    final.fit(X_fs, y_fs)

    # Custom recursive forecast in residual space (need to invert y-scaling),
    # then add the extrapolated trend back on for the final level forecast.
    preds_resid = []
    curr   = residual.copy()
    step   = (residual.index[-1] - residual.index[-2]) if len(residual) >= 2 and isinstance(residual.index, pd.DatetimeIndex) else pd.Timedelta(days=1)
    last   = residual.index[-1] if isinstance(residual.index, pd.DatetimeIndex) else None

    for i in range(forecast_horizon):
        feat = _create_features(curr)
        row  = feat[cols_f].dropna()
        if len(row) == 0:
            preds_resid.append(float(curr.iloc[-1]))
        else:
            x    = sc_f.transform(row.iloc[[-1]])
            ys   = final.predict(x)
            rhat = float(ysc_f.inverse_transform(ys.reshape(-1, 1))[0, 0])
            preds_resid.append(rhat)

        if last is not None:
            new_idx = pd.DatetimeIndex([last + step * (i + 1)])
        else:
            new_idx = pd.RangeIndex(len(curr), len(curr) + 1)
        curr = pd.concat([curr, pd.Series([preds_resid[-1]], index=new_idx)])

    future_t = np.arange(len(series), len(series) + forecast_horizon).reshape(-1, 1)
    preds = np.array(preds_resid) + trend_model.predict(future_t)
    if series.min() >= 0:
        preds = np.clip(preds, 0, None)

    return {
        "model_name":  "SVM",
        **avg_metrics,
        "train_time_sec": round(avg_train_time, 4),
        "pred_time_sec": round(avg_pred_time, 4),
        "forecast":    [round(float(v), 4) for v in preds],
        "best_C":      best_params["C"],
        "best_epsilon": best_params["epsilon"],
    }
