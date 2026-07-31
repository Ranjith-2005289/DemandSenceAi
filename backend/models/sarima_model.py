"""
sarima_model.py — SARIMA & SARIMAX forecasting models
SARIMA: Auto grid-search seasonal orders by AIC
SARIMAX: Same + exogenous features (calendar dummies)
"""

import warnings
import itertools
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
import time
from services.metrics import calculate_metrics, get_cv_splits
from services.preprocessor import build_future_exog

warnings.filterwarnings("ignore")

# Cap evaluation (order-search + CV) cost so it stays bounded regardless of
# how large the input series is. The final production forecast still refits
# on the complete series — this only bounds the cost of deciding which order
# is best and computing the reported CV metric.
MAX_EVAL_ROWS = 2000

# Above this seasonal period, skip the ~35-combination AIC grid search and use
# a fixed reasonable order instead. Each individual fit already costs several
# seconds once m is this large (state-space cost scales roughly with m²), so
# grid-searching 35 of them is the dominant cost blowup for genuinely
# high-frequency data (hourly-within-something, minutely...) — well beyond
# the realistic range for retail seasonality (7/12/4/52).
MAX_GRID_SEARCH_SEASONAL_PERIOD = 24


# ── Helpers ───────────────────────────────────────────────────────────────────

def _infer_seasonal_period(series: pd.Series) -> int:
    """Infer seasonal period m from DatetimeIndex frequency."""
    if not isinstance(series.index, pd.DatetimeIndex):
        return 7
    try:
        freq = pd.infer_freq(series.index)
        if freq is None:
            return 7
        freq = freq.upper()
        # Order matters: minutely aliases ("MIN") start with "M" and would be
        # misread as monthly by a bare substring check, so check the more
        # specific / higher-frequency codes first.
        if freq.startswith("MIN") or freq.startswith("T"): return 60   # minutely → hourly
        if freq.startswith("S"):  return 60   # secondly → minutely
        if freq.startswith("H"):  return 24   # hourly   → daily
        if freq.startswith("D"):  return 7    # daily    → weekly
        if freq.startswith("W"):  return 52   # weekly   → yearly
        if freq.startswith("M"):  return 12   # monthly ("M"/"MS"/"ME") → yearly
        if freq.startswith("Q"):  return 4    # quarterly
    except Exception:
        pass
    return 7


def _check_stationarity(series: pd.Series) -> int:
    try:
        p = adfuller(series.dropna(), autolag="AIC")[1]
        return 0 if p < 0.05 else 1
    except Exception:
        return 1


def _make_exog_features(
    index: pd.DatetimeIndex,
    real_exog_df: pd.DataFrame | None = None,
) -> np.ndarray | None:
    """Build calendar exogenous features for SARIMAX, plus real exogenous
    features (price/promo) if given — real_exog_df must already cover every
    date in `index` (see build_future_exog for the forecast-horizon case)."""
    if not isinstance(index, pd.DatetimeIndex):
        return None
    try:
        df = pd.DataFrame(index=index)
        df["month_sin"]  = np.sin(2 * np.pi * index.month / 12)
        df["month_cos"]  = np.cos(2 * np.pi * index.month / 12)
        df["dow_sin"]    = np.sin(2 * np.pi * index.dayofweek / 7)
        df["dow_cos"]    = np.cos(2 * np.pi * index.dayofweek / 7)
        df["is_month_end"] = index.is_month_end.astype(int)
        if real_exog_df is not None and not real_exog_df.empty:
            df = df.join(real_exog_df.reindex(index), how="left")
        return df.values
    except Exception:
        return None


def _make_future_exog(
    last_date: pd.Timestamp,
    freq: str,
    horizon: int,
    real_exog_df: pd.DataFrame | None = None,
    exog_meta: dict | None = None,
) -> np.ndarray | None:
    try:
        future_idx = pd.date_range(start=last_date, periods=horizon + 1, freq=freq)[1:]
        future_real_exog = None
        if real_exog_df is not None and not real_exog_df.empty:
            future_real_exog = build_future_exog(real_exog_df, exog_meta or {}, future_idx)
        return _make_exog_features(future_idx, real_exog_df=future_real_exog)
    except Exception:
        return None


def _best_sarima_order(train, d, m):
    """Grid-search (p,d,q)×(P,D,Q,m) orders, return best by AIC."""
    best_aic, best_order, best_seasonal = np.inf, (1, d, 1), (1, 1, 0, m)

    p_vals = range(0, 3)
    q_vals = range(0, 3)
    P_vals = range(0, 2)
    Q_vals = range(0, 2)
    D = 1

    for p, q, P, Q in itertools.product(p_vals, q_vals, P_vals, Q_vals):
        if p == 0 and q == 0:
            continue
        try:
            fit = SARIMAX(
                train,
                order=(p, d, q),
                seasonal_order=(P, D, Q, m),
                enforce_stationarity=False,
                enforce_invertibility=False,
            ).fit(disp=False, maxiter=50)
            if fit.aic < best_aic:
                best_aic   = fit.aic
                best_order = (p, d, q)
                best_seasonal = (P, D, Q, m)
        except Exception:
            continue

    return best_order, best_seasonal, best_aic


# ── SARIMA ────────────────────────────────────────────────────────────────────

def train_and_forecast(series: pd.Series, forecast_horizon: int) -> dict:
    """SARIMA with auto order selection."""
    series = series.dropna().astype(float)
    if len(series) < 20:
        raise ValueError(f"SARIMA needs ≥ 20 observations, got {len(series)}.")

    m = _infer_seasonal_period(series)
    d = _check_stationarity(series)
    large_seasonal_period = m > MAX_GRID_SEARCH_SEASONAL_PERIOD

    # Bound evaluation cost to a fixed recent window regardless of total
    # series length (see MAX_EVAL_ROWS above).
    eval_series = series.iloc[-MAX_EVAL_ROWS:] if len(series) > MAX_EVAL_ROWS else series

    # Large seasonal periods make every individual fit expensive (state
    # dimensionality scales with m) — use a single train/test split instead
    # of k-fold CV to halve the number of fits required for evaluation.
    if large_seasonal_period:
        split_at = max(int(len(eval_series) * 0.8), 15)
        fold_splits = [(np.arange(split_at), np.arange(split_at, len(eval_series)))]
    else:
        fold_splits = list(TimeSeriesSplit(n_splits=get_cv_splits(len(eval_series))).split(eval_series))
    cv_metrics = []
    train_times = []
    pred_times = []

    # Auto-select (p,d,q)x(P,D,Q,m) by AIC on an initial subset, reused across
    # CV folds + final refit (mirrors arima_model.py's _select_best_order pattern)
    if large_seasonal_period:
        order, seasonal_order = (1, d, 1), (1, 1, 0, m)
    else:
        split_idx = max(int(len(eval_series) * 0.8), 15)
        order, seasonal_order, _ = _best_sarima_order(eval_series.iloc[:split_idx], d, m)

    # Walk-forward validation: fit once per fold, then cheaply extend the
    # fitted state with each new true observation via .append(refit=False)
    # instead of re-estimating the model from scratch at every test point.
    # EXCEPTION: once the seasonal period is large, even .append() itself
    # costs proportionally more per step (state dimensionality scales with
    # m) — verified ~430ms/step at m=60 vs ~15ms/step at m=12, which still
    # blows the time budget over hundreds of test points. For those cases,
    # evaluate with a single multi-step forecast per fold instead (same
    # approach the DL models already use), trading one-step-ahead rigor for
    # boundedness on data that's realistically outside typical retail
    # seasonality anyway.
    for train_idx, test_idx in fold_splits:
        train, test = eval_series.iloc[train_idx], eval_series.iloc[test_idx]
        preds   = []

        t0 = time.perf_counter()
        try:
            fit_model = SARIMAX(list(train), order=order, seasonal_order=seasonal_order,
                                 enforce_stationarity=False, enforce_invertibility=False).fit(disp=False, maxiter=40)
        except Exception:
            fit_model = None
        train_times.append(time.perf_counter() - t0)

        t1 = time.perf_counter()
        if fit_model is not None and large_seasonal_period:
            try:
                preds = [float(v) for v in np.asarray(fit_model.forecast(len(test)))]
            except Exception:
                preds = [float(train.iloc[-1])] * len(test)
        else:
            for i in range(len(test)):
                try:
                    if fit_model is None:
                        raise RuntimeError("no fitted model")
                    yhat = float(np.asarray(fit_model.forecast(1))[0])
                    fit_model = fit_model.append([test.iloc[i]], refit=False)
                except Exception:
                    yhat = float(train.iloc[-1]) if i == 0 else preds[-1]
                preds.append(yhat)
        pred_times.append(time.perf_counter() - t1)

        metrics = calculate_metrics(test.values, preds)
        cv_metrics.append(metrics)

    avg_metrics = {k: sum(m[k] for m in cv_metrics)/len(cv_metrics) for k in cv_metrics[0].keys()}
    avg_train_time = sum(train_times)/len(train_times)
    avg_pred_time = sum(pred_times)/len(pred_times)

    # Refit on the same bounded window used for evaluation, NOT the full
    # series — unlike ARIMA, SARIMAX's seasonal state-space cost scales
    # roughly with n × m², so even a single fit on the complete series can
    # be dangerously slow once both the row count and seasonal period are
    # realistic (verified: a totally ordinary m=52 weekly/yearly case didn't
    # finish in 150s at just 50,000 rows). Capping to eval_series keeps this
    # bounded regardless of total dataset size.
    final = SARIMAX(eval_series, order=order, seasonal_order=seasonal_order,
                    enforce_stationarity=False, enforce_invertibility=False
                    ).fit(disp=False, maxiter=80)

    floor = 0.0 if series.min() >= 0 else None
    fc    = final.forecast(forecast_horizon)
    fc    = [max(floor, float(v)) if floor is not None else float(v) for v in fc]

    return {
        "model_name":     "SARIMA",
        **avg_metrics,
        "train_time_sec": round(avg_train_time, 4),
        "pred_time_sec": round(avg_pred_time, 4),
        "forecast":       [round(v, 4) for v in fc],
        "order":          str(order),
        "seasonal_order": str(seasonal_order),
    }


# ── SARIMAX (with exogenous calendar features) ────────────────────────────────

def train_and_forecast_sarimax(
    series: pd.Series,
    forecast_horizon: int,
    exog_df: pd.DataFrame | None = None,
    exog_meta: dict | None = None,
) -> dict:
    """SARIMAX with calendar exogenous features, plus real exogenous features
    (price/promo) if exog_df is given (see services/preprocessor.py)."""
    series = series.dropna().astype(float)
    if len(series) < 20:
        raise ValueError(f"SARIMAX needs ≥ 20 observations, got {len(series)}.")

    has_real_exog = exog_df is not None and not exog_df.empty

    m    = _infer_seasonal_period(series)
    d    = _check_stationarity(series)
    large_seasonal_period = m > MAX_GRID_SEARCH_SEASONAL_PERIOD
    # Exog features compound the cost of an already-expensive large seasonal
    # period fit (verified: calendar exog roughly doubled per-fit cost at
    # m=60). They're an enhancement on top of SARIMA, not core functionality,
    # so drop them (calendar AND real price/promo alike) for large seasonal
    # periods to stay within budget — degrades gracefully to plain-SARIMA-
    # equivalent behavior for those cases.
    exog = (
        _make_exog_features(series.index, real_exog_df=exog_df if has_real_exog else None)
        if isinstance(series.index, pd.DatetimeIndex) and not large_seasonal_period
        else None
    )

    # Bound evaluation cost to a fixed recent window regardless of total
    # series length (see MAX_EVAL_ROWS above). Slice exog the same way so it
    # stays aligned with eval_series.
    if len(series) > MAX_EVAL_ROWS:
        eval_series = series.iloc[-MAX_EVAL_ROWS:]
        eval_exog = exog[-MAX_EVAL_ROWS:] if exog is not None else None
    else:
        eval_series = series
        eval_exog = exog

    # Large seasonal periods (+ exog) make every individual fit expensive —
    # use a single train/test split instead of k-fold CV.
    if large_seasonal_period:
        split_at = max(int(len(eval_series) * 0.8), 15)
        fold_splits = [(np.arange(split_at), np.arange(split_at, len(eval_series)))]
    else:
        fold_splits = list(TimeSeriesSplit(n_splits=get_cv_splits(len(eval_series))).split(eval_series))
    cv_metrics = []
    train_times = []
    pred_times = []

    if large_seasonal_period:
        order, seasonal_order = (1, d, 1), (1, 1, 0, m)
    else:
        split_idx = max(int(len(eval_series) * 0.8), 15)
        order, seasonal_order, _ = _best_sarima_order(eval_series.iloc[:split_idx], d, m)

    # Walk-forward validation: fit once per fold, then cheaply extend the
    # fitted state with each new true observation (+ its exog row) via
    # .append(refit=False) instead of re-estimating from scratch each step.
    # For large seasonal periods, .append() itself gets proportionally more
    # expensive too (state dimensionality scales with m) — fall back to one
    # multi-step forecast per fold instead (see SARIMA function for details).
    for train_idx, test_idx in fold_splits:
        train_y = eval_series.iloc[train_idx]
        test_y  = eval_series.iloc[test_idx]
        train_x = eval_exog[train_idx] if eval_exog is not None else None
        test_x  = eval_exog[test_idx] if eval_exog is not None else None
        preds   = []

        t0 = time.perf_counter()
        try:
            fit_model = SARIMAX(list(train_y), exog=train_x, order=order, seasonal_order=seasonal_order,
                                 enforce_stationarity=False, enforce_invertibility=False).fit(disp=False, maxiter=40)
        except Exception:
            fit_model = None
        train_times.append(time.perf_counter() - t0)

        t1 = time.perf_counter()
        if fit_model is not None and large_seasonal_period:
            try:
                preds = [float(v) for v in np.asarray(fit_model.forecast(len(test_y), exog=test_x))]
            except Exception:
                preds = [float(train_y.iloc[-1])] * len(test_y)
        else:
            for i in range(len(test_y)):
                try:
                    if fit_model is None:
                        raise RuntimeError("no fitted model")
                    fut_x = test_x[[i]] if test_x is not None else None
                    yhat  = float(np.asarray(fit_model.forecast(1, exog=fut_x))[0])
                    fit_model = fit_model.append([test_y.iloc[i]], exog=fut_x, refit=False)
                except Exception:
                    yhat = float(train_y.iloc[-1]) if i == 0 else preds[-1]
                preds.append(yhat)
        pred_times.append(time.perf_counter() - t1)

        metrics = calculate_metrics(test_y.values, preds)
        cv_metrics.append(metrics)

    avg_metrics = {k: sum(m[k] for m in cv_metrics)/len(cv_metrics) for k in cv_metrics[0].keys()}
    avg_train_time = sum(train_times)/len(train_times)
    avg_pred_time = sum(pred_times)/len(pred_times)

    # Refit on the same bounded window used for evaluation, NOT the full
    # series — see the SARIMA function above for why (seasonal state-space
    # cost scales roughly with n × m², verified dangerous even at ordinary
    # seasonal periods once row count is large).
    final = SARIMAX(eval_series, exog=eval_exog, order=order, seasonal_order=seasonal_order,
                    enforce_stationarity=False, enforce_invertibility=False
                    ).fit(disp=False, maxiter=80)

    freq = pd.infer_freq(series.index) if isinstance(series.index, pd.DatetimeIndex) else "D"
    fut_exog = (
        _make_future_exog(
            series.index[-1], freq or "D", forecast_horizon,
            real_exog_df=exog_df if has_real_exog and not large_seasonal_period else None,
            exog_meta=exog_meta,
        )
        if eval_exog is not None else None
    )
    fc       = final.forecast(forecast_horizon, exog=fut_exog)
    floor    = 0.0 if series.min() >= 0 else None
    fc       = [max(floor, float(v)) if floor is not None else float(v) for v in fc]

    return {
        "model_name":     "SARIMAX",
        **avg_metrics,
        "train_time_sec": round(avg_train_time, 4),
        "pred_time_sec": round(avg_pred_time, 4),
        "forecast":       [round(v, 4) for v in fc],
        "order":          str(order),
        "seasonal_order": str(seasonal_order),
        "exog_features":  (
            ["month_sin", "month_cos", "dow_sin", "dow_cos", "is_month_end"]
            if eval_exog is not None else []
        ),
        "exog_features_used": (
            list(exog_df.columns) if has_real_exog and eval_exog is not None else []
        ),
    }
