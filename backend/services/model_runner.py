"""
model_runner.py — Parallel model orchestration, common-backtest ranking and
ensembling.

Registry (15 models + an ensemble built per request)
  Statistical : ARIMA, SARIMA, SARIMAX, HoltWinters, ExponentialSmoothing,
                MovingAverage, Prophet, Theta, AutoETS, MSTL
  ML          : XGBoost, RandomForest, SVM
  Intermittent: CrostonSBA (sparse/zero-inflated demand)
  Baseline    : SeasonalNaive
  Ensemble    : mean of the top-3 backtested models (added after ranking)

Removed (benchmark-driven)
  RNN, LSTM, GRU, BiLSTM, CNN1D, TCN, Transformer, KNN. On a held-out
  benchmark — 10 series (5 real uploads, 5 synthetic with known structure),
  every model trained without the last 28 points and scored on them by MASE —
  all seven deep-learning models had a geometric-mean MASE of 1.35-1.68, i.e.
  WORSE than simply repeating last season (1.17), while also being the
  slowest models in the registry. KNN ranked last among the ML models (mean
  rank 11.3, geo-mean MASE 1.165, no better than seasonal naive). The
  full table is in README.md, "What changed in v0.2".

Why a common backtest
  Each model also computes its own internal CV metrics, but they were never
  comparable: XGBoost / RandomForest / SVM / ARIMA scored themselves on
  ONE-step-ahead errors (true lags fed into every test step), whereas
  HoltWinters / Prophet / Croston scored genuine multi-step forecasts. Picking
  the lowest of those numbers systematically favoured the models with the
  easier test, not the ones that forecast the requested horizon best. Now
  every model is re-run from the same forecast origins, forecasting the same
  windows multi-step exactly as it will in production, and ranked on MASE
  over those windows. The internal metrics are kept as `internal_cv`.
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import os
import pickle
import threading
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, wait
from concurrent.futures.process import BrokenProcessPool

import numpy as np
import pandas as pd

# ── Statistical models ────────────────────────────────────────────────────────
from models.arima_model       import train_and_forecast         as arima_forecast
from models.sarima_model      import train_and_forecast         as sarima_forecast
from models.sarima_model      import train_and_forecast_sarimax as sarimax_forecast
from models.sarima_model      import _infer_seasonal_period
from models.holtwinters_model import train_and_forecast         as holtwinters_forecast
from models.holtwinters_model import train_and_forecast_exp     as exp_smoothing_forecast
from models.prophet_model     import train_and_forecast         as prophet_forecast
from models.advanced_stat_models import (
    train_and_forecast_theta   as theta_forecast,
    train_and_forecast_autoets as autoets_forecast,
    train_and_forecast_mstl    as mstl_forecast,
    train_and_forecast_snaive  as snaive_forecast,
)

# ── ML models ─────────────────────────────────────────────────────────────────
from models.xgboost_model     import train_and_forecast         as xgboost_forecast
from models.rf_model          import train_and_forecast         as rf_forecast
from models.ml_models         import train_and_forecast_ma      as ma_forecast
from models.ml_models         import train_and_forecast_svm     as svm_forecast

# ── Intermittent-demand model ────────────────────────────────────────────────
from models.croston_model     import train_and_forecast         as croston_forecast

from services.metrics import calculate_metrics, get_stability_penalty

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Timeouts / parallelism ────────────────────────────────────────────────────
STAT_TIMEOUT = 120
ML_TIMEOUT   = 180
MAX_WORKERS  = max(2, min(8, (os.cpu_count() or 4) - 1))

# Models run in a persistent PROCESS pool, not threads. statsmodels' Cython
# state-space code (ARIMA/SARIMA/ETS) holds the GIL, so a thread pool ran the
# registry essentially one model at a time: a 731-row daily series took ~100s
# wall-clock for ~100s of total CPU. Separate processes give real parallelism.
# Registries that can't be pickled (e.g. test fakes defined as closures) fall
# back to threads automatically.
USE_PROCESS_POOL = os.environ.get("FORECAST_USE_PROCESSES", "1") != "0"
_process_pool: ProcessPoolExecutor | None = None
_pool_lock = threading.Lock()


def _worker_init() -> None:
    """One BLAS/OpenMP thread per worker process — parallelism comes from the
    pool itself, and nested multi-threading only oversubscribes the CPU."""
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[var] = "1"
    try:
        from threadpoolctl import threadpool_limits
        global _THREAD_LIMITS
        _THREAD_LIMITS = threadpool_limits(1)  # keep a reference so it stays applied
    except Exception:
        pass
    logging.getLogger("cmdstanpy").setLevel(logging.WARNING)


def _get_process_pool() -> ProcessPoolExecutor:
    global _process_pool
    with _pool_lock:
        if _process_pool is None:
            _process_pool = ProcessPoolExecutor(
                max_workers=MAX_WORKERS,
                mp_context=mp.get_context("spawn"),
                initializer=_worker_init,
            )
        return _process_pool


def _discard_process_pool() -> None:
    """Kill the pool's workers (used after a timeout so a stuck model can't
    keep occupying a worker for later requests) — a fresh pool is created on
    the next call."""
    global _process_pool
    with _pool_lock:
        pool, _process_pool = _process_pool, None
    if pool is None:
        return
    for proc in list(getattr(pool, "_processes", {}).values()):
        try:
            proc.terminate()
        except Exception:
            pass
    pool.shutdown(wait=False, cancel_futures=True)


def warm_up() -> None:
    """Start the worker processes and import the model stack in them, so the
    first forecast request doesn't pay the spawn + import cost."""
    if not USE_PROCESS_POOL:
        return
    try:
        pool = _get_process_pool()
        futures = [pool.submit(_noop) for _ in range(MAX_WORKERS)]
        wait(futures, timeout=120)
        for f in futures:
            f.result(timeout=0)
        logger.info(f"Model process pool ready ({MAX_WORKERS} workers).")
    except Exception as exc:
        logger.warning(f"Model process pool warm-up failed ({exc}); will retry on first request.")
        _discard_process_pool()


def _noop() -> bool:
    import models.prophet_model  # noqa: F401  (heaviest import)
    import models.xgboost_model  # noqa: F401
    return True


def _picklable(obj) -> bool:
    try:
        pickle.dumps(obj)
        return True
    except Exception:
        return False

# ── Backtest / ensemble configuration ─────────────────────────────────────────
BACKTEST_MAX_ORIGINS = 2      # forecast origins per model (each = one refit)
BACKTEST_MIN_TRAIN   = 30     # never backtest from fewer training points
ENSEMBLE_SIZE        = 3      # top-k models averaged into "Ensemble"
INTERVAL_Z           = 1.96   # ≈95% normal interval

# Models train on at most this many of the MOST RECENT observations. A
# 49,782-row hourly series (one transaction per hour for 5.7 years) made
# SARIMA/SARIMAX run for 6+ minutes and RandomForest/AutoETS for 1-2 minutes
# each, for no accuracy benefit: a 30-step forecast depends on recent
# dynamics, and 2,000 points still covers ~83 days of hourly data (12 weekly
# cycles) or ~5.5 years of daily data (5 yearly cycles) — enough for every
# model's seasonality, including MSTL's daily+yearly / hourly+weekly pairs.
MAX_TRAIN_ROWS = 2000

# Optional training-time budget for selection (seconds). None = accuracy only.
# See tests/test_selection_policy.py for the measured rationale.
SELECTION_TIME_BUDGET_SEC: float | None = None


def _infer_frequency_tier(series: pd.Series) -> str:
    """Coarse frequency label, reported in run_info for the UI/explainer."""
    if not isinstance(series.index, pd.DatetimeIndex):
        return "unknown"
    try:
        freq = pd.infer_freq(series.index)
        if freq is None:
            return "unknown"
        freq = freq.upper()
        if freq.startswith("MIN") or freq.startswith("T") or freq.startswith("S"):
            return "minutely"
        for prefix, tier in (("H", "hourly"), ("D", "daily"), ("B", "daily"),
                             ("W", "weekly"), ("M", "monthly"), ("Q", "quarterly"),
                             ("A", "yearly"), ("Y", "yearly")):
            if freq.startswith(prefix):
                return tier
    except Exception:
        pass
    return "unknown"


# ── Full model registry ───────────────────────────────────────────────────────
# Format: "DisplayName": (function, timeout_seconds, "category")
MODEL_REGISTRY: dict[str, tuple] = {
    # Statistical
    "ARIMA":                (arima_forecast,         STAT_TIMEOUT, "statistical"),
    "SARIMA":               (sarima_forecast,        STAT_TIMEOUT, "statistical"),
    "SARIMAX":              (sarimax_forecast,       STAT_TIMEOUT, "statistical"),
    "HoltWinters":          (holtwinters_forecast,   STAT_TIMEOUT, "statistical"),
    "ExponentialSmoothing": (exp_smoothing_forecast, STAT_TIMEOUT, "statistical"),
    "MovingAverage":        (ma_forecast,            STAT_TIMEOUT, "statistical"),
    "Prophet":              (prophet_forecast,       STAT_TIMEOUT, "statistical"),
    "Theta":                (theta_forecast,         STAT_TIMEOUT, "statistical"),
    "AutoETS":              (autoets_forecast,       STAT_TIMEOUT, "statistical"),
    "MSTL":                 (mstl_forecast,          STAT_TIMEOUT, "statistical"),
    # ML
    "XGBoost":              (xgboost_forecast,       ML_TIMEOUT,   "ml"),
    "RandomForest":         (rf_forecast,            ML_TIMEOUT,   "ml"),
    "SVM":                  (svm_forecast,           ML_TIMEOUT,   "ml"),
    # Intermittent demand
    "CrostonSBA":           (croston_forecast,       STAT_TIMEOUT, "intermittent"),
    # Baseline
    "SeasonalNaive":        (snaive_forecast,        STAT_TIMEOUT, "baseline"),
}


# ── Backtest plan ─────────────────────────────────────────────────────────────

def _season_for_mase(series: pd.Series) -> int:
    m = _infer_seasonal_period(series)
    return m if 1 < m < len(series) // 2 else 1


def _backtest_plan(n_obs: int, horizon: int) -> tuple[int, list[int]]:
    """
    (backtest_horizon, origins). Each origin is a cut index: the model is fit
    on series[:cut] and must forecast series[cut:cut+h]. Origins are
    non-overlapping windows at the end of the series, newest first.
    """
    h = int(min(horizon, max(1, n_obs // 5)))
    origins = []
    for k in range(1, BACKTEST_MAX_ORIGINS + 1):
        cut = n_obs - k * h
        if cut < BACKTEST_MIN_TRAIN:
            break
        origins.append(cut)
    return h, origins


def _mase_denominator(train: np.ndarray, m: int) -> float:
    if len(train) <= m:
        m = 1
    d = np.mean(np.abs(train[m:] - train[:-m])) if len(train) > m else 0.0
    if not np.isfinite(d) or d <= 1e-12:
        d = np.mean(np.abs(np.diff(train))) if len(train) > 1 else 0.0
    return float(d) if np.isfinite(d) and d > 1e-12 else 1.0


def _score_backtest(series: pd.Series, h: int, origins: list[int],
                    forecasts: list[list[float]], m: int) -> dict:
    """Pool errors over all origins into one comparable metric set."""
    y = series.values.astype(float)
    actual_all, pred_all, mases, sq_by_step = [], [], [], []
    for cut, fc in zip(origins, forecasts):
        actual = y[cut:cut + h]
        pred = np.asarray(fc, dtype=float)[:h]
        if len(pred) < len(actual) or not np.all(np.isfinite(pred)):
            raise ValueError("backtest forecast unusable")
        actual_all.append(actual)
        pred_all.append(pred)
        mases.append(float(np.mean(np.abs(actual - pred))) / _mase_denominator(y[:cut], m))
        sq_by_step.append((actual - pred) ** 2)
    a, p = np.concatenate(actual_all), np.concatenate(pred_all)
    metrics = calculate_metrics(a, p)
    # Error scale per horizon step, assuming sigma_h = c * sqrt(h): fit c on
    # the observed multi-step squared errors. Used for honest intervals.
    steps = np.arange(1, h + 1, dtype=float)
    sq = np.mean(np.vstack(sq_by_step), axis=0)
    c = float(np.sqrt(np.mean(sq / steps)))
    return {**metrics, "mase": float(np.mean(mases)), "sigma_scale": c}


def _attach_intervals(result: dict, series: pd.Series) -> None:
    fc = result.get("forecast")
    if not fc:
        result["forecast_lower"] = result["forecast_upper"] = None
        return
    fc = np.asarray(fc, dtype=float)
    steps = np.sqrt(np.arange(1, len(fc) + 1))
    if np.isfinite(result.get("sigma_scale", np.nan)):
        sigma = result["sigma_scale"] * steps
        method = "backtest_calibrated"
    elif "forecast_lower" in result and "forecast_upper" in result:
        result.setdefault("interval_method", "model_native")
        return
    elif np.isfinite(result.get("rmse", np.nan)):
        sigma = float(result["rmse"]) * steps
        method = "normal_approximation_rmse"
    else:
        result["forecast_lower"] = result["forecast_upper"] = None
        return
    lower, upper = fc - INTERVAL_Z * sigma, fc + INTERVAL_Z * sigma
    if float(series.min()) >= 0:
        lower = np.clip(lower, 0, None)
    result["forecast_lower"] = [round(float(v), 4) for v in lower]
    result["forecast_upper"] = [round(float(v), 4) for v in upper]
    result["interval_method"] = method


# ── Task runners ──────────────────────────────────────────────────────────────

def _run_single_model(name, fn, timeout, series, forecast_horizon) -> dict:
    start = time.perf_counter()
    logger.info(f"[{name}] Starting | obs={len(series)} | horizon={forecast_horizon}")
    try:
        result = fn(series, forecast_horizon)
        elapsed = round(time.perf_counter() - start, 2)
        if result.get("forecast") is not None:
            result["forecast"] = [float(v) for v in result["forecast"]]
        logger.info(f"[{name}] ✓ Done {elapsed}s")
        return {**result, "model_name": name, "status": "success",
                "elapsed_sec": elapsed, "error": None}
    except Exception as exc:
        elapsed = round(time.perf_counter() - start, 2)
        logger.error(f"[{name}] ✗ Failed {elapsed}s — {exc}\n{traceback.format_exc()}")
        return {"model_name": name, "status": "failed", "elapsed_sec": elapsed,
                "rmse": float("inf"), "forecast": [], "error": str(exc)}


def _run_backtest_fit(fn, series: pd.Series, cut: int, h: int) -> list[float] | None:
    try:
        out = fn(series.iloc[:cut], h)
        fc = [float(v) for v in (out.get("forecast") or [])][:h]
        return fc if len(fc) == h and all(np.isfinite(fc)) else None
    except Exception as exc:
        logger.info(f"[backtest] fit at cut={cut} failed: {exc}")
        return None


# ── Public API ────────────────────────────────────────────────────────────────

def run_all_models(
    series: pd.Series,
    forecast_horizon: int,
    include_dl: bool = True,          # kept for API compatibility; no DL models remain
    categories: list[str] | None = None,
    models: dict | None = None,
    backtest: bool = True,
    ensemble: bool = True,
    _force_threads: bool = False,
) -> tuple[list[dict], dict]:
    """
    Run all registered models in parallel, score them on a common holdout
    backtest, add a top-k ensemble, and return results best-first.

    Returns (results, run_info). Each successful result carries rmse/mae/mape/
    r2/mase measured on the backtest windows (`evaluation="holdout_backtest"`)
    and its own internal CV metrics under `internal_cv`. If the series is too
    short to backtest, or `backtest=False`, ranking falls back to each model's
    internal CV RMSE (`evaluation="internal_cv"`).
    """
    if not isinstance(series, pd.Series):
        raise TypeError("series must be a pd.Series.")
    if len(series) < 10:
        raise ValueError(f"Need ≥ 10 observations, got {len(series)}.")
    if forecast_horizon < 1:
        raise ValueError(f"forecast_horizon must be ≥ 1, got {forecast_horizon}.")

    series = series.dropna().sort_index()
    original_rows = len(series)
    if original_rows > MAX_TRAIN_ROWS:
        series = series.iloc[-MAX_TRAIN_ROWS:]
        logger.info(f"[WINDOW] Training on the most recent {MAX_TRAIN_ROWS} of {original_rows} observations.")
    row_count = len(series)

    registry = models or MODEL_REGISTRY
    if not include_dl:
        registry = {k: v for k, v in registry.items() if v[2] != "dl"}
    if categories:
        registry = {k: v for k, v in registry.items() if v[2] in categories}

    bt_h, origins = _backtest_plan(row_count, forecast_horizon) if backtest else (0, [])
    use_backtest = bool(origins)
    m = _season_for_mase(series)

    logger.info(
        f"Launching {len(registry)} models | obs={row_count} | horizon={forecast_horizon} | "
        f"backtest={'h=%d origins=%s' % (bt_h, origins) if use_backtest else 'off'}"
    )

    results: list[dict] = []
    main_futures, bt_futures = {}, {}
    use_processes = (USE_PROCESS_POOL and not _force_threads
                     and _picklable(registry) and _picklable(series))
    if use_processes:
        executor = _get_process_pool()
    else:
        # Not a context manager: shutdown(wait=True) would block the request
        # on any model that has already timed out.
        executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)
    timed_out = False
    try:
        for name, (fn, timeout, _cat) in registry.items():
            main_futures[executor.submit(_run_single_model, name, fn, timeout,
                                         series, forecast_horizon)] = name
            for cut in origins:
                bt_futures[executor.submit(_run_backtest_fit, fn, series, cut, bt_h)] = (name, cut)

        timeouts = [t for (_f, t, _c) in registry.values()] or [STAT_TIMEOUT]
        # Tasks are spread over the pool; allow for every fit of the slowest
        # model running back to back, plus slack.
        global_timeout = max(timeouts) * (1 + len(origins)) + 30
        done, not_done = wait(list(main_futures) + list(bt_futures), timeout=global_timeout)
        timed_out = bool(not_done)

        for fut, name in main_futures.items():
            if fut in done:
                try:
                    results.append(fut.result())
                except Exception as exc:
                    results.append({"model_name": name, "status": "failed", "rmse": float("inf"),
                                    "forecast": [], "elapsed_sec": 0, "error": str(exc)})
            else:
                logger.error(f"[{name}] Timed out after {global_timeout}s")
                results.append({"model_name": name, "status": "timeout", "rmse": float("inf"),
                                "forecast": [], "elapsed_sec": global_timeout,
                                "error": f"Timed out after {global_timeout}s"})
                fut.cancel()

        bt_forecasts: dict[str, dict[int, list[float]]] = {}
        for fut, (name, cut) in bt_futures.items():
            fc = None
            if fut in done:
                try:
                    fc = fut.result()
                except Exception as exc:
                    logger.info(f"[{name}] backtest task failed: {exc}")
            if fc is not None:
                bt_forecasts.setdefault(name, {})[cut] = fc
    except BrokenProcessPool as exc:
        # A crashed worker breaks the whole pool. Reset it for the next
        # request and finish this one on threads rather than failing it.
        logger.error(f"Process pool broke ({exc}); retrying this run on threads.")
        _discard_process_pool()
        return run_all_models(series, forecast_horizon, include_dl, categories, models,
                              backtest, ensemble, _force_threads=True)
    finally:
        if use_processes:
            if timed_out:
                _discard_process_pool()
        else:
            executor.shutdown(wait=False, cancel_futures=True)

    # ── Replace internal metrics with common-backtest metrics ──
    for r in results:
        if r.get("status") != "success":
            continue
        r["internal_cv"] = {k: r.get(k) for k in ("rmse", "mae", "mape", "r2")}
        r["evaluation"] = "internal_cv"
        if use_backtest:
            per_origin = bt_forecasts.get(r["model_name"], {})
            if all(c in per_origin for c in origins):
                try:
                    fcs = [per_origin[c] for c in origins]
                    r.update(_score_backtest(series, bt_h, origins, fcs, m))
                    r["backtest_forecasts"] = fcs
                    r["evaluation"] = "holdout_backtest"
                except Exception as exc:
                    logger.info(f"[{r['model_name']}] backtest scoring failed: {exc}")

    _rank(results, series, use_backtest)

    # ── Ensemble of the top-k backtested models ──
    if ensemble and use_backtest:
        ens = _build_ensemble(results, series, bt_h, origins, m)
        if ens is not None:
            results.append(ens)
            _rank(results, series, use_backtest)

    for r in results:
        if r.get("status") == "success":
            _attach_intervals(r, series)
            r.pop("sigma_scale", None)

    _log_leaderboard(results)

    run_info = {
        "row_count": row_count,
        "original_row_count": original_rows,
        "training_window_truncated": original_rows > row_count,
        "frequency_tier": _infer_frequency_tier(series),
        "seasonal_period": m,
        "evaluation": "holdout_backtest" if use_backtest else "internal_cv",
        "backtest_horizon": bt_h if use_backtest else None,
        "backtest_origins": len(origins) if use_backtest else 0,
        "models_run": len(registry),
        # Kept so older clients reading these keys don't break.
        "dl_skipped": False,
        "dl_skip_reason": None,
    }
    return results, run_info


# ── Ranking ───────────────────────────────────────────────────────────────────

def _robust_scale(values) -> float:
    """
    Tukey-fence denominator (Q3 + 1.5·IQR) so one catastrophic model can't set
    the scale for everyone else. Only used on the internal-CV fallback path;
    backtest MASE is already scale-free.
    """
    finite = [v for v in values if np.isfinite(v)]
    if not finite:
        return 1e-6
    arr = np.array(finite, dtype=float)
    q1, q3 = np.percentile(arr, [25, 75])
    fence = q3 + 1.5 * (q3 - q1)
    return float(fence) if fence > 1e-6 else float(arr.max()) or 1e-6


def _is_scorable(r: dict) -> bool:
    return r.get("status") == "success" and bool(r.get("forecast"))


def _train_cost(r: dict) -> float:
    return float(r.get("train_time_sec", r.get("elapsed_sec", 0.0)) or 0.0)


def _rank(results: list[dict], series: pd.Series, use_backtest: bool) -> None:
    try:
        last_val = float(series.iloc[-1])
        hist_std = max(float(series.std()), 1e-6)
        valid = [r for r in results if _is_scorable(r)]

        for r in valid:
            try:
                stab = get_stability_penalty(last_val, hist_std, list(r["forecast"]))
            except Exception:
                stab = float("inf")
            r["stability_penalty"] = round(float(stab), 4) if np.isfinite(stab) else float("inf")

        # Primary key: backtest MASE where available. Models whose backtest
        # failed are ranked after every backtested model, by internal RMSE.
        normalized: dict[int, float] = {}
        backtested = [r for r in valid if r.get("evaluation") == "holdout_backtest"]
        others = [r for r in valid if r.get("evaluation") != "holdout_backtest"]
        for r in backtested:
            normalized[id(r)] = float(r["mase"])
        if others:
            scale = _robust_scale([float(r.get("rmse", np.inf)) for r in others])
            offset = (max(normalized.values()) + 1.0) if (use_backtest and normalized) else 0.0
            for r in others:
                normalized[id(r)] = float(r.get("rmse", np.inf)) / scale + offset

        finite_norms = [v for v in normalized.values() if np.isfinite(v)]
        over_budget_offset = (max(finite_norms) + 1.0) if finite_norms else 1.0

        budget = SELECTION_TIME_BUDGET_SEC
        affordable = ([r for r in valid if _train_cost(r) <= budget]
                      if budget is not None else valid)
        budget_active = bool(budget is not None and affordable)
        affordable_ids = {id(r) for r in affordable}
        if budget is not None and not affordable and valid:
            cheapest = min(valid, key=_train_cost)
            logger.warning(f"[SELECTION] No model within {budget}s; using cheapest "
                           f"({cheapest['model_name']}).")
            affordable_ids, budget_active = {id(cheapest)}, True

        for r in results:
            if not _is_scorable(r) or id(r) not in normalized:
                r["adjusted_score"] = float("inf")
                continue
            score = normalized[id(r)]
            within = (not budget_active) or (id(r) in affordable_ids)
            if not within:
                score += over_budget_offset
            r["within_time_budget"] = bool(within)
            r["adjusted_score"] = round(float(score), 6) if np.isfinite(score) else float("inf")
    except Exception as e:
        logger.error(f"Error ranking models: {e}", exc_info=True)
        for r in results:
            r.setdefault("adjusted_score", float("inf"))

    results.sort(key=lambda r: r.get("adjusted_score", float("inf"))
                 if np.isfinite(r.get("adjusted_score", float("inf"))) else float("inf"))


def _build_ensemble(results, series, h, origins, m) -> dict | None:
    """
    Equal-weight mean of the top-k backtested models. Combining forecasts is
    the most reliable accuracy gain in the forecasting literature (M4: most
    top entries were combinations). Note the members are chosen on the same
    backtest the ensemble is scored on, so its backtest score is mildly
    optimistic; the held-out benchmark is the honest check.
    """
    members = [r for r in results
               if r.get("evaluation") == "holdout_backtest" and _is_scorable(r)
               and r.get("within_time_budget", True) and r["model_name"] != "SeasonalNaive"
               and np.isfinite(r.get("adjusted_score", np.inf))][:ENSEMBLE_SIZE]
    if len(members) < 2:
        return None
    try:
        fc = np.mean([np.asarray(r["forecast"], float) for r in members], axis=0)
        bt = [np.mean([np.asarray(r["backtest_forecasts"][i], float) for r in members], axis=0).tolist()
              for i in range(len(origins))]
        scored = _score_backtest(series, h, origins, bt, m)
    except Exception as exc:
        logger.info(f"[Ensemble] skipped: {exc}")
        return None
    return {
        "model_name": "Ensemble",
        "status": "success",
        "error": None,
        **scored,
        "forecast": [round(float(v), 4) for v in fc],
        "backtest_forecasts": bt,
        "evaluation": "holdout_backtest",
        "ensemble_members": [r["model_name"] for r in members],
        "train_time_sec": round(sum(_train_cost(r) for r in members), 4),
        "pred_time_sec": 0.0,
        "elapsed_sec": round(max(float(r.get("elapsed_sec", 0)) for r in members), 2),
    }


def _log_leaderboard(results: list[dict]) -> None:
    logger.info("─" * 78)
    logger.info(f"{'Rank':<5} {'Model':<22} {'MASE':<8} {'RMSE':<12} {'MAE':<12} {'Eval':<16} Time")
    for i, r in enumerate(results, 1):
        mase = f"{r['mase']:.3f}" if np.isfinite(r.get("mase", np.nan)) else "-"
        rmse = f"{r.get('rmse', np.inf):.2f}" if np.isfinite(r.get("rmse", np.inf)) else "-"
        mae = f"{r.get('mae', np.inf):.2f}" if np.isfinite(r.get("mae", np.inf)) else "-"
        logger.info(f"{i:<5} {r['model_name']:<22} {mase:<8} {rmse:<12} {mae:<12} "
                    f"{r.get('evaluation', r.get('status')):<16} {r.get('elapsed_sec', 0)}s")
    logger.info("─" * 78)
