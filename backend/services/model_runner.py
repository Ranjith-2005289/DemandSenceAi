"""
model_runner.py — Parallel model orchestration (ALL 19 MODELS)
Statistical : ARIMA, SARIMA, SARIMAX, HoltWinters, ExponentialSmoothing, MovingAverage, Prophet
ML          : XGBoost, RandomForest, KNN, SVM
DL          : RNN, LSTM, GRU, BiLSTM, CNN1D, TCN, Transformer
Intermittent: CrostonSBA (sparse/zero-inflated demand)

Runs every model concurrently via ThreadPoolExecutor.
Returns results sorted by RMSE ascending (best model first).
"""

import time
import logging
import traceback
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor

# ── Statistical models ────────────────────────────────────────────────────────
from models.arima_model      import train_and_forecast          as arima_forecast
from models.sarima_model     import train_and_forecast          as sarima_forecast
from models.sarima_model     import train_and_forecast_sarimax  as sarimax_forecast
from models.holtwinters_model import train_and_forecast         as holtwinters_forecast
from models.holtwinters_model import train_and_forecast_exp     as exp_smoothing_forecast
from models.prophet_model     import train_and_forecast         as prophet_forecast

# ── ML models ─────────────────────────────────────────────────────────────────
from models.xgboost_model    import train_and_forecast          as xgboost_forecast
from models.rf_model         import train_and_forecast          as rf_forecast
from models.ml_models        import train_and_forecast_ma       as ma_forecast
from models.ml_models        import train_and_forecast_knn      as knn_forecast
from models.ml_models        import train_and_forecast_svm      as svm_forecast

# ── Intermittent-demand model ────────────────────────────────────────────────
from models.croston_model    import train_and_forecast          as croston_forecast

# ── Deep Learning models ──────────────────────────────────────────────────────
from models.dl_models import (
    train_and_forecast_rnn         as rnn_forecast,
    train_and_forecast_lstm        as lstm_forecast,
    train_and_forecast_gru         as gru_forecast,
    train_and_forecast_bilstm      as bilstm_forecast,
    train_and_forecast_cnn         as cnn_forecast,
    train_and_forecast_tcn         as tcn_forecast,
    train_and_forecast_transformer as transformer_forecast,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Timeouts ──────────────────────────────────────────────────────────────────
STAT_TIMEOUT = 120     # seconds — statistical models (reduced from 180)
ML_TIMEOUT   = 180     # ML models (reduced from 240)
DL_TIMEOUT   = 120     # DL models (reduced from 600 — skip if too slow)
MAX_WORKERS  = 8       # threads (I/O-bound; GIL released by numpy/torch)

# Deep learning sequence models (30-step lookback here) need enough distinct
# training windows to learn real patterns rather than overfit noise — they
# reliably underperform simpler statistical/ML models on small datasets. This
# is a hard constraint, not a heuristic: below the relevant row count, DL
# models are never attempted at all (rather than trained anyway and only
# flagged as unreliable after the fact), saving both compute time and a
# misleading leaderboard entry.
#
# The threshold is frequency-aware rather than a single flat number: a naive
# ratio against a "seasonal period" doesn't work here because that period's
# meaning isn't consistent across frequency tiers (e.g. 24 means "hours per
# day" at the hourly tier but 7 means "days per week" at the daily tier), so
# scaling a row threshold by that number mixes incompatible units. Instead
# each tier has its own calibrated minimum, anchored so DAILY stays at 500
# rows (~1.5 years — validated against a real 366-row dataset that correctly
# skips DL). Lower-frequency tiers use realistically-achievable thresholds
# rather than a mechanical scale-up: 500 *months* of history is 41 years and
# will essentially never be reached, so demanding it would just mean DL never
# runs for monthly/quarterly data — this makes that trade-off explicit and
# tunable instead of an accidental, silent ban.
MIN_ROWS_FOR_DL_BY_TIER = {
    "minutely":  2000,  # need many days of minute data to see daily+weekly repeats
    "hourly":    1000,  # ~42 days — enough to see several weekly repeats
    "daily":     500,   # anchor — validated against a real dataset
    "weekly":    104,   # ~2 years — enough to see 2 yearly cycles
    "monthly":   36,    # ~3 years — thin, but a realistic ceiling for monthly retail data
    "quarterly": 20,    # ~5 years — DL is essentially never viable here; that's the honest answer
    "unknown":   500,   # fall back to the conservative daily anchor
}


def _infer_frequency_tier(series: pd.Series) -> str:
    """
    Map a series' inferred pandas frequency to a coarse tier for the DL
    row-count gate. Mirrors the prefix-matching approach used by
    sarima_model._infer_seasonal_period, but returns a tier label rather
    than a numeric seasonal period — the two aren't interchangeable since a
    "seasonal period" means a different unit at each tier.
    """
    if not isinstance(series.index, pd.DatetimeIndex):
        return "unknown"
    try:
        freq = pd.infer_freq(series.index)
        if freq is None:
            return "unknown"
        freq = freq.upper()
        if freq.startswith("MIN") or freq.startswith("T"): return "minutely"
        if freq.startswith("S"):  return "minutely"
        if freq.startswith("H"):  return "hourly"
        if freq.startswith("D"):  return "daily"
        if freq.startswith("W"):  return "weekly"
        if freq.startswith("M"):  return "monthly"
        if freq.startswith("Q"):  return "quarterly"
    except Exception:
        pass
    return "unknown"

# ── Full model registry ───────────────────────────────────────────────────────
# Format: "DisplayName": (function, timeout_seconds, "category")
MODEL_REGISTRY: dict[str, tuple] = {
    # Statistical
    "ARIMA":               (arima_forecast,        STAT_TIMEOUT, "statistical"),
    "SARIMA":              (sarima_forecast,        STAT_TIMEOUT, "statistical"),
    "SARIMAX":             (sarimax_forecast,       STAT_TIMEOUT, "statistical"),
    "HoltWinters":         (holtwinters_forecast,   STAT_TIMEOUT, "statistical"),
    "ExponentialSmoothing":(exp_smoothing_forecast, STAT_TIMEOUT, "statistical"),
    "MovingAverage":       (ma_forecast,            STAT_TIMEOUT, "statistical"),
    "Prophet":             (prophet_forecast,       STAT_TIMEOUT, "statistical"),
    # ML
    "XGBoost":             (xgboost_forecast,       ML_TIMEOUT,   "ml"),
    "RandomForest":        (rf_forecast,            ML_TIMEOUT,   "ml"),
    "KNN":                 (knn_forecast,           ML_TIMEOUT,   "ml"),
    "SVM":                 (svm_forecast,           ML_TIMEOUT,   "ml"),
    # Deep Learning
    "RNN":                 (rnn_forecast,           DL_TIMEOUT,   "dl"),
    "LSTM":                (lstm_forecast,          DL_TIMEOUT,   "dl"),
    "GRU":                 (gru_forecast,           DL_TIMEOUT,   "dl"),
    "BiLSTM":              (bilstm_forecast,        DL_TIMEOUT,   "dl"),
    "CNN1D":               (cnn_forecast,           DL_TIMEOUT,   "dl"),
    "TCN":                 (tcn_forecast,           DL_TIMEOUT,   "dl"),
    "Transformer":         (transformer_forecast,   DL_TIMEOUT,   "dl"),
    # Intermittent demand
    "CrostonSBA":          (croston_forecast,       STAT_TIMEOUT, "statistical"),
}


# ── Internal runner ───────────────────────────────────────────────────────────

def _run_single_model(
    name: str,
    fn: callable,
    timeout: int,
    series: pd.Series,
    forecast_horizon: int,
) -> dict:
    start = time.perf_counter()
    logger.info(f"[{name}] Starting | obs={len(series)} | horizon={forecast_horizon}")
    try:
        result  = fn(series, forecast_horizon)
        elapsed = round(time.perf_counter() - start, 2)
        logger.info(f"[{name}] ✓ Done {elapsed}s — RMSE={result.get('rmse', '?')}")
        # Always ensure forecast is a plain Python list to avoid numpy array truth-value errors
        if "forecast" in result and result["forecast"] is not None:
            result["forecast"] = [float(v) for v in result["forecast"]]

        # Real forecast intervals — not a fabricated flat percentage. If the
        # model already provides its own (e.g. RandomForest's tree-variance
        # based forecast_lower/upper), that's kept as-is since it's more
        # informative than this generic fallback. Otherwise, derive a normal-
        # approximation interval from THIS model's own historical CV RMSE —
        # honest in that different models get correctly different (wider for
        # less accurate models) bands, though still a simplification (a true
        # per-step conformal/quantile interval would be more rigorous).
        if "forecast_lower" not in result and "forecast_upper" not in result and result.get("forecast"):
            rmse = result.get("rmse")
            if rmse is not None and np.isfinite(rmse):
                z = 1.96
                fc = np.array(result["forecast"])
                # Widen with horizon step (naive random-walk-style error
                # compounding) rather than a flat band at every step.
                steps = np.sqrt(np.arange(1, len(fc) + 1))
                lower = fc - z * rmse * steps
                upper = fc + z * rmse * steps
                if float(series.min()) >= 0:
                    lower = np.clip(lower, 0, None)
                result["forecast_lower"] = [round(float(v), 4) for v in lower]
                result["forecast_upper"] = [round(float(v), 4) for v in upper]
                result["interval_method"] = "normal_approximation_rmse"
            else:
                result["forecast_lower"] = None
                result["forecast_upper"] = None
        else:
            result.setdefault("interval_method", "tree_variance")

        return {
            **result,
            "model_name":  name,
            "status":      "success",
            "elapsed_sec": elapsed,
            "error":       None,
        }
    except Exception as exc:
        elapsed = round(time.perf_counter() - start, 2)
        logger.error(f"[{name}] ✗ Failed {elapsed}s — {exc}\n{traceback.format_exc()}")
        return {
            "model_name":  name,
            "status":      "failed",
            "elapsed_sec": elapsed,
            "rmse":        float("inf"),
            "forecast":    [],
            "error":       str(exc),
        }


# ── Public API ────────────────────────────────────────────────────────────────

def run_all_models(
    series: pd.Series,
    forecast_horizon: int,
    include_dl: bool = True,
    categories: list[str] | None = None,
    models: dict | None = None,
) -> tuple[list[dict], dict]:
    """
    Run all registered models IN PARALLEL.

    Args:
        series           : Clean pd.Series with DatetimeIndex.
        forecast_horizon : Steps to forecast.
        include_dl       : Set False to skip Deep Learning models (faster).
                           Also auto-forced False if the series has fewer
                           rows than its frequency tier's threshold in
                           MIN_ROWS_FOR_DL_BY_TIER, regardless of this flag.
        categories       : Optional filter list, e.g. ["statistical", "ml"].
                           None = run all categories.
        models           : Override MODEL_REGISTRY (for testing).

    Returns:
        (results, run_info) where:
          results  : list of result dicts sorted by adjusted_score ascending.
          run_info : {"dl_skipped": bool, "dl_skip_reason": str | None,
                      "min_rows_for_dl": int, "row_count": int,
                      "frequency_tier": str}
    """
    if not isinstance(series, pd.Series):
        raise TypeError("series must be a pd.Series.")
    if len(series) < 10:
        raise ValueError(f"Need ≥ 10 observations, got {len(series)}.")
    if forecast_horizon < 1:
        raise ValueError(f"forecast_horizon must be ≥ 1, got {forecast_horizon}.")

    series = series.dropna().sort_index()
    row_count = len(series)
    frequency_tier = _infer_frequency_tier(series)
    min_rows_for_dl = MIN_ROWS_FOR_DL_BY_TIER[frequency_tier]

    # Hard row-count constraint: DL models are never attempted below this
    # threshold, regardless of what the caller requested — see
    # MIN_ROWS_FOR_DL_BY_TIER's docstring above for why.
    dl_skipped = False
    dl_skip_reason = None
    if include_dl and row_count < min_rows_for_dl:
        include_dl = False
        dl_skipped = True
        dl_skip_reason = (
            f"Deep learning models require at least {min_rows_for_dl} observations "
            f"of {frequency_tier} data for reliable results; this dataset has only {row_count}."
        )
        logger.info(f"[DL GATE] {dl_skip_reason} Skipping DL models for this run.")

    run_info = {
        "dl_skipped": dl_skipped,
        "dl_skip_reason": dl_skip_reason,
        "min_rows_for_dl": min_rows_for_dl,
        "row_count": row_count,
        "frequency_tier": frequency_tier,
    }

    registry = models or MODEL_REGISTRY

    # Filter by category
    if not include_dl:
        registry = {k: v for k, v in registry.items() if v[2] != "dl"}
    if categories:
        registry = {k: v for k, v in registry.items() if v[2] in categories}

    logger.info(
        f"Launching {len(registry)} models in parallel | "
        f"obs={len(series)} | horizon={forecast_horizon} | workers={MAX_WORKERS}"
    )

    results: list[dict]  = []
    futures_map: dict    = {}
    timeout_map: dict    = {}

    # NOTE: deliberately NOT using `with ThreadPoolExecutor(...) as executor:`.
    # That context manager calls shutdown(wait=True) on exit, which blocks
    # until EVERY submitted thread finishes — including ones we're about to
    # mark "timeout" below. A single stuck/slow model (e.g. a statistical
    # model fitting badly on irregular data) would silently hang this entire
    # HTTP request for as long as that thread takes to finish naturally
    # (confirmed in practice: threads marked "timeout" after 210s were still
    # running — and blocking their request — over an hour later). Using
    # shutdown(wait=False) below lets this function return promptly at
    # global_timeout regardless of what those threads are still doing.
    executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)
    try:
        for name, (fn, timeout, _category) in registry.items():
            future = executor.submit(_run_single_model, name, fn, timeout, series, forecast_horizon)
            futures_map[future] = name
            timeout_map[future] = timeout

        # Use the maximum timeout across all submitted models
        global_timeout = max(timeout_map.values()) + 30

        from concurrent.futures import wait
        done, not_done = wait(futures_map.keys(), timeout=global_timeout)

        for future in done:
            name    = futures_map[future]
            timeout = timeout_map[future]
            try:
                result = future.result()
                results.append(result)
            except Exception as exc:
                results.append({
                    "model_name":  name,
                    "status":      "failed",
                    "rmse":        float("inf"),
                    "forecast":    [],
                    "elapsed_sec": 0,
                    "error":       str(exc),
                })

        for future in not_done:
            name    = futures_map[future]
            timeout = timeout_map[future]
            logger.error(f"[{name}] Timed out after {global_timeout}s — abandoning it "
                         f"(it may keep running in the background; its result is discarded)")
            results.append({
                "model_name":  name,
                "status":      "timeout",
                "rmse":        float("inf"),
                "forecast":    [],
                "elapsed_sec": global_timeout,
                "error":       f"Timed out after {global_timeout}s",
            })
            future.cancel()  # only has any effect if the task hadn't started yet
    finally:
        # cancel_futures=True drops any still-QUEUED (not yet started) tasks
        # immediately; wait=False means we don't block this request on
        # whatever's still actually running.
        executor.shutdown(wait=False, cancel_futures=True)

    # ── Calculate Multi-Criteria Weighted Score ──
    from services.metrics import get_stability_penalty

    def _robust_scale(values):
        """
        Normalization denominator using a Tukey outlier fence (Q3 + 1.5*IQR —
        the same robust-statistics convention already used for outlier
        capping in preprocessor.py) instead of the raw max.

        Why: a single catastrophically bad model (e.g. SARIMA/SARIMAX fitting
        terribly on an irregular dataset, RMSE in the hundreds of millions
        vs. everyone else in the hundreds of thousands) would otherwise
        become the denominator for ALL models. That squashes every other
        model's normalized RMSE/MAE/MAPE down to near-zero noise, so the
        80%-weighted accuracy terms stop differentiating between the good
        models at all and the ranking gets decided by the remaining 20%
        (speed/R²) almost by accident — confirmed: this is exactly how a
        model with a *worse* raw RMSE than another still won "best model."
        Outlier models still land above this fence (ratio > 1, correctly
        penalized) — they just can't set the scale for everyone else.
        """
        finite = [v for v in values if np.isfinite(v)]
        if not finite:
            return 1e-6
        arr = np.array(finite, dtype=float)
        q1, q3 = np.percentile(arr, [25, 75])
        fence = q3 + 1.5 * (q3 - q1)
        return float(fence) if fence > 1e-6 else float(arr.max()) or 1e-6

    try:
        last_val = float(series.iloc[-1])
        hist_std = max(float(series.std()), 1e-6)

        valid_results = [r for r in results if r["status"] == "success" and r.get("forecast") is not None and len(r.get("forecast", [])) > 0]

        if valid_results:
            max_rmse  = _robust_scale([r.get("rmse", 0) for r in valid_results])
            max_mae   = _robust_scale([r.get("mae", 0) for r in valid_results])
            max_mape  = _robust_scale([r.get("mape", 0) for r in valid_results])
            max_train = _robust_scale([r.get("train_time_sec", 0) for r in valid_results])
            max_pred  = _robust_scale([r.get("pred_time_sec", 0) for r in valid_results])

            # First pass for stability to get the normalization scale
            stabs = []
            for r in valid_results:
                try:
                    fc = list(r["forecast"])  # ensure plain list, never numpy array
                    stab = get_stability_penalty(last_val, hist_std, fc)
                except Exception:
                    stab = float('inf')
                r["_raw_stab"] = stab
                stabs.append(stab)
            max_stab = _robust_scale(stabs)

            for r in results:
                if r["status"] == "success" and r.get("forecast") is not None and len(r.get("forecast", [])) > 0:
                    n_rmse = r.get("rmse", 0) / max_rmse
                    n_mae  = r.get("mae", 0) / max_mae
                    n_mape = r.get("mape", 0) / max_mape
                    
                    t_train = r.get("train_time_sec", r.get("elapsed_sec", 0))
                    t_pred  = r.get("pred_time_sec", 0)
                    speed   = (t_train / max_train + t_pred / max_pred) / 2
                    
                    stability = r["_raw_stab"] / max_stab
                    r2 = r.get("r2", 0)
                    
                    final_score = (
                        0.40 * n_rmse +
                        0.20 * n_mae +
                        0.20 * n_mape +
                        0.10 * stability +
                        0.10 * speed -
                        0.10 * r2
                    )
                    
                    r["adjusted_score"] = round(float(final_score), 4)
                    r["stability_penalty"] = round(float(r["_raw_stab"]), 4)
                    del r["_raw_stab"]
                else:
                    r["adjusted_score"] = float("inf")
        else:
            for r in results:
                r["adjusted_score"] = float("inf")
    except Exception as e:
        logger.error(f"Error calculating weighted scores: {e}")
        for r in results:
            r["adjusted_score"] = float("inf")

    # Sort by Adjusted Score (failed models sink to bottom)
    results.sort(key=lambda r: r.get("adjusted_score", float("inf")) if np.isfinite(r.get("adjusted_score", float("inf"))) else float("inf"))

    # Summary
    logger.info("─" * 80)
    logger.info(f"{'Rank':<5} {'Model':<18} {'Score':<10} {'RMSE':<10} {'MAE':<10} {'R2':<8} {'Time'}")
    logger.info("─" * 80)
    for i, r in enumerate(results, 1):
        score = f"{r.get('adjusted_score', float('inf')):.4f}" if np.isfinite(r.get("adjusted_score", float("inf"))) else "FAILED"
        rmse = f"{r.get('rmse', float('inf')):.2f}" if np.isfinite(r.get("rmse", float("inf"))) else "-"
        mae  = f"{r.get('mae', float('inf')):.2f}" if np.isfinite(r.get("mae", float("inf"))) else "-"
        r2   = f"{r.get('r2', -99):.2f}" if "r2" in r else "-"
        logger.info(f"{i:<5} {r['model_name']:<18} {score:<10} {rmse:<10} {mae:<10} {r2:<8} {r['elapsed_sec']}s")
    logger.info("─" * 80)

    return results, run_info