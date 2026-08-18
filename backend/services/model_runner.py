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

# ── Model ranking policy ──────────────────────────────────────────────────────
# Models are ranked by cross-validated RMSE, optionally restricted to those that
# fit a training-time budget.
#
# This replaced a six-term weighted score
# (0.40*RMSE + 0.20*MAE + 0.20*MAPE + 0.10*stability + 0.10*speed - 0.10*R2),
# which was benchmarked against held-out data on 480 M5 series across four
# aggregation levels and lost to cross-validated RMSE alone at every one of them
# (p=2.1e-7 at store x department, p=3.6e-6 at item level). The defect was
# structural, not a bad choice of weights. Every term was divided by its own
# Tukey fence, which bounds the DENOMINATOR but not the RATIO: training time
# spans six orders of magnitude across this registry (MovingAverage ~5e-5s vs
# SARIMA ~118s) while RMSE spans a factor of roughly 1.2, so normalized speed
# reached 43 while normalized RMSE stayed inside 0.5-1.0. At a nominal weight of
# 0.10 the speed term therefore decided 63-88% of selections, and the models it
# pushed out (SARIMA/SARIMAX) were the ones that generalized best out-of-sample.
# R2 compounded it, being the only term never normalized at all.
#
# SELECTION_TIME_BUDGET_SEC defaults to None — ranking by accuracy alone —
# because the useful budget is dataset-dependent and cannot be inferred at
# request time. Constraining to ~1.5s beat unconstrained selection by about 5%
# MASE on fine-grained series (item and item-store levels: 5/5 folds, 95% CI
# excluding zero, stable across seeds), but the same budget is WORSE than no
# budget on coarse aggregates, where nested selection instead chose a bound so
# loose it was equivalent to no constraint. Accuracy-only ranking is the setting
# that measured better than the old score at every level, so it is the default;
# set a budget explicitly when deployment latency actually matters.
SELECTION_TIME_BUDGET_SEC: float | None = None

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

    # ── Rank models (see SELECTION_TIME_BUDGET_SEC above) ──
    from services.metrics import get_stability_penalty

    def _robust_scale(values):
        """
        Normalization denominator using a Tukey outlier fence (Q3 + 1.5*IQR —
        the same robust-statistics convention already used for outlier
        capping in preprocessor.py) instead of the raw max.

        Why: a single catastrophically bad model (e.g. SARIMA/SARIMAX fitting
        terribly on an irregular dataset, RMSE in the hundreds of millions vs.
        everyone else in the hundreds of thousands) would otherwise become the
        denominator for ALL models, squashing every other model's normalized
        error into near-zero noise where they stop being distinguishable.
        Outlier models still land above this fence (ratio > 1, correctly
        penalized) — they just can't set the scale for everyone else.

        Note this bounds the DENOMINATOR, not the resulting ratio. That was
        survivable for RMSE, whose spread across the registry is small, but it
        is exactly what broke the old multi-criteria score once training time —
        spanning six orders of magnitude — was fed through the same treatment.
        """
        finite = [v for v in values if np.isfinite(v)]
        if not finite:
            return 1e-6
        arr = np.array(finite, dtype=float)
        q1, q3 = np.percentile(arr, [25, 75])
        fence = q3 + 1.5 * (q3 - q1)
        return float(fence) if fence > 1e-6 else float(arr.max()) or 1e-6

    def _is_scorable(r: dict) -> bool:
        return (r["status"] == "success"
                and r.get("forecast") is not None
                and len(r.get("forecast", [])) > 0)

    def _train_cost(r: dict) -> float:
        """Seconds spent fitting. Falls back to wall-clock when a model doesn't
        report train_time_sec separately."""
        return float(r.get("train_time_sec", r.get("elapsed_sec", 0.0)) or 0.0)

    try:
        last_val = float(series.iloc[-1])
        hist_std = max(float(series.std()), 1e-6)

        valid_results = [r for r in results if _is_scorable(r)]

        if valid_results:
            max_rmse = _robust_scale([r.get("rmse", 0) for r in valid_results])

            # Stability is still computed and reported — it's genuinely useful
            # diagnostic information — but it no longer influences the ranking.
            # Its discontinuity term |forecast[0] - last_actual| penalizes a
            # model for failing to anchor to the final observation, which
            # rewards persistence-like forecasts over statistical ones; on the
            # M5 benchmark it consistently pushed against SARIMA/SARIMAX, the
            # models that actually generalized best.
            for r in valid_results:
                try:
                    stab = get_stability_penalty(last_val, hist_std, list(r["forecast"]))
                except Exception:
                    stab = float("inf")
                r["stability_penalty"] = (
                    round(float(stab), 4) if np.isfinite(stab) else float("inf"))

            normalized = {id(r): float(r.get("rmse", 0)) / max_rmse for r in valid_results}
            finite_norms = [v for v in normalized.values() if np.isfinite(v)]

            # Over-budget models must rank below every affordable one. A fixed
            # constant can't guarantee that — a catastrophically bad model can
            # sit far above the Tukey fence (SARIMA at ~823M RMSE against a
            # field in the hundreds of thousands) — so the offset is derived
            # from the observed spread instead.
            over_budget_offset = (max(finite_norms) + 1.0) if finite_norms else 1.0

            affordable = (
                [r for r in valid_results if _train_cost(r) <= SELECTION_TIME_BUDGET_SEC]
                if SELECTION_TIME_BUDGET_SEC is not None else valid_results
            )
            budget_active = bool(SELECTION_TIME_BUDGET_SEC is not None and affordable)
            affordable_ids = {id(r) for r in affordable}

            if SELECTION_TIME_BUDGET_SEC is not None and not affordable:
                # Nothing fits. Mirrors the benchmarked policy exactly: fall
                # back to the cheapest model, since the caller asked for a
                # latency bound and this is the closest it can be honoured.
                # Unreachable with the current registry — MovingAverage and KNN
                # fit in microseconds — and `n_fallback` was 0 for every budget
                # at every aggregation level tested.
                cheapest = min(valid_results, key=_train_cost)
                logger.warning(
                    f"[SELECTION] No model trained within {SELECTION_TIME_BUDGET_SEC}s; "
                    f"falling back to the cheapest ({cheapest['model_name']}, "
                    f"{_train_cost(cheapest):.4f}s)."
                )
                affordable_ids = {id(cheapest)}
                budget_active = True

            for r in results:
                if not _is_scorable(r):
                    r["adjusted_score"] = float("inf")
                    continue
                score = normalized[id(r)]
                within = (not budget_active) or (id(r) in affordable_ids)
                if not within:
                    score += over_budget_offset
                r["within_time_budget"] = bool(within)
                r["adjusted_score"] = round(float(score), 6)

            if budget_active:
                logger.info(
                    f"[SELECTION] Time budget {SELECTION_TIME_BUDGET_SEC}s: "
                    f"{len(affordable_ids)}/{len(valid_results)} models eligible."
                )
        else:
            for r in results:
                r["adjusted_score"] = float("inf")
    except Exception as e:
        logger.error(f"Error ranking models: {e}", exc_info=True)
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