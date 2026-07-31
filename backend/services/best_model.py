"""
best_model.py — Best model selector
Picks the model with the lowest RMSE from run_all_models() results,
with fallback logic if the winner's forecast is unusable.
"""

import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ── Validation helpers ────────────────────────────────────────────────────────

def _forecast_is_usable(forecast: list, series: pd.Series | None = None) -> bool:
    """
    Check that a forecast list is non-empty, finite, and not obviously exploded.
    Optionally checks that values aren't wildly out of range relative to history.
    """
    if not forecast or len(forecast) == 0:
        return False

    arr = np.array(forecast, dtype=float)

    # Must be all finite
    if not np.all(np.isfinite(arr)):
        return False

    # Guard against models that produce huge / negative spikes
    if series is not None and len(series) > 0:
        series_mean = float(series.mean())
        series_std  = float(series.std())
        upper_bound = series_mean + 10 * series_std
        lower_bound = series_mean - 10 * series_std
        if np.any(arr > upper_bound) or np.any(arr < lower_bound):
            return False

    return True


def _score_is_valid(score) -> bool:
    """Return True if score is a real finite number."""
    try:
        v = float(score)
        return np.isfinite(v)
    except (TypeError, ValueError):
        return False


# ── Public API ────────────────────────────────────────────────────────────────

def select_best(
    results: list[dict],
    series: pd.Series | None = None,
) -> dict:
    """
    Pick the best model from run_all_models() output.

    Strategy:
      1. Filter to models that succeeded and have usable forecasts.
      2. Among those, pick the one with the lowest RMSE.
      3. If only failed models exist, return a graceful error dict.

    Args:
        results : Output of run_all_models() — list of dicts sorted by RMSE.
        series  : Optional original series used to sanity-check forecast range.

    Returns:
        {
            "best_model_name" : str,
            "rmse"            : float,
            "mae"             : float,
            "mape"            : float,
            "r2"              : float,
            "score"           : float,
            "forecast"        : list[float],
            "rank"            : int,             position in the sorted results (1 = best)
            "all_models"      : list[dict],      full leaderboard
            "winner_details"  : dict,            full result dict of the winner
        }
    """
    if not results:
        raise ValueError("results list is empty — no models were run.")

    # ── Build leaderboard ─────────────────────────────────────────────────────
    leaderboard = []
    for rank, r in enumerate(results, start=1):
        leaderboard.append({
            "rank":        rank,
            "model_name":  r.get("model_name", "Unknown"),
            "score":       r.get("adjusted_score", float("inf")),
            "rmse":        r.get("rmse", float("inf")),
            "mae":         r.get("mae", float("inf")),
            "r2":          r.get("r2", -99.0),
            "status":      r.get("status", "unknown"),
            "elapsed_sec": r.get("elapsed_sec", 0),
            "error":       r.get("error"),
        })

    # ── Find best usable model (by adjusted_score, fallback to RMSE) ───────────
    winner = None
    winner_rank = None

    # First try: pick model with valid adjusted_score
    for rank, r in enumerate(results, start=1):
        score    = r.get("adjusted_score", float("inf"))
        forecast = r.get("forecast", [])
        status   = r.get("status", "failed")

        if status != "success":
            continue
        if not _score_is_valid(score):
            continue
        if not _forecast_is_usable(forecast, series=series):
            logger.warning(
                f"[{r.get('model_name')}] Skipped — forecast failed usability check."
            )
            continue

        winner      = r
        winner_rank = rank
        break

    # Fallback: if no valid adjusted_score, pick by lowest RMSE
    if winner is None:
        logger.warning("No valid adjusted_score found — falling back to RMSE-based selection.")
        # Sort successful models by RMSE
        successful = [
            (rank, r) for rank, r in enumerate(results, start=1)
            if r.get("status") == "success"
            and np.isfinite(r.get("rmse", float("inf")))
            and _forecast_is_usable(r.get("forecast", []), series=series)
        ]
        if successful:
            successful.sort(key=lambda x: x[1].get("rmse", float("inf")))
            winner_rank, winner = successful[0]
            # Give it a synthetic adjusted_score based on RMSE rank
            winner["adjusted_score"] = winner.get("rmse", 0)

    # ── Fallback: all models failed ───────────────────────────────────────────
    if winner is None:
        failed_names = [r.get("model_name", "?") for r in results]
        error_msgs   = [r.get("error", "unknown error") for r in results if r.get("error")]
        logger.error(f"All models failed: {failed_names}")
        return {
            "best_model_name": None,
            "score":           None,
            "rmse":            None,
            "mae":             None,
            "mape":            None,
            "r2":              None,
            "forecast":        [],
            "rank":            None,
            "all_models":      leaderboard,
            "winner_details":  None,
            "error": (
                f"All {len(results)} models failed to produce a valid forecast. "
                f"Errors: {'; '.join(error_msgs[:2])}"
            ),
        }

    # ── Build winner details (clean copy, no internal keys) ───────────────────
    winner_details = {
        k: v for k, v in winner.items()
        if k not in ("status", "error")
    }

    logger.info(
        f"Best model: {winner['model_name']} "
        f"(Score={winner.get('adjusted_score'):.4f}, RMSE={winner.get('rmse'):.4f}, rank={winner_rank}/{len(results)})"
    )

    return {
        "best_model_name": winner["model_name"],
        "score":           round(float(winner.get("adjusted_score", 0)), 4),
        "rmse":            round(float(winner.get("rmse", 0)), 4),
        "mae":             round(float(winner.get("mae", 0)), 4),
        "mape":            round(float(winner.get("mape", 0)), 4),
        "r2":              round(float(winner.get("r2", 0)), 4),
        "forecast":        winner["forecast"],
        "forecast_lower":  winner.get("forecast_lower"),
        "forecast_upper":  winner.get("forecast_upper"),
        "interval_method": winner.get("interval_method"),
        "rank":            winner_rank,
        "all_models":      leaderboard,
        "winner_details":  winner_details,
        "error":           None,
    }


# ── Convenience: leaderboard summary string ───────────────────────────────────

def format_leaderboard(results: list[dict]) -> str:
    """
    Return a formatted text leaderboard for logging / API response.

    Example:
        Rank  Model            RMSE        Status     Time
        1     XGBoost          142.3800    success    8.2s
        2     RandomForest     158.1200    success    9.1s
        3     Prophet          201.4500    success    5.3s
        4     ARIMA            312.8800    success    12.7s
    """
    header = f"{'Rank':<6}{'Model':<17}{'Score':<10}{'RMSE':<10}{'MAE':<10}{'R2':<8}{'Status':<12}{'Time'}"
    rows   = [header, "-" * 78]

    for rank, r in enumerate(results, start=1):
        score_str = f"{r.get('adjusted_score', float('inf')):.4f}" if np.isfinite(r.get("adjusted_score", float("inf"))) else "FAILED"
        rmse_str  = f"{r.get('rmse', float('inf')):.2f}" if np.isfinite(r.get("rmse", float("inf"))) else "-"
        mae_str   = f"{r.get('mae', float('inf')):.2f}" if np.isfinite(r.get("mae", float("inf"))) else "-"
        r2_str    = f"{r.get('r2', -99):.2f}" if "r2" in r else "-"
        
        rows.append(
            f"{rank:<6}{r.get('model_name','?'):<17}"
            f"{score_str:<10}{rmse_str:<10}{mae_str:<10}{r2_str:<8}{r.get('status','?'):<12}"
            f"{r.get('elapsed_sec', 0):.1f}s"
        )

    return "\n".join(rows)