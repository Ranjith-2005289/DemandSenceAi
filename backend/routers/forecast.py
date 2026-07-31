from fastapi import APIRouter, HTTPException
from pathlib import Path
import pandas as pd
from pydantic import BaseModel
import logging
import pickle

from services.preprocessor import preprocess
from services.best_model import select_best
from services.model_runner import run_all_models, MODEL_REGISTRY
from services.product_analytics import compute_group_summary
from services import session_store

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/forecast", tags=["forecast"])

# Storage folders
UPLOAD_DIR = Path("uploads")
PROCESSED_DIR = Path("processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# In-memory cache of the most recent forecast run, keyed by session_id so
# concurrent users/tabs don't silently overwrite each other's results.
# "default" is used when no session_id is provided, preserving the old
# single-global behavior for any caller that doesn't pass one. Also mirrored
# to disk (services/session_store.py) so a backend restart doesn't lose it —
# the in-memory dict stays the fast path, disk is purely a restart backstop.
_latest_forecast_by_session: dict[str, dict] = {}


class ForecastRequest(BaseModel):
    """Request body for POST /forecast. Provide exactly one of target_col
    (single target, the original behavior) or target_cols (forecast several
    numeric columns in one request — see the "targets" key in the response)."""
    filename: str
    date_col: str
    target_col: str | None = None
    target_cols: list[str] | None = None    # Multi-target: forecast each of these independently
    forecast_horizon: int = 30
    skip_dl: bool = False  # Set True to skip Deep Learning models (faster results)
    group_col: str | None = None    # Optional: forecast one group instead of the whole dataset
    group_value: str | None = None  # Required if group_col is set
    country: str | None = None      # Optional ISO country code (e.g. "US") for Prophet holiday effects
    session_id: str | None = None   # Optional client-generated session id, for cache isolation
    feature_cols: list[str] | None = None  # Optional extra columns (price/promo) for
                                            # XGBoost/RandomForest/SARIMAX — see upload's
                                            # detected.exog_columns for what's available


class ProductSummaryRequest(BaseModel):
    """Request body for POST /forecast/product-summary."""
    filename: str
    date_col: str
    target_col: str
    group_col: str
    top_n: int = 50


def _forecast_single_target(
    df: pd.DataFrame,
    processed_name_hint: str,
    date_col: str,
    target_col: str,
    forecast_horizon: int,
    skip_dl: bool,
    country: str | None,
    feature_cols: list[str] | None,
    group_col: str | None,
    group_value: str | None,
) -> dict:
    """
    Runs the full preprocess -> all-models -> best-model pipeline for ONE
    target column and returns the response payload. Shared by both the
    single-target path (POST /forecast with target_col) and the multi-target
    path (target_cols) — for multi-target, this is called once per column,
    so one target's failure doesn't take down the others (see run_forecast).
    """
    # ── Preprocess ─────────────────────────────────────────────────────────
    try:
        logger.info(f"[FORECAST] Preprocessing with date_col={date_col}, target_col={target_col}")
        ts, preprocess_report = preprocess(
            df, date_col=date_col, target_col=target_col, feature_cols=feature_cols,
        )
        logger.info(f"[FORECAST] Preprocessed series: shape={ts.shape}, index type={type(ts.index)}")
        logger.info(f"[FORECAST] Preprocess report steps: {preprocess_report.get('steps')}")
    except Exception as e:
        logger.error(f"[FORECAST] Preprocessing failed: {e}")
        raise HTTPException(status_code=400, detail=f"Preprocessing failed: {e}")

    # ── Save processed data ────────────────────────────────────────────────
    try:
        processed_filename = processed_name_hint.replace("_preprocessed_walmart_dataset.csv", "_processed.pkl")
        processed_path = PROCESSED_DIR / processed_filename
        with open(processed_path, "wb") as f:
            pickle.dump(ts, f)
        csv_processed_path = PROCESSED_DIR / processed_filename.replace(".pkl", ".csv")
        ts.to_csv(csv_processed_path)
        logger.info(f"[FORECAST] Saved processed data to {processed_filename}")
    except Exception as e:
        logger.error(f"[FORECAST] Failed to save processed data: {e}")
        # Don't fail the forecast if saving fails, just warn

    if not isinstance(ts.index, pd.DatetimeIndex):
        raise HTTPException(
            status_code=400,
            detail="After preprocessing, the index should be DatetimeIndex."
        )

    series = ts

    # ── Run all models ─────────────────────────────────────────────────────
    # Prophet is the only model that accepts a country code (for holiday
    # effects) — every other model in the registry is called uniformly as
    # fn(series, horizon) by model_runner, so injecting "country" is done by
    # swapping in a one-off registry override for this request rather than
    # threading a new parameter through the shared orchestration code.
    models_override = None
    if country:
        fn, timeout, category = MODEL_REGISTRY["Prophet"]
        models_override = dict(MODEL_REGISTRY)
        models_override["Prophet"] = (
            lambda s, h, _c=country: fn(s, h, country=_c), timeout, category
        )

    # Same pattern for real exogenous features (price/promo): only
    # XGBoost, RandomForest, and SARIMAX know what to do with them.
    exog_df = preprocess_report.get("exog_df")
    exog_meta = preprocess_report.get("exog_meta")
    if exog_df is not None and not exog_df.empty:
        models_override = models_override or dict(MODEL_REGISTRY)
        for name in ("XGBoost", "RandomForest", "SARIMAX"):
            fn, timeout, category = MODEL_REGISTRY[name]
            models_override[name] = (
                lambda s, h, _fn=fn, _e=exog_df, _m=exog_meta: _fn(s, h, exog_df=_e, exog_meta=_m),
                timeout, category,
            )
        logger.info(f"[FORECAST] Using exogenous features {list(exog_df.columns)} for XGBoost/RandomForest/SARIMAX")

    try:
        logger.info(f"[FORECAST] Running models with series length={len(series)}, horizon={forecast_horizon}, skip_dl={skip_dl}, country={country}")
        all_models, model_run_info = run_all_models(
            series, forecast_horizon=forecast_horizon,
            include_dl=not skip_dl, models=models_override,
        )
        logger.info(f"[FORECAST] Models completed: {len(all_models)} models, results={[m.get('model_name') for m in all_models]}")
        if model_run_info.get("dl_skipped"):
            logger.info(f"[FORECAST] {model_run_info['dl_skip_reason']}")
    except Exception as e:
        logger.error(f"[FORECAST] Model execution failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Model execution failed: {e}")

    # ── Select best model ───────────────────────────────────────────────────
    try:
        logger.info(f"[FORECAST] Selecting best model from {len(all_models)} results")
        best_result = select_best(all_models, series=series)
        logger.info(f"[FORECAST] Best model selected: {best_result.get('best_model_name')}")
    except Exception as e:
        logger.error(f"[FORECAST] Best model selection failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Best model selection failed: {e}")

    # ── Generate future dates ───────────────────────────────────────────────
    last_date = ts.index[-1]
    freq = pd.infer_freq(ts.index)
    if freq is None:
        gaps = ts.index.to_series().diff().dropna()
        median_gap = gaps.median()
        days = median_gap.days if hasattr(median_gap, 'days') else median_gap / pd.Timedelta('1D')
        freq = f"{int(max(1, days))}D"

    future_dates = pd.date_range(
        start=last_date + pd.tseries.frequencies.to_offset(freq),
        periods=forecast_horizon,
        freq=freq
    )
    forecast_dates = future_dates.strftime("%Y-%m-%d").tolist()

    # ── Format response ─────────────────────────────────────────────────────
    # Real historical series for the frontend's chart — capped to the most
    # recent 365 points (a full year) for legibility/payload size; showing
    # every point of a 300K-row series on a line chart isn't useful anyway.
    return {
        "status": "success",
        "target_col": target_col,
        "all_models": all_models,
        "best_model": best_result,
        "forecast_dates": forecast_dates,
        "leaderboard": best_result.get("all_models", []),
        "group": (
            {"group_col": group_col, "group_value": group_value}
            if group_col else None
        ),
        "model_run_info": model_run_info,
        "historical": {
            "dates": [str(d.date()) for d in series.index[-365:]],
            "values": [float(v) for v in series.values[-365:]],
        },
        "exog_features_used": list(exog_df.columns) if exog_df is not None and not exog_df.empty else [],
    }


@router.post("")
def run_forecast(request: ForecastRequest) -> dict:
    """
    Load a CSV file, preprocess it, run all models in parallel, and return
    forecasts — for one target column (request.target_col), or independently
    for several at once (request.target_cols; exactly one of the two must
    be set).

    Single-target response (unchanged from before multi-target support):
        {
            "status": "success",
            "all_models": [...],    # Full result list for all models
            "best_model": {...},    # Winner with lowest RMSE
            "forecast_dates": [...], # List of future ISO dates matching forecast length
            "leaderboard": [...],   # Compact leaderboard (rank, model_name, rmse, status, time)
        }

    Multi-target response:
        {
            "status": "success",
            "targets": {
                "<target_col>": { ...same shape as above, or... },
                "<other_target_col>": { "status": "error", "error": "..." },
            }
        }
    """
    logger.info(f"[FORECAST] Request received: filename={request.filename}, date_col={request.date_col}, target_col={request.target_col}, target_cols={request.target_cols}, horizon={request.forecast_horizon}")

    # Validate forecast_horizon
    if not (1 <= request.forecast_horizon <= 365):
        raise HTTPException(
            status_code=400,
            detail="forecast_horizon must be between 1 and 365."
        )

    # group_col and group_value must be provided together
    if bool(request.group_col) != bool(request.group_value):
        raise HTTPException(
            status_code=400,
            detail="group_col and group_value must both be provided together, or both omitted.",
        )

    # Exactly one of target_col / target_cols
    if bool(request.target_col) == bool(request.target_cols):
        raise HTTPException(
            status_code=400,
            detail="Provide exactly one of target_col (single target) or target_cols (multiple).",
        )

    # ── 1. Load CSV ───────────────────────────────────────────────────────────
    csv_path = UPLOAD_DIR / request.filename
    logger.info(f"[FORECAST] Looking for file at: {csv_path}")

    if not csv_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"File '{request.filename}' not found. Please upload a CSV first."
        )

    try:
        df = pd.read_csv(csv_path)
        logger.info(f"[FORECAST] Loaded CSV: shape={df.shape}, columns={list(df.columns)}")
    except Exception as e:
        logger.error(f"[FORECAST] Failed to read CSV: {e}")
        raise HTTPException(status_code=400, detail=f"Failed to read CSV: {e}")

    # ── 1b. Filter to a single group, if requested ────────────────────────────
    if request.group_col:
        if request.group_col not in df.columns:
            raise HTTPException(
                status_code=400,
                detail=f"Column '{request.group_col}' not found in dataset.",
            )
        df = df[df[request.group_col].astype(str) == request.group_value]
        if df.empty:
            raise HTTPException(
                status_code=404,
                detail=f"No rows found where '{request.group_col}' == '{request.group_value}'.",
            )
        logger.info(f"[FORECAST] Filtered to group {request.group_col}={request.group_value}: {df.shape[0]} rows")

    key = request.session_id or "default"

    # ── 2. Single-target path (original behavior, unchanged response shape) ──
    if request.target_col:
        payload = _forecast_single_target(
            df, request.filename, request.date_col, request.target_col,
            request.forecast_horizon, request.skip_dl, request.country,
            request.feature_cols, request.group_col, request.group_value,
        )
        _latest_forecast_by_session[key] = payload
        try:
            session_store.save("forecast", key, payload)
        except Exception as e:
            # Disk persistence is a restart backstop, not a hard dependency —
            # the in-memory cache above already makes this request succeed either way.
            logger.warning(f"[FORECAST] Failed to persist session {key!r} to disk: {e}")
        return payload

    # ── 2b. Multi-target path: run each target independently, one failing
    #        doesn't take down the others ────────────────────────────────────
    targets: dict[str, dict] = {}
    for target_col in request.target_cols:
        if target_col not in df.columns:
            targets[target_col] = {"status": "error", "error": f"Column '{target_col}' not found in dataset."}
            continue
        try:
            name_hint = f"{Path(request.filename).stem}_{target_col}{Path(request.filename).suffix}"
            targets[target_col] = _forecast_single_target(
                df, name_hint, request.date_col, target_col,
                request.forecast_horizon, request.skip_dl, request.country,
                request.feature_cols, request.group_col, request.group_value,
            )
        except HTTPException as e:
            targets[target_col] = {"status": "error", "error": e.detail}

    multi_payload = {"status": "success", "targets": targets}
    _latest_forecast_by_session[key] = multi_payload
    try:
        session_store.save("forecast", key, multi_payload)
    except Exception as e:
        logger.warning(f"[FORECAST] Failed to persist session {key!r} to disk: {e}")
    return multi_payload


@router.get("/latest")
def latest_forecast(session_id: str | None = None) -> dict:
    """Return the most recent forecast result — from the in-memory cache, falling
    back to disk (survives a backend restart) if this process hasn't seen it."""
    key = session_id or "default"
    cached = _latest_forecast_by_session.get(key)
    if cached is None:
        cached = session_store.load("forecast", key)
        if cached is not None:
            _latest_forecast_by_session[key] = cached
    if cached is None:
        raise HTTPException(
            status_code=404,
            detail="No forecast has been run yet. Call POST /forecast first.",
        )
    return cached


@router.post("/product-summary")
def product_summary(request: ProductSummaryRequest) -> dict:
    """
    Rank groups (e.g. products/stores/categories) by total sales and label
    each with a trend. No forecasting models involved — fast regardless of
    how many groups the dataset has.
    """
    csv_path = UPLOAD_DIR / request.filename
    if not csv_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"File '{request.filename}' not found. Please upload a CSV first.",
        )

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read CSV: {e}")

    try:
        return compute_group_summary(
            df,
            date_col=request.date_col,
            target_col=request.target_col,
            group_col=request.group_col,
            top_n=request.top_n,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"[PRODUCT-SUMMARY] Failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Product summary failed: {e}")
