import numpy as np
from math import sqrt
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score

MIN_CV_SPLITS = 2
MAX_CV_SPLITS = 5


def get_cv_splits(n_obs: int) -> int:
    """
    Adaptive walk-forward fold count for TimeSeriesSplit(n_splits=...).

    Every model in this codebase previously used a dead conditional
    (`2 if n_obs >= 20 else 2` — both branches always evaluate to 2), so
    "best model" selection was always based on a fixed 2-fold walk-forward
    evaluation regardless of how much data was available. More folds when
    there's enough data reduces the risk of a model looking best purely by
    luck on a single train/test split; capped so evaluation cost (each
    fold re-fits the model) stays bounded for expensive models.
    """
    return min(MAX_CV_SPLITS, max(MIN_CV_SPLITS, n_obs // 100))

def calculate_metrics(y_true, y_pred):
    """
    Calculate standard forecasting metrics.
    Returns: { rmse, mae, mape, r2 }
    """
    if len(y_true) == 0 or len(y_pred) == 0:
        return {"rmse": float('inf'), "mae": float('inf'), "mape": float('inf'), "r2": float('-inf')}
        
    try:
        rmse = sqrt(mean_squared_error(y_true, y_pred))
    except Exception:
        rmse = float('inf')
        
    try:
        mae = mean_absolute_error(y_true, y_pred)
    except Exception:
        mae = float('inf')
    
    try:
        # Handle zero division in MAPE
        mape = mean_absolute_percentage_error(y_true, y_pred)
    except Exception:
        mape = float('inf')
        
    try:
        r2 = r2_score(y_true, y_pred)
        if not np.isfinite(r2):
            r2 = float('-inf')
    except Exception:
        r2 = float('-inf')
    
    return {
        "rmse": float(rmse),
        "mae": float(mae),
        "mape": float(mape),
        "r2": float(r2)
    }

def get_stability_penalty(last_val, hist_std, forecast):
    """
    Calculate the stability penalty based on discontinuity and volatility.
    """
    if forecast is None or len(forecast) == 0:
        return float('inf')
        
    fc = np.array(forecast)
    
    discontinuity_penalty = abs(fc[0] - last_val) / hist_std
    
    fc_std = np.std(fc)
    volatility_penalty = abs(fc_std - hist_std) / hist_std
    
    return float(discontinuity_penalty + volatility_penalty)
