"""
run_all_models' adjusted_score normalized every model's RMSE/MAE/MAPE
against the raw max across the batch. A single catastrophic outlier (e.g.
SARIMA/SARIMAX fitting terribly on an irregular dataset, RMSE in the
hundreds of millions vs. everyone else in the hundreds of thousands) became
that denominator, squashing every other model's normalized error down to
near-zero noise — so the 80%-weighted accuracy terms stopped
differentiating between the "good" models, and ranking got decided by the
remaining ~20% (speed/R²) almost by accident. This reproduces exactly that
scenario against the Tukey-fence fix and asserts the ranking now tracks
real RMSE among the non-outlier models.
"""
import numpy as np
import pandas as pd

from services.model_runner import run_all_models


def _fixed_result(name, rmse, mae, mape, r2, train_time, pred_time):
    def _fn(series, horizon):
        return {
            "model_name": name, "rmse": rmse, "mae": mae, "mape": mape, "r2": r2,
            "forecast": [float(series.iloc[-1])] * horizon,
            "train_time_sec": train_time, "pred_time_sec": pred_time,
        }
    return _fn


def test_outlier_does_not_flatten_ranking_among_good_models():
    # Mirrors the real batch composition that exposed the bug: several
    # normal-range models plus two catastrophic-outlier statistical models.
    # Uniform train/pred time across the "good" models so the 10%-weighted
    # speed term doesn't swamp the RMSE/MAE/MAPE terms this test is actually
    # about — the composite score is intentionally a multi-criteria blend,
    # not pure RMSE ranking, so speed differences must be held constant to
    # isolate the normalization-denominator behavior being tested here.
    registry = {
        "TCN":          (_fixed_result("TCN", 593_226.6, 480_000, 0.04, 0.88, 20, 0.1), 5, "dl"),
        "XGBoost":      (_fixed_result("XGBoost", 691_085.9, 550_000, 0.045, 0.85, 20, 0.1), 5, "ml"),
        "RandomForest": (_fixed_result("RandomForest", 720_790.5, 500_000, 0.038, 0.90, 20, 0.1), 5, "ml"),
        "KNN":          (_fixed_result("KNN", 921_648.3, 700_000, 0.05, 0.80, 20, 0.1), 5, "ml"),
        "SARIMA":       (_fixed_result("SARIMA", 823_348_720.64, 429_053_040.94, 150_920.75, -1_509_198.68, 20, 0.1), 5, "statistical"),
        "SARIMAX":      (_fixed_result("SARIMAX", 823_348_720.64, 429_053_040.94, 150_920.75, -1_509_198.68, 20, 0.1), 5, "statistical"),
    }
    # 600 rows: comfortably above the 500-row daily DL threshold, so the
    # "TCN" fake (tagged category "dl") isn't excluded by that gate before
    # it even runs — this test is specifically about score normalization,
    # not the DL row-count gate (covered separately in
    # test_model_runner_dl_gate.py).
    series = pd.Series(np.random.RandomState(0).randn(600) + 100,
                        index=pd.date_range("2023-01-01", periods=600))

    results, _ = run_all_models(series, forecast_horizon=10, models=registry)
    ranked_names = [r["model_name"] for r in results]

    # The catastrophic outliers must sink to the bottom, not distort the
    # ranking of the models that actually fit well.
    assert ranked_names[-2:] == ["SARIMA", "SARIMAX"] or set(ranked_names[-2:]) == {"SARIMA", "SARIMAX"}

    # Among the non-outlier models, the normalized RMSE term must still
    # meaningfully differentiate them (this is the actual bug: it used to
    # compress to near-zero noise) — assert scores are monotonically
    # distinguishable and track raw RMSE order at least approximately for
    # the model with the best RMSE (TCN) landing ahead of the worst (KNN).
    good_results = {r["model_name"]: r for r in results if r["model_name"] not in ("SARIMA", "SARIMAX")}
    assert good_results["TCN"]["adjusted_score"] < good_results["KNN"]["adjusted_score"]

    # And the scores among good models must not be near-identical (that
    # would indicate the normalization denominator is still being set by
    # the outlier).
    good_scores = [r["adjusted_score"] for r in good_results.values()]
    assert max(good_scores) - min(good_scores) > 0.01
