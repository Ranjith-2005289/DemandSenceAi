# Model Updates — DemandSense AI v0.2

A short summary of what changed in the forecasting models, and why.

---

## 1. At a glance

| | Before (v0.1) | Now (v0.2) |
|---|---|---|
| Models | 19 | **15 + a top-3 Ensemble** |
| How the winner is picked | Each model's own CV score (not comparable across models) | **Same holdout backtest for every model, ranked by MASE** |
| Accuracy of the chosen model (geo-mean MASE, lower is better) | 0.862 | **0.765** |
| Run time (731-row daily series) | ~100 s | **~20 s** |
| Large series (e.g. 50k hourly rows) | Timed out after 6+ min | **~16 s** |

**MASE** = the model's error divided by the error of a "seasonal naive" forecast (copy last season's values). Below 1 means the model beats that simple baseline.

---

## 2. Models removed (8)

All models were tested on 10 series (5 real uploads, 5 synthetic) by hiding the last part of each series and scoring the forecast on it.

| Model | Held-out MASE | Reason removed |
|---|---|---|
| Transformer, LSTM, CNN1D, BiLSTM, GRU, TCN, RNN | 1.35 – 1.68 | All **worse than the seasonal-naive baseline (1.17)**, and the slowest models. Retail series are usually too short for deep learning. |
| KNN | 1.165 | No better than the baseline; worst ML model. |

Removing the deep learning models also removed the PyTorch dependency.

---

## 3. Models added (4 + Ensemble)

| Model | What it does | When it helps |
|---|---|---|
| **Theta** | Splits the series into a long-term trend line and a short-term smoothed part, then combines them. Won the M3 forecasting competition. | Trending series; a strong, fast general-purpose model. |
| **AutoETS** | Tries the error/trend/seasonality forms of exponential smoothing and keeps the best one by AICc. | Series with clear level, trend and seasonal patterns. |
| **MSTL** | Separates several seasonal cycles at once (e.g. weekly + yearly for daily data, daily + weekly for hourly data). | High-frequency data with more than one seasonal cycle. |
| **SeasonalNaive** | Repeats the values from the last season. | A baseline: any model that can't beat it shouldn't win. |
| **Ensemble** | Equal-weight average of the 3 best models. | Averaging usually reduces error and makes forecasts more stable. |

---

## 4. Final model list (15)

| Family | Models |
|---|---|
| Statistical | ARIMA, SARIMA, SARIMAX |
| Exponential smoothing | HoltWinters, ExponentialSmoothing, AutoETS, Theta |
| Decomposition | MSTL, Prophet |
| Machine learning | XGBoost, RandomForest, SVM |
| Intermittent demand | CrostonSBA |
| Baselines | MovingAverage, SeasonalNaive |
| + | **Ensemble** (top 3) |

---

## 5. Improvements to how models are trained and selected

1. **Fair comparison (common backtest).** Previously some models (XGBoost, RF, SVM, ARIMA) were scored one step at a time with the true past values fed in, while others forecast many steps blind. That made the ML models look better than they really were. Now every model is retrained from the **same 2 forecast origins** and forecasts the **same unseen windows**, and they are ranked by MASE.
2. **Better prediction intervals.** The 95% band is now calibrated from each model's actual backtest errors, and widens with the forecast horizon.
3. **Faster training.** Models run in parallel **processes** instead of threads (statsmodels blocks threads from running in parallel).
4. **Large datasets.** Models train on the **most recent 2,000 points**. For hourly data that's about 83 days, which is plenty for a short forecast. The results page tells you when this happens.
5. **SARIMA/SARIMAX speed-up.** They now score each check with one multi-step forecast and only run the full order search for short seasonal cycles, so hourly data takes ~5 s instead of timing out.
6. **Bug fixes.** Holt-Winters now detects weekly/quarterly data correctly (it used to treat it as daily) and can also try a no-trend model.

---

## 6. Where the code lives

| File | What's in it |
|---|---|
| `backend/services/model_runner.py` | Model registry, backtest, ranking, ensemble, process pool |
| `backend/models/advanced_stat_models.py` | Theta, AutoETS, MSTL, SeasonalNaive |
| `backend/models/sarima_model.py` | Faster SARIMA / SARIMAX evaluation |
| `backend/services/best_model.py` | Leaderboard with MASE and ensemble members |
| `backend/tests/test_backtest_ranking.py` | Tests for the new ranking |

All **141 backend tests pass**.
