# Choosing Between Forecasting Model Families

No single model family is universally best — the right choice depends on
the data's shape, not on which model is newest or most sophisticated.

**Statistical models (ARIMA, SARIMA, Holt-Winters, Prophet):** strong
when there's a clear trend and/or seasonal pattern and a reasonably
consistent history. They can extrapolate a trend correctly (unlike
tree-based models, see below) and Prophet/SARIMA explicitly handle
holiday/calendar effects. They need enough history to estimate their
parameters reliably and generally assume the underlying pattern is fairly
stable — a sudden structural break (a big pricing change, a new
competitor) can confuse them until enough post-break data accumulates.

**Tree-based and instance-based ML models (XGBoost, Random Forest, KNN,
SVM) on lag/rolling features:** strong at capturing complex, nonlinear
interactions between lag values, rolling statistics, and calendar
features — useful when there are many potentially relevant features
beyond just "time." Their key structural limitation: they cannot predict
values outside the range seen during training, so on a series with a
strong sustained trend they tend to systematically undershoot continued
growth (or overshoot continued decline) unless the trend is separately
modeled and removed before training (e.g. detrending, then modeling the
residual) — a plain moving-average or naive baseline can sometimes beat
these models specifically on strongly trending data unless this
limitation is addressed.

**Deep learning models (LSTM, GRU, Transformer, TCN, etc.):** can in
principle capture complex temporal patterns, but they're the most
data-hungry family — with only a few hundred data points they usually
underperform simpler methods, and their advantage typically only shows up
with substantial history (typically thousands of observations) and/or
when there are rich multivariate signals to exploit. For a single
product's weekly or even daily sales history, they're often not the best
choice unless there's a lot of history and/or additional correlated
series to learn from.

**Practical guidance for running an ensemble and picking a winner:**
- Always compare candidate models against a naive baseline (e.g. a
  seasonal-naive forecast: predict this period equals the same period in
  the last seasonal cycle) — a sophisticated model that can't beat a
  naive baseline on a given dataset is a real warning sign, not just an
  academic curiosity.
- Weight model selection by the actual forecast horizon needed (see
  supplier lead-time notes) — a model that wins at 1-step-ahead accuracy
  isn't necessarily the best choice for a 30-day-ahead purchasing
  decision.
- For small or highly volatile datasets, prefer the simpler, more stable
  models — added model complexity needs enough signal in the data to
  actually pay off, and on thin or noisy data it mostly just adds
  overfitting risk.
