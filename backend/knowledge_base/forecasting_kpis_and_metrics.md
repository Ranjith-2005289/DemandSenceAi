# Forecasting KPIs and What They Actually Mean for the Business

**MAPE (Mean Absolute Percentage Error)** — average error as a percentage
of actual value. Easy to communicate ("the forecast is off by about
X% on average") but breaks down for products with actual values near
zero (a tiny actual value turns a small absolute error into a huge
percentage) and is asymmetric — it penalizes over-forecasting less
harshly than under-forecasting in percentage terms. A rough,
commonly-used translation to a business-friendly "accuracy" figure is
`Accuracy% ≈ 100 − MAPE%`, but this should be treated as a rough
indicator, not a precise statistic, especially for low-volume or
intermittent-demand items.

**RMSE (Root Mean Squared Error)** — penalizes large errors much more
than small ones (because errors are squared before averaging). Useful
when a few big misses matter more than many small ones (e.g. a single
massive stockout is worse than being consistently 5% off), but it's in
the same units as the data, so it isn't directly comparable across
products with very different volume scales.

**MAE (Mean Absolute Error)** — average absolute error in the same units
as the data, without RMSE's extra penalty on large errors. Easier to
interpret directly ("on average the forecast misses by N units") but
less sensitive to occasional large misses than RMSE.

**R² (coefficient of determination)** — how much of the variance in
actual demand the model's predictions explain, relative to a naive
"just predict the average" baseline. An R² near 1 means the model tracks
real movements well; near 0 or negative means the model is doing no
better (or worse) than predicting a flat average — a genuinely important
red flag if the model's more conventional error metrics look OK yet R² is
poor, since it means the model isn't capturing the actual pattern of
ups and downs even if the average error size looks acceptable.

**Bias** (average of actual − forecast, not absolute) reveals whether a
model is systematically over- or under-forecasting rather than just how
far off it is on average. A model can have "good" MAPE/RMSE while still
being consistently biased in one direction — which matters enormously for
inventory decisions, since a systematic under-forecast bias will
repeatedly cause stockouts even if the average error magnitude looks
small.

**Practical guidance for choosing/interpreting a "best" model:** no
single metric tells the whole story — a model selection process that
weighs multiple metrics (error magnitude, bias, and stability of the
forecast trajectory) is more robust than optimizing for any one number in
isolation, and the business context (is a large single miss worse than
many small ones? does under- or over-forecasting cost more here?) should
influence which metric to weight most heavily for a given category.
