# Forecasting New Products with Little or No History

Standard time-series forecasting (ARIMA, exponential smoothing, lag-based
ML models) fundamentally needs historical data to learn from — a brand
new product has none, so these methods either fail outright (too few
observations) or produce unreliable, high-variance estimates from a thin
history in the first few weeks.

**Analog forecasting** is the standard workaround: find a comparable
existing product (similar category, price point, launch season, target
customer) and use its early sales trajectory, shape, and eventual
steady-state level as a template for the new product, scaled by
judgment or by relative marketing spend/distribution breadth.

**Practical staged approach:**
1. **Pre-launch (no data):** rely entirely on analog products and
   planner/category judgment. Don't attempt statistical forecasting yet.
2. **Early weeks (a handful of data points):** blend the analog-based
   estimate with the thin real data, weighted mostly toward the analog —
   a few real data points aren't enough to override a well-chosen
   comparison yet, and models trained on 5-10 points will overfit noise.
3. **Ramp period (enough history for simple methods):** once there are
   roughly 15-30 clean observations, simpler models (moving average,
   simple exponential smoothing, basic regression on trend + calendar)
   become viable and should start being weighted more heavily than the
   analog.
4. **Maturity (enough history for full modeling):** once there's enough
   history to support proper train/validation splits (generally 30+
   points, more if there's meaningful seasonality to capture), the full
   model suite becomes appropriate and the analog can be retired.

**Common mistakes:** launching a full ensemble of sophisticated models
immediately on a brand-new SKU (they'll either error out from insufficient
data or badly overfit the first few noisy points), and failing to
re-evaluate the analog choice once real data starts to diverge from it —
if week 3-4 actuals are consistently 2x the analog's early trajectory,
that's a signal to revise the reference product or launch assumptions,
not to keep forecasting off the original analog.
