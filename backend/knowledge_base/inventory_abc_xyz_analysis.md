# ABC/XYZ Inventory Classification

ABC analysis ranks products by revenue or volume contribution using the
Pareto principle: typically 10-20% of SKUs (A-items) drive 70-80% of
revenue, the next 30% (B-items) drive about 15-25%, and the remaining
50-60% (C-items) drive only 5-10%. Use A-items to prioritize forecasting
effort, safety stock investment, and frequent review cycles — errors here
are the most costly. C-items can tolerate simpler forecasting methods
(moving average, simple exponential smoothing) and looser review cycles
since the cost of being wrong is small relative to revenue.

XYZ analysis classifies products by demand variability instead of value:
X-items have stable, predictable demand (low coefficient of variation,
roughly under 0.5), Y-items have moderate variability often tied to
seasonality or trend, and Z-items have high, erratic variability (new
products, promotional-driven items, or very low-volume SKUs where a few
large orders swing the average).

Combining both gives nine segments. The two extremes matter most in
practice:

- **AX** (high value, stable demand): the safest candidates for tight
  inventory targets and automated replenishment — small forecast errors
  here are cheap to buffer against.
- **CZ** (low value, erratic demand): don't over-invest in sophisticated
  forecasting for these — a simple reorder-point rule with modest safety
  stock is usually more cost-effective than chasing forecast accuracy on
  noise.

Practical guidance for a forecasting system: segment the catalog into
ABC/XYZ first, then match model complexity to the segment. Running a full
ensemble of statistical, ML, and deep learning models on every single SKU
is often wasted effort for C/Z items; the payoff from better models is
concentrated in A and X segments where volume and predictability make
accuracy improvements actually move the bottom line.
