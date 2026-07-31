# Safety Stock and Reorder Points

Safety stock exists to absorb two kinds of uncertainty: variability in
demand during the lead time, and variability in the lead time itself (a
supplier occasionally shipping late). A common formula:

`Safety Stock = Z × sqrt((lead_time × demand_variance) + (avg_demand² × lead_time_variance))`

where Z is the service-level factor (e.g. Z≈1.65 for ~95% service level,
Z≈2.33 for ~99%). The key intuition even without doing the exact math:
safety stock should scale with the *forecast's own uncertainty* for that
item, not be a flat number of days/units applied uniformly across the
catalog. A product with a highly accurate, stable forecast needs much
less safety stock than one with high forecast error, even at the same
average volume — this is why a forecasting system's error metrics
(RMSE, prediction intervals) are directly useful for inventory decisions,
not just for judging "which model is best."

**Reorder point** = (average demand during lead time) + safety stock.
When on-hand inventory drops to this level, it's time to place the next
order so the new stock arrives before the current stock is depleted.

Practical guidance:
- **High-value, low-variability items (AX in ABC/XYZ terms):** can run
  leaner safety stock relative to volume since demand is predictable —
  the capital saved is usually worth more than the small stockout risk.
- **Long lead-time or single-source items:** need disproportionately more
  safety stock than short-lead-time items with the same demand
  variability, because there's more time for something to go wrong and
  less ability to react once it does.
- **Widening prediction intervals or rising forecast error** (visible in
  a system that reports RMSE/confidence per model) is itself a signal to
  raise safety stock for that item *before* a stockout happens — don't
  wait for the stockout to occur to react to declining forecastability.
- **New products** have no real demand history, so safety stock
  calculated from a short, thin history is often wrong in either
  direction; use a category analog or a wider initial buffer until enough
  real data accumulates to trust a tailored estimate.
