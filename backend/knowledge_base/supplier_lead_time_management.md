# Supplier Lead Time Management

Lead time — the gap between placing an order and having usable stock —
directly determines how far ahead a forecast needs to be accurate to be
useful. A forecast that's excellent at predicting next week is close to
useless for a purchase decision that requires 12 weeks of lead time; what
matters for that decision is accuracy at the 12-week-ahead horizon
specifically, which is typically much worse than short-horizon accuracy
for almost any model (forecast error compounds with horizon length).

**Practical implications for using a multi-horizon forecast:**
- Always evaluate/trust a forecast at the horizon that matches the actual
  decision lead time, not at the model's best-case (usually shortest)
  horizon. A model that looks highly accurate at a 7-day forecast can
  still be far less reliable at 60-90 days out — check the accuracy
  metric specifically for the horizon relevant to the purchase decision
  being made, if available.
- **Variable lead times** (a supplier that sometimes ships in 4 weeks,
  sometimes 8) add their own uncertainty on top of demand forecast error
  — both sources compound, so safety stock and reorder timing need to
  account for lead-time variability separately from demand variability
  (see safety-stock notes).
- **Long lead-time categories** (imported goods, made-to-order,
  multi-stage manufacturing) benefit disproportionately from earlier,
  longer-horizon demand signals — pre-order data, search trends, or
  category-level leading indicators — since by the time point-of-sale
  data confirms a trend, it's often too late to react within the lead
  time window.
- **Splitting orders** (part of the order via a fast/expensive path, part
  via the slow/cheap path) is a common practical hedge against long
  lead-time uncertainty rather than trying to solely improve forecast
  accuracy far out — sometimes the right answer to "the forecast isn't
  accurate enough at this horizon" is to change the lead time itself
  (nearshoring, dual-sourcing, safety buffer) rather than demanding a
  better model.
