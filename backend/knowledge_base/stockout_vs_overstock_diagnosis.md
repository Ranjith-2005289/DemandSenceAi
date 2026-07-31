# Diagnosing Stockouts vs. Overstock

Both problems can look similar in raw sales data (a dip or plateau), so
the diagnosis has to look past units sold alone.

**Stockout signature:**
- Sales flatten or drop abruptly to near-zero while on-hand inventory
  also drops to zero or near-zero at the same time — the sales "ceiling"
  is inventory-constrained, not demand-constrained. If you only have
  sales data (no inventory feed), a tell-tale sign is unusually flat,
  capped-looking sales (e.g. exactly the same number for several
  consecutive periods) right before a plateau or dip — real unconstrained
  demand is noisy, a hard ceiling usually means supply ran out.
- A demand forecast built on stockout-affected history will
  systematically under-forecast future demand, because the historical
  "sales" data undercounts true demand (customers who wanted to buy but
  couldn't). This is one of the most common, hardest-to-spot forecasting
  errors — the fix is demand *un*-censoring (imputing what sales likely
  would have been) before training, not just feeding raw sales history
  into a model.
- Recovery after restock often shows a short-lived spike above the
  "true" trend as backlogged/deferred demand clears — don't mistake this
  bounce for a new higher baseline.

**Overstock signature:**
- Weeks of supply climbing steadily, sell-through rate below plan or
  below the category average, and a growing gap between the forecast's
  implied "should sell" rate and what's actually moving.
- A forecast whose actual demand keeps coming in below the model's
  prediction, especially if the gap is *widening* over time, suggests the
  original forecast (and the resulting purchase order) was built on stale
  assumptions — check whether a step change happened (a competitor
  launch, a price change, a shift to a different channel) around when the
  forecast started missing.
- Overstock risk compounds for perishable or highly seasonal goods —
  carrying cost isn't just capital, it's the risk of the product becoming
  worthless (expiration, season end, model-year change).

**Practical rule of thumb:** if a declining trend in the data coincides
with declining on-hand inventory too, suspect a stockout depressing
observed sales, not weakening demand. If a declining trend coincides with
flat or rising inventory, it's more likely genuine softening demand or an
overstock situation building — the appropriate response (expedite
replenishment vs. discount to clear) is opposite in each case, so getting
this diagnosis right matters more than the forecast's raw accuracy.
