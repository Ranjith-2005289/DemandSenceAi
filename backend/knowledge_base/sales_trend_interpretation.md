# Interpreting Sales Trends Correctly

A trend line by itself doesn't say *why* sales are moving — that requires
combining the statistical trend with business context. Some patterns
worth checking before drawing conclusions:

**Distinguishing trend from noise.** A few consecutive down weeks is not
necessarily a "declining trend" — check whether the recent movement is
within the normal historical volatility band (e.g. within roughly 1-2
standard deviations of past week-over-week changes) or whether it's a
genuine break from the historical pattern. A model's own confidence
interval width is a reasonable proxy: if the recent dip is still inside
the forecast's prediction interval, treat it as noise, not a new trend.

**Distinguishing trend from seasonality.** A downward move during a
period the same product historically dips every year (a known seasonal
low) is expected, not a warning sign. Compare the current trajectory
against the same point in prior cycles (year-over-year), not just the
immediately preceding weeks, before calling something "declining."

**Common real causes of a genuine declining trend**, roughly in order of
how often they explain what looks like organic demand erosion:
1. A stockout depressing observed sales (see stockout-vs-overstock notes)
   — check inventory data before concluding demand itself is falling.
2. A price increase or promotion cadence change around the date the
   trend started.
3. A new competitor or substitute product entering the market.
4. Distribution loss (the product went out of stock at a channel or
   store, not overall) — check whether the decline is broad-based across
   all locations/channels or concentrated in a few.
5. Genuine category decline (the whole category is shrinking, not just
   this product) — check whether category-level or competitor sales show
   the same pattern; if only this one product is falling while the
   category is flat, the cause is product-specific (assortment, pricing,
   quality perception), not macro.
6. Product lifecycle maturity — many products naturally decline after a
   growth phase; compare against the product's own historical lifecycle
   shape rather than assuming every decline needs a "fix."

**Growing trends** deserve the same scrutiny before assuming success:
check whether growth is broad or concentrated in one channel/promotion,
and whether the current run-rate is sustainable or promotion-inflated
before committing to higher production/inventory based on it.
