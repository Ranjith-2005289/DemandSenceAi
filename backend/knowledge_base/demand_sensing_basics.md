# Demand Sensing: Short-Term Signal Beyond Pure History

Traditional statistical forecasting projects the future mainly from a
product's own past pattern (trend, seasonality, recent lags). Demand
sensing supplements that with more immediate, short-horizon signals to
catch shifts faster than a model that only looks backward at its own
history can:

- **Point-of-sale velocity in the last few days**, not just weekly/
  monthly aggregates — a sudden acceleration or deceleration this week is
  often visible days before it shows up clearly in a longer rolling
  average.
- **Leading indicators**: web traffic/search interest for a product or
  category, cart-adds, wishlist activity, or weather for
  weather-sensitive categories — these can move before actual purchases
  do and give an early read on a demand shift a lag-based model won't
  see until it's already happened.
- **Local/regional events**: a store or region hosting a local event,
  experiencing unusual weather, or a nearby competitor closing can create
  a real, short-term local demand shift that a model trained on
  aggregate history has no way to anticipate.

**When demand sensing matters most:** short lead-time categories (where
there's actually enough time to react to a fast signal before it's too
late), highly volatile or promotional categories, and situations where
the cost of being wrong in either direction is high (perishables,
capacity-constrained categories).

**When it matters less:** long lead-time categories where by the time a
short-term signal is detected there's no time left to act on it anyway
(better to invest in improving the base statistical forecast and
lead-time reduction instead), and very stable, low-variability categories
where the base forecast is already reliable and short-term signals mostly
just add noise.

**Practical integration point:** rather than replacing the statistical/ML
forecast, demand-sensing signals are usually best used as an adjustment
layer on top of it — nudging the baseline forecast up or down for the
next few periods when a strong signal appears, while leaving the
longer-horizon forecast (used for purchasing decisions with real lead
time) driven by the more stable underlying model.
