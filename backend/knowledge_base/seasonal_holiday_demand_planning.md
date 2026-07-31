# Seasonal and Holiday/Festival Demand Planning

Seasonal demand has a repeating, calendar-driven pattern (weekly,
monthly, or yearly), while holiday/festival demand is a sharper, often
larger spike concentrated around specific dates that can shift year to
year (e.g. lunar-calendar festivals, moveable holidays like Easter or
Diwali). Treating both the same way in a forecast usually underestimates
the holiday spike and overestimates the trailing days after it.

Practical approach:
1. **Separate the baseline from the event.** Model the underlying
   seasonal pattern (e.g. weekly seasonality) first, then treat known
   holidays/festivals as explicit calendar events layered on top —
   this is exactly what Prophet's holiday-effects mechanism and SARIMAX's
   exogenous calendar dummies are designed for, rather than expecting a
   generic seasonal term to capture both at once.
2. **Model pre- and post-event effects, not just the day itself.** Many
   retail categories see a build-up in the 3-14 days before a major
   holiday (gifting, stocking up) and a trough immediately after. A
   single-day spike indicator misses most of the actual demand shift.
3. **Watch for calendar drift.** Holidays tied to a lunar or lunisolar
   calendar (many festivals) land on different Gregorian dates each year.
   A model trained on "day of year" alone will misalign the effect by
   days or weeks in the next cycle; use the actual event date per year as
   a feature instead of a fixed day-of-year assumption.
4. **New or growing holidays need judgment overlays, not pure history.**
   If a promotional shopping event is only 2-3 years old, historical data
   is thin and trend-fitting can wildly over- or under-shoot; blend the
   statistical forecast with category benchmarks or manual adjustment for
   these cases rather than trusting the model in isolation.
5. **Post-holiday returns and markdowns depress the following weeks.**
   If you're forecasting revenue rather than units, expect a real dip in
   net demand after a holiday even if unit sales look flat, driven by
   returns and clearance pricing.

For a system running multiple candidate models, models with explicit
seasonality/holiday support (Prophet, SARIMA/SARIMAX with calendar exog
features, Holt-Winters) are structurally better suited to holiday-heavy
categories than models that only see lag/rolling features unless those
features explicitly encode the holiday calendar.
