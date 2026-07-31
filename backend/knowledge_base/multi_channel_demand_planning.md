# Multi-Channel and Omnichannel Demand Planning

When a product sells through multiple channels (physical stores,
e-commerce, marketplaces, wholesale), forecasting the combined total and
forecasting each channel separately can give very different, both useful,
answers.

**Why channel-level detail matters even if only the total is needed for
purchasing:** channels often have different seasonality (e-commerce may
spike around online-specific shopping events; physical retail may spike
around local foot-traffic patterns or weather), different lead times to
replenish, and different promotional calendars. A combined forecast can
look stable in total while masking one channel growing and another
declining — which matters enormously for allocation decisions even if
total production/purchase quantity is forecast correctly.

**Cross-channel substitution effects:** when a product goes out of stock
in one channel, some demand shifts to another channel rather than being
lost entirely (a customer who can't find an item in-store may buy it
online, or vice versa). This means a channel's own sales history can be
misleading in isolation — a channel showing "growth" might actually be
absorbing stockout overflow from a sibling channel, not genuine organic
growth in that channel.

**Marketplace/wholesale channels** often have less granular or delayed
data (a wholesale partner may report sell-through weekly or monthly, with
a lag), so forecasts based purely on shipment-to-partner data (sell-in)
can diverge substantially from actual end-customer demand (sell-through)
— a spike in shipments to a wholesale partner might just be the partner
restocking their own inventory, not a genuine demand increase.

**Practical guidance:** forecast at the channel level when channels have
materially different seasonality, lead times, or promotional calendars,
then aggregate for total purchasing decisions — but always sanity-check
the sum against a total-level forecast, since channel-level models can
individually be reasonable while compounding errors when summed.
