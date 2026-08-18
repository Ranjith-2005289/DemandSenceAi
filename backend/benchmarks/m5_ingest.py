"""
m5_ingest.py — Reshape the M5 (Walmart) competition dataset into the long-format
CSVs this system's /upload endpoint expects, at a chosen aggregation level.

Why this exists
---------------
M5 ships sales in WIDE format: one row per item-store, one COLUMN per day
(d_1 ... d_1941). This system expects LONG format: one row per (date, series),
with a date column, a numeric target column, and optional exogenous columns.
This module does that reshape, and joins in the two exogenous signals M5
provides that this system can actually use:

  - sell_price  (from sell_prices.csv)  -> matched by PRICE_KEYWORDS ("price")
  - promo_flag  (derived from sell_price markdowns) -> matched by PROMO_KEYWORDS

Those column names are chosen deliberately so that upload.py's
_detect_exog_columns() auto-detects them and forecast.py routes them to
XGBoost / RandomForest / SARIMAX as real exogenous features. See
services/preprocessor.py PRICE_KEYWORDS / PROMO_KEYWORDS.

IMPORTANT — where promo_flag actually comes from
--------------------------------------------------
M5 does NOT ship an explicit promotion/markdown indicator. calendar.csv's
event_name_1/2 columns are CALENDAR HOLIDAYS (Super Bowl, Easter, Orthodox
Christmas, ...) — not retailer promotions, and conflating the two is a real
correctness bug an earlier version of this module had: it labeled the holiday
flag "promo_flag", which would silently misrepresent holiday effects as
promotional exogenous data in any downstream claim about "real promo features".

Instead, promo_flag here is DERIVED from sell_prices.csv: a week is flagged as
promotional when a series' aggregated sell_price drops materially below its own
trailing baseline price (see compute_markdown_promo_flag() below) — the
standard way the M5 literature approximates promotions, since M5 provides no
better signal. The original holiday/event flag is kept as a separate,
honestly-named `calendar_event` column (not matched by PROMO_KEYWORDS, so it is
NOT auto-routed as an exog promo feature — it is retained only as descriptive
data; Prophet's own country-code holiday calendar, wired through
routers/forecast.py's `country` parameter, is the system's actual holiday
mechanism and is unrelated to this column).

Aggregation level matters more than anything else here
-----------------------------------------------------
M5's bottom level (item x store) is heavily intermittent — most item-store-day
cells are zero — so forecasting there produces series with essentially no
learnable signal, and every model collapses to predicting the mean (R^2 ~ 0).
Aggregating upward concentrates the signal. The `--level` flag exposes M5's
standard 12-level hierarchy so this is a deliberate, reportable experimental
variable rather than an accident.

Usage
-----
    # One aggregate series (total US sales), for a quick sanity check:
    python -m benchmarks.m5_ingest --m5-dir ~/m5 --level 1 --out-dir data/m5

    # 70 store-department series — the recommended level for benchmarking:
    python -m benchmarks.m5_ingest --m5-dir ~/m5 --level 9 --out-dir data/m5

    # Cap how many series are emitted (largest by total volume first):
    python -m benchmarks.m5_ingest --m5-dir ~/m5 --level 12 --max-series 200 \
        --out-dir data/m5

Outputs
-------
  <out-dir>/m5_level<N>_<series_id>.csv   one CSV per series (upload-ready)
  <out-dir>/m5_level<N>_manifest.csv      index of every emitted series + stats

Expected M5 input files in --m5-dir (from the Kaggle competition):
  sales_train_evaluation.csv   (30490 rows x 1947 cols; d_1..d_1941)
  calendar.csv                 (1969 rows)
  sell_prices.csv              (~6.8M rows)
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── M5's 12 standard aggregation levels ──────────────────────────────────────
# Level 1 is the grand total (no grouping keys); levels 2-12 group by these
# columns. Names follow the M5 competitors' guide.
AGGREGATION_LEVELS: dict[int, tuple[str, list[str]]] = {
    1:  ("total",              []),
    2:  ("state",              ["state_id"]),
    3:  ("store",              ["store_id"]),
    4:  ("category",           ["cat_id"]),
    5:  ("department",         ["dept_id"]),
    6:  ("state_category",     ["state_id", "cat_id"]),
    7:  ("state_department",   ["state_id", "dept_id"]),
    8:  ("store_category",     ["store_id", "cat_id"]),
    9:  ("store_department",   ["store_id", "dept_id"]),
    10: ("item",               ["item_id"]),
    11: ("item_state",         ["item_id", "state_id"]),
    12: ("item_store",         ["item_id", "store_id"]),
}

# Emitted column names. These are load-bearing: they must match the keyword
# lists in services/preprocessor.py for auto-detection to route them as exog.
DATE_COL = "date"
TARGET_COL = "sales"
PRICE_COL = "sell_price"       # contains "price" -> PRICE_KEYWORDS
PROMO_COL = "promo_flag"       # contains "promo" -> PROMO_KEYWORDS; DERIVED FROM
                               # PRICE MARKDOWNS, not from calendar events
EVENT_COL = "calendar_event"   # holiday/event indicator. Deliberately named so it
                               # matches NEITHER keyword list — it is descriptive
                               # only and must not be routed as a promo exog.

_SAFE_ID = re.compile(r"[^A-Za-z0-9_.-]+")


def _safe_filename(value: str) -> str:
    """Make a series id safe to use as a filename component."""
    return _SAFE_ID.sub("_", str(value)).strip("_") or "series"


# ── Loading ──────────────────────────────────────────────────────────────────

def load_m5(m5_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load the three M5 CSVs. Accepts sales_train_evaluation.csv (1941 days,
    preferred) or falls back to sales_train_validation.csv (1913 days).
    """
    sales_path = m5_dir / "sales_train_evaluation.csv"
    if not sales_path.exists():
        fallback = m5_dir / "sales_train_validation.csv"
        if not fallback.exists():
            raise FileNotFoundError(
                f"Neither sales_train_evaluation.csv nor sales_train_validation.csv "
                f"found in {m5_dir}. Download the M5 dataset from "
                f"https://www.kaggle.com/competitions/m5-forecasting-accuracy/data"
            )
        logger.warning("sales_train_evaluation.csv not found — using %s (1913 days)", fallback.name)
        sales_path = fallback

    for name in ("calendar.csv", "sell_prices.csv"):
        if not (m5_dir / name).exists():
            raise FileNotFoundError(f"{name} not found in {m5_dir}")

    logger.info("Loading %s ...", sales_path.name)
    sales = pd.read_csv(sales_path)
    logger.info("  sales: %d rows x %d cols", *sales.shape)

    calendar = pd.read_csv(m5_dir / "calendar.csv", parse_dates=["date"])
    logger.info("  calendar: %d rows", len(calendar))

    # sell_prices is the big one (~6.8M rows) — narrow dtypes keep it cheap.
    prices = pd.read_csv(
        m5_dir / "sell_prices.csv",
        dtype={"store_id": "category", "item_id": "category",
               "wm_yr_wk": "int32", "sell_price": "float32"},
    )
    logger.info("  sell_prices: %d rows", len(prices))

    return sales, calendar, prices


# ── Aggregation ──────────────────────────────────────────────────────────────

def _day_columns(sales: pd.DataFrame) -> list[str]:
    """The d_1 ... d_N columns, in numeric order (not lexicographic)."""
    cols = [c for c in sales.columns if c.startswith("d_")]
    if not cols:
        raise ValueError("No d_* day columns found — is this really the M5 sales file?")
    return sorted(cols, key=lambda c: int(c.split("_")[1]))


def aggregate_sales(sales: pd.DataFrame, level: int) -> tuple[pd.DataFrame, list[str]]:
    """
    Sum the wide sales frame up to `level`'s grouping keys.

    Done on the WIDE frame deliberately: grouping 30,490 rows and then melting
    the (much smaller) result is far cheaper in both time and memory than
    melting 30,490 x 1,941 = ~59M rows first and grouping afterwards.

    Returns (wide_aggregated, day_cols) where wide_aggregated has the grouping
    keys as columns plus one column per day.
    """
    if level not in AGGREGATION_LEVELS:
        raise ValueError(f"level must be one of {sorted(AGGREGATION_LEVELS)}, got {level}")

    _, keys = AGGREGATION_LEVELS[level]
    day_cols = _day_columns(sales)

    if not keys:
        # Level 1: grand total — one synthetic group.
        agg = sales[day_cols].sum().to_frame().T
        agg = pd.concat([pd.DataFrame({"series_id": ["TOTAL"]}), agg.reset_index(drop=True)], axis=1)
        return agg, day_cols

    grouped = sales.groupby(keys, observed=True)[day_cols].sum().reset_index()
    # Build the id column and concat once rather than insert()-ing into a frame
    # that already has ~1,941 day columns — repeated insert on a wide frame
    # triggers pandas' block fragmentation and is markedly slower at M5 scale.
    series_id = grouped[keys].astype(str).agg("--".join, axis=1).rename("series_id")
    agg = pd.concat([series_id, grouped], axis=1)
    logger.info("Level %d (%s): %d series", level, AGGREGATION_LEVELS[level][0], len(agg))
    return agg, day_cols


def aggregate_prices(
    sales: pd.DataFrame,
    prices: pd.DataFrame,
    level: int,
) -> pd.DataFrame | None:
    """
    Average sell_price up to the same aggregation level, per M5 week (wm_yr_wk).

    Returns a frame [series_id, wm_yr_wk, sell_price], or None for level 1
    (a mean price across all 3,049 products is not a meaningful signal, so no
    price exog is emitted there).

    Note this is an unweighted mean across the constituent item-stores, not a
    sales-weighted one. Sales-weighting would require joining prices onto the
    melted ~59M-row sales frame; the unweighted mean is a deliberate cost
    trade-off and is stated here so it can be reported honestly in the paper.
    """
    _, keys = AGGREGATION_LEVELS[level]
    if not keys:
        return None

    # Map every (item_id, store_id) to its grouping keys, from the sales file.
    lookup = sales[["item_id", "store_id", "dept_id", "cat_id", "state_id"]].drop_duplicates()
    # sell_prices is read with category dtypes for memory; cast the join keys
    # back to str so the merge doesn't depend on category levels matching.
    prices = prices.copy()
    for col in ("item_id", "store_id"):
        prices[col] = prices[col].astype(str)
        lookup[col] = lookup[col].astype(str)
    merged = prices.merge(lookup, on=["item_id", "store_id"], how="inner")
    if merged.empty:
        logger.warning("Price join produced no rows — emitting no price exog.")
        return None

    merged["series_id"] = merged[keys].astype(str).agg("--".join, axis=1)
    out = (
        merged.groupby(["series_id", "wm_yr_wk"], observed=True)["sell_price"]
        .mean()
        .reset_index()
    )
    logger.info("Aggregated prices: %d (series, week) rows", len(out))
    return out


def build_calendar_features(calendar: pd.DataFrame, level: int) -> pd.DataFrame:
    """
    Build a per-day frame of [d, date, wm_yr_wk, calendar_event, snap_CA/TX/WI].

    calendar_event is 1 when M5 records a special event on that date
    (event_name_1 non-null). These are HOLIDAYS — Super Bowl, Easter, Orthodox
    Christmas — NOT retailer promotions, and the column is named accordingly so
    it is never auto-detected as promotional exogenous data. See
    compute_markdown_promo_flag() for the actual promotion proxy.

    SNAP flags are kept separately because they are state-specific and only
    resolvable at state/store levels.
    """
    cal = calendar.copy()
    cal[EVENT_COL] = cal["event_name_1"].notna().astype("int8")
    keep = ["d", "date", "wm_yr_wk", EVENT_COL]
    for c in ("snap_CA", "snap_TX", "snap_WI"):
        if c in cal.columns:
            keep.append(c)
    return cal[keep]


def compute_markdown_promo_flag(
    price: np.ndarray,
    window: int = 56,
    threshold: float = 0.05,
    min_periods: int = 7,
) -> np.ndarray:
    """
    Derive a promotion indicator from price markdowns.

    M5 ships no explicit promotion column, so a promotion is inferred the way
    the M5 literature does it: a day is flagged when the selling price sits
    materially below the product's own recent baseline "regular" price.

    The baseline is a trailing rolling MAX. Central-tendency baselines fail
    here: a rolling mean is dragged down by the discount itself, and a rolling
    median silently stops detecting any markdown lasting longer than half the
    window (verified — a 50-day discount under a 56-day median window
    disappears from day 28 onward, exactly when a clearance event matters most).
    Max encodes "the highest price recently charged", which is what a shopper
    would call the regular price.

    Known false positive, stated rather than hidden: a PERMANENT price cut looks
    like a markdown until it ages out of the window (up to `window` days). That
    is the accepted cost of never missing a genuine sustained promotion.

    Args:
        price      : daily price series (already forward-filled).
        window     : trailing days defining "regular" price (default 8 weeks).
        threshold  : fractional drop below baseline counted as a markdown
                     (default 0.05 = 5%).
        min_periods: days needed before a baseline is computed; earlier days
                     backfill from the first valid baseline.

    Returns an int8 array of 0/1, same length as `price`. A product whose price
    never moves correctly yields all zeros rather than spurious flags.
    """
    s = pd.Series(price, dtype="float64")
    baseline = s.rolling(window, min_periods=min_periods).max().bfill()
    flag = (s < baseline * (1.0 - threshold)) & baseline.notna() & s.notna()
    return flag.to_numpy().astype("int8")


def _snap_for_series(
    cal_feats: pd.DataFrame,
    level: int,
    series_id: str,
) -> pd.Series | None:
    """
    Resolve the state-specific SNAP flag for a series where the state is
    knowable (any level whose keys include state_id or store_id — M5 store ids
    are prefixed with their state, e.g. "CA_1"). Returns None otherwise, rather
    than averaging across states, which would invent a signal that doesn't
    exist for that series.
    """
    _, keys = AGGREGATION_LEVELS[level]
    state = None
    parts = str(series_id).split("--")
    for key, part in zip(keys, parts):
        if key == "state_id":
            state = part
            break
        if key == "store_id":
            state = part.split("_")[0]
            break
    col = f"snap_{state}" if state else None
    if col and col in cal_feats.columns:
        return cal_feats[col].astype("int8")
    return None


# ── Long-format emission ─────────────────────────────────────────────────────

def to_long_frames(
    sales: pd.DataFrame,
    calendar: pd.DataFrame,
    prices: pd.DataFrame,
    level: int,
    max_series: int | None = None,
    min_nonzero_frac: float = 0.0,
    trim_leading_zeros: bool = False,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """
    Produce {series_id: long DataFrame} plus a manifest DataFrame.

    Each long frame has columns: date, sales, [sell_price], [promo_flag],
    [snap] — exactly the shape POST /upload expects.

    Args:
        max_series         : keep only the N highest-volume series (None = all).
        min_nonzero_frac   : drop series whose nonzero-day fraction is below
                             this. Useful at levels 10-12 to exclude series
                             that are almost entirely zeros.
        trim_leading_zeros : drop every observation before a series' first
                             sale. In M5 products are introduced over time, so
                             leading zeros mean "not stocked yet", not "zero
                             demand". Training on them teaches models a
                             non-existent flat-zero regime and depresses every
                             error metric. Strongly recommended at item levels.
    """
    wide, day_cols = aggregate_sales(sales, level)
    cal_feats = build_calendar_features(calendar, level)
    price_agg = aggregate_prices(sales, prices, level)

    # Order days by the calendar, not by column order, so the date mapping is
    # driven by calendar.csv rather than assumed.
    cal_indexed = cal_feats.set_index("d")
    day_cols = [d for d in day_cols if d in cal_indexed.index]
    aligned_cal = cal_indexed.loc[day_cols].reset_index()
    dates = aligned_cal["date"].to_numpy()
    weeks = aligned_cal["wm_yr_wk"].to_numpy()
    events = aligned_cal[EVENT_COL].to_numpy().astype("int8")

    # Index prices by series ONCE. Filtering price_agg inside the per-series
    # loop is a linear scan over a frame that reaches ~8.6M rows at level 12,
    # which makes ingestion quadratic and effectively unusable at that level.
    price_lookup: dict[str, dict] = {}
    if price_agg is not None:
        for sid, sub in price_agg.groupby("series_id", observed=True):
            price_lookup[str(sid)] = dict(zip(sub["wm_yr_wk"], sub["sell_price"]))

    # Rank series by total volume so --max-series keeps the meaningful ones.
    totals = wide[day_cols].sum(axis=1)
    order = np.argsort(-totals.to_numpy())
    wide = wide.iloc[order].reset_index(drop=True)

    frames: dict[str, pd.DataFrame] = {}
    manifest_rows: list[dict] = []

    for _, row in wide.iterrows():
        series_id = str(row["series_id"])
        values = row[day_cols].to_numpy(dtype="float64")

        start = 0
        if trim_leading_zeros:
            nonzero = np.flatnonzero(values)
            if nonzero.size == 0:
                continue                       # never sold — nothing to forecast
            start = int(nonzero[0])

        values = values[start:]
        nonzero_frac = float((values != 0).mean())
        if nonzero_frac < min_nonzero_frac:
            continue

        frame = pd.DataFrame({DATE_COL: dates[start:], TARGET_COL: values})

        wk_to_price = price_lookup.get(series_id)
        if wk_to_price:
            mapped = pd.Series(weeks[start:]).map(wk_to_price)
            # Only emit price when it's actually populated; a mostly-empty exog
            # column is worse than no exog column. In M5 a product has no price
            # row for weeks it wasn't stocked, so this legitimately drops price
            # for intermittently-listed items.
            if mapped.notna().mean() >= 0.8:
                price_values = mapped.ffill().bfill().to_numpy()
                frame[PRICE_COL] = price_values
                # The promo proxy is derived from price, so it only exists where
                # price does. Emitting a promo column without price would be
                # asserting a signal that was never measured.
                frame[PROMO_COL] = compute_markdown_promo_flag(price_values)

        frame[EVENT_COL] = events[start:]

        snap = _snap_for_series(aligned_cal, level, series_id)
        if snap is not None:
            frame["snap"] = snap.to_numpy()[start:]

        frames[series_id] = frame
        manifest_rows.append({
            "series_id": series_id,
            "level": level,
            "level_name": AGGREGATION_LEVELS[level][0],
            "n_obs": len(frame),
            "trimmed_leading": start,
            "total_sales": float(values.sum()),
            "mean": float(values.mean()),
            "std": float(values.std()),
            "cv": float(values.std() / values.mean()) if values.mean() > 1e-9 else np.nan,
            "nonzero_frac": nonzero_frac,
            "has_price": PRICE_COL in frame.columns,
            "has_promo": PROMO_COL in frame.columns,
            # Share of days flagged as a markdown — a sanity check on the promo
            # proxy. A value near 0 means prices never moved; near 1 means the
            # threshold is mis-tuned for this series.
            "promo_rate": (float(frame[PROMO_COL].mean())
                           if PROMO_COL in frame.columns else np.nan),
        })

        if max_series is not None and len(frames) >= max_series:
            break

    manifest = pd.DataFrame(manifest_rows)
    return frames, manifest


# ── CLI ──────────────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Reshape M5 into upload-ready long-format CSVs at a chosen aggregation level.",
    )
    parser.add_argument("--m5-dir", type=Path, required=True,
                        help="Directory containing the three M5 CSVs.")
    parser.add_argument("--out-dir", type=Path, required=True,
                        help="Directory to write per-series CSVs and the manifest into.")
    parser.add_argument("--level", type=int, default=9,
                        choices=sorted(AGGREGATION_LEVELS),
                        help="M5 aggregation level 1-12 (default 9 = store x department).")
    parser.add_argument("--max-series", type=int, default=None,
                        help="Emit only the N highest-volume series.")
    parser.add_argument("--min-nonzero-frac", type=float, default=0.0,
                        help="Drop series with a lower fraction of nonzero days (e.g. 0.5).")
    parser.add_argument("--trim-leading-zeros", action="store_true",
                        help="Drop observations before a series' first sale. M5 products are "
                             "introduced over time, so leading zeros mean 'not stocked yet', "
                             "not zero demand. Recommended at levels 10-12.")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(message)s",
    )

    sales, calendar, prices = load_m5(args.m5_dir)
    frames, manifest = to_long_frames(
        sales, calendar, prices,
        level=args.level,
        max_series=args.max_series,
        min_nonzero_frac=args.min_nonzero_frac,
        trim_leading_zeros=args.trim_leading_zeros,
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"m5_level{args.level}"
    for series_id, frame in frames.items():
        frame.to_csv(args.out_dir / f"{prefix}_{_safe_filename(series_id)}.csv", index=False)

    manifest_path = args.out_dir / f"{prefix}_manifest.csv"
    manifest.to_csv(manifest_path, index=False)

    print(f"Wrote {len(frames)} series to {args.out_dir}/ (level {args.level} = "
          f"{AGGREGATION_LEVELS[args.level][0]})")
    if not manifest.empty:
        print(f"  observations per series : {int(manifest['n_obs'].iloc[0])}")
        print(f"  median nonzero fraction : {manifest['nonzero_frac'].median():.3f}")
        print(f"  median CV (std/mean)    : {manifest['cv'].median():.3f}")
        print(f"  series with price exog  : {int(manifest['has_price'].sum())}/{len(manifest)}")
    print(f"  manifest                : {manifest_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
