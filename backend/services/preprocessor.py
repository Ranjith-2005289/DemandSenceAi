"""
preprocessor.py — Universal time series preprocessor
Handles any real-world dataset: retail, finance, energy, IoT, healthcare, etc.

Pipeline:
  1. Parse & validate date column
  2. Parse & validate target column
  3. Set date as index, sort chronologically
  4. Remove duplicate timestamps (take mean)
  5. Detect & fill gaps in the time index (missing dates)
  6. Fill missing values (forward fill → backward fill → interpolation)
  7. Detect & cap outliers (IQR method, configurable sensitivity)
  8. Final validation & quality report
"""

import logging
import re
import warnings
import numpy as np
import pandas as pd
from typing import Optional

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)


# ── Date format patterns ──────────────────────────────────────────────────────
# Tried in order when pandas infer_datetime_format fails
DATE_FORMATS = [
    "%Y-%m-%d", "%Y/%m/%d", "%Y.%m.%d",
    "%d-%m-%Y", "%d/%m/%Y", "%d.%m.%Y",
    "%m-%d-%Y", "%m/%d/%Y", "%m.%d.%Y",
    "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M:%SZ",
    "%Y-%m-%d %H:%M",    "%d-%m-%Y %H:%M:%S",
    "%m/%d/%Y %H:%M:%S", "%m/%d/%Y %I:%M:%S %p",
    "%d %b %Y", "%d %B %Y", "%b %d, %Y", "%B %d, %Y",
    "%d-%b-%Y", "%d-%B-%Y",
    "%d/%m/%y", "%m/%d/%y", "%y-%m-%d",
    "%Y-%m", "%Y/%m", "%b-%Y", "%b %Y",
    "%Y-Q%q",                               # e.g. 2023-Q1 (handled separately)
]

# IQR multiplier → how aggressive outlier capping is
# 1.5 = standard (aggressive), 3.0 = conservative, 5.0 = very lenient
IQR_MULTIPLIER = 2.5

# Exogenous feature classification, by column-name keyword — determines both
# how duplicate timestamps are aggregated and how *future* values (for the
# forecast horizon, which hasn't happened yet) are filled in:
#   - "promo"-type: binary/flag-like → future = 0 (assume no promotion,
#     rather than guessing one continues or recurs)
#   - "price"-type (default for anything else): continuous & usually stable →
#     future = last known value carried forward
PROMO_KEYWORDS = ("promo", "promotion", "discount", "markdown", "on_sale", "onsale", "sale_flag")
PRICE_KEYWORDS = ("price", "cost", "msrp")


def classify_exog_column(col_name: str) -> str:
    """Returns 'promo' or 'price' — see PROMO_KEYWORDS/PRICE_KEYWORDS above."""
    lower = col_name.lower()
    if any(kw in lower for kw in PROMO_KEYWORDS):
        return "promo"
    return "price"


def build_future_exog(
    exog_df: pd.DataFrame,
    exog_meta: dict,
    future_index: pd.Index,
) -> pd.DataFrame:
    """
    Extend historical exog values into the forecast horizon — used identically
    by every model that accepts real exogenous features (XGBoost, RandomForest,
    SARIMAX), since none of them can know the *actual* future price/promotion.
    Promo-type columns assume no promotion (0) rather than guessing one
    continues or recurs; price-type columns carry the last known value
    forward (reasonable — prices are usually stable over a short horizon).
    """
    future = pd.DataFrame(index=future_index)
    for col in exog_df.columns:
        if col in exog_meta.get("promo_cols", []):
            future[col] = 0.0
        else:
            future[col] = float(exog_df[col].iloc[-1])
    return future


# ── Helpers ───────────────────────────────────────────────────────────────────

# "03/04/2023", "3-4-23", "03.04.2023 08:15" — day and month are both numbers,
# so the order is ambiguous whenever both are ≤ 12.
_NUMERIC_DMY = re.compile(r"^\s*(\d{1,2})([/.\-])(\d{1,2})\2(\d{4}|\d{2})(?=$|[\sT])(.*)$")


def _grid_coverage(dates: pd.Series) -> float:
    """
    How completely the parsed dates fill a regular calendar: unique dates
    divided by the slots a regular grid at the median spacing would have.
    A correct reading of a monthly series scores ~1.0; reading day-first
    monthly dates month-first piles them into a few days of January
    (median gap 1 day over a 3-year span) and scores ~0.03.
    """
    u = pd.Series(dates.dropna().unique()).sort_values()
    if len(u) < 3:
        return 0.0
    gaps = u.diff().dropna()
    median_gap = gaps.median()
    if median_gap <= pd.Timedelta(0):
        return 0.0
    slots = (u.iloc[-1] - u.iloc[0]) / median_gap + 1
    return float(len(u) / slots)


def _resolve_numeric_date_order(series: pd.Series):
    """
    For purely numeric dates (dd/mm/yyyy vs mm/dd/yyyy), parse the WHOLE
    column under both readings and keep the one that is valid and forms a
    regular calendar. Element-wise parsers swap order per value when the day
    exceeds 12, mixing both readings in one column; this never does.

    Returns (parsed_series, order, ambiguous) or None if the column is not
    in that numeric form.
    """
    s = series.dropna().astype(str)
    if s.empty:
        return None
    m = s.str.extract(_NUMERIC_DMY)
    if m[0].notna().mean() < 0.8:
        return None

    full = series.astype(str).str.extract(_NUMERIC_DMY)
    a, b = pd.to_numeric(full[0], errors="coerce"), pd.to_numeric(full[2], errors="coerce")
    year = pd.to_numeric(full[3], errors="coerce")
    year = year.where(full[3].str.len() == 4, np.where(year < 69, 2000 + year, 1900 + year))
    rest = full[4].fillna("").str.strip().str.lstrip("T")

    def build(day, month):
        iso = (year.astype("Int64").astype(str) + "-" + month.astype("Int64").astype(str).str.zfill(2)
               + "-" + day.astype("Int64").astype(str).str.zfill(2) + " " + rest).str.strip()
        return pd.to_datetime(iso.where(full[0].notna()), errors="coerce", format="mixed")

    month_first, day_first = build(b, a), build(a, b)
    ok_mf, ok_df = month_first.notna().mean(), day_first.notna().mean()

    if ok_mf >= 0.8 and ok_df < 0.8:
        return month_first, "month-first", False
    if ok_df >= 0.8 and ok_mf < 0.8:
        return day_first, "day-first", False
    if ok_mf < 0.8 and ok_df < 0.8:
        return None
    if month_first.equals(day_first):
        return month_first, "month-first", False
    # Both readings are valid (every day ≤ 12): pick the one that forms a regular calendar.
    cov_mf, cov_df = _grid_coverage(month_first), _grid_coverage(day_first)
    if cov_df > cov_mf * 1.5:
        return day_first, "day-first", True
    if cov_mf > cov_df * 1.5:
        return month_first, "month-first", True
    return month_first, "month-first", True


def _parse_dates(series: pd.Series) -> pd.Series:
    """
    Try every date parsing strategy until one succeeds.
    Handles ISO, European, American, named-month, unix timestamp, and quarterly formats.

    For numeric day/month dates the chosen order is recorded in
    `result.attrs["date_order"]` (and `attrs["date_order_ambiguous"]`).
    """

    # 0. Already datetime
    if pd.api.types.is_datetime64_any_dtype(series):
        return series

    # 0b. Numeric day/month dates: decide the order for the whole column.
    if not pd.api.types.is_numeric_dtype(series):
        resolved = _resolve_numeric_date_order(series)
        if resolved is not None:
            parsed, order, ambiguous = resolved
            parsed.attrs["date_order"] = order
            parsed.attrs["date_order_ambiguous"] = ambiguous
            return parsed

    # 1. Handle quarterly strings like "2023-Q1", "Q1 2023"
    quarterly_pattern = series.astype(str).str.strip().str.match(
        r"^\d{4}[-\s]?Q[1-4]$|^Q[1-4][-\s]?\d{4}$", case=False
    )
    if quarterly_pattern.mean() > 0.7:
        def _parse_quarter(s: str) -> pd.Timestamp:
            s = s.strip().upper()
            if s.startswith("Q"):
                q, yr = s[1], s[-4:]
            else:
                yr, q = s[:4], s[-1]
            month = (int(q) - 1) * 3 + 1
            return pd.Timestamp(year=int(yr), month=month, day=1)
        try:
            return series.apply(_parse_quarter)
        except Exception:
            pass

    # 2. Unix timestamps (integer or large float)
    # Require the raw values to be in a plausible modern-era range before even
    # attempting this — otherwise small numeric columns (temperature, price,
    # sales figures...) all convert to "valid" dates near 1970-01-01 and
    # false-positive here (this is what silently turned a Weekly_Sales column
    # into a fabricated single-day series in an earlier real-world run).
    if pd.api.types.is_numeric_dtype(series):
        if series.between(946_684_800, 4_102_444_800).mean() > 0.8:      # 2000-01-01 .. 2100-01-01, seconds
            try:
                parsed = pd.to_datetime(series, unit="s", errors="coerce")
                if parsed.notna().mean() > 0.8 and parsed.dt.year.between(2000, 2100).mean() > 0.8:
                    return parsed
            except Exception:
                pass
        if series.between(946_684_800_000, 4_102_444_800_000).mean() > 0.8:  # same range, milliseconds
            try:
                parsed = pd.to_datetime(series, unit="ms", errors="coerce")
                if parsed.notna().mean() > 0.8 and parsed.dt.year.between(2000, 2100).mean() > 0.8:
                    return parsed
            except Exception:
                pass

    # 3. Pandas infer (fastest for standard formats). NOTE: no
    # `infer_datetime_format` argument — pandas infers format automatically
    # by default, and that parameter was removed entirely in pandas 3.0
    # (passing it raises TypeError, silently swallowed by the except below,
    # quietly disabling this step for every dataset until fixed).
    try:
        parsed = pd.to_datetime(series, errors="coerce")
        if parsed.notna().mean() > 0.85:
            return parsed
    except Exception:
        pass

    # 4. Explicit format cycling
    for fmt in DATE_FORMATS:
        try:
            parsed = pd.to_datetime(series, format=fmt, errors="coerce")
            if parsed.notna().mean() > 0.80:
                logger.info(f"Date parsed with format: {fmt}")
                return parsed
        except Exception:
            continue

    # 5. Last resort: mixed format (slowest but handles truly messy data)
    # NOTE: the correct pandas API is format="mixed" (a string value) — the
    # previous `mixed=True` was never valid pandas syntax at all (TypeError,
    # silently swallowed), regardless of pandas version.
    try:
        parsed = pd.to_datetime(series, format="mixed", dayfirst=False, errors="coerce")
        if parsed.notna().mean() > 0.6:
            return parsed
    except Exception:
        pass

    # 6. dayfirst=True variant (European date order)
    try:
        parsed = pd.to_datetime(series, dayfirst=True, errors="coerce")
        if parsed.notna().mean() > 0.6:
            return parsed
    except Exception:
        pass

    raise ValueError(
        f"Could not parse column as dates. Sample values: {series.dropna().head(5).tolist()}"
    )


def _uses_decimal_comma(s: pd.Series) -> bool:
    """
    Decide the separator convention from the whole column, not per value.

    Unambiguous evidence:  "1,234.56" or "12.5"  → decimal point (US/UK)
                           "1.234,56" or "12,5"  → decimal comma (European)
    Ambiguous on its own:  "1,234" is read as thousands (the usual export
    format), "1.234" as European thousands (the previous behaviour).
    The old rule matched "1,234" as European "1.234", shrinking US-formatted
    values 1000×.
    """
    s = s.str.replace(r"\s", "", regex=True)
    point = (
        s.str.contains(r"\d,\d{3}\.\d", regex=True)
        | s.str.match(r"^-?\d+\.\d{1,2}$")
        | s.str.match(r"^-?0\.\d+$")
    ).sum()
    comma = (s.str.contains(r"\d\.\d{3},\d", regex=True) | s.str.match(r"^-?\d+,\d{1,2}$")).sum()
    if comma != point:
        return comma > point
    if s.str.match(r"^-?\d{1,3}(,\d{3})+$").any():
        return False
    return bool(s.str.match(r"^-?[1-9]\d{0,2}(\.\d{3})+$").mean() > 0.5)


def _parse_target(series: pd.Series, col_name: str) -> pd.Series:
    """
    Coerce the target column to float. Handles:
    - Currency strings: "$1,234.56" → 1234.56
    - Percentage strings: "12.5%" → 12.5
    - European decimals: "1.234,56" → 1234.56
    - Commas as thousands separator
    """
    if pd.api.types.is_numeric_dtype(series):
        return series.astype(float)

    s = series.astype(str).str.strip()

    # Remove currency symbols and percent signs
    s = s.str.replace(r"[$€£¥₹]", "", regex=True)
    s = s.str.replace(r"%", "", regex=True)

    if _uses_decimal_comma(s):
        # European format: "1.234,56" → "1234.56"
        s = s.str.replace(".", "", regex=False).str.replace(",", ".", regex=False)
    else:
        # Remove thousands commas: "1,234,567" → "1234567"
        s = s.str.replace(",", "", regex=False)

    try:
        converted = pd.to_numeric(s, errors="coerce")
        if converted.notna().mean() < 0.5:
            raise ValueError(
                f"Less than 50% of values in '{col_name}' could be parsed as numbers. "
                f"Sample: {series.dropna().head(5).tolist()}"
            )
        return converted
    except Exception as e:
        raise ValueError(f"Cannot convert target column '{col_name}' to numeric: {e}")


def _infer_frequency(index: pd.DatetimeIndex) -> Optional[str]:
    """
    Infer the dominant time frequency from a DatetimeIndex.
    Returns pandas offset string (e.g. 'D', 'W', 'MS', 'QS', 'H').
    """
    if len(index) < 3:
        return None

    # Try pandas built-in first
    try:
        freq = pd.infer_freq(index)
        if freq:
            return freq
    except Exception:
        pass

    # Estimate from median gap (irregular data). Aliases are pandas 3 ones:
    # "H"/"T" no longer exist, and passing them made gap filling fail silently.
    u = pd.Series(index.unique()).sort_values()
    seconds = u.diff().dropna().median().total_seconds()
    intraday = bool((u.dt.normalize() != u).any())
    if seconds <= 90:
        return "min"
    if seconds <= 12 * 3600 and intraday:
        return "h"
    if seconds <= 86400 * 1.5:
        span_days = (u.iloc[-1] - u.iloc[0]).days
        if span_days >= 28 and not (u.dt.dayofweek >= 5).any():
            return "B"   # trading/business days: don't invent weekends
        return "D"
    if seconds <= 86400 * 8:
        # Anchor weeks on the weekday the data actually uses.
        day = ["MON", "TUE", "WED", "THU", "FRI", "SAT", "SUN"][int(u.dt.dayofweek.mode().iloc[0])]
        return f"W-{day}"
    if seconds <= 86400 * 35:
        return "ME" if u.dt.is_month_end.mean() > 0.5 else "MS"
    if seconds <= 86400 * 100:
        return "QE" if u.dt.is_quarter_end.mean() > 0.5 else "QS"
    return "YE" if u.dt.is_year_end.mean() > 0.5 else "YS"


def _fill_date_gaps(series: pd.Series, freq: str, how: str = "sum") -> tuple[pd.Series, pd.Index, bool]:
    """
    Put the series on a complete, regular grid at `freq`.

    If every timestamp already sits on the grid, reindex (new periods are NaN).
    If some do not (irregular intraday timestamps, weeks dated on another
    weekday, month-end vs month-start), aggregate into grid periods with `how`
    ("sum" for flows, "mean" for levels). Reindexing off-grid data used to
    drop every point not on the grid: 740 hourly readings became 26.

    Returns (series, empty_periods, resampled).
    """
    try:
        full_index = pd.date_range(start=series.index.min(), end=series.index.max(), freq=freq)
        if len(full_index) and series.index.isin(full_index).all():
            out = series.reindex(full_index)
            return out, full_index.difference(series.index), False
        grouped = series.resample(freq)
        out = grouped.sum(min_count=1) if how == "sum" else grouped.mean()
        counts = grouped.count()
        return out, out.index[counts.values == 0], True
    except Exception as e:
        logger.warning(f"Could not regularise the time index (freq={freq}): {e}")
        return series, pd.Index([]), False


# Name words that mark a LEVEL (a reading at a point in time: average it,
# never zero-fill it) rather than a FLOW (an amount during the period: sum
# it, and zero when nothing was recorded). Everything else is a flow.
LEVEL_KEYWORDS = {
    "price", "rate", "temp", "temperature", "humidity", "pressure", "close", "open",
    "high", "low", "level", "balance", "index", "score", "percent", "pct", "ratio",
    "avg", "average", "mean", "stock", "inventory", "onhand", "occupancy", "speed",
    "yield", "margin", "nav", "utilization", "utilisation",
}


def name_tokens(col: str) -> list[str]:
    """'Paid_Amount' → ['paid', 'amount']; 'StoreID' → ['store', 'id']; 'unitsSold2' → ['units', 'sold', '2']."""
    return [t.lower() for t in re.findall(r"[A-Z]+(?![a-z])|[A-Z]?[a-z]+|\d+", str(col))]


def metric_kind(col: str) -> str:
    """'level' for readings (price, temperature, stock level…), else 'flow'."""
    tokens = name_tokens(col)
    if "on" in tokens and "hand" in tokens:
        return "level"
    return "level" if any(t in LEVEL_KEYWORDS for t in tokens) else "flow"


def _extreme_spikes(series: pd.Series) -> pd.Series:
    """
    Isolated values so far from their neighbours that they are almost
    certainly data errors (an extra zero, a unit slip): more than 8x the
    median of the surrounding 15 periods AND beyond 10 IQRs. Recurring
    peaks (December, weekends) are not flagged because their neighbours are
    high too; intermittent series (neighbour median 0) are left alone.
    """
    s = series.dropna()
    if len(s) < 15:
        return pd.Series(False, index=series.index)
    q1, q3 = s.quantile([0.25, 0.75])
    iqr = q3 - q1
    if iqr <= 0:
        return pd.Series(False, index=series.index)
    neigh = series.rolling(15, center=True, min_periods=5).median()
    dev = (series - neigh).abs()
    mask = (dev > 10 * iqr) & (neigh.abs() > 0) & (dev > 7 * neigh.abs())
    return mask.fillna(False)


def _outlying_dates(index: pd.DatetimeIndex) -> np.ndarray:
    """
    Rows whose date sits far outside the bulk of the data: a small cluster
    of dates at either end separated from the rest by a gap of more than
    50x the typical spacing (and at least 60 days). Typically a typo such as
    2032 for 2023, or a 1970 date from a bad epoch value. Left in, gap
    filling would invent every period in between (3,124 days for one typo).
    Returns a boolean mask over `index` (True = drop).
    """
    drop = np.zeros(len(index), dtype=bool)
    u = pd.Series(index.unique()).sort_values().reset_index(drop=True)
    if len(u) < 10:
        return drop
    gaps = u.diff()
    typical = gaps.median()
    threshold = max(typical * 50, pd.Timedelta(days=60))
    limit = max(3, int(0.01 * len(u)))
    lo, hi = 0, len(u) - 1
    for _ in range(3):  # peel isolated clusters from both ends
        tail_breaks = [i for i in range(hi, lo, -1) if gaps.iloc[i] > threshold and hi - i + 1 <= limit]
        if tail_breaks:
            hi = tail_breaks[-1] - 1
        head_breaks = [i for i in range(lo + 1, hi + 1) if gaps.iloc[i] > threshold and i - lo <= limit]
        if head_breaks:
            lo = head_breaks[-1]
        if not tail_breaks and not head_breaks:
            break
    keep_from, keep_to = u.iloc[lo], u.iloc[hi]
    return np.asarray((index < keep_from) | (index > keep_to))


def _cap_outliers_iqr(
    series: pd.Series,
    multiplier: float = IQR_MULTIPLIER,
) -> tuple[pd.Series, dict]:
    """
    Detect outliers with IQR method and cap (Winsorize) them.
    Does NOT drop — capping preserves the time structure.

    Returns:
        (capped_series, outlier_report)
    """
    Q1  = series.quantile(0.25)
    Q3  = series.quantile(0.75)
    IQR = Q3 - Q1

    # Degenerate case: all values identical (IQR = 0)
    if IQR == 0:
        return series, {"lower_bound": float(Q1), "upper_bound": float(Q3), "n_outliers": 0}

    lower = Q1 - multiplier * IQR
    upper = Q3 + multiplier * IQR

    # For sales/demand: never cap below 0 if data is non-negative
    if series.min() >= 0:
        lower = max(0.0, lower)

    n_below = int((series < lower).sum())
    n_above = int((series > upper).sum())
    n_outliers = n_below + n_above

    if n_outliers > 0:
        logger.info(
            f"Outlier capping (IQR×{multiplier}): "
            f"{n_below} low, {n_above} high → capped to [{lower:.2f}, {upper:.2f}]"
        )

    capped = series.clip(lower=lower, upper=upper)

    return capped, {
        "lower_bound":  round(float(lower),  4),
        "upper_bound":  round(float(upper),  4),
        "n_outliers":   n_outliers,
        "n_below":      n_below,
        "n_above":      n_above,
        "iqr":          round(float(IQR), 4),
    }


# ── Main pipeline ──────────────────────────────────────────────────────────────

def preprocess(
    df: pd.DataFrame,
    date_col: str,
    target_col: str,
    iqr_multiplier: float = IQR_MULTIPLIER,
    cap_outliers: bool = False,
    fill_gaps: bool = True,
    feature_cols: Optional[list[str]] = None,
) -> tuple[pd.Series, dict]:
    """
    Full preprocessing pipeline for any time series dataset.

    Args:
        df             : Raw DataFrame from upload.
        date_col       : Name of the column containing date/timestamp values.
        target_col     : Name of the numeric column to forecast.
        iqr_multiplier : Sensitivity of outlier capping (default 2.5).
                         Lower = more aggressive, Higher = more lenient.
        cap_outliers   : Global IQR capping. Off by default: a single global
                         bound cannot tell a data error from a real seasonal
                         peak and erased December peaks in testing (see
                         DATA_HEALTH_DESIGN.md §2). Kept for benchmarks that
                         reproduce the earlier behaviour.
        fill_gaps      : Set False to skip gap-filling (already complete series).
        feature_cols   : Optional extra numeric columns (e.g. price, a promo
                         flag) to align to the same index as the target series,
                         for models that can use real exogenous features. Not
                         JSON-serialized anywhere — only ever read back out of
                         the returned report by backend code, never sent to a
                         client, so it's safe for report["exog_df"] to hold an
                         actual DataFrame.

    Returns:
        (series, quality_report)

        series        : Clean pd.Series with DatetimeIndex, ready for models.
        quality_report: Dict summarising every transformation applied. When
                        feature_cols is given, also has report["exog_df"]
                        (pd.DataFrame aligned to series.index, or None if none
                        of feature_cols were usable) and report["exog_meta"]
                        ({"price_cols": [...], "promo_cols": [...]}).
    """

    report: dict = {
        "original_rows":       len(df),
        "original_columns":    list(df.columns),
        "date_col":            date_col,
        "target_col":          target_col,
        "steps":               [],
        "warnings":            [],
        "errors":              [],
        "exog_df":             None,
        "exog_meta":           {"price_cols": [], "promo_cols": []},
    }

    # ── 0. Column validation ──────────────────────────────────────────────────
    if date_col not in df.columns:
        raise ValueError(
            f"Date column '{date_col}' not found in dataframe. "
            f"Available columns: {list(df.columns)}"
        )
    if target_col not in df.columns:
        raise ValueError(
            f"Target column '{target_col}' not found in dataframe. "
            f"Available columns: {list(df.columns)}"
        )
    if date_col == target_col:
        raise ValueError(
            f"Date column and target column can't both be '{date_col}'. "
            "Please choose two different columns."
        )

    exog_cols = [
        c for c in (feature_cols or [])
        if c in df.columns and c not in (date_col, target_col)
    ]

    # Work on a copy — never mutate the original
    df = df[[date_col, target_col] + exog_cols].copy()

    # ── 1. Parse date column ──────────────────────────────────────────────────
    try:
        parsed_dates = _parse_dates(df[date_col])
        date_order = parsed_dates.attrs.get("date_order")
        if date_order:
            report["date_order"] = date_order
            if parsed_dates.attrs.get("date_order_ambiguous"):
                sample = str(df[date_col].dropna().iloc[0])
                report["warnings"].append(
                    f"Dates like '{sample}' could be read day-first or month-first; "
                    f"read as {date_order} because that gives a regular calendar."
                )
        df[date_col] = parsed_dates
        n_unparsed = df[date_col].isna().sum()
        if n_unparsed > 0:
            report["warnings"].append(
                f"{n_unparsed} date values could not be parsed and will be dropped."
            )
            df = df.dropna(subset=[date_col])
        report["steps"].append(f"[1] Date column '{date_col}' parsed successfully.")
    except Exception as e:
        report["errors"].append(str(e))
        raise ValueError(f"Date parsing failed: {e}")

    # ── 2. Parse target column ────────────────────────────────────────────────
    try:
        df[target_col] = _parse_target(df[target_col], target_col)
        n_non_numeric = df[target_col].isna().sum()
        if n_non_numeric > 0:
            report["warnings"].append(
                f"{n_non_numeric} non-numeric values in target column will be treated as missing."
            )
        report["steps"].append(
            f"[2] Target column '{target_col}' parsed as float "
            f"(range: [{df[target_col].min():.2f}, {df[target_col].max():.2f}])."
        )
    except Exception as e:
        report["errors"].append(str(e))
        raise ValueError(f"Target column parsing failed: {e}")

    # ── 2b. Coerce exog columns to numeric ────────────────────────────────────
    for col in list(exog_cols):
        coerced = pd.to_numeric(df[col], errors="coerce")
        if coerced.notna().mean() < 0.5:
            # Not usable as a numeric exogenous feature — drop it rather than
            # feed models a mostly-NaN column.
            exog_cols.remove(col)
            df = df.drop(columns=[col])
            report["warnings"].append(
                f"Feature column '{col}' was mostly non-numeric and was dropped."
            )
        else:
            df[col] = coerced

    # ── 3. Set date as index & sort ───────────────────────────────────────────
    df = df.set_index(date_col)
    df = df.sort_index()

    # Remove timezone info (models don't need it; it causes comparison issues)
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)

    outlying = _outlying_dates(df.index)
    if outlying.any():
        dropped = sorted({str(d.date()) for d in df.index[outlying]})
        df = df[~outlying]
        msg = (
            f"{int(outlying.sum())} row(s) dated {', '.join(dropped[:3])}"
            f"{' …' if len(dropped) > 3 else ''} were excluded: they lie far outside the rest of the "
            f"data ({str(df.index.min().date())} – {str(df.index.max().date())}), separated by a long gap; "
            f"they look like test entries or typos."
        )
        report["warnings"].append(msg)
        report["steps"].append(f"[3b] {msg}")

    # Snapshot exog columns now — deduped/aligned separately from the target
    # below, since "sum duplicate timestamps" (right for demand) is the wrong
    # aggregation for a price or promo-flag column: promo-type columns take
    # the max per timestamp (any promo that day = promo that day), price-type
    # columns take the mean.
    exog_raw = None
    if exog_cols:
        exog_raw = df[exog_cols].copy()
        if exog_raw.index.duplicated().any():
            agg = {c: ("max" if classify_exog_column(c) == "promo" else "mean") for c in exog_cols}
            exog_raw = exog_raw.groupby(level=0).agg(agg)

    report["date_range"] = {
        "start": str(df.index.min().date()),
        "end":   str(df.index.max().date()),
        "span_days": (df.index.max() - df.index.min()).days,
    }
    report["steps"].append(
        f"[3] Index set to '{date_col}', sorted chronologically "
        f"({report['date_range']['start']} → {report['date_range']['end']})."
    )

    # ── 4. Remove duplicate timestamps (smart handling) ───────────────────────
    n_before = len(df)
    aggregated_by_sum = False
    kind = metric_kind(target_col)
    report["metric_kind"] = kind
    if df.index.duplicated().any() and kind == "level":
        # Several readings per period of a level (price, temperature, stock
        # on hand): their total means nothing, their average does.
        n_dupes = int(df.index.duplicated().sum())
        df = df.groupby(level=0).mean(numeric_only=True).sort_index()
        report["warnings"].append(
            f"'{target_col}' looks like a reading (not an amount), so rows sharing a date "
            f"({n_dupes} extra) were averaged, not added up."
        )
        report["steps"].append(f"[4] {n_before} readings averaged into {len(df)} periods (level metric).")
    elif df.index.duplicated().any():
        n_dupes = int(df.index.duplicated().sum())
        n_unique = int(df.index.nunique())
        unique_idx = df.index.unique().sort_values()
        sub_daily = len(unique_idx) > 2 and pd.Series(unique_idx).diff().median() < pd.Timedelta(days=1)
        dup_ts = df.index[df.index.duplicated()]
        years = max(1, int((unique_idx[-1] - unique_idx[0]).days // 365) + 1)
        repeated_readings = (
            sub_daily
            and n_dupes <= years * 2
            and int(df.index.value_counts().max()) == 2
            # Clock changes happen at night in March/April and October/November.
            and bool(((dup_ts.hour <= 3) & dup_ts.month.isin([3, 4, 10, 11])).all())
        )
        if repeated_readings:
            # Sub-daily data with one row per timestamp except a rare repeat:
            # a daylight-saving clock change or a re-sent reading, not two
            # separate sales. Summing would create a false 2x spike.
            df = df.groupby(level=0).mean(numeric_only=True)
            report["warnings"].append(
                f"{n_dupes} timestamp(s) appeared twice (for example a daylight-saving clock change); "
                "their values were averaged."
            )
            report["steps"].append(
                f"[4] {n_dupes} repeated timestamps averaged ({n_before} → {len(df)} rows)."
            )
        else:
            # Many rows per period: individual records (transactions, stores,
            # products) — total them. min_count=1 keeps a period whose values
            # are all blank as missing instead of turning it into 0.
            df = df.groupby(level=0).sum(numeric_only=True, min_count=1)
            aggregated_by_sum = True
            report["steps"].append(
                f"[4] {n_before} rows combined into {len(df)} period totals "
                f"(several rows per date were summed)."
            )
        df = df.sort_index()
    else:
        report["steps"].append("[4] No duplicate timestamps found.")

    # Work as a Series from here for simplicity
    series: pd.Series = df[target_col].copy()

    # ── 5. Infer frequency & fill date gaps ───────────────────────────────────
    freq = _infer_frequency(series.index)
    report["inferred_frequency"] = freq or "unknown"

    if fill_gaps and freq:
        n_rows_before = len(series)
        series, empty, resampled = _fill_date_gaps(series, freq, "mean" if kind == "level" else "sum")
        if resampled:
            aggregated_by_sum = aggregated_by_sum or kind == "flow"
            msg = (
                f"Timestamps did not fall on a regular {freq} grid; {n_rows_before} rows were "
                f"{'averaged' if kind == 'level' else 'totalled'} into {len(series)} periods."
            )
            report["warnings"].append(msg)
            report["steps"].append(f"[5a] {msg}")
        n_gaps_added = len(empty)
        sparse = n_gaps_added >= 0.2 * len(series)
        if n_gaps_added and (aggregated_by_sum or sparse) and kind == "flow" and series.min() >= 0:
            # Rows were individual records summed per period, so a period
            # with no rows had none: zero, not an interpolated guess.
            series.loc[empty] = 0.0
            reason = (
                "rows are individual records totalled per period"
                if aggregated_by_sum
                else f"{n_gaps_added} of {len(series)} periods are empty, so '{target_col}' looks like event data"
            )
            report["warnings"].append(f"{n_gaps_added} period(s) had no records and were counted as 0 ({reason}).")
            report["steps"].append(
                f"[5] {n_gaps_added} periods with no records set to 0 "
                f"(freq={freq}, total rows now: {len(series)})."
            )
        else:
            report["steps"].append(
                f"[5] Date gaps filled — added {n_gaps_added} missing periods "
                f"(freq={freq}, total rows now: {len(series)})."
            )
    else:
        report["steps"].append(
            f"[5] Gap filling skipped (freq={'not inferred' if not freq else freq}, "
            f"fill_gaps={fill_gaps})."
        )

    # ── 5b. Extreme isolated spikes (data errors) → missing ──────────────────
    spikes = _extreme_spikes(series)
    if spikes.any():
        examples = ", ".join(f"{series[d]:,.0f} on {d.date()}" for d in series.index[spikes][:3])
        series[spikes] = np.nan
        msg = (
            f"{int(spikes.sum())} value(s) ({examples}) were far outside the surrounding values "
            "(over 8x) and look like data errors; they were replaced by estimates."
        )
        report["warnings"].append(msg)
        report["steps"].append(f"[5b] {msg}")

    # ── 6. Fill missing values ────────────────────────────────────────────────
    n_missing_before = int(series.isna().sum())

    if n_missing_before > 0:
        # Step 1: Linear interpolation (best for time series — respects trend)
        series = series.interpolate(method="time", limit_direction="both")

        # Step 2: Forward fill for any remaining at boundaries
        series = series.ffill()

        # Step 3: Backward fill for leading NaNs
        series = series.bfill()

        n_missing_after = int(series.isna().sum())

        report["steps"].append(
            f"[6] Missing values: {n_missing_before} before → {n_missing_after} after "
            f"(time interpolation → ffill → bfill)."
        )
        if n_missing_after > 0:
            report["warnings"].append(
                f"{n_missing_after} values could not be filled — these rows will be dropped."
            )
            series = series.dropna()
    else:
        report["steps"].append("[6] No missing values detected.")

    report["missing_values"] = {
        "before": n_missing_before,
        "after":  int(series.isna().sum()),
    }

    # ── 7. Outlier detection & capping (IQR) ─────────────────────────────────
    if cap_outliers:
        series, outlier_info = _cap_outliers_iqr(series, multiplier=iqr_multiplier)
        report["outliers"] = outlier_info
        report["steps"].append(
            f"[7] Outlier capping (IQR×{iqr_multiplier}): "
            f"{outlier_info['n_outliers']} values capped "
            f"to [{outlier_info['lower_bound']}, {outlier_info['upper_bound']}]."
        )
    else:
        report["outliers"] = {"n_outliers": 0}
        report["steps"].append(
            "[7] Outlier capping off: unusual values are kept; models are compared on held-out data instead."
        )

    # ── 8b. Align exog columns to the final series index ─────────────────────
    if exog_raw is not None:
        exog_df = exog_raw.reindex(series.index)
        price_cols, promo_cols = [], []
        for col in exog_cols:
            if classify_exog_column(col) == "promo":
                exog_df[col] = exog_df[col].fillna(0)
                promo_cols.append(col)
            else:
                # Carry the last known value forward through any gaps —
                # matches how the *future* horizon is filled for these
                # columns too (see model-side future-exog construction).
                exog_df[col] = exog_df[col].ffill().bfill().fillna(0)
                price_cols.append(col)
        report["exog_df"] = exog_df
        report["exog_meta"] = {"price_cols": price_cols, "promo_cols": promo_cols}
        report["steps"].append(
            f"[8b] Exogenous features aligned: {len(price_cols)} price-type "
            f"({price_cols}), {len(promo_cols)} promo-type ({promo_cols})."
        )

    # ── 9. Final validation ───────────────────────────────────────────────────
    final_rows = len(series)
    report["final_rows"] = final_rows

    if final_rows < 10:
        msg = (
            f"After preprocessing, only {final_rows} rows remain. "
            "Forecasting requires at least 10 observations."
        )
        report["errors"].append(msg)
        raise ValueError(msg)

    if final_rows < 30:
        report["warnings"].append(
            f"Only {final_rows} rows after preprocessing. "
            "Results may be less reliable with fewer than 30 data points."
        )

    # Descriptive stats for the quality report
    report["final_stats"] = {
        "rows":   final_rows,
        "min":    round(float(series.min()),  4),
        "max":    round(float(series.max()),  4),
        "mean":   round(float(series.mean()), 4),
        "std":    round(float(series.std()),  4),
        "median": round(float(series.median()), 4),
        "null_count": int(series.isna().sum()),
    }

    # Ensure series name is set for downstream use
    series.name = target_col

    logger.info(
        f"Preprocessing complete: {report['original_rows']} → {final_rows} rows | "
        f"freq={freq} | outliers={report['outliers']['n_outliers']} | "
        f"missing={n_missing_before}"
    )

    return series, report


# ── Convenience wrapper (returns only series, for simple use) ─────────────────

def preprocess_simple(
    df: pd.DataFrame,
    date_col: str,
    target_col: str,
) -> pd.Series:
    """
    Thin wrapper — returns only the cleaned series (drops the quality report).
    Use preprocess() directly when you need the report for the API response.
    """
    series, _ = preprocess(df, date_col, target_col)
    return series