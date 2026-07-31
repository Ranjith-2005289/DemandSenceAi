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
import warnings
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

def _parse_dates(series: pd.Series) -> pd.Series:
    """
    Try every date parsing strategy until one succeeds.
    Handles ISO, European, American, named-month, unix timestamp, and quarterly formats.
    """

    # 0. Already datetime
    if pd.api.types.is_datetime64_any_dtype(series):
        return series

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

    # European format: "1.234,56" → "1234.56"
    european_mask = s.str.match(r"^\d{1,3}(\.\d{3})*(,\d+)?$")
    if european_mask.mean() > 0.5:
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

    # Estimate from median gap
    gaps = pd.Series(index).diff().dropna()
    median_gap = gaps.median()

    seconds = median_gap.total_seconds()
    if seconds <= 60:
        return "T"           # minute
    elif seconds <= 3600:
        return "H"           # hourly
    elif seconds <= 86400 * 1.5:
        return "D"           # daily
    elif seconds <= 86400 * 8:
        return "W"           # weekly
    elif seconds <= 86400 * 35:
        return "MS"          # monthly
    elif seconds <= 86400 * 100:
        return "QS"          # quarterly
    else:
        return "YS"          # annual


def _fill_date_gaps(series: pd.Series, freq: str) -> pd.Series:
    """
    Reindex to a complete date range so every period exists.
    Newly created rows have NaN values (filled in the next step).
    """
    try:
        full_index = pd.date_range(
            start=series.index.min(),
            end=series.index.max(),
            freq=freq,
        )
        series = series.reindex(full_index)
        gaps_added = len(series) - len(series.dropna())
        if gaps_added > 0:
            logger.info(f"Filled {gaps_added} missing date gaps (freq={freq})")
    except Exception as e:
        logger.warning(f"Could not reindex to fill date gaps: {e}")

    return series


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
    cap_outliers: bool = True,
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
        cap_outliers   : Set False to skip outlier capping (e.g. financial data).
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
        df[date_col] = _parse_dates(df[date_col])
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
    if df.index.duplicated().any():
        n_dupes = int(df.index.duplicated().sum())
        
        # Strategy: Aggregate duplicate timestamps by summing them
        # This is standard for retail/sales data where we want total daily/weekly demand
        df = df.groupby(level=0).sum(numeric_only=True)
        df = df.sort_index()
        
        report["steps"].append(
            f"[4] Removed {n_dupes} duplicate timestamps (aggregated by sum). "
            f"({n_before} → {len(df)} rows)"
        )
    else:
        report["steps"].append("[4] No duplicate timestamps found.")

    # Work as a Series from here for simplicity
    series: pd.Series = df[target_col].copy()

    # ── 5. Infer frequency & fill date gaps ───────────────────────────────────
    freq = _infer_frequency(series.index)
    report["inferred_frequency"] = freq or "unknown"

    if fill_gaps and freq:
        n_before_gap = len(series)
        series = _fill_date_gaps(series, freq)
        n_gaps_added = len(series) - n_before_gap
        report["steps"].append(
            f"[5] Date gaps filled — added {n_gaps_added} missing periods "
            f"(freq={freq}, total rows now: {len(series)})."
        )
    else:
        report["steps"].append(
            f"[5] Gap filling skipped (freq={'not inferred' if not freq else freq}, "
            f"fill_gaps={fill_gaps})."
        )

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
        report["steps"].append("[7] Outlier capping skipped.")

    # ── 8. Smooth extreme noise (optional rolling median correction) ──────────
    # For very high-frequency data (hourly/minute), a light smooth improves model accuracy
    if freq and freq in ("T", "H") and len(series) > 48:
        window = 3
        smoothed = series.rolling(window=window, center=True, min_periods=1).median()
        # Only smooth points that are more than 2 std from rolling median (true noise)
        deviation = (series - smoothed).abs()
        threshold = deviation.std() * 2
        noise_mask = deviation > threshold
        n_smoothed = int(noise_mask.sum())
        if n_smoothed > 0:
            series[noise_mask] = smoothed[noise_mask]
            report["steps"].append(
                f"[8] High-frequency smoothing: {n_smoothed} noisy points corrected "
                f"(rolling median, window={window})."
            )
        else:
            report["steps"].append("[8] High-frequency smoothing: no noisy points detected.")
    else:
        report["steps"].append("[8] High-frequency smoothing skipped.")

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