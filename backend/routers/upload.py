"""
upload.py — FastAPI router for CSV upload
Handles ANY time series dataset: retail, stock, energy, IoT, healthcare, etc.
"""

import uuid
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from fastapi import APIRouter, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse

from services.preprocessor import PRICE_KEYWORDS, PROMO_KEYWORDS

router = APIRouter()

# ── Storage folder ────────────────────────────────────────────────────────────
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# ── Date format patterns to try (covers almost every real-world format) ───────
DATE_FORMATS = [
    # ISO / standard
    "%Y-%m-%d", "%Y/%m/%d", "%Y.%m.%d",
    "%d-%m-%Y", "%d/%m/%Y", "%d.%m.%Y",
    "%m-%d-%Y", "%m/%d/%Y", "%m.%d.%Y",
    # With time
    "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S",
    "%Y-%m-%d %H:%M",    "%d-%m-%Y %H:%M:%S",
    "%m/%d/%Y %H:%M:%S", "%m/%d/%Y %I:%M:%S %p",
    # Month names
    "%d %b %Y", "%d %B %Y", "%b %d, %Y", "%B %d, %Y",
    "%d-%b-%Y", "%d-%B-%Y",
    # Short year
    "%d/%m/%y", "%m/%d/%y", "%y-%m-%d",
    # Period-style (quarterly, monthly)
    "%Y-%m", "%Y/%m", "%b-%Y", "%b %Y",
    # Unix timestamps handled separately
]

# Columns that are almost certainly not time series targets
NON_TARGET_KEYWORDS = {
    "id", "index", "key", "code", "sku", "upc", "barcode",
    "zip", "postal", "phone", "latitude", "longitude", "lat", "lon",
    "flag", "bool", "active", "enabled",
}

# Columns likely to be grouping / categorical dimensions
GROUPING_KEYWORDS = {
    "product", "item", "sku", "category", "brand", "store",
    "region", "country", "city", "channel", "segment",
    "department", "division", "employee", "customer", "user",
    "symbol", "ticker", "sensor", "device", "station", "meter",
}

FREQ_MAP = {
    "T": "minute-level",
    "H": "hourly",
    "D": "daily",
    "W": "weekly",
    "M": "monthly",
    "Q": "quarterly",
    "A": "annual",
    "Y": "annual",
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _try_parse_date_series(series: pd.Series) -> pd.Series | None:
    """Try to parse a Series as dates. Returns parsed Series or None."""
    # 1. Let pandas infer (fastest path) — but ONLY for non-numeric (string/
    # object) series. NOTE: no `infer_datetime_format` argument — pandas
    # already infers format automatically by default, and that parameter was
    # removed entirely in pandas 3.0 (passing it raises TypeError, which —
    # before this fix — was silently swallowed by the except below, quietly
    # disabling this entire fast path for every dataset).
    #
    # Numeric series are deliberately skipped here and deferred to step 2
    # below: pandas' default `unit="ns"` for bare numeric input means small
    # integers (e.g. sales figures like 34390) parse "successfully" as
    # timestamps within nanoseconds of 1970-01-01 — every value comes back
    # non-null, so the >=80% threshold passes and a column like
    # "Weekly_Sales" gets silently misclassified as a date column before
    # step 2's proper 2000-2100 plausible-range gate ever runs. Deferring
    # numeric series to step 2 (which is range-gated) closes that hole.
    if not pd.api.types.is_numeric_dtype(series):
        try:
            parsed = pd.to_datetime(series, errors="raise")
            if parsed.notna().mean() >= 0.8:          # at least 80 % parsed
                return parsed
        except Exception:
            pass

    # 2. Try unix timestamps (integer columns like 1700000000)
    # Require the raw values themselves to be in a plausible modern-era range
    # (roughly year 2000 - 2100 in seconds-since-epoch) before even attempting
    # this — otherwise small numeric columns (temperature, price, ratings...)
    # all convert to "valid" dates near 1970-01-01 and false-positive here.
    if pd.api.types.is_numeric_dtype(series):
        plausible_seconds = series.between(946_684_800, 4_102_444_800)  # 2000-01-01 .. 2100-01-01
        if plausible_seconds.mean() >= 0.8:
            try:
                parsed = pd.to_datetime(series, unit="s", errors="raise")
                if (parsed.dt.year.between(2000, 2100)).mean() >= 0.8:
                    return parsed
            except Exception:
                pass

    # 3. Cycle through explicit format strings
    for fmt in DATE_FORMATS:
        try:
            parsed = pd.to_datetime(series, format=fmt, errors="raise")
            if parsed.notna().mean() >= 0.8:
                return parsed
        except Exception:
            continue

    # 4. Last resort: coerce (accepts partial matches)
    try:
        parsed = pd.to_datetime(series, errors="coerce")
        if parsed.notna().mean() >= 0.6:
            return parsed
    except Exception:
        pass

    # 5. Truly last resort: mixed formats within the same column
    try:
        parsed = pd.to_datetime(series, format="mixed", errors="coerce")
        if parsed.notna().mean() >= 0.6:
            return parsed
    except Exception:
        pass

    return None


def _detect_date_columns(df: pd.DataFrame) -> dict[str, str]:
    """
    Returns {col_name: sample_format_string} for every column that looks like dates.
    Checks: dtype already datetime, column name hints, and value-level parsing.
    """
    date_cols: dict[str, str] = {}

    for col in df.columns:
        col_lower = col.lower().replace(" ", "_")

        # Already a datetime dtype
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            sample = str(df[col].dropna().iloc[0]) if df[col].notna().any() else "datetime"
            date_cols[col] = sample
            continue

        # Name hint: only attempt parsing if the name looks date-like.
        # ("at" was previously in this list to catch "created_at"/"updated_at",
        # but as a bare substring it also matches unrelated words like
        # "temperature" or "rate" — checked separately below as a suffix instead.)
        name_hints = {"date", "time", "datetime", "timestamp", "period",
                      "week", "month", "year", "day", "quarter",
                      "created", "updated", "recorded"}
        name_looks_like_date = (
            any(hint in col_lower for hint in name_hints)
            or col_lower.endswith("_at")
        )

        # Also try object/string columns regardless of name (catches unlabeled cols)
        if df[col].dtype == object or name_looks_like_date:
            sample_vals = df[col].dropna().head(50)
            if sample_vals.empty:
                continue
            parsed = _try_parse_date_series(sample_vals)
            if parsed is not None:
                date_cols[col] = str(sample_vals.iloc[0])

    return date_cols


def _detect_numeric_columns(df: pd.DataFrame, date_cols: list[str]) -> dict[str, dict]:
    """
    Returns metadata for numeric columns that are plausible forecast targets.
    Excludes date columns, ID-like columns, and pure-integer index columns.
    """
    numeric_meta: dict[str, dict] = {}

    for col in df.select_dtypes(include=[np.number]).columns:
        if col in date_cols:
            continue

        col_lower = col.lower().replace(" ", "_")

        # Skip obvious non-targets
        if any(kw in col_lower for kw in NON_TARGET_KEYWORDS):
            continue

        series = df[col].dropna()
        if series.empty:
            continue

        # Skip columns that look like row indices (0,1,2,3…)
        if series.is_monotonic_increasing and series.nunique() == len(series) and series.min() == 0:
            continue

        is_likely_target = any(kw in col_lower for kw in {
            "sale", "sales", "revenue", "demand", "quantity", "qty",
            "units", "price", "amount", "value", "cost", "profit",
            "stock", "inventory", "volume", "count", "total",
            "close", "open", "high", "low", "adj",       # stock
            "kwh", "mwh", "consumption", "production",   # energy
            "passengers", "trips", "visitors",            # transport
            "calls", "orders", "transactions",            # ops
            "temp", "temperature", "humidity", "pressure" # IoT / weather
        })

        numeric_meta[col] = {
            "min": round(float(series.min()), 4),
            "max": round(float(series.max()), 4),
            "mean": round(float(series.mean()), 4),
            "null_count": int(df[col].isna().sum()),
            "is_likely_target": is_likely_target,
        }

    return numeric_meta


def _detect_grouping_columns(df: pd.DataFrame, date_cols: list[str], numeric_cols: list[str]) -> list[str]:
    """Detect columns that are likely product/store/category groupers."""
    grouping = []
    for col in df.columns:
        if col in date_cols or col in numeric_cols:
            continue
        col_lower = col.lower().replace(" ", "_")
        if any(kw in col_lower for kw in GROUPING_KEYWORDS):
            grouping.append(col)
        elif df[col].dtype == object and df[col].nunique() < min(50, len(df) * 0.2):
            # Low-cardinality string columns are likely categories
            grouping.append(col)
    return grouping


def _detect_exog_columns(df: pd.DataFrame, date_cols: list[str]) -> dict[str, list[str]]:
    """
    Flag numeric columns that look like usable exogenous features — price or
    a promo/discount flag — for the models that can use real exogenous data
    (XGBoost, RandomForest, SARIMAX; see services/preprocessor.py). Checked
    independently of _detect_numeric_columns' target-oriented exclusions,
    since e.g. a column named "promo_flag" is a bad forecast target but a
    perfectly good exog feature.
    """
    price_cols, promo_cols = [], []
    for col in df.select_dtypes(include=[np.number]).columns:
        if col in date_cols:
            continue
        col_lower = col.lower().replace(" ", "_")
        if any(kw in col_lower for kw in PROMO_KEYWORDS):
            promo_cols.append(col)
        elif any(kw in col_lower for kw in PRICE_KEYWORDS):
            price_cols.append(col)
    return {"price_cols": price_cols, "promo_cols": promo_cols}


def _infer_frequency(df: pd.DataFrame, date_col: str) -> str:
    """Infer the time series frequency from the date column."""
    try:
        parsed = pd.to_datetime(df[date_col], errors="coerce")
        parsed = parsed.dropna().sort_values()
        if len(parsed) < 3:
            return "unknown"
        freq = pd.infer_freq(parsed)
        if freq is None:
            # Estimate from median gap
            gaps = parsed.diff().dropna()
            median_gap = gaps.median()
            if median_gap.days <= 1:
                return "daily (estimated)"
            elif median_gap.days <= 8:
                return "weekly (estimated)"
            elif median_gap.days <= 35:
                return "monthly (estimated)"
            elif median_gap.days <= 100:
                return "quarterly (estimated)"
            else:
                return "annual (estimated)"
        # Strip trailing digits (e.g. "MS" → "M", "QS-JAN" → "Q")
        freq_base = freq.rstrip("0123456789").split("-")[0].upper()
        return FREQ_MAP.get(freq_base, freq)
    except Exception:
        return "unknown"


def _validate(
    date_cols: list[str],
    numeric_cols: list[str],
    row_count: int,
) -> tuple[bool, list[str], list[str]]:
    """Return (is_valid, errors, warnings)."""
    errors, warnings = [], []

    if not date_cols:
        errors.append(
            "No date/time column detected. "
            "Please ensure your CSV has a column with dates (e.g. 'Date', 'Timestamp', 'Week')."
        )
    if not numeric_cols:
        errors.append(
            "No numeric target column detected. "
            "Your CSV must have at least one numeric column to forecast (e.g. 'Sales', 'Revenue', 'Demand')."
        )
    if row_count < 30:
        warnings.append(
            f"Only {row_count} rows found. "
            "Forecasting models need at least 30 data points for reliable results."
        )
    elif row_count < 60:
        warnings.append(
            f"Only {row_count} rows found. Results may be less accurate with fewer than 60 data points."
        )

    is_valid = len(errors) == 0
    return is_valid, errors, warnings


def _safe_json(obj):
    """Make a value JSON-serialisable."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
        return None
    if isinstance(obj, (pd.Timestamp, datetime)):
        return obj.isoformat()
    return obj


def _df_to_json_safe(df: pd.DataFrame) -> list[dict]:
    """Convert a DataFrame to a JSON-safe list of dicts."""
    records = df.to_dict(orient="records")
    return [
        {k: _safe_json(v) for k, v in row.items()}
        for row in records
    ]


# ── Endpoint ──────────────────────────────────────────────────────────────────

SUPPORTED_EXTENSIONS = (".csv", ".xlsx", ".xlsm", ".json")


def _read_csv_bytes(save_path: Path) -> tuple[pd.DataFrame | None, list[str]]:
    """Try every encoding/separator combination until one produces a usable frame."""
    read_errors: list[str] = []
    for encoding in ["utf-8", "latin-1", "cp1252", "utf-16"]:
        for sep in [",", ";", "\t", "|"]:
            try:
                candidate = pd.read_csv(
                    save_path, encoding=encoding, sep=sep,
                    engine="python", on_bad_lines="skip",
                )
                if candidate.shape[1] >= 2 and candidate.shape[0] >= 1:
                    return candidate, read_errors
            except Exception as e:
                read_errors.append(str(e))
    return None, read_errors


def _read_excel_file(save_path: Path) -> tuple[pd.DataFrame | None, list[str]]:
    """Reads only the first sheet — no sheet-picker UI in this version."""
    try:
        df = pd.read_excel(save_path, engine="openpyxl", sheet_name=0)
        return df, []
    except Exception as e:
        return None, [str(e)]


def _read_json_file(save_path: Path) -> tuple[pd.DataFrame | None, list[str]]:
    """Tries the common flat-records-array shape first, then falls back to
    flattening nested structures."""
    errors: list[str] = []
    try:
        df = pd.read_json(save_path)
        if df.shape[1] >= 1 and df.shape[0] >= 1:
            return df, errors
    except Exception as e:
        errors.append(str(e))
    try:
        raw = json.loads(save_path.read_text())
        df = pd.json_normalize(raw)
        if df.shape[1] >= 1 and df.shape[0] >= 1:
            return df, errors
    except Exception as e:
        errors.append(str(e))
    return None, errors


@router.post("/upload")
async def upload_csv(file: UploadFile = File(...)):
    """
    Accept a CSV, Excel (.xlsx/.xlsm), or JSON file, save it, analyse its
    structure, and return everything the frontend needs to populate
    dropdowns and start forecasting.

    Handles any time series dataset:
      - Retail (sales, demand, inventory)
      - Finance (stock prices, revenue)
      - Energy (kWh, production)
      - IoT / sensors (temperature, pressure)
      - Transport (passenger counts, trips)
      - Healthcare, web analytics, HR, etc.

    Note: only the first sheet of an Excel workbook is read (no sheet-picker
    yet); classic pre-2007 .xls files are not supported (would need the
    separate `xlrd` dependency).
    """

    # ── 1. Validate file type ─────────────────────────────────────────────────
    filename = file.filename or ""
    filename_lower = filename.lower()
    if not filename_lower.endswith(SUPPORTED_EXTENSIONS):
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type. Please upload one of: {', '.join(SUPPORTED_EXTENSIONS)}."
        )

    # ── 2. Save file ──────────────────────────────────────────────────────────
    unique_name = f"{uuid.uuid4().hex}_{filename}"
    save_path = UPLOAD_DIR / unique_name

    try:
        content = await file.read()
        save_path.write_bytes(content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {e}")

    # ── 3. Read by file type ──────────────────────────────────────────────────
    if filename_lower.endswith((".xlsx", ".xlsm")):
        df, read_errors = _read_excel_file(save_path)
        format_label = "Excel"
    elif filename_lower.endswith(".json"):
        df, read_errors = _read_json_file(save_path)
        format_label = "JSON"
    else:
        df, read_errors = _read_csv_bytes(save_path)
        format_label = "CSV"

    if df is None:
        save_path.unlink(missing_ok=True)
        raise HTTPException(
            status_code=400,
            detail=(
                f"Could not read your {format_label} file. "
                f"Details: {read_errors[-1] if read_errors else 'unknown error'}"
            )
        )

    # ── 4. Basic clean-up ─────────────────────────────────────────────────────
    df.columns = [str(c).strip() for c in df.columns]   # strip whitespace from headers
    df = df.loc[:, ~df.columns.duplicated()]              # drop duplicate column names
    df = df.dropna(how="all")                             # drop fully-empty rows

    rows, cols = df.shape

    # ── 5. Column analysis ────────────────────────────────────────────────────
    date_col_map = _detect_date_columns(df)
    date_col_names = list(date_col_map.keys())

    numeric_col_map = _detect_numeric_columns(df, date_col_names)
    numeric_col_names = list(numeric_col_map.keys())

    grouping_cols = _detect_grouping_columns(df, date_col_names, numeric_col_names)
    exog_cols = _detect_exog_columns(df, date_col_names)

    # Suggest the best default date and target columns
    suggested_date_col = date_col_names[0] if date_col_names else None
    likely_targets = [c for c, m in numeric_col_map.items() if m["is_likely_target"]]
    suggested_target_col = likely_targets[0] if likely_targets else (
        numeric_col_names[0] if numeric_col_names else None
    )

    # Infer frequency from the best date column
    frequency = (
        _infer_frequency(df, suggested_date_col)
        if suggested_date_col else "unknown"
    )

    # ── 6. Data quality summary ───────────────────────────────────────────────
    null_summary = {
        col: int(df[col].isna().sum())
        for col in df.columns
        if df[col].isna().sum() > 0
    }

    # ── 7. Validation ─────────────────────────────────────────────────────────
    is_valid, errors, warnings = _validate(date_col_names, numeric_col_names, rows)

    # ── 8. Preview (first 5 rows, JSON-safe) ──────────────────────────────────
    preview = _df_to_json_safe(df.head(5))

    # ── 9. Column dtype summary ───────────────────────────────────────────────
    all_columns = [
        {
            "name": col,
            "dtype": str(df[col].dtype),
            "role": (
                "date"     if col in date_col_names    else
                "target"   if col in numeric_col_names else
                "grouping" if col in grouping_cols     else
                "other"
            ),
            "sample": _safe_json(df[col].dropna().iloc[0]) if df[col].notna().any() else None,
        }
        for col in df.columns
    ]

    # ── 10. Build response ────────────────────────────────────────────────────
    detected = {
        "date_columns": date_col_names,
        "numeric_columns": numeric_col_names,
        "grouping_columns": grouping_cols,
        "date_samples": date_col_map,       # {col: sample_value}
        "numeric_meta": numeric_col_map,    # {col: {min,max,mean,null_count,is_likely_target}}
        "exog_columns": exog_cols,          # {price_cols: [...], promo_cols: [...]} — optional
                                             # extra features for XGBoost/RandomForest/SARIMAX
    }
    suggestions = {
        "date_col": suggested_date_col,
        "target_col": suggested_target_col,
        "frequency": frequency,
    }
    validation = {
        "is_valid": is_valid,
        "errors": errors,
        "warnings": warnings,
    }

    return JSONResponse(content={
        "status": "success",
        "file": {
            "filename": filename,
            "saved_as": unique_name,
            "size_bytes": save_path.stat().st_size,
        },
        "shape": {
            "rows": rows,
            "cols": cols,
        },
        "preview": preview,
        "columns": all_columns,
        "detected": detected,
        "suggestions": suggestions,
        "data_quality": {
            "null_counts": null_summary,
            "total_nulls": sum(null_summary.values()),
        },
        "validation": validation,
    })
