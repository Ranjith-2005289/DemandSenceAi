"""
product_analytics.py — Fast, model-free per-group sales ranking and trend detection.

Answers "which products/stores/categories sell well or poorly and what's the
trend" without running any forecasting models, so it stays cheap regardless
of how many groups (SKUs, stores, categories...) a dataset has.
"""

import pandas as pd

from services.preprocessor import _parse_dates, _parse_target

GROWTH_THRESHOLD_PCT = 5.0


def _trend_for_group(dates: pd.Series, values: pd.Series) -> tuple[str, float | None]:
    """Compare mean of first half vs second half (chronologically) of a group's series."""
    valid = dates.notna()
    dates, values = dates[valid], values[valid]

    if len(values) < 4:
        return "insufficient_data", None

    order = dates.values.argsort()
    values = values.iloc[order].reset_index(drop=True)

    mid = len(values) // 2
    first_half_mean = values.iloc[:mid].mean()
    second_half_mean = values.iloc[mid:].mean()

    if first_half_mean == 0 or pd.isna(first_half_mean) or pd.isna(second_half_mean):
        return "insufficient_data", None

    growth_rate_pct = float((second_half_mean - first_half_mean) / abs(first_half_mean) * 100)

    if growth_rate_pct > GROWTH_THRESHOLD_PCT:
        trend = "growing"
    elif growth_rate_pct < -GROWTH_THRESHOLD_PCT:
        trend = "declining"
    else:
        trend = "flat"

    return trend, round(growth_rate_pct, 2)


def compute_group_summary(
    df: pd.DataFrame,
    date_col: str,
    target_col: str,
    group_col: str,
    top_n: int = 50,
) -> dict:
    """
    Rank groups (e.g. products/stores/categories) by total target value and
    label each with a trend. No forecasting models involved — safe for any
    cardinality.

    Returns:
        {
            "group_col": str,
            "total_groups": int,
            "truncated": bool,
            "groups": [
                {group_value, total, mean, share_pct, row_count,
                 trend, growth_rate_pct, rank}, ...
            ]  # sorted by total desc, capped at top_n
        }
    """
    for col in (date_col, target_col, group_col):
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in dataset.")

    working = df[[date_col, target_col, group_col]].copy()
    working[target_col] = _parse_target(working[target_col], target_col)
    working = working.dropna(subset=[target_col])

    grand_total = float(working[target_col].sum())
    total_groups = working[group_col].nunique(dropna=True)

    rows = []
    for group_value, group_df in working.groupby(group_col, dropna=True):
        total = float(group_df[target_col].sum())
        mean = float(group_df[target_col].mean())
        row_count = int(len(group_df))

        try:
            parsed_dates = _parse_dates(group_df[date_col])
            trend, growth_rate_pct = _trend_for_group(parsed_dates, group_df[target_col])
        except Exception:
            trend, growth_rate_pct = "insufficient_data", None

        share_pct = round((total / grand_total * 100), 2) if grand_total else 0.0

        rows.append({
            "group_value": str(group_value),
            "total": round(total, 4),
            "mean": round(mean, 4),
            "row_count": row_count,
            "share_pct": share_pct,
            "trend": trend,
            "growth_rate_pct": growth_rate_pct,
        })

    rows.sort(key=lambda r: r["total"], reverse=True)

    truncated = len(rows) > top_n
    top_rows = rows[:top_n]
    for i, row in enumerate(top_rows, start=1):
        row["rank"] = i

    return {
        "group_col": group_col,
        "total_groups": int(total_groups),
        "truncated": truncated,
        "groups": top_rows,
    }
