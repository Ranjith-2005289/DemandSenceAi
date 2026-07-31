"""
insights_service.py — Generates the "AI Insights Report": a proactive,
single-shot briefing (dataset overview, sales outlook, recommendations,
profit/loss signal, stock alerts) — separate from the conversational
chatbot in explainer.py.

Reuses the same building blocks as the chatbot rather than duplicating them:
  - format_forecast_context / format_product_context (explainer.py) for the
    deterministic metrics/product-ranking context blocks.
  - retrieve_relevant_knowledge (rag_service.py) for RAG grounding, queried
    with a composite string built from what this module detects.
  - _trend_for_group (product_analytics.py) for the same first-half vs
    second-half trend method, applied to the aggregate series.

Honesty constraint: this system has sales/demand data, not real inventory
or cost data. Stock alerts are a deterministic heuristic scan of the sales
series for the classic "stockout signature" (flat low plateau -> sharp
recovery), not a guess. Profit/loss is framed as a revenue/trend signal with
an explicit caveat that true profit needs cost data this system doesn't have.
"""
import json
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from google.genai import types

from services.explainer import _client, CHAT_MODEL, format_forecast_context, format_product_context
from services.llm_retry import call_with_retry
from services.product_analytics import _trend_for_group, GROWTH_THRESHOLD_PCT
from services.rag_service import retrieve_relevant_knowledge


def compute_dataset_overview(series: pd.Series) -> dict:
    """Deterministic overview stats for the aggregate series — no LLM involved."""
    dates = pd.Series(series.index)
    trend, growth_rate_pct = _trend_for_group(dates, pd.Series(series.values))

    freq = pd.infer_freq(series.index)
    mean = float(series.mean())
    std = float(series.std())
    # Coefficient of variation — a bare std is meaningless without the mean
    # for scale (std=4.32 is huge if mean=10, negligible if mean=500). This
    # is what "volatility is high/low" claims should actually be based on.
    cv = round(std / mean, 4) if abs(mean) > 1e-9 else None
    return {
        "row_count": int(len(series)),
        "date_start": str(series.index.min().date()),
        "date_end": str(series.index.max().date()),
        "frequency": freq or "irregular",
        "mean": round(mean, 4),
        "min": round(float(series.min()), 4),
        "max": round(float(series.max()), 4),
        "std": round(std, 4),
        "coefficient_of_variation": cv,
        "trend": trend,
        "growth_rate_pct": growth_rate_pct,
    }


def detect_stockout_alerts(
    series: pd.Series,
    flat_window: int = 5,
    flat_cv_threshold: float = 0.08,
    low_percentile: float = 25,
    recovery_jump_pct: float = 50,
) -> List[dict]:
    """
    Heuristic scan for the classic stockout signature: a run of `flat_window`+
    consecutive points that are both (a) unusually flat (coefficient of
    variation below `flat_cv_threshold`) and (b) unusually low (below the
    `low_percentile` of the whole series), immediately followed by a jump of
    at least `recovery_jump_pct`% back up. This is a deterministic scan, not
    an LLM guess — returns concrete flagged date ranges for the LLM to explain.
    """
    if len(series) < flat_window + 2:
        return []

    values = series.values
    low_threshold = np.percentile(values, low_percentile)
    alerts = []

    i = 0
    n = len(values)
    while i <= n - flat_window:
        window = values[i:i + flat_window]
        window_mean = window.mean()
        window_std = window.std()
        cv = (window_std / window_mean) if window_mean > 1e-9 else 0.0

        is_flat = cv < flat_cv_threshold
        is_low = window_mean <= low_threshold

        if is_flat and is_low:
            # Extend the flat/low run as far as it continues
            end = i + flat_window
            while end < n:
                extended = values[i:end + 1]
                ext_mean = extended.mean()
                ext_cv = (extended.std() / ext_mean) if ext_mean > 1e-9 else 0.0
                if ext_cv < flat_cv_threshold and ext_mean <= low_threshold:
                    end += 1
                else:
                    break

            # Check for a recovery jump right after the flat run
            if end < n:
                recovery_val = values[end]
                plateau_mean = values[i:end].mean()
                if plateau_mean > 1e-9:
                    jump_pct = (recovery_val - plateau_mean) / plateau_mean * 100
                    if jump_pct >= recovery_jump_pct:
                        alerts.append({
                            "start_date": str(pd.Timestamp(series.index[i]).date()),
                            "end_date": str(pd.Timestamp(series.index[end - 1]).date()),
                            "avg_value_during": round(float(plateau_mean), 4),
                            "recovery_jump_pct": round(float(jump_pct), 1),
                        })
            i = end + 1
        else:
            i += 1

    return alerts


def _format_stockout_context(alerts: List[dict]) -> str:
    if not alerts:
        return "\nSTOCKOUT RISK SCAN:\nNo flat-then-recovery patterns detected in the sales history."
    lines = ["\nSTOCKOUT RISK SCAN (detected from sales data patterns, not real inventory data):", "=" * 80]
    for a in alerts:
        lines.append(
            f"- {a['start_date']} to {a['end_date']}: sales flattened at ~{a['avg_value_during']} "
            f"then jumped {a['recovery_jump_pct']:+.1f}% right after — consistent with a stockout "
            f"(supply ran out) rather than a genuine demand dip."
        )
    return "\n".join(lines)


def _format_dataset_overview_context(overview: dict) -> str:
    cv = overview.get("coefficient_of_variation")
    cv_line = ""
    if cv is not None:
        if cv < 0.15:
            volatility_label = "low"
        elif cv < 0.40:
            volatility_label = "moderate"
        else:
            volatility_label = "high"
        cv_line = f"\nVolatility (coefficient of variation = std/mean): {cv:.2f} — {volatility_label} relative to the average level."

    return (
        "\nDATASET OVERVIEW:\n" + "=" * 80 + "\n"
        f"Rows: {overview['row_count']} | Date range: {overview['date_start']} to {overview['date_end']} "
        f"| Frequency: {overview['frequency']}\n"
        f"Sales — mean: {overview['mean']}, min: {overview['min']}, max: {overview['max']}, std: {overview['std']}"
        f"{cv_line}\n"
        f"Overall trend (computed with a fixed ±{GROWTH_THRESHOLD_PCT:.0f}% threshold — changes inside this "
        f"band are labeled 'flat', NOT growing or declining, regardless of the sign of the number): "
        f"{overview['trend']}"
        + (f" ({overview['growth_rate_pct']:+.1f}%)" if overview.get("growth_rate_pct") is not None else "")
    )


REPORT_SCHEMA = {
    "type": "object",
    "properties": {
        "dataset_summary": {"type": "string", "description": "Plain-language explanation of what this dataset contains and its key characteristics."},
        "sales_forecast_summary": {"type": "string", "description": "Plain-language explanation of the future sales forecast and how confident it is."},
        "recommendations": {"type": "string", "description": "Concrete, actionable business recommendations grounded in the retrieved retail knowledge."},
        "profit_loss_analysis": {"type": "string", "description": "Revenue/sales trend direction and business implications, explicitly noting that true profit/margin requires cost data not present in this dataset."},
        "stock_alerts_summary": {"type": "string", "description": "Explanation of any detected stockout-risk periods and what to do about them, or a note that none were found."},
    },
    "required": ["dataset_summary", "sales_forecast_summary", "recommendations", "profit_loss_analysis", "stock_alerts_summary"],
}


def generate_ai_report(
    series: pd.Series,
    forecast_result: Dict[str, Any],
    product_summary: Optional[Dict[str, Any]] = None,
) -> dict:
    """
    Runs the deterministic analysis, retrieves grounding knowledge based on
    what was detected, and makes one structured Gemini call to produce the
    five report sections.
    """
    overview = compute_dataset_overview(series)
    stockout_alerts = detect_stockout_alerts(series)

    forecast_context = format_forecast_context(forecast_result)
    product_context = format_product_context(product_summary)
    overview_context = _format_dataset_overview_context(overview)
    stockout_context = _format_stockout_context(stockout_alerts)

    # Composite RAG query built from what was actually detected, not a typed question.
    query_parts = [f"retail sales trend {overview['trend']}", "choosing forecasting models"]
    if stockout_alerts:
        query_parts.append("stockout signature diagnosis inventory")
    if product_summary and product_summary.get("groups"):
        query_parts.append("assortment planning ABC XYZ inventory classification")
    query_parts.append("sales trend interpretation causes")
    rag_query = ", ".join(query_parts)

    retrieved = retrieve_relevant_knowledge(rag_query, top_k=5)
    knowledge_block = "\n\n".join(f"[Source: {r['source']}]\n{r['text']}" for r in retrieved)

    # Whether XGBoost/RandomForest/SARIMAX actually used real exogenous
    # features (price/promo) THIS run — set by routers/forecast.py based on
    # what was actually detected/selected at upload time, never assumed.
    exog_used = forecast_result.get("exog_features_used") or []
    if exog_used:
        exog_capability_line = (
            f"- This run DID use real exogenous features from the uploaded file: "
            f"{', '.join(exog_used)}. SARIMAX/XGBoost/RandomForest incorporated these "
            f"alongside their usual lag/rolling/calendar features — you may accurately "
            f"say so if relevant."
        )
    else:
        exog_capability_line = (
            "- SARIMAX's exogenous features in this run were calendar cyclicals only "
            "(month/day-of-week sin-cos, is_month_end) — no price/promo columns were "
            "selected for this run, even if such columns exist in the uploaded file. "
            "XGBoost and Random Forest used ONLY lags/rolling stats of the sales series "
            "itself and calendar features. If you recommend tracking/integrating "
            "promotional calendars or price changes, frame this explicitly as something "
            "the retailer should select as a feature column on their next run — the "
            "system already supports it, it just wasn't used this time."
        )

    prompt = f"""You are an expert retail data analyst producing a proactive briefing report
for a retail manager who has just uploaded their sales data. Generate the report using ONLY
the real data below — never invent numbers. Where data is insufficient (e.g. no cost data
for true profit), say so explicitly rather than guessing.

{overview_context}
{forecast_context}
{product_context}
{stockout_context}

RELEVANT RETAIL KNOWLEDGE (ground your recommendations in this where applicable):
{knowledge_block}

CURRENT SYSTEM CAPABILITIES (state these accurately — do not imply the system already
uses data it doesn't, and do not contradict labels it already computed):
- Trend labels (growing/declining/flat) above use a FIXED ±{GROWTH_THRESHOLD_PCT:.0f}% threshold.
  Never re-label or contradict them in your own words — e.g. if a trend is labeled "flat",
  do not call it "declining" or "growing" even if the underlying percentage is negative or
  positive. Describe it as flat/stable/within normal noise instead.
- When discussing volatility, reference the coefficient of variation (relative to the mean)
  given above, not the raw standard deviation alone — a std has no inherent "high" or "low"
  meaning without knowing the mean.
{exog_capability_line}
  Prophet's holiday effects are a separate, always-available mechanism (by country code) —
  don't confuse the two.

Write in clear, business-friendly language for a retail manager, not a data scientist."""

    response = call_with_retry(
        _client.models.generate_content,
        model=CHAT_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=REPORT_SCHEMA,
        ),
    )
    report = json.loads(response.text)

    return {
        "dataset_overview": overview,
        "stockout_alerts": stockout_alerts,
        **report,
    }
