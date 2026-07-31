"""
Insights Router — the "AI Insights Report" page (separate from the chatbot).
"""
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from services.preprocessor import preprocess
from services.insights_service import generate_ai_report

router = APIRouter(prefix="/api/insights", tags=["insights"])

UPLOAD_DIR = Path("uploads")


class InsightsRequest(BaseModel):
    """Request body for POST /api/insights/generate."""
    filename: str
    date_col: str
    target_col: str
    group_col: Optional[str] = None
    forecast_result: Dict[str, Any]
    product_summary: Optional[Dict[str, Any]] = None


@router.post("/generate")
def generate_insights(request: InsightsRequest) -> dict:
    """
    Generate the proactive AI Insights Report: dataset overview, sales
    forecast outlook, recommendations, profit/loss signal, and stock alerts.
    """
    csv_path = UPLOAD_DIR / request.filename
    if not csv_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"File '{request.filename}' not found. Please upload a CSV first.",
        )

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read CSV: {e}")

    try:
        series, _ = preprocess(df, date_col=request.date_col, target_col=request.target_col)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Preprocessing failed: {e}")

    try:
        report = generate_ai_report(
            series=series,
            forecast_result=request.forecast_result,
            product_summary=request.product_summary,
        )
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Insights generation failed: {e}")

    # Chart data for the frontend: raw historical series + per-product shares.
    report["chart_data"] = {
        "dates": [str(d.date()) for d in series.index],
        "values": [float(v) for v in series.values],
    }

    return report
