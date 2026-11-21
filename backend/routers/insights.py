"""
Insights Router — the "AI Insights Report" page (separate from the chatbot).
"""
import json
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
from fastapi import APIRouter, HTTPException, BackgroundTasks
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
def generate_insights(request: InsightsRequest, background_tasks: BackgroundTasks) -> dict:
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

    # Fast-path: attempt a quick AI generation with reduced retries so the
    # UI isn't blocked for long when the LLM is busy. Regardless of the
    # outcome, schedule a background job that will attempt a full report
    # with longer retry/backoff and persist it to disk for the UI to fetch.
    try:
        report = generate_ai_report(
            series=series,
            forecast_result=request.forecast_result,
            product_summary=request.product_summary,
            llm_max_retries=2,
            llm_base_delay=1.0,
        )
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Insights generation failed: {e}")

    # Schedule background generation of the full report (longer retries).
    out_path = UPLOAD_DIR / f"{request.filename}.insights.json"
    background_tasks.add_task(
        # import here to avoid circular imports in some runtimes
        __import__("services.insights_service", fromlist=["generate_and_save_report"]).generate_and_save_report,
        str(csv_path),
        request.date_col,
        request.target_col,
        request.forecast_result,
        request.product_summary,
        str(out_path),
        6,
        2.0,
    )

    # Chart data for the frontend: raw historical series + per-product shares.
    report["chart_data"] = {
        "dates": [str(d.date()) for d in series.index],
        "values": [float(v) for v in series.values],
    }
    # Inform the UI where a fuller async report will be written (if/when ready).
    report["async_report_path"] = str(out_path)
    report["async_report_ready"] = out_path.exists()

    return report


@router.get("/async_report_status")
def async_report_status(filename: str) -> dict:
    """
    Poll for async AI report completion. Returns the full report JSON once
    the background job finishes, or a 'ready' flag if not yet complete.
    """
    out_path = UPLOAD_DIR / f"{filename}.insights.json"
    if not out_path.exists():
        return {"ready": False, "report": None}
    
    try:
        with open(out_path, "r", encoding="utf-8") as f:
            report_json = json.load(f)
        return {"ready": True, "report": report_json}
    except Exception as e:
        return {"ready": False, "error": str(e), "report": None}
