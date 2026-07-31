"""
Chat Router - API endpoints for explainable AI chatbot
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, List, Optional
from services.explainer import get_or_create_chat_session, remove_chat_session, get_persisted_history, session_exists
import uuid

router = APIRouter(prefix="/api/chat", tags=["chat"])


class ChatRequest(BaseModel):
    """Chat message request"""
    question: str
    session_id: Optional[str] = None
    forecast_data: Optional[Dict[str, Any]] = None  # Optional: pass forecast data in request
    product_summary: Optional[Dict[str, Any]] = None  # Optional: per-product sales ranking, if a grouping column was used


class ChatResponse(BaseModel):
    """Chat message response"""
    response: str
    session_id: str
    message_count: int


class ChatHistoryResponse(BaseModel):
    """Chat history response"""
    history: List[Dict[str, str]]
    session_id: str


@router.post("/ask")
def ask_chatbot(request: ChatRequest) -> ChatResponse:
    """
    Ask the AI chatbot a question about the forecast
    
    Expected request:
    {
        "question": "Why was XGBoost selected as the best model?",
        "session_id": "optional-session-id",
        "forecast_data": {...},      // optional, required for first message
        "product_summary": {...}     // optional, from GET /forecast/product-summary
    }
    """
    try:
        # Generate or use provided session ID
        session_id = request.session_id or str(uuid.uuid4())
        
        # Get or create chat session with forecast data (or empty dict if not provided)
        forecast_data = request.forecast_data or {}
        chat = get_or_create_chat_session(session_id, forecast_data, request.product_summary)
        
        # Get response from AI
        response = chat.ask(request.question)
        message_count = len(chat.get_history())
        
        return ChatResponse(
            response=response,
            session_id=session_id,
            message_count=message_count
        )
        
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        error_detail = f"Chat error: {str(e)}\n{traceback.format_exc()}"
        print(error_detail)
        raise HTTPException(
            status_code=500,
            detail=f"Chat error: {str(e)}"
        )


@router.get("/history/{session_id}")
def get_chat_history(session_id: str) -> ChatHistoryResponse:
    """Get chat history for a session (checks the live in-memory session first,
    falling back to disk so history survives a backend restart)."""
    try:
        history = get_persisted_history(session_id)
        if history is None:
            raise HTTPException(
                status_code=404,
                detail=f"Session {session_id} not found"
            )

        return ChatHistoryResponse(
            history=history,
            session_id=session_id
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving history: {str(e)}"
        )


@router.delete("/session/{session_id}")
def clear_chat_session(session_id: str) -> Dict[str, str]:
    """Clear a chat session"""
    try:
        remove_chat_session(session_id)
        return {"status": "success", "message": f"Session {session_id} cleared"}
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error clearing session: {str(e)}"
        )


@router.post("/reset/{session_id}")
def reset_session(session_id: str) -> Dict[str, str]:
    """Reset chat history while keeping the session (its forecast/product
    context). Works whether the session is currently in memory or only on
    disk from a prior process — get_or_create_chat_session hydrates it either way."""
    try:
        if session_exists(session_id):
            chat = get_or_create_chat_session(session_id, {})
            chat.clear_history()
        return {"status": "success", "message": f"Session {session_id} history cleared"}
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error resetting session: {str(e)}"
        )
