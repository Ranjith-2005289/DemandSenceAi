"""
Explainable AI Service — generates contextual, grounded explanations of
forecast results using Google Gemini.

Three distinct sources of context are combined, deliberately kept separate:
  1. Forecast/model metrics (deterministic — computed by the pipeline)
  2. Per-product/group sales facts (deterministic — from product_analytics.py)
  3. Retail domain knowledge (retrieved — from rag_service.py), used only to
     ground qualitative advice ("how do I improve this"), never for numeric
     facts, since retrieval-based grounding for numbers a system already
     knows exactly would only add hallucination risk, not reduce it.
"""
import os
from typing import Dict, Any, List, Optional

from dotenv import load_dotenv
from google import genai
from google.genai import types

from services.llm_retry import call_with_retry
from services.product_analytics import GROWTH_THRESHOLD_PCT
from services.rag_service import retrieve_relevant_knowledge
from services import session_store

# role name Gemini's API expects for the assistant side of a chat turn.
_GEMINI_ASSISTANT_ROLE = "model"

load_dotenv()
API_KEY = os.environ.get("GEMINI_API_KEY")
if not API_KEY:
    raise RuntimeError(
        "GEMINI_API_KEY is not set. Create a .env file in backend/ "
        "(see .env.example) or export GEMINI_API_KEY before starting the server."
    )
_client = genai.Client(api_key=API_KEY)

CHAT_MODEL = "gemini-flash-latest"


def _accuracy_percent(mape: float | None) -> float:
    """Convert MAPE (a fraction, e.g. 0.05 = 5%) to a rough 0-100 accuracy score."""
    if mape is None or not (mape == mape):  # None or NaN
        return 0.0
    return max(0.0, min(100.0, 100.0 * (1 - mape)))


def format_forecast_context(forecast_result: Dict[str, Any]) -> str:
    """Format forecast results (real metrics: rmse/mae/mape/r2) into a context string."""
    best_model = forecast_result.get("best_model", {}) or {}
    all_models = forecast_result.get("all_models", []) or []
    run_info = forecast_result.get("model_run_info") or {}

    def _score(m):
        return _accuracy_percent(m.get("mape"))

    sorted_models = sorted(
        [m for m in all_models if m.get("status") == "success"],
        key=_score, reverse=True,
    )

    lines = []
    if run_info.get("dl_skipped"):
        lines.append(
            f"NOTE: Deep learning models (RNN, LSTM, GRU, BiLSTM, CNN1D, TCN, Transformer) "
            f"were automatically skipped for this run — this dataset has only "
            f"{run_info.get('row_count')} rows, below the {run_info.get('min_rows_for_dl')}-row "
            f"minimum used for reliable deep learning forecasts. This was a deliberate system "
            f"decision, not an error — state it as a fact if relevant, don't re-derive it."
        )
        lines.append("")

    lines.append("MODEL PERFORMANCE COMPARISON:")
    lines.append("=" * 80)
    for i, model in enumerate(sorted_models, 1):
        acc = _score(model)
        lines.append(f"{i}. {model.get('model_name', 'Unknown')}:")
        lines.append(f"   - Accuracy: {acc:.2f}%")
        lines.append(f"   - RMSE: {float(model.get('rmse', 0) or 0):.4f}")
        lines.append(f"   - MAE: {float(model.get('mae', 0) or 0):.4f}")
        lines.append(f"   - R² Score: {float(model.get('r2', 0) or 0):.4f}")
        if i == 1:
            lines.append("   ⭐ BEST MODEL (Selected)")
        lines.append("")

    lines.append("FORECAST SUMMARY:")
    lines.append("=" * 80)
    lines.append(f"Best Model Selected: {best_model.get('best_model_name', 'Unknown')}")
    best_acc = _accuracy_percent(best_model.get("mape"))
    lines.append(f"Model Accuracy: {best_acc:.2f}%")
    lines.append(f"Confidence (R²): {float(best_model.get('r2', 0) or 0):.4f}")

    forecast_values = best_model.get("forecast", []) or []
    lines.append(f"Forecast Range: {len(forecast_values)} time periods")
    if forecast_values:
        lines.append(f"Forecast Min: {min(forecast_values):.2f}")
        lines.append(f"Forecast Max: {max(forecast_values):.2f}")
        lines.append(f"Forecast Avg: {sum(forecast_values)/len(forecast_values):.2f}")

    if best_acc >= 90:
        confidence_level = "Very High (90%+) - Highly Reliable"
    elif best_acc >= 80:
        confidence_level = "High (80-90%) - Reliable"
    elif best_acc >= 70:
        confidence_level = "Moderate (70-80%) - Acceptable"
    else:
        confidence_level = "Low (<70%) - Use with Caution"
    lines.append(f"Confidence Level: {confidence_level}")

    return "\n".join(lines)


def format_product_context(product_summary: Optional[Dict[str, Any]]) -> str:
    """Format a product_analytics.compute_group_summary() result into a context block."""
    if not product_summary or not product_summary.get("groups"):
        return ""

    group_col = product_summary.get("group_col", "group")
    total_groups = product_summary.get("total_groups", 0)
    truncated = product_summary.get("truncated", False)

    lines = [f"\nSALES BY {group_col.upper()}:", "=" * 80]
    lines.append(
        f"{total_groups} total {group_col} values"
        + (f" (showing top {len(product_summary['groups'])} by total sales)" if truncated else "")
    )
    lines.append(
        f"(trend labels use a fixed ±{GROWTH_THRESHOLD_PCT:.0f}% threshold — changes inside this band "
        f"are labeled 'flat', NOT growing or declining, regardless of the sign of the number)"
    )
    for row in product_summary["groups"]:
        lines.append(
            f"{row['rank']}. {row['group_value']}: total={row['total']:.2f} "
            f"({row['share_pct']}% of total), trend={row['trend']}"
            + (f" ({row['growth_rate_pct']:+.1f}%)" if row.get("growth_rate_pct") is not None else "")
        )
    return "\n".join(lines)


def create_system_prompt(forecast_context: str, product_context: str = "") -> str:
    """Create the system prompt combining forecast metrics + product facts."""
    return f"""You are an expert data scientist and retail business analyst specializing in demand forecasting.
You have deep knowledge of time series analysis, machine learning models, and retail operations.

Your role is to explain forecast results to retail managers and business stakeholders in clear, non-technical language.

HERE IS THE FORECAST DATA YOU'RE ANALYZING:
{forecast_context}
{product_context}

GUIDELINES:
1. Always explain complex concepts in simple, business-friendly language
2. Reference specific metrics (accuracy %, RMSE, R²) when explaining model performance
3. Be honest about confidence levels - if accuracy is low, mention limitations
4. Provide actionable insights - not just predictions but what the numbers mean for business
5. When asked about specific products, use the exact numbers from the sales-by-product data above - never invent figures
6. When you're given "RELEVANT RETAIL KNOWLEDGE" alongside a question, ground your advice in it explicitly rather than generic guesses
7. If asked about something outside the forecast context, politely redirect to forecast-related topics

Remember: You're speaking to retail professionals, not data scientists. Be clear, practical, and business-focused."""


def _history_to_gemini_content(chat_history: List[Dict[str, str]]) -> List[types.Content]:
    """Convert our persisted {"role": "user"/"assistant", "content": str} turns into
    the Content objects chats.create(history=...) expects, so a restored session
    resumes with full conversational context instead of replaying every turn
    through the API (which would cost real quota just to rebuild state)."""
    return [
        types.Content(
            role="user" if turn["role"] == "user" else _GEMINI_ASSISTANT_ROLE,
            parts=[types.Part(text=turn["content"])],
        )
        for turn in chat_history
    ]


class ExplainerChat:
    """Chat interface for forecast explanations, grounded in real metrics/product facts
    and (for advice-style questions) retrieved retail domain knowledge."""

    def __init__(
        self,
        forecast_result: Dict[str, Any],
        product_summary: Optional[Dict[str, Any]] = None,
        history: Optional[List[Dict[str, str]]] = None,
        session_id: Optional[str] = None,
    ):
        self.forecast_result = forecast_result
        self.product_summary = product_summary
        self.session_id = session_id
        self.context = format_forecast_context(forecast_result)
        self.product_context = format_product_context(product_summary)
        self.system_prompt = create_system_prompt(self.context, self.product_context)
        self.chat_history: List[Dict[str, str]] = list(history) if history else []
        self._chat = _client.chats.create(
            model=CHAT_MODEL,
            config=types.GenerateContentConfig(system_instruction=self.system_prompt),
            history=_history_to_gemini_content(self.chat_history) if self.chat_history else None,
        )

    def _persist(self) -> None:
        """Mirror this session's state to disk so it survives a backend restart.
        Best-effort — a disk write failure shouldn't break the chat response
        the user is already waiting on."""
        if not self.session_id:
            return
        try:
            session_store.save("chat", self.session_id, {
                "forecast_result": self.forecast_result,
                "product_summary": self.product_summary,
                "chat_history": self.chat_history,
            })
        except Exception as e:
            print(f"⚠️ Failed to persist chat session {self.session_id!r}: {e}")

    def ask(self, user_question: str) -> str:
        """Ask the chatbot a question, grounding advice-style answers in retrieved
        retail knowledge without polluting the persisted conversation history with it."""
        try:
            retrieved = retrieve_relevant_knowledge(user_question, top_k=3)
            if retrieved:
                knowledge_block = "\n\n".join(
                    f"[Source: {r['source']}]\n{r['text']}" for r in retrieved
                )
                augmented_message = (
                    f"RELEVANT RETAIL KNOWLEDGE (use this to ground your answer if relevant "
                    f"to the question; ignore it if the question is purely about the numbers above):\n"
                    f"{knowledge_block}\n\n"
                    f"USER QUESTION: {user_question}"
                )
            else:
                augmented_message = user_question

            response = call_with_retry(self._chat.send_message, augmented_message)
            response_text = response.text

            self.chat_history.append({"role": "user", "content": user_question})
            self.chat_history.append({"role": "assistant", "content": response_text})
            self._persist()
            return response_text
        except Exception as e:
            error_msg = f"Error generating response: {str(e)}"
            print(f"❌ Gemini API Error: {error_msg}")
            return error_msg

    def get_history(self) -> List[Dict[str, str]]:
        return self.chat_history.copy()

    def clear_history(self):
        self.chat_history = []
        self._chat = _client.chats.create(
            model=CHAT_MODEL,
            config=types.GenerateContentConfig(system_instruction=self.system_prompt),
        )
        self._persist()


# Global chat sessions (in production, use proper session management). Mirrored
# to disk (services/session_store.py) so a backend restart doesn't lose active
# conversations — see get_or_create_chat_session's disk-restore fallback below.
_chat_sessions: Dict[str, ExplainerChat] = {}


def get_or_create_chat_session(
    session_id: str,
    forecast_result: Dict[str, Any],
    product_summary: Optional[Dict[str, Any]] = None,
) -> ExplainerChat:
    if session_id in _chat_sessions:
        return _chat_sessions[session_id]

    persisted = session_store.load("chat", session_id)
    if persisted is not None:
        # A session that already existed (even in a prior process) resumes
        # from its own saved context — the forecast_result/product_summary
        # passed in on a follow-up call is ignored, same as the existing
        # in-memory hit above already does.
        chat = ExplainerChat(
            persisted.get("forecast_result") or {},
            persisted.get("product_summary"),
            history=persisted.get("chat_history"),
            session_id=session_id,
        )
    else:
        chat = ExplainerChat(forecast_result, product_summary, session_id=session_id)
        chat._persist()

    _chat_sessions[session_id] = chat
    return chat


def remove_chat_session(session_id: str):
    if session_id in _chat_sessions:
        del _chat_sessions[session_id]
    session_store.delete("chat", session_id)


def session_exists(session_id: str) -> bool:
    """True if this session is live in memory or recoverable from disk."""
    return session_id in _chat_sessions or session_store.load("chat", session_id) is not None


def get_persisted_history(session_id: str) -> Optional[List[Dict[str, str]]]:
    """Read a session's chat history without spinning up a live Gemini chat
    object — used for read-only history lookups (GET /api/chat/history).
    Returns None only if the session doesn't exist anywhere (memory or disk)."""
    if session_id in _chat_sessions:
        return _chat_sessions[session_id].get_history()
    persisted = session_store.load("chat", session_id)
    if persisted is not None:
        return persisted.get("chat_history", [])
    return None
