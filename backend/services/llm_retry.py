"""
llm_retry.py — Resilience wrapper for Gemini API calls.

Hit repeatedly during this project's development: transient 503 "model is
currently experiencing high demand" errors (worth a retry — they usually
clear within seconds) and 429 daily-quota-exhausted errors (NOT worth a
retry — it's a hard limit on the API key's plan, retrying just burns more
time without helping). This distinguishes the two and handles each the
way that actually helps.
"""
import logging
import time

from google.genai import errors

logger = logging.getLogger(__name__)


def call_with_retry(fn, *args, max_retries: int = 5, base_delay: float = 2.0, **kwargs):
    """
    Call fn(*args, **kwargs), retrying with exponential backoff on
    transient 503 ServerError responses. A 429 quota-exhausted ClientError
    fails fast with a clear, actionable message instead of retrying.
    
    For 503 "High Demand" errors, uses adaptive delays: first 3 retries
    use standard backoff, then subsequent retries use longer delays (8-16s)
    to give Google's infrastructure time to recover.
    """
    last_exc = None
    for attempt in range(max_retries):
        try:
            return fn(*args, **kwargs)
        except errors.ServerError as e:
            last_exc = e
            # Check if it's a 429 embedded in the ServerError (API returns 500-599 but includes 429 info)
            error_str = str(e)
            if "429" in error_str or "quota" in error_str.lower():
                raise RuntimeError(
                    "Gemini API daily quota exceeded for this API key's current plan. "
                    "This is a limit on your Google API key, not an app error — either "
                    "wait for the daily quota to reset, or upgrade the key's billing plan "
                    "in Google AI Studio (https://aistudio.google.com/apikey)."
                ) from e
            # Fast-fail on 404 (model not found) — no point retrying a non-existent model
            if "404" in error_str or "not found" in error_str.lower():
                raise
            if attempt < max_retries - 1:
                # Adaptive delay: fast backoff for first 3 attempts, then slow backoff
                if attempt < 3:
                    delay = base_delay * (2 ** attempt)
                else:
                    # After 3 attempts, use longer delays (8-16s) for 503 recovery
                    delay = base_delay * (2 ** (attempt - 1))
                logger.warning(f"Gemini API transient error (attempt {attempt + 1}/{max_retries}), "
                                f"retrying in {delay:.0f}s: {e}")
                time.sleep(delay)
            continue
        except errors.ClientError as e:
            if getattr(e, "code", None) == 429:
                raise RuntimeError(
                    "Gemini API daily quota exceeded for this API key's current plan. "
                    "This is a limit on your Google API key, not an app error — either "
                    "wait for the daily quota to reset, or upgrade the key's billing plan "
                    "in Google AI Studio (https://aistudio.google.com/apikey)."
                ) from e
            raise
    raise last_exc
