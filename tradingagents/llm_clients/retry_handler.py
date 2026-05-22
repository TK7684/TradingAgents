"""Retry handler for LLM invocations with exponential backoff and rate-limit awareness."""

import logging
import random
import time
from typing import Any, List, Optional

log = logging.getLogger(__name__)


class LLMRetryHandler:
    """Wraps LLM invoke calls with retry logic.

    Handles rate-limiting (HTTP 429) and transient errors with jittered
    exponential backoff. For 429s, uses longer base delays to respect
    provider rate limits.
    """

    # Rate-limit specific delays (seconds) — longer base to let quotas recover
    RATE_LIMIT_BASE_DELAY = 8.0
    RATE_LIMIT_MAX_DELAY = 120.0
    RATE_LIMIT_JITTER = 3.0

    # General error delays
    BASE_DELAY = 2.0
    MAX_DELAY = 30.0

    def __init__(self, max_retries: int = 5, base_delay: float = 2.0):
        self.max_retries = max_retries
        self.base_delay = base_delay

    def _is_rate_limit_error(self, error: Exception) -> bool:
        """Check if error is a rate-limit (429) response."""
        error_str = str(error).lower()
        return (
            "429" in error_str
            or "rate" in error_str and "limit" in error_str
            or "1302" in error_str  # Z.AI specific error code
            or "too many requests" in error_str
        )

    def _calculate_delay(self, attempt: int, is_rate_limit: bool) -> float:
        """Calculate backoff delay with jitter."""
        if is_rate_limit:
            base = self.RATE_LIMIT_BASE_DELAY
            max_delay = self.RATE_LIMIT_MAX_DELAY
            jitter = self.RATE_LIMIT_JITTER
        else:
            base = self.base_delay
            max_delay = self.MAX_DELAY
            jitter = 1.0

        delay = min(base * (2 ** attempt) + random.uniform(0, jitter), max_delay)
        return delay

    def invoke_with_retry(
        self,
        llm: Any,
        messages: List,
        fallback_response: Optional[str] = None,
    ) -> str:
        last_error = None
        for attempt in range(self.max_retries + 1):
            try:
                result = llm.invoke(messages)
                return result.content if hasattr(result, 'content') else str(result)
            except Exception as e:
                last_error = e
                error_name = type(e).__name__
                is_rate_limit = self._is_rate_limit_error(e)

                if attempt < self.max_retries:
                    delay = self._calculate_delay(attempt, is_rate_limit)
                    prefix = "RATE LIMIT" if is_rate_limit else "ERROR"
                    log.warning(
                        f"{prefix} ({error_name}), retry {attempt + 1}/{self.max_retries} in {delay:.1f}s: {e}"
                    )
                    time.sleep(delay)
                else:
                    log.error(f"LLM call failed after {self.max_retries} retries: {e}")

        if fallback_response is not None:
            log.warning(f"Using fallback response after {self.max_retries} retries")
            return fallback_response
        raise last_error
