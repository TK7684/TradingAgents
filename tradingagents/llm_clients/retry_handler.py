"""Retry handler for LLM invocations with exponential backoff."""

import logging
import time
from typing import Any, List, Optional

log = logging.getLogger(__name__)


class LLMRetryHandler:
    """Wraps LLM invoke calls with retry logic."""

    def __init__(self, max_retries: int = 3, base_delay: float = 2.0):
        self.max_retries = max_retries
        self.base_delay = base_delay

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
                if attempt < self.max_retries:
                    delay = self.base_delay * (2 ** attempt)
                    log.warning(f"LLM call failed ({error_name}), retry {attempt + 1}/{self.max_retries} in {delay}s: {e}")
                    time.sleep(delay)
                else:
                    log.error(f"LLM call failed after {self.max_retries} retries: {e}")

        if fallback_response is not None:
            log.warning(f"Using fallback response after {self.max_retries} retries")
            return fallback_response
        raise last_error
