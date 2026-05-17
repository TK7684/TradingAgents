# TradingAgents/graph/signal_processing.py

import logging
import re
from typing import Any
from tradingagents.llm_clients.retry_handler import LLMRetryHandler

log = logging.getLogger(__name__)

VALID_SIGNALS = {"BUY", "OVERWEIGHT", "HOLD", "UNDERWEIGHT", "SELL"}


class SignalProcessor:
    """Processes trading signals to extract actionable decisions."""

    def __init__(self, quick_thinking_llm: Any):
        """Initialize with an LLM for processing."""
        self.quick_thinking_llm = quick_thinking_llm
        self.retry_handler = LLMRetryHandler(max_retries=2, base_delay=1.0)

    def process_signal(self, full_signal: str) -> str:
        """
        Process a full trading signal to extract the core decision.

        Args:
            full_signal: Complete trading signal text

        Returns:
            Extracted rating (BUY, OVERWEIGHT, HOLD, UNDERWEIGHT, or SELL)
        """
        if not full_signal or not isinstance(full_signal, str) or full_signal.strip() == "":
            log.warning("Empty or None signal provided, defaulting to HOLD")
            return "HOLD"

        messages = [
            (
                "system",
                "You are an efficient assistant that extracts the trading decision from analyst reports. "
                "Extract the rating as exactly one of: BUY, OVERWEIGHT, HOLD, UNDERWEIGHT, SELL. "
                "Output only the single rating word, nothing else.",
            ),
            ("human", full_signal),
        ]

        llm_response = self.retry_handler.invoke_with_retry(
            self.quick_thinking_llm,
            messages,
            fallback_response="HOLD",
        )

        # Attempt 1: Exact match (case-insensitive)
        cleaned = llm_response.strip().upper()
        if cleaned in VALID_SIGNALS:
            return cleaned

        # Attempt 2: Regex extraction - look for exact word boundaries
        match = re.search(r"\b(BUY|OVERWEIGHT|HOLD|UNDERWEIGHT|SELL)\b", cleaned)
        if match:
            return match.group(1)

        # Attempt 3: Substring search as fallback
        for signal in VALID_SIGNALS:
            if signal in cleaned:
                log.warning(
                    f"LLM returned '{llm_response}'; matched via substring fallback to {signal}"
                )
                return signal

        # Default to HOLD if no valid signal found
        log.warning(
            f"LLM returned '{llm_response}'; no valid signal matched, defaulting to HOLD"
        )
        return "HOLD"
