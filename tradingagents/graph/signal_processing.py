# TradingAgents/graph/signal_processing.py

import logging
import re
from typing import Any

log = logging.getLogger(__name__)

VALID_SIGNALS = {"BUY", "OVERWEIGHT", "HOLD", "UNDERWEIGHT", "SELL"}


class SignalProcessor:
    """Processes trading signals to extract actionable decisions.

    Uses regex-first extraction (zero API calls) and only falls back to
    LLM if regex cannot find a clear signal. This saves 1 LLM call per
    ticker (~10 calls per daily run), significantly reducing rate-limit risk.
    """

    def __init__(self, quick_thinking_llm: Any = None):
        """Initialize with an optional LLM for fallback processing."""
        self.quick_thinking_llm = quick_thinking_llm

    def process_signal(self, full_signal: str) -> str:
        """
        Process a full trading signal to extract the core decision.

        Strategy:
        1. Regex extraction from the raw signal text (free, no API call)
        2. Look for "Rating: BUY/OVERWEIGHT/HOLD/UNDERWEIGHT/SELL" pattern
        3. Look for "FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL**"
        4. Scan for any valid signal word with word boundaries
        5. If nothing found and LLM available, use LLM as last resort

        Args:
            full_signal: Complete trading signal text

        Returns:
            Extracted rating (BUY, OVERWEIGHT, HOLD, UNDERWEIGHT, or SELL)
        """
        if not full_signal or not isinstance(full_signal, str) or full_signal.strip() == "":
            log.warning("Empty or None signal provided, defaulting to HOLD")
            return "HOLD"

        # Attempt 1: Look for explicit "Rating: XXX" pattern (most reliable)
        rating_match = re.search(
            r"(?:\*\*)?Rating(?:\*\*)?\s*[:：]\s*(?:\*\*)?(BUY|OVERWEIGHT|HOLD|UNDERWEIGHT|SELL)(?:\*\*)?",
            full_signal,
            re.IGNORECASE,
        )
        if rating_match:
            return rating_match.group(1).upper()

        # Attempt 2: Look for "FINAL TRANSACTION PROPOSAL: **XXX**" pattern
        proposal_match = re.search(
            r"FINAL\s+TRANSACTION\s+PROPOSAL\s*[:：]\s*\*\*(BUY|HOLD|SELL)\*\*",
            full_signal,
            re.IGNORECASE,
        )
        if proposal_match:
            signal = proposal_match.group(1).upper()
            # Map to valid signals
            if signal in VALID_SIGNALS:
                return signal
            return signal  # BUY/HOLD/SELL are all valid

        # Attempt 3: Look for "1. **Rating**: BUY" numbered list pattern
        numbered_match = re.search(
            r"1\.\s*\*\*Rating\*\*\s*[:：]\s*(?:\*\*)?(BUY|OVERWEIGHT|HOLD|UNDERWEIGHT|SELL)(?:\*\*)?",
            full_signal,
            re.IGNORECASE,
        )
        if numbered_match:
            return numbered_match.group(1).upper()

        # Attempt 4: Scan for any valid signal word with word boundaries
        # Prioritize: count occurrences, pick the most frequent
        signal_counts = {}
        for signal in VALID_SIGNALS:
            matches = re.findall(rf"\b{signal}\b", full_signal, re.IGNORECASE)
            if matches:
                signal_counts[signal] = len(matches)

        if signal_counts:
            # Return the most frequently mentioned signal
            best = max(signal_counts, key=signal_counts.get)
            # But if HOLD appears everywhere and there's a stronger signal, prefer the stronger one
            if best == "HOLD" and len(signal_counts) > 1:
                for preferred in ["SELL", "UNDERWEIGHT", "BUY", "OVERWEIGHT"]:
                    if preferred in signal_counts and signal_counts[preferred] >= signal_counts[best]:
                        best = preferred
                        break
            return best

        # Attempt 5: LLM fallback (only if LLM is available)
        if self.quick_thinking_llm:
            log.warning("Regex extraction failed, falling back to LLM for signal processing")
            return self._llm_fallback(full_signal)

        # Default to HOLD if nothing found
        log.warning("No valid signal found in text, defaulting to HOLD")
        return "HOLD"

    def _llm_fallback(self, full_signal: str) -> str:
        """Use LLM to extract signal when regex fails."""
        try:
            from tradingagents.llm_clients.retry_handler import LLMRetryHandler
            retry_handler = LLMRetryHandler(max_retries=2, base_delay=1.0)

            messages = [
                (
                    "system",
                    "You are an efficient assistant that extracts the trading decision from analyst reports. "
                    "Extract the rating as exactly one of: BUY, OVERWEIGHT, HOLD, UNDERWEIGHT, SELL. "
                    "Output only the single rating word, nothing else.",
                ),
                ("human", full_signal),
            ]

            llm_response = retry_handler.invoke_with_retry(
                self.quick_thinking_llm,
                messages,
                fallback_response="HOLD",
            )

            cleaned = llm_response.strip().upper()
            match = re.search(r"\b(BUY|OVERWEIGHT|HOLD|UNDERWEIGHT|SELL)\b", cleaned)
            if match:
                return match.group(1)

            return "HOLD"
        except Exception as e:
            log.error(f"LLM fallback failed: {e}, defaulting to HOLD")
            return "HOLD"
