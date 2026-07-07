"""Trader: turns the Research Manager's investment plan into a concrete transaction proposal.

Ported from upstream v0.3.1 — uses structured output (TraderProposal schema)
to force the LLM to commit to a typed enum (Buy/Hold/Sell), eliminating the
free-text BUY bias that caused 82% BUY predictions.
"""

from __future__ import annotations

import functools

from langchain_core.messages import AIMessage

from tradingagents.agents.schemas import TraderProposal, render_trader_proposal
from tradingagents.agents.utils.agent_utils import (
    build_instrument_context,
    get_instrument_context_from_state,
    get_language_instruction,
)
from tradingagents.agents.utils.portfolio_context import get_portfolio_context
from tradingagents.agents.utils.structured import (
    bind_structured,
    invoke_structured_or_freetext,
)


def create_trader(llm, memory):
    structured_llm = bind_structured(llm, TraderProposal, "Trader")

    def trader_node(state, name):
        company_name = state["company_of_interest"]
        ticker = company_name.split(" (")[0].split("(")[0].strip() if " (" in company_name else company_name.strip()
        instrument_context = get_instrument_context_from_state(state)
        investment_plan = state["investment_plan"]

        # Portfolio context — critical for sell decisions (kept from our fork)
        portfolio_context = get_portfolio_context(ticker)

        # Past memory from our append-only memory system (backward compat)
        curr_situation = (
            f"{state.get('market_report', '')}\n\n"
            f"{state.get('sentiment_report', '')}\n\n"
            f"{state.get('news_report', '')}\n\n"
            f"{state.get('fundamentals_report', '')}"
        )
        past_memory_str = ""
        try:
            past_memories = memory.get_memories(curr_situation, n_matches=2)
            if past_memories:
                past_memory_str = "\n".join(rec.get("recommendation", "") for rec in past_memories)
        except Exception:
            past_memory_str = ""

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a trading agent analyzing market data to make investment decisions. "
                    "Based on your analysis, provide a specific recommendation to buy, sell, or hold. "
                    "Anchor your reasoning in the analysts' reports and the research plan."
                    + get_language_instruction()
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Based on a comprehensive analysis by a team of analysts, here is an investment "
                    f"plan tailored for {company_name}. {instrument_context} This plan incorporates "
                    f"insights from current technical market trends, macroeconomic indicators, and "
                    f"social media sentiment. Use this plan as a foundation for evaluating your next "
                    f"trading decision.\n\n"
                    f"Proposed Investment Plan: {investment_plan}\n\n"
                    f"Current Portfolio Context: {portfolio_context}\n\n"
                    f"Leverage these insights to make an informed and strategic decision."
                    + (f"\n\nPast reflections:\n{past_memory_str}" if past_memory_str.strip() else "")
                ),
            },
        ]

        trader_plan = invoke_structured_or_freetext(
            structured_llm,
            llm,
            messages,
            render_trader_proposal,
            "Trader",
        )

        return {
            "messages": [AIMessage(content=trader_plan)],
            "trader_investment_plan": trader_plan,
            "sender": name,
        }

    return functools.partial(trader_node, name="Trader")
