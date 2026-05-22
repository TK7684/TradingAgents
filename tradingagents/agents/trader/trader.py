import functools

from tradingagents.agents.utils.agent_utils import build_instrument_context
from tradingagents.agents.utils.portfolio_context import get_portfolio_context


def create_trader(llm, memory):
    def trader_node(state, name):
        company_name = state["company_of_interest"]
        ticker = company_name.split(" (")[0].split("(")[0].strip() if " (" in company_name else company_name.strip()
        instrument_context = build_instrument_context(company_name)
        investment_plan = state["investment_plan"]
        market_research_report = state["market_report"]
        sentiment_report = state["sentiment_report"]
        news_report = state["news_report"]
        fundamentals_report = state["fundamentals_report"]

        # Get portfolio context — critical for sell decisions
        portfolio_context = get_portfolio_context(ticker)

        curr_situation = f"{market_research_report}\n\n{sentiment_report}\n\n{news_report}\n\n{fundamentals_report}"
        past_memories = memory.get_memories(curr_situation, n_matches=2)

        past_memory_str = ""
        if past_memories:
            for i, rec in enumerate(past_memories, 1):
                past_memory_str += rec["recommendation"] + "\n\n"
        else:
            past_memory_str = "No past memories found."

        context = {
            "role": "user",
            "content": f"Based on a comprehensive analysis by a team of analysts, here is an investment plan tailored for {company_name}. {instrument_context} This plan incorporates insights from current technical market trends, macroeconomic indicators, and social media sentiment. Use this plan as a foundation for evaluating your next trading decision.\n\nProposed Investment Plan: {investment_plan}\n\nLeverage these insights to make an informed and strategic decision.",
        }

        messages = [
            {
                "role": "system",
                "content": f"""You are a trading agent analyzing market data to make investment decisions. Based on your analysis, provide a specific recommendation to buy, sell, or hold.

**PORTFOLIO CONTEXT — CRITICAL FOR YOUR DECISION:**
{portfolio_context}

**SELL CRITERIA — Recommend SELL when:**
- You hold the stock and it is down >5% from entry with no recovery signal
- You hold the stock and fundamentals are deteriorating (declining revenue, shrinking margins, rising debt)
- You hold the stock and technical indicators show bearish divergence or death cross patterns
- Risk significantly outweighs potential reward
- You hold the stock and it has gone up >15% — consider taking profits
- The investment thesis that justified the original purchase is no longer valid

**BUY CRITERIA — Recommend BUY only when:**
- You do NOT currently hold the stock
- Strong bullish thesis with multiple confirming signals
- Risk/reward ratio is favorable (at least 2:1)

End with a firm decision and always conclude your response with 'FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL**' to confirm your recommendation. Apply lessons from past decisions to strengthen your analysis. Here are reflections from similar situations you traded in and the lessons learned: {past_memory_str}""",
            },
            context,
        ]

        result = llm.invoke(messages)

        return {
            "messages": [result],
            "trader_investment_plan": result.content,
            "sender": name,
        }

    return functools.partial(trader_node, name="Trader")
