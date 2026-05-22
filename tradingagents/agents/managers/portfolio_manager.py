from tradingagents.agents.utils.agent_utils import build_instrument_context, get_language_instruction
from tradingagents.agents.utils.portfolio_context import get_portfolio_context


def create_portfolio_manager(llm, memory):
    def portfolio_manager_node(state) -> dict:

        company_name = state["company_of_interest"]
        ticker = company_name.split(" (")[0].split("(")[0].strip() if " (" in company_name else company_name.strip()
        instrument_context = build_instrument_context(company_name)

        # Get portfolio context — critical for sell decisions
        portfolio_context = get_portfolio_context(ticker)

        history = state["risk_debate_state"]["history"]
        risk_debate_state = state["risk_debate_state"]
        market_research_report = state["market_report"]
        news_report = state["news_report"]
        fundamentals_report = state["fundamentals_report"]
        sentiment_report = state["sentiment_report"]
        research_plan = state["investment_plan"]
        trader_plan = state["trader_investment_plan"]

        curr_situation = f"{market_research_report}\n\n{sentiment_report}\n\n{news_report}\n\n{fundamentals_report}"
        past_memories = memory.get_memories(curr_situation, n_matches=2)

        past_memory_str = ""
        for i, rec in enumerate(past_memories, 1):
            past_memory_str += rec["recommendation"] + "\n\n"

        prompt = f"""As the Portfolio Manager, synthesize the risk analysts' debate and deliver the final trading decision.

{instrument_context}

**PORTFOLIO CONTEXT — YOU MUST CONSIDER THIS:**
{portfolio_context}

---

**Rating Scale** (use exactly one):
- **Buy**: Strong conviction to enter NEW position (only if no existing position)
- **Overweight**: Favorable outlook, gradually increase exposure
- **Hold**: Maintain current position, no action needed
- **Underweight**: Reduce exposure, take partial profits
- **Sell**: Exit position or avoid entry

**EXPLICIT SELL CRITERIA — You MUST rate Sell or Underweight if ANY apply:**
- We hold this stock and it is down >5% from average entry price
- Fundamentals deteriorating: declining revenue, shrinking margins, rising debt, negative earnings revisions
- Technicals show bearish divergence (price lower highs vs RSI/MACD higher highs, death cross on moving averages)
- Bear case significantly outweighs bull case in the debate
- Negative catalysts present: regulatory, competitive, management, or sector headwinds
- We hold this stock and the original investment thesis is no longer valid
- If no strong bullish thesis exists and multiple risk factors are present, prefer Sell over Hold

**TAKE-PROFIT CRITERIA — Consider Sell/Underweight if:**
- We hold this stock and it is up >15% from entry — lock in gains
- Risk/reward no longer favors holding (limited upside, significant downside risk)

**Context:**
- Research Manager's investment plan: **{research_plan}**
- Trader's transaction proposal: **{trader_plan}**
- Lessons from past decisions: **{past_memory_str}**

**Required Output Structure:**
1. **Rating**: State one of Buy / Overweight / Hold / Underweight / Sell.
2. **Executive Summary**: A concise action plan covering entry strategy, position sizing, key risk levels, and time horizon.
3. **Investment Thesis**: Detailed reasoning anchored in the analysts' debate and past reflections.
4. **Sell Analysis** (required): Explicitly state whether any sell criteria are triggered and why or why not.

---

**Risk Analysts Debate History:**
{history}

---

Be decisive and ground every conclusion in specific evidence from the analysts. Do NOT default to Buy or Hold — if risk factors are present, favor Sell/Underweight to protect capital.{get_language_instruction()}"""

        response = llm.invoke(prompt)

        new_risk_debate_state = {
            "judge_decision": response.content,
            "history": risk_debate_state["history"],
            "aggressive_history": risk_debate_state["aggressive_history"],
            "conservative_history": risk_debate_state["conservative_history"],
            "neutral_history": risk_debate_state["neutral_history"],
            "latest_speaker": "Judge",
            "current_aggressive_response": risk_debate_state["current_aggressive_response"],
            "current_conservative_response": risk_debate_state["current_conservative_response"],
            "current_neutral_response": risk_debate_state["current_neutral_response"],
            "count": risk_debate_state["count"],
        }

        return {
            "risk_debate_state": new_risk_debate_state,
            "final_trade_decision": response.content,
        }

    return portfolio_manager_node
