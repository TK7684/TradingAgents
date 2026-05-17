from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import MessagesState


def _last_value(left, right):
    """Reducer that keeps the last value written (enables parallel writes)."""
    return right


# Researcher team state
class InvestDebateState(TypedDict):
    bull_history: Annotated[
        str, "Bullish Conversation history"
    ]  # Bullish Conversation history
    bear_history: Annotated[
        str, "Bearish Conversation history"
    ]  # Bullish Conversation history
    history: Annotated[str, "Conversation history"]  # Conversation history
    current_response: Annotated[str, "Latest response"]  # Last response
    judge_decision: Annotated[str, "Final judge decision"]  # Last response
    count: Annotated[int, "Length of the current conversation"]  # Conversation length


# Risk management team state
class RiskDebateState(TypedDict):
    aggressive_history: Annotated[
        str, "Aggressive Agent's Conversation history"
    ]  # Conversation history
    conservative_history: Annotated[
        str, "Conservative Agent's Conversation history"
    ]  # Conversation history
    neutral_history: Annotated[
        str, "Neutral Agent's Conversation history"
    ]  # Conversation history
    history: Annotated[str, "Conversation history"]  # Conversation history
    latest_speaker: Annotated[str, "Analyst that spoke last"]
    current_aggressive_response: Annotated[
        str, "Latest response by the aggressive analyst"
    ]  # Last response
    current_conservative_response: Annotated[
        str, "Latest response by the conservative analyst"
    ]  # Last response
    current_neutral_response: Annotated[
        str, "Latest response by the neutral analyst"
    ]  # Last response
    judge_decision: Annotated[str, "Judge's decision"]
    count: Annotated[int, "Length of the current conversation"]  # Conversation length


class AgentState(MessagesState):
    company_of_interest: Annotated[str, _last_value]
    trade_date: Annotated[str, _last_value]

    sender: Annotated[str, _last_value]

    # research step
    market_report: Annotated[str, _last_value]
    sentiment_report: Annotated[str, _last_value]
    news_report: Annotated[str, _last_value]
    fundamentals_report: Annotated[str, _last_value]
    polymarket_report: Annotated[str, _last_value]

    # researcher team discussion step
    investment_debate_state: Annotated[InvestDebateState, _last_value]
    investment_plan: Annotated[str, _last_value]

    trader_investment_plan: Annotated[str, _last_value]

    # risk management team discussion step
    risk_debate_state: Annotated[RiskDebateState, _last_value]
    final_trade_decision: Annotated[str, _last_value]
