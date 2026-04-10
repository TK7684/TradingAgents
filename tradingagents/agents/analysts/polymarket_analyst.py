"""Polymarket Prediction Market Analyst.

Analyzes crowd-sourced probability estimates from Polymarket to assess
market sentiment and price target consensus for trading decisions.
"""

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from tradingagents.agents.utils.agent_utils import (
    build_instrument_context,
    get_language_instruction,
)
from tradingagents.agents.utils.polymarket_tools import (
    get_polymarket_odds,
    get_polymarket_sentiment,
)


def create_polymarket_analyst(llm):
    def polymarket_analyst_node(state):
        current_date = state["trade_date"]
        instrument_context = build_instrument_context(state["company_of_interest"])

        tools = [
            get_polymarket_odds,
            get_polymarket_sentiment,
        ]

        system_message = (
            "You are a prediction market analyst specializing in Polymarket data. "
            "Your job is to analyze crowd-sourced probability estimates to gauge market sentiment "
            "and consensus on price targets. "
            "Use get_polymarket_odds(ticker, curr_date) to fetch active prediction markets "
            "and their implied probabilities. "
            "Use get_polymarket_sentiment(ticker, curr_date) for aggregated bullish/bearish sentiment. "
            "Provide analysis on: "
            "1) What the crowd thinks will happen (consensus probabilities) "
            "2) Where the crowd sees price targets (high-probability outcomes) "
            "3) How strong the conviction is (volume, liquidity) "
            "4) Any divergence between prediction market odds and current market trends "
            "Remember: prediction market prices = implied probabilities. "
            "$0.65 means 65% crowd-estimated probability of that outcome. "
            "High volume means more participants with skin in the game."
            """ Make sure to append a Markdown table at the end of the report to organize key findings."""
            + get_language_instruction()
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a helpful AI assistant, collaborating with other assistants."
                    " Use the provided tools to progress towards answering the question."
                    " If you are unable to fully answer, that's OK; another assistant with different tools"
                    " will help where you left off. Execute what you can to make progress."
                    " If you or any other assistant has the FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL** or deliverable,"
                    " prefix your response with FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL** so the team knows to stop."
                    " You have access to the following tools: {tool_names}.\n{system_message}"
                    "For your reference, the current date is {current_date}. {instrument_context}",
                ),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )

        prompt = prompt.partial(system_message=system_message)
        prompt = prompt.partial(tool_names=", ".join([tool.name for tool in tools]))
        prompt = prompt.partial(current_date=current_date)
        prompt = prompt.partial(instrument_context=instrument_context)

        chain = prompt | llm.bind_tools(tools)
        result = chain.invoke(state["messages"])

        report = ""

        if len(result.tool_calls) == 0:
            report = result.content

        return {
            "messages": [result],
            "polymarket_report": report,
        }

    return polymarket_analyst_node
