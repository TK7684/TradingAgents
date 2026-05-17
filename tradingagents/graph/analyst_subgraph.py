# TradingAgents/graph/analyst_subgraph.py
#
# Encapsulates each analyst's tool-calling loop into a self-contained subgraph
# for parallel (fan-out) execution in the parent LangGraph pipeline.
#
# Design:
#   - Each analyst gets its own compiled subgraph with an internal tool-calling loop.
#   - The subgraph reads the shared state, runs the analyst + tools loop, and
#     outputs ONLY the report field.
#   - The parent graph fans out from START to all analyst subgraphs concurrently,
#     then fans in to a sync node before the debate phase.
#
# This replaces the sequential analyst chain with a parallel fan-out/fan-in pattern,
# reducing total pipeline latency from sum(all_analysts) to max(single_analyst).

from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from tradingagents.agents.utils.agent_states import AgentState


def create_analyst_subgraph(
    analyst_type: str,
    analyst_node,
    tool_node: ToolNode,
):
    """Build a compiled subgraph for a single analyst with a full tool-calling loop.

    The subgraph internally runs:
        Analyst -> [tools -> Analyst]* -> END

    It only outputs the report field (e.g. "market_report") to the parent state,
    along with a minimal messages update (just the final analyst response).
    Tool-call messages stay internal and don't pollute the parent graph's state.

    Args:
        analyst_type: Key for naming nodes and state fields (e.g. "market").
        analyst_node: Callable(state) -> dict that runs the analyst LLM logic.
                      Must return {"messages": [...], "<analyst_type>_report": report}.
        tool_node: ToolNode with the tools this analyst can call.

    Returns:
        A compiled LangGraph subgraph.
    """

    def _should_continue(state):
        """Route to tools if the last message has tool_calls, otherwise finish."""
        messages = state["messages"]
        last_message = messages[-1]
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return f"tools_{analyst_type}"
        return END

    builder = StateGraph(AgentState)

    analyst_name = f"{analyst_type}_analyst"
    tools_name = f"tools_{analyst_type}"

    builder.add_node(analyst_name, analyst_node)
    builder.add_node(tools_name, tool_node)

    # Entry point -> analyst
    builder.add_edge(START, analyst_name)

    # Analyst conditionally loops through tools or finishes
    builder.add_conditional_edges(
        analyst_name,
        _should_continue,
        {tools_name: tools_name, END: END},
    )

    # After tools, loop back to analyst
    builder.add_edge(tools_name, analyst_name)

    return builder.compile()
