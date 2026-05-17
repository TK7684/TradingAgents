# TradingAgents/graph/setup.py

from typing import Any, Dict
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from tradingagents.agents import *
from tradingagents.agents.utils.agent_states import AgentState

from .conditional_logic import ConditionalLogic
from .analyst_subgraph import create_analyst_subgraph


class GraphSetup:
    """Handles the setup and configuration of the agent graph."""

    def __init__(
        self,
        quick_thinking_llm: Any,
        deep_thinking_llm: Any,
        tool_nodes: Dict[str, ToolNode],
        bull_memory,
        bear_memory,
        trader_memory,
        invest_judge_memory,
        portfolio_manager_memory,
        conditional_logic: ConditionalLogic,
    ):
        """Initialize with required components."""
        self.quick_thinking_llm = quick_thinking_llm
        self.deep_thinking_llm = deep_thinking_llm
        self.tool_nodes = tool_nodes
        self.bull_memory = bull_memory
        self.bear_memory = bear_memory
        self.trader_memory = trader_memory
        self.invest_judge_memory = invest_judge_memory
        self.portfolio_manager_memory = portfolio_manager_memory
        self.conditional_logic = conditional_logic

    def setup_graph(
        self, selected_analysts=["market", "social", "news", "fundamentals"]
    ):
        """Set up and compile the agent workflow graph with parallel analyst execution.

        The analyst nodes (market, social, news, fundamentals, polymarket) run
        concurrently using a fan-out/fan-in pattern. Each analyst is wrapped in
        its own subgraph that handles the tool-calling loop internally. All
        subgraphs start from START and their outputs are merged before the
        debate phase begins.

        Args:
            selected_analysts (list): List of analyst types to include. Options are:
                - "market": Market analyst
                - "social": Social media analyst
                - "news": News analyst
                - "fundamentals": Fundamentals analyst
                - "polymarket": Polymarket prediction market analyst
        """
        if len(selected_analysts) == 0:
            raise ValueError("Trading Agents Graph Setup Error: no analysts selected!")

        # ------------------------------------------------------------------
        # 1. Create analyst nodes and their tool-calling subgraphs
        # ------------------------------------------------------------------
        analyst_subgraphs = {}

        if "market" in selected_analysts:
            analyst_subgraphs["market"] = create_analyst_subgraph(
                analyst_type="market",
                analyst_node=create_market_analyst(self.quick_thinking_llm),
                tool_node=self.tool_nodes["market"],
            )

        if "social" in selected_analysts:
            analyst_subgraphs["social"] = create_analyst_subgraph(
                analyst_type="social",
                analyst_node=create_social_media_analyst(self.quick_thinking_llm),
                tool_node=self.tool_nodes["social"],
            )

        if "news" in selected_analysts:
            analyst_subgraphs["news"] = create_analyst_subgraph(
                analyst_type="news",
                analyst_node=create_news_analyst(self.quick_thinking_llm),
                tool_node=self.tool_nodes["news"],
            )

        if "fundamentals" in selected_analysts:
            analyst_subgraphs["fundamentals"] = create_analyst_subgraph(
                analyst_type="fundamentals",
                analyst_node=create_fundamentals_analyst(self.quick_thinking_llm),
                tool_node=self.tool_nodes["fundamentals"],
            )

        if "polymarket" in selected_analysts:
            analyst_subgraphs["polymarket"] = create_analyst_subgraph(
                analyst_type="polymarket",
                analyst_node=create_polymarket_analyst(self.quick_thinking_llm),
                tool_node=self.tool_nodes["polymarket"],
            )

        # ------------------------------------------------------------------
        # 2. Create researcher and manager nodes
        # ------------------------------------------------------------------
        bull_researcher_node = create_bull_researcher(
            self.quick_thinking_llm, self.bull_memory
        )
        bear_researcher_node = create_bear_researcher(
            self.quick_thinking_llm, self.bear_memory
        )
        research_manager_node = create_research_manager(
            self.deep_thinking_llm, self.invest_judge_memory
        )
        trader_node = create_trader(self.quick_thinking_llm, self.trader_memory)

        # Create risk analysis nodes
        aggressive_analyst = create_aggressive_debator(self.quick_thinking_llm)
        neutral_analyst = create_neutral_debator(self.quick_thinking_llm)
        conservative_analyst = create_conservative_debator(self.quick_thinking_llm)
        portfolio_manager_node = create_portfolio_manager(
            self.deep_thinking_llm, self.portfolio_manager_memory
        )

        # ------------------------------------------------------------------
        # 3. Build the parent workflow graph
        # ------------------------------------------------------------------
        workflow = StateGraph(AgentState)

        # -- Fan-out: add each analyst subgraph as a node --
        for analyst_type, subgraph in analyst_subgraphs.items():
            workflow.add_node(f"{analyst_type}_analyst_group", subgraph)

        # -- Sync node: clear messages after all parallel analysts finish --
        def clear_analyst_messages(state):
            """Clear accumulated analyst messages and keep only a placeholder.
            
            After parallel analyst execution, the messages list contains
            tool-call artifacts from all analysts. Since downstream nodes
            (Bull/Bear researchers, etc.) don't read messages — they use
            the report fields — we clear them to keep state clean.
            """
            from langchain_core.messages import HumanMessage, RemoveMessage
            messages = state["messages"]
            removal_operations = [RemoveMessage(id=m.id) for m in messages]
            placeholder = HumanMessage(content="Analysts complete. Proceeding to debate.")
            return {"messages": removal_operations + [placeholder]}

        workflow.add_node("clear_analyst_messages", clear_analyst_messages)

        # Add debate/risk nodes
        workflow.add_node("Bull Researcher", bull_researcher_node)
        workflow.add_node("Bear Researcher", bear_researcher_node)
        workflow.add_node("Research Manager", research_manager_node)
        workflow.add_node("Trader", trader_node)
        workflow.add_node("Aggressive Analyst", aggressive_analyst)
        workflow.add_node("Neutral Analyst", neutral_analyst)
        workflow.add_node("Conservative Analyst", conservative_analyst)
        workflow.add_node("Portfolio Manager", portfolio_manager_node)

        # ------------------------------------------------------------------
        # 4. Define edges — fan-out / fan-in pattern
        # ------------------------------------------------------------------

        # Fan-out: START -> all analyst subgraphs (runs in parallel)
        for analyst_type in analyst_subgraphs:
            workflow.add_edge(START, f"{analyst_type}_analyst_group")

        # Fan-in: all analyst subgraphs -> sync node -> debate
        for analyst_type in analyst_subgraphs:
            workflow.add_edge(f"{analyst_type}_analyst_group", "clear_analyst_messages")

        workflow.add_edge("clear_analyst_messages", "Bull Researcher")

        # Debate edges (unchanged)
        workflow.add_conditional_edges(
            "Bull Researcher",
            self.conditional_logic.should_continue_debate,
            {
                "Bear Researcher": "Bear Researcher",
                "Research Manager": "Research Manager",
            },
        )
        workflow.add_conditional_edges(
            "Bear Researcher",
            self.conditional_logic.should_continue_debate,
            {
                "Bull Researcher": "Bull Researcher",
                "Research Manager": "Research Manager",
            },
        )
        workflow.add_edge("Research Manager", "Trader")
        workflow.add_edge("Trader", "Aggressive Analyst")
        workflow.add_conditional_edges(
            "Aggressive Analyst",
            self.conditional_logic.should_continue_risk_analysis,
            {
                "Conservative Analyst": "Conservative Analyst",
                "Portfolio Manager": "Portfolio Manager",
            },
        )
        workflow.add_conditional_edges(
            "Conservative Analyst",
            self.conditional_logic.should_continue_risk_analysis,
            {
                "Neutral Analyst": "Neutral Analyst",
                "Portfolio Manager": "Portfolio Manager",
            },
        )
        workflow.add_conditional_edges(
            "Neutral Analyst",
            self.conditional_logic.should_continue_risk_analysis,
            {
                "Aggressive Analyst": "Aggressive Analyst",
                "Portfolio Manager": "Portfolio Manager",
            },
        )

        workflow.add_edge("Portfolio Manager", END)

        # Compile and return
        return workflow.compile()
