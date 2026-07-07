# TradingAgents/graph/trading_graph.py

import logging
import os
from pathlib import Path
import json
from datetime import date
from typing import Dict, Any, Tuple, List, Optional

from langgraph.prebuilt import ToolNode

log = logging.getLogger(__name__)

from tradingagents.llm_clients import create_llm_client

from tradingagents.agents import *
from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.agents.utils.memory import FinancialSituationMemory
from tradingagents.agents.utils.agent_states import (
    AgentState,
    InvestDebateState,
    RiskDebateState,
)
from tradingagents.dataflows.config import set_config

# Import the new abstract tool methods from agent_utils
from tradingagents.agents.utils.agent_utils import (
    get_stock_data,
    get_indicators,
    get_fundamentals,
    get_balance_sheet,
    get_cashflow,
    get_income_statement,
    get_news,
    get_insider_transactions,
    get_global_news
)
from tradingagents.agents.utils.polymarket_tools import (
    get_polymarket_odds,
    get_polymarket_sentiment,
)

from .conditional_logic import ConditionalLogic
from .setup import GraphSetup
from .propagation import Propagator
from .reflection import Reflector
from .signal_processing import SignalProcessor
from .consensus import ConsensusEngine


class TradingAgentsGraph:
    """Main class that orchestrates the trading agents framework."""

    def __init__(
        self,
        selected_analysts=["market", "social", "news", "fundamentals"],
        debug=False,
        config: Dict[str, Any] = None,
        callbacks: Optional[List] = None,
    ):
        """Initialize the trading agents graph and components.

        Args:
            selected_analysts: List of analyst types to include
            debug: Whether to run in debug mode
            config: Configuration dictionary. If None, uses default config
            callbacks: Optional list of callback handlers (e.g., for tracking LLM/tool stats)
        """
        self.debug = debug
        self.config = config or DEFAULT_CONFIG
        self.callbacks = callbacks or []

        # Update the interface's config
        set_config(self.config)

        # Create necessary directories
        os.makedirs(
            os.path.join(self.config["project_dir"], "dataflows/data_cache"),
            exist_ok=True,
        )

        # Initialize LLMs with provider-specific thinking configuration
        llm_kwargs = self._get_provider_kwargs()

        # Add callbacks to kwargs if provided (passed to LLM constructor)
        if self.callbacks:
            llm_kwargs["callbacks"] = self.callbacks

        deep_client = create_llm_client(
            provider=self.config["llm_provider"],
            model=self.config["deep_think_llm"],
            base_url=self.config.get("backend_url"),
            **llm_kwargs,
        )
        quick_client = create_llm_client(
            provider=self.config["llm_provider"],
            model=self.config["quick_think_llm"],
            base_url=self.config.get("backend_url"),
            **llm_kwargs,
        )

        self.deep_thinking_llm = deep_client.get_llm()
        self.quick_thinking_llm = quick_client.get_llm()
        
        # Initialize memories
        self.bull_memory = FinancialSituationMemory("bull_memory", self.config)
        self.bear_memory = FinancialSituationMemory("bear_memory", self.config)
        self.trader_memory = FinancialSituationMemory("trader_memory", self.config)
        self.invest_judge_memory = FinancialSituationMemory("invest_judge_memory", self.config)
        self.portfolio_manager_memory = FinancialSituationMemory("portfolio_manager_memory", self.config)

        # Append-only decision log (v0.3.x) — provides past_context for PM
        from tradingagents.agents.utils.memory import TradingMemoryLog
        self.memory_log = TradingMemoryLog(self.config)

        # Lazy-load crypto trade history flag
        self._crypto_trades_loaded = False

        # Create tool nodes
        self.tool_nodes = self._create_tool_nodes()

        # Initialize components
        self.conditional_logic = ConditionalLogic(
            max_debate_rounds=self.config["max_debate_rounds"],
            max_risk_discuss_rounds=self.config["max_risk_discuss_rounds"],
        )
        self.graph_setup = GraphSetup(
            self.quick_thinking_llm,
            self.deep_thinking_llm,
            self.tool_nodes,
            self.bull_memory,
            self.bear_memory,
            self.trader_memory,
            self.invest_judge_memory,
            self.portfolio_manager_memory,
            self.conditional_logic,
        )

        self.propagator = Propagator()
        self.reflector = Reflector(self.quick_thinking_llm)
        self.signal_processor = SignalProcessor(self.quick_thinking_llm)
        self.consensus_engine = ConsensusEngine()

        # State tracking
        self.curr_state = None
        self.ticker = None
        self.log_states_dict = {}  # date to full state dict

        # Set up the graph
        self.graph = self.graph_setup.setup_graph(selected_analysts)

    def _get_provider_kwargs(self) -> Dict[str, Any]:
        """Get provider-specific kwargs for LLM client creation."""
        kwargs = {}
        provider = self.config.get("llm_provider", "").lower()

        # Forward timeout from config to prevent hangs
        if "llm_timeout_seconds" in self.config:
            kwargs["timeout"] = self.config["llm_timeout_seconds"]

        # LangChain retry for transient errors (timeouts, 429s).
        # Keep low (2) — our rate_limiter prevents burst 429s,
        # and inter-ticker delays prevent cross-ticker exhaustion.
        kwargs["max_retries"] = 2

        if provider == "google":
            thinking_level = self.config.get("google_thinking_level")
            if thinking_level:
                kwargs["thinking_level"] = thinking_level

        elif provider == "openai":
            reasoning_effort = self.config.get("openai_reasoning_effort")
            if reasoning_effort:
                kwargs["reasoning_effort"] = reasoning_effort

        elif provider == "anthropic":
            effort = self.config.get("anthropic_effort")
            if effort:
                kwargs["effort"] = effort

        return kwargs

    def _create_tool_nodes(self) -> Dict[str, ToolNode]:
        """Create tool nodes for different data sources using abstract methods."""
        return {
            "market": ToolNode(
                [
                    # Core stock data tools
                    get_stock_data,
                    # Technical indicators
                    get_indicators,
                ]
            ),
            "social": ToolNode(
                [
                    # News tools for social media analysis
                    get_news,
                ]
            ),
            "news": ToolNode(
                [
                    # News and insider information
                    get_news,
                    get_global_news,
                    get_insider_transactions,
                ]
            ),
            "fundamentals": ToolNode(
                [
                    # Fundamental analysis tools
                    get_fundamentals,
                    get_balance_sheet,
                    get_cashflow,
                    get_income_statement,
                ]
            ),
            "polymarket": ToolNode(
                [
                    # Prediction market tools
                    get_polymarket_odds,
                    get_polymarket_sentiment,
                ]
            ),
        }

    def _load_crypto_trades(self):
        """Load crypto trade history from database if configured."""
        try:
            from tradingagents.agents.utils.crypto_memory_loader import CryptoTradeHistoryLoader
            db_path = self.config.get("crypto_history_db", "/home/tk578/hyperliquid-dex/trading.db")
            loader = CryptoTradeHistoryLoader(db_path)
            trades = loader.load_closed_trades(limit=200)
            if trades:
                self.bull_memory.add_situations(trades)
                self.bear_memory.add_situations(trades)
                self.trader_memory.add_situations(trades)
                import logging
                logging.getLogger(__name__).info(f"Loaded {len(trades)} crypto trade memories")
        except Exception as e:
            import logging
            logging.getLogger(__name__).error(f"Failed to load crypto trades: {e}")

    def propagate(self, company_name, trade_date, market_context=""):
        """Run the trading agents graph for a company on a specific date.

        Args:
            company_name: Ticker symbol or company name
            trade_date: Analysis date string
            market_context: Optional real-time market context text from TradingView
        """

        self.ticker = company_name

        # Lazy-load crypto history on first propagate call if configured
        if not self._crypto_trades_loaded and self.config.get("load_crypto_history", False):
            self._load_crypto_trades()
            self._crypto_trades_loaded = True

        # Initialize state (with optional TradingView market context)
        init_agent_state = self.propagator.create_initial_state(
            company_name, trade_date, market_context=market_context
        )
        args = self.propagator.get_graph_args()

        # Always use stream() — invoke() deadlocks with subgraphs on langgraph 1.1.8.
        # stream() collects all node outputs; the last chunk is the final merged state.
        final_state = None
        for chunk in self.graph.stream(init_agent_state, **args):
            final_state = chunk
            if self.debug and chunk.get("messages"):
                chunk["messages"][-1].pretty_print()

        if final_state is None:
            raise RuntimeError("Graph stream produced no output")

        # Store current state for reflection
        self.curr_state = final_state

        # Log state
        self._log_state(trade_date, final_state)

        # Return decision and processed signal
        signal = self.process_signal(final_state["final_trade_decision"])

        # Run consensus voting across all 4 decision points
        consensus_results = self.consensus_engine.evaluate(self.log_states_dict)
        # Get the result for this trade_date
        consensus = consensus_results.get(str(trade_date))
        consensus_data = None
        if consensus:
            consensus_data = {
                "confidence": consensus.confidence,
                "recommendation": consensus.recommendation,
                "source_signals": consensus.source_signals,
                "unanimous": consensus.unanimous,
            }
            log.info("Consensus: %s (confidence=%.2f, unanimous=%s)",
                     consensus.recommendation, consensus.confidence, consensus.unanimous)

        return final_state, signal, consensus_data

    def _log_state(self, trade_date, final_state):
        """Log the final state to a JSON file."""
        self.log_states_dict[str(trade_date)] = {
            "company_of_interest": final_state["company_of_interest"],
            "trade_date": final_state["trade_date"],
            "market_report": final_state["market_report"],
            "sentiment_report": final_state["sentiment_report"],
            "news_report": final_state["news_report"],
            "fundamentals_report": final_state["fundamentals_report"],
            "polymarket_report": final_state.get("polymarket_report", ""),
            "investment_debate_state": {
                "bull_history": final_state["investment_debate_state"]["bull_history"],
                "bear_history": final_state["investment_debate_state"]["bear_history"],
                "history": final_state["investment_debate_state"]["history"],
                "current_response": final_state["investment_debate_state"][
                    "current_response"
                ],
                "judge_decision": final_state["investment_debate_state"][
                    "judge_decision"
                ],
            },
            "trader_investment_decision": final_state["trader_investment_plan"],
            "risk_debate_state": {
                "aggressive_history": final_state["risk_debate_state"]["aggressive_history"],
                "conservative_history": final_state["risk_debate_state"]["conservative_history"],
                "neutral_history": final_state["risk_debate_state"]["neutral_history"],
                "history": final_state["risk_debate_state"]["history"],
                "judge_decision": final_state["risk_debate_state"]["judge_decision"],
            },
            "investment_plan": final_state["investment_plan"],
            "final_trade_decision": final_state["final_trade_decision"],
        }

        # Save to file
        directory = Path(self.config["results_dir"]) / self.ticker / "TradingAgentsStrategy_logs"
        directory.mkdir(parents=True, exist_ok=True)

        log_path = directory / f"full_states_log_{trade_date}.json"
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(self.log_states_dict[str(trade_date)], f, indent=4)

    def reflect_and_remember(self, returns_losses):
        """Reflect on decisions and update memory based on returns."""
        self.reflector.reflect_bull_researcher(
            self.curr_state, returns_losses, self.bull_memory
        )
        self.reflector.reflect_bear_researcher(
            self.curr_state, returns_losses, self.bear_memory
        )
        self.reflector.reflect_trader(
            self.curr_state, returns_losses, self.trader_memory
        )
        self.reflector.reflect_invest_judge(
            self.curr_state, returns_losses, self.invest_judge_memory
        )
        self.reflector.reflect_portfolio_manager(
            self.curr_state, returns_losses, self.portfolio_manager_memory
        )

    def process_signal(self, full_signal):
        """Process a signal to extract the core decision."""
        return self.signal_processor.process_signal(full_signal)
