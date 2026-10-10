"""Tests for Polymarket analyst fan-out wiring.

Hypothesis: the Polymarket analyst existed but was unreachable (defaults
excluded it) and its report was never consumed downstream. These tests pin:
  1. production default includes the polymarket analyst
  2. initial state carries polymarket_report (no KeyError downstream)
  3. bull/bear researchers consume the report when present, ignore when empty
"""
import inspect


def test_default_analysts_include_polymarket():
    """The production graph must fan out to the polymarket analyst by default."""
    from tradingagents.graph.trading_graph import TradingAgentsGraph

    sig = inspect.signature(TradingAgentsGraph.__init__)
    default = sig.parameters["selected_analysts"].default
    assert "polymarket" in default, f"polymarket missing from defaults: {default}"


def test_setup_graph_default_includes_polymarket():
    from tradingagents.graph.setup import GraphSetup

    sig = inspect.signature(GraphSetup.setup_graph)
    default = sig.parameters["selected_analysts"].default
    assert "polymarket" in default, f"polymarket missing from setup defaults: {default}"


def test_initial_state_has_polymarket_report():
    from tradingagents.graph.propagation import Propagator

    state = Propagator().create_initial_state("BTC", "2026-01-01")
    assert state.get("polymarket_report") == ""


class _FakeMemory:
    def get_memories(self, situation, n_matches=2):
        self.last_situation = situation
        return []


class _FakeLLM:
    def __init__(self):
        self.last_prompt = None

    def invoke(self, prompt):
        self.last_prompt = prompt
        from langchain_core.messages import AIMessage

        return AIMessage(content="ok")


def _make_state(polymarket=""):
    from tradingagents.agents.utils.agent_states import InvestDebateState

    invest = InvestDebateState(
        {
            "bull_history": "",
            "bear_history": "",
            "history": "",
            "current_response": "",
            "judge_decision": "",
            "count": 0,
        }
    )
    return {
        "company_of_interest": "BTC",
        "market_report": "MR",
        "sentiment_report": "SR",
        "news_report": "NR",
        "fundamentals_report": "FR",
        "polymarket_report": polymarket,
        "investment_debate_state": invest,
        "market_context": "",
    }


def test_bull_researcher_consumes_polymarket_report():
    from tradingagents.agents.researchers.bull_researcher import create_bull_researcher

    llm = _FakeLLM()
    mem = _FakeMemory()
    node = create_bull_researcher(llm, mem)
    node(_make_state(polymarket="CROWD: 65% above 120k"))
    assert "CROWD: 65% above 120k" in llm.last_prompt, "bull prompt missing polymarket report"
    assert "CROWD: 65% above 120k" in mem.last_situation


def test_bull_researcher_ignores_empty_polymarket_report():
    from tradingagents.agents.researchers.bull_researcher import create_bull_researcher

    llm = _FakeLLM()
    node = create_bull_researcher(llm, _FakeMemory())
    node(_make_state(polymarket=""))
    assert "Polymarket" not in llm.last_prompt or "report: " not in llm.last_prompt


def test_bear_researcher_consumes_polymarket_report():
    from tradingagents.agents.researchers.bear_researcher import create_bear_researcher

    llm = _FakeLLM()
    node = create_bear_researcher(llm, _FakeMemory())
    node(_make_state(polymarket="CROWD: 70% below 50k"))
    assert "CROWD: 70% below 50k" in llm.last_prompt, "bear prompt missing polymarket report"
