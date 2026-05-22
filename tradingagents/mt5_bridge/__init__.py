"""MT5 + TradingView Bridge for TradingAgents.

Architecture:
  TradingView (Premium webhooks) -> Flask Receiver -> Signal Validator
    -> TradingAgents LLM Analysis -> MT5 Execution -> Discord Notification

The MT5 executor communicates with a Windows-side bridge server via HTTP,
since the MetaTrader5 Python package is Windows-only.
"""

from .webhook_receiver import create_webhook_app
from .mt5_executor import MT5HttpExecutor as MT5Executor
from .signal_validator import TradingViewSignalParser
from .pipeline import TradingPipeline

__all__ = [
    "create_webhook_app",
    "MT5Executor",
    "TradingViewSignalParser",
    "TradingPipeline",
]
