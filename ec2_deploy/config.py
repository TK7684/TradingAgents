"""Centralized TradingAgent configuration — Z.AI GLM models."""
import os
from tradingagents.default_config import DEFAULT_CONFIG

ZAI_API_KEY = os.getenv("ZHIPU_API_KEY", "") or os.getenv("OPENAI_API_KEY", "")
ZAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://open.bigmodel.cn/api/paas/v4")


def get_config(profile="default"):
    """Get config by profile name."""
    config = DEFAULT_CONFIG.copy()
    # Route through glm provider registry — upstream v0.3.x registers GLM as
    # OpenAI-compatible with the correct base_url, avoiding the Responses API
    # that native OpenAI uses (which would break our Z.AI endpoint).
    config["llm_provider"] = "glm"
    config["backend_url"] = ZAI_BASE_URL

    # Rate-limit-safe settings for Z.AI
    config["llm_timeout_seconds"] = 120      # 2min — Z.AI slow on complex prompts
    config["llm_max_retries"] = 2            # LangChain retries for transient errors
    config["llm_base_retry_delay"] = 4.0     # longer base delay

    if profile == "turbo":
        # glm-4.7 hangs on tool-calling (ReadTimeout on coding endpoint).
        # glm-5.1 handles tools correctly (~6s/call).
        config["deep_think_llm"] = "glm-5.1"
        config["quick_think_llm"] = "glm-5.1"
        config["max_debate_rounds"] = 0
        config["max_risk_discuss_rounds"] = 0
    elif profile == "deep":
        config["deep_think_llm"] = "glm-5.1"
        config["quick_think_llm"] = "glm-5.1"
        config["max_debate_rounds"] = 2
        config["max_risk_discuss_rounds"] = 2
    else:
        config["deep_think_llm"] = "glm-5.1"
        config["quick_think_llm"] = "glm-5.1"
        config["max_debate_rounds"] = 1
        config["max_risk_discuss_rounds"] = 1

    return config


# Watchlists by category
WATCHLIST_TECH = ["NVDA", "AAPL", "TSLA", "MSFT", "GOOGL", "AMZN", "META", "AMD", "NFLX", "PLTR"]
WATCHLIST_ETF = ["SPY", "QQQ", "IWM", "DIA", "ARKK"]
WATCHLIST_FINANCE = ["JPM", "GS", "V", "MA", "BRK-B"]
WATCHLIST_HEALTH = ["UNH", "JNJ", "LLY", "PFE", "ABBV"]
WATCHLIST_ENERGY = ["XOM", "CVX", "NEE", "ENPH", "FSLR"]
WATCHLIST_CRYPTO = [
    "BTC-USD", "ETH-USD", "SOL-USD", "AVAX-USD", "LINK-USD",
    "DOT-USD", "MATIC-USD", "ADA-USD", "XRP-USD", "DOGE-USD",
]

# Default watchlist (used by daily cron)
WATCHLIST = WATCHLIST_TECH

# All watchlists for full scan
ALL_WATCHLISTS = {
    "tech": WATCHLIST_TECH,
    "etf": WATCHLIST_ETF,
    "finance": WATCHLIST_FINANCE,
    "health": WATCHLIST_HEALTH,
    "energy": WATCHLIST_ENERGY,
    "crypto": WATCHLIST_CRYPTO,
}
