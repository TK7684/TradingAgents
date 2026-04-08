"""Centralized TradingAgent configuration."""
import os
from tradingagents.default_config import DEFAULT_CONFIG

def get_config(profile="default"):
    """Get config by profile name."""
    config = DEFAULT_CONFIG.copy()
    config["llm_provider"] = "anthropic"
    config["backend_url"] = os.getenv("ZAI_BASE_URL", "https://api.z.ai/api/anthropic")

    if profile == "turbo":
        config["deep_think_llm"] = "glm-4.5-air"
        config["quick_think_llm"] = "glm-4.5-air"
        config["max_debate_rounds"] = 0
        config["max_risk_discuss_rounds"] = 0
    elif profile == "deep":
        config["deep_think_llm"] = "glm-4.7"
        config["quick_think_llm"] = "glm-4.7"
        config["max_debate_rounds"] = 2
        config["max_risk_discuss_rounds"] = 2
    else:
        config["deep_think_llm"] = "glm-4.7"
        config["quick_think_llm"] = "glm-4.5-air"
        config["max_debate_rounds"] = 1
        config["max_risk_discuss_rounds"] = 1

    return config


# Watchlists by category
WATCHLIST_TECH = ["NVDA", "AAPL", "TSLA", "MSFT", "GOOGL", "AMZN", "META", "AMD", "NFLX", "PLTR"]
WATCHLIST_ETF = ["SPY", "QQQ", "IWM", "DIA", "ARKK"]
WATCHLIST_FINANCE = ["JPM", "GS", "V", "MA", "BRK-B"]
WATCHLIST_HEALTH = ["UNH", "JNJ", "LLY", "PFE", "ABBV"]
WATCHLIST_ENERGY = ["XOM", "CVX", "NEE", "ENPH", "FSLR"]

# Default watchlist (used by daily cron)
WATCHLIST = WATCHLIST_TECH

# All watchlists for full scan
ALL_WATCHLISTS = {
    "tech": WATCHLIST_TECH,
    "etf": WATCHLIST_ETF,
    "finance": WATCHLIST_FINANCE,
    "health": WATCHLIST_HEALTH,
    "energy": WATCHLIST_ENERGY,
}
