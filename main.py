from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.default_config import DEFAULT_CONFIG
from dotenv import load_dotenv
import os
import sys

# Load environment variables from .env file
load_dotenv()

# Validate required environment variables
if not os.getenv("OPENAI_API_KEY"):
    print("ERROR: OPENAI_API_KEY not set. Add it to .env or export it.")
    sys.exit(1)

# Create a custom config — using Z.AI GLM models
config = DEFAULT_CONFIG.copy()
config["llm_provider"] = "openai"
config["backend_url"] = os.getenv("OPENAI_BASE_URL", "https://open.bigmodel.cn/api/coding/paas/v4")
config["deep_think_llm"] = "glm-5.1"
config["quick_think_llm"] = "glm-5.1"
config["max_debate_rounds"] = 1
config["load_crypto_history"] = True

# Configure data vendors (default uses yfinance, no extra API keys needed)
config["data_vendors"] = {
    "core_stock_apis": "yfinance",
    "technical_indicators": "yfinance",
    "fundamental_data": "yfinance",
    "news_data": "yfinance",
}

# Initialize with custom config
ta = TradingAgentsGraph(debug=True, config=config)

# forward propagate — test with BTC and today's date
try:
    _, decision = ta.propagate("BTC", "2026-04-18")
    print(f"\nFinal Decision: {decision}")
except Exception as e:
    print(f"\nPipeline error: {e}")
    import traceback
    traceback.print_exc()
