"""Test GLM-5.1 tool calling support via Z.AI endpoint."""

import os
from dotenv import load_dotenv
from langchain_core.tools import tool

load_dotenv()


@tool
def get_price(symbol: str) -> str:
    """Get the current price of a stock or crypto symbol."""
    return f"{symbol}: $100.00"


def test_tool_calling():
    from tradingagents.llm_clients import create_llm_client

    client = create_llm_client(
        provider="openai",
        model="glm-5.1",
        base_url=os.getenv("OPENAI_BASE_URL", "https://open.bigmodel.cn/api/coding/paas/v4"),
    )
    llm = client.get_llm()

    print("1. Testing basic invoke...")
    try:
        result = llm.invoke([("human", "Say 'hello' and nothing else.")])
        print(f"   OK: {result.content[:100]}")
    except Exception as e:
        print(f"   FAILED: {e}")
        return

    print("\n2. Testing bind_tools()...")
    try:
        bound = llm.bind_tools([get_price])
        print("   bind_tools() succeeded")
    except Exception as e:
        print(f"   bind_tools() FAILED: {e}")
        print("   -> Tool-binding fallback IS needed (Phase 1B)")
        return

    print("\n3. Testing tool call invocation...")
    try:
        result = bound.invoke([("human", "What is the price of BTC?")])
        if hasattr(result, 'tool_calls') and result.tool_calls:
            print(f"   Tool calls: {result.tool_calls}")
            print("   -> Full tool calling SUPPORTED")
        else:
            print(f"   No tool_calls in response. Content: {result.content[:200]}")
            print("   -> Model responds but doesn't use tools. Fallback recommended.")
    except Exception as e:
        print(f"   Tool call invocation FAILED: {e}")
        print("   -> Tool-binding fallback IS needed (Phase 1B)")


if __name__ == "__main__":
    test_tool_calling()
