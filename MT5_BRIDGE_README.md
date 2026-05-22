# TradingAgents MT5 Bridge - Setup Guide

## 🏗️ Architecture

```
┌─────────────────┐     webhook (JSON)     ┌──────────────────┐
│  TradingView     │ ──────────────────────▶│  Flask Receiver   │
│  Premium         │     POST :5001         │  (WSL)           │
│  (Your fav       │                         │  /webhook/tradingview
│   indicators)    │                         └────────┬─────────┘
└─────────────────┘                                  │
                                                     ▼
                                           ┌──────────────────┐
                                           │  Signal Queue     │
                                           │  (in-memory)      │
                                           └────────┬─────────┘
                                                    │
                                    ┌───────────────┼───────────────┐
                                    ▼               ▼               ▼
                            ┌───────────┐   ┌───────────┐   ┌───────────┐
                            │  Validate  │──▶│  LLM      │──▶│  Execute  │
                            │  Signal    │   │  Confirm  │   │  on MT5   │
                            │            │   │ (optional)│   │  or Paper │
                            └───────────┘   └───────────┘   └─────┬─────┘
                                                                      │
                                                                      ▼
                                                            ┌───────────────┐
                                                            │  Discord      │
                                                            │  #trading-    │
                                                            │  signals      │
                                                            └───────────────┘
```

## 📋 Prerequisites

### 1. Install MetaTrader 5 on Windows
1. Download from: https://www.metatrader5.com/en/download
2. Install on Windows (standard installation)
3. Open MT5 → Login to your broker (use DEMO account first!)
4. Enable Algo Trading: Tools → Options → Expert Advisors → ✓ Allow Algo Trading
5. Keep MT5 terminal running while using the bridge

### 2. Install Python Dependencies
```bash
cd ~/TradingAgents
source .venv/bin/activate
pip install flask MetaTrader5
```

### 3. Configure TradingView Webhooks
In TradingView, create alerts with this JSON template:

```json
{
  "symbol": "{{ticker}}",
  "side": "{{strategy.order.action}}",
  "price": "{{close}}",
  "time": "{{timenow}}",
  "interval": "{{interval}}",
  "strategy": "MyStrategy",
  "volume": "{{volume}}"
}
```

**Webhook URL:** `http://localhost:5001/webhook/tradingview`

### 4. (Optional) Custom Pine Script Alert
Add this to your Pine Script strategy:
```pine
//@version=5
strategy("My Strategy", overlay=true)

// Your indicator logic here
// ...

// Custom alert message with JSON
alert_message = '{"symbol": "' + syminfo.tickerid + '", ' +
    '"side": "' + (strategy.position_size > 0 ? "buy" : "sell") + '", ' +
    '"price": "' + str.tostring(close) + '", ' +
    '"strategy": "MyStrategy", ' +
    '"timeframe": "' + timeframe.period + '"}'

alertcondition(condition, title="Trade Signal", message=alert_message)
```

## 🚀 Running

### Paper Mode (Safe Testing)
```bash
cd ~/TradingAgents
python run_bridge.py --mode paper --port 5001
```

### Live Mode (Real Trades on MT5)
```bash
python run_bridge.py --mode live --lot 0.01
```

### With LLM Confirmation
```bash
python run_bridge.py --mode confirm --lot 0.01
```

### Background Daemon (PM2)
```bash
pm2 start run_bridge.py --name "mt5-bridge" -- \
    --mode paper --port 5001 --interval 5

pm2 start run_bridge.py --name "mt5-bridge-live" -- \
    --mode live --lot 0.01 --interval 3
```

## 🔑 Environment Variables (.env)

```env
# TradingView webhook
TV_WEBHOOK_SECRET=your-secret-here

# Discord notifications
DISCORD_WEBHOOK_URL=https://discord.com/api/webhooks/...
# OR use bot token:
DISCORD_TOKEN=your-bot-token
DISCORD_TRADE_CHANNEL=your-channel-id

# LLM (for confirm mode)
OPENAI_API_KEY=your-zai-key
OPENAI_BASE_URL=https://open.bigmodel.cn/api/coding/paas/v4

# Bridge config
BRIDGE_MODE=paper
```

## 📊 Monitoring

| Endpoint | Description |
|---|---|
| `GET /health` | Server health + queue size |
| `GET /signals` | List pending signals |
| `POST /signals/next` | Dequeue next signal |
| `POST /signals/clear` | Clear signal queue |

## 🧪 Testing

```bash
# Send a test signal via curl:
curl -X POST http://localhost:5001/webhook/tradingview \
  -H "Content-Type: application/json" \
  -d '{
    "symbol": "EURUSD",
    "side": "buy",
    "price": 1.0850,
    "strategy": "test",
    "timeframe": "1H"
  }'

# Check health:
curl http://localhost:5001/health

# View signals:
curl http://localhost:5001/signals
```

## ⚠️ Risk Controls

- **Daily trade limit:** 50 trades/day (configurable with `--max-daily`)
- **Position dedup:** Won't open duplicate positions on same symbol
- **Paper mode:** No real money at risk — always test first!
- **LLM confirmation:** Optional second brain to validate signals
- **Trade logging:** All trades logged to `results/trades/bridge_trades.jsonl`

## 🔗 Integration with Existing TradingAgents

The bridge complements your existing setup:
- **TradingAgents daemon** (`ec2_deploy/trading_daemon.py`) — runs multi-agent LLM analysis on schedule
- **Paper trader** (`ec2_deploy/paper_trader.py`) — tracks simulated portfolio
- **Discord signals** (`ec2_deploy/discord_signal.py`) — posts to #trading-signals

The bridge adds:
1. **Real-time signal ingestion** from TradingView
2. **Live execution** via MT5 (forex, metals, indices)
3. **Hybrid LLM + indicator** confirmation layer
