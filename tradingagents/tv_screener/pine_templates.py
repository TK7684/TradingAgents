"""
Pine Script Template Library for TradingView Premium Webhook Alerts.

These templates connect TradingView chart alerts directly to the
MT5 bridge webhook receiver (port 5001).

HOW TO USE:
1. Open TradingView → Pine Editor
2. Paste the template you need
3. Add to chart
4. Create alert → set webhook URL to http://localhost:5001/webhook/tradingview
5. Set message template to the JSON shown in each template
"""

# === TEMPLATE 1: Simple Moving Average Crossover ===
# Buy when fast EMA crosses above slow EMA, sell on cross below.
# Best for: Swing trading, medium-term trends.

PINE_EMA_CROSSOVER = '''
//@version=5
strategy("EMA Crossover Signal", overlay=true, default_qty_type=strategy.percent_of_equity, default_qty_value=10)

// Inputs
fast_len = input.int(9, "Fast EMA Length", minval=1)
slow_len = input.int(21, "Slow EMA Length", minval=1)
atr_mult = input.float(1.5, "ATR Stop Loss Multiplier", minval=0.1, step=0.1)
rr_ratio = input.float(2.0, "Risk:Reward Ratio", minval=0.5, step=0.1)

// Calculations
fast_ema = ta.ema(close, fast_len)
slow_ema = ta.ema(close, slow_len)
atr = ta.atr(14)

// Signals
bull_cross = ta.crossover(fast_ema, slow_ema)
bear_cross = ta.crossunder(fast_ema, slow_ema)

var float sl = na
var float tp = na

if bull_cross and strategy.position_size == 0
    strategy.entry("Long", strategy.long)
    sl := close - atr * atr_mult
    tp := close + (close - sl) * rr_ratio
    strategy.exit("Exit Long", "Long", stop=sl, limit=tp)

if bear_cross
    strategy.close("Long")

// Plot
plot(fast_ema, color=color.new(color.aqua, 0), linewidth=2)
plot(slow_ema, color=color.new(color.orange, 0), linewidth=2)

// Alert message for webhook
alert_message = '{"symbol": "' + syminfo.tickerid + '", ' +
    '"side": "' + (bull_cross ? "buy" : bear_cross ? "sell" : "hold") + '", ' +
    '"price": "' + str.tostring(close) + '", ' +
    '"sl": "' + str.tostring(sl) + '", ' +
    '"tp": "' + str.tostring(tp) + '", ' +
    '"strategy": "EMA_Cross_' + str.tostring(fast_len) + '_' + str.tostring(slow_len) + '", ' +
    '"timeframe": "' + timeframe.period + '"}'

alertcondition(bull_cross or bear_cross, title="EMA Cross Signal", message=alert_message)
'''

# === TEMPLATE 2: RSI Divergence + BB Squeeze ===
# Combines RSI oversold with Bollinger Band squeeze for high-probability entries.
# Best for: Reversal trading, mean reversion.

PINE_RSI_BB = '''
//@version=5
strategy("RSI + BB Squeeze Signal", overlay=true)

// Inputs
rsi_len = input.int(14, "RSI Length")
rsi_oversold = input.int(30, "RSI Oversold Level")
bb_len = input.int(20, "BB Length")
bb_mult = input.float(2.0, "BB StdDev")
atr_mult = input.float(2.0, "SL ATR Multiplier")

// Calculations
rsi = ta.rsi(close, rsi_len)
[middle, upper, lower] = ta.bb(close, bb_len, bb_mult)
bb_width = (upper - lower) / middle
atr = ta.atr(14)

// Conditions
rsi_oversold = rsi < rsi_oversold
bb_squeeze = bb_width < ta.sma(bb_width, 50) * 0.8
volume_spike = volume > ta.sma(volume, 20) * 1.5

buy_signal = rsi_oversold and bb_squeeze and ta.barssince(rsi_oversold) == 0
sell_signal = rsi > 70 and ta.crossover(rsi, 70)

var float entry_price = na
if buy_signal and strategy.position_size == 0
    strategy.entry("Long", strategy.long)
    entry_price := close
    strategy.exit("Exit", "Long", stop=close - atr * atr_mult, limit=close + atr * atr_mult * 3)

if sell_signal
    strategy.close("Long")

// Plots
plot(upper, color=color.red, linewidth=1)
plot(lower, color=color.green, linewidth=1)
plot(middle, color=color.gray, linewidth=1)

alert_message = '{"symbol": "' + syminfo.tickerid + '", ' +
    '"side": "' + (buy_signal ? "buy" : sell_signal ? "sell" : "hold") + '", ' +
    '"price": "' + str.tostring(close) + '", ' +
    '"rsi": "' + str.tostring(rsi, "#.1") + '", ' +
    '"bb_width": "' + str.tostring(bb_width, "#.##") + '", ' +
    '"strategy": "RSI_BB_Squeeze", ' +
    '"timeframe": "' + timeframe.period + '"}'

alertcondition(buy_signal or sell_signal, title="RSI+BB Signal", message=alert_message)
'''

# === TEMPLATE 3: Volume Breakout Detector ===
# Detects unusual volume + price breakout above resistance.
# Best for: Momentum trading, breakout entries.

PINE_VOLUME_BREAKOUT = '''
//@version=5
strategy("Volume Breakout Signal", overlay=true)

// Inputs
vol_mult = input.float(3.0, "Volume Multiplier", minval=1.0, step=0.1)
lookback = input.int(20, "Resistance Lookback", minval=5)
atr_mult = input.float(1.5, "SL ATR Mult", minval=0.5, step=0.1)
min_change = input.float(1.0, "Min Change %", minval=0.0, step=0.1)

// Calculations
rel_vol = volume / ta.sma(volume, 20)
resistance = ta.highest(high, lookback)
atr = ta.atr(14)
pct_change = (close - close[1]) / close[1] * 100

// Conditions
vol_breakout = rel_vol > vol_mult
price_breakout = close > resistance[1]  // broke above yesterday's resistance
momentum = pct_change > min_change

buy_signal = vol_breakout and price_breakout and momentum and strategy.position_size == 0
sell_signal = close < ta.lowest(low, 10)[1]  // break below recent support

if buy_signal
    strategy.entry("Long", strategy.long)
    strategy.exit("Exit", "Long", stop=close - atr * atr_mult)

if sell_signal
    strategy.close("Long")

// Visual
resistance_line = ta.valuewhen(buy_signal, resistance, 0)
plot(buy_signal ? close : na, style=plot.style.triangleup, size=size.small, color=color.green, title="Buy")
bgcolor(buy_signal ? color.new(color.green, 90) : na)

alert_message = '{"symbol": "' + syminfo.tickerid + '", ' +
    '"side": "' + (buy_signal ? "buy" : sell_signal ? "sell" : "hold") + '", ' +
    '"price": "' + str.tostring(close) + '", ' +
    '"rel_volume": "' + str.tostring(rel_vol, "#.1") + '", ' +
    '"resistance": "' + str.tostring(resistance) + '", ' +
    '"strategy": "Volume_Breakout", ' +
    '"timeframe": "' + timeframe.period + '"}'

alertcondition(buy_signal or sell_signal, title="Volume Breakout", message=alert_message)
'''

# === TEMPLATE 4: Multi-Indicator Confluence ===
# Combines EMA trend + RSI + MACD for high-confidence signals.
# Best for: Filtered swing trades, conservative entries.

PINE_CONFLUENCE = '''
//@version=5
strategy("Multi-Indicator Confluence", overlay=true)

// Inputs
ema_fast = input.int(12, "EMA Fast")
ema_slow = input.int(26, "EMA Slow")
rsi_buy = input.int(35, "RSI Buy Level")
rsi_sell = input.int(65, "RSI Sell Level")
atr_sl = input.float(2.0, "ATR Stop Loss")

// Calculations
ema_f = ta.ema(close, ema_fast)
ema_s = ta.ema(close, ema_slow)
rsi = ta.rsi(close, 14)
[macd_line, signal_line, hist] = ta.macd(close, 12, 26, 9)
atr = ta.atr(14)

// Trend + Momentum + MACD alignment
uptrend = ema_f > ema_s
downtrend = ema_f < ema_s
momentum_bull = rsi > rsi_buy and rsi < rsi_sell
macd_bull = macd_line > signal_line
macd_bear = macd_line < signal_line

// Confluence: all 3 aligned
buy_signal = uptrend and rsi < rsi_buy and macd_bull and strategy.position_size == 0
sell_signal = downtrend and rsi > rsi_sell and macd_bear

if buy_signal
    strategy.entry("Long", strategy.long)
    strategy.exit("Exit", "Long", stop=close - atr * atr_sl, limit=close + atr * atr_sl * 2.5)

if sell_signal
    strategy.close("Long")

plot(ema_f, color=color.aqua, linewidth=2)
plot(ema_s, color=color.orange, linewidth=2)

// Confluence score for the alert
score = (uptrend ? 1 : 0) + (momentum_bull ? 1 : 0) + (macd_bull ? 1 : 0)

alert_message = '{"symbol": "' + syminfo.tickerid + '", ' +
    '"side": "' + (buy_signal ? "buy" : sell_signal ? "sell" : "hold") + '", ' +
    '"price": "' + str.tostring(close) + '", ' +
    '"confluence_score": "' + str.tostring(score) + '/3", ' +
    '"rsi": "' + str.tostring(rsi, "#.1") + '", ' +
    '"macd_hist": "' + str.tostring(hist, "#.###") + '", ' +
    '"strategy": "Multi_Confluence", ' +
    '"timeframe": "' + timeframe.period + '"}'

alertcondition(buy_signal or sell_signal, title="Confluence Signal", message=alert_message)
'''

# === TEMPLATE 5: Crypto-Specific — EMA + Volume Profile ===
# Designed for crypto perpetual futures on Hyperliquid/TradingView.
# Best for: BTC, ETH, SOL swing trades.

PINE_CRYPTO_SWING = '''
//@version=5
strategy("Crypto Swing EMA+Vol", overlay=true)

// Inputs
ema_len = input.int(21, "EMA Length", minval=5)
vol_sma = input.int(20, "Volume SMA Length")
vol_mult = input.float(2.0, "Volume Spike Mult")
rr = input.float(2.5, "Risk:Reward")
sl_pct = input.float(3.0, "Stop Loss %", minval=0.5, step=0.5)

// Calculations
ema = ta.ema(close, ema_len)
rel_vol = volume / ta.sma(volume, vol_sma)
atr = ta.atr(14)

// Trend
uptrend = close > ema and ema > ema[1]
downtrend = close < ema and ema < ema[1]

// Pullback entries (buy dips in uptrend, sell rallies in downtrend)
pullback_buy = uptrend and low < ema and close > ema and rel_vol > vol_mult
pullback_sell = downtrend and high > ema and close < ema and rel_vol > vol_mult

if pullback_buy and strategy.position_size == 0
    sl = close * (1 - sl_pct / 100)
    tp = close + (close - sl) * rr
    strategy.entry("Long", strategy.long)
    strategy.exit("Exit Long", "Long", stop=sl, limit=tp)

if pullback_sell and strategy.position_size == 0
    sl = close * (1 + sl_pct / 100)
    tp = close - (sl - close) * rr
    strategy.entry("Short", strategy.short)
    strategy.exit("Exit Short", "Short", stop=sl, limit=tp)

plot(ema, color=color.yellow, linewidth=2)
bgcolor(pullback_buy ? color.new(color.green, 90) : pullback_sell ? color.new(color.red, 90) : na)

alert_message = '{"symbol": "' + syminfo.tickerid + '", ' +
    '"side": "' + (pullback_buy ? "buy" : pullback_sell ? "sell" : "hold") + '", ' +
    '"price": "' + str.tostring(close) + '", ' +
    '"rel_volume": "' + str.tostring(rel_vol, "#.1") + '", ' +
    '"strategy": "Crypto_Swing", ' +
    '"timeframe": "' + timeframe.period + '"}'

alertcondition(pullback_buy or pullback_sell, title="Crypto Swing", message=alert_message)
'''

# Export map
TEMPLATES = {
    "ema_crossover": PINE_EMA_CROSSOVER,
    "rsi_bb_squeeze": PINE_RSI_BB,
    "volume_breakout": PINE_VOLUME_BREAKOUT,
    "multi_confluence": PINE_CONFLUENCE,
    "crypto_swing": PINE_CRYPTO_SWING,
}

TEMPLATE_DESCRIPTIONS = {
    "ema_crossover": "EMA 9/21 crossover with ATR-based SL/TP. Good for swing trades.",
    "rsi_bb_squeeze": "RSI oversold + Bollinger Band squeeze. High-probability reversals.",
    "volume_breakout": "3x volume + price breakout above resistance. Momentum plays.",
    "multi_confluence": "EMA trend + RSI + MACD confluence score. Conservative entries.",
    "crypto_swing": "EMA pullback + volume spike for crypto. BTC/ETH/SOL swing trades.",
}
