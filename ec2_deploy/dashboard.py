#!/usr/bin/env python3
"""Generate a static HTML dashboard from TradingAgents results.

Creates an HTML file with:
- Latest decisions table
- Decision history
- Paper portfolio status
- Backtest accuracy

Usage:
    python3.12 dashboard.py                # generate dashboard
    python3.12 dashboard.py --serve 8080   # generate and serve on port
"""
import json
import glob
import os
import sys
from datetime import datetime
from dotenv import load_dotenv
load_dotenv()

RESULTS_DIR = os.path.expanduser("~/TradingAgents/results")
DASHBOARD_FILE = os.path.expanduser("~/TradingAgents/results/dashboard.html")


def load_latest_results():
    files = sorted(glob.glob(os.path.join(RESULTS_DIR, "daily/*.json")))
    if not files:
        return None
    with open(files[-1]) as f:
        return json.load(f)


def load_all_results():
    files = sorted(glob.glob(os.path.join(RESULTS_DIR, "daily/*.json")))
    all_data = []
    for f in files[-30:]:  # last 30 files
        with open(f) as fh:
            all_data.append(json.load(fh))
    return all_data


def load_portfolio():
    pf = os.path.join(RESULTS_DIR, "portfolio.json")
    if os.path.exists(pf):
        with open(pf) as f:
            return json.load(f)
    return None


def load_backtest():
    files = sorted(glob.glob(os.path.join(RESULTS_DIR, "backtest/*.json")))
    if not files:
        return None
    with open(files[-1]) as f:
        return json.load(f)


def load_decision_history():
    hf = os.path.join(RESULTS_DIR, "decision_history.json")
    if os.path.exists(hf):
        with open(hf) as f:
            return json.load(f)
    return {}


def generate_dashboard():
    latest = load_latest_results()
    history = load_decision_history()
    portfolio = load_portfolio()
    backtest = load_backtest()
    all_results = load_all_results()

    color_map = {"BUY": "#57F287", "SELL": "#ED4245", "HOLD": "#FEE75C"}

    # Build history table data
    history_rows = ""
    for data in reversed(all_results[-10:]):
        for r in data["results"]:
            dec = r.get("decision", "N/A")
            color = color_map.get(dec, "#999")
            history_rows += """<tr>
                <td>{}</td><td>{}</td>
                <td style="color:{}; font-weight:bold">{}</td>
                <td>{}</td>
            </tr>""".format(data["date"], r["ticker"], color, dec, data.get("profile", ""))

    # Latest signals
    signal_cards = ""
    if latest:
        for r in latest["results"]:
            dec = r.get("decision", "N/A")
            color = color_map.get(dec, "#999")
            signal_cards += """<div class="card" style="border-left: 4px solid {}">
                <h3>{}</h3><span class="decision" style="color:{}">{}</span>
            </div>""".format(color, r["ticker"], color, dec)

    # Portfolio section
    portfolio_html = "<p>No portfolio data yet. Run paper_trader.py first.</p>"
    if portfolio:
        total = portfolio["cash"]
        pos_html = ""
        for t, p in portfolio.get("positions", {}).items():
            total += p.get("cost", 0)
            pos_html += "<tr><td>{}</td><td>{}</td><td>${:.2f}</td><td>{}</td></tr>".format(
                t, p["shares"], p["avg_price"], p["date"])
        pnl = total - 100000
        pnl_color = "#57F287" if pnl >= 0 else "#ED4245"
        portfolio_html = """
            <p>Total Value: <strong>${:,.0f}</strong>
            <span style="color:{}">({}{:.1f}%)</span></p>
            <p>Cash: ${:,.0f} | Positions: {} | Trades: {}</p>
            {}
        """.format(total, pnl_color, "+" if pnl >= 0 else "", pnl / 1000,
                   portfolio["cash"], len(portfolio.get("positions", {})),
                   portfolio.get("total_trades", 0),
                   "<table><tr><th>Ticker</th><th>Shares</th><th>Avg Price</th><th>Since</th></tr>{}</table>".format(pos_html) if pos_html else "")

    # Backtest section
    backtest_html = "<p>No backtest data yet. Run backtest.py first.</p>"
    if backtest:
        acc_color = "#57F287" if backtest["accuracy"] >= 60 else "#FEE75C" if backtest["accuracy"] >= 40 else "#ED4245"
        backtest_html = """
            <p>Accuracy: <strong style="color:{}">{:.1f}%</strong></p>
            <p>Correct: {} | Wrong: {} | Total: {}</p>
        """.format(acc_color, backtest["accuracy"], backtest["correct"],
                   backtest["wrong"], backtest["total"])

    html = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>TradingAgents Dashboard</title>
<meta http-equiv="refresh" content="300">
<style>
    * {{ margin: 0; padding: 0; box-sizing: border-box; }}
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            background: #1a1a2e; color: #e0e0e0; padding: 20px; }}
    .container {{ max-width: 1200px; margin: 0 auto; }}
    h1 {{ color: #5865F2; margin-bottom: 5px; }}
    h2 {{ color: #a0a0b0; margin: 20px 0 10px; border-bottom: 1px solid #333; padding-bottom: 5px; }}
    .subtitle {{ color: #888; margin-bottom: 20px; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(140px, 1fr)); gap: 10px; margin: 15px 0; }}
    .card {{ background: #16213e; padding: 15px; border-radius: 8px; }}
    .card h3 {{ font-size: 1.1em; margin-bottom: 5px; }}
    .decision {{ font-size: 1.3em; }}
    table {{ width: 100%; border-collapse: collapse; margin: 10px 0; }}
    th, td {{ padding: 8px 12px; text-align: left; border-bottom: 1px solid #333; }}
    th {{ color: #888; font-size: 0.85em; text-transform: uppercase; }}
    .section {{ background: #16213e; padding: 20px; border-radius: 8px; margin: 15px 0; }}
    .updated {{ color: #666; font-size: 0.8em; margin-top: 20px; }}
</style>
</head>
<body>
<div class="container">
    <h1>TradingAgents Dashboard</h1>
    <p class="subtitle">Latest: {date} | Profile: {profile}</p>

    <h2>Current Signals</h2>
    <div class="grid">{signals}</div>

    <h2>Paper Portfolio</h2>
    <div class="section">{portfolio}</div>

    <h2>Backtest Accuracy</h2>
    <div class="section">{backtest}</div>

    <h2>Decision History (Last 10)</h2>
    <div class="section">
    <table>
        <tr><th>Date</th><th>Ticker</th><th>Decision</th><th>Profile</th></tr>
        {history}
    </table>
    </div>

    <p class="updated">Updated: {updated}</p>
</div>
</body>
</html>""".format(
        date=latest["date"] if latest else "N/A",
        profile=latest.get("profile", "N/A") if latest else "N/A",
        signals=signal_cards,
        portfolio=portfolio_html,
        backtest=backtest_html,
        history=history_rows,
        updated=datetime.now().strftime("%Y-%m-%d %H:%M UTC"),
    )

    with open(DASHBOARD_FILE, "w") as f:
        f.write(html)
    print("Dashboard generated: {}".format(DASHBOARD_FILE))
    return DASHBOARD_FILE


if __name__ == "__main__":
    path = generate_dashboard()

    if "--serve" in sys.argv:
        port = 8080
        for i, a in enumerate(sys.argv):
            if a == "--serve" and i + 1 < len(sys.argv):
                port = int(sys.argv[i + 1])
        import http.server
        import socketserver
        os.chdir(os.path.dirname(path))
        handler = http.server.SimpleHTTPRequestHandler
        with socketserver.TCPServer(("", port), handler) as httpd:
            print("Dashboard at http://0.0.0.0:{}/ ".format(port))
            httpd.serve_forever()
