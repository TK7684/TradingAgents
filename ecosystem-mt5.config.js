module.exports = {
  apps: [
    {
      name: "mt5-bridge",
      script: "/home/tk578/TradingAgents/run_bridge.py",
      interpreter: "/home/tk578/TradingAgents/.venv/bin/python3",
      cwd: "/home/tk578/TradingAgents",
      args: "--mode paper --port 5001 --interval 5",
      env: {
        NODE_ENV: "production",
        BRIDGE_MODE: "paper",
      },
      watch: false,
      max_memory_restart: "200M",
      restart_delay: 5000,
      max_restarts: 10,
      log_date_format: "YYYY-MM-DD HH:mm:ss Z",
      error_file: "/home/tk578/.pm2/logs/mt5-bridge-error.log",
      out_file: "/home/tk578/.pm2/logs/mt5-bridge-out.log",
    },
  ],
};
