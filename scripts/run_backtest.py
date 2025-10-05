#scripts/run_backtest.py
import argparse, os
import json
import pandas as pd
import importlib


from src.utils.io import load_config
from src.env.trading_env import TradingEnv
from src.agents.baselines import EMABaselineAgent
from src.evaluation.metrics import compute_pnl, compute_sharpe, compute_maxdd
from src.evaluation.plots import plot_equity_curve


def run_backtest(exp_path: str):
    # 1. Load experiment config
    exp_cfg = load_config(exp_path)
    env_cfg = load_config(exp_cfg["environment"]["config"])
    data_cfg = load_config(exp_cfg["data"]["config"])

    # 2. Load dataset
    symbol = data_cfg["data"]["symbol"].replace("/", "").lower()
    timeframe = data_cfg["data"]["timeframe"]
    filename = f"{symbol}_{timeframe}.parquet"
    data_path = os.path.join(data_cfg["data"]["paths"]["interim"], filename)

    df = pd.read_parquet(data_path)

    # filter backtest window
    start_date = pd.to_datetime(exp_cfg["evaluation"]["backtest"]["start_date"]).tz_localize("UTC")
    end_date = pd.to_datetime(exp_cfg["evaluation"]["backtest"]["end_date"]).tz_localize("UTC")
    df = df[(df.index >= start_date) & (df.index <= end_date)]

    df.columns = [c.capitalize() for c in df.columns] # Ensure 'Close' column is capitalized


    # 3. Init env + agent
    env = TradingEnv(df, env_cfg)
    agent_name = exp_cfg["agent"]["name"]
    agent_params = exp_cfg["agent"].get("params", {})

    # Try importing agent dynamically from known agent modules
    possible_modules = ["src.agents.baselines", "src.agents.q_learning"]
    agent_cls = None
    for mod in possible_modules:
        module = importlib.import_module(mod)
        if hasattr(module, agent_name):
            agent_cls = getattr(module, agent_name)
            break

    if agent_cls is None:
        raise ImportError(f"Agent class '{agent_name}' not found in known modules.")

    agent = agent_cls(**agent_params)

# 4. Run backtest loop (aware of info dict)
    obs, info = env.reset()
    terminated, truncated = False, False
    equity_history, trades = [], []

    while not (terminated or truncated):
        # --- Context-aware action call ---
        try:
            action = agent.act(obs, info)
        except TypeError:
            # fallback for baselines (act(obs) only)
            action = agent.act(obs)

        # --- Step environment ---
        obs, reward, terminated, truncated, info = env.step(action)

        equity_history.append(info.get("equity"))
        if "trade" in info and info["trade"] is not None:
            trades.append(info["trade"])
    # 5. Compute metrics
    metrics = {}
    for m in exp_cfg["evaluation"]["backtest"]["metrics"]:
        if m == "pnl":
            metrics["pnl"] = compute_pnl(equity_history)
        elif m == "sharpe":
            metrics["sharpe"] = compute_sharpe(equity_history)
        elif m == "max_drawdown":
            metrics["max_drawdown"] = compute_maxdd(equity_history)

    # 6. Save results with unique agent label
    agent_label = exp_cfg["agent"]["name"].lower()

    artifacts_dir = "results/artifacts"
    plots_dir = "results/plots"
    os.makedirs(artifacts_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    metrics_path = os.path.join(artifacts_dir, f"{agent_label}.json")
    plot_path = os.path.join(plots_dir, f"{agent_label}_equity.png")

    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    plot_equity_curve(equity_history, trades, plot_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", type=str, default="experiments/exp_qlearn.yaml")
    args = parser.parse_args()

    run_backtest(args.exp)
