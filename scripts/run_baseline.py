#scripts/run_backtest.py
import argparse, os
import json
import pandas as pd

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
    agent_cls = globals()[exp_cfg["agent"]["name"]]  # e.g. EMABaselineAgent
    agent = agent_cls(**exp_cfg["agent"]["params"])

    # 4. Run backtest loop
    obs, info = env.reset()
    terminated, truncated = False, False
    equity_history, trades = [], []

    while not (terminated or truncated):
        action = agent.act(obs)
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

    # 6. Save results
    with open("results/artifacts/exp001.json", "w") as f:
        json.dump(metrics, f, indent=2)

    plot_equity_curve(equity_history, trades, "results/plots/exp001.png")

    print("Backtest complete. Results saved in results/artifacts/ and results/plots/.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", type=str, default="experiments/exp001_baselines.yaml")
    args = parser.parse_args()

    run_backtest(args.exp)
