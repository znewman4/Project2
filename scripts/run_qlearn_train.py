# scripts/run_qlearn_train.py

import logging
from pathlib import Path
import numpy as np
import pandas as pd
import json
from datetime import datetime

from src.utils.io import load_config
from src.env.trading_env import TradingEnv
from src.agents.q_learning import QLearningAgent

# -----------------------------------------------------------
# Setup logging
# -----------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger("run_qlearn_train")

# -----------------------------------------------------------
# Discretization helper
# -----------------------------------------------------------
def discretize_observation(obs: np.ndarray, info: dict) -> tuple:
    """
    Convert array of recent prices + portfolio info into a discrete (momentum, trend, position) state.
    """
    # 1. Momentum sign: recent price relative to average
    momentum_sign = int(np.sign(obs[-1] - np.mean(obs)))

    # 2. EMA signal: short-term vs long-term trend (simple proxy)
    short_ema = np.mean(obs[-3:]) if len(obs) >= 3 else np.mean(obs)
    long_ema = np.mean(obs)
    ema_signal = int(np.sign(short_ema - long_ema))

    # 3. Position sign: current portfolio stance
    position = int(np.sign(info.get("position", 0)))

    return (momentum_sign, ema_signal, position)

# -----------------------------------------------------------
# Training loop
# -----------------------------------------------------------
def train_qlearner(
    config_path="configs/env.yaml",
    data_path="data/processed/btcusdt_5m.parquet",
    episodes=50,
    save_dir="results/artifacts/"
):
    # --- Load config + data ---
    cfg = load_config(config_path)
    df = pd.read_parquet(data_path)

    # --- Initialize environment and agent ---
    env = TradingEnv(df, cfg)
    agent = QLearningAgent(alpha=0.1, gamma=0.95, epsilon=0.1)

    # --- Prepare save paths ---
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(save_dir) / f"qlearn_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # --- Diagnostics ---
    episode_rewards = []
    episode_q_means = []

    # --- Training loop ---
    for ep in range(episodes):
        obs, info = env.reset()
        state = discretize_observation(obs, info)
        done = False
        total_reward = 0
        steps = 0

        while True:
            # Agent picks action
            action = agent.act(state)

            # Environment responds
            next_obs, reward, terminated, truncated, next_info = env.step(action)
            next_state = discretize_observation(next_obs, next_info)

            # Q-learning update
            agent.update(state, action, reward, next_state)

            # Transition
            state = next_state
            total_reward += reward
            steps += 1

            if terminated or truncated:
                break

        # Diagnostics
        episode_rewards.append(total_reward)
        q_values_all = [q for s in agent.q_table.values() for q in s.values()]
        episode_q_means.append(float(np.mean(q_values_all)) if q_values_all else 0.0)

        logger.info(
            f"Episode {ep+1}/{episodes} | Steps: {steps} | "
            f"Total Reward: {total_reward:.4f} | Avg Q: {episode_q_means[-1]:.4f}"
        )

    # --- Save artifacts ---
    q_table_path = run_dir / "q_table.pkl"
    agent.save(str(q_table_path))

    # Save diagnostics to JSON
    diag_data = {
        "episodes": episodes,
        "episode_rewards": episode_rewards,
        "episode_q_means": episode_q_means,
        "config_path": config_path,
        "data_path": data_path,
        "alpha": agent.alpha,
        "gamma": agent.gamma,
        "epsilon": agent.epsilon,
    }
    with open(run_dir / "training_diagnostics.json", "w") as f:
        json.dump(diag_data, f, indent=2)

    # Optional: save to CSV for plotting later
    pd.DataFrame({
        "episode": np.arange(1, episodes + 1),
        "reward": episode_rewards,
        "avg_q_value": episode_q_means
    }).to_csv(run_dir / "training_log.csv", index=False)

    logger.info(f"✅ Training complete. Artifacts saved in {run_dir}")
    logger.info(f"Average reward: {np.mean(episode_rewards):.4f}")

# -----------------------------------------------------------
if __name__ == "__main__":
    train_qlearner()
