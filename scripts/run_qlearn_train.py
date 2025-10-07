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
    Convert the latest observation and portfolio info into a richer discrete state:
    (momentum, ema_signal, vol_bin, rsi_signal, position)
    """
    # fallback if info doesn't include feature values
    momentum = int(np.sign(info.get("momentum_sign", 0)))
    ema_signal = int(np.sign(info.get("ema_signal", 0)))
    rsi_signal = int(np.sign(info.get("rsi_signal", 0)))

    vol = info.get("volatility", 0)
    if np.isnan(vol) or vol == 0:
        vol_bin = 0
    else:
        vol_bin = int(vol > np.nanmedian([vol]))  # binary vol regime

    position = int(np.sign(info.get("position", 0)))

    return (momentum, ema_signal, vol_bin, rsi_signal, position)


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
    agent = QLearningAgent(alpha=0.1, gamma=0.995, epsilon=0.3)

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
            
        # Decay exploration rate
        agent.epsilon = max(0.05, agent.epsilon * 0.98)


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
