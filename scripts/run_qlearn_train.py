# scripts/run_qlearn_train.py
import logging
from pathlib import Path
import numpy as np
import pandas as pd

from src.utils.io import load_yaml_config
from src.env.trading_env import TradingEnv
from src.agents.q_learning import QLearningAgent

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("run_qlearn_train")

def discretize_observation(obs: dict) -> tuple:
    """
    Convert continuous observation into discrete state tuple.
    Example: momentum_sign, ema_signal, and current position.
    """
    momentum_sign = int(np.sign(obs.get("momentum_sign", 0)))
    ema_signal = int(np.sign(obs.get("ema_signal", 0)))
    position = int(np.sign(obs.get("position", 0)))
    return (momentum_sign, ema_signal, position)

def train_qlearner(config_path="configs/env.yaml", episodes=20, save_path="results/artifacts/q_table.pkl"):
    # Load environment config
    cfg = load_yaml_config(config_path)
    df = pd.read_parquet("data/processed/btcusdt_5m.parquet")

    # Initialize environment and agent
    env = TradingEnv(df, cfg)
    agent = QLearningAgent(alpha=0.1, gamma=0.95, epsilon=0.1)

    rewards_per_episode = []

    for ep in range(episodes):
        obs = env.reset()
        state = discretize_observation(obs)
        done = False
        total_reward = 0

        while not done:
            action = agent.act(state)
            next_obs, reward, done, info = env.step(action)
            next_state = discretize_observation(next_obs)
            agent.update(state, action, reward, next_state)
            state = next_state
            total_reward += reward

        rewards_per_episode.append(total_reward)
        logger.info(f"Episode {ep+1}/{episodes} | Total Reward: {total_reward:.4f}")

    # Save the learned Q-table
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    agent.save(save_path)
    logger.info(f"✅ Training complete. Q-table saved to {save_path}")
    logger.info(f"Average reward per episode: {np.mean(rewards_per_episode):.4f}")

if __name__ == "__main__":
    train_qlearner()
