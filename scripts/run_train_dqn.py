import os
import yaml
import torch
import numpy as np
from src.utils.io import load_yaml_config
from src.env.trading_env import TradingEnv
from src.agents.dqn import DQNAgent
from src.utils.logging import get_logger


def train_dqn(config_path: str):
    # --------------------------------------------------
    # 1. Load config
    # --------------------------------------------------
    cfg = load_yaml_config(config_path)
    exp_cfg = cfg["experiment"]
    env_cfg = cfg["env"]
    agent_cfg = cfg["agent"]
    log_cfg = cfg.get("logging", {})

    logger = get_logger("train_dqn")
    logger.info(f"Starting DQN training: {exp_cfg['name']}")

    # --------------------------------------------------
    # 2. Init environment
    # --------------------------------------------------
    env = TradingEnv(
        data_path=env_cfg["data_path"],
        window_size=env_cfg["window_size"],
        trading_fee=env_cfg["trading_fee"],
        initial_balance=env_cfg["initial_balance"],
        reward_mode=env_cfg.get("reward_mode", "pnl"),
    )
    obs, _ = env.reset()

    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    # --------------------------------------------------
    # 3. Init agent
    # --------------------------------------------------
    agent = DQNAgent(obs_dim, action_dim, agent_cfg)

    total_timesteps = exp_cfg["total_timesteps"]
    eval_interval = exp_cfg["eval_interval"]
    checkpoint_path = os.path.join(exp_cfg["save_dir"], "dqn.pt")
    os.makedirs(exp_cfg["save_dir"], exist_ok=True)

    # --------------------------------------------------
    # 4. Main training loop
    # --------------------------------------------------
    rewards = []
    episode_reward = 0
    obs, _ = env.reset()
    for step in range(1, total_timesteps + 1):
        action = agent.act(obs)
        next_obs, reward, done, _, info = env.step(action)
        agent.push(obs, action, reward, next_obs, done)

        loss = agent.update()
        obs = next_obs
        episode_reward += reward

        if done:
            rewards.append(episode_reward)
            logger.info(f"[Episode {len(rewards)}] Reward: {episode_reward:.2f}")
            obs, _ = env.reset()
            episode_reward = 0

        if step % eval_interval == 0:
            avg_reward = np.mean(rewards[-10:]) if rewards else 0
            logger.info(f"Step {step} | Epsilon: {agent.epsilon:.3f} | AvgReward(10ep): {avg_reward:.2f}")
            agent.save(checkpoint_path)

    logger.info(f"Training finished. Model saved to {checkpoint_path}")


if __name__
