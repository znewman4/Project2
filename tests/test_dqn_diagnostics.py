import os
import sys
import yaml
import torch
import numpy as np
import pandas as pd

# --- ensure project root is on path ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.env.trading_env import TradingEnv
from src.agents.dqn import DQNAgent


def test_dqn_diagnostics():
    """Run detailed diagnostics on TradingEnv + DQNAgent pipeline."""
    print("\n=== DQN Diagnostics ===")

    # --- Load config ---
    cfg_path = os.path.join(PROJECT_ROOT, "experiments", "exp010_dqn_v1.yaml")
    assert os.path.exists(cfg_path), f"Config not found: {cfg_path}"
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    # --- Load data & init env ---
    data_path = cfg["env"]["data_path"]
    df = pd.read_parquet(os.path.join(PROJECT_ROOT, data_path))
    env = TradingEnv(df, cfg)
    obs, info = env.reset()

    print(f"\nObservation shape: {obs.shape}")
    print(pd.Series(obs).describe())

    # --- Check per-feature magnitudes ---
    print("\nFeature magnitudes (first 20 features):")
    print(obs[:20])

    # --- Roll out random episode to check rewards ---
    rewards = []
    for _ in range(200):
        a = np.random.randint(env.action_space.n)
        obs, r, done, _, _ = env.step(a)
        rewards.append(r)
        if done:
            break
    rewards = np.array(rewards)
    print("\nReward stats over 200 random steps:")
    print(pd.Series(rewards).describe())

    # --- Init agent ---
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n
    agent = DQNAgent(obs_dim, act_dim, cfg["agent"])

    # --- Push some samples to buffer ---
    env.reset()
    for _ in range(100):
        a = np.random.randint(act_dim)
        obs2, r, done, _, _ = env.step(a)
        agent.push(obs, a, r, obs2, done)
        obs = obs2
        if done:
            env.reset()

    print(f"\nReplay buffer length after warmup: {len(agent.replay)}")
    if len(agent.replay) >= agent.batch_size:
        s, a, r, s2, d = agent.replay.sample(agent.batch_size)
        print(f"Sampled batch shapes: s={s.shape}, a={a.shape}, r={r.shape}")
        print("Reward batch stats:", pd.Series(r).describe())

    # --- Forward pass check ---
    test_obs = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        q_vals = agent.q_net(test_obs)
    print("\nForward pass check:")
    print("Q-values:", q_vals.cpu().numpy().round(4))

    print("\n=== Diagnostics complete ===\n")


if __name__ == "__main__":
    test_dqn_diagnostics()
