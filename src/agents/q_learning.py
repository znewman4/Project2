# src/agents/q_learning.py
import numpy as np
import pickle
from typing import Tuple, Dict
from pathlib import Path

class QLearningAgent:
    """
    Tabular Q-Learning Agent for discrete state-action spaces.

    States: tuples like (momentum_sign, ema_signal, position)
    Actions: [-1, 0, 1]  →  Sell, Hold, Buy
    """

    def __init__(self,
                 actions=(-1, 0, 1),
                 alpha=0.1,       # learning rate
                 gamma=0.95,      # discount factor
                 epsilon=0.1):    # exploration probability
        self.actions = list(actions)
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_table: Dict[Tuple, Dict[int, float]] = {}

    def _ensure_state(self, state: Tuple):
        """Initialise empty Q-values for unseen states."""
        if state not in self.q_table:
            self.q_table[state] = {a: 0.0 for a in self.actions}

    def act(self, state: Tuple) -> int:
        """Epsilon-greedy policy for action selection."""
        self._ensure_state(state)
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.actions)
        else:
            q_values = self.q_table[state]
            return max(q_values, key=q_values.get)

    def update(self, state: Tuple, action: int, reward: float, next_state: Tuple):
        """Standard Q-learning update rule."""
        self._ensure_state(state)
        self._ensure_state(next_state)
        q_sa = self.q_table[state][action]
        next_max = max(self.q_table[next_state].values())
        #here you will find Q-learning formula propa basic 
        new_q = q_sa + self.alpha * (reward + self.gamma * next_max - q_sa)
        self.q_table[state][action] = new_q

    def reset(self):
        """Clear Q-table."""
        self.q_table.clear()

    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump(self.q_table, f)

    def load(self, path: str):
        with open(path, "rb") as f:
            self.q_table = pickle.load(f)

# -----------------------------------------------------------
# Wrapper for backtesting a trained Q-learning policy
# -----------------------------------------------------------

class QLearningPolicyAgent:
    """
    Lightweight wrapper for using a trained Q-table in backtests.
    Exposes act(obs) so it behaves like EMA or Random baselines.
    """

    def __init__(self, q_table_path: str, alpha=0.1, gamma=0.95):
        self.alpha = alpha
        self.gamma = gamma
        self.q_table_path = Path(q_table_path)

        # Load trained Q-table
        with open(self.q_table_path, "rb") as f:
            self.q_table = pickle.load(f)

        # Recreate action space
        self.actions = [-1, 0, 1]

    def discretize_observation(self, obs, info=None):
        if info is None:
            info = {"position": 0}

        momentum_sign = int(np.sign(obs[-1] - np.mean(obs)))
        short_ema = np.mean(obs[-3:]) if len(obs) >= 3 else np.mean(obs)
        long_ema = np.mean(obs)
        ema_signal = int(np.sign(short_ema - long_ema))
        position = int(np.sign(info.get("position", 0)))
        return (momentum_sign, ema_signal, position)

    def act(self, obs):
        """Select greedy action from Q-table for given observation."""
        state = self.discretize_observation(obs)
        if state not in self.q_table:
            return 0  # hold if unseen
        q_values = self.q_table[state]
        return max(q_values, key=q_values.get)

