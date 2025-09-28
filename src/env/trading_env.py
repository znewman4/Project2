import numpy as np
import pandas as pd
import gymnasium
from gymnasium import spaces


class TradingEnv(gym.Env):
    """A simple Gym-style trading environment skeleton."""

    metadata = {"render.modes": ["human"]}

    def __init__(self, df: pd.DataFrame, config: dict):
        super().__init__()

        self.df = df.reset_index(drop=True)
        self.window_size = config["env"]["window_size"]
        self.initial_cash = config["env"]["initial_cash"]
        self.max_steps = config["env"].get("max_steps")

        # Action space: 0=hold, 1=buy, 2=sell
        self.action_space = spaces.Discrete(3)

        # Observation space: N past closes
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.window_size,), dtype=np.float32
        )

        # Internal state
        self.current_step = None
        self.cash = None
        self.position = None

    def reset(self):
        """Reset the environment to the initial state."""
        self.current_step = self.window_size
        self.cash = self.initial_cash
        self.position = 0
        return self._get_observation()

    def step(self, action):
        """Advance one step. Placeholder for now."""
        self.current_step += 1
        done = self.current_step >= len(self.df) - 1
        reward = 0.0  # placeholder
        obs = self._get_observation()
        info = {}
        return obs, reward, done, info

    def _get_observation(self):
        """Return the last N closes."""
        start = self.current_step - self.window_size
        end = self.current_step
        obs = self.df["Close"].iloc[start:end].values
        return obs.astype(np.float32)

    def render(self, mode="human"):
        """Placeholder for rendering/logging."""
        pass
