# src/agents/baselines.py

import numpy as np
import pandas as pd

class EMABaselineAgent:
    """
    A simple baseline agent that trades based on EMA crossovers.
    Actions:
      -1 = sell
       0 = hold
       1 = buy
    """

    def __init__(self, short_window: int = 10, long_window: int = 30):
        if short_window >= long_window:
            raise ValueError("short_window must be smaller than long_window")
        self.short_window = short_window
        self.long_window = long_window
        self.reset()

    def reset(self):
        """Reset internal state (price history and last action)."""
        self.prices = []
        self.last_action = 0

    def act(self, obs):
        """
        Decide an action based on observation.
        obs: dict or ndarray-like, expected to contain latest price (float).
        """
        # Extract price
        if isinstance(obs, dict) and "Close" in obs:
            price = obs["Close"]
        elif isinstance(obs, (list, tuple, np.ndarray, pd.Series)):
            price = obs[-1]  # assume last element is the price
        else:
            price = float(obs)

        self.prices.append(price)

        # Not enough data yet → hold
        if len(self.prices) < self.long_window:
            return 0

        prices_series = pd.Series(self.prices)

        short_ema = prices_series.ewm(span=self.short_window, adjust=False).mean().iloc[-1]
        long_ema = prices_series.ewm(span=self.long_window, adjust=False).mean().iloc[-1]

        if short_ema > long_ema:
            action = 1  # buy
        elif short_ema < long_ema:
            action = -1  # sell
        else:
            action = 0  # hold

        self.last_action = action
        return action
