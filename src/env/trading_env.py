#src/env/trading_env.py
from typing import Optional, Dict, Any
import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces

from .portfolio import Portfolio, PortfolioConfig
from .render_utils import EpisodeRenderer


def _read_config(cfg: Dict[str, Any]) -> tuple[int, Optional[int], dict[str, Any], PortfolioConfig]:
    env_cfg = cfg.get("env", {})
    reward_cfg = cfg.get("reward", {})

    window_size = env_cfg.get("window_size", 10)
    max_steps = env_cfg.get("max_steps", None)

    # map env/reward keys into PortfolioConfig
    p_cfg = PortfolioConfig(
        initial_cash   = env_cfg.get("initial_cash", 10_000.0),
        trade_size     = env_cfg.get("trade_size", 1.0),
        allow_short    = env_cfg.get("allow_short", True),
        max_position   = env_cfg.get("max_position", None),
        commission_pct = env_cfg.get("commission_pct", 0.0),
        slippage_pct   = env_cfg.get("slippage_pct", 0.0),

        sharpe_alpha     = reward_cfg.get("sharpe_alpha", 0.0),
        sharpe_lookback  = reward_cfg.get("sharpe_lookback", 20),
        sharpe_annualizer= reward_cfg.get("sharpe_annualizer", 1.0),
        dd_beta          = reward_cfg.get("dd_beta", 0.0),
    )
    return window_size, max_steps, env_cfg, p_cfg


class TradingEnv(gym.Env):
    """
    Gymnasium environment wrapper.
    Actions: 0=hold, 1=buy(+size), 2=sell(-size)
    Observation: last N closes
    Reward: Δequity + sharpe_alpha*Sharpe - dd_beta*drawdown
    """
    metadata = {"render_modes": ["human"]}

    def __init__(self, df: pd.DataFrame, config: Dict[str, Any]):
        super().__init__()
        assert "Close" in df.columns, "DataFrame must contain a 'Close' column."

        self.df = df.reset_index(drop=True).copy()
        self.window_size, self.max_steps, self._env_cfg_raw, self.port_cfg = _read_config(config)
        self.portfolio = Portfolio(self.port_cfg)
        self.renderer = EpisodeRenderer()

        # spaces
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.window_size,), dtype=np.float32
        )

        # state
        self.current_step: int = 0

    # ---- API ----
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self.current_step = self.window_size
        self.portfolio.reset()
        # initialize equity at current price
        self.portfolio.mark_to_market(float(self.df["Close"].iloc[self.current_step]))
        obs = self._get_obs()
        info = self._info()
        return obs, info

    def step(self, action: int):
        # equity before trade
        px_now = float(self.df["Close"].iloc[self.current_step])
        prev_equity = self.portfolio.equity

        # execute action
        self.portfolio.apply_action(self.current_step, action, px_now)

        # advance time
        self.current_step += 1

        # mark to market after move
        px_next = float(self.df["Close"].iloc[self.current_step])
        new_equity = self.portfolio.mark_to_market(px_next)

        # reward
        reward = self.portfolio.reward(prev_equity, new_equity)

        # termination / truncation
        terminated = self.current_step >= (len(self.df) - 1)
        truncated = False
        if self.max_steps is not None:
            truncated = (self.current_step - self.window_size) >= self.max_steps

        obs = self._get_obs()
        info = self._info()
        return obs, reward, terminated, truncated, info

    def render(self):
        closes = self.df["Close"].values
        self.renderer.render(
            closes=closes,
            cur_step=self.current_step,
            window_offset=self.window_size,
            equity_history=self.portfolio.equity_history,
            trades=self.portfolio.trades,
        )

    # ---- helpers ----
    def _get_obs(self) -> np.ndarray:
        s = self.current_step - self.window_size
        e = self.current_step
        return self.df["Close"].iloc[s:e].values.astype(np.float32)

    def _info(self) -> dict:
        return {
            "equity": self.portfolio.equity,
            "cash": self.portfolio.cash,
            "position": self.portfolio.position,
            "reward_components": self.portfolio.last_reward_components,
            "step": self.current_step,
        }
