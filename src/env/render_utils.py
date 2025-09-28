from typing import List, Tuple, Optional
import numpy as np
import matplotlib.pyplot as plt


class EpisodeRenderer:
    """Simple matplotlib renderer: price w/ trade markers + equity curve."""
    def __init__(self):
        self.fig = None
        self.ax_price = None
        self.ax_equity = None

    def render(
        self,
        closes: np.ndarray,
        cur_step: int,
        window_offset: int,
        equity_history: List[float],
        trades: List[Tuple[int, int, float, float]],
    ):
        steps = np.arange(window_offset, window_offset + len(equity_history))

        if self.fig is None:
            self.fig, (self.ax_price, self.ax_equity) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
            self.ax_price.set_title("Price & Trades")
            self.ax_equity.set_title("Equity")
            self.ax_price.grid(True, alpha=0.3)
            self.ax_equity.grid(True, alpha=0.3)

        self.ax_price.clear()
        self.ax_equity.clear()

        self.ax_price.plot(closes[:cur_step + 1], linewidth=1.2)

        if trades:
            buys = [(s, px) for (s, a, px, sz) in trades if a == 1]
            sells = [(s, px) for (s, a, px, sz) in trades if a == 2]
            if buys:
                s_b, p_b = zip(*buys)
                self.ax_price.scatter(s_b, p_b, marker="^")
            if sells:
                s_s, p_s = zip(*sells)
                self.ax_price.scatter(s_s, p_s, marker="v")

        self.ax_equity.plot(steps, equity_history, linewidth=1.2)
        self.ax_price.set_ylabel("Price")
        self.ax_equity.set_ylabel("Equity")
        self.ax_equity.set_xlabel("Step")
        plt.tight_layout()
        plt.pause(0.001)
