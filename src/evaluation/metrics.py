# src/evaluation/metrics.py
import numpy as np

def compute_pnl(equity_history):
    """
    Compute net profit/loss as final equity - initial equity.
    """
    if len(equity_history) < 2:
        return 0.0
    return float(equity_history[-1] - equity_history[0])


def compute_sharpe(equity_history, risk_free_rate=0.0):
    """
    Compute Sharpe ratio: mean excess return / std of returns.
    """
    equity = np.array(equity_history, dtype=float)
    if len(equity) < 2:
        return 0.0

    returns = np.diff(equity) / equity[:-1]
    if returns.std() == 0:
        return 0.0

    return float((returns.mean() - risk_free_rate) / returns.std())


def compute_maxdd(equity_history):
    """
    Compute maximum drawdown (min relative drop from peak).
    """
    equity = np.array(equity_history, dtype=float)
    if len(equity) == 0:
        return 0.0

    roll_max = np.maximum.accumulate(equity)
    dd = (equity - roll_max) / roll_max
    return float(dd.min())  # negative value, e.g. -0.25 = -25%
