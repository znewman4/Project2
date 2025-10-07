# src/utils/wasserstein.py
import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance


def rolling_wasserstein(series: pd.Series, window: int = 500, step: int = 1) -> pd.Series:
    """
    Compute rolling 1-D Wasserstein distance between consecutive windows of a series.
    Use this to measure how much the recent return distribution has shifted.

    Parameters
    ----------
    series : pd.Series
        The numeric time series (e.g. returns or log returns).
    window : int, default=500
        Number of samples in each comparison window.
    step : int, default=1
        Step size when sliding the windows forward.

    Returns
    -------
    pd.Series
        A time series of Wasserstein distances aligned to the end of each second window.
    """
    series = series.dropna()
    distances, idx = [], []
    n = len(series)

    for i in range(0, n - 2 * window, step):
        a = series.iloc[i:i + window]
        b = series.iloc[i + window:i + 2 * window]
        d = wasserstein_distance(a, b)
        distances.append(d)
        idx.append(series.index[i + 2 * window])

    return pd.Series(distances, index=idx, name="wasserstein_shift")
