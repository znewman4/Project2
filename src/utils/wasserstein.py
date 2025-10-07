# src/utils/wasserstein.py
import pandas as pd
import numpy as np
from scipy.stats import wasserstein_distance

def rolling_wasserstein(series: pd.Series, window: int = 500, step: int = 1) -> pd.Series:
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


def smooth_wasserstein(series: pd.Series, span: int = None, zscore: bool = False) -> pd.Series:
    """
    Smooth and optionally z-normalize a Wasserstein drift series.
    Applies two-stage EMA for stronger noise suppression.
    If span not given, defaults to ~2x the typical rolling window used
    in rolling_wasserstein (e.g. 1000 if window=500).
    """
    if span is None:
        span = 1000  # auto default for 5m data with window=500

    # Optional pre-median to damp micro spikes
    s = series.rolling(20, center=True).median()

    # Two-stage exponential smoothing
    s = s.ewm(span=span // 2, adjust=False).mean()
    s = s.ewm(span=span, adjust=False).mean()

    if zscore:
        s = (s - s.mean()) / (s.std() + 1e-9)

    return s.rename("wasserstein_smooth")
