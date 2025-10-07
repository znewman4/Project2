# src/regimes/wasserstein_features.py
import pandas as pd
import numpy as np
from typing import List, Dict, Optional
from src.utils.wasserstein import rolling_wasserstein, smooth_wasserstein

class WassersteinFeatureGenerator:
    """
    Generates Wasserstein-based regime features from market data.
    Supports single- or multi-scale smoothing, and optional diagnostics.
    """

    def __init__(
        self,
        window: int = 500,
        spans: Optional[List[int]] = None,
        multivariate: bool = False,
        cols: List[str] = ["returns"],
        compute_diagnostics: bool = True,
    ):
        self.window = window
        self.spans = spans or [250, 1000]
        self.multivariate = multivariate
        self.cols = cols
        self.compute_diagnostics = compute_diagnostics

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["returns"] = df["Close"].pct_change()

        # --- Step 1: Base Wasserstein drift
        df["wasserstein_shift"] = rolling_wasserstein(df["returns"], window=self.window)

        # --- Step 2: Multi-scale smoothing
        for span in self.spans:
            df[f"wasserstein_smooth_{span}"] = smooth_wasserstein(df["wasserstein_shift"], span=span)

        # --- Step 3: Optional diagnostics
        if self.compute_diagnostics and "realized_vol" in df:
            df["wasserstein_corr_vol"] = (
                df["wasserstein_shift"].rolling(self.window).corr(df["realized_vol"])
            )

        # --- Step 4: Clean up
        df = df.dropna(subset=["wasserstein_shift"])

        return df

    def summary(self, df: pd.DataFrame) -> Dict[str, float]:
        """Compute simple summary stats useful for sanity checks."""
        result = {}
        for span in self.spans:
            col = f"wasserstein_smooth_{span}"
            if col in df:
                result[f"corr_vol_{span}"] = df[col].corr(df.get("realized_vol"))
                result[f"std_{span}"] = df[col].std()
        return result
