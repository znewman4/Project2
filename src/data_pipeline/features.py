# src/data_pipeline/features.py
import pandas as pd

def add_basic_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute simple technical indicators for early agents.
    Returns a copy of df with new columns appended.
    """
    df = df.copy()

    #make sure that columns are capitalised (it likes that)
    rename_map = {c: c.capitalize() for c in df.columns}
    df = df.rename(columns=rename_map)


    # --- Returns & Momentum ---
    df["return"] = df["Close"].pct_change()
    df["momentum_sign"] = (df["return"] > 0).astype(int) - (df["return"] < 0).astype(int)

    # --- Exponential Moving Averages ---
    df["ema_short"] = df["Close"].ewm(span=10, adjust=False).mean()
    df["ema_long"]  = df["Close"].ewm(span=30, adjust=False).mean()
    df["ema_signal"] = (df["ema_short"] > df["ema_long"]).astype(int) - (df["ema_short"] < df["ema_long"]).astype(int)

    # --- Drop first rows with NaNs from indicators ---
    df = df.dropna().copy()
    return df
