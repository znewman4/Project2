# src/data_pipeline/features.py
import pandas as pd
import numpy as np

def add_basic_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    rename_map = {c: c.capitalize() for c in df.columns}
    df = df.rename(columns=rename_map)

    # --- Returns & Momentum ---
    df["return"] = df["Close"].pct_change()
    df["momentum_sign"] = np.sign(df["return"]).fillna(0)

    # --- Exponential Moving Averages ---
    df["ema_short"] = df["Close"].ewm(span=10, adjust=False).mean()
    df["ema_long"]  = df["Close"].ewm(span=30, adjust=False).mean()
    df["ema_signal"] = np.sign(df["ema_short"] - df["ema_long"]).fillna(0)

    # --- Volatility (rolling std of returns) ---
    df["volatility"] = df["return"].rolling(window=20).std()

    # --- RSI (momentum strength) ---
    delta = df["Close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = -delta.clip(upper=0).rolling(14).mean()
    rs = gain / (loss + 1e-9)
    df["rsi"] = 100 - (100 / (1 + rs))
    df["rsi_signal"] = np.where(df["rsi"] > 60, 1, np.where(df["rsi"] < 40, -1, 0))

    # --- ATR (price range volatility) ---
    df["tr"] = (df["High"] - df["Low"]).abs()
    df["atr"] = df["tr"].rolling(window=14).mean()
    df["normalized_range"] = df["tr"] / df["Close"]

    df = df.dropna().copy()
    return df
