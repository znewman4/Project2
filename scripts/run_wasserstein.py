# src/scripts/run_wasserstein.py
import pandas as pd
from src.utils.wasserstein import rolling_wasserstein, smooth_wasserstein

def main():
    # 1. Load your processed data
    df = pd.read_parquet("data/processed/btcusdt_5m.parquet")

    # 2. Compute returns
    df["returns"] = df["Close"].pct_change()

    # 3. Compute raw Wasserstein drift
    print("Computing rolling Wasserstein distance...")
    df["wasserstein_shift"] = rolling_wasserstein(df["returns"], window=500)

    # 4. Compute smoothed Wasserstein drift (EMA + z-score)
    print("Smoothing Wasserstein signal...")
    df["wasserstein_smooth"] = smooth_wasserstein(df["wasserstein_shift"], span=1000, zscore=False)

    # 5. Save combined results
    out_path = "data/processed/btcusdt_with_wasserstein_signals.parquet"
    df.to_parquet(out_path)
    print(f"✅ Saved both signals to {out_path}")

if __name__ == "__main__":
    main()
