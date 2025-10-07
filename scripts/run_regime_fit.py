# scripts/run_regime_fit.py
import pandas as pd
from src.utils.wasserstein import rolling_wasserstein

def main():
    # 1. Load processed data (change filename if needed)
    df = pd.read_parquet("data/processed/btcusdt_5m.parquet")

    # 2. Compute simple returns
    df["returns"] = df["Close"].pct_change()

    # 3. Compute rolling Wasserstein drift
    print("Computing rolling Wasserstein distance...")
    df["wasserstein_shift"] = rolling_wasserstein(df["returns"], window=500)

    # 4. Save result
    out_path = "data/processed/btcusdt_with_wasserstein.parquet"
    df.to_parquet(out_path)
    print(f"✅ Saved with Wasserstein column to {out_path}")

if __name__ == "__main__":
    main()
