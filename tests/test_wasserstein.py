# src/scripts/test_wasserstein_signals.py
import pandas as pd
import matplotlib.pyplot as plt

def main():
    path = "data/processed/btcusdt_with_wasserstein_signals.parquet"
    df = pd.read_parquet(path)
    print(f"Loaded {len(df):,} rows from {path}")
    print("Columns:", df.columns.tolist())

    # Check basic stats to make sure columns have values
    for c in ["returns", "wasserstein_shift", "wasserstein_smooth"]:
        if c in df:
            print(f"\n{c} describe():\n", df[c].describe())

    # Compute realized volatility (1-day rolling std of returns)
    if "returns" not in df:
        raise ValueError("'returns' column not found")
    df["realized_vol"] = df["returns"].rolling(288).std()

    # Drop NaNs for plotting
    df = df.dropna(subset=["wasserstein_shift", "wasserstein_smooth", "realized_vol"])
    print(f"\nAfter dropna: {len(df):,} rows remain")

    corr_raw = df["wasserstein_shift"].corr(df["realized_vol"])
    corr_smooth = df["wasserstein_smooth"].corr(df["realized_vol"])
    print(f"\nCorrelation with realized vol → Raw: {corr_raw:.3f}, Smoothed: {corr_smooth:.3f}")

    # Normalize robustly (avoid outlier scaling)
    vol_scaled = df["realized_vol"] / df["realized_vol"].quantile(0.99)
    raw_scaled = df["wasserstein_shift"] / df["wasserstein_shift"].quantile(0.99)
    smooth_scaled = df["wasserstein_smooth"] / df["wasserstein_smooth"].quantile(0.99)


    # --- Plot
    fig, ax1 = plt.subplots(figsize=(12, 5))
    ax1.plot(df.index, df["Close"], color="black", label="Close Price", alpha=0.6)
    ax1.set_ylabel("Price", color="black")

    ax2 = ax1.twinx()
    ax2.plot(df.index, vol_scaled, color="tab:blue", alpha=0.5, label="Realized Vol (scaled)")
    ax2.plot(df.index, raw_scaled, color="tab:red", alpha=0.3, label="Raw Wasserstein (scaled)")
    ax2.plot(df.index, smooth_scaled, color="darkred", linewidth=1.8, label="Smoothed Wasserstein (scaled)")
    ax2.set_ylabel("Scaled Drift / Volatility")

    # Combine legends
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc="upper right")

    plt.title("Raw vs Smoothed Wasserstein Drift (BTCUSDT 5m)")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
