# src/scripts/test_wasserstein_signals_basic.py
import pandas as pd
import matplotlib.pyplot as plt

def main():
    path = "data/processed/btcusdt_with_wasserstein_signals.parquet"
    df = pd.read_parquet(path)
    print(f"Loaded {len(df):,} rows from {path}")

    # Detect Wasserstein columns dynamically
    wasser_cols = [c for c in df.columns if "wasserstein" in c]
    print("Wasserstein-related columns:", wasser_cols)

    # Check for returns and realized volatility
    if "returns" not in df:
        df["returns"] = df["Close"].pct_change()
    df["realized_vol"] = df["returns"].rolling(288).std()

    # Drop NaNs
    df = df.dropna(subset=["realized_vol"] + wasser_cols)
    print(f"After dropna: {len(df):,} rows remain")

    # Correlation summary
    for c in wasser_cols:
        corr = df[c].corr(df["realized_vol"])
        print(f"Correlation with realized vol ({c}): {corr:.3f}")

    # Normalize for plotting (robust quantile scaling)
    vol_scaled = df["realized_vol"] / df["realized_vol"].quantile(0.99)
    scaled = {c: df[c] / df[c].quantile(0.99) for c in wasser_cols}

    # --- Plot
    fig, ax1 = plt.subplots(figsize=(12, 5))
    ax1.plot(df.index, df["Close"], color="black", label="Close Price", alpha=0.6)
    ax1.set_ylabel("Price", color="black")

    ax2 = ax1.twinx()
    ax2.plot(df.index, vol_scaled, color="tab:blue", alpha=0.5, label="Realized Vol (scaled)")
    for c in wasser_cols:
        ax2.plot(df.index, scaled[c], linewidth=1.4, alpha=0.8, label=c)
    ax2.set_ylabel("Scaled Drift / Volatility")

    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc="upper right")

    plt.title("Wasserstein Drift Features vs Realized Volatility (BTCUSDT 5m)")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
