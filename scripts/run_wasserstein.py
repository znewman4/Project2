# src/scripts/run_wasserstein_signals.py
import pandas as pd
from src.regimes.wasserstein_features import WassersteinFeatureGenerator
import yaml
from pathlib import Path

def main():
    # --- Load config if available
    cfg_path = Path("configs/regimes.yaml")
    if cfg_path.exists():
        with open(cfg_path, "r") as f:
            cfg = yaml.safe_load(f)
        wasserstein_cfg = cfg.get("wasserstein", {})
    else:
        print("⚠️ configs/regimes.yaml not found — using defaults")
        wasserstein_cfg = dict(window=500, spans=[250, 1000])

    # --- Load data
    path = "data/processed/btcusdt_5m.parquet"
    print(f"Loading {path} ...")
    df = pd.read_parquet(path)
    print(f"Loaded {len(df):,} rows")

    # --- Compute realized volatility for diagnostics
    df["returns"] = df["Close"].pct_change()
    df["realized_vol"] = df["returns"].rolling(288).std()

    # --- Generate Wasserstein features
    print("Generating Wasserstein regime features...")
    wasserstein_cfg.pop("enabled", None)   # <── add this line
    gen = WassersteinFeatureGenerator(**wasserstein_cfg)
    df_feat = gen.compute(df)

    # --- Display summary diagnostics
    summary = gen.summary(df_feat)
    print("\n📈 Summary statistics:")
    for k, v in summary.items():
        print(f"  {k}: {v:.3f}")

    # --- Save output
    out_path = "data/processed/btcusdt_with_wasserstein_signals.parquet"
    df_feat.to_parquet(out_path)
    print(f"\n✅ Saved with Wasserstein features to {out_path}")

if __name__ == "__main__":
    main()
