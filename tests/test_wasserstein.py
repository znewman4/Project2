import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_parquet("data/processed/btcusdt_with_wasserstein.parquet")

plt.figure(figsize=(12,4))
plt.plot(df["wasserstein_shift"], label="Wasserstein Drift", color="tab:red")
plt.title("Rolling Wasserstein Drift (L1 Baseline)")
plt.legend()
plt.show()
