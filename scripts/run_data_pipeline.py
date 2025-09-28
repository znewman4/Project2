import sys
from pathlib import Path

# Add repo root to Python path (parent of 'scripts/')
repo_root = Path(__file__).resolve().parents[1]
sys.path.append(str(repo_root))

from src.utils.io import load_config
from src.data_pipeline.ingest import fetch_ohlcv, save_raw

if __name__ == "__main__":
    cfg = load_config("configs/data.yaml")

    df = fetch_ohlcv(cfg)
    path = save_raw(df, cfg)

    print(f"✅ Raw data saved to {path}")
    print(df.head())
