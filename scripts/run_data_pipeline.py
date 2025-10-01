import logging
from pathlib import Path
from src.utils.io import load_config
from src.data_pipeline.ingest import seed_ohlcv, append_new_ohlcv
from src.data_pipeline.clean import clean_and_resample, save_clean

logging.basicConfig(level=logging.INFO)

if __name__ == "__main__":
    cfg = load_config("configs/data.yaml")
    interim_file = Path(cfg["data"]["paths"]["interim"]) / "btcusdt_5m.parquet"
    processed_file = Path(cfg["data"]["paths"]["processed"]) / "btcusdt_5m.parquet"

    if not interim_file.exists():
        logging.info("Seeding dataset...")
        df = seed_ohlcv(cfg, interim_file)
    else:
        logging.info("Appending new bars...")
        df = append_new_ohlcv(cfg, interim_file)

    logging.info("Cleaning dataset...")
    df_clean = clean_and_resample(df, freq=cfg["data"]["timeframe"])
    save_clean(df_clean, processed_file)

    logging.info("✅ Pipeline complete")
