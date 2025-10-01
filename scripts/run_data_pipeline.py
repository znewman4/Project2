import logging
from pathlib import Path

from src.utils.io import load_config
from src.data_pipeline.ingest import seed_ohlcv, append_new_ohlcv
from src.data_pipeline.clean import clean_and_resample, save_clean

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)
logger = logging.getLogger("run_data_pipeline")

# Map CCXT freq -> pandas freq
FREQ_MAP = {
    "1m": "1min",
    "5m": "5min",
    "15m": "15min",
    "30m": "30min",
    "1h": "1H",
    "4h": "4H",
    "1d": "1D"
}

def main():
    # 1. Load config
    cfg = load_config("configs/data.yaml")
    ex_cfg = cfg["data"]

    # 2. Build file paths
    symbol = ex_cfg["symbol"].replace("/", "").lower()
    timeframe = ex_cfg["timeframe"]

    raw_dir = Path(ex_cfg["paths"]["raw"])
    interim_dir = Path(ex_cfg["paths"]["interim"])

    raw_file = raw_dir / f"{symbol}_{timeframe}.parquet"
    interim_file = interim_dir / f"{symbol}_{timeframe}.parquet"

    # 3. Ingest (seed if first time, else append)
    if not raw_file.exists():
        logger.info("Seeding dataset from %s to %s",
                    ex_cfg["start_date"], ex_cfg.get("end_date", "now"))
        df = seed_ohlcv(cfg, raw_file)
    else:
        logger.info("Appending new data to %s", raw_file)
        df = append_new_ohlcv(cfg, raw_file)

    # 4. Clean
    pandas_freq = FREQ_MAP.get(timeframe, timeframe)
    logger.info("Cleaning dataset with pandas freq=%s", pandas_freq)
    df_clean = clean_and_resample(df, freq=pandas_freq)

    # 5. Save cleaned to interim
    save_clean(df_clean, interim_file)

    logger.info("✅ Pipeline complete: %d rows saved to %s",
                len(df_clean), interim_file)

if __name__ == "__main__":
    main()
