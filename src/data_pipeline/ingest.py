import time
import logging
import pandas as pd
import ccxt
from pathlib import Path
from src.utils.io import load_config

logger = logging.getLogger(__name__)

def fetch_with_retry(fetch_fn, retries=3, backoff=5):
    """Generic retry wrapper for API calls."""
    for attempt in range(1, retries + 1):
        try:
            return fetch_fn()
        except Exception as e:
            if attempt < retries:
                logger.warning("Fetch attempt %d/%d failed: %s – retrying in %ds",
                               attempt, retries, e, backoff)
                time.sleep(backoff)
            else:
                logger.error("All %d attempts failed: %s", retries, e)
                raise

def append_new_ohlcv(cfg: dict, file_path: Path) -> pd.DataFrame:
    """Append new OHLCV bars to existing dataset, save as parquet."""
    ex_cfg   = cfg["data"]
    exchange = getattr(ccxt, ex_cfg["exchange"])({"enableRateLimit": True})
    symbol   = ex_cfg["symbol"]
    timeframe = ex_cfg["timeframe"]
    limit    = ex_cfg.get("limit", 1000)

    if not file_path.exists():
        raise FileNotFoundError(f"No seed file at {file_path} – seed first")

    # Load existing
    df = pd.read_parquet(file_path)
    last_ts = df.index.max()
    since   = int(last_ts.timestamp() * 1000) + 1

    # Fetch new batch
    batch = fetch_with_retry(
        lambda: exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=limit),
        retries=ex_cfg.get("retries", 3),
        backoff=ex_cfg.get("retry_backoff", 5),
    )

    if not batch:
        logger.info("No new bars returned")
        return df

    df_new = pd.DataFrame(batch, columns=["timestamp","open","high","low","close","volume"])
    df_new["timestamp"] = pd.to_datetime(df_new["timestamp"], unit="ms", utc=True)
    df_new.set_index("timestamp", inplace=True)

    df = pd.concat([df, df_new]).sort_index()
    df = df[~df.index.duplicated(keep="first")]

    file_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(file_path)

    logger.info("Appended %d rows up to %s (total %d rows)",
                len(df_new), df_new.index[-1], len(df))
    return df
