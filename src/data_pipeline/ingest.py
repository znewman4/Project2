import ccxt
import pandas as pd
from pathlib import Path
from src.utils.io import load_config

def fetch_ohlcv(cfg: dict) -> pd.DataFrame:
    exchange_name = cfg["data"]["exchange"]
    symbol = cfg["data"]["symbol"]
    timeframe = cfg["data"]["timeframe"]
    limit = cfg["data"]["limit"]

    exchange = getattr(ccxt, exchange_name)()
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)

    df = pd.DataFrame(
        ohlcv,
        columns=["timestamp", "open", "high", "low", "close", "volume"]
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    return df

def save_raw(df: pd.DataFrame, cfg: dict) -> Path:
    symbol = cfg["data"]["symbol"].replace("/", "").lower()
    timeframe = cfg["data"]["timeframe"]

    outdir = Path(cfg["data"]["paths"]["raw"])
    outdir.mkdir(parents=True, exist_ok=True)

    outpath = outdir / f"{symbol}_{timeframe}.csv"
    df.to_csv(outpath, index=False)
    return outpath
