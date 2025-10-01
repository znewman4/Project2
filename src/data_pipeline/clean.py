import logging
import pandas as pd
from pathlib import Path

logger = logging.getLogger(__name__)

def enforce_continuity(df: pd.DataFrame, freq: str) -> pd.DataFrame:
    """Ensure time series continuity and log anomalies."""
    df = df.sort_index()
    full_index = pd.date_range(df.index.min(), df.index.max(), freq=freq)

    missing = full_index.difference(df.index)
    if len(missing) > 0:
        logger.warning("⚠️ Detected %d missing bars (%.2f%%). Example: %s",
                       len(missing),
                       len(missing) / len(full_index) * 100,
                       missing[:5].tolist())

    return df.reindex(full_index)

def clean_and_resample(df: pd.DataFrame, freq="5min") -> pd.DataFrame:
    """Drop duplicates, enforce continuity, optional resample."""
    # Drop dups
    before = len(df)
    df = df[~df.index.duplicated(keep="first")]
    dropped = before - len(df)
    if dropped > 0:
        logger.warning("Dropped %d duplicate rows", dropped)

    # Continuity
    df = enforce_continuity(df, freq)

    return df

def save_clean(df: pd.DataFrame, file_path: Path):
    """Save cleaned dataset to parquet."""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(file_path)
    logger.info("✅ Saved cleaned dataset to %s", file_path)
