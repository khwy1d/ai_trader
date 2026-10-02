"""Canonical OHLCV schema. Every provider must return exactly this."""
import pandas as pd

OHLCV_COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]

def empty_ohlcv() -> pd.DataFrame:
    df = pd.DataFrame({c: [] for c in OHLCV_COLUMNS})
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df.astype({c: "float64" for c in OHLCV_COLUMNS[1:]})

def check_schema(df: pd.DataFrame) -> None:
    """Raise if columns are wrong. Does NOT silently fix data."""
    if list(df.columns) != OHLCV_COLUMNS:
        raise ValueError(f"Bad columns: {list(df.columns)}; expected {OHLCV_COLUMNS}")
    if str(df["timestamp"].dtype) != "datetime64[ns, UTC]":
        raise ValueError(f"timestamp must be tz-aware UTC, got {df['timestamp'].dtype}")
