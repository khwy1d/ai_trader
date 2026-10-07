"""Aggregate OHLCV to a coarser timeframe. Incomplete buckets are dropped, never filled."""
import pandas as pd

AGG = dict(open=("open", "first"), high=("high", "max"), low=("low", "min"),
           close=("close", "last"), volume=("volume", "sum"), n=("open", "size"))


def resample_ohlcv(df: pd.DataFrame, tf_from: int, tf_to: int) -> pd.DataFrame:
    if tf_to % tf_from:
        raise ValueError("tf_to must be a multiple of tf_from")
    ratio = tf_to // tf_from
    d = df.set_index("timestamp")
    bucket = d.index.floor(f"{tf_to}min")
    out = d.groupby(bucket).agg(**AGG)
    out = out[out["n"] == ratio].drop(columns="n")
    out.index.name = "timestamp"
    return out.reset_index()
