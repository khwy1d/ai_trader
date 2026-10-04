"""Raw Kraken CSV -> processed OHLCV. No filling, no resampling, no price edits."""
import numpy as np
import pandas as pd
from ai_trader.data.schema import OHLCV_COLUMNS, check_schema

RAW_COLS = ["timestamp", "open", "high", "low", "close", "volume", "trades"]

def load_raw_csv(path) -> pd.DataFrame:
    df = pd.read_csv(path, header=None, names=RAW_COLS)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s", utc=True).astype("datetime64[ns, UTC]")
    return df

def to_processed(raw: pd.DataFrame, start: str) -> pd.DataFrame:
    df = raw[OHLCV_COLUMNS].copy()
    for c in OHLCV_COLUMNS[1:]:
        df[c] = df[c].astype("float64")
    df = df.sort_values("timestamp")
    if df["timestamp"].duplicated().any():
        raise ValueError("Duplicate timestamps in raw data")
    df = df[df["timestamp"] >= pd.Timestamp(start, tz="UTC")].reset_index(drop=True)
    if df.empty:
        raise ValueError("No rows after start date")
    check_schema(df)
    return df

def coverage_report(df: pd.DataFrame, tf: int) -> dict:
    step = pd.Timedelta(minutes=tf)
    ts = df["timestamp"]
    full = pd.date_range(ts.iloc[0], ts.iloc[-1], freq=f"{tf}min")
    present = pd.Series(full.isin(ts), index=full)
    diffs = ts.diff().dropna()
    gaps = diffs[diffs > step]
    return {
        "rows": int(len(df)),
        "expected_rows": int(len(full)),
        "coverage_pct": round(100 * len(df) / len(full), 3),
        "gap_count": int(len(gaps)),
        "longest_gap_hours": float(gaps.max() / pd.Timedelta(hours=1)) if len(gaps) else 0.0,
        "yearly_coverage_pct": (present.groupby(present.index.year).mean() * 100).round(2).to_dict(),
    }

def sanity_report(df: pd.DataFrame) -> dict:
    o, h, l, c, v = (df[x] for x in ["open", "high", "low", "close", "volume"])
    bad = (h < np.maximum(o, c)) | (l > np.minimum(o, c)) | (h < l)
    ret = np.log(c).diff().abs()
    return {
        "bad_ohlc_rows": int(bad.sum()),
        "non_positive_prices": int(((df[["open", "high", "low", "close"]] <= 0).any(axis=1)).sum()),
        "zero_volume_rows": int((v == 0).sum()),
        "negative_volume_rows": int((v < 0).sum()),
        "max_abs_log_return": float(ret.max()),
        "bars_with_abs_return_gt_10pct": int((ret > 0.10).sum()),
    }

def resample_consistency(df_fine, df_coarse, coarse_min: int, fine_min: int) -> dict:
    n = coarse_min // fine_min
    f = df_fine.set_index("timestamp")
    agg = f.resample(f"{coarse_min}min").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
    cnt = f["close"].resample(f"{coarse_min}min").count()
    agg = agg[cnt == n]
    j = agg.join(df_coarse.set_index("timestamp"), how="inner", lsuffix="_f", rsuffix="_c")
    out = {"windows_compared": int(len(j))}
    for col in ["open", "high", "low", "close"]:
        out[f"{col}_mismatch"] = int((~np.isclose(j[col + "_f"], j[col + "_c"], rtol=1e-9, atol=0)).sum())
    out["volume_mismatch"] = int((~np.isclose(j["volume_f"], j["volume_c"], rtol=1e-6, atol=0)).sum())
    return out
