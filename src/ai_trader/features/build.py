"""Causal features: the value at row i uses data up to the close of bar i only (verified by a prefix-invariance test)."""
import numpy as np
import pandas as pd

WARMUP = 100


def build_features(df: pd.DataFrame, tf_minutes: int) -> pd.DataFrame:
    o, h, l, c, v = (df[k].astype(float) for k in ("open", "high", "low", "close", "volume"))
    pc = c.shift(1)
    f = {}
    lr = np.log(c).diff()
    for n in (1, 3, 6, 12, 24, 48):
        f[f"ret_{n}"] = np.log(c / c.shift(n))
    f["vol_12"] = lr.rolling(12).std()
    f["vol_48"] = lr.rolling(48).std()
    f["vol_ratio"] = f["vol_12"] / f["vol_48"]
    tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    f["atr_pct"] = atr / c
    rng = (h - l).where((h - l) > 0)
    f["range_pct"] = (h - l) / c
    f["body_pct"] = (c - o) / c
    f["close_loc"] = ((c - l) / rng).fillna(0.5)
    f["upper_wick"] = ((h - np.maximum(o, c)) / rng).fillna(0.0)
    f["lower_wick"] = ((np.minimum(o, c) - l) / rng).fillna(0.0)
    sma20, sma50 = c.rolling(20).mean(), c.rolling(50).mean()
    f["dist_sma20_atr"] = (c - sma20) / atr
    f["dist_sma50_atr"] = (c - sma50) / atr
    f["z_50"] = (c - sma50) / c.rolling(50).std()
    d = c.diff()
    up = d.clip(lower=0).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    f["rsi_14"] = 100 - 100 / (1 + up / dn)
    lv = np.log1p(v)
    f["vol_z_48"] = (lv - lv.rolling(48).mean()) / lv.rolling(48).std()
    lo48, hi48 = l.rolling(48).min(), h.rolling(48).max()
    f["donchian_pos_48"] = (c - lo48) / (hi48 - lo48)
    f["dd_from_high_100"] = c / h.rolling(100).max() - 1
    if tf_minutes < 1440:
        ts = pd.DatetimeIndex(df["timestamp"])
        hr = ts.hour + ts.minute / 60
        f["hour_sin"] = np.sin(2 * np.pi * hr / 24).to_numpy()
        f["hour_cos"] = np.cos(2 * np.pi * hr / 24).to_numpy()
    ts = pd.DatetimeIndex(df["timestamp"])
    f["dow_sin"] = np.sin(2 * np.pi * ts.dayofweek / 7).to_numpy()
    f["dow_cos"] = np.cos(2 * np.pi * ts.dayofweek / 7).to_numpy()
    out = pd.DataFrame(f, index=df.index).replace([np.inf, -np.inf], np.nan)
    out.iloc[:WARMUP] = np.nan
    return out
