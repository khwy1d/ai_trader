"""Untuned baseline signals. Value at bar i uses data up to the close of bar i only.
Signal +1 = go long, -1 = go short, 0 = nothing. Parameters are fixed in decision 0004 before any result is seen."""
import numpy as np
import pandas as pd


def ma_cross(df: pd.DataFrame, fast: int = 20, slow: int = 50) -> np.ndarray:
    f = df["close"].rolling(fast).mean()
    s = df["close"].rolling(slow).mean()
    up = (f > s) & (f.shift(1) <= s.shift(1))
    dn = (f < s) & (f.shift(1) >= s.shift(1))
    return (up.astype(np.int8) - dn.astype(np.int8)).to_numpy()


def momentum(df: pd.DataFrame, n: int = 48) -> np.ndarray:
    r = df["close"] / df["close"].shift(n) - 1
    return np.sign(r).fillna(0).astype(np.int8).to_numpy()


def mean_reversion(df: pd.DataFrame, n: int = 50, z: float = 2.0) -> np.ndarray:
    m = df["close"].rolling(n).mean()
    sd = df["close"].rolling(n).std()
    zs = (df["close"] - m) / sd
    return ((zs < -z).astype(np.int8) - (zs > z).astype(np.int8)).to_numpy()


def random_signals(n: int, p: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    fire = rng.random(n) < p
    side = np.where(rng.random(n) < 0.5, 1, -1)
    return np.where(fire, side, 0).astype(np.int8)
