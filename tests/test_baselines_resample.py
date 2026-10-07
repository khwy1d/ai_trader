import numpy as np
import pandas as pd
from ai_trader.strategies.baselines import ma_cross, momentum, mean_reversion, random_signals
from ai_trader.data.resample import resample_ohlcv


def walk(n, seed=0):
    rng = np.random.default_rng(seed)
    c = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    return pd.DataFrame({"timestamp": pd.date_range("2024-01-01", periods=n, freq="60min", tz="UTC"), "close": c})


def test_baselines_do_not_look_ahead():
    df = walk(600)
    k = 400
    changed = df.copy()
    changed.loc[k + 1:, "close"] *= 1.5
    for fn in (ma_cross, momentum, mean_reversion):
        assert (fn(df)[: k + 1] == fn(changed)[: k + 1]).all()


def test_random_signals_reproducible_and_rate():
    a, b = random_signals(10000, 0.05, 7), random_signals(10000, 0.05, 7)
    assert (a == b).all() and abs((a != 0).mean() - 0.05) < 0.01


def test_resample_values_and_incomplete_bucket_dropped():
    ts = pd.date_range("2024-01-01 00:00", periods=8, freq="60min", tz="UTC")
    df = pd.DataFrame({"timestamp": ts, "open": range(8), "high": [5, 6, 9, 7, 1, 2, 3, 4],
                       "low": [1] * 8, "close": [10 + i for i in range(8)], "volume": [1.0] * 8})
    out = resample_ohlcv(df.drop(index=5), 60, 240)
    assert len(out) == 1
    r = out.iloc[0]
    assert r["open"] == 0 and r["high"] == 9 and r["close"] == 13 and r["volume"] == 4.0
    assert out["timestamp"].iloc[0] == pd.Timestamp("2024-01-01 00:00", tz="UTC")
