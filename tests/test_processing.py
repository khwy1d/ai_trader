import numpy as np
import pandas as pd
import pytest
from ai_trader.data.processing import (RAW_COLS, to_processed, coverage_report,
                                       sanity_report, resample_consistency)

def raw_frame(start, n, tf=15):
    ts = pd.date_range(start, periods=n, freq=f"{tf}min", tz="UTC").astype("datetime64[ns, UTC]")
    c = 100 + np.arange(n, dtype=float)
    return pd.DataFrame({"timestamp": ts, "open": c, "high": c + 1, "low": c - 1,
                         "close": c + 0.5, "volume": 1.0, "trades": 1})[RAW_COLS]

def test_cut_removes_everything_before_start():
    out = to_processed(raw_frame("2020-06-30 22:00", 40), "2020-07-01")
    assert out["timestamp"].iloc[0] == pd.Timestamp("2020-07-01", tz="UTC")
    assert "trades" not in out.columns

def test_duplicates_rejected():
    raw = raw_frame("2020-07-01", 10)
    raw = pd.concat([raw, raw.iloc[[3]]])
    with pytest.raises(ValueError):
        to_processed(raw, "2020-07-01")

def test_gap_is_reported_not_filled():
    raw = raw_frame("2020-07-01", 20).drop(index=[5, 6])
    out = to_processed(raw, "2020-07-01")
    rep = coverage_report(out, 15)
    assert len(out) == 18
    assert rep["gap_count"] == 1 and rep["expected_rows"] == 20

def test_sanity_detects_bad_ohlc():
    raw = raw_frame("2020-07-01", 5)
    raw.loc[2, "high"] = 0.0
    assert sanity_report(to_processed(raw, "2020-07-01"))["bad_ohlc_rows"] == 1

def test_resample_consistency_zero_mismatch_on_clean_data():
    f = to_processed(raw_frame("2020-07-01", 16), "2020-07-01")
    g = f.set_index("timestamp").resample("60min").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}).reset_index()
    rep = resample_consistency(f, g, 60, 15)
    assert rep["windows_compared"] == 4 and rep["close_mismatch"] == 0
