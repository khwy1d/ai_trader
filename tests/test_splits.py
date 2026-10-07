from pathlib import Path
import pandas as pd
import pytest
from ai_trader.splits import load_splits, dev_only, assert_no_final_test

CFG = Path(__file__).resolve().parents[1] / "configs" / "splits.yaml"


def test_frozen_values():
    s = load_splits(CFG)
    assert s.data_start == pd.Timestamp("2020-07-01", tz="UTC")
    assert s.dev_end == pd.Timestamp("2025-01-01", tz="UTC")
    assert s.final_test_end == pd.Timestamp("2026-07-01", tz="UTC")
    assert s.max_trade_hours == 120


def test_dev_only_excludes_final_test_and_guard_raises():
    s = load_splits(CFG)
    ts = pd.to_datetime(["2024-12-31 23:45", "2025-01-01 00:00"], utc=True)
    df = pd.DataFrame({"timestamp": ts, "x": [1, 2]})
    out = dev_only(df, s)
    assert len(out) == 1 and out["timestamp"].iloc[0] == ts[0]
    assert_no_final_test(out, s)
    with pytest.raises(ValueError):
        assert_no_final_test(df, s)
