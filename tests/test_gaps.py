import pandas as pd
from ai_trader.data.gaps import find_gaps

def frame(drop):
    ts = pd.date_range("2024-01-01", periods=20, freq="15min", tz="UTC").delete(drop)
    return pd.DataFrame({"timestamp": ts.astype("datetime64[ns, UTC]")})

def test_gaps_found_with_correct_sizes():
    g = find_gaps(frame([5, 6, 10]), 15)
    assert list(g["missing_bars"]) == [2, 1]
    assert list(g["gap_hours"]) == [0.5, 0.25]

def test_no_gaps_returns_empty_table():
    assert len(find_gaps(frame([]), 15)) == 0

def test_input_is_not_modified():
    df = frame([5])
    before = df.copy()
    find_gaps(df, 15)
    pd.testing.assert_frame_equal(df, before)
