"""Gap table: documents missing bars. Never fills them."""
import pandas as pd

GAP_COLUMNS = ["last_bar_before", "first_bar_after", "missing_bars", "gap_hours"]

def find_gaps(df: pd.DataFrame, tf: int) -> pd.DataFrame:
    step = pd.Timedelta(minutes=tf)
    df = df.reset_index(drop=True)
    ts = df["timestamp"]
    idx = ts.diff()[lambda s: s > step].index
    out = pd.DataFrame({
        "last_bar_before": ts.loc[idx - 1].reset_index(drop=True),
        "first_bar_after": ts.loc[idx].reset_index(drop=True),
    })
    out["missing_bars"] = ((out["first_bar_after"] - out["last_bar_before"]) / step).round().astype(int) - 1
    out["gap_hours"] = out["missing_bars"] * tf / 60
    return out[GAP_COLUMNS]
