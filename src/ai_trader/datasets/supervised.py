"""Supervised dataset for one asset/timeframe/side/barrier set.
y = 1 if the take-profit barrier was hit before the stop (and the cap); r_mult = realized R before costs (stop = -1R)."""
import numpy as np
import pandas as pd
from ..labeling.triple_barrier import make_labels
from ..features.build import build_features

REQUIRED = ["signal_idx", "signal_ts", "valid", "exit_idx", "outcome", "entry_price", "exit_price", "atr"]


def build_dataset(df: pd.DataFrame, side: int, k_tp: float, k_sl: float, max_hours: float, tf_minutes: int, feats=None):
    df = df.reset_index(drop=True)
    feats = build_features(df, tf_minutes) if feats is None else feats
    lab = make_labels(df, side, k_tp, k_sl, max_hours, tf_minutes)
    miss = [c for c in REQUIRED if c not in lab.columns]
    if miss:
        raise KeyError(f"labeler output lacks columns: {miss}")
    lab = lab[lab["valid"].astype(bool)]
    X = feats.iloc[lab["signal_idx"].to_numpy()].reset_index(drop=True)
    lab = lab.reset_index(drop=True)
    ok = np.isfinite(X.to_numpy(dtype=float)).all(axis=1)
    X, lab = X[ok].reset_index(drop=True), lab[ok].reset_index(drop=True)
    sl_frac = k_sl * lab["atr"] / lab["entry_price"]
    r = side * (lab["exit_price"] - lab["entry_price"]) / lab["entry_price"] / sl_frac
    meta = pd.DataFrame({
        "signal_idx": lab["signal_idx"].to_numpy(), "signal_ts": pd.DatetimeIndex(lab["signal_ts"]),
        "exit_idx": lab["exit_idx"].to_numpy(), "outcome": lab["outcome"].to_numpy(),
        "y": (r >= k_tp / k_sl - 1e-6).astype(int).to_numpy(), "r_mult": r.to_numpy(),
        "sl_dist": sl_frac.to_numpy(), "side": side,
    })
    assert (meta["exit_idx"] > meta["signal_idx"]).all()
    return X, meta
