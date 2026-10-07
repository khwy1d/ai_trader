"""Triple-barrier labels with take-profit / stop-loss as the decision, time only as a cap.

Conventions (decisions 0001-0003):
- Signal bar i is known at its close. Entry = OPEN of bar i+1.
- Barriers = entry +/- k * ATR(i); ATR uses bars up to i only.
- A bar touching both barriers is ambiguous: the stop is assumed first (worst case) and flagged.
- If a bar opens beyond a barrier, exit is at that open (gap-through), not at the barrier.
- Labels whose entry or path touches a data gap, or that run past the end of data, are invalid.
"""
import numpy as np
import pandas as pd

try:
    from numba import njit
except ImportError:  # pure-python fallback
    def njit(f=None, **kw):
        return f if f is not None else (lambda g: g)


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    prev_c = df["close"].shift(1)
    tr = pd.concat([df["high"] - df["low"], (df["high"] - prev_c).abs(), (df["low"] - prev_c).abs()], axis=1).max(axis=1)
    return tr.rolling(period).mean()


@njit
def _scan(ts, o, h, l, c, a, side, k_tp, k_sl, max_ns, step_ns):
    n = len(ts)
    outcome = np.zeros(n, np.int8)
    entry_p = np.full(n, np.nan)
    tp_p = np.full(n, np.nan)
    sl_p = np.full(n, np.nan)
    exit_p = np.full(n, np.nan)
    resolved = np.zeros(n, np.int64)
    amb = np.zeros(n, np.bool_)
    valid = np.zeros(n, np.bool_)
    for i in range(n - 1):
        ai = a[i]
        if not (ai > 0):
            continue
        e = i + 1
        entry = o[e]
        tp = entry + side * k_tp * ai
        sl = entry - side * k_sl * ai
        limit = ts[e] + max_ns
        bad = (ts[e] - ts[i]) > step_ns
        res = 0
        ex = np.nan
        ex_i = e
        is_amb = False
        j = e
        done = False
        while j < n and ts[j] <= limit:
            if j > e and (ts[j] - ts[j - 1]) > step_ns:
                bad = True
            if side == 1:
                hit_tp = h[j] >= tp
                hit_sl = l[j] <= sl
                gap_tp = o[j] >= tp
                gap_sl = o[j] <= sl
            else:
                hit_tp = l[j] <= tp
                hit_sl = h[j] >= sl
                gap_tp = o[j] <= tp
                gap_sl = o[j] >= sl
            if j > e and gap_sl:
                res = -1
                ex = o[j]
                done = True
            elif j > e and gap_tp:
                res = 1
                ex = o[j]
                done = True
            elif hit_tp and hit_sl:
                res = -1
                ex = sl
                is_amb = True
                done = True
            elif hit_sl:
                res = -1
                ex = sl
                done = True
            elif hit_tp:
                res = 1
                ex = tp
                done = True
            if done:
                ex_i = j
                break
            j += 1
        truncated = False
        if not done:
            ex_i = j - 1
            ex = c[ex_i]
            truncated = j >= n and ts[n - 1] < limit
        outcome[i] = res
        entry_p[i] = entry
        tp_p[i] = tp
        sl_p[i] = sl
        exit_p[i] = ex
        resolved[i] = ts[ex_i] + step_ns
        amb[i] = is_amb
        valid[i] = (not bad) and (not truncated)
    return outcome, entry_p, tp_p, sl_p, exit_p, resolved, amb, valid


def label_from_arrays(timestamps, o, h, l, c, a, side, k_tp, k_sl, max_hours, tf_minutes):
    ts = pd.DatetimeIndex(timestamps).as_unit("ns").asi8
    step_ns = int(tf_minutes) * 60 * 10**9
    max_ns = int(max_hours * 3600) * 10**9
    f = lambda x: np.ascontiguousarray(x, dtype=np.float64)
    out = _scan(ts, f(o), f(h), f(l), f(c), f(a), int(side), float(k_tp), float(k_sl), max_ns, step_ns)
    outcome, entry_p, tp_p, sl_p, exit_p, resolved, amb, valid = out
    keep = np.isfinite(entry_p)
    a_arr = f(a)
    res = pd.DataFrame({
        "signal_ts": pd.DatetimeIndex(timestamps)[keep],
        "side": side,
        "entry_price": entry_p[keep],
        "atr": a_arr[keep],
        "tp": tp_p[keep],
        "sl": sl_p[keep],
        "outcome": outcome[keep],
        "exit_price": exit_p[keep],
        "resolved_at": pd.to_datetime(resolved[keep], unit="ns", utc=True),
        "ambiguous": amb[keep],
        "valid": valid[keep],
    })
    res["r_mult"] = side * (res["exit_price"] - res["entry_price"]) / (k_sl * res["atr"])
    return res.reset_index(drop=True)


def make_labels(df: pd.DataFrame, side: int, k_tp: float, k_sl: float, max_hours: float,
                tf_minutes: int, atr_period: int = 14) -> pd.DataFrame:
    df = df.reset_index(drop=True)
    a = atr(df, atr_period).to_numpy()
    return label_from_arrays(df["timestamp"], df["open"], df["high"], df["low"], df["close"],
                             a, side, k_tp, k_sl, max_hours, tf_minutes)


def finalize_dev(labels: pd.DataFrame, dev_end: pd.Timestamp) -> pd.DataFrame:
    """Keep valid labels fully resolved before dev_end (purge at the split boundary)."""
    m = labels["valid"] & (labels["signal_ts"] < dev_end) & (labels["resolved_at"] <= dev_end)
    return labels.loc[m].reset_index(drop=True)


def summarize(labels: pd.DataFrame, k_tp: float, k_sl: float) -> dict:
    v = labels[labels["valid"]]
    n = len(v)
    if n == 0:
        return {"n_valid": 0}
    return {
        "n_valid": n,
        "excluded_pct": round(100 * (1 - n / len(labels)), 3),
        "tp_pct": round(100 * (v["outcome"] == 1).mean(), 2),
        "sl_pct": round(100 * (v["outcome"] == -1).mean(), 2),
        "timeout_pct": round(100 * (v["outcome"] == 0).mean(), 2),
        "ambiguous_pct": round(100 * v["ambiguous"].mean(), 2),
        "mean_R_no_cost": round(v["r_mult"].mean(), 4),
        "p_tp_zero_cost_breakeven": round(100 * k_sl / (k_tp + k_sl), 2),
        "median_sl_dist_pct": round(100 * (k_sl * v["atr"] / v["entry_price"]).median(), 3),
        "median_tp_dist_pct": round(100 * (k_tp * v["atr"] / v["entry_price"]).median(), 3),
    }
