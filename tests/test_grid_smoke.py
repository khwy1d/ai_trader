import numpy as np
import pandas as pd
from ai_trader.experiments.baseline_grid import run_grid
from ai_trader.labeling.triple_barrier import make_labels


def walk(n, tf, seed):
    rng = np.random.default_rng(seed)
    c = 100 * np.exp(np.cumsum(rng.normal(0, 0.004, n)))
    o = np.r_[c[0], c[:-1]]
    return pd.DataFrame({"timestamp": pd.date_range("2024-01-01", periods=n, freq=f"{tf}min", tz="UTC").astype("datetime64[ns, UTC]"),
                         "open": o, "high": np.maximum(o, c) * 1.001, "low": np.minimum(o, c) * 0.999, "close": c, "volume": 1.0})


def test_exit_idx_matches_resolved_at():
    df = walk(3000, 60, 1)
    lab = make_labels(df, 1, 2.0, 1.0, 120, 60)
    ok = lab[lab["valid"]]
    ts = df["timestamp"].to_numpy()
    assert (ok["exit_idx"] >= ok["signal_idx"] + 1).all()
    got = pd.to_datetime(ts[ok["exit_idx"].to_numpy()], utc=True) + pd.Timedelta(minutes=60)
    assert (got == ok["resolved_at"]).all()


def test_run_grid_smoke_and_random_null_present():
    res = run_grid({("T", 60): walk(4000, 60, 2)}, [(2.0, 1.0)], 120, n_seeds=5)
    assert {"buy_and_hold", "random_median", "random_p95", "ma_cross", "momentum", "mean_rev"} <= set(res["strategy"])
    assert set(res["scenario"]) == {"perp_A", "spot_B", "spot_C"}
    assert res["edge_flag"].dropna().isin([True, False]).all()
