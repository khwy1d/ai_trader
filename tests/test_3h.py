import numpy as np
import pandas as pd
from ai_trader.backtest.engine import SCENARIOS, label_set, run_backtest
from ai_trader.datasets.supervised import build_dataset
from ai_trader.features.build import build_features
from ai_trader.experiments.model_eval import (ev_signals, slice_labels, evaluate_config, apply_acceptance,
                                              mixed_random_signals, calibration_table)


def synth(n, phi, seed, tf=240):
    rng = np.random.default_rng(seed)
    eps = rng.normal(0, 0.012, n)
    r = np.zeros(n)
    for i in range(1, n):
        r[i] = phi * r[i - 1] + eps[i]
    c = 100 * np.exp(np.cumsum(r))
    o = np.r_[c[0], c[:-1]]
    return pd.DataFrame({"timestamp": pd.date_range("2020-01-01", periods=n, freq=f"{tf}min", tz="UTC").astype("datetime64[ns, UTC]"),
                         "open": o, "high": np.maximum(o, c) * 1.003, "low": np.minimum(o, c) * 0.997, "close": c,
                         "volume": rng.uniform(1, 10, n)})


def test_ev_rule_signs_and_side_choice():
    meta = pd.DataFrame({"signal_idx": [0, 1, 2], "sl_dist": [0.016] * 3})
    probs = {1: np.array([0.5, 0.3, 0.6]), -1: np.array([0.2, 0.7, 0.62])}
    sig = ev_signals(4, probs, {1: meta, -1: meta}, 2.0, 1.0, 0.0016)
    assert list(sig) == [1, -1, -1, 0]


def test_ev_rule_no_trade_when_cost_exceeds_edge():
    meta = pd.DataFrame({"signal_idx": [0], "sl_dist": [0.0016]})
    sig = ev_signals(2, {1: np.array([0.34]), -1: np.array([0.0])}, {1: meta, -1: meta}, 2.0, 1.0, 0.0016)
    assert sig[0] == 0


def test_slice_labels_matches_full_run():
    d = synth(3000, 0.0, 3)
    ls = label_set(d, 2.0, 1.0, 120, 240)
    rng = np.random.default_rng(1)
    start = 1500
    sig = np.where(rng.random(len(d)) < 0.05, np.where(rng.random(len(d)) < 0.5, 1, -1), 0).astype(np.int8)
    sig[:start] = 0
    sc = SCENARIOS["perp_A"]
    _, full = run_backtest(d, sig, ls, sc)
    _, part = run_backtest(d.iloc[start:].reset_index(drop=True), sig[start:], slice_labels(ls, start), sc)
    assert len(full) == len(part) > 10
    assert np.allclose(full["r_net"], part["r_net"])


def test_mixed_random_share():
    s = mixed_random_signals(100000, 0.05, 1, 0.8)
    assert abs((s[s != 0] == 1).mean() - 0.8) < 0.03


def _run(phi, seeds, n=9000):
    rows, ds = [], {}
    for s in seeds:
        d = synth(n, phi, s)
        code = f"S{s}"
        f = build_features(d, 240)
        for side in (1, -1):
            ds[(code, 240, 2.0, 1.0, side)] = build_dataset(d, side, 2.0, 1.0, 120, 240, feats=f)
        r, ex = evaluate_config(d, ds, code, 240, 2.0, 1.0, 120, "2022-06-01", n_seeds=20)
        rows += r
    return apply_acceptance(pd.DataFrame(rows), 124), ex


def test_positive_control_is_detected_and_noise_is_not_accepted():
    pos, ex = _run(0.45, (1, 2))
    assert pos["accept_0005"].all() and (pos["exp_R_net"] > 0.2).all()
    neg, _ = _run(0.0, (1, 2))
    assert not neg["accept_0005"].any()
    ct = calibration_table(ex["logreg"]["probs"][1], ex["logreg"]["metas"][1]["y"].to_numpy())
    assert ct["hit_rate"].iloc[-1] > ct["hit_rate"].iloc[0]
