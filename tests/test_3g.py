import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from ai_trader.features.build import build_features, WARMUP
from ai_trader.datasets.supervised import build_dataset
from ai_trader.validation.walk_forward import make_folds, oof_predict
from ai_trader.validation.stats import bootstrap_mean_ci, expected_max_normal
from ai_trader.experiments.trial_log import log_trial, total_comparisons


def walk(n, tf=240, seed=0):
    rng = np.random.default_rng(seed)
    c = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    o = np.r_[c[0], c[:-1]]
    return pd.DataFrame({"timestamp": pd.date_range("2020-01-01", periods=n, freq=f"{tf}min", tz="UTC").astype("datetime64[ns, UTC]"),
                         "open": o, "high": np.maximum(o, c) * 1.004, "low": np.minimum(o, c) * 0.996, "close": c,
                         "volume": rng.uniform(1, 10, n)})


def test_features_prefix_invariance_no_lookahead():
    df = walk(900)
    full = build_features(df, 240)
    for k in (150, 400, 899):
        pre = build_features(df.iloc[: k + 1].reset_index(drop=True), 240)
        assert np.allclose(pre.iloc[k].to_numpy(float), full.iloc[k].to_numpy(float), equal_nan=True, rtol=1e-9, atol=1e-12)


def test_features_future_change_does_not_alter_past():
    df = walk(900)
    a = build_features(df, 240)
    df2 = df.copy()
    df2.loc[600:, ["open", "high", "low", "close", "volume"]] *= 3
    b = build_features(df2, 240)
    assert np.allclose(a.iloc[:600].to_numpy(float), b.iloc[:600].to_numpy(float), equal_nan=True)


def test_features_nan_only_in_warmup():
    f = build_features(walk(600), 240)
    assert f.iloc[:WARMUP].isna().all().all()
    assert np.isfinite(f.iloc[WARMUP + 5:].to_numpy(float)).all()


def test_dataset_alignment_and_labels():
    df = walk(3000)
    X, meta = build_dataset(df, 1, 2.0, 1.0, 120, 240)
    assert len(X) == len(meta) > 500 and np.isfinite(X.to_numpy(float)).all()
    assert (meta["exit_idx"] > meta["signal_idx"]).all()
    assert (meta.loc[meta.y == 1, "r_mult"] > 0).all()
    assert np.allclose(meta.loc[meta.y == 1, "r_mult"], 2.0, atol=1e-5)
    full = build_features(df.reset_index(drop=True), 240)
    assert np.allclose(X.iloc[10].to_numpy(float), full.iloc[meta["signal_idx"].iloc[10]].to_numpy(float))


def test_folds_are_forward_only_and_purged():
    df = walk(6000)
    X, meta = build_dataset(df, 1, 2.0, 1.0, 120, 240)
    folds = make_folds(meta["signal_ts"], meta["exit_idx"], df["timestamp"], "2020-12-01", test_months=3)
    assert len(folds) >= 3
    prev_end = None
    for f in folds:
        start_bar = pd.DatetimeIndex(df["timestamp"]).searchsorted(f.test_start)
        assert meta["exit_idx"].to_numpy()[f.train_idx].max() < start_bar
        assert meta["signal_ts"].iloc[f.train_idx].max() < f.test_start
        assert not set(f.train_idx) & set(f.test_idx)
        assert (meta["signal_ts"].iloc[f.test_idx] >= f.test_start).all() and (meta["signal_ts"].iloc[f.test_idx] < f.test_end).all()
        if prev_end is not None:
            assert f.test_start == prev_end
        prev_end = f.test_end


def test_noise_model_has_no_skill_and_canary_leak_is_detected():
    df = walk(8000, seed=5)
    X, meta = build_dataset(df, 1, 2.0, 1.0, 120, 240)
    folds = make_folds(meta["signal_ts"], meta["exit_idx"], df["timestamp"], "2021-01-01", test_months=3)
    mk = lambda: LogisticRegression(max_iter=500, C=0.1)
    _, rep = oof_predict(mk, X.to_numpy(float), meta["y"].to_numpy(), folds)
    assert abs(rep["auc"].mean() - 0.5) < 0.05
    leak = np.c_[X.to_numpy(float), meta["r_mult"].to_numpy()]
    _, rep2 = oof_predict(mk, leak, meta["y"].to_numpy(), folds)
    assert rep2["auc"].mean() > 0.95


def test_stats_and_trial_log(tmp_path):
    assert expected_max_normal(100) == pytest.approx(2.5, abs=0.1)
    assert expected_max_normal(1) == 0.0
    r = np.random.default_rng(0).normal(0.1, 1, 400)
    m, (lo, hi) = bootstrap_mean_ci(r)
    assert lo < m < hi
    p = str(tmp_path / "t.csv")
    log_trial(p, "a", n_comparisons=108)
    log_trial(p, "b", n_comparisons=12, note="x")
    assert total_comparisons(p) == 120
