import numpy as np
import pandas as pd
import pytest
from ai_trader.backtest.engine import Scenario, run_backtest, label_set, buy_and_hold
from ai_trader.backtest.metrics import compute_metrics


def make_df(n, close=None):
    ts = pd.date_range("2024-01-01", periods=n, freq="60min", tz="UTC").astype("datetime64[ns, UTC]")
    c = np.full(n, 100.0) if close is None else np.asarray(close, float)
    return pd.DataFrame({"timestamp": ts, "open": c, "high": c, "low": c, "close": c})


def empty_ls(n):
    return {"valid": np.zeros((2, n), bool), "exit_idx": np.zeros((2, n), np.int64),
            "exit_px": np.full((2, n), np.nan), "entry_px": np.full((2, n), np.nan),
            "sl_dist": np.full((2, n), np.nan), "outcome": np.zeros((2, n), np.int8)}


def put(ls, r, i, x, p0, px, sd, out=1):
    ls["valid"][r, i] = True
    ls["exit_idx"][r, i] = x
    ls["entry_px"][r, i] = p0
    ls["exit_px"][r, i] = px
    ls["sl_dist"][r, i] = sd
    ls["outcome"][r, i] = out


SC = Scenario("t", fee_bps=10, slip_bps=0, allow_short=True, max_leverage=5.0)


def test_no_signals_flat_equity():
    df = make_df(10)
    eq, tr = run_backtest(df, np.zeros(10, np.int8), empty_ls(10), SC)
    assert (eq == 1.0).all() and len(tr) == 0


def test_long_trade_exact_numbers_with_costs():
    df = make_df(6, [100, 100, 101, 102, 102, 102])
    ls = empty_ls(6)
    put(ls, 0, 1, 3, 100.0, 102.0, 0.01)
    sig = np.zeros(6, np.int8); sig[1] = 1
    eq, tr = run_backtest(df, sig, ls, SC)
    assert eq.iloc[2] == pytest.approx(0.999 + 0.01)
    assert eq.iloc[3] == pytest.approx(0.999 + 0.02 - 1.02 * 0.001)
    assert tr["pnl"].iloc[0] == pytest.approx(0.999 + 0.02 - 1.02 * 0.001 - 1)
    assert tr["r_net"].iloc[0] == pytest.approx(tr["pnl"].iloc[0] / 0.01)
    assert eq.iloc[-1] == pytest.approx(eq.iloc[3])


def test_short_profit_when_price_falls():
    df = make_df(6, [100, 100, 99, 98, 98, 98])
    ls = empty_ls(6)
    put(ls, 1, 1, 3, 100.0, 98.0, 0.01)
    sig = np.zeros(6, np.int8); sig[1] = -1
    eq, tr = run_backtest(df, sig, ls, SC)
    assert tr["pnl"].iloc[0] == pytest.approx(0.02 - 0.001 - 0.98 * 0.001) and tr["side"].iloc[0] == -1


def test_spot_scenario_ignores_short_signals():
    df = make_df(6)
    ls = empty_ls(6)
    put(ls, 1, 1, 3, 100.0, 98.0, 0.01)
    sig = np.zeros(6, np.int8); sig[1] = -1
    spot = Scenario("s", 10, 0, False, 1.0)
    eq, tr = run_backtest(df, sig, ls, spot)
    assert len(tr) == 0


def test_signals_inside_trade_ignored_but_signal_on_exit_bar_allowed():
    df = make_df(10)
    ls = empty_ls(10)
    put(ls, 0, 1, 4, 100.0, 100.0, 0.01)
    put(ls, 0, 2, 5, 100.0, 100.0, 0.01)
    put(ls, 0, 4, 7, 100.0, 100.0, 0.01)
    sig = np.zeros(10, np.int8); sig[[1, 2, 4]] = 1
    eq, tr = run_backtest(df, sig, ls, SC)
    assert len(tr) == 2 and list(tr["bars"]) == [3, 3]
    assert tr["entry_ts"].iloc[1] > tr["exit_ts"].iloc[0]


def test_leverage_capped_and_invalid_label_skipped():
    df = make_df(10)
    ls = empty_ls(10)
    put(ls, 0, 1, 3, 100.0, 100.0, 0.001)
    ls["valid"][0, 5] = False
    sig = np.zeros(10, np.int8); sig[[1, 5]] = 1
    eq, tr = run_backtest(df, sig, ls, SC)
    assert len(tr) == 1 and tr["leverage"].iloc[0] == 5.0


def test_equity_end_equals_product_of_trades_on_random_walk():
    rng = np.random.default_rng(3)
    n = 4000
    c = 100 * np.exp(np.cumsum(rng.normal(0, 0.004, n)))
    o = np.r_[c[0], c[:-1]]
    df = make_df(n, c).assign(open=o, high=np.maximum(o, c) * 1.001, low=np.minimum(o, c) * 0.999)
    ls = label_set(df, 2.0, 1.0, 120, 60)
    sig = np.where(rng.random(n) < 0.05, 1, 0).astype(np.int8)
    eq, tr = run_backtest(df, sig, ls, SC)
    assert len(tr) > 20
    assert eq.iloc[-1] == pytest.approx(np.prod(1 + tr["pnl"]))
    assert (tr["entry_ts"].iloc[1:].to_numpy() > tr["exit_ts"].iloc[:-1].to_numpy()).all()
    assert (eq > 0).all()


def test_metrics_known_values():
    idx = pd.date_range("2024-01-01", periods=4, freq="1D", tz="UTC")
    eq = pd.Series([1.0, 1.1, 0.99, 1.2], index=idx)
    m = compute_metrics(eq)
    assert m["max_dd_pct"] == pytest.approx(100 * (0.99 / 1.1 - 1))
    assert m["total_return_pct"] == pytest.approx(20.0)


def test_buy_and_hold_pays_costs_twice():
    df = make_df(5)
    eq = buy_and_hold(df, Scenario("x", 10, 0, False, 1.0))
    assert eq.iloc[-1] == pytest.approx((1 - 0.001) ** 2)
