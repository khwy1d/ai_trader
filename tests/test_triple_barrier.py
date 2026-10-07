import numpy as np
import pandas as pd
import pytest
from ai_trader.labeling.triple_barrier import label_from_arrays, finalize_dev

TF = 15


def frame(rows, start="2024-01-01 00:00", drop=()):
    ts = pd.date_range(start, periods=len(rows) + len(drop), freq=f"{TF}min", tz="UTC")
    ts = ts.delete(list(drop)) if drop else ts
    o, h, l, c = (np.array(x, float) for x in zip(*rows))
    return ts.astype("datetime64[ns, UTC]"), o, h, l, c


def run(rows, side=1, k_tp=2.0, k_sl=1.0, max_hours=2.0, atr_val=1.0, drop=()):
    ts, o, h, l, c = frame(rows, drop=drop)
    a = np.full(len(ts), atr_val)
    return label_from_arrays(ts, o, h, l, c, a, side, k_tp, k_sl, max_hours, TF)

FLAT = (100, 100.2, 99.8, 100)


def test_long_take_profit_first():
    r = run([FLAT, (100, 100.3, 99.7, 100), (100, 102.5, 99.5, 102)]).iloc[0]
    assert r["outcome"] == 1 and r["entry_price"] == 100 and r["r_mult"] == pytest.approx(2.0) and r["valid"]


def test_long_stop_loss_first():
    r = run([FLAT, (100, 100.3, 99.7, 100), (100, 100.5, 98.5, 99)]).iloc[0]
    assert r["outcome"] == -1 and r["r_mult"] == pytest.approx(-1.0)


def test_both_barriers_same_bar_is_stop_and_flagged():
    r = run([FLAT, (100, 100.3, 99.7, 100), (100, 103, 98, 100)]).iloc[0]
    assert r["outcome"] == -1 and bool(r["ambiguous"])


def test_short_mirrors_long():
    r = run([FLAT, (100, 100.3, 99.7, 100), (100, 100.5, 97.5, 98)], side=-1).iloc[0]
    assert r["outcome"] == 1 and r["r_mult"] == pytest.approx(2.0)
    r = run([FLAT, (100, 100.3, 99.7, 100), (100, 101.5, 99.5, 101)], side=-1).iloc[0]
    assert r["outcome"] == -1 and r["r_mult"] == pytest.approx(-1.0)


def test_gap_through_stop_exits_at_open_not_at_stop():
    r = run([FLAT, (100, 100.2, 99.8, 100), (97, 97.5, 96.5, 97)]).iloc[0]
    assert r["outcome"] == -1 and r["exit_price"] == 97 and r["r_mult"] == pytest.approx(-3.0)


def test_entry_is_next_open_not_signal_close():
    r = run([(100, 100.2, 99.8, 100.9), (101.2, 101.4, 100.9, 101.1), (101.1, 101.2, 101.0, 101.1)], max_hours=0.25).iloc[0]
    assert r["entry_price"] == 101.2


def test_timeout_exits_at_close_and_is_valid():
    r = run([FLAT] * 20, max_hours=1.0).iloc[0]
    assert r["outcome"] == 0 and r["valid"] and r["exit_price"] == 100


def test_truncated_at_end_of_data_is_invalid():
    r = run([FLAT] * 4, max_hours=5.0).iloc[0]
    assert not r["valid"]


def test_gap_inside_path_is_invalid():
    r = run([FLAT] * 12, max_hours=2.0, drop=(4,)).iloc[0]
    assert not r["valid"]


def test_gap_right_after_signal_is_invalid():
    r = run([FLAT] * 12, max_hours=2.0, drop=(1,)).iloc[0]
    assert not r["valid"]


def test_future_bars_after_resolution_do_not_change_label():
    base = [FLAT, (100, 100.3, 99.7, 100), (100, 102.5, 99.5, 102)]
    a = run(base + [FLAT] * 10)
    b = run(base + [(50, 200, 1, 80)] * 10)
    cols = ["outcome", "exit_price", "r_mult", "ambiguous"]
    pd.testing.assert_frame_equal(a.loc[[0], cols], b.loc[[0], cols])


def test_finalize_dev_purges_labels_resolving_after_boundary():
    lab = run([FLAT] * 30, max_hours=1.0)
    cut = lab["signal_ts"].iloc[10]
    kept = finalize_dev(lab, cut)
    assert (kept["resolved_at"] <= cut).all() and (kept["signal_ts"] < cut).all()
    assert len(kept) < 10
