"""Single-asset, one-position-at-a-time backtester using the triple-barrier exit rules.

Rules (decision 0004):
- Signal at bar close i -> entry at OPEN of bar i+1 (decision 0002). Signals while in a trade are ignored.
- Exits: take-profit / stop-loss / 120 h cap, exactly as in ai_trader.labeling.triple_barrier.
- Size: risk_per_trade / stop distance, capped at max_leverage. Costs: (fee + slippage) per side on traded notional.
- Equity is marked to market at every bar close (intrabar extremes are not marked).
- Not modelled yet: funding, borrow fees, liquidation, order-book depth, partial fills.
"""
from dataclasses import dataclass
import numpy as np
import pandas as pd
from ..labeling.triple_barrier import make_labels

try:
    from numba import njit
except ImportError:
    def njit(f=None, **kw):
        return f if f is not None else (lambda g: g)


@dataclass(frozen=True)
class Scenario:
    name: str
    fee_bps: float
    slip_bps: float
    allow_short: bool
    max_leverage: float
    risk_per_trade: float = 0.01

    @property
    def cost(self) -> float:
        return (self.fee_bps + self.slip_bps) / 1e4


# Fees: Kraken schedule (support.kraken.com, July 2026). Slippage 3 bps per side is an ASSUMPTION to be refined.
SCENARIOS = {
    "perp_A": Scenario("perp_A", fee_bps=5.0, slip_bps=3.0, allow_short=True, max_leverage=3.0),
    "spot_B": Scenario("spot_B", fee_bps=22.0, slip_bps=3.0, allow_short=False, max_leverage=1.0),
    "spot_C": Scenario("spot_C", fee_bps=38.0, slip_bps=3.0, allow_short=False, max_leverage=1.0),
}


def label_set(df, k_tp, k_sl, max_hours, tf_minutes, atr_period=14) -> dict:
    n = len(df)
    out = {k: np.zeros((2, n), dtype=t) for k, t in
           [("valid", np.bool_), ("exit_idx", np.int64), ("outcome", np.int8)]}
    for k in ("exit_px", "entry_px", "sl_dist"):
        out[k] = np.full((2, n), np.nan)
    for r, side in enumerate((1, -1)):
        lab = make_labels(df, side, k_tp, k_sl, max_hours, tf_minutes, atr_period)
        idx = lab["signal_idx"].to_numpy()
        out["valid"][r, idx] = lab["valid"].to_numpy()
        out["exit_idx"][r, idx] = lab["exit_idx"].to_numpy()
        out["outcome"][r, idx] = lab["outcome"].to_numpy()
        out["exit_px"][r, idx] = lab["exit_price"].to_numpy()
        out["entry_px"][r, idx] = lab["entry_price"].to_numpy()
        out["sl_dist"][r, idx] = (k_sl * lab["atr"] / lab["entry_price"]).to_numpy()
    return out


@njit
def _simulate(sig, c, valid, exit_idx, exit_px, entry_px, sl_dist, outcome, cost, risk, lev_cap, allow_short):
    n = len(sig)
    eq = np.full(n, np.nan)
    cap = n // 2 + 2
    t_e = np.zeros(cap, np.int64)
    t_x = np.zeros(cap, np.int64)
    t_side = np.zeros(cap, np.int8)
    t_lev = np.zeros(cap)
    t_pre = np.zeros(cap)
    t_rn = np.zeros(cap)
    t_rg = np.zeros(cap)
    t_out = np.zeros(cap, np.int8)
    t_pnl = np.zeros(cap)
    k = 0
    cur = 1.0
    i = 0
    while i < n - 1:
        if np.isnan(eq[i]):
            eq[i] = cur
        s = sig[i]
        if s != 0 and (s == 1 or allow_short):
            r = 0 if s == 1 else 1
            if valid[r, i]:
                e = i + 1
                x = exit_idx[r, i]
                p0 = entry_px[r, i]
                px = exit_px[r, i]
                sd = sl_dist[r, i]
                lev = min(risk / sd, lev_cap)
                N = lev * cur
                eq1 = cur - N * cost
                for t in range(e, x):
                    eq[t] = eq1 + s * N * (c[t] - p0) / p0
                eq_exit = eq1 + s * N * (px - p0) / p0 - N * (px / p0) * cost
                eq[x] = eq_exit
                t_e[k] = e
                t_x[k] = x
                t_side[k] = s
                t_lev[k] = lev
                t_pre[k] = cur
                t_pnl[k] = eq_exit / cur - 1.0
                t_rn[k] = (eq_exit - cur) / (N * sd)
                t_rg[k] = s * (px - p0) / p0 / sd
                t_out[k] = outcome[r, i]
                k += 1
                cur = eq_exit
                i = x
                continue
        i += 1
    last = 1.0
    for t in range(n):
        if np.isnan(eq[t]):
            eq[t] = last
        else:
            last = eq[t]
    return eq, k, t_e, t_x, t_side, t_lev, t_pre, t_pnl, t_rn, t_rg, t_out


def run_backtest(df: pd.DataFrame, signals: np.ndarray, ls: dict, scen: Scenario):
    c = np.ascontiguousarray(df["close"].to_numpy(), dtype=np.float64)
    sig = np.ascontiguousarray(signals, dtype=np.int8)
    eq, k, te, tx, ts_, tl, tp, tpnl, trn, trg, tout = _simulate(
        sig, c, ls["valid"], ls["exit_idx"], ls["exit_px"], ls["entry_px"], ls["sl_dist"], ls["outcome"],
        scen.cost, scen.risk_per_trade, scen.max_leverage, scen.allow_short)
    ts = df["timestamp"].reset_index(drop=True)
    equity = pd.Series(eq, index=pd.DatetimeIndex(ts), name="equity")
    trades = pd.DataFrame({
        "entry_ts": ts.iloc[te[:k]].to_numpy(), "exit_ts": ts.iloc[tx[:k]].to_numpy(),
        "bars": tx[:k] - te[:k] + 1, "side": ts_[:k], "leverage": tl[:k], "equity_before": tp[:k], "pnl": tpnl[:k],
        "r_net": trn[:k], "r_gross": trg[:k], "outcome": tout[:k],
    })
    return equity, trades


def buy_and_hold(df: pd.DataFrame, scen: Scenario) -> pd.Series:
    cost = scen.cost
    eq = (1 - cost) * df["close"].to_numpy() / df["open"].iloc[0]
    eq[-1] *= (1 - cost)
    return pd.Series(eq, index=pd.DatetimeIndex(df["timestamp"]), name="equity")
