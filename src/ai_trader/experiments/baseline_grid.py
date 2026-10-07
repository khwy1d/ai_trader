"""Baselines vs a random-entry null, same exits, same costs, same data (decision 0004)."""
import numpy as np
import pandas as pd
from ..backtest.engine import SCENARIOS, label_set, run_backtest, buy_and_hold
from ..backtest.metrics import compute_metrics
from ..strategies.baselines import ma_cross, momentum, mean_reversion, random_signals

KEEP = ["trades", "trades_per_year", "win_rate_pct", "avg_win_over_avg_loss", "profit_factor", "expectancy_R_net", "expectancy_R_gross",
        "total_return_pct", "cagr_pct", "sharpe", "sortino", "max_dd_pct", "calmar", "turnover_per_year",
        "time_in_market_pct", "years_positive", "worst_year_pct"]


def _row(meta, m):
    return {**meta, **{k: m.get(k, np.nan) for k in KEEP}}


def run_grid(frames: dict, barriers, max_hours: float, n_seeds: int = 100, p_random: float = 0.05,
             scenarios=None) -> pd.DataFrame:
    scenarios = scenarios or list(SCENARIOS)
    rows = []
    for (code, tf), df in frames.items():
        df = df.reset_index(drop=True)
        sigs = {"ma_cross": ma_cross(df), "momentum": momentum(df), "mean_rev": mean_reversion(df)}
        for sc_name in scenarios:
            sc = SCENARIOS[sc_name]
            m = compute_metrics(buy_and_hold(df, sc))
            rows.append(_row({"asset": code, "tf": tf, "scenario": sc_name, "k_tp": np.nan, "k_sl": np.nan,
                              "strategy": "buy_and_hold"}, m))
        for k_tp, k_sl in barriers:
            ls = label_set(df, k_tp, k_sl, max_hours, tf)
            for sc_name in scenarios:
                sc = SCENARIOS[sc_name]
                null = pd.DataFrame([compute_metrics(*run_backtest(df, random_signals(len(df), p_random, s), ls, sc))
                                     for s in range(n_seeds)])
                meta = {"asset": code, "tf": tf, "scenario": sc_name, "k_tp": k_tp, "k_sl": k_sl}
                rows.append(_row({**meta, "strategy": "random_median"}, null.median(numeric_only=True).to_dict()))
                rows.append(_row({**meta, "strategy": "random_p95"}, null.quantile(0.95, numeric_only=True).to_dict()))
                for name, sig in sigs.items():
                    eq, tr = run_backtest(df, sig, ls, sc)
                    m = compute_metrics(eq, tr)
                    r = _row({**meta, "strategy": name}, m)
                    r["sharpe_pctile_vs_random"] = 100 * (null["sharpe"] < m["sharpe"]).mean() if np.isfinite(m["sharpe"]) else np.nan
                    r["expR_pctile_vs_random"] = 100 * (null["expectancy_R_net"] < m.get("expectancy_R_net", np.nan)).mean()
                    r["edge_flag" ] = bool(r["expectancy_R_net"] > 0 and r["expR_pctile_vs_random"] >= 95)
                    rows.append(r)
    return pd.DataFrame(rows)
