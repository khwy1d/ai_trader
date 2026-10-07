"""Performance metrics from a mark-to-market equity curve (starts at 1.0) and a trade table."""
import numpy as np
import pandas as pd


def compute_metrics(equity: pd.Series, trades: pd.DataFrame | None = None) -> dict:
    days = (equity.index[-1] - equity.index[0]).total_seconds() / 86400
    years = max(days / 365.25, 1e-9)
    total = equity.iloc[-1] / 1.0 - 1
    cagr = (equity.iloc[-1]) ** (1 / years) - 1 if equity.iloc[-1] > 0 else -1.0
    daily = equity.resample("1D").last().pct_change().dropna()
    sd = daily.std()
    dd_sd = np.sqrt(np.mean(np.minimum(daily, 0) ** 2))
    sharpe = daily.mean() / sd * np.sqrt(365) if sd > 0 else np.nan
    sortino = daily.mean() / dd_sd * np.sqrt(365) if dd_sd > 0 else np.nan
    maxdd = (equity / equity.cummax() - 1).min()
    ye = equity.groupby(equity.index.year).last()
    yearly = (ye / ye.shift(1).fillna(equity.iloc[0]) - 1)
    m = {
        "total_return_pct": 100 * total, "cagr_pct": 100 * cagr, "sharpe": sharpe, "sortino": sortino,
        "max_dd_pct": 100 * maxdd, "calmar": cagr / abs(maxdd) if maxdd < 0 else np.nan,
        "years_positive": f"{int((yearly > 0).sum())}/{len(yearly)}",
        "worst_year_pct": 100 * yearly.min(),
    }
    if trades is not None and len(trades):
        p = trades["pnl"]
        wins, losses = p[p > 0], p[p < 0]
        mean_eq = equity.mean()
        m.update({
            "trades": len(trades), "trades_per_year": len(trades) / years,
            "win_rate_pct": 100 * (p > 0).mean(),
            "avg_win_over_avg_loss": wins.mean() / abs(losses.mean()) if len(wins) and len(losses) else np.nan,
            "profit_factor": wins.sum() / abs(losses.sum()) if len(losses) and losses.sum() != 0 else np.nan,
            "expectancy_R_net": trades["r_net"].mean(), "expectancy_R_gross": trades["r_gross"].mean(),
            "turnover_per_year": (2 * trades["leverage"] * trades["equity_before"]).sum() / mean_eq / years,
            "time_in_market_pct": 100 * trades["bars"].sum() / len(equity),
        })
    else:
        m["trades"] = 0
    return m
