"""Phase 3H: out-of-sample model evaluation through the real backtest engine, against random-entry nulls.

Per (asset, timeframe, barrier, model):
 - probabilities P(take-profit before stop) per side come from forward-only purged walk-forward folds (decision 0005);
 - entry rule: expected value  EV = p*(k_tp/k_sl) - (1-p) - cost_R > 0, with cost_R = round-trip cost / stop distance
   (timeouts are treated as full losses, which is conservative); if both sides qualify take the larger EV;
 - signals exist only inside test windows and are executed by the same engine, exits and costs as the baselines;
 - nulls: (a) random entries, 50/50 sides; (b) random entries with the same long/short mix as the model.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from ..backtest.engine import SCENARIOS, label_set, run_backtest
from ..backtest.metrics import compute_metrics
from ..strategies.baselines import random_signals
from ..validation.stats import bootstrap_mean_ci, expected_max_normal
from ..validation.walk_forward import make_folds, oof_predict

MODELS = {
    "logreg": lambda: make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=2000)),
    "hgb": lambda: HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=150, min_samples_leaf=100,
                                                  l2_regularization=1.0, random_state=0),
}


def mixed_random_signals(n, p, seed, p_long):
    rng = np.random.default_rng(seed)
    fire = rng.random(n) < p
    side = np.where(rng.random(n) < p_long, 1, -1)
    return np.where(fire, side, 0).astype(np.int8)


def ev_signals(n_bars, probs, metas, k_tp, k_sl, cost_rt):
    ev = {}
    for side in (1, -1):
        arr = np.full(n_bars, -np.inf)
        p, m = probs[side], metas[side]
        ok = ~np.isnan(p)
        arr[m["signal_idx"].to_numpy()[ok]] = p[ok] * (k_tp / k_sl) - (1 - p[ok]) - cost_rt / m["sl_dist"].to_numpy()[ok]
        ev[side] = arr
    sig = np.where((ev[1] > 0) & (ev[1] >= ev[-1]), 1, np.where(ev[-1] > 0, -1, 0)).astype(np.int8)
    return sig


def slice_labels(ls, start):
    out = {k: v[:, start:].copy() for k, v in ls.items()}
    out["exit_idx"] = out["exit_idx"] - start
    return out


def calibration_table(p, y, bins=10):
    d = pd.DataFrame({"p": p, "y": y}).dropna()
    d["bin"] = pd.qcut(d["p"], bins, duplicates="drop")
    return d.groupby("bin", observed=True).agg(n=("y", "size"), mean_p=("p", "mean"), hit_rate=("y", "mean")).round(3)


def evaluate_config(d, datasets, code, tf, k_tp, k_sl, max_hours, first_test, model_names=("logreg", "hgb"),
                    n_seeds=100, p_random=0.05, scen_name="perp_A", drop_cols=("body_pct",)):
    d = d.reset_index(drop=True)
    scen = SCENARIOS[scen_name]
    cost_rt = 2 * scen.cost
    ls = label_set(d, k_tp, k_sl, max_hours, tf)
    start = int(pd.DatetimeIndex(d["timestamp"]).searchsorted(pd.Timestamp(first_test, tz="UTC")))
    ls_sub, d_sub = slice_labels(ls, start), d.iloc[start:].reset_index(drop=True)
    rows, extras = [], {}
    nulls_sym = [run_backtest(d_sub, random_signals(len(d_sub), p_random, s), ls_sub, scen)[1]["r_net"].mean()
                 for s in range(n_seeds)]
    for mname in model_names:
        probs, metas, reps = {}, {}, {}
        for side in (1, -1):
            X, meta = datasets[(code, tf, k_tp, k_sl, side)]
            folds = make_folds(meta["signal_ts"], meta["exit_idx"], d["timestamp"], first_test)
            Xv = X.drop(columns=[c for c in drop_cols if c in X.columns]).to_numpy(float)
            probs[side], reps[side] = oof_predict(MODELS[mname], Xv, meta["y"].to_numpy(), folds)
            metas[side] = meta
        sig = ev_signals(len(d), probs, metas, k_tp, k_sl, cost_rt)[start:]
        eq, tr = run_backtest(d_sub, sig, ls_sub, scen)
        m = compute_metrics(eq, tr)
        row = {"asset": code, "tf": tf, "barrier": f"{k_tp:g}:{k_sl:g}", "model": mname, "trades": m.get("trades", 0)}
        auc = {}
        for side in (1, -1):
            msk = ~np.isnan(probs[side])
            auc[side] = roc_auc_score(metas[side]["y"][msk], probs[side][msk])
        row.update({"auc_long": auc[1], "auc_short": auc[-1]})
        if len(tr) >= 5:
            mean, (lo, hi) = bootstrap_mean_ci(tr["r_net"].to_numpy())
            share_long = float((tr["side"] == 1).mean())
            nulls_mix = [run_backtest(d_sub, mixed_random_signals(len(d_sub), p_random, s, share_long), ls_sub, scen)[1]["r_net"].mean()
                         for s in range(n_seeds)]
            se = tr["r_net"].std(ddof=1) / np.sqrt(len(tr))
            bounds = [(f.test_start, f.test_end) for f in make_folds(metas[1]["signal_ts"], metas[1]["exit_idx"], d["timestamp"], first_test)]
            ex = pd.to_datetime(tr["exit_ts"], utc=True)
            win = [tr.loc[(ex >= a) & (ex < b), "r_net"] for a, b in bounds]
            row.update({"share_long": share_long, "exp_R_net": mean, "exp_R_gross": m["expectancy_R_gross"],
                        "ci_lo": lo, "ci_hi": hi, "z": mean / se if se > 0 else np.nan,
                        "null_sym_med": np.median(nulls_sym), "null_mix_med": np.median(nulls_mix),
                        "null_mix_p95": np.percentile(nulls_mix, 95),
                        "pctile_vs_mix": 100 * np.mean(np.array(nulls_mix) < mean),
                        "windows_pos": f"{sum(len(x) > 0 and x.mean() > 0 for x in win)}/{len(win)}",
                        "sharpe": m["sharpe"], "max_dd_pct": m["max_dd_pct"], "total_return_pct": m["total_return_pct"],
                        "win_rate_pct": m["win_rate_pct"], "profit_factor": m["profit_factor"]})
        rows.append(row)
        extras[mname] = {"probs": probs, "metas": metas, "reports": reps, "trades": tr}
    return rows, extras


def apply_acceptance(res: pd.DataFrame, n_comparisons_total: int) -> pd.DataFrame:
    thr = expected_max_normal(n_comparisons_total)
    r = res.copy()
    r["z_threshold"] = thr
    r["c1_net_pos"] = r["exp_R_net"] > 0
    r["c2_ci_above_null"] = r["ci_lo"] > r[["null_sym_med", "null_mix_med"]].max(axis=1)
    r["c3_windows"] = r["windows_pos"].fillna("0/0").map(lambda s: int(s.split("/")[0]) >= 3)
    r["c4_z"] = r["z"] > thr
    r["pass_single"] = r[["c1_net_pos", "c2_ci_above_null", "c3_windows", "c4_z"]].all(axis=1)
    g = r.groupby(["tf", "barrier", "model"])["pass_single"].transform("all")
    r["accept_0005"] = g & (r.groupby(["tf", "barrier", "model"])["asset"].transform("nunique") >= 2)
    return r
