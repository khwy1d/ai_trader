"""Forward-only (expanding) walk-forward with purging.
A training sample is used only if its label is fully resolved BEFORE the test window starts
(exit bar < first test bar - embargo). Test window membership is by signal timestamp."""
from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass
class Fold:
    k: int
    train_idx: np.ndarray
    test_idx: np.ndarray
    test_start: pd.Timestamp
    test_end: pd.Timestamp


def _utc(x):
    if isinstance(x, pd.DatetimeIndex):
        return x.tz_localize("UTC") if x.tz is None else x.tz_convert("UTC")
    t = pd.Timestamp(x)
    return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")


def make_folds(sample_ts, exit_idx, bar_ts, first_test_start, test_months=6, end=None, embargo_bars=0) -> list:
    sample_ts, bar_ts = _utc(pd.DatetimeIndex(sample_ts)), _utc(pd.DatetimeIndex(bar_ts))
    exit_idx = np.asarray(exit_idx)
    s = _utc(first_test_start)
    end_ts = _utc(end) if end is not None else sample_ts.max() + pd.Timedelta(1, "ns")
    folds, k = [], 0
    while s < end_ts:
        e = min(s + pd.DateOffset(months=test_months), end_ts)
        start_bar = bar_ts.searchsorted(s, side="left")
        train = np.nonzero(exit_idx < start_bar - embargo_bars)[0]
        test = np.nonzero((sample_ts >= s) & (sample_ts < e))[0]
        if len(train) and len(test):
            folds.append(Fold(k, train, test, s, e))
            k += 1
        s = e
    return folds


def oof_predict(make_model, X, y, folds):
    from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss
    X, y = np.asarray(X, float), np.asarray(y)
    p = np.full(len(y), np.nan)
    rows = []
    for f in folds:
        ytr, yte = y[f.train_idx], y[f.test_idx]
        if len(np.unique(ytr)) < 2:
            continue
        m = make_model().fit(X[f.train_idx], ytr)
        pr = m.predict_proba(X[f.test_idx])[:, 1]
        p[f.test_idx] = pr
        base = ytr.mean()
        both = len(np.unique(yte)) == 2
        rows.append({"fold": f.k, "test_start": f.test_start.date(), "n_train": len(ytr), "n_test": len(yte),
                     "base_train": base, "base_test": yte.mean(),
                     "auc": roc_auc_score(yte, pr) if both else np.nan,
                     "logloss": log_loss(yte, np.clip(pr, 1e-6, 1 - 1e-6), labels=[0, 1]),
                     "logloss_base": log_loss(yte, np.full(len(yte), np.clip(base, 1e-6, 1 - 1e-6)), labels=[0, 1]),
                     "brier": brier_score_loss(yte, pr), "brier_base": brier_score_loss(yte, np.full(len(yte), base))})
    return p, pd.DataFrame(rows)
