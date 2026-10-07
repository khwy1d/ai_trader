"""Every experiment/comparison is appended here so the number of trials behind any 'best' result is known."""
import os
import pandas as pd


def log_trial(path: str, name: str, **fields) -> None:
    row = {"logged_at": pd.Timestamp.now(tz="UTC").isoformat(), "name": name, **fields}
    new = pd.DataFrame([row])
    if os.path.exists(path):
        new = pd.concat([pd.read_csv(path), new], ignore_index=True)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    new.to_csv(path, index=False)


def total_comparisons(path: str) -> int:
    if not os.path.exists(path):
        return 0
    return int(pd.read_csv(path).get("n_comparisons", pd.Series(dtype=float)).fillna(0).sum())
