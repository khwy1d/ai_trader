"""Frozen time split. Development code must only ever see data before dev_end."""
from dataclasses import dataclass
from pathlib import Path
import pandas as pd
import yaml


@dataclass(frozen=True)
class Splits:
    data_start: pd.Timestamp
    dev_end: pd.Timestamp
    final_test_end: pd.Timestamp
    max_trade_hours: int


def load_splits(path) -> Splits:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return Splits(
        data_start=pd.Timestamp(raw["data_start"]),
        dev_end=pd.Timestamp(raw["dev_end"]),
        final_test_end=pd.Timestamp(raw["final_test_end"]),
        max_trade_hours=int(raw["max_trade_hours"]),
    )


def dev_only(df: pd.DataFrame, splits: Splits, ts_col: str = "timestamp") -> pd.DataFrame:
    m = (df[ts_col] >= splits.data_start) & (df[ts_col] < splits.dev_end)
    return df.loc[m].reset_index(drop=True)


def assert_no_final_test(df: pd.DataFrame, splits: Splits, ts_col: str = "timestamp") -> None:
    if len(df) and df[ts_col].max() >= splits.dev_end:
        raise ValueError("Final-test data detected in a development dataset.")
