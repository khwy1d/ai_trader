"""Kraken REST provider (recent data only: max ~720 candles per call)."""
from __future__ import annotations
import time
import pandas as pd
import requests

from ai_trader.config import load_config
from ai_trader.data.providers.base import DataProvider
from ai_trader.data.schema import OHLCV_COLUMNS, check_schema

class KrakenError(RuntimeError):
    pass

class KrakenProvider(DataProvider):
    name = "kraken"
    URL = "https://api.kraken.com/0/public/OHLC"
    VALID_INTERVALS = {1, 5, 15, 30, 60, 240, 1440, 10080, 21600}

    def __init__(self, session=None, pair_map: dict | None = None, max_retries: int = 3):
        self.session = session or requests.Session()
        if pair_map is None:
            cfg = load_config("data")
            pair_map = {a: v["kraken_pair"] for a, v in cfg["assets"].items()}
        self.pair_map = pair_map
        self.max_retries = max_retries

    def _get(self, params: dict) -> dict:
        last_exc = None
        for attempt in range(self.max_retries):
            try:
                resp = self.session.get(self.URL, params=params, timeout=30)
                resp.raise_for_status()
                return resp.json()
            except requests.RequestException as exc:
                last_exc = exc
                time.sleep(2 ** attempt)
        raise KrakenError(f"Network failure after {self.max_retries} tries: {last_exc}")

    def fetch_ohlcv(self, asset, timeframe_minutes, since=None, now=None):
        if asset not in self.pair_map:
            raise KrakenError(f"Unknown asset {asset!r}; known: {sorted(self.pair_map)}")
        if timeframe_minutes not in self.VALID_INTERVALS:
            raise KrakenError(f"Unsupported interval {timeframe_minutes}")

        params = {"pair": self.pair_map[asset], "interval": timeframe_minutes}
        if since is not None:
            params["since"] = int(pd.Timestamp(since).timestamp())

        payload = self._get(params)
        if payload.get("error"):
            raise KrakenError(f"Kraken API error: {payload['error']}")

        result = payload["result"]
        key = next(k for k in result if k != "last")
        df = pd.DataFrame(
            result[key],
            columns=["timestamp", "open", "high", "low", "close", "vwap", "volume", "trades"],
        )
        df["timestamp"] = pd.to_datetime(df["timestamp"].astype("int64"), unit="s", utc=True)
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = df[col].astype(float)

        now = pd.Timestamp.now(tz="UTC") if now is None else pd.Timestamp(now)
        closed = df["timestamp"] + pd.Timedelta(minutes=timeframe_minutes) <= now
        df = df.loc[closed, OHLCV_COLUMNS].reset_index(drop=True)
        check_schema(df)
        return df
