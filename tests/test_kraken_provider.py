import pandas as pd
import pytest
from ai_trader.data.providers.kraken import KrakenProvider, KrakenError

T0 = 1_699_999_200  # divisible by 900

class FakeResp:
    def __init__(self, payload): self._p = payload
    def raise_for_status(self): pass
    def json(self): return self._p

class FakeSession:
    def __init__(self, payload):
        self.payload, self.calls = payload, []
    def get(self, url, params=None, timeout=None):
        self.calls.append(params)
        return FakeResp(self.payload)

def row(i):
    return [T0 + 900 * i, "100", "110", "90", "105", "101", "5.5", 12]

def make_provider(payload):
    sess = FakeSession(payload)
    return KrakenProvider(session=sess, pair_map={"BTC/USD": "XBTUSD"}), sess

def test_drops_incomplete_candle_and_sets_schema():
    payload = {"error": [], "result": {"XXBTZUSD": [row(0), row(1), row(2)], "last": T0}}
    prov, sess = make_provider(payload)
    now = pd.Timestamp(T0 + 1800 + 450, unit="s", tz="UTC")
    df = prov.fetch_ohlcv("BTC/USD", 15, now=now)
    assert len(df) == 2
    assert df["timestamp"].is_monotonic_increasing
    assert str(df["timestamp"].dtype) == "datetime64[ns, UTC]"
    assert sess.calls[0] == {"pair": "XBTUSD", "interval": 15}

def test_api_error_raises():
    prov, _ = make_provider({"error": ["EQuery:Unknown asset pair"], "result": {}})
    with pytest.raises(KrakenError):
        prov.fetch_ohlcv("BTC/USD", 15)

def test_unknown_asset_and_interval_rejected():
    prov, _ = make_provider({"error": [], "result": {}})
    with pytest.raises(KrakenError):
        prov.fetch_ohlcv("DOGE/USD", 15)
    with pytest.raises(KrakenError):
        prov.fetch_ohlcv("BTC/USD", 7)
