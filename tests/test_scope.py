from ai_trader.config import load_config
from ai_trader.paths import data_dir
import pytest

def test_scope_is_locked():
    data = load_config("data")
    risk = load_config("risk")
    assert data["provider"] == "kraken"
    assert data["timeframes_minutes"] == [15, 60]
    assert set(data["assets"]) == {"BTC/USD", "ETH/USD"}
    assert risk["initial_capital"] == 1000
    assert risk["leverage"] == 1.0
    assert risk["long_only"] is True

def test_unknown_stage_rejected():
    with pytest.raises(ValueError):
        data_dir("random_folder")
