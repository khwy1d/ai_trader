"""Provider interface. The rest of the project depends on THIS, not on Kraken."""
from abc import ABC, abstractmethod
import pandas as pd

class DataProvider(ABC):
    name: str = "base"

    @abstractmethod
    def fetch_ohlcv(self, asset: str, timeframe_minutes: int,
                    since: pd.Timestamp | None = None,
                    now: pd.Timestamp | None = None) -> pd.DataFrame:
        """Return CLOSED candles only, columns = OHLCV_COLUMNS, UTC timestamps.
        timestamp = candle OPEN time."""
