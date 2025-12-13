import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional


class MarketDataLoader:
    """
    Responsible for:
    - Loading raw market data
    - Cleaning and validating
    - Returning standardized DataFrames
    """

    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)

    def load_csv(
        self,
        symbol: str,
        timeframe: str,
        parse_dates: bool = True
    ) -> pd.DataFrame:
        """
        Load CSV data for a given symbol and timeframe.
        Expected columns: [timestamp, open, high, low, close, volume]
        """

        file_path = self.data_dir / f"{symbol}_{timeframe}.csv"

        if not file_path.exists():
            raise FileNotFoundError(f"Data file not found: {file_path}")

        df = pd.read_csv(file_path)

        if parse_dates:
            df["timestamp"] = pd.to_datetime(df["timestamp"])

        return self._clean_data(df)

    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Standard cleaning:
        - Sort by time
        - Drop duplicates
        - Handle missing values
        """

        df = df.sort_values("timestamp")
        df = df.drop_duplicates(subset="timestamp")

        df = df.set_index("timestamp")

        # Forward-fill prices, zero-fill volume
        price_cols = ["open", "high", "low", "close"]
        df[price_cols] = df[price_cols].ffill()
        if "volume" in df.columns:
            df["volume"] = df["volume"].fillna(0)

        return df

    def resample(
        self,
        df: pd.DataFrame,
        rule: str
    ) -> pd.DataFrame:
        """
        Resample OHLCV data (e.g. 1min → 1H)
        """

        ohlc = {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum"
        }

        return df.resample(rule).apply(ohlc).dropna()
