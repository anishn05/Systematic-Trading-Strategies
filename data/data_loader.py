import pandas as pd
from pathlib import Path
from typing import Optional, Dict


class DataLoader:
    """
    Responsible for:
    - Loading raw market data
    - Cleaning format
    - Returning single or multi-asset OHLCV DataFrames
    """

    def __init__(self, data_dir: str, frequency):
        self.data_dir = Path(data_dir)
        self.frequency = frequency

    # ==========================================================
    # SINGLE SYMBOL LOADING  (existing behaviour preserved)
    # ==========================================================
    def load_csv(
        self,
        symbol: str,
        timeframe: str,
        parse_dates: bool = True
    ) -> pd.DataFrame:

        df = self._load_single(symbol, timeframe, parse_dates)
        return self._clean_data(df)

    def _load_single(self, symbol: str, timeframe: str, parse_dates=True):
        file_path = self.data_dir / f"{symbol}_{timeframe}.csv"

        if not file_path.exists():
            raise FileNotFoundError(f"Data file not found: {file_path}")

        df = pd.read_csv(file_path)

        if parse_dates:
            df["Date"] = pd.to_datetime(df["Date"])

        return df

    # ==========================================================
    # MULTI-ASSET LOADING
    # ==========================================================
    def load_multiple(
        self,
        symbols: list[str],
        timeframe: str,
        parse_dates: bool = True
    ) -> Dict[str, pd.DataFrame]:
        """
        Returns:
            { "AAPL": df, "MSFT": df }

        — NO validation yet
        — NO alignment yet
        """

        universe = {}

        for sym in symbols:
            df = self._load_single(sym, timeframe, parse_dates)
            universe[sym] = self._clean_data(df)

        return universe

    # ==========================================================
    # COMMON CLEANING LOGIC
    # ==========================================================
    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.sort_values("Date")
        df = df.drop_duplicates(subset="Date")
        df = df.set_index("Date")

        price_cols = ["Open","High","Low","Close"]

        df[price_cols] = df[price_cols].ffill()

        if "Volume" in df.columns:
            df["Volume"] = df["Volume"].fillna(0)

        return df

    # ==========================================================
    # RESAMPLING (still works for single symbol output)
    # ==========================================================
    def resample(self, df: pd.DataFrame, rule: str) -> pd.DataFrame:

        ohlc = {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum"
        }

        return df.resample(rule).apply(ohlc).dropna()
