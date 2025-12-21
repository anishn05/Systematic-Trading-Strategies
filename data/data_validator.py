import pandas as pd
from typing import Dict, List, Tuple, Optional
from config.config_loader import Config

class DataValidationError(Exception):
    """Raised when dataset quality is unacceptable."""
    pass


class DataValidator:
    """
    Provide structural validation for OHLCV datasets.

    Works with:
      - single DataFrame
      - dict[str, DataFrame]
    """
    def __init__(self):
        # Load global config
        settings = Config.load()
        self.required_cols = settings["data_validation"]["required_columns"]
        self.volume_cols = settings["data_validation"]["volume_columns"]
        self.max_price_jump = settings["data_validation"]["max_price_jump"]
        self.strict = settings["data_validation"]["strict"]
        
    # ========================================
    # PUBLIC ENTRY API
    # ========================================

    def validate_single(self, df: pd.DataFrame, symbol: str = "") -> None:
        """
        Raises an exception if something is wrong.
        """

        self._check_index(df, symbol)
        self._check_structure(df, symbol)
        self._check_missing_values(df, symbol)
        self._check_price_relationship(df, symbol)

    def validate_multiple(
        self,
        universe: Dict[str, pd.DataFrame]
    ) -> None:
        """
        Validate each symbol & cross-validate universe.
        """

        # --- run individual validation
        for sym, df in universe.items():
            self.validate_single(df, sym)

        # --- cross checks
        self._check_calendar_alignment(universe)
        self._check_common_period(universe)

    # ========================================
    # SINGLE SYMBOL TESTS
    # ========================================

    def _check_index(self, df: pd.DataFrame, symbol: str) -> None:

        if not isinstance(df.index, pd.DatetimeIndex):
            raise DataValidationError(
                f"[{symbol}] Index must be datetime."
            )

        if df.index.has_duplicates:
            raise DataValidationError(
                f"[{symbol}] Duplicate timestamps found."
            )

        if not df.index.is_monotonic_increasing:
            raise DataValidationError(
                f"[{symbol}] Index must be sorted ascending."
            )

    def _check_structure(self, df: pd.DataFrame, symbol: str):

        for col in self.required_cols:
            if col not in df.columns:
                raise DataValidationError(
                    f"[{symbol}] Missing required column: {col}"
                )

        # optional check – volume not mandatory
        # but warn if exists and bad
        if "Volume" in df.columns:
            if df["Volume"].isna().any():
                raise DataValidationError(
                    f"[{symbol}] Volume contains NaNs."
                )

    def _check_missing_values(self, df: pd.DataFrame, symbol: str):

        if df[self.required_cols].isna().any().any():
            raise DataValidationError(
                f"[{symbol}] NaN found in OHLC data."
            )

    def _check_price_relationship(self, df: pd.DataFrame, symbol: str):

        bad = df[(df["Low"] > df["High"]) |
                 (df["Open"] > df["High"]) |
                 (df["Close"] > df["High"])]

        if not bad.empty:
            raise DataValidationError(
                f"[{symbol}] Invalid OHLC relationships detected."
            )

    # ========================================
    # MULTI SYMBOL CROSS-VALIDATION
    # ========================================

    def _check_calendar_alignment(
        self,
        universe: Dict[str, pd.DataFrame]
    ) -> None:

        # pick first asset as reference
        ref_symbol = next(iter(universe))
        ref_index = universe[ref_symbol].index

        for sym, df in universe.items():
            if not df.index.equals(ref_index):
                raise DataValidationError(
                    f"Calendar mismatch between {ref_symbol} and {sym}."
                )

    def _check_common_period(
        self,
        universe: Dict[str, pd.DataFrame]
    ) -> None:

        starts = [(sym, df.index.min()) for sym, df in universe.items()]
        ends   = [(sym, df.index.max()) for sym, df in universe.items()]

        latest_start = max(t[1] for t in starts)
        earliest_end = min(t[1] for t in ends)

        if latest_start >= earliest_end:
            raise DataValidationError(
                "No overlapping date window across assets."
            )
