from abc import ABC, abstractmethod
import pandas as pd

class Strategy(ABC):
    """
    Base class for all strategies.
    """
    execution_mode = "vectorized"  # or "event

    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        For vectorized strategies: returns a DataFrame of signals
        for all timestamps.
        """
        pass

    def generate_signal_row(self, row: pd.Series) -> int:
        """
        Optional: For row-by-row strategies (stateful/live)
        Override if strategy needs incremental updates.
        Returns a signal for a single row.
        """
        return 0
