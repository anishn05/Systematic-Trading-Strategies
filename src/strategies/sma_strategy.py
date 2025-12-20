import pandas as pd
from src.strategies.base_strategy import Strategy
#Output signal dataframe has to have columns: ['Close', 'Volume', 'sma_short', 'sma_long', 'signal]


class SMAStrategy(Strategy):
    execution_mode = "vectorized"

    def __init__(self, short_window: int = 20, long_window: int = 50):
        super().__init__()

        if short_window >= long_window:
            raise ValueError("short_window must be less than long_window")

        self.short = short_window
        self.long = long_window

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generates directional trading signals:
        +1 = long
        -1 = short
        0  = flat

        Signals are shifted by 1 bar to avoid lookahead bias.
        """

        signals = pd.DataFrame(index=data.index)
        data.drop(['Open', 'High','Low'], axis = 1, inplace=True)
        signals = data
        signals["sma_short"] = (
            data["Close"]
            .rolling(window=self.short, min_periods=self.short)
            .mean()
        )

        signals["sma_long"] = (
            data["Close"]
            .rolling(window=self.long, min_periods=self.long)
            .mean()
        )

        signals["signal"] = 0
        signals.loc[signals["sma_short"] > signals["sma_long"], "signal"] = 1
        signals.loc[signals["sma_short"] < signals["sma_long"], "signal"] = -1

        # 🚨 Remove lookahead bias
        signals["signal"] = signals["signal"].shift(1)
        final_df = signals.dropna()    #dropping NaNs from SMA_short, SMA_long and signals columns
        return final_df