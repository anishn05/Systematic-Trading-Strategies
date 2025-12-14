import numpy as np
import pandas as pd

class PerformanceAnalyzer:
    """
    Computes performance metrics from backtest outputs.
    """
    def __init__(
        self,
        equity_curve: pd.DataFrame,
        returns: pd.Series,
        trades: pd.DataFrame,
        periods_per_year: int = 252,
        risk_free_rate: float = 0.0,
    ):
        self.equity = equity_curve["equity"]
        self.returns = returns
        self.trades = trades
        self.ppy = periods_per_year
        self.rf = risk_free_rate

    # =========================
    # RETURN METRICS
    # =========================
    def total_return(self):
        return self.equity.iloc[-1] / self.equity.iloc[0] - 1

    def annualized_return(self):
        return (1 + self.total_return()) ** (
            self.ppy / len(self.returns)
        ) - 1

    def annualized_volatility(self):
        return self.returns.std() * np.sqrt(self.ppy)

    # =========================
    # RISK-ADJUSTED METRICS
    # =========================
    def sharpe_ratio(self):
        excess = self.returns - self.rf / self.ppy
        return np.sqrt(self.ppy) * excess.mean() / excess.std()

    def sortino_ratio(self):
        downside = self.returns[self.returns < 0]
        downside_std = downside.std()
        return np.sqrt(self.ppy) * self.returns.mean() / downside_std

    def calmar_ratio(self):
        return self.annualized_return() / abs(self.max_drawdown())

    # =========================
    # DRAWDOWN METRICS
    # =========================
    def drawdown_series(self):
        running_max = self.equity.cummax()
        return self.equity / running_max - 1

    def max_drawdown(self):
        return self.drawdown_series().min()

    def avg_drawdown(self):
        dd = self.drawdown_series()
        return dd[dd < 0].mean()

    # =========================
    # TRADE QUALITY
    # =========================
    def hit_rate(self):
        if self.trades.empty:
            return np.nan
        pnl = self._trade_pnl()
        return (pnl > 0).mean()

    def profit_factor(self):
        pnl = self._trade_pnl()
        wins = pnl[pnl > 0].sum()
        losses = abs(pnl[pnl < 0].sum())
        return wins / losses if losses != 0 else np.nan

    def expectancy(self):
        pnl = self._trade_pnl()
        return pnl.mean()

    def turnover(self):
        if self.trades.empty:
            return 0.0
        traded_value = (self.trades["quantity"] * self.trades["price"]).sum()
        avg_equity = self.equity.mean()
        return traded_value / avg_equity

    # =========================
    # EXPOSURE METRICS
    # =========================
    def time_in_market(self):
        return (self.returns != 0).mean()

    # =========================
    # DISTRIBUTION METRICS
    # =========================
    def skew(self):
        return self.returns.skew()

    def kurtosis(self):
        return self.returns.kurtosis()

    # =========================
    # HELPERS
    # =========================
    def _trade_pnl(self):
        """
        Approximate per-trade PnL using fills.
        """
        signed_qty = np.where(
            self.trades["side"] == "BUY",
            self.trades["quantity"],
            -self.trades["quantity"],
        )
        return -signed_qty * self.trades["price"] - self.trades["commission"]

    # =========================
    # SUMMARY
    # =========================
    def summary(self):
        return {
            "Total Return": self.total_return(),
            "Annualized Return": self.annualized_return(),
            "Annualized Volatility": self.annualized_volatility(),
            "Sharpe Ratio": self.sharpe_ratio(),
            "Sortino Ratio": self.sortino_ratio(),
            "Calmar Ratio": self.calmar_ratio(),
            "Max Drawdown": self.max_drawdown(),
            "Average Drawdown": self.avg_drawdown(),
            "Hit Rate": self.hit_rate(),
            "Profit Factor": self.profit_factor(),
            "Expectancy": self.expectancy(),
            "Turnover": self.turnover(),
            "Time in Market": self.time_in_market(),
            "Skew": self.skew(),
            "Kurtosis": self.kurtosis(),
        }
