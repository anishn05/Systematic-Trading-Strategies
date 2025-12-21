import pandas as pd
import numpy as np
from dataclasses import dataclass
from src.execution.execution import ExecutionModel, Order, Fill
from src.portfolio.portfolio import Portfolio
from src.risk.risk_manager import RiskManager, RiskManagerConfig


@dataclass
class Order:
    timestamp: pd.Timestamp
    instrument: str
    side: str
    quantity: float
    price: float | None = None


@dataclass
class Fill:
    timestamp: pd.Timestamp
    side: str
    quantity: float
    price: float
    commission: float
    slippage: float


class Backtester:
    """
    Professional-grade, event-driven, strategy-agnostic backtester.
    Supports vectorized and iterative strategies.
    """

    def __init__(self, data: pd.DataFrame, strategy, settings: dict):

        self.data = data
        self.strategy = strategy
        self.settings = settings
        self.symbol = settings["symbol"]

        risk_conf = RiskManagerConfig(
        max_leverage=settings["risk"]["max_leverage"],
        max_position_notional=settings["risk"]["max_position_notional"],
        #max_daily_loss=settings["risk"]["max_daily_loss"],
        #max_drawdown=settings["risk"]["max_drawdown"],
        #min_cash_buffer=settings["risk"]["min_cash_buffer"],
        allow_shorting=settings["risk"]["allow_shorting"],
        )
        self.risk_manager = RiskManager(risk_conf)

        self.portfolio = Portfolio(
            initial_capital=settings["capital"]["initial_capital"],
            leverage=settings["capital"]["leverage"]
        )

        self.execution = ExecutionModel(settings)

        self.equity_curve = []
        self.trades = []


    # =========================
    # MAIN LOOP
    # =========================
    def run(self):

        if self.strategy.execution_mode == "vectorized":
            signals_df = self.strategy.generate_signals(self.data)
            for timestamp, row in signals_df.iterrows():
                signal = row.get("signal", 0)
                orders = self._create_orders(timestamp, signal)
                orders = self.risk_manager.filter_orders(orders, self.portfolio)
                fills = self.execution.execute(timestamp, orders, row)
                for fill in fills:
                    self.portfolio.apply_fill(fill, self.symbol)
                    self.trades.append(fill)
                self._mark_to_market(timestamp, row["Close"])
                #self.risk_manager.intraday_kill_switch(self.portfolio)

        else:
            for timestamp, row in self.data.iterrows():
                signal = self.strategy.generate_signal_row(row)
                orders = self._create_orders(timestamp, signal)
                fills = self.execution.execute(timestamp, orders, row)
                for fill in fills:
                    self.portfolio.apply_fill(fill, self.symbol)
                    self.trades.append(fill)
                self._mark_to_market(timestamp, row["Close"])
        return self._results()


    # =========================
    # ORDER CREATION
    # =========================
    def _create_orders(self, timestamp, signal):

        if signal == 0:
            return []
        target_qty = signal * self._position_size()
        # get portfolio position size
        current_qty = self.portfolio.positions.get(self.symbol, None)
        current_qty = current_qty.quantity if current_qty else 0.0
        delta = target_qty - current_qty

        if abs(delta) < 1e-12:
            return []

        side = "BUY" if delta > 0 else "SELL"
        return [Order(timestamp, self.symbol, side, abs(delta))]


    # =========================
    # MARK TO MARKET
    # =========================
    def _mark_to_market(self, timestamp, price):

        price_map = {self.symbol: price}
        self.portfolio.update_prices(price_map)
        equity = self.portfolio.equity(price_map)
        self.equity_curve.append({
            "timestamp": timestamp,
            "equity": equity
        })


    # =========================
    # POSITION SIZING
    # =========================
    def _position_size(self):

        capital = self.portfolio.cash
        max_pct = self.settings["capital"]["max_position_pct"]
        price = self.data.iloc[0]["Close"]
        position_value = capital * max_pct
        qty = position_value / price

        return qty


    # =========================
    # RESULTS
    # =========================
    def _results(self):

        equity_df = pd.DataFrame(self.equity_curve).set_index("timestamp")
        initial_cap = self.settings["capital"]["initial_capital"]
        final_equity = equity_df["equity"].iloc[-1]
        returns = equity_df["equity"].pct_change().dropna()
        running_max = equity_df["equity"].cummax()
        dd = equity_df["equity"] / running_max - 1.0
        max_dd = dd.min()
        trades_df = pd.DataFrame([f.__dict__ for f in self.trades])
        return {
            "equity_curve": equity_df,
            "returns": returns,
            "trades": trades_df,
            "final_equity": final_equity,
            "total_return": final_equity / initial_cap - 1,
            "num_trades": len(trades_df),
            "max_drawdown": max_dd,
        }
