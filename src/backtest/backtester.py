import pandas as pd
import numpy as np
from dataclasses import dataclass

@dataclass
class Order:
    timestamp: pd.Timestamp
    side: str                 # 'BUY' or 'SELL'
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
    Supports vectorized and row-by-row (stateful) strategies.
    """

    def __init__(self, data: pd.DataFrame, strategy, settings: dict):
        self.data = data
        self.strategy = strategy
        self.settings = settings

        self.cash = settings["capital"]["initial_capital"]
        self.position = 0.0
        self.avg_price = 0.0

        self.equity_curve = []
        self.trades = []

    # =========================
    # MAIN LOOP
    # =========================
    def run(self):
        # Decide if strategy supports vectorized signals
        if self.strategy.execution_mode == "vectorized":
            # Vectorized: precompute signals
            signals_df = self.strategy.generate_signals(self.data)
            for timestamp, row in signals_df.iterrows():
                signal = row.get("signal", 0)
                orders = self._create_orders(timestamp, signal)
                fills = self._execute_orders(timestamp, orders, row)
                self._update_portfolio(fills, row["Close"])
                self._mark_to_market(row["Close"], timestamp)
        else:
            # Row-by-row: call generate_signal_row for each bar
            for timestamp, row in self.data.iterrows():
                signal = self.strategy.generate_signal_row(row)
                orders = self._create_orders(timestamp, signal)
                fills = self._execute_orders(timestamp, orders, row)
                self._update_portfolio(fills, row["Close"])
                self._mark_to_market(row["Close"], timestamp)

        return self._results()

    # =========================
    # ORDER CREATION
    # =========================
    def _create_orders(self, timestamp, signal):
        orders = []
        if signal == 0:
            return orders

        target_qty = signal * self._position_size()
        delta = target_qty - self.position

        if abs(delta) > 0:
            side = "BUY" if delta > 0 else "SELL"
            orders.append(Order(timestamp, side, abs(delta)))

        return orders

    # =========================
    # EXECUTION MODEL
    # =========================
    def _execute_orders(self, timestamp, orders, bar):
        fills = []
        for order in orders:
            max_qty = bar["Volume"] * self.settings["execution"]["max_participation_rate"]
            filled_qty = min(order.quantity, max_qty)

            fill_price = self._apply_slippage(bar)
            commission = self._commission(filled_qty, fill_price)

            fills.append(
                Fill(
                    timestamp=timestamp,
                    side=order.side,
                    quantity=filled_qty,
                    price=fill_price,
                    commission=commission,
                    slippage=abs(fill_price - bar["Close"]) * filled_qty
                )
            )
        return fills

    # =========================
    # PORTFOLIO UPDATE
    # =========================
    def _update_portfolio(self, fills, price):
        for fill in fills:
            signed_qty = fill.quantity if fill.side == "BUY" else -fill.quantity

            new_position = self.position + signed_qty

            if new_position != 0:
                self.avg_price = (
                    (self.position * self.avg_price + signed_qty * fill.price) / new_position
                )
            else:
                self.avg_price = 0.0

            self.position = new_position
            self.cash -= signed_qty * fill.price
            self.cash -= fill.commission

            self.trades.append(fill)

    # =========================
    # COST MODELS
    # =========================
    def _apply_slippage(self, bar):
        slippage_cfg = self.settings["fees"]["slippage"]

        if slippage_cfg["model"] == "fixed":
            return bar["Close"] * (1 + slippage_cfg["fixed_bps"] / 1e4)

        # Volume-impact model
        impact = slippage_cfg.get("impact_coefficient", 0) * (bar["Volume"] ** -0.5)
        return bar["Close"] * (1 + impact)

    def _commission(self, qty, price):
        cfg = self.settings["fees"]["commission"]
        if cfg["type"] == "proportional":
            return max(qty * price * cfg["rate"], cfg["min_commission"])
        return cfg["min_commission"]

    # =========================
    # MARK TO MARKET
    # =========================
    def _mark_to_market(self, price, timestamp):
        equity = self.cash + self.position * price
        self.equity_curve.append({"timestamp": timestamp, "equity": equity})

    # =========================
    # POSITION SIZING
    # =========================
    def _position_size(self):
        max_pct = self.settings["capital"]["max_position_pct"]
        return (self.cash * max_pct) / self.data.iloc[0]["Close"]

    # =========================
    # RESULTS
    # =========================
    def _results(self):
        equity_df = pd.DataFrame(self.equity_curve).set_index("timestamp")
        final_equity = equity_df["equity"].iloc[-1]
        returns = equity_df["equity"].pct_change().dropna()
        initial_capital = self.settings["capital"]["initial_capital"]
        total_return = final_equity / initial_capital - 1.0
        running_max = equity_df["equity"].cummax()
        drawdown = equity_df["equity"] / running_max - 1.0
        max_drawdown = drawdown.min()
        trades_df = pd.DataFrame([f.__dict__ for f in self.trades])

        return {
           "equity_curve": equity_df,
            "returns": returns,
            "trades": trades_df,
            "final_equity": final_equity,
            "total_return": total_return,
            "num_trades": len(self.trades),
            "max_drawdown": max_drawdown,
        }
