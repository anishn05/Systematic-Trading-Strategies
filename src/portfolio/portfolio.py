import pandas as pd
from dataclasses import dataclass

@dataclass
class Position:
    symbol: str
    quantity: float = 0.0
    avg_price: float = 0.0
    realized_pnl: float = 0.0

    def update(self, fill_price, fill_qty, fill_side):
        signed_qty = fill_qty if fill_side == "BUY" else -fill_qty
        new_qty = self.quantity + signed_qty

        # --- Realised PnL on closed portion ---
        if self.quantity != 0 and (self.quantity * new_qty < 0 or abs(new_qty) < abs(self.quantity)):

            closed = min(abs(signed_qty), abs(self.quantity))
            direction = 1 if self.quantity > 0 else -1
            pnl = direction * closed * (fill_price - self.avg_price)
            self.realized_pnl += pnl

        # --- Update average price ---
        if new_qty != 0:
            if self.quantity == 0:
                self.avg_price = fill_price
            else:
                self.avg_price = (
                    self.avg_price * self.quantity + signed_qty * fill_price
                ) / new_qty
        else:
            self.avg_price = 0.0

        self.quantity = new_qty



class Portfolio:
    
    def __init__(self, initial_capital, leverage=2.0, maintenance_margin=0.5):

        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.positions: dict[str, Position] = {}
        self.trade_log: list[dict] = []

        self.last_prices = {}   # stores latest market prices

        self.leverage = leverage
        self.maintenance_margin = maintenance_margin


    # ============================
    # FILL → UPDATE PORTFOLIO STATE
    # ============================
    def apply_fill(self, fill, symbol):

        if symbol not in self.positions:
            self.positions[symbol] = Position(symbol)

        pos = self.positions[symbol]
        pos.update(fill.price, fill.quantity, fill.side)

        signed = fill.quantity if fill.side == "BUY" else -fill.quantity

        self.cash -= signed * fill.price
        self.cash -= fill.commission

        self.trade_log.append({
            "timestamp": fill.timestamp,
            "symbol": symbol,
            "side": fill.side,
            "qty": fill.quantity,
            "price": fill.price,
            "commission": fill.commission,
        })


    # ============================
    # MARK-TO-MARKET (INSTANT ONLY)
    # ============================
    def unrealised_pnl(self, price_map: dict) -> float:
        return sum(
            pos.quantity * (price_map[sym] - pos.avg_price)
            for sym, pos in self.positions.items()
        )

    def equity(self, price_map: dict) -> float:
        return self.cash + sum(
            pos.quantity * price_map[sym]
            for sym, pos in self.positions.items()
        )

    def realized_pnl(self):
        return sum(p.realized_pnl for p in self.positions.values())


    # ============================
    # EXPOSURE & RISK
    # ============================
    def net_exposure(self):
        return sum(pos.quantity * pos.avg_price for pos in self.positions.values())

    def available_buying_power(self):
        return self.initial_capital * self.leverage - abs(self.net_exposure())


    def update_prices(self, price_map):
        """
        price_map: dict {instrument: price}
        Called every tick/bar.
        """
        for inst, px in price_map.items():
            self.last_prices[inst] = px

    def get_last_price(self, instrument:str):
        """
        Return latest known price for instrument.
        Raises if no price exists.
        """
        try:
            return self.last_prices[instrument]
        except KeyError:
            raise ValueError(f"No price available for {instrument}")
        
    def current_equity(self):
        """
        Cash + MTM all positions using latest prices
        """
        total = self.cash
        for sym, pos in self.positions.items():
            if sym in self.last_prices:
                total += pos.quantity * self.last_prices[sym]
        return total
    
    def market_value(self, symbol):
        """
        Current notional exposure in dollars for one instrument
        """
        if symbol not in self.positions:
            return 0.0
        if symbol not in self.last_prices:
            return 0.0

        pos = self.positions[symbol]
        price = self.last_prices[symbol]

        return pos.quantity * price

    def market_value_after_order(self, order, price):
        """
        Project notional exposure after order is filled
        """
        current = self.market_value(order.instrument)

        sign = 1 if order.side == "BUY" else -1
        projected = current + price * order.quantity * sign

        return projected

    def leverage_after_order(self, order, price):
        """
        Calculate projected portfolio leverage
        """
        equity = self.current_equity()
        if equity == 0:
            return 999  # block trading

        # total exposure = sum absolute notional after order
        exposures = []

        # existing positions
        for sym in self.positions:
            exposures.append(abs(self.market_value(sym)))

        # add projected order exposure
        exposures.append(abs(
            self.market_value_after_order(order, price)
        ))

        total_exposure = sum(exposures)
        return total_exposure / equity



    # ============================
    # EXPORT
    # ============================
    def to_dataframe(self):
        return pd.DataFrame([
            {
                "symbol": p.symbol,
                "qty": p.quantity,
                "avg_price": p.avg_price,
                "realized_pnl": p.realized_pnl,
            }
            for p in self.positions.values()
        ])
