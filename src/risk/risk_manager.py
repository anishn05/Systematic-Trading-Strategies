from dataclasses import dataclass

class RiskViolation(Exception):
    pass


@dataclass
class RiskManagerConfig:
    max_leverage: float
    max_position_notional: float  # 10% of equity per asset
    allow_shorting: bool


class RiskManager:

    def __init__(self, config: RiskManagerConfig):
        self.config = config

    def filter_orders(self, orders, portfolio):
        """Return only orders that satisfy portfolio risk constraints"""
        safe = []
        for order in orders:
            if self._order_allowed(order, portfolio):
                safe.append(order)
        return safe


    def _order_allowed(self, order, portfolio):

        # --- no shorting? ---
        if not self.config.allow_shorting and order.side == "SELL":
            return False

        symbol = order.instrument

        # --- price must exist ---
        try:
            price = portfolio.get_last_price(symbol)
        except:
            return False

        # --- current equity ---
        equity = portfolio.current_equity()

        # --- calculate market value after trade ---
        new_mv = portfolio.market_value_after_order(order, price)

        # --- per-instrument notional limit ---
        if abs(new_mv) > self.config.max_position_notional * equity:
            return False

        # --- leverage check ---
        if portfolio.leverage_after_order(order, price) > self.config.max_leverage:
            return False

        return True
