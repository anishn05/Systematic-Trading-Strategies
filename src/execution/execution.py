import pandas as pd
from dataclasses import dataclass


@dataclass
class Order:
    timestamp: pd.Timestamp
    side: str                  # "BUY" or "SELL"
    quantity: float
    price: float | None = None     # optional (for limit orders later)


@dataclass
class Fill:
    timestamp: pd.Timestamp
    side: str
    quantity: float
    price: float
    commission: float
    slippage: float


class ExecutionModel:
    """
    Professional execution model.
    Applies slippage, commissions, participation limits, and microstructure rules.
    """

    def __init__(self, settings: dict):
        self.settings = settings

    # =====================================================
    # Main fill generation method
    # =====================================================
    def execute(self, timestamp, orders, bar):
        fills = []
        for order in orders:
            filled_qty = self._apply_participation_limit(order, bar)
            if filled_qty <= 0:
                continue

            price = self._apply_slippage(bar)
            commission = self._commission(filled_qty, price)

            fills.append(
                Fill(
                    timestamp=timestamp,
                    side=order.side,
                    quantity=filled_qty,
                    price=price,
                    commission=commission,
                    slippage=abs(price - bar["Close"]) * filled_qty
                )
            )
        return fills

    # =====================================================
    # Execution components
    # =====================================================
    def _apply_participation_limit(self, order, bar):
        max_rate = self.settings["execution"]["max_participation_rate"]
        max_qty = bar["Volume"] * max_rate
        return min(order.quantity, max_qty)

    def _apply_slippage(self, bar):
        cfg = self.settings["fees"]["slippage"]

        if cfg["model"] == "fixed":
            return bar["Close"] * (1 + cfg["fixed_bps"] / 1e4)

        # Volume impact model
        impact = cfg.get("impact_coefficient", 0) * (bar["Volume"] ** -0.5)
        return bar["Close"] * (1 + impact)

    def _commission(self, qty, price):
        cfg = self.settings["fees"]["commission"]
        if cfg["type"] == "proportional":
            return max(qty * price * cfg["rate"], cfg["min_commission"])
        return cfg["min_commission"]
