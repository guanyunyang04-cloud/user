from dataclasses import dataclass
from typing import Dict

from .models import AccountSnapshot, OrderRequest, OrderSide, PositionSnapshot, RiskCheckResult


@dataclass
class RiskConfig:
    max_position_ratio: float = 0.2
    max_single_order_ratio: float = 0.1
    min_lot_size: int = 100
    max_daily_trade_per_stock: int = 6


class RiskManager:
    def __init__(self, config: RiskConfig | None = None):
        self.config = config or RiskConfig()

    def check_order(
        self,
        order: OrderRequest,
        account: AccountSnapshot,
        positions: Dict[str, PositionSnapshot],
        daily_trade_count: int = 0,
    ) -> RiskCheckResult:
        if order.quantity <= 0:
            return RiskCheckResult(False, "quantity <= 0")

        if daily_trade_count >= self.config.max_daily_trade_per_stock:
            return RiskCheckResult(False, "daily trade limit reached", tags=["trade_limit"])

        quantity = max((order.quantity // self.config.min_lot_size) * self.config.min_lot_size, 0)
        if quantity == 0:
            return RiskCheckResult(False, "quantity below minimum lot", tags=["lot"])

        order_value = quantity * order.limit_price
        max_order_value = account.total_asset * self.config.max_single_order_ratio
        if order_value > max_order_value:
            quantity = int(max_order_value // max(order.limit_price, 1e-8))
            quantity = max((quantity // self.config.min_lot_size) * self.config.min_lot_size, 0)

        if quantity == 0:
            return RiskCheckResult(False, "single order exceeds risk budget", tags=["budget"])

        if order.side == OrderSide.BUY:
            if order_value > account.available_cash:
                quantity = int(account.available_cash // max(order.limit_price, 1e-8))
                quantity = max((quantity // self.config.min_lot_size) * self.config.min_lot_size, 0)
            if quantity == 0:
                return RiskCheckResult(False, "insufficient cash", tags=["cash"])

            position = positions.get(order.stock)
            current_value = 0.0 if position is None else position.quantity * order.limit_price
            if current_value + quantity * order.limit_price > account.total_asset * self.config.max_position_ratio:
                max_position_value = account.total_asset * self.config.max_position_ratio - current_value
                quantity = int(max_position_value // max(order.limit_price, 1e-8))
                quantity = max((quantity // self.config.min_lot_size) * self.config.min_lot_size, 0)
            if quantity == 0:
                return RiskCheckResult(False, "position cap reached", tags=["position_cap"])
        else:
            position = positions.get(order.stock)
            if position is None or position.available_quantity <= 0:
                return RiskCheckResult(False, "no sellable position", tags=["position"])
            quantity = min(quantity, position.available_quantity)
            quantity = max((quantity // self.config.min_lot_size) * self.config.min_lot_size, 0)
            if quantity == 0:
                return RiskCheckResult(False, "insufficient available position", tags=["position"])

        return RiskCheckResult(True, adjusted_quantity=quantity)

