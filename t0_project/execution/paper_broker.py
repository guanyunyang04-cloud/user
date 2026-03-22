from typing import Dict, List, Optional
import itertools

from .broker import BrokerInterface
from .models import (
    AccountSnapshot,
    OrderRecord,
    OrderRequest,
    OrderSide,
    OrderStatus,
    PositionSnapshot,
)


class PaperBroker(BrokerInterface):
    def __init__(self, initial_cash: float = 1_000_000.0):
        self._order_counter = itertools.count(1)
        self._account = AccountSnapshot(total_asset=initial_cash, available_cash=initial_cash, market_value=0.0)
        self._orders: Dict[str, OrderRecord] = {}
        self._positions: Dict[str, PositionSnapshot] = {}

    def seed_position(self, stock: str, quantity: int, cost_price: float) -> None:
        quantity = max(int(quantity), 0)
        if quantity <= 0:
            return
        self._positions[stock] = PositionSnapshot(
            stock=stock,
            quantity=quantity,
            available_quantity=quantity,
            cost_price=cost_price,
            last_price=cost_price,
        )
        self._account.available_cash = max(0.0, self._account.available_cash - quantity * cost_price)
        self._revalue_account()

    def place_order(self, order: OrderRequest) -> OrderRecord:
        order_id = f"PAPER-{next(self._order_counter):06d}"
        record = OrderRecord(
            order_id=order_id,
            request=order,
            status=OrderStatus.FILLED,
            filled_quantity=order.quantity,
            avg_fill_price=order.limit_price,
        )
        self._orders[order_id] = record
        self._apply_fill(record)
        return record

    def _apply_fill(self, record: OrderRecord) -> None:
        stock = record.request.stock
        qty = record.filled_quantity
        px = record.avg_fill_price
        amount = qty * px

        if record.request.side == OrderSide.BUY:
            position = self._positions.get(stock)
            if position is None:
                self._positions[stock] = PositionSnapshot(
                    stock=stock,
                    quantity=qty,
                    available_quantity=qty,
                    cost_price=px,
                    last_price=px,
                )
            else:
                new_qty = position.quantity + qty
                new_cost = ((position.quantity * position.cost_price) + amount) / max(new_qty, 1)
                position.quantity = new_qty
                position.available_quantity += qty
                position.cost_price = new_cost
                position.last_price = px
            self._account.available_cash -= amount
        else:
            position = self._positions.get(stock)
            if position is None:
                return
            position.quantity -= qty
            position.available_quantity -= qty
            position.last_price = px
            self._account.available_cash += amount
            if position.quantity <= 0:
                del self._positions[stock]

        self._revalue_account()

    def _revalue_account(self) -> None:
        market_value = sum(pos.quantity * pos.last_price for pos in self._positions.values())
        self._account.market_value = market_value
        self._account.total_asset = self._account.available_cash + market_value

    def cancel_order(self, order_id: str) -> bool:
        record = self._orders.get(order_id)
        if record is None or record.status == OrderStatus.FILLED:
            return False
        record.status = OrderStatus.CANCELED
        return True

    def get_order(self, order_id: str) -> Optional[OrderRecord]:
        return self._orders.get(order_id)

    def list_open_orders(self) -> List[OrderRecord]:
        return [order for order in self._orders.values() if order.status in {OrderStatus.NEW, OrderStatus.SUBMITTED, OrderStatus.PARTIAL_FILLED}]

    def get_positions(self) -> Dict[str, PositionSnapshot]:
        return {stock: PositionSnapshot(**vars(pos)) for stock, pos in self._positions.items()}

    def get_account(self) -> AccountSnapshot:
        return AccountSnapshot(**vars(self._account))
