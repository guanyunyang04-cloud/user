from dataclasses import dataclass, field
from typing import Dict, Optional

from .broker import BrokerInterface
from .models import OrderRecord, OrderRequest, OrderSide, SignalEvent, SignalType
from .risk import RiskManager


@dataclass
class ExecutionState:
    daily_trade_count: Dict[str, int] = field(default_factory=dict)
    latest_orders: Dict[str, OrderRecord] = field(default_factory=dict)


class OrderManager:
    def __init__(self, broker: BrokerInterface, risk_manager: RiskManager | None = None):
        self.broker = broker
        self.risk_manager = risk_manager or RiskManager()
        self.state = ExecutionState()

    def _signal_to_side(self, signal: SignalEvent) -> OrderSide:
        if signal.signal_type in {SignalType.BUY_ENTRY, SignalType.BUY_ADD}:
            return OrderSide.BUY
        return OrderSide.SELL

    def build_order_request(self, signal: SignalEvent) -> OrderRequest:
        return OrderRequest(
            stock=signal.stock,
            side=self._signal_to_side(signal),
            quantity=signal.suggested_qty,
            limit_price=signal.signal_price,
            reason=signal.reason,
            signal_type=signal.signal_type,
        )

    def process_signal(self, signal: SignalEvent) -> Optional[OrderRecord]:
        request = self.build_order_request(signal)
        account = self.broker.get_account()
        positions = self.broker.get_positions()
        daily_count = self.state.daily_trade_count.get(signal.stock, 0)
        risk_result = self.risk_manager.check_order(request, account, positions, daily_count)
        if not risk_result.approved:
            return None

        request.quantity = int(risk_result.adjusted_quantity or request.quantity)
        record = self.broker.place_order(request)
        self.state.latest_orders[signal.stock] = record
        self.state.daily_trade_count[signal.stock] = daily_count + 1
        return record

