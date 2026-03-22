from typing import Dict, List, Optional

from .broker import BrokerInterface
from .config import LiveBrokerConfig
from .models import AccountSnapshot, OrderRecord, OrderRequest, OrderStatus, PositionSnapshot


class LiveBrokerAdapter(BrokerInterface):
    """
    真实交易适配器骨架。

    当前只定义 live broker 需要实现的行为边界，不直接假设你已经有可用的交易 API。
    真正接券商/交易客户端时，只需要把这些方法补成具体实现。
    """

    def __init__(self, config: LiveBrokerConfig):
        self.config = config
        self._connected = False

    def connect(self) -> None:
        if not self.config.account_id:
            raise RuntimeError("live broker missing account_id")
        if not self.config.endpoint and not self.config.client_path:
            raise RuntimeError("live broker requires endpoint or client_path")
        self._connected = True

    def ensure_connected(self) -> None:
        if not self._connected:
            self.connect()

    def place_order(self, order: OrderRequest) -> OrderRecord:
        self.ensure_connected()
        raise NotImplementedError(
            "LiveBrokerAdapter.place_order is not implemented. "
            "You need to bind a real trading API here."
        )

    def cancel_order(self, order_id: str) -> bool:
        self.ensure_connected()
        raise NotImplementedError("LiveBrokerAdapter.cancel_order is not implemented.")

    def get_order(self, order_id: str) -> Optional[OrderRecord]:
        self.ensure_connected()
        raise NotImplementedError("LiveBrokerAdapter.get_order is not implemented.")

    def list_open_orders(self) -> List[OrderRecord]:
        self.ensure_connected()
        raise NotImplementedError("LiveBrokerAdapter.list_open_orders is not implemented.")

    def get_positions(self) -> Dict[str, PositionSnapshot]:
        self.ensure_connected()
        raise NotImplementedError("LiveBrokerAdapter.get_positions is not implemented.")

    def get_account(self) -> AccountSnapshot:
        self.ensure_connected()
        raise NotImplementedError("LiveBrokerAdapter.get_account is not implemented.")

