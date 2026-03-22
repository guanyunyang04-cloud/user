from abc import ABC, abstractmethod
from typing import Dict, List, Optional

from .models import AccountSnapshot, OrderRecord, OrderRequest, PositionSnapshot


class BrokerInterface(ABC):
    @abstractmethod
    def place_order(self, order: OrderRequest) -> OrderRecord:
        raise NotImplementedError

    @abstractmethod
    def cancel_order(self, order_id: str) -> bool:
        raise NotImplementedError

    @abstractmethod
    def get_order(self, order_id: str) -> Optional[OrderRecord]:
        raise NotImplementedError

    @abstractmethod
    def list_open_orders(self) -> List[OrderRecord]:
        raise NotImplementedError

    @abstractmethod
    def get_positions(self) -> Dict[str, PositionSnapshot]:
        raise NotImplementedError

    @abstractmethod
    def get_account(self) -> AccountSnapshot:
        raise NotImplementedError

