from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class SignalType(str, Enum):
    BUY_ENTRY = "buy_entry"
    BUY_ADD = "buy_add"
    SELL_REDUCE = "sell_reduce"
    SELL_EXIT = "sell_exit"


class OrderSide(str, Enum):
    BUY = "buy"
    SELL = "sell"


class OrderStatus(str, Enum):
    NEW = "new"
    SUBMITTED = "submitted"
    PARTIAL_FILLED = "partial_filled"
    FILLED = "filled"
    CANCELED = "canceled"
    REJECTED = "rejected"


class TimeInForce(str, Enum):
    DAY = "day"
    IOC = "ioc"
    FOK = "fok"


@dataclass
class SignalEvent:
    stock: str
    signal_type: SignalType
    signal_price: float
    signal_time: str
    reason: str
    suggested_qty: int = 100
    score: Optional[float] = None


@dataclass
class OrderRequest:
    stock: str
    side: OrderSide
    quantity: int
    limit_price: float
    tif: TimeInForce = TimeInForce.DAY
    reason: str = ""
    signal_type: Optional[SignalType] = None


@dataclass
class OrderRecord:
    order_id: str
    request: OrderRequest
    status: OrderStatus = OrderStatus.NEW
    filled_quantity: int = 0
    avg_fill_price: float = 0.0
    reject_reason: str = ""


@dataclass
class PositionSnapshot:
    stock: str
    quantity: int
    available_quantity: int
    cost_price: float
    last_price: float = 0.0


@dataclass
class AccountSnapshot:
    total_asset: float
    available_cash: float
    market_value: float = 0.0


@dataclass
class RiskCheckResult:
    approved: bool
    reason: str = ""
    adjusted_quantity: Optional[int] = None
    tags: list[str] = field(default_factory=list)

