from .config import load_live_broker_config
from .live_broker import LiveBrokerAdapter
from .order_manager import OrderManager
from .paper_broker import PaperBroker
from .risk import RiskManager


def build_execution_manager(mode: str = "paper", broker_config_path: str | None = None) -> OrderManager:
    mode = (mode or "paper").lower()
    if mode == "paper":
        return OrderManager(PaperBroker(initial_cash=1_000_000.0), RiskManager())
    if mode == "live":
        config = load_live_broker_config(broker_config_path)
        return OrderManager(LiveBrokerAdapter(config), RiskManager())
    raise ValueError(f"unsupported execution mode: {mode}")

