from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import json
import os


@dataclass
class LiveBrokerConfig:
    broker_name: str = "tdx_live"
    account_id: str = ""
    endpoint: str = ""
    client_path: str = ""
    dry_run: bool = True
    allow_market_orders: bool = False
    password_env: str = "T0_BROKER_PASSWORD"

    @property
    def password(self) -> str:
        return os.getenv(self.password_env, "")

    @classmethod
    def from_json(cls, path: str | Path) -> "LiveBrokerConfig":
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        return cls(**raw)


def load_live_broker_config(path: Optional[str]) -> LiveBrokerConfig:
    if not path:
        return LiveBrokerConfig()
    return LiveBrokerConfig.from_json(path)

