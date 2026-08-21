"""Historical Intraday External Archive Repair: config responsibilities."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPAIR_ID = "historical_intraday_5m_external_archive_repair_v2"


BASELINE_ID = "pretraining_data_repair_baseline_v1"


START_DATE = "2010-01-01"


END_DATE = "2025-12-31"


DEFAULT_ARCHIVE = Path(r"H:\BaiduNetdiskDownload\量化数据\5分钟(2000-2025).zip")


DEFAULT_WORKERS = 4


PRICE_RELATIVE_TOLERANCE = 0.02


VOLUME_RELATIVE_TOLERANCE = 0.05


AMOUNT_RELATIVE_TOLERANCE = 0.05


SOURCE_NAME = "external_quant_archive_daily_validated_v2"


EXPECTED_CANDIDATE_DAYS = 130_060


EXPECTED_CANDIDATE_SYMBOLS = 174


EXPECTED_ACCEPTED_DAYS = 129_116


EXPECTED_FORMAL_QUALITY_ACCEPTED_DAYS = 35_450


EXPECTED_FORMAL_QUALITY_REMAINING_DAYS = 152


EXPECTED_RECENT_QUALITY_REMAINING_DAYS = 3


EXPECTED_FORMAL_QUALITY_POOL_ROWS = 4_476_997


_MEMBER_PATTERN = re.compile(r"^(?P<exchange>sh|sz)(?P<code>\d{6})\.csv$", re.IGNORECASE)


_NUMERIC_COLUMNS = ("open", "high", "low", "close", "volume", "amount")


_PRICE_COLUMNS = ("open", "high", "low", "close")


_DECISION_COLUMNS = (
    "symbol",
    "trade_date",
    "quality_liquidity_keep",
    "archive_member",
    "decision",
    "rejection_reason",
    "source_bar_count",
    "normalized_bar_count",
    "price_relative_error",
    "volume_relative_error",
    "amount_relative_error",
    "accepted_row_count",
    "source",
)


class ExternalArchiveRepairError(RuntimeError):
    pass


@dataclass(frozen=True)
class ArchiveMember:
    symbol: str
    member_name: str
    file_size: int
    compressed_size: int
    crc32: str


@dataclass(frozen=True)
class SymbolResult:
    symbol: str
    requested_day_count: int
    accepted_day_count: int
    rejected_day_count: int
    accepted_path: str
    decision_path: str
    receipt_path: str
    accepted_sha256: str
    decision_sha256: str
    reused: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "requested_day_count": self.requested_day_count,
            "accepted_day_count": self.accepted_day_count,
            "rejected_day_count": self.rejected_day_count,
            "accepted_path": self.accepted_path,
            "decision_path": self.decision_path,
            "receipt_path": self.receipt_path,
            "accepted_sha256": self.accepted_sha256,
            "decision_sha256": self.decision_sha256,
            "reused": self.reused,
        }
