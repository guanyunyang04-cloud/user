"""Read-only adapter for Baostock point-in-time market data."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import pandas as pd


class BaostockSourceError(RuntimeError):
    """Raised when Baostock cannot be queried safely."""


@dataclass(frozen=True)
class BaostockSourceConfig:
    timeout_seconds: int = 30
    retry_count: int = 3
    retry_base_delay_seconds: float = 0.5
    request_interval_seconds: float = 0.0


TRADE_DATES_COLUMNS = ["calendar_date", "is_trading_day"]
ALL_STOCK_COLUMNS = ["code", "tradeStatus", "code_name"]
STOCK_BASIC_COLUMNS = ["code", "code_name", "ipoDate", "outDate", "type", "status"]
DAILY_BAR_COLUMNS = ["date", "code", "open", "high", "low", "close", "volume", "amount", "tradestatus", "isST"]


class BaostockSource:
    """Context-managed wrapper around the Baostock Python client."""

    def __init__(self, config: BaostockSourceConfig | None = None) -> None:
        self.config = config or BaostockSourceConfig()
        self._bs: Any | None = None
        self._logged_in = False
        self.errors: list[dict[str, str]] = []
        self._last_request_at = 0.0

    def __enter__(self) -> "BaostockSource":
        self.open()
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.close()

    @property
    def version(self) -> str:
        if self._bs is None:
            return ""
        return str(getattr(self._bs, "__version__", "unknown"))

    def open(self) -> None:
        try:
            import baostock as bs  # type: ignore[import-not-found]
        except Exception as exc:  # noqa: BLE001
            raise BaostockSourceError(f"failed to import baostock: {exc}") from exc
        login = bs.login()
        if str(login.error_code) != "0":
            raise BaostockSourceError(f"baostock login failed: {login.error_msg}")
        self._bs = bs
        self._logged_in = True

    def close(self) -> None:
        if self._bs is not None and self._logged_in:
            self._bs.logout()
        self._logged_in = False
        self._bs = None

    @property
    def bs(self) -> Any:
        if self._bs is None:
            raise BaostockSourceError("Baostock source is not initialized")
        return self._bs

    def query_trade_dates(self, start_date: str, end_date: str) -> pd.DataFrame:
        return self._query_with_retry(
            "query_trade_dates",
            lambda: self.bs.query_trade_dates(start_date=start_date, end_date=end_date),
            TRADE_DATES_COLUMNS,
        )

    def query_all_stock(self, day: str) -> pd.DataFrame:
        return self._query_with_retry("query_all_stock", lambda: self.bs.query_all_stock(day=day), ALL_STOCK_COLUMNS)

    def query_stock_basic(self, code: str = "") -> pd.DataFrame:
        return self._query_with_retry(
            "query_stock_basic",
            lambda: self.bs.query_stock_basic(code=code),
            STOCK_BASIC_COLUMNS,
        )

    def query_daily_bars(self, code: str, start_date: str, end_date: str) -> pd.DataFrame:
        fields = ",".join(
            [
                "date",
                "code",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "amount",
                "tradestatus",
                "isST",
            ]
        )
        return self._query_with_retry(
            "query_history_k_data_plus",
            lambda: self.bs.query_history_k_data_plus(
                code,
                fields,
                start_date=start_date,
                end_date=end_date,
                frequency="d",
                adjustflag="3",
            ),
            DAILY_BAR_COLUMNS,
        )

    def _query_with_retry(self, name: str, query: Any, columns: list[str]) -> pd.DataFrame:
        last_error = ""
        for attempt in range(1, self.config.retry_count + 1):
            self._throttle()
            try:
                return _result_to_frame(query(), columns)
            except Exception as exc:  # noqa: BLE001
                last_error = str(exc)
                self.errors.append({"query": name, "attempt": str(attempt), "error": last_error})
                if attempt >= self.config.retry_count:
                    break
                delay = self.config.retry_base_delay_seconds * (2 ** (attempt - 1))
                time.sleep(delay)
        raise BaostockSourceError(f"{name} failed after {self.config.retry_count} attempts: {last_error}")

    def _throttle(self) -> None:
        interval = self.config.request_interval_seconds
        if interval <= 0:
            return
        elapsed = time.monotonic() - self._last_request_at
        if elapsed < interval:
            time.sleep(interval - elapsed)
        self._last_request_at = time.monotonic()


def _result_to_frame(result: Any, columns: list[str]) -> pd.DataFrame:
    if str(result.error_code) != "0":
        raise BaostockSourceError(f"baostock query failed: {result.error_msg}")
    rows: list[list[str]] = []
    while result.next():
        rows.append(result.get_row_data())
    result_columns = list(result.fields) if getattr(result, "fields", None) else columns
    frame = pd.DataFrame(rows, columns=result_columns)
    for column in columns:
        if column not in frame.columns:
            frame[column] = pd.Series(dtype="object")
    return frame[columns]
