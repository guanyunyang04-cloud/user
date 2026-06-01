"""Read-only adapter for Baostock point-in-time market data."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd


class BaostockSourceError(RuntimeError):
    """Raised when Baostock cannot be queried safely."""


@dataclass(frozen=True)
class BaostockSourceConfig:
    timeout_seconds: int = 30


class BaostockSource:
    """Context-managed wrapper around the Baostock Python client."""

    def __init__(self, config: BaostockSourceConfig | None = None) -> None:
        self.config = config or BaostockSourceConfig()
        self._bs: Any | None = None
        self._logged_in = False

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
        rs = self.bs.query_trade_dates(start_date=start_date, end_date=end_date)
        return _result_to_frame(rs)

    def query_all_stock(self, day: str) -> pd.DataFrame:
        rs = self.bs.query_all_stock(day=day)
        return _result_to_frame(rs)

    def query_stock_basic(self, code: str = "") -> pd.DataFrame:
        rs = self.bs.query_stock_basic(code=code)
        return _result_to_frame(rs)

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
        rs = self.bs.query_history_k_data_plus(
            code,
            fields,
            start_date=start_date,
            end_date=end_date,
            frequency="d",
            adjustflag="3",
        )
        return _result_to_frame(rs)


def _result_to_frame(result: Any) -> pd.DataFrame:
    if str(result.error_code) != "0":
        raise BaostockSourceError(f"baostock query failed: {result.error_msg}")
    rows: list[list[str]] = []
    while result.next():
        rows.append(result.get_row_data())
    return pd.DataFrame(rows, columns=list(result.fields))
