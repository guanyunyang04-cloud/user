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
        self._install_safe_socket_recv()
        login = bs.login()
        if str(login.error_code) != "0":
            raise BaostockSourceError(f"baostock login failed: {login.error_msg}")
        self._bs = bs
        self._logged_in = True
        self._set_socket_timeout()

    def close(self) -> None:
        if self._bs is not None and self._logged_in:
            try:
                self._bs.logout()
            except Exception as exc:  # noqa: BLE001
                self.errors.append({"query": "logout", "attempt": "0", "error": str(exc)})
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
                self._reopen()
                if attempt >= self.config.retry_count:
                    break
                delay = self.config.retry_base_delay_seconds * (2 ** (attempt - 1))
                time.sleep(delay)
        raise BaostockSourceError(f"{name} failed after {self.config.retry_count} attempts: {last_error}")

    def _reopen(self) -> None:
        if self._bs is None:
            return
        self.close()
        self.open()

    def _set_socket_timeout(self) -> None:
        if self._bs is None:
            return
        try:
            import baostock.common.context as context  # type: ignore[import-not-found]

            default_socket = getattr(context, "default_socket", None)
            if default_socket is not None:
                default_socket.settimeout(self.config.timeout_seconds)
        except Exception as exc:  # noqa: BLE001
            self.errors.append({"query": "set_socket_timeout", "attempt": "0", "error": str(exc)})

    def _install_safe_socket_recv(self) -> None:
        try:
            import zlib

            import baostock.common.contants as cons  # type: ignore[import-not-found]
            import baostock.common.context as context  # type: ignore[import-not-found]
            import baostock.util.socketutil as socketutil  # type: ignore[import-not-found]
        except Exception as exc:  # noqa: BLE001
            self.errors.append({"query": "install_safe_socket_recv", "attempt": "0", "error": str(exc)})
            return
        if getattr(socketutil, "_tqr_safe_send_msg_installed", False):
            return

        def safe_send_msg(msg: str) -> str | None:
            if not hasattr(context, "default_socket"):
                print("you don't login.")
                return None
            default_socket = getattr(context, "default_socket")
            if default_socket is None:
                return None
            default_socket.send(bytes(msg + "\n", encoding="utf-8"))
            receive = b""
            while True:
                recv = default_socket.recv(8192)
                if not recv:
                    raise ConnectionError("baostock socket closed while receiving data")
                receive += recv
                if receive[-13:] == b"<![CDATA[]]>\n":
                    break
            head_bytes = receive[0 : cons.MESSAGE_HEADER_LENGTH]
            head_str = bytes.decode(head_bytes)
            head_arr = head_str.split(cons.MESSAGE_SPLIT)
            if head_arr[1] in cons.COMPRESSED_MESSAGE_TYPE_TUPLE:
                head_inner_length = int(head_arr[2])
                body = receive[cons.MESSAGE_HEADER_LENGTH : cons.MESSAGE_HEADER_LENGTH + head_inner_length]
                return head_str + bytes.decode(zlib.decompress(body))
            return bytes.decode(receive)

        socketutil.send_msg = safe_send_msg
        socketutil._tqr_safe_send_msg_installed = True

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
