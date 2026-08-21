"""Recent Market Repair: session responsibilities."""

from __future__ import annotations

import io
import time
from contextlib import redirect_stderr, redirect_stdout
from typing import Any

import pandas as pd

from .config import (
    INTRADAY_COLUMNS,
    RecentMarketRepairError,
    _BaostockTask,
)


class _DirectBaostockSession:
    """One BaoStock login reused for sequential 5-minute range queries."""

    def __init__(self, *, socket_timeout_seconds: float = 90.0) -> None:
        self._socket_timeout_seconds = max(1.0, float(socket_timeout_seconds))
        self._bs: Any | None = None

    def _ensure_login(self) -> Any:
        if self._bs is not None:
            return self._bs
        import baostock as bs  # type: ignore

        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            login = bs.login()
        if str(getattr(login, "error_code", "1")) != "0":
            raise RuntimeError(f"baostock_login_failed:{getattr(login, 'error_msg', '')}")
        from baostock.common import context as baostock_context  # type: ignore

        active_socket = getattr(baostock_context, "default_socket", None)
        if active_socket is not None:
            active_socket.settimeout(self._socket_timeout_seconds)
        self._bs = bs
        return bs

    def fetch(self, task: _BaostockTask) -> pd.DataFrame:
        bs = self._ensure_login()
        code, exchange = task.symbol.split(".", 1)
        query = bs.query_history_k_data_plus(
            f"{exchange.lower()}.{code}",
            "date,time,code,open,high,low,close,volume,amount,adjustflag",
            start_date=task.start_date,
            end_date=task.end_date,
            frequency="5",
            adjustflag="3",
        )
        error_code = str(getattr(query, "error_code", "1"))
        if error_code != "0":
            raise RuntimeError(f"baostock_intraday_5m_query_error:{error_code}:{getattr(query, 'error_msg', '')}")
        fields = [str(item) for item in (getattr(query, "fields", None) or [])]
        rows: list[list[Any]] = []
        while query.next():
            rows.append(query.get_row_data())
        raw = pd.DataFrame(rows, columns=fields) if fields else pd.DataFrame(rows)
        return _normalize_baostock_raw_5m(raw, symbol=task.symbol)

    def reset(self) -> None:
        self.close()

    def close(self) -> None:
        bs = self._bs
        self._bs = None
        if bs is None:
            return
        try:
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                bs.logout()
        except Exception:  # provider logout is best effort
            pass


_BAOSTOCK_PROCESS_SESSION: _DirectBaostockSession | None = None


def _baostock_process_session(*, startup_delay_seconds: float = 0.0) -> _DirectBaostockSession:
    global _BAOSTOCK_PROCESS_SESSION
    if _BAOSTOCK_PROCESS_SESSION is None:
        if float(startup_delay_seconds) > 0:
            time.sleep(float(startup_delay_seconds))
        _BAOSTOCK_PROCESS_SESSION = _DirectBaostockSession()
    return _BAOSTOCK_PROCESS_SESSION


def _normalize_baostock_raw_5m(raw: pd.DataFrame, *, symbol: str) -> pd.DataFrame:
    if raw is None or raw.empty:
        return pd.DataFrame(columns=INTRADAY_COLUMNS)
    frame = raw.rename(columns={"date": "trade_date", "time": "bar_time"}).copy()
    if "trade_date" not in frame or "bar_time" not in frame:
        raise RecentMarketRepairError("baostock_intraday_missing_date_or_time")
    digits = frame["bar_time"].astype(str).str.replace(r"\D", "", regex=True)
    clock = digits.where(digits.str.len() < 12, digits.str.slice(8, 12))
    clock = clock.str.slice(0, 4).str.pad(4, side="left", fillchar="0")
    frame["bar_time"] = clock.str.slice(0, 2) + ":" + clock.str.slice(2, 4)
    frame["symbol"] = str(symbol).strip().upper()
    return frame
