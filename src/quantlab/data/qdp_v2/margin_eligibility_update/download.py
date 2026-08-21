"""Margin Eligibility Update: download responsibilities."""

from __future__ import annotations

from collections import deque
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq

from quantlab.data.qdp_v2.research_event_update import (
    _sha256,
)

from .config import (
    END_DATE,
    ENDPOINTS,
    MAX_WORKERS,
    START_DATE,
    MarginEligibilityUpdateError,
)
from .context import (
    _raw_day_path,
    _read_state,
    _trade_dates,
    _workspace,
    _write_state,
)
from .sources import (
    _fetch_day,
)


def download(
    *,
    workspace_root: str | Path | None = None,
    max_workers: int = MAX_WORKERS,
) -> dict[str, Any]:
    workspace = _workspace(workspace_root)
    state = _read_state(workspace)
    dates = _trade_dates(workspace)
    days = dict(state.get("days", {}) or {})
    pending: deque[str] = deque()
    for trade_date in dates:
        path = _raw_day_path(workspace, trade_date)
        record = dict(days.get(trade_date, {}) or {})
        if path.is_file():
            try:
                endpoints = set(pd.read_parquet(path, columns=["endpoint"])["endpoint"])
                if endpoints == set(ENDPOINTS):
                    record.update(
                        {
                            "status": "observed",
                            "path": str(path),
                            "row_count": int(pq.ParquetFile(path).metadata.num_rows),
                            "sha256": _sha256(path),
                        }
                    )
                    days[trade_date] = record
                    continue
            except (OSError, ValueError):
                pass
        pending.append(trade_date)
        days[trade_date] = {
            **record,
            "status": "pending",
            "attempt_count": int(record.get("attempt_count", 0) or 0),
            "last_error": "",
        }

    workers = min(MAX_WORKERS, max(1, int(max_workers)))

    def checkpoint(status: str) -> None:
        state.update(
            {
                "status": status,
                "start_date": START_DATE,
                "end_date": END_DATE,
                "requested_date_count": len(dates),
                "maximum_workers": workers,
                "days": days,
                "request_2026_count": 0,
            }
        )
        _write_state(workspace, state)

    active: dict[Future[dict[str, Any]], str] = {}

    def fill(pool: ThreadPoolExecutor) -> None:
        while pending and len(active) < workers:
            trade_date = pending.popleft()
            days[trade_date]["status"] = "downloading"
            active[pool.submit(_fetch_day, workspace, trade_date)] = trade_date

    processed = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        fill(pool)
        while active:
            completed, _ = wait(tuple(active), return_when=FIRST_COMPLETED)
            for future in completed:
                trade_date = active.pop(future)
                prior_attempts = int(days[trade_date].get("attempt_count", 0) or 0)
                try:
                    days[trade_date] = {
                        **future.result(),
                        "attempt_count": prior_attempts + 1,
                        "last_error": "",
                    }
                except Exception as exc:  # noqa: BLE001 - source failures are ledger data
                    days[trade_date] = {
                        **days[trade_date],
                        "status": "failed",
                        "attempt_count": prior_attempts + 1,
                        "error_type": type(exc).__name__,
                        "last_error": f"{type(exc).__name__}:{str(exc)[:480]}",
                    }
                processed += 1
                if processed % 25 == 0:
                    checkpoint("downloading")
                fill(pool)
    failed = [date for date in dates if days[date].get("status") != "observed"]
    checkpoint("failed" if failed else "downloaded")
    if failed:
        raise MarginEligibilityUpdateError(f"exchange_daily_tasks_incomplete:{len(failed)}")
    return state
