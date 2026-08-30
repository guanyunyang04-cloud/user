"""Research Event Update: report_download responsibilities."""

from __future__ import annotations

from collections import deque
from collections.abc import Sequence
from concurrent.futures import (
    FIRST_COMPLETED,
    Future,
    ThreadPoolExecutor,
    as_completed,
    wait,
)
from pathlib import Path
from typing import Any

import pandas as pd

from quantlab.data.qdp_v2.auxiliary_update.context import _resolve_tushare_token, _TushareClient
from quantlab.data.qdp_v2.manifest import (
    utc_now,
)

from .config import (
    END_DATE,
    MAX_WORKERS,
    REPORT_EASTMONEY_START,
    REPORT_RC_EMPTY_CONFIRMATIONS,
    REPORT_RC_FIELDS,
    REPORT_RC_PAGE_SIZE,
    ResearchEventUpdateError,
)
from .context import (
    _cached_report_day,
    _frame_schema_hash,
    _identity_symbols,
    _legacy_runtime,
    _next_report_offset,
    _parquet_safe_provider_frame,
    _read_state,
    _report_rc_page_path,
    _report_request_dates,
    _request_json,
    _runtime,
    _sha256,
    _workspace,
    _write_parquet,
    _write_state,
)


def _initialize_tushare_report_days(
    workspace: Path,
    *,
    state: dict[str, Any],
    dates: Sequence[str],
) -> tuple[dict[str, Any], deque[str], dict[str, int]]:
    day_state: dict[str, Any] = {}
    pending: deque[str] = deque()
    next_offsets: dict[str, int] = {}
    stored_days = dict(state.get("tushare_report_rc", {}) or {}).get("days", {})
    previous_days = stored_days if isinstance(stored_days, dict) else {}
    for report_date in dates:
        offset, complete, pages, terminal_status = _cached_report_day(workspace, report_date)
        previous = dict(previous_days.get(report_date, {}) or {})
        day_state[report_date] = {
            "status": terminal_status if complete else "pending",
            "pages": pages,
            "raw_row_count": int(sum(item["row_count"] for item in pages.values())),
            "attempt_count": int(previous.get("attempt_count", 0) or 0),
            "empty_confirmation_count": int(previous.get("empty_confirmation_count", 0) or 0),
            "last_error": "",
        }
        if not complete:
            pending.append(report_date)
            next_offsets[report_date] = offset
    return day_state, pending, next_offsets


def _fetch_tushare_report_page(
    client: _TushareClient,
    report_date: str,
    offset: int,
) -> pd.DataFrame:
    compact = report_date.replace("-", "")
    return client.fetch(
        "report_rc",
        params={
            "start_date": compact,
            "end_date": compact,
            "limit": REPORT_RC_PAGE_SIZE,
            "offset": int(offset),
        },
        fields=REPORT_RC_FIELDS,
    )


def _checkpoint_tushare_reports(
    workspace: Path,
    *,
    state: dict[str, Any],
    day_state: dict[str, Any],
    dates: Sequence[str],
    workers: int,
    status: str,
) -> None:
    state["status"] = status
    terminal = all(item["status"] in {"observed", "confirmed_empty"} for item in day_state.values())
    state["tushare_report_rc"] = {
        "status": "completed" if terminal else status,
        "maximum_workers": workers,
        "page_size": REPORT_RC_PAGE_SIZE,
        "empty_confirmations_required": REPORT_RC_EMPTY_CONFIRMATIONS,
        "request_granularity": "report_date",
        "requested_date_count": len(dates),
        "days": day_state,
        "credential_persisted": False,
    }
    _write_state(workspace, state)


def _fill_tushare_report_slots(
    pool: ThreadPoolExecutor,
    *,
    client: _TushareClient,
    pending: deque[str],
    next_offsets: dict[str, int],
    day_state: dict[str, Any],
    active: dict[Future[pd.DataFrame], tuple[str, int]],
    workers: int,
) -> None:
    while pending and len(active) < workers:
        report_date = pending.popleft()
        offset = next_offsets[report_date]
        day_state[report_date]["status"] = "downloading"
        active[pool.submit(_fetch_tushare_report_page, client, report_date, offset)] = (
            report_date,
            offset,
        )


def _handle_tushare_report_result(
    workspace: Path,
    *,
    client: _TushareClient,
    pool: ThreadPoolExecutor,
    future: Future[pd.DataFrame],
    report_date: str,
    offset: int,
    day_state: dict[str, Any],
    next_offsets: dict[str, int],
    active: dict[Future[pd.DataFrame], tuple[str, int]],
) -> str:
    record = day_state[report_date]
    record["attempt_count"] = int(record.get("attempt_count", 0) or 0) + 1
    try:
        frame = _parquet_safe_provider_frame(future.result())
    except Exception as exc:  # noqa: BLE001 - provider failures are ledger data
        record["status"] = "failed"
        record["error_type"] = type(exc).__name__
        record["last_error"] = f"{type(exc).__name__}:{str(exc)[:240]}"
        return "failed"
    record["last_error"] = ""
    record["error_type"] = ""
    if frame.empty:
        confirmations = int(record.get("empty_confirmation_count", 0) or 0) + 1
        record["empty_confirmation_count"] = confirmations
        if confirmations < REPORT_RC_EMPTY_CONFIRMATIONS:
            record["status"] = "confirming_empty"
            active[pool.submit(_fetch_tushare_report_page, client, report_date, offset)] = (report_date, offset)
            return "resubmitted"
    else:
        record["empty_confirmation_count"] = 0
    path = _report_rc_page_path(workspace, report_date, offset)
    _write_parquet(frame, path)
    page_record = {
        "path": str(path),
        "row_count": len(frame),
        "sha256": _sha256(path),
        "response_hash": _sha256(path),
        "schema_hash": _frame_schema_hash(frame),
        "request_status": "success",
        "completed_at": utc_now(),
    }
    record["pages"][str(offset)] = page_record
    record["raw_row_count"] = int(sum(item["row_count"] for item in record["pages"].values()))
    following = _next_report_offset(len(frame), offset)
    if following is None:
        record["status"] = "confirmed_empty" if not record["raw_row_count"] else "observed"
    else:
        record["status"] = "downloading"
        next_offsets[report_date] = following
        active[pool.submit(_fetch_tushare_report_page, client, report_date, following)] = (report_date, following)
    return "processed"


def download_tushare_reports(
    *,
    workspace_root: str | Path | None = None,
    max_workers: int = MAX_WORKERS,
) -> dict[str, Any]:
    workspace = _workspace(workspace_root)
    state = _read_state(workspace)
    token = _resolve_tushare_token(workspace)
    if not token:
        raise ResearchEventUpdateError("tushare_token_required")
    client = _TushareClient(token, workspace_root=workspace)
    dates = _report_request_dates()
    day_state, pending, next_offsets = _initialize_tushare_report_days(
        workspace,
        state=state,
        dates=dates,
    )

    active: dict[Future[pd.DataFrame], tuple[str, int]] = {}
    workers = min(max(1, int(max_workers)), MAX_WORKERS)
    processed = 0

    with ThreadPoolExecutor(max_workers=workers) as pool:
        _fill_tushare_report_slots(
            pool,
            client=client,
            pending=pending,
            next_offsets=next_offsets,
            day_state=day_state,
            active=active,
            workers=workers,
        )
        while active:
            completed, _ = wait(tuple(active), return_when=FIRST_COMPLETED)
            for future in completed:
                report_date, offset = active.pop(future)
                outcome = _handle_tushare_report_result(
                    workspace,
                    client=client,
                    pool=pool,
                    future=future,
                    report_date=report_date,
                    offset=offset,
                    day_state=day_state,
                    next_offsets=next_offsets,
                    active=active,
                )
                if outcome == "resubmitted":
                    continue
                processed += 1
                if processed % 25 == 0:
                    _checkpoint_tushare_reports(
                        workspace,
                        state=state,
                        day_state=day_state,
                        dates=dates,
                        workers=workers,
                        status="failed" if outcome == "failed" else "downloading_tushare_reports",
                    )
                _fill_tushare_report_slots(
                    pool,
                    client=client,
                    pending=pending,
                    next_offsets=next_offsets,
                    day_state=day_state,
                    active=active,
                    workers=workers,
                )
    failed = [
        report_date
        for report_date, record in day_state.items()
        if record["status"] not in {"observed", "confirmed_empty"}
    ]
    _checkpoint_tushare_reports(
        workspace,
        state=state,
        day_state=day_state,
        dates=dates,
        workers=workers,
        status="failed" if failed else "tushare_reports_downloaded",
    )
    if failed:
        raise ResearchEventUpdateError(f"tushare_report_daily_tasks_incomplete:{len(failed)}")
    return dict(state["tushare_report_rc"])


def _eastmoney_report_path(workspace: Path, symbol: str) -> Path:
    relative = Path("raw") / "eastmoney_reports" / f"symbol={symbol.replace('.', '_')}.parquet"
    current = _runtime(workspace) / relative
    legacy = _legacy_runtime(workspace) / relative
    if not current.is_file() and legacy.is_file():
        return legacy
    return current


def _eastmoney_report_paths(workspace: Path) -> list[Path]:
    paths = {
        path.resolve()
        for root in (_legacy_runtime(workspace), _runtime(workspace))
        for path in (root / "raw" / "eastmoney_reports").glob("*.parquet")
    }
    return sorted(paths)


def _eastmoney_report_output_path(workspace: Path, symbol: str) -> Path:
    return _runtime(workspace) / "raw" / "eastmoney_reports" / f"symbol={symbol.replace('.', '_')}.parquet"


def _fetch_eastmoney_reports(symbol: str) -> pd.DataFrame:
    code = str(symbol).split(".", 1)[0]
    url = "https://reportapi.eastmoney.com/report/list"
    common = {
        "industryCode": "*",
        "pageSize": "5000",
        "industry": "*",
        "rating": "*",
        "ratingChange": "*",
        "beginTime": REPORT_EASTMONEY_START,
        "endTime": END_DATE,
        "fields": "",
        "qType": "0",
        "orgCode": "",
        "code": code,
        "rcode": "",
    }
    pages: list[pd.DataFrame] = []
    page = 1
    total_pages = 1
    while page <= total_pages:
        params = {
            **common,
            "pageNo": str(page),
            "p": str(page),
            "pageNum": str(page),
            "pageNumber": str(page),
        }
        payload = _request_json("GET", url, params=params)
        total_pages = max(1, int(payload.get("TotalPage", 0) or 0))
        frame = pd.DataFrame(list(payload.get("data", []) or []))
        if not frame.empty:
            frame["_query_symbol"] = symbol
            pages.append(frame)
        page += 1
    if pages:
        return pd.concat(pages, ignore_index=True)
    return pd.DataFrame({"_query_symbol": pd.Series(dtype="object")})


def _download_symbol_files(
    *,
    workspace: Path,
    state_key: str,
    symbols: Sequence[str],
    path_for: Any,
    fetch: Any,
    max_workers: int,
    allow_failures: bool,
) -> dict[str, Any]:
    state = _read_state(workspace)
    completed = {symbol for symbol in symbols if path_for(workspace, symbol).is_file()}
    failures: dict[str, str] = {}
    pending = [symbol for symbol in symbols if symbol not in completed]
    workers = min(max(1, int(max_workers)), MAX_WORKERS)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(fetch, symbol): symbol for symbol in pending}
        for number, future in enumerate(as_completed(futures), start=1):
            symbol = futures[future]
            try:
                frame = _parquet_safe_provider_frame(future.result())
                path = path_for(workspace, symbol)
                _write_parquet(frame, path)
                completed.add(symbol)
                failures.pop(symbol, None)
            except (
                OSError,
                ResearchEventUpdateError,
                RuntimeError,
                TypeError,
                ValueError,
            ) as exc:
                failures[symbol] = f"{type(exc).__name__}:{str(exc)[:240]}"
            if number % 25 == 0 or number == len(futures):
                state[state_key] = {
                    "status": "completed" if len(completed) == len(symbols) else "downloading",
                    "maximum_workers": workers,
                    "symbol_count": len(symbols),
                    "completed_symbol_count": len(completed),
                    "failed_symbol_count": len(failures),
                    "failures": failures,
                }
                state["status"] = f"downloading_{state_key}"
                _write_state(workspace, state)
    state[state_key] = {
        "status": "completed" if len(completed) == len(symbols) else "partial",
        "maximum_workers": workers,
        "symbol_count": len(symbols),
        "completed_symbol_count": len(completed),
        "failed_symbol_count": len(failures),
        "failures": failures,
    }
    _write_state(workspace, state)
    if failures and not allow_failures:
        raise ResearchEventUpdateError(f"{state_key}_incomplete:{len(failures)}")
    return dict(state[state_key])


def download_eastmoney_reports(
    *,
    workspace_root: str | Path | None = None,
    max_workers: int = MAX_WORKERS,
) -> dict[str, Any]:
    workspace = _workspace(workspace_root)
    return _download_symbol_files(
        workspace=workspace,
        state_key="eastmoney_reports",
        symbols=_identity_symbols(workspace),
        path_for=_eastmoney_report_path,
        fetch=_fetch_eastmoney_reports,
        max_workers=max_workers,
        allow_failures=False,
    )
