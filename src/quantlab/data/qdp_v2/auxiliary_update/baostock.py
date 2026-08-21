"""Auxiliary update baostock operations."""

from __future__ import annotations

import io
import os
import time
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Any

import pandas as pd

from quantlab.data.qdp_v2 import normalization as _normalization

from .context import (
    BAOSTOCK_WORKERS,
    INDEX_SPECS,
    AuxiliaryContext,
    AuxiliaryUpdateError,
    _parquet_schema_columns,
    _split_evenly,
)


def _quiet_baostock_login(bs: Any) -> Any:
    with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
        return bs.login()


def _quiet_baostock_logout(bs: Any) -> None:
    with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
        try:
            bs.logout()
        except Exception:
            pass


def _industry_snapshot_rows(bs: Any, trade_date: str) -> list[dict[str, Any]]:
    query = bs.query_stock_industry(date=str(trade_date))
    if str(query.error_code) != "0":
        raise AuxiliaryUpdateError(f"baostock_industry_query_failed:{trade_date}")
    rows: list[dict[str, Any]] = []
    while query.error_code == "0" and query.next():
        row = dict(zip(query.fields, query.get_row_data(), strict=True))
        code = _from_baostock_code(row.get("code", ""))
        if code:
            rows.append(
                {
                    "symbol": code,
                    "snapshot_query_date": str(trade_date),
                    "industry": str(row.get("industry", "") or "").strip(),
                    "industry_standard": str(
                        row.get("industryClassification", "") or "证监会行业分类"
                    ).strip(),
                    "source_date": _provider_date(row.get("updateDate"), trade_date),
                    "source": "baostock.query_stock_industry",
                }
            )
    if not rows:
        raise AuxiliaryUpdateError(f"baostock_industry_query_empty:{trade_date}")
    return rows


def _index_snapshot_rows(bs: Any, trade_date: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    index_counts: dict[str, int] = {}
    for index_symbol, index_name, endpoint in INDEX_SPECS:
        before = len(rows)
        query = getattr(bs, f"query_{endpoint}_stocks")(date=str(trade_date))
        if str(query.error_code) != "0":
            raise AuxiliaryUpdateError(f"baostock_index_query_failed:{endpoint}:{trade_date}")
        while query.error_code == "0" and query.next():
            row = dict(zip(query.fields, query.get_row_data(), strict=True))
            code = _from_baostock_code(row.get("code", ""))
            if code:
                rows.append(
                    {
                        "index_symbol": index_symbol,
                        "symbol": code,
                        "index_name": index_name,
                        "snapshot_query_date": str(trade_date),
                        "source_snapshot_date": str(trade_date),
                        "source": "baostock",
                    }
                )
        index_counts[index_symbol] = len(rows) - before
    missing = [symbol for symbol, _, _ in INDEX_SPECS if index_counts.get(symbol, 0) <= 0]
    if missing:
        raise AuxiliaryUpdateError(f"baostock_index_query_empty:{trade_date}:{','.join(missing)}")
    return rows


def _snapshot_rows(bs: Any, kind: str, trade_date: str) -> list[dict[str, Any]]:
    if kind == "industry":
        return _industry_snapshot_rows(bs, trade_date)
    if kind == "index":
        return _index_snapshot_rows(bs, trade_date)
    raise ValueError(f"unsupported_baostock_snapshot_kind:{kind}")


def _write_snapshot_part(
    destination: Path,
    *,
    trade_date: str,
    rows: list[dict[str, Any]],
    columns: Sequence[str],
) -> Path:
    output = destination / _baostock_snapshot_part_name(trade_date)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    temporary.unlink(missing_ok=True)
    try:
        pd.DataFrame(rows, columns=columns).to_parquet(
            temporary,
            index=False,
            compression="zstd",
        )
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    return output


def _baostock_snapshot_worker(
    kind: str,
    dates: Sequence[str],
    output_dir: str,
) -> dict[str, Any]:
    import baostock as bs

    columns = _baostock_snapshot_columns(kind)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    row_count = 0
    failures: list[str] = []
    outputs: list[str] = []
    login = _quiet_baostock_login(bs)
    if str(getattr(login, "error_code", "1")) != "0":
        raise AuxiliaryUpdateError("baostock_snapshot_login_failed")
    try:
        for trade_date in dates:
            for attempt in range(3):
                try:
                    rows = _snapshot_rows(bs, kind, str(trade_date))
                    output = _write_snapshot_part(
                        destination,
                        trade_date=str(trade_date),
                        rows=rows,
                        columns=columns,
                    )
                    row_count += len(rows)
                    outputs.append(str(output))
                    break
                except Exception:
                    if attempt == 2:
                        failures.append(str(trade_date))
                        break
                    _quiet_baostock_logout(bs)
                    login = _quiet_baostock_login(bs)
                    time.sleep(1 + attempt)
                    if str(getattr(login, "error_code", "1")) != "0":
                        continue
    finally:
        _quiet_baostock_logout(bs)
    return {
        "row_count": row_count,
        "failure_dates": failures,
        "output_paths": outputs,
    }


def _baostock_snapshot_columns(kind: str) -> list[str]:
    if kind == "industry":
        return [
            "symbol",
            "snapshot_query_date",
            "industry",
            "industry_standard",
            "source_date",
            "source",
        ]
    if kind == "index":
        return [
            "index_symbol",
            "symbol",
            "index_name",
            "snapshot_query_date",
            "source_snapshot_date",
            "source",
        ]
    raise ValueError(f"unsupported_baostock_snapshot_kind:{kind}")


def _baostock_snapshot_part_name(trade_date: str) -> str:
    return f"snapshot_{str(trade_date).replace('-', '')}.parquet"


def _valid_parquet_columns(path: Path, expected: Sequence[str]) -> bool:
    if not path.is_file():
        return False
    try:
        return _parquet_schema_columns(path) == list(expected)
    except Exception:
        return False


def _fetch_baostock_snapshots(
    ctx: AuxiliaryContext,
    *,
    kind: str,
    dates: Sequence[str],
) -> list[Path]:
    unique_dates = sorted(set(str(item) for item in dates))
    if not unique_dates:
        raise AuxiliaryUpdateError(f"baostock_{kind}_snapshot_dates_empty")
    output_dir = ctx.runtime / f"baostock_{kind}_parts"
    output_dir.mkdir(parents=True, exist_ok=True)
    expected_columns = _baostock_snapshot_columns(kind)
    outputs = [output_dir / _baostock_snapshot_part_name(trade_date) for trade_date in unique_dates]
    pending_dates: list[str] = []
    for trade_date, output in zip(unique_dates, outputs, strict=True):
        if _valid_parquet_columns(output, expected_columns):
            continue
        output.unlink(missing_ok=True)
        pending_dates.append(trade_date)
    chunks = _split_evenly(pending_dates, BAOSTOCK_WORKERS)
    if chunks:
        with ProcessPoolExecutor(max_workers=len(chunks)) as pool:
            futures = {
                pool.submit(
                    _baostock_snapshot_worker,
                    kind,
                    chunk,
                    str(output_dir),
                ): chunk
                for chunk in chunks
            }
            failures: list[str] = []
            for future in as_completed(futures):
                result = future.result()
                failures.extend(list(result.get("failure_dates", []) or []))
        if failures:
            raise AuxiliaryUpdateError(f"baostock_{kind}_snapshot_failures:{len(failures)}:" + ",".join(failures[:10]))
    incomplete = [
        trade_date
        for trade_date, output in zip(unique_dates, outputs, strict=True)
        if not _valid_parquet_columns(output, expected_columns)
    ]
    if incomplete:
        raise AuxiliaryUpdateError(
            f"baostock_{kind}_snapshot_incomplete:{len(incomplete)}:" + ",".join(incomplete[:10])
        )
    return outputs


_provider_date = _normalization.provider_date


_from_baostock_code = _normalization.from_baostock_code
