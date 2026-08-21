"""Recent Market Repair: inventory responsibilities."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import pandas as pd

from quantlab.data.qdp_v2.duckdb_resources import (
    DuckDbMemoryFloorError,
    open_guarded_duckdb,
)
from quantlab.data.qdp_v2.manifest import (
    dataset_manifest_for_id,
    qdp_v2_root,
    read_active_manifest,
    read_dataset_manifest,
    resolve_manifest_path,
)

from .config import (
    DAILY_DOMAIN,
    INTRADAY_DOMAIN,
    RecentMarketRepairError,
    _DomainInput,
)
from .state import (
    _path_texts,
    _runtime_root,
)


def _missing_daily_symbols(
    inputs: Mapping[str, _DomainInput],
    *,
    trade_date: str,
    workspace: Path,
) -> tuple[str, ...]:
    with open_guarded_duckdb(
        temp_directory=_runtime_root(workspace) / "inventory_spill",
        threads=4,
    ) as connection:
        rows = connection.execute(
            """
            WITH eligible AS (
              SELECT DISTINCT cast(u.symbol AS VARCHAR) AS symbol
              FROM read_parquet(?, union_by_name=true) AS u
              LEFT JOIN read_parquet(?, union_by_name=true) AS s
                ON cast(s.symbol AS VARCHAR)=cast(u.symbol AS VARCHAR)
               AND cast(s.trade_date AS VARCHAR)=cast(u.trade_date AS VARCHAR)
              WHERE cast(u.trade_date AS VARCHAR)=?
                AND upper(cast(u.list_status AS VARCHAR))='L'
                AND NOT coalesce(try_cast(s.is_suspended AS BOOLEAN), false)
                AND NOT coalesce(try_cast(s.is_delisted AS BOOLEAN), false)
              UNION
              SELECT DISTINCT cast(i.current_symbol AS VARCHAR) AS symbol
              FROM read_parquet(?, union_by_name=true) AS i
              WHERE coalesce(cast(i.current_symbol AS VARCHAR),'')<>''
                AND NOT EXISTS (
                  SELECT 1 FROM read_parquet(?, union_by_name=true) AS u2
                  WHERE cast(u2.trade_date AS VARCHAR)=?
                )
            ), existing AS (
              SELECT DISTINCT cast(symbol AS VARCHAR) AS symbol
              FROM read_parquet(?, union_by_name=true)
              WHERE cast(trade_date AS VARCHAR)=?
            )
            SELECT e.symbol FROM eligible e
            LEFT JOIN existing x USING(symbol)
            WHERE x.symbol IS NULL
            ORDER BY e.symbol
            """,
            [
                _path_texts(inputs["universe_snapshot"]),
                _path_texts(inputs["security_status"]),
                trade_date,
                _path_texts(inputs["security_identity"]),
                _path_texts(inputs["universe_snapshot"]),
                trade_date,
                _path_texts(inputs[DAILY_DOMAIN]),
                trade_date,
            ],
        ).fetchall()
    return tuple(str(row[0]) for row in rows)


def _missing_intraday_pairs(
    inputs: Mapping[str, _DomainInput],
    *,
    start_date: str,
    end_date: str,
    workspace: Path,
) -> tuple[tuple[str, str], ...]:
    rows: list[tuple[str, str]] = []
    pending = list(_calendar_year_windows(start_date, end_date))
    while pending:
        window_start, window_end = pending.pop(0)
        try:
            rows.extend(
                _missing_intraday_pairs_window(
                    inputs,
                    start_date=window_start,
                    end_date=window_end,
                    workspace=workspace,
                )
            )
        except DuckDbMemoryFloorError:
            first = pd.Timestamp(window_start)
            last = pd.Timestamp(window_end)
            if first >= last:
                raise
            midpoint = first + (last - first) // 2
            left = (first.strftime("%Y-%m-%d"), midpoint.strftime("%Y-%m-%d"))
            right = (
                (midpoint + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
                last.strftime("%Y-%m-%d"),
            )
            pending[0:0] = [left, right]
    return tuple(sorted(set(rows)))


def _calendar_year_windows(start_date: str, end_date: str) -> tuple[tuple[str, str], ...]:
    first = pd.Timestamp(start_date).normalize()
    last = pd.Timestamp(end_date).normalize()
    if first > last:
        raise ValueError("start_date_after_end_date")
    windows: list[tuple[str, str]] = []
    cursor = first
    while cursor <= last:
        year_end = min(last, pd.Timestamp(year=cursor.year, month=12, day=31))
        windows.append((cursor.strftime("%Y-%m-%d"), year_end.strftime("%Y-%m-%d")))
        cursor = year_end + pd.Timedelta(days=1)
    return tuple(windows)


def _missing_intraday_pairs_window(
    inputs: Mapping[str, _DomainInput],
    *,
    start_date: str,
    end_date: str,
    workspace: Path,
) -> tuple[tuple[str, str], ...]:
    with open_guarded_duckdb(
        temp_directory=_runtime_root(workspace) / "inventory_spill",
        threads=4,
    ) as connection:
        rows = connection.execute(
            """
            WITH eligible AS (
              SELECT DISTINCT cast(d.symbol AS VARCHAR) AS symbol,
                              cast(d.trade_date AS VARCHAR) AS trade_date
              FROM read_parquet(?, union_by_name=true) AS d
              WHERE cast(d.trade_date AS VARCHAR) BETWEEN ? AND ?
                AND coalesce(try_cast(d.volume AS DOUBLE),0)>0
            ), complete AS (
              SELECT cast(symbol AS VARCHAR) AS symbol,
                     cast(trade_date AS VARCHAR) AS trade_date
              FROM read_parquet(?, union_by_name=true)
              WHERE cast(trade_date AS VARCHAR) BETWEEN ? AND ?
              GROUP BY 1,2
              HAVING count(*)=48 AND count(DISTINCT cast(bar_time AS VARCHAR))=48
            )
            SELECT e.symbol,e.trade_date FROM eligible e
            LEFT JOIN complete c USING(symbol,trade_date)
            WHERE c.symbol IS NULL
            ORDER BY e.symbol,e.trade_date
            """,
            [
                _path_texts(inputs[DAILY_DOMAIN]),
                start_date,
                end_date,
                _path_texts(inputs[INTRADAY_DOMAIN]),
                start_date,
                end_date,
            ],
        ).fetchall()
    return tuple((str(symbol), str(trade_date)) for symbol, trade_date in rows)


def _active_inputs(
    workspace: Path,
    domains: Sequence[str],
    *,
    start_date: str = "",
    end_date: str = "",
) -> dict[str, _DomainInput]:
    root = qdp_v2_root(workspace).resolve()
    active = read_active_manifest(root)
    inputs: dict[str, _DomainInput] = {}
    for domain in domains:
        dataset_id = str(active.get("datasets", {}).get(domain, ""))
        manifest_path = dataset_manifest_for_id(root, dataset_id, domain)
        if manifest_path is None:
            raise RecentMarketRepairError(f"recent_repair_active_domain_missing:{domain}")
        manifest = read_dataset_manifest(manifest_path)
        entries = manifest.shards
        if start_date and end_date:
            overlapping = [
                item
                for item in entries
                if not item.start_date
                or not item.end_date
                or (str(item.start_date) <= end_date and str(item.end_date) >= start_date)
            ]
            if overlapping:
                entries = overlapping
        paths = tuple(resolve_manifest_path(item.path, root=root) for item in entries)
        if not paths or any(not item.is_file() for item in paths):
            raise RecentMarketRepairError(f"recent_repair_active_shards_missing:{domain}")
        inputs[domain] = _DomainInput(
            domain=domain,
            dataset_id=dataset_id,
            manifest_path=manifest_path,
            paths=paths,
        )
    return inputs
