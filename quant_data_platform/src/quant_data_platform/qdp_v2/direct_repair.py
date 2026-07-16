from __future__ import annotations

"""Directly repair QDP v2 low-frequency active datasets from trusted facts.

The module is intentionally conservative around publication: the current
``active/active.json`` is read as an immutable input and is never written.  An
apply run prepares all five domains under the H-backed runtime, validates them,
then updates the already-active dataset manifests/shards through the guarded
repair API in calendar/daily/universe/status/factor order.  Each domain is
transactional; completed correct domains are retained on a later failure and
the state file supports a missing-only resume.  Full prepared files are always
deleted.
"""

import argparse
import bisect
import contextlib
import hashlib
import json
import os
import re
import shutil
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Sequence

import duckdb
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as pads
import pyarrow.parquet as pq

from quant_data_platform.core.json_io import json_safe
from quant_data_platform.qdp_v2.duckdb_resources import (
    DuckDbMemoryFloorError,
    DuckDbResourceSettings,
    GuardedDuckDbConnection,
    available_memory_bytes,
    configure_dynamic_duckdb,
    guard_configured_duckdb,
    total_physical_memory_bytes,
)
from quant_data_platform.qdp_v2.manifest import (
    DatasetManifest,
    atomic_write_json,
    dataset_manifest_for_id,
    qdp_v2_root,
    read_active_manifest,
    read_dataset_manifest,
    resolve_manifest_path,
    stable_hash,
    utc_now,
)
from quant_data_platform.qdp_v2.repair_sources import (
    TRUSTED_START_DATE,
    V2_ADJUST_FACTOR_COLUMNS,
    V2_MARKET_DAILY_COLUMNS,
    V2_SECURITY_STATUS_COLUMNS,
    V2_TRADING_CALENDAR_COLUMNS,
    IdentityRepairTables,
    RepairBatch,
    build_mainboard_identity_tables,
    iter_adjust_factor_repairs_by_security,
    tushare_daily_to_v2,
    tushare_trade_calendar_to_v2,
)
from quant_data_platform.qdp_v2.repair import (
    bulk_append_active_shards_from_parquet,
    replace_active_shard_from_parquet,
)
from quant_data_platform.qdp_v2.status import active_dataset_map
from quant_data_platform.qdp_v3.constants import (
    RAW_TUSHARE_PROXY_ADJ_FACTOR,
    RAW_TUSHARE_PROXY_DAILY,
    RAW_TUSHARE_PROXY_NAMECHANGE,
    RAW_TUSHARE_PROXY_STOCK_BASIC,
    RAW_TUSHARE_PROXY_SUSPEND,
    RAW_TUSHARE_PROXY_TRADE_CALENDAR,
)
from quant_data_platform.qdp_v3.identity import normalize_symbol
from quant_data_platform.qdp_v3.storage import (
    RawPartitionRef,
    iter_raw_partitions,
    read_raw_partition,
)


DIRECT_REPAIR_VERSION = "qdp_v2_direct_repair_trusted_tushare_v3"
REPAIR_DOMAINS = (
    "trading_calendar",
    "market_daily_raw",
    "universe_snapshot",
    "security_status",
    "adjust_factor",
)
MEMORY_FLOOR_BYTES = int(0.5 * 1024**3)
MEMORY_LOW_SECONDS = 5.0


class DirectRepairError(RuntimeError):
    pass


class DirectRepairPaused(DirectRepairError):
    pass


@dataclass(frozen=True)
class _ActiveInputs:
    root: Path
    active_path: Path
    active_bytes: bytes
    active_sha256: str
    active: dict[str, Any]
    manifests: dict[str, DatasetManifest]


@dataclass(frozen=True)
class _UniverseIdentity:
    security_id: str
    symbol: str
    name: str
    exchange: str
    list_date: str
    delist_date: str
    intervals: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class _ResolvedPlan:
    public: dict[str, Any]
    active: _ActiveInputs
    identities: IdentityRepairTables
    universe_identities: tuple[_UniverseIdentity, ...]
    open_dates: tuple[str, ...]
    calendar_refs: tuple[RawPartitionRef, ...]
    stock_basic_refs: tuple[RawPartitionRef, ...]
    daily_refs: tuple[RawPartitionRef, ...]
    factor_refs: tuple[RawPartitionRef, ...]
    suspend_refs: tuple[RawPartitionRef, ...]
    namechange_refs: tuple[RawPartitionRef, ...]


class _MemoryGuard:
    def __init__(
        self,
        *,
        sampler: Callable[[], int] | None = None,
        clock: Callable[[], float] = time.monotonic,
        sleeper: Callable[[float], None] = time.sleep,
        floor_bytes: int = MEMORY_FLOOR_BYTES,
        low_seconds: float = MEMORY_LOW_SECONDS,
    ) -> None:
        self._sampler = sampler or _available_memory_bytes
        self._clock = clock
        self._sleeper = sleeper
        self._floor = int(floor_bytes)
        self._low_seconds = float(low_seconds)

    def checkpoint(self) -> int:
        available = int(self._sampler())
        if available >= self._floor:
            return available
        low_since = self._clock()
        while available < self._floor:
            elapsed = self._clock() - low_since
            if elapsed >= self._low_seconds:
                raise DirectRepairPaused(
                    "available_memory_below_0.5_gib_for_5_seconds"
                )
            self._sleeper(min(0.25, max(0.0, self._low_seconds - elapsed)))
            available = int(self._sampler())
        return available


def plan_direct_repair(
    *,
    workspace_root: str | Path | None = None,
    identity_config_path: str | Path | None = None,
    start_date: str = TRUSTED_START_DATE,
    calendar_refs: Iterable[RawPartitionRef] | None = None,
    stock_basic_refs: Iterable[RawPartitionRef] | None = None,
    daily_refs: Iterable[RawPartitionRef] | None = None,
    factor_refs: Iterable[RawPartitionRef] | None = None,
    suspend_refs: Iterable[RawPartitionRef] | None = None,
    namechange_refs: Iterable[RawPartitionRef] | None = None,
    identity_tables: IdentityRepairTables | None = None,
    memory_sampler: Callable[[], int] | None = None,
    clock: Callable[[], float] = time.monotonic,
    sleeper: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    """Read inputs and return a deterministic, non-materializing repair plan."""

    guard = _MemoryGuard(
        sampler=memory_sampler,
        clock=clock,
        sleeper=sleeper,
    )
    resolved = _resolve_plan(
        workspace_root=workspace_root,
        identity_config_path=identity_config_path,
        start_date=start_date,
        calendar_refs=calendar_refs,
        stock_basic_refs=stock_basic_refs,
        daily_refs=daily_refs,
        factor_refs=factor_refs,
        suspend_refs=suspend_refs,
        namechange_refs=namechange_refs,
        identity_tables=identity_tables,
        guard=guard,
    )
    return resolved.public


def run_direct_repair(
    *,
    workspace_root: str | Path | None = None,
    apply: bool = False,
    factor_output_path: str | Path | None = None,
    identity_config_path: str | Path | None = None,
    start_date: str = TRUSTED_START_DATE,
    calendar_refs: Iterable[RawPartitionRef] | None = None,
    stock_basic_refs: Iterable[RawPartitionRef] | None = None,
    daily_refs: Iterable[RawPartitionRef] | None = None,
    factor_refs: Iterable[RawPartitionRef] | None = None,
    suspend_refs: Iterable[RawPartitionRef] | None = None,
    namechange_refs: Iterable[RawPartitionRef] | None = None,
    identity_tables: IdentityRepairTables | None = None,
    memory_sampler: Callable[[], int] | None = None,
    clock: Callable[[], float] = time.monotonic,
    sleeper: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    """Plan or directly repair the five current v2 datasets.

    ``apply=False`` is the default and never prepares full data.  ``--apply``
    updates shards/manifests inside the dataset IDs already named by active;
    the active pointer itself remains byte-for-byte unchanged.
    """

    guard = _MemoryGuard(
        sampler=memory_sampler,
        clock=clock,
        sleeper=sleeper,
    )
    try:
        resolved = _resolve_plan(
            workspace_root=workspace_root,
            identity_config_path=identity_config_path,
            start_date=start_date,
            calendar_refs=calendar_refs,
            stock_basic_refs=stock_basic_refs,
            daily_refs=daily_refs,
            factor_refs=factor_refs,
            suspend_refs=suspend_refs,
            namechange_refs=namechange_refs,
            identity_tables=identity_tables,
            guard=guard,
        )
    except (DirectRepairPaused, DuckDbMemoryFloorError) as exc:
        return _record_early_pause(workspace_root, reason=str(exc), apply=apply)

    plan = resolved.public
    run_id = str(plan["run_id"])
    state_path = _state_path(resolved.active.root, run_id)
    if not apply:
        payload = {
            **plan,
            "status": "dry_run",
            "apply": False,
            "active_manifest_unchanged": True,
            "state_path": str(state_path.resolve()),
        }
        _write_state(state_path, payload)
        return payload

    runtime_dir = state_path.parent
    stage = runtime_dir / f"prepared.{uuid.uuid4().hex}"
    explicit_factor = Path(factor_output_path).resolve() if factor_output_path else None
    if explicit_factor is not None:
        try:
            explicit_factor.relative_to(resolved.active.root.parent.resolve())
        except ValueError as exc:
            raise ValueError(
                f"factor_output_must_be_under_qdp_data_root:{resolved.active.root.parent}"
            ) from exc
    completed_domains: list[str] = []
    _write_state(
        state_path,
        {
            **plan,
            "status": "running",
            "apply": True,
            "stage": str(stage),
        },
    )
    try:
        guard.checkpoint()
        stage.mkdir(parents=True, exist_ok=False)
        calendar_prepared, calendar_inserted = _prepare_calendar_missing(
            resolved,
            stage / "trading_calendar",
            guard=guard,
        )
        if calendar_inserted != int(plan["domains"]["trading_calendar"]["missing_rows"]):
            raise DirectRepairError("calendar_source_changed_after_plan")
        (
            daily_prepared,
            daily_inserted,
            status_prepared,
            status_inserted,
        ) = _prepare_daily_status(
            resolved,
            stage / "market_daily_raw",
            stage / "security_status",
            guard=guard,
        )
        if daily_inserted != int(plan["domains"]["market_daily_raw"]["missing_rows"]):
            raise DirectRepairError("daily_source_changed_after_plan")

        universe_prepared, universe_inserted = _prepare_universe_missing(
            resolved,
            stage / "universe_snapshot",
            identities=resolved.universe_identities,
            open_dates=resolved.open_dates,
            guard=guard,
        )
        if universe_inserted != int(plan["domains"]["universe_snapshot"]["missing_rows"]):
            raise DirectRepairError("universe_source_changed_after_plan")

        if status_inserted != int(plan["domains"]["security_status"]["missing_rows"]):
            raise DirectRepairError("status_source_changed_after_plan")

        if explicit_factor is not None:
            if explicit_factor.exists():
                raise FileExistsError(f"factor_output_already_exists:{explicit_factor}")
            factor_path = explicit_factor
        else:
            factor_path = (
                stage
                / "adjust_factor"
                / "part_000000_adjust_factor_direct_repair.parquet"
            )
        active_daily_paths = _active_shard_paths(resolved.active, "market_daily_raw")
        final_daily_paths = [*active_daily_paths, *daily_prepared]
        factor_stats = write_adjust_factor_parquet(
            daily_paths=final_daily_paths,
            factor_batches=_factor_batches(resolved),
            output_path=factor_path,
            work_dir=stage / "factor_work",
            memory_guard=guard,
        )
        final_daily_rows = int(plan["domains"]["market_daily_raw"]["output_rows"])
        if int(factor_stats["row_count"]) != final_daily_rows:
            raise DirectRepairError("factor_row_count_not_equal_daily")

        validation_work = stage / "validation_work"
        _validate_primary_key(
            final_daily_paths,
            expected_rows=final_daily_rows,
            work_dir=validation_work / "market_daily_raw",
            memory_guard=guard,
        )
        _validate_primary_key(
            [
                *_active_shard_paths(resolved.active, "trading_calendar"),
                *calendar_prepared,
            ],
            expected_rows=int(plan["domains"]["trading_calendar"]["output_rows"]),
            primary_key=("trade_date",),
            work_dir=validation_work / "trading_calendar",
            memory_guard=guard,
        )
        final_universe_paths = [
            *_active_shard_paths(resolved.active, "universe_snapshot"),
            *universe_prepared,
        ]
        final_status_paths = [
            *_active_shard_paths(resolved.active, "security_status"),
            *status_prepared,
        ]
        _validate_primary_key(
            final_universe_paths,
            expected_rows=int(plan["domains"]["universe_snapshot"]["output_rows"]),
            work_dir=validation_work / "universe_snapshot",
            memory_guard=guard,
        )
        _validate_primary_key(
            final_status_paths,
            expected_rows=int(plan["domains"]["security_status"]["output_rows"]),
            work_dir=validation_work / "security_status",
            memory_guard=guard,
        )
        _validate_bidirectional_key_alignment(
            final_universe_paths,
            final_status_paths,
            left_label="universe_snapshot",
            right_label="security_status",
            work_dir=validation_work / "universe_status_alignment",
            memory_guard=guard,
        )
        _validate_primary_key(
            [factor_path],
            expected_rows=final_daily_rows,
            work_dir=validation_work / "adjust_factor",
            memory_guard=guard,
        )
        _validate_factor_alignment(
            final_daily_paths,
            factor_path,
            work_dir=validation_work / "factor_alignment",
            memory_guard=guard,
        )
        _assert_active_unchanged(resolved.active)

        factor_manifest = resolved.active.manifests["adjust_factor"]
        if len(factor_manifest.shards) != 1:
            raise DirectRepairError(
                f"active_adjust_factor_requires_one_shard:{len(factor_manifest.shards)}"
            )
        old_factor_path = resolve_manifest_path(
            factor_manifest.shards[0].path,
            root=resolved.active.root,
        )
        reason = f"trusted Tushare direct repair {plan['plan_id']}"
        commit_results: dict[str, Any] = {}
        commit_results["trading_calendar"] = _bulk_append_or_noop(
            "trading_calendar",
            calendar_prepared,
            reason=reason,
            workspace_root=workspace_root,
        )
        completed_domains.append("trading_calendar")
        _write_commit_progress(state_path, plan, completed_domains, commit_results)
        commit_results["market_daily_raw"] = _bulk_append_or_noop(
            "market_daily_raw",
            daily_prepared,
            reason=reason,
            workspace_root=workspace_root,
        )
        completed_domains.append("market_daily_raw")
        _write_commit_progress(state_path, plan, completed_domains, commit_results)
        commit_results["universe_snapshot"] = _bulk_append_or_noop(
            "universe_snapshot",
            universe_prepared,
            reason=reason,
            workspace_root=workspace_root,
        )
        completed_domains.append("universe_snapshot")
        _write_commit_progress(state_path, plan, completed_domains, commit_results)
        commit_results["security_status"] = _bulk_append_or_noop(
            "security_status",
            status_prepared,
            reason=reason,
            workspace_root=workspace_root,
        )
        completed_domains.append("security_status")
        _write_commit_progress(state_path, plan, completed_domains, commit_results)
        commit_results["adjust_factor"] = replace_active_shard_from_parquet(
            "adjust_factor",
            factor_path,
            old_factor_path,
            reason,
            workspace_root=workspace_root,
        )
        completed_domains.append("adjust_factor")
        _write_commit_progress(state_path, plan, completed_domains, commit_results)
        _assert_active_dataset_ids_unchanged(resolved.active)
        _assert_active_unchanged(resolved.active)
        manifest_paths = {
            domain: str(
                dataset_manifest_for_id(
                    resolved.active.root,
                    resolved.active.manifests[domain].dataset_id,
                    domain,
                ).resolve()
            )
            for domain in REPAIR_DOMAINS
        }
        result = {
            **plan,
            "status": "applied",
            "apply": True,
            "dataset_ids": dict(plan["active_dataset_ids"]),
            "manifest_paths": manifest_paths,
            "commit_results": commit_results,
            "completed_domains": list(completed_domains),
            "prepared_files_deleted": True,
            "factor_stats": factor_stats,
            "active_manifest_unchanged": True,
            "active_sha256_after": _sha256_bytes(resolved.active.active_path.read_bytes()),
            "completed_at": utc_now(),
        }
        _write_state(state_path, result)
        return result
    except (DirectRepairPaused, DuckDbMemoryFloorError) as exc:
        shutil.rmtree(stage, ignore_errors=True)
        if explicit_factor is not None:
            explicit_factor.unlink(missing_ok=True)
        payload = {
            **plan,
            "status": "paused_resource_guard",
            "apply": True,
            "reason": str(exc),
            "completed_domains": list(completed_domains),
            "prepared_files_deleted": True,
            "active_manifest_unchanged": _active_is_unchanged(resolved.active),
        }
        _write_state(state_path, payload)
        return payload
    except BaseException as exc:
        shutil.rmtree(stage, ignore_errors=True)
        if explicit_factor is not None:
            explicit_factor.unlink(missing_ok=True)
        _write_state(
            state_path,
            {
                **plan,
                "status": (
                    "partial_applied_recoverable"
                    if completed_domains
                    else "failed_recoverable"
                ),
                "apply": True,
                "completed_domains": list(completed_domains),
                "resume_required": True,
                "prepared_files_deleted": True,
                "error_type": type(exc).__name__,
                "error": _sanitized_error(exc),
                "active_manifest_unchanged": _active_is_unchanged(resolved.active),
            },
        )
        raise
    finally:
        shutil.rmtree(stage, ignore_errors=True)
        if explicit_factor is not None:
            explicit_factor.unlink(missing_ok=True)
        _assert_active_unchanged(resolved.active)


def write_adjust_factor_parquet(
    *,
    daily_paths: Sequence[str | Path],
    factor_batches: Iterable[RepairBatch],
    output_path: str | Path,
    work_dir: str | Path,
    memory_guard: _MemoryGuard | None = None,
) -> dict[str, Any]:
    """Write exactly one dense factor Parquet with one ``ParquetWriter``.

    Factor events are staged in a spillable DuckDB table, then an ASOF join is
    streamed to PyArrow.  This keeps the full symbol-day panel out of pandas.
    """

    paths = [str(Path(item).resolve()) for item in daily_paths]
    if not paths:
        raise ValueError("daily_paths_required")
    output = Path(output_path).resolve()
    if output.exists():
        raise FileExistsError(f"factor_output_already_exists:{output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    work = Path(work_dir).resolve()
    work.mkdir(parents=True, exist_ok=True)
    database = work / "factor_events.duckdb"
    guard = memory_guard or _MemoryGuard()
    schema = pa.schema(
        [
            pa.field("symbol", pa.string()),
            pa.field("trade_date", pa.string()),
            pa.field("fore_adjust_factor", pa.float64()),
            pa.field("back_adjust_factor", pa.float64()),
            pa.field("adjust_factor", pa.float64()),
            pa.field("factor_provider", pa.string()),
            pa.field("factor_semantics", pa.string()),
            pa.field("source", pa.string()),
            pa.field("factor_source_date", pa.string()),
            pa.field("ffill_days", pa.int64()),
        ]
    )
    event_rows = 0
    security_count = 0
    writer: pq.ParquetWriter | None = None
    row_count = 0
    con = _open_direct_repair_duckdb(
        str(database),
        temp_directory=work / "spill",
    )
    try:
        con.execute(
            """
            create table factor_events(
              symbol varchar,
              trade_date varchar,
              fore_adjust_factor double,
              back_adjust_factor double,
              adjust_factor double,
              factor_source_date varchar
            )
            """
        )
        for batch in factor_batches:
            guard.checkpoint()
            frame = batch.rows.loc[
                :,
                [
                    "symbol",
                    "trade_date",
                    "fore_adjust_factor",
                    "back_adjust_factor",
                    "adjust_factor",
                    "factor_source_date",
                ],
            ].copy()
            if frame.empty:
                continue
            con.register("factor_batch", frame)
            try:
                con.execute("insert into factor_events select * from factor_batch")
            finally:
                con.unregister("factor_batch")
            event_rows += len(frame)
            security_count += int(frame["symbol"].nunique())
        guard.checkpoint()
        duplicate = con.execute(
            """
            select symbol, trade_date
            from factor_events
            group by symbol, trade_date
            having count(*) > 1
            limit 1
            """
        ).fetchone()
        guard.checkpoint()
        if duplicate:
            raise DirectRepairError(f"factor_event_duplicate_key:{duplicate}")
        guard.checkpoint()
        missing_symbol = con.execute(
            """
            select d.symbol
            from (select distinct symbol from read_parquet(?)) d
            left join (select distinct symbol from factor_events) f using(symbol)
            where f.symbol is null
            limit 1
            """,
            [paths],
        ).fetchone()
        guard.checkpoint()
        if missing_symbol:
            raise DirectRepairError(
                f"tushare_factor_security_missing:{missing_symbol[0]}"
            )
        guard.checkpoint()
        late_first_factor = con.execute(
            """
            with first_daily as (
              select symbol, min(cast(trade_date as date)) as first_daily_date
              from read_parquet(?)
              group by symbol
            ), first_factor as (
              select symbol, min(cast(trade_date as date)) as first_factor_date
              from factor_events
              group by symbol
            )
            select
              d.symbol,
              cast(d.first_daily_date as varchar),
              coalesce(cast(f.first_factor_date as varchar), '')
            from first_daily d
            left join first_factor f using(symbol)
            where f.first_factor_date is null
               or f.first_factor_date > d.first_daily_date
            order by d.symbol
            limit 1
            """,
            [paths],
        ).fetchone()
        guard.checkpoint()
        if late_first_factor:
            raise DirectRepairError(
                "tushare_factor_first_date_not_covered:"
                f"symbol={late_first_factor[0]}:"
                f"first_daily_date={late_first_factor[1]}:"
                f"first_factor_date={late_first_factor[2]}"
            )
        guard.checkpoint()
        query = con.execute(
            """
            select
              cast(d.symbol as varchar) as symbol,
              cast(d.trade_date as varchar) as trade_date,
              cast(f.fore_adjust_factor as double) as fore_adjust_factor,
              cast(f.back_adjust_factor as double) as back_adjust_factor,
              cast(f.adjust_factor as double) as adjust_factor,
              'tushare_proxy'::varchar as factor_provider,
              'trusted_source_first_2010_observation_normalized_to_1_asof'::varchar as factor_semantics,
              'tushare_proxy.adj_factor+direct_repair_asof'::varchar as source,
              cast(f.factor_source_date as varchar) as factor_source_date,
              cast(
                date_diff('day', cast(f.factor_source_date as date), cast(d.trade_date as date))
                as bigint
              ) as ffill_days
            from read_parquet(?) d
            asof left join factor_events f
              on d.symbol = f.symbol and d.trade_date >= f.trade_date
            order by d.symbol, d.trade_date
            """,
            [paths],
        )
        reader = query.to_arrow_reader(batch_size=131_072)
        writer = pq.ParquetWriter(
            output,
            schema,
            compression="zstd",
            use_dictionary=True,
        )
        for batch in reader:
            guard.checkpoint()
            table = pa.Table.from_batches([batch]).cast(schema, safe=True)
            writer.write_table(table, row_group_size=131_072)
            row_count += table.num_rows
        guard.checkpoint()
        writer.close()
        writer = None
    except BaseException:
        if writer is not None:
            writer.close()
        output.unlink(missing_ok=True)
        raise
    finally:
        con.close()
    if row_count <= 0:
        output.unlink(missing_ok=True)
        raise DirectRepairError("factor_output_empty")
    footer_rows = int(pq.ParquetFile(output).metadata.num_rows)
    if footer_rows != row_count:
        output.unlink(missing_ok=True)
        raise DirectRepairError("factor_footer_row_count_mismatch")
    return {
        "row_count": row_count,
        "event_rows": event_rows,
        "security_count": security_count,
        "file_count": 1,
        "sha256": _sha256_file(output),
    }


def _resolve_plan(
    *,
    workspace_root: str | Path | None,
    identity_config_path: str | Path | None,
    start_date: str,
    calendar_refs: Iterable[RawPartitionRef] | None,
    stock_basic_refs: Iterable[RawPartitionRef] | None,
    daily_refs: Iterable[RawPartitionRef] | None,
    factor_refs: Iterable[RawPartitionRef] | None,
    suspend_refs: Iterable[RawPartitionRef] | None,
    namechange_refs: Iterable[RawPartitionRef] | None,
    identity_tables: IdentityRepairTables | None,
    guard: _MemoryGuard,
) -> _ResolvedPlan:
    active = _active_inputs(workspace_root)
    guard.checkpoint()
    calendars = _materialize_refs(
        RAW_TUSHARE_PROXY_TRADE_CALENDAR,
        workspace_root,
        calendar_refs,
    )
    stock = _materialize_refs(
        RAW_TUSHARE_PROXY_STOCK_BASIC,
        workspace_root,
        stock_basic_refs,
    )
    daily = _materialize_refs(
        RAW_TUSHARE_PROXY_DAILY,
        workspace_root,
        daily_refs,
    )
    factors = _materialize_refs(
        RAW_TUSHARE_PROXY_ADJ_FACTOR,
        workspace_root,
        factor_refs,
    )
    suspends = _materialize_refs(
        RAW_TUSHARE_PROXY_SUSPEND,
        workspace_root,
        suspend_refs,
    )
    names = _materialize_refs(
        RAW_TUSHARE_PROXY_NAMECHANGE,
        workspace_root,
        namechange_refs,
    )
    if not calendars:
        raise FileNotFoundError("tushare_trade_calendar_raw_missing")
    if not daily:
        raise FileNotFoundError("tushare_daily_raw_missing")
    if not factors:
        raise FileNotFoundError("tushare_adjust_factor_raw_missing")
    identities = identity_tables or build_mainboard_identity_tables(
        workspace_root=workspace_root,
        stock_basic_refs=stock,
        config_path=identity_config_path,
    )
    universe_identities = _universe_identities(identities, start_date=start_date)
    if not universe_identities:
        raise DirectRepairError("mainboard_identity_set_empty")

    calendar_missing, ordered_dates, calendar_summary = _calendar_repair_rows(
        active,
        calendar_refs=calendars,
        start_date=start_date,
        guard=guard,
    )
    if not ordered_dates:
        raise DirectRepairError("open_trading_dates_empty")

    daily_summary: list[dict[str, Any]] = []
    status_summary: list[dict[str, Any]] = []
    missing_daily = 0
    missing_status = 0
    for daily_batch, status_batch in _iter_daily_status_batches(
        active=active,
        identities=identities,
        universe_identities=universe_identities,
        open_dates=ordered_dates,
        daily_refs=daily,
        suspend_refs=suspends,
        namechange_refs=names,
        start_date=start_date,
        guard=guard,
    ):
        missing_daily += int(daily_batch.insert_count)
        missing_status += int(status_batch.insert_count)
        daily_summary.append(_batch_summary(daily_batch))
        status_summary.append(_batch_summary(status_batch))

    # Planning uses catalog metadata only.  Reading/normalizing the full dense
    # factor history here and again during --apply would double the dominant
    # factor I/O without changing the planned output row count.
    mapped_factor_refs = [
        ref
        for ref in factors
        if identities.registry.security_id_for_provider_symbol(ref.partition_value)
    ]
    factor_source_rows = sum(int(ref.row_count) for ref in mapped_factor_refs)
    factor_security_count = len(
        {
            identities.registry.security_id_for_provider_symbol(ref.partition_value)
            for ref in mapped_factor_refs
        }
    )
    factor_summary = {
        "partition_count": len(mapped_factor_refs),
        "row_count": factor_source_rows,
        "content_signature": stable_hash(
            {
                "items": [
                    (ref.partition_value, ref.content_sha256, int(ref.row_count))
                    for ref in mapped_factor_refs
                ]
            },
            length=32,
        ),
    }

    target_universe_rows = _count_universe_rows(universe_identities, ordered_dates)
    missing_universe, universe_summary = _count_missing_universe(
        active,
        identities=universe_identities,
        open_dates=ordered_dates,
        guard=guard,
    )
    final_daily_rows = int(active.manifests["market_daily_raw"].row_count) + missing_daily
    final_calendar_rows = (
        int(active.manifests["trading_calendar"].row_count) + len(calendar_missing)
    )
    final_status_rows = int(active.manifests["security_status"].row_count) + missing_status
    final_universe_rows = int(active.manifests["universe_snapshot"].row_count) + missing_universe
    source_signature = {
        "raw_refs": _refs_signature(
            (*calendars, *stock, *daily, *factors, *suspends, *names)
        ),
        "calendar": calendar_summary,
        "daily_batches": daily_summary,
        "status_batches": status_summary,
        "universe_missing": universe_summary,
        "factor_batches": factor_summary,
        "identity": _identity_signature(identities),
        "open_dates": {
            "count": len(ordered_dates),
            "start": ordered_dates[0],
            "end": ordered_dates[-1],
        },
    }
    plan_hash = stable_hash(
        {
            "version": DIRECT_REPAIR_VERSION,
            "active_sha256": active.active_sha256,
            "active_datasets": {
                key: active.manifests[key].dataset_id
                for key in REPAIR_DOMAINS
            },
            "start_date": start_date,
            "source": source_signature,
        },
        length=24,
    )
    run_id = f"direct_repair__{plan_hash}"
    dataset_ids = {
        domain: active.manifests[domain].dataset_id
        for domain in REPAIR_DOMAINS
    }
    public = {
        "status": "planned",
        "plan_version": DIRECT_REPAIR_VERSION,
        "plan_id": plan_hash,
        "run_id": run_id,
        "qdp_v2_root": str(active.root.resolve()),
        "active_sha256_before": active.active_sha256,
        "active_dataset_ids": {
            domain: active.manifests[domain].dataset_id
            for domain in REPAIR_DOMAINS
        },
        "dataset_ids": dataset_ids,
        "start_date": start_date,
        "domains": {
            "trading_calendar": {
                "merge_policy": "missing_only_preserve_all_old_dates",
                "active_rows": int(active.manifests["trading_calendar"].row_count),
                "source_rows": int(calendar_summary["source_rows"]),
                "missing_rows": len(calendar_missing),
                "output_rows": final_calendar_rows,
                "open_date_count": len(ordered_dates),
            },
            "market_daily_raw": {
                "merge_policy": "missing_only_preserve_all_old_keys",
                "active_rows": int(active.manifests["market_daily_raw"].row_count),
                "missing_rows": missing_daily,
                "output_rows": final_daily_rows,
            },
            "universe_snapshot": {
                "merge_policy": "missing_only_identity_x_open_date_with_valid_intervals",
                "active_rows": int(active.manifests["universe_snapshot"].row_count),
                "identity_count": len(universe_identities),
                "open_date_count": len(ordered_dates),
                "target_identity_rows": target_universe_rows,
                "missing_rows": missing_universe,
                "output_rows": final_universe_rows,
            },
            "security_status": {
                "merge_policy": "missing_only_preserve_all_old_keys",
                "active_rows": int(active.manifests["security_status"].row_count),
                "missing_rows": missing_status,
                "output_rows": final_status_rows,
            },
            "adjust_factor": {
                "merge_policy": "full_tushare_rebuild_asof_aligned_to_final_daily",
                "active_rows": int(active.manifests["adjust_factor"].row_count),
                "source_rows": factor_source_rows,
                "source_security_count": factor_security_count,
                "output_rows": final_daily_rows,
                "single_parquet_file": True,
            },
        },
        "source_signature": source_signature,
        "created_at": utc_now(),
    }
    return _ResolvedPlan(
        public=public,
        active=active,
        identities=identities,
        universe_identities=universe_identities,
        open_dates=ordered_dates,
        calendar_refs=calendars,
        stock_basic_refs=stock,
        daily_refs=daily,
        factor_refs=factors,
        suspend_refs=suspends,
        namechange_refs=names,
    )


def _prepare_calendar_missing(
    resolved: _ResolvedPlan,
    directory: Path,
    *,
    guard: _MemoryGuard,
) -> tuple[list[Path], int]:
    missing, open_dates, _ = _calendar_repair_rows(
        resolved.active,
        calendar_refs=resolved.calendar_refs,
        start_date=str(resolved.public["start_date"]),
        guard=guard,
    )
    if tuple(open_dates) != resolved.open_dates:
        raise DirectRepairError("calendar_open_dates_changed_after_plan")
    if missing.empty:
        return [], 0
    schema = pq.ParquetFile(
        _active_shard_paths(resolved.active, "trading_calendar")[0]
    ).schema_arrow
    path = directory / "part_trading_calendar_missing.parquet"
    _write_frame(path, missing.loc[:, V2_TRADING_CALENDAR_COLUMNS], schema=schema)
    return [path], len(missing)


def _prepare_daily_status(
    resolved: _ResolvedPlan,
    daily_directory: Path,
    status_directory: Path,
    *,
    guard: _MemoryGuard,
) -> tuple[list[Path], int, list[Path], int]:
    daily_directory.mkdir(parents=True, exist_ok=True)
    status_directory.mkdir(parents=True, exist_ok=True)
    daily_schema = pq.ParquetFile(
        _active_shard_paths(resolved.active, "market_daily_raw")[0]
    ).schema_arrow
    status_schema = pq.ParquetFile(
        _active_shard_paths(resolved.active, "security_status")[0]
    ).schema_arrow
    daily_paths: list[Path] = []
    status_paths: list[Path] = []
    daily_inserted = 0
    status_inserted = 0
    for index, (daily_batch, status_batch) in enumerate(
        _iter_daily_status_batches(
            active=resolved.active,
            identities=resolved.identities,
            universe_identities=resolved.universe_identities,
            open_dates=resolved.open_dates,
            daily_refs=resolved.daily_refs,
            suspend_refs=resolved.suspend_refs,
            namechange_refs=resolved.namechange_refs,
            start_date=str(resolved.public["start_date"]),
            guard=guard,
        )
    ):
        guard.checkpoint()
        if not daily_batch.rows.empty:
            path = daily_directory / f"repair_{index:04d}_{daily_batch.partition_key}.parquet"
            _write_frame(path, daily_batch.rows, schema=daily_schema)
            daily_paths.append(path)
            daily_inserted += int(daily_batch.insert_count)
        if not status_batch.rows.empty:
            path = status_directory / f"repair_{index:04d}_{status_batch.partition_key}.parquet"
            _write_frame(path, status_batch.rows, schema=status_schema)
            status_paths.append(path)
            status_inserted += int(status_batch.insert_count)
    return daily_paths, daily_inserted, status_paths, status_inserted


def _prepare_universe_missing(
    resolved: _ResolvedPlan,
    directory: Path,
    *,
    identities: Sequence[_UniverseIdentity],
    open_dates: Sequence[str],
    guard: _MemoryGuard,
) -> tuple[list[Path], int]:
    schema = pa.schema(
        [
            pa.field("symbol", pa.string()),
            pa.field("trade_date", pa.string()),
            pa.field("name", pa.string()),
            pa.field("exchange", pa.string()),
            pa.field("board", pa.string()),
            pa.field("list_status", pa.string()),
            pa.field("list_date", pa.string()),
            pa.field("delist_date", pa.string()),
            pa.field("source", pa.string()),
        ]
    )
    by_year: dict[str, list[str]] = {}
    for date in open_dates:
        by_year.setdefault(date[:4], []).append(date)
    directory.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    inserted = 0
    existing_loader = _year_loader(
        resolved.active,
        "universe_snapshot",
        ("trade_date", "symbol"),
    )
    for year, dates in sorted(by_year.items()):
        path = directory / f"part_{year}_universe_snapshot_missing.parquet"
        writer = pq.ParquetWriter(path, schema, compression="zstd", use_dictionary=True)
        year_rows = 0
        existing = existing_loader(year)
        try:
            for frame in _iter_universe_frames(identities, dates):
                guard.checkpoint()
                missing = _missing_only(frame, existing)
                if missing.empty:
                    continue
                table = pa.Table.from_pandas(missing, preserve_index=False).cast(schema, safe=True)
                writer.write_table(table, row_group_size=131_072)
                year_rows += table.num_rows
        finally:
            writer.close()
        if year_rows:
            paths.append(path)
            inserted += year_rows
        else:
            path.unlink(missing_ok=True)
    return paths, inserted


def _factor_batches(resolved: _ResolvedPlan) -> Iterator[RepairBatch]:
    yield from iter_adjust_factor_repairs_by_security(
        identity_registry=resolved.identities.registry,
        factor_refs=resolved.factor_refs,
        start_date=str(resolved.public["start_date"]),
    )


def _active_inputs(workspace_root: str | Path | None) -> _ActiveInputs:
    root = qdp_v2_root(workspace_root).resolve()
    active_path = root / "active" / "active.json"
    active_bytes = active_path.read_bytes()
    active = read_active_manifest(root)
    mapping = active_dataset_map(active)
    manifests: dict[str, DatasetManifest] = {}
    for domain in REPAIR_DOMAINS:
        dataset_id = str(mapping.get(domain, "") or "")
        if not dataset_id:
            raise DirectRepairError(f"active_domain_missing:{domain}")
        path = dataset_manifest_for_id(root, dataset_id, domain)
        if path is None:
            raise FileNotFoundError(f"active_dataset_manifest_missing:{domain}:{dataset_id}")
        manifest = read_dataset_manifest(path)
        if manifest.domain != domain or manifest.dataset_id != dataset_id:
            raise DirectRepairError(f"active_manifest_identity_mismatch:{domain}")
        manifests[domain] = manifest
    return _ActiveInputs(
        root=root,
        active_path=active_path,
        active_bytes=active_bytes,
        active_sha256=_sha256_bytes(active_bytes),
        active=active,
        manifests=manifests,
    )


def _materialize_refs(
    domain: str,
    workspace_root: str | Path | None,
    supplied: Iterable[RawPartitionRef] | None,
) -> tuple[RawPartitionRef, ...]:
    if supplied is not None:
        refs = list(supplied)
    else:
        # Direct repair is a self-contained H/data operation.  Do not inherit a
        # desktop bootstrap runtime pointing at C:, because that would make a
        # repair silently depend on an unrelated SQLite catalog.
        runtime = qdp_v2_root(workspace_root).parent / "qdp_runtime"
        with _temporary_environment("QDP_RUNTIME_ROOT", str(runtime.resolve())):
            refs = iter_raw_partitions(domain, workspace_root=workspace_root)
    return tuple(
        sorted(
            refs,
            key=lambda item: (
                item.partition_field,
                item.partition_value,
                item.content_sha256,
            ),
        )
    )


@contextlib.contextmanager
def _temporary_environment(key: str, value: str) -> Iterator[None]:
    previous = os.environ.get(key)
    os.environ[key] = value
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = previous


def _year_loader(
    active: _ActiveInputs,
    domain: str,
    columns: Sequence[str],
) -> Callable[[str], pd.DataFrame]:
    manifest = active.manifests[domain]
    paths = [str(resolve_manifest_path(item.path, root=active.root)) for item in manifest.shards]
    dataset = pads.dataset(paths, format="parquet")
    missing = sorted(set(columns).difference(dataset.schema.names))
    if missing:
        raise DirectRepairError(f"active_{domain}_schema_missing:{missing}")

    def load(year: str) -> pd.DataFrame:
        start = f"{year}-01-01"
        end = f"{year}-12-31"
        table = dataset.to_table(
            columns=list(columns),
            filter=(pads.field("trade_date") >= start) & (pads.field("trade_date") <= end),
        )
        return table.to_pandas()

    return load


def _iter_daily_status_batches(
    *,
    active: _ActiveInputs,
    identities: IdentityRepairTables,
    universe_identities: Sequence[_UniverseIdentity],
    open_dates: Sequence[str],
    daily_refs: Sequence[RawPartitionRef],
    suspend_refs: Sequence[RawPartitionRef],
    namechange_refs: Sequence[RawPartitionRef],
    start_date: str,
    guard: _MemoryGuard,
) -> Iterator[tuple[RepairBatch, RepairBatch]]:
    """Read each compact daily year once and derive both missing-only domains."""

    by_year = _trade_refs_by_year(daily_refs)
    suspend_by_year = _trade_refs_by_year(suspend_refs)
    daily_loader = _year_loader(active, "market_daily_raw", V2_MARKET_DAILY_COLUMNS)
    status_loader = _year_loader(active, "security_status", V2_SECURITY_STATUS_COLUMNS)
    name_intervals = _trusted_name_intervals(
        namechange_refs,
        identities=identities,
        guard=guard,
    )
    open_dates_by_year: dict[str, list[str]] = {}
    for date in open_dates:
        open_dates_by_year.setdefault(str(date)[:4], []).append(str(date))
    for year in sorted(open_dates_by_year):
        guard.checkpoint()
        year_daily_refs = by_year.get(year, ())
        if year_daily_refs:
            raw = _read_refs_bulk(year_daily_refs)
            incoming_daily = tushare_daily_to_v2(
                raw,
                identity_registry=identities.registry,
                start_date=start_date,
            )
        else:
            incoming_daily = pd.DataFrame(columns=V2_MARKET_DAILY_COLUMNS)
        existing_daily = daily_loader(year)
        daily_missing = _missing_only(incoming_daily, existing_daily)
        daily_batch = RepairBatch(
            partition_key=year,
            rows=daily_missing.loc[:, V2_MARKET_DAILY_COLUMNS],
            insert_count=len(daily_missing),
            update_count=0,
            source_partition_count=len(year_daily_refs),
        )

        suspend_raw = (
            _read_refs_bulk(suspend_by_year.get(year, ()))
            if suspend_by_year.get(year)
            else pd.DataFrame()
        )
        incoming_status = _status_from_trusted_year(
            incoming_daily,
            universe_identities=universe_identities,
            open_dates=open_dates_by_year[year],
            suspend_raw=suspend_raw,
            name_intervals=name_intervals,
            identities=identities,
            start_date=start_date,
        )
        existing_status = status_loader(year)
        status_missing = _missing_only(incoming_status, existing_status)
        status_batch = RepairBatch(
            partition_key=year,
            rows=status_missing.loc[:, V2_SECURITY_STATUS_COLUMNS],
            insert_count=len(status_missing),
            update_count=0,
            source_partition_count=(
                len(year_daily_refs)
                + len(suspend_by_year.get(year, ()))
                + len(namechange_refs)
            ),
        )
        yield daily_batch, status_batch


def _trade_refs_by_year(
    refs: Sequence[RawPartitionRef],
) -> dict[str, tuple[RawPartitionRef, ...]]:
    grouped: dict[str, list[RawPartitionRef]] = {}
    for ref in refs:
        if ref.partition_field != "trade_date":
            continue
        value = str(ref.partition_value)
        if len(value) < 4 or not value[:4].isdigit():
            continue
        grouped.setdefault(value[:4], []).append(ref)
    return {year: tuple(values) for year, values in grouped.items()}


def _read_refs_bulk(refs: Sequence[RawPartitionRef]) -> pd.DataFrame:
    """Read compacted bundle row groups once per physical Parquet file."""

    if not refs:
        return pd.DataFrame()
    frames: list[pd.DataFrame] = []
    bundle_groups: dict[Path, set[int]] = {}
    legacy: list[RawPartitionRef] = []
    for ref in refs:
        if ref.storage_kind != "bundle":
            legacy.append(ref)
            continue
        for segment in ref.bundle_segments:
            bundle_groups.setdefault(segment.bundle_path.resolve(), set()).add(
                int(segment.row_group)
            )
    for path, row_groups in sorted(bundle_groups.items(), key=lambda item: str(item[0])):
        file = pq.ParquetFile(path)
        table = file.read_row_groups(sorted(row_groups))
        frame = table.to_pandas()
        internal = [column for column in frame.columns if str(column).startswith("__qdp_")]
        if internal:
            frame = frame.drop(columns=internal)
        frames.append(frame)
    frames.extend(read_raw_partition(ref) for ref in legacy)
    if not frames:
        return pd.DataFrame()
    result = pd.concat(frames, ignore_index=True, sort=False)
    expected_rows = sum(int(ref.row_count) for ref in refs)
    if len(result) != expected_rows:
        raise DirectRepairError(
            f"bulk_raw_row_count_mismatch:expected={expected_rows}:actual={len(result)}"
        )
    return result


def _missing_only(
    incoming: pd.DataFrame,
    existing: pd.DataFrame,
    *,
    keys: Sequence[str] = ("trade_date", "symbol"),
) -> pd.DataFrame:
    keys = list(keys)
    if incoming.empty:
        return incoming.copy()
    if incoming.duplicated(keys).any():
        raise DirectRepairError("trusted_source_duplicate_key")
    if existing.duplicated(keys).any():
        raise DirectRepairError("active_source_duplicate_key")
    old_keys = pd.MultiIndex.from_frame(existing.loc[:, keys])
    incoming_keys = pd.MultiIndex.from_frame(incoming.loc[:, keys])
    return incoming.loc[~incoming_keys.isin(old_keys)].reset_index(drop=True)


def _trusted_name_intervals(
    refs: Sequence[RawPartitionRef],
    *,
    identities: IdentityRepairTables,
    guard: _MemoryGuard,
) -> dict[str, list[tuple[str, str, bool]]]:
    if not refs:
        return {}
    guard.checkpoint()
    raw = _read_refs_bulk(refs)
    required = {"ts_code", "name", "start_date"}
    if not required.issubset(raw.columns):
        raise DirectRepairError(
            f"tushare_namechange_fields_missing:{sorted(required - set(raw.columns))}"
        )
    starts = _iso_dates(raw["start_date"])
    ends = _iso_dates(
        raw["end_date"] if "end_date" in raw.columns else pd.Series("", index=raw.index)
    )
    grouped: dict[str, list[tuple[str, str, bool]]] = {}
    for index, symbol in raw["ts_code"].map(normalize_symbol).items():
        security_id = identities.registry.security_id_for_provider_symbol(symbol)
        start = str(starts.loc[index])
        if not security_id or not start:
            continue
        end = str(ends.loc[index]) or "9999-12-31"
        is_st = "ST" in str(raw.loc[index, "name"] or "").upper()
        grouped.setdefault(security_id, []).append((start, end, is_st))
    return {key: sorted(values) for key, values in grouped.items()}


def _status_from_trusted_year(
    daily: pd.DataFrame,
    *,
    universe_identities: Sequence[_UniverseIdentity],
    open_dates: Sequence[str],
    suspend_raw: pd.DataFrame,
    name_intervals: dict[str, list[tuple[str, str, bool]]],
    identities: IdentityRepairTables,
    start_date: str,
) -> pd.DataFrame:
    stable = {
        str(row.security_id): str(row.current_symbol)
        for row in identities.security_identity.itertuples(index=False)
    }
    skeleton_frames: list[pd.DataFrame] = []
    security_by_symbol = {
        identity.symbol: identity.security_id for identity in universe_identities
    }
    for frame in _iter_universe_frames(universe_identities, open_dates):
        skeleton = frame.loc[:, ["symbol", "trade_date", "list_status"]].copy()
        skeleton["security_id"] = skeleton["symbol"].map(security_by_symbol).fillna("")
        skeleton["is_delisted"] = skeleton["list_status"].eq("D")
        skeleton_frames.append(
            skeleton.loc[
                skeleton["security_id"].ne(""),
                ["security_id", "symbol", "trade_date", "is_delisted"],
            ]
        )
    if not skeleton_frames:
        return pd.DataFrame(columns=V2_SECURITY_STATUS_COLUMNS)
    keys = pd.concat(skeleton_frames, ignore_index=True)
    if keys.duplicated(["trade_date", "symbol"]).any():
        raise DirectRepairError("trusted_universe_status_skeleton_duplicate_key")

    daily_keys = daily.loc[:, ["symbol", "trade_date"]].copy()
    daily_keys["security_id"] = daily_keys["symbol"].map(
        identities.registry.security_id_for_provider_symbol
    )
    daily_keys["has_bar"] = True
    daily_keys = daily_keys.loc[
        daily_keys["security_id"].notna(),
        ["security_id", "symbol", "trade_date", "has_bar"],
    ]
    suspended = pd.DataFrame(
        columns=["security_id", "symbol", "trade_date", "is_suspended"]
    )
    if not suspend_raw.empty:
        required = {"ts_code", "trade_date"}
        if not required.issubset(suspend_raw.columns):
            raise DirectRepairError(
                f"tushare_suspend_fields_missing:{sorted(required - set(suspend_raw.columns))}"
            )
        base = pd.DataFrame(
            {
                "provider_symbol": suspend_raw["ts_code"].map(normalize_symbol),
                "trade_date": _iso_dates(suspend_raw["trade_date"]),
            }
        )
        mapped = identities.registry.map_frame(base)
        mapped = mapped.loc[
            mapped["identity_mapping_status"].eq("mapped")
            & mapped["board_on_date"].eq("MainBoard")
            & mapped["trade_date"].ge(start_date)
        ].copy()
        mapped["symbol"] = mapped["security_id"].map(stable).fillna("")
        suspended = mapped.loc[
            mapped["symbol"].ne(""),
            ["security_id", "symbol", "trade_date"],
        ].drop_duplicates(["trade_date", "symbol"])
        suspended["is_suspended"] = True
    keys = keys.merge(
        daily_keys,
        on=["security_id", "symbol", "trade_date"],
        how="left",
        sort=False,
    ).merge(
        suspended,
        on=["security_id", "symbol", "trade_date"],
        how="left",
        sort=False,
    )
    keys["has_bar"] = keys["has_bar"].eq(True)
    keys["is_suspended"] = keys["is_suspended"].eq(True)
    keys["is_st"] = False
    for security_id, indices in keys.groupby("security_id", sort=False).groups.items():
        dates = keys.loc[indices, "trade_date"]
        flags = pd.Series(False, index=indices)
        for begin, end, is_st in name_intervals.get(str(security_id), ()): 
            if is_st:
                flags |= dates.between(begin, end)
        keys.loc[indices, "is_st"] = flags
    keys["status_reason"] = np.select(
        [keys["is_st"], keys["is_suspended"], ~keys["has_bar"]],
        ["st", "suspended", "missing_or_invalid_bar"],
        default="tradeable",
    )
    keys["source"] = (
        "tushare_proxy.stock_basic+symbol_history+trade_cal+daily+namechange+suspend_d"
    )
    return keys.loc[:, V2_SECURITY_STATUS_COLUMNS].sort_values(
        ["trade_date", "symbol"],
        kind="mergesort",
    ).reset_index(drop=True)


def _iso_dates(values: pd.Series) -> pd.Series:
    raw = values.fillna("").astype(str).str.strip().str.replace("-", "", regex=False)
    valid = raw.str.fullmatch(r"\d{8}")
    result = pd.Series("", index=values.index, dtype="string")
    result.loc[valid] = (
        raw.loc[valid].str.slice(0, 4)
        + "-"
        + raw.loc[valid].str.slice(4, 6)
        + "-"
        + raw.loc[valid].str.slice(6, 8)
    )
    return result.astype(str)


def _calendar_repair_rows(
    active: _ActiveInputs,
    *,
    calendar_refs: Sequence[RawPartitionRef],
    start_date: str,
    guard: _MemoryGuard,
) -> tuple[pd.DataFrame, tuple[str, ...], dict[str, Any]]:
    """Return missing calendar rows and the final trusted open-date axis."""

    guard.checkpoint()
    raw = _read_refs_bulk(calendar_refs)
    incoming = tushare_trade_calendar_to_v2(raw, start_date=start_date)
    if incoming.empty:
        raise DirectRepairError("tushare_trade_calendar_empty")
    if str(incoming["trade_date"].min()) > str(start_date):
        raise DirectRepairError(
            "tushare_trade_calendar_start_not_covered:"
            f"expected={start_date}:actual={incoming['trade_date'].min()}"
        )

    manifest = active.manifests["trading_calendar"]
    paths = [str(resolve_manifest_path(item.path, root=active.root)) for item in manifest.shards]
    dataset = pads.dataset(paths, format="parquet")
    missing_columns = sorted(
        set(V2_TRADING_CALENDAR_COLUMNS).difference(dataset.schema.names)
    )
    if missing_columns:
        raise DirectRepairError(
            f"active_trading_calendar_schema_missing:{missing_columns}"
        )
    existing = dataset.to_table(columns=list(V2_TRADING_CALENDAR_COLUMNS)).to_pandas()
    if existing.duplicated("trade_date").any():
        raise DirectRepairError("active_trading_calendar_duplicate_key")
    missing = _missing_only(incoming, existing, keys=("trade_date",))
    final = pd.concat([existing, missing], ignore_index=True, sort=False)
    open_dates = tuple(
        sorted(
            set(
                final.loc[
                    final["is_open"].eq(True)
                    & final["trade_date"].astype(str).ge(start_date),
                    "trade_date",
                ].astype(str)
            )
        )
    )
    summary = {
        "source_partition_count": len(calendar_refs),
        "source_rows": len(incoming),
        "active_rows": len(existing),
        "missing_rows": len(missing),
        "open_date_count": len(open_dates),
        "source_start_date": str(incoming["trade_date"].min()),
        "source_end_date": str(incoming["trade_date"].max()),
    }
    return missing, open_dates, summary


def _universe_identities(
    tables: IdentityRepairTables,
    *,
    start_date: str,
) -> tuple[_UniverseIdentity, ...]:
    identity_by_id = {
        str(item.security_id): item
        for item in tables.security_identity.itertuples(index=False)
    }
    results: list[_UniverseIdentity] = []
    for security_id, history in tables.symbol_history.groupby("security_id", sort=True):
        identity = identity_by_id.get(str(security_id))
        if identity is None:
            continue
        intervals: list[tuple[str, str]] = []
        for row in history.sort_values(["effective_from", "effective_to"]).itertuples(index=False):
            if str(row.board_on_date) != "MainBoard":
                continue
            begin = max(start_date, str(row.effective_from or start_date))
            end = str(row.effective_to or "9999-12-31")
            if begin <= end:
                intervals.append((begin, end))
        merged = _merge_intervals(intervals)
        if not merged:
            continue
        finite_ends = [end for _, end in merged if end < "9999-12-31"]
        open_ended = any(end >= "9999-12-31" for _, end in merged)
        symbol = str(identity.current_symbol)
        results.append(
            _UniverseIdentity(
                security_id=str(security_id),
                symbol=symbol,
                name=str(identity.issuer_name or ""),
                exchange=symbol.rsplit(".", 1)[-1] if "." in symbol else "",
                list_date=str(identity.list_date or merged[0][0]),
                delist_date="" if open_ended else (max(finite_ends) if finite_ends else ""),
                intervals=tuple(merged),
            )
        )
    return tuple(sorted(results, key=lambda item: item.symbol))


def _merge_intervals(intervals: Sequence[tuple[str, str]]) -> list[tuple[str, str]]:
    merged: list[list[str]] = []
    for start, end in sorted(intervals):
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return [(item[0], item[1]) for item in merged]


def _count_universe_rows(
    identities: Sequence[_UniverseIdentity],
    open_dates: Sequence[str],
) -> int:
    total = 0
    dates = list(open_dates)
    for identity in identities:
        for start, end in identity.intervals:
            total += bisect.bisect_right(dates, end) - bisect.bisect_left(dates, start)
    return total


def _count_missing_universe(
    active: _ActiveInputs,
    *,
    identities: Sequence[_UniverseIdentity],
    open_dates: Sequence[str],
    guard: _MemoryGuard,
) -> tuple[int, list[dict[str, Any]]]:
    by_year: dict[str, list[str]] = {}
    for date in open_dates:
        by_year.setdefault(date[:4], []).append(date)
    loader = _year_loader(active, "universe_snapshot", ("trade_date", "symbol"))
    total = 0
    summary: list[dict[str, Any]] = []
    for year, dates in sorted(by_year.items()):
        existing = loader(year)
        year_missing = 0
        target_rows = 0
        for frame in _iter_universe_frames(identities, dates):
            guard.checkpoint()
            target_rows += len(frame)
            year_missing += len(_missing_only(frame, existing))
        total += year_missing
        summary.append(
            {
                "year": year,
                "target_rows": target_rows,
                "active_rows": len(existing),
                "missing_rows": year_missing,
            }
        )
    return total, summary


def _iter_universe_frames(
    identities: Sequence[_UniverseIdentity],
    open_dates: Sequence[str],
    *,
    identity_batch_size: int = 128,
) -> Iterator[pd.DataFrame]:
    dates = np.asarray(list(open_dates), dtype=object)
    columns = (
        "symbol",
        "trade_date",
        "name",
        "exchange",
        "board",
        "list_status",
        "list_date",
        "delist_date",
        "source",
    )
    for offset in range(0, len(identities), identity_batch_size):
        rows: list[pd.DataFrame] = []
        for identity in identities[offset : offset + identity_batch_size]:
            for start, end in identity.intervals:
                selected = dates[(dates >= start) & (dates <= end)]
                if not len(selected):
                    continue
                frame = pd.DataFrame({"trade_date": selected.astype(str)})
                frame["symbol"] = identity.symbol
                frame["name"] = identity.name
                frame["exchange"] = identity.exchange
                frame["board"] = "main"
                frame["list_status"] = np.where(
                    frame["trade_date"].eq(identity.delist_date),
                    "D",
                    "L",
                )
                frame["list_date"] = identity.list_date
                frame["delist_date"] = identity.delist_date
                frame["source"] = "tushare_proxy.stock_basic+symbol_history"
                rows.append(frame.loc[:, columns])
        if rows:
            yield pd.concat(rows, ignore_index=True).sort_values(
                ["trade_date", "symbol"],
                kind="mergesort",
            )


def _active_shard_paths(active: _ActiveInputs, domain: str) -> list[Path]:
    paths = [
        resolve_manifest_path(item.path, root=active.root).resolve()
        for item in active.manifests[domain].shards
    ]
    if not paths or any(not path.exists() for path in paths):
        raise DirectRepairError(f"active_{domain}_shards_missing")
    return paths


def _bulk_append_or_noop(
    domain: str,
    paths: Sequence[Path],
    *,
    reason: str,
    workspace_root: str | Path | None,
) -> dict[str, Any]:
    if not paths:
        return {"status": "no_missing_rows", "domain": domain, "row_count": 0}
    return bulk_append_active_shards_from_parquet(
        domain,
        list(paths),
        reason,
        workspace_root=workspace_root,
    )


def _assert_active_dataset_ids_unchanged(active: _ActiveInputs) -> None:
    current = active_dataset_map(read_active_manifest(active.root))
    for domain in REPAIR_DOMAINS:
        expected = active.manifests[domain].dataset_id
        if current.get(domain) != expected:
            raise DirectRepairError(
                f"active_dataset_id_changed:{domain}:{expected}:{current.get(domain, '')}"
            )


def _write_frame(path: Path, frame: pd.DataFrame, *, schema: pa.Schema) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pandas(frame, preserve_index=False).cast(schema, safe=True)
    pq.write_table(
        table,
        path,
        compression="zstd",
        use_dictionary=True,
        row_group_size=131_072,
    )


def _validate_primary_key(
    paths: Sequence[Path],
    *,
    expected_rows: int,
    primary_key: Sequence[str] = ("trade_date", "symbol"),
    work_dir: str | Path,
    memory_guard: _MemoryGuard,
) -> None:
    strings = [str(Path(item).resolve()) for item in paths]
    if not primary_key:
        raise ValueError("primary_key_required")
    quoted = ", ".join(_quote_identifier(item) for item in primary_key)
    with _open_direct_repair_duckdb(
        ":memory:", temp_directory=Path(work_dir) / "spill"
    ) as con:
        memory_guard.checkpoint()
        row_count = int(
            con.execute("select count(*) from read_parquet(?)", [strings]).fetchone()[0]
        )
        memory_guard.checkpoint()
        memory_guard.checkpoint()
        duplicate = con.execute(
            f"""
            select {quoted}
            from read_parquet(?)
            group by {quoted}
            having count(*) > 1
            limit 1
            """,
            [strings],
        ).fetchone()
        memory_guard.checkpoint()
    if row_count != int(expected_rows):
        raise DirectRepairError(
            f"row_count_mismatch:expected={expected_rows}:actual={row_count}"
        )
    if duplicate:
        raise DirectRepairError(f"duplicate_primary_key:{duplicate}")


def _validate_bidirectional_key_alignment(
    left_paths: Sequence[Path],
    right_paths: Sequence[Path],
    *,
    left_label: str,
    right_label: str,
    work_dir: str | Path,
    memory_guard: _MemoryGuard,
) -> None:
    left = [str(Path(item).resolve()) for item in left_paths]
    right = [str(Path(item).resolve()) for item in right_paths]
    with _open_direct_repair_duckdb(
        ":memory:", temp_directory=Path(work_dir) / "spill"
    ) as con:
        memory_guard.checkpoint()
        left_only = con.execute(
            """
            select trade_date, symbol
            from read_parquet(?)
            except
            select trade_date, symbol
            from read_parquet(?)
            limit 1
            """,
            [left, right],
        ).fetchone()
        memory_guard.checkpoint()
        memory_guard.checkpoint()
        right_only = con.execute(
            """
            select trade_date, symbol
            from read_parquet(?)
            except
            select trade_date, symbol
            from read_parquet(?)
            limit 1
            """,
            [right, left],
        ).fetchone()
        memory_guard.checkpoint()
    if left_only:
        raise DirectRepairError(
            f"key_alignment_mismatch:{left_label}_only:{left_only}"
        )
    if right_only:
        raise DirectRepairError(
            f"key_alignment_mismatch:{right_label}_only:{right_only}"
        )


def _validate_factor_alignment(
    daily_paths: Sequence[Path],
    factor_path: Path,
    *,
    work_dir: str | Path,
    memory_guard: _MemoryGuard,
) -> None:
    paths = [str(Path(item).resolve()) for item in daily_paths]
    with _open_direct_repair_duckdb(
        ":memory:", temp_directory=Path(work_dir) / "spill"
    ) as con:
        memory_guard.checkpoint()
        missing = con.execute(
            """
            (select trade_date, symbol from read_parquet(?)
             except
             select trade_date, symbol from read_parquet(?))
            union all
            (select trade_date, symbol from read_parquet(?)
             except
             select trade_date, symbol from read_parquet(?))
            limit 1
            """,
            [paths, str(factor_path), str(factor_path), paths],
        ).fetchone()
        memory_guard.checkpoint()
        memory_guard.checkpoint()
        invalid = con.execute(
            """
            select count(*)
            from read_parquet(?)
            where not isfinite(fore_adjust_factor)
               or not isfinite(back_adjust_factor)
               or not isfinite(adjust_factor)
               or fore_adjust_factor <= 0
               or back_adjust_factor <= 0
               or adjust_factor <= 0
            """,
            [str(factor_path)],
        ).fetchone()[0]
        memory_guard.checkpoint()
        memory_guard.checkpoint()
        uncovered_first = con.execute(
            """
            with first_daily as (
              select symbol, min(cast(trade_date as date)) as first_daily_date
              from read_parquet(?)
              group by symbol
            ), first_factor as (
              select symbol, min(try_cast(factor_source_date as date)) as first_factor_date
              from read_parquet(?)
              where factor_provider = 'tushare_proxy'
                and factor_source_date is not null
                and factor_source_date <> ''
              group by symbol
            )
            select
              d.symbol,
              cast(d.first_daily_date as varchar),
              coalesce(cast(f.first_factor_date as varchar), '')
            from first_daily d
            left join first_factor f using(symbol)
            where f.first_factor_date is null
               or f.first_factor_date > d.first_daily_date
            order by d.symbol
            limit 1
            """,
            [paths, str(factor_path)],
        ).fetchone()
        memory_guard.checkpoint()
        memory_guard.checkpoint()
        invalid_source_date = con.execute(
            """
            select symbol, trade_date, factor_source_date
            from read_parquet(?)
            where factor_provider is distinct from 'tushare_proxy'
               or factor_source_date is null
               or factor_source_date = ''
               or try_cast(factor_source_date as date) is null
               or cast(factor_source_date as date) > cast(trade_date as date)
            limit 1
            """,
            [str(factor_path)],
        ).fetchone()
        memory_guard.checkpoint()
        memory_guard.checkpoint()
        variants = con.execute(
            """
            select count(distinct adjust_factor)
            from read_parquet(?)
            where symbol='600076.SH' and trade_date between '2024-01-01' and '2024-12-31'
            """,
            [str(factor_path)],
        ).fetchone()[0]
        memory_guard.checkpoint()
    if missing:
        raise DirectRepairError(f"factor_daily_key_mismatch:{missing}")
    if int(invalid):
        raise DirectRepairError(f"factor_non_positive_or_non_finite:{invalid}")
    if uncovered_first:
        raise DirectRepairError(
            "factor_first_date_not_covered:"
            f"symbol={uncovered_first[0]}:"
            f"first_daily_date={uncovered_first[1]}:"
            f"first_factor_date={uncovered_first[2]}"
        )
    if invalid_source_date:
        raise DirectRepairError(
            f"factor_source_date_invalid_or_future:{invalid_source_date}"
        )
    if int(variants) > 1:
        raise DirectRepairError("factor_600076_2024_changed")


def _batch_summary(batch: RepairBatch) -> dict[str, Any]:
    return {
        "partition_key": batch.partition_key,
        "row_count": len(batch.rows),
        "insert_count": int(batch.insert_count),
        "update_count": int(batch.update_count),
        "source_partition_count": int(batch.source_partition_count),
    }


def _refs_signature(refs: Sequence[RawPartitionRef]) -> dict[str, Any]:
    by_domain: dict[str, dict[str, int]] = {}
    items: list[tuple[str, str, str, str, int]] = []
    for item in refs:
        summary = by_domain.setdefault(item.raw_domain, {"partitions": 0, "rows": 0})
        summary["partitions"] += 1
        summary["rows"] += int(item.row_count)
        items.append(
            (
                item.raw_domain,
                item.partition_field,
                item.partition_value,
                item.content_sha256,
                int(item.row_count),
            )
        )
    return {
        "by_domain": by_domain,
        "content_signature": stable_hash({"items": items}, length=32),
    }


def _identity_signature(tables: IdentityRepairTables) -> str:
    records = {
        "identity": tables.security_identity.sort_values("security_id").to_dict("records"),
        "history": tables.symbol_history.sort_values(
            ["security_id", "effective_from", "symbol"]
        ).to_dict("records"),
    }
    return stable_hash(records, length=32)


def _state_path(root: Path, run_id: str) -> Path:
    return root.parent / "qdp_runtime" / "direct_repair" / run_id / "state.json"


def _write_state(path: Path, payload: dict[str, Any]) -> None:
    atomic_write_json(path, {**payload, "state_updated_at": utc_now()})


def _write_commit_progress(
    state_path: Path,
    plan: dict[str, Any],
    completed_domains: Sequence[str],
    commit_results: dict[str, Any],
) -> None:
    _write_state(
        state_path,
        {
            **plan,
            "status": "applying",
            "apply": True,
            "completed_domains": list(completed_domains),
            "commit_results": dict(commit_results),
            "resume_supported": True,
            "active_manifest_unchanged": True,
        },
    )


def _record_early_pause(
    workspace_root: str | Path | None,
    *,
    reason: str,
    apply: bool,
) -> dict[str, Any]:
    root = qdp_v2_root(workspace_root).resolve()
    active_path = root / "active" / "active.json"
    active_bytes = active_path.read_bytes()
    run_id = "direct_repair__paused_" + _sha256_bytes(active_bytes)[:16]
    payload = {
        "status": "paused_resource_guard",
        "apply": bool(apply),
        "run_id": run_id,
        "reason": reason,
        "active_sha256_before": _sha256_bytes(active_bytes),
        "active_manifest_unchanged": True,
    }
    state = _state_path(root, run_id)
    payload["state_path"] = str(state.resolve())
    _write_state(state, payload)
    return payload


def _assert_active_unchanged(active: _ActiveInputs) -> None:
    if not _active_is_unchanged(active):
        raise DirectRepairError("active_manifest_changed_during_direct_repair")


def _active_is_unchanged(active: _ActiveInputs) -> bool:
    try:
        return active.active_path.read_bytes() == active.active_bytes
    except OSError:
        return False


def _available_memory_bytes() -> int:
    try:
        import psutil  # type: ignore

        return int(psutil.virtual_memory().available)
    except Exception:
        return 2**63 - 1


def _sanitized_error(exc: BaseException) -> str:
    text = str(exc)
    configured = os.environ.get("QDP_TUSHARE_PROXY_TOKEN", "")
    if configured:
        text = text.replace(configured, "<redacted>")
    text = re.sub(
        r"(?i)\b(token|authorization|bearer)\b\s*[:=]?\s*[^\s,;]+",
        r"\1=<redacted>",
        text,
    )
    return text[:500]


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sql_literal(value: str) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def _configure_duckdb(
    connection: duckdb.DuckDBPyConnection,
    *,
    temp_directory: str | Path,
    memory_sampler: Callable[[], int] | None = None,
    total_memory_sampler: Callable[[], int] | None = None,
) -> DuckDbResourceSettings:
    """Use available RAM above the 0.5 GiB floor and repository-backed spill."""

    return configure_dynamic_duckdb(
        connection,
        temp_directory=temp_directory,
        threads=max(1, min(os.cpu_count() or 1, 8)),
        memory_sampler=memory_sampler or available_memory_bytes,
        total_memory_sampler=total_memory_sampler or total_physical_memory_bytes,
        floor_bytes=MEMORY_FLOOR_BYTES,
        low_memory_seconds=MEMORY_LOW_SECONDS,
    )


def _open_direct_repair_duckdb(
    database: str | Path,
    *,
    temp_directory: str | Path,
) -> GuardedDuckDbConnection:
    connection = duckdb.connect(str(database))
    try:
        settings = _configure_duckdb(
            connection,
            temp_directory=temp_directory,
        )
        return guard_configured_duckdb(
            connection,
            settings=settings,
            error_factory=DirectRepairPaused,
        )
    except BaseException:
        connection.close()
        raise


def _quote_identifier(value: str) -> str:
    return '"' + str(value).replace('"', '""') + '"'


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m quant_data_platform.qdp_v2.direct_repair",
        description="Plan or directly repair trusted-source QDP v2 active datasets.",
    )
    parser.add_argument("--workspace-root", default="")
    parser.add_argument("--identity-config", default="")
    parser.add_argument("--start-date", default=TRUSTED_START_DATE)
    parser.add_argument("--factor-output-path", default="")
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Repair current dataset IDs in place. Omit for the default dry-run.",
    )
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    payload = run_direct_repair(
        workspace_root=str(args.workspace_root or "") or None,
        apply=bool(args.apply),
        factor_output_path=str(args.factor_output_path or "") or None,
        identity_config_path=str(args.identity_config or "") or None,
        start_date=str(args.start_date),
    )
    if args.json:
        print(json.dumps(json_safe(payload), ensure_ascii=False, indent=2))
    else:
        print(f"status: {payload.get('status', '')}")
        print(f"run_id: {payload.get('run_id', '')}")
        print(f"active_manifest_unchanged: {payload.get('active_manifest_unchanged', '')}")
    return 0 if str(payload.get("status", "")) in {"dry_run", "applied"} else 3


__all__ = [
    "DIRECT_REPAIR_VERSION",
    "DirectRepairError",
    "DirectRepairPaused",
    "build_arg_parser",
    "main",
    "plan_direct_repair",
    "run_direct_repair",
    "write_adjust_factor_parquet",
]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
