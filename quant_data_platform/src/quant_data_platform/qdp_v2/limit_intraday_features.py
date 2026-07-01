from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import shutil
from pathlib import Path
from typing import Any, Iterable, Mapping

from quant_data_platform.core.json_io import json_safe
from quant_data_platform.qdp_v2.environment import runtime_environment
from quant_data_platform.qdp_v2.manifest import (
    DatasetManifest,
    ShardManifestEntry,
    atomic_write_json,
    dataset_manifest_for_id,
    path_for_manifest,
    qdp_v2_root,
    read_active_manifest,
    read_dataset_manifest,
    schema_hash,
    stable_hash,
    utc_now,
    write_active_manifest,
    write_dataset_manifest,
)
from quant_data_platform.qdp_v2.runtime import resolve_runtime_profile
from quant_data_platform.qdp_v2.status import active_dataset_map


DOMAIN = "limit_intraday_features"
CONTRACT = "qdp_v2_limit_intraday_features_1m_v1"
PRIMARY_KEY = ["trade_date", "symbol"]


def build_limit_intraday_features(
    *,
    workspace_root: str | Path | None = None,
    source_dataset_id: str = "",
    runtime: str = "balanced",
    duckdb_memory_limit: str = "",
    threads: int = 0,
    workers: int = 1,
    batch_shards: int = 64,
    max_batches: int = 0,
    resume: bool = False,
    activate: bool = False,
) -> dict[str, Any]:
    root = qdp_v2_root(workspace_root)
    active = read_active_manifest(root)
    datasets = active_dataset_map(active)
    one_id = source_dataset_id or str(datasets.get("market_intraday_1m", "") or "")
    daily_id = str(datasets.get("market_daily_raw", "") or "")
    status_id = str(datasets.get("security_status", "") or "")
    if not one_id or not daily_id or not status_id:
        return {"status": "error", "errors": ["market_intraday_1m_or_daily_or_status_missing"]}

    one = read_dataset_manifest(_require_manifest(root, one_id, "market_intraday_1m"))
    daily = read_dataset_manifest(_require_manifest(root, daily_id, "market_daily_raw"))
    status = read_dataset_manifest(_require_manifest(root, status_id, "security_status"))
    profile = resolve_runtime_profile(runtime)
    memory_limit = str(duckdb_memory_limit or "").strip() or profile.duckdb_memory_limit
    thread_count = max(1, int(threads or profile.duckdb_threads or 1))
    worker_count = max(1, int(workers or 1))
    batch_size = max(1, int(batch_shards or 1))

    dataset_id = f"{DOMAIN}__{stable_hash({'one': one.dataset_id, 'daily': daily.dataset_id, 'status': status.dataset_id, 'contract': CONTRACT})}"
    target_dir = root / "datasets" / DOMAIN / dataset_id
    staging = target_dir / ".staging"
    staging_shards = staging / "shards"
    if staging.exists() and not resume:
        shutil.rmtree(staging)
    staging_shards.mkdir(parents=True, exist_ok=True)

    reference_path = staging / "daily_limit_reference.parquet"
    if not reference_path.exists() or not resume:
        _write_daily_limit_reference(
            root=root,
            daily=daily,
            status=status,
            target_path=reference_path,
            memory_limit=memory_limit,
            threads=thread_count,
        )

    shard_batches = list(_batches(list(enumerate(one.shards)), batch_size))
    if max_batches and int(max_batches) > 0:
        shard_batches = shard_batches[: int(max_batches)]
    tasks: list[dict[str, Any]] = []
    for batch_index, batch in enumerate(shard_batches):
        first_index = int(batch[0][0])
        last_index = int(batch[-1][0])
        target_path = staging_shards / f"part_{batch_index:06d}_{DOMAIN}.parquet"
        if resume and target_path.exists() and target_path.stat().st_size > 0:
            continue
        paths = [_resolve_data_path(root, shard.path) for _, shard in batch]
        starts = [str(shard.start_date or "") for _, shard in batch if str(shard.start_date or "")]
        ends = [str(shard.end_date or "") for _, shard in batch if str(shard.end_date or "")]
        tasks.append(
            {
                "batch_index": batch_index,
                "first_shard_index": first_index,
                "last_shard_index": last_index,
                "input_paths": [str(path.resolve()) for path in paths],
                "start_date": min(starts) if starts else "",
                "end_date": max(ends) if ends else "",
                "target_path": str(target_path.resolve()),
                "reference_path": str(reference_path.resolve()),
                "memory_limit": _worker_memory_limit(memory_limit, worker_count),
                "threads": max(1, min(2, thread_count)),
            }
        )

    entries_by_batch: dict[int, ShardManifestEntry] = {}
    errors: list[dict[str, Any]] = []
    processed = 0
    if tasks:
        if worker_count == 1:
            for task in tasks:
                result = _derive_batch_worker(task)
                if result.get("status") == "ok":
                    entries_by_batch[int(result["batch_index"])] = _entry_from_result(root=root, dataset_id=dataset_id, result=result)
                    processed += 1
                else:
                    errors.append(result)
        else:
            with ProcessPoolExecutor(max_workers=worker_count) as executor:
                futures = {executor.submit(_derive_batch_worker, task): task for task in tasks}
                for future in as_completed(futures):
                    result = future.result()
                    if result.get("status") == "ok":
                        entries_by_batch[int(result["batch_index"])] = _entry_from_result(root=root, dataset_id=dataset_id, result=result)
                        processed += 1
                    else:
                        errors.append(result)
    if errors:
        payload = {
            "status": "error",
            "target_dataset_id": dataset_id,
            "processed_batches": processed,
            "error_count": len(errors),
            "errors": errors[:50],
        }
        atomic_write_json(root / "runs" / f"limit_intraday_features_failed_{_stamp()}.json", payload)
        return payload

    if resume:
        for path in sorted(staging_shards.glob("part_*_limit_intraday_features.parquet")):
            batch_index = int(path.name.split("_")[1])
            if batch_index in entries_by_batch:
                continue
            stats = _parquet_stats(path)
            entries_by_batch[batch_index] = ShardManifestEntry(
                path=path_for_manifest(path, root=root).replace("/.staging/", "/"),
                row_count=int(stats["row_count"]),
                start_date=str(stats["start_date"]),
                end_date=str(stats["end_date"]),
                status="stored",
                file_size=int(path.stat().st_size),
                schema_hash=str(stats["schema_hash"]),
                source_path="",
                content_key=CONTRACT,
                metadata={"resumed_existing": True, "source_1m_dataset_id": one.dataset_id},
            )

    final_shards = target_dir / "shards"
    target_dir.mkdir(parents=True, exist_ok=True)
    if final_shards.exists():
        shutil.rmtree(final_shards)
    staging_shards.replace(final_shards)
    shutil.rmtree(staging, ignore_errors=True)

    entries: list[ShardManifestEntry] = []
    for batch_index, entry in sorted(entries_by_batch.items()):
        final_path = final_shards / Path(entry.path).name
        stats = _parquet_stats(final_path)
        entries.append(
            ShardManifestEntry(
                path=path_for_manifest(final_path, root=root),
                row_count=int(stats["row_count"]),
                start_date=str(stats["start_date"]),
                end_date=str(stats["end_date"]),
                status="stored",
                file_size=int(final_path.stat().st_size),
                schema_hash=str(stats["schema_hash"]),
                source_path="",
                content_key=CONTRACT,
                metadata={"batch_index": batch_index, "source_1m_dataset_id": one.dataset_id},
            )
        )
    schema = _parquet_schema(final_shards / Path(entries[0].path).name) if entries else []
    manifest = DatasetManifest(
        dataset_id=dataset_id,
        domain=DOMAIN,
        layer="derived",
        frequency="1d",
        contract_version=CONTRACT,
        primary_key=PRIMARY_KEY,
        start_date=min([item.start_date for item in entries if item.start_date], default=""),
        end_date=max([item.end_date for item in entries if item.end_date], default=""),
        row_count=sum(int(item.row_count) for item in entries),
        schema_hash=schema_hash(schema),
        schema=schema,
        shards=entries,
        source={
            "provider": "qdp_v2",
            "created_by": "qdp rebuild limit-intraday",
            "created_at": utc_now(),
            "source_1m_dataset_id": one.dataset_id,
            "daily_dataset_id": daily.dataset_id,
            "security_status_dataset_id": status.dataset_id,
        },
        quality={
            "path_refs_exist": True,
            "primary_key_unique": "not_checked",
            "derived_from": "market_intraday_1m + market_daily_raw + security_status",
            "no_l2_order_book_fields": True,
        },
    )
    manifest_path = write_dataset_manifest(root, manifest)
    active_path = ""
    if activate:
        updated = read_active_manifest(root)
        updated_datasets = active_dataset_map(updated)
        updated_datasets[DOMAIN] = dataset_id
        updated["datasets"] = updated_datasets
        for old_key in ("raw", "derived", "research_panels", "memmap"):
            updated.pop(old_key, None)
        source = dict(updated.get("source", {}) or {})
        source["limit_intraday_features"] = {
            "updated_at": utc_now(),
            "dataset_id": dataset_id,
            "created_by": "qdp rebuild limit-intraday",
        }
        updated["source"] = source
        active_path = str(write_active_manifest(root, updated).resolve())

    payload = {
        "status": "ok",
        "target_dataset_id": dataset_id,
        "manifest_path": str(manifest_path.resolve()),
        "active_manifest": active_path,
        "row_count": manifest.row_count,
        "start_date": manifest.start_date,
        "end_date": manifest.end_date,
        "shard_count": len(entries),
        "processed_batches": processed,
        "batch_shards": batch_size,
        "runtime_environment": runtime_environment(),
    }
    atomic_write_json(root / "runs" / f"limit_intraday_features_{_stamp()}.json", payload)
    return payload


def _write_daily_limit_reference(*, root: Path, daily: DatasetManifest, status: DatasetManifest, target_path: Path, memory_limit: str, threads: int) -> None:
    import duckdb  # type: ignore

    daily_paths = [_resolve_data_path(root, shard.path) for shard in daily.shards]
    status_paths = [_resolve_data_path(root, shard.path) for shard in status.shards]
    target_path.parent.mkdir(parents=True, exist_ok=True)
    if target_path.exists():
        target_path.unlink()
    with duckdb.connect(":memory:") as con:
        con.execute(f"set memory_limit='{memory_limit}'")
        con.execute(f"set threads={int(threads)}")
        con.execute(
            f"""
            copy (
              with daily as (
                select
                  symbol,
                  trade_date,
                  open as daily_open,
                  close as daily_close,
                  lag(close) over (partition by symbol order by trade_date) as prev_close
                from read_parquet({_path_list_sql(daily_paths)}, union_by_name=true)
              ),
              status as (
                select symbol, trade_date, is_st
                from read_parquet({_path_list_sql(status_paths)}, union_by_name=true)
              )
              select
                d.symbol,
                d.trade_date,
                d.daily_open,
                d.daily_close,
                d.prev_close,
                coalesce(s.is_st, false) as is_st,
                case when d.prev_close is not null and d.prev_close > 0
                  then round(cast(d.prev_close as double) * (1 + case when coalesce(s.is_st, false) then 0.05 else 0.10 end), 2)
                  else null end as up_limit,
                case when d.prev_close is not null and d.prev_close > 0
                  then round(cast(d.prev_close as double) * (1 - case when coalesce(s.is_st, false) then 0.05 else 0.10 end), 2)
                  else null end as down_limit
              from daily d
              left join status s using(symbol, trade_date)
              order by trade_date, symbol
            ) to '{_sql_path(target_path)}' (format parquet)
            """
        )


def _derive_batch_worker(task: Mapping[str, Any]) -> dict[str, Any]:
    try:
        import duckdb  # type: ignore

        input_paths = [Path(item) for item in list(task.get("input_paths", []) or [])]
        target_path = Path(str(task.get("target_path", "")))
        reference_path = Path(str(task.get("reference_path", "")))
        target_path.parent.mkdir(parents=True, exist_ok=True)
        if target_path.exists():
            target_path.unlink()
        start_date = str(task.get("start_date", "") or "")
        end_date = str(task.get("end_date", "") or "")
        with duckdb.connect(":memory:") as con:
            con.execute(f"set memory_limit='{task.get('memory_limit', '4GB')}'")
            con.execute(f"set threads={int(task.get('threads', 1) or 1)}")
            con.execute(
                f"""
                copy (
                  with bars_raw as (
                    select
                      symbol,
                      trade_date,
                      regexp_replace(cast(bar_time as varchar), '[^0-9]', '', 'g') as bt,
                      open,
                      high,
                      low,
                      close,
                      volume,
                      amount
                    from read_parquet({_path_list_sql(input_paths)}, union_by_name=true)
                  ),
                  bars as (
                    select
                      *,
                      substr(bt, 1, 4) as bar_time,
                      cast(substr(bt, 1, 2) as integer) * 60 + cast(substr(bt, 3, 2) as integer) as minute_of_day,
                      row_number() over (
                        partition by symbol, trade_date
                        order by cast(substr(bt, 1, 2) as integer) * 60 + cast(substr(bt, 3, 2) as integer)
                      ) as bar_index
                    from bars_raw
                    where length(bt) >= 4
                  ),
                  ref as (
                    select *
                    from read_parquet('{_sql_path(reference_path)}')
                    where trade_date >= '{start_date}' and trade_date <= '{end_date}'
                  ),
                  flagged as (
                    select
                      b.symbol,
                      b.trade_date,
                      b.bar_time,
                      b.minute_of_day,
                      b.bar_index,
                      cast(b.open as double) as open,
                      cast(b.high as double) as high,
                      cast(b.low as double) as low,
                      cast(b.close as double) as close,
                      cast(b.volume as double) as volume,
                      cast(b.amount as double) as amount,
                      r.prev_close,
                      r.up_limit,
                      r.down_limit,
                      r.daily_open,
                      r.daily_close,
                      r.is_st,
                      r.up_limit is not null and cast(b.high as double) >= r.up_limit - 0.011 as touch_limit_up,
                      r.down_limit is not null and cast(b.low as double) <= r.down_limit + 0.011 as touch_limit_down,
                      r.up_limit is not null and cast(b.close as double) >= r.up_limit - 0.011 as close_limit_up,
                      r.down_limit is not null and cast(b.close as double) <= r.down_limit + 0.011 as close_limit_down,
                      r.up_limit is not null and cast(b.open as double) >= r.up_limit - 0.011 as open_limit_up,
                      r.down_limit is not null and cast(b.open as double) <= r.down_limit + 0.011 as open_limit_down
                    from bars b
                    left join ref r using(symbol, trade_date)
                  ),
                  enriched as (
                    select
                      *,
                      lead(close_limit_up) over (partition by symbol, trade_date order by minute_of_day) as next_close_limit_up,
                      lead(close_limit_down) over (partition by symbol, trade_date order by minute_of_day) as next_close_limit_down,
                      max(case when not close_limit_up then bar_index else null end) over (partition by symbol, trade_date) as last_non_limit_up_idx,
                      max(case when not close_limit_down then bar_index else null end) over (partition by symbol, trade_date) as last_non_limit_down_idx
                    from flagged
                  ),
                  agg as (
                    select
                      symbol,
                      trade_date,
                      any_value(prev_close) as prev_close,
                      any_value(up_limit) as up_limit,
                      any_value(down_limit) as down_limit,
                      any_value(is_st) as is_st,
                      count(*) as bar_count,
                      arg_min(open, minute_of_day) as first_open,
                      arg_max(close, minute_of_day) as last_close,
                      max(high) as intraday_high,
                      min(low) as intraday_low,
                      bool_or(open_limit_up and bar_index = 1) as is_open_limit_up,
                      bool_or(open_limit_down and bar_index = 1) as is_open_limit_down,
                      bool_or(close_limit_up and bar_index = 240) as is_close_limit_up,
                      bool_or(close_limit_down and bar_index = 240) as is_close_limit_down,
                      bool_or(touch_limit_up) as touch_limit_up,
                      bool_or(touch_limit_down) as touch_limit_down,
                      min(case when touch_limit_up then bar_time else null end) as first_touch_up_time,
                      max(case when touch_limit_up then bar_time else null end) as last_touch_up_time,
                      min(case when touch_limit_down then bar_time else null end) as first_touch_down_time,
                      max(case when touch_limit_down then bar_time else null end) as last_touch_down_time,
                      min(case when touch_limit_up then bar_index else null end) as first_touch_up_index,
                      min(case when touch_limit_down then bar_index else null end) as first_touch_down_index,
                      sum(case when touch_limit_up then 1 else 0 end) as limit_up_touch_minutes,
                      sum(case when touch_limit_down then 1 else 0 end) as limit_down_touch_minutes,
                      sum(case when close_limit_up then 1 else 0 end) as limit_up_close_minutes,
                      sum(case when close_limit_down then 1 else 0 end) as limit_down_close_minutes,
                      sum(case when close_limit_up and coalesce(next_close_limit_up, true) = false then 1 else 0 end) as break_limit_up_count,
                      sum(case when close_limit_down and coalesce(next_close_limit_down, true) = false then 1 else 0 end) as break_limit_down_count,
                      sum(case when close_limit_up and (last_non_limit_up_idx is null or bar_index > last_non_limit_up_idx) then 1 else 0 end) as sealed_up_minutes_to_close,
                      sum(case when close_limit_down and (last_non_limit_down_idx is null or bar_index > last_non_limit_down_idx) then 1 else 0 end) as sealed_down_minutes_to_close,
                      min(open) as min_open,
                      max(open) as max_open,
                      min(high) as min_high,
                      max(high) as max_high,
                      min(low) as min_low,
                      max(low) as max_low,
                      min(close) as min_close,
                      max(close) as max_close
                    from enriched
                    group by symbol, trade_date
                  )
                  select
                    symbol,
                    trade_date,
                    prev_close,
                    up_limit,
                    down_limit,
                    is_st,
                    bar_count,
                    is_open_limit_up,
                    is_open_limit_down,
                    is_close_limit_up,
                    is_close_limit_down,
                    touch_limit_up,
                    touch_limit_down,
                    (
                      up_limit is not null
                      and min_low >= up_limit - 0.011
                      and max_high <= up_limit + 0.011
                    ) as is_one_word_limit_up,
                    (
                      down_limit is not null
                      and max_high <= down_limit + 0.011
                      and min_low >= down_limit - 0.011
                    ) as is_one_word_limit_down,
                    first_touch_up_time,
                    last_touch_up_time,
                    first_touch_down_time,
                    last_touch_down_time,
                    first_touch_up_index,
                    first_touch_down_index,
                    limit_up_touch_minutes,
                    limit_down_touch_minutes,
                    limit_up_close_minutes,
                    limit_down_close_minutes,
                    break_limit_up_count,
                    break_limit_down_count,
                    break_limit_up_count > 0 as opened_after_limit_up,
                    break_limit_down_count > 0 as opened_after_limit_down,
                    is_close_limit_up and touch_limit_up as close_sealed_up,
                    is_close_limit_down and touch_limit_down as close_sealed_down,
                    case when is_close_limit_up then sealed_up_minutes_to_close else 0 end as sealed_up_minutes_to_close,
                    case when is_close_limit_down then sealed_down_minutes_to_close else 0 end as sealed_down_minutes_to_close,
                    least(100.0, greatest(0.0,
                      (case when is_close_limit_up then 40.0 else 0.0 end)
                      + (case when up_limit is not null and min_low >= up_limit - 0.011 and max_high <= up_limit + 0.011 then 30.0 else 0.0 end)
                      + (case when first_touch_up_index is not null then 20.0 * (1.0 - ((first_touch_up_index - 1.0) / greatest(bar_count - 1.0, 1.0))) else 0.0 end)
                      + least((limit_up_close_minutes / greatest(bar_count, 1.0)) * 10.0, 10.0)
                      - break_limit_up_count * 5.0
                    )) as limit_up_strength_score,
                    least(100.0, greatest(0.0,
                      (case when is_close_limit_down then 40.0 else 0.0 end)
                      + (case when down_limit is not null and max_high <= down_limit + 0.011 and min_low >= down_limit - 0.011 then 30.0 else 0.0 end)
                      + (case when first_touch_down_index is not null then 20.0 * (1.0 - ((first_touch_down_index - 1.0) / greatest(bar_count - 1.0, 1.0))) else 0.0 end)
                      + least((limit_down_close_minutes / greatest(bar_count, 1.0)) * 10.0, 10.0)
                      - break_limit_down_count * 5.0
                    )) as limit_down_strength_score,
                    'qdp_v2_1m_limit_intraday_features' as source
                  from agg
                  order by trade_date, symbol
                ) to '{_sql_path(target_path)}' (format parquet)
                """
            )
        stats = _parquet_stats(target_path)
        return {
            "status": "ok",
            "batch_index": int(task["batch_index"]),
            "first_shard_index": int(task["first_shard_index"]),
            "last_shard_index": int(task["last_shard_index"]),
            "target_path": str(target_path.resolve()),
            **stats,
        }
    except Exception as exc:
        return {
            "status": "error",
            "batch_index": int(task.get("batch_index", -1)),
            "error_type": type(exc).__name__,
            "message": str(exc),
        }


def _entry_from_result(*, root: Path, dataset_id: str, result: Mapping[str, Any]) -> ShardManifestEntry:
    path = Path(str(result["target_path"]))
    return ShardManifestEntry(
        path=f"datasets/{DOMAIN}/{dataset_id}/shards/{path.name}",
        row_count=int(result["row_count"]),
        start_date=str(result["start_date"]),
        end_date=str(result["end_date"]),
        status="stored",
        file_size=int(path.stat().st_size),
        schema_hash=str(result["schema_hash"]),
        source_path="",
        content_key=CONTRACT,
        metadata={
            "batch_index": int(result["batch_index"]),
            "first_shard_index": int(result["first_shard_index"]),
            "last_shard_index": int(result["last_shard_index"]),
        },
    )


def _parquet_stats(path: Path) -> dict[str, Any]:
    import duckdb  # type: ignore

    with duckdb.connect(":memory:") as con:
        row = con.execute(
            f"""
            select
              count(*) as row_count,
              strftime(min(cast(trade_date as date)), '%Y-%m-%d') as start_date,
              strftime(max(cast(trade_date as date)), '%Y-%m-%d') as end_date
            from read_parquet('{_sql_path(path)}')
            """
        ).fetchone()
    return {
        "row_count": int(row[0] or 0),
        "start_date": str(row[1] or ""),
        "end_date": str(row[2] or ""),
        "schema_hash": schema_hash(_parquet_schema(path)),
    }


def _parquet_schema(path: Path) -> list[dict[str, str]]:
    import pyarrow.parquet as pq  # type: ignore

    schema = pq.ParquetFile(path).schema_arrow
    return [{"name": field.name, "type": str(field.type)} for field in schema]


def _worker_memory_limit(memory_limit: str, workers: int) -> str:
    text = str(memory_limit or "4GB").upper().strip()
    if not text.endswith("GB"):
        return memory_limit
    try:
        gb = float(text[:-2])
    except ValueError:
        return memory_limit
    return f"{max(2.0, gb / max(1, int(workers))):.1f}GB"


def _require_manifest(root: Path, dataset_id: str, domain: str) -> Path:
    path = dataset_manifest_for_id(root, dataset_id, domain)
    if path is None:
        raise FileNotFoundError(f"dataset_manifest_missing:{domain}:{dataset_id}")
    return path


def _resolve_data_path(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (root / path).resolve()


def _batches(items: list[Any], size: int) -> Iterable[list[Any]]:
    for index in range(0, len(items), max(1, int(size))):
        yield items[index : index + max(1, int(size))]


def _path_list_sql(paths: Iterable[Path]) -> str:
    return "[" + ", ".join("'" + _sql_path(path) + "'" for path in paths) + "]"


def _sql_path(path: Path) -> str:
    return str(path).replace("\\", "/").replace("'", "''")


def _stamp() -> str:
    return utc_now().replace(":", "").replace("-", "")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="qdp rebuild limit-intraday", description="Rebuild detailed limit-up/down intraday features from active 1m bars.")
    parser.add_argument("--workspace-root", default="")
    parser.add_argument("--source-dataset-id", default="")
    parser.add_argument("--runtime", default="balanced", choices=("safe", "balanced", "fast"))
    parser.add_argument("--duckdb-memory-limit", default="")
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--batch-shards", type=int, default=64)
    parser.add_argument("--max-batches", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--activate", action="store_true")
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    payload = build_limit_intraday_features(
        workspace_root=str(args.workspace_root or "") or None,
        source_dataset_id=str(args.source_dataset_id or ""),
        runtime=str(args.runtime or "balanced"),
        duckdb_memory_limit=str(args.duckdb_memory_limit or ""),
        threads=int(args.threads or 0),
        workers=int(args.workers or 1),
        batch_shards=int(args.batch_shards or 64),
        max_batches=int(args.max_batches or 0),
        resume=bool(args.resume),
        activate=bool(args.activate),
    )
    if bool(args.json):
        print(json.dumps(json_safe(payload), ensure_ascii=False, indent=2))
    else:
        print("\n".join(f"{key}: {value}" for key, value in payload.items()))
    return 0 if str(payload.get("status", "")) == "ok" else 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
