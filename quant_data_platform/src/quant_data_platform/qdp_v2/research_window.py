from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import os
import shutil
from pathlib import Path
from typing import Any, Mapping

from quant_data_platform.core.json_io import json_safe
from quant_data_platform.qdp_v2.environment import runtime_environment
from quant_data_platform.qdp_v2.manifest import (
    DatasetManifest,
    ShardManifestEntry,
    atomic_write_json,
    dataset_manifest_for_id,
    qdp_v2_root,
    read_active_manifest,
    read_dataset_manifest,
    resolve_manifest_path,
    schema_hash,
    stable_hash,
    utc_now,
    write_active_manifest,
    write_dataset_manifest,
)
from quant_data_platform.qdp_v2.runtime import resolve_runtime_profile


DEFAULT_START_DATE = "2011-11-22"
DEFAULT_SYMBOL_STARTS = {"600036.SH": "2016-07-25"}


def apply_research_window(
    *,
    workspace_root: str | Path | None = None,
    start_date: str = DEFAULT_START_DATE,
    symbol_starts: Mapping[str, str] | None = None,
    domains: str = "",
    runtime: str = "balanced",
    workers: int = 4,
    duckdb_memory_limit: str = "",
    activate: bool = False,
    hardlink_reused: bool = True,
) -> dict[str, Any]:
    root = qdp_v2_root(workspace_root)
    active = read_active_manifest(root)
    profile = resolve_runtime_profile(runtime)
    memory_limit = str(duckdb_memory_limit or "").strip() or profile.duckdb_memory_limit
    selected_domains = _selected_domains(active, domains)
    symbol_rules = {str(k).upper(): str(v) for k, v in dict(symbol_starts or DEFAULT_SYMBOL_STARTS).items() if str(k).strip() and str(v).strip()}
    outputs: dict[str, dict[str, Any]] = {}
    new_active = json.loads(json.dumps(active, ensure_ascii=False))

    for section_name in ("raw", "derived", "research_panels"):
        section = dict(active.get(section_name, {}) or {})
        for domain, dataset_id in section.items():
            if selected_domains and domain not in selected_domains:
                continue
            manifest_path = dataset_manifest_for_id(root, str(dataset_id), str(domain))
            if manifest_path is None:
                outputs[domain] = {"status": "skipped", "reason": "manifest_not_found", "dataset_id": dataset_id}
                continue
            manifest = read_dataset_manifest(manifest_path)
            if "trade_date" not in _column_names(manifest):
                outputs[domain] = {"status": "skipped", "reason": "no_trade_date_column", "dataset_id": dataset_id}
                continue
            result = _window_dataset(
                root=root,
                source=manifest,
                start_date=start_date,
                symbol_rules=symbol_rules if "symbol" in _column_names(manifest) else {},
                runtime=runtime,
                workers=workers,
                memory_limit=memory_limit,
                hardlink_reused=hardlink_reused,
            )
            outputs[domain] = result
            if result.get("status") == "ok":
                new_active.setdefault(section_name, {})[domain] = result["target_dataset_id"]

    new_active["active_start_date"] = start_date
    new_active.setdefault("source", {})
    new_active["source"] = {
        **dict(new_active.get("source", {}) or {}),
        "research_window": {
            "created_by": "qdp_v2.apply_research_window",
            "created_at": utc_now(),
            "start_date": start_date,
            "symbol_starts": symbol_rules,
        },
    }
    new_active["updated_at"] = utc_now()

    status = "ok" if all(item.get("status") in {"ok", "skipped"} for item in outputs.values()) else "error"
    active_path = ""
    if status == "ok" and activate:
        active_path = str(write_active_manifest(root, new_active).resolve())

    payload = {
        "status": status,
        "qdp_v2_root": str(root.resolve()),
        "start_date": start_date,
        "symbol_starts": symbol_rules,
        "activate": bool(activate),
        "active_manifest": active_path,
        "outputs": outputs,
        "runtime_environment": runtime_environment(),
    }
    run_path = root / "runs" / f"research_window_{utc_now().replace(':', '').replace('-', '')}.json"
    atomic_write_json(run_path, payload)
    payload["run_manifest"] = str(run_path.resolve())
    return payload


def _window_dataset(
    *,
    root: Path,
    source: DatasetManifest,
    start_date: str,
    symbol_rules: Mapping[str, str],
    runtime: str,
    workers: int,
    memory_limit: str,
    hardlink_reused: bool,
) -> dict[str, Any]:
    target_dataset_id = f"{source.domain}__{stable_hash({'source': source.dataset_id, 'research_window': start_date, 'symbol_starts': dict(symbol_rules)})}"
    target_dir = root / "datasets" / source.domain / target_dataset_id
    staging_dir = target_dir / ".staging"
    staging_shards = staging_dir / "shards"
    if staging_dir.exists():
        shutil.rmtree(staging_dir)
    staging_shards.mkdir(parents=True, exist_ok=True)
    final_shards = target_dir / "shards"
    target_dir.mkdir(parents=True, exist_ok=True)

    columns = _column_names(source)
    has_symbol = "symbol" in columns and bool(symbol_rules)
    tasks: list[dict[str, Any]] = []
    linked_entries: dict[int, ShardManifestEntry] = {}
    skipped_before_window = 0
    for index, shard in enumerate(source.shards):
        source_path = resolve_manifest_path(shard.path, root=root)
        if not source_path.exists():
            tasks.append({"index": index, "status": "error", "error": "missing_source_path", "source_path": str(source_path)})
            continue
        if shard.end_date and shard.end_date < start_date:
            skipped_before_window += 1
            continue
        target_path = staging_shards / Path(shard.path).name
        needs_rewrite = (not shard.start_date) or shard.start_date < start_date
        if not needs_rewrite and has_symbol and _may_have_symbol_before_cutoff(root=root, shard=shard, source_path=source_path, symbol_rules=symbol_rules, memory_limit=memory_limit):
            needs_rewrite = True
        if needs_rewrite:
            tasks.append(
                {
                    "index": index,
                    "source_path": str(source_path),
                    "target_path": str(target_path),
                    "target_dataset_id": target_dataset_id,
                    "domain": source.domain,
                    "source_shard": shard.to_dict(),
                    "start_date": start_date,
                    "symbol_rules": dict(symbol_rules),
                    "has_symbol": has_symbol,
                    "memory_limit": memory_limit,
                }
            )
        else:
            _copy_or_link_file(source_path, target_path, hardlink=hardlink_reused)
            linked_entries[index] = ShardManifestEntry(
                path=f"datasets/{source.domain}/{target_dataset_id}/shards/{target_path.name}",
                row_count=shard.row_count,
                start_date=shard.start_date,
                end_date=shard.end_date,
                status="stored",
                file_size=int(target_path.stat().st_size),
                schema_hash=shard.schema_hash,
                source_path=str(source_path.resolve()),
                content_key=shard.content_key,
                metadata={**dict(shard.metadata or {}), "research_window_reused": True, "source_dataset_id": source.dataset_id},
            )

    worker_count = max(1, min(int(workers or 1), len([t for t in tasks if "source_path" in t]) or 1))
    entries_by_index: dict[int, ShardManifestEntry] = dict(linked_entries)
    errors: list[dict[str, Any]] = []
    rewrite_tasks = [task for task in tasks if "source_path" in task]
    immediate_errors = [task for task in tasks if task.get("status") == "error"]
    errors.extend(immediate_errors)
    if worker_count <= 1 or len(rewrite_tasks) <= 1:
        for task in rewrite_tasks:
            result = _window_shard_task(task)
            _collect_task_result(result, entries_by_index, errors)
    else:
        with ProcessPoolExecutor(max_workers=worker_count) as pool:
            futures = [pool.submit(_window_shard_task, task) for task in rewrite_tasks]
            for future in as_completed(futures):
                _collect_task_result(future.result(), entries_by_index, errors)

    if errors:
        return {
            "status": "error",
            "source_dataset_id": source.dataset_id,
            "target_dataset_id": target_dataset_id,
            "errors": errors[:50],
            "error_count": len(errors),
        }

    entries = [entries_by_index[key] for key in sorted(entries_by_index)]
    if final_shards.exists():
        shutil.rmtree(final_shards)
    staging_shards.replace(final_shards)
    if staging_dir.exists():
        shutil.rmtree(staging_dir, ignore_errors=True)

    schema = source.schema or (_parquet_schema(resolve_manifest_path(entries[0].path, root=root)) if entries else [])
    manifest = DatasetManifest(
        dataset_id=target_dataset_id,
        domain=source.domain,
        layer=source.layer,
        frequency=source.frequency,
        contract_version=source.contract_version,
        primary_key=list(source.primary_key),
        start_date=min([item.start_date for item in entries if item.start_date], default=""),
        end_date=max([item.end_date for item in entries if item.end_date], default=""),
        row_count=sum(int(item.row_count) for item in entries),
        schema_hash=schema_hash(schema),
        schema=schema,
        shards=entries,
        source={
            "provider": "qdp_v2",
            "created_by": "apply_research_window",
            "created_at": utc_now(),
            "source_dataset_id": source.dataset_id,
            "research_window_start_date": start_date,
            "symbol_starts": dict(symbol_rules),
        },
        quality={
            **dict(source.quality or {}),
            "path_refs_exist": True,
            "research_window_start_date": start_date,
            "symbol_starts": dict(symbol_rules),
            "primary_key_unique": "not_checked_after_research_window",
        },
        legacy=dict(source.legacy or {}),
        notes=[
            *list(source.notes or []),
            f"research active window starts at {start_date}",
            *[f"{symbol} treated as active from {date}" for symbol, date in sorted(symbol_rules.items())],
        ],
    )
    manifest_path = write_dataset_manifest(root, manifest)
    return {
        "status": "ok",
        "source_dataset_id": source.dataset_id,
        "target_dataset_id": target_dataset_id,
        "manifest_path": str(manifest_path.resolve()),
        "row_count": manifest.row_count,
        "start_date": manifest.start_date,
        "end_date": manifest.end_date,
        "linked_shards": len(linked_entries),
        "rewritten_shards": len(rewrite_tasks),
        "skipped_before_window_shards": skipped_before_window,
        "shard_count": len(entries),
    }


def _collect_task_result(result: Mapping[str, Any], entries_by_index: dict[int, ShardManifestEntry], errors: list[dict[str, Any]]) -> None:
    if result.get("status") == "ok":
        entries_by_index[int(result["index"])] = ShardManifestEntry.from_mapping(dict(result["entry"]))
    elif result.get("status") != "empty":
        errors.append(dict(result))


def _window_shard_task(task: Mapping[str, Any]) -> dict[str, Any]:
    try:
        import duckdb  # type: ignore

        source_path = Path(str(task["source_path"]))
        target_path = Path(str(task["target_path"]))
        target_path.parent.mkdir(parents=True, exist_ok=True)
        if target_path.exists():
            target_path.unlink()
        source_shard = ShardManifestEntry.from_mapping(dict(task["source_shard"]))
        condition = _where_condition(str(task["start_date"]), dict(task.get("symbol_rules", {}) or {}), bool(task.get("has_symbol")))
        with duckdb.connect(":memory:") as con:
            con.execute(f"set memory_limit='{str(task['memory_limit'])}'")
            con.execute("set threads=1")
            con.execute(
                f"copy (select * from read_parquet({_sql_literal(str(source_path))}) where {condition}) "
                f"to {_sql_literal(str(target_path))} (format parquet)"
            )
            stats = con.execute(
                "select count(*) as n, "
                "strftime(min(try_cast(trade_date as date)), '%Y-%m-%d') as start_date, "
                "strftime(max(try_cast(trade_date as date)), '%Y-%m-%d') as end_date "
                f"from read_parquet({_sql_literal(str(target_path))})"
            ).fetchone()
        row_count = int(stats[0] or 0)
        if row_count <= 0:
            target_path.unlink(missing_ok=True)
            return {"status": "empty", "index": int(task["index"]), "source_path": str(source_path)}
        entry = ShardManifestEntry(
            path=f"datasets/{str(task['domain'])}/{str(task['target_dataset_id'])}/shards/{target_path.name}",
            row_count=row_count,
            start_date=str(stats[1] or ""),
            end_date=str(stats[2] or ""),
            status="stored",
            file_size=int(target_path.stat().st_size),
            schema_hash=schema_hash(_parquet_schema(target_path)),
            source_path=str(source_path.resolve()),
            content_key=source_shard.content_key,
            metadata={**dict(source_shard.metadata or {}), "research_window_rewritten": True},
        )
        return {"status": "ok", "index": int(task["index"]), "entry": entry.to_dict()}
    except Exception as exc:
        return {"status": "error", "index": int(task.get("index", -1)), "source_path": str(task.get("source_path", "")), "error": str(exc)}


def _may_have_symbol_before_cutoff(
    *,
    root: Path,
    shard: ShardManifestEntry,
    source_path: Path,
    symbol_rules: Mapping[str, str],
    memory_limit: str,
) -> bool:
    if not symbol_rules:
        return False
    possible = False
    for cutoff in symbol_rules.values():
        if not shard.start_date or shard.start_date < cutoff:
            possible = True
            break
    if not possible:
        return False
    import duckdb  # type: ignore

    clauses = [
        f"(upper(cast(symbol as varchar)) = {_sql_literal(symbol)} and try_cast(trade_date as date) < date {_sql_literal(cutoff)})"
        for symbol, cutoff in symbol_rules.items()
    ]
    with duckdb.connect(":memory:") as con:
        con.execute(f"set memory_limit='{memory_limit}'")
        con.execute("set threads=1")
        row = con.execute(
            f"select 1 from read_parquet({_sql_literal(str(source_path))}) "
            f"where {' or '.join(clauses)} limit 1"
        ).fetchone()
    return bool(row)


def _where_condition(start_date: str, symbol_rules: Mapping[str, str], has_symbol: bool) -> str:
    condition = f"try_cast(trade_date as date) >= date {_sql_literal(start_date)}"
    if has_symbol and symbol_rules:
        clauses = [
            f"(upper(cast(symbol as varchar)) = {_sql_literal(symbol)} and try_cast(trade_date as date) < date {_sql_literal(cutoff)})"
            for symbol, cutoff in symbol_rules.items()
        ]
        condition += f" and not ({' or '.join(clauses)})"
    return condition


def _selected_domains(active: Mapping[str, Any], domains: str) -> set[str]:
    if domains.strip():
        return {item.strip() for item in domains.split(",") if item.strip()}
    out: set[str] = set()
    for section in ("raw", "derived", "research_panels"):
        out.update(str(key) for key in dict(active.get(section, {}) or {}).keys())
    return out


def _column_names(manifest: DatasetManifest) -> set[str]:
    names = {str(item.get("name", "")) for item in list(manifest.schema or []) if item.get("name")}
    names.update(manifest.primary_key)
    return names


def _copy_or_link_file(source_path: Path, target_path: Path, *, hardlink: bool) -> None:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    if target_path.exists():
        target_path.unlink()
    if hardlink:
        try:
            os.link(source_path, target_path)
            return
        except OSError:
            pass
    shutil.copy2(source_path, target_path)


def _parquet_schema(path: Path) -> list[dict[str, str]]:
    import duckdb  # type: ignore

    with duckdb.connect(":memory:") as con:
        rows = con.execute(f"describe select * from read_parquet({_sql_literal(str(path))})").fetchall()
    return [{"name": str(row[0]), "type": str(row[1])} for row in rows]


def _sql_literal(value: str) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def _parse_symbol_starts(value: str) -> dict[str, str]:
    if not value.strip():
        return dict(DEFAULT_SYMBOL_STARTS)
    out: dict[str, str] = {}
    for item in value.split(","):
        if not item.strip():
            continue
        if ":" not in item:
            raise ValueError(f"invalid_symbol_start:{item}")
        symbol, date = item.split(":", 1)
        out[symbol.strip().upper()] = date.strip()
    return out


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="qdp research-window", description="Build qdp_v2 active datasets for a simplified continuous research window.")
    parser.add_argument("--workspace-root", default="")
    parser.add_argument("--start-date", default=DEFAULT_START_DATE)
    parser.add_argument("--symbol-starts", default="600036.SH:2016-07-25")
    parser.add_argument("--domains", default="")
    parser.add_argument("--runtime", choices=("safe", "balanced", "fast"), default="balanced")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--duckdb-memory-limit", default="")
    parser.add_argument("--copy-reused", dest="hardlink_reused", action="store_false")
    parser.add_argument("--activate", action="store_true")
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    payload = apply_research_window(
        workspace_root=str(args.workspace_root or "") or None,
        start_date=str(args.start_date),
        symbol_starts=_parse_symbol_starts(str(args.symbol_starts or "")),
        domains=str(args.domains or ""),
        runtime=str(args.runtime or "balanced"),
        workers=int(args.workers or 1),
        duckdb_memory_limit=str(args.duckdb_memory_limit or ""),
        activate=bool(args.activate),
        hardlink_reused=bool(args.hardlink_reused),
    )
    if bool(args.json):
        print(json.dumps(json_safe(payload), ensure_ascii=False, indent=2))
    else:
        print("\n".join(f"{key}: {value}" for key, value in payload.items() if key != "outputs"))
    return 0 if payload.get("status") == "ok" else 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
