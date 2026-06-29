from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

from quant_data_platform.core.json_io import json_safe
from quant_data_platform.qdp_v2.manifest import (
    DatasetManifest,
    ShardManifestEntry,
    atomic_write_json,
    dataset_manifest_for_id,
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


def derive_5m_from_1m(
    *,
    workspace_root: str | Path | None = None,
    source_dataset_id: str = "",
    max_shards: int = 0,
    runtime: str = "balanced",
    activate_domain: bool = False,
) -> dict[str, Any]:
    root = qdp_v2_root(workspace_root)
    source_manifest = _resolve_source_manifest(root, source_dataset_id, "market_intraday_1m")
    source = read_dataset_manifest(source_manifest)
    profile = resolve_runtime_profile(runtime)
    target_dataset_id = f"market_intraday_5m__{stable_hash({'source': source.dataset_id, 'contract': 'mootdx_5m_48_v1'})}"
    target_dir = root / "datasets" / "market_intraday_5m" / target_dataset_id
    staging = target_dir / ".staging"
    if staging.exists():
        shutil.rmtree(staging)
    staging_shards = staging / "shards"
    staging_shards.mkdir(parents=True, exist_ok=True)
    processed = 0
    shard_entries: list[ShardManifestEntry] = []
    errors: list[dict[str, Any]] = []
    for index, shard in enumerate(source.shards):
        if max_shards and processed >= max_shards:
            break
        source_path = root / shard.path if not Path(shard.path).is_absolute() else Path(shard.path)
        target_path = staging_shards / f"part_{index:06d}_5m.parquet"
        try:
            stats = _derive_5m_shard(source_path=source_path, target_path=target_path, memory_limit=profile.duckdb_memory_limit, threads=profile.duckdb_threads)
            if int(stats["row_count"]) <= 0:
                continue
            shard_entries.append(
                ShardManifestEntry(
                    path=f"datasets/market_intraday_5m/{target_dataset_id}/shards/{target_path.name}",
                    row_count=int(stats["row_count"]),
                    start_date=str(stats.get("start_date", "") or shard.start_date),
                    end_date=str(stats.get("end_date", "") or shard.end_date),
                    status="stored",
                    file_size=int(target_path.stat().st_size),
                    schema_hash=str(stats.get("schema_hash", "") or ""),
                    source_path=str(source_path.resolve()),
                    content_key="derived_from_1m_240",
                )
            )
            processed += 1
        except Exception as exc:
            errors.append({"source_path": str(source_path), "target_path": str(target_path), "error": str(exc)})
    if errors:
        return {"status": "error", "target_dataset_id": target_dataset_id, "processed_shards": processed, "errors": errors[:50], "error_count": len(errors)}
    final_shards = target_dir / "shards"
    target_dir.mkdir(parents=True, exist_ok=True)
    if final_shards.exists():
        shutil.rmtree(final_shards)
    if staging_shards.exists():
        staging_shards.replace(final_shards)
    if staging.exists():
        shutil.rmtree(staging, ignore_errors=True)
    schema = _duckdb_schema(final_shards / shard_entries[0].path.split("/")[-1]) if shard_entries else []
    manifest = DatasetManifest(
        dataset_id=target_dataset_id,
        domain="market_intraday_5m",
        layer="raw",
        frequency="5m",
        contract_version="mootdx_5m_48_v1",
        primary_key=["trade_date", "symbol", "bar_time"],
        start_date=min([item.start_date for item in shard_entries if item.start_date], default=""),
        end_date=max([item.end_date for item in shard_entries if item.end_date], default=""),
        row_count=sum(int(item.row_count) for item in shard_entries),
        schema_hash=schema_hash(schema),
        schema=schema,
        shards=shard_entries,
        source={"provider": "qdp_v2", "created_by": "derive_5m_from_1m", "created_at": utc_now(), "source_dataset_id": source.dataset_id},
        quality={"path_refs_exist": True, "bar_count_contract": "48", "primary_key_unique": "not_checked"},
        notes=["derived from mootdx-style 1m 240 bars using 5-minute end labels"],
    )
    manifest_path = write_dataset_manifest(root, manifest)
    active_path = ""
    if activate_domain:
        active = read_active_manifest(root)
        if active:
            raw = dict(active.get("raw", {}) or {})
            raw["market_intraday_5m"] = target_dataset_id
            active["raw"] = raw
            active_path = str(write_active_manifest(root, active).resolve())
    payload = {
        "status": "ok",
        "target_dataset_id": target_dataset_id,
        "manifest_path": str(manifest_path.resolve()),
        "source_dataset_id": source.dataset_id,
        "processed_shards": processed,
        "row_count": manifest.row_count,
        "start_date": manifest.start_date,
        "end_date": manifest.end_date,
        "active_manifest": active_path,
        "max_shards": int(max_shards or 0),
    }
    atomic_write_json(root / "runs" / f"derive_5m_{utc_now().replace(':', '').replace('-', '')}.json", payload)
    return payload


def normalize_valuation(
    *,
    workspace_root: str | Path | None = None,
    source_dataset_id: str = "",
    max_shards: int = 0,
    runtime: str = "balanced",
    activate_domain: bool = False,
) -> dict[str, Any]:
    root = qdp_v2_root(workspace_root)
    source_manifest = _resolve_source_manifest(root, source_dataset_id, "valuation")
    source = read_dataset_manifest(source_manifest)
    profile = resolve_runtime_profile(runtime)
    target_dataset_id = f"valuation__{stable_hash({'source': source.dataset_id, 'contract': 'qdp_v2_valuation_v1'})}"
    target_dir = root / "datasets" / "valuation" / target_dataset_id
    staging = target_dir / ".staging"
    if staging.exists():
        shutil.rmtree(staging)
    staging_shards = staging / "shards"
    staging_shards.mkdir(parents=True, exist_ok=True)
    processed = 0
    entries: list[ShardManifestEntry] = []
    errors: list[dict[str, Any]] = []
    for index, shard in enumerate(source.shards):
        if max_shards and processed >= max_shards:
            break
        source_path = root / shard.path if not Path(shard.path).is_absolute() else Path(shard.path)
        target_path = staging_shards / f"part_{index:06d}_valuation.parquet"
        try:
            stats = _normalize_valuation_shard(source_path=source_path, target_path=target_path, memory_limit=profile.duckdb_memory_limit, threads=profile.duckdb_threads)
            entries.append(
                ShardManifestEntry(
                    path=f"datasets/valuation/{target_dataset_id}/shards/{target_path.name}",
                    row_count=int(stats["row_count"]),
                    start_date=str(stats.get("start_date", "") or shard.start_date),
                    end_date=str(stats.get("end_date", "") or shard.end_date),
                    status="stored",
                    file_size=int(target_path.stat().st_size),
                    schema_hash=str(stats.get("schema_hash", "") or ""),
                    source_path=str(source_path.resolve()),
                    content_key="normalized_valuation",
                )
            )
            processed += 1
        except Exception as exc:
            errors.append({"source_path": str(source_path), "target_path": str(target_path), "error": str(exc)})
    if errors:
        return {"status": "error", "target_dataset_id": target_dataset_id, "processed_shards": processed, "errors": errors[:50], "error_count": len(errors)}
    final_shards = target_dir / "shards"
    target_dir.mkdir(parents=True, exist_ok=True)
    if final_shards.exists():
        shutil.rmtree(final_shards)
    if staging_shards.exists():
        staging_shards.replace(final_shards)
    if staging.exists():
        shutil.rmtree(staging, ignore_errors=True)
    schema = _duckdb_schema(final_shards / entries[0].path.split("/")[-1]) if entries else []
    manifest = DatasetManifest(
        dataset_id=target_dataset_id,
        domain="valuation",
        layer="raw",
        frequency="1d",
        contract_version="qdp_v2_valuation_v1",
        primary_key=["trade_date", "symbol"],
        start_date=min([item.start_date for item in entries if item.start_date], default=""),
        end_date=max([item.end_date for item in entries if item.end_date], default=""),
        row_count=sum(int(item.row_count) for item in entries),
        schema_hash=schema_hash(schema),
        schema=schema,
        shards=entries,
        source={"provider": "qdp_v2", "created_by": "normalize_valuation", "created_at": utc_now(), "source_dataset_id": source.dataset_id},
        quality={"path_refs_exist": True, "schema_contract": "qdp_v2_valuation_v1", "primary_key_unique": "not_checked"},
    )
    manifest_path = write_dataset_manifest(root, manifest)
    active_path = ""
    if activate_domain:
        active = read_active_manifest(root)
        if active:
            raw = dict(active.get("raw", {}) or {})
            raw["valuation"] = target_dataset_id
            active["raw"] = raw
            active_path = str(write_active_manifest(root, active).resolve())
    payload = {
        "status": "ok",
        "target_dataset_id": target_dataset_id,
        "manifest_path": str(manifest_path.resolve()),
        "source_dataset_id": source.dataset_id,
        "processed_shards": processed,
        "row_count": manifest.row_count,
        "start_date": manifest.start_date,
        "end_date": manifest.end_date,
        "active_manifest": active_path,
        "max_shards": int(max_shards or 0),
    }
    atomic_write_json(root / "runs" / f"normalize_valuation_{utc_now().replace(':', '').replace('-', '')}.json", payload)
    return payload


def _resolve_source_manifest(root: Path, dataset_id: str, domain: str) -> Path:
    if dataset_id:
        path = dataset_manifest_for_id(root, dataset_id, domain)
        if path is None:
            raise FileNotFoundError(f"dataset_manifest_not_found:{domain}:{dataset_id}")
        return path
    active = read_active_manifest(root)
    active_id = str(dict(active.get("raw", {}) or {}).get(domain, "") or "")
    if active_id:
        path = dataset_manifest_for_id(root, active_id, domain)
        if path is not None:
            return path
    candidates = sorted((root / "datasets" / domain).glob("*/dataset.json"))
    if len(candidates) == 1:
        return candidates[0]
    raise FileNotFoundError(f"source_dataset_required:{domain}")


def _derive_5m_shard(*, source_path: Path, target_path: Path, memory_limit: str, threads: int) -> dict[str, Any]:
    import duckdb  # type: ignore

    target_path.parent.mkdir(parents=True, exist_ok=True)
    source_literal = _sql_literal(str(source_path))
    target_literal = _sql_literal(str(target_path))
    with duckdb.connect(":memory:") as con:
        con.execute(f"set memory_limit='{memory_limit}'")
        con.execute(f"set threads={max(1, int(threads))}")
        columns = {row[0] for row in con.execute("describe select * from read_parquet(?)", [str(source_path)]).fetchall()}
        source_expr = "any_value(source) as source" if "source" in columns else "'derived_from_1m_240' as source"
        adjusted_expr = "any_value(adjusted_flag) as adjusted_flag" if "adjusted_flag" in columns else "'' as adjusted_flag"
        sql = f"""
        copy (
            with raw as (
                select
                    *,
                    regexp_replace(cast(bar_time as varchar), '[^0-9]', '', 'g') as bt
                from read_parquet({source_literal})
            ),
            parsed as (
                select
                    *,
                    cast(substr(bt, 1, 2) as integer) * 60 + cast(substr(bt, 3, 2) as integer) as minute_of_day
                from raw
                where length(bt) >= 4
            ),
            bucketed as (
                select
                    *,
                    case
                        when minute_of_day between 571 and 690 then 570 + cast(ceil((minute_of_day - 570) / 5.0) * 5 as integer)
                        when minute_of_day between 781 and 900 then 780 + cast(ceil((minute_of_day - 780) / 5.0) * 5 as integer)
                        else null
                    end as bucket_end
                from parsed
            ),
            grouped as (
                select
                    symbol,
                    trade_date,
                    printf('%02d%02d00000', cast(floor(bucket_end / 60) as integer), cast(bucket_end % 60 as integer)) as bar_time,
                    arg_min(open, minute_of_day) as open,
                    max(high) as high,
                    min(low) as low,
                    arg_max(close, minute_of_day) as close,
                    sum(volume) as volume,
                    sum(amount) as amount,
                    {source_expr},
                    {adjusted_expr}
                from bucketed
                where bucket_end is not null
                group by symbol, trade_date, bucket_end
            )
            select * from grouped order by trade_date, symbol, bar_time
        ) to {target_literal} (format parquet)
        """
        con.execute(sql)
        row = con.execute("select count(*) as n, min(trade_date) as start_date, max(trade_date) as end_date from read_parquet(?)", [str(target_path)]).fetchone()
    return {
        "row_count": int(row[0] or 0),
        "start_date": str(row[1] or ""),
        "end_date": str(row[2] or ""),
        "schema_hash": schema_hash(_duckdb_schema(target_path)),
    }


def _normalize_valuation_shard(*, source_path: Path, target_path: Path, memory_limit: str, threads: int) -> dict[str, Any]:
    import duckdb  # type: ignore

    target_path.parent.mkdir(parents=True, exist_ok=True)
    source_literal = _sql_literal(str(source_path))
    target_literal = _sql_literal(str(target_path))
    with duckdb.connect(":memory:") as con:
        con.execute(f"set memory_limit='{memory_limit}'")
        con.execute(f"set threads={max(1, int(threads))}")
        cols = {str(row[0]) for row in con.execute("describe select * from read_parquet(?)", [str(source_path)]).fetchall()}
        exprs = [
            _column_expr(cols, "symbol", aliases=("code", "stock_code")),
            _column_expr(cols, "trade_date", aliases=("date",)),
            _column_expr(cols, "total_mv", aliases=("totalMarketValue", "total_mv")),
            _column_expr(cols, "circ_mv", aliases=("circulatingMarketValue", "circ_mv")),
            _column_expr(cols, "pe", aliases=("peTTM", "pe_ttm", "pe")),
            _column_expr(cols, "pb", aliases=("pbMRQ", "pb_mrq", "pb")),
            _column_expr(cols, "turnover_rate", aliases=("turn", "turnover", "turnover_rate")),
        ]
        source_expr = "source" if "source" in cols else "'legacy'"
        sql = f"""
        copy (
            select
                {exprs[0]} as symbol,
                {exprs[1]} as trade_date,
                cast({exprs[2]} as double) as total_mv,
                cast({exprs[3]} as double) as circ_mv,
                cast({exprs[4]} as double) as pe,
                cast({exprs[5]} as double) as pb,
                cast({exprs[6]} as double) as turnover_rate,
                cast({source_expr} as varchar) as source
            from read_parquet({source_literal})
        ) to {target_literal} (format parquet)
        """
        con.execute(sql)
        row = con.execute("select count(*) as n, min(trade_date) as start_date, max(trade_date) as end_date from read_parquet(?)", [str(target_path)]).fetchone()
    return {
        "row_count": int(row[0] or 0),
        "start_date": str(row[1] or ""),
        "end_date": str(row[2] or ""),
        "schema_hash": schema_hash(_duckdb_schema(target_path)),
    }


def _column_expr(columns: set[str], canonical: str, *, aliases: tuple[str, ...]) -> str:
    for name in (canonical, *aliases):
        if name in columns:
            return name
    return "null"


def _sql_literal(value: str) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def _duckdb_schema(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    import duckdb  # type: ignore

    with duckdb.connect(":memory:") as con:
        rows = con.execute("describe select * from read_parquet(?)", [str(path)]).fetchall()
    return [{"name": str(row[0]), "type": str(row[1])} for row in rows]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="qdp clean", description="qdp_v2 data cleaning and standardization commands.")
    parser.add_argument("--workspace-root", default="")
    parser.add_argument("--runtime", default="balanced", choices=("safe", "balanced", "fast"))
    sub = parser.add_subparsers(dest="clean_command", required=True)
    five = sub.add_parser("5m-from-1m", help="Derive standard mootdx_5m_48_v1 shards from qdp_v2 1m 240 shards.")
    five.add_argument("--source-dataset-id", default="")
    five.add_argument("--max-shards", type=int, default=0)
    five.add_argument("--activate-domain", action="store_true")
    five.add_argument("--json", action="store_true")
    valuation = sub.add_parser("valuation", help="Normalize valuation shards to qdp_v2 valuation schema.")
    valuation.add_argument("--source-dataset-id", default="")
    valuation.add_argument("--max-shards", type=int, default=0)
    valuation.add_argument("--activate-domain", action="store_true")
    valuation.add_argument("--json", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    workspace = str(args.workspace_root or "") or None
    if args.clean_command == "5m-from-1m":
        payload = derive_5m_from_1m(
            workspace_root=workspace,
            source_dataset_id=str(args.source_dataset_id or ""),
            max_shards=int(args.max_shards or 0),
            runtime=str(args.runtime or "balanced"),
            activate_domain=bool(args.activate_domain),
        )
    elif args.clean_command == "valuation":
        payload = normalize_valuation(
            workspace_root=workspace,
            source_dataset_id=str(args.source_dataset_id or ""),
            max_shards=int(args.max_shards or 0),
            runtime=str(args.runtime or "balanced"),
            activate_domain=bool(args.activate_domain),
        )
    else:
        raise ValueError(f"unsupported_clean_command:{args.clean_command}")
    if bool(args.json):
        print(json.dumps(json_safe(payload), ensure_ascii=False, indent=2))
    else:
        print("\n".join(f"{key}: {value}" for key, value in payload.items()))
    return 0 if payload.get("status") == "ok" else 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
