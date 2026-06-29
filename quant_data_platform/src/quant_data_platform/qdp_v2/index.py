from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from quant_data_platform.core.json_io import json_safe
from quant_data_platform.qdp_v2.manifest import atomic_write_json, iter_dataset_manifests, qdp_v2_root, read_dataset_manifest, utc_now


def rebuild_index(*, workspace_root: str | Path | None = None) -> dict[str, Any]:
    root = qdp_v2_root(workspace_root)
    index_dir = root / "index"
    index_dir.mkdir(parents=True, exist_ok=True)
    catalog = index_dir / "catalog.duckdb"
    try:
        import duckdb  # type: ignore
    except Exception as exc:
        return {"status": "error", "qdp_v2_root": str(root.resolve()), "error": f"duckdb_unavailable:{exc}"}
    dataset_rows: list[tuple[Any, ...]] = []
    shard_rows: list[tuple[Any, ...]] = []
    for path in iter_dataset_manifests(root):
        manifest = read_dataset_manifest(path)
        dataset_rows.append(
            (
                manifest.dataset_id,
                manifest.domain,
                manifest.layer,
                manifest.frequency,
                manifest.contract_version,
                manifest.start_date,
                manifest.end_date,
                manifest.row_count,
                len(manifest.shards),
                manifest.schema_hash,
                str(path.resolve()),
            )
        )
        for index, shard in enumerate(manifest.shards):
            shard_rows.append(
                (
                    manifest.dataset_id,
                    manifest.domain,
                    index,
                    shard.path,
                    shard.row_count,
                    shard.start_date,
                    shard.end_date,
                    shard.file_size,
                    shard.schema_hash,
                    shard.status,
                )
            )
    with duckdb.connect(str(catalog)) as con:
        con.execute("drop table if exists datasets")
        con.execute("drop table if exists shards")
        con.execute(
            """
            create table datasets (
                dataset_id varchar,
                domain varchar,
                layer varchar,
                frequency varchar,
                contract_version varchar,
                start_date varchar,
                end_date varchar,
                row_count bigint,
                shard_count integer,
                schema_hash varchar,
                manifest_path varchar
            )
            """
        )
        con.execute(
            """
            create table shards (
                dataset_id varchar,
                domain varchar,
                shard_index integer,
                path varchar,
                row_count bigint,
                start_date varchar,
                end_date varchar,
                file_size bigint,
                schema_hash varchar,
                status varchar
            )
            """
        )
        if dataset_rows:
            con.executemany("insert into datasets values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", dataset_rows)
        if shard_rows:
            con.executemany("insert into shards values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", shard_rows)
    payload = {
        "status": "ok",
        "qdp_v2_root": str(root.resolve()),
        "catalog_path": str(catalog.resolve()),
        "dataset_count": len(dataset_rows),
        "shard_count": len(shard_rows),
        "rebuilt_at": utc_now(),
        "role": "rebuildable_query_index",
    }
    atomic_write_json(index_dir / "catalog_manifest.json", payload)
    return payload


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="qdp index", description="Manage qdp_v2 rebuildable query index.")
    parser.add_argument("--workspace-root", default="")
    sub = parser.add_subparsers(dest="index_command", required=True)
    rebuild = sub.add_parser("rebuild")
    rebuild.add_argument("--json", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if args.index_command != "rebuild":
        raise ValueError(f"unsupported_index_command:{args.index_command}")
    payload = rebuild_index(workspace_root=str(args.workspace_root or "") or None)
    if bool(args.json):
        print(json.dumps(json_safe(payload), ensure_ascii=False, indent=2))
    else:
        print("\n".join(f"{key}: {value}" for key, value in payload.items()))
    return 0 if payload.get("status") == "ok" else 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
