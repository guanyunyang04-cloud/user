from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from quant_data_platform.core.json_io import json_safe
from quant_data_platform.qdp_v2.manifest import (
    dataset_manifest_for_id,
    iter_dataset_manifests,
    qdp_v2_root,
    read_dataset_manifest,
    resolve_manifest_path,
)


def list_datasets(*, workspace_root: str | Path | None = None) -> dict[str, Any]:
    root = qdp_v2_root(workspace_root)
    rows: list[dict[str, Any]] = []
    for path in iter_dataset_manifests(root):
        manifest = read_dataset_manifest(path)
        rows.append(
            {
                "dataset_id": manifest.dataset_id,
                "domain": manifest.domain,
                "layer": manifest.layer,
                "frequency": manifest.frequency,
                "contract_version": manifest.contract_version,
                "start_date": manifest.start_date,
                "end_date": manifest.end_date,
                "row_count": manifest.row_count,
                "shard_count": len(manifest.shards),
                "manifest_path": str(path.resolve()),
            }
        )
    return {
        "status": "ok",
        "qdp_v2_root": str(root.resolve()),
        "dataset_count": len(rows),
        "datasets": sorted(rows, key=lambda item: (str(item["domain"]), str(item["dataset_id"]))),
    }


def describe_dataset(dataset_id: str, *, workspace_root: str | Path | None = None, domain: str = "") -> dict[str, Any]:
    root = qdp_v2_root(workspace_root)
    path = dataset_manifest_for_id(root, dataset_id, domain)
    if path is None:
        return {"status": "not_found", "dataset_id": dataset_id, "qdp_v2_root": str(root.resolve())}
    manifest = read_dataset_manifest(path)
    payload = manifest.to_dict()
    payload["status"] = "ok"
    payload["manifest_path"] = str(path.resolve())
    return payload


def validate_dataset(dataset_id: str, *, workspace_root: str | Path | None = None, domain: str = "") -> dict[str, Any]:
    root = qdp_v2_root(workspace_root)
    path = dataset_manifest_for_id(root, dataset_id, domain)
    if path is None:
        return {"status": "error", "dataset_id": dataset_id, "errors": ["dataset_manifest_missing"]}
    manifest = read_dataset_manifest(path)
    errors: list[str] = []
    existing = 0
    footer_rows = 0
    for shard in manifest.shards:
        shard_path = resolve_manifest_path(shard.path, root=root)
        if not shard_path.exists():
            errors.append(f"shard_missing:{shard.path}")
            continue
        existing += 1
        try:
            rows = _parquet_row_count(shard_path)
            footer_rows += int(rows)
            if shard.row_count > 0 and rows > 0 and int(rows) != int(shard.row_count):
                errors.append(f"row_count_mismatch:{shard.path}:manifest={shard.row_count}:footer={rows}")
        except Exception as exc:
            errors.append(f"footer_unreadable:{shard.path}:{exc}")
    if manifest.row_count and footer_rows and int(manifest.row_count) != int(footer_rows):
        errors.append(f"dataset_row_count_mismatch:manifest={manifest.row_count}:footer={footer_rows}")
    return {
        "status": "ok" if not errors else "error",
        "dataset_id": manifest.dataset_id,
        "domain": manifest.domain,
        "manifest_path": str(path.resolve()),
        "shard_count": len(manifest.shards),
        "existing_shards": existing,
        "row_count": manifest.row_count,
        "footer_row_count": footer_rows,
        "errors": errors[:200],
        "error_count": len(errors),
    }


def _parquet_row_count(path: Path) -> int:
    try:
        import pyarrow.parquet as pq  # type: ignore

        return int(pq.ParquetFile(path).metadata.num_rows)
    except Exception:
        import duckdb  # type: ignore

        with duckdb.connect(":memory:") as con:
            return int(con.execute("select count(*) as n from read_parquet(?)", [str(path)]).fetchone()[0])


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="qdp dataset", description="Inspect qdp_v2 dataset manifests.")
    parser.add_argument("--workspace-root", default="")
    sub = parser.add_subparsers(dest="dataset_command", required=True)
    list_cmd = sub.add_parser("list")
    list_cmd.add_argument("--json", action="store_true")
    describe_cmd = sub.add_parser("describe")
    describe_cmd.add_argument("dataset_id")
    describe_cmd.add_argument("--domain", default="")
    describe_cmd.add_argument("--json", action="store_true")
    validate_cmd = sub.add_parser("validate")
    validate_cmd.add_argument("dataset_id")
    validate_cmd.add_argument("--domain", default="")
    validate_cmd.add_argument("--json", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    workspace = str(args.workspace_root or "") or None
    if args.dataset_command == "list":
        payload = list_datasets(workspace_root=workspace)
    elif args.dataset_command == "describe":
        payload = describe_dataset(str(args.dataset_id), workspace_root=workspace, domain=str(args.domain or ""))
    elif args.dataset_command == "validate":
        payload = validate_dataset(str(args.dataset_id), workspace_root=workspace, domain=str(args.domain or ""))
    else:
        raise ValueError(f"unsupported_dataset_command:{args.dataset_command}")
    if bool(getattr(args, "json", False)):
        print(json.dumps(json_safe(payload), ensure_ascii=False, indent=2))
    else:
        print(_format(payload))
    return 0 if str(payload.get("status", "")) == "ok" else 2


def _format(payload: dict[str, Any]) -> str:
    if "datasets" in payload:
        lines = [f"status: {payload.get('status')}", f"dataset_count: {payload.get('dataset_count', 0)}"]
        for item in list(payload.get("datasets", []) or []):
            lines.append(
                f"{item.get('domain')} {item.get('dataset_id')} "
                f"{item.get('start_date', '')}..{item.get('end_date', '')} "
                f"rows={item.get('row_count', 0)} shards={item.get('shard_count', 0)}"
            )
        return "\n".join(lines)
    if payload.get("status") == "not_found":
        return f"status: not_found\ndataset_id: {payload.get('dataset_id')}"
    return "\n".join(f"{key}: {json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else value}" for key, value in payload.items())


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
