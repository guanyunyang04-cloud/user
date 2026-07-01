from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

from quant_data_platform.core.json_io import json_safe
from quant_data_platform.qdp_v2.manifest import iter_dataset_manifests, qdp_v2_root, read_active_manifest, read_dataset_manifest
from quant_data_platform.qdp_v2.status import _active_dataset_refs


def lake_gc(
    *,
    workspace_root: str | Path | None = None,
    delete: bool = False,
    yes: bool = False,
    with_size: bool = False,
    max_items: int = 200,
) -> dict[str, Any]:
    if delete and not yes:
        raise ValueError("qdp_v2_gc_delete_requires_yes")
    root = qdp_v2_root(workspace_root)
    active = read_active_manifest(root)
    referenced = {dataset_id for _, _, dataset_id in _active_dataset_refs(active)}
    inventory = _dataset_dir_inventory(root, with_size=with_size or delete)
    unreferenced = [item for item in inventory if item["dataset_id"] not in referenced]
    referenced_items = [item for item in inventory if item["dataset_id"] in referenced]
    unreferenced = sorted(unreferenced, key=lambda item: int(item["bytes"]), reverse=True)
    referenced_items = sorted(referenced_items, key=lambda item: int(item["bytes"]), reverse=True)
    deleted: list[dict[str, Any]] = []
    if delete:
        for item in unreferenced:
            path = Path(str(item["path"]))
            if path.exists() and _is_inside(path, root / "datasets"):
                shutil.rmtree(path)
                deleted.append(item)
    limit = max(0, int(max_items or 0))
    return {
        "status": "deleted" if delete else "dry_run",
        "qdp_v2_root": str(root.resolve()),
        "destructive_actions_performed": bool(delete and deleted),
        "referenced_dataset_count": len(referenced),
        "dataset_dir_count": len(inventory),
        "unreferenced_dataset_dir_count": len(unreferenced),
        "unreferenced_bytes": sum(int(item["bytes"]) for item in unreferenced),
        "referenced_bytes": sum(int(item["bytes"]) for item in referenced_items),
        "deleted_dataset_dir_count": len(deleted),
        "deleted_bytes": sum(int(item["bytes"]) for item in deleted),
        "unreferenced": unreferenced[:limit] if limit else unreferenced,
        "deleted": deleted[:limit] if limit else deleted,
    }


def _dataset_dir_inventory(root: Path, *, with_size: bool) -> list[dict[str, Any]]:
    datasets_root = root / "datasets"
    if not datasets_root.exists():
        return []
    manifest_by_dir: dict[Path, dict[str, Any]] = {}
    for manifest_path in iter_dataset_manifests(root):
        manifest = read_dataset_manifest(manifest_path)
        manifest_by_dir[manifest_path.parent] = {
            "dataset_id": manifest.dataset_id,
            "domain": manifest.domain,
            "start_date": manifest.start_date,
            "end_date": manifest.end_date,
            "row_count": manifest.row_count,
        }
    items: list[dict[str, Any]] = []
    for domain_dir in sorted([item for item in datasets_root.iterdir() if item.is_dir()], key=lambda item: item.name):
        for dataset_dir in sorted([item for item in domain_dir.iterdir() if item.is_dir()], key=lambda item: item.name):
            if dataset_dir.name.startswith("."):
                continue
            meta = manifest_by_dir.get(dataset_dir, {})
            dataset_id = str(meta.get("dataset_id", "") or dataset_dir.name)
            size = _directory_size(dataset_dir) if with_size else 0
            items.append(
                {
                    "dataset_id": dataset_id,
                    "domain": str(meta.get("domain", "") or domain_dir.name),
                    "path": str(dataset_dir.resolve()),
                    "bytes": int(size),
                    "gb": round(size / 1024**3, 4),
                    "start_date": str(meta.get("start_date", "") or ""),
                    "end_date": str(meta.get("end_date", "") or ""),
                    "row_count": int(meta.get("row_count", 0) or 0),
                    "manifest_exists": bool(meta),
                }
            )
    return items


def _directory_size(path: Path) -> int:
    total = 0
    for child in path.rglob("*"):
        if child.is_file():
            try:
                total += int(child.stat().st_size)
            except OSError:
                pass
    return total


def _is_inside(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="qdp gc", description="qdp_v2 manifest-first GC.")
    parser.add_argument("--workspace-root", default="")
    parser.add_argument("--dry-run", action="store_true", default=True)
    parser.add_argument("--delete", action="store_true")
    parser.add_argument("--yes", action="store_true")
    parser.add_argument("--with-size", action="store_true")
    parser.add_argument("--max-items", type=int, default=200)
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    payload = lake_gc(
        workspace_root=str(args.workspace_root or "") or None,
        delete=bool(args.delete),
        yes=bool(args.yes),
        with_size=bool(args.with_size),
        max_items=int(args.max_items or 0),
    )
    if bool(args.json):
        print(json.dumps(json_safe(payload), ensure_ascii=False, indent=2))
    else:
        print(_format(payload))
    return 0


def _format(payload: dict[str, Any]) -> str:
    lines = [
        f"status: {payload.get('status')}",
        f"qdp_v2_root: {payload.get('qdp_v2_root')}",
        f"referenced_dataset_count: {payload.get('referenced_dataset_count', 0)}",
        f"unreferenced_dataset_dir_count: {payload.get('unreferenced_dataset_dir_count', 0)}",
        f"unreferenced_gb: {round(int(payload.get('unreferenced_bytes', 0)) / 1024**3, 4)}",
    ]
    for item in list(payload.get("unreferenced", []) or [])[:20]:
        lines.append(f"unreferenced: {item.get('domain')} {item.get('dataset_id')} {item.get('gb')}GB {item.get('path')}")
    return "\n".join(lines)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
