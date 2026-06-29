from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from quant_data_platform.core.registry import load_memmap_registry, load_root_manifest
from quant_data_platform.core.paths import QdpPaths, qdp_paths
from quant_data_platform.lake.canonical import load_canonical_manifest
from quant_data_platform.lake.catalog import DEFAULT_DATA_LAKE_ROOT, ResearchDataLake


@dataclass(frozen=True)
class LakeGcConfig:
    lake_root: Path = DEFAULT_DATA_LAKE_ROOT
    workspace_root: Path | None = None
    dry_run: bool = True
    delete: bool = False
    yes: bool = False
    include_referenced: bool = False
    with_size: bool = False
    max_items: int = 200
    max_delete_items: int = 0
    max_delete_bytes: int = 0


@dataclass(frozen=True)
class LakeGcResult:
    status: str
    destructive_actions_performed: bool
    referenced_dataset_count: int
    dataset_dir_count: int
    unreferenced_dataset_dir_count: int
    unreferenced_bytes: int
    referenced_bytes: int
    deleted_dataset_dir_count: int = 0
    deleted_bytes: int = 0
    unreferenced: list[dict[str, Any]] = field(default_factory=list)
    referenced: list[dict[str, Any]] = field(default_factory=list)
    deleted: list[dict[str, Any]] = field(default_factory=list)


def lake_gc(config: LakeGcConfig) -> LakeGcResult:
    cfg = config
    if cfg.delete and not cfg.yes:
        raise ValueError("lake_gc_delete_requires_yes: pass --delete --yes")
    lake = ResearchDataLake(cfg.lake_root)
    referenced = _referenced_dataset_ids(lake, paths=qdp_paths(cfg.workspace_root) if cfg.workspace_root else None)
    inventory = _dataset_dir_inventory(lake, with_size=bool(cfg.with_size and not cfg.delete))
    unreferenced_items = [item for item in inventory if item["dataset_id"] not in referenced]
    referenced_items = [item for item in inventory if item["dataset_id"] in referenced]
    if cfg.delete:
        unreferenced_items = [_with_directory_size(item) for item in unreferenced_items]
    unreferenced_items = sorted(unreferenced_items, key=lambda item: int(item["bytes"]), reverse=True)
    referenced_items = sorted(referenced_items, key=lambda item: int(item["bytes"]), reverse=True)
    max_items = max(0, int(cfg.max_items or 0))
    deleted_items: list[dict[str, Any]] = []
    deleted_bytes = 0
    if cfg.delete:
        candidates = list(unreferenced_items)
        if cfg.max_delete_items > 0:
            candidates = candidates[: int(cfg.max_delete_items)]
        if cfg.max_delete_bytes > 0:
            bounded: list[dict[str, Any]] = []
            running = 0
            for item in candidates:
                size = int(item.get("bytes", 0) or 0)
                if running + size > int(cfg.max_delete_bytes):
                    continue
                bounded.append(item)
                running += size
            candidates = bounded
        deleted_items, deleted_bytes = _delete_unreferenced_dataset_dirs(lake, candidates)
        if deleted_items:
            _delete_catalog_rows(lake, deleted_items)
            lake.write_catalog_manifest()

    return LakeGcResult(
        status="deleted" if cfg.delete else "dry_run",
        destructive_actions_performed=bool(cfg.delete and deleted_items),
        referenced_dataset_count=len(referenced),
        dataset_dir_count=len(inventory),
        unreferenced_dataset_dir_count=len(unreferenced_items),
        unreferenced_bytes=sum(int(item["bytes"]) for item in unreferenced_items),
        referenced_bytes=sum(int(item["bytes"]) for item in referenced_items),
        deleted_dataset_dir_count=len(deleted_items),
        deleted_bytes=int(deleted_bytes),
        unreferenced=unreferenced_items[:max_items] if max_items else unreferenced_items,
        referenced=referenced_items[:max_items] if cfg.include_referenced and max_items else (referenced_items if cfg.include_referenced else []),
        deleted=deleted_items[:max_items] if max_items else deleted_items,
    )


def _referenced_dataset_ids(lake: ResearchDataLake, *, paths: QdpPaths | None = None) -> set[str]:
    paths = paths or qdp_paths()
    root = load_root_manifest(paths)
    canonical = load_canonical_manifest(lake)
    memmap_registry = load_memmap_registry(paths)
    direct: set[str] = set()
    _collect_dataset_ids(root, direct)
    _collect_dataset_ids(canonical, direct)
    _collect_dataset_ids(memmap_registry, direct)
    seen: set[str] = set()
    pending = list(direct)
    while pending:
        dataset_id = pending.pop()
        if dataset_id in seen:
            continue
        seen.add(dataset_id)
        try:
            metadata = lake.describe_dataset(dataset_id)
        except Exception:
            continue
        nested: set[str] = set()
        _collect_dataset_ids(metadata, nested)
        _collect_content_path_dataset_ids(lake, metadata, nested)
        for item in nested:
            if item not in seen:
                pending.append(item)
    return seen


def _collect_dataset_ids(value: Any, out: set[str]) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            key_text = str(key)
            if key_text.endswith("dataset_id") or key_text.endswith("dataset_ids") or key_text in {"sidecar_dataset_ids", "source_dataset_ids", "combined_from_dataset_ids"}:
                _collect_dataset_ids(item, out)
            else:
                _collect_dataset_ids(item, out)
        return
    if isinstance(value, (list, tuple, set)):
        for item in value:
            _collect_dataset_ids(item, out)
        return
    text = str(value or "").strip()
    if "__" not in text:
        return
    if text.startswith(("data_platform_", "policy_input_bundle__", "policy_pool_view__", "policy_sector_board_view__", "gold_training_dataset__")):
        out.add(text)


def _collect_content_path_dataset_ids(lake: ResearchDataLake, metadata: Mapping[str, Any], out: set[str]) -> None:
    content_paths = dict(metadata.get("content_paths", {}) or {})
    for value in content_paths.values():
        text = str(value or "").strip()
        if not text or "*" in text:
            continue
        inferred = _dataset_id_from_content_path(lake, Path(text))
        if inferred:
            out.add(inferred)
    shard_manifest = str(content_paths.get("shard_manifest", "") or "").strip()
    if not shard_manifest or not Path(shard_manifest).exists():
        return
    try:
        manifest = json.loads(Path(shard_manifest).read_text(encoding="utf-8"))
    except Exception:
        return
    _collect_dataset_ids(manifest, out)
    for shard in list(manifest.get("shards", []) or []):
        if not isinstance(shard, Mapping):
            continue
        for key in ("path", "file_path", "shard_path", "source_path", "source_feature_path"):
            text = str(shard.get(key, "") or "").strip()
            if not text:
                continue
            inferred = _dataset_id_from_content_path(lake, Path(text))
            if inferred:
                out.add(inferred)


def _dataset_id_from_content_path(lake: ResearchDataLake, path: Path) -> str:
    try:
        resolved = path.resolve()
        parquet_root = lake.parquet_root.resolve()
        relative = resolved.relative_to(parquet_root)
    except Exception:
        return ""
    parts = relative.parts
    if len(parts) < 3:
        return ""
    zone = parts[0]
    if zone in {"bronze_silver", "silver_view"}:
        return _dataset_id_from_dir(str(parts[1]), str(parts[2]))
    if zone == "gold" and len(parts) >= 4:
        return _dataset_id_from_dir(str(parts[1]), str(parts[3]))
    return ""


def _dataset_dir_inventory(lake: ResearchDataLake, *, with_size: bool) -> list[dict[str, Any]]:
    rows = lake.list_datasets()
    by_id: dict[str, dict[str, Any]] = {}
    for _, row in rows.iterrows():
        dataset_id = str(row.get("dataset_id", "") or "")
        if not dataset_id:
            continue
        by_id[dataset_id] = {
            "dataset_id": dataset_id,
            "dataset_kind": str(row.get("dataset_kind", "") or ""),
            "source": str(row.get("source", "") or ""),
            "start_date": str(row.get("start_date", "") or ""),
            "end_date": str(row.get("end_date", "") or ""),
            "status": str(row.get("status", "") or ""),
        }
    items: list[dict[str, Any]] = []
    for root in _dataset_dir_roots(lake):
        if not root.exists():
            continue
        for dataset_kind_dir in sorted([item for item in root.iterdir() if item.is_dir()], key=lambda item: item.name):
            for fingerprint_dir in _fingerprint_dirs(dataset_kind_dir):
                dataset_id = _dataset_id_from_dir(dataset_kind_dir.name, fingerprint_dir.name)
                size = _directory_size(fingerprint_dir) if with_size else 0
                meta = by_id.get(dataset_id, {})
                items.append(
                    {
                        "dataset_id": dataset_id,
                        "dataset_kind": str(meta.get("dataset_kind", "") or dataset_kind_dir.name),
                        "path": str(fingerprint_dir.resolve()),
                        "bytes": int(size),
                        "gb": round(size / 1024**3, 4),
                        "source": str(meta.get("source", "") or ""),
                        "start_date": str(meta.get("start_date", "") or ""),
                        "end_date": str(meta.get("end_date", "") or ""),
                        "catalog_status": str(meta.get("status", "") or ("cataloged" if dataset_id in by_id else "not_in_catalog")),
                    }
                )
    return items


def _dataset_dir_roots(lake: ResearchDataLake) -> list[Path]:
    return [
        lake.parquet_root / "bronze_silver",
        lake.parquet_root / "silver_view",
        lake.parquet_root / "gold",
    ]


def _fingerprint_dirs(dataset_kind_dir: Path) -> list[Path]:
    if dataset_kind_dir.parent.name == "gold":
        dirs: list[Path] = []
        for zone_dir in dataset_kind_dir.iterdir():
            if zone_dir.is_dir():
                dirs.extend([item for item in zone_dir.iterdir() if item.is_dir()])
        return sorted(dirs, key=lambda item: str(item))
    return sorted([item for item in dataset_kind_dir.iterdir() if item.is_dir()], key=lambda item: item.name)


def _dataset_id_from_dir(dataset_kind: str, fingerprint: str) -> str:
    if dataset_kind.startswith("data_platform_"):
        return f"{dataset_kind}__{fingerprint}"
    return f"{dataset_kind}__{fingerprint}"


def _directory_size(path: Path) -> int:
    total = 0
    for child in path.rglob("*"):
        try:
            if child.is_file():
                total += child.stat().st_size
        except OSError:
            continue
    return total


def _with_directory_size(item: dict[str, Any]) -> dict[str, Any]:
    size = _directory_size(Path(str(item.get("path", "") or "")))
    return {**item, "bytes": int(size), "gb": round(size / 1024**3, 4)}


def _delete_unreferenced_dataset_dirs(lake: ResearchDataLake, items: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    parquet_root = lake.parquet_root.resolve()
    deleted: list[dict[str, Any]] = []
    deleted_bytes = 0
    for item in items:
        path = Path(str(item.get("path", "") or "")).resolve()
        _assert_deletable_dataset_dir(path, parquet_root)
        size = int(item.get("bytes", 0) or 0)
        if size <= 0:
            size = _directory_size(path)
            item = {**item, "bytes": size, "gb": round(size / 1024**3, 4)}
        shutil.rmtree(path)
        deleted.append(dict(item))
        deleted_bytes += size
    return deleted, deleted_bytes


def _assert_deletable_dataset_dir(path: Path, parquet_root: Path) -> None:
    try:
        path.relative_to(parquet_root)
    except ValueError as exc:
        raise ValueError(f"lake_gc_refuses_path_outside_parquet_root: {path}") from exc
    if path == parquet_root or path.parent == parquet_root:
        raise ValueError(f"lake_gc_refuses_root_delete: {path}")
    if not path.exists():
        raise ValueError(f"lake_gc_delete_path_missing: {path}")
    if not path.is_dir():
        raise ValueError(f"lake_gc_delete_path_not_directory: {path}")


def _delete_catalog_rows(lake: ResearchDataLake, items: list[dict[str, Any]]) -> None:
    dataset_ids = sorted({str(item.get("dataset_id", "") or "") for item in items if str(item.get("dataset_id", "") or "")})
    fingerprints = sorted({str(Path(str(item.get("path", "") or "")).name) for item in items if str(item.get("path", "") or "")})
    if not dataset_ids and not fingerprints:
        return
    with lake._connect() as con:
        for dataset_id in dataset_ids:
            con.execute("delete from datasets where dataset_id = ?", [dataset_id])
        for fingerprint in fingerprints:
            con.execute("delete from datasets where fingerprint = ?", [fingerprint])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="QDP data lake garbage collection inventory and cleanup.")
    parser.add_argument("--lake-root", default=str(DEFAULT_DATA_LAKE_ROOT))
    parser.add_argument("--workspace-root", default="")
    parser.add_argument("--dry-run", action="store_true", default=True)
    parser.add_argument("--delete", action="store_true", help="Delete unreferenced dataset directories. Requires --yes.")
    parser.add_argument("--yes", action="store_true", help="Confirm destructive deletion when used with --delete.")
    parser.add_argument("--include-referenced", action="store_true")
    parser.add_argument("--with-size", action="store_true", help="Recursively stat dataset directories; slow on large lakes.")
    parser.add_argument("--max-items", type=int, default=200)
    parser.add_argument("--max-delete-items", type=int, default=0, help="Limit deleted candidates after size-desc sorting; 0 means no item limit.")
    parser.add_argument("--max-delete-gb", type=float, default=0.0, help="Maximum total bytes to delete; 0 means no byte limit.")
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = lake_gc(
        LakeGcConfig(
            lake_root=Path(args.lake_root),
            workspace_root=Path(args.workspace_root) if str(args.workspace_root or "").strip() else None,
            dry_run=not bool(args.delete),
            delete=bool(args.delete),
            yes=bool(args.yes),
            include_referenced=bool(args.include_referenced),
            with_size=bool(args.with_size or args.delete),
            max_items=int(args.max_items),
            max_delete_items=int(args.max_delete_items),
            max_delete_bytes=int(float(args.max_delete_gb or 0.0) * 1024**3),
        )
    )
    payload = result.__dict__
    print(json.dumps(payload, ensure_ascii=False, indent=2 if args.json else None, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
