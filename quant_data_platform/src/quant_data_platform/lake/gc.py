from __future__ import annotations

import argparse
import json
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
    include_referenced: bool = False
    with_size: bool = False
    max_items: int = 200


@dataclass(frozen=True)
class LakeGcResult:
    status: str
    destructive_actions_performed: bool
    referenced_dataset_count: int
    dataset_dir_count: int
    unreferenced_dataset_dir_count: int
    unreferenced_bytes: int
    referenced_bytes: int
    unreferenced: list[dict[str, Any]] = field(default_factory=list)
    referenced: list[dict[str, Any]] = field(default_factory=list)


def lake_gc(config: LakeGcConfig) -> LakeGcResult:
    cfg = config
    if not cfg.dry_run:
        raise ValueError("lake_gc_delete_not_implemented: run with --dry-run")
    lake = ResearchDataLake(cfg.lake_root)
    referenced = _referenced_dataset_ids(lake, paths=qdp_paths(cfg.workspace_root) if cfg.workspace_root else None)
    inventory = _dataset_dir_inventory(lake, with_size=bool(cfg.with_size))
    unreferenced_items = [item for item in inventory if item["dataset_id"] not in referenced]
    referenced_items = [item for item in inventory if item["dataset_id"] in referenced]
    unreferenced_items = sorted(unreferenced_items, key=lambda item: int(item["bytes"]), reverse=True)
    referenced_items = sorted(referenced_items, key=lambda item: int(item["bytes"]), reverse=True)
    max_items = max(0, int(cfg.max_items or 0))
    return LakeGcResult(
        status="dry_run",
        destructive_actions_performed=False,
        referenced_dataset_count=len(referenced),
        dataset_dir_count=len(inventory),
        unreferenced_dataset_dir_count=len(unreferenced_items),
        unreferenced_bytes=sum(int(item["bytes"]) for item in unreferenced_items),
        referenced_bytes=sum(int(item["bytes"]) for item in referenced_items),
        unreferenced=unreferenced_items[:max_items] if max_items else unreferenced_items,
        referenced=referenced_items[:max_items] if cfg.include_referenced and max_items else (referenced_items if cfg.include_referenced else []),
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Dry-run QDP data lake garbage collection inventory.")
    parser.add_argument("--lake-root", default=str(DEFAULT_DATA_LAKE_ROOT))
    parser.add_argument("--workspace-root", default="")
    parser.add_argument("--dry-run", action="store_true", default=True)
    parser.add_argument("--include-referenced", action="store_true")
    parser.add_argument("--with-size", action="store_true", help="Recursively stat dataset directories; slow on large lakes.")
    parser.add_argument("--max-items", type=int, default=200)
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = lake_gc(
        LakeGcConfig(
            lake_root=Path(args.lake_root),
            workspace_root=Path(args.workspace_root) if str(args.workspace_root or "").strip() else None,
            dry_run=True,
            include_referenced=bool(args.include_referenced),
            with_size=bool(args.with_size),
            max_items=int(args.max_items),
        )
    )
    payload = result.__dict__
    print(json.dumps(payload, ensure_ascii=False, indent=2 if args.json else None, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
