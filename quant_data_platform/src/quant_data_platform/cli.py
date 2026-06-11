from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from quant_data_platform.core.json_io import json_safe
from quant_data_platform.core.paths import qdp_paths
from quant_data_platform.core.registry import migrate_legacy_registry, registry_status
from quant_data_platform.lake.adapters import audit_inventory, build_bundle, bundle_summary_dict, cleanup_dry_run
from quant_data_platform.memmap.sharded import write_sharded_memmap_plan
from quant_data_platform.memmap.validation import validate_active_memmap


def _print(payload: dict[str, Any], *, as_json: bool) -> None:
    if as_json:
        print(json.dumps(json_safe(payload), ensure_ascii=False, indent=2))
        return
    for key, value in payload.items():
        if isinstance(value, (dict, list)):
            print(f"{key}: {json.dumps(json_safe(value), ensure_ascii=False)}")
        else:
            print(f"{key}: {value}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="qdp", description="Quant Data Platform governance CLI.")
    parser.add_argument("--workspace-root", default="", help="Workspace root, defaults to auto-detection.")
    sub = parser.add_subparsers(dest="command", required=True)

    status = sub.add_parser("status", help="Show canonical registry, bundle, and active memmap status.")
    status.add_argument("--json", action="store_true")

    init_registry = sub.add_parser("init-registry", help="Migrate canonical_data registry into quant_data_platform/registry.")
    init_registry.add_argument("--dry-run", action="store_true")
    init_registry.add_argument("--json", action="store_true")

    audit = sub.add_parser("audit", help="Write or print coverage, duplicate, orphan, and cleanup dry-run inventory.")
    audit.add_argument("--no-write", action="store_true")
    audit.add_argument("--json", action="store_true")

    bundle = sub.add_parser("build-bundle", help="Build canonical policy bundle with all v1 sidecars.")
    bundle.add_argument("--dry-run", action="store_true")
    bundle.add_argument("--json", action="store_true")

    validate = sub.add_parser("validate-memmap", help="Validate active or explicit forecast memmap manifest.")
    validate.add_argument("--manifest", default="")
    validate.add_argument("--min-universe-size", type=int, default=0)
    validate.add_argument("--min-train-rows", type=int, default=0)
    validate.add_argument("--json", action="store_true")

    sharded = sub.add_parser("build-sharded-memmap", help="Create the sharded memmap build plan scaffold.")
    sharded.add_argument("--profile", default="short_horizon_core_v1")
    sharded.add_argument("--max-universe-size", type=int, default=0)
    sharded.add_argument("--dry-run", action="store_true")
    sharded.add_argument("--json", action="store_true")

    cleanup = sub.add_parser("cleanup", help="Generate cleanup dry-run plan. This command never deletes files in v1.")
    cleanup.add_argument("--dry-run", action="store_true", default=True)
    cleanup.add_argument("--json", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    paths = qdp_paths(args.workspace_root or None)
    if args.command == "status":
        _print(registry_status(paths), as_json=bool(args.json))
        return 0
    if args.command == "init-registry":
        _print(migrate_legacy_registry(paths, write=not bool(args.dry_run)), as_json=bool(args.json))
        return 0
    if args.command == "audit":
        inventory = audit_inventory(paths, write=not bool(args.no_write))
        summary = {
            "status": inventory.get("status", ""),
            "dataset_count": inventory.get("dataset_count", 0),
            "duplicate_group_count": len(inventory.get("duplicate_dataset_groups", []) or []),
            "orphan_fingerprint_dir_count": len(inventory.get("filesystem", {}).get("orphan_fingerprint_dirs", []) or []),
            "forecast_dat_count": len(inventory.get("memmaps", {}).get("dat_files", []) or []),
            "report_path": inventory.get("report_path", ""),
            "cleanup_dry_run": inventory.get("cleanup_dry_run", {}),
        }
        _print(summary if not bool(args.json) else inventory, as_json=bool(args.json))
        return 0
    if args.command == "build-bundle":
        summary = build_bundle(paths, write=not bool(args.dry_run))
        _print(bundle_summary_dict(summary), as_json=bool(args.json))
        return 0
    if args.command == "validate-memmap":
        report = validate_active_memmap(
            paths,
            manifest=Path(args.manifest) if str(args.manifest or "").strip() else None,
            min_universe_size=int(args.min_universe_size),
            min_train_rows=int(args.min_train_rows),
        )
        _print(report, as_json=bool(args.json))
        return 0 if str(report.get("status", "")) == "ok" else 1
    if args.command == "build-sharded-memmap":
        payload = write_sharded_memmap_plan(
            paths,
            profile=str(args.profile or ""),
            max_universe_size=int(args.max_universe_size),
            write=not bool(args.dry_run),
        )
        _print(payload, as_json=bool(args.json))
        return 0
    if args.command == "cleanup":
        _print(cleanup_dry_run(paths), as_json=bool(args.json))
        return 0
    raise ValueError(f"unsupported_command: {args.command}")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
