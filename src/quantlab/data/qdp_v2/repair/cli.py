"""Repair: cli responsibilities."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from quantlab.data.core.json_io import json_safe
from quantlab.data.qdp_v2.manifest import (
    _manifest_schema_from_arrow,
)

from .mutation import (
    mutate_active_shards_from_parquet,
    patch_active_cells,
    replace_active_table_from_parquet,
)
from .validation import (
    _common_schema,
    _resolve_prepared_path,
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Repair active QDP data directly.")
    parser.add_argument("--workspace-root", type=Path)
    subparsers = parser.add_subparsers(dest="repair_command", required=True)

    patch = subparsers.add_parser("patch", help="Patch explicitly named cells.")
    patch.add_argument("--request", type=Path, required=True)
    patch.add_argument("--reason", default="patch cells")
    patch.add_argument("--apply", action="store_true")

    mutate = subparsers.add_parser("mutate", help="Replace, remove, or append shards.")
    mutate.add_argument("--domain", required=True)
    mutate.add_argument("--replace", action="append", default=[])
    mutate.add_argument("--remove", action="append", default=[])
    mutate.add_argument("--append", action="append", default=[])
    mutate.add_argument("--reason", default="mutate active shards")
    mutate.add_argument("--apply", action="store_true")

    replace_table = subparsers.add_parser("replace-table", help="Replace an active table.")
    replace_table.add_argument("--domain", required=True)
    replace_table.add_argument("--prepared", action="append", required=True)
    replace_table.add_argument("--primary-key", action="append", default=[])
    replace_table.add_argument("--contract-version", default="")
    replace_table.add_argument("--reason", default="replace active table")
    replace_table.add_argument("--apply", action="store_true")
    return parser


def _parse_replacements(values: Sequence[str]) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for value in values:
        if "=" not in value:
            raise ValueError(f"qdp_v2_repair_replace_invalid:{value}")
        old, new = value.split("=", 1)
        pairs.append((old, new))
    return pairs


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    workspace = str(args.workspace_root or "") or None
    if args.repair_command == "patch":
        result = patch_active_cells(
            args.request,
            reason=args.reason,
            workspace_root=workspace,
            apply=bool(args.apply),
        )
    elif args.repair_command == "mutate":
        if not args.apply:
            result = {
                "status": "would_mutate",
                "dry_run": True,
                "domain": args.domain,
                "replacements": _parse_replacements(args.replace),
                "removals": list(args.remove),
                "appends": list(args.append),
            }
        else:
            result = mutate_active_shards_from_parquet(
                args.domain,
                replacements=_parse_replacements(args.replace),
                removals=args.remove,
                appends=args.append,
                reason=args.reason,
                workspace_root=workspace,
            )
    else:
        if not args.apply:
            schema = _common_schema([_resolve_prepared_path(item, workspace) for item in args.prepared])
            result = {
                "status": "would_replace_table",
                "dry_run": True,
                "domain": args.domain,
                "prepared": list(args.prepared),
                "schema": _manifest_schema_from_arrow(schema),
            }
        else:
            result = replace_active_table_from_parquet(
                args.domain,
                args.prepared,
                reason=args.reason,
                workspace_root=workspace,
                primary_key=args.primary_key or None,
                contract_version=args.contract_version,
            )
    print(json.dumps(json_safe(result), ensure_ascii=False, indent=2))
    return 0
