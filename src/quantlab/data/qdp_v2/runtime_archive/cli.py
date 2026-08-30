"""Runtime Archive: cli responsibilities."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from quantlab.core.io import json_safe

from .archive import (
    restore_archive,
)
from .seal import (
    _resolve_manifest_artifact_path,
    seal_workflows,
    verify_unit_manifest,
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="qdp runtime-archive")
    parser.add_argument("--workspace-root", default="")
    subparsers = parser.add_subparsers(dest="command", required=True)
    seal = subparsers.add_parser("seal")
    seal.add_argument("--workflow", action="append", default=[])
    seal.add_argument("--delete-expanded", action="store_true")
    seal.add_argument("--yes", action="store_true")
    verify = subparsers.add_parser("verify")
    verify.add_argument("--manifest", required=True)
    restore = subparsers.add_parser("restore")
    restore.add_argument("--manifest", required=True)
    restore.add_argument("--target-root", required=True)
    restore.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    workspace = str(args.workspace_root or "") or None
    if args.command == "seal":
        payload = seal_workflows(
            workspace_root=workspace,
            workflows=tuple(args.workflow),
            delete_expanded=bool(args.delete_expanded),
            yes=bool(args.yes),
        )
    elif args.command == "verify":
        payload = verify_unit_manifest(args.manifest)
    else:
        manifest_path = Path(args.manifest).resolve()
        manifest = dict(json.loads(manifest_path.read_text(encoding="utf-8")))
        payload = restore_archive(
            _resolve_manifest_artifact_path(manifest["archive_path"], manifest_path),
            _resolve_manifest_artifact_path(manifest["ledger_path"], manifest_path),
            target_root=args.target_root,
            overwrite=bool(args.overwrite),
        )
    print(json.dumps(json_safe(payload), ensure_ascii=False, indent=2))
    return 0 if payload.get("status") not in {"error", "failed"} else 2
