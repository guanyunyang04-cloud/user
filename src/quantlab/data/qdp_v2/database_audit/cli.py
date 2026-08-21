"""Database audit cli checks."""

from __future__ import annotations

import argparse
import json

from quantlab.data.core.json_io import json_safe
from quantlab.data.qdp_v2.manifest import (
    qdp_v2_root,
)

from .asof import (
    audit_database_as_of,
)
from .common import (
    _format,
)
from .orchestrator import (
    audit_database,
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="qdp check --full",
        description="Run structural and row-level checks on the current QDP tables.",
    )
    parser.add_argument("--workspace-root", default="")
    parser.add_argument("--runtime", default="balanced", choices=("safe", "balanced", "fast"))
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--max-shards", type=int, default=0)
    parser.add_argument("--sample-limit", type=int, default=20)
    parser.add_argument("--as-of", default="")
    parser.add_argument("--skip-global-primary-key", action="store_true")
    parser.add_argument("--write-audit", action="store_true")
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    common = {
        "workspace_root": str(args.workspace_root or "") or None,
        "deep": True,
        "runtime": str(args.runtime),
        "threads": int(args.threads) or None,
        "max_shards": int(args.max_shards),
        "sample_limit": int(args.sample_limit),
        "skip_global_primary_key": bool(args.skip_global_primary_key),
    }
    if args.as_of:
        write_path = None
        if args.write_audit:
            root = qdp_v2_root(common["workspace_root"])
            write_path = root / "audits" / f"database_audit_as_of_{str(args.as_of).replace('-', '')}.json"
        payload = audit_database_as_of(as_of_date=str(args.as_of), write_path=write_path, **common)
    else:
        payload = audit_database(write=bool(args.write_audit), **common)
    if args.json:
        print(json.dumps(json_safe(payload), ensure_ascii=False, indent=2))
    else:
        print(_format(payload))
    return 0 if payload.get("status") == "ok" else 2


