from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from quantlab.data.core.json_io import json_safe
from quantlab.data.qdp_v2.audit import audit_active
from quantlab.data.qdp_v2.database_audit import (
    REQUIRED_DOMAINS,
    audit_database,
    audit_latest_keys,
)
from quantlab.data.qdp_v2.manifest import qdp_v2_root, read_active_manifest
from quantlab.data.qdp_v2.semantic_audit import audit_semantics
from quantlab.data.qdp_v2.status import active_dataset_map


def run_check(
    *,
    workspace_root: str | Path | None = None,
    full: bool = False,
    semantic: bool = False,
    runtime: str = "balanced",
    write: bool = False,
) -> dict[str, Any]:
    if semantic:
        return audit_semantics(workspace_root=workspace_root, write=write)
    if full:
        return audit_database(workspace_root=workspace_root, deep=True, runtime=runtime, write=write)
    active = audit_active(workspace_root=workspace_root, write=write, verify_footers=False)
    manifest = read_active_manifest(qdp_v2_root(workspace_root))
    active_domains = set(active_dataset_map(manifest))
    missing_required = sorted(REQUIRED_DOMAINS.difference(active_domains))
    latest = audit_latest_keys(workspace_root=workspace_root)
    database_status = (
        "ok"
        if not missing_required and latest.get("status") in {"ok", "warning"}
        else "needs_attention"
    )
    status = "ok" if active.get("status") == "ok" and database_status == "ok" else "needs_attention"
    return {
        "status": status,
        "mode": "quick",
        "scope": "physical_contract_and_latest_market_key_checks",
        "all_active_domains_semantically_certified": False,
        "active": active,
        "database": {
            "status": database_status,
            "dataset_count": len(active_domains),
            "finding_count": len(missing_required) + int(latest.get("finding_count", 0)),
            "coverage": {
                "active_domains": sorted(active_domains),
                "required_domains": sorted(REQUIRED_DOMAINS),
                "missing_required_domains": missing_required,
            },
            "latest_keys": latest,
            "errors": [],
            "warnings": [f"required_domains_missing:{','.join(missing_required)}"] if missing_required else [],
        },
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="qdp check", description="Check the active qdp_v2 data base.")
    parser.add_argument("--workspace-root", default="")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--quick",
        action="store_true",
        help="Run manifest, Parquet-footer schema, file-presence and latest-key checks.",
    )
    mode.add_argument(
        "--full",
        action="store_true",
        help="Run deep schema, primary-key, OHLC, factor, identity and 48-bar checks.",
    )
    mode.add_argument(
        "--semantic",
        action="store_true",
        help="Aggregate specialty audits and selected stable semantic invariants.",
    )
    parser.add_argument("--runtime", default="balanced", choices=("safe", "balanced", "fast"))
    parser.add_argument("--write-audit", action="store_true", help="Persist audit output; checks are read-only by default.")
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    payload = run_check(
        workspace_root=str(args.workspace_root or "") or None,
        full=bool(args.full),
        semantic=bool(args.semantic),
        runtime=str(args.runtime or "balanced"),
        write=bool(args.write_audit),
    )
    if bool(args.json):
        print(json.dumps(json_safe(payload), ensure_ascii=False, indent=2))
    else:
        print(_format(payload))
    return 0 if str(payload.get("status", "")) == "ok" else 2


def _format(payload: dict[str, Any]) -> str:
    lines = [
        f"status: {payload.get('status')}",
        f"mode: {payload.get('mode', 'quick')}",
    ]
    if "active" in payload:
        active = dict(payload.get("active", {}) or {})
        lines.append(f"active_dataset_count: {active.get('dataset_count', 0)}")
        lines.append(f"active_errors: {len(active.get('errors', []) or [])}")
        lines.append(f"active_warnings: {len(active.get('warnings', []) or [])}")
        database = dict(payload.get("database", {}) or {})
        lines.append(f"finding_count: {database.get('finding_count', 0)}")
    else:
        lines.append(f"dataset_count: {payload.get('dataset_count', 0)}")
        lines.append(f"finding_count: {payload.get('finding_count', 0)}")
    return "\n".join(lines)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
