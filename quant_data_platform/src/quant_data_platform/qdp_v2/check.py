from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from quant_data_platform.core.json_io import json_safe
from quant_data_platform.qdp_v2.audit import audit_active
from quant_data_platform.qdp_v2.database_audit import audit_database


def run_check(
    *,
    workspace_root: str | Path | None = None,
    full: bool = False,
    runtime: str = "balanced",
    no_write: bool = False,
) -> dict[str, Any]:
    if full:
        return audit_database(workspace_root=workspace_root, deep=True, runtime=runtime, write=not no_write)
    active = audit_active(workspace_root=workspace_root, write=not no_write)
    database = audit_database(workspace_root=workspace_root, deep=False, runtime=runtime, write=False)
    status = "ok" if active.get("status") == "ok" and database.get("status") == "ok" else "needs_attention"
    return {
        "status": status,
        "mode": "quick",
        "active": active,
        "database": {
            "status": database.get("status"),
            "dataset_count": database.get("dataset_count"),
            "finding_count": database.get("finding_count"),
            "coverage": database.get("coverage", {}),
            "errors": database.get("errors", []),
            "warnings": database.get("warnings", []),
        },
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="qdp check", description="Check the active qdp_v2 data base.")
    parser.add_argument("--workspace-root", default="")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--quick", action="store_true", help="Run manifest/footer and coverage checks.")
    mode.add_argument("--full", action="store_true", help="Run deep row-level and cross-frequency checks.")
    parser.add_argument("--runtime", default="balanced", choices=("safe", "balanced", "fast"))
    parser.add_argument("--no-write", action="store_true")
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    payload = run_check(
        workspace_root=str(args.workspace_root or "") or None,
        full=bool(args.full),
        runtime=str(args.runtime or "balanced"),
        no_write=bool(args.no_write),
    )
    if bool(args.json):
        print(json.dumps(json_safe(payload), ensure_ascii=False, indent=2))
    else:
        print(_format(payload))
    return 0 if str(payload.get("status", "")) == "ok" else 2


def _format(payload: dict[str, Any]) -> str:
    lines = [
        f"status: {payload.get('status')}",
        f"mode: {payload.get('mode', 'full' if payload.get('deep') else 'quick')}",
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

