"""Margin Eligibility Update: workflow responsibilities."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from quantlab.core.io import json_safe
from quantlab.data.domains.contracts.schema import DataDomain
from quantlab.data.qdp_v2.manifest import (
    atomic_write_json,
    qdp_v2_root,
    read_active_manifest,
    utc_now,
    write_active_manifest,
)
from quantlab.data.qdp_v2.status import active_dataset_map

from .config import (
    MAX_WORKERS,
    UPDATE_ID,
    MarginEligibilityUpdateError,
)
from .context import (
    _dataset_paths,
    _read_state,
    _workspace,
    _write_state,
)
from .download import (
    download,
)
from .install import (
    _install,
)
from .prepare import (
    prepare,
)
from .repair import (
    repair_margin_detail_provenance,
    repair_sse_eligibility_semantics,
)


def commit(*, workspace_root: str | Path | None = None) -> dict[str, Any]:
    workspace = _workspace(workspace_root)
    state = _read_state(workspace)
    if state.get("status") == "applied":
        return state
    if state.get("status") != "prepared":
        raise MarginEligibilityUpdateError(f"margin_eligibility_not_prepared:{state.get('status')}")
    margin_manifest, _ = _dataset_paths(workspace, DataDomain.MARGIN_DETAIL)
    ids: dict[str, str] = {}
    installed: dict[str, Any] = {}
    for domain in (DataDomain.MARGIN_ELIGIBILITY, DataDomain.MARGIN_DETAIL):
        dataset_id, record = _install(
            workspace,
            domain=domain,
            paths=[Path(path) for path in state["prepared"][domain]],
            input_manifest=(margin_manifest if domain == DataDomain.MARGIN_DETAIL else None),
        )
        ids[domain] = dataset_id
        installed[domain] = record
    root = qdp_v2_root(workspace)
    active = read_active_manifest(root)
    current = active_dataset_map(active)
    if current.get(DataDomain.MARGIN_DETAIL) != dict(state.get("input_dataset_ids", {}) or {}).get(
        DataDomain.MARGIN_DETAIL
    ):
        raise MarginEligibilityUpdateError("active_margin_detail_drifted")
    active["datasets"] = {**current, **ids}
    active["updated_at"] = utc_now()
    write_active_manifest(root, active)
    state.update({"status": "applied", "installed_domains": installed})
    _write_state(workspace, state)
    atomic_write_json(root / "audits" / f"{UPDATE_ID}.json", state)
    return state


def evaluate(*, workspace_root: str | Path | None = None) -> dict[str, Any]:
    workspace = _workspace(workspace_root)
    state = _read_state(workspace)
    active = active_dataset_map(read_active_manifest(qdp_v2_root(workspace)))
    installed = dict(state.get("installed_domains", {}) or {})
    checks = {
        **dict(state.get("checks", {}) or {}),
        "eligibility_dataset_active": active.get(DataDomain.MARGIN_ELIGIBILITY)
        == dict(installed.get(DataDomain.MARGIN_ELIGIBILITY, {}) or {}).get("dataset_id"),
        "margin_detail_repair_dataset_active": active.get(DataDomain.MARGIN_DETAIL)
        == dict(installed.get(DataDomain.MARGIN_DETAIL, {}) or {}).get("dataset_id"),
    }
    return {
        "status": "ok" if checks and all(checks.values()) else "error",
        "update_id": UPDATE_ID,
        "checks": checks,
        "installed_domains": installed,
        "exchange_detail_missing_count": state.get("exchange_detail_missing_count", 0),
        "exchange_qdp_conflict_field_count": state.get("exchange_qdp_conflict_field_count", 0),
        "annual_statistics": state.get("annual_statistics", []),
    }


def run_pending(
    *,
    workspace_root: str | Path | None = None,
    max_workers: int = MAX_WORKERS,
    seal_runtime: bool = True,
) -> dict[str, Any]:
    download(workspace_root=workspace_root, max_workers=max_workers)
    prepare(workspace_root=workspace_root)
    result = commit(workspace_root=workspace_root)
    if seal_runtime:
        from quantlab.data.qdp_v2.runtime_archive.seal import seal_completed_workflow

        result = {
            **result,
            "runtime_archive": seal_completed_workflow(
                UPDATE_ID,
                workspace_root=workspace_root,
            ),
        }
    return result


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="qdp margin-eligibility-update")
    parser.add_argument("--workspace-root", default="")
    parser.add_argument("--max-workers", type=int, default=MAX_WORKERS)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--run-pending", action="store_true")
    mode.add_argument("--evaluate", action="store_true")
    mode.add_argument("--status", action="store_true")
    mode.add_argument("--repair-provenance", action="store_true")
    mode.add_argument("--repair-sse-semantics", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    workspace = str(args.workspace_root or "") or None
    if args.run_pending:
        payload = run_pending(
            workspace_root=workspace,
            max_workers=int(args.max_workers),
        )
    elif args.evaluate:
        payload = evaluate(workspace_root=workspace)
    elif args.repair_provenance:
        payload = repair_margin_detail_provenance(workspace_root=workspace)
    elif args.repair_sse_semantics:
        payload = repair_sse_eligibility_semantics(workspace_root=workspace)
    else:
        payload = _read_state(_workspace(workspace))
    print(json.dumps(json_safe(payload), ensure_ascii=False, indent=2))
    return 0 if payload.get("status") not in {"error", "failed"} else 2
