"""Historical Intraday External Archive Repair: workflow responsibilities."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from quantlab.data.core.json_io import json_safe
from quantlab.data.qdp_v2.manifest import (
    atomic_write_json,
    qdp_v2_root,
    read_active_manifest,
)
from quantlab.data.qdp_v2.recent_market_repair import (
    INTRADAY_DOMAIN,
)
from quantlab.data.qdp_v2.research_event_update import (
    _sha256,
)
from quantlab.data.qdp_v2.status import active_dataset_map

from .config import (
    DEFAULT_ARCHIVE,
    DEFAULT_WORKERS,
    REPAIR_ID,
    ExternalArchiveRepairError,
    SymbolResult,
)
from .context import (
    _read_state,
    _workspace,
    _write_state,
)
from .install import (
    _install_immutable_version,
)
from .prepare import (
    _assemble_decisions,
    _bundle_accepted,
    _initialize,
    _validate_prepared,
)
from .process import (
    _process_symbol,
)


def _process_archive_symbols(
    workspace: Path,
    *,
    archive: Path,
    inventory: Any,
    members: dict[str, Any],
    candidate_symbols: list[str],
    max_workers: int,
    state: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    grouped = {
        str(symbol): frame.copy()
        for symbol, frame in inventory.loc[inventory["symbol"].isin(candidate_symbols)].groupby("symbol", sort=True)
    }
    workers = max(1, min(DEFAULT_WORKERS, int(max_workers)))
    results: dict[str, dict[str, Any]] = dict(state.get("symbols", {}) or {})
    failures: dict[str, str] = {}
    futures: dict[Future[SymbolResult], str] = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for symbol in candidate_symbols:
            futures[
                pool.submit(
                    _process_symbol,
                    workspace,
                    archive_path=archive,
                    member=members[symbol],
                    reference=grouped[symbol],
                )
            ] = symbol
        for completed, future in enumerate(as_completed(futures), start=1):
            symbol = futures[future]
            try:
                results[symbol] = future.result().to_dict()
            except Exception as exc:  # noqa: BLE001 - preserve task error in ledger
                failures[symbol] = f"{type(exc).__name__}:{str(exc)[:500]}"
            if completed % 5 == 0 or completed == len(futures):
                state.update(
                    {
                        "status": "processing",
                        "maximum_workers": workers,
                        "completed_symbol_count": len(results),
                        "failed_symbols": failures,
                        "symbols": results,
                    }
                )
                _write_state(workspace, state)
    if failures:
        state.update({"status": "failed", "failed_symbols": failures})
        _write_state(workspace, state)
        raise ExternalArchiveRepairError(f"external_archive_symbol_failures:{len(failures)}")
    return results


def _prepare_archive_bundle(
    workspace: Path,
    *,
    inventory: Any,
    candidate_symbols: list[str],
    results: dict[str, dict[str, Any]],
    input_manifest: Any,
    state: dict[str, Any],
) -> tuple[list[Path], dict[str, Any]]:
    ordered = [
        SymbolResult(**{**results[symbol], "reused": bool(results[symbol].get("reused", False))})
        for symbol in candidate_symbols
    ]
    decisions_path = _assemble_decisions(workspace, inventory, ordered)
    bundle_paths = _bundle_accepted(workspace, ordered)
    validation = _validate_prepared(
        workspace,
        inventory=inventory,
        decisions_path=decisions_path,
        bundle_paths=bundle_paths,
        input_manifest=input_manifest,
    )
    state.update(
        {
            "status": "prepared",
            "symbols": results,
            "failed_symbols": {},
            "decision_path": str(decisions_path),
            "decision_sha256": _sha256(decisions_path),
            "prepared_bundle_paths": [str(path) for path in bundle_paths],
            "prepared_bundle_sha256": {str(path): _sha256(path) for path in bundle_paths},
            **validation,
        }
    )
    _write_state(workspace, state)
    return bundle_paths, validation


def _apply_archive_bundle(
    workspace: Path,
    *,
    input_manifest: Any,
    bundle_paths: list[Path],
    validation: dict[str, Any],
    state: dict[str, Any],
) -> None:
    dataset_id, installed = _install_immutable_version(
        workspace,
        input_manifest=input_manifest,
        bundle_paths=bundle_paths,
        validation=validation,
    )
    state.update(
        {
            "status": "applied",
            "installed_dataset_id": dataset_id,
            "installed_bundle_paths": [str(path) for path in installed],
            "output_row_count": input_manifest.row_count + int(validation["accepted_row_count"]),
        }
    )
    _write_state(workspace, state)


def _archive_repair_audit(workspace: Path, state: dict[str, Any], results: dict[str, dict[str, Any]]) -> None:
    audit = {key: value for key, value in state.items() if key != "symbols"} | {
        "symbol_task_count": len(results),
        "reused_symbol_task_count": sum(bool(item.get("reused", False)) for item in results.values()),
    }
    atomic_write_json(qdp_v2_root(workspace) / "audits" / f"{REPAIR_ID}.json", audit)


def run_pending(
    *,
    workspace_root: str | Path | None = None,
    archive_path: str | Path = DEFAULT_ARCHIVE,
    max_workers: int = DEFAULT_WORKERS,
    seal_runtime: bool = True,
) -> dict[str, Any]:
    workspace = _workspace(workspace_root)
    state = _read_state(workspace)
    if state.get("status") == "applied":
        return state
    archive = Path(archive_path).resolve()
    inventory, members, input_manifest, state = _initialize(workspace, archive)
    candidate_symbols = sorted(set(inventory["symbol"]).intersection(members))
    results = _process_archive_symbols(
        workspace,
        archive=archive,
        inventory=inventory,
        members=members,
        candidate_symbols=candidate_symbols,
        max_workers=max_workers,
        state=state,
    )
    bundle_paths, validation = _prepare_archive_bundle(
        workspace,
        inventory=inventory,
        candidate_symbols=candidate_symbols,
        results=results,
        input_manifest=input_manifest,
        state=state,
    )
    _apply_archive_bundle(
        workspace,
        input_manifest=input_manifest,
        bundle_paths=bundle_paths,
        validation=validation,
        state=state,
    )
    _archive_repair_audit(workspace, state, results)
    if seal_runtime:
        from quantlab.data.qdp_v2.runtime_archive import (
            seal_completed_workflow,
        )

        return {
            **state,
            "runtime_archive": seal_completed_workflow(
                REPAIR_ID,
                workspace_root=workspace,
            ),
        }
    return state


def evaluate(*, workspace_root: str | Path | None = None) -> dict[str, Any]:
    workspace = _workspace(workspace_root)
    state = _read_state(workspace)
    active = active_dataset_map(read_active_manifest(qdp_v2_root(workspace)))
    checks = {
        **dict(state.get("checks", {}) or {}),
        "repair_applied": state.get("status") == "applied",
        "immutable_dataset_active": active.get(INTRADAY_DOMAIN) == state.get("installed_dataset_id"),
        "request_2026_count_zero": int(state.get("request_2026_count", -1)) == 0,
        "write_2026_count_zero": int(state.get("write_2026_count", -1)) == 0,
    }
    return {
        "status": "ok" if checks and all(checks.values()) else "error",
        "repair_id": REPAIR_ID,
        "checks": checks,
        "installed_dataset_id": state.get("installed_dataset_id", ""),
        "input_intraday_dataset_id": state.get("input_intraday_dataset_id", ""),
        "decision_counts": state.get("decision_counts", {}),
        "accepted_row_count": state.get("accepted_row_count", 0),
        "formal_quality_complete_pool_row_count": state.get("formal_quality_complete_pool_row_count", 0),
        "rejection_reason_counts": state.get("rejection_reason_counts", {}),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="qdp external-archive-5m-repair")
    parser.add_argument("--workspace-root", default="")
    parser.add_argument("--archive-path", default=str(DEFAULT_ARCHIVE))
    parser.add_argument("--max-workers", type=int, default=DEFAULT_WORKERS)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--run-pending", action="store_true")
    mode.add_argument("--evaluate", action="store_true")
    mode.add_argument("--status", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    workspace = str(args.workspace_root or "") or None
    if args.run_pending:
        payload = run_pending(
            workspace_root=workspace,
            archive_path=args.archive_path,
            max_workers=args.max_workers,
        )
    elif args.evaluate:
        payload = evaluate(workspace_root=workspace)
    else:
        payload = _read_state(_workspace(workspace))
    print(json.dumps(json_safe(payload), ensure_ascii=False, indent=2))
    return 0 if payload.get("status") not in {"error", "failed"} else 2
