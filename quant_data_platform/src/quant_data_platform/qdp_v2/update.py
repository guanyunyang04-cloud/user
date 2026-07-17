from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from quant_data_platform.core.json_io import json_safe
from quant_data_platform.core.paths import qdp_paths
from quant_data_platform.qdp_v2.baostock_update import run_baostock_core_update
from quant_data_platform.qdp_v2.database_audit import audit_latest_keys
from quant_data_platform.qdp_v2.factor_update import run_factor_tail_update
from quant_data_platform.qdp_v2.manifest import (
    dataset_manifest_for_id,
    qdp_v2_root,
    read_active_manifest,
    read_dataset_manifest,
    write_active_manifest,
)
from quant_data_platform.qdp_v2.permanent_exclusions import (
    purge_permanent_exclusions,
    register_current_exclusions,
)
from quant_data_platform.qdp_v2.recent_market_repair import (
    run_baostock_intraday_repair,
    run_recent_intraday_repair,
)


def plan_update(
    *,
    as_of_date: str,
    start_date: str = "",
    lookback_days: int = 10,
    workspace_root: str | Path | None = None,
) -> dict[str, Any]:
    root = qdp_v2_root(workspace_root)
    active = read_active_manifest(root)
    if not active:
        return {"status": "error", "errors": ["active_manifest_missing"]}
    target = pd.Timestamp(as_of_date).normalize()
    coverage: dict[str, dict[str, Any]] = {}
    next_dates: list[pd.Timestamp] = []
    for domain in ("market_daily_raw", "market_intraday_5m"):
        dataset_id = str(active.get("datasets", {}).get(domain, ""))
        manifest_path = dataset_manifest_for_id(root, dataset_id, domain)
        if manifest_path is None:
            return {"status": "error", "errors": [f"missing_current_table:{domain}"]}
        manifest = read_dataset_manifest(manifest_path)
        coverage[domain] = {
            "dataset_id": dataset_id,
            "start_date": manifest.start_date,
            "end_date": manifest.end_date,
            "row_count": manifest.row_count,
        }
        if manifest.end_date:
            next_dates.append(pd.Timestamp(manifest.end_date) + pd.Timedelta(days=1))
    if start_date:
        start = pd.Timestamp(start_date).normalize()
    else:
        lookback = target - pd.Timedelta(days=max(0, int(lookback_days) - 1))
        start = min([lookback, *next_dates]) if next_dates else lookback
    if start > target:
        start = target
    return {
        "status": "planned",
        "start_date": start.strftime("%Y-%m-%d"),
        "as_of_date": target.strftime("%Y-%m-%d"),
        "workers": 4,
        "provider_order": [
            "baostock_core",
            "mootdx_5m_accelerator",
            "baostock_5m_fallback",
        ],
        "coverage_before": coverage,
    }


def run_update(
    *,
    as_of_date: str,
    start_date: str = "",
    lookback_days: int = 10,
    workers: int = 4,
    workspace_root: str | Path | None = None,
    keep_runtime: bool = False,
) -> dict[str, Any]:
    if int(workers) not in {1, 2, 3, 4}:
        raise ValueError("workers_must_be_between_1_and_4")
    workspace = Path(workspace_root or Path.cwd()).resolve()
    plan = plan_update(
        as_of_date=as_of_date,
        start_date=start_date,
        lookback_days=lookback_days,
        workspace_root=workspace,
    )
    if plan.get("status") != "planned":
        return plan
    start = str(plan["start_date"])
    end = str(plan["as_of_date"])
    payload = {
        **plan,
        "status": "updating",
        "workers": int(workers),
    }
    stage = "baostock_core"
    try:
        core = run_baostock_core_update(
            start_date=start,
            end_date=end,
            workspace_root=workspace,
        )
        payload[stage] = _state_summary(core)

        stage = "permanent_exclusions"
        exclusions = register_current_exclusions(
            as_of_date=end,
            workspace_root=workspace,
        )
        exclusion_state: dict[str, Any] = dict(exclusions)
        if int(exclusions.get("new_exclusion_count", 0) or 0) > 0:
            exclusion_state["purge"] = purge_permanent_exclusions(
                workspace_root=workspace
            )
        payload[stage] = _state_summary(exclusion_state)

        stage = "adjust_factor"
        factor = run_factor_tail_update(
            start_date=start,
            end_date=end,
            workspace_root=workspace,
        )
        payload[stage] = _state_summary(factor)

        stage = "mootdx_5m"
        fast_5m = run_recent_intraday_repair(
            start_date=start,
            end_date=end,
            workspace_root=workspace,
            workers=int(workers),
        )
        payload[stage] = _state_summary(fast_5m)

        stage = "baostock_5m"
        baostock_5m = run_baostock_intraday_repair(
            start_date=start,
            end_date=end,
            workspace_root=workspace,
            workers=int(workers),
        )
        payload[stage] = _state_summary(baostock_5m)

        stage = "latest_check"
        latest = audit_latest_keys(workspace_root=workspace)
        payload[stage] = latest
        if latest.get("status") not in {"ok", "warning"}:
            raise RuntimeError(
                f"post_update_latest_check_failed:{latest.get('status')}"
            )
        payload["active_as_of_date"] = _advance_active_date(workspace)
        unresolved = int(baostock_5m.get("unresolved_day_count", 0) or 0)
        provider_errors = int(baostock_5m.get("provider_error_count", 0) or 0)
        payload["status"] = (
            "updated_with_gaps" if unresolved or provider_errors else "updated"
        )
    except Exception as exc:
        payload.update(
            {
                "status": "failed",
                "failed_stage": stage,
                "error_type": type(exc).__name__,
                "error": str(exc)[:1000],
                "runtime_cleanup": "retained_after_failure",
            }
        )
        return payload

    if not keep_runtime and payload["status"] == "updated":
        runtime = qdp_paths(workspace).data_dir / "qdp_runtime" / "recent_market_repair"
        if runtime.exists():
            resolved = runtime.resolve(strict=True)
            resolved.relative_to(workspace)
            shutil.rmtree(resolved)
        core_runtime = qdp_paths(workspace).data_dir / "qdp_runtime" / "baostock_update"
        if core_runtime.exists():
            resolved = core_runtime.resolve(strict=True)
            resolved.relative_to(workspace)
            shutil.rmtree(resolved)
        payload["runtime_cleanup"] = "deleted_after_success"
    elif payload["status"] == "updated_with_gaps":
        payload["runtime_cleanup"] = "retained_for_unresolved_days"
    return payload


def _state_summary(state: dict[str, Any]) -> dict[str, Any]:
    return {
        key: state.get(key)
        for key in (
            "status",
            "task_count",
            "row_count",
            "accepted_day_count",
            "unresolved_day_count",
            "provider_error_count",
            "workers",
            "minimum_available_bytes",
            "missing_rows",
            "open_date_count",
            "missing_key_count",
            "event_count",
            "initialized_symbol_count",
            "remaining_missing_key_count",
            "exclusion_count",
            "new_exclusion_count",
            "removed_rows",
        )
        if key in state
    }


def _advance_active_date(workspace: Path) -> str:
    root = qdp_v2_root(workspace)
    active = read_active_manifest(root)
    daily_id = str(dict(active.get("datasets", {}) or {}).get("market_daily_raw", ""))
    manifest_path = dataset_manifest_for_id(root, daily_id, "market_daily_raw")
    if manifest_path is None:
        raise RuntimeError("updated_daily_manifest_missing")
    latest = str(read_dataset_manifest(manifest_path).end_date or "")
    if not latest:
        raise RuntimeError("updated_daily_end_date_missing")
    active["active_as_of_date"] = latest
    scope = dict(active.get("scope", {}) or {})
    scope["end_date"] = latest
    active["scope"] = scope
    write_active_manifest(root, active)
    return latest


def _default_as_of_date() -> str:
    now = datetime.now(ZoneInfo("Asia/Shanghai"))
    candidate = pd.Timestamp(now.date())
    if now.hour < 17:
        candidate -= pd.offsets.BDay(1)
    while candidate.weekday() >= 5:
        candidate -= pd.Timedelta(days=1)
    return candidate.strftime("%Y-%m-%d")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="qdp update",
        description="Update the single current market store in place.",
    )
    parser.add_argument("--workspace-root", default="")
    parser.add_argument("--as-of-date", default=_default_as_of_date())
    parser.add_argument("--start-date", default="")
    parser.add_argument("--lookback-days", type=int, default=10)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--keep-runtime", action="store_true")
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    kwargs = {
        "as_of_date": str(args.as_of_date),
        "start_date": str(args.start_date or ""),
        "lookback_days": int(args.lookback_days),
        "workspace_root": str(args.workspace_root or "") or None,
    }
    payload = plan_update(**kwargs) if args.dry_run else run_update(
        **kwargs,
        workers=int(args.workers),
        keep_runtime=bool(args.keep_runtime),
    )
    if args.json:
        print(json.dumps(json_safe(payload), ensure_ascii=False, indent=2))
    else:
        print(_format(payload))
    return 0 if payload.get("status") in {"planned", "updated", "updated_with_gaps"} else 2


def _format(payload: dict[str, Any]) -> str:
    lines = [
        f"status: {payload.get('status')}",
        f"range: {payload.get('start_date', '')}..{payload.get('as_of_date', '')}",
        f"workers: {payload.get('workers', '')}",
    ]
    for name in ("baostock_core", "permanent_exclusions", "adjust_factor", "mootdx_5m", "baostock_5m", "latest_check"):
        if name in payload:
            lines.append(f"{name}: {payload[name]}")
    return "\n".join(lines)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
