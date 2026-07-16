from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import pandas as pd

from quant_data_platform.core.json_io import json_safe
from quant_data_platform.core.paths import qdp_paths
from quant_data_platform.qdp_v2.manifest import (
    dataset_manifest_for_id,
    qdp_v2_root,
    read_active_manifest,
    read_dataset_manifest,
)
from quant_data_platform.qdp_v2.recent_market_repair import (
    run_baostock_intraday_repair,
    run_recent_daily_repair,
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
        "provider_order": ["mootdx_online", "baostock"],
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
    daily = run_recent_daily_repair(
        start_date=start,
        trade_date=end,
        workspace_root=workspace,
        workers=int(workers),
    )
    fast_5m = run_recent_intraday_repair(
        start_date=start,
        end_date=end,
        workspace_root=workspace,
        workers=int(workers),
    )
    baostock_5m = run_baostock_intraday_repair(
        start_date=start,
        end_date=end,
        workspace_root=workspace,
        workers=int(workers),
    )
    payload = {
        **plan,
        "status": "updated",
        "workers": int(workers),
        "daily": _state_summary(daily),
        "mootdx_5m": _state_summary(fast_5m),
        "baostock_5m": _state_summary(baostock_5m),
    }
    if not keep_runtime:
        runtime = qdp_paths(workspace).data_dir / "qdp_runtime" / "recent_market_repair"
        if runtime.exists():
            resolved = runtime.resolve(strict=True)
            resolved.relative_to(workspace)
            shutil.rmtree(resolved)
            payload["runtime_cleanup"] = "deleted_after_success"
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
        )
        if key in state
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="qdp update",
        description="Update the single current market store in place.",
    )
    parser.add_argument("--workspace-root", default="")
    parser.add_argument("--as-of-date", required=True)
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
    return 0 if payload.get("status") in {"planned", "updated"} else 2


def _format(payload: dict[str, Any]) -> str:
    lines = [
        f"status: {payload.get('status')}",
        f"range: {payload.get('start_date', '')}..{payload.get('as_of_date', '')}",
        f"workers: {payload.get('workers', '')}",
    ]
    for name in ("daily", "mootdx_5m", "baostock_5m"):
        if name in payload:
            lines.append(f"{name}: {payload[name]}")
    return "\n".join(lines)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
