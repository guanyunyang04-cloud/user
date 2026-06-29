from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from quant_data_platform.core.json_io import json_safe
from quant_data_platform.qdp_v2.environment import runtime_environment
from quant_data_platform.qdp_v2.manifest import atomic_write_json, dataset_manifest_for_id, qdp_v2_root, read_active_manifest, read_dataset_manifest, utc_now
from quant_data_platform.qdp_v2.runtime import resolve_runtime_profile
from quant_data_platform.qdp_v2.status import _active_dataset_refs


def plan_update(*, as_of_date: str, runtime: str, workspace_root: str | Path | None = None) -> dict[str, Any]:
    root = qdp_v2_root(workspace_root)
    active = read_active_manifest(root)
    if not active:
        return {"status": "error", "qdp_v2_root": str(root.resolve()), "errors": ["active_manifest_missing"]}
    profile = resolve_runtime_profile(runtime)
    target = str(as_of_date or "").strip()
    gaps: list[dict[str, Any]] = []
    coverage: list[dict[str, Any]] = []
    for section, domain, dataset_id in _active_dataset_refs(active):
        manifest_path = dataset_manifest_for_id(root, dataset_id, domain)
        if manifest_path is None:
            gaps.append({"section": section, "domain": domain, "dataset_id": dataset_id, "gap": "manifest_missing"})
            continue
        manifest = read_dataset_manifest(manifest_path)
        current_end = manifest.end_date
        coverage.append(
            {
                "section": section,
                "domain": domain,
                "dataset_id": dataset_id,
                "start_date": manifest.start_date,
                "end_date": current_end,
                "row_count": manifest.row_count,
            }
        )
        if target and current_end and current_end < target and _domain_is_date_updated(domain):
            gaps.append({"section": section, "domain": domain, "dataset_id": dataset_id, "from_exclusive": current_end, "to_inclusive": target})
    return {
        "status": "planned",
        "qdp_v2_root": str(root.resolve()),
        "as_of_date": target,
        "runtime": asdict(profile),
        "runtime_environment": runtime_environment(),
        "active_as_of_date": str(active.get("active_as_of_date", "") or ""),
        "coverage": coverage,
        "gaps": gaps,
        "provider_execution": "not_started",
        "memmap": "not_part_of_data_base",
    }


def _domain_is_date_updated(domain: str) -> bool:
    return domain in {
        "market_daily_raw",
        "market_daily_panel",
        "market_intraday_1m",
        "market_intraday_5m",
        "trading_calendar",
        "universe_snapshot",
        "security_status",
        "valuation",
        "adjust_factor",
        "intraday_daily_features",
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="qdp update", description="Plan or run qdp_v2 data base update.")
    parser.add_argument("--workspace-root", default="")
    parser.add_argument("--as-of-date", required=True)
    parser.add_argument("--runtime", default="balanced", choices=("safe", "balanced", "fast"))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    payload = plan_update(as_of_date=str(args.as_of_date), runtime=str(args.runtime or "balanced"), workspace_root=str(args.workspace_root or "") or None)
    root = qdp_v2_root(str(args.workspace_root or "") or None)
    if not bool(args.dry_run) and payload.get("status") == "planned":
        payload["status"] = "blocked"
        payload["blocker"] = "provider_execution_not_wired_yet"
        payload["message"] = "qdp_v2 update planning is implemented; provider write/validate/activate DAG is not enabled in this patch."
    run_id = f"update_plan_{utc_now().replace(':', '').replace('-', '')}"
    atomic_write_json(root / "runs" / f"{run_id}.json", payload)
    if bool(args.json):
        print(json.dumps(json_safe(payload), ensure_ascii=False, indent=2))
    else:
        print(_format(payload))
    return 0 if str(payload.get("status", "")) == "planned" else 2


def _format(payload: dict[str, Any]) -> str:
    lines = [
        f"status: {payload.get('status')}",
        f"as_of_date: {payload.get('as_of_date', '')}",
        f"gap_count: {len(payload.get('gaps', []) or [])}",
        f"provider_execution: {payload.get('provider_execution', '')}",
    ]
    for gap in list(payload.get("gaps", []) or [])[:20]:
        lines.append(f"gap: {gap}")
    if payload.get("blocker"):
        lines.append(f"blocker: {payload.get('blocker')}")
    return "\n".join(lines)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
