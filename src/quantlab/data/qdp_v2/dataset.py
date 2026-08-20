from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from quantlab.data.core.json_io import json_safe
from quantlab.data.qdp_v2.manifest import (
    dataset_manifest_for_id,
    iter_dataset_manifests,
    qdp_v2_root,
    read_active_manifest,
    read_dataset_manifest,
)
from quantlab.data.qdp_v2.status import active_dataset_map


def list_datasets(*, workspace_root: str | Path | None = None) -> dict[str, Any]:
    root = qdp_v2_root(workspace_root)
    rows: list[dict[str, Any]] = []
    for path in iter_dataset_manifests(root):
        manifest = read_dataset_manifest(path)
        rows.append(
            {
                "dataset_id": manifest.dataset_id,
                "domain": manifest.domain,
                "layer": manifest.layer,
                "frequency": manifest.frequency,
                "contract_version": manifest.contract_version,
                "start_date": manifest.start_date,
                "end_date": manifest.end_date,
                "row_count": manifest.row_count,
                "shard_count": len(manifest.shards),
                "manifest_path": str(path.resolve()),
            }
        )
    return {
        "status": "ok",
        "qdp_v2_root": str(root.resolve()),
        "dataset_count": len(rows),
        "datasets": sorted(rows, key=lambda item: (str(item["domain"]), str(item["dataset_id"]))),
    }


def describe_dataset(dataset_id: str, *, workspace_root: str | Path | None = None, domain: str = "", full: bool = False) -> dict[str, Any]:
    root = qdp_v2_root(workspace_root)
    identifier = str(dataset_id or "").strip()
    active = read_active_manifest(root)
    active_map = active_dataset_map(active)
    resolved_domain = str(domain or "").strip()
    resolved_id = active_map.get(identifier, identifier)
    if not resolved_domain and identifier in active_map:
        resolved_domain = identifier
    path = dataset_manifest_for_id(root, resolved_id, resolved_domain)
    if path is None:
        return {"status": "not_found", "dataset_id": identifier, "qdp_v2_root": str(root.resolve())}
    manifest = read_dataset_manifest(path)
    payload = manifest.to_dict()
    payload["status"] = "ok"
    payload["manifest_path"] = str(path.resolve())
    if full:
        return payload
    return _summarize_manifest(payload)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="qdp", description="Inspect qdp_v2 active table manifests.")
    parser.add_argument("--workspace-root", default="")
    sub = parser.add_subparsers(dest="dataset_command", required=True)
    list_cmd = sub.add_parser("list")
    list_cmd.add_argument("--json", action="store_true")
    describe_cmd = sub.add_parser("describe")
    describe_cmd.add_argument("dataset_id")
    describe_cmd.add_argument("--domain", default="")
    describe_cmd.add_argument("--full", action="store_true", help="Print the complete dataset.json manifest.")
    describe_cmd.add_argument("--json", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    workspace = str(args.workspace_root or "") or None
    if args.dataset_command == "list":
        payload = list_datasets(workspace_root=workspace)
    elif args.dataset_command == "describe":
        payload = describe_dataset(
            str(args.dataset_id),
            workspace_root=workspace,
            domain=str(args.domain or ""),
            full=bool(getattr(args, "full", False)),
        )
    else:
        raise ValueError(f"unsupported_dataset_command:{args.dataset_command}")
    if bool(getattr(args, "json", False)):
        print(json.dumps(json_safe(payload), ensure_ascii=False, indent=2))
    else:
        print(_format(payload))
    return 0 if str(payload.get("status", "")) == "ok" else 2


def _format(payload: dict[str, Any]) -> str:
    if "datasets" in payload:
        lines = [f"status: {payload.get('status')}", f"dataset_count: {payload.get('dataset_count', 0)}"]
        for item in list(payload.get("datasets", []) or []):
            lines.append(
                f"{item.get('domain')} {item.get('dataset_id')} "
                f"{item.get('start_date', '')}..{item.get('end_date', '')} "
                f"rows={item.get('row_count', 0)} shards={item.get('shard_count', 0)}"
            )
        return "\n".join(lines)
    if payload.get("status") == "not_found":
        return f"status: not_found\ndataset_id: {payload.get('dataset_id')}"
    if payload.get("summary_version") == 1:
        lines = [
            f"status: {payload.get('status')}",
            f"table: {payload.get('table')}",
            f"role: {payload.get('role')}",
            f"dataset_id: {payload.get('dataset_id')}",
            f"range: {payload.get('start_date')}..{payload.get('end_date')}",
            f"rows: {payload.get('row_count', 0)}",
            f"shards: {payload.get('shard_count', 0)}",
            f"frequency: {payload.get('frequency', '')}",
            f"contract: {payload.get('contract_version', '')}",
            f"primary_key: {', '.join(list(payload.get('primary_key', []) or []))}",
        ]
        quality = dict(payload.get("quality_summary", {}) or {})
        if quality:
            lines.append(
                "quality: "
                + ", ".join(f"{key}={value}" for key, value in quality.items() if value not in ("", None))
            )
        notes = list(payload.get("notes", []) or [])
        if notes:
            lines.append("notes: " + " | ".join(str(item) for item in notes))
        return "\n".join(lines)
    return "\n".join(f"{key}: {json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else value}" for key, value in payload.items())


def _summarize_manifest(payload: dict[str, Any]) -> dict[str, Any]:
    quality = dict(payload.get("quality", {}) or {})
    notes: list[str] = []
    overrides = quality.get("symbol_start_overrides") or {}
    if isinstance(overrides, dict) and overrides:
        notes.append("symbol_start_overrides=" + json.dumps(overrides, ensure_ascii=False, sort_keys=True))
    if quality.get("no_l2_order_book_fields") is True:
        notes.append("contains no L2/order-book fields")
    if quality.get("repaired_rows_excluded_by_data_scope"):
        notes.append(f"outside_scope_repaired_rows={quality.get('repaired_rows_excluded_by_data_scope')}")
    return {
        "status": "ok",
        "summary_version": 1,
        "table": payload.get("domain", ""),
        "role": _domain_role(str(payload.get("domain", "")), str(payload.get("layer", ""))),
        "dataset_id": payload.get("dataset_id", ""),
        "start_date": payload.get("start_date", ""),
        "end_date": payload.get("end_date", ""),
        "row_count": payload.get("row_count", 0),
        "shard_count": len(list(payload.get("shards", []) or [])),
        "frequency": payload.get("frequency", ""),
        "contract_version": payload.get("contract_version", ""),
        "primary_key": list(payload.get("primary_key", []) or []),
        "quality_summary": {
            "files": "ok" if quality.get("path_refs_exist") is True else str(quality.get("path_refs_exist", "unknown")),
            "primary_key": _quality_state(quality.get("primary_key_unique"), quality.get("primary_key_audit")),
            "date_coverage": str(quality.get("date_coverage_ok", "not_recorded")),
            "ohlcv": "ok" if quality.get("ohlcv_non_null") is True else str(quality.get("ohlcv_non_null", "not_applicable")),
        },
        "notes": notes,
    }


def _quality_state(value: Any, audit: Any) -> str:
    if value is True:
        return "passed"
    if isinstance(audit, dict) and str(audit.get("status", "")) == "passed":
        return "passed"
    if value in (False, None, ""):
        return "not_recorded"
    return str(value)


def _domain_role(domain: str, layer: str) -> str:
    roles = {
        "market_daily_raw": "raw daily OHLCV facts",
        "market_intraday_5m": "canonical complete 48-bar 5m market facts",
        "trading_calendar": "trading calendar",
        "universe_snapshot": "PIT universe snapshot",
        "security_status": "PIT listing/ST/suspension status",
        "valuation": "daily valuation facts",
        "adjust_factor": "adjustment factor facts",
        "industry_concept": "industry labels",
        "index_constituents": "index constituent facts",
        "corporate_actions": "corporate-action event facts",
        "share_capital": "share-capital facts",
        "name_change": "name-change event facts",
    }
    return roles.get(domain, layer or "active table")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
