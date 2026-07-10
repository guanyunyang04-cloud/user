from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from quant_data_platform.core.json_io import json_safe
from quant_data_platform.qdp_v2.manifest import (
    atomic_write_json,
    dataset_manifest_for_id,
    qdp_v2_root,
    read_active_manifest,
    read_dataset_manifest,
    utc_now,
)
from quant_data_platform.qdp_v2.status import _active_dataset_refs, _dataset_file_index, _manifest_path_key


def audit_active(*, workspace_root: str | Path | None = None, write: bool = False, verify_footers: bool = True) -> dict[str, Any]:
    root = qdp_v2_root(workspace_root)
    active = read_active_manifest(root)
    if not active:
        return {"status": "error", "qdp_v2_root": str(root.resolve()), "errors": ["active_manifest_missing"]}
    dataset_reports: list[dict[str, Any]] = []
    errors: list[str] = []
    warnings: list[str] = []
    for _, domain, dataset_id in _active_dataset_refs(active):
        manifest_path = dataset_manifest_for_id(root, dataset_id, domain)
        if manifest_path is None:
            errors.append(f"dataset_manifest_missing:{domain}:{dataset_id}")
            continue
        manifest = read_dataset_manifest(manifest_path)
        missing_shards: list[str] = []
        footer_errors: list[str] = []
        footer_rows = 0
        file_index = _dataset_file_index(manifest_path.parent)
        for shard in manifest.shards:
            shard_path = Path(shard.path) if Path(shard.path).is_absolute() else root / shard.path
            if _manifest_path_key(shard.path, root) not in file_index:
                missing_shards.append(str(shard_path))
                continue
            if verify_footers:
                try:
                    rows = _parquet_row_count(shard_path)
                    footer_rows += int(rows)
                    if shard.row_count > 0 and rows > 0 and int(rows) != int(shard.row_count):
                        footer_errors.append(f"row_count_mismatch:{shard.path}:manifest={shard.row_count}:footer={rows}")
                except Exception as exc:
                    footer_errors.append(f"footer_unreadable:{shard.path}:{exc}")
        if missing_shards:
            errors.append(f"missing_shards:{domain}:{len(missing_shards)}")
        if footer_errors:
            errors.extend(footer_errors[:20])
            if len(footer_errors) > 20:
                warnings.append(f"footer_errors_truncated:{domain}:{len(footer_errors)}")
        if verify_footers and manifest.row_count and footer_rows and int(manifest.row_count) != int(footer_rows):
            errors.append(f"dataset_row_count_mismatch:{domain}:manifest={manifest.row_count}:footer={footer_rows}")
        contract_errors, contract_warnings = _manifest_contract_findings(domain, manifest.to_dict())
        errors.extend(contract_errors)
        warnings.extend(contract_warnings)
        dataset_reports.append(
            {
                "domain": domain,
                "layer": manifest.layer,
                "dataset_id": dataset_id,
                "manifest_path": str(manifest_path.resolve()),
                "start_date": manifest.start_date,
                "end_date": manifest.end_date,
                "row_count": manifest.row_count,
                "footer_row_count": footer_rows,
                "shard_count": len(manifest.shards),
                "missing_shard_count": len(missing_shards),
                "footer_error_count": len(footer_errors),
                "quality": manifest.quality,
            }
        )
    payload = {
        "status": "ok" if not errors else "error",
        "qdp_v2_root": str(root.resolve()),
        "active_manifest": str((root / "active" / "active.json").resolve()),
        "audited_at": utc_now(),
        "dataset_count": len(dataset_reports),
        "datasets": dataset_reports,
        "warnings": warnings,
        "errors": errors,
    }
    if write:
        audit_id = f"active_audit_{utc_now().replace(':', '').replace('-', '')}"
        path = root / "audits" / f"{audit_id}.json"
        atomic_write_json(path, payload)
        payload["audit_path"] = str(path.resolve())
    return payload


def _manifest_contract_findings(domain: str, manifest: dict[str, Any]) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    contract = str(manifest.get("contract_version", "") or "")
    dataset_id = str(manifest.get("dataset_id", "") or "")
    quality = dict(manifest.get("quality", {}) or {})
    schema = [str(dict(item).get("name", "") or "") for item in list(manifest.get("schema", []) or []) if isinstance(item, dict)]
    if "pending_rebuild" in contract or "pending_qdp_v2_normalization" in contract or "pending_qdp_v2_split" in contract:
        errors.append(f"pending_contract_active:{domain}:{dataset_id}:{contract}")
    expected_contracts = {
        "market_intraday_1m": "mootdx_1m_240_v1",
        "market_intraday_5m": "mootdx_5m_48_v1",
        "market_daily_raw": "qdp_v2_market_daily_raw_v1",
        "market_daily_panel": "qdp_v2_market_daily_panel_v1",
    }
    expected = expected_contracts.get(domain)
    if expected and contract != expected:
        errors.append(f"wrong_contract:{domain}:{dataset_id}:expected={expected}:actual={contract}")
    if domain == "valuation" and contract not in {"qdp_v2_valuation_v1", "qdp_v2_valuation_v2"}:
        errors.append(f"wrong_contract:{domain}:{dataset_id}:expected=qdp_v2_valuation_v1|qdp_v2_valuation_v2:actual={contract}")
    if domain == "market_intraday_5m" and str(quality.get("bar_count_contract", "") or "") != "48":
        errors.append(f"missing_5m_48_quality:{dataset_id}")
    if domain == "market_daily_raw" and quality.get("ohlcv_non_null") is not True:
        errors.append(f"market_daily_raw_not_marked_ohlcv_non_null:{dataset_id}")
    if domain == "market_daily_panel" and quality.get("has_bar_contract") is not True:
        errors.append(f"market_daily_panel_missing_has_bar_contract:{dataset_id}")
    if domain == "valuation":
        required = {"symbol", "trade_date", "total_mv", "circ_mv", "pe", "pb", "turnover_rate", "source"}
        missing = sorted(required.difference(schema))
        if missing:
            errors.append(f"valuation_schema_missing:{dataset_id}:{','.join(missing)}")
    if domain in {"market_daily_raw", "market_daily_panel", "market_intraday_1m", "market_intraday_5m", "valuation"}:
        if quality.get("primary_key_unique") not in {True, "checked"}:
            warnings.append(f"primary_key_uniqueness_not_deep_checked:{domain}:{dataset_id}")
    return errors, warnings


def _parquet_row_count(path: Path) -> int:
    try:
        import pyarrow.parquet as pq  # type: ignore

        return int(pq.ParquetFile(path).metadata.num_rows)
    except Exception:
        import duckdb  # type: ignore

        with duckdb.connect(":memory:") as con:
            return int(con.execute("select count(*) as n from read_parquet(?)", [str(path)]).fetchone()[0])


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="qdp check --quick", description="Audit qdp_v2 active manifests.")
    parser.add_argument("--workspace-root", default="")
    parser.add_argument("--write-audit", action="store_true", help="Persist an audit JSON; checks are read-only by default.")
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    payload = audit_active(workspace_root=str(args.workspace_root or "") or None, write=bool(args.write_audit))
    print(json.dumps(json_safe(payload), ensure_ascii=False, indent=2) if args.json else _format(payload))
    return 0 if payload.get("status") == "ok" else 2


def _format(payload: dict[str, Any]) -> str:
    lines = [
        f"status: {payload.get('status', '')}",
        f"qdp_v2_root: {payload.get('qdp_v2_root', '')}",
        f"dataset_count: {payload.get('dataset_count', 0)}",
        f"errors: {len(payload.get('errors', []) or [])}",
        f"warnings: {len(payload.get('warnings', []) or [])}",
    ]
    if payload.get("audit_path"):
        lines.append(f"audit_path: {payload.get('audit_path')}")
    return "\n".join(lines)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
