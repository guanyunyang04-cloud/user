from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from quantlab.data.core.json_io import json_safe
from quantlab.data.qdp_v2.manifest import (
    _manifest_schema_from_arrow,
    atomic_write_json,
    canonical_manifest_schema,
    dataset_manifest_for_id,
    qdp_v2_root,
    read_active_manifest,
    read_dataset_manifest,
    utc_now,
)
from quantlab.data.qdp_v2.status import _active_dataset_refs, _manifest_path

VALUATION_REQUIRED_COLUMNS = {
    "symbol",
    "trade_date",
    "total_mv",
    "circ_mv",
    "pe",
    "pb",
    "turnover_rate",
    "source",
}


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
        declared_schema = list(manifest.schema)
        declared_names = [str(item.get("name", "") or "") for item in declared_schema]
        # A legacy declaration carries no convertible ``type``, so only its
        # column names and order can be compared against a Parquet footer.
        # Normalizing a typed declaration keeps equivalent spellings such as
        # ``string``/``VARCHAR`` from being reported as drift.
        canonical_schema = canonical_manifest_schema(declared_schema)
        if not declared_schema:
            footer_errors.append(f"manifest_schema_missing:{domain}:{dataset_id}")
        elif canonical_schema is None:
            warnings.append(
                f"legacy_untyped_manifest_schema:{domain}:{dataset_id}"
            )
        for shard in manifest.shards:
            shard_path = _manifest_path(shard.path, root)
            if not shard_path.is_file():
                missing_shards.append(str(shard_path))
                continue
            try:
                parquet = _parquet_footer(shard_path)
                footer_schema = _manifest_schema_from_arrow(parquet.schema_arrow)
                footer_names = [item["name"] for item in footer_schema]
                if footer_names != declared_names and set(footer_names) == set(declared_names):
                    footer_errors.append(
                        f"schema_column_order_mismatch:{shard.path}:"
                        f"manifest={declared_names}:footer={footer_names}"
                    )
                elif footer_names != declared_names:
                    footer_errors.append(
                        f"schema_column_set_mismatch:{shard.path}:"
                        f"manifest={declared_names}:footer={footer_names}"
                    )
                elif canonical_schema is not None and footer_schema != canonical_schema:
                    footer_errors.append(f"footer_schema_mismatch:{shard.path}")
                if verify_footers:
                    rows = int(parquet.metadata.num_rows)
                    footer_rows += int(rows)
                    if shard.row_count > 0 and rows > 0 and int(rows) != int(shard.row_count):
                        footer_errors.append(f"row_count_mismatch:{shard.path}:manifest={shard.row_count}:footer={rows}")
            except Exception as exc:  # noqa: BLE001 - an audit must report every unreadable shard
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
        "scope": "active_manifest_files_schema_and_declared_contracts",
        "all_active_domains_semantically_certified": False,
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
    expected_contracts = {
        "market_intraday_5m": ("qdp_current_intraday_5m_48_v1",),
        "market_daily_raw": ("qdp_v2_market_daily_raw_v1",),
        "margin_eligibility": (
            "qdp_v2_margin_eligibility_exchange_tristate_v1",
            "qdp_v2_margin_eligibility_exchange_tristate_v2",
        ),
    }
    expected = expected_contracts.get(domain, ())
    if expected and contract not in expected:
        expected_text = "|".join(expected)
        errors.append(f"wrong_contract:{domain}:{dataset_id}:expected={expected_text}:actual={contract}")
    valuation_contracts = {
        "qdp_v2_valuation_v1",
        "qdp_v2_valuation_v2",
        "qdp_v2_valuation_strict_pit_v3",
        "qdp_v2_valuation_formula_v4",
    }
    if domain == "valuation" and contract not in valuation_contracts:
        expected_text = "|".join(sorted(valuation_contracts))
        errors.append(
            f"wrong_contract:{domain}:{dataset_id}:"
            f"expected={expected_text}:actual={contract}"
        )
    if domain == "market_intraday_5m" and str(quality.get("bar_count_contract", "") or "") != "48":
        errors.append(f"missing_5m_48_quality:{dataset_id}")
    if domain == "market_daily_raw" and quality.get("ohlcv_non_null") is not True:
        errors.append(f"market_daily_raw_not_marked_ohlcv_non_null:{dataset_id}")
    if domain == "valuation":
        missing = sorted(VALUATION_REQUIRED_COLUMNS.difference(schema))
        if missing:
            errors.append(f"valuation_schema_missing:{dataset_id}:{','.join(missing)}")
    if domain in {
        "market_daily_raw",
        "market_intraday_5m",
        "valuation",
    } and quality.get("primary_key_unique") not in {True, "checked"}:
        warnings.append(f"primary_key_uniqueness_not_deep_checked:{domain}:{dataset_id}")
    return errors, warnings


def _parquet_row_count(path: Path) -> int:
    try:
        import pyarrow.parquet as pq  # type: ignore

        return int(pq.ParquetFile(path).metadata.num_rows)
    except Exception:  # noqa: BLE001 - DuckDB is the deliberate fallback for any PyArrow failure
        import duckdb  # type: ignore

        with duckdb.connect(":memory:") as con:
            return int(con.execute("select count(*) as n from read_parquet(?)", [str(path)]).fetchone()[0])


def _parquet_footer(path: Path) -> Any:
    import pyarrow.parquet as pq  # type: ignore

    return pq.ParquetFile(path)


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
