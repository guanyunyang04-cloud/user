"""Compact final checklist for QDP specialty data repairs."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import duckdb

from quant_data_platform.qdp_v2.manifest import (
    atomic_write_json,
    dataset_manifest_for_id,
    qdp_v2_root,
    read_active_manifest,
    read_dataset_manifest,
    resolve_manifest_path,
    utc_now,
)
from quant_data_platform.qdp_v2.status import active_dataset_map

SPECIALTY_AUDITS = (
    "research_report_rc_backfill_v2.json",
    "financial_statement_quarterly_pit_v2.json",
    "balance_extension_conflict_semantics_repair_v2.json",
    "share_capital_total_share_repair_v1.json",
    "share_capital_float_share_invariant_repair_v1.json",
    "margin_eligibility_sse_semantics_repair_v1.json",
    "historical_intraday_5m_external_archive_repair_v2.json",
    "tushare_extended_backfill_v1.json",
)
TERMINAL_OK_STATUSES = {"applied", "ok", "completed", "already_repaired"}


def _specialty_record(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {
            "name": path.name,
            "status": "missing",
            "ok": False,
            "failed_checks": ["audit_file_missing"],
        }
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, ValueError) as exc:
        return {
            "name": path.name,
            "status": "unreadable",
            "ok": False,
            "failed_checks": [f"audit_unreadable:{type(exc).__name__}"],
        }
    if not isinstance(payload, Mapping):
        return {
            "name": path.name,
            "status": "invalid",
            "ok": False,
            "failed_checks": ["audit_payload_not_mapping"],
        }
    status = str(payload.get("status", "") or "")
    checks = dict(payload.get("checks", {}) or {})
    failed = sorted(str(key) for key, value in checks.items() if value is not True)
    statistics = payload.get("statistics", {})
    if not isinstance(statistics, Mapping):
        statistics = {}
    request_2026 = payload.get(
        "request_2026_count", statistics.get("request_2026_count")
    )
    if request_2026 is not None and int(request_2026 or 0) != 0:
        failed.append("request_2026_count_nonzero")
    return {
        "name": path.name,
        "path": str(path.resolve()),
        "status": status,
        "ok": status in TERMINAL_OK_STATUSES and not failed,
        "check_count": len(checks),
        "failed_checks": failed,
    }


def _active_domain(
    root: Path, active: Mapping[str, str], domain: str
) -> tuple[Any, list[Path]]:
    dataset_id = str(active.get(domain, ""))
    manifest_path = dataset_manifest_for_id(root, dataset_id, domain)
    if manifest_path is None:
        raise FileNotFoundError(f"active_manifest_missing:{domain}:{dataset_id}")
    manifest = read_dataset_manifest(manifest_path)
    paths = [resolve_manifest_path(item.path, root=root) for item in manifest.shards]
    if not paths or any(not path.is_file() for path in paths):
        raise FileNotFoundError(f"active_shards_missing:{domain}:{dataset_id}")
    return manifest, paths


def _scan(paths: Sequence[Path]) -> str:
    quoted = ",".join(
        "'" + path.resolve().as_posix().replace("'", "''") + "'" for path in paths
    )
    return f"read_parquet([{quoted}], union_by_name=true, hive_partitioning=false)"


def _semantic_invariants(root: Path, active: Mapping[str, str]) -> dict[str, Any]:
    margin_manifest, margin_paths = _active_domain(root, active, "margin_eligibility")
    balance_manifest, balance_paths = _active_domain(
        root, active, "balance_sheet_quarterly"
    )
    margin_scan = _scan(margin_paths)
    balance_scan = _scan(balance_paths)
    with duckdb.connect() as connection:
        margin = connection.execute(
            "SELECT count(*),"
            "count(*) FILTER(WHERE exchange='SH' AND eligibility_state='known_ineligible'),"
            "count(*) FILTER(WHERE exchange='SH' AND eligibility_state='source_unavailable'),"
            "count(*) FILTER(WHERE exchange='SH' AND NOT detail_observed),"
            "count(*) FILTER(WHERE NOT coalesce(("
            " (eligibility_state='eligible_observed' AND eligible IS true) OR"
            " (eligibility_state='known_ineligible' AND eligible IS false) OR"
            " (eligibility_state='source_unavailable' AND eligible IS NULL)),false)) "
            f"FROM {margin_scan}"
        ).fetchone()
        balance_columns = [
            str(row[0])
            for row in connection.execute(
                f"DESCRIBE SELECT * FROM {balance_scan}"
            ).fetchall()
        ]
        conflict_count = (
            int(
                connection.execute(
                    "SELECT count(*) FILTER(WHERE "
                    "balance_extension_source_conflict) "
                    f"FROM {balance_scan}"
                ).fetchone()[0]
                or 0
            )
            if "balance_extension_source_conflict" in balance_columns
            else -1
        )
    checks = {
        "margin_contract_is_corrected_v2": margin_manifest.contract_version
        == "qdp_v2_margin_eligibility_exchange_tristate_v2",
        "sse_known_ineligible_count_zero": int(margin[1]) == 0,
        "sse_unknown_count_matches_absent_detail": int(margin[2]) == int(margin[3]),
        "margin_nullable_state_contract": int(margin[4]) == 0,
        "balance_contract_has_exact_extension_conflicts": (
            balance_manifest.contract_version == "qdp_v2_balance_sheet_quarterly_pit_v4"
        ),
        "balance_extension_conflict_flag_present": conflict_count >= 0,
        "balance_extension_conflict_count_matches_manifest": conflict_count
        == int(
            dict(balance_manifest.quality or {}).get(
                "balance_extension_source_conflict_count", -1
            )
        ),
    }
    return {
        "checks": checks,
        "margin_eligibility": {
            "dataset_id": margin_manifest.dataset_id,
            "contract_version": margin_manifest.contract_version,
            "row_count": int(margin[0]),
            "sse_source_unavailable_count": int(margin[2]),
        },
        "balance_sheet_quarterly": {
            "dataset_id": balance_manifest.dataset_id,
            "contract_version": balance_manifest.contract_version,
            "extension_source_conflict_count": conflict_count,
        },
    }


def audit_semantics(
    *, workspace_root: str | Path | None = None, write: bool = False
) -> dict[str, Any]:
    root = qdp_v2_root(workspace_root)
    active = active_dataset_map(read_active_manifest(root))
    audits = [_specialty_record(root / "audits" / name) for name in SPECIALTY_AUDITS]
    errors = [
        f"specialty_audit_failed:{item['name']}:{','.join(item['failed_checks'])}"
        for item in audits
        if not item["ok"]
    ]
    try:
        invariants = _semantic_invariants(root, active)
        errors.extend(
            f"semantic_invariant_failed:{name}"
            for name, passed in invariants["checks"].items()
            if not passed
        )
    except (FileNotFoundError, duckdb.Error) as exc:
        invariants = {"checks": {}}
        errors.append(f"semantic_invariant_scan_failed:{type(exc).__name__}:{exc}")
    payload = {
        "status": "ok" if not errors else "needs_attention",
        "scope": "specialty_audit_aggregation_and_selected_stable_invariants",
        "all_active_domains_semantically_certified": False,
        "qdp_v2_root": str(root.resolve()),
        "audited_at": utc_now(),
        "specialty_audits": audits,
        "invariants": invariants,
        "errors": errors,
    }
    if write:
        path = root / "audits" / "final_semantic_audit_v2.json"
        atomic_write_json(path, payload)
        payload["audit_path"] = str(path.resolve())
    return payload
