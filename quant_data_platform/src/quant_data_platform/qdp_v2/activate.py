from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from quant_data_platform.core.json_io import json_safe
from quant_data_platform.qdp_v2.manifest import ACTIVE_MANIFEST_VERSION, iter_dataset_manifests, qdp_v2_root, read_dataset_manifest, utc_now, write_active_manifest


RAW_DOMAINS = (
    "market_daily_raw",
    "market_intraday_1m",
    "market_intraday_5m",
    "trading_calendar",
    "universe_snapshot",
    "security_status",
    "valuation",
    "adjust_factor",
    "industry_concept",
    "index_constituents",
    "announcement",
)
DERIVED_DOMAINS = ("intraday_daily_features", "v2_status_sidecar")
RESEARCH_PANEL_DOMAINS = ("market_daily_panel",)
REQUIRED_DOMAINS = (
    "market_daily_raw",
    "market_intraday_1m",
    "market_intraday_5m",
    "trading_calendar",
    "universe_snapshot",
    "security_status",
    "valuation",
    "adjust_factor",
    "intraday_daily_features",
    "market_daily_panel",
)
PREFERRED_CONTRACTS = {
    "market_intraday_1m": ("mootdx_1m_240_v1",),
    "market_intraday_5m": ("mootdx_5m_48_v1",),
    "valuation": ("qdp_v2_valuation_v1",),
    "market_daily_raw": ("qdp_v2_market_daily_raw_v1",),
}


def activate_v2(
    *,
    workspace_root: str | Path | None = None,
    as_of_date: str = "",
    yes: bool = False,
) -> dict[str, Any]:
    root = qdp_v2_root(workspace_root)
    by_domain: dict[str, list[dict[str, Any]]] = {}
    for path in iter_dataset_manifests(root):
        manifest = read_dataset_manifest(path)
        by_domain.setdefault(manifest.domain, []).append(
            {
                "dataset_id": manifest.dataset_id,
                "domain": manifest.domain,
                "contract_version": manifest.contract_version,
                "start_date": manifest.start_date,
                "end_date": manifest.end_date,
                "created_at": manifest.created_at,
                "row_count": manifest.row_count,
                "manifest_path": str(path.resolve()),
            }
        )
    selected: dict[str, str] = {}
    errors: list[str] = []
    warnings: list[str] = []
    for domain in (*RAW_DOMAINS, *DERIVED_DOMAINS, *RESEARCH_PANEL_DOMAINS):
        candidates = by_domain.get(domain, [])
        if not candidates:
            if domain in REQUIRED_DOMAINS:
                errors.append(f"required_domain_missing:{domain}")
            continue
        candidate = _select_candidate(domain, candidates)
        if _is_pending_contract(str(candidate.get("contract_version", "") or "")):
            if domain in REQUIRED_DOMAINS:
                errors.append(f"required_domain_pending_contract:{domain}:{candidate.get('dataset_id')}:{candidate.get('contract_version')}")
            else:
                warnings.append(f"optional_domain_pending_contract:{domain}:{candidate.get('dataset_id')}:{candidate.get('contract_version')}")
                continue
        selected[domain] = str(candidate["dataset_id"])
    active_as_of = str(as_of_date or "").strip() or _infer_active_as_of(by_domain, selected)
    payload = {
        "version": ACTIVE_MANIFEST_VERSION,
        "active_as_of_date": active_as_of,
        "raw": {domain: selected[domain] for domain in RAW_DOMAINS if domain in selected},
        "derived": {domain: selected[domain] for domain in DERIVED_DOMAINS if domain in selected},
        "research_panels": {domain: selected[domain] for domain in RESEARCH_PANEL_DOMAINS if domain in selected},
        "memmap": {"active_manifest": "", "status": "not_part_of_data_base"},
        "source": {"created_by": "qdp activate-v2", "created_at": utc_now()},
    }
    result = {
        "status": "blocked" if errors else ("activated" if yes else "dry_run"),
        "qdp_v2_root": str(root.resolve()),
        "active_manifest": "",
        "active": payload,
        "selected": selected,
        "errors": errors,
        "warnings": warnings,
    }
    if errors or not yes:
        return result
    path = write_active_manifest(root, payload)
    result["active_manifest"] = str(path.resolve())
    return result


def _select_candidate(domain: str, candidates: list[dict[str, Any]]) -> dict[str, Any]:
    preferred = set(PREFERRED_CONTRACTS.get(domain, ()))

    def key(item: dict[str, Any]) -> tuple[int, str, str]:
        contract = str(item.get("contract_version", "") or "")
        return (1 if contract in preferred else 0, str(item.get("end_date", "") or ""), str(item.get("created_at", "") or ""))

    valid = [item for item in candidates if not _is_pending_contract(str(item.get("contract_version", "") or ""))]
    pool = valid or candidates
    return sorted(pool, key=key, reverse=True)[0]


def _is_pending_contract(contract: str) -> bool:
    return "pending_rebuild" in contract or "pending_qdp_v2_normalization" in contract


def _infer_active_as_of(by_domain: dict[str, list[dict[str, Any]]], selected: dict[str, str]) -> str:
    ends: list[str] = []
    for domain in REQUIRED_DOMAINS:
        dataset_id = selected.get(domain)
        if not dataset_id:
            continue
        for item in by_domain.get(domain, []):
            if item.get("dataset_id") == dataset_id and item.get("end_date"):
                ends.append(str(item["end_date"]))
                break
    return min(ends) if ends else ""


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="qdp activate-v2", description="Atomically activate a qdp_v2 manifest set.")
    parser.add_argument("--workspace-root", default="")
    parser.add_argument("--as-of-date", default="")
    parser.add_argument("--yes", action="store_true", help="Write active/active.json. Without --yes this is a dry-run.")
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    payload = activate_v2(workspace_root=str(args.workspace_root or "") or None, as_of_date=str(args.as_of_date or ""), yes=bool(args.yes))
    if bool(args.json):
        print(json.dumps(json_safe(payload), ensure_ascii=False, indent=2))
    else:
        print("\n".join(f"{key}: {value}" for key, value in payload.items() if key != "active"))
    return 0 if payload.get("status") in {"dry_run", "activated"} else 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
