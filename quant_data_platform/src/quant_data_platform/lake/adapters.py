from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import pandas as pd

from daily_research.data_lake.build_canonical_policy_bundle import (
    BuildCanonicalPolicyBundleConfig,
    build_canonical_policy_bundle,
)
from daily_research.data_lake.canonical import (
    DEFAULT_EXTERNAL_QUANT_DATA_ROOT,
    build_lake_inventory,
    load_canonical_manifest,
    write_inventory_report,
)
from daily_research.data_lake.catalog import ResearchDataLake
from daily_research.data_platform.contracts import DataDomain

from quant_data_platform.core.json_io import utc_now
from quant_data_platform.core.paths import QdpPaths, qdp_paths
from quant_data_platform.core.registry import load_root_manifest, write_root_manifest_update
from quant_data_platform.domains.contracts import CANONICAL_BUNDLE_SIDECAR_DOMAINS, CANONICAL_START_DATE


@dataclass(frozen=True)
class BundleBuildSummary:
    status: str
    dataset_id: str
    canonical_manifest_path: str
    sidecar_dataset_ids: dict[str, str]
    missing_sidecar_domains: list[str]
    row_count: int = 0
    membership_symbols: int = 0


def lake(paths: QdpPaths | None = None) -> ResearchDataLake:
    resolved = paths or qdp_paths()
    return ResearchDataLake(resolved.lake_root)


def discover_latest_domain_dataset_ids(
    data_lake: ResearchDataLake,
    domains: Iterable[str],
    *,
    start_date: str = CANONICAL_START_DATE,
) -> dict[str, str]:
    rows = data_lake.list_datasets()
    if rows.empty:
        return {}
    out: dict[str, str] = {}
    expected_start = pd.Timestamp(start_date)
    for domain in domains:
        normalized = str(domain)
        kind = f"data_platform_{normalized}"
        sub = rows.loc[rows["dataset_kind"].astype(str).eq(kind)].copy()
        if sub.empty and "domain" in rows.columns:
            sub = rows.loc[rows["domain"].astype(str).eq(normalized)].copy()
        if sub.empty:
            continue
        sub["_start"] = pd.to_datetime(sub.get("start_date", ""), errors="coerce")
        sub["_end"] = pd.to_datetime(sub.get("end_date", ""), errors="coerce")
        sub["_full_start"] = sub["_start"].le(expected_start).fillna(False)
        sub["_combined"] = sub.get("source", "").astype(str).eq("canonical_reuse_combined")
        sub["_stored"] = sub.get("status", "").astype(str).eq("stored")
        sub = sub.sort_values(
            ["_stored", "_full_start", "_combined", "_end", "_start"],
            ascending=[True, True, True, True, False],
        )
        selected = sub.iloc[-1]
        out[normalized] = str(selected.get("dataset_id", "") or "")
    return {key: value for key, value in out.items() if value}


def canonical_sidecar_dataset_ids(paths: QdpPaths | None = None) -> dict[str, str]:
    resolved = paths or qdp_paths()
    data_lake = lake(resolved)
    root_manifest = load_root_manifest(resolved)
    components = dict(root_manifest.get("canonical_component_dataset_ids", {}) or {})
    current_manifest = load_canonical_manifest(data_lake)
    sidecars = dict(current_manifest.get("sidecar_dataset_ids", {}) or {})
    discovered = discover_latest_domain_dataset_ids(data_lake, CANONICAL_BUNDLE_SIDECAR_DOMAINS)
    out: dict[str, str] = {}
    for domain in CANONICAL_BUNDLE_SIDECAR_DOMAINS:
        out[domain] = str(
            sidecars.get(domain, "")
            or components.get(domain, "")
            or discovered.get(domain, "")
            or ""
        )
    return {key: value for key, value in out.items() if value}


def build_bundle(paths: QdpPaths | None = None, *, write: bool = True) -> BundleBuildSummary:
    resolved = paths or qdp_paths()
    data_lake = lake(resolved)
    root_manifest = load_root_manifest(resolved)
    components = dict(root_manifest.get("canonical_component_dataset_ids", {}) or {})
    market_daily = str(components.get(DataDomain.MARKET_DAILY, "") or root_manifest.get("market_daily_dataset_id", "") or "")
    if not market_daily:
        discovered = discover_latest_domain_dataset_ids(data_lake, [DataDomain.MARKET_DAILY])
        market_daily = str(discovered.get(DataDomain.MARKET_DAILY, "") or "")
    if not market_daily:
        raise ValueError("canonical_market_daily_dataset_id_required")
    sidecars = canonical_sidecar_dataset_ids(resolved)
    missing = [domain for domain in CANONICAL_BUNDLE_SIDECAR_DOMAINS if domain not in sidecars]
    if missing:
        raise ValueError(f"canonical_bundle_missing_sidecars: {missing}")
    if not write:
        return BundleBuildSummary(
            status="dry_run",
            dataset_id=str(root_manifest.get("canonical_dataset_id", "") or ""),
            canonical_manifest_path=str(root_manifest.get("canonical_manifest", "") or ""),
            sidecar_dataset_ids=sidecars,
            missing_sidecar_domains=missing,
        )
    result = build_canonical_policy_bundle(
        BuildCanonicalPolicyBundleConfig(
            lake_root=resolved.lake_root,
            market_daily_dataset_id=market_daily,
            sidecar_dataset_ids=sidecars,
            start_date=CANONICAL_START_DATE,
            benchmark="000300.SH",
            benchmark_source_dataset_id=str(components.get("benchmark_daily", "") or ""),
            source="canonical_data_v1",
            alias="canonical_data_v1",
            write_manifest=True,
            reuse=True,
        )
    )
    component_ids = {
        **components,
        DataDomain.MARKET_DAILY: market_daily,
        **sidecars,
    }
    write_root_manifest_update(
        {
            "status": "canonical_bundle_ready_full_memmap_pending",
            "canonical_dataset_id": result.dataset_id,
            "canonical_manifest": result.canonical_manifest_path,
            "canonical_component_dataset_ids": component_ids,
            "canonical_bundle_sidecar_dataset_ids": sidecars,
            "structural_style_domains": [
                DataDomain.VALUATION,
                DataDomain.INDUSTRY_CONCEPT,
                DataDomain.INDEX_CONSTITUENTS,
            ],
            "default_feature_policy": {
                "research_style": "profile_selected",
                "canonical_included_domains": [
                    DataDomain.MARKET_DAILY,
                    DataDomain.MARKET_INTRADAY_5M,
                    DataDomain.INTRADAY_DAILY_FEATURES,
                    DataDomain.ADJUST_FACTOR,
                    DataDomain.VALUATION,
                    DataDomain.INDUSTRY_CONCEPT,
                    DataDomain.INDEX_CONSTITUENTS,
                    DataDomain.TRADING_CALENDAR,
                    DataDomain.UNIVERSE_SNAPSHOT,
                    DataDomain.SECURITY_STATUS,
                ],
                "short_horizon_core_v1_domains": [
                    DataDomain.MARKET_DAILY,
                    DataDomain.INTRADAY_DAILY_FEATURES,
                    DataDomain.ADJUST_FACTOR,
                    DataDomain.TRADING_CALENDAR,
                    DataDomain.UNIVERSE_SNAPSHOT,
                    DataDomain.SECURITY_STATUS,
                ],
                "style_structural_v1_extra_domains": [
                    DataDomain.VALUATION,
                    DataDomain.INDUSTRY_CONCEPT,
                    DataDomain.INDEX_CONSTITUENTS,
                ],
                "v1_excluded_domains": [
                    DataDomain.FINANCIAL_QUARTERLY,
                    DataDomain.PERFORMANCE_FORECAST,
                    DataDomain.PERFORMANCE_EXPRESS,
                ],
                "notes": (
                    "Canonical keeps valuation, industry, and index membership as auditable structure/style data. "
                    "Feature profiles decide usage; short_horizon_core_v1 does not use them, style_structural_v1 and "
                    "medium_horizon_v1 may use them. Slow disclosure financial/performance domains stay out of v1."
                ),
            },
            "v1_excluded_domains": [
                DataDomain.FINANCIAL_QUARTERLY,
                DataDomain.PERFORMANCE_FORECAST,
                DataDomain.PERFORMANCE_EXPRESS,
            ],
            "bundle_updated_at": utc_now(),
        },
        paths=resolved,
    )
    return BundleBuildSummary(
        status=result.status,
        dataset_id=result.dataset_id,
        canonical_manifest_path=result.canonical_manifest_path,
        sidecar_dataset_ids=dict(result.sidecar_dataset_ids),
        missing_sidecar_domains=missing,
        row_count=int(result.row_count),
        membership_symbols=int(result.membership_symbols),
    )


def audit_inventory(paths: QdpPaths | None = None, *, write: bool = True) -> dict[str, Any]:
    resolved = paths or qdp_paths()
    inventory = build_lake_inventory(
        resolved.lake_root,
        external_data_root=DEFAULT_EXTERNAL_QUANT_DATA_ROOT,
        path_policy_root=resolved.workspace_root / "daily_research" / "output" / "path_policy",
        start_date=CANONICAL_START_DATE,
    )
    inventory = _protect_active_cleanup_candidates(inventory, paths=resolved)
    if write:
        report_path = write_inventory_report(inventory, output_dir=resolved.audits_dir)
        inventory["report_path"] = str(Path(report_path).as_posix())
    return inventory


def cleanup_dry_run(paths: QdpPaths | None = None) -> dict[str, Any]:
    inventory = audit_inventory(paths, write=True)
    return {
        "status": "dry_run",
        "destructive_actions_performed": False,
        "cleanup_dry_run": dict(inventory.get("cleanup_dry_run", {}) or {}),
        "report_path": str(inventory.get("report_path", "") or ""),
    }


def bundle_summary_dict(summary: BundleBuildSummary) -> dict[str, Any]:
    return asdict(summary)


def _protect_active_cleanup_candidates(inventory: Mapping[str, Any], *, paths: QdpPaths) -> dict[str, Any]:
    protected = dict(inventory)
    root_manifest = load_root_manifest(paths)
    active_dataset_id = str(root_manifest.get("canonical_dataset_id", "") or "")
    cleanup = dict(protected.get("cleanup_dry_run", {}) or {})
    bundle_candidates = [
        str(item)
        for item in list(cleanup.get("policy_input_bundles_to_mark_deprecated_after_canonical_cutover", []) or [])
    ]
    if active_dataset_id and active_dataset_id in bundle_candidates:
        cleanup["policy_input_bundles_to_mark_deprecated_after_canonical_cutover"] = [
            item for item in bundle_candidates if item != active_dataset_id
        ]
        cleanup["protected_current_canonical_bundle"] = active_dataset_id
    protected["cleanup_dry_run"] = cleanup
    return protected
