from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable
import os
import sqlite3
import tempfile

import pandas as pd

from quant_data_platform.core.json_io import read_json
from quant_data_platform.qdp_v3.constants import (
    DOMAIN_ADJUST_FACTOR_DAILY,
    DOMAIN_ADJUST_FACTOR_EVENT,
    DOMAIN_CORPORATE_ACTIONS,
    DOMAIN_FINANCIAL_QUARTERLY,
    DOMAIN_INDEX_CONSTITUENTS,
    DOMAIN_INDUSTRY,
    DOMAIN_ELIGIBLE_SIGNAL_D,
    DOMAIN_MARKET_DAILY_RAW,
    DOMAIN_MARKET_INTRADAY_5M,
    DOMAIN_PERFORMANCE_EXPRESS,
    DOMAIN_PERFORMANCE_FORECAST,
    DOMAIN_SECURITY_IDENTITY,
    DOMAIN_SECURITY_STATUS_DAILY,
    DOMAIN_SHARE_CAPITAL_DAILY,
    DOMAIN_SHARE_CAPITAL_EVENT,
    DOMAIN_SYMBOL_HISTORY,
    DOMAIN_TRADABLE_OPEN_D1,
    DOMAIN_TRADING_CALENDAR,
    DOMAIN_VALUATION_DAILY,
    QDP_V3_CONTRACT_VERSION,
    QUALITY_PROVISIONAL,
    QUALITY_QUARANTINED,
    QUALITY_STRICT,
    RAW_ADJUST_FACTOR_EVENT,
    RAW_ADJUST_FACTOR_SYMBOL_HISTORY,
    RAW_ALL_STOCK,
    RAW_CORPORATE_ACTION_XDXR,
    RAW_DAILY_ASTOCK,
    RAW_INTRADAY_5M_SELECTED,
    RAW_SECURITY_MASTER,
    RAW_TRADING_CALENDAR,
)
from quant_data_platform.qdp_v3.corporate_actions import (
    FACTOR_CANONICAL_COLUMNS,
    arbitrate_adjust_factor_disputes,
    build_share_capital_daily,
    canonicalize_mootdx_xdxr,
    load_official_factor_evidence,
    reconstruct_xdxr_reference_prices,
)
from quant_data_platform.qdp_v3.datasets import dataset_input_ref, write_dataset, write_partitioned_dataset
from quant_data_platform.qdp_v3.identity import SecurityIdentityRegistry, board_for_symbol, normalize_symbol
from quant_data_platform.qdp_v3.intraday import canonicalize_selected_5m, is_complete_5m_day
from quant_data_platform.qdp_v3.manifest import ProviderEvidence, dataset_manifest_for_id, manifest_sha256
from quant_data_platform.qdp_v3.paths import ensure_qdp_v3_layout
from quant_data_platform.qdp_v3.quality import (
    QualityFinding,
    QualityReport,
    audit_canonical_daily,
    audit_factor_semantics,
    audit_intraday_5m,
    audit_table_contract,
    report_for,
)
from quant_data_platform.qdp_v3.release import CandidateManifestV3, write_candidate
from quant_data_platform.qdp_v3.secondary import canonicalize_secondary_domain, refs_for_secondary_domain
from quant_data_platform.qdp_v3.storage import RawPartitionRef, iter_raw_partitions, read_raw_partition, read_raw_receipt
from quant_data_platform.qdp_v3.storage import atomic_write_parquet
from quant_data_platform.qdp_v3.transforms import (
    build_adjust_factor_daily,
    build_eligible_signal_view,
    build_signal_and_open_pit_views,
    build_tradable_open_view,
    derive_adjust_factor_events,
    derive_daily_domains,
    derive_symbol_adjust_factor_events,
    reconcile_adjust_factor_events,
    trading_calendar_from_snapshot_dates,
)


def _finding_from_mapping(payload: dict[str, Any], *, domain: str) -> QualityFinding:
    return QualityFinding(
        code=str(payload.get("code", "upstream_quality_finding") or "upstream_quality_finding"),
        severity=str(payload.get("severity", "warning") or "warning"),
        message=str(payload.get("message", "") or ""),
        domain=str(payload.get("domain", "") or domain),
        count=int(payload.get("count", 0) or 0),
        sample=list(payload.get("sample", []) or [])[:20],
        provisional=bool(payload.get("provisional", False)),
    )


def _raw_quality_findings(refs: Iterable[RawPartitionRef], *, domain: str) -> list[QualityFinding]:
    findings: list[QualityFinding] = []
    for ref in refs:
        receipt = read_raw_receipt(ref)
        report = dict(receipt.get("quality_report", {}) or {})
        for item in list(report.get("findings", []) or []):
            if isinstance(item, dict):
                findings.append(_finding_from_mapping(item, domain=domain))
        if ref.quality_tier == QUALITY_QUARANTINED and not list(report.get("blockers", []) or []):
            findings.append(
                QualityFinding(
                    code="upstream_raw_partition_quarantined",
                    severity="blocker",
                    message=f"Raw partition {ref.partition_value} is quarantined.",
                    domain=domain,
                    count=1,
                    sample=[ref.partition_value],
                )
            )
        elif ref.quality_tier == QUALITY_PROVISIONAL and not any(bool(item.get("provisional", False)) for item in list(report.get("findings", []) or []) if isinstance(item, dict)):
            findings.append(
                QualityFinding(
                    code="upstream_raw_partition_provisional",
                    severity="warning",
                    message=f"Raw partition {ref.partition_value} lacks strict source proof.",
                    domain=domain,
                    count=1,
                    sample=[ref.partition_value],
                    provisional=True,
                )
            )
    return findings


def _merge_reports(domain: str, reports: Iterable[QualityReport], extra: Iterable[QualityFinding] = ()) -> QualityReport:
    findings: list[QualityFinding] = []
    metrics: dict[str, Any] = {}
    for report in reports:
        findings.extend(report.findings)
        metrics.update({f"{report.domain}.{key}": value for key, value in report.metrics.items()})
    findings.extend(extra)
    aggregated: dict[tuple[str, str, str, str, bool], dict[str, Any]] = {}
    for finding in findings:
        key = (finding.code, finding.severity, finding.message, finding.domain, finding.provisional)
        bucket = aggregated.setdefault(key, {"count": 0, "sample": [], "seen": set()})
        bucket["count"] += int(finding.count)
        for sample in finding.sample:
            marker = repr(sample)
            if marker not in bucket["seen"] and len(bucket["sample"]) < 20:
                bucket["seen"].add(marker)
                bucket["sample"].append(sample)
    compact = [
        QualityFinding(
            code=key[0],
            severity=key[1],
            message=key[2],
            domain=key[3],
            count=int(value["count"]),
            sample=list(value["sample"]),
            provisional=key[4],
        )
        for key, value in aggregated.items()
    ]
    return report_for(domain, compact, metrics=metrics)


def _provider_evidence(refs: list[RawPartitionRef]) -> list[ProviderEvidence]:
    if not refs:
        return []
    receipts = [read_json(ref.receipt_path) for ref in refs]
    by_endpoint: dict[tuple[str, str, str, str], list[tuple[RawPartitionRef, dict[str, Any]]]] = {}
    for ref, receipt in zip(refs, receipts):
        key = (
            str(receipt.get("provider", "") or ""),
            str(receipt.get("endpoint", "") or ""),
            str(receipt.get("package_version", "") or ""),
            str(receipt.get("wheel_sha256", "") or ""),
        )
        by_endpoint.setdefault(key, []).append((ref, receipt))
    evidence: list[ProviderEvidence] = []
    for (provider, endpoint, package_version, wheel_sha), items in sorted(by_endpoint.items()):
        values = sorted(ref.partition_value for ref, _ in items)
        collected = sorted(str(receipt.get("stored_at", "") or receipt.get("requested_at", "") or "") for _, receipt in items)
        evidence.append(
            ProviderEvidence(
                provider=provider,
                endpoint=endpoint,
                package_version=package_version,
                wheel_sha256=wheel_sha,
                request_range={"start": values[0], "end": values[-1], "partition_count": len(values)},
                collected_at=collected[-1] if collected else "",
                metadata={"first_collected_at": collected[0] if collected else ""},
            )
        )
    return evidence


def _latest_security_master(workspace_root: str | Path | None) -> tuple[pd.DataFrame, RawPartitionRef | None]:
    refs = iter_raw_partitions(RAW_SECURITY_MASTER, workspace_root=workspace_root)
    if not refs:
        return pd.DataFrame(), None
    ref = refs[-1]
    return read_raw_partition(ref), ref


def _pit_name_observations(refs: Iterable[RawPartitionRef]) -> tuple[pd.DataFrame, set[str]]:
    """Reduce daily all-stock snapshots to first/name-change observations."""

    records: list[dict[str, str]] = []
    previous_names: dict[str, str] = {}
    provider_symbols: set[str] = set()
    for ref in sorted(refs, key=lambda item: item.partition_value):
        frame = read_raw_partition(ref)
        if frame.empty or "symbol" not in frame.columns or "name" not in frame.columns:
            continue
        current = frame.loc[:, ["symbol", "name"]].copy()
        current["symbol"] = current["symbol"].map(normalize_symbol)
        current["name"] = current["name"].fillna("").astype(str).str.strip()
        current = current.loc[current["symbol"].ne("")].drop_duplicates("symbol", keep="last")
        provider_symbols.update(current["symbol"].astype(str).tolist())
        for item in current.itertuples(index=False):
            symbol, name = str(item.symbol), str(item.name)
            if name and previous_names.get(symbol) != name:
                records.append({"symbol": symbol, "trade_date": ref.partition_value, "name": name})
            if name:
                previous_names[symbol] = name
    return pd.DataFrame(records, columns=["symbol", "trade_date", "name"]), provider_symbols


def _calendar_frame(workspace_root: str | Path | None, snapshot_dates: list[str]) -> tuple[pd.DataFrame, list[RawPartitionRef], list[QualityFinding]]:
    refs = iter_raw_partitions(RAW_TRADING_CALENDAR, workspace_root=workspace_root)
    findings: list[QualityFinding] = []
    if not refs:
        findings.append(
            QualityFinding(
                code="trading_calendar_provider_evidence_missing",
                severity="warning",
                message="Calendar was inferred from snapshot dates rather than query_trade_dates.",
                domain=DOMAIN_TRADING_CALENDAR,
                provisional=True,
            )
        )
        return trading_calendar_from_snapshot_dates(snapshot_dates, source="qdp_v3_snapshot_date_inference"), [], findings
    frames = [read_raw_partition(ref) for ref in refs]
    calendar = pd.concat(frames, ignore_index=True)
    calendar["trade_date"] = calendar["trade_date"].astype(str).str.slice(0, 10)
    calendar = calendar.drop_duplicates(["trade_date", "exchange"], keep="last").sort_values(["trade_date", "exchange"]).reset_index(drop=True)
    return calendar, refs, findings


def _read_selected_intraday_group(
    refs: list[RawPartitionRef],
    *,
    registry: SecurityIdentityRegistry,
    start_date: str,
    end_date: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    frames: list[pd.DataFrame] = []
    for ref in refs:
        frame = read_raw_partition(ref)
        if frame.empty:
            continue
        dates = frame["trade_date"].astype(str).str.slice(0, 10)
        frames.append(frame.loc[dates.between(start_date, end_date)].copy())
    raw = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return canonicalize_selected_5m(raw, identity_registry=registry)


def _build_intraday_dataset(
    *,
    paths: Any,
    workspace_root: str | Path | None,
    registry: SecurityIdentityRegistry,
    status_parts: list[tuple[str, Path]],
    inputs: list[Any],
    start_date: str,
    end_date: str,
) -> tuple[Any | None, list[RawPartitionRef], dict[str, Any]]:
    if str(end_date) < "2020-01-01":
        return None, [], {
            "findings": [],
            "coverage": {
                "release_applicable": False,
                "required_start_date": "2020-01-01",
                "end_date": str(end_date),
                "expected_stock_day_count": 0,
                "strict_covered_stock_day_count": 0,
                "strict_missing_stock_day_count": 0,
                "strict_coverage_rate": 0.0,
            },
            "quarantine": [],
            "quarantine_count": 0,
            "missing_sample": [],
        }
    refs = iter_raw_partitions(RAW_INTRADAY_5M_SELECTED, workspace_root=workspace_root)
    if not refs:
        return None, [], {
            "findings": [
                QualityFinding(
                    code="strict_5m_raw_missing",
                    severity="blocker",
                    message="No selected mootdx/BaoStock 5-minute raw partitions exist.",
                    domain=DOMAIN_MARKET_INTRADAY_5M,
                )
            ],
            "coverage": {"expected_stock_day_count": 0, "strict_covered_stock_day_count": 0, "strict_coverage_rate": 0.0},
            "quarantine": [],
        }

    groups: dict[tuple[str, str], list[RawPartitionRef]] = {}
    relevant_refs: list[RawPartitionRef] = []
    for ref in refs:
        frame = read_raw_partition(ref)
        if frame.empty or "trade_date" not in frame.columns or "provider_symbol" not in frame.columns:
            continue
        dates = frame["trade_date"].astype(str).str.slice(0, 10)
        relevant = frame.loc[dates.between(start_date, end_date), ["provider_symbol", "trade_date"]]
        if relevant.empty:
            continue
        relevant_refs.append(ref)
        for row in relevant.drop_duplicates().itertuples(index=False):
            provider_symbol = str(row.provider_symbol)
            security_id = registry.security_id_for_provider_symbol(provider_symbol) or f"UNMAPPED::{provider_symbol}"
            groups.setdefault((security_id, str(row.trade_date)[:7].replace("-", "")), []).append(ref)
    for key in list(groups):
        groups[key] = sorted({ref.payload_path: ref for ref in groups[key]}.values(), key=lambda item: str(item.payload_path))

    database_fd, database_name = tempfile.mkstemp(prefix="qdp_v3_5m_coverage_", suffix=".sqlite", dir=str(paths.jobs))
    os.close(database_fd)
    findings: list[QualityFinding] = []
    quarantine_count = 0
    quarantine_sample: list[dict[str, Any]] = []
    strict_row_count = 0
    provisional_row_count = 0
    connection: sqlite3.Connection | None = None
    try:
        connection = sqlite3.connect(database_name)
        connection.execute("PRAGMA journal_mode=OFF")
        connection.execute("PRAGMA synchronous=OFF")
        connection.execute("CREATE TABLE expected (security_id TEXT NOT NULL, trade_date TEXT NOT NULL, PRIMARY KEY (security_id, trade_date)) WITHOUT ROWID")
        connection.execute("CREATE TABLE actual (security_id TEXT NOT NULL, trade_date TEXT NOT NULL, PRIMARY KEY (security_id, trade_date)) WITHOUT ROWID")
        for _, status_path in status_parts:
            expected = pd.read_parquet(status_path, engine="pyarrow")
            expected["trade_date"] = expected["trade_date"].astype(str).str.slice(0, 10)
            expected = expected.loc[
                expected["trade_date"].between(max(str(start_date), "2020-01-01"), str(end_date))
                & expected["tradestatus"].astype(str).eq("1")
                & expected["symbol_on_date"].map(board_for_symbol).eq("MainBoard"),
                ["security_id", "trade_date"],
            ].drop_duplicates()
            connection.executemany("INSERT OR IGNORE INTO expected VALUES (?, ?)", expected.itertuples(index=False, name=None))
        connection.commit()

        for (security_id, month), group_refs in sorted(groups.items()):
            canonical, quarantined = _read_selected_intraday_group(
                group_refs,
                registry=registry,
                start_date=start_date,
                end_date=end_date,
            )
            if not quarantined.empty:
                quarantine_count += int(len(quarantined))
                if len(quarantine_sample) < 20:
                    quarantine_sample.extend(quarantined.head(20 - len(quarantine_sample)).to_dict("records"))
            if canonical.empty:
                continue
            valid_indices: list[int] = []
            actual_keys: list[tuple[str, str]] = []
            for (_, trade_date), day in canonical.groupby(["security_id", "trade_date"], sort=False):
                if not is_complete_5m_day(day):
                    quarantine_count += int(len(day))
                    if len(quarantine_sample) < 20:
                        quarantine_sample.append(
                            {
                                "security_id": security_id,
                                "trade_date": str(trade_date),
                                "conflict_type": "selected_intraday_day_not_complete",
                                "bar_count": int(len(day)),
                            }
                        )
                    continue
                valid_indices.extend(day.index.tolist())
                if day["quality_tier"].eq(QUALITY_STRICT).all() and str(trade_date) >= "2020-01-01":
                    actual_keys.append((str(day["security_id"].iloc[0]), str(trade_date)))
            valid = canonical.loc[valid_indices] if valid_indices else canonical.iloc[0:0]
            strict_row_count += int(valid["quality_tier"].eq(QUALITY_STRICT).sum())
            provisional_row_count += int(valid["quality_tier"].eq(QUALITY_PROVISIONAL).sum())
            connection.executemany("INSERT OR IGNORE INTO actual VALUES (?, ?)", actual_keys)
        connection.commit()
        expected_count = int(connection.execute("SELECT COUNT(*) FROM expected").fetchone()[0])
        covered_count = int(connection.execute("SELECT COUNT(*) FROM expected INNER JOIN actual USING (security_id, trade_date)").fetchone()[0])
        missing_count = expected_count - covered_count
        missing_sample = [
            {"security_id": row[0], "trade_date": row[1], "reason": "strict_5m_missing_or_quarantined"}
            for row in connection.execute(
                "SELECT expected.security_id, expected.trade_date FROM expected LEFT JOIN actual USING (security_id, trade_date) WHERE actual.security_id IS NULL ORDER BY expected.trade_date, expected.security_id LIMIT 20"
            ).fetchall()
        ]
        coverage_rate = float(covered_count / expected_count) if expected_count else 0.0
        if expected_count == 0:
            findings.append(QualityFinding(code="strict_5m_expected_universe_empty", severity="blocker", message="No tradable PIT main-board stock-days were available for 5m coverage proof.", domain=DOMAIN_MARKET_INTRADAY_5M))
        elif coverage_rate < 0.9995:
            findings.append(
                QualityFinding(
                    code="strict_5m_coverage_below_99_95_percent",
                    severity="blocker",
                    message="Strict 5m coverage is below the 99.95% release floor.",
                    domain=DOMAIN_MARKET_INTRADAY_5M,
                    count=missing_count,
                    sample=missing_sample,
                )
            )
        if quarantine_count:
            findings.append(
                QualityFinding(
                    code="intraday_rows_quarantined",
                    severity="warning",
                    message="Unmapped, conflicting, or incomplete 5m rows were excluded from strict visibility.",
                    domain=DOMAIN_MARKET_INTRADAY_5M,
                    count=quarantine_count,
                    sample=quarantine_sample,
                )
            )
        coverage = {
            "required_start_date": max(str(start_date), "2020-01-01"),
            "end_date": str(end_date),
            "expected_stock_day_count": expected_count,
            "strict_covered_stock_day_count": covered_count,
            "strict_missing_stock_day_count": missing_count,
            "strict_coverage_rate": coverage_rate,
            "strict_row_count": strict_row_count,
            "provisional_row_count": provisional_row_count,
        }
    finally:
        try:
            if connection is not None:
                connection.close()
        except Exception:
            pass
        Path(database_name).unlink(missing_ok=True)

    report = report_for(DOMAIN_MARKET_INTRADAY_5M, findings, metrics=coverage)

    def partition_frames() -> Iterable[tuple[str, str, pd.DataFrame]]:
        yielded = False
        for (security_id, month), group_refs in sorted(groups.items()):
            canonical, _ = _read_selected_intraday_group(group_refs, registry=registry, start_date=start_date, end_date=end_date)
            if canonical.empty:
                continue
            valid_days = [day for _, day in canonical.groupby(["security_id", "trade_date"], sort=False) if is_complete_5m_day(day)]
            valid = pd.concat(valid_days, ignore_index=True) if valid_days else canonical.iloc[0:0]
            if valid.empty:
                continue
            yielded = True
            yield f"{security_id}_{month}", month, valid
        if not yielded:
            empty = canonicalize_selected_5m(pd.DataFrame(), identity_registry=registry)[0]
            yield "empty", "", empty

    manifest = write_partitioned_dataset(
        root=paths.root,
        domain=DOMAIN_MARKET_INTRADAY_5M,
        partition_frames=partition_frames(),
        layer="canonical_tiered",
        frequency="5m",
        primary_key=["security_id", "trade_date", "bar_end"],
        quality_report=report,
        partitioning="natural_month",
        inputs=inputs,
        provider_evidence=_provider_evidence(relevant_refs),
        raw_content_hashes=[ref.content_sha256 for ref in relevant_refs],
        build={"contract": QDP_V3_CONTRACT_VERSION, "source_policy": "complete_mootdx_else_complete_baostock_never_stitch", "default_visibility": QUALITY_STRICT},
        coverage=coverage,
        units={"price": "CNY/share", "volume": "share", "amount": "CNY"},
        quarantine=[{"row_count": quarantine_count, "sample": quarantine_sample}, {"stock_day_count": coverage["strict_missing_stock_day_count"], "sample": missing_sample}],
    )
    return manifest, relevant_refs, {"findings": findings, "coverage": coverage, "quarantine": quarantine_sample, "quarantine_count": quarantine_count, "missing_sample": missing_sample}


def _stage_daily_domains(
    *,
    refs: list[RawPartitionRef],
    registry: SecurityIdentityRegistry,
    staging_root: Path,
) -> dict[str, Any]:
    domains = (DOMAIN_MARKET_DAILY_RAW, DOMAIN_SECURITY_STATUS_DAILY, DOMAIN_VALUATION_DAILY, DOMAIN_ELIGIBLE_SIGNAL_D)
    staged: dict[str, list[tuple[str, Path]]] = {domain: [] for domain in domains}
    reports: dict[str, list[QualityReport]] = {domain: [] for domain in domains}
    quarantine_count = 0
    quarantine_sample: list[dict[str, Any]] = []
    resolved_conflict_count = 0
    resolved_conflict_sample: list[dict[str, Any]] = []
    security_ids: set[str] = set()
    current_year = ""
    buffers: dict[str, list[pd.DataFrame]] = {domain: [] for domain in domains}

    def flush(year: str) -> None:
        nonlocal quarantine_count
        if not year:
            return
        market = pd.concat(buffers[DOMAIN_MARKET_DAILY_RAW], ignore_index=True)
        status = pd.concat(buffers[DOMAIN_SECURITY_STATUS_DAILY], ignore_index=True)
        valuation = pd.concat(buffers[DOMAIN_VALUATION_DAILY], ignore_index=True)
        eligible = build_eligible_signal_view(status)
        frames = {
            DOMAIN_MARKET_DAILY_RAW: market,
            DOMAIN_SECURITY_STATUS_DAILY: status,
            DOMAIN_VALUATION_DAILY: valuation,
            DOMAIN_ELIGIBLE_SIGNAL_D: eligible,
        }
        reports[DOMAIN_MARKET_DAILY_RAW].append(audit_canonical_daily(market))
        for domain in (DOMAIN_SECURITY_STATUS_DAILY, DOMAIN_VALUATION_DAILY, DOMAIN_ELIGIBLE_SIGNAL_D):
            reports[domain].append(audit_table_contract(frames[domain], domain=domain, primary_key=["security_id", "trade_date"]))
        for domain, frame in frames.items():
            path = staging_root / domain / f"{year}.parquet"
            atomic_write_parquet(path, frame.sort_values(["security_id", "trade_date"], kind="mergesort").reset_index(drop=True))
            staged[domain].append((year, path))
        for domain in domains:
            buffers[domain].clear()

    for ref in refs:
        year = str(ref.partition_value)[:4]
        if current_year and year != current_year:
            flush(current_year)
        current_year = year
        derived = derive_daily_domains(read_raw_partition(ref), query_date=ref.partition_value, identity_registry=registry)
        buffers[DOMAIN_MARKET_DAILY_RAW].append(derived.market_daily_raw)
        buffers[DOMAIN_SECURITY_STATUS_DAILY].append(derived.security_status_daily)
        buffers[DOMAIN_VALUATION_DAILY].append(derived.valuation_daily)
        security_ids.update(derived.market_daily_raw["security_id"].astype(str))
        if not derived.quarantine.empty:
            resolution = derived.quarantine.get(
                "resolution_status", pd.Series("unresolved", index=derived.quarantine.index)
            ).fillna("unresolved").astype(str)
            resolved = derived.quarantine.loc[resolution.eq("resolved")]
            unresolved = derived.quarantine.loc[~resolution.eq("resolved")]
            quarantine_count += int(len(unresolved))
            resolved_conflict_count += int(len(resolved))
            if len(quarantine_sample) < 100:
                quarantine_sample.extend(unresolved.head(100 - len(quarantine_sample)).to_dict("records"))
            if len(resolved_conflict_sample) < 100:
                resolved_conflict_sample.extend(resolved.head(100 - len(resolved_conflict_sample)).to_dict("records"))
    flush(current_year)
    return {
        "staged": staged,
        "reports": reports,
        "quarantine_count": quarantine_count,
        "quarantine_sample": quarantine_sample,
        "resolved_conflict_count": resolved_conflict_count,
        "resolved_conflict_sample": resolved_conflict_sample,
        "security_ids": security_ids,
    }


def _stage_tradable_open(
    *,
    market_parts: list[tuple[str, Path]],
    staging_root: Path,
) -> tuple[list[tuple[str, Path]], list[QualityReport]]:
    staged: list[tuple[str, Path]] = []
    reports: list[QualityReport] = []
    future_first = pd.DataFrame()
    for year, market_path in reversed(market_parts):
        current = pd.read_parquet(market_path, engine="pyarrow")
        combined = pd.concat([current, future_first], ignore_index=True) if not future_first.empty else current
        tradable = build_tradable_open_view(combined)
        tradable = tradable.loc[tradable["trade_date"].astype(str).str.startswith(year)].copy()
        reports.append(audit_table_contract(tradable, domain=DOMAIN_TRADABLE_OPEN_D1, primary_key=["security_id", "trade_date"]))
        path = staging_root / DOMAIN_TRADABLE_OPEN_D1 / f"{year}.parquet"
        atomic_write_parquet(path, tradable.sort_values(["security_id", "trade_date"], kind="mergesort").reset_index(drop=True))
        staged.append((year, path))
        current_first = current.sort_values(["security_id", "trade_date"]).drop_duplicates("security_id", keep="first")
        if future_first.empty:
            future_first = current_first
        else:
            future_first = pd.concat([current_first, future_first], ignore_index=True).sort_values(["security_id", "trade_date"]).drop_duplicates("security_id", keep="first")
    return sorted(staged), reports


def _partition_frames_from_stage(parts: list[tuple[str, Path]]) -> Iterable[tuple[str, str, pd.DataFrame]]:
    for partition, path in sorted(parts):
        yield partition, partition, pd.read_parquet(path, engine="pyarrow")


def build_candidate(
    *,
    workspace_root: str | Path | None = None,
    start_date: str = "2010-01-01",
    end_date: str = "",
    identity_config: str | Path | None = None,
    require_factor_dual_path: bool = True,
) -> CandidateManifestV3:
    paths = ensure_qdp_v3_layout(workspace_root)
    daily_refs = iter_raw_partitions(RAW_DAILY_ASTOCK, workspace_root=workspace_root, start_value=start_date, end_value=end_date)
    if not daily_refs:
        raise RuntimeError("qdp_v3_daily_raw_partitions_missing")
    snapshot_dates = [ref.partition_value for ref in daily_refs]
    effective_end = str(end_date or snapshot_dates[-1])
    security_master, security_master_ref = _latest_security_master(workspace_root)
    all_stock_refs = iter_raw_partitions(
        RAW_ALL_STOCK,
        workspace_root=workspace_root,
        start_value=start_date,
        end_value=effective_end,
    )
    name_observations, all_stock_symbols = _pit_name_observations(all_stock_refs)

    provider_symbols: set[str] = set(all_stock_symbols)
    for ref in daily_refs:
        raw = read_raw_partition(ref)
        if "code" in raw.columns:
            provider_symbols.update(raw["code"].astype(str).tolist())
    factor_refs = iter_raw_partitions(RAW_ADJUST_FACTOR_EVENT, workspace_root=workspace_root, end_value=effective_end)
    factor_raw_frames = [read_raw_partition(ref) for ref in factor_refs]
    for raw in factor_raw_frames:
        if "code" in raw.columns:
            provider_symbols.update(raw["code"].astype(str).tolist())
    symbol_factor_refs = iter_raw_partitions(RAW_ADJUST_FACTOR_SYMBOL_HISTORY, workspace_root=workspace_root)
    provider_symbols.update(ref.partition_value for ref in symbol_factor_refs)
    xdxr_refs = iter_raw_partitions(RAW_CORPORATE_ACTION_XDXR, workspace_root=workspace_root)
    provider_symbols.update(ref.partition_value for ref in xdxr_refs)
    secondary_domains = (
        DOMAIN_FINANCIAL_QUARTERLY,
        DOMAIN_PERFORMANCE_FORECAST,
        DOMAIN_PERFORMANCE_EXPRESS,
        DOMAIN_INDUSTRY,
        DOMAIN_INDEX_CONSTITUENTS,
    )
    secondary_refs = {
        domain: refs_for_secondary_domain(domain, workspace_root=workspace_root)
        for domain in secondary_domains
    }
    for domain, refs in secondary_refs.items():
        for ref in refs:
            if ref.partition_field == "provider_symbol":
                provider_symbols.add(ref.partition_value)
            else:
                evidence = read_raw_partition(ref)
                if "symbol" in evidence.columns:
                    provider_symbols.update(evidence["symbol"].astype(str).tolist())
    registry = SecurityIdentityRegistry.from_sources(
        provider_symbols=provider_symbols,
        security_master=security_master,
        config_path=identity_config,
        workspace_root=workspace_root,
    )
    registry = registry.with_pit_name_observations(name_observations)
    identity_frame, history_frame = registry.identity_frames()
    xdxr_actions, share_capital_events, xdxr_conflicts = canonicalize_mootdx_xdxr(
        [(ref.partition_value, read_raw_partition(ref)) for ref in xdxr_refs],
        identity_registry=registry,
    )
    official_factor_evidence = load_official_factor_evidence(
        identity_registry=registry,
        workspace_root=workspace_root,
    )
    identity_findings: list[QualityFinding] = []
    if security_master_ref is None:
        identity_findings.append(
            QualityFinding(
                code="security_master_provider_evidence_missing",
                severity="warning",
                message="Identity rows inferred from stable provider symbols lack stock-basic proof.",
                domain=DOMAIN_SECURITY_IDENTITY,
                provisional=True,
            )
        )
    history_findings = _raw_quality_findings(all_stock_refs, domain=DOMAIN_SYMBOL_HISTORY)
    missing_name_snapshot_dates = sorted(set(snapshot_dates) - {ref.partition_value for ref in all_stock_refs})
    if missing_name_snapshot_dates:
        history_findings.append(
            QualityFinding(
                code="symbol_history_name_snapshot_dates_missing",
                severity="blocker",
                message="Date-local query_all_stock name evidence is missing for daily snapshot dates.",
                domain=DOMAIN_SYMBOL_HISTORY,
                count=len(missing_name_snapshot_dates),
                sample=missing_name_snapshot_dates[:20],
            )
        )
    regression_security_id = registry.security_id_for_provider_symbol("302132.SZ")
    if not regression_security_id or registry.symbol_for_date(regression_security_id, "2016-01-04") != "300114.SZ" or registry.symbol_for_date(regression_security_id, "2025-02-17") != "302132.SZ":
        identity_findings.append(
            QualityFinding(
                code="security_identity_code_change_regression_failed",
                severity="blocker",
                message="300114/302132 PIT code-history regression failed.",
                domain=DOMAIN_SYMBOL_HISTORY,
            )
        )
    identity_report = _merge_reports(
        DOMAIN_SECURITY_IDENTITY,
        [audit_table_contract(identity_frame, domain=DOMAIN_SECURITY_IDENTITY, primary_key=["security_id"])],
        identity_findings,
    )
    history_report = _merge_reports(
        DOMAIN_SYMBOL_HISTORY,
        [audit_table_contract(history_frame, domain=DOMAIN_SYMBOL_HISTORY, primary_key=["security_id", "effective_from"])],
        [item for item in identity_findings if item.severity == "blocker"] + history_findings,
    )
    identity_manifest = write_dataset(
        root=paths.root,
        domain=DOMAIN_SECURITY_IDENTITY,
        frame=identity_frame,
        layer="canonical",
        frequency="event",
        primary_key=["security_id"],
        quality_report=identity_report,
        raw_content_hashes=[security_master_ref.content_sha256] if security_master_ref else [],
        provider_evidence=_provider_evidence([security_master_ref] if security_master_ref else []),
        build={"contract": QDP_V3_CONTRACT_VERSION, "identity_config": str(identity_config or "default")},
        coverage={"security_count": int(len(identity_frame))},
        date_column="",
        partitioning="single",
    )
    identity_input = dataset_input_ref(paths.root, identity_manifest)
    history_manifest = write_dataset(
        root=paths.root,
        domain=DOMAIN_SYMBOL_HISTORY,
        frame=history_frame,
        layer="canonical_pit",
        frequency="event",
        primary_key=["security_id", "effective_from"],
        quality_report=history_report,
        inputs=[identity_input],
        raw_content_hashes=(
            ([security_master_ref.content_sha256] if security_master_ref else [])
            + [ref.content_sha256 for ref in all_stock_refs]
        ),
        provider_evidence=_provider_evidence(
            ([security_master_ref] if security_master_ref else []) + all_stock_refs
        ),
        build={
            "contract": QDP_V3_CONTRACT_VERSION,
            "official_code_changes_only": True,
            "pit_name_source": "baostock.query_all_stock(date_snapshot)",
            "current_stock_basic_name_backfill_forbidden": True,
        },
        coverage={
            "history_row_count": int(len(history_frame)),
            "name_observation_count": int(len(name_observations)),
            "name_snapshot_count": len(all_stock_refs),
            "missing_name_snapshot_count": len(missing_name_snapshot_dates),
        },
        date_column="effective_from",
        partitioning="single",
    )
    history_input = dataset_input_ref(paths.root, history_manifest)

    # TemporaryDirectory cleans up on ordinary exceptions as well as normal
    # completion.  A stale directory from a hard process kill remains clearly
    # marked and contains no canonical truth; raw and datasets stay immutable.
    staging_directory = tempfile.TemporaryDirectory(prefix=".candidate_stage_", dir=str(paths.jobs))
    staging_root = Path(staging_directory.name)
    staged_daily = _stage_daily_domains(refs=daily_refs, registry=registry, staging_root=staging_root)
    market_parts = staged_daily["staged"][DOMAIN_MARKET_DAILY_RAW]
    status_parts = staged_daily["staged"][DOMAIN_SECURITY_STATUS_DAILY]
    valuation_parts = staged_daily["staged"][DOMAIN_VALUATION_DAILY]
    eligible_parts = staged_daily["staged"][DOMAIN_ELIGIBLE_SIGNAL_D]
    tradable_parts, tradable_partition_reports = _stage_tradable_open(market_parts=market_parts, staging_root=staging_root)
    daily_quarantine_count = int(staged_daily["quarantine_count"])
    daily_quarantine_sample = list(staged_daily["quarantine_sample"])
    daily_resolved_conflict_count = int(staged_daily["resolved_conflict_count"])
    daily_resolved_conflict_sample = list(staged_daily["resolved_conflict_sample"])
    raw_daily_findings = _raw_quality_findings(daily_refs, domain=DOMAIN_MARKET_DAILY_RAW)
    if daily_quarantine_count:
        raw_daily_findings.append(
            QualityFinding(
                code="daily_identity_conflicts_quarantined",
                severity="blocker",
                message="Mapped duplicate histories disagree and were removed from canonical output.",
                domain=DOMAIN_MARKET_DAILY_RAW,
                count=daily_quarantine_count,
                sample=daily_quarantine_sample[:20],
            )
        )
    if daily_resolved_conflict_count:
        raw_daily_findings.append(
            QualityFinding(
                code="provider_current_symbol_duplicates_resolved",
                severity="warning",
                message="Conflicting provider-current duplicate rows were excluded in favor of the official PIT symbol_on_date row.",
                domain=DOMAIN_MARKET_DAILY_RAW,
                count=daily_resolved_conflict_count,
                sample=daily_resolved_conflict_sample[:20],
            )
        )
    market_report = _merge_reports(DOMAIN_MARKET_DAILY_RAW, staged_daily["reports"][DOMAIN_MARKET_DAILY_RAW], raw_daily_findings)
    status_report = _merge_reports(
        DOMAIN_SECURITY_STATUS_DAILY,
        staged_daily["reports"][DOMAIN_SECURITY_STATUS_DAILY],
        [item for item in raw_daily_findings if item.severity == "blocker"],
    )
    valuation_report = _merge_reports(
        DOMAIN_VALUATION_DAILY,
        staged_daily["reports"][DOMAIN_VALUATION_DAILY],
        [item for item in raw_daily_findings if item.severity == "blocker"],
    )
    eligible_report = _merge_reports(
        DOMAIN_ELIGIBLE_SIGNAL_D,
        staged_daily["reports"][DOMAIN_ELIGIBLE_SIGNAL_D],
        [item for item in raw_daily_findings if item.severity == "blocker"],
    )
    tradable_report = _merge_reports(
        DOMAIN_TRADABLE_OPEN_D1,
        tradable_partition_reports,
        [item for item in raw_daily_findings if item.severity == "blocker"],
    )
    identity_inputs = [identity_input, history_input]
    daily_evidence = _provider_evidence(daily_refs)
    common_build = {"contract": QDP_V3_CONTRACT_VERSION, "start_date": start_date, "end_date": effective_end, "raw_universe": "all_a"}
    market_manifest = write_partitioned_dataset(
        root=paths.root,
        domain=DOMAIN_MARKET_DAILY_RAW,
        partition_frames=_partition_frames_from_stage(market_parts),
        layer="canonical_raw_fact",
        frequency="1d",
        primary_key=["security_id", "trade_date"],
        quality_report=market_report,
        inputs=identity_inputs,
        provider_evidence=daily_evidence,
        raw_content_hashes=[ref.content_sha256 for ref in daily_refs],
        build=common_build,
        coverage={"start_date": snapshot_dates[0], "end_date": snapshot_dates[-1], "security_count": len(set(staged_daily["security_ids"]) - {""})},
        units={"price": "CNY/share", "volume": "share", "amount": "CNY", "pct_chg": "percent"},
        quarantine=(
            ([{"resolution_status": "unresolved", "row_count": daily_quarantine_count, "sample": daily_quarantine_sample}] if daily_quarantine_count else [])
            + ([{"resolution_status": "resolved", "row_count": daily_resolved_conflict_count, "sample": daily_resolved_conflict_sample}] if daily_resolved_conflict_count else [])
        ),
        partitioning="natural_year",
    )
    market_input = dataset_input_ref(paths.root, market_manifest)
    status_manifest = write_partitioned_dataset(
        root=paths.root,
        domain=DOMAIN_SECURITY_STATUS_DAILY,
        partition_frames=_partition_frames_from_stage(status_parts),
        layer="canonical_pit",
        frequency="1d",
        primary_key=["security_id", "trade_date"],
        quality_report=status_report,
        inputs=identity_inputs + [market_input],
        provider_evidence=daily_evidence,
        raw_content_hashes=[ref.content_sha256 for ref in daily_refs],
        build=common_build,
        coverage={"start_date": snapshot_dates[0], "end_date": snapshot_dates[-1]},
        partitioning="natural_year",
    )
    valuation_manifest = write_partitioned_dataset(
        root=paths.root,
        domain=DOMAIN_VALUATION_DAILY,
        partition_frames=_partition_frames_from_stage(valuation_parts),
        layer="canonical_pit",
        frequency="1d",
        primary_key=["security_id", "trade_date"],
        quality_report=valuation_report,
        inputs=identity_inputs + [market_input],
        provider_evidence=daily_evidence,
        raw_content_hashes=[ref.content_sha256 for ref in daily_refs],
        build=common_build,
        coverage={"start_date": snapshot_dates[0], "end_date": snapshot_dates[-1]},
        units={"turnover_rate": "percent"},
        partitioning="natural_year",
    )
    status_input = dataset_input_ref(paths.root, status_manifest)
    eligible_manifest = write_partitioned_dataset(
        root=paths.root,
        domain=DOMAIN_ELIGIBLE_SIGNAL_D,
        partition_frames=_partition_frames_from_stage(eligible_parts),
        layer="canonical_pit_view",
        frequency="1d",
        primary_key=["security_id", "trade_date"],
        quality_report=eligible_report,
        inputs=[history_input, status_input],
        provider_evidence=daily_evidence,
        raw_content_hashes=[ref.content_sha256 for ref in daily_refs],
        build={**common_build, "availability": "signal_date_close", "future_fields_forbidden": True},
        coverage={"start_date": snapshot_dates[0], "end_date": snapshot_dates[-1]},
        partitioning="natural_year",
    )
    tradable_manifest = write_partitioned_dataset(
        root=paths.root,
        domain=DOMAIN_TRADABLE_OPEN_D1,
        partition_frames=_partition_frames_from_stage(tradable_parts),
        layer="research_label_substrate",
        frequency="1d",
        primary_key=["security_id", "trade_date"],
        quality_report=tradable_report,
        inputs=[history_input, market_input, status_input],
        provider_evidence=daily_evidence,
        raw_content_hashes=[ref.content_sha256 for ref in daily_refs],
        build={**common_build, "availability": "label_only_after_next_open", "feature_visibility": "forbidden"},
        coverage={"signal_start_date": snapshot_dates[0], "signal_end_date": snapshot_dates[-1]},
        units={"open_d1": "CNY/share"},
        partitioning="natural_year",
    )

    calendar, calendar_refs, calendar_findings = _calendar_frame(workspace_root, snapshot_dates)
    calendar_report = _merge_reports(
        DOMAIN_TRADING_CALENDAR,
        [audit_table_contract(calendar, domain=DOMAIN_TRADING_CALENDAR, primary_key=["trade_date", "exchange"])],
        calendar_findings,
    )
    calendar_manifest = write_dataset(
        root=paths.root,
        domain=DOMAIN_TRADING_CALENDAR,
        frame=calendar,
        layer="canonical",
        frequency="1d",
        primary_key=["trade_date", "exchange"],
        quality_report=calendar_report,
        provider_evidence=_provider_evidence(calendar_refs),
        raw_content_hashes=[ref.content_sha256 for ref in calendar_refs],
        build={"contract": QDP_V3_CONTRACT_VERSION},
        coverage={"start_date": str(calendar["trade_date"].min()), "end_date": str(calendar["trade_date"].max())},
        partitioning="year",
    )
    calendar_input = dataset_input_ref(paths.root, calendar_manifest)

    secondary_manifests: list[Any] = []
    secondary_conflicts: dict[str, pd.DataFrame] = {}
    for secondary_domain in secondary_domains:
        refs = secondary_refs[secondary_domain]
        if not refs:
            continue
        secondary_frame, conflicts = canonicalize_secondary_domain(
            domain=secondary_domain,
            refs=refs,
            identity_registry=registry,
            calendar=calendar,
        )
        secondary_conflicts[secondary_domain] = conflicts
        if secondary_domain in {DOMAIN_FINANCIAL_QUARTERLY, DOMAIN_PERFORMANCE_FORECAST, DOMAIN_PERFORMANCE_EXPRESS}:
            primary_key = ["security_id", "report_date", "publish_date", "source"]
            date_column = "report_date"
            frequency = "report_event"
            unavailable = secondary_frame.loc[secondary_frame["availability_status"].ne("next_trading_day_after_publish_date")]
        elif secondary_domain == DOMAIN_INDEX_CONSTITUENTS:
            primary_key = ["trade_date", "index_symbol", "security_id", "source"]
            date_column = "trade_date"
            frequency = "snapshot_event"
            unavailable = secondary_frame.iloc[0:0]
        else:
            primary_key = ["trade_date", "security_id", "source"]
            date_column = "trade_date"
            frequency = "snapshot_event"
            unavailable = secondary_frame.iloc[0:0]
        findings = _raw_quality_findings(refs, domain=secondary_domain)
        findings.append(
            QualityFinding(
                code="secondary_domain_single_source_provisional",
                severity="warning",
                message="Secondary PIT data remains explicitly provisional until independent or official reconciliation.",
                domain=secondary_domain,
                provisional=True,
            )
        )
        if not unavailable.empty:
            findings.append(
                QualityFinding(
                    code="report_publish_availability_unproven",
                    severity="warning",
                    message="Rows without an explicit provider publication date remain unavailable to strict PIT queries.",
                    domain=secondary_domain,
                    count=int(len(unavailable)),
                    sample=unavailable[[column for column in ("security_id", "report_date", "publish_date", "availability_status") if column in unavailable.columns]].head(20).to_dict("records"),
                    provisional=True,
                )
            )
        if not conflicts.empty:
            findings.append(
                QualityFinding(
                    code="secondary_identity_conflicts_quarantined",
                    severity="blocker",
                    message="Secondary rows could not be mapped to a stable PIT security identity.",
                    domain=secondary_domain,
                    count=int(len(conflicts)),
                    sample=conflicts.head(20).to_dict("records"),
                )
            )
        report = _merge_reports(
            secondary_domain,
            [audit_table_contract(secondary_frame, domain=secondary_domain, primary_key=primary_key)],
            findings,
        )
        manifest = write_dataset(
            root=paths.root,
            domain=secondary_domain,
            frame=secondary_frame,
            layer="canonical_pit_event",
            frequency=frequency,
            primary_key=primary_key,
            quality_report=report,
            inputs=[*identity_inputs, calendar_input],
            provider_evidence=_provider_evidence(refs),
            raw_content_hashes=[ref.content_sha256 for ref in refs],
            build={
                "contract": QDP_V3_CONTRACT_VERSION,
                "unknown_values": "preserved",
                "future_backfill": False,
                "date_only_publication_policy": "next_exchange_trading_day",
            },
            coverage={
                "row_count": int(len(secondary_frame)),
                "strict_available_row_count": int(len(secondary_frame) - len(unavailable)),
                "unavailable_row_count": int(len(unavailable)),
            },
            quarantine=conflicts.head(100).to_dict("records") if not conflicts.empty else [],
            date_column=date_column,
            partitioning="year",
        )
        secondary_manifests.append(manifest)

    corporate_action_manifest = None
    share_capital_event_manifest = None
    share_capital_daily_manifest = None
    corporate_action_input = None
    reference_proof = pd.DataFrame()
    if xdxr_refs:
        proof_frames: list[pd.DataFrame] = []
        previous_last = pd.DataFrame()
        for year, market_path in market_parts:
            market_year = pd.read_parquet(market_path, engine="pyarrow")
            proof_market = pd.concat([previous_last, market_year], ignore_index=True) if not previous_last.empty else market_year
            year_actions = xdxr_actions.loc[xdxr_actions["event_date"].astype(str).str.startswith(year)]
            if not year_actions.empty:
                proof_frames.append(reconstruct_xdxr_reference_prices(proof_market, year_actions))
            previous_last = market_year.sort_values(["security_id", "trade_date"]).drop_duplicates("security_id", keep="last")
        reference_proof = pd.concat(proof_frames, ignore_index=True) if proof_frames else pd.DataFrame()
        action_findings = _raw_quality_findings(xdxr_refs, domain=DOMAIN_CORPORATE_ACTIONS)
        action_findings.append(
            QualityFinding(
                code="xdxr_requires_official_disclosure_for_strict_use",
                severity="warning",
                message="TDX xdxr is independent corroboration, not official factor authority.",
                domain=DOMAIN_CORPORATE_ACTIONS,
                provisional=True,
            )
        )
        if not xdxr_conflicts.empty:
            action_findings.append(
                QualityFinding(
                    code="xdxr_identity_or_value_conflicts",
                    severity="blocker",
                    message="TDX xdxr rows conflict after stable-identity mapping.",
                    domain=DOMAIN_CORPORATE_ACTIONS,
                    count=int(len(xdxr_conflicts)),
                    sample=xdxr_conflicts.head(20).to_dict("records"),
                )
            )
        if not reference_proof.empty:
            mismatched = reference_proof.loc[reference_proof["proof_status"].eq("reference_price_mismatch")]
            unproven = reference_proof.loc[~reference_proof["proof_status"].isin(["proved", "reference_price_mismatch"])]
            if not mismatched.empty:
                action_findings.append(
                    QualityFinding(
                        code="xdxr_reference_price_mismatch",
                        severity="warning",
                        message="Reconstructed ex-right reference price differs by more than one tick; factor use remains blocked unless separately arbitrated.",
                        domain=DOMAIN_CORPORATE_ACTIONS,
                        count=int(len(mismatched)),
                        sample=mismatched.head(20).to_dict("records"),
                        provisional=True,
                    )
                )
            if not unproven.empty:
                action_findings.append(
                    QualityFinding(
                        code="xdxr_reference_price_inputs_unproven",
                        severity="warning",
                        message="Some xdxr events lack market or action terms needed for reference-price reconstruction.",
                        domain=DOMAIN_CORPORATE_ACTIONS,
                        count=int(len(unproven)),
                        sample=unproven.head(20).to_dict("records"),
                        provisional=True,
                    )
                )
        action_report = _merge_reports(
            DOMAIN_CORPORATE_ACTIONS,
            [audit_table_contract(xdxr_actions, domain=DOMAIN_CORPORATE_ACTIONS, primary_key=["security_id", "event_date", "category"])],
            action_findings,
        )
        corporate_action_manifest = write_dataset(
            root=paths.root,
            domain=DOMAIN_CORPORATE_ACTIONS,
            frame=xdxr_actions,
            layer="canonical_evidence",
            frequency="event",
            primary_key=["security_id", "event_date", "category"],
            quality_report=action_report,
            inputs=identity_inputs,
            provider_evidence=_provider_evidence(xdxr_refs),
            raw_content_hashes=[ref.content_sha256 for ref in xdxr_refs],
            build={"contract": QDP_V3_CONTRACT_VERSION, "authority": "corroborating_only", "raw_share_unit": "10k_shares"},
            coverage={"event_count": int(len(xdxr_actions)), "reference_proof_count": int(len(reference_proof))},
            quarantine=xdxr_conflicts.head(100).to_dict("records") if not xdxr_conflicts.empty else [],
            date_column="event_date",
            partitioning="year",
        )
        corporate_action_input = dataset_input_ref(paths.root, corporate_action_manifest)

        capital_event_report = _merge_reports(
            DOMAIN_SHARE_CAPITAL_EVENT,
            [audit_table_contract(share_capital_events, domain=DOMAIN_SHARE_CAPITAL_EVENT, primary_key=["security_id", "event_date", "category"])],
            [
                QualityFinding(
                    code="share_capital_xdxr_provisional",
                    severity="warning",
                    message="TDX share-capital events remain provisional until official disclosure reconciliation.",
                    domain=DOMAIN_SHARE_CAPITAL_EVENT,
                    provisional=True,
                )
            ],
        )
        share_capital_event_manifest = write_dataset(
            root=paths.root,
            domain=DOMAIN_SHARE_CAPITAL_EVENT,
            frame=share_capital_events,
            layer="canonical_event",
            frequency="event",
            primary_key=["security_id", "event_date", "category"],
            quality_report=capital_event_report,
            inputs=identity_inputs,
            provider_evidence=_provider_evidence(xdxr_refs),
            raw_content_hashes=[ref.content_sha256 for ref in xdxr_refs],
            build={"contract": QDP_V3_CONTRACT_VERSION, "unit_conversion": "TDX_10k_shares_to_shares"},
            coverage={"event_count": int(len(share_capital_events))},
            date_column="event_date",
            partitioning="year",
        )
        capital_event_input = dataset_input_ref(paths.root, share_capital_event_manifest)
        capital_daily_parts: list[tuple[str, Path]] = []
        capital_daily_reports: list[QualityReport] = []
        capital_baseline_unproven = 0
        for year, market_path in market_parts:
            capital_daily = build_share_capital_daily(pd.read_parquet(market_path, engine="pyarrow"), share_capital_events)
            capital_daily_reports.append(audit_table_contract(capital_daily, domain=DOMAIN_SHARE_CAPITAL_DAILY, primary_key=["security_id", "trade_date"]))
            capital_baseline_unproven += int(capital_daily["baseline_status"].eq("baseline_unproven").sum())
            capital_path = staging_root / DOMAIN_SHARE_CAPITAL_DAILY / f"{year}.parquet"
            atomic_write_parquet(capital_path, capital_daily.sort_values(["security_id", "trade_date"], kind="mergesort").reset_index(drop=True))
            capital_daily_parts.append((year, capital_path))
        capital_daily_report = _merge_reports(
            DOMAIN_SHARE_CAPITAL_DAILY,
            capital_daily_reports,
            [
                QualityFinding(
                    code="share_capital_daily_provisional",
                    severity="warning",
                    message="PIT share capital is forward-filled only from provisional xdxr events; no future backfill is used.",
                    domain=DOMAIN_SHARE_CAPITAL_DAILY,
                    count=capital_baseline_unproven,
                    provisional=True,
                )
            ],
        )
        share_capital_daily_manifest = write_partitioned_dataset(
            root=paths.root,
            domain=DOMAIN_SHARE_CAPITAL_DAILY,
            partition_frames=_partition_frames_from_stage(capital_daily_parts),
            layer="canonical_pit",
            frequency="1d",
            primary_key=["security_id", "trade_date"],
            quality_report=capital_daily_report,
            inputs=[market_input, history_input, capital_event_input],
            provider_evidence=_provider_evidence(xdxr_refs),
            raw_content_hashes=[ref.content_sha256 for ref in xdxr_refs],
            build={"contract": QDP_V3_CONTRACT_VERSION, "forward_fill_events_only": True, "future_backfill": False},
            coverage={"start_date": snapshot_dates[0], "end_date": snapshot_dates[-1], "baseline_unproven_rows": capital_baseline_unproven},
            units={"total_share": "share", "float_share": "share", "restricted_share": "share"},
            partitioning="natural_year",
        )

    event_frames: list[pd.DataFrame] = []
    factor_identity_conflicts: list[pd.DataFrame] = []
    for ref, raw in zip(factor_refs, factor_raw_frames):
        events_for_date, conflicts = derive_adjust_factor_events(raw, query_date=ref.partition_value, identity_registry=registry)
        event_frames.append(events_for_date)
        if not conflicts.empty:
            factor_identity_conflicts.append(conflicts)
    event_columns = [
        "security_id",
        "divid_operate_date",
        "symbol_on_date",
        "provider_symbol",
        "fore_adjust_factor",
        "back_adjust_factor",
        "adjust_factor",
        "query_date",
        "source_method",
        "verification_status",
        "identity_mapping_status",
        "source",
    ]
    nonempty_event_frames = [frame for frame in event_frames if not frame.empty]
    batch_events = pd.concat(nonempty_event_frames, ignore_index=True) if nonempty_event_frames else pd.DataFrame(columns=event_columns)
    symbol_event_frames: list[pd.DataFrame] = []
    for ref in symbol_factor_refs:
        symbol_events_for_security, conflicts = derive_symbol_adjust_factor_events(
            read_raw_partition(ref),
            provider_symbol=ref.partition_value,
            identity_registry=registry,
        )
        symbol_event_frames.append(symbol_events_for_security)
        if not conflicts.empty:
            factor_identity_conflicts.append(conflicts)
    nonempty_symbol_event_frames = [frame for frame in symbol_event_frames if not frame.empty]
    symbol_events = pd.concat(nonempty_symbol_event_frames, ignore_index=True) if nonempty_symbol_event_frames else pd.DataFrame(columns=event_columns)
    comparable_start = min((ref.partition_value for ref in factor_refs), default=str(start_date))
    comparable_end = max((ref.partition_value for ref in factor_refs), default=effective_end)
    events, disputed_events, dual_path_metrics = reconcile_adjust_factor_events(
        batch_events,
        symbol_events,
        comparable_start=comparable_start,
        comparable_end=comparable_end,
    )
    arbitrated_events, remaining_disputes, arbitration_metrics = arbitrate_adjust_factor_disputes(
        disputed_events,
        xdxr_events=xdxr_actions,
        official_evidence=official_factor_evidence,
    )
    for column in FACTOR_CANONICAL_COLUMNS:
        if column not in events.columns:
            events[column] = False if column == "arbitration_xdxr_confirmed" else ""
    events = events.loc[:, FACTOR_CANONICAL_COLUMNS]
    if not arbitrated_events.empty:
        events = pd.concat([events, arbitrated_events], ignore_index=True).sort_values(
            ["divid_operate_date", "security_id"], kind="mergesort"
        ).reset_index(drop=True)
    dual_path_metrics.update(
        {
            **arbitration_metrics,
            "pre_arbitration_disputed_event_count": int(len(disputed_events)),
            "remaining_disputed_event_count": int(len(remaining_disputes)),
        }
    )
    if not require_factor_dual_path and events.empty and not batch_events.empty:
        events = batch_events.copy()
        events["verification_status"] = "batch_only_unverified"
        for column in FACTOR_CANONICAL_COLUMNS:
            if column not in events.columns:
                events[column] = False if column == "arbitration_xdxr_confirmed" else ""
        events = events.loc[:, FACTOR_CANONICAL_COLUMNS]
        dual_path_metrics["unverified_batch_events_admitted_for_nonrelease_build"] = int(len(events))
    factor_identity_quarantine = (
        pd.concat(factor_identity_conflicts, ignore_index=True, sort=False)
        if factor_identity_conflicts
        else pd.DataFrame()
    )
    factor_resolution = (
        factor_identity_quarantine.get(
            "resolution_status", pd.Series("unresolved", index=factor_identity_quarantine.index)
        )
        .fillna("unresolved")
        .astype(str)
        if not factor_identity_quarantine.empty
        else pd.Series(dtype=str)
    )
    factor_resolved = (
        factor_identity_quarantine.loc[factor_resolution.eq("resolved")]
        if not factor_identity_quarantine.empty
        else pd.DataFrame()
    )
    factor_unresolved = (
        factor_identity_quarantine.loc[~factor_resolution.eq("resolved")]
        if not factor_identity_quarantine.empty
        else pd.DataFrame()
    )
    factor_quarantine_parts = [frame for frame in (factor_identity_quarantine, remaining_disputes) if not frame.empty]
    factor_quarantine = (
        pd.concat(factor_quarantine_parts, ignore_index=True, sort=False)
        if factor_quarantine_parts
        else pd.DataFrame()
    )
    factor_extra: list[QualityFinding] = _raw_quality_findings(factor_refs, domain=DOMAIN_ADJUST_FACTOR_EVENT)
    if not factor_refs:
        factor_extra.append(
            QualityFinding(
                code="factor_batch_event_history_missing",
                severity="warning",
                message="No batch factor-event partitions are available.",
                domain=DOMAIN_ADJUST_FACTOR_EVENT,
                provisional=True,
            )
        )
    daily_security_ids = {
        registry.security_id_for_provider_symbol(symbol)
        for symbol in provider_symbols
        if registry.security_id_for_provider_symbol(symbol)
    }
    observed_factor_security_ids = {
        registry.security_id_for_provider_symbol(ref.partition_value)
        for ref in symbol_factor_refs
        if registry.security_id_for_provider_symbol(ref.partition_value)
    }
    missing_factor_security_ids = sorted(daily_security_ids - observed_factor_security_ids)
    missing_factor_batch_dates = sorted(set(snapshot_dates) - {ref.partition_value for ref in factor_refs})
    dual_path_metrics.update(
        {
            "expected_security_count": len(daily_security_ids),
            "queried_security_count": len(observed_factor_security_ids),
            "missing_security_count": len(missing_factor_security_ids),
            "missing_batch_trade_date_count": len(missing_factor_batch_dates),
            "proof_complete": bool(symbol_factor_refs and not missing_factor_security_ids and not missing_factor_batch_dates and remaining_disputes.empty),
        }
    )
    if require_factor_dual_path and not symbol_factor_refs:
        factor_extra.append(
            QualityFinding(
                code="factor_dual_path_proof_missing",
                severity="blocker",
                message="Initial rebuild requires batch events versus per-symbol query_adjust_factor proof.",
                domain=DOMAIN_ADJUST_FACTOR_EVENT,
            )
        )
    if require_factor_dual_path and missing_factor_security_ids:
        factor_extra.append(
            QualityFinding(
                code="factor_symbol_history_security_coverage_incomplete",
                severity="blocker",
                message="The legacy factor endpoint has not been queried once for every security identity.",
                domain=DOMAIN_ADJUST_FACTOR_EVENT,
                count=len(missing_factor_security_ids),
                sample=missing_factor_security_ids[:20],
            )
        )
    if require_factor_dual_path and missing_factor_batch_dates:
        factor_extra.append(
            QualityFinding(
                code="factor_batch_trade_dates_missing",
                severity="blocker",
                message="Factor batch event partitions are missing for one or more daily snapshot dates.",
                domain=DOMAIN_ADJUST_FACTOR_EVENT,
                count=len(missing_factor_batch_dates),
                sample=missing_factor_batch_dates[:20],
            )
        )
    if require_factor_dual_path and not remaining_disputes.empty:
        factor_extra.append(
            QualityFinding(
                code="factor_dual_path_events_disputed",
                severity="blocker",
                message="Events present in only one path or with mismatched values require xdxr/official arbitration.",
                domain=DOMAIN_ADJUST_FACTOR_EVENT,
                count=int(len(remaining_disputes)),
                sample=remaining_disputes.head(20).to_dict("records"),
            )
        )
    if not require_factor_dual_path:
        factor_extra.append(
            QualityFinding(
                code="factor_dual_path_requirement_disabled",
                severity="warning",
                message="Unverified factor events are visible only in a non-release candidate.",
                domain=DOMAIN_ADJUST_FACTOR_EVENT,
                provisional=True,
            )
        )
    if not factor_unresolved.empty:
        factor_extra.append(
            QualityFinding(
                code="factor_identity_conflicts_quarantined",
                severity="blocker",
                message="Factor events disagree after stable-identity mapping.",
                domain=DOMAIN_ADJUST_FACTOR_EVENT,
                count=int(len(factor_unresolved)),
            )
        )
    if not factor_resolved.empty:
        factor_extra.append(
            QualityFinding(
                code="factor_provider_symbol_duplicates_resolved",
                severity="warning",
                message="Provider-current factor duplicates were excluded in favor of the official PIT symbol_on_date row.",
                domain=DOMAIN_ADJUST_FACTOR_EVENT,
                count=int(len(factor_resolved)),
                sample=factor_resolved.head(20).to_dict("records"),
            )
        )
    market_event_mask = events["divid_operate_date"].astype(str).between(snapshot_dates[0], snapshot_dates[-1]) if not events.empty else pd.Series(dtype=bool)
    events_requiring_reference = events.loc[market_event_mask, ["security_id", "divid_operate_date"]].drop_duplicates() if not events.empty else pd.DataFrame(columns=["security_id", "divid_operate_date"])
    if not events_requiring_reference.empty:
        proof = reference_proof.rename(columns={"event_date": "divid_operate_date"}) if not reference_proof.empty else pd.DataFrame(columns=["security_id", "divid_operate_date", "proof_status"])
        checked = events_requiring_reference.merge(
            proof[["security_id", "divid_operate_date", "proof_status", "absolute_error"]] if not proof.empty else proof,
            on=["security_id", "divid_operate_date"],
            how="left",
        )
        missing_reference = checked.loc[checked["proof_status"].isna()]
        bad_reference = checked.loc[checked["proof_status"].notna() & checked["proof_status"].ne("proved")]
        if not missing_reference.empty:
            factor_extra.append(
                QualityFinding(
                    code="factor_reference_price_proof_missing",
                    severity="blocker",
                    message="Validated factor events in the market window require an xdxr/official ex-right reference-price proof.",
                    domain=DOMAIN_ADJUST_FACTOR_EVENT,
                    count=int(len(missing_reference)),
                    sample=missing_reference.head(20).to_dict("records"),
                )
            )
        if not bad_reference.empty:
            factor_extra.append(
                QualityFinding(
                    code="factor_reference_price_reconstruction_failed",
                    severity="blocker",
                    message="Ex-right reference-price reconstruction exceeds one tick or lacks required inputs.",
                    domain=DOMAIN_ADJUST_FACTOR_EVENT,
                    count=int(len(bad_reference)),
                    sample=bad_reference.head(20).to_dict("records"),
                )
            )
    event_contract_report = audit_table_contract(events, domain=DOMAIN_ADJUST_FACTOR_EVENT, primary_key=["security_id", "divid_operate_date"]) if not events.empty else report_for(DOMAIN_ADJUST_FACTOR_EVENT, [], metrics={"row_count": 0, "zero_event_history": True})
    factor_daily_parts: list[tuple[str, Path]] = []
    factor_partition_reports: list[QualityReport] = []
    baseline_unproven_rows = 0
    for year, market_path in market_parts:
        factor_daily = build_adjust_factor_daily(pd.read_parquet(market_path, engine="pyarrow"), events)
        factor_partition_reports.append(audit_factor_semantics(events, factor_daily))
        baseline_unproven_rows += int(factor_daily["baseline_status"].eq("baseline_unproven").sum())
        factor_path = staging_root / DOMAIN_ADJUST_FACTOR_DAILY / f"{year}.parquet"
        atomic_write_parquet(factor_path, factor_daily.sort_values(["security_id", "trade_date"], kind="mergesort").reset_index(drop=True))
        factor_daily_parts.append((year, factor_path))
    event_report = _merge_reports(DOMAIN_ADJUST_FACTOR_EVENT, [event_contract_report], factor_extra)
    factor_daily_report = _merge_reports(
        DOMAIN_ADJUST_FACTOR_DAILY,
        factor_partition_reports,
        [item for item in factor_extra if item.severity == "blocker"],
    )
    event_manifest = write_dataset(
        root=paths.root,
        domain=DOMAIN_ADJUST_FACTOR_EVENT,
        frame=events,
        layer="canonical_event",
        frequency="event",
        primary_key=["security_id", "divid_operate_date"],
        quality_report=event_report,
        inputs=[*identity_inputs, *([corporate_action_input] if corporate_action_input is not None else [])],
        provider_evidence=_provider_evidence([*factor_refs, *symbol_factor_refs, *xdxr_refs]),
        raw_content_hashes=[ref.content_sha256 for ref in [*factor_refs, *symbol_factor_refs, *xdxr_refs]],
        build={"contract": QDP_V3_CONTRACT_VERSION, "source_method": "date_batch+symbol_history+official_arbitration", "dual_path_required": require_factor_dual_path, "dual_path_proof": dual_path_metrics, "official_evidence_count": int(len(official_factor_evidence))},
        coverage={"event_count": int(len(events)), "start_date": str(events["divid_operate_date"].min()) if not events.empty else "", "end_date": str(events["divid_operate_date"].max()) if not events.empty else "", **dual_path_metrics},
        quarantine=factor_quarantine.head(100).to_dict("records") if not factor_quarantine.empty else [],
        date_column="divid_operate_date",
        partitioning="year",
    )
    event_input = dataset_input_ref(paths.root, event_manifest)
    factor_daily_manifest = write_partitioned_dataset(
        root=paths.root,
        domain=DOMAIN_ADJUST_FACTOR_DAILY,
        partition_frames=_partition_frames_from_stage(factor_daily_parts),
        layer="canonical_derived",
        frequency="1d",
        primary_key=["security_id", "trade_date"],
        quality_report=factor_daily_report,
        inputs=[market_input, event_input, history_input],
        provider_evidence=_provider_evidence([*factor_refs, *symbol_factor_refs]),
        raw_content_hashes=[ref.content_sha256 for ref in [*factor_refs, *symbol_factor_refs]],
        build={"contract": QDP_V3_CONTRACT_VERSION, "baseline_policy": "no_unproven_1.0", "forward_fill_events_only": True},
        coverage={"start_date": snapshot_dates[0], "end_date": snapshot_dates[-1], "baseline_unproven_rows": baseline_unproven_rows},
        partitioning="natural_year",
    )

    intraday_manifest, intraday_refs, intraday_info = _build_intraday_dataset(
        paths=paths,
        workspace_root=workspace_root,
        registry=registry,
        status_parts=status_parts,
        inputs=[identity_input, history_input, market_input, status_input],
        start_date=max(str(start_date), "2011-11-22"),
        end_date=effective_end,
    )

    manifests = [
        calendar_manifest,
        identity_manifest,
        history_manifest,
        market_manifest,
        status_manifest,
        valuation_manifest,
        event_manifest,
        factor_daily_manifest,
        eligible_manifest,
        tradable_manifest,
    ]
    for optional_manifest in (corporate_action_manifest, share_capital_event_manifest, share_capital_daily_manifest):
        if optional_manifest is not None:
            manifests.append(optional_manifest)
    manifests.extend(secondary_manifests)
    if intraday_manifest is not None:
        manifests.append(intraday_manifest)
    dataset_ids = {manifest.domain: manifest.dataset_id for manifest in manifests}
    dataset_shas: dict[str, str] = {}
    for manifest in manifests:
        path = dataset_manifest_for_id(paths.root, manifest.dataset_id, manifest.domain)
        if path is None:
            raise RuntimeError(f"built_dataset_manifest_missing:{manifest.dataset_id}")
        dataset_shas[manifest.domain] = manifest_sha256(path)
    quality_tiers = {manifest.domain: manifest.quality_tier for manifest in manifests}
    blockers: list[dict[str, Any]] = []
    for manifest in manifests:
        blockers.extend({**item, "dataset_id": manifest.dataset_id} for item in manifest.blockers)
    open_calendar = calendar.loc[calendar["is_open"].astype(str).str.lower().isin({"1", "true", "t", "yes"}), "trade_date"].astype(str)
    expected_open_dates = set(open_calendar.loc[(open_calendar >= str(start_date)) & (open_calendar <= effective_end)])
    missing_snapshot_dates = sorted(expected_open_dates - set(snapshot_dates))
    if missing_snapshot_dates:
        blockers.append(
            {
                "code": "daily_snapshot_trading_dates_missing",
                "count": len(missing_snapshot_dates),
                "sample": missing_snapshot_dates[:20],
            }
        )
    first_snapshot = min(snapshot_dates)
    required_first_open = min(expected_open_dates) if expected_open_dates else str(start_date)
    required_last_open = max(expected_open_dates) if expected_open_dates else effective_end
    if first_snapshot > required_first_open:
        blockers.append({"code": "daily_coverage_starts_after_required_open_date", "required": required_first_open, "actual": first_snapshot})
    if max(snapshot_dates) < required_last_open:
        blockers.append({"code": "daily_coverage_ends_before_required_open_date", "required": required_last_open, "actual": max(snapshot_dates)})
    if intraday_manifest is None:
        blockers.extend(item.to_dict() for item in intraday_info.get("findings", []) if item.severity == "blocker")
    secondary_raw_refs = [ref for refs in secondary_refs.values() for ref in refs]
    raw_refs = [
        *daily_refs,
        *all_stock_refs,
        *calendar_refs,
        *factor_refs,
        *symbol_factor_refs,
        *xdxr_refs,
        *secondary_raw_refs,
        *intraday_refs,
    ]
    if security_master_ref:
        raw_refs.append(security_master_ref)
    quarantine_summary: list[dict[str, Any]] = []
    if daily_quarantine_count:
        quarantine_summary.append({"domain": DOMAIN_MARKET_DAILY_RAW, "resolution_status": "unresolved", "row_count": daily_quarantine_count, "sample": daily_quarantine_sample[:20]})
    if daily_resolved_conflict_count:
        quarantine_summary.append({"domain": DOMAIN_MARKET_DAILY_RAW, "resolution_status": "resolved", "row_count": daily_resolved_conflict_count, "sample": daily_resolved_conflict_sample[:20]})
    if not factor_unresolved.empty:
        quarantine_summary.append({"domain": DOMAIN_ADJUST_FACTOR_EVENT, "resolution_status": "unresolved", "row_count": int(len(factor_unresolved)), "sample": factor_unresolved.head(20).to_dict("records")})
    if not factor_resolved.empty:
        quarantine_summary.append({"domain": DOMAIN_ADJUST_FACTOR_EVENT, "resolution_status": "resolved", "row_count": int(len(factor_resolved)), "sample": factor_resolved.head(20).to_dict("records")})
    if not xdxr_conflicts.empty:
        quarantine_summary.append({"domain": DOMAIN_CORPORATE_ACTIONS, "row_count": int(len(xdxr_conflicts)), "sample": xdxr_conflicts.head(20).to_dict("records")})
    for domain, conflicts in secondary_conflicts.items():
        if not conflicts.empty:
            quarantine_summary.append({"domain": domain, "row_count": int(len(conflicts)), "sample": conflicts.head(20).to_dict("records")})
    if intraday_info.get("quarantine_count") or intraday_info.get("coverage", {}).get("strict_missing_stock_day_count"):
        quarantine_summary.append(
            {
                "domain": DOMAIN_MARKET_INTRADAY_5M,
                "row_count": int(intraday_info.get("quarantine_count", 0) or 0),
                "missing_stock_day_count": int(intraday_info.get("coverage", {}).get("strict_missing_stock_day_count", 0) or 0),
                "sample": list(intraday_info.get("quarantine", []) or [])[:10] + list(intraday_info.get("missing_sample", []) or [])[:10],
            }
        )
    build_contract = {
        "contract": QDP_V3_CONTRACT_VERSION,
        "start_date": str(start_date),
        "end_date": effective_end,
        "raw_scope": "all_a",
        "strict_research_view": "shanghai_shenzhen_mainboard",
        "factor_dual_path_required": bool(require_factor_dual_path),
    }
    candidate = write_candidate(
        datasets=dataset_ids,
        dataset_manifest_sha256=dataset_shas,
        quality_tiers=quality_tiers,
        blockers=blockers,
        coverage={
            "start_date": first_snapshot,
            "end_date": max(snapshot_dates),
            "snapshot_count": len(snapshot_dates),
            "expected_open_date_count": len(expected_open_dates),
            "missing_snapshot_date_count": len(missing_snapshot_dates),
            "raw_security_count": len(set(staged_daily["security_ids"]) - {""}),
            "research_view": "mainboard_hs_pit",
            "intraday_5m": dict(intraday_info.get("coverage", {}) or {}),
        },
        build=build_contract,
        raw_partitions=[ref.to_dict(root=paths.root) for ref in raw_refs],
        quarantine=quarantine_summary,
        workspace_root=workspace_root,
    )
    staging_directory.cleanup()
    return candidate
