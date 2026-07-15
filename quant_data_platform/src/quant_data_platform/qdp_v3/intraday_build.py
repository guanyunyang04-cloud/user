from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Iterable, Mapping

import pandas as pd

from quant_data_platform.core.json_io import read_json
from quant_data_platform.qdp_v3.constants import (
    DOMAIN_MARKET_INTRADAY_5M,
    QDP_V3_CONTRACT_VERSION,
    QUALITY_QUARANTINED,
    QUALITY_STRICT,
    RAW_EXTERNAL_QUANT_INTRADAY_5M,
    RAW_INTRADAY_5M_SELECTED,
    RAW_TUSHARE_PROXY_INTRADAY_5M,
)
from quant_data_platform.qdp_v3.datasets import write_partitioned_dataset
from quant_data_platform.qdp_v3.identity import board_for_symbol, normalize_symbol
from quant_data_platform.qdp_v3.intraday import (
    CANONICAL_5M_COLUMNS,
    canonicalize_selected_5m,
    is_complete_5m_day,
    normalize_provider_5m,
    normalize_tushare_proxy_5m,
    reconcile_5m_with_daily,
    stable_security_bucket,
)
from quant_data_platform.qdp_v3.manifest import ProviderEvidence
from quant_data_platform.qdp_v3.quality import QualityFinding, audit_strict_5m_coverage, report_for
from quant_data_platform.qdp_v3.storage import (
    RawPartitionRef,
    atomic_write_parquet,
    get_raw_partition_version,
    iter_raw_partitions,
    read_raw_partition,
    read_raw_receipt,
)


def _selected_upstream_lineage(
    selected_refs: Iterable[RawPartitionRef],
    *,
    workspace_root: str | Path | None,
) -> tuple[list[RawPartitionRef], set[str], list[str]]:
    """Resolve selected-source raw inputs and retain every referenced hash.

    New receipts carry the full raw partition identity so older immutable
    versions remain addressable after an incremental revision. Legacy receipts
    are resolved against the current raw catalog when possible; their hashes
    are still returned even when provider metadata cannot be recovered.
    """

    resolved: list[RawPartitionRef] = []
    hashes: set[str] = set()
    unresolved: list[str] = []
    latest_by_domain_hash: dict[tuple[str, str], RawPartitionRef] = {}
    seen_refs: set[tuple[str, str, str, str]] = set()
    for selected_ref in selected_refs:
        receipt = read_raw_receipt(selected_ref)
        for item in list(receipt.get("inputs", ()) or ()):
            if not isinstance(item, Mapping):
                continue
            raw_domain = str(item.get("raw_domain", "") or "")
            content_sha = str(item.get("content_sha256", "") or "")
            if not raw_domain or not content_sha:
                continue
            hashes.add(content_sha)
            partition_field = str(item.get("partition_field", "") or "")
            partition_value = str(item.get("partition_value", "") or "")
            upstream = None
            if partition_field and partition_value:
                upstream = get_raw_partition_version(
                    raw_domain,
                    partition_field=partition_field,
                    partition_value=partition_value,
                    content_sha256=content_sha,
                    workspace_root=workspace_root,
                )
            if upstream is None:
                key = (raw_domain, content_sha)
                if key not in latest_by_domain_hash:
                    for candidate in iter_raw_partitions(raw_domain, workspace_root=workspace_root):
                        latest_by_domain_hash.setdefault(
                            (candidate.raw_domain, candidate.content_sha256), candidate
                        )
                upstream = latest_by_domain_hash.get(key)
            if upstream is None:
                unresolved.append(f"{raw_domain}:{content_sha}")
                continue
            identity = (
                upstream.raw_domain,
                upstream.partition_field,
                upstream.partition_value,
                upstream.content_sha256,
            )
            if identity not in seen_refs:
                seen_refs.add(identity)
                resolved.append(upstream)
    return resolved, hashes, sorted(set(unresolved))


def _provider_evidence(refs: Iterable[RawPartitionRef]) -> list[ProviderEvidence]:
    aggregates: dict[tuple[str, str, str], dict[str, Any]] = {}
    for ref in refs:
        receipt = read_raw_receipt(ref)
        provider = str(receipt.get("provider", "unknown") or "unknown")
        default_endpoint = "stk_mins" if provider == "tushare_proxy" else "local_archive"
        endpoint = str(receipt.get("endpoint", receipt.get("api_name", default_endpoint)) or default_endpoint)
        fingerprint = str(receipt.get("token_sha256", "") or "")
        key = (provider, endpoint, fingerprint)
        request = receipt.get("request") if isinstance(receipt.get("request"), Mapping) else {}
        start = str(
            receipt.get("coverage_start_date", "")
            or receipt.get("requested_start_date", "")
            or receipt.get("start_at", "")
            or request.get("start_date", "")
            or ""
        )
        end = str(
            receipt.get("coverage_end_date", "")
            or receipt.get("requested_end_date", "")
            or receipt.get("end_at", "")
            or request.get("end_date", "")
            or ""
        )
        aggregate = aggregates.setdefault(
            key,
            {
                "provider": provider,
                "endpoint": endpoint,
                "package_versions": set(),
                "starts": [],
                "ends": [],
                "collected": [],
                "protocols": set(),
                "upstream": set(),
                "roles": set(),
                "source_urls": set(),
                "source_inventory_fingerprints": set(),
                "container_sha256": set(),
            },
        )
        if start:
            aggregate["starts"].append(start)
        if end:
            aggregate["ends"].append(end)
        package_version = str(receipt.get("package_version", "") or "")
        if package_version:
            aggregate["package_versions"].add(package_version)
        collected_at = str(receipt.get("stored_at", "") or "")
        if collected_at:
            aggregate["collected"].append(collected_at)
        aggregate["protocols"].add(
            str(
                receipt.get(
                    "protocol",
                    "tushare_compatible_http"
                    if provider == "tushare_proxy"
                    else "local_archive_files"
                    if provider == "external_quant_archive"
                    else "tdx_quote_protocol",
                )
                or ""
            )
        )
        aggregate["upstream"].add(
            str(
                receipt.get(
                    "upstream_provenance",
                    "not_exposed" if provider == "tushare_proxy" else provider,
                )
                or ""
            )
        )
        aggregate["roles"].add(
            str(
                receipt.get(
                    "production_role",
                    "historical_bootstrap"
                    if provider in {"tushare_proxy", "external_quant_archive"}
                    else "ongoing_incremental",
                )
                or ""
            )
        )
        source_url = str(receipt.get("source_reference_url", "") or "")
        if source_url:
            aggregate["source_urls"].add(source_url)
        source_inventory = str(receipt.get("source_inventory_fingerprint", "") or "")
        if source_inventory:
            aggregate["source_inventory_fingerprints"].add(source_inventory)
        for segment in list(receipt.get("source_segments", ()) or ()):
            if isinstance(segment, Mapping):
                container_sha = str(segment.get("container_sha256", "") or "")
                if container_sha:
                    aggregate["container_sha256"].add(container_sha)

    evidence: list[ProviderEvidence] = []
    for (provider, endpoint, fingerprint), aggregate in sorted(aggregates.items()):
        versions = sorted(aggregate["package_versions"])
        evidence.append(
            ProviderEvidence(
                provider=provider,
                endpoint=endpoint,
                package_version=versions[0] if len(versions) == 1 else ",".join(versions),
                request_range={
                    "start_at": min(aggregate["starts"]) if aggregate["starts"] else "",
                    "end_at": max(aggregate["ends"]) if aggregate["ends"] else "",
                },
                collected_at=max(aggregate["collected"]) if aggregate["collected"] else "",
                metadata={
                    "protocols": sorted(filter(None, aggregate["protocols"])),
                    "upstream_provenance": sorted(filter(None, aggregate["upstream"])),
                    "production_roles": sorted(filter(None, aggregate["roles"])),
                    "token_sha256": fingerprint,
                    "source_reference_urls": sorted(aggregate["source_urls"]),
                    "source_inventory_fingerprints": sorted(
                        aggregate["source_inventory_fingerprints"]
                    ),
                    "container_sha256": sorted(aggregate["container_sha256"]),
                },
            )
        )
    return evidence


def _raw_groups(
    refs: Iterable[RawPartitionRef],
    *,
    registry: Any,
) -> tuple[dict[str, list[tuple[str, RawPartitionRef]]], list[str]]:
    groups: dict[str, list[tuple[str, RawPartitionRef]]] = {}
    unmapped: list[str] = []
    for ref in refs:
        receipt = read_raw_receipt(ref)
        symbol = normalize_symbol(receipt.get("provider_symbol", ref.partition_value))
        if not symbol:
            unmapped.append(str(ref.partition_value))
            continue
        security_id = registry.security_id_for_provider_symbol(symbol)
        if not security_id:
            unmapped.append(symbol)
            continue
        groups.setdefault(str(security_id), []).append((symbol, ref))
    return groups, sorted(set(unmapped))


def _create_reference_database(
    *,
    database_path: Path,
    market_parts: list[tuple[str, Path]],
    status_parts: list[tuple[str, Path]],
    start_date: str,
    end_date: str,
) -> Any:
    import duckdb

    connection = duckdb.connect(str(database_path))
    connection.execute(
        "CREATE TABLE daily (security_id VARCHAR, trade_date VARCHAR, open DOUBLE, high DOUBLE, low DOUBLE, close DOUBLE, volume DOUBLE, amount DOUBLE)"
    )
    connection.execute(
        "CREATE TABLE expected (security_id VARCHAR, trade_date VARCHAR, PRIMARY KEY (security_id, trade_date))"
    )
    connection.execute(
        "CREATE TABLE actual_5m (security_id VARCHAR, trade_date VARCHAR, PRIMARY KEY (security_id, trade_date))"
    )
    connection.execute(
        "CREATE TABLE explained_5m (security_id VARCHAR, trade_date VARCHAR, PRIMARY KEY (security_id, trade_date))"
    )
    for _, path in sorted(market_parts):
        frame = pd.read_parquet(path, engine="pyarrow")
        if frame.empty:
            continue
        frame["trade_date"] = frame["trade_date"].astype(str).str.slice(0, 10)
        frame = frame.loc[
            frame["trade_date"].between(str(start_date), str(end_date)),
            ["security_id", "trade_date", "open", "high", "low", "close", "volume", "amount"],
        ]
        connection.register("market_chunk", frame)
        connection.execute("INSERT INTO daily SELECT * FROM market_chunk")
        connection.unregister("market_chunk")
    for _, path in sorted(status_parts):
        frame = pd.read_parquet(path, engine="pyarrow")
        if frame.empty:
            continue
        frame["trade_date"] = frame["trade_date"].astype(str).str.slice(0, 10)
        expected = frame.loc[
            frame["trade_date"].between(str(start_date), str(end_date))
            & frame["tradestatus"].astype(str).eq("1")
            & frame["symbol_on_date"].map(board_for_symbol).eq("MainBoard"),
            ["security_id", "trade_date"],
        ].drop_duplicates()
        connection.register("expected_chunk", expected)
        connection.execute("INSERT OR IGNORE INTO expected SELECT * FROM expected_chunk")
        connection.unregister("expected_chunk")
    connection.execute("CREATE INDEX daily_security_date ON daily(security_id, trade_date)")
    return connection


def _daily_reference_for_security(connection: Any, security_id: str) -> dict[str, dict[str, Any]]:
    frame = connection.execute(
        "SELECT trade_date, open, high, low, close, volume, amount FROM daily WHERE security_id = ? ORDER BY trade_date",
        [str(security_id)],
    ).fetchdf()
    return {
        str(row.trade_date): {
            "open": row.open,
            "high": row.high,
            "low": row.low,
            "close": row.close,
            "volume": row.volume,
            "amount": row.amount,
        }
        for row in frame.itertuples(index=False)
    }


def _load_source_5m(
    group_refs: list[tuple[str, RawPartitionRef]],
    *,
    start_date: str,
    end_date: str,
    source: str,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for symbol, ref in group_refs:
        raw = read_raw_partition(ref)
        if source == "tushare_proxy":
            normalized = normalize_tushare_proxy_5m(raw, provider_symbol=symbol)
        else:
            normalized = normalize_provider_5m(raw, provider_symbol=symbol, source=source)
        if normalized.empty:
            continue
        frames.append(normalized.loc[normalized["trade_date"].between(str(start_date), str(end_date))])
    return pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()


def _load_proxy_5m(
    group_refs: list[tuple[str, RawPartitionRef]],
    *,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """Backward-compatible wrapper retained for focused callers/tests."""

    return _load_source_5m(
        group_refs,
        start_date=start_date,
        end_date=end_date,
        source="tushare_proxy",
    )


def _choose_provider_day(
    frame: pd.DataFrame,
    *,
    security_id: str,
    trade_date: str,
    registry: Any,
) -> tuple[pd.DataFrame, str]:
    if frame.empty:
        return frame, "missing"
    expected_symbol = normalize_symbol(registry.symbol_for_date(security_id, trade_date))
    groups = [(str(symbol), group.copy()) for symbol, group in frame.groupby("provider_symbol", sort=True)]
    if len(groups) == 1:
        return groups[0][1], "provider_current_symbol_only"
    compare_columns = ["bar_end", "open", "high", "low", "close", "volume", "amount"]
    fingerprints = {
        tuple(group.sort_values("bar_end")[compare_columns].astype("string").fillna("<NULL>").to_numpy().ravel())
        for _, group in groups
    }
    if len(fingerprints) != 1:
        return frame.iloc[0:0], "provider_code_restatement_value_conflict"
    for symbol, group in groups:
        if symbol == expected_symbol:
            return group, "identical_provider_code_restatement_pit_symbol_preferred"
    return groups[0][1], "identical_provider_code_restatement_deduplicated"


def _validate_source_day(
    frame: pd.DataFrame,
    *,
    source: str,
    security_id: str,
    trade_date: str,
    registry: Any,
    daily: dict[str, Any] | None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Validate one provider's whole stock-day without borrowing any bars."""

    chosen, identity_reason = _choose_provider_day(
        frame,
        security_id=security_id,
        trade_date=trade_date,
        registry=registry,
    )
    if chosen.empty:
        return chosen, {
            "source": source,
            "valid": False,
            "reason": "missing" if frame.empty else identity_reason,
        }
    if not is_complete_5m_day(chosen):
        return chosen.iloc[0:0], {
            "source": source,
            "valid": False,
            "reason": f"{source}_5m_structure_invalid",
            "row_count": int(len(chosen)),
        }
    daily_result = reconcile_5m_with_daily(chosen, daily)
    if daily_result.get("conflict"):
        return chosen.iloc[0:0], {
            "source": source,
            "valid": False,
            "reason": f"{source}_5m_daily_conflict",
            "evidence": daily_result,
        }
    if not daily_result.get("comparable"):
        return chosen.iloc[0:0], {
            "source": source,
            "valid": False,
            "reason": f"{source}_complete_daily_proof_missing",
            "evidence": daily_result,
        }
    return chosen, {
        "source": source,
        "valid": True,
        "reason": identity_reason,
        "evidence": daily_result,
    }


def _select_historical_source_day(
    *,
    local_day: pd.DataFrame,
    proxy_day: pd.DataFrame,
    free_day: pd.DataFrame | None = None,
    security_id: str,
    trade_date: str,
    registry: Any,
    daily: dict[str, Any] | None,
) -> tuple[pd.DataFrame, str, dict[str, Any]]:
    """Select exactly one complete source for a historical stock-day.

    The local direct-5m archive is authoritative when its *whole* day passes
    structure and same-source daily reconciliation.  A complete selected
    mootdx/BaoStock day is the next residual source; Tushare is consulted last.
    Rows from different providers are never concatenated or bar-spliced.
    """

    local, local_result = _validate_source_day(
        local_day,
        source="external_quant_archive",
        security_id=security_id,
        trade_date=trade_date,
        registry=registry,
        daily=daily,
    )
    if bool(local_result.get("valid")):
        selected = local.copy()
        selected["quality_tier"] = QUALITY_STRICT
        reason = "external_quant_archive_complete_and_daily_consistent"
        selected["source_selection_reason"] = reason
        return selected, reason, {"selected_source": "external_quant_archive", "local": local_result}

    materialized_free = free_day if isinstance(free_day, pd.DataFrame) else pd.DataFrame()
    free, free_result = _validate_source_day(
        materialized_free,
        source="selected_free_5m",
        security_id=security_id,
        trade_date=trade_date,
        registry=registry,
        daily=daily,
    )
    if bool(free_result.get("valid")):
        raw_tiers = set(
            free.get("quality_tier", pd.Series(QUALITY_QUARANTINED, index=free.index))
            .fillna(QUALITY_QUARANTINED)
            .astype(str)
        )
        if raw_tiers == {QUALITY_STRICT}:
            selected = free.copy()
            selected["quality_tier"] = QUALITY_STRICT
            selected_sources = sorted(
                set(selected.get("source", pd.Series(dtype=str)).dropna().astype(str))
            )
            selected_source = selected_sources[0] if len(selected_sources) == 1 else "selected_free_5m"
            reason = f"{selected_source}_complete_free_residual_and_daily_consistent"
            selected["source_selection_reason"] = reason
            return selected, reason, {
                "selected_source": selected_source,
                "local": local_result,
                "free": free_result,
            }
        free_result = {
            **free_result,
            "valid": False,
            "reason": "selected_free_5m_not_strict",
            "quality_tiers": sorted(raw_tiers),
        }

    proxy, proxy_result = _validate_source_day(
        proxy_day,
        source="tushare_proxy",
        security_id=security_id,
        trade_date=trade_date,
        registry=registry,
        daily=daily,
    )
    if bool(proxy_result.get("valid")):
        selected = proxy.copy()
        selected["quality_tier"] = QUALITY_STRICT
        reason = "tushare_proxy_residual_complete_and_daily_consistent"
        selected["source_selection_reason"] = reason
        return selected, reason, {
            "selected_source": "tushare_proxy",
            "local": local_result,
            "free": free_result,
            "proxy": proxy_result,
        }

    return local.iloc[0:0], "historical_5m_all_sources_rejected", {
        "selected_source": "",
        "local": local_result,
        "free": free_result,
        "proxy": proxy_result,
    }


def _quality_coverage(connection: Any) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    by_date = connection.execute(
        "SELECT e.trade_date, COUNT(*) AS expected_count, COUNT(x.security_id) AS explained_count "
        "FROM expected e LEFT JOIN explained_5m x USING (security_id, trade_date) "
        "GROUP BY e.trade_date ORDER BY e.trade_date"
    ).fetchall()
    watermark = ""
    first_incomplete_index = len(by_date)
    for index, (trade_date, expected_count, explained_count) in enumerate(by_date):
        if int(explained_count) != int(expected_count):
            first_incomplete_index = index
            break
        watermark = str(trade_date)
    later_explained_dates = (
        [
            str(trade_date)
            for trade_date, _, explained_count in by_date[first_incomplete_index + 1 :]
            if int(explained_count) > 0
        ]
        if first_incomplete_index < len(by_date)
        else []
    )
    if watermark:
        expected = int(
            connection.execute("SELECT COUNT(*) FROM expected WHERE trade_date <= ?", [watermark]).fetchone()[0]
        )
        covered = int(
            connection.execute(
                "SELECT COUNT(*) FROM expected e INNER JOIN actual_5m a USING (security_id, trade_date) WHERE e.trade_date <= ?",
                [watermark],
            ).fetchone()[0]
        )
        sample_rows = connection.execute(
            "SELECT e.security_id, e.trade_date FROM expected e LEFT JOIN actual_5m a USING (security_id, trade_date) "
            "WHERE e.trade_date <= ? AND a.security_id IS NULL ORDER BY e.trade_date, e.security_id LIMIT 20",
            [watermark],
        ).fetchall()
    else:
        expected = 0
        covered = 0
        sample_rows = []
    missing = max(0, expected - covered)
    trailing_dates = [str(item[0]) for item in by_date if not watermark or str(item[0]) > watermark]
    trailing_expected = (
        int(connection.execute("SELECT COUNT(*) FROM expected WHERE trade_date > ?", [watermark]).fetchone()[0])
        if watermark
        else int(connection.execute("SELECT COUNT(*) FROM expected").fetchone()[0])
    )
    coverage = {
        "expected_stock_day_count": expected,
        "strict_covered_stock_day_count": covered,
        "strict_missing_stock_day_count": missing,
        "strict_coverage_rate": float(covered / expected) if expected else 0.0,
        "watermark": watermark,
        "date_count": len(by_date),
        "trailing_lag_trade_date_count": len(trailing_dates),
        "trailing_lag_expected_stock_day_count": trailing_expected,
        "non_continuous_history_hole": bool(later_explained_dates),
        "later_explained_date_sample": later_explained_dates[:20],
    }
    sample = [
        {
            "security_id": str(row[0]),
            "trade_date": str(row[1]),
            "reason": "strict_5m_missing_or_quarantined",
        }
        for row in sample_rows
    ]
    return coverage, sample


def _selected_incremental_frame(
    refs: Iterable[RawPartitionRef],
    *,
    start_date: str,
    end_date: str,
    bootstrap_cutoff: str,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for ref in refs:
        raw = read_raw_partition(ref)
        if raw.empty:
            continue
        partition_symbol = normalize_symbol(str(ref.partition_value).rsplit("_", 1)[0])
        normalized = normalize_provider_5m(raw, provider_symbol=partition_symbol, source="qdp_selected_5m")
        metadata_columns = [
            column
            for column in (
                "provider_symbol",
                "trade_date",
                "bar_end",
                "quality_tier",
                "source_selection_reason",
                "source",
            )
            if column in raw.columns
        ]
        if {"provider_symbol", "trade_date", "bar_end"}.issubset(metadata_columns):
            metadata = raw.loc[:, metadata_columns].copy()
            if "source" in metadata.columns:
                metadata = metadata.rename(columns={"source": "selected_source"})
            metadata["provider_symbol"] = metadata["provider_symbol"].map(normalize_symbol)
            metadata["trade_date"] = pd.to_datetime(metadata["trade_date"], errors="coerce").dt.strftime("%Y-%m-%d")
            metadata["bar_end"] = metadata["bar_end"].astype(str).str.slice(-5)
            normalized = normalized.merge(metadata, on=["provider_symbol", "trade_date", "bar_end"], how="left")
            if "selected_source" in normalized.columns:
                normalized["source"] = normalized["selected_source"].fillna(normalized["source"]).astype(str)
                normalized = normalized.drop(columns=["selected_source"])
        normalized["quality_tier"] = normalized.get(
            "quality_tier", pd.Series(QUALITY_QUARANTINED, index=normalized.index)
        ).fillna(QUALITY_QUARANTINED).astype(str)
        normalized["source_selection_reason"] = normalized.get(
            "source_selection_reason", pd.Series("selected_complete_5m", index=normalized.index)
        ).fillna("selected_complete_5m").astype(str)
        # The selected mootdx/BaoStock domain is also the free historical-gap
        # source for dates that the local archive does not contain (notably
        # 2020+).  Source precedence is enforced by the caller at stock-day
        # granularity, so filtering it to dates after the nominal Tushare
        # cutoff would leave valid archive-tail gaps permanently invisible.
        normalized = normalized.loc[
            normalized["trade_date"].between(str(start_date), str(end_date))
        ]
        if not normalized.empty:
            frames.append(normalized)
    return pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()


def _frames_from_bucketed_stage(
    staged: Iterable[tuple[str, int, Path]],
) -> Iterable[tuple[str, str, pd.DataFrame]]:
    grouped: dict[tuple[str, int], list[Path]] = {}
    for year, bucket, path in staged:
        grouped.setdefault((str(year), int(bucket)), []).append(Path(path))
    for (year, bucket), paths in sorted(grouped.items()):
        frames = [pd.read_parquet(path, engine="pyarrow") for path in sorted(paths)]
        frame = pd.concat(frames, ignore_index=True).loc[:, CANONICAL_5M_COLUMNS]
        yield f"{year}_b{bucket:02d}", f"{year}/bucket={bucket:02d}", frame


def build_proxy_intraday_dataset(
    *,
    paths: Any,
    workspace_root: str | Path | None,
    registry: Any,
    market_parts: list[tuple[str, Path]],
    status_parts: list[tuple[str, Path]],
    inputs: list[Any],
    start_date: str,
    end_date: str,
    staging_root: Path,
) -> tuple[Any | None, list[RawPartitionRef], dict[str, Any]]:
    local_refs = iter_raw_partitions(RAW_EXTERNAL_QUANT_INTRADAY_5M, workspace_root=workspace_root)
    proxy_refs = iter_raw_partitions(RAW_TUSHARE_PROXY_INTRADAY_5M, workspace_root=workspace_root)
    cutoff_payload = read_json(paths.metadata / "tushare_proxy_bootstrap_cutoff.json")
    bootstrap_cutoff = str(cutoff_payload.get("bootstrap_cutoff", "") or "")[:10]
    selected_refs = iter_raw_partitions(RAW_INTRADAY_5M_SELECTED, workspace_root=workspace_root)
    selected_upstream_refs, selected_upstream_hashes, unresolved_selected_inputs = (
        _selected_upstream_lineage(selected_refs, workspace_root=workspace_root)
    )
    all_source_refs = [*local_refs, *proxy_refs, *selected_refs]
    if not local_refs and not proxy_refs and not selected_refs:
        finding = QualityFinding(
            code="strict_5m_raw_missing",
            severity="blocker",
            message="No completed local-archive or Tushare-residual 5m raw security histories exist.",
            domain=DOMAIN_MARKET_INTRADAY_5M,
        )
        return None, all_source_refs, {
            "findings": [finding],
            "coverage": {"strict_coverage_rate": 0.0, "watermark": ""},
            "quarantine": [],
            "quarantine_count": 0,
            "missing_sample": [],
        }

    local_groups, local_unmapped = _raw_groups(local_refs, registry=registry)
    proxy_groups, proxy_unmapped = _raw_groups(proxy_refs, registry=registry)
    # Proxy histories enter candidate lineage only when they are actually
    # selected or needed to explain a rejected residual. Histories duplicated
    # by either the local archive or a complete free-source day remain outside
    # active lineage and can be retired later.
    proxy_evidence_security_ids: set[str] = set()
    selected = _selected_incremental_frame(
        selected_refs,
        start_date=start_date,
        end_date=end_date,
        bootstrap_cutoff=bootstrap_cutoff,
    )
    selected_unmapped: list[str] = []
    selected_groups: dict[str, list[pd.DataFrame]] = {}
    if not selected.empty:
        for provider_symbol, selected_symbol in selected.groupby("provider_symbol", sort=True):
            security_id = registry.security_id_for_provider_symbol(provider_symbol)
            if not security_id:
                selected_unmapped.append(str(provider_symbol))
                continue
            selected_groups.setdefault(str(security_id), []).append(selected_symbol.copy())
    selected_by_security = {
        security_id: pd.concat(frames, ignore_index=True, sort=False)
        for security_id, frames in selected_groups.items()
    }
    database_path = staging_root / "intraday_reference.duckdb"
    connection = _create_reference_database(
        database_path=database_path,
        market_parts=market_parts,
        status_parts=status_parts,
        start_date=start_date,
        end_date=end_date,
    )
    staged: list[tuple[str, int, Path]] = []
    quarantine_count = 0
    quarantine_sample: list[dict[str, Any]] = []
    strict_stock_days = 0
    historical_selected_source_stock_days: dict[str, int] = {
        "external_quant_archive": 0,
        "tushare_proxy": 0,
    }

    def quarantine(item: dict[str, Any]) -> None:
        nonlocal quarantine_count
        quarantine_count += 1
        if len(quarantine_sample) < 50:
            quarantine_sample.append(item)

    try:
        historical_security_ids = sorted(
            set(local_groups) | set(proxy_groups) | set(selected_by_security)
        )
        for security_id in historical_security_ids:
            local = _load_source_5m(
                local_groups.get(security_id, []),
                start_date=start_date,
                end_date=end_date,
                source="external_quant_archive",
            )
            proxy = _load_source_5m(
                proxy_groups.get(security_id, []),
                start_date=start_date,
                end_date=end_date,
                source="tushare_proxy",
            )
            free = selected_by_security.get(security_id, pd.DataFrame()).copy()
            if local.empty and proxy.empty and free.empty:
                continue
            daily_by_date = _daily_reference_for_security(connection, security_id)
            canonical_years: dict[str, list[pd.DataFrame]] = {}
            explained_keys: list[tuple[str, str]] = []
            strict_keys: list[tuple[str, str]] = []
            local_dates = set(local["trade_date"].astype(str)) if not local.empty else set()
            proxy_dates = set(proxy["trade_date"].astype(str)) if not proxy.empty else set()
            free_dates = set(free["trade_date"].astype(str)) if not free.empty else set()
            for trade_date in sorted(local_dates | proxy_dates | free_dates):
                trade_date = str(trade_date)
                explained_keys.append((security_id, trade_date))
                local_day = local.loc[local["trade_date"].astype(str).eq(trade_date)].copy()
                proxy_day = proxy.loc[proxy["trade_date"].astype(str).eq(trade_date)].copy()
                free_day = free.loc[free["trade_date"].astype(str).eq(trade_date)].copy()
                chosen, selection_reason, source_evidence = _select_historical_source_day(
                    local_day=local_day,
                    proxy_day=proxy_day,
                    free_day=free_day,
                    security_id=security_id,
                    trade_date=trade_date,
                    registry=registry,
                    daily=daily_by_date.get(trade_date),
                )
                if chosen.empty:
                    if "proxy" in source_evidence:
                        proxy_evidence_security_ids.add(security_id)
                    quarantine(
                        {
                            "security_id": security_id,
                            "trade_date": trade_date,
                            "reason": selection_reason,
                            "evidence": source_evidence,
                        }
                    )
                    continue
                canonical, identity_quarantine = canonicalize_selected_5m(chosen, identity_registry=registry)
                if not identity_quarantine.empty or canonical.empty:
                    quarantine(
                        {
                            "security_id": security_id,
                            "trade_date": trade_date,
                            "reason": "identity_unmapped_or_conflicted",
                        }
                    )
                    continue
                canonical_years.setdefault(trade_date[:4], []).append(canonical)
                strict_stock_days += 1
                selected_source = str(source_evidence.get("selected_source", "") or "")
                if selected_source:
                    historical_selected_source_stock_days[selected_source] = (
                        historical_selected_source_stock_days.get(selected_source, 0) + 1
                    )
                if selected_source == "tushare_proxy":
                    proxy_evidence_security_ids.add(security_id)
                strict_keys.append((security_id, trade_date))
            if explained_keys:
                connection.executemany("INSERT OR IGNORE INTO explained_5m VALUES (?, ?)", explained_keys)
            if strict_keys:
                connection.executemany("INSERT OR IGNORE INTO actual_5m VALUES (?, ?)", strict_keys)
            bucket = stable_security_bucket(security_id)
            for year, frames in sorted(canonical_years.items()):
                frame = pd.concat(frames, ignore_index=True).loc[:, CANONICAL_5M_COLUMNS]
                path = staging_root / "historical_intraday_5m" / year / f"bucket_{bucket:02d}" / f"{security_id}.parquet"
                atomic_write_parquet(path, frame)
                staged.append((year, bucket, path))

        coverage, missing_sample = _quality_coverage(connection)
    finally:
        connection.close()
        try:
            os.remove(database_path)
        except OSError:
            pass

    effective_proxy_refs: list[RawPartitionRef] = []
    for security_id in sorted(proxy_evidence_security_ids):
        effective_proxy_refs.extend(ref for _, ref in proxy_groups.get(security_id, []))
    if proxy_unmapped:
        unmapped_symbols = set(proxy_unmapped)
        effective_proxy_refs.extend(
            ref
            for ref in proxy_refs
            if normalize_symbol(ref.partition_value) in unmapped_symbols
        )
    deduped_proxy_refs: list[RawPartitionRef] = []
    seen_proxy_refs: set[tuple[str, str, str, str]] = set()
    for ref in effective_proxy_refs:
        key = (ref.raw_domain, ref.partition_field, ref.partition_value, ref.content_sha256)
        if key not in seen_proxy_refs:
            seen_proxy_refs.add(key)
            deduped_proxy_refs.append(ref)
    all_refs: list[RawPartitionRef] = []
    seen_all_refs: set[tuple[str, str, str, str]] = set()
    for ref in [*local_refs, *deduped_proxy_refs, *selected_refs, *selected_upstream_refs]:
        key = (ref.raw_domain, ref.partition_field, ref.partition_value, ref.content_sha256)
        if key not in seen_all_refs:
            seen_all_refs.add(key)
            all_refs.append(ref)

    coverage = {
        **coverage,
        "required_start_date": str(start_date),
        "end_date": str(end_date),
        "bootstrap_cutoff": bootstrap_cutoff,
        "selected_incremental_partition_count": len(selected_refs),
        "strict_stock_day_count": strict_stock_days,
        "historical_selected_source_stock_days": historical_selected_source_stock_days,
        "external_quant_archive_raw_partition_count": len(local_refs),
        "tushare_proxy_residual_raw_partition_count": len(deduped_proxy_refs),
        "tushare_proxy_duplicate_raw_partition_excluded_count": len(proxy_refs) - len(deduped_proxy_refs),
        "provisional_stock_day_count": 0,
        "canonical_year_bucket_shard_count": len({(year, bucket) for year, bucket, _ in staged}),
    }
    findings: list[QualityFinding] = []
    unmapped = sorted(set(local_unmapped + proxy_unmapped + selected_unmapped))
    if unmapped:
        findings.append(
            QualityFinding(
                code="intraday_5m_identity_unmapped",
                severity="blocker",
                message="One or more 5m raw partitions cannot map to a stable security identity.",
                domain=DOMAIN_MARKET_INTRADAY_5M,
                count=len(unmapped),
                sample=[{"provider_symbol": item} for item in unmapped[:20]],
            )
        )
    if unresolved_selected_inputs:
        findings.append(
            QualityFinding(
                code="intraday_5m_selected_upstream_lineage_unresolved",
                severity="blocker",
                message="Selected mootdx/BaoStock 5m raw input versions cannot be resolved.",
                domain=DOMAIN_MARKET_INTRADAY_5M,
                count=len(unresolved_selected_inputs),
                sample=[{"raw_input": item} for item in unresolved_selected_inputs[:20]],
            )
        )
    coverage["strict_missing_sample"] = missing_sample
    findings.extend(audit_strict_5m_coverage(coverage))
    if bool(coverage.get("non_continuous_history_hole", False)):
        findings.append(
            QualityFinding(
                code="strict_5m_non_continuous_history_hole",
                severity="warning",
                message="5m evidence exists after an earlier unexplained date; incomplete days remain quarantined.",
                domain=DOMAIN_MARKET_INTRADAY_5M,
                sample=list(coverage.get("later_explained_date_sample", []) or []),
            )
        )
    if quarantine_count:
        findings.append(
            QualityFinding(
                code="intraday_5m_stock_days_quarantined",
                severity="warning",
                message="Conflicting or invalid 5m stock-days are excluded from canonical visibility.",
                domain=DOMAIN_MARKET_INTRADAY_5M,
                count=quarantine_count,
                sample=quarantine_sample[:20],
            )
        )
    quality_report = report_for(DOMAIN_MARKET_INTRADAY_5M, findings, metrics=coverage)

    def frames_from_stage() -> Iterable[tuple[str, str, pd.DataFrame]]:
        if not staged:
            yield "empty", "", pd.DataFrame(columns=CANONICAL_5M_COLUMNS)
            return
        yield from _frames_from_bucketed_stage(staged)

    manifest = write_partitioned_dataset(
        root=paths.root,
        domain=DOMAIN_MARKET_INTRADAY_5M,
        partition_frames=frames_from_stage(),
        layer="canonical_tiered",
        frequency="5m",
        primary_key=["security_id", "trade_date", "bar_end"],
        quality_report=quality_report,
        partitioning="natural_year_security_bucket",
        inputs=inputs,
        provider_evidence=_provider_evidence(all_refs),
        raw_content_hashes=sorted(
            {*(ref.content_sha256 for ref in all_refs), *selected_upstream_hashes}
        ),
        build={
            "contract": QDP_V3_CONTRACT_VERSION,
            "canonical_source_through_bootstrap": "complete_external_quant_archive_5m_with_free_2020plus_and_tushare_residuals",
            "historical_source_selection_policy": "whole_stock_day_local_primary_then_complete_residual_never_stitch",
            "historical_network_route": "selected_mootdx_baostock_for_2020plus_before_tushare_residual",
            "historical_participating_sources": [
                source
                for source, count in historical_selected_source_stock_days.items()
                if int(count) > 0
            ],
            "post_bootstrap_source_policy": "complete_mootdx_else_complete_baostock_never_stitch",
            "bar_contract": "right_closed_48_bars_0935_1130_1305_1500",
            "quality_policy": "stock_day_evidence_driven",
            "tushare_proxy_raw_units": {"volume": "share", "amount": "CNY"},
            "tushare_proxy_canonical_scales": {"volume": 1.0, "amount": 1.0},
            "external_quant_archive_canonical_scales": {"volume": 1.0, "amount": 1.0},
            "bootstrap_cutoff": bootstrap_cutoff,
            "default_visibility": QUALITY_STRICT,
        },
        coverage=coverage,
        units={"price": "CNY/share", "volume": "share", "amount": "CNY"},
        quarantine=[
            {"stock_day_count": quarantine_count, "sample": quarantine_sample[:50]},
            {
                "stock_day_count": int(coverage.get("strict_missing_stock_day_count", 0)),
                "sample": missing_sample,
            },
        ],
    )
    return manifest, all_refs, {
        "findings": findings,
        "coverage": coverage,
        "quarantine": quarantine_sample,
        "quarantine_count": quarantine_count,
        "missing_sample": missing_sample,
    }
