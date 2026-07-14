from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from quant_data_platform.core.json_io import read_json
from quant_data_platform.qdp_v3.constants import (
    DOMAIN_MARKET_INTRADAY_1M,
    DOMAIN_MARKET_INTRADAY_5M,
    QDP_V3_CONTRACT_VERSION,
    QUALITY_PROVISIONAL,
    QUALITY_QUARANTINED,
    QUALITY_STRICT,
    RAW_INTRADAY_1M_MOOTDX,
    RAW_INTRADAY_5M_SELECTED,
    RAW_TUSHARE_PROXY_INTRADAY_1M,
    RAW_TUSHARE_PROXY_INTRADAY_5M,
)
from quant_data_platform.qdp_v3.datasets import write_partitioned_dataset
from quant_data_platform.qdp_v3.identity import board_for_symbol, normalize_symbol
from quant_data_platform.qdp_v3.intraday_1m import (
    CANONICAL_ONE_MINUTE_COLUMNS,
    aggregate_1m_to_5m,
    canonicalize_1m_day,
    classify_1m_stock_day,
    normalize_direct_proxy_5m,
    normalize_provider_1m,
    stable_security_bucket,
)
from quant_data_platform.qdp_v3.intraday import is_complete_5m_day, normalize_provider_5m
from quant_data_platform.qdp_v3.manifest import ProviderEvidence
from quant_data_platform.qdp_v3.quality import QualityFinding, report_for
from quant_data_platform.qdp_v3.storage import (
    RawPartitionRef,
    atomic_write_parquet,
    iter_raw_partitions,
    read_raw_partition,
    read_raw_receipt,
)


def _provider_evidence(refs: Iterable[RawPartitionRef]) -> list[ProviderEvidence]:
    evidence: dict[tuple[str, str, str], ProviderEvidence] = {}
    for ref in refs:
        receipt = read_raw_receipt(ref)
        provider = str(receipt.get("provider", "tushare_proxy") or "tushare_proxy")
        endpoint = str(receipt.get("endpoint", receipt.get("api_name", "stk_mins")) or "stk_mins")
        fingerprint = str(receipt.get("token_sha256", "") or "")
        key = (provider, endpoint, fingerprint)
        evidence[key] = ProviderEvidence(
            provider=provider,
            endpoint=endpoint,
            package_version=str(receipt.get("package_version", "") or ""),
            request_range={
                "start_at": str(receipt.get("start_at", "") or ""),
                "end_at": str(receipt.get("end_at", "") or ""),
            },
            collected_at=str(receipt.get("stored_at", "") or ""),
            metadata={
                "protocol": str(receipt.get("protocol", "tushare_compatible_http" if provider == "tushare_proxy" else "tdx_quote_protocol") or ""),
                "upstream_provenance": str(receipt.get("upstream_provenance", "not_exposed" if provider == "tushare_proxy" else provider) or ""),
                "production_role": str(receipt.get("production_role", "historical_bootstrap" if provider == "tushare_proxy" else "ongoing_incremental") or ""),
                "token_sha256": fingerprint,
            },
        )
    return list(evidence.values())


def _symbol_from_partition(ref: RawPartitionRef, frequency: str) -> str:
    suffix = f"__{frequency}"
    value = str(ref.partition_value)
    if not value.endswith(suffix):
        return ""
    return normalize_symbol(value[: -len(suffix)])


def _raw_groups(
    refs: Iterable[RawPartitionRef],
    *,
    frequency: str,
    registry: Any,
) -> tuple[dict[str, list[tuple[str, RawPartitionRef]]], list[str]]:
    groups: dict[str, list[tuple[str, RawPartitionRef]]] = {}
    unmapped: list[str] = []
    for ref in refs:
        symbol = _symbol_from_partition(ref, frequency)
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
    connection.execute("CREATE TABLE expected (security_id VARCHAR, trade_date VARCHAR, PRIMARY KEY (security_id, trade_date))")
    connection.execute("CREATE TABLE actual_1m (security_id VARCHAR, trade_date VARCHAR, PRIMARY KEY (security_id, trade_date))")
    connection.execute("CREATE TABLE explained_1m (security_id VARCHAR, trade_date VARCHAR, PRIMARY KEY (security_id, trade_date))")
    connection.execute("CREATE TABLE actual_5m (security_id VARCHAR, trade_date VARCHAR, PRIMARY KEY (security_id, trade_date))")
    connection.execute("CREATE TABLE explained_5m (security_id VARCHAR, trade_date VARCHAR, PRIMARY KEY (security_id, trade_date))")
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


def _daily_reference_for_date(connection: Any, trade_date: str) -> dict[str, dict[str, Any]]:
    frame = connection.execute(
        "SELECT security_id, open, high, low, close, volume, amount FROM daily WHERE trade_date = ?",
        [str(trade_date)],
    ).fetchdf()
    return {
        str(row.security_id): {
            "open": row.open,
            "high": row.high,
            "low": row.low,
            "close": row.close,
            "volume": row.volume,
            "amount": row.amount,
        }
        for row in frame.itertuples(index=False)
    }


def _load_normalized_1m(
    group_refs: list[tuple[str, RawPartitionRef]],
    *,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for symbol, ref in group_refs:
        raw = read_raw_partition(ref)
        normalized = normalize_provider_1m(
            raw,
            provider_symbol=symbol,
            source="tushare_proxy",
            merge_0930_into_0931=True,
        )
        if normalized.empty:
            continue
        normalized = normalized.loc[normalized["trade_date"].between(str(start_date), str(end_date))]
        frames.append(normalized)
    return pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()


def _load_normalized_5m(
    group_refs: list[tuple[str, RawPartitionRef]],
    *,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for symbol, ref in group_refs:
        raw = read_raw_partition(ref)
        normalized = normalize_direct_proxy_5m(raw, provider_symbol=symbol)
        if normalized.empty:
            continue
        normalized = normalized.loc[normalized["trade_date"].between(str(start_date), str(end_date))]
        frames.append(normalized)
    return pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()


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
    for symbol, group in groups:
        if symbol == expected_symbol:
            return group, "pit_symbol_preferred"
    if len(groups) == 1:
        return groups[0][1], "provider_current_symbol_only"
    compare_columns = ["bar_end", "open", "high", "low", "close", "volume", "amount"]
    fingerprints = {
        tuple(group.sort_values("bar_end")[compare_columns].astype("string").fillna("<NULL>").to_numpy().ravel())
        for _, group in groups
    }
    if len(fingerprints) == 1:
        return groups[0][1], "identical_provider_code_restatement_deduplicated"
    return frame.iloc[0:0], "provider_code_restatement_value_conflict"


def _quality_coverage(
    connection: Any,
    *,
    actual_table: str = "actual_1m",
    explained_table: str = "explained_1m",
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if actual_table not in {"actual_1m", "actual_5m"} or explained_table not in {"explained_1m", "explained_5m"}:
        raise ValueError("intraday_coverage_table_invalid")
    by_date = connection.execute(
        f"SELECT e.trade_date, COUNT(*) AS expected_count, COUNT(x.security_id) AS explained_count FROM expected e LEFT JOIN {explained_table} x USING (security_id, trade_date) GROUP BY e.trade_date ORDER BY e.trade_date"
    ).fetchall()
    watermark = ""
    first_incomplete_index = len(by_date)
    for trade_date, expected_count, explained_count in by_date:
        if int(explained_count) != int(expected_count):
            first_incomplete_index = next(
                index for index, item in enumerate(by_date) if str(item[0]) == str(trade_date)
            )
            break
        watermark = str(trade_date)
    later_explained_dates = [
        str(trade_date)
        for trade_date, _, explained_count in by_date[first_incomplete_index + 1 :]
        if int(explained_count) > 0
    ] if first_incomplete_index < len(by_date) else []
    if watermark:
        expected = int(connection.execute("SELECT COUNT(*) FROM expected WHERE trade_date <= ?", [watermark]).fetchone()[0])
        covered = int(
            connection.execute(
                f"SELECT COUNT(*) FROM expected e INNER JOIN {actual_table} a USING (security_id, trade_date) WHERE e.trade_date <= ?",
                [watermark],
            ).fetchone()[0]
        )
        sample_rows = connection.execute(
            f"SELECT e.security_id, e.trade_date FROM expected e LEFT JOIN {actual_table} a USING (security_id, trade_date) WHERE e.trade_date <= ? AND a.security_id IS NULL ORDER BY e.trade_date, e.security_id LIMIT 20",
            [watermark],
        ).fetchall()
    else:
        expected = 0
        covered = 0
        sample_rows = []
    missing = max(0, expected - covered)
    sample = [
        {"security_id": str(row[0]), "trade_date": str(row[1]), "reason": f"strict_{actual_table.removeprefix('actual_')}_missing_or_quarantined"}
        for row in sample_rows
    ]
    trailing_dates = [str(item[0]) for item in by_date if not watermark or str(item[0]) > watermark]
    trailing_expected = int(
        connection.execute("SELECT COUNT(*) FROM expected WHERE trade_date > ?", [watermark]).fetchone()[0]
    ) if watermark else int(connection.execute("SELECT COUNT(*) FROM expected").fetchone()[0])
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
    return coverage, sample


def build_proxy_intraday_datasets(
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
) -> tuple[Any | None, Any | None, list[RawPartitionRef], dict[str, Any]]:
    one_refs = iter_raw_partitions(RAW_TUSHARE_PROXY_INTRADAY_1M, workspace_root=workspace_root)
    five_refs = iter_raw_partitions(RAW_TUSHARE_PROXY_INTRADAY_5M, workspace_root=workspace_root)
    cutoff_payload = read_json(paths.metadata / "tushare_proxy_bootstrap_cutoff.json")
    bootstrap_cutoff = str(cutoff_payload.get("bootstrap_cutoff", "") or "")[:10]
    mootdx_refs = [
        ref
        for ref in iter_raw_partitions(RAW_INTRADAY_1M_MOOTDX, workspace_root=workspace_root)
        if (not bootstrap_cutoff or ref.partition_value > bootstrap_cutoff)
        and str(start_date) <= ref.partition_value <= str(end_date)
    ]
    selected_refs = [
        ref
        for ref in iter_raw_partitions(RAW_INTRADAY_5M_SELECTED, workspace_root=workspace_root)
        if (not bootstrap_cutoff or str(ref.partition_value).rsplit("_", 1)[-1][:6] >= bootstrap_cutoff.replace("-", "")[:6])
    ]
    all_refs = [*one_refs, *five_refs, *mootdx_refs, *selected_refs]
    if not one_refs:
        finding = QualityFinding(
            code="strict_1m_raw_missing",
            severity="blocker",
            message="No completed Tushare-proxy 1m raw security histories exist.",
            domain=DOMAIN_MARKET_INTRADAY_1M,
        )
        return None, None, all_refs, {
            "findings": [finding],
            "coverage_1m": {"strict_coverage_rate": 0.0, "watermark": ""},
            "coverage_5m": {"strict_coverage_rate": 0.0, "watermark": ""},
            "quarantine": [],
            "quarantine_count": 0,
        }
    one_groups, one_unmapped = _raw_groups(one_refs, frequency="1m", registry=registry)
    five_groups, five_unmapped = _raw_groups(five_refs, frequency="5m", registry=registry)
    selected_frames: list[pd.DataFrame] = []
    for ref in selected_refs:
        raw_selected = read_raw_partition(ref)
        if raw_selected.empty:
            continue
        partition_symbol = normalize_symbol(str(ref.partition_value).rsplit("_", 1)[0])
        normalized_selected = normalize_provider_5m(
            raw_selected,
            provider_symbol=partition_symbol,
            source="qdp_selected_5m",
        )
        if {"provider_symbol", "trade_date", "bar_end", "quality_tier"}.issubset(raw_selected.columns):
            metadata_columns = [
                column
                for column in ("provider_symbol", "trade_date", "bar_end", "quality_tier", "source_selection_reason", "source")
                if column in raw_selected.columns
            ]
            selected_metadata = raw_selected.loc[:, metadata_columns].copy()
            if "source" in selected_metadata.columns:
                selected_metadata = selected_metadata.rename(columns={"source": "selected_source"})
            selected_metadata["provider_symbol"] = selected_metadata["provider_symbol"].map(normalize_symbol)
            selected_metadata["trade_date"] = pd.to_datetime(selected_metadata["trade_date"], errors="coerce").dt.strftime("%Y-%m-%d")
            selected_metadata["bar_end"] = selected_metadata["bar_end"].astype(str).str.slice(-5)
            normalized_selected = normalized_selected.merge(
                selected_metadata,
                on=["provider_symbol", "trade_date", "bar_end"],
                how="left",
            )
            if "selected_source" in normalized_selected.columns:
                normalized_selected["source"] = normalized_selected["selected_source"].fillna("qdp_selected_5m").astype(str)
        normalized_selected = normalized_selected.loc[
            normalized_selected["trade_date"].between(str(start_date), str(end_date))
            & (normalized_selected["trade_date"] > str(bootstrap_cutoff or "0000-00-00"))
        ]
        if not normalized_selected.empty:
            selected_frames.append(normalized_selected)
    selected_5m = pd.concat(selected_frames, ignore_index=True, sort=False) if selected_frames else pd.DataFrame()
    database_path = staging_root / "intraday_reference.duckdb"
    connection = _create_reference_database(
        database_path=database_path,
        market_parts=market_parts,
        status_parts=status_parts,
        start_date=start_date,
        end_date=end_date,
    )
    one_staged: list[tuple[str, str, Path]] = []
    five_staged: list[tuple[str, str, Path]] = []
    quarantine_count = 0
    quarantine_sample: list[dict[str, Any]] = []
    strict_stock_days = 0
    provisional_stock_days = 0
    recent_derived_5m_keys: set[tuple[str, str]] = set()
    try:
        for security_id, refs_for_security in sorted(one_groups.items()):
            one = _load_normalized_1m(refs_for_security, start_date=start_date, end_date=end_date)
            if one.empty:
                continue
            explained_keys = [(security_id, str(item)) for item in sorted(set(one["trade_date"].astype(str)))]
            if explained_keys:
                connection.executemany("INSERT OR IGNORE INTO explained_1m VALUES (?, ?)", explained_keys)
                connection.executemany("INSERT OR IGNORE INTO explained_5m VALUES (?, ?)", explained_keys)
            direct = _load_normalized_5m(
                five_groups.get(security_id, []),
                start_date=start_date,
                end_date=end_date,
            )
            daily_by_date = _daily_reference_for_security(connection, security_id)
            canonical_years: dict[str, list[pd.DataFrame]] = {}
            five_years: dict[str, list[pd.DataFrame]] = {}
            strict_keys: list[tuple[str, str]] = []
            for trade_date, raw_day in one.groupby("trade_date", sort=True):
                chosen, selection_reason = _choose_provider_day(
                    raw_day,
                    security_id=security_id,
                    trade_date=str(trade_date),
                    registry=registry,
                )
                if chosen.empty:
                    quarantine_count += 1
                    if len(quarantine_sample) < 50:
                        quarantine_sample.append(
                            {
                                "security_id": security_id,
                                "trade_date": str(trade_date),
                                "reason": selection_reason,
                            }
                        )
                    continue
                direct_day_raw = direct.loc[direct["trade_date"].eq(str(trade_date))] if not direct.empty else direct
                direct_day, direct_selection = _choose_provider_day(
                    direct_day_raw,
                    security_id=security_id,
                    trade_date=str(trade_date),
                    registry=registry,
                )
                expected_symbol = registry.symbol_for_date(security_id, str(trade_date))
                identity_mapped = bool(expected_symbol)
                tier, reason, evidence = classify_1m_stock_day(
                    chosen,
                    daily_reference=daily_by_date.get(str(trade_date)),
                    direct_5m=direct_day if not direct_day.empty else None,
                    identity_mapped=identity_mapped,
                )
                if tier == QUALITY_QUARANTINED:
                    quarantine_count += 1
                    if len(quarantine_sample) < 50:
                        quarantine_sample.append(
                            {
                                "security_id": security_id,
                                "trade_date": str(trade_date),
                                "reason": reason,
                                "provider_selection": selection_reason,
                                "direct_5m_selection": direct_selection,
                                "evidence": evidence,
                            }
                        )
                    continue
                canonical, identity_quarantine = canonicalize_1m_day(
                    chosen,
                    identity_registry=registry,
                    quality_tier=tier,
                    quality_reason=reason,
                )
                if not identity_quarantine.empty or canonical.empty:
                    quarantine_count += 1
                    if len(quarantine_sample) < 50:
                        quarantine_sample.extend(identity_quarantine.head(50 - len(quarantine_sample)).to_dict("records"))
                    continue
                aggregated = aggregate_1m_to_5m(chosen)
                aggregated_canonical, five_identity_quarantine = canonicalize_1m_day(
                    aggregated,
                    identity_registry=registry,
                    quality_tier=tier,
                    quality_reason="derived_from_strict_1m" if tier == QUALITY_STRICT else "derived_from_provisional_1m",
                )
                if not five_identity_quarantine.empty or aggregated_canonical.empty:
                    quarantine_count += 1
                    continue
                year = str(trade_date)[:4]
                canonical_years.setdefault(year, []).append(canonical)
                five_years.setdefault(year, []).append(aggregated_canonical)
                if tier == QUALITY_STRICT:
                    strict_stock_days += 1
                    strict_keys.append((security_id, str(trade_date)))
                else:
                    provisional_stock_days += 1
            if strict_keys:
                connection.executemany("INSERT OR IGNORE INTO actual_1m VALUES (?, ?)", strict_keys)
                connection.executemany("INSERT OR IGNORE INTO actual_5m VALUES (?, ?)", strict_keys)
            bucket = stable_security_bucket(security_id)
            for year, frames in sorted(canonical_years.items()):
                frame = pd.concat(frames, ignore_index=True).loc[:, CANONICAL_ONE_MINUTE_COLUMNS]
                path = staging_root / "proxy_intraday_1m" / year / f"bucket_{bucket:02d}" / f"{security_id}.parquet"
                atomic_write_parquet(path, frame)
                one_staged.append((f"{year}_b{bucket:02d}_{security_id}", f"{year}/bucket={bucket:02d}", path))
            for year, frames in sorted(five_years.items()):
                frame = pd.concat(frames, ignore_index=True).loc[:, CANONICAL_ONE_MINUTE_COLUMNS]
                path = staging_root / "proxy_intraday_5m" / year / f"bucket_{bucket:02d}" / f"{security_id}.parquet"
                atomic_write_parquet(path, frame)
                five_staged.append((f"{year}_b{bucket:02d}_{security_id}", f"{year}/bucket={bucket:02d}", path))
        mootdx_unmapped: list[str] = []
        for ref in sorted(mootdx_refs, key=lambda item: item.partition_value):
            trade_date = str(ref.partition_value)[:10]
            raw_date = read_raw_partition(ref)
            if raw_date.empty:
                continue
            required = {"provider_symbol", "trade_date", "bar_end", "open", "high", "low", "close", "volume", "amount"}
            if not required.issubset(raw_date.columns):
                quarantine_count += 1
                if len(quarantine_sample) < 50:
                    quarantine_sample.append(
                        {"trade_date": trade_date, "reason": "mootdx_1m_raw_schema_invalid", "missing": sorted(required - set(raw_date.columns))}
                    )
                continue
            raw_date = raw_date.copy()
            raw_date["provider_symbol"] = raw_date["provider_symbol"].map(normalize_symbol)
            raw_date["trade_date"] = raw_date["trade_date"].astype(str).str.slice(0, 10)
            raw_date["bar_end"] = raw_date["bar_end"].astype(str).str.slice(-5)
            daily_by_security = _daily_reference_for_date(connection, trade_date)
            direct_date = selected_5m.loc[selected_5m["trade_date"].eq(trade_date)] if not selected_5m.empty else selected_5m
            canonical_for_date: list[pd.DataFrame] = []
            five_for_date: list[pd.DataFrame] = []
            explained_keys: list[tuple[str, str]] = []
            strict_keys: list[tuple[str, str]] = []
            for provider_symbol, raw_day in raw_date.groupby("provider_symbol", sort=True):
                security_id = registry.security_id_for_provider_symbol(provider_symbol)
                if not security_id:
                    mootdx_unmapped.append(str(provider_symbol))
                    quarantine_count += 1
                    continue
                security_id = str(security_id)
                explained_keys.append((security_id, trade_date))
                direct_raw = direct_date.loc[direct_date["provider_symbol"].eq(provider_symbol)] if not direct_date.empty else direct_date
                direct_day, direct_selection = _choose_provider_day(
                    direct_raw,
                    security_id=security_id,
                    trade_date=trade_date,
                    registry=registry,
                )
                expected_symbol = registry.symbol_for_date(security_id, trade_date)
                tier, reason, evidence = classify_1m_stock_day(
                    raw_day,
                    daily_reference=daily_by_security.get(security_id),
                    direct_5m=direct_day if not direct_day.empty else None,
                    identity_mapped=bool(expected_symbol),
                )
                if tier == QUALITY_QUARANTINED:
                    quarantine_count += 1
                    if len(quarantine_sample) < 50:
                        quarantine_sample.append(
                            {
                                "security_id": security_id,
                                "trade_date": trade_date,
                                "reason": reason,
                                "direct_5m_selection": direct_selection,
                                "evidence": evidence,
                            }
                        )
                    continue
                canonical, identity_quarantine = canonicalize_1m_day(
                    raw_day,
                    identity_registry=registry,
                    quality_tier=tier,
                    quality_reason=reason,
                )
                aggregated = aggregate_1m_to_5m(raw_day)
                aggregated_canonical, five_identity_quarantine = canonicalize_1m_day(
                    aggregated,
                    identity_registry=registry,
                    quality_tier=tier,
                    quality_reason="derived_from_strict_1m" if tier == QUALITY_STRICT else "derived_from_provisional_1m",
                )
                if (
                    not identity_quarantine.empty
                    or not five_identity_quarantine.empty
                    or canonical.empty
                    or aggregated_canonical.empty
                ):
                    quarantine_count += 1
                    continue
                canonical_for_date.append(canonical)
                if direct_day.empty:
                    five_for_date.append(aggregated_canonical)
                    recent_derived_5m_keys.add((security_id, trade_date))
                if tier == QUALITY_STRICT:
                    strict_stock_days += 1
                    strict_keys.append((security_id, trade_date))
                else:
                    provisional_stock_days += 1
            if explained_keys:
                connection.executemany("INSERT OR IGNORE INTO explained_1m VALUES (?, ?)", explained_keys)
                connection.executemany("INSERT OR IGNORE INTO explained_5m VALUES (?, ?)", explained_keys)
            if strict_keys:
                connection.executemany("INSERT OR IGNORE INTO actual_1m VALUES (?, ?)", strict_keys)
                connection.executemany("INSERT OR IGNORE INTO actual_5m VALUES (?, ?)", strict_keys)
            if canonical_for_date:
                canonical_date = pd.concat(canonical_for_date, ignore_index=True).loc[:, CANONICAL_ONE_MINUTE_COLUMNS]
                canonical_date["_bucket"] = canonical_date["security_id"].map(stable_security_bucket)
                for bucket, bucket_frame in canonical_date.groupby("_bucket", sort=True):
                    bucket_frame = bucket_frame.drop(columns=["_bucket"])
                    year = trade_date[:4]
                    path = staging_root / "mootdx_intraday_1m" / year / trade_date / f"bucket_{int(bucket):02d}.parquet"
                    atomic_write_parquet(path, bucket_frame)
                    one_staged.append((f"mootdx_{trade_date}_b{int(bucket):02d}", f"{year}/bucket={int(bucket):02d}", path))
            if five_for_date:
                canonical_five = pd.concat(five_for_date, ignore_index=True).loc[:, CANONICAL_ONE_MINUTE_COLUMNS]
                canonical_five["_bucket"] = canonical_five["security_id"].map(stable_security_bucket)
                for bucket, bucket_frame in canonical_five.groupby("_bucket", sort=True):
                    bucket_frame = bucket_frame.drop(columns=["_bucket"])
                    year = trade_date[:4]
                    path = staging_root / "mootdx_intraday_5m" / year / trade_date / f"bucket_{int(bucket):02d}.parquet"
                    atomic_write_parquet(path, bucket_frame)
                    five_staged.append((f"mootdx_{trade_date}_b{int(bucket):02d}", f"{year}/bucket={int(bucket):02d}", path))
        if not selected_5m.empty:
            for trade_date, selected_date in selected_5m.groupby("trade_date", sort=True):
                canonical_selected: list[pd.DataFrame] = []
                explained_5m_keys: list[tuple[str, str]] = []
                strict_5m_keys: list[tuple[str, str]] = []
                for provider_symbol, selected_day in selected_date.groupby("provider_symbol", sort=True):
                    security_id = registry.security_id_for_provider_symbol(provider_symbol)
                    if not security_id:
                        mootdx_unmapped.append(str(provider_symbol))
                        continue
                    security_id = str(security_id)
                    key = (security_id, str(trade_date))
                    explained_5m_keys.append(key)
                    if key in recent_derived_5m_keys:
                        continue
                    if not is_complete_5m_day(selected_day):
                        quarantine_count += 1
                        continue
                    tier_values = set(selected_day.get("quality_tier", pd.Series(QUALITY_PROVISIONAL, index=selected_day.index)).fillna(QUALITY_PROVISIONAL).astype(str))
                    tier = QUALITY_STRICT if tier_values == {QUALITY_STRICT} else QUALITY_PROVISIONAL
                    reason_values = selected_day.get("source_selection_reason", pd.Series("selected_5m_complete", index=selected_day.index)).fillna("selected_5m_complete").astype(str)
                    reason = str(reason_values.iloc[0])
                    canonical, identity_quarantine = canonicalize_1m_day(
                        selected_day,
                        identity_registry=registry,
                        quality_tier=tier,
                        quality_reason=reason,
                    )
                    if not identity_quarantine.empty or canonical.empty:
                        quarantine_count += 1
                        continue
                    canonical_selected.append(canonical)
                    if tier == QUALITY_STRICT:
                        strict_5m_keys.append(key)
                if explained_5m_keys:
                    connection.executemany("INSERT OR IGNORE INTO explained_5m VALUES (?, ?)", explained_5m_keys)
                if strict_5m_keys:
                    connection.executemany("INSERT OR IGNORE INTO actual_5m VALUES (?, ?)", strict_5m_keys)
                if canonical_selected:
                    canonical_date = pd.concat(canonical_selected, ignore_index=True).loc[:, CANONICAL_ONE_MINUTE_COLUMNS]
                    canonical_date["_bucket"] = canonical_date["security_id"].map(stable_security_bucket)
                    for bucket, bucket_frame in canonical_date.groupby("_bucket", sort=True):
                        bucket_frame = bucket_frame.drop(columns=["_bucket"])
                        year = str(trade_date)[:4]
                        path = staging_root / "selected_intraday_5m" / year / str(trade_date) / f"bucket_{int(bucket):02d}.parquet"
                        atomic_write_parquet(path, bucket_frame)
                        five_staged.append((f"selected_{trade_date}_b{int(bucket):02d}", f"{year}/bucket={int(bucket):02d}", path))
        coverage, missing_sample = _quality_coverage(
            connection,
            actual_table="actual_1m",
            explained_table="explained_1m",
        )
        coverage_5m_raw, missing_5m_sample = _quality_coverage(
            connection,
            actual_table="actual_5m",
            explained_table="explained_5m",
        )
    finally:
        connection.close()
        try:
            os.remove(database_path)
        except OSError:
            pass
    findings: list[QualityFinding] = []
    if one_unmapped or five_unmapped or mootdx_unmapped:
        findings.append(
            QualityFinding(
                code="proxy_intraday_identity_unmapped",
                severity="blocker",
                message="One or more proxy intraday raw partitions cannot map to a stable security identity.",
                domain=DOMAIN_MARKET_INTRADAY_1M,
                count=len(set(one_unmapped + five_unmapped + mootdx_unmapped)),
                sample=[{"provider_symbol": item} for item in sorted(set(one_unmapped + five_unmapped + mootdx_unmapped))[:20]],
            )
        )
    if not five_refs:
        findings.append(
            QualityFinding(
                code="proxy_direct_5m_audit_raw_missing",
                severity="blocker",
                message="Direct proxy 5m history is required as full audit evidence for historical 1m.",
                domain=DOMAIN_MARKET_INTRADAY_5M,
            )
        )
    if int(coverage.get("expected_stock_day_count", 0)) == 0:
        findings.append(
            QualityFinding(
                code="strict_intraday_expected_universe_empty",
                severity="blocker",
                message="No PIT main-board traded stock-days exist in the expected coverage table.",
                domain=DOMAIN_MARKET_INTRADAY_1M,
            )
        )
    elif float(coverage.get("strict_coverage_rate", 0.0)) < 0.9995:
        findings.append(
            QualityFinding(
                code="strict_1m_coverage_below_99_95_percent",
                severity="blocker",
                message="Strict historical 1m coverage is below the 99.95% release floor.",
                domain=DOMAIN_MARKET_INTRADAY_1M,
                count=int(coverage.get("strict_missing_stock_day_count", 0)),
                sample=missing_sample,
            )
        )
    if bool(coverage.get("non_continuous_history_hole", False)):
        findings.append(
            QualityFinding(
                code="strict_1m_non_continuous_history_hole",
                severity="blocker",
                message="Minute evidence exists after an earlier incomplete date; the watermark cannot skip a historical hole.",
                domain=DOMAIN_MARKET_INTRADAY_1M,
                count=len(list(coverage.get("later_explained_date_sample", []) or [])),
                sample=list(coverage.get("later_explained_date_sample", []) or []),
            )
        )
    if quarantine_count:
        findings.append(
            QualityFinding(
                code="proxy_intraday_stock_days_quarantined",
                severity="warning",
                message="Conflicting or invalid proxy stock-days are excluded from canonical visibility.",
                domain=DOMAIN_MARKET_INTRADAY_1M,
                count=quarantine_count,
                sample=quarantine_sample[:20],
            )
        )
    coverage_1m = {
        **coverage,
        "required_start_date": str(start_date),
        "end_date": str(end_date),
        "bootstrap_cutoff": bootstrap_cutoff,
        "mootdx_incremental_date_count": len(mootdx_refs),
        "strict_stock_day_count": strict_stock_days,
        "provisional_stock_day_count": provisional_stock_days,
    }
    coverage_5m = {
        **coverage_5m_raw,
        "required_start_date": str(start_date),
        "end_date": str(end_date),
        "bootstrap_cutoff": bootstrap_cutoff,
        "selected_incremental_partition_count": len(selected_refs),
    }
    one_report = report_for(DOMAIN_MARKET_INTRADAY_1M, findings, metrics=coverage_1m)
    five_findings = [
        QualityFinding(
            code=item.code.replace("1m", "5m"),
            severity=item.severity,
            message=item.message.replace("1m", "5m"),
            domain=DOMAIN_MARKET_INTRADAY_5M,
            count=item.count,
            sample=item.sample,
            provisional=item.provisional,
        )
        for item in findings
        if item.code not in {"strict_1m_coverage_below_99_95_percent", "strict_1m_non_continuous_history_hole"}
    ]
    if float(coverage_5m.get("strict_coverage_rate", 0.0)) < 0.9995 and int(coverage_5m.get("expected_stock_day_count", 0)):
        five_findings.append(
            QualityFinding(
                code="strict_5m_coverage_below_99_95_percent",
                severity="blocker",
                message="Strict 5m coverage through its continuous watermark is below 99.95%.",
                domain=DOMAIN_MARKET_INTRADAY_5M,
                count=int(coverage_5m.get("strict_missing_stock_day_count", 0)),
                sample=missing_5m_sample,
            )
        )
    if bool(coverage_5m.get("non_continuous_history_hole", False)):
        five_findings.append(
            QualityFinding(
                code="strict_5m_non_continuous_history_hole",
                severity="blocker",
                message="5m evidence exists after an earlier incomplete date.",
                domain=DOMAIN_MARKET_INTRADAY_5M,
                sample=list(coverage_5m.get("later_explained_date_sample", []) or []),
            )
        )
    five_report = report_for(DOMAIN_MARKET_INTRADAY_5M, five_findings, metrics=coverage_5m)

    def frames_from_stage(items: list[tuple[str, str, Path]]) -> Iterable[tuple[str, str, pd.DataFrame]]:
        if not items:
            empty = pd.DataFrame(columns=CANONICAL_ONE_MINUTE_COLUMNS)
            yield "empty", "", empty
            return
        for shard_key, partition, path in sorted(items):
            yield shard_key, partition, pd.read_parquet(path, engine="pyarrow")

    evidence = _provider_evidence(all_refs)
    raw_hashes = [ref.content_sha256 for ref in all_refs]
    one_manifest = write_partitioned_dataset(
        root=paths.root,
        domain=DOMAIN_MARKET_INTRADAY_1M,
        partition_frames=frames_from_stage(one_staged),
        layer="canonical_tiered",
        frequency="1m",
        primary_key=["security_id", "trade_date", "bar_end"],
        quality_report=one_report,
        partitioning="natural_year_security_bucket",
        inputs=inputs,
        provider_evidence=evidence,
        raw_content_hashes=raw_hashes,
        build={
            "contract": QDP_V3_CONTRACT_VERSION,
            "raw_bar_contract": "tushare_proxy_241_including_0930",
            "canonical_bar_contract": "mootdx_240_0930_merged_into_0931",
            "incremental_source": "mootdx_frequency_8_240_without_additional_merge",
            "bootstrap_cutoff": bootstrap_cutoff,
            "quality_policy": "stock_day_evidence_driven_no_fixed_2020_boundary",
            "default_visibility": QUALITY_STRICT,
        },
        coverage=coverage_1m,
        units={"price": "CNY/share", "volume": "share", "amount": "CNY"},
        quarantine=[{"stock_day_count": quarantine_count, "sample": quarantine_sample[:50]}, {"stock_day_count": coverage["strict_missing_stock_day_count"], "sample": missing_sample}],
    )
    five_manifest = write_partitioned_dataset(
        root=paths.root,
        domain=DOMAIN_MARKET_INTRADAY_5M,
        partition_frames=frames_from_stage(five_staged),
        layer="canonical_tiered_derived",
        frequency="5m",
        primary_key=["security_id", "trade_date", "bar_end"],
        quality_report=five_report,
        partitioning="natural_year_security_bucket",
        inputs=inputs,
        provider_evidence=evidence,
        raw_content_hashes=raw_hashes,
        build={
            "contract": QDP_V3_CONTRACT_VERSION,
            "canonical_source": "strict_or_provisional_1m_aggregation",
            "direct_5m_role": "full_audit_evidence",
            "post_bootstrap_source_policy": "complete_mootdx_else_complete_baostock_never_stitch",
            "default_visibility": QUALITY_STRICT,
        },
        coverage=coverage_5m,
        units={"price": "CNY/share", "volume": "share", "amount": "CNY"},
        quarantine=[{"stock_day_count": quarantine_count, "sample": quarantine_sample[:50]}, {"stock_day_count": coverage["strict_missing_stock_day_count"], "sample": missing_sample}],
    )
    return one_manifest, five_manifest, all_refs, {
        "findings": findings,
        "coverage_1m": coverage_1m,
        "coverage_5m": coverage_5m,
        "quarantine": quarantine_sample,
        "quarantine_count": quarantine_count,
        "missing_sample": missing_sample,
    }
