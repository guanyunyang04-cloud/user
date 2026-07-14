from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from quant_data_platform.core.json_io import read_json
from quant_data_platform.qdp_v3.constants import (
    DOMAIN_MARKET_INTRADAY_5M,
    QDP_V3_CONTRACT_VERSION,
    QUALITY_QUARANTINED,
    QUALITY_STRICT,
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
                "protocol": str(
                    receipt.get(
                        "protocol",
                        "tushare_compatible_http" if provider == "tushare_proxy" else "tdx_quote_protocol",
                    )
                    or ""
                ),
                "upstream_provenance": str(
                    receipt.get("upstream_provenance", "not_exposed" if provider == "tushare_proxy" else provider)
                    or ""
                ),
                "production_role": str(
                    receipt.get(
                        "production_role",
                        "historical_bootstrap" if provider == "tushare_proxy" else "ongoing_incremental",
                    )
                    or ""
                ),
                "token_sha256": fingerprint,
            },
        )
    return list(evidence.values())


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


def _load_proxy_5m(
    group_refs: list[tuple[str, RawPartitionRef]],
    *,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for symbol, ref in group_refs:
        normalized = normalize_tushare_proxy_5m(read_raw_partition(ref), provider_symbol=symbol)
        if normalized.empty:
            continue
        frames.append(normalized.loc[normalized["trade_date"].between(str(start_date), str(end_date))])
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
        normalized = normalized.loc[
            normalized["trade_date"].between(str(start_date), str(end_date))
            & (normalized["trade_date"] > str(bootstrap_cutoff or "0000-00-00"))
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
    proxy_refs = iter_raw_partitions(RAW_TUSHARE_PROXY_INTRADAY_5M, workspace_root=workspace_root)
    cutoff_payload = read_json(paths.metadata / "tushare_proxy_bootstrap_cutoff.json")
    bootstrap_cutoff = str(cutoff_payload.get("bootstrap_cutoff", "") or "")[:10]
    selected_refs = [
        ref
        for ref in iter_raw_partitions(RAW_INTRADAY_5M_SELECTED, workspace_root=workspace_root)
        if not bootstrap_cutoff
        or str(ref.partition_value).rsplit("_", 1)[-1][:6] >= bootstrap_cutoff.replace("-", "")[:6]
    ]
    all_refs = [*proxy_refs, *selected_refs]
    if not proxy_refs:
        finding = QualityFinding(
            code="strict_5m_raw_missing",
            severity="blocker",
            message="No completed Tushare-proxy 5m raw security histories exist.",
            domain=DOMAIN_MARKET_INTRADAY_5M,
        )
        return None, all_refs, {
            "findings": [finding],
            "coverage": {"strict_coverage_rate": 0.0, "watermark": ""},
            "quarantine": [],
            "quarantine_count": 0,
            "missing_sample": [],
        }

    proxy_groups, proxy_unmapped = _raw_groups(proxy_refs, registry=registry)
    selected = _selected_incremental_frame(
        selected_refs,
        start_date=start_date,
        end_date=end_date,
        bootstrap_cutoff=bootstrap_cutoff,
    )
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
    selected_unmapped: list[str] = []

    def quarantine(item: dict[str, Any]) -> None:
        nonlocal quarantine_count
        quarantine_count += 1
        if len(quarantine_sample) < 50:
            quarantine_sample.append(item)

    try:
        for security_id, refs_for_security in sorted(proxy_groups.items()):
            proxy = _load_proxy_5m(refs_for_security, start_date=start_date, end_date=end_date)
            if proxy.empty:
                continue
            daily_by_date = _daily_reference_for_security(connection, security_id)
            canonical_years: dict[str, list[pd.DataFrame]] = {}
            explained_keys: list[tuple[str, str]] = []
            strict_keys: list[tuple[str, str]] = []
            for trade_date, raw_day in proxy.groupby("trade_date", sort=True):
                trade_date = str(trade_date)
                explained_keys.append((security_id, trade_date))
                chosen, selection_reason = _choose_provider_day(
                    raw_day,
                    security_id=security_id,
                    trade_date=trade_date,
                    registry=registry,
                )
                if chosen.empty or not is_complete_5m_day(chosen):
                    quarantine(
                        {
                            "security_id": security_id,
                            "trade_date": trade_date,
                            "reason": selection_reason if chosen.empty else "tushare_proxy_5m_structure_invalid",
                        }
                    )
                    continue
                daily_result = reconcile_5m_with_daily(chosen, daily_by_date.get(trade_date))
                if daily_result.get("conflict"):
                    quarantine(
                        {
                            "security_id": security_id,
                            "trade_date": trade_date,
                            "reason": "tushare_proxy_5m_daily_conflict",
                            "evidence": daily_result,
                        }
                    )
                    continue
                if not daily_result.get("comparable"):
                    quarantine(
                        {
                            "security_id": security_id,
                            "trade_date": trade_date,
                            "reason": "tushare_proxy_complete_daily_proof_missing",
                        }
                    )
                    continue
                tier = QUALITY_STRICT
                reason = "tushare_proxy_complete_and_daily_consistent"
                chosen = chosen.copy()
                chosen["quality_tier"] = tier
                chosen["source_selection_reason"] = reason
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
                strict_keys.append((security_id, trade_date))
            if explained_keys:
                connection.executemany("INSERT OR IGNORE INTO explained_5m VALUES (?, ?)", explained_keys)
            if strict_keys:
                connection.executemany("INSERT OR IGNORE INTO actual_5m VALUES (?, ?)", strict_keys)
            bucket = stable_security_bucket(security_id)
            for year, frames in sorted(canonical_years.items()):
                frame = pd.concat(frames, ignore_index=True).loc[:, CANONICAL_5M_COLUMNS]
                path = staging_root / "proxy_intraday_5m" / year / f"bucket_{bucket:02d}" / f"{security_id}.parquet"
                atomic_write_parquet(path, frame)
                staged.append((year, bucket, path))

        if not selected.empty:
            for trade_date, selected_date in selected.groupby("trade_date", sort=True):
                trade_date = str(trade_date)
                canonical_by_bucket: dict[int, list[pd.DataFrame]] = {}
                explained_keys: list[tuple[str, str]] = []
                strict_keys: list[tuple[str, str]] = []
                for provider_symbol, selected_day in selected_date.groupby("provider_symbol", sort=True):
                    security_id = registry.security_id_for_provider_symbol(provider_symbol)
                    if not security_id:
                        selected_unmapped.append(str(provider_symbol))
                        quarantine({"provider_symbol": str(provider_symbol), "trade_date": trade_date, "reason": "identity_unmapped"})
                        continue
                    security_id = str(security_id)
                    explained_keys.append((security_id, trade_date))
                    if not is_complete_5m_day(selected_day):
                        quarantine({"security_id": security_id, "trade_date": trade_date, "reason": "selected_5m_structure_invalid"})
                        continue
                    daily = _daily_reference_for_security(connection, security_id).get(trade_date)
                    daily_result = reconcile_5m_with_daily(selected_day, daily)
                    if daily_result.get("conflict"):
                        quarantine(
                            {
                                "security_id": security_id,
                                "trade_date": trade_date,
                                "reason": "selected_5m_daily_conflict",
                                "evidence": daily_result,
                            }
                        )
                        continue
                    raw_tiers = set(selected_day["quality_tier"].fillna(QUALITY_QUARANTINED).astype(str))
                    if raw_tiers != {QUALITY_STRICT} or not daily_result.get("comparable"):
                        quarantine(
                            {
                                "security_id": security_id,
                                "trade_date": trade_date,
                                "reason": "selected_5m_not_strict_or_daily_proof_missing",
                            }
                        )
                        continue
                    tier = QUALITY_STRICT
                    selected_day = selected_day.copy()
                    selected_day["quality_tier"] = tier
                    canonical, identity_quarantine = canonicalize_selected_5m(
                        selected_day,
                        identity_registry=registry,
                    )
                    if not identity_quarantine.empty or canonical.empty:
                        quarantine({"security_id": security_id, "trade_date": trade_date, "reason": "identity_unmapped_or_conflicted"})
                        continue
                    bucket = stable_security_bucket(security_id)
                    canonical_by_bucket.setdefault(bucket, []).append(canonical)
                    strict_stock_days += 1
                    strict_keys.append((security_id, trade_date))
                if explained_keys:
                    connection.executemany("INSERT OR IGNORE INTO explained_5m VALUES (?, ?)", explained_keys)
                if strict_keys:
                    connection.executemany("INSERT OR IGNORE INTO actual_5m VALUES (?, ?)", strict_keys)
                for bucket, frames in sorted(canonical_by_bucket.items()):
                    frame = pd.concat(frames, ignore_index=True).loc[:, CANONICAL_5M_COLUMNS]
                    year = trade_date[:4]
                    path = staging_root / "selected_intraday_5m" / year / trade_date / f"bucket_{bucket:02d}.parquet"
                    atomic_write_parquet(path, frame)
                    staged.append((year, bucket, path))

        coverage, missing_sample = _quality_coverage(connection)
    finally:
        connection.close()
        try:
            os.remove(database_path)
        except OSError:
            pass

    coverage = {
        **coverage,
        "required_start_date": str(start_date),
        "end_date": str(end_date),
        "bootstrap_cutoff": bootstrap_cutoff,
        "selected_incremental_partition_count": len(selected_refs),
        "strict_stock_day_count": strict_stock_days,
        "provisional_stock_day_count": 0,
        "canonical_year_bucket_shard_count": len({(year, bucket) for year, bucket, _ in staged}),
    }
    findings: list[QualityFinding] = []
    unmapped = sorted(set(proxy_unmapped + selected_unmapped))
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
        raw_content_hashes=[ref.content_sha256 for ref in all_refs],
        build={
            "contract": QDP_V3_CONTRACT_VERSION,
            "canonical_source_through_bootstrap": "complete_tushare_proxy_5m_reconciled_to_daily",
            "post_bootstrap_source_policy": "complete_mootdx_else_complete_baostock_never_stitch",
            "bar_contract": "right_closed_48_bars_0935_1130_1305_1500",
            "quality_policy": "stock_day_evidence_driven",
            "tushare_proxy_raw_units": {"volume": "share", "amount": "CNY"},
            "tushare_proxy_canonical_scales": {"volume": 1.0, "amount": 1.0},
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
