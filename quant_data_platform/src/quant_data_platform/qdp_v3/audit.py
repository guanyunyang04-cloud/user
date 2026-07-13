from __future__ import annotations

from pathlib import Path
from typing import Any

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
    MANIFEST_VERSION,
    RAW_ALL_STOCK,
    RAW_DAILY_ASTOCK,
)
from quant_data_platform.qdp_v3.corporate_actions import build_share_capital_daily, reconstruct_xdxr_reference_prices
from quant_data_platform.qdp_v3.datasets import read_dataset_frame
from quant_data_platform.qdp_v3.identity import SecurityIdentityRegistry, normalize_symbol
from quant_data_platform.qdp_v3.manifest import (
    atomic_write_json,
    dataset_manifest_for_id,
    manifest_sha256,
    read_dataset_manifest,
    resolve_manifest_path,
    sha256_file,
    utc_now,
)
from quant_data_platform.qdp_v3.paths import ensure_qdp_v3_layout
from quant_data_platform.qdp_v3.quality import audit_canonical_daily, audit_factor_semantics, audit_intraday_5m, audit_table_contract
from quant_data_platform.qdp_v3.release import read_candidate, validate_candidate_graph
from quant_data_platform.qdp_v3.storage import frame_content_sha256
from quant_data_platform.qdp_v3.transforms import build_eligible_signal_view, build_tradable_open_view


STREAMING_DOMAINS = {
    DOMAIN_MARKET_DAILY_RAW,
    DOMAIN_SECURITY_STATUS_DAILY,
    DOMAIN_ELIGIBLE_SIGNAL_D,
    DOMAIN_TRADABLE_OPEN_D1,
    DOMAIN_ADJUST_FACTOR_DAILY,
    DOMAIN_SHARE_CAPITAL_DAILY,
    DOMAIN_MARKET_INTRADAY_5M,
    DOMAIN_VALUATION_DAILY,
}


def _blocker(code: str, message: str, **details: Any) -> dict[str, Any]:
    return {"code": code, "severity": "blocker", "message": message, **details}


def _warning(code: str, message: str, **details: Any) -> dict[str, Any]:
    return {"code": code, "severity": "warning", "message": message, **details}


def _audit_raw_references(candidate: Any, *, root: Path, full: bool) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    checked = 0
    checked_revision_dirs: set[Path] = set()
    for item in candidate.raw_partitions:
        payload_path = resolve_manifest_path(str(item.get("payload_path", "")), root=root)
        receipt_path = resolve_manifest_path(str(item.get("receipt_path", "")), root=root)
        if not payload_path.exists() or not receipt_path.exists():
            findings.append(_blocker("candidate_raw_partition_missing", "Raw payload or receipt is missing.", raw_domain=item.get("raw_domain"), partition_value=item.get("partition_value")))
            continue
        receipt = read_json(receipt_path)
        expected_content = str(item.get("content_sha256", "") or "")
        if str(receipt.get("content_sha256", "") or "") != expected_content:
            findings.append(_blocker("candidate_raw_receipt_content_hash_mismatch", "Candidate raw hash differs from receipt.", receipt_path=str(receipt_path)))
        quality_assessment = str(item.get("quality_assessment_path", "") or "")
        quality_assessment_sha = str(item.get("quality_assessment_sha256", "") or "")
        if quality_assessment or quality_assessment_sha:
            assessment_path = resolve_manifest_path(quality_assessment, root=root)
            if not assessment_path.exists():
                findings.append(
                    _blocker(
                        "candidate_raw_quality_assessment_missing",
                        "The immutable raw quality assessment referenced by the candidate is missing.",
                        assessment_path=str(assessment_path),
                    )
                )
            elif not quality_assessment_sha or sha256_file(assessment_path) != quality_assessment_sha:
                findings.append(
                    _blocker(
                        "candidate_raw_quality_assessment_hash_mismatch",
                        "Raw quality assessment bytes differ from the candidate reference.",
                        assessment_path=str(assessment_path),
                    )
                )
        versions_dir = receipt_path.parent.parent
        if versions_dir not in checked_revision_dirs and versions_dir.exists():
            checked_revision_dirs.add(versions_dir)
            version_receipts: dict[str, dict[str, Any]] = {}
            for version in [path for path in versions_dir.iterdir() if path.is_dir()]:
                version_receipt = read_json(version / "receipt.json")
                content_hash = str(version_receipt.get("content_sha256", "") or "")
                if content_hash:
                    version_receipts[content_hash] = version_receipt
            roots = [content_hash for content_hash, item in version_receipts.items() if not str(item.get("revision_of", "") or "")]
            broken = [
                {"content_sha256": content_hash, "revision_of": str(item.get("revision_of", "") or "")}
                for content_hash, item in version_receipts.items()
                if str(item.get("revision_of", "") or "") and str(item.get("revision_of", "") or "") not in version_receipts
            ]
            latest = read_json(versions_dir.parent / "latest.json")
            cursor = str(latest.get("content_sha256", "") or "")
            visited: set[str] = set()
            while cursor and cursor in version_receipts and cursor not in visited:
                visited.add(cursor)
                cursor = str(version_receipts[cursor].get("revision_of", "") or "")
            chain_invalid = bool(cursor) or (version_receipts and visited != set(version_receipts))
            if len(version_receipts) > 1 and (len(roots) != 1 or broken or chain_invalid):
                findings.append(
                    _blocker(
                        "raw_revision_chain_invalid",
                        "Raw provider revisions must form one explicit predecessor chain.",
                        versions_dir=str(versions_dir),
                        root_count=len(roots),
                        broken=broken[:20],
                        chain_visited_count=len(visited),
                        version_count=len(version_receipts),
                    )
                )
        if full:
            expected_parquet = str(receipt.get("parquet_sha256", "") or "")
            if not expected_parquet or sha256_file(payload_path) != expected_parquet:
                findings.append(_blocker("raw_parquet_hash_mismatch", "Raw parquet bytes differ from receipt.", payload_path=str(payload_path)))
            else:
                frame = pd.read_parquet(payload_path, engine="pyarrow")
                if frame_content_sha256(frame) != expected_content:
                    findings.append(_blocker("raw_decompressed_content_hash_mismatch", "Raw decompressed content differs from receipt.", payload_path=str(payload_path)))
        checked += 1
    return findings, {"raw_partition_checked_count": checked}


def _audit_dataset_manifest_and_shards(
    *,
    root: Path,
    domain: str,
    dataset_id: str,
    expected_manifest_sha: str,
    full: bool,
) -> tuple[list[dict[str, Any]], Any, pd.DataFrame | None]:
    findings: list[dict[str, Any]] = []
    path = dataset_manifest_for_id(root, dataset_id, domain)
    if path is None:
        return [_blocker("dataset_manifest_missing", "Dataset manifest is missing.", domain=domain, dataset_id=dataset_id)], None, None
    if manifest_sha256(path) != expected_manifest_sha:
        findings.append(_blocker("dataset_manifest_sha_mismatch", "Dataset manifest hash differs from candidate.", domain=domain, dataset_id=dataset_id))
    manifest = read_dataset_manifest(path)
    if manifest.manifest_version != MANIFEST_VERSION:
        findings.append(_blocker("dataset_manifest_version_invalid", "Dataset manifest is not v3.", domain=domain, dataset_id=dataset_id))
    row_count = 0
    frames: list[pd.DataFrame] = []
    intraday_partitions: set[tuple[str, str]] = set()
    declared_partitions: set[str] = set()
    for shard in manifest.shards:
        shard_path = resolve_manifest_path(shard.path, root=root)
        if not shard_path.exists():
            findings.append(_blocker("dataset_shard_missing", "Dataset shard is missing.", domain=domain, dataset_id=dataset_id, path=str(shard_path)))
            continue
        if int(shard.file_size) and shard_path.stat().st_size != int(shard.file_size):
            findings.append(_blocker("dataset_shard_size_mismatch", "Dataset shard size differs from manifest.", domain=domain, path=str(shard_path)))
        if full and sha256_file(shard_path) != shard.sha256:
            findings.append(_blocker("dataset_shard_sha_mismatch", "Dataset shard SHA256 differs from manifest.", domain=domain, path=str(shard_path)))
            continue
        if full:
            part = pd.read_parquet(shard_path, engine="pyarrow")
            if len(part) != int(shard.row_count):
                findings.append(_blocker("dataset_shard_row_count_mismatch", "Shard row count differs from manifest.", domain=domain, path=str(shard_path), expected=shard.row_count, actual=len(part)))
            expected_columns = [str(item.get("name", "")) for item in manifest.schema]
            actual_columns = [str(item) for item in part.columns]
            if actual_columns != expected_columns:
                findings.append(_blocker("dataset_schema_columns_mismatch", "Parquet columns differ from manifest schema.", domain=domain, expected=expected_columns, actual=actual_columns, path=str(shard_path)))
            if domain in STREAMING_DOMAINS:
                partition_month = str(shard.partition.get("value", "") or "")
                source_partition = str(shard.partition.get("source_partition", "") or "")
                if domain == DOMAIN_MARKET_INTRADAY_5M:
                    security_ids = sorted(set(part.get("security_id", pd.Series(dtype=str)).astype(str)))
                    observed_months = sorted(set(part.get("trade_date", pd.Series(dtype=str)).astype(str).str.slice(0, 7).str.replace("-", "", regex=False)))
                    partition_identity = (security_ids[0], partition_month) if len(security_ids) == 1 else (source_partition, partition_month)
                    if partition_identity in intraday_partitions:
                        findings.append(_blocker("intraday_cross_shard_partition_duplicate", "A security-month appears in more than one canonical shard.", domain=domain, partition=partition_identity))
                    intraday_partitions.add(partition_identity)
                    if len(security_ids) != 1 or observed_months != [partition_month] or source_partition != f"{security_ids[0]}_{partition_month}":
                        findings.append(
                            _blocker(
                                "intraday_shard_partition_contract_invalid",
                                "5m shard does not contain exactly its declared security-month.",
                                domain=domain,
                                path=str(shard_path),
                                security_ids=security_ids[:10],
                                observed_months=observed_months[:10],
                                declared_month=partition_month,
                                source_partition=source_partition,
                            )
                        )
                else:
                    if partition_month in declared_partitions:
                        findings.append(_blocker("annual_partition_duplicate", "A natural-year partition is declared more than once.", domain=domain, partition=partition_month))
                    declared_partitions.add(partition_month)
                    observed_years = sorted(set(part.get("trade_date", pd.Series(dtype=str)).astype(str).str.slice(0, 4)))
                    if observed_years != [partition_month] or source_partition != partition_month:
                        findings.append(_blocker("annual_partition_contract_invalid", "Shard rows do not match the declared natural year.", domain=domain, path=str(shard_path), declared=partition_month, observed=observed_years[:10]))
                report = audit_table_contract(part, domain=domain, primary_key=manifest.primary_key)
                findings.extend(item.to_dict() for item in report.blockers)
                if domain == DOMAIN_MARKET_INTRADAY_5M:
                    strict = part.loc[part["quality_tier"].eq("strict")].copy() if "quality_tier" in part.columns else part
                    findings.extend(item.to_dict() for item in audit_intraday_5m(strict).blockers)
                elif domain == DOMAIN_MARKET_DAILY_RAW:
                    findings.extend(item.to_dict() for item in audit_canonical_daily(part).blockers)
            else:
                frames.append(part)
            row_count += int(len(part))
        else:
            row_count += int(shard.row_count)
    if row_count != int(manifest.row_count):
        findings.append(_blocker("dataset_manifest_row_count_mismatch", "Dataset row count differs from shards.", domain=domain, expected=manifest.row_count, actual=row_count))
    frame = pd.concat(frames, ignore_index=True) if full and frames else (None if domain in STREAMING_DOMAINS else pd.DataFrame(columns=[item["name"] for item in manifest.schema]) if full else None)
    if full and frame is not None:
        actual_columns = [str(item) for item in frame.columns]
        expected_columns = [str(item.get("name", "")) for item in manifest.schema]
        if actual_columns != expected_columns:
            findings.append(_blocker("dataset_schema_columns_mismatch", "Parquet columns differ from manifest schema.", domain=domain, expected=expected_columns, actual=actual_columns))
        if not (domain == DOMAIN_ADJUST_FACTOR_EVENT and frame.empty):
            report = audit_table_contract(frame, domain=domain, primary_key=manifest.primary_key)
            findings.extend(item.to_dict() for item in report.blockers)
    return findings, manifest, frame


def _partition_paths(root: Path, manifest: Any) -> dict[str, Path]:
    return {
        str(shard.partition.get("value", "") or ""): resolve_manifest_path(shard.path, root=root)
        for shard in manifest.shards
    }


def _compare_pit_frame(actual: pd.DataFrame, expected: pd.DataFrame, *, kind: str) -> tuple[int, list[dict[str, Any]]]:
    keys = ["security_id", "trade_date"]
    compared = actual.merge(expected, on=keys, how="outer", suffixes=("_actual", "_expected"), indicator=True)
    bad = compared["_merge"].ne("both")
    if kind == "eligible":
        for column in ("symbol_on_date", "is_eligible_signal", "eligibility_reason"):
            bad |= compared[f"{column}_actual"].ne(compared[f"{column}_expected"])
    else:
        for column in ("next_trade_date", "next_symbol_on_date", "tradable_open_d1", "tradability_reason"):
            bad |= compared[f"{column}_actual"].ne(compared[f"{column}_expected"])
        left = pd.to_numeric(compared["open_d1_actual"], errors="coerce")
        right = pd.to_numeric(compared["open_d1_expected"], errors="coerce")
        bad |= ~(left.eq(right) | (left.isna() & right.isna()))
    return int(bad.sum()), compared.loc[bad].head(20).to_dict("records")


def _audit_symbol_history_name_evidence(
    history: pd.DataFrame,
    candidate: Any,
    *,
    root: Path,
) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    if history.empty:
        return findings
    data = history.copy()
    names = data.get("name_on_date", pd.Series("", index=data.index)).fillna("").astype(str).str.strip()
    evidence = data.get("evidence_source", pd.Series("", index=data.index)).fillna("").astype(str)
    document_hash = data.get("official_document_hash", pd.Series("", index=data.index)).fillna("").astype(str)
    has_date_snapshot = evidence.str.contains("query_all_stock", regex=False)
    has_official_document = document_hash.str.fullmatch(r"[0-9a-fA-F]{64}", na=False)
    unsupported = data.loc[names.ne("") & ~(has_date_snapshot | has_official_document)]
    if not unsupported.empty:
        findings.append(
            _blocker(
                "symbol_history_name_without_pit_evidence",
                "Historical names require a date-local snapshot or an official document hash.",
                count=int(len(unsupported)),
                sample=unsupported.head(20).to_dict("records"),
            )
        )

    raw_all_stock = {
        str(item.get("partition_value", "")): item
        for item in candidate.raw_partitions
        if str(item.get("raw_domain", "")) == RAW_ALL_STOCK
    }
    daily_dates = {
        str(item.get("partition_value", ""))
        for item in candidate.raw_partitions
        if str(item.get("raw_domain", "")) == RAW_DAILY_ASTOCK
    }
    missing_snapshot_dates = sorted(daily_dates - set(raw_all_stock))
    if missing_snapshot_dates:
        findings.append(
            _blocker(
                "semantic_symbol_name_snapshot_dates_missing",
                "Every daily raw partition must have date-local query_all_stock name evidence.",
                count=len(missing_snapshot_dates),
                sample=missing_snapshot_dates[:20],
            )
        )

    observed = data.loc[names.ne("") & has_date_snapshot, ["symbol", "effective_from", "name_on_date"]].copy()
    observed["symbol"] = observed["symbol"].map(normalize_symbol)
    for effective_from, expected in observed.groupby("effective_from", sort=False):
        item = raw_all_stock.get(str(effective_from))
        if item is None:
            findings.append(
                _blocker(
                    "symbol_history_name_observation_raw_missing",
                    "A query_all_stock-derived name interval has no matching raw date snapshot.",
                    effective_from=str(effective_from),
                    count=int(len(expected)),
                    sample=expected.head(20).to_dict("records"),
                )
            )
            continue
        payload_path = resolve_manifest_path(str(item.get("payload_path", "")), root=root)
        if not payload_path.exists():
            continue
        raw = pd.read_parquet(payload_path, columns=["symbol", "name"], engine="pyarrow")
        raw["symbol"] = raw["symbol"].map(normalize_symbol)
        raw["name"] = raw["name"].fillna("").astype(str).str.strip()
        actual = set(zip(raw["symbol"].astype(str), raw["name"].astype(str)))
        missing = [
            {"symbol": str(row.symbol), "effective_from": str(effective_from), "name_on_date": str(row.name_on_date)}
            for row in expected.itertuples(index=False)
            if (str(row.symbol), str(row.name_on_date)) not in actual
        ]
        if missing:
            findings.append(
                _blocker(
                    "symbol_history_name_observation_mismatch",
                    "Canonical historical names do not match their declared date-local raw evidence.",
                    effective_from=str(effective_from),
                    count=len(missing),
                    sample=missing[:20],
                )
            )
    return findings


def audit_candidate(
    *,
    candidate_id: str,
    mode: str = "quick",
    workspace_root: str | Path | None = None,
    write_report: bool = True,
) -> dict[str, Any]:
    normalized_mode = str(mode or "quick").strip().lower()
    if normalized_mode not in {"quick", "full", "semantic"}:
        raise ValueError(f"unsupported_candidate_audit_mode:{mode}")
    paths = ensure_qdp_v3_layout(workspace_root)
    candidate = read_candidate(candidate_id, workspace_root=workspace_root)
    full = normalized_mode in {"full", "semantic"}
    findings: list[dict[str, Any]] = []
    findings.extend(_blocker(str(item.get("code", "candidate_blocker")), str(item.get("message", item.get("code", "candidate blocker"))), **{key: value for key, value in item.items() if key not in {"code", "message", "severity"}}) for item in candidate.blockers)
    findings.extend(_blocker(str(item.get("code", "candidate_graph_error")), "Candidate lineage graph is invalid.", **{key: value for key, value in item.items() if key != "code"}) for item in validate_candidate_graph(candidate, workspace_root=workspace_root))
    raw_findings, raw_metrics = _audit_raw_references(candidate, root=paths.root, full=full)
    findings.extend(raw_findings)
    manifests: dict[str, Any] = {}
    frames: dict[str, pd.DataFrame] = {}
    for domain, dataset_id in candidate.datasets.items():
        dataset_findings, manifest, frame = _audit_dataset_manifest_and_shards(
            root=paths.root,
            domain=domain,
            dataset_id=dataset_id,
            expected_manifest_sha=str(candidate.dataset_manifest_sha256.get(domain, "") or ""),
            full=full,
        )
        findings.extend(dataset_findings)
        if manifest is not None:
            manifests[domain] = manifest
        if frame is not None:
            frames[domain] = frame
    if full and DOMAIN_MARKET_DAILY_RAW in manifests and DOMAIN_SECURITY_STATUS_DAILY in manifests:
        market_parts = _partition_paths(paths.root, manifests[DOMAIN_MARKET_DAILY_RAW])
        status_parts = _partition_paths(paths.root, manifests[DOMAIN_SECURITY_STATUS_DAILY])
        for year in sorted(set(market_parts) | set(status_parts)):
            if year not in market_parts or year not in status_parts:
                findings.append(_blocker("market_status_partition_misalignment", "Market and status natural-year partitions differ.", year=year))
                continue
            market_keys = pd.read_parquet(market_parts[year], columns=["security_id", "trade_date"], engine="pyarrow").drop_duplicates()
            status_keys = pd.read_parquet(status_parts[year], columns=["security_id", "trade_date"], engine="pyarrow").drop_duplicates()
            mismatch = market_keys.merge(status_keys, on=["security_id", "trade_date"], how="outer", indicator=True)
            mismatch = mismatch.loc[mismatch["_merge"].ne("both")]
            if not mismatch.empty:
                findings.append(_blocker("market_status_key_misalignment", "Market and status facts do not have identical keys.", count=int(len(mismatch)), sample=mismatch.head(20).to_dict("records"), year=year))
    if normalized_mode == "semantic":
        identity = frames.get(DOMAIN_SECURITY_IDENTITY)
        history = frames.get(DOMAIN_SYMBOL_HISTORY)
        market_manifest = manifests.get(DOMAIN_MARKET_DAILY_RAW)
        if identity is None or history is None or market_manifest is None:
            findings.append(_blocker("semantic_identity_inputs_missing", "Semantic identity audit requires identity, history, and market datasets."))
        else:
            registry = SecurityIdentityRegistry(identity, history)
            findings.extend(_audit_symbol_history_name_evidence(history, candidate, root=paths.root))
            security_id = registry.security_id_for_provider_symbol("302132.SZ")
            if not security_id or registry.symbol_for_date(security_id, "2016-01-04") != "300114.SZ":
                findings.append(_blocker("semantic_code_history_regression_failed", "2016 history for provider 302132 did not restore 300114.SZ."))
            for shard_path in _partition_paths(paths.root, market_manifest).values():
                market_part = pd.read_parquet(shard_path, columns=["trade_date", "provider_symbol", "symbol_on_date"], engine="pyarrow")
                rewritten = market_part.loc[(market_part["provider_symbol"].eq("302132.SZ")) & (market_part["trade_date"].lt("2025-02-17"))]
                leaked = rewritten.loc[rewritten["symbol_on_date"].ne("300114.SZ")]
                if not leaked.empty:
                    findings.append(_blocker("provider_current_code_leaked_into_history", "Provider current code leaked into historical symbol_on_date.", count=int(len(leaked)), sample=leaked.head(20).to_dict("records")))
        status_manifest = manifests.get(DOMAIN_SECURITY_STATUS_DAILY)
        eligible_manifest = manifests.get(DOMAIN_ELIGIBLE_SIGNAL_D)
        tradable_manifest = manifests.get(DOMAIN_TRADABLE_OPEN_D1)
        if market_manifest is None or status_manifest is None or eligible_manifest is None or tradable_manifest is None:
            findings.append(_blocker("semantic_pit_views_missing", "PIT signal eligibility and D+1 raw-open label substrate are required."))
        else:
            status_parts = _partition_paths(paths.root, status_manifest)
            eligible_parts = _partition_paths(paths.root, eligible_manifest)
            for year in sorted(set(status_parts) | set(eligible_parts)):
                if year not in status_parts or year not in eligible_parts:
                    findings.append(_blocker("eligible_signal_partition_missing", "eligible_signal_D and status partitions differ.", year=year))
                    continue
                actual = pd.read_parquet(eligible_parts[year], engine="pyarrow")
                expected_eligible = build_eligible_signal_view(pd.read_parquet(status_parts[year], engine="pyarrow"))
                bad_count, sample = _compare_pit_frame(actual, expected_eligible, kind="eligible")
                if bad_count:
                    findings.append(_blocker("eligible_signal_pit_rebuild_mismatch", "eligible_signal_D is not a pure signal-date status derivation.", count=bad_count, sample=sample, year=year))
            market_parts = _partition_paths(paths.root, market_manifest)
            tradable_parts = _partition_paths(paths.root, tradable_manifest)
            future_first = pd.DataFrame()
            for year in sorted(set(market_parts) | set(tradable_parts), reverse=True):
                if year not in market_parts or year not in tradable_parts:
                    findings.append(_blocker("tradable_open_partition_missing", "tradable_open_D1 and market partitions differ.", year=year))
                    continue
                market_part = pd.read_parquet(market_parts[year], engine="pyarrow")
                combined = pd.concat([market_part, future_first], ignore_index=True) if not future_first.empty else market_part
                expected_tradable = build_tradable_open_view(combined)
                expected_tradable = expected_tradable.loc[expected_tradable["trade_date"].astype(str).str.startswith(year)]
                actual = pd.read_parquet(tradable_parts[year], engine="pyarrow")
                bad_count, sample = _compare_pit_frame(actual, expected_tradable, kind="tradable")
                if bad_count:
                    findings.append(_blocker("tradable_open_d1_rebuild_mismatch", "tradable_open_D1 differs from the next raw market observation.", count=bad_count, sample=sample, year=year))
                invalid = actual["next_trade_date"].astype(str).ne("") & actual["next_trade_date"].astype(str).le(actual["trade_date"].astype(str))
                if invalid.any():
                    findings.append(_blocker("tradable_open_d1_nonfuture_date", "D+1 label rows must point strictly after the signal date.", count=int(invalid.sum()), year=year))
                current_first = market_part.sort_values(["security_id", "trade_date"]).drop_duplicates("security_id", keep="first")
                future_first = current_first if future_first.empty else pd.concat([current_first, future_first], ignore_index=True).sort_values(["security_id", "trade_date"]).drop_duplicates("security_id", keep="first")
        events = frames.get(DOMAIN_ADJUST_FACTOR_EVENT)
        factors_manifest = manifests.get(DOMAIN_ADJUST_FACTOR_DAILY)
        if events is None or factors_manifest is None:
            findings.append(_blocker("semantic_factor_inputs_missing", "Semantic factor audit requires event and daily factor datasets."))
        else:
            previous_last = pd.DataFrame()
            event_keys = set(zip(events.get("security_id", []), events.get("divid_operate_date", [])))
            for year, factor_path in sorted(_partition_paths(paths.root, factors_manifest).items()):
                factors = pd.read_parquet(factor_path, engine="pyarrow")
                factor_report = audit_factor_semantics(events, factors)
                findings.extend(item.to_dict() for item in factor_report.blockers)
                first = factors.sort_values(["security_id", "trade_date"]).drop_duplicates("security_id", keep="first")
                if not previous_last.empty:
                    boundary = previous_last.merge(first, on="security_id", suffixes=("_previous", "_current"))
                    changed = pd.Series(False, index=boundary.index)
                    for column in ("fore_adjust_factor", "back_adjust_factor", "adjust_factor"):
                        left = pd.to_numeric(boundary[f"{column}_previous"], errors="coerce")
                        right = pd.to_numeric(boundary[f"{column}_current"], errors="coerce")
                        changed |= left.notna() & right.notna() & left.ne(right)
                    illegal = boundary.loc[changed & ~boundary.apply(lambda row: (row["security_id"], row["trade_date_current"]) in event_keys, axis=1)]
                    if not illegal.empty:
                        findings.append(_blocker("factor_cross_year_non_event_jump", "Factor changed across a year boundary without an event.", count=int(len(illegal)), sample=illegal[["security_id", "trade_date_current"]].head(20).to_dict("records")))
                previous_last = factors.sort_values(["security_id", "trade_date"]).drop_duplicates("security_id", keep="last")
            corporate_actions = frames.get(DOMAIN_CORPORATE_ACTIONS)
            if corporate_actions is None:
                findings.append(_blocker("factor_reference_price_proof_missing", "Ex-right reference-price reconstruction needs validated corporate actions."))
            elif market_manifest is not None and not events.empty:
                proof_frames: list[pd.DataFrame] = []
                previous_last = pd.DataFrame()
                market_parts = _partition_paths(paths.root, market_manifest)
                for year, market_path in sorted(market_parts.items()):
                    market_part = pd.read_parquet(market_path, engine="pyarrow")
                    proof_market = pd.concat([previous_last, market_part], ignore_index=True) if not previous_last.empty else market_part
                    year_actions = corporate_actions.loc[corporate_actions["event_date"].astype(str).str.startswith(year)]
                    if not year_actions.empty:
                        proof_frames.append(reconstruct_xdxr_reference_prices(proof_market, year_actions))
                    previous_last = market_part.sort_values(["security_id", "trade_date"]).drop_duplicates("security_id", keep="last")
                proof = pd.concat(proof_frames, ignore_index=True) if proof_frames else pd.DataFrame(columns=["security_id", "event_date", "proof_status"])
                event_window = events.loc[
                    events["divid_operate_date"].astype(str).between(str(min(market_parts)), str(max(market_parts)) + "-12-31"),
                    ["security_id", "divid_operate_date"],
                ].drop_duplicates()
                compared = event_window.merge(
                    proof[["security_id", "event_date", "proof_status"]].rename(columns={"event_date": "divid_operate_date"}),
                    on=["security_id", "divid_operate_date"],
                    how="left",
                )
                failed = compared.loc[compared["proof_status"].ne("proved")]
                if not failed.empty:
                    findings.append(
                        _blocker(
                            "factor_reference_price_semantic_gate_failed",
                            "Every validated in-window factor event must reconstruct the ex-right reference price within one tick.",
                            count=int(len(failed)),
                            sample=failed.head(20).to_dict("records"),
                        )
                    )
        capital_events = frames.get(DOMAIN_SHARE_CAPITAL_EVENT)
        capital_daily_manifest = manifests.get(DOMAIN_SHARE_CAPITAL_DAILY)
        if capital_daily_manifest is not None:
            if capital_events is None or market_manifest is None:
                findings.append(_blocker("semantic_share_capital_inputs_missing", "PIT share-capital daily requires event and market inputs."))
            else:
                market_parts = _partition_paths(paths.root, market_manifest)
                capital_parts = _partition_paths(paths.root, capital_daily_manifest)
                for year in sorted(set(market_parts) | set(capital_parts)):
                    if year not in market_parts or year not in capital_parts:
                        findings.append(_blocker("share_capital_partition_missing", "Share-capital and market natural-year partitions differ.", year=year))
                        continue
                    expected = build_share_capital_daily(pd.read_parquet(market_parts[year], engine="pyarrow"), capital_events)
                    actual = pd.read_parquet(capital_parts[year], engine="pyarrow")
                    compared = actual.merge(expected, on=["security_id", "trade_date"], how="outer", suffixes=("_actual", "_expected"), indicator=True)
                    bad = compared["_merge"].ne("both")
                    for column in ("total_share", "float_share", "restricted_share"):
                        left = pd.to_numeric(compared[f"{column}_actual"], errors="coerce")
                        right = pd.to_numeric(compared[f"{column}_expected"], errors="coerce")
                        bad |= ~(left.eq(right) | (left.isna() & right.isna()))
                    bad |= compared["capital_event_date_actual"].fillna("").ne(compared["capital_event_date_expected"].fillna(""))
                    if bad.any():
                        findings.append(_blocker("share_capital_pit_rebuild_mismatch", "Share-capital daily contains future fill or differs from event-only forward fill.", count=int(bad.sum()), sample=compared.loc[bad].head(20).to_dict("records"), year=year))
        calendar_frame = frames.get(DOMAIN_TRADING_CALENDAR)
        open_dates = set(
            calendar_frame.loc[
                calendar_frame["is_open"].astype(str).str.lower().isin({"1", "true", "t", "yes"}),
                "trade_date",
            ].astype(str).str.slice(0, 10)
        ) if calendar_frame is not None and not calendar_frame.empty else set()
        for report_domain in (DOMAIN_FINANCIAL_QUARTERLY, DOMAIN_PERFORMANCE_FORECAST, DOMAIN_PERFORMANCE_EXPRESS):
            report_frame = frames.get(report_domain)
            if report_frame is None:
                continue
            explicit = report_frame["availability_status"].eq("next_trading_day_after_publish_date")
            invalid_explicit = explicit & (
                report_frame["available_date"].astype(str).le(report_frame["publish_date"].astype(str))
                | ~report_frame["available_date"].astype(str).isin(open_dates)
            )
            leaked_unknown = ~explicit & report_frame["available_date"].fillna("").astype(str).ne("")
            if invalid_explicit.any():
                findings.append(
                    _blocker(
                        "report_available_date_not_next_open_day",
                        "Date-only disclosures must become available strictly on an exchange trading day after publication.",
                        domain=report_domain,
                        count=int(invalid_explicit.sum()),
                        sample=report_frame.loc[invalid_explicit, ["security_id", "report_date", "publish_date", "available_date"]].head(20).to_dict("records"),
                    )
                )
            if leaked_unknown.any():
                findings.append(
                    _blocker(
                        "unproven_publish_date_leaked_into_pit",
                        "A report with unknown publication evidence must not receive an available_date.",
                        domain=report_domain,
                        count=int(leaked_unknown.sum()),
                        sample=report_frame.loc[leaked_unknown, ["security_id", "report_date", "publish_date", "available_date", "availability_status"]].head(20).to_dict("records"),
                    )
                )
        intraday = frames.get(DOMAIN_MARKET_INTRADAY_5M)
        intraday_manifest = manifests.get(DOMAIN_MARKET_INTRADAY_5M)
        if intraday_manifest is None:
            findings.append(_blocker("strict_5m_dataset_missing", "Final v3 release requires the strict/provisional 5m dataset and coverage proof."))
        else:
            if intraday is not None:
                strict_intraday = intraday.loc[intraday["quality_tier"].eq("strict")].copy() if "quality_tier" in intraday.columns else intraday
                findings.extend(item.to_dict() for item in audit_intraday_5m(strict_intraday).blockers)
            coverage = dict(intraday_manifest.coverage if intraday_manifest is not None else {})
            expected = int(coverage.get("expected_stock_day_count", 0) or 0)
            covered = int(coverage.get("strict_covered_stock_day_count", 0) or 0)
            rate = float(coverage.get("strict_coverage_rate", 0.0) or 0.0)
            if expected <= 0:
                findings.append(_blocker("strict_5m_coverage_denominator_missing", "5m release proof has no PIT tradable-stock-day denominator."))
            elif covered > expected or rate < 0.9995:
                findings.append(
                    _blocker(
                        "strict_5m_coverage_gate_failed",
                        "5m strict coverage must be internally consistent and at least 99.95%.",
                        count=max(0, expected - covered),
                        sample=[{"expected": expected, "covered": covered, "rate": rate}],
                    )
                )
    blocker_count = sum(str(item.get("severity", "")) == "blocker" for item in findings)
    payload = {
        "manifest_version": MANIFEST_VERSION,
        "status": "passed" if blocker_count == 0 else "failed",
        "mode": normalized_mode,
        "candidate_id": candidate.candidate_id,
        "candidate_manifest_sha256": manifest_sha256(paths.candidates / f"{candidate.candidate_id}.json"),
        "audited_at": utc_now(),
        "dataset_count": len(candidate.datasets),
        "finding_count": len(findings),
        "blocker_count": blocker_count,
        "findings": findings,
        "metrics": raw_metrics,
    }
    if write_report:
        stamp = utc_now().replace(":", "").replace("-", "")
        report_path = paths.audits / candidate.candidate_id / f"{stamp}__{normalized_mode}.json"
        atomic_write_json(report_path, payload)
        payload["report_path"] = str(report_path.resolve())
    return payload
