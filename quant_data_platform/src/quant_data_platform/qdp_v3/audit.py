from __future__ import annotations

import json
import os
import re
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
    QUALITY_PROVISIONAL,
    RAW_ALL_STOCK,
    RAW_DAILY_ASTOCK,
    RAW_TUSHARE_PROXY_NAMECHANGE,
    STRICT_RELEASE_DOMAINS,
)
from quant_data_platform.qdp_v3.corporate_actions import build_share_capital_daily
from quant_data_platform.qdp_v3.datasets import read_dataset_frame
from quant_data_platform.qdp_v3.identity import SecurityIdentityRegistry, normalize_symbol
from quant_data_platform.qdp_v3.intraday import stable_security_bucket
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
from quant_data_platform.qdp_v3.quality import (
    audit_canonical_daily,
    audit_intraday_5m,
    audit_strict_5m_coverage,
    audit_table_contract,
    audit_trusted_factor_daily,
)
from quant_data_platform.qdp_v3.release import read_candidate, validate_candidate_graph
from quant_data_platform.qdp_v3.storage import (
    RawPartitionRef,
    get_raw_partition_version,
    read_raw_partition,
    read_raw_receipt,
)
from quant_data_platform.qdp_v3.transforms import build_eligible_signal_view, build_tradable_open_view
from quant_data_platform.tushare_proxy import TUSHARE_PROXY_TOKEN_ENV


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


_SECRET_ARTIFACT_SUFFIXES = frozenset({".json", ".jsonl", ".log", ".md", ".txt", ".toml", ".yaml", ".yml"})
_EMBEDDED_TOKEN_URL = re.compile(r"/token=(?!<redacted>)[^\s\"'?#]+", re.IGNORECASE)


def _json_contains_unredacted_secret_key(value: Any) -> bool:
    if isinstance(value, dict):
        for key, item in value.items():
            normalized = str(key).strip().lower()
            if normalized in {"token", "access_token", "api_token", "authorization", "password", "secret"}:
                serialized = str(item or "").strip()
                is_content_hash = bool(re.fullmatch(r"[0-9a-fA-F]{64}", serialized))
                if serialized not in {"", "<redacted>"} and not is_content_hash:
                    return True
            if _json_contains_unredacted_secret_key(item):
                return True
    elif isinstance(value, list):
        return any(_json_contains_unredacted_secret_key(item) for item in value)
    return False


def _audit_secret_artifacts(
    paths: Any,
    *,
    workspace_root: str | Path | None,
    candidate: Any | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Reject credentials in repository text and QDP runtime control artifacts."""

    token = str(os.environ.get(TUSHARE_PROXY_TOKEN_ENV, "") or "")
    workspace = Path(workspace_root or paths.root.parents[2]).resolve()
    artifact_roots = (
        paths.jobs,
        paths.metadata,
        paths.compatibility,
        paths.candidates,
        paths.active,
        paths.rollbacks,
        paths.pins,
        paths.audits,
    )
    repository_roots = (
        workspace / "brain",
        workspace / "quant_data_platform" / "brain",
        workspace / "quant_data_platform" / "configs",
        workspace / "quant_data_platform" / "src",
    )
    files: dict[Path, bool] = {}
    for root in artifact_roots:
        if root.exists():
            for path in root.rglob("*"):
                if path.is_file() and path.suffix.lower() in _SECRET_ARTIFACT_SUFFIXES:
                    files[path.resolve()] = True
    if candidate is not None:
        for item in candidate.raw_partitions:
            for field_name in ("quality_assessment_path",):
                value = str(item.get(field_name, "") or "")
                if not value:
                    continue
                path = resolve_manifest_path(value, root=paths.root)
                if path.exists() and path.is_file():
                    files[path.resolve()] = True
        for domain, dataset_id in candidate.datasets.items():
            path = dataset_manifest_for_id(paths.root, str(dataset_id), str(domain))
            if path is not None and path.exists():
                files[path.resolve()] = True
    for root in repository_roots:
        if root.exists():
            for path in root.rglob("*"):
                if path.is_file() and path.suffix.lower() in _SECRET_ARTIFACT_SUFFIXES | {".py"}:
                    files.setdefault(path.resolve(), False)

    exact_matches: list[str] = []
    token_url_matches: list[str] = []
    secret_key_matches: list[str] = []
    for path, is_runtime_artifact in sorted(files.items(), key=lambda item: str(item[0])):
        try:
            if path.stat().st_size > 64 * 1024 * 1024:
                continue
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeError):
            continue
        display = path.relative_to(workspace).as_posix() if path.is_relative_to(workspace) else str(path)
        if token and token in text:
            exact_matches.append(display)
        if not is_runtime_artifact:
            continue
        if _EMBEDDED_TOKEN_URL.search(text):
            token_url_matches.append(display)
        if path.suffix.lower() in {".json", ".jsonl"}:
            try:
                payload = json.loads(text)
            except (TypeError, ValueError):
                payload = None
            if payload is not None and _json_contains_unredacted_secret_key(payload):
                secret_key_matches.append(display)

    findings: list[dict[str, Any]] = []
    if exact_matches:
        findings.append(
            _blocker(
                "tushare_proxy_token_material_leaked",
                "The configured proxy credential appears in repository or QDP runtime text.",
                count=len(set(exact_matches)),
                sample=sorted(set(exact_matches))[:20],
            )
        )
    if token_url_matches:
        findings.append(
            _blocker(
                "tushare_proxy_token_url_leaked",
                "A QDP runtime artifact contains an unredacted token-in-path URL.",
                count=len(set(token_url_matches)),
                sample=sorted(set(token_url_matches))[:20],
            )
        )
    if secret_key_matches:
        findings.append(
            _blocker(
                "runtime_artifact_unredacted_secret_key",
                "A QDP runtime JSON artifact contains a non-redacted credential field.",
                count=len(set(secret_key_matches)),
                sample=sorted(set(secret_key_matches))[:20],
            )
        )
    return findings, {
        "secret_scanned_file_count": len(files),
        "secret_exact_match_count": len(set(exact_matches)),
        "secret_token_url_match_count": len(set(token_url_matches)),
        "secret_key_match_count": len(set(secret_key_matches)),
    }


def _candidate_raw_ref(
    item: dict[str, Any],
    *,
    root: Path,
    workspace_root: str | Path | None,
) -> RawPartitionRef:
    raw_domain = str(item.get("raw_domain", "") or "")
    partition_field = str(item.get("partition_field", "") or "")
    partition_value = str(item.get("partition_value", "") or "")
    content_sha256 = str(item.get("content_sha256", "") or "")
    if not partition_field:
        receipt_path = resolve_manifest_path(str(item.get("receipt_path", "")), root=root)
        if receipt_path.suffix.lower() == ".json" and receipt_path.exists():
            partition_field = str(read_json(receipt_path).get("partition_field", "") or "")
    if not raw_domain or not partition_field or not partition_value or not content_sha256:
        raise RuntimeError(
            "candidate_raw_partition_identity_incomplete:"
            f"{raw_domain}:{partition_field}={partition_value}:{content_sha256}"
        )
    effective_workspace = workspace_root
    if effective_workspace is None:
        if root.name == "qdp_v3" and root.parent.name == "data":
            effective_workspace = root.parents[2]
        else:
            raise RuntimeError(f"candidate_raw_workspace_root_required:{root}")
    ref = get_raw_partition_version(
        raw_domain,
        partition_field=partition_field,
        partition_value=partition_value,
        content_sha256=content_sha256,
        workspace_root=effective_workspace,
    )
    if ref is None:
        raise FileNotFoundError(
            "candidate_raw_partition_not_found:"
            f"{raw_domain}:{partition_field}={partition_value}:{content_sha256}"
        )
    return ref


def _audit_raw_references(
    candidate: Any,
    *,
    root: Path,
    full: bool,
    workspace_root: str | Path | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    checked = 0
    configured_token = str(os.environ.get(TUSHARE_PROXY_TOKEN_ENV, "") or "")
    secret_exact_paths: list[str] = []
    secret_url_paths: list[str] = []
    secret_key_paths: list[str] = []
    checked_revision_dirs: set[Path] = set()
    for item in candidate.raw_partitions:
        try:
            ref = _candidate_raw_ref(
                item,
                root=root,
                workspace_root=workspace_root,
            )
            receipt = read_raw_receipt(ref)
        except FileNotFoundError as exc:
            findings.append(
                _blocker(
                    "candidate_raw_partition_missing",
                    "The exact raw content version referenced by the candidate is missing.",
                    raw_domain=item.get("raw_domain"),
                    partition_value=item.get("partition_value"),
                    content_sha256=item.get("content_sha256"),
                    error=str(exc),
                )
            )
            continue
        except (OSError, RuntimeError, ValueError) as exc:
            findings.append(
                _blocker(
                    "candidate_raw_partition_invalid",
                    "The exact raw content version could not be resolved safely.",
                    raw_domain=item.get("raw_domain"),
                    partition_value=item.get("partition_value"),
                    content_sha256=item.get("content_sha256"),
                    error=str(exc),
                )
            )
            continue
        payload_path = ref.payload_path
        receipt_path = ref.receipt_path
        serialized_receipt = json.dumps(receipt, ensure_ascii=False, default=str)
        if configured_token and configured_token in serialized_receipt:
            secret_exact_paths.append(str(receipt_path))
        if _EMBEDDED_TOKEN_URL.search(serialized_receipt):
            secret_url_paths.append(str(receipt_path))
        if _json_contains_unredacted_secret_key(receipt):
            secret_key_paths.append(str(receipt_path))
        expected_content = str(item.get("content_sha256", "") or "")
        if str(receipt.get("content_sha256", "") or "") != expected_content:
            findings.append(_blocker("candidate_raw_receipt_content_hash_mismatch", "Candidate raw hash differs from receipt.", receipt_path=str(receipt_path)))
        quality_assessment = str(item.get("quality_assessment_path", "") or "")
        quality_assessment_sha = str(item.get("quality_assessment_sha256", "") or "")
        if quality_assessment or quality_assessment_sha:
            assessment_path = resolve_manifest_path(quality_assessment, root=root)
            bundled_assessment_sha = str(
                receipt.get("quality_assessment_sha256", "") or ""
            )
            if (
                not assessment_path.exists()
                and ref.storage_kind == "bundle"
                and quality_assessment_sha
                and bundled_assessment_sha == quality_assessment_sha
            ):
                pass
            elif not assessment_path.exists():
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
        if (
            ref.storage_kind == "legacy"
            and versions_dir not in checked_revision_dirs
            and versions_dir.exists()
        ):
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
            try:
                read_raw_partition(ref, verify_hash=True)
            except (OSError, RuntimeError, ValueError) as exc:
                findings.append(
                    _blocker(
                        "raw_partition_verification_failed",
                        "Raw storage bytes or decoded content differ from the immutable reference.",
                        payload_path=str(payload_path),
                        storage_kind=ref.storage_kind,
                        error=str(exc),
                    )
                )
        checked += 1
    if secret_exact_paths:
        findings.append(
            _blocker(
                "tushare_proxy_token_material_leaked",
                "The configured proxy credential appears in a candidate raw receipt.",
                count=len(set(secret_exact_paths)),
                sample=sorted(set(secret_exact_paths))[:20],
            )
        )
    if secret_url_paths:
        findings.append(
            _blocker(
                "tushare_proxy_token_url_leaked",
                "A candidate raw receipt contains an unredacted token-in-path URL.",
                count=len(set(secret_url_paths)),
                sample=sorted(set(secret_url_paths))[:20],
            )
        )
    if secret_key_paths:
        findings.append(
            _blocker(
                "runtime_artifact_unredacted_secret_key",
                "A candidate raw receipt contains a non-redacted credential field.",
                count=len(set(secret_key_paths)),
                sample=sorted(set(secret_key_paths))[:20],
            )
        )
    return findings, {
        "raw_partition_checked_count": checked,
        "raw_receipt_secret_exact_match_count": len(set(secret_exact_paths)),
        "raw_receipt_secret_token_url_match_count": len(set(secret_url_paths)),
        "raw_receipt_secret_key_match_count": len(set(secret_key_paths)),
    }


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
                    security_ids = sorted(
                        item
                        for item in set(part.get("security_id", pd.Series(dtype=str)).astype(str))
                        if item
                    )
                    partition_kind = str(shard.partition.get("kind", "") or "")
                    if partition_kind == "natural_year_security_bucket":
                        partition_match = re.fullmatch(r"(\d{4})/bucket=(\d{2})", partition_month)
                        observed_periods = sorted(set(part.get("trade_date", pd.Series(dtype=str)).astype(str).str.slice(0, 4)))
                        declared_year = partition_match.group(1) if partition_match else partition_month[:4]
                        declared_bucket = int(partition_match.group(2)) if partition_match else -1
                        partition_identity = (declared_year, f"{declared_bucket:02d}")
                        source_valid = bool(
                            partition_match
                            and source_partition == f"{declared_year}_b{declared_bucket:02d}"
                        )
                        period_valid = observed_periods == [declared_year]
                        bucket_valid = bool(security_ids) and declared_bucket >= 0 and all(
                            stable_security_bucket(security_id) == declared_bucket
                            for security_id in security_ids
                        )
                        partition_contract_valid = period_valid and source_valid and bucket_valid
                    else:
                        observed_periods = sorted(set(part.get("trade_date", pd.Series(dtype=str)).astype(str).str.slice(0, 7).str.replace("-", "", regex=False)))
                        declared_year = partition_month
                        partition_identity = (security_ids[0], partition_month) if len(security_ids) == 1 else (source_partition, partition_month)
                        source_valid = len(security_ids) == 1 and source_partition == f"{security_ids[0]}_{partition_month}"
                        period_valid = observed_periods == [partition_month]
                        declared_bucket = -1
                        partition_contract_valid = len(security_ids) == 1 and period_valid and source_valid
                    if partition_identity in intraday_partitions:
                        findings.append(_blocker("intraday_cross_shard_partition_duplicate", "A security-period appears in more than one canonical shard.", domain=domain, partition=partition_identity))
                    intraday_partitions.add(partition_identity)
                    if not partition_contract_valid:
                        findings.append(
                            _blocker(
                                "intraday_shard_partition_contract_invalid",
                                "Intraday shard rows do not match the declared time and security-bucket partition.",
                                domain=domain,
                                path=str(shard_path),
                                security_ids=security_ids[:10],
                                observed_periods=observed_periods[:10],
                                declared_period=declared_year,
                                declared_bucket=declared_bucket,
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
    workspace_root: str | Path | None = None,
) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    if history.empty:
        return findings
    data = history.copy()
    names = data.get("name_on_date", pd.Series("", index=data.index)).fillna("").astype(str).str.strip()
    evidence = data.get("evidence_source", pd.Series("", index=data.index)).fillna("").astype(str)
    document_hash = data.get("official_document_hash", pd.Series("", index=data.index)).fillna("").astype(str)
    has_date_snapshot = evidence.str.contains("query_all_stock", regex=False)
    has_trusted_namechange = evidence.str.contains("tushare_proxy.namechange", regex=False)
    has_official_document = document_hash.str.fullmatch(r"[0-9a-fA-F]{64}", na=False)
    unsupported = data.loc[
        names.ne("") & ~(has_date_snapshot | has_trusted_namechange | has_official_document)
    ]
    if not unsupported.empty:
        findings.append(
            _blocker(
                "symbol_history_name_without_pit_evidence",
                "Historical names require a date-local snapshot or an official document hash.",
                count=int(len(unsupported)),
                sample=unsupported.head(20).to_dict("records"),
            )
        )

    if has_trusted_namechange.any() and not any(
        str(item.get("raw_domain", "")) == RAW_TUSHARE_PROXY_NAMECHANGE
        for item in candidate.raw_partitions
    ):
        findings.append(
            _blocker(
                "symbol_history_trusted_namechange_raw_missing",
                "Tushare namechange-derived history has no immutable raw lineage.",
                count=int(has_trusted_namechange.sum()),
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
        try:
            ref = _candidate_raw_ref(
                item,
                root=root,
                workspace_root=workspace_root,
            )
            raw = read_raw_partition(ref)
        except (FileNotFoundError, OSError, RuntimeError, ValueError):
            continue
        if not {"symbol", "name"}.issubset(raw.columns):
            continue
        raw = raw.loc[:, ["symbol", "name"]].copy()
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


def _audit_market_status_key_alignment(
    market_path: Path,
    status_path: Path,
    *,
    year: str,
) -> list[dict[str, Any]]:
    """Require every market row to have status while allowing explicit suspensions."""

    key = ["security_id", "trade_date"]
    market_keys = pd.read_parquet(market_path, columns=key, engine="pyarrow").drop_duplicates()
    status = pd.read_parquet(
        status_path,
        columns=[*key, "tradestatus", "is_suspended"],
        engine="pyarrow",
    ).drop_duplicates(key, keep="last")
    findings: list[dict[str, Any]] = []
    missing_status = market_keys.merge(status[key], on=key, how="left", indicator=True)
    missing_status = missing_status.loc[missing_status["_merge"].ne("both")]
    if not missing_status.empty:
        findings.append(
            _blocker(
                "market_status_key_missing",
                "Every market fact must have a matching status fact.",
                count=int(len(missing_status)),
                sample=missing_status.head(20).to_dict("records"),
                year=year,
            )
        )
    status_only = status.merge(market_keys, on=key, how="left", indicator=True)
    status_only = status_only.loc[status_only["_merge"].eq("left_only")]
    if not status_only.empty:
        suspended = status_only["is_suspended"].astype(str).str.lower().isin(
            {"1", "true", "t", "yes"}
        )
        stopped = status_only["tradestatus"].astype(str).eq("0")
        invalid = status_only.loc[~(suspended & stopped)]
        if not invalid.empty:
            findings.append(
                _blocker(
                    "status_only_row_not_suspended",
                    "A status fact without a market bar must be an explicit suspension.",
                    count=int(len(invalid)),
                    sample=invalid.head(20).to_dict("records"),
                    year=year,
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
    expected_domains = set(STRICT_RELEASE_DOMAINS)
    actual_domains = set(candidate.datasets)
    if actual_domains != expected_domains:
        findings.append(
            _blocker(
                "candidate_core_domain_contract_mismatch",
                "A production candidate must expose exactly the nine trusted-source core domains.",
                missing_domains=sorted(expected_domains - actual_domains),
                unexpected_domains=sorted(actual_domains - expected_domains),
            )
        )
    provisional_domains = sorted(
        domain for domain, tier in candidate.quality_tiers.items() if tier == QUALITY_PROVISIONAL
    )
    if provisional_domains:
        findings.append(
            _blocker(
                "candidate_provisional_quality_forbidden",
                "New production candidates may contain only strict or quarantined datasets.",
                domains=provisional_domains,
            )
        )
    findings.extend(_blocker(str(item.get("code", "candidate_blocker")), str(item.get("message", item.get("code", "candidate blocker"))), **{key: value for key, value in item.items() if key not in {"code", "message", "severity"}}) for item in candidate.blockers)
    findings.extend(_blocker(str(item.get("code", "candidate_graph_error")), "Candidate lineage graph is invalid.", **{key: value for key, value in item.items() if key != "code"}) for item in validate_candidate_graph(candidate, workspace_root=workspace_root))
    raw_findings, raw_metrics = _audit_raw_references(
        candidate,
        root=paths.root,
        full=full,
        workspace_root=workspace_root,
    )
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
            findings.extend(
                _audit_market_status_key_alignment(
                    market_parts[year],
                    status_parts[year],
                    year=year,
                )
            )
    if normalized_mode == "semantic":
        secret_findings, secret_metrics = _audit_secret_artifacts(
            paths,
            workspace_root=workspace_root,
            candidate=candidate,
        )
        findings.extend(secret_findings)
        raw_metrics.update(secret_metrics)
        identity = frames.get(DOMAIN_SECURITY_IDENTITY)
        history = frames.get(DOMAIN_SYMBOL_HISTORY)
        market_manifest = manifests.get(DOMAIN_MARKET_DAILY_RAW)
        if identity is None or history is None or market_manifest is None:
            findings.append(_blocker("semantic_identity_inputs_missing", "Semantic identity audit requires identity, history, and market datasets."))
        else:
            registry = SecurityIdentityRegistry(identity, history)
            findings.extend(
                _audit_symbol_history_name_evidence(
                    history,
                    candidate,
                    root=paths.root,
                    workspace_root=workspace_root,
                )
            )
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
        factors_manifest = manifests.get(DOMAIN_ADJUST_FACTOR_DAILY)
        if factors_manifest is None:
            findings.append(_blocker("semantic_factor_inputs_missing", "Semantic factor audit requires the trusted-source daily factor dataset."))
        else:
            for year, factor_path in sorted(_partition_paths(paths.root, factors_manifest).items()):
                factors = pd.read_parquet(factor_path, engine="pyarrow")
                factor_report = audit_trusted_factor_daily(factors)
                findings.extend(item.to_dict() for item in factor_report.blockers)
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
        candidate_intraday_coverage = dict(candidate.coverage.get("intraday_5m", {}) or {})
        candidate_end = str(candidate.coverage.get("end_date", "") or "")
        required_start = str(candidate_intraday_coverage.get("required_start_date", "2010-01-01") or "2010-01-01")
        explicitly_not_applicable = candidate_intraday_coverage.get("release_applicable") is False
        intraday_release_applicable = not (explicitly_not_applicable and candidate_end and candidate_end < required_start)
        if explicitly_not_applicable and (not candidate_end or candidate_end >= required_start):
            findings.append(
                _blocker(
                    "strict_5m_release_applicability_invalid",
                    "A candidate reaching the strict 5m policy window cannot declare intraday release checks not applicable.",
                    candidate_end_date=candidate_end,
                    required_start_date=required_start,
                )
            )
        if intraday_release_applicable:
            if intraday_manifest is None:
                findings.append(_blocker("strict_5m_dataset_missing", "Final v3 release requires the strict/provisional 5m dataset and coverage proof."))
            else:
                if intraday is not None:
                    strict_intraday = intraday.loc[intraday["quality_tier"].eq("strict")].copy() if "quality_tier" in intraday.columns else intraday
                    findings.extend(item.to_dict() for item in audit_intraday_5m(strict_intraday).blockers)
                coverage = dict(intraday_manifest.coverage if intraday_manifest is not None else {})
                findings.extend(item.to_dict() for item in audit_strict_5m_coverage(coverage))
                if bool(coverage.get("non_continuous_history_hole", False)):
                    findings.append(
                        _warning(
                            "intraday_5m_non_continuous_history_hole",
                            "Incomplete historical stock-days remain quarantined and research windows must be continuous.",
                            sample=list(coverage.get("later_explained_date_sample", []) or []),
                        )
                    )
                cutoff_payload = read_json(paths.metadata / "tushare_proxy_bootstrap_cutoff.json")
                cutoff = str(cutoff_payload.get("bootstrap_cutoff", "") or "")[:10]
                candidate_end = str(candidate.coverage.get("end_date", "") or "")[:10]
                if cutoff and candidate_end >= cutoff:
                    proof = read_json(paths.metadata / "tushare_proxy_bootstrap_cutoff_5m_proof.json")
                    minute_proofs = list(proof.get("minute_proofs", []) or [])
                    proof_valid = (
                        str(proof.get("bootstrap_cutoff", "") or "") == cutoff
                        and str(proof.get("frequency", "") or "") == "5m"
                        and bool(minute_proofs)
                        and all(
                            int(item.get("row_count", 0) or 0) == 48
                            and str(item.get("max_timestamp", "") or "").endswith("15:00:00")
                            for item in minute_proofs
                        )
                    )
                    if not proof_valid:
                        findings.append(
                            _blocker(
                                "bootstrap_cutoff_5m_proof_missing_or_invalid",
                                "The immutable bootstrap cutoff must be re-proved with a complete 48-bar 5m sample.",
                                bootstrap_cutoff=cutoff,
                            )
                        )
                manifest_5m_watermark = str(coverage.get("watermark", "") or "")
                candidate_5m_watermark = str(candidate.coverage.get("market_intraday_5m_watermark", "") or "")
                if candidate_5m_watermark != manifest_5m_watermark:
                    findings.append(
                        _blocker(
                            "intraday_5m_watermark_manifest_mismatch",
                            "Candidate and 5m dataset watermarks must be identical.",
                            candidate_watermark=candidate_5m_watermark,
                            manifest_watermark=manifest_5m_watermark,
                        )
                    )
                daily_watermark = str(candidate.coverage.get("market_daily_watermark", candidate.coverage.get("end_date", "")) or "")
                if manifest_5m_watermark != daily_watermark:
                    findings.append(
                        _blocker(
                            "intraday_5m_watermark_daily_mismatch",
                            "The sole intraday watermark must equal the daily watermark.",
                            market_intraday_5m_watermark=manifest_5m_watermark,
                            market_daily_watermark=daily_watermark,
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
