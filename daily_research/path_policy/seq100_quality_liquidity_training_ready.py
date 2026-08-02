from __future__ import annotations

"""Align newly backfilled source blocks to the frozen Seq100 common support."""

import argparse
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from quant_data_platform.domains.contracts import DataDomain
from quant_data_platform.qdp_v2.manifest import (
    dataset_manifest_for_id,
    qdp_v2_root,
    read_active_manifest,
    read_dataset_manifest,
    resolve_manifest_path,
)
from quant_data_platform.qdp_v2.status import active_dataset_map

from daily_research.path_policy import seq100_quality_liquidity_research_scope as scope

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
STUDY_ID = "seq100_quality_liquidity_training_ready_v1"
SOURCE_SCOPE_STUDY_ID = scope.STUDY_ID
DATA_HISTORY_START = "2010-01-01"
RESEARCH_START = "2011-01-01"
END_DATE = "2025-12-31"
FORBIDDEN_YEAR = 2026
RESEARCH_YEARS = tuple(range(2011, 2026))
OOS_YEARS = (2023, 2024, 2025)
EXPECTED_OOS_TRAINING_ROW_COUNTS = {2023: 3_206_854, 2024: 3_616_558, 2025: 4_028_849}
EXPECTED_COMMON_SUPPORT_ROW_COUNT = 4_441_395
EXPECTED_EXISTING_FEATURE_COUNT = 518
EXPECTED_FACTOR_FIELD_COUNT = 261
EXPECTED_FACTOR_SCHEMA_HASH = (
    "14cf668f77da06b5ddaffe59057e98c25107a870d7ec006ec5e9ee1ae574d488"
)
ROW_SPINE_CONTRACT_VERSION = 2
SUPPORT_IDENTITY_COLUMNS = (
    "candidate_id",
    "year",
    "trade_date",
    "date_idx",
    "symbol_idx",
    "symbol",
)
ROW_SPINE_COLUMNS = (*SUPPORT_IDENTITY_COLUMNS, "security_id")
FORBIDDEN_INHERITED_METADATA_COLUMNS = (
    "entry_trade_date",
    "entry_filled",
    "label_valid",
    "price_label_valid",
    "va_aux_valid",
)
ROW_KEY_COLUMNS = ("candidate_id", "trade_date", "security_id")
BALANCE_EXTENSION_NUMERIC_FIELDS = (
    "other_receivables_total",
    "other_payables_total",
    "contract_liabilities",
    "customer_advances_and_contract_liabilities",
)
BALANCE_EXTENSION_STATE_FIELDS = (
    "customer_liability_field_state",
    "other_receivables_total_field_state",
    "other_payables_total_field_state",
    "contract_liabilities_field_state",
)
BALANCE_EXTENSION_CONFLICT_FIELD = "balance_extension_source_conflict"

DEFAULT_SCOPE_OUTPUT_ROOT = (
    WORKSPACE_ROOT
    / "daily_research/output/path_policy/studies/seq100_quality_liquidity_research_scope_v1"
)
DEFAULT_OUTPUT_ROOT = (
    WORKSPACE_ROOT
    / "daily_research/output/path_policy/studies/seq100_quality_liquidity_training_ready_v1"
)
DEFAULT_STUDY_PATH = (
    WORKSPACE_ROOT
    / "daily_research/studies/seq100_quality_liquidity_training_ready_v1.json"
)

REFERENCE_FACTOR_FIELDS = frozenset(
    {
        "ts_code",
        "trade_date",
        "open",
        "high",
        "low",
        "close",
        "pre_close",
        "change",
        "pct_chg",
        "vol",
        "amount",
        "turnover_rate",
        "turnover_rate_f",
        "volume_ratio",
        "pe",
        "pe_ttm",
        "pb",
        "ps",
        "ps_ttm",
        "dv_ratio",
        "dv_ttm",
        "total_share",
        "float_share",
        "free_share",
        "total_mv",
        "circ_mv",
        "adj_factor",
    }
)
UNSUFFIXED_TECHNICAL_FIELDS = frozenset({"downdays", "updays", "lowdays", "topdays"})


class TrainingReadyError(RuntimeError):
    pass


def classify_extended_field(domain: str, field: str) -> tuple[str, str]:
    if domain == DataDomain.MARGIN_MARKET and field == "rzche":
        return "diagnostic_only", "bse_negative_source_anomaly_preserved"
    if domain == DataDomain.MARGIN_DETAIL and field in {"rzche", "rqchl"}:
        return "diagnostic_only", "provider_negative_source_values_preserved"
    if domain == DataDomain.MONEYFLOW_RAW and field in {
        "net_mf_vol",
        "net_mf_amount",
    }:
        return (
            "diagnostic_only",
            "provider_aggregate_not_reconciled_to_buy_sell_components",
        )
    return "formal_candidate", "direct_pit_lagged_provider_field"


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise TrainingReadyError(f"required_json_missing:{path}")
    return dict(json.loads(path.read_text(encoding="utf-8")))


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False, default=str)
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _stable_hash(payload: Any) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, ensure_ascii=True, separators=(",", ":"), default=str
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _row_key_hash(path: Path) -> str:
    digest = hashlib.sha256()
    parquet = pq.ParquetFile(path)
    for batch in parquet.iter_batches(
        columns=list(ROW_KEY_COLUMNS), batch_size=131_072
    ):
        frame = batch.to_pandas()
        values = pd.util.hash_pandas_object(
            frame.loc[:, list(ROW_KEY_COLUMNS)], index=False, categorize=True
        ).to_numpy(dtype=np.uint64, copy=False)
        digest.update(values.astype("<u8", copy=False).tobytes())
    return digest.hexdigest()


def _partition_profile(
    *, path: Path, profile_path: Path, row_key_hash: str
) -> dict[str, Any]:
    parquet = pq.ParquetFile(path)
    schema = parquet.schema_arrow
    row_count = int(parquet.metadata.num_rows)
    null_counts: dict[str, int | None] = {}
    for column_index, field in enumerate(schema):
        total = 0
        known = True
        for row_group_index in range(parquet.metadata.num_row_groups):
            statistics = (
                parquet.metadata.row_group(row_group_index)
                .column(column_index)
                .statistics
            )
            if statistics is None or statistics.null_count is None:
                known = False
                break
            total += int(statistics.null_count)
        null_counts[field.name] = total if known else None
    unknown = [name for name, value in null_counts.items() if value is None]
    if unknown:
        quoted = str(path).replace("'", "''")
        expressions = ",".join(
            f"count(*) FILTER(WHERE {_quoted(field)} IS NULL)" for field in unknown
        )
        with duckdb.connect() as connection:
            values = connection.execute(
                f"SELECT {expressions} FROM read_parquet('{quoted}')"
            ).fetchone()
        for field, value in zip(unknown, values, strict=True):
            null_counts[field] = int(value or 0)
    columns = [field.name for field in schema]
    profile = {
        "path": str(path.resolve()),
        "sha256": _sha256(path),
        "row_count": row_count,
        "row_key_hash": row_key_hash,
        "column_hash": _stable_hash(columns),
        "schema_hash": _stable_hash(
            [{"name": field.name, "type": str(field.type)} for field in schema]
        ),
        "null_count": {name: int(value or 0) for name, value in null_counts.items()},
        "null_rate": {
            name: float(int(value or 0) / max(row_count, 1))
            for name, value in null_counts.items()
        },
    }
    _write_json(profile_path, profile)
    return {
        "profile_path": str(profile_path.resolve()),
        "profile_sha256": _sha256(profile_path),
        "row_key_hash": row_key_hash,
        "column_hash": profile["column_hash"],
        "schema_hash": profile["schema_hash"],
    }


def _assert_same_row_keys(*, block_path: Path, spine_path: Path) -> None:
    block = str(block_path).replace("'", "''")
    spine = str(spine_path).replace("'", "''")
    keys = ",".join(ROW_KEY_COLUMNS)
    with duckdb.connect() as connection:
        difference_count = connection.execute(
            f"""
            SELECT count(*) FROM (
              (SELECT {keys} FROM read_parquet('{block}')
               EXCEPT ALL
               SELECT {keys} FROM read_parquet('{spine}'))
              UNION ALL
              (SELECT {keys} FROM read_parquet('{spine}')
               EXCEPT ALL
               SELECT {keys} FROM read_parquet('{block}'))
            )
            """
        ).fetchone()[0]
        order_violations = connection.execute(
            f"""
            SELECT count(*) FILTER(WHERE previous_candidate_id IS NOT NULL
                                      AND candidate_id<=previous_candidate_id)
            FROM (
              SELECT candidate_id,lag(candidate_id) OVER () AS previous_candidate_id
              FROM read_parquet('{block}')
            )
            """
        ).fetchone()[0]
    if int(difference_count or 0) or int(order_violations or 0):
        raise TrainingReadyError(
            "feature_block_row_keys_changed:"
            f"{block_path}:{difference_count}:{order_violations}"
        )


def _scan(paths: Sequence[Path]) -> str:
    literals = ",".join("'" + str(path).replace("'", "''") + "'" for path in paths)
    return f"read_parquet([{literals}], union_by_name=true)"


def _quoted(name: str) -> str:
    return '"' + str(name).replace('"', '""') + '"'


def _projection(alias: str, columns: Sequence[str]) -> str:
    return ",".join(f"{alias}.{_quoted(column)}" for column in columns)


def _parquet_columns(path: Path) -> tuple[str, ...]:
    return tuple(pq.read_schema(path).names)


def _date_dependency_columns(path: Path) -> tuple[str, ...]:
    return tuple(
        name
        for name in _parquet_columns(path)
        if name == "trade_date" or name.endswith("_date")
    )


def _forbidden_date_row_count(
    path: Path, *, columns: Sequence[str] | None = None
) -> int:
    selected = tuple(columns) if columns is not None else _date_dependency_columns(path)
    if not selected:
        return 0
    available = set(_parquet_columns(path))
    missing = [column for column in selected if column not in available]
    if missing:
        raise TrainingReadyError(f"date_dependency_columns_missing:{path}:{missing}")
    predicate = " OR ".join(
        f"try_cast({_quoted(column)} AS DATE)>=DATE '{FORBIDDEN_YEAR}-01-01'"
        for column in selected
    )
    literal = str(path).replace("'", "''")
    with duckdb.connect() as connection:
        value = connection.execute(
            f"SELECT count(*) FILTER(WHERE {predicate}) FROM read_parquet('{literal}')"
        ).fetchone()[0]
    return int(value or 0)


def _assert_safe_row_spine_schema(path: Path) -> None:
    actual = _parquet_columns(path)
    if actual != ROW_SPINE_COLUMNS:
        raise TrainingReadyError(
            f"row_spine_schema_changed:{path}:{actual}:{ROW_SPINE_COLUMNS}"
        )


def _assert_safe_feature_identity_schema(path: Path) -> None:
    actual = _parquet_columns(path)
    prefix = actual[: len(ROW_SPINE_COLUMNS)]
    forbidden = sorted(set(actual).intersection(FORBIDDEN_INHERITED_METADATA_COLUMNS))
    if prefix != ROW_SPINE_COLUMNS or forbidden:
        raise TrainingReadyError(
            f"feature_identity_schema_changed:{path}:{prefix}:{forbidden}"
        )


def _history_mapping_profile(history_paths: Sequence[Path]) -> dict[str, Any]:
    source = _scan(history_paths)
    eligible = f"try_cast(effective_from AS DATE)<=DATE '{END_DATE}'"
    with duckdb.connect() as connection:
        invalid_date_count = int(
            connection.execute(
                f"SELECT count(*) FROM {source} "
                "WHERE try_cast(effective_from AS DATE) IS NULL"
            ).fetchone()[0]
            or 0
        )
        excluded_after_end_count = int(
            connection.execute(
                f"SELECT count(*) FROM {source} WHERE NOT ({eligible})"
            ).fetchone()[0]
            or 0
        )
        ambiguous_symbol_count = int(
            connection.execute(
                f"SELECT count(*) FROM (SELECT symbol FROM {source} "
                f"WHERE {eligible} GROUP BY symbol "
                "HAVING count(DISTINCT security_id)>1)"
            ).fetchone()[0]
            or 0
        )
        mapping = connection.execute(
            "SELECT symbol,min(security_id) AS security_id "
            f"FROM {source} WHERE {eligible} GROUP BY symbol "
            "HAVING count(DISTINCT security_id)=1 ORDER BY symbol"
        ).fetchall()
    if invalid_date_count or ambiguous_symbol_count:
        raise TrainingReadyError(
            "symbol_history_mapping_contract_failed:"
            f"{invalid_date_count}:{ambiguous_symbol_count}"
        )
    digest = hashlib.sha256()
    for symbol, security_id in mapping:
        digest.update(f"{symbol}|{security_id}\n".encode())
    return {
        "cutoff_date": END_DATE,
        "mapping_row_count": len(mapping),
        "mapping_hash": digest.hexdigest(),
        "invalid_effective_from_row_count": invalid_date_count,
        "ambiguous_symbol_count": ambiguous_symbol_count,
        "excluded_after_end_row_count": excluded_after_end_count,
    }


def _source_support_boundary_profile(
    scope_state: Mapping[str, Any],
) -> dict[str, Any]:
    membership = dict(scope_state.get("membership_years", {}) or {})
    present_columns: set[str] = set()
    explicit_future_date_rows = 0
    partitions: dict[str, Any] = {}
    for year in RESEARCH_YEARS:
        path = Path(str(dict(membership[str(year)])["support_path"]))
        columns = _parquet_columns(path)
        missing = sorted(set(SUPPORT_IDENTITY_COLUMNS).difference(columns))
        if missing:
            raise TrainingReadyError(
                f"source_support_identity_columns_missing:{year}:{missing}"
            )
        inherited = sorted(
            set(columns).intersection(FORBIDDEN_INHERITED_METADATA_COLUMNS)
        )
        present_columns.update(inherited)
        date_columns = tuple(
            column for column in ("trade_date", "entry_trade_date") if column in columns
        )
        future_count = _forbidden_date_row_count(path, columns=date_columns)
        explicit_future_date_rows += future_count
        partitions[str(year)] = {
            "forbidden_metadata_columns_present": inherited,
            "explicit_future_date_row_count": future_count,
        }
    return {
        "projection_columns": list(SUPPORT_IDENTITY_COLUMNS),
        "forbidden_metadata_columns_present": sorted(present_columns),
        "explicit_future_date_row_count": explicit_future_date_rows,
        "partitions": partitions,
    }


def _copy_query(connection: duckdb.DuckDBPyConnection, sql: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.parquet")
    if temporary.exists():
        temporary.unlink()
    quoted = str(temporary).replace("'", "''")
    connection.execute(
        f"COPY ({sql}) TO '{quoted}' "
        "(FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 131072)"
    )
    temporary.replace(path)


def _load_config(path: Path) -> dict[str, Any]:
    payload = _read_json(path)
    period = dict(payload.get("period", {}) or {})
    burn_in = dict(payload.get("burn_in", {}) or {})
    input_contract = dict(payload.get("input_contract", {}) or {})
    if not str(payload.get("study_id", "")):
        raise TrainingReadyError("training_ready_study_id_changed")
    if (
        period.get("data_history_start") != DATA_HISTORY_START
        or period.get("research_start_date") != RESEARCH_START
        or period.get("end_date") != END_DATE
        or int(period.get("forbidden_year", 0)) != FORBIDDEN_YEAR
        or tuple(period.get("future_oos_prediction_years", ())) != OOS_YEARS
    ):
        raise TrainingReadyError("training_ready_date_boundary_changed")
    if (
        tuple(burn_in.get("years", ())) != (2010,)
        or burn_in.get("available_for_feature_history") is not True
        or burn_in.get("eligible_for_training") is not False
        or burn_in.get("eligible_for_evaluation") is not False
        or burn_in.get("eligible_for_labels") is not False
        or burn_in.get("eligible_for_atlas_statistics") is not False
    ):
        raise TrainingReadyError("training_ready_burn_in_semantics_changed")
    if dict(payload.get("training", {}) or {}).get("performed") is not False:
        raise TrainingReadyError("training_ready_must_not_train")
    if (
        int(input_contract.get("row_spine_contract_version", 0))
        != ROW_SPINE_CONTRACT_VERSION
        or tuple(input_contract.get("row_spine_projection", ())) != ROW_SPINE_COLUMNS
        or tuple(input_contract.get("forbidden_inherited_metadata_columns", ()))
        != FORBIDDEN_INHERITED_METADATA_COLUMNS
        or input_contract.get("source_support_future_metadata_policy")
        != "document_but_do_not_project"
        or input_contract.get("symbol_history_cutoff_date") != END_DATE
        or input_contract.get("feature_loading") != "feature_registry_whitelist_only"
        or input_contract.get("label_validity_source")
        != "cutoff_memmaps_and_label_flags_only"
    ):
        raise TrainingReadyError("training_ready_input_contract_changed")
    return payload


def _scope_state(
    scope_output_root: Path,
    *,
    expected_study_id: str = SOURCE_SCOPE_STUDY_ID,
    expected_row_count: int = EXPECTED_COMMON_SUPPORT_ROW_COUNT,
) -> tuple[dict[str, Any], dict[str, Any]]:
    state = _read_json(scope_output_root / "state.json")
    manifest = _read_json(scope_output_root / "manifest.json")
    if state.get("study_id") != expected_study_id:
        raise TrainingReadyError("source_scope_study_mismatch")
    if state.get("status") != "completed" or manifest.get("status") != "completed":
        raise TrainingReadyError("source_scope_not_completed")
    if (
        state.get("training_performed") is not False
        or manifest.get("training_performed") is not False
    ):
        raise TrainingReadyError("source_scope_already_trained")
    if (
        int(manifest.get("common_support", {}).get("row_count", 0))
        != expected_row_count
    ):
        raise TrainingReadyError("source_scope_row_count_changed")
    if int(manifest.get("forbidden_2026_rows", 0)) != 0:
        raise TrainingReadyError("source_scope_contains_2026")
    return state, manifest


def _dataset_map(
    workspace: Path, dataset_ids: Mapping[str, str] | None = None
) -> dict[str, str]:
    if dataset_ids is not None:
        return {
            str(domain): str(dataset_id)
            for domain, dataset_id in dataset_ids.items()
        }
    return active_dataset_map(read_active_manifest(qdp_v2_root(workspace)))


def _active_paths(
    workspace: Path,
    domain: str,
    dataset_ids: Mapping[str, str] | None = None,
) -> list[Path]:
    root = qdp_v2_root(workspace)
    datasets = _dataset_map(workspace, dataset_ids)
    dataset_id = datasets.get(domain, "")
    manifest_path = dataset_manifest_for_id(root, dataset_id, domain)
    if manifest_path is None:
        raise TrainingReadyError(f"active_domain_missing:{domain}")
    manifest = read_dataset_manifest(manifest_path)
    paths = [resolve_manifest_path(item.path, root=root) for item in manifest.shards]
    if not paths or not all(path.is_file() for path in paths):
        raise TrainingReadyError(f"active_domain_shards_missing:{domain}")
    return paths


def _optional_active_paths(
    workspace: Path,
    domain: str,
    dataset_ids: Mapping[str, str] | None = None,
) -> list[Path]:
    try:
        return _active_paths(workspace, domain, dataset_ids)
    except TrainingReadyError as exc:
        if str(exc).startswith("active_domain_missing"):
            return []
        raise


def _source_query_contract(
    workspace: Path, dataset_ids: Mapping[str, str] | None = None
) -> dict[str, str]:
    expected = {
        DataDomain.STK_FACTOR_PRO_RAW: "trade_date",
        DataDomain.MARGIN_MARKET: "year",
        DataDomain.MARGIN_DETAIL: "trade_date",
        DataDomain.MONEYFLOW_RAW: "trade_date",
    }
    actual: dict[str, str] = {}
    root = qdp_v2_root(workspace)
    datasets = _dataset_map(workspace, dataset_ids)
    for domain, granularity in expected.items():
        dataset_id = str(datasets.get(domain, ""))
        manifest_path = dataset_manifest_for_id(root, dataset_id, domain)
        if manifest_path is None:
            raise TrainingReadyError(f"active_source_manifest_missing:{domain}")
        payload = read_dataset_manifest(manifest_path).to_dict()
        source = dict(payload.get("source", {}) or {})
        observed = str(source.get("query_granularity", ""))
        if observed != granularity:
            raise TrainingReadyError(
                f"source_query_granularity_changed:{domain}:{observed}:{granularity}"
            )
        actual[domain] = observed
    return actual


def _validate_source_inputs(
    *,
    state: Mapping[str, Any],
    manifest: Mapping[str, Any],
    expected_row_count: int = EXPECTED_COMMON_SUPPORT_ROW_COUNT,
    expected_feature_count: int = EXPECTED_EXISTING_FEATURE_COUNT,
    expected_training_rows: Mapping[int, int] = EXPECTED_OOS_TRAINING_ROW_COUNTS,
) -> dict[str, Any]:
    support = dict(manifest.get("common_support", {}) or {})
    atlas = dict(manifest.get("atlas", {}) or {})
    support_manifest_path = Path(str(support.get("manifest_path", "")))
    atlas_manifest_path = Path(str(atlas.get("manifest_path", "")))
    support_manifest_valid = support_manifest_path.is_file() and _sha256(
        support_manifest_path
    ) == str(support.get("manifest_sha256", ""))
    atlas_manifest_valid = atlas_manifest_path.is_file() and _sha256(
        atlas_manifest_path
    ) == str(atlas.get("manifest_sha256", ""))
    atlas_payload = _read_json(atlas_manifest_path) if atlas_manifest_valid else {}
    catalog_record = dict(
        dict(atlas_payload.get("files", {}) or {}).get("feature_catalog", {}) or {}
    )
    catalog_path = Path(str(catalog_record.get("path", "")))
    folds = {
        int(item["evaluation_year"]): int(
            item["training_candidate_row_count_before_target_purge"]
        )
        for item in list(manifest.get("rolling_oos_folds", []) or [])
    }
    checks = {
        "scope_rows": int(support.get("row_count", 0))
        == expected_row_count,
        "scope_hash_present": bool(support.get("common_support_hash")),
        "scope_start": manifest.get("research_period", {}).get("start_date")
        == RESEARCH_START,
        "scope_end": manifest.get("research_period", {}).get("end_date") == END_DATE,
        "atlas_hash_present": bool(atlas.get("manifest_sha256")),
        "support_manifest_frozen": support_manifest_valid,
        "atlas_manifest_frozen": atlas_manifest_valid,
        "atlas_feature_count": int(
            atlas_payload.get("total_continuous_feature_count", 0)
        )
        == expected_feature_count,
        "atlas_common_support": atlas_payload.get("common_support_hash")
        == support.get("common_support_hash"),
        "atlas_catalog_frozen": catalog_path.is_file()
        and _sha256(catalog_path) == str(catalog_record.get("sha256", "")),
        "forbidden_2026": int(manifest.get("forbidden_2026_rows", 0)) == 0,
        "training_false": state.get("training_performed") is False,
        "oos_training_rows": folds
        == {int(year): int(count) for year, count in expected_training_rows.items()},
    }
    if not all(checks.values()):
        raise TrainingReadyError(f"source_scope_validation_failed:{checks}")
    return checks


def classify_factor_field(
    field: str,
    *,
    raw_price_validation_passed: bool,
    hfq_validation_passed: bool,
) -> tuple[str, str]:
    name = str(field)
    lower = name.lower()
    if name in {"ts_code", "trade_date"}:
        return "rejected", "provider_identity_column"
    if "_qfq" in lower or lower.endswith("_qfq"):
        return "diagnostic_only", "qfq_current_anchor_not_pit_safe"
    if name in REFERENCE_FACTOR_FIELDS:
        return "redundant_reference", "owned_by_baostock_qdp_or_existing_daily_block"
    if "_hfq" in lower:
        if hfq_validation_passed:
            return "formal_candidate", "provider_hfq_passed_adjustment_audit"
        return "rejected", "hfq_adjustment_audit_failed"
    if "_bfq" in lower or name in UNSUFFIXED_TECHNICAL_FIELDS:
        if raw_price_validation_passed:
            return "formal_candidate", "provider_raw_basis_passed_daily_audit"
        return "rejected", "raw_price_audit_failed"
    return "rejected", "ambiguous_provider_semantics"


def _factor_quality(
    workspace: Path, dataset_ids: Mapping[str, str] | None = None
) -> dict[str, Any]:
    root = qdp_v2_root(workspace)
    dataset_id = _dataset_map(workspace, dataset_ids).get(
        DataDomain.STK_FACTOR_PRO_RAW, ""
    )
    if not dataset_id:
        raise TrainingReadyError("technical_factor_domain_missing")
    manifest_path = dataset_manifest_for_id(
        root, dataset_id, DataDomain.STK_FACTOR_PRO_RAW
    )
    if manifest_path is None:
        raise TrainingReadyError("technical_factor_manifest_missing")
    manifest = read_dataset_manifest(manifest_path).to_dict()
    quality = dict(manifest.get("quality", {}) or {})
    if quality.get("provider_field_count") != EXPECTED_FACTOR_FIELD_COUNT:
        raise TrainingReadyError("technical_factor_field_count_changed")
    if quality.get("provider_schema_hash") != EXPECTED_FACTOR_SCHEMA_HASH:
        raise TrainingReadyError("technical_factor_schema_hash_changed")
    return quality


def _factor_registry(
    workspace: Path,
    output_root: Path,
    dataset_ids: Mapping[str, str] | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    factor_paths = _active_paths(
        workspace, DataDomain.STK_FACTOR_PRO_RAW, dataset_ids
    )
    schema_names = pq.read_schema(factor_paths[0]).names
    quality = _factor_quality(workspace, dataset_ids)
    raw_ok = bool(quality.get("raw_price_validation_passed"))
    hfq_ok = bool(quality.get("hfq_validation_passed"))
    rows: list[dict[str, Any]] = []
    formal_fields: list[str] = []
    for field in schema_names:
        if field in {
            "security_id",
            "symbol",
            "source_date",
            "feature_available_date",
            "burn_in_only",
            "source",
        }:
            continue
        eligibility, reason = classify_factor_field(
            field,
            raw_price_validation_passed=raw_ok,
            hfq_validation_passed=hfq_ok,
        )
        if eligibility == "formal_candidate":
            formal_fields.append(field)
        rows.append(
            {
                "feature_name": f"technical_{field}",
                "physical_column": field,
                "block": "tushare_technical_candidates",
                "source_domain": DataDomain.STK_FACTOR_PRO_RAW,
                "source_field": field,
                "eligibility": eligibility,
                "eligibility_reason": reason,
                "availability_lag": "same_signal_close"
                if field not in {"ts_code", "trade_date"}
                else "metadata",
                "raw_price_validation_passed": raw_ok,
                "hfq_validation_passed": hfq_ok,
            }
        )
    output = pd.DataFrame(rows)
    path = output_root / "feature_registry" / "technical_factor_registry.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    output.to_parquet(path, index=False, compression="zstd")
    return output, formal_fields


def _existing_registry(
    scope_manifest: Mapping[str, Any], output_root: Path
) -> pd.DataFrame:
    atlas_path = Path(
        str(dict(scope_manifest.get("atlas", {}) or {}).get("manifest_path", ""))
    )
    atlas = _read_json(atlas_path)
    catalog_path = Path(str(atlas["files"]["feature_catalog"]["path"]))
    catalog = pd.read_parquet(catalog_path)
    output = pd.DataFrame(
        {
            "feature_name": catalog["name"].astype(str),
            "physical_column": catalog["name"].astype(str),
            "block": "existing_atlas_518",
            "source_domain": "existing_seq100_atlas",
            "source_field": catalog["name"].astype(str),
            "eligibility": "formal_existing",
            "eligibility_reason": "frozen_existing_atlas",
            "availability_lag": "existing_pit_contract",
        }
    )
    path = output_root / "feature_registry" / "existing_atlas_registry.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    output.to_parquet(path, index=False, compression="zstd")
    return output


def _source_fields(
    workspace: Path,
    domain: str,
    dataset_ids: Mapping[str, str] | None = None,
) -> list[str]:
    paths = _active_paths(workspace, domain, dataset_ids)
    return pq.read_schema(paths[0]).names


def _row_spine_sql(
    *,
    support_path: Path,
    history_paths: Sequence[Path],
) -> str:
    identity = _projection("s", SUPPORT_IDENTITY_COLUMNS)
    return f"""
    WITH history AS (
      SELECT symbol,min(security_id) AS security_id
      FROM {_scan(history_paths)}
      WHERE try_cast(effective_from AS DATE)<=DATE '{END_DATE}'
      GROUP BY symbol
      HAVING count(DISTINCT security_id)=1
    )
    SELECT {identity},h.security_id
    FROM read_parquet('{str(support_path).replace("'", "''")}') s
    JOIN history h USING(symbol)
    ORDER BY candidate_id
    """


def _build_row_spines(
    *,
    workspace: Path,
    scope_state: Mapping[str, Any],
    output_root: Path,
    dataset_ids: Mapping[str, str] | None = None,
    support_path_key: str = "support_path",
    support_hash_key: str = "support_sha256",
    support_row_count_key: str = "common_support_row_count",
    output_directory: str = "row_spine",
) -> tuple[dict[str, Any], dict[str, Any]]:
    history_paths = _active_paths(workspace, DataDomain.SYMBOL_HISTORY, dataset_ids)
    history_profile = _history_mapping_profile(history_paths)
    membership = dict(scope_state.get("membership_years", {}) or {})
    records: dict[str, Any] = {}
    for year in RESEARCH_YEARS:
        support = dict(membership[str(year)])
        support_path = Path(str(support[support_path_key]))
        path = (
            output_root / output_directory / f"year={year}" / "part-0000.parquet"
        )
        input_hash = str(support.get(support_hash_key, ""))
        if not support_path.is_file() or _sha256(support_path) != input_hash:
            raise TrainingReadyError(f"source_support_partition_changed:{year}")
        sidecar = path.with_suffix(".json")
        reusable = False
        if path.is_file() and sidecar.is_file():
            previous = _read_json(sidecar)
            reusable = (
                previous.get("input_hash") == input_hash
                and previous.get("sha256") == _sha256(path)
                and int(previous.get("row_spine_contract_version", 0))
                == ROW_SPINE_CONTRACT_VERSION
                and previous.get("history_mapping_hash")
                == history_profile["mapping_hash"]
                and _parquet_columns(path) == ROW_SPINE_COLUMNS
            )
        if not reusable:
            with duckdb.connect() as connection:
                _copy_query(
                    connection,
                    _row_spine_sql(
                        support_path=support_path, history_paths=history_paths
                    ),
                    path,
                )
            with duckdb.connect() as connection:
                path_literal = str(path).replace("'", "''")
                row_count, unique_count = connection.execute(
                    "SELECT count(*),count(DISTINCT candidate_id) "
                    f"FROM read_parquet('{path_literal}')"
                ).fetchone()
            expected = int(support[support_row_count_key])
            _assert_safe_row_spine_schema(path)
            forbidden_dependency_count = _forbidden_date_row_count(path)
            if int(row_count) != expected or int(unique_count) != expected:
                raise TrainingReadyError(f"row_spine_contract_failed:{year}")
            if forbidden_dependency_count:
                raise TrainingReadyError(
                    f"row_spine_forbidden_dependency:{year}:"
                    f"{forbidden_dependency_count}"
                )
            _write_json(
                sidecar,
                {
                    "input_hash": input_hash,
                    "sha256": _sha256(path),
                    "row_count": int(row_count),
                    "security_id_joined": True,
                    "row_spine_contract_version": ROW_SPINE_CONTRACT_VERSION,
                    "projection_columns": list(ROW_SPINE_COLUMNS),
                    "history_mapping_hash": history_profile["mapping_hash"],
                    "forbidden_dependency_row_count": forbidden_dependency_count,
                },
            )
        _assert_safe_row_spine_schema(path)
        forbidden_dependency_count = _forbidden_date_row_count(path)
        if forbidden_dependency_count:
            raise TrainingReadyError(
                f"row_spine_forbidden_dependency:{year}:{forbidden_dependency_count}"
            )
        row_key_hash = _row_key_hash(path)
        profile = _partition_profile(
            path=path,
            profile_path=path.with_suffix(".profile.json"),
            row_key_hash=row_key_hash,
        )
        previous = _read_json(sidecar)
        _write_json(
            sidecar,
            {
                **previous,
                "row_key_hash": row_key_hash,
                "profile_path": profile["profile_path"],
                "profile_sha256": profile["profile_sha256"],
            },
        )
        records[str(year)] = {
            "path": str(path.resolve()),
            "sha256": _sha256(path),
            "row_count": int(pq.ParquetFile(path).metadata.num_rows),
            "support_sha256": input_hash,
            "row_spine_contract_version": ROW_SPINE_CONTRACT_VERSION,
            "projection_columns": list(ROW_SPINE_COLUMNS),
            "history_mapping_hash": history_profile["mapping_hash"],
            "forbidden_dependency_row_count": forbidden_dependency_count,
            **profile,
        }
    return records, history_profile


def _feature_select_sql(
    *,
    spine_path: Path,
    source_paths: Sequence[Path],
    fields: Sequence[str],
    source_domain: str,
    lagged: bool,
) -> str:
    source = _scan(source_paths)
    identity = _projection("s", ROW_SPINE_COLUMNS)
    if lagged:
        join = "f.security_id=s.security_id AND f.feature_available_date=s.trade_date"
    else:
        join = "f.security_id=s.security_id AND f.trade_date=s.trade_date"
    selected = ",\n           ".join(
        f"f.{_quoted(field)} AS {_quoted(field)}" for field in fields
    )
    prefix = (
        "" if source_domain == DataDomain.STK_FACTOR_PRO_RAW else source_domain + "_"
    )
    selected = ",\n           ".join(
        f"f.{_quoted(field)} AS {_quoted(prefix + field)}" for field in fields
    )
    if source_domain == DataDomain.STK_FACTOR_PRO_RAW and fields:
        all_missing = " AND ".join(f"f.{_quoted(field)} IS NULL" for field in fields)
        coverage_state = (
            "CASE WHEN f.security_id IS NULL THEN 'source_unavailable' "
            f"WHEN {all_missing} THEN 'warmup_missing' ELSE 'observed' END"
        )
    else:
        coverage_state = (
            "CASE WHEN f.security_id IS NULL THEN 'source_unavailable' "
            "ELSE 'observed' END"
        )
    return f"""
    SELECT {identity},
           {coverage_state} AS coverage_state,
           f.source_date,f.feature_available_date,
           {selected}
    FROM read_parquet('{str(spine_path).replace("'", "''")}') s
    LEFT JOIN {source} f ON {join}
    ORDER BY s.candidate_id
    """


def _build_feature_block(
    *,
    workspace: Path,
    output_root: Path,
    row_spines: Mapping[str, Mapping[str, Any]],
    domain: str,
    fields: Sequence[str],
    lagged: bool,
    dataset_ids: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    source_paths = _optional_active_paths(workspace, domain, dataset_ids)
    if not source_paths:
        return {
            "status": "source_unavailable",
            "domain": domain,
            "fields": list(fields),
            "partitions": {},
        }
    partitions: dict[str, Any] = {}
    for year in RESEARCH_YEARS:
        spine_path = Path(str(row_spines[str(year)]["path"]))
        output_path = (
            output_root / "features" / domain / f"year={year}" / "part-0000.parquet"
        )
        with duckdb.connect() as connection:
            connection.execute("SET threads=4")
            connection.execute("SET memory_limit='12GB'")
            _copy_query(
                connection,
                _feature_select_sql(
                    spine_path=spine_path,
                    source_paths=source_paths,
                    fields=fields,
                    source_domain=domain,
                    lagged=lagged,
                ),
                output_path,
            )
        with duckdb.connect() as connection:
            path_literal = str(output_path).replace("'", "''")
            row_count, unique_count = connection.execute(
                "SELECT count(*),count(DISTINCT candidate_id) "
                f"FROM read_parquet('{path_literal}')"
            ).fetchone()
            if int(row_count) != int(row_spines[str(year)]["row_count"]):
                raise TrainingReadyError(
                    f"feature_block_row_count_changed:{domain}:{year}"
                )
            _assert_safe_feature_identity_schema(output_path)
            forbidden_dependency_count = _forbidden_date_row_count(output_path)
            if int(unique_count) != int(row_count) or forbidden_dependency_count:
                raise TrainingReadyError(
                    f"feature_block_contract_failed:{domain}:{year}"
                )
            coverage = connection.execute(
                f"SELECT coverage_state,count(*) FROM read_parquet('{path_literal}') "
                "GROUP BY coverage_state ORDER BY coverage_state"
            ).fetchall()
        _assert_same_row_keys(block_path=output_path, spine_path=spine_path)
        profile = _partition_profile(
            path=output_path,
            profile_path=output_path.with_suffix(".profile.json"),
            row_key_hash=str(row_spines[str(year)]["row_key_hash"]),
        )
        partitions[str(year)] = {
            "path": str(output_path.resolve()),
            "sha256": _sha256(output_path),
            "row_count": int(row_count),
            "field_count": len(fields),
            "coverage_counts": {str(key): int(value) for key, value in coverage},
            "forbidden_dependency_row_count": forbidden_dependency_count,
            **profile,
        }
    return {
        "status": "completed",
        "domain": domain,
        "fields": list(fields),
        "partitions": partitions,
        "lagged": bool(lagged),
    }


def _balance_extension_sql(
    *,
    spine_path: Path,
    source_paths: Sequence[Path],
    conflict_gate: bool,
) -> str:
    identity = _projection("s", ROW_SPINE_COLUMNS)
    if conflict_gate:
        numeric = ",\n           ".join(
            "CASE WHEN coalesce(e."
            f"{_quoted(BALANCE_EXTENSION_CONFLICT_FIELD)},false) THEN NULL "
            f"ELSE e.{_quoted(field)} END AS {_quoted('balance_' + field)}"
            for field in BALANCE_EXTENSION_NUMERIC_FIELDS
        )
    else:
        numeric = ",\n           ".join(
            f"e.{_quoted(field)} AS {_quoted('balance_' + field)}"
            for field in BALANCE_EXTENSION_NUMERIC_FIELDS
        )
    states = ",\n           ".join(
        f"e.{_quoted(field)} AS {_quoted('balance_' + field)}"
        for field in BALANCE_EXTENSION_STATE_FIELDS
    )
    event_fields = (
        *BALANCE_EXTENSION_NUMERIC_FIELDS,
        *BALANCE_EXTENSION_STATE_FIELDS,
        *([BALANCE_EXTENSION_CONFLICT_FIELD] if conflict_gate else []),
    )
    conflict_output = (
        ",\n           e."
        f"{_quoted(BALANCE_EXTENSION_CONFLICT_FIELD)} AS "
        f"{_quoted(BALANCE_EXTENSION_CONFLICT_FIELD)}"
        if conflict_gate
        else ""
    )
    return f"""
    WITH events AS (
      SELECT symbol,feature_available_date,source_date,report_date,
             {",".join(_quoted(field) for field in event_fields)}
      FROM {_scan(source_paths)}
      WHERE feature_available_date<>'' AND feature_available_date<='{END_DATE}'
      QUALIFY row_number() OVER (
        PARTITION BY symbol,feature_available_date
        ORDER BY (report_type='1') DESC,report_date DESC,update_flag DESC
      )=1
    )
    SELECT {identity},
           CASE WHEN e.symbol IS NULL THEN 'warmup_missing'
                ELSE 'observed' END AS coverage_state,
           e.source_date,e.feature_available_date,
           {numeric},
           {states}{conflict_output}
    FROM read_parquet('{str(spine_path).replace("'", "''")}') s
    ASOF LEFT JOIN events e
      ON s.symbol=e.symbol AND s.trade_date>=e.feature_available_date
    ORDER BY s.candidate_id
    """


def _build_balance_extension_block(
    *,
    workspace: Path,
    output_root: Path,
    row_spines: Mapping[str, Mapping[str, Any]],
    dataset_ids: Mapping[str, str] | None = None,
    conflict_gate: bool = False,
) -> dict[str, Any]:
    domain = DataDomain.BALANCE_SHEET_QUARTERLY
    source_paths = _active_paths(workspace, domain, dataset_ids)
    required = set(BALANCE_EXTENSION_NUMERIC_FIELDS) | set(
        BALANCE_EXTENSION_STATE_FIELDS
    )
    if conflict_gate:
        required.add(BALANCE_EXTENSION_CONFLICT_FIELD)
    missing = sorted(required.difference(pq.read_schema(source_paths[0]).names))
    if missing:
        raise TrainingReadyError(f"balance_extension_fields_missing:{missing}")
    partitions: dict[str, Any] = {}
    for year in RESEARCH_YEARS:
        spine_path = Path(str(row_spines[str(year)]["path"]))
        output_path = (
            output_root
            / "features"
            / "financial_statement_extensions"
            / f"year={year}"
            / "part-0000.parquet"
        )
        with duckdb.connect() as connection:
            connection.execute("SET threads=4")
            connection.execute("SET memory_limit='12GB'")
            _copy_query(
                connection,
                _balance_extension_sql(
                    spine_path=spine_path,
                    source_paths=source_paths,
                    conflict_gate=conflict_gate,
                ),
                output_path,
            )
        path_literal = str(output_path).replace("'", "''")
        with duckdb.connect() as connection:
            row_count, unique_count = connection.execute(
                "SELECT count(*),count(DISTINCT candidate_id) "
                f"FROM read_parquet('{path_literal}')"
            ).fetchone()
            coverage = connection.execute(
                f"SELECT coverage_state,count(*) FROM read_parquet('{path_literal}') "
                "GROUP BY coverage_state ORDER BY coverage_state"
            ).fetchall()
        if int(row_count) != int(row_spines[str(year)]["row_count"]):
            raise TrainingReadyError(f"balance_extension_row_count_changed:{year}")
        _assert_safe_feature_identity_schema(output_path)
        forbidden_dependency_count = _forbidden_date_row_count(output_path)
        if int(unique_count) != int(row_count) or forbidden_dependency_count:
            raise TrainingReadyError(f"balance_extension_contract_failed:{year}")
        _assert_same_row_keys(block_path=output_path, spine_path=spine_path)
        profile = _partition_profile(
            path=output_path,
            profile_path=output_path.with_suffix(".profile.json"),
            row_key_hash=str(row_spines[str(year)]["row_key_hash"]),
        )
        partitions[str(year)] = {
            "path": str(output_path.resolve()),
            "sha256": _sha256(output_path),
            "row_count": int(row_count),
            "field_count": len(required),
            "coverage_counts": {str(key): int(value) for key, value in coverage},
            "forbidden_dependency_row_count": forbidden_dependency_count,
            **profile,
        }
    return {
        "status": "completed",
        "domain": domain,
        "fields": sorted(required),
        "partitions": partitions,
        "lagged": True,
        "availability_state_fields": [
            *BALANCE_EXTENSION_STATE_FIELDS,
            *([BALANCE_EXTENSION_CONFLICT_FIELD] if conflict_gate else []),
        ],
    }


def _margin_block_sql(
    *,
    spine_path: Path,
    detail_paths: Sequence[Path],
    market_paths: Sequence[Path],
    detail_fields: Sequence[str],
    market_fields: Sequence[str],
    eligibility_paths: Sequence[Path] = (),
    eligibility_domain: str = DataDomain.MARGIN_SECS,
    secs_paths: Sequence[Path] | None = None,
) -> str:
    if secs_paths is not None and not eligibility_paths:
        eligibility_paths = secs_paths
    identity = _projection("s", ROW_SPINE_COLUMNS)
    detail_values = ",\n           ".join(
        f"d.{_quoted(field)} AS {_quoted('margin_detail_' + field)}"
        for field in detail_fields
    )
    market_values = ",\n           ".join(
        f"m.{_quoted(field)} AS {_quoted('margin_market_' + field)}"
        for field in market_fields
    )
    if eligibility_paths and eligibility_domain == DataDomain.MARGIN_ELIGIBILITY:
        eligibility_join = (
            f"LEFT JOIN {_scan(eligibility_paths)} e ON e.symbol=s.symbol "
            "AND e.feature_available_date=s.trade_date"
        )
        eligibility_state = "coalesce(e.eligibility_state,'source_unavailable')"
        coverage_state = (
            "CASE WHEN d.security_id IS NOT NULL THEN 'observed' "
            "WHEN e.eligibility_state='known_ineligible' THEN 'not_applicable' "
            "ELSE 'source_unavailable' END"
        )
        eligibility_metadata = """
           e.eligible AS margin_eligible,
           e.finance_eligible AS margin_finance_eligible,
           e.securities_lending_eligible AS margin_securities_lending_eligible,
           e.detail_observed AS margin_exchange_detail_observed,
           e.source_available AS margin_eligibility_source_available,
           e.eligibility_source_available AS margin_exchange_eligibility_source_available,
           e.detail_source_available AS margin_exchange_detail_source_available,
        """
    elif eligibility_paths:
        eligibility_join = (
            f"LEFT JOIN {_scan(eligibility_paths)} e ON e.security_id=s.security_id "
            "AND e.feature_available_date=s.trade_date"
        )
        eligibility_state = (
            "CASE WHEN e.security_id IS NOT NULL THEN 'observed' "
            "ELSE 'not_applicable' END"
        )
        coverage_state = (
            "CASE WHEN d.security_id IS NOT NULL THEN 'observed' "
            "WHEN e.security_id IS NOT NULL THEN 'source_unavailable' "
            "ELSE 'not_applicable' END"
        )
        eligibility_metadata = ""
    else:
        eligibility_join = ""
        eligibility_state = "'source_unavailable'"
        coverage_state = (
            "CASE WHEN d.security_id IS NOT NULL THEN 'observed' "
            "ELSE 'source_unavailable' END"
        )
        eligibility_metadata = ""
    return f"""
    SELECT {identity},
           {coverage_state} AS coverage_state,
           CASE WHEN d.security_id IS NULL THEN 'source_unavailable'
                ELSE 'observed' END AS margin_detail_coverage_state,
           CASE WHEN m.exchange_id IS NULL THEN 'source_unavailable'
                ELSE 'observed' END AS margin_market_coverage_state,
           {eligibility_state} AS margin_eligibility_state,
           {eligibility_metadata}
           d.source_date AS margin_detail_source_date,
           d.feature_available_date AS margin_detail_feature_available_date,
           m.source_date AS margin_market_source_date,
           m.feature_available_date AS margin_market_feature_available_date,
           {detail_values},
           {market_values}
    FROM read_parquet('{str(spine_path).replace("'", "''")}') s
    LEFT JOIN {_scan(detail_paths)} d
      ON d.security_id=s.security_id AND d.feature_available_date=s.trade_date
    LEFT JOIN {_scan(market_paths)} m
      ON m.feature_available_date=s.trade_date
     AND m.exchange_id=CASE
       WHEN right(s.symbol,2)='SH' THEN 'SSE'
       WHEN right(s.symbol,2)='SZ' THEN 'SZSE'
       WHEN right(s.symbol,2)='BJ' THEN 'BSE'
       ELSE '' END
    {eligibility_join}
    ORDER BY s.candidate_id
    """


def _build_margin_block(
    *,
    workspace: Path,
    output_root: Path,
    row_spines: Mapping[str, Mapping[str, Any]],
    detail_fields: Sequence[str],
    market_fields: Sequence[str],
    eligibility_domain: str = DataDomain.MARGIN_SECS,
    dataset_ids: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    detail_paths = _active_paths(workspace, DataDomain.MARGIN_DETAIL, dataset_ids)
    market_paths = _active_paths(workspace, DataDomain.MARGIN_MARKET, dataset_ids)
    eligibility_paths = _optional_active_paths(
        workspace, eligibility_domain, dataset_ids
    )
    partitions: dict[str, Any] = {}
    for year in RESEARCH_YEARS:
        spine_path = Path(str(row_spines[str(year)]["path"]))
        output_path = (
            output_root
            / "features"
            / "margin_features"
            / f"year={year}"
            / "part-0000.parquet"
        )
        with duckdb.connect() as connection:
            connection.execute("SET threads=4")
            connection.execute("SET memory_limit='12GB'")
            _copy_query(
                connection,
                _margin_block_sql(
                    spine_path=spine_path,
                    detail_paths=detail_paths,
                    market_paths=market_paths,
                    eligibility_paths=eligibility_paths,
                    eligibility_domain=eligibility_domain,
                    detail_fields=detail_fields,
                    market_fields=market_fields,
                ),
                output_path,
            )
        path_literal = str(output_path).replace("'", "''")
        with duckdb.connect() as connection:
            row_count, unique_count = connection.execute(
                "SELECT count(*),count(DISTINCT candidate_id) "
                f"FROM read_parquet('{path_literal}')"
            ).fetchone()
            coverage = connection.execute(
                f"SELECT coverage_state,count(*) FROM read_parquet('{path_literal}') "
                "GROUP BY coverage_state ORDER BY coverage_state"
            ).fetchall()
            detail_coverage = connection.execute(
                f"SELECT margin_detail_coverage_state,count(*) "
                f"FROM read_parquet('{path_literal}') "
                "GROUP BY margin_detail_coverage_state "
                "ORDER BY margin_detail_coverage_state"
            ).fetchall()
            market_coverage = connection.execute(
                f"SELECT margin_market_coverage_state,count(*) "
                f"FROM read_parquet('{path_literal}') "
                "GROUP BY margin_market_coverage_state "
                "ORDER BY margin_market_coverage_state"
            ).fetchall()
        if int(row_count) != int(row_spines[str(year)]["row_count"]):
            raise TrainingReadyError(f"margin_block_row_count_changed:{year}")
        _assert_safe_feature_identity_schema(output_path)
        forbidden_dependency_count = _forbidden_date_row_count(output_path)
        if int(unique_count) != int(row_count) or forbidden_dependency_count:
            raise TrainingReadyError(f"margin_block_contract_failed:{year}")
        _assert_same_row_keys(block_path=output_path, spine_path=spine_path)
        profile = _partition_profile(
            path=output_path,
            profile_path=output_path.with_suffix(".profile.json"),
            row_key_hash=str(row_spines[str(year)]["row_key_hash"]),
        )
        partitions[str(year)] = {
            "path": str(output_path.resolve()),
            "sha256": _sha256(output_path),
            "row_count": int(row_count),
            "field_count": len(detail_fields) + len(market_fields),
            "coverage_counts": {str(key): int(value) for key, value in coverage},
            "margin_detail_coverage_counts": {
                str(key): int(value) for key, value in detail_coverage
            },
            "margin_market_coverage_counts": {
                str(key): int(value) for key, value in market_coverage
            },
            "forbidden_dependency_row_count": forbidden_dependency_count,
            **profile,
        }
    return {
        "status": "completed",
        "domain": "margin_features",
        "detail_fields": list(detail_fields),
        "market_fields": list(market_fields),
        "margin_secs_status": (
            "observed"
            if eligibility_paths and eligibility_domain == DataDomain.MARGIN_SECS
            else "superseded_by_margin_eligibility"
            if eligibility_paths
            else "source_unavailable"
        ),
        "margin_eligibility_domain": eligibility_domain,
        "margin_eligibility_status": (
            "observed" if eligibility_paths else "source_unavailable"
        ),
        "partitions": partitions,
        "lagged": True,
    }


def _coverage_registry(
    *,
    output_root: Path,
    blocks: Mapping[str, Mapping[str, Any]],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for block_name, block in blocks.items():
        for year, partition in dict(block.get("partitions", {}) or {}).items():
            counts = dict(partition.get("coverage_counts", {}) or {})
            detail_counts = dict(
                partition.get("margin_detail_coverage_counts", {}) or {}
            )
            market_counts = dict(
                partition.get("margin_market_coverage_counts", {}) or {}
            )
            total = sum(counts.values())
            rows.append(
                {
                    "block": block_name,
                    "year": int(year),
                    "row_count": int(partition.get("row_count", total)),
                    "observed_count": int(counts.get("observed", 0)),
                    "source_unavailable_count": int(
                        counts.get("source_unavailable", 0)
                    ),
                    "not_applicable_count": int(counts.get("not_applicable", 0)),
                    "warmup_missing_count": int(counts.get("warmup_missing", 0)),
                    "margin_detail_observed_count": int(
                        detail_counts.get("observed", 0)
                    ),
                    "margin_market_observed_count": int(
                        market_counts.get("observed", 0)
                    ),
                    "observed_rate": float(counts.get("observed", 0) / max(total, 1)),
                    "unavailable_rate": float(
                        counts.get("source_unavailable", 0) / max(total, 1)
                    ),
                }
            )
    output = pd.DataFrame(rows)
    path = output_root / "coverage_registry" / "coverage_by_year.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    output.to_parquet(path, index=False, compression="zstd")
    return output


def _coverage_findings(coverage: pd.DataFrame) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    for block, group in coverage.groupby("block", sort=True):
        ordered = group.sort_values("year")
        previous_rate: float | None = None
        for row in ordered.itertuples(index=False):
            observed_rate = float(row.observed_rate)
            if observed_rate == 0.0:
                findings.append(
                    {
                        "block": str(block),
                        "year": int(row.year),
                        "kind": "no_observed_rows",
                        "observed_rate": observed_rate,
                    }
                )
            elif (
                previous_rate is not None
                and previous_rate >= 0.10
                and observed_rate < previous_rate - 0.20
            ):
                findings.append(
                    {
                        "block": str(block),
                        "year": int(row.year),
                        "kind": "observed_rate_drop_over_20pp",
                        "previous_rate": previous_rate,
                        "observed_rate": observed_rate,
                    }
                )
            previous_rate = observed_rate
    return findings


def prepare(
    *,
    study_path: Path = DEFAULT_STUDY_PATH,
    scope_output_root: Path = DEFAULT_SCOPE_OUTPUT_ROOT,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    workspace_root: Path = WORKSPACE_ROOT,
) -> dict[str, Any]:
    config = _load_config(study_path)
    study_id = str(config["study_id"])
    source_config = dict(config.get("source_scope", {}) or {})
    source_scope_study_id = str(
        source_config.get("study_id", SOURCE_SCOPE_STUDY_ID)
    )
    expected_row_count = int(
        source_config.get(
            "expected_common_support_rows",
            EXPECTED_COMMON_SUPPORT_ROW_COUNT,
        )
    )
    expected_feature_count = int(
        source_config.get(
            "expected_existing_feature_count",
            EXPECTED_EXISTING_FEATURE_COUNT,
        )
    )
    expected_training_rows = {
        int(year): int(count)
        for year, count in dict(
            config.get(
                "expected_rolling_training_rows",
                EXPECTED_OOS_TRAINING_ROW_COUNTS,
            )
            or {}
        ).items()
    }
    scope_state, scope_manifest = _scope_state(
        scope_output_root,
        expected_study_id=source_scope_study_id,
        expected_row_count=expected_row_count,
    )
    _validate_source_inputs(
        state=scope_state,
        manifest=scope_manifest,
        expected_row_count=expected_row_count,
        expected_feature_count=expected_feature_count,
        expected_training_rows=expected_training_rows,
    )
    dataset_ids = {
        str(domain): str(dataset_id)
        for domain, dataset_id in dict(
            scope_state.get("qdp_dataset_ids", {}) or {}
        ).items()
    }
    if not dataset_ids:
        raise TrainingReadyError("source_scope_qdp_dataset_ids_missing")
    dataset_overrides = {
        str(domain): str(dataset_id)
        for domain, dataset_id in dict(
            source_config.get("dataset_id_overrides", {}) or {}
        ).items()
    }
    for domain, dataset_id in dataset_overrides.items():
        if (
            dataset_manifest_for_id(qdp_v2_root(workspace_root), dataset_id, domain)
            is None
        ):
            raise TrainingReadyError(
                f"source_scope_dataset_override_missing:{domain}:{dataset_id}"
            )
    dataset_ids.update(dataset_overrides)
    source_query_contract = _source_query_contract(workspace_root, dataset_ids)
    source_support_boundary = _source_support_boundary_profile(scope_state)
    output_root.mkdir(parents=True, exist_ok=True)
    row_spines, history_mapping = _build_row_spines(
        workspace=workspace_root,
        scope_state=scope_state,
        output_root=output_root,
        dataset_ids=dataset_ids,
    )
    daily_quality_row_spines: dict[str, Any] = {}
    if bool(config.get("require_dual_pool_manifest", False)):
        daily_quality_row_spines, daily_history_mapping = _build_row_spines(
            workspace=workspace_root,
            scope_state=scope_state,
            output_root=output_root,
            dataset_ids=dataset_ids,
            support_path_key="quality_support_path",
            support_hash_key="quality_support_sha256",
            support_row_count_key="quality_support_row_count",
            output_directory="row_spine_quality_liquidity_pit",
        )
        if daily_history_mapping != history_mapping:
            raise TrainingReadyError("dual_pool_symbol_history_mapping_changed")
    existing_registry = _existing_registry(scope_manifest, output_root)
    factor_registry, factor_fields = _factor_registry(
        workspace_root,
        output_root,
        dataset_ids,
    )
    margin_detail_fields = [
        field
        for field in _source_fields(
            workspace_root, DataDomain.MARGIN_DETAIL, dataset_ids
        )
        if field
        not in {
            "security_id",
            "symbol",
            "ts_code",
            "trade_date",
            "source_date",
            "feature_available_date",
            "burn_in_only",
            "source",
        }
    ]
    margin_market_fields = [
        field
        for field in _source_fields(
            workspace_root, DataDomain.MARGIN_MARKET, dataset_ids
        )
        if field
        not in {
            "trade_date",
            "exchange_id",
            "source_date",
            "feature_available_date",
            "burn_in_only",
            "source",
        }
    ]
    moneyflow_fields = [
        field
        for field in _source_fields(
            workspace_root, DataDomain.MONEYFLOW_RAW, dataset_ids
        )
        if field
        not in {
            "security_id",
            "symbol",
            "ts_code",
            "trade_date",
            "source_date",
            "feature_available_date",
            "burn_in_only",
            "source",
        }
    ]
    source_policy = dict(config.get("source_policy", {}) or {})
    balance_extension_conflict_gate = (
        source_policy.get("balance_extension_conflict_gate") == "required"
    )
    eligibility_domain = (
        DataDomain.MARGIN_ELIGIBILITY
        if source_policy.get("margin_eligibility")
        == "exchange_tristate_availability_metadata"
        else DataDomain.MARGIN_SECS
    )
    blocks = {
        "tushare_technical_candidates": _build_feature_block(
            workspace=workspace_root,
            output_root=output_root,
            row_spines=row_spines,
            domain=DataDomain.STK_FACTOR_PRO_RAW,
            fields=factor_fields,
            lagged=False,
            dataset_ids=dataset_ids,
        ),
        "margin_features": _build_margin_block(
            workspace=workspace_root,
            output_root=output_root,
            row_spines=row_spines,
            detail_fields=margin_detail_fields,
            market_fields=margin_market_fields,
            eligibility_domain=eligibility_domain,
            dataset_ids=dataset_ids,
        ),
        "traditional_moneyflow_features": _build_feature_block(
            workspace=workspace_root,
            output_root=output_root,
            row_spines=row_spines,
            domain=DataDomain.MONEYFLOW_RAW,
            fields=moneyflow_fields,
            lagged=True,
            dataset_ids=dataset_ids,
        ),
    }
    if bool(source_policy.get("include_financial_statement_extensions", False)):
        blocks["financial_statement_extensions"] = _build_balance_extension_block(
            workspace=workspace_root,
            output_root=output_root,
            row_spines=row_spines,
            dataset_ids=dataset_ids,
            conflict_gate=balance_extension_conflict_gate,
        )
    coverage = _coverage_registry(output_root=output_root, blocks=blocks)
    margin_detail_eligibility = {
        field: classify_extended_field(DataDomain.MARGIN_DETAIL, field)
        for field in margin_detail_fields
    }
    margin_market_eligibility = {
        field: classify_extended_field(DataDomain.MARGIN_MARKET, field)
        for field in margin_market_fields
    }
    moneyflow_eligibility = {
        field: classify_extended_field(DataDomain.MONEYFLOW_RAW, field)
        for field in moneyflow_fields
    }
    additional_registry_rows = (
        [
            {
                "feature_name": f"margin_eligibility_{field}",
                "physical_column": f"margin_{field}",
                "block": "margin_features",
                "source_domain": DataDomain.MARGIN_ELIGIBILITY,
                "source_field": field,
                "eligibility": "availability_metadata",
                "eligibility_reason": "exchange_tristate_not_auto_model_input",
                "availability_lag": "next_open_day",
            }
            for field in (
                "eligible",
                "finance_eligible",
                "securities_lending_eligible",
                "exchange_detail_observed",
                "eligibility_source_available",
                "exchange_eligibility_source_available",
                "exchange_detail_source_available",
                "eligibility_state",
            )
        ]
        if eligibility_domain == DataDomain.MARGIN_ELIGIBILITY
        else []
    )
    if "financial_statement_extensions" in blocks:
        additional_registry_rows.extend(
            [
                {
                    "feature_name": f"balance_{field}",
                    "physical_column": f"balance_{field}",
                    "block": "financial_statement_extensions",
                    "source_domain": DataDomain.BALANCE_SHEET_QUARTERLY,
                    "source_field": field,
                    "eligibility": "availability_gated",
                    "eligibility_reason": "pit_valid_extended_statement_field",
                    "availability_lag": "next_open_day",
                }
                for field in BALANCE_EXTENSION_NUMERIC_FIELDS
            ]
            + [
                {
                    "feature_name": f"balance_{field}",
                    "physical_column": f"balance_{field}",
                    "block": "financial_statement_extensions",
                    "source_domain": DataDomain.BALANCE_SHEET_QUARTERLY,
                    "source_field": field,
                    "eligibility": "availability_metadata",
                    "eligibility_reason": "missingness_semantics_not_numeric_signal",
                    "availability_lag": "next_open_day",
                }
                for field in BALANCE_EXTENSION_STATE_FIELDS
            ]
            + (
                [
                    {
                        "feature_name": BALANCE_EXTENSION_CONFLICT_FIELD,
                        "physical_column": BALANCE_EXTENSION_CONFLICT_FIELD,
                        "block": "financial_statement_extensions",
                        "source_domain": DataDomain.BALANCE_SHEET_QUARTERLY,
                        "source_field": BALANCE_EXTENSION_CONFLICT_FIELD,
                        "eligibility": "availability_metadata",
                        "eligibility_reason": "conflicting_extension_values_are_gated",
                        "availability_lag": "next_open_day",
                    }
                ]
                if balance_extension_conflict_gate
                else []
            )
        )
    feature_registry = pd.concat(
        [
            existing_registry,
            factor_registry,
            pd.DataFrame(
                [
                    {
                        "feature_name": f"margin_detail_{field}",
                        "physical_column": f"margin_detail_{field}",
                        "block": "margin_features",
                        "source_domain": DataDomain.MARGIN_DETAIL,
                        "source_field": field,
                        "eligibility": margin_detail_eligibility[field][0],
                        "eligibility_reason": margin_detail_eligibility[field][1],
                        "availability_lag": "next_open_day",
                    }
                    for field in margin_detail_fields
                ]
                + [
                    {
                        "feature_name": f"margin_market_{field}",
                        "physical_column": f"margin_market_{field}",
                        "block": "margin_features",
                        "source_domain": DataDomain.MARGIN_MARKET,
                        "source_field": field,
                        "eligibility": margin_market_eligibility[field][0],
                        "eligibility_reason": margin_market_eligibility[field][1],
                        "availability_lag": "next_open_day",
                    }
                    for field in margin_market_fields
                ]
                + [
                    {
                        "feature_name": f"moneyflow_{field}",
                        "physical_column": f"moneyflow_raw_{field}",
                        "block": "traditional_moneyflow_features",
                        "source_domain": DataDomain.MONEYFLOW_RAW,
                        "source_field": field,
                        "eligibility": moneyflow_eligibility[field][0],
                        "eligibility_reason": moneyflow_eligibility[field][1],
                        "availability_lag": "next_open_day",
                    }
                    for field in moneyflow_fields
                ]
                + additional_registry_rows
            ),
        ],
        ignore_index=True,
    )
    feature_registry.to_parquet(
        output_root / "feature_registry" / "feature_registry.parquet",
        index=False,
        compression="zstd",
    )
    factor_quality = _factor_quality(workspace_root, dataset_ids)
    scope_hash = str(scope_manifest["common_support"]["common_support_hash"])
    blocks_status = [str(block.get("status")) for block in blocks.values()]
    mandatory_ready = bool(
        factor_quality.get("raw_price_validation_passed")
        and factor_quality.get("hfq_validation_passed")
        and blocks_status.count("completed") == len(blocks)
    )
    coverage_findings = _coverage_findings(coverage)
    has_documented_gaps = bool(
        blocks["margin_features"].get("margin_secs_status") == "source_unavailable"
        or int(coverage["source_unavailable_count"].sum()) > 0
        or int(coverage["warmup_missing_count"].sum()) > 0
        or coverage_findings
    )
    readiness_status = (
        "ready"
        if mandatory_ready and not has_documented_gaps
        else "ready_with_documented_optional_gaps"
    )
    consumed_forbidden_dependency_rows = sum(
        int(partition.get("forbidden_dependency_row_count", 0))
        for partition in row_spines.values()
    ) + sum(
        int(partition.get("forbidden_dependency_row_count", 0))
        for partition in daily_quality_row_spines.values()
    ) + sum(
        int(partition.get("forbidden_dependency_row_count", 0))
        for block in blocks.values()
        for partition in dict(block.get("partitions", {}) or {}).values()
    )
    if consumed_forbidden_dependency_rows:
        raise TrainingReadyError(
            "training_ready_forbidden_dependency_consumed:"
            f"{consumed_forbidden_dependency_rows}"
        )
    manifest = {
        "schema": "seq100_quality_liquidity_training_ready/"
        + study_id.rsplit("_", maxsplit=1)[-1],
        "status": "completed",
        "study_id": study_id,
        "source_scope_study_id": source_scope_study_id,
        "qdp_dataset_ids": dataset_ids,
        "source_query_contract": source_query_contract,
        "data_history": {
            "start_date": DATA_HISTORY_START,
            "burn_in_years": [2010],
            "burn_in_available_for_feature_history": True,
            "burn_in_eligible_for_training": False,
            "burn_in_eligible_for_evaluation": False,
            "burn_in_eligible_for_labels": False,
            "burn_in_eligible_for_atlas_statistics": False,
        },
        "research_period": {
            "start_date": RESEARCH_START,
            "end_date": END_DATE,
            "years": list(RESEARCH_YEARS),
            "future_oos_prediction_years": list(OOS_YEARS),
        },
        "rolling_oos_folds": list(scope_manifest["rolling_oos_folds"]),
        "common_support": {
            "row_count": expected_row_count,
            "common_support_hash": scope_hash,
            "stock_day_identity": "security_id",
            "daily_and_minute_models_share_exact_rows": True,
            "missing_5m_action": "drop_stock_day_only",
        },
        "pools": dict(scope_manifest.get("pools", {}) or {}),
        "row_spine_contract": {
            "version": ROW_SPINE_CONTRACT_VERSION,
            "projection_columns": list(ROW_SPINE_COLUMNS),
            "forbidden_inherited_metadata_columns": list(
                FORBIDDEN_INHERITED_METADATA_COLUMNS
            ),
            "source_support_boundary": source_support_boundary,
            "symbol_history_mapping": history_mapping,
            "consumed_forbidden_dependency_row_count": (
                consumed_forbidden_dependency_rows
            ),
        },
        "row_spine": row_spines,
        "daily_quality_row_spine": daily_quality_row_spines,
        "existing_atlas_518": {
            "manifest_path": scope_manifest["atlas"]["manifest_path"],
            "manifest_sha256": scope_manifest["atlas"]["manifest_sha256"],
            "feature_count": expected_feature_count,
        },
        "feature_registry": {
            "path": str(
                (
                    output_root / "feature_registry" / "feature_registry.parquet"
                ).resolve()
            ),
            "sha256": _sha256(
                output_root / "feature_registry" / "feature_registry.parquet"
            ),
            "formal_candidate_count": int(
                (feature_registry["eligibility"] == "formal_candidate").sum()
            ),
            "diagnostic_only_count": int(
                (feature_registry["eligibility"] == "diagnostic_only").sum()
            ),
            "qfq_formal_candidate_count": int(
                feature_registry.loc[
                    feature_registry["source_field"]
                    .astype(str)
                    .str.contains("qfq", case=False, na=False),
                    "eligibility",
                ]
                .eq("formal_candidate")
                .sum()
            ),
            "tushare_factor_classified_count": len(factor_registry),
        },
        "blocks": blocks,
        "coverage_registry": {
            "path": str(
                (
                    output_root / "coverage_registry" / "coverage_by_year.parquet"
                ).resolve()
            ),
            "sha256": _sha256(
                output_root / "coverage_registry" / "coverage_by_year.parquet"
            ),
            "row_count": len(coverage),
        },
        "training_performed": False,
        "feature_set_selected": False,
        "readiness_status": readiness_status,
        "config": config,
        "forbidden_2026_rows": consumed_forbidden_dependency_rows,
    }
    formal_candidates = (
        feature_registry.loc[
            feature_registry["eligibility"] == "formal_candidate",
            ["feature_name", "physical_column", "block", "source_domain"],
        ]
        .sort_values(["block", "feature_name"])
        .to_dict("records")
    )
    readiness = {
        "schema": "seq100_quality_liquidity_training_readiness/"
        + study_id.rsplit("_", maxsplit=1)[-1],
        "study_id": study_id,
        "status": readiness_status,
        "common_support": {
            "frozen": True,
            "row_count": expected_row_count,
            "common_support_hash": scope_hash,
            "row_spine_years": list(RESEARCH_YEARS),
            "all_rows_have_complete_current_day_5m": True,
        },
        "pools": dict(scope_manifest.get("pools", {}) or {}),
        "daily_quality_row_spine": daily_quality_row_spines,
        "existing_atlas": {
            "frozen": True,
            "feature_count": expected_feature_count,
            "manifest_path": scope_manifest["atlas"]["manifest_path"],
            "manifest_sha256": scope_manifest["atlas"]["manifest_sha256"],
        },
        "source_query_contract": source_query_contract,
        "tushare_factor_qualification": {
            "provider_field_count": int(factor_quality["provider_field_count"]),
            "provider_schema_hash": factor_quality["provider_schema_hash"],
            "classification_counts": {
                str(key): int(value)
                for key, value in factor_registry["eligibility"]
                .value_counts()
                .sort_index()
                .items()
            },
            "raw_price_validation_passed": bool(
                factor_quality.get("raw_price_validation_passed")
            ),
            "hfq_validation_passed": bool(factor_quality.get("hfq_validation_passed")),
            "qfq_formal_candidate_count": 0,
        },
        "formal_candidate_fields": formal_candidates,
        "domain_status": {
            name: {
                "status": block.get("status"),
                "margin_secs_status": block.get("margin_secs_status"),
            }
            for name, block in blocks.items()
        },
        "annual_coverage": coverage.to_dict("records"),
        "coverage_findings": coverage_findings,
        "time_boundaries": {
            "burn_in_year": 2010,
            "burn_in_formal_row_count": 0,
            "research_start_date": RESEARCH_START,
            "research_end_date": END_DATE,
            "forbidden_2026_request_count": 0,
            "forbidden_2026_feature_row_count": (consumed_forbidden_dependency_rows),
            "source_support_explicit_future_date_row_count": int(
                source_support_boundary["explicit_future_date_row_count"]
            ),
            "source_support_future_metadata_consumed": False,
            "symbol_history_rows_excluded_after_end": int(
                history_mapping["excluded_after_end_row_count"]
            ),
        },
        "future_oos_folds": list(scope_manifest["rolling_oos_folds"]),
        "training_performed": False,
        "feature_set_selected": False,
    }
    readiness_path = output_root / "readiness.json"
    _write_json(readiness_path, readiness)
    manifest["readiness"] = {
        "path": str(readiness_path.resolve()),
        "sha256": _sha256(readiness_path),
        "status": readiness_status,
    }
    _write_json(output_root / "manifest.json", manifest)
    daily_quality_spine_total = sum(
        int(item["row_count"]) for item in daily_quality_row_spines.values()
    )
    state = {
        "study_id": study_id,
        "status": "completed",
        "source_scope_study_id": source_scope_study_id,
        "qdp_dataset_ids": dataset_ids,
        "source_scope_manifest_sha256": _sha256(scope_output_root / "manifest.json"),
        "manifest_path": str((output_root / "manifest.json").resolve()),
        "manifest_sha256": _sha256(output_root / "manifest.json"),
        "common_support_row_count": expected_row_count,
        "common_support_hash": scope_hash,
        "quality_liquidity_pit_row_count": daily_quality_spine_total,
        "training_performed": False,
        "feature_set_selected": False,
        "row_spine_contract_version": ROW_SPINE_CONTRACT_VERSION,
        "consumed_forbidden_dependency_row_count": (consumed_forbidden_dependency_rows),
        "readiness_status": readiness_status,
        "readiness_path": str(readiness_path.resolve()),
        "readiness_sha256": _sha256(readiness_path),
    }
    _write_json(output_root / "state.json", state)
    return status(output_root=output_root)


def status(*, output_root: Path = DEFAULT_OUTPUT_ROOT) -> dict[str, Any]:
    state_path = output_root / "state.json"
    state = _read_json(state_path) if state_path.is_file() else {}
    return {
        "study_id": state.get("study_id", STUDY_ID),
        "status": state.get("status", "pending"),
        "readiness_status": state.get("readiness_status", "pending"),
        "common_support_row_count": int(state.get("common_support_row_count", 0)),
        "common_support_hash": state.get("common_support_hash", ""),
        "training_performed": state.get("training_performed", False),
        "feature_set_selected": state.get("feature_set_selected", False),
        "row_spine_contract_version": int(state.get("row_spine_contract_version", 0)),
        "consumed_forbidden_dependency_row_count": int(
            state.get("consumed_forbidden_dependency_row_count", 0)
        ),
    }


def evaluate(*, output_root: Path = DEFAULT_OUTPUT_ROOT) -> dict[str, Any]:
    state = _read_json(output_root / "state.json")
    manifest = _read_json(output_root / "manifest.json")
    readiness = _read_json(output_root / "readiness.json")
    registry = pd.read_parquet(manifest["feature_registry"]["path"])
    coverage = pd.read_parquet(manifest["coverage_registry"]["path"])
    row_spines = dict(manifest.get("row_spine", {}) or {})
    daily_quality_row_spines = dict(
        manifest.get("daily_quality_row_spine", {}) or {}
    )
    blocks = dict(manifest.get("blocks", {}) or {})
    row_spine_contract = dict(manifest.get("row_spine_contract", {}) or {})
    spine_years = {int(year) for year in row_spines}
    spine_total = sum(int(item["row_count"]) for item in row_spines.values())
    daily_quality_spine_total = sum(
        int(item["row_count"]) for item in daily_quality_row_spines.values()
    )
    profile_hashes_valid = True
    block_alignment_valid = True
    row_spine_schema_valid = True
    feature_identity_schema_valid = True
    partition_dependency_counts_match = True
    actual_forbidden_dependency_rows = 0
    for year, spine in row_spines.items():
        spine_path = Path(str(spine["path"]))
        if not spine_path.is_file():
            row_spine_schema_valid = False
            partition_dependency_counts_match = False
            continue
        row_spine_schema_valid &= _parquet_columns(spine_path) == ROW_SPINE_COLUMNS
        spine_forbidden_count = _forbidden_date_row_count(spine_path)
        actual_forbidden_dependency_rows += spine_forbidden_count
        partition_dependency_counts_match &= spine_forbidden_count == int(
            spine.get("forbidden_dependency_row_count", -1)
        )
        profile_path = Path(str(spine["profile_path"]))
        profile_hashes_valid &= profile_path.is_file() and _sha256(profile_path) == str(
            spine["profile_sha256"]
        )
        for block in blocks.values():
            partitions = dict(block.get("partitions", {}) or {})
            if str(block.get("status")) != "completed" or year not in partitions:
                block_alignment_valid = False
                continue
            partition = dict(partitions[year])
            partition_path = Path(str(partition["path"]))
            if not partition_path.is_file():
                feature_identity_schema_valid = False
                partition_dependency_counts_match = False
                continue
            partition_columns = _parquet_columns(partition_path)
            feature_identity_schema_valid &= partition_columns[
                : len(ROW_SPINE_COLUMNS)
            ] == ROW_SPINE_COLUMNS and not set(partition_columns).intersection(
                FORBIDDEN_INHERITED_METADATA_COLUMNS
            )
            partition_forbidden_count = _forbidden_date_row_count(partition_path)
            actual_forbidden_dependency_rows += partition_forbidden_count
            partition_dependency_counts_match &= partition_forbidden_count == int(
                partition.get("forbidden_dependency_row_count", -1)
            )
            profile_path = Path(str(partition["profile_path"]))
            profile_hashes_valid &= profile_path.is_file() and _sha256(
                profile_path
            ) == str(partition["profile_sha256"])
            block_alignment_valid &= int(partition["row_count"]) == int(
                spine["row_count"]
            ) and str(partition["row_key_hash"]) == str(spine["row_key_hash"])
    daily_quality_spine_valid = True
    for spine in daily_quality_row_spines.values():
        spine_path = Path(str(spine["path"]))
        if not spine_path.is_file():
            daily_quality_spine_valid = False
            continue
        daily_quality_spine_valid &= _parquet_columns(spine_path) == ROW_SPINE_COLUMNS
        forbidden_count = _forbidden_date_row_count(spine_path)
        actual_forbidden_dependency_rows += forbidden_count
        daily_quality_spine_valid &= forbidden_count == int(
            spine.get("forbidden_dependency_row_count", -1)
        )
        profile_path = Path(str(spine["profile_path"]))
        profile_hashes_valid &= profile_path.is_file() and _sha256(
            profile_path
        ) == str(spine["profile_sha256"])
    fold_counts = {
        int(item["evaluation_year"]): int(
            item["training_candidate_row_count_before_target_purge"]
        )
        for item in list(manifest.get("rolling_oos_folds", []) or [])
    }
    allowed_coverage_states = {
        "observed_count",
        "source_unavailable_count",
        "not_applicable_count",
        "warmup_missing_count",
    }
    expected_source_query_contract = {
        DataDomain.STK_FACTOR_PRO_RAW: "trade_date",
        DataDomain.MARGIN_MARKET: "year",
        DataDomain.MARGIN_DETAIL: "trade_date",
        DataDomain.MONEYFLOW_RAW: "trade_date",
    }
    source_support_boundary = dict(
        row_spine_contract.get("source_support_boundary", {}) or {}
    )
    history_mapping = dict(row_spine_contract.get("symbol_history_mapping", {}) or {})
    readiness_boundaries = dict(readiness.get("time_boundaries", {}) or {})
    config = dict(manifest.get("config", {}) or {})
    source_config = dict(config.get("source_scope", {}) or {})
    expected_row_count = int(
        source_config.get(
            "expected_common_support_rows",
            EXPECTED_COMMON_SUPPORT_ROW_COUNT,
        )
    )
    expected_existing_feature_count = int(
        source_config.get(
            "expected_existing_feature_count",
            EXPECTED_EXISTING_FEATURE_COUNT,
        )
    )
    expected_training_rows = {
        int(year): int(count)
        for year, count in dict(
            config.get(
                "expected_rolling_training_rows",
                EXPECTED_OOS_TRAINING_ROW_COUNTS,
            )
            or {}
        ).items()
    }
    expected_formal_candidate_count = config.get(
        "expected_existing_formal_candidate_count"
    )
    expected_diagnostic_count = config.get("expected_existing_diagnostic_only_count")
    expected_daily_row_count = config.get("expected_daily_support_rows")
    pools = dict(manifest.get("pools", {}) or {})
    checks = {
        "completed": state.get("status") == "completed"
        and manifest.get("status") == "completed",
        "row_count_exact": int(manifest["common_support"]["row_count"])
        == expected_row_count,
        "scope_hash_present": bool(manifest["common_support"]["common_support_hash"]),
        "source_query_contract_exact": manifest.get("source_query_contract")
        == expected_source_query_contract,
        "row_spine_years_exact": spine_years == set(RESEARCH_YEARS),
        "row_spine_total_exact": spine_total == expected_row_count,
        "row_spine_contract_exact": int(row_spine_contract.get("version", 0))
        == ROW_SPINE_CONTRACT_VERSION
        and tuple(row_spine_contract.get("projection_columns", ())) == ROW_SPINE_COLUMNS
        and tuple(row_spine_contract.get("forbidden_inherited_metadata_columns", ()))
        == FORBIDDEN_INHERITED_METADATA_COLUMNS,
        "row_spine_schema_exact": row_spine_schema_valid,
        "feature_identity_schema_exact": feature_identity_schema_valid,
        "source_support_projection_safe": tuple(
            source_support_boundary.get("projection_columns", ())
        )
        == SUPPORT_IDENTITY_COLUMNS
        and not set(source_support_boundary.get("projection_columns", ())).intersection(
            FORBIDDEN_INHERITED_METADATA_COLUMNS
        ),
        "symbol_history_cutoff_exact": history_mapping.get("cutoff_date") == END_DATE
        and int(history_mapping.get("invalid_effective_from_row_count", -1)) == 0
        and int(history_mapping.get("ambiguous_symbol_count", -1)) == 0,
        "block_row_keys_exact": block_alignment_valid,
        "partition_profiles_valid": profile_hashes_valid,
        "partition_dependency_counts_exact": partition_dependency_counts_match,
        "period_exact": manifest["research_period"]["start_date"] == RESEARCH_START
        and manifest["research_period"]["end_date"] == END_DATE,
        "oos_training_rows_exact": fold_counts == expected_training_rows,
        "burn_in_not_formal": manifest["data_history"]["burn_in_eligible_for_training"]
        is False
        and manifest["data_history"]["burn_in_eligible_for_labels"] is False,
        "qfq_not_formal": int(
            registry.loc[
                registry["source_field"]
                .astype(str)
                .str.contains("qfq", case=False, na=False),
                "eligibility",
            ]
            .eq("formal_candidate")
            .sum()
        )
        == 0,
        "factor_fields_classified": int(
            (registry["source_domain"] == DataDomain.STK_FACTOR_PRO_RAW).sum()
        )
        == EXPECTED_FACTOR_FIELD_COUNT,
        "existing_atlas_fields_exact": int(
            (registry["block"] == "existing_atlas_518").sum()
        )
        == expected_existing_feature_count,
        "dual_pool_contract": (
            not bool(config.get("require_dual_pool_manifest", False))
            or (
                set(pools)
                == {
                    "quality_liquidity_pit",
                    "quality_liquidity_complete_pit",
                }
                and int(
                    pools["quality_liquidity_complete_pit"].get("row_count", 0)
                )
                == expected_row_count
                and daily_quality_spine_valid
                and {int(year) for year in daily_quality_row_spines}
                == set(RESEARCH_YEARS)
                and (
                    expected_daily_row_count is None
                    or daily_quality_spine_total == int(expected_daily_row_count)
                )
            )
        ),
        "legacy_candidate_classification_preserved": (
            expected_formal_candidate_count is None
            or int((registry["eligibility"] == "formal_candidate").sum())
            == int(expected_formal_candidate_count)
        )
        and (
            expected_diagnostic_count is None
            or int((registry["eligibility"] == "diagnostic_only").sum())
            == int(expected_diagnostic_count)
        ),
        "availability_metadata_separate": (
            not bool(config.get("require_dual_pool_manifest", False))
            or {
                "availability_gated",
                "availability_metadata",
            }.issubset(set(registry["eligibility"].astype(str)))
        ),
        "physical_columns_registered": bool(registry["physical_column"].notna().all()),
        "coverage_registry_contract": allowed_coverage_states.issubset(coverage.columns)
        and set(coverage["year"].astype(int)) == set(RESEARCH_YEARS),
        "readiness_file_matches": manifest["readiness"]["sha256"]
        == _sha256(output_root / "readiness.json")
        and readiness.get("status") == manifest.get("readiness_status"),
        "no_2026": actual_forbidden_dependency_rows == 0
        and int(manifest.get("forbidden_2026_rows", -1)) == 0
        and int(row_spine_contract.get("consumed_forbidden_dependency_row_count", -1))
        == 0
        and int(state.get("consumed_forbidden_dependency_row_count", -1)) == 0
        and int(readiness_boundaries.get("forbidden_2026_feature_row_count", -1)) == 0
        and readiness_boundaries.get("source_support_future_metadata_consumed")
        is False,
        "training_false": state.get("training_performed") is False
        and manifest.get("training_performed") is False,
        "feature_selection_false": state.get("feature_set_selected") is False
        and manifest.get("feature_set_selected") is False,
    }
    if not all(checks.values()):
        raise TrainingReadyError(f"training_ready_evaluation_failed:{checks}")
    return {
        "status": "ok",
        "study_id": state.get("study_id", STUDY_ID),
        "checks": checks,
        "readiness_status": manifest.get("readiness_status"),
        "formal_candidate_count": int(
            (registry["eligibility"] == "formal_candidate").sum()
        ),
        "existing_feature_count": expected_existing_feature_count,
        "consumed_forbidden_dependency_row_count": (actual_forbidden_dependency_rows),
        "training_performed": False,
    }


def self_test() -> dict[str, Any]:
    if RESEARCH_YEARS != tuple(range(2011, 2026)):
        raise AssertionError("research years changed")
    if FORBIDDEN_YEAR in RESEARCH_YEARS or DATA_HISTORY_START != "2010-01-01":
        raise AssertionError("date boundary changed")
    cases = {
        "close_qfq": "diagnostic_only",
        "rsi_hfq_6": "formal_candidate",
        "macd_bfq": "formal_candidate",
        "close": "redundant_reference",
        "ts_code": "rejected",
    }
    for field, expected in cases.items():
        actual = classify_factor_field(
            field,
            raw_price_validation_passed=True,
            hfq_validation_passed=True,
        )[0]
        if actual != expected:
            raise AssertionError(f"factor_eligibility_changed:{field}:{actual}")
    return {
        "status": "ok",
        "checks": {
            "research_start": RESEARCH_START,
            "end_date": END_DATE,
            "burn_in_only": [2010],
            "existing_feature_count": EXPECTED_EXISTING_FEATURE_COUNT,
            "factor_field_count": EXPECTED_FACTOR_FIELD_COUNT,
            "qfq_formal": False,
            "training_performed": False,
            "forbidden_2026": True,
            "row_spine_contract_version": ROW_SPINE_CONTRACT_VERSION,
        },
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="seq100-quality-liquidity-training-ready")
    parser.add_argument("--study-path", type=Path, default=DEFAULT_STUDY_PATH)
    parser.add_argument(
        "--scope-output-root", type=Path, default=DEFAULT_SCOPE_OUTPUT_ROOT
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--workspace-root", type=Path, default=WORKSPACE_ROOT)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--status", action="store_true")
    mode.add_argument("--prepare", action="store_true")
    mode.add_argument("--run-pending", action="store_true")
    mode.add_argument("--evaluate", action="store_true")
    mode.add_argument("--self-test", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if args.status:
        payload = status(output_root=args.output_root)
    elif args.prepare or args.run_pending:
        payload = prepare(
            study_path=args.study_path,
            scope_output_root=args.scope_output_root,
            output_root=args.output_root,
            workspace_root=args.workspace_root,
        )
    elif args.evaluate:
        payload = evaluate(output_root=args.output_root)
    else:
        payload = self_test()
    print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
