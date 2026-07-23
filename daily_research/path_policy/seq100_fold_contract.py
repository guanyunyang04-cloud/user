from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
DEVELOPMENT_CANDIDATE_COLUMNS = {
    "candidate_id",
    "split",
    "year",
    "trade_date",
    "date_idx",
    "symbol_idx",
    "symbol",
    "entry_trade_date",
    "entry_filled",
    "label_valid",
    "price_label_valid",
    "va_aux_valid",
}


def _workspace_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else WORKSPACE_ROOT / path


def _sha256_file(value: str | Path) -> str:
    digest = hashlib.sha256()
    with _workspace_path(value).open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _source_backing_file_inventory(manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    paths: set[Path] = set()

    def collect(value: Any) -> None:
        if isinstance(value, Mapping):
            raw_path = value.get("path")
            if isinstance(raw_path, str) and raw_path.strip():
                paths.add(_workspace_path(raw_path).resolve())
            for nested in value.values():
                collect(nested)
        elif isinstance(value, list):
            for nested in value:
                collect(nested)

    for section in ("feature_channels", "label_arrays", "execution_arrays", "masks"):
        collect(dict(manifest.get(section, {}) or {}))
    rows: list[dict[str, Any]] = []
    for path in sorted(paths, key=lambda item: str(item).lower()):
        if not path.is_file():
            raise FileNotFoundError(path)
        stat = path.stat()
        rows.append({"path": str(path), "size": int(stat.st_size), "mtime_ns": int(stat.st_mtime_ns)})
    return rows


def validate_source_view_provenance(manifest: Mapping[str, Any]) -> dict[str, Any]:
    stored = dict(manifest.get("source_view_provenance", {}) or {})
    required = {
        "schema_version",
        "manifest_path",
        "manifest_sha256",
        "sample_index_path",
        "sample_index_sha256",
        "artifact_type",
        "backing_files",
        "backing_files_sha256",
    }
    missing = sorted(required.difference(stored))
    if missing:
        raise ValueError(f"fold source_view_provenance is missing fields: {missing}")
    source_path = _workspace_path(str(stored["manifest_path"])).resolve()
    sample_path = _workspace_path(str(stored["sample_index_path"])).resolve()
    if not source_path.is_file() or _sha256_file(source_path) != str(stored["manifest_sha256"]):
        raise ValueError("fold source manifest is missing or its SHA-256 changed")
    if not sample_path.is_file() or _sha256_file(sample_path) != str(stored["sample_index_sha256"]):
        raise ValueError("fold source sample index is missing or its SHA-256 changed")
    candidate_value = str(stored.get("candidate_index_path", "") or "")
    candidate_path = _workspace_path(candidate_value).resolve() if candidate_value else None
    if candidate_path is not None and (
        not candidate_path.is_file()
        or _sha256_file(candidate_path) != str(stored.get("candidate_index_sha256", ""))
    ):
        raise ValueError("fold source candidate index is missing or its SHA-256 changed")
    current = json.loads(source_path.read_text(encoding="utf-8"))
    if str(current.get("artifact_type", "")) != str(stored["artifact_type"]):
        raise ValueError("fold source artifact type changed")
    if _workspace_path(str(current.get("sample_index_path", "") or "")).resolve() != sample_path:
        raise ValueError("fold source sample-index path changed")
    current_candidate = str(current.get("candidate_index_path", "") or "")
    if bool(candidate_value) != bool(current_candidate):
        raise ValueError("fold source candidate-index declaration changed")
    if candidate_path is not None and _workspace_path(current_candidate).resolve() != candidate_path:
        raise ValueError("fold source candidate-index path changed")
    inventory = _source_backing_file_inventory(current)
    if inventory != list(stored.get("backing_files", []) or []):
        raise ValueError("fold source backing-file identity changed")
    if _canonical_sha256(inventory) != str(stored["backing_files_sha256"]):
        raise ValueError("fold source backing-file inventory digest changed")
    return stored


def compute_fold_training_contract(
    manifest: Mapping[str, Any],
    *,
    sample_frame: pd.DataFrame | None = None,
) -> dict[str, Any]:
    sample_path = _workspace_path(str(manifest.get("sample_index_path", "") or ""))
    if not sample_path.is_file():
        raise FileNotFoundError(f"missing fold sample index: {sample_path}")
    frame = pd.read_parquet(sample_path) if sample_frame is None else sample_frame.copy()
    required = {"split", "trade_date", "date_idx", "symbol_idx"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"fold sample index missing training-contract columns: {missing}")
    if frame.empty or bool(frame.duplicated(["date_idx", "symbol_idx"]).any()):
        raise ValueError("fold sample index must be non-empty with unique date/symbol rows")
    frame["split"] = frame["split"].astype(str)
    if set(frame["split"].unique()) != {"train", "oos"}:
        raise ValueError("fold training contract requires train/oos splits")
    train = frame[frame["split"].eq("train")]
    oos = frame[frame["split"].eq("oos")]
    lookback_days = int(manifest.get("lookback_days", 0) or 0)
    forward_days = int(manifest.get("forward_days", 0) or 0)
    if train.empty or oos.empty or lookback_days <= 0 or forward_days <= 0:
        raise ValueError("fold training contract has an empty split or invalid horizon")
    oos_start_idx = int(oos["date_idx"].astype(int).min())
    overlap = int(((train["date_idx"].astype(int) + forward_days) >= oos_start_idx).sum())
    if overlap:
        raise ValueError(f"fold training labels overlap OOS: {overlap}")
    oos_years = sorted({int(str(value)[:4]) for value in oos["trade_date"].astype(str)})
    purge = dict(manifest.get("purged_walkforward", {}) or {})
    if len(oos_years) != 1 or not purge:
        raise ValueError("fold requires one OOS year and purged_walkforward metadata")
    if int(purge.get("oos_year", 0) or 0) != oos_years[0]:
        raise ValueError("fold purge OOS year does not match sample index")
    if int(purge.get("oos_start_date_idx", -1)) != oos_start_idx:
        raise ValueError("fold purge OOS start index does not match sample index")
    if str(purge.get("purge_rule", "")) != "date_idx + forward_days < oos_start_date_idx":
        raise ValueError("fold purge rule is not the registered strict boundary")
    if int(purge.get("label_overlap_count", -1)) != 0:
        raise ValueError("fold purge metadata reports label overlap")
    normalization = dict(manifest.get("normalization", {}) or {})
    if normalization.get("fit_scope") != "feature_dates_before_oos_start":
        raise ValueError("fold normalization fit scope is not pre-OOS")
    if str(normalization.get("fit_date_end_exclusive", "")) != str(purge.get("oos_start", "")):
        raise ValueError("fold normalization cutoff does not match OOS start")
    if int(normalization.get("oos_feature_date_count", -1) or 0) != 0:
        raise ValueError("fold normalization includes OOS feature dates")
    for name, meta in dict(manifest.get("feature_channels", {}) or {}).items():
        width = len(list(dict(meta).get("columns", []) or []))
        stats = dict(normalization.get(name, {}) or {})
        if width and (
            len(list(stats.get("mean", []) or [])) != width
            or len(list(stats.get("std", []) or [])) != width
        ):
            raise ValueError(f"fold normalization width mismatch for {name}")
    sample_sha = _sha256_file(sample_path)
    purge_keys = (
        "schema_version", "method", "split_roles", "train_start_year", "oos_year",
        "oos_start", "oos_start_trade_date", "oos_end", "oos_start_date_idx",
        "safe_train_signal_end", "safe_train_signal_end_trade_date", "max_train_label_end",
        "max_train_label_end_trade_date", "purge_rule", "purged_row_count",
        "purged_signal_date_count", "purged_signal_start", "purged_signal_end",
        "label_overlap_count", "normalization_cutoff_exclusive",
        "normalization_last_feature_date", "normalization_includes_purge_period_features",
    )
    payload: dict[str, Any] = {
        "schema_version": 1,
        "artifact_type": "seq100_purged_fold_training_contract",
        "lookback_days": lookback_days,
        "forward_days": forward_days,
        "sample_index": {
            "sha256": sample_sha,
            "row_count": int(len(frame)),
            "columns": [str(column) for column in frame.columns],
            "dtypes": {str(column): str(dtype) for column, dtype in frame.dtypes.items()},
        },
        "split": {
            "roles": {"fit": "train", "evaluation": "oos"},
            "counts": {str(key): int(value) for key, value in frame["split"].value_counts().sort_index().items()},
            "train_date_idx_min": int(train["date_idx"].astype(int).min()),
            "train_date_idx_max": int(train["date_idx"].astype(int).max()),
            "oos_date_idx_min": oos_start_idx,
            "oos_date_idx_max": int(oos["date_idx"].astype(int).max()),
            "train_years": sorted({int(str(value)[:4]) for value in train["trade_date"].astype(str)}),
            "oos_years": oos_years,
        },
        "purge": {key: purge.get(key) for key in purge_keys},
        "normalization": normalization,
        "feature_channels": dict(manifest.get("feature_channels", {}) or {}),
        "label_arrays": dict(manifest.get("label_arrays", {}) or {}),
        "label_semantics": dict(manifest.get("label_semantics", {}) or {}),
        "masks": dict(manifest.get("masks", {}) or {}),
        "scope": dict(manifest.get("scope", {}) or {}),
        "source_view_provenance": dict(manifest.get("source_view_provenance", {}) or {}),
        "date_values_sha256": _canonical_sha256(list(manifest.get("date_values", []) or [])),
        "symbol_values_sha256": _canonical_sha256(list(manifest.get("symbol_values", []) or [])),
    }
    if manifest.get("relative_turnover_supplement"):
        payload["relative_turnover_supplement"] = dict(manifest["relative_turnover_supplement"])
    if manifest.get("legal_exit_contract"):
        payload["legal_exit_contract"] = dict(manifest["legal_exit_contract"])
    return {
        "schema_version": 1,
        "algorithm": "sha256",
        "sha256": _canonical_sha256(payload),
        "sample_index_sha256": sample_sha,
        "normalization_sha256": _canonical_sha256(normalization),
        "payload": payload,
    }


def validate_fold_training_contract(manifest: Mapping[str, Any]) -> dict[str, Any]:
    stored = dict(manifest.get("fold_training_contract", {}) or {})
    if not stored or stored != compute_fold_training_contract(manifest):
        raise ValueError("fold_training_contract does not match current training material")
    return stored


def compute_development_fold_training_contract(
    manifest: Mapping[str, Any],
    *,
    sample_frame: pd.DataFrame | None = None,
    candidate_frame: pd.DataFrame | None = None,
) -> dict[str, Any]:
    sample_path = _workspace_path(str(manifest.get("sample_index_path", "") or ""))
    candidate_path = _workspace_path(str(manifest.get("candidate_index_path", "") or ""))
    if not sample_path.is_file() or not candidate_path.is_file():
        raise FileNotFoundError("development sample or candidate index is missing")
    contract_schema = int(
        dict(manifest.get("development_walkforward", {}) or {}).get("schema_version", 1)
        or 1
    )
    if contract_schema >= 2 and sample_frame is None and candidate_frame is None:
        return _compute_development_fold_training_contract_v2(
            manifest,
            sample_path=sample_path,
            candidate_path=candidate_path,
        )
    samples = pd.read_parquet(sample_path) if sample_frame is None else sample_frame.copy()
    candidates = pd.read_parquet(candidate_path) if candidate_frame is None else candidate_frame.copy()
    sample_missing = sorted({"split", "trade_date", "date_idx", "symbol_idx", "symbol"}.difference(samples.columns))
    candidate_missing = sorted(DEVELOPMENT_CANDIDATE_COLUMNS.difference(candidates.columns))
    if sample_missing or candidate_missing:
        raise ValueError(f"development index columns missing: {sample_missing or candidate_missing}")
    if samples.empty or candidates.empty:
        raise ValueError("development indexes must be non-empty")
    if bool(samples.duplicated(["date_idx", "symbol_idx"]).any()) or bool(
        candidates.duplicated(["date_idx", "symbol_idx"]).any()
    ):
        raise ValueError("development indexes require unique date/symbol rows")
    if bool(candidates["candidate_id"].duplicated().any()):
        raise ValueError("development candidate_id values must be unique")
    samples["split"] = samples["split"].astype(str)
    candidates["split"] = candidates["split"].astype(str)
    if set(samples["split"].unique()) != {"train", "development"}:
        raise ValueError("development supervised index requires train/development splits")
    if set(candidates["split"].unique()) != {"development"}:
        raise ValueError("development candidate index must contain only development rows")
    train = samples[samples["split"].eq("train")]
    development = samples[samples["split"].eq("development")]
    contract = dict(manifest.get("development_walkforward", {}) or {})
    research_contract = dict(manifest.get("research_contract", {}) or {})
    if train.empty or development.empty or not contract or not research_contract:
        raise ValueError("development fold is incomplete")
    if dict(manifest.get("development_contract", {}) or {}) != research_contract:
        raise ValueError("fold development contract alias changed")
    for field in ("contract_id", "contract_sha256", "contract_file_sha256"):
        if not str(research_contract.get(field, "") or ""):
            raise ValueError(f"fold research contract is missing {field}")
    year = int(contract.get("development_year", 0) or 0)
    if sorted({int(str(value)[:4]) for value in development["trade_date"].astype(str)}) != [year]:
        raise ValueError("development year does not match supervised index")
    if sorted({int(value) for value in candidates["year"].astype(int)}) != [year]:
        raise ValueError("development year does not match candidate index")
    start_idx = int(candidates["date_idx"].astype(int).min())
    contract_schema = int(contract.get("schema_version", 1) or 1)
    dependency_days = int(
        contract.get(
            "training_label_dependency_days",
            contract.get("max_label_dependency_days", 0),
        )
        or 0
    )
    forward_days = int(manifest.get("forward_days", 0) or 0)
    execution_tail_days = int(manifest.get("execution_tail_days", 0) or 0)
    if contract_schema >= 2:
        if dependency_days != forward_days:
            raise ValueError("development training purge must equal forward_days")
        if int(contract.get("execution_dependency_days", 0) or 0) != (
            forward_days + execution_tail_days
        ):
            raise ValueError("development execution dependency metadata is inconsistent")
    overlap = int(((train["date_idx"].astype(int) + dependency_days) >= start_idx).sum())
    if dependency_days <= 0 or overlap:
        raise ValueError(f"training label dependencies overlap development: {overlap}")
    expected_purge_rule = (
        "training_label_end_date_idx < development_start_date_idx"
        if contract_schema >= 2
        else "max_label_dependency_date_idx < development_start_date_idx"
    )
    if str(contract.get("purge_rule", "")) != expected_purge_rule:
        raise ValueError("development purge rule is not the registered strict boundary")
    if int(contract.get("label_dependency_overlap_count", -1)) != 0:
        raise ValueError("development metadata reports label dependency overlap")
    candidate_keys = candidates.set_index(["date_idx", "symbol_idx"], drop=False)
    supervised_keys = development.set_index(["date_idx", "symbol_idx"], drop=False)
    if len(supervised_keys.index.difference(candidate_keys.index)):
        raise ValueError("supervised development rows are missing from the full candidate index")
    aligned = candidate_keys.loc[supervised_keys.index]
    if not bool(np.equal(supervised_keys["trade_date"].astype(str), aligned["trade_date"].astype(str)).all()):
        raise ValueError("supervised/candidate trade dates differ")
    if not bool(np.equal(supervised_keys["symbol"].astype(str), aligned["symbol"].astype(str)).all()):
        raise ValueError("supervised/candidate symbols differ")
    normalization = dict(manifest.get("normalization", {}) or {})
    if normalization.get("fit_scope") != "feature_dates_before_development_start":
        raise ValueError("development normalization fit scope is not pre-development")
    if str(normalization.get("fit_date_end_exclusive", "")) != str(contract.get("development_start", "")):
        raise ValueError("development normalization cutoff does not match development start")
    if int(normalization.get("development_feature_date_count", -1) or 0) != 0:
        raise ValueError("development normalization includes development feature dates")
    for name, meta in dict(manifest.get("feature_channels", {}) or {}).items():
        width = len(list(dict(meta).get("columns", []) or []))
        stats = dict(normalization.get(name, {}) or {})
        if width and (
            len(list(stats.get("mean", []) or [])) != width
            or len(list(stats.get("std", []) or [])) != width
        ):
            raise ValueError(f"development normalization width mismatch for {name}")
    sample_sha = _sha256_file(sample_path)
    candidate_sha = _sha256_file(candidate_path)
    contract_keys = (
        "schema_version", "method", "split_roles", "train_start_year", "development_year",
        "development_start", "development_end", "source_padding_end",
        "source_padding_trade_date_count", "development_start_date_idx", "forward_days",
        "execution_tail_days", "max_label_dependency_days", "safe_train_signal_end",
        "max_train_dependency_end", "purge_rule", "purged_row_count",
        "purged_signal_date_count", "label_dependency_overlap_count", "candidate_universe_rule",
        "candidate_count", "supervised_development_count", "unsupervised_candidate_count",
        "normalization_cutoff_exclusive",
    )
    if contract_schema >= 2:
        contract_keys = (
            *contract_keys,
            "training_label_dependency_days",
            "execution_dependency_days",
        )
    payload = {
        "schema_version": 2 if contract_schema >= 2 else 1,
        "artifact_type": "seq100_development_fold_training_contract",
        "lookback_days": int(manifest.get("lookback_days", 0) or 0),
        "forward_days": int(manifest.get("forward_days", 0) or 0),
        "max_label_dependency_days": dependency_days,
        "supervised_sample_index": {
            "sha256": sample_sha,
            "row_count": int(len(samples)),
            "columns": [str(column) for column in samples.columns],
            "dtypes": {str(column): str(dtype) for column, dtype in samples.dtypes.items()},
        },
        "candidate_index": {
            "sha256": candidate_sha,
            "row_count": int(len(candidates)),
            "columns": [str(column) for column in candidates.columns],
            "dtypes": {str(column): str(dtype) for column, dtype in candidates.dtypes.items()},
            "full_signal_day_universe": True,
            "future_label_required_for_membership": False,
            "entry_fill_required_for_membership": False,
        },
        "split": {
            "roles": {"fit": "train", "evaluation": "development"},
            "supervised_counts": {str(key): int(value) for key, value in samples["split"].value_counts().sort_index().items()},
            "candidate_counts": {"development": int(len(candidates))},
            "train_date_idx_min": int(train["date_idx"].astype(int).min()),
            "train_date_idx_max": int(train["date_idx"].astype(int).max()),
            "development_date_idx_min": start_idx,
            "development_date_idx_max": int(candidates["date_idx"].astype(int).max()),
            "development_year": year,
        },
        "development_walkforward": {key: contract.get(key) for key in contract_keys},
        "normalization": normalization,
        "feature_channels": dict(manifest.get("feature_channels", {}) or {}),
        "label_arrays": dict(manifest.get("label_arrays", {}) or {}),
        "label_semantics": dict(manifest.get("label_semantics", {}) or {}),
        "execution_contract": dict(manifest.get("execution_contract", {}) or {}),
        "research_contract": research_contract,
        "development_contract": dict(manifest.get("development_contract", {}) or {}),
        "masks": dict(manifest.get("masks", {}) or {}),
        "scope": dict(manifest.get("scope", {}) or {}),
        "source_view_provenance": dict(manifest.get("source_view_provenance", {}) or {}),
        "date_values_sha256": _canonical_sha256(list(manifest.get("date_values", []) or [])),
        "symbol_values_sha256": _canonical_sha256(list(manifest.get("symbol_values", []) or [])),
    }
    if contract_schema >= 2:
        payload["training_label_dependency_days"] = dependency_days
        payload["execution_dependency_days"] = int(contract["execution_dependency_days"])
    return {
        "schema_version": 2 if contract_schema >= 2 else 1,
        "algorithm": "sha256",
        "sha256": _canonical_sha256(payload),
        "sample_index_sha256": sample_sha,
        "candidate_index_sha256": candidate_sha,
        "normalization_sha256": _canonical_sha256(normalization),
        "payload": payload,
    }


def _parquet_contract_meta(path: Path) -> dict[str, Any]:
    parquet = pq.ParquetFile(path)
    schema = parquet.schema_arrow
    return {
        "row_count": int(parquet.metadata.num_rows),
        "columns": list(schema.names),
        "dtypes": {field.name: str(field.type) for field in schema},
    }


def _duckdb_parquet_sql(path: Path) -> str:
    return "'" + str(path.resolve()).replace("'", "''") + "'"


def _compute_development_fold_training_contract_v2(
    manifest: Mapping[str, Any],
    *,
    sample_path: Path,
    candidate_path: Path,
) -> dict[str, Any]:
    import duckdb  # type: ignore

    contract = dict(manifest.get("development_walkforward", {}) or {})
    research_contract = dict(manifest.get("research_contract", {}) or {})
    if dict(manifest.get("development_contract", {}) or {}) != research_contract:
        raise ValueError("fold development contract alias changed")
    for field in ("contract_id", "contract_sha256", "contract_file_sha256"):
        if not str(research_contract.get(field, "") or ""):
            raise ValueError(f"fold research contract is missing {field}")
    forward_days = int(manifest.get("forward_days", 0) or 0)
    execution_tail_days = int(manifest.get("execution_tail_days", 0) or 0)
    dependency_days = int(contract.get("training_label_dependency_days", 0) or 0)
    execution_dependency_days = int(contract.get("execution_dependency_days", 0) or 0)
    if dependency_days != forward_days or dependency_days <= 0:
        raise ValueError("development training purge must equal forward_days")
    if execution_dependency_days != forward_days + execution_tail_days:
        raise ValueError("development execution dependency metadata is inconsistent")
    if str(contract.get("purge_rule", "")) != (
        "training_label_end_date_idx < development_start_date_idx"
    ):
        raise ValueError("development purge rule is not the registered strict boundary")
    if int(contract.get("label_dependency_overlap_count", -1)) != 0:
        raise ValueError("development metadata reports label dependency overlap")

    sample_meta = _parquet_contract_meta(sample_path)
    candidate_meta = _parquet_contract_meta(candidate_path)
    sample_missing = sorted(
        {"split", "trade_date", "date_idx", "symbol_idx", "symbol"}.difference(
            sample_meta["columns"]
        )
    )
    candidate_missing = sorted(
        DEVELOPMENT_CANDIDATE_COLUMNS.difference(candidate_meta["columns"])
    )
    if sample_missing or candidate_missing:
        raise ValueError(f"development index columns missing: {sample_missing or candidate_missing}")
    sample_sql = _duckdb_parquet_sql(sample_path)
    candidate_sql = _duckdb_parquet_sql(candidate_path)
    con = duckdb.connect(":memory:")
    try:
        con.execute("set threads=2")
        con.execute("set memory_limit='1GB'")
        sample_splits = {
            str(key): int(value)
            for key, value in con.execute(
                f"select cast(split as varchar), count(*) from read_parquet({sample_sql}) group by 1"
            ).fetchall()
        }
        candidate_splits = {
            str(key): int(value)
            for key, value in con.execute(
                f"select cast(split as varchar), count(*) from read_parquet({candidate_sql}) group by 1"
            ).fetchall()
        }
        if set(sample_splits) != {"train", "development"}:
            raise ValueError("development supervised index requires train/development splits")
        if set(candidate_splits) != {"development"}:
            raise ValueError("development candidate index must contain only development rows")
        sample_duplicates = int(
            con.execute(
                f"select count(*) - count(distinct (date_idx, symbol_idx)) from read_parquet({sample_sql})"
            ).fetchone()[0]
        )
        candidate_duplicates, candidate_id_duplicates = con.execute(
            f"""
            select
              count(*) - count(distinct (date_idx, symbol_idx)),
              count(*) - count(distinct candidate_id)
            from read_parquet({candidate_sql})
            """
        ).fetchone()
        if int(sample_duplicates) or int(candidate_duplicates) or int(candidate_id_duplicates):
            raise ValueError("development indexes require unique date/symbol and candidate_id rows")
        train_stats = con.execute(
            f"""
            select min(date_idx), max(date_idx)
            from read_parquet({sample_sql})
            where cast(split as varchar) = 'train'
            """
        ).fetchone()
        development_stats = con.execute(
            f"""
            select min(date_idx), max(date_idx),
                   min(cast(substr(cast(trade_date as varchar), 1, 4) as integer)),
                   max(cast(substr(cast(trade_date as varchar), 1, 4) as integer))
            from read_parquet({sample_sql})
            where cast(split as varchar) = 'development'
            """
        ).fetchone()
        candidate_stats = con.execute(
            f"""
            select min(date_idx), max(date_idx), min(cast(year as integer)), max(cast(year as integer))
            from read_parquet({candidate_sql})
            """
        ).fetchone()
        start_idx = int(candidate_stats[0])
        year = int(contract.get("development_year", 0) or 0)
        if int(development_stats[2]) != year or int(development_stats[3]) != year:
            raise ValueError("development year does not match supervised index")
        if int(candidate_stats[2]) != year or int(candidate_stats[3]) != year:
            raise ValueError("development year does not match candidate index")
        if start_idx != int(contract.get("development_start_date_idx", -1)):
            raise ValueError("development start index does not match candidate index")
        overlap = int(
            con.execute(
                f"""
                select count(*) from read_parquet({sample_sql})
                where cast(split as varchar) = 'train'
                  and cast(date_idx as bigint) + {dependency_days} >= {start_idx}
                """
            ).fetchone()[0]
        )
        if overlap:
            raise ValueError(f"training label dependencies overlap development: {overlap}")
        missing_candidates, mismatches = con.execute(
            f"""
            select
              count(*) filter (where c.candidate_id is null),
              count(*) filter (
                where c.candidate_id is not null
                  and (cast(s.trade_date as varchar) <> cast(c.trade_date as varchar)
                       or cast(s.symbol as varchar) <> cast(c.symbol as varchar))
              )
            from read_parquet({sample_sql}) s
            left join read_parquet({candidate_sql}) c
              on s.date_idx = c.date_idx and s.symbol_idx = c.symbol_idx
            where cast(s.split as varchar) = 'development'
            """
        ).fetchone()
        if int(missing_candidates):
            raise ValueError("supervised development rows are missing from the full candidate index")
        if int(mismatches):
            raise ValueError("supervised/candidate trade dates or symbols differ")
        train_years = [
            int(row[0])
            for row in con.execute(
                f"""
                select distinct cast(substr(cast(trade_date as varchar), 1, 4) as integer) as year
                from read_parquet({sample_sql})
                where cast(split as varchar) = 'train'
                order by year
                """
            ).fetchall()
        ]
    finally:
        con.close()

    normalization = dict(manifest.get("normalization", {}) or {})
    if normalization.get("fit_scope") != "feature_dates_before_development_start":
        raise ValueError("development normalization fit scope is not pre-development")
    if str(normalization.get("fit_date_end_exclusive", "")) != str(
        contract.get("development_start", "")
    ):
        raise ValueError("development normalization cutoff does not match development start")
    if int(normalization.get("development_feature_date_count", -1) or 0) != 0:
        raise ValueError("development normalization includes development feature dates")
    for name, meta in dict(manifest.get("feature_channels", {}) or {}).items():
        width = len(list(dict(meta).get("columns", []) or []))
        stats = dict(normalization.get(name, {}) or {})
        if width and (
            len(list(stats.get("mean", []) or [])) != width
            or len(list(stats.get("std", []) or [])) != width
        ):
            raise ValueError(f"development normalization width mismatch for {name}")

    sample_sha = _sha256_file(sample_path)
    candidate_sha = _sha256_file(candidate_path)
    contract_keys = (
        "schema_version", "method", "split_roles", "train_start_year", "development_year",
        "development_start", "development_end", "source_padding_end",
        "source_padding_trade_date_count", "development_start_date_idx", "forward_days",
        "execution_tail_days", "max_label_dependency_days", "safe_train_signal_end",
        "max_train_dependency_end", "purge_rule", "purged_row_count",
        "purged_signal_date_count", "label_dependency_overlap_count", "candidate_universe_rule",
        "candidate_count", "supervised_development_count", "unsupervised_candidate_count",
        "normalization_cutoff_exclusive", "training_label_dependency_days",
        "execution_dependency_days",
    )
    payload = {
        "schema_version": 2,
        "artifact_type": "seq100_development_fold_training_contract",
        "lookback_days": int(manifest.get("lookback_days", 0) or 0),
        "forward_days": forward_days,
        "max_label_dependency_days": dependency_days,
        "training_label_dependency_days": dependency_days,
        "execution_dependency_days": execution_dependency_days,
        "supervised_sample_index": {
            "sha256": sample_sha,
            **sample_meta,
        },
        "candidate_index": {
            "sha256": candidate_sha,
            **candidate_meta,
            "full_signal_day_universe": True,
            "future_label_required_for_membership": False,
            "entry_fill_required_for_membership": False,
        },
        "split": {
            "roles": {"fit": "train", "evaluation": "development"},
            "supervised_counts": dict(sorted(sample_splits.items())),
            "candidate_counts": {"development": int(candidate_splits["development"])},
            "train_date_idx_min": int(train_stats[0]),
            "train_date_idx_max": int(train_stats[1]),
            "development_date_idx_min": start_idx,
            "development_date_idx_max": int(candidate_stats[1]),
            "development_year": year,
            "train_years": train_years,
        },
        "development_walkforward": {key: contract.get(key) for key in contract_keys},
        "normalization": normalization,
        "feature_channels": dict(manifest.get("feature_channels", {}) or {}),
        "label_arrays": dict(manifest.get("label_arrays", {}) or {}),
        "label_semantics": dict(manifest.get("label_semantics", {}) or {}),
        "execution_contract": dict(manifest.get("execution_contract", {}) or {}),
        "research_contract": research_contract,
        "development_contract": dict(manifest.get("development_contract", {}) or {}),
        "masks": dict(manifest.get("masks", {}) or {}),
        "scope": dict(manifest.get("scope", {}) or {}),
        "source_view_provenance": dict(manifest.get("source_view_provenance", {}) or {}),
        "date_values_sha256": _canonical_sha256(list(manifest.get("date_values", []) or [])),
        "symbol_values_sha256": _canonical_sha256(list(manifest.get("symbol_values", []) or [])),
    }
    return {
        "schema_version": 2,
        "algorithm": "sha256",
        "sha256": _canonical_sha256(payload),
        "sample_index_sha256": sample_sha,
        "candidate_index_sha256": candidate_sha,
        "normalization_sha256": _canonical_sha256(normalization),
        "payload": payload,
    }


def validate_development_fold_training_contract(manifest: Mapping[str, Any]) -> dict[str, Any]:
    stored = dict(manifest.get("development_fold_training_contract", {}) or {})
    if not stored or stored != compute_development_fold_training_contract(manifest):
        raise ValueError("development_fold_training_contract does not match current training material")
    return stored


def _fit_feature_normalization_before(
    meta: Mapping[str, Any],
    *,
    end_date_idx_exclusive: int,
    date_chunk_size: int = 32,
) -> dict[str, list[float]]:
    shape = tuple(int(value) for value in meta.get("shape", []))
    columns = list(meta.get("columns", []) or [])
    if len(shape) != 3 or shape[2] != len(columns):
        raise ValueError("feature channel must be a dense [date, symbol, feature] panel")
    cutoff = int(end_date_idx_exclusive)
    if cutoff <= 0 or cutoff > shape[0]:
        raise ValueError("normalization cutoff is outside the feature panel")
    panel = np.memmap(
        _workspace_path(str(meta["path"])),
        dtype="float32",
        mode="r",
        shape=shape,
    )
    width = int(shape[2])
    count = np.zeros(width, dtype=np.int64)
    total = np.zeros(width, dtype=np.float64)
    total_sq = np.zeros(width, dtype=np.float64)
    for start in range(0, cutoff, max(1, int(date_chunk_size))):
        block = np.asarray(panel[start : min(cutoff, start + int(date_chunk_size))], dtype=np.float64)
        flat = block.reshape(-1, width)
        finite = np.isfinite(flat)
        count += finite.sum(axis=0, dtype=np.int64)
        clean = np.where(finite, flat, 0.0)
        total += clean.sum(axis=0, dtype=np.float64)
        total_sq += np.square(clean).sum(axis=0, dtype=np.float64)
    mean = np.divide(total, count, out=np.zeros(width, dtype=np.float64), where=count > 0)
    second = np.divide(total_sq, count, out=np.zeros(width, dtype=np.float64), where=count > 0)
    variance = np.maximum(second - np.square(mean), 0.0)
    std = np.sqrt(variance)
    std = np.where(np.isfinite(std) & (std > 1.0e-6), std, 1.0)
    return {
        "mean": mean.astype(np.float32).tolist(),
        "std": std.astype(np.float32).tolist(),
    }


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(dict(payload), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def build_development_fold_view(
    *,
    source_manifest: str | Path,
    output_root: str | Path,
    development_year: int,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Create one expanding OOF view with a D60-only training purge."""

    source_path = _workspace_path(source_manifest).resolve()
    source = json.loads(source_path.read_text(encoding="utf-8"))
    if str(source.get("artifact_type", "")) != "qdp_v2_sequence_path_pack":
        raise ValueError("source manifest is not a sequence path pack")
    forward_days = int(source.get("forward_days", 0) or 0)
    execution_tail_days = int(source.get("execution_tail_days", 0) or 0)
    if forward_days <= 0 or execution_tail_days < 0:
        raise ValueError("source pack has invalid prediction or execution horizons")
    if int(source.get("max_label_dependency_days", 0) or 0) != forward_days:
        raise ValueError("source pack labels are not bound to the D60-only dependency contract")
    execution_dependency_days = forward_days + execution_tail_days
    declared_execution_dependency = int(
        source.get("max_execution_dependency_days", execution_dependency_days) or 0
    )
    if declared_execution_dependency != execution_dependency_days:
        raise ValueError("source pack execution dependency metadata is inconsistent")
    research_contract = dict(source.get("research_contract", {}) or {})
    if not research_contract:
        raise ValueError("source pack is not bound to a research contract")

    source_sample_path = _workspace_path(str(source.get("sample_index_path", "") or "")).resolve()
    source_candidate_path = _workspace_path(str(source.get("candidate_index_path", "") or "")).resolve()
    if not source_sample_path.is_file() or not source_candidate_path.is_file():
        raise FileNotFoundError("source sample or candidate index is missing")
    required = {
        "year", "trade_date", "date_idx", "symbol_idx", "symbol", "entry_trade_date",
        "entry_filled", "label_valid", "price_label_valid", "va_aux_valid",
    }
    source_sample_meta = _parquet_contract_meta(source_sample_path)
    source_candidate_meta = _parquet_contract_meta(source_candidate_path)
    if required.difference(source_sample_meta["columns"]) or required.difference(
        source_candidate_meta["columns"]
    ):
        raise ValueError("source indexes are missing required fold columns")
    year = int(development_year)
    output = _workspace_path(output_root).resolve()
    index_dir = output / "indexes"
    view_dir = output / "views"
    sample_path = index_dir / f"development_{year}_purge{forward_days}.parquet"
    candidate_path = index_dir / f"candidates_{year}.parquet"
    view_path = view_dir / f"l35v2_pit_{year}.json"
    for target in (sample_path, candidate_path, view_path):
        if target.exists() and not bool(overwrite):
            raise FileExistsError(target)
        if target.exists():
            target.unlink()
    index_dir.mkdir(parents=True, exist_ok=True)
    source_sample_sql = _duckdb_parquet_sql(source_sample_path)
    source_candidate_sql = _duckdb_parquet_sql(source_candidate_path)
    sample_output_sql = _duckdb_parquet_sql(sample_path)
    candidate_output_sql = _duckdb_parquet_sql(candidate_path)
    import duckdb  # type: ignore

    con = duckdb.connect(":memory:")
    try:
        con.execute("set threads=2")
        con.execute("set memory_limit='1GB'")
        candidate_count, development_start_idx, development_end_idx = con.execute(
            f"""
            select count(*), min(date_idx), max(date_idx)
            from read_parquet({source_candidate_sql})
            where cast(year as integer) = {year}
            """
        ).fetchone()
        candidate_count = int(candidate_count or 0)
        if candidate_count <= 0:
            raise ValueError(f"source pack has no candidate rows for {year}")
        development_start_idx = int(development_start_idx)
        development_end_idx = int(development_end_idx)
        safe_train_signal_end_idx = development_start_idx - forward_days - 1
        if safe_train_signal_end_idx < 0:
            raise ValueError("development fold has no room for a strict training purge")
        train_count, development_supervised_count, purged_row_count, train_start_year = con.execute(
            f"""
            select
              count(*) filter (
                where cast(year as integer) < {year}
                  and cast(date_idx as bigint) <= {safe_train_signal_end_idx}
              ),
              count(*) filter (where cast(year as integer) = {year}),
              count(*) filter (
                where cast(year as integer) < {year}
                  and cast(date_idx as bigint) > {safe_train_signal_end_idx}
                  and cast(date_idx as bigint) < {development_start_idx}
              ),
              min(cast(year as integer)) filter (
                where cast(year as integer) < {year}
                  and cast(date_idx as bigint) <= {safe_train_signal_end_idx}
              )
            from read_parquet({source_sample_sql})
            """
        ).fetchone()
        train_count = int(train_count or 0)
        development_supervised_count = int(development_supervised_count or 0)
        purged_row_count = int(purged_row_count or 0)
        train_start_year = int(train_start_year or 0)
        if train_count <= 0 or development_supervised_count <= 0:
            raise ValueError("development fold has an empty supervised split")
        con.execute(
            f"""
            copy (
              select
                sample_id,
                cast(split as varchar) as source_split,
                case when cast(year as integer) = {year} then 'development' else 'train' end as split,
                cast(year as smallint) as year,
                cast(trade_date as varchar) as trade_date,
                cast(date_idx as bigint) as date_idx,
                cast(symbol_idx as integer) as symbol_idx,
                cast(symbol as varchar) as symbol,
                cast(entry_trade_date as varchar) as entry_trade_date,
                cast(entry_filled as boolean) as entry_filled,
                cast(label_valid as boolean) as label_valid,
                cast(price_label_valid as boolean) as price_label_valid,
                cast(va_aux_valid as boolean) as va_aux_valid,
                cast(date_idx as bigint) + {forward_days} as label_end_date_idx,
                cast(date_idx as bigint) + {forward_days} as training_label_end_date_idx,
                cast(date_idx as bigint) + {execution_dependency_days} as execution_dependency_end_date_idx
              from read_parquet({source_sample_sql})
              where (cast(year as integer) < {year}
                     and cast(date_idx as bigint) <= {safe_train_signal_end_idx})
                 or cast(year as integer) = {year}
              order by date_idx, symbol_idx
            ) to {sample_output_sql}
            (format parquet, compression zstd, row_group_size 250000)
            """
        )
        con.execute(
            f"""
            copy (
              select
                row_number() over (order by date_idx, symbol_idx) - 1 as candidate_id,
                'development' as split,
                {year}::integer as year,
                cast(trade_date as varchar) as trade_date,
                cast(date_idx as bigint) as date_idx,
                cast(symbol_idx as integer) as symbol_idx,
                cast(symbol as varchar) as symbol,
                cast(entry_trade_date as varchar) as entry_trade_date,
                cast(entry_filled as boolean) as entry_filled,
                cast(label_valid as boolean) as label_valid,
                cast(price_label_valid as boolean) as price_label_valid,
                cast(va_aux_valid as boolean) as va_aux_valid,
                cast(split as varchar) as source_split
              from read_parquet({source_candidate_sql})
              where cast(year as integer) = {year}
              order by date_idx, symbol_idx
            ) to {candidate_output_sql}
            (format parquet, compression zstd, row_group_size 250000)
            """
        )
    finally:
        con.close()

    date_values = [str(value) for value in list(source.get("date_values", []) or [])]
    if development_end_idx + execution_dependency_days >= len(date_values):
        raise ValueError("source pack lacks the full execution tail for this fold")
    normalization: dict[str, Any] = {
        "fit_scope": "feature_dates_before_development_start",
        "fit_date_start": date_values[0],
        "fit_date_end": date_values[development_start_idx - 1],
        "fit_date_count": development_start_idx,
        "fit_date_end_exclusive": date_values[development_start_idx],
        "development_feature_date_count": 0,
    }
    for name, meta in dict(source.get("feature_channels", {}) or {}).items():
        normalization[str(name)] = _fit_feature_normalization_before(
            dict(meta),
            end_date_idx_exclusive=development_start_idx,
        )

    development_walkforward = {
        "schema_version": 2,
        "method": "expanding_train_development_walkforward",
        "split_roles": {"fit": "train", "evaluation": "development"},
        "train_start_year": train_start_year,
        "development_year": year,
        "development_start": date_values[development_start_idx],
        "development_end": date_values[development_end_idx],
        "source_padding_end": date_values[development_end_idx + execution_dependency_days],
        "source_padding_trade_date_count": execution_dependency_days,
        "development_start_date_idx": development_start_idx,
        "forward_days": forward_days,
        "execution_tail_days": execution_tail_days,
        "max_label_dependency_days": forward_days,
        "training_label_dependency_days": forward_days,
        "execution_dependency_days": execution_dependency_days,
        "safe_train_signal_end": date_values[safe_train_signal_end_idx],
        "max_train_dependency_end": date_values[safe_train_signal_end_idx + forward_days],
        "purge_rule": "training_label_end_date_idx < development_start_date_idx",
        "purged_row_count": purged_row_count,
        "purged_signal_date_count": forward_days,
        "purged_signal_start": date_values[development_start_idx - forward_days],
        "purged_signal_end": date_values[development_start_idx - 1],
        "label_dependency_overlap_count": 0,
        "candidate_universe_rule": "signal_day_input_valid_and_signal_eligible",
        "candidate_count": candidate_count,
        "supervised_development_count": development_supervised_count,
        "unsupervised_candidate_count": candidate_count - development_supervised_count,
        "normalization_cutoff_exclusive": date_values[development_start_idx],
    }
    if int(development_walkforward["label_dependency_overlap_count"]) != 0:
        raise ValueError("constructed fold still overlaps the development year")

    inventory = _source_backing_file_inventory(source)
    provenance = {
        "schema_version": 1,
        "manifest_path": str(source_path),
        "manifest_sha256": _sha256_file(source_path),
        "sample_index_path": str(source_sample_path),
        "sample_index_sha256": _sha256_file(source_sample_path),
        "candidate_index_path": str(source_candidate_path),
        "candidate_index_sha256": _sha256_file(source_candidate_path),
        "artifact_type": str(source.get("artifact_type", "")),
        "artifact_view_id": "",
        "backing_files": inventory,
        "backing_files_sha256": _canonical_sha256(inventory),
    }
    view = dict(source)
    view.update(
        {
            "sample_index_path": str(sample_path),
            "candidate_index_path": str(candidate_path),
            "sample_count": train_count + development_supervised_count,
            "sample_count_by_split": {
                "development": development_supervised_count,
                "train": train_count,
            },
            "candidate_count": candidate_count,
            "candidate_count_by_split": {"development": candidate_count},
            "normalization": normalization,
            "development_walkforward": development_walkforward,
            "development_contract": research_contract,
            "source_view_provenance": provenance,
            "qdp_source_freshness_policy": {
                "mode": "immutable_research_pack_v1",
                "attachments": [
                    {
                        "path": str(_workspace_path(str(research_contract["path"])).resolve()),
                        "sha256": _sha256_file(research_contract["path"]),
                    }
                ],
            },
        }
    )
    semantics = dict(view.get("data_semantics", {}) or {})
    semantics["training_purge"] = {
        "days": forward_days,
        "uses_execution_tail": False,
        "execution_tail_days": execution_tail_days,
    }
    view["data_semantics"] = semantics
    view["development_fold_training_contract"] = compute_development_fold_training_contract(view)
    _atomic_write_json(view_path, view)
    validate_source_view_provenance(view)
    return {
        "year": year,
        "view_path": str(view_path),
        "sample_index_path": str(sample_path),
        "candidate_index_path": str(candidate_path),
        "train_count": train_count,
        "development_supervised_count": development_supervised_count,
        "development_candidate_count": candidate_count,
        "purge_trading_days": forward_days,
        "execution_dependency_days": execution_dependency_days,
        "contract_sha256": str(view["development_fold_training_contract"]["sha256"]),
    }


def build_development_fold_views(
    *,
    source_manifest: str | Path,
    output_root: str | Path,
    years: list[int] | tuple[int, ...],
    overwrite: bool = False,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for year in years:
        results.append(
            build_development_fold_view(
                source_manifest=source_manifest,
                output_root=output_root,
                development_year=int(year),
                overwrite=overwrite,
            )
        )
    return results


def _parse_years(value: str) -> list[int]:
    years: list[int] = []
    for chunk in str(value).split(","):
        item = chunk.strip()
        if not item:
            continue
        if "-" in item:
            start, end = (int(part) for part in item.split("-", 1))
            years.extend(range(start, end + 1))
        else:
            years.append(int(item))
    return sorted(set(years))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build immutable seq100 OOF fold views.")
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--years", required=True, help="Comma-separated years or ranges, e.g. 2020-2025")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    results = build_development_fold_views(
        source_manifest=args.source_manifest,
        output_root=args.output_root,
        years=_parse_years(args.years),
        overwrite=bool(args.overwrite),
    )
    print(json.dumps(results, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
