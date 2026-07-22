from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd


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
    dependency_days = int(contract.get("max_label_dependency_days", 0) or 0)
    overlap = int(((train["date_idx"].astype(int) + dependency_days) >= start_idx).sum())
    if dependency_days <= 0 or overlap:
        raise ValueError(f"training label dependencies overlap development: {overlap}")
    if str(contract.get("purge_rule", "")) != "max_label_dependency_date_idx < development_start_date_idx":
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
    payload = {
        "schema_version": 1,
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
    return {
        "schema_version": 1,
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
