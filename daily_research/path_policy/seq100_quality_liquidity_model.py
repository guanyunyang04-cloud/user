from __future__ import annotations

"""Prepare and run the first rolling quality-liquidity LightGBM comparison."""

import argparse
import gc
import hashlib
import itertools
import json
import math
import os
import time
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import average_precision_score

from daily_research.path_policy import seq100_path_label_learnability as learnability
from daily_research.path_policy import seq100_quality_liquidity_data_prep as data_prep
from daily_research.path_policy import (
    seq100_quality_liquidity_descriptive_feature_audit as descriptive,
)
from daily_research.path_policy import seq100_quality_liquidity_training_ready as ready

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
STUDY_ID = "seq100_quality_liquidity_model"
YEARS = tuple(range(2011, 2026))
ROLLING_YEARS = (2023, 2024, 2025)
FORBIDDEN_YEAR = 2026
MAXIMUM_OUTCOME_DATE = "2025-12-31"
HORIZONS = (10, 20)
LABEL_HORIZON_INDEX = {10: 1, 20: 2}
STATE_HORIZON = 10
FLAG_MFE_PRE_PEAK_MAE_VALID = 128
FLAG_STATE_ASSIGNED = 256
TARGETS = ("mfe_10", "mfe_20", "risk_10", "risk_20", "state_10")
CORE_VARIANTS = (
    "legacy_core",
    "legacy_plus_technical",
    "legacy_plus_moneyflow",
    "full_core",
)
GATED_FAMILIES = ("margin", "research", "financial_extensions")
RISK_STATE_CORE_VARIANTS = ("legacy_core", "full_core")
ALL_GATED_VARIANT = "selected_core_plus_all_gated"
MODEL_INPUT_DIR_NAME = "model_inputs"

DEFAULT_STUDY_PATH = (
    WORKSPACE_ROOT / "daily_research/studies/seq100_quality_liquidity_model.json"
)
DEFAULT_READY_ROOT = (
    WORKSPACE_ROOT
    / "daily_research/output/path_policy/studies/seq100_quality_liquidity_training_ready"
)
DEFAULT_AUDIT_ROOT = (
    WORKSPACE_ROOT
    / "daily_research/output/path_policy/studies/seq100_quality_liquidity_descriptive_feature_audit"
)
DEFAULT_OUTPUT_ROOT = (
    WORKSPACE_ROOT
    / "daily_research/output/path_policy/studies/seq100_quality_liquidity_model"
)


class ModelError(RuntimeError):
    pass


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ModelError(f"required_json_missing:{path}")
    return dict(json.loads(path.read_text(encoding="utf-8")))


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False, default=str)
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _write_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    frame.to_parquet(temporary, index=False, compression="zstd")
    os.replace(temporary, path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _stable_hash(payload: Any) -> str:
    encoded = json.dumps(
        payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"), default=str
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _file_record(path: Path, **extra: Any) -> dict[str, Any]:
    result = {
        "path": str(path.resolve()),
        "sha256": _sha256(path),
        "size": int(path.stat().st_size),
    }
    result.update(extra)
    return result


def _resolve_source_path(path: str | Path) -> Path:
    value = Path(path)
    if not value.is_absolute():
        value = WORKSPACE_ROOT / value
    return value.resolve()


def _load_config(study_path: Path = DEFAULT_STUDY_PATH) -> dict[str, Any]:
    config = _read_json(study_path)
    if config.get("study_id") != STUDY_ID:
        raise ModelError("study_id_mismatch")
    period = dict(config.get("period", {}) or {})
    if tuple(int(year) for year in period.get("research_years", ())) != YEARS:
        raise ModelError("research_year_contract_mismatch")
    if (
        tuple(int(year) for year in period.get("rolling_prediction_years", ()))
        != ROLLING_YEARS
    ):
        raise ModelError("rolling_year_contract_mismatch")
    if int(period.get("burn_in_year", -1)) != 2010:
        raise ModelError("burn_in_contract_mismatch")
    if (
        int(dict(config.get("source", {}) or {}).get("forbidden_year", -1))
        != FORBIDDEN_YEAR
    ):
        raise ModelError("forbidden_year_contract_mismatch")
    if (
        str(dict(config.get("source", {}) or {}).get("maximum_outcome_date"))
        != MAXIMUM_OUTCOME_DATE
    ):
        raise ModelError("outcome_cutoff_contract_mismatch")
    if not bool(
        dict(config.get("execution", {}) or {}).get("training_performed", False)
    ):
        raise ModelError("model_execution_training_contract_disabled")
    return config


def _source_manifest(ready_root: Path = DEFAULT_READY_ROOT) -> dict[str, Any]:
    manifest = _read_json(ready_root / "manifest.json")
    readiness = _read_json(ready_root / "readiness.json")
    if manifest.get("study_id") != "seq100_quality_liquidity_training_ready":
        raise ModelError("training_ready_study_mismatch")
    if manifest.get("training_performed") or manifest.get("feature_set_selected"):
        raise ModelError("training_ready_already_trained")
    if int(manifest.get("forbidden_2026_rows", -1)) != 0:
        raise ModelError("training_ready_contains_2026_rows")
    if readiness.get("status") not in {"ready", "ready_with_documented_optional_gaps"}:
        raise ModelError("training_ready_not_ready")
    return manifest


def _audit_manifest(audit_root: Path = DEFAULT_AUDIT_ROOT) -> dict[str, Any]:
    manifest = _read_json(audit_root / "manifest.json")
    if manifest.get("study_id") != "seq100_quality_liquidity_descriptive_feature_audit":
        raise ModelError("descriptive_audit_study_mismatch")
    if manifest.get("status") != "completed":
        raise ModelError("descriptive_audit_not_completed")
    if manifest.get("training_performed") or manifest.get("feature_set_selected"):
        raise ModelError("descriptive_audit_training_contract_violated")
    if (
        int(dict(manifest.get("scope", {}) or {}).get("forbidden_2026_read_count", -1))
        != 0
    ):
        raise ModelError("descriptive_audit_contains_2026_reads")
    return manifest


def _catalog_and_registry(
    *, ready_root: Path, audit_root: Path, ready_manifest: Mapping[str, Any]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    audit_catalog_path = audit_root / "numeric_feature_catalog.parquet"
    catalog = pd.read_parquet(audit_catalog_path)
    registry = pd.read_parquet(Path(ready_manifest["feature_registry"]["path"]))
    required = {
        "feature_name",
        "physical_column",
        "block",
        "eligibility",
        "analytic_family",
    }
    if not required.issubset(catalog.columns):
        raise ModelError(
            f"numeric_catalog_columns_missing:{sorted(required - set(catalog.columns))}"
        )
    if len(catalog) != 628 or catalog["feature_name"].duplicated().any():
        raise ModelError("numeric_catalog_count_or_uniqueness_mismatch")
    if bool(catalog["feature_name"].isin(["listing_age_days"]).any()):
        raise ModelError("defective_listing_age_days_in_model_catalog")
    if int(catalog["feature_name"].eq("listing_age_open_days").sum()) != 1:
        raise ModelError("canonical_listing_age_open_days_missing")
    if bool(
        catalog["feature_name"]
        .astype(str)
        .str.contains("qfq|hfq", case=False, regex=True)
        .any()
    ):
        raise ModelError("adjusted_price_field_in_model_catalog")
    if bool(
        catalog["physical_column"]
        .astype(str)
        .isin(["net_mf_vol", "net_mf_amount"])
        .any()
    ):
        raise ModelError("unreconciled_moneyflow_field_in_model_catalog")
    return catalog, registry


def _base_manifest() -> dict[str, Any]:
    return _read_json(data_prep.BASE_FEATURE_MANIFEST)


def _label_manifest() -> dict[str, Any]:
    return _read_json(data_prep.LABEL_MANIFEST)


def _feature_contract(
    *, catalog: pd.DataFrame, registry: pd.DataFrame, base_manifest: Mapping[str, Any]
) -> dict[str, Any]:
    base_catalog = pd.DataFrame(base_manifest["continuous_catalog"])
    base_names = base_catalog["name"].astype(str).tolist()
    if ready.LEGACY_LISTING_AGE_FIELD not in base_names:
        raise ModelError("base_manifest_listing_age_field_missing")
    base_valid_names = [
        name for name in base_names if name != ready.LEGACY_LISTING_AGE_FIELD
    ]
    base_rows = catalog[catalog["block"].eq("existing_seq100_base")].copy()
    if len(base_rows) != 295 or set(base_rows["feature_name"]) != set(base_valid_names):
        raise ModelError("base_reuse_feature_contract_mismatch")
    base_index = dict(
        zip(
            base_catalog["name"].astype(str),
            base_catalog["column_index"].astype(int),
            strict=True,
        )
    )
    base_valid_names.sort(key=lambda name: base_index[name])
    extra = catalog[~catalog["block"].eq("existing_seq100_base")].copy()
    if len(extra) != 333:
        raise ModelError("extra_numeric_feature_contract_mismatch")
    extra_names = extra["feature_name"].astype(str).tolist()
    extra_index = {name: index for index, name in enumerate(extra_names)}
    metadata = registry[registry["eligibility"].eq("availability_metadata")].copy()
    if len(metadata) != 13 or metadata["feature_name"].duplicated().any():
        raise ModelError("availability_metadata_contract_mismatch")

    def names_where(mask: pd.Series) -> list[str]:
        return catalog.loc[mask, "feature_name"].astype(str).tolist()

    legacy = names_where(
        catalog["eligibility"].eq("formal_existing")
        & ~catalog["analytic_family"].eq("research_reports")
    )
    technical = names_where(
        catalog["block"].eq("tushare_technical_candidates")
        & catalog["eligibility"].eq("formal_candidate")
    )
    moneyflow = names_where(
        catalog["block"].eq("traditional_moneyflow_features")
        & catalog["eligibility"].eq("formal_candidate")
    )
    margin = names_where(
        catalog["block"].eq("margin_features")
        & catalog["eligibility"].eq("formal_candidate")
    )
    research = names_where(catalog["analytic_family"].eq("research_reports"))
    financial = names_where(
        catalog["block"].eq("financial_statement_extensions")
        & catalog["eligibility"].eq("availability_gated")
    )
    full_core = [*legacy, *technical, *moneyflow]
    gated = [*margin, *research, *financial]
    groups = {
        "legacy_core": legacy,
        "legacy_plus_technical": [*legacy, *technical],
        "legacy_plus_moneyflow": [*legacy, *moneyflow],
        "full_core": full_core,
        "margin": margin,
        "research": research,
        "financial_extensions": financial,
        "all_gated": gated,
    }
    expected = {
        "legacy_core": 493,
        "legacy_plus_technical": 571,
        "legacy_plus_moneyflow": 509,
        "full_core": 587,
        "margin": 12,
        "research": 25,
        "financial_extensions": 4,
        "all_gated": 41,
    }
    for name, count in expected.items():
        if len(groups[name]) != count:
            raise ModelError(
                f"feature_group_count_mismatch:{name}:{len(groups[name])}:{count}"
            )
    if len(set(full_core + gated)) != 628:
        raise ModelError("feature_group_union_count_mismatch")
    return {
        "base_names": base_valid_names,
        "base_index": base_index,
        "extra_names": extra_names,
        "extra_index": extra_index,
        "metadata": metadata[
            ["feature_name", "physical_column", "block", "source_field"]
        ].to_dict("records"),
        "groups": groups,
        "catalog": catalog,
    }


def _row_index(ready_manifest: Mapping[str, Any]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for year in YEARS:
        path = Path(ready_manifest["row_spine"][str(year)]["path"])
        frame = pd.read_parquet(
            path,
            columns=["candidate_id", "date_idx", "trade_date", "symbol", "security_id"],
        )
        if frame.empty or not frame["candidate_id"].is_unique:
            raise ModelError(f"row_index_year_empty_or_duplicate:{year}")
        frame["trade_date"] = frame["trade_date"].astype(str)
        frames.append(frame)
    result = pd.concat(frames, ignore_index=True)
    if len(result) != int(ready_manifest["common_support"]["row_count"]):
        raise ModelError("row_index_count_mismatch")
    if not result["candidate_id"].is_unique:
        raise ModelError("row_index_candidate_id_duplicate")
    candidate_ids = result["candidate_id"].to_numpy(dtype=np.int64)
    if not bool((candidate_ids[1:] > candidate_ids[:-1]).all()):
        raise ModelError("row_index_candidate_id_not_strictly_increasing")
    date_idx = result["date_idx"].to_numpy(dtype=np.int32)
    if not bool((date_idx[1:] >= date_idx[:-1]).all()):
        raise ModelError("row_index_date_not_ordered")
    if bool(result["trade_date"].str.startswith("2026-").any()):
        raise ModelError("row_index_contains_2026")
    return result[["candidate_id", "date_idx", "trade_date", "symbol", "security_id"]]


def _source_paths(ready_manifest: Mapping[str, Any], year: int) -> dict[str, Path]:
    paths = descriptive._physical_sources(ready_manifest, year)
    for block in (
        "membership_context",
        "tushare_technical_candidates",
        "margin_features",
        "traditional_moneyflow_features",
        "financial_statement_extensions",
    ):
        paths[block] = Path(
            ready_manifest["blocks"][block]["partitions"][str(year)]["path"]
        )
    return paths


def _align_source_frame(frame: pd.DataFrame, expected_ids: np.ndarray) -> pd.DataFrame:
    ids = frame["candidate_id"].to_numpy(dtype=np.int64)
    if np.array_equal(ids, expected_ids):
        return frame.reset_index(drop=True)
    if not frame["candidate_id"].is_unique:
        raise ModelError("source_candidate_id_duplicate")
    if not bool(np.isin(expected_ids, ids, assume_unique=True).all()):
        raise ModelError("source_candidate_id_missing")
    indexed = frame.set_index("candidate_id")
    try:
        aligned = indexed.reindex(expected_ids)
    except Exception as exc:  # pragma: no cover - pandas error wording varies
        raise ModelError("source_candidate_id_alignment_failed") from exc
    aligned.insert(0, "candidate_id", expected_ids)
    return aligned.reset_index(drop=True)


def _as_numeric(frame: pd.DataFrame, columns: Sequence[str]) -> np.ndarray:
    values = []
    for column in columns:
        series = pd.to_numeric(frame[column], errors="coerce")
        values.append(series.to_numpy(dtype=np.float32, na_value=np.nan))
    return np.column_stack(values).astype(np.float32, copy=False)


def _metadata_mapping(values: pd.Series) -> tuple[str, dict[str, int]]:
    nonnull = values.dropna()
    if nonnull.empty:
        return "string", {}
    if pd.api.types.is_bool_dtype(values) or (
        pd.api.types.is_numeric_dtype(values)
        and set(pd.to_numeric(nonnull, errors="coerce").dropna().unique()).issubset(
            {0, 1}
        )
    ):
        return "boolean", {"0": 0, "1": 1, "False": 0, "True": 1}
    normalized = nonnull.astype(str).str.strip()
    unknown = {"", "unknown", "nan", "none", "<na>"}
    categories = sorted({value for value in normalized if value.lower() not in unknown})
    return "string", {value: index for index, value in enumerate(categories)}


def _encode_metadata(
    values: pd.Series, *, kind: str, mapping: Mapping[str, int]
) -> np.ndarray:
    if kind == "boolean":
        numeric = pd.to_numeric(values, errors="coerce")
        if pd.api.types.is_bool_dtype(values):
            numeric = values.astype("Float64")
        output = numeric.to_numpy(dtype=np.float64, na_value=np.nan)
        invalid = np.isfinite(output) & ~np.isin(output, (0.0, 1.0))
        if bool(invalid.any()):
            raise ModelError("availability_boolean_value_outside_0_1")
        return np.where(np.isfinite(output), output, -1).astype(np.int8)
    normalized = values.astype("string").str.strip()
    output = np.full(len(values), -1, dtype=np.int8)
    for value, code in mapping.items():
        matches = normalized.eq(value).fillna(False).to_numpy(dtype=bool)
        output[matches] = np.int8(code)
    unknown = normalized.isna() | normalized.str.lower().isin(
        {"", "unknown", "nan", "none", "<na>"}
    )
    unknown_mask = unknown.fillna(True).to_numpy(dtype=bool)
    output[unknown_mask] = -1
    unmatched = ~unknown_mask & (output == -1)
    if bool(unmatched.any()):
        raise ModelError("availability_string_value_unmapped")
    return output


def _metadata_mappings(
    *, ready_manifest: Mapping[str, Any], metadata: Sequence[Mapping[str, Any]]
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for item in metadata:
        name = str(item["feature_name"])
        block = str(item["block"])
        physical = str(item["physical_column"])
        values: list[pd.Series] = []
        for year in YEARS:
            path = _source_paths(ready_manifest, year)[block]
            values.append(pd.read_parquet(path, columns=[physical])[physical])
        combined = pd.concat(values, ignore_index=True)
        kind, mapping = _metadata_mapping(combined)
        result[name] = {
            "kind": kind,
            "mapping": mapping,
            "physical_column": physical,
            "block": block,
            "source_field": str(item["source_field"]),
        }
    return result


def _write_feature_storage(
    *,
    ready_manifest: Mapping[str, Any],
    row_index: pd.DataFrame,
    contract: Mapping[str, Any],
    input_dir: Path,
) -> dict[str, Any]:
    row_count = len(row_index)
    extra_names = list(contract["extra_names"])
    extra_index = dict(contract["extra_index"])
    catalog = pd.DataFrame(contract["catalog"])
    extra_path = input_dir / "extra_numeric.float32.dat"
    extra_partial = input_dir / "extra_numeric.float32.dat.partial"
    availability_path = input_dir / "availability.int8.dat"
    availability_partial = input_dir / "availability.int8.dat.partial"
    extra_mm = np.memmap(
        extra_partial, dtype=np.float32, mode="w+", shape=(len(extra_names), row_count)
    )
    extra_mm[:] = np.nan
    metadata = list(contract["metadata"])
    metadata_mappings = _metadata_mappings(
        ready_manifest=ready_manifest, metadata=metadata
    )
    availability_mm = np.memmap(
        availability_partial, dtype=np.int8, mode="w+", shape=(len(metadata), row_count)
    )
    availability_mm[:] = -1
    metadata_index = {
        str(item["feature_name"]): index for index, item in enumerate(metadata)
    }
    block_order = [
        "minute",
        "fundamental",
        "event",
        "membership_context",
        "tushare_technical_candidates",
        "margin_features",
        "traditional_moneyflow_features",
        "financial_statement_extensions",
    ]
    for year in YEARS:
        year_rows = row_index["trade_date"].str.startswith(f"{year}-").to_numpy()
        positions = np.flatnonzero(year_rows)
        expected_ids = row_index.loc[positions, "candidate_id"].to_numpy(dtype=np.int64)
        paths = _source_paths(ready_manifest, year)
        for block in block_order:
            block_features = catalog[catalog["block"].eq(block)]
            block_metadata = [item for item in metadata if str(item["block"]) == block]
            if block_features.empty and not block_metadata:
                continue
            path = paths.get(block)
            if path is None:
                raise ModelError(f"source_block_path_missing:{year}:{block}")
            physical = block_features["physical_column"].astype(str).tolist()
            metadata_physical = [
                str(item["physical_column"]) for item in block_metadata
            ]
            columns = list(
                dict.fromkeys(["candidate_id", *physical, *metadata_physical])
            )
            frame = _align_source_frame(
                pd.read_parquet(path, columns=columns), expected_ids
            )
            if physical:
                numeric = _as_numeric(frame, physical)
                for offset, name in enumerate(
                    block_features["feature_name"].astype(str)
                ):
                    extra_mm[extra_index[name], positions] = numeric[:, offset]
            for item in block_metadata:
                name = str(item["feature_name"])
                encoded = _encode_metadata(
                    frame[str(item["physical_column"])],
                    kind=str(metadata_mappings[name]["kind"]),
                    mapping=metadata_mappings[name]["mapping"],
                )
                availability_mm[metadata_index[name], positions] = encoded
            del frame
        extra_mm.flush()
        availability_mm.flush()
    extra_sample_rows = np.asarray([0, row_count // 2, row_count - 1], dtype=np.int64)
    extra_sample_hash = _stable_hash(
        extra_mm[:, extra_sample_rows].astype(np.float32).tolist()
    )
    availability_sample_hash = _stable_hash(
        availability_mm[:, extra_sample_rows].astype(np.int8).tolist()
    )
    del extra_mm, availability_mm
    gc.collect()
    return {
        "extra": {
            "path": str(extra_partial.resolve()),
            "final_path": str(extra_path.resolve()),
            "dtype": "float32",
            "shape": [len(extra_names), row_count],
            "sha256": _sha256(extra_partial),
            "sample_rows": extra_sample_rows.tolist(),
            "sample_hash": extra_sample_hash,
        },
        "availability": {
            "path": str(availability_partial.resolve()),
            "final_path": str(availability_path.resolve()),
            "dtype": "int8",
            "shape": [len(metadata), row_count],
            "sha256": _sha256(availability_partial),
            "sample_rows": extra_sample_rows.tolist(),
            "sample_hash": availability_sample_hash,
            "mappings": metadata_mappings,
        },
    }


def _feature_records(contract: Mapping[str, Any]) -> list[dict[str, Any]]:
    catalog = pd.DataFrame(contract["catalog"])
    base_index = dict(contract["base_index"])
    extra_index = dict(contract["extra_index"])
    metadata = {str(item["feature_name"]): item for item in contract["metadata"]}
    rows: list[dict[str, Any]] = []
    for row in catalog.to_dict("records"):
        name = str(row["feature_name"])
        if name in base_index and name != ready.LEGACY_LISTING_AGE_FIELD:
            storage, index = "base", int(base_index[name])
        else:
            storage, index = "extra", int(extra_index[name])
        rows.append(
            {
                "feature_name": name,
                "storage": storage,
                "column_index": index,
                "block": str(row["block"]),
                "analytic_family": str(row["analytic_family"]),
                "eligibility": str(row["eligibility"]),
                "physical_column": str(row["physical_column"]),
            }
        )
    for name, row in metadata.items():
        rows.append(
            {
                "feature_name": name,
                "storage": "availability",
                "column_index": int(list(metadata).index(name)),
                "block": str(row["block"]),
                "analytic_family": "availability_metadata",
                "eligibility": "availability_metadata",
                "physical_column": str(row["physical_column"]),
            }
        )
    return rows


def _model_input_fingerprint(
    *,
    config_path: Path,
    ready_root: Path,
    audit_root: Path,
    contract: Mapping[str, Any],
) -> str:
    return _stable_hash(
        {
            "study_id": STUDY_ID,
            "config_sha256": _sha256(config_path),
            "ready_manifest_sha256": _sha256(ready_root / "manifest.json"),
            "ready_readiness_sha256": _sha256(ready_root / "readiness.json"),
            "audit_manifest_sha256": _sha256(audit_root / "manifest.json"),
            "base_feature_manifest_sha256": _sha256(data_prep.BASE_FEATURE_MANIFEST),
            "label_manifest_sha256": _sha256(data_prep.LABEL_MANIFEST),
            "numeric_feature_names": pd.DataFrame(contract["catalog"])["feature_name"]
            .astype(str)
            .tolist(),
            "groups": contract["groups"],
        }
    )


def _verify_model_input_files(
    manifest: Mapping[str, Any], *, full_hash: bool
) -> dict[str, Any]:
    row_record = dict(manifest["row_index"])
    row_path = Path(row_record["path"])
    if not row_path.is_file() or (
        full_hash and _sha256(row_path) != row_record["sha256"]
    ):
        raise ModelError("model_input_row_index_hash_mismatch")
    if int(row_path.stat().st_size) != int(row_record["size"]):
        raise ModelError("model_input_row_index_size_mismatch")
    row_index = pd.read_parquet(row_path)
    expected_columns = [
        "candidate_id",
        "date_idx",
        "trade_date",
        "symbol",
        "security_id",
    ]
    if list(row_index.columns) != expected_columns:
        raise ModelError("model_input_row_index_schema_mismatch")
    if (
        len(row_index) != int(manifest["row_count"])
        or not row_index["candidate_id"].is_unique
    ):
        raise ModelError("model_input_row_index_count_or_key_mismatch")
    for key in ("extra", "availability"):
        record = dict(manifest["storage"][key])
        path = Path(record["path"])
        if not path.is_file() or (full_hash and _sha256(path) != record["sha256"]):
            raise ModelError(f"model_input_{key}_hash_mismatch")
        expected_size = (
            int(np.prod(record["shape"])) * np.dtype(record["dtype"]).itemsize
        )
        if int(path.stat().st_size) != expected_size:
            raise ModelError(f"model_input_{key}_size_mismatch")
        values = np.memmap(
            path,
            mode="r",
            dtype=np.dtype(record["dtype"]),
            shape=tuple(int(value) for value in record["shape"]),
        )
        sample_rows = np.asarray(record["sample_rows"], dtype=np.int64)
        sample_hash = _stable_hash(values[:, sample_rows].tolist())
        del values
        if sample_hash != record["sample_hash"]:
            raise ModelError(f"model_input_{key}_sample_hash_mismatch")
    return {"row_index": row_index}


def prepare(
    *,
    study_path: Path = DEFAULT_STUDY_PATH,
    ready_root: Path = DEFAULT_READY_ROOT,
    audit_root: Path = DEFAULT_AUDIT_ROOT,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
) -> dict[str, Any]:
    _load_config(study_path)
    ready_manifest = _source_manifest(ready_root)
    _audit_manifest(audit_root)
    catalog, registry = _catalog_and_registry(
        ready_root=ready_root, audit_root=audit_root, ready_manifest=ready_manifest
    )
    base_manifest = _base_manifest()
    label_manifest = _label_manifest()
    if str(label_manifest.get("outcome_cutoff")) != MAXIMUM_OUTCOME_DATE:
        raise ModelError("label_manifest_cutoff_mismatch")
    contract = _feature_contract(
        catalog=catalog, registry=registry, base_manifest=base_manifest
    )
    fingerprint = _model_input_fingerprint(
        config_path=study_path,
        ready_root=ready_root,
        audit_root=audit_root,
        contract=contract,
    )
    input_dir = ready_root / MODEL_INPUT_DIR_NAME
    manifest_path = input_dir / "manifest.json"
    if manifest_path.is_file():
        existing = _read_json(manifest_path)
        if existing.get("input_fingerprint") == fingerprint:
            _verify_model_input_files(existing, full_hash=False)
            return existing
    input_dir.mkdir(parents=True, exist_ok=True)
    row_index = _row_index(ready_manifest)
    row_partial = input_dir / "row_index.parquet.partial"
    row_path = input_dir / "row_index.parquet"
    _write_parquet(row_index, row_partial)
    storage = _write_feature_storage(
        ready_manifest=ready_manifest,
        row_index=row_index,
        contract=contract,
        input_dir=input_dir,
    )
    base_record = {
        "path": str(data_prep.BASE_FEATURE_MANIFEST.resolve()),
        "sha256": _sha256(data_prep.BASE_FEATURE_MANIFEST),
        "shape": list(base_manifest["continuous_shape"]),
        "dtype": "float32",
    }
    label_record = {
        "path": str(data_prep.LABEL_MANIFEST.resolve()),
        "sha256": _sha256(data_prep.LABEL_MANIFEST),
        "candidate_count": int(label_manifest["candidate_count"]),
    }
    features = _feature_records(contract)
    manifest = {
        "schema": "seq100_quality_liquidity_model_inputs/1",
        "status": "completed",
        "study_id": STUDY_ID,
        "created_at": _now(),
        "input_fingerprint": fingerprint,
        "row_count": len(row_index),
        "row_index": _file_record(
            row_partial,
            columns=list(row_index.columns),
            row_count=len(row_index),
        ),
        "base_feature_manifest": base_record,
        "label_manifest": label_record,
        "storage": storage,
        "features": features,
        "feature_groups": contract["groups"],
        "metadata_mappings": storage["availability"]["mappings"],
        "source": {
            "training_ready_manifest": _file_record(ready_root / "manifest.json"),
            "descriptive_audit_manifest": _file_record(audit_root / "manifest.json"),
            "maximum_outcome_date": MAXIMUM_OUTCOME_DATE,
            "forbidden_year": FORBIDDEN_YEAR,
        },
        "training_performed": False,
        "feature_set_selected": False,
    }
    _verify_model_input_files(manifest, full_hash=False)
    os.replace(row_partial, row_path)
    manifest["row_index"]["path"] = str(row_path.resolve())
    for key in ("extra", "availability"):
        record = manifest["storage"][key]
        os.replace(Path(record["path"]), Path(record["final_path"]))
        record["path"] = record.pop("final_path")
    _write_json(manifest_path, manifest)
    return manifest


class ModelInputs:
    """Read the compact model inputs while retaining candidate-id label semantics."""

    def __init__(self, manifest: Mapping[str, Any]) -> None:
        self.manifest = dict(manifest)
        self.row_count = int(manifest["row_count"])
        self.row_index = pd.read_parquet(Path(manifest["row_index"]["path"]))
        self.candidate_ids = self.row_index["candidate_id"].to_numpy(dtype=np.int64)
        self.date_idx = self.row_index["date_idx"].to_numpy(dtype=np.int32)
        self.trade_date = np.asarray(
            self.row_index["trade_date"].astype(str).to_numpy(), dtype=str
        )
        self.years = (
            self.row_index["trade_date"].str.slice(0, 4).astype(np.int16).to_numpy()
        )
        self.base_manifest = _read_json(Path(manifest["base_feature_manifest"]["path"]))
        base_record = self.base_manifest["files"]["continuous"]
        self.base = np.memmap(
            Path(base_record["path"]),
            dtype=np.float32,
            mode="r",
            shape=tuple(int(value) for value in self.base_manifest["continuous_shape"]),
        )
        extra_record = manifest["storage"]["extra"]
        self.extra = np.memmap(
            Path(extra_record["path"]),
            dtype=np.float32,
            mode="r",
            shape=tuple(int(value) for value in extra_record["shape"]),
        )
        availability_record = manifest["storage"]["availability"]
        self.availability = np.memmap(
            Path(availability_record["path"]),
            dtype=np.int8,
            mode="r",
            shape=tuple(int(value) for value in availability_record["shape"]),
        )
        label_manifest = _read_json(Path(manifest["label_manifest"]["path"]))
        self.label_manifest = label_manifest
        label_files = label_manifest["files"]
        candidate_count = int(label_manifest["candidate_count"])
        self.labels = np.memmap(
            Path(label_files["candidate_labels"]["path"]),
            dtype=np.float32,
            mode="r",
            shape=tuple(
                int(value) for value in label_files["candidate_labels"]["shape"]
            ),
        )
        self.flags = np.memmap(
            Path(label_files["label_flags"]["path"]),
            dtype=np.uint16,
            mode="r",
            shape=tuple(int(value) for value in label_files["label_flags"]["shape"]),
        )
        self.states = np.memmap(
            Path(label_files["state_labels"]["path"]),
            dtype=np.int8,
            mode="r",
            shape=tuple(int(value) for value in label_files["state_labels"]["shape"]),
        )
        if candidate_count <= int(self.candidate_ids.max()):
            raise ModelError("model_input_candidate_id_exceeds_label_memmap")
        pack_manifest = _read_json(Path(label_manifest["source"]["pack_manifest"]))
        self.date_values = np.asarray(pack_manifest["date_values"], dtype=str)
        self.feature_records = [dict(item) for item in manifest["features"]]
        self.feature_map = {
            str(item["feature_name"]): item for item in self.feature_records
        }
        self.feature_groups = {
            str(name): list(values)
            for name, values in dict(manifest["feature_groups"]).items()
        }
        self.metadata_mappings = dict(manifest.get("metadata_mappings", {}) or {})
        self._target_cache: dict[str, np.ndarray] = {}
        self._valid_cache: dict[str, np.ndarray] = {}

    def rows_for_year(self, year: int) -> np.ndarray:
        rows = np.flatnonzero(self.years == int(year)).astype(np.int64, copy=False)
        if not rows.size:
            raise ModelError(f"model_input_year_missing:{year}")
        return rows

    def task_values(self, target: str) -> np.ndarray:
        cache = getattr(self, "_target_cache", None)
        if cache is None:
            cache = {}
            self._target_cache = cache
        if target in cache:
            return cache[target]
        if target.startswith("state"):
            values = np.asarray(
                self.states[self.candidate_ids, LABEL_HORIZON_INDEX[STATE_HORIZON]],
                dtype=np.int8,
            )
        else:
            horizon = int(target.rsplit("_", 1)[1])
            label = "mfe" if target.startswith("mfe") else "pre_peak_mae"
            column = self.label_manifest["label_columns"].index(f"{label}_{horizon}")
            values = np.asarray(
                self.labels[self.candidate_ids, column], dtype=np.float32
            )
        cache[target] = values
        return values

    def valid_mask(self, target: str) -> np.ndarray:
        cache = getattr(self, "_valid_cache", None)
        if cache is None:
            cache = {}
            self._valid_cache = cache
        if target in cache:
            return cache[target]
        horizon_index = LABEL_HORIZON_INDEX[int(target.rsplit("_", 1)[1])]
        flags = np.asarray(
            self.flags[self.candidate_ids, horizon_index], dtype=np.uint16
        )
        if target.startswith("state"):
            state = self.task_values(target)
            valid = ((flags & FLAG_STATE_ASSIGNED) != 0) & np.isin(state, (0, 1, 2))
        else:
            values = self.task_values(target)
            valid = ((flags & FLAG_MFE_PRE_PEAK_MAE_VALID) != 0) & np.isfinite(values)
        cache[target] = valid
        return valid

    def fold(self, *, year: int, horizon: int, target: str) -> dict[str, Any]:
        year_rows = self.rows_for_year(year)
        oos_start_date_idx = int(self.date_idx[year_rows[0]])
        oos_end_date_idx = int(self.date_idx[year_rows[-1]])
        maximum_train_signal_date_idx = oos_start_date_idx - int(horizon) - 1
        if maximum_train_signal_date_idx < int(self.date_idx[0]):
            raise ModelError("fold_purge_leaves_no_training_history")
        valid = self.valid_mask(target)
        train_rows = np.flatnonzero(
            (self.date_idx <= maximum_train_signal_date_idx) & valid
        ).astype(np.int64, copy=False)
        evaluation_rows = year_rows[valid[year_rows]]
        if not train_rows.size or not evaluation_rows.size:
            raise ModelError(f"fold_empty:{target}:{year}")
        if int(self.date_idx[train_rows[-1]]) + int(horizon) >= oos_start_date_idx:
            raise ModelError("horizon_purge_contract_failed")
        return {
            "year": int(year),
            "horizon": int(horizon),
            "target": target,
            "oos_start_date_idx": oos_start_date_idx,
            "oos_end_date_idx": oos_end_date_idx,
            "maximum_train_signal_date_idx": maximum_train_signal_date_idx,
            "train_rows": train_rows,
            "evaluation_rows": evaluation_rows,
        }

    def layout(self, feature_names: Sequence[str]) -> dict[str, Any]:
        names = [str(value) for value in feature_names]
        if not names or len(set(names)) != len(names):
            raise ModelError("model_feature_names_empty_or_duplicate")
        records = []
        for name in names:
            if name not in self.feature_map:
                raise ModelError(f"model_feature_not_registered:{name}")
            records.append(self.feature_map[name])
        positions: dict[str, list[tuple[int, int]]] = {
            "base": [],
            "extra": [],
            "availability": [],
        }
        for position, record in enumerate(records):
            positions[str(record["storage"])].append(
                (position, int(record["column_index"]))
            )
        categorical_positions = [
            position
            for position, record in enumerate(records)
            if record["storage"] == "availability"
        ]
        return {
            "feature_names": names,
            "records": records,
            "positions": positions,
            "categorical_positions": categorical_positions,
        }

    def availability_mask(self, family: str, rows: np.ndarray) -> np.ndarray:
        local = np.asarray(rows, dtype=np.int64)

        def values(name: str) -> np.ndarray:
            record = self.feature_map[name]
            if record["storage"] != "availability":
                raise ModelError(f"availability_feature_not_metadata:{name}")
            return np.asarray(
                self.availability[int(record["column_index"]), local], dtype=np.int8
            )

        if family == "margin":
            return values("margin_eligibility_eligible") == 1
        if family == "research":
            names = (
                "research_metadata_source_covered",
                "research_forecast_source_complete",
                "research_forecast_source_partial",
            )
            result = np.zeros(len(local), dtype=bool)
            for name in names:
                if (
                    name in self.feature_map
                    and self.feature_map[name]["storage"] == "extra"
                ):
                    value = np.asarray(
                        self.extra[int(self.feature_map[name]["column_index"]), local]
                    )
                    result |= np.isfinite(value) & (value > 0)
            return result
        if family == "financial_extensions":
            conflict = values("balance_extension_source_conflict")
            result = np.zeros(len(local), dtype=bool)
            for name in self.feature_groups["financial_extensions"]:
                record = self.feature_map[name]
                value = np.asarray(self.extra[int(record["column_index"]), local])
                result |= np.isfinite(value)
            return result & (conflict != 1)
        if family == "all_gated":
            return (
                self.availability_mask("margin", local)
                | self.availability_mask("research", local)
                | self.availability_mask("financial_extensions", local)
            )
        raise ModelError(f"unknown_availability_family:{family}")


def _make_sequence(
    *, inputs: ModelInputs, rows: np.ndarray, layout: Mapping[str, Any], batch_size: int
) -> Any:
    import lightgbm as lgb

    selected_rows = np.asarray(rows, dtype=np.int64)
    positions = dict(layout["positions"])

    class MatrixSequence(lgb.Sequence):
        def __init__(self) -> None:
            self.batch_size = int(batch_size)

        def __len__(self) -> int:
            return len(selected_rows)

        def _block(self, local: np.ndarray) -> np.ndarray:
            local = np.asarray(local, dtype=np.int64)
            global_rows = selected_rows[local]
            output = np.empty(
                (len(local), len(layout["feature_names"])), dtype=np.float64
            )
            for storage, pairs in positions.items():
                if not pairs:
                    continue
                target_positions = np.asarray(
                    [pair[0] for pair in pairs], dtype=np.int64
                )
                columns = np.asarray([pair[1] for pair in pairs], dtype=np.int32)
                if storage == "base":
                    values = np.asarray(
                        inputs.base[np.ix_(inputs.candidate_ids[global_rows], columns)],
                        dtype=np.float64,
                    )
                elif storage == "extra":
                    values = np.asarray(
                        inputs.extra[np.ix_(columns, global_rows)].T, dtype=np.float64
                    )
                else:
                    values = np.asarray(
                        inputs.availability[np.ix_(columns, global_rows)].T,
                        dtype=np.float64,
                    )
                output[:, target_positions] = values
            return output

        def __getitem__(self, index: Any) -> np.ndarray:
            if isinstance(index, slice):
                start = 0 if index.start is None else int(index.start)
                stop = len(selected_rows) if index.stop is None else int(index.stop)
                step = 1 if index.step is None else int(index.step)
                return self._block(np.arange(start, stop, step, dtype=np.int64))
            if isinstance(index, (list, tuple, np.ndarray)):
                return self._block(np.asarray(index, dtype=np.int64))
            if isinstance(index, (int, np.integer)):
                return self._block(np.asarray([int(index)], dtype=np.int64))[0]
            raise TypeError(f"unsupported_sequence_index:{type(index).__name__}")

    return MatrixSequence()


def _target_parameters(config: Mapping[str, Any], kind: str) -> dict[str, Any]:
    model = dict(config["model"])
    parameters: dict[str, Any] = {
        "boosting_type": "gbdt",
        "device_type": "cpu",
        "learning_rate": float(model["learning_rate"]),
        "num_leaves": int(model["num_leaves"]),
        "max_depth": int(model["max_depth"]),
        "min_data_in_leaf": int(model["min_data_in_leaf"]),
        "lambda_l2": float(model["lambda_l2"]),
        "feature_fraction": float(model["feature_fraction"]),
        "bagging_fraction": float(model["bagging_fraction"]),
        "bagging_freq": int(model["bagging_freq"]),
        "max_bin": int(model["max_bin"]),
        "num_threads": int(model["num_threads"]),
        "histogram_pool_size": int(model["histogram_pool_size_mb"]),
        "deterministic": True,
        "force_col_wise": True,
        "seed": int(model["seed"]),
        "feature_fraction_seed": int(model["seed"]),
        "bagging_seed": int(model["seed"]),
        "data_random_seed": int(model["seed"]),
        "feature_pre_filter": False,
        "verbosity": -1,
    }
    if kind in {"mfe", "risk"}:
        parameters.update(
            {
                "objective": "huber",
                "metric": "huber",
                "alpha": float(model["huber_alpha"]),
            }
        )
    elif kind == "state":
        parameters.update(
            {"objective": "multiclass", "metric": "multi_logloss", "num_class": 3}
        )
    else:
        raise ModelError(f"unknown_model_kind:{kind}")
    return parameters


def _aligned_labels(
    *, inputs: ModelInputs, target: str, rows: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = np.asarray(inputs.task_values(target)[rows])
    valid = np.asarray(inputs.valid_mask(target)[rows], dtype=bool)
    date_idx = inputs.date_idx[rows]
    weights = np.zeros(len(rows), dtype=np.float32)
    if not bool(valid.any()):
        raise ModelError(f"target_has_no_valid_rows:{target}")
    weights[valid] = learnability.date_equal_weights(date_idx[valid])
    labels = values.copy()
    if target.startswith("state"):
        labels[~valid] = 0
        labels = labels.astype(np.int32, copy=False)
    else:
        labels[~valid] = 0.0
        labels = labels.astype(np.float32, copy=False)
    return labels, weights, values


def _build_datasets(
    *,
    inputs: ModelInputs,
    config: Mapping[str, Any],
    target: str,
    feature_names: Sequence[str],
    train_rows: np.ndarray,
    evaluation_rows: np.ndarray,
) -> dict[str, Any]:
    import lightgbm as lgb

    layout = inputs.layout(feature_names)
    train_label, train_weight, _ = _aligned_labels(
        inputs=inputs, target=target, rows=train_rows
    )
    evaluation_label, evaluation_weight, evaluation_raw = _aligned_labels(
        inputs=inputs, target=target, rows=evaluation_rows
    )
    batch_size = int(config["model"]["sequence_batch_size"])
    train_sequence = _make_sequence(
        inputs=inputs, rows=train_rows, layout=layout, batch_size=batch_size
    )
    evaluation_sequence = _make_sequence(
        inputs=inputs, rows=evaluation_rows, layout=layout, batch_size=batch_size
    )
    train_sequence = learnability._memory_trimmed_sequence(train_sequence, config)
    evaluation_sequence = learnability._memory_trimmed_sequence(
        evaluation_sequence, config
    )
    construction = {
        "max_bin": int(config["model"]["max_bin"]),
        "data_random_seed": int(config["model"]["seed"]),
        "feature_pre_filter": False,
        "verbosity": -1,
    }
    categorical_positions = list(layout["categorical_positions"])
    train_set = lgb.Dataset(
        train_sequence,
        label=train_label,
        weight=train_weight,
        feature_name=list(feature_names),
        categorical_feature=categorical_positions,
        free_raw_data=True,
        params=construction,
    )
    evaluation_set = lgb.Dataset(
        evaluation_sequence,
        label=evaluation_label,
        weight=evaluation_weight,
        feature_name=list(feature_names),
        categorical_feature=categorical_positions,
        reference=train_set,
        free_raw_data=True,
        params=construction,
    )
    train_set.construct()
    evaluation_set.construct()
    return {
        "layout": layout,
        "train_sequence": train_sequence,
        "evaluation_sequence": evaluation_sequence,
        "train_set": train_set,
        "evaluation_set": evaluation_set,
        "train_label": train_label,
        "train_weight": train_weight,
        "evaluation_label": evaluation_label,
        "evaluation_weight": evaluation_weight,
        "evaluation_raw": evaluation_raw,
    }


def _predict(model: Any, sequence: Any, iterations: int) -> np.ndarray:
    output: np.ndarray | None = None
    for start in range(0, len(sequence), 250_000):
        stop = min(start + 250_000, len(sequence))
        part = np.asarray(
            model.predict(sequence[start:stop], num_iteration=int(iterations))
        )
        if output is None:
            shape = (
                (len(sequence),) if part.ndim == 1 else (len(sequence), part.shape[1])
            )
            output = np.empty(shape, dtype=np.float32)
        output[start:stop] = part.astype(np.float32, copy=False)
    if output is None:
        raise ModelError("empty_prediction_sequence")
    return output


def _safe_spearman(left: np.ndarray, right: np.ndarray) -> float:
    if len(left) < 3:
        return float("nan")
    value = stats.spearmanr(left, right).statistic
    return float(value) if np.isfinite(value) else float("nan")


def _date_boundaries(date_idx: np.ndarray) -> np.ndarray:
    dates = np.asarray(date_idx, dtype=np.int32)
    if dates.ndim != 1 or bool(np.any(dates[1:] < dates[:-1])):
        raise ModelError("metric_rows_not_date_ordered")
    return np.flatnonzero(np.r_[True, dates[1:] != dates[:-1], True])


def _mfe_daily_metrics(
    *,
    date_idx: np.ndarray,
    actual: np.ndarray,
    prediction: np.ndarray,
    risk: np.ndarray,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    records: list[dict[str, Any]] = []
    deciles: list[np.ndarray] = []
    boundaries = _date_boundaries(date_idx)
    for left, right in itertools.pairwise(boundaries):
        y = np.asarray(actual[left:right], dtype=np.float64)
        p = np.asarray(prediction[left:right], dtype=np.float64)
        valid = np.isfinite(y) & np.isfinite(p)
        if int(valid.sum()) < 20:
            continue
        y = y[valid]
        p = p[valid]
        risk_values = np.asarray(risk[left:right], dtype=np.float64)[valid]
        order_pred = np.argsort(p, kind="mergesort")
        order_true = np.argsort(y, kind="mergesort")
        count = len(y)
        top1_count = max(1, math.ceil(0.01 * count))
        top5_count = max(1, math.ceil(0.05 * count))
        top1_pred = order_pred[-top1_count:]
        top5_pred = order_pred[-top5_count:]
        top1_true = set(order_true[-top1_count:].tolist())
        top5_true = set(order_true[-top5_count:].tolist())
        top5_risk = -risk_values[top5_pred]
        top5_risk = top5_risk[np.isfinite(top5_risk)]
        decile_means = np.asarray(
            [float(y[part].mean()) for part in np.array_split(order_pred, 10)],
            dtype=np.float64,
        )
        deciles.append(decile_means)
        records.append(
            {
                "date_idx": int(date_idx[left]),
                "rank_ic": _safe_spearman(y, p),
                "top1_mfe_mean": float(y[top1_pred].mean()),
                "top5_mfe_mean": float(y[top5_pred].mean()),
                "top1_mfe_median": float(np.median(y[top1_pred])),
                "top5_mfe_median": float(np.median(y[top5_pred])),
                "top1_capture": float(
                    len(top1_true.intersection(set(top1_pred.tolist())))
                    / len(top1_true)
                ),
                "top5_capture": float(
                    len(top5_true.intersection(set(top5_pred.tolist())))
                    / len(top5_true)
                ),
                "top5_adverse_median": float(np.median(top5_risk))
                if len(top5_risk)
                else float("nan"),
                "mae": float(np.mean(np.abs(y - p))),
                "row_count": int(count),
            }
        )
    frame = pd.DataFrame(records)
    if frame.empty:
        raise ModelError("mfe_metrics_no_valid_dates")
    decile = np.nanmean(np.asarray(deciles, dtype=np.float64), axis=0)
    summary = {
        "date_count": len(frame),
        "rank_ic": float(frame["rank_ic"].mean()),
        "top1_mfe_mean": float(frame["top1_mfe_mean"].mean()),
        "top5_mfe_mean": float(frame["top5_mfe_mean"].mean()),
        "top1_mfe_median": float(frame["top1_mfe_median"].mean()),
        "top5_mfe_median": float(frame["top5_mfe_median"].mean()),
        "top1_capture": float(frame["top1_capture"].mean()),
        "top5_capture": float(frame["top5_capture"].mean()),
        "top5_adverse_median": float(frame["top5_adverse_median"].mean()),
        "mae": float(frame["mae"].mean()),
        "decile_spearman": _safe_spearman(np.arange(10, dtype=np.float64), decile),
    }
    return frame, summary


def _risk_daily_metrics(
    *, date_idx: np.ndarray, actual: np.ndarray, prediction: np.ndarray
) -> tuple[pd.DataFrame, dict[str, Any]]:
    records: list[dict[str, Any]] = []
    boundaries = _date_boundaries(date_idx)
    for left, right in itertools.pairwise(boundaries):
        y = np.asarray(actual[left:right], dtype=np.float64)
        p = np.asarray(prediction[left:right], dtype=np.float64)
        valid = np.isfinite(y) & np.isfinite(p)
        if int(valid.sum()) < 20:
            continue
        y = y[valid]
        p = p[valid]
        order = np.argsort(y, kind="mergesort")
        count = max(1, math.ceil(0.20 * len(y)))
        event = np.zeros(len(y), dtype=np.int8)
        event[order[:count]] = 1
        records.append(
            {
                "date_idx": int(date_idx[left]),
                "rank_ic": _safe_spearman(p, y),
                "mae": float(np.mean(np.abs(p - y))),
                "deep_adverse_pr_auc": float(average_precision_score(event, -p)),
                "row_count": len(y),
            }
        )
    frame = pd.DataFrame(records)
    if frame.empty:
        raise ModelError("risk_metrics_no_valid_dates")
    return frame, {
        "date_count": len(frame),
        "rank_ic": float(frame["rank_ic"].mean()),
        "mae": float(frame["mae"].mean()),
        "deep_adverse_pr_auc": float(frame["deep_adverse_pr_auc"].mean()),
    }


def _state_daily_metrics(
    *, date_idx: np.ndarray, actual: np.ndarray, prediction: np.ndarray
) -> tuple[pd.DataFrame, dict[str, Any]]:
    records: list[dict[str, Any]] = []
    boundaries = _date_boundaries(date_idx)
    for left, right in itertools.pairwise(boundaries):
        y = np.asarray(actual[left:right], dtype=np.int8)
        p = np.asarray(prediction[left:right], dtype=np.float64)
        valid = np.isin(y, (0, 1, 2)) & np.isfinite(p).all(axis=1)
        if int(valid.sum()) < 20:
            continue
        y = y[valid]
        p = np.clip(p[valid], 1.0e-7, 1.0)
        p /= p.sum(axis=1, keepdims=True)
        expected = p[:, 1] + 2.0 * p[:, 2]
        one_hot = np.eye(3, dtype=np.float64)[y]
        count = max(1, math.ceil(0.05 * len(y)))
        selected = np.argsort(p[:, 2], kind="mergesort")[-count:]
        high = y == 2
        records.append(
            {
                "date_idx": int(date_idx[left]),
                "ordinal_ic": _safe_spearman(expected, y.astype(np.float64)),
                "high_state_top5_lift": float(high[selected].mean() - high.mean()),
                "brier": float(np.mean(np.square(p - one_hot).sum(axis=1))),
                "logloss": float(
                    np.mean(-np.log(np.clip(p[np.arange(len(y)), y], 1.0e-12, 1.0)))
                ),
                "row_count": len(y),
            }
        )
    frame = pd.DataFrame(records)
    if frame.empty:
        raise ModelError("state_metrics_no_valid_dates")
    return frame, {
        "date_count": len(frame),
        "ordinal_ic": float(frame["ordinal_ic"].mean()),
        "high_state_top5_lift": float(frame["high_state_top5_lift"].mean()),
        "brier": float(frame["brier"].mean()),
        "logloss": float(frame["logloss"].mean()),
    }


def _task_plan(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    targets = dict(config["targets"])
    tasks: list[dict[str, Any]] = []

    def add(
        *,
        stage: str,
        target: str,
        year: int,
        variant: str,
        gated_family: str | None = None,
    ) -> None:
        target_config = dict(targets[target])
        task_id = f"{stage}__{target}__{variant}__{year}"
        tasks.append(
            {
                "task_id": task_id,
                "stage": stage,
                "target": target,
                "kind": str(target_config["kind"]),
                "horizon": int(target_config["horizon"]),
                "year": int(year),
                "variant": variant,
                "gated_family": gated_family,
            }
        )

    adaptive_targets = ("risk_10", "risk_20", "state_10")
    for target in adaptive_targets:
        for year in ROLLING_YEARS:
            add(stage="tuning", target=target, year=year, variant="legacy_core")
    for target in ("mfe_10", "mfe_20"):
        for variant in CORE_VARIANTS:
            for year in ROLLING_YEARS:
                add(stage="mfe_core", target=target, year=year, variant=variant)
    for target in ("mfe_10", "mfe_20"):
        for family in GATED_FAMILIES:
            variant = f"selected_core_plus_{family}"
            for year in ROLLING_YEARS:
                add(
                    stage="mfe_gated",
                    target=target,
                    year=year,
                    variant=variant,
                    gated_family=family,
                )
    for target in adaptive_targets:
        for variant in RISK_STATE_CORE_VARIANTS:
            for year in ROLLING_YEARS:
                add(stage="risk_state_core", target=target, year=year, variant=variant)
    for target in adaptive_targets:
        for year in ROLLING_YEARS:
            add(
                stage="risk_state_gated",
                target=target,
                year=year,
                variant=ALL_GATED_VARIANT,
                gated_family="all_gated",
            )
    if len(tasks) != 78 or len({task["task_id"] for task in tasks}) != 78:
        raise ModelError(f"task_plan_contract_mismatch:{len(tasks)}")
    return tasks


def _save_npy(path: Path, values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    with temporary.open("wb") as stream:
        np.save(stream, np.asarray(values), allow_pickle=False)
    os.replace(temporary, path)


def _task_result_path(output_root: Path, task_id: str) -> Path:
    return output_root / "tasks" / task_id / "task_result.json"


def _task_fingerprint(
    *,
    task: Mapping[str, Any],
    feature_names: Sequence[str],
    model_input_fingerprint: str,
    config_sha256: str,
) -> str:
    return _stable_hash(
        {
            "task": dict(task),
            "feature_names": list(feature_names),
            "model_input_fingerprint": model_input_fingerprint,
            "config_sha256": config_sha256,
        }
    )


def _task_complete(path: Path, *, fingerprint: str) -> bool:
    if not path.is_file():
        return False
    try:
        result = _read_json(path)
        if (
            result.get("status") != "completed"
            or result.get("task_fingerprint") != fingerprint
        ):
            return False
        return all(
            Path(record["path"]).is_file()
            and _sha256(Path(record["path"])) == record["sha256"]
            for record in dict(result.get("files", {}) or {}).values()
        )
    except (ModelError, OSError, KeyError, ValueError, TypeError):
        return False


def _metadata_names(inputs: ModelInputs, block: str | None = None) -> list[str]:
    return [
        str(item["feature_name"])
        for item in inputs.feature_records
        if item["storage"] == "availability"
        and (block is None or str(item["block"]) == block)
    ]


def _load_selection(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ModelError(f"required_selection_missing:{path}")
    return _read_json(path)


def _effective_features(
    *,
    inputs: ModelInputs,
    task: Mapping[str, Any],
    output_root: Path,
) -> tuple[list[str], str]:
    variant = str(task["variant"])
    stage = str(task["stage"])
    target = str(task["target"])
    if stage in {"tuning", "mfe_core", "risk_state_core"}:
        return list(inputs.feature_groups[variant]), variant
    if stage == "mfe_gated":
        selection = _load_selection(
            output_root / "selection" / "mfe_core_selection.json"
        )
        base_variant = str(selection["targets"][target]["selected_core"])
        family = str(task["gated_family"])
        names = [*inputs.feature_groups[base_variant], *inputs.feature_groups[family]]
        if family == "margin":
            names.extend(_metadata_names(inputs, "margin_features"))
        elif family == "financial_extensions":
            names.extend(_metadata_names(inputs, "financial_statement_extensions"))
        return list(dict.fromkeys(names)), base_variant
    if stage == "risk_state_gated":
        selection = _load_selection(
            output_root / "selection" / "risk_state_core_selection.json"
        )
        base_variant = str(selection["targets"][target]["selected_core"])
        names = [
            *inputs.feature_groups[base_variant],
            *inputs.feature_groups["all_gated"],
            *_metadata_names(inputs),
        ]
        return list(dict.fromkeys(names)), base_variant
    raise ModelError(f"unknown_task_stage:{stage}")


def _evaluate_prediction(
    *,
    inputs: ModelInputs,
    task: Mapping[str, Any],
    rows: np.ndarray,
    prediction: np.ndarray,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    target = str(task["target"])
    kind = str(task["kind"])
    actual = inputs.task_values(target)[rows]
    if kind == "mfe":
        risk = inputs.task_values(f"risk_{int(task['horizon'])}")[rows]
        return _mfe_daily_metrics(
            date_idx=inputs.date_idx[rows],
            actual=actual,
            prediction=prediction,
            risk=risk,
        )
    if kind == "risk":
        return _risk_daily_metrics(
            date_idx=inputs.date_idx[rows], actual=actual, prediction=prediction
        )
    return _state_daily_metrics(
        date_idx=inputs.date_idx[rows], actual=actual, prediction=prediction
    )


def _observed_subset_metrics(
    *,
    inputs: ModelInputs,
    task: Mapping[str, Any],
    rows: np.ndarray,
    prediction: np.ndarray,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for family in (*GATED_FAMILIES, "all_gated"):
        mask = inputs.availability_mask(family, rows)
        selected_rows = rows[mask]
        if int(mask.sum()) < 100:
            result[family] = {"row_count": int(mask.sum()), "metrics": None}
            continue
        try:
            _, metrics = _evaluate_prediction(
                inputs=inputs,
                task=task,
                rows=selected_rows,
                prediction=np.asarray(prediction)[mask],
            )
        except ModelError:
            metrics = None
        result[family] = {"row_count": int(mask.sum()), "metrics": metrics}
    return result


def _importance_frame(
    *,
    model: Any,
    inputs: ModelInputs,
    feature_names: Sequence[str],
    task: Mapping[str, Any],
) -> pd.DataFrame:
    gain = np.asarray(
        model.feature_importance(importance_type="gain"), dtype=np.float64
    )
    split = np.asarray(
        model.feature_importance(importance_type="split"), dtype=np.float64
    )
    rows: list[dict[str, Any]] = []
    for name, gain_value, split_value in zip(feature_names, gain, split, strict=True):
        record = inputs.feature_map[str(name)]
        family = str(record["analytic_family"])
        if record["storage"] == "availability":
            family = f"availability_{record['block']}"
        rows.append(
            {
                "task_id": str(task["task_id"]),
                "target": str(task["target"]),
                "year": int(task["year"]),
                "variant": str(task["variant"]),
                "feature_family": family,
                "gain": float(gain_value),
                "split": float(split_value),
                "feature_count": 1,
            }
        )
    frame = pd.DataFrame(rows)
    return (
        frame.groupby(
            ["task_id", "target", "year", "variant", "feature_family"],
            as_index=False,
        )[["gain", "split", "feature_count"]]
        .sum()
        .sort_values(["task_id", "gain"], ascending=[True, False])
    )


def _release_datasets(datasets: dict[str, Any]) -> None:
    datasets["train_set"] = None
    datasets["evaluation_set"] = None
    datasets["train_sequence"] = None
    datasets["evaluation_sequence"] = None
    gc.collect()


def _run_training_task(
    *,
    task: Mapping[str, Any],
    inputs: ModelInputs,
    config: Mapping[str, Any],
    output_root: Path,
    config_sha256: str,
) -> dict[str, Any]:
    import lightgbm as lgb

    feature_names, base_variant = _effective_features(
        inputs=inputs, task=task, output_root=output_root
    )
    fingerprint = _task_fingerprint(
        task=task,
        feature_names=feature_names,
        model_input_fingerprint=str(inputs.manifest["input_fingerprint"]),
        config_sha256=config_sha256,
    )
    result_path = _task_result_path(output_root, str(task["task_id"]))
    if _task_complete(result_path, fingerprint=fingerprint):
        return _read_json(result_path)
    outer_year = int(task["year"])
    evaluation_year = outer_year - 1 if task["stage"] == "tuning" else outer_year
    fold = inputs.fold(
        year=evaluation_year,
        horizon=int(task["horizon"]),
        target=str(task["target"]),
    )
    datasets = _build_datasets(
        inputs=inputs,
        config=config,
        target=str(task["target"]),
        feature_names=feature_names,
        train_rows=fold["train_rows"],
        evaluation_rows=fold["evaluation_rows"],
    )
    parameters = _target_parameters(config, str(task["kind"]))
    started = time.perf_counter()
    if task["stage"] == "tuning":
        model = lgb.train(
            parameters,
            datasets["train_set"],
            num_boost_round=int(config["model"]["adaptive_max_rounds"]),
            valid_sets=[datasets["evaluation_set"]],
            valid_names=["inner_validation"],
            callbacks=[
                lgb.early_stopping(
                    int(config["model"]["adaptive_patience"]),
                    first_metric_only=True,
                    verbose=False,
                )
            ],
        )
        iterations = int(model.best_iteration)
    else:
        if task["kind"] == "mfe":
            iterations = int(config["model"]["fixed_mfe_rounds"])
        else:
            tuning_id = f"tuning__{task['target']}__legacy_core__{outer_year}"
            tuning = _read_json(_task_result_path(output_root, tuning_id))
            iterations = int(tuning["best_iteration"])
        model = lgb.train(
            parameters,
            datasets["train_set"],
            num_boost_round=iterations,
            valid_sets=[datasets["evaluation_set"]],
            valid_names=["outer_evaluation"],
        )
    training_seconds = float(time.perf_counter() - started)
    prediction_rows = inputs.rows_for_year(evaluation_year)
    prediction_sequence = _make_sequence(
        inputs=inputs,
        rows=prediction_rows,
        layout=datasets["layout"],
        batch_size=int(config["model"]["sequence_batch_size"]),
    )
    prediction = _predict(model, prediction_sequence, iterations)
    evaluation_positions = np.searchsorted(prediction_rows, fold["evaluation_rows"])
    if not np.array_equal(
        prediction_rows[evaluation_positions], fold["evaluation_rows"]
    ):
        raise ModelError("evaluation_prediction_row_alignment_failed")
    evaluation_prediction = prediction[evaluation_positions]
    daily, metrics = _evaluate_prediction(
        inputs=inputs,
        task=task,
        rows=fold["evaluation_rows"],
        prediction=evaluation_prediction,
    )
    daily.insert(
        1,
        "trade_date",
        [str(inputs.date_values[int(value)]) for value in daily["date_idx"]],
    )
    observed = _observed_subset_metrics(
        inputs=inputs,
        task=task,
        rows=fold["evaluation_rows"],
        prediction=evaluation_prediction,
    )
    task_dir = result_path.parent
    task_dir.mkdir(parents=True, exist_ok=True)
    model_path = task_dir / "model.txt"
    model_partial = model_path.with_suffix(".txt.partial")
    model.save_model(str(model_partial), num_iteration=iterations)
    os.replace(model_partial, model_path)
    prediction_path = task_dir / "prediction.npy"
    candidate_path = task_dir / "candidate_id.npy"
    daily_path = task_dir / "daily_metrics.parquet"
    importance_path = task_dir / "family_importance.parquet"
    _save_npy(prediction_path, prediction)
    _save_npy(candidate_path, inputs.candidate_ids[prediction_rows])
    _write_parquet(daily, daily_path)
    importance = _importance_frame(
        model=model, inputs=inputs, feature_names=feature_names, task=task
    )
    _write_parquet(importance, importance_path)
    result = {
        "schema": "seq100_quality_liquidity_model_task/1",
        "status": "completed",
        "completed_at": _now(),
        "study_id": STUDY_ID,
        "task_id": str(task["task_id"]),
        "task_fingerprint": fingerprint,
        "stage": str(task["stage"]),
        "target": str(task["target"]),
        "kind": str(task["kind"]),
        "horizon": int(task["horizon"]),
        "model_year": outer_year,
        "evaluation_year": evaluation_year,
        "variant": str(task["variant"]),
        "base_variant": base_variant,
        "gated_family": task.get("gated_family"),
        "feature_count": len(feature_names),
        "feature_names": feature_names,
        "availability_feature_count": int(
            sum(
                inputs.feature_map[name]["storage"] == "availability"
                for name in feature_names
            )
        ),
        "train_row_count": len(fold["train_rows"]),
        "evaluation_row_count": len(fold["evaluation_rows"]),
        "candidate_prediction_row_count": len(prediction_rows),
        "maximum_train_signal_date_idx": int(fold["maximum_train_signal_date_idx"]),
        "maximum_train_signal_date": str(
            inputs.date_values[int(fold["maximum_train_signal_date_idx"])]
        ),
        "purge_days": int(task["horizon"]),
        "best_iteration": iterations,
        "training_seconds": training_seconds,
        "parameters": parameters,
        "metrics": metrics,
        "observed_subset_metrics": observed,
        "files": {
            "model": _file_record(model_path),
            "prediction": _file_record(
                prediction_path,
                shape=list(prediction.shape),
                dtype=str(prediction.dtype),
            ),
            "candidate_id": _file_record(
                candidate_path,
                shape=[len(prediction_rows)],
                dtype=str(inputs.candidate_ids.dtype),
            ),
            "daily_metrics": _file_record(daily_path, row_count=len(daily)),
            "family_importance": _file_record(
                importance_path, row_count=len(importance)
            ),
        },
    }
    _write_json(result_path, result)
    _release_datasets(datasets)
    del model, prediction, evaluation_prediction, prediction_sequence
    gc.collect()
    return result


def _completed_results(output_root: Path) -> dict[str, dict[str, Any]]:
    results: dict[str, dict[str, Any]] = {}
    for path in sorted((output_root / "tasks").glob("*/task_result.json")):
        result = _read_json(path)
        if result.get("status") == "completed":
            results[str(result["task_id"])] = result
    return results


def _results_for(
    results: Mapping[str, Mapping[str, Any]], *, target: str, variant: str
) -> dict[int, dict[str, Any]]:
    selected: dict[int, dict[str, Any]] = {}
    for result in results.values():
        if (
            result.get("target") == target
            and result.get("variant") == variant
            and result.get("stage") not in {"tuning"}
        ):
            selected[int(result["model_year"])] = dict(result)
    return selected


def _metric_deltas(
    candidate: Mapping[int, Mapping[str, Any]],
    reference: Mapping[int, Mapping[str, Any]],
    metric: str,
) -> list[float]:
    return [
        float(candidate[year]["metrics"][metric])
        - float(reference[year]["metrics"][metric])
        for year in ROLLING_YEARS
    ]


def _observed_deltas(
    candidate: Mapping[int, Mapping[str, Any]],
    reference: Mapping[int, Mapping[str, Any]],
    family: str,
    metric: str,
) -> list[float] | None:
    values: list[float] = []
    for year in ROLLING_YEARS:
        candidate_metrics = (
            candidate[year].get("observed_subset_metrics", {}).get(family, {})
        )
        reference_metrics = (
            reference[year].get("observed_subset_metrics", {}).get(family, {})
        )
        if not candidate_metrics.get("metrics") or not reference_metrics.get("metrics"):
            return None
        values.append(
            float(candidate_metrics["metrics"][metric])
            - float(reference_metrics["metrics"][metric])
        )
    return values


def _mfe_gate(
    *,
    candidate: Mapping[int, Mapping[str, Any]],
    reference: Mapping[int, Mapping[str, Any]],
    config: Mapping[str, Any],
    observed_family: str | None = None,
) -> dict[str, Any]:
    rules = dict(config["selection"]["mfe"])
    rank = _metric_deltas(candidate, reference, "rank_ic")
    top5_mfe = _metric_deltas(candidate, reference, "top5_mfe_mean")
    capture = _metric_deltas(candidate, reference, "top5_capture")
    top1 = _metric_deltas(candidate, reference, "top1_capture")
    adverse = _metric_deltas(candidate, reference, "top5_adverse_median")
    checks = {
        "rank_deltas": rank,
        "rank_improved_years": sum(value > 0 for value in rank),
        "rank_median_delta": float(np.median(rank)),
        "rank_worst_delta": float(min(rank)),
        "top5_mfe_deltas": top5_mfe,
        "top5_mfe_improved_years": sum(value > 0 for value in top5_mfe),
        "top5_mfe_worst_delta": float(min(top5_mfe)),
        "top5_capture_deltas": capture,
        "top5_capture_improved_years": sum(value > 0 for value in capture),
        "top5_capture_worst_delta": float(min(capture)),
        "top5_adverse_worst_increase": float(max(adverse)),
        "top5_adverse_deltas": adverse,
        "top1_capture_deltas": top1,
    }
    passed = bool(
        checks["rank_improved_years"] >= int(rules["minimum_improved_years"])
        and checks["rank_median_delta"] >= float(rules["minimum_rank_ic_median_delta"])
        and checks["rank_worst_delta"] >= float(rules["minimum_worst_rank_ic_delta"])
        and checks["top5_mfe_improved_years"]
        >= int(rules["minimum_top5_mfe_improved_years"])
        and checks["top5_mfe_worst_delta"]
        >= -float(rules["maximum_worst_top5_mfe_decline"])
        and checks["top5_capture_improved_years"]
        >= int(rules["minimum_top5_capture_improved_years"])
        and checks["top5_capture_worst_delta"]
        >= -float(rules["maximum_worst_top5_capture_decline"])
        and checks["top5_adverse_worst_increase"]
        <= float(rules["maximum_worst_top5_adverse_median_increase"])
        and (
            not bool(rules["top1_cannot_decline_all_years"])
            or not all(value < 0 for value in top1)
        )
    )
    if observed_family is not None:
        observed_rank = _observed_deltas(
            candidate, reference, observed_family, "rank_ic"
        )
        observed_capture = _observed_deltas(
            candidate, reference, observed_family, "top5_capture"
        )
        checks["observed_rank_ic_deltas"] = observed_rank
        checks["observed_top5_capture_deltas"] = observed_capture
        observed_direction = (
            observed_capture if observed_capture is not None else observed_rank
        )
        passed = bool(
            passed
            and observed_direction is not None
            and sum(value >= 0 for value in observed_direction)
            >= int(config["selection"]["gated_observed_minimum_nonreversed_years"])
        )
    checks["passed"] = passed
    return checks


def _risk_gate(
    *,
    candidate: Mapping[int, Mapping[str, Any]],
    reference: Mapping[int, Mapping[str, Any]],
    config: Mapping[str, Any],
    observed_family: str | None = None,
) -> dict[str, Any]:
    rules = dict(config["selection"]["risk"])
    rank = _metric_deltas(candidate, reference, "rank_ic")
    mae_harm = [
        (
            float(candidate[year]["metrics"]["mae"])
            - float(reference[year]["metrics"]["mae"])
        )
        / max(abs(float(reference[year]["metrics"]["mae"])), 1.0e-12)
        for year in ROLLING_YEARS
    ]
    pr_auc = _metric_deltas(candidate, reference, "deep_adverse_pr_auc")
    checks = {
        "rank_median_delta": float(np.median(rank)),
        "rank_worst_delta": float(min(rank)),
        "relative_mae_harm": mae_harm,
        "relative_mae_median": float(np.median(mae_harm)),
        "relative_mae_worst": float(max(mae_harm)),
        "deep_adverse_pr_auc_deltas": pr_auc,
        "deep_adverse_pr_auc_nonnegative_years": sum(value >= 0 for value in pr_auc),
        "deep_adverse_pr_auc_worst_delta": float(min(pr_auc)),
    }
    passed = bool(
        checks["rank_median_delta"] >= float(rules["minimum_rank_ic_median_delta"])
        and checks["rank_worst_delta"] >= float(rules["minimum_worst_rank_ic_delta"])
        and checks["relative_mae_median"]
        <= float(rules["maximum_median_relative_mae_harm"])
        and checks["relative_mae_worst"]
        <= float(rules["maximum_worst_relative_mae_harm"])
        and checks["deep_adverse_pr_auc_nonnegative_years"]
        >= int(rules["minimum_nonnegative_deep_adverse_pr_auc_years"])
        and checks["deep_adverse_pr_auc_worst_delta"]
        >= -float(rules["maximum_worst_deep_adverse_pr_auc_decline"])
    )
    if observed_family is not None:
        observed = _observed_deltas(candidate, reference, observed_family, "rank_ic")
        checks["observed_rank_ic_deltas"] = observed
        passed = bool(
            passed
            and observed is not None
            and sum(value >= 0 for value in observed)
            >= int(config["selection"]["gated_observed_minimum_nonreversed_years"])
        )
    checks["passed"] = passed
    return checks


def _state_gate(
    *,
    candidate: Mapping[int, Mapping[str, Any]],
    reference: Mapping[int, Mapping[str, Any]],
    config: Mapping[str, Any],
    observed_family: str | None = None,
) -> dict[str, Any]:
    rules = dict(config["selection"]["state"])
    ordinal = _metric_deltas(candidate, reference, "ordinal_ic")
    high = _metric_deltas(candidate, reference, "high_state_top5_lift")
    brier_harm = [
        (
            float(candidate[year]["metrics"]["brier"])
            - float(reference[year]["metrics"]["brier"])
        )
        / max(abs(float(reference[year]["metrics"]["brier"])), 1.0e-12)
        for year in ROLLING_YEARS
    ]
    logloss_harm = [
        (
            float(candidate[year]["metrics"]["logloss"])
            - float(reference[year]["metrics"]["logloss"])
        )
        / max(abs(float(reference[year]["metrics"]["logloss"])), 1.0e-12)
        for year in ROLLING_YEARS
    ]
    checks = {
        "ordinal_ic_median_delta": float(np.median(ordinal)),
        "ordinal_ic_worst_delta": float(min(ordinal)),
        "high_state_top5_deltas": high,
        "high_state_top5_nonnegative_years": sum(value >= 0 for value in high),
        "high_state_top5_worst_delta": float(min(high)),
        "relative_brier_harm": brier_harm,
        "relative_brier_median": float(np.median(brier_harm)),
        "relative_brier_worst": float(max(brier_harm)),
        "relative_logloss_harm": logloss_harm,
        "relative_logloss_median": float(np.median(logloss_harm)),
        "relative_logloss_worst": float(max(logloss_harm)),
    }
    passed = bool(
        checks["ordinal_ic_median_delta"]
        >= float(rules["minimum_ordinal_ic_median_delta"])
        and checks["ordinal_ic_worst_delta"]
        >= float(rules["minimum_worst_ordinal_ic_delta"])
        and checks["high_state_top5_nonnegative_years"]
        >= int(rules["minimum_nonnegative_high_state_top5_years"])
        and checks["high_state_top5_worst_delta"]
        >= -float(rules["maximum_worst_high_state_top5_decline"])
        and checks["relative_brier_median"]
        <= float(rules["maximum_median_relative_brier_harm"])
        and checks["relative_brier_worst"]
        <= float(rules["maximum_worst_relative_brier_harm"])
        and checks["relative_logloss_median"]
        <= float(rules["maximum_median_relative_logloss_harm"])
        and checks["relative_logloss_worst"]
        <= float(rules["maximum_worst_relative_logloss_harm"])
    )
    if observed_family is not None:
        observed = _observed_deltas(candidate, reference, observed_family, "ordinal_ic")
        checks["observed_ordinal_ic_deltas"] = observed
        passed = bool(
            passed
            and observed is not None
            and sum(value >= 0 for value in observed)
            >= int(config["selection"]["gated_observed_minimum_nonreversed_years"])
        )
    checks["passed"] = passed
    return checks


def _write_selection(path: Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    _write_json(path, payload)
    return dict(payload)


def _select_mfe_core(
    *,
    results: Mapping[str, Mapping[str, Any]],
    config: Mapping[str, Any],
    output_root: Path,
    inputs: ModelInputs,
) -> dict[str, Any]:
    output: dict[str, Any] = {
        "schema": "seq100_quality_liquidity_mfe_core_selection/1",
        "targets": {},
    }
    for target in ("mfe_10", "mfe_20"):
        reference = _results_for(results, target=target, variant="legacy_core")
        if set(reference) != set(ROLLING_YEARS):
            raise ModelError(f"mfe_core_reference_incomplete:{target}")
        candidates: list[dict[str, Any]] = []
        for variant in CORE_VARIANTS[1:]:
            candidate = _results_for(results, target=target, variant=variant)
            if set(candidate) != set(ROLLING_YEARS):
                raise ModelError(f"mfe_core_candidate_incomplete:{target}:{variant}")
            gate = _mfe_gate(candidate=candidate, reference=reference, config=config)
            candidates.append(
                {
                    "variant": variant,
                    "feature_count": len(inputs.feature_groups[variant]),
                    "gate": gate,
                }
            )
        passed = [item for item in candidates if item["gate"]["passed"]]
        if passed:
            selected = max(
                passed,
                key=lambda item: (
                    float(np.median(item["gate"]["top5_capture_deltas"])),
                    float(np.median(item["gate"]["top1_capture_deltas"])),
                    float(item["gate"]["rank_median_delta"]),
                    -int(item["feature_count"]),
                ),
            )
            selected_variant = str(selected["variant"])
        else:
            selected_variant = "legacy_core"
        output["targets"][target] = {
            "selected_core": selected_variant,
            "candidates": candidates,
            "reference_variant": "legacy_core",
        }
    return _write_selection(
        output_root / "selection" / "mfe_core_selection.json", output
    )


def _select_mfe_gated(
    *,
    results: Mapping[str, Mapping[str, Any]],
    config: Mapping[str, Any],
    output_root: Path,
    inputs: ModelInputs,
) -> dict[str, Any]:
    core = _load_selection(output_root / "selection" / "mfe_core_selection.json")
    output: dict[str, Any] = {
        "schema": "seq100_quality_liquidity_mfe_head_selection/1",
        "targets": {},
    }
    for target in ("mfe_10", "mfe_20"):
        base_variant = str(core["targets"][target]["selected_core"])
        reference = _results_for(results, target=target, variant=base_variant)
        candidates: list[dict[str, Any]] = []
        for family in GATED_FAMILIES:
            variant = f"selected_core_plus_{family}"
            candidate = _results_for(results, target=target, variant=variant)
            gate = _mfe_gate(
                candidate=candidate,
                reference=reference,
                config=config,
                observed_family=family,
            )
            candidates.append(
                {
                    "variant": variant,
                    "family": family,
                    "feature_count": len(inputs.feature_groups[base_variant])
                    + len(inputs.feature_groups[family])
                    + (
                        len(_metadata_names(inputs, "margin_features"))
                        if family == "margin"
                        else len(
                            _metadata_names(inputs, "financial_statement_extensions")
                        )
                        if family == "financial_extensions"
                        else 0
                    ),
                    "gate": gate,
                }
            )
        passed = [item for item in candidates if item["gate"]["passed"]]
        if passed:
            selected = max(
                passed,
                key=lambda item: (
                    float(np.median(item["gate"]["top5_capture_deltas"])),
                    float(np.median(item["gate"]["top1_capture_deltas"])),
                    float(item["gate"]["rank_median_delta"]),
                    -int(item["feature_count"]),
                ),
            )
            selected_variant = str(selected["variant"])
            selected_family = str(selected["family"])
        else:
            selected_variant = base_variant
            selected_family = None
        output["targets"][target] = {
            "selected_core": base_variant,
            "selected_variant": selected_variant,
            "selected_gated_family": selected_family,
            "candidates": candidates,
        }
    return _write_selection(
        output_root / "selection" / "mfe_selected_heads.json", output
    )


def _select_risk_state_core(
    *,
    results: Mapping[str, Mapping[str, Any]],
    config: Mapping[str, Any],
    output_root: Path,
) -> dict[str, Any]:
    output: dict[str, Any] = {
        "schema": "seq100_quality_liquidity_risk_state_core_selection/1",
        "targets": {},
    }
    for target in ("risk_10", "risk_20", "state_10"):
        reference = _results_for(results, target=target, variant="legacy_core")
        candidate = _results_for(results, target=target, variant="full_core")
        if target.startswith("risk"):
            gate = _risk_gate(candidate=candidate, reference=reference, config=config)
        else:
            gate = _state_gate(candidate=candidate, reference=reference, config=config)
        selected = "full_core" if gate["passed"] else "legacy_core"
        output["targets"][target] = {
            "selected_core": selected,
            "reference_variant": "legacy_core",
            "full_core_gate": gate,
        }
    return _write_selection(
        output_root / "selection" / "risk_state_core_selection.json", output
    )


def _select_risk_state_gated(
    *,
    results: Mapping[str, Mapping[str, Any]],
    config: Mapping[str, Any],
    output_root: Path,
) -> dict[str, Any]:
    core = _load_selection(output_root / "selection" / "risk_state_core_selection.json")
    output: dict[str, Any] = {
        "schema": "seq100_quality_liquidity_risk_state_selection/1",
        "targets": {},
    }
    for target in ("risk_10", "risk_20", "state_10"):
        base_variant = str(core["targets"][target]["selected_core"])
        reference = _results_for(results, target=target, variant=base_variant)
        candidate = _results_for(results, target=target, variant=ALL_GATED_VARIANT)
        if target.startswith("risk"):
            gate = _risk_gate(
                candidate=candidate,
                reference=reference,
                config=config,
                observed_family="all_gated",
            )
        else:
            gate = _state_gate(
                candidate=candidate,
                reference=reference,
                config=config,
                observed_family="all_gated",
            )
        selected = ALL_GATED_VARIANT if gate["passed"] else base_variant
        output["targets"][target] = {
            "selected_core": base_variant,
            "selected_variant": selected,
            "full_gated_gate": gate,
        }
    return _write_selection(
        output_root / "selection" / "risk_state_selected_heads.json", output
    )


def _selected_head_contract(output_root: Path, inputs: ModelInputs) -> dict[str, Any]:
    mfe = _load_selection(output_root / "selection" / "mfe_selected_heads.json")
    risk = _load_selection(output_root / "selection" / "risk_state_selected_heads.json")
    targets: dict[str, Any] = {}
    for target, item in {**mfe["targets"], **risk["targets"]}.items():
        variant = str(item["selected_variant"])
        if variant in inputs.feature_groups:
            count = len(inputs.feature_groups[variant])
        elif variant.startswith("selected_core_plus_"):
            base_variant = str(item["selected_core"])
            family = str(item.get("selected_gated_family") or "all_gated")
            count = len(
                {
                    *inputs.feature_groups[base_variant],
                    *inputs.feature_groups[family],
                }
            )
            if family == "margin":
                count += len(_metadata_names(inputs, "margin_features"))
            elif family == "financial_extensions":
                count += len(_metadata_names(inputs, "financial_statement_extensions"))
            elif family == "all_gated":
                count += len(_metadata_names(inputs))
        else:
            count = None
        targets[target] = {**item, "feature_count": count}
    return {
        "schema": "seq100_quality_liquidity_selected_head_contract/1",
        "study_id": STUDY_ID,
        "targets": targets,
        "training_semantics": "retrospective_rolling_oos",
        "training_performed": True,
    }


def _root_manifest(
    *,
    output_root: Path,
    config_path: Path,
    input_manifest: Mapping[str, Any],
    input_manifest_path: Path,
    tasks: Sequence[Mapping[str, Any]],
    status: str,
    files: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "schema": "seq100_quality_liquidity_model_manifest/1",
        "study_id": STUDY_ID,
        "status": status,
        "created_at": _now(),
        "config": _file_record(config_path),
        "model_inputs": _file_record(
            input_manifest_path,
            input_fingerprint=str(input_manifest.get("input_fingerprint", "")),
        ),
        "task_count": len(tasks),
        "tasks": [dict(task) for task in tasks],
        "outputs": dict(files or {}),
        "training_performed": status in {"training_completed", "evaluated", "audited"},
        "feature_set_selected": status in {"evaluated", "audited"},
        "evaluation_semantics": "retrospective_rolling_oos",
        "forbidden_year": FORBIDDEN_YEAR,
    }


def _update_ledger(
    *,
    output_root: Path,
    tasks: Sequence[Mapping[str, Any]],
    results: Mapping[str, Mapping[str, Any]],
) -> None:
    entries = []
    for task in tasks:
        result = results.get(str(task["task_id"]))
        entries.append(
            {
                **dict(task),
                "status": "completed" if result is not None else "pending",
                "result_path": str(
                    _task_result_path(output_root, str(task["task_id"])).resolve()
                )
                if result is not None
                else None,
                "best_iteration": result.get("best_iteration") if result else None,
                "updated_at": _now(),
            }
        )
    _write_json(
        output_root / "task_ledger.json",
        {
            "schema": "seq100_quality_liquidity_model_task_ledger/1",
            "study_id": STUDY_ID,
            "task_count": len(tasks),
            "completed_count": sum(item["status"] == "completed" for item in entries),
            "tasks": entries,
        },
    )


def run(
    *,
    study_path: Path = DEFAULT_STUDY_PATH,
    ready_root: Path = DEFAULT_READY_ROOT,
    audit_root: Path = DEFAULT_AUDIT_ROOT,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    max_tasks: int | None = None,
) -> dict[str, Any]:
    config = _load_config(study_path)
    input_manifest = prepare(
        study_path=study_path,
        ready_root=ready_root,
        audit_root=audit_root,
        output_root=output_root,
    )
    inputs = ModelInputs(input_manifest)
    tasks = _task_plan(config)
    output_root.mkdir(parents=True, exist_ok=True)
    config_sha256 = _sha256(study_path)
    results = _completed_results(output_root)
    _update_ledger(output_root=output_root, tasks=tasks, results=results)
    stage_order = (
        "tuning",
        "mfe_core",
        "mfe_gated",
        "risk_state_core",
        "risk_state_gated",
    )
    completed_this_run = 0
    for stage in stage_order:
        for task in [item for item in tasks if item["stage"] == stage]:
            if max_tasks is not None and completed_this_run >= int(max_tasks):
                break
            result = _run_training_task(
                task=task,
                inputs=inputs,
                config=config,
                output_root=output_root,
                config_sha256=config_sha256,
            )
            results[str(task["task_id"])] = result
            completed_this_run += 1
            _update_ledger(output_root=output_root, tasks=tasks, results=results)
        stage_results = [item for item in tasks if item["stage"] == stage]
        if not all(str(item["task_id"]) in results for item in stage_results):
            break
        if stage == "mfe_core":
            _select_mfe_core(
                results=results, config=config, output_root=output_root, inputs=inputs
            )
        elif stage == "mfe_gated":
            _select_mfe_gated(
                results=results, config=config, output_root=output_root, inputs=inputs
            )
        elif stage == "risk_state_core":
            _select_risk_state_core(
                results=results, config=config, output_root=output_root
            )
        elif stage == "risk_state_gated":
            _select_risk_state_gated(
                results=results, config=config, output_root=output_root
            )
    all_complete = len(results) >= len(tasks) and all(
        str(task["task_id"]) in results for task in tasks
    )
    status = "training_completed" if all_complete else "training_in_progress"
    manifest = _root_manifest(
        output_root=output_root,
        config_path=study_path,
        input_manifest=input_manifest,
        input_manifest_path=ready_root / MODEL_INPUT_DIR_NAME / "manifest.json",
        tasks=tasks,
        status=status,
    )
    _write_json(output_root / "manifest.json", manifest)
    if all_complete:
        contract = _selected_head_contract(output_root, inputs)
        _write_json(output_root / "selected_head_contract.json", contract)
    return {
        "status": status,
        "study_id": STUDY_ID,
        "task_count": len(tasks),
        "completed_count": len(results),
        "completed_this_run": completed_this_run,
        "training_performed": all_complete,
    }


def _annual_metrics(results: Mapping[str, Mapping[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for result in results.values():
        row = {
            "task_id": result["task_id"],
            "stage": result["stage"],
            "target": result["target"],
            "kind": result["kind"],
            "model_year": int(result["model_year"]),
            "evaluation_year": int(result["evaluation_year"]),
            "variant": result["variant"],
            "base_variant": result.get("base_variant"),
            "gated_family": result.get("gated_family"),
            "feature_count": int(result["feature_count"]),
            "best_iteration": int(result["best_iteration"]),
            "train_row_count": int(result["train_row_count"]),
            "evaluation_row_count": int(result["evaluation_row_count"]),
        }
        row.update({str(key): value for key, value in dict(result["metrics"]).items()})
        rows.append(row)
    return pd.DataFrame(rows).sort_values(
        ["target", "evaluation_year", "variant", "stage"]
    )


def _paired_variant_deltas(
    *, results: Mapping[str, Mapping[str, Any]], output_root: Path
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    mfe_core = _load_selection(output_root / "selection" / "mfe_core_selection.json")
    risk_core = _load_selection(
        output_root / "selection" / "risk_state_core_selection.json"
    )
    comparisons: list[tuple[str, str, str, str]] = []
    for target in ("mfe_10", "mfe_20"):
        for variant in CORE_VARIANTS[1:]:
            comparisons.append((target, variant, "legacy_core", "core"))
        reference = str(mfe_core["targets"][target]["selected_core"])
        for variant in [f"selected_core_plus_{family}" for family in GATED_FAMILIES]:
            comparisons.append((target, variant, reference, "gated"))
    for target in ("risk_10", "risk_20", "state_10"):
        comparisons.append((target, "full_core", "legacy_core", "core"))
        reference = str(risk_core["targets"][target]["selected_core"])
        comparisons.append((target, ALL_GATED_VARIANT, reference, "gated"))
    for target, candidate_variant, reference_variant, comparison_kind in comparisons:
        candidate = _results_for(results, target=target, variant=candidate_variant)
        reference = _results_for(results, target=target, variant=reference_variant)
        for year in ROLLING_YEARS:
            if year not in candidate or year not in reference:
                continue
            row = {
                "target": target,
                "year": year,
                "candidate_variant": candidate_variant,
                "reference_variant": reference_variant,
                "comparison_kind": comparison_kind,
            }
            keys = set(candidate[year]["metrics"]) | set(reference[year]["metrics"])
            for key in sorted(keys):
                if (
                    key in candidate[year]["metrics"]
                    and key in reference[year]["metrics"]
                ):
                    row[f"candidate_{key}"] = candidate[year]["metrics"][key]
                    row[f"reference_{key}"] = reference[year]["metrics"][key]
                    row[f"delta_{key}"] = float(
                        candidate[year]["metrics"][key]
                    ) - float(reference[year]["metrics"][key])
            rows.append(row)
    return pd.DataFrame(rows)


def evaluate(
    *, output_root: Path = DEFAULT_OUTPUT_ROOT, study_path: Path = DEFAULT_STUDY_PATH
) -> dict[str, Any]:
    config = _load_config(study_path)
    manifest = _read_json(output_root / "manifest.json")
    if manifest.get("status") not in {"training_completed", "evaluated", "audited"}:
        raise ModelError("training_not_completed")
    results = _completed_results(output_root)
    tasks = _task_plan(config)
    if len(results) != len(tasks):
        raise ModelError(f"evaluation_task_count_mismatch:{len(results)}:{len(tasks)}")
    annual = _annual_metrics(results)
    paired = _paired_variant_deltas(results=results, output_root=output_root)
    importance_parts = []
    for result in results.values():
        importance_parts.append(
            pd.read_parquet(Path(result["files"]["family_importance"]["path"]))
        )
    importance = pd.concat(importance_parts, ignore_index=True)
    annual_path = output_root / "annual_metrics.parquet"
    paired_path = output_root / "paired_variant_deltas.parquet"
    importance_path = output_root / "family_importance.parquet"
    _write_parquet(annual, annual_path)
    _write_parquet(paired, paired_path)
    _write_parquet(importance, importance_path)
    inputs = ModelInputs(_read_json(Path(manifest["model_inputs"]["path"])))
    contract = _selected_head_contract(output_root, inputs)
    contract_path = output_root / "selected_head_contract.json"
    _write_json(contract_path, contract)
    files = {
        "annual_metrics": _file_record(annual_path, row_count=len(annual)),
        "paired_variant_deltas": _file_record(paired_path, row_count=len(paired)),
        "family_importance": _file_record(importance_path, row_count=len(importance)),
        "selected_head_contract": _file_record(contract_path),
    }
    manifest["status"] = "evaluated"
    manifest["outputs"] = files
    manifest["training_performed"] = True
    manifest["feature_set_selected"] = True
    _write_json(output_root / "manifest.json", manifest)
    return {
        "status": "evaluated",
        "task_count": len(results),
        "annual_metric_rows": len(annual),
        "paired_delta_rows": len(paired),
        "family_importance_rows": len(importance),
        "selected_head_contract": str(contract_path.resolve()),
    }


def audit(
    *, output_root: Path = DEFAULT_OUTPUT_ROOT, study_path: Path = DEFAULT_STUDY_PATH
) -> dict[str, Any]:
    config = _load_config(study_path)
    manifest = _read_json(output_root / "manifest.json")
    input_manifest = _read_json(Path(manifest["model_inputs"]["path"]))
    input_checks = _verify_model_input_files(input_manifest, full_hash=True)
    del input_checks
    inputs = ModelInputs(input_manifest)
    tasks = _task_plan(config)
    results = _completed_results(output_root)
    task_ids = {str(task["task_id"]) for task in tasks}
    checks: dict[str, Any] = {
        "manifest_status": manifest.get("status")
        in {"training_completed", "evaluated", "audited"},
        "task_count_exact": set(results) == task_ids,
        "input_fingerprint_present": bool(input_manifest.get("input_fingerprint")),
        "row_count_exact": inputs.row_count == int(input_manifest["row_count"]),
        "forbidden_2026_rows": not bool(
            np.char.startswith(inputs.trade_date, "2026-").any()
        ),
        "feature_contract_counts": input_manifest.get("feature_groups", {}).get(
            "full_core"
        )
        is not None
        and len(input_manifest.get("feature_groups", {}).get("full_core", [])) == 587,
        "forbidden_listing_age": not any(
            str(item.get("feature_name")) == "listing_age_days"
            for item in input_manifest.get("features", [])
        ),
        "outputs_hashed": True,
        "task_files_valid": True,
        "purge_valid": True,
    }
    for task in tasks:
        result = results.get(str(task["task_id"]))
        if result is None:
            checks["task_files_valid"] = False
            continue
        try:
            feature_names, _ = _effective_features(
                inputs=inputs, task=task, output_root=output_root
            )
            fingerprint = _task_fingerprint(
                task=task,
                feature_names=feature_names,
                model_input_fingerprint=str(input_manifest["input_fingerprint"]),
                config_sha256=_sha256(study_path),
            )
            checks["task_files_valid"] &= _task_complete(
                _task_result_path(output_root, str(task["task_id"])),
                fingerprint=fingerprint,
            )
            prediction_path = Path(result["files"]["prediction"]["path"])
            candidate_path = Path(result["files"]["candidate_id"]["path"])
            prediction = np.load(prediction_path, mmap_mode="r", allow_pickle=False)
            candidate_ids = np.load(candidate_path, mmap_mode="r", allow_pickle=False)
            expected_rows = inputs.rows_for_year(int(result["evaluation_year"]))
            expected_ids = inputs.candidate_ids[expected_rows]
            checks["task_files_valid"] &= np.array_equal(candidate_ids, expected_ids)
            checks["task_files_valid"] &= prediction.shape[0] == len(expected_ids)
            fold = inputs.fold(
                year=int(result["evaluation_year"]),
                horizon=int(result["horizon"]),
                target=str(result["target"]),
            )
            checks["purge_valid"] &= int(
                result["maximum_train_signal_date_idx"]
            ) == int(fold["maximum_train_signal_date_idx"])
        except (ModelError, OSError, KeyError, ValueError, TypeError, IndexError):
            checks["task_files_valid"] = False
    for record in dict(manifest.get("outputs", {}) or {}).values():
        path = Path(record["path"])
        checks["outputs_hashed"] &= path.is_file() and _sha256(path) == record["sha256"]
    audit_payload = {
        "schema": "seq100_quality_liquidity_model_audit/1",
        "study_id": STUDY_ID,
        "status": "ok" if all(checks.values()) else "failed",
        "created_at": _now(),
        "checks": checks,
        "task_count": len(results),
        "training_performed": True,
        "evaluation_semantics": "retrospective_rolling_oos",
    }
    _write_json(output_root / "audit.json", audit_payload)
    if audit_payload["status"] != "ok":
        raise ModelError("model_audit_failed")
    manifest["status"] = "audited"
    manifest["audit"] = _file_record(output_root / "audit.json")
    _write_json(output_root / "manifest.json", manifest)
    return audit_payload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-path", type=Path, default=DEFAULT_STUDY_PATH)
    parser.add_argument("--ready-root", type=Path, default=DEFAULT_READY_ROOT)
    parser.add_argument("--audit-root", type=Path, default=DEFAULT_AUDIT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--prepare", action="store_true")
    modes.add_argument("--run", action="store_true")
    modes.add_argument("--evaluate", action="store_true")
    modes.add_argument("--audit", action="store_true")
    parser.add_argument("--max-tasks", type=int, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.prepare:
        result = prepare(
            study_path=args.study_path,
            ready_root=args.ready_root,
            audit_root=args.audit_root,
            output_root=args.output_root,
        )
    elif args.run:
        result = run(
            study_path=args.study_path,
            ready_root=args.ready_root,
            audit_root=args.audit_root,
            output_root=args.output_root,
            max_tasks=args.max_tasks,
        )
    elif args.evaluate:
        result = evaluate(output_root=args.output_root, study_path=args.study_path)
    else:
        result = audit(output_root=args.output_root, study_path=args.study_path)
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    return (
        0
        if result.get("status")
        in {
            "completed",
            "training_completed",
            "training_in_progress",
            "evaluated",
            "ok",
            "audited",
        }
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())
