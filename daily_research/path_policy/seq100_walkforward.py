from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
from datetime import datetime
from pathlib import Path
from statistics import NormalDist
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from daily_research.path_policy.qdp_v2_sequence_path_pack import validate_sequence_pack


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PYTHON = Path("C:/Users/ASUS/miniconda3/envs/yolos/python.exe")
DEFAULT_STORE_ROOT = Path("daily_research/data/research_store")
DEFAULT_SOURCE_VIEW = DEFAULT_STORE_ROOT / "views/seq100_path60_todayclose_ohlcva.json"
DEFAULT_OOS_YEARS = (2022, 2023, 2024, 2025)
DEFAULT_DEVELOPMENT_YEARS = (2022, 2023, 2024, 2025)
DEFAULT_STUDY_ROOT = Path("daily_research/output/path_policy/studies/seq100_purged_walkforward_2022_2025")
DEFAULT_BOOTSTRAP_REPLICATIONS = 10_000
DEFAULT_BLOCK_LENGTH = 60
DEFAULT_SEED = 7
APPROVED_DEVELOPMENT_CONTRACT_PATH = Path(
    "daily_research/brain/references/seq100_candidate_complete_development_walkforward_contract_20260711.json"
)
APPROVED_DEVELOPMENT_CONTRACT_ID = "seq100_candidate_complete_development_walkforward_contract_20260711_v1"
APPROVED_DEVELOPMENT_CONTRACT_SHA256 = "e461c3e4654b0e97b31508874f2eacf89258d5219bf0fb452d8f1690ee193f11"

PROFILE_COMMANDS = {
    "summary_v2_all_channels": "train-summary-v2",
    "daily_only_summary_v2_ohlcva_aux_low": "train-daily-only-summary-v2-ohlcva-aux-low",
    "daily_only_summary_v2_ohlcva_aux_low_hard_st": "train-daily-only-summary-v2-ohlcva-aux-low-hard-st",
    "daily_only_summary_v2_ohlcva_aux_low_hard_st_global_tail": "train-daily-only-summary-v2-ohlcva-aux-low-hard-st-global-tail",
}
LEGACY_REQUIRED_PROFILES = (
    "summary_v2_all_channels",
    "daily_only_summary_v2_ohlcva_aux_low",
)
INNER_SCREEN_PROFILES = (
    "daily_only_summary_v2_ohlcva_aux_low",
    "daily_only_summary_v2_ohlcva_aux_low_hard_st",
    "daily_only_summary_v2_ohlcva_aux_low_hard_st_global_tail",
)
STUDY_RUN_TAG = "seq100_purged_walkforward_2022_2025"
FOLD_TRAINING_CONTRACT_BINDING_METHOD = "post_run_reconstruction_v1"
FOLD_METADATA_RECONCILIATION_METHOD = "staged_metadata_only_v1"


def _workspace_path(path: str | Path) -> Path:
    raw = Path(path)
    return raw if raw.is_absolute() else WORKSPACE_ROOT / raw


def _relative(path: str | Path) -> str:
    resolved = _workspace_path(path).resolve()
    try:
        return resolved.relative_to(WORKSPACE_ROOT.resolve()).as_posix()
    except ValueError:
        return str(resolved)


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _read_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(_workspace_path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _write_json(path: str | Path, payload: Mapping[str, Any]) -> Path:
    target = _workspace_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(
        json.dumps(dict(payload), ensure_ascii=False, indent=2, default=_json_default) + "\n",
        encoding="utf-8",
    )
    temporary.replace(target)
    return target


def _write_parquet(path: str | Path, frame: pd.DataFrame) -> Path:
    target = _workspace_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    frame.to_parquet(temporary, index=False)
    temporary.replace(target)
    return target


def _sha256_file(path: str | Path) -> str:
    target = _workspace_path(path)
    digest = hashlib.sha256()
    with target.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=_json_default,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _serialized_json_sha256(payload: Mapping[str, Any]) -> str:
    serialized = json.dumps(dict(payload), ensure_ascii=False, indent=2, default=_json_default) + "\n"
    encoded = serialized.replace("\n", os.linesep).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _canonicalize_json(payload: Any) -> Any:
    return json.loads(json.dumps(payload, default=_json_default, allow_nan=False))


def approved_development_contract_binding() -> dict[str, Any]:
    path = _workspace_path(APPROVED_DEVELOPMENT_CONTRACT_PATH).resolve()
    payload = _read_json(path)
    declared = str(payload.get("contract_sha256", "") or "")
    semantic_payload = {key: value for key, value in payload.items() if key != "contract_sha256"}
    computed = _canonical_json_sha256(semantic_payload)
    if str(payload.get("contract_id", "")) != APPROVED_DEVELOPMENT_CONTRACT_ID:
        raise ValueError("approved development contract id changed")
    if declared != APPROVED_DEVELOPMENT_CONTRACT_SHA256 or computed != declared:
        raise ValueError("approved development contract semantic digest changed")
    return {
        "path": str(path),
        "contract_id": APPROVED_DEVELOPMENT_CONTRACT_ID,
        "contract_sha256": declared,
        "contract_file_sha256": _sha256_file(path),
    }


def _source_backing_file_inventory(source_manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
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
        collect(dict(source_manifest.get(section, {}) or {}))
    inventory: list[dict[str, Any]] = []
    for path in sorted(paths, key=lambda item: str(item).lower()):
        if not path.is_file():
            raise FileNotFoundError(path)
        stat = path.stat()
        inventory.append(
            {
                "path": str(path),
                "size": int(stat.st_size),
                "mtime_ns": int(stat.st_mtime_ns),
            }
        )
    return inventory


def _source_view_provenance(
    source_path: str | Path,
    source_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    manifest_path = _workspace_path(source_path).resolve()
    sample_path = _workspace_path(str(source_manifest.get("sample_index_path", "") or "")).resolve()
    candidate_value = str(source_manifest.get("candidate_index_path", "") or "")
    candidate_path = _workspace_path(candidate_value).resolve() if candidate_value else None
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    if not sample_path.is_file():
        raise FileNotFoundError(sample_path)
    if candidate_path is not None and not candidate_path.is_file():
        raise FileNotFoundError(candidate_path)
    artifact_view = dict(source_manifest.get("artifact_view", {}) or {})
    backing_files = _source_backing_file_inventory(source_manifest)
    provenance = {
        "schema_version": 1,
        "manifest_path": str(manifest_path),
        "manifest_sha256": _sha256_file(manifest_path),
        "sample_index_path": str(sample_path),
        "sample_index_sha256": _sha256_file(sample_path),
        "artifact_type": str(source_manifest.get("artifact_type", "")),
        "artifact_view_id": str(artifact_view.get("view_id", "")),
        "created_at": str(source_manifest.get("created_at", "")),
        "backing_files": backing_files,
        "backing_files_sha256": _canonical_json_sha256(backing_files),
    }
    if candidate_path is not None:
        provenance.update(
            {
                "candidate_index_path": str(candidate_path),
                "candidate_index_sha256": _sha256_file(candidate_path),
            }
        )
    return provenance


def _validated_source_view_provenance(manifest: Mapping[str, Any]) -> dict[str, Any]:
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
    if candidate_value:
        candidate_path = _workspace_path(candidate_value).resolve()
        if not candidate_path.is_file() or _sha256_file(candidate_path) != str(
            stored.get("candidate_index_sha256", "")
        ):
            raise ValueError("fold source candidate index is missing or its SHA-256 changed")
    current = _read_json(source_path)
    if str(current.get("artifact_type", "")) != str(stored["artifact_type"]):
        raise ValueError("fold source artifact type changed")
    if _workspace_path(str(current.get("sample_index_path", "") or "")).resolve() != sample_path:
        raise ValueError("fold source sample-index path changed")
    current_candidate_value = str(current.get("candidate_index_path", "") or "")
    if bool(candidate_value) != bool(current_candidate_value):
        raise ValueError("fold source candidate-index declaration changed")
    if candidate_value and _workspace_path(current_candidate_value).resolve() != candidate_path:
        raise ValueError("fold source candidate-index path changed")
    current_backing_files = _source_backing_file_inventory(current)
    if current_backing_files != list(stored.get("backing_files", []) or []):
        raise ValueError("fold source backing-file identity changed")
    if _canonical_json_sha256(current_backing_files) != str(stored.get("backing_files_sha256", "")):
        raise ValueError("fold source backing-file inventory digest changed")
    return stored


def _normalize_profiles(profiles: Sequence[str]) -> tuple[str, ...]:
    values = tuple(str(item).strip() for item in profiles if str(item).strip())
    if not values:
        raise ValueError("at least one profile is required")
    unknown = sorted(set(values).difference(PROFILE_COMMANDS))
    if unknown:
        raise ValueError(f"unsupported profiles: {unknown}")
    if len(values) != len(set(values)):
        raise ValueError("profiles must be unique")
    return values


def _parse_years(raw: str | Iterable[int]) -> tuple[int, ...]:
    if not isinstance(raw, str):
        return tuple(sorted({int(item) for item in raw}))
    years: set[int] = set()
    for token in [item.strip() for item in raw.split(",") if item.strip()]:
        if "-" in token:
            start, end = token.split("-", 1)
            years.update(range(int(start), int(end) + 1))
        else:
            years.add(int(token))
    if not years:
        raise ValueError("at least one OOS year is required")
    return tuple(sorted(years))


def _open_memmap(meta: Mapping[str, Any], *, dtype: str) -> np.memmap:
    if list(meta.get("shards", []) or []):
        raise ValueError("walk-forward normalization expects non-sharded feature and mask arrays")
    path = _workspace_path(str(meta.get("path", "") or ""))
    shape = tuple(int(item) for item in list(meta.get("shape", []) or []))
    if not path.exists() or not shape:
        raise FileNotFoundError(path)
    return np.memmap(path, dtype=dtype, mode="r", shape=shape)


def _fit_channel_normalization(
    meta: Mapping[str, Any],
    *,
    start_idx: int,
    end_idx_exclusive: int,
    chunk_dates: int = 32,
) -> dict[str, list[float]]:
    panel = _open_memmap(meta, dtype="float32")
    if panel.ndim != 3:
        raise ValueError(f"feature panel must be 3D, got {panel.shape}")
    if not 0 <= int(start_idx) < int(end_idx_exclusive) <= int(panel.shape[0]):
        raise ValueError("invalid normalization date range")
    feature_count = int(panel.shape[2])
    counts = np.zeros(feature_count, dtype=np.int64)
    sums = np.zeros(feature_count, dtype=np.float64)
    square_sums = np.zeros(feature_count, dtype=np.float64)
    for chunk_start in range(int(start_idx), int(end_idx_exclusive), max(int(chunk_dates), 1)):
        chunk_end = min(chunk_start + max(int(chunk_dates), 1), int(end_idx_exclusive))
        values = np.asarray(panel[chunk_start:chunk_end], dtype=np.float32)
        finite = np.isfinite(values)
        counts += finite.sum(axis=(0, 1), dtype=np.int64)
        values64 = values.astype(np.float64, copy=False)
        sums += np.where(finite, values64, 0.0).sum(axis=(0, 1), dtype=np.float64)
        square_sums += np.where(finite, values64 * values64, 0.0).sum(axis=(0, 1), dtype=np.float64)
    safe_counts = np.maximum(counts, 1)
    mean = sums / safe_counts
    variance = square_sums / safe_counts - mean * mean
    variance = np.maximum(variance, 0.0)
    std = np.sqrt(variance)
    mean = np.where((counts > 0) & np.isfinite(mean), mean, 0.0).astype(np.float32)
    std = np.where((counts > 0) & np.isfinite(std) & (std > 1.0e-6), std, 1.0).astype(np.float32)
    return {"mean": mean.tolist(), "std": std.tolist()}


def _fit_fold_normalization(
    source: Mapping[str, Any],
    *,
    start_idx: int,
    end_idx_exclusive: int,
    date_values: Sequence[str],
) -> dict[str, Any]:
    normalization: dict[str, Any] = {
        "fit_scope": "feature_dates_before_oos_start",
        "fit_date_start": str(date_values[int(start_idx)]),
        "fit_date_end": str(date_values[int(end_idx_exclusive) - 1]),
        "fit_date_count": int(end_idx_exclusive - start_idx),
        "fit_date_end_exclusive": str(date_values[int(end_idx_exclusive)]),
        "oos_feature_date_count": 0,
    }
    for name, meta in dict(source.get("feature_channels", {}) or {}).items():
        stats = _fit_channel_normalization(
            dict(meta),
            start_idx=int(start_idx),
            end_idx_exclusive=int(end_idx_exclusive),
        )
        expected = len(list(dict(meta).get("columns", []) or []))
        if expected and (len(stats["mean"]) != expected or len(stats["std"]) != expected):
            raise ValueError(f"normalization width mismatch for {name}")
        normalization[str(name)] = stats
    return normalization


def _complete_case_audit(
    source: Mapping[str, Any],
    *,
    oos_date_indices: Sequence[int],
) -> dict[str, Any]:
    masks = dict(source.get("masks", {}) or {})
    required = ("input_valid", "entry_buyable", "label_valid")
    if any(name not in masks for name in required):
        return {
            "available": False,
            "candidate_row_count_before_label_valid": 0,
            "complete_case_row_count": 0,
            "complete_case_excluded_row_count": 0,
            "complete_case_excluded_rate": 0.0,
        }
    input_valid = _open_memmap(dict(masks["input_valid"]), dtype="bool")
    entry_buyable = _open_memmap(dict(masks["entry_buyable"]), dtype="bool")
    label_valid = _open_memmap(dict(masks["label_valid"]), dtype="bool")
    indices = np.asarray(list(oos_date_indices), dtype=np.int64)
    candidates = np.asarray(input_valid[indices], dtype=bool) & np.asarray(entry_buyable[indices], dtype=bool)
    complete = candidates & np.asarray(label_valid[indices], dtype=bool)
    candidate_count = int(candidates.sum())
    complete_count = int(complete.sum())
    excluded = int(candidate_count - complete_count)
    return {
        "available": True,
        "candidate_row_count_before_label_valid": candidate_count,
        "complete_case_row_count": complete_count,
        "complete_case_excluded_row_count": excluded,
        "complete_case_excluded_rate": float(excluded / max(candidate_count, 1)),
        "interpretation": "OOS metrics use complete future-60d labels and may be optimistic versus the signal-day tradable universe.",
    }


def purged_view_id(oos_year: int) -> str:
    return f"seq100_path60_todayclose_ohlcva_purged_oos{int(oos_year)}"


def purged_view_path(
    oos_year: int,
    *,
    store_root: str | Path = DEFAULT_STORE_ROOT,
) -> Path:
    return _workspace_path(store_root) / "views" / f"{purged_view_id(oos_year)}.json"


def _compute_fold_training_contract(
    manifest: Mapping[str, Any],
    *,
    sample_frame: pd.DataFrame | None = None,
) -> dict[str, Any]:
    sample_path = _workspace_path(str(manifest.get("sample_index_path", "") or ""))
    if not sample_path.exists():
        raise FileNotFoundError(f"missing fold sample index: {sample_path}")
    frame = pd.read_parquet(sample_path) if sample_frame is None else sample_frame.copy()
    required_columns = {"split", "trade_date", "date_idx", "symbol_idx"}
    missing = sorted(required_columns - set(frame.columns))
    if missing:
        raise ValueError(f"fold sample index missing training-contract columns: {missing}")
    if frame.empty or bool(frame.duplicated(["date_idx", "symbol_idx"]).any()):
        raise ValueError("fold sample index must be non-empty with unique date/symbol rows")
    frame["split"] = frame["split"].astype(str)
    split_values = set(frame["split"].unique().tolist())
    if split_values != {"train", "oos"}:
        raise ValueError(f"fold training contract requires train/oos splits, got {sorted(split_values)}")
    train = frame[frame["split"].eq("train")]
    oos = frame[frame["split"].eq("oos")]
    if train.empty or oos.empty:
        raise ValueError("fold training contract requires non-empty train and oos splits")

    lookback_days = int(manifest.get("lookback_days", 0) or 0)
    forward_days = int(manifest.get("forward_days", 0) or 0)
    if lookback_days <= 0 or forward_days <= 0:
        raise ValueError("fold training contract requires positive lookback_days and forward_days")
    oos_start_idx = int(oos["date_idx"].astype(int).min())
    overlap_count = int(((train["date_idx"].astype(int) + forward_days) >= oos_start_idx).sum())
    if overlap_count:
        raise ValueError(f"fold training labels overlap OOS: {overlap_count}")
    oos_years = sorted({int(str(value)[:4]) for value in oos["trade_date"].astype(str)})
    if len(oos_years) != 1:
        raise ValueError(f"fold OOS must contain exactly one year, got {oos_years}")

    purge = dict(manifest.get("purged_walkforward", {}) or {})
    if not purge:
        raise ValueError("fold is missing purged_walkforward metadata")
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

    sample_sha256 = _sha256_file(sample_path)
    normalization_sha256 = _canonical_json_sha256(normalization)
    split_payload = {
        "roles": {"fit": "train", "evaluation": "oos"},
        "counts": {str(key): int(value) for key, value in frame["split"].value_counts().sort_index().items()},
        "train_date_idx_min": int(train["date_idx"].astype(int).min()),
        "train_date_idx_max": int(train["date_idx"].astype(int).max()),
        "oos_date_idx_min": oos_start_idx,
        "oos_date_idx_max": int(oos["date_idx"].astype(int).max()),
        "train_years": sorted({int(str(value)[:4]) for value in train["trade_date"].astype(str)}),
        "oos_years": oos_years,
    }
    purge_payload = {
        key: purge.get(key)
        for key in (
            "schema_version",
            "method",
            "split_roles",
            "train_start_year",
            "oos_year",
            "oos_start",
            "oos_start_trade_date",
            "oos_end",
            "oos_start_date_idx",
            "safe_train_signal_end",
            "safe_train_signal_end_trade_date",
            "max_train_label_end",
            "max_train_label_end_trade_date",
            "purge_rule",
            "purged_row_count",
            "purged_signal_date_count",
            "purged_signal_start",
            "purged_signal_end",
            "label_overlap_count",
            "normalization_cutoff_exclusive",
            "normalization_last_feature_date",
            "normalization_includes_purge_period_features",
        )
    }
    payload = {
        "schema_version": 1,
        "artifact_type": "seq100_purged_fold_training_contract",
        "lookback_days": lookback_days,
        "forward_days": forward_days,
        "sample_index": {
            "sha256": sample_sha256,
            "row_count": int(len(frame)),
            "columns": [str(column) for column in frame.columns],
            "dtypes": {str(column): str(dtype) for column, dtype in frame.dtypes.items()},
        },
        "split": split_payload,
        "purge": purge_payload,
        "normalization": normalization,
        "feature_channels": dict(manifest.get("feature_channels", {}) or {}),
        "label_arrays": dict(manifest.get("label_arrays", {}) or {}),
        "label_semantics": dict(manifest.get("label_semantics", {}) or {}),
        "masks": dict(manifest.get("masks", {}) or {}),
        "scope": dict(manifest.get("scope", {}) or {}),
        "source_view_provenance": dict(manifest.get("source_view_provenance", {}) or {}),
        "date_values_sha256": _canonical_json_sha256(list(manifest.get("date_values", []) or [])),
        "symbol_values_sha256": _canonical_json_sha256(list(manifest.get("symbol_values", []) or [])),
    }
    return {
        "schema_version": 1,
        "algorithm": "sha256",
        "sha256": _canonical_json_sha256(payload),
        "sample_index_sha256": sample_sha256,
        "normalization_sha256": normalization_sha256,
        "payload": payload,
    }


def _validated_fold_training_contract(
    manifest: Mapping[str, Any],
    *,
    sample_frame: pd.DataFrame | None = None,
) -> dict[str, Any]:
    stored = dict(manifest.get("fold_training_contract", {}) or {})
    if not stored:
        raise ValueError("fold manifest is missing fold_training_contract")
    computed = _compute_fold_training_contract(manifest, sample_frame=sample_frame)
    if stored != computed:
        raise ValueError("fold_training_contract does not match current training material")
    return computed


def build_purged_walkforward_fold(
    source_view: str | Path = DEFAULT_SOURCE_VIEW,
    oos_year: int = 2025,
    train_start_year: int = 2012,
    store_root: str | Path = DEFAULT_STORE_ROOT,
    overwrite: bool = False,
) -> dict[str, Any]:
    source_path = _workspace_path(source_view).resolve()
    source = _read_json(source_path)
    if source.get("artifact_type") != "qdp_v2_sequence_path_pack":
        raise ValueError(f"not a sequence path pack view: {source_path}")
    forward_days = int(source.get("forward_days", 0) or 0)
    if forward_days <= 0:
        raise ValueError("source view forward_days must be positive")
    date_values = [str(item) for item in list(source.get("date_values", []) or [])]
    if not date_values or len(date_values) != len(set(date_values)):
        raise ValueError("source date_values must be non-empty and unique")
    sample_index_path = _workspace_path(str(source.get("sample_index_path", "") or ""))
    source_index = pd.read_parquet(sample_index_path)
    source_provenance = _source_view_provenance(source_path, source)
    required_columns = {"split", "trade_date", "date_idx", "symbol_idx", "symbol"}
    missing = sorted(required_columns - set(source_index.columns))
    if missing:
        raise ValueError(f"source sample index missing columns: {missing}")
    duplicate_mask = source_index.duplicated(["date_idx", "symbol_idx"], keep=False)
    if bool(duplicate_mask.any()):
        raise ValueError("source sample index contains duplicate (date_idx, symbol_idx) rows")

    index = source_index.copy()
    index["trade_date"] = index["trade_date"].astype(str)
    index["date_idx"] = index["date_idx"].astype(np.int64)
    if bool(((index["date_idx"] < 0) | (index["date_idx"] >= len(date_values))).any()):
        raise ValueError("source sample index date_idx is outside date_values")
    expected_trade_dates = np.asarray(date_values, dtype=object)[index["date_idx"].to_numpy(dtype=np.int64)]
    if not bool(np.equal(index["trade_date"].to_numpy(dtype=object), expected_trade_dates).all()):
        raise ValueError("source sample index trade_date does not match date_values[date_idx]")
    index_year = index["trade_date"].str.slice(0, 4).astype(int)
    oos_mask = index_year.eq(int(oos_year))
    if not bool(oos_mask.any()):
        raise ValueError(f"source sample index has no rows for OOS year {oos_year}")
    oos_dates = sorted(index.loc[oos_mask, "trade_date"].unique().tolist())
    oos_start = str(oos_dates[0])
    oos_end = str(oos_dates[-1])
    date_to_idx = {date: idx for idx, date in enumerate(date_values)}
    if oos_start not in date_to_idx:
        raise ValueError(f"OOS start {oos_start} missing from date_values")
    oos_start_idx = int(date_to_idx[oos_start])
    label_end_idx = index["date_idx"].to_numpy(dtype=np.int64, copy=True) + int(forward_days)
    if int(label_end_idx.max(initial=-1)) >= len(date_values):
        raise ValueError("source sample index contains labels beyond date_values")
    pre_oos = index["date_idx"].to_numpy(dtype=np.int64, copy=False) < oos_start_idx
    after_train_start = index_year.to_numpy(dtype=np.int64, copy=False) >= int(train_start_year)
    train_mask = pre_oos & after_train_start & (label_end_idx < oos_start_idx)
    purge_mask = pre_oos & after_train_start & (label_end_idx >= oos_start_idx)

    selected_mask = train_mask | oos_mask.to_numpy(dtype=bool, copy=False)
    fold_index = index.loc[selected_mask].copy()
    fold_label_end = label_end_idx[selected_mask]
    fold_index.insert(1, "source_split", fold_index["split"].astype(str).to_numpy(copy=True))
    fold_index["split"] = np.where(train_mask[selected_mask], "train", "oos")
    fold_index["year"] = fold_index["trade_date"].str.slice(0, 4).astype(np.int16)
    fold_index["label_end_date_idx"] = fold_label_end.astype(np.int32)
    fold_index["label_end_trade_date"] = np.asarray(date_values, dtype=object)[fold_label_end]
    fold_index = fold_index.sort_values(["date_idx", "symbol_idx"], kind="mergesort").reset_index(drop=True)
    fold_index["sample_id"] = np.arange(len(fold_index), dtype=np.int64)

    train_rows = fold_index[fold_index["split"].eq("train")]
    oos_rows = fold_index[fold_index["split"].eq("oos")]
    if train_rows.empty or oos_rows.empty:
        raise ValueError("purged fold requires non-empty train and oos splits")
    label_overlap_count = int((train_rows["label_end_date_idx"].astype(int) >= oos_start_idx).sum())
    if label_overlap_count:
        raise AssertionError("purge invariant failed: training labels overlap OOS")

    normalization_start_idx = next(
        (idx for idx, date in enumerate(date_values) if int(str(date)[:4]) >= int(train_start_year)),
        -1,
    )
    if normalization_start_idx < 0 or normalization_start_idx >= oos_start_idx:
        raise ValueError("normalization range is empty")
    normalization = _fit_fold_normalization(
        source,
        start_idx=normalization_start_idx,
        end_idx_exclusive=oos_start_idx,
        date_values=date_values,
    )

    store_root_path = _workspace_path(store_root)
    view_id = purged_view_id(oos_year)
    sample_target = store_root_path / "sample_index" / f"{view_id}.parquet"
    view_target = store_root_path / "views" / f"{view_id}.json"
    if not overwrite and (sample_target.exists() or view_target.exists()):
        raise FileExistsError(f"walk-forward fold already exists for {oos_year}")
    _write_parquet(sample_target, fold_index)

    oos_date_indices = sorted({int(item) for item in oos_rows["date_idx"].astype(int).tolist()})
    complete_case = _complete_case_audit(source, oos_date_indices=oos_date_indices)
    purge_dates = sorted(index.loc[purge_mask, "trade_date"].astype(str).unique().tolist())
    safe_train_signal_end = str(train_rows["trade_date"].max())
    max_train_label_end = str(train_rows["label_end_trade_date"].max())
    purged_walkforward = {
        "schema_version": 1,
        "method": "expanding_train_fixed_oos",
        "split_roles": {"fit": "train", "evaluation": "oos"},
        "train_start_year": int(train_start_year),
        "oos_year": int(oos_year),
        "oos_start": oos_start,
        "oos_start_trade_date": oos_start,
        "oos_end": oos_end,
        "oos_start_date_idx": int(oos_start_idx),
        "safe_train_signal_end": safe_train_signal_end,
        "safe_train_signal_end_trade_date": safe_train_signal_end,
        "max_train_label_end": max_train_label_end,
        "max_train_label_end_trade_date": max_train_label_end,
        "purge_rule": "date_idx + forward_days < oos_start_date_idx",
        "purged_row_count": int(purge_mask.sum()),
        "purged_signal_date_count": int(len(purge_dates)),
        "purged_signal_start": str(purge_dates[0]) if purge_dates else "",
        "purged_signal_end": str(purge_dates[-1]) if purge_dates else "",
        "label_overlap_count": label_overlap_count,
        "normalization_cutoff_exclusive": oos_start,
        "normalization_last_feature_date": str(date_values[oos_start_idx - 1]),
        "normalization_includes_purge_period_features": True,
        "complete_case_audit": complete_case,
    }
    artifact_view = dict(source.get("artifact_view", {}) or {})
    artifact_view.update(
        {
            "schema_version": 1,
            "view_id": view_id,
            "view_type": "purged_walk_forward_research_store_view",
            "source_view": str(source_path),
            "purge_audit": purged_walkforward,
        }
    )
    manifest = dict(source)
    manifest.update(
        {
            "created_at": _now(),
            "start_date": f"{int(train_start_year)}-01-01",
            "end_date": oos_end,
            "train_years": sorted({int(value) for value in train_rows["year"].astype(int).tolist()}),
            "validation_years": [],
            "test_years": [],
            "oos_years": [int(oos_year)],
            "split_roles": {"fit": "train", "evaluation": "oos"},
            "sample_index_path": str(sample_target.resolve()),
            "sample_count": int(len(fold_index)),
            "sample_count_by_split": {
                "train": int(len(train_rows)),
                "oos": int(len(oos_rows)),
            },
            "normalization": normalization,
            "artifact_view": artifact_view,
            "purged_walkforward": purged_walkforward,
            "source_view_provenance": source_provenance,
        }
    )
    manifest["fold_training_contract"] = _compute_fold_training_contract(
        manifest,
        sample_frame=fold_index,
    )
    _write_json(view_target, manifest)
    verification = verify_purged_walkforward_view(view_target)
    if verification["status"] != "ok":
        raise RuntimeError(f"purged fold verification failed: {verification['blockers']}")
    return {
        "status": "ok",
        "oos_year": int(oos_year),
        "view_id": view_id,
        "view_path": str(view_target.resolve()),
        "sample_index_path": str(sample_target.resolve()),
        "sample_count_by_split": manifest["sample_count_by_split"],
        "fold_training_contract": manifest["fold_training_contract"],
        "purged_walkforward": purged_walkforward,
        "normalization": {
            key: normalization[key]
            for key in (
                "fit_scope",
                "fit_date_start",
                "fit_date_end",
                "fit_date_count",
                "fit_date_end_exclusive",
                "oos_feature_date_count",
            )
        },
        "verification": verification,
    }


def verify_purged_walkforward_view(view_path: str | Path) -> dict[str, Any]:
    path = _workspace_path(view_path)
    manifest = _read_json(path)
    blockers: list[str] = []
    base_validation = validate_sequence_pack(path)
    blockers.extend(str(item) for item in list(base_validation.get("blockers", []) or []))
    contract = dict(manifest.get("purged_walkforward", {}) or {})
    if not contract:
        blockers.append("missing_purged_walkforward_contract")
    if str(contract.get("method", "")) == "expanding_train_fixed_oos":
        try:
            _validated_source_view_provenance(manifest)
        except (FileNotFoundError, ValueError) as exc:
            blockers.append(f"source_view_provenance:{exc}")
    sample_path = _workspace_path(str(manifest.get("sample_index_path", "") or ""))
    frame = pd.read_parquet(sample_path) if sample_path.exists() else pd.DataFrame()
    if frame.empty:
        blockers.append("empty_sample_index")
    else:
        split_values = set(frame["split"].astype(str).unique().tolist())
        if split_values != {"train", "oos"}:
            blockers.append(f"unexpected_split_values:{sorted(split_values)}")
        if bool(frame.duplicated(["date_idx", "symbol_idx"]).any()):
            blockers.append("duplicate_date_symbol_rows")
        train = frame[frame["split"].astype(str).eq("train")]
        oos = frame[frame["split"].astype(str).eq("oos")]
        forward_days = int(manifest.get("forward_days", 0) or 0)
        if train.empty or oos.empty:
            blockers.append("empty_train_or_oos")
        else:
            oos_start_idx = int(oos["date_idx"].astype(int).min())
            recomputed_overlap = int(((train["date_idx"].astype(int) + forward_days) >= oos_start_idx).sum())
            if recomputed_overlap:
                blockers.append(f"label_overlap_count:{recomputed_overlap}")
            oos_year = int(contract.get("oos_year", 0) or 0)
            observed_years = sorted({int(str(value)[:4]) for value in oos["trade_date"].astype(str)})
            if observed_years != [oos_year]:
                blockers.append(f"unexpected_oos_years:{observed_years}")
            observed_train_years = sorted({int(str(value)[:4]) for value in train["trade_date"].astype(str)})
            if list(manifest.get("train_years", []) or []) != observed_train_years:
                blockers.append("train_years_metadata_mismatch")
            declared_train_start = int(contract.get("train_start_year", -1))
            if (
                not observed_train_years
                or observed_train_years[0] < declared_train_start
                or str(manifest.get("start_date", "")) != f"{declared_train_start}-01-01"
            ):
                blockers.append("train_start_year_mismatch")
            if list(manifest.get("oos_years", []) or []) != [oos_year]:
                blockers.append("oos_years_metadata_mismatch")
            if list(manifest.get("validation_years", []) or []) or list(manifest.get("test_years", []) or []):
                blockers.append("legacy_validation_test_years_present")
            if dict(manifest.get("split_roles", {}) or {}) != {"fit": "train", "evaluation": "oos"}:
                blockers.append("split_roles_metadata_mismatch")
            expected_counts = {str(key): int(value) for key, value in frame["split"].value_counts().to_dict().items()}
            if expected_counts != {
                str(key): int(value)
                for key, value in dict(manifest.get("sample_count_by_split", {}) or {}).items()
            }:
                blockers.append("sample_count_by_split_mismatch")
    normalization = dict(manifest.get("normalization", {}) or {})
    if normalization.get("fit_scope") != "feature_dates_before_oos_start":
        blockers.append("normalization_fit_scope_mismatch")
    if str(normalization.get("fit_date_end_exclusive", "")) != str(contract.get("oos_start", "")):
        blockers.append("normalization_cutoff_mismatch")
    if int(normalization.get("oos_feature_date_count", -1) or 0) != 0:
        blockers.append("normalization_uses_oos_dates")
    for name, meta in dict(manifest.get("feature_channels", {}) or {}).items():
        width = len(list(dict(meta).get("columns", []) or []))
        stats = dict(normalization.get(name, {}) or {})
        if width and (len(list(stats.get("mean", []) or [])) != width or len(list(stats.get("std", []) or [])) != width):
            blockers.append(f"normalization_width_mismatch:{name}")
    if frame.empty:
        blockers.append("fold_training_contract_unverifiable")
    else:
        try:
            _validated_fold_training_contract(manifest, sample_frame=frame)
        except (FileNotFoundError, ValueError) as exc:
            blockers.append(f"fold_training_contract_invalid:{exc}")
    return {
        "status": "blocked" if blockers else "ok",
        "blockers": blockers,
        "view_path": str(path.resolve()),
        "view_id": str(dict(manifest.get("artifact_view", {}) or {}).get("view_id", path.stem)),
        "sample_count_by_split": dict(manifest.get("sample_count_by_split", {}) or {}),
        "label_overlap_count": int(contract.get("label_overlap_count", -1) or 0),
        "fold_training_contract_sha256": str(
            dict(manifest.get("fold_training_contract", {}) or {}).get("sha256", "")
        ),
        "purged_walkforward": contract,
    }


def build_purged_walkforward_folds(
    *,
    source_view: str | Path = DEFAULT_SOURCE_VIEW,
    oos_years: Iterable[int] = DEFAULT_OOS_YEARS,
    train_start_year: int = 2012,
    store_root: str | Path = DEFAULT_STORE_ROOT,
    overwrite: bool = False,
) -> dict[str, Any]:
    folds = [
        build_purged_walkforward_fold(
            source_view=source_view,
            oos_year=int(year),
            train_start_year=int(train_start_year),
            store_root=store_root,
            overwrite=bool(overwrite),
        )
        for year in _parse_years(oos_years)
    ]
    return {
        "status": "ok",
        "source_view": str(_workspace_path(source_view).resolve()),
        "oos_years": [int(item["oos_year"]) for item in folds],
        "folds": folds,
    }


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


def development_view_id(development_year: int) -> str:
    return f"seq100_path60_todayclose_ohlcva_development_{int(development_year)}"


def development_view_path(
    development_year: int,
    *,
    store_root: str | Path = DEFAULT_STORE_ROOT,
) -> Path:
    return _workspace_path(store_root) / "views" / f"{development_view_id(development_year)}.json"


def _max_label_dependency_days(source: Mapping[str, Any]) -> int:
    forward_days = int(source.get("forward_days", 0) or 0)
    if forward_days <= 0:
        raise ValueError("source view forward_days must be positive")
    execution = dict(source.get("execution_contract", {}) or {})
    candidates = [
        int(source.get("max_label_dependency_days", 0) or 0),
        int(execution.get("max_label_dependency_days", 0) or 0),
        int(execution.get("label_dependency_days", 0) or 0),
    ]
    tail_days = int(execution.get("execution_tail_days", source.get("execution_tail_days", 0)) or 0)
    if tail_days < 0:
        raise ValueError("execution_tail_days must be non-negative")
    candidates.append(forward_days + tail_days)
    dependency_days = max(candidates)
    if dependency_days < forward_days:
        raise ValueError("max_label_dependency_days cannot be shorter than forward_days")
    return dependency_days


def _fit_development_normalization(
    source: Mapping[str, Any],
    *,
    start_idx: int,
    end_idx_exclusive: int,
    date_values: Sequence[str],
) -> dict[str, Any]:
    normalization = _fit_fold_normalization(
        source,
        start_idx=int(start_idx),
        end_idx_exclusive=int(end_idx_exclusive),
        date_values=date_values,
    )
    normalization["fit_scope"] = "feature_dates_before_development_start"
    normalization["development_feature_date_count"] = int(normalization.pop("oos_feature_date_count", 0))
    return normalization


def _compute_development_fold_training_contract(
    manifest: Mapping[str, Any],
    *,
    sample_frame: pd.DataFrame | None = None,
    candidate_frame: pd.DataFrame | None = None,
) -> dict[str, Any]:
    sample_path = _workspace_path(str(manifest.get("sample_index_path", "") or ""))
    candidate_path = _workspace_path(str(manifest.get("candidate_index_path", "") or ""))
    if not sample_path.is_file():
        raise FileNotFoundError(f"missing development supervised sample index: {sample_path}")
    if not candidate_path.is_file():
        raise FileNotFoundError(f"missing development candidate index: {candidate_path}")
    samples = pd.read_parquet(sample_path) if sample_frame is None else sample_frame.copy()
    candidates = pd.read_parquet(candidate_path) if candidate_frame is None else candidate_frame.copy()
    sample_required = {"split", "trade_date", "date_idx", "symbol_idx", "symbol"}
    sample_missing = sorted(sample_required.difference(samples.columns))
    candidate_missing = sorted(DEVELOPMENT_CANDIDATE_COLUMNS.difference(candidates.columns))
    if sample_missing:
        raise ValueError(f"development supervised index missing columns: {sample_missing}")
    if candidate_missing:
        raise ValueError(f"development candidate index missing columns: {candidate_missing}")
    if samples.empty or bool(samples.duplicated(["date_idx", "symbol_idx"]).any()):
        raise ValueError("development supervised index must be non-empty with unique date/symbol rows")
    if candidates.empty or bool(candidates.duplicated(["date_idx", "symbol_idx"]).any()):
        raise ValueError("development candidate index must be non-empty with unique date/symbol rows")
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
    if train.empty or development.empty:
        raise ValueError("development fold requires non-empty train and supervised development rows")
    contract = dict(manifest.get("development_walkforward", {}) or {})
    if not contract:
        raise ValueError("fold is missing development_walkforward metadata")
    research_contract = dict(manifest.get("research_contract", {}) or {})
    if not research_contract:
        raise ValueError("fold is not bound to a development research contract")
    if dict(manifest.get("development_contract", {}) or {}) != research_contract:
        raise ValueError("fold development contract alias changed")
    for field in ("contract_id", "contract_sha256", "contract_file_sha256"):
        if not str(research_contract.get(field, "") or ""):
            raise ValueError(f"fold research contract is missing {field}")
    development_year = int(contract.get("development_year", 0) or 0)
    observed_sample_years = sorted({int(str(value)[:4]) for value in development["trade_date"].astype(str)})
    observed_candidate_years = sorted({int(value) for value in candidates["year"].astype(int)})
    if observed_sample_years != [development_year] or observed_candidate_years != [development_year]:
        raise ValueError("development year does not match supervised/candidate indexes")
    development_start_idx = int(candidates["date_idx"].astype(int).min())
    dependency_days = int(contract.get("max_label_dependency_days", 0) or 0)
    if dependency_days <= 0:
        raise ValueError("development contract requires positive max_label_dependency_days")
    overlap_count = int(((train["date_idx"].astype(int) + dependency_days) >= development_start_idx).sum())
    if overlap_count:
        raise ValueError(f"training label dependencies overlap development: {overlap_count}")
    if str(contract.get("purge_rule", "")) != (
        "max_label_dependency_date_idx < development_start_date_idx"
    ):
        raise ValueError("development purge rule is not the registered strict boundary")
    if int(contract.get("label_dependency_overlap_count", -1)) != 0:
        raise ValueError("development metadata reports label dependency overlap")

    candidate_keys = candidates.set_index(["date_idx", "symbol_idx"], drop=False)
    supervised_keys = development.set_index(["date_idx", "symbol_idx"], drop=False)
    missing_supervised = supervised_keys.index.difference(candidate_keys.index)
    if len(missing_supervised):
        raise ValueError("supervised development rows are missing from the full candidate index")
    aligned = candidate_keys.loc[supervised_keys.index]
    if not bool(
        np.equal(
            supervised_keys["trade_date"].astype(str).to_numpy(),
            aligned["trade_date"].astype(str).to_numpy(),
        ).all()
    ) or not bool(
        np.equal(
            supervised_keys["symbol"].astype(str).to_numpy(),
            aligned["symbol"].astype(str).to_numpy(),
        ).all()
    ):
        raise ValueError("supervised/candidate key metadata differs")

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

    sample_sha256 = _sha256_file(sample_path)
    candidate_sha256 = _sha256_file(candidate_path)
    split_payload = {
        "roles": {"fit": "train", "evaluation": "development"},
        "supervised_counts": {
            str(key): int(value) for key, value in samples["split"].value_counts().sort_index().items()
        },
        "candidate_counts": {"development": int(len(candidates))},
        "train_date_idx_min": int(train["date_idx"].astype(int).min()),
        "train_date_idx_max": int(train["date_idx"].astype(int).max()),
        "development_date_idx_min": development_start_idx,
        "development_date_idx_max": int(candidates["date_idx"].astype(int).max()),
        "development_year": development_year,
    }
    contract_payload = {
        key: contract.get(key)
        for key in (
            "schema_version",
            "method",
            "split_roles",
            "train_start_year",
            "development_year",
            "development_start",
            "development_end",
            "source_padding_end",
            "source_padding_trade_date_count",
            "development_start_date_idx",
            "forward_days",
            "execution_tail_days",
            "max_label_dependency_days",
            "safe_train_signal_end",
            "max_train_dependency_end",
            "purge_rule",
            "purged_row_count",
            "purged_signal_date_count",
            "label_dependency_overlap_count",
            "candidate_universe_rule",
            "candidate_count",
            "supervised_development_count",
            "unsupervised_candidate_count",
            "normalization_cutoff_exclusive",
        )
    }
    payload = {
        "schema_version": 1,
        "artifact_type": "seq100_development_fold_training_contract",
        "lookback_days": int(manifest.get("lookback_days", 0) or 0),
        "forward_days": int(manifest.get("forward_days", 0) or 0),
        "max_label_dependency_days": dependency_days,
        "supervised_sample_index": {
            "sha256": sample_sha256,
            "row_count": int(len(samples)),
            "columns": [str(column) for column in samples.columns],
            "dtypes": {str(column): str(dtype) for column, dtype in samples.dtypes.items()},
        },
        "candidate_index": {
            "sha256": candidate_sha256,
            "row_count": int(len(candidates)),
            "columns": [str(column) for column in candidates.columns],
            "dtypes": {str(column): str(dtype) for column, dtype in candidates.dtypes.items()},
            "full_signal_day_universe": True,
            "future_label_required_for_membership": False,
            "entry_fill_required_for_membership": False,
        },
        "split": split_payload,
        "development_walkforward": contract_payload,
        "normalization": normalization,
        "feature_channels": dict(manifest.get("feature_channels", {}) or {}),
        "label_arrays": dict(manifest.get("label_arrays", {}) or {}),
        "label_semantics": dict(manifest.get("label_semantics", {}) or {}),
        "execution_contract": dict(manifest.get("execution_contract", {}) or {}),
        "research_contract": dict(manifest.get("research_contract", {}) or {}),
        "development_contract": dict(manifest.get("development_contract", {}) or {}),
        "masks": dict(manifest.get("masks", {}) or {}),
        "scope": dict(manifest.get("scope", {}) or {}),
        "source_view_provenance": dict(manifest.get("source_view_provenance", {}) or {}),
        "date_values_sha256": _canonical_json_sha256(list(manifest.get("date_values", []) or [])),
        "symbol_values_sha256": _canonical_json_sha256(list(manifest.get("symbol_values", []) or [])),
    }
    return {
        "schema_version": 1,
        "algorithm": "sha256",
        "sha256": _canonical_json_sha256(payload),
        "sample_index_sha256": sample_sha256,
        "candidate_index_sha256": candidate_sha256,
        "normalization_sha256": _canonical_json_sha256(normalization),
        "payload": payload,
    }


def _validated_development_fold_training_contract(
    manifest: Mapping[str, Any],
    *,
    sample_frame: pd.DataFrame | None = None,
    candidate_frame: pd.DataFrame | None = None,
) -> dict[str, Any]:
    stored = dict(manifest.get("development_fold_training_contract", {}) or {})
    if not stored:
        raise ValueError("fold manifest is missing development_fold_training_contract")
    computed = _compute_development_fold_training_contract(
        manifest,
        sample_frame=sample_frame,
        candidate_frame=candidate_frame,
    )
    if stored != computed:
        raise ValueError("development_fold_training_contract does not match current training material")
    return computed


def build_development_walkforward_fold(
    source_view: str | Path = DEFAULT_SOURCE_VIEW,
    development_year: int = 2025,
    train_start_year: int = 2012,
    store_root: str | Path = DEFAULT_STORE_ROOT,
    overwrite: bool = False,
) -> dict[str, Any]:
    source_path = _workspace_path(source_view).resolve()
    source = _read_json(source_path)
    if source.get("artifact_type") != "qdp_v2_sequence_path_pack":
        raise ValueError(f"not a sequence path pack view: {source_path}")
    forward_days = int(source.get("forward_days", 0) or 0)
    dependency_days = _max_label_dependency_days(source)
    execution = dict(source.get("execution_contract", {}) or {})
    execution_tail_days = int(
        execution.get("execution_tail_days", source.get("execution_tail_days", dependency_days - forward_days))
        or 0
    )
    date_values = [str(item) for item in list(source.get("date_values", []) or [])]
    if not date_values or len(date_values) != len(set(date_values)):
        raise ValueError("source date_values must be non-empty and unique")
    sample_path = _workspace_path(str(source.get("sample_index_path", "") or ""))
    candidate_path = _workspace_path(str(source.get("candidate_index_path", "") or ""))
    if not candidate_path.is_file():
        raise FileNotFoundError("development folds require source candidate_index_path")
    source_samples = pd.read_parquet(sample_path)
    source_candidates = pd.read_parquet(candidate_path)
    source_provenance = _source_view_provenance(source_path, source)
    sample_required = {"split", "trade_date", "date_idx", "symbol_idx", "symbol"}
    sample_missing = sorted(sample_required.difference(source_samples.columns))
    candidate_missing = sorted(DEVELOPMENT_CANDIDATE_COLUMNS.difference(source_candidates.columns))
    if sample_missing:
        raise ValueError(f"source supervised index missing columns: {sample_missing}")
    if candidate_missing:
        raise ValueError(f"source candidate index missing columns: {candidate_missing}")
    if bool(source_samples.duplicated(["date_idx", "symbol_idx"]).any()):
        raise ValueError("source supervised index contains duplicate date/symbol rows")
    if bool(source_candidates.duplicated(["date_idx", "symbol_idx"]).any()):
        raise ValueError("source candidate index contains duplicate date/symbol rows")

    def normalize_index(frame: pd.DataFrame, *, name: str) -> pd.DataFrame:
        result = frame.copy()
        result["trade_date"] = result["trade_date"].astype(str)
        result["date_idx"] = result["date_idx"].astype(np.int64)
        if bool(((result["date_idx"] < 0) | (result["date_idx"] >= len(date_values))).any()):
            raise ValueError(f"source {name} date_idx is outside date_values")
        expected = np.asarray(date_values, dtype=object)[result["date_idx"].to_numpy(dtype=np.int64)]
        if not bool(np.equal(result["trade_date"].to_numpy(dtype=object), expected).all()):
            raise ValueError(f"source {name} trade_date does not match date_values[date_idx]")
        return result

    samples = normalize_index(source_samples, name="supervised index")
    candidates = normalize_index(source_candidates, name="candidate index")
    sample_year = samples["trade_date"].str.slice(0, 4).astype(int)
    candidate_year = candidates["trade_date"].str.slice(0, 4).astype(int)
    development_candidate_mask = candidate_year.eq(int(development_year))
    development_sample_mask = sample_year.eq(int(development_year))
    if not bool(development_candidate_mask.any()) or not bool(development_sample_mask.any()):
        raise ValueError(f"source has no candidate/supervised rows for development year {development_year}")
    development_candidates = candidates.loc[development_candidate_mask].copy()
    development_dates = sorted(development_candidates["trade_date"].unique().tolist())
    development_start = str(development_dates[0])
    development_end = str(development_dates[-1])
    date_to_idx = {date: idx for idx, date in enumerate(date_values)}
    development_start_idx = int(date_to_idx[development_start])
    development_end_idx = int(date_to_idx[development_end])
    if development_end_idx + dependency_days >= len(date_values):
        raise ValueError(
            "development candidates require a complete max-label-dependency source padding window"
        )
    source_padding_end = str(date_values[development_end_idx + dependency_days])

    sample_date_idx = samples["date_idx"].to_numpy(dtype=np.int64, copy=False)
    pre_development = sample_date_idx < development_start_idx
    after_train_start = sample_year.to_numpy(dtype=np.int64, copy=False) >= int(train_start_year)
    train_mask = pre_development & after_train_start & (
        sample_date_idx + dependency_days < development_start_idx
    )
    purge_mask = pre_development & after_train_start & (
        sample_date_idx + dependency_days >= development_start_idx
    )
    selected_mask = train_mask | development_sample_mask.to_numpy(dtype=bool, copy=False)
    fold_samples = samples.loc[selected_mask].copy()
    fold_samples.insert(1, "source_split", fold_samples["split"].astype(str).to_numpy(copy=True))
    fold_samples["split"] = np.where(train_mask[selected_mask], "train", "development")
    fold_samples["year"] = fold_samples["trade_date"].str.slice(0, 4).astype(np.int16)
    fold_samples["label_end_date_idx"] = fold_samples["date_idx"].astype(np.int64) + forward_days
    fold_samples["dependency_end_date_idx"] = fold_samples["date_idx"].astype(np.int64) + dependency_days
    fold_samples = fold_samples.sort_values(["date_idx", "symbol_idx"], kind="mergesort").reset_index(drop=True)
    fold_samples["sample_id"] = np.arange(len(fold_samples), dtype=np.int64)
    train_rows = fold_samples[fold_samples["split"].eq("train")]
    development_rows = fold_samples[fold_samples["split"].eq("development")]
    if train_rows.empty or development_rows.empty:
        raise ValueError("development fold requires non-empty train and supervised development splits")
    overlap_count = int(
        (train_rows["dependency_end_date_idx"].astype(int) >= development_start_idx).sum()
    )
    if overlap_count:
        raise AssertionError("purge invariant failed: training dependencies overlap development")

    development_candidates["source_split"] = development_candidates["split"].astype(str)
    development_candidates["split"] = "development"
    development_candidates["year"] = int(development_year)
    development_candidates = development_candidates.sort_values(
        ["date_idx", "symbol_idx"], kind="mergesort"
    ).reset_index(drop=True)
    development_candidates["candidate_id"] = np.arange(len(development_candidates), dtype=np.int64)
    supervised_keys = pd.MultiIndex.from_frame(development_rows[["date_idx", "symbol_idx"]])
    candidate_keys = pd.MultiIndex.from_frame(development_candidates[["date_idx", "symbol_idx"]])
    if len(supervised_keys.difference(candidate_keys)):
        raise ValueError("development candidates do not cover all supervised development rows")

    normalization_start_idx = next(
        (idx for idx, date in enumerate(date_values) if int(str(date)[:4]) >= int(train_start_year)),
        -1,
    )
    if normalization_start_idx < 0 or normalization_start_idx >= development_start_idx:
        raise ValueError("development normalization range is empty")
    normalization = _fit_development_normalization(
        source,
        start_idx=normalization_start_idx,
        end_idx_exclusive=development_start_idx,
        date_values=date_values,
    )

    store_root_path = _workspace_path(store_root)
    view_id = development_view_id(development_year)
    sample_target = store_root_path / "sample_index" / f"{view_id}.parquet"
    candidate_target = store_root_path / "candidate_index" / f"{view_id}.parquet"
    view_target = store_root_path / "views" / f"{view_id}.json"
    if not overwrite and (sample_target.exists() or candidate_target.exists() or view_target.exists()):
        raise FileExistsError(f"development fold already exists for {development_year}")
    _write_parquet(sample_target, fold_samples)
    _write_parquet(candidate_target, development_candidates)

    purge_dates = sorted(samples.loc[purge_mask, "trade_date"].astype(str).unique().tolist())
    safe_train_signal_end = str(train_rows["trade_date"].max())
    max_train_dependency_idx = int(train_rows["dependency_end_date_idx"].max())
    max_train_dependency_end = str(date_values[max_train_dependency_idx])
    development_contract = {
        "schema_version": 1,
        "method": "expanding_train_development_walkforward",
        "split_roles": {"fit": "train", "evaluation": "development"},
        "train_start_year": int(train_start_year),
        "development_year": int(development_year),
        "development_start": development_start,
        "development_end": development_end,
        "source_padding_end": source_padding_end,
        "source_padding_trade_date_count": dependency_days,
        "development_start_date_idx": development_start_idx,
        "forward_days": forward_days,
        "execution_tail_days": execution_tail_days,
        "max_label_dependency_days": dependency_days,
        "safe_train_signal_end": safe_train_signal_end,
        "max_train_dependency_end": max_train_dependency_end,
        "purge_rule": "max_label_dependency_date_idx < development_start_date_idx",
        "purged_row_count": int(purge_mask.sum()),
        "purged_signal_date_count": int(len(purge_dates)),
        "purged_signal_start": str(purge_dates[0]) if purge_dates else "",
        "purged_signal_end": str(purge_dates[-1]) if purge_dates else "",
        "label_dependency_overlap_count": overlap_count,
        "candidate_universe_rule": "signal_day_input_valid_and_signal_eligible",
        "candidate_count": int(len(development_candidates)),
        "supervised_development_count": int(len(development_rows)),
        "unsupervised_candidate_count": int(len(development_candidates) - len(development_rows)),
        "normalization_cutoff_exclusive": development_start,
    }
    artifact_view = dict(source.get("artifact_view", {}) or {})
    artifact_view.update(
        {
            "schema_version": 1,
            "view_id": view_id,
            "view_type": "development_walkforward_research_store_view",
            "source_view": str(source_path),
            "development_audit": development_contract,
        }
    )
    manifest = dict(source)
    source_contract_binding = dict(source.get("research_contract", {}) or {})
    if not source_contract_binding:
        source_contract_binding = approved_development_contract_binding()
    for legacy_key in ("validation_years", "test_years", "oos_years", "purged_walkforward"):
        manifest.pop(legacy_key, None)
    manifest.update(
        {
            "created_at": _now(),
            "start_date": f"{int(train_start_year)}-01-01",
            "end_date": development_end,
            "train_years": sorted({int(value) for value in train_rows["year"].astype(int).tolist()}),
            "development_years": [int(development_year)],
            "split_roles": {"fit": "train", "evaluation": "development"},
            "sample_index_path": str(sample_target.resolve()),
            "sample_count": int(len(fold_samples)),
            "sample_count_by_split": {
                "train": int(len(train_rows)),
                "development": int(len(development_rows)),
            },
            "candidate_index_path": str(candidate_target.resolve()),
            "candidate_count": int(len(development_candidates)),
            "candidate_count_by_split": {"development": int(len(development_candidates))},
            "normalization": normalization,
            "artifact_view": artifact_view,
            "development_walkforward": development_contract,
            "source_view_provenance": source_provenance,
            "research_contract": source_contract_binding,
            "development_contract": source_contract_binding,
            "max_label_dependency_days": dependency_days,
        }
    )
    manifest["development_fold_training_contract"] = _compute_development_fold_training_contract(
        manifest,
        sample_frame=fold_samples,
        candidate_frame=development_candidates,
    )
    _write_json(view_target, manifest)
    verification = verify_development_walkforward_view(view_target)
    if verification["status"] != "ok":
        raise RuntimeError(f"development fold verification failed: {verification['blockers']}")
    return {
        "status": "ok",
        "development_year": int(development_year),
        "view_id": view_id,
        "view_path": str(view_target.resolve()),
        "sample_index_path": str(sample_target.resolve()),
        "candidate_index_path": str(candidate_target.resolve()),
        "sample_count_by_split": manifest["sample_count_by_split"],
        "candidate_count_by_split": manifest["candidate_count_by_split"],
        "development_fold_training_contract": manifest["development_fold_training_contract"],
        "development_walkforward": development_contract,
        "verification": verification,
    }


def verify_development_walkforward_view(view_path: str | Path) -> dict[str, Any]:
    path = _workspace_path(view_path)
    manifest = _read_json(path)
    blockers: list[str] = []
    base_validation = validate_sequence_pack(path)
    blockers.extend(str(item) for item in list(base_validation.get("blockers", []) or []))
    contract = dict(manifest.get("development_walkforward", {}) or {})
    if str(contract.get("method", "")) != "expanding_train_development_walkforward":
        blockers.append("missing_development_walkforward_contract")
    try:
        _validated_source_view_provenance(manifest)
    except (FileNotFoundError, ValueError) as exc:
        blockers.append(f"source_view_provenance:{exc}")
    sample_path = _workspace_path(str(manifest.get("sample_index_path", "") or ""))
    candidate_path = _workspace_path(str(manifest.get("candidate_index_path", "") or ""))
    samples = pd.read_parquet(sample_path) if sample_path.is_file() else pd.DataFrame()
    candidates = pd.read_parquet(candidate_path) if candidate_path.is_file() else pd.DataFrame()
    if samples.empty:
        blockers.append("empty_development_supervised_index")
    if candidates.empty:
        blockers.append("empty_development_candidate_index")
    if not samples.empty and not candidates.empty:
        try:
            _validated_development_fold_training_contract(
                manifest,
                sample_frame=samples,
                candidate_frame=candidates,
            )
        except (FileNotFoundError, ValueError) as exc:
            blockers.append(f"development_fold_training_contract_invalid:{exc}")
    if dict(manifest.get("split_roles", {}) or {}) != {"fit": "train", "evaluation": "development"}:
        blockers.append("development_split_roles_metadata_mismatch")
    development_year = int(contract.get("development_year", 0) or 0)
    if list(manifest.get("development_years", []) or []) != [development_year]:
        blockers.append("development_years_metadata_mismatch")
    if any(key in manifest for key in ("test_years", "oos_years")):
        blockers.append("test_or_oos_semantics_present")
    return {
        "status": "blocked" if blockers else "ok",
        "blockers": blockers,
        "view_path": str(path.resolve()),
        "view_id": str(dict(manifest.get("artifact_view", {}) or {}).get("view_id", path.stem)),
        "development_year": development_year,
        "sample_count_by_split": dict(manifest.get("sample_count_by_split", {}) or {}),
        "candidate_count_by_split": dict(manifest.get("candidate_count_by_split", {}) or {}),
        "label_dependency_overlap_count": int(contract.get("label_dependency_overlap_count", -1)),
        "development_fold_training_contract_sha256": str(
            dict(manifest.get("development_fold_training_contract", {}) or {}).get("sha256", "")
        ),
        "development_walkforward": contract,
    }


def build_development_walkforward_folds(
    *,
    source_view: str | Path = DEFAULT_SOURCE_VIEW,
    development_years: Iterable[int] = DEFAULT_DEVELOPMENT_YEARS,
    train_start_year: int = 2012,
    store_root: str | Path = DEFAULT_STORE_ROOT,
    overwrite: bool = False,
) -> dict[str, Any]:
    years = _parse_years(development_years)
    folds = [
        build_development_walkforward_fold(
            source_view=source_view,
            development_year=int(year),
            train_start_year=int(train_start_year),
            store_root=store_root,
            overwrite=bool(overwrite),
        )
        for year in years
    ]
    return {
        "status": "ok",
        "source_view": str(_workspace_path(source_view).resolve()),
        "development_years": [int(item["development_year"]) for item in folds],
        "folds": folds,
    }


def _clean_values(values: Sequence[float] | np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    return array[np.isfinite(array)]


def hac_mean_interval(
    values: Sequence[float] | np.ndarray,
    *,
    max_lag: int = 59,
    confidence: float = 0.95,
) -> dict[str, Any]:
    array = _clean_values(values)
    n = int(array.size)
    if n == 0:
        raise ValueError("HAC interval requires at least one finite value")
    mean = float(array.mean())
    centered = array - mean
    lag_count = min(max(int(max_lag), 0), n - 1)
    long_run_variance = float(np.dot(centered, centered) / n)
    for lag in range(1, lag_count + 1):
        weight = 1.0 - lag / float(lag_count + 1)
        covariance = float(np.dot(centered[lag:], centered[:-lag]) / n)
        long_run_variance += 2.0 * weight * covariance
    standard_error = math.sqrt(max(long_run_variance, 0.0) / n)
    z_value = NormalDist().inv_cdf(0.5 + float(confidence) / 2.0)
    return {
        "n": n,
        "mean": mean,
        "max_lag": int(lag_count),
        "long_run_variance": float(long_run_variance),
        "standard_error": float(standard_error),
        "se": float(standard_error),
        "confidence": float(confidence),
        "ci_low": float(mean - z_value * standard_error),
        "ci_high": float(mean + z_value * standard_error),
    }


def _moving_block_sample_mean(array: np.ndarray, *, block_length: int, rng: np.random.Generator) -> float:
    n = int(array.size)
    block = min(max(int(block_length), 1), n)
    blocks_needed = int(math.ceil(n / block))
    max_start = n - block
    starts = rng.integers(0, max_start + 1, size=blocks_needed) if max_start > 0 else np.zeros(blocks_needed, dtype=int)
    sample = np.concatenate([array[int(start) : int(start) + block] for start in starts])[:n]
    return float(sample.mean())


def moving_block_bootstrap_mean(
    values: Sequence[float] | np.ndarray,
    *,
    block_length: int = DEFAULT_BLOCK_LENGTH,
    replications: int = DEFAULT_BOOTSTRAP_REPLICATIONS,
    seed: int = DEFAULT_SEED,
    confidence: float = 0.95,
) -> dict[str, Any]:
    array = _clean_values(values)
    if array.size == 0:
        raise ValueError("moving-block bootstrap requires at least one finite value")
    if int(replications) <= 0:
        raise ValueError("replications must be positive")
    rng = np.random.default_rng(int(seed))
    estimates = np.empty(int(replications), dtype=np.float64)
    for idx in range(int(replications)):
        estimates[idx] = _moving_block_sample_mean(array, block_length=int(block_length), rng=rng)
    alpha = (1.0 - float(confidence)) / 2.0
    return {
        "n": int(array.size),
        "mean": float(array.mean()),
        "block_length": int(min(max(int(block_length), 1), int(array.size))),
        "replications": int(replications),
        "seed": int(seed),
        "bootstrap_standard_error": float(estimates.std(ddof=1)) if len(estimates) > 1 else 0.0,
        "bootstrap_std": float(estimates.std(ddof=1)) if len(estimates) > 1 else 0.0,
        "confidence": float(confidence),
        "ci_low": float(np.quantile(estimates, alpha)),
        "ci_high": float(np.quantile(estimates, 1.0 - alpha)),
    }


def _grouped_moving_block_interval(
    groups: Sequence[np.ndarray],
    *,
    block_length: int,
    replications: int,
    seed: int,
    fold_equal: bool,
    confidence: float = 0.95,
) -> dict[str, Any]:
    clean_groups = [_clean_values(group) for group in groups]
    clean_groups = [group for group in clean_groups if group.size]
    if not clean_groups:
        raise ValueError("grouped bootstrap requires finite values")
    rng = np.random.default_rng(int(seed))
    estimates = np.empty(int(replications), dtype=np.float64)
    weights = np.ones(len(clean_groups), dtype=np.float64)
    if not fold_equal:
        weights = np.asarray([group.size for group in clean_groups], dtype=np.float64)
    weights /= weights.sum()
    for replication in range(int(replications)):
        group_means = np.asarray(
            [_moving_block_sample_mean(group, block_length=int(block_length), rng=rng) for group in clean_groups],
            dtype=np.float64,
        )
        estimates[replication] = float(np.dot(group_means, weights))
    observed_means = np.asarray([group.mean() for group in clean_groups], dtype=np.float64)
    observed = float(np.dot(observed_means, weights))
    alpha = (1.0 - float(confidence)) / 2.0
    return {
        "mean": observed,
        "group_count": int(len(clean_groups)),
        "fold_equal": bool(fold_equal),
        "block_length": int(block_length),
        "replications": int(replications),
        "seed": int(seed),
        "bootstrap_standard_error": float(estimates.std(ddof=1)) if len(estimates) > 1 else 0.0,
        "ci_low": float(np.quantile(estimates, alpha)),
        "ci_high": float(np.quantile(estimates, 1.0 - alpha)),
    }


def paired_topk_comparison(
    daily_topk: pd.DataFrame,
    *,
    profile_a: str = "summary_v2_all_channels",
    profile_b: str = "daily_only_summary_v2_ohlcva_aux_low",
    block_length: int = DEFAULT_BLOCK_LENGTH,
    replications: int = DEFAULT_BOOTSTRAP_REPLICATIONS,
    seed: int = DEFAULT_SEED,
) -> pd.DataFrame:
    frame = daily_topk.copy()
    if "score_column" in frame.columns:
        frame = frame[frame["score_column"].astype(str).eq("score")].copy()
    required = {"profile", "oos_year", "seed", "trade_date", "split", "top_k", "universe_count", "universe_hash"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"daily TopK frame missing columns: {missing}")
    key_columns = ["oos_year", "seed", "trade_date", "split", "top_k"]
    for profile in (profile_a, profile_b):
        selected = frame[frame["profile"].astype(str).eq(str(profile))]
        if bool(selected.duplicated(key_columns).any()):
            raise ValueError(f"duplicate daily TopK keys for profile {profile}")
    left = frame[frame["profile"].astype(str).eq(str(profile_a))].copy()
    right = frame[frame["profile"].astype(str).eq(str(profile_b))].copy()
    merged = left.merge(right, on=key_columns, how="outer", suffixes=("_a", "_b"), indicator=True, validate="one_to_one")
    if not bool(merged["_merge"].eq("both").all()):
        raise ValueError("paired TopK profiles have unmatched daily keys")
    if not bool(
        merged["universe_count_a"].astype(int).eq(merged["universe_count_b"].astype(int)).all()
        and merged["universe_hash_a"].astype(str).eq(merged["universe_hash_b"].astype(str)).all()
    ):
        raise ValueError("paired TopK profiles use different daily universes")
    value_candidates = sorted(
        column.removeprefix("alpha_").removesuffix("_a")
        for column in merged.columns
        if column.startswith("alpha_") and column.endswith("_a") and "path_trade_value" in column
    )
    if len(value_candidates) != 1:
        raise ValueError(f"expected one path-value alpha column, got {value_candidates}")
    value_column = value_candidates[0]
    metric_columns = [
        f"alpha_{value_column}",
        "selected_hit_10pct_rate",
        "selected_loss_5pct_rate",
    ]
    rows: list[dict[str, Any]] = []
    for top_k, top_frame in merged.groupby("top_k", sort=True):
        ordered = top_frame.sort_values(["oos_year", "trade_date"], kind="mergesort")
        for metric in metric_columns:
            left_col = f"{metric}_a"
            right_col = f"{metric}_b"
            if left_col not in ordered.columns or right_col not in ordered.columns:
                continue
            left_values = pd.to_numeric(ordered[left_col], errors="coerce").to_numpy(dtype=np.float64)
            right_values = pd.to_numeric(ordered[right_col], errors="coerce").to_numpy(dtype=np.float64)
            if not bool(np.isfinite(left_values).all() and np.isfinite(right_values).all()):
                raise ValueError(f"paired TopK metric contains non-finite values: top_k={int(top_k)} metric={metric}")
            differences = left_values - right_values
            metric_frame = ordered[["oos_year", "trade_date"]].copy()
            metric_frame["difference"] = differences
            year_means = metric_frame.groupby("oos_year", sort=True)["difference"].mean()
            groups = [group["difference"].to_numpy(dtype=np.float64) for _, group in metric_frame.groupby("oos_year", sort=True)]
            hac = hac_mean_interval(metric_frame["difference"].to_numpy(dtype=np.float64), max_lag=int(block_length) - 1)
            pooled_bootstrap = moving_block_bootstrap_mean(
                metric_frame["difference"].to_numpy(dtype=np.float64),
                block_length=int(block_length),
                replications=int(replications),
                seed=int(seed) + int(top_k) * 1009 + len(rows),
            )
            grouped_bootstrap = _grouped_moving_block_interval(
                groups,
                block_length=int(block_length),
                replications=int(replications),
                seed=int(seed) + int(top_k) * 1009 + len(rows),
                fold_equal=False,
            )
            fold_equal_bootstrap = _grouped_moving_block_interval(
                groups,
                block_length=int(block_length),
                replications=int(replications),
                seed=int(seed) + int(top_k) * 2027 + len(rows),
                fold_equal=True,
            )
            rows.append(
                {
                    "profile_a": str(profile_a),
                    "profile_b": str(profile_b),
                    "difference_direction": "profile_a_minus_profile_b",
                    "top_k": int(top_k),
                    "metric": metric,
                    "value_column": value_column,
                    "day_count": int(len(metric_frame)),
                    "fold_count": int(len(year_means)),
                    "pooled_mean_difference": float(metric_frame["difference"].mean()),
                    "fold_equal_mean_difference": float(year_means.mean()),
                    "worst_year_difference": float(year_means.min()),
                    "best_year_difference": float(year_means.max()),
                    "positive_fold_count": int((year_means > 0.0).sum()),
                    "year_differences": json.dumps(
                        {str(int(year)): float(value) for year, value in year_means.items()},
                        ensure_ascii=False,
                    ),
                    "hac_standard_error": float(hac["standard_error"]),
                    "hac_ci_low": float(hac["ci_low"]),
                    "hac_ci_high": float(hac["ci_high"]),
                    "pooled_mbb_ci_low": float(pooled_bootstrap["ci_low"]),
                    "pooled_mbb_ci_high": float(pooled_bootstrap["ci_high"]),
                    "pooled_mbb_method": "continuous_ordered_moving_block_bootstrap",
                    "grouped_pooled_mbb_ci_low": float(grouped_bootstrap["ci_low"]),
                    "grouped_pooled_mbb_ci_high": float(grouped_bootstrap["ci_high"]),
                    "fold_equal_mbb_ci_low": float(fold_equal_bootstrap["ci_low"]),
                    "fold_equal_mbb_ci_high": float(fold_equal_bootstrap["ci_high"]),
                    "block_length": int(block_length),
                    "bootstrap_replications": int(replications),
                }
            )
    return pd.DataFrame(rows)


def paired_ic_comparison(
    daily_ic: pd.DataFrame,
    *,
    profile_a: str = "summary_v2_all_channels",
    profile_b: str = "daily_only_summary_v2_ohlcva_aux_low",
    block_length: int = DEFAULT_BLOCK_LENGTH,
    replications: int = DEFAULT_BOOTSTRAP_REPLICATIONS,
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    frame = daily_ic.copy()
    keys = ["oos_year", "seed", "trade_date", "split"]
    left = frame[frame["profile"].astype(str).eq(str(profile_a))].copy()
    right = frame[frame["profile"].astype(str).eq(str(profile_b))].copy()
    if bool(left.duplicated(keys).any()) or bool(right.duplicated(keys).any()):
        raise ValueError("duplicate daily IC keys")
    merged = left.merge(right, on=keys, how="outer", suffixes=("_a", "_b"), indicator=True, validate="one_to_one")
    if not bool(merged["_merge"].eq("both").all()):
        raise ValueError("paired IC profiles have unmatched daily keys")
    if "count_a" in merged.columns and not bool(merged["count_a"].astype(int).eq(merged["count_b"].astype(int)).all()):
        raise ValueError("paired IC profiles use different daily counts")
    merged = merged.sort_values(["oos_year", "trade_date"], kind="mergesort")
    left_values = pd.to_numeric(merged["rank_ic_a"], errors="coerce").to_numpy(dtype=np.float64)
    right_values = pd.to_numeric(merged["rank_ic_b"], errors="coerce").to_numpy(dtype=np.float64)
    if not bool(np.isfinite(left_values).all() and np.isfinite(right_values).all()):
        raise ValueError("paired IC contains non-finite rank_ic values")
    merged["difference"] = left_values - right_values
    year_means = merged.groupby("oos_year", sort=True)["difference"].mean()
    groups = [group["difference"].to_numpy(dtype=np.float64) for _, group in merged.groupby("oos_year", sort=True)]
    hac = hac_mean_interval(merged["difference"].to_numpy(dtype=np.float64), max_lag=int(block_length) - 1)
    bootstrap = moving_block_bootstrap_mean(
        merged["difference"].to_numpy(dtype=np.float64),
        block_length=int(block_length),
        replications=int(replications),
        seed=int(seed),
    )
    grouped_bootstrap = _grouped_moving_block_interval(
        groups,
        block_length=int(block_length),
        replications=int(replications),
        seed=int(seed),
        fold_equal=False,
    )
    return {
        "profile_a": str(profile_a),
        "profile_b": str(profile_b),
        "difference_direction": "profile_a_minus_profile_b",
        "day_count": int(len(merged)),
        "pooled_mean_difference": float(merged["difference"].mean()),
        "fold_equal_mean_difference": float(year_means.mean()),
        "positive_fold_count": int((year_means > 0.0).sum()),
        "year_differences": {str(int(year)): float(value) for year, value in year_means.items()},
        "hac": hac,
        "moving_block_bootstrap": bootstrap,
        "moving_block_bootstrap_method": "continuous_ordered_moving_block_bootstrap",
        "grouped_pooled_moving_block_bootstrap_sensitivity": grouped_bootstrap,
        "fold_equal_moving_block_bootstrap_sensitivity": _grouped_moving_block_interval(
            groups,
            block_length=int(block_length),
            replications=int(replications),
            seed=int(seed) + 2027,
            fold_equal=True,
        ),
    }


def _study_manifest_path(study_root: str | Path) -> Path:
    return _workspace_path(study_root) / "study_manifest.json"


def _resolved_profile_config(
    profile: str,
    *,
    epochs: int,
    seed: int,
    device: str,
) -> dict[str, Any]:
    from daily_research.path_policy.seq100_mainline import PROFILE_SPECS

    command = PROFILE_COMMANDS[str(profile)]
    resolved = PROFILE_SPECS[command].default_profile()
    return {
        "epochs": int(epochs),
        "batch_size": int(resolved.batch_size),
        "model_type": str(resolved.model_type),
        "hidden_dim": int(resolved.hidden_dim),
        "layers": int(resolved.layers),
        "dropout": float(resolved.dropout),
        "symbol_embedding_dim": 16,
        "learning_rate": float(resolved.learning_rate),
        "weight_decay": float(resolved.weight_decay),
        "path_loss_weight": float(resolved.path_loss_weight),
        "path_loss_profile": str(resolved.path_loss_profile),
        "summary_loss_weight": float(resolved.summary_loss_weight),
        "richer_loss_weight": float(resolved.richer_loss_weight),
        "price_delta_loss_weight": float(resolved.price_delta_loss_weight),
        "va_level_loss_weight": float(resolved.va_level_loss_weight),
        "va_delta_loss_weight": float(resolved.va_delta_loss_weight),
        "value_loss_weight": float(resolved.value_loss_weight),
        "rank_loss_weight": float(resolved.rank_loss_weight),
        "residual_score_weight": 0.25,
        "residual_penalty_weight": 0.01,
        "summary_loss_profile": str(resolved.summary_loss_profile),
        "input_channel_profile": str(resolved.input_channel_profile),
        "direct_value_horizon": int(resolved.direct_value_horizon),
        "rank_max_per_side": int(resolved.rank_max_per_side),
        "path_value_gradient_profile": str(resolved.path_value_gradient_profile),
        "rank_training_profile": str(resolved.rank_training_profile),
        "rank_batch_size": int(resolved.rank_batch_size),
        "rank_interval": int(resolved.rank_interval),
        "prefetch_batches": int(resolved.prefetch_batches),
        "device": str(device),
        "amp": True,
        "seed": int(seed),
        "top_k": [int(item) for item in str(resolved.top_k).split(",") if item],
        "max_samples_per_split": int(resolved.max_samples_per_split),
        "prediction_mode": "none",
        "early_stopping_patience": 0,
        "early_stopping_min_delta": 0.0,
        "evaluation_mode": "fixed_oos",
    }


def _expected_input_channels(input_profile: str) -> list[str]:
    if str(input_profile) == "all":
        return ["daily_raw", "daily_state", "intraday_summary", "limit_structure"]
    if str(input_profile) == "daily_only":
        return ["daily_raw", "daily_state"]
    raise ValueError(f"unsupported walk-forward input profile: {input_profile}")


def _validate_final_checkpoint_provenance(
    summary: Mapping[str, Any],
    *,
    expected_config: Mapping[str, Any],
    expected_view: str,
    expected_run_tag: str,
    epochs: int,
    fold_contract: Mapping[str, Any],
) -> None:
    checkpoint = _workspace_path(str(summary.get("best_checkpoint", "") or ""))
    if checkpoint.name != "final_model.pt":
        raise ValueError("best_checkpoint must be final_model.pt for fixed-OOS provenance")
    if not checkpoint.is_file():
        raise FileNotFoundError(f"best_checkpoint is missing: {checkpoint}")
    try:
        import torch

        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    except Exception as exc:  # pragma: no cover - torch errors vary by release.
        raise ValueError(f"unable to read final checkpoint: {checkpoint}: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise ValueError("final checkpoint is not a mapping payload")
    if str(payload.get("checkpoint_policy", "")) != "final_epoch":
        raise ValueError("final checkpoint policy is not final_epoch")
    if int(payload.get("best_epoch", -1) or -1) != int(epochs):
        raise ValueError("final checkpoint best_epoch does not match completed epochs")
    raw_config = payload.get("config")
    if not isinstance(raw_config, Mapping):
        raise ValueError("final checkpoint is missing its training config")
    checkpoint_config = _canonicalize_json(dict(raw_config))
    if not isinstance(checkpoint_config, dict):
        raise ValueError("final checkpoint config is not a JSON object")
    ignored_config_keys = {"pack_manifest", "output_root", "run_tag"}
    unexpected_config_keys = sorted(set(checkpoint_config) - set(expected_config) - ignored_config_keys)
    if unexpected_config_keys:
        raise ValueError(f"final checkpoint config has unexpected keys: {unexpected_config_keys}")
    resolved_checkpoint_config = {
        key: checkpoint_config.get(key)
        for key in expected_config
    }
    if resolved_checkpoint_config != dict(expected_config):
        raise ValueError("final checkpoint config does not match the registered profile")
    checkpoint_view = _workspace_path(str(checkpoint_config.get("pack_manifest", "") or "")).resolve()
    if str(checkpoint_view) != str(expected_view):
        raise ValueError("final checkpoint pack_manifest does not match the fold view")
    if str(checkpoint_config.get("run_tag", "")) != str(expected_run_tag):
        raise ValueError("final checkpoint run_tag does not match the study entry")
    checkpoint_output_root = _workspace_path(str(checkpoint_config.get("output_root", "") or "")).resolve()
    if checkpoint_output_root != checkpoint.parent.parent.resolve():
        raise ValueError("final checkpoint output_root does not match its run directory")
    checkpoint_feature_channels = payload.get("feature_channels")
    contract_feature_channels = dict(dict(fold_contract.get("payload", {}) or {}).get("feature_channels", {}) or {})
    if checkpoint_feature_channels is not None and _canonicalize_json(checkpoint_feature_channels) != _canonicalize_json(contract_feature_channels):
        raise ValueError("final checkpoint feature metadata does not match the fold contract")


def _validate_summary_provenance(
    summary: Mapping[str, Any],
    *,
    profile: str,
    year: int,
    seed: int,
    epochs: int,
    device: str,
    view_path: str | Path,
    fold_contract: Mapping[str, Any],
    allow_unbound_legacy_preflight: bool = False,
) -> dict[str, Any]:
    expected_config = _resolved_profile_config(
        profile,
        epochs=int(epochs),
        seed=int(seed),
        device=str(device),
    )
    expected_run_tag = f"{profile}_purged_oos{int(year)}_seed{int(seed)}"
    expected_view = str(_workspace_path(view_path).resolve())
    problems: list[str] = []

    exact_fields = {
        "artifact_type": "qdp_v2_sequence_path_training",
        "run_tag": expected_run_tag,
        "seed": int(seed),
        "pack_manifest": expected_view,
        "evaluation_mode": "fixed_oos",
        "evaluation_splits": ["oos"],
        "evaluation_split": "oos",
        "checkpoint_policy": "final_epoch",
        "fold_year": int(year),
        "epochs": int(epochs),
        "completed_epochs": int(epochs),
        "best_epoch": int(epochs),
        "prediction_mode": "none",
        "batch_size": int(expected_config["batch_size"]),
        "max_samples_per_split": int(expected_config["max_samples_per_split"]),
        "input_channel_profile": str(expected_config["input_channel_profile"]),
        "input_channels": _expected_input_channels(str(expected_config["input_channel_profile"])),
        "direct_value_horizon": int(expected_config["direct_value_horizon"]),
    }
    for key, expected in exact_fields.items():
        if summary.get(key) != expected:
            problems.append(f"{key}={summary.get(key)!r} expected={expected!r}")
    if list(summary.get("top_k", []) or []) != list(expected_config["top_k"]):
        problems.append("top_k does not match the registered profile")
    if bool(summary.get("amp_enabled", False)) != bool(expected_config["amp"] and str(summary.get("device", "")) == "cuda"):
        problems.append("amp_enabled does not match the resolved device/config")
    summary_device = str(summary.get("device", ""))
    if str(expected_config["device"]) in {"cpu", "cuda"} and summary_device != str(expected_config["device"]):
        problems.append("device does not match the registered profile")
    if str(expected_config["device"]) == "auto" and summary_device not in {"cpu", "cuda"}:
        problems.append("auto device did not resolve to cpu or cuda")

    payload = dict(fold_contract.get("payload", {}) or {})
    if int(summary.get("lookback_days", 0) or 0) != int(payload.get("lookback_days", 0) or 0):
        problems.append("lookback_days does not match fold contract")
    if int(summary.get("forward_days", 0) or 0) != int(payload.get("forward_days", 0) or 0):
        problems.append("forward_days does not match fold contract")
    purge = dict(payload.get("purge", {}) or {})
    if str(summary.get("train_label_end_before", "")) != str(purge.get("oos_start", "")):
        problems.append("train_label_end_before does not match fold contract")
    if str(summary.get("normalization_cutoff", "")) != str(purge.get("normalization_cutoff_exclusive", "")):
        problems.append("normalization_cutoff does not match fold contract")

    model = dict(summary.get("model", {}) or {})
    expected_model = {
        "type": f"SequencePathModel_{expected_config['model_type']}",
        "hidden_dim": int(expected_config["hidden_dim"]),
        "layers": int(expected_config["layers"]),
        "dropout": float(expected_config["dropout"]),
    }
    for key, expected in expected_model.items():
        if model.get(key) != expected:
            problems.append(f"model.{key}={model.get(key)!r} expected={expected!r}")
    expected_loss = {
        "path": float(expected_config["path_loss_weight"]),
        "path_profile": str(expected_config["path_loss_profile"]),
        "summary": float(expected_config["summary_loss_weight"]),
        "richer": float(expected_config["richer_loss_weight"]),
        "price_delta": float(expected_config["price_delta_loss_weight"]),
        "value": float(expected_config["value_loss_weight"]),
        "rank": float(expected_config["rank_loss_weight"]),
        "rank_max_per_side": int(expected_config["rank_max_per_side"]),
        "summary_profile": str(expected_config["summary_loss_profile"]),
        "va_level": float(expected_config["va_level_loss_weight"]),
        "va_delta": float(expected_config["va_delta_loss_weight"]),
    }
    loss_weights = dict(summary.get("loss_weights", {}) or {})
    for key, expected in expected_loss.items():
        if loss_weights.get(key) != expected:
            problems.append(f"loss_weights.{key}={loss_weights.get(key)!r} expected={expected!r}")
    early_stopping = dict(summary.get("early_stopping", {}) or {})
    if int(early_stopping.get("patience", -1)) != 0:
        problems.append("early_stopping.patience must be zero")
    if float(early_stopping.get("min_delta", math.nan)) != 0.0:
        problems.append("early_stopping.min_delta must be zero")

    stored_contract = dict(summary.get("fold_training_contract", {}) or {})
    if stored_contract != dict(fold_contract) and not (allow_unbound_legacy_preflight and not stored_contract):
        problems.append("fold_training_contract is missing or mismatched")
    recorded_config = summary.get("resolved_training_config")
    if recorded_config is None:
        if not allow_unbound_legacy_preflight:
            problems.append("resolved_training_config is missing")
    else:
        normalized_recorded = dict(recorded_config)
        # Historical completed runs predate the additive training-core controls.
        # Their missing values have the exact legacy/default semantics below.
        normalized_recorded.setdefault("path_value_gradient_profile", "smooth_current")
        normalized_recorded.setdefault("rank_training_profile", "local_chunk")
        normalized_recorded.setdefault("rank_batch_size", 512)
        normalized_recorded.setdefault("rank_interval", 4)
        normalized_recorded.setdefault("prefetch_batches", 1)
        if _canonicalize_json(normalized_recorded) != expected_config:
            problems.append("resolved_training_config does not match the registered profile")

    checkpoint = _workspace_path(str(summary.get("best_checkpoint", "") or ""))
    if not checkpoint.is_file():
        problems.append(f"best_checkpoint is missing: {checkpoint}")
    else:
        if _workspace_path(str(summary.get("output_dir", "") or "")).resolve() != checkpoint.parent.resolve():
            problems.append("output_dir does not match the final checkpoint directory")
        try:
            _validate_final_checkpoint_provenance(
                summary,
                expected_config=expected_config,
                expected_view=expected_view,
                expected_run_tag=expected_run_tag,
                epochs=int(epochs),
                fold_contract=fold_contract,
            )
        except (FileNotFoundError, ValueError) as exc:
            problems.append(str(exc))
    outputs = dict(summary.get("outputs", {}) or {})
    output_filenames = {
        "split_metrics_csv": "split_metrics.csv",
        "topk_metrics_csv": "topk_metrics.csv",
        "daily_rank_ic_csv": "daily_rank_ic.csv",
        "daily_topk_metrics_csv": "daily_topk_metrics.csv",
        "topk_candidates_parquet": "topk_candidates.parquet",
        "training_history_csv": "training_history.csv",
    }
    for name, filename in output_filenames.items():
        target = _workspace_path(str(outputs.get(name, "") or ""))
        if not target.is_file():
            problems.append(f"output is missing: {name}")
        elif checkpoint.is_file() and target.resolve() != (checkpoint.parent / filename).resolve():
            problems.append(f"output path does not match the validated run directory: {name}")
    if problems:
        raise ValueError("training provenance mismatch: " + "; ".join(problems))
    return expected_config


def _command_option(command: Sequence[str], name: str) -> str:
    try:
        index = list(command).index(name)
    except ValueError as exc:
        raise ValueError(f"study ledger command is missing {name}") from exc
    if index + 1 >= len(command):
        raise ValueError(f"study ledger command has no value for {name}")
    return str(command[index + 1])


def _validate_fold_contract_binding(
    binding: Mapping[str, Any],
    *,
    fold_contract: Mapping[str, Any],
) -> dict[str, Any]:
    normalized = dict(binding or {})
    required_keys = {
        "schema_version",
        "method",
        "bound_at",
        "pre_correction_full_view_sha256",
        "sample_index_sha256",
        "normalization_sha256",
        "fold_training_contract_sha256",
        "provenance_validation",
    }
    if set(normalized) != required_keys:
        raise ValueError("legacy fold contract binding schema is not exact")
    if int(normalized.get("schema_version", 0) or 0) != 1:
        raise ValueError("legacy fold contract binding schema version is invalid")
    if str(normalized.get("method", "")) != FOLD_TRAINING_CONTRACT_BINDING_METHOD:
        raise ValueError("legacy fold contract binding method is invalid")
    if not str(normalized.get("bound_at", "")):
        raise ValueError("legacy fold contract binding has no bound_at timestamp")
    pre_hash = str(normalized.get("pre_correction_full_view_sha256", ""))
    if len(pre_hash) != 64 or any(character not in "0123456789abcdef" for character in pre_hash.lower()):
        raise ValueError("legacy fold contract binding has an invalid pre-correction hash")
    expected_hashes = {
        "sample_index_sha256": str(fold_contract.get("sample_index_sha256", "")),
        "normalization_sha256": str(fold_contract.get("normalization_sha256", "")),
        "fold_training_contract_sha256": str(fold_contract.get("sha256", "")),
    }
    for key, expected in expected_hashes.items():
        if str(normalized.get(key, "")) != expected:
            raise ValueError(f"legacy fold contract binding digest mismatch: {key}")
    return normalized


def _summary_contract_binding(binding: Mapping[str, Any]) -> dict[str, Any]:
    return {
        **dict(binding),
        "resolved_training_config_source": "final_checkpoint_config",
    }


def _validate_completed_entry(
    entry: Mapping[str, Any],
    *,
    manifest: Mapping[str, Any],
    fold_manifest: Mapping[str, Any],
    validated_fold_contract: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if str(entry.get("status", "")) != "completed":
        raise ValueError("study ledger entry is not completed")
    profile = str(entry.get("profile", ""))
    year = int(entry.get("oos_year", 0) or 0)
    seed = int(manifest.get("seed", -1))
    epochs = int(manifest.get("epochs", 0) or 0)
    device = str(manifest.get("device", ""))
    expected_view = purged_view_path(year, store_root=str(manifest.get("store_root", ""))).resolve()
    if str(entry.get("profile_command", "")) != PROFILE_COMMANDS.get(profile, ""):
        raise ValueError("study ledger profile command mismatch")
    if int(entry.get("seed", -1)) != seed:
        raise ValueError("study ledger seed mismatch")
    if _workspace_path(str(entry.get("view_path", "") or "")).resolve() != expected_view:
        raise ValueError("study ledger view path mismatch")
    fold_contract = (
        dict(validated_fold_contract)
        if validated_fold_contract is not None
        else _validated_fold_training_contract(fold_manifest)
    )
    if dict(entry.get("fold_training_contract", {}) or {}) != fold_contract:
        raise ValueError("study ledger fold_training_contract is missing or mismatched")

    summary_path = _workspace_path(str(entry.get("summary_path", "") or ""))
    run_dir = _workspace_path(str(entry.get("run_dir", "") or ""))
    if not summary_path.is_file() or summary_path.resolve() != (run_dir / "sequence_path_training_summary.json").resolve():
        raise ValueError("study ledger summary/run directory is missing or inconsistent")
    summary = _read_json(summary_path)
    summary_binding = dict(summary.get("fold_training_contract_binding", {}) or {})
    entry_binding = dict(entry.get("fold_training_contract_binding", {}) or {})
    fold_binding = dict(fold_manifest.get("fold_training_contract_binding", {}) or {})
    if summary_binding or entry_binding or fold_binding:
        if not summary_binding or not entry_binding or not fold_binding:
            raise ValueError("legacy fold contract binding is incomplete across fold/summary/ledger")
        validated_binding = _validate_fold_contract_binding(fold_binding, fold_contract=fold_contract)
        if entry_binding != validated_binding:
            raise ValueError("legacy fold contract binding differs between fold and ledger")
        if summary_binding != _summary_contract_binding(validated_binding):
            raise ValueError("legacy fold contract binding differs between fold and summary")
    expected_config = _validate_summary_provenance(
        summary,
        profile=profile,
        year=year,
        seed=seed,
        epochs=epochs,
        device=device,
        view_path=expected_view,
        fold_contract=fold_contract,
    )
    if dict(entry.get("resolved_profile_config", {}) or {}) != expected_config:
        raise ValueError("study ledger resolved_profile_config is missing or mismatched")
    command = [str(item) for item in list(entry.get("command", []) or [])]
    if len(command) < 4 or command[2:4] != ["daily_research.path_policy.seq100_mainline", PROFILE_COMMANDS[profile]]:
        raise ValueError("study ledger training command/profile mismatch")
    expected_options = {
        "--store-view": str(expected_view),
        "--run-tag": f"{profile}_purged_oos{year}_seed{seed}",
        "--epochs": str(epochs),
        "--device": str(device),
        "--seed": str(seed),
        "--early-stopping-patience": "0",
        "--prediction-mode": "none",
        "--evaluation-mode": "fixed_oos",
    }
    for name, expected in expected_options.items():
        if _command_option(command, name) != expected:
            raise ValueError(f"study ledger command option mismatch for {name}")
    return summary


def _load_or_initialize_study(
    *,
    study_root: str | Path,
    source_view: str | Path,
    years: Sequence[int],
    profiles: Sequence[str],
    seed: int,
    epochs: int,
    train_start_year: int,
    store_root: str | Path,
    device: str,
) -> dict[str, Any]:
    path = _study_manifest_path(study_root)
    if path.exists():
        payload = _read_json(path)
        expected = {
            "source_view": str(_workspace_path(source_view).resolve()),
            "oos_years": [int(year) for year in years],
            "profiles": [str(profile) for profile in profiles],
            "seed": int(seed),
            "epochs": int(epochs),
            "train_start_year": int(train_start_year),
            "store_root": str(_workspace_path(store_root).resolve()),
            "device": str(device),
            "run_tag": STUDY_RUN_TAG,
        }
        for key, value in expected.items():
            if payload.get(key) != value:
                raise ValueError(f"existing study manifest mismatch for {key}")
        return payload
    payload = {
        "schema_version": 2,
        "artifact_type": "seq100_purged_walkforward_study_manifest",
        "created_at": _now(),
        "updated_at": _now(),
        "source_view": str(_workspace_path(source_view).resolve()),
        "oos_years": [int(year) for year in years],
        "profiles": [str(profile) for profile in profiles],
        "seed": int(seed),
        "epochs": int(epochs),
        "train_start_year": int(train_start_year),
        "store_root": str(_workspace_path(store_root).resolve()),
        "device": str(device),
        "run_tag": STUDY_RUN_TAG,
        "evaluation_mode": "fixed_oos",
        "early_stopping_patience": 0,
        "prediction_mode": "none",
        "entries": {},
    }
    _write_json(path, payload)
    return payload


def run_walkforward_study(
    *,
    source_view: str | Path = DEFAULT_SOURCE_VIEW,
    oos_years: Iterable[int] = DEFAULT_OOS_YEARS,
    profiles: Sequence[str] = LEGACY_REQUIRED_PROFILES,
    study_root: str | Path = DEFAULT_STUDY_ROOT,
    store_root: str | Path = DEFAULT_STORE_ROOT,
    train_start_year: int = 2012,
    seed: int = DEFAULT_SEED,
    epochs: int = 1,
    device: str = "cuda",
    python_executable: str | Path = DEFAULT_PYTHON,
    overwrite_folds: bool = False,
    rerun_completed: bool = False,
    bootstrap_replications: int = DEFAULT_BOOTSTRAP_REPLICATIONS,
) -> dict[str, Any]:
    if bool(overwrite_folds) and not bool(rerun_completed):
        raise ValueError("--overwrite-folds requires --rerun-completed")
    years = _parse_years(oos_years)
    profile_values = _normalize_profiles(profiles)
    study_root_path = _workspace_path(study_root)
    runs_root = study_root_path / "runs"
    logs_root = study_root_path / "logs"
    runs_root.mkdir(parents=True, exist_ok=True)
    logs_root.mkdir(parents=True, exist_ok=True)
    manifest = _load_or_initialize_study(
        study_root=study_root_path,
        source_view=source_view,
        years=years,
        profiles=profile_values,
        seed=int(seed),
        epochs=int(epochs),
        train_start_year=int(train_start_year),
        store_root=store_root,
        device=str(device),
    )
    entries = dict(manifest.get("entries", {}) or {})

    for year in years:
        view_path = purged_view_path(year, store_root=store_root)
        if view_path.exists() and not bool(overwrite_folds):
            verification = verify_purged_walkforward_view(view_path)
            if verification["status"] != "ok":
                raise RuntimeError(f"existing fold {year} failed verification: {verification['blockers']}")
        else:
            build_purged_walkforward_fold(
                source_view=source_view,
                oos_year=int(year),
                train_start_year=int(train_start_year),
                store_root=store_root,
                overwrite=bool(overwrite_folds),
            )
        fold_manifest = _read_json(view_path)
        fold_contract = _validated_fold_training_contract(fold_manifest)
        for profile in profile_values:
            key = f"{profile}:oos{int(year)}:seed{int(seed)}"
            existing = dict(entries.get(key, {}) or {})
            if str(existing.get("status", "")) == "completed" and not bool(rerun_completed):
                _validate_completed_entry(
                    existing,
                    manifest=manifest,
                    fold_manifest=fold_manifest,
                    validated_fold_contract=fold_contract,
                )
                continue
            resolved_profile = _resolved_profile_config(
                profile,
                epochs=int(epochs),
                seed=int(seed),
                device=str(device),
            )
            run_tag = f"{profile}_purged_oos{int(year)}_seed{int(seed)}"
            log_path = logs_root / f"{run_tag}.log"
            command = [
                str(_workspace_path(python_executable)),
                "-m",
                "daily_research.path_policy.seq100_mainline",
                PROFILE_COMMANDS[profile],
                "--store-view",
                str(view_path.resolve()),
                "--output-root",
                str(runs_root.resolve()),
                "--run-tag",
                run_tag,
                "--epochs",
                str(int(epochs)),
                "--device",
                str(device),
                "--seed",
                str(int(seed)),
                "--early-stopping-patience",
                "0",
                "--prediction-mode",
                "none",
                "--evaluation-mode",
                "fixed_oos",
                "--json",
            ]
            existing_summaries = {
                path.resolve()
                for path in runs_root.glob(f"{run_tag}_*/sequence_path_training_summary.json")
            }
            entries[key] = {
                "status": "running",
                "profile": profile,
                "profile_command": PROFILE_COMMANDS[profile],
                "oos_year": int(year),
                "seed": int(seed),
                "view_path": str(view_path.resolve()),
                "fold_training_contract": fold_contract,
                "resolved_profile_config": resolved_profile,
                "run_tag": run_tag,
                "command": command,
                "log_path": str(log_path.resolve()),
                "started_at": _now(),
            }
            manifest["entries"] = entries
            manifest["updated_at"] = _now()
            _write_json(_study_manifest_path(study_root_path), manifest)
            with log_path.open("w", encoding="utf-8") as log_handle:
                completed = subprocess.run(
                    command,
                    cwd=WORKSPACE_ROOT,
                    stdout=log_handle,
                    stderr=subprocess.STDOUT,
                    text=True,
                    check=False,
                )
            if int(completed.returncode) != 0:
                entries[key]["status"] = "failed"
                entries[key]["returncode"] = int(completed.returncode)
                entries[key]["finished_at"] = _now()
                manifest["entries"] = entries
                manifest["updated_at"] = _now()
                _write_json(_study_manifest_path(study_root_path), manifest)
                raise RuntimeError(f"training failed for {key}; see {log_path}")
            candidates = sorted(
                (
                    path
                    for path in runs_root.glob(f"{run_tag}_*/sequence_path_training_summary.json")
                    if path.resolve() not in existing_summaries
                ),
                key=lambda item: item.stat().st_mtime,
            )
            if not candidates:
                raise FileNotFoundError(f"completed run did not write a summary for {key}")
            summary_path = candidates[-1]
            summary = _read_json(summary_path)
            try:
                _validate_summary_provenance(
                    summary,
                    profile=profile,
                    year=int(year),
                    seed=int(seed),
                    epochs=int(epochs),
                    device=str(device),
                    view_path=view_path,
                    fold_contract=fold_contract,
                )
            except ValueError as exc:
                raise RuntimeError(f"training provenance mismatch for {key}: {exc}") from exc
            entries[key].update(
                {
                    "status": "completed",
                    "returncode": 0,
                    "finished_at": _now(),
                    "run_dir": str(summary_path.parent.resolve()),
                    "summary_path": str(summary_path.resolve()),
                    "fold_training_contract": fold_contract,
                    "resolved_profile_config": resolved_profile,
                }
            )
            manifest["entries"] = entries
            manifest["updated_at"] = _now()
            _write_json(_study_manifest_path(study_root_path), manifest)
    return summarize_walkforward_study(
        study_root=study_root_path,
        bootstrap_replications=int(bootstrap_replications),
        block_length=DEFAULT_BLOCK_LENGTH,
        bootstrap_seed=int(seed),
    )


def _new_fold_contract_binding(view_path: Path, fold_contract: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "method": FOLD_TRAINING_CONTRACT_BINDING_METHOD,
        "bound_at": _now(),
        "pre_correction_full_view_sha256": _sha256_file(view_path),
        "sample_index_sha256": str(fold_contract["sample_index_sha256"]),
        "normalization_sha256": str(fold_contract["normalization_sha256"]),
        "fold_training_contract_sha256": str(fold_contract["sha256"]),
        "provenance_validation": "checkpoint config, recorded run fields, and required outputs validated before binding",
    }


def bind_legacy_walkforward_run_contracts(
    *,
    study_root: str | Path = DEFAULT_STUDY_ROOT,
    store_root: str | Path = DEFAULT_STORE_ROOT,
    train_start_year: int = 2012,
    device: str = "cuda",
) -> dict[str, Any]:
    study_root_path = _workspace_path(study_root)
    manifest_path = _study_manifest_path(study_root_path)
    manifest = _read_json(manifest_path)
    years = _parse_years(list(manifest.get("oos_years", []) or []))
    profiles = _normalize_profiles(list(manifest.get("profiles", []) or []))
    seed = int(manifest.get("seed", -1))
    epochs = int(manifest.get("epochs", 0) or 0)
    if seed < 0 or epochs <= 0:
        raise ValueError("legacy study manifest has invalid seed or epochs")
    resolved_store_root = _workspace_path(store_root).resolve()
    for key, expected in {
        "store_root": str(resolved_store_root),
        "train_start_year": int(train_start_year),
        "device": str(device),
        "run_tag": STUDY_RUN_TAG,
    }.items():
        existing = manifest.get(key)
        if existing is not None and existing != expected:
            raise ValueError(f"legacy study manifest mismatch for {key}")

    entries = dict(manifest.get("entries", {}) or {})
    expected_keys = {
        f"{profile}:oos{int(year)}:seed{seed}"
        for year in years
        for profile in profiles
    }
    if set(entries) != expected_keys:
        raise ValueError("legacy study ledger entries do not match the registered year/profile matrix")

    folds: dict[int, dict[str, Any]] = {}
    planned_summaries: list[dict[str, Any]] = []
    for year in years:
        view_path = purged_view_path(int(year), store_root=resolved_store_root)
        if not view_path.is_file():
            raise FileNotFoundError(f"missing legacy fold view: {view_path}")
        fold_manifest = _read_json(view_path)
        computed_contract = _compute_fold_training_contract(fold_manifest)
        stored_contract = dict(fold_manifest.get("fold_training_contract", {}) or {})
        stored_binding = dict(fold_manifest.get("fold_training_contract_binding", {}) or {})
        if stored_contract:
            if stored_contract != computed_contract:
                raise ValueError(f"fold {year} stored contract differs from current training material")
            if not stored_binding:
                raise ValueError(f"fold {year} has a contract without its legacy binding")
            binding = _validate_fold_contract_binding(stored_binding, fold_contract=computed_contract)
        else:
            if stored_binding:
                raise ValueError(f"fold {year} has a legacy binding without its contract")
            binding = _new_fold_contract_binding(view_path, computed_contract)
        actual_train_years = list(dict(computed_contract["payload"]).get("split", {}).get("train_years", []) or [])
        if not actual_train_years or int(min(actual_train_years)) != int(train_start_year):
            raise ValueError(f"fold {year} train_start_year does not match the requested migration contract")
        folds[int(year)] = {
            "view_path": view_path,
            "manifest": fold_manifest,
            "contract": computed_contract,
            "binding": binding,
        }

        for profile in profiles:
            key = f"{profile}:oos{int(year)}:seed{seed}"
            entry = dict(entries[key] or {})
            if str(entry.get("status", "")) != "completed":
                raise ValueError(f"legacy entry is not completed: {key}")
            if str(entry.get("profile", "")) != profile or str(entry.get("profile_command", "")) != PROFILE_COMMANDS[profile]:
                raise ValueError(f"legacy entry profile mismatch: {key}")
            if int(entry.get("oos_year", 0) or 0) != int(year) or int(entry.get("seed", -1)) != seed:
                raise ValueError(f"legacy entry fold/seed mismatch: {key}")
            if _workspace_path(str(entry.get("view_path", "") or "")).resolve() != view_path.resolve():
                raise ValueError(f"legacy entry view path mismatch: {key}")
            command = [str(item) for item in list(entry.get("command", []) or [])]
            if len(command) < 4 or command[2:4] != ["daily_research.path_policy.seq100_mainline", PROFILE_COMMANDS[profile]]:
                raise ValueError(f"legacy entry command mismatch: {key}")
            for name, expected in {
                "--store-view": str(view_path.resolve()),
                "--run-tag": f"{profile}_purged_oos{int(year)}_seed{seed}",
                "--epochs": str(epochs),
                "--device": str(device),
                "--seed": str(seed),
                "--early-stopping-patience": "0",
                "--prediction-mode": "none",
                "--evaluation-mode": "fixed_oos",
            }.items():
                if _command_option(command, name) != expected:
                    raise ValueError(f"legacy entry command option mismatch for {key}: {name}")
            summary_path = _workspace_path(str(entry.get("summary_path", "") or ""))
            run_dir = _workspace_path(str(entry.get("run_dir", "") or ""))
            if not summary_path.is_file() or summary_path.resolve() != (run_dir / "sequence_path_training_summary.json").resolve():
                raise ValueError(f"legacy entry summary/run directory mismatch: {key}")
            summary = _read_json(summary_path)
            resolved_profile = _validate_summary_provenance(
                summary,
                profile=profile,
                year=int(year),
                seed=seed,
                epochs=epochs,
                device=str(device),
                view_path=view_path,
                fold_contract=computed_contract,
                allow_unbound_legacy_preflight=True,
            )
            entry_contract = dict(entry.get("fold_training_contract", {}) or {})
            if entry_contract and entry_contract != computed_contract:
                raise ValueError(f"legacy entry contract mismatch: {key}")
            entry_config = entry.get("resolved_profile_config")
            if entry_config is not None and _canonicalize_json(entry_config) != resolved_profile:
                raise ValueError(f"legacy entry resolved config mismatch: {key}")
            entry_binding = dict(entry.get("fold_training_contract_binding", {}) or {})
            if entry_binding and entry_binding != binding:
                raise ValueError(f"legacy entry binding mismatch: {key}")
            summary_binding = dict(summary.get("fold_training_contract_binding", {}) or {})
            if summary_binding and summary_binding != _summary_contract_binding(binding):
                raise ValueError(f"legacy summary binding mismatch: {key}")
            planned_summaries.append(
                {
                    "key": key,
                    "entry": entry,
                    "summary_path": summary_path,
                    "summary": summary,
                    "resolved_profile": resolved_profile,
                    "fold_contract": computed_contract,
                    "binding": binding,
                }
            )

    existing_migration = dict(manifest.get("fold_training_contract_binding_migration", {}) or {})
    if existing_migration:
        if (
            int(existing_migration.get("schema_version", 0) or 0) != 1
            or str(existing_migration.get("method", "")) != FOLD_TRAINING_CONTRACT_BINDING_METHOD
            or int(existing_migration.get("bound_fold_count", -1)) != len(folds)
            or int(existing_migration.get("bound_run_count", -1)) != len(planned_summaries)
            or dict(existing_migration.get("folds", {}) or {})
            != {str(year): dict(fold["binding"]) for year, fold in sorted(folds.items())}
        ):
            raise ValueError("existing legacy binding migration record is inconsistent")

    changed_file_count = 0
    for fold in folds.values():
        desired_fold = dict(fold["manifest"])
        desired_fold["fold_training_contract"] = fold["contract"]
        desired_fold["fold_training_contract_binding"] = fold["binding"]
        if desired_fold != fold["manifest"]:
            _write_json(fold["view_path"], desired_fold)
            changed_file_count += 1
    for planned in planned_summaries:
        desired_summary = dict(planned["summary"])
        desired_summary["fold_training_contract"] = planned["fold_contract"]
        desired_summary["resolved_training_config"] = planned["resolved_profile"]
        desired_summary["fold_training_contract_binding"] = _summary_contract_binding(planned["binding"])
        if desired_summary != planned["summary"]:
            _write_json(planned["summary_path"], desired_summary)
            changed_file_count += 1
        desired_entry = dict(planned["entry"])
        desired_entry["fold_training_contract"] = planned["fold_contract"]
        desired_entry["resolved_profile_config"] = planned["resolved_profile"]
        desired_entry["fold_training_contract_binding"] = planned["binding"]
        entries[str(planned["key"])] = desired_entry

    completed_at = str(existing_migration.get("completed_at", "") or _now())
    migration_record = {
        "schema_version": 1,
        "method": FOLD_TRAINING_CONTRACT_BINDING_METHOD,
        "completed_at": completed_at,
        "bound_fold_count": int(len(folds)),
        "bound_run_count": int(len(planned_summaries)),
        "folds": {
            str(year): dict(fold["binding"])
            for year, fold in sorted(folds.items())
        },
    }
    desired_manifest = dict(manifest)
    desired_manifest.update(
        {
            "schema_version": 2,
            "store_root": str(resolved_store_root),
            "train_start_year": int(train_start_year),
            "device": str(device),
            "run_tag": STUDY_RUN_TAG,
            "entries": entries,
            "fold_training_contract_binding_migration": migration_record,
        }
    )
    if desired_manifest != manifest:
        desired_manifest["updated_at"] = _now()
        _write_json(manifest_path, desired_manifest)
        changed_file_count += 1
    manifest = desired_manifest

    for year, fold in folds.items():
        persisted_fold = _read_json(fold["view_path"])
        for profile in profiles:
            key = f"{profile}:oos{int(year)}:seed{seed}"
            _validate_completed_entry(
                entries[key],
                manifest=manifest,
                fold_manifest=persisted_fold,
                validated_fold_contract=fold["contract"],
            )
    return {
        "status": "completed",
        "method": FOLD_TRAINING_CONTRACT_BINDING_METHOD,
        "study_manifest": str(manifest_path.resolve()),
        "bound_fold_count": int(len(folds)),
        "bound_run_count": int(len(planned_summaries)),
        "changed_file_count": int(changed_file_count),
        "fold_training_contract_sha256": {
            str(year): str(fold["contract"]["sha256"])
            for year, fold in sorted(folds.items())
        },
    }


def _assert_fold_sample_frames_equal(
    live_frame: pd.DataFrame,
    staged_frame: pd.DataFrame,
    *,
    year: int,
) -> None:
    try:
        pd.testing.assert_frame_equal(
            live_frame.reset_index(drop=True),
            staged_frame.reset_index(drop=True),
            check_dtype=True,
            check_exact=True,
            check_like=False,
        )
    except AssertionError as exc:
        raise ValueError(f"staged fold {year} sample frame differs from the bound live fold: {exc}") from exc


def reconcile_bound_walkforward_fold_metadata(
    *,
    study_root: str | Path = DEFAULT_STUDY_ROOT,
    staging_root: str | Path | None = None,
) -> dict[str, Any]:
    study_root_path = _workspace_path(study_root).resolve()
    manifest_path = _study_manifest_path(study_root_path)
    manifest = _read_json(manifest_path)
    years = _parse_years(list(manifest.get("oos_years", []) or []))
    profiles = _normalize_profiles(list(manifest.get("profiles", []) or []))
    seed = int(manifest.get("seed", -1))
    train_start_year = int(manifest.get("train_start_year", 0) or 0)
    store_root_value = str(manifest.get("store_root", "") or "")
    source_view_value = str(manifest.get("source_view", "") or "")
    if seed < 0 or train_start_year <= 0 or not store_root_value or not source_view_value:
        raise ValueError("study manifest is missing bound reconciliation provenance")
    store_root_path = _workspace_path(store_root_value).resolve()
    source_view_path = _workspace_path(source_view_value).resolve()
    staging_root_path = (
        _workspace_path(staging_root).resolve()
        if staging_root is not None
        else (study_root_path / "fold_metadata_reconciliation_staging").resolve()
    )
    if staging_root_path == store_root_path:
        raise ValueError("metadata reconciliation staging_root must differ from the live research store")
    audit_path = study_root_path / "fold_metadata_reconciliation.json"
    existing_audit = _read_json(audit_path) if audit_path.is_file() else {}
    existing_manifest_record = dict(manifest.get("fold_metadata_reconciliation", {}) or {})

    live_folds: dict[int, dict[str, Any]] = {}
    entries = dict(manifest.get("entries", {}) or {})
    for year in years:
        live_view_path = purged_view_path(int(year), store_root=store_root_path).resolve()
        live_manifest = _read_json(live_view_path)
        live_contract = _validated_fold_training_contract(live_manifest)
        live_binding = _validate_fold_contract_binding(
            dict(live_manifest.get("fold_training_contract_binding", {}) or {}),
            fold_contract=live_contract,
        )
        for profile in profiles:
            key = f"{profile}:oos{int(year)}:seed{seed}"
            if key not in entries:
                raise ValueError(f"study ledger is missing reconciliation entry: {key}")
            _validate_completed_entry(
                dict(entries[key] or {}),
                manifest=manifest,
                fold_manifest=live_manifest,
                validated_fold_contract=live_contract,
            )
        live_sample_path = _workspace_path(str(live_manifest.get("sample_index_path", "") or "")).resolve()
        live_folds[int(year)] = {
            "view_path": live_view_path,
            "manifest": live_manifest,
            "contract": live_contract,
            "binding": live_binding,
            "sample_path": live_sample_path,
            "sample_sha256": _sha256_file(live_sample_path),
            "view_sha256": _sha256_file(live_view_path),
        }

    if str(existing_audit.get("status", "")) == "completed":
        if (
            str(existing_audit.get("method", "")) != FOLD_METADATA_RECONCILIATION_METHOD
            or str(existing_audit.get("study_manifest", "")) != str(manifest_path.resolve())
            or dict(existing_manifest_record) != dict(existing_audit.get("manifest_record", {}) or {})
        ):
            raise ValueError("completed fold metadata reconciliation audit is inconsistent")
        audit_folds = dict(existing_audit.get("folds", {}) or {})
        for year, live in live_folds.items():
            record = dict(audit_folds.get(str(year), {}) or {})
            if (
                str(record.get("post_view_sha256", "")) != str(live["view_sha256"])
                or str(record.get("live_sample_sha256", "")) != str(live["sample_sha256"])
                or str(record.get("fold_training_contract_sha256", "")) != str(live["contract"]["sha256"])
            ):
                raise ValueError(f"completed fold metadata reconciliation drifted for {year}")
        return {
            **existing_audit,
            "changed_view_count": 0,
            "idempotent_recheck": True,
        }

    if existing_audit and str(existing_audit.get("status", "")) != "planned":
        raise ValueError("fold metadata reconciliation audit has an unsupported partial status")
    planned_at = str(existing_audit.get("planned_at", "") or _now())
    audit_folds_existing = dict(existing_audit.get("folds", {}) or {})
    staged_folds: dict[int, dict[str, Any]] = {}
    metadata_fields = (
        "start_date",
        "end_date",
        "train_years",
        "validation_years",
        "test_years",
        "oos_years",
        "split_roles",
        "sample_count",
        "sample_count_by_split",
        "normalization",
        "artifact_view",
        "purged_walkforward",
    )
    for year in years:
        build_purged_walkforward_fold(
            source_view=source_view_path,
            oos_year=int(year),
            train_start_year=int(train_start_year),
            store_root=staging_root_path,
            overwrite=True,
        )
        staged_view_path = purged_view_path(int(year), store_root=staging_root_path).resolve()
        staged_manifest = _read_json(staged_view_path)
        staged_contract = _validated_fold_training_contract(staged_manifest)
        live = live_folds[int(year)]
        staged_sample_path = _workspace_path(str(staged_manifest.get("sample_index_path", "") or "")).resolve()
        live_frame = pd.read_parquet(live["sample_path"])
        staged_frame = pd.read_parquet(staged_sample_path)
        _assert_fold_sample_frames_equal(live_frame, staged_frame, year=int(year))
        if _canonicalize_json(staged_manifest.get("normalization", {})) != _canonicalize_json(
            live["manifest"].get("normalization", {})
        ):
            raise ValueError(f"staged fold {year} normalization differs from the bound live fold")
        if staged_contract != live["contract"]:
            raise ValueError(f"staged fold {year} material contract differs from the bound live fold")

        existing_fold_plan = dict(audit_folds_existing.get(str(year), {}) or {})
        pre_view_sha256 = str(existing_fold_plan.get("pre_view_sha256", "") or live["view_sha256"])
        desired_manifest = dict(live["manifest"])
        for field in metadata_fields:
            desired_manifest[field] = staged_manifest[field]
        desired_manifest["sample_index_path"] = str(live["sample_path"])
        desired_manifest["fold_training_contract"] = live["contract"]
        desired_manifest["fold_training_contract_binding"] = live["binding"]
        desired_manifest["metadata_reconciliation"] = {
            "schema_version": 1,
            "method": FOLD_METADATA_RECONCILIATION_METHOD,
            "planned_at": planned_at,
            "audit_path": str(audit_path.resolve()),
            "pre_view_sha256": pre_view_sha256,
            "fold_training_contract_sha256": str(live["contract"]["sha256"]),
        }
        post_view_sha256 = _serialized_json_sha256(desired_manifest)
        staged_sample_sha256 = _sha256_file(staged_sample_path)
        fold_plan = {
            "oos_year": int(year),
            "live_view_path": str(live["view_path"]),
            "staged_view_path": str(staged_view_path),
            "live_sample_index_path": str(live["sample_path"]),
            "staged_sample_index_path": str(staged_sample_path),
            "sample_row_count": int(len(live_frame)),
            "pre_view_sha256": pre_view_sha256,
            "post_view_sha256": post_view_sha256,
            "live_sample_sha256": str(live["sample_sha256"]),
            "staged_sample_sha256": staged_sample_sha256,
            "fold_training_contract_sha256": str(live["contract"]["sha256"]),
            "normalization_sha256": str(live["contract"]["normalization_sha256"]),
        }
        if existing_fold_plan and existing_fold_plan != fold_plan:
            raise ValueError(f"partial reconciliation plan changed for fold {year}")
        if str(live["view_sha256"]) not in {pre_view_sha256, post_view_sha256}:
            raise ValueError(f"live fold {year} is neither the planned pre- nor post-reconciliation view")
        staged_folds[int(year)] = {
            "desired_manifest": desired_manifest,
            "plan": fold_plan,
            "live": live,
        }

    plan = {
        "schema_version": 1,
        "artifact_type": "seq100_bound_fold_metadata_reconciliation",
        "status": "planned",
        "method": FOLD_METADATA_RECONCILIATION_METHOD,
        "planned_at": planned_at,
        "study_manifest": str(manifest_path.resolve()),
        "study_manifest_pre_sha256": str(
            existing_audit.get("study_manifest_pre_sha256", "") or _sha256_file(manifest_path)
        ),
        "source_view": str(source_view_path),
        "store_root": str(store_root_path),
        "staging_root": str(staging_root_path),
        "oos_years": list(years),
        "folds": {
            str(year): dict(staged["plan"])
            for year, staged in sorted(staged_folds.items())
        },
    }
    if existing_audit:
        if existing_audit != plan:
            raise ValueError("partial reconciliation audit differs from the reconstructed plan")
    else:
        _write_json(audit_path, plan)

    changed_view_count = 0
    for year, staged in sorted(staged_folds.items()):
        current_sha256 = _sha256_file(staged["live"]["view_path"])
        if current_sha256 == str(staged["plan"]["pre_view_sha256"]):
            _write_json(staged["live"]["view_path"], staged["desired_manifest"])
            changed_view_count += 1
        elif current_sha256 != str(staged["plan"]["post_view_sha256"]):
            raise RuntimeError(f"live fold {year} changed after reconciliation planning")
        if _sha256_file(staged["live"]["view_path"]) != str(staged["plan"]["post_view_sha256"]):
            raise RuntimeError(f"live fold {year} post-reconciliation hash mismatch")
        if _sha256_file(staged["live"]["sample_path"]) != str(staged["plan"]["live_sample_sha256"]):
            raise RuntimeError(f"live fold {year} sample parquet changed during metadata reconciliation")

    completed_at = str(existing_manifest_record.get("completed_at", "") or _now())
    manifest_record = {
        "schema_version": 1,
        "method": FOLD_METADATA_RECONCILIATION_METHOD,
        "status": "completed",
        "planned_at": planned_at,
        "completed_at": completed_at,
        "audit_path": str(audit_path.resolve()),
        "staging_root": str(staging_root_path),
        "folds": {
            str(year): {
                "pre_view_sha256": str(staged["plan"]["pre_view_sha256"]),
                "post_view_sha256": str(staged["plan"]["post_view_sha256"]),
                "live_sample_sha256": str(staged["plan"]["live_sample_sha256"]),
                "fold_training_contract_sha256": str(staged["plan"]["fold_training_contract_sha256"]),
            }
            for year, staged in sorted(staged_folds.items())
        },
    }
    if existing_manifest_record and existing_manifest_record != manifest_record:
        raise ValueError("existing study manifest reconciliation record is inconsistent")
    desired_study_manifest = dict(manifest)
    desired_study_manifest["fold_metadata_reconciliation"] = manifest_record
    if desired_study_manifest != manifest:
        desired_study_manifest["updated_at"] = _now()
        _write_json(manifest_path, desired_study_manifest)
    completed_audit = {
        **plan,
        "status": "completed",
        "completed_at": completed_at,
        "study_manifest_post_sha256": _sha256_file(manifest_path),
        "manifest_record": manifest_record,
    }
    _write_json(audit_path, completed_audit)
    return {
        **completed_audit,
        "changed_view_count": int(changed_view_count),
        "idempotent_recheck": False,
    }


def summarize_walkforward_study(
    *,
    study_root: str | Path = DEFAULT_STUDY_ROOT,
    bootstrap_replications: int = DEFAULT_BOOTSTRAP_REPLICATIONS,
    block_length: int = DEFAULT_BLOCK_LENGTH,
    bootstrap_seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    study_root_path = _workspace_path(study_root)
    manifest = _read_json(_study_manifest_path(study_root_path))
    years = _parse_years(list(manifest.get("oos_years", []) or []))
    profiles = _normalize_profiles(list(manifest.get("profiles", []) or []))
    store_root_value = str(manifest.get("store_root", "") or "")
    if not store_root_value:
        raise ValueError("study manifest is missing store_root; bind/migrate legacy runs explicitly")
    store_root_path = _workspace_path(store_root_value).resolve()
    entries_by_key = dict(manifest.get("entries", {}) or {})
    expected_keys = {
        f"{profile}:oos{int(year)}:seed{int(manifest.get('seed', -1))}"
        for year in years
        for profile in profiles
    }
    if set(entries_by_key) != expected_keys:
        raise RuntimeError("study ledger does not match the registered year/profile matrix")
    completed: list[dict[str, Any]] = []
    fold_manifests: dict[int, dict[str, Any]] = {}
    for year in years:
        fold_path = purged_view_path(int(year), store_root=store_root_path)
        fold_manifest = _read_json(fold_path)
        fold_manifests[int(year)] = fold_manifest
        fold_contract = _validated_fold_training_contract(fold_manifest)
        for profile in profiles:
            key = f"{profile}:oos{int(year)}:seed{int(manifest.get('seed', -1))}"
            entry = dict(entries_by_key[key] or {})
            try:
                _validate_completed_entry(
                    entry,
                    manifest=manifest,
                    fold_manifest=fold_manifest,
                    validated_fold_contract=fold_contract,
                )
            except (FileNotFoundError, ValueError) as exc:
                raise RuntimeError(f"invalid completed study entry {key}: {exc}") from exc
            completed.append(entry)
    split_frames: list[pd.DataFrame] = []
    topk_frames: list[pd.DataFrame] = []
    daily_topk_frames: list[pd.DataFrame] = []
    daily_ic_frames: list[pd.DataFrame] = []
    run_rows: list[dict[str, Any]] = []
    value_columns: set[str] = set()
    for entry in sorted(completed, key=lambda item: (int(item["oos_year"]), str(item["profile"]))):
        run_dir = _workspace_path(str(entry["run_dir"]))
        summary = _read_json(run_dir / "sequence_path_training_summary.json")
        value_columns.add(str(summary.get("value_column", "")))
        metadata = {
            "profile": str(entry["profile"]),
            "oos_year": int(entry["oos_year"]),
            "seed": int(entry["seed"]),
        }
        split_frames.append(pd.read_csv(run_dir / "split_metrics.csv").assign(**metadata))
        topk_frames.append(pd.read_csv(run_dir / "topk_metrics.csv").assign(**metadata))
        daily_topk_frames.append(pd.read_csv(run_dir / "daily_topk_metrics.csv").assign(**metadata))
        daily_ic_frames.append(pd.read_csv(run_dir / "daily_rank_ic.csv").assign(**metadata))
        run_rows.append(
            {
                **metadata,
                "run_dir": str(run_dir.resolve()),
                "summary_path": str((run_dir / "sequence_path_training_summary.json").resolve()),
                "input_channel_profile": str(summary.get("input_channel_profile", "")),
                "input_dim": int(dict(summary.get("model", {}) or {}).get("input_dim", 0) or 0),
                "checkpoint_policy": str(summary.get("checkpoint_policy", "")),
                "completed_epochs": int(summary.get("completed_epochs", 0) or 0),
                "value_column": str(summary.get("value_column", "")),
            }
        )
    if len(value_columns) != 1:
        raise ValueError(f"walk-forward runs use inconsistent value columns: {sorted(value_columns)}")
    value_column = next(iter(value_columns))
    split_metrics = pd.concat(split_frames, ignore_index=True)
    topk_metrics = pd.concat(topk_frames, ignore_index=True)
    daily_topk = pd.concat(daily_topk_frames, ignore_index=True)
    daily_ic = pd.concat(daily_ic_frames, ignore_index=True)
    paired_topk = paired_topk_comparison(
        daily_topk,
        block_length=int(block_length),
        replications=int(bootstrap_replications),
        seed=int(bootstrap_seed),
    )
    paired_ic = paired_ic_comparison(
        daily_ic,
        block_length=int(block_length),
        replications=int(bootstrap_replications),
        seed=int(bootstrap_seed),
    )
    runs_path = study_root_path / "walkforward_runs.csv"
    split_path = study_root_path / "fold_split_metrics.csv"
    topk_path = study_root_path / "fold_topk_metrics.csv"
    daily_topk_path = study_root_path / "daily_topk_all_folds.csv"
    daily_ic_path = study_root_path / "daily_rank_ic_all_folds.csv"
    comparison_path = study_root_path / "paired_profile_comparison.csv"
    pd.DataFrame(run_rows).to_csv(runs_path, index=False, encoding="utf-8-sig")
    split_metrics.to_csv(split_path, index=False, encoding="utf-8-sig")
    topk_metrics.to_csv(topk_path, index=False, encoding="utf-8-sig")
    daily_topk.to_csv(daily_topk_path, index=False, encoding="utf-8-sig")
    daily_ic.to_csv(daily_ic_path, index=False, encoding="utf-8-sig")
    paired_topk.to_csv(comparison_path, index=False, encoding="utf-8-sig")

    primary = paired_topk[
        paired_topk["top_k"].astype(int).eq(3)
        & paired_topk["metric"].astype(str).eq(f"alpha_{value_column}")
    ]
    if len(primary) != 1:
        raise RuntimeError("paired comparison did not produce the registered Top3 primary metric")
    primary_row = primary.iloc[0].to_dict()
    fold_top3 = topk_metrics[topk_metrics["top_k"].astype(int).eq(3)].copy()
    alpha_column = f"alpha_{value_column}"
    profile_fold_summary: list[dict[str, Any]] = []
    for profile, group in fold_top3.groupby("profile", sort=True):
        values = pd.to_numeric(group[alpha_column], errors="coerce")
        profile_fold_summary.append(
            {
                "profile": str(profile),
                "fold_count": int(len(group)),
                "fold_equal_top3_alpha_mean": float(values.mean()),
                "worst_fold_top3_alpha": float(values.min()),
                "positive_fold_count": int((values > 0.0).sum()),
                "year_values": {
                    str(int(row["oos_year"])): float(row[alpha_column])
                    for row in group.sort_values("oos_year").to_dict("records")
                },
            }
        )
    fold_audits = {
        str(year): dict(fold_manifests[int(year)].get("purged_walkforward", {}) or {})
        for year in years
    }
    summary = {
        "schema_version": 1,
        "artifact_type": "seq100_purged_walkforward_study",
        "status": "completed",
        "run_tag": str(manifest.get("run_tag", STUDY_RUN_TAG)),
        "generated_at": _now(),
        "study_root": str(study_root_path.resolve()),
        "store_root": str(store_root_path),
        "source_view": manifest.get("source_view", ""),
        "oos_years": list(manifest.get("oos_years", []) or []),
        "profiles": list(manifest.get("profiles", []) or []),
        "seed": int(manifest.get("seed", DEFAULT_SEED)),
        "epochs": int(manifest.get("epochs", 1)),
        "evaluation_mode": "fixed_oos",
        "checkpoint_policy": "final_epoch",
        "value_column": value_column,
        "primary_metric": "Top3 daily path-value alpha; summary_v2_all_channels minus daily-only default",
        "primary_inference_method": "continuous ordered 60-day moving-block bootstrap across the full paired OOS date sequence",
        "primary_result": primary_row,
        "paired_ic": paired_ic,
        "profile_fold_summary": profile_fold_summary,
        "fold_audits": fold_audits,
        "residual_bias": (
            "The source sample index requires complete future-60d labels. Each fold records the signal-day candidate rows "
            "excluded by label_valid; results are complete-case OOS evidence, not a full execution backtest."
        ),
        "evidence_verdict": {
            "promotion_allowed": False,
            "decision": "retain_daily_only_summary_v2_ohlcva_aux_low_as_default",
            "default_profile_changed": False,
            "reason": "The all-channel bundle did not establish stable dominance across purged OOS folds.",
            "active_execution_artifact_changed": False,
            "active_execution_policy": "unchanged",
        },
        "outputs": {
            "study_manifest_json": str(_study_manifest_path(study_root_path).resolve()),
            "walkforward_runs_csv": str(runs_path.resolve()),
            "fold_split_metrics_csv": str(split_path.resolve()),
            "fold_topk_metrics_csv": str(topk_path.resolve()),
            "daily_topk_all_folds_csv": str(daily_topk_path.resolve()),
            "daily_rank_ic_all_folds_csv": str(daily_ic_path.resolve()),
            "paired_profile_comparison_csv": str(comparison_path.resolve()),
        },
    }
    report_lines = [
        "# Seq100 Purged Walk-Forward 2022-2025",
        "",
        "## Contract",
        "",
        "- Expanding train window; each train label ends strictly before the first OOS trading day.",
        "- Fixed one-epoch checkpoint; OOS is read once after training and never selects weights.",
        "- Top3 path-value alpha is the registered primary comparison metric.",
        "",
        "## Fold Top3",
        "",
    ]
    for row in fold_top3.sort_values(["profile", "oos_year"]).to_dict("records"):
        report_lines.append(
            f"- {row['profile']} / {int(row['oos_year'])}: alpha={float(row[alpha_column]):.6f}, "
            f"rank_ic={float(split_metrics[(split_metrics['profile'] == row['profile']) & (split_metrics['oos_year'] == row['oos_year'])]['rank_ic_mean'].iloc[0]):.6f}"
        )
    report_lines.extend(
        [
            "",
            "## Paired Primary Result",
            "",
            f"- pooled mean difference: {float(primary_row['pooled_mean_difference']):.6f}",
            f"- fold-equal mean difference: {float(primary_row['fold_equal_mean_difference']):.6f}",
            f"- HAC 95% CI: [{float(primary_row['hac_ci_low']):.6f}, {float(primary_row['hac_ci_high']):.6f}]",
            f"- primary continuous ordered 60-day MBB 95% CI: [{float(primary_row['pooled_mbb_ci_low']):.6f}, {float(primary_row['pooled_mbb_ci_high']):.6f}]",
            f"- grouped-by-year pooled MBB sensitivity 95% CI: [{float(primary_row['grouped_pooled_mbb_ci_low']):.6f}, {float(primary_row['grouped_pooled_mbb_ci_high']):.6f}]",
            f"- fold-equal MBB sensitivity 95% CI: [{float(primary_row['fold_equal_mbb_ci_low']):.6f}, {float(primary_row['fold_equal_mbb_ci_high']):.6f}]",
            f"- positive folds: {int(primary_row['positive_fold_count'])}/{int(primary_row['fold_count'])}",
            "",
            "## Boundary",
            "",
            f"- {summary['residual_bias']}",
            "- Promotion verdict: not allowed; the daily-only low-VA profile remains the default.",
            "- This study does not change active execution artifacts or QDP active pointers.",
        ]
    )
    report_path = study_root_path / "walkforward_report.md"
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    summary["outputs"]["report_md"] = str(report_path.resolve())
    summary_path = _write_json(study_root_path / "study_summary.json", summary)
    summary["outputs"]["study_summary_json"] = str(summary_path.resolve())
    _write_json(summary_path, summary)
    return summary


def run_inner_screen_study(
    *,
    source_view: str | Path = DEFAULT_SOURCE_VIEW,
    study_root: str | Path = "daily_research/output/path_policy/studies/seq100_pit_inner_2018_2021",
    store_root: str | Path = DEFAULT_STORE_ROOT,
    train_start_year: int = 2012,
    seed: int = DEFAULT_SEED,
    epochs: int = 1,
    device: str = "cuda",
    python_executable: str | Path = DEFAULT_PYTHON,
    overwrite_folds: bool = False,
    rerun_completed: bool = False,
    bootstrap_replications: int = DEFAULT_BOOTSTRAP_REPLICATIONS,
) -> dict[str, Any]:
    """Run only the frozen 2018-2021 inner profile screen.

    This entry point deliberately excludes 2022-2025. Those years remain sealed
    until a champion is frozen from the inner evidence.
    """

    return run_walkforward_study(
        source_view=source_view,
        oos_years=(2018, 2019, 2020, 2021),
        profiles=INNER_SCREEN_PROFILES,
        study_root=study_root,
        store_root=store_root,
        train_start_year=int(train_start_year),
        seed=int(seed),
        epochs=int(epochs),
        device=str(device),
        python_executable=python_executable,
        overwrite_folds=bool(overwrite_folds),
        rerun_completed=bool(rerun_completed),
        bootstrap_replications=int(bootstrap_replications),
    )


def _print(payload: Any, *, as_json: bool) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default) if as_json else payload)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Purged expanding walk-forward utilities for the seq100 path model.")
    sub = parser.add_subparsers(dest="command", required=True)

    build = sub.add_parser("build-fold")
    build.add_argument("--source-view", type=Path, default=DEFAULT_SOURCE_VIEW)
    build.add_argument("--oos-year", type=int, required=True)
    build.add_argument("--train-start-year", type=int, default=2012)
    build.add_argument("--store-root", type=Path, default=DEFAULT_STORE_ROOT)
    build.add_argument("--overwrite", action="store_true")
    build.add_argument("--json", action="store_true")

    build_all = sub.add_parser("build-folds")
    build_all.add_argument("--source-view", type=Path, default=DEFAULT_SOURCE_VIEW)
    build_all.add_argument("--oos-years", required=True)
    build_all.add_argument("--train-start-year", type=int, default=2012)
    build_all.add_argument("--store-root", type=Path, default=DEFAULT_STORE_ROOT)
    build_all.add_argument("--overwrite", action="store_true")
    build_all.add_argument("--json", action="store_true")

    verify = sub.add_parser("verify")
    verify.add_argument("--view", type=Path, required=True)
    verify.add_argument("--json", action="store_true")

    build_development = sub.add_parser("build-development-fold")
    build_development.add_argument("--source-view", type=Path, default=DEFAULT_SOURCE_VIEW)
    build_development.add_argument("--development-year", type=int, required=True)
    build_development.add_argument("--train-start-year", type=int, default=2012)
    build_development.add_argument("--store-root", type=Path, default=DEFAULT_STORE_ROOT)
    build_development.add_argument("--overwrite", action="store_true")
    build_development.add_argument("--json", action="store_true")

    build_all_development = sub.add_parser("build-development-folds")
    build_all_development.add_argument("--source-view", type=Path, default=DEFAULT_SOURCE_VIEW)
    build_all_development.add_argument(
        "--development-years", default=",".join(map(str, DEFAULT_DEVELOPMENT_YEARS))
    )
    build_all_development.add_argument("--train-start-year", type=int, default=2012)
    build_all_development.add_argument("--store-root", type=Path, default=DEFAULT_STORE_ROOT)
    build_all_development.add_argument("--overwrite", action="store_true")
    build_all_development.add_argument("--json", action="store_true")

    verify_development = sub.add_parser("verify-development")
    verify_development.add_argument("--view", type=Path, required=True)
    verify_development.add_argument("--json", action="store_true")

    run = sub.add_parser("run-study")
    run.add_argument("--source-view", type=Path, default=DEFAULT_SOURCE_VIEW)
    run.add_argument("--oos-years", required=True)
    run.add_argument("--profiles", default=",".join(LEGACY_REQUIRED_PROFILES))
    run.add_argument("--study-root", type=Path, default=DEFAULT_STUDY_ROOT)
    run.add_argument("--store-root", type=Path, default=DEFAULT_STORE_ROOT)
    run.add_argument("--train-start-year", type=int, default=2012)
    run.add_argument("--seed", type=int, default=DEFAULT_SEED)
    run.add_argument("--epochs", type=int, default=1)
    run.add_argument("--device", default="cuda", choices=("cpu", "cuda", "auto"))
    run.add_argument("--python", type=Path, default=DEFAULT_PYTHON)
    run.add_argument("--overwrite-folds", action="store_true")
    run.add_argument("--rerun-completed", action="store_true")
    run.add_argument("--bootstrap-replications", type=int, default=DEFAULT_BOOTSTRAP_REPLICATIONS)
    run.add_argument("--json", action="store_true")

    inner = sub.add_parser("run-inner-screen")
    inner.add_argument("--source-view", type=Path, default=DEFAULT_SOURCE_VIEW)
    inner.add_argument("--study-root", type=Path, default=Path("daily_research/output/path_policy/studies/seq100_pit_inner_2018_2021"))
    inner.add_argument("--store-root", type=Path, default=DEFAULT_STORE_ROOT)
    inner.add_argument("--train-start-year", type=int, default=2012)
    inner.add_argument("--seed", type=int, default=DEFAULT_SEED)
    inner.add_argument("--epochs", type=int, default=1)
    inner.add_argument("--device", default="cuda", choices=("cpu", "cuda", "auto"))
    inner.add_argument("--python", type=Path, default=DEFAULT_PYTHON)
    inner.add_argument("--overwrite-folds", action="store_true")
    inner.add_argument("--rerun-completed", action="store_true")
    inner.add_argument("--bootstrap-replications", type=int, default=DEFAULT_BOOTSTRAP_REPLICATIONS)
    inner.add_argument("--json", action="store_true")

    bind = sub.add_parser(
        "bind-legacy-run-contracts",
        help="Explicitly validate and bind pre-contract completed runs without retraining.",
    )
    bind.add_argument("--study-root", type=Path, default=DEFAULT_STUDY_ROOT)
    bind.add_argument("--store-root", type=Path, default=DEFAULT_STORE_ROOT)
    bind.add_argument("--train-start-year", type=int, default=2012)
    bind.add_argument("--device", default="cuda", choices=("cpu", "cuda", "auto"))
    bind.add_argument("--json", action="store_true")

    reconcile = sub.add_parser(
        "reconcile-bound-fold-metadata",
        help="Stage and compare rebuilt folds, then update only bound live view metadata.",
    )
    reconcile.add_argument("--study-root", type=Path, default=DEFAULT_STUDY_ROOT)
    reconcile.add_argument("--staging-root", type=Path)
    reconcile.add_argument("--json", action="store_true")

    summarize = sub.add_parser("summarize-study")
    summarize.add_argument("--study-root", type=Path, default=DEFAULT_STUDY_ROOT)
    summarize.add_argument("--bootstrap-replications", type=int, default=DEFAULT_BOOTSTRAP_REPLICATIONS)
    summarize.add_argument("--block-length", type=int, default=DEFAULT_BLOCK_LENGTH)
    summarize.add_argument("--bootstrap-seed", type=int, default=DEFAULT_SEED)
    summarize.add_argument("--json", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command == "build-fold":
        result = build_purged_walkforward_fold(
            source_view=args.source_view,
            oos_year=int(args.oos_year),
            train_start_year=int(args.train_start_year),
            store_root=args.store_root,
            overwrite=bool(args.overwrite),
        )
    elif args.command == "build-folds":
        result = build_purged_walkforward_folds(
            source_view=args.source_view,
            oos_years=_parse_years(str(args.oos_years)),
            train_start_year=int(args.train_start_year),
            store_root=args.store_root,
            overwrite=bool(args.overwrite),
        )
    elif args.command == "verify":
        result = verify_purged_walkforward_view(args.view)
    elif args.command == "build-development-fold":
        result = build_development_walkforward_fold(
            source_view=args.source_view,
            development_year=int(args.development_year),
            train_start_year=int(args.train_start_year),
            store_root=args.store_root,
            overwrite=bool(args.overwrite),
        )
    elif args.command == "build-development-folds":
        result = build_development_walkforward_folds(
            source_view=args.source_view,
            development_years=_parse_years(str(args.development_years)),
            train_start_year=int(args.train_start_year),
            store_root=args.store_root,
            overwrite=bool(args.overwrite),
        )
    elif args.command == "verify-development":
        result = verify_development_walkforward_view(args.view)
    elif args.command == "run-study":
        result = run_walkforward_study(
            source_view=args.source_view,
            oos_years=_parse_years(str(args.oos_years)),
            profiles=tuple(item.strip() for item in str(args.profiles).split(",") if item.strip()),
            study_root=args.study_root,
            store_root=args.store_root,
            train_start_year=int(args.train_start_year),
            seed=int(args.seed),
            epochs=int(args.epochs),
            device=str(args.device),
            python_executable=args.python,
            overwrite_folds=bool(args.overwrite_folds),
            rerun_completed=bool(args.rerun_completed),
            bootstrap_replications=int(args.bootstrap_replications),
        )
    elif args.command == "run-inner-screen":
        result = run_inner_screen_study(
            source_view=args.source_view,
            study_root=args.study_root,
            store_root=args.store_root,
            train_start_year=int(args.train_start_year),
            seed=int(args.seed),
            epochs=int(args.epochs),
            device=str(args.device),
            python_executable=args.python,
            overwrite_folds=bool(args.overwrite_folds),
            rerun_completed=bool(args.rerun_completed),
            bootstrap_replications=int(args.bootstrap_replications),
        )
    elif args.command == "bind-legacy-run-contracts":
        result = bind_legacy_walkforward_run_contracts(
            study_root=args.study_root,
            store_root=args.store_root,
            train_start_year=int(args.train_start_year),
            device=str(args.device),
        )
    elif args.command == "reconcile-bound-fold-metadata":
        result = reconcile_bound_walkforward_fold_metadata(
            study_root=args.study_root,
            staging_root=args.staging_root,
        )
    else:
        result = summarize_walkforward_study(
            study_root=args.study_root,
            bootstrap_replications=int(args.bootstrap_replications),
            block_length=int(args.block_length),
            bootstrap_seed=int(args.bootstrap_seed),
        )
    _print(result, as_json=bool(args.json))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
