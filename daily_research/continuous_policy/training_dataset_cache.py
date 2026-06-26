from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from daily_research.continuous_policy.runtime import CONTINUOUS_POLICY_ROOT, now_iso


TRAINING_DATASET_CACHE_ROOT = CONTINUOUS_POLICY_ROOT / "training_datasets"
TRAINING_DATASET_CACHE_VERSION = 1


@dataclass(frozen=True)
class TrainingDatasetCacheRecord:
    cache_key: str
    cache_dir: Path
    sample_frame: pd.DataFrame
    daily_frame: pd.DataFrame
    teacher_summary: dict[str, Any]
    metadata: dict[str, Any]


@dataclass(frozen=True)
class ReusableTrainingDatasetRecord:
    store: str
    cache_key: str
    cache_dir: Path
    sample_frame: pd.DataFrame
    daily_frame: pd.DataFrame
    teacher_summary: dict[str, Any]
    metadata: dict[str, Any]


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def build_training_dataset_fingerprint(spec: Mapping[str, Any]) -> str:
    payload = {
        "cache_version": TRAINING_DATASET_CACHE_VERSION,
        "spec": _json_safe(dict(spec)),
    }
    text = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:24]


def _paths(cache_root: Path, cache_key: str) -> dict[str, Path]:
    cache_dir = Path(cache_root) / str(cache_key)
    return {
        "cache_dir": cache_dir,
        "metadata": cache_dir / "metadata.json",
        "sample_frame": cache_dir / "sample_frame.pkl",
        "daily_frame": cache_dir / "daily_frame.pkl",
        "teacher_summary": cache_dir / "teacher_summary.json",
    }


def save_training_dataset_cache(
    *,
    cache_root: str | Path | None = None,
    spec: Mapping[str, Any],
    sample_frame: pd.DataFrame,
    daily_frame: pd.DataFrame,
    teacher_summary: Mapping[str, Any],
) -> TrainingDatasetCacheRecord:
    root = Path(cache_root) if cache_root is not None else TRAINING_DATASET_CACHE_ROOT
    cache_key = build_training_dataset_fingerprint(spec)
    paths = _paths(root, cache_key)
    cache_dir = paths["cache_dir"]
    cache_dir.mkdir(parents=True, exist_ok=True)
    sample_frame.to_pickle(paths["sample_frame"])
    daily_frame.to_pickle(paths["daily_frame"])
    safe_teacher = _json_safe(dict(teacher_summary or {}))
    paths["teacher_summary"].write_text(json.dumps(safe_teacher, ensure_ascii=False, indent=2), encoding="utf-8")
    metadata = {
        "cache_version": TRAINING_DATASET_CACHE_VERSION,
        "cache_key": cache_key,
        "created_at": now_iso(),
        "spec": _json_safe(dict(spec)),
        "sample_rows": int(len(sample_frame)),
        "daily_rows": int(len(daily_frame)),
        "sample_frame_path": str(paths["sample_frame"].resolve()),
        "daily_frame_path": str(paths["daily_frame"].resolve()),
        "teacher_summary_path": str(paths["teacher_summary"].resolve()),
    }
    paths["metadata"].write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return TrainingDatasetCacheRecord(
        cache_key=cache_key,
        cache_dir=cache_dir,
        sample_frame=sample_frame,
        daily_frame=daily_frame,
        teacher_summary=dict(safe_teacher),
        metadata=metadata,
    )


def load_training_dataset_cache(
    *,
    cache_root: str | Path | None = None,
    spec: Mapping[str, Any],
) -> TrainingDatasetCacheRecord | None:
    root = Path(cache_root) if cache_root is not None else TRAINING_DATASET_CACHE_ROOT
    cache_key = build_training_dataset_fingerprint(spec)
    paths = _paths(root, cache_key)
    required = (paths["metadata"], paths["sample_frame"], paths["daily_frame"], paths["teacher_summary"])
    if any(not path.exists() for path in required):
        return None
    try:
        metadata = json.loads(paths["metadata"].read_text(encoding="utf-8"))
        if str(metadata.get("cache_key", "") or "") != cache_key:
            return None
        if int(metadata.get("cache_version", 0) or 0) != TRAINING_DATASET_CACHE_VERSION:
            return None
        sample_frame = pd.read_pickle(paths["sample_frame"])
        daily_frame = pd.read_pickle(paths["daily_frame"])
        teacher_summary = json.loads(paths["teacher_summary"].read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(sample_frame, pd.DataFrame) or not isinstance(daily_frame, pd.DataFrame):
        return None
    if not isinstance(teacher_summary, dict):
        return None
    return TrainingDatasetCacheRecord(
        cache_key=cache_key,
        cache_dir=paths["cache_dir"],
        sample_frame=sample_frame,
        daily_frame=daily_frame,
        teacher_summary=teacher_summary,
        metadata=dict(metadata),
    )


def _legacy_to_reusable(record: TrainingDatasetCacheRecord) -> ReusableTrainingDatasetRecord:
    return ReusableTrainingDatasetRecord(
        store="legacy_pickle",
        cache_key=record.cache_key,
        cache_dir=record.cache_dir,
        sample_frame=record.sample_frame,
        daily_frame=record.daily_frame,
        teacher_summary=record.teacher_summary,
        metadata=record.metadata,
    )


def _lake_to_reusable(record: Any) -> ReusableTrainingDatasetRecord:
    return ReusableTrainingDatasetRecord(
        store="data_lake",
        cache_key=str(getattr(record, "dataset_id", "") or ""),
        cache_dir=Path(getattr(record, "root", Path("."))),
        sample_frame=record.sample_frame,
        daily_frame=record.daily_frame,
        teacher_summary=dict(record.teacher_summary),
        metadata=dict(getattr(record, "metadata", {}) or {}),
    )


def load_reusable_training_dataset(
    *,
    cache_root: str | Path | None = None,
    lake_root: str | Path | None = None,
    spec: Mapping[str, Any],
    prefer_data_lake: bool = True,
    zone: str = "strict_train",
) -> ReusableTrainingDatasetRecord | None:
    if prefer_data_lake:
        try:
            from quant_data_platform.lake import ResearchDataLake

            lake_record = ResearchDataLake(lake_root).find_training_dataset(spec=spec, zone=zone)
            if lake_record is not None:
                return _lake_to_reusable(lake_record)
        except Exception:
            pass
    legacy = load_training_dataset_cache(cache_root=cache_root, spec=spec)
    if legacy is not None:
        return _legacy_to_reusable(legacy)
    return None


def save_reusable_training_dataset(
    *,
    cache_root: str | Path | None = None,
    lake_root: str | Path | None = None,
    spec: Mapping[str, Any],
    sample_frame: pd.DataFrame,
    daily_frame: pd.DataFrame,
    teacher_summary: Mapping[str, Any],
    prefer_data_lake: bool = True,
    zone: str = "strict_train",
    label_completeness_summary: Mapping[str, Any] | None = None,
) -> ReusableTrainingDatasetRecord:
    legacy = save_training_dataset_cache(
        cache_root=cache_root,
        spec=spec,
        sample_frame=sample_frame,
        daily_frame=daily_frame,
        teacher_summary=teacher_summary,
    )
    if prefer_data_lake:
        try:
            from quant_data_platform.lake import ResearchDataLake

            lake_record = ResearchDataLake(lake_root).save_training_dataset(
                spec=spec,
                sample_frame=sample_frame,
                daily_frame=daily_frame,
                teacher_summary=teacher_summary,
                zone=zone,
                label_completeness_summary=label_completeness_summary,
            )
            return _lake_to_reusable(lake_record)
        except Exception:
            return _legacy_to_reusable(legacy)
    return _legacy_to_reusable(legacy)


def build_training_dataset_cache_spec(
    *,
    args: Any,
    prepared_summary: Mapping[str, Any],
    universe: list[str] | tuple[str, ...],
) -> dict[str, Any]:
    return {
        "dataset": "continuous_policy_training_matrices",
        "pool_name": str(getattr(args, "pool_name", "") or ""),
        "start_date": str(getattr(args, "start_date", "") or ""),
        "end_date": str(getattr(args, "end_date", "") or ""),
        "benchmark": str(getattr(args, "benchmark", "") or ""),
        "data_source": str(getattr(args, "data_source", "") or ""),
        "csv_folder": str(getattr(args, "csv_folder", "") or ""),
        "pool_rebalance_days": int(getattr(args, "pool_rebalance_days", 0) or 0),
        "pool_adv_window": int(getattr(args, "pool_adv_window", 0) or 0),
        "max_universe_size": int(getattr(args, "max_universe_size", 0) or 0),
        "random_seed": int(getattr(args, "random_seed", 0) or 0),
        "skip_multiplier": float(getattr(args, "skip_multiplier", 0.0) or 0.0),
        "label_preset": str(getattr(args, "label_preset", "") or ""),
        "execution_semantics": str(getattr(args, "execution_semantics", "") or ""),
        "budget_semantics": str(getattr(args, "budget_semantics", "") or ""),
        "budget_calibration": str(getattr(args, "budget_calibration", "") or ""),
        "budget_objective": str(getattr(args, "budget_objective", "") or ""),
        "alpha_prior_source": str(getattr(args, "alpha_prior_source", "") or ""),
        "alpha_prior_score_panel": str(getattr(args, "alpha_prior_score_panel", "") or ""),
        "alpha_prior_target_weight_panel": str(getattr(args, "alpha_prior_target_weight_panel", "") or ""),
        "transaction_cost_bps": float(getattr(args, "transaction_cost_bps", 0.0) or 0.0),
        "slippage_bps": float(getattr(args, "slippage_bps", 0.0) or 0.0),
        "sell_tax_bps": float(getattr(args, "sell_tax_bps", 0.0) or 0.0),
        "prepared_end_date": str(dict(prepared_summary or {}).get("end_date", "") or ""),
        "prepared_universe_size": int(dict(prepared_summary or {}).get("universe_size", len(universe)) or 0),
        "universe_digest": hashlib.sha256(
            json.dumps([str(item) for item in universe], ensure_ascii=False, sort_keys=True).encode("utf-8")
        ).hexdigest()[:24],
    }
