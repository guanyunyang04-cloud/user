from __future__ import annotations

"""Train and audit strict T+1 open-to-open and open-to-D2-close heads."""

import argparse
import json
import os
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from daily_research.path_policy import seq100_quality_liquidity_model as model
from daily_research.path_policy import (
    seq100_quality_liquidity_t1_targets as source_targets,
)

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
STUDY_ID = "seq100_quality_liquidity_t1_return_models"
SOURCE_MODEL_STUDY_ID = model.STUDY_ID
SOURCE_TARGET_STUDY_ID = source_targets.STUDY_ID
YEARS = tuple(range(2011, 2026))
ROLLING_YEARS = (2023, 2024, 2025)
FORBIDDEN_YEAR = 2026
MAXIMUM_OUTCOME_DATE = "2025-12-31"
FEATURE_COUNT = model.COMPACT_FEATURE_COUNT
TARGETS = ("open_to_open_1", "open_to_close_2")
DERIVED_COLUMNS = ("open_to_close_2", "exit_open_to_close_2")
TASK_COUNT = 6

FLAG_OUTCOME_WITHIN_CUTOFF = np.uint16(1 << 0)
FLAG_ENTRY_OPEN_OBSERVED = np.uint16(1 << 1)
FLAG_EXIT_OPEN_OBSERVED = np.uint16(1 << 2)
FLAG_EXIT_CLOSE_OBSERVED = np.uint16(1 << 3)
FLAG_SOURCE_OPEN_TO_OPEN_VALID = np.uint16(1 << 4)
FLAG_OPEN_TO_CLOSE_2_VALID = np.uint16(1 << 5)
FLAG_DECOMPOSITION_VALID = np.uint16(1 << 6)

FLAG_DEFINITIONS = {
    "outcome_within_cutoff": int(FLAG_OUTCOME_WITHIN_CUTOFF),
    "entry_open_observed": int(FLAG_ENTRY_OPEN_OBSERVED),
    "exit_open_observed": int(FLAG_EXIT_OPEN_OBSERVED),
    "exit_close_observed": int(FLAG_EXIT_CLOSE_OBSERVED),
    "source_open_to_open_valid": int(FLAG_SOURCE_OPEN_TO_OPEN_VALID),
    "open_to_close_2_valid": int(FLAG_OPEN_TO_CLOSE_2_VALID),
    "decomposition_valid": int(FLAG_DECOMPOSITION_VALID),
}

MANIFEST_SCHEMA = "seq100_quality_liquidity_t1_return_models_manifest/1"
LABEL_SCHEMA = "seq100_quality_liquidity_t1_return_model_labels/1"
AUDIT_SCHEMA = "seq100_quality_liquidity_t1_return_models_audit/1"

DEFAULT_STUDY_PATH = (
    WORKSPACE_ROOT
    / "daily_research/studies/seq100_quality_liquidity_t1_return_models.json"
)
DEFAULT_OUTPUT_ROOT = (
    WORKSPACE_ROOT
    / "daily_research/output/path_policy/studies"
    / STUDY_ID
)


class T1ReturnModelError(RuntimeError):
    pass


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _resolve(value: str | Path) -> Path:
    path = Path(value)
    return (
        (WORKSPACE_ROOT / path).resolve() if not path.is_absolute() else path.resolve()
    )


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise T1ReturnModelError(f"required_json_missing:{path}")
    return dict(json.loads(path.read_text(encoding="utf-8")))


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    model._write_json(path, payload)


def _write_parquet(frame: pd.DataFrame, path: Path) -> None:
    model._write_parquet(frame, path)


def _sha256(path: Path) -> str:
    return model._sha256(path)


def _file_record(path: Path, **extra: Any) -> dict[str, Any]:
    return model._file_record(path, **extra)


def _record_valid(record: Mapping[str, Any], *, verify_hash: bool = True) -> bool:
    try:
        path = Path(str(record["path"]))
        if not path.is_file() or int(path.stat().st_size) != int(record["size"]):
            return False
        return not verify_hash or _sha256(path) == str(record["sha256"])
    except (KeyError, OSError, TypeError, ValueError):
        return False


def _emit(event: str, **payload: Any) -> None:
    print(
        json.dumps({"event": event, "at": _now(), **payload}, ensure_ascii=False),
        flush=True,
    )


def load_study(path: Path = DEFAULT_STUDY_PATH) -> dict[str, Any]:
    study = _read_json(path)
    if study.get("study_id") != STUDY_ID:
        raise T1ReturnModelError("study_id_mismatch")
    source = dict(study.get("source", {}) or {})
    if (
        str(source.get("source_model_study_id")) != SOURCE_MODEL_STUDY_ID
        or str(source.get("source_target_study_id")) != SOURCE_TARGET_STUDY_ID
        or str(source.get("maximum_outcome_date")) != MAXIMUM_OUTCOME_DATE
        or int(source.get("forbidden_year", -1)) != FORBIDDEN_YEAR
        or bool(source.get("candidate_selection_uses_future_fill", True))
    ):
        raise T1ReturnModelError("source_contract_mismatch")
    period = dict(study.get("period", {}) or {})
    if (
        int(period.get("burn_in_year", -1)) != 2010
        or tuple(int(value) for value in period.get("research_years", ())) != YEARS
        or tuple(int(value) for value in period.get("rolling_prediction_years", ()))
        != ROLLING_YEARS
    ):
        raise T1ReturnModelError("period_contract_mismatch")
    feature = dict(study.get("feature_contract", {}) or {})
    if (
        str(feature.get("variant")) != model.COMPACT_VARIANT
        or int(feature.get("feature_count", -1)) != FEATURE_COUNT
        or bool(feature.get("feature_set_selected", True))
    ):
        raise T1ReturnModelError("feature_contract_mismatch")
    target_config = dict(study.get("targets", {}) or {})
    if tuple(target_config) != TARGETS:
        raise T1ReturnModelError("target_order_or_names_mismatch")
    for target in TARGETS:
        current = dict(target_config[target])
        if str(current.get("kind")) != "return" or int(
            current.get("horizon", -1)
        ) != 2:
            raise T1ReturnModelError(f"target_contract_mismatch:{target}")
    label = dict(study.get("label_contract", {}) or {})
    if (
        int(label.get("outcome_dependency_trading_days", -1)) != 2
        or bool(label.get("future_fill_used_for_label_support", True))
    ):
        raise T1ReturnModelError("label_dependency_contract_mismatch")
    execution = dict(study.get("execution", {}) or {})
    if (
        not bool(execution.get("training_performed", False))
        or int(execution.get("task_count", -1)) != TASK_COUNT
        or bool(execution.get("execution_replay_performed", True))
    ):
        raise T1ReturnModelError("execution_contract_mismatch")
    if int(dict(study.get("model", {}) or {}).get("fixed_mfe_rounds", -1)) != 512:
        raise T1ReturnModelError("fixed_round_contract_mismatch")
    return study


def _source_context(study: Mapping[str, Any]) -> dict[str, Any]:
    source_study = source_targets.load_study()
    target_context = source_targets._source_context(source_study)
    input_path = _resolve(str(study["source"]["model_input_manifest"]))
    if input_path != Path(target_context["input_path"]).resolve():
        raise T1ReturnModelError("model_input_source_mismatch")
    input_manifest = dict(target_context["input_manifest"])
    model._verify_model_input_files(input_manifest, full_hash=False)
    if (
        input_manifest.get("study_id") != SOURCE_MODEL_STUDY_ID
        or input_manifest.get("status") != "completed"
        or int(input_manifest.get("row_count", -1)) != len(target_context["row_index"])
    ):
        raise T1ReturnModelError("model_input_contract_mismatch")

    target_manifest_path = _resolve(str(study["source"]["t1_target_manifest"]))
    target_manifest = _read_json(target_manifest_path)
    if (
        target_manifest.get("study_id") != SOURCE_TARGET_STUDY_ID
        or target_manifest.get("status") != "audited"
        or int(target_manifest.get("row_count", -1)) != int(input_manifest["row_count"])
        or str(target_manifest.get("maximum_outcome_date")) != MAXIMUM_OUTCOME_DATE
        or int(target_manifest.get("forbidden_2026_read_count", -1)) != 0
        or str(target_manifest.get("model_inputs", {}).get("input_fingerprint"))
        != str(input_manifest.get("input_fingerprint"))
    ):
        raise T1ReturnModelError("t1_target_manifest_contract_mismatch")
    audit_record = dict(target_manifest.get("audit", {}) or {})
    if not _record_valid(audit_record) or _read_json(Path(audit_record["path"])).get(
        "status"
    ) != "ok":
        raise T1ReturnModelError("t1_target_audit_invalid")
    contract_record = dict(
        target_manifest.get("preparation_files", {}).get("label_contract", {}) or {}
    )
    if not _record_valid(contract_record):
        raise T1ReturnModelError("t1_target_contract_record_invalid")
    target_contract = _read_json(Path(contract_record["path"]))
    if (
        target_contract.get("schema") != source_targets.LABEL_SCHEMA
        or target_contract.get("status") != "completed"
        or target_contract.get("primary_label") != "open_to_open_1"
        or int(target_contract.get("row_count", -1)) != int(input_manifest["row_count"])
        or bool(target_contract.get("future_fill_used_for_label_support", True))
        or int(target_contract.get("forbidden_2026_read_count", -1)) != 0
        or not all(
            _record_valid(record)
            for record in dict(target_contract.get("files", {}) or {}).values()
        )
    ):
        raise T1ReturnModelError("t1_target_label_contract_invalid")
    baseline_root = _resolve(str(study["source"]["frozen_g1_root"]))
    return {
        **target_context,
        "input_path": input_path,
        "input_manifest": input_manifest,
        "target_manifest_path": target_manifest_path,
        "target_manifest": target_manifest,
        "target_contract_path": Path(contract_record["path"]).resolve(),
        "target_contract": target_contract,
        "baseline_root": baseline_root,
    }


def _experiment_fingerprint(
    *, study_path: Path, context: Mapping[str, Any]
) -> str:
    return model._stable_hash(
        {
            "schema": LABEL_SCHEMA,
            "study_sha256": _sha256(study_path),
            "model_input_manifest_sha256": _sha256(Path(context["input_path"])),
            "model_input_fingerprint": str(
                context["input_manifest"]["input_fingerprint"]
            ),
            "source_target_manifest_sha256": _sha256(
                Path(context["target_manifest_path"])
            ),
            "source_target_contract_sha256": _sha256(
                Path(context["target_contract_path"])
            ),
            "pack_manifest_sha256": _sha256(Path(context["pack_path"])),
            "targets": list(TARGETS),
            "derived_columns": list(DERIVED_COLUMNS),
            "rolling_years": list(ROLLING_YEARS),
            "feature_variant": model.COMPACT_VARIANT,
            "feature_count": FEATURE_COUNT,
            "rounds": 512,
            "objective": "regression_l2",
            "outcome_dependency_trading_days": 2,
            "maximum_outcome_date": MAXIMUM_OUTCOME_DATE,
            "forbidden_year": FORBIDDEN_YEAR,
            "future_fill_used_for_label_support": False,
            "flag_definitions": FLAG_DEFINITIONS,
        }
    )


def _derive_d2_batch(
    *,
    signal_date_idx: np.ndarray,
    symbol_idx: np.ndarray,
    cutoff_idx: int,
    daily_raw: np.ndarray,
    price_observed: np.ndarray,
    source_open_to_open: np.ndarray,
    source_valid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    dates = np.asarray(signal_date_idx, dtype=np.int64)
    symbols = np.asarray(symbol_idx, dtype=np.int64)
    source_values = np.asarray(source_open_to_open, dtype=np.float64)
    source_mask = np.asarray(source_valid, dtype=bool)
    if (
        dates.ndim != 1
        or symbols.shape != dates.shape
        or source_values.shape != dates.shape
        or source_mask.shape != dates.shape
    ):
        raise T1ReturnModelError("derive_batch_shape_mismatch")
    row_count = len(dates)
    returns = np.full((row_count, len(DERIVED_COLUMNS)), np.nan, dtype=np.float32)
    valid_store = np.zeros(row_count, dtype=np.uint8)
    flags = np.zeros(row_count, dtype=np.uint16)
    within = (dates >= 0) & (dates + 2 <= int(cutoff_idx))
    flags[within] |= FLAG_OUTCOME_WITHIN_CUTOFF
    positions = np.flatnonzero(within)
    if not len(positions):
        return returns, valid_store, flags

    signal_dates = dates[positions]
    current_symbols = symbols[positions]
    entry_dates = signal_dates + 1
    exit_dates = signal_dates + 2
    entry_open = np.asarray(
        daily_raw[entry_dates, current_symbols, 0], dtype=np.float64
    )
    exit_open = np.asarray(
        daily_raw[exit_dates, current_symbols, 0], dtype=np.float64
    )
    exit_close = np.asarray(
        daily_raw[exit_dates, current_symbols, 3], dtype=np.float64
    )
    entry_row_observed = np.asarray(
        price_observed[entry_dates, current_symbols], dtype=bool
    )
    exit_row_observed = np.asarray(
        price_observed[exit_dates, current_symbols], dtype=bool
    )
    entry_observed = (
        entry_row_observed & np.isfinite(entry_open) & (entry_open > 0.0)
    )
    exit_open_observed = (
        exit_row_observed & np.isfinite(exit_open) & (exit_open > 0.0)
    )
    exit_close_observed = (
        exit_row_observed & np.isfinite(exit_close) & (exit_close > 0.0)
    )
    flags[positions[entry_observed]] |= FLAG_ENTRY_OPEN_OBSERVED
    flags[positions[exit_open_observed]] |= FLAG_EXIT_OPEN_OBSERVED
    flags[positions[exit_close_observed]] |= FLAG_EXIT_CLOSE_OBSERVED

    local_source_valid = (
        source_mask[positions] & np.isfinite(source_values[positions])
    )
    flags[positions[local_source_valid]] |= FLAG_SOURCE_OPEN_TO_OPEN_VALID
    target_valid = entry_observed & exit_close_observed
    global_valid = positions[target_valid]
    flags[global_valid] |= FLAG_OPEN_TO_CLOSE_2_VALID
    valid_store[global_valid] = 1
    returns[global_valid, 0] = (
        exit_close[target_valid] / entry_open[target_valid] - 1.0
    ).astype(np.float32)

    decomposition_valid = target_valid & exit_open_observed & local_source_valid
    global_decomposition = positions[decomposition_valid]
    flags[global_decomposition] |= FLAG_DECOMPOSITION_VALID
    returns[global_decomposition, 1] = (
        exit_close[decomposition_valid] / exit_open[decomposition_valid] - 1.0
    ).astype(np.float32)
    return returns, valid_store, flags


def _flag_mask(flags: np.ndarray, flag: np.uint16) -> np.ndarray:
    return (np.asarray(flags, dtype=np.uint16) & flag) != 0


def _label_statistics(
    *,
    source_open_to_open: np.ndarray,
    derived: np.ndarray,
    valid: np.ndarray,
    flags: np.ndarray,
) -> dict[str, Any]:
    target_valid = np.asarray(valid, dtype=bool)
    values = np.asarray(derived[:, 0], dtype=np.float64)[target_valid]
    decomposition = _flag_mask(flags, FLAG_DECOMPOSITION_VALID)
    reconstructed = (
        1.0
        + np.asarray(source_open_to_open[decomposition], dtype=np.float64)
    ) * (
        1.0 + np.asarray(derived[decomposition, 1], dtype=np.float64)
    ) - 1.0
    identity_error = np.abs(
        reconstructed - np.asarray(derived[decomposition, 0], dtype=np.float64)
    )
    quantiles = np.quantile(values, [0.0, 0.001, 0.01, 0.5, 0.99, 0.999, 1.0])
    return {
        "valid_count": int(target_valid.sum()),
        "invalid_count": int((~target_valid).sum()),
        "decomposition_valid_count": int(decomposition.sum()),
        "mean": float(values.mean()),
        "std": float(values.std(ddof=0)),
        "quantiles": {
            key: float(value)
            for key, value in zip(
                ("minimum", "q001", "q01", "median", "q99", "q999", "maximum"),
                quantiles,
                strict=True,
            )
        },
        "maximum_decomposition_identity_error": float(
            identity_error.max(initial=0.0)
        ),
    }


def _coverage_frame(
    *,
    row_index: pd.DataFrame,
    date_values: np.ndarray,
    raw_values: Mapping[str, np.ndarray],
    raw_valid: Mapping[str, np.ndarray],
) -> pd.DataFrame:
    years = row_index["trade_date"].astype(str).str[:4].astype(int).to_numpy()
    date_idx = row_index["date_idx"].to_numpy(dtype=np.int32)
    records: list[dict[str, Any]] = []
    for target in TARGETS:
        _, normalized_valid = model._winsorized_zscore_by_date(
            values=np.asarray(raw_values[target], dtype=np.float32),
            date_idx=date_idx,
            valid=np.asarray(raw_valid[target], dtype=bool),
        )
        for year in YEARS:
            rows = years == year
            local_raw = np.asarray(raw_valid[target], dtype=bool)[rows]
            local_normalized = normalized_valid[rows]
            valid_dates = np.unique(date_idx[rows & normalized_valid])
            records.append(
                {
                    "target": target,
                    "signal_year": year,
                    "row_count": int(rows.sum()),
                    "raw_valid_count": int(local_raw.sum()),
                    "normalized_valid_count": int(local_normalized.sum()),
                    "raw_coverage": float(local_raw.mean()),
                    "normalized_coverage": float(local_normalized.mean()),
                    "normalized_date_count": len(valid_dates),
                    "first_normalized_trade_date": (
                        str(date_values[int(valid_dates[0])])
                        if len(valid_dates)
                        else None
                    ),
                    "last_normalized_trade_date": (
                        str(date_values[int(valid_dates[-1])])
                        if len(valid_dates)
                        else None
                    ),
                }
            )
        del normalized_valid
    return pd.DataFrame(records).sort_values(["target", "signal_year"])


def _prepared_manifest_valid(
    manifest: Mapping[str, Any], *, fingerprint: str
) -> bool:
    if (
        manifest.get("schema") != MANIFEST_SCHEMA
        or manifest.get("status")
        not in {"prepared", "training_in_progress", "training_completed", "evaluated", "audited"}
        or str(manifest.get("experiment_fingerprint")) != fingerprint
    ):
        return False
    records = dict(manifest.get("preparation_files", {}) or {})
    if not records or not all(_record_valid(record) for record in records.values()):
        return False
    contract = _read_json(Path(records["label_contract"]["path"]))
    return all(
        _record_valid(record)
        for record in dict(contract.get("files", {}) or {}).values()
    )


def prepare(
    *,
    study_path: Path = DEFAULT_STUDY_PATH,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    chunk_size: int = 250_000,
) -> dict[str, Any]:
    study = load_study(study_path)
    context = _source_context(study)
    fingerprint = _experiment_fingerprint(study_path=study_path, context=context)
    manifest_path = output_root / "manifest.json"
    if manifest_path.is_file():
        current = _read_json(manifest_path)
        if str(current.get("experiment_fingerprint")) != fingerprint:
            raise T1ReturnModelError("existing_output_fingerprint_mismatch")
        if _prepared_manifest_valid(current, fingerprint=fingerprint):
            return current
    if int(chunk_size) <= 0:
        raise T1ReturnModelError("chunk_size_must_be_positive")

    row_count = int(context["input_manifest"]["row_count"])
    source_files = dict(context["target_contract"]["files"])
    source_return_record = dict(source_files["returns"])
    source_valid_record = dict(source_files["valid"])
    source_returns = np.memmap(
        Path(source_return_record["path"]),
        dtype=np.dtype(source_return_record["dtype"]),
        mode="r",
        shape=tuple(int(value) for value in source_return_record["shape"]),
    )
    source_valid = np.memmap(
        Path(source_valid_record["path"]),
        dtype=np.dtype(source_valid_record["dtype"]),
        mode="r",
        shape=tuple(int(value) for value in source_valid_record["shape"]),
    )
    source_open_to_open = source_returns[:, 0]
    daily_raw = source_targets._open_array(
        context["daily_record"], dtype=np.float32
    )
    price_observed = source_targets._open_array(
        context["source_records"]["price_observed"], dtype=np.bool_
    )

    label_root = output_root / "labels"
    label_root.mkdir(parents=True, exist_ok=True)
    derived_path = label_root / "t1_d2_close_returns.float32.dat"
    valid_path = label_root / "t1_d2_close_valid.uint8.dat"
    flags_path = label_root / "t1_d2_close_flags.uint16.dat"
    partials = {
        derived_path: Path(str(derived_path) + ".partial"),
        valid_path: Path(str(valid_path) + ".partial"),
        flags_path: Path(str(flags_path) + ".partial"),
    }
    for partial in partials.values():
        partial.unlink(missing_ok=True)
    derived_store = np.memmap(
        partials[derived_path],
        dtype=np.float32,
        mode="w+",
        shape=(row_count, len(DERIVED_COLUMNS)),
    )
    valid_store = np.memmap(
        partials[valid_path], dtype=np.uint8, mode="w+", shape=(row_count,)
    )
    flags_store = np.memmap(
        partials[flags_path], dtype=np.uint16, mode="w+", shape=(row_count,)
    )
    derived_store[:] = np.nan
    valid_store[:] = 0
    flags_store[:] = 0
    _emit("t1_return_label_prepare_started", row_count=row_count)
    for start in range(0, row_count, int(chunk_size)):
        stop = min(start + int(chunk_size), row_count)
        derived, valid, flags = _derive_d2_batch(
            signal_date_idx=context["date_idx"][start:stop],
            symbol_idx=context["symbol_idx"][start:stop],
            cutoff_idx=int(context["cutoff_idx"]),
            daily_raw=daily_raw,
            price_observed=price_observed,
            source_open_to_open=source_open_to_open[start:stop],
            source_valid=source_valid[start:stop],
        )
        derived_store[start:stop] = derived
        valid_store[start:stop] = valid
        flags_store[start:stop] = flags
    derived_store.flush()
    valid_store.flush()
    flags_store.flush()
    del derived_store, valid_store, flags_store
    for final, partial in partials.items():
        os.replace(partial, final)

    derived_store = np.memmap(
        derived_path,
        dtype=np.float32,
        mode="r",
        shape=(row_count, len(DERIVED_COLUMNS)),
    )
    valid_store = np.memmap(valid_path, dtype=np.uint8, mode="r", shape=(row_count,))
    flags_store = np.memmap(flags_path, dtype=np.uint16, mode="r", shape=(row_count,))
    raw_values = {
        "open_to_open_1": source_open_to_open,
        "open_to_close_2": derived_store[:, 0],
    }
    raw_valid = {
        "open_to_open_1": np.asarray(source_valid, dtype=bool)
        & np.isfinite(np.asarray(source_open_to_open, dtype=np.float32)),
        "open_to_close_2": np.asarray(valid_store, dtype=bool),
    }
    coverage = _coverage_frame(
        row_index=context["row_index"],
        date_values=context["date_values"],
        raw_values=raw_values,
        raw_valid=raw_valid,
    )
    statistics = _label_statistics(
        source_open_to_open=source_open_to_open,
        derived=derived_store,
        valid=valid_store,
        flags=flags_store,
    )
    within = _flag_mask(flags_store, FLAG_OUTCOME_WITHIN_CUTOFF)
    maximum_signal_idx = int(context["date_idx"][within].max())
    coverage_path = output_root / "label_coverage.parquet"
    statistics_path = output_root / "label_statistics.json"
    contract_path = output_root / "label_contract.json"
    _write_parquet(coverage, coverage_path)
    _write_json(statistics_path, statistics)
    contract = {
        "schema": LABEL_SCHEMA,
        "status": "completed",
        "created_at": _now(),
        "study_id": STUDY_ID,
        "experiment_fingerprint": fingerprint,
        "row_count": row_count,
        "targets": list(TARGETS),
        "derived_columns": list(DERIVED_COLUMNS),
        "source_open_to_open_column": 0,
        "formulas": {
            "open_to_open_1": "adjusted_open[t+2]/adjusted_open[t+1]-1",
            "open_to_close_2": "adjusted_close[t+2]/adjusted_open[t+1]-1",
            "exit_open_to_close_2": "adjusted_close[t+2]/adjusted_open[t+2]-1",
            "decomposition": (
                "(1+open_to_open_1)*(1+exit_open_to_close_2)-1"
            ),
        },
        "price_semantics": "back_adjusted_total_return",
        "outcome_dependency_trading_days": 2,
        "training_purge_days": 2,
        "future_fill_used_for_label_support": False,
        "candidate_selection_uses_future_fill": False,
        "read_dependencies": [
            "signal_date_idx",
            "adjusted_entry_open_t_plus_1",
            "adjusted_exit_open_t_plus_2_for_decomposition_only",
            "adjusted_exit_close_t_plus_2",
            "price_observed",
            "audited_open_to_open_1",
            "maximum_outcome_date",
        ],
        "excluded_label_dependencies": [
            "entry_filled",
            "entry_buyable",
            "exit_open_sellable",
            "future_execution_state",
        ],
        "training_transform": {
            "group": "signal_trade_date_cross_section",
            "winsor_quantiles": [0.01, 0.99],
            "quantile_method": "nearest",
            "zscore_ddof": 0,
            "minimum_cross_section": model.RETURN_MIN_CROSS_SECTION,
        },
        "flag_definitions": FLAG_DEFINITIONS,
        "maximum_signal_trade_date_read": str(
            context["date_values"][maximum_signal_idx]
        ),
        "maximum_outcome_date_read": MAXIMUM_OUTCOME_DATE,
        "forbidden_2026_read_count": 0,
        "statistics": statistics,
        "sources": {
            "model_input_manifest": _file_record(Path(context["input_path"])),
            "source_target_manifest": _file_record(
                Path(context["target_manifest_path"])
            ),
            "source_target_contract": _file_record(
                Path(context["target_contract_path"])
            ),
            "source_open_to_open_values": dict(source_return_record),
            "source_open_to_open_valid": dict(source_valid_record),
            "pack_manifest": _file_record(Path(context["pack_path"])),
            "daily_raw": dict(context["daily_record"]),
            "price_observed": dict(context["source_records"]["price_observed"]),
        },
        "files": {
            "derived_returns": _file_record(
                derived_path,
                shape=[row_count, len(DERIVED_COLUMNS)],
                dtype="float32",
                columns=list(DERIVED_COLUMNS),
            ),
            "derived_valid": _file_record(
                valid_path, shape=[row_count], dtype="uint8"
            ),
            "derived_flags": _file_record(
                flags_path,
                shape=[row_count],
                dtype="uint16",
                definitions=FLAG_DEFINITIONS,
            ),
        },
        "training_performed": False,
        "execution_replay_performed": False,
    }
    _write_json(contract_path, contract)
    tasks = _task_plan(study)
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "status": "prepared",
        "created_at": _now(),
        "study_id": STUDY_ID,
        "experiment_fingerprint": fingerprint,
        "study_config": _file_record(study_path),
        "model_inputs": _file_record(
            Path(context["input_path"]),
            input_fingerprint=str(context["input_manifest"]["input_fingerprint"]),
        ),
        "source_target_manifest": _file_record(
            Path(context["target_manifest_path"])
        ),
        "row_count": row_count,
        "formal_years": list(YEARS),
        "rolling_years": list(ROLLING_YEARS),
        "feature_variant": model.COMPACT_VARIANT,
        "feature_count": FEATURE_COUNT,
        "task_count": len(tasks),
        "tasks": tasks,
        "maximum_outcome_date": MAXIMUM_OUTCOME_DATE,
        "forbidden_year": FORBIDDEN_YEAR,
        "forbidden_2026_read_count": 0,
        "preparation_files": {
            "label_contract": _file_record(contract_path),
            "label_coverage": _file_record(coverage_path, row_count=len(coverage)),
            "label_statistics": _file_record(statistics_path),
        },
        "outputs": {},
        "training_performed": False,
        "execution_replay_performed": False,
        "feature_set_selected": False,
        "production_policy_selected": False,
    }
    _write_json(manifest_path, manifest)
    _emit(
        "t1_return_label_prepare_completed",
        open_to_open_valid_count=int(raw_valid["open_to_open_1"].sum()),
        open_to_close_2_valid_count=int(raw_valid["open_to_close_2"].sum()),
    )
    return manifest


class T1ReturnModelInputs(model.ModelInputs):
    """Expose the two row-aligned T+1 targets to the shared trainer."""

    def __init__(
        self, manifest: Mapping[str, Any], label_contract: Mapping[str, Any]
    ) -> None:
        super().__init__(manifest)
        if int(label_contract.get("row_count", -1)) != self.row_count:
            raise T1ReturnModelError("label_row_count_mismatch")
        source_values_record = dict(
            label_contract["sources"]["source_open_to_open_values"]
        )
        source_valid_record = dict(
            label_contract["sources"]["source_open_to_open_valid"]
        )
        derived_record = dict(label_contract["files"]["derived_returns"])
        derived_valid_record = dict(label_contract["files"]["derived_valid"])
        self._source_returns = np.memmap(
            Path(source_values_record["path"]),
            dtype=np.dtype(source_values_record["dtype"]),
            mode="r",
            shape=tuple(int(value) for value in source_values_record["shape"]),
        )
        self._source_valid = np.memmap(
            Path(source_valid_record["path"]),
            dtype=np.dtype(source_valid_record["dtype"]),
            mode="r",
            shape=tuple(int(value) for value in source_valid_record["shape"]),
        )
        self._derived_returns = np.memmap(
            Path(derived_record["path"]),
            dtype=np.dtype(derived_record["dtype"]),
            mode="r",
            shape=tuple(int(value) for value in derived_record["shape"]),
        )
        self._derived_valid = np.memmap(
            Path(derived_valid_record["path"]),
            dtype=np.dtype(derived_valid_record["dtype"]),
            mode="r",
            shape=tuple(int(value) for value in derived_valid_record["shape"]),
        )
        if (
            self._source_returns.shape[0] != self.row_count
            or self._source_valid.shape != (self.row_count,)
            or self._derived_returns.shape != (self.row_count, len(DERIVED_COLUMNS))
            or self._derived_valid.shape != (self.row_count,)
        ):
            raise T1ReturnModelError("label_array_shape_mismatch")
        self._raw_target_cache: dict[str, np.ndarray] = {
            "open_to_open_1": self._source_returns[:, 0],
            "open_to_close_2": self._derived_returns[:, 0],
        }
        self._raw_target_valid_cache: dict[str, np.ndarray] = {
            "open_to_open_1": np.asarray(self._source_valid, dtype=bool)
            & np.isfinite(np.asarray(self._source_returns[:, 0], dtype=np.float32)),
            "open_to_close_2": np.asarray(self._derived_valid, dtype=bool)
            & np.isfinite(np.asarray(self._derived_returns[:, 0], dtype=np.float32)),
        }
        self._normalized_target_cache: dict[str, np.ndarray] = {}
        self._normalized_target_valid_cache: dict[str, np.ndarray] = {}

    def raw_task_values(self, target: str) -> np.ndarray:
        if target in self._raw_target_cache:
            return self._raw_target_cache[target]
        return super().raw_task_values(target)

    def raw_task_valid_mask(self, target: str) -> np.ndarray:
        if target not in self._raw_target_valid_cache:
            raise T1ReturnModelError(f"unsupported_t1_return_target:{target}")
        return self._raw_target_valid_cache[target]

    def task_values(self, target: str) -> np.ndarray:
        if target not in TARGETS:
            return super().task_values(target)
        if target not in self._normalized_target_cache:
            values, valid = model._winsorized_zscore_by_date(
                values=self.raw_task_values(target),
                date_idx=self.date_idx,
                valid=self.raw_task_valid_mask(target),
            )
            self._normalized_target_cache[target] = values
            self._normalized_target_valid_cache[target] = valid
        return self._normalized_target_cache[target]

    def valid_mask(self, target: str) -> np.ndarray:
        if target not in TARGETS:
            return super().valid_mask(target)
        if target not in self._normalized_target_valid_cache:
            self.task_values(target)
        return self._normalized_target_valid_cache[target]


def _task_plan(study: Mapping[str, Any] | None = None) -> list[dict[str, Any]]:
    config = load_study() if study is None else dict(study)
    target_config = dict(config["targets"])
    tasks = [
        {
            "study_id": STUDY_ID,
            "task_id": (
                f"t1_return_zscore__{target}__{model.COMPACT_VARIANT}__{year}"
            ),
            "stage": model.RETURN_TASK_STAGE,
            "target": target,
            "source_label": str(target_config[target]["source"]),
            "kind": "return",
            "horizon": 2,
            "year": int(year),
            "variant": model.COMPACT_VARIANT,
            "gated_family": None,
            "label_transform": "signal_date_winsor_01_99_zscore",
        }
        for target in TARGETS
        for year in ROLLING_YEARS
    ]
    if len(tasks) != TASK_COUNT or len({str(task["task_id"]) for task in tasks}) != TASK_COUNT:
        raise T1ReturnModelError(f"task_plan_contract_mismatch:{len(tasks)}")
    return tasks


def _update_ledger(
    *,
    output_root: Path,
    tasks: Sequence[Mapping[str, Any]],
    results: Mapping[str, Mapping[str, Any]],
) -> None:
    entries = []
    for task in tasks:
        task_id = str(task["task_id"])
        result = results.get(task_id)
        entries.append(
            {
                **dict(task),
                "status": "completed" if result is not None else "pending",
                "result_path": (
                    str(model._task_result_path(output_root, task_id).resolve())
                    if result is not None
                    else None
                ),
                "best_iteration": result.get("best_iteration") if result else None,
                "updated_at": _now(),
            }
        )
    _write_json(
        output_root / "task_ledger.json",
        {
            "schema": "seq100_quality_liquidity_t1_return_model_ledger/1",
            "study_id": STUDY_ID,
            "task_count": len(tasks),
            "completed_count": sum(item["status"] == "completed" for item in entries),
            "tasks": entries,
        },
    )


def run(
    *,
    study_path: Path = DEFAULT_STUDY_PATH,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    max_tasks: int | None = None,
) -> dict[str, Any]:
    study = load_study(study_path)
    prepare(study_path=study_path, output_root=output_root)
    manifest_path = output_root / "manifest.json"
    manifest = _read_json(manifest_path)
    input_manifest = _read_json(Path(manifest["model_inputs"]["path"]))
    label_contract = _read_json(
        Path(manifest["preparation_files"]["label_contract"]["path"])
    )
    inputs = T1ReturnModelInputs(input_manifest, label_contract)
    tasks = _task_plan(study)
    results, diagnostic = model._partition_results(
        model._completed_results(output_root), tasks
    )
    if diagnostic:
        raise T1ReturnModelError(f"unexpected_training_tasks:{sorted(diagnostic)}")
    _update_ledger(output_root=output_root, tasks=tasks, results=results)
    completed_this_run = 0
    fingerprint = str(manifest["experiment_fingerprint"])
    config_sha256 = _sha256(study_path)
    for task in tasks:
        task_id = str(task["task_id"])
        if task_id in results:
            feature_names, _ = model._effective_features(
                inputs=inputs, task=task, output_root=output_root
            )
            task_fingerprint = model._task_fingerprint(
                task=task,
                feature_names=feature_names,
                model_input_fingerprint=str(input_manifest["input_fingerprint"]),
                config_sha256=config_sha256,
                experiment_fingerprint=fingerprint,
            )
            if model._task_complete(
                model._task_result_path(output_root, task_id),
                fingerprint=task_fingerprint,
            ):
                continue
            results.pop(task_id)
        if max_tasks is not None and completed_this_run >= int(max_tasks):
            break
        _emit(
            "t1_return_training_task_started",
            task_id=task_id,
            completed_count=len(results),
            task_count=len(tasks),
        )
        result = model._run_training_task(
            task=task,
            inputs=inputs,
            config=study,
            output_root=output_root,
            config_sha256=config_sha256,
            experiment_fingerprint=fingerprint,
        )
        results[task_id] = result
        completed_this_run += 1
        _update_ledger(output_root=output_root, tasks=tasks, results=results)
        _emit(
            "t1_return_training_task_completed",
            task_id=task_id,
            training_seconds=float(result["training_seconds"]),
            completed_count=len(results),
            task_count=len(tasks),
        )
    all_complete = len(results) == len(tasks)
    manifest["status"] = "training_completed" if all_complete else "training_in_progress"
    manifest["updated_at"] = _now()
    manifest["training_performed"] = all_complete
    manifest["outputs"] = {} if not all_complete else dict(manifest.get("outputs", {}))
    _write_json(manifest_path, manifest)
    return {
        "status": manifest["status"],
        "study_id": STUDY_ID,
        "task_count": len(tasks),
        "completed_count": len(results),
        "completed_this_run": completed_this_run,
        "training_performed": all_complete,
    }


def _task_prediction(
    *, result: Mapping[str, Any], inputs: T1ReturnModelInputs, year: int
) -> tuple[np.ndarray, np.ndarray]:
    prediction = np.load(
        Path(result["files"]["prediction"]["path"]),
        mmap_mode="r",
        allow_pickle=False,
    )
    candidate_ids = np.load(
        Path(result["files"]["candidate_id"]["path"]),
        mmap_mode="r",
        allow_pickle=False,
    )
    year_rows = inputs.rows_for_year(year)
    if not np.array_equal(candidate_ids, inputs.candidate_ids[year_rows]) or len(
        prediction
    ) != len(year_rows):
        raise T1ReturnModelError(f"prediction_candidate_alignment_failed:{year}")
    return year_rows, prediction


def _frozen_g1_result(
    *, baseline_root: Path, year: int
) -> dict[str, Any]:
    task_id = f"{model.RETURN_TASK_STAGE}__g_01__{model.COMPACT_VARIANT}__{year}"
    result = _read_json(model._task_result_path(baseline_root, task_id))
    if (
        result.get("status") != "completed"
        or result.get("study_id") != SOURCE_MODEL_STUDY_ID
        or result.get("target") != "return_1"
        or int(result.get("evaluation_year", -1)) != int(year)
        or int(result.get("feature_count", -1)) != FEATURE_COUNT
    ):
        raise T1ReturnModelError(f"frozen_g1_task_contract_mismatch:{year}")
    if not all(_record_valid(record) for record in result["files"].values()):
        raise T1ReturnModelError(f"frozen_g1_task_files_invalid:{year}")
    return result


def _metric_row(
    *,
    target: str,
    year: int,
    model_source: str,
    task_id: str,
    evaluation_row_count: int,
    metrics: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "target": target,
        "year": int(year),
        "model_source": model_source,
        "task_id": task_id,
        "evaluation_row_count": int(evaluation_row_count),
        **{str(key): value for key, value in metrics.items()},
    }


def evaluate(
    *,
    study_path: Path = DEFAULT_STUDY_PATH,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
) -> dict[str, Any]:
    study = load_study(study_path)
    context = _source_context(study)
    manifest_path = output_root / "manifest.json"
    manifest = _read_json(manifest_path)
    if manifest.get("status") not in {"training_completed", "evaluated", "audited"}:
        raise T1ReturnModelError("training_not_completed")
    input_manifest = _read_json(Path(manifest["model_inputs"]["path"]))
    label_contract = _read_json(
        Path(manifest["preparation_files"]["label_contract"]["path"])
    )
    inputs = T1ReturnModelInputs(input_manifest, label_contract)
    tasks = _task_plan(study)
    results, diagnostic = model._partition_results(
        model._completed_results(output_root), tasks
    )
    if len(results) != TASK_COUNT or diagnostic:
        raise T1ReturnModelError(
            f"evaluation_task_count_mismatch:{len(results)}:{len(diagnostic)}"
        )

    daily_parts: list[pd.DataFrame] = []
    annual_rows: list[dict[str, Any]] = []
    paired_rows: list[dict[str, Any]] = []
    comparison_metrics = [str(value) for value in study["comparison"]["metrics"]]
    for task in tasks:
        task_id = str(task["task_id"])
        year = int(task["year"])
        target = str(task["target"])
        direct_result = results[task_id]
        year_rows, direct_prediction = _task_prediction(
            result=direct_result, inputs=inputs, year=year
        )
        baseline_result = _frozen_g1_result(
            baseline_root=Path(context["baseline_root"]), year=year
        )
        baseline_rows, baseline_prediction = _task_prediction(
            result=baseline_result, inputs=inputs, year=year
        )
        if not np.array_equal(year_rows, baseline_rows):
            raise T1ReturnModelError(f"baseline_year_rows_mismatch:{target}:{year}")
        fold = inputs.fold(year=year, horizon=2, target=target)
        evaluation_rows = fold["evaluation_rows"]
        positions = np.searchsorted(year_rows, evaluation_rows)
        if not np.array_equal(year_rows[positions], evaluation_rows):
            raise T1ReturnModelError(f"evaluation_row_alignment_failed:{target}:{year}")
        metrics_by_source: dict[str, dict[str, Any]] = {}
        for model_source, result, prediction in (
            ("direct_target", direct_result, direct_prediction),
            ("frozen_g1", baseline_result, baseline_prediction),
        ):
            daily, metrics = model._return_daily_metrics(
                date_idx=inputs.date_idx[evaluation_rows],
                actual_raw=inputs.raw_task_values(target)[evaluation_rows],
                actual_zscore=inputs.task_values(target)[evaluation_rows],
                prediction=np.asarray(prediction[positions], dtype=np.float32),
            )
            daily.insert(
                1,
                "trade_date",
                [str(inputs.date_values[int(value)]) for value in daily["date_idx"]],
            )
            daily.insert(0, "target", target)
            daily.insert(1, "year", year)
            daily.insert(2, "model_source", model_source)
            daily.insert(3, "task_id", str(result["task_id"]))
            daily_parts.append(daily)
            metrics_by_source[model_source] = metrics
            annual_rows.append(
                _metric_row(
                    target=target,
                    year=year,
                    model_source=model_source,
                    task_id=str(result["task_id"]),
                    evaluation_row_count=len(evaluation_rows),
                    metrics=metrics,
                )
            )
        direct_metrics = metrics_by_source["direct_target"]
        baseline_metrics = metrics_by_source["frozen_g1"]
        paired = {
            "target": target,
            "year": year,
            "direct_task_id": task_id,
            "baseline_task_id": str(baseline_result["task_id"]),
            "evaluation_row_count": len(evaluation_rows),
        }
        for metric_name in comparison_metrics:
            direct_value = float(direct_metrics[metric_name])
            baseline_value = float(baseline_metrics[metric_name])
            paired[f"direct_{metric_name}"] = direct_value
            paired[f"baseline_{metric_name}"] = baseline_value
            paired[f"delta_{metric_name}"] = direct_value - baseline_value
        paired_rows.append(paired)

    daily = pd.concat(daily_parts, ignore_index=True).sort_values(
        ["target", "year", "model_source", "date_idx"]
    )
    annual = pd.DataFrame(annual_rows).sort_values(
        ["target", "year", "model_source"]
    )
    paired = pd.DataFrame(paired_rows).sort_values(["target", "year"])
    importance = pd.concat(
        [
            pd.read_parquet(Path(result["files"]["family_importance"]["path"]))
            for result in results.values()
        ],
        ignore_index=True,
    )
    decision_targets: dict[str, Any] = {}
    for target, part in paired.groupby("target", sort=True):
        decision_targets[str(target)] = {
            "year_count": len(part),
            "rank_ic_improved_years": int(part["delta_rank_ic"].gt(0.0).sum()),
            "top5_excess_improved_years": int(
                part["delta_top5_excess_mean"].gt(0.0).sum()
            ),
            "top5_capture_improved_years": int(
                part["delta_top5_capture"].gt(0.0).sum()
            ),
            "zscore_mse_improved_years": int(
                part["delta_zscore_mse"].lt(0.0).sum()
            ),
            "direct_positive_rank_ic_years": int(
                part["direct_rank_ic"].gt(0.0).sum()
            ),
            "direct_positive_top5_excess_years": int(
                part["direct_top5_excess_mean"].gt(0.0).sum()
            ),
        }
    decision = {
        "schema": "seq100_quality_liquidity_t1_return_model_comparison/1",
        "status": "completed",
        "created_at": _now(),
        "study_id": STUDY_ID,
        "targets": decision_targets,
        "baseline": "frozen_g_1",
        "identical_target_support": True,
        "execution_replay_performed": False,
        "feature_set_selected": False,
        "production_policy_selected": False,
        "interpretation": "predictive comparison only; no execution policy decision",
    }
    paths = {
        "daily_metrics": output_root / "daily_metrics.parquet",
        "annual_metrics": output_root / "annual_metrics.parquet",
        "paired_comparison": output_root / "paired_comparison.parquet",
        "family_importance": output_root / "family_importance.parquet",
        "comparison_decision": output_root / "comparison_decision.json",
    }
    for frame, path in (
        (daily, paths["daily_metrics"]),
        (annual, paths["annual_metrics"]),
        (paired, paths["paired_comparison"]),
        (importance, paths["family_importance"]),
    ):
        _write_parquet(frame, path)
    _write_json(paths["comparison_decision"], decision)
    manifest["status"] = "evaluated"
    manifest["updated_at"] = _now()
    manifest["training_performed"] = True
    manifest["execution_replay_performed"] = False
    manifest["feature_set_selected"] = False
    manifest["production_policy_selected"] = False
    manifest["outputs"] = {
        "daily_metrics": _file_record(paths["daily_metrics"], row_count=len(daily)),
        "annual_metrics": _file_record(
            paths["annual_metrics"], row_count=len(annual)
        ),
        "paired_comparison": _file_record(
            paths["paired_comparison"], row_count=len(paired)
        ),
        "family_importance": _file_record(
            paths["family_importance"], row_count=len(importance)
        ),
        "comparison_decision": _file_record(paths["comparison_decision"]),
    }
    _write_json(manifest_path, manifest)
    return {
        "status": "evaluated",
        "task_count": len(results),
        "daily_metric_rows": len(daily),
        "annual_metric_rows": len(annual),
        "paired_comparison_rows": len(paired),
        "decision": decision,
    }


def _audit_label_arrays(
    *, context: Mapping[str, Any], contract: Mapping[str, Any], chunk_size: int = 250_000
) -> dict[str, Any]:
    row_count = int(contract["row_count"])
    source_record = dict(contract["sources"]["source_open_to_open_values"])
    source_valid_record = dict(contract["sources"]["source_open_to_open_valid"])
    derived_record = dict(contract["files"]["derived_returns"])
    valid_record = dict(contract["files"]["derived_valid"])
    flags_record = dict(contract["files"]["derived_flags"])
    source_returns = np.memmap(
        Path(source_record["path"]),
        dtype=np.dtype(source_record["dtype"]),
        mode="r",
        shape=tuple(int(value) for value in source_record["shape"]),
    )
    source_valid = np.memmap(
        Path(source_valid_record["path"]),
        dtype=np.dtype(source_valid_record["dtype"]),
        mode="r",
        shape=tuple(int(value) for value in source_valid_record["shape"]),
    )
    derived = np.memmap(
        Path(derived_record["path"]),
        dtype=np.dtype(derived_record["dtype"]),
        mode="r",
        shape=tuple(int(value) for value in derived_record["shape"]),
    )
    valid = np.memmap(
        Path(valid_record["path"]),
        dtype=np.dtype(valid_record["dtype"]),
        mode="r",
        shape=tuple(int(value) for value in valid_record["shape"]),
    )
    flags = np.memmap(
        Path(flags_record["path"]),
        dtype=np.dtype(flags_record["dtype"]),
        mode="r",
        shape=tuple(int(value) for value in flags_record["shape"]),
    )
    daily_raw = source_targets._open_array(context["daily_record"], dtype=np.float32)
    price_observed = source_targets._open_array(
        context["source_records"]["price_observed"], dtype=np.bool_
    )
    arrays_exact = True
    maximum_value_error = 0.0
    for start in range(0, row_count, int(chunk_size)):
        stop = min(start + int(chunk_size), row_count)
        expected_returns, expected_valid, expected_flags = _derive_d2_batch(
            signal_date_idx=context["date_idx"][start:stop],
            symbol_idx=context["symbol_idx"][start:stop],
            cutoff_idx=int(context["cutoff_idx"]),
            daily_raw=daily_raw,
            price_observed=price_observed,
            source_open_to_open=source_returns[start:stop, 0],
            source_valid=source_valid[start:stop],
        )
        actual_returns = np.asarray(derived[start:stop], dtype=np.float32)
        finite = np.isfinite(expected_returns) & np.isfinite(actual_returns)
        maximum_value_error = max(
            maximum_value_error,
            float(
                np.abs(expected_returns[finite] - actual_returns[finite]).max(
                    initial=0.0
                )
            ),
        )
        arrays_exact &= np.array_equal(
            np.isnan(expected_returns), np.isnan(actual_returns)
        )
        arrays_exact &= maximum_value_error <= 1.0e-7
        arrays_exact &= np.array_equal(expected_valid, valid[start:stop])
        arrays_exact &= np.array_equal(expected_flags, flags[start:stop])
    decomposition = _flag_mask(flags, FLAG_DECOMPOSITION_VALID)
    reconstructed = (
        1.0 + np.asarray(source_returns[decomposition, 0], dtype=np.float64)
    ) * (1.0 + np.asarray(derived[decomposition, 1], dtype=np.float64)) - 1.0
    identity_error = np.abs(
        reconstructed - np.asarray(derived[decomposition, 0], dtype=np.float64)
    )
    return {
        "arrays_exact": bool(arrays_exact),
        "maximum_value_error": maximum_value_error,
        "maximum_identity_error": float(identity_error.max(initial=0.0)),
        "valid_count": int(np.asarray(valid, dtype=bool).sum()),
        "decomposition_valid_count": int(decomposition.sum()),
        "stored_target_finite_exact": bool(
            np.array_equal(
                np.isfinite(np.asarray(derived[:, 0], dtype=np.float32)),
                np.asarray(valid, dtype=bool),
            )
        ),
    }


def audit(
    *,
    study_path: Path = DEFAULT_STUDY_PATH,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
) -> dict[str, Any]:
    study = load_study(study_path)
    context = _source_context(study)
    expected_fingerprint = _experiment_fingerprint(
        study_path=study_path, context=context
    )
    manifest_path = output_root / "manifest.json"
    manifest = _read_json(manifest_path)
    preparation_files = dict(manifest.get("preparation_files", {}) or {})
    contract = _read_json(Path(preparation_files["label_contract"]["path"]))
    label_audit = _audit_label_arrays(context=context, contract=contract)
    input_manifest = _read_json(Path(manifest["model_inputs"]["path"]))
    inputs = T1ReturnModelInputs(input_manifest, contract)
    tasks = _task_plan(study)
    results, diagnostic = model._partition_results(
        model._completed_results(output_root), tasks
    )
    checks: dict[str, bool] = {
        "manifest_status": manifest.get("status") in {"evaluated", "audited"},
        "experiment_fingerprint": str(manifest.get("experiment_fingerprint"))
        == expected_fingerprint,
        "label_contract_schema": contract.get("schema") == LABEL_SCHEMA
        and contract.get("status") == "completed",
        "preparation_files_valid": all(
            _record_valid(record) for record in preparation_files.values()
        ),
        "label_files_valid": all(
            _record_valid(record)
            for record in dict(contract.get("files", {}) or {}).values()
        ),
        "derived_labels_exact": bool(label_audit["arrays_exact"]),
        "stored_target_finite_exact": bool(
            label_audit["stored_target_finite_exact"]
        ),
        "decomposition_identity": float(label_audit["maximum_identity_error"])
        <= 2.0e-6,
        "future_fill_not_label_gate": not bool(
            contract.get("future_fill_used_for_label_support", True)
        )
        and not bool(contract.get("candidate_selection_uses_future_fill", True))
        and set(contract.get("excluded_label_dependencies", ()))
        == {
            "entry_filled",
            "entry_buyable",
            "exit_open_sellable",
            "future_execution_state",
        },
        "feature_count_exact": len(inputs.feature_groups[model.COMPACT_VARIANT])
        == FEATURE_COUNT,
        "target_raw_support_identical": np.array_equal(
            inputs.raw_task_valid_mask(TARGETS[0]),
            inputs.raw_task_valid_mask(TARGETS[1]),
        ),
        "target_normalized_support_identical": np.array_equal(
            inputs.valid_mask(TARGETS[0]), inputs.valid_mask(TARGETS[1])
        ),
        "task_count_exact": len(results) == TASK_COUNT and not diagnostic,
        "formal_rows_exclude_2010": not bool((inputs.years == 2010).any()),
        "forbidden_2026_rows": not bool(
            np.char.startswith(inputs.trade_date, "2026-").any()
        ),
        "forbidden_2026_reads": int(
            contract.get("forbidden_2026_read_count", -1)
        )
        == 0
        and str(contract.get("maximum_outcome_date_read")) == MAXIMUM_OUTCOME_DATE,
        "task_files_valid": True,
        "candidate_alignment": True,
        "purge_valid": True,
        "fixed_rounds_exact": True,
        "result_study_id_exact": True,
        "baseline_candidate_alignment": True,
        "outputs_hashed": bool(manifest.get("outputs")),
        "evaluation_rows_exact": False,
        "evaluation_metrics_reconcile": True,
        "comparison_support_identical": True,
        "no_execution_replay": not bool(
            manifest.get("execution_replay_performed", True)
        )
        and not bool(contract.get("execution_replay_performed", True)),
        "no_html_report": not any(output_root.rglob("*.html")),
    }
    config_sha256 = _sha256(study_path)
    baseline_checked: set[int] = set()
    for task in tasks:
        task_id = str(task["task_id"])
        result = results.get(task_id)
        if result is None:
            checks["task_files_valid"] = False
            continue
        try:
            feature_names, _ = model._effective_features(
                inputs=inputs, task=task, output_root=output_root
            )
            task_fingerprint = model._task_fingerprint(
                task=task,
                feature_names=feature_names,
                model_input_fingerprint=str(input_manifest["input_fingerprint"]),
                config_sha256=config_sha256,
                experiment_fingerprint=expected_fingerprint,
            )
            checks["task_files_valid"] &= model._task_complete(
                model._task_result_path(output_root, task_id),
                fingerprint=task_fingerprint,
            )
            year = int(task["year"])
            year_rows, _ = _task_prediction(result=result, inputs=inputs, year=year)
            checks["candidate_alignment"] &= len(year_rows) == int(
                result["candidate_prediction_row_count"]
            )
            fold = inputs.fold(year=year, horizon=2, target=str(task["target"]))
            checks["purge_valid"] &= int(
                result["maximum_train_signal_date_idx"]
            ) == int(fold["maximum_train_signal_date_idx"])
            checks["purge_valid"] &= int(
                result["maximum_train_signal_date_idx"]
            ) == int(fold["oos_start_date_idx"]) - 3
            checks["purge_valid"] &= int(result["purge_days"]) == 2
            checks["fixed_rounds_exact"] &= int(result["best_iteration"]) == 512
            checks["result_study_id_exact"] &= result.get("study_id") == STUDY_ID
            if year not in baseline_checked:
                baseline_result = _frozen_g1_result(
                    baseline_root=Path(context["baseline_root"]), year=year
                )
                baseline_rows, _ = _task_prediction(
                    result=baseline_result, inputs=inputs, year=year
                )
                checks["baseline_candidate_alignment"] &= np.array_equal(
                    baseline_rows, year_rows
                )
                baseline_checked.add(year)
        except (KeyError, OSError, TypeError, ValueError, IndexError, model.ModelError):
            checks["task_files_valid"] = False
    for record in dict(manifest.get("outputs", {}) or {}).values():
        checks["outputs_hashed"] &= _record_valid(record)
    annual_path = Path(manifest["outputs"]["annual_metrics"]["path"])
    paired_path = Path(manifest["outputs"]["paired_comparison"]["path"])
    daily_path = Path(manifest["outputs"]["daily_metrics"]["path"])
    annual = pd.read_parquet(annual_path)
    paired = pd.read_parquet(paired_path)
    daily = pd.read_parquet(daily_path)
    checks["evaluation_rows_exact"] = (
        len(annual) == TASK_COUNT * 2
        and len(paired) == TASK_COUNT
        and set(annual["model_source"].astype(str)) == {"direct_target", "frozen_g1"}
        and set(paired["target"].astype(str)) == set(TARGETS)
        and set(paired["year"].astype(int)) == set(ROLLING_YEARS)
    )
    comparison_metrics = [str(value) for value in study["comparison"]["metrics"]]
    for task in tasks:
        target = str(task["target"])
        year = int(task["year"])
        result = results.get(str(task["task_id"]))
        direct = annual.loc[
            annual["target"].eq(target)
            & annual["year"].eq(year)
            & annual["model_source"].eq("direct_target")
        ]
        baseline = annual.loc[
            annual["target"].eq(target)
            & annual["year"].eq(year)
            & annual["model_source"].eq("frozen_g1")
        ]
        comparison = paired.loc[
            paired["target"].eq(target) & paired["year"].eq(year)
        ]
        if result is None or len(direct) != 1 or len(baseline) != 1 or len(comparison) != 1:
            checks["evaluation_metrics_reconcile"] = False
            checks["comparison_support_identical"] = False
            continue
        fold = inputs.fold(year=year, horizon=2, target=target)
        expected_count = len(fold["evaluation_rows"])
        checks["comparison_support_identical"] &= (
            int(direct.iloc[0]["evaluation_row_count"]) == expected_count
            and int(baseline.iloc[0]["evaluation_row_count"]) == expected_count
            and int(comparison.iloc[0]["evaluation_row_count"]) == expected_count
        )
        direct_daily = daily.loc[
            daily["target"].eq(target)
            & daily["year"].eq(year)
            & daily["model_source"].eq("direct_target")
        ]
        baseline_daily = daily.loc[
            daily["target"].eq(target)
            & daily["year"].eq(year)
            & daily["model_source"].eq("frozen_g1")
        ]
        checks["comparison_support_identical"] &= np.array_equal(
            direct_daily["date_idx"].to_numpy(dtype=np.int32),
            baseline_daily["date_idx"].to_numpy(dtype=np.int32),
        )
        for metric_name in comparison_metrics:
            direct_value = float(direct.iloc[0][metric_name])
            baseline_value = float(baseline.iloc[0][metric_name])
            checks["evaluation_metrics_reconcile"] &= bool(
                np.isclose(
                    direct_value,
                    float(result["metrics"][metric_name]),
                    rtol=0.0,
                    atol=1.0e-12,
                    equal_nan=True,
                )
            )
            checks["evaluation_metrics_reconcile"] &= bool(
                np.isclose(
                    float(comparison.iloc[0][f"direct_{metric_name}"]),
                    direct_value,
                    rtol=0.0,
                    atol=1.0e-12,
                    equal_nan=True,
                )
                and np.isclose(
                    float(comparison.iloc[0][f"baseline_{metric_name}"]),
                    baseline_value,
                    rtol=0.0,
                    atol=1.0e-12,
                    equal_nan=True,
                )
                and np.isclose(
                    float(comparison.iloc[0][f"delta_{metric_name}"]),
                    direct_value - baseline_value,
                    rtol=0.0,
                    atol=1.0e-12,
                    equal_nan=True,
                )
            )
    payload = {
        "schema": AUDIT_SCHEMA,
        "status": "ok" if all(checks.values()) else "failed",
        "created_at": _now(),
        "study_id": STUDY_ID,
        "checks": checks,
        "task_count": len(results),
        "targets": list(TARGETS),
        "rolling_years": list(ROLLING_YEARS),
        "feature_count": FEATURE_COUNT,
        "rounds": 512,
        "maximum_label_value_error": float(label_audit["maximum_value_error"]),
        "maximum_decomposition_identity_error": float(
            label_audit["maximum_identity_error"]
        ),
        "open_to_close_2_valid_count": int(label_audit["valid_count"]),
        "forbidden_2026_read_count": 0,
        "training_performed": True,
        "execution_replay_performed": False,
        "feature_set_selected": False,
        "production_policy_selected": False,
        "evaluation_semantics": study["execution"]["evaluation_semantics"],
    }
    audit_path = output_root / "audit.json"
    _write_json(audit_path, payload)
    if payload["status"] != "ok":
        failed = sorted(name for name, passed in checks.items() if not passed)
        raise T1ReturnModelError(f"t1_return_model_audit_failed:{failed}")
    manifest["status"] = "audited"
    manifest["audit"] = _file_record(audit_path)
    manifest["updated_at"] = _now()
    _write_json(manifest_path, manifest)
    return payload


def status(*, output_root: Path = DEFAULT_OUTPUT_ROOT) -> dict[str, Any]:
    manifest_path = output_root / "manifest.json"
    audit_path = output_root / "audit.json"
    manifest = _read_json(manifest_path) if manifest_path.is_file() else {}
    audit_payload = _read_json(audit_path) if audit_path.is_file() else {}
    completed = len(model._completed_results(output_root)) if output_root.is_dir() else 0
    return {
        "study_id": STUDY_ID,
        "status": manifest.get("status", "not_prepared"),
        "audit_status": audit_payload.get("status", "not_audited"),
        "task_count": TASK_COUNT,
        "completed_count": completed,
        "training_performed": bool(manifest.get("training_performed", False)),
        "execution_replay_performed": bool(
            manifest.get("execution_replay_performed", False)
        ),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-path", type=Path, default=DEFAULT_STUDY_PATH)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument("--prepare", action="store_true")
    actions.add_argument("--run", action="store_true")
    actions.add_argument("--evaluate", action="store_true")
    actions.add_argument("--audit", action="store_true")
    actions.add_argument("--status", action="store_true")
    parser.add_argument("--max-tasks", type=int, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.prepare:
        result = prepare(study_path=args.study_path, output_root=args.output_root)
    elif args.run:
        result = run(
            study_path=args.study_path,
            output_root=args.output_root,
            max_tasks=args.max_tasks,
        )
    elif args.evaluate:
        result = evaluate(study_path=args.study_path, output_root=args.output_root)
    elif args.audit:
        result = audit(study_path=args.study_path, output_root=args.output_root)
    else:
        result = status(output_root=args.output_root)
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    return 0 if result.get("status") in {
        "prepared",
        "training_in_progress",
        "training_completed",
        "evaluated",
        "audited",
        "ok",
    } else 1


if __name__ == "__main__":
    raise SystemExit(main())
