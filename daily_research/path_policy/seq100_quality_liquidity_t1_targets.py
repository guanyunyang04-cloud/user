from __future__ import annotations

"""Prepare an A-share T+1 open-to-open label contract and frozen baseline."""

import argparse
import hashlib
import json
import math
import os
from collections.abc import Mapping, Sequence
from datetime import datetime
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
STUDY_ID = "seq100_quality_liquidity_t1_targets"
SOURCE_STUDY_ID = "seq100_quality_liquidity_model"
YEARS = tuple(range(2011, 2026))
ROLLING_YEARS = (2023, 2024, 2025)
FORBIDDEN_YEAR = 2026
MAXIMUM_OUTCOME_DATE = "2025-12-31"
FEATURE_COUNT = 557
LABEL_COLUMNS = (
    "open_to_open_1",
    "open_to_close_1",
    "close_to_open_1",
)
PRIMARY_LABEL = LABEL_COLUMNS[0]

FLAG_OUTCOME_WITHIN_CUTOFF = np.uint16(1 << 0)
FLAG_ENTRY_PRICE_OBSERVED = np.uint16(1 << 1)
FLAG_EXIT_PRICE_OBSERVED = np.uint16(1 << 2)
FLAG_ENTRY_EXECUTION_STATE_KNOWN = np.uint16(1 << 3)
FLAG_EXIT_EXECUTION_STATE_KNOWN = np.uint16(1 << 4)
FLAG_ENTRY_BUYABLE = np.uint16(1 << 5)
FLAG_EXIT_OPEN_SELLABLE = np.uint16(1 << 6)
FLAG_TARGET_VALID = np.uint16(1 << 7)
FLAG_DECOMPOSITION_VALID = np.uint16(1 << 8)

FLAG_DEFINITIONS = {
    "outcome_within_cutoff": int(FLAG_OUTCOME_WITHIN_CUTOFF),
    "entry_price_observed": int(FLAG_ENTRY_PRICE_OBSERVED),
    "exit_price_observed": int(FLAG_EXIT_PRICE_OBSERVED),
    "entry_execution_state_known": int(FLAG_ENTRY_EXECUTION_STATE_KNOWN),
    "exit_execution_state_known": int(FLAG_EXIT_EXECUTION_STATE_KNOWN),
    "entry_buyable": int(FLAG_ENTRY_BUYABLE),
    "exit_open_sellable": int(FLAG_EXIT_OPEN_SELLABLE),
    "target_valid": int(FLAG_TARGET_VALID),
    "decomposition_valid": int(FLAG_DECOMPOSITION_VALID),
}

LABEL_SCHEMA = "seq100_quality_liquidity_t1_labels/1"
MANIFEST_SCHEMA = "seq100_quality_liquidity_t1_targets_manifest/1"
BASELINE_SCHEMA = "seq100_quality_liquidity_t1_frozen_g1_baseline/1"
AUDIT_SCHEMA = "seq100_quality_liquidity_t1_targets_audit/1"

DEFAULT_STUDY_PATH = (
    WORKSPACE_ROOT / "daily_research/studies/seq100_quality_liquidity_t1_targets.json"
)
DEFAULT_OUTPUT_ROOT = (
    WORKSPACE_ROOT
    / "daily_research/output/path_policy/studies"
    / "seq100_quality_liquidity_t1_targets"
)
DEFAULT_MODEL_OUTPUT_ROOT = (
    WORKSPACE_ROOT
    / "daily_research/output/path_policy/studies"
    / "seq100_quality_liquidity_model"
)


class T1TargetError(RuntimeError):
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
        raise T1TargetError(f"required_json_missing:{path}")
    return dict(json.loads(path.read_text(encoding="utf-8")))


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"not_json_serializable:{type(value).__name__}")


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
            default=_json_default,
        )
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
        payload,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        default=_json_default,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _file_record(path: Path, **extra: Any) -> dict[str, Any]:
    record = {
        "path": str(path.resolve()),
        "sha256": _sha256(path),
        "size": int(path.stat().st_size),
    }
    record.update(extra)
    return record


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
        json.dumps(
            {"event": event, "at": _now(), **payload},
            ensure_ascii=False,
            default=_json_default,
        ),
        flush=True,
    )


def load_study(path: Path = DEFAULT_STUDY_PATH) -> dict[str, Any]:
    study = _read_json(path)
    if study.get("study_id") != STUDY_ID:
        raise T1TargetError("study_id_mismatch")
    source = dict(study.get("source", {}) or {})
    if str(source.get("source_study_id")) != SOURCE_STUDY_ID:
        raise T1TargetError("source_study_id_mismatch")
    if str(source.get("maximum_outcome_date")) != MAXIMUM_OUTCOME_DATE:
        raise T1TargetError("maximum_outcome_date_mismatch")
    if int(source.get("forbidden_year", -1)) != FORBIDDEN_YEAR:
        raise T1TargetError("forbidden_year_mismatch")
    if bool(source.get("candidate_selection_uses_future_fill", True)):
        raise T1TargetError("future_fill_candidate_gate_forbidden")
    period = dict(study.get("period", {}) or {})
    if int(period.get("burn_in_year", -1)) != 2010:
        raise T1TargetError("burn_in_year_mismatch")
    if tuple(int(value) for value in period.get("research_years", ())) != YEARS:
        raise T1TargetError("research_years_mismatch")
    if (
        tuple(int(value) for value in period.get("rolling_prediction_years", ()))
        != ROLLING_YEARS
    ):
        raise T1TargetError("rolling_years_mismatch")
    target = dict(study.get("target", {}) or {})
    if (
        str(target.get("primary")) != PRIMARY_LABEL
        or tuple(str(value) for value in target.get("components", ()))
        != LABEL_COLUMNS[1:]
        or str(target.get("entry")) != "next_trading_day_open"
        or str(target.get("exit")) != "second_next_trading_day_open"
        or str(target.get("price_semantics")) != "back_adjusted_total_return"
    ):
        raise T1TargetError("target_contract_mismatch")
    contract = dict(study.get("label_contract", {}) or {})
    if (
        str(contract.get("schema_version")) != LABEL_SCHEMA
        or tuple(str(value) for value in contract.get("label_columns", ()))
        != LABEL_COLUMNS
    ):
        raise T1TargetError("label_contract_schema_mismatch")
    execution = dict(study.get("execution", {}) or {})
    if bool(execution.get("training_performed", True)):
        raise T1TargetError("target_contract_must_not_train")
    return study


def _open_array(record: Mapping[str, Any], *, dtype: str | np.dtype) -> np.memmap:
    path = _resolve(str(record["path"]))
    shape = tuple(int(value) for value in record["shape"])
    expected_size = int(np.prod(shape)) * np.dtype(dtype).itemsize
    if not path.is_file() or int(path.stat().st_size) != expected_size:
        raise T1TargetError(f"source_array_invalid:{path}")
    return np.memmap(path, dtype=dtype, mode="r", shape=shape)


def _source_context(study: Mapping[str, Any]) -> dict[str, Any]:
    input_path = _resolve(str(study["source"]["model_input_manifest"]))
    input_manifest = _read_json(input_path)
    if (
        input_manifest.get("study_id") != SOURCE_STUDY_ID
        or input_manifest.get("status") != "completed"
    ):
        raise T1TargetError("model_input_manifest_contract_mismatch")
    row_record = dict(input_manifest["row_index"])
    row_path = _resolve(str(row_record["path"]))
    if not _record_valid(row_record):
        raise T1TargetError("model_input_row_index_invalid")
    row_index = pd.read_parquet(row_path)
    expected_columns = [
        "candidate_id",
        "date_idx",
        "trade_date",
        "symbol",
        "security_id",
    ]
    if list(row_index.columns) != expected_columns:
        raise T1TargetError("model_input_row_index_schema_mismatch")
    if len(row_index) != int(input_manifest.get("row_count", -1)):
        raise T1TargetError("model_input_row_count_mismatch")
    candidate_id = row_index["candidate_id"].to_numpy(dtype=np.int64)
    date_idx = row_index["date_idx"].to_numpy(dtype=np.int32)
    if (
        not row_index["candidate_id"].is_unique
        or bool((candidate_id[1:] <= candidate_id[:-1]).any())
        or bool((date_idx[1:] < date_idx[:-1]).any())
    ):
        raise T1TargetError("model_input_row_order_invalid")
    trade_date = row_index["trade_date"].astype(str)
    formal_years = tuple(sorted(trade_date.str[:4].astype(int).unique().tolist()))
    if formal_years != YEARS or bool(trade_date.str.startswith("2026-").any()):
        raise T1TargetError("model_input_formal_period_mismatch")

    label_path = _resolve(str(input_manifest["label_manifest"]["path"]))
    label_manifest = _read_json(label_path)
    pack_path = _resolve(str(label_manifest["source"]["pack_manifest"]))
    pack = _read_json(pack_path)
    semantics = dict(pack.get("data_semantics", {}) or {})
    if str(semantics.get("price_adjustment")) != "back_adjust" or bool(
        semantics.get("candidate_selection_uses_future_labels", True)
    ):
        raise T1TargetError("pack_price_or_candidate_semantics_mismatch")
    date_values = np.asarray(pack.get("date_values", ()), dtype=str)
    if (
        date_values.ndim != 1
        or not len(date_values)
        or bool((date_values[1:] <= date_values[:-1]).any())
    ):
        raise T1TargetError("pack_date_values_invalid")
    cutoff_positions = np.flatnonzero(date_values == MAXIMUM_OUTCOME_DATE)
    if len(cutoff_positions) != 1:
        raise T1TargetError("pack_outcome_cutoff_missing_or_duplicate")
    cutoff_idx = int(cutoff_positions[0])
    if cutoff_idx + 1 >= len(date_values) or not bool(
        np.char.startswith(date_values[cutoff_idx + 1 :], "2026-").any()
    ):
        raise T1TargetError("pack_forbidden_year_boundary_not_explicit")
    if int(date_idx.max()) >= len(date_values):
        raise T1TargetError("model_input_date_exceeds_pack")
    for start in range(0, len(row_index), 500_000):
        stop = min(start + 500_000, len(row_index))
        if not np.array_equal(
            date_values[date_idx[start:stop]],
            trade_date.iloc[start:stop].to_numpy(dtype=str),
        ):
            raise T1TargetError("model_input_trade_date_pack_alignment_failed")

    symbol_values = pd.Index([str(value) for value in pack.get("symbol_values", ())])
    if not symbol_values.is_unique:
        raise T1TargetError("pack_symbol_values_duplicate")
    symbol_idx = symbol_values.get_indexer(row_index["symbol"].astype(str))
    if bool((symbol_idx < 0).any()):
        raise T1TargetError("model_input_symbol_missing_from_pack")
    symbol_idx = symbol_idx.astype(np.int32, copy=False)

    daily_record = dict(pack["feature_channels"]["daily_raw"])
    daily_columns = tuple(str(value) for value in daily_record.get("columns", ()))
    if daily_columns[:4] != ("open", "high", "low", "close"):
        raise T1TargetError("daily_raw_ohlc_contract_mismatch")
    panel_shape = tuple(int(value) for value in daily_record["shape"][:2])
    expected_panel_shape = (len(date_values), len(symbol_values))
    if panel_shape != expected_panel_shape:
        raise T1TargetError("daily_raw_panel_shape_mismatch")
    required_records = {
        "price_observed": dict(pack["masks"]["price_observed"]),
        "entry_filled": dict(pack["masks"]["entry_filled"]),
        "status_valid": dict(pack["masks"]["status_valid"]),
        "is_suspended": dict(pack["masks"]["is_suspended"]),
        "is_delisted": dict(pack["masks"]["is_delisted"]),
        "raw_open": dict(pack["execution_arrays"]["entry_open_raw"]),
        "raw_down_limit": dict(pack["execution_arrays"]["exit_down_limit_raw"]),
    }
    if any(
        tuple(int(value) for value in record["shape"]) != expected_panel_shape
        for record in required_records.values()
    ):
        raise T1TargetError("execution_source_panel_shape_mismatch")
    return {
        "input_path": input_path,
        "input_manifest": input_manifest,
        "row_path": row_path,
        "row_index": row_index,
        "candidate_id": candidate_id,
        "date_idx": date_idx,
        "symbol_idx": symbol_idx,
        "label_path": label_path,
        "label_manifest": label_manifest,
        "pack_path": pack_path,
        "pack": pack,
        "date_values": date_values,
        "cutoff_idx": cutoff_idx,
        "daily_record": daily_record,
        "source_records": required_records,
    }


def _experiment_fingerprint(*, study_path: Path, context: Mapping[str, Any]) -> str:
    return _stable_hash(
        {
            "schema": LABEL_SCHEMA,
            "study_sha256": _sha256(study_path),
            "model_input_manifest_sha256": _sha256(Path(context["input_path"])),
            "model_input_fingerprint": str(
                context["input_manifest"].get("input_fingerprint", "")
            ),
            "row_index_sha256": _sha256(Path(context["row_path"])),
            "pack_manifest_sha256": _sha256(Path(context["pack_path"])),
            "label_columns": list(LABEL_COLUMNS),
            "date_offsets": {"entry_open": 1, "entry_close": 1, "exit_open": 2},
            "primary_formula": "adjusted_open[t+2]/adjusted_open[t+1]-1",
            "label_support": "both_open_prices_observed_finite_positive",
            "future_fill_used_for_label_support": False,
            "maximum_outcome_date": MAXIMUM_OUTCOME_DATE,
            "forbidden_year": FORBIDDEN_YEAR,
            "flag_definitions": FLAG_DEFINITIONS,
        }
    )


def _derive_open_sellable(
    *,
    raw_open: np.ndarray,
    raw_down_limit: np.ndarray,
    status_valid: np.ndarray,
    suspended: np.ndarray,
    delisted: np.ndarray,
) -> np.ndarray:
    open_values = np.asarray(raw_open, dtype=np.float64)
    limit_values = np.asarray(raw_down_limit, dtype=np.float64)
    base = (
        np.isfinite(open_values)
        & (open_values > 0.0)
        & np.asarray(status_valid, dtype=bool)
        & ~np.asarray(suspended, dtype=bool)
        & ~np.asarray(delisted, dtype=bool)
    )
    blocked = (
        base
        & np.isfinite(limit_values)
        & (np.floor(open_values / 0.01 + 0.5) <= np.floor(limit_values / 0.01 + 0.5))
    )
    return base & ~blocked


def _derive_batch(
    *,
    signal_date_idx: np.ndarray,
    symbol_idx: np.ndarray,
    cutoff_idx: int,
    daily_raw: np.ndarray,
    price_observed: np.ndarray,
    entry_filled: np.ndarray,
    status_valid: np.ndarray,
    suspended: np.ndarray,
    delisted: np.ndarray,
    raw_open: np.ndarray,
    raw_down_limit: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    dates = np.asarray(signal_date_idx, dtype=np.int64)
    symbols = np.asarray(symbol_idx, dtype=np.int64)
    if dates.ndim != 1 or symbols.shape != dates.shape:
        raise T1TargetError("derive_batch_index_shape_mismatch")
    row_count = len(dates)
    returns = np.full((row_count, len(LABEL_COLUMNS)), np.nan, dtype=np.float32)
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
    entry_close = np.asarray(
        daily_raw[entry_dates, current_symbols, 3], dtype=np.float64
    )
    exit_open = np.asarray(daily_raw[exit_dates, current_symbols, 0], dtype=np.float64)
    entry_source_observed = np.asarray(
        price_observed[entry_dates, current_symbols], dtype=bool
    )
    exit_source_observed = np.asarray(
        price_observed[exit_dates, current_symbols], dtype=bool
    )
    entry_observed = (
        entry_source_observed & np.isfinite(entry_open) & (entry_open > 0.0)
    )
    exit_observed = exit_source_observed & np.isfinite(exit_open) & (exit_open > 0.0)
    flags[positions[entry_observed]] |= FLAG_ENTRY_PRICE_OBSERVED
    flags[positions[exit_observed]] |= FLAG_EXIT_PRICE_OBSERVED

    entry_status = np.asarray(status_valid[entry_dates, current_symbols], dtype=bool)
    exit_status = np.asarray(status_valid[exit_dates, current_symbols], dtype=bool)
    entry_suspended = np.asarray(suspended[entry_dates, current_symbols], dtype=bool)
    exit_suspended = np.asarray(suspended[exit_dates, current_symbols], dtype=bool)
    entry_delisted = np.asarray(delisted[entry_dates, current_symbols], dtype=bool)
    exit_delisted = np.asarray(delisted[exit_dates, current_symbols], dtype=bool)
    entry_state_known = entry_status & (
        entry_observed | entry_suspended | entry_delisted
    )
    exit_state_known = exit_status & (exit_observed | exit_suspended | exit_delisted)
    flags[positions[entry_state_known]] |= FLAG_ENTRY_EXECUTION_STATE_KNOWN
    flags[positions[exit_state_known]] |= FLAG_EXIT_EXECUTION_STATE_KNOWN

    entry_buyable = (
        np.asarray(entry_filled[signal_dates, current_symbols], dtype=bool)
        & entry_state_known
    )
    flags[positions[entry_buyable]] |= FLAG_ENTRY_BUYABLE
    exit_sellable = (
        _derive_open_sellable(
            raw_open=np.asarray(
                raw_open[exit_dates, current_symbols], dtype=np.float64
            ),
            raw_down_limit=np.asarray(
                raw_down_limit[exit_dates, current_symbols], dtype=np.float64
            ),
            status_valid=exit_status,
            suspended=exit_suspended,
            delisted=exit_delisted,
        )
        & exit_state_known
    )
    flags[positions[exit_sellable]] |= FLAG_EXIT_OPEN_SELLABLE

    target_valid = entry_observed & exit_observed
    global_valid = positions[target_valid]
    flags[global_valid] |= FLAG_TARGET_VALID
    valid_store[global_valid] = 1
    returns[global_valid, 0] = (
        exit_open[target_valid] / entry_open[target_valid] - 1.0
    ).astype(np.float32)

    decomposition_valid = target_valid & np.isfinite(entry_close) & (entry_close > 0.0)
    global_decomposition = positions[decomposition_valid]
    flags[global_decomposition] |= FLAG_DECOMPOSITION_VALID
    returns[global_decomposition, 1] = (
        entry_close[decomposition_valid] / entry_open[decomposition_valid] - 1.0
    ).astype(np.float32)
    returns[global_decomposition, 2] = (
        exit_open[decomposition_valid] / entry_close[decomposition_valid] - 1.0
    ).astype(np.float32)
    return returns, valid_store, flags


def _flag_mask(flags: np.ndarray, flag: np.uint16) -> np.ndarray:
    return (np.asarray(flags, dtype=np.uint16) & flag) != 0


def _coverage_frame(row_index: pd.DataFrame, flags: np.ndarray) -> pd.DataFrame:
    years = row_index["trade_date"].astype(str).str[:4].astype(int).to_numpy()
    current = np.asarray(flags, dtype=np.uint16)
    masks = {
        name: _flag_mask(current, np.uint16(value))
        for name, value in FLAG_DEFINITIONS.items()
    }
    fully_executable = (
        masks["target_valid"]
        & masks["entry_execution_state_known"]
        & masks["exit_execution_state_known"]
        & masks["entry_buyable"]
        & masks["exit_open_sellable"]
    )
    entry_known_blocked = (
        masks["target_valid"]
        & masks["entry_execution_state_known"]
        & ~masks["entry_buyable"]
    )
    exit_known_blocked = (
        masks["target_valid"]
        & masks["exit_execution_state_known"]
        & ~masks["exit_open_sellable"]
    )
    rows: list[dict[str, Any]] = []
    for year in YEARS:
        selected = years == year
        count = int(selected.sum())
        record: dict[str, Any] = {"signal_year": year, "row_count": count}
        for name, mask in masks.items():
            record[f"{name}_count"] = int(np.sum(mask & selected))
        record["fully_executable_count"] = int(np.sum(fully_executable & selected))
        record["entry_known_blocked_count"] = int(
            np.sum(entry_known_blocked & selected)
        )
        record["exit_known_blocked_count"] = int(np.sum(exit_known_blocked & selected))
        record["target_coverage"] = (
            float(record["target_valid_count"] / count) if count else float("nan")
        )
        valid_count = int(record["target_valid_count"])
        record["fully_executable_on_valid"] = (
            float(record["fully_executable_count"] / valid_count)
            if valid_count
            else float("nan")
        )
        rows.append(record)
    return pd.DataFrame(rows)


def _label_statistics(returns: np.ndarray, valid: np.ndarray) -> dict[str, Any]:
    primary_valid = np.asarray(valid, dtype=bool)
    primary = np.asarray(returns[:, 0], dtype=np.float64)[primary_valid]
    decomposition = np.isfinite(np.asarray(returns[:, 1:], dtype=np.float64)).all(
        axis=1
    )
    intraday = np.asarray(returns[decomposition, 1], dtype=np.float64)
    overnight = np.asarray(returns[decomposition, 2], dtype=np.float64)
    reconstructed = (1.0 + intraday) * (1.0 + overnight) - 1.0
    identity_error = np.abs(
        reconstructed - np.asarray(returns[decomposition, 0], dtype=np.float64)
    )
    quantiles = np.quantile(primary, [0.0, 0.001, 0.01, 0.5, 0.99, 0.999, 1.0])
    return {
        "valid_count": int(primary_valid.sum()),
        "decomposition_valid_count": int(decomposition.sum()),
        "primary_mean": float(primary.mean()),
        "primary_std": float(primary.std(ddof=0)),
        "primary_quantiles": {
            key: float(value)
            for key, value in zip(
                ("minimum", "q001", "q01", "median", "q99", "q999", "maximum"),
                quantiles,
                strict=True,
            )
        },
        "intraday_mean": float(intraday.mean()),
        "overnight_mean": float(overnight.mean()),
        "maximum_decomposition_identity_error": float(identity_error.max(initial=0.0)),
    }


def _prepared_manifest_valid(manifest: Mapping[str, Any], *, fingerprint: str) -> bool:
    if (
        manifest.get("schema") != MANIFEST_SCHEMA
        or manifest.get("status") not in {"prepared", "audited"}
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
            raise T1TargetError("existing_output_fingerprint_mismatch")
        if _prepared_manifest_valid(current, fingerprint=fingerprint):
            return current
    if int(chunk_size) <= 0:
        raise T1TargetError("chunk_size_must_be_positive")

    row_index = context["row_index"]
    row_count = len(row_index)
    label_root = output_root / "labels"
    label_root.mkdir(parents=True, exist_ok=True)
    returns_path = label_root / "t1_open_returns.float32.dat"
    valid_path = label_root / "t1_open_return_valid.uint8.dat"
    flags_path = label_root / "t1_open_return_flags.uint16.dat"
    partial_paths = {
        returns_path: Path(str(returns_path) + ".partial"),
        valid_path: Path(str(valid_path) + ".partial"),
        flags_path: Path(str(flags_path) + ".partial"),
    }
    for partial in partial_paths.values():
        partial.unlink(missing_ok=True)
    returns_store = np.memmap(
        partial_paths[returns_path],
        dtype=np.float32,
        mode="w+",
        shape=(row_count, len(LABEL_COLUMNS)),
    )
    valid_store = np.memmap(
        partial_paths[valid_path],
        dtype=np.uint8,
        mode="w+",
        shape=(row_count,),
    )
    flags_store = np.memmap(
        partial_paths[flags_path],
        dtype=np.uint16,
        mode="w+",
        shape=(row_count,),
    )
    returns_store[:] = np.nan
    valid_store[:] = 0
    flags_store[:] = 0

    records = context["source_records"]
    daily_raw = _open_array(context["daily_record"], dtype=np.float32)
    price_observed = _open_array(records["price_observed"], dtype=np.bool_)
    entry_filled = _open_array(records["entry_filled"], dtype=np.bool_)
    status_valid = _open_array(records["status_valid"], dtype=np.bool_)
    suspended = _open_array(records["is_suspended"], dtype=np.bool_)
    delisted = _open_array(records["is_delisted"], dtype=np.bool_)
    raw_open = _open_array(records["raw_open"], dtype=np.float32)
    raw_down_limit = _open_array(records["raw_down_limit"], dtype=np.float32)
    _emit("t1_target_prepare_started", row_count=row_count)
    for start in range(0, row_count, int(chunk_size)):
        stop = min(start + int(chunk_size), row_count)
        batch_returns, batch_valid, batch_flags = _derive_batch(
            signal_date_idx=context["date_idx"][start:stop],
            symbol_idx=context["symbol_idx"][start:stop],
            cutoff_idx=int(context["cutoff_idx"]),
            daily_raw=daily_raw,
            price_observed=price_observed,
            entry_filled=entry_filled,
            status_valid=status_valid,
            suspended=suspended,
            delisted=delisted,
            raw_open=raw_open,
            raw_down_limit=raw_down_limit,
        )
        returns_store[start:stop] = batch_returns
        valid_store[start:stop] = batch_valid
        flags_store[start:stop] = batch_flags
    returns_store.flush()
    valid_store.flush()
    flags_store.flush()
    del returns_store, valid_store, flags_store
    for final, partial in partial_paths.items():
        os.replace(partial, final)

    returns_store = np.memmap(
        returns_path,
        dtype=np.float32,
        mode="r",
        shape=(row_count, len(LABEL_COLUMNS)),
    )
    valid_store = np.memmap(valid_path, dtype=np.uint8, mode="r", shape=(row_count,))
    flags_store = np.memmap(flags_path, dtype=np.uint16, mode="r", shape=(row_count,))
    coverage = _coverage_frame(row_index, flags_store)
    statistics = _label_statistics(returns_store, valid_store)
    coverage_path = output_root / "label_coverage.parquet"
    statistics_path = output_root / "label_statistics.json"
    _write_parquet(coverage, coverage_path)
    _write_json(statistics_path, statistics)

    within = _flag_mask(flags_store, FLAG_OUTCOME_WITHIN_CUTOFF)
    maximum_signal_row = int(np.flatnonzero(within)[-1])
    maximum_outcome_idx = int(np.max(context["date_idx"][within].astype(np.int64) + 2))
    target_valid = np.asarray(valid_store, dtype=bool)
    entry_buyable = _flag_mask(flags_store, FLAG_ENTRY_BUYABLE)
    exit_sellable = _flag_mask(flags_store, FLAG_EXIT_OPEN_SELLABLE)
    entry_known = _flag_mask(flags_store, FLAG_ENTRY_EXECUTION_STATE_KNOWN)
    exit_known = _flag_mask(flags_store, FLAG_EXIT_EXECUTION_STATE_KNOWN)
    fully_executable = (
        target_valid & entry_known & exit_known & entry_buyable & exit_sellable
    )
    files = {
        "returns": _file_record(
            returns_path,
            shape=[row_count, len(LABEL_COLUMNS)],
            dtype=str(np.dtype(np.float32)),
            columns=list(LABEL_COLUMNS),
        ),
        "valid": _file_record(
            valid_path,
            shape=[row_count],
            dtype=str(np.dtype(np.uint8)),
        ),
        "flags": _file_record(
            flags_path,
            shape=[row_count],
            dtype=str(np.dtype(np.uint16)),
            definitions=FLAG_DEFINITIONS,
        ),
    }
    label_contract = {
        "schema": LABEL_SCHEMA,
        "status": "completed",
        "created_at": _now(),
        "study_id": STUDY_ID,
        "experiment_fingerprint": fingerprint,
        "row_count": row_count,
        "label_columns": list(LABEL_COLUMNS),
        "primary_label": PRIMARY_LABEL,
        "valid_count": int(target_valid.sum()),
        "invalid_count": int((~target_valid).sum()),
        "fully_executable_count": int(fully_executable.sum()),
        "signal_decision": "signal_day_close",
        "date_offsets": {"entry_open": 1, "entry_close": 1, "exit_open": 2},
        "holding_period_trading_days_after_entry": 1,
        "primary_formula": "adjusted_open[t+2]/adjusted_open[t+1]-1",
        "decomposition_formula": ("(1+open_to_close_1)*(1+close_to_open_1)-1"),
        "price_semantics": "back_adjusted_total_return",
        "label_validity": ("entry and exit open prices observed, finite, and positive"),
        "label_validity_dependencies": [
            "signal_date_idx",
            "entry_price_observed",
            "exit_price_observed",
            "adjusted_entry_open",
            "adjusted_exit_open",
            "maximum_outcome_date",
        ],
        "future_fill_used_for_label_support": False,
        "candidate_selection_uses_future_fill": False,
        "execution_state_semantics": {
            "entry_buyable": (
                "separate future state from the source entry_filled mask"
            ),
            "exit_open_sellable": (
                "separate future state from raw exit open, down limit, status, "
                "suspension, and delisting evidence"
            ),
            "false_without_execution_state_known": "unknown_not_known_blocked",
            "training_use": "not_part_of_primary_label_validity",
        },
        "flag_definitions": FLAG_DEFINITIONS,
        "maximum_signal_trade_date_read": str(
            row_index["trade_date"].iloc[maximum_signal_row]
        ),
        "maximum_outcome_date_read": str(context["date_values"][maximum_outcome_idx]),
        "forbidden_2026_read_count": 0,
        "statistics": statistics,
        "sources": {
            "model_input_manifest": _file_record(Path(context["input_path"])),
            "row_index": _file_record(Path(context["row_path"]), row_count=row_count),
            "label_manifest": _file_record(Path(context["label_path"])),
            "pack_manifest": _file_record(Path(context["pack_path"])),
            "daily_raw": {
                "path": str(_resolve(context["daily_record"]["path"])),
                "shape": list(context["daily_record"]["shape"]),
                "columns": list(context["daily_record"]["columns"]),
                "read_fields": ["open", "close"],
            },
        },
        "files": files,
        "training_performed": False,
    }
    contract_path = output_root / "label_contract.json"
    _write_json(contract_path, label_contract)
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "status": "prepared",
        "created_at": _now(),
        "study_id": STUDY_ID,
        "experiment_fingerprint": fingerprint,
        "study_config": _file_record(study_path),
        "model_inputs": _file_record(
            Path(context["input_path"]),
            input_fingerprint=str(
                context["input_manifest"].get("input_fingerprint", "")
            ),
        ),
        "row_count": row_count,
        "formal_years": list(YEARS),
        "rolling_years": list(ROLLING_YEARS),
        "maximum_outcome_date": MAXIMUM_OUTCOME_DATE,
        "forbidden_year": FORBIDDEN_YEAR,
        "forbidden_2026_read_count": 0,
        "primary_label": PRIMARY_LABEL,
        "training_performed": False,
        "feature_set_selected": False,
        "preparation_files": {
            "label_contract": _file_record(contract_path),
            "label_coverage": _file_record(coverage_path, row_count=len(coverage)),
            "label_statistics": _file_record(statistics_path),
        },
    }
    _write_json(manifest_path, manifest)
    _emit(
        "t1_target_prepare_completed",
        row_count=row_count,
        valid_count=int(target_valid.sum()),
        fully_executable_count=int(fully_executable.sum()),
    )
    return manifest


def _safe_spearman(left: np.ndarray, right: np.ndarray) -> float:
    if len(left) < 3:
        return float("nan")
    value = stats.spearmanr(left, right).statistic
    return float(value) if np.isfinite(value) else float("nan")


def _daily_return_metrics(
    *, date_idx: np.ndarray, actual: np.ndarray, prediction: np.ndarray
) -> pd.DataFrame:
    dates = np.asarray(date_idx, dtype=np.int32)
    truth = np.asarray(actual, dtype=np.float64)
    score = np.asarray(prediction, dtype=np.float64)
    if dates.shape != truth.shape or score.shape != truth.shape:
        raise T1TargetError("baseline_metric_shape_mismatch")
    if bool((dates[1:] < dates[:-1]).any()):
        raise T1TargetError("baseline_metric_dates_not_ordered")
    boundaries = np.flatnonzero(np.r_[True, dates[1:] != dates[:-1], True])
    rows: list[dict[str, Any]] = []
    for left, right in pairwise(boundaries):
        valid = np.isfinite(truth[left:right]) & np.isfinite(score[left:right])
        if int(valid.sum()) < 20:
            continue
        current_truth = truth[left:right][valid]
        current_score = score[left:right][valid]
        order_prediction = np.argsort(current_score, kind="stable")
        order_truth = np.argsort(current_truth, kind="stable")
        count = len(current_truth)
        top1_count = max(1, math.ceil(0.01 * count))
        top5_count = max(1, math.ceil(0.05 * count))
        top1_prediction = order_prediction[-top1_count:]
        top5_prediction = order_prediction[-top5_count:]
        top1_truth = set(order_truth[-top1_count:].tolist())
        top5_truth = set(order_truth[-top5_count:].tolist())
        universe_mean = float(current_truth.mean())
        top1_mean = float(current_truth[top1_prediction].mean())
        top5_mean = float(current_truth[top5_prediction].mean())
        rows.append(
            {
                "date_idx": int(dates[left]),
                "row_count": count,
                "rank_ic": _safe_spearman(current_truth, current_score),
                "universe_return_mean": universe_mean,
                "top1_return_mean": top1_mean,
                "top5_return_mean": top5_mean,
                "top1_excess_mean": top1_mean - universe_mean,
                "top5_excess_mean": top5_mean - universe_mean,
                "top1_capture": float(
                    len(top1_truth.intersection(set(top1_prediction.tolist())))
                    / len(top1_truth)
                ),
                "top5_capture": float(
                    len(top5_truth.intersection(set(top5_prediction.tolist())))
                    / len(top5_truth)
                ),
            }
        )
    if not rows:
        raise T1TargetError("baseline_metrics_have_no_valid_dates")
    return pd.DataFrame(rows)


def _baseline_task_path(model_output_root: Path, year: int) -> Path:
    return (
        model_output_root
        / "direct_returns/tasks"
        / f"return_zscore__g_01__compact_core__{int(year)}"
        / "task_result.json"
    )


def evaluate_frozen_g1_baseline(
    *,
    study_path: Path = DEFAULT_STUDY_PATH,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    model_output_root: Path = DEFAULT_MODEL_OUTPUT_ROOT,
) -> dict[str, Any]:
    study = load_study(study_path)
    manifest_path = output_root / "manifest.json"
    if not manifest_path.is_file():
        raise T1TargetError("t1_labels_not_prepared")
    manifest = _read_json(manifest_path)
    context = _source_context(study)
    expected_fingerprint = _experiment_fingerprint(
        study_path=study_path, context=context
    )
    if not _prepared_manifest_valid(manifest, fingerprint=expected_fingerprint):
        raise T1TargetError("t1_label_manifest_invalid")
    contract = _read_json(Path(manifest["preparation_files"]["label_contract"]["path"]))
    returns_record = dict(contract["files"]["returns"])
    valid_record = dict(contract["files"]["valid"])
    row_count = int(contract["row_count"])
    returns_store = np.memmap(
        Path(returns_record["path"]),
        dtype=np.dtype(returns_record["dtype"]),
        mode="r",
        shape=tuple(int(value) for value in returns_record["shape"]),
    )
    valid_store = np.memmap(
        Path(valid_record["path"]),
        dtype=np.dtype(valid_record["dtype"]),
        mode="r",
        shape=tuple(int(value) for value in valid_record["shape"]),
    )
    if row_count != len(context["row_index"]):
        raise T1TargetError("baseline_row_count_mismatch")
    years = (
        context["row_index"]["trade_date"].astype(str).str[:4].astype(int).to_numpy()
    )
    daily_parts: list[pd.DataFrame] = []
    annual_rows: list[dict[str, Any]] = []
    task_references: list[dict[str, Any]] = []
    metric_columns = (
        "rank_ic",
        "universe_return_mean",
        "top1_return_mean",
        "top5_return_mean",
        "top1_excess_mean",
        "top5_excess_mean",
        "top1_capture",
        "top5_capture",
    )
    for year in ROLLING_YEARS:
        task_path = _baseline_task_path(model_output_root, year)
        task = _read_json(task_path)
        if (
            task.get("status") != "completed"
            or task.get("source_label") != "g_1"
            or int(task.get("horizon", -1)) != 1
            or int(task.get("feature_count", -1)) != FEATURE_COUNT
            or int(task.get("evaluation_year", -1)) != year
        ):
            raise T1TargetError(f"frozen_g1_task_contract_mismatch:{year}")
        prediction_record = dict(task["files"]["prediction"])
        candidate_record = dict(task["files"]["candidate_id"])
        if not _record_valid(prediction_record) or not _record_valid(candidate_record):
            raise T1TargetError(f"frozen_g1_task_files_invalid:{year}")
        prediction = np.load(
            Path(prediction_record["path"]), mmap_mode="r", allow_pickle=False
        )
        candidate_ids = np.load(
            Path(candidate_record["path"]), mmap_mode="r", allow_pickle=False
        )
        year_rows = np.flatnonzero(years == year).astype(np.int64, copy=False)
        if len(prediction) != len(year_rows) or not np.array_equal(
            candidate_ids, context["candidate_id"][year_rows]
        ):
            raise T1TargetError(f"frozen_g1_candidate_alignment_failed:{year}")
        current_valid = np.asarray(valid_store[year_rows], dtype=bool)
        positions = np.flatnonzero(
            current_valid & np.isfinite(np.asarray(prediction, dtype=np.float64))
        )
        daily = _daily_return_metrics(
            date_idx=context["date_idx"][year_rows][positions],
            actual=np.asarray(returns_store[year_rows, 0], dtype=np.float32)[positions],
            prediction=np.asarray(prediction, dtype=np.float32)[positions],
        )
        daily.insert(
            1,
            "trade_date",
            [str(context["date_values"][int(value)]) for value in daily["date_idx"]],
        )
        daily.insert(0, "evaluation_year", year)
        daily_parts.append(daily)
        record: dict[str, Any] = {
            "evaluation_year": year,
            "candidate_row_count": len(year_rows),
            "target_valid_count": int(current_valid.sum()),
            "target_coverage": float(current_valid.mean()),
            "date_count": len(daily),
        }
        for column in metric_columns:
            record[column] = float(pd.to_numeric(daily[column], errors="coerce").mean())
        annual_rows.append(record)
        task_references.append(
            {
                "evaluation_year": year,
                "task_id": str(task["task_id"]),
                "task_result": _file_record(task_path),
                "prediction_sha256": str(prediction_record["sha256"]),
                "candidate_id_sha256": str(candidate_record["sha256"]),
            }
        )
    daily = pd.concat(daily_parts, ignore_index=True).sort_values(
        ["evaluation_year", "trade_date"]
    )
    annual = pd.DataFrame(annual_rows).sort_values("evaluation_year")
    baseline_config = dict(study["frozen_baseline"])
    positive_rank_years = int(
        annual["rank_ic"].gt(float(baseline_config["minimum_rank_ic"])).sum()
    )
    positive_top5_years = int(
        annual["top5_excess_mean"]
        .gt(float(baseline_config["minimum_top5_excess"]))
        .sum()
    )
    retraining_justified = positive_rank_years >= int(
        baseline_config["minimum_positive_rank_ic_years"]
    ) and positive_top5_years >= int(
        baseline_config["minimum_positive_top5_excess_years"]
    )
    decision = {
        "status": "completed",
        "overall": (
            "bounded_t1_retraining_is_justified"
            if retraining_justified
            else "frozen_signal_does_not_justify_t1_retraining_yet"
        ),
        "retraining_justified": bool(retraining_justified),
        "positive_rank_ic_years": positive_rank_years,
        "positive_top5_excess_years": positive_top5_years,
        "required_positive_rank_ic_years": int(
            baseline_config["minimum_positive_rank_ic_years"]
        ),
        "required_positive_top5_excess_years": int(
            baseline_config["minimum_positive_top5_excess_years"]
        ),
        "training_performed": False,
        "feature_set_selected": False,
        "production_policy_selected": False,
        "next_scope": (
            "three 2023-2025 rolling compact_core open-to-open heads with a "
            "two-trading-day outcome purge"
            if retraining_justified
            else "stop before training and revisit target evidence"
        ),
    }
    baseline_root = output_root / "frozen_g1_baseline"
    daily_path = baseline_root / "daily_metrics.parquet"
    annual_path = baseline_root / "annual_metrics.parquet"
    decision_path = baseline_root / "decision.json"
    _write_parquet(daily, daily_path)
    _write_parquet(annual, annual_path)
    _write_json(decision_path, decision)
    baseline_manifest = {
        "schema": BASELINE_SCHEMA,
        "status": "completed",
        "created_at": _now(),
        "study_id": STUDY_ID,
        "experiment_fingerprint": expected_fingerprint,
        "source_model_label": "g_1",
        "evaluated_target": PRIMARY_LABEL,
        "feature_count": FEATURE_COUNT,
        "rolling_years": list(ROLLING_YEARS),
        "task_count": len(task_references),
        "task_references": task_references,
        "decision": decision,
        "files": {
            "daily_metrics": _file_record(daily_path, row_count=len(daily)),
            "annual_metrics": _file_record(annual_path, row_count=len(annual)),
            "decision": _file_record(decision_path),
        },
        "forbidden_2026_read_count": 0,
        "training_performed": False,
    }
    baseline_manifest_path = baseline_root / "manifest.json"
    _write_json(baseline_manifest_path, baseline_manifest)
    manifest["frozen_g1_baseline"] = _file_record(baseline_manifest_path)
    manifest["retraining_decision"] = decision
    manifest["updated_at"] = _now()
    _write_json(manifest_path, manifest)
    return baseline_manifest


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
    files = dict(contract.get("files", {}) or {})
    row_count = len(context["row_index"])
    returns_record = dict(files["returns"])
    valid_record = dict(files["valid"])
    flags_record = dict(files["flags"])
    source_files_valid = all(_record_valid(record) for record in files.values())
    returns_store = np.memmap(
        Path(returns_record["path"]),
        dtype=np.dtype(returns_record["dtype"]),
        mode="r",
        shape=tuple(int(value) for value in returns_record["shape"]),
    )
    valid_store = np.asarray(
        np.memmap(
            Path(valid_record["path"]),
            dtype=np.dtype(valid_record["dtype"]),
            mode="r",
            shape=tuple(int(value) for value in valid_record["shape"]),
        ),
        dtype=bool,
    )
    flags_store = np.memmap(
        Path(flags_record["path"]),
        dtype=np.dtype(flags_record["dtype"]),
        mode="r",
        shape=tuple(int(value) for value in flags_record["shape"]),
    )
    within = _flag_mask(flags_store, FLAG_OUTCOME_WITHIN_CUTOFF)
    expected_within = context["date_idx"].astype(np.int64) + 2 <= int(
        context["cutoff_idx"]
    )
    target_flag = _flag_mask(flags_store, FLAG_TARGET_VALID)
    decomposition_flag = _flag_mask(flags_store, FLAG_DECOMPOSITION_VALID)
    entry_observed = _flag_mask(flags_store, FLAG_ENTRY_PRICE_OBSERVED)
    exit_observed = _flag_mask(flags_store, FLAG_EXIT_PRICE_OBSERVED)
    entry_known = _flag_mask(flags_store, FLAG_ENTRY_EXECUTION_STATE_KNOWN)
    exit_known = _flag_mask(flags_store, FLAG_EXIT_EXECUTION_STATE_KNOWN)
    entry_buyable = _flag_mask(flags_store, FLAG_ENTRY_BUYABLE)
    exit_sellable = _flag_mask(flags_store, FLAG_EXIT_OPEN_SELLABLE)
    primary_finite = np.isfinite(np.asarray(returns_store[:, 0], dtype=np.float64))
    component_finite = np.isfinite(
        np.asarray(returns_store[:, 1:], dtype=np.float64)
    ).all(axis=1)
    reconstructed = (
        1.0 + np.asarray(returns_store[decomposition_flag, 1], dtype=np.float64)
    ) * (1.0 + np.asarray(returns_store[decomposition_flag, 2], dtype=np.float64)) - 1.0
    identity_error = np.abs(
        reconstructed
        - np.asarray(returns_store[decomposition_flag, 0], dtype=np.float64)
    )
    fully_executable = (
        valid_store & entry_known & exit_known & entry_buyable & exit_sellable
    )
    target_without_entry = valid_store & entry_known & ~entry_buyable
    target_without_exit = valid_store & exit_known & ~exit_sellable
    coverage = pd.read_parquet(Path(preparation_files["label_coverage"]["path"]))
    baseline_record = manifest.get("frozen_g1_baseline")
    baseline_valid = True
    if baseline_record is not None:
        baseline_valid = _record_valid(dict(baseline_record))
        if baseline_valid:
            baseline_manifest = _read_json(Path(baseline_record["path"]))
            baseline_valid = (
                baseline_manifest.get("schema") == BASELINE_SCHEMA
                and baseline_manifest.get("status") == "completed"
                and str(baseline_manifest.get("experiment_fingerprint"))
                == expected_fingerprint
                and int(baseline_manifest.get("forbidden_2026_read_count", -1)) == 0
                and all(
                    _record_valid(record)
                    for record in dict(baseline_manifest.get("files", {})).values()
                )
            )
    checks = {
        "manifest_prepared": manifest.get("status") in {"prepared", "audited"},
        "experiment_fingerprint": str(manifest.get("experiment_fingerprint"))
        == expected_fingerprint,
        "label_contract_schema": contract.get("schema") == LABEL_SCHEMA
        and contract.get("status") == "completed",
        "label_files_valid": bool(source_files_valid),
        "row_count_exact": int(contract.get("row_count", -1)) == row_count,
        "label_columns_exact": tuple(contract.get("label_columns", ()))
        == LABEL_COLUMNS,
        "cutoff_support_exact": np.array_equal(within, expected_within),
        "target_valid_consistent": np.array_equal(valid_store, target_flag)
        and np.array_equal(valid_store, entry_observed & exit_observed),
        "primary_finite_exact": np.array_equal(primary_finite, valid_store),
        "decomposition_finite_exact": np.array_equal(
            component_finite, decomposition_flag
        ),
        "decomposition_identity": float(identity_error.max(initial=0.0)) <= 2.0e-6,
        "future_fill_not_label_gate": bool(target_without_entry.any())
        and bool(target_without_exit.any())
        and int(valid_store.sum()) > int(fully_executable.sum())
        and not bool(contract.get("future_fill_used_for_label_support", True)),
        "coverage_years_exact": tuple(coverage["signal_year"].astype(int)) == YEARS,
        "coverage_counts_reconcile": int(coverage["row_count"].sum()) == row_count
        and int(coverage["target_valid_count"].sum()) == int(valid_store.sum())
        and int(coverage["fully_executable_count"].sum())
        == int(fully_executable.sum()),
        "formal_rows_exclude_2010": not bool(
            context["row_index"]["trade_date"].astype(str).str.startswith("2010-").any()
        ),
        "forbidden_2026_rows": not bool(
            context["row_index"]["trade_date"].astype(str).str.startswith("2026-").any()
        ),
        "forbidden_2026_reads": int(contract.get("forbidden_2026_read_count", -1)) == 0
        and str(contract.get("maximum_outcome_date_read")) <= MAXIMUM_OUTCOME_DATE,
        "preparation_files_valid": all(
            _record_valid(record) for record in preparation_files.values()
        ),
        "frozen_baseline_valid_if_present": bool(baseline_valid),
        "no_training_performed": not bool(contract.get("training_performed", True))
        and not bool(manifest.get("training_performed", True)),
        "no_html_report": not any(output_root.rglob("*.html")),
    }
    payload = {
        "schema": AUDIT_SCHEMA,
        "status": "ok" if all(checks.values()) else "failed",
        "created_at": _now(),
        "study_id": STUDY_ID,
        "checks": checks,
        "row_count": row_count,
        "target_valid_count": int(valid_store.sum()),
        "fully_executable_count": int(fully_executable.sum()),
        "target_valid_without_entry_buyable_count": int(target_without_entry.sum()),
        "target_valid_without_exit_sellable_count": int(target_without_exit.sum()),
        "maximum_decomposition_identity_error": float(identity_error.max(initial=0.0)),
        "forbidden_2026_read_count": 0,
        "training_performed": False,
        "research_semantics": study["execution"]["research_semantics"],
    }
    audit_path = output_root / "audit.json"
    _write_json(audit_path, payload)
    if payload["status"] != "ok":
        failed = sorted(name for name, passed in checks.items() if not passed)
        raise T1TargetError(f"t1_target_audit_failed:{failed}")
    manifest["status"] = "audited"
    manifest["audit"] = _file_record(audit_path)
    manifest["updated_at"] = _now()
    _write_json(manifest_path, manifest)
    return payload


def status(*, output_root: Path = DEFAULT_OUTPUT_ROOT) -> dict[str, Any]:
    manifest_path = output_root / "manifest.json"
    audit_path = output_root / "audit.json"
    baseline_path = output_root / "frozen_g1_baseline/manifest.json"
    manifest = _read_json(manifest_path) if manifest_path.is_file() else {}
    audit_payload = _read_json(audit_path) if audit_path.is_file() else {}
    baseline = _read_json(baseline_path) if baseline_path.is_file() else {}
    return {
        "study_id": STUDY_ID,
        "status": manifest.get("status", "not_prepared"),
        "audit_status": audit_payload.get("status", "not_audited"),
        "baseline_status": baseline.get("status", "not_evaluated"),
        "retraining_decision": baseline.get("decision", {}).get("overall"),
        "training_performed": False,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-path", type=Path, default=DEFAULT_STUDY_PATH)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--model-output-root", type=Path, default=DEFAULT_MODEL_OUTPUT_ROOT
    )
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument("--prepare", action="store_true")
    actions.add_argument("--baseline", action="store_true")
    actions.add_argument("--audit", action="store_true")
    actions.add_argument("--status", action="store_true")
    parser.add_argument("--chunk-size", type=int, default=250_000)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.prepare:
        result = prepare(
            study_path=args.study_path,
            output_root=args.output_root,
            chunk_size=args.chunk_size,
        )
    elif args.baseline:
        result = evaluate_frozen_g1_baseline(
            study_path=args.study_path,
            output_root=args.output_root,
            model_output_root=args.model_output_root,
        )
    elif args.audit:
        result = audit(study_path=args.study_path, output_root=args.output_root)
    else:
        result = status(output_root=args.output_root)
    print(json.dumps(result, ensure_ascii=False, indent=2, default=_json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
