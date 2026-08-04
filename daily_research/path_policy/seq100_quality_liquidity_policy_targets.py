from __future__ import annotations

"""Train and replay policy-aligned TP8 plus D10/D20 timeout targets."""

import argparse
import gc
import json
import math
import os
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    log_loss,
    roc_auc_score,
)

from daily_research.path_policy import seq100_quality_liquidity_execution as execution
from daily_research.path_policy import seq100_quality_liquidity_model as canonical
from daily_research.path_policy import (
    seq100_quality_liquidity_profit_timeout as profit_timeout,
)
from daily_research.path_policy import seq100_v4_economic_realizability as economic

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
STUDY_ID = "seq100_quality_liquidity_policy_targets"
YEARS = tuple(range(2011, 2026))
ROLLING_YEARS = (2023, 2024, 2025)
HORIZONS = (10, 20)
FORBIDDEN_YEAR = 2026
MAXIMUM_OUTCOME_DATE = "2025-12-31"
TAKE_PROFIT = 0.08
FEATURE_VARIANT = "compact_core"
FEATURE_COUNT = 557

LABEL_COLUMNS = tuple(
    name
    for horizon in HORIZONS
    for name in (
        f"tp08_hit_d{horizon}",
        f"d{horizon}_timeout_return",
        f"d{horizon}_endpoint_return",
    )
)
TARGET_KIND = {
    **{f"tp08_hit_d{horizon}": "binary" for horizon in HORIZONS},
    **{f"d{horizon}_timeout_return": "timeout_return" for horizon in HORIZONS},
    **{f"d{horizon}_endpoint_return": "endpoint_return" for horizon in HORIZONS},
}
REPLAY_VARIANTS = (
    "tp_hit_probability",
    "dual_mfe_hit_maximin",
    "dual_mfe_timeout_maximin",
    "dual_mfe_hit_timeout_maximin",
)
SLOT_COUNTS = (6, 12)
COST_SCENARIOS = ("base", "stress")

OUTCOME_INVALID = 0
OUTCOME_TAKE_PROFIT = 1
OUTCOME_TIMEOUT = 2
FILL_NONE = 0
FILL_GAP_OPEN = 1
FILL_STANDING_LIMIT = 2
FILL_DELAYED_OPEN = 3
FILL_TIMEOUT_CLOSE = 4

LABEL_SCHEMA = "seq100_quality_liquidity_policy_targets_labels/1"
TASK_SCHEMA = "seq100_quality_liquidity_policy_targets_task/1"
REPLAY_TASK_SCHEMA = "seq100_quality_liquidity_policy_targets_replay_task/1"
MANIFEST_SCHEMA = "seq100_quality_liquidity_policy_targets_manifest/1"
AUDIT_SCHEMA = "seq100_quality_liquidity_policy_targets_audit/1"

DEFAULT_STUDY_PATH = (
    WORKSPACE_ROOT
    / "daily_research/studies/seq100_quality_liquidity_policy_targets.json"
)
DEFAULT_OUTPUT_ROOT = (
    WORKSPACE_ROOT
    / "daily_research/output/path_policy/studies/seq100_quality_liquidity_policy_targets"
)


class PolicyTargetError(RuntimeError):
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
        raise PolicyTargetError(f"required_json_missing:{path}")
    return dict(json.loads(path.read_text(encoding="utf-8")))


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    canonical._write_json(path, payload)


def _write_parquet(frame: pd.DataFrame, path: Path) -> None:
    canonical._write_parquet(frame, path)


def _file_record(path: Path, **extra: Any) -> dict[str, Any]:
    return canonical._file_record(path, **extra)


def _emit(event: str, **payload: Any) -> None:
    print(
        json.dumps(
            {"event": event, "at": _now(), **payload},
            ensure_ascii=False,
            default=economic._json_default,
        ),
        flush=True,
    )


def _load_config(path: Path = DEFAULT_STUDY_PATH) -> dict[str, Any]:
    config = _read_json(path)
    if config.get("study_id") != STUDY_ID:
        raise PolicyTargetError("study_id_mismatch")
    period = dict(config["period"])
    if tuple(int(value) for value in period["research_years"]) != YEARS:
        raise PolicyTargetError("research_year_contract_mismatch")
    if tuple(int(value) for value in period["rolling_prediction_years"]) != (
        ROLLING_YEARS
    ):
        raise PolicyTargetError("rolling_year_contract_mismatch")
    if str(period["maximum_outcome_date"]) != MAXIMUM_OUTCOME_DATE:
        raise PolicyTargetError("outcome_cutoff_contract_mismatch")
    if int(period["forbidden_year"]) != FORBIDDEN_YEAR:
        raise PolicyTargetError("forbidden_year_contract_mismatch")
    targets = dict(config["target_contract"])
    if tuple(int(value) for value in targets["timeout_days"]) != HORIZONS:
        raise PolicyTargetError("target_horizon_contract_mismatch")
    if not math.isclose(
        float(targets["take_profit"]), TAKE_PROFIT, rel_tol=0.0, abs_tol=1.0e-12
    ):
        raise PolicyTargetError("take_profit_contract_mismatch")
    if tuple(str(value) for value in targets["target_names"]) != LABEL_COLUMNS:
        raise PolicyTargetError("target_name_contract_mismatch")
    model = dict(config["model"])
    if (
        str(model["feature_variant"]) != FEATURE_VARIANT
        or int(model["feature_count"]) != FEATURE_COUNT
        or int(model["fixed_rounds"]) != 512
    ):
        raise PolicyTargetError("model_contract_mismatch")
    replay = dict(config["economic_replay"])
    if tuple(str(value) for value in replay["ranking_variants"]) != REPLAY_VARIANTS:
        raise PolicyTargetError("replay_variant_contract_mismatch")
    if tuple(int(value) for value in replay["timeout_days"]) != HORIZONS:
        raise PolicyTargetError("replay_horizon_contract_mismatch")
    if tuple(int(value) for value in replay["slot_counts"]) != SLOT_COUNTS:
        raise PolicyTargetError("replay_slot_contract_mismatch")
    if tuple(str(value) for value in replay["cost_scenarios"]) != COST_SCENARIOS:
        raise PolicyTargetError("replay_cost_contract_mismatch")
    outputs = dict(config["outputs"])
    if (
        int(outputs["expected_training_task_count"]) != 18
        or int(outputs["expected_new_replay_task_count"]) != 32
        or int(outputs["expected_baseline_reference_count"]) != 8
        or bool(outputs["html_report"])
    ):
        raise PolicyTargetError("output_contract_mismatch")
    return config


def _source_paths(config: Mapping[str, Any]) -> dict[str, Path]:
    return {str(name): _resolve(value) for name, value in config["sources"].items()}


def _horizon(target: str) -> int:
    for horizon in HORIZONS:
        if f"d{horizon}" in target:
            return horizon
    raise PolicyTargetError(f"target_horizon_unknown:{target}")


def _target_plan() -> list[dict[str, Any]]:
    tasks = [
        {
            "task_id": f"{target}__{FEATURE_VARIANT}__{year}",
            "target": target,
            "kind": TARGET_KIND[target],
            "horizon": _horizon(target),
            "year": year,
            "variant": FEATURE_VARIANT,
        }
        for target in LABEL_COLUMNS
        for year in ROLLING_YEARS
    ]
    if len(tasks) != 18 or len({task["task_id"] for task in tasks}) != 18:
        raise PolicyTargetError("training_task_inventory_changed")
    return tasks


@dataclass(frozen=True)
class BatchOutcome:
    tp_hit: np.ndarray
    tp_valid: np.ndarray
    timeout_return: np.ndarray
    timeout_valid: np.ndarray
    endpoint_return: np.ndarray
    endpoint_valid: np.ndarray
    outcome_code: np.ndarray
    trigger_day: np.ndarray
    fill_day: np.ndarray
    fill_phase: np.ndarray
    policy_return: np.ndarray


def _resolve_batch_outcome(
    *,
    signal_idx: int,
    symbols: np.ndarray,
    horizon: int,
    cutoff_idx: int,
    daily_raw: np.ndarray,
    entry_filled: np.ndarray,
    exit_sellable: np.ndarray,
    next_open_sellable_idx: np.ndarray,
) -> BatchOutcome:
    """Resolve independent candidates using the frozen portfolio exit semantics."""

    symbols = np.asarray(symbols, dtype=np.int32)
    count = len(symbols)
    tp_hit = np.zeros(count, dtype=np.float32)
    tp_valid = np.zeros(count, dtype=bool)
    timeout_return = np.full(count, np.nan, dtype=np.float32)
    timeout_valid = np.zeros(count, dtype=bool)
    endpoint_return = np.full(count, np.nan, dtype=np.float32)
    endpoint_valid = np.zeros(count, dtype=bool)
    outcome_code = np.zeros(count, dtype=np.int8)
    trigger_day = np.full(count, -1, dtype=np.int16)
    fill_day = np.full(count, -1, dtype=np.int16)
    fill_phase = np.zeros(count, dtype=np.int8)
    policy_return = np.full(count, np.nan, dtype=np.float32)
    endpoint_idx = int(signal_idx) + int(horizon)
    entry_idx = int(signal_idx) + 1
    if endpoint_idx > int(cutoff_idx) or not count:
        return BatchOutcome(
            tp_hit,
            tp_valid,
            timeout_return,
            timeout_valid,
            endpoint_return,
            endpoint_valid,
            outcome_code,
            trigger_day,
            fill_day,
            fill_phase,
            policy_return,
        )
    entry_open = np.asarray(daily_raw[entry_idx, symbols, 0], dtype=np.float64)
    entry_ok = (
        np.asarray(entry_filled[int(signal_idx), symbols], dtype=bool)
        & np.isfinite(entry_open)
        & (entry_open > 0.0)
    )
    endpoint_close = np.asarray(daily_raw[endpoint_idx, symbols, 3], dtype=np.float64)
    endpoint_ok = entry_ok & np.isfinite(endpoint_close) & (endpoint_close > 0.0)
    endpoint_return[endpoint_ok] = (
        endpoint_close[endpoint_ok] / entry_open[endpoint_ok] - 1.0
    ).astype(np.float32)
    endpoint_valid[endpoint_ok] = True
    active = entry_ok.copy()
    target_price = entry_open * (1.0 + TAKE_PROFIT)

    def resolve_blocked(indices: np.ndarray, *, reason: int, trigger: int) -> None:
        if not len(indices):
            return
        next_idx = np.asarray(
            next_open_sellable_idx[int(trigger) + 1, symbols[indices]],
            dtype=np.int32,
        )
        filled = (next_idx >= 0) & (next_idx <= int(cutoff_idx))
        if not bool(filled.any()):
            active[indices] = False
            return
        filled_indices = indices[filled]
        filled_dates = next_idx[filled]
        adjusted_open = np.asarray(
            daily_raw[filled_dates, symbols[filled_indices], 0], dtype=np.float64
        )
        price_ok = np.isfinite(adjusted_open) & (adjusted_open > 0.0)
        accepted = filled_indices[price_ok]
        accepted_dates = filled_dates[price_ok]
        prices = adjusted_open[price_ok]
        if len(accepted):
            outcome_code[accepted] = np.int8(reason)
            trigger_day[accepted] = np.int16(trigger - int(signal_idx))
            fill_day[accepted] = (accepted_dates - int(signal_idx)).astype(np.int16)
            fill_phase[accepted] = FILL_DELAYED_OPEN
            policy_return[accepted] = (prices / entry_open[accepted] - 1.0).astype(
                np.float32
            )
        active[indices] = False

    for day in range(2, int(horizon) + 1):
        if not bool(active.any()):
            break
        date_idx = int(signal_idx) + day
        indices = np.flatnonzero(active)
        ohlc = np.asarray(daily_raw[date_idx, symbols[indices], :4], dtype=np.float64)
        open_price = ohlc[:, 0]
        high_price = ohlc[:, 1]
        gap = np.isfinite(open_price) & (open_price >= target_price[indices])
        standing = (
            (~gap) & np.isfinite(high_price) & (high_price >= target_price[indices])
        )
        triggered = gap | standing
        if bool(triggered.any()):
            triggered_indices = indices[triggered]
            sellable = np.asarray(
                exit_sellable[date_idx, symbols[triggered_indices]], dtype=bool
            )
            direct = triggered_indices[sellable]
            if len(direct):
                direct_gap = np.isfinite(
                    np.asarray(
                        daily_raw[date_idx, symbols[direct], 0], dtype=np.float64
                    )
                ) & (
                    np.asarray(
                        daily_raw[date_idx, symbols[direct], 0], dtype=np.float64
                    )
                    >= target_price[direct]
                )
                prices = np.where(
                    direct_gap,
                    np.asarray(
                        daily_raw[date_idx, symbols[direct], 0], dtype=np.float64
                    ),
                    target_price[direct],
                )
                outcome_code[direct] = OUTCOME_TAKE_PROFIT
                trigger_day[direct] = day
                fill_day[direct] = day
                fill_phase[direct] = np.where(
                    direct_gap, FILL_GAP_OPEN, FILL_STANDING_LIMIT
                ).astype(np.int8)
                policy_return[direct] = (prices / entry_open[direct] - 1.0).astype(
                    np.float32
                )
                active[direct] = False
            blocked = triggered_indices[~sellable]
            resolve_blocked(
                blocked,
                reason=OUTCOME_TAKE_PROFIT,
                trigger=date_idx,
            )
        if day != int(horizon) or not bool(active.any()):
            continue
        timeout_indices = np.flatnonzero(active)
        close_price = np.asarray(
            daily_raw[date_idx, symbols[timeout_indices], 3], dtype=np.float64
        )
        sellable = np.asarray(
            exit_sellable[date_idx, symbols[timeout_indices]], dtype=bool
        )
        direct_mask = sellable & np.isfinite(close_price) & (close_price > 0.0)
        direct = timeout_indices[direct_mask]
        if len(direct):
            prices = close_price[direct_mask]
            outcome_code[direct] = OUTCOME_TIMEOUT
            trigger_day[direct] = day
            fill_day[direct] = day
            fill_phase[direct] = FILL_TIMEOUT_CLOSE
            policy_return[direct] = (prices / entry_open[direct] - 1.0).astype(
                np.float32
            )
            active[direct] = False
        resolve_blocked(
            timeout_indices[~direct_mask],
            reason=OUTCOME_TIMEOUT,
            trigger=date_idx,
        )

    resolved = np.isin(outcome_code, (OUTCOME_TAKE_PROFIT, OUTCOME_TIMEOUT))
    tp_valid[resolved] = True
    tp_hit[outcome_code == OUTCOME_TAKE_PROFIT] = 1.0
    timeout_mask = outcome_code == OUTCOME_TIMEOUT
    timeout_valid[timeout_mask] = True
    timeout_return[timeout_mask] = policy_return[timeout_mask]
    return BatchOutcome(
        tp_hit,
        tp_valid,
        timeout_return,
        timeout_valid,
        endpoint_return,
        endpoint_valid,
        outcome_code,
        trigger_day,
        fill_day,
        fill_phase,
        policy_return,
    )


def _next_open_sellable_index(
    *, pack: Mapping[str, Any], first_idx: int, cutoff_idx: int
) -> np.ndarray:
    symbol_count = int(pack["symbol_count"])
    execution_arrays = dict(pack["execution_arrays"])
    masks = dict(pack["masks"])
    raw_open = economic._open_array(
        execution_arrays["entry_open_raw"], dtype=np.float32
    )
    raw_down_limit = economic._open_array(
        execution_arrays["exit_down_limit_raw"], dtype=np.float32
    )
    status_valid = economic._open_array(masks["status_valid"], dtype=np.bool_)
    suspended = economic._open_array(masks["is_suspended"], dtype=np.bool_)
    delisted = economic._open_array(masks["is_delisted"], dtype=np.bool_)
    daily_raw = economic._open_array(
        pack["feature_channels"]["daily_raw"], dtype=np.float32
    )
    output = np.full((cutoff_idx + 2, symbol_count), -1, dtype=np.int32)
    next_idx = np.full(symbol_count, -1, dtype=np.int32)
    for date_idx in range(int(cutoff_idx), int(first_idx) - 1, -1):
        sellable = economic.derive_open_sellable(
            raw_open=np.asarray(raw_open[date_idx], dtype=np.float64),
            raw_down_limit=np.asarray(raw_down_limit[date_idx], dtype=np.float64),
            status_valid=np.asarray(status_valid[date_idx], dtype=bool),
            suspended=np.asarray(suspended[date_idx], dtype=bool),
            delisted=np.asarray(delisted[date_idx], dtype=bool),
        )
        adjusted_open = np.asarray(daily_raw[date_idx, :, 0], dtype=np.float64)
        sellable &= np.isfinite(adjusted_open) & (adjusted_open > 0.0)
        next_idx[sellable] = date_idx
        output[date_idx] = next_idx
    return output


def _labels_fingerprint(
    *,
    config_path: Path,
    model_input_path: Path,
    pack_path: Path,
) -> str:
    return canonical._stable_hash(
        {
            "study_sha256": canonical._sha256(config_path),
            "model_input_sha256": canonical._sha256(model_input_path),
            "pack_sha256": canonical._sha256(pack_path),
            "label_columns": LABEL_COLUMNS,
            "take_profit": TAKE_PROFIT,
            "maximum_outcome_date": MAXIMUM_OUTCOME_DATE,
        }
    )


def _records_valid(records: Mapping[str, Mapping[str, Any]]) -> bool:
    try:
        return all(
            Path(record["path"]).is_file()
            and canonical._sha256(Path(record["path"])) == str(record["sha256"])
            for record in records.values()
        )
    except (KeyError, OSError, TypeError, ValueError):
        return False


def _labels_complete(path: Path, *, fingerprint: str | None = None) -> bool:
    if not path.is_file():
        return False
    try:
        manifest = _read_json(path)
        return bool(
            manifest.get("schema") == LABEL_SCHEMA
            and manifest.get("status") == "completed"
            and (fingerprint is None or manifest.get("fingerprint") == fingerprint)
            and _records_valid(dict(manifest["files"]))
        )
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
        return False


def prepare(
    *,
    study_path: Path = DEFAULT_STUDY_PATH,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
) -> dict[str, Any]:
    config = _load_config(study_path)
    sources = _source_paths(config)
    model_input_path = sources["model_input_manifest"]
    model_manifest = _read_json(model_input_path)
    label_source = _read_json(Path(model_manifest["label_manifest"]["path"]))
    pack_path = Path(label_source["source"]["pack_manifest"])
    pack = _read_json(pack_path)
    fingerprint = _labels_fingerprint(
        config_path=study_path,
        model_input_path=model_input_path,
        pack_path=pack_path,
    )
    manifest_path = output_root / "labels/manifest.json"
    if _labels_complete(manifest_path, fingerprint=fingerprint):
        return _read_json(manifest_path)

    row_index = pd.read_parquet(Path(model_manifest["row_index"]["path"]))
    required = {"candidate_id", "date_idx", "trade_date", "symbol"}
    if not required.issubset(row_index.columns):
        raise PolicyTargetError("model_row_index_columns_missing")
    if (
        len(row_index) != int(model_manifest["row_count"])
        or row_index["candidate_id"].duplicated().any()
        or not row_index["date_idx"].is_monotonic_increasing
    ):
        raise PolicyTargetError("model_row_index_contract_failed")
    years = row_index["trade_date"].astype(str).str[:4].astype(int)
    if set(years.unique()) != set(YEARS) or bool((years == FORBIDDEN_YEAR).any()):
        raise PolicyTargetError("formal_row_year_contract_failed")
    date_values = np.asarray(pack["date_values"], dtype=str)
    cutoff_matches = np.flatnonzero(date_values == MAXIMUM_OUTCOME_DATE)
    if len(cutoff_matches) != 1:
        raise PolicyTargetError("outcome_cutoff_absent_or_duplicate")
    cutoff_idx = int(cutoff_matches[0])
    if bool(np.any(row_index["date_idx"].to_numpy(dtype=np.int32) > cutoff_idx)):
        raise PolicyTargetError("formal_row_after_outcome_cutoff")
    symbol_values = np.asarray(pack["symbol_values"], dtype=str)
    symbol_map = {symbol: idx for idx, symbol in enumerate(symbol_values)}
    symbol_idx = row_index["symbol"].astype(str).map(symbol_map)
    if symbol_idx.isna().any():
        raise PolicyTargetError("model_symbol_absent_from_pack")
    symbols = symbol_idx.to_numpy(dtype=np.int32)
    date_idx = row_index["date_idx"].to_numpy(dtype=np.int32)
    first_read_idx = int(date_idx.min()) + 1
    next_sellable = _next_open_sellable_index(
        pack=pack,
        first_idx=first_read_idx,
        cutoff_idx=cutoff_idx,
    )
    daily_raw = economic._open_array(
        pack["feature_channels"]["daily_raw"], dtype=np.float32
    )
    entry_filled = economic._open_array(pack["masks"]["entry_filled"], dtype=np.bool_)
    exit_sellable = economic._open_array(pack["masks"]["exit_sellable"], dtype=np.bool_)
    row_count = len(row_index)
    labels = np.full((row_count, len(LABEL_COLUMNS)), np.nan, dtype=np.float32)
    valid = np.zeros((row_count, len(LABEL_COLUMNS)), dtype=bool)
    diagnostics = {
        "outcome_code": np.zeros((row_count, len(HORIZONS)), dtype=np.int8),
        "trigger_day": np.full((row_count, len(HORIZONS)), -1, dtype=np.int16),
        "fill_day": np.full((row_count, len(HORIZONS)), -1, dtype=np.int16),
        "fill_phase": np.zeros((row_count, len(HORIZONS)), dtype=np.int8),
        "policy_return": np.full((row_count, len(HORIZONS)), np.nan, dtype=np.float32),
    }
    boundaries = np.flatnonzero(np.r_[True, date_idx[1:] != date_idx[:-1], True])
    started = time.monotonic()
    for group_number, (start, stop) in enumerate(pairwise(boundaries), start=1):
        current_idx = int(date_idx[start])
        current_symbols = np.asarray(symbols[start:stop], dtype=np.int32)
        if len(np.unique(current_symbols)) != len(current_symbols):
            raise PolicyTargetError(
                f"duplicate_candidate_symbol:{date_values[current_idx]}"
            )
        for horizon_position, horizon in enumerate(HORIZONS):
            outcome = _resolve_batch_outcome(
                signal_idx=current_idx,
                symbols=current_symbols,
                horizon=horizon,
                cutoff_idx=cutoff_idx,
                daily_raw=daily_raw,
                entry_filled=entry_filled,
                exit_sellable=exit_sellable,
                next_open_sellable_idx=next_sellable,
            )
            column = horizon_position * 3
            labels[start:stop, column] = outcome.tp_hit
            valid[start:stop, column] = outcome.tp_valid
            labels[start:stop, column + 1] = outcome.timeout_return
            valid[start:stop, column + 1] = outcome.timeout_valid
            labels[start:stop, column + 2] = outcome.endpoint_return
            valid[start:stop, column + 2] = outcome.endpoint_valid
            diagnostics["outcome_code"][start:stop, horizon_position] = (
                outcome.outcome_code
            )
            diagnostics["trigger_day"][start:stop, horizon_position] = (
                outcome.trigger_day
            )
            diagnostics["fill_day"][start:stop, horizon_position] = outcome.fill_day
            diagnostics["fill_phase"][start:stop, horizon_position] = outcome.fill_phase
            diagnostics["policy_return"][start:stop, horizon_position] = (
                outcome.policy_return
            )
        if group_number % 500 == 0:
            _emit(
                "policy_label_progress",
                completed_dates=group_number,
                total_dates=len(boundaries) - 1,
                elapsed_seconds=round(time.monotonic() - started, 2),
            )
    del next_sellable
    gc.collect()

    label_root = output_root / "labels"
    arrays: dict[str, np.ndarray] = {
        "labels": labels,
        "valid": valid,
        **diagnostics,
    }
    files: dict[str, dict[str, Any]] = {}
    for name, values in arrays.items():
        path = label_root / f"{name}.npy"
        canonical._save_npy(path, values)
        files[name] = _file_record(
            path, shape=list(values.shape), dtype=str(values.dtype)
        )
    summary_rows: list[dict[str, Any]] = []
    for year in YEARS:
        year_mask = years.to_numpy() == year
        for column, target in enumerate(LABEL_COLUMNS):
            target_valid = valid[year_mask, column]
            target_values = labels[year_mask, column][target_valid]
            summary_rows.append(
                {
                    "year": year,
                    "target": target,
                    "row_count": int(year_mask.sum()),
                    "valid_count": int(target_valid.sum()),
                    "valid_rate": float(target_valid.mean()),
                    "mean": (
                        float(np.mean(target_values))
                        if len(target_values)
                        else math.nan
                    ),
                    "median": (
                        float(np.median(target_values))
                        if len(target_values)
                        else math.nan
                    ),
                }
            )
    summary = pd.DataFrame(summary_rows)
    summary_path = label_root / "annual_summary.parquet"
    _write_parquet(summary, summary_path)
    files["annual_summary"] = _file_record(summary_path, row_count=len(summary))
    manifest = {
        "schema": LABEL_SCHEMA,
        "status": "completed",
        "completed_at": _now(),
        "study_id": STUDY_ID,
        "fingerprint": fingerprint,
        "row_count": row_count,
        "label_columns": list(LABEL_COLUMNS),
        "diagnostic_horizons": list(HORIZONS),
        "outcome_codes": {
            "invalid_or_unresolved": OUTCOME_INVALID,
            "take_profit": OUTCOME_TAKE_PROFIT,
            "timeout": OUTCOME_TIMEOUT,
        },
        "fill_phase_codes": {
            "none": FILL_NONE,
            "gap_open": FILL_GAP_OPEN,
            "standing_limit": FILL_STANDING_LIMIT,
            "delayed_open": FILL_DELAYED_OPEN,
            "timeout_close": FILL_TIMEOUT_CLOSE,
        },
        "contract": {
            "take_profit": TAKE_PROFIT,
            "horizons": list(HORIZONS),
            "entry_day": 1,
            "take_profit_first_active_day": 2,
            "blocked_exit": "next_sellable_open",
            "maximum_outcome_date": MAXIMUM_OUTCOME_DATE,
            "minimum_formal_year": 2011,
            "forbidden_year": FORBIDDEN_YEAR,
            "maximum_source_date_idx_read": cutoff_idx,
            "maximum_source_date_read": str(date_values[cutoff_idx]),
            "forbidden_2026_read_count": 0,
        },
        "sources": {
            "study": _file_record(study_path),
            "model_inputs": _file_record(
                model_input_path,
                input_fingerprint=str(model_manifest["input_fingerprint"]),
                feature_count=len(model_manifest["feature_groups"][FEATURE_VARIANT]),
            ),
            "row_index": _file_record(Path(model_manifest["row_index"]["path"])),
            "pack": _file_record(pack_path),
        },
        "files": files,
    }
    _write_json(manifest_path, manifest)
    return manifest


class PolicyInputs(canonical.ModelInputs):
    def __init__(
        self,
        model_manifest: Mapping[str, Any],
        label_manifest: Mapping[str, Any],
    ) -> None:
        super().__init__(model_manifest)
        if int(label_manifest["row_count"]) != self.row_count:
            raise PolicyTargetError("policy_label_row_count_mismatch")
        files = dict(label_manifest["files"])
        self.policy_labels = np.load(
            economic._verify_record(files["labels"]), mmap_mode="r", allow_pickle=False
        )
        self.policy_valid = np.load(
            economic._verify_record(files["valid"]), mmap_mode="r", allow_pickle=False
        )
        if self.policy_labels.shape != (self.row_count, len(LABEL_COLUMNS)):
            raise PolicyTargetError("policy_label_shape_mismatch")
        if self.policy_valid.shape != self.policy_labels.shape:
            raise PolicyTargetError("policy_valid_shape_mismatch")
        self.policy_column = {name: idx for idx, name in enumerate(LABEL_COLUMNS)}

    def task_values(self, target: str) -> np.ndarray:
        if target not in self.policy_column:
            return super().task_values(target)
        return np.asarray(self.policy_labels[:, self.policy_column[target]])

    def valid_mask(self, target: str) -> np.ndarray:
        if target not in self.policy_column:
            return super().valid_mask(target)
        column = self.policy_column[target]
        values = np.asarray(self.policy_labels[:, column])
        return np.asarray(self.policy_valid[:, column], dtype=bool) & np.isfinite(
            values
        )

    def policy_fold(self, *, year: int, target: str) -> dict[str, Any]:
        horizon = _horizon(target)
        year_rows = self.rows_for_year(year)
        oos_start = int(self.date_idx[year_rows[0]])
        oos_end = int(self.date_idx[year_rows[-1]])
        maximum_train_idx = oos_start - horizon - 1
        valid = self.valid_mask(target)
        train_rows = np.flatnonzero(
            (self.date_idx <= maximum_train_idx) & valid
        ).astype(np.int64, copy=False)
        evaluation_rows = year_rows[valid[year_rows]]
        if not len(train_rows) or not len(evaluation_rows):
            raise PolicyTargetError(f"empty_policy_fold:{target}:{year}")
        if int(self.date_idx[train_rows[-1]]) + horizon >= oos_start:
            raise PolicyTargetError("policy_horizon_purge_failed")
        if bool(np.any(self.years[train_rows] < 2011)):
            raise PolicyTargetError("burn_in_row_entered_training")
        return {
            "year": year,
            "target": target,
            "horizon": horizon,
            "oos_start_date_idx": oos_start,
            "oos_end_date_idx": oos_end,
            "maximum_train_signal_date_idx": maximum_train_idx,
            "train_rows": train_rows,
            "evaluation_rows": evaluation_rows,
        }


def _model_parameters(config: Mapping[str, Any], kind: str) -> dict[str, Any]:
    parameters = canonical._target_parameters(
        {"model": dict(config["model"])},
        "mfe",
    )
    if kind == "binary":
        parameters.update({"objective": "binary", "metric": "binary_logloss"})
        parameters.pop("alpha", None)
    elif kind not in {"timeout_return", "endpoint_return"}:
        raise PolicyTargetError(f"unknown_policy_target_kind:{kind}")
    return parameters


def _safe_mean(values: Sequence[float] | np.ndarray) -> float:
    array = np.asarray(values, dtype=np.float64)
    finite = array[np.isfinite(array)]
    return float(np.mean(finite)) if len(finite) else math.nan


def _safe_auc(actual: np.ndarray, score: np.ndarray) -> float:
    if len(np.unique(actual)) != 2:
        return math.nan
    return float(roc_auc_score(actual, score))


def _binary_daily_metrics(
    *,
    date_idx: np.ndarray,
    actual: np.ndarray,
    prediction: np.ndarray,
    fractions: Sequence[float],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    date_idx = np.asarray(date_idx, dtype=np.int32)
    actual = np.asarray(actual, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    probability = np.clip(prediction, 1.0e-7, 1 - 1.0e-7)
    boundaries = np.flatnonzero(np.r_[True, date_idx[1:] != date_idx[:-1], True])
    for start, stop in pairwise(boundaries):
        y = actual[start:stop]
        p = prediction[start:stop]
        probability_part = probability[start:stop]
        base_rate = float(np.mean(y))
        row: dict[str, Any] = {
            "date_idx": int(date_idx[start]),
            "row_count": len(y),
            "positive_count": int(np.sum(y)),
            "base_rate": base_rate,
            "roc_auc": _safe_auc(y, p),
            "pr_auc": (
                float(average_precision_score(y, p))
                if bool(np.any(y == 1.0))
                else math.nan
            ),
            "brier": float(np.mean((probability_part - y) ** 2)),
            "logloss": float(log_loss(y, probability_part, labels=[0.0, 1.0])),
        }
        ranked = np.lexsort((np.arange(len(p)), -p))
        for fraction in fractions:
            name = f"top{round(float(fraction) * 100):02d}"
            count = max(1, math.ceil(len(y) * float(fraction)))
            chosen = ranked[:count]
            hit_rate = float(np.mean(y[chosen]))
            row[f"{name}_hit_rate"] = hit_rate
            row[f"{name}_lift"] = (
                float(hit_rate / base_rate) if base_rate > 0.0 else math.nan
            )
            row[f"{name}_capture"] = (
                float(np.sum(y[chosen]) / np.sum(y)) if np.sum(y) > 0.0 else math.nan
            )
        rows.append(row)
    frame = pd.DataFrame(rows)
    metrics = {
        column: _safe_mean(frame[column].to_numpy(dtype=np.float64))
        for column in frame.columns
        if column not in {"date_idx", "row_count", "positive_count"}
    }
    metrics.update(
        {
            "evaluation_row_count": len(actual),
            "evaluation_date_count": len(frame),
            "positive_count": int(np.sum(actual)),
        }
    )
    return frame, metrics


def _regression_daily_metrics(
    *,
    date_idx: np.ndarray,
    actual: np.ndarray,
    prediction: np.ndarray,
    kind: str,
    config: Mapping[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    date_idx = np.asarray(date_idx, dtype=np.int32)
    actual = np.asarray(actual, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    evaluation = dict(config["evaluation"])
    boundaries = np.flatnonzero(np.r_[True, date_idx[1:] != date_idx[:-1], True])
    for start, stop in pairwise(boundaries):
        y = actual[start:stop]
        p = prediction[start:stop]
        row: dict[str, Any] = {
            "date_idx": int(date_idx[start]),
            "row_count": len(y),
            "rank_ic": canonical._safe_spearman(p, y),
            "mae": float(np.mean(np.abs(p - y))),
            "actual_mean": float(np.mean(y)),
            "actual_median": float(np.median(y)),
        }
        if kind == "timeout_return":
            threshold = float(evaluation["large_timeout_loss"])
            base_loss_rate = float(np.mean(y <= threshold))
            row["large_loss_base_rate"] = base_loss_rate
            ranked = np.lexsort((np.arange(len(p)), p))
            for fraction in evaluation["bottom_fractions"]:
                name = f"bottom{round(float(fraction) * 100):02d}"
                count = max(1, math.ceil(len(y) * float(fraction)))
                chosen = ranked[:count]
                loss_rate = float(np.mean(y[chosen] <= threshold))
                row[f"{name}_large_loss_rate"] = loss_rate
                row[f"{name}_large_loss_lift"] = (
                    float(loss_rate / base_loss_rate)
                    if base_loss_rate > 0.0
                    else math.nan
                )
                row[f"{name}_actual_mean"] = float(np.mean(y[chosen]))
        else:
            ranked = np.lexsort((np.arange(len(p)), -p))
            for fraction in evaluation["top_fractions"]:
                name = f"top{round(float(fraction) * 100):02d}"
                count = max(1, math.ceil(len(y) * float(fraction)))
                chosen = ranked[:count]
                row[f"{name}_actual_mean"] = float(np.mean(y[chosen]))
                row[f"{name}_actual_median"] = float(np.median(y[chosen]))
        rows.append(row)
    frame = pd.DataFrame(rows)
    metrics = {
        column: _safe_mean(frame[column].to_numpy(dtype=np.float64))
        for column in frame.columns
        if column not in {"date_idx", "row_count"}
    }
    metrics.update(
        {"evaluation_row_count": len(actual), "evaluation_date_count": len(frame)}
    )
    return frame, metrics


def _evaluate_target(
    *,
    inputs: PolicyInputs,
    task: Mapping[str, Any],
    rows: np.ndarray,
    prediction: np.ndarray,
    config: Mapping[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    target = str(task["target"])
    actual = np.asarray(inputs.task_values(target)[rows], dtype=np.float64)
    if str(task["kind"]) == "binary":
        return _binary_daily_metrics(
            date_idx=inputs.date_idx[rows],
            actual=actual,
            prediction=prediction,
            fractions=config["evaluation"]["top_fractions"],
        )
    return _regression_daily_metrics(
        date_idx=inputs.date_idx[rows],
        actual=actual,
        prediction=prediction,
        kind=str(task["kind"]),
        config=config,
    )


def _task_result_path(output_root: Path, task_id: str) -> Path:
    return output_root / "tasks" / task_id / "task_result.json"


def _task_fingerprint(
    *,
    task: Mapping[str, Any],
    config_sha256: str,
    input_fingerprint: str,
    label_fingerprint: str,
    feature_names: Sequence[str],
) -> str:
    return canonical._stable_hash(
        {
            "task": dict(task),
            "config_sha256": config_sha256,
            "input_fingerprint": input_fingerprint,
            "label_fingerprint": label_fingerprint,
            "feature_names": list(feature_names),
        }
    )


def _task_complete(path: Path, *, fingerprint: str | None = None) -> bool:
    if not path.is_file():
        return False
    try:
        result = _read_json(path)
        return bool(
            result.get("schema") == TASK_SCHEMA
            and result.get("status") == "completed"
            and (fingerprint is None or result.get("task_fingerprint") == fingerprint)
            and _records_valid(dict(result["files"]))
        )
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
        return False


def _load_inputs(
    *, config: Mapping[str, Any], output_root: Path
) -> tuple[PolicyInputs, dict[str, Any], dict[str, Any]]:
    sources = _source_paths(config)
    model_manifest = _read_json(sources["model_input_manifest"])
    label_path = output_root / "labels/manifest.json"
    if not _labels_complete(label_path):
        raise PolicyTargetError("policy_labels_not_prepared")
    label_manifest = _read_json(label_path)
    inputs = PolicyInputs(model_manifest, label_manifest)
    feature_names = inputs.feature_groups.get(FEATURE_VARIANT, [])
    if len(feature_names) != FEATURE_COUNT or len(set(feature_names)) != FEATURE_COUNT:
        raise PolicyTargetError("compact_feature_contract_changed")
    return inputs, model_manifest, label_manifest


def _run_training_task(
    *,
    task: Mapping[str, Any],
    inputs: PolicyInputs,
    config: Mapping[str, Any],
    output_root: Path,
    config_sha256: str,
    label_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    import lightgbm as lgb

    feature_names = list(inputs.feature_groups[FEATURE_VARIANT])
    fingerprint = _task_fingerprint(
        task=task,
        config_sha256=config_sha256,
        input_fingerprint=str(inputs.manifest["input_fingerprint"]),
        label_fingerprint=str(label_manifest["fingerprint"]),
        feature_names=feature_names,
    )
    result_path = _task_result_path(output_root, str(task["task_id"]))
    if _task_complete(result_path, fingerprint=fingerprint):
        return _read_json(result_path)
    fold = inputs.policy_fold(year=int(task["year"]), target=str(task["target"]))
    datasets = canonical._build_datasets(
        inputs=inputs,
        config=config,
        target=str(task["target"]),
        feature_names=feature_names,
        train_rows=fold["train_rows"],
        evaluation_rows=fold["evaluation_rows"],
    )
    parameters = _model_parameters(config, str(task["kind"]))
    iterations = int(config["model"]["fixed_rounds"])
    started = time.perf_counter()
    booster = lgb.train(
        parameters,
        datasets["train_set"],
        num_boost_round=iterations,
        valid_sets=[datasets["evaluation_set"]],
        valid_names=["outer_evaluation"],
    )
    training_seconds = float(time.perf_counter() - started)
    prediction_rows = inputs.rows_for_year(int(task["year"]))
    sequence = canonical._make_sequence(
        inputs=inputs,
        rows=prediction_rows,
        layout=datasets["layout"],
        batch_size=int(config["model"]["sequence_batch_size"]),
    )
    prediction = canonical._predict(booster, sequence, iterations)
    positions = np.searchsorted(prediction_rows, fold["evaluation_rows"])
    if not np.array_equal(
        prediction_rows[positions], np.asarray(fold["evaluation_rows"])
    ):
        raise PolicyTargetError("policy_evaluation_prediction_alignment_failed")
    daily, metrics = _evaluate_target(
        inputs=inputs,
        task=task,
        rows=fold["evaluation_rows"],
        prediction=np.asarray(prediction[positions]),
        config=config,
    )
    daily.insert(
        1,
        "trade_date",
        [str(inputs.date_values[int(value)]) for value in daily["date_idx"]],
    )
    task_dir = result_path.parent
    task_dir.mkdir(parents=True, exist_ok=True)
    model_path = task_dir / "model.txt"
    model_partial = model_path.with_suffix(".txt.partial")
    booster.save_model(str(model_partial), num_iteration=iterations)
    os.replace(model_partial, model_path)
    prediction_path = task_dir / "prediction.npy"
    candidate_path = task_dir / "candidate_id.npy"
    daily_path = task_dir / "daily_metrics.parquet"
    importance_path = task_dir / "family_importance.parquet"
    canonical._save_npy(prediction_path, prediction)
    canonical._save_npy(candidate_path, inputs.candidate_ids[prediction_rows])
    _write_parquet(daily, daily_path)
    importance = canonical._importance_frame(
        model=booster,
        inputs=inputs,
        feature_names=feature_names,
        task=task,
    )
    _write_parquet(importance, importance_path)
    result = {
        "schema": TASK_SCHEMA,
        "status": "completed",
        "completed_at": _now(),
        "study_id": STUDY_ID,
        "task_id": str(task["task_id"]),
        "task_fingerprint": fingerprint,
        "target": str(task["target"]),
        "kind": str(task["kind"]),
        "horizon": int(task["horizon"]),
        "evaluation_year": int(task["year"]),
        "variant": FEATURE_VARIANT,
        "feature_count": len(feature_names),
        "feature_names": feature_names,
        "train_row_count": len(fold["train_rows"]),
        "evaluation_row_count": len(fold["evaluation_rows"]),
        "candidate_prediction_row_count": len(prediction_rows),
        "maximum_train_signal_date_idx": int(fold["maximum_train_signal_date_idx"]),
        "maximum_train_signal_date": str(
            inputs.date_values[int(fold["maximum_train_signal_date_idx"])]
        ),
        "purge_days": int(task["horizon"]),
        "iterations": iterations,
        "training_seconds": training_seconds,
        "parameters": parameters,
        "metrics": metrics,
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
    canonical._release_datasets(datasets)
    del booster, prediction, sequence
    gc.collect()
    return result


def _completed_training_results(output_root: Path) -> dict[str, dict[str, Any]]:
    results: dict[str, dict[str, Any]] = {}
    for task in _target_plan():
        path = _task_result_path(output_root, str(task["task_id"]))
        if _task_complete(path):
            results[str(task["task_id"])] = _read_json(path)
    return results


def _write_training_ledger(
    *, output_root: Path, tasks: Sequence[Mapping[str, Any]]
) -> None:
    rows = []
    for task in tasks:
        path = _task_result_path(output_root, str(task["task_id"]))
        complete = _task_complete(path)
        rows.append(
            {
                **dict(task),
                "status": "completed" if complete else "pending",
                "result_path": str(path.resolve()),
            }
        )
    _write_parquet(pd.DataFrame(rows), output_root / "training_ledger.parquet")


def run(
    *,
    study_path: Path = DEFAULT_STUDY_PATH,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    max_tasks: int | None = None,
) -> dict[str, Any]:
    config = _load_config(study_path)
    if not _labels_complete(output_root / "labels/manifest.json"):
        prepare(study_path=study_path, output_root=output_root)
    inputs, _, label_manifest = _load_inputs(config=config, output_root=output_root)
    tasks = _target_plan()
    config_sha256 = canonical._sha256(study_path)
    completed = 0
    executed = 0
    started = time.monotonic()
    for task in tasks:
        feature_names = list(inputs.feature_groups[FEATURE_VARIANT])
        fingerprint = _task_fingerprint(
            task=task,
            config_sha256=config_sha256,
            input_fingerprint=str(inputs.manifest["input_fingerprint"]),
            label_fingerprint=str(label_manifest["fingerprint"]),
            feature_names=feature_names,
        )
        path = _task_result_path(output_root, str(task["task_id"]))
        if _task_complete(path, fingerprint=fingerprint):
            completed += 1
            continue
        if max_tasks is not None and executed >= int(max_tasks):
            break
        _run_training_task(
            task=task,
            inputs=inputs,
            config=config,
            output_root=output_root,
            config_sha256=config_sha256,
            label_manifest=label_manifest,
        )
        completed += 1
        executed += 1
        _write_training_ledger(output_root=output_root, tasks=tasks)
        _emit(
            "policy_training_progress",
            completed=completed,
            total=len(tasks),
            task_id=str(task["task_id"]),
            elapsed_seconds=round(time.monotonic() - started, 2),
        )
    _write_training_ledger(output_root=output_root, tasks=tasks)
    return {
        "status": "completed" if completed == len(tasks) else "in_progress",
        "completed": completed,
        "total": len(tasks),
        "executed_this_run": executed,
    }


def _load_task_arrays(
    result: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    candidate_ids = np.load(
        economic._verify_record(result["files"]["candidate_id"]),
        mmap_mode="r",
        allow_pickle=False,
    )
    prediction = np.load(
        economic._verify_record(result["files"]["prediction"]),
        mmap_mode="r",
        allow_pickle=False,
    )
    return candidate_ids, prediction


def _daily_dual_mfe_score(
    *,
    date_idx: np.ndarray,
    mfe10: np.ndarray,
    mfe20: np.ndarray,
) -> np.ndarray:
    output = np.empty(len(date_idx), dtype=np.float32)
    boundaries = np.flatnonzero(np.r_[True, date_idx[1:] != date_idx[:-1], True])
    for start, stop in pairwise(boundaries):
        rank10 = economic._rank01(np.asarray(mfe10[start:stop], dtype=np.float64))
        rank20 = economic._rank01(np.asarray(mfe20[start:stop], dtype=np.float64))
        output[start:stop] = economic._rank01(np.minimum(rank10, rank20))
    return output


def _prediction_comparison(
    *,
    inputs: PolicyInputs,
    results: Mapping[str, Mapping[str, Any]],
    config: Mapping[str, Any],
) -> pd.DataFrame:
    canonical_root = _source_paths(config)["canonical_model_root"]
    rows: list[dict[str, Any]] = []
    for year in ROLLING_YEARS:
        year_rows = inputs.rows_for_year(year)
        expected_ids = inputs.candidate_ids[year_rows]
        _, mfe10_ids, mfe10 = execution._load_prediction_task(
            model_root=canonical_root, target="mfe_10", year=year
        )
        _, mfe20_ids, mfe20 = execution._load_prediction_task(
            model_root=canonical_root, target="mfe_20", year=year
        )
        if not (
            np.array_equal(np.asarray(mfe10_ids), expected_ids)
            and np.array_equal(np.asarray(mfe20_ids), expected_ids)
        ):
            raise PolicyTargetError(f"canonical_prediction_alignment_failed:{year}")
        dual = _daily_dual_mfe_score(
            date_idx=inputs.date_idx[year_rows],
            mfe10=np.asarray(mfe10),
            mfe20=np.asarray(mfe20),
        )
        for horizon in HORIZONS:
            target = f"tp08_hit_d{horizon}"
            result = results[f"{target}__{FEATURE_VARIANT}__{year}"]
            policy_ids, policy_prediction = _load_task_arrays(result)
            if not np.array_equal(np.asarray(policy_ids), expected_ids):
                raise PolicyTargetError(
                    f"policy_prediction_alignment_failed:{target}:{year}"
                )
            valid = inputs.valid_mask(target)[year_rows]
            actual = inputs.task_values(target)[year_rows][valid]
            date_values = inputs.date_idx[year_rows][valid]
            scores = {
                "tp_hit_probability": np.asarray(policy_prediction)[valid],
                f"mfe_{horizon}": (
                    np.asarray(mfe10)[valid]
                    if horizon == 10
                    else np.asarray(mfe20)[valid]
                ),
                "dual_mfe": dual[valid],
            }
            for score_name, score in scores.items():
                _, metrics = _binary_daily_metrics(
                    date_idx=date_values,
                    actual=actual,
                    prediction=score,
                    fractions=config["evaluation"]["top_fractions"],
                )
                rows.append(
                    {
                        "year": year,
                        "horizon": horizon,
                        "score": score_name,
                        **metrics,
                    }
                )
    return pd.DataFrame(rows)


def _annual_training_metrics(
    results: Mapping[str, Mapping[str, Any]],
) -> pd.DataFrame:
    rows = []
    for result in results.values():
        row = {
            "task_id": str(result["task_id"]),
            "target": str(result["target"]),
            "kind": str(result["kind"]),
            "horizon": int(result["horizon"]),
            "year": int(result["evaluation_year"]),
            "train_row_count": int(result["train_row_count"]),
            "evaluation_row_count": int(result["evaluation_row_count"]),
            "training_seconds": float(result["training_seconds"]),
        }
        row.update(dict(result["metrics"]))
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["horizon", "target", "year"])


def _importance_summary(
    results: Mapping[str, Mapping[str, Any]],
) -> pd.DataFrame:
    frames = [
        pd.read_parquet(economic._verify_record(result["files"]["family_importance"]))
        for result in results.values()
    ]
    combined = pd.concat(frames, ignore_index=True)
    return (
        combined.groupby(["target", "year", "feature_family"], as_index=False)[
            ["gain", "split", "feature_count"]
        ]
        .sum()
        .sort_values(["target", "year", "gain"], ascending=[True, True, False])
    )


def _root_manifest(
    *,
    study_path: Path,
    output_root: Path,
    config: Mapping[str, Any],
    status: str,
    files: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    sources = _source_paths(config)
    labels = _read_json(output_root / "labels/manifest.json")
    payload: dict[str, Any] = {
        "schema": MANIFEST_SCHEMA,
        "status": status,
        "created_at": _now(),
        "study_id": STUDY_ID,
        "study_config": _file_record(study_path),
        "model_inputs": _file_record(sources["model_input_manifest"]),
        "labels": _file_record(
            output_root / "labels/manifest.json",
            fingerprint=str(labels["fingerprint"]),
        ),
        "feature_variant": FEATURE_VARIANT,
        "feature_count": FEATURE_COUNT,
        "training_task_count": 18,
        "rolling_years": list(ROLLING_YEARS),
        "horizons": list(HORIZONS),
        "forbidden_2026_read_count": 0,
        "research_semantics": str(config["period"]["research_semantics"]),
        "training_performed": True,
        "hyperparameter_search_performed": False,
        "html_report": False,
        "files": dict(files),
    }
    existing = output_root / "manifest.json"
    if existing.is_file():
        old = _read_json(existing)
        if "replay" in old:
            payload["replay"] = old["replay"]
    return payload


def evaluate(
    *,
    study_path: Path = DEFAULT_STUDY_PATH,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
) -> dict[str, Any]:
    config = _load_config(study_path)
    inputs, _, _ = _load_inputs(config=config, output_root=output_root)
    results = _completed_training_results(output_root)
    if len(results) != 18:
        raise PolicyTargetError(f"policy_training_tasks_incomplete:{len(results)}/18")
    annual = _annual_training_metrics(results)
    comparison = _prediction_comparison(
        inputs=inputs,
        results=results,
        config=config,
    )
    importance = _importance_summary(results)
    evaluation_root = output_root / "evaluation"
    annual_path = evaluation_root / "annual_metrics.parquet"
    comparison_path = evaluation_root / "prediction_comparison.parquet"
    importance_path = evaluation_root / "family_importance.parquet"
    _write_parquet(annual, annual_path)
    _write_parquet(comparison, comparison_path)
    _write_parquet(importance, importance_path)
    files = {
        "annual_metrics": _file_record(annual_path, row_count=len(annual)),
        "prediction_comparison": _file_record(
            comparison_path, row_count=len(comparison)
        ),
        "family_importance": _file_record(importance_path, row_count=len(importance)),
        "training_ledger": _file_record(
            output_root / "training_ledger.parquet", row_count=18
        ),
    }
    manifest = _root_manifest(
        study_path=study_path,
        output_root=output_root,
        config=config,
        status="evaluated",
        files=files,
    )
    _write_json(output_root / "manifest.json", manifest)
    return {
        "status": "evaluated",
        "training_task_count": len(results),
        "comparison_row_count": len(comparison),
    }


@dataclass(frozen=True)
class PolicyReplaySpec:
    ranking_variant: str
    slot_count: int
    timeout_day: int
    take_profit: float
    cost_scenario: str

    @property
    def policy_name(self) -> str:
        return f"tp08_d{self.timeout_day:02d}"

    @property
    def task_id(self) -> str:
        return (
            f"{self.ranking_variant}__{self.policy_name}"
            f"__k{self.slot_count:02d}__{self.cost_scenario}"
        )

    def validate(self) -> None:
        if self.ranking_variant not in REPLAY_VARIANTS:
            raise ValueError(f"invalid ranking variant: {self.ranking_variant}")
        if self.slot_count not in SLOT_COUNTS:
            raise ValueError(f"invalid slot count: {self.slot_count}")
        if self.timeout_day not in HORIZONS:
            raise ValueError(f"invalid timeout day: {self.timeout_day}")
        if not math.isclose(
            self.take_profit, TAKE_PROFIT, rel_tol=0.0, abs_tol=1.0e-12
        ):
            raise ValueError(f"invalid take profit: {self.take_profit}")
        if self.cost_scenario not in COST_SCENARIOS:
            raise ValueError(f"invalid cost scenario: {self.cost_scenario}")


def _replay_specs() -> list[PolicyReplaySpec]:
    specs = [
        PolicyReplaySpec(
            ranking_variant=variant,
            slot_count=slots,
            timeout_day=horizon,
            take_profit=TAKE_PROFIT,
            cost_scenario=cost,
        )
        for horizon in HORIZONS
        for variant in REPLAY_VARIANTS
        for slots in SLOT_COUNTS
        for cost in COST_SCENARIOS
    ]
    if len(specs) != 32 or len({spec.task_id for spec in specs}) != 32:
        raise PolicyTargetError("replay_task_inventory_changed")
    return specs


class PolicySignalBook:
    def __init__(self, base_book: execution.SignalBook, orders: np.ndarray) -> None:
        self._base = base_book
        self._orders = np.asarray(orders, dtype=np.int32)
        expected = (base_book.day_count, base_book.symbol_count)
        if self._orders.shape != expected:
            raise PolicyTargetError("policy_order_shape_mismatch")

    def __getattr__(self, name: str) -> Any:
        return getattr(self._base, name)

    def symbols_for_day(self, family: str, day: int) -> np.ndarray:
        del family
        count = int(self._base.candidate_counts[int(day)])
        return np.asarray(self._orders[int(day), :count], dtype=np.int32)


def _order_path(output_root: Path, variant: str, horizon: int) -> Path:
    return output_root / "replay/orders" / f"{variant}__d{horizon:02d}.npy"


def _build_replay_orders(
    *,
    inputs: PolicyInputs,
    results: Mapping[str, Mapping[str, Any]],
    base_book: execution.SignalBook,
    output_root: Path,
) -> dict[tuple[str, int], dict[str, Any]]:
    symbol_map = {
        str(symbol): idx for idx, symbol in enumerate(base_book.symbol_values)
    }
    date_to_day = {
        int(value): day for day, value in enumerate(base_book.signal_date_idx)
    }
    records: dict[tuple[str, int], dict[str, Any]] = {}
    for horizon in HORIZONS:
        orders = {
            variant: np.full(
                (base_book.day_count, base_book.symbol_count), -1, dtype=np.int32
            )
            for variant in REPLAY_VARIANTS
        }
        for year in ROLLING_YEARS:
            year_rows = inputs.rows_for_year(year)
            expected_ids = inputs.candidate_ids[year_rows]
            hit_result = results[f"tp08_hit_d{horizon}__{FEATURE_VARIANT}__{year}"]
            timeout_result = results[
                f"d{horizon}_timeout_return__{FEATURE_VARIANT}__{year}"
            ]
            hit_ids, hit_prediction = _load_task_arrays(hit_result)
            timeout_ids, timeout_prediction = _load_task_arrays(timeout_result)
            if not (
                np.array_equal(np.asarray(hit_ids), expected_ids)
                and np.array_equal(np.asarray(timeout_ids), expected_ids)
            ):
                raise PolicyTargetError(f"replay_prediction_alignment_failed:{year}")
            current_dates = inputs.date_idx[year_rows]
            current_symbols = (
                inputs.row_index.iloc[year_rows]["symbol"]
                .astype(str)
                .map(symbol_map)
                .to_numpy(dtype=np.int32)
            )
            boundaries = np.flatnonzero(
                np.r_[True, current_dates[1:] != current_dates[:-1], True]
            )
            for start, stop in pairwise(boundaries):
                date_idx = int(current_dates[start])
                if date_idx not in date_to_day:
                    raise PolicyTargetError(f"replay_signal_date_missing:{date_idx}")
                day = date_to_day[date_idx]
                symbols = np.asarray(current_symbols[start:stop], dtype=np.int32)
                expected_count = int(base_book.candidate_counts[day])
                base_symbols = np.asarray(
                    base_book.orders[2, day, :expected_count], dtype=np.int32
                )
                if len(symbols) != expected_count or not np.array_equal(
                    np.sort(symbols), np.sort(base_symbols)
                ):
                    raise PolicyTargetError(
                        f"replay_candidate_set_changed:{base_book.date_text(day)}"
                    )
                hit_rank = economic._rank01(
                    np.asarray(hit_prediction[start:stop], dtype=np.float64)
                )
                timeout_rank = economic._rank01(
                    np.asarray(timeout_prediction[start:stop], dtype=np.float64)
                )
                dual_rank = np.asarray(
                    base_book.rank_panel[day, symbols, 7], dtype=np.float64
                )
                if not (
                    np.isfinite(hit_rank).all()
                    and np.isfinite(timeout_rank).all()
                    and np.isfinite(dual_rank).all()
                ):
                    raise PolicyTargetError("replay_score_contains_nonfinite")
                scores = {
                    "tp_hit_probability": hit_rank,
                    "dual_mfe_hit_maximin": economic._rank01(
                        np.minimum(dual_rank, hit_rank)
                    ),
                    "dual_mfe_timeout_maximin": economic._rank01(
                        np.minimum(dual_rank, timeout_rank)
                    ),
                    "dual_mfe_hit_timeout_maximin": economic._rank01(
                        np.minimum.reduce((dual_rank, hit_rank, timeout_rank))
                    ),
                }
                for variant, score in scores.items():
                    ranked = np.lexsort((symbols, -np.asarray(score, dtype=np.float64)))
                    orders[variant][day, : len(symbols)] = symbols[ranked]
        for variant, values in orders.items():
            if bool(
                np.any(
                    values[
                        np.arange(base_book.day_count),
                        np.asarray(base_book.candidate_counts, dtype=np.int32) - 1,
                    ]
                    < 0
                )
            ):
                raise PolicyTargetError(f"replay_order_incomplete:{variant}:d{horizon}")
            path = _order_path(output_root, variant, horizon)
            canonical._save_npy(path, values)
            records[(variant, horizon)] = _file_record(
                path, shape=list(values.shape), dtype=str(values.dtype)
            )
    return records


def _replay_task_path(output_root: Path, task_id: str) -> Path:
    return output_root / "replay/tasks" / task_id / "task_result.json"


def _replay_task_complete(path: Path, *, study_sha256: str, order_sha256: str) -> bool:
    if not path.is_file():
        return False
    try:
        result = _read_json(path)
        return bool(
            result.get("schema") == REPLAY_TASK_SCHEMA
            and result.get("status") == "completed"
            and result.get("study_config_sha256") == study_sha256
            and result.get("policy_order_sha256") == order_sha256
            and _records_valid(dict(result["files"]))
        )
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
        return False


def _unit_time_metrics(trades: pd.DataFrame, *, slot_count: int) -> dict[str, Any]:
    completed = trades[
        trades["side"].astype(str).eq("sell")
        & trades["status"].astype(str).eq("filled")
    ].copy()
    if completed.empty:
        return {
            "completed_trade_count": 0,
            "completed_trades_per_slot_year": 0.0,
            "mean_net_return_per_holding_day": math.nan,
            "mean_log_return_per_holding_day": math.nan,
            "median_net_return_per_holding_day": math.nan,
        }
    returns = pd.to_numeric(completed["net_return"], errors="coerce").to_numpy(
        dtype=np.float64
    )
    holding = pd.to_numeric(completed["holding_days"], errors="coerce").to_numpy(
        dtype=np.float64
    )
    valid = np.isfinite(returns) & np.isfinite(holding) & (holding > 0.0)
    arithmetic = returns[valid] / holding[valid]
    log_valid = valid & (returns > -1.0)
    log_daily = np.log1p(returns[log_valid]) / holding[log_valid]
    return {
        "completed_trade_count": int(valid.sum()),
        "completed_trades_per_slot_year": float(valid.sum() / (slot_count * 3.0)),
        "mean_net_return_per_holding_day": _safe_mean(arithmetic),
        "mean_log_return_per_holding_day": _safe_mean(log_daily),
        "median_net_return_per_holding_day": (
            float(np.median(arithmetic)) if len(arithmetic) else math.nan
        ),
    }


def _write_replay_task(
    *,
    output_root: Path,
    spec: PolicyReplaySpec,
    result: dict[str, Any],
    equity: pd.DataFrame,
    trades: pd.DataFrame,
    monthly: pd.DataFrame,
    study_sha256: str,
    order_record: Mapping[str, Any],
) -> dict[str, Any]:
    directory = _replay_task_path(output_root, spec.task_id).parent
    equity_path = directory / "equity.parquet"
    trades_path = directory / "trades.parquet"
    monthly_path = directory / "monthly.parquet"
    _write_parquet(equity, equity_path)
    _write_parquet(trades, trades_path)
    _write_parquet(monthly, monthly_path)
    result["metrics"]["unit_time"] = _unit_time_metrics(
        trades, slot_count=spec.slot_count
    )
    payload = {
        **result,
        "schema": REPLAY_TASK_SCHEMA,
        "study_id": STUDY_ID,
        "study_config_sha256": study_sha256,
        "policy_order_sha256": str(order_record["sha256"]),
        "ranking_variant": spec.ranking_variant,
        "files": {
            "equity": _file_record(equity_path, row_count=len(equity)),
            "trades": _file_record(trades_path, row_count=len(trades)),
            "monthly": _file_record(monthly_path, row_count=len(monthly)),
        },
    }
    _write_json(_replay_task_path(output_root, spec.task_id), payload)
    return payload


def _baseline_task_path(
    *, source_root: Path, horizon: int, slots: int, cost: str
) -> Path:
    task_id = f"dual_mfe_risk_veto__tp08_d{horizon:02d}__k{slots:02d}__{cost}"
    return source_root / "tasks" / task_id / "task_result.json"


def _replay_metric_row(
    *,
    result: Mapping[str, Any],
    ranking_variant: str,
    horizon: int,
    slots: int,
    cost: str,
    unit_time: Mapping[str, Any],
) -> dict[str, Any]:
    metrics = dict(result["metrics"])
    net = dict(metrics["net"])
    gross = dict(metrics["gross_same_trade_sequence"])
    total_cost = float(dict(metrics["costs"])["total"])
    terminal_return = float(metrics["terminal_cost_accrued_return"])
    gross_terminal_return = float(gross["cumulative_return"])
    return {
        "ranking_variant": ranking_variant,
        "horizon": horizon,
        "slot_count": slots,
        "cost_scenario": cost,
        "terminal_return": terminal_return,
        "gross_terminal_return": gross_terminal_return,
        "terminal_cost_drag": gross_terminal_return - terminal_return,
        "terminal_excess": float(
            metrics["terminal_cost_accrued_relative_excess_return"]
        ),
        "cagr": float(net["cagr"]),
        "gross_cagr": float(gross["cagr"]),
        "sharpe": float(net["sharpe"]),
        "maximum_drawdown": float(net["maximum_drawdown"]),
        "turnover_to_starting_cash": float(metrics["turnover_to_starting_cash"]),
        "total_cost_cny": total_cost,
        "total_cost_to_starting_cash": total_cost / 1_000_000.0,
        "filled_buy_count": int(metrics["filled_buy_count"]),
        "take_profit_hit_rate": float(metrics["take_profit_hit_rate"]),
        "timeout_gross_return_mean": float(metrics["timeout_gross_return_mean"]),
        "holding_days_mean": float(metrics["holding_days_mean"]),
        "holding_days_median": float(metrics["holding_days_median"]),
        **dict(unit_time),
    }


def _collect_replay_evaluation(
    *,
    config: Mapping[str, Any],
    output_root: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    source_root = _source_paths(config)["profit_timeout_output"]
    metric_rows: list[dict[str, Any]] = []
    annual_rows: list[dict[str, Any]] = []
    for horizon in HORIZONS:
        for slots in SLOT_COUNTS:
            for cost in COST_SCENARIOS:
                baseline_path = _baseline_task_path(
                    source_root=source_root,
                    horizon=horizon,
                    slots=slots,
                    cost=cost,
                )
                baseline = _read_json(baseline_path)
                if baseline.get("status") != "completed":
                    raise PolicyTargetError(
                        f"baseline_replay_incomplete:{baseline_path}"
                    )
                baseline_trades = pd.read_parquet(
                    economic._verify_record(baseline["files"]["trades"])
                )
                unit_time = _unit_time_metrics(baseline_trades, slot_count=slots)
                metric_rows.append(
                    _replay_metric_row(
                        result=baseline,
                        ranking_variant="dual_mfe_risk_veto",
                        horizon=horizon,
                        slots=slots,
                        cost=cost,
                        unit_time=unit_time,
                    )
                )
                annual_rows.extend(
                    {
                        "ranking_variant": "dual_mfe_risk_veto",
                        "horizon": horizon,
                        "slot_count": slots,
                        "cost_scenario": cost,
                        **dict(row),
                    }
                    for row in baseline["annual"]
                )
    for spec in _replay_specs():
        result = _read_json(_replay_task_path(output_root, spec.task_id))
        unit_time = dict(result["metrics"]["unit_time"])
        metric_rows.append(
            _replay_metric_row(
                result=result,
                ranking_variant=spec.ranking_variant,
                horizon=spec.timeout_day,
                slots=spec.slot_count,
                cost=spec.cost_scenario,
                unit_time=unit_time,
            )
        )
        annual_rows.extend(
            {
                "ranking_variant": spec.ranking_variant,
                "horizon": spec.timeout_day,
                "slot_count": spec.slot_count,
                "cost_scenario": spec.cost_scenario,
                **dict(row),
            }
            for row in result["annual"]
        )
    metrics = pd.DataFrame(metric_rows).sort_values(
        ["horizon", "slot_count", "cost_scenario", "ranking_variant"]
    )
    annual = pd.DataFrame(annual_rows).sort_values(
        ["horizon", "slot_count", "cost_scenario", "ranking_variant", "year"]
    )
    paired = metrics.merge(
        metrics[metrics["ranking_variant"].eq("dual_mfe_risk_veto")],
        on=["horizon", "slot_count", "cost_scenario"],
        suffixes=("", "_baseline"),
    )
    paired = paired[~paired["ranking_variant"].eq("dual_mfe_risk_veto")].copy()
    metric_columns = [
        "terminal_return",
        "gross_terminal_return",
        "terminal_cost_drag",
        "terminal_excess",
        "cagr",
        "gross_cagr",
        "sharpe",
        "maximum_drawdown",
        "turnover_to_starting_cash",
        "total_cost_cny",
        "total_cost_to_starting_cash",
        "take_profit_hit_rate",
        "timeout_gross_return_mean",
        "holding_days_mean",
        "completed_trades_per_slot_year",
        "mean_net_return_per_holding_day",
        "mean_log_return_per_holding_day",
    ]
    for column in metric_columns:
        paired[f"{column}_delta"] = paired[column].astype(float) - paired[
            f"{column}_baseline"
        ].astype(float)
    paired = paired[
        [
            "ranking_variant",
            "horizon",
            "slot_count",
            "cost_scenario",
            *[f"{column}_delta" for column in metric_columns],
        ]
    ]
    horizon = metrics[metrics["ranking_variant"].isin(REPLAY_VARIANTS)].merge(
        metrics[metrics["ranking_variant"].isin(REPLAY_VARIANTS)],
        on=["ranking_variant", "slot_count", "cost_scenario"],
        suffixes=("_d10", "_d20"),
    )
    horizon = horizon[
        horizon["horizon_d10"].eq(10) & horizon["horizon_d20"].eq(20)
    ].copy()
    for column in metric_columns:
        horizon[f"{column}_d10_minus_d20"] = horizon[f"{column}_d10"].astype(
            float
        ) - horizon[f"{column}_d20"].astype(float)
    horizon = horizon[
        [
            "ranking_variant",
            "slot_count",
            "cost_scenario",
            *[f"{column}_d10_minus_d20" for column in metric_columns],
        ]
    ]
    return metrics, annual, paired, horizon


def replay(
    *,
    study_path: Path = DEFAULT_STUDY_PATH,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    max_tasks: int | None = None,
) -> dict[str, Any]:
    config = _load_config(study_path)
    results = _completed_training_results(output_root)
    if len(results) != 18:
        raise PolicyTargetError("policy_training_required_before_replay")
    inputs, _, _ = _load_inputs(config=config, output_root=output_root)
    sources = _source_paths(config)
    execution_study = execution.load_study(sources["execution_study"])
    execution_audit = _read_json(sources["execution_output"] / "audit.json")
    source_profit_audit = _read_json(sources["profit_timeout_output"] / "audit.json")
    if (
        execution_audit.get("status") != "ok"
        or source_profit_audit.get("status") != "ok"
    ):
        raise PolicyTargetError("source_execution_not_audited")
    base_book = execution.SignalBook(
        study=execution_study, output_root=sources["execution_output"]
    )
    order_records = _build_replay_orders(
        inputs=inputs,
        results=results,
        base_book=base_book,
        output_root=output_root,
    )
    profit_study = profit_timeout.load_study(sources["profit_timeout_study"])
    study_sha256 = canonical._sha256(study_path)
    completed = 0
    executed_count = 0
    started = time.monotonic()
    for spec in _replay_specs():
        order_record = order_records[(spec.ranking_variant, spec.timeout_day)]
        path = _replay_task_path(output_root, spec.task_id)
        if _replay_task_complete(
            path,
            study_sha256=study_sha256,
            order_sha256=str(order_record["sha256"]),
        ):
            completed += 1
            continue
        if max_tasks is not None and executed_count >= int(max_tasks):
            break
        orders = np.load(
            economic._verify_record(order_record), mmap_mode="r", allow_pickle=False
        )
        book = PolicySignalBook(base_book, orders)
        result, equity, trades, monthly = profit_timeout.simulate_task(
            book=book, spec=spec, study=profit_study
        )
        _write_replay_task(
            output_root=output_root,
            spec=spec,
            result=result,
            equity=equity,
            trades=trades,
            monthly=monthly,
            study_sha256=study_sha256,
            order_record=order_record,
        )
        completed += 1
        executed_count += 1
        _emit(
            "policy_replay_progress",
            completed=completed,
            total=32,
            task_id=spec.task_id,
            elapsed_seconds=round(time.monotonic() - started, 2),
        )
    if completed != 32:
        return {
            "status": "in_progress",
            "completed": completed,
            "total": 32,
            "executed_this_run": executed_count,
        }
    metrics, annual, paired, horizon = _collect_replay_evaluation(
        config=config, output_root=output_root
    )
    replay_root = output_root / "replay/evaluation"
    metric_path = replay_root / "metrics.parquet"
    annual_path = replay_root / "annual_metrics.parquet"
    paired_path = replay_root / "paired_baseline_deltas.parquet"
    horizon_path = replay_root / "d10_vs_d20.parquet"
    _write_parquet(metrics, metric_path)
    _write_parquet(annual, annual_path)
    _write_parquet(paired, paired_path)
    _write_parquet(horizon, horizon_path)
    manifest = _read_json(output_root / "manifest.json")
    manifest["status"] = "replayed"
    manifest["replay"] = {
        "new_task_count": 32,
        "baseline_reference_count": 8,
        "ranking_variants": list(REPLAY_VARIANTS),
        "horizons": list(HORIZONS),
        "slot_counts": list(SLOT_COUNTS),
        "cost_scenarios": list(COST_SCENARIOS),
        "parameter_search_performed": False,
        "order_files": {
            f"{variant}_d{horizon}": record
            for (variant, horizon), record in order_records.items()
        },
        "files": {
            "metrics": _file_record(metric_path, row_count=len(metrics)),
            "annual_metrics": _file_record(annual_path, row_count=len(annual)),
            "paired_baseline_deltas": _file_record(paired_path, row_count=len(paired)),
            "d10_vs_d20": _file_record(horizon_path, row_count=len(horizon)),
        },
    }
    _write_json(output_root / "manifest.json", manifest)
    return {
        "status": "completed",
        "completed": completed,
        "total": 32,
        "metric_row_count": len(metrics),
    }


def status(
    *,
    study_path: Path = DEFAULT_STUDY_PATH,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
) -> dict[str, Any]:
    _load_config(study_path)
    training = _completed_training_results(output_root)
    replay_completed = sum(
        _replay_task_path(output_root, spec.task_id).is_file()
        for spec in _replay_specs()
    )
    return {
        "labels": (
            "completed"
            if _labels_complete(output_root / "labels/manifest.json")
            else "pending"
        ),
        "training_completed": len(training),
        "training_total": 18,
        "replay_completed": replay_completed,
        "replay_total": 32,
        "manifest_status": (
            _read_json(output_root / "manifest.json").get("status")
            if (output_root / "manifest.json").is_file()
            else "pending"
        ),
    }


def audit(
    *,
    study_path: Path = DEFAULT_STUDY_PATH,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
) -> dict[str, Any]:
    config = _load_config(study_path)
    label_path = output_root / "labels/manifest.json"
    label_valid = _labels_complete(label_path)
    labels = _read_json(label_path) if label_valid else {}
    results = _completed_training_results(output_root)
    training_valid = len(results) == 18
    task_contract_valid = training_valid
    if training_valid:
        for task in _target_plan():
            result = results[str(task["task_id"])]
            task_contract_valid &= bool(
                int(result["feature_count"]) == FEATURE_COUNT
                and int(result["purge_days"]) == int(task["horizon"])
                and int(result["evaluation_year"]) in ROLLING_YEARS
                and not str(result["maximum_train_signal_date"]).startswith("2010")
                and not str(result["maximum_train_signal_date"]).startswith("2026")
            )
    manifest_path = output_root / "manifest.json"
    manifest = _read_json(manifest_path) if manifest_path.is_file() else {}
    evaluation_files_valid = _records_valid(dict(manifest.get("files", {})))
    replay_section = dict(manifest.get("replay", {}))
    replay_files_valid = _records_valid(dict(replay_section.get("files", {})))
    order_files_valid = _records_valid(dict(replay_section.get("order_files", {})))
    replay_tasks_valid = True
    replay_dates_valid = True
    study_sha256 = canonical._sha256(study_path)
    for spec in _replay_specs():
        key = f"{spec.ranking_variant}_d{spec.timeout_day}"
        order = dict(replay_section.get("order_files", {})).get(key, {})
        path = _replay_task_path(output_root, spec.task_id)
        complete = bool(order) and _replay_task_complete(
            path,
            study_sha256=study_sha256,
            order_sha256=str(order.get("sha256", "")),
        )
        replay_tasks_valid &= complete
        if not complete:
            replay_dates_valid = False
            continue
        result = _read_json(path)
        equity = pd.read_parquet(
            economic._verify_record(result["files"]["equity"]),
            columns=["trade_date"],
        )
        dates = equity["trade_date"].astype(str)
        replay_dates_valid &= bool(
            len(dates) == 727
            and dates.iloc[0] == "2023-01-03"
            and dates.iloc[-1] == MAXIMUM_OUTCOME_DATE
            and not dates.str.startswith(str(FORBIDDEN_YEAR)).any()
            and int(result["period"]["forbidden_2026_row_count"]) == 0
        )
    no_html = not any(output_root.rglob("*.html"))
    checks = {
        "label_files_valid": label_valid,
        "label_row_count_matches_model": (
            int(labels.get("row_count", -1)) == 4_476_851
        ),
        "label_columns_exact": tuple(labels.get("label_columns", ())) == LABEL_COLUMNS,
        "label_cutoff_and_2026_boundary_valid": (
            labels.get("contract", {}).get("maximum_source_date_read")
            == MAXIMUM_OUTCOME_DATE
            and int(labels.get("contract", {}).get("forbidden_2026_read_count", -1))
            == 0
        ),
        "training_task_count_exact": training_valid,
        "training_feature_and_purge_contract_valid": task_contract_valid,
        "evaluation_files_valid": evaluation_files_valid,
        "replay_task_count_exact": replay_tasks_valid,
        "replay_order_files_valid": order_files_valid,
        "replay_evaluation_files_valid": replay_files_valid,
        "replay_dates_and_2026_boundary_valid": replay_dates_valid,
        "manifest_feature_count_557": int(manifest.get("feature_count", -1))
        == FEATURE_COUNT,
        "no_hyperparameter_search": not bool(
            manifest.get("hyperparameter_search_performed", True)
        ),
        "no_html_report": no_html,
    }
    payload = {
        "schema": AUDIT_SCHEMA,
        "status": "ok" if all(checks.values()) else "failed",
        "created_at": _now(),
        "study_id": STUDY_ID,
        "checks": checks,
        "training_task_count": len(results),
        "new_replay_task_count": 32 if replay_tasks_valid else 0,
        "baseline_reference_count": 8,
        "forbidden_2026_read_count": 0,
        "research_semantics": str(config["period"]["research_semantics"]),
    }
    _write_json(output_root / "audit.json", payload)
    if payload["status"] != "ok":
        raise PolicyTargetError("policy_target_audit_failed")
    manifest["status"] = "audited"
    manifest["audit"] = _file_record(output_root / "audit.json")
    _write_json(manifest_path, manifest)
    return payload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-path", type=Path, default=DEFAULT_STUDY_PATH)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument("--status", action="store_true")
    actions.add_argument("--prepare", action="store_true")
    actions.add_argument("--run", action="store_true")
    actions.add_argument("--evaluate", action="store_true")
    actions.add_argument("--replay", action="store_true")
    actions.add_argument("--audit", action="store_true")
    parser.add_argument("--max-tasks", type=int, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.status:
        result = status(study_path=args.study_path, output_root=args.output_root)
    elif args.prepare:
        result = prepare(study_path=args.study_path, output_root=args.output_root)
    elif args.run:
        result = run(
            study_path=args.study_path,
            output_root=args.output_root,
            max_tasks=args.max_tasks,
        )
    elif args.evaluate:
        result = evaluate(study_path=args.study_path, output_root=args.output_root)
    elif args.replay:
        result = replay(
            study_path=args.study_path,
            output_root=args.output_root,
            max_tasks=args.max_tasks,
        )
    else:
        result = audit(study_path=args.study_path, output_root=args.output_root)
    print(json.dumps(economic._json_safe(result), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
