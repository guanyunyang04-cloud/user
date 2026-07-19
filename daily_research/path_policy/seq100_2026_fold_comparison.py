from __future__ import annotations

import gc
import hashlib
import json
import math
import os
import shutil
import subprocess
import warnings
from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import psutil
import pyarrow.parquet as pq
import torch

from daily_research.path_policy import qdp_v2_sequence_path_pack as pack_builder
from daily_research.path_policy import qdp_v2_sequence_path_training as training
from daily_research.path_policy import seq100_candidate_execution as candidate_execution
from daily_research.path_policy import seq100_checkpoint_freshness as freshness
from daily_research.path_policy import seq100_development as development
from daily_research.path_policy import seq100_structured_experiment as structured
from daily_research.path_policy import seq100_walkforward as walkforward
from daily_research.path_policy.seq100_development import (
    PYTHON,
    WORKSPACE_ROOT,
    build_todayclose_path_only_train_argv,
)


STUDY_ID = "seq100_legal_structured_path_2026_fold_v1"
ANALYSIS_ID = "seq100_checkpoint_vintage_on_2026_v1"
TARGET_YEAR = 2026
CHECKPOINT_VINTAGES = (2023, 2024, 2025, 2026)
PROFILE_ORDER = ("legal_flat_baseline", "structured_joint_turnover")
TOP_K_VALUES = (1, 3, 5, 10)
SEED = 7

EXPECTED_QDP_AS_OF = "2026-07-16"
SIGNAL_START = "2026-01-05"
SIGNAL_END = "2026-03-19"
SAFE_TRAIN_SIGNAL_END = "2025-09-02"
PURGE_START = "2025-09-03"
PURGE_END = "2025-12-31"
FORWARD_DAYS = 60
EXECUTION_TAIL_DAYS = 20
DEPENDENCY_DAYS = FORWARD_DAYS + EXECUTION_TAIL_DAYS
EXPECTED_SIGNAL_DATE_COUNT = 48
EXPECTED_CANDIDATE_COUNT = 143_840
EXPECTED_SYMBOL_COUNT = 3_034
EXPECTED_TRAIN_ROW_COUNT = 7_773_480
EXPECTED_PURGED_ROW_COUNT = 239_081
EXPECTED_PURGED_DATE_COUNT = 80
OVERLAY_SIZE_LIMIT_BYTES = int(1.25 * (1024**3))
MEMORY_GUARD_GIB = 0.5

BASE_PACK_MANIFEST = (
    WORKSPACE_ROOT
    / "daily_research"
    / "data"
    / "research_store"
    / "seq100_current"
    / "pack"
    / "manifest.json"
)
BASE_TURNOVER_MANIFEST = (
    WORKSPACE_ROOT
    / "daily_research"
    / "data"
    / "research_store"
    / "seq100_current"
    / "supplements"
    / "relative_turnover_v1"
    / "manifest.json"
)
BASE_STUDY_ROOT = (
    WORKSPACE_ROOT
    / "daily_research"
    / "output"
    / "path_policy"
    / "studies"
    / "seq100_legal_structured_path_rolling_2023_2025_v1"
)
OVERLAY_ROOT = (
    WORKSPACE_ROOT
    / "daily_research"
    / "data"
    / "research_store"
    / "seq100_2026_overlay_v1"
)
STUDY_ROOT = (
    WORKSPACE_ROOT
    / "daily_research"
    / "output"
    / "path_policy"
    / "studies"
    / STUDY_ID
)
ANALYSIS_ROOT = STUDY_ROOT / "analysis" / "checkpoint_vintage_2026_20260719"

EXPECTED_IMPLEMENTATION_HASHES = {
    "freshness": "e2ac69e0401775c3672f57aa6f9430a2a45a28bf50a9d25e8a711763a908ea88",
    "prediction": "8d58265cddad48d92f06d0bb1d9f8a4d86491569bcefeacf341bab875852fef8",
    "execution": "7988f955e4406eaeece90ef617619942668e66ed08abac450b6fe1c1076dfe87",
    "base_pack": "6dc5dc0dac612a6e5bf53732d374facf67c8cc4e11c532a85dcc03f771b1a16a",
}

LEGAL_EXIT_CONTRACT = {
    "earliest_legal_exit_day": 2,
    "exit_argmax_domain": [2, 60],
    "tie_break": "earliest_legal_day",
}


def _now() -> str:
    return datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _json_bytes(payload: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(dict(payload), ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    ).encode("utf-8")


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(_json_bytes(payload))
    os.replace(temporary, path)


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False, encoding="utf-8-sig")
    os.replace(temporary, path)


def _write_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False)
    os.replace(temporary, path)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_material(
    path: Path,
    *,
    dtype: str | None = None,
    shape: Sequence[int] | None = None,
) -> dict[str, Any]:
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    payload: dict[str, Any] = {
        "path": str(resolved),
        "file_size": int(resolved.stat().st_size),
        "sha256": _file_sha256(resolved),
    }
    if dtype is not None:
        payload["dtype"] = str(dtype)
    if shape is not None:
        payload["shape"] = [int(value) for value in shape]
    return payload


def _canonical_digest(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _replace_path_prefix(value: Any, old: str, new: str) -> Any:
    if isinstance(value, str):
        return value.replace(old, new)
    if isinstance(value, list):
        return [_replace_path_prefix(item, old, new) for item in value]
    if isinstance(value, dict):
        return {key: _replace_path_prefix(item, old, new) for key, item in value.items()}
    return value


def _directory_size(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def _remove_path(path: Path) -> None:
    resolved = path.resolve()
    allowed = {
        OVERLAY_ROOT.resolve(),
        OVERLAY_ROOT.with_name(OVERLAY_ROOT.name + ".staging").resolve(),
    }
    if resolved not in allowed and OVERLAY_ROOT.with_name(
        OVERLAY_ROOT.name + ".staging"
    ).resolve() not in resolved.parents:
        raise ValueError(f"refusing to remove path outside the 2026 overlay staging area: {path}")
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()


def _assert_no_other_research_process() -> None:
    current = os.getpid()
    excluded = {current}
    try:
        parent = psutil.Process(current).parent()
        while parent is not None:
            excluded.add(int(parent.pid))
            parent = parent.parent()
    except (psutil.AccessDenied, psutil.NoSuchProcess):
        pass
    markers = (
        "qdp_v2_sequence_path_training",
        "seq100_2026_fold_comparison",
        "seq100_development run-2026-fold",
        "seq100_development prepare-2026-fold",
        "seq100_development evaluate-2026-vintages",
    )
    matches: list[int] = []
    for process in psutil.process_iter(["pid", "cmdline"]):
        try:
            if int(process.info["pid"]) in excluded:
                continue
            command = " ".join(process.info.get("cmdline") or [])
            if any(marker in command for marker in markers):
                matches.append(int(process.info["pid"]))
        except (psutil.AccessDenied, psutil.NoSuchProcess):
            continue
    if matches:
        raise RuntimeError(f"another Seq100 research process is active: pids={matches}")


def _guarded_command(command: Sequence[str], *, log_path: Path) -> None:
    guard = [
        str(PYTHON),
        str((WORKSPACE_ROOT / "tools" / "memory_guard.py").resolve()),
        "--min-available-gb",
        str(MEMORY_GUARD_GIB),
        "--interval-seconds",
        "1",
        "--consecutive-breaches",
        "2",
        "--log-json",
        str(log_path.resolve()),
        "--",
        *[str(item) for item in command],
    ]
    completed = subprocess.run(guard, cwd=WORKSPACE_ROOT, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"guarded command failed with exit code {completed.returncode}")


def _assert_frozen_inputs() -> dict[str, Any]:
    implementation_paths = {
        "freshness": Path(freshness.__file__).resolve(),
        "prediction": Path(training.__file__).resolve(),
        "execution": Path(candidate_execution.__file__).resolve(),
        "base_pack": BASE_PACK_MANIFEST.resolve(),
    }
    observed = {key: _file_sha256(path) for key, path in implementation_paths.items()}
    if observed != EXPECTED_IMPLEMENTATION_HASHES:
        raise ValueError(
            f"frozen Seq100 implementation or base pack changed: {observed}"
        )
    qdp_root = Path(development.QDP_ROOT).resolve()
    active = development._read_active(qdp_root)
    if str(active.get("active_as_of_date", "")) != EXPECTED_QDP_AS_OF:
        raise ValueError("QDP active_as_of changed from the frozen 2026-07-16 boundary")
    audit_path, audit = development._latest_full_audit(qdp_root)
    preflight = development._preflight_audit_metrics(audit)
    return {
        "implementation_sha256": observed,
        "qdp_root": str(qdp_root),
        "qdp_active_as_of": str(active["active_as_of_date"]),
        "qdp_active_manifest": str((qdp_root / "active" / "active.json").resolve()),
        "qdp_active_manifest_sha256": _file_sha256(
            qdp_root / "active" / "active.json"
        ),
        "qdp_full_audit": str(audit_path.resolve()),
        "qdp_full_audit_sha256": _file_sha256(audit_path),
        "qdp_preflight": preflight,
    }


def _profile_contracts() -> dict[str, dict[str, Any]]:
    return structured._profile_contracts(512)


def _study_semantic_contract() -> dict[str, Any]:
    profiles = _profile_contracts()
    for payload in profiles.values():
        for key in ("store_view", "output_root", "run_tag"):
            payload.pop(key, None)
    return {
        "schema_version": 1,
        "contract_id": STUDY_ID,
        "experiment": "structured-path-2026-fold-v1",
        "data": {
            "qdp_expected_as_of": EXPECTED_QDP_AS_OF,
            "base_pack": str(BASE_PACK_MANIFEST.resolve()),
            "overlay": str(OVERLAY_ROOT.resolve()),
            "base_pack_mutated": False,
            "input_dim": 32,
            "lookback_days": 100,
            "forward_days": FORWARD_DAYS,
            "execution_tail_days": EXECUTION_TAIL_DAYS,
            "signal_window": [SIGNAL_START, SIGNAL_END],
            "candidate_count": EXPECTED_CANDIDATE_COUNT,
            "symbol_count": EXPECTED_SYMBOL_COUNT,
        },
        "legal_exit_contract": dict(LEGAL_EXIT_CONTRACT),
        "development_protocol": {
            "years": [TARGET_YEAR],
            "method": "purged_expanding_development_walkforward",
            "split_roles": {"fit": "train", "evaluation": "development"},
            "train_start_year": 2010,
            "safe_train_signal_end": SAFE_TRAIN_SIGNAL_END,
            "purge_window": [PURGE_START, PURGE_END],
            "purge_trade_date_count": EXPECTED_PURGED_DATE_COUNT,
            "seed": SEED,
            "task_order": [f"{profile}:{TARGET_YEAR}" for profile in PROFILE_ORDER],
            "final_fit_in_scope": False,
        },
        "early_stopping": {
            "metric": training.EARLY_STOPPING_METRIC_DEVELOPMENT_PRICE_TOTAL_LOSS,
            "formula": "0.45*path + 0.20*summary + 0.20*legal_value + 0.15*rank",
            "excludes": ["volume", "amount", "turnover", "geometry", "utility_curve"],
            "mode": "min",
            "minimum_complete_epochs": 1,
            "maximum_epochs": 10,
            "patience": 2,
            "restore_best_checkpoint": True,
        },
        "profiles": profiles,
        "metric_contract": {
            "primary": "2026_top3_net_realized_plan_return_base_alpha",
            "top_k": list(TOP_K_VALUES),
            "automatic_winner": False,
            "formal_60d_block_inference": False,
        },
        "protected_boundaries": {
            "update_qdp": False,
            "call_provider": False,
            "mutate_base_pack": False,
            "change_active_execution": False,
            "create_final_model": False,
        },
    }


def _initial_study() -> dict[str, Any]:
    semantic = _study_semantic_contract()
    return {
        "schema_version": 1,
        "artifact_type": "seq100_current_study",
        "study_id": STUDY_ID,
        "contract": semantic,
        "contract_sha256": _canonical_digest(semantic),
        "runtime": {
            "status": "preparing",
            "completed_tasks": [],
            "created_at": _now(),
            "updated_at": _now(),
            "error": None,
            "current_task": None,
        },
        "material": {},
    }


def _load_or_create_study() -> tuple[Path, dict[str, Any]]:
    STUDY_ROOT.mkdir(parents=True, exist_ok=True)
    path = STUDY_ROOT / "study.json"
    expected = _initial_study()
    if not path.is_file():
        _write_json(path, expected)
        return path, expected
    study = _read_json(path)
    semantic = dict(study.get("contract", {}) or {})
    if semantic != expected["contract"] or str(study.get("contract_sha256", "")) != str(
        expected["contract_sha256"]
    ):
        raise ValueError("existing 2026 study contract differs from the registered contract")
    return path, study


def _update_runtime(study_path: Path, study: dict[str, Any], **changes: Any) -> None:
    runtime = dict(study.get("runtime", {}) or {})
    runtime.update(changes)
    runtime["updated_at"] = _now()
    study["runtime"] = runtime
    _write_json(study_path, study)


def _open_memmap(meta: Mapping[str, Any], *, dtype: str) -> np.memmap:
    path = Path(str(meta.get("path", "") or "")).resolve()
    shape = tuple(int(item) for item in list(meta.get("shape", []) or []))
    if not path.is_file() or not shape:
        raise FileNotFoundError(path)
    return np.memmap(path, dtype=dtype, mode="r", shape=shape)


def _create_memmap(path: Path, *, dtype: str, shape: Sequence[int], fill: Any) -> np.memmap:
    path.parent.mkdir(parents=True, exist_ok=True)
    array = np.memmap(path, dtype=dtype, mode="w+", shape=tuple(int(v) for v in shape))
    array[:] = fill
    array.flush()
    return array


def _close_memmap(array: Any) -> None:
    handle = getattr(array, "_mmap", None)
    if handle is not None:
        handle.close()


def _copy_prefix_and_tail_panel(
    *,
    output_path: Path,
    base_meta: Mapping[str, Any],
    compact_meta: Mapping[str, Any],
    dtype: str,
    extended_date_values: Sequence[str],
    compact_date_to_idx: Mapping[str, int],
    compact_symbol_indices: np.ndarray,
    fill: Any,
) -> dict[str, Any]:
    base = _open_memmap(base_meta, dtype=dtype)
    compact = _open_memmap(compact_meta, dtype=dtype)
    tail_shape = tuple(int(item) for item in base.shape[1:])
    output_shape = (len(extended_date_values), *tail_shape)
    output = _create_memmap(output_path, dtype=dtype, shape=output_shape, fill=fill)
    for start in range(0, int(base.shape[0]), 32):
        stop = min(start + 32, int(base.shape[0]))
        output[start:stop] = np.asarray(base[start:stop])
    for global_idx in range(int(base.shape[0]), len(extended_date_values)):
        compact_idx = int(compact_date_to_idx[str(extended_date_values[global_idx])])
        output[global_idx] = np.asarray(compact[compact_idx, compact_symbol_indices])
    output.flush()
    _close_memmap(base)
    _close_memmap(compact)
    _close_memmap(output)
    return _file_material(output_path, dtype=dtype, shape=output_shape)


def _assert_mapped_panel_overlap(
    *,
    name: str,
    base_meta: Mapping[str, Any],
    compact_meta: Mapping[str, Any],
    dtype: str,
    base_date_indices: Sequence[int],
    compact_date_indices: Sequence[int],
    compact_symbol_indices: np.ndarray,
    atol: float = 1.0e-6,
    rtol: float = 1.0e-6,
) -> None:
    if len(base_date_indices) != len(compact_date_indices):
        raise ValueError(f"{name} overlap date mapping length differs")
    base = _open_memmap(base_meta, dtype=dtype)
    compact = _open_memmap(compact_meta, dtype=dtype)
    try:
        for base_idx, compact_idx in zip(base_date_indices, compact_date_indices):
            left = np.asarray(base[int(base_idx)])
            right = np.asarray(compact[int(compact_idx), compact_symbol_indices])
            if dtype == "bool":
                matches = bool(np.array_equal(left, right))
            else:
                matches = bool(
                    np.allclose(
                        left,
                        right,
                        equal_nan=True,
                        atol=float(atol),
                        rtol=float(rtol),
                    )
                )
            if not matches:
                raise ValueError(
                    f"Jan 5 overlap regression failed for {name} at "
                    f"base_date_idx={int(base_idx)} compact_date_idx={int(compact_idx)}"
                )
    finally:
        _close_memmap(base)
        _close_memmap(compact)


def _derive_today_close_label_and_summary(
    *,
    daily_raw_meta: Mapping[str, Any],
    date_idx: int,
    label_valid: np.ndarray,
    forward_days: int = FORWARD_DAYS,
) -> tuple[np.ndarray, np.ndarray]:
    daily = _open_memmap(daily_raw_meta, dtype="float32")
    columns = [str(value) for value in list(daily_raw_meta.get("columns", []) or [])]
    required = ("open", "high", "low", "close", "volume_log", "amount_log")
    if any(name not in columns for name in required):
        _close_memmap(daily)
        raise ValueError("daily_raw is missing fields required for Jan 5 label regression")
    try:
        start = int(date_idx) + 1
        end = start + int(forward_days)
        if end > int(daily.shape[0]):
            raise ValueError("base daily panel does not cover the Jan 5 price horizon")
        signal_close = np.asarray(
            daily[int(date_idx), :, columns.index("close")], dtype=np.float64
        ).copy()
        future = {
            name: np.asarray(daily[start:end, :, columns.index(name)], dtype=np.float32)
            .T.copy()
            for name in required
            if name != "close"
        }
        future["close"] = np.asarray(
            daily[start:end, :, columns.index("close")], dtype=np.float32
        ).T.copy()
        history_start = max(0, int(date_idx) - 19)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            trailing_volume = np.nanmean(
                np.asarray(
                    daily[
                        history_start : int(date_idx) + 1,
                        :,
                        columns.index("volume_log"),
                    ],
                    dtype=np.float32,
                ),
                axis=0,
            )
            trailing_amount = np.nanmean(
                np.asarray(
                    daily[
                        history_start : int(date_idx) + 1,
                        :,
                        columns.index("amount_log"),
                    ],
                    dtype=np.float32,
                ),
                axis=0,
            )
    finally:
        _close_memmap(daily)
    denom = np.where(signal_close != 0.0, signal_close, np.nan)
    price = np.stack(
        [future[name] / denom[:, None] - 1.0 for name in ("open", "high", "low", "close")],
        axis=2,
    ).astype(np.float32, copy=False)
    volume_rel = (future["volume_log"] - trailing_volume[:, None]).astype(
        np.float32, copy=False
    )
    amount_rel = (future["amount_log"] - trailing_amount[:, None]).astype(
        np.float32, copy=False
    )
    label = np.concatenate(
        [price, volume_rel[:, :, None], amount_rel[:, :, None]], axis=2
    ).astype(np.float32, copy=False)
    valid = np.asarray(label_valid, dtype=bool).reshape(-1)
    if valid.shape[0] != label.shape[0]:
        raise ValueError("Jan 5 label-valid mask width differs from the base symbol panel")
    label[~valid] = np.nan

    entry_anchor = np.maximum(
        1.0 + price[:, :1, 0].astype(np.float64, copy=False), 1.0e-6
    )
    entry_relative = (
        (1.0 + price.astype(np.float64, copy=False)) / entry_anchor[:, :, None]
        - 1.0
    )
    high_ret = entry_relative[:, :, 1]
    low_ret = entry_relative[:, :, 2]
    close_ret = entry_relative[:, :, 3]
    safe_high = np.where(np.isfinite(high_ret), high_ret, -np.inf)
    safe_low = np.where(np.isfinite(low_ret), low_ret, np.inf)
    max_ret = np.max(safe_high, axis=1)
    min_ret = np.min(safe_low, axis=1)
    max_ret[~np.isfinite(max_ret)] = np.nan
    min_ret[~np.isfinite(min_ret)] = np.nan
    final_ret = close_ret[:, -1]
    peak_idx = np.argmax(safe_high, axis=1)
    trough_idx = np.argmin(safe_low, axis=1)
    min_after_peak = np.full(label.shape[0], np.nan, dtype=np.float64)
    for symbol_idx in np.flatnonzero(valid):
        min_after_peak[symbol_idx] = np.nanmin(
            low_ret[symbol_idx, int(peak_idx[symbol_idx]) :]
        )
    drawdown_after_peak = (1.0 + min_after_peak) / (1.0 + max_ret) - 1.0
    time_above = (close_ret > 0.0).mean(axis=1)
    time_below = (close_ret < 0.0).mean(axis=1)
    value = (
        final_ret
        + 0.50 * max_ret
        + 0.35 * min_ret
        + 0.20 * drawdown_after_peak
    )
    summary = np.column_stack(
        [
            max_ret,
            min_ret,
            final_ret,
            peak_idx + 1,
            trough_idx + 1,
            drawdown_after_peak,
            time_above,
            time_below,
            value,
        ]
    ).astype(np.float32, copy=False)
    summary[~valid] = np.nan
    return label, summary


def _compact_pack(staging: Path, study_path: Path) -> Path:
    compact_root = staging / "_compact"
    compact_manifest = compact_root / "manifest.json"
    if compact_manifest.is_file():
        validation = pack_builder.validate_sequence_pack(compact_manifest)
        if str(validation.get("status", "")) == "ok":
            return compact_manifest
        _remove_path(compact_root)
    elif compact_root.exists():
        _remove_path(compact_root)
    base = _read_json(BASE_PACK_MANIFEST)
    semantics = dict(base.get("data_semantics", {}) or {})
    config = pack_builder.SequencePackConfig(
        qdp_root=Path(development.QDP_ROOT).resolve(),
        output_root=staging,
        run_tag="_compact",
        lookback_days=100,
        forward_days=FORWARD_DAYS,
        start_date=SIGNAL_START,
        end_date=SIGNAL_END,
        train_years=(),
        validation_years=(),
        test_years=(TARGET_YEAR,),
        write_legacy_ohlc_label=False,
        label_shard_size=64,
        price_anchor="today_close",
        price_adjustment=str(semantics["price_adjustment"]),
        entry_rule=str(semantics["entry_rule"]),
        sample_filter=pack_builder.SAMPLE_FILTER_CURRENT_QDP,
        suspension_fill=pack_builder.SUSPENSION_FILL_CARRY_CLOSE,
        execution_tail_days=EXECUTION_TAIL_DAYS,
        require_full_dependency_padding=True,
        minimum_free_memory_gb=MEMORY_GUARD_GIB,
        unresolved_exit_recovery_fraction=0.0,
        execution_cost=pack_builder.AShareExecutionCostConfig(),
        research_contract=study_path,
        pit_universe_manifest=None,
        dataset_view=None,
        feature_profile=pack_builder.FEATURE_PROFILE_DAILY_ONLY,
    )
    pack_builder.build_sequence_pack(config)
    validation = pack_builder.validate_sequence_pack(compact_manifest)
    if str(validation.get("status", "")) != "ok":
        raise ValueError(f"compact 2026 pack validation failed: {validation}")
    return compact_manifest


def _compact_turnover(staging: Path, compact_manifest: Path) -> Path:
    target = staging / "_compact_turnover"
    payload = structured.build_relative_turnover_supplement(
        pack_manifest=compact_manifest,
        output_root=target,
    )
    path = target / "manifest.json"
    if not path.is_file() or str(payload.get("artifact_type", "")) != (
        "seq100_relative_turnover_supplement"
    ):
        raise ValueError("compact turnover supplement was not completed")
    return path


def _subset_compact_indexes(
    *,
    base: Mapping[str, Any],
    compact: Mapping[str, Any],
    extended_date_values: Sequence[str],
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    base_symbols = [str(item) for item in list(base["symbol_values"])]
    compact_symbols = [str(item) for item in list(compact["symbol_values"])]
    if len(base_symbols) != EXPECTED_SYMBOL_COUNT:
        raise ValueError("base pack symbol count drifted")
    compact_symbol_to_idx = {symbol: idx for idx, symbol in enumerate(compact_symbols)}
    missing = [symbol for symbol in base_symbols if symbol not in compact_symbol_to_idx]
    if missing:
        raise ValueError(f"compact pack is missing base symbols: {missing[:10]}")
    compact_symbol_indices = np.asarray(
        [compact_symbol_to_idx[symbol] for symbol in base_symbols], dtype=np.int64
    )
    global_date_to_idx = {date: idx for idx, date in enumerate(extended_date_values)}
    compact_date_to_idx = {
        str(date): idx for idx, date in enumerate(list(compact["date_values"]))
    }
    signal_dates = [
        date for date in extended_date_values if SIGNAL_START <= date <= SIGNAL_END
    ]
    if (
        len(signal_dates) != EXPECTED_SIGNAL_DATE_COUNT
        or signal_dates[0] != SIGNAL_START
        or signal_dates[-1] != SIGNAL_END
        or any(date not in compact_date_to_idx for date in signal_dates)
    ):
        raise ValueError("compact pack does not cover the registered 2026 signal window")

    base_candidate_meta = dict(dict(base["masks"])["candidate_eligible"])
    base_candidate = _open_memmap(base_candidate_meta, dtype="bool")
    compact_masks = dict(compact["masks"])
    compact_entry = _open_memmap(compact_masks["entry_buyable"], dtype="bool")
    compact_label = _open_memmap(compact_masks["label_valid"], dtype="bool")
    compact_price_label = _open_memmap(
        compact_masks["price_label_valid"], dtype="bool"
    )
    compact_va_aux = _open_memmap(compact_masks["va_aux_valid"], dtype="bool")
    candidate_rows: list[pd.DataFrame] = []
    try:
        for trade_date in signal_dates:
            global_idx = int(global_date_to_idx[trade_date])
            compact_idx = int(compact_date_to_idx[trade_date])
            symbol_idx = np.flatnonzero(base_candidate[global_idx]).astype(np.int64)
            compact_idx_values = compact_symbol_indices[symbol_idx]
            candidate_rows.append(
                pd.DataFrame(
                    {
                        "split": "development",
                        "year": np.int16(TARGET_YEAR),
                        "trade_date": str(trade_date),
                        "date_idx": np.full(len(symbol_idx), global_idx, dtype=np.int64),
                        "symbol_idx": symbol_idx,
                        "symbol": np.asarray(base_symbols, dtype=object)[symbol_idx],
                        "entry_trade_date": str(
                            extended_date_values[global_idx + 1]
                            if global_idx + 1 < len(extended_date_values)
                            else ""
                        ),
                        "entry_filled": np.asarray(
                            compact_entry[compact_idx, compact_idx_values], dtype=bool
                        ),
                        "label_valid": np.asarray(
                            compact_label[compact_idx, compact_idx_values], dtype=bool
                        ),
                        "price_label_valid": np.asarray(
                            compact_price_label[compact_idx, compact_idx_values],
                            dtype=bool,
                        ),
                        "va_aux_valid": np.asarray(
                            compact_va_aux[compact_idx, compact_idx_values], dtype=bool
                        ),
                    }
                )
            )
    finally:
        for array in (
            base_candidate,
            compact_entry,
            compact_label,
            compact_price_label,
            compact_va_aux,
        ):
            _close_memmap(array)
    if not candidate_rows:
        raise ValueError("frozen 2026 candidate mask produced no rows")
    candidates = pd.concat(candidate_rows, ignore_index=True)
    candidates = candidates.sort_values(
        ["date_idx", "symbol_idx"], kind="mergesort"
    ).reset_index(drop=True)
    candidates.insert(0, "candidate_id", np.arange(len(candidates), dtype=np.int64))

    supervised = candidates[
        candidates["price_label_valid"].astype(bool)
        & candidates["va_aux_valid"].astype(bool)
    ].copy()
    supervised.insert(0, "sample_id", np.arange(len(supervised), dtype=np.int64))
    supervised.insert(2, "source_split", "compact_label_complete")
    supervised["label_end_date_idx"] = (
        supervised["date_idx"].astype(np.int64) + FORWARD_DAYS
    )
    supervised["dependency_end_date_idx"] = (
        supervised["date_idx"].astype(np.int64) + DEPENDENCY_DAYS
    )
    supervised = supervised.drop(columns=["candidate_id"]).reset_index(drop=True)
    return candidates, supervised, compact_symbol_indices


def _validate_candidate_membership(
    *, base: Mapping[str, Any], candidates: pd.DataFrame
) -> None:
    dates = [str(item) for item in list(base["date_values"])]
    start_idx = dates.index(SIGNAL_START)
    end_idx = dates.index(SIGNAL_END)
    mask = _open_memmap(dict(base["masks"])["candidate_eligible"], dtype="bool")
    observed = np.asarray(mask[start_idx : end_idx + 1], dtype=bool).copy()
    expected_count = int(np.count_nonzero(observed))
    _close_memmap(mask)
    if expected_count != EXPECTED_CANDIDATE_COUNT or len(candidates) != expected_count:
        raise ValueError(
            f"2026 candidate count drift: expected={expected_count}, observed={len(candidates)}"
        )
    expected_hash = hashlib.sha256()
    for offset, row in enumerate(observed):
        symbols = np.flatnonzero(row).astype(np.int64)
        pairs = np.column_stack(
            [np.full(len(symbols), start_idx + offset, dtype=np.int64), symbols]
        )
        expected_hash.update(pairs.tobytes())
    observed_pairs = candidates[["date_idx", "symbol_idx"]].to_numpy(dtype=np.int64)
    observed_hash = hashlib.sha256(observed_pairs.tobytes()).hexdigest()
    if observed_hash != expected_hash.hexdigest():
        raise ValueError("2026 candidate keys differ from the frozen pack candidate mask")


def _build_fold_indexes(
    *,
    base: Mapping[str, Any],
    candidates: pd.DataFrame,
    supervised: pd.DataFrame,
    study_root: Path,
) -> tuple[Path, Path, dict[str, Any]]:
    source_samples = pd.read_parquet(str(base["sample_index_path"]))
    source_samples["trade_date"] = source_samples["trade_date"].astype(str)
    start_idx = [str(item) for item in list(base["date_values"])].index(SIGNAL_START)
    date_idx = source_samples["date_idx"].astype(np.int64)
    after_start = source_samples["trade_date"].str[:4].astype(int).ge(2010)
    train_mask = after_start & ((date_idx + DEPENDENCY_DAYS) < start_idx)
    purge_mask = after_start & (date_idx < start_idx) & (
        (date_idx + DEPENDENCY_DAYS) >= start_idx
    )
    train = source_samples.loc[train_mask].copy()
    if "source_split" not in train.columns:
        train.insert(1, "source_split", train["split"].astype(str).to_numpy(copy=True))
    train["split"] = "train"
    train["year"] = train["trade_date"].str[:4].astype(np.int16)
    train["label_end_date_idx"] = train["date_idx"].astype(np.int64) + FORWARD_DAYS
    train["dependency_end_date_idx"] = train["date_idx"].astype(np.int64) + DEPENDENCY_DAYS
    train = train.drop(columns=["label_symbol_idx"], errors="ignore")
    supervised = supervised.drop(columns=["label_symbol_idx"], errors="ignore")
    candidates = candidates.drop(columns=["label_symbol_idx"], errors="ignore")
    samples = pd.concat([train, supervised], ignore_index=True, sort=False)
    samples = samples.sort_values(["date_idx", "symbol_idx"], kind="mergesort").reset_index(drop=True)
    samples["sample_id"] = np.arange(len(samples), dtype=np.int64)
    purge = source_samples.loc[purge_mask]
    purge_dates = sorted(purge["trade_date"].unique().tolist())
    if (
        len(train) != EXPECTED_TRAIN_ROW_COUNT
        or str(train["trade_date"].max()) != SAFE_TRAIN_SIGNAL_END
        or len(purge) != EXPECTED_PURGED_ROW_COUNT
        or len(purge_dates) != EXPECTED_PURGED_DATE_COUNT
        or purge_dates[0] != PURGE_START
        or purge_dates[-1] != PURGE_END
    ):
        raise ValueError("2026 training or purge boundary drifted")
    index_root = study_root / "folds"
    sample_path = index_root / "sample_index" / "development_2026.parquet"
    candidate_path = index_root / "candidate_index" / "development_2026.parquet"
    _write_parquet(sample_path, samples)
    _write_parquet(candidate_path, candidates)
    audit = {
        "train_row_count": int(len(train)),
        "train_date_count": int(train["trade_date"].nunique()),
        "train_start": str(train["trade_date"].min()),
        "train_end": str(train["trade_date"].max()),
        "purged_row_count": int(len(purge)),
        "purged_date_count": int(len(purge_dates)),
        "purged_start": str(purge_dates[0]),
        "purged_end": str(purge_dates[-1]),
        "development_supervised_count": int(len(supervised)),
        "development_candidate_count": int(len(candidates)),
    }
    return sample_path, candidate_path, audit


def _extract_overlay(
    *,
    staging: Path,
    compact_manifest_path: Path,
    compact_turnover_path: Path,
    frozen: Mapping[str, Any],
) -> dict[str, Any]:
    base = _read_json(BASE_PACK_MANIFEST)
    compact = _read_json(compact_manifest_path)
    base_turnover = _read_json(BASE_TURNOVER_MANIFEST)
    compact_turnover = _read_json(compact_turnover_path)
    qdp_root = Path(str(frozen["qdp_root"]))
    active = development._read_active(qdp_root)
    extended_dates = development._open_dates(qdp_root, active)
    if extended_dates[-1] != EXPECTED_QDP_AS_OF:
        raise ValueError("extended calendar does not end at the frozen QDP cutoff")
    base_dates = [str(item) for item in list(base["date_values"])]
    compact_dates = [str(item) for item in list(compact["date_values"])]
    if base_dates[-1] != "2026-05-08" or extended_dates[: len(base_dates)] != base_dates:
        raise ValueError("base pack calendar is not a prefix of the frozen QDP calendar")
    compact_date_to_idx = {date: idx for idx, date in enumerate(compact_dates)}
    global_date_to_idx = {date: idx for idx, date in enumerate(extended_dates)}
    candidates, supervised, compact_symbol_indices = _subset_compact_indexes(
        base=base, compact=compact, extended_date_values=extended_dates
    )
    _validate_candidate_membership(base=base, candidates=candidates)

    label_dir = staging / "labels"
    mask_dir = staging / "masks"
    execution_dir = staging / "execution"
    supplement_dir = staging / "supplements" / "relative_turnover_v1"
    summary_meta = dict(dict(base["label_arrays"])["path_summary"])
    base_summary = _open_memmap(summary_meta, dtype="float32")
    compact_summary = _open_memmap(
        dict(dict(compact["label_arrays"])["path_summary"]), dtype="float32"
    )
    summary_shape = (len(extended_dates), EXPECTED_SYMBOL_COUNT, int(base_summary.shape[2]))
    summary_path = label_dir / "path_summary.float32.dat"
    summary_out = _create_memmap(
        summary_path, dtype="float32", shape=summary_shape, fill=np.nan
    )
    for start in range(0, len(base_dates), 32):
        stop = min(start + 32, len(base_dates))
        summary_out[start:stop] = np.asarray(base_summary[start:stop])
    for date in extended_dates[
        global_date_to_idx[SIGNAL_START] : global_date_to_idx[SIGNAL_END] + 1
    ]:
        summary_out[global_date_to_idx[date]] = np.asarray(
            compact_summary[compact_date_to_idx[date], compact_symbol_indices]
        )
    summary_out.flush()

    compact_label_reader = training._open_label_array(
        dict(dict(compact["label_arrays"])["future_ohlcva_path"]), dtype="float32"
    )
    signal_dates = extended_dates[
        global_date_to_idx[SIGNAL_START] : global_date_to_idx[SIGNAL_END] + 1
    ]
    label_shape = (len(signal_dates), EXPECTED_SYMBOL_COUNT, FORWARD_DAYS, 6)
    label_path = label_dir / "future_ohlcva_path.2026_development.float32.dat"
    label_out = _create_memmap(
        label_path, dtype="float32", shape=label_shape, fill=np.nan
    )
    for local_idx, date in enumerate(signal_dates):
        compact_date_idx = int(compact_date_to_idx[date])
        date_vector = np.full(EXPECTED_SYMBOL_COUNT, compact_date_idx, dtype=np.int64)
        label_out[local_idx] = compact_label_reader.take(
            date_vector, compact_symbol_indices
        )
    label_out.flush()

    execution_arrays = dict(base["execution_arrays"])
    compact_execution = dict(compact["execution_arrays"])
    entry_meta = _copy_prefix_and_tail_panel(
        output_path=execution_dir / "entry_open_raw.float32.dat",
        base_meta=execution_arrays["entry_open_raw"],
        compact_meta=compact_execution["entry_open_raw"],
        dtype="float32",
        extended_date_values=extended_dates,
        compact_date_to_idx=compact_date_to_idx,
        compact_symbol_indices=compact_symbol_indices,
        fill=np.nan,
    )
    exit_meta = _copy_prefix_and_tail_panel(
        output_path=execution_dir / "exit_close_raw.float32.dat",
        base_meta=execution_arrays["exit_close_raw"],
        compact_meta=compact_execution["exit_close_raw"],
        dtype="float32",
        extended_date_values=extended_dates,
        compact_date_to_idx=compact_date_to_idx,
        compact_symbol_indices=compact_symbol_indices,
        fill=np.nan,
    )

    masks = dict(base["masks"])
    compact_masks = dict(compact["masks"])
    extended_masks: dict[str, dict[str, Any]] = {}
    for name, compact_name in (
        ("tradable", "tradable"),
        ("has_bar", "has_bar"),
        ("exit_sellable", "exit_sellable"),
    ):
        extended_masks[name] = _copy_prefix_and_tail_panel(
            output_path=mask_dir / f"{name}.bool.dat",
            base_meta=masks[name],
            compact_meta=compact_masks[compact_name],
            dtype="bool",
            extended_date_values=extended_dates,
            compact_date_to_idx=compact_date_to_idx,
            compact_symbol_indices=compact_symbol_indices,
            fill=False,
        )

    turnover_meta: dict[str, dict[str, Any]] = {}
    for name in ("log_turnover_pct", "past20_positive_median"):
        turnover_meta[name] = _copy_prefix_and_tail_panel(
            output_path=supplement_dir / f"{name}.float32.dat",
            base_meta=dict(base_turnover[name]),
            compact_meta=dict(compact_turnover[name]),
            dtype="float32",
            extended_date_values=extended_dates,
            compact_date_to_idx=compact_date_to_idx,
            compact_symbol_indices=compact_symbol_indices,
            fill=np.nan,
        )

    jan5_global = global_date_to_idx[SIGNAL_START]
    jan5_compact = compact_date_to_idx[SIGNAL_START]
    compact_label_valid_panel = _open_memmap(
        compact_masks["label_valid"], dtype="bool"
    )
    compact_jan5_valid = np.asarray(
        compact_label_valid_panel[jan5_compact, compact_symbol_indices], dtype=bool
    ).copy()
    _close_memmap(compact_label_valid_panel)
    base_jan5, base_jan5_summary = _derive_today_close_label_and_summary(
        daily_raw_meta=dict(dict(base["feature_channels"])["daily_raw"]),
        date_idx=jan5_global,
        label_valid=compact_jan5_valid,
    )
    compact_jan5 = compact_label_reader.take(
        np.full(EXPECTED_SYMBOL_COUNT, jan5_compact, dtype=np.int64),
        compact_symbol_indices,
    )
    if not np.allclose(base_jan5, compact_jan5, equal_nan=True, atol=1.0e-6, rtol=1.0e-6):
        raise ValueError("Jan 5 OHLCVA overlap regression failed")
    if not np.allclose(
        base_jan5_summary,
        np.asarray(compact_summary[jan5_compact, compact_symbol_indices]),
        equal_nan=True,
        atol=1.0e-6,
        rtol=1.0e-6,
    ):
        raise ValueError("Jan 5 path-summary overlap regression failed")

    overlap_dates = extended_dates[jan5_global : jan5_global + DEPENDENCY_DAYS]
    if len(overlap_dates) != DEPENDENCY_DAYS or overlap_dates[-1] != base_dates[-1]:
        raise ValueError("Jan 5 overlap window no longer ends at the base-pack boundary")
    base_overlap_indices = [global_date_to_idx[date] for date in overlap_dates]
    compact_overlap_indices = [compact_date_to_idx[date] for date in overlap_dates]
    for name, base_meta, compact_meta, dtype in (
        (
            "entry_open_raw",
            execution_arrays["entry_open_raw"],
            compact_execution["entry_open_raw"],
            "float32",
        ),
        (
            "exit_close_raw",
            execution_arrays["exit_close_raw"],
            compact_execution["exit_close_raw"],
            "float32",
        ),
        ("tradable", masks["tradable"], compact_masks["tradable"], "bool"),
        ("has_bar", masks["has_bar"], compact_masks["has_bar"], "bool"),
        (
            "exit_sellable",
            masks["exit_sellable"],
            compact_masks["exit_sellable"],
            "bool",
        ),
        (
            "relative_turnover_log",
            dict(base_turnover["log_turnover_pct"]),
            dict(compact_turnover["log_turnover_pct"]),
            "float32",
        ),
        (
            "relative_turnover_past20_positive_median",
            dict(base_turnover["past20_positive_median"]),
            dict(compact_turnover["past20_positive_median"]),
            "float32",
        ),
    ):
        _assert_mapped_panel_overlap(
            name=name,
            base_meta=base_meta,
            compact_meta=compact_meta,
            dtype=dtype,
            base_date_indices=base_overlap_indices,
            compact_date_indices=compact_overlap_indices,
            compact_symbol_indices=compact_symbol_indices,
        )

    _close_memmap(base_summary)
    _close_memmap(compact_summary)
    _close_memmap(summary_out)
    _close_memmap(label_out)
    del (
        base_jan5,
        base_jan5_summary,
        compact_jan5,
        compact_jan5_valid,
        compact_label_reader,
    )
    gc.collect()

    label_arrays = dict(base["label_arrays"])
    old_label_meta = dict(label_arrays["future_ohlcva_path"])
    retained_shards = [
        dict(shard)
        for shard in list(old_label_meta.get("shards", []) or [])
        if int(shard.get("date_end_idx", -1)) < global_date_to_idx[SIGNAL_START]
    ]
    if not retained_shards or int(retained_shards[-1]["date_end_idx"]) < (
        global_date_to_idx[SAFE_TRAIN_SIGNAL_END]
    ):
        raise ValueError("retained base label shards do not cover the 2026 training set")
    new_label_meta = dict(old_label_meta)
    new_label_meta["shape"] = [len(extended_dates), EXPECTED_SYMBOL_COUNT, FORWARD_DAYS, 6]
    new_label_meta["shards"] = [
        *retained_shards,
        {
            **_file_material(label_path, dtype="float32", shape=label_shape),
            "date_start_idx": global_date_to_idx[SIGNAL_START],
            "date_end_idx": global_date_to_idx[SIGNAL_END],
        },
    ]
    label_arrays = {
        "future_ohlcva_path": new_label_meta,
        "path_summary": {
            **summary_meta,
            **_file_material(summary_path, dtype="float32", shape=summary_shape),
        },
    }
    new_masks = dict(masks)
    new_masks.update(extended_masks)
    new_masks["price_observed"] = dict(extended_masks["has_bar"])
    new_masks["entry_filled"] = dict(new_masks["entry_buyable"])
    new_execution = {
        "entry_open_raw": {**dict(execution_arrays["entry_open_raw"]), **entry_meta},
        "exit_close_raw": {**dict(execution_arrays["exit_close_raw"]), **exit_meta},
    }
    supplement = dict(base_turnover)
    supplement.update(
        {
            "shape": [len(extended_dates), EXPECTED_SYMBOL_COUNT],
            "date_start": extended_dates[0],
            "date_end": extended_dates[-1],
            "date_values_sha256": _canonical_digest(extended_dates),
            "symbol_values_sha256": _canonical_digest(base["symbol_values"]),
            "log_turnover_pct": turnover_meta["log_turnover_pct"],
            "past20_positive_median": turnover_meta["past20_positive_median"],
        }
    )
    _write_json(supplement_dir / "manifest.json", supplement)

    candidate_staging_path = staging / "development_candidates.parquet"
    supervised_staging_path = staging / "development_supervised.parquet"
    _write_parquet(candidate_staging_path, candidates)
    _write_parquet(supervised_staging_path, supervised)
    material_files = {
        "future_ohlcva_path_2026": _file_material(
            label_path, dtype="float32", shape=label_shape
        ),
        "path_summary": _file_material(
            summary_path, dtype="float32", shape=summary_shape
        ),
        "entry_open_raw": dict(entry_meta),
        "exit_close_raw": dict(exit_meta),
        **{f"mask_{name}": dict(meta) for name, meta in extended_masks.items()},
        "relative_turnover_log": dict(turnover_meta["log_turnover_pct"]),
        "relative_turnover_past20_positive_median": dict(
            turnover_meta["past20_positive_median"]
        ),
        "development_candidates": _file_material(candidate_staging_path),
        "development_supervised": _file_material(supervised_staging_path),
    }

    manifest = dict(base)
    manifest.update(
        {
            "artifact_type": "qdp_v2_sequence_path_pack",
            "artifact_view": {
                "schema_version": 1,
                "view_id": "seq100_2026_dependency_overlay_v1",
                "view_type": "read_only_base_pack_dependency_overlay",
                "source_view": str(BASE_PACK_MANIFEST.resolve()),
            },
            "created_at": _now(),
            "date_values": list(extended_dates),
            "date_count": len(extended_dates),
            "end_date": SIGNAL_END,
            "development_years": [TARGET_YEAR],
            "dependency_padding_complete": True,
            "available_dependency_padding_days": DEPENDENCY_DAYS,
            "max_label_dependency_days": DEPENDENCY_DAYS,
            "label_arrays": label_arrays,
            "execution_arrays": new_execution,
            "masks": new_masks,
            "relative_turnover_supplement": {
                "manifest": str((supplement_dir / "manifest.json").resolve()),
                "log_turnover_pct": turnover_meta["log_turnover_pct"],
                "past20_positive_median": turnover_meta["past20_positive_median"],
            },
            "execution_views": {
                "exit_close_raw_path": {
                    "source_array": "exit_close_raw",
                    "shape": [len(extended_dates), EXPECTED_SYMBOL_COUNT, DEPENDENCY_DAYS],
                    "indexing": "source[signal_date_idx+1:signal_date_idx+81, symbol_idx]",
                    "materialized": False,
                },
                "exit_sellable_path": {
                    "source_mask": "exit_sellable",
                    "shape": [len(extended_dates), EXPECTED_SYMBOL_COUNT, DEPENDENCY_DAYS],
                    "indexing": "source[signal_date_idx+1:signal_date_idx+81, symbol_idx]",
                    "materialized": False,
                },
            },
            "overlay_contract": {
                "base_pack_manifest": str(BASE_PACK_MANIFEST.resolve()),
                "base_pack_sha256": EXPECTED_IMPLEMENTATION_HASHES["base_pack"],
                "qdp_active_as_of": EXPECTED_QDP_AS_OF,
                "signal_window": [SIGNAL_START, SIGNAL_END],
                "candidate_count": int(len(candidates)),
                "symbol_count": EXPECTED_SYMBOL_COUNT,
                "feature_panels_copied": False,
                "jan5_overlap_regression": {
                    "status": "passed",
                    "signal_date": SIGNAL_START,
                    "dependency_end": base_dates[-1],
                    "date_count": DEPENDENCY_DAYS,
                    "materials": [
                        "future_ohlcva_path",
                        "path_summary",
                        "entry_open_raw",
                        "exit_close_raw",
                        "tradable",
                        "has_bar",
                        "exit_sellable",
                        "relative_turnover_log",
                        "relative_turnover_past20_positive_median",
                    ],
                },
                "material_files": material_files,
                "material_files_sha256": _canonical_digest(material_files),
            },
        }
    )
    manifest["overlay_contract"].update(
        {
            "development_candidate_staging": str(candidate_staging_path.resolve()),
            "development_supervised_staging": str(supervised_staging_path.resolve()),
        }
    )
    _write_json(staging / "manifest.json", manifest)
    validation = pack_builder.validate_sequence_pack(staging / "manifest.json")
    if str(validation.get("status", "")) != "ok":
        raise ValueError(f"2026 overlay validation failed: {validation}")
    _validate_overlay_material(staging / "manifest.json")
    return manifest


def _validate_overlay_material(manifest_path: Path) -> dict[str, Any]:
    path = manifest_path.resolve()
    manifest = _read_json(path)
    contract = dict(manifest.get("overlay_contract", {}) or {})
    if (
        str(contract.get("base_pack_sha256", ""))
        != EXPECTED_IMPLEMENTATION_HASHES["base_pack"]
        or str(contract.get("qdp_active_as_of", "")) != EXPECTED_QDP_AS_OF
        or list(contract.get("signal_window", []) or []) != [SIGNAL_START, SIGNAL_END]
        or int(contract.get("candidate_count", -1)) != EXPECTED_CANDIDATE_COUNT
        or int(contract.get("symbol_count", -1)) != EXPECTED_SYMBOL_COUNT
    ):
        raise ValueError("2026 overlay contract boundary drifted")
    overlap = dict(contract.get("jan5_overlap_regression", {}) or {})
    if (
        str(overlap.get("status", "")) != "passed"
        or str(overlap.get("signal_date", "")) != SIGNAL_START
        or str(overlap.get("dependency_end", "")) != "2026-05-08"
        or int(overlap.get("date_count", -1)) != DEPENDENCY_DAYS
    ):
        raise ValueError("2026 overlay Jan 5 overlap contract is incomplete")
    materials = dict(contract.get("material_files", {}) or {})
    if not materials or str(contract.get("material_files_sha256", "")) != _canonical_digest(
        materials
    ):
        raise ValueError("2026 overlay material inventory digest changed")
    root = path.parent
    verified_bytes = 0
    for name, raw_meta in materials.items():
        meta = dict(raw_meta)
        material_path = Path(str(meta.get("path", "") or "")).resolve()
        if root != material_path.parent and root not in material_path.parents:
            raise ValueError(f"overlay material escapes its root: {name}={material_path}")
        if not material_path.is_file():
            raise FileNotFoundError(material_path)
        observed_size = int(material_path.stat().st_size)
        if observed_size != int(meta.get("file_size", -1)):
            raise ValueError(f"overlay material size drift: {name}")
        if _file_sha256(material_path) != str(meta.get("sha256", "")):
            raise ValueError(f"overlay material hash drift: {name}")
        if meta.get("dtype") and meta.get("shape"):
            expected_size = int(
                np.prod(np.asarray(meta["shape"], dtype=np.int64), dtype=np.int64)
            ) * int(np.dtype(str(meta["dtype"])).itemsize)
            if observed_size != expected_size:
                raise ValueError(f"overlay dense-array byte size drift: {name}")
        verified_bytes += observed_size
    candidates = pd.read_parquet(root / "development_candidates.parquet")
    if (
        len(candidates) != EXPECTED_CANDIDATE_COUNT
        or candidates["trade_date"].astype(str).nunique() != EXPECTED_SIGNAL_DATE_COUNT
        or str(candidates["trade_date"].astype(str).min()) != SIGNAL_START
        or str(candidates["trade_date"].astype(str).max()) != SIGNAL_END
        or len(list(manifest.get("symbol_values", []) or [])) != EXPECTED_SYMBOL_COUNT
    ):
        raise ValueError("2026 overlay candidate or symbol shape drifted")
    return {
        "status": "ok",
        "manifest": str(path),
        "material_count": int(len(materials)),
        "verified_material_bytes": int(verified_bytes),
    }


def build_overlay(*, study_path: Path) -> dict[str, Any]:
    if OVERLAY_ROOT.is_dir():
        manifest_path = OVERLAY_ROOT / "manifest.json"
        validation = pack_builder.validate_sequence_pack(manifest_path)
        if str(validation.get("status", "")) != "ok":
            raise ValueError(f"existing 2026 overlay is invalid: {validation}")
        _validate_overlay_material(manifest_path)
        size = _directory_size(OVERLAY_ROOT)
        if size > OVERLAY_SIZE_LIMIT_BYTES:
            raise ValueError("existing 2026 overlay exceeds the registered size cap")
        return _read_json(manifest_path)
    frozen = _assert_frozen_inputs()
    staging = OVERLAY_ROOT.with_name(OVERLAY_ROOT.name + ".staging")
    staging.parent.mkdir(parents=True, exist_ok=True)
    if staging.exists() and not (staging / "_compact" / "manifest.json").is_file():
        _remove_path(staging)
    staging.mkdir(parents=True, exist_ok=True)
    compact_manifest = _compact_pack(staging, study_path)
    compact_turnover = _compact_turnover(staging, compact_manifest)
    manifest = _extract_overlay(
        staging=staging,
        compact_manifest_path=compact_manifest,
        compact_turnover_path=compact_turnover,
        frozen=frozen,
    )
    for temporary_name in ("_compact", "_compact_turnover"):
        temporary_path = staging / temporary_name
        if temporary_path.exists():
            _remove_path(temporary_path)
    promoted = _replace_path_prefix(
        manifest, str(staging.resolve()), str(OVERLAY_ROOT.resolve())
    )
    promoted_contract = dict(promoted["overlay_contract"])
    promoted_contract["material_files_sha256"] = _canonical_digest(
        promoted_contract["material_files"]
    )
    promoted["overlay_contract"] = promoted_contract
    supplement_manifest_path = (
        staging / "supplements" / "relative_turnover_v1" / "manifest.json"
    )
    promoted_supplement = _replace_path_prefix(
        _read_json(supplement_manifest_path),
        str(staging.resolve()),
        str(OVERLAY_ROOT.resolve()),
    )
    _write_json(supplement_manifest_path, promoted_supplement)
    _write_json(staging / "manifest.json", promoted)
    size = _directory_size(staging)
    if size > OVERLAY_SIZE_LIMIT_BYTES:
        raise ValueError(
            f"2026 overlay exceeds size cap: {size / (1024**3):.3f} GiB"
        )
    os.replace(staging, OVERLAY_ROOT)
    validation = pack_builder.validate_sequence_pack(OVERLAY_ROOT / "manifest.json")
    if str(validation.get("status", "")) != "ok":
        raise ValueError(f"promoted 2026 overlay is invalid: {validation}")
    _validate_overlay_material(OVERLAY_ROOT / "manifest.json")
    return _read_json(OVERLAY_ROOT / "manifest.json")


def _research_contract_binding(study_path: Path, study: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "path": str(study_path.resolve()),
        "contract_id": STUDY_ID,
        "contract_sha256": str(study["contract_sha256"]),
        "contract_file_sha256": str(study["contract_sha256"]),
    }


def build_2026_fold_view(
    *, study_path: Path, study: dict[str, Any], overlay: Mapping[str, Any]
) -> tuple[Path, dict[str, Any]]:
    view_path = STUDY_ROOT / "views" / "development_2026.json"
    if view_path.is_file():
        verification = walkforward.verify_development_walkforward_view(view_path)
        existing = _read_json(view_path)
        sample_index = Path(str(existing.get("sample_index_path", "") or ""))
        candidate_index = Path(str(existing.get("candidate_index_path", "") or ""))
        has_redundant_label_index = bool(
            sample_index.is_file()
            and "label_symbol_idx" in pq.ParquetFile(sample_index).schema.names
        ) or bool(
            candidate_index.is_file()
            and "label_symbol_idx" in pq.ParquetFile(candidate_index).schema.names
        )
        if str(verification.get("status", "")) == "ok" and not has_redundant_label_index:
            return view_path, existing
        if view_path.resolve().parent != (STUDY_ROOT / "views").resolve():
            raise ValueError(f"refusing to replace unexpected fold view: {view_path}")
        view_path.unlink()
    candidates = pd.read_parquet(OVERLAY_ROOT / "development_candidates.parquet")
    supervised = pd.read_parquet(OVERLAY_ROOT / "development_supervised.parquet")
    _validate_candidate_membership(base=_read_json(BASE_PACK_MANIFEST), candidates=candidates)
    sample_path, candidate_path, index_audit = _build_fold_indexes(
        base=_read_json(BASE_PACK_MANIFEST),
        candidates=candidates,
        supervised=supervised,
        study_root=STUDY_ROOT,
    )
    date_values = [str(item) for item in list(overlay["date_values"])]
    development_start_idx = date_values.index(SIGNAL_START)
    development_end_idx = date_values.index(SIGNAL_END)
    normalization = walkforward._fit_development_normalization(
        overlay,
        start_idx=0,
        end_idx_exclusive=development_start_idx,
        date_values=date_values,
    )
    contract = {
        "schema_version": 1,
        "method": "expanding_train_development_walkforward",
        "split_roles": {"fit": "train", "evaluation": "development"},
        "train_start_year": 2010,
        "development_year": TARGET_YEAR,
        "development_start": SIGNAL_START,
        "development_end": SIGNAL_END,
        "source_padding_end": EXPECTED_QDP_AS_OF,
        "source_padding_trade_date_count": DEPENDENCY_DAYS,
        "development_start_date_idx": development_start_idx,
        "forward_days": FORWARD_DAYS,
        "execution_tail_days": EXECUTION_TAIL_DAYS,
        "max_label_dependency_days": DEPENDENCY_DAYS,
        "safe_train_signal_end": SAFE_TRAIN_SIGNAL_END,
        "max_train_dependency_end": PURGE_END,
        "purge_rule": "max_label_dependency_date_idx < development_start_date_idx",
        "purged_row_count": EXPECTED_PURGED_ROW_COUNT,
        "purged_signal_date_count": EXPECTED_PURGED_DATE_COUNT,
        "purged_signal_start": PURGE_START,
        "purged_signal_end": PURGE_END,
        "label_dependency_overlap_count": 0,
        "candidate_universe_rule": "signal_day_input_valid_and_signal_eligible",
        "candidate_count": int(len(candidates)),
        "supervised_development_count": int(len(supervised)),
        "unsupervised_candidate_count": int(len(candidates) - len(supervised)),
        "normalization_cutoff_exclusive": SIGNAL_START,
    }
    view = dict(overlay)
    view.pop("test_years", None)
    view.pop("oos_years", None)
    view.update(
        {
            "created_at": _now(),
            "start_date": "2010-01-01",
            "end_date": SIGNAL_END,
            "train_years": list(range(2010, 2026)),
            "development_years": [TARGET_YEAR],
            "split_roles": {"fit": "train", "evaluation": "development"},
            "sample_index_path": str(sample_path.resolve()),
            "sample_count": int(index_audit["train_row_count"] + len(supervised)),
            "sample_count_by_split": {
                "train": int(index_audit["train_row_count"]),
                "development": int(len(supervised)),
            },
            "candidate_index_path": str(candidate_path.resolve()),
            "candidate_count": int(len(candidates)),
            "candidate_count_by_split": {"development": int(len(candidates))},
            "normalization": normalization,
            "artifact_view": {
                "schema_version": 1,
                "view_id": "seq100_path60_todayclose_ohlcva_development_2026",
                "view_type": "development_walkforward_research_store_view",
                "source_view": str((OVERLAY_ROOT / "manifest.json").resolve()),
                "development_audit": contract,
            },
            "development_walkforward": contract,
            "source_view_provenance": walkforward._source_view_provenance(
                OVERLAY_ROOT / "manifest.json", overlay
            ),
            "research_contract": _research_contract_binding(study_path, study),
            "development_contract": _research_contract_binding(study_path, study),
            "max_label_dependency_days": DEPENDENCY_DAYS,
            "legal_exit_contract": dict(LEGAL_EXIT_CONTRACT),
        }
    )
    view["development_fold_training_contract"] = (
        walkforward._compute_development_fold_training_contract(view)
    )
    _write_json(view_path, view)
    verification = walkforward.verify_development_walkforward_view(view_path)
    if str(verification.get("status", "")) != "ok":
        raise ValueError(f"2026 fold verification failed: {verification}")
    return view_path, view


def prepare_2026_fold(*, study_root: Path = STUDY_ROOT) -> dict[str, Any]:
    if study_root.resolve() != STUDY_ROOT.resolve():
        raise ValueError("2026 fold uses the registered canonical study root")
    _assert_no_other_research_process()
    frozen = _assert_frozen_inputs()
    study_path, study = _load_or_create_study()
    _update_runtime(study_path, study, status="preparing", error=None)
    try:
        overlay = build_overlay(study_path=study_path)
        view_path, view = build_2026_fold_view(
            study_path=study_path, study=study, overlay=overlay
        )
        study = _read_json(study_path)
        study["material"] = {
            "base_pack_manifest": str(BASE_PACK_MANIFEST.resolve()),
            "base_pack_sha256": EXPECTED_IMPLEMENTATION_HASHES["base_pack"],
            "overlay_manifest": str((OVERLAY_ROOT / "manifest.json").resolve()),
            "overlay_manifest_sha256": _file_sha256(OVERLAY_ROOT / "manifest.json"),
            "overlay_size_bytes": _directory_size(OVERLAY_ROOT),
            "fold_views": {str(TARGET_YEAR): str(view_path.resolve())},
            "fold_view_sha256": _file_sha256(view_path),
            "candidate_index_path": str(Path(view["candidate_index_path"]).resolve()),
            "candidate_index_sha256": _file_sha256(Path(view["candidate_index_path"])),
            "sample_index_path": str(Path(view["sample_index_path"]).resolve()),
            "sample_index_sha256": _file_sha256(Path(view["sample_index_path"])),
            "frozen_inputs": frozen,
            "qdp_changed": False,
            "base_pack_changed": False,
            "provider_called": False,
            "live_execution_changed": False,
        }
        study["runtime"] = {
            **dict(study.get("runtime", {}) or {}),
            "status": "prepared",
            "updated_at": _now(),
            "error": None,
            "current_task": None,
        }
        _write_json(study_path, study)
        return study
    except Exception as exc:
        study = _read_json(study_path)
        _update_runtime(study_path, study, status="prepare_failed", error=str(exc))
        raise


def _validate_prepared_2026_study(*, require_runs: bool = False) -> dict[str, Any]:
    study_path = STUDY_ROOT / "study.json"
    study = _read_json(study_path)
    material = dict(study.get("material", {}) or {})
    frozen_now = _assert_frozen_inputs()
    frozen_then = dict(material.get("frozen_inputs", {}) or {})
    for field in (
        "implementation_sha256",
        "qdp_active_as_of",
        "qdp_active_manifest_sha256",
        "qdp_full_audit_sha256",
        "qdp_preflight",
    ):
        if frozen_now.get(field) != frozen_then.get(field):
            raise ValueError(f"prepared 2026 study frozen input drift: {field}")
    overlay_manifest = Path(str(material.get("overlay_manifest", "") or "")).resolve()
    if (
        overlay_manifest != (OVERLAY_ROOT / "manifest.json").resolve()
        or _file_sha256(overlay_manifest)
        != str(material.get("overlay_manifest_sha256", ""))
        or _directory_size(OVERLAY_ROOT) > OVERLAY_SIZE_LIMIT_BYTES
    ):
        raise ValueError("prepared 2026 overlay binding drifted")
    overlay_validation = _validate_overlay_material(overlay_manifest)
    view_path = Path(str(dict(material.get("fold_views", {}) or {}).get("2026", ""))).resolve()
    if (
        not view_path.is_file()
        or _file_sha256(view_path) != str(material.get("fold_view_sha256", ""))
    ):
        raise ValueError("prepared 2026 fold view binding drifted")
    verification = walkforward.verify_development_walkforward_view(view_path)
    if str(verification.get("status", "")) != "ok":
        raise ValueError(f"prepared 2026 fold view is invalid: {verification}")
    view = _read_json(view_path)
    normalization = dict(view.get("normalization", {}) or {})
    audit = dict(view.get("development_walkforward", {}) or {})
    if (
        str(normalization.get("fit_date_end_exclusive", "")) != SIGNAL_START
        or str(audit.get("normalization_cutoff_exclusive", "")) != SIGNAL_START
        or int(audit.get("candidate_count", -1)) != EXPECTED_CANDIDATE_COUNT
        or int(audit.get("purged_row_count", -1)) != EXPECTED_PURGED_ROW_COUNT
        or int(audit.get("purged_signal_date_count", -1))
        != EXPECTED_PURGED_DATE_COUNT
    ):
        raise ValueError("prepared 2026 fold boundary or normalization drifted")
    for material_name, view_name in (
        ("candidate_index", "candidate_index_path"),
        ("sample_index", "sample_index_path"),
    ):
        index_path = Path(str(view[view_name])).resolve()
        if (
            index_path != Path(str(material[f"{material_name}_path"])).resolve()
            or _file_sha256(index_path)
            != str(material[f"{material_name}_sha256"])
        ):
            raise ValueError(f"prepared 2026 {material_name} binding drifted")
    if require_runs:
        for profile in PROFILE_ORDER:
            complete, partial = _run_dirs(profile)
            if len(complete) != 1 or partial:
                raise ValueError(f"2026 profile run is incomplete: {profile}")
            _validate_2026_run_summary(
                complete[0] / "sequence_path_training_summary.json", profile
            )
    return {
        "status": "ok",
        "study": str(study_path.resolve()),
        "overlay": overlay_validation,
        "fold_verification": verification,
    }


def _profile_from_study(study: Mapping[str, Any], profile_name: str) -> Any:
    payload = dict(dict(study["contract"])["profiles"][profile_name])
    for key in ("store_view", "output_root", "run_tag"):
        payload.pop(key, None)
    return structured.TodayClosePathOnlyProfile(**payload)


def _run_dirs(profile_name: str) -> tuple[list[Path], list[Path]]:
    root = STUDY_ROOT / "runs" / "development"
    matches = sorted(
        root.glob(f"seq100_structured_{profile_name}_{TARGET_YEAR}_seed{SEED}_*")
    )
    complete = [item for item in matches if (item / "sequence_path_training_summary.json").is_file()]
    partial = [item for item in matches if item not in complete]
    return complete, partial


def _validate_2026_run_summary(path: Path, profile_name: str) -> dict[str, Any]:
    summary = structured._validate_run_summary(path, TARGET_YEAR)
    expected_model = (
        "gru_ohlcva_aux_path_value"
        if profile_name == "legal_flat_baseline"
        else "gru_structured_joint_turnover"
    )
    if str(dict(summary["resolved_training_config"])["model_type"]) != expected_model:
        raise ValueError(f"2026 {profile_name} run used the wrong model")
    if str(summary.get("candidate_index_sha256", "")) != _file_sha256(
        Path(_read_json(STUDY_ROOT / "study.json")["material"]["candidate_index_path"])
    ):
        raise ValueError("2026 training summary candidate binding drifted")
    record = freshness._checkpoint_metadata(
        path.resolve().parent, profile=profile_name, vintage=TARGET_YEAR
    )
    older = freshness._checkpoint_metadata(
        _checkpoint_run_dir(profile_name, 2025), profile=profile_name, vintage=2025
    )
    if str(record["architecture_sha256"]) != str(older["architecture_sha256"]):
        raise ValueError(f"2026 {profile_name} checkpoint architecture drifted")
    if str(record["normalization_fit_date_end_exclusive"]) != SIGNAL_START:
        raise ValueError(f"2026 {profile_name} normalization cutoff drifted")
    view_path = Path(
        str(_read_json(STUDY_ROOT / "study.json")["material"]["fold_views"]["2026"])
    )
    view_contract = dict(
        _read_json(view_path).get("development_fold_training_contract", {}) or {}
    )
    if str(record["checkpoint_fold_contract_sha256"]) != str(
        view_contract.get("sha256", "")
    ):
        raise ValueError(f"2026 {profile_name} checkpoint fold contract drifted")
    return summary


def run_2026_fold(*, study_root: Path = STUDY_ROOT, max_tasks: int = 0) -> dict[str, Any]:
    if int(max_tasks) < 0:
        raise ValueError("max_tasks must be non-negative")
    if study_root.resolve() != STUDY_ROOT.resolve():
        raise ValueError("2026 fold uses the registered canonical study root")
    _assert_no_other_research_process()
    study_path = STUDY_ROOT / "study.json"
    if not study_path.is_file():
        prepare_2026_fold()
    _validate_prepared_2026_study(require_runs=False)
    study = _read_json(study_path)
    if str(dict(study.get("runtime", {}) or {}).get("status", "")) not in {
        "prepared",
        "running",
        "run_failed",
        "runs_completed",
    }:
        raise ValueError("2026 study is not runnable")
    view_path = Path(str(dict(study["material"])["fold_views"][str(TARGET_YEAR)]))
    completed_tasks: list[str] = []
    launched = 0
    _update_runtime(study_path, study, status="running", error=None)
    try:
        for profile_name in PROFILE_ORDER:
            task = f"{profile_name}:{TARGET_YEAR}"
            complete, partial = _run_dirs(profile_name)
            if len(complete) > 1:
                raise ValueError(f"task {task} has multiple completed runs")
            if complete:
                _validate_2026_run_summary(
                    complete[0] / "sequence_path_training_summary.json", profile_name
                )
                completed_tasks.append(task)
                continue
            for partial_path in partial:
                resolved = partial_path.resolve()
                expected_parent = (STUDY_ROOT / "runs" / "development").resolve()
                if expected_parent not in resolved.parents:
                    raise ValueError(f"refusing to remove unexpected partial run: {partial_path}")
                shutil.rmtree(partial_path)
            if int(max_tasks) > 0 and launched >= int(max_tasks):
                break
            base_profile = _profile_from_study(study, profile_name)
            profile = replace(
                base_profile,
                store_view=view_path,
                output_root=STUDY_ROOT / "runs" / "development",
                run_tag=f"seq100_structured_{profile_name}_{TARGET_YEAR}_seed{SEED}",
                epochs=10,
                seed=SEED,
                top_k="1,3,5,10",
                prediction_mode="compact",
                evaluation_mode="development",
                early_stopping_patience=2,
                early_stopping_min_delta=0.0,
                early_stopping_metric=training.EARLY_STOPPING_METRIC_DEVELOPMENT_PRICE_TOTAL_LOSS,
                early_stopping_mode="min",
                min_complete_epochs=1,
                max_samples_per_split=0,
            )
            command = [
                str(PYTHON),
                "-m",
                "daily_research.path_policy.qdp_v2_sequence_path_training",
                *build_todayclose_path_only_train_argv(profile),
                "--development-contract",
                str(study_path),
                "--json",
            ]
            current = _read_json(study_path)
            _update_runtime(
                study_path,
                current,
                status="running",
                completed_tasks=completed_tasks,
                current_task=task,
            )
            _guarded_command(
                command,
                log_path=STUDY_ROOT / "logs" / f"memory_guard_{profile_name}_2026.json",
            )
            complete, _partial = _run_dirs(profile_name)
            if len(complete) != 1:
                raise RuntimeError(f"task {task} did not produce exactly one completed run")
            _validate_2026_run_summary(
                complete[0] / "sequence_path_training_summary.json", profile_name
            )
            completed_tasks.append(task)
            launched += 1
            current = _read_json(study_path)
            _update_runtime(
                study_path,
                current,
                status="running",
                completed_tasks=completed_tasks,
                current_task=None,
            )
        final_status = "runs_completed" if len(completed_tasks) == len(PROFILE_ORDER) else "running"
        current = _read_json(study_path)
        _update_runtime(
            study_path,
            current,
            status=final_status,
            completed_tasks=completed_tasks,
            current_task=None,
        )
    except Exception as exc:
        current = _read_json(study_path)
        _update_runtime(
            study_path,
            current,
            status="run_failed",
            completed_tasks=completed_tasks,
            current_task=None,
            error=str(exc),
        )
        raise
    return status_2026_fold()


def status_2026_fold() -> dict[str, Any]:
    study_path = STUDY_ROOT / "study.json"
    study = _read_json(study_path)
    tasks: dict[str, Any] = {}
    for profile in PROFILE_ORDER:
        complete, partial = _run_dirs(profile)
        tasks[f"{profile}:{TARGET_YEAR}"] = {
            "completed_run": str(complete[0]) if len(complete) == 1 else None,
            "partial_runs": [str(item) for item in partial],
        }
    return {
        "study": str(study_path.resolve()),
        "runtime": dict(study.get("runtime", {}) or {}),
        "tasks": tasks,
    }


def _base_run_dirs(profile: str, vintage: int) -> tuple[list[Path], list[Path]]:
    return structured._run_dirs(BASE_STUDY_ROOT, profile, vintage)


def _checkpoint_run_dir(profile: str, vintage: int) -> Path:
    complete, _partial = (
        _run_dirs(profile) if int(vintage) == TARGET_YEAR else _base_run_dirs(profile, vintage)
    )
    if len(complete) != 1:
        raise ValueError(f"expected one completed checkpoint run for {profile}/{vintage}")
    return complete[0]


def _candidate_material(source_view: Mapping[str, Any]) -> dict[str, Any]:
    candidate_path = Path(str(source_view["candidate_index_path"])).resolve()
    candidates = pd.read_parquet(candidate_path)
    required = {
        "candidate_id",
        "trade_date",
        "date_idx",
        "symbol_idx",
        "symbol",
        "entry_filled",
        "label_valid",
        "price_label_valid",
        "va_aux_valid",
    }
    missing = sorted(required.difference(candidates.columns))
    if missing:
        raise ValueError(f"2026 candidate index is missing columns: {missing}")
    if candidates.empty or bool(candidates["candidate_id"].duplicated().any()):
        raise ValueError("2026 candidate index is empty or has duplicate candidate IDs")
    candidate_ids = candidates["candidate_id"].to_numpy(dtype=np.int64, copy=False)
    if not np.array_equal(candidate_ids, np.arange(len(candidates), dtype=np.int64)):
        raise ValueError("2026 candidate IDs are not the registered contiguous ordering")
    key_columns = ["candidate_id", "trade_date", "date_idx", "symbol_idx", "symbol"]
    coverage_columns = [
        "trade_date",
        "entry_filled",
        "label_valid",
        "price_label_valid",
        "va_aux_valid",
    ]
    key_hashes = pd.util.hash_pandas_object(
        candidates[key_columns], index=False, categorize=True
    ).to_numpy(dtype=np.uint64, copy=False)
    coverage = (
        candidates[coverage_columns]
        .groupby("trade_date", sort=True)
        .agg(
            candidate_count=("entry_filled", "size"),
            entry_filled_count=("entry_filled", "sum"),
            label_valid_count=("label_valid", "sum"),
            price_label_valid_count=("price_label_valid", "sum"),
            va_aux_valid_count=("va_aux_valid", "sum"),
        )
        .reset_index()
    )
    result = {
        "path": str(candidate_path),
        "file_sha256": _file_sha256(candidate_path),
        "row_count": int(len(candidates)),
        "date_count": int(candidates["trade_date"].nunique()),
        "trade_date_start": str(candidates["trade_date"].min()),
        "trade_date_end": str(candidates["trade_date"].max()),
        "pack_symbol_count": int(len(list(source_view.get("symbol_values", []) or []))),
        "candidate_key_sha256": hashlib.sha256(key_hashes.tobytes()).hexdigest(),
        "candidate_key_hash_policy": "pandas_hash_object_uint64_v1",
        "label_coverage_sha256": _canonical_digest(coverage.to_dict("records")),
        "label_coverage_by_date": coverage.to_dict("records"),
    }
    if (
        result["row_count"] != EXPECTED_CANDIDATE_COUNT
        or result["date_count"] != EXPECTED_SIGNAL_DATE_COUNT
        or result["trade_date_start"] != SIGNAL_START
        or result["trade_date_end"] != SIGNAL_END
        or result["pack_symbol_count"] != EXPECTED_SYMBOL_COUNT
    ):
        raise ValueError(f"2026 candidate material drifted: {result}")
    return result


def _fairness_material(
    *, source_view_path: Path, source_view: Mapping[str, Any], candidate: Mapping[str, Any]
) -> dict[str, Any]:
    fold_contract = dict(source_view.get("development_fold_training_contract", {}) or {})
    if str(fold_contract.get("candidate_index_sha256", "")) != str(
        candidate["file_sha256"]
    ):
        raise ValueError("2026 candidate index no longer matches the fold contract")
    label_payload = {
        "label_arrays": source_view.get("label_arrays", {}),
        "label_semantics": source_view.get("label_semantics", {}),
        "mask_views": source_view.get("mask_views", {}),
        "candidate_label_coverage_sha256": candidate["label_coverage_sha256"],
    }
    execution_payload = {
        "execution_arrays": source_view.get("execution_arrays", {}),
        "execution_views": source_view.get("execution_views", {}),
        "execution_cost_contract": source_view.get("execution_cost_contract", {}),
        "terminal_execution_contract": source_view.get(
            "terminal_execution_contract", {}
        ),
        "execution_tail_days": source_view.get("execution_tail_days"),
        "exit_sellable_mask": dict(source_view.get("masks", {}) or {}).get(
            "exit_sellable", {}
        ),
        "entry_filled_mask": dict(source_view.get("masks", {}) or {}).get(
            "entry_filled", {}
        ),
    }
    auxiliary_payload = {
        "relative_turnover_supplement": source_view.get(
            "relative_turnover_supplement", {}
        ),
        "input_feature_channels": source_view.get("feature_channels", {}),
        "input_mask_features": dict(source_view.get("data_semantics", {}) or {}).get(
            "model_input_mask_features", []
        ),
    }
    evaluation_rules = {
        "candidate_membership": "exact_2026_candidate_index_without_label_filtering",
        "ranking": "predicted_price_path_only",
        "legal_exit": dict(LEGAL_EXIT_CONTRACT),
        "top_k": list(TOP_K_VALUES),
        "costs_and_execution": "exact_2026_view_contract",
        "terminal_recovery": source_view.get("terminal_execution_contract", {}),
        "score_only_tail_included": False,
    }
    evaluation_implementation = {
        "aggregation_module": str(Path(__file__).resolve()),
        "aggregation_module_sha256": _file_sha256(Path(__file__).resolve()),
        "prediction_module": str(Path(training.__file__).resolve()),
        "prediction_module_sha256": _file_sha256(Path(training.__file__).resolve()),
        "execution_module": str(Path(candidate_execution.__file__).resolve()),
        "execution_module_sha256": _file_sha256(
            Path(candidate_execution.__file__).resolve()
        ),
        "prediction_function": "qdp_v2_sequence_path_training._predict_split",
        "execution_function": "seq100_candidate_execution.evaluate_candidate_execution",
        "metric_function": "seq100_2026_fold_comparison._evaluation_metrics_2026",
    }
    overlay_contract = dict(source_view.get("overlay_contract", {}) or {})
    material: dict[str, Any] = {
        "target_year": TARGET_YEAR,
        "source_view_path": str(source_view_path.resolve()),
        "source_view_sha256": _file_sha256(source_view_path),
        "overlay_manifest_sha256": _file_sha256(OVERLAY_ROOT / "manifest.json"),
        "base_pack_sha256": EXPECTED_IMPLEMENTATION_HASHES["base_pack"],
        "candidate_index_path": str(candidate["path"]),
        "candidate_index_sha256": str(candidate["file_sha256"]),
        "candidate_key_sha256": str(candidate["candidate_key_sha256"]),
        "candidate_count": int(candidate["row_count"]),
        "candidate_date_count": int(candidate["date_count"]),
        "candidate_trade_date_start": str(candidate["trade_date_start"]),
        "candidate_trade_date_end": str(candidate["trade_date_end"]),
        "label_coverage_sha256": str(candidate["label_coverage_sha256"]),
        "label_material_sha256": _canonical_digest(label_payload),
        "execution_material_sha256": _canonical_digest(execution_payload),
        "auxiliary_material_sha256": _canonical_digest(auxiliary_payload),
        "overlay_material_files_sha256": str(
            overlay_contract.get("material_files_sha256", "")
        ),
        "execution_cost_contract_sha256": candidate_execution.execution_cost_contract_sha256(
            source_view
        ),
        "date_values_sha256": _canonical_digest(source_view.get("date_values", [])),
        "symbol_values_sha256": _canonical_digest(source_view.get("symbol_values", [])),
        "evaluation_rules": evaluation_rules,
        "evaluation_implementation": evaluation_implementation,
    }
    material["evaluation_contract_sha256"] = _canonical_digest(
        {
            "rules": evaluation_rules,
            "implementation": evaluation_implementation,
            "execution_cost_contract_sha256": material[
                "execution_cost_contract_sha256"
            ],
            "label_material_sha256": material["label_material_sha256"],
            "execution_material_sha256": material["execution_material_sha256"],
            "auxiliary_material_sha256": material["auxiliary_material_sha256"],
        }
    )
    return material


def _comparison_task_id(profile: str, vintage: int) -> str:
    return f"{profile}_checkpoint_{int(vintage)}_on_{TARGET_YEAR}"


def _evaluation_view(
    *,
    source_view: Mapping[str, Any],
    profile: str,
    vintage: int,
    checkpoint: Mapping[str, Any],
    fairness_material: Mapping[str, Any],
) -> dict[str, Any]:
    view = json.loads(json.dumps(dict(source_view), ensure_ascii=False))
    source_fold = dict(view.pop("development_fold_training_contract", {}) or {})
    view["artifact_type"] = "seq100_checkpoint_vintage_2026_evaluation_view"
    view["artifact_view"] = (
        f"{ANALYSIS_ID}_{profile}_{int(vintage)}_on_{TARGET_YEAR}"
    )
    view["normalization"] = dict(checkpoint["normalization"])
    view["checkpoint_vintage_evaluation"] = {
        "analysis_id": ANALYSIS_ID,
        "profile": str(profile),
        "checkpoint_vintage": int(vintage),
        "target_year": TARGET_YEAR,
        "checkpoint_path": str(checkpoint["checkpoint_path"]),
        "checkpoint_sha256": str(checkpoint["checkpoint_sha256"]),
        "checkpoint_fold_contract_sha256": str(
            checkpoint["checkpoint_fold_contract_sha256"]
        ),
        "normalization_sha256": str(checkpoint["normalization_sha256"]),
        "normalization_source": "checkpoint.fold_training_contract.payload.normalization",
        "source_2026_fold_contract_sha256": str(source_fold.get("sha256", "")),
        "candidate_index_sha256": str(fairness_material["candidate_index_sha256"]),
        "label_material_sha256": str(fairness_material["label_material_sha256"]),
        "execution_material_sha256": str(
            fairness_material["execution_material_sha256"]
        ),
        "future_label_required_for_candidate_membership": False,
    }
    return view


def _write_or_validate_json(path: Path, payload: Mapping[str, Any]) -> None:
    if path.is_file():
        if _read_json(path) != dict(payload):
            raise ValueError(f"existing 2026 comparison artifact changed: {path}")
        return
    _write_json(path, payload)


def prepare_2026_comparison(*, output_root: Path = ANALYSIS_ROOT) -> dict[str, Any]:
    _validate_prepared_2026_study(require_runs=True)
    study_path = STUDY_ROOT / "study.json"
    study = _read_json(study_path)
    if str(dict(study.get("runtime", {}) or {}).get("status", "")) != "runs_completed":
        raise ValueError("both 2026 profile runs must complete before vintage evaluation")
    source_view_path = Path(
        str(dict(study["material"])["fold_views"][str(TARGET_YEAR)])
    ).resolve()
    source_view = _read_json(source_view_path)
    candidate = _candidate_material(source_view)
    fairness_material = _fairness_material(
        source_view_path=source_view_path,
        source_view=source_view,
        candidate=candidate,
    )
    root = output_root.resolve()
    checkpoint_records: dict[tuple[str, int], dict[str, Any]] = {}
    profile_architecture: dict[str, str] = {}
    for profile in PROFILE_ORDER:
        for vintage in CHECKPOINT_VINTAGES:
            record = freshness._checkpoint_metadata(
                _checkpoint_run_dir(profile, vintage),
                profile=profile,
                vintage=vintage,
            )
            checkpoint_records[(profile, vintage)] = record
            previous = profile_architecture.setdefault(
                profile, str(record["architecture_sha256"])
            )
            if previous != str(record["architecture_sha256"]):
                raise ValueError(
                    f"architecture or training config drift within profile {profile}"
                )
    tasks: dict[str, Any] = {}
    for profile in PROFILE_ORDER:
        for vintage in CHECKPOINT_VINTAGES:
            record = checkpoint_records[(profile, vintage)]
            task_id = _comparison_task_id(profile, vintage)
            view_path = root / "views" / f"{task_id}.json"
            view = _evaluation_view(
                source_view=source_view,
                profile=profile,
                vintage=vintage,
                checkpoint=record,
                fairness_material=fairness_material,
            )
            _write_or_validate_json(view_path, view)
            task_dir = root / "tasks" / task_id
            tasks[task_id] = {
                "task_id": task_id,
                "profile": profile,
                "checkpoint_vintage": int(vintage),
                "target_year": TARGET_YEAR,
                "run_dir": str(record["run_dir"]),
                "checkpoint_path": str(record["checkpoint_path"]),
                "checkpoint_sha256": str(record["checkpoint_sha256"]),
                "checkpoint_summary_sha256": str(record["summary_sha256"]),
                "checkpoint_fold_contract_sha256": str(
                    record["checkpoint_fold_contract_sha256"]
                ),
                "normalization_sha256": str(record["normalization_sha256"]),
                "normalization_fit_date_end_exclusive": str(
                    record["normalization_fit_date_end_exclusive"]
                ),
                "architecture_sha256": str(record["architecture_sha256"]),
                "view_path": str(view_path.resolve()),
                "view_sha256": _file_sha256(view_path),
                "output_dir": str(task_dir.resolve()),
                "result_path": str((task_dir / "evaluation.json").resolve()),
                "inference_required": bool(vintage != TARGET_YEAR),
                "result_source": (
                    "fresh_inference"
                    if vintage != TARGET_YEAR
                    else "reuse_2026_training_evaluation_if_consistent"
                ),
            }
    contract_payload = {
        "schema_version": 1,
        "artifact_type": "seq100_checkpoint_vintage_2026_contract",
        "analysis_id": ANALYSIS_ID,
        "study_root": str(STUDY_ROOT.resolve()),
        "study_contract_sha256": str(study["contract_sha256"]),
        "target_year": TARGET_YEAR,
        "checkpoint_vintages": list(CHECKPOINT_VINTAGES),
        "profiles": list(PROFILE_ORDER),
        "seed": SEED,
        "top_k": list(TOP_K_VALUES),
        "fairness_material": fairness_material,
        "profile_architecture_sha256": profile_architecture,
        "tasks": tasks,
        "gate_contract": {
            "primary": "2026 Top3 executable base alpha must exceed all 2023-2025 checkpoints within profile",
            "breadth": "at least two of Top1, Top5 and Top10 executable base alpha must exceed all older checkpoints",
            "ranking_support": "Rank IC or Top3 opportunity alpha must exceed all older checkpoints",
            "path_support": "exit regret or OHLC path MAE must be lower than all older checkpoints",
            "overall": "both profiles and all fairness/coverage/reuse checks must pass",
        },
        "uncertainty_contract": {
            "signal_date_count": EXPECTED_SIGNAL_DATE_COUNT,
            "forward_dependency_days": FORWARD_DAYS,
            "formal_moving_block_interval": False,
            "reason": "signal-date count is shorter than the overlapping 60-day outcome dependency",
            "paired_daily_deltas_are_descriptive_only": True,
        },
        "protected_boundaries": {
            "qdp_updated": False,
            "provider_called": False,
            "base_pack_mutated": False,
            "live_execution_changed": False,
            "score_only_tail_included": False,
        },
    }
    contract = {**contract_payload, "contract_sha256": _canonical_digest(contract_payload)}
    contract_path = root / "contract.json"
    _write_or_validate_json(contract_path, contract)
    return contract


def _load_comparison_contract(path: Path) -> dict[str, Any]:
    _assert_frozen_inputs()
    _validate_overlay_material(OVERLAY_ROOT / "manifest.json")
    contract = _read_json(path.resolve())
    declared = str(contract.get("contract_sha256", "") or "")
    payload = {key: value for key, value in contract.items() if key != "contract_sha256"}
    if declared != _canonical_digest(payload):
        raise ValueError("2026 comparison contract digest changed")
    if str(contract.get("analysis_id", "")) != ANALYSIS_ID:
        raise ValueError("wrong 2026 comparison contract")
    implementation = dict(
        dict(contract.get("fairness_material", {}) or {}).get(
            "evaluation_implementation", {}
        )
        or {}
    )
    for prefix in ("aggregation", "prediction", "execution"):
        source = Path(str(implementation.get(f"{prefix}_module", "") or "")).resolve()
        declared_hash = str(implementation.get(f"{prefix}_module_sha256", "") or "")
        if not source.is_file() or _file_sha256(source) != declared_hash:
            raise ValueError(f"2026 comparison {prefix} implementation changed")
    return contract


def _task_artifact_paths(task: Mapping[str, Any]) -> dict[str, Path]:
    output_dir = Path(str(task["output_dir"])).resolve()
    return {
        "topk_metrics_csv": output_dir / "topk_metrics.csv",
        "daily_rank_ic_csv": output_dir / "daily_rank_ic.csv",
        "daily_topk_metrics_csv": output_dir / "daily_topk_metrics.csv",
        "topk_candidates_parquet": output_dir / "topk_candidates.parquet",
    }


def _finite_topk_metric(topk: pd.DataFrame, top_k: int, column: str) -> float:
    if column not in topk.columns:
        raise ValueError(f"TopK aggregate is missing {column}")
    row = topk[topk["top_k"].astype(int).eq(int(top_k))]
    if len(row) != 1:
        raise ValueError(f"expected one Top{int(top_k)} aggregate row")
    value = float(pd.to_numeric(row.iloc[0][column], errors="coerce"))
    if not math.isfinite(value):
        raise ValueError(f"Top{int(top_k)} metric {column} is not finite")
    return value


def _evaluation_metrics_2026(
    *, topk: pd.DataFrame, daily_topk: pd.DataFrame, split_metrics: Mapping[str, Any]
) -> dict[str, Any]:
    metrics = freshness._evaluation_metrics(
        topk=topk, daily_topk=daily_topk, split_metrics=split_metrics
    )
    for top_k in TOP_K_VALUES:
        prefix = f"top{int(top_k)}"
        for output_name, column in (
            ("selected_stress", "selected_net_realized_plan_return_stress"),
            ("universe_stress", "universe_net_realized_plan_return_stress"),
            ("selected_label_coverage", "selected_label_coverage"),
            ("universe_label_coverage", "universe_label_coverage"),
            (
                "selected_opportunity_coverage",
                "selected_opportunity_value_coverage",
            ),
            (
                "universe_opportunity_coverage",
                "universe_opportunity_value_coverage",
            ),
            ("selected_exit_regret", "selected_oracle_regret"),
            ("universe_exit_regret", "universe_oracle_regret"),
            ("selected_exit_regret_coverage", "selected_oracle_regret_coverage"),
            ("universe_exit_regret_coverage", "universe_oracle_regret_coverage"),
            ("selected_entry_fill_rate", "selected_entry_fill_rate"),
            ("universe_entry_fill_rate", "universe_entry_fill_rate"),
            (
                "selected_entry_flag_coverage",
                "selected_realized_plan_entry_filled_coverage",
            ),
            (
                "universe_entry_flag_coverage",
                "universe_realized_plan_entry_filled_coverage",
            ),
            (
                "selected_execution_coverage",
                "selected_realized_plan_return_coverage",
            ),
            (
                "universe_execution_coverage",
                "universe_realized_plan_return_coverage",
            ),
        ):
            metrics[f"{prefix}_{output_name}"] = _finite_topk_metric(
                topk, top_k, column
            )
    return metrics


def _evaluation_result(
    *,
    contract: Mapping[str, Any],
    task: Mapping[str, Any],
    metrics: Mapping[str, Any],
    daily_topk: pd.DataFrame,
    artifacts: Mapping[str, Path],
    completed_at: str,
    inference_performed: bool,
    result_source: str,
) -> dict[str, Any]:
    fairness_material = dict(contract["fairness_material"])
    output_meta: dict[str, Any] = {}
    for name, path in artifacts.items():
        output_meta[name] = str(path.resolve())
        output_meta[f"{name}_sha256"] = _file_sha256(path)
    return {
        "schema_version": 1,
        "artifact_type": "seq100_checkpoint_vintage_2026_evaluation",
        "status": "completed",
        "completed_at": completed_at,
        "analysis_id": ANALYSIS_ID,
        "suite_contract_sha256": str(contract["contract_sha256"]),
        "task_id": str(task["task_id"]),
        "profile": str(task["profile"]),
        "checkpoint_vintage": int(task["checkpoint_vintage"]),
        "target_year": TARGET_YEAR,
        "result_source": str(result_source),
        "inference_performed": bool(inference_performed),
        "checkpoint_path": str(task["checkpoint_path"]),
        "checkpoint_sha256": str(task["checkpoint_sha256"]),
        "normalization_sha256": str(task["normalization_sha256"]),
        "normalization_fit_date_end_exclusive": str(
            task["normalization_fit_date_end_exclusive"]
        ),
        "candidate_index_sha256": str(fairness_material["candidate_index_sha256"]),
        "candidate_key_sha256": str(fairness_material["candidate_key_sha256"]),
        "label_coverage_sha256": str(fairness_material["label_coverage_sha256"]),
        "label_material_sha256": str(fairness_material["label_material_sha256"]),
        "execution_material_sha256": str(
            fairness_material["execution_material_sha256"]
        ),
        "auxiliary_material_sha256": str(
            fairness_material["auxiliary_material_sha256"]
        ),
        "execution_cost_contract_sha256": str(
            fairness_material["execution_cost_contract_sha256"]
        ),
        "evaluation_contract_sha256": str(
            fairness_material["evaluation_contract_sha256"]
        ),
        "evaluation_implementation": dict(
            fairness_material["evaluation_implementation"]
        ),
        "common_daily_coverage_sha256": _canonical_digest(
            freshness._common_daily_coverage(daily_topk).to_dict("records")
        ),
        "metrics": dict(metrics),
        "outputs": output_meta,
        "full_prediction_written": False,
        "qdp_changed": False,
        "provider_called": False,
        "base_pack_changed": False,
        "live_execution_changed": False,
    }


def _validate_completed_result(
    result_path: Path, *, contract: Mapping[str, Any], task: Mapping[str, Any]
) -> dict[str, Any]:
    result = _read_json(result_path)
    suite_contract_sha256 = str(contract["contract_sha256"])
    fairness = dict(contract["fairness_material"])
    if str(result.get("status", "")) != "completed":
        raise ValueError(f"2026 evaluation is not completed: {result_path}")
    if str(result.get("suite_contract_sha256", "")) != suite_contract_sha256:
        raise ValueError(f"2026 evaluation contract drift: {result_path}")
    if str(result.get("task_id", "")) != str(task["task_id"]):
        raise ValueError(f"2026 evaluation task identity drift: {result_path}")
    if str(result.get("checkpoint_sha256", "")) != str(task["checkpoint_sha256"]):
        raise ValueError(f"2026 evaluation checkpoint drift: {result_path}")
    identity_fields = {
        "profile": str(task["profile"]),
        "checkpoint_vintage": int(task["checkpoint_vintage"]),
        "target_year": TARGET_YEAR,
        "normalization_sha256": str(task["normalization_sha256"]),
        "candidate_index_sha256": str(fairness["candidate_index_sha256"]),
        "candidate_key_sha256": str(fairness["candidate_key_sha256"]),
        "label_coverage_sha256": str(fairness["label_coverage_sha256"]),
        "label_material_sha256": str(fairness["label_material_sha256"]),
        "execution_material_sha256": str(fairness["execution_material_sha256"]),
        "auxiliary_material_sha256": str(fairness["auxiliary_material_sha256"]),
        "execution_cost_contract_sha256": str(
            fairness["execution_cost_contract_sha256"]
        ),
        "evaluation_contract_sha256": str(fairness["evaluation_contract_sha256"]),
    }
    for field, expected in identity_fields.items():
        observed = result.get(field)
        if observed != expected:
            raise ValueError(
                f"2026 evaluation {field} drift: expected={expected!r} observed={observed!r}"
            )
    if dict(result.get("evaluation_implementation", {}) or {}) != dict(
        fairness["evaluation_implementation"]
    ):
        raise ValueError(f"2026 evaluation implementation drift: {result_path}")
    metrics = dict(result.get("metrics", {}) or {})
    if (
        int(metrics.get("candidate_count", -1)) != int(fairness["candidate_count"])
        or int(metrics.get("date_count", -1))
        != int(fairness["candidate_date_count"])
        or float(metrics.get("candidate_score_coverage", -1.0)) != 1.0
        or float(metrics.get("execution_return_coverage", -1.0)) != 1.0
    ):
        raise ValueError(f"2026 evaluation coverage drift: {result_path}")
    outputs = dict(result.get("outputs", {}) or {})
    expected_paths = _task_artifact_paths(task)
    if str(result.get("result_source", "")) == "reused_2026_training_evaluation":
        run_dir = Path(str(task["run_dir"])).resolve()
        expected_paths = {name: run_dir / path.name for name, path in expected_paths.items()}
    for name, expected_path in expected_paths.items():
        path = Path(str(outputs.get(name, "") or "")).resolve()
        if path != expected_path.resolve():
            raise ValueError(f"2026 evaluation artifact path drift: {name}={path}")
        if not path.is_file() or _file_sha256(path) != str(
            outputs.get(f"{name}_sha256", "")
        ):
            raise ValueError(f"2026 evaluation artifact drift: {path}")
    daily_topk = pd.read_csv(expected_paths["daily_topk_metrics_csv"])
    observed_daily_hash = _canonical_digest(
        freshness._common_daily_coverage(daily_topk).to_dict("records")
    )
    if observed_daily_hash != str(result.get("common_daily_coverage_sha256", "")):
        raise ValueError(f"2026 evaluation daily coverage drift: {result_path}")
    topk = pd.read_csv(expected_paths["topk_metrics_csv"])
    cost_hashes = {
        str(value)
        for value in topk["execution_cost_contract_sha256"].dropna().tolist()
    }
    if cost_hashes != {str(fairness["execution_cost_contract_sha256"])}:
        raise ValueError(f"2026 evaluation TopK cost contract drift: {result_path}")
    if any(
        bool(result.get(field, True))
        for field in (
            "qdp_changed",
            "provider_called",
            "base_pack_changed",
            "live_execution_changed",
        )
    ):
        raise ValueError(f"2026 evaluation crossed a protected boundary: {result_path}")
    return result


def _reset_task_output(*, task: Mapping[str, Any], suite_root: Path) -> None:
    output_dir = Path(str(task["output_dir"])).resolve()
    expected_parent = (suite_root.resolve() / "tasks").resolve()
    if output_dir.parent != expected_parent or output_dir.name != str(task["task_id"]):
        raise ValueError(f"refusing to reset unexpected evaluation task directory: {output_dir}")
    if output_dir.is_dir():
        shutil.rmtree(output_dir)
    elif output_dir.exists():
        output_dir.unlink()


def _reuse_2026_training_result(
    *, contract: Mapping[str, Any], task: Mapping[str, Any]
) -> dict[str, Any]:
    result_path = Path(str(task["result_path"])).resolve()
    if result_path.is_file():
        return _validate_completed_result(
            result_path,
            contract=contract,
            task=task,
        )
    run_dir = Path(str(task["run_dir"])).resolve()
    summary_path = run_dir / "sequence_path_training_summary.json"
    if _file_sha256(summary_path) != str(task["checkpoint_summary_sha256"]):
        raise ValueError("2026 training summary drifted")
    summary = _read_json(summary_path)
    fairness_material = dict(contract["fairness_material"])
    if (
        str(summary.get("candidate_index_sha256", ""))
        != str(fairness_material["candidate_index_sha256"])
        or str(summary.get("execution_cost_contract_sha256", ""))
        != str(fairness_material["execution_cost_contract_sha256"])
    ):
        raise ValueError("2026 training evaluation is not bound to the common view")
    split_metrics = next(
        item
        for item in summary.get("split_metrics", [])
        if str(item.get("split", "")) == "development"
    )
    artifacts = {
        "topk_metrics_csv": run_dir / "topk_metrics.csv",
        "daily_rank_ic_csv": run_dir / "daily_rank_ic.csv",
        "daily_topk_metrics_csv": run_dir / "daily_topk_metrics.csv",
        "topk_candidates_parquet": run_dir / "topk_candidates.parquet",
    }
    topk = pd.read_csv(artifacts["topk_metrics_csv"])
    daily_topk = pd.read_csv(artifacts["daily_topk_metrics_csv"])
    topk = topk[topk["split"].astype(str).eq("development")].copy()
    daily_topk = daily_topk[
        daily_topk["split"].astype(str).eq("development")
    ].copy()
    metrics = _evaluation_metrics_2026(
        topk=topk, daily_topk=daily_topk, split_metrics=split_metrics
    )
    if (
        int(metrics["candidate_count"]) != int(fairness_material["candidate_count"])
        or int(metrics["date_count"]) != int(
            fairness_material["candidate_date_count"]
        )
    ):
        raise ValueError("2026 training evaluation candidate material drifted")
    result = _evaluation_result(
        contract=contract,
        task=task,
        metrics=metrics,
        daily_topk=daily_topk,
        artifacts=artifacts,
        completed_at=_now(),
        inference_performed=False,
        result_source="reused_2026_training_evaluation",
    )
    _write_json(result_path, result)
    return result


def evaluate_2026_vintage_task(
    *, suite_contract_path: Path, profile: str, vintage: int, force_inference: bool = False
) -> dict[str, Any]:
    contract = _load_comparison_contract(suite_contract_path)
    task_id = _comparison_task_id(profile, vintage)
    task = dict(dict(contract["tasks"])[task_id])
    if int(vintage) == TARGET_YEAR and not bool(force_inference):
        return _reuse_2026_training_result(contract=contract, task=task)
    result_path = Path(str(task["result_path"])).resolve()
    if result_path.is_file():
        return _validate_completed_result(
            result_path,
            contract=contract,
            task=task,
        )
    output_dir = Path(str(task["output_dir"])).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = output_dir / "progress.json"
    _write_json(
        progress_path,
        {
            "status": "loading",
            "task_id": task_id,
            "suite_contract_sha256": str(contract["contract_sha256"]),
            "updated_at": _now(),
        },
    )
    view_path = Path(str(task["view_path"])).resolve()
    if _file_sha256(view_path) != str(task["view_sha256"]):
        raise ValueError(f"evaluation view drift for {task_id}")
    view = _read_json(view_path)
    evaluation = dict(view.get("checkpoint_vintage_evaluation", {}) or {})
    if (
        str(evaluation.get("normalization_sha256", ""))
        != str(task["normalization_sha256"])
        or _canonical_digest(view.get("normalization", {}))
        != str(task["normalization_sha256"])
    ):
        raise ValueError(f"evaluation normalization drift for {task_id}")
    dataset = training.SequencePathPackDataset(
        view,
        split="development",
        max_samples=0,
        input_channel_profile=training.INPUT_CHANNEL_PROFILE_DAILY_ONLY,
        index_role="candidate",
    )
    fairness_material = dict(contract["fairness_material"])
    if (
        int(len(dataset)) != int(fairness_material["candidate_count"])
        or str(dataset.sample_selection["index_sha256"])
        != str(fairness_material["candidate_index_sha256"])
    ):
        raise ValueError(f"candidate material drift for {task_id}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, summary = structured._load_checkpoint_model(
        Path(str(task["run_dir"])), dataset, device=device
    )
    try:
        architecture = freshness._architecture_contract(
            summary,
            {
                "input_dim": dataset.input_dim,
                "resolved_training_config": summary.get("resolved_training_config", {}),
            },
        )
        if _canonical_digest(architecture) != str(task["architecture_sha256"]):
            raise ValueError(f"loaded architecture drift for {task_id}")
        _write_json(
            progress_path,
            {
                "status": "evaluating",
                "task_id": task_id,
                "device": str(device),
                "candidate_count": int(len(dataset)),
                "updated_at": _now(),
            },
        )
        config = dict(summary.get("resolved_training_config", {}) or {})
        ic, topk, daily_topk, topk_candidates, split_metrics = training._predict_split(
            model=model,
            dataset=dataset,
            device=device,
            output_dir=output_dir,
            split="development",
            batch_size=int(config.get("batch_size", 512)),
            amp_enabled=bool(device.type == "cuda" and config.get("amp", True)),
            top_k=TOP_K_VALUES,
            write_predictions=False,
            write_path_predictions=False,
            direct_value_horizon=int(config.get("direct_value_horizon", 0)),
        )
    finally:
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        gc.collect()
    if (
        int(split_metrics["row_count"]) != int(fairness_material["candidate_count"])
        or int(split_metrics["date_count"])
        != int(fairness_material["candidate_date_count"])
    ):
        raise ValueError(f"evaluated candidate material drift for {task_id}")
    metrics = _evaluation_metrics_2026(
        topk=topk, daily_topk=daily_topk, split_metrics=split_metrics
    )
    artifacts = _task_artifact_paths(task)
    _write_csv(artifacts["topk_metrics_csv"], topk)
    _write_csv(artifacts["daily_rank_ic_csv"], ic)
    _write_csv(artifacts["daily_topk_metrics_csv"], daily_topk)
    _write_parquet(artifacts["topk_candidates_parquet"], topk_candidates)
    result = _evaluation_result(
        contract=contract,
        task=task,
        metrics=metrics,
        daily_topk=daily_topk,
        artifacts=artifacts,
        completed_at=_now(),
        inference_performed=True,
        result_source="fresh_inference",
    )
    _write_json(result_path, result)
    _write_json(
        progress_path,
        {
            "status": "completed",
            "task_id": task_id,
            "result_path": str(result_path),
            "updated_at": _now(),
        },
    )
    return result


def _expanded_freshness_gate(results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_profile: dict[str, dict[int, Mapping[str, Any]]] = {}
    for result in results:
        by_profile.setdefault(str(result["profile"]), {})[
            int(result["checkpoint_vintage"])
        ] = result
    decisions: dict[str, Any] = {}
    for profile in PROFILE_ORDER:
        rows = by_profile.get(profile, {})
        if set(rows) != set(CHECKPOINT_VINTAGES):
            raise ValueError(f"2026 freshness gate is missing vintages for {profile}")

        def better(metric: str, *, higher: bool) -> bool:
            current = float(dict(rows[TARGET_YEAR]["metrics"])[metric])
            older = [
                float(dict(rows[year]["metrics"])[metric])
                for year in CHECKPOINT_VINTAGES
                if year != TARGET_YEAR
            ]
            return bool(current > max(older)) if higher else bool(current < min(older))

        breadth_evidence = {
            metric: better(metric, higher=True)
            for metric in ("top1_base_alpha", "top5_base_alpha", "top10_base_alpha")
        }
        ranking_evidence = {
            metric: better(metric, higher=True)
            for metric in ("rank_ic", "top3_opportunity_alpha")
        }
        path_evidence = {
            metric: better(metric, higher=False)
            for metric in ("exit_regret", "path_mae")
        }
        primary = better("top3_base_alpha", higher=True)
        breadth = int(sum(breadth_evidence.values())) >= 2
        ranking = bool(any(ranking_evidence.values()))
        path = bool(any(path_evidence.values()))
        decisions[profile] = {
            "primary_top3_pass": primary,
            "breadth_pass": breadth,
            "ranking_support_pass": ranking,
            "path_support_pass": path,
            "breadth_evidence": breadth_evidence,
            "ranking_evidence": ranking_evidence,
            "path_evidence": path_evidence,
            "profile_pass": bool(primary and breadth and ranking and path),
        }
    return {
        "profiles": decisions,
        "metric_gate_pass": bool(
            all(bool(decisions[profile]["profile_pass"]) for profile in PROFILE_ORDER)
        ),
    }


def _daily_pairwise_deltas(
    *, contract: Mapping[str, Any], results: Sequence[Mapping[str, Any]]
) -> pd.DataFrame:
    by_key = {
        (str(result["profile"]), int(result["checkpoint_vintage"])): result
        for result in results
    }
    rows: list[dict[str, Any]] = []
    metrics = {
        "top3_base_alpha": ("alpha_net_realized_plan_return_base", True),
        "top3_stress_alpha": ("alpha_net_realized_plan_return_stress", True),
        "top3_opportunity_alpha": ("alpha_opportunity_value", True),
        "top3_exit_regret": ("universe_oracle_regret", False),
    }
    for profile in PROFILE_ORDER:
        current_result = by_key[(profile, TARGET_YEAR)]
        current_daily = pd.read_csv(
            Path(str(dict(current_result["outputs"])["daily_topk_metrics_csv"]))
        )
        current_daily = current_daily[current_daily["top_k"].astype(int).eq(3)].copy()
        current_ic = pd.read_csv(
            Path(str(dict(current_result["outputs"])["daily_rank_ic_csv"]))
        )[["trade_date", "rank_ic"]]
        for vintage in (2023, 2024, 2025):
            older_result = by_key[(profile, vintage)]
            older_daily = pd.read_csv(
                Path(str(dict(older_result["outputs"])["daily_topk_metrics_csv"]))
            )
            older_daily = older_daily[older_daily["top_k"].astype(int).eq(3)].copy()
            merged = current_daily.merge(
                older_daily,
                on="trade_date",
                suffixes=("_2026", "_older"),
                validate="one_to_one",
            )
            if len(merged) != EXPECTED_SIGNAL_DATE_COUNT:
                raise ValueError("daily Top3 comparison does not cover all 48 dates")
            for metric, (column, higher) in metrics.items():
                current = pd.to_numeric(merged[f"{column}_2026"], errors="coerce")
                older = pd.to_numeric(merged[f"{column}_older"], errors="coerce")
                delta = current - older if higher else older - current
                for trade_date, current_value, older_value, difference in zip(
                    merged["trade_date"], current, older, delta
                ):
                    rows.append(
                        {
                            "profile": profile,
                            "older_vintage": vintage,
                            "trade_date": str(trade_date),
                            "metric": metric,
                            "higher_is_better": bool(higher),
                            "checkpoint_2026_value": float(current_value),
                            "older_checkpoint_value": float(older_value),
                            "improvement_delta": float(difference),
                            "checkpoint_2026_won": bool(difference > 0.0),
                        }
                    )
            older_ic = pd.read_csv(
                Path(str(dict(older_result["outputs"])["daily_rank_ic_csv"]))
            )[["trade_date", "rank_ic"]]
            ic_merged = current_ic.merge(
                older_ic,
                on="trade_date",
                suffixes=("_2026", "_older"),
                validate="one_to_one",
            )
            for row in ic_merged.itertuples(index=False):
                delta = float(row.rank_ic_2026) - float(row.rank_ic_older)
                rows.append(
                    {
                        "profile": profile,
                        "older_vintage": vintage,
                        "trade_date": str(row.trade_date),
                        "metric": "rank_ic",
                        "higher_is_better": True,
                        "checkpoint_2026_value": float(row.rank_ic_2026),
                        "older_checkpoint_value": float(row.rank_ic_older),
                        "improvement_delta": delta,
                        "checkpoint_2026_won": bool(delta > 0.0),
                    }
                )
    return pd.DataFrame(rows)


def summarize_2026_comparison(*, suite_contract_path: Path) -> dict[str, Any]:
    contract = _load_comparison_contract(suite_contract_path)
    results: list[dict[str, Any]] = []
    for task in dict(contract["tasks"]).values():
        result_path = Path(str(task["result_path"])).resolve()
        if not result_path.is_file():
            raise ValueError("cannot summarize incomplete 2026 vintage tasks")
        results.append(
            _validate_completed_result(
                result_path,
                contract=contract,
                task=task,
            )
        )
    fairness_fields = (
        "candidate_index_sha256",
        "candidate_key_sha256",
        "label_coverage_sha256",
        "label_material_sha256",
        "execution_material_sha256",
        "auxiliary_material_sha256",
        "execution_cost_contract_sha256",
        "evaluation_contract_sha256",
        "common_daily_coverage_sha256",
    )
    common_values = {
        field: sorted({str(result[field]) for result in results})
        for field in fairness_fields
    }
    coverage_checks = {
        "candidate_index_identical": len(common_values["candidate_index_sha256"])
        == 1,
        "candidate_keys_identical": len(common_values["candidate_key_sha256"]) == 1,
        "label_coverage_identical": len(common_values["label_coverage_sha256"]) == 1,
        "label_material_identical": len(common_values["label_material_sha256"]) == 1,
        "execution_material_identical": len(common_values["execution_material_sha256"]) == 1,
        "auxiliary_material_identical": len(
            common_values["auxiliary_material_sha256"]
        )
        == 1,
        "execution_cost_contract_identical": len(
            common_values["execution_cost_contract_sha256"]
        )
        == 1,
        "evaluation_contract_identical": len(
            common_values["evaluation_contract_sha256"]
        )
        == 1,
        "daily_execution_coverage_identical": len(
            common_values["common_daily_coverage_sha256"]
        )
        == 1,
        "score_coverage_complete": all(
            float(dict(result["metrics"])["candidate_score_coverage"]) == 1.0
            for result in results
        ),
        "execution_coverage_complete": all(
            float(dict(result["metrics"])["execution_return_coverage"]) == 1.0
            for result in results
        ),
    }
    fairness_pass = bool(all(coverage_checks.values()))
    reuse_checks: dict[str, Any] = {}
    for profile in PROFILE_ORDER:
        result = next(
            row
            for row in results
            if str(row["profile"]) == profile
            and int(row["checkpoint_vintage"]) == TARGET_YEAR
        )
        reuse_checks[profile] = {
            "result_source": str(result["result_source"]),
            "inference_performed": bool(result["inference_performed"]),
            "pass": str(result["result_source"])
            in {"reused_2026_training_evaluation", "fresh_inference"},
        }
    gate = _expanded_freshness_gate(results)
    rows: list[dict[str, Any]] = []
    for result in sorted(
        results, key=lambda value: (str(value["profile"]), int(value["checkpoint_vintage"]))
    ):
        rows.append(
            {
                "profile": str(result["profile"]),
                "checkpoint_vintage": int(result["checkpoint_vintage"]),
                "target_year": TARGET_YEAR,
                "normalization_fit_date_end_exclusive": str(
                    result["normalization_fit_date_end_exclusive"]
                ),
                "result_source": str(result["result_source"]),
                **dict(result["metrics"]),
            }
        )
    root = suite_contract_path.resolve().parent
    metrics_path = root / "vintage_metrics.csv"
    _write_csv(metrics_path, pd.DataFrame(rows))
    daily_pairwise = _daily_pairwise_deltas(contract=contract, results=results)
    daily_pairwise_path = root / "daily_pairwise_deltas.csv"
    _write_csv(daily_pairwise_path, daily_pairwise)
    pairwise_summary = (
        daily_pairwise.groupby(["profile", "older_vintage", "metric"], sort=True)
        .agg(
            date_count=("trade_date", "size"),
            mean_improvement_delta=("improvement_delta", "mean"),
            median_improvement_delta=("improvement_delta", "median"),
            checkpoint_2026_win_rate=("checkpoint_2026_won", "mean"),
        )
        .reset_index()
    )
    pairwise_summary_path = root / "pairwise_summary.csv"
    _write_csv(pairwise_summary_path, pairwise_summary)
    summary = {
        "schema_version": 1,
        "artifact_type": "seq100_checkpoint_vintage_2026_summary",
        "status": "completed",
        "completed_at": _now(),
        "analysis_id": ANALYSIS_ID,
        "suite_contract": str(suite_contract_path.resolve()),
        "suite_contract_sha256": str(contract["contract_sha256"]),
        "target_year": TARGET_YEAR,
        "signal_window": [SIGNAL_START, SIGNAL_END],
        "signal_date_count": EXPECTED_SIGNAL_DATE_COUNT,
        "candidate_count": EXPECTED_CANDIDATE_COUNT,
        "symbol_count": EXPECTED_SYMBOL_COUNT,
        "profiles": list(PROFILE_ORDER),
        "checkpoint_vintages": list(CHECKPOINT_VINTAGES),
        "fairness": {
            "checks": coverage_checks,
            "common_values": common_values,
            "pass": fairness_pass,
        },
        "existing_2026_consistency": {
            "profiles": reuse_checks,
            "pass": bool(all(value["pass"] for value in reuse_checks.values())),
        },
        "gate": gate,
        "overall_2026_freshness_gate_pass": bool(
            fairness_pass
            and all(value["pass"] for value in reuse_checks.values())
            and gate["metric_gate_pass"]
        ),
        "uncertainty": dict(contract["uncertainty_contract"]),
        "metrics": rows,
        "pairwise_summary": pairwise_summary.to_dict("records"),
        "outputs": {
            "vintage_metrics_csv": str(metrics_path.resolve()),
            "vintage_metrics_csv_sha256": _file_sha256(metrics_path),
            "daily_pairwise_deltas_csv": str(daily_pairwise_path.resolve()),
            "daily_pairwise_deltas_csv_sha256": _file_sha256(daily_pairwise_path),
            "pairwise_summary_csv": str(pairwise_summary_path.resolve()),
            "pairwise_summary_csv_sha256": _file_sha256(pairwise_summary_path),
        },
        "qdp_changed": False,
        "provider_called": False,
        "base_pack_changed": False,
        "live_execution_changed": False,
        "score_only_tail_included": False,
        "finite_capital_cagr_reported": False,
    }
    summary_path = root / "comparison_summary.json"
    _write_json(summary_path, summary)
    lines = [
        "# Seq100 checkpoint vintages on the complete 2026 realized window",
        "",
        f"Evaluation window: {SIGNAL_START} through {SIGNAL_END}; {EXPECTED_SIGNAL_DATE_COUNT} signal dates and {EXPECTED_CANDIDATE_COUNT:,} candidates.",
        "",
        "| profile | checkpoint | Top1 alpha | Top3 alpha | Top5 alpha | Top10 alpha | Rank IC | opportunity alpha | exit regret | path MAE |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['profile']} | {row['checkpoint_vintage']} | "
            f"{row['top1_base_alpha']:.4%} | {row['top3_base_alpha']:.4%} | "
            f"{row['top5_base_alpha']:.4%} | {row['top10_base_alpha']:.4%} | "
            f"{row['rank_ic']:.6f} | {row['top3_opportunity_alpha']:.4%} | "
            f"{row['exit_regret']:.6f} | {row['path_mae']:.6f} |"
        )
    lines.extend(
        [
            "",
            f"Fairness checks: `{fairness_pass}`.",
            f"2026 freshness gate: `{summary['overall_2026_freshness_gate_pass']}`.",
            "Formal moving-block confidence intervals are intentionally omitted because 48 signal dates are shorter than the 60-day overlapping outcome horizon.",
        ]
    )
    (root / "comparison_summary.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    return summary


def evaluate_2026_vintages(
    *, output_root: Path = ANALYSIS_ROOT, max_tasks: int = 0
) -> dict[str, Any]:
    if int(max_tasks) < 0:
        raise ValueError("max_tasks must be non-negative")
    _assert_no_other_research_process()
    _assert_frozen_inputs()
    contract = prepare_2026_comparison(output_root=output_root)
    root = output_root.resolve()
    contract_path = root / "contract.json"
    state_path = root / "state.json"
    completed: list[str] = []
    reuse_fallback_tasks: list[str] = []
    launched = 0
    _write_json(
        state_path,
        {
            "status": "running",
            "analysis_id": ANALYSIS_ID,
            "suite_contract_sha256": str(contract["contract_sha256"]),
            "completed_tasks": completed,
            "reuse_fallback_tasks": reuse_fallback_tasks,
            "updated_at": _now(),
        },
    )
    try:
        for task_id, raw_task in dict(contract["tasks"]).items():
            task = dict(raw_task)
            result_path = Path(str(task["result_path"])).resolve()
            if result_path.is_file():
                try:
                    _validate_completed_result(
                        result_path, contract=contract, task=task
                    )
                except (FileNotFoundError, ValueError):
                    _reset_task_output(task=task, suite_root=root)
                else:
                    if task_id not in completed:
                        completed.append(task_id)
                    continue
            elif Path(str(task["output_dir"])).exists():
                _reset_task_output(task=task, suite_root=root)

            force_inference = False
            if int(task["checkpoint_vintage"]) == TARGET_YEAR:
                try:
                    result = _reuse_2026_training_result(contract=contract, task=task)
                    _validate_completed_result(
                        Path(str(task["result_path"])), contract=contract, task=task
                    )
                except (FileNotFoundError, StopIteration, ValueError):
                    _reset_task_output(task=task, suite_root=root)
                    force_inference = True
                    reuse_fallback_tasks.append(task_id)
                else:
                    completed.append(str(result["task_id"]))
                    _write_json(
                        state_path,
                        {
                            "status": "running",
                            "analysis_id": ANALYSIS_ID,
                            "suite_contract_sha256": str(contract["contract_sha256"]),
                            "completed_tasks": completed,
                            "reuse_fallback_tasks": reuse_fallback_tasks,
                            "updated_at": _now(),
                        },
                    )
                    continue
            if int(max_tasks) > 0 and launched >= int(max_tasks):
                break
            command = [
                str(PYTHON),
                "-m",
                "daily_research.path_policy.seq100_development",
                "evaluate-2026-vintage-task",
                "--suite-contract",
                str(contract_path),
                "--profile",
                str(task["profile"]),
                "--vintage",
                str(task["checkpoint_vintage"]),
            ]
            if force_inference:
                command.append("--force-inference")
            _guarded_command(
                command,
                log_path=root / "logs" / f"memory_guard_{task_id}.json",
            )
            _validate_completed_result(
                result_path,
                contract=contract,
                task=task,
            )
            completed.append(task_id)
            launched += 1
            _write_json(
                state_path,
                {
                    "status": "running",
                    "analysis_id": ANALYSIS_ID,
                    "suite_contract_sha256": str(contract["contract_sha256"]),
                    "completed_tasks": completed,
                    "reuse_fallback_tasks": reuse_fallback_tasks,
                    "updated_at": _now(),
                },
            )
        if len(completed) != len(contract["tasks"]):
            return _read_json(state_path)
        summary = summarize_2026_comparison(suite_contract_path=contract_path)
        _write_json(
            state_path,
            {
                "status": "completed",
                "analysis_id": ANALYSIS_ID,
                "suite_contract_sha256": str(contract["contract_sha256"]),
                "completed_tasks": completed,
                "reuse_fallback_tasks": reuse_fallback_tasks,
                "summary": str((root / "comparison_summary.json").resolve()),
                "updated_at": _now(),
            },
        )
        return summary
    except Exception as exc:
        _write_json(
            state_path,
            {
                "status": "failed",
                "analysis_id": ANALYSIS_ID,
                "suite_contract_sha256": str(contract["contract_sha256"]),
                "completed_tasks": completed,
                "reuse_fallback_tasks": reuse_fallback_tasks,
                "error": str(exc),
                "updated_at": _now(),
            },
        )
        raise
