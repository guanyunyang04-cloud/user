"""Structured 180x35 2026 fold and fixed-D7 account experiment.

The study is intentionally isolated from the existing Seq100 studies. It
reuses their immutable training labels and execution panels, then evaluates
only the preselected Structured 180x35 model and its two fixed-D7 accounts.
"""

from __future__ import annotations

import gc
import json
import math
import os
import shutil
import subprocess
import time
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import psutil
import torch

from daily_research.path_policy import qdp_v2_sequence_path_pack as pack_builder
from daily_research.path_policy import qdp_v2_sequence_path_training as training
from daily_research.path_policy import seq100_2026_fold_comparison as comparison_2026
from daily_research.path_policy import seq100_checkpoint_freshness as freshness
from daily_research.path_policy import seq100_development as development
from daily_research.path_policy import seq100_finite_capital_backtest as finite
from daily_research.path_policy import seq100_integrity_v2 as integrity
from daily_research.path_policy import seq100_structured_experiment as structured
from daily_research.path_policy import seq100_structured_input_ablation as input_ablation
from daily_research.path_policy import seq100_walkforward as walkforward
from daily_research.path_policy.seq100_exit_policy_audit import CandidateCompleteAuditPack
from daily_research.path_policy.seq100_mainline import build_todayclose_path_only_train_argv
from quant_data_platform.qdp_v2.manifest import qdp_snapshot_sha256


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
PYTHON = Path("C:/Users/ASUS/miniconda3/envs/yolos/python.exe")
STUDY_ID = "seq100_structured_180x35_2026_fold_v1"
EXPERIMENT_ID = "structured-180x35-2026-d7-v1"
STUDY_ROOT = WORKSPACE_ROOT / (
    "daily_research/output/path_policy/studies/"
    "seq100_structured_180x35_2026_fold_v1"
)
OVERLAY_ROOT = WORKSPACE_ROOT / (
    "daily_research/data/research_store/"
    "seq100_structured_180x35_2026_overlay_v1"
)

BASE_PACK_MANIFEST = input_ablation.BASE_PACK_MANIFEST
INPUT_OVERLAY_ROOT = input_ablation.OVERLAY_ROOT
INPUT_STUDY_ROOT = input_ablation.STUDY_ROOT
FROZEN_2026_OVERLAY = comparison_2026.OVERLAY_ROOT / "manifest.json"
FROZEN_2026_VIEW = comparison_2026.STUDY_ROOT / "views/development_2026.json"
QDP_ROOT = input_ablation.QDP_ROOT
LIVE_STATE = WORKSPACE_ROOT / "daily_research/output/active_execution_strategy.json"

TARGET_YEAR = 2026
SEED = 7
LOOKBACK_DAYS = 180
INPUT_DIM = 35
INPUT_CHANNEL_PROFILE = training.INPUT_CHANNEL_PROFILE_DAILY_ONLY_TURNOVER
FORWARD_DAYS = 60
EXECUTION_TAIL_DAYS = 20
BATCH_SIZE = 512
EPOCHS = 1

SIGNAL_START = "2026-01-05"
FULL_LABEL_END = "2026-03-19"
STRICT_SIGNAL_END = "2026-06-08"
EXTENDED_SIGNAL_END = "2026-07-07"
QDP_AS_OF = "2026-07-16"
OVERLAP_DATE = "2026-05-08"
SUFFIX_START = "2026-05-11"
SAFE_TRAIN_SIGNAL_END = "2025-09-02"
PURGE_START = "2025-09-03"
PURGE_END = "2025-12-31"

EXPECTED_SYMBOL_COUNT = 3_034
EXPECTED_FULL_LABEL_DATES = 48
EXPECTED_FULL_LABEL_CANDIDATES = 143_840
EXPECTED_EXTENDED_DATES = 121
EXPECTED_STRICT_DATES = 101
EXPECTED_COMMON_TRAIN_ROWS = 7_674_720
EXPECTED_NATIVE_TRAIN_ROWS = 7_396_133
EXPECTED_NATIVE_FULL_LABEL_CANDIDATES = 143_300
EXPECTED_NATIVE_FULL_LABEL_SUPERVISED = 143_265
EXPECTED_PURGE_ROWS = 239_081
EXPECTED_PURGE_DATES = 80
EXPECTED_SUFFIX_DATES = 42
OVERLAY_SIZE_LIMIT_BYTES = 1024**3
MEMORY_GUARD_GIB = 0.5
MEMORY_INTERVAL_SECONDS = 1.0
MEMORY_CONSECUTIVE_BREACHES = 2
MONITOR_POLL_SECONDS = 5.0
STALE_SECONDS = 15 * 60
STARTING_CASH_CNY = 1_000_000.0
TOP_K_VALUES = (1, 3, 5, 10)

MODEL_2026_180 = "structured_180x35_2026"
MODEL_ORDER = (MODEL_2026_180,)


@dataclass(frozen=True)
class AccountSpec:
    top_k: int
    slots: int

    @property
    def strategy_id(self) -> str:
        return f"top{int(self.top_k)}_slots{int(self.slots)}_fixed_d7"


ACCOUNT_SPECS = (AccountSpec(1, 1), AccountSpec(3, 3))
COST_SCENARIOS = ("base", "double_slippage")


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    input_ablation._write_json(path, payload)


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _task_identity(scope: str) -> dict[str, str]:
    material = dict(_read_json(STUDY_ROOT / "study.json")["material"])
    shared = dict(material["integrity"])
    detail = dict(shared["candidate_semantic_detail"])
    candidate_key = {
        "full_label": "full_label_native_180",
        "extended_121": "extended_native_180",
        "strict_101": "strict_native_180",
    }[scope]
    return {
        "data_snapshot_sha256": str(shared["data_snapshot_sha256"]),
        "candidate_semantic_sha256": str(detail[candidate_key]),
        "evaluation_contract_sha256": str(shared["evaluation_contract_sha256"]),
    }


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    input_ablation._write_csv(path, frame)


def _write_parquet(path: Path, frame: pd.DataFrame) -> None:
    input_ablation._write_parquet(path, frame)


def _file_sha256(path: Path) -> str:
    return input_ablation._file_sha256(path)


def _canonical_digest(payload: Any) -> str:
    return input_ablation._canonical_digest(payload)


def _directory_size(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def _open_memmap(meta: Mapping[str, Any], *, dtype: str) -> np.memmap:
    return input_ablation._open_memmap(meta, dtype=dtype)


def _close_memmap(array: Any) -> None:
    input_ablation._close_memmap(array)


def _create_memmap(
    path: Path, *, dtype: str, shape: Sequence[int], fill: float | bool
) -> np.memmap:
    path.parent.mkdir(parents=True, exist_ok=True)
    array = np.memmap(
        path,
        dtype=dtype,
        mode="w+",
        shape=tuple(int(value) for value in shape),
    )
    array[:] = fill
    array.flush()
    return array


def _memmap_meta(
    path: Path, shape: Sequence[int], *, dtype: str, columns: Sequence[str] | None = None
) -> dict[str, Any]:
    payload = input_ablation._memmap_meta(path, shape, dtype=dtype)
    if columns is not None:
        payload["columns"] = [str(value) for value in columns]
    return payload


def signal_window_indices(date_values: Sequence[str]) -> dict[str, Any]:
    dates = [str(value) for value in date_values]
    required = (
        SIGNAL_START,
        FULL_LABEL_END,
        STRICT_SIGNAL_END,
        EXTENDED_SIGNAL_END,
        OVERLAP_DATE,
        QDP_AS_OF,
    )
    missing = [value for value in required if value not in dates]
    if missing:
        raise ValueError(f"registered dates are absent from the market calendar: {missing}")
    full = [value for value in dates if SIGNAL_START <= value <= FULL_LABEL_END]
    strict = [value for value in dates if SIGNAL_START <= value <= STRICT_SIGNAL_END]
    extended = [value for value in dates if SIGNAL_START <= value <= EXTENDED_SIGNAL_END]
    suffix = [value for value in dates if OVERLAP_DATE <= value <= EXTENDED_SIGNAL_END]
    if len(full) != EXPECTED_FULL_LABEL_DATES:
        raise ValueError("full-label signal-date count drifted")
    if len(strict) != EXPECTED_STRICT_DATES:
        raise ValueError("strict-tail signal-date count drifted")
    if len(extended) != EXPECTED_EXTENDED_DATES:
        raise ValueError("extended D7 signal-date count drifted")
    if len(suffix) != EXPECTED_SUFFIX_DATES:
        raise ValueError("feature suffix date count drifted")
    if dates.index(EXTENDED_SIGNAL_END) + 7 != dates.index(QDP_AS_OF):
        raise ValueError("the last extended signal no longer maps to D7 on QDP as-of")
    if dates.index(STRICT_SIGNAL_END) + 7 + 20 != dates.index(QDP_AS_OF):
        raise ValueError("the strict signal boundary no longer has a complete D7+20 tail")
    return {
        "full_label_dates": full,
        "strict_dates": strict,
        "extended_dates": extended,
        "suffix_dates": suffix,
        "full_label_start_idx": dates.index(SIGNAL_START),
        "full_label_end_idx": dates.index(FULL_LABEL_END),
        "strict_end_idx": dates.index(STRICT_SIGNAL_END),
        "extended_end_idx": dates.index(EXTENDED_SIGNAL_END),
        "overlap_idx": dates.index(OVERLAP_DATE),
        "qdp_as_of_idx": dates.index(QDP_AS_OF),
    }


def _assert_stage3_absent() -> None:
    study = _read_json(INPUT_STUDY_ROOT / "study.json")
    runtime = dict(study.get("runtime", {}) or {})
    if bool(runtime.get("stage3_started", False)):
        raise ValueError("Structured input Stage 3 has started")
    if (INPUT_STUDY_ROOT / "runs/stage_3").exists():
        raise ValueError("Structured input Stage 3 directory exists")


def _assert_no_research_process() -> None:
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
        "seq100_structured_180x35_2026",
        "run-structured-180x35-2026",
        "prepare-structured-180x35-2026",
    )
    matches: list[int] = []
    for process in psutil.process_iter(["pid", "cmdline"]):
        try:
            pid = int(process.info["pid"])
            if pid in excluded:
                continue
            command = " ".join(process.info.get("cmdline") or [])
            if any(marker in command for marker in markers):
                matches.append(pid)
        except (psutil.AccessDenied, psutil.NoSuchProcess):
            continue
    if matches:
        raise RuntimeError(f"another Structured 2026 research process is active: {matches}")


def _source_identity() -> dict[str, Any]:
    frozen = input_ablation._assert_frozen_inputs()
    comparison_2026._validate_overlay_material(FROZEN_2026_OVERLAY)
    _assert_stage3_absent()
    old_2025 = _old_2025_run()
    live = {
        "exists": LIVE_STATE.is_file(),
        "sha256": _file_sha256(LIVE_STATE) if LIVE_STATE.is_file() else None,
    }
    return {
        "qdp_active_as_of": str(frozen["qdp_active_as_of"]),
        "qdp_active_sha256": str(frozen["qdp_active_sha256"]),
        "qdp_full_audit_sha256": str(frozen["qdp_full_audit_sha256"]),
        "qdp_snapshot_sha256": qdp_snapshot_sha256(QDP_ROOT),
        "base_pack_path": str(BASE_PACK_MANIFEST.resolve()),
        "base_pack_sha256": _file_sha256(BASE_PACK_MANIFEST),
        "input_overlay_manifest": str((INPUT_OVERLAY_ROOT / "manifest.json").resolve()),
        "input_overlay_sha256": _file_sha256(INPUT_OVERLAY_ROOT / "manifest.json"),
        "frozen_2026_overlay": str(FROZEN_2026_OVERLAY.resolve()),
        "frozen_2026_overlay_sha256": _file_sha256(FROZEN_2026_OVERLAY),
        "old_2025_checkpoint": str((old_2025 / "best_model.pt").resolve()),
        "old_2025_checkpoint_sha256": _file_sha256(old_2025 / "best_model.pt"),
        "live_state": live,
        "stage3_started": False,
    }


def _old_2025_run() -> Path:
    descriptor = {
        "source": "ablation_run",
        "origin_stage": 2,
        "variant": input_ablation.InputVariant(
            LOOKBACK_DAYS, turnover=True, intraday=False
        ).to_dict(),
    }
    return input_ablation._descriptor_run_dir(
        descriptor, year=2025, study_path=INPUT_STUDY_ROOT / "study.json"
    )


def _semantic_contract() -> dict[str, Any]:
    profile = asdict(input_ablation._base_structured_profile(BATCH_SIZE))
    for key in ("store_view", "output_root", "run_tag"):
        profile.pop(key, None)
    profile.update(
        {
            "epochs": EPOCHS,
            "input_channel_profile": INPUT_CHANNEL_PROFILE,
            "evaluation_mode": training.EVALUATION_MODE_DEVELOPMENT,
            "early_stopping_patience": 1,
            "top_k": "1,3,5,10",
        }
    )
    return {
        "schema_version": 1,
        "contract_id": STUDY_ID,
        "experiment": EXPERIMENT_ID,
        "development_protocol": {
            "years": [TARGET_YEAR],
            "method": "purged_expanding_development_walkforward_native_180_input",
            "split_roles": {"fit": "train", "evaluation": "development"},
            "seed": SEED,
            "purge_days": FORWARD_DAYS + EXECUTION_TAIL_DAYS,
            "candidate_membership_uses_future_information": False,
            "final_fit_in_scope": False,
        },
        "early_stopping": {
            "metric": training.EARLY_STOPPING_METRIC_DEVELOPMENT_PRICE_TOTAL_LOSS,
            "mode": training.EARLY_STOPPING_MODE_MIN,
            "minimum_complete_epochs": 1,
            "maximum_epochs": EPOCHS,
            "patience": 1,
            "restore_best_checkpoint": True,
            "epoch_selection_uses_2026_labels": False,
        },
        "profile": {
            "name": "structured_joint_turnover",
            "input_dim": INPUT_DIM,
            "parameters": profile,
        },
        "data": {
            "qdp_active_as_of": QDP_AS_OF,
            "symbol_count": EXPECTED_SYMBOL_COUNT,
            "full_label_window": [SIGNAL_START, FULL_LABEL_END],
            "extended_d7_window": [SIGNAL_START, EXTENDED_SIGNAL_END],
            "strict_tail_window": [SIGNAL_START, STRICT_SIGNAL_END],
            "base_pack_mutated": False,
            "old_checkpoint_mutated": False,
        },
        "training": {
            "train_signal_window": ["2010-09-29", SAFE_TRAIN_SIGNAL_END],
            "prequalification_train_row_count": EXPECTED_COMMON_TRAIN_ROWS,
            "native_180_train_row_count": EXPECTED_NATIVE_TRAIN_ROWS,
            "purge_window": [PURGE_START, PURGE_END],
            "purge_row_count": EXPECTED_PURGE_ROWS,
            "purge_date_count": EXPECTED_PURGE_DATES,
            "lookback_days": LOOKBACK_DAYS,
            "input_dim": INPUT_DIM,
            "input_channel_profile": INPUT_CHANNEL_PROFILE,
            "seed": SEED,
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS,
            "epoch_selection_uses_2026_labels": False,
            "input_qualification": "complete_180_session_close_history_without_continuity_break",
            "profile": profile,
        },
        "evaluation": {
            "full_label_metrics": [
                "rank_ic",
                "rank_ic_positive_day_rate",
                "path_mae",
                "top1_top3_alpha",
                "opportunity_alpha",
                "exit_regret",
            ],
            "account_strategies": [spec.strategy_id for spec in ACCOUNT_SPECS],
            "entry": "next_trading_day_open",
            "exit": "fixed_d7_close_then_first_sellable_close_up_to_20_days",
            "starting_cash_cny": STARTING_CASH_CNY,
            "allow_leverage": False,
            "allow_pyramiding": False,
            "same_day_ordering": "entry_checks_before_exits",
            "scan_other_exit_days": False,
            "cross_model_rank_exit_hybrid": False,
            "headline_cost_scenario": "double_slippage",
        },
        "protected_boundaries": {
            "update_qdp": False,
            "call_provider": False,
            "mutate_base_pack": False,
            "mutate_old_checkpoint": False,
            "mutate_stage3": False,
            "change_live_state": False,
            "place_order": False,
        },
    }


def _load_or_create_study(study_root: Path) -> tuple[Path, dict[str, Any]]:
    root = study_root.resolve()
    if root != STUDY_ROOT.resolve():
        raise ValueError("Structured 180x35 2026 uses the canonical study root")
    path = root / "study.json"
    semantic = _semantic_contract()
    if path.is_file():
        study = _read_json(path)
        if (
            str(study.get("study_id", "")) != STUDY_ID
            or dict(study.get("contract", {}) or {}) != semantic
            or str(study.get("contract_sha256", "")) != _canonical_digest(semantic)
        ):
            raise ValueError("existing Structured 180x35 2026 contract drifted")
        return path, study
    study = {
        "schema_version": 1,
        "artifact_type": "seq100_current_study",
        "study_id": STUDY_ID,
        "contract": semantic,
        "contract_sha256": _canonical_digest(semantic),
        "runtime": {
            "status": "preparing",
            "created_at": _now(),
            "updated_at": _now(),
            "current_task": None,
            "error": None,
        },
        "material": {},
    }
    _write_json(path, study)
    return path, study


def _update_runtime(study_path: Path, **changes: Any) -> None:
    study = _read_json(study_path)
    runtime = dict(study.get("runtime", {}) or {})
    runtime.update(changes)
    runtime["updated_at"] = _now()
    study["runtime"] = runtime
    _write_json(study_path, study)


def _compact_suffix_pack(staging: Path, study_path: Path) -> Path:
    compact_root = staging / "_compact"
    manifest_path = compact_root / "manifest.json"
    if manifest_path.is_file():
        validation = pack_builder.validate_sequence_pack(manifest_path)
        if str(validation.get("status", "")) == "ok":
            return manifest_path
        shutil.rmtree(compact_root)
    base = _read_json(BASE_PACK_MANIFEST)
    semantics = dict(base.get("data_semantics", {}) or {})
    config = pack_builder.SequencePackConfig(
        qdp_root=QDP_ROOT.resolve(),
        output_root=staging,
        run_tag="_compact",
        lookback_days=LOOKBACK_DAYS,
        forward_days=7,
        start_date=SIGNAL_START,
        end_date=EXTENDED_SIGNAL_END,
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
        execution_tail_days=0,
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
    validation = pack_builder.validate_sequence_pack(manifest_path)
    if str(validation.get("status", "")) != "ok":
        raise ValueError(f"compact suffix pack validation failed: {validation}")
    return manifest_path


def _extract_mapped_panel(
    *,
    path: Path,
    compact_meta: Mapping[str, Any],
    compact_date_indices: Sequence[int],
    compact_symbol_indices: np.ndarray,
    dtype: str,
    columns: Sequence[str] | None = None,
) -> dict[str, Any]:
    source = _open_memmap(compact_meta, dtype=dtype)
    tail = tuple(int(value) for value in source.shape[2:])
    shape = (len(compact_date_indices), len(compact_symbol_indices), *tail)
    output = _create_memmap(
        path,
        dtype=dtype,
        shape=shape,
        fill=(False if dtype == "bool" else np.nan),
    )
    try:
        for out_idx, source_idx in enumerate(compact_date_indices):
            output[out_idx] = np.asarray(
                source[int(source_idx), compact_symbol_indices], dtype=dtype
            )
        output.flush()
    finally:
        _close_memmap(source)
        _close_memmap(output)
    return _memmap_meta(path, shape, dtype=dtype, columns=columns)


def _assert_overlap(
    *, name: str, left_meta: Mapping[str, Any], right_meta: Mapping[str, Any],
    left_idx: int, right_idx: int, dtype: str
) -> None:
    left = _open_memmap(left_meta, dtype=dtype)
    right = _open_memmap(right_meta, dtype=dtype)
    try:
        lhs = np.asarray(left[int(left_idx)])
        rhs = np.asarray(right[int(right_idx)])
        matches = (
            bool(np.array_equal(lhs, rhs))
            if dtype == "bool"
            else bool(np.allclose(lhs, rhs, equal_nan=True, atol=1.0e-6, rtol=1.0e-6))
        )
        if not matches:
            raise ValueError(f"May 8 overlap regression failed for {name}")
    finally:
        _close_memmap(left)
        _close_memmap(right)


def _restore_base_candidate_semantics(
    *,
    base: Mapping[str, Any],
    channels: Mapping[str, Any],
    masks: Mapping[str, Any],
    suffix_dates: Sequence[str],
    global_dates: Sequence[str],
) -> dict[str, Any]:
    """Use the frozen pack's 100-day candidate rule with 180-day model inputs."""
    base_daily = _open_memmap(dict(base["feature_channels"])["daily_raw"], dtype="float32")
    base_continuity = _open_memmap(dict(base["masks"])["continuity_break"], dtype="bool")
    suffix_daily = _open_memmap(channels["daily_raw"], dtype="float32")
    suffix_continuity = _open_memmap(masks["continuity_break"], dtype="bool")
    signal_eligible = _open_memmap(masks["signal_eligible"], dtype="bool")
    input_valid = np.memmap(
        Path(str(masks["input_valid"]["path"])),
        dtype="bool",
        mode="r+",
        shape=tuple(int(value) for value in masks["input_valid"]["shape"]),
    )
    candidate = np.memmap(
        Path(str(masks["candidate_eligible"]["path"])),
        dtype="bool",
        mode="r+",
        shape=tuple(int(value) for value in masks["candidate_eligible"]["shape"]),
    )
    suffix_start_idx = [str(value) for value in global_dates].index(str(suffix_dates[0]))
    try:
        for out_idx, trade_date in enumerate(suffix_dates):
            global_idx = [str(value) for value in global_dates].index(str(trade_date))
            start = global_idx - 99
            if start < 0:
                raise ValueError("candidate lookback starts before the global panel")
            close_parts: list[np.ndarray] = []
            continuity_parts: list[np.ndarray] = []
            if start < suffix_start_idx:
                base_stop = min(global_idx + 1, suffix_start_idx)
                close_parts.append(np.asarray(base_daily[start:base_stop, :, 3], dtype=np.float32))
                continuity_parts.append(
                    np.asarray(base_continuity[start:base_stop], dtype=bool)
                )
            local_start = max(start, suffix_start_idx) - suffix_start_idx
            local_stop = global_idx - suffix_start_idx + 1
            if local_stop > local_start:
                close_parts.append(
                    np.asarray(suffix_daily[local_start:local_stop, :, 3], dtype=np.float32)
                )
                continuity_parts.append(
                    np.asarray(suffix_continuity[local_start:local_stop], dtype=bool)
                )
            close = np.concatenate(close_parts, axis=0)
            continuity = np.concatenate(continuity_parts, axis=0)
            if close.shape[0] != 100 or continuity.shape[0] != 100:
                raise ValueError("candidate semantic reconstruction has a wrong lookback")
            valid = np.isfinite(close).all(axis=0) & ~continuity[1:].any(axis=0)
            input_valid[out_idx] = valid
            candidate[out_idx] = valid & np.asarray(signal_eligible[out_idx], dtype=bool)
        input_valid.flush()
        candidate.flush()
    finally:
        for array in (
            base_daily,
            base_continuity,
            suffix_daily,
            suffix_continuity,
            signal_eligible,
            input_valid,
            candidate,
        ):
            _close_memmap(array)
    updated = {
        "input_valid": _file_sha256(Path(str(masks["input_valid"]["path"]))),
        "candidate_eligible": _file_sha256(
            Path(str(masks["candidate_eligible"]["path"]))
        ),
    }
    return {
        "lookback_days": 100,
        "reason": "candidate_membership_is_frozen_to_the_base_pack_and_does_not_change_with_model_lookback",
        "updated_sha256": updated,
    }


def _build_prefix_native_mask(*, base: Mapping[str, Any], path: Path) -> dict[str, Any]:
    """Materialize the complete-history (180 trading-day) prefix qualification mask."""
    raw_meta = dict(base["feature_channels"])["daily_raw"]
    continuity_meta = dict(base["masks"])["continuity_break"]
    raw = _open_memmap(raw_meta, dtype="float32")
    continuity = _open_memmap(continuity_meta, dtype="bool")
    shape = (int(raw.shape[0]), int(raw.shape[1]))
    output = _create_memmap(path, dtype="bool", shape=shape, fill=False)
    try:
        for date_idx in range(LOOKBACK_DAYS - 1, shape[0]):
            output[date_idx] = (
                np.isfinite(np.asarray(raw[date_idx - LOOKBACK_DAYS + 1 : date_idx + 1, :, 3])).all(axis=0)
                & ~np.asarray(
                    continuity[date_idx - LOOKBACK_DAYS + 2 : date_idx + 1]
                ).any(axis=0)
            )
        output.flush()
    finally:
        _close_memmap(raw)
        _close_memmap(continuity)
        _close_memmap(output)
    return _memmap_meta(path, shape, dtype="bool")


def _build_extended_candidates(
    *,
    staging: Path,
    global_manifest: Mapping[str, Any],
    suffix_manifest: Mapping[str, Any],
    entry_filled_meta: Mapping[str, Any],
    entry_date_values: Sequence[str],
) -> tuple[dict[str, Path], dict[str, Any]]:
    date_values = [str(value) for value in global_manifest["date_values"]]
    windows = signal_window_indices(date_values)
    symbols = np.asarray(global_manifest["symbol_values"], dtype=object)
    if len(symbols) != EXPECTED_SYMBOL_COUNT:
        raise ValueError("global symbol count drifted")
    base = _read_json(BASE_PACK_MANIFEST)
    base_candidate = _open_memmap(dict(base["masks"])["candidate_eligible"], dtype="bool")
    base_native = _open_memmap(suffix_manifest["prefix_native_input_valid"], dtype="bool")
    suffix_candidate = _open_memmap(
        dict(suffix_manifest["masks"])["candidate_eligible"], dtype="bool"
    )
    suffix_native = _open_memmap(suffix_manifest["masks"]["native_input_valid"], dtype="bool")
    entry_filled = _open_memmap(entry_filled_meta, dtype="bool")
    entry_dates = [str(value) for value in entry_date_values]
    if entry_dates != list(windows["extended_dates"]):
        raise ValueError("extended entry-mask dates drifted")
    suffix_dates = [str(value) for value in suffix_manifest["date_values"]]
    suffix_start_idx = date_values.index(OVERLAP_DATE)
    rows_by_mode: dict[str, list[pd.DataFrame]] = {"common_100": [], "native_180": []}
    try:
        for trade_date in windows["extended_dates"]:
            global_idx = date_values.index(trade_date)
            if global_idx < suffix_start_idx:
                common = np.asarray(base_candidate[global_idx], dtype=bool)
                native = np.asarray(base_native[global_idx], dtype=bool) & common
                source = "base_prefix"
            else:
                local_idx = suffix_dates.index(trade_date)
                common = np.asarray(suffix_candidate[local_idx], dtype=bool)
                native = np.asarray(suffix_native[local_idx], dtype=bool) & common
                source = "feature_suffix"
            entry = np.asarray(entry_filled[entry_dates.index(trade_date)], dtype=bool)
            for mode, eligible in (("common_100", common), ("native_180", native)):
                symbol_idx = np.flatnonzero(eligible).astype(np.int32)
                rows_by_mode[mode].append(
                    pd.DataFrame(
                        {
                            "split": "development",
                            "year": np.int16(TARGET_YEAR),
                            "trade_date": trade_date,
                            "date_idx": np.full(len(symbol_idx), global_idx, dtype=np.int32),
                            "symbol_idx": symbol_idx,
                            "symbol": symbols[symbol_idx],
                            "entry_trade_date": date_values[global_idx + 1],
                            "entry_filled": entry[symbol_idx].astype(bool),
                            "source_segment": source,
                            "candidate_mode": mode,
                        }
                    )
                )
    finally:
        for array in (base_candidate, base_native, suffix_candidate, suffix_native, entry_filled):
            _close_memmap(array)
    paths: dict[str, Path] = {}
    audit: dict[str, Any] = {}
    for mode, rows in rows_by_mode.items():
        frame = pd.concat(rows, ignore_index=True)
        frame = frame.sort_values(["date_idx", "symbol_idx"], kind="mergesort").reset_index(drop=True)
        frame.insert(0, "candidate_id", np.arange(len(frame), dtype=np.int64))
        if frame["trade_date"].nunique() != EXPECTED_EXTENDED_DATES:
            raise ValueError(f"{mode} extended candidate dates are incomplete")
        if bool(frame.duplicated(["trade_date", "symbol"]).any()):
            raise ValueError(f"{mode} extended candidate keys are duplicated")
        strict = frame[frame["trade_date"].astype(str).le(STRICT_SIGNAL_END)].copy()
        strict["candidate_id"] = np.arange(len(strict), dtype=np.int64)
        if strict["trade_date"].nunique() != EXPECTED_STRICT_DATES:
            raise ValueError(f"{mode} strict-tail candidate dates are incomplete")
        extended_path = staging / f"indexes/extended_d7_candidates_{mode}.parquet"
        strict_path = staging / f"indexes/strict_d7_candidates_{mode}.parquet"
        _write_parquet(extended_path, frame)
        _write_parquet(strict_path, strict)
        paths[f"extended_{mode}"] = extended_path
        paths[f"strict_{mode}"] = strict_path
        audit[f"{mode}_extended_candidate_count"] = int(len(frame))
        audit[f"{mode}_strict_candidate_count"] = int(len(strict))
        audit[f"{mode}_entry_fill_rate"] = float(frame["entry_filled"].mean())
        if not 0.95 <= float(audit[f"{mode}_entry_fill_rate"]) <= 1.0:
            raise ValueError(f"{mode} extended entry-fill coverage is unexpectedly low")
    audit["extended_date_count"] = EXPECTED_EXTENDED_DATES
    audit["strict_date_count"] = EXPECTED_STRICT_DATES
    return paths, audit


def _validate_suffix_overlay(path: Path) -> dict[str, Any]:
    manifest = _read_json(path)
    if str(manifest.get("artifact_type", "")) != "seq100_structured_180x35_2026_suffix":
        raise ValueError("not a Structured 180x35 2026 suffix overlay")
    if manifest.get("shape") != [EXPECTED_SUFFIX_DATES, EXPECTED_SYMBOL_COUNT]:
        raise ValueError("suffix overlay shape drifted")
    size = int(path.stat().st_size)
    for section in ("feature_channels", "masks"):
        for name, raw in dict(manifest[section]).items():
            meta = dict(raw)
            material = Path(str(meta["path"])).resolve()
            if not material.is_file():
                raise FileNotFoundError(material)
            expected = int(np.prod(meta["shape"])) * np.dtype(meta["dtype"]).itemsize
            if material.stat().st_size != expected or _file_sha256(material) != str(meta["sha256"]):
                raise ValueError(f"suffix material drifted: {section}/{name}")
            size += int(material.stat().st_size)
    execution_masks = dict(manifest.get("execution_masks", {}) or {})
    entry_meta = dict(execution_masks.get("entry_filled_extended", {}) or {})
    if entry_meta.get("shape") != [EXPECTED_EXTENDED_DATES, EXPECTED_SYMBOL_COUNT]:
        raise ValueError("extended entry-fill mask shape drifted")
    entry_path = Path(str(entry_meta.get("path", ""))).resolve()
    expected_entry_size = int(np.prod(entry_meta["shape"])) * np.dtype(entry_meta["dtype"]).itemsize
    if (
        not entry_path.is_file()
        or entry_path.stat().st_size != expected_entry_size
        or _file_sha256(entry_path) != str(entry_meta.get("sha256", ""))
    ):
        raise ValueError("extended entry-fill mask drifted")
    size += int(entry_path.stat().st_size)
    prefix_native = dict(manifest["prefix_native_input_valid"])
    prefix_path = Path(str(prefix_native["path"])).resolve()
    prefix_expected = int(np.prod(prefix_native["shape"])) * np.dtype(
        prefix_native["dtype"]
    ).itemsize
    if (
        not prefix_path.is_file()
        or prefix_path.stat().st_size != prefix_expected
        or _file_sha256(prefix_path) != str(prefix_native["sha256"])
    ):
        raise ValueError("native prefix qualification mask drifted")
    size += int(prefix_path.stat().st_size)
    for key, raw in dict(manifest["candidate_indexes"]).items():
        material = Path(str(raw["path"])).resolve()
        if not material.is_file() or _file_sha256(material) != str(raw["sha256"]):
            raise ValueError(f"suffix candidate material drifted: {key}")
        size += int(material.stat().st_size)
    if size > OVERLAY_SIZE_LIMIT_BYTES:
        raise ValueError("suffix overlay exceeds the 1 GiB cap")
    signal_window_indices(_read_json(FROZEN_2026_OVERLAY)["date_values"])
    manifest["validated_size_bytes"] = size
    return manifest


def build_suffix_overlay(*, study_path: Path) -> dict[str, Any]:
    manifest_path = OVERLAY_ROOT / "manifest.json"
    if manifest_path.is_file():
        return _validate_suffix_overlay(manifest_path)
    staging = OVERLAY_ROOT.with_name(OVERLAY_ROOT.name + ".staging")
    if staging.exists() and not (staging / "_compact/manifest.json").is_file():
        shutil.rmtree(staging)
    staging.mkdir(parents=True, exist_ok=True)
    compact_path = _compact_suffix_pack(staging, study_path)
    compact = _read_json(compact_path)
    global_manifest = _read_json(FROZEN_2026_OVERLAY)
    base = _read_json(BASE_PACK_MANIFEST)
    old_input = _read_json(INPUT_OVERLAY_ROOT / "manifest.json")
    global_dates = [str(value) for value in global_manifest["date_values"]]
    windows = signal_window_indices(global_dates)
    suffix_dates = list(windows["suffix_dates"])
    compact_dates = [str(value) for value in compact["date_values"]]
    compact_date_indices = [compact_dates.index(value) for value in suffix_dates]
    extended_dates = list(windows["extended_dates"])
    compact_extended_date_indices = [compact_dates.index(value) for value in extended_dates]
    base_symbols = [str(value) for value in global_manifest["symbol_values"]]
    compact_symbols = [str(value) for value in compact["symbol_values"]]
    compact_symbol_to_idx = {value: idx for idx, value in enumerate(compact_symbols)}
    missing = [value for value in base_symbols if value not in compact_symbol_to_idx]
    if missing:
        raise ValueError(f"compact suffix is missing frozen symbols: {missing[:10]}")
    compact_symbol_indices = np.asarray(
        [compact_symbol_to_idx[value] for value in base_symbols], dtype=np.int64
    )
    channels: dict[str, Any] = {}
    for name in ("daily_raw", "daily_state"):
        meta = dict(dict(compact["feature_channels"])[name])
        channels[name] = _extract_mapped_panel(
            path=staging / f"panels/{name}.float32.dat",
            compact_meta=meta,
            compact_date_indices=compact_date_indices,
            compact_symbol_indices=compact_symbol_indices,
            dtype="float32",
            columns=meta.get("columns", []),
        )
    mask_names = (
        "input_valid",
        "candidate_eligible",
        "entry_buyable",
        "signal_eligible",
        "has_bar",
        "status_valid",
        "is_st",
        "is_suspended",
        "continuity_break",
        "is_delisted",
    )
    masks: dict[str, Any] = {}
    for name in mask_names:
        masks[name] = _extract_mapped_panel(
            path=staging / f"masks/{name}.bool.dat",
            compact_meta=dict(dict(compact["masks"])[name]),
            compact_date_indices=compact_date_indices,
            compact_symbol_indices=compact_symbol_indices,
            dtype="bool",
        )
    masks["native_input_valid"] = _extract_mapped_panel(
        path=staging / "masks/native_input_valid.bool.dat",
        compact_meta=dict(dict(compact["masks"])["input_valid"]),
        compact_date_indices=compact_date_indices,
        compact_symbol_indices=compact_symbol_indices,
        dtype="bool",
    )
    prefix_native = _build_prefix_native_mask(
        base=base, path=staging / "masks/native_input_valid_prefix.bool.dat"
    )
    entry_filled_extended = _extract_mapped_panel(
        path=staging / "masks/entry_filled_extended.bool.dat",
        compact_meta=dict(dict(compact["masks"])["entry_buyable"]),
        compact_date_indices=compact_extended_date_indices,
        compact_symbol_indices=compact_symbol_indices,
        dtype="bool",
    )
    candidate_semantics = _restore_base_candidate_semantics(
        base=base,
        channels=channels,
        masks=masks,
        suffix_dates=suffix_dates,
        global_dates=global_dates,
    )
    for name in ("input_valid", "candidate_eligible"):
        masks[name]["sha256"] = _file_sha256(Path(str(masks[name]["path"])))
    turnover_source = dict(global_manifest["relative_turnover_supplement"])
    turnover_log = _open_memmap(turnover_source["log_turnover_pct"], dtype="float32")
    turnover_median = _open_memmap(
        turnover_source["past20_positive_median"], dtype="float32"
    )
    turnover_shape = (len(suffix_dates), EXPECTED_SYMBOL_COUNT, 2)
    turnover_path = staging / "panels/turnover.float32.dat"
    turnover = _create_memmap(
        turnover_path, dtype="float32", shape=turnover_shape, fill=np.nan
    )
    turnover_valid_path = staging / "masks/turnover_valid.bool.dat"
    turnover_valid = _create_memmap(
        turnover_valid_path,
        dtype="bool",
        shape=(len(suffix_dates), EXPECTED_SYMBOL_COUNT),
        fill=False,
    )
    try:
        for out_idx, trade_date in enumerate(suffix_dates):
            global_idx = global_dates.index(trade_date)
            level = np.asarray(turnover_log[global_idx], dtype=np.float32)
            relative = level - np.asarray(turnover_median[global_idx], dtype=np.float32)
            valid = np.isfinite(level) & np.isfinite(relative)
            turnover[out_idx, :, 0] = np.where(valid, level, np.nan)
            turnover[out_idx, :, 1] = np.where(valid, relative, np.nan)
            turnover_valid[out_idx] = valid
        turnover.flush()
        turnover_valid.flush()
    finally:
        for array in (turnover_log, turnover_median, turnover, turnover_valid):
            _close_memmap(array)
    channels["turnover"] = _memmap_meta(
        turnover_path,
        turnover_shape,
        dtype="float32",
        columns=input_ablation.TURNOVER_COLUMNS,
    )
    masks["turnover_valid"] = _memmap_meta(
        turnover_valid_path,
        (len(suffix_dates), EXPECTED_SYMBOL_COUNT),
        dtype="bool",
    )

    base_overlap_idx = [str(value) for value in base["date_values"]].index(OVERLAP_DATE)
    suffix_overlap_idx = 0
    overlap_checks = []
    for name in ("daily_raw", "daily_state"):
        _assert_overlap(
            name=name,
            left_meta=dict(base["feature_channels"])[name],
            right_meta=channels[name],
            left_idx=base_overlap_idx,
            right_idx=suffix_overlap_idx,
            dtype="float32",
        )
        overlap_checks.append(name)
    for name in ("input_valid", "candidate_eligible", "signal_eligible", "has_bar"):
        _assert_overlap(
            name=name,
            left_meta=dict(base["masks"])[name],
            right_meta=masks[name],
            left_idx=base_overlap_idx,
            right_idx=suffix_overlap_idx,
            dtype="bool",
        )
        overlap_checks.append(name)
    _assert_overlap(
        name="turnover",
        left_meta=dict(old_input["feature_channels"])["turnover"],
        right_meta=channels["turnover"],
        left_idx=base_overlap_idx,
        right_idx=suffix_overlap_idx,
        dtype="float32",
    )
    _assert_overlap(
        name="turnover_valid",
        left_meta=dict(old_input["masks"])["turnover_valid"],
        right_meta=masks["turnover_valid"],
        left_idx=base_overlap_idx,
        right_idx=suffix_overlap_idx,
        dtype="bool",
    )
    overlap_checks.extend(["turnover", "turnover_valid"])

    candidate_paths, candidate_audit = _build_extended_candidates(
        staging=staging,
        global_manifest=global_manifest,
        suffix_manifest={
            "date_values": suffix_dates,
            "masks": masks,
            "prefix_native_input_valid": prefix_native,
        },
        entry_filled_meta=entry_filled_extended,
        entry_date_values=extended_dates,
    )
    frozen_candidates = pd.read_parquet(
        Path(str(_read_json(FROZEN_2026_VIEW)["candidate_index_path"])).resolve(),
        columns=["trade_date", "symbol_idx", "entry_filled"],
    )
    entry_reader = _open_memmap(entry_filled_extended, dtype="bool")
    try:
        for date_idx, trade_date in enumerate(extended_dates[:EXPECTED_FULL_LABEL_DATES]):
            expected = frozen_candidates[
                frozen_candidates["trade_date"].astype(str).eq(trade_date)
            ].sort_values("symbol_idx", kind="mergesort")
            actual = np.asarray(entry_reader[date_idx, expected["symbol_idx"].to_numpy(dtype=np.int64)])
            if not np.array_equal(actual, expected["entry_filled"].to_numpy(dtype=bool)):
                raise ValueError(f"entry-fill overlap regression failed for {trade_date}")
    finally:
        _close_memmap(entry_reader)
    payload = {
        "schema_version": 1,
        "artifact_type": "seq100_structured_180x35_2026_suffix",
        "created_at": _now(),
        "shape": [len(suffix_dates), EXPECTED_SYMBOL_COUNT],
        "date_values": suffix_dates,
        "date_start": suffix_dates[0],
        "date_end": suffix_dates[-1],
        "global_date_start_idx": global_dates.index(suffix_dates[0]),
        "global_date_end_idx": global_dates.index(suffix_dates[-1]),
        "symbol_values": base_symbols,
        "feature_channels": channels,
        "masks": masks,
        "execution_masks": {
            "entry_filled_extended": entry_filled_extended,
        },
        "prefix_native_input_valid": prefix_native,
        "candidate_indexes": {
            key: {
                "path": str(path.resolve()),
                "sha256": _file_sha256(path),
                "row_count": int(
                    candidate_audit[
                        f"{key.removeprefix('extended_').removeprefix('strict_')}_"
                        f"{'extended' if key.startswith('extended_') else 'strict'}_candidate_count"
                    ]
                ),
            }
            for key, path in candidate_paths.items()
        },
        "candidate_audit": candidate_audit,
        "candidate_semantics": candidate_semantics,
        "overlap_regression": {
            "trade_date": OVERLAP_DATE,
            "status": "ok",
            "checked_material": overlap_checks,
        },
        "source": {
            "qdp_snapshot_sha256": qdp_snapshot_sha256(QDP_ROOT),
            "base_pack_sha256": _file_sha256(BASE_PACK_MANIFEST),
            "frozen_2026_overlay_sha256": _file_sha256(FROZEN_2026_OVERLAY),
            "compact_qdp_source_manifests": compact.get("qdp_source_manifests", {}),
        },
        "protected_boundaries": {
            "qdp_read_only": True,
            "provider_called": False,
            "base_pack_copied": False,
            "historical_feature_panel_copied": False,
        },
    }
    if (staging / "_compact").exists():
        shutil.rmtree(staging / "_compact")
    promoted = input_ablation._replace_path_prefix(
        payload, str(staging.resolve()), str(OVERLAY_ROOT.resolve())
    )
    _write_json(staging / "manifest.json", promoted)
    if _directory_size(staging) > OVERLAY_SIZE_LIMIT_BYTES:
        raise ValueError("prepared suffix overlay exceeds the 1 GiB cap")
    if OVERLAY_ROOT.exists():
        raise FileExistsError(OVERLAY_ROOT)
    os.replace(staging, OVERLAY_ROOT)
    return _validate_suffix_overlay(OVERLAY_ROOT / "manifest.json")


def _build_training_view(
    *, study_path: Path, suffix: Mapping[str, Any]
) -> tuple[Path, dict[str, Any]]:
    source = _read_json(FROZEN_2026_VIEW)
    source_index = Path(str(source["sample_index_path"])).resolve()
    target_index = OVERLAY_ROOT / "indexes/development_2026_lookback180.parquet"
    frame = pd.read_parquet(source_index)
    native_meta = dict(suffix["prefix_native_input_valid"])
    native = _open_memmap(native_meta, dtype="bool")
    date_idx = frame["date_idx"].to_numpy(dtype=np.int64)
    symbol_idx = frame["symbol_idx"].to_numpy(dtype=np.int64)
    qualification = np.asarray(native[date_idx, symbol_idx], dtype=bool)
    _close_memmap(native)
    split = frame["split"].astype(str)
    selected = frame[
        qualification
        & (
            ~split.eq("train")
            | frame["date_idx"].astype(np.int64).ge(LOOKBACK_DAYS - 1)
        )
    ].copy()
    selected = selected.sort_values(["date_idx", "symbol_idx"], kind="mergesort").reset_index(drop=True)
    train = selected[selected["split"].astype(str).eq("train")]
    development = selected[selected["split"].astype(str).eq("development")]
    if (
        len(train) != EXPECTED_NATIVE_TRAIN_ROWS
        or str(train["trade_date"].min()) != "2010-09-29"
        or str(train["trade_date"].max()) != SAFE_TRAIN_SIGNAL_END
        or len(development) != EXPECTED_NATIVE_FULL_LABEL_SUPERVISED
    ):
        raise ValueError("180-day 2026 sample boundary drifted")
    if target_index.is_file():
        observed = pd.read_parquet(target_index)
        if not observed.equals(selected):
            raise ValueError("stored 180-day 2026 sample index drifted")
    else:
        _write_parquet(target_index, selected)

    old_overlay = _read_json(INPUT_OVERLAY_ROOT / "manifest.json")
    source_candidates = pd.read_parquet(Path(str(source["candidate_index_path"])).resolve())
    candidate_native = _open_memmap(native_meta, dtype="bool")
    candidate_qualified = np.asarray(
        candidate_native[
            source_candidates["date_idx"].to_numpy(dtype=np.int64),
            source_candidates["symbol_idx"].to_numpy(dtype=np.int64),
        ],
        dtype=bool,
    )
    _close_memmap(candidate_native)
    candidates = source_candidates[candidate_qualified].copy().reset_index(drop=True)
    candidates["candidate_id"] = np.arange(len(candidates), dtype=np.int64)
    if len(candidates) != EXPECTED_NATIVE_FULL_LABEL_CANDIDATES:
        raise ValueError("native 180-day full-label candidate count drifted")
    target_candidates = OVERLAY_ROOT / "indexes/full_label_candidates_native_180.parquet"
    _write_parquet(target_candidates, candidates)
    view = json.loads(json.dumps(source, ensure_ascii=False))
    view["created_at"] = _now()
    view["lookback_days"] = LOOKBACK_DAYS
    view["sample_index_path"] = str(target_index.resolve())
    view["sample_count"] = int(len(selected))
    view["sample_count_by_split"] = {
        "train": int(len(train)),
        "development": int(len(development)),
    }
    view["candidate_index_path"] = str(target_candidates.resolve())
    view["candidate_count"] = int(len(candidates))
    view["candidate_count_by_split"] = {"development": int(len(candidates))}
    channels = dict(view["feature_channels"])
    channels["turnover"] = dict(old_overlay["feature_channels"])["turnover"]
    view["feature_channels"] = channels
    masks = dict(view["masks"])
    masks["turnover_valid"] = dict(old_overlay["masks"])["turnover_valid"]
    view["masks"] = masks
    semantics = dict(view["data_semantics"])
    semantics["model_input_mask_features"] = ["turnover_valid"]
    semantics["structured_180x35_2026"] = {
        "lookback_days": LOOKBACK_DAYS,
        "input_dim": INPUT_DIM,
        "input_channel_profile": INPUT_CHANNEL_PROFILE,
    }
    view["data_semantics"] = semantics
    development_start_idx = [str(value) for value in view["date_values"]].index(SIGNAL_START)
    normalization = dict(view["normalization"])
    normalization["turnover"] = walkforward._fit_channel_normalization(
        channels["turnover"], start_idx=0, end_idx_exclusive=development_start_idx
    )
    view["normalization"] = normalization
    binding = {
        "path": str(study_path.resolve()),
        "contract_id": STUDY_ID,
        "contract_sha256": str(_read_json(study_path)["contract_sha256"]),
        "contract_file_sha256": str(_read_json(study_path)["contract_sha256"]),
    }
    view["research_contract"] = binding
    view["development_contract"] = binding
    artifact = dict(view.get("artifact_view", {}) or {})
    artifact.update(
        {
            "schema_version": 1,
            "view_id": "seq100_structured_180x35_development_2026",
            "view_type": "structured_180x35_2026_development_view",
        }
    )
    view["artifact_view"] = artifact
    view["structured_180x35_2026"] = {
        "suffix_overlay": str((OVERLAY_ROOT / "manifest.json").resolve()),
        "suffix_overlay_sha256": _file_sha256(OVERLAY_ROOT / "manifest.json"),
        "full_label_candidate_count_native_180": int(len(candidates)),
        "full_label_candidate_count_common_100": EXPECTED_FULL_LABEL_CANDIDATES,
        "extended_candidate_count_native_180": int(
            suffix["candidate_audit"]["native_180_extended_candidate_count"]
        ),
        "extended_candidate_count_common_100": int(
            suffix["candidate_audit"]["common_100_extended_candidate_count"]
        ),
        "qualification_is_signal_time_only": True,
    }
    walk = dict(view["development_walkforward"])
    walk.update(
        {
            "candidate_universe_rule": "signal_day_common_candidate_and_complete_native_180_input_history",
            "candidate_count": int(len(candidates)),
            "supervised_development_count": int(len(development)),
            "unsupervised_candidate_count": int(len(candidates) - len(development)),
        }
    )
    view["development_walkforward"] = walk
    view["development_fold_training_contract"] = (
        walkforward._compute_development_fold_training_contract(view)
    )
    target_view = STUDY_ROOT / "fold_view_2026.json"
    _write_json(target_view, view)
    verification = walkforward.verify_development_walkforward_view(target_view)
    if str(verification.get("status", "")) != "ok":
        raise ValueError(f"Structured 180x35 2026 view verification failed: {verification}")

    return target_view, {
        "train_row_count": int(len(train)),
        "development_row_count": int(len(development)),
        "first_train_signal": str(train["trade_date"].min()),
        "last_train_signal": str(train["trade_date"].max()),
        "sample_index_sha256": _file_sha256(target_index),
        "fold_contract_sha256": str(view["development_fold_training_contract"]["sha256"]),
        "normalization_cutoff_exclusive": str(normalization["fit_date_end_exclusive"]),
        "common_train_row_count": EXPECTED_COMMON_TRAIN_ROWS,
        "native_full_label_candidate_count": int(len(candidates)),
        "common_full_label_candidate_count": EXPECTED_FULL_LABEL_CANDIDATES,
        "excluded_full_label_candidate_count": int(
            EXPECTED_FULL_LABEL_CANDIDATES - len(candidates)
        ),
    }


def _shared_integrity(
    *, suffix: Mapping[str, Any], training_view: Path
) -> dict[str, Any]:
    source_view = _read_json(training_view)
    full_candidate = Path(str(source_view["candidate_index_path"])).resolve()
    common_full_candidate = Path(str(_read_json(FROZEN_2026_VIEW)["candidate_index_path"])).resolve()
    indexes = dict(suffix["candidate_indexes"])
    candidate_detail = {
        "full_label_native_180": integrity.candidate_semantic_sha256(full_candidate),
        "full_label_common_100_audit": integrity.candidate_semantic_sha256(common_full_candidate),
        **{
            key: integrity.candidate_semantic_sha256(Path(str(raw["path"])).resolve())
            for key, raw in indexes.items()
        },
    }
    data_payload = {
        "schema_version": "seq100_structured_180x35_2026_data_snapshot_v1",
        "qdp_snapshot_sha256": qdp_snapshot_sha256(QDP_ROOT),
        "base_pack_sha256": _file_sha256(BASE_PACK_MANIFEST),
        "frozen_2026_overlay_sha256": _file_sha256(FROZEN_2026_OVERLAY),
        "input_overlay_sha256": _file_sha256(INPUT_OVERLAY_ROOT / "manifest.json"),
        "suffix_overlay_sha256": _file_sha256(OVERLAY_ROOT / "manifest.json"),
        "extended_entry_filled_mask": dict(suffix["execution_masks"])["entry_filled_extended"],
        "labels": source_view["label_arrays"],
        "execution_arrays": source_view["execution_arrays"],
        "execution_masks": {
            "entry_filled": dict(source_view["masks"])["entry_filled"],
            "exit_sellable": dict(source_view["masks"])["exit_sellable"],
        },
    }
    evaluation_rules = {
        "schema_version": "seq100_structured_180x35_2026_evaluation_v1",
        "metric_semantic_version": "seq100_structured_180x35_2026_metrics_v1",
        "full_label_window": [SIGNAL_START, FULL_LABEL_END],
        "extended_d7_window": [SIGNAL_START, EXTENDED_SIGNAL_END],
        "strict_tail_window": [SIGNAL_START, STRICT_SIGNAL_END],
        "top_k": [1, 3],
        "fixed_exit_day": 7,
        "maximum_deferral_sessions": 20,
        "same_day_ordering": "entry_checks_before_exits",
        "no_fallback_selection": True,
        "no_leverage": True,
        "no_pyramiding": True,
        "starting_cash_cny": STARTING_CASH_CNY,
        "candidate_mode": "native_180",
        "common_100_candidate_used_for_coverage_audit_only": True,
    }
    return {
        "schema_version": "seq100_integrity_v2",
        "data_snapshot_sha256": integrity.canonical_sha256(data_payload),
        "candidate_semantic_sha256": integrity.canonical_sha256(
            {"schema_version": 1, "candidate_sets": candidate_detail}
        ),
        "candidate_semantic_detail": candidate_detail,
        "evaluation_contract_sha256": integrity.evaluation_contract_sha256(
            source_view, evaluation_rules
        ),
        "evaluation_rules": evaluation_rules,
    }


def prepare_structured_180x35_2026(
    *, study_root: Path = STUDY_ROOT
) -> dict[str, Any]:
    _assert_no_research_process()
    study_path, study = _load_or_create_study(study_root)
    material = dict(study.get("material", {}) or {})
    suffix_manifest = OVERLAY_ROOT / "manifest.json"
    material_matches_overlay = (
        suffix_manifest.is_file()
        and _file_sha256(suffix_manifest) == str(material.get("suffix_overlay_sha256", ""))
    )
    if material.get("training_view") and material_matches_overlay:
        verify_structured_180x35_2026(study_root=study_root, require_complete=False)
        return _read_json(study_path)
    _update_runtime(study_path, status="preparing", error=None)
    try:
        source = _source_identity()
        suffix = build_suffix_overlay(study_path=study_path)
        if material.get("training_view"):
            existing_view = Path(str(material["training_view"])).resolve()
            existing_payload = _read_json(existing_view)
            backing_indexes = (
                Path(str(existing_payload["sample_index_path"])).resolve(),
                Path(str(existing_payload["candidate_index_path"])).resolve(),
            )
        else:
            existing_view = Path()
            backing_indexes = ()
        if material.get("training_view") and all(path.is_file() for path in backing_indexes):
            training_view = existing_view
            fold_audit = dict(material["fold_audit"])
        else:
            training_view, fold_audit = _build_training_view(
                study_path=study_path, suffix=suffix
            )
        shared = _shared_integrity(suffix=suffix, training_view=training_view)
        study = _read_json(study_path)
        study["material"] = {
            "source_identity": source,
            "suffix_overlay": str((OVERLAY_ROOT / "manifest.json").resolve()),
            "suffix_overlay_sha256": _file_sha256(OVERLAY_ROOT / "manifest.json"),
            "suffix_overlay_size_bytes": int(suffix["validated_size_bytes"]),
            "training_view": str(training_view.resolve()),
            "training_view_sha256": _file_sha256(training_view),
            "fold_audit": fold_audit,
            "integrity": shared,
            "qdp_changed": False,
            "provider_called": False,
            "base_pack_changed": False,
            "old_checkpoint_changed": False,
            "stage3_changed": False,
            "live_state_changed": False,
        }
        study["runtime"] = {
            **dict(study.get("runtime", {}) or {}),
            "status": "prepared",
            "updated_at": _now(),
            "current_task": None,
            "error": None,
        }
        _write_json(study_path, study)
        verify_structured_180x35_2026(study_root=study_root, require_complete=False)
        return _read_json(study_path)
    except Exception as exc:
        _update_runtime(study_path, status="prepare_failed", error=str(exc))
        raise


def _run_glob() -> list[Path]:
    return sorted(
        (STUDY_ROOT / "runs/development").glob(
            f"seq100_structured_180x35_{TARGET_YEAR}_seed{SEED}_*"
        )
    )


def _completed_runs() -> tuple[list[Path], list[Path]]:
    complete: list[Path] = []
    partial: list[Path] = []
    for path in _run_glob():
        if (path / "sequence_path_training_summary.json").is_file() and (
            path / "best_model.pt"
        ).is_file():
            try:
                _validate_training_run(path)
            except (EOFError, FileNotFoundError, KeyError, OSError, RuntimeError, TypeError, ValueError):
                partial.append(path)
            else:
                complete.append(path)
        else:
            partial.append(path)
    if len(complete) > 1:
        raise ValueError("multiple valid 2026 Structured 180x35 runs exist")
    return complete, partial


def _validate_training_run(run_dir: Path) -> dict[str, Any]:
    study_path = STUDY_ROOT / "study.json"
    summary_path = run_dir / "sequence_path_training_summary.json"
    progress_path = run_dir / "progress.json"
    checkpoint_path = run_dir / "best_model.pt"
    for path in (summary_path, progress_path, checkpoint_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    summary = development_validate_run_summary(summary_path)
    progress = _read_json(progress_path)
    if str(progress.get("status", "")) != "completed":
        raise ValueError("2026 training progress is not terminal")
    config = dict(summary.get("resolved_training_config", {}) or {})
    expected = {
        "model_type": "gru_structured_joint_turnover",
        "input_channel_profile": INPUT_CHANNEL_PROFILE,
        "seed": SEED,
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
    }
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError(f"2026 training config drifted for {key}")
    if int(summary.get("completed_epochs", 0)) != EPOCHS:
        raise ValueError("2026 training did not complete exactly one epoch")
    if int(summary.get("lookback_days", 0)) != LOOKBACK_DAYS:
        raise ValueError("2026 training lookback drifted")
    if _file_sha256(checkpoint_path) != str(summary.get("best_checkpoint_sha256", "")):
        raise ValueError("2026 checkpoint checksum drifted")
    view = _read_json(STUDY_ROOT / "fold_view_2026.json")
    if Path(str(summary.get("pack_manifest", ""))).resolve() != (
        STUDY_ROOT / "fold_view_2026.json"
    ).resolve():
        raise ValueError("2026 training used a different fold view")
    fold_contract = dict(view["development_fold_training_contract"])
    if dict(summary.get("development_fold_training_contract", {}) or {}) != fold_contract:
        raise ValueError("2026 training fold contract drifted")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if int(checkpoint.get("input_dim", 0)) != INPUT_DIM:
        raise ValueError("2026 checkpoint input dimension drifted")
    projection = tuple(checkpoint["model_state_dict"]["proj.weight"].shape)
    if projection != (128, INPUT_DIM):
        raise ValueError("2026 checkpoint projection shape drifted")
    if dict(checkpoint.get("fold_training_contract", {}) or {}) != fold_contract:
        raise ValueError("2026 checkpoint fold contract drifted")
    candidate_hash = _file_sha256(Path(str(view["candidate_index_path"])).resolve())
    if str(summary.get("candidate_index_sha256", "")) != candidate_hash:
        raise ValueError("2026 checkpoint candidate binding drifted")
    return summary


def development_validate_run_summary(path: Path) -> dict[str, Any]:
    """Keep run validation in one place while tolerating development-only views."""
    return development._validate_run_summary(path, TARGET_YEAR)


def _append_event(payload: Mapping[str, Any]) -> None:
    path = STUDY_ROOT / "monitor_events.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    event = {"timestamp": _now(), **dict(payload)}
    with path.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(event, ensure_ascii=False, default=input_ablation._json_default, allow_nan=False)
            + "\n"
        )
    _write_json(STUDY_ROOT / "monitor.json", event)
    try:
        print(f"structured-180x35-2026: {event.get('message', event.get('event', 'status'))}", flush=True)
    except OSError:
        pass


def _archive_partial_runs(paths: Sequence[Path]) -> list[str]:
    archived: list[str] = []
    target_root = STUDY_ROOT / "runs/failed/structured_180x35_2026"
    target_root.mkdir(parents=True, exist_ok=True)
    for number, source in enumerate(paths, start=1):
        if STUDY_ROOT.resolve() not in source.resolve().parents:
            raise ValueError(f"refusing to archive a run outside the study: {source}")
        target = target_root / f"{source.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{number}"
        shutil.move(str(source), str(target))
        archived.append(str(target.resolve()))
    return archived


def _parse_timestamp(value: Any) -> float | None:
    try:
        return datetime.fromisoformat(str(value)).timestamp()
    except (TypeError, ValueError):
        return None


def _supervise_training(command: Sequence[str]) -> Path:
    log_root = STUDY_ROOT / "logs"
    log_root.mkdir(parents=True, exist_ok=True)
    memory_path = log_root / "memory_guard_structured_180x35_2026.json"
    stdout_path = log_root / "worker_structured_180x35_2026.log"
    guard_command = [
        str(PYTHON),
        str((WORKSPACE_ROOT / "tools/memory_guard.py").resolve()),
        "--min-available-gb",
        str(MEMORY_GUARD_GIB),
        "--interval-seconds",
        str(MEMORY_INTERVAL_SECONDS),
        "--consecutive-breaches",
        str(MEMORY_CONSECUTIVE_BREACHES),
        "--log-json",
        str(memory_path),
        "--",
        *[str(value) for value in command],
    ]
    before = set(_run_glob())
    _append_event({"event": "training_started", "status": "running", "message": "start Structured 180x35 2026 training"})
    current_run: Path | None = None
    last_phase: str | None = None
    last_epoch: int | None = None
    last_bucket: int | None = None
    stale_reported = False
    with stdout_path.open("w", encoding="utf-8") as stdout:
        process = subprocess.Popen(
            guard_command,
            cwd=str(WORKSPACE_ROOT),
            stdout=stdout,
            stderr=subprocess.STDOUT,
        )
        while process.poll() is None:
            if current_run is None:
                created = sorted(set(_run_glob()).difference(before))
                if len(created) > 1:
                    process.kill()
                    raise RuntimeError("training created multiple run directories")
                if created:
                    current_run = created[0]
            progress: dict[str, Any] = {}
            if current_run is not None and (current_run / "progress.json").is_file():
                try:
                    progress = _read_json(current_run / "progress.json")
                except (OSError, json.JSONDecodeError):
                    progress = {}
            phase = str(progress.get("status", "starting"))
            epoch = int(progress.get("epoch", 0) or 0)
            total = int(progress.get("total_batches", 0) or 0)
            batch = int(progress.get("batch", 0) or 0)
            bucket = min(100, int(math.floor(10.0 * batch / total) * 10)) if total else 0
            if phase != last_phase or epoch != last_epoch or bucket != last_bucket:
                available = float(psutil.virtual_memory().available / 1024**3)
                _append_event(
                    {
                        "event": "progress",
                        "status": "running",
                        "phase": phase,
                        "epoch": epoch,
                        "batch": batch,
                        "total_batches": total,
                        "batch_percent_bucket": bucket,
                        "available_memory_gib": available,
                        "message": f"phase={phase} epoch={epoch} batch={batch}/{total}",
                    }
                )
                last_phase, last_epoch, last_bucket = phase, epoch, bucket
            updated = _parse_timestamp(progress.get("updated_at"))
            stale = updated is not None and time.time() - updated > STALE_SECONDS
            if stale and not stale_reported:
                _append_event(
                    {
                        "event": "stale_progress",
                        "status": "warning",
                        "stale_seconds": time.time() - float(updated),
                        "message": "training progress stale for more than 15 minutes",
                    }
                )
                stale_reported = True
            elif not stale:
                stale_reported = False
            time.sleep(MONITOR_POLL_SECONDS)
        exit_code = int(process.returncode or 0)
    memory = _read_json(memory_path) if memory_path.is_file() else {}
    if current_run is None:
        created = sorted(set(_run_glob()).difference(before))
        current_run = created[-1] if created else None
    if current_run is not None:
        _write_json(current_run / "memory_guard.json", memory)
    if exit_code != 0:
        event = "low_memory_terminated" if str(memory.get("status", "")) == "killed_low_available_memory" else "training_failed"
        _append_event({"event": event, "status": "failed", "exit_code": exit_code, "message": f"training exited with code {exit_code}"})
        raise subprocess.CalledProcessError(exit_code, guard_command)
    if current_run is None:
        raise RuntimeError("training produced no run directory")
    _validate_training_run(current_run)
    _append_event({"event": "training_completed", "status": "completed", "run_dir": str(current_run.resolve()), "message": "training completed"})
    return current_run


def run_structured_180x35_2026(
    *, study_root: Path = STUDY_ROOT, max_tasks: int = 1
) -> dict[str, Any]:
    if int(max_tasks) < 0:
        raise ValueError("max_tasks must be non-negative")
    _assert_no_research_process()
    study_path = study_root.resolve() / "study.json"
    if not study_path.is_file():
        prepare_structured_180x35_2026(study_root=study_root)
    verify_structured_180x35_2026(study_root=study_root, require_complete=False)
    complete, partial = _completed_runs()
    if complete:
        _update_runtime(study_path, status="training_completed", current_task=None, error=None)
        return status_structured_180x35_2026(study_root=study_root)
    if partial:
        archived = _archive_partial_runs(partial)
        _append_event({"event": "partial_archived", "status": "recovering", "archived_runs": archived, "message": "archived partial training artifacts"})
    if int(max_tasks) == 0:
        max_tasks = 1
    view = Path(str(_read_json(study_path)["material"]["training_view"])).resolve()
    profile = input_ablation._base_structured_profile(BATCH_SIZE)
    profile = replace(
        profile,
        store_view=view,
        output_root=STUDY_ROOT / "runs/development",
        run_tag=f"seq100_structured_180x35_{TARGET_YEAR}_seed{SEED}",
        epochs=EPOCHS,
        seed=SEED,
        top_k="1,3,5,10",
        input_channel_profile=INPUT_CHANNEL_PROFILE,
        prediction_mode="compact",
        evaluation_mode=training.EVALUATION_MODE_DEVELOPMENT,
        early_stopping_patience=1,
        early_stopping_min_delta=0.0,
        early_stopping_metric=training.EARLY_STOPPING_METRIC_DEVELOPMENT_PRICE_TOTAL_LOSS,
        early_stopping_mode=training.EARLY_STOPPING_MODE_MIN,
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
    _update_runtime(study_path, status="training", current_task="structured_180x35_2026", error=None)
    run_dir = _supervise_training(command)
    for source, target in (
        (run_dir / "best_model.pt", STUDY_ROOT / "best_model.pt"),
        (
            run_dir / "sequence_path_training_summary.json",
            STUDY_ROOT / "sequence_path_training_summary.json",
        ),
    ):
        temporary = target.with_name(target.name + ".tmp")
        shutil.copy2(source, temporary)
        os.replace(temporary, target)
    _update_runtime(study_path, status="training_completed", current_task=None, error=None)
    return status_structured_180x35_2026(study_root=study_root)


class ExtendedFeatureAccessor:
    """Read base-prefix and compact-suffix panels as one logical panel."""

    def __init__(self, dataset: training.SequencePathPackDataset, suffix: Mapping[str, Any]):
        self.dataset = dataset
        self.suffix = dict(suffix)
        frozen = _read_json(FROZEN_2026_OVERLAY)
        self.date_values = [str(value) for value in frozen["date_values"]]
        self.suffix_start_idx = self.date_values.index(OVERLAP_DATE)
        self.suffix_dates = [str(value) for value in suffix["date_values"]]
        self.suffix_meta = dict(suffix["feature_channels"])
        self.suffix_masks = dict(suffix["masks"])
        self._base_arrays: dict[str, np.memmap] = {}
        self._suffix_arrays: dict[str, np.memmap] = {}
        for name in dataset.channel_order:
            self._base_arrays[name] = dataset.feature_arrays[name]
            if name not in self.suffix_meta:
                raise ValueError(f"suffix overlay has no channel {name}")
            self._suffix_arrays[name] = _open_memmap(self.suffix_meta[name], dtype="float32")
        self._base_masks: dict[str, np.memmap] = dataset.input_mask_arrays
        self._suffix_masks_open: dict[str, np.memmap] = {}
        for name in dataset.input_mask_features:
            if name not in self.suffix_masks:
                raise ValueError(f"suffix overlay has no input mask {name}")
            self._suffix_masks_open[name] = _open_memmap(self.suffix_masks[name], dtype="bool")

    def close(self) -> None:
        for array in self._suffix_arrays.values():
            _close_memmap(array)
        for array in self._suffix_masks_open.values():
            _close_memmap(array)
        self._suffix_arrays.clear()
        self._suffix_masks_open.clear()

    def _block(
        self,
        *,
        name: str,
        start: int,
        stop: int,
        symbols: np.ndarray,
        mask: bool = False,
    ) -> np.ndarray:
        if start < 0 or stop <= start:
            raise ValueError("invalid feature block bounds")
        width = int(stop - start)
        suffix_local_start = int(start - self.suffix_start_idx)
        suffix_local_stop = int(stop - self.suffix_start_idx)
        base_array = self._base_masks[name] if mask else self._base_arrays[name]
        suffix_array = self._suffix_masks_open[name] if mask else self._suffix_arrays[name]
        base_shape = tuple(int(value) for value in base_array.shape)
        tail = tuple(int(value) for value in base_shape[2:])
        out_shape = (width, len(symbols), *tail)
        if mask:
            out = np.zeros(out_shape, dtype=bool)
        else:
            out = np.full(out_shape, np.nan, dtype=np.float32)
        cursor = 0
        if start < self.suffix_start_idx:
            base_stop = min(stop, self.suffix_start_idx, base_shape[0])
            if base_stop > start:
                out[: base_stop - start] = np.asarray(
                    base_array[start:base_stop, symbols], dtype=out.dtype
                )
                cursor = base_stop - start
        suffix_start = max(start, self.suffix_start_idx)
        if stop > suffix_start:
            local_start = suffix_start - self.suffix_start_idx
            local_stop = local_start + (stop - suffix_start)
            out[cursor:] = np.asarray(
                suffix_array[local_start:local_stop, symbols], dtype=out.dtype
            )
        return out

    def batch_x(
        self, date_indices: np.ndarray, symbol_indices: np.ndarray
    ) -> np.ndarray:
        dates = np.asarray(date_indices, dtype=np.int64)
        symbols = np.asarray(symbol_indices, dtype=np.int64)
        if dates.ndim != 1 or symbols.ndim != 1 or len(dates) != len(symbols):
            raise ValueError("extended feature batch indices are invalid")
        batch = int(len(dates))
        parts: list[np.ndarray] = []
        for name in self.dataset.channel_order:
            count = len(self.dataset.feature_columns[name])
            values = np.empty((batch, self.dataset.lookback_days, count), dtype=np.float32)
            for current in np.unique(dates):
                selected = dates == int(current)
                rows = np.flatnonzero(selected)
                start = int(current) - int(self.dataset.lookback_days) + 1
                stop = int(current) + 1
                block = self._block(
                    name=name,
                    start=start,
                    stop=stop,
                    symbols=symbols[rows],
                )
                values[rows] = np.transpose(block, (1, 0, 2))
            mean = self.dataset.normalization_mean[name]
            std = self.dataset.normalization_std[name]
            values = (values - mean.reshape(1, 1, -1)) / std.reshape(1, 1, -1)
            parts.append(np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32))
        if self.dataset.input_mask_features:
            masks = np.empty(
                (batch, self.dataset.lookback_days, len(self.dataset.input_mask_features)),
                dtype=np.float32,
            )
            for field_idx, name in enumerate(self.dataset.input_mask_features):
                for current in np.unique(dates):
                    selected = dates == int(current)
                    rows = np.flatnonzero(selected)
                    start = int(current) - int(self.dataset.lookback_days) + 1
                    stop = int(current) + 1
                    block = self._block(
                        name=name,
                        start=start,
                        stop=stop,
                        symbols=symbols[rows],
                        mask=True,
                    )
                    masks[rows, :, field_idx] = np.transpose(block, (1, 0))
            parts.append(masks)
        return np.concatenate(parts, axis=2).astype(np.float32, copy=False)


def _model_paths(model_id: str) -> tuple[Path, Path, str, int]:
    if model_id == MODEL_2026_180:
        complete, partial = _completed_runs()
        if len(complete) != 1 or partial:
            raise ValueError("new 2026 Structured 180x35 checkpoint is incomplete")
        return complete[0], STUDY_ROOT / "fold_view_2026.json", "structured_joint_turnover", TARGET_YEAR
    raise ValueError(f"unknown model id: {model_id}")


def _load_model_and_dataset(
    model_id: str, *, device: torch.device
) -> tuple[torch.nn.Module, training.SequencePathPackDataset, Path, dict[str, Any]]:
    run_dir, view_path, _profile, vintage = _model_paths(model_id)
    view = _read_json(view_path)
    dataset = training.SequencePathPackDataset(
        view,
        split="development",
        max_samples=0,
        input_channel_profile=INPUT_CHANNEL_PROFILE,
        index_role="candidate",
    )
    if len(dataset) != EXPECTED_NATIVE_FULL_LABEL_CANDIDATES:
        raise ValueError(f"full-label candidate count drifted for {model_id}")
    model, summary = structured._load_checkpoint_model(run_dir, dataset, device=device)
    return model, dataset, run_dir, summary


def _split_metrics_from_summary(summary: Mapping[str, Any]) -> dict[str, Any]:
    rows = [row for row in list(summary.get("split_metrics", []) or []) if str(row.get("split", "")) == "development"]
    if len(rows) != 1:
        raise ValueError("training summary has no unique development split metrics")
    return dict(rows[0])


def _filter_development(frame: pd.DataFrame) -> pd.DataFrame:
    if "split" in frame.columns:
        return frame[frame["split"].astype(str).eq("development")].copy()
    return frame.copy()


def _full_label_task_dir(model_id: str) -> Path:
    return STUDY_ROOT / "evaluations/full_label" / model_id


def _write_full_label_result(
    *, model_id: str, topk: pd.DataFrame, daily_ic: pd.DataFrame,
    daily_topk: pd.DataFrame, topk_candidates: pd.DataFrame,
    split_metrics: Mapping[str, Any], summary: Mapping[str, Any],
    inference_performed: bool,
) -> dict[str, Any]:
    output_dir = _full_label_task_dir(model_id)
    output_dir.mkdir(parents=True, exist_ok=True)
    topk = _filter_development(topk)
    daily_ic = _filter_development(daily_ic)
    daily_topk = _filter_development(daily_topk)
    topk_candidates = _filter_development(topk_candidates)
    paths = {
        "topk_metrics_csv": output_dir / "topk_metrics.csv",
        "daily_rank_ic_csv": output_dir / "daily_rank_ic.csv",
        "daily_topk_metrics_csv": output_dir / "daily_topk_metrics.csv",
        "topk_candidates_parquet": output_dir / "topk_candidates.parquet",
    }
    _write_csv(paths["topk_metrics_csv"], topk)
    _write_csv(paths["daily_rank_ic_csv"], daily_ic)
    _write_csv(paths["daily_topk_metrics_csv"], daily_topk)
    _write_parquet(paths["topk_candidates_parquet"], topk_candidates)
    metrics = comparison_2026._evaluation_metrics_2026(
        topk=topk, daily_topk=daily_topk, split_metrics=split_metrics
    )
    metrics["candidate_score_coverage"] = float(metrics.get("candidate_score_coverage", 1.0))
    metrics["full_label_window"] = f"{SIGNAL_START}..{FULL_LABEL_END}"
    metrics["candidate_count"] = int(split_metrics.get("row_count", 0))
    metrics["date_count"] = int(split_metrics.get("date_count", EXPECTED_FULL_LABEL_DATES))
    run_dir, view_path, _profile, vintage = _model_paths(model_id)
    checkpoint_path = Path(str(summary.get("best_checkpoint", run_dir / "best_model.pt"))).resolve()
    checkpoint_sha = str(summary.get("best_checkpoint_sha256", _file_sha256(checkpoint_path)))
    normalization = _read_json(view_path).get("normalization", {})
    architecture = {
        "model_type": str(dict(summary.get("resolved_training_config", {}) or {}).get("model_type", "gru_structured_joint_turnover")),
        "input_dim": int(summary.get("input_dim", INPUT_DIM) or INPUT_DIM),
        "hidden_dim": int(dict(summary.get("resolved_training_config", {}) or {}).get("hidden_dim", 128)),
        "layers": int(dict(summary.get("resolved_training_config", {}) or {}).get("layers", 2)),
        "dropout": float(dict(summary.get("resolved_training_config", {}) or {}).get("dropout", 0.1)),
        "lookback_days": int(summary.get("lookback_days", _read_json(view_path).get("lookback_days", 100))),
    }
    model_record = {
        "checkpoint_sha256": checkpoint_sha,
        "normalization": normalization,
        "architecture": architecture,
    }
    task = {
        "schema_version": 1,
        "status": "completed",
        **_task_identity("full_label"),
        "model_id": model_id,
        "checkpoint_vintage": int(vintage),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha,
        "normalization_sha256": _canonical_digest(normalization),
        "model_bundle_sha256": integrity.model_bundle_sha256(model_record),
        "metrics": metrics,
        "inference_performed": bool(inference_performed),
        "outputs": {name: str(path.resolve()) for name, path in paths.items()},
        "output_sha256": {name: _file_sha256(path) for name, path in paths.items()},
        "result_artifact_sha256": integrity.result_artifact_sha256(paths),
        "candidate_count": EXPECTED_NATIVE_FULL_LABEL_CANDIDATES,
        "date_count": EXPECTED_FULL_LABEL_DATES,
        "completed_at": _now(),
    }
    _write_json(output_dir / "evaluation.json", task)
    return task


def _evaluate_full_label_model(model_id: str) -> dict[str, Any]:
    target = _full_label_task_dir(model_id) / "evaluation.json"
    if _job_is_complete("full_label", (model_id,)):
        return _read_json(target)
    run_dir, view_path, _profile, _vintage = _model_paths(model_id)
    summary = _read_json(run_dir / "sequence_path_training_summary.json")
    if model_id == MODEL_2026_180:
        source_dir = run_dir
        artifacts = {
            "topk_metrics_csv": source_dir / "topk_metrics.csv",
            "daily_rank_ic_csv": source_dir / "daily_rank_ic.csv",
            "daily_topk_metrics_csv": source_dir / "daily_topk_metrics.csv",
            "topk_candidates_parquet": source_dir / "topk_candidates.parquet",
        }
        if not all(path.is_file() for path in artifacts.values()):
            raise FileNotFoundError(f"training evaluation artifacts missing for {model_id}")
        topk = pd.read_csv(artifacts["topk_metrics_csv"])
        daily_ic = pd.read_csv(artifacts["daily_rank_ic_csv"])
        daily_topk = pd.read_csv(artifacts["daily_topk_metrics_csv"])
        topk_candidates = pd.read_parquet(artifacts["topk_candidates_parquet"])
        return _write_full_label_result(
            model_id=model_id,
            topk=topk,
            daily_ic=daily_ic,
            daily_topk=daily_topk,
            topk_candidates=topk_candidates,
            split_metrics=_split_metrics_from_summary(summary),
            summary=summary,
            inference_performed=False,
        )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, dataset, loaded_run, loaded_summary = _load_model_and_dataset(model_id, device=device)
    output_dir = _full_label_task_dir(model_id) / "inference"
    try:
        ic, topk, daily_topk, topk_candidates, split_metrics = training._predict_split(
            model=model,
            dataset=dataset,
            device=device,
            output_dir=output_dir,
            split="development",
            batch_size=BATCH_SIZE,
            amp_enabled=bool(device.type == "cuda"),
            top_k=TOP_K_VALUES,
            write_predictions=False,
            write_path_predictions=False,
            direct_value_horizon=0,
        )
    finally:
        del model, dataset
        if device.type == "cuda":
            torch.cuda.empty_cache()
        gc.collect()
    return _write_full_label_result(
        model_id=model_id,
        topk=topk,
        daily_ic=ic,
        daily_topk=daily_topk,
        topk_candidates=topk_candidates,
        split_metrics=split_metrics,
        summary=loaded_summary,
        inference_performed=True,
    )


def _extended_prediction_path(model_id: str) -> Path:
    return STUDY_ROOT / "predictions" / f"extended_d7_{model_id}.parquet"


def _structured_ranking_score(
    output: Mapping[str, torch.Tensor],
    *,
    price_anchor: str,
    forward_days: int,
) -> np.ndarray:
    if "future_path" not in output:
        raise ValueError("Structured ranking requires a predicted OHLC path")
    predicted_path = output["future_path"].detach().float().cpu().numpy()
    predicted_summary = training._derive_path_summary_numpy(
        predicted_path,
        price_anchor=price_anchor,
    )
    columns = training.derived_path_summary_columns(int(forward_days), path_dim=4)
    value_column = training.value_column_for_path(int(forward_days), path_dim=4)
    score = predicted_summary[:, columns.index(value_column)]
    return np.asarray(score, dtype=np.float64)


def _evaluate_extended_model(model_id: str, suffix: Mapping[str, Any]) -> dict[str, Any]:
    path = _extended_prediction_path(model_id)
    result_path = STUDY_ROOT / "evaluations/extended" / model_id / "evaluation.json"
    if _job_is_complete("extended_score", (model_id,)):
        return _read_json(result_path)
    candidate_path = Path(
        str(dict(suffix["candidate_indexes"])["extended_native_180"]["path"])
    ).resolve()
    candidates = pd.read_parquet(candidate_path)
    if len(candidates) == 0 or candidates["trade_date"].nunique() != EXPECTED_EXTENDED_DATES:
        raise ValueError("extended candidate material is incomplete")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, dataset, run_dir, summary = _load_model_and_dataset(model_id, device=device)
    accessor = ExtendedFeatureAccessor(dataset, suffix)
    rows: list[pd.DataFrame] = []
    try:
        for start in range(0, len(candidates), BATCH_SIZE):
            stop = min(start + BATCH_SIZE, len(candidates))
            part = candidates.iloc[start:stop]
            dates = part["date_idx"].to_numpy(dtype=np.int64)
            symbols = part["symbol_idx"].to_numpy(dtype=np.int64)
            x = torch.from_numpy(accessor.batch_x(dates, symbols)).to(device)
            symbol_tensor = torch.from_numpy(symbols.astype(np.int64, copy=False)).to(device)
            with torch.no_grad(), torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda"):
                output = model(x, symbol_idx=symbol_tensor)
            score = _structured_ranking_score(
                output,
                price_anchor=dataset.price_anchor,
                forward_days=dataset.forward_days,
            )
            if not bool(np.isfinite(score).all()):
                raise ValueError(f"non-finite extended scores for {model_id}")
            rows.append(
                part[
                    ["candidate_id", "trade_date", "date_idx", "symbol_idx", "symbol", "entry_filled"]
                ].assign(score=score)
            )
            del x, symbol_tensor, output, score
    finally:
        accessor.close()
        del model, dataset
        if device.type == "cuda":
            torch.cuda.empty_cache()
        gc.collect()
    forecast = pd.concat(rows, ignore_index=True)
    forecast = forecast.sort_values(["date_idx", "symbol_idx"], kind="mergesort").reset_index(drop=True)
    if len(forecast) != len(candidates) or not bool(np.isfinite(forecast["score"]).all()):
        raise ValueError("extended score coverage is incomplete")
    _write_parquet(path, forecast)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    task = {
        "schema_version": 1,
        "status": "completed",
        **_task_identity("extended_121"),
        "model_id": model_id,
        "window": [SIGNAL_START, EXTENDED_SIGNAL_END],
        "candidate_count": int(len(forecast)),
        "date_count": EXPECTED_EXTENDED_DATES,
        "score_coverage": 1.0,
        "prediction_path": str(path.resolve()),
        "prediction_sha256": _file_sha256(path),
        "checkpoint_path": str((run_dir / "best_model.pt").resolve()),
        "checkpoint_sha256": _file_sha256(run_dir / "best_model.pt"),
        "normalization_sha256": _canonical_digest(_read_json(_model_paths(model_id)[1])["normalization"]),
        "completed_at": _now(),
    }
    _write_json(result_path, task)
    return task


def _safe_ratio(numerator: float, denominator: float) -> float | None:
    if not math.isfinite(numerator) or not math.isfinite(denominator) or denominator == 0.0:
        return None
    value = numerator / denominator
    return float(value) if math.isfinite(value) else None


def _mark_equity(
    *, cash: float, positions: Mapping[int, Mapping[str, Any]], date_idx: int,
    close_panel: np.ndarray,
) -> float:
    equity = float(cash)
    for position in positions.values():
        price = float(close_panel[int(date_idx), int(position["symbol_idx"])])
        if not math.isfinite(price) or price < 0.0:
            price = float(position["last_mark_price"])
        equity += float(position["shares"]) * price
    return float(equity)


def _selected_signals(forecast: pd.DataFrame, *, top_k: int) -> dict[int, list[dict[str, Any]]]:
    selected: dict[int, list[dict[str, Any]]] = {}
    for date_idx_raw, group in forecast.groupby("date_idx", sort=True):
        ordered = group.sort_values(
            ["score", "symbol_idx"], ascending=[False, True], kind="mergesort"
        ).head(int(top_k))
        if len(ordered) != int(top_k):
            raise ValueError(f"not enough candidates on date_idx={date_idx_raw}")
        selected[int(date_idx_raw)] = ordered.to_dict("records")
    return selected


def simulate_partial_d7_account(
    *,
    model_id: str,
    forecast: pd.DataFrame,
    spec: AccountSpec,
    cost_scenario: str,
    last_signal_date: str,
    audit_pack: CandidateCompleteAuditPack,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if cost_scenario not in COST_SCENARIOS:
        raise ValueError(f"unknown cost scenario: {cost_scenario}")
    date_values = np.asarray(audit_pack.date_values, dtype=object)
    date_to_idx = {str(value): idx for idx, value in enumerate(date_values)}
    first_signal_idx = date_to_idx[SIGNAL_START]
    last_signal_idx = date_to_idx[str(last_signal_date)]
    final_idx = date_to_idx[QDP_AS_OF]
    frame = forecast[
        forecast["trade_date"].astype(str).between(SIGNAL_START, str(last_signal_date))
    ].copy()
    selections = _selected_signals(frame, top_k=spec.top_k)
    expected_dates = (
        EXPECTED_EXTENDED_DATES
        if str(last_signal_date) == EXTENDED_SIGNAL_END
        else EXPECTED_STRICT_DATES
    )
    if len(selections) != expected_dates:
        raise ValueError("selected D7 signal dates are incomplete")
    if min(selections) != first_signal_idx or max(selections) != last_signal_idx:
        raise ValueError("D7 selection window drifted")
    multiplier = (
        1.0
        if cost_scenario == "base"
        else float(audit_pack.contract.stress_slippage_multiplier)
    )
    cash = float(STARTING_CASH_CNY)
    positions: dict[int, dict[str, Any]] = {}
    pending: list[dict[str, Any]] = []
    next_position_id = 0
    daily_rows: list[dict[str, Any]] = []
    trade_rows: list[dict[str, Any]] = []
    counters: dict[str, int | float] = {
        "buy_count": 0,
        "sell_count": 0,
        "failed_entry_count": 0,
        "skipped_duplicate_signal_count": 0,
        "skipped_duplicate_pending_count": 0,
        "skipped_no_slot_signal_count": 0,
        "deferred_exit_count": 0,
        "terminal_recovery_count": 0,
        "fees_and_slippage_cny": 0.0,
        "turnover_notional_cny": 0.0,
    }

    for date_idx in range(first_signal_idx, final_idx + 1):
        if pending:
            equity_open = _mark_equity(
                cash=cash,
                positions=positions,
                date_idx=date_idx,
                close_panel=audit_pack.entry_open_raw,
            )
            for order in pending:
                duplicate = any(
                    int(position["symbol_idx"]) == int(order["symbol_idx"])
                    for position in positions.values()
                )
                if duplicate:
                    counters["skipped_duplicate_pending_count"] += 1
                    continue
                if len(positions) >= int(spec.slots):
                    counters["skipped_no_slot_signal_count"] += 1
                    continue
                entry_price = float(audit_pack.entry_open_raw[date_idx, int(order["symbol_idx"])])
                if (
                    not bool(order["entry_filled"])
                    or not math.isfinite(entry_price)
                    or entry_price <= 0.0
                ):
                    counters["failed_entry_count"] += 1
                    continue
                allocation = min(cash, max(equity_open, 0.0) / float(spec.slots))
                shares, buy_cash, buy_cost, buy_notional = finite._buy_order(
                    available_cash=cash,
                    allocated_cash=allocation,
                    entry_price=entry_price,
                    contract=audit_pack.contract,
                    slippage_multiplier=multiplier,
                )
                if shares <= 0:
                    counters["failed_entry_count"] += 1
                    continue
                requested = int(order["signal_date_idx"]) + 7
                cash -= float(buy_cash)
                counters["fees_and_slippage_cny"] += float(buy_cost)
                counters["turnover_notional_cny"] += float(buy_notional)
                counters["buy_count"] += 1
                positions[next_position_id] = {
                    "signal_date_idx": int(order["signal_date_idx"]),
                    "symbol_idx": int(order["symbol_idx"]),
                    "symbol": str(order["symbol"]),
                    "shares": int(shares),
                    "entry_price_raw": entry_price,
                    "buy_cash": float(buy_cash),
                    "entry_date_idx": int(date_idx),
                    "requested_exit_date_idx": requested,
                    "terminal_date_idx": requested + 20,
                    "last_mark_price": entry_price,
                }
                next_position_id += 1
            pending = []

        for position in positions.values():
            mark = float(audit_pack.exit_close_raw[date_idx, int(position["symbol_idx"])])
            if math.isfinite(mark) and mark >= 0.0:
                position["last_mark_price"] = mark

        for position_id, position in list(positions.items()):
            if date_idx < int(position["requested_exit_date_idx"]):
                continue
            symbol_idx = int(position["symbol_idx"])
            close_price = float(audit_pack.exit_close_raw[date_idx, symbol_idx])
            sellable = bool(audit_pack.exit_sellable[date_idx, symbol_idx]) and math.isfinite(close_price) and close_price >= 0.0
            terminal = date_idx >= int(position["terminal_date_idx"]) and not sellable
            if not sellable and not terminal:
                continue
            exit_price = close_price
            exit_reason = "fixed_d7"
            if terminal:
                exit_price = float(position["entry_price_raw"]) * float(
                    audit_pack.terminal_recovery_fraction
                )
                exit_reason = "terminal_recovery_after_d7_plus_20"
                counters["terminal_recovery_count"] += 1
            elif date_idx > int(position["requested_exit_date_idx"]):
                exit_reason = "deferred_sellable_close"
                counters["deferred_exit_count"] += 1
            proceeds, sell_cost, sell_notional = finite._sell_order(
                shares=int(position["shares"]),
                exit_price=float(exit_price),
                exit_date_idx=date_idx,
                date_values=date_values,
                contract=audit_pack.contract,
                slippage_multiplier=multiplier,
            )
            cash += float(proceeds)
            counters["fees_and_slippage_cny"] += float(sell_cost)
            counters["turnover_notional_cny"] += float(sell_notional)
            counters["sell_count"] += 1
            trade_rows.append(
                {
                    "model_id": model_id,
                    "strategy_id": spec.strategy_id,
                    "cost_scenario": cost_scenario,
                    "window_end": str(last_signal_date),
                    "symbol": str(position["symbol"]),
                    "signal_date": str(date_values[int(position["signal_date_idx"])]),
                    "entry_date": str(date_values[int(position["entry_date_idx"])]),
                    "exit_date": str(date_values[date_idx]),
                    "signal_date_idx": int(position["signal_date_idx"]),
                    "entry_date_idx": int(position["entry_date_idx"]),
                    "exit_date_idx": int(date_idx),
                    "entry_price_raw": float(position["entry_price_raw"]),
                    "exit_price_raw": float(exit_price),
                    "shares": int(position["shares"]),
                    "buy_cash_cny": float(position["buy_cash"]),
                    "sell_proceeds_cny": float(proceeds),
                    "net_pnl_cny": float(proceeds - float(position["buy_cash"])),
                    "net_return_on_buy_cash": float(proceeds / float(position["buy_cash"]) - 1.0),
                    "occupied_sessions": int(date_idx - int(position["entry_date_idx"]) + 1),
                    "requested_exit_date": str(date_values[int(position["requested_exit_date_idx"])]),
                    "exit_reason": exit_reason,
                }
            )
            positions.pop(position_id)

        equity = _mark_equity(
            cash=cash,
            positions=positions,
            date_idx=date_idx,
            close_panel=audit_pack.exit_close_raw,
        )
        utilization = 0.0 if equity <= 0.0 else float(1.0 - cash / equity)
        daily_rows.append(
            {
                "model_id": model_id,
                "strategy_id": spec.strategy_id,
                "cost_scenario": cost_scenario,
                "window_end": str(last_signal_date),
                "trade_date": str(date_values[date_idx]),
                "date_idx": int(date_idx),
                "equity": equity,
                "cash": float(cash),
                "position_count": int(len(positions)),
                "capital_utilization": utilization,
                "inside_signal_period": bool(date_idx <= last_signal_idx),
            }
        )

        if date_idx <= last_signal_idx:
            for row in selections.get(date_idx, []):
                symbol_idx = int(row["symbol_idx"])
                duplicate = any(
                    int(position["symbol_idx"]) == symbol_idx
                    for position in positions.values()
                ) or any(int(order["symbol_idx"]) == symbol_idx for order in pending)
                if duplicate:
                    counters["skipped_duplicate_signal_count"] += 1
                    continue
                if len(positions) + len(pending) >= int(spec.slots):
                    counters["skipped_no_slot_signal_count"] += 1
                    continue
                pending.append(
                    {
                        "signal_date_idx": int(date_idx),
                        "symbol_idx": symbol_idx,
                        "symbol": str(row["symbol"]),
                        "entry_filled": bool(row["entry_filled"]),
                    }
                )

    if pending:
        raise AssertionError("pending D7 orders remain at the market boundary")
    daily = pd.DataFrame(daily_rows)
    trades = pd.DataFrame(trade_rows)
    if trades.empty:
        trades = pd.DataFrame(
            columns=[
                "model_id",
                "strategy_id",
                "cost_scenario",
                "window_end",
                "symbol",
                "signal_date",
                "entry_date",
                "exit_date",
                "signal_date_idx",
                "entry_date_idx",
                "exit_date_idx",
                "entry_price_raw",
                "exit_price_raw",
                "shares",
                "buy_cash_cny",
                "sell_proceeds_cny",
                "net_pnl_cny",
                "net_return_on_buy_cash",
                "occupied_sessions",
                "requested_exit_date",
                "exit_reason",
            ]
        )
    terminal_rows: list[dict[str, Any]] = []
    for position in positions.values():
        mark = float(position["last_mark_price"])
        terminal_rows.append(
            {
                "model_id": model_id,
                "strategy_id": spec.strategy_id,
                "cost_scenario": cost_scenario,
                "window_end": str(last_signal_date),
                "symbol": str(position["symbol"]),
                "signal_date": str(date_values[int(position["signal_date_idx"])]),
                "entry_date": str(date_values[int(position["entry_date_idx"])]),
                "requested_exit_date": str(date_values[int(position["requested_exit_date_idx"])]),
                "entry_price_raw": float(position["entry_price_raw"]),
                "last_mark_price": mark,
                "shares": int(position["shares"]),
                "buy_cash_cny": float(position["buy_cash"]),
                "mark_value_cny": float(mark * int(position["shares"])),
                "unrealized_pnl_cny": float(mark * int(position["shares"]) - float(position["buy_cash"])),
            }
        )
    terminal = pd.DataFrame(terminal_rows)
    if terminal.empty:
        terminal = pd.DataFrame(
            columns=[
                "model_id",
                "strategy_id",
                "cost_scenario",
                "window_end",
                "symbol",
                "signal_date",
                "entry_date",
                "requested_exit_date",
                "entry_price_raw",
                "last_mark_price",
                "shares",
                "buy_cash_cny",
                "mark_value_cny",
                "unrealized_pnl_cny",
            ]
        )
    equity_values = daily["equity"].to_numpy(dtype=np.float64)
    running_peak = np.maximum.accumulate(equity_values)
    drawdown = equity_values / np.maximum(running_peak, 1.0e-12) - 1.0
    daily_returns = pd.Series(equity_values).pct_change().dropna().to_numpy(dtype=np.float64)
    sharpe = (
        float(np.mean(daily_returns) / np.std(daily_returns, ddof=1) * math.sqrt(252.0))
        if len(daily_returns) > 1 and float(np.std(daily_returns, ddof=1)) > 0.0
        else None
    )
    trade_returns = (
        trades["net_return_on_buy_cash"].to_numpy(dtype=np.float64)
        if not trades.empty
        else np.asarray([], dtype=np.float64)
    )
    wins = trade_returns[trade_returns > 0.0]
    losses = trade_returns[trade_returns < 0.0]
    sessions = int(len(daily))
    realized_cash = float(cash)
    mark_to_market = float(equity_values[-1])
    metric: dict[str, Any] = {
        "model_id": model_id,
        "strategy_id": spec.strategy_id,
        "top_k": int(spec.top_k),
        "slot_count": int(spec.slots),
        "fixed_exit_day": 7,
        "cost_scenario": cost_scenario,
        "signal_start": SIGNAL_START,
        "signal_end": str(last_signal_date),
        "signal_date_count": int(len(selections)),
        "starting_cash_cny": STARTING_CASH_CNY,
        "realized_cash_equity": realized_cash,
        "mark_to_market_equity": mark_to_market,
        "mark_to_market_total_return": float(mark_to_market / STARTING_CASH_CNY - 1.0),
        "annualized_log_growth": float(math.log(max(mark_to_market, 1.0e-12) / STARTING_CASH_CNY) * 252.0 / max(sessions - 1, 1)),
        "maximum_drawdown": float(np.min(drawdown)),
        "sharpe": sharpe,
        "winning_trade_rate": float(np.mean(trade_returns > 0.0)) if len(trade_returns) else None,
        "payoff_ratio": _safe_ratio(float(np.mean(wins)) if len(wins) else math.nan, abs(float(np.mean(losses))) if len(losses) else math.nan),
        "profit_factor": _safe_ratio(float(np.sum(wins)), abs(float(np.sum(losses)))),
        "expectancy_per_trade": float(np.mean(trade_returns)) if len(trade_returns) else None,
        "trade_count": int(len(trades)),
        "mean_occupied_sessions": float(trades["occupied_sessions"].mean()) if not trades.empty else None,
        "mean_capital_utilization": float(daily.loc[daily["inside_signal_period"], "capital_utilization"].mean()),
        "terminal_position_count": int(len(terminal)),
        "all_positions_resolved": bool(len(terminal) == 0),
        **counters,
    }
    return metric, daily, trades, terminal


def _account_job_dir(
    model_id: str, spec: AccountSpec, cost_scenario: str, window_name: str
) -> Path:
    return STUDY_ROOT / "accounts" / model_id / spec.strategy_id / cost_scenario / window_name


def _evaluate_account_job(
    *, model_id: str, spec: AccountSpec, cost_scenario: str, window_name: str
) -> dict[str, Any]:
    if window_name not in {"extended_121", "strict_101"}:
        raise ValueError("unknown D7 account window")
    root = _account_job_dir(model_id, spec, cost_scenario, window_name)
    result_path = root / "evaluation.json"
    if _job_is_complete("account", (model_id, spec, cost_scenario, window_name)):
        return _read_json(result_path)
    forecast = pd.read_parquet(_extended_prediction_path(model_id))
    last_signal = EXTENDED_SIGNAL_END if window_name == "extended_121" else STRICT_SIGNAL_END
    audit = CandidateCompleteAuditPack(FROZEN_2026_OVERLAY)
    metric, daily, trades, terminal = simulate_partial_d7_account(
        model_id=model_id,
        forecast=forecast,
        spec=spec,
        cost_scenario=cost_scenario,
        last_signal_date=last_signal,
        audit_pack=audit,
    )
    root.mkdir(parents=True, exist_ok=True)
    paths = {
        "daily": root / "daily.parquet",
        "trades": root / "trades.parquet",
        "terminal": root / "terminal.parquet",
    }
    _write_parquet(paths["daily"], daily)
    _write_parquet(paths["trades"], trades)
    _write_parquet(paths["terminal"], terminal)
    result = {
        "schema_version": 1,
        "status": "completed",
        **_task_identity(window_name),
        "job_id": f"{model_id}:{spec.strategy_id}:{cost_scenario}:{window_name}",
        "window_name": window_name,
        "metrics": metric,
        "outputs": {key: str(path.resolve()) for key, path in paths.items()},
        "output_sha256": {key: _file_sha256(path) for key, path in paths.items()},
        "completed_at": _now(),
    }
    _write_json(result_path, result)
    return result


def _evaluation_jobs() -> list[tuple[str, tuple[Any, ...]]]:
    jobs: list[tuple[str, tuple[Any, ...]]] = []
    jobs.extend(("full_label", (model_id,)) for model_id in MODEL_ORDER)
    jobs.extend(("extended_score", (model_id,)) for model_id in MODEL_ORDER)
    for model_id in MODEL_ORDER:
        for spec in ACCOUNT_SPECS:
            for cost in COST_SCENARIOS:
                for window in ("strict_101", "extended_121"):
                    jobs.append(("account", (model_id, spec, cost, window)))
    return jobs


def _job_is_complete(kind: str, args: tuple[Any, ...]) -> bool:
    if kind == "full_label":
        path = _full_label_task_dir(str(args[0])) / "evaluation.json"
        scope = "full_label"
    elif kind == "extended_score":
        path = STUDY_ROOT / "evaluations/extended" / str(args[0]) / "evaluation.json"
        scope = "extended_121"
    else:
        model_id, spec, cost, window = args
        path = _account_job_dir(str(model_id), spec, str(cost), str(window)) / "evaluation.json"
        scope = str(window)
    if not path.is_file():
        return False
    try:
        result = _read_json(path)
        if str(result.get("status", "")) != "completed":
            return False
        expected_identity = _task_identity(scope)
        if any(str(result.get(key, "")) != value for key, value in expected_identity.items()):
            return False
        if kind == "extended_score":
            prediction = Path(str(result.get("prediction_path", ""))).resolve()
            return prediction.is_file() and _file_sha256(prediction) == str(
                result.get("prediction_sha256", "")
            )
        outputs = dict(result.get("outputs", {}) or {})
        hashes = dict(result.get("output_sha256", {}) or {})
        if not outputs or set(outputs) != set(hashes):
            return False
        return all(
            Path(str(output)).is_file()
            and _file_sha256(Path(str(output))) == str(hashes[name])
            for name, output in outputs.items()
        )
    except (KeyError, OSError, json.JSONDecodeError, ValueError):
        return False


def evaluate_structured_180x35_2026(
    *, study_root: Path = STUDY_ROOT, max_jobs: int = 1
) -> dict[str, Any]:
    if int(max_jobs) < 0:
        raise ValueError("max_jobs must be non-negative")
    _assert_no_research_process()
    verify_structured_180x35_2026(study_root=study_root, require_complete=False)
    complete, partial = _completed_runs()
    if len(complete) != 1 or partial:
        raise ValueError("train the new 2026 checkpoint before evaluation")
    suffix = _validate_suffix_overlay(OVERLAY_ROOT / "manifest.json")
    study_path = STUDY_ROOT / "study.json"
    launched = 0
    limit = int(max_jobs)
    try:
        for kind, args in _evaluation_jobs():
            if _job_is_complete(kind, args):
                continue
            if limit > 0 and launched >= limit:
                break
            job_id = f"{kind}:" + ":".join(
                value.strategy_id if isinstance(value, AccountSpec) else str(value)
                for value in args
            )
            phase = {
                "full_label": "full_label_inference",
                "extended_score": "extended_d7_inference",
                "account": "account_backtest",
            }[kind]
            _update_runtime(study_path, status=phase, current_task=job_id, error=None)
            _write_json(
                STUDY_ROOT / "progress.json",
                {
                    "status": phase,
                    "current_task": job_id,
                    "completed_jobs": int(sum(_job_is_complete(k, a) for k, a in _evaluation_jobs())),
                    "total_jobs": len(_evaluation_jobs()),
                    "updated_at": _now(),
                },
            )
            if kind == "full_label":
                _evaluate_full_label_model(str(args[0]))
            elif kind == "extended_score":
                _evaluate_extended_model(str(args[0]), suffix)
            else:
                model_id, spec, cost, window = args
                _evaluate_account_job(
                    model_id=str(model_id),
                    spec=spec,
                    cost_scenario=str(cost),
                    window_name=str(window),
                )
            launched += 1
        remaining = [(kind, args) for kind, args in _evaluation_jobs() if not _job_is_complete(kind, args)]
        if remaining:
            _update_runtime(study_path, status="evaluation_pending", current_task=None, error=None)
            return status_structured_180x35_2026(study_root=study_root)
        _update_runtime(study_path, status="summarizing", current_task=None, error=None)
        return summarize_structured_180x35_2026(study_root=study_root)
    except Exception as exc:
        _update_runtime(study_path, status="evaluation_failed", current_task=None, error=str(exc))
        raise


def _scalar_metrics_row(result: Mapping[str, Any]) -> dict[str, Any]:
    row: dict[str, Any] = {
        "model_id": str(result["model_id"]),
        "checkpoint_vintage": int(result["checkpoint_vintage"]),
        "checkpoint_sha256": str(result["checkpoint_sha256"]),
        "normalization_sha256": str(result["normalization_sha256"]),
        "model_bundle_sha256": str(result["model_bundle_sha256"]),
    }
    for key, value in dict(result["metrics"]).items():
        if isinstance(value, (str, int, float, bool, np.integer, np.floating)) or value is None:
            row[str(key)] = value
    return row


def _summary_markdown(summary: Mapping[str, Any]) -> str:
    lines = [
        "# Structured 180x35 2026 Fold and Fixed-D7 Accounts",
        "",
        f"Full-label window: `{SIGNAL_START}..{FULL_LABEL_END}` ({EXPECTED_FULL_LABEL_DATES} signal dates).",
        f"Extended D7 window: `{SIGNAL_START}..{EXTENDED_SIGNAL_END}` ({EXPECTED_EXTENDED_DATES} signal dates).",
        f"Strict D7+20 window: `{SIGNAL_START}..{STRICT_SIGNAL_END}` ({EXPECTED_STRICT_DATES} signal dates).",
        "",
        "The new checkpoint is an experimental research fold. No deployment state was changed.",
        "",
        "## Selected fixed-D7 account results",
        "",
        "| Strategy | Window | Cost | Ending equity | Total return | Max drawdown | Trades |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    rows = list(summary.get("account_results", []) or [])
    if not rows:
        rows = list(summary.get("selected_account_results", []) or [])
    for row in rows:
        lines.append(
            f"| {row['strategy_id']} | {row['window_name']} | {row['cost_scenario']} | "
            f"{row['mark_to_market_equity']:.2f} | {row['mark_to_market_total_return']:.4%} | "
            f"{row['maximum_drawdown']:.4%} | {row['trade_count']} |"
        )
    lines.extend(
        [
            "",
            "The strict and extended windows are reported separately. This study evaluates the preselected model and does not run a promotion or model-selection gate.",
            "",
        ]
    )
    return "\n".join(lines)


def summarize_structured_180x35_2026(
    *, study_root: Path = STUDY_ROOT
) -> dict[str, Any]:
    missing = [(kind, args) for kind, args in _evaluation_jobs() if not _job_is_complete(kind, args)]
    if missing:
        raise ValueError(f"cannot summarize incomplete evaluation jobs: {len(missing)}")
    full_results = {
        model_id: _read_json(_full_label_task_dir(model_id) / "evaluation.json")
        for model_id in MODEL_ORDER
    }
    full_frame = pd.DataFrame([_scalar_metrics_row(full_results[model_id]) for model_id in MODEL_ORDER])
    _write_csv(STUDY_ROOT / "full_label_metrics.csv", full_frame)

    account_results: list[dict[str, Any]] = []
    daily_frames: list[pd.DataFrame] = []
    trade_frames: list[pd.DataFrame] = []
    terminal_frames: list[pd.DataFrame] = []
    for model_id in MODEL_ORDER:
        for spec in ACCOUNT_SPECS:
            for cost in COST_SCENARIOS:
                for window in ("strict_101", "extended_121"):
                    result = _read_json(
                        _account_job_dir(model_id, spec, cost, window) / "evaluation.json"
                    )
                    metric = dict(result["metrics"])
                    metric["window_name"] = window
                    account_results.append(metric)
                    outputs = dict(result["outputs"])
                    daily_frames.append(pd.read_parquet(outputs["daily"]))
                    trades = pd.read_parquet(outputs["trades"])
                    terminal = pd.read_parquet(outputs["terminal"])
                    if not trades.empty:
                        trade_frames.append(trades)
                    if not terminal.empty:
                        terminal_frames.append(terminal)
    account = pd.DataFrame(account_results)
    strict_lookup = account[account["window_name"].astype(str).eq("strict_101")].set_index(
        ["model_id", "strategy_id", "cost_scenario"]
    )["mark_to_market_equity"]
    account["strict_tail_equity"] = [
        float(strict_lookup.loc[(row.model_id, row.strategy_id, row.cost_scenario)])
        for row in account.itertuples(index=False)
    ]
    _write_csv(STUDY_ROOT / "d7_account_metrics.csv", account)
    daily = pd.concat(daily_frames, ignore_index=True)
    trades = pd.concat(trade_frames, ignore_index=True) if trade_frames else pd.DataFrame()
    terminal = pd.concat(terminal_frames, ignore_index=True) if terminal_frames else pd.DataFrame()
    _write_csv(STUDY_ROOT / "d7_account_daily.csv", daily)
    _write_csv(STUDY_ROOT / "d7_trade_ledger.csv", trades)
    _write_csv(STUDY_ROOT / "terminal_positions.csv", terminal)
    suffix = _read_json(OVERLAY_ROOT / "manifest.json")
    candidate_audit = dict(suffix["candidate_audit"])
    fold_audit = dict(_read_json(STUDY_ROOT / "study.json")["material"]["fold_audit"])
    qualification = pd.DataFrame(
        [
            {
                "scope": "full_label",
                "common_candidate_count": fold_audit["common_full_label_candidate_count"],
                "native_candidate_count": fold_audit["native_full_label_candidate_count"],
                "excluded_candidate_count": fold_audit["excluded_full_label_candidate_count"],
                "native_coverage": fold_audit["native_full_label_candidate_count"]
                / fold_audit["common_full_label_candidate_count"],
            },
            {
                "scope": "extended_d7",
                "common_candidate_count": candidate_audit["common_100_extended_candidate_count"],
                "native_candidate_count": candidate_audit["native_180_extended_candidate_count"],
                "excluded_candidate_count": candidate_audit["common_100_extended_candidate_count"]
                - candidate_audit["native_180_extended_candidate_count"],
                "native_coverage": candidate_audit["native_180_extended_candidate_count"]
                / candidate_audit["common_100_extended_candidate_count"],
            },
            {
                "scope": "strict_tail",
                "common_candidate_count": candidate_audit["common_100_strict_candidate_count"],
                "native_candidate_count": candidate_audit["native_180_strict_candidate_count"],
                "excluded_candidate_count": candidate_audit["common_100_strict_candidate_count"]
                - candidate_audit["native_180_strict_candidate_count"],
                "native_coverage": candidate_audit["native_180_strict_candidate_count"]
                / candidate_audit["common_100_strict_candidate_count"],
            },
        ]
    )
    _write_csv(STUDY_ROOT / "candidate_qualification_audit.csv", qualification)
    selected_rows = _json_safe(
        account[account["cost_scenario"].astype(str).eq("double_slippage")].to_dict("records")
    )
    shared = dict(_read_json(STUDY_ROOT / "study.json")["material"]["integrity"])
    summary = {
        "schema_version": 1,
        "artifact_type": "seq100_structured_180x35_2026_evaluation",
        "status": "completed",
        "created_at": _now(),
        "study_id": STUDY_ID,
        "full_label_window": {
            "start": SIGNAL_START,
            "end": FULL_LABEL_END,
            "signal_dates": EXPECTED_FULL_LABEL_DATES,
            "candidate_rows": EXPECTED_NATIVE_FULL_LABEL_CANDIDATES,
        },
        "extended_d7_window": {
            "start": SIGNAL_START,
            "end": EXTENDED_SIGNAL_END,
            "signal_dates": EXPECTED_EXTENDED_DATES,
        },
        "strict_tail_window": {
            "start": SIGNAL_START,
            "end": STRICT_SIGNAL_END,
            "signal_dates": EXPECTED_STRICT_DATES,
        },
        "full_label_metrics": dict(full_results[MODEL_2026_180]["metrics"]),
        "account_results": _json_safe(account.to_dict("records")),
        "selected_account_results": selected_rows,
        "candidate_qualification_audit": qualification.to_dict("records"),
        "model_selection_performed": False,
        "checkpoint_comparison_performed": False,
        "promotion_gate_applied": False,
        "deployment_upgrade": False,
        "integrity": {
            **shared,
            "model_bundle_sha256": str(full_results[MODEL_2026_180]["model_bundle_sha256"]),
            "result_artifact_sha256": str(full_results[MODEL_2026_180]["result_artifact_sha256"]),
        },
        "outputs": {
            "full_label_metrics": str((STUDY_ROOT / "full_label_metrics.csv").resolve()),
            "d7_account_metrics": str((STUDY_ROOT / "d7_account_metrics.csv").resolve()),
            "d7_account_daily": str((STUDY_ROOT / "d7_account_daily.csv").resolve()),
            "d7_trade_ledger": str((STUDY_ROOT / "d7_trade_ledger.csv").resolve()),
            "terminal_positions": str((STUDY_ROOT / "terminal_positions.csv").resolve()),
            "candidate_qualification_audit": str((STUDY_ROOT / "candidate_qualification_audit.csv").resolve()),
        },
        "protected_boundaries": {
            "qdp_changed": False,
            "provider_called": False,
            "base_pack_changed": False,
            "old_checkpoint_changed": False,
            "stage3_changed": False,
            "live_state_changed": False,
            "orders_placed": False,
        },
    }
    _write_json(STUDY_ROOT / "comparison_summary.json", summary)
    temporary_md = STUDY_ROOT / "comparison_summary.md.tmp"
    temporary_md.write_text(_summary_markdown(summary), encoding="utf-8")
    os.replace(temporary_md, STUDY_ROOT / "comparison_summary.md")
    _update_runtime(STUDY_ROOT / "study.json", status="completed", current_task=None, error=None)
    _write_json(
        STUDY_ROOT / "progress.json",
        {
            "status": "completed",
            "completed_jobs": len(_evaluation_jobs()),
            "total_jobs": len(_evaluation_jobs()),
            "summary": str((STUDY_ROOT / "comparison_summary.json").resolve()),
            "updated_at": _now(),
        },
    )
    verify_structured_180x35_2026(study_root=study_root, require_complete=True)
    return summary


def _architecture_signature(run_dir: Path) -> dict[str, Any]:
    summary = _read_json(run_dir / "sequence_path_training_summary.json")
    checkpoint = torch.load(run_dir / "best_model.pt", map_location="cpu", weights_only=False)
    config = dict(summary.get("resolved_training_config", {}) or {})
    return {
        "model_type": str(config.get("model_type", "")),
        "input_dim": int(checkpoint["input_dim"]),
        "hidden_dim": int(config.get("hidden_dim", 128)),
        "layers": int(config.get("layers", 2)),
        "dropout": float(config.get("dropout", 0.1)),
        "forward_days": int(config.get("forward_days", FORWARD_DAYS)),
        "projection_shape": list(checkpoint["model_state_dict"]["proj.weight"].shape),
        "state_shapes": {
            key: list(value.shape)
            for key, value in checkpoint["model_state_dict"].items()
        },
    }


def status_structured_180x35_2026(
    *, study_root: Path = STUDY_ROOT
) -> dict[str, Any]:
    root = study_root.resolve()
    study_path = root / "study.json"
    if not study_path.is_file():
        return {
            "study": str(study_path),
            "status": "not_prepared",
            "training": {"complete": False, "partial_runs": []},
            "evaluation": {"completed_jobs": 0, "total_jobs": len(_evaluation_jobs())},
        }
    study = _read_json(study_path)
    complete, partial = _completed_runs()
    current_progress = None
    if partial:
        progress_path = partial[-1] / "progress.json"
        if progress_path.is_file():
            try:
                current_progress = _read_json(progress_path)
            except (OSError, json.JSONDecodeError):
                current_progress = None
    completed_jobs = int(sum(_job_is_complete(kind, args) for kind, args in _evaluation_jobs()))
    return {
        "study": str(study_path.resolve()),
        "runtime": dict(study.get("runtime", {}) or {}),
        "training": {
            "complete": len(complete) == 1 and not partial,
            "run_dir": str(complete[0].resolve()) if len(complete) == 1 else None,
            "partial_runs": [str(path.resolve()) for path in partial],
            "progress": current_progress,
        },
        "evaluation": {
            "completed_jobs": completed_jobs,
            "total_jobs": len(_evaluation_jobs()),
            "remaining_jobs": len(_evaluation_jobs()) - completed_jobs,
        },
        "monitor": (
            _read_json(root / "monitor.json") if (root / "monitor.json").is_file() else None
        ),
        "summary": str((root / "comparison_summary.json").resolve())
        if (root / "comparison_summary.json").is_file()
        else None,
    }


def verify_structured_180x35_2026(
    *, study_root: Path = STUDY_ROOT, require_complete: bool = False
) -> dict[str, Any]:
    root = study_root.resolve()
    if root != STUDY_ROOT.resolve():
        raise ValueError("Structured 180x35 2026 verification requires the canonical root")
    study_path = root / "study.json"
    study = _read_json(study_path)
    if (
        str(study.get("study_id", "")) != STUDY_ID
        or dict(study.get("contract", {}) or {}) != _semantic_contract()
        or str(study.get("contract_sha256", "")) != _canonical_digest(_semantic_contract())
    ):
        raise ValueError("Structured 180x35 2026 study contract drifted")
    material = dict(study.get("material", {}) or {})
    if not material:
        raise ValueError("Structured 180x35 2026 material is not prepared")
    current_source = _source_identity()
    if current_source != dict(material.get("source_identity", {}) or {}):
        raise ValueError("a protected source changed after study preparation")
    suffix_path = Path(str(material["suffix_overlay"])).resolve()
    suffix = _validate_suffix_overlay(suffix_path)
    if (
        suffix_path != (OVERLAY_ROOT / "manifest.json").resolve()
        or _file_sha256(suffix_path) != str(material["suffix_overlay_sha256"])
        or int(suffix["validated_size_bytes"]) > OVERLAY_SIZE_LIMIT_BYTES
    ):
        raise ValueError("suffix overlay binding drifted")
    training_view = Path(str(material["training_view"])).resolve()
    for path, hash_key in ((training_view, "training_view_sha256"),):
        if not path.is_file() or _file_sha256(path) != str(material[hash_key]):
            raise ValueError(f"prepared view binding drifted: {path}")
    view = _read_json(training_view)
    audit = dict(material["fold_audit"])
    if (
        int(audit["train_row_count"]) != EXPECTED_NATIVE_TRAIN_ROWS
        or str(audit["first_train_signal"]) != "2010-09-29"
        or str(audit["last_train_signal"]) != SAFE_TRAIN_SIGNAL_END
        or str(audit["normalization_cutoff_exclusive"]) != SIGNAL_START
        or int(view["lookback_days"]) != LOOKBACK_DAYS
        or list(dict(view["data_semantics"])["model_input_mask_features"])
        != ["turnover_valid"]
    ):
        raise ValueError("prepared fold boundary or input contract drifted")
    sample_path = Path(str(view["sample_index_path"])).resolve()
    if _file_sha256(sample_path) != str(audit["sample_index_sha256"]):
        raise ValueError("180-day sample index drifted")
    candidate_path = Path(str(view["candidate_index_path"])).resolve()
    if len(pd.read_parquet(candidate_path, columns=["candidate_id"])) != EXPECTED_NATIVE_FULL_LABEL_CANDIDATES:
        raise ValueError("full-label candidate count drifted")
    shared = _shared_integrity(suffix=suffix, training_view=training_view)
    if shared != dict(material["integrity"]):
        raise ValueError("shared integrity identity drifted")
    protected_flags = (
        "qdp_changed",
        "provider_called",
        "base_pack_changed",
        "old_checkpoint_changed",
        "stage3_changed",
        "live_state_changed",
    )
    if any(bool(material.get(key, True)) for key in protected_flags):
        raise ValueError("a protected-boundary flag is not false")
    complete, partial = _completed_runs()
    training_ok = len(complete) == 1 and not partial
    if training_ok:
        _validate_training_run(complete[0])
        if _architecture_signature(complete[0]) != _architecture_signature(_old_2025_run()):
            raise ValueError("new and 2025 180x35 checkpoint architectures differ")
    if require_complete and not training_ok:
        raise ValueError("new 2026 training task is incomplete")
    completed_jobs = int(sum(_job_is_complete(kind, args) for kind, args in _evaluation_jobs()))
    if require_complete:
        if completed_jobs != len(_evaluation_jobs()):
            raise ValueError("evaluation jobs are incomplete")
        for model_id in MODEL_ORDER:
            prediction = pd.read_parquet(_extended_prediction_path(model_id), columns=["score", "trade_date"])
            if (
                prediction["trade_date"].nunique() != EXPECTED_EXTENDED_DATES
                or not bool(np.isfinite(prediction["score"]).all())
            ):
                raise ValueError(f"extended score coverage drifted for {model_id}")
        account = pd.read_csv(root / "d7_account_metrics.csv")
        strict = account[account["window_name"].astype(str).eq("strict_101")]
        if not bool(strict["all_positions_resolved"].astype(bool).all()):
            raise ValueError("strict D7+20 account has unresolved positions")
        expected_outputs = (
            "full_label_metrics.csv",
            "d7_account_metrics.csv",
            "d7_account_daily.csv",
            "d7_trade_ledger.csv",
            "terminal_positions.csv",
            "candidate_qualification_audit.csv",
            "comparison_summary.json",
            "comparison_summary.md",
        )
        for name in expected_outputs:
            if not (root / name).is_file():
                raise FileNotFoundError(root / name)
    return {
        "status": "ok",
        "study": str(study_path.resolve()),
        "training_complete": training_ok,
        "completed_evaluation_jobs": completed_jobs,
        "total_evaluation_jobs": len(_evaluation_jobs()),
        "suffix_size_bytes": int(suffix["validated_size_bytes"]),
        "provider_called": False,
        "qdp_changed": False,
        "base_pack_changed": False,
        "old_checkpoint_changed": False,
        "stage3_changed": False,
        "live_state_changed": False,
    }
