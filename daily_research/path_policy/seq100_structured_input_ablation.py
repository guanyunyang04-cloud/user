"""Structured Seq100 input-window and feature-channel ablation.

The workflow is deliberately separate from the current Structured study.  It
reuses the immutable daily pack, labels, execution arrays, and candidate
indexes while adding only lightweight fold indexes and two feature overlays.
"""

from __future__ import annotations

import gc
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from pathlib import Path
from statistics import median
from typing import Any, Iterable, Mapping, Sequence

import duckdb
import numpy as np
import pandas as pd
import psutil
import torch

from daily_research.path_policy import qdp_v2_sequence_path_training as training
from daily_research.path_policy import seq100_development as development
from daily_research.path_policy import seq100_structured_experiment as structured
from daily_research.path_policy import seq100_walkforward as walkforward
from daily_research.path_policy.seq100_candidate_execution import (
    evaluate_candidate_execution,
)
from daily_research.path_policy.seq100_mainline import (
    TodayClosePathOnlyProfile,
    build_todayclose_path_only_train_argv,
)


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
PYTHON = Path("C:/Users/ASUS/miniconda3/envs/yolos/python.exe")
STUDY_ID = "seq100_structured_input_ablation_rolling_2023_2025_v1"
EXPERIMENT_ID = "structured-input-ablation-v1"
STUDY_ROOT = WORKSPACE_ROOT / (
    "daily_research/output/path_policy/studies/"
    "seq100_structured_input_ablation_rolling_2023_2025_v1"
)
OVERLAY_ROOT = WORKSPACE_ROOT / (
    "daily_research/data/research_store/seq100_structured_input_ablation_v1"
)
BASE_PACK_MANIFEST = WORKSPACE_ROOT / (
    "daily_research/data/research_store/seq100_current/pack/manifest.json"
)
BASE_STUDY_ROOT = WORKSPACE_ROOT / (
    "daily_research/output/path_policy/studies/"
    "seq100_legal_structured_path_rolling_2023_2025_v1"
)
BASE_TURNOVER_MANIFEST = WORKSPACE_ROOT / (
    "daily_research/data/research_store/seq100_current/supplements/"
    "relative_turnover_v1/manifest.json"
)
QDP_ROOT = WORKSPACE_ROOT / "quant_data_platform/data/qdp_v2"
ACTIVE_QDP_PATH = QDP_ROOT / "active/active.json"

DEVELOPMENT_YEARS = (2023, 2024, 2025)
SEED = 7
BASE_LOOKBACK = 100
LONG_LOOKBACK = 180
FORWARD_DAYS = 60
EXECUTION_TAIL_DAYS = 20
DEPENDENCY_DAYS = FORWARD_DAYS + EXECUTION_TAIL_DAYS
EXPECTED_QDP_AS_OF = "2026-07-16"
EXPECTED_SYMBOL_COUNT = 3_034
EXPECTED_DATE_COUNT = 3_966
EXPECTED_LONG_FIRST_DATE = "2010-09-29"
EXPECTED_LONG_TRAIN_ROWS = {2023: 5_542_816, 2024: 6_239_523, 2025: 6_952_766}
EXPECTED_CANDIDATE_ROWS = {2023: 703_468, 2024: 715_878, 2025: 724_125}
TOP_K_VALUES = (1, 3, 5, 10)
MEMORY_GUARD_GIB = 0.5
MEMORY_INTERVAL_SECONDS = 1.0
MEMORY_CONSECUTIVE_BREACHES = 2
OVERLAY_SIZE_LIMIT_BYTES = int(0.5 * 1024**3)
STALE_SECONDS = 15 * 60
MONITOR_POLL_SECONDS = 5.0
BOOTSTRAP_BLOCK_LENGTH = 60
BOOTSTRAP_REPLICATIONS = 10_000
CAPITAL_REVIEW_FIXED_DAYS = (14, 15, 16, 17, 18)
CAPITAL_REVIEW_SLOTS = (3, 6, 12, 24, 48)
CAPITAL_REVIEW_COST_SCENARIOS = ("base", "double_slippage")

EXPECTED_FROZEN_SHA256 = {
    "qdp_active": "13eb11087901ff7207dd4cc20faf52f8f6c5ed021c1364760bf7b39c6a093afc",
    "base_pack": "6dc5dc0dac612a6e5bf53732d374facf67c8cc4e11c532a85dcc03f771b1a16a",
    "structured_2023": "79a25d1fae4a5c0634e5b67127a5396bdda57770299e09bdaf0158fe05e246bc",
    "structured_2024": "5283e91e617e54088ce6e7f35ba8499767897efad92b2702092c699fbb33bede",
    "structured_2025": "9498081cfdbc6b69c3488c33a81ca2a5c52a6b85ee0e4bf8efac785ab43a77d2",
}

TURNOVER_COLUMNS = ("log_turnover_pct", "relative_turnover_20")
INTRADAY_COLUMNS = (
    "signed_path_efficiency",
    "close_to_vwap",
    "intraday_realized_vol",
    "low_time_frac",
    "high_before_low",
    "close_pressure_30m",
    "amount_concentration_hhi",
)


@dataclass(frozen=True)
class InputVariant:
    lookback_days: int
    turnover: bool = False
    intraday: bool = False

    @property
    def variant_id(self) -> str:
        suffix = "daily"
        if self.turnover and self.intraday:
            suffix = "turnover_intraday"
        elif self.turnover:
            suffix = "turnover"
        elif self.intraday:
            suffix = "intraday"
        return f"lookback{int(self.lookback_days)}_{suffix}"

    @property
    def input_channel_profile(self) -> str:
        if self.turnover and self.intraday:
            return training.INPUT_CHANNEL_PROFILE_DAILY_ONLY_TURNOVER_INTRADAY
        if self.turnover:
            return training.INPUT_CHANNEL_PROFILE_DAILY_ONLY_TURNOVER
        if self.intraday:
            return training.INPUT_CHANNEL_PROFILE_DAILY_ONLY_INTRADAY
        return training.INPUT_CHANNEL_PROFILE_DAILY_ONLY

    @property
    def input_dim(self) -> int:
        return 32 + (3 if self.turnover else 0) + (8 if self.intraday else 0)

    def to_dict(self) -> dict[str, Any]:
        return {
            "variant_id": self.variant_id,
            "lookback_days": int(self.lookback_days),
            "turnover": bool(self.turnover),
            "intraday": bool(self.intraday),
            "input_channel_profile": self.input_channel_profile,
            "input_dim": self.input_dim,
        }


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y-%m-%d")
    if isinstance(value, float) and not math.isfinite(value):
        return None
    raise TypeError(f"not JSON serializable: {type(value).__name__}")


def _canonical_digest(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=_json_default,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(
            dict(payload),
            ensure_ascii=False,
            indent=2,
            default=_json_default,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    frame.to_csv(temporary, index=False, encoding="utf-8-sig")
    os.replace(temporary, path)


def _write_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    frame.to_parquet(temporary, index=False)
    os.replace(temporary, path)


def _memmap_meta(path: Path, shape: Sequence[int], *, dtype: str) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "shape": [int(value) for value in shape],
        "dtype": str(dtype),
        "file_size": int(path.stat().st_size),
        "sha256": _file_sha256(path),
    }


def _open_memmap(meta: Mapping[str, Any], *, dtype: str, mode: str = "r") -> np.memmap:
    path = Path(str(meta["path"])).resolve()
    shape = tuple(int(value) for value in meta["shape"])
    if not path.is_file():
        raise FileNotFoundError(path)
    return np.memmap(path, dtype=dtype, mode=mode, shape=shape)


def _close_memmap(array: Any) -> None:
    if isinstance(array, np.memmap):
        array.flush()
        mmap = getattr(array, "_mmap", None)
        if mmap is not None:
            mmap.close()


def _directory_size(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def _replace_path_prefix(value: Any, old: str, new: str) -> Any:
    if isinstance(value, str):
        return value.replace(old, new)
    if isinstance(value, list):
        return [_replace_path_prefix(item, old, new) for item in value]
    if isinstance(value, dict):
        return {
            key: _replace_path_prefix(item, old, new) for key, item in value.items()
        }
    return value


def _safe_remove_staging(path: Path) -> None:
    resolved = path.resolve()
    expected = OVERLAY_ROOT.with_name(OVERLAY_ROOT.name + ".staging").resolve()
    if resolved != expected:
        raise ValueError(f"refusing to remove non-overlay staging path: {resolved}")
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()


def _old_structured_run(year: int) -> Path:
    matches = sorted(
        (BASE_STUDY_ROOT / "runs/development").glob(
            f"seq100_structured_structured_joint_turnover_{int(year)}_seed{SEED}_*"
        )
    )
    complete = [
        path
        for path in matches
        if (path / "sequence_path_training_summary.json").is_file()
        and (path / "best_model.pt").is_file()
    ]
    if len(complete) != 1:
        raise ValueError(f"expected exactly one frozen Structured run for {year}")
    return complete[0]


def _assert_frozen_inputs() -> dict[str, Any]:
    observed = {
        "qdp_active": _file_sha256(ACTIVE_QDP_PATH),
        "base_pack": _file_sha256(BASE_PACK_MANIFEST),
    }
    active = _read_json(ACTIVE_QDP_PATH)
    if str(active.get("active_as_of_date", "")) != EXPECTED_QDP_AS_OF:
        raise ValueError("QDP active_as_of drifted from the registered experiment")
    for year in DEVELOPMENT_YEARS:
        observed[f"structured_{year}"] = _file_sha256(
            _old_structured_run(year) / "best_model.pt"
        )
    for name, expected in EXPECTED_FROZEN_SHA256.items():
        if observed.get(name) != expected:
            raise ValueError(f"protected input drift: {name}")
    audit_path, audit = development._latest_full_audit(QDP_ROOT)
    live_state = WORKSPACE_ROOT / "daily_research/output/active_execution_strategy.json"
    return {
        "qdp_active_path": str(ACTIVE_QDP_PATH.resolve()),
        "qdp_active_sha256": observed["qdp_active"],
        "qdp_active_as_of": EXPECTED_QDP_AS_OF,
        "qdp_full_audit": str(audit_path.resolve()),
        "qdp_full_audit_sha256": _file_sha256(audit_path),
        "qdp_full_audit_status": str(audit["status"]),
        "base_pack_path": str(BASE_PACK_MANIFEST.resolve()),
        "base_pack_sha256": observed["base_pack"],
        "structured_checkpoint_sha256": {
            str(year): observed[f"structured_{year}"] for year in DEVELOPMENT_YEARS
        },
        "live_state_exists": bool(live_state.exists()),
    }


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
        "seq100_structured_input_ablation",
        "run-structured-input-ablation",
        "prepare-structured-input-ablation",
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
        raise RuntimeError(f"another Seq100 research process is active: pids={matches}")


def _base_structured_profile(batch_size: int = 512) -> TodayClosePathOnlyProfile:
    payload = structured._profile_contracts(int(batch_size))["structured_joint_turnover"]
    for key in ("store_view", "output_root", "run_tag"):
        payload.pop(key, None)
    return TodayClosePathOnlyProfile(**payload)


def _semantic_contract(batch_size: int) -> dict[str, Any]:
    profile = asdict(_base_structured_profile(batch_size))
    for key in ("store_view", "output_root", "run_tag"):
        profile.pop(key, None)
    return {
        "schema_version": 1,
        "contract_id": STUDY_ID,
        "experiment": EXPERIMENT_ID,
        "data": {
            "qdp_expected_as_of": EXPECTED_QDP_AS_OF,
            "base_pack": str(BASE_PACK_MANIFEST.resolve()),
            "base_pack_mutated": False,
            "development_years": list(DEVELOPMENT_YEARS),
            "lookback_candidates": [BASE_LOOKBACK, LONG_LOOKBACK],
            "forward_days": FORWARD_DAYS,
            "execution_tail_days": EXECUTION_TAIL_DAYS,
            "input_dims": [32, 35, 40, 43],
        },
        "development_protocol": {
            "years": list(DEVELOPMENT_YEARS),
            "method": "purged_expanding_development_walkforward",
            "split_roles": {"fit": "train", "evaluation": "development"},
            "seed": SEED,
            "purge_days": DEPENDENCY_DAYS,
            "candidate_membership_changes_with_input": False,
            "final_fit_in_scope": False,
        },
        "early_stopping": {
            "metric": training.EARLY_STOPPING_METRIC_DEVELOPMENT_PRICE_TOTAL_LOSS,
            "mode": training.EARLY_STOPPING_MODE_MIN,
            "minimum_complete_epochs": 1,
            "maximum_epochs": 10,
            "patience": 2,
            "restore_best_checkpoint": True,
        },
        "profile": {
            "name": "structured_joint_turnover",
            "input_dim": 0,
            "batch_size": int(batch_size),
            "parameters": profile,
        },
        "stage_order": [
            "100x32_vs_180x32",
            "incumbent_plus_turnover",
            "incumbent_plus_intraday_micro",
        ],
        "selection_rule": {
            "primary_objective": (
                "maximize equal-slot mean log liquidated terminal wealth over "
                "the identical development calendar under rolling_path and "
                "base costs"
            ),
            "slot_counts": list(CAPITAL_REVIEW_SLOTS),
            "holding_time_idle_cash_slot_pressure_and_skipped_signals": "included",
            "cohort_metrics_role": "diagnostic_only",
            "stress_and_risk_metrics_role": "reported_not_selection_gate",
        },
        "protected_boundaries": {
            "update_qdp": False,
            "call_provider": False,
            "mutate_base_pack": False,
            "mutate_old_checkpoint": False,
            "change_active_execution": False,
            "evaluate_2026": False,
            "create_final_model": False,
        },
    }


def _probe_cuda_batch_size() -> dict[str, Any]:
    if not torch.cuda.is_available():
        return {
            "device": "cpu",
            "selected_batch_size": 512,
            "effective_batch_size": 512,
            "gradient_accumulation_steps": 1,
            "attempts": [],
        }
    attempts: list[dict[str, Any]] = []
    props = torch.cuda.get_device_properties(0)
    total = int(props.total_memory)
    for batch_size in (512, 256):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        try:
            model = training.SequencePathModel(
                input_dim=43,
                hidden_dim=128,
                layers=2,
                forward_days=60,
                summary_dim=12,
                dropout=0.1,
                model_type="gru_structured_joint_turnover",
            ).cuda()
            x = torch.randn(batch_size, LONG_LOOKBACK, 43, device="cuda")
            target_raw = torch.randn(batch_size, 60, 4, device="cuda") * 0.1
            _geometry, y_path = training._structured_geometry_to_ohlc_torch(target_raw)
            y_activity = torch.randn(batch_size, 60, device="cuda")
            date_idx = torch.zeros(batch_size, dtype=torch.long, device="cuda")
            with torch.amp.autocast(device_type="cuda", enabled=True):
                output = model(x)
                loss, _parts = training._compute_loss(
                    output,
                    y_path.detach(),
                    torch.empty(batch_size, 0, device="cuda"),
                    date_idx,
                    y_activity_path=y_activity,
                    value_index=0,
                    path_weight=0.35,
                    summary_weight=0.20,
                    value_weight=0.15,
                    rank_weight=0.15,
                    geometry_weight=0.10,
                    utility_curve_weight=0.05,
                    turnover_level_weight=0.02,
                    turnover_delta_weight=0.01,
                    price_anchor="today_close",
                    summary_loss_profile=training.SUMMARY_LOSS_PROFILE_MULTI_HORIZON_OHLC,
                    path_value_gradient_profile=training.PATH_VALUE_GRADIENT_PROFILE_HARD_ST,
                )
            loss.backward()
            peak = int(torch.cuda.max_memory_reserved())
            attempts.append(
                {
                    "batch_size": int(batch_size),
                    "status": "ok",
                    "peak_reserved_bytes": peak,
                    "total_bytes": total,
                    "peak_reserved_ratio": float(peak / max(total, 1)),
                }
            )
            del model, x, target_raw, y_path, y_activity, date_idx, output, loss
            torch.cuda.empty_cache()
            return {
                "device": torch.cuda.get_device_name(0),
                "selected_batch_size": int(batch_size),
                "effective_batch_size": int(batch_size),
                "gradient_accumulation_steps": 1,
                "attempts": attempts,
            }
        except torch.cuda.OutOfMemoryError:
            attempts.append({"batch_size": int(batch_size), "status": "oom"})
            torch.cuda.empty_cache()
    raise RuntimeError("180-day CUDA batch probe failed at batch sizes 512 and 256")


def derive_intraday_features(
    *,
    open_values: Sequence[float],
    high_values: Sequence[float],
    low_values: Sequence[float],
    close_values: Sequence[float],
    volume_values: Sequence[float],
    amount_values: Sequence[float],
) -> tuple[np.ndarray, bool, float]:
    """Derive the registered seven-field microstructure vector from one day."""

    open_array = np.asarray(open_values, dtype=np.float64)
    high_array = np.asarray(high_values, dtype=np.float64)
    low_array = np.asarray(low_values, dtype=np.float64)
    close_array = np.asarray(close_values, dtype=np.float64)
    volume_array = np.asarray(volume_values, dtype=np.float64)
    amount_array = np.asarray(amount_values, dtype=np.float64)
    invalid = np.full(len(INTRADAY_COLUMNS), np.nan, dtype=np.float32)
    if any(array.shape != (48,) for array in (
        open_array,
        high_array,
        low_array,
        close_array,
        volume_array,
        amount_array,
    )):
        return invalid, False, float("nan")
    prices = np.column_stack((open_array, high_array, low_array, close_array))
    valid = bool(
        np.isfinite(prices).all()
        and (prices > 0.0).all()
        and np.isfinite(volume_array).all()
        and np.isfinite(amount_array).all()
        and (volume_array >= 0.0).all()
        and (amount_array >= 0.0).all()
        and (high_array >= np.maximum.reduce((open_array, low_array, close_array))).all()
        and (low_array <= np.minimum.reduce((open_array, high_array, close_array))).all()
        and float(volume_array.sum()) > 0.0
        and float(amount_array.sum()) > 0.0
    )
    if not valid:
        return invalid, False, float("nan")
    eps = 1.0e-12
    first_move = math.log(float(close_array[0] / open_array[0]))
    close_moves = np.log(close_array[1:] / close_array[:-1])
    denominator = abs(first_move) + float(np.abs(close_moves).sum()) + eps
    efficiency = float(np.clip(math.log(float(close_array[-1] / open_array[0])) / denominator, -1.0, 1.0))
    vwap = float(amount_array.sum() / volume_array.sum())
    close_to_vwap = math.log(float(close_array[-1] / vwap))
    realized_vol = math.sqrt(first_move * first_move + float(np.square(close_moves).sum()))
    low_position = int(np.flatnonzero(low_array == np.min(low_array))[0])
    high_position = int(np.flatnonzero(high_array == np.max(high_array))[0])
    low_time_frac = float(low_position / 47.0)
    high_before_low = 1.0 if high_position < low_position else (0.0 if high_position > low_position else 0.5)
    last_30m_ret = math.log(float(close_array[-1] / close_array[41]))
    last_30m_amount_share = float(amount_array[42:].sum() / amount_array.sum())
    close_pressure = last_30m_ret * last_30m_amount_share
    amount_weights = amount_array / amount_array.sum()
    hhi = float(np.square(amount_weights).sum())
    day_range = float(np.max(high_array) - np.min(low_array))
    close_position = (
        float((close_array[-1] - np.min(low_array)) / day_range)
        if day_range > 0.0
        else 0.5
    )
    return (
        np.asarray(
            [
                efficiency,
                close_to_vwap,
                realized_vol,
                low_time_frac,
                high_before_low,
                close_pressure,
                hhi,
            ],
            dtype=np.float32,
        ),
        True,
        close_position,
    )


def _build_turnover_overlay(
    staging: Path,
    *,
    pack: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    supplement = structured._validate_supplement_manifest(
        BASE_TURNOVER_MANIFEST, pack
    )
    shape = (int(pack["date_count"]), int(pack["symbol_count"]))
    log_panel = _open_memmap(supplement["log_turnover_pct"], dtype="float32")
    baseline_panel = _open_memmap(
        supplement["past20_positive_median"], dtype="float32"
    )
    values_path = staging / "panels/turnover.float32.dat"
    valid_path = staging / "masks/turnover_valid.bool.dat"
    values_path.parent.mkdir(parents=True, exist_ok=True)
    valid_path.parent.mkdir(parents=True, exist_ok=True)
    output = np.memmap(values_path, dtype=np.float32, mode="w+", shape=(*shape, 2))
    valid_output = np.memmap(valid_path, dtype=np.bool_, mode="w+", shape=shape)
    valid_count = 0
    for start in range(0, shape[0], 32):
        end = min(start + 32, shape[0])
        absolute = np.asarray(log_panel[start:end], dtype=np.float32)
        baseline = np.asarray(baseline_panel[start:end], dtype=np.float32)
        valid = np.isfinite(absolute) & np.isfinite(baseline)
        relative = absolute - baseline
        block = np.stack((absolute, relative), axis=2)
        block[~valid] = np.nan
        output[start:end] = block
        valid_output[start:end] = valid
        valid_count += int(valid.sum())
    _close_memmap(output)
    _close_memmap(valid_output)
    _close_memmap(log_panel)
    _close_memmap(baseline_panel)
    channel = {
        **_memmap_meta(values_path, (*shape, 2), dtype="float32"),
        "columns": list(TURNOVER_COLUMNS),
    }
    mask = _memmap_meta(valid_path, shape, dtype="bool")
    audit = {
        "valid_cell_count": int(valid_count),
        "total_cell_count": int(shape[0] * shape[1]),
        "coverage": float(valid_count / max(shape[0] * shape[1], 1)),
        "formula": "log1p(100 * daily_volume / same_day_float_share)",
        "relative_formula": "log_turnover_pct - median(last 20 positive observations through signal date)",
    }
    return {"channel": channel, "mask": mask}, audit


def _intraday_aggregate_query(shard_path: Path, start_date: str, end_date: str) -> str:
    escaped = str(shard_path).replace("'", "''")
    return f"""
        WITH ordered AS (
            SELECT
                q.symbol,
                CAST(q.trade_date AS VARCHAR) AS trade_date,
                CAST(q.bar_time AS BIGINT) AS bar_time,
                CAST(q.open AS DOUBLE) AS open_value,
                CAST(q.high AS DOUBLE) AS high_value,
                CAST(q.low AS DOUBLE) AS low_value,
                CAST(q.close AS DOUBLE) AS close_value,
                CAST(q.volume AS DOUBLE) AS volume_value,
                CAST(q.amount AS DOUBLE) AS amount_value,
                row_number() OVER (
                    PARTITION BY q.symbol, q.trade_date ORDER BY q.bar_time
                ) AS rn,
                lag(CAST(q.close AS DOUBLE)) OVER (
                    PARTITION BY q.symbol, q.trade_date ORDER BY q.bar_time
                ) AS previous_close
            FROM read_parquet('{escaped}') AS q
            INNER JOIN pack_symbols AS s ON q.symbol = s.symbol
            WHERE CAST(q.trade_date AS VARCHAR) BETWEEN '{start_date}' AND '{end_date}'
        )
        SELECT
            symbol,
            trade_date,
            count(*) AS bar_count,
            count(DISTINCT bar_time) AS distinct_bar_count,
            bool_and(
                isfinite(open_value) AND isfinite(high_value)
                AND isfinite(low_value) AND isfinite(close_value)
                AND open_value > 0 AND high_value > 0
                AND low_value > 0 AND close_value > 0
                AND high_value >= greatest(open_value, low_value, close_value)
                AND low_value <= least(open_value, high_value, close_value)
            ) AS valid_ohlc,
            bool_and(
                isfinite(volume_value) AND isfinite(amount_value)
                AND volume_value >= 0 AND amount_value >= 0
            ) AS valid_flow,
            first(open_value ORDER BY bar_time) AS first_open,
            first(close_value ORDER BY bar_time) AS first_close,
            last(close_value ORDER BY bar_time) AS last_close,
            sum(CASE WHEN previous_close IS NULL THEN 0.0
                     ELSE abs(ln(close_value / previous_close)) END) AS later_abs_move,
            sum(CASE WHEN previous_close IS NULL
                     THEN power(ln(close_value / open_value), 2)
                     ELSE power(ln(close_value / previous_close), 2) END) AS squared_move,
            sum(volume_value) AS total_volume,
            sum(amount_value) AS total_amount,
            first(rn ORDER BY high_value DESC, rn ASC) AS high_rn,
            first(rn ORDER BY low_value ASC, rn ASC) AS low_rn,
            max(CASE WHEN rn = 42 THEN close_value END) AS close_rn42,
            sum(CASE WHEN rn > 42 THEN amount_value ELSE 0.0 END) AS last6_amount,
            sum(amount_value * amount_value) AS amount_square_sum,
            max(high_value) AS day_high,
            min(low_value) AS day_low
        FROM ordered
        GROUP BY symbol, trade_date
        ORDER BY trade_date, symbol
    """


def _build_intraday_overlay(
    staging: Path,
    *,
    pack: Mapping[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    source, shards = structured._resolve_qdp_shards(pack, "market_intraday_5m")
    dates = [str(value) for value in pack["date_values"]]
    symbols = [str(value) for value in pack["symbol_values"]]
    date_index = {value: idx for idx, value in enumerate(dates)}
    symbol_index = {value: idx for idx, value in enumerate(symbols)}
    shape = (len(dates), len(symbols))
    values_path = staging / "panels/intraday_micro.float32.dat"
    valid_path = staging / "masks/intraday_valid.bool.dat"
    values_path.parent.mkdir(parents=True, exist_ok=True)
    valid_path.parent.mkdir(parents=True, exist_ok=True)
    output = np.memmap(values_path, dtype=np.float32, mode="w+", shape=(*shape, 7))
    output[:] = np.nan
    valid_output = np.memmap(valid_path, dtype=np.bool_, mode="w+", shape=shape)
    valid_output[:] = False

    connection = duckdb.connect(database=":memory:")
    connection.execute("SET threads=4")
    connection.execute("SET memory_limit='4GB'")
    temp_dir = staging / "duckdb_tmp"
    temp_dir.mkdir(parents=True, exist_ok=True)
    escaped_temp_dir = str(temp_dir).replace("'", "''")
    connection.execute(f"SET temp_directory='{escaped_temp_dir}'")
    connection.register("pack_symbols", pd.DataFrame({"symbol": symbols}))
    yearly: list[dict[str, Any]] = []
    try:
        for shard in list(source.get("shards", []) or []):
            shard_start = str(shard["start_date"])
            shard_end = str(shard["end_date"])
            start = max(shard_start, dates[0])
            end = min(shard_end, dates[-1])
            if start > end:
                continue
            shard_path = Path(str(pack["qdp_root"])) / str(shard["path"])
            reader = connection.execute(
                _intraday_aggregate_query(shard_path, start, end)
            ).fetch_record_batch(100_000)
            row_count = 0
            valid_count = 0
            feature_sum = np.zeros(7, dtype=np.float64)
            feature_square_sum = np.zeros(7, dtype=np.float64)
            close_position_sum = 0.0
            close_position_count = 0
            for record_batch in reader:
                frame = record_batch.to_pandas()
                row_count += int(len(frame))
                for row in frame.itertuples(index=False):
                    date_idx = date_index.get(str(row.trade_date))
                    symbol_idx = symbol_index.get(str(row.symbol))
                    if date_idx is None or symbol_idx is None:
                        raise ValueError("intraday aggregate contains an unregistered key")
                    is_valid = bool(
                        int(row.bar_count) == 48
                        and int(row.distinct_bar_count) == 48
                        and bool(row.valid_ohlc)
                        and bool(row.valid_flow)
                        and float(row.total_volume) > 0.0
                        and float(row.total_amount) > 0.0
                        and math.isfinite(float(row.close_rn42))
                    )
                    if not is_valid:
                        continue
                    first_move = math.log(float(row.first_close / row.first_open))
                    denominator = abs(first_move) + float(row.later_abs_move) + 1.0e-12
                    efficiency = float(
                        np.clip(
                            math.log(float(row.last_close / row.first_open)) / denominator,
                            -1.0,
                            1.0,
                        )
                    )
                    vwap = float(row.total_amount / row.total_volume)
                    close_to_vwap = math.log(float(row.last_close / vwap))
                    realized_vol = math.sqrt(max(float(row.squared_move), 0.0))
                    low_time_frac = float((int(row.low_rn) - 1) / 47.0)
                    high_before_low = (
                        1.0
                        if int(row.high_rn) < int(row.low_rn)
                        else (0.0 if int(row.high_rn) > int(row.low_rn) else 0.5)
                    )
                    close_pressure = math.log(float(row.last_close / row.close_rn42)) * float(
                        row.last6_amount / row.total_amount
                    )
                    hhi = float(row.amount_square_sum / (row.total_amount * row.total_amount))
                    values = np.asarray(
                        [
                            efficiency,
                            close_to_vwap,
                            realized_vol,
                            low_time_frac,
                            high_before_low,
                            close_pressure,
                            hhi,
                        ],
                        dtype=np.float32,
                    )
                    if not bool(np.isfinite(values).all()):
                        continue
                    output[date_idx, symbol_idx] = values
                    valid_output[date_idx, symbol_idx] = True
                    valid_count += 1
                    feature_sum += values.astype(np.float64)
                    feature_square_sum += np.square(values.astype(np.float64))
                    day_range = float(row.day_high - row.day_low)
                    close_position = (
                        float((row.last_close - row.day_low) / day_range)
                        if day_range > 0.0
                        else 0.5
                    )
                    close_position_sum += close_position
                    close_position_count += 1
            means = feature_sum / max(valid_count, 1)
            variances = np.maximum(
                feature_square_sum / max(valid_count, 1) - np.square(means), 0.0
            )
            yearly.append(
                {
                    "year": int(start[:4]),
                    "source_shard": str(shard_path.resolve()),
                    "source_shard_sha256": _file_sha256(shard_path),
                    "aggregated_symbol_day_count": int(row_count),
                    "valid_symbol_day_count": int(valid_count),
                    "valid_rate": float(valid_count / max(row_count, 1)),
                    "feature_mean": {
                        name: float(value)
                        for name, value in zip(INTRADAY_COLUMNS, means, strict=True)
                    },
                    "feature_std": {
                        name: float(value)
                        for name, value in zip(
                            INTRADAY_COLUMNS, np.sqrt(variances), strict=True
                        )
                    },
                    "close_position_diagnostic_mean": float(
                        close_position_sum / max(close_position_count, 1)
                    ),
                }
            )
            output.flush()
            valid_output.flush()
    finally:
        connection.close()
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
    _close_memmap(output)
    _close_memmap(valid_output)
    channel = {
        **_memmap_meta(values_path, (*shape, 7), dtype="float32"),
        "columns": list(INTRADAY_COLUMNS),
    }
    mask = _memmap_meta(valid_path, shape, dtype="bool")
    return {"channel": channel, "mask": mask, "qdp_source": source}, yearly


def _validate_overlay_manifest(path: Path) -> dict[str, Any]:
    manifest = _read_json(path)
    pack = _read_json(BASE_PACK_MANIFEST)
    if manifest.get("artifact_type") != "seq100_structured_input_ablation_overlay":
        raise ValueError("not a Structured input-ablation overlay")
    if manifest.get("shape") != [int(pack["date_count"]), int(pack["symbol_count"])]:
        raise ValueError("overlay shape drifted")
    if str(manifest.get("base_pack_sha256", "")) != _file_sha256(BASE_PACK_MANIFEST):
        raise ValueError("overlay base-pack binding drifted")
    referenced_size = int(path.stat().st_size)
    for section, names in (
        ("feature_channels", ("turnover", "intraday_micro")),
        ("masks", ("turnover_valid", "intraday_valid")),
    ):
        for name in names:
            meta = dict(dict(manifest[section])[name])
            data_path = Path(str(meta["path"])).resolve()
            if not data_path.is_file():
                raise FileNotFoundError(data_path)
            expected_size = int(np.prod(meta["shape"])) * np.dtype(meta["dtype"]).itemsize
            if int(data_path.stat().st_size) != expected_size:
                raise ValueError(f"overlay file size drifted: {name}")
            if _file_sha256(data_path) != str(meta["sha256"]):
                raise ValueError(f"overlay checksum drifted: {name}")
            referenced_size += int(data_path.stat().st_size)
    if referenced_size > OVERLAY_SIZE_LIMIT_BYTES:
        raise ValueError("overlay exceeds the registered 0.5 GiB limit")
    manifest["validated_size_bytes"] = int(referenced_size)
    return manifest


def build_overlay() -> dict[str, Any]:
    target = OVERLAY_ROOT.resolve()
    manifest_path = target / "manifest.json"
    if manifest_path.is_file():
        return _validate_overlay_manifest(manifest_path)
    staging = target.with_name(target.name + ".staging")
    if staging.exists():
        _safe_remove_staging(staging)
    staging.mkdir(parents=True)
    pack = _read_json(BASE_PACK_MANIFEST)
    if int(pack["date_count"]) != EXPECTED_DATE_COUNT or int(pack["symbol_count"]) != EXPECTED_SYMBOL_COUNT:
        raise ValueError("base pack date/symbol shape drifted")
    turnover, turnover_audit = _build_turnover_overlay(staging, pack=pack)
    intraday, intraday_audit = _build_intraday_overlay(staging, pack=pack)
    payload = {
        "schema_version": 1,
        "artifact_type": "seq100_structured_input_ablation_overlay",
        "created_at": _now(),
        "shape": [int(pack["date_count"]), int(pack["symbol_count"])],
        "date_start": str(pack["date_values"][0]),
        "date_end": str(pack["date_values"][-1]),
        "date_values_sha256": _canonical_digest(pack["date_values"]),
        "symbol_values_sha256": _canonical_digest(pack["symbol_values"]),
        "base_pack_path": str(BASE_PACK_MANIFEST.resolve()),
        "base_pack_sha256": _file_sha256(BASE_PACK_MANIFEST),
        "feature_channels": {
            "turnover": turnover["channel"],
            "intraday_micro": intraday["channel"],
        },
        "masks": {
            "turnover_valid": turnover["mask"],
            "intraday_valid": intraday["mask"],
        },
        "turnover_audit": turnover_audit,
        "intraday_audit": intraday_audit,
        "qdp_intraday_source": {
            "dataset_id": str(intraday["qdp_source"]["dataset_id"]),
            "dataset_json_sha256": str(
                dict(pack["qdp_source_manifests"])["market_intraday_5m"][
                    "dataset_json_sha256"
                ]
            ),
        },
        "protected_boundaries": {
            "qdp_read_only": True,
            "base_pack_copied": False,
            "provider_called": False,
        },
    }
    _write_json(staging / "manifest.json", payload)
    _validate_overlay_manifest(staging / "manifest.json")
    promoted_payload = _replace_path_prefix(
        payload, str(staging.resolve()), str(target.resolve())
    )
    _write_json(staging / "manifest.json", promoted_payload)
    if target.exists():
        raise FileExistsError(target)
    os.replace(staging, target)
    return _validate_overlay_manifest(target / "manifest.json")


def _normalization_for_view(
    source_view: Mapping[str, Any],
    overlay: Mapping[str, Any],
    variant: InputVariant,
) -> dict[str, Any]:
    normalization = json.loads(
        json.dumps(dict(source_view["normalization"]), ensure_ascii=False)
    )
    start_idx = 0
    development_start = str(
        dict(source_view["development_walkforward"])["development_start"]
    )
    dates = [str(value) for value in source_view["date_values"]]
    end_idx = dates.index(development_start)
    if variant.turnover:
        normalization["turnover"] = walkforward._fit_channel_normalization(
            dict(dict(overlay["feature_channels"])["turnover"]),
            start_idx=start_idx,
            end_idx_exclusive=end_idx,
        )
    if variant.intraday:
        normalization["intraday_micro"] = walkforward._fit_channel_normalization(
            dict(dict(overlay["feature_channels"])["intraday_micro"]),
            start_idx=start_idx,
            end_idx_exclusive=end_idx,
        )
    return normalization


def _long_sample_index(source_view: Mapping[str, Any], year: int) -> tuple[Path, dict[str, Any]]:
    target = OVERLAY_ROOT / "folds/indexes" / f"development_{year}_lookback180.parquet"
    source_path = Path(str(source_view["sample_index_path"])).resolve()
    source = pd.read_parquet(source_path)
    selected = source[
        ~source["split"].astype(str).eq("train")
        | source["date_idx"].astype(np.int64).ge(LONG_LOOKBACK - 1)
    ].copy()
    selected = selected.sort_values(["date_idx", "symbol_idx"], kind="mergesort").reset_index(drop=True)
    train = selected[selected["split"].astype(str).eq("train")]
    if int(len(train)) != EXPECTED_LONG_TRAIN_ROWS[int(year)]:
        raise ValueError(f"180-day training row count drifted for {year}")
    if str(train["trade_date"].min()) != EXPECTED_LONG_FIRST_DATE:
        raise ValueError(f"180-day first training date drifted for {year}")
    if target.is_file():
        observed = pd.read_parquet(target)
        if not observed.equals(selected):
            raise ValueError(f"stored 180-day index drifted for {year}")
    else:
        _write_parquet(target, selected)
    return target.resolve(), {
        "source_sample_index": str(source_path),
        "source_sample_index_sha256": _file_sha256(source_path),
        "sample_index": str(target.resolve()),
        "sample_index_sha256": _file_sha256(target),
        "train_row_count": int(len(train)),
        "development_row_count": int(
            selected["split"].astype(str).eq("development").sum()
        ),
        "first_train_signal": str(train["trade_date"].min()),
    }


def _view_fairness_material(view: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "candidate_index_sha256": _file_sha256(
            Path(str(view["candidate_index_path"])).resolve()
        ),
        "candidate_key_semantics": "unchanged_base_candidate_index_bytes",
        "label_material_sha256": _canonical_digest(view["label_arrays"]),
        "execution_material_sha256": _canonical_digest(
            {
                "execution_arrays": view["execution_arrays"],
                "execution_contract": dict(view.get("execution_contract", {}) or {}),
                "exit_sellable": dict(view["masks"])["exit_sellable"],
                "entry_filled": dict(view["masks"])["entry_filled"],
            }
        ),
        "cost_contract_sha256": _canonical_digest(view["execution_cost_contract"]),
    }


def _build_view(
    *,
    source_path: Path,
    source_view: Mapping[str, Any],
    overlay: Mapping[str, Any],
    variant: InputVariant,
    year: int,
    study_path: Path,
    sample_index_path: Path,
) -> Path:
    target = OVERLAY_ROOT / "folds/views" / f"{variant.variant_id}_{int(year)}.json"
    contract = _read_json(study_path)
    binding = {
        "contract_id": STUDY_ID,
        "contract_sha256": str(contract["contract_sha256"]),
        "contract_file_sha256": str(contract["contract_sha256"]),
        "path": str(study_path.resolve()),
    }
    view = json.loads(json.dumps(dict(source_view), ensure_ascii=False))
    channels = {
        "daily_raw": dict(dict(source_view["feature_channels"])["daily_raw"]),
        "daily_state": dict(dict(source_view["feature_channels"])["daily_state"]),
    }
    masks = dict(source_view["masks"])
    input_masks: list[str] = []
    if variant.turnover:
        channels["turnover"] = dict(dict(overlay["feature_channels"])["turnover"])
        masks["turnover_valid"] = dict(dict(overlay["masks"])["turnover_valid"])
        input_masks.append("turnover_valid")
    if variant.intraday:
        channels["intraday_micro"] = dict(
            dict(overlay["feature_channels"])["intraday_micro"]
        )
        masks["intraday_valid"] = dict(dict(overlay["masks"])["intraday_valid"])
        input_masks.append("intraday_valid")
    semantics = dict(source_view["data_semantics"])
    semantics["model_input_mask_features"] = input_masks
    semantics["structured_input_ablation"] = variant.to_dict()
    samples = pd.read_parquet(sample_index_path, columns=["split", "trade_date"])
    source_artifact_view = source_view.get("artifact_view", {})
    artifact_view = (
        dict(source_artifact_view)
        if isinstance(source_artifact_view, Mapping)
        else {}
    )
    artifact_view.update(
        {
            "schema_version": 1,
            "view_id": f"{STUDY_ID}_{variant.variant_id}_{year}",
            "view_type": "structured_input_ablation_development_view",
        }
    )
    view.update(
        {
            "created_at": _now(),
            "lookback_days": int(variant.lookback_days),
            "sample_index_path": str(sample_index_path.resolve()),
            "sample_count": int(len(samples)),
            "sample_count_by_split": {
                str(key): int(value)
                for key, value in samples["split"].astype(str).value_counts().items()
            },
            "feature_channels": channels,
            "masks": masks,
            "normalization": _normalization_for_view(source_view, overlay, variant),
            "data_semantics": semantics,
            "research_contract": binding,
            "development_contract": binding,
            "artifact_view": artifact_view,
            "structured_input_ablation": {
                **variant.to_dict(),
                "source_view": str(source_path.resolve()),
                "source_view_sha256": _file_sha256(source_path),
                "overlay_manifest": str((OVERLAY_ROOT / "manifest.json").resolve()),
                "overlay_manifest_sha256": _file_sha256(OVERLAY_ROOT / "manifest.json"),
            },
        }
    )
    view["development_fold_training_contract"] = (
        walkforward._compute_development_fold_training_contract(view)
    )
    if not target.is_file() or _canonical_digest(_read_json(target)) != _canonical_digest(view):
        _write_json(target, view)
    verification = walkforward.verify_development_walkforward_view(target)
    if verification["status"] != "ok":
        raise RuntimeError(
            f"Structured input-ablation view failed verification: {verification['blockers']}"
        )
    return target.resolve()


def _all_variants() -> list[InputVariant]:
    return [
        InputVariant(lookback, turnover, intraday)
        for lookback in (BASE_LOOKBACK, LONG_LOOKBACK)
        for turnover, intraday in ((False, False), (True, False), (False, True), (True, True))
    ]


def prepare_structured_input_ablation(
    *, study_root: Path = STUDY_ROOT
) -> dict[str, Any]:
    _assert_no_other_research_process()
    frozen = _assert_frozen_inputs()
    study_path = study_root.resolve() / "study.json"
    if study_path.is_file():
        study = _read_json(study_path)
        if str(study.get("study_id", "")) != STUDY_ID:
            raise ValueError("refusing to reuse an unrelated study root")
        if dict(study.get("material", {}) or {}).get("views"):
            verify_prepared_study(study_root=study_root)
            return study
        if _canonical_digest(study["contract"]) != str(study["contract_sha256"]):
            raise ValueError("incomplete study contract drifted")
        study["runtime"].update(
            {"status": "preparing", "updated_at": _now(), "error": None}
        )
        _write_json(study_path, study)
    else:
        probe = _probe_cuda_batch_size()
        semantic = _semantic_contract(int(probe["selected_batch_size"]))
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
                "current_stage": 1,
                "completed_training_tasks": [],
            },
            "cuda_probe": probe,
            "material": {},
        }
        _write_json(study_path, study)
    try:
        overlay = build_overlay()
        base_study = _read_json(BASE_STUDY_ROOT / "study.json")
        source_views = dict(dict(base_study["material"])["fold_views"])
        long_indexes: dict[str, Any] = {}
        source_payloads: dict[int, tuple[Path, dict[str, Any]]] = {}
        for year in DEVELOPMENT_YEARS:
            source_path = Path(str(source_views[str(year)])).resolve()
            source_view = _read_json(source_path)
            source_payloads[year] = (source_path, source_view)
            index_path, audit = _long_sample_index(source_view, year)
            long_indexes[str(year)] = {**audit, "path": str(index_path)}

        views: dict[str, dict[str, str]] = {}
        fairness: dict[str, dict[str, Any]] = {}
        for variant in _all_variants():
            variant_views: dict[str, str] = {}
            for year in DEVELOPMENT_YEARS:
                source_path, source_view = source_payloads[year]
                sample_path = (
                    Path(long_indexes[str(year)]["path"])
                    if variant.lookback_days == LONG_LOOKBACK
                    else Path(str(source_view["sample_index_path"])).resolve()
                )
                target = _build_view(
                    source_path=source_path,
                    source_view=source_view,
                    overlay=overlay,
                    variant=variant,
                    year=year,
                    study_path=study_path,
                    sample_index_path=sample_path,
                )
                variant_views[str(year)] = str(target)
                material = _view_fairness_material(_read_json(target))
                fairness.setdefault(str(year), material)
                if fairness[str(year)] != material:
                    raise ValueError(
                        f"candidate/label/execution/cost material differs across variants for {year}"
                    )
            views[variant.variant_id] = variant_views

        for year in DEVELOPMENT_YEARS:
            source_path, source_view = source_payloads[year]
            candidate_count = len(
                pd.read_parquet(
                    Path(str(source_view["candidate_index_path"])),
                    columns=["candidate_id"],
                )
            )
            if candidate_count != EXPECTED_CANDIDATE_ROWS[year]:
                raise ValueError(f"candidate count drifted for {year}")
            daily_normalization = {
                key: dict(source_view["normalization"])[key]
                for key in ("daily_raw", "daily_state")
            }
            for variant in _all_variants():
                prepared = _read_json(Path(views[variant.variant_id][str(year)]))
                observed = {
                    key: dict(prepared["normalization"])[key]
                    for key in ("daily_raw", "daily_state")
                }
                if observed != daily_normalization:
                    raise ValueError(f"daily normalization changed for {variant.variant_id}/{year}")

        study = _read_json(study_path)
        study["material"] = {
            "base_pack_manifest": str(BASE_PACK_MANIFEST.resolve()),
            "base_study": str(BASE_STUDY_ROOT.resolve()),
            "overlay_manifest": str((OVERLAY_ROOT / "manifest.json").resolve()),
            "overlay_size_bytes": int(overlay["validated_size_bytes"]),
            "long_indexes": long_indexes,
            "views": views,
            "fairness_material": fairness,
            "frozen_inputs": frozen,
            "base_pack_copied": False,
        }
        study["runtime"].update(
            {
                "status": "prepared",
                "updated_at": _now(),
                "current_stage": 1,
                "current_task": None,
                "error": None,
            }
        )
        _write_json(study_path, study)
        verify_prepared_study(study_root=study_root)
        return study
    except Exception as exc:
        failed = _read_json(study_path)
        failed["runtime"].update(
            {"status": "prepare_failed", "updated_at": _now(), "error": str(exc)}
        )
        _write_json(study_path, failed)
        raise


def verify_prepared_study(
    *, study_root: Path = STUDY_ROOT, deep: bool = True
) -> dict[str, Any]:
    study_path = study_root.resolve() / "study.json"
    study = _read_json(study_path)
    if str(study.get("study_id", "")) != STUDY_ID:
        raise ValueError("Structured input-ablation study identity drifted")
    if _canonical_digest(study["contract"]) != str(study["contract_sha256"]):
        raise ValueError("Structured input-ablation contract drifted")
    frozen = _assert_frozen_inputs()
    recorded = dict(dict(study["material"])["frozen_inputs"])
    for key in (
        "qdp_active_sha256",
        "base_pack_sha256",
        "structured_checkpoint_sha256",
        "live_state_exists",
    ):
        if recorded.get(key) != frozen.get(key):
            raise ValueError(f"protected boundary drifted: {key}")
    overlay = _validate_overlay_manifest(
        Path(str(dict(study["material"])["overlay_manifest"]))
    )
    views = dict(dict(study["material"])["views"])
    expected_variants = {variant.variant_id for variant in _all_variants()}
    if set(views) != expected_variants:
        raise ValueError("prepared variant set drifted")
    for variant_id, yearly in views.items():
        if set(dict(yearly)) != {str(year) for year in DEVELOPMENT_YEARS}:
            raise ValueError(f"prepared year set drifted for {variant_id}")
        for path in dict(yearly).values():
            if not Path(str(path)).is_file():
                raise FileNotFoundError(path)
    if not deep:
        return {
            "status": "ok",
            "study": str(study_path),
            "verification_depth": "run_preflight",
            "view_count": int(len(_all_variants()) * len(DEVELOPMENT_YEARS)),
            "overlay_size_bytes": int(overlay["validated_size_bytes"]),
            "protected_inputs": frozen,
        }
    for variant in _all_variants():
        for year in DEVELOPMENT_YEARS:
            path = Path(str(dict(views[variant.variant_id])[str(year)]))
            verification = walkforward.verify_development_walkforward_view(path)
            if verification["status"] != "ok":
                raise ValueError(f"prepared view invalid: {variant.variant_id}/{year}")
            view = _read_json(path)
            dataset = training.SequencePathPackDataset(
                view,
                split="development",
                max_samples=0,
                input_channel_profile=variant.input_channel_profile,
                index_role="candidate",
            )
            try:
                if int(dataset.input_dim) != variant.input_dim:
                    raise ValueError(
                        f"input dimension drifted: {variant.variant_id}/{year}"
                    )
                if len(dataset) != EXPECTED_CANDIDATE_ROWS[year]:
                    raise ValueError(
                        f"candidate count drifted: {variant.variant_id}/{year}"
                    )
            finally:
                del dataset
                gc.collect()
    return {
        "status": "ok",
        "study": str(study_path),
        "verification_depth": "deep",
        "view_count": int(len(_all_variants()) * len(DEVELOPMENT_YEARS)),
        "overlay_size_bytes": int(overlay["validated_size_bytes"]),
        "protected_inputs": frozen,
    }


def _variant_from_payload(payload: Mapping[str, Any]) -> InputVariant:
    return InputVariant(
        lookback_days=int(payload["lookback_days"]),
        turnover=bool(payload.get("turnover", False)),
        intraday=bool(payload.get("intraday", False)),
    )


def _initial_incumbent() -> dict[str, Any]:
    return {
        "source": "existing_100x32",
        "origin_stage": 0,
        "variant": InputVariant(BASE_LOOKBACK).to_dict(),
    }


def _load_stage_decisions(study_root: Path) -> dict[str, Any]:
    path = study_root.resolve() / "stage_decisions.json"
    if not path.is_file():
        return {
            "schema_version": 2,
            "artifact_type": "seq100_structured_input_ablation_stage_decisions",
            "study_id": STUDY_ID,
            "stages": [],
        }
    payload = _read_json(path)
    if str(payload.get("study_id", "")) != STUDY_ID:
        raise ValueError("stage-decision study identity drifted")
    changed = int(payload.get("schema_version", 1)) < 2
    stages: list[dict[str, Any]] = []
    for raw in list(payload.get("stages", []) or []):
        row = dict(raw)
        legacy_checks = row.pop("gates", None)
        legacy_passed = row.pop("challenger_passed", None)
        row.pop("cohort_reference_preferred", None)
        row.pop("selected_incumbent", None)
        row.pop("overall_model_rejection", None)
        if legacy_checks is not None or "cohort_diagnostics" not in row:
            observations = (
                dict(legacy_checks)
                if isinstance(legacy_checks, Mapping)
                else {}
            )
            row["cohort_diagnostics"] = {
                "role": "diagnostic_only",
                "selection_authority": False,
                "registered_observations": observations,
                "legacy_all_observations_favorable": (
                    bool(legacy_passed) if legacy_passed is not None else None
                ),
                "interpretation": (
                    "These observations describe cohort ranking, path, and exit "
                    "behavior. They cannot promote or reject a strategy."
                ),
            }
            changed = True
        speed_review = dict(row.get("capital_speed_review", {}) or {})
        review = dict(row.get("capital_efficiency_review", {}) or {})
        if str(speed_review.get("status", "")) == "completed":
            row["decision_scope"] = "joint_model_capital_speed_primary"
            row["interpretation"] = (
                "A complete model is selected by continuous-account capital "
                "speed. Its own exit is used only when it beats that model's "
                "best fixed exit."
            )
        elif str(review.get("status", "")) == "completed":
            row["decision_scope"] = "continuous_account_profit_primary"
            row["interpretation"] = (
                "The stage strategy is selected only by continuous-account "
                "liquidated terminal wealth over the common calendar window. "
                "Cohort metrics are diagnostic."
            )
        else:
            row["decision_scope"] = "cohort_diagnostics_pending_capital_review"
            row["interpretation"] = (
                "No strategy is selected until the continuous-account capital "
                "review is complete."
            )
        stages.append(row)
    if changed:
        payload["schema_version"] = 2
        payload["stages"] = stages
        payload["updated_at"] = _now()
        _write_json(path, payload)
    else:
        payload["stages"] = stages
    return payload


def _decision_for_stage(decisions: Mapping[str, Any], stage: int) -> dict[str, Any] | None:
    rows = [
        dict(row)
        for row in list(decisions.get("stages", []) or [])
        if int(row.get("stage", 0)) == int(stage)
    ]
    if len(rows) > 1:
        raise ValueError(f"multiple decisions recorded for stage {stage}")
    return rows[0] if rows else None


def _incumbent_before_stage(decisions: Mapping[str, Any], stage: int) -> dict[str, Any]:
    if int(stage) == 1:
        return _initial_incumbent()
    previous = _decision_for_stage(decisions, int(stage) - 1)
    if previous is None:
        raise ValueError(f"stage {stage} cannot start before stage {stage - 1} is decided")
    speed_review = dict(previous.get("capital_speed_review", {}) or {})
    if speed_review:
        if str(speed_review.get("status", "")) != "completed":
            raise ValueError(
                f"Stage {stage - 1} capital-speed review is incomplete"
            )
        selected_model = speed_review.get("selected_model_descriptor")
        if not isinstance(selected_model, Mapping):
            raise ValueError(
                f"Stage {stage - 1} capital-speed review has no selected model"
            )
        return dict(selected_model)
    review = dict(previous.get("capital_efficiency_review", {}) or {})
    if str(review.get("status", "")) != "completed":
        raise ValueError(
            f"stage {stage} is blocked until the Stage {stage - 1} "
            "finite-capital review completes"
        )
    branch = review.get("training_branch_descriptor")
    if not isinstance(branch, Mapping):
        raise ValueError(
            f"Stage {stage - 1} capital review has no training branch"
        )
    return dict(branch)


def _challenger_for_stage(decisions: Mapping[str, Any], stage: int) -> dict[str, Any]:
    incumbent = _incumbent_before_stage(decisions, stage)
    current = _variant_from_payload(dict(incumbent["variant"]))
    if int(stage) == 1:
        variant = InputVariant(LONG_LOOKBACK)
    elif int(stage) == 2:
        variant = InputVariant(current.lookback_days, turnover=True, intraday=False)
    elif int(stage) == 3:
        variant = InputVariant(
            current.lookback_days,
            turnover=current.turnover,
            intraday=True,
        )
    else:
        raise ValueError(f"invalid stage: {stage}")
    return {
        "source": "ablation_run",
        "origin_stage": int(stage),
        "variant": variant.to_dict(),
    }


def _task_id(stage: int, variant: InputVariant, year: int) -> str:
    return f"stage{int(stage)}:{variant.variant_id}:{int(year)}"


def _run_tag(stage: int, variant: InputVariant, year: int) -> str:
    return (
        f"seq100_structured_ablation_s{int(stage)}_{variant.variant_id}_"
        f"{int(year)}_seed{SEED}"
    )


def _task_run_dirs(
    *, study_root: Path, stage: int, variant: InputVariant, year: int
) -> list[Path]:
    root = study_root.resolve() / "runs" / f"stage_{int(stage)}"
    return sorted(root.glob(f"{_run_tag(stage, variant, year)}_*"))


def _validate_training_run(
    run_dir: Path,
    *,
    study_path: Path,
    stage: int,
    variant: InputVariant,
    year: int,
) -> dict[str, Any]:
    summary_path = run_dir / "sequence_path_training_summary.json"
    progress_path = run_dir / "progress.json"
    checkpoint_path = run_dir / "best_model.pt"
    if not summary_path.is_file() or not progress_path.is_file() or not checkpoint_path.is_file():
        raise FileNotFoundError(f"incomplete training artifacts: {run_dir}")
    summary = development._validate_run_summary(summary_path, year)
    progress = _read_json(progress_path)
    if str(progress.get("status", "")) != "completed":
        raise ValueError(f"training progress is not terminal-completed: {run_dir}")
    if Path(str(progress.get("summary_json", ""))).resolve() != summary_path.resolve():
        raise ValueError("training progress points to a different summary")
    if Path(str(summary.get("best_checkpoint", ""))).resolve() != checkpoint_path.resolve():
        raise ValueError("training summary points to a different checkpoint")
    if _file_sha256(checkpoint_path) != str(summary.get("best_checkpoint_sha256", "")):
        raise ValueError("training checkpoint checksum drifted")
    config = dict(summary.get("resolved_training_config", {}) or {})
    exact = {
        "model_type": "gru_structured_joint_turnover",
        "input_channel_profile": variant.input_channel_profile,
        "seed": SEED,
        "batch_size": int(
            dict(_read_json(study_path)["cuda_probe"])["selected_batch_size"]
        ),
    }
    for key, expected in exact.items():
        if config.get(key) != expected:
            raise ValueError(
                f"training config drifted for {key}: {config.get(key)!r} != {expected!r}"
            )
    if int(summary.get("lookback_days", 0)) != int(variant.lookback_days):
        raise ValueError("training lookback drifted")
    if list(summary.get("input_channels", []) or []) != training._input_channel_order(
        variant.input_channel_profile
    ):
        raise ValueError("training input channel order drifted")
    expected_masks = [
        name
        for name, active in (
            ("turnover_valid", variant.turnover),
            ("intraday_valid", variant.intraday),
        )
        if active
    ]
    if list(summary.get("input_mask_features", []) or []) != expected_masks:
        raise ValueError("training input mask order drifted")
    model = dict(summary.get("model", {}) or {})
    if int(model.get("input_dim", summary.get("input_dim", 0)) or 0) not in {
        0,
        variant.input_dim,
    }:
        raise ValueError("training model input dimension drifted")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if int(checkpoint.get("input_dim", 0)) != variant.input_dim:
        raise ValueError("checkpoint input dimension drifted")
    checkpoint_config = dict(checkpoint.get("resolved_training_config", {}) or {})
    if str(checkpoint_config.get("input_channel_profile", "")) != variant.input_channel_profile:
        raise ValueError("checkpoint input profile drifted")
    if tuple(checkpoint["model_state_dict"]["proj.weight"].shape) != (
        128,
        variant.input_dim,
    ):
        raise ValueError("checkpoint input projection shape drifted")
    view_path = Path(str(summary["pack_manifest"])).resolve()
    expected_view = Path(
        str(
            dict(dict(_read_json(study_path)["material"])["views"])[
                variant.variant_id
            ][str(year)]
        )
    ).resolve()
    if view_path != expected_view:
        raise ValueError("training used the wrong fold view")
    view = _read_json(view_path)
    fold_contract = dict(view["development_fold_training_contract"])
    if dict(summary.get("development_fold_training_contract", {}) or {}) != fold_contract:
        raise ValueError("summary fold contract drifted")
    if dict(checkpoint.get("fold_training_contract", {}) or {}) != fold_contract:
        raise ValueError("checkpoint fold contract drifted")
    if str(summary.get("candidate_index_sha256", "")) != _file_sha256(
        Path(str(view["candidate_index_path"]))
    ):
        raise ValueError("candidate index binding drifted")
    return summary


def _classify_task_runs(
    *, study_path: Path, stage: int, variant: InputVariant, year: int
) -> tuple[list[Path], list[Path]]:
    completed: list[Path] = []
    partial: list[Path] = []
    for path in _task_run_dirs(
        study_root=study_path.parent, stage=stage, variant=variant, year=year
    ):
        try:
            _validate_training_run(
                path,
                study_path=study_path,
                stage=stage,
                variant=variant,
                year=year,
            )
        except (
            EOFError,
            FileNotFoundError,
            KeyError,
            OSError,
            RuntimeError,
            TypeError,
            ValueError,
        ):
            partial.append(path)
        else:
            completed.append(path)
    if len(completed) > 1:
        raise ValueError(f"task has multiple valid completed runs: {stage}/{year}")
    return completed, partial


def _archive_partial_runs(
    paths: Sequence[Path], *, study_root: Path, task_id: str
) -> list[str]:
    archived: list[str] = []
    failed_root = study_root.resolve() / "runs/failed" / task_id.replace(":", "__")
    failed_root.mkdir(parents=True, exist_ok=True)
    for number, source in enumerate(paths, start=1):
        if study_root.resolve() not in source.resolve().parents:
            raise ValueError(f"refusing to archive a run outside the study: {source}")
        target = failed_root / f"{source.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{number}"
        shutil.move(str(source), str(target))
        archived.append(str(target.resolve()))
    return archived


def _append_monitor_event(study_root: Path, payload: Mapping[str, Any]) -> None:
    event = {"timestamp": _now(), **dict(payload)}
    path = study_root.resolve() / "monitor_events.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(event, ensure_ascii=False, default=_json_default, allow_nan=False)
            + "\n"
        )
    _write_json(study_root.resolve() / "monitor.json", event)
    message = str(event.get("message", event.get("event", "status")))
    try:
        print(f"structured-ablation: {message}", flush=True)
    except OSError:
        # A closed parent console must not stop the file-backed supervisor.
        pass


def _parse_timestamp(value: Any) -> float | None:
    try:
        return datetime.fromisoformat(str(value)).timestamp()
    except (TypeError, ValueError):
        return None


def classify_monitor_terminal(
    *, exit_code: int, memory_guard_status: str, artifacts_valid: bool
) -> str:
    if str(memory_guard_status) == "killed_low_available_memory":
        return "low_memory_terminated"
    if int(exit_code) != 0:
        return "task_failed"
    if not bool(artifacts_valid):
        return "invalid_terminal_artifacts"
    return "task_completed"


def _supervise_training_command(
    *,
    command: Sequence[str],
    study_root: Path,
    stage: int,
    variant: InputVariant,
    year: int,
    task_id: str,
) -> Path:
    log_root = study_root.resolve() / "logs"
    log_root.mkdir(parents=True, exist_ok=True)
    memory_path = log_root / f"memory_guard_{task_id.replace(':', '__')}.json"
    stdout_path = log_root / f"worker_{task_id.replace(':', '__')}.log"
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
    before = set(
        _task_run_dirs(
            study_root=study_root, stage=stage, variant=variant, year=year
        )
    )
    _append_monitor_event(
        study_root,
        {
            "event": "task_started",
            "status": "running",
            "task_id": task_id,
            "stage": int(stage),
            "fold": int(year),
            "variant": variant.to_dict(),
            "message": f"start {task_id}",
        },
    )
    current_run: Path | None = None
    last_phase: str | None = None
    last_epoch: int | None = None
    last_bucket: int | None = None
    stale_reported = False
    memory_warning_reported = False
    with stdout_path.open("w", encoding="utf-8") as stdout:
        process = subprocess.Popen(
            guard_command,
            cwd=str(WORKSPACE_ROOT),
            stdout=stdout,
            stderr=subprocess.STDOUT,
        )
        while process.poll() is None:
            if current_run is None:
                after = set(
                    _task_run_dirs(
                        study_root=study_root,
                        stage=stage,
                        variant=variant,
                        year=year,
                    )
                )
                created = sorted(after.difference(before))
                if len(created) > 1:
                    process.kill()
                    raise RuntimeError(f"task created multiple run directories: {task_id}")
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
            total_batches = int(progress.get("total_batches", 0) or 0)
            batch = int(progress.get("batch", 0) or 0)
            bucket = (
                min(100, int(math.floor(10.0 * batch / total_batches) * 10))
                if total_batches > 0
                else 0
            )
            changed = phase != last_phase or epoch != last_epoch
            bucket_changed = total_batches > 0 and bucket != last_bucket
            if changed or bucket_changed:
                _append_monitor_event(
                    study_root,
                    {
                        "event": "progress",
                        "status": "running",
                        "task_id": task_id,
                        "stage": int(stage),
                        "fold": int(year),
                        "phase": phase,
                        "epoch": epoch,
                        "batch": batch,
                        "total_batches": total_batches,
                        "batch_percent_bucket": bucket,
                        "available_memory_gib": float(
                            psutil.virtual_memory().available / 1024**3
                        ),
                        "run_dir": str(current_run) if current_run else None,
                        "message": (
                            f"{task_id} phase={phase} epoch={epoch} batch={batch}/{total_batches}"
                        ),
                    },
                )
                last_phase, last_epoch, last_bucket = phase, epoch, bucket
            updated = _parse_timestamp(progress.get("updated_at"))
            stale = updated is not None and time.time() - updated > STALE_SECONDS
            if stale and not stale_reported:
                _append_monitor_event(
                    study_root,
                    {
                        "event": "stale_progress",
                        "status": "warning",
                        "task_id": task_id,
                        "stale_seconds": float(time.time() - updated),
                        "message": f"{task_id} progress stale for more than 15 minutes",
                    },
                )
                stale_reported = True
            elif not stale:
                stale_reported = False
            if memory_path.is_file():
                try:
                    memory = _read_json(memory_path)
                except (OSError, json.JSONDecodeError):
                    memory = {}
                breaches = int(memory.get("available_breaches", 0) or 0)
                if breaches > 0 and not memory_warning_reported:
                    _append_monitor_event(
                        study_root,
                        {
                            "event": "memory_warning",
                            "status": "warning",
                            "task_id": task_id,
                            "available_breaches": breaches,
                            "current_available_gib": memory.get("current_available_gb"),
                            "message": f"{task_id} memory guard warning",
                        },
                    )
                    memory_warning_reported = True
                elif breaches == 0:
                    memory_warning_reported = False
            time.sleep(MONITOR_POLL_SECONDS)
        exit_code = int(process.returncode or 0)

    if current_run is None:
        after = set(
            _task_run_dirs(
                study_root=study_root, stage=stage, variant=variant, year=year
            )
        )
        created = sorted(after.difference(before))
        current_run = created[-1] if created else None
    memory = _read_json(memory_path) if memory_path.is_file() else {}
    if current_run is not None:
        _write_json(current_run / "memory_guard.json", memory)
    if exit_code != 0:
        event = classify_monitor_terminal(
            exit_code=exit_code,
            memory_guard_status=str(memory.get("status", "")),
            artifacts_valid=False,
        )
        _append_monitor_event(
            study_root,
            {
                "event": event,
                "status": "failed",
                "task_id": task_id,
                "exit_code": exit_code,
                "run_dir": str(current_run) if current_run else None,
                "memory_guard": memory,
                "message": f"{task_id} failed with exit code {exit_code}",
            },
        )
        raise subprocess.CalledProcessError(exit_code, guard_command)
    if current_run is None:
        raise RuntimeError(f"task produced no run directory: {task_id}")
    _validate_training_run(
        current_run,
        study_path=study_root / "study.json",
        stage=stage,
        variant=variant,
        year=year,
    )
    _append_monitor_event(
        study_root,
        {
            "event": "training_completed",
            "status": "completed",
            "task_id": task_id,
            "stage": int(stage),
            "fold": int(year),
            "run_dir": str(current_run.resolve()),
            "memory_guard_status": memory.get("status"),
            "message": f"training completed {task_id}",
        },
    )
    return current_run


def _descriptor_run_dir(
    descriptor: Mapping[str, Any], *, year: int, study_path: Path
) -> Path:
    if str(descriptor["source"]) == "existing_100x32":
        return _old_structured_run(year)
    stage = int(descriptor["origin_stage"])
    variant = _variant_from_payload(dict(descriptor["variant"]))
    completed, _partial = _classify_task_runs(
        study_path=study_path, stage=stage, variant=variant, year=year
    )
    if len(completed) != 1:
        raise ValueError(f"missing completed challenger run for stage {stage}/{year}")
    return completed[0]


def _descriptor_view_path(
    descriptor: Mapping[str, Any], *, year: int, study_path: Path
) -> Path:
    if str(descriptor["source"]) == "existing_100x32":
        source = _read_json(BASE_STUDY_ROOT / "study.json")
        return Path(str(dict(source["material"])["fold_views"][str(year)])).resolve()
    variant = _variant_from_payload(dict(descriptor["variant"]))
    study = _read_json(study_path)
    return Path(
        str(dict(dict(study["material"])["views"])[variant.variant_id][str(year)])
    ).resolve()


def _year_evaluation(
    descriptor: Mapping[str, Any], *, year: int, study_path: Path
) -> dict[str, Any]:
    run_dir = _descriptor_run_dir(descriptor, year=year, study_path=study_path)
    view_path = _descriptor_view_path(descriptor, year=year, study_path=study_path)
    variant = _variant_from_payload(dict(descriptor["variant"]))
    metrics, summary = development._year_metrics(run_dir, year)
    diagnostics = structured.stream_checkpoint_diagnostics(
        run_dir=run_dir,
        view_path=view_path,
        output_dir=run_dir / "path_diagnostics",
        year=year,
        compare_legacy_domain=False,
        fixed_exit_comparison=False,
        input_channel_profile=variant.input_channel_profile,
    )
    split = next(
        row
        for row in list(summary.get("split_metrics", []) or [])
        if str(row.get("split", "")) == "development"
    )
    metrics.update(
        {
            "path_mae": float(split["path_mae"]),
            "path_open_mae": float(split["path_open_mae"]),
            "path_high_mae": float(split["path_high_mae"]),
            "path_low_mae": float(split["path_low_mae"]),
            "path_close_mae": float(split["path_close_mae"]),
            "rank_ic_positive_day_rate": float(split["rank_ic_positive_day_rate"]),
            "exit_regret": float(diagnostics["exit_regret"]),
            "exact_raw_oracle_executable_regret": float(
                diagnostics["exact_raw_oracle_executable_regret"]
            ),
            "close_cumulative_path_mae": float(
                diagnostics["close_cumulative_path_mae"]
            ),
            "top3_legal_exit_max_day_share": float(
                diagnostics["top3_legal_exit_max_day_share"]
            ),
            "top3_exit_deferral_rate": float(
                diagnostics["top3_exit_deferral_rate"]
            ),
            "ohlc_geometry_violation_count": int(
                diagnostics["ohlc_geometry_violation_count"]
            ),
            "candidate_count": int(split["row_count"]),
            "date_count": int(split["date_count"]),
        }
    )
    view = _read_json(view_path)
    fairness = _view_fairness_material(view)
    if str(summary.get("candidate_index_sha256", "")) != str(
        fairness["candidate_index_sha256"]
    ):
        raise ValueError("evaluation candidate binding drifted")
    if str(summary.get("execution_cost_contract_sha256", "")) != str(
        fairness["cost_contract_sha256"]
    ):
        raise ValueError("evaluation cost binding drifted")
    return {
        "development_year": int(year),
        "descriptor": dict(descriptor),
        "run_dir": str(run_dir.resolve()),
        "view_path": str(view_path.resolve()),
        "checkpoint_path": str(Path(str(summary["best_checkpoint"])).resolve()),
        "checkpoint_sha256": str(summary["best_checkpoint_sha256"]),
        "summary_sha256": _file_sha256(run_dir / "sequence_path_training_summary.json"),
        "path_diagnostics_sha256": _file_sha256(
            run_dir / "path_diagnostics" / f"checkpoint_path_diagnostics_{year}.json"
        ),
        "fairness_material": fairness,
        "metrics": metrics,
    }


def _descriptor_evaluation(
    descriptor: Mapping[str, Any], *, study_path: Path
) -> dict[str, Any]:
    yearly = [
        _year_evaluation(descriptor, year=year, study_path=study_path)
        for year in DEVELOPMENT_YEARS
    ]
    metric_keys = sorted(
        key
        for key, value in dict(yearly[0]["metrics"]).items()
        if isinstance(value, (int, float, np.integer, np.floating))
        and key not in {"development_year", "best_epoch", "completed_epochs"}
    )
    equal = {
        key: float(
            np.mean([float(dict(row["metrics"])[key]) for row in yearly])
        )
        for key in metric_keys
    }
    equal.update(
        {
            "positive_top3_years": int(
                sum(float(dict(row["metrics"])["top3_base_alpha"]) > 0.0 for row in yearly)
            ),
            "worst_year_top3_base_alpha": float(
                min(float(dict(row["metrics"])["top3_base_alpha"]) for row in yearly)
            ),
            "maximum_top3_exit_day_share": float(
                max(
                    float(dict(row["metrics"])["top3_legal_exit_max_day_share"])
                    for row in yearly
                )
            ),
            "geometry_violation_count": int(
                sum(
                    int(dict(row["metrics"])["ohlc_geometry_violation_count"])
                    for row in yearly
                )
            ),
        }
    )
    fairness_by_year = {
        str(row["development_year"]): dict(row["fairness_material"])
        for row in yearly
    }
    return {
        "descriptor": dict(descriptor),
        "yearly": yearly,
        "equal_year": equal,
        "fairness_by_year": fairness_by_year,
    }


def stage_cohort_diagnostics(
    *,
    stage: int,
    incumbent: Mapping[str, Any],
    challenger: Mapping[str, Any],
) -> dict[str, Any]:
    incumbent_equal = dict(incumbent["equal_year"])
    challenger_equal = dict(challenger["equal_year"])
    incumbent_yearly = {
        int(row["development_year"]): dict(row["metrics"])
        for row in incumbent["yearly"]
    }
    challenger_yearly = {
        int(row["development_year"]): dict(row["metrics"])
        for row in challenger["yearly"]
    }
    fairness_identical = bool(
        dict(incumbent["fairness_by_year"]) == dict(challenger["fairness_by_year"])
    )
    close_noninferior = {
        str(year): bool(
            float(challenger_yearly[year]["path_close_mae"])
            <= float(incumbent_yearly[year]["path_close_mae"])
        )
        for year in DEVELOPMENT_YEARS
    }
    exit_limit = max(
        0.50, float(incumbent_equal["maximum_top3_exit_day_share"]) + 0.05
    )
    observations = {
        "equal_year_top3_base_alpha_strictly_higher": bool(
            float(challenger_equal["top3_base_alpha"])
            > float(incumbent_equal["top3_base_alpha"])
        ),
        "all_year_top3_alpha_positive": bool(
            all(
                float(challenger_yearly[year]["top3_base_alpha"]) > 0.0
                for year in DEVELOPMENT_YEARS
            )
        ),
        "worst_year_top3_not_lower": bool(
            float(challenger_equal["worst_year_top3_base_alpha"])
            >= float(incumbent_equal["worst_year_top3_base_alpha"])
        ),
        "equal_year_exit_regret_strictly_lower": bool(
            float(challenger_equal["exit_regret"])
            < float(incumbent_equal["exit_regret"])
        ),
        "close_path_mae_noninferior_at_least_two_folds": bool(
            sum(close_noninferior.values()) >= 2
        ),
        "ohlc_geometry_violations_zero": bool(
            int(challenger_equal["geometry_violation_count"]) == 0
        ),
        "exit_concentration_within_limit": bool(
            float(challenger_equal["maximum_top3_exit_day_share"]) <= exit_limit
        ),
        "score_coverage_complete": bool(
            all(
                float(challenger_yearly[year]["candidate_score_coverage"]) == 1.0
                for year in DEVELOPMENT_YEARS
            )
        ),
        "fairness_material_identical": fairness_identical,
    }
    return {
        "stage": int(stage),
        "created_at": _now(),
        "incumbent": dict(incumbent["descriptor"]),
        "challenger": dict(challenger["descriptor"]),
        "incumbent_equal_year_metrics": incumbent_equal,
        "challenger_equal_year_metrics": challenger_equal,
        "close_path_mae_noninferior_by_fold": close_noninferior,
        "exit_concentration_limit": float(exit_limit),
        "cohort_diagnostics": {
            "role": "diagnostic_only",
            "selection_authority": False,
            "registered_observations": observations,
            "legacy_all_observations_favorable": None,
            "interpretation": (
                "These observations describe cohort ranking, path, and exit "
                "behavior. They cannot promote or reject a strategy."
            ),
        },
        "decision_scope": "cohort_diagnostics_pending_capital_review",
        "capital_efficiency_review": {
            "status": "pending",
            "reason": (
                "continuous-account evaluation is required because cohort "
                "returns do not price holding time, idle cash, slot pressure, "
                "or skipped signals"
            ),
        },
        "interpretation": (
            "No strategy is selected until the continuous-account capital "
            "review is complete."
        ),
    }


def _write_task_evaluations(
    evaluation: Mapping[str, Any], *, study_path: Path
) -> None:
    descriptor = dict(evaluation["descriptor"])
    if str(descriptor["source"]) != "ablation_run":
        return
    for row in evaluation["yearly"]:
        run_dir = Path(str(row["run_dir"]))
        payload = {
            "schema_version": 1,
            "artifact_type": "seq100_structured_input_ablation_evaluation",
            "status": "completed",
            "completed_at": _now(),
            "study_id": STUDY_ID,
            "study_contract_sha256": str(_read_json(study_path)["contract_sha256"]),
            "descriptor": descriptor,
            "development_year": int(row["development_year"]),
            "checkpoint_path": str(row["checkpoint_path"]),
            "checkpoint_sha256": str(row["checkpoint_sha256"]),
            "summary_sha256": str(row["summary_sha256"]),
            "path_diagnostics_sha256": str(row["path_diagnostics_sha256"]),
            "fairness_material": dict(row["fairness_material"]),
            "metrics": dict(row["metrics"]),
            "qdp_changed": False,
            "provider_called": False,
            "live_state_changed": False,
        }
        _write_json(run_dir / "evaluation.json", payload)


def _record_stage_decision(
    *, study_path: Path, stage: int, incumbent: Mapping[str, Any], challenger: Mapping[str, Any]
) -> dict[str, Any]:
    decisions = _load_stage_decisions(study_path.parent)
    existing = _decision_for_stage(decisions, stage)
    computed = stage_cohort_diagnostics(
        stage=stage, incumbent=incumbent, challenger=challenger
    )
    if existing is not None:
        comparable = dict(existing)
        comparable.pop("created_at", None)
        comparable.pop("capital_efficiency_review", None)
        comparable.pop("decision_scope", None)
        comparable.pop("interpretation", None)
        expected = dict(computed)
        expected.pop("created_at", None)
        expected.pop("capital_efficiency_review", None)
        expected.pop("decision_scope", None)
        expected.pop("interpretation", None)
        for payload in (comparable, expected):
            diagnostics = dict(payload.get("cohort_diagnostics", {}) or {})
            diagnostics.pop("legacy_all_observations_favorable", None)
            payload["cohort_diagnostics"] = diagnostics
        if comparable != expected:
            raise ValueError(f"recorded stage {stage} diagnostics drifted")
        return existing
    _write_task_evaluations(challenger, study_path=study_path)
    decisions["stages"].append(computed)
    decisions["updated_at"] = _now()
    _write_json(study_path.parent / "stage_decisions.json", decisions)
    _append_monitor_event(
        study_path.parent,
        {
            "event": "stage_cohort_diagnostics_completed",
            "status": "completed",
            "stage": int(stage),
            "selection_authority": False,
            "message": f"stage {stage} cohort diagnostics completed",
        },
    )
    return computed


def _update_study_runtime(study_path: Path, **changes: Any) -> None:
    study = _read_json(study_path)
    runtime = dict(study.get("runtime", {}) or {})
    runtime.update(changes)
    runtime["updated_at"] = _now()
    study["runtime"] = runtime
    _write_json(study_path, study)


def run_structured_input_ablation(
    *, study_root: Path = STUDY_ROOT, max_tasks: int = 0
) -> dict[str, Any]:
    _assert_no_other_research_process()
    verify_prepared_study(study_root=study_root, deep=False)
    study_path = study_root.resolve() / "study.json"
    launched = 0
    _update_study_runtime(study_path, status="running", error=None)
    try:
        for stage in (1, 2, 3):
            decisions = _load_stage_decisions(study_path.parent)
            if int(stage) > 1:
                previous_stage = int(stage) - 1
                previous = _decision_for_stage(decisions, previous_stage)
                review = (
                    dict(previous.get("capital_efficiency_review", {}) or {})
                    if previous is not None
                    else {}
                )
                if str(review.get("status", "")) != "completed":
                    if previous_stage == 1:
                        run_stage1_capital_review(study_root=study_path.parent)
                    else:
                        run_stage_capital_review(
                            stage=previous_stage, study_root=study_path.parent
                        )
                    decisions = _load_stage_decisions(study_path.parent)
            if _decision_for_stage(decisions, stage) is not None:
                continue
            incumbent_descriptor = _incumbent_before_stage(decisions, stage)
            challenger_descriptor = _challenger_for_stage(decisions, stage)
            variant = _variant_from_payload(dict(challenger_descriptor["variant"]))
            for year in DEVELOPMENT_YEARS:
                task = _task_id(stage, variant, year)
                completed, partial = _classify_task_runs(
                    study_path=study_path,
                    stage=stage,
                    variant=variant,
                    year=year,
                )
                if completed:
                    continue
                if partial:
                    archived = _archive_partial_runs(
                        partial, study_root=study_path.parent, task_id=task
                    )
                    _append_monitor_event(
                        study_path.parent,
                        {
                            "event": "partial_archived",
                            "status": "recovering",
                            "task_id": task,
                            "archived_runs": archived,
                            "message": f"archived partial artifacts for {task}",
                        },
                    )
                if int(max_tasks) > 0 and launched >= int(max_tasks):
                    _update_study_runtime(
                        study_path,
                        status="running",
                        current_stage=int(stage),
                        current_task=None,
                    )
                    return status_structured_input_ablation(study_root=study_root)
                study = _read_json(study_path)
                view = Path(
                    str(
                        dict(dict(study["material"])["views"])[variant.variant_id][
                            str(year)
                        ]
                    )
                )
                base = _base_structured_profile(
                    int(dict(study["cuda_probe"])["selected_batch_size"])
                )
                profile = replace(
                    base,
                    store_view=view,
                    output_root=study_path.parent / "runs" / f"stage_{stage}",
                    run_tag=_run_tag(stage, variant, year),
                    epochs=10,
                    seed=SEED,
                    top_k="1,3,5,10",
                    input_channel_profile=variant.input_channel_profile,
                    prediction_mode="compact",
                    evaluation_mode="development",
                    early_stopping_patience=2,
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
                _update_study_runtime(
                    study_path,
                    status="running",
                    current_stage=int(stage),
                    current_task=task,
                )
                _supervise_training_command(
                    command=command,
                    study_root=study_path.parent,
                    stage=stage,
                    variant=variant,
                    year=year,
                    task_id=task,
                )
                launched += 1
                _update_study_runtime(
                    study_path,
                    status="running",
                    current_stage=int(stage),
                    current_task=None,
                )

            completed_all = all(
                len(
                    _classify_task_runs(
                        study_path=study_path,
                        stage=stage,
                        variant=variant,
                        year=year,
                    )[0]
                )
                == 1
                for year in DEVELOPMENT_YEARS
            )
            if not completed_all:
                return status_structured_input_ablation(study_root=study_root)
            _update_study_runtime(
                study_path,
                status="evaluating_stage",
                current_stage=int(stage),
                current_task=None,
            )
            incumbent_evaluation = _descriptor_evaluation(
                incumbent_descriptor, study_path=study_path
            )
            challenger_evaluation = _descriptor_evaluation(
                challenger_descriptor, study_path=study_path
            )
            _record_stage_decision(
                study_path=study_path,
                stage=stage,
                incumbent=incumbent_evaluation,
                challenger=challenger_evaluation,
            )
            if int(stage) == 1:
                run_stage1_capital_review(study_root=study_path.parent)
            else:
                run_stage_capital_review(
                    stage=int(stage), study_root=study_path.parent
                )
            if int(max_tasks) > 0 and launched >= int(max_tasks):
                return status_structured_input_ablation(study_root=study_root)

        decisions = _load_stage_decisions(study_path.parent)
        if len(list(decisions.get("stages", []) or [])) != 3:
            raise RuntimeError("all training completed without three stage decisions")
        _update_study_runtime(
            study_path,
            status="runs_completed",
            current_stage=None,
            current_task=None,
            error=None,
        )
    except Exception as exc:
        _update_study_runtime(
            study_path,
            status="run_failed",
            current_task=None,
            error=str(exc),
        )
        _append_monitor_event(
            study_path.parent,
            {
                "event": "workflow_failed",
                "status": "failed",
                "error": str(exc),
                "message": f"workflow failed: {exc}",
            },
        )
        raise
    return status_structured_input_ablation(study_root=study_root)


def status_structured_input_ablation(
    *, study_root: Path = STUDY_ROOT
) -> dict[str, Any]:
    study_path = study_root.resolve() / "study.json"
    if not study_path.is_file():
        return {
            "status": "not_prepared",
            "study": str(study_path),
            "available_memory_gib": float(psutil.virtual_memory().available / 1024**3),
        }
    study = _read_json(study_path)
    decisions = _load_stage_decisions(study_path.parent)
    tasks: dict[str, Any] = {}
    for stage in (1, 2, 3):
        decision = _decision_for_stage(decisions, stage)
        if decision is not None:
            descriptor = dict(decision["challenger"])
        else:
            try:
                descriptor = _challenger_for_stage(decisions, stage)
            except ValueError:
                tasks[f"stage{stage}"] = {"status": "blocked_on_previous_stage"}
                continue
        variant = _variant_from_payload(dict(descriptor["variant"]))
        for year in DEVELOPMENT_YEARS:
            complete, partial = _classify_task_runs(
                study_path=study_path,
                stage=stage,
                variant=variant,
                year=year,
            )
            progress = None
            if partial and (partial[-1] / "progress.json").is_file():
                try:
                    progress = _read_json(partial[-1] / "progress.json")
                except (OSError, json.JSONDecodeError):
                    progress = None
            evaluation_path = complete[0] / "evaluation.json" if complete else None
            tasks[_task_id(stage, variant, year)] = {
                "status": (
                    "evaluated"
                    if evaluation_path is not None and evaluation_path.is_file()
                    else (
                        "training_completed"
                        if complete
                        else ("partial" if partial else "pending")
                    )
                ),
                "completed_run": str(complete[0]) if complete else None,
                "partial_runs": [str(path) for path in partial],
                "progress": progress,
            }
    monitor_path = study_path.parent / "monitor.json"
    return {
        "status": str(dict(study.get("runtime", {}) or {}).get("status", "unknown")),
        "study": str(study_path),
        "runtime": dict(study.get("runtime", {}) or {}),
        "decided_stage_count": int(len(list(decisions.get("stages", []) or []))),
        "stage_decisions": str(study_path.parent / "stage_decisions.json"),
        "monitor": _read_json(monitor_path) if monitor_path.is_file() else None,
        "tasks": tasks,
        "available_memory_gib": float(psutil.virtual_memory().available / 1024**3),
        "h_free_gib": float(shutil.disk_usage(WORKSPACE_ROOT).free / 1024**3),
    }


def _daily_evidence(
    descriptor: Mapping[str, Any], *, year: int, study_path: Path
) -> pd.DataFrame:
    run_dir = _descriptor_run_dir(descriptor, year=year, study_path=study_path)
    daily = pd.read_csv(run_dir / "daily_topk_metrics.csv")
    daily = daily[
        daily["split"].astype(str).eq("development")
        & daily["top_k"].astype(int).eq(3)
    ].copy()
    ic = pd.read_csv(run_dir / "daily_rank_ic.csv")
    ic = ic[ic["split"].astype(str).eq("development")].copy()
    if bool(daily["trade_date"].astype(str).duplicated().any()) or bool(
        ic["trade_date"].astype(str).duplicated().any()
    ):
        raise ValueError("daily stage evidence contains duplicate dates")
    columns = {
        "alpha_net_realized_plan_return_base": "top3_base_alpha",
        "alpha_net_realized_plan_return_stress": "top3_stress_alpha",
        "alpha_opportunity_value": "top3_opportunity_alpha",
        "universe_oracle_regret": "top3_exit_regret",
    }
    selected = daily[
        [
            "trade_date",
            "universe_hash",
            "universe_count",
            "execution_cost_contract_sha256",
            *columns,
        ]
    ].rename(columns=columns)
    selected["trade_date"] = selected["trade_date"].astype(str)
    ic = ic[["trade_date", "rank_ic", "count"]].copy()
    ic["trade_date"] = ic["trade_date"].astype(str)
    merged = selected.merge(ic, on="trade_date", how="outer", validate="one_to_one")
    if len(merged) != int(daily["trade_date"].nunique()):
        raise ValueError("daily evidence date coverage drifted")
    return merged.sort_values("trade_date", kind="mergesort").reset_index(drop=True)


def _daily_pairwise_deltas(
    *, decisions: Mapping[str, Any], study_path: Path
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    metrics = (
        "top3_base_alpha",
        "top3_stress_alpha",
        "top3_opportunity_alpha",
        "top3_exit_regret",
        "rank_ic",
    )
    for decision in list(decisions.get("stages", []) or []):
        stage = int(decision["stage"])
        incumbent = dict(decision["incumbent"])
        challenger = dict(decision["challenger"])
        for year in DEVELOPMENT_YEARS:
            left = _daily_evidence(incumbent, year=year, study_path=study_path)
            right = _daily_evidence(challenger, year=year, study_path=study_path)
            common = (
                left[
                    [
                        "trade_date",
                        "universe_hash",
                        "universe_count",
                        "execution_cost_contract_sha256",
                        "count",
                    ]
                ]
                .merge(
                    right[
                        [
                            "trade_date",
                            "universe_hash",
                            "universe_count",
                            "execution_cost_contract_sha256",
                            "count",
                        ]
                    ],
                    on="trade_date",
                    how="outer",
                    suffixes=("_incumbent", "_challenger"),
                    validate="one_to_one",
                    indicator=True,
                )
            )
            if not bool(common["_merge"].eq("both").all()):
                raise ValueError(f"stage {stage}/{year} daily keys differ")
            for field in (
                "universe_hash",
                "universe_count",
                "execution_cost_contract_sha256",
                "count",
            ):
                if not bool(
                    common[f"{field}_incumbent"].astype(str).eq(
                        common[f"{field}_challenger"].astype(str)
                    ).all()
                ):
                    raise ValueError(f"stage {stage}/{year} daily {field} differs")
            paired = left.merge(
                right,
                on="trade_date",
                suffixes=("_incumbent", "_challenger"),
                how="inner",
                validate="one_to_one",
            )
            for metric in metrics:
                incumbent_values = pd.to_numeric(
                    paired[f"{metric}_incumbent"], errors="coerce"
                ).to_numpy(dtype=np.float64)
                challenger_values = pd.to_numeric(
                    paired[f"{metric}_challenger"], errors="coerce"
                ).to_numpy(dtype=np.float64)
                finite = np.isfinite(incumbent_values) & np.isfinite(challenger_values)
                if not bool(finite.all()):
                    raise ValueError(f"stage {stage}/{year}/{metric} is not fully finite")
                delta = challenger_values - incumbent_values
                lower_better = metric == "top3_exit_regret"
                for idx, trade_date in enumerate(paired["trade_date"].astype(str)):
                    rows.append(
                        {
                            "stage": stage,
                            "development_year": int(year),
                            "trade_date": trade_date,
                            "metric": metric,
                            "difference_direction": "challenger_minus_incumbent",
                            "incumbent_value": float(incumbent_values[idx]),
                            "challenger_value": float(challenger_values[idx]),
                            "delta": float(delta[idx]),
                            "challenger_win": bool(
                                delta[idx] < 0.0 if lower_better else delta[idx] > 0.0
                            ),
                            "incumbent_variant": str(
                                dict(incumbent["variant"])["variant_id"]
                            ),
                            "challenger_variant": str(
                                dict(challenger["variant"])["variant_id"]
                            ),
                        }
                    )
    return pd.DataFrame(rows).sort_values(
        ["stage", "metric", "development_year", "trade_date"], kind="mergesort"
    ).reset_index(drop=True)


def noncircular_grouped_moving_block_interval(
    groups: Sequence[Sequence[float] | np.ndarray],
    *,
    block_length: int = BOOTSTRAP_BLOCK_LENGTH,
    replications: int = BOOTSTRAP_REPLICATIONS,
    seed: int = SEED,
) -> dict[str, Any]:
    arrays = [np.asarray(group, dtype=np.float64).reshape(-1) for group in groups]
    if not arrays or any(not bool(np.isfinite(group).all()) for group in arrays):
        raise ValueError("moving-block groups must be non-empty and finite")
    if any(len(group) < int(block_length) for group in arrays):
        raise ValueError("moving-block group is shorter than the block length")
    rng = np.random.default_rng(int(seed))
    samples = np.empty(int(replications), dtype=np.float64)
    for replication in range(int(replications)):
        fold_means: list[float] = []
        for group in arrays:
            selected: list[np.ndarray] = []
            count = 0
            while count < len(group):
                start = int(rng.integers(0, len(group) - int(block_length) + 1))
                block = group[start : start + int(block_length)]
                selected.append(block)
                count += len(block)
            fold_means.append(float(np.concatenate(selected)[: len(group)].mean()))
        samples[replication] = float(np.mean(fold_means))
    observed = float(np.mean([float(group.mean()) for group in arrays]))
    return {
        "method": "noncircular_grouped_moving_block_fold_equal_mean",
        "block_length": int(block_length),
        "replications": int(replications),
        "seed": int(seed),
        "fold_count": int(len(arrays)),
        "day_count": int(sum(len(group) for group in arrays)),
        "observed_fold_equal_mean_delta": observed,
        "ci_low": float(np.quantile(samples, 0.025)),
        "ci_high": float(np.quantile(samples, 0.975)),
        "positive_replication_rate": float((samples > 0.0).mean()),
        "negative_replication_rate": float((samples < 0.0).mean()),
        "formal_gate": False,
    }


def _bootstrap_diagnostics(daily: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for (stage, metric), frame in daily.groupby(["stage", "metric"], sort=True):
        groups = [
            group.sort_values("trade_date", kind="mergesort")["delta"].to_numpy(
                dtype=np.float64
            )
            for _, group in frame.groupby("development_year", sort=True)
        ]
        result = noncircular_grouped_moving_block_interval(
            groups,
            seed=SEED + int(stage) * 1009 + sum(ord(char) for char in str(metric)),
        )
        rows.append(
            {
                "stage": int(stage),
                "metric": str(metric),
                "challenger_win_day_rate": float(frame["challenger_win"].astype(bool).mean()),
                **result,
            }
        )
    return rows


def _vintage_metrics_frame(
    evaluations: Sequence[Mapping[str, Any]],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for evaluation in evaluations:
        descriptor = dict(evaluation["descriptor"])
        variant = dict(descriptor["variant"])
        for yearly in evaluation["yearly"]:
            rows.append(
                {
                    "source": str(descriptor["source"]),
                    "origin_stage": int(descriptor["origin_stage"]),
                    **variant,
                    "aggregation": "year",
                    "development_year": int(yearly["development_year"]),
                    **dict(yearly["metrics"]),
                }
            )
        rows.append(
            {
                "source": str(descriptor["source"]),
                "origin_stage": int(descriptor["origin_stage"]),
                **variant,
                "aggregation": "equal_year",
                "development_year": "equal_year",
                **dict(evaluation["equal_year"]),
            }
        )
    return pd.DataFrame(rows)


def _comparison_markdown(summary: Mapping[str, Any]) -> str:
    lines = [
        "# Structured Input Ablation 2023-2025",
        "",
        "Only Structured joint turnover was trained. The original 100x32 result is the registered incumbent; no 2026 data, final fit, deployment, QDP update, or provider call is included.",
        "",
        "Strategy selection maximizes continuous-account unit-time profit over the same calendar window. Cohort metrics below diagnose behavior only; they do not pass, fail, promote, or reject a model.",
        "",
        "| Stage | Reference | New input | Top3 cohort alpha: reference | Top3 cohort alpha: new input | Capital-selected strategy |",
        "|---:|---|---|---:|---:|---|",
    ]
    for decision in list(summary["stage_decisions"]):
        selected = dict(
            dict(decision["capital_efficiency_review"])["selected_strategy"]
        )
        lines.append(
            "| {stage} | {incumbent} | {challenger} | {inc:.4%} | {chal:.4%} | {selected} |".format(
                stage=int(decision["stage"]),
                incumbent=dict(decision["incumbent"])["variant"]["variant_id"],
                challenger=dict(decision["challenger"])["variant"]["variant_id"],
                inc=float(decision["incumbent_equal_year_metrics"]["top3_base_alpha"]),
                chal=float(decision["challenger_equal_year_metrics"]["top3_base_alpha"]),
                selected=str(selected["configuration"]),
            )
        )
    final_strategy = dict(summary["final_strategy"])
    final_rank = dict(
        dict(final_strategy["ranking_descriptor"])["variant"]
    )["variant_id"]
    final_exit = dict(
        dict(final_strategy["exit_descriptor"])["variant"]
    )["variant_id"]
    lines.extend(
        [
            "",
            (
                "Final capital-selected strategy: "
                f"ranking `{final_rank}`, path exit `{final_exit}`."
            ),
            "",
            "Stress, drawdown, cohort metrics, and 60-day moving-block intervals are diagnostics; none overrides the continuous-account profit objective.",
        ]
    )
    return "\n".join(lines) + "\n"


FEATURE_DUPLICATE_PAIRS = (
    ("daily_raw:open_ret_prev_close", "daily_state:open_gap_1d"),
    ("daily_raw:intraday_range_raw", "daily_state:range_1d"),
    ("daily_raw:close_ret_prev_close", "daily_state:ret_1d"),
    ("daily_state:distance_to_20d_high", "daily_state:drawdown_from_20d_high"),
)

OCCLUSION_GROUP_COLUMNS = {
    "ohlc_level": (
        "daily_raw:open",
        "daily_raw:high",
        "daily_raw:low",
        "daily_raw:close",
    ),
    "raw_volume_amount": (
        "daily_raw:volume",
        "daily_raw:amount",
        "daily_raw:volume_log",
        "daily_raw:amount_log",
    ),
    "momentum": (
        "daily_state:ret_3d",
        "daily_state:ret_5d",
        "daily_state:ret_10d",
        "daily_state:ret_20d",
    ),
    "volatility_liquidity": (
        "daily_state:volatility_20d",
        "daily_state:amount_mean_20d_log",
        "daily_state:amount_ratio_5_20",
        "daily_state:volume_ratio_5_20",
    ),
    "price_position": (
        "daily_state:distance_to_60d_high",
        "daily_state:distance_to_20d_low",
    ),
    "candlestick_shape": (
        "daily_raw:high_ret_prev_close",
        "daily_raw:low_ret_prev_close",
        "daily_state:body_to_range_1d",
        "daily_state:upper_shadow_to_range_1d",
        "daily_state:lower_shadow_to_range_1d",
        "daily_state:close_to_open_1d",
    ),
    "duplicate_aliases": tuple(
        dict.fromkeys(item for pair in FEATURE_DUPLICATE_PAIRS for item in pair)
    ),
}


def _base_feature_names(pack: Mapping[str, Any]) -> list[str]:
    names: list[str] = []
    for channel in ("daily_raw", "daily_state"):
        names.extend(
            f"{channel}:{column}"
            for column in dict(dict(pack["feature_channels"])[channel])["columns"]
        )
    if len(names) != 32:
        raise ValueError("base daily feature dimension drifted from 32")
    return names


def _daily_feature_statistics() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    pack = _read_json(BASE_PACK_MANIFEST)
    names = _base_feature_names(pack)
    counts = np.zeros(32, dtype=np.int64)
    sums = np.zeros(32, dtype=np.float64)
    square_sums = np.zeros(32, dtype=np.float64)
    duplicate_indices = [
        (names.index(left), names.index(right)) for left, right in FEATURE_DUPLICATE_PAIRS
    ]
    duplicate_max_difference = np.zeros(len(duplicate_indices), dtype=np.float64)
    duplicate_same_missing = np.ones(len(duplicate_indices), dtype=bool)
    panels = [
        _open_memmap(dict(dict(pack["feature_channels"])[name]), dtype="float32")
        for name in ("daily_raw", "daily_state")
    ]
    try:
        for start in range(0, int(pack["date_count"]), 16):
            end = min(start + 16, int(pack["date_count"]))
            block = np.concatenate(
                [np.asarray(panel[start:end], dtype=np.float32) for panel in panels],
                axis=2,
            ).reshape(-1, 32)
            finite = np.isfinite(block)
            values = block.astype(np.float64, copy=False)
            counts += finite.sum(axis=0, dtype=np.int64)
            sums += np.where(finite, values, 0.0).sum(axis=0, dtype=np.float64)
            square_sums += np.where(finite, values * values, 0.0).sum(
                axis=0, dtype=np.float64
            )
            for pair_idx, (left_idx, right_idx) in enumerate(duplicate_indices):
                left_values = values[:, left_idx]
                right_values = values[:, right_idx]
                left_finite = finite[:, left_idx]
                right_finite = finite[:, right_idx]
                duplicate_same_missing[pair_idx] &= bool(
                    np.equal(left_finite, right_finite).all()
                )
                both = left_finite & right_finite
                if bool(both.any()):
                    duplicate_max_difference[pair_idx] = max(
                        duplicate_max_difference[pair_idx],
                        float(np.max(np.abs(left_values[both] - right_values[both]))),
                    )
        means = sums / np.maximum(counts, 1)
        variances = np.maximum(
            square_sums / np.maximum(counts, 1) - np.square(means), 0.0
        )
        total = int(pack["date_count"]) * int(pack["symbol_count"])
        statistics = [
            {
                "record_type": "feature_statistics",
                "feature": name,
                "finite_count": int(count),
                "coverage": float(count / max(total, 1)),
                "mean": float(mean),
                "std": float(math.sqrt(variance)),
            }
            for name, count, mean, variance in zip(
                names, counts, means, variances, strict=True
            )
        ]

        date_positions = np.unique(
            np.rint(np.linspace(0, int(pack["date_count"]) - 1, 256)).astype(np.int64)
        )
        symbol_positions = np.unique(
            np.rint(np.linspace(0, int(pack["symbol_count"]) - 1, 320)).astype(np.int64)
        )
        if int(date_positions.size * symbol_positions.size) != 81_920:
            raise AssertionError("feature-audit deterministic sample is not 81,920 rows")
        sample = np.concatenate(
            [
                np.asarray(panel[date_positions], dtype=np.float32)[:, symbol_positions]
                for panel in panels
            ],
            axis=2,
        ).reshape(-1, 32)
    finally:
        for panel in panels:
            _close_memmap(panel)
    frame = pd.DataFrame(sample, columns=names)
    correlation = frame.corr(min_periods=100).to_numpy(dtype=np.float64)
    filled = frame.copy()
    for column in names:
        values = pd.to_numeric(filled[column], errors="coerce")
        filled[column] = values.fillna(float(values.mean()))
    matrix = filled.to_numpy(dtype=np.float64)
    matrix -= matrix.mean(axis=0, keepdims=True)
    scale = matrix.std(axis=0, ddof=0, keepdims=True)
    matrix /= np.where(scale > 1.0e-12, scale, 1.0)
    eigenvalues = np.linalg.eigvalsh(np.corrcoef(matrix, rowvar=False))
    eigenvalues = np.clip(eigenvalues, 0.0, None)
    probabilities = eigenvalues / max(float(eigenvalues.sum()), 1.0e-12)
    positive = probabilities > 0.0
    effective_rank = float(
        math.exp(-float(np.sum(probabilities[positive] * np.log(probabilities[positive]))))
    )
    duplicate_rows: list[dict[str, Any]] = []
    for pair_idx, (left, right) in enumerate(FEATURE_DUPLICATE_PAIRS):
        left_values = frame[left].to_numpy(dtype=np.float64)
        right_values = frame[right].to_numpy(dtype=np.float64)
        max_difference = float(duplicate_max_difference[pair_idx])
        same_missing = bool(duplicate_same_missing[pair_idx])
        duplicate_rows.append(
            {
                "left": left,
                "right": right,
                "same_missing_pattern": same_missing,
                "maximum_absolute_difference": max_difference,
                "correlation": float(
                    frame[[left, right]].corr(min_periods=100).iloc[0, 1]
                ),
                "exact_duplicate_on_full_panel": bool(
                    same_missing and max_difference == 0.0
                ),
            }
        )
    pairwise_rows: list[dict[str, Any]] = []
    for left_idx, left in enumerate(names):
        for right_idx in range(left_idx + 1, len(names)):
            pairwise_rows.append(
                {
                    "left": left,
                    "right": names[right_idx],
                    "correlation": float(correlation[left_idx, right_idx]),
                }
            )
    audit = {
        "sample_contract": {
            "method": "256 evenly spaced dates x 320 evenly spaced symbols",
            "row_count": 81_920,
            "date_count": int(date_positions.size),
            "symbol_count": int(symbol_positions.size),
        },
        "effective_rank": effective_rank,
        "effective_rank_method": "entropy_effective_rank_of_sample_correlation_eigenvalues",
        "duplicate_pairs": duplicate_rows,
        "pairwise_correlations": pairwise_rows,
    }
    return statistics, audit


def _projection_sensitivity() -> list[dict[str, Any]]:
    pack = _read_json(BASE_PACK_MANIFEST)
    names = _base_feature_names(pack)
    rows: list[dict[str, Any]] = []
    for year in DEVELOPMENT_YEARS:
        checkpoint_path = _old_structured_run(year) / "best_model.pt"
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state = dict(payload["model_state_dict"])
        projection = state["proj.weight"].detach().float().numpy()
        layer_norm = np.abs(state["input_norm.weight"].detach().float().numpy())
        if projection.shape != (128, 32) or layer_norm.shape != (32,):
            raise ValueError(f"old Structured input projection shape drifted for {year}")
        sensitivity = np.linalg.norm(projection, axis=0) * layer_norm
        normalized = sensitivity / max(float(sensitivity.sum()), 1.0e-12)
        for feature, value, share in zip(names, sensitivity, normalized, strict=True):
            rows.append(
                {
                    "record_type": "checkpoint_projection_sensitivity",
                    "checkpoint_year": int(year),
                    "feature": feature,
                    "projection_norm_times_layernorm_scale": float(value),
                    "normalized_share": float(share),
                    "causal_interpretation": False,
                }
            )
    return rows


def _descriptor_input_names(
    dataset: training.SequencePathPackDataset,
) -> list[str]:
    names: list[str] = []
    for channel in dataset.channel_order:
        names.extend(
            f"{channel}:{column}" for column in dataset.feature_columns[channel]
        )
    names.extend(f"mask:{name}" for name in dataset.input_mask_features)
    if len(names) != int(dataset.input_dim):
        raise AssertionError("dataset input-name width mismatch")
    return names


def _stream_group_occlusion_fold(
    descriptor: Mapping[str, Any], *, year: int, study_path: Path
) -> list[dict[str, Any]]:
    run_dir = _descriptor_run_dir(descriptor, year=year, study_path=study_path)
    view_path = _descriptor_view_path(descriptor, year=year, study_path=study_path)
    variant = _variant_from_payload(dict(descriptor["variant"]))
    manifest = _read_json(view_path)
    dataset = training.SequencePathPackDataset(
        manifest,
        split="development",
        max_samples=0,
        input_channel_profile=variant.input_channel_profile,
        index_role="candidate",
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _summary = structured._load_checkpoint_model(run_dir, dataset, device=device)
    names = _descriptor_input_names(dataset)
    name_to_idx = {name: idx for idx, name in enumerate(names)}
    groups = {
        group: [name_to_idx[name] for name in columns]
        for group, columns in OCCLUSION_GROUP_COLUMNS.items()
    }
    version_names = ["baseline", *groups]
    group_indices = [[], *[groups[name] for name in groups]]
    accumulator: dict[str, dict[str, Any]] = {
        name: {
            "path_abs_sum": np.zeros(4, dtype=np.float64),
            "path_abs_count": np.zeros(4, dtype=np.int64),
            "rank_ic": [],
            "topk_alpha": {top_k: [] for top_k in TOP_K_VALUES},
            "topk_overlap": {top_k: [] for top_k in TOP_K_VALUES},
            "score_abs_drift_sum": 0.0,
            "score_abs_drift_count": 0,
        }
        for name in version_names
    }
    date_groups = dataset.sample_index.groupby("date_idx", sort=True).indices
    date_values = np.asarray(manifest["date_values"], dtype=object)
    model.eval()
    try:
        with torch.no_grad():
            for date_idx_raw, positions_raw in date_groups.items():
                if psutil.virtual_memory().available < int(MEMORY_GUARD_GIB * 1024**3):
                    raise MemoryError("feature audit available memory fell below 0.5 GiB")
                positions = np.asarray(positions_raw, dtype=np.int64)
                predicted_parts: dict[str, list[np.ndarray]] = {
                    name: [] for name in version_names
                }
                true_parts: list[np.ndarray] = []
                tradable_parts: list[np.ndarray] = []
                entry_filled_parts: list[np.ndarray] = []
                entry_open_parts: list[np.ndarray] = []
                exit_close_parts: list[np.ndarray] = []
                exit_sellable_parts: list[np.ndarray] = []
                symbols: list[str] = []
                for start in range(0, len(positions), 64):
                    chunk = positions[start : start + 64]
                    batch = dataset.get_batch(
                        chunk,
                        include_ohlcva_path=False,
                        include_richer_path=False,
                        include_summary=False,
                        include_activity_path=False,
                    )
                    x = batch["x"].to(device, non_blocking=device.type == "cuda")
                    symbol_idx = batch["symbol_idx"].to(
                        device, non_blocking=device.type == "cuda"
                    )
                    stacked = x.unsqueeze(0).repeat(len(version_names), 1, 1, 1)
                    for version_idx, indices in enumerate(group_indices):
                        if indices:
                            stacked[version_idx, :, :, indices] = 0.0
                    flat = stacked.reshape(
                        len(version_names) * int(x.shape[0]),
                        int(x.shape[1]),
                        int(x.shape[2]),
                    )
                    repeated_symbols = symbol_idx.repeat(len(version_names))
                    with torch.amp.autocast(
                        device_type=device.type, enabled=device.type == "cuda"
                    ):
                        output = model(flat, symbol_idx=repeated_symbols)
                    paths = (
                        output["future_path"]
                        .detach()
                        .float()
                        .cpu()
                        .numpy()
                        .reshape(
                            len(version_names), int(x.shape[0]), FORWARD_DAYS, 4
                        )
                    )
                    for version_idx, name in enumerate(version_names):
                        predicted_parts[name].append(paths[version_idx])
                    true_parts.append(batch["y_path"].numpy())
                    tradable_parts.append(batch["y_tradable_path"].numpy().astype(bool))
                    entry_filled_parts.append(batch["entry_filled"].numpy().astype(bool))
                    entry_open_parts.append(batch["entry_open_raw"].numpy())
                    exit_close_parts.append(batch["exit_close_raw_path"].numpy())
                    exit_sellable_parts.append(
                        batch["exit_sellable_path"].numpy().astype(bool)
                    )
                    symbols.extend(str(value) for value in batch["symbol"])
                    del batch, x, symbol_idx, stacked, flat, repeated_symbols, output, paths
                true_path = np.concatenate(true_parts, axis=0)
                tradable = np.concatenate(tradable_parts, axis=0)
                entry_filled = np.concatenate(entry_filled_parts, axis=0)
                entry_open = np.concatenate(entry_open_parts, axis=0)
                exit_close = np.concatenate(exit_close_parts, axis=0)
                exit_sellable = np.concatenate(exit_sellable_parts, axis=0)
                true_summary = training._derive_path_summary_numpy(
                    true_path,
                    price_anchor=dataset.price_anchor,
                    tradable_path=tradable,
                    earliest_exit_day=2,
                )
                true_score = true_summary[:, -1]
                outputs: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
                for name in version_names:
                    predicted = np.concatenate(predicted_parts[name], axis=0)
                    summary = training._derive_path_summary_numpy(
                        predicted,
                        price_anchor=dataset.price_anchor,
                        earliest_exit_day=2,
                    )
                    score = summary[:, -1]
                    planned_day = summary[:, 8]
                    outputs[name] = (predicted, score, planned_day)
                    finite = np.isfinite(true_path)
                    absolute = np.abs(predicted - true_path)
                    for field_idx in range(4):
                        field_mask = finite[:, :, field_idx]
                        accumulator[name]["path_abs_sum"][field_idx] += float(
                            absolute[:, :, field_idx][field_mask].sum()
                        )
                        accumulator[name]["path_abs_count"][field_idx] += int(
                            field_mask.sum()
                        )
                    accumulator[name]["rank_ic"].append(
                        float(
                            pd.Series(score).corr(
                                pd.Series(true_score), method="spearman"
                            )
                        )
                    )
                baseline_score = outputs["baseline"][1]
                baseline_order = np.argsort(-baseline_score, kind="mergesort")
                trade_date = str(dataset.trade_date_values[positions[0]])
                signal_indices = np.full(len(positions), int(date_idx_raw), dtype=np.int64)
                for name in version_names:
                    predicted, score, planned_day = outputs[name]
                    if name != "baseline":
                        drift = np.abs(score - baseline_score)
                        accumulator[name]["score_abs_drift_sum"] += float(drift.sum())
                        accumulator[name]["score_abs_drift_count"] += int(drift.size)
                    order = np.argsort(-score, kind="mergesort")
                    for top_k in TOP_K_VALUES:
                        overlap = len(
                            set(order[:top_k].tolist())
                            & set(baseline_order[:top_k].tolist())
                        ) / float(top_k)
                        accumulator[name]["topk_overlap"][top_k].append(float(overlap))
                    execution = evaluate_candidate_execution(
                        pd.DataFrame(
                            {
                                "trade_date": trade_date,
                                "symbol": symbols,
                                "score": score,
                                "predicted_exit_day": planned_day,
                                "entry_filled": entry_filled,
                                "entry_open_raw": entry_open,
                            }
                        ),
                        exit_close,
                        exit_sellable,
                        manifest=manifest,
                        signal_date_idx=signal_indices,
                        date_values=date_values,
                        top_k_values=TOP_K_VALUES,
                    )
                    for row in execution.daily_topk.itertuples(index=False):
                        accumulator[name]["topk_alpha"][int(row.top_k)].append(
                            float(row.alpha_net_realized_plan_return_base)
                        )
                del (
                    true_path,
                    tradable,
                    entry_filled,
                    entry_open,
                    exit_close,
                    exit_sellable,
                    true_summary,
                    true_score,
                    outputs,
                )
    finally:
        del model, dataset
        if device.type == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

    baseline = accumulator["baseline"]
    baseline_path_mae = float(
        np.sum(baseline["path_abs_sum"]) / max(np.sum(baseline["path_abs_count"]), 1)
    )
    baseline_close_mae = float(
        baseline["path_abs_sum"][3] / max(baseline["path_abs_count"][3], 1)
    )
    baseline_rank_ic = float(np.mean(baseline["rank_ic"]))
    baseline_alpha = {
        top_k: float(np.mean(baseline["topk_alpha"][top_k]))
        for top_k in TOP_K_VALUES
    }
    rows: list[dict[str, Any]] = []
    for name in version_names:
        item = accumulator[name]
        path_mae = float(
            np.sum(item["path_abs_sum"]) / max(np.sum(item["path_abs_count"]), 1)
        )
        close_mae = float(
            item["path_abs_sum"][3] / max(item["path_abs_count"][3], 1)
        )
        rank_ic = float(np.mean(item["rank_ic"]))
        row: dict[str, Any] = {
            "record_type": "group_occlusion",
            "development_year": int(year),
            "group": name,
            "occluded_features": list(
                OCCLUSION_GROUP_COLUMNS.get(name, ())
            ),
            "score_mean_absolute_drift": (
                float(item["score_abs_drift_sum"] / item["score_abs_drift_count"])
                if item["score_abs_drift_count"]
                else 0.0
            ),
            "rank_ic": rank_ic,
            "rank_ic_delta": rank_ic - baseline_rank_ic,
            "path_mae": path_mae,
            "path_mae_delta": path_mae - baseline_path_mae,
            "path_close_mae": close_mae,
            "path_close_mae_delta": close_mae - baseline_close_mae,
            "candidate_count": int(EXPECTED_CANDIDATE_ROWS[year]),
            "date_count": int(len(date_groups)),
        }
        for top_k in TOP_K_VALUES:
            alpha = float(np.mean(item["topk_alpha"][top_k]))
            row[f"top{top_k}_base_alpha"] = alpha
            row[f"top{top_k}_base_alpha_delta"] = alpha - baseline_alpha[top_k]
            row[f"top{top_k}_selection_overlap"] = float(
                np.mean(item["topk_overlap"][top_k])
            )
        rows.append(row)
    return rows


def build_feature_audit(
    final_incumbent: Mapping[str, Any], *, study_path: Path
) -> dict[str, Any]:
    output_json = study_path.parent / "feature_audit.json"
    output_csv = study_path.parent / "feature_audit.csv"
    if output_json.is_file() and output_csv.is_file():
        payload = _read_json(output_json)
        if dict(payload.get("final_incumbent", {}) or {}) != dict(final_incumbent):
            raise ValueError("stored feature audit belongs to a different final incumbent")
        return payload
    statistics, base_audit = _daily_feature_statistics()
    projection = _projection_sensitivity()
    occlusion: list[dict[str, Any]] = []
    for year in DEVELOPMENT_YEARS:
        occlusion.extend(
            _stream_group_occlusion_fold(
                final_incumbent, year=year, study_path=study_path
            )
        )
    aggregate: list[dict[str, Any]] = []
    numeric_keys = [
        key
        for key, value in occlusion[0].items()
        if isinstance(value, (int, float, np.integer, np.floating))
        and key not in {"development_year", "candidate_count", "date_count"}
    ]
    for group, frame in pd.DataFrame(occlusion).groupby("group", sort=False):
        aggregate.append(
            {
                "group": str(group),
                **{
                    key: float(pd.to_numeric(frame[key], errors="raise").mean())
                    for key in numeric_keys
                },
            }
        )
    payload = {
        "schema_version": 1,
        "artifact_type": "seq100_structured_input_feature_audit",
        "created_at": _now(),
        "final_incumbent": dict(final_incumbent),
        "base_32_feature_audit": {
            **base_audit,
            "feature_statistics": statistics,
        },
        "checkpoint_projection_sensitivity": projection,
        "checkpoint_projection_sensitivity_is_causal": False,
        "group_occlusion_yearly": occlusion,
        "group_occlusion_equal_year": aggregate,
        "feature_deletion_retraining_performed": False,
        "recommendation_scope": "diagnostic_only_separate_32_to_28_experiment_required",
    }
    csv_rows = [*statistics, *projection]
    for row in occlusion:
        csv_rows.append(
            {
                **row,
                "occluded_features": json.dumps(
                    row["occluded_features"], ensure_ascii=False
                ),
            }
        )
    _write_csv(output_csv, pd.DataFrame(csv_rows))
    _write_json(output_json, payload)
    return payload


def _descriptor_forecast_book(
    descriptor: Mapping[str, Any],
    *,
    study_path: Path,
    profile_name: str,
) -> tuple[Any, dict[str, str]]:
    from daily_research.path_policy import seq100_finite_capital_backtest as finite

    book = finite.ForecastBook(profile_name)
    source_hashes: dict[str, str] = {}
    variant = _variant_from_payload(dict(descriptor["variant"]))
    for year in DEVELOPMENT_YEARS:
        view_path = _descriptor_view_path(
            descriptor, year=year, study_path=study_path
        )
        manifest = _read_json(view_path)
        dataset = training.SequencePathPackDataset(
            manifest,
            split="development",
            max_samples=0,
            input_channel_profile=variant.input_channel_profile,
            index_role="candidate",
        )
        run_dir = _descriptor_run_dir(
            descriptor, year=year, study_path=study_path
        )
        prediction_path = run_dir / "predictions/development_predictions.csv"
        if not prediction_path.is_file():
            raise FileNotFoundError(prediction_path)
        source_hashes[str(year)] = _file_sha256(prediction_path)
        frame = pd.read_csv(
            prediction_path,
            usecols=["trade_date", "symbol", "score", "predicted_exit_day"],
            dtype={"trade_date": str, "symbol": str, "score": np.float64},
        )
        candidates = dataset.sample_index
        if len(frame) != len(candidates):
            raise ValueError(f"forecast row count drifted for {profile_name}/{year}")
        if not np.array_equal(
            frame["trade_date"].astype(str).to_numpy(),
            candidates["trade_date"].astype(str).to_numpy(),
        ) or not np.array_equal(
            frame["symbol"].astype(str).to_numpy(),
            candidates["symbol"].astype(str).to_numpy(),
        ):
            raise ValueError(f"forecast candidate order drifted for {profile_name}/{year}")
        scores = pd.to_numeric(frame["score"], errors="coerce").to_numpy(
            dtype=np.float64
        )
        planned = pd.to_numeric(
            frame["predicted_exit_day"], errors="coerce"
        ).to_numpy(dtype=np.float64)
        if not bool(np.isfinite(scores).all()) or not bool(np.isfinite(planned).all()):
            raise ValueError(f"forecast coverage is incomplete for {profile_name}/{year}")
        planned = np.clip(np.rint(planned), 2, FORWARD_DAYS).astype(np.int16)
        symbol_indices = candidates["symbol_idx"].to_numpy(dtype=np.int64)
        for date_idx_raw, positions_raw in candidates.groupby(
            "date_idx", sort=True
        ).indices.items():
            positions = np.asarray(positions_raw, dtype=np.int64)
            book.add_day(
                date_idx=int(date_idx_raw),
                symbol_idx=symbol_indices[positions],
                score=scores[positions],
                planned_day=planned[positions],
            )
        del dataset, frame, candidates, scores, planned, symbol_indices
        gc.collect()
    return book, source_hashes


def _hybrid_forecast_book(
    ranking_book: Any, exit_book: Any, *, profile_name: str
) -> Any:
    from daily_research.path_policy import seq100_finite_capital_backtest as finite

    if ranking_book.signal_date_indices != exit_book.signal_date_indices:
        raise ValueError("hybrid ranking and exit signal-date coverage differs")
    hybrid = finite.ForecastBook(profile_name)
    for date_idx in ranking_book.signal_date_indices:
        ranking_day = ranking_book.days[int(date_idx)]
        exit_day = exit_book.days[int(date_idx)]
        if not np.array_equal(ranking_day.symbol_idx, exit_day.symbol_idx):
            raise ValueError(f"hybrid candidate symbols differ on date_idx={date_idx}")
        hybrid.days[int(date_idx)] = finite.ForecastDay(
            symbol_idx=np.asarray(exit_day.symbol_idx, dtype=np.int32).copy(),
            score=np.asarray(exit_day.score, dtype=np.float64).copy(),
            planned_day=np.asarray(exit_day.planned_day, dtype=np.int16).copy(),
            top3_symbol_idx=tuple(int(value) for value in ranking_day.top3_symbol_idx),
        )
    return hybrid


def _cohort_outcome_row(
    *,
    audit_pack: Any,
    finite: Any,
    exit_audit: Any,
    date_idx: int,
    symbol_indices: np.ndarray,
    planned_days: np.ndarray,
    cost_scenario: str,
) -> dict[str, Any]:
    entry, exit_close, exit_sellable = audit_pack.execution_paths(
        int(date_idx), symbol_indices
    )
    entry_filled = np.asarray(
        audit_pack.entry_filled[int(date_idx) + 1, symbol_indices], dtype=bool
    )
    lookup = exit_audit._next_valid_exit_indices(exit_close, exit_sellable)
    plan = exit_audit.resolve_planned_exit_batch(
        signal_date_idx=int(date_idx),
        entry_filled=entry_filled,
        entry_prices=entry,
        exit_prices=exit_close,
        next_valid_exit_idx=lookup,
        planned_days=planned_days,
        forward_days=audit_pack.forward_days,
        execution_days=audit_pack.execution_days,
        terminal_recovery_fraction=audit_pack.terminal_recovery_fraction,
    )
    multiplier = (
        1.0
        if str(cost_scenario) == "base"
        else float(audit_pack.contract.stress_slippage_multiplier)
    )
    cash = exit_audit.cashflow_batch(
        allocated_cash=finite.STARTING_CASH_CNY,
        entry_filled=entry_filled,
        entry_prices=entry,
        plan=plan,
        date_values=audit_pack.date_values,
        contract=audit_pack.contract,
        slippage_multiplier=multiplier,
    )
    _oracle_plan, oracle_cash = exit_audit.oracle_executable_outcome_batch(
        signal_date_idx=int(date_idx),
        allocated_cash=finite.STARTING_CASH_CNY,
        entry_filled=entry_filled,
        entry_prices=entry,
        exit_prices=exit_close,
        next_valid_exit_idx=lookup,
        date_values=audit_pack.date_values,
        contract=audit_pack.contract,
        slippage_multiplier=multiplier,
        forward_days=audit_pack.forward_days,
        execution_days=audit_pack.execution_days,
        terminal_recovery_fraction=audit_pack.terminal_recovery_fraction,
    )
    valid_exit = plan.exit_day >= 0
    resolved = (
        float(np.mean(plan.exit_day[valid_exit]))
        if bool(valid_exit.any())
        else float("nan")
    )
    selected_return = float(np.mean(cash.net_return))
    return {
        "selected_count": int(len(symbol_indices)),
        "selected_return": selected_return,
        "oracle_return": float(np.mean(oracle_cash.net_return)),
        "exit_regret": float(np.mean(oracle_cash.net_return - cash.net_return)),
        "entry_fill_rate": float(np.mean(cash.order_filled.astype(np.float64))),
        "terminal_recovery_rate": float(
            np.mean(plan.terminal_recovery.astype(np.float64))
        ),
        "exit_deferral_rate": float(
            np.mean(
                (
                    (plan.exit_day > plan.planned_day)
                    & (~plan.terminal_recovery)
                    & cash.order_filled
                ).astype(np.float64)
            )
        ),
        "mean_planned_exit_day": float(np.mean(plan.planned_day)),
        "mean_resolved_exit_day": resolved,
        "capital_release_speed": float(1.0 / resolved) if resolved > 0.0 else None,
        "return_per_resolved_holding_day": (
            float(selected_return / resolved) if resolved > 0.0 else None
        ),
    }


def _stage1_cohort_review(
    *,
    audit_pack: Any,
    finite: Any,
    exit_audit: Any,
    book_100: Any,
    book_180: Any,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    if book_100.signal_date_indices != book_180.signal_date_indices:
        raise ValueError("Stage 1 cohort signal-date coverage differs")
    for date_idx in book_100.signal_date_indices:
        trade_date = str(audit_pack.date_values[int(date_idx)])
        year = int(trade_date[:4])
        day_100 = book_100.days[int(date_idx)]
        day_180 = book_180.days[int(date_idx)]
        selected_100 = np.asarray(day_100.top3_symbol_idx, dtype=np.int64)
        selected_180 = np.asarray(day_180.top3_symbol_idx, dtype=np.int64)
        for ranking_name, selected in (
            ("100x32", selected_100),
            ("180x32", selected_180),
        ):
            for fixed_day in CAPITAL_REVIEW_FIXED_DAYS:
                for cost_scenario in CAPITAL_REVIEW_COST_SCENARIOS:
                    outcome = _cohort_outcome_row(
                        audit_pack=audit_pack,
                        finite=finite,
                        exit_audit=exit_audit,
                        date_idx=int(date_idx),
                        symbol_indices=selected,
                        planned_days=np.full(
                            len(selected), int(fixed_day), dtype=np.int16
                        ),
                        cost_scenario=cost_scenario,
                    )
                    rows.append(
                        {
                            "comparison_layer": "fixed_exit_ranking",
                            "configuration": f"rank_{ranking_name}",
                            "ranking_source": ranking_name,
                            "exit_source": f"fixed_d{int(fixed_day)}",
                            "policy": f"fixed_d{int(fixed_day)}",
                            "cost_scenario": cost_scenario,
                            "year": year,
                            "trade_date": trade_date,
                            **outcome,
                        }
                    )
        for exit_name, exit_day in (
            ("100x32", day_100),
            ("180x32", day_180),
        ):
            planned = np.asarray(
                [
                    exit_day.planned_day[
                        int(np.searchsorted(exit_day.symbol_idx, symbol_idx))
                    ]
                    for symbol_idx in selected_100
                ],
                dtype=np.int16,
            )
            for cost_scenario in CAPITAL_REVIEW_COST_SCENARIOS:
                outcome = _cohort_outcome_row(
                    audit_pack=audit_pack,
                    finite=finite,
                    exit_audit=exit_audit,
                    date_idx=int(date_idx),
                    symbol_indices=selected_100,
                    planned_days=planned,
                    cost_scenario=cost_scenario,
                )
                rows.append(
                    {
                        "comparison_layer": "fixed_selection_exit",
                        "configuration": f"rank_100x32_exit_{exit_name}",
                        "ranking_source": "100x32",
                        "exit_source": exit_name,
                        "policy": "model_plan",
                        "cost_scenario": cost_scenario,
                        "year": year,
                        "trade_date": trade_date,
                        **outcome,
                    }
                )
    daily = pd.DataFrame(rows)
    group = [
        "comparison_layer",
        "configuration",
        "ranking_source",
        "exit_source",
        "policy",
        "cost_scenario",
        "year",
    ]
    numeric = [
        "selected_return",
        "oracle_return",
        "exit_regret",
        "entry_fill_rate",
        "terminal_recovery_rate",
        "exit_deferral_rate",
        "mean_planned_exit_day",
        "mean_resolved_exit_day",
        "capital_release_speed",
        "return_per_resolved_holding_day",
    ]
    yearly = daily.groupby(group, as_index=False, sort=True)[numeric].mean()
    counts = (
        daily.groupby(group, as_index=False, sort=True)
        .size()
        .rename(columns={"size": "date_count"})
    )
    yearly = yearly.merge(counts, on=group, how="left", validate="one_to_one")
    equal_group = group[:-1]
    equal_year = yearly.groupby(equal_group, as_index=False, sort=True)[
        [*numeric, "date_count"]
    ].mean()
    equal_year["year"] = "equal_year"
    return daily, pd.concat([yearly, equal_year], ignore_index=True, sort=False)


def _stage1_account_review(
    *,
    market: Any,
    finite: Any,
    books: Mapping[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    jobs: list[tuple[str, str, Any]] = []
    for name in ("rank100_exit100", "rank180_exit180"):
        for fixed_day in CAPITAL_REVIEW_FIXED_DAYS:
            jobs.append(
                (
                    "fixed_exit_ranking",
                    name,
                    finite.PolicySpec(
                        name=f"fixed_d{int(fixed_day)}",
                        kind="fixed",
                        fixed_day=int(fixed_day),
                    ),
                )
            )
    for name in ("rank100_exit100", "rank180_exit180", "rank100_exit180"):
        jobs.extend(
            [
                (
                    "full_model_path_exit",
                    name,
                    finite.PolicySpec(name="model_plan", kind="model_plan"),
                ),
                (
                    "full_model_path_exit",
                    name,
                    finite.PolicySpec(name="rolling_path", kind="rolling"),
                ),
            ]
        )
    signal_dates = tuple(books["rank100_exit100"].signal_date_indices)
    if not signal_dates or any(
        tuple(book.signal_date_indices) != signal_dates for book in books.values()
    ):
        raise ValueError("Stage 1 account signal-date coverage differs")
    guard = finite._MemoryGuard()
    metric_rows: list[dict[str, Any]] = []
    annual_rows: list[dict[str, Any]] = []
    for layer, configuration, policy in jobs:
        for slots in CAPITAL_REVIEW_SLOTS:
            for cost_scenario in CAPITAL_REVIEW_COST_SCENARIOS:
                metric, _equity, trades, annual = finite.simulate_portfolio(
                    market=market,
                    book=books[configuration],
                    raw_top3_paths={},
                    policy=policy,
                    slots=int(slots),
                    cost_scenario=cost_scenario,
                    first_signal_date_idx=int(signal_dates[0]),
                    last_signal_date_idx=int(signal_dates[-1]),
                    starting_cash=finite.STARTING_CASH_CNY,
                    memory_guard=guard,
                )
                occupied_capital_days = (
                    float(
                        np.sum(
                            trades["buy_cash_cny"].to_numpy(dtype=np.float64)
                            * trades["occupied_sessions"].to_numpy(dtype=np.float64)
                        )
                    )
                    if not trades.empty
                    else 0.0
                )
                pnl = (
                    float(trades["net_pnl_cny"].sum()) if not trades.empty else 0.0
                )
                metric_rows.append(
                    {
                        "comparison_layer": layer,
                        "configuration": configuration,
                        **metric,
                        "worst_calendar_year_return": (
                            float(min(row["net_return"] for row in annual))
                            if annual
                            else None
                        ),
                        "net_pnl_per_deployed_capital_session": (
                            float(pnl / occupied_capital_days)
                            if occupied_capital_days > 0.0
                            else None
                        ),
                    }
                )
                annual_rows.extend(
                    {
                        "comparison_layer": layer,
                        "configuration": configuration,
                        "policy": str(policy.name),
                        "slot_count": int(slots),
                        "cost_scenario": cost_scenario,
                        **row,
                    }
                    for row in annual
                )
                del _equity, trades
    return pd.DataFrame(metric_rows), pd.DataFrame(annual_rows)


def _dynamic_account_review(
    *,
    market: Any,
    finite: Any,
    books: Mapping[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    signal_dates = tuple(next(iter(books.values())).signal_date_indices)
    if not signal_dates or any(
        tuple(book.signal_date_indices) != signal_dates for book in books.values()
    ):
        raise ValueError("capital-review signal-date coverage differs")
    guard = finite._MemoryGuard()
    metric_rows: list[dict[str, Any]] = []
    annual_rows: list[dict[str, Any]] = []
    policies = (
        finite.PolicySpec(name="model_plan", kind="model_plan"),
        finite.PolicySpec(name="rolling_path", kind="rolling"),
    )
    for configuration, book in books.items():
        for policy in policies:
            for slots in CAPITAL_REVIEW_SLOTS:
                for cost_scenario in CAPITAL_REVIEW_COST_SCENARIOS:
                    metric, _equity, trades, annual = finite.simulate_portfolio(
                        market=market,
                        book=book,
                        raw_top3_paths={},
                        policy=policy,
                        slots=int(slots),
                        cost_scenario=cost_scenario,
                        first_signal_date_idx=int(signal_dates[0]),
                        last_signal_date_idx=int(signal_dates[-1]),
                        starting_cash=finite.STARTING_CASH_CNY,
                        memory_guard=guard,
                    )
                    occupied_capital_days = (
                        float(
                            np.sum(
                                trades["buy_cash_cny"].to_numpy(dtype=np.float64)
                                * trades["occupied_sessions"].to_numpy(dtype=np.float64)
                            )
                        )
                        if not trades.empty
                        else 0.0
                    )
                    pnl = (
                        float(trades["net_pnl_cny"].sum())
                        if not trades.empty
                        else 0.0
                    )
                    metric_rows.append(
                        {
                            "comparison_layer": "full_model_path_exit",
                            "configuration": str(configuration),
                            **metric,
                            "worst_calendar_year_return": (
                                float(min(row["net_return"] for row in annual))
                                if annual
                                else None
                            ),
                            "net_pnl_per_deployed_capital_session": (
                                float(pnl / occupied_capital_days)
                                if occupied_capital_days > 0.0
                                else None
                            ),
                        }
                    )
                    annual_rows.extend(
                        {
                            "comparison_layer": "full_model_path_exit",
                            "configuration": str(configuration),
                            "policy": str(policy.name),
                            "slot_count": int(slots),
                            "cost_scenario": cost_scenario,
                            **row,
                        }
                        for row in annual
                    )
                    del _equity, trades
    return pd.DataFrame(metric_rows), pd.DataFrame(annual_rows)


def _account_pairwise_against_reference(
    account: pd.DataFrame, *, reference: str
) -> pd.DataFrame:
    keys = ["comparison_layer", "policy", "slot_count", "cost_scenario"]
    metrics = [
        "signal_period_total_return",
        "signal_period_cagr_trading_days",
        "signal_period_maximum_drawdown",
        "worst_calendar_year_return",
        "mean_signal_capital_utilization",
        "skipped_no_slot_signal_count",
        "closed_trade_count",
        "mean_occupied_sessions",
        "net_pnl_per_deployed_capital_session",
    ]
    reference_frame = account[account["configuration"].eq(reference)]
    if reference_frame.empty:
        raise ValueError(f"capital-review reference is missing: {reference}")
    rows: list[dict[str, Any]] = []
    for challenger in sorted(
        set(account["configuration"].astype(str)).difference({str(reference)})
    ):
        challenger_frame = account[account["configuration"].eq(challenger)]
        merged = challenger_frame.merge(
            reference_frame,
            on=keys,
            how="inner",
            suffixes=("_challenger", "_reference"),
            validate="one_to_one",
        )
        if len(merged) != len(reference_frame):
            raise ValueError(
                f"capital-review comparison coverage differs: {challenger}"
            )
        for row in merged.to_dict(orient="records"):
            payload = {key: row[key] for key in keys}
            payload.update({"challenger": challenger, "reference": reference})
            for metric in metrics:
                payload[f"{metric}_challenger"] = row[f"{metric}_challenger"]
                payload[f"{metric}_reference"] = row[f"{metric}_reference"]
                payload[f"{metric}_delta"] = (
                    float(row[f"{metric}_challenger"])
                    - float(row[f"{metric}_reference"])
                )
            rows.append(payload)
    return pd.DataFrame(rows)


def _stage1_account_pairwise(account: pd.DataFrame) -> pd.DataFrame:
    comparisons = (
        ("rank180_exit180", "rank100_exit100"),
        ("rank100_exit180", "rank100_exit100"),
        ("rank100_exit180", "rank180_exit180"),
    )
    rows: list[dict[str, Any]] = []
    keys = ["comparison_layer", "policy", "slot_count", "cost_scenario"]
    metrics = [
        "signal_period_total_return",
        "signal_period_cagr_trading_days",
        "signal_period_maximum_drawdown",
        "worst_calendar_year_return",
        "mean_signal_capital_utilization",
        "skipped_no_slot_signal_count",
        "closed_trade_count",
        "mean_occupied_sessions",
        "net_pnl_per_deployed_capital_session",
    ]
    for challenger, reference in comparisons:
        left = account[account["configuration"].eq(challenger)]
        right = account[account["configuration"].eq(reference)]
        merged = left.merge(
            right,
            on=keys,
            how="inner",
            suffixes=("_challenger", "_reference"),
            validate="one_to_one",
        )
        for row in merged.to_dict(orient="records"):
            payload = {
                key: row[key] for key in keys
            }
            payload.update(
                {
                    "challenger": challenger,
                    "reference": reference,
                }
            )
            for metric in metrics:
                payload[f"{metric}_challenger"] = row[f"{metric}_challenger"]
                payload[f"{metric}_reference"] = row[f"{metric}_reference"]
                payload[f"{metric}_delta"] = (
                    float(row[f"{metric}_challenger"])
                    - float(row[f"{metric}_reference"])
                )
            rows.append(payload)
    return pd.DataFrame(rows)


def _primary_account_selection(
    account: pd.DataFrame, *, expected_configurations: Sequence[str]
) -> dict[str, Any]:
    primary = account[
        account["comparison_layer"].eq("full_model_path_exit")
        & account["policy"].eq("rolling_path")
        & account["cost_scenario"].eq("base")
        & account["slot_count"].isin(CAPITAL_REVIEW_SLOTS)
    ].copy()
    expected = {str(value) for value in expected_configurations}
    if set(primary["configuration"].astype(str)) != expected:
        raise ValueError("primary capital selection is missing a configuration")
    horizon_values = sorted(
        set(primary["equity_path_session_count"].astype(int).tolist())
    )
    if len(horizon_values) != 1 or int(horizon_values[0]) <= 0:
        raise ValueError(
            "primary capital selection requires one common positive calendar horizon"
        )
    common_horizon_sessions = int(horizon_values[0])
    rows: list[dict[str, Any]] = []
    for configuration, frame in primary.groupby("configuration", sort=True):
        frame = frame.sort_values("slot_count", kind="mergesort")
        if frame["slot_count"].astype(int).tolist() != list(CAPITAL_REVIEW_SLOTS):
            raise ValueError(
                f"primary capital selection has incomplete slots: {configuration}"
            )
        wealth = (
            frame["liquidated_ending_equity_cny"].to_numpy(dtype=np.float64)
            / frame["starting_cash_cny"].to_numpy(dtype=np.float64)
        )
        if not bool(np.isfinite(wealth).all()) or bool(np.any(wealth <= 0.0)):
            raise ValueError("primary capital selection contains invalid terminal wealth")
        annualized_log_growth = np.log(wealth) * (
            252.0 / float(common_horizon_sessions)
        )
        rows.append(
            {
                "configuration": str(configuration),
                "equal_slot_mean_log_terminal_wealth": float(np.log(wealth).mean()),
                "equal_slot_mean_annualized_log_growth": float(
                    annualized_log_growth.mean()
                ),
                "equal_slot_geometric_mean_terminal_wealth_multiple": float(
                    np.exp(np.log(wealth).mean())
                ),
                "equal_slot_mean_cagr": float(
                    frame["signal_period_cagr_trading_days"].astype(float).mean()
                ),
                "equal_slot_mean_total_return": float(
                    frame["signal_period_total_return"].astype(float).mean()
                ),
                "slot_win_material": {
                    str(int(row["slot_count"])): {
                        "liquidated_ending_equity_cny": float(
                            row["liquidated_ending_equity_cny"]
                        ),
                        "signal_period_cagr": float(
                            row["signal_period_cagr_trading_days"]
                        ),
                    }
                    for row in frame.to_dict(orient="records")
                },
            }
        )
    ordered = sorted(
        rows,
        key=lambda row: (
            -float(row["equal_slot_mean_annualized_log_growth"]),
            str(row["configuration"]),
        ),
    )
    selected_by_slot: dict[str, str] = {}
    for slots, frame in primary.groupby("slot_count", sort=True):
        ranked = frame.sort_values(
            ["liquidated_ending_equity_cny", "configuration"],
            ascending=[False, True],
            kind="mergesort",
        )
        selected_by_slot[str(int(slots))] = str(ranked.iloc[0]["configuration"])
    return {
        "objective": (
            "maximize equal-slot mean annualized log growth of liquidated "
            "terminal wealth over one common calendar horizon under rolling_path "
            "and base costs"
        ),
        "objective_is_unit_time_profit": True,
        "common_calendar_horizon_sessions": common_horizon_sessions,
        "policy": "rolling_path",
        "cost_scenario": "base",
        "slot_counts": list(CAPITAL_REVIEW_SLOTS),
        "selected_configuration": str(ordered[0]["configuration"]),
        "selected_configuration_by_slot": selected_by_slot,
        "ranking": ordered,
        "risk_metrics_used_as_selection_gate": False,
        "cohort_metrics_used_as_selection_gate": False,
    }


def _stage1_primary_account_selection(account: pd.DataFrame) -> dict[str, Any]:
    return _primary_account_selection(
        account,
        expected_configurations=(
            "rank100_exit100",
            "rank180_exit180",
            "rank100_exit180",
        ),
    )


def _migrate_capital_review_summary(
    summary: Mapping[str, Any], *, study_root: Path, stage: int
) -> dict[str, Any]:
    migrated = dict(summary)
    outputs = dict(migrated["outputs"])
    account_path = Path(str(outputs["account_metrics_csv"])).resolve()
    account = pd.read_csv(account_path)
    if int(stage) == 1:
        primary = _stage1_primary_account_selection(account)
    else:
        compared = tuple(dict(migrated["compared_strategies"]))
        primary = _primary_account_selection(
            account, expected_configurations=compared
        )
        selected_name = str(primary["selected_configuration"])
        selected_components = dict(
            dict(migrated["compared_strategies"])[selected_name]
        )
        selected_rank = dict(selected_components["ranking_descriptor"])
        selected_exit = dict(selected_components["exit_descriptor"])
        migrated["selected_strategy"] = {
            "configuration": selected_name,
            "ranking_descriptor": selected_rank,
            "exit_descriptor": selected_exit,
            "hybrid": bool(
                _canonical_digest(selected_rank)
                != _canonical_digest(selected_exit)
            ),
        }
    migrated["primary_account_selection"] = primary

    decisions = _load_stage_decisions(study_root)
    decision = _decision_for_stage(decisions, int(stage))
    if decision is None:
        raise ValueError(f"Stage {stage} diagnostics are missing")
    diagnostics = dict(decision["cohort_diagnostics"])
    migrated.pop("cohort_gate_result", None)
    migrated.pop("cohort_reference", None)
    migrated["cohort_diagnostics"] = {
        "scope": "ranking_path_and_exit_behavior_only",
        "selection_authority": False,
        "observations": dict(diagnostics["registered_observations"]),
    }
    migrated["selection_semantics"] = {
        "primary": "continuous_account_unit_time_profit",
        "common_calendar_horizon_sessions": int(
            primary["common_calendar_horizon_sessions"]
        ),
        "cohort_metrics": "diagnostic_only",
        "risk_and_stress": "reported_not_selection_gate",
    }
    active_amendment = _ensure_profit_selection_amendment(
        study_root.resolve() / "study.json"
    )
    migrated["active_selection_amendment"] = {
        "path": active_amendment["path"],
        "sha256": active_amendment["sha256"],
    }
    return migrated


def _ensure_capital_selection_amendment(study_path: Path) -> dict[str, Any]:
    path = study_path.parent / "protocol_amendment_capital_selection_v2.json"
    payload = {
        "schema_version": 1,
        "artifact_type": "seq100_structured_input_ablation_protocol_amendment",
        "amendment_id": "capital-selection-v2",
        "study_id": STUDY_ID,
        "original_study_contract_sha256": str(
            _read_json(study_path)["contract_sha256"]
        ),
        "authorization_timing": (
            "user-authorized after Stage 1 cohort metrics and before any Stage 1 "
            "finite-capital comparison was run"
        ),
        "reason": (
            "daily independent-cohort alpha does not internalize holding time, "
            "idle cash, slot pressure, or skipped signals"
        ),
        "changes": {
            "cohort_gate_role": "reference_only",
            "primary_objective": (
                "maximize equal-slot mean log liquidated terminal wealth under "
                "rolling_path and base costs"
            ),
            "slot_counts": list(CAPITAL_REVIEW_SLOTS),
            "compared_configurations": [
                "rank100_exit100",
                "rank180_exit180",
                "rank100_exit180",
            ],
            "stress_and_risk_metrics_are_selection_gates": False,
            "stage2_blocked_until_review": True,
        },
        "protected_material_unchanged": {
            "qdp": True,
            "base_pack": True,
            "candidate_keys": True,
            "labels": True,
            "execution_material": True,
            "checkpoints": True,
        },
    }
    if path.is_file():
        observed = _read_json(path)
        if observed != payload:
            raise ValueError("capital-selection protocol amendment drifted")
    else:
        _write_json(path, payload)
    return {
        "path": str(path.resolve()),
        "sha256": _file_sha256(path),
        "payload": payload,
    }


def _ensure_all_stage_capital_amendment(study_path: Path) -> dict[str, Any]:
    path = study_path.parent / "protocol_amendment_all_stage_capital_selection_v3.json"
    payload = {
        "schema_version": 1,
        "artifact_type": "seq100_structured_input_ablation_protocol_amendment",
        "amendment_id": "all-stage-capital-selection-v3",
        "study_id": STUDY_ID,
        "supersedes_for_stage_selection": "capital-selection-v2",
        "original_study_contract_sha256": str(
            _read_json(study_path)["contract_sha256"]
        ),
        "authorization_timing": (
            "user-authorized after the Stage 1 capital review and before any "
            "complete three-fold Stage 2 challenger evaluation existed"
        ),
        "reason": (
            "the user's objective is continuous-account profit; cohort metrics "
            "cannot promote or reject any input challenger"
        ),
        "selection_rule": {
            "applies_to_stages": [2, 3],
            "primary_objective": (
                "maximize equal-slot mean log liquidated terminal wealth under "
                "rolling_path and base costs"
            ),
            "slot_counts": list(CAPITAL_REVIEW_SLOTS),
            "candidate_strategies": [
                "incumbent_strategy",
                "challenger_own",
                "incumbent_rank_challenger_exit",
                "challenger_rank_incumbent_exit",
            ],
            "cohort_metrics_role": "reference_only",
            "stress_and_risk_metrics_role": "reported_not_gate",
        },
        "protected_material_unchanged": {
            "qdp": True,
            "base_pack": True,
            "candidate_keys": True,
            "labels": True,
            "execution_material": True,
            "completed_checkpoints": True,
        },
    }
    if path.is_file():
        if _read_json(path) != payload:
            raise ValueError("all-stage capital-selection amendment drifted")
    else:
        _write_json(path, payload)
    return {
        "path": str(path.resolve()),
        "sha256": _file_sha256(path),
        "payload": payload,
    }


def _ensure_profit_selection_amendment(study_path: Path) -> dict[str, Any]:
    path = study_path.parent / "protocol_amendment_profit_selection_v4.json"
    payload = {
        "schema_version": 1,
        "artifact_type": "seq100_structured_input_ablation_protocol_amendment",
        "amendment_id": "profit-selection-v4",
        "study_id": STUDY_ID,
        "supersedes_for_all_stage_selection": [
            "original study contract selection_rule",
            "capital-selection-v2",
            "all-stage-capital-selection-v3",
        ],
        "authorization_timing": (
            "user-authorized while the third Stage 2 fold was training and "
            "before the Stage 2 continuous-account review completed"
        ),
        "reason": (
            "the research objective is maximum profit per unit of calendar time; "
            "daily independent-cohort return cannot price holding time, idle "
            "cash, slot pressure, or skipped signals"
        ),
        "selection_rule": {
            "applies_to_stages": [1, 2, 3],
            "primary_objective": (
                "maximize equal-slot mean annualized log growth of liquidated "
                "terminal wealth over the identical development calendar under "
                "rolling_path and base costs"
            ),
            "slot_counts": list(CAPITAL_REVIEW_SLOTS),
            "candidate_strategies_include_ranking_exit_hybrids": True,
            "cohort_metrics_role": "diagnostic_only",
            "cohort_pass_fail_or_preference_is_defined": False,
            "risk_and_stress_metrics_role": "reported_not_selection_gate",
        },
        "equivalence_note": (
            "All configurations share one calendar horizon, so annualized log "
            "growth and log terminal wealth induce the same ordering."
        ),
        "protected_material_unchanged": {
            "qdp": True,
            "base_pack": True,
            "candidate_keys": True,
            "labels": True,
            "execution_material": True,
            "completed_checkpoints": True,
        },
    }
    if path.is_file():
        if _read_json(path) != payload:
            raise ValueError("profit-selection protocol amendment drifted")
    else:
        _write_json(path, payload)
    return {
        "path": str(path.resolve()),
        "sha256": _file_sha256(path),
        "payload": payload,
    }


def _update_stage1_capital_review_decision(
    *, study_root: Path, summary_path: Path, summary: Mapping[str, Any]
) -> None:
    decisions = _load_stage_decisions(study_root)
    stages = list(decisions.get("stages", []) or [])
    for index, raw in enumerate(stages):
        if int(raw["stage"]) != 1:
            continue
        row = dict(raw)
        row["decision_scope"] = "continuous_account_profit_primary"
        selection = dict(summary["primary_account_selection"])
        selected_configuration = str(selection["selected_configuration"])
        descriptor_100 = dict(row["incumbent"])
        descriptor_180 = dict(row["challenger"])
        if selected_configuration == "rank100_exit100":
            ranking_descriptor = descriptor_100
            exit_descriptor = descriptor_100
        elif selected_configuration == "rank180_exit180":
            ranking_descriptor = descriptor_180
            exit_descriptor = descriptor_180
        elif selected_configuration == "rank100_exit180":
            ranking_descriptor = descriptor_100
            exit_descriptor = descriptor_180
        else:
            raise ValueError(
                f"unknown Stage 1 capital configuration: {selected_configuration}"
            )
        selected_strategy = {
            "configuration": selected_configuration,
            "ranking_descriptor": ranking_descriptor,
            "exit_descriptor": exit_descriptor,
            "hybrid": bool(ranking_descriptor != exit_descriptor),
        }
        row["capital_efficiency_review"] = {
            "status": "completed",
            "summary_json": str(summary_path.resolve()),
            "summary_sha256": _file_sha256(summary_path),
            "comparison_status": str(summary["comparison_status"]),
            "selection_objective": str(selection["objective"]),
            "selected_strategy": selected_strategy,
            "training_branch_descriptor": ranking_descriptor,
            "capital_baseline_selection_performed": True,
        }
        row["interpretation"] = (
            "The Stage 1 strategy is selected only by continuous-account "
            "liquidated terminal wealth over the common calendar window. "
            "Cohort metrics are diagnostic."
        )
        stages[index] = row
        decisions["stages"] = stages
        decisions["updated_at"] = _now()
        _write_json(study_root / "stage_decisions.json", decisions)
        return
    raise ValueError("Stage 1 decision is missing")


def run_stage1_capital_review(
    *, study_root: Path = STUDY_ROOT
) -> dict[str, Any]:
    from daily_research.path_policy import seq100_finite_capital_backtest as finite
    from daily_research.path_policy import seq100_exit_policy_audit as exit_audit
    from daily_research.path_policy.seq100_exit_policy_audit import (
        CandidateCompleteAuditPack,
    )

    verify_prepared_study(study_root=study_root, deep=False)
    study_path = study_root.resolve() / "study.json"
    amendment = _ensure_capital_selection_amendment(study_path)
    active_amendment = _ensure_profit_selection_amendment(study_path)
    output_root = study_path.parent / "stage1_capital_review"
    summary_path = output_root / "summary.json"
    if summary_path.is_file():
        summary = _read_json(summary_path)
        for name, value in dict(summary["outputs"]).items():
            if name.endswith("_sha256"):
                continue
            path = Path(str(value))
            declared = dict(summary["outputs"]).get(f"{name}_sha256")
            if not path.is_file() or (
                declared is not None and _file_sha256(path) != str(declared)
            ):
                raise ValueError(f"Stage 1 capital-review output drifted: {name}")
        migrated = _migrate_capital_review_summary(
            summary, study_root=study_path.parent, stage=1
        )
        if migrated != summary:
            _write_json(summary_path, migrated)
            summary = migrated
        _update_stage1_capital_review_decision(
            study_root=study_path.parent,
            summary_path=summary_path,
            summary=summary,
        )
        return summary
    decisions = _load_stage_decisions(study_path.parent)
    stage1 = _decision_for_stage(decisions, 1)
    if stage1 is None:
        raise ValueError("Stage 1 capital review requires a completed Stage 1 decision")
    descriptor_100 = dict(stage1["incumbent"])
    descriptor_180 = dict(stage1["challenger"])
    book_100, hashes_100 = _descriptor_forecast_book(
        descriptor_100,
        study_path=study_path,
        profile_name="rank100_exit100",
    )
    book_180, hashes_180 = _descriptor_forecast_book(
        descriptor_180,
        study_path=study_path,
        profile_name="rank180_exit180",
    )
    hybrid = _hybrid_forecast_book(
        book_100, book_180, profile_name="rank100_exit180"
    )
    first_view = _descriptor_view_path(
        descriptor_100, year=DEVELOPMENT_YEARS[0], study_path=study_path
    )
    audit_pack = CandidateCompleteAuditPack(first_view)
    market = finite.BacktestMarket(
        date_values=np.asarray(audit_pack.date_values, dtype=object),
        symbol_values=np.asarray(audit_pack.symbol_values, dtype=object),
        entry_open_raw=audit_pack.entry_open_raw,
        exit_close_raw=audit_pack.exit_close_raw,
        exit_sellable=audit_pack.exit_sellable,
        entry_filled=audit_pack.entry_filled,
        contract=audit_pack.contract,
        terminal_recovery_fraction=float(audit_pack.terminal_recovery_fraction),
        forward_days=int(audit_pack.forward_days),
        execution_days=int(audit_pack.execution_days),
    )
    cohort_daily, cohort_summary = _stage1_cohort_review(
        audit_pack=audit_pack,
        finite=finite,
        exit_audit=exit_audit,
        book_100=book_100,
        book_180=book_180,
    )
    account, annual = _stage1_account_review(
        market=market,
        finite=finite,
        books={
            "rank100_exit100": book_100,
            "rank180_exit180": book_180,
            "rank100_exit180": hybrid,
        },
    )
    pairwise = _stage1_account_pairwise(account)
    primary_selection = _stage1_primary_account_selection(account)
    output_root.mkdir(parents=True, exist_ok=True)
    outputs = {
        "cohort_daily_csv": output_root / "cohort_daily_metrics.csv",
        "cohort_summary_csv": output_root / "cohort_summary_metrics.csv",
        "account_metrics_csv": output_root / "account_metrics.csv",
        "account_annual_metrics_csv": output_root / "account_annual_metrics.csv",
        "account_pairwise_csv": output_root / "account_pairwise.csv",
    }
    for path, frame in (
        (outputs["cohort_daily_csv"], cohort_daily),
        (outputs["cohort_summary_csv"], cohort_summary),
        (outputs["account_metrics_csv"], account),
        (outputs["account_annual_metrics_csv"], annual),
        (outputs["account_pairwise_csv"], pairwise),
    ):
        _write_csv(path, frame)
    dynamic = pairwise[
        pairwise["comparison_layer"].eq("full_model_path_exit")
        & pairwise["challenger"].eq("rank180_exit180")
        & pairwise["reference"].eq("rank100_exit100")
    ]
    fixed = pairwise[
        pairwise["comparison_layer"].eq("fixed_exit_ranking")
        & pairwise["challenger"].eq("rank180_exit180")
        & pairwise["reference"].eq("rank100_exit100")
    ]
    hybrid_rows = pairwise[
        pairwise["comparison_layer"].eq("full_model_path_exit")
        & pairwise["challenger"].eq("rank100_exit180")
    ]
    summary = {
        "schema_version": 1,
        "artifact_type": "seq100_structured_stage1_capital_review",
        "status": "completed",
        "created_at": _now(),
        "study_id": STUDY_ID,
        "stage": 1,
        "comparison_status": "capital_efficiency_baseline_selected",
        "primary_account_selection": primary_selection,
        "cohort_diagnostics": {
            "scope": "ranking_path_and_exit_behavior_only",
            "selection_authority": False,
            "observations": dict(
                dict(stage1["cohort_diagnostics"])["registered_observations"]
            ),
        },
        "selection_semantics": {
            "primary": "continuous_account_unit_time_profit",
            "common_calendar_horizon_sessions": int(
                primary_selection["common_calendar_horizon_sessions"]
            ),
            "cohort_metrics": "diagnostic_only",
            "risk_and_stress": "reported_not_selection_gate",
        },
        "account_contract": {
            "starting_cash_cny": float(finite.STARTING_CASH_CNY),
            "daily_selection_count": 3,
            "slots": list(CAPITAL_REVIEW_SLOTS),
            "fixed_exit_days": list(CAPITAL_REVIEW_FIXED_DAYS),
            "dynamic_exit_policies": ["model_plan", "rolling_path"],
            "cost_scenarios": list(CAPITAL_REVIEW_COST_SCENARIOS),
            "no_leverage": True,
            "allow_pyramiding": False,
            "entry": "next_open",
            "exit": "legal_close_with_sellability_deferral",
        },
        "source_prediction_sha256": {
            "100x32": hashes_100,
            "180x32": hashes_180,
        },
        "protocol_amendment": {
            "path": amendment["path"],
            "sha256": amendment["sha256"],
        },
        "active_selection_amendment": {
            "path": active_amendment["path"],
            "sha256": active_amendment["sha256"],
        },
        "fixed_exit_rank_comparison": {
            "comparison_count": int(len(fixed)),
            "rank180_cagr_win_rate": float(
                (
                    fixed["signal_period_cagr_trading_days_delta"].astype(float)
                    > 0.0
                ).mean()
            ),
        },
        "full_model_comparison": {
            "comparison_count": int(len(dynamic)),
            "model180_cagr_win_rate": float(
                (
                    dynamic["signal_period_cagr_trading_days_delta"].astype(float)
                    > 0.0
                ).mean()
            ),
            "model180_total_return_win_rate": float(
                (
                    dynamic["signal_period_total_return_delta"].astype(float) > 0.0
                ).mean()
            ),
            "model180_capital_day_efficiency_win_rate": float(
                (
                    dynamic[
                        "net_pnl_per_deployed_capital_session_delta"
                    ].astype(float)
                    > 0.0
                ).mean()
            ),
        },
        "hybrid_comparison": {
            "comparison_count": int(len(hybrid_rows)),
            "hybrid_cagr_win_rate": float(
                (
                    hybrid_rows[
                        "signal_period_cagr_trading_days_delta"
                    ].astype(float)
                    > 0.0
                ).mean()
            ),
        },
        "interpretation": (
            "The Stage 1 baseline is selected by continuous-account liquidated "
            "terminal wealth, which internalizes holding time, idle cash, and "
            "skipped signals. Cohort alpha is retained only as a diagnostic."
        ),
        "outputs": {
            name: str(path.resolve()) for name, path in outputs.items()
        },
        "qdp_changed": False,
        "base_pack_changed": False,
        "old_checkpoints_changed": False,
        "live_state_changed": False,
    }
    for name, path in outputs.items():
        summary["outputs"][f"{name}_sha256"] = _file_sha256(path)
    _write_json(summary_path, summary)
    _update_stage1_capital_review_decision(
        study_root=study_path.parent,
        summary_path=summary_path,
        summary=summary,
    )
    _append_monitor_event(
        study_path.parent,
        {
            "event": "stage1_capital_review_completed",
            "status": "completed",
            "summary": str(summary_path.resolve()),
            "comparison_status": summary["comparison_status"],
            "message": "Stage 1 finite-capital review completed",
        },
    )
    return summary


def _previous_selected_strategy(
    decisions: Mapping[str, Any], *, stage: int
) -> dict[str, Any]:
    previous = _decision_for_stage(decisions, int(stage) - 1)
    if previous is None:
        raise ValueError(f"Stage {stage - 1} decision is missing")
    review = dict(previous.get("capital_efficiency_review", {}) or {})
    if str(review.get("status", "")) != "completed":
        raise ValueError(f"Stage {stage - 1} capital review is incomplete")
    strategy = review.get("selected_strategy")
    if not isinstance(strategy, Mapping):
        raise ValueError(f"Stage {stage - 1} selected strategy is missing")
    return dict(strategy)


def _update_stage_capital_review_decision(
    *,
    study_root: Path,
    stage: int,
    summary_path: Path,
    summary: Mapping[str, Any],
) -> None:
    decisions = _load_stage_decisions(study_root)
    stages = list(decisions.get("stages", []) or [])
    for index, raw in enumerate(stages):
        if int(raw["stage"]) != int(stage):
            continue
        row = dict(raw)
        selected_strategy = dict(summary["selected_strategy"])
        ranking_descriptor = dict(selected_strategy["ranking_descriptor"])
        row["decision_scope"] = "continuous_account_profit_primary"
        row["capital_efficiency_review"] = {
            "status": "completed",
            "summary_json": str(summary_path.resolve()),
            "summary_sha256": _file_sha256(summary_path),
            "comparison_status": str(summary["comparison_status"]),
            "selection_objective": str(
                dict(summary["primary_account_selection"])["objective"]
            ),
            "selected_strategy": selected_strategy,
            "training_branch_descriptor": ranking_descriptor,
            "capital_baseline_selection_performed": True,
        }
        row["interpretation"] = (
            "The stage strategy is selected only by continuous-account "
            "liquidated terminal wealth over the common calendar window. "
            "Cohort metrics are diagnostic."
        )
        stages[index] = row
        decisions["stages"] = stages
        decisions["updated_at"] = _now()
        _write_json(study_root / "stage_decisions.json", decisions)
        return
    raise ValueError(f"Stage {stage} decision is missing")


def run_stage_capital_review(
    *, stage: int, study_root: Path = STUDY_ROOT
) -> dict[str, Any]:
    from daily_research.path_policy import seq100_finite_capital_backtest as finite
    from daily_research.path_policy.seq100_exit_policy_audit import (
        CandidateCompleteAuditPack,
    )

    if int(stage) not in {2, 3}:
        raise ValueError("generic stage capital review supports only stages 2 and 3")
    verify_prepared_study(study_root=study_root, deep=False)
    study_path = study_root.resolve() / "study.json"
    amendment = _ensure_all_stage_capital_amendment(study_path)
    active_amendment = _ensure_profit_selection_amendment(study_path)
    output_root = study_path.parent / f"stage{int(stage)}_capital_review"
    summary_path = output_root / "summary.json"
    if summary_path.is_file():
        summary = _read_json(summary_path)
        for name, value in dict(summary["outputs"]).items():
            if name.endswith("_sha256"):
                continue
            path = Path(str(value))
            declared = dict(summary["outputs"]).get(f"{name}_sha256")
            if not path.is_file() or (
                declared is not None and _file_sha256(path) != str(declared)
            ):
                raise ValueError(f"Stage {stage} capital-review output drifted: {name}")
        migrated = _migrate_capital_review_summary(
            summary, study_root=study_path.parent, stage=int(stage)
        )
        if migrated != summary:
            _write_json(summary_path, migrated)
            summary = migrated
        _update_stage_capital_review_decision(
            study_root=study_path.parent,
            stage=int(stage),
            summary_path=summary_path,
            summary=summary,
        )
        return summary
    decisions = _load_stage_decisions(study_path.parent)
    decision = _decision_for_stage(decisions, int(stage))
    if decision is None:
        raise ValueError(f"Stage {stage} cohort reference is missing")
    previous_strategy = _previous_selected_strategy(decisions, stage=int(stage))
    incumbent_rank = dict(previous_strategy["ranking_descriptor"])
    incumbent_exit = dict(previous_strategy["exit_descriptor"])
    challenger = dict(decision["challenger"])
    descriptor_cache: dict[str, tuple[Any, dict[str, str]]] = {}

    def load_component(descriptor: Mapping[str, Any]) -> Any:
        key = _canonical_digest(descriptor)
        if key not in descriptor_cache:
            descriptor_cache[key] = _descriptor_forecast_book(
                descriptor,
                study_path=study_path,
                profile_name=f"stage{int(stage)}_component_{len(descriptor_cache) + 1}",
            )
        return descriptor_cache[key][0]

    rank_book = load_component(incumbent_rank)
    exit_book = load_component(incumbent_exit)
    challenger_book = load_component(challenger)

    def strategy_book(
        ranking: Mapping[str, Any],
        exits: Mapping[str, Any],
        *,
        profile_name: str,
    ) -> Any:
        ranking_component = load_component(ranking)
        exit_component = load_component(exits)
        if _canonical_digest(ranking) == _canonical_digest(exits):
            return ranking_component
        return _hybrid_forecast_book(
            ranking_component, exit_component, profile_name=profile_name
        )

    strategy_descriptors = {
        "incumbent_strategy": (incumbent_rank, incumbent_exit),
        "challenger_own": (challenger, challenger),
        "incumbent_rank_challenger_exit": (incumbent_rank, challenger),
        "challenger_rank_incumbent_exit": (challenger, incumbent_exit),
    }
    books = {
        name: strategy_book(ranking, exits, profile_name=name)
        for name, (ranking, exits) in strategy_descriptors.items()
    }
    first_view = _descriptor_view_path(
        incumbent_rank, year=DEVELOPMENT_YEARS[0], study_path=study_path
    )
    audit_pack = CandidateCompleteAuditPack(first_view)
    market = finite.BacktestMarket(
        date_values=np.asarray(audit_pack.date_values, dtype=object),
        symbol_values=np.asarray(audit_pack.symbol_values, dtype=object),
        entry_open_raw=audit_pack.entry_open_raw,
        exit_close_raw=audit_pack.exit_close_raw,
        exit_sellable=audit_pack.exit_sellable,
        entry_filled=audit_pack.entry_filled,
        contract=audit_pack.contract,
        terminal_recovery_fraction=float(audit_pack.terminal_recovery_fraction),
        forward_days=int(audit_pack.forward_days),
        execution_days=int(audit_pack.execution_days),
    )
    account, annual = _dynamic_account_review(
        market=market, finite=finite, books=books
    )
    pairwise = _account_pairwise_against_reference(
        account, reference="incumbent_strategy"
    )
    primary = _primary_account_selection(
        account, expected_configurations=tuple(books)
    )
    selected_name = str(primary["selected_configuration"])
    selected_rank, selected_exit = strategy_descriptors[selected_name]
    selected_strategy = {
        "configuration": selected_name,
        "ranking_descriptor": dict(selected_rank),
        "exit_descriptor": dict(selected_exit),
        "hybrid": bool(
            _canonical_digest(selected_rank) != _canonical_digest(selected_exit)
        ),
    }
    output_root.mkdir(parents=True, exist_ok=True)
    outputs = {
        "account_metrics_csv": output_root / "account_metrics.csv",
        "account_annual_metrics_csv": output_root / "account_annual_metrics.csv",
        "account_pairwise_csv": output_root / "account_pairwise.csv",
    }
    for path, frame in (
        (outputs["account_metrics_csv"], account),
        (outputs["account_annual_metrics_csv"], annual),
        (outputs["account_pairwise_csv"], pairwise),
    ):
        _write_csv(path, frame)
    summary = {
        "schema_version": 1,
        "artifact_type": "seq100_structured_stage_capital_review",
        "status": "completed",
        "created_at": _now(),
        "study_id": STUDY_ID,
        "stage": int(stage),
        "comparison_status": "capital_efficiency_baseline_selected",
        "primary_account_selection": primary,
        "selected_strategy": selected_strategy,
        "compared_strategies": {
            name: {
                "ranking_descriptor": dict(ranking),
                "exit_descriptor": dict(exits),
            }
            for name, (ranking, exits) in strategy_descriptors.items()
        },
        "cohort_diagnostics": {
            "scope": "ranking_path_and_exit_behavior_only",
            "selection_authority": False,
            "observations": dict(
                dict(decision["cohort_diagnostics"])["registered_observations"]
            ),
        },
        "selection_semantics": {
            "primary": "continuous_account_unit_time_profit",
            "common_calendar_horizon_sessions": int(
                primary["common_calendar_horizon_sessions"]
            ),
            "cohort_metrics": "diagnostic_only",
            "risk_and_stress": "reported_not_selection_gate",
        },
        "source_prediction_sha256": {
            key: hashes for key, (_book, hashes) in descriptor_cache.items()
        },
        "protocol_amendment": {
            "path": amendment["path"],
            "sha256": amendment["sha256"],
        },
        "active_selection_amendment": {
            "path": active_amendment["path"],
            "sha256": active_amendment["sha256"],
        },
        "outputs": {name: str(path.resolve()) for name, path in outputs.items()},
        "qdp_changed": False,
        "base_pack_changed": False,
        "old_checkpoints_changed": False,
        "live_state_changed": False,
    }
    for name, path in outputs.items():
        summary["outputs"][f"{name}_sha256"] = _file_sha256(path)
    _write_json(summary_path, summary)
    _update_stage_capital_review_decision(
        study_root=study_path.parent,
        stage=int(stage),
        summary_path=summary_path,
        summary=summary,
    )
    _append_monitor_event(
        study_path.parent,
        {
            "event": f"stage{int(stage)}_capital_review_completed",
            "status": "completed",
            "summary": str(summary_path.resolve()),
            "selected_strategy": selected_strategy,
            "message": f"Stage {int(stage)} finite-capital review completed",
        },
    )
    return summary


def run_focused_finite_capital_diagnostic(
    final_incumbent: Mapping[str, Any], *, study_path: Path
) -> dict[str, Any]:
    from daily_research.path_policy import seq100_finite_capital_backtest as finite
    from daily_research.path_policy.seq100_exit_policy_audit import (
        CandidateCompleteAuditPack,
    )

    output_root = study_path.parent / "finite_capital_diagnostic"
    result_path = output_root / "summary.json"
    if result_path.is_file():
        result = _read_json(result_path)
        if dict(result.get("final_incumbent", {}) or {}) != dict(final_incumbent):
            raise ValueError("finite-capital diagnostic belongs to another incumbent")
        return result
    output_root.mkdir(parents=True, exist_ok=True)
    first_view = _descriptor_view_path(
        final_incumbent, year=DEVELOPMENT_YEARS[0], study_path=study_path
    )
    audit_pack = CandidateCompleteAuditPack(first_view)
    market = finite.BacktestMarket(
        date_values=np.asarray(audit_pack.date_values, dtype=object),
        symbol_values=np.asarray(audit_pack.symbol_values, dtype=object),
        entry_open_raw=audit_pack.entry_open_raw,
        exit_close_raw=audit_pack.exit_close_raw,
        exit_sellable=audit_pack.exit_sellable,
        entry_filled=audit_pack.entry_filled,
        contract=audit_pack.contract,
        terminal_recovery_fraction=float(audit_pack.terminal_recovery_fraction),
        forward_days=int(audit_pack.forward_days),
        execution_days=int(audit_pack.execution_days),
    )
    top3_book = finite.ForecastBook("structured_joint_turnover")
    top1_book = finite.ForecastBook("structured_joint_turnover_top1")
    source_hashes: dict[str, str] = {}
    variant = _variant_from_payload(dict(final_incumbent["variant"]))
    for year in DEVELOPMENT_YEARS:
        view_path = _descriptor_view_path(
            final_incumbent, year=year, study_path=study_path
        )
        manifest = _read_json(view_path)
        dataset = training.SequencePathPackDataset(
            manifest,
            split="development",
            max_samples=0,
            input_channel_profile=variant.input_channel_profile,
            index_role="candidate",
        )
        run_dir = _descriptor_run_dir(
            final_incumbent, year=year, study_path=study_path
        )
        prediction_path = run_dir / "predictions/development_predictions.csv"
        if not prediction_path.is_file():
            raise FileNotFoundError(prediction_path)
        source_hashes[str(year)] = _file_sha256(prediction_path)
        frame = pd.read_csv(
            prediction_path,
            usecols=["trade_date", "symbol", "score", "predicted_exit_day"],
            dtype={"trade_date": str, "symbol": str, "score": np.float64},
        )
        candidates = dataset.sample_index
        if len(frame) != len(candidates):
            raise ValueError(f"finite-capital prediction row count drifted for {year}")
        if not np.array_equal(
            frame["trade_date"].astype(str).to_numpy(),
            candidates["trade_date"].astype(str).to_numpy(),
        ) or not np.array_equal(
            frame["symbol"].astype(str).to_numpy(),
            candidates["symbol"].astype(str).to_numpy(),
        ):
            raise ValueError(f"finite-capital candidate order drifted for {year}")
        scores = pd.to_numeric(frame["score"], errors="coerce").to_numpy(
            dtype=np.float64
        )
        planned = np.clip(
            np.rint(
                pd.to_numeric(frame["predicted_exit_day"], errors="coerce").to_numpy(
                    dtype=np.float64
                )
            ),
            2,
            60,
        ).astype(np.int16)
        if not bool(np.isfinite(scores).all()):
            raise ValueError(f"finite-capital scores are incomplete for {year}")
        symbol_indices = candidates["symbol_idx"].to_numpy(dtype=np.int64)
        for date_idx_raw, positions_raw in candidates.groupby(
            "date_idx", sort=True
        ).indices.items():
            positions = np.asarray(positions_raw, dtype=np.int64)
            top3_book.add_day(
                date_idx=int(date_idx_raw),
                symbol_idx=symbol_indices[positions],
                score=scores[positions],
                planned_day=planned[positions],
            )
            local_top = int(np.argsort(-scores[positions], kind="mergesort")[0])
            selected = positions[[local_top]]
            top1_book.add_day(
                date_idx=int(date_idx_raw),
                symbol_idx=symbol_indices[selected],
                score=scores[selected],
                planned_day=planned[selected],
            )
        del dataset, frame, scores, planned
        gc.collect()
    signal_indices = list(top3_book.signal_date_indices)
    if signal_indices != list(top1_book.signal_date_indices) or not signal_indices:
        raise ValueError("finite-capital signal-date coverage drifted")
    jobs = (
        (
            "structured_d16_top3_slots3",
            top3_book,
            finite.PolicySpec(name="fixed_d16", kind="fixed", fixed_day=16),
            3,
        ),
        (
            "structured_d18_top1_slots1",
            top1_book,
            finite.PolicySpec(name="fixed_d18", kind="fixed", fixed_day=18),
            1,
        ),
    )
    results: dict[str, Any] = {}
    guard = finite._MemoryGuard()
    for name, book, policy, slots in jobs:
        metric, equity, trades, annual = finite.simulate_portfolio(
            market=market,
            book=book,
            raw_top3_paths={},
            policy=policy,
            slots=slots,
            cost_scenario="base",
            first_signal_date_idx=int(signal_indices[0]),
            last_signal_date_idx=int(signal_indices[-1]),
            starting_cash=finite.STARTING_CASH_CNY,
            memory_guard=guard,
        )
        equity_path = output_root / f"{name}_equity.csv"
        trades_path = output_root / f"{name}_trades.csv"
        _write_csv(equity_path, equity)
        _write_csv(trades_path, trades)
        results[name] = {
            "metric": metric,
            "annual_metrics": annual,
            "equity_csv": str(equity_path.resolve()),
            "equity_csv_sha256": _file_sha256(equity_path),
            "trades_csv": str(trades_path.resolve()),
            "trades_csv_sha256": _file_sha256(trades_path),
        }
    result = {
        "schema_version": 1,
        "artifact_type": "seq100_structured_input_ablation_focused_finite_capital",
        "created_at": _now(),
        "final_incumbent": dict(final_incumbent),
        "source_prediction_sha256": source_hashes,
        "starting_cash_cny": float(finite.STARTING_CASH_CNY),
        "results": results,
        "parameter_selection_performed": False,
        "qdp_changed": False,
        "live_state_changed": False,
    }
    _write_json(result_path, result)
    return result


def _architecture_loss_compatibility(
    *, study_path: Path, descriptor: Mapping[str, Any], year: int
) -> dict[str, Any]:
    base_run = _old_structured_run(year)
    candidate_run = _descriptor_run_dir(descriptor, year=year, study_path=study_path)
    base_checkpoint = torch.load(
        base_run / "best_model.pt", map_location="cpu", weights_only=False
    )
    candidate_checkpoint = torch.load(
        candidate_run / "best_model.pt", map_location="cpu", weights_only=False
    )
    ignored_shape_keys = {"input_norm.weight", "input_norm.bias", "proj.weight"}
    base_shapes = {
        key: tuple(value.shape)
        for key, value in dict(base_checkpoint["model_state_dict"]).items()
        if key not in ignored_shape_keys
    }
    candidate_shapes = {
        key: tuple(value.shape)
        for key, value in dict(candidate_checkpoint["model_state_dict"]).items()
        if key not in ignored_shape_keys
    }
    if base_shapes != candidate_shapes:
        raise ValueError(f"model architecture drifted beyond input projection for {year}")
    ignored_config = {
        "pack_manifest",
        "output_root",
        "run_tag",
        "input_channel_profile",
        "batch_size",
        "development_contract",
    }
    base_config = {
        key: value
        for key, value in dict(base_checkpoint["resolved_training_config"]).items()
        if key not in ignored_config
    }
    candidate_config = {
        key: value
        for key, value in dict(candidate_checkpoint["resolved_training_config"]).items()
        if key not in ignored_config
    }
    if base_config != candidate_config:
        raise ValueError(f"model/loss contract drifted for {year}")
    return {
        "year": int(year),
        "architecture_equal_excluding_input_projection": True,
        "training_and_loss_config_equal_excluding_input_material": True,
        "candidate_input_dim": int(candidate_checkpoint["input_dim"]),
        "candidate_checkpoint_sha256": _file_sha256(candidate_run / "best_model.pt"),
    }


def summarize_structured_input_ablation(
    *, study_root: Path = STUDY_ROOT
) -> dict[str, Any]:
    verify_prepared_study(study_root=study_root)
    study_path = study_root.resolve() / "study.json"
    decisions = _load_stage_decisions(study_path.parent)
    if [int(row["stage"]) for row in list(decisions.get("stages", []) or [])] != [1, 2, 3]:
        raise ValueError("summarize requires three completed stage decisions")
    stage1_capital = run_stage1_capital_review(study_root=study_path.parent)
    stage2_capital = run_stage_capital_review(
        stage=2, study_root=study_path.parent
    )
    stage3_capital = run_stage_capital_review(
        stage=3, study_root=study_path.parent
    )
    decisions = _load_stage_decisions(study_path.parent)
    descriptors = [_initial_incumbent()]
    descriptors.extend(dict(row["challenger"]) for row in decisions["stages"])
    evaluations = [
        _descriptor_evaluation(descriptor, study_path=study_path)
        for descriptor in descriptors
    ]
    for evaluation in evaluations:
        _write_task_evaluations(evaluation, study_path=study_path)
    metrics = _vintage_metrics_frame(evaluations)
    metrics_path = study_path.parent / "vintage_metrics.csv"
    _write_csv(metrics_path, metrics)
    daily = _daily_pairwise_deltas(decisions=decisions, study_path=study_path)
    daily_path = study_path.parent / "daily_pairwise_deltas.csv"
    _write_csv(daily_path, daily)
    bootstrap = _bootstrap_diagnostics(daily)
    final_strategy = dict(
        dict(decisions["stages"][-1]["capital_efficiency_review"])[
            "selected_strategy"
        ]
    )
    final_ranking_descriptor = dict(final_strategy["ranking_descriptor"])
    feature_audit = build_feature_audit(
        final_ranking_descriptor, study_path=study_path
    )
    finite_capital = run_focused_finite_capital_diagnostic(
        final_ranking_descriptor, study_path=study_path
    )
    summary = {
        "schema_version": 1,
        "artifact_type": "seq100_structured_input_ablation_summary",
        "status": "completed",
        "created_at": _now(),
        "study_id": STUDY_ID,
        "study_contract_sha256": str(_read_json(study_path)["contract_sha256"]),
        "stage_decisions": list(decisions["stages"]),
        "final_ranking_descriptor": final_ranking_descriptor,
        "final_strategy": final_strategy,
        "evaluated_descriptors": descriptors,
        "moving_block_diagnostics": bootstrap,
        "moving_block_diagnostics_are_formal_gate": False,
        "outputs": {
            "stage_decisions_json": str(
                (study_path.parent / "stage_decisions.json").resolve()
            ),
            "vintage_metrics_csv": str(metrics_path.resolve()),
            "vintage_metrics_csv_sha256": _file_sha256(metrics_path),
            "daily_pairwise_deltas_csv": str(daily_path.resolve()),
            "daily_pairwise_deltas_csv_sha256": _file_sha256(daily_path),
            "feature_audit_json": str(
                (study_path.parent / "feature_audit.json").resolve()
            ),
            "feature_audit_json_sha256": _file_sha256(
                study_path.parent / "feature_audit.json"
            ),
            "feature_audit_csv": str(
                (study_path.parent / "feature_audit.csv").resolve()
            ),
            "feature_audit_csv_sha256": _file_sha256(
                study_path.parent / "feature_audit.csv"
            ),
            "finite_capital_summary": str(
                (study_path.parent / "finite_capital_diagnostic/summary.json").resolve()
            ),
            "finite_capital_summary_sha256": _file_sha256(
                study_path.parent / "finite_capital_diagnostic/summary.json"
            ),
            "stage1_capital_review_summary": str(
                (study_path.parent / "stage1_capital_review/summary.json").resolve()
            ),
            "stage1_capital_review_summary_sha256": _file_sha256(
                study_path.parent / "stage1_capital_review/summary.json"
            ),
            "stage2_capital_review_summary": str(
                (study_path.parent / "stage2_capital_review/summary.json").resolve()
            ),
            "stage2_capital_review_summary_sha256": _file_sha256(
                study_path.parent / "stage2_capital_review/summary.json"
            ),
            "stage3_capital_review_summary": str(
                (study_path.parent / "stage3_capital_review/summary.json").resolve()
            ),
            "stage3_capital_review_summary_sha256": _file_sha256(
                study_path.parent / "stage3_capital_review/summary.json"
            ),
        },
        "feature_audit_effective_rank": float(
            dict(feature_audit["base_32_feature_audit"])["effective_rank"]
        ),
        "focused_finite_capital_results": {
            key: dict(value["metric"])
            for key, value in dict(finite_capital["results"]).items()
        },
        "stage1_primary_account_selection": dict(
            stage1_capital["primary_account_selection"]
        ),
        "stage2_primary_account_selection": dict(
            stage2_capital["primary_account_selection"]
        ),
        "stage3_primary_account_selection": dict(
            stage3_capital["primary_account_selection"]
        ),
        "final_fit_performed": False,
        "deployment_changed": False,
        "qdp_changed": False,
        "provider_called": False,
        "evaluated_2026": False,
    }
    summary_path = study_path.parent / "comparison_summary.json"
    markdown_path = study_path.parent / "comparison_summary.md"
    summary["outputs"]["comparison_summary_md"] = str(markdown_path.resolve())
    _write_json(summary_path, summary)
    markdown_path.write_text(_comparison_markdown(summary), encoding="utf-8")
    _update_study_runtime(
        study_path,
        status="completed",
        current_stage=None,
        current_task=None,
        comparison_summary_json=str(summary_path.resolve()),
        comparison_summary_md=str(markdown_path.resolve()),
        error=None,
    )
    _append_monitor_event(
        study_path.parent,
        {
            "event": "workflow_completed",
            "status": "completed",
            "final_strategy": final_strategy,
            "summary": str(summary_path.resolve()),
            "message": "Structured input-ablation workflow completed",
        },
    )
    return _read_json(summary_path)


def verify_structured_input_ablation(
    *, study_root: Path = STUDY_ROOT
) -> dict[str, Any]:
    prepared = verify_prepared_study(study_root=study_root)
    study_path = study_root.resolve() / "study.json"
    study = _read_json(study_path)
    decisions = _load_stage_decisions(study_path.parent)
    if len(list(decisions.get("stages", []) or [])) != 3:
        raise ValueError("verification requires all three stage decisions")
    stage1_review = run_stage1_capital_review(study_root=study_path.parent)
    if str(stage1_review.get("comparison_status", "")) != (
        "capital_efficiency_baseline_selected"
    ):
        raise ValueError("Stage 1 capital-efficiency selection is incomplete")
    for review_stage in (2, 3):
        review = run_stage_capital_review(
            stage=review_stage, study_root=study_path.parent
        )
        if str(review.get("comparison_status", "")) != (
            "capital_efficiency_baseline_selected"
        ):
            raise ValueError(
                f"Stage {review_stage} capital-efficiency selection is incomplete"
            )
    decisions = _load_stage_decisions(study_path.parent)
    task_count = 0
    architecture: list[dict[str, Any]] = []
    fairness_reference = dict(dict(study["material"])["fairness_material"])
    for decision in decisions["stages"]:
        descriptor = dict(decision["challenger"])
        stage = int(descriptor["origin_stage"])
        variant = _variant_from_payload(dict(descriptor["variant"]))
        for year in DEVELOPMENT_YEARS:
            completed, partial = _classify_task_runs(
                study_path=study_path,
                stage=stage,
                variant=variant,
                year=year,
            )
            if len(completed) != 1 or partial:
                raise ValueError(f"task terminal state is invalid: stage {stage}/{year}")
            evaluation_path = completed[0] / "evaluation.json"
            if not evaluation_path.is_file():
                raise FileNotFoundError(evaluation_path)
            evaluation = _read_json(evaluation_path)
            if str(evaluation.get("status", "")) != "completed":
                raise ValueError("task evaluation is not completed")
            if dict(evaluation["fairness_material"]) != dict(
                fairness_reference[str(year)]
            ):
                raise ValueError("task fairness material drifted")
            architecture.append(
                _architecture_loss_compatibility(
                    study_path=study_path, descriptor=descriptor, year=year
                )
            )
            task_count += 1
    if task_count != 9:
        raise AssertionError("verification did not cover nine challenger folds")
    summary_path = study_path.parent / "comparison_summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(summary_path)
    summary = _read_json(summary_path)
    for name, value in dict(summary["outputs"]).items():
        if not str(name).endswith("_sha256") and str(name) not in {
            "stage_decisions_json"
        }:
            path = Path(str(value)).resolve()
            if not path.is_file():
                raise FileNotFoundError(path)
            declared = dict(summary["outputs"]).get(f"{name}_sha256")
            if declared is not None and _file_sha256(path) != str(declared):
                raise ValueError(f"summary output drifted: {name}")
    monitor = _read_json(study_path.parent / "monitor.json")
    if str(monitor.get("event", "")) != "workflow_completed" or str(
        monitor.get("status", "")
    ) != "completed":
        raise ValueError("monitor did not reach the workflow terminal event")
    if str(dict(study.get("runtime", {}) or {}).get("status", "")) != "completed":
        raise ValueError("study runtime is not completed")
    _assert_no_other_research_process()
    frozen = _assert_frozen_inputs()
    return {
        "status": "ok",
        "study": str(study_path),
        "prepared": prepared,
        "training_task_count": task_count,
        "architecture_loss_checks": architecture,
        "stage_decision_count": 3,
        "final_ranking_descriptor": dict(summary["final_ranking_descriptor"]),
        "final_strategy": dict(summary["final_strategy"]),
        "monitor_terminal": monitor,
        "protected_inputs": frozen,
        "provider_called": False,
        "research_processes_remaining": 0,
    }
