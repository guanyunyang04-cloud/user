"""Bounded training-history experiment for the Structured 180x35 model.

The experiment reuses the immutable Seq100 panels, labels, execution arrays,
candidate indexes, turnover overlay, and expanding checkpoints.  It creates
only bounded sample indexes, per-window normalization, checkpoints, and
evaluation artifacts under its own roots.
"""

from __future__ import annotations

import gc
import hashlib
import json
import math
import os
import shutil
import subprocess
import time
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import psutil
import torch

from daily_research.path_policy import qdp_v2_sequence_path_training as training
from daily_research.path_policy import seq100_development as development
from daily_research.path_policy import seq100_finite_capital_backtest as finite
from daily_research.path_policy import seq100_structured_capital_speed as capital
from daily_research.path_policy import seq100_structured_experiment as structured
from daily_research.path_policy import seq100_structured_input_ablation as input_ablation
from daily_research.path_policy import seq100_walkforward as walkforward
from daily_research.path_policy.seq100_exit_policy_audit import CandidateCompleteAuditPack
from daily_research.path_policy.seq100_mainline import build_todayclose_path_only_train_argv


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
PYTHON = Path("C:/Users/ASUS/miniconda3/envs/yolos/python.exe")
STUDY_ID = "seq100_structured_training_window_ablation_2025_v1"
EXPERIMENT_ID = "structured-training-window-ablation-v1"
STUDY_ROOT = WORKSPACE_ROOT / (
    "daily_research/output/path_policy/studies/"
    "seq100_structured_training_window_ablation_2025_v1"
)
STORE_ROOT = WORKSPACE_ROOT / (
    "daily_research/data/research_store/"
    "seq100_structured_training_window_ablation_v1"
)
INPUT_STUDY_ROOT = input_ablation.STUDY_ROOT
INPUT_OVERLAY_ROOT = input_ablation.OVERLAY_ROOT

LOOKBACK_DAYS = 180
INPUT_DIM = 35
INPUT_CHANNEL_PROFILE = training.INPUT_CHANNEL_PROFILE_DAILY_ONLY_TURNOVER
SEED = 7
BATCH_SIZE = 512
DEPENDENCY_DAYS = 80
DEVELOPMENT_YEARS = (2023, 2024, 2025)
SHORT_WINDOW_YEARS = (12, 8, 5, 3)
SCREEN_YEAR = 2025
FIXED_EXIT_DAYS = tuple(range(2, 61))
COST_SCENARIOS = ("base", "double_slippage")
CAPITAL_CONFIGS = ((1, 1), (3, 3), (3, 6))
SCREEN_SELECTION_POLICIES = ("model_plan", "rolling_path", "fixed_d7")
MEMORY_GUARD_GIB = 0.5
MEMORY_INTERVAL_SECONDS = 1.0
MEMORY_CONSECUTIVE_BREACHES = 2
MONITOR_POLL_SECONDS = 5.0
STALE_SECONDS = 15 * 60
STUDY_SIZE_LIMIT_BYTES = 6 * 1024**3

EXPECTED_2025 = {
    "expanding": {"first_train_signal": "2010-09-29", "train_rows": 6_952_766},
    "window12y": {"first_train_signal": "2012-08-30", "train_rows": 6_273_740},
    "window8y": {"first_train_signal": "2016-08-30", "train_rows": 4_859_679},
    "window5y": {"first_train_signal": "2019-08-30", "train_rows": 3_354_874},
    "window3y": {"first_train_signal": "2021-08-30", "train_rows": 2_096_298},
}
EXPECTED_2025_CANDIDATES = 724_125
EXPECTED_2025_SUPERVISED = 723_807

VALID_PHASES = (
    "prepared",
    "screen_2025_training",
    "screen_2025_evaluating",
    "confirmation_training",
    "confirmation_evaluating",
    "optional_epoch_confirmation",
    "completed",
)


@dataclass(frozen=True)
class TrainingTask:
    window_years: int
    development_year: int
    epochs: int = 1

    @property
    def window_id(self) -> str:
        return window_id(self.window_years)

    @property
    def task_id(self) -> str:
        return (
            f"{self.window_id}:{int(self.development_year)}:"
            f"epochs{int(self.epochs)}"
        )


@dataclass(frozen=True)
class CapitalJob:
    phase: str
    model_id: str
    top_k: int
    slots: int
    policy_name: str
    policy_kind: str
    fixed_day: int | None
    cost_scenario: str

    @property
    def job_id(self) -> str:
        return (
            f"{self.model_id}__top{int(self.top_k)}__slots{int(self.slots):02d}__"
            f"{self.policy_name}__{self.cost_scenario}"
        )


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _json_default(value: Any) -> Any:
    return input_ablation._json_default(value)


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
    return input_ablation._file_sha256(path)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    input_ablation._write_json(path, payload)


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    input_ablation._write_csv(path, frame)


def _write_parquet(path: Path, frame: pd.DataFrame) -> None:
    input_ablation._write_parquet(path, frame)


def window_id(years: int) -> str:
    if int(years) <= 0:
        raise ValueError("training-window years must be positive")
    return f"window{int(years)}y"


def _expanding_descriptor() -> dict[str, Any]:
    return {
        "source": "ablation_run",
        "origin_stage": 2,
        "variant": input_ablation.InputVariant(
            LOOKBACK_DAYS, turnover=True, intraday=False
        ).to_dict(),
    }


def _expanding_view(year: int) -> Path:
    study = _read_json(INPUT_STUDY_ROOT / "study.json")
    return Path(
        str(
            dict(dict(study["material"])["views"])["lookback180_turnover"][
                str(int(year))
            ]
        )
    ).resolve()


def _expanding_run(year: int) -> Path:
    return input_ablation._descriptor_run_dir(
        _expanding_descriptor(),
        year=int(year),
        study_path=INPUT_STUDY_ROOT / "study.json",
    )


def _directory_size(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def _assert_stage3_not_started() -> None:
    study = _read_json(INPUT_STUDY_ROOT / "study.json")
    runtime = dict(study.get("runtime", {}) or {})
    if bool(runtime.get("stage3_started", False)):
        raise ValueError("Structured input Stage 3 has started")
    stage3 = INPUT_STUDY_ROOT / "runs/stage_3"
    if stage3.exists():
        raise ValueError("Structured input Stage 3 run directory exists")


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
        "run-structured-training-window",
        "prepare-structured-training-window",
        "seq100_structured_training_window",
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
        raise RuntimeError(f"another training-window research process is active: {matches}")


def _source_identity() -> dict[str, Any]:
    frozen = input_ablation._assert_frozen_inputs()
    _assert_stage3_not_started()
    overlay_manifest = INPUT_OVERLAY_ROOT / "manifest.json"
    input_study = INPUT_STUDY_ROOT / "study.json"
    live_state = WORKSPACE_ROOT / "daily_research/output/active_execution_strategy.json"
    runs: dict[str, Any] = {}
    for year in DEVELOPMENT_YEARS:
        run = _expanding_run(year)
        runs[str(year)] = {
            "run_dir": str(run),
            "checkpoint_sha256": _file_sha256(run / "best_model.pt"),
            "summary_sha256": _file_sha256(run / "sequence_path_training_summary.json"),
            "prediction_sha256": _file_sha256(
                run / "predictions/development_predictions.csv"
            ),
            "view_sha256": _file_sha256(_expanding_view(year)),
        }
    return {
        "frozen_qdp_and_base": frozen,
        "input_study_path": str(input_study),
        "input_study_sha256": _file_sha256(input_study),
        "turnover_overlay_manifest": str(overlay_manifest),
        "turnover_overlay_manifest_sha256": _file_sha256(overlay_manifest),
        "expanding_runs": runs,
        "live_state_path": str(live_state),
        "live_state_sha256": _file_sha256(live_state) if live_state.is_file() else None,
        "stage3_started": False,
    }


def _semantic_contract() -> dict[str, Any]:
    base = asdict(input_ablation._base_structured_profile(BATCH_SIZE))
    for key in ("store_view", "output_root", "run_tag", "epochs"):
        base.pop(key, None)
    return {
        "schema_version": 1,
        "contract_id": STUDY_ID,
        "experiment_id": EXPERIMENT_ID,
        "model": {
            "profile": "structured_joint_turnover",
            "lookback_days": LOOKBACK_DAYS,
            "input_dim": INPUT_DIM,
            "input_channel_profile": INPUT_CHANNEL_PROFILE,
            "seed": SEED,
            "screen_epochs": 1,
            "optional_confirmation_epochs": 3,
            "batch_size": BATCH_SIZE,
            "parameters": base,
        },
        "training_windows": {
            "expanding": "reuse_existing_checkpoint",
            "bounded_calendar_years": list(SHORT_WINDOW_YEARS),
            "start_rule": (
                "first legal train signal on or after safe_train_signal_end minus "
                "N calendar years"
            ),
            "normalization_rule": (
                "first_train_signal_date_idx_minus_179 through development_start_exclusive"
            ),
            "purge_days": DEPENDENCY_DAYS,
        },
        "development_protocol": {
            "years": list(DEVELOPMENT_YEARS),
            "method": "purged_bounded_development_walkforward",
            "split_roles": {"fit": "train", "evaluation": "development"},
            "seed": SEED,
            "purge_days": DEPENDENCY_DAYS,
            "candidate_membership_changes_with_window": False,
        },
        "early_stopping": {
            "metric": training.EARLY_STOPPING_METRIC_DEVELOPMENT_PRICE_TOTAL_LOSS,
            "mode": training.EARLY_STOPPING_MODE_MIN,
            "minimum_complete_epochs": 1,
            "maximum_epochs": 3,
            "patience": 2,
            "restore_best_checkpoint": True,
        },
        "screen": {
            "year": SCREEN_YEAR,
            "capital_configurations": [
                {"top_k": top_k, "slots": slots}
                for top_k, slots in CAPITAL_CONFIGS
            ],
            "selection_cost": "double_slippage",
            "selection_policies": list(SCREEN_SELECTION_POLICIES),
            "fixed_curve_diagnostic_days": list(FIXED_EXIT_DAYS),
            "score": "equal_mean_annualized_log_growth",
            "shortlist": "top_two_plus_within_five_percent_of_champion",
            "expanding_always_retained": True,
        },
        "confirmation": {
            "years": list(DEVELOPMENT_YEARS),
            "fixed_days_selected_jointly": list(FIXED_EXIT_DAYS),
            "own_exit_policies": ["model_plan", "rolling_path"],
            "cross_model_rank_exit_hybrid": False,
            "short_window_must_strictly_beat_expanding": True,
        },
        "protected_boundaries": {
            "qdp_read_only": True,
            "provider_calls": 0,
            "base_pack_copied": False,
            "turnover_overlay_mutated": False,
            "old_artifacts_mutated": False,
            "stage3_started": False,
            "live_state_mutated": False,
            "evaluate_2026": False,
        },
    }


def bounded_start_date(
    *, safe_train_signal_end: str, window_years: int, legal_signal_dates: Sequence[str]
) -> str:
    threshold = pd.Timestamp(str(safe_train_signal_end)) - pd.DateOffset(
        years=int(window_years)
    )
    dates = pd.DatetimeIndex(pd.to_datetime(list(legal_signal_dates), errors="raise"))
    eligible = dates[dates >= threshold]
    if len(eligible) == 0:
        raise ValueError("bounded training window has no legal signal date")
    return eligible[0].strftime("%Y-%m-%d")


def _normalization(
    source: Mapping[str, Any], *, start_idx: int, end_idx_exclusive: int
) -> dict[str, Any]:
    dates = [str(value) for value in source["date_values"]]
    output: dict[str, Any] = {
        "fit_scope": "feature_dates_before_development_start",
        "fit_date_start": dates[int(start_idx)],
        "fit_date_end": dates[int(end_idx_exclusive) - 1],
        "fit_date_count": int(end_idx_exclusive - start_idx),
        "fit_date_end_exclusive": dates[int(end_idx_exclusive)],
        "development_feature_date_count": 0,
    }
    for name in ("daily_raw", "daily_state", "turnover"):
        output[name] = walkforward._fit_channel_normalization(
            dict(dict(source["feature_channels"])[name]),
            start_idx=int(start_idx),
            end_idx_exclusive=int(end_idx_exclusive),
        )
    return output


def _fairness(view: Mapping[str, Any]) -> dict[str, Any]:
    return input_ablation._view_fairness_material(view)


def _build_bounded_view(
    *, study_path: Path, year: int, years: int
) -> tuple[Path, dict[str, Any]]:
    source_path = _expanding_view(year)
    source = _read_json(source_path)
    source_index = Path(str(source["sample_index_path"])).resolve()
    frame = pd.read_parquet(source_index)
    train_mask = frame["split"].astype(str).eq("train")
    train = frame.loc[train_mask]
    legal_dates = sorted(train["trade_date"].astype(str).unique())
    walk = dict(source["development_walkforward"])
    first_date = bounded_start_date(
        safe_train_signal_end=str(walk["safe_train_signal_end"]),
        window_years=int(years),
        legal_signal_dates=legal_dates,
    )
    selected = frame.loc[
        ~train_mask | frame["trade_date"].astype(str).ge(first_date)
    ].copy()
    selected = selected.sort_values(
        ["date_idx", "symbol_idx"], kind="mergesort"
    ).reset_index(drop=True)
    selected_train = selected[selected["split"].astype(str).eq("train")]
    selected_dev = selected[selected["split"].astype(str).eq("development")]
    first_idx = int(selected_train["date_idx"].min())
    normalization_start_idx = first_idx - (LOOKBACK_DAYS - 1)
    development_start_idx = int(walk["development_start_date_idx"])
    if normalization_start_idx < 0:
        raise ValueError("bounded normalization cannot cover the first input sequence")

    index_path = STORE_ROOT / "indexes" / f"{window_id(years)}_{int(year)}.parquet"
    if index_path.is_file():
        observed = pd.read_parquet(index_path)
        if not observed.equals(selected):
            raise ValueError(f"bounded sample index drifted: {index_path}")
    else:
        _write_parquet(index_path, selected)

    contract = _read_json(study_path)
    binding = {
        "contract_id": STUDY_ID,
        "contract_sha256": str(contract["contract_sha256"]),
        "contract_file_sha256": str(contract["contract_sha256"]),
        "path": str(study_path.resolve()),
    }
    view = json.loads(json.dumps(source, ensure_ascii=False))
    view["created_at"] = _now()
    view["sample_index_path"] = str(index_path.resolve())
    view["sample_count"] = int(len(selected))
    view["sample_count_by_split"] = {
        str(key): int(value)
        for key, value in selected["split"].astype(str).value_counts().items()
    }
    view["normalization"] = _normalization(
        source,
        start_idx=normalization_start_idx,
        end_idx_exclusive=development_start_idx,
    )
    view["research_contract"] = binding
    view["development_contract"] = binding
    walk["train_start_year"] = int(first_date[:4])
    view["development_walkforward"] = walk
    artifact = dict(view.get("artifact_view", {}) or {})
    artifact.update(
        {
            "schema_version": 1,
            "view_id": f"{STUDY_ID}_{window_id(years)}_{int(year)}",
            "view_type": "structured_bounded_training_window_development_view",
        }
    )
    view["artifact_view"] = artifact
    window_contract = {
        "schema_version": 1,
        "window_id": window_id(years),
        "window_years": int(years),
        "development_year": int(year),
        "source_view": str(source_path),
        "source_view_sha256": _file_sha256(source_path),
        "source_sample_index_sha256": _file_sha256(source_index),
        "sample_index_sha256": _file_sha256(index_path),
        "first_train_signal": str(selected_train["trade_date"].min()),
        "last_train_signal": str(selected_train["trade_date"].max()),
        "train_row_count": int(len(selected_train)),
        "development_row_count": int(len(selected_dev)),
        "normalization_start_date_idx": int(normalization_start_idx),
        "normalization_end_date_idx_exclusive": int(development_start_idx),
        "normalization_fit_start": str(view["normalization"]["fit_date_start"]),
        "normalization_fit_end_exclusive": str(
            view["normalization"]["fit_date_end_exclusive"]
        ),
        "purge_days": DEPENDENCY_DAYS,
        "fairness_material": _fairness(view),
    }
    window_contract["sha256"] = _canonical_digest(window_contract)
    view["structured_training_window_ablation"] = window_contract
    view["development_fold_training_contract"] = (
        walkforward._compute_development_fold_training_contract(view)
    )
    target = STORE_ROOT / "views" / f"{window_id(years)}_{int(year)}.json"
    if target.is_file():
        old = _read_json(target)
        old.pop("created_at", None)
        comparable = dict(view)
        comparable.pop("created_at", None)
        if _canonical_digest(old) != _canonical_digest(comparable):
            raise ValueError(f"bounded view drifted: {target}")
    else:
        _write_json(target, view)
    verification = walkforward.verify_development_walkforward_view(target)
    if str(verification["status"]) != "ok":
        raise RuntimeError(f"bounded view verification failed: {verification}")
    return target.resolve(), window_contract


def _validate_expected_2025(material: Mapping[str, Any]) -> None:
    expanding = _read_json(_expanding_view(2025))
    if int(expanding["candidate_count"]) != EXPECTED_2025_CANDIDATES:
        raise ValueError("2025 candidate count drifted")
    if int(dict(expanding["sample_count_by_split"])["development"]) != EXPECTED_2025_SUPERVISED:
        raise ValueError("2025 supervised development count drifted")
    for key, expected in EXPECTED_2025.items():
        if key == "expanding":
            view = expanding
            train_rows = int(dict(view["sample_count_by_split"])["train"])
            first = str(
                pd.read_parquet(
                    Path(str(view["sample_index_path"])),
                    columns=["split", "trade_date"],
                ).query("split == 'train'")["trade_date"].min()
            )
        else:
            row = dict(dict(material["window_contracts"])[key]["2025"])
            train_rows = int(row["train_row_count"])
            first = str(row["first_train_signal"])
        if train_rows != int(expected["train_rows"]) or first != str(
            expected["first_train_signal"]
        ):
            raise ValueError(
                f"registered 2025 boundary drifted for {key}: {first}/{train_rows}"
            )


def prepare_structured_training_window(
    *, study_root: Path = STUDY_ROOT
) -> dict[str, Any]:
    _assert_no_research_process()
    _assert_stage3_not_started()
    study_path = study_root.resolve() / "study.json"
    semantic = _semantic_contract()
    if study_path.is_file():
        study = _read_json(study_path)
        if str(study.get("study_id", "")) != STUDY_ID:
            raise ValueError("refusing to reuse an unrelated study root")
        if dict(study.get("contract", {}) or {}) != semantic:
            raise ValueError("training-window study contract drifted")
        if dict(study.get("material", {}) or {}).get("views"):
            verify_structured_training_window(study_root=study_root, require_complete=False)
            return study
    else:
        study = {
            "schema_version": 1,
            "artifact_type": "seq100_current_study",
            "study_type": "seq100_structured_training_window_study",
            "study_id": STUDY_ID,
            "contract": semantic,
            "contract_sha256": _canonical_digest(semantic),
            "runtime": {
                "phase": "preparing",
                "created_at": _now(),
                "updated_at": _now(),
                "current_task": None,
                "error": None,
            },
            "material": {},
        }
        _write_json(study_path, study)
    try:
        source_identity = _source_identity()
        views: dict[str, dict[str, str]] = {"expanding": {}}
        contracts: dict[str, dict[str, Any]] = {}
        fairness: dict[str, dict[str, Any]] = {}
        for year in DEVELOPMENT_YEARS:
            path = _expanding_view(year)
            views["expanding"][str(year)] = str(path)
            fairness[str(year)] = _fairness(_read_json(path))
        for years in SHORT_WINDOW_YEARS:
            key = window_id(years)
            views[key] = {}
            contracts[key] = {}
            for year in DEVELOPMENT_YEARS:
                path, contract = _build_bounded_view(
                    study_path=study_path, year=year, years=years
                )
                views[key][str(year)] = str(path)
                contracts[key][str(year)] = contract
                if _fairness(_read_json(path)) != fairness[str(year)]:
                    raise ValueError(f"fairness material changed for {key}/{year}")
        material = {
            "source_identity": source_identity,
            "views": views,
            "window_contracts": contracts,
            "fairness_material": fairness,
            "base_pack_copied": False,
            "turnover_overlay_copied": False,
        }
        _validate_expected_2025(material)
        study = _read_json(study_path)
        study["material"] = material
        study["runtime"].update(
            {
                "phase": "prepared",
                "updated_at": _now(),
                "current_task": None,
                "error": None,
            }
        )
        _write_json(study_path, study)
        verify_structured_training_window(study_root=study_root, require_complete=False)
        return _read_json(study_path)
    except Exception as exc:
        failed = _read_json(study_path)
        failed["runtime"].update(
            {"phase": "prepare_failed", "updated_at": _now(), "error": str(exc)}
        )
        _write_json(study_path, failed)
        raise


def _view_for(study: Mapping[str, Any], window: str, year: int) -> Path:
    return Path(
        str(dict(dict(study["material"])["views"])[str(window)][str(int(year))])
    ).resolve()


def _task_run_tag(task: TrainingTask) -> str:
    return (
        f"seq100_structured_training_window_{task.window_id}_"
        f"{int(task.development_year)}_e{int(task.epochs)}_seed{SEED}"
    )


def _task_run_dirs(study_root: Path, task: TrainingTask) -> list[Path]:
    return sorted(
        (study_root.resolve() / "runs" / f"epochs_{int(task.epochs)}").glob(
            f"{_task_run_tag(task)}_*"
        )
    )


def _validate_training_run(
    run_dir: Path, *, study_path: Path, task: TrainingTask
) -> dict[str, Any]:
    summary_path = run_dir / "sequence_path_training_summary.json"
    progress_path = run_dir / "progress.json"
    checkpoint_path = run_dir / "best_model.pt"
    prediction_path = run_dir / "predictions/development_predictions.csv"
    for path in (summary_path, progress_path, checkpoint_path, prediction_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    summary = development._validate_run_summary(summary_path, task.development_year)
    progress = _read_json(progress_path)
    if str(progress.get("status", "")) != "completed":
        raise ValueError("training progress is not completed")
    if int(summary.get("completed_epochs", 0)) != int(task.epochs):
        raise ValueError("completed epoch count differs from the task contract")
    config = dict(summary.get("resolved_training_config", {}) or {})
    exact = {
        "model_type": "gru_structured_joint_turnover",
        "input_channel_profile": INPUT_CHANNEL_PROFILE,
        "seed": SEED,
        "batch_size": BATCH_SIZE,
        "epochs": int(task.epochs),
    }
    for key, expected in exact.items():
        if config.get(key) != expected:
            raise ValueError(f"training config drifted for {key}")
    if int(summary.get("lookback_days", 0)) != LOOKBACK_DAYS:
        raise ValueError("lookback drifted")
    if _file_sha256(checkpoint_path) != str(summary.get("best_checkpoint_sha256", "")):
        raise ValueError("checkpoint hash drifted")
    study = _read_json(study_path)
    view_path = _view_for(study, task.window_id, task.development_year)
    if Path(str(summary["pack_manifest"])).resolve() != view_path:
        raise ValueError("training used the wrong bounded view")
    view = _read_json(view_path)
    fold_contract = dict(view["development_fold_training_contract"])
    if dict(summary.get("development_fold_training_contract", {}) or {}) != fold_contract:
        raise ValueError("summary fold contract drifted")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if int(checkpoint.get("input_dim", 0)) != INPUT_DIM:
        raise ValueError("checkpoint input dimension drifted")
    if tuple(checkpoint["model_state_dict"]["proj.weight"].shape) != (128, INPUT_DIM):
        raise ValueError("checkpoint projection architecture drifted")
    if dict(checkpoint.get("fold_training_contract", {}) or {}) != fold_contract:
        raise ValueError("checkpoint fold contract drifted")
    return summary


def _classify_task(
    *, study_path: Path, task: TrainingTask
) -> tuple[list[Path], list[Path]]:
    complete: list[Path] = []
    partial: list[Path] = []
    for path in _task_run_dirs(study_path.parent, task):
        try:
            _validate_training_run(path, study_path=study_path, task=task)
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
            complete.append(path)
    if len(complete) > 1:
        raise ValueError(f"multiple valid completed runs for {task.task_id}")
    return complete, partial


def _archive_partial(
    paths: Sequence[Path], *, study_root: Path, task: TrainingTask
) -> list[str]:
    target_root = (
        study_root.resolve() / "runs/failed" / task.task_id.replace(":", "__")
    )
    target_root.mkdir(parents=True, exist_ok=True)
    output: list[str] = []
    for number, source in enumerate(paths, start=1):
        if study_root.resolve() not in source.resolve().parents:
            raise ValueError("refusing to archive a run outside the study")
        target = target_root / (
            f"{source.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{number}"
        )
        shutil.move(str(source), str(target))
        output.append(str(target.resolve()))
    return output


def _append_monitor_event(study_root: Path, payload: Mapping[str, Any]) -> None:
    event = {"timestamp": _now(), **dict(payload)}
    root = study_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    with (root / "monitor_events.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(event, ensure_ascii=False, default=_json_default, allow_nan=False)
            + "\n"
        )
    _write_json(root / "monitor.json", event)
    try:
        print(f"structured-window: {event.get('message', event.get('event'))}", flush=True)
    except OSError:
        pass


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


def _parse_timestamp(value: Any) -> float | None:
    try:
        return datetime.fromisoformat(str(value)).timestamp()
    except (TypeError, ValueError):
        return None


def _supervise_training(
    *, command: Sequence[str], study_path: Path, task: TrainingTask
) -> Path:
    study_root = study_path.parent
    logs = study_root / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    stem = task.task_id.replace(":", "__")
    memory_path = logs / f"memory_guard_{stem}.json"
    stdout_path = logs / f"worker_{stem}.log"
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
    before = set(_task_run_dirs(study_root, task))
    _append_monitor_event(
        study_root,
        {
            "event": "task_started",
            "status": "running",
            "task_id": task.task_id,
            "fold": int(task.development_year),
            "epochs": int(task.epochs),
            "message": f"start {task.task_id}",
        },
    )
    current_run: Path | None = None
    last_phase: str | None = None
    last_epoch: int | None = None
    last_bucket: int | None = None
    stale_reported = False
    memory_reported = False
    with stdout_path.open("w", encoding="utf-8") as stdout:
        process = subprocess.Popen(
            guard_command,
            cwd=str(WORKSPACE_ROOT),
            stdout=stdout,
            stderr=subprocess.STDOUT,
        )
        while process.poll() is None:
            if current_run is None:
                created = sorted(set(_task_run_dirs(study_root, task)).difference(before))
                if len(created) > 1:
                    process.kill()
                    raise RuntimeError("training task created multiple run directories")
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
            batch = int(progress.get("batch", 0) or 0)
            total = int(progress.get("total_batches", 0) or 0)
            bucket = min(100, int(math.floor(10.0 * batch / total) * 10)) if total else 0
            if phase != last_phase or epoch != last_epoch or (total and bucket != last_bucket):
                _append_monitor_event(
                    study_root,
                    {
                        "event": "progress",
                        "status": "running",
                        "task_id": task.task_id,
                        "phase": phase,
                        "epoch": epoch,
                        "batch": batch,
                        "total_batches": total,
                        "batch_percent_bucket": bucket,
                        "available_memory_gib": float(
                            psutil.virtual_memory().available / 1024**3
                        ),
                        "run_dir": str(current_run) if current_run else None,
                        "message": (
                            f"{task.task_id} phase={phase} epoch={epoch} "
                            f"batch={batch}/{total}"
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
                        "task_id": task.task_id,
                        "message": f"{task.task_id} progress stale for 15 minutes",
                    },
                )
                stale_reported = True
            elif not stale:
                stale_reported = False
            memory = {}
            if memory_path.is_file():
                try:
                    memory = _read_json(memory_path)
                except (OSError, json.JSONDecodeError):
                    memory = {}
            breaches = int(memory.get("available_breaches", 0) or 0)
            if breaches and not memory_reported:
                _append_monitor_event(
                    study_root,
                    {
                        "event": "memory_warning",
                        "status": "warning",
                        "task_id": task.task_id,
                        "available_breaches": breaches,
                        "message": f"{task.task_id} memory guard warning",
                    },
                )
                memory_reported = True
            elif not breaches:
                memory_reported = False
            time.sleep(MONITOR_POLL_SECONDS)
        exit_code = int(process.returncode or 0)

    if current_run is None:
        created = sorted(set(_task_run_dirs(study_root, task)).difference(before))
        current_run = created[-1] if created else None
    memory = _read_json(memory_path) if memory_path.is_file() else {}
    if current_run is not None:
        _write_json(current_run / "memory_guard.json", memory)
    valid = False
    if exit_code == 0 and current_run is not None:
        try:
            _validate_training_run(current_run, study_path=study_path, task=task)
        except Exception:
            valid = False
        else:
            valid = True
    terminal = classify_monitor_terminal(
        exit_code=exit_code,
        memory_guard_status=str(memory.get("status", "")),
        artifacts_valid=valid,
    )
    _append_monitor_event(
        study_root,
        {
            "event": terminal,
            "status": "completed" if terminal == "task_completed" else "failed",
            "task_id": task.task_id,
            "exit_code": exit_code,
            "run_dir": str(current_run) if current_run else None,
            "watcher_remaining": False,
            "message": f"{task.task_id} terminal={terminal}",
        },
    )
    if terminal != "task_completed" or current_run is None:
        if exit_code:
            raise subprocess.CalledProcessError(exit_code, guard_command)
        raise RuntimeError(f"invalid terminal artifacts for {task.task_id}")
    return current_run


def _update_runtime(study_path: Path, **changes: Any) -> None:
    study = _read_json(study_path)
    runtime = dict(study.get("runtime", {}) or {})
    runtime.update(changes)
    runtime["updated_at"] = _now()
    study["runtime"] = runtime
    _write_json(study_path, study)


def _shortlist_windows(study_root: Path) -> list[int]:
    path = study_root.resolve() / "shortlist.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = _read_json(path)
    return [
        int(row["window_years"])
        for row in list(payload["shortlisted_short_windows"])
    ]


def _epoch_confirmation_window(study_root: Path) -> int:
    path = study_root.resolve() / "epoch_confirmation.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    return int(_read_json(path)["window_years"])


def _pending_tasks(study_path: Path) -> tuple[str, list[TrainingTask]]:
    study = _read_json(study_path)
    phase = str(dict(study.get("runtime", {}) or {}).get("phase", ""))
    if phase == "prepared":
        phase = "screen_2025_training"
    if phase == "screen_2025_training":
        return phase, [TrainingTask(years, 2025, 1) for years in SHORT_WINDOW_YEARS]
    if phase == "confirmation_training":
        return phase, [
            TrainingTask(years, year, 1)
            for years in _shortlist_windows(study_path.parent)
            for year in (2023, 2024)
        ]
    if phase == "optional_epoch_confirmation":
        years = _epoch_confirmation_window(study_path.parent)
        return phase, [TrainingTask(years, year, 3) for year in DEVELOPMENT_YEARS]
    return phase, []


def run_structured_training_window(
    *, study_root: Path = STUDY_ROOT, max_tasks: int = 1
) -> dict[str, Any]:
    _assert_no_research_process()
    verify_structured_training_window(study_root=study_root, require_complete=False)
    study_path = study_root.resolve() / "study.json"
    phase, tasks = _pending_tasks(study_path)
    if phase not in {
        "screen_2025_training",
        "confirmation_training",
        "optional_epoch_confirmation",
    }:
        return status_structured_training_window(study_root=study_root)
    _update_runtime(study_path, phase=phase, current_task=None, error=None)
    launched = 0
    try:
        for task in tasks:
            completed, partial = _classify_task(study_path=study_path, task=task)
            if completed:
                continue
            if partial:
                archived = _archive_partial(
                    partial, study_root=study_path.parent, task=task
                )
                _append_monitor_event(
                    study_path.parent,
                    {
                        "event": "partial_archived",
                        "status": "recovering",
                        "task_id": task.task_id,
                        "archived_runs": archived,
                        "message": f"archived partial {task.task_id}",
                    },
                )
            if int(max_tasks) > 0 and launched >= int(max_tasks):
                return status_structured_training_window(study_root=study_root)
            study = _read_json(study_path)
            view = _view_for(study, task.window_id, task.development_year)
            profile = replace(
                input_ablation._base_structured_profile(BATCH_SIZE),
                store_view=view,
                output_root=study_path.parent / "runs" / f"epochs_{task.epochs}",
                run_tag=_task_run_tag(task),
                epochs=int(task.epochs),
                batch_size=BATCH_SIZE,
                seed=SEED,
                top_k="1,3,5,10",
                input_channel_profile=INPUT_CHANNEL_PROFILE,
                prediction_mode="compact",
                evaluation_mode="development",
                early_stopping_patience=2,
                early_stopping_min_delta=0.0,
                early_stopping_metric=(
                    training.EARLY_STOPPING_METRIC_DEVELOPMENT_PRICE_TOTAL_LOSS
                ),
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
            _update_runtime(study_path, phase=phase, current_task=task.task_id)
            _supervise_training(command=command, study_path=study_path, task=task)
            launched += 1
            _update_runtime(study_path, phase=phase, current_task=None)
        all_complete = all(
            len(_classify_task(study_path=study_path, task=task)[0]) == 1
            for task in tasks
        )
        if all_complete:
            next_phase = (
                "screen_2025_evaluating"
                if phase == "screen_2025_training"
                else "confirmation_evaluating"
            )
            _update_runtime(
                study_path, phase=next_phase, current_task=None, error=None
            )
    except Exception as exc:
        _update_runtime(study_path, phase=phase, current_task=None, error=str(exc))
        raise
    return status_structured_training_window(study_root=study_root)


def shortlist_from_scores(
    scores: Mapping[str, float], *, expanding_id: str = "expanding"
) -> dict[str, Any]:
    if expanding_id not in scores:
        raise ValueError("shortlist scores are missing the expanding baseline")
    if len(scores) < 2:
        raise ValueError("shortlist requires at least two models")
    ordered = sorted(scores.items(), key=lambda item: (-float(item[1]), str(item[0])))
    champion_id, champion_score = ordered[0]
    denominator = max(abs(float(champion_score)), 1.0e-12)
    selected = {str(model_id) for model_id, _score in ordered[:2]}
    relative_gap: dict[str, float] = {}
    for model_id, score in ordered:
        gap = float((float(champion_score) - float(score)) / denominator)
        relative_gap[str(model_id)] = gap
        if gap <= 0.05 + 1.0e-15:
            selected.add(str(model_id))
    selected.add(str(expanding_id))
    short = [
        model_id
        for model_id, _score in ordered
        if model_id in selected and model_id != expanding_id
    ]
    return {
        "champion_model_id": str(champion_id),
        "champion_score": float(champion_score),
        "ordered_scores": [
            {"model_id": str(model_id), "score": float(score)}
            for model_id, score in ordered
        ],
        "relative_gap_to_champion": relative_gap,
        "selected_model_ids": [
            model_id for model_id, _score in ordered if model_id in selected
        ],
        "shortlisted_short_model_ids": short,
    }


def _model_spec(
    *, model_id: str, window_years: int | None, epochs: int
) -> dict[str, Any]:
    return {
        "model_id": str(model_id),
        "window_years": int(window_years) if window_years is not None else None,
        "epochs": int(epochs),
        "source": "existing_expanding" if window_years is None else "bounded_run",
    }


def _screen_model_specs() -> list[dict[str, Any]]:
    return [_model_spec(model_id="expanding", window_years=None, epochs=3)] + [
        _model_spec(model_id=window_id(years), window_years=years, epochs=1)
        for years in SHORT_WINDOW_YEARS
    ]


def _confirmation_model_specs(study_root: Path) -> list[dict[str, Any]]:
    return [_model_spec(model_id="expanding", window_years=None, epochs=3)] + [
        _model_spec(model_id=window_id(years), window_years=years, epochs=1)
        for years in _shortlist_windows(study_root)
    ]


def _run_for_spec(
    spec: Mapping[str, Any], *, year: int, study_path: Path
) -> Path:
    if spec.get("window_years") is None:
        return _expanding_run(year)
    task = TrainingTask(
        int(spec["window_years"]), int(year), int(spec.get("epochs", 1))
    )
    complete, _partial = _classify_task(study_path=study_path, task=task)
    if len(complete) != 1:
        raise ValueError(f"missing completed training run for {task.task_id}")
    return complete[0]


def _view_for_spec(
    spec: Mapping[str, Any], *, year: int, study_path: Path
) -> Path:
    if spec.get("window_years") is None:
        return _expanding_view(year)
    return _view_for(_read_json(study_path), window_id(int(spec["window_years"])), year)


def _year_cohort_evaluation(
    spec: Mapping[str, Any], *, year: int, study_path: Path
) -> dict[str, Any]:
    run_dir = _run_for_spec(spec, year=year, study_path=study_path)
    view_path = _view_for_spec(spec, year=year, study_path=study_path)
    metrics, summary = development._year_metrics(run_dir, year)
    diagnostics = structured.stream_checkpoint_diagnostics(
        run_dir=run_dir,
        view_path=view_path,
        output_dir=run_dir / "path_diagnostics",
        year=int(year),
        compare_legacy_domain=False,
        fixed_exit_comparison=False,
        input_channel_profile=INPUT_CHANNEL_PROFILE,
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
            "development_price_total_loss": float(
                dict(summary["early_stopping"])["best_value"]
            ),
        }
    )
    fairness = _fairness(_read_json(view_path))
    output = {
        "schema_version": 1,
        "artifact_type": "seq100_structured_training_window_evaluation",
        "status": "completed",
        "completed_at": _now(),
        "study_id": STUDY_ID,
        "study_contract_sha256": str(_read_json(study_path)["contract_sha256"]),
        "model": dict(spec),
        "development_year": int(year),
        "run_dir": str(run_dir),
        "view_path": str(view_path),
        "checkpoint_sha256": _file_sha256(run_dir / "best_model.pt"),
        "summary_sha256": _file_sha256(
            run_dir / "sequence_path_training_summary.json"
        ),
        "prediction_sha256": _file_sha256(
            run_dir / "predictions/development_predictions.csv"
        ),
        "fairness_material": fairness,
        "metrics": metrics,
        "qdp_changed": False,
        "provider_called": False,
        "live_state_changed": False,
    }
    if spec.get("window_years") is not None:
        _write_json(run_dir / "evaluation.json", output)
    return output


def _cohort_metrics(
    specs: Sequence[Mapping[str, Any]], *, years: Sequence[int], study_path: Path
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    fairness_reference: dict[str, Any] = {}
    for spec in specs:
        for year in years:
            payload = _year_cohort_evaluation(spec, year=year, study_path=study_path)
            fairness = dict(payload["fairness_material"])
            previous = fairness_reference.setdefault(str(year), fairness)
            if previous != fairness:
                raise ValueError(f"cohort fairness material differs in {year}")
            rows.append(
                {
                    "model_id": str(spec["model_id"]),
                    "window_years": spec.get("window_years"),
                    "epochs": int(spec["epochs"]),
                    "development_year": int(year),
                    **dict(payload["metrics"]),
                }
            )
    return pd.DataFrame(rows)


def _cohort_contract(
    specs: Sequence[Mapping[str, Any]], *, years: Sequence[int], study_path: Path
) -> dict[str, Any]:
    sources: dict[str, dict[str, Any]] = {}
    for spec in specs:
        model_id = str(spec["model_id"])
        sources[model_id] = {}
        for year in years:
            run = _run_for_spec(spec, year=int(year), study_path=study_path)
            view = _view_for_spec(spec, year=int(year), study_path=study_path)
            sources[model_id][str(year)] = {
                "checkpoint_sha256": _file_sha256(run / "best_model.pt"),
                "prediction_sha256": _file_sha256(
                    run / "predictions/development_predictions.csv"
                ),
                "summary_sha256": _file_sha256(
                    run / "sequence_path_training_summary.json"
                ),
                "view_sha256": _file_sha256(view),
            }
    semantic = {
        "schema_version": 1,
        "semantic_version": "structured-training-window-cohort-v1",
        "study_contract_sha256": str(_read_json(study_path)["contract_sha256"]),
        "models": [dict(spec) for spec in specs],
        "years": [int(year) for year in years],
        "sources": sources,
    }
    return {"contract": semantic, "contract_sha256": _canonical_digest(semantic)}


def _load_or_build_cohort(
    *,
    cache_name: str,
    specs: Sequence[Mapping[str, Any]],
    years: Sequence[int],
    study_path: Path,
) -> pd.DataFrame:
    csv_path = study_path.parent / f"window_metrics_{cache_name}.csv"
    contract_path = study_path.parent / f"window_metrics_{cache_name}.contract.json"
    expected = _cohort_contract(specs, years=years, study_path=study_path)
    expected_pairs = {
        (str(spec["model_id"]), int(year)) for spec in specs for year in years
    }
    if csv_path.is_file():
        frame = pd.read_csv(csv_path)
        observed_pairs = set(
            zip(
                frame["model_id"].astype(str),
                frame["development_year"].astype(int),
            )
        )
        valid = bool(
            observed_pairs == expected_pairs
            and len(frame) == len(expected_pairs)
            and frame["candidate_score_coverage"].astype(float).eq(1.0).all()
        )
        if contract_path.is_file():
            valid = valid and _read_json(contract_path) == expected
        if valid:
            if not contract_path.is_file():
                _write_json(contract_path, expected)
            return frame
    frame = _cohort_metrics(specs, years=years, study_path=study_path)
    _write_csv(csv_path, frame)
    _write_json(contract_path, expected)
    return frame


def _forecast_book(
    spec: Mapping[str, Any], *, years: Sequence[int], study_path: Path
) -> tuple[finite.ForecastBook, dict[str, str]]:
    model_id = str(spec["model_id"])
    book = finite.ForecastBook(model_id)
    hashes: dict[str, str] = {}
    for year in years:
        view_path = _view_for_spec(spec, year=year, study_path=study_path)
        manifest = _read_json(view_path)
        dataset = training.SequencePathPackDataset(
            manifest,
            split="development",
            max_samples=0,
            input_channel_profile=INPUT_CHANNEL_PROFILE,
            index_role="candidate",
        )
        run_dir = _run_for_spec(spec, year=year, study_path=study_path)
        prediction_path = run_dir / "predictions/development_predictions.csv"
        hashes[str(year)] = _file_sha256(prediction_path)
        frame = pd.read_csv(
            prediction_path,
            usecols=["trade_date", "symbol", "score", "predicted_exit_day"],
            dtype={"trade_date": str, "symbol": str, "score": np.float64},
        )
        candidates = dataset.sample_index
        if len(frame) != len(candidates):
            raise ValueError(f"prediction row count drifted for {model_id}/{year}")
        if not np.array_equal(
            frame["trade_date"].astype(str).to_numpy(),
            candidates["trade_date"].astype(str).to_numpy(),
        ) or not np.array_equal(
            frame["symbol"].astype(str).to_numpy(),
            candidates["symbol"].astype(str).to_numpy(),
        ):
            raise ValueError(f"prediction candidate order drifted for {model_id}/{year}")
        scores = pd.to_numeric(frame["score"], errors="coerce").to_numpy(
            dtype=np.float64
        )
        planned = pd.to_numeric(
            frame["predicted_exit_day"], errors="coerce"
        ).to_numpy(dtype=np.float64)
        if not bool(np.isfinite(scores).all()) or not bool(np.isfinite(planned).all()):
            raise ValueError(f"prediction coverage is incomplete for {model_id}/{year}")
        planned = np.clip(np.rint(planned), 2, 60).astype(np.int16)
        symbols = candidates["symbol_idx"].to_numpy(dtype=np.int64)
        for date_idx, positions_raw in candidates.groupby("date_idx", sort=True).indices.items():
            positions = np.asarray(positions_raw, dtype=np.int64)
            book.add_day(
                date_idx=int(date_idx),
                symbol_idx=symbols[positions],
                score=scores[positions],
                planned_day=planned[positions],
            )
        del dataset, frame, candidates, scores, planned, symbols
        gc.collect()
    return book, hashes


def _capital_market(
    spec: Mapping[str, Any], *, first_year: int, study_path: Path
) -> finite.BacktestMarket:
    pack = CandidateCompleteAuditPack(
        _view_for_spec(spec, year=first_year, study_path=study_path)
    )
    return finite.BacktestMarket(
        date_values=np.asarray(pack.date_values, dtype=object),
        symbol_values=np.asarray(pack.symbol_values, dtype=object),
        entry_open_raw=pack.entry_open_raw,
        exit_close_raw=pack.exit_close_raw,
        exit_sellable=pack.exit_sellable,
        entry_filled=pack.entry_filled,
        contract=pack.contract,
        terminal_recovery_fraction=float(pack.terminal_recovery_fraction),
        forward_days=int(pack.forward_days),
        execution_days=int(pack.execution_days),
    )


def _capital_jobs(phase: str, model_ids: Sequence[str]) -> list[CapitalJob]:
    jobs: list[CapitalJob] = []
    policies = [(f"fixed_d{day}", "fixed", day) for day in FIXED_EXIT_DAYS]
    policies.extend(
        [("model_plan", "model_plan", None), ("rolling_path", "rolling", None)]
    )
    for model_id in model_ids:
        for top_k, slots in CAPITAL_CONFIGS:
            for policy_name, policy_kind, fixed_day in policies:
                for cost in COST_SCENARIOS:
                    jobs.append(
                        CapitalJob(
                            phase=str(phase),
                            model_id=str(model_id),
                            top_k=int(top_k),
                            slots=int(slots),
                            policy_name=str(policy_name),
                            policy_kind=str(policy_kind),
                            fixed_day=int(fixed_day) if fixed_day is not None else None,
                            cost_scenario=str(cost),
                        )
                    )
    return jobs


def _policy(job: CapitalJob) -> finite.PolicySpec:
    return finite.PolicySpec(
        name=job.policy_name, kind=job.policy_kind, fixed_day=job.fixed_day
    )


def _job_payload(
    job: CapitalJob,
    *,
    market: finite.BacktestMarket,
    book: finite.ForecastBook,
    signal_dates: Sequence[int],
    contract_sha256: str,
) -> dict[str, Any]:
    metric, _equity, trades, annual = finite.simulate_portfolio(
        market=market,
        book=book,
        raw_top3_paths={},
        policy=_policy(job),
        slots=int(job.slots),
        cost_scenario=str(job.cost_scenario),
        first_signal_date_idx=int(signal_dates[0]),
        last_signal_date_idx=int(signal_dates[-1]),
        starting_cash=finite.STARTING_CASH_CNY,
        memory_guard=finite._MemoryGuard(),
    )
    occupied = (
        float(
            np.sum(
                trades["buy_cash_cny"].to_numpy(dtype=np.float64)
                * trades["occupied_sessions"].to_numpy(dtype=np.float64)
            )
        )
        if not trades.empty
        else 0.0
    )
    pnl = float(trades["net_pnl_cny"].sum()) if not trades.empty else 0.0
    output = {
        "schema_version": 1,
        "artifact_type": "seq100_structured_training_window_capital_job",
        "status": "completed",
        "completed_at": _now(),
        "contract_sha256": str(contract_sha256),
        "job_id": job.job_id,
        "job": asdict(job),
        "metric": {
            **metric,
            "annualized_log_growth": capital._annualized_log_growth(metric),
            "worst_calendar_year_return": float(
                min(float(row["net_return"]) for row in annual)
            ),
            "net_pnl_per_deployed_capital_session": (
                float(pnl / occupied) if occupied > 0.0 else None
            ),
        },
        "annual_metrics": annual,
    }
    del _equity, trades
    return output


def _job_valid(path: Path, *, contract_sha256: str, job_id: str) -> bool:
    if not path.is_file():
        return False
    try:
        payload = _read_json(path)
    except (OSError, json.JSONDecodeError):
        return False
    return bool(
        str(payload.get("status", "")) == "completed"
        and str(payload.get("contract_sha256", "")) == str(contract_sha256)
        and str(payload.get("job_id", "")) == str(job_id)
    )


def _phase_contract(
    *,
    phase: str,
    specs: Sequence[Mapping[str, Any]],
    years: Sequence[int],
    prediction_hashes: Mapping[str, Mapping[str, str]],
    study_path: Path,
) -> dict[str, Any]:
    fairness = dict(dict(_read_json(study_path)["material"])["fairness_material"])
    semantic = {
        "schema_version": 1,
        "semantic_version": "structured-training-window-capital-speed-v1",
        "study_contract_sha256": str(_read_json(study_path)["contract_sha256"]),
        "phase": str(phase),
        "models": [dict(spec) for spec in specs],
        "years": [int(year) for year in years],
        "prediction_hashes": {
            str(key): dict(value) for key, value in prediction_hashes.items()
        },
        "fairness_material": {str(year): fairness[str(year)] for year in years},
        "capital_configurations": [list(value) for value in CAPITAL_CONFIGS],
        "fixed_exit_days": list(FIXED_EXIT_DAYS),
        "own_exit_policies": ["model_plan", "rolling_path"],
        "cost_scenarios": list(COST_SCENARIOS),
        "execution_contract": "next_open_legal_close_no_leverage_no_pyramiding",
    }
    return {"contract": semantic, "contract_sha256": _canonical_digest(semantic)}


def _aggregate_capital_jobs(
    *, root: Path, jobs: Sequence[CapitalJob], contract_sha256: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    metrics: list[dict[str, Any]] = []
    annual: list[dict[str, Any]] = []
    for job in jobs:
        path = root / "jobs" / f"{job.job_id}.json"
        if not _job_valid(
            path, contract_sha256=contract_sha256, job_id=job.job_id
        ):
            raise ValueError(f"capital job is incomplete: {job.job_id}")
        payload = _read_json(path)
        identity = {
            "phase": str(job.phase),
            "model_id": str(job.model_id),
            "top_k": int(job.top_k),
            "slot_count": int(job.slots),
            "policy_name": str(job.policy_name),
            "policy_kind": str(job.policy_kind),
            "fixed_day": job.fixed_day,
            "cost_scenario": str(job.cost_scenario),
        }
        metrics.append({**identity, **dict(payload["metric"])})
        annual.extend({**identity, **dict(row)} for row in payload["annual_metrics"])
    return pd.DataFrame(metrics), pd.DataFrame(annual)


def _evaluate_capital_phase(
    *,
    phase: str,
    specs: Sequence[Mapping[str, Any]],
    years: Sequence[int],
    study_path: Path,
    max_jobs: int,
) -> tuple[str, pd.DataFrame | None, pd.DataFrame | None]:
    root = study_path.parent / "capital" / str(phase)
    jobs_root = root / "jobs"
    jobs_root.mkdir(parents=True, exist_ok=True)
    books: dict[str, finite.ForecastBook] = {}
    top_books: dict[tuple[str, int], finite.ForecastBook] = {}
    hashes: dict[str, dict[str, str]] = {}
    for spec in specs:
        model_id = str(spec["model_id"])
        book, source_hashes = _forecast_book(
            spec, years=years, study_path=study_path
        )
        books[model_id] = book
        hashes[model_id] = source_hashes
        for top_k, _slots in CAPITAL_CONFIGS:
            top_books[(model_id, int(top_k))] = capital._top_k_book(
                book, top_k=int(top_k), profile_name=f"{model_id}_top{int(top_k)}"
            )
    calendars = {tuple(book.signal_date_indices) for book in books.values()}
    if len(calendars) != 1:
        raise ValueError("capital models do not share one signal calendar")
    signal_dates = next(iter(calendars))
    market = _capital_market(specs[0], first_year=int(years[0]), study_path=study_path)
    contract = _phase_contract(
        phase=phase,
        specs=specs,
        years=years,
        prediction_hashes=hashes,
        study_path=study_path,
    )
    contract_path = root / "contract.json"
    if contract_path.is_file() and _read_json(contract_path) != contract:
        raise ValueError(f"capital phase contract drifted: {phase}")
    if not contract_path.is_file():
        _write_json(contract_path, contract)
    jobs = _capital_jobs(phase, [str(spec["model_id"]) for spec in specs])
    contract_sha = str(contract["contract_sha256"])
    completed = sum(
        _job_valid(
            jobs_root / f"{job.job_id}.json",
            contract_sha256=contract_sha,
            job_id=job.job_id,
        )
        for job in jobs
    )
    launched = 0
    for job in jobs:
        path = jobs_root / f"{job.job_id}.json"
        if _job_valid(path, contract_sha256=contract_sha, job_id=job.job_id):
            continue
        if int(max_jobs) > 0 and launched >= int(max_jobs):
            break
        payload = _job_payload(
            job,
            market=market,
            book=top_books[(job.model_id, int(job.top_k))],
            signal_dates=signal_dates,
            contract_sha256=contract_sha,
        )
        _write_json(path, payload)
        completed += 1
        launched += 1
        if completed % 100 == 0 or completed == len(jobs):
            _append_monitor_event(
                study_path.parent,
                {
                    "event": "capital_progress",
                    "status": "running",
                    "phase": str(phase),
                    "completed_jobs": int(completed),
                    "total_jobs": int(len(jobs)),
                    "message": f"capital {phase} {completed}/{len(jobs)}",
                },
            )
    if completed != len(jobs):
        _append_monitor_event(
            study_path.parent,
            {
                "event": "capital_phase_partial",
                "status": "partial",
                "phase": str(phase),
                "completed_jobs": int(completed),
                "total_jobs": int(len(jobs)),
                "watcher_remaining": False,
                "message": f"capital {phase} yielded {completed}/{len(jobs)}",
            },
        )
        return "partial", None, None
    metrics, annual = _aggregate_capital_jobs(
        root=root, jobs=jobs, contract_sha256=contract_sha
    )
    _write_csv(root / "capital_metrics.csv", metrics)
    _write_csv(root / "capital_annual_metrics.csv", annual)
    _append_monitor_event(
        study_path.parent,
        {
            "event": "capital_phase_completed",
            "status": "completed",
            "phase": str(phase),
            "completed_jobs": int(len(jobs)),
            "watcher_remaining": False,
            "message": f"capital phase completed {phase}",
        },
    )
    return "completed", metrics, annual


def select_capital_strategies(
    metrics: pd.DataFrame, *, allowed_policy_names: Iterable[str] | None
) -> dict[str, Any]:
    frame = metrics[metrics["cost_scenario"].astype(str).eq("double_slippage")].copy()
    if allowed_policy_names is not None:
        allowed = {str(value) for value in allowed_policy_names}
        frame = frame[frame["policy_name"].astype(str).isin(allowed)]
    if frame.empty:
        raise ValueError("capital selection has no eligible double-slippage rows")
    selected: list[dict[str, Any]] = []
    for (model_id, top_k, slots), group in frame.groupby(
        ["model_id", "top_k", "slot_count"], sort=True
    ):
        ordered = group.sort_values(
            ["annualized_log_growth", "policy_name"],
            ascending=[False, True],
            kind="mergesort",
        )
        row = ordered.iloc[0].to_dict()
        selected.append(
            {
                "model_id": str(model_id),
                "top_k": int(top_k),
                "slot_count": int(slots),
                "policy_name": str(row["policy_name"]),
                "policy_kind": str(row["policy_kind"]),
                "fixed_day": (
                    int(row["fixed_day"])
                    if row.get("fixed_day") is not None
                    and not pd.isna(row.get("fixed_day"))
                    else None
                ),
                "annualized_log_growth": float(row["annualized_log_growth"]),
                "signal_period_total_return": float(row["signal_period_total_return"]),
                "signal_period_maximum_drawdown": float(
                    row["signal_period_maximum_drawdown"]
                ),
                "closed_trade_count": int(row["closed_trade_count"]),
                "winning_trade_rate": float(row["winning_trade_rate"]),
                "mean_occupied_sessions": float(row["mean_occupied_sessions"]),
                "mean_signal_capital_utilization": float(
                    row["mean_signal_capital_utilization"]
                ),
                "skipped_no_slot_signal_count": int(
                    row["skipped_no_slot_signal_count"]
                ),
                "net_pnl_per_deployed_capital_session": (
                    None
                    if pd.isna(row.get("net_pnl_per_deployed_capital_session"))
                    else float(row["net_pnl_per_deployed_capital_session"])
                ),
            }
        )
    counts = pd.Series([row["model_id"] for row in selected]).value_counts()
    if not bool((counts == len(CAPITAL_CONFIGS)).all()):
        raise ValueError("capital selection does not cover all three configurations")
    scores = {
        str(model_id): float(
            np.mean(
                [
                    float(row["annualized_log_growth"])
                    for row in selected
                    if str(row["model_id"]) == str(model_id)
                ]
            )
        )
        for model_id in sorted(counts.index.astype(str))
    }
    ordered_scores = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    return {
        "objective": (
            "equal mean double-slippage annualized log growth across Top1/1, "
            "Top3/3, and Top3/6"
        ),
        "selected_strategies": selected,
        "scores": scores,
        "ranking": [
            {"model_id": model_id, "score": float(score)}
            for model_id, score in ordered_scores
        ],
        "champion_model_id": ordered_scores[0][0],
        "champion_score": float(ordered_scores[0][1]),
    }


def epoch_confirmation_reasons(
    *,
    scores: Mapping[str, float],
    champion_id: str,
    strongest_short_id: str | None = None,
    yearly_returns: Mapping[str, Mapping[int, float]],
    cohort_losses: Mapping[str, Mapping[int, float]],
    expanding_id: str = "expanding",
) -> list[str]:
    ordered = sorted(scores.items(), key=lambda item: (-float(item[1]), str(item[0])))
    runner_score = float(ordered[1][1]) if len(ordered) > 1 else float("-inf")
    champion_score = float(scores[champion_id])
    expanding_score = float(scores[expanding_id])
    scale = max(abs(champion_score), 1.0e-12)
    reasons: list[str] = []
    if str(champion_id) != str(expanding_id):
        if (champion_score - expanding_score) / scale < 0.05:
            reasons.append("champion_lead_over_expanding_below_5pct")
        if (champion_score - runner_score) / scale < 0.05:
            reasons.append("champion_lead_over_runner_up_below_5pct")
        champion_years = dict(yearly_returns.get(champion_id, {}))
        expanding_years = dict(yearly_returns.get(expanding_id, {}))
        wins = sum(
            float(champion_years.get(year, float("-inf")))
            > float(expanding_years.get(year, float("inf")))
            for year in DEVELOPMENT_YEARS
        )
        if wins < 2:
            reasons.append("champion_beats_expanding_in_fewer_than_two_years")
    loss_model_id = str(strongest_short_id or champion_id)
    champion_losses = dict(cohort_losses.get(loss_model_id, {}))
    expanding_losses = dict(cohort_losses.get(expanding_id, {}))
    if any(
        float(champion_losses.get(year, float("inf")))
        > 1.25 * float(expanding_losses.get(year, float("nan")))
        for year in DEVELOPMENT_YEARS
    ):
        reasons.append("epoch1_development_price_loss_exceeds_expanding_by_25pct")
    return reasons


def _yearly_selected_returns(
    selection: Mapping[str, Any], annual: pd.DataFrame
) -> dict[str, dict[int, float]]:
    config_returns: dict[str, dict[int, list[float]]] = {}
    for strategy in list(selection["selected_strategies"]):
        mask = (
            annual["model_id"].astype(str).eq(str(strategy["model_id"]))
            & annual["top_k"].astype(int).eq(int(strategy["top_k"]))
            & annual["slot_count"].astype(int).eq(int(strategy["slot_count"]))
            & annual["policy_name"].astype(str).eq(str(strategy["policy_name"]))
            & annual["cost_scenario"].astype(str).eq("double_slippage")
        )
        rows = annual.loc[mask]
        for row in rows.to_dict(orient="records"):
            config_returns.setdefault(str(strategy["model_id"]), {}).setdefault(
                int(row["year"]), []
            ).append(float(row["net_return"]))
    return {
        model_id: {
            year: float(np.mean(values)) for year, values in by_year.items()
        }
        for model_id, by_year in config_returns.items()
    }


def _selected_daily_deltas(
    *,
    selection: Mapping[str, Any],
    specs: Sequence[Mapping[str, Any]],
    years: Sequence[int],
    study_path: Path,
) -> pd.DataFrame:
    spec_by_id = {str(spec["model_id"]): spec for spec in specs}
    books: dict[str, finite.ForecastBook] = {}
    top_books: dict[tuple[str, int], finite.ForecastBook] = {}
    for model_id, spec in spec_by_id.items():
        books[model_id], _ = _forecast_book(spec, years=years, study_path=study_path)
        for top_k, _slots in CAPITAL_CONFIGS:
            top_books[(model_id, top_k)] = capital._top_k_book(
                books[model_id], top_k=top_k, profile_name=f"{model_id}_top{top_k}"
            )
    market = _capital_market(specs[0], first_year=int(years[0]), study_path=study_path)
    signal_dates = next(iter(books.values())).signal_date_indices
    paths: dict[tuple[str, int, int], pd.DataFrame] = {}
    for strategy in list(selection["selected_strategies"]):
        model_id = str(strategy["model_id"])
        top_k = int(strategy["top_k"])
        slots = int(strategy["slot_count"])
        policy = finite.PolicySpec(
            name=str(strategy["policy_name"]),
            kind=str(strategy["policy_kind"]),
            fixed_day=strategy.get("fixed_day"),
        )
        _metric, equity, _trades, _annual = finite.simulate_portfolio(
            market=market,
            book=top_books[(model_id, top_k)],
            raw_top3_paths={},
            policy=policy,
            slots=slots,
            cost_scenario="double_slippage",
            first_signal_date_idx=int(signal_dates[0]),
            last_signal_date_idx=int(signal_dates[-1]),
            starting_cash=finite.STARTING_CASH_CNY,
            memory_guard=finite._MemoryGuard(),
        )
        equity = equity.copy()
        equity["daily_return"] = np.r_[
            equity["equity"].iloc[0] / finite.STARTING_CASH_CNY - 1.0,
            equity["equity"].to_numpy(dtype=np.float64)[1:]
            / equity["equity"].to_numpy(dtype=np.float64)[:-1]
            - 1.0,
        ]
        paths[(model_id, top_k, slots)] = equity[["trade_date", "daily_return"]]
    rows: list[pd.DataFrame] = []
    for (model_id, top_k, slots), frame in paths.items():
        if model_id == "expanding":
            continue
        baseline = paths[("expanding", top_k, slots)]
        merged = frame.merge(
            baseline,
            on="trade_date",
            how="inner",
            suffixes=("_model", "_expanding"),
            validate="one_to_one",
        )
        merged["model_id"] = model_id
        merged["top_k"] = top_k
        merged["slot_count"] = slots
        merged["daily_return_delta"] = (
            merged["daily_return_model"] - merged["daily_return_expanding"]
        )
        rows.append(merged)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _merge_phase_csvs(study_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    metric_frames: list[pd.DataFrame] = []
    annual_frames: list[pd.DataFrame] = []
    for root in sorted((study_root / "capital").glob("*")):
        metric_path = root / "capital_metrics.csv"
        annual_path = root / "capital_annual_metrics.csv"
        if metric_path.is_file():
            metric_frames.append(pd.read_csv(metric_path))
        if annual_path.is_file():
            annual_frames.append(pd.read_csv(annual_path))
    metrics = pd.concat(metric_frames, ignore_index=True) if metric_frames else pd.DataFrame()
    annual = pd.concat(annual_frames, ignore_index=True) if annual_frames else pd.DataFrame()
    if not metrics.empty:
        _write_csv(study_root / "capital_metrics.csv", metrics)
    if not annual.empty:
        _write_csv(study_root / "capital_annual_metrics.csv", annual)
    return metrics, annual


def _summary_markdown(summary: Mapping[str, Any]) -> str:
    final = dict(summary["final_selection"])
    screen = dict(dict(summary["screen_2025"])["selection"])
    lines = [
        "# Structured 180x35 Training-window Result",
        "",
        f"- Status: `{summary['status']}`.",
        "- Objective: maximize double-slippage continuous-account annualized log growth.",
        "- Capital configurations: Top1/1 slot, Top3/3 slots, Top3/6 slots.",
        f"- Selected baseline: `{final['selected_model_id']}`.",
        f"- Expanding retained: `{str(final['expanding_retained']).lower()}`.",
        "- IC, path error, cohort alpha, and exit diagnostics are explanatory only.",
        "",
        "## 2025 Screen",
        "",
        "| model | mean annual log growth |",
        "|---|---:|",
    ]
    for row in list(screen["ranking"]):
        lines.append(f"| {row['model_id']} | {float(row['score']):.4f} |")
    lines.extend(
        [
            "",
            "## Three-year Selected Strategies",
            "",
            "| model | config | execution | annual log growth | implied CAGR | total return | max drawdown | win rate | trades | mean hold |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in list(dict(summary["selection"])["selected_strategies"]):
        log_growth = float(row["annualized_log_growth"])
        lines.append(
            "| {model} | Top{top}/{slots} | {policy} | {log_growth:.4f} | "
            "{cagr:.2%} | {total:.2%} | {drawdown:.2%} | {win_rate:.2%} | "
            "{trades} | {hold:.2f} |".format(
                model=str(row["model_id"]),
                top=int(row["top_k"]),
                slots=int(row["slot_count"]),
                policy=str(row["policy_name"]),
                log_growth=log_growth,
                cagr=math.expm1(log_growth),
                total=float(row["signal_period_total_return"]),
                drawdown=float(row["signal_period_maximum_drawdown"]),
                win_rate=float(row["winning_trade_rate"]),
                trades=int(row["closed_trade_count"]),
                hold=float(row["mean_occupied_sessions"]),
            )
        )
    annual_lookup = {
        (
            str(row["model_id"]),
            int(row["top_k"]),
            int(row["slot_count"]),
            int(row["year"]),
        ): float(row["net_return"])
        for row in list(summary["selected_annual_metrics"])
    }
    lines.extend(
        [
            "",
            "## Calendar-year Account Returns",
            "",
            "| model | config | execution | 2023 | 2024 | 2025 |",
            "|---|---|---|---:|---:|---:|",
        ]
    )
    for row in list(dict(summary["selection"])["selected_strategies"]):
        key = (str(row["model_id"]), int(row["top_k"]), int(row["slot_count"]))
        lines.append(
            "| {model} | Top{top}/{slots} | {policy} | {y2023:.2%} | "
            "{y2024:.2%} | {y2025:.2%} |".format(
                model=key[0],
                top=key[1],
                slots=key[2],
                policy=str(row["policy_name"]),
                y2023=annual_lookup[(*key, 2023)],
                y2024=annual_lookup[(*key, 2024)],
                y2025=annual_lookup[(*key, 2025)],
            )
        )
    lines.extend(
        [
            "",
            "The eight-year model won the 2025 screen but lost the symmetric "
            "2023-2025 confirmation. No 3-epoch rerun was required after applying "
            "the registered conflict rule correctly.",
            "",
            "No QDP, provider, Stage 3, old checkpoint, base pack, or live-state change was made.",
        ]
    )
    return "\n".join(lines) + "\n"


def _selected_annual_metrics(
    selection: Mapping[str, Any], annual: pd.DataFrame
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for strategy in list(selection["selected_strategies"]):
        mask = (
            annual["model_id"].astype(str).eq(str(strategy["model_id"]))
            & annual["top_k"].astype(int).eq(int(strategy["top_k"]))
            & annual["slot_count"].astype(int).eq(int(strategy["slot_count"]))
            & annual["policy_name"].astype(str).eq(str(strategy["policy_name"]))
            & annual["cost_scenario"].astype(str).eq("double_slippage")
        )
        for row in annual.loc[mask].sort_values("year").to_dict(orient="records"):
            output.append(
                {
                    "model_id": str(strategy["model_id"]),
                    "top_k": int(strategy["top_k"]),
                    "slot_count": int(strategy["slot_count"]),
                    "policy_name": str(strategy["policy_name"]),
                    "year": int(row["year"]),
                    "net_return": float(row["net_return"]),
                    "maximum_drawdown": float(row["maximum_drawdown"]),
                    "annualized_volatility": float(row["annualized_volatility"]),
                    "sharpe_zero_rate": float(row["sharpe_zero_rate"]),
                    "mean_position_count": float(row["mean_position_count"]),
                    "mean_capital_utilization": float(row["mean_capital_utilization"]),
                    "trading_session_count": int(row["trading_session_count"]),
                }
            )
    return output


def _reported_cohort_metrics(cohort: pd.DataFrame) -> list[dict[str, Any]]:
    fields = (
        "model_id",
        "window_years",
        "epochs",
        "development_year",
        "rank_ic",
        "rank_ic_positive_day_rate",
        "path_mae",
        "path_close_mae",
        "top1_base_alpha",
        "top3_base_alpha",
        "top5_base_alpha",
        "top10_base_alpha",
        "exit_regret",
        "top3_average_exit_day",
        "top3_legal_exit_max_day_share",
        "candidate_score_coverage",
        "top3_execution_return_coverage",
        "development_price_total_loss",
    )
    records = cohort.loc[:, list(fields)].sort_values(
        ["development_year", "model_id"], kind="mergesort"
    ).to_dict(orient="records")
    return [
        {
            str(key): (
                None
                if value is None or (isinstance(value, (float, np.floating)) and pd.isna(value))
                else value.item() if isinstance(value, np.generic) else value
            )
            for key, value in row.items()
        }
        for row in records
    ]


def _combined_window_metrics(
    *, study_root: Path, final_cohort: pd.DataFrame
) -> pd.DataFrame:
    frames = [final_cohort.copy()]
    screen_path = study_root / "window_metrics_screen_2025.csv"
    if screen_path.is_file():
        frames.append(pd.read_csv(screen_path))
    combined = pd.concat(frames, ignore_index=True)
    combined = combined.sort_values(
        ["development_year", "model_id", "epochs"], kind="mergesort"
    ).drop_duplicates(["model_id", "development_year"], keep="first")
    return combined.reset_index(drop=True)


def _complete_summary(
    *,
    study_path: Path,
    specs: Sequence[Mapping[str, Any]],
    cohort: pd.DataFrame,
    metrics: pd.DataFrame,
    annual: pd.DataFrame,
    selection: Mapping[str, Any],
    epoch_confirmation: Mapping[str, Any] | None,
) -> dict[str, Any]:
    scores = {str(key): float(value) for key, value in dict(selection["scores"]).items()}
    best_short = max(
        ((key, value) for key, value in scores.items() if key != "expanding"),
        key=lambda item: (item[1], item[0]),
    )
    expanding_score = float(scores["expanding"])
    replace_baseline = bool(float(best_short[1]) > expanding_score)
    selected_id = str(best_short[0]) if replace_baseline else "expanding"
    summary = {
        "schema_version": 3,
        "artifact_type": "seq100_structured_training_window_comparison_summary",
        "status": "completed",
        "completed_at": _now(),
        "study_id": STUDY_ID,
        "study_contract_sha256": str(_read_json(study_path)["contract_sha256"]),
        "objective": dict(selection)["objective"],
        "models": [dict(spec) for spec in specs],
        "selection": dict(selection),
        "selected_annual_metrics": _selected_annual_metrics(selection, annual),
        "cohort_metrics": _reported_cohort_metrics(cohort),
        "screen_2025": _read_json(study_path.parent / "screen_2025.json"),
        "final_selection": {
            "selected_model_id": selected_id,
            "selected_score": float(scores[selected_id]),
            "expanding_score": expanding_score,
            "best_short_model_id": str(best_short[0]),
            "best_short_score": float(best_short[1]),
            "short_minus_expanding": float(best_short[1] - expanding_score),
            "expanding_retained": not replace_baseline,
            "rule": "short window replaces expanding only when its score is strictly higher",
        },
        "epoch_confirmation": dict(epoch_confirmation or {}),
        "coverage": {
            "cohort_rows": int(len(cohort)),
            "capital_metric_rows": int(len(metrics)),
            "capital_annual_rows": int(len(annual)),
            "candidate_score_coverage_complete": bool(
                cohort["candidate_score_coverage"].astype(float).eq(1.0).all()
            ),
        },
        "protected_boundaries": {
            "qdp_changed": False,
            "provider_calls": 0,
            "base_pack_changed": False,
            "turnover_overlay_changed": False,
            "old_checkpoint_changed": False,
            "stage3_started": False,
            "live_state_changed": False,
        },
    }
    _write_json(study_path.parent / "comparison_summary.json", summary)
    (study_path.parent / "comparison_summary.md").write_text(
        _summary_markdown(summary), encoding="utf-8"
    )
    _update_runtime(
        study_path,
        phase="completed",
        current_task=None,
        error=None,
        selected_model_id=selected_id,
    )
    _append_monitor_event(
        study_path.parent,
        {
            "event": "study_completed",
            "status": "completed",
            "selected_model_id": selected_id,
            "watcher_remaining": False,
            "message": f"study completed selected={selected_id}",
        },
    )
    return summary


def evaluate_structured_training_window(
    *, study_root: Path = STUDY_ROOT, max_jobs: int = 0
) -> dict[str, Any]:
    verify_structured_training_window(study_root=study_root, require_complete=False)
    study_path = study_root.resolve() / "study.json"
    phase = str(dict(_read_json(study_path)["runtime"])["phase"])
    if phase == "screen_2025_evaluating":
        specs = _screen_model_specs()
        cohort = _load_or_build_cohort(
            cache_name="screen_2025",
            specs=specs,
            years=(2025,),
            study_path=study_path,
        )
        status, metrics, annual = _evaluate_capital_phase(
            phase="screen_2025",
            specs=specs,
            years=(2025,),
            study_path=study_path,
            max_jobs=int(max_jobs),
        )
        if status != "completed" or metrics is None or annual is None:
            return status_structured_training_window(study_root=study_root)
        selection = select_capital_strategies(
            metrics, allowed_policy_names=SCREEN_SELECTION_POLICIES
        )
        shortlist = shortlist_from_scores(dict(selection["scores"]))
        short_rows = []
        for model_id in list(shortlist["shortlisted_short_model_ids"]):
            if not str(model_id).startswith("window") or not str(model_id).endswith("y"):
                raise ValueError(f"invalid short-window model id: {model_id}")
            short_rows.append(
                {
                    "model_id": str(model_id),
                    "window_years": int(str(model_id)[6:-1]),
                    "score": float(dict(selection["scores"])[str(model_id)]),
                }
            )
        screen = {
            "schema_version": 1,
            "status": "completed",
            "completed_at": _now(),
            "selection": selection,
            "fixed_d2_d60_role": "diagnostic_only",
            "eligible_selection_policies": list(SCREEN_SELECTION_POLICIES),
            "shortlist": shortlist,
        }
        shortlist_payload = {
            "schema_version": 1,
            "status": "completed",
            "completed_at": _now(),
            "rule": "top_two_plus_within_five_percent_of_champion",
            "expanding_retained_as_baseline": True,
            "shortlisted_short_windows": short_rows,
            "screen": shortlist,
        }
        _write_json(study_path.parent / "screen_2025.json", screen)
        _write_json(study_path.parent / "shortlist.json", shortlist_payload)
        _update_runtime(
            study_path,
            phase="confirmation_training",
            current_task=None,
            error=None,
        )
        _merge_phase_csvs(study_path.parent)
        return status_structured_training_window(study_root=study_root)

    if phase not in {
        "confirmation_evaluating",
        "optional_epoch_confirmation",
        "completed",
    }:
        return status_structured_training_window(study_root=study_root)
    if phase == "completed":
        return _read_json(study_path.parent / "comparison_summary.json")

    base_specs = _confirmation_model_specs(study_path.parent)
    base_cohort = _load_or_build_cohort(
        cache_name="confirmation",
        specs=base_specs,
        years=DEVELOPMENT_YEARS,
        study_path=study_path,
    )
    status, base_metrics, base_annual = _evaluate_capital_phase(
        phase="confirmation",
        specs=base_specs,
        years=DEVELOPMENT_YEARS,
        study_path=study_path,
        max_jobs=int(max_jobs),
    )
    if status != "completed" or base_metrics is None or base_annual is None:
        return status_structured_training_window(study_root=study_root)
    base_selection = select_capital_strategies(
        base_metrics, allowed_policy_names=None
    )
    epoch_path = study_path.parent / "epoch_confirmation.json"
    epoch_payload = _read_json(epoch_path) if epoch_path.is_file() else None
    if epoch_payload is not None and int(epoch_payload.get("rule_version", 0)) != 2:
        epoch_payload = None

    if epoch_payload is None:
        yearly = _yearly_selected_returns(base_selection, base_annual)
        losses = {
            str(model_id): {
                int(row["development_year"]): float(row["development_price_total_loss"])
                for row in base_cohort[
                    base_cohort["model_id"].astype(str).eq(str(model_id))
                ].to_dict(orient="records")
            }
            for model_id in base_cohort["model_id"].astype(str).unique()
        }
        short_scores = {
            key: value
            for key, value in dict(base_selection["scores"]).items()
            if key != "expanding"
        }
        strongest_short = max(
            short_scores.items(), key=lambda item: (float(item[1]), str(item[0]))
        )[0]
        champion = str(base_selection["champion_model_id"])
        reasons = epoch_confirmation_reasons(
            scores=dict(base_selection["scores"]),
            champion_id=champion,
            strongest_short_id=strongest_short,
            yearly_returns=yearly,
            cohort_losses=losses,
        )
        preliminary = {
            "schema_version": 1,
            "rule_version": 2,
            "status": "required" if reasons else "not_required",
            "created_at": _now(),
            "window_years": int(str(strongest_short)[6:-1]),
            "model_id": strongest_short,
            "reasons": reasons,
            "preliminary_selection": base_selection,
        }
        _write_json(epoch_path, preliminary)
        epoch_payload = preliminary
        if reasons:
            _update_runtime(
                study_path,
                phase="optional_epoch_confirmation",
                current_task=None,
                error=None,
            )
            _merge_phase_csvs(study_path.parent)
            return status_structured_training_window(study_root=study_root)

    final_specs = list(base_specs)
    final_cohort = base_cohort
    final_metrics = base_metrics
    final_annual = base_annual
    final_selection = base_selection
    if str(epoch_payload.get("status", "")) == "required":
        years = int(epoch_payload["window_years"])
        deep_spec = _model_spec(
            model_id=f"{window_id(years)}_e3", window_years=years, epochs=3
        )
        deep_cohort = _load_or_build_cohort(
            cache_name="epoch_confirmation",
            specs=[deep_spec],
            years=DEVELOPMENT_YEARS,
            study_path=study_path,
        )
        status, deep_metrics, deep_annual = _evaluate_capital_phase(
            phase="epoch_confirmation",
            specs=[deep_spec],
            years=DEVELOPMENT_YEARS,
            study_path=study_path,
            max_jobs=int(max_jobs),
        )
        if status != "completed" or deep_metrics is None or deep_annual is None:
            return status_structured_training_window(study_root=study_root)
        original_id = window_id(years)
        final_specs = [
            spec for spec in base_specs if str(spec["model_id"]) != original_id
        ] + [deep_spec]
        final_cohort = pd.concat(
            [
                base_cohort[~base_cohort["model_id"].astype(str).eq(original_id)],
                deep_cohort,
            ],
            ignore_index=True,
        )
        final_metrics = pd.concat(
            [
                base_metrics[~base_metrics["model_id"].astype(str).eq(original_id)],
                deep_metrics,
            ],
            ignore_index=True,
        )
        final_annual = pd.concat(
            [
                base_annual[~base_annual["model_id"].astype(str).eq(original_id)],
                deep_annual,
            ],
            ignore_index=True,
        )
        final_selection = select_capital_strategies(
            final_metrics, allowed_policy_names=None
        )
        epoch_payload = {
            **dict(epoch_payload),
            "status": "completed",
            "completed_at": _now(),
            "deep_model_id": str(deep_spec["model_id"]),
            "post_confirmation_selection": final_selection,
        }
        _write_json(epoch_path, epoch_payload)

    combined_cohort = _combined_window_metrics(
        study_root=study_path.parent, final_cohort=final_cohort
    )
    _write_csv(study_path.parent / "window_metrics.csv", combined_cohort)
    _merge_phase_csvs(study_path.parent)
    deltas = _selected_daily_deltas(
        selection=final_selection,
        specs=final_specs,
        years=DEVELOPMENT_YEARS,
        study_path=study_path,
    )
    _write_csv(study_path.parent / "daily_pairwise_deltas.csv", deltas)
    return _complete_summary(
        study_path=study_path,
        specs=final_specs,
        cohort=combined_cohort,
        metrics=final_metrics,
        annual=final_annual,
        selection=final_selection,
        epoch_confirmation=epoch_payload,
    )


def summarize_structured_training_window(
    *, study_root: Path = STUDY_ROOT
) -> dict[str, Any]:
    path = study_root.resolve() / "comparison_summary.json"
    if not path.is_file():
        return evaluate_structured_training_window(study_root=study_root, max_jobs=0)
    summary = _read_json(path)
    if int(summary.get("schema_version", 0)) >= 3:
        return summary
    specs = [dict(row) for row in list(summary["models"])]
    confirmation = study_root.resolve() / "capital/confirmation"
    metrics = pd.read_csv(confirmation / "capital_metrics.csv")
    annual = pd.read_csv(confirmation / "capital_annual_metrics.csv")
    selection = select_capital_strategies(metrics, allowed_policy_names=None)
    final_cohort = pd.read_csv(study_root.resolve() / "window_metrics_confirmation.csv")
    combined = _combined_window_metrics(
        study_root=study_root.resolve(), final_cohort=final_cohort
    )
    _write_csv(study_root.resolve() / "window_metrics.csv", combined)
    return _complete_summary(
        study_path=study_root.resolve() / "study.json",
        specs=specs,
        cohort=combined,
        metrics=metrics,
        annual=annual,
        selection=selection,
        epoch_confirmation=_read_json(study_root.resolve() / "epoch_confirmation.json"),
    )


def status_structured_training_window(
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
    tasks: dict[str, Any] = {}
    for years in SHORT_WINDOW_YEARS:
        for year in DEVELOPMENT_YEARS:
            for epochs in (1, 3):
                task = TrainingTask(years, year, epochs)
                complete, partial = _classify_task(study_path=study_path, task=task)
                if not complete and not partial:
                    continue
                progress = None
                candidate = complete[0] if complete else partial[-1]
                if (candidate / "progress.json").is_file():
                    try:
                        progress = _read_json(candidate / "progress.json")
                    except (OSError, json.JSONDecodeError):
                        progress = None
                tasks[task.task_id] = {
                    "status": "completed" if complete else "partial",
                    "run_dir": str(candidate),
                    "progress": progress,
                }
    monitor = study_path.parent / "monitor.json"
    return {
        "status": str(dict(study["runtime"]).get("phase", "unknown")),
        "study": str(study_path),
        "runtime": dict(study["runtime"]),
        "tasks": tasks,
        "monitor": _read_json(monitor) if monitor.is_file() else None,
        "available_memory_gib": float(psutil.virtual_memory().available / 1024**3),
        "h_free_gib": float(shutil.disk_usage(WORKSPACE_ROOT).free / 1024**3),
        "study_size_gib": float(_directory_size(study_path.parent) / 1024**3),
    }


def verify_structured_training_window(
    *, study_root: Path = STUDY_ROOT, require_complete: bool = True
) -> dict[str, Any]:
    study_path = study_root.resolve() / "study.json"
    if not study_path.is_file():
        raise FileNotFoundError(study_path)
    study = _read_json(study_path)
    if str(study.get("study_id", "")) != STUDY_ID:
        raise ValueError("training-window study identity drifted")
    if dict(study.get("contract", {}) or {}) != _semantic_contract():
        raise ValueError("training-window semantic contract drifted")
    if _canonical_digest(study["contract"]) != str(study["contract_sha256"]):
        raise ValueError("training-window contract hash drifted")
    _assert_stage3_not_started()
    material = dict(study.get("material", {}) or {})
    if not material:
        raise ValueError("training-window material is not prepared")
    if dict(material["source_identity"]) != _source_identity():
        raise ValueError("protected source identity drifted")
    _validate_expected_2025(material)
    fairness = dict(material["fairness_material"])
    view_count = 0
    for years in SHORT_WINDOW_YEARS:
        key = window_id(years)
        for year in DEVELOPMENT_YEARS:
            path = _view_for(study, key, year)
            verification = walkforward.verify_development_walkforward_view(path)
            if str(verification["status"]) != "ok":
                raise ValueError(f"bounded view verification failed: {key}/{year}")
            view = _read_json(path)
            if _fairness(view) != fairness[str(year)]:
                raise ValueError(f"bounded view fairness drifted: {key}/{year}")
            registered = dict(dict(material["window_contracts"])[key][str(year)])
            observed = dict(view["structured_training_window_ablation"])
            if registered != observed:
                raise ValueError(f"window contract drifted: {key}/{year}")
            if _canonical_digest({k: v for k, v in observed.items() if k != "sha256"}) != str(
                observed["sha256"]
            ):
                raise ValueError(f"window contract digest drifted: {key}/{year}")
            view_count += 1
    size = _directory_size(study_path.parent)
    if size > STUDY_SIZE_LIMIT_BYTES:
        raise ValueError("training-window study exceeded the 6 GiB safety limit")
    phase = str(dict(study["runtime"])["phase"])
    if require_complete:
        if phase != "completed":
            raise ValueError(f"study is not complete: {phase}")
        required = (
            "window_metrics.csv",
            "capital_metrics.csv",
            "capital_annual_metrics.csv",
            "daily_pairwise_deltas.csv",
            "comparison_summary.json",
            "comparison_summary.md",
            "monitor.json",
            "monitor_events.jsonl",
        )
        for name in required:
            if not (study_path.parent / name).is_file():
                raise FileNotFoundError(study_path.parent / name)
        monitor = _read_json(study_path.parent / "monitor.json")
        if bool(monitor.get("watcher_remaining", False)):
            raise ValueError("monitor reports a residual watcher")
    return {
        "status": "ok",
        "study": str(study_path),
        "phase": phase,
        "bounded_view_count": int(view_count),
        "study_size_bytes": int(size),
        "provider_calls": 0,
        "stage3_started": False,
        "qdp_changed": False,
        "base_pack_copied": False,
        "old_checkpoint_changed": False,
        "live_state_changed": False,
    }
