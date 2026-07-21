"""Local Structured intraday, capital-speed, and true batch-1024 study.

This workflow intentionally retires the simplified global-tail experiment.  It
reuses immutable 180-day fold material and the completed L35V2 checkpoints,
while all new checkpoints keep date-grouped local-chunk ranking and a fixed
single development epoch.
"""

from __future__ import annotations

import gc
import hashlib
import json
import math
import os
import shutil
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import psutil
import torch

from daily_research.path_policy import qdp_v2_sequence_path_training as training
from daily_research.path_policy import seq100_development as development
from daily_research.path_policy import seq100_finite_capital_backtest as finite
from daily_research.path_policy import seq100_exit_policy_audit as exit_audit
from daily_research.path_policy import seq100_structured_experiment as structured
from daily_research.path_policy import seq100_structured_global_intraday as retired_global
from daily_research.path_policy import seq100_structured_input_ablation as ablation
from daily_research.path_policy import seq100_walkforward as walkforward
from daily_research.path_policy.seq100_candidate_execution import (
    parse_execution_cost_contract,
)
from daily_research.path_policy.seq100_exit_policy_audit import CandidateCompleteAuditPack
from daily_research.path_policy.seq100_mainline import build_todayclose_path_only_train_argv


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
PYTHON = Path("C:/Users/ASUS/miniconda3/envs/yolos/python.exe")
STUDY_ID = "seq100_structured_local_intraday_capital_speed_rolling_2023_2025_v1"
STUDY_ROOT = WORKSPACE_ROOT / (
    "daily_research/output/path_policy/studies/"
    "seq100_structured_local_intraday_capital_speed_rolling_2023_2025_v1"
)
SOURCE_STUDY_ROOT = ablation.STUDY_ROOT
OVERLAY_ROOT = ablation.OVERLAY_ROOT
GLOBAL_STUDY_ROOT = retired_global.STUDY_ROOT

DEVELOPMENT_YEARS = (2023, 2024, 2025)
LOOKBACK_DAYS = 180
FORWARD_DAYS = 60
EXECUTION_TAIL_DAYS = 20
SEED = 7
TOP_K_VALUES = retired_global.TOP_K_VALUES
SLOTS_BY_TOP_K = retired_global.SLOTS_BY_TOP_K
FIXED_EXIT_DAYS = retired_global.FIXED_EXIT_DAYS
OWN_EXIT_POLICIES = retired_global.OWN_EXIT_POLICIES
COST_SCENARIOS = retired_global.COST_SCENARIOS
PRIMARY_COST_SCENARIO = retired_global.PRIMARY_COST_SCENARIO
TRAINING_MODEL_ORDER = ("l43v2", "l35v3", "l43v3")
EXPECTED_TRAIN_ROWS = {2023: 5_542_816, 2024: 6_239_523, 2025: 6_952_766}
EXPECTED_CANDIDATES = {2023: 703_468, 2024: 715_878, 2025: 724_125}
EXPECTED_SUPERVISED_CANDIDATES = {2023: 703_312, 2024: 715_717, 2025: 723_807}
EXPECTED_SIGNAL_DATES = 727


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    input_dim: int
    input_channel_profile: str
    path_value_semantic: str
    phase: str
    batch_size: int = 512
    activation_checkpoint_profile: str = training.ACTIVATION_CHECKPOINT_PROFILE_NONE
    reuse_legacy: bool = False
    rank_training_profile: str = training.RANK_TRAINING_PROFILE_LOCAL_CHUNK

    @property
    def input_variant_id(self) -> str:
        return "lookback180_turnover_intraday" if self.input_dim == 43 else "lookback180_turnover"

    @property
    def is_v3(self) -> bool:
        return self.path_value_semantic == training.PATH_VALUE_SEMANTIC_CAPITAL_SPEED_V3

    @property
    def is_global(self) -> bool:
        return False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


MODEL_SPECS = {
    "l35v2": ModelSpec(
        "l35v2",
        35,
        training.INPUT_CHANNEL_PROFILE_DAILY_ONLY_TURNOVER,
        training.PATH_VALUE_SEMANTIC_V2,
        "baseline",
        reuse_legacy=True,
    ),
    "l43v2": ModelSpec(
        "l43v2",
        43,
        training.INPUT_CHANNEL_PROFILE_DAILY_ONLY_TURNOVER_INTRADAY,
        training.PATH_VALUE_SEMANTIC_V2,
        "l43v2",
    ),
    "l35v3": ModelSpec(
        "l35v3",
        35,
        training.INPUT_CHANNEL_PROFILE_DAILY_ONLY_TURNOVER,
        training.PATH_VALUE_SEMANTIC_CAPITAL_SPEED_V3,
        "l35v3",
    ),
    "l43v3": ModelSpec(
        "l43v3",
        43,
        training.INPUT_CHANNEL_PROFILE_DAILY_ONLY_TURNOVER_INTRADAY,
        training.PATH_VALUE_SEMANTIC_CAPITAL_SPEED_V3,
        "l43v3",
    ),
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


def _digest(payload: Any) -> str:
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


def _snapshot_file(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "exists": path.is_file(),
        "size": int(path.stat().st_size) if path.is_file() else None,
        "sha256": _file_sha256(path) if path.is_file() else None,
    }


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = path.with_name(path.name + ".tmp")
    staging.write_text(
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
    os.replace(staging, path)


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = path.with_name(path.name + ".tmp")
    frame.to_csv(staging, index=False, encoding="utf-8-sig")
    os.replace(staging, path)


def _append_event(root: Path, payload: Mapping[str, Any]) -> None:
    event = {"timestamp": _now(), **dict(payload)}
    root.mkdir(parents=True, exist_ok=True)
    with (root / "monitor_events.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(event, ensure_ascii=False, default=_json_default, allow_nan=False)
            + "\n"
        )
    _write_json(root / "monitor.json", event)
    print(f"structured-local: {event.get('message', event.get('event', 'status'))}", flush=True)


def _state_path(study_root: Path) -> Path:
    return study_root.resolve() / "state.json"


def _load_state(study_root: Path) -> dict[str, Any]:
    path = _state_path(study_root)
    return _read_json(path) if path.is_file() else {"status": "not_prepared"}


def _set_state(study_root: Path, status: str, **changes: Any) -> dict[str, Any]:
    current = _load_state(study_root)
    payload = {
        "schema_version": 1,
        "study_id": STUDY_ID,
        **current,
        "status": str(status),
        "updated_at": _now(),
        **changes,
    }
    _write_json(_state_path(study_root), payload)
    return payload


def _semantic_contract() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "contract_id": STUDY_ID,
        "objective": "maximize_double_slippage_continuous_account_annualized_log_growth",
        "selection_years": list(DEVELOPMENT_YEARS),
        "development_protocol": {
            "years": list(DEVELOPMENT_YEARS),
            "method": "purged_expanding_development_walkforward",
            "split_roles": {"fit": "train", "evaluation": "development"},
            "seed": SEED,
            "purge_days": FORWARD_DAYS + EXECUTION_TAIL_DAYS,
        },
        "early_stopping": {
            "metric": training.EARLY_STOPPING_METRIC_DEVELOPMENT_PRICE_TOTAL_LOSS,
            "mode": training.EARLY_STOPPING_MODE_MIN,
            "minimum_complete_epochs": 1,
            "maximum_epochs": 1,
            "patience": 1,
            "restore_best_checkpoint": True,
            "checkpoint_policy": "fixed_final_development_epoch",
            "development_labels_select_epoch": False,
        },
        "profile": {"name": "structured_joint_turnover", "input_dim": 0},
        "models": {key: value.to_dict() for key, value in MODEL_SPECS.items()},
        "training": {
            "lookback_days": LOOKBACK_DAYS,
            "seed": SEED,
            "epochs": 1,
            "batch_size": 512,
            "rank_training_profile": training.RANK_TRAINING_PROFILE_LOCAL_CHUNK,
            "rank_max_per_side": 64,
            "path_value_gradient_profile": training.PATH_VALUE_GRADIENT_PROFILE_HARD_ST,
            "hard_negative_mining": False,
            "development_fixed_final_epoch": True,
        },
        "capital_speed_v3": {
            "semantic_version": training.PATH_VALUE_SEMANTIC_CAPITAL_SPEED_V3,
            "formula": "log((1+close_return)*date_specific_proportional_cost_multiplier)/capital_days",
            "legal_exit_days": [2, 60],
            "earliest_tie_break": True,
            "minimum_commission_and_lot_constraints": "account_execution_only",
            "formula_auto_tuning_from_preflight": False,
        },
        "evaluation": {
            "signal_date_count": EXPECTED_SIGNAL_DATES,
            "top_k": list(TOP_K_VALUES),
            "slots_by_top_k": {str(key): list(value) for key, value in SLOTS_BY_TOP_K.items()},
            "fixed_exit_days": list(FIXED_EXIT_DAYS),
            "own_exit_policies": list(OWN_EXIT_POLICIES),
            "cost_scenarios": list(COST_SCENARIOS),
            "primary_cost_scenario": PRIMARY_COST_SCENARIO,
            "unit_time_alpha": (
                "mean_by_signal_date[log1p(selected_return)/selected_resolved_day-"
                "log1p(universe_return)/universe_resolved_day]*252"
            ),
            "cross_model_rank_exit_hybrid": False,
            "d7_special_weight": False,
        },
        "batch1024": {
            "scope": "only_best_batch512_configuration",
            "learning_rate_scaling": False,
            "gradient_accumulation": False,
            "fallback_activation_checkpoint_profile": (
                training.ACTIVATION_CHECKPOINT_PROFILE_STRUCTURED_GRU
            ),
            "try_batch768": False,
        },
        "protected_boundaries": {
            "update_qdp": False,
            "call_provider": False,
            "copy_base_pack": False,
            "modify_overlay": False,
            "modify_old_checkpoint": False,
            "start_old_stage3": False,
            "change_live_state": False,
            "modify_main_report": False,
            "train_2026": False,
        },
    }


def _protected_snapshot() -> dict[str, Any]:
    payload = retired_global._protected_snapshot()
    payload["old_global_study"] = _snapshot_file(GLOBAL_STUDY_ROOT / "study.json")
    payload["old_global_g35v2_2023_checkpoint"] = _snapshot_file(
        retired_global._run_dir_for(
            GLOBAL_STUDY_ROOT, retired_global.V2_SPECS["g35v2"], 2023
        )
        / "best_model.pt"
    )
    return payload


def _clone_view(
    source_path: Path,
    *,
    target_path: Path,
    study_path: Path,
    contract_sha256: str,
    input_dim: int,
    year: int,
) -> Path:
    source = _read_json(source_path)
    view = json.loads(json.dumps(source, ensure_ascii=False))
    binding = {
        "contract_id": STUDY_ID,
        "contract_sha256": str(contract_sha256),
        "contract_file_sha256": str(contract_sha256),
        "path": str(study_path.resolve()),
    }
    view["created_at"] = _now()
    view["research_contract"] = binding
    view["development_contract"] = binding
    view["artifact_view"] = {
        "schema_version": 1,
        "view_id": f"{STUDY_ID}_input{input_dim}_{year}",
        "view_type": "structured_local_intraday_capital_speed_development_view",
    }
    view["structured_local_intraday_capital_speed"] = {
        "source_view": str(source_path.resolve()),
        "source_view_sha256": _file_sha256(source_path),
        "base_material_copied": False,
    }
    view["development_fold_training_contract"] = (
        walkforward._compute_development_fold_training_contract(view)
    )
    _write_json(target_path, view)
    result = walkforward.verify_development_walkforward_view(target_path)
    if str(result.get("status", "")) != "ok":
        raise ValueError(f"cloned fold view failed verification: {result.get('blockers')}")
    return target_path.resolve()


def _fairness(view_path: Path) -> dict[str, Any]:
    return ablation._view_fairness_material(_read_json(view_path))


def _view_for(study_root: Path, spec: ModelSpec, year: int) -> Path:
    study = _read_json(study_root.resolve() / "study.json")
    return Path(
        str(dict(dict(study["material"])["views"])[str(spec.input_dim)][str(year)])
    ).resolve()


def _run_tag(spec: ModelSpec, year: int) -> str:
    return retired_global._run_tag(spec, year)


def _task_run_dirs(study_root: Path, spec: ModelSpec, year: int) -> list[Path]:
    return sorted(
        (study_root.resolve() / "runs" / spec.model_id).glob(f"{_run_tag(spec, year)}_*")
    )


def _validate_run(
    study_root: Path, spec: ModelSpec, year: int, run_dir: Path
) -> dict[str, Any]:
    if spec.reuse_legacy:
        return development._validate_run_summary(
            run_dir / "sequence_path_training_summary.json", year
        )
    summary = retired_global._validate_run(study_root, spec, year, run_dir)
    resolved = dict(summary.get("resolved_training_config", {}) or {})
    exact = {
        "epochs": 1,
        "batch_size": int(spec.batch_size),
        "development_fixed_final_epoch": True,
        "rank_training_profile": training.RANK_TRAINING_PROFILE_LOCAL_CHUNK,
        "rank_max_per_side": 64,
        "activation_checkpoint_profile": spec.activation_checkpoint_profile,
    }
    for key, expected in exact.items():
        if resolved.get(key) != expected:
            raise ValueError(
                f"training config drifted for {spec.model_id}/{year} {key}: "
                f"{resolved.get(key)!r} != {expected!r}"
            )
    if str(summary.get("checkpoint_policy", "")) != "fixed_final_development_epoch":
        raise ValueError("single-epoch development task did not keep the fixed final epoch")
    if int(summary.get("best_epoch", 0)) != 1:
        raise ValueError("single-epoch task best_epoch must be 1")
    if int(summary.get("training_sample_count", 0)) not in {
        0,
        EXPECTED_TRAIN_ROWS[int(year)],
    }:
        raise ValueError("training row count drifted")
    return summary


def _classify_runs(
    study_root: Path, spec: ModelSpec, year: int
) -> tuple[list[Path], list[Path]]:
    if spec.reuse_legacy:
        return [retired_global._legacy_l35_run(year)], []
    complete: list[Path] = []
    partial: list[Path] = []
    for path in _task_run_dirs(study_root, spec, year):
        try:
            _validate_run(study_root, spec, year, path)
        except (EOFError, FileNotFoundError, KeyError, OSError, RuntimeError, TypeError, ValueError):
            partial.append(path)
        else:
            complete.append(path)
    if len(complete) > 1:
        raise ValueError(f"multiple valid runs for {spec.model_id}/{year}")
    return complete, partial


def _run_dir_for(study_root: Path, spec: ModelSpec, year: int) -> Path:
    complete, _partial = _classify_runs(study_root, spec, year)
    if len(complete) != 1:
        raise ValueError(f"missing completed run for {spec.model_id}/{year}")
    return complete[0]


def _archive_partial(
    study_root: Path, spec: ModelSpec, year: int, paths: Sequence[Path]
) -> None:
    if not paths:
        return
    target_root = study_root.resolve() / "runs/failed" / f"{spec.model_id}__{year}"
    target_root.mkdir(parents=True, exist_ok=True)
    for number, source in enumerate(paths, start=1):
        if study_root.resolve() not in source.resolve().parents:
            raise ValueError("refusing to archive a run outside the study")
        target = target_root / (
            f"{source.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{number}"
        )
        shutil.move(str(source), str(target))


def _batch1024_spec(study_root: Path) -> ModelSpec | None:
    path = study_root.resolve() / "stage_decisions.json"
    if not path.is_file():
        return None
    raw = dict(_read_json(path).get("batch1024_spec", {}) or {})
    return ModelSpec(**raw) if raw else None


def _all_specs(study_root: Path) -> dict[str, ModelSpec]:
    output = dict(MODEL_SPECS)
    batch_spec = _batch1024_spec(study_root)
    if batch_spec is not None:
        output[batch_spec.model_id] = batch_spec
    return output


def prepare_structured_local_intraday_capital_speed(
    *, study_root: Path = STUDY_ROOT
) -> dict[str, Any]:
    root = study_root.resolve()
    study_path = root / "study.json"
    if study_path.is_file():
        existing = _read_json(study_path)
        state = _load_state(root)
        prepared = bool(dict(existing.get("material", {}) or {}).get("views"))
        if prepared and (root / "path_value_preflight.json").is_file():
            verify_structured_local_intraday_capital_speed(
                study_root=root, require_complete=False
            )
            return existing
        if str(existing.get("study_id", "")) != STUDY_ID:
            raise ValueError("existing study identity drifted")
        if _digest(existing.get("contract", {})) != str(
            existing.get("contract_sha256", "")
        ):
            raise ValueError("incomplete study contract drifted")
        if str(state.get("status", "")) not in {
            "not_prepared",
            "prepare_failed",
            "prepared",
        }:
            raise ValueError("incomplete study is not in a recoverable prepare state")
    ablation._assert_no_other_research_process()
    retired_global.retire_structured_global_intraday(
        study_root=GLOBAL_STUDY_ROOT, successor_study_id=STUDY_ID
    )
    source = _read_json(SOURCE_STUDY_ROOT / "study.json")
    overlay = ablation._validate_overlay_manifest(OVERLAY_ROOT / "manifest.json")
    contract = _semantic_contract()
    contract_sha = _digest(contract)
    root.mkdir(parents=True, exist_ok=True)
    if study_path.is_file():
        study = _read_json(study_path)
        if str(study.get("contract_sha256", "")) != contract_sha:
            raise ValueError("registered study contract differs from current code")
    else:
        study = {
            "schema_version": 1,
            "artifact_type": "seq100_structured_local_intraday_capital_speed_study",
            "study_id": STUDY_ID,
            "created_at": _now(),
            "contract": contract,
            "contract_sha256": contract_sha,
            "material": {},
        }
        _write_json(study_path, study)
    _write_json(
        root / "contract.json",
        {"contract": contract, "contract_sha256": contract_sha},
    )
    try:
        source_views = dict(dict(source["material"])["views"])
        views: dict[str, dict[str, str]] = {"35": {}, "43": {}}
        for input_dim, variant in (
            (35, "lookback180_turnover"),
            (43, "lookback180_turnover_intraday"),
        ):
            for year in DEVELOPMENT_YEARS:
                source_view = Path(str(dict(source_views[variant])[str(year)])).resolve()
                target = root / "views" / f"input{input_dim}_{year}.json"
                views[str(input_dim)][str(year)] = str(
                    _clone_view(
                        source_view,
                        target_path=target,
                        study_path=study_path,
                        contract_sha256=contract_sha,
                        input_dim=input_dim,
                        year=year,
                    )
                )
        fairness = {
            str(year): _fairness(Path(views["35"][str(year)]))
            for year in DEVELOPMENT_YEARS
        }
        for year in DEVELOPMENT_YEARS:
            if _fairness(Path(views["43"][str(year)])) != fairness[str(year)]:
                raise ValueError(f"35/43 fairness material differs for {year}")
            view = _read_json(Path(views["35"][str(year)]))
            train_rows = int(dict(view.get("sample_count_by_split", {}) or {}).get("train", 0))
            candidate_rows = int(
                len(
                    pd.read_parquet(
                        str(Path(str(view["candidate_index_path"])).resolve()),
                        columns=["trade_date"],
                    )
                )
            )
            if train_rows != EXPECTED_TRAIN_ROWS[year]:
                raise ValueError(f"training row count drifted for {year}")
            if candidate_rows != EXPECTED_CANDIDATES[year]:
                raise ValueError(f"candidate row count drifted for {year}")
        study["material"] = {
            "views": views,
            "fairness": fairness,
            "overlay": str((OVERLAY_ROOT / "manifest.json").resolve()),
            "overlay_sha256": _file_sha256(OVERLAY_ROOT / "manifest.json"),
            "overlay_validated_size_bytes": int(overlay["validated_size_bytes"]),
            "legacy_l35_runs": {
                str(year): str(retired_global._legacy_l35_run(year))
                for year in DEVELOPMENT_YEARS
            },
            "retired_global": str((GLOBAL_STUDY_ROOT / "retirement.json").resolve()),
            "protected_snapshot": _protected_snapshot(),
            "base_material_copied": False,
        }
        _write_json(study_path, study)
        _set_state(
            root,
            "prepared",
            completed_training_tasks=[],
            completed_evaluation_models=[],
            current_task=None,
            current_evaluation_model=None,
            error=None,
        )
        _build_path_value_preflight(root)
        _append_event(
            root,
            {"event": "prepared", "status": "prepared", "message": "study prepared"},
        )
        return _read_json(study_path)
    except Exception as exc:
        _set_state(root, "prepare_failed", error=str(exc))
        raise


def _training_spec_for_state(study_root: Path) -> ModelSpec | None:
    status = str(_load_state(study_root).get("status", ""))
    mapping = {
        "l43v2_training": MODEL_SPECS["l43v2"],
        "l35v3_training": MODEL_SPECS["l35v3"],
        "l43v3_training": MODEL_SPECS["l43v3"],
    }
    if status in mapping:
        return mapping[status]
    if status == "run_failed":
        failed = str(_load_state(study_root).get("failed_phase", ""))
        if failed in mapping:
            return mapping[failed]
        if failed == "batch1024_training":
            return _batch1024_spec(study_root)
    if status == "batch1024_training":
        return _batch1024_spec(study_root)
    return None


def _next_evaluation_state(spec: ModelSpec) -> str:
    if spec.model_id == "l43v2":
        return "l43v2_evaluating"
    if spec.model_id == "l35v3":
        return "l35v3_evaluating"
    if spec.model_id == "l43v3":
        return "l43v3_evaluating"
    return "batch1024_evaluating"


def _train_one(study_root: Path, spec: ModelSpec, year: int) -> Path:
    root = study_root.resolve()
    complete, partial = _classify_runs(root, spec, year)
    if complete:
        return complete[0]
    _archive_partial(root, spec, year, partial)
    profile = replace(
        ablation._base_structured_profile(int(spec.batch_size)),
        store_view=_view_for(root, spec, year),
        output_root=root / "runs" / spec.model_id,
        run_tag=_run_tag(spec, year),
        epochs=1,
        batch_size=int(spec.batch_size),
        seed=SEED,
        top_k="1,2,3,5,10",
        input_channel_profile=spec.input_channel_profile,
        prediction_mode="compact",
        evaluation_mode=training.EVALUATION_MODE_DEVELOPMENT,
        early_stopping_patience=1,
        early_stopping_min_delta=0.0,
        early_stopping_metric=training.EARLY_STOPPING_METRIC_DEVELOPMENT_PRICE_TOTAL_LOSS,
        early_stopping_mode=training.EARLY_STOPPING_MODE_MIN,
        min_complete_epochs=1,
        max_samples_per_split=0,
        path_value_gradient_profile=training.PATH_VALUE_GRADIENT_PROFILE_HARD_ST,
        path_value_semantic=spec.path_value_semantic,
        rank_training_profile=training.RANK_TRAINING_PROFILE_LOCAL_CHUNK,
        rank_batch_size=int(spec.batch_size),
        development_fixed_final_epoch=True,
        activation_checkpoint_profile=spec.activation_checkpoint_profile,
    )
    command = [
        str(PYTHON),
        "-m",
        "daily_research.path_policy.qdp_v2_sequence_path_training",
        *build_todayclose_path_only_train_argv(profile),
        "--development-contract",
        str(root / "study.json"),
        "--json",
    ]
    run_dir = retired_global._supervise(
        command, study_root=root, spec=spec, year=year
    )
    _validate_run(root, spec, year, run_dir)
    return run_dir


def run_structured_local_intraday_capital_speed(
    *, study_root: Path = STUDY_ROOT, max_tasks: int = 1
) -> dict[str, Any]:
    root = study_root.resolve()
    if not (root / "study.json").is_file():
        prepare_structured_local_intraday_capital_speed(study_root=root)
    state = _load_state(root)
    status = str(state.get("status", ""))
    if status == "batch1024_probing":
        _run_batch1024_probe(root)
        return status_structured_local_intraday_capital_speed(study_root=root)
    spec = _training_spec_for_state(root)
    if spec is None:
        return status_structured_local_intraday_capital_speed(study_root=root)
    phase = "batch1024_training" if spec.phase == "batch1024" else f"{spec.phase}_training"
    _set_state(root, phase, current_task=None, error=None)
    launched = 0
    try:
        for year in DEVELOPMENT_YEARS:
            complete, _partial = _classify_runs(root, spec, year)
            if complete:
                continue
            if int(max_tasks) > 0 and launched >= int(max_tasks):
                return status_structured_local_intraday_capital_speed(study_root=root)
            task_id = f"{spec.model_id}:{year}"
            _set_state(root, phase, current_task=task_id)
            _train_one(root, spec, year)
            launched += 1
            completed = list(_load_state(root).get("completed_training_tasks", []) or [])
            if task_id not in completed:
                completed.append(task_id)
            _set_state(root, phase, current_task=None, completed_training_tasks=completed)
        if all(_classify_runs(root, spec, year)[0] for year in DEVELOPMENT_YEARS):
            _set_state(root, _next_evaluation_state(spec), current_task=None)
    except Exception as exc:
        _set_state(
            root,
            "run_failed",
            failed_phase=phase,
            current_task=None,
            error=str(exc),
        )
        raise
    return status_structured_local_intraday_capital_speed(study_root=root)


def status_structured_local_intraday_capital_speed(
    *, study_root: Path = STUDY_ROOT
) -> dict[str, Any]:
    root = study_root.resolve()
    if not (root / "study.json").is_file():
        return {"status": "not_prepared", "study_root": str(root)}
    tasks: dict[str, Any] = {}
    for spec in _all_specs(root).values():
        for year in DEVELOPMENT_YEARS:
            complete, partial = _classify_runs(root, spec, year)
            tasks[f"{spec.model_id}:{year}"] = {
                "status": (
                    "reused"
                    if spec.reuse_legacy
                    else "completed"
                    if complete
                    else "partial"
                    if partial
                    else "pending"
                ),
                "run_dir": str(complete[0]) if complete else None,
                "partial_runs": [str(path) for path in partial],
                "model_evaluated": (
                    root / "evaluations" / spec.model_id / "evaluation.json"
                ).is_file(),
            }
    state = _load_state(root)
    return {
        "status": str(state.get("status", "unknown")),
        "study_root": str(root),
        "state": state,
        "tasks": tasks,
        "available_memory_gib": float(psutil.virtual_memory().available / 1024**3),
        "monitor": _read_json(root / "monitor.json") if (root / "monitor.json").is_file() else None,
    }


def _score_distribution(values: np.ndarray) -> dict[str, Any]:
    numeric = np.asarray(values, dtype=np.float64).reshape(-1)
    finite = numeric[np.isfinite(numeric)]
    return {
        "count": int(numeric.size),
        "finite_count": int(finite.size),
        "nonfinite_count": int(numeric.size - finite.size),
        "negative_rate": float((finite < 0.0).mean()) if finite.size else None,
        "zero_rate": float((finite == 0.0).mean()) if finite.size else None,
        "positive_rate": float((finite > 0.0).mean()) if finite.size else None,
        "mean": float(finite.mean()) if finite.size else None,
        "std": float(finite.std()) if finite.size else None,
        "minimum": float(finite.min()) if finite.size else None,
        "p01": float(np.quantile(finite, 0.01)) if finite.size else None,
        "p10": float(np.quantile(finite, 0.10)) if finite.size else None,
        "median": float(np.median(finite)) if finite.size else None,
        "p90": float(np.quantile(finite, 0.90)) if finite.size else None,
        "p99": float(np.quantile(finite, 0.99)) if finite.size else None,
        "maximum": float(finite.max()) if finite.size else None,
    }


def _smooth_l1_mean(predicted: np.ndarray, target: np.ndarray) -> float | None:
    predicted = np.asarray(predicted, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    valid = np.isfinite(predicted) & np.isfinite(target)
    if not bool(valid.any()):
        return None
    delta = np.abs(predicted[valid] - target[valid])
    return float(np.mean(np.where(delta < 1.0, 0.5 * delta**2, delta - 0.5)))


def _exit_distribution_summary(raw: Mapping[str, Any], total: int) -> dict[str, Any]:
    counts = {int(key): int(value) for key, value in raw.items()}
    invalid = sorted(day for day in counts if day < 2 or day > 60)
    if invalid:
        raise ValueError(f"preflight produced illegal exit days: {invalid}")
    count = int(sum(counts.values()))
    if count != int(total):
        raise ValueError(f"preflight exit distribution coverage drifted: {count} != {total}")
    maximum = max(counts.values(), default=0)
    return {
        "counts": {str(key): value for key, value in sorted(counts.items())},
        "count": count,
        "d2_share": float(counts.get(2, 0) / max(count, 1)),
        "d60_share": float(counts.get(60, 0) / max(count, 1)),
        "maximum_day_share": float(maximum / max(count, 1)),
    }


def _build_path_value_preflight(study_root: Path) -> dict[str, Any]:
    root = study_root.resolve()
    result_path = root / "path_value_preflight.json"
    if result_path.is_file():
        payload = _read_json(result_path)
        if str(payload.get("status", "")) != "ok":
            raise ValueError("cached path-value preflight is not valid")
        return payload
    rows: list[dict[str, Any]] = []
    yearly: list[dict[str, Any]] = []
    warnings: list[str] = []
    for year in DEVELOPMENT_YEARS:
        spec = MODEL_SPECS["l35v2"]
        run_dir = _run_dir_for(root, spec, year)
        prediction_path = run_dir / "predictions/development_predictions.csv"
        v2 = pd.read_csv(
            prediction_path,
            usecols=[
                "score",
                "true_path_trade_value_v2_60d",
                "predicted_exit_day",
                "true_best_exit_day_60d",
            ],
        )
        if len(v2) != EXPECTED_CANDIDATES[year]:
            raise ValueError(f"V2 preflight candidate count drifted for {year}")
        v2_pred = pd.to_numeric(v2["score"], errors="coerce").to_numpy(dtype=np.float64)
        v2_true = pd.to_numeric(
            v2["true_path_trade_value_v2_60d"], errors="coerce"
        ).to_numpy(dtype=np.float64)
        v2_pred_exit = np.rint(
            pd.to_numeric(v2["predicted_exit_day"], errors="coerce").to_numpy(dtype=np.float64)
        )
        v2_true_exit = np.rint(
            pd.to_numeric(v2["true_best_exit_day_60d"], errors="coerce").to_numpy(dtype=np.float64)
        )
        for scope, values, exits in (
            ("predicted", v2_pred, v2_pred_exit),
            ("true", v2_true, v2_true_exit),
        ):
            distribution = _score_distribution(values)
            expected_finite = (
                EXPECTED_CANDIDATES[year]
                if scope == "predicted"
                else EXPECTED_SUPERVISED_CANDIDATES[year]
            )
            if int(distribution["finite_count"]) != expected_finite:
                raise ValueError(
                    f"V2 {scope} label coverage drifted for {year}: "
                    f"{distribution['finite_count']} != {expected_finite}"
                )
            exit_values, exit_counts = np.unique(exits[np.isfinite(exits)].astype(int), return_counts=True)
            exit_summary = _exit_distribution_summary(
                {str(key): int(value) for key, value in zip(exit_values, exit_counts, strict=True)},
                expected_finite,
            )
            rows.append(
                {
                    "year": year,
                    "path_value_semantic": training.PATH_VALUE_SEMANTIC_V2,
                    "scope": scope,
                    **distribution,
                    "d2_share": exit_summary["d2_share"],
                    "d60_share": exit_summary["d60_share"],
                    "maximum_exit_day_share": exit_summary["maximum_day_share"],
                    "tie_rate": None,
                    "paired_score_smooth_l1": _smooth_l1_mean(v2_pred, v2_true),
                }
            )
        del v2

        diagnostic = structured.stream_checkpoint_diagnostics(
            run_dir=run_dir,
            view_path=_view_for(root, spec, year),
            output_dir=root / "preflight" / "capital_speed_v3" / str(year),
            year=year,
            compare_legacy_domain=False,
            fixed_exit_comparison=False,
            input_channel_profile=spec.input_channel_profile,
            path_value_semantic=training.PATH_VALUE_SEMANTIC_CAPITAL_SPEED_V3,
        )
        if int(diagnostic["candidate_count"]) != EXPECTED_CANDIDATES[year]:
            raise ValueError(f"V3 preflight candidate count drifted for {year}")
        if int(diagnostic["ohlc_geometry_violation_count"]) != 0:
            raise ValueError("legacy checkpoint emitted invalid OHLC geometry during V3 preflight")
        v3_pred = dict(diagnostic["predicted_score_distribution"])
        v3_true = dict(diagnostic["true_score_distribution"])
        if int(v3_pred["finite_count"]) != EXPECTED_CANDIDATES[year]:
            raise ValueError(f"V3 predicted score coverage drifted for {year}")
        if int(v3_true["finite_count"]) != EXPECTED_SUPERVISED_CANDIDATES[year]:
            raise ValueError(f"V3 true label coverage drifted for {year}")
        pred_exit = _exit_distribution_summary(
            diagnostic["legal_exit_day_distribution"], EXPECTED_CANDIDATES[year]
        )
        true_exit = _exit_distribution_summary(
            diagnostic["true_legal_exit_day_distribution"],
            EXPECTED_SUPERVISED_CANDIDATES[year],
        )
        if float(pred_exit["maximum_day_share"]) > 0.50:
            warnings.append(
                f"{year} predicted V3 exits are concentrated: "
                f"{pred_exit['maximum_day_share']:.2%}"
            )
        for scope, distribution, exit_summary in (
            ("predicted", v3_pred, pred_exit),
            ("true", v3_true, true_exit),
        ):
            rows.append(
                {
                    "year": year,
                    "path_value_semantic": training.PATH_VALUE_SEMANTIC_CAPITAL_SPEED_V3,
                    "scope": scope,
                    **distribution,
                    "d2_share": exit_summary["d2_share"],
                    "d60_share": exit_summary["d60_share"],
                    "maximum_exit_day_share": exit_summary["maximum_day_share"],
                    "tie_rate": float(diagnostic["tie_rate"]),
                    "paired_score_smooth_l1": diagnostic["paired_score_smooth_l1"],
                }
            )
        yearly.append(
            {
                "year": year,
                "candidate_count": EXPECTED_CANDIDATES[year],
                "supervised_candidate_count": EXPECTED_SUPERVISED_CANDIDATES[year],
                "unsupervised_candidate_count": (
                    EXPECTED_CANDIDATES[year] - EXPECTED_SUPERVISED_CANDIDATES[year]
                ),
                "v2_value_loss_scale": _smooth_l1_mean(v2_pred, v2_true),
                "v3_value_loss_scale": diagnostic["paired_score_smooth_l1"],
                "v3_predicted_exit": pred_exit,
                "v3_true_exit": true_exit,
                "v3_tie_rate": diagnostic["tie_rate"],
                "v3_earliest_tie_break_count": diagnostic["earliest_tie_break_count"],
                "cost_mapping_validated": True,
            }
        )
        del v2_pred, v2_true, v2_pred_exit, v2_true_exit
        gc.collect()
    frame = pd.DataFrame(rows)
    _write_csv(root / "path_value_preflight.csv", frame)
    payload = {
        "schema_version": 1,
        "artifact_type": "seq100_structured_local_path_value_preflight",
        "status": "ok",
        "created_at": _now(),
        "study_id": STUDY_ID,
        "yearly": yearly,
        "warnings": warnings,
        "formula_changed_from_registered_v3": False,
        "training_allowed": True,
        "outputs": {
            "csv": str((root / "path_value_preflight.csv").resolve()),
        },
    }
    _write_json(result_path, payload)
    return payload


def _forecast_book(
    study_root: Path, spec: ModelSpec
) -> tuple[finite.ForecastBook, dict[str, str]]:
    return retired_global._forecast_book(study_root, spec)


def _lookup_forecast(
    book: finite.ForecastBook,
    date_idx: int,
    symbol_idx: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    symbols = np.asarray(symbol_idx, dtype=np.int64)
    scores = np.full(symbols.shape, np.nan, dtype=np.float64)
    planned = np.full(symbols.shape, 60, dtype=np.int16)
    day = book.days.get(int(date_idx))
    if day is None:
        return scores, planned, np.zeros(symbols.shape, dtype=bool)
    positions = np.searchsorted(day.symbol_idx, symbols)
    clipped = np.clip(positions, 0, max(len(day.symbol_idx) - 1, 0))
    found = (positions < len(day.symbol_idx)) & (day.symbol_idx[clipped] == symbols)
    scores[found] = day.score[clipped[found]]
    planned[found] = day.planned_day[clipped[found]]
    return scores, planned, found


def _resolve_plan_matrix(
    *,
    signal_date_idx: int,
    entry_filled: np.ndarray,
    entry_prices: np.ndarray,
    exit_prices: np.ndarray,
    next_valid_exit_idx: np.ndarray,
    planned_days: np.ndarray,
    terminal_recovery_fraction: float,
) -> dict[str, np.ndarray]:
    entries = np.asarray(entry_prices, dtype=np.float64).reshape(-1)
    count = int(entries.size)
    planned = np.asarray(planned_days, dtype=np.float64)
    if planned.ndim == 1:
        if planned.size == count:
            planned = planned.reshape(count, 1)
        else:
            planned = np.broadcast_to(planned.reshape(1, -1), (count, planned.size)).copy()
    if planned.ndim != 2 or planned.shape[0] != count:
        raise ValueError("planned-day matrix must be [candidate, policy]")
    if not bool(np.isfinite(planned).all()):
        raise ValueError("planned-day matrix contains non-finite values")
    planned = np.clip(np.rint(planned), 2, FORWARD_DAYS).astype(np.int16)
    requested = np.take_along_axis(
        np.asarray(next_valid_exit_idx, dtype=np.int16),
        planned.astype(np.int64) - 1,
        axis=1,
    )
    terminal = requested < 0
    actual_idx = np.where(terminal, FORWARD_DAYS + EXECUTION_TAIL_DAYS - 1, requested)
    prices = np.asarray(exit_prices, dtype=np.float64)
    gathered = np.take_along_axis(prices, actual_idx.astype(np.int64), axis=1)
    gathered = np.where(
        terminal,
        entries.reshape(-1, 1) * float(terminal_recovery_fraction),
        gathered,
    )
    valid_entry = (
        np.asarray(entry_filled, dtype=bool).reshape(-1)
        & np.isfinite(entries)
        & (entries > 0.0)
    )
    exit_day = np.where(valid_entry.reshape(-1, 1), actual_idx + 1, -1).astype(np.int16)
    return {
        "planned_day": planned,
        "exit_day": exit_day,
        "exit_date_idx": np.where(
            valid_entry.reshape(-1, 1), int(signal_date_idx) + exit_day, -1
        ).astype(np.int32),
        "exit_price": np.where(valid_entry.reshape(-1, 1), gathered, np.nan),
        "terminal_recovery": terminal & valid_entry.reshape(-1, 1),
    }


def _rolling_plan_matrix(
    *,
    book: finite.ForecastBook,
    signal_date_idx: int,
    symbol_idx: np.ndarray,
    initial_planned_day: np.ndarray,
    entry_filled: np.ndarray,
    entry_prices: np.ndarray,
    exit_prices: np.ndarray,
    exit_sellable: np.ndarray,
    terminal_recovery_fraction: float,
) -> dict[str, np.ndarray]:
    symbols = np.asarray(symbol_idx, dtype=np.int64)
    entries = np.asarray(entry_prices, dtype=np.float64)
    initial = np.clip(np.rint(initial_planned_day), 2, FORWARD_DAYS).astype(np.int16)
    requested = initial.astype(np.int16, copy=True)
    valid_entry = np.asarray(entry_filled, dtype=bool) & np.isfinite(entries) & (entries > 0.0)
    valid_exit = (
        np.asarray(exit_sellable, dtype=bool)
        & np.isfinite(exit_prices)
        & (np.asarray(exit_prices, dtype=np.float64) >= 0.0)
    )
    resolved_day = np.full(symbols.shape, -1, dtype=np.int16)
    resolved_price = np.full(symbols.shape, np.nan, dtype=np.float64)
    active = valid_entry.copy()
    for day in range(1, FORWARD_DAYS + EXECUTION_TAIL_DAYS + 1):
        sell = active & (day >= requested) & valid_exit[:, day - 1]
        if bool(sell.any()):
            resolved_day[sell] = np.int16(day)
            resolved_price[sell] = exit_prices[sell, day - 1]
            active[sell] = False
        if day > FORWARD_DAYS or not bool(active.any()):
            continue
        fresh_score, fresh_plan, found = _lookup_forecast(
            book, int(signal_date_idx) + day, symbols
        )
        update = active & found & np.isfinite(fresh_score)
        candidate = np.where(
            fresh_score <= 0.0,
            day + 1,
            day + np.clip(fresh_plan.astype(np.int64), 2, FORWARD_DAYS),
        )
        requested[update] = np.minimum(
            requested[update].astype(np.int64),
            np.minimum(candidate[update], FORWARD_DAYS),
        ).astype(np.int16)
    terminal = active & valid_entry
    resolved_day[terminal] = np.int16(FORWARD_DAYS + EXECUTION_TAIL_DAYS)
    resolved_price[terminal] = entries[terminal] * float(terminal_recovery_fraction)
    return {
        "planned_day": initial.reshape(-1, 1),
        "exit_day": resolved_day.reshape(-1, 1),
        "exit_date_idx": np.where(
            resolved_day.reshape(-1, 1) >= 0,
            int(signal_date_idx) + resolved_day.reshape(-1, 1),
            -1,
        ).astype(np.int32),
        "exit_price": resolved_price.reshape(-1, 1),
        "terminal_recovery": terminal.reshape(-1, 1),
    }


def _cashflow_matrix(
    *,
    allocation: float,
    entry_filled: np.ndarray,
    entry_prices: np.ndarray,
    plan: Mapping[str, np.ndarray],
    date_values: np.ndarray,
    contract: Any,
    slippage_multiplier: float,
) -> dict[str, np.ndarray]:
    terms = exit_audit._buy_terms_batch(
        allocated_cash=float(allocation),
        entry_filled=entry_filled,
        entry_prices=entry_prices,
        contract=contract,
        slippage_multiplier=float(slippage_multiplier),
    )
    exit_price = np.asarray(plan["exit_price"], dtype=np.float64)
    exit_date_idx = np.asarray(plan["exit_date_idx"], dtype=np.int32)
    active = terms["order_filled"].reshape(-1, 1) & np.isfinite(exit_price) & (exit_date_idx >= 0)
    slippage_rate = contract.slippage_bps * float(slippage_multiplier) / 10_000.0
    sell_price = exit_price * max(0.0, 1.0 - slippage_rate)
    sell_notional = terms["shares"].reshape(-1, 1) * sell_price
    commission_rate = contract.commission_bps / 10_000.0
    transfer_rate = contract.transfer_fee_bps / 10_000.0
    stamp_bps = exit_audit._stamp_tax_bps_by_date_idx(
        exit_date_idx, date_values=date_values, contract=contract
    )
    sell_commission = np.where(
        active,
        np.maximum(contract.minimum_commission_cny, sell_notional * commission_rate),
        0.0,
    )
    sell_transfer = np.where(active, sell_notional * transfer_rate, 0.0)
    stamp_tax = np.where(active, sell_notional * stamp_bps / 10_000.0, 0.0)
    ending = np.where(
        active,
        float(allocation)
        - terms["buy_cash"].reshape(-1, 1)
        + sell_notional
        - sell_commission
        - sell_transfer
        - stamp_tax,
        float(allocation),
    )
    explicit = (
        terms["buy_commission"].reshape(-1, 1)
        + terms["buy_transfer"].reshape(-1, 1)
        + sell_commission
        + sell_transfer
        + stamp_tax
    )
    slippage_cost = terms["shares"].reshape(-1, 1) * (
        (terms["buy_price"] - np.asarray(entry_prices, dtype=np.float64)).reshape(-1, 1)
        + (exit_price - sell_price)
    )
    return {
        "order_filled": active,
        "net_return": ending / float(allocation) - 1.0,
        "total_cost": np.where(active, explicit + slippage_cost, 0.0),
        "cash_utilization": np.where(
            active, terms["buy_cash"].reshape(-1, 1) / float(allocation), 0.0
        ),
    }


def _unit_metric_rows(
    *,
    model_id: str,
    year: int,
    trade_date: str,
    policy_names: Sequence[str],
    policy_kinds: Sequence[str],
    selected_idx: np.ndarray,
    selected_symbols: Sequence[str],
    top_k: int,
    cost_scenario: str,
    plan: Mapping[str, np.ndarray],
    cash: Mapping[str, np.ndarray],
    universe_count: int,
) -> list[dict[str, Any]]:
    selected = np.asarray(selected_idx, dtype=np.int64)
    net = np.asarray(cash["net_return"], dtype=np.float64)
    exit_day = np.asarray(plan["exit_day"], dtype=np.float64)
    planned_day = np.asarray(plan["planned_day"], dtype=np.float64)
    terminal = np.asarray(plan["terminal_recovery"], dtype=bool)
    filled = np.asarray(cash["order_filled"], dtype=bool)
    cost = np.asarray(cash["total_cost"], dtype=np.float64)
    utilization = np.asarray(cash["cash_utilization"], dtype=np.float64)
    output: list[dict[str, Any]] = []
    for column, (policy_name, policy_kind) in enumerate(zip(policy_names, policy_kinds, strict=True)):
        selected_return = float(net[selected, column].mean())
        universe_return = float(net[:, column].mean())
        selected_days = np.where(exit_day[selected, column] >= 0, exit_day[selected, column], np.nan)
        universe_days = np.where(exit_day[:, column] >= 0, exit_day[:, column], np.nan)
        selected_finite_days = selected_days[np.isfinite(selected_days)]
        universe_finite_days = universe_days[np.isfinite(universe_days)]
        selected_day = (
            float(selected_finite_days.mean()) if selected_finite_days.size else math.nan
        )
        universe_day = (
            float(universe_finite_days.mean()) if universe_finite_days.size else math.nan
        )
        valid = (
            selected_return > -1.0
            and universe_return > -1.0
            and selected_day > 0.0
            and universe_day > 0.0
        )
        daily_alpha = (
            (
                math.log1p(selected_return) / selected_day
                - math.log1p(universe_return) / universe_day
            )
            * 252.0
            if valid
            else math.nan
        )
        output.append(
            {
                "model_id": model_id,
                "year": int(year),
                "trade_date": str(trade_date),
                "top_k": int(top_k),
                "policy_name": str(policy_name),
                "policy_kind": str(policy_kind),
                "fixed_day": (
                    int(str(policy_name).removeprefix("fixed_d"))
                    if str(policy_name).startswith("fixed_d")
                    else None
                ),
                "cost_scenario": str(cost_scenario),
                "selected_count": int(selected.size),
                "universe_count": int(universe_count),
                "selected_symbols_json": json.dumps(
                    list(selected_symbols), ensure_ascii=False, separators=(",", ":")
                ),
                "selected_return": selected_return,
                "universe_return": universe_return,
                "selected_resolved_day": selected_day,
                "universe_resolved_day": universe_day,
                "annualized_unit_time_alpha": daily_alpha,
                "selected_entry_fill_rate": float(filled[selected, column].mean()),
                "universe_entry_fill_rate": float(filled[:, column].mean()),
                "selected_terminal_recovery_rate": float(
                    (terminal[selected, column] & filled[selected, column]).mean()
                ),
                "universe_terminal_recovery_rate": float(
                    (terminal[:, column] & filled[:, column]).mean()
                ),
                "selected_deferred_exit_rate": float(
                    (
                        (exit_day[selected, column] > planned_day[selected, column])
                        & (~terminal[selected, column])
                        & filled[selected, column]
                    ).mean()
                ),
                "universe_deferred_exit_rate": float(
                    (
                        (exit_day[:, column] > planned_day[:, column])
                        & (~terminal[:, column])
                        & filled[:, column]
                    ).mean()
                ),
                "selected_execution_cost_cny": float(cost[selected, column].mean()),
                "universe_execution_cost_cny": float(cost[:, column].mean()),
                "selected_cash_utilization": float(utilization[selected, column].mean()),
                "universe_cash_utilization": float(utilization[:, column].mean()),
            }
        )
    return output


def _unit_time_evaluation(
    study_root: Path, spec: ModelSpec, book: finite.ForecastBook
) -> tuple[pd.DataFrame, pd.DataFrame]:
    root = study_root.resolve()
    daily_rows: list[dict[str, Any]] = []
    fixed_names = [f"fixed_d{day}" for day in FIXED_EXIT_DAYS]
    fixed_kinds = ["fixed"] * len(fixed_names)
    own_names = ["model_plan", "rolling_path"]
    own_kinds = ["model_plan", "rolling"]
    for year in DEVELOPMENT_YEARS:
        view_path = _view_for(root, spec, year)
        pack = CandidateCompleteAuditPack(view_path)
        candidates = pack.candidates_for_year(year)
        for trade_date, date_frame in candidates.groupby("trade_date", sort=True):
            date_frame = date_frame.reset_index(drop=True)
            date_idx_values = date_frame["date_idx"].astype(int).unique()
            if len(date_idx_values) != 1:
                raise ValueError("candidate date_idx is not unique")
            date_idx = int(date_idx_values[0])
            symbols = date_frame["symbol_idx"].to_numpy(dtype=np.int64)
            score, planned, found = _lookup_forecast(book, date_idx, symbols)
            if not bool(found.all()) or not bool(np.isfinite(score).all()):
                raise ValueError(f"score coverage is incomplete for {spec.model_id}/{trade_date}")
            order = np.lexsort((symbols, -score))
            entries, exit_prices, exit_sellable = pack.execution_paths(date_idx, symbols)
            entry_filled = date_frame["entry_filled"].to_numpy(dtype=bool)
            next_valid = exit_audit._next_valid_exit_indices(exit_prices, exit_sellable)
            fixed_plan = _resolve_plan_matrix(
                signal_date_idx=date_idx,
                entry_filled=entry_filled,
                entry_prices=entries,
                exit_prices=exit_prices,
                next_valid_exit_idx=next_valid,
                planned_days=np.asarray(FIXED_EXIT_DAYS, dtype=np.int16),
                terminal_recovery_fraction=pack.terminal_recovery_fraction,
            )
            model_plan = _resolve_plan_matrix(
                signal_date_idx=date_idx,
                entry_filled=entry_filled,
                entry_prices=entries,
                exit_prices=exit_prices,
                next_valid_exit_idx=next_valid,
                planned_days=planned,
                terminal_recovery_fraction=pack.terminal_recovery_fraction,
            )
            rolling_plan = _rolling_plan_matrix(
                book=book,
                signal_date_idx=date_idx,
                symbol_idx=symbols,
                initial_planned_day=planned,
                entry_filled=entry_filled,
                entry_prices=entries,
                exit_prices=exit_prices,
                exit_sellable=exit_sellable,
                terminal_recovery_fraction=pack.terminal_recovery_fraction,
            )
            own_plan = {
                key: np.concatenate([model_plan[key], rolling_plan[key]], axis=1)
                for key in model_plan
            }
            for top_k in TOP_K_VALUES:
                selected = order[: int(top_k)]
                selected_symbols = date_frame.iloc[selected]["symbol"].astype(str).tolist()
                allocation = finite.STARTING_CASH_CNY / int(top_k)
                for scenario, multiplier in (
                    ("base", 1.0),
                    ("double_slippage", pack.contract.stress_slippage_multiplier),
                ):
                    fixed_cash = _cashflow_matrix(
                        allocation=allocation,
                        entry_filled=entry_filled,
                        entry_prices=entries,
                        plan=fixed_plan,
                        date_values=pack.date_values,
                        contract=pack.contract,
                        slippage_multiplier=multiplier,
                    )
                    daily_rows.extend(
                        _unit_metric_rows(
                            model_id=spec.model_id,
                            year=year,
                            trade_date=str(trade_date),
                            policy_names=fixed_names,
                            policy_kinds=fixed_kinds,
                            selected_idx=selected,
                            selected_symbols=selected_symbols,
                            top_k=top_k,
                            cost_scenario=scenario,
                            plan=fixed_plan,
                            cash=fixed_cash,
                            universe_count=len(date_frame),
                        )
                    )
                    own_cash = _cashflow_matrix(
                        allocation=allocation,
                        entry_filled=entry_filled,
                        entry_prices=entries,
                        plan=own_plan,
                        date_values=pack.date_values,
                        contract=pack.contract,
                        slippage_multiplier=multiplier,
                    )
                    daily_rows.extend(
                        _unit_metric_rows(
                            model_id=spec.model_id,
                            year=year,
                            trade_date=str(trade_date),
                            policy_names=own_names,
                            policy_kinds=own_kinds,
                            selected_idx=selected,
                            selected_symbols=selected_symbols,
                            top_k=top_k,
                            cost_scenario=scenario,
                            plan=own_plan,
                            cash=own_cash,
                            universe_count=len(date_frame),
                        )
                    )
            del entries, exit_prices, exit_sellable, next_valid, fixed_plan, model_plan
            del rolling_plan, own_plan
        del pack, candidates
        gc.collect()
    daily = pd.DataFrame(daily_rows)
    aggregate = (
        daily.groupby(
            ["model_id", "year", "top_k", "policy_name", "policy_kind", "fixed_day", "cost_scenario"],
            dropna=False,
            sort=True,
        )
        .agg(
            annualized_unit_time_alpha=("annualized_unit_time_alpha", "mean"),
            median_daily_unit_time_alpha=("annualized_unit_time_alpha", "median"),
            positive_day_rate=("annualized_unit_time_alpha", lambda values: float((values > 0.0).mean())),
            signal_date_count=("trade_date", "nunique"),
            selected_return=("selected_return", "mean"),
            universe_return=("universe_return", "mean"),
            selected_resolved_day=("selected_resolved_day", "mean"),
            universe_resolved_day=("universe_resolved_day", "mean"),
            selected_entry_fill_rate=("selected_entry_fill_rate", "mean"),
            universe_entry_fill_rate=("universe_entry_fill_rate", "mean"),
        )
        .reset_index()
    )
    return aggregate, daily


def _yearly_model_metrics(
    study_root: Path,
    spec: ModelSpec,
    unit_daily: pd.DataFrame,
) -> list[dict[str, Any]]:
    root = study_root.resolve()
    rows: list[dict[str, Any]] = []
    for year in DEVELOPMENT_YEARS:
        run_dir = _run_dir_for(root, spec, year)
        summary = development._validate_run_summary(
            run_dir / "sequence_path_training_summary.json", year
        )
        split = next(
            row
            for row in list(summary.get("split_metrics", []) or [])
            if str(row.get("split", "")) == "development"
        )
        daily = pd.read_csv(run_dir / "daily_topk_metrics.csv")
        daily = daily[daily["split"].astype(str).eq("development")].copy()
        metrics: dict[str, float] = {}
        for top_k in (1, 3, 5, 10):
            group = daily[daily["top_k"].astype(int).eq(top_k)]
            if group.empty:
                raise ValueError(f"missing Top{top_k} metrics for {spec.model_id}/{year}")
            for scenario, suffix in (("base", "base"), ("stress", "stress")):
                metrics[f"top{top_k}_{scenario}_alpha"] = float(
                    pd.to_numeric(
                        group[f"alpha_net_realized_plan_return_{suffix}"], errors="coerce"
                    ).mean()
                )
        for scenario, output_name in (("base", "base"), ("double_slippage", "stress")):
            group = unit_daily[
                unit_daily["year"].astype(int).eq(year)
                & unit_daily["top_k"].astype(int).eq(2)
                & unit_daily["policy_name"].eq("model_plan")
                & unit_daily["cost_scenario"].eq(scenario)
            ]
            if group.empty:
                raise ValueError(f"missing Top2 unit evidence for {spec.model_id}/{year}")
            metrics[f"top2_{output_name}_alpha"] = float(
                (group["selected_return"] - group["universe_return"]).mean()
            )
        diagnostic = structured.stream_checkpoint_diagnostics(
            run_dir=run_dir,
            view_path=_view_for(root, spec, year),
            output_dir=root / "evaluations" / spec.model_id / "path_diagnostics" / str(year),
            year=year,
            compare_legacy_domain=False,
            fixed_exit_comparison=False,
            input_channel_profile=spec.input_channel_profile,
            path_value_semantic=spec.path_value_semantic,
        )
        rows.append(
            {
                "model_id": spec.model_id,
                "year": year,
                "rank_ic": float(split["rank_ic_mean"]),
                "rank_ic_positive_day_rate": float(split["rank_ic_positive_day_rate"]),
                "path_mae": float(split["path_mae"]),
                "path_open_mae": float(split["path_open_mae"]),
                "path_high_mae": float(split["path_high_mae"]),
                "path_low_mae": float(split["path_low_mae"]),
                "path_close_mae": float(split["path_close_mae"]),
                "top1_base_alpha": metrics["top1_base_alpha"],
                "top2_base_alpha": metrics["top2_base_alpha"],
                "top3_base_alpha": metrics["top3_base_alpha"],
                "top5_base_alpha": metrics["top5_base_alpha"],
                "top10_base_alpha": metrics["top10_base_alpha"],
                "top1_stress_alpha": metrics["top1_stress_alpha"],
                "top2_stress_alpha": metrics["top2_stress_alpha"],
                "top3_stress_alpha": metrics["top3_stress_alpha"],
                "top5_stress_alpha": metrics["top5_stress_alpha"],
                "top10_stress_alpha": metrics["top10_stress_alpha"],
                "opportunity_alpha": float(diagnostic["legal_opportunity_alpha"]),
                "exit_regret": float(diagnostic["exit_regret"]),
                "exit_deferral_rate": float(diagnostic["top3_exit_deferral_rate"]),
                "geometry_violation_count": int(
                    diagnostic["ohlc_geometry_violation_count"]
                ),
                "candidate_count": int(split["row_count"]),
                "date_count": int(split["date_count"]),
                "score_coverage": float(split["candidate_complete_score_coverage"]),
                "checkpoint_sha256": _file_sha256(run_dir / "best_model.pt"),
                "prediction_sha256": _file_sha256(
                    run_dir / "predictions/development_predictions.csv"
                ),
            }
        )
    return rows


def _evaluation_valid(path: Path, spec: ModelSpec) -> bool:
    if not path.is_file():
        return False
    try:
        payload = _read_json(path)
    except (OSError, json.JSONDecodeError):
        return False
    return bool(
        str(payload.get("status", "")) == "completed"
        and str(dict(payload.get("model", {}) or {}).get("model_id", "")) == spec.model_id
        and int(payload.get("job_count", 0)) == len(retired_global._account_jobs(spec.model_id))
    )


def _evaluate_model(study_root: Path, spec: ModelSpec) -> dict[str, Any]:
    root = study_root.resolve()
    output = root / "evaluations" / spec.model_id
    output.mkdir(parents=True, exist_ok=True)
    book, prediction_hashes = _forecast_book(root, spec)
    signal_dates = tuple(book.signal_date_indices)
    if len(signal_dates) != EXPECTED_SIGNAL_DATES:
        raise ValueError(
            f"signal-date coverage drifted for {spec.model_id}: {len(signal_dates)}"
        )
    top_books = {
        top_k: retired_global._top_k_book(book, top_k) for top_k in TOP_K_VALUES
    }
    market = retired_global._market(_view_for(root, spec, DEVELOPMENT_YEARS[0]))
    semantic = {
        "schema_version": 1,
        "study_contract_sha256": _read_json(root / "study.json")["contract_sha256"],
        "model": spec.to_dict(),
        "prediction_sha256": prediction_hashes,
        "signal_date_indices": [int(signal_dates[0]), int(signal_dates[-1]), len(signal_dates)],
        "fairness": _read_json(root / "study.json")["material"]["fairness"],
        "top_k": list(TOP_K_VALUES),
        "slots": {str(key): list(value) for key, value in SLOTS_BY_TOP_K.items()},
        "fixed_days": list(FIXED_EXIT_DAYS),
        "own_exits": list(OWN_EXIT_POLICIES),
        "cost_scenarios": list(COST_SCENARIOS),
        "unit_time_alpha_semantic": _semantic_contract()["evaluation"]["unit_time_alpha"],
    }
    contract_sha = _digest(semantic)
    _write_json(
        output / "contract.json",
        {"contract": semantic, "contract_sha256": contract_sha},
    )
    _write_json(
        output / "prediction_manifest.json",
        {
            "schema_version": 1,
            "model_id": spec.model_id,
            "prediction_sha256": prediction_hashes,
            "score_coverage": 1.0,
            "signal_date_count": len(signal_dates),
        },
    )
    jobs = retired_global._account_jobs(spec.model_id)
    jobs_root = output / "jobs"
    jobs_root.mkdir(exist_ok=True)
    for number, job in enumerate(jobs, start=1):
        path = jobs_root / f"{job.job_id}.json"
        if not retired_global._job_valid(path, job, contract_sha):
            payload = retired_global._run_account_job(
                job,
                market=market,
                book=top_books[job.top_k],
                signal_dates=signal_dates,
                contract_sha256=contract_sha,
            )
            _write_json(path, payload)
        if number == 1 or number % 100 == 0 or number == len(jobs):
            _write_json(
                output / "progress.json",
                {
                    "status": "running" if number < len(jobs) else "accounts_completed",
                    "completed_jobs": number,
                    "total_jobs": len(jobs),
                    "updated_at": _now(),
                },
            )
    metric_rows: list[dict[str, Any]] = []
    annual_rows: list[dict[str, Any]] = []
    for job in jobs:
        payload = _read_json(jobs_root / f"{job.job_id}.json")
        identity = {
            "model_id": spec.model_id,
            "top_k": job.top_k,
            "slot_count": job.slots,
            "policy_name": job.policy_name,
            "policy_kind": job.policy_kind,
            "fixed_day": job.fixed_day,
            "cost_scenario": job.cost_scenario,
        }
        metric_rows.append({**identity, **dict(payload["metric"])})
        annual_rows.extend(
            {**identity, **dict(row)} for row in payload["annual_metrics"]
        )
    account = pd.DataFrame(metric_rows)
    annual = pd.DataFrame(annual_rows)
    unit, unit_daily = _unit_time_evaluation(root, spec, book)
    yearly = pd.DataFrame(_yearly_model_metrics(root, spec, unit_daily))
    if not bool(yearly["score_coverage"].eq(1.0).all()):
        raise ValueError(f"score coverage is not 100% for {spec.model_id}")
    if int(yearly["geometry_violation_count"].sum()) != 0:
        raise ValueError(f"OHLC geometry violations found for {spec.model_id}")
    _write_csv(output / "capital_metrics.csv", account)
    _write_csv(output / "capital_annual_metrics.csv", annual)
    _write_csv(output / "vintage_metrics.csv", yearly)
    _write_csv(output / "unit_time_alpha.csv", unit)
    _write_csv(output / "unit_time_alpha_daily.csv", unit_daily)
    result_files = (
        "capital_metrics.csv",
        "capital_annual_metrics.csv",
        "vintage_metrics.csv",
        "unit_time_alpha.csv",
        "unit_time_alpha_daily.csv",
    )
    evaluation = {
        "schema_version": 1,
        "artifact_type": "seq100_structured_local_model_evaluation",
        "status": "completed",
        "completed_at": _now(),
        "model": spec.to_dict(),
        "contract_sha256": contract_sha,
        "job_count": len(jobs),
        "prediction_sha256": prediction_hashes,
        "coverage": {
            "signal_date_count": len(signal_dates),
            "score_coverage": 1.0,
            "unit_time_policy_count": len(FIXED_EXIT_DAYS) + len(OWN_EXIT_POLICIES),
        },
        "outputs": {
            name.removesuffix(".csv"): str((output / name).resolve())
            for name in result_files
        },
        "result_artifact_sha256": _digest(
            {name: _file_sha256(output / name) for name in result_files}
        ),
    }
    _write_json(output / "evaluation.json", evaluation)
    _write_json(
        output / "progress.json",
        {
            "status": "completed",
            "evaluation": str((output / "evaluation.json").resolve()),
            "updated_at": _now(),
        },
    )
    del market, book, top_books
    gc.collect()
    return evaluation


def _evaluation_spec_for_state(study_root: Path) -> ModelSpec | None:
    state = _load_state(study_root)
    status = str(state.get("status", ""))
    if status in {"prepared", "baseline_evaluating"}:
        return MODEL_SPECS["l35v2"]
    mapping = {
        "l43v2_evaluating": MODEL_SPECS["l43v2"],
        "l35v3_evaluating": MODEL_SPECS["l35v3"],
        "l43v3_evaluating": MODEL_SPECS["l43v3"],
    }
    if status in mapping:
        return mapping[status]
    if status == "batch1024_evaluating":
        return _batch1024_spec(study_root)
    if status == "evaluation_failed":
        failed = str(state.get("failed_phase", ""))
        if failed == "baseline_evaluating":
            return MODEL_SPECS["l35v2"]
        if failed in mapping:
            return mapping[failed]
        if failed == "batch1024_evaluating":
            return _batch1024_spec(study_root)
    return None


def _post_evaluation_state(spec: ModelSpec) -> str:
    return {
        "l35v2": "l43v2_training",
        "l43v2": "l35v3_training",
        "l35v3": "l43v3_training",
        "l43v3": "factorial_summarizing",
    }.get(spec.model_id, "completed")


def evaluate_structured_local_intraday_capital_speed(
    *, study_root: Path = STUDY_ROOT, max_jobs: int = 1
) -> dict[str, Any]:
    root = study_root.resolve()
    spec = _evaluation_spec_for_state(root)
    if spec is None:
        return status_structured_local_intraday_capital_speed(study_root=root)
    evaluation_path = root / "evaluations" / spec.model_id / "evaluation.json"
    phase = "baseline_evaluating" if spec.model_id == "l35v2" else (
        "batch1024_evaluating" if spec.phase == "batch1024" else f"{spec.phase}_evaluating"
    )
    try:
        if not _evaluation_valid(evaluation_path, spec):
            if int(max_jobs) == 0:
                return status_structured_local_intraday_capital_speed(study_root=root)
            _set_state(root, phase, current_evaluation_model=spec.model_id, error=None)
            _append_event(
                root,
                {
                    "event": "evaluation_started",
                    "status": "running",
                    "model_id": spec.model_id,
                    "message": f"evaluate {spec.model_id}",
                },
            )
            _evaluate_model(root, spec)
            _append_event(
                root,
                {
                    "event": "evaluation_completed",
                    "status": "completed",
                    "model_id": spec.model_id,
                    "message": f"evaluated {spec.model_id}",
                },
            )
        completed = list(_load_state(root).get("completed_evaluation_models", []) or [])
        if spec.model_id not in completed:
            completed.append(spec.model_id)
        next_state = _post_evaluation_state(spec)
        _set_state(
            root,
            next_state,
            completed_evaluation_models=completed,
            current_evaluation_model=None,
            error=None,
        )
        if spec.phase == "batch1024":
            summarize_structured_local_intraday_capital_speed(study_root=root)
    except Exception as exc:
        _set_state(
            root,
            "evaluation_failed",
            failed_phase=phase,
            current_evaluation_model=None,
            error=str(exc),
        )
        raise
    return status_structured_local_intraday_capital_speed(study_root=root)


def _probe_backward(
    *,
    dataset: training.SequencePathPackDataset,
    batch: Mapping[str, Any],
    spec: ModelSpec,
    activation_profile: str,
) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    device = torch.device("cuda")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    model = training.SequencePathModel(
        input_dim=dataset.input_dim,
        hidden_dim=128,
        layers=2,
        forward_days=dataset.forward_days,
        summary_dim=len(dataset.path_summary_columns),
        dropout=0.1,
        model_type="gru_structured_joint_turnover",
        symbol_count=dataset.symbol_count,
        richer_path_dim=dataset.richer_path_dim,
        activation_checkpoint_profile=activation_profile,
    ).to(device)
    profile = ablation._base_structured_profile(1024)
    x, y_path, y_ohlcva, y_richer, y_summary, y_activity, date_idx, symbol_idx = (
        training._batch_to_device(batch, device)
    )
    y_tradable = batch["y_tradable_path"].to(device)
    growth = batch["path_value_growth_multiplier"].to(device)
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    model.train()
    started = datetime.now().timestamp()
    with torch.amp.autocast(device_type="cuda", enabled=True):
        outputs = model(x, symbol_idx=symbol_idx)
        loss, parts = training._compute_loss(
            outputs,
            y_path,
            y_summary,
            date_idx,
            y_ohlcva_path=y_ohlcva,
            y_richer_path=y_richer,
            y_activity_path=y_activity,
            value_index=dataset.value_index,
            path_weight=float(profile.path_loss_weight),
            path_loss_profile=str(profile.path_loss_profile),
            summary_weight=float(profile.summary_loss_weight),
            value_weight=float(profile.value_loss_weight),
            rank_weight=float(profile.rank_loss_weight),
            richer_weight=float(profile.richer_loss_weight),
            rank_max_per_side=64,
            price_delta_weight=float(profile.price_delta_loss_weight),
            va_level_weight=float(profile.va_level_loss_weight),
            va_delta_weight=float(profile.va_delta_loss_weight),
            geometry_weight=float(profile.geometry_loss_weight),
            utility_curve_weight=float(profile.utility_curve_loss_weight),
            turnover_level_weight=float(profile.turnover_level_loss_weight),
            turnover_delta_weight=float(profile.turnover_delta_loss_weight),
            residual_weight=0.25,
            residual_penalty_weight=0.01,
            price_anchor=dataset.price_anchor,
            summary_loss_profile=str(profile.summary_loss_profile),
            direct_value_horizon=int(profile.direct_value_horizon),
            path_value_gradient_profile=training.PATH_VALUE_GRADIENT_PROFILE_HARD_ST,
            path_value_semantic=spec.path_value_semantic,
            path_value_growth_multiplier=growth,
            rank_training_profile=training.RANK_TRAINING_PROFILE_LOCAL_CHUNK,
            y_tradable_path=y_tradable,
            return_tensor_parts=True,
        )
    scaler.scale(loss).backward()
    gradients = [
        parameter.grad
        for parameter in model.parameters()
        if parameter.requires_grad and parameter.grad is not None
    ]
    if not gradients or not all(bool(torch.isfinite(value).all()) for value in gradients):
        raise ValueError("batch1024 probe produced missing or non-finite gradients")
    result = {
        "status": "success",
        "activation_checkpoint_profile": activation_profile,
        "batch_size": int(x.shape[0]),
        "input_shape": list(x.shape),
        "loss": float(loss.detach().cpu()),
        "loss_parts": {
            key: float(value.detach().cpu())
            for key, value in parts.items()
            if isinstance(value, torch.Tensor) and not str(key).startswith("_")
        },
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        "seconds": float(datetime.now().timestamp() - started),
        "state_dict_keys": list(model.state_dict().keys()),
    }
    del model, scaler, x, y_path, y_ohlcva, y_richer, y_summary, y_activity
    del date_idx, symbol_idx, y_tradable, growth, outputs, loss, parts, gradients
    torch.cuda.empty_cache()
    gc.collect()
    return result


def _is_cuda_oom(exc: BaseException) -> bool:
    return isinstance(exc, torch.cuda.OutOfMemoryError) or (
        isinstance(exc, RuntimeError)
        and "out of memory" in str(exc).lower()
        and "cuda" in str(exc).lower()
    )


def _run_batch1024_probe(study_root: Path) -> dict[str, Any]:
    root = study_root.resolve()
    result_path = root / "batch1024_probe.json"
    if result_path.is_file():
        result = _read_json(result_path)
        if str(result.get("status", "")) == "success":
            _set_state(root, "batch1024_training", error=None)
        elif str(result.get("status", "")) == "infeasible":
            _set_state(root, "batch1024_infeasible", error=None)
        return result
    decisions = _read_json(root / "stage_decisions.json")
    winner_id = str(dict(decisions["batch512_selection"])["winner_model_id"])
    winner = MODEL_SPECS[winner_id]
    probe_view = _view_for(root, winner, 2023)
    dataset = training.SequencePathPackDataset(
        _read_json(probe_view),
        split="train",
        max_samples=0,
        input_channel_profile=winner.input_channel_profile,
    )
    sampler = training.DateGroupedBatchSampler(
        dataset.sample_index, batch_size=1024, shuffle=False, seed=SEED
    )
    indices = next((value for value in sampler if len(value) == 1024), None)
    if indices is None:
        raise ValueError("no complete real batch1024 exists in the training view")
    batch = dataset.get_batch(
        indices,
        include_ohlcva_path=False,
        include_richer_path=False,
        include_summary=True,
        include_metadata=False,
        include_observation=False,
        include_execution=False,
        include_activity_path=True,
    )
    attempts: list[dict[str, Any]] = []
    selected_profile: str | None = None
    for profile in (
        training.ACTIVATION_CHECKPOINT_PROFILE_NONE,
        training.ACTIVATION_CHECKPOINT_PROFILE_STRUCTURED_GRU,
    ):
        try:
            attempt = _probe_backward(
                dataset=dataset,
                batch=batch,
                spec=winner,
                activation_profile=profile,
            )
        except Exception as exc:
            if not _is_cuda_oom(exc):
                raise
            attempts.append(
                {
                    "status": "oom",
                    "activation_checkpoint_profile": profile,
                    "error": str(exc),
                }
            )
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            continue
        attempts.append(attempt)
        selected_profile = profile
        break
    del batch, dataset
    gc.collect()
    if selected_profile is None:
        result = {
            "schema_version": 1,
            "status": "infeasible",
            "created_at": _now(),
            "winner_batch512_model_id": winner_id,
            "batch_size": 1024,
            "attempts": attempts,
            "tried_batch768": False,
            "used_gradient_accumulation": False,
        }
        _write_json(result_path, result)
        _set_state(root, "batch1024_infeasible", error=None)
        return result
    batch_spec = replace(
        winner,
        model_id=f"{winner.model_id}_b1024",
        phase="batch1024",
        batch_size=1024,
        activation_checkpoint_profile=selected_profile,
        reuse_legacy=False,
    )
    decisions["batch1024_spec"] = batch_spec.to_dict()
    decisions["batch1024_probe"] = {
        "status": "success",
        "activation_checkpoint_profile": selected_profile,
        "source_batch512_model_id": winner_id,
    }
    decisions["updated_at"] = _now()
    _write_json(root / "stage_decisions.json", decisions)
    result = {
        "schema_version": 1,
        "status": "success",
        "created_at": _now(),
        "winner_batch512_model_id": winner_id,
        "batch1024_model_id": batch_spec.model_id,
        "batch_size": 1024,
        "selected_activation_checkpoint_profile": selected_profile,
        "attempts": attempts,
        "tried_batch768": False,
        "used_gradient_accumulation": False,
    }
    _write_json(result_path, result)
    _set_state(root, "batch1024_training", error=None)
    return result


def _selection_by_model(selection: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for raw in list(selection.get("frontier", []) or []):
        row = dict(raw)
        model_id = str(row["model_id"])
        selected = dict(row["selected"])
        previous = output.get(model_id)
        if previous is None or float(selected["annualized_log_growth"]) > float(
            dict(previous["selected"])["annualized_log_growth"]
        ):
            output[model_id] = row
    return output


def _fixed_horizon_robustness(capital: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    fixed = capital[capital["policy_kind"].eq("fixed")].copy()
    for keys, group in fixed.groupby(
        ["model_id", "top_k", "slot_count", "cost_scenario"], sort=True
    ):
        group = group.sort_values("fixed_day", kind="mergesort")
        if len(group) != len(FIXED_EXIT_DAYS):
            raise ValueError(f"fixed-horizon grid is incomplete for {keys}")
        best = group.sort_values(
            "annualized_log_growth", ascending=False, kind="mergesort"
        ).iloc[0]
        best_day = int(best["fixed_day"])
        adjacent = group[
            group["fixed_day"].astype(int).between(best_day - 2, best_day + 2)
        ]
        values = pd.to_numeric(group["annualized_log_growth"], errors="coerce")
        rows.append(
            {
                "model_id": str(keys[0]),
                "top_k": int(keys[1]),
                "slot_count": int(keys[2]),
                "cost_scenario": str(keys[3]),
                "best_fixed_day": best_day,
                "best_annualized_log_growth": float(best["annualized_log_growth"]),
                "adjacent_plus_minus_2_mean_growth": float(
                    pd.to_numeric(adjacent["annualized_log_growth"], errors="coerce").mean()
                ),
                "fixed_horizon_mean_growth": float(values.mean()),
                "fixed_horizon_median_growth": float(values.median()),
                "positive_growth_horizon_rate": float((values > 0.0).mean()),
                "fixed_horizon_count": int(len(group)),
            }
        )
    return pd.DataFrame(rows)


def _factorial_effects(selection: Mapping[str, Any]) -> pd.DataFrame:
    by_model = _selection_by_model(selection)
    required = set(MODEL_SPECS)
    if not required.issubset(by_model):
        raise ValueError("the 2x2 factorial selection is incomplete")
    value = {
        model_id: float(dict(by_model[model_id]["selected"])["annualized_log_growth"])
        for model_id in required
    }
    rows = [
        {
            "effect": "intraday_main_effect_v2",
            "contrast": "L43V2-L35V2",
            "annualized_log_growth_delta": value["l43v2"] - value["l35v2"],
        },
        {
            "effect": "intraday_main_effect_v3",
            "contrast": "L43V3-L35V3",
            "annualized_log_growth_delta": value["l43v3"] - value["l35v3"],
        },
        {
            "effect": "v3_main_effect_35",
            "contrast": "L35V3-L35V2",
            "annualized_log_growth_delta": value["l35v3"] - value["l35v2"],
        },
        {
            "effect": "v3_main_effect_43",
            "contrast": "L43V3-L43V2",
            "annualized_log_growth_delta": value["l43v3"] - value["l43v2"],
        },
        {
            "effect": "intraday_v3_interaction",
            "contrast": "(L43V3-L35V3)-(L43V2-L35V2)",
            "annualized_log_growth_delta": (
                value["l43v3"] - value["l35v3"]
            )
            - (value["l43v2"] - value["l35v2"]),
        },
    ]
    return pd.DataFrame(rows)


def _bootstrap_non_circular(
    values: np.ndarray, *, block_length: int = 60, repetitions: int = 10_000
) -> dict[str, Any]:
    numeric = np.asarray(values, dtype=np.float64)
    numeric = numeric[np.isfinite(numeric)]
    if numeric.size < block_length:
        return {
            "status": "insufficient",
            "observation_count": int(numeric.size),
            "block_length": int(block_length),
            "repetitions": 0,
        }
    starts = np.arange(0, numeric.size - block_length + 1, dtype=np.int64)
    blocks_needed = int(math.ceil(numeric.size / block_length))
    rng = np.random.default_rng(SEED)
    means = np.empty(repetitions, dtype=np.float64)
    for index in range(repetitions):
        sampled = rng.choice(starts, size=blocks_needed, replace=True)
        series = np.concatenate(
            [numeric[start : start + block_length] for start in sampled]
        )[: numeric.size]
        means[index] = float(series.mean())
    return {
        "status": "diagnostic_only",
        "observation_count": int(numeric.size),
        "block_length": int(block_length),
        "non_circular": True,
        "repetitions": int(repetitions),
        "seed": SEED,
        "observed_mean": float(numeric.mean()),
        "bootstrap_mean": float(means.mean()),
        "ci_2_5": float(np.quantile(means, 0.025)),
        "ci_97_5": float(np.quantile(means, 0.975)),
        "positive_probability": float((means > 0.0).mean()),
        "selection_gate": False,
    }


def _selected_account_curve(
    study_root: Path, spec: ModelSpec, frontier_row: Mapping[str, Any]
) -> pd.DataFrame:
    root = study_root.resolve()
    selected = dict(frontier_row["selected"])
    book, _hashes = _forecast_book(root, spec)
    top_book = retired_global._top_k_book(book, int(selected["top_k"]))
    market = retired_global._market(_view_for(root, spec, DEVELOPMENT_YEARS[0]))
    policy = finite.PolicySpec(
        str(selected["policy_name"]),
        str(selected["policy_kind"]),
        int(selected["fixed_day"]) if selected.get("fixed_day") is not None else None,
    )
    dates = tuple(book.signal_date_indices)
    _metric, equity, _trades, _annual = finite.simulate_portfolio(
        market=market,
        book=top_book,
        raw_top3_paths={},
        policy=policy,
        slots=int(selected["slot_count"]),
        cost_scenario=PRIMARY_COST_SCENARIO,
        first_signal_date_idx=int(dates[0]),
        last_signal_date_idx=int(dates[-1]),
        starting_cash=finite.STARTING_CASH_CNY,
        memory_guard=finite._MemoryGuard(),
        allow_pyramiding=False,
    )
    frame = equity[
        equity["date_idx"].astype(int).between(int(dates[0]), int(dates[-1]))
    ].copy()
    values = frame["equity"].to_numpy(dtype=np.float64)
    previous = np.r_[finite.STARTING_CASH_CNY, values[:-1]]
    frame["daily_log_return"] = np.log(values / previous)
    frame["model_id"] = spec.model_id
    frame["policy_name"] = str(selected["policy_name"])
    frame["top_k"] = int(selected["top_k"])
    frame["slot_count"] = int(selected["slot_count"])
    return frame[
        [
            "trade_date",
            "date_idx",
            "model_id",
            "policy_name",
            "top_k",
            "slot_count",
            "equity",
            "daily_log_return",
        ]
    ]


def _pairwise_diagnostics(
    study_root: Path,
    specs: Sequence[ModelSpec],
    selection: Mapping[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    root = study_root.resolve()
    by_model = _selection_by_model(selection)
    baseline_id = "l35v2"
    baseline_curve = _selected_account_curve(
        root, MODEL_SPECS[baseline_id], by_model[baseline_id]
    )
    baseline_unit = pd.read_csv(
        root / "evaluations" / baseline_id / "unit_time_alpha_daily.csv"
    )
    baseline_selected = dict(by_model[baseline_id]["selected"])
    baseline_unit = baseline_unit[
        baseline_unit["top_k"].astype(int).eq(int(baseline_selected["top_k"]))
        & baseline_unit["policy_name"].eq(str(baseline_selected["policy_name"]))
        & baseline_unit["cost_scenario"].eq(PRIMARY_COST_SCENARIO)
    ][["trade_date", "annualized_unit_time_alpha"]].rename(
        columns={"annualized_unit_time_alpha": "baseline_value"}
    )
    rows: list[pd.DataFrame] = []
    bootstrap: dict[str, Any] = {}
    for spec in specs:
        if spec.model_id == baseline_id:
            continue
        contender_curve = _selected_account_curve(root, spec, by_model[spec.model_id])
        account = contender_curve[["trade_date", "daily_log_return"]].merge(
            baseline_curve[["trade_date", "daily_log_return"]],
            on="trade_date",
            suffixes=("_contender", "_baseline"),
            validate="one_to_one",
        )
        account["model_id"] = spec.model_id
        account["baseline_model_id"] = baseline_id
        account["metric"] = "account_daily_log_return"
        account["contender_value"] = account["daily_log_return_contender"]
        account["baseline_value"] = account["daily_log_return_baseline"]
        account["delta"] = account["contender_value"] - account["baseline_value"]
        rows.append(
            account[
                [
                    "trade_date",
                    "model_id",
                    "baseline_model_id",
                    "metric",
                    "contender_value",
                    "baseline_value",
                    "delta",
                ]
            ]
        )
        bootstrap[f"{spec.model_id}:account_daily_log_return"] = _bootstrap_non_circular(
            account["delta"].to_numpy(dtype=np.float64)
        )

        selected = dict(by_model[spec.model_id]["selected"])
        unit = pd.read_csv(
            root / "evaluations" / spec.model_id / "unit_time_alpha_daily.csv"
        )
        unit = unit[
            unit["top_k"].astype(int).eq(int(selected["top_k"]))
            & unit["policy_name"].eq(str(selected["policy_name"]))
            & unit["cost_scenario"].eq(PRIMARY_COST_SCENARIO)
        ][["trade_date", "annualized_unit_time_alpha"]].rename(
            columns={"annualized_unit_time_alpha": "contender_value"}
        )
        unit = unit.merge(baseline_unit, on="trade_date", validate="one_to_one")
        unit["model_id"] = spec.model_id
        unit["baseline_model_id"] = baseline_id
        unit["metric"] = "annualized_unit_time_alpha"
        unit["delta"] = unit["contender_value"] - unit["baseline_value"]
        rows.append(
            unit[
                [
                    "trade_date",
                    "model_id",
                    "baseline_model_id",
                    "metric",
                    "contender_value",
                    "baseline_value",
                    "delta",
                ]
            ]
        )
        bootstrap[f"{spec.model_id}:annualized_unit_time_alpha"] = _bootstrap_non_circular(
            unit["delta"].to_numpy(dtype=np.float64)
        )
    return pd.concat(rows, ignore_index=True), bootstrap


def _markdown_summary(summary: Mapping[str, Any]) -> str:
    winner = dict(summary["selection"]["winner"])
    lines = [
        "# Structured Local Intraday / Capital Speed Result",
        "",
        f"- Status: {summary['status']}.",
        f"- Selected model: `{summary['selection']['winner_model_id']}`.",
        (
            f"- Selected execution: Top{int(winner['top_k'])}, "
            f"{int(winner['slot_count'])} slots, `{winner['policy_name']}`."
        ),
        f"- Double-slippage annualized log growth: {float(winner['annualized_log_growth']):.4%}.",
        (
            f"- Total return: {float(winner['signal_period_total_return']):.4%}; "
            f"drawdown: {float(winner['signal_period_maximum_drawdown']):.4%}."
        ),
        "- D7 is one ordinary fixed horizon; autonomous exits use the model's own ranking and path.",
        "- The 60-day moving-block intervals are diagnostics, not promotion gates.",
        "- No QDP/provider/live-state/main-report/2026 change was made.",
    ]
    return "\n".join(lines) + "\n"


def summarize_structured_local_intraday_capital_speed(
    *, study_root: Path = STUDY_ROOT
) -> dict[str, Any]:
    root = study_root.resolve()
    state = _load_state(root)
    status = str(state.get("status", ""))
    allowed = {
        "factorial_summarizing",
        "batch1024_probing",
        "batch1024_training",
        "batch1024_evaluating",
        "batch1024_infeasible",
        "completed",
    }
    if status not in allowed:
        raise ValueError("the four-model factorial evaluation is not complete")
    specs: list[ModelSpec] = [MODEL_SPECS[key] for key in MODEL_SPECS]
    batch_spec = _batch1024_spec(root)
    if batch_spec is not None and _evaluation_valid(
        root / "evaluations" / batch_spec.model_id / "evaluation.json", batch_spec
    ):
        specs.append(batch_spec)
    for spec in specs:
        if not _evaluation_valid(
            root / "evaluations" / spec.model_id / "evaluation.json", spec
        ):
            raise ValueError(f"model evaluation is incomplete: {spec.model_id}")
    capital = pd.concat(
        [
            pd.read_csv(root / "evaluations" / spec.model_id / "capital_metrics.csv")
            for spec in specs
        ],
        ignore_index=True,
    )
    annual = pd.concat(
        [
            pd.read_csv(
                root / "evaluations" / spec.model_id / "capital_annual_metrics.csv"
            )
            for spec in specs
        ],
        ignore_index=True,
    )
    vintage = pd.concat(
        [
            pd.read_csv(root / "evaluations" / spec.model_id / "vintage_metrics.csv")
            for spec in specs
        ],
        ignore_index=True,
    )
    unit = pd.concat(
        [
            pd.read_csv(root / "evaluations" / spec.model_id / "unit_time_alpha.csv")
            for spec in specs
        ],
        ignore_index=True,
    )
    unit_daily = pd.concat(
        [
            pd.read_csv(
                root / "evaluations" / spec.model_id / "unit_time_alpha_daily.csv"
            )
            for spec in specs
        ],
        ignore_index=True,
    )
    selection_all = retired_global._select_models(capital)
    batch512_capital = capital[capital["model_id"].isin(MODEL_SPECS)].copy()
    selection512 = retired_global._select_models(batch512_capital)
    final_selection = selection512
    batch_outcome: dict[str, Any] | None = None
    if batch_spec is not None and batch_spec.model_id in set(capital["model_id"].astype(str)):
        by_model = _selection_by_model(selection_all)
        batch_growth = float(
            dict(by_model[batch_spec.model_id]["selected"])["annualized_log_growth"]
        )
        baseline_growth = float(
            dict(by_model[str(selection512["winner_model_id"])]["selected"])[
                "annualized_log_growth"
            ]
        )
        batch_wins = batch_growth > baseline_growth
        final_selection = selection_all if batch_wins else selection512
        batch_outcome = {
            "model_id": batch_spec.model_id,
            "batch1024_growth": batch_growth,
            "batch512_champion_growth": baseline_growth,
            "strictly_improved": batch_wins,
            "selected": batch_wins,
        }
    factorial = _factorial_effects(selection512)
    robustness = _fixed_horizon_robustness(capital)
    pairwise, uncertainty = _pairwise_diagnostics(root, specs, selection_all)
    _write_csv(root / "capital_metrics.csv", capital)
    _write_csv(root / "capital_annual_metrics.csv", annual)
    _write_csv(root / "annual_metrics.csv", annual)
    _write_csv(root / "vintage_metrics.csv", vintage)
    _write_csv(root / "unit_time_alpha.csv", unit)
    _write_csv(root / "unit_time_alpha_daily.csv", unit_daily)
    _write_csv(root / "factorial_metrics.csv", factorial)
    _write_csv(root / "fixed_horizon_robustness.csv", robustness)
    _write_csv(root / "daily_pairwise_deltas.csv", pairwise)
    retired_global._feature_audit(root)
    decisions_path = root / "stage_decisions.json"
    decisions = _read_json(decisions_path) if decisions_path.is_file() else {
        "schema_version": 1,
        "study_id": STUDY_ID,
    }
    decisions["batch512_selection"] = selection512
    decisions["factorial_effects"] = factorial.to_dict(orient="records")
    decisions["batch1024_outcome"] = batch_outcome
    decisions["final_selection"] = final_selection
    decisions["uncertainty_diagnostics"] = uncertainty
    decisions["updated_at"] = _now()
    _write_json(decisions_path, decisions)
    terminal = status in {"batch1024_infeasible", "completed"} or batch_outcome is not None
    summary = {
        "schema_version": 1,
        "artifact_type": "seq100_structured_local_intraday_capital_speed_summary",
        "status": "completed" if terminal else "batch512_selected_awaiting_batch1024",
        "created_at": _now(),
        "study_id": STUDY_ID,
        "selection": final_selection,
        "batch512_selection": selection512,
        "batch1024_outcome": batch_outcome,
        "factorial_effects": factorial.to_dict(orient="records"),
        "uncertainty_diagnostics": uncertainty,
        "outputs": {
            name: str((root / name).resolve())
            for name in (
                "path_value_preflight.json",
                "path_value_preflight.csv",
                "factorial_metrics.csv",
                "vintage_metrics.csv",
                "annual_metrics.csv",
                "unit_time_alpha.csv",
                "unit_time_alpha_daily.csv",
                "capital_metrics.csv",
                "capital_annual_metrics.csv",
                "fixed_horizon_robustness.csv",
                "daily_pairwise_deltas.csv",
                "feature_audit.json",
                "feature_audit.csv",
            )
        },
        "provider_called": False,
        "qdp_changed": False,
        "live_state_changed": False,
        "main_report_changed": False,
        "trained_2026": False,
    }
    _write_json(root / "comparison_summary.json", summary)
    (root / "comparison_summary.md").write_text(
        _markdown_summary(summary), encoding="utf-8"
    )
    if terminal:
        _set_state(root, "completed", current_task=None, current_evaluation_model=None)
    elif status == "factorial_summarizing":
        _set_state(root, "batch1024_probing", current_task=None)
    return summary


def verify_structured_local_intraday_capital_speed(
    *, study_root: Path = STUDY_ROOT, require_complete: bool = True
) -> dict[str, Any]:
    root = study_root.resolve()
    study_path = root / "study.json"
    if not study_path.is_file():
        raise FileNotFoundError(study_path)
    study = _read_json(study_path)
    if str(study.get("study_id", "")) != STUDY_ID:
        raise ValueError("study identity drifted")
    if _digest(study["contract"]) != str(study.get("contract_sha256", "")):
        raise ValueError("study contract drifted")
    retirement = _read_json(GLOBAL_STUDY_ROOT / "retirement.json")
    global_state = _read_json(GLOBAL_STUDY_ROOT / "state.json")
    if (
        str(retirement.get("status", "")) != "abandoned"
        or str(retirement.get("successor_study_id", "")) != STUDY_ID
        or str(global_state.get("status", "")) != "abandoned"
        or global_state.get("current_task") is not None
    ):
        raise ValueError("retired global study is not safely frozen")
    recorded = dict(dict(study["material"])["protected_snapshot"])
    current = _protected_snapshot()
    for key in (
        "qdp_active",
        "base_pack",
        "overlay",
        "source_study",
        "legacy_l35_checkpoints",
        "legacy_stage3_exists",
        "live_state",
        "old_global_study",
        "old_global_g35v2_2023_checkpoint",
    ):
        if recorded.get(key) != current.get(key):
            raise ValueError(f"protected boundary drifted: {key}")
    views = dict(dict(study["material"])["views"])
    fairness = dict(dict(study["material"])["fairness"])
    for input_dim in (35, 43):
        for year in DEVELOPMENT_YEARS:
            path = Path(str(dict(views[str(input_dim)])[str(year)]))
            result = walkforward.verify_development_walkforward_view(path)
            if str(result.get("status", "")) != "ok":
                raise ValueError(f"view failed verification: {input_dim}/{year}")
            if _fairness(path) != fairness[str(year)]:
                raise ValueError(f"view fairness drifted: {input_dim}/{year}")
            manifest = _read_json(path)
            dataset = training.SequencePathPackDataset(
                manifest,
                split="development",
                max_samples=0,
                input_channel_profile=(
                    training.INPUT_CHANNEL_PROFILE_DAILY_ONLY_TURNOVER_INTRADAY
                    if input_dim == 43
                    else training.INPUT_CHANNEL_PROFILE_DAILY_ONLY_TURNOVER
                ),
                index_role="candidate",
            )
            if int(dataset.input_dim) != input_dim:
                raise ValueError(f"input dimension drifted: {input_dim}/{year}")
            if len(dataset) != EXPECTED_CANDIDATES[year]:
                raise ValueError(f"candidate count drifted: {input_dim}/{year}")
            del dataset
    preflight = _read_json(root / "path_value_preflight.json")
    if str(preflight.get("status", "")) != "ok":
        raise ValueError("path-value preflight is incomplete")
    state = _load_state(root)
    if require_complete and str(state.get("status", "")) != "completed":
        raise ValueError("study is not completed")
    for spec in MODEL_SPECS.values():
        for year in DEVELOPMENT_YEARS:
            complete, partial = _classify_runs(root, spec, year)
            if (require_complete or spec.reuse_legacy) and (len(complete) != 1 or partial):
                raise ValueError(f"training task is not clean: {spec.model_id}/{year}")
        evaluation_path = root / "evaluations" / spec.model_id / "evaluation.json"
        if require_complete and not _evaluation_valid(evaluation_path, spec):
            raise ValueError(f"evaluation is incomplete: {spec.model_id}")
    probe_path = root / "batch1024_probe.json"
    if require_complete:
        if not probe_path.is_file():
            raise FileNotFoundError(probe_path)
        probe = _read_json(probe_path)
        if str(probe.get("status", "")) not in {"success", "infeasible"}:
            raise ValueError("batch1024 probe has no terminal result")
        batch_spec = _batch1024_spec(root)
        if str(probe.get("status", "")) == "success":
            if batch_spec is None:
                raise ValueError("successful batch1024 probe has no model specification")
            for year in DEVELOPMENT_YEARS:
                complete, partial = _classify_runs(root, batch_spec, year)
                if len(complete) != 1 or partial:
                    raise ValueError(f"batch1024 task is not clean: {year}")
            if not _evaluation_valid(
                root / "evaluations" / batch_spec.model_id / "evaluation.json", batch_spec
            ):
                raise ValueError("batch1024 evaluation is incomplete")
        for name in (
            "comparison_summary.json",
            "comparison_summary.md",
            "stage_decisions.json",
            "factorial_metrics.csv",
            "vintage_metrics.csv",
            "annual_metrics.csv",
            "unit_time_alpha.csv",
            "unit_time_alpha_daily.csv",
            "capital_metrics.csv",
            "capital_annual_metrics.csv",
            "fixed_horizon_robustness.csv",
            "daily_pairwise_deltas.csv",
            "feature_audit.json",
            "feature_audit.csv",
        ):
            if not (root / name).is_file():
                raise FileNotFoundError(root / name)
    if require_complete:
        ablation._assert_no_other_research_process()
    return {
        "status": "ok",
        "study": str(study_path),
        "workflow_status": state.get("status"),
        "protected_boundaries_unchanged": True,
        "global_study_frozen": True,
        "view_count": 6,
        "batch1024_terminal": probe_path.is_file(),
        "provider_called": False,
        "qdp_changed": False,
        "trained_2026": False,
    }
