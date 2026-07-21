"""Structured global-tail, intraday, and capital-speed research workflow.

The study is intentionally isolated from the earlier input ablation.  It
reuses immutable fold material and the L35V2 checkpoints, while every new
model owns its ranking and exit path end to end.
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
from daily_research.path_policy import seq100_structured_experiment as structured
from daily_research.path_policy import seq100_structured_input_ablation as ablation
from daily_research.path_policy import seq100_walkforward as walkforward
from daily_research.path_policy.seq100_exit_policy_audit import CandidateCompleteAuditPack
from daily_research.path_policy.seq100_candidate_execution import evaluate_candidate_execution
from daily_research.path_policy.seq100_mainline import build_todayclose_path_only_train_argv


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
PYTHON = Path("C:/Users/ASUS/miniconda3/envs/yolos/python.exe")
STUDY_ID = "seq100_structured_global_intraday_capital_speed_rolling_2023_2025_v1"
STUDY_ROOT = WORKSPACE_ROOT / (
    "daily_research/output/path_policy/studies/"
    "seq100_structured_global_intraday_capital_speed_rolling_2023_2025_v1"
)
SOURCE_STUDY_ROOT = ablation.STUDY_ROOT
OVERLAY_ROOT = ablation.OVERLAY_ROOT
LEGACY_2026_STUDY_ROOT = WORKSPACE_ROOT / (
    "daily_research/output/path_policy/studies/seq100_structured_180x35_2026_fold_v1"
)
QDP_ACTIVE = WORKSPACE_ROOT / "quant_data_platform/data/qdp_v2/active/active.json"
BASE_PACK_MANIFEST = ablation.BASE_PACK_MANIFEST
LIVE_STATE = WORKSPACE_ROOT / "daily_research/output/active_execution_strategy.json"

DEVELOPMENT_YEARS = (2023, 2024, 2025)
OOS_YEAR = 2026
LOOKBACK_DAYS = 180
FORWARD_DAYS = 60
EXECUTION_TAIL_DAYS = 20
SEED = 7
TOP_K_VALUES = (1, 2, 3)
SLOTS_BY_TOP_K = {
    1: (1, 3, 6, 12, 24, 48),
    2: (2, 4, 6, 12, 24, 48),
    3: (3, 6, 12, 24, 48),
}
FIXED_EXIT_DAYS = tuple(range(2, 61))
OWN_EXIT_POLICIES = ("model_plan", "rolling_path")
COST_SCENARIOS = ("base", "double_slippage")
PRIMARY_COST_SCENARIO = "double_slippage"
MEMORY_GUARD_GIB = 0.5
MONITOR_POLL_SECONDS = 5.0
STALE_SECONDS = 15 * 60
EXPECTED_QDP_AS_OF = "2026-07-16"
OOS_START = "2026-01-05"
OOS_END = "2026-03-19"


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    input_dim: int
    input_channel_profile: str
    rank_training_profile: str
    path_value_semantic: str
    phase: str
    reuse_legacy: bool = False

    @property
    def input_variant_id(self) -> str:
        return (
            "lookback180_turnover_intraday"
            if int(self.input_dim) == 43
            else "lookback180_turnover"
        )

    @property
    def is_global(self) -> bool:
        return self.rank_training_profile == training.RANK_TRAINING_PROFILE_GLOBAL_TAIL_512

    @property
    def is_v3(self) -> bool:
        return self.path_value_semantic == training.PATH_VALUE_SEMANTIC_CAPITAL_SPEED_V3

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


V2_SPECS = {
    "l35v2": ModelSpec(
        "l35v2", 35, training.INPUT_CHANNEL_PROFILE_DAILY_ONLY_TURNOVER,
        training.RANK_TRAINING_PROFILE_LOCAL_CHUNK,
        training.PATH_VALUE_SEMANTIC_V2, "v2", True,
    ),
    "g35v2": ModelSpec(
        "g35v2", 35, training.INPUT_CHANNEL_PROFILE_DAILY_ONLY_TURNOVER,
        training.RANK_TRAINING_PROFILE_GLOBAL_TAIL_512,
        training.PATH_VALUE_SEMANTIC_V2, "v2",
    ),
    "l43v2": ModelSpec(
        "l43v2", 43, training.INPUT_CHANNEL_PROFILE_DAILY_ONLY_TURNOVER_INTRADAY,
        training.RANK_TRAINING_PROFILE_LOCAL_CHUNK,
        training.PATH_VALUE_SEMANTIC_V2, "v2",
    ),
    "g43v2": ModelSpec(
        "g43v2", 43, training.INPUT_CHANNEL_PROFILE_DAILY_ONLY_TURNOVER_INTRADAY,
        training.RANK_TRAINING_PROFILE_GLOBAL_TAIL_512,
        training.PATH_VALUE_SEMANTIC_V2, "v2",
    ),
}


def v3_spec_for(v2_model_id: str) -> ModelSpec:
    base = V2_SPECS[str(v2_model_id)]
    return ModelSpec(
        model_id=f"{str(v2_model_id)[:3]}v3",
        input_dim=base.input_dim,
        input_channel_profile=base.input_channel_profile,
        rank_training_profile=base.rank_training_profile,
        path_value_semantic=training.PATH_VALUE_SEMANTIC_CAPITAL_SPEED_V3,
        phase="v3",
    )


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
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        default=_json_default, allow_nan=False,
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
    staging = path.with_name(path.name + ".tmp")
    staging.write_text(
        json.dumps(dict(payload), ensure_ascii=False, indent=2, default=_json_default, allow_nan=False) + "\n",
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
        handle.write(json.dumps(event, ensure_ascii=False, default=_json_default, allow_nan=False) + "\n")
    _write_json(root / "monitor.json", event)
    print(f"structured-global: {event.get('message', event.get('event', 'status'))}", flush=True)


def _snapshot_file(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "exists": path.is_file(),
        "size": int(path.stat().st_size) if path.is_file() else None,
        "sha256": _file_sha256(path) if path.is_file() else None,
    }


def _legacy_l35_run(year: int) -> Path:
    roots = sorted(
        (SOURCE_STUDY_ROOT / "runs/stage_2").glob(
            f"seq100_structured_ablation_s2_lookback180_turnover_{int(year)}_seed7_*"
        )
    )
    valid = [
        path for path in roots
        if (path / "best_model.pt").is_file()
        and (path / "sequence_path_training_summary.json").is_file()
        and (path / "progress.json").is_file()
        and str(_read_json(path / "progress.json").get("status", "")) == "completed"
    ]
    if len(valid) != 1:
        raise ValueError(f"expected one reusable L35V2 run for {year}, found {len(valid)}")
    return valid[0].resolve()


def _legacy_l35_2026_run() -> Path:
    roots = sorted((LEGACY_2026_STUDY_ROOT / "runs/development").glob("seq100_structured_180x35_2026_seed7_*"))
    valid = [path for path in roots if (path / "best_model.pt").is_file() and (path / "sequence_path_training_summary.json").is_file()]
    if len(valid) != 1:
        raise ValueError(f"expected one reusable 2026 L35V2 run, found {len(valid)}")
    return valid[0].resolve()


def _protected_snapshot() -> dict[str, Any]:
    active = _read_json(QDP_ACTIVE)
    if str(active.get("active_as_of_date", "")) != EXPECTED_QDP_AS_OF:
        raise ValueError("QDP active_as_of drifted")
    source_study = _read_json(SOURCE_STUDY_ROOT / "study.json")
    if (SOURCE_STUDY_ROOT / "runs/stage_3").exists():
        raise ValueError("the protected prior Stage 3 has started")
    return {
        "qdp_active": _snapshot_file(QDP_ACTIVE),
        "qdp_active_as_of": EXPECTED_QDP_AS_OF,
        "base_pack": _snapshot_file(BASE_PACK_MANIFEST),
        "overlay": _snapshot_file(OVERLAY_ROOT / "manifest.json"),
        "source_study": _snapshot_file(SOURCE_STUDY_ROOT / "study.json"),
        "source_study_contract_sha256": str(source_study["contract_sha256"]),
        "legacy_l35_checkpoints": {
            str(year): _snapshot_file(_legacy_l35_run(year) / "best_model.pt")
            for year in DEVELOPMENT_YEARS
        },
        "legacy_stage3_exists": False,
        "live_state": _snapshot_file(LIVE_STATE),
    }


def _semantic_contract() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "contract_id": STUDY_ID,
        "objective": "maximize_double_slippage_continuous_account_annualized_log_growth",
        "selection_years": list(DEVELOPMENT_YEARS),
        "oos_year": OOS_YEAR,
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
            "maximum_epochs": 10,
            "patience": 1,
            "restore_best_checkpoint": True,
        },
        "profile": {"name": "structured_joint_turnover", "input_dim": 0},
        "input": {
            "lookback_days": LOOKBACK_DAYS,
            "variants": {key: value.to_dict() for key, value in V2_SPECS.items()},
            "intraday_columns": list(ablation.INTRADAY_COLUMNS),
            "candidate_membership_uses_future": False,
        },
        "training": {
            "seed": SEED,
            "maximum_epochs": 10,
            "early_stopping_patience": 1,
            "minimum_complete_epochs": 1,
            "batch_size": 512,
            "path_value_gradient_profile": training.PATH_VALUE_GRADIENT_PROFILE_HARD_ST,
            "global_tail_profile": training.RANK_TRAINING_PROFILE_GLOBAL_TAIL_512,
            "global_tail_allocation": dict(training.GLOBAL_TAIL_GROUP_COUNTS),
            "literal_all_market_tensor": False,
        },
        "capital_speed_v3": {
            "semantic_version": training.PATH_VALUE_SEMANTIC_CAPITAL_SPEED_V3,
            "formula": "log((1+close_return)*date_specific_proportional_cost_multiplier)/capital_days",
            "legal_exit_days": [2, 60],
            "earliest_tie_break": True,
            "minimum_commission_and_lot_constraints": "account_execution_only",
        },
        "evaluation": {
            "top_k": list(TOP_K_VALUES),
            "slots_by_top_k": {str(k): list(v) for k, v in SLOTS_BY_TOP_K.items()},
            "fixed_exit_days": list(FIXED_EXIT_DAYS),
            "own_exit_policies": list(OWN_EXIT_POLICIES),
            "cost_scenarios": list(COST_SCENARIOS),
            "primary_cost_scenario": PRIMARY_COST_SCENARIO,
            "cross_model_rank_exit_hybrid": False,
            "d7_special_weight": False,
            "cash_filter_selection_authority": False,
        },
        "oos_2026": {
            "window": [OOS_START, OOS_END],
            "reselect_model_or_policy": False,
            "score_only_dates_included": False,
        },
        "protected_boundaries": {
            "update_qdp": False,
            "call_provider": False,
            "copy_base_pack": False,
            "modify_old_checkpoint": False,
            "start_old_stage3": False,
            "change_live_state": False,
            "modify_main_report": False,
        },
    }


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


def _clone_view(
    source_path: Path,
    *,
    target_path: Path,
    study_path: Path,
    contract_sha256: str,
    view_id: str,
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
        "view_id": str(view_id),
        "view_type": "structured_global_intraday_development_view",
    }
    view["structured_global_intraday"] = {
        "source_view": str(source_path.resolve()),
        "source_view_sha256": _file_sha256(source_path),
        "base_material_copied": False,
    }
    view["development_fold_training_contract"] = walkforward._compute_development_fold_training_contract(view)
    _write_json(target_path, view)
    result = walkforward.verify_development_walkforward_view(target_path)
    if str(result.get("status", "")) != "ok":
        raise ValueError(f"cloned fold view failed verification: {result.get('blockers')}")
    return target_path.resolve()


def _fairness(view_path: Path) -> dict[str, Any]:
    return ablation._view_fairness_material(_read_json(view_path))


def prepare_structured_global_intraday(*, study_root: Path = STUDY_ROOT) -> dict[str, Any]:
    root = study_root.resolve()
    study_path = root / "study.json"
    if study_path.is_file():
        existing = _read_json(study_path)
        expected_contract = _semantic_contract()
        expected_sha = _digest(expected_contract)
        if str(existing.get("contract_sha256", "")) != expected_sha:
            if any(root.glob("runs/**/best_model.pt")) or any(root.glob("evaluations/**/evaluation.json")):
                raise ValueError("cannot migrate the study contract after training or evaluation")
            existing["contract"] = expected_contract
            existing["contract_sha256"] = expected_sha
            _write_json(study_path, existing)
            _write_json(root / "contract.json", {"contract": expected_contract, "contract_sha256": expected_sha})
            source = _read_json(SOURCE_STUDY_ROOT / "study.json")
            source_views = dict(dict(source["material"])["views"])
            views: dict[str, dict[str, str]] = {"35": {}, "43": {}}
            for input_dim, variant_id in ((35, "lookback180_turnover"), (43, "lookback180_turnover_intraday")):
                for year in DEVELOPMENT_YEARS:
                    source_view = Path(str(dict(source_views[variant_id])[str(year)])).resolve()
                    target = root / "views" / f"input{input_dim}_{year}.json"
                    views[str(input_dim)][str(year)] = str(_clone_view(
                        source_view, target_path=target, study_path=study_path,
                        contract_sha256=expected_sha,
                        view_id=f"{STUDY_ID}_input{input_dim}_{year}",
                    ))
            existing = _read_json(study_path)
            existing["material"]["views"] = views
            existing["material"]["fairness"] = {
                str(year): _fairness(Path(views["35"][str(year)])) for year in DEVELOPMENT_YEARS
            }
            _write_json(study_path, existing)
            _set_state(root, "prepared", completed_training_tasks=[], completed_evaluation_models=[], error=None)
        verify_structured_global_intraday(study_root=root, require_complete=False)
        return _read_json(study_path)
    ablation._assert_no_other_research_process()
    source = _read_json(SOURCE_STUDY_ROOT / "study.json")
    overlay = ablation._validate_overlay_manifest(OVERLAY_ROOT / "manifest.json")
    contract = _semantic_contract()
    contract_sha = _digest(contract)
    root.mkdir(parents=True, exist_ok=True)
    study = {
        "schema_version": 1,
        "artifact_type": "seq100_structured_global_intraday_study",
        "study_id": STUDY_ID,
        "created_at": _now(),
        "contract": contract,
        "contract_sha256": contract_sha,
        "material": {},
    }
    _write_json(study_path, study)
    _write_json(root / "contract.json", {"contract": contract, "contract_sha256": contract_sha})
    try:
        source_views = dict(dict(source["material"])["views"])
        views: dict[str, dict[str, str]] = {"35": {}, "43": {}}
        for input_dim, variant_id in ((35, "lookback180_turnover"), (43, "lookback180_turnover_intraday")):
            for year in DEVELOPMENT_YEARS:
                source_view = Path(str(dict(source_views[variant_id])[str(year)])).resolve()
                target = root / "views" / f"input{input_dim}_{year}.json"
                views[str(input_dim)][str(year)] = str(
                    _clone_view(
                        source_view,
                        target_path=target,
                        study_path=study_path,
                        contract_sha256=contract_sha,
                        view_id=f"{STUDY_ID}_input{input_dim}_{year}",
                    )
                )
        fairness = {str(year): _fairness(Path(views["35"][str(year)])) for year in DEVELOPMENT_YEARS}
        for year in DEVELOPMENT_YEARS:
            if _fairness(Path(views["43"][str(year)])) != fairness[str(year)]:
                raise ValueError(f"35/43 fairness material differs for {year}")
        snapshot = _protected_snapshot()
        study["material"] = {
            "views": views,
            "fairness": fairness,
            "overlay": str((OVERLAY_ROOT / "manifest.json").resolve()),
            "overlay_sha256": _file_sha256(OVERLAY_ROOT / "manifest.json"),
            "overlay_validated_size_bytes": int(overlay["validated_size_bytes"]),
            "legacy_l35_runs": {str(y): str(_legacy_l35_run(y)) for y in DEVELOPMENT_YEARS},
            "protected_snapshot": snapshot,
            "base_material_copied": False,
        }
        _write_json(study_path, study)
        _set_state(root, "prepared", completed_training_tasks=[], completed_evaluation_models=[])
        _append_event(root, {"event": "prepared", "status": "prepared", "message": "study prepared"})
        return study
    except Exception as exc:
        _set_state(root, "prepare_failed", error=str(exc))
        raise


def _all_declared_specs(study_root: Path) -> dict[str, ModelSpec]:
    specs = dict(V2_SPECS)
    decisions_path = study_root.resolve() / "stage_decisions.json"
    if decisions_path.is_file():
        payload = _read_json(decisions_path)
        for raw in list(payload.get("v3_models", []) or []):
            spec = ModelSpec(**dict(raw))
            specs[spec.model_id] = spec
    return specs


def _v3_specs(v2_champion: str) -> dict[str, ModelSpec]:
    ids = ["g35v2", "g43v2"]
    if str(v2_champion) in {"l35v2", "l43v2"}:
        ids.append(str(v2_champion))
    output = {spec.model_id: spec for spec in (v3_spec_for(model_id) for model_id in ids)}
    if len(output) != len(ids):
        raise AssertionError("duplicate V3 model specification")
    return output


def _task_id(spec: ModelSpec, year: int) -> str:
    return f"{spec.model_id}:{int(year)}"


def _run_tag(spec: ModelSpec, year: int) -> str:
    return f"seq100_structured_global_{spec.model_id}_{int(year)}_seed{SEED}"


def _task_run_dirs(study_root: Path, spec: ModelSpec, year: int) -> list[Path]:
    return sorted(
        (study_root.resolve() / "runs" / spec.model_id).glob(f"{_run_tag(spec, year)}_*")
    )


def _view_for(study_root: Path, spec: ModelSpec, year: int) -> Path:
    study = _read_json(study_root.resolve() / "study.json")
    if int(year) == OOS_YEAR:
        return Path(str(dict(dict(study["material"])["oos_views"])[str(spec.input_dim)])).resolve()
    return Path(str(dict(dict(study["material"])["views"])[str(spec.input_dim)][str(year)])).resolve()


def _run_dir_for(study_root: Path, spec: ModelSpec, year: int) -> Path:
    if int(year) == OOS_YEAR and spec.model_id == "l35v2" and spec.reuse_legacy:
        return _legacy_l35_2026_run()
    if spec.reuse_legacy:
        return _legacy_l35_run(year)
    complete, _partial = _classify_runs(study_root, spec, year)
    if len(complete) != 1:
        raise ValueError(f"missing completed run for {spec.model_id}/{year}")
    return complete[0]


def _validate_run(study_root: Path, spec: ModelSpec, year: int, run_dir: Path) -> dict[str, Any]:
    summary_path = run_dir / "sequence_path_training_summary.json"
    progress_path = run_dir / "progress.json"
    checkpoint_path = run_dir / "best_model.pt"
    if not all(path.is_file() for path in (summary_path, progress_path, checkpoint_path)):
        raise FileNotFoundError(run_dir)
    summary = development._validate_run_summary(summary_path, int(year))
    progress = _read_json(progress_path)
    if str(progress.get("status", "")) != "completed":
        raise ValueError("training progress is not completed")
    if Path(str(summary["best_checkpoint"])).resolve() != checkpoint_path.resolve():
        raise ValueError("best checkpoint path drifted")
    if _file_sha256(checkpoint_path) != str(summary.get("best_checkpoint_sha256", "")):
        raise ValueError("best checkpoint hash drifted")
    resolved = dict(summary.get("resolved_training_config", {}) or {})
    expected = {
        "model_type": "gru_structured_joint_turnover",
        "input_channel_profile": spec.input_channel_profile,
        "rank_training_profile": spec.rank_training_profile,
        "path_value_gradient_profile": training.PATH_VALUE_GRADIENT_PROFILE_HARD_ST,
        "seed": SEED,
        "early_stopping_patience": 1,
    }
    for key, value in expected.items():
        observed = resolved.get(key, summary.get(key))
        if key == "path_value_semantic" and observed is None:
            observed = training.PATH_VALUE_SEMANTIC_V2
        if observed != value:
            raise ValueError(f"training config drifted for {key}: {observed!r} != {value!r}")
    observed_semantic = str(
        resolved.get("path_value_semantic")
        or summary.get("path_value_semantic")
        or training.PATH_VALUE_SEMANTIC_V2
    )
    if observed_semantic != spec.path_value_semantic:
        raise ValueError("path-value semantic drifted")
    if int(summary.get("lookback_days", 0)) != LOOKBACK_DAYS:
        raise ValueError("lookback drifted")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if int(checkpoint.get("input_dim", 0)) != spec.input_dim:
        raise ValueError("checkpoint input dimension drifted")
    if tuple(dict(checkpoint["model_state_dict"])["proj.weight"].shape) != (128, spec.input_dim):
        raise ValueError("checkpoint projection shape drifted")
    if Path(str(summary["pack_manifest"])).resolve() != _view_for(study_root, spec, year):
        raise ValueError("training used the wrong fold view")
    if int(year) == OOS_YEAR:
        if not bool(resolved.get("development_fixed_final_epoch", False)):
            raise ValueError("2026 checkpoint was not fixed before OOS labels")
        if str(summary.get("checkpoint_policy", "")) != "fixed_final_development_epoch":
            raise ValueError("2026 checkpoint policy used OOS labels for selection")
    return summary


def _classify_runs(study_root: Path, spec: ModelSpec, year: int) -> tuple[list[Path], list[Path]]:
    if int(year) == OOS_YEAR and spec.model_id == "l35v2" and spec.reuse_legacy:
        return [_legacy_l35_2026_run()], []
    if spec.reuse_legacy:
        return [_legacy_l35_run(year)], []
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


def _archive_partial(study_root: Path, spec: ModelSpec, year: int, paths: Sequence[Path]) -> None:
    if not paths:
        return
    target_root = study_root.resolve() / "runs/failed" / f"{spec.model_id}__{year}"
    target_root.mkdir(parents=True, exist_ok=True)
    for number, source in enumerate(paths, start=1):
        if study_root.resolve() not in source.resolve().parents:
            raise ValueError("refusing to archive a run outside the study")
        target = target_root / f"{source.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{number}"
        shutil.move(str(source), str(target))


def _parse_time(value: Any) -> float | None:
    try:
        return datetime.fromisoformat(str(value)).timestamp()
    except (TypeError, ValueError):
        return None


def _supervise(command: Sequence[str], *, study_root: Path, spec: ModelSpec, year: int) -> Path:
    root = study_root.resolve()
    task = _task_id(spec, year)
    log_root = root / "logs"
    log_root.mkdir(parents=True, exist_ok=True)
    memory_path = log_root / f"memory_guard_{spec.model_id}_{year}.json"
    stdout_path = log_root / f"worker_{spec.model_id}_{year}.log"
    guarded = [
        str(PYTHON), str((WORKSPACE_ROOT / "tools/memory_guard.py").resolve()),
        "--min-available-gb", str(MEMORY_GUARD_GIB),
        "--interval-seconds", "1", "--consecutive-breaches", "2",
        "--log-json", str(memory_path), "--", *[str(value) for value in command],
    ]
    before = set(_task_run_dirs(root, spec, year))
    _append_event(root, {"event": "task_started", "status": "running", "task_id": task, "message": f"start {task}"})
    run_dir: Path | None = None
    last_key: tuple[str, int, int] | None = None
    stale_reported = False
    with stdout_path.open("w", encoding="utf-8") as stdout:
        process = subprocess.Popen(guarded, cwd=str(WORKSPACE_ROOT), stdout=stdout, stderr=subprocess.STDOUT)
        while process.poll() is None:
            if run_dir is None:
                created = sorted(set(_task_run_dirs(root, spec, year)).difference(before))
                if len(created) > 1:
                    process.kill()
                    raise RuntimeError("worker created multiple run directories")
                run_dir = created[0] if created else None
            progress: dict[str, Any] = {}
            if run_dir is not None and (run_dir / "progress.json").is_file():
                try:
                    progress = _read_json(run_dir / "progress.json")
                except (OSError, json.JSONDecodeError):
                    progress = {}
            phase = str(progress.get("status", "starting"))
            epoch = int(progress.get("epoch", 0) or 0)
            batch = int(progress.get("batch", 0) or 0)
            total = int(progress.get("total_batches", 0) or 0)
            bucket = min(100, int(math.floor(10.0 * batch / total) * 10)) if total else 0
            key = (phase, epoch, bucket)
            if key != last_key:
                _append_event(root, {
                    "event": "progress", "status": "running", "task_id": task,
                    "phase": phase, "epoch": epoch, "batch": batch, "total_batches": total,
                    "batch_percent_bucket": bucket,
                    "available_memory_gib": float(psutil.virtual_memory().available / 1024**3),
                    "message": f"{task} phase={phase} epoch={epoch} batch={batch}/{total}",
                })
                last_key = key
            updated = _parse_time(progress.get("updated_at"))
            stale = updated is not None and time.time() - updated > STALE_SECONDS
            if stale and not stale_reported:
                _append_event(root, {"event": "stale_progress", "status": "warning", "task_id": task, "message": f"{task} progress stale"})
                stale_reported = True
            elif not stale:
                stale_reported = False
            time.sleep(MONITOR_POLL_SECONDS)
        exit_code = int(process.returncode or 0)
    if run_dir is None:
        created = sorted(set(_task_run_dirs(root, spec, year)).difference(before))
        run_dir = created[-1] if created else None
    memory = _read_json(memory_path) if memory_path.is_file() else {}
    if run_dir is not None:
        _write_json(run_dir / "memory_guard.json", memory)
    if exit_code != 0:
        event = "low_memory_terminated" if str(memory.get("status", "")) == "killed_low_available_memory" else "task_failed"
        _append_event(root, {"event": event, "status": "failed", "task_id": task, "exit_code": exit_code, "message": f"{task} failed"})
        raise subprocess.CalledProcessError(exit_code, guarded)
    if run_dir is None:
        raise RuntimeError("worker produced no run directory")
    _validate_run(root, spec, year, run_dir)
    _append_event(root, {"event": "task_completed", "status": "completed", "task_id": task, "run_dir": str(run_dir), "message": f"completed {task}"})
    return run_dir


def _training_specs_for_state(study_root: Path) -> list[ModelSpec]:
    state = _load_state(study_root)
    status = str(state.get("status", ""))
    failed_phase = str(state.get("failed_phase", ""))
    if status in {"prepared", "v2_training"} or (status == "run_failed" and failed_phase == "v2"):
        return [V2_SPECS[key] for key in ("g35v2", "l43v2", "g43v2")]
    if status == "v3_training" or (status == "run_failed" and failed_phase == "v3"):
        specs = _all_declared_specs(study_root)
        return [spec for spec in specs.values() if spec.phase == "v3"]
    if status == "finalized" or (status == "run_failed" and failed_phase == "oos_2026"):
        decisions = _read_json(study_root.resolve() / "stage_decisions.json")
        selected_id = str(dict(dict(decisions["final_selection"])["selection"])["winner_model_id"])
        return [_all_declared_specs(study_root)[selected_id]]
    return []


def _prepare_oos_views(study_root: Path) -> dict[str, str]:
    root = study_root.resolve()
    study_path = root / "study.json"
    study = _read_json(study_path)
    existing = dict(dict(study.get("material", {}) or {}).get("oos_views", {}) or {})
    if set(existing) == {"35", "43"} and all(Path(path).is_file() for path in existing.values()):
        return {key: str(value) for key, value in existing.items()}
    source_path = LEGACY_2026_STUDY_ROOT / "fold_view_2026.json"
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    source = _read_json(source_path)
    overlay = ablation._validate_overlay_manifest(OVERLAY_ROOT / "manifest.json")
    outputs: dict[str, str] = {}
    for input_dim in (35, 43):
        target = root / "views" / f"input{input_dim}_2026.json"
        view = json.loads(json.dumps(source, ensure_ascii=False))
        if input_dim == 43:
            channels = dict(view["feature_channels"])
            channels["intraday_micro"] = dict(dict(overlay["feature_channels"])["intraday_micro"])
            masks = dict(view["masks"])
            masks["intraday_valid"] = dict(dict(overlay["masks"])["intraday_valid"])
            semantics = dict(view["data_semantics"])
            semantics["model_input_mask_features"] = ["turnover_valid", "intraday_valid"]
            semantics["structured_global_intraday_2026"] = {
                "lookback_days": LOOKBACK_DAYS, "input_dim": 43,
                "input_channel_profile": training.INPUT_CHANNEL_PROFILE_DAILY_ONLY_TURNOVER_INTRADAY,
            }
            view["feature_channels"] = channels
            view["masks"] = masks
            view["data_semantics"] = semantics
            view["normalization"] = ablation._normalization_for_view(
                source, overlay, ablation.InputVariant(LOOKBACK_DAYS, turnover=True, intraday=True)
            )
        binding = {
            "contract_id": STUDY_ID,
            "contract_sha256": str(study["contract_sha256"]),
            "contract_file_sha256": str(study["contract_sha256"]),
            "path": str(study_path.resolve()),
        }
        view["created_at"] = _now()
        view["research_contract"] = binding
        view["development_contract"] = binding
        view["artifact_view"] = {
            "schema_version": 1, "view_id": f"{STUDY_ID}_input{input_dim}_2026",
            "view_type": "structured_global_intraday_oos_view",
        }
        view["development_fold_training_contract"] = walkforward._compute_development_fold_training_contract(view)
        _write_json(target, view)
        verification = walkforward.verify_development_walkforward_view(target)
        if str(verification.get("status", "")) != "ok":
            raise ValueError(f"2026 input{input_dim} view failed verification")
        outputs[str(input_dim)] = str(target.resolve())
    if _fairness(Path(outputs["35"])) != _fairness(Path(outputs["43"])):
        raise ValueError("2026 35/43 fairness material differs")
    study = _read_json(study_path)
    study["material"]["oos_views"] = outputs
    study["material"]["oos_fairness"] = _fairness(Path(outputs["35"]))
    _write_json(study_path, study)
    return outputs


def run_structured_global_intraday(
    *, study_root: Path = STUDY_ROOT, max_tasks: int = 1
) -> dict[str, Any]:
    root = study_root.resolve()
    if not (root / "study.json").is_file():
        prepare_structured_global_intraday(study_root=root)
    verify_structured_global_intraday(study_root=root, require_complete=False)
    pre_state = _load_state(root)
    if str(pre_state.get("status", "")) == "finalized" or (
        str(pre_state.get("status", "")) == "run_failed"
        and str(pre_state.get("failed_phase", "")) == "oos_2026"
    ):
        _prepare_oos_views(root)
    specs = _training_specs_for_state(root)
    if not specs:
        return status_structured_global_intraday(study_root=root)
    active_state = _load_state(root)
    oos = str(active_state.get("status", "")) == "finalized" or (
        str(active_state.get("status", "")) == "run_failed"
        and str(active_state.get("failed_phase", "")) == "oos_2026"
    )
    phase = "oos_2026" if oos else specs[0].phase
    if oos and specs[0].reuse_legacy and specs[0].model_id == "l35v2":
        decisions = _read_json(root / "stage_decisions.json")
        decisions["oos_2026_checkpoint"] = {
            "status": "reused", "model_id": "l35v2",
            "run_dir": str(_legacy_l35_2026_run()),
            "checkpoint_sha256": _file_sha256(_legacy_l35_2026_run() / "best_model.pt"),
        }
        _write_json(root / "stage_decisions.json", decisions)
        _set_state(root, "oos_2026_evaluating", current_task=None)
        return status_structured_global_intraday(study_root=root)
    _set_state(root, f"{phase}_training", current_task=None, error=None)
    launched = 0
    try:
        for spec in specs:
            for year in ((OOS_YEAR,) if oos else DEVELOPMENT_YEARS):
                complete, partial = _classify_runs(root, spec, year)
                if complete:
                    continue
                _archive_partial(root, spec, year, partial)
                if int(max_tasks) > 0 and launched >= int(max_tasks):
                    return status_structured_global_intraday(study_root=root)
                oos_epoch_depth = 1
                if oos:
                    prior_best = [
                        int(_read_json(_run_dir_for(root, spec, fold) / "sequence_path_training_summary.json")["best_epoch"])
                        for fold in DEVELOPMENT_YEARS
                    ]
                    oos_epoch_depth = max(
                        2 if spec.is_global else 1,
                        int(round(float(np.median(prior_best)))),
                    )
                profile = replace(
                    ablation._base_structured_profile(512),
                    store_view=_view_for(root, spec, year),
                    output_root=root / "runs" / spec.model_id,
                    run_tag=_run_tag(spec, year),
                    # The 2026 epoch depth is fixed before reading 2026 labels;
                    # one epoch matches the existing OOS contract. Development
                    # folds retain the registered early-stopping search.
                    epochs=oos_epoch_depth if oos else 10,
                    batch_size=512,
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
                    rank_training_profile=spec.rank_training_profile,
                    rank_batch_size=512,
                    development_fixed_final_epoch=bool(oos),
                )
                command = [
                    str(PYTHON), "-m", "daily_research.path_policy.qdp_v2_sequence_path_training",
                    *build_todayclose_path_only_train_argv(profile),
                    "--development-contract", str(root / "study.json"), "--json",
                ]
                _set_state(root, f"{phase}_training", current_task=_task_id(spec, year))
                _supervise(command, study_root=root, spec=spec, year=year)
                launched += 1
                _set_state(root, f"{phase}_training", current_task=None)
        next_status = (
            "oos_2026_evaluating" if oos
            else ("v2_evaluating" if phase == "v2" else "v3_evaluating")
        )
        _set_state(root, next_status, current_task=None)
    except Exception as exc:
        _set_state(root, "run_failed", failed_phase=phase, current_task=None, error=str(exc))
        raise
    return status_structured_global_intraday(study_root=root)


def status_structured_global_intraday(*, study_root: Path = STUDY_ROOT) -> dict[str, Any]:
    root = study_root.resolve()
    if not (root / "study.json").is_file():
        return {"status": "not_prepared", "study_root": str(root)}
    state = _load_state(root)
    tasks: dict[str, Any] = {}
    for spec in _all_declared_specs(root).values():
        for year in DEVELOPMENT_YEARS:
            complete, partial = _classify_runs(root, spec, year)
            evaluation = root / "evaluations" / spec.model_id / "evaluation.json"
            tasks[_task_id(spec, year)] = {
                "status": "reused" if spec.reuse_legacy else ("completed" if complete else ("partial" if partial else "pending")),
                "run_dir": str(complete[0]) if complete else None,
                "partial_runs": [str(path) for path in partial],
                "model_evaluated": evaluation.is_file(),
            }
    return {
        "status": str(state.get("status", "unknown")),
        "study_root": str(root),
        "state": state,
        "tasks": tasks,
        "available_memory_gib": float(psutil.virtual_memory().available / 1024**3),
        "monitor": _read_json(root / "monitor.json") if (root / "monitor.json").is_file() else None,
    }


@dataclass(frozen=True)
class AccountJob:
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
            f"top{self.top_k}__slots{self.slots:02d}__{self.policy_name}__"
            f"{self.cost_scenario}"
        )


def _account_jobs(model_id: str) -> list[AccountJob]:
    jobs: list[AccountJob] = []
    policies = [
        (f"fixed_d{day}", "fixed", int(day)) for day in FIXED_EXIT_DAYS
    ] + [
        ("model_plan", "model_plan", None),
        ("rolling_path", "rolling", None),
    ]
    for top_k in TOP_K_VALUES:
        for slots in SLOTS_BY_TOP_K[top_k]:
            for policy_name, policy_kind, fixed_day in policies:
                for cost in COST_SCENARIOS:
                    jobs.append(AccountJob(model_id, top_k, slots, policy_name, policy_kind, fixed_day, cost))
    return jobs


def _forecast_book(study_root: Path, spec: ModelSpec) -> tuple[finite.ForecastBook, dict[str, str]]:
    book = finite.ForecastBook(spec.model_id)
    hashes: dict[str, str] = {}
    for year in DEVELOPMENT_YEARS:
        view = _read_json(_view_for(study_root, spec, year))
        dataset = training.SequencePathPackDataset(
            view, split="development", max_samples=0,
            input_channel_profile=spec.input_channel_profile, index_role="candidate",
        )
        run_dir = _run_dir_for(study_root, spec, year)
        prediction_path = run_dir / "predictions/development_predictions.csv"
        if not prediction_path.is_file():
            raise FileNotFoundError(prediction_path)
        hashes[str(year)] = _file_sha256(prediction_path)
        frame = pd.read_csv(
            prediction_path,
            usecols=["trade_date", "symbol", "score", "predicted_exit_day"],
            dtype={"trade_date": str, "symbol": str, "score": np.float64},
        )
        candidates = dataset.sample_index
        if len(frame) != len(candidates):
            raise ValueError(f"prediction row count drifted for {spec.model_id}/{year}")
        if not np.array_equal(frame["trade_date"].astype(str), candidates["trade_date"].astype(str)):
            raise ValueError("prediction trade-date order drifted")
        if not np.array_equal(frame["symbol"].astype(str), candidates["symbol"].astype(str)):
            raise ValueError("prediction symbol order drifted")
        score = pd.to_numeric(frame["score"], errors="coerce").to_numpy(dtype=np.float64)
        planned = pd.to_numeric(frame["predicted_exit_day"], errors="coerce").to_numpy(dtype=np.float64)
        if not bool(np.isfinite(score).all()) or not bool(np.isfinite(planned).all()):
            raise ValueError("prediction coverage is incomplete")
        planned = np.clip(np.rint(planned), 2, FORWARD_DAYS).astype(np.int16)
        symbol_idx = candidates["symbol_idx"].to_numpy(dtype=np.int64)
        for date_idx_raw, positions_raw in candidates.groupby("date_idx", sort=True).indices.items():
            positions = np.asarray(positions_raw, dtype=np.int64)
            book.add_day(
                date_idx=int(date_idx_raw), symbol_idx=symbol_idx[positions],
                score=score[positions], planned_day=planned[positions],
            )
        del dataset, frame, candidates, score, planned, symbol_idx
        gc.collect()
    return book, hashes


def _top_k_book(book: finite.ForecastBook, top_k: int) -> finite.ForecastBook:
    output = finite.ForecastBook(f"{book.profile}_top{top_k}")
    for date_idx in book.signal_date_indices:
        day = book.days[int(date_idx)]
        positions = np.argsort(-day.score, kind="mergesort")[: int(top_k)]
        output.days[int(date_idx)] = finite.ForecastDay(
            symbol_idx=day.symbol_idx,
            score=day.score,
            planned_day=day.planned_day,
            top3_symbol_idx=tuple(int(value) for value in day.symbol_idx[positions]),
        )
    return output


def _market(view_path: Path) -> finite.BacktestMarket:
    audit = CandidateCompleteAuditPack(view_path)
    return finite.BacktestMarket(
        date_values=np.asarray(audit.date_values, dtype=object),
        symbol_values=np.asarray(audit.symbol_values, dtype=object),
        entry_open_raw=audit.entry_open_raw,
        exit_close_raw=audit.exit_close_raw,
        exit_sellable=audit.exit_sellable,
        entry_filled=audit.entry_filled,
        contract=audit.contract,
        terminal_recovery_fraction=float(audit.terminal_recovery_fraction),
        forward_days=int(audit.forward_days),
        execution_days=int(audit.execution_days),
    )


def _annualized_log_growth(metric: Mapping[str, Any]) -> float:
    total = float(metric["signal_period_total_return"])
    sessions = int(metric["signal_session_count"])
    return float(math.log1p(total) * 252.0 / sessions) if total > -1.0 and sessions > 0 else float("-inf")


def _trade_quality(trades: pd.DataFrame) -> dict[str, Any]:
    if trades.empty:
        return {
            "winning_trade_rate": 0.0, "payoff_ratio": None,
            "profit_factor": None, "mean_trade_expectancy": 0.0,
        }
    returns = trades["net_return_on_buy_cash"].to_numpy(dtype=np.float64)
    wins = returns[returns > 0.0]
    losses = returns[returns < 0.0]
    mean_win = float(wins.mean()) if wins.size else 0.0
    mean_loss = float(-losses.mean()) if losses.size else 0.0
    gross_profit = float(trades.loc[trades["net_pnl_cny"] > 0.0, "net_pnl_cny"].sum())
    gross_loss = float(-trades.loc[trades["net_pnl_cny"] < 0.0, "net_pnl_cny"].sum())
    return {
        "winning_trade_rate": float((returns > 0.0).mean()),
        "payoff_ratio": float(mean_win / mean_loss) if mean_loss > 0.0 else None,
        "profit_factor": float(gross_profit / gross_loss) if gross_loss > 0.0 else None,
        "mean_trade_expectancy": float(returns.mean()),
    }


def _top2_alpha_for_year(study_root: Path, spec: ModelSpec, year: int) -> float:
    """Compute the missing Top2 cohort row for legacy checkpoints.

    New runs request Top2 directly.  The old L35V2 artifact predates that
    report column, so this small date-streamed adapter derives it from the
    same candidate/execution contract without changing the old files.
    """

    view_path = _view_for(study_root, spec, year)
    view = _read_json(view_path)
    dataset = training.SequencePathPackDataset(
        view, split="development", max_samples=0,
        input_channel_profile=spec.input_channel_profile, index_role="candidate",
    )
    run_dir = _run_dir_for(study_root, spec, year)
    prediction = pd.read_csv(
        run_dir / "predictions/development_predictions.csv",
        usecols=["trade_date", "symbol", "score", "predicted_exit_day"],
        dtype={"trade_date": str, "symbol": str, "score": np.float64},
    )
    audit = CandidateCompleteAuditPack(view_path)
    candidate = dataset.sample_index.reset_index(drop=True)
    if len(candidate) != len(prediction):
        raise ValueError("Top2 candidate row count drifted")
    daily: list[float] = []
    for date_idx_raw, positions_raw in candidate.groupby("date_idx", sort=True).indices.items():
        positions = np.asarray(positions_raw, dtype=np.int64)
        symbols = candidate.iloc[positions]["symbol_idx"].to_numpy(dtype=np.int64)
        entry, close, sellable = audit.execution_paths(int(date_idx_raw), symbols)
        frame = prediction.iloc[positions].copy().reset_index(drop=True)
        frame["entry_filled"] = candidate.iloc[positions]["entry_filled"].to_numpy(dtype=bool)
        frame["entry_open_raw"] = entry
        result = evaluate_candidate_execution(
            frame, close, sellable, manifest=view,
            signal_date_idx=np.full(len(positions), int(date_idx_raw), dtype=np.int64),
            date_values=audit.date_values, top_k_values=(2,),
        )
        row = result.daily_topk.iloc[0]
        daily.append(float(row["alpha_net_realized_plan_return_base"]))
    return float(np.mean(daily)) if daily else float("nan")


def _run_account_job(
    job: AccountJob,
    *,
    market: finite.BacktestMarket,
    book: finite.ForecastBook,
    signal_dates: Sequence[int],
    contract_sha256: str,
) -> dict[str, Any]:
    policy = finite.PolicySpec(job.policy_name, job.policy_kind, job.fixed_day)
    metric, _equity, trades, annual = finite.simulate_portfolio(
        market=market, book=book, raw_top3_paths={}, policy=policy,
        slots=int(job.slots), cost_scenario=job.cost_scenario,
        first_signal_date_idx=int(signal_dates[0]), last_signal_date_idx=int(signal_dates[-1]),
        starting_cash=finite.STARTING_CASH_CNY, memory_guard=finite._MemoryGuard(),
        allow_pyramiding=False,
    )
    occupied = float(
        np.sum(trades["buy_cash_cny"].to_numpy(dtype=np.float64) * trades["occupied_sessions"].to_numpy(dtype=np.float64))
    ) if not trades.empty else 0.0
    net_pnl = float(trades["net_pnl_cny"].sum()) if not trades.empty else 0.0
    if not annual:
        signal_equity = _equity[
            _equity["date_idx"].astype(int).between(int(signal_dates[0]), int(signal_dates[-1]), inclusive="both")
        ]
        values = signal_equity["equity"].to_numpy(dtype=np.float64)
        path = np.r_[finite.STARTING_CASH_CNY, values]
        drawdown = path / np.maximum.accumulate(path) - 1.0
        annual = [{
            "year": int(pd.Timestamp(signal_equity["trade_date"].iloc[0]).year),
            "starting_equity_cny": float(finite.STARTING_CASH_CNY),
            "ending_equity_cny": float(values[-1]),
            "net_return": float(values[-1] / finite.STARTING_CASH_CNY - 1.0),
            "maximum_drawdown": float(drawdown.min()),
            "annualized_volatility": None,
            "sharpe_zero_rate": float(metric["signal_period_sharpe_zero_rate"]),
            "mean_position_count": float(signal_equity["position_count"].mean()),
            "mean_capital_utilization": float(signal_equity["capital_utilization"].mean()),
            "trading_session_count": int(len(signal_equity)),
        }]
    result = {
        "schema_version": 1, "status": "completed", "job_id": job.job_id,
        "contract_sha256": contract_sha256, "job": asdict(job),
        "metric": {
            **metric, **_trade_quality(trades),
            "annualized_log_growth": _annualized_log_growth(metric),
            "worst_calendar_year_return": min(float(row["net_return"]) for row in annual),
            "net_pnl_per_deployed_capital_session": float(net_pnl / occupied) if occupied > 0.0 else None,
        },
        "annual_metrics": annual,
    }
    del _equity, trades
    return result


def _job_valid(path: Path, job: AccountJob, contract_sha256: str) -> bool:
    if not path.is_file():
        return False
    try:
        payload = _read_json(path)
    except (OSError, json.JSONDecodeError):
        return False
    return bool(
        str(payload.get("status", "")) == "completed"
        and str(payload.get("job_id", "")) == job.job_id
        and str(payload.get("contract_sha256", "")) == contract_sha256
    )


def _yearly_model_metrics(study_root: Path, spec: ModelSpec) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for year in DEVELOPMENT_YEARS:
        run_dir = _run_dir_for(study_root, spec, year)
        summary = development._validate_run_summary(run_dir / "sequence_path_training_summary.json", year)
        daily = pd.read_csv(run_dir / "daily_topk_metrics.csv")
        daily = daily[daily["split"].astype(str).eq("development")].copy()
        metrics: dict[str, Any] = {}
        for top_k in (1, 3, 5, 10):
            group = daily[daily["top_k"].astype(int).eq(top_k)]
            if group.empty:
                raise ValueError(f"missing Top{top_k} daily metrics for {spec.model_id}/{year}")
            metrics[f"top{top_k}_base_alpha"] = float(pd.to_numeric(group["alpha_net_realized_plan_return_base"], errors="coerce").mean())
        split = next(
            row for row in list(summary.get("split_metrics", []) or [])
            if str(row.get("split", "")) == "development"
        )
        metrics["rank_ic"] = float(split["rank_ic_mean"])
        metrics["candidate_score_coverage"] = float(split["candidate_complete_score_coverage"])
        top2_group = daily[daily["top_k"].astype(int).eq(2)]
        metrics["top2_base_alpha"] = (
            float(pd.to_numeric(top2_group["alpha_net_realized_plan_return_base"], errors="coerce").mean())
            if not top2_group.empty else _top2_alpha_for_year(study_root, spec, year)
        )
        diagnostics = structured.stream_checkpoint_diagnostics(
            run_dir=run_dir, view_path=_view_for(study_root, spec, year),
            output_dir=run_dir / f"path_diagnostics_{STUDY_ID}", year=year,
            compare_legacy_domain=False, fixed_exit_comparison=False,
            input_channel_profile=spec.input_channel_profile,
            path_value_semantic=spec.path_value_semantic,
        )
        rows.append({
            "model_id": spec.model_id, "year": int(year),
            "rank_ic": float(metrics["rank_ic"]),
            "rank_ic_positive_day_rate": float(split["rank_ic_positive_day_rate"]),
            "path_mae": float(split["path_mae"]),
            "path_open_mae": float(split["path_open_mae"]),
            "path_high_mae": float(split["path_high_mae"]),
            "path_low_mae": float(split["path_low_mae"]),
            "path_close_mae": float(split["path_close_mae"]),
            "top1_base_alpha": float(metrics["top1_base_alpha"]),
            "top2_base_alpha": float(metrics["top2_base_alpha"]),
            "top3_base_alpha": float(metrics["top3_base_alpha"]),
            "top5_base_alpha": float(metrics["top5_base_alpha"]),
            "top10_base_alpha": float(metrics["top10_base_alpha"]),
            "exit_regret": float(diagnostics["exit_regret"]),
            "exit_deferral_rate": float(diagnostics["top3_exit_deferral_rate"]),
            "geometry_violation_count": int(diagnostics["ohlc_geometry_violation_count"]),
            "candidate_count": int(split["row_count"]),
            "date_count": int(split["date_count"]),
            "score_coverage": float(metrics["candidate_score_coverage"]),
            "checkpoint_sha256": _file_sha256(run_dir / "best_model.pt"),
            "prediction_sha256": _file_sha256(run_dir / "predictions/development_predictions.csv"),
        })
    return rows


def _evaluate_model(study_root: Path, spec: ModelSpec) -> dict[str, Any]:
    root = study_root.resolve()
    output = root / "evaluations" / spec.model_id
    output.mkdir(parents=True, exist_ok=True)
    book, prediction_hashes = _forecast_book(root, spec)
    signal_dates = tuple(book.signal_date_indices)
    if not signal_dates:
        raise ValueError("model has no signal dates")
    top_books = {top_k: _top_k_book(book, top_k) for top_k in TOP_K_VALUES}
    market = _market(_view_for(root, spec, DEVELOPMENT_YEARS[0]))
    semantic = {
        "schema_version": 1, "study_contract_sha256": _read_json(root / "study.json")["contract_sha256"],
        "model": spec.to_dict(), "prediction_sha256": prediction_hashes,
        "signal_date_indices": [int(signal_dates[0]), int(signal_dates[-1]), len(signal_dates)],
        "fairness": _read_json(root / "study.json")["material"]["fairness"],
        "top_k": list(TOP_K_VALUES), "slots": {str(k): list(v) for k, v in SLOTS_BY_TOP_K.items()},
        "fixed_days": list(FIXED_EXIT_DAYS), "own_exits": list(OWN_EXIT_POLICIES),
        "cost_scenarios": list(COST_SCENARIOS),
    }
    contract_sha = _digest(semantic)
    _write_json(output / "contract.json", {"contract": semantic, "contract_sha256": contract_sha})
    jobs = _account_jobs(spec.model_id)
    jobs_root = output / "jobs"
    jobs_root.mkdir(exist_ok=True)
    completed = 0
    for number, job in enumerate(jobs, start=1):
        path = jobs_root / f"{job.job_id}.json"
        if not _job_valid(path, job, contract_sha):
            payload = _run_account_job(
                job, market=market, book=top_books[job.top_k],
                signal_dates=signal_dates, contract_sha256=contract_sha,
            )
            _write_json(path, payload)
        completed += 1
        if number == 1 or number % 100 == 0 or number == len(jobs):
            _write_json(output / "progress.json", {
                "status": "running" if number < len(jobs) else "accounts_completed",
                "completed_jobs": number, "total_jobs": len(jobs), "updated_at": _now(),
            })
    metric_rows: list[dict[str, Any]] = []
    annual_rows: list[dict[str, Any]] = []
    for job in jobs:
        payload = _read_json(jobs_root / f"{job.job_id}.json")
        identity = {
            "model_id": spec.model_id, "top_k": job.top_k, "slot_count": job.slots,
            "policy_name": job.policy_name, "policy_kind": job.policy_kind,
            "fixed_day": job.fixed_day, "cost_scenario": job.cost_scenario,
        }
        metric_rows.append({**identity, **dict(payload["metric"])})
        annual_rows.extend({**identity, **dict(row)} for row in payload["annual_metrics"])
    account = pd.DataFrame(metric_rows)
    annual = pd.DataFrame(annual_rows)
    yearly = pd.DataFrame(_yearly_model_metrics(root, spec))
    _write_csv(output / "capital_metrics.csv", account)
    _write_csv(output / "capital_annual_metrics.csv", annual)
    _write_csv(output / "vintage_metrics.csv", yearly)
    evaluation = {
        "schema_version": 1, "artifact_type": "seq100_structured_global_model_evaluation",
        "status": "completed", "completed_at": _now(), "model": spec.to_dict(),
        "contract_sha256": contract_sha, "job_count": len(jobs),
        "prediction_sha256": prediction_hashes,
        "coverage": {"signal_date_count": len(signal_dates), "score_coverage": 1.0},
        "outputs": {
            "capital_metrics": str((output / "capital_metrics.csv").resolve()),
            "capital_annual_metrics": str((output / "capital_annual_metrics.csv").resolve()),
            "vintage_metrics": str((output / "vintage_metrics.csv").resolve()),
        },
        "result_artifact_sha256": _digest({
            name: _file_sha256(output / name)
            for name in ("capital_metrics.csv", "capital_annual_metrics.csv", "vintage_metrics.csv")
        }),
    }
    _write_json(output / "evaluation.json", evaluation)
    _write_json(output / "progress.json", {"status": "completed", "evaluation": str((output / "evaluation.json").resolve()), "updated_at": _now()})
    return evaluation


def _selection_row(row: Mapping[str, Any]) -> dict[str, Any]:
    keys = (
        "model_id", "top_k", "slot_count", "policy_name", "policy_kind", "fixed_day",
        "cost_scenario", "annualized_log_growth", "signal_period_total_return",
        "signal_period_cagr_trading_days", "signal_period_maximum_drawdown",
        "worst_calendar_year_return", "closed_trade_count", "winning_trade_rate",
        "payoff_ratio", "profit_factor", "mean_trade_expectancy",
        "mean_occupied_sessions", "mean_signal_capital_utilization",
        "skipped_no_slot_signal_count", "net_pnl_per_deployed_capital_session",
    )
    return {key: (None if key not in row or pd.isna(row[key]) else row[key].item() if isinstance(row[key], np.generic) else row[key]) for key in keys}


def _select_models(frame: pd.DataFrame) -> dict[str, Any]:
    primary = frame[frame["cost_scenario"].eq(PRIMARY_COST_SCENARIO)].copy()
    if primary.empty:
        raise ValueError("double-slippage account metrics are missing")
    frontier: list[dict[str, Any]] = []
    for (model_id, top_k, slots), group in primary.groupby(["model_id", "top_k", "slot_count"], sort=True):
        fixed = group[group["policy_kind"].eq("fixed")].sort_values("annualized_log_growth", ascending=False, kind="mergesort")
        own = group[group["policy_kind"].isin(["model_plan", "rolling"])].sort_values("annualized_log_growth", ascending=False, kind="mergesort")
        if fixed.empty or own.empty:
            raise ValueError("fixed/own execution comparison is incomplete")
        best_fixed = fixed.iloc[0]
        best_own = own.iloc[0]
        selected = best_own if float(best_own["annualized_log_growth"]) > float(best_fixed["annualized_log_growth"]) else best_fixed
        frontier.append({
            "model_id": str(model_id), "top_k": int(top_k), "slot_count": int(slots),
            "selected_execution": "own_model_exit" if selected is best_own else "fixed_exit_fallback",
            "best_fixed": _selection_row(best_fixed.to_dict()),
            "best_own": _selection_row(best_own.to_dict()),
            "selected": _selection_row(selected.to_dict()),
        })
    winner = max(frontier, key=lambda row: (float(dict(row["selected"])["annualized_log_growth"]), str(row["model_id"])))
    return {
        "primary_cost_scenario": PRIMARY_COST_SCENARIO,
        "frontier": frontier,
        "winner": dict(winner["selected"]),
        "winner_model_id": str(winner["model_id"]),
        "model_owns_ranking_and_exit": True,
        "d7_special_weight": False,
    }


def _aggregate_phase(study_root: Path, phase: str, specs: Sequence[ModelSpec]) -> dict[str, Any]:
    root = study_root.resolve()
    capital = pd.concat(
        [pd.read_csv(root / "evaluations" / spec.model_id / "capital_metrics.csv") for spec in specs],
        ignore_index=True,
    )
    annual = pd.concat(
        [pd.read_csv(root / "evaluations" / spec.model_id / "capital_annual_metrics.csv") for spec in specs],
        ignore_index=True,
    )
    vintage = pd.concat(
        [pd.read_csv(root / "evaluations" / spec.model_id / "vintage_metrics.csv") for spec in specs],
        ignore_index=True,
    )
    selection = _select_models(capital)
    _write_csv(root / f"{phase}_capital_metrics.csv", capital)
    _write_csv(root / f"{phase}_capital_annual_metrics.csv", annual)
    _write_csv(root / f"{phase}_vintage_metrics.csv", vintage)
    return {"phase": phase, "model_ids": [spec.model_id for spec in specs], "selection": selection}


def _oos_forecast_book(study_root: Path, spec: ModelSpec) -> tuple[finite.ForecastBook, str]:
    view = _read_json(_view_for(study_root, spec, OOS_YEAR))
    dataset = training.SequencePathPackDataset(
        view, split="development", max_samples=0,
        input_channel_profile=spec.input_channel_profile, index_role="candidate",
    )
    run_dir = _run_dir_for(study_root, spec, OOS_YEAR)
    path = run_dir / "predictions/development_predictions.csv"
    frame = pd.read_csv(
        path, usecols=["trade_date", "symbol", "score", "predicted_exit_day"],
        dtype={"trade_date": str, "symbol": str, "score": np.float64},
    )
    candidates = dataset.sample_index.reset_index(drop=True)
    if len(frame) != len(candidates):
        raise ValueError("2026 prediction row count drifted")
    if str(frame["trade_date"].min()) != OOS_START or str(frame["trade_date"].max()) != OOS_END:
        raise ValueError("2026 prediction window drifted")
    score = pd.to_numeric(frame["score"], errors="coerce").to_numpy(dtype=np.float64)
    planned = np.clip(
        np.rint(pd.to_numeric(frame["predicted_exit_day"], errors="coerce").to_numpy(dtype=np.float64)),
        2, FORWARD_DAYS,
    ).astype(np.int16)
    if not bool(np.isfinite(score).all()):
        raise ValueError("2026 score coverage is incomplete")
    symbol_idx = candidates["symbol_idx"].to_numpy(dtype=np.int64)
    book = finite.ForecastBook(f"{spec.model_id}_2026")
    for date_idx_raw, positions_raw in candidates.groupby("date_idx", sort=True).indices.items():
        positions = np.asarray(positions_raw, dtype=np.int64)
        book.add_day(
            date_idx=int(date_idx_raw), symbol_idx=symbol_idx[positions],
            score=score[positions], planned_day=planned[positions],
        )
    if len(book.signal_date_indices) != 48:
        raise ValueError(f"2026 signal date count drifted: {len(book.signal_date_indices)}")
    return book, _file_sha256(path)


def _evaluate_oos(study_root: Path) -> dict[str, Any]:
    root = study_root.resolve()
    decisions = _read_json(root / "stage_decisions.json")
    final = dict(dict(decisions["final_selection"])["selection"])
    model_id = str(final["winner_model_id"])
    spec = _all_declared_specs(root)[model_id]
    book, prediction_sha = _oos_forecast_book(root, spec)
    top_k = int(dict(final["winner"])["top_k"])
    slots = int(dict(final["winner"])["slot_count"])
    selected_policy = str(dict(final["winner"])["policy_name"])
    selected_book = _top_k_book(book, top_k)
    market = _market(_view_for(root, spec, OOS_YEAR))
    signal_dates = tuple(book.signal_date_indices)
    output = root / "oos_2026"
    jobs_root = output / "jobs"
    jobs_root.mkdir(parents=True, exist_ok=True)
    semantic = {
        "schema_version": 1, "model": spec.to_dict(), "prediction_sha256": prediction_sha,
        "window": [OOS_START, OOS_END], "signal_date_count": 48,
        "selected_2023_2025_execution": dict(final["winner"]),
        "diagnostic_policies": [*[f"fixed_d{day}" for day in FIXED_EXIT_DAYS], *OWN_EXIT_POLICIES],
        "reselection_authority": False,
    }
    contract_sha = _digest(semantic)
    jobs: list[AccountJob] = []
    for day in FIXED_EXIT_DAYS:
        for cost in COST_SCENARIOS:
            jobs.append(AccountJob(model_id, top_k, slots, f"fixed_d{day}", "fixed", day, cost))
    for name, kind in (("model_plan", "model_plan"), ("rolling_path", "rolling")):
        for cost in COST_SCENARIOS:
            jobs.append(AccountJob(model_id, top_k, slots, name, kind, None, cost))
    for number, job in enumerate(jobs, start=1):
        path = jobs_root / f"{job.job_id}.json"
        if not _job_valid(path, job, contract_sha):
            _write_json(path, _run_account_job(
                job, market=market, book=selected_book,
                signal_dates=signal_dates, contract_sha256=contract_sha,
            ))
        if number == 1 or number % 25 == 0 or number == len(jobs):
            _write_json(output / "progress.json", {
                "status": "running" if number < len(jobs) else "completed",
                "completed_jobs": number, "total_jobs": len(jobs), "updated_at": _now(),
            })
    metrics: list[dict[str, Any]] = []
    annual: list[dict[str, Any]] = []
    for job in jobs:
        payload = _read_json(jobs_root / f"{job.job_id}.json")
        identity = {
            "model_id": model_id, "top_k": top_k, "slot_count": slots,
            "policy_name": job.policy_name, "policy_kind": job.policy_kind,
            "fixed_day": job.fixed_day, "cost_scenario": job.cost_scenario,
        }
        metrics.append({**identity, **dict(payload["metric"])})
        annual.extend({**identity, **dict(row)} for row in payload["annual_metrics"])
    frame = pd.DataFrame(metrics)
    annual_frame = pd.DataFrame(annual)
    _write_csv(output / "capital_metrics.csv", frame)
    _write_csv(output / "capital_annual_metrics.csv", annual_frame)
    selected = frame[
        frame["policy_name"].eq(selected_policy)
        & frame["cost_scenario"].eq(PRIMARY_COST_SCENARIO)
    ]
    if len(selected) != 1:
        raise ValueError("selected 2023-2025 execution is missing from 2026 results")
    diagnostic = frame[frame["cost_scenario"].eq(PRIMARY_COST_SCENARIO)].sort_values(
        "annualized_log_growth", ascending=False, kind="mergesort"
    ).iloc[0]
    result = {
        "schema_version": 1, "status": "completed", "completed_at": _now(),
        "model": spec.to_dict(), "window": [OOS_START, OOS_END], "signal_date_count": 48,
        "candidate_count": int(sum(len(book.days[idx].symbol_idx) for idx in book.signal_date_indices)),
        "selected_2023_2025_execution_result": _selection_row(selected.iloc[0].to_dict()),
        "ex_post_best_diagnostic": _selection_row(diagnostic.to_dict()),
        "ex_post_best_has_selection_authority": False,
        "prediction_sha256": prediction_sha,
        "outputs": {
            "capital_metrics": str((output / "capital_metrics.csv").resolve()),
            "capital_annual_metrics": str((output / "capital_annual_metrics.csv").resolve()),
        },
    }
    _write_json(output / "evaluation.json", result)
    decisions["oos_2026"] = result
    _write_json(root / "stage_decisions.json", decisions)
    _set_state(root, "completed", current_evaluation_model=None)
    return result


def evaluate_structured_global_intraday(
    *, study_root: Path = STUDY_ROOT, max_jobs: int = 1
) -> dict[str, Any]:
    root = study_root.resolve()
    state = _load_state(root)
    status = str(state.get("status", ""))
    if status == "oos_2026_evaluating":
        _evaluate_oos(root)
        summarize_structured_global_intraday(study_root=root)
        return status_structured_global_intraday(study_root=root)
    if status not in {"v2_evaluating", "v3_evaluating", "evaluation_failed"}:
        return status_structured_global_intraday(study_root=root)
    phase = str(state.get("failed_phase", "")) if status == "evaluation_failed" else status[:2]
    specs = [spec for spec in _all_declared_specs(root).values() if spec.phase == phase]
    if phase == "v2":
        specs = [V2_SPECS[key] for key in ("l35v2", "g35v2", "l43v2", "g43v2")]
    launched = 0
    try:
        for spec in specs:
            evaluation_path = root / "evaluations" / spec.model_id / "evaluation.json"
            if evaluation_path.is_file() and str(_read_json(evaluation_path).get("status", "")) == "completed":
                continue
            if int(max_jobs) > 0 and launched >= int(max_jobs):
                return status_structured_global_intraday(study_root=root)
            _set_state(root, f"{phase}_evaluating", current_evaluation_model=spec.model_id)
            _append_event(root, {"event": "evaluation_started", "status": "running", "model_id": spec.model_id, "message": f"evaluate {spec.model_id}"})
            _evaluate_model(root, spec)
            launched += 1
            _append_event(root, {"event": "evaluation_completed", "status": "completed", "model_id": spec.model_id, "message": f"evaluated {spec.model_id}"})
        if not all((root / "evaluations" / spec.model_id / "evaluation.json").is_file() for spec in specs):
            return status_structured_global_intraday(study_root=root)
        phase_summary = _aggregate_phase(root, phase, specs)
        decisions_path = root / "stage_decisions.json"
        decisions = _read_json(decisions_path) if decisions_path.is_file() else {
            "schema_version": 1, "study_id": STUDY_ID,
        }
        decisions[f"{phase}_selection"] = phase_summary
        if phase == "v2":
            v3 = _v3_specs(str(phase_summary["selection"]["winner_model_id"]))
            decisions["v3_models"] = [spec.to_dict() for spec in v3.values()]
            decisions["updated_at"] = _now()
            _write_json(decisions_path, decisions)
            _set_state(root, "v3_training", current_evaluation_model=None)
        else:
            all_specs = [V2_SPECS[key] for key in V2_SPECS] + [spec for spec in _all_declared_specs(root).values() if spec.phase == "v3"]
            final = _aggregate_phase(root, "final", all_specs)
            decisions["final_selection"] = final
            decisions["updated_at"] = _now()
            _write_json(decisions_path, decisions)
            _set_state(root, "finalized", final_model_id=final["selection"]["winner_model_id"], current_evaluation_model=None)
    except Exception as exc:
        _set_state(root, "evaluation_failed", failed_phase=phase, current_evaluation_model=None, error=str(exc))
        raise
    return status_structured_global_intraday(study_root=root)


def unit_time_alpha(
    selected_return: np.ndarray | Sequence[float],
    selected_resolved_day: np.ndarray | Sequence[float],
    universe_return: np.ndarray | Sequence[float],
    universe_resolved_day: np.ndarray | Sequence[float],
) -> float:
    selected = np.asarray(selected_return, dtype=np.float64)
    selected_day = np.asarray(selected_resolved_day, dtype=np.float64)
    universe = np.asarray(universe_return, dtype=np.float64)
    universe_day = np.asarray(universe_resolved_day, dtype=np.float64)
    valid = (
        np.isfinite(selected) & np.isfinite(selected_day)
        & np.isfinite(universe) & np.isfinite(universe_day)
        & (selected > -1.0) & (universe > -1.0)
        & (selected_day > 0.0) & (universe_day > 0.0)
    )
    if not bool(valid.any()):
        return float("nan")
    daily = np.log1p(selected[valid]) / selected_day[valid] - np.log1p(universe[valid]) / universe_day[valid]
    return float(daily.mean() * 252.0)


def _unit_time_rows(study_root: Path, specs: Sequence[ModelSpec]) -> tuple[pd.DataFrame, pd.DataFrame]:
    aggregate: list[dict[str, Any]] = []
    evidence: list[pd.DataFrame] = []
    for spec in specs:
        for year in DEVELOPMENT_YEARS:
            path = _run_dir_for(study_root, spec, year) / "daily_topk_metrics.csv"
            daily = pd.read_csv(path)
            daily = daily[daily["split"].astype(str).eq("development")].copy()
            for top_k in TOP_K_VALUES:
                group = daily[daily["top_k"].astype(int).eq(top_k)].copy()
                if group.empty:
                    continue
                for cost, suffix in (("base", "base"), ("double_slippage", "stress")):
                    selected_col = f"selected_net_realized_plan_return_{suffix}"
                    universe_col = f"universe_net_realized_plan_return_{suffix}"
                    value = unit_time_alpha(
                        group[selected_col], group["selected_realized_plan_exit_day"],
                        group[universe_col], group["universe_realized_plan_exit_day"],
                    )
                    per_day = (
                        np.log1p(pd.to_numeric(group[selected_col], errors="coerce"))
                        / pd.to_numeric(group["selected_realized_plan_exit_day"], errors="coerce")
                        - np.log1p(pd.to_numeric(group[universe_col], errors="coerce"))
                        / pd.to_numeric(group["universe_realized_plan_exit_day"], errors="coerce")
                    ) * 252.0
                    evidence.append(pd.DataFrame({
                        "trade_date": group["trade_date"].astype(str),
                        "model_id": spec.model_id, "year": int(year), "top_k": int(top_k),
                        "policy_name": "model_plan", "cost_scenario": cost,
                        "annualized_unit_time_alpha": per_day,
                    }))
                    aggregate.append({
                        "model_id": spec.model_id, "year": int(year), "top_k": int(top_k),
                        "policy_name": "model_plan", "cost_scenario": cost,
                        "annualized_unit_time_alpha": value,
                        "signal_date_count": int(len(group)),
                    })
    return pd.DataFrame(aggregate), pd.concat(evidence, ignore_index=True) if evidence else pd.DataFrame()


def _feature_audit(study_root: Path) -> dict[str, Any]:
    overlay = ablation._validate_overlay_manifest(OVERLAY_ROOT / "manifest.json")
    intraday_meta = dict(dict(overlay["feature_channels"])["intraday_micro"])
    mask_meta = dict(dict(overlay["masks"])["intraday_valid"])
    values = ablation._open_memmap(intraday_meta, dtype="float32")
    valid = ablation._open_memmap(mask_meta, dtype="bool")
    try:
        valid_count = int(np.asarray(valid, dtype=bool).sum())
        total = int(valid.size)
        rows: list[dict[str, Any]] = []
        for index, name in enumerate(ablation.INTRADAY_COLUMNS):
            channel = np.asarray(values[:, :, index], dtype=np.float32)
            mask = np.asarray(valid, dtype=bool) & np.isfinite(channel)
            rows.append({
                "feature": name, "valid_count": int(mask.sum()),
                "coverage": float(mask.mean()),
                "mean": float(channel[mask].mean()) if bool(mask.any()) else None,
                "std": float(channel[mask].std()) if bool(mask.any()) else None,
            })
    finally:
        ablation._close_memmap(values)
        ablation._close_memmap(valid)
    payload = {
        "schema_version": 1, "artifact_type": "structured_global_intraday_feature_audit",
        "created_at": _now(), "intraday_columns": list(ablation.INTRADAY_COLUMNS),
        "intraday_valid_count": valid_count, "intraday_total_count": total,
        "intraday_valid_coverage": float(valid_count / max(total, 1)),
        "features": rows, "overlay_reused": True, "qdp_rescanned": False,
    }
    _write_json(study_root / "feature_audit.json", payload)
    _write_csv(study_root / "feature_audit.csv", pd.DataFrame(rows))
    return payload


def _path_value_rescore(study_root: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    spec = V2_SPECS["l35v2"]
    for year in DEVELOPMENT_YEARS:
        run = _run_dir_for(study_root, spec, year)
        for semantic in (
            training.PATH_VALUE_SEMANTIC_V2,
            training.PATH_VALUE_SEMANTIC_CAPITAL_SPEED_V3,
        ):
            output = study_root / "path_value_rescore" / semantic / str(year)
            result = structured.stream_checkpoint_diagnostics(
                run_dir=run, view_path=_view_for(study_root, spec, year),
                output_dir=output, year=year, compare_legacy_domain=False,
                fixed_exit_comparison=True,
                input_channel_profile=spec.input_channel_profile,
                path_value_semantic=semantic,
            )
            exit_distribution = {
                int(key): int(value)
                for key, value in dict(result["top3_legal_exit_day_distribution"]).items()
            }
            exit_total = sum(exit_distribution.values())
            rows.append({
                "model_id": spec.model_id, "year": year, "path_value_semantic": semantic,
                "rank_ic": result["legal_score_rank_ic"],
                "exit_regret": result["exit_regret"],
                "top3_average_exit_day": (
                    sum(day * count for day, count in exit_distribution.items()) / exit_total
                    if exit_total else None
                ),
                "top3_exit_max_day_share": result["top3_legal_exit_max_day_share"],
                **{f"fixed_{key.lower()}_alpha": value for key, value in dict(result["fixed_exit_top3_alpha"]).items()},
            })
    frame = pd.DataFrame(rows)
    _write_csv(study_root / "path_value_rescore.csv", frame)
    return frame


def _markdown(summary: Mapping[str, Any]) -> str:
    winner = dict(summary["selection"]["winner"])
    lines = [
        "# Structured Global / Intraday / Capital-Speed Result", "",
        f"- Status: {summary['status']}.",
        f"- Selected model: `{winner['model_id']}`.",
        f"- Selected execution: Top{int(winner['top_k'])}, {int(winner['slot_count'])} slots, `{winner['policy_name']}`.",
        f"- Double-slippage annualized log growth: {float(winner['annualized_log_growth']):.4%}.",
        f"- Total return: {float(winner['signal_period_total_return']):.4%}; drawdown: {float(winner['signal_period_maximum_drawdown']):.4%}.",
        "- D7 was treated as one ordinary fixed horizon. Model-plan/rolling exits were used only when they beat the same model's best fixed exit.",
        "- No cross-model ranking/exit hybrid, QDP update, provider call, or deployment change was performed.",
    ]
    return "\n".join(lines) + "\n"


def summarize_structured_global_intraday(*, study_root: Path = STUDY_ROOT) -> dict[str, Any]:
    root = study_root.resolve()
    state = _load_state(root)
    if str(state.get("status", "")) not in {"finalized", "oos_2026_training", "oos_2026_evaluating", "completed"}:
        raise ValueError("2023-2025 model selection is not finalized")
    decisions = _read_json(root / "stage_decisions.json")
    final = dict(decisions["final_selection"])
    specs = [V2_SPECS[key] for key in V2_SPECS] + [spec for spec in _all_declared_specs(root).values() if spec.phase == "v3"]
    capital = pd.read_csv(root / "final_capital_metrics.csv")
    annual = pd.read_csv(root / "final_capital_annual_metrics.csv")
    vintage = pd.read_csv(root / "final_vintage_metrics.csv")
    _write_csv(root / "capital_metrics.csv", capital)
    _write_csv(root / "capital_annual_metrics.csv", annual)
    _write_csv(root / "annual_metrics.csv", annual)
    _write_csv(root / "vintage_metrics.csv", vintage)
    unit, daily = _unit_time_rows(root, specs)
    _write_csv(root / "unit_time_alpha.csv", unit)
    _write_csv(root / "daily_pairwise_deltas.csv", daily)
    _feature_audit(root)
    _path_value_rescore(root)
    summary = {
        "schema_version": 1, "artifact_type": "seq100_structured_global_intraday_summary",
        "status": "selected_2023_2025" if str(state.get("status")) != "completed" else "completed",
        "created_at": _now(), "study_id": STUDY_ID,
        "selection": dict(final["selection"]),
        "v2_selection": dict(decisions["v2_selection"]),
        "v3_selection": dict(decisions["v3_selection"]),
        "oos_2026": decisions.get("oos_2026"),
        "outputs": {
            name: str((root / name).resolve()) for name in (
                "path_value_rescore.csv", "vintage_metrics.csv", "annual_metrics.csv",
                "unit_time_alpha.csv", "capital_metrics.csv", "capital_annual_metrics.csv",
                "daily_pairwise_deltas.csv", "feature_audit.json", "feature_audit.csv",
            )
        },
        "qdp_changed": False, "provider_called": False, "live_state_changed": False,
        "main_report_changed": False,
    }
    _write_json(root / "comparison_summary.json", summary)
    (root / "comparison_summary.md").write_text(_markdown(summary), encoding="utf-8")
    return summary


def verify_structured_global_intraday(
    *, study_root: Path = STUDY_ROOT, require_complete: bool = False
) -> dict[str, Any]:
    root = study_root.resolve()
    study_path = root / "study.json"
    if not study_path.is_file():
        raise FileNotFoundError(study_path)
    study = _read_json(study_path)
    if str(study.get("study_id", "")) != STUDY_ID:
        raise ValueError("study identity drifted")
    if _digest(study["contract"]) != str(study["contract_sha256"]):
        raise ValueError("study contract drifted")
    recorded = dict(dict(study["material"])["protected_snapshot"])
    current = _protected_snapshot()
    for key in ("qdp_active", "base_pack", "overlay", "source_study", "legacy_l35_checkpoints", "legacy_stage3_exists", "live_state"):
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
                raise ValueError("fairness material drifted")
    state = _load_state(root)
    if require_complete and str(state.get("status", "")) != "completed":
        raise ValueError("study is not completed")
    if str(state.get("status", "")) == "completed":
        for name in ("comparison_summary.json", "comparison_summary.md", "capital_metrics.csv", "vintage_metrics.csv"):
            if not (root / name).is_file():
                raise FileNotFoundError(root / name)
    return {
        "status": "ok", "study": str(study_path), "workflow_status": state.get("status"),
        "protected_boundaries_unchanged": True,
        "view_count": 6, "provider_called": False, "qdp_changed": False,
    }
