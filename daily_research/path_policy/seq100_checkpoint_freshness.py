from __future__ import annotations

import gc
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import torch

from daily_research.path_policy import qdp_v2_sequence_path_training as training
from daily_research.path_policy import seq100_candidate_execution as candidate_execution
from daily_research.path_policy.seq100_candidate_execution import (
    execution_cost_contract_sha256,
)
from daily_research.path_policy.seq100_development import (
    PYTHON,
    SEED,
    _assert_no_other_research_process,
    _guarded_command,
)
from daily_research.path_policy.seq100_structured_experiment import (
    LEGAL_EXIT_CONTRACT,
    PROFILE_ORDER,
    STUDY_ROOT,
    _file_sha256,
    _load_checkpoint_model,
    _now,
    _read_json,
    _run_dirs,
    _write_json,
)


TARGET_YEAR = 2025
CHECKPOINT_VINTAGES = (2023, 2024, 2025)
TOP_K_VALUES = (1, 3, 5, 10)
ANALYSIS_ID = "seq100_checkpoint_freshness_2025_v1"
DEFAULT_OUTPUT_ROOT = STUDY_ROOT / "analysis" / "checkpoint_freshness_20260719"
GATE_CONTRACT = {
    "primary": (
        "within each profile, the 2025 checkpoint's 2025 Top3 executable base alpha "
        "must be strictly higher than both the 2023 and 2024 checkpoints"
    ),
    "breadth": (
        "within each profile, at least two of Top1, Top5, and Top10 executable base "
        "alpha must be strictly higher than both older checkpoints"
    ),
    "ranking_support": (
        "within each profile, Rank IC or Top3 opportunity alpha must be strictly "
        "higher than both older checkpoints"
    ),
    "path_support": (
        "within each profile, full-universe exit regret or OHLC path MAE must be "
        "strictly lower than both older checkpoints"
    ),
    "overall": (
        "both profiles must pass primary, breadth, ranking support, and path support; "
        "the existing 2025 baseline consistency and all fairness/coverage checks must pass"
    ),
}


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


def _normalization_from_checkpoint(checkpoint: Mapping[str, Any]) -> tuple[dict[str, Any], str]:
    fold_contract = dict(checkpoint.get("fold_training_contract", {}) or {})
    payload = dict(fold_contract.get("payload", {}) or {})
    normalization = dict(payload.get("normalization", {}) or {})
    if not normalization:
        raise ValueError("checkpoint fold contract has no input normalization")
    digest = _canonical_digest(normalization)
    declared = str(fold_contract.get("normalization_sha256", "") or "")
    if digest != declared:
        raise ValueError("checkpoint normalization digest does not match its fold contract")
    return normalization, digest


def _architecture_contract(
    summary: Mapping[str, Any], checkpoint: Mapping[str, Any]
) -> dict[str, Any]:
    config = dict(
        checkpoint.get("resolved_training_config", checkpoint.get("config", {})) or {}
    )
    config.pop("development_contract", None)
    return {
        "seed": int(summary.get("seed", config.get("seed", -1))),
        "lookback_days": int(summary.get("lookback_days", 0)),
        "forward_days": int(summary.get("forward_days", 0)),
        "price_anchor": str(summary.get("price_anchor", "")),
        "input_dim": int(checkpoint.get("input_dim", 0)),
        "input_channel_profile": str(summary.get("input_channel_profile", "")),
        "input_channels": [str(value) for value in summary.get("input_channels", [])],
        "input_mask_features": [
            str(value) for value in summary.get("input_mask_features", [])
        ],
        "legal_exit_contract": dict(summary.get("legal_exit_contract", {}) or {}),
        "resolved_training_config": config,
    }


def _checkpoint_metadata(run_dir: Path, *, profile: str, vintage: int) -> dict[str, Any]:
    summary_path = run_dir / "sequence_path_training_summary.json"
    summary = _read_json(summary_path)
    checkpoint_path = Path(str(summary.get("best_checkpoint", "") or "")).resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"missing checkpoint for {profile}/{vintage}: {checkpoint_path}")
    checkpoint_sha256 = _file_sha256(checkpoint_path)
    if checkpoint_sha256 != str(summary.get("best_checkpoint_sha256", "") or ""):
        raise ValueError(f"checkpoint hash drift for {profile}/{vintage}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    try:
        normalization, normalization_sha256 = _normalization_from_checkpoint(checkpoint)
        checkpoint_fold = dict(checkpoint.get("fold_training_contract", {}) or {})
        summary_fold = dict(summary.get("development_fold_training_contract", {}) or {})
        if checkpoint_fold != summary_fold:
            raise ValueError(f"checkpoint/summary fold contract drift for {profile}/{vintage}")
        architecture = _architecture_contract(summary, checkpoint)
        if int(architecture["seed"]) != SEED:
            raise ValueError(f"checkpoint seed drift for {profile}/{vintage}")
        if int(architecture["lookback_days"]) != 100 or int(
            architecture["forward_days"]
        ) != 60:
            raise ValueError(f"checkpoint sequence shape drift for {profile}/{vintage}")
        if architecture["legal_exit_contract"] != LEGAL_EXIT_CONTRACT:
            raise ValueError(f"checkpoint legal-exit contract drift for {profile}/{vintage}")
        return {
            "profile": str(profile),
            "vintage": int(vintage),
            "run_dir": str(run_dir.resolve()),
            "summary_path": str(summary_path.resolve()),
            "summary_sha256": _file_sha256(summary_path),
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_sha256": checkpoint_sha256,
            "checkpoint_fold_contract_sha256": str(checkpoint_fold.get("sha256", "")),
            "normalization": normalization,
            "normalization_sha256": normalization_sha256,
            "normalization_fit_date_end_exclusive": str(
                normalization.get("fit_date_end_exclusive", "")
            ),
            "architecture": architecture,
            "architecture_sha256": _canonical_digest(architecture),
        }
    finally:
        del checkpoint
        gc.collect()


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
        raise ValueError(f"2025 candidate index is missing columns: {missing}")
    if candidates.empty or bool(candidates["candidate_id"].duplicated().any()):
        raise ValueError("2025 candidate index is empty or has duplicate candidate IDs")
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
    return {
        "path": str(candidate_path),
        "file_sha256": _file_sha256(candidate_path),
        "row_count": int(len(candidates)),
        "date_count": int(candidates["trade_date"].nunique()),
        "trade_date_start": str(candidates["trade_date"].min()),
        "trade_date_end": str(candidates["trade_date"].max()),
        "candidate_key_sha256": hashlib.sha256(key_hashes.tobytes()).hexdigest(),
        "candidate_key_hash_policy": "pandas_hash_object_uint64_v1",
        "label_coverage_sha256": _canonical_digest(coverage.to_dict("records")),
        "label_coverage_by_date": coverage.to_dict("records"),
    }


def _fairness_material(
    *, source_view_path: Path, source_view: Mapping[str, Any], candidate: Mapping[str, Any]
) -> dict[str, Any]:
    fold_contract = dict(source_view.get("development_fold_training_contract", {}) or {})
    if str(fold_contract.get("candidate_index_sha256", "")) != str(
        candidate["file_sha256"]
    ):
        raise ValueError("2025 candidate index no longer matches the fold contract")
    pack_manifest = Path(
        str(source_view.get("research_dataset_view", source_view_path) or source_view_path)
    )
    pack_manifest_sha256 = _file_sha256(pack_manifest) if pack_manifest.is_file() else ""
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
    return {
        "target_year": TARGET_YEAR,
        "source_view_path": str(source_view_path.resolve()),
        "source_view_sha256": _file_sha256(source_view_path),
        "pack_manifest_sha256": pack_manifest_sha256,
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
        "execution_cost_contract_sha256": execution_cost_contract_sha256(source_view),
        "date_values_sha256": _canonical_digest(source_view.get("date_values", [])),
        "symbol_values_sha256": _canonical_digest(source_view.get("symbol_values", [])),
        "evaluation_rules": {
            "candidate_membership": "exact_2025_candidate_index_without_label_filtering",
            "ranking": "predicted_price_path_only",
            "legal_exit": dict(LEGAL_EXIT_CONTRACT),
            "top_k": list(TOP_K_VALUES),
            "costs_and_execution": "exact_2025_view_contract",
        },
        "evaluation_implementation": {
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
        },
    }


def _evaluation_view(
    *,
    source_view: Mapping[str, Any],
    profile: str,
    vintage: int,
    checkpoint: Mapping[str, Any],
    fairness: Mapping[str, Any],
) -> dict[str, Any]:
    view = json.loads(json.dumps(dict(source_view), ensure_ascii=False))
    source_fold = dict(view.pop("development_fold_training_contract", {}) or {})
    view["artifact_type"] = "seq100_checkpoint_freshness_evaluation_view"
    view["artifact_view"] = f"{ANALYSIS_ID}_{profile}_{vintage}_on_{TARGET_YEAR}"
    view["normalization"] = dict(checkpoint["normalization"])
    view["checkpoint_freshness_evaluation"] = {
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
        "source_2025_fold_contract_sha256": str(source_fold.get("sha256", "")),
        "candidate_index_sha256": str(fairness["candidate_index_sha256"]),
        "label_material_sha256": str(fairness["label_material_sha256"]),
        "execution_material_sha256": str(fairness["execution_material_sha256"]),
        "future_label_required_for_candidate_membership": False,
    }
    return view


def _write_or_validate_json(path: Path, payload: Mapping[str, Any]) -> None:
    if path.is_file():
        if _read_json(path) != dict(payload):
            raise ValueError(f"existing checkpoint-freshness artifact changed: {path}")
        return
    _write_json(path, payload)


def _task_id(profile: str, vintage: int) -> str:
    return f"{profile}_checkpoint_{int(vintage)}_on_{TARGET_YEAR}"


def prepare_checkpoint_freshness(
    *, study_root: Path = STUDY_ROOT, output_root: Path = DEFAULT_OUTPUT_ROOT
) -> dict[str, Any]:
    study_path = study_root.resolve() / "study.json"
    study = _read_json(study_path)
    source_view_path = Path(
        str(dict(dict(study["material"])["fold_views"])[str(TARGET_YEAR)])
    ).resolve()
    source_view = _read_json(source_view_path)
    candidate = _candidate_material(source_view)
    fairness = _fairness_material(
        source_view_path=source_view_path,
        source_view=source_view,
        candidate=candidate,
    )
    root = output_root.resolve()
    checkpoint_records: dict[tuple[str, int], dict[str, Any]] = {}
    profile_architecture: dict[str, str] = {}
    for profile in PROFILE_ORDER:
        for vintage in CHECKPOINT_VINTAGES:
            complete, _partial = _run_dirs(study_path.parent, profile, vintage)
            if len(complete) != 1:
                raise ValueError(
                    f"checkpoint freshness requires one completed run for {profile}/{vintage}"
                )
            record = _checkpoint_metadata(complete[0], profile=profile, vintage=vintage)
            checkpoint_records[(profile, vintage)] = record
            previous = profile_architecture.setdefault(
                profile, str(record["architecture_sha256"])
            )
            if previous != str(record["architecture_sha256"]):
                raise ValueError(f"architecture or training config drift within profile {profile}")

    tasks: dict[str, Any] = {}
    for profile in PROFILE_ORDER:
        for vintage in CHECKPOINT_VINTAGES:
            record = checkpoint_records[(profile, vintage)]
            task_id = _task_id(profile, vintage)
            view_path = root / "views" / f"{task_id}.json"
            view = _evaluation_view(
                source_view=source_view,
                profile=profile,
                vintage=vintage,
                checkpoint=record,
                fairness=fairness,
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
                    else "reuse_existing_final_evaluation"
                ),
            }

    contract_payload = {
        "schema_version": 1,
        "artifact_type": "seq100_checkpoint_freshness_contract",
        "analysis_id": ANALYSIS_ID,
        "study_root": str(study_path.parent.resolve()),
        "study_contract_sha256": str(study.get("contract_sha256", "")),
        "target_year": TARGET_YEAR,
        "checkpoint_vintages": list(CHECKPOINT_VINTAGES),
        "profiles": list(PROFILE_ORDER),
        "seed": SEED,
        "top_k": list(TOP_K_VALUES),
        "fairness_material": fairness,
        "gate_contract": dict(GATE_CONTRACT),
        "profile_architecture_sha256": profile_architecture,
        "tasks": tasks,
        "protected_boundaries": {
            "qdp_updated": False,
            "provider_called": False,
            "live_execution_changed": False,
            "full_pack_copied": False,
        },
    }
    contract = {
        **contract_payload,
        "contract_sha256": _canonical_digest(contract_payload),
    }
    contract_path = root / "contract.json"
    _write_or_validate_json(contract_path, contract)
    return contract


def _load_suite_contract(path: Path) -> dict[str, Any]:
    contract = _read_json(path.resolve())
    declared = str(contract.get("contract_sha256", "") or "")
    payload = {key: value for key, value in contract.items() if key != "contract_sha256"}
    if declared != _canonical_digest(payload):
        raise ValueError("checkpoint-freshness suite contract digest changed")
    if str(contract.get("analysis_id", "")) != ANALYSIS_ID:
        raise ValueError("wrong checkpoint-freshness analysis contract")
    implementation = dict(
        dict(contract.get("fairness_material", {}) or {}).get(
            "evaluation_implementation", {}
        )
        or {}
    )
    for prefix in ("aggregation", "prediction", "execution"):
        source = Path(str(implementation.get(f"{prefix}_module", "") or "")).resolve()
        declared_source_hash = str(
            implementation.get(f"{prefix}_module_sha256", "") or ""
        )
        if not source.is_file() or _file_sha256(source) != declared_source_hash:
            raise ValueError(f"checkpoint-freshness {prefix} implementation changed")
    return contract


def _task_artifact_paths(task: Mapping[str, Any]) -> dict[str, Path]:
    output_dir = Path(str(task["output_dir"])).resolve()
    return {
        "topk_metrics_csv": output_dir / "topk_metrics.csv",
        "daily_rank_ic_csv": output_dir / "daily_rank_ic.csv",
        "daily_topk_metrics_csv": output_dir / "daily_topk_metrics.csv",
        "topk_candidates_parquet": output_dir / "topk_candidates.parquet",
    }


def _validate_completed_result(
    result_path: Path, *, suite_contract_sha256: str, task: Mapping[str, Any]
) -> dict[str, Any]:
    result = _read_json(result_path)
    if str(result.get("status", "")) != "completed":
        raise ValueError(f"checkpoint-freshness result is not completed: {result_path}")
    if str(result.get("suite_contract_sha256", "")) != suite_contract_sha256:
        raise ValueError(f"checkpoint-freshness result contract drift: {result_path}")
    if str(result.get("task_id", "")) != str(task["task_id"]):
        raise ValueError(f"checkpoint-freshness task identity drift: {result_path}")
    if str(result.get("checkpoint_sha256", "")) != str(task["checkpoint_sha256"]):
        raise ValueError(f"checkpoint-freshness checkpoint drift: {result_path}")
    outputs = dict(result.get("outputs", {}) or {})
    for name in (
        "topk_metrics_csv",
        "daily_rank_ic_csv",
        "daily_topk_metrics_csv",
        "topk_candidates_parquet",
    ):
        path = Path(str(outputs.get(name, "") or "")).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"completed result is missing {name}: {path}")
        if _file_sha256(path) != str(outputs.get(f"{name}_sha256", "")):
            raise ValueError(f"completed result artifact drifted: {path}")
    return result


def _finite_metric(frame: pd.DataFrame, top_k: int, column: str) -> float:
    row = frame[frame["top_k"].astype(int).eq(int(top_k))]
    if len(row) != 1:
        raise ValueError(f"expected one Top{top_k} aggregate row")
    value = float(pd.to_numeric(row.iloc[0][column], errors="coerce"))
    if not math.isfinite(value):
        raise ValueError(f"Top{top_k} metric {column} is not finite")
    return value


def _common_daily_coverage(daily_topk: pd.DataFrame) -> pd.DataFrame:
    required = [
        "trade_date",
        "top_k",
        "universe_count",
        "universe_hash",
        "universe_realized_plan_return_coverage",
        "execution_cost_contract_sha256",
    ]
    missing = sorted(set(required).difference(daily_topk.columns))
    if missing:
        raise ValueError(f"daily TopK output is missing common coverage columns: {missing}")
    return daily_topk[required].sort_values(
        ["trade_date", "top_k"], kind="mergesort"
    )


def _evaluation_metrics(
    *, topk: pd.DataFrame, daily_topk: pd.DataFrame, split_metrics: Mapping[str, Any]
) -> dict[str, Any]:
    if set(topk["top_k"].astype(int)) != set(TOP_K_VALUES):
        raise ValueError("TopK aggregate is incomplete")
    coverage_values = pd.to_numeric(
        daily_topk["selected_realized_plan_return_coverage"], errors="coerce"
    ).to_numpy(dtype=np.float64)
    if not bool(np.isfinite(coverage_values).all()) or not bool(
        np.equal(coverage_values, 1.0).all()
    ):
        raise ValueError("execution coverage is incomplete")
    metrics: dict[str, Any] = {
        "candidate_count": int(split_metrics["row_count"]),
        "date_count": int(split_metrics["date_count"]),
        "candidate_score_coverage": float(
            split_metrics["candidate_complete_score_coverage"]
        ),
        "execution_return_coverage": float(coverage_values.mean()),
        "rank_ic": float(split_metrics["rank_ic_mean"]),
        "rank_ic_positive_day_rate": float(
            split_metrics["rank_ic_positive_day_rate"]
        ),
        "path_mae": float(split_metrics["path_mae"]),
        "path_open_mae": float(split_metrics["path_open_mae"]),
        "path_high_mae": float(split_metrics["path_high_mae"]),
        "path_low_mae": float(split_metrics["path_low_mae"]),
        "path_close_mae": float(split_metrics["path_close_mae"]),
        "entry_fill_rate": float(split_metrics["entry_fill_rate"]),
        "label_coverage": _finite_metric(topk, 3, "universe_label_coverage"),
        "exit_regret": _finite_metric(topk, 3, "universe_oracle_regret"),
        "exit_regret_coverage": _finite_metric(
            topk, 3, "universe_oracle_regret_coverage"
        ),
    }
    if metrics["candidate_score_coverage"] != 1.0:
        raise ValueError("score coverage is incomplete")
    for top_k in TOP_K_VALUES:
        metrics.update(
            {
                f"top{top_k}_base_alpha": _finite_metric(
                    topk, top_k, "alpha_net_realized_plan_return_base"
                ),
                f"top{top_k}_stress_alpha": _finite_metric(
                    topk, top_k, "alpha_net_realized_plan_return_stress"
                ),
                f"top{top_k}_opportunity_alpha": _finite_metric(
                    topk, top_k, "alpha_opportunity_value"
                ),
                f"top{top_k}_selected_base": _finite_metric(
                    topk, top_k, "selected_net_realized_plan_return_base"
                ),
                f"top{top_k}_universe_base": _finite_metric(
                    topk, top_k, "universe_net_realized_plan_return_base"
                ),
            }
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
) -> dict[str, Any]:
    fairness = dict(contract["fairness_material"])
    output_meta: dict[str, Any] = {}
    for name, path in artifacts.items():
        output_meta[name] = str(path.resolve())
        output_meta[f"{name}_sha256"] = _file_sha256(path)
    return {
        "schema_version": 1,
        "artifact_type": "seq100_checkpoint_freshness_evaluation",
        "status": "completed",
        "completed_at": completed_at,
        "analysis_id": ANALYSIS_ID,
        "suite_contract_sha256": str(contract["contract_sha256"]),
        "task_id": str(task["task_id"]),
        "profile": str(task["profile"]),
        "checkpoint_vintage": int(task["checkpoint_vintage"]),
        "target_year": TARGET_YEAR,
        "result_source": str(task["result_source"]),
        "inference_performed": bool(task["inference_required"]),
        "checkpoint_path": str(task["checkpoint_path"]),
        "checkpoint_sha256": str(task["checkpoint_sha256"]),
        "normalization_sha256": str(task["normalization_sha256"]),
        "normalization_fit_date_end_exclusive": str(
            task["normalization_fit_date_end_exclusive"]
        ),
        "candidate_index_sha256": str(fairness["candidate_index_sha256"]),
        "candidate_key_sha256": str(fairness["candidate_key_sha256"]),
        "label_coverage_sha256": str(fairness["label_coverage_sha256"]),
        "label_material_sha256": str(fairness["label_material_sha256"]),
        "execution_material_sha256": str(fairness["execution_material_sha256"]),
        "evaluation_implementation": dict(fairness["evaluation_implementation"]),
        "common_daily_coverage_sha256": _canonical_digest(
            _common_daily_coverage(daily_topk).to_dict("records")
        ),
        "metrics": dict(metrics),
        "outputs": output_meta,
        "full_prediction_written": False,
        "qdp_changed": False,
        "provider_called": False,
        "live_execution_changed": False,
    }


def reuse_existing_2025_result(
    *, suite_contract_path: Path, profile: str
) -> dict[str, Any]:
    contract = _load_suite_contract(suite_contract_path)
    task_id = _task_id(profile, TARGET_YEAR)
    task = dict(dict(contract["tasks"])[task_id])
    if bool(task["inference_required"]):
        raise ValueError(f"task {task_id} is not registered for existing-result reuse")
    result_path = Path(str(task["result_path"])).resolve()
    if result_path.is_file():
        return _validate_completed_result(
            result_path,
            suite_contract_sha256=str(contract["contract_sha256"]),
            task=task,
        )
    run_dir = Path(str(task["run_dir"])).resolve()
    summary_path = run_dir / "sequence_path_training_summary.json"
    if _file_sha256(summary_path) != str(task["checkpoint_summary_sha256"]):
        raise ValueError(f"existing 2025 summary drift for {task_id}")
    summary = _read_json(summary_path)
    if str(summary.get("best_checkpoint_sha256", "")) != str(
        task["checkpoint_sha256"]
    ):
        raise ValueError(f"existing 2025 checkpoint binding drift for {task_id}")
    if str(summary.get("candidate_index_sha256", "")) != str(
        contract["fairness_material"]["candidate_index_sha256"]
    ):
        raise ValueError(f"existing 2025 candidate binding drift for {task_id}")
    if str(summary.get("execution_cost_contract_sha256", "")) != str(
        contract["fairness_material"]["execution_cost_contract_sha256"]
    ):
        raise ValueError(f"existing 2025 execution-cost binding drift for {task_id}")
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
    for name, path in artifacts.items():
        declared = str(dict(summary.get("outputs", {}) or {}).get(name, "") or "")
        if not path.is_file() or Path(declared).resolve() != path.resolve():
            raise ValueError(f"existing 2025 output binding drift for {task_id}/{name}")
    topk = pd.read_csv(artifacts["topk_metrics_csv"])
    daily_topk = pd.read_csv(artifacts["daily_topk_metrics_csv"])
    topk = topk[topk["split"].astype(str).eq("development")].copy()
    daily_topk = daily_topk[
        daily_topk["split"].astype(str).eq("development")
    ].copy()
    metrics = _evaluation_metrics(
        topk=topk, daily_topk=daily_topk, split_metrics=split_metrics
    )
    fairness = dict(contract["fairness_material"])
    if int(metrics["candidate_count"]) != int(fairness["candidate_count"]):
        raise ValueError(f"existing 2025 candidate count drift for {task_id}")
    if int(metrics["date_count"]) != int(fairness["candidate_date_count"]):
        raise ValueError(f"existing 2025 candidate date count drift for {task_id}")
    result = _evaluation_result(
        contract=contract,
        task=task,
        metrics=metrics,
        daily_topk=daily_topk,
        artifacts=artifacts,
        completed_at=_now(),
    )
    _write_json(result_path, result)
    return result


def evaluate_vintage_task(
    *, suite_contract_path: Path, profile: str, vintage: int
) -> dict[str, Any]:
    contract = _load_suite_contract(suite_contract_path)
    task_id = _task_id(profile, vintage)
    task = dict(dict(contract["tasks"])[task_id])
    if not bool(task["inference_required"]):
        return reuse_existing_2025_result(
            suite_contract_path=suite_contract_path, profile=profile
        )
    result_path = Path(str(task["result_path"])).resolve()
    if result_path.is_file():
        return _validate_completed_result(
            result_path,
            suite_contract_sha256=str(contract["contract_sha256"]),
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
    evaluation = dict(view.get("checkpoint_freshness_evaluation", {}) or {})
    if str(evaluation.get("normalization_sha256", "")) != str(
        task["normalization_sha256"]
    ) or _canonical_digest(view.get("normalization", {})) != str(
        task["normalization_sha256"]
    ):
        raise ValueError(f"evaluation normalization drift for {task_id}")
    dataset = training.SequencePathPackDataset(
        view,
        split="development",
        max_samples=0,
        input_channel_profile=training.INPUT_CHANNEL_PROFILE_DAILY_ONLY,
        index_role="candidate",
    )
    fairness = dict(contract["fairness_material"])
    if int(len(dataset)) != int(fairness["candidate_count"]):
        raise ValueError(f"candidate count drift for {task_id}")
    if str(dataset.sample_selection["index_sha256"]) != str(
        fairness["candidate_index_sha256"]
    ):
        raise ValueError(f"candidate index drift for {task_id}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, summary = _load_checkpoint_model(
        Path(str(task["run_dir"])), dataset, device=device
    )
    try:
        architecture = _architecture_contract(summary, {
            "input_dim": dataset.input_dim,
            "resolved_training_config": summary.get("resolved_training_config", {}),
        })
        if _canonical_digest(architecture) != str(task["architecture_sha256"]):
            raise ValueError(f"loaded model architecture drift for {task_id}")
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

    if int(split_metrics["row_count"]) != int(fairness["candidate_count"]):
        raise ValueError(f"evaluated row count drift for {task_id}")
    if int(split_metrics["date_count"]) != int(fairness["candidate_date_count"]):
        raise ValueError(f"evaluated date count drift for {task_id}")
    metrics = _evaluation_metrics(
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


def _strictly_better_than_both(
    rows: Mapping[int, Mapping[str, Any]], metric: str, *, higher: bool
) -> bool:
    current = float(dict(rows[2025]["metrics"])[metric])
    older = [float(dict(rows[year]["metrics"])[metric]) for year in (2023, 2024)]
    return bool(current > max(older)) if higher else bool(current < min(older))


def freshness_gate(results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_profile: dict[str, dict[int, Mapping[str, Any]]] = {}
    for result in results:
        by_profile.setdefault(str(result["profile"]), {})[
            int(result["checkpoint_vintage"])
        ] = result
    decisions: dict[str, Any] = {}
    for profile in PROFILE_ORDER:
        rows = by_profile.get(profile, {})
        if set(rows) != set(CHECKPOINT_VINTAGES):
            raise ValueError(f"freshness gate is missing vintages for {profile}")
        primary = _strictly_better_than_both(
            rows, "top3_base_alpha", higher=True
        )
        breadth_metrics = ("top1_base_alpha", "top5_base_alpha", "top10_base_alpha")
        breadth_evidence = {
            metric: _strictly_better_than_both(rows, metric, higher=True)
            for metric in breadth_metrics
        }
        ranking_evidence = {
            metric: _strictly_better_than_both(rows, metric, higher=True)
            for metric in ("rank_ic", "top3_opportunity_alpha")
        }
        path_evidence = {
            metric: _strictly_better_than_both(rows, metric, higher=False)
            for metric in ("exit_regret", "path_mae")
        }
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
        "contract": dict(GATE_CONTRACT),
        "profiles": decisions,
        "metric_gate_pass": bool(
            all(bool(decisions[profile]["profile_pass"]) for profile in PROFILE_ORDER)
        ),
    }


def _source_2025_expected_metrics(task: Mapping[str, Any]) -> dict[str, float]:
    run_dir = Path(str(task["run_dir"])).resolve()
    summary = _read_json(run_dir / "sequence_path_training_summary.json")
    split = next(
        item
        for item in summary.get("split_metrics", [])
        if str(item.get("split", "")) == "development"
    )
    topk = pd.read_csv(run_dir / "topk_metrics.csv")
    topk = topk[topk["split"].astype(str).eq("development")].copy()
    expected = {
        "rank_ic": float(split["rank_ic_mean"]),
        "path_mae": float(split["path_mae"]),
        "path_close_mae": float(split["path_close_mae"]),
        "exit_regret": _finite_metric(topk, 3, "universe_oracle_regret"),
    }
    for top_k in TOP_K_VALUES:
        expected[f"top{top_k}_base_alpha"] = _finite_metric(
            topk, top_k, "alpha_net_realized_plan_return_base"
        )
        expected[f"top{top_k}_opportunity_alpha"] = _finite_metric(
            topk, top_k, "alpha_opportunity_value"
        )
    return expected


def _existing_2025_consistency_check(
    contract: Mapping[str, Any], results: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    checks: dict[str, Any] = {}
    for result in results:
        if int(result["checkpoint_vintage"]) != TARGET_YEAR:
            continue
        task = dict(dict(contract["tasks"])[str(result["task_id"])])
        expected = _source_2025_expected_metrics(task)
        observed = dict(result["metrics"])
        differences = {
            key: abs(float(observed[key]) - float(value))
            for key, value in expected.items()
        }
        checks[str(result["profile"])] = {
            "maximum_absolute_difference": float(max(differences.values(), default=0.0)),
            "differences": differences,
            "pass": bool(max(differences.values(), default=0.0) <= 1.0e-9),
        }
    return {
        "tolerance": 1.0e-9,
        "profiles": checks,
        "pass": bool(
            set(checks) == set(PROFILE_ORDER)
            and all(bool(value["pass"]) for value in checks.values())
        ),
    }


def summarize_checkpoint_freshness(
    *, suite_contract_path: Path
) -> dict[str, Any]:
    contract = _load_suite_contract(suite_contract_path)
    results: list[dict[str, Any]] = []
    for task in dict(contract["tasks"]).values():
        result_path = Path(str(task["result_path"])).resolve()
        if not result_path.is_file():
            raise ValueError("cannot summarize incomplete checkpoint-freshness tasks")
        results.append(
            _validate_completed_result(
                result_path,
                suite_contract_sha256=str(contract["contract_sha256"]),
                task=task,
            )
        )
    fairness_fields = (
        "candidate_index_sha256",
        "candidate_key_sha256",
        "label_coverage_sha256",
        "label_material_sha256",
        "execution_material_sha256",
        "common_daily_coverage_sha256",
    )
    common_values = {
        field: sorted({str(result[field]) for result in results})
        for field in fairness_fields
    }
    coverage_checks = {
        "candidate_keys_identical": len(common_values["candidate_key_sha256"]) == 1,
        "label_coverage_identical": len(common_values["label_coverage_sha256"]) == 1,
        "label_material_identical": len(common_values["label_material_sha256"]) == 1,
        "execution_material_identical": len(common_values["execution_material_sha256"])
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
    existing_2025_consistency = _existing_2025_consistency_check(contract, results)
    gate = freshness_gate(results)
    overall_pass = bool(
        fairness_pass
        and existing_2025_consistency["pass"]
        and gate["metric_gate_pass"]
    )
    rows = []
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
                **dict(result["metrics"]),
            }
        )
    root = suite_contract_path.resolve().parent
    metrics_path = root / "vintage_metrics.csv"
    _write_csv(metrics_path, pd.DataFrame(rows))
    summary = {
        "schema_version": 1,
        "artifact_type": "seq100_checkpoint_freshness_summary",
        "status": "completed",
        "completed_at": _now(),
        "analysis_id": ANALYSIS_ID,
        "suite_contract": str(suite_contract_path.resolve()),
        "suite_contract_sha256": str(contract["contract_sha256"]),
        "target_year": TARGET_YEAR,
        "profiles": list(PROFILE_ORDER),
        "checkpoint_vintages": list(CHECKPOINT_VINTAGES),
        "fairness": {
            "checks": coverage_checks,
            "common_values": common_values,
            "pass": fairness_pass,
        },
        "existing_2025_consistency": existing_2025_consistency,
        "gate": gate,
        "overall_freshness_gate_pass": overall_pass,
        "next_step": "train_2026_fold" if overall_pass else "stop_without_2026_training",
        "metrics": rows,
        "outputs": {
            "vintage_metrics_csv": str(metrics_path.resolve()),
            "vintage_metrics_csv_sha256": _file_sha256(metrics_path),
        },
        "qdp_changed": False,
        "provider_called": False,
        "live_execution_changed": False,
    }
    summary_path = root / "freshness_summary.json"
    _write_json(summary_path, summary)
    lines = [
        "# Seq100 checkpoint freshness on the frozen 2025 candidate set",
        "",
        "All six evaluations use the exact same 2025 candidates, labels, costs, and execution rules. Each checkpoint uses its own training-time input normalization.",
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
            f"Existing 2025 baseline consistency: `{existing_2025_consistency['pass']}`.",
            f"Pre-registered freshness gate: `{overall_pass}`.",
            f"Next step: `{summary['next_step']}`.",
        ]
    )
    (root / "freshness_summary.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    return summary


def evaluate_vintages(
    *,
    study_root: Path = STUDY_ROOT,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    max_tasks: int = 0,
) -> dict[str, Any]:
    _assert_no_other_research_process()
    contract = prepare_checkpoint_freshness(
        study_root=study_root, output_root=output_root
    )
    root = output_root.resolve()
    contract_path = root / "contract.json"
    state_path = root / "state.json"
    completed: list[str] = []
    for profile in PROFILE_ORDER:
        reused = reuse_existing_2025_result(
            suite_contract_path=contract_path, profile=profile
        )
        completed.append(str(reused["task_id"]))
    launched = 0
    _write_json(
        state_path,
        {
            "status": "running",
            "analysis_id": ANALYSIS_ID,
            "suite_contract_sha256": str(contract["contract_sha256"]),
            "completed_tasks": completed,
            "updated_at": _now(),
        },
    )
    try:
        for task_id, raw_task in dict(contract["tasks"]).items():
            task = dict(raw_task)
            result_path = Path(str(task["result_path"])).resolve()
            if result_path.is_file():
                _validate_completed_result(
                    result_path,
                    suite_contract_sha256=str(contract["contract_sha256"]),
                    task=task,
                )
                if task_id not in completed:
                    completed.append(task_id)
                continue
            if int(max_tasks) > 0 and launched >= int(max_tasks):
                break
            _write_json(
                state_path,
                {
                    "status": "running",
                    "analysis_id": ANALYSIS_ID,
                    "suite_contract_sha256": str(contract["contract_sha256"]),
                    "completed_tasks": completed,
                    "current_task": task_id,
                    "updated_at": _now(),
                },
            )
            command = [
                str(PYTHON),
                "-m",
                "daily_research.path_policy.seq100_development",
                "evaluate-vintage-task",
                "--experiment",
                "structured-path-v1",
                "--suite-contract",
                str(contract_path),
                "--profile",
                str(task["profile"]),
                "--vintage",
                str(task["checkpoint_vintage"]),
            ]
            _guarded_command(
                command,
                log_path=root / "logs" / f"memory_guard_{task_id}.json",
            )
            _validate_completed_result(
                result_path,
                suite_contract_sha256=str(contract["contract_sha256"]),
                task=task,
            )
            completed.append(task_id)
            launched += 1
        if len(completed) == len(dict(contract["tasks"])):
            summary = summarize_checkpoint_freshness(
                suite_contract_path=contract_path
            )
            _write_json(
                state_path,
                {
                    "status": "completed",
                    "analysis_id": ANALYSIS_ID,
                    "suite_contract_sha256": str(contract["contract_sha256"]),
                    "completed_tasks": completed,
                    "summary": str((root / "freshness_summary.json").resolve()),
                    "updated_at": _now(),
                },
            )
            return summary
        result = {
            "status": "partial",
            "analysis_id": ANALYSIS_ID,
            "suite_contract": str(contract_path),
            "completed_tasks": completed,
            "pending_tasks": [
                task_id
                for task_id in dict(contract["tasks"])
                if task_id not in set(completed)
            ],
        }
        _write_json(state_path, {**result, "updated_at": _now()})
        return result
    except Exception as exc:
        _write_json(
            state_path,
            {
                "status": "failed",
                "analysis_id": ANALYSIS_ID,
                "suite_contract_sha256": str(contract["contract_sha256"]),
                "completed_tasks": completed,
                "error": str(exc),
                "updated_at": _now(),
            },
        )
        raise
