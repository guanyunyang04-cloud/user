"""Evaluate Structured 180x35 checkpoint vintages on common 2026 material.

The 2026 training study intentionally selected one final model and two fixed-D7
accounts.  This diagnostic layer answers a different question: how the 2023,
2024, 2025, and 2026 checkpoints behave on the same native-180 2026 candidate
set.  It never mutates checkpoints, the QDP snapshot, the base pack, or live
state.
"""

from __future__ import annotations

import gc
import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import torch

from daily_research.path_policy import qdp_v2_sequence_path_training as training
from daily_research.path_policy import seq100_2026_fold_comparison as comparison_2026
from daily_research.path_policy import seq100_finite_capital_backtest as finite
from daily_research.path_policy import seq100_integrity_v2 as integrity
from daily_research.path_policy import seq100_exit_policy_audit as exit_audit
from daily_research.path_policy import seq100_structured_180x35_2026 as study
from daily_research.path_policy import seq100_structured_180x35_2026_diagnostics as prior
from daily_research.path_policy import seq100_structured_experiment as structured
from daily_research.path_policy.seq100_exit_policy_audit import CandidateCompleteAuditPack
from daily_research.path_policy.seq100_candidate_execution import (
    DEFAULT_DAILY_COHORT_CASH_CNY,
)


DIAGNOSTIC_ID = "structured-180x35-2026-checkpoint-vintages-2023-2026-v1"
OUTPUT_ROOT = study.STUDY_ROOT / "analysis/checkpoint_vintages_2023_2026_20260721"
ABLATION_STUDY_ROOT = (
    study.INPUT_STUDY_ROOT
)
HISTORICAL_CAPITAL_ROOT = ABLATION_STUDY_ROOT / "stage2_capital_speed_review"
VINTAGES = (2023, 2024, 2025, 2026)
HISTORICAL_YEARS = (2023, 2024, 2025)
NEW_INFERENCE_VINTAGES = (2023, 2024)
EXPECTED_EXTENDED_CANDIDATES = 361_980
POLICY_ALPHA_NAMES = tuple(
    [f"fixed_d{day}" for day in range(2, 61)]
    + ["model_plan", "rolling_path"]
)
EXPECTED_POLICY_ALPHA_DAILY_ROWS = (
    len(VINTAGES)
    * 48
    * len(POLICY_ALPHA_NAMES)
    * 2
    * len(study.COST_SCENARIOS)
)
HISTORICAL_POLICY_ALPHA_NAMES = POLICY_ALPHA_NAMES
HISTORICAL_POLICY_ALPHA_DATE_COUNTS = {2023: 242, 2024: 242, 2025: 243}
EXPECTED_HISTORICAL_POLICY_ALPHA_DAILY_ROWS = (
    sum(HISTORICAL_POLICY_ALPHA_DATE_COUNTS.values())
    * len(HISTORICAL_POLICY_ALPHA_NAMES)
    * 2
    * len(study.COST_SCENARIOS)
)
MODEL_IDS = {
    year: f"structured_180x35_{year}_on_2026" for year in VINTAGES
}
MODEL_IDS[2026] = study.MODEL_2026_180
MODEL_PLAN = finite.PolicySpec(name="model_plan", kind="model_plan")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    study._write_json(path, payload)


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    study._write_csv(path, frame)


def _write_parquet(path: Path, frame: pd.DataFrame) -> None:
    study._write_parquet(path, frame)


def _run_dir(vintage: int) -> Path:
    year = int(vintage)
    if year == 2025:
        return study._old_2025_run()
    if year == 2026:
        return prior._new_run()
    pattern = (
        "runs/stage_2/"
        f"seq100_structured_ablation_s2_lookback180_turnover_{year}_seed7_*"
    )
    matches = sorted(ABLATION_STUDY_ROOT.glob(pattern))
    if len(matches) != 1:
        raise ValueError(f"expected one Structured 180x35 run for {year}, found {matches}")
    run_dir = matches[0].resolve()
    required = (
        run_dir / "best_model.pt",
        run_dir / "sequence_path_training_summary.json",
    )
    if not all(path.is_file() for path in required):
        raise FileNotFoundError(f"incomplete checkpoint run for {year}: {run_dir}")
    return run_dir


def _view_path(vintage: int) -> Path:
    year = int(vintage)
    if year == 2025:
        return prior.VIEW_2025
    if year == 2026:
        return study.STUDY_ROOT / "fold_view_2026.json"
    return OUTPUT_ROOT / "views" / f"evaluation_{year}_on_2026_native_180.json"


def _checkpoint_normalization(run_dir: Path) -> tuple[dict[str, Any], str]:
    return prior._checkpoint_normalization(run_dir)


def _prepare_view(vintage: int) -> Path:
    year = int(vintage)
    if year == 2025:
        prior.prepare_diagnostics()
        return prior.VIEW_2025
    if year == 2026:
        return study.STUDY_ROOT / "fold_view_2026.json"

    run_dir = _run_dir(year)
    normalization, normalization_sha256 = _checkpoint_normalization(run_dir)
    source_view = _read_json(study.STUDY_ROOT / "fold_view_2026.json")
    view = json.loads(json.dumps(source_view, ensure_ascii=False))
    source_fold = dict(view.pop("development_fold_training_contract", {}) or {})
    view["normalization"] = normalization
    artifact = dict(view.get("artifact_view", {}) or {})
    artifact.update(
        {
            "schema_version": 1,
            "view_id": f"seq100_structured_180x35_{year}_checkpoint_on_2026_native_180",
            "view_type": "checkpoint_normalization_evaluation_view",
        }
    )
    view["artifact_view"] = artifact
    view["checkpoint_vintage_diagnostic"] = {
        "diagnostic_id": DIAGNOSTIC_ID,
        "checkpoint_vintage": year,
        "target_year": 2026,
        "normalization_source": "checkpoint.fold_training_contract.payload.normalization",
        "normalization_sha256": normalization_sha256,
        "source_2026_fold_contract_sha256": str(source_fold.get("sha256", "")),
        "candidate_mode": "native_180",
        "candidate_membership_uses_future_information": False,
    }
    target = _view_path(year)
    if target.is_file():
        if _read_json(target) != view:
            raise ValueError(f"the {year}-on-2026 evaluation view drifted")
    else:
        _write_json(target, view)

    dataset = training.SequencePathPackDataset(
        view,
        split="development",
        max_samples=0,
        input_channel_profile=study.INPUT_CHANNEL_PROFILE,
        index_role="candidate",
    )
    try:
        if (
            len(dataset) != study.EXPECTED_NATIVE_FULL_LABEL_CANDIDATES
            or int(dataset.lookback_days) != study.LOOKBACK_DAYS
            or int(dataset.input_dim) != study.INPUT_DIM
        ):
            raise ValueError(f"the {year} view does not preserve the native 180x35 contract")
    finally:
        del dataset
        gc.collect()
    return target


def _checkpoint_health(vintage: int) -> dict[str, Any]:
    year = int(vintage)
    run_dir = _run_dir(year)
    summary = _read_json(run_dir / "sequence_path_training_summary.json")
    checkpoint = torch.load(run_dir / "best_model.pt", map_location="cpu", weights_only=False)
    try:
        state = dict(checkpoint["model_state_dict"])
        finite_tensors = all(
            bool(torch.isfinite(value).all())
            for value in state.values()
            if torch.is_tensor(value)
        )
        scaler = dict(checkpoint.get("scaler_state_dict", {}) or {})
        scaler_values_finite = all(
            math.isfinite(float(value))
            for value in scaler.values()
            if isinstance(value, (int, float))
        )
    finally:
        del checkpoint
        gc.collect()
    history = list(summary.get("history", []) or [])
    numeric_loss_keys = (
        "loss",
        "path_loss",
        "summary_loss",
        "geometry_loss",
        "utility_curve_loss",
        "turnover_level_loss",
        "turnover_delta_loss",
        "value_loss",
        "rank_loss",
        "development_total_loss",
        "development_price_total_loss",
    )
    losses_finite = all(
        math.isfinite(float(row[key]))
        for row in history
        for key in numeric_loss_keys
        if row.get(key) is not None
    )
    trim_events = [
        event
        for row in history
        for event in list(row.get("working_set_trim_events", []) or [])
    ]
    sample = dict(summary.get("sample_selection", {}) or {})
    return {
        "checkpoint_vintage": year,
        "run_dir": str(run_dir),
        "checkpoint_sha256": study._file_sha256(run_dir / "best_model.pt"),
        "architecture": study._architecture_signature(run_dir),
        "completed_epochs": int(summary.get("completed_epochs", 0)),
        "best_epoch": int(summary.get("best_epoch", 0)),
        "best_optimizer_step": int(summary.get("best_optimizer_step", 0)),
        "train_row_count": int(dict(sample.get("train", {}) or {}).get("selected_row_count", 0)),
        "development_candidate_count": int(
            dict(sample.get("development_candidates", {}) or {}).get("selected_row_count", 0)
        ),
        "normalization_cutoff": str(summary.get("normalization_cutoff", "")),
        "best_development_price_loss": float(
            dict(summary.get("early_stopping", {}) or {}).get("best_value", float("nan"))
        ),
        "device": str(summary.get("device", "")),
        "amp_enabled": bool(summary.get("amp_enabled", False)),
        "model_state_finite": finite_tensors,
        "scaler_state_finite": scaler_values_finite,
        "losses_finite": losses_finite,
        "working_set_trim_count": len(trim_events),
        "all_working_set_trims_succeeded": all(
            bool(event.get("trim_succeeded", False)) for event in trim_events
        ),
    }


def _training_health() -> dict[str, Any]:
    rows = [_checkpoint_health(year) for year in VINTAGES]
    architectures = [row["architecture"] for row in rows]
    progress = _read_json(study.STUDY_ROOT / "progress.json")
    monitor = _read_json(study.STUDY_ROOT / "monitor.json")
    contract = _read_json(study.STUDY_ROOT / "study.json")["contract"]
    training_contract = dict(contract["training"])
    row_2026 = next(row for row in rows if row["checkpoint_vintage"] == 2026)
    checks = {
        "progress_completed": str(progress.get("status")) == "completed",
        "monitor_completed": str(monitor.get("status")) == "completed",
        "monitor_event_completed": str(monitor.get("event")) == "training_completed",
        "checkpoint_tensor_finite": bool(row_2026["model_state_finite"]),
        "scaler_state_finite": bool(row_2026["scaler_state_finite"]),
        "losses_finite": bool(row_2026["losses_finite"]),
        "native_train_count_matches": int(row_2026["train_row_count"])
        == int(training_contract["native_180_train_row_count"]),
        "prequalification_count_documented": int(
            training_contract["prequalification_train_row_count"]
        )
        == 7_674_720,
        "normalization_cutoff_matches": row_2026["normalization_cutoff"]
        == study.SIGNAL_START,
        "one_epoch_fixed": int(row_2026["completed_epochs"]) == 1
        and int(row_2026["best_epoch"]) == 1,
        "architecture_matches_2023_2025": all(
            architecture == architectures[0] for architecture in architectures[1:]
        ),
        "memory_trims_succeeded": bool(row_2026["all_working_set_trims_succeeded"]),
        "epoch_selection_uses_2026_labels": bool(
            training_contract.get("epoch_selection_uses_2026_labels", True)
        ),
    }
    healthy = all(
        value
        for key, value in checks.items()
        if key != "epoch_selection_uses_2026_labels"
    ) and not checks["epoch_selection_uses_2026_labels"]
    return {
        "status": "healthy" if healthy else "failed",
        "checks": checks,
        "checkpoints": rows,
        "training_row_reconciliation": {
            "prequalification_rows": int(training_contract["prequalification_train_row_count"]),
            "native_180_qualified_rows": int(training_contract["native_180_train_row_count"]),
            "excluded_by_native_180_qualification": int(
                training_contract["prequalification_train_row_count"]
                - training_contract["native_180_train_row_count"]
            ),
            "qualification_rule": str(training_contract["input_qualification"]),
        },
        "interpretation": (
            "The 2026 fold completed normally and is contract-consistent. "
            "Weak 2026 Top-K returns are an out-of-sample behavior finding, not evidence "
            "of a crashed or numerically corrupt training run."
        ),
    }


def prepare_vintage_diagnostics(*, output_root: Path = OUTPUT_ROOT) -> dict[str, Any]:
    if output_root.resolve() != OUTPUT_ROOT.resolve():
        raise ValueError("vintage diagnostics require the canonical output root")
    study.verify_structured_180x35_2026(require_complete=True)
    prior.prepare_diagnostics()
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    for year in VINTAGES:
        _prepare_view(year)
    source_view = _read_json(study.STUDY_ROOT / "fold_view_2026.json")
    candidate_path = Path(str(source_view["candidate_index_path"])).resolve()
    checkpoints: dict[str, Any] = {}
    for year in VINTAGES:
        run_dir = _run_dir(year)
        normalization, normalization_sha256 = _checkpoint_normalization(run_dir)
        checkpoints[str(year)] = {
            "model_id": MODEL_IDS[year],
            "checkpoint_path": str((run_dir / "best_model.pt").resolve()),
            "checkpoint_sha256": study._file_sha256(run_dir / "best_model.pt"),
            "normalization_sha256": normalization_sha256,
            "model_bundle_sha256": integrity.model_bundle_sha256(
                {
                    "checkpoint_sha256": study._file_sha256(run_dir / "best_model.pt"),
                    "normalization": normalization,
                    "architecture": study._architecture_signature(run_dir),
                }
            ),
        }
    contract = {
        "schema_version": 1,
        "diagnostic_id": DIAGNOSTIC_ID,
        "source_study_contract_sha256": str(
            _read_json(study.STUDY_ROOT / "study.json")["contract_sha256"]
        ),
        "candidate_semantic_sha256": integrity.candidate_semantic_sha256(candidate_path),
        "candidate_count": study.EXPECTED_NATIVE_FULL_LABEL_CANDIDATES,
        "candidate_mode": "native_180",
        "candidate_membership_uses_future_information": False,
        "full_label_window": [study.SIGNAL_START, study.FULL_LABEL_END],
        "extended_d7_window": [study.SIGNAL_START, study.EXTENDED_SIGNAL_END],
        "checkpoint_vintages": checkpoints,
        "evaluation": {
            "full_label_metrics": True,
            "model_plan_common_48": True,
            "fixed_d7_strict_101_and_extended_121": True,
            "cost_scenarios": list(study.COST_SCENARIOS),
            "top_k_slots": [[1, 1], [3, 3]],
            "cross_model_rank_exit_hybrid": False,
        },
        "protected_boundaries": {
            "qdp_changed": False,
            "provider_called": False,
            "checkpoint_changed": False,
            "live_state_changed": False,
            "deployment_changed": False,
        },
    }
    payload = {"contract": contract, "contract_sha256": study._canonical_digest(contract)}
    target = OUTPUT_ROOT / "contract.json"
    if target.is_file():
        if _read_json(target) != payload:
            raise ValueError("the checkpoint-vintage diagnostic contract drifted")
    else:
        _write_json(target, payload)
    _write_json(OUTPUT_ROOT / "training_health.json", _training_health())
    return payload


def _load_model_dataset(
    vintage: int, *, device: torch.device
) -> tuple[torch.nn.Module, training.SequencePathPackDataset, Path]:
    run_dir = _run_dir(vintage)
    view = _read_json(_view_path(vintage))
    dataset = training.SequencePathPackDataset(
        view,
        split="development",
        max_samples=0,
        input_channel_profile=study.INPUT_CHANNEL_PROFILE,
        index_role="candidate",
    )
    if len(dataset) != study.EXPECTED_NATIVE_FULL_LABEL_CANDIDATES:
        raise ValueError(f"candidate coverage drifted for checkpoint {vintage}")
    model, _summary = structured._load_checkpoint_model(run_dir, dataset, device=device)
    return model, dataset, run_dir


def _full_label_dir(vintage: int) -> Path:
    return OUTPUT_ROOT / "full_label" / MODEL_IDS[int(vintage)]


def _extended_prediction_path(vintage: int) -> Path:
    year = int(vintage)
    if year == 2025:
        return prior._extended_prediction_path(prior.MODEL_2025)
    if year == 2026:
        return prior._extended_prediction_path(prior.MODEL_2026)
    return OUTPUT_ROOT / "predictions" / f"extended_score_and_exit_{MODEL_IDS[year]}.parquet"


def _full_label_result(vintage: int) -> dict[str, Any]:
    year = int(vintage)
    if year == 2025:
        return prior._evaluate_old_full_label()
    if year == 2026:
        return _read_json(study._full_label_task_dir(prior.MODEL_2026) / "evaluation.json")
    return _read_json(_full_label_dir(year) / "evaluation.json")


def _evaluate_loaded_full_label(
    vintage: int,
    *,
    model: torch.nn.Module,
    dataset: training.SequencePathPackDataset,
    run_dir: Path,
    device: torch.device,
) -> dict[str, Any]:
    year = int(vintage)
    root = _full_label_dir(year)
    result_path = root / "evaluation.json"
    if result_path.is_file():
        return _read_json(result_path)
    daily_ic, topk, daily_topk, topk_candidates, split_metrics = training._predict_split(
        model=model,
        dataset=dataset,
        device=device,
        output_dir=root / "inference",
        split="development",
        batch_size=study.BATCH_SIZE,
        amp_enabled=bool(device.type == "cuda"),
        top_k=study.TOP_K_VALUES,
        write_predictions=False,
        write_path_predictions=False,
        direct_value_horizon=0,
    )
    paths = {
        "topk_metrics": root / "topk_metrics.csv",
        "daily_rank_ic": root / "daily_rank_ic.csv",
        "daily_topk_metrics": root / "daily_topk_metrics.csv",
        "topk_candidates": root / "topk_candidates.parquet",
    }
    _write_csv(paths["topk_metrics"], topk)
    _write_csv(paths["daily_rank_ic"], daily_ic)
    _write_csv(paths["daily_topk_metrics"], daily_topk)
    _write_parquet(paths["topk_candidates"], topk_candidates)
    normalization, normalization_sha256 = _checkpoint_normalization(run_dir)
    metrics = comparison_2026._evaluation_metrics_2026(
        topk=topk, daily_topk=daily_topk, split_metrics=split_metrics
    )
    result = {
        "schema_version": 1,
        "status": "completed",
        "diagnostic_id": DIAGNOSTIC_ID,
        "model_id": MODEL_IDS[year],
        "checkpoint_vintage": year,
        "checkpoint_sha256": study._file_sha256(run_dir / "best_model.pt"),
        "normalization_sha256": normalization_sha256,
        "model_bundle_sha256": integrity.model_bundle_sha256(
            {
                "checkpoint_sha256": study._file_sha256(run_dir / "best_model.pt"),
                "normalization": normalization,
                "architecture": study._architecture_signature(run_dir),
            }
        ),
        "metrics": study._json_safe(metrics),
        "candidate_count": int(split_metrics["row_count"]),
        "date_count": int(split_metrics["date_count"]),
        "outputs": {key: str(path.resolve()) for key, path in paths.items()},
        "output_sha256": {
            key: study._file_sha256(path) for key, path in paths.items()
        },
        "completed_at": study._now(),
    }
    _write_json(result_path, result)
    return result


def _evaluate_loaded_extended(
    vintage: int,
    *,
    model: torch.nn.Module,
    dataset: training.SequencePathPackDataset,
    device: torch.device,
) -> Path:
    year = int(vintage)
    target = _extended_prediction_path(year)
    if target.is_file():
        frame = pd.read_parquet(
            target, columns=["trade_date", "date_idx", "symbol_idx", "score", "predicted_exit_day"]
        )
        if (
            len(frame) == 0
            or frame["trade_date"].nunique() != study.EXPECTED_EXTENDED_DATES
            or bool(frame.duplicated(["date_idx", "symbol_idx"]).any())
            or not bool(np.isfinite(frame[["score", "predicted_exit_day"]]).all().all())
        ):
            raise ValueError(f"stored extended forecast is invalid for {year}")
        return target
    suffix = _read_json(study.OVERLAY_ROOT / "manifest.json")
    candidate_path = Path(
        str(dict(suffix["candidate_indexes"])["extended_native_180"]["path"])
    ).resolve()
    candidates = pd.read_parquet(candidate_path)
    accessor = study.ExtendedFeatureAccessor(dataset, suffix)
    guard = finite._MemoryGuard()
    rows: list[pd.DataFrame] = []
    next_bucket = 10
    try:
        for start in range(0, len(candidates), study.BATCH_SIZE):
            guard.check()
            stop = min(start + study.BATCH_SIZE, len(candidates))
            part = candidates.iloc[start:stop]
            dates = part["date_idx"].to_numpy(dtype=np.int64)
            symbols = part["symbol_idx"].to_numpy(dtype=np.int64)
            x = torch.from_numpy(accessor.batch_x(dates, symbols)).to(device)
            symbol_tensor = torch.from_numpy(symbols.astype(np.int64, copy=False)).to(device)
            with torch.no_grad(), torch.amp.autocast(
                device_type=device.type, enabled=device.type == "cuda"
            ):
                output = model(x, symbol_idx=symbol_tensor)
            path = output["future_path"].detach().float().cpu().numpy()
            summary = training._derive_path_summary_numpy(
                path, price_anchor=dataset.price_anchor, earliest_exit_day=2
            )
            columns = training.derived_path_summary_columns(dataset.forward_days, path_dim=4)
            score = summary[
                :,
                columns.index(
                    training.value_column_for_path(dataset.forward_days, path_dim=4)
                ),
            ]
            planned = summary[:, columns.index(f"best_exit_day_{dataset.forward_days}d")]
            if not bool(np.isfinite(score).all()) or not bool(np.isfinite(planned).all()):
                raise ValueError(f"non-finite extended prediction for checkpoint {year}")
            rows.append(
                part[
                    [
                        "candidate_id",
                        "trade_date",
                        "date_idx",
                        "symbol_idx",
                        "symbol",
                        "entry_filled",
                    ]
                ].assign(
                    score=score.astype(np.float64),
                    predicted_exit_day=planned.astype(np.int16),
                )
            )
            percent = int(stop * 100 / len(candidates))
            if percent >= next_bucket:
                print(f"checkpoint {year}: extended inference {next_bucket}%", flush=True)
                next_bucket += 10
            del x, symbol_tensor, output, path, summary, score, planned
    finally:
        accessor.close()
    forecast = pd.concat(rows, ignore_index=True).sort_values(
        ["date_idx", "symbol_idx"], kind="mergesort"
    ).reset_index(drop=True)
    if (
        len(forecast) != len(candidates)
        or forecast["trade_date"].nunique() != study.EXPECTED_EXTENDED_DATES
        or bool(forecast.duplicated(["date_idx", "symbol_idx"]).any())
    ):
        raise ValueError(f"extended candidate coverage drifted for checkpoint {year}")
    _write_parquet(target, forecast)
    return target


def _vintage_complete(vintage: int) -> bool:
    year = int(vintage)
    if year not in NEW_INFERENCE_VINTAGES:
        return True
    result = _full_label_dir(year) / "evaluation.json"
    forecast = _extended_prediction_path(year)
    if not result.is_file() or not forecast.is_file():
        return False
    payload = _read_json(result)
    keys = pd.read_parquet(forecast, columns=["trade_date", "date_idx", "symbol_idx", "score"])
    return (
        str(payload.get("status")) == "completed"
        and int(payload.get("candidate_count", 0)) == study.EXPECTED_NATIVE_FULL_LABEL_CANDIDATES
        and keys["trade_date"].nunique() == study.EXPECTED_EXTENDED_DATES
        and not bool(keys.duplicated(["date_idx", "symbol_idx"]).any())
        and bool(np.isfinite(keys["score"]).all())
    )


def evaluate_vintages(*, max_jobs: int = 0, output_root: Path = OUTPUT_ROOT) -> dict[str, Any]:
    prepare_vintage_diagnostics(output_root=output_root)
    limit = len(NEW_INFERENCE_VINTAGES) if int(max_jobs) <= 0 else int(max_jobs)
    completed = 0
    for year in NEW_INFERENCE_VINTAGES:
        if _vintage_complete(year):
            continue
        if completed >= limit:
            break
        print(f"checkpoint {year}: full-label and extended 2026 inference", flush=True)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model, dataset, run_dir = _load_model_dataset(year, device=device)
        try:
            _evaluate_loaded_full_label(
                year, model=model, dataset=dataset, run_dir=run_dir, device=device
            )
            _evaluate_loaded_extended(
                year, model=model, dataset=dataset, device=device
            )
        finally:
            del model, dataset
            if device.type == "cuda":
                torch.cuda.empty_cache()
            gc.collect()
        completed += 1
        print(f"checkpoint {year}: inference completed", flush=True)
    if all(_vintage_complete(year) for year in NEW_INFERENCE_VINTAGES):
        return summarize_vintages(output_root=output_root)
    return {
        "status": "partial",
        "completed_this_invocation": completed,
        "pending_vintages": [
            year for year in NEW_INFERENCE_VINTAGES if not _vintage_complete(year)
        ],
    }


def _full_label_metrics() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    fields = (
        "rank_ic",
        "rank_ic_positive_day_rate",
        "path_mae",
        "path_open_mae",
        "path_high_mae",
        "path_low_mae",
        "path_close_mae",
        "exit_regret",
        "top1_base_alpha",
        "top1_stress_alpha",
        "top3_base_alpha",
        "top3_stress_alpha",
        "candidate_score_coverage",
        "execution_return_coverage",
    )
    for year in VINTAGES:
        result = _full_label_result(year)
        metrics = dict(result["metrics"])
        rows.append(
            {
                "checkpoint_vintage": year,
                "model_id": MODEL_IDS[year],
                **{field: metrics.get(field) for field in fields},
                "candidate_count": int(result.get("candidate_count", 0)),
                "date_count": int(result.get("date_count", 0)),
            }
        )
    frame = pd.DataFrame(rows)
    _write_csv(OUTPUT_ROOT / "full_label_vintage_metrics.csv", frame)
    return frame


def _forecast_book(frame: pd.DataFrame, *, vintage: int, top_k: int) -> finite.ForecastBook:
    book = finite.ForecastBook(f"{MODEL_IDS[int(vintage)]}_top{top_k}")
    for date_idx_raw, group in frame.groupby("date_idx", sort=True):
        book.add_day(
            date_idx=int(date_idx_raw),
            symbol_idx=group["symbol_idx"].to_numpy(dtype=np.int32),
            score=group["score"].to_numpy(dtype=np.float64),
            planned_day=group["predicted_exit_day"].to_numpy(dtype=np.int16),
        )
    for date_idx, day in list(book.days.items()):
        selected = np.argsort(-day.score, kind="mergesort")[: int(top_k)]
        book.days[date_idx] = finite.ForecastDay(
            symbol_idx=day.symbol_idx,
            score=day.score,
            planned_day=day.planned_day,
            top3_symbol_idx=tuple(int(value) for value in day.symbol_idx[selected]),
        )
    return book


def _annualized_log_growth(total_return: float, sessions: int) -> float:
    if int(sessions) <= 0 or float(total_return) <= -1.0:
        return float("-inf")
    return float(math.log1p(float(total_return)) * 252.0 / float(sessions))


def _model_plan_accounts() -> pd.DataFrame:
    target = OUTPUT_ROOT / "model_plan_account_metrics.csv"
    market = prior._market()
    date_to_idx = {str(value): idx for idx, value in enumerate(market.date_values)}
    first_idx = int(date_to_idx[study.SIGNAL_START])
    last_idx = int(date_to_idx[study.FULL_LABEL_END])
    guard = finite._MemoryGuard()
    rows: list[dict[str, Any]] = []
    for year in VINTAGES:
        forecast = pd.read_parquet(_extended_prediction_path(year))
        for top_k, slots in ((1, 1), (3, 3)):
            book = _forecast_book(forecast, vintage=year, top_k=top_k)
            for cost in study.COST_SCENARIOS:
                root = (
                    OUTPUT_ROOT
                    / "model_plan_accounts"
                    / MODEL_IDS[year]
                    / f"top{top_k}_slots{slots}"
                    / cost
                )
                result_path = root / "evaluation.json"
                if result_path.is_file():
                    result = _read_json(result_path)
                else:
                    metric, daily, trades, annual = finite.simulate_portfolio(
                        market=market,
                        book=book,
                        raw_top3_paths={},
                        policy=MODEL_PLAN,
                        slots=slots,
                        cost_scenario=cost,
                        first_signal_date_idx=first_idx,
                        last_signal_date_idx=last_idx,
                        starting_cash=study.STARTING_CASH_CNY,
                        memory_guard=guard,
                    )
                    paths = {
                        "daily": root / "daily.parquet",
                        "trades": root / "trades.parquet",
                        "annual": root / "annual.parquet",
                    }
                    _write_parquet(paths["daily"], daily)
                    _write_parquet(paths["trades"], trades)
                    _write_parquet(paths["annual"], pd.DataFrame(annual))
                    row = {
                        "checkpoint_vintage": year,
                        "model_id": MODEL_IDS[year],
                        "top_k": top_k,
                        "slot_count": slots,
                        "policy_name": "model_plan",
                        "cost_scenario": cost,
                        **metric,
                        "signal_period_annualized_log_growth": _annualized_log_growth(
                            float(metric["signal_period_total_return"]),
                            int(metric["signal_session_count"]),
                        ),
                        "liquidated_annualized_log_growth": _annualized_log_growth(
                            float(metric["liquidated_total_return"]),
                            int(metric["equity_path_session_count"]),
                        ),
                    }
                    result = {
                        "schema_version": 1,
                        "status": "completed",
                        "diagnostic_id": DIAGNOSTIC_ID,
                        "metrics": study._json_safe(row),
                        "outputs": {key: str(path.resolve()) for key, path in paths.items()},
                        "completed_at": study._now(),
                    }
                    _write_json(result_path, result)
                rows.append(dict(result["metrics"]))
    frame = pd.DataFrame(rows).sort_values(
        ["checkpoint_vintage", "top_k", "cost_scenario"], kind="mergesort"
    )
    _write_csv(target, frame)
    return frame


def _d7_accounts() -> pd.DataFrame:
    prior_rows = prior._run_d7_accounts().copy()
    prior_rows["checkpoint_vintage"] = prior_rows["model_id"].map(
        {prior.MODEL_2025: 2025, prior.MODEL_2026: 2026}
    )
    rows = prior_rows.to_dict("records")
    audit = CandidateCompleteAuditPack(study.FROZEN_2026_OVERLAY)
    for year in NEW_INFERENCE_VINTAGES:
        forecast = pd.read_parquet(_extended_prediction_path(year))
        for spec in study.ACCOUNT_SPECS:
            for cost in study.COST_SCENARIOS:
                for window, end in (
                    ("strict_101", study.STRICT_SIGNAL_END),
                    ("extended_121", study.EXTENDED_SIGNAL_END),
                ):
                    root = (
                        OUTPUT_ROOT
                        / "d7_accounts"
                        / MODEL_IDS[year]
                        / spec.strategy_id
                        / cost
                        / window
                    )
                    result_path = root / "evaluation.json"
                    if result_path.is_file():
                        result = _read_json(result_path)
                    else:
                        metric, daily, trades, terminal = study.simulate_partial_d7_account(
                            model_id=MODEL_IDS[year],
                            forecast=forecast,
                            spec=spec,
                            cost_scenario=cost,
                            last_signal_date=end,
                            audit_pack=audit,
                        )
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
                            "diagnostic_id": DIAGNOSTIC_ID,
                            "metrics": study._json_safe(metric),
                            "outputs": {
                                key: str(path.resolve()) for key, path in paths.items()
                            },
                            "completed_at": study._now(),
                        }
                        _write_json(result_path, result)
                    row = dict(result["metrics"])
                    row["window_name"] = window
                    row["checkpoint_vintage"] = year
                    rows.append(row)
    frame = pd.DataFrame(rows)
    frame["checkpoint_vintage"] = frame["checkpoint_vintage"].astype(int)
    frame = frame.sort_values(
        ["checkpoint_vintage", "strategy_id", "cost_scenario", "window_name"],
        kind="mergesort",
    )
    _write_csv(OUTPUT_ROOT / "d7_vintage_metrics.csv", frame)
    return frame


def _pairwise_metrics() -> tuple[pd.DataFrame, pd.DataFrame]:
    forecasts = {
        year: pd.read_parquet(_extended_prediction_path(year)) for year in VINTAGES
    }
    key_columns = ["candidate_id", "trade_date", "date_idx", "symbol_idx", "symbol"]
    reference = forecasts[2026][key_columns]
    for year, frame in forecasts.items():
        if not frame[key_columns].equals(reference):
            raise ValueError(f"checkpoint {year} candidate keys differ on 2026 material")
    daily_rows: list[dict[str, Any]] = []
    for older in (2023, 2024, 2025):
        left = forecasts[older]
        right = forecasts[2026]
        for date_idx_raw, positions in left.groupby("date_idx", sort=True).indices.items():
            idx = np.asarray(positions, dtype=np.int64)
            old_scores = left.iloc[idx]["score"].to_numpy(dtype=np.float64)
            new_scores = right.iloc[idx]["score"].to_numpy(dtype=np.float64)
            old_symbols = left.iloc[idx]["symbol_idx"].to_numpy(dtype=np.int64)
            new_symbols = right.iloc[idx]["symbol_idx"].to_numpy(dtype=np.int64)
            old_order = np.argsort(-old_scores, kind="mergesort")
            new_order = np.argsort(-new_scores, kind="mergesort")
            daily_rows.append(
                {
                    "trade_date": str(left.iloc[idx[0]]["trade_date"]),
                    "date_idx": int(date_idx_raw),
                    "older_vintage": older,
                    "new_vintage": 2026,
                    "score_spearman": float(
                        pd.Series(old_scores).corr(pd.Series(new_scores), method="spearman")
                    ),
                    "top1_same": bool(old_symbols[old_order[0]] == new_symbols[new_order[0]]),
                    "top3_overlap_fraction": float(
                        len(
                            set(old_symbols[old_order[:3]])
                            & set(new_symbols[new_order[:3]])
                        )
                        / 3.0
                    ),
                }
            )
    daily = pd.DataFrame(daily_rows)
    aggregate = (
        daily.groupby(["older_vintage", "new_vintage"], as_index=False)
        .agg(
            date_count=("trade_date", "count"),
            mean_daily_score_spearman=("score_spearman", "mean"),
            top1_same_day_rate=("top1_same", "mean"),
            mean_top3_overlap_fraction=("top3_overlap_fraction", "mean"),
        )
        .sort_values("older_vintage", kind="mergesort")
    )
    _write_csv(OUTPUT_ROOT / "checkpoint_pairwise_daily.csv", daily)
    _write_csv(OUTPUT_ROOT / "checkpoint_pairwise_metrics.csv", aggregate)
    return daily, aggregate


def _forecast_lookup_by_date(
    forecast: pd.DataFrame,
) -> dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    lookup: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for date_idx_raw, group in forecast.groupby("date_idx", sort=True):
        ordered = group.sort_values("symbol_idx", kind="mergesort")
        lookup[int(date_idx_raw)] = (
            ordered["symbol_idx"].to_numpy(dtype=np.int64),
            ordered["score"].to_numpy(dtype=np.float64),
            ordered["predicted_exit_day"].to_numpy(dtype=np.int16),
        )
    return lookup


def _rolling_planned_days(
    *,
    signal_date_idx: int,
    symbol_idx: np.ndarray,
    initial_planned_day: np.ndarray,
    forecast_lookup: Mapping[int, tuple[np.ndarray, np.ndarray, np.ndarray]],
    forward_days: int = 60,
) -> np.ndarray:
    symbols = np.asarray(symbol_idx, dtype=np.int64)
    initial = np.clip(
        np.rint(np.asarray(initial_planned_day, dtype=np.float64)), 2, int(forward_days)
    ).astype(np.int16)
    requested = int(signal_date_idx) + initial.astype(np.int64)
    hard_cap = int(signal_date_idx) + int(forward_days)
    requested = np.minimum(requested, hard_cap)
    for current_date_idx in range(int(signal_date_idx) + 1, hard_cap):
        active = current_date_idx < requested
        if not bool(active.any()):
            break
        material = forecast_lookup.get(int(current_date_idx))
        if material is None:
            continue
        lookup_symbols, lookup_scores, lookup_plans = material
        positions = np.searchsorted(lookup_symbols, symbols)
        valid = positions < len(lookup_symbols)
        matched = np.zeros(len(symbols), dtype=bool)
        matched[valid] = lookup_symbols[positions[valid]] == symbols[valid]
        use = active & matched
        if not bool(use.any()):
            continue
        scores = np.zeros(len(symbols), dtype=np.float64)
        plans = np.full(len(symbols), 60, dtype=np.int16)
        scores[use] = lookup_scores[positions[use]]
        plans[use] = lookup_plans[positions[use]]
        candidates = np.where(
            scores <= 0.0,
            int(current_date_idx) + 1,
            int(current_date_idx) + np.clip(np.rint(plans), 2, 60).astype(np.int64),
        )
        requested[use] = np.minimum(
            requested[use], np.minimum(candidates[use], hard_cap)
        )
    return np.clip(requested - int(signal_date_idx), 2, int(forward_days)).astype(
        np.int16
    )


def _mean(values: np.ndarray) -> float:
    numeric = np.asarray(values, dtype=np.float64)
    finite_values = numeric[np.isfinite(numeric)]
    return float(finite_values.mean()) if finite_values.size else float("nan")


def _cohort_policy_row(
    *,
    vintage: int,
    model_id: str | None = None,
    trade_date: str,
    date_idx: int,
    policy_name: str,
    policy_kind: str,
    fixed_day: int | None,
    top_k: int,
    cost_scenario: str,
    selected_idx: np.ndarray,
    plan: exit_audit.ResolvedPlanBatch,
    cash: exit_audit.CashflowBatch,
) -> dict[str, Any]:
    selected = np.asarray(selected_idx, dtype=np.int64)
    universe = np.arange(cash.net_return.size, dtype=np.int64)
    selected_return = _mean(cash.net_return[selected])
    universe_return = _mean(cash.net_return[universe])
    selected_resolved = _mean(
        np.where(plan.exit_day[selected] >= 0, plan.exit_day[selected], np.nan)
    )
    universe_resolved = _mean(
        np.where(plan.exit_day[universe] >= 0, plan.exit_day[universe], np.nan)
    )
    selected_log_rate = (
        float(np.log1p(max(selected_return, -0.999999999)) / selected_resolved)
        if math.isfinite(selected_resolved) and selected_resolved > 0.0
        else float("nan")
    )
    universe_log_rate = (
        float(np.log1p(max(universe_return, -0.999999999)) / universe_resolved)
        if math.isfinite(universe_resolved) and universe_resolved > 0.0
        else float("nan")
    )
    selected_filled = cash.order_filled[selected]
    universe_filled = cash.order_filled[universe]
    return {
        "checkpoint_vintage": int(vintage),
        "model_id": str(model_id or MODEL_IDS[int(vintage)]),
        "trade_date": str(trade_date),
        "date_idx": int(date_idx),
        "policy_name": str(policy_name),
        "policy_kind": str(policy_kind),
        "fixed_day": fixed_day,
        "top_k": int(top_k),
        "cost_scenario": str(cost_scenario),
        "selected_count": int(selected.size),
        "universe_count": int(universe.size),
        "selected_mean_net_return": selected_return,
        "universe_mean_net_return": universe_return,
        "alpha_mean_net_return": selected_return - universe_return,
        "selected_log_growth_per_session": selected_log_rate,
        "universe_log_growth_per_session": universe_log_rate,
        "alpha_log_growth_per_session": selected_log_rate - universe_log_rate,
        "selected_entry_fill_rate": _mean(selected_filled.astype(float)),
        "universe_entry_fill_rate": _mean(universe_filled.astype(float)),
        "selected_terminal_recovery_rate": _mean(
            (plan.terminal_recovery[selected] & selected_filled).astype(float)
        ),
        "universe_terminal_recovery_rate": _mean(
            (plan.terminal_recovery[universe] & universe_filled).astype(float)
        ),
        "selected_deferred_exit_rate": _mean(
            (
                (plan.exit_day[selected] > plan.planned_day[selected])
                & (~plan.terminal_recovery[selected])
                & selected_filled
            ).astype(float)
        ),
        "universe_deferred_exit_rate": _mean(
            (
                (plan.exit_day[universe] > plan.planned_day[universe])
                & (~plan.terminal_recovery[universe])
                & universe_filled
            ).astype(float)
        ),
        "selected_mean_planned_exit_day": _mean(plan.planned_day[selected]),
        "universe_mean_planned_exit_day": _mean(plan.planned_day[universe]),
        "selected_mean_resolved_exit_day": _mean(
            np.where(plan.exit_day[selected] >= 0, plan.exit_day[selected], np.nan)
        ),
        "universe_mean_resolved_exit_day": _mean(
            np.where(plan.exit_day[universe] >= 0, plan.exit_day[universe], np.nan)
        ),
        "selected_mean_cash_utilization": _mean(cash.cash_utilization[selected]),
        "universe_mean_cash_utilization": _mean(cash.cash_utilization[universe]),
    }


def _policy_alpha_contract() -> dict[str, Any]:
    contract = {
        "schema_version": 3,
        "contract_id": "structured-180x35-2026-cohort-policy-alpha-v3",
        "diagnostic_id": DIAGNOSTIC_ID,
        "checkpoint_vintages": list(VINTAGES),
        "signal_window": [study.SIGNAL_START, study.FULL_LABEL_END],
        "signal_date_count": 48,
        "candidate_mode": "native_180",
        "policies": list(POLICY_ALPHA_NAMES),
        "top_k": [1, 3],
        "cost_scenarios": list(study.COST_SCENARIOS),
        "daily_aggregation": "equal_weight_by_signal_date",
        "alpha_definition": "selected_mean_net_return_minus_universe_mean_net_return",
        "rolling_semantics": {
            "fresh_forecast_observed_at_close": True,
            "same_close_exit": False,
            "nonpositive_score": "request_next_trading_day_close",
            "positive_score": "accelerate_only",
            "hard_cap": "original_signal_D60",
            "sellability_deferral": "through_signal_D80",
        },
        "selection_objective": (
            "maximum_double_slippage_mean_daily_alpha_log_growth_per_resolved_session"
        ),
        "secondary_objectives": [
            "annualized_mean_selected_log_growth_per_resolved_session",
            "mean_daily_cumulative_alpha",
        ],
        "post_hoc_fixed_day_warning": True,
    }
    return {"contract": contract, "contract_sha256": study._canonical_digest(contract)}


def _ensure_policy_speed_columns(daily: pd.DataFrame) -> pd.DataFrame:
    required = {
        "selected_log_growth_per_session",
        "universe_log_growth_per_session",
        "alpha_log_growth_per_session",
    }
    if required.issubset(daily.columns):
        return daily
    frame = daily.copy()
    selected_return = np.maximum(
        frame["selected_mean_net_return"].to_numpy(dtype=np.float64), -0.999999999
    )
    universe_return = np.maximum(
        frame["universe_mean_net_return"].to_numpy(dtype=np.float64), -0.999999999
    )
    selected_days = frame["selected_mean_resolved_exit_day"].to_numpy(dtype=np.float64)
    universe_days = frame["universe_mean_resolved_exit_day"].to_numpy(dtype=np.float64)
    selected_rate = np.divide(
        np.log1p(selected_return),
        selected_days,
        out=np.full(len(frame), np.nan, dtype=np.float64),
        where=np.isfinite(selected_days) & (selected_days > 0.0),
    )
    universe_rate = np.divide(
        np.log1p(universe_return),
        universe_days,
        out=np.full(len(frame), np.nan, dtype=np.float64),
        where=np.isfinite(universe_days) & (universe_days > 0.0),
    )
    frame["selected_log_growth_per_session"] = selected_rate
    frame["universe_log_growth_per_session"] = universe_rate
    frame["alpha_log_growth_per_session"] = selected_rate - universe_rate
    return frame


def _trimmed_mean_10pct(values: pd.Series) -> float:
    numeric = np.sort(values.to_numpy(dtype=np.float64))
    numeric = numeric[np.isfinite(numeric)]
    if numeric.size == 0:
        return float("nan")
    trim = int(math.floor(numeric.size * 0.10))
    if trim == 0 or numeric.size <= 2 * trim:
        return float(numeric.mean())
    return float(numeric[trim:-trim].mean())


def _aggregate_policy_alpha(daily: pd.DataFrame) -> pd.DataFrame:
    daily = _ensure_policy_speed_columns(daily)
    group = (["evaluation_year"] if "evaluation_year" in daily.columns else []) + [
        "checkpoint_vintage",
        "model_id",
        "policy_name",
        "policy_kind",
        "fixed_day",
        "top_k",
        "cost_scenario",
    ]
    distribution = (
        daily.groupby(group, as_index=False, dropna=False, sort=True)
        .agg(
            alpha_median=("alpha_mean_net_return", "median"),
            alpha_trimmed_mean_10pct=("alpha_mean_net_return", _trimmed_mean_10pct),
            alpha_q10=("alpha_mean_net_return", lambda values: float(values.quantile(0.10))),
            alpha_q90=("alpha_mean_net_return", lambda values: float(values.quantile(0.90))),
            selected_return_median=("selected_mean_net_return", "median"),
        )
    )
    speed_distribution = daily.groupby(
        group, as_index=False, dropna=False, sort=True
    ).agg(
        selected_log_speed_median=("selected_log_growth_per_session", "median"),
        selected_log_speed_trimmed_mean_10pct=(
            "selected_log_growth_per_session",
            _trimmed_mean_10pct,
        ),
        alpha_log_speed_median=("alpha_log_growth_per_session", "median"),
        alpha_log_speed_trimmed_mean_10pct=(
            "alpha_log_growth_per_session",
            _trimmed_mean_10pct,
        ),
    )
    speed_columns = [
        "selected_log_speed_median",
        "selected_log_speed_trimmed_mean_10pct",
        "alpha_log_speed_median",
        "alpha_log_speed_trimmed_mean_10pct",
    ]
    speed_distribution[speed_columns] = speed_distribution[speed_columns] * 252.0
    aggregate = (
        daily.groupby(group, as_index=False, dropna=False, sort=True)
        .agg(
            signal_date_count=("trade_date", "count"),
            selected_mean_net_return=("selected_mean_net_return", "mean"),
            universe_mean_net_return=("universe_mean_net_return", "mean"),
            alpha_mean_net_return=("alpha_mean_net_return", "mean"),
            selected_return_std=("selected_mean_net_return", "std"),
            alpha_std=("alpha_mean_net_return", "std"),
            annualized_selected_log_speed=(
                "selected_log_growth_per_session",
                lambda values: float(values.mean() * 252.0),
            ),
            annualized_universe_log_speed=(
                "universe_log_growth_per_session",
                lambda values: float(values.mean() * 252.0),
            ),
            annualized_alpha_log_speed=(
                "alpha_log_growth_per_session",
                lambda values: float(values.mean() * 252.0),
            ),
            selected_entry_fill_rate=("selected_entry_fill_rate", "mean"),
            selected_terminal_recovery_rate=("selected_terminal_recovery_rate", "mean"),
            selected_deferred_exit_rate=("selected_deferred_exit_rate", "mean"),
            selected_mean_planned_exit_day=("selected_mean_planned_exit_day", "mean"),
            selected_mean_resolved_exit_day=("selected_mean_resolved_exit_day", "mean"),
        )
        .merge(distribution, on=group, how="left", validate="one_to_one")
        .merge(speed_distribution, on=group, how="left", validate="one_to_one")
        .merge(
            daily.assign(
                positive_selected_day=daily["selected_mean_net_return"].astype(float) > 0.0,
                positive_alpha_day=daily["alpha_mean_net_return"].astype(float) > 0.0,
                positive_selected_log_speed_day=(
                    daily["selected_log_growth_per_session"].astype(float) > 0.0
                ),
                positive_alpha_log_speed_day=(
                    daily["alpha_log_growth_per_session"].astype(float) > 0.0
                ),
            )
            .groupby(group, as_index=False, dropna=False, sort=True)[
                [
                    "positive_selected_day",
                    "positive_alpha_day",
                    "positive_selected_log_speed_day",
                    "positive_alpha_log_speed_day",
                ]
            ]
            .mean(),
            on=group,
            how="left",
            validate="one_to_one",
        )
    )
    return aggregate


def _best_policy_rows(
    frame: pd.DataFrame,
    *,
    group_columns: list[str],
    objective: str,
) -> pd.DataFrame:
    ordered = frame.sort_values(
        [*group_columns, objective, "policy_name"],
        ascending=[True] * len(group_columns) + [False, True],
        kind="mergesort",
    )
    return ordered.drop_duplicates(group_columns, keep="first").reset_index(drop=True)


def _policy_alpha_scan() -> tuple[pd.DataFrame, pd.DataFrame]:
    contract_path = OUTPUT_ROOT / "policy_alpha_contract_v3.json"
    expected_contract = _policy_alpha_contract()
    if contract_path.is_file():
        if _read_json(contract_path) != expected_contract:
            raise ValueError("cohort policy-alpha contract drifted")
    else:
        _write_json(contract_path, expected_contract)
    daily_path = OUTPUT_ROOT / "policy_alpha_daily.csv"
    aggregate_path = OUTPUT_ROOT / "policy_alpha_metrics.csv"
    if daily_path.is_file():
        daily = _ensure_policy_speed_columns(pd.read_csv(daily_path))
        if len(daily) == EXPECTED_POLICY_ALPHA_DAILY_ROWS:
            _write_csv(daily_path, daily)
            aggregate = _aggregate_policy_alpha(daily)
            _write_csv(aggregate_path, aggregate)
            return daily, aggregate

    audit = CandidateCompleteAuditPack(study.FROZEN_2026_OVERLAY)
    corrected_entry_filled = prior._correct_entry_filled()
    if corrected_entry_filled.shape != audit.exit_sellable.shape:
        raise ValueError("corrected entry mask and execution panel shapes differ")
    rows: list[dict[str, Any]] = []
    completed_dates = 0
    total_dates = len(VINTAGES) * 48
    for year in VINTAGES:
        forecast = pd.read_parquet(_extended_prediction_path(year)).sort_values(
            ["date_idx", "symbol_idx"], kind="mergesort"
        )
        lookup = _forecast_lookup_by_date(forecast)
        signal_frame = forecast[
            forecast["trade_date"].astype(str).le(study.FULL_LABEL_END)
        ]
        if signal_frame["trade_date"].nunique() != 48:
            raise ValueError(f"full-label signal coverage drifted for checkpoint {year}")
        for date_idx_raw, group in signal_frame.groupby("date_idx", sort=True):
            ordered = group.sort_values("symbol_idx", kind="mergesort")
            date_idx = int(date_idx_raw)
            trade_date = str(ordered.iloc[0]["trade_date"])
            symbols = ordered["symbol_idx"].to_numpy(dtype=np.int64)
            scores = ordered["score"].to_numpy(dtype=np.float64)
            initial_plan = ordered["predicted_exit_day"].to_numpy(dtype=np.int16)
            entry_filled = np.asarray(
                corrected_entry_filled[date_idx, symbols], dtype=bool
            )
            if not np.array_equal(
                entry_filled, ordered["entry_filled"].to_numpy(dtype=bool)
            ):
                raise ValueError(f"entry-fill identity drift for checkpoint {year}/{trade_date}")
            entry_prices, exit_prices, sellable = audit.execution_paths(date_idx, symbols)
            next_valid = exit_audit._next_valid_exit_indices(exit_prices, sellable)
            rolling_plan = _rolling_planned_days(
                signal_date_idx=date_idx,
                symbol_idx=symbols,
                initial_planned_day=initial_plan,
                forecast_lookup=lookup,
                forward_days=audit.forward_days,
            )
            plan_days: dict[str, int | np.ndarray] = {
                **{f"fixed_d{day}": day for day in range(2, 61)},
                "model_plan": initial_plan,
                "rolling_path": rolling_plan,
            }
            plans = {
                name: exit_audit.resolve_planned_exit_batch(
                    signal_date_idx=date_idx,
                    entry_filled=entry_filled,
                    entry_prices=entry_prices,
                    exit_prices=exit_prices,
                    next_valid_exit_idx=next_valid,
                    planned_days=days,
                    forward_days=audit.forward_days,
                    execution_days=audit.execution_days,
                    terminal_recovery_fraction=audit.terminal_recovery_fraction,
                )
                for name, days in plan_days.items()
            }
            ranking = np.argsort(-scores, kind="mergesort")
            for top_k in (1, 3):
                selected = ranking[:top_k]
                allocation = float(DEFAULT_DAILY_COHORT_CASH_CNY) / int(top_k)
                for cost in study.COST_SCENARIOS:
                    multiplier = (
                        1.0
                        if cost == "base"
                        else float(audit.contract.stress_slippage_multiplier)
                    )
                    for policy_name, plan in plans.items():
                        cash = exit_audit.cashflow_batch(
                            allocated_cash=allocation,
                            entry_filled=entry_filled,
                            entry_prices=entry_prices,
                            plan=plan,
                            date_values=audit.date_values,
                            contract=audit.contract,
                            slippage_multiplier=multiplier,
                        )
                        rows.append(
                            _cohort_policy_row(
                                vintage=year,
                                trade_date=trade_date,
                                date_idx=date_idx,
                                policy_name=policy_name,
                                policy_kind=(
                                    "fixed"
                                    if policy_name.startswith("fixed_d")
                                    else policy_name
                                ),
                                fixed_day=(
                                    int(policy_name.removeprefix("fixed_d"))
                                    if policy_name.startswith("fixed_d")
                                    else None
                                ),
                                top_k=top_k,
                                cost_scenario=cost,
                                selected_idx=selected,
                                plan=plan,
                                cash=cash,
                            )
                        )
            completed_dates += 1
            if completed_dates % 24 == 0 or completed_dates == total_dates:
                print(
                    f"cohort policy-alpha scan {completed_dates}/{total_dates} signal dates",
                    flush=True,
                )
    daily = _ensure_policy_speed_columns(pd.DataFrame(rows))
    if len(daily) != EXPECTED_POLICY_ALPHA_DAILY_ROWS:
        raise ValueError(
            f"cohort policy-alpha row count drifted: {len(daily)}"
        )
    aggregate = _aggregate_policy_alpha(daily)
    _write_csv(daily_path, daily)
    _write_csv(aggregate_path, aggregate)
    return daily, aggregate


def _historical_policy_alpha_contract() -> dict[str, Any]:
    sources: dict[str, dict[str, Any]] = {}
    for year in HISTORICAL_YEARS:
        run_dir = _run_dir(year)
        summary = _read_json(run_dir / "sequence_path_training_summary.json")
        view_path = Path(str(summary["pack_manifest"])).resolve()
        prediction_path = run_dir / "predictions" / "development_predictions.csv"
        sources[str(year)] = {
            "checkpoint_sha256": str(summary["best_checkpoint_sha256"]),
            "view_sha256": study._file_sha256(view_path),
            "prediction_sha256": study._file_sha256(prediction_path),
            "candidate_index_sha256": str(summary["candidate_index_sha256"]),
            "execution_cost_contract_sha256": str(
                summary["execution_cost_contract_sha256"]
            ),
        }
    contract = {
        "schema_version": 2,
        "contract_id": "structured-180x35-historical-cohort-policy-alpha-v2",
        "diagnostic_id": DIAGNOSTIC_ID,
        "evaluation_years": list(HISTORICAL_YEARS),
        "signal_date_counts": {
            str(year): count
            for year, count in HISTORICAL_POLICY_ALPHA_DATE_COUNTS.items()
        },
        "candidate_mode": "original_fold_candidate_index",
        "checkpoint_assignment": "each_checkpoint_evaluates_its_own_development_year",
        "policies": list(HISTORICAL_POLICY_ALPHA_NAMES),
        "top_k": [1, 3],
        "cost_scenarios": list(study.COST_SCENARIOS),
        "daily_aggregation": "equal_weight_by_signal_date",
        "year_aggregation": "equal_weight_by_development_year",
        "alpha_definition": (
            "selected_mean_net_return_minus_universe_mean_net_return_under_same_policy"
        ),
        "rolling_semantics": {
            "fresh_forecast_observed_at_close": True,
            "same_close_exit": False,
            "nonpositive_score": "request_next_trading_day_close",
            "positive_score": "accelerate_only",
            "hard_cap": "original_signal_D60",
            "sellability_deferral": "through_signal_D80",
            "calendar_boundary": (
                "switch_to_next_causal_fold_checkpoint_when_available; "
                "retain_initial_plan_after_the_2025_forecast_tail"
            ),
        },
        "selection_objective": (
            "maximum_double_slippage_mean_daily_alpha_log_growth_per_resolved_session"
        ),
        "robustness_diagnostics": [
            "median_daily_alpha",
            "10pct_trimmed_mean_daily_alpha",
            "positive_alpha_day_rate",
            "annualized_selected_log_speed",
        ],
        "post_hoc_policy_scan_warning": True,
        "sources": sources,
    }
    return {"contract": contract, "contract_sha256": study._canonical_digest(contract)}


def _historical_forecast_frame(
    *, year: int, audit: CandidateCompleteAuditPack
) -> pd.DataFrame:
    run_dir = _run_dir(year)
    prediction_path = run_dir / "predictions" / "development_predictions.csv"
    prediction = pd.read_csv(
        prediction_path,
        usecols=[
            "trade_date",
            "symbol",
            "score",
            "predicted_exit_day",
            "realized_plan_entry_filled",
        ],
        dtype={"trade_date": str, "symbol": str, "score": np.float64},
    )
    candidates = audit.candidates_for_year(year)
    if len(prediction) != len(candidates):
        raise ValueError(f"historical forecast row count drifted for {year}")
    if not np.array_equal(
        prediction["trade_date"].astype(str).to_numpy(),
        candidates["trade_date"].astype(str).to_numpy(),
    ) or not np.array_equal(
        prediction["symbol"].astype(str).to_numpy(),
        candidates["symbol"].astype(str).to_numpy(),
    ):
        raise ValueError(f"historical forecast candidate order drifted for {year}")
    score = pd.to_numeric(prediction["score"], errors="coerce").to_numpy(
        dtype=np.float64
    )
    planned = pd.to_numeric(
        prediction["predicted_exit_day"], errors="coerce"
    ).to_numpy(dtype=np.float64)
    if not bool(np.isfinite(score).all()) or not bool(np.isfinite(planned).all()):
        raise ValueError(f"historical forecast coverage is incomplete for {year}")
    entry_filled = np.asarray(
        audit.entry_filled[
            candidates["date_idx"].to_numpy(dtype=np.int64),
            candidates["symbol_idx"].to_numpy(dtype=np.int64),
        ],
        dtype=bool,
    )
    if not np.array_equal(entry_filled, candidates["entry_filled"].to_numpy(bool)):
        raise ValueError(f"historical candidate entry identity drifted for {year}")
    if not np.array_equal(
        entry_filled,
        prediction["realized_plan_entry_filled"].to_numpy(dtype=bool),
    ):
        raise ValueError(f"historical prediction entry identity drifted for {year}")
    frame = candidates[
        ["candidate_id", "trade_date", "date_idx", "symbol_idx", "symbol", "entry_filled"]
    ].copy()
    frame["score"] = score
    frame["predicted_exit_day"] = np.clip(
        np.rint(planned), 2, 60
    ).astype(np.int16)
    return frame


def _historical_policy_alpha_scan() -> tuple[pd.DataFrame, pd.DataFrame]:
    contract_path = OUTPUT_ROOT / "historical_policy_alpha_contract_v2.json"
    expected_contract = _historical_policy_alpha_contract()
    if contract_path.is_file():
        if _read_json(contract_path) != expected_contract:
            raise ValueError("historical cohort policy-alpha contract drifted")
    else:
        _write_json(contract_path, expected_contract)

    daily_path = OUTPUT_ROOT / "historical_policy_alpha_daily.csv"
    aggregate_path = OUTPUT_ROOT / "historical_policy_alpha_metrics.csv"
    if daily_path.is_file():
        daily = _ensure_policy_speed_columns(pd.read_csv(daily_path))
        if len(daily) == EXPECTED_HISTORICAL_POLICY_ALPHA_DAILY_ROWS:
            aggregate = _aggregate_policy_alpha(daily)
            _write_csv(daily_path, daily)
            _write_csv(aggregate_path, aggregate)
            return daily, aggregate

    audits: dict[int, CandidateCompleteAuditPack] = {}
    forecasts: dict[int, pd.DataFrame] = {}
    forecast_lookup: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    reference_dates: np.ndarray | None = None
    reference_symbols: np.ndarray | None = None
    for year in HISTORICAL_YEARS:
        summary = _read_json(_run_dir(year) / "sequence_path_training_summary.json")
        audit = CandidateCompleteAuditPack(Path(str(summary["pack_manifest"])))
        if reference_dates is None:
            reference_dates = np.asarray(audit.date_values, dtype=object)
            reference_symbols = np.asarray(audit.symbol_values, dtype=object)
        elif not np.array_equal(reference_dates, audit.date_values) or not np.array_equal(
            reference_symbols, audit.symbol_values
        ):
            raise ValueError("historical fold date/symbol axes differ")
        frame = _historical_forecast_frame(year=year, audit=audit)
        observed_dates = int(frame["trade_date"].nunique())
        if observed_dates != HISTORICAL_POLICY_ALPHA_DATE_COUNTS[year]:
            raise ValueError(f"historical signal-date coverage drifted for {year}")
        lookup = _forecast_lookup_by_date(frame)
        overlap = set(forecast_lookup).intersection(lookup)
        if overlap:
            raise ValueError(f"historical forecast date indices overlap: {sorted(overlap)[:3]}")
        forecast_lookup.update(lookup)
        audits[year] = audit
        forecasts[year] = frame

    guard = finite._MemoryGuard()
    completed_dates = 0
    total_dates = sum(HISTORICAL_POLICY_ALPHA_DATE_COUNTS.values())
    year_frames: list[pd.DataFrame] = []
    partial_root = OUTPUT_ROOT / "historical_policy_alpha"
    for year in HISTORICAL_YEARS:
        expected_rows = (
            HISTORICAL_POLICY_ALPHA_DATE_COUNTS[year]
            * len(HISTORICAL_POLICY_ALPHA_NAMES)
            * 2
            * len(study.COST_SCENARIOS)
        )
        year_path = partial_root / f"{year}_daily.csv"
        if year_path.is_file():
            cached = _ensure_policy_speed_columns(pd.read_csv(year_path))
            if len(cached) == expected_rows:
                year_frames.append(cached)
                completed_dates += HISTORICAL_POLICY_ALPHA_DATE_COUNTS[year]
                print(
                    f"historical policy-alpha scan reused {year} "
                    f"({completed_dates}/{total_dates} signal dates)",
                    flush=True,
                )
                continue

        audit = audits[year]
        frame = forecasts[year]
        rows: list[dict[str, Any]] = []
        for date_idx_raw, group in frame.groupby("date_idx", sort=True):
            guard.check()
            ordered = group.sort_values("symbol_idx", kind="mergesort")
            date_idx = int(date_idx_raw)
            trade_date = str(ordered.iloc[0]["trade_date"])
            symbols = ordered["symbol_idx"].to_numpy(dtype=np.int64)
            scores = ordered["score"].to_numpy(dtype=np.float64)
            initial_plan = ordered["predicted_exit_day"].to_numpy(dtype=np.int16)
            entry_filled = ordered["entry_filled"].to_numpy(dtype=bool)
            entry_prices, exit_prices, sellable = audit.execution_paths(date_idx, symbols)
            next_valid = exit_audit._next_valid_exit_indices(exit_prices, sellable)
            rolling_plan = _rolling_planned_days(
                signal_date_idx=date_idx,
                symbol_idx=symbols,
                initial_planned_day=initial_plan,
                forecast_lookup=forecast_lookup,
                forward_days=audit.forward_days,
            )
            plan_days: dict[str, int | np.ndarray] = {
                **{f"fixed_d{day}": day for day in range(2, 61)},
                "model_plan": initial_plan,
                "rolling_path": rolling_plan,
            }
            plans = {
                name: exit_audit.resolve_planned_exit_batch(
                    signal_date_idx=date_idx,
                    entry_filled=entry_filled,
                    entry_prices=entry_prices,
                    exit_prices=exit_prices,
                    next_valid_exit_idx=next_valid,
                    planned_days=days,
                    forward_days=audit.forward_days,
                    execution_days=audit.execution_days,
                    terminal_recovery_fraction=audit.terminal_recovery_fraction,
                )
                for name, days in plan_days.items()
            }
            ranking = np.argsort(-scores, kind="mergesort")
            for top_k in (1, 3):
                selected = ranking[:top_k]
                allocation = float(DEFAULT_DAILY_COHORT_CASH_CNY) / int(top_k)
                for cost in study.COST_SCENARIOS:
                    multiplier = (
                        1.0
                        if cost == "base"
                        else float(audit.contract.stress_slippage_multiplier)
                    )
                    for policy_name, plan in plans.items():
                        cash = exit_audit.cashflow_batch(
                            allocated_cash=allocation,
                            entry_filled=entry_filled,
                            entry_prices=entry_prices,
                            plan=plan,
                            date_values=audit.date_values,
                            contract=audit.contract,
                            slippage_multiplier=multiplier,
                        )
                        row = _cohort_policy_row(
                            vintage=year,
                            model_id=f"structured_180x35_{year}_own_fold",
                            trade_date=trade_date,
                            date_idx=date_idx,
                            policy_name=policy_name,
                            policy_kind=(
                                "fixed"
                                if policy_name.startswith("fixed_d")
                                else policy_name
                            ),
                            fixed_day=(
                                int(policy_name.removeprefix("fixed_d"))
                                if policy_name.startswith("fixed_d")
                                else None
                            ),
                            top_k=top_k,
                            cost_scenario=cost,
                            selected_idx=selected,
                            plan=plan,
                            cash=cash,
                        )
                        row["evaluation_year"] = year
                        rows.append(row)
            completed_dates += 1
            if completed_dates % 25 == 0 or completed_dates == total_dates:
                print(
                    f"historical policy-alpha scan {completed_dates}/{total_dates} "
                    "signal dates",
                    flush=True,
                )
        year_frame = _ensure_policy_speed_columns(pd.DataFrame(rows))
        if len(year_frame) != expected_rows:
            raise ValueError(
                f"historical policy-alpha row count drifted for {year}: "
                f"{len(year_frame)}"
            )
        _write_csv(year_path, year_frame)
        year_frames.append(year_frame)
        del rows, plans, year_frame
        gc.collect()

    daily = pd.concat(year_frames, ignore_index=True, sort=False)
    if len(daily) != EXPECTED_HISTORICAL_POLICY_ALPHA_DAILY_ROWS:
        raise ValueError(f"historical policy-alpha row count drifted: {len(daily)}")
    aggregate = _aggregate_policy_alpha(daily)
    _write_csv(daily_path, daily)
    _write_csv(aggregate_path, aggregate)
    return daily, aggregate


def _historical_policy_selection_summary(metrics: pd.DataFrame) -> dict[str, Any]:
    stress = metrics[
        metrics["cost_scenario"].astype(str).eq("double_slippage")
    ].copy()
    group = ["evaluation_year", "top_k"]
    per_year_best_unit_time_alpha = _best_policy_rows(
        stress,
        group_columns=group,
        objective="annualized_alpha_log_speed",
    )
    per_year_best_selected_speed = _best_policy_rows(
        stress,
        group_columns=group,
        objective="annualized_selected_log_speed",
    )
    per_year_best_cumulative_alpha = _best_policy_rows(
        stress,
        group_columns=group,
        objective="alpha_mean_net_return",
    )
    identity = ["top_k", "policy_name", "policy_kind", "fixed_day"]
    numeric = [
        "selected_mean_net_return",
        "universe_mean_net_return",
        "alpha_mean_net_return",
        "alpha_median",
        "alpha_trimmed_mean_10pct",
        "positive_alpha_day",
        "annualized_selected_log_speed",
        "annualized_alpha_log_speed",
        "selected_log_speed_median",
        "selected_log_speed_trimmed_mean_10pct",
        "alpha_log_speed_median",
        "alpha_log_speed_trimmed_mean_10pct",
        "positive_selected_log_speed_day",
        "positive_alpha_log_speed_day",
        "selected_mean_resolved_exit_day",
    ]
    equal_year = stress.groupby(identity, as_index=False, dropna=False, sort=True)[
        numeric
    ].mean()
    equal_year_best_unit_time_alpha = _best_policy_rows(
        equal_year,
        group_columns=["top_k"],
        objective="annualized_alpha_log_speed",
    )
    equal_year_best_selected_speed = _best_policy_rows(
        equal_year,
        group_columns=["top_k"],
        objective="annualized_selected_log_speed",
    )
    equal_year_best_cumulative_alpha = _best_policy_rows(
        equal_year,
        group_columns=["top_k"],
        objective="alpha_mean_net_return",
    )
    payload = {
        "schema_version": 2,
        "selection_role": "diagnostic_not_automatic_deployment_change",
        "primary_metric": "double_slippage_mean_daily_unit_time_alpha_log_speed",
        "primary_metric_formula": (
            "mean_by_signal_date[(log1p(selected_return)/selected_resolved_day) - "
            "(log1p(universe_return)/universe_resolved_day)] * 252"
        ),
        "per_year_best_mean_unit_time_alpha": study._json_safe(
            per_year_best_unit_time_alpha.to_dict("records")
        ),
        "per_year_best_mean_unit_time_selected_return": study._json_safe(
            per_year_best_selected_speed.to_dict("records")
        ),
        "equal_year_best_mean_unit_time_alpha": study._json_safe(
            equal_year_best_unit_time_alpha.to_dict("records")
        ),
        "equal_year_best_mean_unit_time_selected_return": study._json_safe(
            equal_year_best_selected_speed.to_dict("records")
        ),
        "secondary_best_cumulative_alpha": {
            "per_year": study._json_safe(
                per_year_best_cumulative_alpha.to_dict("records")
            ),
            "equal_year": study._json_safe(
                equal_year_best_cumulative_alpha.to_dict("records")
            ),
        },
        "equal_year_policy_metrics": study._json_safe(equal_year.to_dict("records")),
    }
    _write_json(OUTPUT_ROOT / "historical_policy_selection_summary.json", payload)
    return payload


def _historical_model_plan() -> tuple[pd.DataFrame, pd.DataFrame]:
    account = pd.read_csv(HISTORICAL_CAPITAL_ROOT / "account_metrics.csv")
    annual = pd.read_csv(HISTORICAL_CAPITAL_ROOT / "account_annual_metrics.csv")
    mask_account = (
        account["model_id"].astype(str).eq("lookback180_turnover")
        & account["policy_kind"].astype(str).eq("model_plan")
        & (
            ((account["top_k"].astype(int) == 1) & (account["slot_count"].astype(int) == 1))
            | ((account["top_k"].astype(int) == 3) & (account["slot_count"].astype(int) == 3))
        )
    )
    mask_annual = (
        annual["model_id"].astype(str).eq("lookback180_turnover")
        & annual["policy_kind"].astype(str).eq("model_plan")
        & (
            ((annual["top_k"].astype(int) == 1) & (annual["slot_count"].astype(int) == 1))
            | ((annual["top_k"].astype(int) == 3) & (annual["slot_count"].astype(int) == 3))
        )
    )
    aggregate = account.loc[mask_account].copy()
    yearly = annual.loc[mask_annual].copy()
    _write_csv(OUTPUT_ROOT / "historical_model_plan_aggregate.csv", aggregate)
    _write_csv(OUTPUT_ROOT / "historical_model_plan_annual.csv", yearly)
    return aggregate, yearly


def _historical_fold_metrics() -> pd.DataFrame:
    fields = (
        "rank_ic",
        "rank_ic_positive_day_rate",
        "path_mae",
        "path_close_mae",
        "exit_regret",
        "top1_selected_base",
        "top1_universe_base",
        "top1_base_alpha",
        "top1_stress_alpha",
        "top3_selected_base",
        "top3_universe_base",
        "top3_base_alpha",
        "top3_stress_alpha",
        "top3_average_exit_day",
        "candidate_score_coverage",
        "top3_execution_return_coverage",
        "candidate_count",
        "date_count",
    )
    rows: list[dict[str, Any]] = []
    for year in (2023, 2024, 2025):
        evaluation = _read_json(_run_dir(year) / "evaluation.json")
        metrics = dict(evaluation["metrics"])
        rows.append(
            {
                "development_year": year,
                "checkpoint_vintage": year,
                **{field: metrics.get(field) for field in fields},
            }
        )
    frame = pd.DataFrame(rows)
    _write_csv(OUTPUT_ROOT / "historical_fold_metrics.csv", frame)
    return frame


def _historical_d7_annual() -> pd.DataFrame:
    annual = pd.read_csv(HISTORICAL_CAPITAL_ROOT / "account_annual_metrics.csv")
    frame = annual[
        annual["model_id"].astype(str).eq("lookback180_turnover")
        & annual["policy_name"].astype(str).eq("fixed_d7")
        & (
            ((annual["top_k"].astype(int) == 1) & (annual["slot_count"].astype(int) == 1))
            | ((annual["top_k"].astype(int) == 3) & (annual["slot_count"].astype(int) == 3))
        )
    ].copy()
    _write_csv(OUTPUT_ROOT / "historical_d7_account_annual.csv", frame)
    return frame


def _historical_rolling_annual() -> pd.DataFrame:
    annual = pd.read_csv(HISTORICAL_CAPITAL_ROOT / "account_annual_metrics.csv")
    frame = annual[
        annual["model_id"].astype(str).eq("lookback180_turnover")
        & annual["policy_name"].astype(str).eq("rolling_path")
        & (
            ((annual["top_k"].astype(int) == 1) & (annual["slot_count"].astype(int) == 1))
            | ((annual["top_k"].astype(int) == 3) & (annual["slot_count"].astype(int) == 3))
        )
    ].copy()
    _write_csv(OUTPUT_ROOT / "historical_rolling_account_annual.csv", frame)
    return frame


def _summary_markdown(
    *,
    health: Mapping[str, Any],
    full: pd.DataFrame,
    historical_fold: pd.DataFrame,
    historical_policy_alpha: pd.DataFrame,
    policy_alpha: pd.DataFrame,
    model_plan: pd.DataFrame,
    historical_yearly: pd.DataFrame,
    historical_d7: pd.DataFrame,
    historical_rolling: pd.DataFrame,
) -> str:
    stress_plan = model_plan[model_plan["cost_scenario"].astype(str).eq("double_slippage")]
    stress_alpha = policy_alpha[
        policy_alpha["cost_scenario"].astype(str).eq("double_slippage")
    ]
    best_unit_time_alpha = _best_policy_rows(
        stress_alpha,
        group_columns=["checkpoint_vintage", "top_k"],
        objective="annualized_alpha_log_speed",
    )
    best_selected_speed = _best_policy_rows(
        stress_alpha,
        group_columns=["checkpoint_vintage", "top_k"],
        objective="annualized_selected_log_speed",
    )
    hist_stress = historical_yearly[
        historical_yearly["cost_scenario"].astype(str).eq("double_slippage")
    ]
    historical_policy_stress = historical_policy_alpha[
        historical_policy_alpha["cost_scenario"].astype(str).eq("double_slippage")
    ].copy()
    historical_best_unit_time_alpha = _best_policy_rows(
        historical_policy_stress,
        group_columns=["evaluation_year", "top_k"],
        objective="annualized_alpha_log_speed",
    )
    historical_best_selected_speed = _best_policy_rows(
        historical_policy_stress,
        group_columns=["evaluation_year", "top_k"],
        objective="annualized_selected_log_speed",
    )
    equal_year = historical_policy_stress.groupby(
        ["top_k", "policy_name", "policy_kind", "fixed_day"],
        as_index=False,
        dropna=False,
        sort=True,
    )[
        [
            "selected_mean_net_return",
            "alpha_mean_net_return",
            "alpha_median",
            "alpha_trimmed_mean_10pct",
            "positive_alpha_day",
            "annualized_selected_log_speed",
            "annualized_alpha_log_speed",
            "selected_log_speed_median",
            "selected_log_speed_trimmed_mean_10pct",
            "alpha_log_speed_median",
            "alpha_log_speed_trimmed_mean_10pct",
            "positive_selected_log_speed_day",
            "positive_alpha_log_speed_day",
        ]
    ].mean()
    equal_best_unit_time_alpha = _best_policy_rows(
        equal_year,
        group_columns=["top_k"],
        objective="annualized_alpha_log_speed",
    )
    equal_best_selected_speed = _best_policy_rows(
        equal_year,
        group_columns=["top_k"],
        objective="annualized_selected_log_speed",
    )
    lines = [
        "# Structured 180×35 四年代在 2026 的统一诊断",
        "",
        f"训练健康性：**{health['status']}**。2026 checkpoint 数值、合同、architecture、normalization 和内存终态均已核验。",
        "",
        "## 2026 完整标签窗口",
        "",
        "| checkpoint | Rank IC | Top1 alpha | Top3 alpha | Path MAE | Exit regret |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in full.sort_values("checkpoint_vintage").itertuples(index=False):
        lines.append(
            f"| {int(row.checkpoint_vintage)} | {row.rank_ic:.4f} | {row.top1_base_alpha:+.2%} | "
            f"{row.top3_base_alpha:+.2%} | {row.path_mae:.2%} | {row.exit_regret:.2%} |"
        )
    lines.extend(
        [
            "",
            "## 2023–2025 各 fold 的每日独立 cohort 指标",
            "",
            "| 年份 | Rank IC | Top1 alpha | Top3 alpha | Path MAE | Exit regret | 平均退出日 |",
            "|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in historical_fold.sort_values("development_year").itertuples(index=False):
        lines.append(
            f"| {int(row.development_year)} | {row.rank_ic:.4f} | {row.top1_base_alpha:+.2%} | "
            f"{row.top3_base_alpha:+.2%} | {row.path_mae:.2%} | {row.exit_regret:.2%} | "
            f"D{row.top3_average_exit_day:.2f} |"
        )
    lines.extend(
        [
            "",
            "## 2023–2025 各年最高平均单位时间 alpha（双滑点）",
            "",
            "| 年份 | TopK | 最佳执行 | 平均 alpha log speed | 中位 speed | 10% 截尾 speed | 正 speed 日率 | 平均退出 |",
            "|---:|---:|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in historical_best_unit_time_alpha.sort_values(
        ["evaluation_year", "top_k"]
    ).itertuples(index=False):
        lines.append(
            f"| {int(row.evaluation_year)} | Top{int(row.top_k)} | {row.policy_name} | "
            f"{row.annualized_alpha_log_speed:+.2%} | "
            f"{row.alpha_log_speed_median:+.2%} | "
            f"{row.alpha_log_speed_trimmed_mean_10pct:+.2%} | "
            f"{row.positive_alpha_log_speed_day:.2%} | "
            f"D{row.selected_mean_resolved_exit_day:.2f} |"
        )
    lines.extend(
        [
            "",
            "## 2023–2025 各年最高平均单位时间 selected growth（辅助口径）",
            "",
            "| 年份 | TopK | 最佳执行 | 平均 selected log speed | 同策略 alpha log speed |",
            "|---:|---:|---|---:|---:|",
        ]
    )
    for row in historical_best_selected_speed.sort_values(
        ["evaluation_year", "top_k"]
    ).itertuples(index=False):
        lines.append(
            f"| {int(row.evaluation_year)} | Top{int(row.top_k)} | {row.policy_name} | "
            f"{row.annualized_selected_log_speed:+.2%} | "
            f"{row.annualized_alpha_log_speed:+.2%} |"
        )
    lines.extend(
        [
            "",
            "## 三年等权的单位时间赢家（双滑点）",
            "",
            "| TopK | 目标 | 最佳执行 | Alpha log speed | Selected log speed |",
            "|---:|---|---|---:|---:|",
        ]
    )
    for objective, frame in (
        ("单位时间 alpha", equal_best_unit_time_alpha),
        ("单位时间 selected growth", equal_best_selected_speed),
    ):
        for row in frame.sort_values("top_k").itertuples(index=False):
            lines.append(
                f"| Top{int(row.top_k)} | {objective} | {row.policy_name} | "
                f"{row.annualized_alpha_log_speed:+.2%} | "
                f"{row.annualized_selected_log_speed:+.2%} |"
            )
    lines.extend(
        [
            "",
            "## 四年代在 2026 的最高平均单位时间 alpha（双滑点）",
            "",
            "| checkpoint | TopK | 最佳执行 | Alpha log speed | 中位 speed | 10% 截尾 speed | 正 speed 日率 |",
            "|---:|---:|---|---:|---:|---:|---:|",
        ]
    )
    for row in best_unit_time_alpha.sort_values(
        ["checkpoint_vintage", "top_k"]
    ).itertuples(index=False):
        lines.append(
            f"| {int(row.checkpoint_vintage)} | Top{int(row.top_k)} | {row.policy_name} | "
            f"{row.annualized_alpha_log_speed:+.2%} | "
            f"{row.alpha_log_speed_median:+.2%} | "
            f"{row.alpha_log_speed_trimmed_mean_10pct:+.2%} | "
            f"{row.positive_alpha_log_speed_day:.2%} |"
        )
    lines.extend(
        [
            "",
            "## 每单位持有时间的最高 selected growth（双滑点）",
            "",
            "| checkpoint | TopK | 最佳执行 | 年化 selected log speed | 年化 alpha log speed | 累计平均 alpha |",
            "|---:|---:|---|---:|---:|---:|",
        ]
    )
    for row in best_selected_speed.sort_values(
        ["checkpoint_vintage", "top_k"]
    ).itertuples(index=False):
        lines.append(
            f"| {int(row.checkpoint_vintage)} | Top{int(row.top_k)} | {row.policy_name} | "
            f"{row.annualized_selected_log_speed:+.2%} | "
            f"{row.annualized_alpha_log_speed:+.2%} | {row.alpha_mean_net_return:+.2%} |"
        )
    lines.extend(
        [
            "",
            "## 按模型自身预测退出时间的连续账户（双滑点）",
            "",
            "| checkpoint | 组合 | 信号期收益 | 完全清算收益 | 最大回撤 | 交易数 | 平均持有日 |",
            "|---:|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in stress_plan.sort_values(["checkpoint_vintage", "top_k"]).itertuples(index=False):
        lines.append(
            f"| {int(row.checkpoint_vintage)} | Top{int(row.top_k)}/{int(row.slot_count)} | "
            f"{row.signal_period_total_return:+.2%} | {row.liquidated_total_return:+.2%} | "
            f"{row.full_path_maximum_drawdown:.2%} | {int(row.closed_trade_count)} | "
            f"{row.mean_occupied_sessions:.2f} |"
        )
    lines.extend(
        [
            "",
            "## 2023–2025 各自开发年（双滑点，模型自身退出）",
            "",
            "| 年份 | 组合 | 年度账户收益 | 最大回撤 | 胜率/Sharpe 口径 |",
            "|---:|---|---:|---:|---:|",
        ]
    )
    for row in hist_stress.sort_values(["year", "top_k"]).itertuples(index=False):
        lines.append(
            f"| {int(row.year)} | Top{int(row.top_k)}/{int(row.slot_count)} | "
            f"{row.net_return:+.2%} | {row.maximum_drawdown:.2%} | Sharpe {row.sharpe_zero_rate:.2f} |"
        )
    d7_stress = historical_d7[
        historical_d7["cost_scenario"].astype(str).eq("double_slippage")
    ]
    lines.extend(
        [
            "",
            "## 2023–2025 固定 D7 连续账户（双滑点）",
            "",
            "| 年份 | 组合 | 年度账户收益 | 最大回撤 | Sharpe |",
            "|---:|---|---:|---:|---:|",
        ]
    )
    for row in d7_stress.sort_values(["year", "top_k"]).itertuples(index=False):
        lines.append(
            f"| {int(row.year)} | Top{int(row.top_k)}/{int(row.slot_count)} | "
            f"{row.net_return:+.2%} | {row.maximum_drawdown:.2%} | {row.sharpe_zero_rate:.2f} |"
        )
    rolling_stress = historical_rolling[
        historical_rolling["cost_scenario"].astype(str).eq("double_slippage")
    ]
    lines.extend(
        [
            "",
            "## 2023–2025 rolling path 连续账户（双滑点）",
            "",
            "| 年份 | 组合 | 年度账户收益 | 最大回撤 | Sharpe |",
            "|---:|---|---:|---:|---:|",
        ]
    )
    for row in rolling_stress.sort_values(["year", "top_k"]).itertuples(index=False):
        lines.append(
            f"| {int(row.year)} | Top{int(row.top_k)}/{int(row.slot_count)} | "
            f"{row.net_return:+.2%} | {row.maximum_drawdown:.2%} | {row.sharpe_zero_rate:.2f} |"
        )
    lines.extend(
        [
            "",
            "## 解释边界",
            "",
            "- 2023–2025 表是三个 fold checkpoint 各自在对应开发年上的拼接连续账户，不是同一个 checkpoint 连续跑三年。",
            "- 2026 `model_plan` 只能使用 48 个具备完整 D2–D60 路径和执行尾部的信号日；不能与 121 日 D7 窗口直接比较总收益。",
            "- D2–D60、model_plan 和 rolling_path 在 2023–2025 各年及 2026 的执行比较都先逐日换算单位时间 alpha，再跨信号日等权平均；连续账户用于资金约束下的最终复核。",
            "- 平均单位时间 alpha 仍可能被极端日影响，因此对应 speed 的中位数、10% 截尾均值与正日率必须同看。",
            "- 从 61 个执行候选中得到的最佳规则属于样本内选择；2026 仅 48 日，不能单凭该窗口改写部署规则。",
            "- D7 是此前按 2023–2025 共同账户复合增长速度选出的固定执行规则；本诊断不重新挑选部署规则。",
        ]
    )
    return "\n".join(lines) + "\n"


def _workspace_relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(study.WORKSPACE_ROOT.resolve())).replace(
            "\\", "/"
        )
    except ValueError:
        return str(path.resolve()).replace("\\", "/")


def _report_source(
    *,
    source_id: str,
    label: str,
    paths: list[Path],
    description: str,
    metric_definitions: list[str],
) -> dict[str, Any]:
    relative_paths = [_workspace_relative(path) for path in paths]
    statements = []
    for path in relative_paths:
        if path.endswith(".csv"):
            statements.append(f"SELECT * FROM read_csv_auto('{path}')")
        elif path.endswith(".json"):
            statements.append(f"SELECT * FROM read_json_auto('{path}')")
    sql = "; ".join(statements)
    return {
        "id": source_id,
        "label": label,
        "path": relative_paths[0] if relative_paths else None,
        "query": {
            "engine": "duckdb",
            "language": "sql",
            "sql": sql,
            "description": description,
            "executed_at": study._now(),
            "tables_used": relative_paths,
            "metric_definitions": metric_definitions,
        },
    }


def build_report_artifact(*, output_root: Path = OUTPUT_ROOT) -> Path:
    """Build the canonical technical report artifact for MCP validation/rendering."""
    if output_root.resolve() != OUTPUT_ROOT.resolve():
        raise ValueError("vintage report requires the canonical output root")
    full = pd.read_csv(OUTPUT_ROOT / "full_label_vintage_metrics.csv")
    historical_fold = pd.read_csv(OUTPUT_ROOT / "historical_fold_metrics.csv")
    historical_policy = pd.read_csv(
        OUTPUT_ROOT / "historical_policy_alpha_metrics.csv"
    )
    historical_yearly = pd.read_csv(OUTPUT_ROOT / "historical_model_plan_annual.csv")
    historical_d7 = pd.read_csv(OUTPUT_ROOT / "historical_d7_account_annual.csv")
    historical_rolling = pd.read_csv(
        OUTPUT_ROOT / "historical_rolling_account_annual.csv"
    )
    alpha = pd.read_csv(OUTPUT_ROOT / "policy_alpha_metrics.csv")
    accounts = pd.read_csv(OUTPUT_ROOT / "model_plan_account_metrics.csv")
    pairwise = pd.read_csv(OUTPUT_ROOT / "checkpoint_pairwise_metrics.csv")
    health = _read_json(OUTPUT_ROOT / "training_health.json")
    summary = _read_json(OUTPUT_ROOT / "comparison_summary.json")

    stress_alpha = alpha[alpha["cost_scenario"].astype(str).eq("double_slippage")].copy()
    vintage_alpha_rows: list[dict[str, Any]] = []
    for row in stress_alpha[
        stress_alpha["policy_name"].astype(str).eq("model_plan")
    ].sort_values(["checkpoint_vintage", "top_k"]).itertuples(index=False):
        vintage_alpha_rows.append(
            {
                "checkpoint": f"{int(row.checkpoint_vintage)} checkpoint",
                "vintage": int(row.checkpoint_vintage),
                "top_k": f"Top{int(row.top_k)}",
                "top_k_value": int(row.top_k),
                "alpha": float(row.alpha_mean_net_return),
                "selected_return": float(row.selected_mean_net_return),
                "alpha_median": float(row.alpha_median),
                "positive_alpha_day": float(row.positive_alpha_day),
            }
        )
    historical_alpha_rows = []
    for row in historical_fold.sort_values("development_year").itertuples(index=False):
        for top_k in (1, 3):
            historical_alpha_rows.append(
                {
                    "year": int(row.development_year),
                    "checkpoint": f"{int(row.development_year)} checkpoint",
                    "top_k": f"Top{top_k}",
                    "top_k_value": top_k,
                    "alpha": float(getattr(row, f"top{top_k}_stress_alpha")),
                    "rank_ic": float(row.rank_ic),
                    "path_mae": float(row.path_mae),
                    "exit_regret": float(row.exit_regret),
                }
            )
    historical_policy_stress = historical_policy[
        historical_policy["cost_scenario"].astype(str).eq("double_slippage")
    ].copy()
    historical_policy_best_unit_time_alpha = _best_policy_rows(
        historical_policy_stress,
        group_columns=["evaluation_year", "top_k"],
        objective="annualized_alpha_log_speed",
    )
    historical_policy_best_selected_speed = _best_policy_rows(
        historical_policy_stress,
        group_columns=["evaluation_year", "top_k"],
        objective="annualized_selected_log_speed",
    )
    historical_policy_winner_rows = []
    for objective, frame in (
        ("最高平均单位时间 alpha", historical_policy_best_unit_time_alpha),
        ("最高平均单位时间 selected growth", historical_policy_best_selected_speed),
    ):
        historical_policy_winner_rows.extend(
            {
                "year": int(row.evaluation_year),
                "top_k": f"Top{int(row.top_k)}",
                "top_k_value": int(row.top_k),
                "objective": objective,
                "policy": str(row.policy_name),
                "selected_return": float(row.selected_mean_net_return),
                "alpha": float(row.alpha_mean_net_return),
                "alpha_median": float(row.alpha_median),
                "alpha_trimmed_mean": float(row.alpha_trimmed_mean_10pct),
                "positive_alpha_day": float(row.positive_alpha_day),
                "annualized_selected_log_speed": float(
                    row.annualized_selected_log_speed
                ),
                "annualized_alpha_log_speed": float(row.annualized_alpha_log_speed),
                "alpha_log_speed_median": float(row.alpha_log_speed_median),
                "alpha_log_speed_trimmed_mean": float(
                    row.alpha_log_speed_trimmed_mean_10pct
                ),
                "positive_alpha_log_speed_day": float(
                    row.positive_alpha_log_speed_day
                ),
                "mean_resolved_exit_day": float(
                    row.selected_mean_resolved_exit_day
                ),
            }
            for row in frame.itertuples(index=False)
        )
    fixed_curve = stress_alpha[
        (stress_alpha["checkpoint_vintage"].astype(int) == 2026)
        & stress_alpha["policy_kind"].astype(str).eq("fixed")
    ].copy()
    fixed_curve_rows = [
        {
            "exit_day": int(row.fixed_day),
            "top_k": f"Top{int(row.top_k)}",
            "top_k_value": int(row.top_k),
            "alpha": float(row.alpha_mean_net_return),
            "selected_return": float(row.selected_mean_net_return),
            "alpha_median": float(row.alpha_median),
            "alpha_trimmed_mean": float(row.alpha_trimmed_mean_10pct),
            "annualized_selected_log_speed": float(row.annualized_selected_log_speed),
            "annualized_alpha_log_speed": float(row.annualized_alpha_log_speed),
            "alpha_log_speed_median": float(row.alpha_log_speed_median),
            "alpha_log_speed_trimmed_mean": float(
                row.alpha_log_speed_trimmed_mean_10pct
            ),
            "positive_alpha_log_speed_day": float(
                row.positive_alpha_log_speed_day
            ),
            "positive_alpha_day": float(row.positive_alpha_day),
        }
        for row in fixed_curve.sort_values(["top_k", "fixed_day"]).itertuples(index=False)
    ]
    named_names = {
        "fixed_d2",
        "fixed_d7",
        "fixed_d52",
        "fixed_d54",
        "fixed_d60",
        "model_plan",
        "rolling_path",
    }
    named = stress_alpha[
        (stress_alpha["checkpoint_vintage"].astype(int) == 2026)
        & stress_alpha["policy_name"].astype(str).isin(named_names)
    ].copy()
    named_rows = [
        {
            "top_k": f"Top{int(row.top_k)}",
            "top_k_value": int(row.top_k),
            "policy": str(row.policy_name),
            "fixed_day": None if pd.isna(row.fixed_day) else int(row.fixed_day),
            "selected_return": float(row.selected_mean_net_return),
            "alpha": float(row.alpha_mean_net_return),
            "alpha_median": float(row.alpha_median),
            "alpha_trimmed_mean": float(row.alpha_trimmed_mean_10pct),
            "positive_alpha_day": float(row.positive_alpha_day),
            "annualized_selected_log_speed": float(row.annualized_selected_log_speed),
            "annualized_alpha_log_speed": float(row.annualized_alpha_log_speed),
            "alpha_log_speed_median": float(row.alpha_log_speed_median),
            "alpha_log_speed_trimmed_mean": float(
                row.alpha_log_speed_trimmed_mean_10pct
            ),
            "positive_alpha_log_speed_day": float(
                row.positive_alpha_log_speed_day
            ),
        }
        for row in named.sort_values(["top_k", "policy_name"]).itertuples(index=False)
    ]
    account_rows = [
        {
            "checkpoint": f"{int(row.checkpoint_vintage)} checkpoint",
            "vintage": int(row.checkpoint_vintage),
            "strategy": f"Top{int(row.top_k)}/{int(row.slot_count)}",
            "top_k": int(row.top_k),
            "window": "2026-01-05..2026-03-19",
            "signal_return": float(row.signal_period_total_return),
            "liquidated_return": float(row.liquidated_total_return),
            "liquidated_log_growth": float(row.liquidated_annualized_log_growth),
            "maximum_drawdown": float(row.full_path_maximum_drawdown),
            "winning_trade_rate": float(row.winning_trade_rate),
            "trade_count": int(row.closed_trade_count),
            "mean_holding_sessions": float(row.mean_occupied_sessions),
        }
        for row in accounts[accounts["cost_scenario"].astype(str).eq("double_slippage")]
        .sort_values(["checkpoint_vintage", "top_k"])
        .itertuples(index=False)
    ]
    historical_account_rows = [
        {
            "year": int(row.year),
            "strategy": f"Top{int(row.top_k)}/{int(row.slot_count)}",
            "policy": "model_plan",
            "return": float(row.net_return),
            "maximum_drawdown": float(row.maximum_drawdown),
            "sharpe": float(row.sharpe_zero_rate),
            "cost_scenario": str(row.cost_scenario),
        }
        for row in historical_yearly[
            historical_yearly["cost_scenario"].astype(str).eq("double_slippage")
        ].sort_values(["year", "top_k"]).itertuples(index=False)
    ]
    historical_d7_rows = [
        {
            "year": int(row.year),
            "strategy": f"Top{int(row.top_k)}/{int(row.slot_count)}",
            "policy": "fixed_d7",
            "return": float(row.net_return),
            "maximum_drawdown": float(row.maximum_drawdown),
            "sharpe": float(row.sharpe_zero_rate),
            "cost_scenario": str(row.cost_scenario),
        }
        for row in historical_d7[
            historical_d7["cost_scenario"].astype(str).eq("double_slippage")
        ].sort_values(["year", "top_k"]).itertuples(index=False)
    ]
    historical_rolling_rows = [
        {
            "year": int(row.year),
            "strategy": f"Top{int(row.top_k)}/{int(row.slot_count)}",
            "policy": "rolling_path",
            "return": float(row.net_return),
            "maximum_drawdown": float(row.maximum_drawdown),
            "sharpe": float(row.sharpe_zero_rate),
            "cost_scenario": str(row.cost_scenario),
        }
        for row in historical_rolling[
            historical_rolling["cost_scenario"].astype(str).eq("double_slippage")
        ].sort_values(["year", "top_k"]).itertuples(index=False)
    ]

    health_2026 = next(
        item for item in health["checkpoints"] if int(item["checkpoint_vintage"]) == 2026
    )
    headline = [
        {
            "train_rows": int(health_2026["train_row_count"]),
            "dev_loss": float(health_2026["best_development_price_loss"]),
            "health_ok": 1.0 if health["status"] == "healthy" else 0.0,
            "old_top1_alpha": float(
                full.loc[full["checkpoint_vintage"].eq(2025), "top1_stress_alpha"].iloc[0]
            ),
            "new_top1_alpha": float(
                full.loc[full["checkpoint_vintage"].eq(2026), "top1_stress_alpha"].iloc[0]
            ),
            "new_top3_alpha": float(
                full.loc[full["checkpoint_vintage"].eq(2026), "top3_stress_alpha"].iloc[0]
            ),
            "pairwise_spearman_2025": float(
                pairwise.loc[pairwise["older_vintage"].eq(2025), "mean_daily_score_spearman"].iloc[0]
            ),
        }
    ]

    historical_unit_time_alpha_takeaway = "；".join(
        f"{int(row.evaluation_year)} Top{int(row.top_k)}={row.policy_name} "
        f"({row.annualized_alpha_log_speed:+.2%})"
        for row in historical_policy_best_unit_time_alpha.sort_values(
            ["evaluation_year", "top_k"]
        ).itertuples(index=False)
    )
    historical_selected_speed_takeaway = "；".join(
        f"{int(row.evaluation_year)} Top{int(row.top_k)}={row.policy_name} "
        f"({row.annualized_selected_log_speed:+.2%})"
        for row in historical_policy_best_selected_speed.sort_values(
            ["evaluation_year", "top_k"]
        ).itertuples(index=False)
    )
    historical_policy_note_body = (
        "## 前三年退出比较：主口径是平均单位时间 alpha\n\n"
        "61 种执行方式都在每个年份的全部信号日上等权比较，账户能否买到并不会改变 cohort "
        "样本数。按双滑点平均单位时间 alpha log speed 的逐年赢家为："
        f"{historical_unit_time_alpha_takeaway}。作为资产增长辅助口径，selected log speed 的逐年赢家为："
        f"{historical_selected_speed_takeaway}。表中同时保留单位时间 alpha 的中位数、10% "
        "截尾均值和正日率；累计平均 alpha 仅作为二级解释指标，不再用于选择退出。"
    )
    best_2026_unit_time_alpha = _best_policy_rows(
        stress_alpha[stress_alpha["checkpoint_vintage"].astype(int).eq(2026)],
        group_columns=["top_k"],
        objective="annualized_alpha_log_speed",
    )
    top1_unit_time = best_2026_unit_time_alpha[
        best_2026_unit_time_alpha["top_k"].astype(int).eq(1)
    ].iloc[0]
    top3_unit_time = best_2026_unit_time_alpha[
        best_2026_unit_time_alpha["top_k"].astype(int).eq(3)
    ].iloc[0]
    policy_note_body = (
        "## 2026 退出比较：按平均单位时间 alpha 选优\n\n"
        f"Top1 的均值赢家是 `{top1_unit_time.policy_name}`，alpha log speed "
        f"{float(top1_unit_time.annualized_alpha_log_speed):+.2%}，但中位数 "
        f"{float(top1_unit_time.alpha_log_speed_median):+.2%}、正日率 "
        f"{float(top1_unit_time.positive_alpha_log_speed_day):.1%}；"
        f"Top3 的赢家是 `{top3_unit_time.policy_name}`，alpha log speed "
        f"{float(top3_unit_time.annualized_alpha_log_speed):+.2%}，中位数 "
        f"{float(top3_unit_time.alpha_log_speed_median):+.2%}、正日率 "
        f"{float(top3_unit_time.positive_alpha_log_speed_day):.1%}。"
        "因此 Top1 即使均值最高也必须接受稳健性否决，而 Top3 的证据相对一致。"
        "累计 D60 alpha 只说明长持有累计收益较大，不再参与退出方案排名。"
    )

    source_vintage = _report_source(
        source_id="vintage_source",
        label="四年代统一 2026 checkpoint 诊断",
        paths=[
            OUTPUT_ROOT / "comparison_summary.json",
            OUTPUT_ROOT / "full_label_vintage_metrics.csv",
            OUTPUT_ROOT / "policy_alpha_metrics.csv",
            OUTPUT_ROOT / "model_plan_account_metrics.csv",
            OUTPUT_ROOT / "checkpoint_pairwise_metrics.csv",
        ],
        description="四个 Structured 180x35 checkpoint 在同一 2026 原生 180 日候选和执行材料上的评价。",
        metric_definitions=[
            "Top-K alpha = 每个信号日 selected mean net return 减 universe mean net return，再对 48 个信号日等权平均。",
            "主选择指标 = 每个信号日的 selected log return / selected resolved day 减 universe log return / universe resolved day，再跨信号日平均并乘 252。",
            "累计 Top-K alpha 仅为二级解释指标，不用于选择退出。",
            "model_plan 使用各 checkpoint 自己预测的合法退出日；固定 D 与 rolling 使用相同排名。",
        ],
    )
    source_historical = _report_source(
        source_id="historical_source",
        label="2023–2025 Structured 180x35 fold 结果",
        paths=[
            OUTPUT_ROOT / "historical_fold_metrics.csv",
            OUTPUT_ROOT / "historical_policy_alpha_metrics.csv",
            OUTPUT_ROOT / "historical_policy_selection_summary.json",
            OUTPUT_ROOT / "historical_model_plan_annual.csv",
            OUTPUT_ROOT / "historical_d7_account_annual.csv",
            OUTPUT_ROOT / "historical_rolling_account_annual.csv",
        ],
        description="每个 fold checkpoint 在其对应开发年的 out-of-sample 指标与连续账户年度结果。",
        metric_definitions=[
            "每年使用对应年份 checkpoint，不是同一 checkpoint 跨三年滚动。",
            "历史 policy 对 D2–D60、model_plan 和 rolling_path 先逐日计算单位时间 alpha log speed，再等权平均，并同时报告中位数、10% 截尾均值与正日率。",
            "单位时间 selected speed 是资产增长辅助口径；累计平均 alpha 是二级解释指标。",
            "账户年度收益为双滑点、无杠杆、禁止 pyramiding 的年度净收益。",
        ],
    )
    source_training = _report_source(
        source_id="training_source",
        label="2026 fold 训练合同与终态",
        paths=[OUTPUT_ROOT / "training_health.json", study.STUDY_ROOT / "study.json"],
        description="2026 fold 的训练日志、样本资格、normalization cutoff、checkpoint 和 memory guard 终态。",
        metric_definitions=[
            "native 180 训练行数排除了不足 180 日历史或 continuity_break 的样本。",
            "epoch 选择固定为 1，不读取 2026 标签挑选 epoch。",
        ],
    )

    manifest: dict[str, Any] = {
        "version": 1,
        "surface": "report",
        "title": "Structured 180×35：四年代与 2026 执行诊断",
        "description": "训练健康性、前三年逐年表现、四年代 2026 泛化和 cohort alpha 执行方式比较。",
        "generatedAt": study._now(),
        "cards": [
            {
                "id": "health_card",
                "dataset": "headline",
                "sourceId": "training_source",
                "description": "2026 fold 训练终态与 native 180 合法样本。",
                "metrics": [
                    {"label": "训练状态（1=通过）", "field": "health_ok", "format": "number"},
                    {"label": "native 180 训练行", "field": "train_rows", "format": "number"},
                    {"label": "最佳开发 price loss", "field": "dev_loss", "format": "number"},
                ],
            },
            {
                "id": "alpha_card",
                "dataset": "headline",
                "sourceId": "vintage_source",
                "description": "同一 2026 完整标签窗口的 Top1/Top3 stress alpha。",
                "metrics": [
                    {"label": "2025 Top1 alpha", "field": "old_top1_alpha", "format": "percent"},
                    {"label": "2026 Top1 alpha", "field": "new_top1_alpha", "format": "percent", "signed": True},
                    {"label": "2026 Top3 alpha", "field": "new_top3_alpha", "format": "percent", "signed": True},
                ],
            },
            {
                "id": "pairwise_card",
                "dataset": "headline",
                "sourceId": "vintage_source",
                "description": "2025 与 2026 在 121 日扩展候选上的逐日 score 一致性。",
                "metrics": [
                    {"label": "Score Spearman", "field": "pairwise_spearman_2025", "format": "number"}
                ],
            },
        ],
        "charts": [
            {
                "id": "vintage_alpha_chart",
                "title": "2026 完整标签窗口的 checkpoint Top-K alpha",
                "subtitle": "48 个信号日、双滑点；每个柱为每日独立 cohort alpha 的等权平均。",
                "type": "bar",
                "dataset": "vintage_alpha_chart",
                "sourceId": "vintage_source",
                "encodings": {
                    "x": {"field": "checkpoint", "type": "ordinal", "label": "Checkpoint"},
                    "y": {"field": "alpha", "type": "quantitative", "format": "percent", "label": "平均 alpha"},
                    "color": {"field": "top_k", "type": "nominal", "label": "Top-K"},
                    "tooltip": [
                        {"field": "selected_return", "format": "percent", "label": "Selected return"},
                        {"field": "alpha_median", "format": "percent", "label": "Alpha 中位数"},
                        {"field": "positive_alpha_day", "format": "percent", "label": "Alpha 正日率"},
                    ],
                },
                "valueFormat": "percent",
                "legend": {"position": "bottom", "sort": "spec"},
                "settings": {"groupMode": "grouped"},
                "layout": "full",
            },
            {
                "id": "historical_alpha_chart",
                "title": "2023–2025 各 fold 的 Top-K alpha",
                "subtitle": "每个 checkpoint 在其对应开发年的 out-of-sample 结果，双滑点。",
                "type": "bar",
                "dataset": "historical_alpha_chart",
                "sourceId": "historical_source",
                "encodings": {
                    "x": {"field": "year", "type": "ordinal", "label": "开发年"},
                    "y": {"field": "alpha", "type": "quantitative", "format": "percent", "label": "Top-K alpha"},
                    "color": {"field": "top_k", "type": "nominal", "label": "Top-K"},
                    "tooltip": [
                        {"field": "rank_ic", "format": "number", "label": "Rank IC"},
                        {"field": "path_mae", "format": "percent", "label": "Path MAE"},
                        {"field": "exit_regret", "format": "percent", "label": "Exit regret"},
                    ],
                },
                "valueFormat": "percent",
                "legend": {"position": "bottom", "sort": "spec"},
                "settings": {"groupMode": "grouped"},
                "layout": "full",
            },
            {
                "id": "fixed_curve_chart",
                "title": "2026 checkpoint 固定退出日的单位时间 alpha 曲线",
                "subtitle": "D2–D60、48 个信号日、双滑点；逐日 alpha log speed 等权平均后年化。",
                "type": "line",
                "dataset": "fixed_curve_2026",
                "sourceId": "vintage_source",
                "encodings": {
                    "x": {"field": "exit_day", "type": "quantitative", "label": "固定退出日 D"},
                    "y": {"field": "annualized_alpha_log_speed", "type": "quantitative", "format": "percent", "label": "平均单位时间 alpha"},
                    "color": {"field": "top_k", "type": "nominal", "label": "Top-K"},
                    "tooltip": [
                        {"field": "selected_return", "format": "percent", "label": "Selected return"},
                        {"field": "annualized_selected_log_speed", "format": "percent", "label": "Selected log speed"},
                        {"field": "alpha_log_speed_median", "format": "percent", "label": "Alpha speed 中位数"},
                        {"field": "alpha_log_speed_trimmed_mean", "format": "percent", "label": "10% 截尾 alpha speed"},
                    ],
                },
                "valueFormat": "percent",
                "legend": {"position": "bottom", "sort": "spec"},
                "layout": "full",
            },
        ],
        "tables": [
            {
                "id": "full_label_table",
                "title": "四年代在 2026 的完整标签指标",
                "subtitle": "2026-01-05 至 2026-03-19；143,300 个共同 native 180 候选。",
                "dataset": "full_label_metrics",
                "sourceId": "vintage_source",
                "defaultSort": {"field": "checkpoint", "direction": "asc"},
                "density": "spacious",
                "columns": [
                    {"field": "checkpoint", "label": "Checkpoint", "type": "text"},
                    {"field": "rank_ic", "label": "Rank IC", "format": "number"},
                    {"field": "ic_positive_rate", "label": "IC 正日率", "format": "percent"},
                    {"field": "path_mae", "label": "Path MAE", "format": "percent"},
                    {"field": "exit_regret", "label": "Exit regret", "format": "percent"},
                    {"field": "top1_alpha", "label": "Top1 alpha", "format": "percent", "movement": True},
                    {"field": "top3_alpha", "label": "Top3 alpha", "format": "percent", "movement": True},
                ],
            },
            {
                "id": "historical_fold_table",
                "title": "2023–2025 各 fold 年度指标",
                "subtitle": "对应年份 checkpoint 的 out-of-sample cohort 结果；双滑点 alpha。",
                "dataset": "historical_fold",
                "sourceId": "historical_source",
                "defaultSort": {"field": "year", "direction": "asc"},
                "density": "spacious",
                "columns": [
                    {"field": "year", "label": "年份", "format": "number"},
                    {"field": "rank_ic", "label": "Rank IC", "format": "number"},
                    {"field": "top1_alpha", "label": "Top1 alpha", "format": "percent", "movement": True},
                    {"field": "top3_alpha", "label": "Top3 alpha", "format": "percent", "movement": True},
                    {"field": "path_mae", "label": "Path MAE", "format": "percent"},
                    {"field": "exit_regret", "label": "Exit regret", "format": "percent"},
                ],
            },
            {
                "id": "historical_policy_winner_table",
                "title": "前三年逐年退出策略赢家",
                "subtitle": "每年对应 checkpoint、双滑点；单位时间 alpha 为主，selected growth 为辅助。",
                "dataset": "historical_policy_winners",
                "sourceId": "historical_source",
                "defaultSort": {"field": "year", "direction": "asc"},
                "density": "dense",
                "columns": [
                    {"field": "year", "label": "年份", "format": "number"},
                    {"field": "top_k", "label": "Top-K", "type": "text"},
                    {"field": "objective", "label": "目标", "type": "text"},
                    {"field": "policy", "label": "最佳执行", "type": "text"},
                    {"field": "annualized_alpha_log_speed", "label": "Alpha log speed", "format": "percent", "movement": True},
                    {"field": "alpha_log_speed_median", "label": "中位 speed", "format": "percent"},
                    {"field": "alpha_log_speed_trimmed_mean", "label": "10% 截尾 speed", "format": "percent"},
                    {"field": "positive_alpha_log_speed_day", "label": "正 speed 日率", "format": "percent"},
                    {"field": "annualized_selected_log_speed", "label": "Selected log speed", "format": "percent"},
                    {"field": "mean_resolved_exit_day", "label": "平均退出日", "format": "number"},
                ],
            },
            {
                "id": "policy_table",
                "title": "2026 checkpoint 退出策略对照",
                "subtitle": "D7、model_plan、rolling_path 与代表性固定 D；主排序为平均单位时间 alpha。",
                "dataset": "policy_named_2026",
                "sourceId": "vintage_source",
                "defaultSort": {"field": "top_k", "direction": "asc"},
                "density": "dense",
                "columns": [
                    {"field": "top_k", "label": "Top-K", "type": "text"},
                    {"field": "policy", "label": "执行", "type": "text"},
                    {"field": "fixed_day", "label": "固定 D", "format": "number"},
                    {"field": "annualized_alpha_log_speed", "label": "Alpha log speed", "format": "percent", "movement": True},
                    {"field": "alpha_log_speed_median", "label": "中位 speed", "format": "percent"},
                    {"field": "alpha_log_speed_trimmed_mean", "label": "10% 截尾 speed", "format": "percent"},
                    {"field": "positive_alpha_log_speed_day", "label": "正 speed 日率", "format": "percent"},
                    {"field": "annualized_selected_log_speed", "label": "Selected log speed", "format": "percent"},
                    {"field": "selected_return", "label": "Selected return", "format": "percent", "movement": True},
                    {"field": "alpha", "label": "累计平均 alpha（辅助）", "format": "percent", "movement": True},
                    {"field": "alpha_median", "label": "Alpha 中位数", "format": "percent"},
                    {"field": "alpha_trimmed_mean", "label": "10% 截尾 alpha", "format": "percent"},
                    {"field": "positive_alpha_day", "label": "Alpha 正日率", "format": "percent"},
                ],
            },
            {
                "id": "account_table",
                "title": "四年代 model_plan 连续账户",
                "subtitle": "2026-01-05 至 2026-03-19、双滑点；48 日信号期，完全清算含共同执行尾部。",
                "dataset": "model_plan_accounts",
                "sourceId": "vintage_source",
                "defaultSort": {"field": "checkpoint", "direction": "asc"},
                "density": "dense",
                "columns": [
                    {"field": "checkpoint", "label": "Checkpoint", "type": "text"},
                    {"field": "strategy", "label": "策略", "type": "text"},
                    {"field": "signal_return", "label": "信号期收益", "format": "percent", "movement": True},
                    {"field": "liquidated_return", "label": "完全清算收益", "format": "percent", "movement": True},
                    {"field": "liquidated_log_growth", "label": "清算 log growth", "format": "percent"},
                    {"field": "maximum_drawdown", "label": "最大回撤", "format": "percent", "movement": True},
                    {"field": "winning_trade_rate", "label": "胜率", "format": "percent"},
                    {"field": "trade_count", "label": "交易数", "format": "number"},
                ],
            },
            {
                "id": "historical_account_table",
                "title": "前三年连续账户年度结果",
                "subtitle": "对应年份 checkpoint、双滑点；model_plan、rolling_path 与预注册 D7 分列。",
                "dataset": "historical_accounts",
                "sourceId": "historical_source",
                "defaultSort": {"field": "year", "direction": "asc"},
                "density": "dense",
                "columns": [
                    {"field": "year", "label": "年份", "format": "number"},
                    {"field": "strategy", "label": "策略", "type": "text"},
                    {"field": "policy", "label": "退出", "type": "text"},
                    {"field": "return", "label": "年度收益", "format": "percent", "movement": True},
                    {"field": "maximum_drawdown", "label": "最大回撤", "format": "percent", "movement": True},
                    {"field": "sharpe", "label": "Sharpe", "format": "number"},
                ],
            },
            {
                "id": "pairwise_table",
                "title": "旧 checkpoint 与 2026 checkpoint 的逐日排序一致性",
                "subtitle": "121 日 extended candidate；Top 端一致性明显低于整体 score 相关性。",
                "dataset": "pairwise",
                "sourceId": "vintage_source",
                "defaultSort": {"field": "older_vintage", "direction": "asc"},
                "density": "spacious",
                "columns": [
                    {"field": "older_vintage", "label": "旧 vintage", "format": "number"},
                    {"field": "mean_daily_score_spearman", "label": "Score Spearman", "format": "number"},
                    {"field": "top1_same_day_rate", "label": "Top1 同选率", "format": "percent"},
                    {"field": "mean_top3_overlap_fraction", "label": "Top3 重合率", "format": "percent"},
                ],
            },
        ],
        "sources": [source_vintage, source_historical, source_training],
        "blocks": [
            {"id": "title", "type": "markdown", "body": "# Structured 180×35：四年代与 2026 执行诊断", "layout": "full"},
            {
                "id": "summary",
                "type": "markdown",
                "body": "## 技术摘要\n\n- **2026 fold 训练本身正常。** 训练完成 1 epoch，checkpoint、normalization、architecture、loss 和 memory guard 均通过；7,674,720 是资格筛选前行数，真正合法 native-180 训练行是 7,396,133。\n- **前三年对应开发年均为正，但 2024→2026 泛化失败。** 2023/2025 checkpoint 在 2026 的 Top1/Top3 alpha 为正，2024 为负；2025 checkpoint 的 Top1 最好，2023 checkpoint 的 Top3 最好。\n- **2026 checkpoint 的整体 Rank IC 较高，不代表 Top 端更好。** 2026 Rank IC 为 0.0946，但 Top1 alpha 为负；差异集中在极端头部，且 model_plan 退出会进一步放大损失。\n- **退出方案现在以平均单位时间 alpha 为主。** 每个信号日先按 resolved holding day 换算 selected 与 universe 的 log growth 差，再跨日期平均；累计 alpha 和 selected speed 只作辅助，最终仍需有限资金账户复核。",
                "layout": "full",
                "sourceId": "vintage_source",
            },
            {"id": "headline", "type": "metric-strip", "cardIds": ["health_card", "alpha_card", "pairwise_card"], "layout": "full"},
            {
                "id": "training",
                "type": "markdown",
                "body": "## 训练健康性：没有发现数值或流程故障\n\n2026 训练的 best epoch 与 completed epoch 都是 1，所有模型张量、AMP scaler 和损失有限；normalization 截止 `2026-01-05`，epoch 选择没有读取 2026 标签。与 2023–2025 同 profile checkpoint 的输入维度、GRU 结构、隐藏层和 dropout 完全一致。低内存 trim 事件均成功，监督样本差异来自预先锁定的 `complete_180_session_close_history_without_continuity_break` 资格规则，不是意外丢样本。",
                "layout": "full",
                "sourceId": "training_source",
            },
            {"id": "vintage_chart_note", "type": "markdown", "body": "## 2026 泛化：2025 checkpoint 的 Top 端最可靠\n\n图中每个柱是 48 个信号日的每日独立 cohort alpha，不受槽位占用影响。2025 的 Top1/Top3 均为正，2023 也保持正 alpha；2024 两个 Top-K 均为负。2026 checkpoint 的 Top3 接近零，但 Top1 已为负。", "layout": "full", "sourceId": "vintage_source"},
            {"id": "vintage_chart_block", "type": "chart", "chartId": "vintage_alpha_chart", "layout": "full"},
            {"id": "full_table_block", "type": "table", "tableId": "full_label_table", "layout": "full"},
            {"id": "divergence", "type": "markdown", "body": "## 为什么 Rank IC 更高，Top alpha 反而更差\n\nRank IC 使用全横截面排序，约 3,000 个候选共同决定一个日度相关系数；Top1/Top3 只看排序最前端。2025 与 2026 的逐日 score Spearman 仍为 0.672，但 Top1 同选率只有 15.7%、Top3 重合率 17.9%，所以全局排序可以改善，头部证券却完全换了一批。当前 rank loss 是 local-chunk 训练，且分数由预测路径效用派生，并没有直接优化全市场 Top1/Top3 的 realized executable alpha。共同候选、label 和费用完全一致，2025/2026 Top1 alpha 的差距主要来自 selected return（+3.85% 对 -2.17%），不是 universe 基线变化（-0.43% 对 -0.45%）。", "layout": "full", "sourceId": "vintage_source"},
            {"id": "pairwise_block", "type": "table", "tableId": "pairwise_table", "layout": "full"},
            {"id": "historical_note", "type": "markdown", "body": "## 前三年逐年表现：IC 没有逐年下降，但路径和 alpha 有年份差异\n\n2023、2024、2025 的对应 fold Rank IC 分别为 0.0965、0.1088、0.1136，不能说 IC 逐年下降；Top1 stress alpha 为 +6.30%、+5.90%、+3.71%，Top3 为 +4.69%、+5.23%、+3.60%。2024 的路径 MAE 与 exit regret 明显较高，2025 的 IC 最好但 Top-K alpha 较低，说明排序质量、路径误差和可执行尾部不是同一个指标。", "layout": "full", "sourceId": "historical_source"},
            {"id": "historical_chart_block", "type": "chart", "chartId": "historical_alpha_chart", "layout": "full"},
            {"id": "historical_table_block", "type": "table", "tableId": "historical_fold_table", "layout": "full"},
            {"id": "historical_policy_note", "type": "markdown", "body": historical_policy_note_body, "layout": "full", "sourceId": "historical_source"},
            {"id": "historical_policy_table_block", "type": "table", "tableId": "historical_policy_winner_table", "layout": "full"},
            {"id": "policy_note", "type": "markdown", "body": policy_note_body, "layout": "full", "sourceId": "vintage_source"},
            {"id": "fixed_curve_block", "type": "chart", "chartId": "fixed_curve_chart", "layout": "full"},
            {"id": "policy_table_block", "type": "table", "tableId": "policy_table", "layout": "full"},
            {"id": "account_note", "type": "markdown", "body": "## 账户结果是可实现性检查，不替代 alpha\n\n在 48 日 2026 窗口中，2025 checkpoint 的 model_plan Top3/3 双滑点完全清算收益为 +18.47%，2026 checkpoint 为 -4.51%；Top1/1 分别为 +2.07% 与 -9.73%。账户只成交 3–14 笔，受到持仓占用和信号重叠强烈影响，因此不能单独否定每日 cohort alpha。相反，alpha 告诉我们每天把股票排到头部是否有持续信息；账户告诉我们在具体槽位、费用和递延约束下能兑现多少。", "layout": "full", "sourceId": "vintage_source"},
            {"id": "account_table_block", "type": "table", "tableId": "account_table", "layout": "full"},
            {"id": "historical_account_note", "type": "markdown", "body": "## 前三年账户：D7 的资金周转优势稳定兑现\n\n虽然 rolling_path 的逐日单位持有时间指标三年等权最高，但固定 D7 在 2023、2024、2025 的 Top1/1 与 Top3/3 六个年度账户比较中全部胜过 model_plan 和 rolling_path。这里没有矛盾：cohort speed 假设每天的信号都能独立部署；有限槽位账户会被持仓重叠、空闲现金、同日进出顺序和跳过信号重新加权。对当前固定槽位执行，D7 仍是已验证的实际回退规则。", "layout": "full", "sourceId": "historical_source"},
            {"id": "historical_account_table_block", "type": "table", "tableId": "historical_account_table", "layout": "full"},
            {"id": "limitations", "type": "markdown", "body": "## 局限性与下一步\n\n- 2026 完整标签只有 48 个信号日；平均单位时间 alpha、其中位数和截尾均值仍是诊断，不是正式置信区间。\n- D2–D60 与 model_plan/rolling_path 的 policy scan 使用同一评价窗口，存在期限选择的事后偏差；D7 仍是此前三年共同账户选出的预注册基线。\n- 2023–2025 年度账户表是各 fold 在自己开发年的结果，不是同一个模型跨三年的独立验证。\n- 下一步应在新增独立数据上预注册“平均单位时间 alpha + 稳健分布 + 有限资金增长”的联合规则，并继续把 2025 checkpoint 作为当前部署候选，2026 checkpoint 保留为实验结果。", "layout": "full", "sourceId": "vintage_source"},
        ],
    }
    snapshot = {
        "version": 1,
        "generatedAt": study._now(),
        "status": "ready",
        "datasets": {
            "headline": headline,
            "vintage_alpha_chart": vintage_alpha_rows,
            "historical_alpha_chart": historical_alpha_rows,
            "fixed_curve_2026": fixed_curve_rows,
            "policy_named_2026": named_rows,
            "full_label_metrics": [
                {
                    "checkpoint": f"{int(row.checkpoint_vintage)} checkpoint",
                    "rank_ic": float(row.rank_ic),
                    "ic_positive_rate": float(row.rank_ic_positive_day_rate),
                    "path_mae": float(row.path_mae),
                    "exit_regret": float(row.exit_regret),
                    "top1_alpha": float(row.top1_stress_alpha),
                    "top3_alpha": float(row.top3_stress_alpha),
                }
                for row in full.sort_values("checkpoint_vintage").itertuples(index=False)
            ],
            "historical_fold": [
                {
                    "year": int(row.development_year),
                    "rank_ic": float(row.rank_ic),
                    "top1_alpha": float(row.top1_stress_alpha),
                    "top3_alpha": float(row.top3_stress_alpha),
                    "path_mae": float(row.path_mae),
                    "exit_regret": float(row.exit_regret),
                }
                for row in historical_fold.sort_values("development_year").itertuples(index=False)
            ],
            "historical_policy_winners": historical_policy_winner_rows,
            "model_plan_accounts": account_rows,
            "historical_accounts": (
                historical_account_rows
                + historical_rolling_rows
                + historical_d7_rows
            ),
            "pairwise": [
                {
                    "older_vintage": int(row.older_vintage),
                    "mean_daily_score_spearman": float(row.mean_daily_score_spearman),
                    "top1_same_day_rate": float(row.top1_same_day_rate),
                    "mean_top3_overlap_fraction": float(row.mean_top3_overlap_fraction),
                }
                for row in pairwise.sort_values("older_vintage").itertuples(index=False)
            ],
        },
    }
    artifact = {"surface": "report", "manifest": manifest, "snapshot": snapshot}
    target = OUTPUT_ROOT / "report_artifact.json"
    _write_json(target, artifact)
    return target


def summarize_vintages(*, output_root: Path = OUTPUT_ROOT) -> dict[str, Any]:
    if output_root.resolve() != OUTPUT_ROOT.resolve():
        raise ValueError("vintage diagnostics require the canonical output root")
    if not all(_vintage_complete(year) for year in NEW_INFERENCE_VINTAGES):
        raise RuntimeError("2023/2024 inference is incomplete")
    health = _training_health()
    _write_json(OUTPUT_ROOT / "training_health.json", health)
    full = _full_label_metrics()
    historical_fold = _historical_fold_metrics()
    _policy_alpha_daily, policy_alpha = _policy_alpha_scan()
    _historical_policy_daily, historical_policy_alpha = (
        _historical_policy_alpha_scan()
    )
    historical_policy_selection = _historical_policy_selection_summary(
        historical_policy_alpha
    )
    model_plan = _model_plan_accounts()
    d7 = _d7_accounts()
    _daily_pairwise, pairwise = _pairwise_metrics()
    historical_aggregate, historical_yearly = _historical_model_plan()
    historical_d7 = _historical_d7_annual()
    historical_rolling = _historical_rolling_annual()
    stress_plan = model_plan[
        model_plan["cost_scenario"].astype(str).eq("double_slippage")
    ].copy()
    best_plan = (
        stress_plan.sort_values(
            ["top_k", "liquidated_total_return", "checkpoint_vintage"],
            ascending=[True, False, False],
            kind="mergesort",
        )
        .groupby("top_k", as_index=False)
        .first()
    )
    stress_alpha = policy_alpha[
        policy_alpha["cost_scenario"].astype(str).eq("double_slippage")
    ]
    best_policy_unit_time_alpha = _best_policy_rows(
        stress_alpha,
        group_columns=["checkpoint_vintage", "top_k"],
        objective="annualized_alpha_log_speed",
    )
    best_policy_selected_speed = _best_policy_rows(
        stress_alpha,
        group_columns=["checkpoint_vintage", "top_k"],
        objective="annualized_selected_log_speed",
    )
    best_policy_cumulative_alpha = _best_policy_rows(
        stress_alpha,
        group_columns=["checkpoint_vintage", "top_k"],
        objective="alpha_mean_net_return",
    )
    summary = {
        "schema_version": 2,
        "status": "completed",
        "diagnostic_id": DIAGNOSTIC_ID,
        "completed_at": study._now(),
        "training_health": health,
        "full_label_vintage_metrics": study._json_safe(full.to_dict("records")),
        "historical_fold_metrics": study._json_safe(
            historical_fold.to_dict("records")
        ),
        "historical_policy_selection": historical_policy_selection,
        "primary_best_mean_unit_time_alpha_double_slippage": study._json_safe(
            best_policy_unit_time_alpha.to_dict("records")
        ),
        "secondary_best_mean_unit_time_selected_return_double_slippage": study._json_safe(
            best_policy_selected_speed.to_dict("records")
        ),
        "secondary_best_cumulative_alpha_double_slippage": study._json_safe(
            best_policy_cumulative_alpha.to_dict("records")
        ),
        "model_plan_account_metrics": study._json_safe(model_plan.to_dict("records")),
        "d7_vintage_metrics": study._json_safe(d7.to_dict("records")),
        "checkpoint_pairwise_metrics": study._json_safe(pairwise.to_dict("records")),
        "best_model_plan_by_top_k_double_slippage": study._json_safe(
            best_plan.to_dict("records")
        ),
        "historical_model_plan_contract": {
            "years": [2023, 2024, 2025],
            "model_id": "lookback180_turnover",
            "meaning": "each fold checkpoint trades its own development year",
            "policy": "model_plan",
            "aggregate_rows": int(len(historical_aggregate)),
            "annual_rows": int(len(historical_yearly)),
        },
        "protected_boundaries": {
            "qdp_changed": False,
            "provider_called": False,
            "checkpoint_changed": False,
            "live_state_changed": False,
            "deployment_changed": False,
        },
        "outputs": {
            "training_health": str((OUTPUT_ROOT / "training_health.json").resolve()),
            "full_label_vintage_metrics": str(
                (OUTPUT_ROOT / "full_label_vintage_metrics.csv").resolve()
            ),
            "model_plan_account_metrics": str(
                (OUTPUT_ROOT / "model_plan_account_metrics.csv").resolve()
            ),
            "policy_alpha_daily": str((OUTPUT_ROOT / "policy_alpha_daily.csv").resolve()),
            "policy_alpha_metrics": str((OUTPUT_ROOT / "policy_alpha_metrics.csv").resolve()),
            "policy_alpha_contract": str(
                (OUTPUT_ROOT / "policy_alpha_contract_v3.json").resolve()
            ),
            "historical_policy_alpha_daily": str(
                (OUTPUT_ROOT / "historical_policy_alpha_daily.csv").resolve()
            ),
            "historical_policy_alpha_metrics": str(
                (OUTPUT_ROOT / "historical_policy_alpha_metrics.csv").resolve()
            ),
            "historical_policy_selection_summary": str(
                (OUTPUT_ROOT / "historical_policy_selection_summary.json").resolve()
            ),
            "historical_policy_alpha_contract": str(
                (OUTPUT_ROOT / "historical_policy_alpha_contract_v2.json").resolve()
            ),
            "d7_vintage_metrics": str((OUTPUT_ROOT / "d7_vintage_metrics.csv").resolve()),
            "historical_model_plan_annual": str(
                (OUTPUT_ROOT / "historical_model_plan_annual.csv").resolve()
            ),
            "historical_fold_metrics": str(
                (OUTPUT_ROOT / "historical_fold_metrics.csv").resolve()
            ),
            "historical_d7_account_annual": str(
                (OUTPUT_ROOT / "historical_d7_account_annual.csv").resolve()
            ),
            "historical_rolling_account_annual": str(
                (OUTPUT_ROOT / "historical_rolling_account_annual.csv").resolve()
            ),
            "checkpoint_pairwise_metrics": str(
                (OUTPUT_ROOT / "checkpoint_pairwise_metrics.csv").resolve()
            ),
        },
    }
    _write_json(OUTPUT_ROOT / "comparison_summary.json", summary)
    (OUTPUT_ROOT / "comparison_summary.md").write_text(
        _summary_markdown(
            health=health,
            full=full,
            historical_fold=historical_fold,
            historical_policy_alpha=historical_policy_alpha,
            policy_alpha=policy_alpha,
            model_plan=model_plan,
            historical_yearly=historical_yearly,
            historical_d7=historical_d7,
            historical_rolling=historical_rolling,
        ),
        encoding="utf-8",
    )
    report_path = build_report_artifact(output_root=output_root)
    summary["outputs"]["report_artifact"] = str(report_path.resolve())
    _write_json(OUTPUT_ROOT / "comparison_summary.json", summary)
    verify_vintages(output_root=output_root)
    return summary


def verify_vintages(*, output_root: Path = OUTPUT_ROOT) -> dict[str, Any]:
    if output_root.resolve() != OUTPUT_ROOT.resolve():
        raise ValueError("vintage diagnostics require the canonical output root")
    study.verify_structured_180x35_2026(require_complete=True)
    contract_payload = _read_json(OUTPUT_ROOT / "contract.json")
    if str(contract_payload.get("contract_sha256", "")) != study._canonical_digest(
        contract_payload["contract"]
    ):
        raise ValueError("vintage diagnostic contract digest drifted")
    if _read_json(OUTPUT_ROOT / "policy_alpha_contract_v3.json") != _policy_alpha_contract():
        raise ValueError("2026 unit-time policy-alpha contract drifted")
    if _read_json(
        OUTPUT_ROOT / "historical_policy_alpha_contract_v2.json"
    ) != _historical_policy_alpha_contract():
        raise ValueError("historical unit-time policy-alpha contract drifted")
    health = _read_json(OUTPUT_ROOT / "training_health.json")
    if str(health.get("status")) != "healthy":
        raise ValueError("2026 checkpoint training health verification failed")
    forecasts = {
        year: pd.read_parquet(
            _extended_prediction_path(year),
            columns=["trade_date", "date_idx", "symbol_idx", "score", "predicted_exit_day"],
        )
        for year in VINTAGES
    }
    reference = forecasts[2026][["date_idx", "symbol_idx"]]
    for year, frame in forecasts.items():
        if (
            len(frame) != EXPECTED_EXTENDED_CANDIDATES
            or frame["trade_date"].nunique() != study.EXPECTED_EXTENDED_DATES
            or bool(frame.duplicated(["date_idx", "symbol_idx"]).any())
            or not bool(np.isfinite(frame[["score", "predicted_exit_day"]]).all().all())
            or not frame[["date_idx", "symbol_idx"]].equals(reference)
        ):
            raise ValueError(f"extended forecast verification failed for {year}")
    full = pd.read_csv(OUTPUT_ROOT / "full_label_vintage_metrics.csv")
    if (
        len(full) != 4
        or set(full["checkpoint_vintage"].astype(int)) != set(VINTAGES)
        or not bool(full["candidate_score_coverage"].eq(1.0).all())
        or not bool(full["execution_return_coverage"].eq(1.0).all())
        or not bool(
            full["candidate_count"].astype(int).eq(
                study.EXPECTED_NATIVE_FULL_LABEL_CANDIDATES
            ).all()
        )
    ):
        raise ValueError("full-label four-vintage metrics are incomplete")
    model_plan = pd.read_csv(OUTPUT_ROOT / "model_plan_account_metrics.csv")
    if (
        len(model_plan) != 16
        or set(model_plan["checkpoint_vintage"].astype(int)) != set(VINTAGES)
        or set(model_plan["top_k"].astype(int)) != {1, 3}
        or set(model_plan["cost_scenario"].astype(str)) != set(study.COST_SCENARIOS)
    ):
        raise ValueError("model-plan account coverage is incomplete")
    policy_alpha_daily = pd.read_csv(OUTPUT_ROOT / "policy_alpha_daily.csv")
    policy_alpha = pd.read_csv(OUTPUT_ROOT / "policy_alpha_metrics.csv")
    if (
        len(policy_alpha_daily) != EXPECTED_POLICY_ALPHA_DAILY_ROWS
        or len(policy_alpha)
        != len(VINTAGES)
        * len(POLICY_ALPHA_NAMES)
        * 2
        * len(study.COST_SCENARIOS)
        or set(policy_alpha["policy_name"].astype(str)) != set(POLICY_ALPHA_NAMES)
        or not bool(policy_alpha["signal_date_count"].astype(int).eq(48).all())
        or not {
            "alpha_median",
            "alpha_trimmed_mean_10pct",
            "alpha_q10",
            "alpha_q90",
            "positive_alpha_day",
            "annualized_selected_log_speed",
            "annualized_alpha_log_speed",
            "alpha_log_speed_median",
            "alpha_log_speed_trimmed_mean_10pct",
            "positive_alpha_log_speed_day",
        }.issubset(policy_alpha.columns)
    ):
        raise ValueError("cohort policy-alpha scan coverage is incomplete")
    model_plan_alpha = policy_alpha[
        policy_alpha["policy_name"].astype(str).eq("model_plan")
        & policy_alpha["cost_scenario"].astype(str).eq("double_slippage")
    ]
    for row in full.itertuples(index=False):
        for top_k in (1, 3):
            observed = model_plan_alpha[
                model_plan_alpha["checkpoint_vintage"].astype(int).eq(
                    int(row.checkpoint_vintage)
                )
                & model_plan_alpha["top_k"].astype(int).eq(top_k)
            ]["alpha_mean_net_return"]
            expected = float(getattr(row, f"top{top_k}_stress_alpha"))
            if len(observed) != 1 or not math.isclose(
                float(observed.iloc[0]), expected, rel_tol=0.0, abs_tol=1.0e-4
            ):
                raise ValueError(
                    f"model-plan cohort alpha does not reproduce Top{top_k} for "
                    f"checkpoint {int(row.checkpoint_vintage)}"
                )
    d7 = pd.read_csv(OUTPUT_ROOT / "d7_vintage_metrics.csv")
    if (
        len(d7) != 32
        or set(d7["checkpoint_vintage"].astype(int)) != set(VINTAGES)
        or set(d7["window_name"].astype(str)) != {"strict_101", "extended_121"}
        or not bool(
            d7[d7["window_name"].astype(str).eq("strict_101")][
                "all_positions_resolved"
            ].astype(bool).all()
        )
    ):
        raise ValueError("D7 four-vintage account coverage is incomplete")
    if not (OUTPUT_ROOT / "comparison_summary.json").is_file():
        raise FileNotFoundError(OUTPUT_ROOT / "comparison_summary.json")
    if not (OUTPUT_ROOT / "report_artifact.json").is_file():
        raise FileNotFoundError(OUTPUT_ROOT / "report_artifact.json")
    historical = pd.read_csv(OUTPUT_ROOT / "historical_fold_metrics.csv")
    historical_d7 = pd.read_csv(OUTPUT_ROOT / "historical_d7_account_annual.csv")
    historical_rolling = pd.read_csv(
        OUTPUT_ROOT / "historical_rolling_account_annual.csv"
    )
    historical_policy_daily = pd.read_csv(
        OUTPUT_ROOT / "historical_policy_alpha_daily.csv"
    )
    historical_policy = pd.read_csv(
        OUTPUT_ROOT / "historical_policy_alpha_metrics.csv"
    )
    if (
        len(historical) != 3
        or set(historical["development_year"].astype(int)) != {2023, 2024, 2025}
        or len(historical_d7) != 12
        or len(historical_rolling) != 12
    ):
        raise ValueError("historical 2023-2025 detail is incomplete")
    expected_historical_metric_rows = (
        len(HISTORICAL_YEARS)
        * len(HISTORICAL_POLICY_ALPHA_NAMES)
        * 2
        * len(study.COST_SCENARIOS)
    )
    if (
        len(historical_policy_daily)
        != EXPECTED_HISTORICAL_POLICY_ALPHA_DAILY_ROWS
        or len(historical_policy) != expected_historical_metric_rows
        or set(historical_policy["evaluation_year"].astype(int))
        != set(HISTORICAL_YEARS)
        or set(historical_policy["policy_name"].astype(str))
        != set(HISTORICAL_POLICY_ALPHA_NAMES)
        or not {
            "alpha_log_speed_median",
            "alpha_log_speed_trimmed_mean_10pct",
            "positive_alpha_log_speed_day",
        }.issubset(historical_policy.columns)
        or not (OUTPUT_ROOT / "historical_policy_selection_summary.json").is_file()
    ):
        raise ValueError("historical policy-alpha coverage is incomplete")
    for year, expected_count in HISTORICAL_POLICY_ALPHA_DATE_COUNTS.items():
        observed_counts = historical_policy.loc[
            historical_policy["evaluation_year"].astype(int).eq(year),
            "signal_date_count",
        ].astype(int)
        if observed_counts.empty or not bool(observed_counts.eq(expected_count).all()):
            raise ValueError(f"historical policy-alpha date coverage drifted for {year}")
    historical_model_plan_alpha = historical_policy[
        historical_policy["policy_name"].astype(str).eq("model_plan")
        & historical_policy["cost_scenario"].astype(str).eq("double_slippage")
    ]
    for row in historical.itertuples(index=False):
        for top_k in (1, 3):
            observed = historical_model_plan_alpha[
                historical_model_plan_alpha["evaluation_year"].astype(int).eq(
                    int(row.development_year)
                )
                & historical_model_plan_alpha["top_k"].astype(int).eq(top_k)
            ]["alpha_mean_net_return"]
            expected = float(getattr(row, f"top{top_k}_stress_alpha"))
            if len(observed) != 1 or not math.isclose(
                float(observed.iloc[0]), expected, rel_tol=0.0, abs_tol=1.0e-4
            ):
                raise ValueError(
                    f"historical model-plan alpha does not reproduce Top{top_k} "
                    f"for {int(row.development_year)}"
                )
    return {
        "status": "ok",
        "checkpoint_vintages": list(VINTAGES),
        "full_label_rows": int(len(full)),
        "extended_candidate_rows_per_vintage": int(len(reference)),
        "model_plan_account_rows": int(len(model_plan)),
        "policy_alpha_daily_rows": int(len(policy_alpha_daily)),
        "policy_alpha_rows": int(len(policy_alpha)),
        "historical_policy_alpha_daily_rows": int(len(historical_policy_daily)),
        "historical_policy_alpha_rows": int(len(historical_policy)),
        "d7_account_rows": int(len(d7)),
        "score_coverage_complete": True,
        "candidate_keys_identical": True,
        "training_health": "healthy",
        "qdp_changed": False,
        "provider_called": False,
        "checkpoint_changed": False,
        "live_state_changed": False,
    }
