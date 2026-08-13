"""Incremental fusion and calibration audit for the frozen D5/D10 study."""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from daily_research.path_policy import seq100_full_market_multitask_forecast as base
from daily_research.path_policy import seq100_full_market_sequence_challenger as seq
from daily_research.path_policy import seq100_multi_horizon_distribution as model

SCHEMA = "seq100_multi_horizon_distribution_incremental/1"
DEFAULT_STUDY_PATH = (
    base.WORKSPACE_ROOT
    / "daily_research/studies/seq100_multi_horizon_distribution_incremental_v1.json"
)
FUSION_VARIANTS = ("add_single_d10_point", "add_multi_d10_point")


class IncrementalAuditError(RuntimeError):
    """Raised when an incremental-audit contract is violated."""


def _workspace_path(value: str) -> Path:
    return (base.WORKSPACE_ROOT / value).resolve()


def load_study(path: Path = DEFAULT_STUDY_PATH) -> dict[str, Any]:
    study = json.loads(path.read_text(encoding="utf-8"))
    if study.get("study_id") != "seq100_multi_horizon_distribution_incremental_v1":
        raise IncrementalAuditError("study_id_changed")
    if tuple(study["incremental_fusions"]) != FUSION_VARIANTS:
        raise IncrementalAuditError("fusion_variants_changed")
    if study["portfolio_contract"] != {
        "horizon": 10,
        "top_k": 10,
        "cohort_equity_fraction": 0.1,
        "cost_scenario": "stress",
        "primary_overlap_policy": "no overlapping same-symbol cohorts",
        "winner_caps": [0.05, 0.1],
        "weight_search_allowed": False,
        "threshold_search_allowed": False,
        "quantile_overlay_allowed": False,
    }:
        raise IncrementalAuditError("portfolio_contract_changed")
    if not bool(study["epistemic_contract"]["no_2026_outcome_may_be_read"]):
        raise IncrementalAuditError("forbidden_outcome_boundary_changed")
    return study


def _model_context(
    study: Mapping[str, Any],
) -> tuple[
    dict[str, Any],
    model.MultiHorizonSources,
    Path,
    list[dict[str, Any]],
]:
    model_study_path = _workspace_path(str(study["sources"]["model_study"]))
    model_study = model.load_study(model_study_path)
    sources = model._load_sources(model_study)
    model_output_root = _workspace_path(str(study["sources"]["model_output_root"]))
    return model_study, sources, model_output_root, model._forward_folds(sources)


def _fold_prediction_frame(
    *,
    sources: model.MultiHorizonSources,
    folds: list[dict[str, Any]],
    output_root: Path,
    variant: str,
) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    horizons, distribution_heads = model._variant_contract(sources.study, variant)
    for fold in folds:
        fold_number = int(fold["fold"])
        task, _ = model._load_task(output_root, variant=variant, fold=fold_number)
        positions = base._fold_rows(
            sources.primary.context.row_index, fold, "validation"
        )
        frame = pd.DataFrame(
            {
                "model_row_position": positions,
                "fold": fold_number,
            }
        )
        for horizon in horizons:
            frame[f"{variant}__point_d{horizon}"] = model._load_prediction_array(
                task, f"point_d{horizon}"
            ).astype(np.float32)
            frame[f"{variant}__exit_probability_d{horizon}"] = (
                model._load_prediction_array(
                    task, f"exit_probability_d{horizon}"
                ).astype(np.float32)
            )
            if distribution_heads:
                quantiles = model._load_prediction_array(
                    task, f"quantiles_d{horizon}"
                ).astype(np.float32)
                for column, quantile in enumerate((10, 50, 90)):
                    frame[f"{variant}__q{quantile}_d{horizon}"] = quantiles[:, column]
        frame[f"{variant}__entry_probability"] = model._load_prediction_array(
            task, "entry_probability"
        ).astype(np.float32)
        parts.append(frame)
    return pd.concat(parts, ignore_index=True)


def _aligned_predictions(
    study: Mapping[str, Any],
    sources: model.MultiHorizonSources,
    model_output_root: Path,
    folds: list[dict[str, Any]],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    current_manifest_path = _workspace_path(
        str(study["sources"]["current_core_evaluation"])
    )
    current_manifest = json.loads(current_manifest_path.read_text(encoding="utf-8"))
    if (
        current_manifest.get("status") != "completed"
        or current_manifest.get("forbidden_2026_read_count") != 0
        or current_manifest.get("tree_feature_variant") != "price_path_core_183"
    ):
        raise IncrementalAuditError("current_core_evaluation_invalid")
    current = pd.read_parquet(
        current_manifest["files"]["component_predictions"]["path"]
    )[
        [
            "candidate_id",
            "date_idx",
            "model_row_position",
            "sequence_rank",
            "tree_rank",
            "ensemble_score",
        ]
    ]
    single = _fold_prediction_frame(
        sources=sources,
        folds=folds,
        output_root=model_output_root,
        variant="single_d10_point",
    )
    multi = _fold_prediction_frame(
        sources=sources,
        folds=folds,
        output_root=model_output_root,
        variant="multi_d5_d10_distribution",
    )
    aligned = current.merge(
        single, on="model_row_position", how="inner", validate="one_to_one"
    ).merge(
        multi,
        on=["model_row_position", "fold"],
        how="inner",
        validate="one_to_one",
    )
    if len(aligned) != len(current):
        raise IncrementalAuditError("prediction_alignment_dropped_rows")
    aligned = aligned.sort_values(
        ["date_idx", "model_row_position"], kind="stable"
    ).reset_index(drop=True)
    for variant in ("single_d10_point", "multi_d5_d10_distribution"):
        point = f"{variant}__point_d10"
        rank = f"{variant}__rank_d10"
        aligned[rank] = aligned.groupby("date_idx", sort=False)[point].rank(
            method="average", pct=True
        )
    aligned["add_single_d10_point"] = aligned[
        ["sequence_rank", "tree_rank", "single_d10_point__rank_d10"]
    ].mean(axis=1)
    aligned["add_multi_d10_point"] = aligned[
        ["sequence_rank", "tree_rank", "multi_d5_d10_distribution__rank_d10"]
    ].mean(axis=1)
    return aligned, current_manifest


def _rank_diagnostics(frame: pd.DataFrame) -> dict[str, Any]:
    columns = [
        "sequence_rank",
        "tree_rank",
        "ensemble_score",
        "single_d10_point__rank_d10",
        "multi_d5_d10_distribution__rank_d10",
    ]
    daily_correlations: list[pd.DataFrame] = []
    top10_overlap = {
        "single_vs_multi": [],
        "multi_vs_current_core": [],
        "multi_vs_sequence": [],
        "multi_vs_tree": [],
    }
    for _, current in frame.groupby("date_idx", sort=False):
        daily_correlations.append(current[columns].corr(method="pearson"))
        sets = {
            name: set(current.nlargest(10, column)["model_row_position"])
            for name, column in {
                "single": "single_d10_point__rank_d10",
                "multi": "multi_d5_d10_distribution__rank_d10",
                "core": "ensemble_score",
                "sequence": "sequence_rank",
                "tree": "tree_rank",
            }.items()
        }
        top10_overlap["single_vs_multi"].append(
            len(sets["single"] & sets["multi"]) / 10.0
        )
        top10_overlap["multi_vs_current_core"].append(
            len(sets["multi"] & sets["core"]) / 10.0
        )
        top10_overlap["multi_vs_sequence"].append(
            len(sets["multi"] & sets["sequence"]) / 10.0
        )
        top10_overlap["multi_vs_tree"].append(len(sets["multi"] & sets["tree"]) / 10.0)
    mean_correlation = sum(daily_correlations) / len(daily_correlations)
    return {
        "date_count": len(daily_correlations),
        "mean_daily_rank_correlation": {
            row: {
                column: float(mean_correlation.loc[row, column]) for column in columns
            }
            for row in columns
        },
        "mean_daily_top10_overlap_fraction": {
            key: float(np.mean(value)) for key, value in top10_overlap.items()
        },
    }


def _quantile_metrics(
    *,
    dates: np.ndarray,
    actual: np.ndarray,
    quantile_prediction: np.ndarray,
    valid: np.ndarray,
) -> dict[str, float | int]:
    quantiles = (0.1, 0.5, 0.9)
    pinballs = {value: [] for value in quantiles}
    coverages = {value: [] for value in quantiles}
    widths: list[float] = []
    for date_idx in np.unique(dates):
        mask = (dates == date_idx) & valid
        if not mask.any():
            continue
        current_actual = actual[mask]
        current_prediction = quantile_prediction[mask]
        for column, quantile in enumerate(quantiles):
            error = current_actual - current_prediction[:, column]
            pinballs[quantile].append(
                float(np.maximum(quantile * error, (quantile - 1.0) * error).mean())
            )
            coverages[quantile].append(
                float((current_actual <= current_prediction[:, column]).mean())
            )
        widths.append(
            float((current_prediction[:, 2] - current_prediction[:, 0]).mean())
        )
    return {
        "date_count": len(widths),
        **{
            f"pinball_q{int(quantile * 100)}": float(np.mean(values))
            for quantile, values in pinballs.items()
        },
        "mean_pinball": float(
            np.mean([np.mean(values) for values in pinballs.values()])
        ),
        **{
            f"coverage_q{int(quantile * 100)}": float(np.mean(values))
            for quantile, values in coverages.items()
        },
        "mean_q90_minus_q10": float(np.mean(widths)),
    }


def _binary_metrics(
    *,
    dates: np.ndarray,
    actual: np.ndarray,
    probability: np.ndarray,
    baseline_probability: np.ndarray,
    valid: np.ndarray,
) -> dict[str, float | int]:
    model_brier: list[float] = []
    baseline_brier: list[float] = []
    observed: list[float] = []
    predicted: list[float] = []
    baseline: list[float] = []
    for date_idx in np.unique(dates):
        mask = (dates == date_idx) & valid
        if not mask.any():
            continue
        model_brier.append(float(np.square(probability[mask] - actual[mask]).mean()))
        baseline_brier.append(
            float(np.square(baseline_probability[mask] - actual[mask]).mean())
        )
        observed.append(float(actual[mask].mean()))
        predicted.append(float(probability[mask].mean()))
        baseline.append(float(baseline_probability[mask].mean()))
    model_value = float(np.mean(model_brier))
    baseline_value = float(np.mean(baseline_brier))
    return {
        "date_count": len(model_brier),
        "model_brier": model_value,
        "training_prior_brier": baseline_value,
        "brier_skill": float(1.0 - model_value / baseline_value)
        if baseline_value > 0.0
        else math.nan,
        "mean_observed_rate": float(np.mean(observed)),
        "mean_predicted_rate": float(np.mean(predicted)),
        "mean_training_prior": float(np.mean(baseline)),
    }


def _training_prefix_references(
    *,
    sources: model.MultiHorizonSources,
    folds: list[dict[str, Any]],
) -> dict[int, dict[int, dict[str, Any]]]:
    references: dict[int, dict[int, dict[str, Any]]] = {}
    row_index = sources.primary.context.row_index
    for fold in folds:
        fold_number = int(fold["fold"])
        train_rows = base._fold_rows(row_index, fold, "train")
        dates = row_index.iloc[train_rows]["date_idx"].to_numpy(dtype=np.int32)
        symbols = row_index.iloc[train_rows]["symbol_idx"].to_numpy(dtype=np.int32)
        entry = np.asarray(
            sources.primary.entry_filled[dates, symbols], dtype=np.float64
        )
        references[fold_number] = {}
        for horizon in (5, 10):
            source = sources.by_horizon[horizon]
            values = np.asarray(
                source.exact_values[train_rows, source.base_column], dtype=np.float64
            )
            valid = np.asarray(
                source.exact_valid[train_rows, source.base_column], dtype=bool
            ) & np.isfinite(values)
            fill_days = np.asarray(
                source.fill_days[train_rows, source.horizon_column], dtype=np.int16
            )
            path_valid = np.asarray(
                source.path_valid[train_rows, source.gross_column], dtype=bool
            )
            exit_valid = entry.astype(bool) & path_valid & (fill_days >= horizon)
            references[fold_number][horizon] = {
                "quantiles": np.quantile(values[valid], [0.1, 0.5, 0.9]).tolist(),
                "return_valid_count": int(valid.sum()),
                "entry_probability": float(entry.mean()),
                "entry_count": len(entry),
                "exit_block_probability": float(
                    (fill_days[exit_valid] > horizon).mean()
                ),
                "exit_valid_count": int(exit_valid.sum()),
            }
    return references


def calibration_audit(
    *,
    frame: pd.DataFrame,
    sources: model.MultiHorizonSources,
    folds: list[dict[str, Any]],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    row_index = sources.primary.context.row_index.iloc[
        frame["model_row_position"].to_numpy(dtype=np.int64)
    ]
    dates = frame["date_idx"].to_numpy(dtype=np.int32)
    symbols = row_index["symbol_idx"].to_numpy(dtype=np.int32)
    years = pd.to_datetime(row_index["trade_date"]).dt.year.to_numpy(dtype=np.int16)
    folds_array = frame["fold"].to_numpy(dtype=np.int16)
    references = _training_prefix_references(sources=sources, folds=folds)
    quantile_rows: list[dict[str, Any]] = []
    binary_rows: list[dict[str, Any]] = []
    for horizon in (5, 10):
        source = sources.by_horizon[horizon]
        rows = frame["model_row_position"].to_numpy(dtype=np.int64)
        actual = np.asarray(
            source.exact_values[rows, source.base_column], dtype=np.float64
        )
        valid = np.asarray(
            source.exact_valid[rows, source.base_column], dtype=bool
        ) & np.isfinite(actual)
        prediction = frame[
            [
                f"multi_d5_d10_distribution__q10_d{horizon}",
                f"multi_d5_d10_distribution__q50_d{horizon}",
                f"multi_d5_d10_distribution__q90_d{horizon}",
            ]
        ].to_numpy(dtype=np.float64)
        baseline_prediction = np.zeros_like(prediction)
        baseline_entry = np.zeros(len(frame), dtype=np.float64)
        baseline_exit = np.zeros(len(frame), dtype=np.float64)
        for fold_number in np.unique(folds_array):
            mask = folds_array == fold_number
            reference = references[int(fold_number)][horizon]
            baseline_prediction[mask] = np.asarray(reference["quantiles"])
            baseline_entry[mask] = float(reference["entry_probability"])
            baseline_exit[mask] = float(reference["exit_block_probability"])
        point = frame[f"multi_d5_d10_distribution__point_d{horizon}"].to_numpy(
            dtype=np.float64
        )
        point_rank = (
            pd.Series(point)
            .groupby(frame["date_idx"], sort=False)
            .rank(method="average", pct=True)
            .to_numpy(dtype=np.float64)
        )
        top10 = (
            pd.Series(point)
            .groupby(frame["date_idx"], sort=False)
            .rank(method="first", ascending=False)
            .le(10)
            .to_numpy(dtype=bool)
        )
        slices: list[tuple[str, str, np.ndarray]] = [
            ("universe", "all", np.ones(len(frame), dtype=bool)),
            ("point_rank", "top_decile", point_rank >= 0.9),
            ("point_rank", "top10", top10),
        ]
        slices.extend(
            ("fold", str(value), folds_array == value)
            for value in np.unique(folds_array)
        )
        slices.extend(
            ("calendar_year", str(value), years == value) for value in np.unique(years)
        )
        for slice_type, slice_value, slice_mask in slices:
            current_valid = valid & slice_mask
            model_metrics = _quantile_metrics(
                dates=dates,
                actual=actual,
                quantile_prediction=prediction,
                valid=current_valid,
            )
            reference_metrics = _quantile_metrics(
                dates=dates,
                actual=actual,
                quantile_prediction=baseline_prediction,
                valid=current_valid,
            )
            quantile_rows.append(
                {
                    "horizon": horizon,
                    "slice_type": slice_type,
                    "slice_value": slice_value,
                    "valid_row_count": int(current_valid.sum()),
                    **{f"model_{key}": value for key, value in model_metrics.items()},
                    **{
                        f"reference_{key}": value
                        for key, value in reference_metrics.items()
                    },
                    "mean_pinball_skill": float(
                        1.0
                        - float(model_metrics["mean_pinball"])
                        / float(reference_metrics["mean_pinball"])
                    ),
                }
            )
        entry_valid = dates + 1 <= sources.cutoff_date_idx
        entry_actual = np.asarray(source.entry_filled[dates, symbols], dtype=np.float64)
        entry_probability = frame[
            "multi_d5_d10_distribution__entry_probability"
        ].to_numpy(dtype=np.float64)
        fill_days = np.asarray(
            source.fill_days[rows, source.horizon_column], dtype=np.int16
        )
        path_valid = np.asarray(
            source.path_valid[rows, source.gross_column], dtype=bool
        )
        exit_valid = entry_actual.astype(bool) & path_valid & (fill_days >= horizon)
        exit_actual = (fill_days > horizon).astype(np.float64)
        exit_probability = frame[
            f"multi_d5_d10_distribution__exit_probability_d{horizon}"
        ].to_numpy(dtype=np.float64)
        for (
            event,
            current_actual,
            current_probability,
            current_baseline,
            event_valid,
        ) in (
            (
                "entry_fill",
                entry_actual,
                entry_probability,
                baseline_entry,
                entry_valid,
            ),
            (
                "exit_block",
                exit_actual,
                exit_probability,
                baseline_exit,
                exit_valid,
            ),
        ):
            event_slices: list[tuple[str, str, np.ndarray]] = [
                ("universe", "all", np.ones(len(frame), dtype=bool))
            ]
            event_slices.extend(
                ("fold", str(value), folds_array == value)
                for value in np.unique(folds_array)
            )
            event_slices.extend(
                ("calendar_year", str(value), years == value)
                for value in np.unique(years)
            )
            for slice_type, slice_value, slice_mask in event_slices:
                mask = event_valid & slice_mask
                binary_rows.append(
                    {
                        "horizon": horizon,
                        "event": event,
                        "slice_type": slice_type,
                        "slice_value": slice_value,
                        "valid_row_count": int(mask.sum()),
                        **_binary_metrics(
                            dates=dates,
                            actual=current_actual,
                            probability=current_probability,
                            baseline_probability=current_baseline,
                            valid=mask,
                        ),
                    }
                )
    return pd.DataFrame(quantile_rows), pd.DataFrame(binary_rows), references


def _evaluate_fusion(
    *,
    frame: pd.DataFrame,
    sources: model.MultiHorizonSources,
    variant: str,
    output_root: Path,
) -> dict[str, Any]:
    source = sources.by_horizon[10]
    date_values = frame["date_idx"].to_numpy(dtype=np.int32)
    prediction = seq.PredictionOutput(
        rows=frame["model_row_position"].to_numpy(dtype=np.int64),
        stock_score=frame[variant].to_numpy(dtype=np.float32),
        market_return={int(value): 0.0 for value in np.unique(date_values)},
        market_probability={int(value): 0.5 for value in np.unique(date_values)},
    )
    daily_parts: list[pd.DataFrame] = []
    decile_parts: list[pd.DataFrame] = []
    selection_parts: list[pd.DataFrame] = []
    fold_summaries: list[dict[str, Any]] = []
    for fold_number in sorted(frame["fold"].unique()):
        mask = frame["fold"].to_numpy(dtype=np.int16) == int(fold_number)
        fold_prediction = seq.PredictionOutput(
            rows=prediction.rows[mask],
            stock_score=prediction.stock_score[mask],
            market_return={
                key: value
                for key, value in prediction.market_return.items()
                if key in set(date_values[mask])
            },
            market_probability={
                key: value
                for key, value in prediction.market_probability.items()
                if key in set(date_values[mask])
            },
        )
        daily, deciles, selections, summary = seq._build_evaluation_frames(
            source,
            fold_prediction,
            fold=int(fold_number),
            unfilled_as_cash=True,
        )
        daily_parts.append(daily)
        decile_parts.append(deciles)
        selection_parts.append(selections)
        fold_summaries.append({"fold": int(fold_number), **summary})
    daily = pd.concat(daily_parts, ignore_index=True).sort_values("date_idx")
    deciles = pd.concat(decile_parts, ignore_index=True).sort_values(
        ["date_idx", "decile"]
    )
    selections = pd.concat(selection_parts, ignore_index=True).sort_values(
        ["date_idx", "selection_rank"]
    )
    root = output_root / "fusion_evaluation" / variant
    daily_path = root / "daily_metrics.parquet"
    decile_path = root / "decile_daily.parquet"
    selection_path = root / "top10_selections.parquet"
    base._write_parquet(daily, daily_path)
    base._write_parquet(deciles, decile_path)
    base._write_parquet(selections, selection_path)
    return {
        "schema": SCHEMA,
        "status": "completed",
        "completed_at": model._now(),
        "variant": variant,
        "fold_summaries": fold_summaries,
        "combined": base._payoff_summary(daily, deciles),
        "forbidden_2026_read_count": 0,
        "files": {
            "daily_metrics": model._file_record(daily_path, row_count=len(daily)),
            "decile_daily": model._file_record(decile_path, row_count=len(deciles)),
            "top10_selections": model._file_record(
                selection_path, row_count=len(selections)
            ),
        },
    }


def _replay_fusion(
    *,
    evaluation: Mapping[str, Any],
    sources: model.MultiHorizonSources,
    variant: str,
    output_root: Path,
) -> dict[str, Any]:
    source = sources.by_horizon[10]
    selections = pd.read_parquet(evaluation["files"]["top10_selections"]["path"])
    schedule, dropped = seq._prepare_exact_payoff_schedule(
        selections=selections,
        sources=source,
        horizon=10,
        variant=variant,
    )
    specs = seq._exact_payoff_account_specs(variant=variant, horizon=10)
    fingerprint = model._stable_hash(
        {
            "schema": SCHEMA,
            "role": "incremental_equal_component_account",
            "selection_sha256": evaluation["files"]["top10_selections"]["sha256"],
            "variant": variant,
            "specs": specs,
            "starting_cash": 1_000_000.0,
        }
    )
    root = output_root / "fusion_account_replay" / variant
    tasks = seq._run_exact_payoff_account_tasks(
        root=root,
        schema=SCHEMA,
        fingerprint=fingerprint,
        schedule=schedule,
        sources=source,
        specs=specs,
        starting_cash=1_000_000.0,
    )
    summaries = tasks["summaries"]
    return {
        "schema": SCHEMA,
        "status": "completed",
        "completed_at": model._now(),
        "fingerprint": fingerprint,
        "variant": variant,
        "primary_stress_top10_no_overlapping_same_symbol": seq._account_result(
            summaries,
            top_k=10,
            cost_scenario="stress",
            allow_overlapping_same_symbol=False,
        ),
        "stress_top10_no_overlap_cap5": seq._account_result(
            summaries,
            top_k=10,
            cost_scenario="stress",
            cap=0.05,
            allow_overlapping_same_symbol=False,
        ),
        "stress_top10_no_overlap_cap10": seq._account_result(
            summaries,
            top_k=10,
            cost_scenario="stress",
            cap=0.10,
            allow_overlapping_same_symbol=False,
        ),
        "dropped_selection_count_after_gross_join": int(dropped),
        "fixed_definition": "equal within-date rank average of existing sequence, existing 183-field tree, and one new point head",
        "weight_or_threshold_search_performed": False,
        "forbidden_2026_read_count": 0,
        "files": {
            "summary": tasks["summary"],
            "selection_schedule": tasks["selection_schedule"],
            "tasks": tasks["tasks"],
        },
    }


def run(
    *,
    study_path: Path = DEFAULT_STUDY_PATH,
) -> dict[str, Any]:
    study = load_study(study_path)
    _, sources, model_output_root, folds = _model_context(study)
    output_root = _workspace_path(str(study["outputs"]["output_root"]))
    frame, current_evaluation = _aligned_predictions(
        study, sources, model_output_root, folds
    )
    diagnostics = _rank_diagnostics(frame)
    quantile_frame, binary_frame, references = calibration_audit(
        frame=frame, sources=sources, folds=folds
    )
    quantile_path = output_root / "calibration" / "quantile_slices.parquet"
    binary_path = output_root / "calibration" / "binary_slices.parquet"
    base._write_parquet(quantile_frame, quantile_path)
    base._write_parquet(binary_frame, binary_path)
    fusion_outputs: dict[str, Any] = {}
    for variant in FUSION_VARIANTS:
        evaluation = _evaluate_fusion(
            frame=frame,
            sources=sources,
            variant=variant,
            output_root=output_root,
        )
        evaluation_path = output_root / "fusion_evaluation" / variant / "manifest.json"
        model._write_json(evaluation_path, evaluation)
        account = _replay_fusion(
            evaluation=evaluation,
            sources=sources,
            variant=variant,
            output_root=output_root,
        )
        account_path = output_root / "fusion_account_replay" / variant / "manifest.json"
        model._write_json(account_path, account)
        fusion_outputs[variant] = {
            "evaluation": model._file_record(evaluation_path),
            "account": model._file_record(account_path),
            "combined": evaluation["combined"],
            "primary": account["primary_stress_top10_no_overlapping_same_symbol"],
            "cap10": account["stress_top10_no_overlap_cap10"],
            "cap5": account["stress_top10_no_overlap_cap5"],
        }
    current_account_path = _workspace_path(
        str(study["sources"]["current_core_account"])
    )
    current_account = json.loads(current_account_path.read_text(encoding="utf-8"))
    if current_account.get("forbidden_2026_read_count") != 0:
        raise IncrementalAuditError("current_core_account_invalid")
    current_primary = current_account["stress_top10_no_overlapping_same_symbol"]
    current_cap10 = current_account["stress_top10_no_overlap_cap10"]
    current_cap5 = current_account["stress_top10_no_overlap_cap5"]
    comparison = {}
    for variant, current in fusion_outputs.items():
        comparison[variant] = {
            "rank_ic": current["combined"]["daily_rank_ic_mean"],
            "rank_ic_delta_vs_current_core": float(
                current["combined"]["daily_rank_ic_mean"]
                - current_evaluation["combined"]["daily_rank_ic_mean"]
            ),
            "no_overlap_return": current["primary"]["total_net_return"],
            "no_overlap_return_delta_vs_current_core": float(
                current["primary"]["total_net_return"]
                - current_primary["total_net_return"]
            ),
            "no_overlap_drawdown": current["primary"]["maximum_drawdown"],
            "no_overlap_drawdown_delta_vs_current_core": float(
                current["primary"]["maximum_drawdown"]
                - current_primary["maximum_drawdown"]
            ),
            "no_overlap_cap10_return": current["cap10"]["total_net_return"],
            "no_overlap_cap10_delta_vs_current_core": float(
                current["cap10"]["total_net_return"] - current_cap10["total_net_return"]
            ),
            "no_overlap_cap5_return": current["cap5"]["total_net_return"],
            "no_overlap_cap5_delta_vs_current_core": float(
                current["cap5"]["total_net_return"] - current_cap5["total_net_return"]
            ),
        }
    fingerprint = model._stable_hash(
        {
            "schema": SCHEMA,
            "implementation_sha256": base._sha256(Path(__file__)),
            "study_sha256": base._sha256(study_path),
            "current_evaluation_sha256": base._sha256(
                _workspace_path(str(study["sources"]["current_core_evaluation"]))
            ),
            "current_account_sha256": base._sha256(current_account_path),
            "model_comparison_sha256": base._sha256(
                model_output_root / "comparison" / "manifest.json"
            ),
            "fusion_definitions": study["incremental_fusions"],
            "portfolio_contract": study["portfolio_contract"],
        }
    )
    result = {
        "schema": SCHEMA,
        "status": "completed",
        "completed_at": model._now(),
        "study_id": study["study_id"],
        "fingerprint": fingerprint,
        "rank_diagnostics": diagnostics,
        "training_prefix_calibration_references": references,
        "current_core": {
            "rank_ic": current_evaluation["combined"]["daily_rank_ic_mean"],
            "no_overlap_return": current_primary["total_net_return"],
            "no_overlap_drawdown": current_primary["maximum_drawdown"],
            "no_overlap_cap10_return": current_cap10["total_net_return"],
            "no_overlap_cap5_return": current_cap5["total_net_return"],
        },
        "incremental_comparison": comparison,
        "decision_boundary": {
            "equal_component_fusions_only": True,
            "weight_or_threshold_search_performed": False,
            "quantiles_used_for_trading": False,
            "D2_D3_trained": False,
            "standalone_results_seen_before_materialized_audit": True,
            "correlation_diagnostics_seen_before_account_fusion_replay": True,
            "adaptive_development_evidence_not_independent_confirmation": True,
            "stable_profit_claim_allowed": False,
            "forbidden_2026_read_count": 0,
        },
        "forbidden_2026_read_count": 0,
        "files": {
            "quantile_slices": model._file_record(
                quantile_path, row_count=len(quantile_frame)
            ),
            "binary_slices": model._file_record(
                binary_path, row_count=len(binary_frame)
            ),
            "fusion_outputs": {
                key: {
                    "evaluation": value["evaluation"],
                    "account": value["account"],
                }
                for key, value in fusion_outputs.items()
            },
        },
        "sources": {
            "study": model._file_record(study_path),
            "model_comparison": model._file_record(
                model_output_root / "comparison" / "manifest.json"
            ),
            "current_core_evaluation": model._file_record(
                _workspace_path(str(study["sources"]["current_core_evaluation"]))
            ),
            "current_core_account": model._file_record(current_account_path),
        },
    }
    manifest_path = output_root / "manifest.json"
    model._write_json(manifest_path, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the frozen multi-horizon incremental and calibration audit."
    )
    parser.add_argument("--study-path", type=Path, default=DEFAULT_STUDY_PATH)
    args = parser.parse_args()
    print(json.dumps(run(study_path=args.study_path), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
