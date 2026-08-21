from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from quantlab.core.io import stable_hash

from .data import (
    OUTPUT_ROOT,
    ResearchDataError,
    cutoff_audit_fields,
    cutoff_violation_count,
    daily_rank_metrics,
    write_json,
)
from .portfolio import newey_west_interval
from .sequence import load_fold_results as load_sequence_fold_results
from .tree import evaluate_predictions
from .tree import load_fold_results as load_tree_fold_results

RUN_NAME = "ensemble_tree158_raw60"
COMPONENT_RUNS = ("tree_158", "sequence_raw60")
ACCOUNT_TASKS = (
    "top10__stress",
    "top10__stress__cap10pct",
    "top10__stress__no_overlap",
    "top10__stress__cap10pct__no_overlap",
)


def _paired_increments(
    ensemble_result: dict[str, Any],
    component_fingerprints: dict[str, list[str]],
) -> dict[str, Any]:
    ensemble_daily = pd.read_parquet(
        Path(ensemble_result["files"]["daily_metrics"]["path"]),
        columns=["date_idx", "rank_ic"],
    ).rename(columns={"rank_ic": "ensemble"})
    ensemble_selections = pd.read_parquet(
        Path(ensemble_result["files"]["top10_selections"]["path"]),
        columns=["date_idx", "selection_rank", "actual"],
    )
    output: dict[str, Any] = {}
    for component in COMPONENT_RUNS:
        path = OUTPUT_ROOT / component / "evaluation/result.json"
        if not path.is_file():
            raise ResearchDataError(f"component evaluation is missing: {path}")
        result = dict(json.loads(path.read_text(encoding="utf-8")))
        if (
            result.get("status") != "completed"
            or cutoff_violation_count(result) != 0
            or result.get("fold_fingerprints") != component_fingerprints[component]
            or result.get("input_fingerprint") != ensemble_result["input_fingerprint"]
            or result.get("target_fingerprint") != ensemble_result["target_fingerprint"]
        ):
            raise ResearchDataError(f"component evaluation is stale: {path}")
        component_daily = pd.read_parquet(
            Path(result["files"]["daily_metrics"]["path"]),
            columns=["date_idx", "rank_ic"],
        ).rename(columns={"rank_ic": "component"})
        joined = component_daily.merge(
            ensemble_daily, on="date_idx", validate="one_to_one"
        )
        rank_increment = newey_west_interval(
            (joined["ensemble"] - joined["component"]).to_numpy(dtype=np.float64),
            lag=20,
        )
        component_selections = pd.read_parquet(
            Path(result["files"]["top10_selections"]["path"]),
            columns=["date_idx", "selection_rank", "actual"],
        )
        topk: dict[str, Any] = {}
        for top_k in (1, 3, 5, 10):
            component_actual = (
                component_selections.loc[
                    component_selections["selection_rank"] <= top_k
                ]
                .groupby("date_idx")["actual"]
                .mean()
                .rename("component")
            )
            ensemble_actual = (
                ensemble_selections.loc[ensemble_selections["selection_rank"] <= top_k]
                .groupby("date_idx")["actual"]
                .mean()
                .rename("ensemble")
            )
            paired = pd.concat(
                [component_actual, ensemble_actual], axis=1, join="inner"
            ).dropna()
            topk[str(top_k)] = newey_west_interval(
                (paired["ensemble"] - paired["component"]).to_numpy(dtype=np.float64),
                lag=20,
            )
        account: dict[str, Any] = {}
        for task_id in ACCOUNT_TASKS:
            component_equity = pd.read_parquet(
                OUTPUT_ROOT / component / "evaluation" / task_id / "equity.parquet",
                columns=["date_idx", "daily_net_return"],
            ).rename(columns={"daily_net_return": "component"})
            ensemble_equity = pd.read_parquet(
                OUTPUT_ROOT / RUN_NAME / "evaluation" / task_id / "equity.parquet",
                columns=["date_idx", "daily_net_return"],
            ).rename(columns={"daily_net_return": "ensemble"})
            paired = component_equity.merge(
                ensemble_equity, on="date_idx", validate="one_to_one"
            )
            account[task_id] = newey_west_interval(
                (paired["ensemble"] - paired["component"]).to_numpy(dtype=np.float64),
                lag=20,
            )
        output[component] = {
            "daily_rank_ic": rank_increment,
            "daily_topk_exact_net_return": topk,
            "account_daily_net_return": account,
        }
    return output


def evaluate() -> dict[str, Any]:
    tree_results, _ = load_tree_fold_results(158)
    sequence_results, _ = load_sequence_fold_results()
    fold_results: list[dict[str, Any]] = []
    predictions: list[pd.DataFrame] = []
    component_correlations: list[dict[str, Any]] = []
    run_root = OUTPUT_ROOT / RUN_NAME
    for fold in range(1, 6):
        tree_result = tree_results[fold - 1]
        sequence_result = sequence_results[fold - 1]
        tree = pd.read_parquet(Path(tree_result["files"]["predictions"]["path"]))
        sequence = pd.read_parquet(
            Path(sequence_result["files"]["predictions"]["path"]),
            columns=["row_position", "score"],
        ).rename(columns={"score": "sequence_score"})
        merged = tree.merge(sequence, on="row_position", validate="one_to_one")
        merged["tree_rank"] = merged.groupby("date_idx", sort=False)["score"].rank(
            pct=True
        )
        merged["sequence_rank"] = merged.groupby("date_idx", sort=False)[
            "sequence_score"
        ].rank(pct=True)
        merged["score"] = 0.5 * merged["tree_rank"] + 0.5 * merged["sequence_rank"]
        valid = merged["target_valid"].astype(bool) & np.isfinite(merged["actual"])
        daily, metrics = daily_rank_metrics(
            dates=merged.loc[valid, "date_idx"].to_numpy(dtype=np.int32),
            actual=merged.loc[valid, "actual"].to_numpy(dtype=np.float32),
            prediction=merged.loc[valid, "score"].to_numpy(dtype=np.float32),
        )
        correlation = merged.groupby("date_idx")[["tree_rank", "sequence_rank"]].corr()
        correlation = correlation.iloc[0::2, -1]
        component_correlations.append(
            {
                "fold": fold,
                "mean_daily_rank_correlation": float(correlation.mean()),
            }
        )
        fold_root = run_root / f"fold_{fold:02d}"
        fold_root.mkdir(parents=True, exist_ok=True)
        prediction_path = fold_root / "predictions.parquet"
        daily_path = fold_root / "daily_metrics.parquet"
        keep = [
            "row_position",
            "candidate_id",
            "date_idx",
            "trade_date",
            "symbol",
            "symbol_idx",
            "fold",
            "score",
            "actual",
            "target_valid",
        ]
        merged[keep].to_parquet(prediction_path, index=False)
        daily.to_parquet(daily_path, index=False)
        result = {
            "status": "completed",
            "model_family": "equal_rank_ensemble",
            "fingerprint": stable_hash(
                {
                    "pipeline": "equal_rank_ensemble",
                    "fold": fold,
                    "components": [
                        tree_result["fingerprint"],
                        sequence_result["fingerprint"],
                    ],
                    "weights": [0.5, 0.5],
                }
            ),
            "fold": tree_result["fold"],
            "metrics": metrics,
            "components": ["tree_158", "sequence_raw60"],
            "weights": [0.5, 0.5],
            "component_rank_correlation": component_correlations[-1],
            **cutoff_audit_fields(),
            "files": {
                "predictions": {"path": str(prediction_path)},
                "daily_metrics": {"path": str(daily_path)},
            },
        }
        write_json(fold_root / "result.json", result)
        fold_results.append(result)
        predictions.append(merged[keep])
    oof = pd.concat(predictions, ignore_index=True).sort_values(
        ["date_idx", "symbol_idx"], kind="stable"
    )
    result = evaluate_predictions(
        run_name=RUN_NAME,
        fold_results=fold_results,
        oof=oof,
        metadata={
            "model_family": "equal_rank_ensemble",
            "components": ["tree_158", "sequence_raw60"],
            "weights": [0.5, 0.5],
            "component_rank_correlations": component_correlations,
        },
    )
    result["paired_increments_vs_components"] = _paired_increments(
        result,
        {
            "tree_158": [item["fingerprint"] for item in tree_results],
            "sequence_raw60": [item["fingerprint"] for item in sequence_results],
        },
    )
    write_json(OUTPUT_ROOT / RUN_NAME / "evaluation/result.json", result)
    return result
