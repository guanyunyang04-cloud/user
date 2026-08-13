from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd

from daily_research.path_policy import (
    seq100_multi_horizon_distribution_incremental as study,
)


def test_incremental_contract_has_only_equal_component_fusions() -> None:
    contract = study.load_study()

    assert tuple(contract["incremental_fusions"]) == study.FUSION_VARIANTS
    assert contract["portfolio_contract"]["weight_search_allowed"] is False
    assert contract["portfolio_contract"]["quantile_overlay_allowed"] is False


def test_quantile_metrics_are_date_equal_and_exact_when_predictions_match() -> None:
    dates = np.asarray([1, 1, 1, 2])
    actual = np.asarray([1.0, 1.0, 1.0, 2.0])
    prediction = np.column_stack([actual, actual, actual])

    metrics = study._quantile_metrics(
        dates=dates,
        actual=actual,
        quantile_prediction=prediction,
        valid=np.ones(4, dtype=bool),
    )

    assert metrics["date_count"] == 2
    assert metrics["mean_pinball"] == 0.0
    assert metrics["coverage_q10"] == 1.0
    assert metrics["mean_q90_minus_q10"] == 0.0


def test_binary_skill_compares_model_with_training_prior() -> None:
    dates = np.asarray([1, 1, 2, 2])
    actual = np.asarray([1.0, 0.0, 1.0, 0.0])
    model_probability = np.asarray([0.9, 0.1, 0.9, 0.1])
    baseline_probability = np.full(4, 0.5)

    metrics = study._binary_metrics(
        dates=dates,
        actual=actual,
        probability=model_probability,
        baseline_probability=baseline_probability,
        valid=np.ones(4, dtype=bool),
    )

    assert metrics["model_brier"] < metrics["training_prior_brier"]
    assert metrics["brier_skill"] > 0.0


def test_training_prefix_references_use_the_base_train_split() -> None:
    row_index = pd.DataFrame(
        {
            "date_idx": [0, 0, 1, 1, 2, 2],
            "symbol_idx": [0, 1, 0, 1, 0, 1],
        }
    )
    primary = SimpleNamespace(
        context=SimpleNamespace(row_index=row_index),
        entry_filled=np.ones((3, 2), dtype=bool),
    )
    horizon_source = SimpleNamespace(
        exact_values=np.asarray([[0.01], [0.02], [0.03], [0.04], [0.05], [0.06]]),
        exact_valid=np.ones((6, 1), dtype=bool),
        base_column=0,
        fill_days=np.full((6, 1), 10, dtype=np.int16),
        horizon_column=0,
        path_valid=np.ones((6, 1), dtype=bool),
        gross_column=0,
    )
    sources = SimpleNamespace(
        primary=primary,
        by_horizon={5: horizon_source, 10: horizon_source},
    )
    folds = [
        {
            "fold": 1,
            "training_maximum_date_idx": 1,
            "validation_start_date_idx": 2,
            "validation_end_date_idx": 2,
        }
    ]

    references = study._training_prefix_references(sources=sources, folds=folds)

    assert references[1][5]["return_valid_count"] == 4
    assert references[1][10]["entry_count"] == 4
