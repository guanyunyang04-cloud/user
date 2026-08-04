from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from daily_research.path_policy import seq100_quality_liquidity_model as model


def test_close_d1_plan_is_isolated_and_exactly_three_tasks() -> None:
    tasks = model._close_d1_task_plan()

    assert len(tasks) == model.CLOSE_D1_TASK_COUNT == 3
    assert {task["year"] for task in tasks} == {2023, 2024, 2025}
    assert {task["horizon"] for task in tasks} == {1}
    assert {task["target"] for task in tasks} == {model.CLOSE_D1_TARGET}
    assert {task["variant"] for task in tasks} == {model.COMPACT_VARIANT}
    assert all(task["stage"] == model.CLOSE_D1_TASK_STAGE for task in tasks)
    assert len(model._direct_return_task_plan()) == model.RETURN_TASK_COUNT == 15


def test_vwap_validation_preserves_missing_and_accepts_rounding_tolerance() -> None:
    frame = pd.DataFrame(
        {
            "first_count": [1, 1, 0],
            "last_count": [1, 1, 0],
            "first_open": [10.0, 10.0, np.nan],
            "first_high": [11.0, 10.0, np.nan],
            "first_low": [9.0, 10.0, np.nan],
            "first_volume": [100.0, 100.0, np.nan],
            "first_amount": [1000.0, 1001.0, np.nan],
            "last_close": [11.0, 11.0, np.nan],
        }
    )

    validated, summary = model._validate_vwap_execution_frame(
        frame,
        existing_open_to_close=np.asarray([0.10, 0.10, np.nan]),
    )

    assert validated["outcome_valid"].tolist() == [True, True, False]
    assert validated["vwap_strictly_inside_ohlc"].tolist() == [True, False, False]
    assert validated["source_state"].tolist() == [
        "observed",
        "observed",
        "missing_or_duplicate_bar",
    ]
    assert validated.loc[0, "first_5m_vwap_to_close_return"] == pytest.approx(0.10)
    assert np.isnan(validated.loc[2, "first_5m_vwap_to_close_return"])
    assert summary["valid_count"] == 2
    assert not summary["source_validation_passed"]


def test_paired_model_delta_is_close_trained_minus_open_trained() -> None:
    annual = pd.DataFrame(
        [
            {
                "model_training_target": training_target,
                "outcome": "next_open_to_close",
                "evaluation_year": 2023,
                "rank_ic": rank_ic,
                "top1_return_mean": 0.01,
                "top5_return_mean": 0.01,
                "top1_excess_mean": 0.01,
                "top5_excess_mean": 0.01,
                "top1_capture": 0.10,
                "top5_capture": 0.10,
                "decile_spearman": 0.10,
                "zscore_mse": 1.0,
            }
            for training_target, rank_ic in (
                ("close_to_close", 0.03),
                ("next_open_to_close", 0.02),
            )
        ]
    )

    paired = model._paired_model_deltas(annual)
    rank_ic = paired.loc[paired["metric"].eq("rank_ic")].iloc[0]

    assert rank_ic["close_minus_open"] == pytest.approx(0.01)
