from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from daily_research.path_policy import seq100_hot_path_state as state


def test_state_contract_keeps_path_distribution_primary() -> None:
    study = state.load_study()
    assert study["target"]["primary"].startswith("three-class")
    assert study["target"]["single_profit_label"] is False
    assert study["evaluation"]["portfolio_selection_performed"] is False
    assert "minute_range" not in study["feature_sets"]["intraday_state"]
    assert "bar_range" in study["feature_sets"]["daily_bar_state"]


def test_multiclass_brier_is_zero_for_perfect_probability() -> None:
    target = np.array([0, 1, 2])
    probability = np.eye(3)
    assert state._multiclass_brier(target, probability) == pytest.approx(0.0)


def test_prediction_evaluation_rewards_information() -> None:
    frame = pd.DataFrame(
        {
            "symbol": ["A", "B", "C", "D", "E", "F"],
            "trade_date": [
                "2019-01-02",
                "2019-01-02",
                "2020-01-02",
                "2020-01-02",
                "2020-01-03",
                "2020-01-03",
            ],
            "date_idx": [1, 1, 2, 2, 3, 3],
            "signal_year": [2019, 2019, 2020, 2020, 2020, 2020],
            "path_cluster": [0, 1, 2, 0, 1, 2],
            "terminal_log_return_5": np.zeros(6),
            "terminal_log_return_20": np.zeros(6),
        }
    )
    target = frame["path_cluster"].to_numpy()
    probability = np.eye(3)[target] * 0.98 + 0.02 / 3.0
    annual, pooled, predictions = state._evaluate_predictions(
        frame,
        target,
        probability,
        np.array([1 / 3, 1 / 3, 1 / 3]),
        np.full((len(frame), 3), 1 / 3),
        model_name="test",
        feature_set="daily",
    )
    assert annual["log_loss_gain"].gt(0).all()
    assert pooled.iloc[0]["log_loss_gain_date_mean"] > 0
    assert predictions["row_log_loss_gain"].gt(0).all()


def test_safe_binary_auc_is_missing_when_only_one_class_is_present() -> None:
    result = state._safe_binary_auc(np.array([False, False]), np.array([0.1, 0.2]))
    assert np.isnan(result)


def test_causal_prior_excludes_labels_inside_delay_window() -> None:
    frame = pd.DataFrame(
        {
            "date_idx": [1, 1, 2, 2, 3, 3, 4, 4],
            "path_cluster": [0, 0, 1, 1, 2, 2, 2, 2],
        }
    )
    target = frame["path_cluster"].to_numpy()
    evaluation = pd.DataFrame({"date_idx": [4]})
    probability = state._causal_rolling_prior(
        frame,
        target,
        evaluation,
        np.array([1 / 3, 1 / 3, 1 / 3]),
        lookback_days=10,
        label_delay_days=2,
        smoothing=0.0,
    )
    assert probability[0].tolist() == pytest.approx([0.5, 0.5, 0.0])


def test_top_decile_ranks_each_model_independently() -> None:
    rows = []
    for model in ("m1", "m2"):
        for index in range(20):
            rows.append(
                {
                    "model": model,
                    "feature_set": "daily",
                    "signal_year": 2019,
                    "trade_date": "2019-01-02",
                    "path_cluster": 2 if index >= 18 else 0,
                    "terminal_log_return_5": 0.0,
                    "terminal_log_return_20": 0.0,
                    "probability_cluster_2": float(index),
                }
            )
    result = state._top_decile_diagnostics(pd.DataFrame(rows))
    assert len(result) == 2
    assert result["top_rows"].eq(2).all()
    assert result["top_rising_path_rate"].eq(1.0).all()
