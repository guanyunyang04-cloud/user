from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from traditional_quant_research.experiments import personal_short_event_model_zoo_research as zoo


def _base_events() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "date": pd.Timestamp("2024-01-02"),
                "entry_date": pd.Timestamp("2024-01-03"),
                "year": 2024,
                "code": "a",
                "target_net_ret_pct": 2.0,
                "executable_entry": True,
                "entry_open_near_limit": False,
                "limit_up_run_ending_today": 1,
                "open_below_kama_break_limitup": True,
                "close_cross_atr_upper": True,
                "event_limit_up_core": True,
                "event_big_up": False,
                "event_new_high_breakout": False,
                "event_kama_breakout": False,
                "event_flag_count": 1,
                "event_strength_score": 2.0,
                "signal_amount_x20": 2.0,
                "signal_vrat5": 3.0,
                "signal_close_position": 0.8,
                "signal_ret20_before_pct": 6.0,
                "volatility_20_pct": 3.0,
                "market_limitup_rate": 0.04,
                "industry_limitup_rate": 0.08,
                "next_open_gap_pct": 1.0,
                "pctChg": 9.8,
            },
            {
                "date": pd.Timestamp("2024-01-02"),
                "entry_date": pd.Timestamp("2024-01-03"),
                "year": 2024,
                "code": "b",
                "target_net_ret_pct": -1.0,
                "executable_entry": True,
                "entry_open_near_limit": False,
                "limit_up_run_ending_today": 2,
                "open_below_kama_break_limitup": False,
                "close_cross_atr_upper": False,
                "event_limit_up_core": False,
                "event_big_up": True,
                "event_new_high_breakout": True,
                "event_kama_breakout": True,
                "event_flag_count": 3,
                "event_strength_score": 2.2,
                "signal_amount_x20": 4.0,
                "signal_vrat5": 2.0,
                "signal_close_position": 0.6,
                "signal_ret20_before_pct": 2.0,
                "volatility_20_pct": 4.0,
                "market_limitup_rate": 0.03,
                "industry_limitup_rate": 0.05,
                "next_open_gap_pct": -0.5,
                "pctChg": 6.0,
            },
        ]
    )


def test_model_zoo_feature_audit_rejects_future_labels() -> None:
    with pytest.raises(ValueError, match="leakage"):
        zoo.audit_model_zoo_feature_columns(["signal_amount_x20", "sell1_close_ret_pct"])

    with pytest.raises(ValueError, match="leakage"):
        zoo.audit_model_zoo_feature_columns(["target_net_ret_pct"])


def test_add_expanded_event_features_builds_ranks_and_interactions() -> None:
    enriched = zoo.add_expanded_event_features(_base_events())

    assert enriched.loc[enriched["code"].eq("a"), "pctChg_date_rank"].iloc[0] == pytest.approx(1.0)
    assert enriched.loc[enriched["code"].eq("b"), "pctChg_date_rank"].iloc[0] == pytest.approx(0.5)
    assert enriched.loc[enriched["code"].eq("a"), "kama_atr_confirm"].iloc[0] == 1.0
    assert enriched.loc[enriched["code"].eq("a"), "volume_close_strength"].iloc[0] == pytest.approx(1.6)
    assert enriched.loc[enriched["code"].eq("b"), "nonlimit_flag"].iloc[0] == 1.0


def test_build_branch_event_frame_keeps_only_executable_practical_branches() -> None:
    limitup = _base_events().copy()
    generalized = _base_events().copy()
    generalized.loc[generalized["code"].eq("a"), "entry_open_near_limit"] = True

    branches = zoo.build_branch_event_frame(limitup, generalized)

    assert "limitup_first_board_exec" in set(branches["branch_name"])
    assert "limitup_kama_atr_exec" in set(branches["branch_name"])
    assert "nonlimit_kama_breakout_exec" in set(branches["branch_name"])
    assert "nonlimit_big_new_high_exec" in set(branches["branch_name"])
    assert not branches["entry_open_near_limit"].any()


def test_encode_train_eval_aligns_categories_without_eval_category_leakage() -> None:
    train = pd.DataFrame({"x": [1.0, 2.0], "bucket": ["a", "b"], "flag": [True, False]})
    eval_frame = pd.DataFrame({"x": [3.0], "bucket": ["c"], "flag": [True]})

    train_x, eval_x, features = zoo.encode_train_eval(train, eval_frame, feature_columns=("x", "bucket", "flag"))

    assert "bucket_c" not in features
    assert "bucket_a" in features
    assert "bucket_b" in features
    assert train_x.shape[1] == eval_x.shape[1]
    assert float(eval_x["bucket_a"].iloc[0]) == 0.0
    assert float(eval_x["bucket_b"].iloc[0]) == 0.0


def test_append_ensemble_predictions_uses_within_model_year_rank_average() -> None:
    rows = []
    for model_name, scores in {"m1": [1.0, 2.0], "m2": [10.0, 0.0]}.items():
        for code, score in zip(["a", "b"], scores):
            rows.append(
                {
                    "date": pd.Timestamp("2024-01-02"),
                    "entry_date": pd.Timestamp("2024-01-03"),
                    "year": 2024,
                    "eval_year": 2024,
                    "code": code,
                    "branch_name": "branch",
                    "model_name": model_name,
                    "target_net_ret_pct": 1.0,
                    "target_raw_ret_pct": 1.3,
                    "exit_date": pd.Timestamp("2024-01-04"),
                    "entry_open_near_limit": False,
                    "executable_entry": True,
                    "predicted_ret_pct": score,
                    "predicted_big_loss_prob": 0.1,
                    "ml_score": score,
                    "model_score": score,
                }
            )
    predictions = pd.DataFrame(rows)

    output = zoo.append_ensemble_predictions(predictions)
    ensemble = output.loc[output["model_name"].eq("ensemble_rank_mean")].sort_values("code")

    assert ensemble["code"].tolist() == ["a", "b"]
    assert ensemble["ml_score"].tolist() == [pytest.approx(0.75), pytest.approx(0.75)]
    assert set(output["model_name"]) == {"m1", "m2", "ensemble_rank_mean"}
