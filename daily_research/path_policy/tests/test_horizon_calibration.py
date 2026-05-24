from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from daily_research.path_policy.horizon_calibration import (
    build_horizon_calibration_report,
    evaluate_horizon_calibration_gate,
    infer_decision_utility_horizons,
    score_horizon_profiles,
    write_horizon_calibration_artifacts,
)


def _prediction_frame(*, role: str = "validation", rows: int = 20, slope: float = 0.01) -> pd.DataFrame:
    horizons = (1, 2, 3, 5, 8, 10, 15, 20, 30)
    items = []
    for idx in range(rows):
        strength = (idx + 1) / rows
        row = {
            "date": f"2024-01-{(idx % 10) + 1:02d}",
            "stock": f"{idx:06d}.SZ",
            "role": role,
            "trade_utility_score": strength,
            "future_decision_score": strength + 0.02,
            "pred_best_horizon": 30 if idx < rows - 2 else 1,
            "future_best_horizon": horizons[idx % len(horizons)],
        }
        for horizon in horizons:
            row[f"pred_decision_utility_{horizon}d"] = strength + slope * horizon
            row[f"future_decision_utility_{horizon}d"] = strength + 0.005 * horizon
            row[f"pred_hit_prob_{horizon}d"] = 0.55 + strength * 0.2
            row[f"future_hit_label_{horizon}d"] = 1 if idx >= rows // 2 else 0
        items.append(row)
    return pd.DataFrame(items)


def test_score_horizon_profiles_infers_dynamic_horizons_and_keeps_original_predictions() -> None:
    frame = _prediction_frame()
    original_columns = list(frame.columns)

    scored, horizons = score_horizon_profiles(frame)

    assert horizons == (1, 2, 3, 5, 8, 10, 15, 20, 30)
    assert list(frame.columns) == original_columns
    for column in [
        "baseline_trade_utility",
        "softmax_expected_utility_t015",
        "long_blend_15_20_30",
        "long_blend_10_15_20_30",
        "top2_blend",
    ]:
        assert column in scored.columns
        assert np.isfinite(scored[column]).all()
    assert scored["baseline_trade_utility"].tolist() == pytest.approx(frame["trade_utility_score"].tolist())


def test_score_horizon_profiles_blocks_missing_required_horizon_columns() -> None:
    frame = _prediction_frame().drop(columns=["pred_decision_utility_30d"])

    with pytest.raises(ValueError, match="missing required decision utility columns"):
        score_horizon_profiles(frame)


def test_build_report_blocks_missing_required_horizon_columns_without_completed_verdict() -> None:
    frame = _prediction_frame().drop(columns=["future_decision_utility_30d"])

    report = build_horizon_calibration_report(frame, _prediction_frame())

    assert report["status"] == "blocked"
    assert "missing required decision utility columns" in report["reason"]
    assert "gate" not in report


def test_build_report_blocks_empty_validation_or_test_predictions() -> None:
    frame = _prediction_frame()

    report = build_horizon_calibration_report(frame.iloc[0:0], frame)

    assert report["status"] == "blocked"
    assert report["reason"] == "validation/test predictions must both be non-empty"


def test_build_report_contains_required_metadata_and_role_metrics() -> None:
    validation = _prediction_frame(role="validation", rows=24)
    test = _prediction_frame(role="test", rows=24)

    report = build_horizon_calibration_report(
        validation,
        test,
        study_metadata={
            "study_tag": "unit_study",
            "dataset_id": "policy_input_bundle__unit",
            "feature_profile": "raw_kline_context_no_alpha_prior_v1",
            "model_family": "gru_sequence_static_context",
            "seed": 7,
            "role_years": {"validation_year": 2023, "test_year": 2024},
        },
    )

    assert report["status"] == "completed"
    assert report["study_tag"] == "unit_study"
    assert report["dataset_id"] == "policy_input_bundle__unit"
    assert report["feature_profile"] == "raw_kline_context_no_alpha_prior_v1"
    assert report["model_family"] == "gru_sequence_static_context"
    assert report["seed"] == 7
    assert report["role_years"] == {"validation_year": 2023, "test_year": 2024}
    assert report["horizons"] == [1, 2, 3, 5, 8, 10, 15, 20, 30]
    assert "baseline_trade_utility" in report["profiles"]
    assert "validation" in report["profiles"]["baseline_trade_utility"]
    assert "test" in report["profiles"]["baseline_trade_utility"]
    assert report["profiles"]["baseline_trade_utility"]["validation"]["row_count"] == 24
    assert report["profiles"]["baseline_trade_utility"]["validation"]["target_column"] == "future_decision_score"
    assert "thirty_d_concentration" in report["profiles"]["baseline_trade_utility"]["validation"]


def test_gate_distinguishes_calibrated_long_family_and_regression() -> None:
    baseline = {
        "validation": {
            "rank_ic": 0.10,
            "top_bottom_spread": 0.03,
            "monthly_spread_positive_rate": 0.82,
            "negative_months": ["2023-01"],
            "long_horizon_share": 0.95,
        },
        "test": {
            "rank_ic": 0.09,
            "top_bottom_spread": 0.04,
            "monthly_spread_positive_rate": 1.0,
            "negative_months": [],
            "long_horizon_share": 0.97,
        },
    }
    profiles = {
        "baseline_trade_utility": baseline,
        "softmax_expected_utility_t015": {
            "validation": {**baseline["validation"], "rank_ic": 0.095, "top_bottom_spread": 0.028, "long_horizon_share": 0.70},
            "test": {**baseline["test"], "rank_ic": 0.086, "top_bottom_spread": 0.037, "long_horizon_share": 0.72},
        },
        "long_blend_15_20_30": {
            "validation": {**baseline["validation"], "rank_ic": 0.096, "top_bottom_spread": 0.029},
            "test": {**baseline["test"], "rank_ic": 0.087, "top_bottom_spread": 0.038},
        },
        "top2_blend": {
            "validation": {**baseline["validation"], "rank_ic": 0.02, "top_bottom_spread": 0.005},
            "test": {**baseline["test"], "rank_ic": 0.01, "top_bottom_spread": 0.002},
        },
    }

    calibrated = evaluate_horizon_calibration_gate({"profiles": profiles})
    assert calibrated["status"] == "passed_calibrated"
    assert calibrated["selected_profile"] == "softmax_expected_utility_t015"

    long_family_profiles = dict(profiles)
    long_family_profiles["softmax_expected_utility_t015"] = profiles["top2_blend"]
    long_family = evaluate_horizon_calibration_gate({"profiles": long_family_profiles})
    assert long_family["status"] == "passed_long_family"
    assert long_family["selected_profile"] == "long_blend_15_20_30"

    failed_profiles = {key: profiles["top2_blend"] for key in profiles}
    failed_profiles["baseline_trade_utility"] = baseline
    failed = evaluate_horizon_calibration_gate({"profiles": failed_profiles})
    assert failed["status"] == "failed_regression"
    assert failed["selected_profile"] == ""


def test_write_artifacts_outputs_report_and_comparison(tmp_path) -> None:
    validation = _prediction_frame(role="validation", rows=16)
    test = _prediction_frame(role="test", rows=16)
    report = build_horizon_calibration_report(validation, test, study_metadata={"study_tag": "unit"})

    paths = write_horizon_calibration_artifacts(report, tmp_path)

    assert paths["report_json"].name == "horizon_calibration_report.json"
    assert paths["comparison_csv"].name == "horizon_score_comparison.csv"
    loaded = json.loads(paths["report_json"].read_text(encoding="utf-8"))
    comparison = pd.read_csv(paths["comparison_csv"])
    assert loaded["study_tag"] == "unit"
    assert {"profile", "role", "rank_ic", "top_bottom_spread", "monthly_spread_positive_rate"}.issubset(
        comparison.columns
    )
    assert "thirty_d_concentration" in comparison.columns


def test_infer_decision_utility_horizons_blocks_empty_input() -> None:
    with pytest.raises(ValueError, match="No pred_decision_utility"):
        infer_decision_utility_horizons(pd.DataFrame({"trade_utility_score": [1.0]}))
