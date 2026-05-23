from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from daily_research.path_policy.target_calibration_audit import (
    build_target_calibration_audit,
    parse_target_grid,
    summarize_target_frame,
)


def _target_frame(*, rows_per_month: int = 20, reverse_february: bool = False) -> pd.DataFrame:
    rows = []
    for month in ("2024-01", "2024-02"):
        for idx in range(rows_per_month):
            strength = (idx + 1) / rows_per_month
            score = strength
            utility = -0.02 + strength * 0.08
            if reverse_february and month == "2024-02":
                score = 1.0 - strength
            row = {
                "date": f"{month}-{idx % 5 + 1:02d}",
                "stock": f"S{idx:03d}",
                "model_family": "stock_mixer_sequence",
                "history_bucket": "high" if idx >= rows_per_month // 2 else "low",
                "stock_seen_in_train": idx % 3 != 0,
                "pred_decision_score": score,
                "future_path_max_drawdown_20d": -0.01,
                "alpha_prior_signal": float(idx % 4) / 10.0,
                "industry_name": "tech" if idx % 2 == 0 else "bank",
            }
            for horizon in (1, 3, 5, 10, 20):
                row[f"future_cum_excess_return_{horizon}d"] = utility * (horizon / 20.0)
            rows.append(row)
    return pd.DataFrame(rows)


def _dynamic_horizon_frame() -> pd.DataFrame:
    rows = []
    horizons = (1, 2, 3, 5, 8, 10, 15, 20, 30)
    for idx in range(12):
        strength = (idx + 1) / 12.0
        row = {
            "date": f"2024-01-{idx % 4 + 1:02d}",
            "stock": f"S{idx:03d}",
            "pred_decision_score": strength,
        }
        for horizon in horizons:
            row[f"future_cum_excess_return_{horizon}d"] = strength * horizon / 100.0
            row[f"future_path_max_drawdown_{horizon}d"] = -0.01 * horizon / 30.0
        rows.append(row)
    return pd.DataFrame(rows)


def test_parse_target_grid_requires_cost_hit_and_drawdown_penalty() -> None:
    grid = parse_target_grid("20:10:0.10, 15:5:0.00")

    assert grid == [
        {"cost_bps": 20.0, "hit_threshold_bps": 10.0, "drawdown_penalty": 0.10},
        {"cost_bps": 15.0, "hit_threshold_bps": 5.0, "drawdown_penalty": 0.00},
    ]

    with pytest.raises(ValueError, match="cost_bps:hit_threshold_bps:drawdown_penalty"):
        parse_target_grid("20:10")


def test_summarize_target_frame_reports_monotonic_utility_deciles() -> None:
    summary = summarize_target_frame(
        _target_frame(rows_per_month=20),
        {"cost_bps": 0.0, "hit_threshold_bps": 10.0, "drawdown_penalty": 0.0},
        deciles=5,
    )

    assert summary["hit_base_rate_status"] == "ok"
    assert summary["utility_decile_monotonicity"]["is_monotonic"] is True
    assert summary["score_decile_calibration"]["future_utility_top_bottom_spread"] > 0.0
    assert summary["score_decile_calibration"]["monthly_spread_positive_rate"] == pytest.approx(1.0)
    assert summary["gate_a"]["passed"] is True


def test_summarize_target_frame_infers_dynamic_horizons_from_columns() -> None:
    horizons = (1, 2, 3, 5, 8, 10, 15, 20, 30)

    summary = summarize_target_frame(
        _dynamic_horizon_frame(),
        {"cost_bps": 0.0, "hit_threshold_bps": 10.0, "drawdown_penalty": 0.0},
        deciles=4,
    )

    assert summary["horizons"] == list(horizons)
    assert summary["horizon_source"] == "columns"
    assert summary["utility_decile_monotonicity"]["is_monotonic"] is True


def test_summarize_target_frame_uses_horizon_specific_drawdown() -> None:
    frame = pd.DataFrame(
        [
            {
                "date": "2024-01-02",
                "stock": "AAA",
                "pred_decision_score": 1.0,
                "future_cum_excess_return_10d": 0.03,
                "future_cum_excess_return_30d": 0.04,
                "future_path_max_drawdown_10d": -0.01,
                "future_path_max_drawdown_30d": -0.30,
            }
        ]
    )

    summary = summarize_target_frame(
        frame,
        {"cost_bps": 0.0, "hit_threshold_bps": -1_000.0, "drawdown_penalty": 0.10},
        cumulative_horizons=(10, 30),
    )

    assert summary["horizons"] == [10, 30]
    assert summary["horizon_source"] == "argument"
    assert summary["utility_deciles"][0]["future_utility_mean"] == pytest.approx(
        0.03 - 0.10 * 0.01 * (10 / 30) ** 0.5
    )


def test_summarize_target_frame_flags_sparse_and_dense_hit_labels() -> None:
    sparse = summarize_target_frame(
        _target_frame(rows_per_month=20),
        {"cost_bps": 0.0, "hit_threshold_bps": 1_000.0, "drawdown_penalty": 0.0},
    )
    dense = summarize_target_frame(
        _target_frame(rows_per_month=20),
        {"cost_bps": 0.0, "hit_threshold_bps": -1_000.0, "drawdown_penalty": 0.0},
    )

    assert sparse["hit_base_rate"] == pytest.approx(0.0)
    assert sparse["hit_base_rate_status"] == "too_sparse"
    assert sparse["gate_a"]["checks"]["hit_base_rate_ok"] is False
    assert dense["hit_base_rate"] == pytest.approx(1.0)
    assert dense["hit_base_rate_status"] == "too_dense"
    assert dense["gate_a"]["checks"]["hit_base_rate_ok"] is False


def test_stock_mixer_rank_spread_with_failed_hit_lift_reports_target_miscalibration_hint() -> None:
    summary = summarize_target_frame(
        _target_frame(rows_per_month=20),
        {"cost_bps": 0.0, "hit_threshold_bps": 1_000.0, "drawdown_penalty": 0.0},
    )

    assert summary["score_decile_calibration"]["future_utility_top_bottom_spread"] > 0.0
    assert summary["score_decile_calibration"]["hit_lift_top20_mean"] <= 0.0
    assert summary["conclusion_hint"] == "ranking_signal_exists_but_hit_target_is_miscalibrated"


def test_build_target_calibration_audit_reads_study_predictions(tmp_path: Path) -> None:
    tag = "fixture_stock_mixer"
    study_root = tmp_path / "studies" / tag
    study_root.mkdir(parents=True)
    (study_root / "study_summary.json").write_text(
        json.dumps(
            {
                "status": "completed",
                "evidence_verdict": "forecast_promising",
                "model_families": ["stock_mixer_sequence"],
                "dataset_manifest": {"role_years": {"train_start_year": 2019, "test_year": 2024}},
            }
        ),
        encoding="utf-8",
    )
    frame = _target_frame(rows_per_month=20, reverse_february=True)
    frame.to_csv(study_root / "forecast_predictions_validation.csv", index=False)
    frame.to_csv(study_root / "forecast_predictions_test.csv", index=False)

    payload = build_target_calibration_audit(
        [tag],
        studies_root=tmp_path / "studies",
        target_grid=[{"cost_bps": 0.0, "hit_threshold_bps": 10.0, "drawdown_penalty": 0.0}],
        cumulative_horizons=(1, 3, 5, 10, 20),
    )

    assert payload["schema_version"] == 1
    assert payload["studies"][0]["tag"] == tag
    assert payload["studies"][0]["targets"][0]["test"]["monthly_negative_context"]["2024-02"]["top20_count"] > 0
    assert payload["studies"][0]["targets"][0]["test"]["score_decile_calibration"]["monthly_spread_positive_rate"] == pytest.approx(0.5)
    assert payload["studies"][0]["targets"][0]["gate_a"]["passed"] is True
    assert payload["gate_a"]["passed"] is True


def test_build_target_calibration_audit_gate_fails_on_validation_test_hit_drift(tmp_path: Path) -> None:
    tag = "fixture_hit_drift"
    study_root = tmp_path / "studies" / tag
    study_root.mkdir(parents=True)
    (study_root / "study_summary.json").write_text(
        json.dumps({"status": "completed", "model_families": ["linear_last_day"]}),
        encoding="utf-8",
    )
    validation = _target_frame(rows_per_month=20)
    test = _target_frame(rows_per_month=20)
    for horizon in (1, 3, 5, 10, 20):
        test[f"future_cum_excess_return_{horizon}d"] = test[f"future_cum_excess_return_{horizon}d"] - 0.20
    validation.to_csv(study_root / "forecast_predictions_validation.csv", index=False)
    test.to_csv(study_root / "forecast_predictions_test.csv", index=False)

    payload = build_target_calibration_audit(
        [tag],
        studies_root=tmp_path / "studies",
        target_grid=[{"cost_bps": 0.0, "hit_threshold_bps": 10.0, "drawdown_penalty": 0.0}],
    )

    target = payload["studies"][0]["targets"][0]
    assert target["validation"]["gate_a"]["passed"] is True
    assert target["test"]["hit_base_rate_status"] == "too_sparse"
    assert target["validation_test_hit_base_rate_abs_diff"] > 0.25
    assert target["gate_a"]["checks"]["hit_base_rate_drift_le_25pct"] is False
    assert payload["gate_a"]["passed"] is False
