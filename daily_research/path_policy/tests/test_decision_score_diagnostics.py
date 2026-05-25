from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from daily_research.path_policy.decision_score_diagnostics import (
    add_score_columns,
    build_diagnostics,
    main,
    score_passes_gate,
    summarize_frame,
)


def _fixture_frame() -> pd.DataFrame:
    rows = []
    for month, good_direction in (("2024-01", True), ("2024-02", False)):
        for day in (1, 2):
            for idx in range(5):
                score = 0.05 - idx * 0.01
                future = 0.05 - idx * 0.01 if good_direction else -0.05 + idx * 0.01
                row = {
                    "date": f"{month}-{day + 1:02d}",
                    "stock": f"S{idx}",
                    "pred_decision_score": score,
                    "future_decision_score": future,
                    "pred_best_horizon": 5 if idx < 2 else 1,
                    "future_best_horizon": 5 if idx < 2 else 1,
                    "pred_cum_mu_5d": score * 0.5,
                    "pred_cum_mu_20d": score * 0.2,
                    "history_bucket": "high" if idx < 2 else "medium",
                    "stock_seen_in_train": idx != 4,
                }
                for horizon in (1, 3, 5, 10, 20):
                    row[f"pred_decision_utility_{horizon}d"] = score + horizon * 0.001
                    row[f"future_decision_utility_{horizon}d"] = future + horizon * 0.001
                    row[f"pred_hit_prob_{horizon}d"] = 0.9 - idx * 0.1
                    row[f"future_hit_label_{horizon}d"] = 1 if future > 0 and idx < 2 else 0
                rows.append(row)
    return pd.DataFrame(rows)


def test_add_score_columns_builds_expected_variants() -> None:
    frame = add_score_columns(_fixture_frame())

    assert frame["max_pred_utility"].iloc[0] == pytest.approx(0.05)
    assert frame["horizon_selected_utility"].iloc[0] == pytest.approx(0.055)
    assert frame["hit_weighted_utility"].iloc[0] > 0.0
    assert frame["short_horizon_blend"].iloc[0] == pytest.approx(0.0524)
    assert frame["decision_forecast_blend"].iloc[0] == pytest.approx(0.0345)


def test_add_score_columns_infers_dynamic_horizons_and_trade_utility_score() -> None:
    rows = []
    horizons = (1, 2, 3, 5, 8, 10, 15, 20, 30)
    for idx in range(5):
        score = 0.05 - idx * 0.01
        row = {
            "date": "2024-01-02",
            "stock": f"S{idx}",
            "pred_decision_score": score,
            "future_decision_score": score,
            "pred_best_horizon": 8,
            "future_best_horizon": 8,
            "pred_cum_mu_5d": score * 0.5,
            "pred_cum_mu_20d": score * 0.2,
        }
        for horizon in horizons:
            row[f"pred_decision_utility_{horizon}d"] = score + horizon * 0.001
            row[f"future_decision_utility_{horizon}d"] = score + horizon * 0.001
            row[f"pred_hit_prob_{horizon}d"] = 0.8
            row[f"future_hit_label_{horizon}d"] = 1 if idx < 2 else 0
        rows.append(row)

    summary = summarize_frame(pd.DataFrame(rows))
    frame = add_score_columns(pd.DataFrame(rows))

    assert summary["horizons"] == list(horizons)
    assert summary["horizon_source"] == "columns"
    assert "horizon_discovery" in summary
    assert summary["horizon_discovery"]["30"]["rank_ic"] > 0.0
    assert frame["trade_utility_score"].iloc[0] == pytest.approx(0.08)
    assert frame["horizon_selected_utility"].iloc[0] == pytest.approx(0.058)


def test_summarize_frame_uses_path_proxy_decision_scores() -> None:
    horizons = (1, 2, 3, 5, 8, 10, 15, 20, 30)
    rows = []
    for idx in range(12):
        strength = (idx + 1) / 12.0
        row = {
            "date": f"2024-01-{idx % 4 + 1:02d}",
            "stock": f"S{idx:03d}",
            "future_path_max_drawdown_20d": -0.01,
        }
        for horizon in horizons:
            row[f"pred_cum_mu_{horizon}d"] = strength * horizon / 100.0
            row[f"future_cum_excess_return_{horizon}d"] = strength * horizon / 100.0
            row[f"future_path_max_drawdown_{horizon}d"] = -0.01 * horizon / 30.0
        rows.append(row)

    summary = summarize_frame(pd.DataFrame(rows))

    assert summary["decision_score_source"] == "path_proxy"
    assert summary["horizons"] == list(horizons)
    assert summary["scores"]["trade_utility_score"]["rank_ic"] > 0.0


def test_summarize_frame_reports_monthly_spread_and_negative_months() -> None:
    summary = summarize_frame(_fixture_frame())
    max_score = summary["scores"]["max_pred_utility"]

    assert max_score["top_bottom_spread"] == pytest.approx(0.0)
    assert max_score["monthly_spread_positive_rate"] == pytest.approx(0.5)
    assert max_score["negative_months"] == ["2024-02"]
    assert max_score["worst_month"] == "2024-02"
    assert max_score["hit_lift_top20_mean"] > 0.0
    assert max_score["negative_month_context"]["2024-02"]["history_bucket"]["high"] == 2


def test_missing_decision_columns_raise_clear_error() -> None:
    with pytest.raises(ValueError, match="missing required columns"):
        add_score_columns(pd.DataFrame({"date": ["2024-01-02"]}))


def test_score_gate_requires_validation_test_and_monthly_stability() -> None:
    passed_validation = {
        "rank_ic": 0.1,
        "top_bottom_spread": 0.01,
        "hit_lift_top20_mean": 0.02,
        "monthly_spread_positive_rate": 1.0,
        "negative_months": [],
    }
    weak_test = {
        "rank_ic": 0.1,
        "top_bottom_spread": 0.01,
        "hit_lift_top20_mean": 0.02,
        "monthly_spread_positive_rate": 0.5,
        "negative_months": ["2024-01"],
    }
    strong_test = {
        **weak_test,
        "monthly_spread_positive_rate": 0.75,
        "negative_months": [],
    }

    assert not score_passes_gate(passed_validation, weak_test, baseline_negative_count=2)["passed"]
    assert score_passes_gate(passed_validation, strong_test, baseline_negative_count=2)["passed"]
    assert score_passes_gate(passed_validation, strong_test, baseline_negative_count=0)["passed"]


def test_build_diagnostics_reads_completed_study_fixture(tmp_path: Path) -> None:
    tag = "fixture_study"
    root = tmp_path / "studies"
    study = root / tag
    study.mkdir(parents=True)
    (study / "study_summary.json").write_text(
        json.dumps(
            {
                "status": "completed",
                "evidence_verdict": "forecast_promising",
                "lake_dataset_id": "dataset",
                "dataset_manifest": {"role_years": {"train_start_year": 2020}},
                "model_families": ["gru_sequence_static_context"],
            }
        ),
        encoding="utf-8",
    )
    frame = _fixture_frame()
    frame.to_csv(study / "forecast_predictions_validation.csv", index=False)
    frame.to_csv(study / "forecast_predictions_test.csv", index=False)

    payload = build_diagnostics([tag], studies_root=root)

    assert payload["studies"][0]["tag"] == tag
    assert payload["studies"][0]["validation"]["date_count"] == 4
    assert payload["conclusion_hint"] in {
        "selection_calibration_candidate_found",
        "no_selection_only_fix_found; inspect target_definition_or_input_regime_contamination",
    }


def test_build_diagnostics_includes_target_audit_misalignment_context(tmp_path: Path) -> None:
    tag = "fixture_stock_mixer"
    root = tmp_path / "studies"
    study = root / tag
    study.mkdir(parents=True)
    (study / "study_summary.json").write_text(
        json.dumps(
            {
                "status": "completed",
                "evidence_verdict": "forecast_promising",
                "dataset_manifest": {"role_years": {"train_start_year": 2020}},
                "model_families": ["stock_mixer_sequence"],
            }
        ),
        encoding="utf-8",
    )
    frame = _fixture_frame()
    frame.to_csv(study / "forecast_predictions_validation.csv", index=False)
    frame.to_csv(study / "forecast_predictions_test.csv", index=False)
    target_audit_path = tmp_path / "target_audit.json"
    target_audit_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "gate_a": {"passed": False},
                "conclusion_hints": ["ranking_signal_exists_but_hit_target_is_miscalibrated"],
                "studies": [
                    {
                        "tag": tag,
                        "targets": [
                            {
                                "target": {"name": "cost20_hit10_dd0.1"},
                                "gate_a": {"passed": False},
                                "validation": {"conclusion_hint": "target_calibration_inconclusive"},
                                "test": {"conclusion_hint": "ranking_signal_exists_but_hit_target_is_miscalibrated"},
                            }
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    payload = build_diagnostics([tag], studies_root=root, target_audit=target_audit_path)

    assert payload["target_audit_context"]["source_path"] == str(target_audit_path.resolve())
    assert payload["target_audit_context"]["gate_a_passed"] is False
    assert payload["target_audit_context"]["study_hints"][0]["tag"] == tag
    assert payload["score_target_inconsistency_reasons"] == [
        "ranking_signal_exists_but_hit_target_is_miscalibrated"
    ]


def test_decision_score_diagnostics_cli_accepts_target_audit(tmp_path: Path) -> None:
    tag = "fixture_study"
    root = tmp_path / "studies"
    study = root / tag
    study.mkdir(parents=True)
    (study / "study_summary.json").write_text(
        json.dumps(
            {
                "status": "completed",
                "evidence_verdict": "forecast_promising",
                "dataset_manifest": {"role_years": {"train_start_year": 2020}},
                "model_families": ["gru_sequence_static_context"],
            }
        ),
        encoding="utf-8",
    )
    frame = _fixture_frame()
    frame.to_csv(study / "forecast_predictions_validation.csv", index=False)
    frame.to_csv(study / "forecast_predictions_test.csv", index=False)
    target_audit = tmp_path / "target_audit.json"
    target_audit.write_text(
        json.dumps({"schema_version": 1, "gate_a": {"passed": False}, "conclusion_hints": ["hit_label_base_rate_out_of_range"], "studies": []}),
        encoding="utf-8",
    )
    output = tmp_path / "diagnostics.json"

    assert main(
        [
            "--tags",
            tag,
            "--studies-root",
            str(root),
            "--output",
            str(output),
            "--target-audit",
            str(target_audit),
        ]
    ) == 0

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["target_audit_context"]["conclusion_hints"] == ["hit_label_base_rate_out_of_range"]
