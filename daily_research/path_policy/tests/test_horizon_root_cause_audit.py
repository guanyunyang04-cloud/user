from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from daily_research.path_policy.horizon_root_cause_audit import (
    build_horizon_root_cause_audit,
    build_stage26_root_cause_summary,
    evaluate_next_experiment_gate,
    target_experiment_configs,
    write_stage26_root_cause_summary_artifacts,
    write_horizon_root_cause_artifacts,
)


def _prediction_frame(*, role: str, rows: int = 36, collapse_to_30d: bool = True) -> pd.DataFrame:
    horizons = (1, 2, 3, 5, 8, 10, 15, 20, 30)
    rows_out = []
    for idx in range(rows):
        strength = (idx + 1) / rows
        horizon_cycle = horizons[idx % len(horizons)]
        row = {
            "date": f"2024-{idx % 3 + 1:02d}-{idx % 6 + 1:02d}",
            "stock": f"{idx:06d}.SZ",
            "role": role,
            "trade_utility_score": strength,
            "future_decision_score": strength * 0.10,
            "pred_best_horizon": 30 if collapse_to_30d or idx < rows - 4 else 1,
            "future_best_horizon": horizon_cycle,
        }
        for horizon in horizons:
            target_scale = 0.004 if horizon <= 5 else 0.006
            prediction_scale = 0.002 if horizon <= 5 else 0.008
            row[f"pred_decision_utility_{horizon}d"] = strength + prediction_scale * horizon
            row[f"future_decision_utility_{horizon}d"] = strength * target_scale * horizon
            row[f"future_hit_label_{horizon}d"] = 1 if strength + horizon / 100.0 > 0.60 else 0
        rows_out.append(row)
    return pd.DataFrame(rows_out)


def _study_summary() -> dict[str, object]:
    return {
        "study_tag": "unit_mh_study",
        "lake_dataset_id": "policy_input_bundle__unit",
        "dataset_manifest": {
            "role_years": {
                "train_start_year": 2019,
                "train_end_year": 2022,
                "validation_year": 2023,
                "test_year": 2024,
            }
        },
    }


def _training_summary() -> dict[str, object]:
    return {
        "feature_profile": "raw_kline_context_no_alpha_prior_v1",
        "selected_model_family": "gru_sequence_static_context",
        "selected_seed": 7,
        "training_config": {"epochs": 16, "min_epochs": 8},
        "model_results": [
            {
                "model_family": "gru_sequence_static_context",
                "seed": 7,
                "epochs_ran": 8,
                "best_epoch": 2,
                "stopped_reason": "early_stopping_patience_exhausted",
            }
        ],
    }


def _target_audit() -> dict[str, object]:
    return {
        "studies": [
            {
                "tag": "unit_mh_study",
                "targets": [
                    {
                        "validation": {
                            "hit_base_rate": 0.81,
                            "utility_decile_monotonicity": {"status": "passed"},
                        },
                        "test": {
                            "hit_base_rate": 0.83,
                            "utility_decile_monotonicity": {"status": "passed"},
                        },
                    }
                ],
            }
        ]
    }


def _horizon_calibration_report() -> dict[str, object]:
    return {
        "status": "completed",
        "gate": {
            "status": "failed_regression",
            "selected_profile": "",
            "reason": "no calibration profile preserved baseline validation/test quality",
        },
        "profiles": {
            "baseline_trade_utility": {
                "validation": {"rank_ic": 0.11, "top_bottom_spread": 0.028, "thirty_d_concentration": 0.84},
                "test": {"rank_ic": 0.10, "top_bottom_spread": 0.040, "thirty_d_concentration": 0.96},
            },
            "top2_blend": {
                "validation": {"rank_ic": 0.10, "top_bottom_spread": 0.020, "thirty_d_concentration": 0.88},
                "test": {"rank_ic": 0.10, "top_bottom_spread": 0.039, "thirty_d_concentration": 0.91},
            },
        },
    }


def test_build_audit_identifies_dynamic_horizons_and_root_causes() -> None:
    report = build_horizon_root_cause_audit(
        _prediction_frame(role="validation"),
        _prediction_frame(role="test"),
        study_summary=_study_summary(),
        training_summary=_training_summary(),
        target_audit=_target_audit(),
        horizon_calibration_report=_horizon_calibration_report(),
        baseline_context={"short_expert_policy_v5b": {"excess_annual_return": 0.7408}},
    )

    assert report["status"] == "completed"
    assert report["horizons"] == [1, 2, 3, 5, 8, 10, 15, 20, 30]
    assert report["metadata"]["study_tag"] == "unit_mh_study"
    assert report["metadata"]["dataset_id"] == "policy_input_bundle__unit"
    assert report["metadata"]["feature_profile"] == "raw_kline_context_no_alpha_prior_v1"
    assert report["metadata"]["model_family"] == "gru_sequence_static_context"
    assert report["metadata"]["seed"] == 7
    assert report["roles"]["validation"]["per_horizon"]["30"]["rank_ic"] > 0.0
    assert report["roles"]["test"]["per_horizon"]["1"]["target_std"] >= 0.0
    assert report["roles"]["test"]["horizon_alignment"]["pred_30d_share"] > 0.90
    assert report["roles"]["test"]["horizon_alignment"]["pred_long_share"] > 0.90
    assert report["roles"]["test"]["horizon_alignment"]["future_long_share"] < 0.70
    assert "true_long_horizon_edge" in report["conclusion_enums"]
    assert "horizon_head_collapse" in report["conclusion_enums"]
    assert "training_instability" in report["conclusion_enums"]
    assert "early_best_epoch_warning" in report["training_audit"]["warnings"]
    assert report["next_experiment_plan"]["phase_2_target_function_experiments"][0]["tag"] == "mh_short_utility_1_3_5d_v1"
    assert report["production_boundaries"]["active_manifest_touched"] is False
    assert report["baseline_context"]["short_expert_policy_v5b"]["excess_annual_return"] == 0.7408


def test_build_audit_blocks_missing_required_columns_without_completed_verdict() -> None:
    validation = _prediction_frame(role="validation").drop(columns=["future_decision_utility_30d"])

    report = build_horizon_root_cause_audit(validation, _prediction_frame(role="test"))

    assert report["status"] == "blocked"
    assert "future_decision_utility_30d" in report["reason"]
    assert "conclusion_enums" not in report


def test_write_artifacts_outputs_json_csv_and_markdown(tmp_path: Path) -> None:
    report = build_horizon_root_cause_audit(
        _prediction_frame(role="validation"),
        _prediction_frame(role="test"),
        study_summary=_study_summary(),
        training_summary=_training_summary(),
    )

    paths = write_horizon_root_cause_artifacts(report, tmp_path)

    assert paths["json"].name == "horizon_root_cause_audit.json"
    assert paths["csv"].name == "horizon_root_cause_audit.csv"
    assert paths["markdown"].name == "horizon_root_cause_audit.md"
    loaded = json.loads(paths["json"].read_text(encoding="utf-8"))
    rows = pd.read_csv(paths["csv"])
    markdown = paths["markdown"].read_text(encoding="utf-8")
    assert loaded["status"] == "completed"
    assert {"role", "horizon", "rank_ic", "top_bottom_spread", "hit_lift_top20_mean"}.issubset(rows.columns)
    assert "无 live/default" in markdown


def test_stage26_summary_classifies_common_seed_and_test_only_negative_months(tmp_path: Path) -> None:
    base = {
        "status": "completed",
        "metadata": {"study_tag": "mh25_path_aux_fullgrid_rebudget_seed7_20260526_01", "seed": 7},
        "conclusion_enums": ["horizon_head_collapse", "training_instability"],
        "roles": {
            "validation": {
                "horizon_alignment": {"pred_long_share": 0.84, "pred_30d_share": 0.80},
                "per_horizon": {
                    "30": {
                        "monthly_spread": {"2023-01": -0.01, "2023-02": 0.02},
                        "monthly_spread_positive_rate": 0.50,
                        "negative_months": ["2023-01"],
                    }
                },
            },
            "test": {
                "horizon_alignment": {"pred_long_share": 0.94, "pred_30d_share": 0.90},
                "per_horizon": {
                    "30": {
                        "monthly_spread": {"2024-01": -0.03, "2024-02": 0.01, "2024-03": -0.02},
                        "monthly_spread_positive_rate": 0.333333,
                        "negative_months": ["2024-01", "2024-03"],
                    }
                },
            },
        },
    }
    seed11 = json.loads(json.dumps(base))
    seed11["metadata"]["study_tag"] = "mh25_path_aux_fullgrid_rebudget_seed11_20260526_01"
    seed11["metadata"]["seed"] = 11
    seed11["roles"]["test"]["per_horizon"]["30"]["negative_months"] = ["2024-01"]
    seed11["roles"]["test"]["per_horizon"]["30"]["monthly_spread"] = {"2024-01": -0.02, "2024-02": 0.02}
    daily = json.loads(json.dumps(base))
    daily["metadata"]["study_tag"] = "mh25_path_aux_daily1_45_multiseed_seed7_20260526_01"
    daily["roles"]["test"]["horizon_alignment"]["pred_long_share"] = 0.70

    report = build_stage26_root_cause_summary([base, seed11, daily], run_tag="unit_stage26")

    assert report["status"] == "completed"
    assert report["run_tag"] == "unit_stage26"
    assert report["candidate_summary"]["fullgrid_rebudget"]["seed_count"] == 2
    assert report["candidate_summary"]["fullgrid_rebudget"]["common_test_negative_months"] == ["2024-01"]
    assert "2024-03" in report["candidate_summary"]["fullgrid_rebudget"]["seed_specific_test_negative_months"]
    assert report["candidate_summary"]["fullgrid_rebudget"]["mean_test_pred_long_share"] > 0.90
    assert report["candidate_summary"]["fullgrid_rebudget"]["recommended_path"] == "loss_regularization_stabilization"
    assert report["production_boundaries"]["training_started_by_summary"] is False

    paths = write_stage26_root_cause_summary_artifacts(report, tmp_path)
    assert paths["matrix_csv"].name == "stage26_root_cause_matrix.csv"
    assert paths["verdict_md"].name == "stage26_root_cause_verdict.md"
    matrix = pd.read_csv(paths["matrix_csv"])
    verdict = paths["verdict_md"].read_text(encoding="utf-8")
    assert {"candidate", "seed", "role", "horizon", "negative_months"}.issubset(matrix.columns)
    assert "Stage 2.6" in verdict


def test_horizon_alignment_treats_daily_grid_horizons_above_30_as_long() -> None:
    frame = _prediction_frame(role="test", rows=6, collapse_to_30d=False)
    frame["pred_best_horizon"] = [31, 35, 40, 45, 20, 5]
    frame["future_best_horizon"] = [31, 35, 8, 10, 20, 5]

    report = build_horizon_root_cause_audit(frame, frame)

    alignment = report["roles"]["test"]["horizon_alignment"]
    assert alignment["pred_long_share"] == 5 / 6
    assert alignment["future_long_share"] == 3 / 6


def test_target_experiment_configs_are_shadow_only_and_do_not_overlap_tags() -> None:
    configs = target_experiment_configs(
        dataset_id="policy_input_bundle__unit",
        pool="liquid500",
        seed=7,
    )

    assert [item["tag"] for item in configs] == [
        "mh_short_utility_1_3_5d_v1",
        "mh_mid_utility_5_10_20d_v1",
        "mh_long_utility_15_20_30d_v1",
    ]
    assert [item["horizons"] for item in configs] == [[1, 3, 5], [5, 10, 20], [15, 20, 30]]
    assert all(item["scope"] == "shadow_only" for item in configs)
    assert all(item["dataset_id"] == "policy_input_bundle__unit" for item in configs)
    assert all(item["pool"] == "liquid500" for item in configs)
    assert all(item["seed"] == 7 for item in configs)
    assert all(item["may_touch_active_manifest"] is False for item in configs)


def test_next_experiment_gate_distinguishes_short_mid_long_and_execution_failure() -> None:
    short_success = evaluate_next_experiment_gate(
        {
            "mh_short_utility_1_3_5d_v1": {
                "prediction_gate": "passed",
                "execution_gate": "passed",
                "beats_production_baseline": True,
            }
        }
    )
    assert short_success["status"] == "shortline_candidate_ready_for_multiseed"

    execution_failed = evaluate_next_experiment_gate(
        {
            "mh_short_utility_1_3_5d_v1": {
                "prediction_gate": "passed",
                "execution_gate": "failed",
                "beats_production_baseline": False,
            }
        }
    )
    assert execution_failed["status"] == "short_prediction_signal_execution_failed"

    long_wins = evaluate_next_experiment_gate(
        {
            "mh_mid_utility_5_10_20d_v1": {"prediction_gate": "failed"},
            "mh_long_utility_15_20_30d_v1": {
                "prediction_gate": "passed",
                "execution_gate": "passed",
                "beats_production_baseline": False,
            },
        }
    )
    assert long_wins["status"] == "long_or_mid_family_wins_research_only"

    insufficient = evaluate_next_experiment_gate({})
    assert insufficient["status"] == "evidence_insufficient"
