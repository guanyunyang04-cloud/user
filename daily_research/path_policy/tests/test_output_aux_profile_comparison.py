from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from daily_research.path_policy.output_aux_profile_comparison import (
    OUTPUT_AUX_VERDICTS,
    build_output_aux_profile_comparison,
    next_stage_decision,
    output_profile_comparison_rows,
    profile_aggregate_rows,
    score_variant_comparison_rows,
    write_output_aux_profile_comparison,
)


def _prediction_frame(*, rows: int = 24, horizons: tuple[int, ...] = (1, 2, 3, 5, 8, 10, 15, 20, 30)) -> pd.DataFrame:
    items = []
    for idx in range(rows):
        strength = (idx + 1) / rows
        row = {
            "date": f"2024-01-{(idx % 8) + 1:02d}",
            "stock": f"{idx:06d}.SZ",
            "role": "test",
            "pred_decision_score": strength,
            "trade_utility_score": strength,
            "hit_weighted_utility": strength + 0.010,
            "short_horizon_blend": strength + 0.020,
            "horizon_selected_utility": strength + 0.030,
            "decision_forecast_blend": strength + 0.040,
            "future_decision_score": strength + 0.01,
            "pred_best_horizon": max(horizons),
            "future_best_horizon": horizons[idx % len(horizons)],
        }
        for horizon in horizons:
            row[f"pred_cum_mu_{horizon}d"] = strength + 0.001 * horizon
            row[f"future_cum_excess_return_{horizon}d"] = strength + 0.002 * horizon
            row[f"pred_decision_utility_{horizon}d"] = strength + 0.001 * horizon
            row[f"future_decision_utility_{horizon}d"] = strength + 0.002 * horizon
            row[f"pred_hit_prob_{horizon}d"] = 0.50 + strength * 0.20
            row[f"future_hit_label_{horizon}d"] = 1 if idx >= rows // 2 else 0
        return_cols = {}
        for step in range(1, max(horizons) + 1):
            return_cols[f"target_excess_{step}d"] = 0.001 * step + strength
            return_cols[f"pred_q10_{step}d"] = strength - 0.02
            return_cols[f"pred_q90_{step}d"] = strength + 0.02
        row.update(return_cols)
        items.append(row)
    return pd.DataFrame(items)


def _study(root: Path, tag: str, *, loss_profile: str, horizons: tuple[int, ...], daily_grid: bool = False) -> Path:
    study = root / tag
    study.mkdir(parents=True)
    frame = _prediction_frame(horizons=horizons)
    frame.to_csv(study / "forecast_predictions_validation.csv", index=False)
    frame.to_csv(study / "forecast_predictions_test.csv", index=False)
    training = {
        "status": "completed",
        "selected_model_family": "gru_sequence_static_context",
        "selected_seed": 7,
        "training_config": {
            "loss_profile": loss_profile,
            "output_profile": "decision_utility_v1",
            "forecast_horizon": max(horizons),
            "cumulative_horizons": list(horizons),
            "horizon_grid_name": "daily_1_45_feasibility" if daily_grid else "full_1_2_3_5_8_10_15_20_30",
            "feature_profile": "raw_kline_context_no_alpha_prior_v1",
        },
        "validation_metrics": {
            "status": "completed",
            "decision_score_rank_ic": 0.10,
            "decision_score_top_bottom_spread": 0.04,
            "decision_hit_lift_top20_mean": 0.02,
            "decision_utility_profile_status": "passed",
        },
        "test_metrics": {
            "status": "completed",
            "decision_score_rank_ic": 0.09,
            "decision_score_top_bottom_spread": 0.03,
            "decision_hit_lift_top20_mean": 0.02,
            "decision_utility_profile_status": "passed",
        },
        "evidence_verdict": "forecast_test_confirmed",
        "shadow_only": True,
        "promotion_allowed": False,
    }
    (study / "forecast_training_summary.json").write_text(json.dumps(training), encoding="utf-8")
    return study


def _study_with_seed(
    root: Path,
    tag: str,
    *,
    loss_profile: str,
    seed: int,
    score_sign: float = 1.0,
    horizons: tuple[int, ...] = (1, 2, 3, 5, 8, 10, 15, 20, 30),
) -> Path:
    study = _study(root, tag, loss_profile=loss_profile, horizons=horizons)
    training = json.loads((study / "forecast_training_summary.json").read_text(encoding="utf-8"))
    training["selected_seed"] = seed
    (study / "forecast_training_summary.json").write_text(json.dumps(training), encoding="utf-8")
    if score_sign < 0:
        for name in ("forecast_predictions_validation.csv", "forecast_predictions_test.csv"):
            frame = pd.read_csv(study / name)
            for column in (
                "pred_decision_score",
                "trade_utility_score",
                "hit_weighted_utility",
                "short_horizon_blend",
                "horizon_selected_utility",
                "decision_forecast_blend",
            ):
                frame[column] = -pd.to_numeric(frame[column], errors="coerce")
            frame.to_csv(study / name, index=False)
    return study


def _study_with_grid(
    root: Path,
    tag: str,
    *,
    loss_profile: str,
    seed: int,
    horizon_grid_name: str,
    horizons: tuple[int, ...],
    score_sign: float = 1.0,
) -> Path:
    study = _study_with_seed(root, tag, loss_profile=loss_profile, seed=seed, horizons=horizons, score_sign=score_sign)
    training = json.loads((study / "forecast_training_summary.json").read_text(encoding="utf-8"))
    training["training_config"]["horizon_grid_name"] = horizon_grid_name
    (study / "forecast_training_summary.json").write_text(json.dumps(training), encoding="utf-8")
    return study


def test_build_output_aux_profile_comparison_blocks_missing_required_columns(tmp_path: Path) -> None:
    study = _study(tmp_path, "bad_missing_cols", loss_profile="decision_utility_path_aux_v1", horizons=(1, 3, 5))
    frame = pd.read_csv(study / "forecast_predictions_test.csv").drop(columns=["future_decision_score"])
    frame.to_csv(study / "forecast_predictions_test.csv", index=False)

    report = build_output_aux_profile_comparison([study], run_tag="unit")

    assert report["status"] == "completed_with_blocked_studies"
    assert report["studies"][0]["status"] == "blocked"
    assert "missing required prediction columns" in report["studies"][0]["reason"]
    assert report["research_verdicts"] == ["needs_input_redesign"]


def test_path_only_baseline_uses_path_proxy_decision_scores(tmp_path: Path) -> None:
    study = _study(
        tmp_path,
        "path_only_full_seed7",
        loss_profile="forecast_path_v1_baseline",
        horizons=(1, 2, 3, 5, 8, 10, 15, 20, 30),
    )
    for name in ("forecast_predictions_validation.csv", "forecast_predictions_test.csv"):
        frame = pd.read_csv(study / name)
        drop_cols = [
            column
            for column in frame.columns
            if column.startswith("pred_decision_utility_")
            or column.startswith("future_decision_utility_")
            or column.startswith("pred_hit_prob_")
            or column.startswith("future_hit_label_")
            or column
            in {
                "pred_decision_score",
                "future_decision_score",
                "trade_utility_score",
                "pred_best_horizon",
                "future_best_horizon",
            }
        ]
        frame.drop(columns=drop_cols).to_csv(study / name, index=False)

    report = build_output_aux_profile_comparison([study], run_tag="unit")

    assert report["status"] == "completed"
    assert report["studies"][0]["status"] == "completed"
    assert report["studies"][0]["decision_score_source"] == "path_proxy"
    rows = output_profile_comparison_rows(report)
    assert {row["decision_score_source"] for row in rows} == {"path_proxy"}


def test_decision_utility_baseline_preserves_native_decision_score_source(tmp_path: Path) -> None:
    study = _study(
        tmp_path,
        "utility_full_seed7",
        loss_profile="decision_utility_v1_baseline",
        horizons=(1, 2, 3, 5, 8, 10, 15, 20, 30),
    )

    report = build_output_aux_profile_comparison([study], run_tag="unit")

    assert report["status"] == "completed"
    assert report["studies"][0]["decision_score_source"] == "model_decision_utility"
    rows = output_profile_comparison_rows(report)
    assert {row["decision_score_source"] for row in rows} == {"model_decision_utility"}


def test_write_output_aux_profile_comparison_outputs_schema_and_daily_grid_feasibility_verdict(tmp_path: Path) -> None:
    full = _study(
        tmp_path,
        "decision_full_seed7",
        loss_profile="decision_utility_v1_baseline",
        horizons=(1, 2, 3, 5, 8, 10, 15, 20, 30),
    )
    daily = _study(
        tmp_path,
        "daily_grid_seed7",
        loss_profile="decision_utility_path_aux_v1",
        horizons=tuple(range(1, 46)),
        daily_grid=True,
    )

    report = build_output_aux_profile_comparison([full, daily], run_tag="unit")
    assert report["status"] == "completed"
    assert set(report["research_verdicts"]).issubset(OUTPUT_AUX_VERDICTS)
    assert "daily_grid_continue_feasibility_only" in report["research_verdicts"]
    assert all(item["promotion_allowed"] is False for item in report["studies"])

    paths = write_output_aux_profile_comparison(report, tmp_path / "comparison")
    assert paths["output_profile_comparison_csv"].name == "output_profile_comparison.csv"
    assert paths["horizon_grid_comparison_csv"].name == "horizon_grid_comparison.csv"
    assert paths["architecture_input_interaction_report_json"].name == "architecture_input_interaction_report.json"
    assert paths["research_verdict_md"].name == "research_verdict.md"
    loaded = json.loads(paths["architecture_input_interaction_report_json"].read_text(encoding="utf-8"))
    assert loaded["run_tag"] == "unit"
    comparison = pd.read_csv(paths["output_profile_comparison_csv"])
    assert {"study_tag", "loss_profile", "role", "decision_score_rank_ic", "decision_score_top_bottom_spread"}.issubset(
        comparison.columns
    )


def test_daily_grid_feasibility_does_not_unlock_stage3_architecture(tmp_path: Path) -> None:
    daily = _study(
        tmp_path,
        "daily_grid_seed7",
        loss_profile="decision_utility_path_aux_v1",
        horizons=tuple(range(1, 46)),
        daily_grid=True,
    )

    report = build_output_aux_profile_comparison([daily], run_tag="unit")
    aggregate = next(
        row
        for row in profile_aggregate_rows(report)
        if row["loss_profile"] == "decision_utility_path_aux_v1"
        and row["role"] == "test"
        and row["score_name"] == "pred_decision_score"
    )
    decision = next_stage_decision(report)

    assert aggregate["daily_grid_feasibility"] is True
    assert aggregate["seed_count"] == 1
    assert aggregate["stage3_weak_gate_pass"] is False
    assert decision["stage3_architecture_allowed"] is False
    assert decision["stage3_candidates"] == []


def test_daily_grid_feasibility_does_not_emit_architecture_verdict(tmp_path: Path) -> None:
    studies = [
        _study(
            tmp_path,
            "daily_grid_path_aux_seed7",
            loss_profile="decision_utility_path_aux_v1",
            horizons=tuple(range(1, 46)),
            daily_grid=True,
        ),
        _study(
            tmp_path,
            "daily_grid_utility_seed7",
            loss_profile="decision_utility_v1_baseline",
            horizons=tuple(range(1, 46)),
            daily_grid=True,
        ),
    ]

    report = build_output_aux_profile_comparison(studies, run_tag="unit")

    assert "daily_grid_continue_feasibility_only" in report["research_verdicts"]
    assert "architecture_compare_allowed" not in report["research_verdicts"]


def test_profile_aggregate_rows_compute_seed_dispersion_and_gate(tmp_path: Path) -> None:
    studies = [
        _study_with_seed(tmp_path, "path_aux_seed7", loss_profile="decision_utility_path_aux_v1", seed=7),
        _study_with_seed(tmp_path, "path_aux_seed11", loss_profile="decision_utility_path_aux_v1", seed=11),
        _study_with_seed(
            tmp_path,
            "path_aux_seed19",
            loss_profile="decision_utility_path_aux_v1",
            seed=19,
            score_sign=-1.0,
        ),
    ]

    report = build_output_aux_profile_comparison(studies, run_tag="unit")
    rows = profile_aggregate_rows(report)
    test_row = next(
        row
        for row in rows
        if row["loss_profile"] == "decision_utility_path_aux_v1"
        and row["role"] == "test"
        and row["score_name"] == "pred_decision_score"
    )

    assert test_row["seed_count"] == 3
    assert test_row["rank_ic_min"] < 0.0
    assert test_row["rank_ic_mean"] < 1.0
    assert test_row["all_seed_rank_ic_spread_hit_positive"] is False
    assert test_row["stage3_weak_gate_pass"] is False


def test_score_variant_comparison_rows_include_existing_score_columns(tmp_path: Path) -> None:
    study = _study_with_seed(tmp_path, "path_aux_seed7", loss_profile="decision_utility_path_aux_v1", seed=7)

    report = build_output_aux_profile_comparison([study], run_tag="unit")
    rows = score_variant_comparison_rows(report)
    test_scores = {row["score_name"] for row in rows if row["role"] == "test"}

    assert {
        "pred_decision_score",
        "trade_utility_score",
        "hit_weighted_utility",
        "short_horizon_blend",
        "horizon_selected_utility",
        "decision_forecast_blend",
    }.issubset(test_scores)
    assert all(row["decision_score_source"] == "model_decision_utility" for row in rows)


def test_score_variant_comparison_rows_derives_missing_score_columns(tmp_path: Path) -> None:
    study = _study_with_seed(tmp_path, "path_aux_seed7", loss_profile="decision_utility_path_aux_v1", seed=7)
    for name in ("forecast_predictions_validation.csv", "forecast_predictions_test.csv"):
        frame = pd.read_csv(study / name)
        frame = frame.drop(
            columns=[
                "hit_weighted_utility",
                "short_horizon_blend",
                "horizon_selected_utility",
                "decision_forecast_blend",
            ]
        )
        frame.to_csv(study / name, index=False)

    report = build_output_aux_profile_comparison([study], run_tag="unit")
    rows = score_variant_comparison_rows(report)
    test_scores = {row["score_name"] for row in rows if row["role"] == "test"}

    assert {
        "max_pred_utility",
        "horizon_selected_utility",
        "hit_weighted_utility",
        "short_horizon_blend",
        "forecast_5d_mu",
        "forecast_20d_mu",
        "decision_forecast_blend",
    }.issubset(test_scores)


def test_profile_aggregate_rows_keep_horizon_grids_separate(tmp_path: Path) -> None:
    studies = [
        _study_with_grid(
            tmp_path,
            "path_aux_full_seed7",
            loss_profile="decision_utility_path_aux_v1",
            seed=7,
            horizon_grid_name="full_1_2_3_5_8_10_15_20_30",
            horizons=(1, 2, 3, 5, 8, 10, 15, 20, 30),
        ),
        _study_with_grid(
            tmp_path,
            "path_aux_full_seed11",
            loss_profile="decision_utility_path_aux_v1",
            seed=11,
            horizon_grid_name="full_1_2_3_5_8_10_15_20_30",
            horizons=(1, 2, 3, 5, 8, 10, 15, 20, 30),
        ),
        _study_with_grid(
            tmp_path,
            "path_aux_sparse_seed7",
            loss_profile="decision_utility_path_aux_v1",
            seed=7,
            horizon_grid_name="sparse_long",
            horizons=(1, 3, 5, 10, 15, 20, 30, 45),
            score_sign=-1.0,
        ),
    ]

    report = build_output_aux_profile_comparison(studies, run_tag="unit")
    rows = [
        row
        for row in profile_aggregate_rows(report)
        if row["loss_profile"] == "decision_utility_path_aux_v1"
        and row["role"] == "test"
        and row["score_name"] == "pred_decision_score"
    ]

    assert {row["horizon_grid_key"] for row in rows} == {
        "full_1_2_3_5_8_10_15_20_30",
        "sparse_long",
    }
    assert {row["horizon_grid"] for row in rows} == {
        "1,2,3,5,8,10,15,20,30",
        "1,3,5,10,15,20,30,45",
    }


def test_daily_grid_long_horizon_share_counts_all_horizons_at_or_above_15(tmp_path: Path) -> None:
    study = _study_with_seed(
        tmp_path,
        "daily_grid_seed7",
        loss_profile="decision_utility_path_aux_v1",
        seed=7,
        horizons=tuple(range(1, 46)),
    )
    for name in ("forecast_predictions_validation.csv", "forecast_predictions_test.csv"):
        frame = pd.read_csv(study / name)
        frame["pred_best_horizon"] = [31 if idx % 2 == 0 else 44 for idx in range(len(frame))]
        frame.to_csv(study / name, index=False)

    report = build_output_aux_profile_comparison([study], run_tag="unit")
    row = next(
        item
        for item in profile_aggregate_rows(report)
        if item["role"] == "test"
        and item["score_name"] == "pred_decision_score"
        and item["loss_profile"] == "decision_utility_path_aux_v1"
    )

    assert row["long_horizon_share_mean"] == 1.0


def test_write_output_aux_profile_comparison_outputs_calibration_artifacts(tmp_path: Path) -> None:
    studies = [
        _study_with_seed(tmp_path, "utility_seed7", loss_profile="decision_utility_v1_baseline", seed=7),
        _study_with_seed(tmp_path, "path_aux_seed7", loss_profile="decision_utility_path_aux_v1", seed=7),
        _study_with_seed(tmp_path, "hit_risk_seed7", loss_profile="decision_utility_hit_risk_aux_v1", seed=7),
        _study_with_seed(tmp_path, "rank_aux_seed7", loss_profile="decision_utility_rank_aux_v1", seed=7),
        _study_with_seed(tmp_path, "path_only_seed7", loss_profile="forecast_path_v1_baseline", seed=7),
    ]
    report = build_output_aux_profile_comparison(studies, run_tag="unit")

    paths = write_output_aux_profile_comparison(report, tmp_path / "comparison")

    assert paths["profile_aggregate_csv"].name == "profile_aggregate.csv"
    assert paths["score_variant_comparison_csv"].name == "score_variant_comparison.csv"
    assert paths["next_stage_decision_json"].name == "next_stage_decision.json"
    assert paths["calibration_review_md"].name == "calibration_review.md"
    aggregate = pd.read_csv(paths["profile_aggregate_csv"])
    score_variants = pd.read_csv(paths["score_variant_comparison_csv"])
    decision = json.loads(paths["next_stage_decision_json"].read_text(encoding="utf-8"))
    review = paths["calibration_review_md"].read_text(encoding="utf-8")
    assert {"rank_ic_mean", "rank_ic_min", "stage3_weak_gate_pass"}.issubset(aggregate.columns)
    assert "short_horizon_blend" in set(score_variants["score_name"])
    assert decision["stage2_selected_loss_profiles"] == [
        "decision_utility_path_aux_v1",
        "decision_utility_v1_baseline",
    ]
    assert "forecast_path_v1_baseline" in decision["rejected_loss_profiles"]
    assert "path-only baseline rejected" in review
    assert "horizon chooser not solved" in review
