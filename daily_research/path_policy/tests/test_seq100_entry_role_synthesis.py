from __future__ import annotations

import numpy as np
import pytest

from daily_research.path_policy import seq100_entry_role_synthesis as audit


def _synthetic_actual() -> tuple[audit.PathActual, np.ndarray]:
    dates = np.repeat(np.arange(3, dtype=np.int32), 100)
    within = np.tile(np.arange(100, dtype=np.float64), 3)
    target = within / 100.0
    adverse = -0.20 + within / 1000.0
    state = np.tile(np.repeat(np.arange(3, dtype=np.int8), [34, 33, 33]), 3)
    return (
        audit.PathActual(
            rows=np.arange(len(dates), dtype=np.int64),
            date_idx=dates,
            target_mfe=target.astype(np.float32),
            early_mfe=(target * 0.5).astype(np.float32),
            peak_day=np.full(len(dates), 5.0, dtype=np.float32),
            pre_peak_mae=adverse.astype(np.float32),
            endpoint_return=(target * 0.5).astype(np.float32),
            state=state,
            mfe_tail_event=audit._within_date_event(
                date_idx=dates, values=target, fraction=0.20, upper=True
            ),
            deep_adverse_event=audit._within_date_event(
                date_idx=dates, values=adverse, fraction=0.20, upper=False
            ),
        ),
        within,
    )


def test_study_is_read_only_and_forbids_fusion() -> None:
    study = audit.load_study()

    assert study["folds"]["fold_years"] == [2023, 2024, 2025]
    assert study["folds"]["maximum_outcome_date"] == "2025-12-31"
    assert study["models"]["train_new_boosters"] is False
    assert study["decision"]["no_fusion"] is True


def test_compact_role_metrics_reads_tail_rate_from_path_summary() -> None:
    compact = audit._compact_role_metrics(
        {
            "broad_ranking": {
                "rank_ic_mean": 0.1,
                "top_1pct_mean": 0.2,
                "top_1pct_lift": 0.1,
                "top_5pct_mean": 0.15,
                "top_5pct_lift": 0.05,
            },
            "upper_tail": {
                "daily_tail_top1_lift": 0.3,
                "daily_tail_top5_lift": 0.2,
            },
            "path_quality": {
                "top1_daily_tail_rate": 0.5,
                "top5_daily_tail_rate": 0.4,
            },
        }
    )

    assert compact["top1_mfe_tail_rate"] == pytest.approx(0.5)
    assert compact["top5_mfe_tail_rate"] == pytest.approx(0.4)


def test_within_date_events_use_each_cross_section() -> None:
    dates = np.repeat(np.array([1, 2], dtype=np.int32), [5, 10])
    values = np.r_[np.arange(5), np.arange(10)].astype(np.float64)

    upper = audit._within_date_event(
        date_idx=dates, values=values, fraction=0.20, upper=True
    )
    lower = audit._within_date_event(
        date_idx=dates, values=values, fraction=0.20, upper=False
    )

    np.testing.assert_array_equal(np.flatnonzero(upper), [4, 13, 14])
    np.testing.assert_array_equal(np.flatnonzero(lower), [0, 5, 6])


def test_relationship_reports_overlap_and_reciprocal_discrimination() -> None:
    actual, within = _synthetic_actual()
    challenger = within.copy()
    challenger[np.arange(len(challenger)) % 100 >= 95] += np.tile(
        np.linspace(-3.0, 3.0, 5), 3
    )

    frame = audit._daily_model_relationship(
        actual=actual,
        left_name="baseline",
        left_score=within,
        right_name="challenger",
        right_score=challenger,
    )

    assert len(frame) == 3
    assert frame["top5_overlap_rate"].between(0.0, 1.0).all()
    assert frame["challenger_within_baseline_top5_high_minus_low_mfe_mean"].mean() > 0.0


def test_conditional_curve_and_grid_are_nested_without_fusion() -> None:
    actual, within = _synthetic_actual()

    curve, grid = audit._conditional_daily_frames(
        actual=actual,
        host_score=within,
        auxiliary_score=within,
        top_fractions=(0.05,),
        retention_fractions=(0.5, 1.0),
        quantile_count=5,
    )
    summary = audit._aggregate_conditional_curve(curve, horizon=10)
    half = next(row for row in summary if row["retention_fraction"] == 0.5)
    grid_summary = audit._aggregate_conditional_grid(
        grid, horizon=10, quantile_count=5
    )[0]

    assert half["selected_count_mean"] == pytest.approx(3.0)
    assert half["metrics"]["mfe_mean"]["mean_delta"] > 0.0
    assert grid_summary["highest_minus_lowest"]["mfe_mean"]["mean_delta"] > 0.0


def test_feature_role_rules_keep_strong_tail_and_broad_rank_separate() -> None:
    study = audit.load_study()
    settings = {
        "turnover_cost_proxy": (0.0005, 0.002, 0.01, 0.003),
        "breakout_retest_levels": (0.0004, 0.003, 0.01, 0.004),
        "traditional_indicators": (0.004, -0.002, -0.01, -0.002),
    }
    horizons = {
        "turnover_cost_proxy": 10,
        "breakout_retest_levels": 20,
        "traditional_indicators": 20,
    }
    paired = []
    relationships = []
    for challenger, (rank_ic, top5, tail, exclusive) in settings.items():
        horizon = horizons[challenger]
        for year in audit.FOLD_YEARS:
            paired.append(
                {
                    "year": year,
                    "horizon": horizon,
                    "challenger": challenger,
                    "metrics": {
                        "rank_ic": {"mean_delta": rank_ic},
                        "top_5pct_lift": {"mean_delta": top5},
                        "daily_tail_top5_lift": {"mean_delta": tail},
                    },
                }
            )
            relationships.append(
                {
                    "year": year,
                    "horizon": horizon,
                    "left": "baseline",
                    "right": challenger,
                    "summary": {
                        "metrics": {
                            "top5_right_only_minus_left_only_mfe_mean": {
                                "mean": exclusive
                            }
                        }
                    },
                }
            )

    roles = audit._compile_feature_roles(
        study=study, paired=paired, relationships=relationships
    )

    assert roles["turnover_cost_proxy"]["replace_baseline"] is True
    assert roles["breakout_retest_levels"]["replace_baseline"] is True
    assert roles["traditional_indicators"]["role"] == "broad_ranking_only"
    assert roles["traditional_indicators"]["replace_baseline"] is False


def test_conditional_coordinate_requires_path_gain_and_opportunity_preservation() -> (
    None
):
    study = audit.load_study()

    def record(year: int, mfe_ratio: float) -> dict[str, object]:
        curve_metrics = {metric: {"mean_delta": 0.0} for metric in audit.PATH_METRICS}
        curve_metrics["mfe_tail_rate"]["mean_delta"] = -0.01
        curve_metrics["pre_peak_mae_mean"]["mean_delta"] = 0.01
        curve_metrics["deep_adverse_rate"]["mean_delta"] = -0.05
        grid_metrics = {metric: {"mean_delta": 0.0} for metric in audit.PATH_METRICS}
        grid_metrics["pre_peak_mae_mean"]["mean_delta"] = 0.02
        grid_metrics["deep_adverse_rate"]["mean_delta"] = -0.10
        return {
            "year": year,
            "horizon": 10,
            "host_variant": "turnover_cost_proxy",
            "auxiliary": "pre_peak_mae_10",
            "curve": [
                {
                    "mfe_top_fraction": 0.05,
                    "retention_fraction": 0.5,
                    "mfe_mean_retention_ratio": mfe_ratio,
                    "metrics": curve_metrics,
                }
            ],
            "grid": [
                {
                    "mfe_top_fraction": 0.05,
                    "highest_minus_lowest": grid_metrics,
                }
            ],
        }

    supported = audit._classify_conditional_coordinate(
        study=study,
        records=[record(year, 0.99) for year in audit.FOLD_YEARS],
        auxiliary="pre_peak_mae_10",
        horizon=10,
        host_variant="turnover_cost_proxy",
    )
    tradeoff = audit._classify_conditional_coordinate(
        study=study,
        records=[record(year, 0.90) for year in audit.FOLD_YEARS],
        auxiliary="pre_peak_mae_10",
        horizon=10,
        host_variant="turnover_cost_proxy",
    )

    assert supported["status"] == "conditional_filter_supported"
    assert tradeoff["status"] == "informative_but_opportunity_tradeoff"


def test_state_projections_are_judged_by_their_declared_path_role() -> None:
    study = audit.load_study()

    def record(
        *, year: int, auxiliary: str, endpoint_delta: float
    ) -> dict[str, object]:
        curve_metrics = {metric: {"mean_delta": 0.0} for metric in audit.PATH_METRICS}
        curve_metrics["mfe_tail_rate"]["mean_delta"] = -0.01
        curve_metrics["endpoint_return_mean"]["mean_delta"] = endpoint_delta
        curve_metrics["state_expected_mean"]["mean_delta"] = 0.05
        curve_metrics["high_state_rate"]["mean_delta"] = 0.03
        grid_metrics = {metric: {"mean_delta": 0.0} for metric in audit.PATH_METRICS}
        grid_metrics["endpoint_return_mean"]["mean_delta"] = endpoint_delta * 2.0
        grid_metrics["state_expected_mean"]["mean_delta"] = 0.10
        grid_metrics["high_state_rate"]["mean_delta"] = 0.06
        return {
            "year": year,
            "horizon": 10,
            "host_variant": "turnover_cost_proxy",
            "auxiliary": auxiliary,
            "curve": [
                {
                    "mfe_top_fraction": 0.05,
                    "retention_fraction": 0.5,
                    "mfe_mean_retention_ratio": 0.99,
                    "metrics": curve_metrics,
                }
            ],
            "grid": [
                {
                    "mfe_top_fraction": 0.05,
                    "highest_minus_lowest": grid_metrics,
                }
            ],
        }

    expected = audit._classify_conditional_coordinate(
        study=study,
        records=[
            record(
                year=year,
                auxiliary="state_expected_value_10",
                endpoint_delta=0.01,
            )
            for year in audit.FOLD_YEARS
        ],
        auxiliary="state_expected_value_10",
        horizon=10,
        host_variant="turnover_cost_proxy",
    )
    high_probability = audit._classify_conditional_coordinate(
        study=study,
        records=[
            record(
                year=year,
                auxiliary="state_high_probability_10",
                endpoint_delta=-0.01,
            )
            for year in audit.FOLD_YEARS
        ],
        auxiliary="state_high_probability_10",
        horizon=10,
        host_variant="turnover_cost_proxy",
    )

    assert expected["status"] == "conditional_filter_supported"
    assert high_probability["status"] == "not_supported_as_pre_entry_filter"
