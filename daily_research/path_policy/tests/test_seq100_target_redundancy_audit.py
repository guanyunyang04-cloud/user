from __future__ import annotations

import numpy as np
from scipy import stats

from daily_research.path_policy import seq100_short_horizon_target_reaudit as source
from daily_research.path_policy import seq100_target_redundancy_audit as audit


def test_load_study_freezes_recent_folds_and_2026_firewall() -> None:
    study = audit.load_study()

    assert study["contract_sha256"] == audit._canonical_json_sha256(
        study["contract"]
    )
    assert study["contract"]["protocol"]["fold_years"] == [2023, 2024, 2025]
    assert study["contract"]["protocol"]["new_booster_count"] == 0
    assert study["contract"]["scientific_firewall"]["forbidden_years"] == [2026]


def test_orthogonalization_preserves_independent_target_score() -> None:
    dates = np.repeat(np.arange(3, dtype=np.int32), 40)
    rows = np.arange(len(dates), dtype=np.int64)
    predictor = np.tile(np.linspace(-1.0, 1.0, 40), 3)
    independent = np.tile(np.cos(np.linspace(-np.pi, np.pi, 40)), 3)
    actual = independent.copy()
    predictor_bundle = source.PredictionBundle(
        year=2023,
        horizon=5,
        target="mfe",
        rows=rows,
        date_idx=dates,
        actual=actual,
        prediction=predictor,
        result={},
    )
    target_bundle = source.PredictionBundle(
        year=2023,
        horizon=10,
        target="mfe",
        rows=rows,
        date_idx=dates,
        actual=actual,
        prediction=2.0 * predictor + independent,
        result={},
    )

    result = audit.orthogonalize_score(
        target_bundle=target_bundle,
        predictor_bundles=(predictor_bundle,),
        ridge_penalty=1.0e-3,
    )

    assert result.score_explained_r2 > 0.7
    assert stats.spearmanr(result.residual_score, actual).statistic > 0.95


def test_residual_gate_separates_core_secondary_and_omit() -> None:
    thresholds = {
        "minimum_strong_residual_rank_ic": 0.01,
        "core_minimum_strong_years": 2,
        "secondary_minimum_positive_years": 2,
        "secondary_minimum_positive_tail_years": 2,
        "secondary_worst_rank_ic": -0.01,
    }

    def classify(rank_ic: tuple[float, float, float], tail: tuple[float, float, float]) -> str:
        rows = [
            {"rank_ic": current_ic, "daily_tail_top5_lift": current_tail}
            for current_ic, current_tail in zip(rank_ic, tail)
        ]
        return audit._classify_residual(rows, thresholds)["status"]

    assert classify((0.02, 0.03, 0.04), (0.01, 0.01, 0.01)) == "core"
    assert classify((0.005, -0.005, 0.02), (0.01, -0.01, 0.01)) == "secondary"
    assert classify((-0.02, -0.01, 0.03), (-0.01, -0.01, 0.01)) == "omit"


def test_probability_increment_gate_requires_stable_improvement() -> None:
    thresholds = {"secondary_maximum_relative_deterioration": 0.005}
    supported = audit._classify_probability_increment(
        [
            {"relative_brier_improvement": 0.01, "relative_logloss_improvement": 0.02},
            {"relative_brier_improvement": 0.02, "relative_logloss_improvement": 0.01},
        ],
        thresholds,
    )
    conditional = audit._classify_probability_increment(
        [
            {"relative_brier_improvement": 0.01, "relative_logloss_improvement": 0.01},
            {"relative_brier_improvement": -0.001, "relative_logloss_improvement": -0.002},
        ],
        thresholds,
    )
    unsupported = audit._classify_probability_increment(
        [
            {"relative_brier_improvement": 0.01, "relative_logloss_improvement": 0.01},
            {"relative_brier_improvement": -0.02, "relative_logloss_improvement": -0.02},
        ],
        thresholds,
    )

    assert supported["status"] == "supported"
    assert conditional["status"] == "conditional"
    assert unsupported["status"] == "unsupported"


def test_daily_percentile_rank_is_cross_sectional() -> None:
    dates = np.asarray([1, 1, 1, 2, 2], dtype=np.int32)
    values = np.asarray([3.0, 1.0, 2.0, 5.0, 4.0])

    ranked = audit._daily_percentile_rank(dates, values)

    assert ranked.tolist() == [5.0 / 6.0, 1.0 / 6.0, 0.5, 0.75, 0.25]
