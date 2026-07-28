from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from daily_research.path_policy import (
    seq100_post_entry_incremental_information as audit,
)


def test_study_freezes_contract_and_forbids_policy_selection() -> None:
    study = audit.load_study()

    assert study["folds"]["fold_years"] == [2023, 2024, 2025]
    assert study["folds"]["maximum_outcome_date"] == "2025-12-31"
    assert study["landmarks"]["ages"] == [1, 3, 5]
    assert study["entry_contract"]["frozen"] is True
    assert study["entry_contract"]["fusion"] is False
    assert study["targets"]["holding_target_requires_new_buy_fill"] is False
    assert set(study["targets"]["matched_entry_diagnostic"]) == {
        "mfe_10",
        "mfe_20",
        "pre_peak_mae_10",
        "pre_peak_mae_20",
    }
    assert study["evaluation"]["baseline_metrics"] == [
        "daily Spearman Rank IC",
        "top 5% target lift",
        "target mean",
    ]
    assert "formal_holding_model" in study["non_selections"]


def test_holding_mfe_includes_legal_d1_but_matched_entry_does_not() -> None:
    close = np.asarray(
        [
            [0.20, 0.05, 0.10],
            [-0.05, -0.10, -0.02],
            [0.10, 0.30, 0.20],
        ],
        dtype=np.float64,
    )
    sellable = np.ones_like(close, dtype=bool)
    valid = np.ones(3, dtype=bool)

    _, holding_mfe, holding_risk = audit._numeric_path_labels(
        close, sellable, valid, first_legal_day=1
    )
    _, entry_mfe, entry_risk = audit._numeric_path_labels(
        close, sellable, valid, first_legal_day=2
    )

    assert holding_mfe[0] == pytest.approx(0.20)
    assert entry_mfe[0] == pytest.approx(0.10)
    assert holding_risk[0] == pytest.approx(0.0)
    assert holding_mfe[1] == entry_mfe[1]
    assert entry_risk[1] == pytest.approx(-0.10)
    assert holding_mfe[2] == entry_mfe[2]


def test_candidate_lookup_preserves_missing_rows() -> None:
    source = np.asarray([100, 104, 109, 120], dtype=np.int64)

    rows, found = audit._candidate_lookup(
        source, np.asarray([99, 100, 105, 120, 121], dtype=np.int64)
    )

    np.testing.assert_array_equal(rows, [-1, 0, -1, 3, -1])
    np.testing.assert_array_equal(found, [False, True, False, True, False])


def test_partial_rank_removes_current_contract_and_keeps_increment() -> None:
    rng = np.random.default_rng(19)
    controls = rng.normal(size=(800, 4))
    incremental = rng.normal(size=800)
    target = (
        3.0 * controls[:, 0]
        - 2.0 * controls[:, 1]
        + incremental
        + rng.normal(scale=0.05, size=800)
    )

    result = audit._partial_rank_statistics(
        controls,
        incremental,
        target,
        minimum_rows=20,
        quantile_count=5,
    )

    assert result["row_count"] == 800
    assert result["partial_rank_ic"] > 0.70
    assert result["residual_quintile_spread"] > 0.0


def test_matrix_partial_rank_matches_single_coordinate_result() -> None:
    rng = np.random.default_rng(23)
    controls = rng.normal(size=(500, 3))
    feature = rng.normal(size=500)
    target = controls[:, 0] + 0.5 * feature + rng.normal(scale=0.1, size=500)
    single = audit._partial_rank_statistics(
        controls, feature, target, minimum_rows=20, quantile_count=5
    )

    counts, correlations, spreads = audit._partial_rank_matrix_statistics(
        controls,
        feature[:, None],
        target[:, None],
        minimum_rows=20,
        quantile_count=5,
    )

    assert counts[0, 0] == single["row_count"]
    assert correlations[0, 0] == pytest.approx(single["partial_rank_ic"])
    assert spreads[0, 0] == pytest.approx(single["residual_quintile_spread"])


def test_matrix_partial_rank_ignores_constant_coordinate() -> None:
    rng = np.random.default_rng(29)
    controls = rng.normal(size=(200, 2))
    features = np.column_stack([np.ones(200), rng.normal(size=200)])
    targets = rng.normal(size=(200, 1))

    counts, correlations, spreads = audit._partial_rank_matrix_statistics(
        controls,
        features,
        targets,
        minimum_rows=20,
        quantile_count=5,
    )

    assert counts[0, 0] == 0
    assert np.isnan(correlations[0, 0])
    assert np.isnan(spreads[0, 0])
    assert counts[1, 0] == 200


def _stable_row(
    *, block: str, target: str = "mfe_10", age: int = 3
) -> dict[str, object]:
    return {
        "age": age,
        "stratum": "entry_mfe10_top5",
        "target": target,
        "feature": "coordinate",
        "block": block,
        "partial_rank_ic_2023_2024_2025": [0.02, 0.02, 0.02],
        "p_value_2023_2024_2025": [0.01, 0.01, 0.01],
        "q_value_2023_2024_2025": [0.02, 0.02, 0.02],
        "same_sign_years": 3,
        "hac_supported_years": 3,
        "fdr_supported_years": 3,
        "worst_year_absolute_partial_ic": 0.02,
        "mean_absolute_partial_ic": 0.02,
        "sparse_event_coordinate": False,
        "stable": True,
    }


def test_low_date_coverage_is_reported_as_sparse_event() -> None:
    study = audit.load_study()
    rows = []
    for feature, coverage in (("dense", 1.0), ("sparse", 0.05)):
        for year in (2023, 2024, 2025):
            rows.append(
                {
                    "year": year,
                    "age": 3,
                    "stratum": "entry_mfe10_top5",
                    "target": "pre_peak_mae_10",
                    "feature": feature,
                    "block": "B3_activity_and_tradability",
                    "partial_rank_ic_mean": -0.02,
                    "partial_rank_ic_p_value": 0.01,
                    "partial_rank_ic_q_value": 0.02,
                    "valid_date_fraction": coverage,
                }
            )

    evidence = audit._stable_evidence(pd.DataFrame.from_records(rows), study)
    dense = evidence.loc[evidence["feature"] == "dense"].iloc[0]
    sparse = evidence.loc[evidence["feature"] == "sparse"].iloc[0]

    assert dense["stable"]
    assert not dense["sparse_event_coordinate"]
    assert not sparse["stable"]
    assert sparse["sparse_event_coordinate"]
    assert not sparse["coverage_supported"]


def test_decision_does_not_confuse_entry_memory_with_realized_path() -> None:
    study = audit.load_study()
    memory = pd.DataFrame.from_records([_stable_row(block="B1_entry_memory")])
    price = pd.DataFrame.from_records(
        [_stable_row(block="B2_realized_price_path", target="state_10")]
    )

    memory_decision = audit._build_decision(memory, study)
    path_decision = audit._build_decision(price, study)

    assert memory_decision["verdict"] == "entry_memory_only"
    assert memory_decision["dedicated_path_updater_candidate"] is False
    assert path_decision["verdict"] == "path_updates_state_or_risk_only"
    assert path_decision["dedicated_path_updater_candidate"] is True
