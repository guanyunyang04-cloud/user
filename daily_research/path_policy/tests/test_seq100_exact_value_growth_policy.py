from __future__ import annotations

import numpy as np
import pandas as pd

from daily_research.path_policy import seq100_exact_value_growth_policy as exact


def test_contract_freezes_exact_weights_scope_and_no_low_price_reward() -> None:
    study, _ = exact.load_study()

    assert study["score_contract"]["family_weights"] == {
        "valuation": 0.30,
        "earnings_and_revision": 0.30,
        "quality": 0.25,
        "price_confirmation": 0.10,
        "governance": 0.05,
    }
    assert tuple(study["signals"]["policies"]) == exact.POLICIES
    assert tuple(study["universe"]["allowed_prefixes"]) == exact.MAIN_BOARD_PREFIXES
    assert study["epistemic_contract"][
        "52_week_low_or_distance_from_low_reward_is_forbidden"
    ]
    assert study["decision_boundary"]["historical_account_replay_performed"] is False


def test_covered_revision_requires_positive_values_and_both_coverage_counts() -> None:
    result = exact._covered_revision(
        np.array([110.0, 110.0, -1.0, 110.0]),
        np.array([100.0, 100.0, 100.0, 0.0]),
        np.array([3, 2, 3, 3]),
        np.array([3, 3, 3, 3]),
        minimum_institutions=3,
    )

    assert result[0] == np.log(1.1)
    assert np.isnan(result[1:]).all()


def test_selection_respects_frozen_breadth_and_industry_cap() -> None:
    study, _ = exact.load_study()
    rows = 30
    frame = pd.DataFrame(
        {
            "row_position": np.arange(rows),
            "candidate_id": np.arange(rows),
            "date_idx": np.ones(rows, dtype=int),
            "trade_date": ["2025-01-31"] * rows,
            "symbol": [f"600{i:03d}.SH" for i in range(rows)],
            "evaluation_year": np.full(rows, 2025),
            "industry_code": np.repeat(np.arange(10), 3),
            "valuation_score": np.linspace(1.0, 0.0, rows),
            "earnings_revision_score": np.linspace(1.0, 0.0, rows),
            "quality_score_exact": np.linspace(1.0, 0.0, rows),
            "price_confirmation_score_exact": np.linspace(1.0, 0.0, rows),
            "governance_score": np.ones(rows),
            "exact_full_score": np.linspace(1.0, 0.0, rows),
            "exact_no_cycle_score": np.linspace(1.0, 0.0, rows),
            "exact_no_governance_score": np.linspace(1.0, 0.0, rows),
            "forward_value_score": np.linspace(1.0, 0.0, rows),
            "ttm_fcf_score": np.linspace(1.0, 0.0, rows),
        }
    )
    for policy in exact.POLICIES:
        frame[f"eligible__{policy}"] = True

    selections = exact._selection_frame(frame, study)
    top10 = selections.loc[selections["policy"].eq("exact_full_top10")]
    top20 = selections.loc[selections["policy"].eq("exact_full_top20")]

    assert len(top10) == 10
    assert top10.groupby("industry_code").size().max() <= 2
    assert len(top20) == 20
    assert top20.groupby("industry_code").size().max() <= 3


def test_gate_does_not_authorize_account_when_one_check_fails() -> None:
    study, _ = exact.load_study()
    periods = list(study["evaluation"]["periods"])
    summaries = []
    for period in [*periods, "full_history"]:
        summaries.append(
            {
                "period": period,
                "policy": "exact_full_top10",
                "stress_net": {
                    "mean": 0.01,
                    "lcb_95": 0.001,
                    "block": {"lcb_95": 0.001},
                },
                "industry_residual": {
                    "lcb_95": 0.001,
                    "block": {"lcb_95": 0.001},
                },
                "positive_year_count": 9,
                "excluding_best_year_mean": 0.001,
            }
        )
    paired = [
        {
            "control": control,
            "period": period,
            "primary_minus_control": {"mean": 0.001},
        }
        for control in ("forward_value_only_top10", "ttm_fcf_only_top10")
        for period in periods
    ]

    decision = exact._decision(summaries, paired, study)

    assert decision["checks"]["minimum_positive_years"] is False
    assert decision["information_gate_passed"] is False
    assert decision["account_replay_authorized"] is False
