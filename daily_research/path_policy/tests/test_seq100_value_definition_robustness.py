from __future__ import annotations

import numpy as np
import pandas as pd

from daily_research.path_policy import seq100_value_definition_robustness as robustness


def test_contract_uses_only_transparent_posthoc_value_variants() -> None:
    study, _ = robustness.load_study()

    assert tuple(study["variants"]) == robustness.VARIANTS
    assert study["epistemic_contract"][
        "all_variants_are_posthoc_retrospective_sensitivity_tests"
    ]
    assert study["epistemic_contract"][
        "52_week_low_or_distance_from_low_reward_is_forbidden"
    ]


def test_variant_selection_keeps_all_declared_definitions() -> None:
    features = pd.DataFrame(
        {
            "row_position": np.arange(4),
            "candidate_id": np.arange(4),
            "date_idx": np.ones(4, dtype=int),
            "trade_date": ["2020-01-31"] * 4,
            "symbol": [f"00000{x}.SZ" for x in range(4)],
            "evaluation_year": [2020] * 4,
            "industry_code": np.arange(4),
            "signed_log_pe": np.log1p([5.0, 10.0, 20.0, 30.0]),
            "signed_log_pb": np.log1p([0.5, 1.0, 2.0, 3.0]),
            "log_total_market_value": np.log1p([100.0] * 4),
            "cashflow_free_cash_flow": [4.0, 3.0, 2.0, 1.0],
            "value_score": [1.0, 0.75, 0.5, 0.25],
        }
    )

    result = robustness._variant_selections(
        features,
        top_k=2,
        industry_cap=1,
        minimum_group_size=2,
    )

    assert set(result["variant"]) == set(robustness.VARIANTS)
    assert result.groupby("variant").size().eq(2).all()


def test_decision_requires_every_definition_and_recent_period() -> None:
    metrics = [
        {
            "terminal_multiple": 1.2,
            "annual": [
                {"year": 2023, "return": 0.01},
                {"year": 2024, "return": 0.02},
                {"year": 2025, "return": 0.03},
            ],
        },
        {
            "terminal_multiple": 0.9,
            "annual": [
                {"year": 2023, "return": 0.01},
                {"year": 2024, "return": 0.02},
                {"year": 2025, "return": 0.03},
            ],
        },
    ]

    assert robustness._decision(metrics)["retrospective_robustness_passed"] is False
