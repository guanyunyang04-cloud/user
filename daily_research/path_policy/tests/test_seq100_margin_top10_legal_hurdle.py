from __future__ import annotations

import pandas as pd

from daily_research.path_policy import seq100_margin_top10_legal_hurdle as study


def test_attach_scores_builds_all_frozen_variants() -> None:
    frame = pd.DataFrame(
        {
            "date_idx": [1, 1],
            "candidate__m2_direct_557_margin__legal_gross_return": [0.02, -0.01],
            "legal_net_positive": [0.7, 0.3],
            "legal_net_q20": [-0.01, -0.03],
        }
    )
    mapping = study.attach_scores(frame)
    assert set(mapping) == {
        "parent_expected_net",
        "legal_win_probability",
        "legal_net_q20",
        "equal_rank_composite",
    }
    assert frame.loc[0, "score__equal_rank_composite"] == 1.0


def test_composite_no_trade_gate_applies_to_each_selected_candidate() -> None:
    frame = pd.DataFrame(
        {
            "date_idx": [1, 1, 2, 2],
            "symbol": ["A", "B", "C", "D"],
            "score": [0.9, 0.8, 0.9, 0.8],
            "score__parent_expected_net": [0.01, -0.01, 0.01, 0.01],
            "score__legal_win_probability": [0.6, 0.7, 0.4, 0.6],
        }
    )
    selected = study._select(
        frame,
        score_name="equal_rank_composite",
        score_column="score",
        top_k=1,
        no_trade=True,
    )
    assert selected["symbol"].tolist() == ["A"]


def test_frozen_hurdle_contract_loads() -> None:
    loaded, path = study.load_study()
    assert path.is_file()
    assert loaded["study_id"] == study.STUDY_ID
    assert loaded["source"]["forbidden_year"] == 2026
