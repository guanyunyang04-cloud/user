from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd

from daily_research.path_policy import seq100_margin_top10_current_cross as study_module


def test_frozen_study_loads() -> None:
    study, path, base_study = study_module.load_study()

    assert path.is_file()
    assert study["study_id"] == study_module.STUDY_ID
    assert study["selection"]["minimum_consecutive_increases"] == 2
    assert base_study["study_id"] == "seq100_margin_top10_adaptive_entry_v1"


def test_current_cross_uses_current_and_previous_valid_bars() -> None:
    pack = SimpleNamespace(
        exit_close_raw=np.asarray(
            [
                [9.0, 9.0],
                [9.0, 9.0],
                [11.0, 10.5],
            ]
        )
    )
    factor = np.ones((3, 2), dtype=np.float32)
    adp = np.asarray(
        [
            [10.0, 10.0],
            [10.0, 10.0],
            [10.0, 10.2],
        ]
    )
    candidates = pd.DataFrame(
        {
            "candidate_id": [1, 2],
            "symbol_idx": [0, 1],
            "date_idx": [2, 2],
        }
    )

    result = study_module._attach_current_cross(
        candidates, pack=pack, factor=factor, adp=adp
    )

    assert result["current_cross"].tolist() == [True, True]
    assert result["current_cross_rising"].tolist() == [False, True]
    assert bool((result["prior_close_to_adp"] < 0.0).all())


def test_daily_summary_keeps_unfilled_candidate_as_cash() -> None:
    frame = pd.DataFrame(
        {
            "candidate_id": [1, 2],
            "date_idx": [10, 10],
            "trade_date": ["2025-01-02", "2025-01-02"],
            "evaluation_year": [2025, 2025],
            "signal_close_return_d1": [0.01, -0.01],
            "entry_to_close_return_d1": [0.02, -0.02],
            "next_close_up": [True, False],
            "one_day_gross_adjusted_return": [0.01, np.nan],
            "one_day_net_return": [0.005, np.nan],
        }
    )

    daily = study_module._daily_summary(frame, np.ones(2, dtype=bool))

    assert daily.loc[0, "selected_count"] == 2
    assert daily.loc[0, "filled_count"] == 1
    assert daily.loc[0, "next_up_fraction"] == 0.5
    assert daily.loc[0, "legal_d2_cash_net_return"] == 0.0025
