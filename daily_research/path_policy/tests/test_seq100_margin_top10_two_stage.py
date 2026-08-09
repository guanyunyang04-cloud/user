from __future__ import annotations

from itertools import pairwise

import numpy as np
import pandas as pd

from daily_research.path_policy import seq100_margin_top10_two_stage as study


def test_forward_folds_are_contiguous_and_purged() -> None:
    history_date_idx = np.arange(80, 100, dtype=np.int32)
    validation_date_idx = np.arange(100, 140, dtype=np.int32)
    date_idx = np.repeat(np.concatenate([history_date_idx, validation_date_idx]), 2)
    labels = np.repeat(
        np.asarray(
            [f"2019-12-{index + 1:02d}" for index in range(20)]
            + [f"2020-01-{index + 1:02d}" for index in range(20)]
            + [f"2021-01-{index + 1:02d}" for index in range(20)]
        ),
        2,
    )
    folds = study.build_forward_folds(
        date_idx=date_idx,
        trade_date=labels,
        validation_start_date="2020-01-01",
        validation_end_date="2021-01-20",
        fold_count=5,
        purge_days=2,
    )
    assert len(folds) == 5
    assert sum(int(fold["validation_date_count"]) for fold in folds) == 40
    for fold in folds:
        assert int(fold["training_maximum_date_idx"]) + 2 < int(
            fold["validation_start_date_idx"]
        )
    for left, right in pairwise(folds):
        assert int(left["validation_end_date_idx"]) + 1 == int(
            right["validation_start_date_idx"]
        )


def test_same_date_winsor_preserves_units_and_missingness() -> None:
    values = np.asarray(
        [0.01, 0.02, 5.0, *np.linspace(-0.1, 0.1, 20)], dtype=np.float32
    )
    valid = np.ones(len(values), dtype=bool)
    valid[1] = False
    dates = np.asarray([1, 1, 1, *([2] * 20)], dtype=np.int32)
    result = study._same_date_winsor(values, valid, dates)
    assert np.isnan(result[:3]).all()  # fewer than twenty valid rows on date one
    np.testing.assert_allclose(result[3:], values[3:])


def test_score_columns_builds_direction_legal_and_composite() -> None:
    frame = pd.DataFrame(
        {
            "date_idx": [1, 1],
            "margin_rank": [1, 2],
            "base__transparent_66__next_close_up": [0.6, 0.4],
            "base__transparent_66__legal_gross_return": [0.01, -0.01],
            "base__compact_557__next_close_up": [0.7, 0.3],
            "base__compact_557__legal_gross_return": [0.02, -0.02],
            "candidate__m2_direct_557_margin__next_close_up": [0.8, 0.2],
            "candidate__m2_direct_557_margin__legal_gross_return": [0.03, -0.03],
            "candidate__m3_two_stage__next_close_up": [0.9, 0.1],
            "candidate__m3_two_stage__legal_gross_return": [0.04, -0.04],
        }
    )
    mapping = study._score_columns(frame)
    assert set(mapping) == {
        "margin_rank",
        "m0_transparent_66",
        "m1_compact_557",
        "m2_direct_557_margin",
        "m3_two_stage",
    }
    assert frame.loc[0, "score__m3_two_stage__composite"] == 1.0
    assert frame.loc[1, "score__m3_two_stage__composite"] == 0.5


def test_no_trade_requires_positive_legal_score() -> None:
    frame = pd.DataFrame(
        {
            "date_idx": [1, 1, 2, 2],
            "symbol": ["A", "B", "C", "D"],
            "score": [-0.01, -0.02, 0.02, 0.01],
        }
    )
    selected = study._select_top(
        frame,
        score_column="score",
        top_k=1,
        no_trade=True,
        score_type="legal",
        variant="m3_two_stage",
    )
    assert selected["symbol"].tolist() == ["C"]


def test_frozen_study_contract_loads() -> None:
    loaded, path = study.load_study()
    assert path.is_file()
    assert loaded["study_id"] == study.STUDY_ID
    assert loaded["source"]["forbidden_year"] == 2026
