from __future__ import annotations

import hashlib
from types import SimpleNamespace

import pandas as pd
import pytest

from daily_research.path_policy.qdp_v2_sequence_path_training import (
    EVALUATION_MODE_FIXED_OOS,
    EVALUATION_MODE_STANDARD,
    TrainConfig,
    _topk_candidate_rows,
    _topk_daily_rows,
    _validate_evaluation_mode,
)


VALUE_COLUMN = "path_trade_value_60d"


def test_fixed_oos_mode_forbids_checkpoint_selection_by_early_stopping() -> None:
    assert TrainConfig.__dataclass_fields__["evaluation_mode"].default == EVALUATION_MODE_STANDARD
    assert (
        _validate_evaluation_mode(
            SimpleNamespace(evaluation_mode=EVALUATION_MODE_FIXED_OOS, early_stopping_patience=0)
        )
        == EVALUATION_MODE_FIXED_OOS
    )
    with pytest.raises(ValueError, match="early_stopping_patience=0"):
        _validate_evaluation_mode(
            SimpleNamespace(evaluation_mode=EVALUATION_MODE_FIXED_OOS, early_stopping_patience=1)
        )


def _topk_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "trade_date": ["2022-01-04"] * 5,
            "symbol": ["CCC", "AAA", "BBB", "DDD", "INVALID"],
            "score": [0.9, 0.9, 0.5, 0.1, float("nan")],
            VALUE_COLUMN: [0.30, 0.10, 0.20, -0.10, 99.0],
            "future_max_return_60d": [0.40, 0.20, 0.30, 0.00, 99.0],
            "future_min_return_60d": [-0.01, -0.02, -0.03, -0.10, -99.0],
            "future_final_return_60d": [0.20, 0.10, 0.15, -0.05, 99.0],
            "future_peak_day_60d": [8.0, 7.0, 6.0, 5.0, 1.0],
            "drawdown_after_peak_60d": [-0.02, -0.03, -0.04, -0.12, -99.0],
        }
    )


def test_topk_ties_counts_and_universe_hash_are_order_independent() -> None:
    frame = _topk_frame()
    expected_hash = hashlib.sha256("AAA\nBBB\nCCC\nDDD".encode("utf-8")).hexdigest()

    rows = _topk_daily_rows(
        frame,
        top_k_values=(1, 10),
        forward_days=60,
        value_column=VALUE_COLUMN,
    )
    shuffled_rows = _topk_daily_rows(
        frame.sample(frac=1.0, random_state=17),
        top_k_values=(1, 10),
        forward_days=60,
        value_column=VALUE_COLUMN,
    )

    assert rows == shuffled_rows
    assert [row["universe_count"] for row in rows] == [4, 4]
    assert [row["selected_count"] for row in rows] == [1, 4]
    assert {row["universe_hash"] for row in rows} == {expected_hash}
    # AAA wins the score tie with CCC by the stable symbol tie-break.
    assert rows[0][f"selected_{VALUE_COLUMN}"] == 0.10


def test_topk_candidate_rows_retain_deterministic_rank_and_audit_fields() -> None:
    frame = _topk_frame()
    expected_hash = hashlib.sha256("AAA\nBBB\nCCC\nDDD".encode("utf-8")).hexdigest()

    rows = _topk_candidate_rows(
        frame.sample(frac=1.0, random_state=23),
        max_top_k=3,
        forward_days=60,
        value_column=VALUE_COLUMN,
    )

    assert [row["symbol"] for row in rows] == ["AAA", "CCC", "BBB"]
    assert [row["score_rank"] for row in rows] == [1, 2, 3]
    assert {row["universe_count"] for row in rows} == {4}
    assert {row["universe_hash"] for row in rows} == {expected_hash}
    assert rows[0][VALUE_COLUMN] == 0.10
