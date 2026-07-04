from __future__ import annotations

import numpy as np
import torch

from daily_research.path_policy.qdp_v2_sequence_path_pack import (
    DAILY_RAW_FEATURES,
    PATH_SUMMARY_COLUMNS,
    _build_sample_index,
    _compute_future_path_and_masks,
    _fit_normalization,
)
from daily_research.path_policy.qdp_v2_sequence_path_training import SequencePathModel, _compute_loss


def _raw_panel(open_values: list[float], high_values: list[float], low_values: list[float], close_values: list[float]) -> np.ndarray:
    panel = np.full((len(open_values), 1, len(DAILY_RAW_FEATURES)), np.nan, dtype=np.float32)
    for name, values in {
        "open": open_values,
        "high": high_values,
        "low": low_values,
        "close": close_values,
    }.items():
        panel[:, 0, DAILY_RAW_FEATURES.index(name)] = np.asarray(values, dtype=np.float32)
    return panel


def test_future_path_anchors_on_next_calendar_trading_day_open() -> None:
    raw = _raw_panel(
        open_values=[8.0, 9.0, 10.0, 12.0, 13.0],
        high_values=[8.5, 9.5, 11.0, 14.0, 13.5],
        low_values=[7.5, 8.5, 9.5, 11.0, 12.5],
        close_values=[8.2, 9.2, 10.5, 13.0, 13.2],
    )
    up_limit = np.full((5, 1), np.nan, dtype=np.float32)

    future_path, summary, input_valid, entry_buyable, label_valid = _compute_future_path_and_masks(
        raw_panel=raw,
        up_limit_panel=up_limit,
        lookback_days=2,
        forward_days=2,
    )

    assert input_valid[1, 0]
    assert entry_buyable[1, 0]
    assert label_valid[1, 0]
    # Signal date index 1 enters at date index 2 open = 10.0.
    assert future_path[1, 0, 0, 0] == 0.0
    assert np.isclose(future_path[1, 0, 0, 1], 0.10)
    assert np.isclose(future_path[1, 0, 1, 3], 0.30)
    final_idx = PATH_SUMMARY_COLUMNS.index("future_final_return_20d")
    max_idx = PATH_SUMMARY_COLUMNS.index("future_max_return_20d")
    assert np.isclose(summary[1, 0, final_idx], 0.30)
    assert np.isclose(summary[1, 0, max_idx], 0.40)


def test_entry_buyable_blocks_next_open_limit_up() -> None:
    raw = _raw_panel(
        open_values=[8.0, 9.0, 10.0, 12.0],
        high_values=[8.5, 9.5, 10.5, 12.5],
        low_values=[7.5, 8.5, 9.5, 11.5],
        close_values=[8.2, 9.2, 10.2, 12.2],
    )
    up_limit = np.full((4, 1), np.nan, dtype=np.float32)
    up_limit[2, 0] = 10.0

    _future_path, _summary, _input_valid, entry_buyable, label_valid = _compute_future_path_and_masks(
        raw_panel=raw,
        up_limit_panel=up_limit,
        lookback_days=2,
        forward_days=1,
    )

    assert not entry_buyable[1, 0]
    assert label_valid[1, 0]


def test_sample_index_requires_input_entry_and_label_masks() -> None:
    dates = ["2024-01-02", "2024-01-03", "2024-01-04"]
    symbols = ["000001.SZ", "000002.SZ"]
    input_valid = np.array([[False, False], [True, True], [True, True]])
    entry_buyable = np.array([[True, True], [True, False], [True, True]])
    label_valid = np.array([[True, True], [True, True], [False, True]])

    sample_index = _build_sample_index(
        date_values=dates,
        symbol_values=symbols,
        start_date="2024-01-02",
        end_date="2024-01-04",
        train_years=(),
        validation_years=(2024,),
        test_years=(),
        input_valid=input_valid,
        entry_buyable=entry_buyable,
        label_valid=label_valid,
    )

    assert sample_index[["trade_date", "symbol"]].to_dict("records") == [
        {"trade_date": "2024-01-03", "symbol": "000001.SZ"},
        {"trade_date": "2024-01-04", "symbol": "000002.SZ"},
    ]


def test_normalization_uses_only_train_date_mask() -> None:
    panel = np.asarray(
        [
            [[1.0], [3.0]],
            [[5.0], [7.0]],
            [[1000.0], [2000.0]],
        ],
        dtype=np.float32,
    )
    stats = _fit_normalization(panel, np.asarray([True, True, False]))

    assert np.isclose(stats["mean"][0], 4.0)
    assert np.isclose(stats["std"][0], np.std([1.0, 3.0, 5.0, 7.0]))


def test_sequence_path_model_outputs_path_summary_and_score() -> None:
    model = SequencePathModel(input_dim=6, hidden_dim=8, layers=1, forward_days=20, summary_dim=9, dropout=0.0)
    x = torch.randn(4, 100, 6)
    y_path = torch.randn(4, 20, 4) * 0.01
    y_summary = torch.randn(4, 9) * 0.01
    date_idx = torch.tensor([1, 1, 1, 1])

    out = model(x)
    loss, parts = _compute_loss(out, y_path, y_summary, date_idx, value_index=8)

    assert out["future_path"].shape == (4, 20, 4)
    assert out["path_summary"].shape == (4, 9)
    assert out["score"].shape == (4,)
    assert torch.isfinite(loss)
    assert set(parts) == {"loss", "path_loss", "summary_loss", "value_loss", "rank_loss"}
