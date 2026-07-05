from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from daily_research.path_policy.qdp_v2_sequence_path_pack import (
    DAILY_RAW_FEATURES,
    PATH_SUMMARY_COLUMNS,
    _build_sample_index,
    _compute_future_path_and_masks,
    _fit_normalization,
    path_summary_columns,
    path_value_column,
)
from daily_research.path_policy.qdp_v2_sequence_path_training import SequencePathModel, SequencePathPackDataset, _compute_loss


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


def test_sequence_pack_dataset_get_batch_reads_date_grouped_windows(tmp_path) -> None:
    def write_memmap(path, array: np.ndarray) -> None:
        mm = np.memmap(path, dtype="float32", mode="w+", shape=array.shape)
        mm[:] = array[:]
        mm.flush()

    panels = tmp_path / "panels"
    labels = tmp_path / "labels"
    panels.mkdir()
    labels.mkdir()
    channel_specs = {
        "daily_raw": 2,
        "daily_state": 1,
        "intraday_summary": 1,
        "limit_structure": 1,
    }
    feature_channels = {}
    for channel_idx, (name, feature_count) in enumerate(channel_specs.items()):
        values = np.zeros((5, 2, feature_count), dtype=np.float32)
        for date_idx in range(5):
            for symbol_idx in range(2):
                values[date_idx, symbol_idx, :] = 100 * channel_idx + 10 * date_idx + symbol_idx
        path = panels / f"{name}.float32.dat"
        write_memmap(path, values)
        feature_channels[name] = {
            "path": str(path),
            "shape": list(values.shape),
            "columns": [f"{name}_{idx}" for idx in range(feature_count)],
        }

    future_path = np.ones((5, 2, 20, 4), dtype=np.float32)
    path_summary = np.ones((5, 2, len(PATH_SUMMARY_COLUMNS)), dtype=np.float32)
    future_path_path = labels / "future_ohlc_path.float32.dat"
    path_summary_path = labels / "path_summary.float32.dat"
    write_memmap(future_path_path, future_path)
    write_memmap(path_summary_path, path_summary)
    sample_index_path = tmp_path / "sample_index.parquet"
    pd.DataFrame(
        [
            {"split": "train", "date_idx": 2, "symbol_idx": 0, "trade_date": "2024-01-03", "symbol": "000001.SZ"},
            {"split": "train", "date_idx": 2, "symbol_idx": 1, "trade_date": "2024-01-03", "symbol": "000002.SZ"},
        ]
    ).to_parquet(sample_index_path, index=False)
    manifest = {
        "lookback_days": 3,
        "forward_days": 20,
        "sample_index_path": str(sample_index_path),
        "feature_channels": feature_channels,
        "label_arrays": {
            "future_ohlc_path": {"path": str(future_path_path), "shape": list(future_path.shape)},
            "path_summary": {"path": str(path_summary_path), "shape": list(path_summary.shape), "columns": PATH_SUMMARY_COLUMNS},
        },
        "normalization": {
            name: {"mean": [0.0] * feature_count, "std": [1.0] * feature_count}
            for name, feature_count in channel_specs.items()
        },
    }

    dataset = SequencePathPackDataset(manifest, split="train")
    batch = dataset.get_batch([0, 1])

    assert batch["x"].shape == (2, 3, 5)
    assert batch["y_path"].shape == (2, 20, 4)
    assert batch["y_summary"].shape == (2, len(PATH_SUMMARY_COLUMNS))
    assert batch["trade_date"] == ["2024-01-03", "2024-01-03"]
    # First channel, first feature: dates 0..2 for symbol 0 and 1.
    assert torch.equal(batch["x"][0, :, 0], torch.tensor([0.0, 10.0, 20.0]))
    assert torch.equal(batch["x"][1, :, 0], torch.tensor([1.0, 11.0, 21.0]))


def test_future_path_columns_support_sixty_day_horizon() -> None:
    cols = path_summary_columns(60)
    assert cols == [
        "future_max_return_60d",
        "future_min_return_60d",
        "future_final_return_60d",
        "future_peak_day_60d",
        "future_trough_day_60d",
        "drawdown_after_peak_60d",
        "time_above_zero_60d",
        "time_below_zero_60d",
        "path_trade_value_60d",
    ]
    assert path_value_column(60) == "path_trade_value_60d"


def test_sequence_pack_dataset_get_batch_supports_sixty_day_labels(tmp_path) -> None:
    def write_memmap(path, array: np.ndarray) -> None:
        mm = np.memmap(path, dtype="float32", mode="w+", shape=array.shape)
        mm[:] = array[:]
        mm.flush()

    panels = tmp_path / "panels60"
    labels = tmp_path / "labels60"
    panels.mkdir()
    labels.mkdir()
    channel_specs = {
        "daily_raw": 1,
        "daily_state": 1,
        "intraday_summary": 1,
        "limit_structure": 1,
    }
    feature_channels = {}
    for name, feature_count in channel_specs.items():
        values = np.ones((4, 1, feature_count), dtype=np.float32)
        path = panels / f"{name}.float32.dat"
        write_memmap(path, values)
        feature_channels[name] = {
            "path": str(path),
            "shape": list(values.shape),
            "columns": [f"{name}_{idx}" for idx in range(feature_count)],
        }
    columns = path_summary_columns(60)
    future_path = np.ones((4, 1, 60, 4), dtype=np.float32)
    path_summary = np.ones((4, 1, len(columns)), dtype=np.float32)
    future_path_path = labels / "future_ohlc_path.float32.dat"
    path_summary_path = labels / "path_summary.float32.dat"
    write_memmap(future_path_path, future_path)
    write_memmap(path_summary_path, path_summary)
    sample_index_path = tmp_path / "sample_index60.parquet"
    pd.DataFrame(
        [{"split": "train", "date_idx": 2, "symbol_idx": 0, "trade_date": "2024-01-03", "symbol": "000001.SZ"}]
    ).to_parquet(sample_index_path, index=False)
    manifest = {
        "lookback_days": 3,
        "forward_days": 60,
        "sample_index_path": str(sample_index_path),
        "feature_channels": feature_channels,
        "label_arrays": {
            "future_ohlc_path": {"path": str(future_path_path), "shape": list(future_path.shape)},
            "path_summary": {"path": str(path_summary_path), "shape": list(path_summary.shape), "columns": columns},
        },
        "normalization": {
            name: {"mean": [0.0] * feature_count, "std": [1.0] * feature_count}
            for name, feature_count in channel_specs.items()
        },
    }

    dataset = SequencePathPackDataset(manifest, split="train")
    batch = dataset.get_batch([0])

    assert dataset.value_column == "path_trade_value_60d"
    assert batch["x"].shape == (1, 3, 4)
    assert batch["y_path"].shape == (1, 60, 4)
    assert batch["y_summary"].shape == (1, len(columns))
