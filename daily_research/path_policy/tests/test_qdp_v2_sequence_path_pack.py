from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from daily_research.path_policy.qdp_v2_sequence_path_pack import (
    DAILY_RAW_FEATURES,
    PATH_OHLCVA_FIELDS,
    PATH_SUMMARY_COLUMNS,
    _build_sample_index,
    _compute_future_path_and_masks,
    _fit_normalization,
    path_summary_columns,
    path_value_column,
)
from daily_research.path_policy.qdp_v2_sequence_flat_lgbm import _feature_names, _select_indices
from daily_research.path_policy.qdp_v2_sequence_path_training import SequencePathModel, SequencePathPackDataset, _compute_loss
from daily_research.path_policy.qdp_v2_sequence_path_training import (
    _derive_path_summary_numpy,
    _derive_path_summary_torch,
    _realize_predicted_plan_numpy,
    derived_path_summary_columns,
    path_value_v2_column,
    unified_path_value_column,
)


def _raw_panel(
    open_values: list[float],
    high_values: list[float],
    low_values: list[float],
    close_values: list[float],
    volume_values: list[float] | None = None,
    amount_values: list[float] | None = None,
) -> np.ndarray:
    panel = np.full((len(open_values), 1, len(DAILY_RAW_FEATURES)), np.nan, dtype=np.float32)
    volume_values = volume_values or [1000.0 + 10.0 * idx for idx in range(len(open_values))]
    amount_values = amount_values or [10000.0 + 100.0 * idx for idx in range(len(open_values))]
    for name, values in {
        "open": open_values,
        "high": high_values,
        "low": low_values,
        "close": close_values,
        "volume": volume_values,
        "amount": amount_values,
    }.items():
        panel[:, 0, DAILY_RAW_FEATURES.index(name)] = np.asarray(values, dtype=np.float32)
    panel[:, 0, DAILY_RAW_FEATURES.index("volume_log")] = np.log1p(np.asarray(volume_values, dtype=np.float32))
    panel[:, 0, DAILY_RAW_FEATURES.index("amount_log")] = np.log1p(np.asarray(amount_values, dtype=np.float32))
    panel[:, 0, DAILY_RAW_FEATURES.index("intraday_range_raw")] = np.asarray(high_values, dtype=np.float32) / np.asarray(
        low_values, dtype=np.float32
    ) - 1.0
    return panel


def test_future_path_anchors_on_next_calendar_trading_day_open() -> None:
    raw = _raw_panel(
        open_values=[8.0, 9.0, 10.0, 12.0, 13.0],
        high_values=[8.5, 9.5, 11.0, 14.0, 13.5],
        low_values=[7.5, 8.5, 9.5, 11.0, 12.5],
        close_values=[8.2, 9.2, 10.5, 13.0, 13.2],
    )
    up_limit = np.full((5, 1), np.nan, dtype=np.float32)

    future_path, future_ohlcva_path, summary, input_valid, entry_buyable, label_valid = _compute_future_path_and_masks(
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
    assert future_ohlcva_path.shape[-1] == len(PATH_OHLCVA_FIELDS)
    assert np.isclose(future_ohlcva_path[1, 0, 0, 0], future_path[1, 0, 0, 0])
    assert np.isfinite(future_ohlcva_path[1, 0, 0, 4])
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

    _future_path, _future_ohlcva_path, _summary, _input_valid, entry_buyable, label_valid = _compute_future_path_and_masks(
        raw_panel=raw,
        up_limit_panel=up_limit,
        lookback_days=2,
        forward_days=1,
    )

    assert not entry_buyable[1, 0]
    assert label_valid[1, 0]


def test_ohlcva_volume_amount_targets_use_signal_day_trailing_history() -> None:
    raw = _raw_panel(
        open_values=[10.0, 10.0, 10.0, 10.0, 10.0],
        high_values=[10.5, 10.5, 10.5, 10.5, 10.5],
        low_values=[9.5, 9.5, 9.5, 9.5, 9.5],
        close_values=[10.0, 10.0, 10.0, 10.0, 10.0],
        volume_values=[100.0, 100.0, 100.0, 10000.0, 10000.0],
        amount_values=[1000.0, 1000.0, 1000.0, 100000.0, 100000.0],
    )
    up_limit = np.full((5, 1), np.nan, dtype=np.float32)

    _future_path, future_ohlcva_path, _summary, _input_valid, _entry_buyable, label_valid = _compute_future_path_and_masks(
        raw_panel=raw,
        up_limit_panel=up_limit,
        lookback_days=2,
        forward_days=2,
    )

    assert label_valid[1, 0]
    trailing_volume = np.mean(np.log1p([100.0, 100.0]))
    trailing_amount = np.mean(np.log1p([1000.0, 1000.0]))
    assert np.isclose(future_ohlcva_path[1, 0, 0, 4], np.log1p(100.0) - trailing_volume)
    assert np.isclose(future_ohlcva_path[1, 0, 1, 4], np.log1p(10000.0) - trailing_volume)
    assert np.isclose(future_ohlcva_path[1, 0, 1, 5], np.log1p(100000.0) - trailing_amount)


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
    assert set(parts) == {"loss", "path_loss", "summary_loss", "richer_loss", "value_loss", "rank_loss", "residual_penalty"}


def test_sequence_path_attention_model_outputs_path_summary_and_score() -> None:
    model = SequencePathModel(
        input_dim=6,
        hidden_dim=8,
        layers=1,
        forward_days=60,
        summary_dim=9,
        dropout=0.0,
        model_type="gru_attention",
    )
    x = torch.randn(4, 100, 6)
    y_path = torch.randn(4, 60, 4) * 0.01
    y_summary = torch.randn(4, 9) * 0.01
    date_idx = torch.tensor([1, 1, 1, 1])

    out = model(x)
    loss, parts = _compute_loss(
        out,
        y_path,
        y_summary,
        date_idx,
        value_index=8,
        path_weight=0.20,
        summary_weight=0.20,
        value_weight=0.20,
        rank_weight=0.40,
        rank_max_per_side=2,
    )

    assert out["future_path"].shape == (4, 60, 4)
    assert out["path_summary"].shape == (4, 9)
    assert out["score"].shape == (4,)
    assert torch.isfinite(loss)
    assert parts["rank_loss"] >= 0.0


def test_path_value_v2_numpy_and_torch_match() -> None:
    path = np.zeros((2, 4, 4), dtype=np.float32)
    path[:, :, 0] = 0.0
    path[0, :, 1] = [0.02, 0.05, 0.04, 0.03]
    path[0, :, 2] = [-0.01, 0.00, 0.01, 0.00]
    path[0, :, 3] = [0.01, 0.04, 0.03, 0.02]
    path[1, :, 1] = [0.01, 0.02, 0.03, 0.06]
    path[1, :, 2] = [-0.02, -0.03, -0.01, 0.00]
    path[1, :, 3] = [0.00, 0.01, 0.02, 0.05]

    np_summary = _derive_path_summary_numpy(path)
    torch_summary = _derive_path_summary_torch(torch.from_numpy(path), smooth_value=False).detach().numpy()

    assert derived_path_summary_columns(4)[-1] == path_value_v2_column(4)
    assert np.allclose(np_summary, torch_summary, atol=1.0e-6)


def test_path_value_v2_penalizes_later_same_return() -> None:
    early = np.zeros((1, 4, 4), dtype=np.float32)
    late = np.zeros((1, 4, 4), dtype=np.float32)
    early[0, :, 1] = 0.05
    early[0, :, 2] = 0.0
    early[0, :, 3] = [0.05, 0.05, 0.05, 0.05]
    late[0, :, 1] = [0.00, 0.00, 0.00, 0.05]
    late[0, :, 2] = 0.0
    late[0, :, 3] = [0.00, 0.00, 0.00, 0.05]

    early_value = _derive_path_summary_numpy(early)[0, -1]
    late_value = _derive_path_summary_numpy(late)[0, -1]

    assert early_value > late_value


def test_path_value_v2_penalizes_drawdown_before_exit() -> None:
    smooth = np.zeros((1, 4, 4), dtype=np.float32)
    volatile = np.zeros((1, 4, 4), dtype=np.float32)
    smooth[0, :, 1] = 0.05
    smooth[0, :, 2] = 0.0
    smooth[0, :, 3] = [0.01, 0.03, 0.05, 0.05]
    volatile[0, :, 1] = 0.05
    volatile[0, :, 2] = [-0.10, -0.10, -0.10, -0.10]
    volatile[0, :, 3] = [0.01, 0.03, 0.05, 0.05]

    smooth_value = _derive_path_summary_numpy(smooth)[0, -1]
    volatile_value = _derive_path_summary_numpy(volatile)[0, -1]

    assert smooth_value > volatile_value


def test_unified_ohlcva_value_uses_entry_before_exit_and_penalizes_wait() -> None:
    path = np.zeros((1, 4, 6), dtype=np.float32)
    path[0, 0, 1] = 0.50
    path[0, 0, 2] = -0.20
    path[0, 0, 3] = 0.00
    path[0, 1:, 1] = [0.04, 0.06, 0.08]
    path[0, 1:, 2] = 0.00
    path[0, 1:, 3] = [0.03, 0.05, 0.07]

    summary = _derive_path_summary_numpy(path)
    cols = derived_path_summary_columns(4, path_dim=6)

    assert cols[-1] == unified_path_value_column(4)
    assert summary[0, cols.index("best_entry_day_4d")] < summary[0, cols.index("best_exit_day_4d")]
    # The same-day high on day 1 is large, but same-day low-buy/high-sell is disallowed.
    assert summary[0, cols.index("best_exit_day_4d")] > 1.0


def test_unified_ohlcva_value_penalizes_low_amount() -> None:
    liquid = np.zeros((1, 4, 6), dtype=np.float32)
    illiquid = np.zeros((1, 4, 6), dtype=np.float32)
    for arr in [liquid, illiquid]:
        arr[0, :, 1] = [0.02, 0.05, 0.08, 0.10]
        arr[0, :, 2] = 0.0
        arr[0, :, 3] = [0.01, 0.04, 0.07, 0.09]
    liquid[0, :, 5] = 1.0
    illiquid[0, :, 5] = -3.0

    liquid_value = _derive_path_summary_numpy(liquid)[0, -1]
    illiquid_value = _derive_path_summary_numpy(illiquid)[0, -1]

    assert liquid_value > illiquid_value


def test_realized_predicted_plan_uses_predicted_entry_exit_on_true_path() -> None:
    pred = np.zeros((1, 4, 6), dtype=np.float32)
    pred[0, :, 1] = [0.01, 0.04, 0.10, 0.08]
    pred[0, :, 2] = [0.00, -0.02, 0.02, 0.03]
    pred[0, :, 3] = [0.00, 0.03, 0.09, 0.07]
    true = pred.copy()
    pred_summary = _derive_path_summary_numpy(pred)

    realized = _realize_predicted_plan_numpy(pred_summary, true, forward_days=4)

    assert realized["realized_fill"][0] == 1.0
    assert np.isfinite(realized["realized_trade_return"][0])
    assert realized["realized_exit_day"][0] > realized["realized_entry_day"][0]


def test_gru_path_value_model_outputs_only_future_path_and_loss_uses_derived_score() -> None:
    model = SequencePathModel(input_dim=6, hidden_dim=8, layers=1, forward_days=20, summary_dim=9, dropout=0.0, model_type="gru_path_value")
    x = torch.randn(4, 100, 6)
    y_path = torch.randn(4, 20, 4) * 0.01
    y_summary = torch.randn(4, 9) * 0.01
    date_idx = torch.tensor([1, 1, 1, 1])

    out = model(x)
    loss, parts = _compute_loss(out, y_path, y_summary, date_idx, value_index=8)

    assert set(out) == {"future_path"}
    assert out["future_path"].shape == (4, 20, 4)
    assert torch.isfinite(loss)
    assert set(parts) == {"loss", "path_loss", "summary_loss", "richer_loss", "value_loss", "rank_loss", "residual_penalty"}


def test_gru_ohlcva_path_value_model_outputs_six_dim_path_and_unified_loss() -> None:
    model = SequencePathModel(
        input_dim=6,
        hidden_dim=8,
        layers=1,
        forward_days=20,
        summary_dim=9,
        dropout=0.0,
        model_type="gru_ohlcva_path_value",
    )
    x = torch.randn(4, 100, 6)
    y_path = torch.randn(4, 20, 6) * 0.01
    y_summary = torch.randn(4, 9) * 0.01
    date_idx = torch.tensor([1, 1, 1, 1])

    out = model(x)
    loss, parts = _compute_loss(out, y_path, y_summary, date_idx, value_index=8)

    assert set(out) == {"future_path"}
    assert out["future_path"].shape == (4, 20, 6)
    assert model.uses_ohlcva_path
    assert torch.isfinite(loss)
    assert parts["value_loss"] >= 0.0


def test_gru_path_value_symbol_model_uses_symbol_embedding() -> None:
    model = SequencePathModel(
        input_dim=6,
        hidden_dim=8,
        layers=1,
        forward_days=20,
        summary_dim=9,
        dropout=0.0,
        model_type="gru_path_value_symbol",
        symbol_count=3,
        symbol_embedding_dim=4,
    )
    x = torch.randn(4, 100, 6)
    symbol_idx = torch.tensor([0, 1, 2, 1])
    out = model(x, symbol_idx=symbol_idx)

    assert set(out) == {"future_path"}
    assert out["future_path"].shape == (4, 20, 4)
    assert model.symbol_embedding.weight.shape == (3, 4)


def test_gru_path_value_residual_model_outputs_residual_score() -> None:
    model = SequencePathModel(
        input_dim=6,
        hidden_dim=8,
        layers=1,
        forward_days=20,
        summary_dim=9,
        dropout=0.0,
        model_type="gru_path_value_residual",
    )
    x = torch.randn(4, 100, 6)
    y_path = torch.randn(4, 20, 4) * 0.01
    y_summary = torch.randn(4, 9) * 0.01
    date_idx = torch.tensor([1, 1, 1, 1])

    out = model(x)
    loss, parts = _compute_loss(
        out,
        y_path,
        y_summary,
        date_idx,
        value_index=8,
        residual_weight=0.25,
        residual_penalty_weight=0.01,
    )

    assert set(out) == {"future_path", "residual_score"}
    assert out["future_path"].shape == (4, 20, 4)
    assert out["residual_score"].shape == (4,)
    assert torch.isfinite(loss)
    assert parts["residual_penalty"] >= 0.0


def test_gru_richer_path_value_model_outputs_richer_path_and_uses_auxiliary_loss() -> None:
    model = SequencePathModel(
        input_dim=6,
        hidden_dim=8,
        layers=1,
        forward_days=20,
        summary_dim=9,
        dropout=0.0,
        model_type="gru_richer_path_value",
        richer_path_dim=12,
    )
    x = torch.randn(4, 100, 6)
    y_path = torch.randn(4, 20, 4) * 0.01
    y_richer_path = torch.randn(4, 20, 12) * 0.01
    y_richer_path[:, :, :4] = y_path
    y_summary = torch.randn(4, 9) * 0.01
    date_idx = torch.tensor([1, 1, 1, 1])

    out = model(x)
    loss, parts = _compute_loss(
        out,
        y_path,
        y_summary,
        date_idx,
        y_richer_path=y_richer_path,
        value_index=8,
        richer_weight=0.10,
    )

    assert set(out) == {"future_path", "future_richer_path"}
    assert out["future_path"].shape == (4, 20, 4)
    assert out["future_richer_path"].shape == (4, 20, 12)
    assert torch.isfinite(loss)
    assert parts["richer_loss"] >= 0.0


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
    names = _feature_names(dataset)
    assert len(names) == 15
    assert names[0] == "t-2__daily_raw__daily_raw_0"
    selected = _select_indices(dataset, samples_per_date=1, max_samples=0, seed=7)
    assert selected.shape == (1,)


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


def test_sequence_pack_dataset_can_read_ohlc_from_ohlcva_without_legacy_label(tmp_path) -> None:
    def write_memmap(path, array: np.ndarray) -> None:
        mm = np.memmap(path, dtype="float32", mode="w+", shape=array.shape)
        mm[:] = array[:]
        mm.flush()

    panels = tmp_path / "panels_ohlcva_only"
    labels = tmp_path / "labels_ohlcva_only"
    panels.mkdir()
    labels.mkdir()
    feature_channels = {}
    for name in ["daily_raw", "daily_state", "intraday_summary", "limit_structure"]:
        values = np.ones((4, 1, 1), dtype=np.float32)
        path = panels / f"{name}.float32.dat"
        write_memmap(path, values)
        feature_channels[name] = {"path": str(path), "shape": list(values.shape), "columns": [f"{name}_0"]}
    future_ohlcva = np.zeros((4, 1, 20, 6), dtype=np.float32)
    future_ohlcva[:, :, :, 3] = 0.05
    path_summary = np.ones((4, 1, len(PATH_SUMMARY_COLUMNS)), dtype=np.float32)
    future_ohlcva_path = labels / "future_ohlcva_path.float32.dat"
    path_summary_path = labels / "path_summary.float32.dat"
    write_memmap(future_ohlcva_path, future_ohlcva)
    write_memmap(path_summary_path, path_summary)
    sample_index_path = tmp_path / "sample_index_ohlcva_only.parquet"
    pd.DataFrame(
        [{"split": "train", "date_idx": 2, "symbol_idx": 0, "trade_date": "2024-01-03", "symbol": "000001.SZ"}]
    ).to_parquet(sample_index_path, index=False)
    manifest = {
        "lookback_days": 3,
        "forward_days": 20,
        "sample_index_path": str(sample_index_path),
        "feature_channels": feature_channels,
        "label_arrays": {
            "future_ohlcva_path": {"path": str(future_ohlcva_path), "shape": list(future_ohlcva.shape), "fields": PATH_OHLCVA_FIELDS},
            "path_summary": {"path": str(path_summary_path), "shape": list(path_summary.shape), "columns": PATH_SUMMARY_COLUMNS},
        },
        "normalization": {name: {"mean": [0.0], "std": [1.0]} for name in feature_channels},
    }

    dataset = SequencePathPackDataset(manifest, split="train")
    batch = dataset.get_batch([0])

    assert batch["y_path"].shape == (1, 20, 4)
    assert batch["y_ohlcva_path"].shape == (1, 20, 6)
    assert torch.equal(batch["y_path"][0, :, 3], torch.full((20,), 0.05))


def test_sequence_pack_dataset_reads_ohlcva_label_shards(tmp_path) -> None:
    def write_memmap(path, array: np.ndarray) -> None:
        mm = np.memmap(path, dtype="float32", mode="w+", shape=array.shape)
        mm[:] = array[:]
        mm.flush()

    panels = tmp_path / "panels_sharded"
    labels = tmp_path / "labels_sharded"
    panels.mkdir()
    labels.mkdir()
    feature_channels = {}
    for name in ["daily_raw", "daily_state", "intraday_summary", "limit_structure"]:
        values = np.ones((6, 1, 1), dtype=np.float32)
        path = panels / f"{name}.float32.dat"
        write_memmap(path, values)
        feature_channels[name] = {"path": str(path), "shape": list(values.shape), "columns": [f"{name}_0"]}
    shard0 = np.zeros((3, 1, 20, 6), dtype=np.float32)
    shard1 = np.zeros((3, 1, 20, 6), dtype=np.float32)
    shard1[:, :, :, 3] = 0.12
    shard0_path = labels / "future_ohlcva_path.000000_000002.float32.dat"
    shard1_path = labels / "future_ohlcva_path.000003_000005.float32.dat"
    write_memmap(shard0_path, shard0)
    write_memmap(shard1_path, shard1)
    path_summary = np.ones((6, 1, len(PATH_SUMMARY_COLUMNS)), dtype=np.float32)
    path_summary_path = labels / "path_summary.float32.dat"
    write_memmap(path_summary_path, path_summary)
    sample_index_path = tmp_path / "sample_index_sharded.parquet"
    pd.DataFrame(
        [{"split": "train", "date_idx": 4, "symbol_idx": 0, "trade_date": "2024-01-05", "symbol": "000001.SZ"}]
    ).to_parquet(sample_index_path, index=False)
    manifest = {
        "lookback_days": 3,
        "forward_days": 20,
        "sample_index_path": str(sample_index_path),
        "feature_channels": feature_channels,
        "label_arrays": {
            "future_ohlcva_path": {
                "shape": [6, 1, 20, 6],
                "fields": PATH_OHLCVA_FIELDS,
                "shards": [
                    {"path": str(shard0_path), "date_start_idx": 0, "date_end_idx": 2, "shape": list(shard0.shape)},
                    {"path": str(shard1_path), "date_start_idx": 3, "date_end_idx": 5, "shape": list(shard1.shape)},
                ],
            },
            "path_summary": {"path": str(path_summary_path), "shape": list(path_summary.shape), "columns": PATH_SUMMARY_COLUMNS},
        },
        "normalization": {name: {"mean": [0.0], "std": [1.0]} for name in feature_channels},
    }

    dataset = SequencePathPackDataset(manifest, split="train")
    batch = dataset.get_batch([0])

    assert batch["y_ohlcva_path"].shape == (1, 20, 6)
    assert torch.equal(batch["y_path"][0, :, 3], torch.full((20,), 0.12))
