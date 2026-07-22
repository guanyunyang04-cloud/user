from __future__ import annotations

import json
import hashlib

import numpy as np
import pandas as pd
import pytest
import torch

import daily_research.path_policy.qdp_v2_sequence_path_pack as sequence_pack
import daily_research.path_policy.qdp_v2_sequence_path_training as sequence_training
from daily_research.path_policy.qdp_v2_raw_rising_path_atlas import (
    RAW_SIGNAL_COLUMNS,
    _add_raw_daily_signals,
)

from daily_research.path_policy.qdp_v2_sequence_path_pack import (
    AShareExecutionCostConfig,
    DAILY_RAW_FEATURES,
    DEFAULT_OUTPUT_ROOT,
    DateShardedFloatStore,
    ENTRY_RULE_OPEN_BELOW_LIMIT,
    PATH_OHLCVA_FIELDS,
    PATH_SUMMARY_COLUMNS,
    _apply_back_adjustment,
    _build_sample_index,
    _compute_future_path_and_masks,
    _exit_fill_mask,
    _fill_suspended_daily_raw,
    _fit_normalization,
    _parse_years,
    _prepare_daily_frame,
    _prepare_daily_with_state_memory_bounded,
    _resolve_deferred_exit_days,
    _write_sample_index_streaming,
    assert_qdp_source_fresh,
    compute_long_suspension_masks,
    derive_mainboard_limit_panels,
    path_summary_columns,
    path_value_column,
    simulate_a_share_round_trip,
)
from daily_research.path_policy.qdp_v2_sequence_path_training import (
    DateGroupedBatchSampler,
    GlobalTailBatchSampler,
    SequencePathModel,
    SequencePathPackDataset,
    ShuffledBatchSampler,
    _compute_loss,
)
from daily_research.path_policy.qdp_v2_sequence_path_training import (
    INPUT_CHANNEL_PROFILE_DAILY_ONLY,
    INPUT_CHANNEL_PROFILE_NO_INTRADAY_SUMMARY,
    INPUT_CHANNEL_PROFILE_NO_LIMIT_STRUCTURE,
    PATH_LOSS_PROFILE_OHLCVA_EQUAL,
    PATH_VALUE_GRADIENT_PROFILE_HARD_ST,
    PATH_VALUE_GRADIENT_PROFILE_SMOOTH,
    RANK_TRAINING_PROFILE_GLOBAL_TAIL_512,
    SUMMARY_LOSS_PROFILE_MULTI_HORIZON_OHLC_NO60,
    SUMMARY_LOSS_PROFILE_MULTI_HORIZON_OHLC,
    _derive_path_summary_numpy,
    _derive_path_summary_torch,
    _derived_path_rank_score_and_target,
    _multi_horizon_ohlc_summary_loss_loop,
    _multi_horizon_ohlc_summary_loss_vectorized,
    _global_tail_rank_loss_by_date,
    _realize_path_value_v2_plan_numpy,
    _realize_predicted_plan_numpy,
    _summary_loss_windows,
    derived_path_summary_columns,
    path_value_v2_column,
    unified_path_value_column,
)


def test_sequence_pack_default_output_root_is_daily_research_store() -> None:
    assert DEFAULT_OUTPUT_ROOT.as_posix() == "daily_research/data/research_store/sequence_pack"


def test_explicit_empty_year_set_does_not_restore_the_default_test_year() -> None:
    assert _parse_years("", default=(2025,)) == ()
    assert _parse_years(None, default=(2025,)) == (2025,)


def _daily_limit_fixture(
    raw_close: np.ndarray,
    factor: np.ndarray,
    raw_high: np.ndarray | None = None,
    raw_low: np.ndarray | None = None,
) -> np.ndarray:
    close = np.asarray(raw_close, dtype=np.float64) * np.asarray(factor, dtype=np.float64)
    high = (
        np.asarray(raw_high, dtype=np.float64)
        if raw_high is not None
        else np.asarray(raw_close, dtype=np.float64) * 1.02
    ) * np.asarray(factor, dtype=np.float64)
    low = (
        np.asarray(raw_low, dtype=np.float64)
        if raw_low is not None
        else np.asarray(raw_close, dtype=np.float64) * 0.98
    ) * np.asarray(factor, dtype=np.float64)
    panel = np.full((*close.shape, len(DAILY_RAW_FEATURES)), np.nan, dtype=np.float32)
    panel[..., DAILY_RAW_FEATURES.index("close")] = close
    panel[..., DAILY_RAW_FEATURES.index("high")] = high
    panel[..., DAILY_RAW_FEATURES.index("low")] = low
    return panel


def test_mainboard_limit_reconstruction_handles_ex_right_reference_and_st_rate() -> None:
    dates = ["2024-01-02", "2024-01-03", "2024-01-04"]
    raw_close = np.asarray([[10.0], [5.0], [5.0]], dtype=np.float32)
    factor = np.asarray([[2.0], [4.0], [4.0]], dtype=np.float32)
    daily_raw = _daily_limit_fixture(raw_close, factor)

    up, down, audit = derive_mainboard_limit_panels(
        daily_raw=daily_raw,
        raw_close=raw_close,
        has_bar=np.ones_like(raw_close, dtype=bool),
        status_valid=np.ones_like(raw_close, dtype=bool),
        is_st=np.asarray([[False], [False], [True]], dtype=bool),
        date_values=dates,
        symbol_values=["600000.SH"],
    )

    assert np.isnan(up[0, 0])
    assert up[1, 0] == pytest.approx(5.50)
    assert down[1, 0] == pytest.approx(4.50)
    assert up[2, 0] == pytest.approx(5.25)
    assert down[2, 0] == pytest.approx(4.75)
    assert audit["standard_limit_day_count"] == 2


def test_mainboard_limit_reconstruction_excludes_registration_ipo_window() -> None:
    dates = pd.date_range("2024-01-02", periods=7, freq="B").strftime("%Y-%m-%d").tolist()
    raw_close = np.full((7, 1), 10.0, dtype=np.float32)
    daily_raw = _daily_limit_fixture(raw_close, np.ones_like(raw_close))

    up, down, audit = derive_mainboard_limit_panels(
        daily_raw=daily_raw,
        raw_close=raw_close,
        has_bar=np.ones_like(raw_close, dtype=bool),
        status_valid=np.ones_like(raw_close, dtype=bool),
        is_st=np.zeros_like(raw_close, dtype=bool),
        date_values=dates,
        symbol_values=["001999.SZ"],
        listing_dates={"001999.SZ": dates[0]},
    )

    assert np.isnan(up[:5, 0]).all()
    assert up[5, 0] == pytest.approx(11.0)
    assert down[5, 0] == pytest.approx(9.0)
    assert audit["ipo_no_limit_day_count"] == 5


def test_mainboard_limit_reconstruction_marks_observed_special_session_no_limit() -> None:
    dates = ["2024-01-02", "2024-01-03"]
    raw_close = np.asarray([[10.0], [11.5]], dtype=np.float32)
    daily_raw = _daily_limit_fixture(
        raw_close,
        np.ones_like(raw_close),
        raw_high=np.asarray([[10.2], [12.0]], dtype=np.float32),
        raw_low=np.asarray([[9.8], [10.0]], dtype=np.float32),
    )

    up, down, audit = derive_mainboard_limit_panels(
        daily_raw=daily_raw,
        raw_close=raw_close,
        has_bar=np.ones_like(raw_close, dtype=bool),
        status_valid=np.ones_like(raw_close, dtype=bool),
        is_st=np.zeros_like(raw_close, dtype=bool),
        date_values=dates,
        symbol_values=["600000.SH"],
    )

    assert np.isnan(up[1, 0])
    assert np.isnan(down[1, 0])
    assert audit["inferred_special_no_limit_day_count"] == 1


def test_memory_bounded_daily_preparation_matches_legacy_composition() -> None:
    dates = pd.date_range("2024-01-02", periods=30, freq="B").strftime("%Y-%m-%d")
    rows: list[dict[str, object]] = []
    for symbol_idx, symbol in enumerate(("000001.SZ", "600000.SH")):
        for day_idx, trade_date in enumerate(dates):
            close = 10.0 + symbol_idx + day_idx * 0.05
            rows.append(
                {
                    "symbol": symbol,
                    "trade_date": trade_date,
                    "open": close - 0.03,
                    "high": close + 0.08,
                    "low": close - 0.09,
                    "close": close,
                    "volume": 1000.0 + day_idx,
                    "amount": 10000.0 + day_idx * 20.0,
                    "raw_open": close - 0.03,
                    "raw_close": close,
                }
            )
    daily = pd.DataFrame(rows).sample(frac=1.0, random_state=7).reset_index(drop=True)
    legacy = _add_raw_daily_signals(_prepare_daily_frame(daily.copy()))
    bounded = _prepare_daily_with_state_memory_bounded(daily.copy())
    columns = [*DAILY_RAW_FEATURES, *RAW_SIGNAL_COLUMNS, "raw_open", "raw_close"]
    pd.testing.assert_frame_equal(
        bounded[["symbol", "trade_date", *columns]].reset_index(drop=True),
        legacy[["symbol", "trade_date", *columns]].reset_index(drop=True),
        check_dtype=False,
        check_exact=False,
        rtol=1.0e-12,
        atol=1.0e-12,
    )


def test_date_sharded_store_keeps_only_one_mapping_open_and_can_revisit(tmp_path) -> None:
    store = DateShardedFloatStore(
        directory=tmp_path / "labels",
        name="future",
        shape=(4, 2, 2, 1),
        shard_size=2,
    )
    first = np.full((2, 2, 1), 1.0, dtype=np.float32)
    third = np.full((2, 2, 1), 3.0, dtype=np.float32)
    second = np.full((2, 2, 1), 2.0, dtype=np.float32)
    store.set_date(0, first)
    store.set_date(2, third)
    assert len(store._shards) == 1
    store.set_date(1, second)
    assert len(store._shards) == 1
    store.flush()
    assert len(store._shards) == 0
    assert len(store.shard_metas) == 2
    reopened = np.memmap(
        tmp_path / "labels" / "future.000000_000001.float32.dat",
        dtype="float32",
        mode="r",
        shape=(2, 2, 2, 1),
    )
    np.testing.assert_allclose(reopened[0], first)
    np.testing.assert_allclose(reopened[1], second)


def test_memory_guard_stops_safely_and_records_progress(tmp_path, monkeypatch) -> None:
    assert sequence_pack.DEFAULT_MINIMUM_FREE_MEMORY_GB == 1.0
    progress = tmp_path / "progress.json"
    monkeypatch.setattr(sequence_pack, "_available_physical_memory_gb", lambda: 1.25)
    with pytest.raises(MemoryError, match="memory guard blocked"):
        sequence_pack._enforce_memory_guard(
            stage="large_stage",
            minimum_free_gb=3.0,
            progress_path=progress,
        )
    payload = json.loads(progress.read_text(encoding="utf-8"))
    assert payload["blocker"] == "minimum_free_memory_guard"
    assert payload["available_memory_gb"] == 1.25


def test_streaming_index_writer_matches_in_memory_candidate_index(tmp_path) -> None:
    dates = ["2024-01-02", "2024-01-03", "2025-01-02"]
    symbols = ["000001.SZ", "000002.SZ"]
    input_valid = np.ones((3, 2), dtype=bool)
    entry = np.asarray([[True, False], [True, True], [False, True]], dtype=bool)
    label = np.asarray([[True, False], [True, True], [False, True]], dtype=bool)
    signal = np.ones((3, 2), dtype=bool)
    expected = _build_sample_index(
        date_values=dates,
        symbol_values=symbols,
        start_date="2024-01-01",
        end_date="2025-12-31",
        train_years=(2024,),
        validation_years=(2025,),
        test_years=(2099,),
        input_valid=input_valid,
        entry_buyable=entry,
        label_valid=label,
        price_label_valid=label,
        va_aux_valid=label,
        signal_eligible=signal,
        require_entry_filled=False,
        require_label_valid=False,
        id_column="candidate_id",
    )
    path = tmp_path / "candidate_index.parquet"
    stats = _write_sample_index_streaming(
        output_path=path,
        date_values=dates,
        symbol_values=symbols,
        start_date="2024-01-01",
        end_date="2025-12-31",
        train_years=(2024,),
        validation_years=(2025,),
        test_years=(2099,),
        input_valid=input_valid,
        entry_buyable=entry,
        label_valid=label,
        price_label_valid=label,
        va_aux_valid=label,
        signal_eligible=signal,
        require_entry_filled=False,
        require_label_valid=False,
        id_column="candidate_id",
        target_row_group_size=2,
    )
    actual = pd.read_parquet(path)
    pd.testing.assert_frame_equal(actual, expected, check_dtype=False)
    assert stats["row_count"] == len(expected)
    assert stats["split_counts"] == {"train": 4, "validation": 2}


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


def test_future_path_can_anchor_on_signal_day_close() -> None:
    raw = _raw_panel(
        open_values=[8.0, 9.0, 10.0, 12.0, 13.0],
        high_values=[8.5, 9.5, 11.0, 14.0, 13.5],
        low_values=[7.5, 8.5, 9.5, 11.0, 12.5],
        close_values=[8.2, 9.2, 10.5, 13.0, 13.2],
    )
    up_limit = np.full((5, 1), np.nan, dtype=np.float32)

    future_path, future_ohlcva_path, _summary, input_valid, entry_buyable, label_valid = _compute_future_path_and_masks(
        raw_panel=raw,
        up_limit_panel=up_limit,
        lookback_days=2,
        forward_days=2,
        price_anchor="today_close",
    )

    assert input_valid[1, 0]
    assert entry_buyable[1, 0]
    assert label_valid[1, 0]
    # Signal date index 1 close = 9.2, so the next open gap is preserved.
    assert np.isclose(future_path[1, 0, 0, 0], 10.0 / 9.2 - 1.0)
    assert np.isclose(future_path[1, 0, 0, 3], 10.5 / 9.2 - 1.0)
    assert np.isclose(future_path[1, 0, 1, 3], 13.0 / 9.2 - 1.0)
    assert np.isclose(future_ohlcva_path[1, 0, 0, 0], future_path[1, 0, 0, 0])


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


def test_strict_tick_entry_rule_allows_one_tick_below_limit() -> None:
    raw = _raw_panel(
        open_values=[8.0, 9.0, 9.99, 10.0, 10.0],
        high_values=[8.5, 9.5, 10.0, 10.1, 10.1],
        low_values=[7.5, 8.5, 9.8, 9.9, 9.9],
        close_values=[8.2, 9.2, 9.99, 10.0, 10.0],
    )
    up_limit = np.full((5, 1), np.nan, dtype=np.float32)
    up_limit[2, 0] = 10.0
    up_limit[3, 0] = 10.0

    _path, _ohlcva, _summary, _input_valid, entry_filled, _label_valid = _compute_future_path_and_masks(
        raw_panel=raw,
        up_limit_panel=up_limit,
        lookback_days=1,
        forward_days=1,
        entry_rule=ENTRY_RULE_OPEN_BELOW_LIMIT,
    )

    assert entry_filled[1, 0]
    assert not entry_filled[2, 0]


def test_back_adjustment_changes_only_ohlc_and_preserves_raw_open() -> None:
    daily = pd.DataFrame(
        {
            "symbol": ["000001.SZ", "000001.SZ"],
            "trade_date": ["2024-01-02", "2024-01-03"],
            "open": [10.0, 9.0],
            "high": [10.5, 9.5],
            "low": [9.5, 8.5],
            "close": [10.0, 9.0],
            "volume": [100.0, 200.0],
            "amount": [1000.0, 2000.0],
        }
    )
    factor = pd.DataFrame(
        {
            "symbol": ["000001.SZ", "000001.SZ"],
            "trade_date": ["2024-01-02", "2024-01-03"],
            "adjust_factor": [1.0, 10.0 / 9.0],
        }
    )

    adjusted = _apply_back_adjustment(daily, factor)

    assert adjusted["raw_open"].tolist() == [10.0, 9.0]
    assert adjusted["raw_close"].tolist() == [10.0, 9.0]
    assert np.allclose(adjusted["close"].to_numpy(), [10.0, 10.0])
    assert adjusted["volume"].tolist() == [100.0, 200.0]
    assert adjusted["amount"].tolist() == [1000.0, 2000.0]


def test_price_label_remains_valid_when_va_aux_is_missing() -> None:
    raw = _raw_panel(
        open_values=[10.0, 10.0, 10.0, 10.0],
        high_values=[10.5, 10.5, 10.5, 10.5],
        low_values=[9.5, 9.5, 9.5, 9.5],
        close_values=[10.0, 10.0, 10.0, 10.0],
    )
    raw[2, 0, DAILY_RAW_FEATURES.index("amount_log")] = np.nan
    price_valid = np.zeros((4, 1), dtype=bool)
    va_valid = np.zeros((4, 1), dtype=bool)

    future_path, future_ohlcva, _summary, _input, _entry, label_valid = _compute_future_path_and_masks(
        raw_panel=raw,
        up_limit_panel=np.full((4, 1), np.nan, dtype=np.float32),
        lookback_days=2,
        forward_days=1,
        separate_price_va_validity=True,
        price_label_valid_out=price_valid,
        va_aux_valid_out=va_valid,
    )

    assert label_valid[1, 0]
    assert price_valid[1, 0]
    assert not va_valid[1, 0]
    assert np.isfinite(future_path[1, 0]).all()
    assert np.isnan(future_ohlcva[1, 0, 0, 5])


def test_suspension_fill_carries_price_but_does_not_create_observed_bar() -> None:
    raw = _raw_panel(
        open_values=[10.0, np.nan, 11.0],
        high_values=[10.5, np.nan, 11.5],
        low_values=[9.5, np.nan, 10.5],
        close_values=[10.0, np.nan, 11.0],
        volume_values=[100.0, np.nan, 200.0],
        amount_values=[1000.0, np.nan, 2000.0],
    )
    suspended = np.array([[False], [True], [False]])
    has_bar = np.array([[True], [False], [True]])

    _fill_suspended_daily_raw(raw, suspended_panel=suspended, has_bar_panel=has_bar)

    for field in ("open", "high", "low", "close"):
        assert raw[1, 0, DAILY_RAW_FEATURES.index(field)] == 10.0
    assert raw[1, 0, DAILY_RAW_FEATURES.index("volume")] == 0.0
    assert raw[1, 0, DAILY_RAW_FEATURES.index("amount")] == 0.0
    assert not has_bar[1, 0]


def test_long_suspension_break_triggers_on_twentieth_open_day_only() -> None:
    nineteen = np.zeros((30, 1), dtype=bool)
    nineteen[5:24, 0] = True
    long_19, breaks_19 = compute_long_suspension_masks(nineteen)
    assert not long_19.any()
    assert not breaks_19.any()

    twenty = np.zeros((30, 1), dtype=bool)
    twenty[5:25, 0] = True
    long_20, breaks_20 = compute_long_suspension_masks(twenty)
    assert long_20[5:25, 0].all()
    assert int(long_20.sum()) == 20
    assert breaks_20[25, 0]
    assert int(breaks_20.sum()) == 1


def test_continuity_break_invalidates_crossing_input_and_dependency_tail() -> None:
    n_dates = 70
    raw = _raw_panel(
        open_values=[10.0] * n_dates,
        high_values=[10.5] * n_dates,
        low_values=[9.5] * n_dates,
        close_values=[10.0] * n_dates,
    )
    suspended = np.zeros((n_dates, 1), dtype=bool)
    suspended[30:50, 0] = True
    long_suspension, continuity_break = compute_long_suspension_masks(suspended)

    _, _, _, input_valid, _, label_valid = _compute_future_path_and_masks(
        raw_panel=raw,
        up_limit_panel=np.full((n_dates, 1), np.nan, dtype=np.float32),
        lookback_days=5,
        forward_days=2,
        suspended_panel=suspended,
        long_suspension_panel=long_suspension,
        continuity_break_panel=continuity_break,
        execution_tail_days=3,
    )

    assert not input_valid[50, 0]  # lookback crosses the reopen break
    assert input_valid[54, 0]  # five observations beginning on the reopen day
    assert not label_valid[27, 0]  # forward path is clear, but the tail reaches the long halt
    assert not label_valid[48, 0]  # forward path crosses the break


def test_qdp_source_hash_mismatch_is_reported_as_stale(tmp_path) -> None:
    root = tmp_path / "qdp_v2"
    dataset_json = root / "datasets" / "market_daily_raw" / "daily__unit" / "dataset.json"
    dataset_json.parent.mkdir(parents=True)
    dataset_json.write_text('{"row_count": 1}\n', encoding="utf-8")
    digest = sequence_pack._file_sha256(dataset_json)
    manifest = {
        "artifact_type": "qdp_v2_sequence_path_pack",
        "qdp_root": str(root),
        "qdp_source_manifests": {
            "market_daily_raw": {
                "dataset_id": "daily__unit",
                "manifest_path": str(dataset_json),
                "dataset_json_sha256": digest,
            }
        },
    }
    assert_qdp_source_fresh(manifest)
    dataset_json.write_text('{"row_count": 2}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="stale_qdp_source"):
        assert_qdp_source_fresh(manifest)


def test_immutable_research_pack_survives_active_qdp_manifest_advance(tmp_path) -> None:
    source = tmp_path / "pack" / "manifest.json"
    source.parent.mkdir(parents=True)
    sample = source.parent / "sample_index.parquet"
    candidate = source.parent / "candidate_index.parquet"
    panel = source.parent / "daily.float32.dat"
    pd.DataFrame({"split": ["train"]}).to_parquet(sample, index=False)
    pd.DataFrame({"split": ["development"]}).to_parquet(candidate, index=False)
    panel.write_bytes(np.asarray([1.0], dtype=np.float32).tobytes())
    source_payload = {
        "artifact_type": "qdp_v2_sequence_path_pack",
        "sample_index_path": str(sample),
        "candidate_index_path": str(candidate),
        "feature_channels": {
            "daily_raw": {"path": str(panel), "shape": [1, 1, 1]}
        },
        "label_arrays": {},
        "execution_arrays": {},
        "masks": {},
    }
    source.write_text(json.dumps(source_payload), encoding="utf-8")
    stat = panel.stat()
    inventory = [{"path": str(panel.resolve()), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}]
    attachment = tmp_path / "overlay.json"
    attachment.write_text('{"version": 1}\n', encoding="utf-8")
    manifest = {
        **source_payload,
        "source_view_provenance": {
            "schema_version": 1,
            "manifest_path": str(source),
            "manifest_sha256": sequence_pack._file_sha256(source),
            "sample_index_path": str(sample),
            "sample_index_sha256": sequence_pack._file_sha256(sample),
            "candidate_index_path": str(candidate),
            "candidate_index_sha256": sequence_pack._file_sha256(candidate),
            "artifact_type": "qdp_v2_sequence_path_pack",
            "backing_files": inventory,
            "backing_files_sha256": hashlib.sha256(
                json.dumps(
                    inventory,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                ).encode("utf-8")
            ).hexdigest(),
        },
        "qdp_source_freshness_policy": {
            "mode": "immutable_research_pack_v1",
            "attachments": [
                {"path": str(attachment), "sha256": sequence_pack._file_sha256(attachment)}
            ],
        },
    }
    assert_qdp_source_fresh(manifest)
    attachment.write_text('{"version": 2}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="immutable_pack_attachment_hash_mismatch"):
        assert_qdp_source_fresh(manifest)


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


def test_pit_sample_index_retains_unfilled_rows_after_signal_day_filter() -> None:
    dates = ["2024-01-02", "2024-01-03", "2024-01-04"]
    symbols = ["000001.SZ", "000002.SZ"]
    input_valid = np.array([[True, True], [True, True], [True, True]])
    entry_filled = np.array([[False, True], [True, True], [True, True]])
    price_valid = np.array([[True, True], [True, True], [True, True]])
    signal_eligible = np.array([[True, False], [True, True], [True, True]])

    sample_index = _build_sample_index(
        date_values=dates,
        symbol_values=symbols,
        start_date="2024-01-02",
        end_date="2024-01-02",
        train_years=(),
        validation_years=(2024,),
        test_years=(),
        input_valid=input_valid,
        entry_buyable=entry_filled,
        label_valid=price_valid,
        signal_eligible=signal_eligible,
        require_entry_filled=False,
    )

    assert sample_index[["trade_date", "symbol", "entry_filled"]].to_dict("records") == [
        {"trade_date": "2024-01-02", "symbol": "000001.SZ", "entry_filled": False}
    ]


def test_candidate_index_is_not_gated_by_future_labels_or_entry_fill() -> None:
    dates = ["2024-01-02", "2024-01-03"]
    symbols = ["000001.SZ", "000002.SZ"]
    input_valid = np.ones((2, 2), dtype=bool)
    entry_filled = np.asarray([[False, True], [True, True]], dtype=bool)
    label_valid = np.asarray([[False, True], [True, True]], dtype=bool)
    price_valid = np.asarray([[False, True], [True, True]], dtype=bool)
    va_valid = np.asarray([[True, False], [True, True]], dtype=bool)
    signal_eligible = np.asarray([[True, True], [True, True]], dtype=bool)

    supervised = _build_sample_index(
        date_values=dates,
        symbol_values=symbols,
        start_date="2024-01-02",
        end_date="2024-01-02",
        train_years=(),
        validation_years=(2024,),
        test_years=(),
        input_valid=input_valid,
        entry_buyable=entry_filled,
        label_valid=label_valid,
        price_label_valid=price_valid,
        va_aux_valid=va_valid,
        signal_eligible=signal_eligible,
        require_entry_filled=False,
    )
    candidates = _build_sample_index(
        date_values=dates,
        symbol_values=symbols,
        start_date="2024-01-02",
        end_date="2024-01-02",
        train_years=(),
        validation_years=(2024,),
        test_years=(),
        input_valid=input_valid,
        entry_buyable=entry_filled,
        label_valid=label_valid,
        price_label_valid=price_valid,
        va_aux_valid=va_valid,
        signal_eligible=signal_eligible,
        require_entry_filled=False,
        require_label_valid=False,
        id_column="candidate_id",
    )

    assert supervised["symbol"].tolist() == ["000002.SZ"]
    assert candidates["symbol"].tolist() == ["000001.SZ", "000002.SZ"]
    invalid = candidates.loc[candidates["symbol"] == "000001.SZ"].iloc[0]
    assert not bool(invalid["entry_filled"])
    assert not bool(invalid["label_valid"])
    assert not bool(invalid["price_label_valid"])
    assert bool(invalid["va_aux_valid"])


def test_exit_sellable_blocks_down_limit_suspension_and_missing_bar() -> None:
    sellable = _exit_fill_mask(
        np.asarray([9.00, 9.01, np.nan, 10.00], dtype=np.float32),
        np.asarray([9.00, 9.00, 9.00, np.nan], dtype=np.float32),
        exit_observed=np.asarray([True, True, False, True]),
        exit_status_valid=np.asarray([True, True, True, True]),
        exit_suspended=np.asarray([False, False, False, True]),
        exit_delisted=np.asarray([False, False, False, False]),
    )

    assert sellable.tolist() == [False, True, False, False]


def test_deferred_exit_uses_first_sellable_day_inside_execution_tail() -> None:
    sellable = np.asarray(
        [
            [True, False, False, True, True],
            [True, False, False, False, False],
            [True, True, True, True, True],
        ],
        dtype=bool,
    )
    resolved = _resolve_deferred_exit_days(
        np.asarray([2.0, 2.0, 1.0]),
        sellable,
        forward_days=3,
        execution_tail_days=2,
    )

    assert resolved[0] == 4.0
    assert np.isnan(resolved[1])
    assert resolved[2] == 2.0


def test_round_trip_costs_apply_board_lot_minimum_commission_and_sell_tax() -> None:
    cost = AShareExecutionCostConfig(slippage_bps=0.0)
    result = simulate_a_share_round_trip(
        allocated_cash=10_000.0,
        entry_price=10.0,
        exit_price=11.0,
        cost=cost,
    )
    too_small = simulate_a_share_round_trip(
        allocated_cash=900.0,
        entry_price=10.0,
        exit_price=11.0,
        cost=cost,
    )
    historical_tax = simulate_a_share_round_trip(
        allocated_cash=10_000.0,
        entry_price=10.0,
        exit_price=11.0,
        cost=cost,
        exit_trade_date="2022-01-04",
    )

    assert result["filled"] is True
    assert result["shares"] == 900
    assert np.isclose(result["ending_cash"], 10_884.861)
    assert np.isclose(result["net_return"], 0.0884861)
    assert np.isclose(historical_tax["ending_cash"], 10_879.911)
    assert too_small == {
        "filled": False,
        "shares": 0,
        "ending_cash": 900.0,
        "net_return": 0.0,
        "total_cost": 0.0,
    }


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
    assert set(parts) == {
        "loss",
        "path_loss",
        "summary_loss",
        "richer_loss",
        "price_delta_loss",
        "va_level_loss",
        "va_delta_loss",
        "geometry_loss",
        "utility_curve_loss",
        "turnover_level_loss",
        "turnover_delta_loss",
        "value_loss",
        "rank_loss",
        "residual_penalty",
    }


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


def test_price_delta_loss_penalizes_close_path_rhythm() -> None:
    true_path = torch.zeros(4, 5, 4)
    pred_path = true_path.clone()
    true_path[:, :, 3] = torch.tensor([0.00, 0.01, 0.02, 0.03, 0.04])
    pred_path[:, :, 3] = torch.tensor([0.00, 0.02, 0.02, 0.04, 0.04])
    y_summary = torch.zeros(4, 12)
    date_idx = torch.tensor([1, 1, 1, 1])

    loss, parts = _compute_loss(
        {"future_path": pred_path},
        true_path,
        y_summary,
        date_idx,
        value_index=11,
        path_weight=0.0,
        summary_weight=0.0,
        value_weight=0.0,
        rank_weight=0.0,
        price_delta_weight=1.0,
    )

    assert torch.isfinite(loss)
    assert parts["price_delta_loss"] > 0.0
    assert parts["path_loss"] > 0.0


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


def test_path_value_v2_hard_st_matches_hard_inference_and_keeps_smooth_gradient() -> None:
    smooth_path = torch.zeros((3, 60, 4), dtype=torch.float32, requires_grad=True)
    smooth_value = _derive_path_summary_torch(
        smooth_path,
        smooth_value=True,
        path_value_gradient_profile=PATH_VALUE_GRADIENT_PROFILE_SMOOTH,
    )[:, -1]
    smooth_value.sum().backward()
    smooth_gradient = smooth_path.grad.detach().clone()

    hard_st_path = torch.zeros((3, 60, 4), dtype=torch.float32, requires_grad=True)
    hard_st_value = _derive_path_summary_torch(
        hard_st_path,
        smooth_value=True,
        path_value_gradient_profile=PATH_VALUE_GRADIENT_PROFILE_HARD_ST,
    )[:, -1]
    hard_st_value.sum().backward()
    hard_st_gradient = hard_st_path.grad.detach().clone()

    hard_inference = _derive_path_summary_numpy(np.zeros((3, 60, 4), dtype=np.float32))[:, -1]
    assert np.allclose(hard_st_value.detach().numpy(), hard_inference, atol=1.0e-7)
    assert not np.allclose(smooth_value.detach().numpy(), hard_inference, atol=1.0e-3)
    assert torch.allclose(hard_st_gradient, smooth_gradient, atol=1.0e-7, rtol=1.0e-6)


def test_path_value_v2_target_excludes_nontradable_exit_days() -> None:
    path = np.zeros((1, 4, 4), dtype=np.float32)
    path[0, :, 3] = [0.01, 0.20, 0.04, 0.03]
    path[0, :, 1] = path[0, :, 3]
    tradable = np.asarray([[True, False, True, True]], dtype=bool)

    numpy_summary = _derive_path_summary_numpy(path, tradable_path=tradable)
    torch_summary = _derive_path_summary_torch(
        torch.from_numpy(path),
        smooth_value=False,
        tradable_path=torch.from_numpy(tradable),
    ).numpy()

    assert numpy_summary[0, 8] != 2.0
    assert np.allclose(numpy_summary, torch_summary, atol=1.0e-6)

    no_exit = np.zeros((1, 4), dtype=bool)
    numpy_no_exit = _derive_path_summary_numpy(path, tradable_path=no_exit)
    torch_no_exit = _derive_path_summary_torch(
        torch.from_numpy(path),
        smooth_value=False,
        tradable_path=torch.from_numpy(no_exit),
    ).numpy()
    assert np.isnan(numpy_no_exit[0, 8:]).all()
    assert np.isnan(torch_no_exit[0, 8:]).all()


def test_path_value_v2_searches_the_legal_domain_instead_of_clamping_day_one() -> None:
    path = np.zeros((1, 4, 4), dtype=np.float32)
    path[0, :, 1] = [0.50, 0.02, 0.20, 0.03]
    path[0, :, 2] = 0.0
    path[0, :, 3] = [0.50, 0.02, 0.20, 0.03]

    legal_numpy = _derive_path_summary_numpy(path)
    legal_torch = _derive_path_summary_torch(
        torch.from_numpy(path), smooth_value=False
    ).detach().numpy()
    legacy = _derive_path_summary_numpy(path, earliest_exit_day=1)

    assert legacy[0, 8] == 1.0
    assert legal_numpy[0, 8] == 3.0
    assert legal_torch[0, 8] == 3.0


def test_path_value_v2_legal_ties_choose_the_earliest_legal_day(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sequence_training, "PATH_VALUE_V2_WAITING_PENALTY", 0.0)
    path = np.zeros((1, 4, 4), dtype=np.float32)
    path[0, 1:3, 3] = 0.05
    path[0, 1:3, 1] = 0.05

    summary = _derive_path_summary_numpy(path)

    assert summary[0, 8] == 2.0


def test_path_value_v2_day_one_tradability_does_not_create_a_legal_exit() -> None:
    path = np.zeros((1, 4, 4), dtype=np.float32)
    tradable = np.asarray([[True, False, False, False]], dtype=bool)

    numpy_summary = _derive_path_summary_numpy(path, tradable_path=tradable)
    torch_summary = _derive_path_summary_torch(
        torch.from_numpy(path),
        smooth_value=False,
        tradable_path=torch.from_numpy(tradable),
    ).detach().numpy()

    assert np.isnan(numpy_summary[0, 8:]).all()
    assert np.isnan(torch_summary[0, 8:]).all()


def test_structured_geometry_round_trip_and_ohlc_legality() -> None:
    raw = torch.randn(5, 60, 4)
    geometry, path = sequence_training._structured_geometry_to_ohlc_torch(raw)
    recovered = sequence_training._ohlc_path_to_geometry_torch(path)

    assert geometry.dtype == torch.float32
    assert path.dtype == torch.float32
    assert torch.allclose(geometry, recovered, atol=2.0e-5, rtol=2.0e-5)
    assert bool(torch.all(path[:, :, 1] >= torch.maximum(path[:, :, 0], path[:, :, 3])))
    assert bool(torch.all(path[:, :, 2] <= torch.minimum(path[:, :, 0], path[:, :, 3])))


def test_structured_turnover_model_has_shared_future_states_and_price_only_score() -> None:
    model = SequencePathModel(
        input_dim=32,
        hidden_dim=16,
        layers=2,
        forward_days=60,
        summary_dim=12,
        dropout=0.0,
        model_type="gru_structured_joint_turnover",
    )
    out = model(torch.randn(4, 100, 32))
    assert set(out) == {"future_path", "future_price_geometry", "future_activity_path"}
    assert out["future_path"].shape == (4, 60, 4)
    assert out["future_price_geometry"].shape == (4, 60, 4)
    assert out["future_activity_path"].shape == (4, 60)

    score_before = _derive_path_summary_torch(
        out["future_path"], smooth_value=False, price_anchor="today_close"
    )[:, -1]
    changed = dict(out)
    changed["future_activity_path"] = out["future_activity_path"] + 10_000.0
    score_after = _derive_path_summary_torch(
        changed["future_path"], smooth_value=False, price_anchor="today_close"
    )[:, -1]
    assert torch.equal(score_before, score_after)


def test_structured_joint_loss_is_finite_and_backpropagates_through_decoder() -> None:
    model = SequencePathModel(
        input_dim=32,
        hidden_dim=16,
        layers=2,
        forward_days=10,
        summary_dim=12,
        dropout=0.0,
        model_type="gru_structured_joint_turnover",
    )
    x = torch.randn(8, 100, 32)
    target_raw = torch.randn(8, 10, 4)
    _target_geometry, y_path = sequence_training._structured_geometry_to_ohlc_torch(target_raw)
    y_activity = torch.randn(8, 10)
    y_activity[0, :] = float("nan")
    tradable = torch.ones(8, 10, dtype=torch.bool)
    tradable[:, 4] = False
    out = model(x)
    loss, parts = _compute_loss(
        out,
        y_path.detach(),
        torch.empty(8, 0),
        torch.zeros(8, dtype=torch.long),
        y_activity_path=y_activity,
        value_index=0,
        path_weight=0.35,
        summary_weight=0.20,
        value_weight=0.15,
        rank_weight=0.15,
        geometry_weight=0.10,
        utility_curve_weight=0.05,
        turnover_level_weight=0.02,
        turnover_delta_weight=0.01,
        price_anchor="today_close",
        summary_loss_profile=SUMMARY_LOSS_PROFILE_MULTI_HORIZON_OHLC,
        path_value_gradient_profile=PATH_VALUE_GRADIENT_PROFILE_HARD_ST,
        y_tradable_path=tradable,
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert all(np.isfinite(float(parts[key])) for key in parts)
    assert model.future_decoder.weight_hh_l0.grad is not None
    assert bool(torch.isfinite(model.future_decoder.weight_hh_l0.grad).all())


def test_path_value_v2_realized_plan_reports_exit_timing_regret_and_defers_suspended_exit() -> None:
    true_path = np.zeros((1, 4, 4), dtype=np.float32)
    true_path[0, :, 3] = [0.01, 0.03, 0.10, 0.04]
    true_path[0, :, 1] = true_path[0, :, 3]
    pred_summary = np.zeros((1, len(derived_path_summary_columns(4))), dtype=np.float32)
    exit_idx = derived_path_summary_columns(4).index("best_exit_day_4d")
    pred_summary[0, exit_idx] = 2.0
    tradable = np.asarray([[True, False, True, True]], dtype=bool)

    realized = _realize_path_value_v2_plan_numpy(
        pred_summary,
        true_path,
        forward_days=4,
        tradable_path=tradable,
    )

    assert realized["predicted_exit_day"][0] == 2.0
    assert realized["realized_plan_exit_day"][0] == 3.0
    assert realized["realized_plan_exit_tradable"][0] == 1.0
    assert realized["realized_plan_return"][0] == np.float32(0.10)
    assert realized["opportunity_value"][0] >= realized["realized_plan_value"][0]
    assert np.isclose(
        realized["oracle_regret"][0],
        realized["opportunity_value"][0] - realized["realized_plan_value"][0],
    )

    unfilled = _realize_path_value_v2_plan_numpy(
        pred_summary,
        true_path,
        forward_days=4,
        tradable_path=tradable,
        entry_filled=np.asarray([False]),
    )
    assert unfilled["realized_plan_entry_filled"][0] == 0.0
    assert unfilled["realized_plan_covered"][0] == 1.0
    assert unfilled["realized_plan_value"][0] == 0.0
    assert unfilled["oracle_regret"][0] == unfilled["opportunity_value"][0]


def test_multi_horizon_ohlc_summary_loss_uses_available_windows() -> None:
    assert _summary_loss_windows(20) == (5, 10, 20)
    assert _summary_loss_windows(60) == (5, 10, 20, 40, 60)
    assert _summary_loss_windows(60, include_full_horizon=False) == (5, 10, 20, 40)

    true_path = torch.zeros(4, 60, 4)
    pred_path = true_path.clone()
    perturbed_path = true_path.clone()
    perturbed_path[:, :10, 3] = 0.05
    y_summary = torch.zeros(4, 12)
    date_idx = torch.tensor([1, 1, 1, 1])

    exact_loss, exact_parts = _compute_loss(
        {"future_path": pred_path},
        true_path,
        y_summary,
        date_idx,
        value_index=11,
        path_weight=0.0,
        summary_weight=1.0,
        value_weight=0.0,
        rank_weight=0.0,
        summary_loss_profile=SUMMARY_LOSS_PROFILE_MULTI_HORIZON_OHLC,
    )
    perturbed_loss, perturbed_parts = _compute_loss(
        {"future_path": perturbed_path},
        true_path,
        y_summary,
        date_idx,
        value_index=11,
        path_weight=0.0,
        summary_weight=1.0,
        value_weight=0.0,
        rank_weight=0.0,
        summary_loss_profile=SUMMARY_LOSS_PROFILE_MULTI_HORIZON_OHLC,
    )

    assert torch.isfinite(exact_loss)
    assert torch.isfinite(perturbed_loss)
    assert exact_parts["summary_loss"] == 0.0
    assert perturbed_parts["summary_loss"] > exact_parts["summary_loss"]


def test_multi_horizon_summary_loss_no60_excludes_full_horizon_window() -> None:
    true_path = torch.zeros(4, 60, 4)
    exact_path = true_path.clone()
    early_perturbed_path = true_path.clone()
    late_perturbed_path = true_path.clone()
    early_perturbed_path[:, :10, 3] = 0.05
    late_perturbed_path[:, 59, 3] = 0.05
    y_summary = torch.zeros(4, 12)
    date_idx = torch.tensor([1, 1, 1, 1])

    exact_loss, exact_parts = _compute_loss(
        {"future_path": exact_path},
        true_path,
        y_summary,
        date_idx,
        value_index=11,
        path_weight=0.0,
        summary_weight=1.0,
        value_weight=0.0,
        rank_weight=0.0,
        summary_loss_profile=SUMMARY_LOSS_PROFILE_MULTI_HORIZON_OHLC_NO60,
    )
    early_loss, early_parts = _compute_loss(
        {"future_path": early_perturbed_path},
        true_path,
        y_summary,
        date_idx,
        value_index=11,
        path_weight=0.0,
        summary_weight=1.0,
        value_weight=0.0,
        rank_weight=0.0,
        summary_loss_profile=SUMMARY_LOSS_PROFILE_MULTI_HORIZON_OHLC_NO60,
    )
    late_loss, late_parts = _compute_loss(
        {"future_path": late_perturbed_path},
        true_path,
        y_summary,
        date_idx,
        value_index=11,
        path_weight=0.0,
        summary_weight=1.0,
        value_weight=0.0,
        rank_weight=0.0,
        summary_loss_profile=SUMMARY_LOSS_PROFILE_MULTI_HORIZON_OHLC_NO60,
    )

    assert torch.isfinite(exact_loss)
    assert torch.isfinite(early_loss)
    assert torch.isfinite(late_loss)
    assert exact_parts["summary_loss"] == 0.0
    assert early_parts["summary_loss"] > exact_parts["summary_loss"]
    assert late_parts["summary_loss"] == exact_parts["summary_loss"]


def test_multi_horizon_ohlc_summary_loss_vectorized_matches_loop() -> None:
    generator = torch.Generator().manual_seed(20260707)
    target_path = torch.randn((6, 60, 4), generator=generator) * 0.03
    pred_path = target_path + torch.randn((6, 60, 4), generator=generator) * 0.01

    for price_anchor in ("next_open", "today_close"):
        loop_loss = _multi_horizon_ohlc_summary_loss_loop(pred_path, target_path, price_anchor=price_anchor)
        vectorized_loss = _multi_horizon_ohlc_summary_loss_vectorized(pred_path, target_path, price_anchor=price_anchor)

        assert torch.allclose(vectorized_loss, loop_loss, atol=1.0e-7)


def test_multi_horizon_target_exit_fields_are_ignored_without_tradable_days() -> None:
    target_path = torch.zeros((2, 60, 4))
    pred_path = target_path.clone().requires_grad_(True)
    tradable = torch.zeros((2, 60), dtype=torch.bool)

    loss = _multi_horizon_ohlc_summary_loss_vectorized(
        pred_path,
        target_path,
        price_anchor="next_open",
        target_tradable_path=tradable,
    )

    assert torch.isfinite(loss)
    loss.backward()
    assert pred_path.grad is not None
    assert torch.isfinite(pred_path.grad).all()


def test_today_close_anchor_path_value_matches_next_open_anchor() -> None:
    today_close = 100.0
    next_open = 110.0
    future_prices = np.asarray(
        [
            [110.0, 116.0, 108.0, 112.0],
            [113.0, 125.0, 111.0, 121.0],
            [120.0, 126.0, 118.0, 119.0],
            [118.0, 124.0, 115.0, 123.0],
        ],
        dtype=np.float32,
    )
    next_open_path = (future_prices / next_open - 1.0).reshape(1, 4, 4)
    today_close_path = (future_prices / today_close - 1.0).reshape(1, 4, 4)

    next_summary = _derive_path_summary_numpy(next_open_path)
    today_summary = _derive_path_summary_numpy(today_close_path, price_anchor="today_close")
    torch_summary = _derive_path_summary_torch(
        torch.from_numpy(today_close_path),
        smooth_value=False,
        price_anchor="today_close",
    ).detach().numpy()

    assert np.allclose(today_summary, next_summary, atol=1.0e-6)
    assert np.allclose(torch_summary, next_summary, atol=1.0e-6)


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
    assert set(parts) == {
        "loss",
        "path_loss",
        "summary_loss",
        "richer_loss",
        "price_delta_loss",
        "va_level_loss",
        "va_delta_loss",
        "geometry_loss",
        "utility_curve_loss",
        "turnover_level_loss",
        "turnover_delta_loss",
        "value_loss",
        "rank_loss",
        "residual_penalty",
    }


def test_gru_direct_value_model_outputs_only_score_and_loss_uses_direct_horizon() -> None:
    model = SequencePathModel(
        input_dim=6,
        hidden_dim=8,
        layers=1,
        forward_days=60,
        summary_dim=9,
        dropout=0.0,
        model_type="gru_direct_value",
    )
    x = torch.randn(4, 100, 6)
    y_path = torch.zeros(4, 60, 4)
    y_path[:, :5, 3] = torch.tensor([0.01, 0.02, 0.03, 0.04, 0.05])
    y_path[:, 5:10, 3] = 0.20
    y_summary = torch.zeros(4, 9)
    date_idx = torch.tensor([1, 1, 1, 1])

    out = model(x)
    loss, parts = _compute_loss(
        out,
        y_path,
        y_summary,
        date_idx,
        value_index=8,
        path_weight=0.0,
        summary_weight=0.0,
        value_weight=0.50,
        rank_weight=0.50,
        direct_value_horizon=5,
    )

    assert set(out) == {"score"}
    assert out["score"].shape == (4,)
    assert torch.isfinite(loss)
    assert parts["path_loss"] == 0.0
    assert parts["summary_loss"] == 0.0
    assert parts["richer_loss"] == 0.0
    assert parts["value_loss"] >= 0.0
    assert parts["rank_loss"] >= 0.0


def test_direct_value_loss_targets_requested_horizon_not_full_path() -> None:
    y_path = torch.zeros(4, 60, 4)
    y_path[:, :5, 3] = 0.05
    y_path[:, 5:60, 3] = -0.20
    y_summary = torch.zeros(4, 9)
    date_idx = torch.tensor([1, 1, 1, 1])
    target_5d = _derive_path_summary_torch(y_path[:, :5, :4], smooth_value=False)[:, -1].detach()

    loss, parts = _compute_loss(
        {"score": target_5d},
        y_path,
        y_summary,
        date_idx,
        value_index=8,
        path_weight=0.0,
        summary_weight=0.0,
        value_weight=1.0,
        rank_weight=0.0,
        direct_value_horizon=5,
    )

    assert torch.isfinite(loss)
    assert loss.item() == 0.0
    assert parts["value_loss"] == 0.0


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


def test_gru_ohlcva_aux_path_value_keeps_price_value_path_and_adds_va_losses() -> None:
    model = SequencePathModel(
        input_dim=6,
        hidden_dim=8,
        layers=1,
        forward_days=20,
        summary_dim=9,
        dropout=0.0,
        model_type="gru_ohlcva_aux_path_value",
    )
    x = torch.randn(4, 100, 6)
    y_path = torch.randn(4, 20, 4) * 0.01
    y_ohlcva_path = torch.randn(4, 20, 6) * 0.01
    date_idx = torch.tensor([1, 1, 1, 1])

    out = model(x)
    loss, parts = _compute_loss(
        out,
        y_path,
        torch.empty(4, 0),
        date_idx,
        y_ohlcva_path=y_ohlcva_path,
        value_index=8,
        va_level_weight=0.05,
        va_delta_weight=0.02,
    )

    assert set(out) == {"future_path", "future_ohlcva_aux_path"}
    assert out["future_path"].shape == (4, 20, 4)
    assert out["future_ohlcva_aux_path"].shape == (4, 20, 6)
    assert model.uses_derived_path_value
    assert model.uses_ohlcva_aux_path
    assert not model.uses_ohlcva_path
    assert torch.isfinite(loss)
    assert parts["price_delta_loss"] >= 0.0
    assert parts["va_level_loss"] >= 0.0
    assert parts["va_delta_loss"] >= 0.0
    assert parts["value_loss"] >= 0.0


def test_ohlcva_equal_path_loss_uses_six_fields_but_price_only_value() -> None:
    pred_price = torch.zeros(4, 20, 4)
    pred_ohlcva = torch.zeros(4, 20, 6)
    true_price = torch.zeros(4, 20, 4)
    true_ohlcva = torch.zeros(4, 20, 6)
    true_ohlcva[:, :, 4] = 0.60
    true_ohlcva[:, :, 5] = -0.20
    date_idx = torch.tensor([1, 1, 1, 1])

    loss, parts = _compute_loss(
        {"future_path": pred_price, "future_ohlcva_aux_path": pred_ohlcva},
        true_price,
        torch.empty(4, 0),
        date_idx,
        y_ohlcva_path=true_ohlcva,
        value_index=8,
        path_weight=1.0,
        summary_weight=0.0,
        value_weight=0.0,
        rank_weight=0.0,
        va_level_weight=0.0,
        va_delta_weight=0.0,
        path_loss_profile=PATH_LOSS_PROFILE_OHLCVA_EQUAL,
    )
    _base_loss, base_parts = _compute_loss(
        {"future_path": pred_price, "future_ohlcva_aux_path": pred_ohlcva},
        true_price,
        torch.empty(4, 0),
        date_idx,
        y_ohlcva_path=torch.zeros_like(true_ohlcva),
        value_index=8,
        path_weight=1.0,
        summary_weight=0.0,
        value_weight=0.0,
        rank_weight=0.0,
        va_level_weight=0.0,
        va_delta_weight=0.0,
        path_loss_profile=PATH_LOSS_PROFILE_OHLCVA_EQUAL,
    )
    expected_fields = []
    for field_idx in range(6):
        expected_fields.append(torch.nn.functional.smooth_l1_loss(pred_ohlcva[:, :, field_idx], true_ohlcva[:, :, field_idx]))
    expected_path_loss = torch.stack(expected_fields).mean()

    assert torch.isclose(loss, expected_path_loss)
    assert np.isclose(parts["path_loss"], float(expected_path_loss.item()))
    assert np.isclose(parts["value_loss"], base_parts["value_loss"])
    assert np.isclose(parts["summary_loss"], base_parts["summary_loss"])
    assert parts["va_level_loss"] > 0.0


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
    lean_batch = dataset.get_batch(
        [0, 1],
        include_ohlcva_path=False,
        include_richer_path=False,
        include_summary=False,
    )
    assert lean_batch["y_path"].shape == (2, 20, 4)
    assert lean_batch["y_ohlcva_path"] is None
    assert lean_batch["y_richer_path"] is None
    assert lean_batch["y_summary"] is None
    # First channel, first feature: dates 0..2 for symbol 0 and 1.
    assert torch.equal(batch["x"][0, :, 0], torch.tensor([0.0, 10.0, 20.0]))
    assert torch.equal(batch["x"][1, :, 0], torch.tensor([1.0, 11.0, 21.0]))
    daily_only = SequencePathPackDataset(manifest, split="train", input_channel_profile=INPUT_CHANNEL_PROFILE_DAILY_ONLY)
    daily_only_batch = daily_only.get_batch([0, 1])

    assert daily_only.channel_order == ["daily_raw", "daily_state"]
    assert daily_only.input_dim == 3
    assert daily_only_batch["x"].shape == (2, 3, 3)
    no_intraday = SequencePathPackDataset(manifest, split="train", input_channel_profile=INPUT_CHANNEL_PROFILE_NO_INTRADAY_SUMMARY)
    assert no_intraday.channel_order == ["daily_raw", "daily_state", "limit_structure"]
    assert no_intraday.input_dim == 4

    no_limit = SequencePathPackDataset(manifest, split="train", input_channel_profile=INPUT_CHANNEL_PROFILE_NO_LIMIT_STRUCTURE)
    assert no_limit.channel_order == ["daily_raw", "daily_state", "intraday_summary"]
    assert no_limit.input_dim == 4

    mask_meta = {}
    for mask_name in ["has_bar", "is_suspended", "previous_close_valid", "zero_range", "corr_valid", "tradable"]:
        mask_path = tmp_path / f"{mask_name}.bool.dat"
        mask_values = np.ones((5, 2), dtype=bool)
        mask_memmap = np.memmap(mask_path, dtype="bool", mode="w+", shape=mask_values.shape)
        mask_memmap[:] = mask_values
        mask_memmap.flush()
        mask_meta[mask_name] = {"path": str(mask_path), "shape": list(mask_values.shape)}
    masked_manifest = {**manifest, "masks": mask_meta}
    masked_daily = SequencePathPackDataset(
        masked_manifest,
        split="train",
        input_channel_profile=INPUT_CHANNEL_PROFILE_DAILY_ONLY,
    )
    explicit_unmasked_daily = SequencePathPackDataset(
        {
            **masked_manifest,
            "data_semantics": {"model_input_mask_features": []},
        },
        split="train",
        input_channel_profile=INPUT_CHANNEL_PROFILE_DAILY_ONLY,
    )
    masked_all = SequencePathPackDataset(masked_manifest, split="train")

    assert masked_daily.input_mask_features == ["has_bar", "is_suspended", "previous_close_valid", "zero_range"]
    assert masked_daily.get_batch([0, 1])["x"].shape == (2, 3, 7)
    assert explicit_unmasked_daily.input_mask_features == []
    assert explicit_unmasked_daily.input_dim == 3
    assert explicit_unmasked_daily.get_batch([0, 1])["x"].shape == (2, 3, 3)
    assert masked_all.input_mask_features[-1] == "corr_valid"
    assert masked_all.get_batch([0, 1])["x"].shape == (2, 3, 10)
    assert masked_all.get_batch([0, 1])["y_tradable_path"].shape == (2, 20)
    assert masked_daily.path_value_targets().shape == (2,)


def test_date_grouped_batch_sampler_merges_small_dates_without_losing_date_groups() -> None:
    sample_index = pd.DataFrame(
        [
            {"date_idx": 1},
            {"date_idx": 1},
            {"date_idx": 2},
            {"date_idx": 2},
            {"date_idx": 3},
            {"date_idx": 3},
        ]
    )
    sampler = DateGroupedBatchSampler(sample_index, batch_size=4, shuffle=False, seed=7)

    batches = list(sampler)

    assert len(sampler) == 2
    assert batches == [[0, 1, 2, 3], [4, 5]]


def test_shuffled_batch_sampler_fills_path_batches_across_dates_without_loss() -> None:
    sampler = ShuffledBatchSampler(10, batch_size=4, shuffle=False, seed=7)

    assert list(sampler) == [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9]]
    assert len(sampler) == 3


def test_global_tail_512_sampler_uses_frozen_unique_daily_groups_and_prior_false_positives() -> None:
    sample_index = pd.DataFrame({"date_idx": np.zeros(700, dtype=np.int32)})
    targets = np.linspace(1.0, 0.0, num=700, dtype=np.float32)
    prior = np.full(700, -1.0, dtype=np.float32)
    prior[300:428] = np.linspace(2.0, 1.0, num=128, dtype=np.float32)
    sampler = GlobalTailBatchSampler(
        sample_index,
        targets,
        batch_size=512,
        shuffle=False,
        seed=7,
        prior_epoch_scores=prior,
    )

    slate, groups = sampler.build_slate(0, epoch=0)

    assert len(slate) == len(set(slate)) == 512
    assert len(groups["true_top"]) == 32
    assert len(groups["true_rank_33_256"]) == 128
    assert len(groups["middle"]) == 128
    assert len(groups["bottom"]) == 96
    assert len(groups["prior_epoch_false_positive"]) == 128
    assert set(groups["true_top"]) == set(range(32))
    assert set(groups["bottom"]) == set(range(604, 700))
    assert set(groups["prior_epoch_false_positive"]) == set(range(300, 428))


def test_global_tail_512_sampler_keeps_every_small_date_as_one_slate() -> None:
    sample_index = pd.DataFrame(
        {"date_idx": np.repeat(np.arange(3, dtype=np.int32), [4, 7, 11])}
    )
    targets = np.linspace(1.0, 0.0, num=len(sample_index), dtype=np.float32)
    sampler = GlobalTailBatchSampler(
        sample_index,
        targets,
        batch_size=512,
        shuffle=False,
        seed=7,
    )

    slates = list(sampler)

    assert len(slates) == 3
    assert [len(slate) for slate in slates] == [4, 7, 11]
    for date_idx, slate in enumerate(slates):
        assert set(sample_index.iloc[slate]["date_idx"].astype(int)) == {date_idx}


def test_global_tail_rank_loss_rewards_correct_target_order() -> None:
    target = torch.linspace(1.0, -1.0, steps=512)
    date_idx = torch.zeros(512, dtype=torch.long)

    correct = _global_tail_rank_loss_by_date(target.clone(), target, date_idx)
    reversed_loss = _global_tail_rank_loss_by_date(-target, target, date_idx)

    assert torch.isfinite(correct)
    assert correct < reversed_loss


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA AMP is unavailable")
def test_global_tail_rank_loss_prevents_fp16_pair_count_overflow() -> None:
    score = torch.linspace(1.0, -1.0, steps=512, device="cuda", dtype=torch.float16).requires_grad_(True)
    target = torch.linspace(1.0, -1.0, steps=512, device="cuda", dtype=torch.float32)
    date_idx = torch.zeros(512, device="cuda", dtype=torch.long)

    with torch.amp.autocast(device_type="cuda", enabled=True):
        actual = _global_tail_rank_loss_by_date(score, target, date_idx)
    expected = _global_tail_rank_loss_by_date(score.detach().float(), target, date_idx)

    assert actual.dtype == torch.float32
    assert float(actual) > 0.0
    assert torch.isinf(torch.tensor(512 * 511 // 2, device="cuda", dtype=torch.float16))
    torch.testing.assert_close(actual, expected, rtol=1.0e-6, atol=1.0e-8)
    actual.backward()
    assert score.grad is not None
    assert torch.isfinite(score.grad).all()
    assert float(score.grad.abs().sum()) > 0.0


def test_compute_loss_accepts_global_tail_profile_on_single_daily_slate() -> None:
    true_path = torch.zeros(8, 5, 4)
    true_path[:, :, 3] = torch.linspace(0.0, 0.14, steps=8).view(-1, 1)
    pred_path = true_path.clone().requires_grad_(True)
    date_idx = torch.ones(8, dtype=torch.long)

    loss, parts = _compute_loss(
        {"future_path": pred_path},
        true_path,
        torch.empty(8, 0),
        date_idx,
        value_index=0,
        path_weight=0.0,
        summary_weight=0.0,
        value_weight=0.0,
        rank_weight=1.0,
        rank_training_profile=RANK_TRAINING_PROFILE_GLOBAL_TAIL_512,
    )

    assert torch.isfinite(loss)
    assert parts["rank_loss"] >= 0.0


def test_separated_rank_score_rejects_unified_ohlcva_value_semantics() -> None:
    target_path = torch.zeros((4, 5, 4))
    tradable = torch.ones((4, 5), dtype=torch.bool)

    try:
        _derived_path_rank_score_and_target(
            {"future_path": torch.zeros((4, 5, 6))},
            target_path,
            price_anchor="next_open",
            path_value_gradient_profile=PATH_VALUE_GRADIENT_PROFILE_HARD_ST,
            target_tradable_path=tradable,
        )
    except ValueError as error:
        assert "unified OHLCVA" in str(error)
    else:
        raise AssertionError("global-tail rank score must reject six-field unified value semantics")


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
