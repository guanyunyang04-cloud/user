from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from daily_research.baseline.advanced_ml_runtime import HistoryWindow
from daily_research.continuous_policy.state_builder import (
    ALPHA_PRIOR_FRAME_NAMES,
    STATE_SEQUENCE_BASES,
    STATE_SEQUENCE_LAGS,
    PreparedPolicyInputs,
)
from daily_research.path_policy.forecast_dataset import ForecastMemmapDataset, ForecastSequenceDataset
from daily_research.path_policy.models import PATH20_DEFAULT_CUMULATIVE_HORIZONS, normalize_path20_cumulative_horizons


def make_prepared_policy_inputs(
    *,
    days: int = 34,
    stocks: tuple[str, ...] = ("AAA", "BBB", "CCC", "DDD"),
    start_date: str = "2024-01-02",
) -> PreparedPolicyInputs:
    dates = pd.bdate_range(str(start_date), periods=int(days))
    columns = list(stocks)
    base = np.arange(len(dates), dtype=float).reshape(-1, 1)
    multipliers = np.linspace(1.0, 1.6, len(columns)).reshape(1, -1)
    open_values = 10.0 + base * multipliers + np.arange(len(columns), dtype=float).reshape(1, -1)
    close_values = open_values * (1.0 + 0.001 * multipliers)
    open_ = pd.DataFrame(open_values, index=dates, columns=columns)
    close = pd.DataFrame(close_values, index=dates, columns=columns)
    high = close * 1.01
    low = open_ * 0.99
    volume_values = 100000.0 + base * 100.0 + np.arange(len(columns), dtype=float).reshape(1, -1) * 1000.0
    volume = pd.DataFrame(volume_values, index=dates, columns=columns)
    amount = volume * close
    benchmark_open = pd.Series(100.0 + np.arange(len(dates), dtype=float) * 0.75, index=dates)
    benchmark_close = benchmark_open * 1.001
    membership = pd.DataFrame(True, index=dates, columns=columns)
    score_none = pd.DataFrame(0.0, index=dates, columns=columns)
    score_v2 = pd.DataFrame(np.tile(np.linspace(0.1, 0.4, len(columns)), (len(dates), 1)), index=dates, columns=columns)
    score_blend = score_v2.copy()
    feature_frames = {
        name: pd.DataFrame(0.0, index=dates, columns=columns)
        for name in ("z_score_none", "z_score_v2", "adv20_rank", "price_rank", "ma20_gap", "ma60_gap", "volume_rank")
    }
    derived_names = [
        "ret_1d",
        "ret_3d",
        "ret_5d",
        "ret_10d",
        "ret_20d",
        "vol_5d",
        "vol_20d",
        "score_delta_1d",
        "score_delta_5d",
        "score_delta_accel",
        "ret_accel_5_20",
        "volume_ratio_5_20",
        "distance_to_20d_high",
        "distance_to_60d_high",
        "distance_to_20d_low",
        "volatility_expansion",
        "adv_ratio_5_20",
        *ALPHA_PRIOR_FRAME_NAMES,
        *[f"{base_name}_lag{lag}" for base_name in STATE_SEQUENCE_BASES for lag in STATE_SEQUENCE_LAGS],
    ]
    derived_frames = {
        name: pd.DataFrame(0.0, index=dates, columns=columns)
        for name in derived_names
    }
    for horizon in (1, 3, 5, 10, 20):
        derived_frames[f"ret_{horizon}d"] = close.pct_change(horizon).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    derived_frames["vol_5d"] = close.pct_change().rolling(5).std().fillna(0.0)
    derived_frames["vol_20d"] = close.pct_change().rolling(20).std().fillna(0.0)
    return PreparedPolicyInputs(
        universe=tuple(columns),
        pool_name="fixture",
        benchmark="BENCH",
        data_source="fixture",
        csv_folder="",
        start_date=dates.min().strftime("%Y%m%d"),
        end_date=dates.max().strftime("%Y%m%d"),
        requested_start_date=dates.min().strftime("%Y%m%d"),
        history_window=HistoryWindow(
            mode="fixture",
            requested_start_date=dates.min().strftime("%Y%m%d"),
            effective_start_date=dates.min().strftime("%Y%m%d"),
            end_date=dates.max().strftime("%Y%m%d"),
            required_trading_days=0,
        ),
        raw_cache_meta={},
        prepared_cache_meta={},
        close=close,
        open_=open_,
        high=high,
        low=low,
        volume=volume,
        amount=amount,
        benchmark_close=benchmark_close,
        benchmark_open=benchmark_open,
        score_none=score_none,
        score_v2=score_v2,
        score_blend=score_blend,
        feature_frames=feature_frames,
        market_features={},
        membership_frame=membership,
        rolling_pool_summary={},
        alpha_prior_summary={},
        derived_frames=derived_frames,
    )


def _tiny_static_context_schema(enabled: bool, *, stock_count: int) -> dict[str, object]:
    fields = ["symbol", "exchange", "industry", "liquidity_bucket", "price_bucket"]
    if not bool(enabled):
        return {"enabled": False, "fields": fields, "id_columns": [f"{field}_id" for field in fields], "vocab_sizes": {}}
    vocab_sizes = {
        "symbol": int(stock_count) + 1,
        "exchange": 3,
        "industry": 4,
        "liquidity_bucket": 6,
        "price_bucket": 6,
    }
    return {
        "enabled": True,
        "fields": fields,
        "id_columns": [f"{field}_id" for field in fields],
        "vocab_sizes": vocab_sizes,
        "embedding_defaults": {
            "symbol": 4,
            "exchange": 2,
            "industry": 3,
            "liquidity_bucket": 2,
            "price_bucket": 2,
            "dropout": 0.0,
        },
        "symbol_vocab_fingerprint": "tiny-symbol-vocab",
        "industry_vocab_fingerprint": "tiny-industry-vocab",
        "board_vocab_fingerprint": "tiny-board-vocab",
    }


def _tiny_manifest(
    *,
    dataset_mode: str,
    feature_count: int,
    lookback_days: int,
    horizon: int,
    cumulative_horizons: tuple[int, ...],
    static_context_schema: dict[str, object],
) -> dict[str, object]:
    feature_columns = [f"feature_{idx}" for idx in range(int(feature_count))]
    feature_manifest = {
        "feature_profile": "tiny_forecast_fixture_v1",
        "feature_count_before_cap": int(feature_count),
        "feature_count_after_cap": int(feature_count),
        "feature_group_counts": {"fixture": int(feature_count)},
    }
    return {
        "dataset_mode": str(dataset_mode),
        "feature_profile": "tiny_forecast_fixture_v1",
        "feature_manifest": feature_manifest,
        "feature_columns": feature_columns,
        "feature_count": int(feature_count),
        "lookback_days": int(lookback_days),
        "horizon": int(horizon),
        "forecast_horizon": int(horizon),
        "cumulative_horizons": [int(item) for item in cumulative_horizons],
        "rank_horizons": [int(item) for item in cumulative_horizons],
        "risk_horizons": [int(item) for item in cumulative_horizons],
        "sample_count": 0,
        "sample_count_by_role": {},
        "normalization": {
            "fit_role": "train_only",
            "method": "zscore",
            "feature_count": int(feature_count),
            "feature_mean": [0.0] * int(feature_count),
            "feature_std": [1.0] * int(feature_count),
        },
        "static_context_schema": dict(static_context_schema),
        "symbol_vocab_fingerprint": str(static_context_schema.get("symbol_vocab_fingerprint", "")),
        "industry_vocab_fingerprint": str(static_context_schema.get("industry_vocab_fingerprint", "")),
        "board_vocab_fingerprint": str(static_context_schema.get("board_vocab_fingerprint", "")),
    }


def _tiny_labels(
    *,
    row_count: int,
    horizon: int,
    cumulative_horizons: tuple[int, ...],
    role_values: np.ndarray,
    stock_positions: np.ndarray,
    date_positions: np.ndarray,
) -> dict[str, np.ndarray]:
    rows = np.arange(int(row_count), dtype=np.float32).reshape(-1, 1)
    steps = np.arange(1, int(horizon) + 1, dtype=np.float32).reshape(1, -1)
    stock_signal = (np.asarray(stock_positions, dtype=np.float32).reshape(-1, 1) - 1.5) * 0.0008
    date_signal = (np.asarray(date_positions, dtype=np.float32).reshape(-1, 1) % 7.0) * 0.00005
    role_shift = np.where(role_values.reshape(-1, 1) == "test", 0.00015, 0.0).astype(np.float32)
    daily = (0.00025 * steps + stock_signal + date_signal + role_shift + 0.00001 * (rows % 5)).astype(np.float32)

    horizon_indices = np.asarray(cumulative_horizons, dtype=np.int64) - 1
    cumulative_1toh = np.cumsum(daily, axis=1).astype(np.float32)
    y_cum = cumulative_1toh[:, horizon_indices].astype(np.float32)

    rank_by_horizon = np.zeros((int(row_count), len(cumulative_horizons)), dtype=np.float32)
    for pos in range(len(cumulative_horizons)):
        values = y_cum[:, pos]
        order = values.argsort(kind="mergesort")
        ranks = np.empty_like(values, dtype=np.float32)
        if len(values) > 1:
            ranks[order] = np.linspace(0.0, 1.0, len(values), dtype=np.float32)
        rank_by_horizon[:, pos] = ranks

    drawdown = (-0.015 - np.abs(stock_signal) - 0.0002 * np.arange(len(cumulative_horizons), dtype=np.float32).reshape(1, -1)).astype(np.float32)
    worst = (-0.010 - np.abs(stock_signal) * 0.5 - 0.0001 * np.arange(len(cumulative_horizons), dtype=np.float32).reshape(1, -1)).astype(np.float32)
    upside = (np.maximum(y_cum, 0.0) + 0.012 + stock_signal).astype(np.float32)
    return {
        "y_daily_excess": daily,
        "y_cum_excess": y_cum,
        "y_rank_by_horizon": rank_by_horizon,
        "y_rank_20d": rank_by_horizon[:, -1].astype(np.float32),
        "y_drawdown_by_horizon": drawdown,
        "y_worst_by_horizon": worst,
        "y_upside_by_horizon": upside,
        "y_max_drawdown_20d": drawdown[:, -1].astype(np.float32),
        "y_worst_1d_20d": worst[:, -1].astype(np.float32),
        "y_upside_20d": upside[:, -1].astype(np.float32),
    }


def _role_dates(*, train_days: int, validation_days: int, test_days: int) -> list[tuple[str, pd.Timestamp]]:
    result: list[tuple[str, pd.Timestamp]] = []
    for role, start, count in (
        ("train", "2019-01-02", train_days),
        ("validation", "2020-01-02", validation_days),
        ("test", "2021-01-04", test_days),
    ):
        for dt in pd.bdate_range(start, periods=int(count)):
            result.append((role, pd.Timestamp(dt).normalize()))
    return result


def make_tiny_forecast_sequence_dataset(
    *,
    lookback_days: int = 5,
    horizon: int = 20,
    cumulative_horizons: tuple[int, ...] | list[int] | str | None = None,
    feature_count: int = 6,
    stocks: tuple[str, ...] = ("AAA.SZ", "BBB.SH", "CCC.SZ", "DDD.SH"),
    train_days: int = 3,
    validation_days: int = 2,
    test_days: int = 2,
    include_static_context: bool = False,
) -> ForecastSequenceDataset:
    resolved_horizons = normalize_path20_cumulative_horizons(
        cumulative_horizons or PATH20_DEFAULT_CUMULATIVE_HORIZONS,
        horizon=int(horizon),
    )
    roles_and_dates = _role_dates(train_days=train_days, validation_days=validation_days, test_days=test_days)
    rows: list[tuple[str, pd.Timestamp, str, int, int]] = []
    for date_pos, (role, dt) in enumerate(roles_and_dates):
        for stock_pos, stock in enumerate(stocks):
            rows.append((role, dt, str(stock), date_pos, stock_pos))
    role_values = np.asarray([item[0] for item in rows], dtype=object)
    date_values = np.asarray([item[1].to_datetime64() for item in rows], dtype="datetime64[ns]")
    stock_values = np.asarray([item[2] for item in rows], dtype=object)
    date_positions = np.asarray([item[3] for item in rows], dtype=np.int32)
    stock_positions = np.asarray([item[4] for item in rows], dtype=np.int32)
    row_count = len(rows)

    time_axis = np.arange(int(lookback_days), dtype=np.float32).reshape(1, -1, 1)
    feature_axis = np.arange(int(feature_count), dtype=np.float32).reshape(1, 1, -1)
    row_axis = np.arange(row_count, dtype=np.float32).reshape(-1, 1, 1)
    x = (0.01 * row_axis + 0.10 * time_axis + 0.03 * feature_axis).astype(np.float32)

    labels = _tiny_labels(
        row_count=row_count,
        horizon=int(horizon),
        cumulative_horizons=resolved_horizons,
        role_values=role_values,
        stock_positions=stock_positions,
        date_positions=date_positions,
    )
    static_schema = _tiny_static_context_schema(include_static_context, stock_count=len(stocks))
    manifest = _tiny_manifest(
        dataset_mode="eager",
        feature_count=int(feature_count),
        lookback_days=int(lookback_days),
        horizon=int(horizon),
        cumulative_horizons=resolved_horizons,
        static_context_schema=static_schema,
    )
    manifest["sample_count"] = int(row_count)
    manifest["sample_count_by_role"] = {role: int(np.sum(role_values == role)) for role in ("train", "validation", "test")}
    static_context_ids = None
    if include_static_context:
        static_context_ids = np.column_stack(
            [
                stock_positions + 1,
                (stock_positions % 2) + 1,
                (stock_positions % 3) + 1,
                (stock_positions % 5) + 1,
                ((stock_positions + 1) % 5) + 1,
            ]
        ).astype(np.int64)
    return ForecastSequenceDataset(
        x=x,
        y_daily_excess=labels["y_daily_excess"],
        y_cum_excess=labels["y_cum_excess"],
        y_rank_by_horizon=labels["y_rank_by_horizon"],
        y_rank_20d=labels["y_rank_20d"],
        y_drawdown_by_horizon=labels["y_drawdown_by_horizon"],
        y_worst_by_horizon=labels["y_worst_by_horizon"],
        y_upside_by_horizon=labels["y_upside_by_horizon"],
        y_max_drawdown_20d=labels["y_max_drawdown_20d"],
        y_worst_1d_20d=labels["y_worst_1d_20d"],
        y_upside_20d=labels["y_upside_20d"],
        date=date_values,
        stock=stock_values,
        role=role_values,
        sequence_start_dates=date_values,
        label_end_dates=date_values,
        feature_columns=[f"feature_{idx}" for idx in range(int(feature_count))],
        normalization_manifest=dict(manifest["normalization"]),
        manifest=manifest,
        static_context_ids=static_context_ids,
    )


def _write_tiny_memmap(path: Path, values: np.ndarray) -> np.memmap:
    path.parent.mkdir(parents=True, exist_ok=True)
    mmap = np.memmap(path, dtype="float32", mode="w+", shape=values.shape)
    mmap[:] = np.asarray(values, dtype=np.float32)
    mmap.flush()
    return np.memmap(path, dtype="float32", mode="r+", shape=values.shape)


def make_tiny_forecast_memmap_dataset(
    root: Path,
    *,
    lookback_days: int = 5,
    horizon: int = 20,
    cumulative_horizons: tuple[int, ...] | list[int] | str | None = None,
    feature_count: int = 6,
    stocks: tuple[str, ...] = ("AAA.SZ", "BBB.SH", "CCC.SZ", "DDD.SH"),
    train_days: int = 3,
    validation_days: int = 2,
    test_days: int = 2,
    include_static_context: bool = False,
) -> ForecastMemmapDataset:
    root = Path(root)
    resolved_horizons = normalize_path20_cumulative_horizons(
        cumulative_horizons or PATH20_DEFAULT_CUMULATIVE_HORIZONS,
        horizon=int(horizon),
    )
    roles_and_dates = _role_dates(train_days=train_days, validation_days=validation_days, test_days=test_days)
    dates = [dt for _, dt in roles_and_dates]
    warmup_count = max(int(lookback_days) - 1, 0)
    warmup_dates = [pd.Timestamp("2018-12-03") + pd.offsets.BDay(idx) for idx in range(warmup_count)]
    date_strings = [pd.Timestamp(dt).strftime("%Y-%m-%d") for dt in [*warmup_dates, *dates]]
    date_count = len(date_strings)
    stock_count = len(stocks)

    date_axis = np.arange(date_count, dtype=np.float32).reshape(-1, 1, 1)
    stock_axis = np.arange(stock_count, dtype=np.float32).reshape(1, -1, 1)
    feature_axis = np.arange(int(feature_count), dtype=np.float32).reshape(1, 1, -1)
    panel = (0.02 * date_axis + 0.05 * stock_axis + 0.03 * feature_axis).astype(np.float32)
    feature_store = _write_tiny_memmap(root / "feature_store.dat", panel)

    sample_rows: list[dict[str, object]] = []
    role_values: list[str] = []
    stock_positions: list[int] = []
    date_positions: list[int] = []
    train_stocks = set(stocks)
    for sample_date_pos, (role, dt) in enumerate(roles_and_dates):
        date_pos = int(sample_date_pos) + int(warmup_count)
        for stock_pos, stock in enumerate(stocks):
            role_values.append(role)
            stock_positions.append(stock_pos)
            date_positions.append(date_pos)
            start_pos = int(date_pos) - int(lookback_days) + 1
            sample_rows.append(
                {
                    "role": role,
                    "date": dt.strftime("%Y-%m-%d"),
                    "stock": str(stock),
                    "date_pos": int(date_pos),
                    "stock_pos": int(stock_pos),
                    "sequence_start_pos": int(start_pos),
                    "sequence_start_date": date_strings[start_pos],
                    "label_end_date": date_strings[min(date_pos + int(horizon), date_count - 1)],
                    "lookback_days": int(lookback_days),
                    "stock_seen_in_train": bool(stock in train_stocks),
                    "history_valid_ratio": 1.0,
                    "history_bucket": "full",
                }
            )
    sample_index = pd.DataFrame(sample_rows)
    role_array = np.asarray(role_values, dtype=object)
    labels = _tiny_labels(
        row_count=len(sample_index),
        horizon=int(horizon),
        cumulative_horizons=resolved_horizons,
        role_values=role_array,
        stock_positions=np.asarray(stock_positions, dtype=np.int32),
        date_positions=np.asarray(date_positions, dtype=np.int32),
    )
    label_mmaps = {
        name: _write_tiny_memmap(root / f"{name}.dat", values)
        for name, values in {
            "daily_excess_return": labels["y_daily_excess"],
            "cumulative_excess_return": labels["y_cum_excess"],
            "rank_by_horizon": labels["y_rank_by_horizon"],
            "rank_20d": labels["y_rank_20d"],
            "drawdown_by_horizon": labels["y_drawdown_by_horizon"],
            "worst_by_horizon": labels["y_worst_by_horizon"],
            "upside_by_horizon": labels["y_upside_by_horizon"],
            "max_drawdown_20d": labels["y_max_drawdown_20d"],
            "worst_1d_20d": labels["y_worst_1d_20d"],
            "upside_20d": labels["y_upside_20d"],
        }.items()
    }
    static_schema = _tiny_static_context_schema(include_static_context, stock_count=len(stocks))
    manifest = _tiny_manifest(
        dataset_mode="memmap",
        feature_count=int(feature_count),
        lookback_days=int(lookback_days),
        horizon=int(horizon),
        cumulative_horizons=resolved_horizons,
        static_context_schema=static_schema,
    )
    manifest["sample_count"] = int(len(sample_index))
    manifest["sample_count_by_role"] = {role: int((sample_index["role"].astype(str) == role).sum()) for role in ("train", "validation", "test")}
    static_context_ids = None
    if include_static_context:
        stock_pos_array = np.asarray(stock_positions, dtype=np.int64)
        static_context_ids = np.column_stack(
            [
                stock_pos_array + 1,
                (stock_pos_array % 2) + 1,
                (stock_pos_array % 3) + 1,
                (stock_pos_array % 5) + 1,
                ((stock_pos_array + 1) % 5) + 1,
            ]
        ).astype(np.int64)
    return ForecastMemmapDataset(
        root=root,
        feature_store_path=root / "feature_store.dat",
        feature_store_shape=tuple(int(item) for item in feature_store.shape),
        sample_index=sample_index,
        y_daily_excess=label_mmaps["daily_excess_return"],
        y_cum_excess=label_mmaps["cumulative_excess_return"],
        y_rank_by_horizon=label_mmaps["rank_by_horizon"],
        y_rank_20d=label_mmaps["rank_20d"],
        y_drawdown_by_horizon=label_mmaps["drawdown_by_horizon"],
        y_worst_by_horizon=label_mmaps["worst_by_horizon"],
        y_upside_by_horizon=label_mmaps["upside_by_horizon"],
        y_max_drawdown_20d=label_mmaps["max_drawdown_20d"],
        y_worst_1d_20d=label_mmaps["worst_1d_20d"],
        y_upside_20d=label_mmaps["upside_20d"],
        feature_columns=[f"feature_{idx}" for idx in range(int(feature_count))],
        normalization_manifest=dict(manifest["normalization"]),
        manifest=manifest,
        feature_mean=np.zeros((int(feature_count),), dtype=np.float32),
        feature_std=np.ones((int(feature_count),), dtype=np.float32),
        date_values=np.asarray(date_strings, dtype=object),
        stock_values=np.asarray(stocks, dtype=object),
        static_context_ids=static_context_ids,
    )
