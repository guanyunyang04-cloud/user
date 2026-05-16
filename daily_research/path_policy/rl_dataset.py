from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any

import numpy as np
import pandas as pd

from daily_research.continuous_policy.pipeline_utils import select_feature_columns, signal_dates_between
from daily_research.continuous_policy.portfolio_simulator import PortfolioState
from daily_research.continuous_policy.state_builder import PreparedPolicyInputs, build_cross_section_state
from daily_research.path_policy import ALPHA_PATH20_SEQUENCE_POLICY_VERSION


DEFAULT_RL_REWARD_PROFILE = "net_excess_turnover_drawdown_v1"
DEFAULT_FULL_YEAR_WINDOWS: dict[int, tuple[str, str]] = {
    2019: ("20190101", "20191231"),
    2020: ("20200101", "20201231"),
    2022: ("20220101", "20221231"),
    2024: ("20240101", "20241231"),
}
FORBIDDEN_RL_INPUT_PREFIXES: tuple[str, ...] = (
    "future_",
    "oracle_path20_",
    "path_q",
    "path_mu_",
)
FORBIDDEN_RL_INPUT_COLUMNS: frozenset[str] = frozenset(
    {
        "future_path_max_drawdown_20d",
        "future_path_worst_1d_20d",
        "future_path_upside_capture_20d",
    }
)


@dataclass(frozen=True)
class Path20TrajectoryDataset:
    daily_frames: list[pd.DataFrame]
    feature_columns: list[str]
    dates: list[pd.Timestamp]
    manifest: dict[str, Any]

    @property
    def empty(self) -> bool:
        return not self.daily_frames

    def to_long_frame(self) -> pd.DataFrame:
        return pd.concat(self.daily_frames, ignore_index=True) if self.daily_frames else pd.DataFrame()


def stable_sequence_dataset_id(payload: dict[str, Any]) -> str:
    normalized = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
    digest = hashlib.sha1(normalized.encode("utf-8")).hexdigest()[:24]
    return f"alpha_path20_sequence_dataset__{digest}"


def full_year_window(year: int | str) -> tuple[str, str]:
    resolved = int(year)
    if resolved not in DEFAULT_FULL_YEAR_WINDOWS:
        raise ValueError(f"Unsupported fixed path20 sequence year: {year!r}")
    return DEFAULT_FULL_YEAR_WINDOWS[resolved]


def leakage_guard_for_feature_columns(columns: list[str]) -> dict[str, Any]:
    forbidden = [
        str(column)
        for column in columns
        if str(column) in FORBIDDEN_RL_INPUT_COLUMNS
        or any(str(column).startswith(prefix) for prefix in FORBIDDEN_RL_INPUT_PREFIXES)
    ]
    return {
        "status": "passed" if not forbidden else "failed",
        "forbidden_column_count": int(len(forbidden)),
        "forbidden_columns": forbidden,
        "forbidden_prefixes": list(FORBIDDEN_RL_INPUT_PREFIXES),
    }


def validate_no_oracle_or_future_inputs(columns: list[str]) -> dict[str, Any]:
    guard = leakage_guard_for_feature_columns(columns)
    if guard["status"] != "passed":
        raise ValueError(f"path20 sequence leakage guard failed: {guard['forbidden_columns']}")
    return guard


def _clean_numeric(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    return frame[columns].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)


def _next_return_panels(
    prepared: PreparedPolicyInputs,
    *,
    execution_mode: str,
) -> tuple[pd.DataFrame, pd.Series]:
    mode = str(execution_mode or "next_open").strip().lower()
    if mode == "next_open":
        price = prepared.open_
        benchmark = prepared.benchmark_open
    elif mode == "close":
        price = prepared.close
        benchmark = prepared.benchmark_close
    else:
        raise ValueError(f"Unsupported path20 sequence execution mode: {execution_mode!r}")
    stock_return = price.shift(-1).div(price).sub(1.0)
    benchmark_return = benchmark.shift(-1).div(benchmark).sub(1.0)
    return stock_return.astype(float), benchmark_return.astype(float)


def _portfolio_feature_row(portfolio: PortfolioState) -> dict[str, float]:
    features = portfolio.portfolio_features()
    return {
        "cash_weight": float(features.get("cash_weight", 1.0) or 1.0),
        "holding_count": float(features.get("holding_count", 0.0) or 0.0),
        "gross_exposure": float(features.get("gross_exposure", 0.0) or 0.0),
        "turnover_5d_mean": float(features.get("turnover_5d_mean", 0.0) or 0.0),
        "realized_return_5d": float(features.get("realized_return_5d", 0.0) or 0.0),
        "realized_vol_20d": float(features.get("realized_vol_20d", 0.0) or 0.0),
        "cash_change_5d": float(features.get("cash_change_5d", 0.0) or 0.0),
        "cash_deficit": float(features.get("cash_deficit", 0.0) or 0.0),
    }


def _resolve_feature_columns(sample: pd.DataFrame, max_feature_columns: int) -> list[str]:
    candidates = [
        column
        for column in select_feature_columns(sample)
        if column not in {"date", "stock", "in_universe"}
    ]
    validate_no_oracle_or_future_inputs([str(column) for column in candidates])
    return [str(column) for column in candidates[: int(max_feature_columns)]]


def build_path20_sequence_trajectory_dataset(
    prepared: PreparedPolicyInputs,
    *,
    start_date: str,
    end_date: str,
    lake_dataset_id: str = "",
    year: int | str | None = None,
    sequence_length: int = 20,
    execution_mode: str = "next_open",
    reward_profile: str = DEFAULT_RL_REWARD_PROFILE,
    max_feature_columns: int = 96,
    min_trading_days: int = 180,
) -> Path20TrajectoryDataset:
    stock_next_return, benchmark_next_return = _next_return_panels(prepared, execution_mode=execution_mode)
    dates = signal_dates_between(prepared, start_date=start_date, end_date=end_date, max_forward_horizon=1)
    portfolio = PortfolioState()
    daily_frames: list[pd.DataFrame] = []
    feature_columns: list[str] | None = None
    portfolio_feature_columns = list(_portfolio_feature_row(portfolio).keys())
    for dt in dates:
        state_frame = build_cross_section_state(prepared, date=dt, portfolio_state=portfolio)
        state_frame = state_frame.copy()
        if feature_columns is None:
            feature_columns = _resolve_feature_columns(state_frame, max_feature_columns=max_feature_columns)
        membership = prepared.membership_frame.reindex(index=[dt], columns=prepared.close.columns).iloc[0].fillna(False).astype(bool)
        next_ret = stock_next_return.loc[dt].reindex(state_frame["stock"].astype(str)).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        bench_ret = float(benchmark_next_return.loc[dt]) if dt in benchmark_next_return.index else 0.0
        portfolio_features = _portfolio_feature_row(portfolio)
        frame_columns = list(dict.fromkeys(["date", "stock", "current_weight", *feature_columns]))
        frame = state_frame[frame_columns].copy()
        frame["tradable_mask"] = frame["stock"].astype(str).map(lambda stock: bool(membership.get(stock, False))).astype(float)
        frame["next_open_return"] = frame["stock"].astype(str).map(lambda stock: float(next_ret.get(stock, 0.0)))
        frame["benchmark_return"] = float(bench_ret if np.isfinite(bench_ret) else 0.0)
        frame["next_open_excess_return"] = frame["next_open_return"].astype(float) - float(frame["benchmark_return"].iloc[0])
        frame["reward_profile"] = str(reward_profile)
        for key, value in portfolio_features.items():
            frame[f"portfolio_{key}"] = float(value)
        daily_frames.append(frame)
    resolved_features = feature_columns or []
    leakage_guard = validate_no_oracle_or_future_inputs(resolved_features)
    actual_start = dates[0].strftime("%Y-%m-%d") if dates else ""
    actual_end = dates[-1].strftime("%Y-%m-%d") if dates else ""
    dataset_id = stable_sequence_dataset_id(
        {
            "policy_version": ALPHA_PATH20_SEQUENCE_POLICY_VERSION,
            "lake_dataset_id": str(lake_dataset_id or ""),
            "year": int(year) if year is not None and str(year).strip() else "",
            "requested_start_date": str(start_date),
            "requested_end_date": str(end_date),
            "actual_signal_start_date": actual_start,
            "actual_signal_end_date": actual_end,
            "sequence_length": int(sequence_length),
            "feature_count": int(len(resolved_features)),
            "trading_day_count": int(len(dates)),
            "reward_profile": str(reward_profile),
        }
    )
    incomplete = int(len(dates)) < int(min_trading_days)
    manifest = {
        "status": "incomplete" if incomplete else "completed",
        "incomplete_reason": "trading_day_count_below_180" if incomplete else "",
        "policy_version": ALPHA_PATH20_SEQUENCE_POLICY_VERSION,
        "dataset_id": dataset_id,
        "lake_dataset_id": str(lake_dataset_id or ""),
        "stage": "path20_sequence_trajectory_dataset",
        "year": int(year) if year is not None and str(year).strip() else "",
        "requested_start_date": str(start_date),
        "requested_end_date": str(end_date),
        "actual_signal_start_date": actual_start,
        "actual_signal_end_date": actual_end,
        "trading_day_count": int(len(dates)),
        "sequence_length": int(sequence_length),
        "feature_columns": resolved_features,
        "feature_count": int(len(resolved_features)),
        "portfolio_feature_columns": portfolio_feature_columns,
        "reward_profile": str(reward_profile),
        "execution_mode": str(execution_mode),
        "leakage_guard": leakage_guard,
        "loose_latest_allowed": False,
        "shadow_only": True,
        "promotion_allowed": False,
    }
    return Path20TrajectoryDataset(
        daily_frames=daily_frames,
        feature_columns=resolved_features,
        dates=dates,
        manifest=manifest,
    )


def build_sequence_tensors(
    trajectory: Path20TrajectoryDataset,
    *,
    sequence_length: int,
) -> dict[str, np.ndarray]:
    if trajectory.empty:
        return {
            "state": np.zeros((0, int(sequence_length), 0, 0), dtype=np.float32),
            "portfolio": np.zeros((0, int(sequence_length), 8), dtype=np.float32),
            "mask": np.zeros((0, 0), dtype=bool),
            "current_weight": np.zeros((0, 0), dtype=np.float32),
            "future_return": np.zeros((0, 0), dtype=np.float32),
            "benchmark_return": np.zeros((0,), dtype=np.float32),
        }
    seq_len = int(sequence_length)
    frames = trajectory.daily_frames
    feature_columns = trajectory.feature_columns
    stocks = [str(item) for item in frames[0]["stock"].astype(str).tolist()]
    portfolio_columns = [column for column in frames[0].columns if str(column).startswith("portfolio_")]
    samples = []
    portfolio_samples = []
    masks = []
    current_weights = []
    future_returns = []
    benchmark_returns = []
    dates = []
    for idx in range(seq_len - 1, len(frames)):
        window = frames[idx - seq_len + 1 : idx + 1]
        state_block = []
        portfolio_block = []
        for daily in window:
            daily_indexed = daily.set_index("stock").reindex(stocks)
            state_block.append(_clean_numeric(daily_indexed, feature_columns).to_numpy(dtype=np.float32))
            portfolio_block.append(_clean_numeric(daily_indexed, portfolio_columns).iloc[0].to_numpy(dtype=np.float32))
        current_daily = frames[idx].set_index("stock").reindex(stocks)
        samples.append(np.stack(state_block, axis=0))
        portfolio_samples.append(np.stack(portfolio_block, axis=0))
        masks.append(current_daily["tradable_mask"].astype(float).fillna(0.0).to_numpy(dtype=float) > 0.5)
        current_weights.append(current_daily["current_weight"].astype(float).fillna(0.0).to_numpy(dtype=np.float32))
        future_returns.append(current_daily["next_open_excess_return"].astype(float).fillna(0.0).to_numpy(dtype=np.float32))
        benchmark_returns.append(float(current_daily["benchmark_return"].iloc[0]))
        dates.append(str(current_daily["date"].iloc[0]))
    return {
        "state": np.stack(samples, axis=0).astype(np.float32),
        "portfolio": np.stack(portfolio_samples, axis=0).astype(np.float32),
        "mask": np.stack(masks, axis=0).astype(bool),
        "current_weight": np.stack(current_weights, axis=0).astype(np.float32),
        "future_return": np.stack(future_returns, axis=0).astype(np.float32),
        "benchmark_return": np.asarray(benchmark_returns, dtype=np.float32),
        "dates": np.asarray(dates, dtype=object),
        "stocks": np.asarray(stocks, dtype=object),
    }
