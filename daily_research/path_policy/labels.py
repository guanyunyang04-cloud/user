from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from daily_research.continuous_policy.portfolio_simulator import PortfolioState
from daily_research.continuous_policy.state_builder import PreparedPolicyInputs, build_cross_section_state


PATH20_HORIZON = 20
PATH20_QUANTILES: tuple[float, ...] = (0.10, 0.50, 0.90)
PATH20_CUMULATIVE_HORIZONS: tuple[int, ...] = (1, 3, 5, 10, 20)


def normalize_cumulative_horizons(
    cumulative_horizons: tuple[int, ...] | list[int] | str | None = None,
    *,
    horizon: int = PATH20_HORIZON,
) -> tuple[int, ...]:
    max_horizon = int(horizon or PATH20_HORIZON)
    if max_horizon <= 0:
        raise ValueError("horizon must be positive.")
    if cumulative_horizons is None:
        values = list(PATH20_CUMULATIVE_HORIZONS)
    elif isinstance(cumulative_horizons, str):
        values = [int(item.strip()) for item in cumulative_horizons.split(",") if item.strip()]
    else:
        values = [int(item) for item in cumulative_horizons]
    if not values:
        raise ValueError("cumulative_horizons must contain at least one horizon.")
    normalized = tuple(sorted(dict.fromkeys(values)))
    invalid = [item for item in normalized if int(item) <= 0 or int(item) > max_horizon]
    if invalid:
        raise ValueError(f"cumulative_horizons must be in [1, {max_horizon}], got {invalid}.")
    return normalized


@dataclass(frozen=True)
class Path20LabelBundle:
    daily_return: dict[int, pd.DataFrame]
    daily_excess_return: dict[int, pd.DataFrame]
    cumulative_return: dict[int, pd.DataFrame]
    cumulative_excess_return: dict[int, pd.DataFrame]
    path_max_drawdown_by_horizon: dict[int, pd.DataFrame]
    path_worst_1d_by_horizon: dict[int, pd.DataFrame]
    path_upside_capture_by_horizon: dict[int, pd.DataFrame]
    path_max_drawdown_20d: pd.DataFrame
    path_worst_1d_20d: pd.DataFrame
    path_upside_capture_20d: pd.DataFrame
    forward_rank: dict[int, pd.DataFrame]
    top_label: dict[int, pd.DataFrame]
    bottom_label: dict[int, pd.DataFrame]
    metadata: dict[str, Any]

    @property
    def max_forward_horizon(self) -> int:
        return int(self.metadata.get("max_forward_horizon", PATH20_HORIZON))


def _safe_div(numerator: pd.DataFrame | pd.Series, denominator: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
    return numerator.div(denominator.replace(0.0, np.nan)).replace([np.inf, -np.inf], np.nan)


def _future_entry_exit_returns(
    *,
    price: pd.DataFrame,
    benchmark: pd.Series,
    horizon: int,
    execution_mode: str,
) -> tuple[pd.DataFrame, pd.Series]:
    mode = str(execution_mode or "next_open").strip().lower()
    if mode == "next_open":
        entry = price.shift(-1)
        exit_ = price.shift(-(int(horizon) + 1))
        bench_entry = benchmark.shift(-1)
        bench_exit = benchmark.shift(-(int(horizon) + 1))
    elif mode == "close":
        entry = price
        exit_ = price.shift(-int(horizon))
        bench_entry = benchmark
        bench_exit = benchmark.shift(-int(horizon))
    else:
        raise ValueError(f"Unsupported path20 execution mode: {execution_mode!r}")
    stock_ret = _safe_div(exit_, entry).sub(1.0)
    bench_ret = _safe_div(bench_exit, bench_entry).sub(1.0)
    return stock_ret.astype(float), bench_ret.astype(float)


def _daily_forward_returns(
    *,
    price: pd.DataFrame,
    benchmark: pd.Series,
    horizon: int,
    execution_mode: str,
) -> tuple[pd.DataFrame, pd.Series]:
    mode = str(execution_mode or "next_open").strip().lower()
    if mode == "next_open":
        start = price.shift(-int(horizon))
        end = price.shift(-(int(horizon) + 1))
        bench_start = benchmark.shift(-int(horizon))
        bench_end = benchmark.shift(-(int(horizon) + 1))
    elif mode == "close":
        start = price.shift(-(int(horizon) - 1))
        end = price.shift(-int(horizon))
        bench_start = benchmark.shift(-(int(horizon) - 1))
        bench_end = benchmark.shift(-int(horizon))
    else:
        raise ValueError(f"Unsupported path20 execution mode: {execution_mode!r}")
    stock_ret = _safe_div(end, start).sub(1.0)
    bench_ret = _safe_div(bench_end, bench_start).sub(1.0)
    return stock_ret.astype(float), bench_ret.astype(float)


def _future_price_path(
    *,
    price: pd.DataFrame,
    execution_mode: str,
    horizon: int = PATH20_HORIZON,
) -> list[pd.DataFrame]:
    mode = str(execution_mode or "next_open").strip().lower()
    if mode == "next_open":
        return [price.shift(-(step + 1)) for step in range(0, int(horizon) + 1)]
    if mode == "close":
        return [price.shift(-step) for step in range(0, int(horizon) + 1)]
    raise ValueError(f"Unsupported path20 execution mode: {execution_mode!r}")


def _nan_reduce(values: np.ndarray, *, mode: str) -> np.ndarray:
    if mode == "min":
        filled = np.where(np.isfinite(values), values, np.inf)
        reduced = np.min(filled, axis=0)
        return np.where(np.isfinite(reduced), reduced, np.nan)
    if mode == "max":
        filled = np.where(np.isfinite(values), values, -np.inf)
        reduced = np.max(filled, axis=0)
        return np.where(np.isfinite(reduced), reduced, np.nan)
    raise ValueError(f"Unsupported nan reduce mode: {mode}")


def build_path20_labels(
    prepared: PreparedPolicyInputs,
    *,
    execution_mode: str = "next_open",
    horizon: int = PATH20_HORIZON,
    cumulative_horizons: tuple[int, ...] | list[int] | str | None = None,
    top_frac: float = 0.20,
    bottom_frac: float = 0.20,
) -> Path20LabelBundle:
    max_horizon = int(horizon or PATH20_HORIZON)
    resolved_horizons = normalize_cumulative_horizons(cumulative_horizons, horizon=max_horizon)
    price = prepared.open_ if str(execution_mode or "next_open").strip().lower() == "next_open" else prepared.close
    benchmark = (
        prepared.benchmark_open
        if str(execution_mode or "next_open").strip().lower() == "next_open"
        else prepared.benchmark_close
    )
    daily_return: dict[int, pd.DataFrame] = {}
    daily_excess_return: dict[int, pd.DataFrame] = {}
    for step in range(1, max_horizon + 1):
        stock_ret, bench_ret = _daily_forward_returns(
            price=price,
            benchmark=benchmark,
            horizon=step,
            execution_mode=execution_mode,
        )
        daily_return[step] = stock_ret
        daily_excess_return[step] = stock_ret.sub(bench_ret, axis=0)

    cumulative_return: dict[int, pd.DataFrame] = {}
    cumulative_excess_return: dict[int, pd.DataFrame] = {}
    forward_rank: dict[int, pd.DataFrame] = {}
    top_label: dict[int, pd.DataFrame] = {}
    bottom_label: dict[int, pd.DataFrame] = {}
    for step in resolved_horizons:
        stock_ret, bench_ret = _future_entry_exit_returns(
            price=price,
            benchmark=benchmark,
            horizon=step,
            execution_mode=execution_mode,
        )
        excess = daily_excess_return[1].copy()
        for daily_step in range(2, int(step) + 1):
            excess = excess.add(daily_excess_return[daily_step], fill_value=np.nan)
        cumulative_return[step] = stock_ret
        cumulative_excess_return[step] = excess
        rank = excess.rank(axis=1, pct=True)
        forward_rank[step] = rank
        top_label[step] = rank.ge(1.0 - float(top_frac)).astype(float).where(rank.notna())
        bottom_label[step] = rank.le(float(bottom_frac)).astype(float).where(rank.notna())

    price_path = _future_price_path(price=price, execution_mode=execution_mode, horizon=max_horizon)
    entry = price_path[0].replace(0.0, np.nan)
    rel_paths = [_safe_div(frame, entry) for frame in price_path[1:]]
    path_returns = [frame.sub(1.0) for frame in rel_paths]
    path_arrays = [frame.to_numpy(dtype=float, copy=False) for frame in path_returns]
    daily_arrays = [daily_return[step].to_numpy(dtype=float, copy=False) for step in range(1, max_horizon + 1)]
    path_max_drawdown_by_horizon: dict[int, pd.DataFrame] = {}
    path_worst_1d_by_horizon: dict[int, pd.DataFrame] = {}
    path_upside_capture_by_horizon: dict[int, pd.DataFrame] = {}
    for step in resolved_horizons:
        stacked = np.stack(path_arrays[: int(step)], axis=0)
        daily_stacked = np.stack(daily_arrays[: int(step)], axis=0)
        path_max_drawdown_by_horizon[int(step)] = pd.DataFrame(
            _nan_reduce(stacked, mode="min"),
            index=price.index,
            columns=price.columns,
        )
        path_worst_1d_by_horizon[int(step)] = pd.DataFrame(
            _nan_reduce(daily_stacked, mode="min"),
            index=price.index,
            columns=price.columns,
        )
        path_upside_capture_by_horizon[int(step)] = pd.DataFrame(
            _nan_reduce(stacked, mode="max"),
            index=price.index,
            columns=price.columns,
        )
    legacy_risk_horizon = 20 if 20 in path_max_drawdown_by_horizon else max(resolved_horizons)
    path_max_drawdown_20d = path_max_drawdown_by_horizon[int(legacy_risk_horizon)]
    path_worst_1d_20d = path_worst_1d_by_horizon[int(legacy_risk_horizon)]
    path_upside_capture_20d = path_upside_capture_by_horizon[int(legacy_risk_horizon)]
    metadata = {
        "execution_mode": str(execution_mode),
        "max_forward_horizon": int(max_horizon),
        "daily_horizons": list(range(1, max_horizon + 1)),
        "cumulative_horizons": list(resolved_horizons),
        "risk_horizons": list(resolved_horizons),
        "legacy_risk_horizon": int(legacy_risk_horizon),
        "top_frac": float(top_frac),
        "bottom_frac": float(bottom_frac),
        "label_semantics": "next_open_entry_to_future_open" if str(execution_mode) == "next_open" else "close_to_future_close",
    }
    return Path20LabelBundle(
        daily_return=daily_return,
        daily_excess_return=daily_excess_return,
        cumulative_return=cumulative_return,
        cumulative_excess_return=cumulative_excess_return,
        path_max_drawdown_by_horizon=path_max_drawdown_by_horizon,
        path_worst_1d_by_horizon=path_worst_1d_by_horizon,
        path_upside_capture_by_horizon=path_upside_capture_by_horizon,
        path_max_drawdown_20d=path_max_drawdown_20d,
        path_worst_1d_20d=path_worst_1d_20d,
        path_upside_capture_20d=path_upside_capture_20d,
        forward_rank=forward_rank,
        top_label=top_label,
        bottom_label=bottom_label,
        metadata=metadata,
    )


def path20_label_frame_for_date(bundle: Path20LabelBundle, date: pd.Timestamp | str) -> pd.DataFrame:
    dt = pd.Timestamp(date).normalize()
    rows: dict[str, pd.Series] = {}
    cumulative_horizons = normalize_cumulative_horizons(
        bundle.metadata.get("cumulative_horizons", PATH20_CUMULATIVE_HORIZONS),
        horizon=bundle.max_forward_horizon,
    )
    for step in range(1, bundle.max_forward_horizon + 1):
        rows[f"future_return_{step}d"] = bundle.daily_return[step].loc[dt]
        rows[f"future_excess_return_{step}d"] = bundle.daily_excess_return[step].loc[dt]
        rows[f"path_mu_{step}d"] = bundle.daily_excess_return[step].loc[dt]
        rows[f"path_q10_{step}d"] = bundle.daily_excess_return[step].loc[dt]
        rows[f"path_q50_{step}d"] = bundle.daily_excess_return[step].loc[dt]
        rows[f"path_q90_{step}d"] = bundle.daily_excess_return[step].loc[dt]
    for step in cumulative_horizons:
        rows[f"future_cum_return_{step}d"] = bundle.cumulative_return[step].loc[dt]
        rows[f"future_cum_excess_return_{step}d"] = bundle.cumulative_excess_return[step].loc[dt]
        rows[f"future_rank_{step}d"] = bundle.forward_rank[step].loc[dt]
        rows[f"future_top_label_{step}d"] = bundle.top_label[step].loc[dt]
        rows[f"future_bottom_label_{step}d"] = bundle.bottom_label[step].loc[dt]
        rows[f"future_path_max_drawdown_{step}d"] = bundle.path_max_drawdown_by_horizon[step].loc[dt]
        rows[f"future_path_worst_1d_{step}d"] = bundle.path_worst_1d_by_horizon[step].loc[dt]
        rows[f"future_path_upside_capture_{step}d"] = bundle.path_upside_capture_by_horizon[step].loc[dt]
    rows["future_path_max_drawdown_20d"] = bundle.path_max_drawdown_20d.loc[dt]
    rows["future_path_worst_1d_20d"] = bundle.path_worst_1d_20d.loc[dt]
    rows["future_path_upside_capture_20d"] = bundle.path_upside_capture_20d.loc[dt]
    frame = pd.DataFrame(rows)
    frame.index.name = "stock"
    return frame.reset_index()


def build_path20_dataset_frame(
    prepared: PreparedPolicyInputs,
    *,
    start_date: str,
    end_date: str = "",
    execution_mode: str = "next_open",
    include_state_features: bool = True,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    bundle = build_path20_labels(prepared, execution_mode=execution_mode)
    start_ts = pd.Timestamp(start_date).normalize()
    end_ts = pd.Timestamp(end_date).normalize() if str(end_date or "").strip() else prepared.close.index.max()
    dates = [
        pd.Timestamp(dt).normalize()
        for dt in prepared.close.index
        if pd.Timestamp(dt).normalize() >= start_ts and pd.Timestamp(dt).normalize() <= end_ts
    ]
    if len(dates) > bundle.max_forward_horizon:
        dates = dates[:-bundle.max_forward_horizon]
    frames = []
    empty_portfolio = PortfolioState()
    for dt in dates:
        label_frame = path20_label_frame_for_date(bundle, dt)
        label_frame.insert(0, "date", dt.strftime("%Y-%m-%d"))
        membership = prepared.membership_frame.reindex(index=[dt], columns=prepared.close.columns).iloc[0].fillna(False)
        label_frame["in_universe"] = label_frame["stock"].map(lambda stock: bool(membership.get(stock, False)))
        daily = label_frame.loc[label_frame["in_universe"]].copy()
        if include_state_features and not daily.empty:
            state_frame = build_cross_section_state(prepared, date=dt, portfolio_state=empty_portfolio)
            state_frame = state_frame.drop(columns=["date"], errors="ignore").copy()
            daily = daily.merge(state_frame, on="stock", how="left", suffixes=("", "_state"))
        frames.append(daily)
    dataset = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    manifest = {
        "status": "completed",
        "execution_mode": str(execution_mode),
        "start_date": str(start_date),
        "end_date": str(end_date),
        "max_forward_horizon": bundle.max_forward_horizon,
        "date_count": int(len(dates)),
        "row_count": int(len(dataset)),
        "include_state_features": bool(include_state_features),
        "label_metadata": dict(bundle.metadata),
    }
    return dataset, manifest
