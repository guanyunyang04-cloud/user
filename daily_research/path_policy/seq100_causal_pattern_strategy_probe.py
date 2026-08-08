"""Causal K-line entry and structure-driven exit screening.

The module evaluates a small frozen set of path-grammar candidates as legal
single-stock trades.  It is intentionally separate from the hindsight oracle:
signals are read from the causal structure panel, entry is the following open,
and an exit request is filled only at a later legal close.  No fixed holding
day is used.  The output is a trade-value diagnostic; it does not replay an
account or select a production policy.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import psutil
import pyarrow.parquet as pq

from daily_research.path_policy import seq100_dynamic_oracle as dynamic_oracle
from daily_research.path_policy import seq100_market_replay as replay
from daily_research.path_policy.seq100_exit_policy_audit import (
    _buy_order,
    _sell_order,
)

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STUDY_PATH = (
    WORKSPACE_ROOT
    / "daily_research/studies/seq100_causal_pattern_strategy_probe_v1.json"
)
DEFAULT_OUTPUT_ROOT = (
    WORKSPACE_ROOT
    / "daily_research/output/path_policy/studies/seq100_causal_pattern_strategy_probe_v1"
)
FORBIDDEN_YEAR = 2026
STRUCTURE_COLUMNS = (
    "date_idx",
    "symbol_idx",
    "pattern_zone_retest_hold_up",
    "pattern_up_exhaustion",
    "pattern_nested_up_reacceleration",
    "pattern_zone_breakout_up",
    "pattern_zone_reentry_after_up_breakout",
    "chan_breakout_active",
    "dc_small_event",
    "dc_medium_event",
    "dc_medium_mode",
    "dc_large_mode",
)
STRATEGY_NAMES = (
    "retest_invalidation",
    "retest_small_reversal",
    "retest_medium_reversal",
    "retest_post_exhaustion_exit",
    "retest_exhaustion_entry_small_reversal",
    "retest_exhaustion_entry_medium_reversal",
    "nested_reacceleration_small_reversal",
    "breakout_invalidation_control",
)
COST_SCENARIOS = replay.COST_SCENARIOS


def _resolve(value: str | Path) -> Path:
    path = Path(value)
    return (
        (WORKSPACE_ROOT / path).resolve() if not path.is_absolute() else path.resolve()
    )


def _read_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(_resolve(path).read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise TypeError(f"expected_json_object:{path}")
    return payload


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with _resolve(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        result = float(value)
        return result if math.isfinite(result) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    raise TypeError(f"json_value_not_serializable:{type(value).__name__}")


def _write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    target = _resolve(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".partial")
    temporary.write_text(
        json.dumps(dict(payload), ensure_ascii=False, indent=2, default=_json_default)
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, target)


def _write_frame(path: str | Path, frame: pd.DataFrame) -> None:
    target = _resolve(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".partial")
    frame.to_parquet(temporary, index=False, compression="zstd", row_group_size=100_000)
    os.replace(temporary, target)


def load_study(path: str | Path = DEFAULT_STUDY_PATH) -> dict[str, Any]:
    study = _read_json(path)
    if study.get("study_id") != "seq100_causal_pattern_strategy_probe_v1":
        raise ValueError("pattern_strategy_study_id_mismatch")
    source = dict(study.get("source", {}))
    if int(source.get("forbidden_year", -1)) != FORBIDDEN_YEAR:
        raise ValueError("pattern_strategy_forbidden_year_contract")
    if source.get("quality_pool_name") != "quality_liquidity_pit":
        raise ValueError("pattern_strategy_quality_pool_contract")
    execution = dict(study.get("execution", {}))
    if tuple(execution.get("cost_scenarios", ())) != COST_SCENARIOS:
        raise ValueError("pattern_strategy_cost_contract")
    period = dict(study.get("period", {}))
    if bool(study.get("boundaries", {}).get("fixed_holding_horizon_used")):
        raise ValueError("pattern_strategy_fixed_horizon_forbidden")
    if str(period.get("formal_end")) > str(source.get("maximum_outcome_date")):
        raise ValueError("pattern_strategy_period_after_source_end")
    if set(study.get("strategies", {})) != set(STRATEGY_NAMES):
        raise ValueError("pattern_strategy_strategy_contract")
    return study


@dataclass
class StructureArrays:
    retest: np.ndarray
    exhaustion: np.ndarray
    nested_reacceleration: np.ndarray
    breakout: np.ndarray
    reentry: np.ndarray
    breakout_active: np.ndarray
    small_event: np.ndarray
    medium_event: np.ndarray
    medium_mode: np.ndarray
    large_mode: np.ndarray
    rows: int
    duplicate_rows: int
    forbidden_rows: int


def _structure_panel_records(study: Mapping[str, Any]) -> list[dict[str, Any]]:
    source = dict(study["source"])
    manifest_path = _resolve(source["path_structure_manifest"])
    expected = str(source["expected_path_structure_manifest_sha256"])
    if _sha256(manifest_path) != expected:
        raise ValueError("pattern_strategy_structure_manifest_hash")
    manifest = _read_json(manifest_path)
    panel_path = _resolve(dict(manifest["panel_manifest"])["path"])
    panel = _read_json(panel_path)
    records = [dict(item) for item in panel.get("structure_panels", [])]
    if len(records) != 14:
        raise ValueError("pattern_strategy_structure_partition_count")
    return records


def load_structure_arrays(
    *, study: Mapping[str, Any], market: replay.ReplayMarket
) -> StructureArrays:
    day_count = market.day_count
    symbol_count = market.symbol_count
    arrays: dict[str, np.ndarray] = {
        "retest": np.zeros((day_count, symbol_count), dtype=bool),
        "exhaustion": np.zeros((day_count, symbol_count), dtype=bool),
        "nested_reacceleration": np.zeros((day_count, symbol_count), dtype=bool),
        "breakout": np.zeros((day_count, symbol_count), dtype=bool),
        "reentry": np.zeros((day_count, symbol_count), dtype=bool),
        "breakout_active": np.zeros((day_count, symbol_count), dtype=np.int8),
        "small_event": np.zeros((day_count, symbol_count), dtype=np.int8),
        "medium_event": np.zeros((day_count, symbol_count), dtype=np.int8),
        "medium_mode": np.zeros((day_count, symbol_count), dtype=np.int8),
        "large_mode": np.zeros((day_count, symbol_count), dtype=np.int8),
    }
    seen = np.zeros((day_count, symbol_count), dtype=bool)
    rows = 0
    duplicate_rows = 0
    forbidden_rows = 0
    records = _structure_panel_records(study)
    for record in records:
        path = _resolve(record["path"])
        table = pq.read_table(path, columns=list(STRUCTURE_COLUMNS))
        dates = table["date_idx"].to_numpy(zero_copy_only=False).astype(np.int64)
        symbols = table["symbol_idx"].to_numpy(zero_copy_only=False).astype(np.int64)
        if int(record.get("year", 0)) >= FORBIDDEN_YEAR:
            forbidden_rows += len(table)
        local = dates - int(market.start_idx)
        valid = (
            (local >= 0)
            & (local < day_count)
            & (symbols >= 0)
            & (symbols < symbol_count)
        )
        if not bool(valid.all()):
            raise ValueError("pattern_strategy_structure_domain")
        duplicate_rows += int(seen[local, symbols].sum())
        if duplicate_rows:
            raise ValueError("pattern_strategy_structure_duplicate_rows")
        seen[local, symbols] = True

        arrays["retest"][local, symbols] = (
            table["pattern_zone_retest_hold_up"]
            .to_numpy(zero_copy_only=False)
            .astype(bool)
        )
        arrays["exhaustion"][local, symbols] = (
            table["pattern_up_exhaustion"].to_numpy(zero_copy_only=False).astype(bool)
        )
        arrays["nested_reacceleration"][local, symbols] = (
            table["pattern_nested_up_reacceleration"]
            .to_numpy(zero_copy_only=False)
            .astype(bool)
        )
        arrays["breakout"][local, symbols] = (
            table["pattern_zone_breakout_up"]
            .to_numpy(zero_copy_only=False)
            .astype(bool)
        )
        arrays["reentry"][local, symbols] = (
            table["pattern_zone_reentry_after_up_breakout"]
            .to_numpy(zero_copy_only=False)
            .astype(bool)
        )
        for target, source in (
            ("breakout_active", "chan_breakout_active"),
            ("small_event", "dc_small_event"),
            ("medium_event", "dc_medium_event"),
            ("medium_mode", "dc_medium_mode"),
            ("large_mode", "dc_large_mode"),
        ):
            arrays[target][local, symbols] = (
                table[source].to_numpy(zero_copy_only=False).astype(np.int8)
            )
        rows += len(table)
        del table
    return StructureArrays(
        **arrays,
        rows=rows,
        duplicate_rows=duplicate_rows,
        forbidden_rows=forbidden_rows,
    )


def _hac(values: np.ndarray, lag: int = 20) -> tuple[float, float]:
    x = np.asarray(values, dtype=np.float64)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return math.nan, math.nan
    mean = float(x.mean())
    if len(x) == 1:
        return mean, math.nan
    centered = x - mean
    maximum_lag = min(int(lag), len(x) - 1)
    long_run = float(np.dot(centered, centered) / len(x))
    for offset in range(1, maximum_lag + 1):
        covariance = float(np.dot(centered[offset:], centered[:-offset]) / len(x))
        long_run += 2.0 * (1.0 - offset / (maximum_lag + 1.0)) * covariance
    return mean, math.sqrt(max(long_run, 0.0) / len(x))


def _bh(values: np.ndarray) -> np.ndarray:
    p = np.asarray(values, dtype=np.float64)
    result = np.full(p.shape, np.nan, dtype=np.float64)
    finite = np.isfinite(p)
    positions = np.flatnonzero(finite)
    if len(positions) == 0:
        return result
    order = positions[np.argsort(p[positions], kind="mergesort")]
    adjusted = np.empty(len(order), dtype=np.float64)
    running = 1.0
    for rank in range(len(order) - 1, -1, -1):
        value = p[order[rank]] * len(order) / float(rank + 1)
        running = min(running, value)
        adjusted[rank] = min(running, 1.0)
    result[order] = adjusted
    return result


def _normal_one_sided_p(mean: float, se: float) -> float:
    if not math.isfinite(mean) or not math.isfinite(se) or se <= 0.0:
        return math.nan
    # P(Z >= observed z) for a positive observed mean.
    return 0.5 * math.erfc(float(mean / se) / math.sqrt(2.0))


def _entry_mask(structure: StructureArrays, name: str) -> np.ndarray:
    if name == "retest":
        return structure.retest
    if name == "retest_exhaustion":
        return structure.retest & structure.exhaustion
    if name == "nested_reacceleration":
        return structure.nested_reacceleration
    if name == "breakout_control":
        return structure.breakout
    raise KeyError(name)


def _exit_mask(
    structure: StructureArrays,
    strategy: str,
    day: int,
    *,
    entry_signal_abs: np.ndarray,
    absolute_day: int,
) -> np.ndarray:
    active_invalid = structure.breakout_active[day] != 1
    if strategy == "retest_invalidation" or strategy == "breakout_invalidation_control":
        return active_invalid
    if strategy in {"retest_small_reversal", "retest_exhaustion_entry_small_reversal"}:
        return active_invalid | (structure.small_event[day] == -1)
    if strategy in {
        "retest_medium_reversal",
        "retest_exhaustion_entry_medium_reversal",
    }:
        return active_invalid | (structure.medium_event[day] == -1)
    if strategy == "retest_post_exhaustion_exit":
        return active_invalid | (
            structure.exhaustion[day] & (absolute_day > entry_signal_abs)
        )
    if strategy == "nested_reacceleration_small_reversal":
        return (
            (structure.medium_mode[day] != 1)
            | (structure.large_mode[day] != 1)
            | (structure.small_event[day] == -1)
        )
    raise KeyError(strategy)


def _pending_fill_mask(
    market: replay.ReplayMarket,
    *,
    pending: np.ndarray,
    signal_abs: np.ndarray,
    absolute_day: int,
) -> tuple[np.ndarray, np.ndarray]:
    symbol_count = len(pending)
    fill = np.zeros(symbol_count, dtype=bool)
    pending_symbols = np.flatnonzero(pending)
    if len(pending_symbols):
        fill[pending_symbols] = np.asarray(
            market.entry_filled[
                signal_abs[pending_symbols].astype(np.int64), pending_symbols
            ],
            dtype=bool,
        )
    prices = np.asarray(market.entry_open_raw[absolute_day], dtype=np.float64)
    fill &= np.isfinite(prices) & (prices > 0.0)
    return fill, prices


def simulate_strategy(
    *,
    market: replay.ReplayMarket,
    structure: StructureArrays,
    strategy: str,
    liquidation_start: str,
    starting_cash: float,
) -> tuple[pd.DataFrame, dict[str, int]]:
    entry_name = {
        "retest_invalidation": "retest",
        "retest_small_reversal": "retest",
        "retest_medium_reversal": "retest",
        "retest_post_exhaustion_exit": "retest",
        "retest_exhaustion_entry_small_reversal": "retest_exhaustion",
        "retest_exhaustion_entry_medium_reversal": "retest_exhaustion",
        "nested_reacceleration_small_reversal": "nested_reacceleration",
        "breakout_invalidation_control": "breakout_control",
    }[strategy]
    signals = _entry_mask(structure, entry_name)
    liquidation_abs = int(
        np.flatnonzero(market.date_values.astype(str) == str(liquidation_start))[0]
    )
    liquidation_local = liquidation_abs - int(market.start_idx)
    day_count, symbol_count = market.day_count, market.symbol_count
    state = np.zeros(
        symbol_count, dtype=np.int8
    )  # 0 flat, 1 pending, 2 hold, 3 exit requested
    signal_abs = np.full(symbol_count, -1, dtype=np.int32)
    entry_abs = np.full(symbol_count, -1, dtype=np.int32)
    request_abs = np.full(symbol_count, -1, dtype=np.int32)
    entry_price = np.full(symbol_count, np.nan, dtype=np.float64)
    max_close = np.full(symbol_count, np.nan, dtype=np.float64)
    min_close = np.full(symbol_count, np.nan, dtype=np.float64)
    records: list[dict[str, Any]] = []
    counts = {
        "signals_seen": 0,
        "entry_filled": 0,
        "entry_unfilled": 0,
        "exit_requests": 0,
        "terminal_unresolved": 0,
    }

    def append_exit(symbol: int, exit_abs: int, terminal: bool) -> None:
        ep = float(entry_price[symbol])
        xp = (
            float(market.exit_close_raw[exit_abs, symbol])
            if exit_abs >= 0
            else math.nan
        )
        records.append(
            {
                "strategy": strategy,
                "symbol_idx": int(symbol),
                "symbol": str(market.symbol_values[symbol]),
                "signal_date_idx": int(signal_abs[symbol]),
                "entry_date_idx": int(entry_abs[symbol]),
                "exit_request_date_idx": int(request_abs[symbol]),
                "exit_date_idx": int(exit_abs),
                "entry_price_raw": ep,
                "exit_price_raw": xp,
                "holding_sessions": int(exit_abs - entry_abs[symbol] + 1)
                if exit_abs >= 0
                else -1,
                "mfe_close": float(max_close[symbol] / ep - 1.0)
                if ep > 0.0
                else math.nan,
                "mae_close": float(min_close[symbol] / ep - 1.0)
                if ep > 0.0
                else math.nan,
                "terminal_liquidation": bool(terminal),
                "terminal_unresolved": bool(exit_abs < 0),
            }
        )

    for day in range(day_count):
        absolute_day = int(market.start_idx + day)
        pending = state == 1
        if bool(pending.any()):
            fill, prices = _pending_fill_mask(
                market,
                pending=pending,
                signal_abs=signal_abs,
                absolute_day=absolute_day,
            )
            failed = pending & ~fill
            counts["entry_unfilled"] += int(failed.sum())
            state[failed] = 0
            signal_abs[failed] = -1
            if bool(fill.any()):
                counts["entry_filled"] += int(fill.sum())
                state[fill] = 2
                entry_abs[fill] = absolute_day
                entry_price[fill] = prices[fill]
                max_close[fill] = prices[fill]
                min_close[fill] = prices[fill]

        requested = state == 3
        if bool(requested.any()):
            close = np.asarray(market.exit_close_raw[absolute_day], dtype=np.float64)
            sell = requested & np.asarray(
                market.exit_sellable[absolute_day], dtype=bool
            )
            sell &= np.isfinite(close) & (close > 0.0)
            for symbol in np.flatnonzero(sell):
                append_exit(int(symbol), absolute_day, terminal=False)
            state[sell] = 0
            signal_abs[sell] = -1
            entry_abs[sell] = -1
            request_abs[sell] = -1
            entry_price[sell] = np.nan
            max_close[sell] = np.nan
            min_close[sell] = np.nan

        held = (state == 2) | (state == 3)
        if bool(held.any()):
            close = np.asarray(market.exit_close_raw[absolute_day], dtype=np.float64)
            valid = held & np.isfinite(close) & (close > 0.0)
            max_close[valid] = np.maximum(max_close[valid], close[valid])
            min_close[valid] = np.minimum(min_close[valid], close[valid])

        active_hold = state == 2
        if bool(active_hold.any()):
            trigger = _exit_mask(
                structure,
                strategy,
                day,
                entry_signal_abs=signal_abs,
                absolute_day=absolute_day,
            )
            request = active_hold & trigger
            if day >= liquidation_local:
                request |= active_hold
            if bool(request.any()):
                state[request] = 3
                request_abs[request] = absolute_day
                counts["exit_requests"] += int(request.sum())

        # The liquidation boundary is known before the study starts.  No new
        # signal is allowed to create a position that would cross it.
        if day + 1 < liquidation_local:
            flat = state == 0
            take = flat & signals[day]
            counts["signals_seen"] += int(take.sum())
            state[take] = 1
            signal_abs[take] = absolute_day

    terminal_abs = int(market.end_idx)
    remaining = (state == 2) | (state == 3)
    if bool(remaining.any()):
        close = np.asarray(market.exit_close_raw[terminal_abs], dtype=np.float64)
        sell = remaining & np.asarray(market.exit_sellable[terminal_abs], dtype=bool)
        sell &= np.isfinite(close) & (close > 0.0)
        for symbol in np.flatnonzero(sell):
            append_exit(int(symbol), terminal_abs, terminal=True)
        unresolved = remaining & ~sell
        for symbol in np.flatnonzero(unresolved):
            append_exit(int(symbol), -1, terminal=True)
        counts["terminal_unresolved"] += int(unresolved.sum())

    frame = pd.DataFrame.from_records(records)
    if frame.empty:
        return frame, counts
    frame["signal_date"] = frame["signal_date_idx"].map(
        lambda value: str(market.date_values[int(value)])
    )
    frame["entry_date"] = frame["entry_date_idx"].map(
        lambda value: str(market.date_values[int(value)])
    )
    frame["exit_date"] = frame["exit_date_idx"].map(
        lambda value: str(market.date_values[int(value)]) if int(value) >= 0 else ""
    )
    frame["signal_year"] = frame["signal_date"].str.slice(0, 4).astype(int)
    for cost in COST_SCENARIOS:
        buy_multiplier, sell_multiplier = replay.proportional_cost_multipliers(
            market, cost
        )
        finite_net: list[float] = []
        finite_filled: list[bool] = []
        proportional_net: list[float] = []
        for row in frame.itertuples(index=False):
            if int(row.exit_date_idx) < 0:
                finite_net.append(-1.0)
                finite_filled.append(True)
                proportional_net.append(-1.0)
                continue
            shares, buy_cash, _buy_cost, _buy_notional = _buy_order(
                available_cash=float(starting_cash),
                allocated_cash=float(starting_cash),
                entry_price=float(row.entry_price_raw),
                contract=market.costs,
                slippage_multiplier=(
                    1.0 if cost == "base" else market.costs.stress_slippage_multiplier
                ),
            )
            if shares <= 0 or buy_cash <= 0.0:
                # An unaffordable board lot leaves the finite account in cash.
                # The proportional diagnostic remains defined for comparison.
                finite_net.append(0.0)
                finite_filled.append(False)
                factor = (
                    float(row.exit_price_raw)
                    * float(sell_multiplier[int(row.exit_date_idx) - market.start_idx])
                    / (float(row.entry_price_raw) * float(buy_multiplier))
                )
                proportional_net.append(factor - 1.0)
                continue
            proceeds, _sell_cost, _sell_notional = _sell_order(
                shares=shares,
                exit_price=float(row.exit_price_raw),
                exit_date_idx=int(row.exit_date_idx),
                date_values=market.date_values,
                contract=market.costs,
                slippage_multiplier=(
                    1.0 if cost == "base" else market.costs.stress_slippage_multiplier
                ),
            )
            finite_net.append(
                (float(starting_cash) - buy_cash + proceeds) / float(starting_cash)
                - 1.0
            )
            finite_filled.append(True)
            factor = (
                float(row.exit_price_raw)
                * float(sell_multiplier[int(row.exit_date_idx) - market.start_idx])
                / (float(row.entry_price_raw) * float(buy_multiplier))
            )
            proportional_net.append(factor - 1.0)
        frame[f"net_return_{cost}"] = np.asarray(finite_net, dtype=np.float64)
        frame[f"finite_order_filled_{cost}"] = np.asarray(finite_filled, dtype=bool)
        frame[f"proportional_net_return_{cost}"] = np.asarray(
            proportional_net, dtype=np.float64
        )
        frame[f"gross_return_{cost}"] = (
            frame["exit_price_raw"] / frame["entry_price_raw"] - 1.0
        )
    return frame, counts


def _summary(
    frame: pd.DataFrame, study: Mapping[str, Any]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if frame.empty:
        return pd.DataFrame(), pd.DataFrame()
    period = dict(study["period"])
    primary_years = {int(value) for value in period["primary_years"]}
    work = frame.loc[frame["signal_year"].isin(primary_years)].copy()
    rows: list[dict[str, Any]] = []
    annual_rows: list[dict[str, Any]] = []
    lag = int(dict(study["evaluation"])["hac_lag"])
    minimum_trades = int(dict(study["evaluation"])["minimum_primary_trades"])
    minimum_dates = int(dict(study["evaluation"])["minimum_primary_dates"])
    for strategy, group in work.groupby("strategy", sort=False):
        date_groups = group.groupby("signal_date", sort=True)
        for cost in COST_SCENARIOS:
            daily = date_groups[f"net_return_{cost}"].mean()
            mean, se = _hac(daily.to_numpy(float), lag)
            daily_frame = daily.rename("mean_net_return").reset_index()
            daily_frame["signal_year"] = (
                daily_frame["signal_date"].str.slice(0, 4).astype(int)
            )
            annual = daily_frame.groupby("signal_year")["mean_net_return"].mean()
            for year, value in annual.items():
                year_rows = group["signal_year"] == year
                annual_rows.append(
                    {
                        "strategy": strategy,
                        "cost_scenario": cost,
                        "signal_year": int(year),
                        "trades": int(year_rows.sum()),
                        "dates": int(group.loc[year_rows, "signal_date"].nunique()),
                        "mean_net_return": float(value),
                    }
                )
            p = _normal_one_sided_p(mean, se)
            rows.append(
                {
                    "strategy": strategy,
                    "cost_scenario": cost,
                    "trades": len(group),
                    "dates": int(group["signal_date"].nunique()),
                    "symbols": int(group["symbol_idx"].nunique()),
                    "finite_orders_filled": int(
                        group[f"finite_order_filled_{cost}"].sum()
                    ),
                    "mean_net_return": mean,
                    "hac_se": se,
                    "hac_lcb95": mean - 1.96 * se if math.isfinite(se) else math.nan,
                    "one_sided_p": p,
                    "mean_gross_return": float(group[f"gross_return_{cost}"].mean()),
                    "mean_proportional_net_return": float(
                        group[f"proportional_net_return_{cost}"].mean()
                    ),
                    "positive_trade_fraction": float(
                        (group[f"net_return_{cost}"] > 0.0).mean()
                    ),
                    "median_holding_sessions": float(
                        group["holding_sessions"].replace(-1, np.nan).median()
                    ),
                    "mean_mfe_close": float(group["mfe_close"].mean()),
                    "mean_mae_close": float(group["mae_close"].mean()),
                    "terminal_unresolved": int(group["terminal_unresolved"].sum()),
                    "minimum_sample_pass": len(group) >= minimum_trades
                    and group["signal_date"].nunique() >= minimum_dates,
                    "positive_year_count": int((annual > 0.0).sum()),
                }
            )
    summary = pd.DataFrame(rows)
    summary["bh_q"] = np.nan
    for cost, indices in summary.groupby("cost_scenario").groups.items():
        positions = np.asarray(list(indices), dtype=np.int64)
        summary.loc[positions, "bh_q"] = _bh(
            summary.loc[positions, "one_sided_p"].to_numpy(float)
        )
    summary["promotion_pass"] = (
        summary["minimum_sample_pass"]
        & (summary["hac_lcb95"] > 0.0)
        & (
            summary["positive_year_count"]
            >= int(dict(study["evaluation"])["majority_positive_years"])
        )
        & (summary["bh_q"] <= 0.05)
    )
    return summary.sort_values(
        ["cost_scenario", "strategy"], kind="mergesort"
    ), pd.DataFrame(annual_rows)


def run_study(
    *,
    study_path: str | Path = DEFAULT_STUDY_PATH,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    force: bool = False,
) -> dict[str, Any]:
    started = time.perf_counter()
    study_file = _resolve(study_path)
    study = load_study(study_file)
    root = _resolve(output_root)
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "manifest.json"
    if manifest_path.is_file() and not force:
        return _read_json(manifest_path)
    oracle_study = dynamic_oracle.load_study(
        _resolve(study["source"]["dynamic_oracle_study"])
    )
    market, market_audit = replay.load_market(oracle_study)
    available_start = int(psutil.virtual_memory().available / 1024**2)
    structure = load_structure_arrays(study=study, market=market)
    count_records: list[dict[str, Any]] = []
    frames: list[pd.DataFrame] = []
    for strategy in STRATEGY_NAMES:
        frame, counts = simulate_strategy(
            market=market,
            structure=structure,
            strategy=strategy,
            liquidation_start=str(study["period"]["terminal_liquidation_start"]),
            starting_cash=float(study["execution"]["starting_cash_cny_per_trade"]),
        )
        if not frame.empty:
            frames.append(frame)
        count_records.append({"strategy": strategy, **counts, "trades": len(frame)})
        print(
            json.dumps(
                {"strategy": strategy, "trades": len(frame), **counts},
                ensure_ascii=False,
            ),
            flush=True,
        )
    trades = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if not trades.empty:
        trades = trades.sort_values(
            ["strategy", "signal_date_idx", "symbol_idx"], kind="mergesort"
        ).reset_index(drop=True)
    summary, annual = _summary(trades, study)
    _write_frame(root / "trades.parquet", trades)
    _write_frame(root / "summary.parquet", summary)
    _write_frame(root / "annual_summary.parquet", annual)
    counts_frame = pd.DataFrame(count_records)
    _write_frame(root / "execution_counts.parquet", counts_frame)
    gate = {
        "status": "passed"
        if bool(summary.get("promotion_pass", pd.Series(dtype=bool)).any())
        else "failed",
        "account_replay_allowed": False,
        "production_policy_allowed": False,
        "passed_candidates": summary.loc[
            summary["promotion_pass"], ["strategy", "cost_scenario"]
        ].to_dict("records")
        if not summary.empty
        else [],
        "reason": "trade-level probe only; account replay is intentionally disabled even if a row passes",
    }
    _write_json(root / "promotion_gate.json", gate)
    minimum_available = min(
        available_start, int(psutil.virtual_memory().available / 1024**2)
    )
    manifest = {
        "schema": "seq100_causal_pattern_strategy_probe_manifest/1",
        "status": "completed",
        "study_id": study["study_id"],
        "study": {"path": str(study_file), "sha256": _sha256(study_file)},
        "market_audit": market_audit,
        "structure_rows": structure.rows,
        "structure_duplicate_rows": structure.duplicate_rows,
        "structure_forbidden_rows": structure.forbidden_rows,
        "trades": len(trades),
        "strategies": list(STRATEGY_NAMES),
        "cost_scenarios": list(COST_SCENARIOS),
        "outputs": {
            "trades": str((root / "trades.parquet").resolve()),
            "summary": str((root / "summary.parquet").resolve()),
            "annual_summary": str((root / "annual_summary.parquet").resolve()),
            "execution_counts": str((root / "execution_counts.parquet").resolve()),
            "promotion_gate": str((root / "promotion_gate.json").resolve()),
        },
        "runtime": {
            "elapsed_seconds": float(time.perf_counter() - started),
            "minimum_available_memory_mb": int(minimum_available),
        },
        "boundaries": dict(study["boundaries"]),
    }
    _write_json(manifest_path, manifest)
    del structure, market
    gc.collect()
    return manifest


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the causal K-line strategy probe."
    )
    parser.add_argument("--study-path", default=str(DEFAULT_STUDY_PATH))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = build_arg_parser().parse_args(argv)
    result = run_study(
        study_path=args.study_path,
        output_root=args.output_root,
        force=bool(args.force),
    )
    print(
        json.dumps(result, ensure_ascii=False, indent=2, default=_json_default)
        if args.json
        else result["outputs"]["summary"]
    )
    return result


if __name__ == "__main__":
    main()
