"""Full-universe dynamic hindsight values under the audited execution contract.

The relaxed Bellman recursion uses fractional shares and proportional costs so
that every legal variable-duration trade can be represented without enumerating
all entry/exit intervals.  Its selected path is then replayed with board lots,
minimum commissions, finite cash, and optional signal-amount capacity.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import psutil

from daily_research.path_policy import seq100_market_replay as replay
from daily_research.path_policy.seq100_exit_policy_audit import (
    _buy_order,
    _sell_order,
)

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STUDY_PATH = (
    WORKSPACE_ROOT / "daily_research/studies/seq100_dynamic_oracle_v1.json"
)
DEFAULT_OUTPUT_ROOT = WORKSPACE_ROOT / (
    "daily_research/output/path_policy/studies/seq100_dynamic_oracle_v1"
)
STUDY_ID = "seq100_dynamic_oracle_v1"
SCHEMA_VERSION = 1
TIE_TOLERANCE = 1.0e-12


def _resolve(path: str | Path) -> Path:
    return replay.resolve_path(path)


def _read_json(path: str | Path) -> dict[str, Any]:
    return replay.read_json(path)


def _sha256(path: str | Path) -> str:
    return replay.sha256(path)


def _write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    target = _resolve(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    temporary.write_text(
        json.dumps(dict(payload), ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, target)


def _write_frame(path: str | Path, frame: pd.DataFrame, *, row_group_size: int) -> None:
    target = _resolve(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    frame.to_parquet(
        temporary,
        index=False,
        compression="zstd",
        row_group_size=int(row_group_size),
    )
    os.replace(temporary, target)


def _write_array(path: str | Path, values: np.ndarray) -> None:
    target = _resolve(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp.npy")
    np.save(temporary, values, allow_pickle=False)
    os.replace(temporary, target)


def _file_record(path: str | Path, *, include_hash: bool = True) -> dict[str, Any]:
    target = _resolve(path)
    record: dict[str, Any] = {
        "path": str(target.resolve()),
        "size": int(target.stat().st_size),
    }
    if include_hash:
        record["sha256"] = _sha256(target)
    return record


def load_study(path: str | Path = DEFAULT_STUDY_PATH) -> dict[str, Any]:
    study = _read_json(path)
    if study.get("study_id") != STUDY_ID:
        raise ValueError("dynamic_oracle_study_id_mismatch")
    if bool(dict(study["oracle"])["fixed_holding_horizon_used"]):
        raise ValueError("dynamic_oracle_fixed_horizon_forbidden")
    if bool(dict(study["oracle"])["top_or_bottom_label_used"]):
        raise ValueError("dynamic_oracle_turning_label_forbidden")
    if str(dict(study["source"])["quality_pool_name"]) != "quality_liquidity_pit":
        raise ValueError("dynamic_oracle_quality_pool_mismatch")
    if int(dict(study["source"])["forbidden_year"]) != 2026:
        raise ValueError("dynamic_oracle_forbidden_year_mismatch")
    if tuple(dict(study["oracle"])["cost_scenarios"]) != replay.COST_SCENARIOS:
        raise ValueError("dynamic_oracle_cost_scenarios_mismatch")
    return study


@dataclass(frozen=True)
class RelaxedSolution:
    cost_scenario: str
    terminal_log_wealth: float
    cash_log: np.ndarray
    cash_action_symbol: np.ndarray
    hold_exit_policy: np.ndarray
    daily_cash: pd.DataFrame
    selected_trades: pd.DataFrame
    label_files: tuple[dict[str, Any], ...]
    label_summary: dict[str, Any]


class _AdvantageSummary:
    def __init__(self) -> None:
        self.buy: list[np.ndarray] = []
        self.hold: list[np.ndarray] = []
        self.rows = 0
        self.buy_valid = 0
        self.hold_comparable = 0
        self.hold_exit = 0

    def update(
        self,
        *,
        buy_advantage: np.ndarray,
        buy_valid: np.ndarray,
        hold_advantage: np.ndarray,
        hold_comparable: np.ndarray,
        oracle_exit: np.ndarray,
    ) -> None:
        self.rows += len(buy_advantage)
        self.buy_valid += int(buy_valid.sum())
        self.hold_comparable += int(hold_comparable.sum())
        self.hold_exit += int((oracle_exit & hold_comparable).sum())
        if bool(buy_valid.any()):
            self.buy.append(buy_advantage[buy_valid].astype(np.float32, copy=True))
        if bool(hold_comparable.any()):
            self.hold.append(
                hold_advantage[hold_comparable].astype(np.float32, copy=True)
            )

    @staticmethod
    def _describe(parts: Sequence[np.ndarray]) -> dict[str, Any]:
        if not parts:
            return {"rows": 0}
        values = np.concatenate(parts).astype(np.float64, copy=False)
        quantiles = np.quantile(values, [0.01, 0.1, 0.5, 0.9, 0.99])
        return {
            "rows": len(values),
            "mean": float(values.mean()),
            "positive_fraction": float((values > 0.0).mean()),
            "q01": float(quantiles[0]),
            "q10": float(quantiles[1]),
            "median": float(quantiles[2]),
            "q90": float(quantiles[3]),
            "q99": float(quantiles[4]),
        }

    def finish(self) -> dict[str, Any]:
        return {
            "quality_rows": self.rows,
            "buy_valid_rows": self.buy_valid,
            "holding_comparable_rows": self.hold_comparable,
            "oracle_exit_rows": self.hold_exit,
            "oracle_exit_fraction_when_comparable": (
                float(self.hold_exit / self.hold_comparable)
                if self.hold_comparable
                else 0.0
            ),
            "buy_advantage_vs_cash": self._describe(self.buy),
            "exit_advantage_vs_continue": self._describe(self.hold),
        }


def _finite_or_nan(values: np.ndarray) -> np.ndarray:
    return np.where(np.isfinite(values), values, np.nan).astype(np.float32)


def _label_frame(
    *,
    market: replay.ReplayMarket,
    absolute_idx: int,
    quality_symbols: np.ndarray,
    cash_selected_symbol: int,
    cash_stay_log: float,
    buy_values: np.ndarray,
    buy_valid: np.ndarray,
    continue_values: np.ndarray,
    exit_values: np.ndarray,
    oracle_exit: np.ndarray,
    next_sell: np.ndarray,
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    q = np.asarray(quality_symbols, dtype=np.int32)
    buy_advantage = np.full(len(q), np.nan, dtype=np.float64)
    buy_advantage[buy_valid] = buy_values[buy_valid] - float(cash_stay_log)
    hold_comparable = np.isfinite(continue_values) & np.isfinite(exit_values)
    hold_advantage = np.full(len(q), np.nan, dtype=np.float64)
    hold_advantage[hold_comparable] = (
        exit_values[hold_comparable] - continue_values[hold_comparable]
    )
    frame = pd.DataFrame(
        {
            "trade_date": np.full(len(q), str(market.date_values[absolute_idx])),
            "date_idx": np.full(len(q), int(absolute_idx), dtype=np.int32),
            "symbol_idx": q,
            "entry_filled_next_open": np.asarray(
                market.entry_filled[absolute_idx, q], dtype=bool
            ),
            "buy_action_valid": buy_valid,
            "oracle_cash_buy_selected": q == int(cash_selected_symbol),
            "buy_terminal_log_value": _finite_or_nan(buy_values),
            "cash_stay_terminal_log_value": np.full(
                len(q), float(cash_stay_log), dtype=np.float64
            ),
            "buy_advantage_vs_cash": _finite_or_nan(buy_advantage),
            "holding_continue_valid": np.isfinite(continue_values),
            "holding_exit_valid": np.isfinite(exit_values),
            "holding_actions_comparable": hold_comparable,
            "oracle_holding_exit": oracle_exit,
            "holding_continue_terminal_log_value": _finite_or_nan(continue_values),
            "holding_exit_terminal_log_value": _finite_or_nan(exit_values),
            "exit_advantage_vs_continue": _finite_or_nan(hold_advantage),
            "next_legal_sell_date_idx": next_sell.astype(np.int32),
        }
    )
    return frame, {
        "buy_advantage": buy_advantage,
        "buy_valid": buy_valid,
        "hold_advantage": hold_advantage,
        "hold_comparable": hold_comparable,
        "oracle_exit": oracle_exit,
    }


def _flush_label_year(
    *,
    output_root: Path,
    cost_scenario: str,
    year: int,
    frames: Sequence[pd.DataFrame],
    row_group_size: int,
) -> dict[str, Any]:
    if not frames:
        raise ValueError(f"dynamic_oracle_empty_label_year:{year}")
    output = pd.concat(frames, ignore_index=True).sort_values(
        ["date_idx", "symbol_idx"], kind="mergesort"
    )
    path = (
        output_root
        / "oracle_action_labels"
        / f"cost={cost_scenario}"
        / f"year={int(year)}"
        / "part-0000.parquet"
    )
    _write_frame(path, output, row_group_size=row_group_size)
    return {"year": int(year), "rows": len(output), **_file_record(path)}


def _next_sell_after(
    market: replay.ReplayMarket, *, absolute_idx: int, symbol_idx: int
) -> int:
    for candidate in range(int(absolute_idx) + 1, int(market.end_idx) + 1):
        price = float(market.exit_close_raw[candidate, int(symbol_idx)])
        if (
            bool(market.exit_sellable[candidate, int(symbol_idx)])
            and math.isfinite(price)
            and price > 0.0
        ):
            return candidate
    return -1


def _reconstruct_selected_trades(
    *,
    market: replay.ReplayMarket,
    cost_scenario: str,
    cash_action_symbol: np.ndarray,
    hold_exit_policy: np.ndarray,
    terminal_log_wealth: float,
) -> pd.DataFrame:
    buy_multiplier, sell_multiplier = replay.proportional_cost_multipliers(
        market, cost_scenario
    )
    rows: list[dict[str, Any]] = []
    absolute_idx = int(market.start_idx)
    held_symbol = -1
    signal_idx = -1
    entry_idx = -1
    cumulative_log = 0.0
    while absolute_idx <= int(market.end_idx):
        local = absolute_idx - int(market.start_idx)
        if held_symbol < 0:
            selected = int(cash_action_symbol[local])
            if selected < 0:
                absolute_idx += 1
                continue
            if absolute_idx + 1 > int(market.end_idx):
                raise AssertionError("dynamic oracle selected an entry beyond the terminal date")
            signal_idx = absolute_idx
            entry_idx = absolute_idx + 1
            if not bool(market.quality_mask[local, selected]):
                raise AssertionError("dynamic oracle selected a non-quality entry")
            if not bool(market.entry_filled[signal_idx, selected]):
                raise AssertionError("dynamic oracle selected an unfilled entry")
            held_symbol = selected
            absolute_idx = entry_idx
            continue

        if not bool(hold_exit_policy[local, held_symbol]):
            absolute_idx += 1
            continue
        exit_idx = _next_sell_after(
            market, absolute_idx=absolute_idx, symbol_idx=held_symbol
        )
        if exit_idx < 0:
            raise AssertionError("dynamic oracle exit has no legal resolution")
        entry_price = float(market.entry_open_raw[entry_idx, held_symbol])
        exit_price = float(market.exit_close_raw[exit_idx, held_symbol])
        factor = (
            exit_price
            * float(sell_multiplier[exit_idx - market.start_idx])
            / (entry_price * float(buy_multiplier))
        )
        if not math.isfinite(factor) or factor <= 0.0:
            raise AssertionError("dynamic oracle reconstructed a nonpositive trade factor")
        trade_log = math.log(factor)
        before = cumulative_log
        cumulative_log += trade_log
        rows.append(
            {
                "cost_scenario": str(cost_scenario),
                "trade_id": len(rows),
                "symbol_idx": int(held_symbol),
                "symbol": str(market.symbol_values[held_symbol]),
                "signal_date": str(market.date_values[signal_idx]),
                "entry_date": str(market.date_values[entry_idx]),
                "exit_request_date": str(market.date_values[absolute_idx]),
                "exit_date": str(market.date_values[exit_idx]),
                "signal_date_idx": int(signal_idx),
                "entry_date_idx": int(entry_idx),
                "exit_request_date_idx": int(absolute_idx),
                "exit_date_idx": int(exit_idx),
                "entry_price_raw": entry_price,
                "exit_price_raw": exit_price,
                "occupied_sessions": int(exit_idx - entry_idx + 1),
                "proportional_wealth_factor": float(factor),
                "proportional_log_return": float(trade_log),
                "cumulative_log_wealth_before": float(before),
                "cumulative_log_wealth_after": float(cumulative_log),
            }
        )
        held_symbol = -1
        signal_idx = -1
        entry_idx = -1
        absolute_idx = exit_idx
    if held_symbol >= 0:
        raise AssertionError("dynamic oracle reconstruction ended with an open position")
    if abs(cumulative_log - float(terminal_log_wealth)) > 2.0e-8:
        raise AssertionError(
            "dynamic oracle path does not reproduce terminal value: "
            f"{cumulative_log} != {terminal_log_wealth}"
        )
    return pd.DataFrame(rows)


def solve_relaxed_oracle(
    *,
    market: replay.ReplayMarket,
    study: Mapping[str, Any],
    cost_scenario: str,
    output_root: str | Path,
    write_outputs: bool = True,
) -> RelaxedSolution:
    root = _resolve(output_root)
    row_group_size = int(dict(study["resources"])["parquet_row_group_size"])
    day_count = market.day_count
    symbol_count = market.symbol_count
    buy_multiplier, sell_multiplier = replay.proportional_cost_multipliers(
        market, cost_scenario
    )

    cash_log = np.full(day_count, -np.inf, dtype=np.float64)
    cash_log[-1] = 0.0
    cash_action_symbol = np.full(day_count, -1, dtype=np.int32)
    hold_exit_policy = np.zeros((day_count, symbol_count), dtype=bool)
    future_hold = np.full(symbol_count, -np.inf, dtype=np.float64)
    next_sell = np.full(symbol_count, -1, dtype=np.int32)
    daily_rows: list[dict[str, Any]] = []
    label_records: list[dict[str, Any]] = []
    label_frames: list[pd.DataFrame] = []
    current_label_year = int(str(market.date_values[market.end_idx])[:4])
    advantages = _AdvantageSummary()

    terminal_q = np.flatnonzero(market.quality_mask[-1]).astype(np.int32)
    terminal_frame, terminal_values = _label_frame(
        market=market,
        absolute_idx=market.end_idx,
        quality_symbols=terminal_q,
        cash_selected_symbol=-1,
        cash_stay_log=0.0,
        buy_values=np.full(len(terminal_q), -np.inf),
        buy_valid=np.zeros(len(terminal_q), dtype=bool),
        continue_values=np.full(len(terminal_q), -np.inf),
        exit_values=np.full(len(terminal_q), -np.inf),
        oracle_exit=np.zeros(len(terminal_q), dtype=bool),
        next_sell=np.full(len(terminal_q), -1, dtype=np.int32),
    )
    label_frames.append(terminal_frame)
    advantages.update(**terminal_values)
    daily_rows.append(
        {
            "cost_scenario": str(cost_scenario),
            "trade_date": str(market.date_values[market.end_idx]),
            "date_idx": int(market.end_idx),
            "quality_candidates": len(terminal_q),
            "valid_buy_candidates": 0,
            "cash_action": "terminal_cash",
            "cash_selected_symbol_idx": -1,
            "cash_selected_symbol": None,
            "cash_terminal_log_value": 0.0,
            "stay_terminal_log_value": 0.0,
            "best_buy_terminal_log_value": None,
            "second_buy_terminal_log_value": None,
            "selected_advantage_over_stay": 0.0,
            "selected_action_margin": 0.0,
        }
    )

    terminal_close = np.asarray(
        market.exit_close_raw[market.end_idx], dtype=np.float64
    )
    terminal_sellable = (
        np.asarray(market.exit_sellable[market.end_idx], dtype=bool)
        & np.isfinite(terminal_close)
        & (terminal_close > 0.0)
    )
    next_sell[terminal_sellable] = int(market.end_idx)

    for local in range(day_count - 2, -1, -1):
        absolute_idx = int(market.start_idx + local)
        year = int(str(market.date_values[absolute_idx])[:4])
        if year != current_label_year:
            if write_outputs:
                label_records.append(
                    _flush_label_year(
                        output_root=root,
                        cost_scenario=cost_scenario,
                        year=current_label_year,
                        frames=label_frames,
                        row_group_size=row_group_size,
                    )
                )
            label_frames = []
            current_label_year = year

        mark_current = market.mark_close[local].astype(np.float64, copy=False)
        mark_next = market.mark_close[local + 1].astype(np.float64, copy=False)
        stay_log = float(cash_log[local + 1])
        q = np.flatnonzero(market.quality_mask[local]).astype(np.int32)
        entry_open = np.asarray(
            market.entry_open_raw[absolute_idx + 1, q], dtype=np.float64
        )
        entry_valid = (
            np.asarray(market.entry_filled[absolute_idx, q], dtype=bool)
            & np.isfinite(entry_open)
            & (entry_open > 0.0)
            & np.isfinite(mark_next[q])
            & (mark_next[q] > 0.0)
            & np.isfinite(future_hold[q])
        )
        buy_values = np.full(len(q), -np.inf, dtype=np.float64)
        buy_values[entry_valid] = (
            np.log(mark_next[q][entry_valid])
            - np.log(entry_open[entry_valid] * buy_multiplier)
            + future_hold[q][entry_valid]
        )
        valid_positions = np.flatnonzero(entry_valid)
        best_buy_value = -np.inf
        second_buy_value = -np.inf
        best_buy_symbol = -1
        if len(valid_positions):
            order = valid_positions[
                np.argsort(-buy_values[valid_positions], kind="mergesort")
            ]
            best_position = int(order[0])
            best_buy_value = float(buy_values[best_position])
            best_buy_symbol = int(q[best_position])
            if len(order) > 1:
                second_buy_value = float(buy_values[int(order[1])])
        if best_buy_value > stay_log + TIE_TOLERANCE:
            cash_log[local] = best_buy_value
            cash_action_symbol[local] = best_buy_symbol
            second_action = max(stay_log, second_buy_value)
            selected_margin = best_buy_value - second_action
            selected_advantage = best_buy_value - stay_log
            cash_action = "buy"
        else:
            cash_log[local] = stay_log
            cash_action_symbol[local] = -1
            selected_margin = stay_log - best_buy_value
            selected_advantage = 0.0
            cash_action = "remain_cash"

        continue_values = np.full(symbol_count, -np.inf, dtype=np.float64)
        continue_valid = (
            np.isfinite(mark_current)
            & (mark_current > 0.0)
            & np.isfinite(mark_next)
            & (mark_next > 0.0)
            & np.isfinite(future_hold)
        )
        continue_values[continue_valid] = (
            np.log(mark_next[continue_valid] / mark_current[continue_valid])
            + future_hold[continue_valid]
        )
        exit_values = np.full(symbol_count, -np.inf, dtype=np.float64)
        exit_symbols = np.flatnonzero(
            (next_sell >= 0) & np.isfinite(mark_current) & (mark_current > 0.0)
        )
        if len(exit_symbols):
            sale_dates = next_sell[exit_symbols]
            sale_prices = np.asarray(
                market.exit_close_raw[sale_dates, exit_symbols], dtype=np.float64
            )
            sale_local = sale_dates - int(market.start_idx)
            exit_values[exit_symbols] = (
                np.log(
                    sale_prices
                    * sell_multiplier[sale_local]
                    / mark_current[exit_symbols]
                )
                + cash_log[sale_local]
            )
        oracle_exit = exit_values > continue_values + TIE_TOLERANCE
        hold_current = np.maximum(continue_values, exit_values)
        hold_exit_policy[local] = oracle_exit

        q_frame, values = _label_frame(
            market=market,
            absolute_idx=absolute_idx,
            quality_symbols=q,
            cash_selected_symbol=int(cash_action_symbol[local]),
            cash_stay_log=stay_log,
            buy_values=buy_values,
            buy_valid=entry_valid,
            continue_values=continue_values[q],
            exit_values=exit_values[q],
            oracle_exit=oracle_exit[q],
            next_sell=next_sell[q],
        )
        label_frames.append(q_frame)
        advantages.update(**values)
        daily_rows.append(
            {
                "cost_scenario": str(cost_scenario),
                "trade_date": str(market.date_values[absolute_idx]),
                "date_idx": absolute_idx,
                "quality_candidates": len(q),
                "valid_buy_candidates": int(entry_valid.sum()),
                "cash_action": cash_action,
                "cash_selected_symbol_idx": int(cash_action_symbol[local]),
                "cash_selected_symbol": (
                    str(market.symbol_values[best_buy_symbol])
                    if cash_action == "buy"
                    else None
                ),
                "cash_terminal_log_value": float(cash_log[local]),
                "stay_terminal_log_value": stay_log,
                "best_buy_terminal_log_value": (
                    best_buy_value if math.isfinite(best_buy_value) else None
                ),
                "second_buy_terminal_log_value": (
                    second_buy_value if math.isfinite(second_buy_value) else None
                ),
                "selected_advantage_over_stay": float(selected_advantage),
                "selected_action_margin": (
                    float(selected_margin) if math.isfinite(selected_margin) else 0.0
                ),
            }
        )
        future_hold = hold_current

        current_close = np.asarray(
            market.exit_close_raw[absolute_idx], dtype=np.float64
        )
        current_sellable = (
            np.asarray(market.exit_sellable[absolute_idx], dtype=bool)
            & np.isfinite(current_close)
            & (current_close > 0.0)
        )
        next_sell[current_sellable] = absolute_idx

    if write_outputs:
        label_records.append(
            _flush_label_year(
                output_root=root,
                cost_scenario=cost_scenario,
                year=current_label_year,
                frames=label_frames,
                row_group_size=row_group_size,
            )
        )
    label_records.sort(key=lambda item: int(item["year"]))
    if sum(int(item["rows"]) for item in label_records) not in {
        0,
        int(market.quality_rows),
    }:
        raise AssertionError("dynamic oracle label row count mismatch")

    daily = pd.DataFrame(daily_rows).sort_values("date_idx", kind="mergesort")
    trades = _reconstruct_selected_trades(
        market=market,
        cost_scenario=cost_scenario,
        cash_action_symbol=cash_action_symbol,
        hold_exit_policy=hold_exit_policy,
        terminal_log_wealth=float(cash_log[0]),
    )
    if write_outputs:
        scenario_root = root / "relaxed_oracle" / f"cost={cost_scenario}"
        _write_frame(
            scenario_root / "daily_cash_actions.parquet",
            daily,
            row_group_size=row_group_size,
        )
        _write_frame(
            scenario_root / "selected_trades.parquet",
            trades,
            row_group_size=row_group_size,
        )
        _write_array(scenario_root / "cash_log_value.npy", cash_log)
        _write_array(scenario_root / "cash_action_symbol.npy", cash_action_symbol)
        _write_array(scenario_root / "holding_exit_policy.npy", hold_exit_policy)
    return RelaxedSolution(
        cost_scenario=str(cost_scenario),
        terminal_log_wealth=float(cash_log[0]),
        cash_log=cash_log,
        cash_action_symbol=cash_action_symbol,
        hold_exit_policy=hold_exit_policy,
        daily_cash=daily,
        selected_trades=trades,
        label_files=tuple(label_records),
        label_summary=advantages.finish(),
    )


def _maximum_drawdown(equity: np.ndarray) -> float:
    values = np.asarray(equity, dtype=np.float64)
    if len(values) == 0:
        return 0.0
    running = np.maximum.accumulate(values)
    drawdown = np.divide(
        values,
        running,
        out=np.ones_like(values),
        where=running > 0.0,
    ) - 1.0
    return float(drawdown.min())


def replay_finite_account(
    *,
    market: replay.ReplayMarket,
    selected_trades: pd.DataFrame,
    cost_scenario: str,
    starting_cash: float,
    capacity_mode: str,
    maximum_signal_amount_fraction: float,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    if capacity_mode not in {"uncapped", "signal_amount_0p5pct"}:
        raise ValueError(f"unknown dynamic oracle capacity mode: {capacity_mode}")
    cash = float(starting_cash)
    if not math.isfinite(cash) or cash <= 0.0:
        raise ValueError("dynamic oracle starting cash must be positive")
    slippage_multiplier = (
        1.0
        if str(cost_scenario) == "base"
        else float(market.costs.stress_slippage_multiplier)
    )
    schedule = selected_trades.sort_values(
        ["signal_date_idx", "trade_id"], kind="mergesort"
    ).reset_index(drop=True)
    entries = {int(row.entry_date_idx): row for row in schedule.itertuples(index=False)}
    exits = {int(row.exit_date_idx): row for row in schedule.itertuples(index=False)}
    if len(entries) != len(schedule) or len(exits) != len(schedule):
        raise ValueError("dynamic oracle selected path has overlapping entries or exits")

    active: dict[str, Any] | None = None
    last_mark = math.nan
    equity_rows: list[dict[str, Any]] = []
    trade_rows: list[dict[str, Any]] = []
    fees_and_slippage = 0.0
    turnover = 0.0
    missed_entries = 0
    capacity_capped = 0
    minimum_lot_failures = 0

    for absolute_idx in range(int(market.start_idx), int(market.end_idx) + 1):
        entry = entries.get(absolute_idx)
        if entry is not None:
            if active is not None:
                raise AssertionError("finite dynamic oracle attempted an overlapping entry")
            symbol_idx = int(entry.symbol_idx)
            signal_idx = int(entry.signal_date_idx)
            entry_price = float(market.entry_open_raw[absolute_idx, symbol_idx])
            signal_amount = float(market.amount_panel[signal_idx, symbol_idx])
            entry_filled = bool(market.entry_filled[signal_idx, symbol_idx])
            allocation = cash
            capacity_limit = math.inf
            if capacity_mode == "signal_amount_0p5pct":
                if not math.isfinite(signal_amount) or signal_amount <= 0.0:
                    allocation = 0.0
                    capacity_limit = 0.0
                else:
                    capacity_limit = (
                        signal_amount * float(maximum_signal_amount_fraction)
                    )
                    if capacity_limit < allocation:
                        capacity_capped += 1
                    allocation = min(allocation, capacity_limit)
            shares, buy_cash, buy_cost, buy_notional = _buy_order(
                available_cash=cash,
                allocated_cash=allocation,
                entry_price=entry_price,
                contract=market.costs,
                slippage_multiplier=slippage_multiplier,
            )
            if not entry_filled or shares <= 0:
                missed_entries += 1
                if entry_filled and allocation > 0.0:
                    minimum_lot_failures += 1
            else:
                cash -= buy_cash
                fees_and_slippage += buy_cost
                turnover += buy_notional
                active = {
                    "source_trade_id": int(entry.trade_id),
                    "symbol_idx": symbol_idx,
                    "symbol": str(entry.symbol),
                    "signal_date_idx": signal_idx,
                    "entry_date_idx": absolute_idx,
                    "expected_exit_date_idx": int(entry.exit_date_idx),
                    "shares": int(shares),
                    "entry_price_raw": entry_price,
                    "buy_cash": float(buy_cash),
                    "buy_notional": float(buy_notional),
                    "buy_cost": float(buy_cost),
                    "signal_amount": signal_amount,
                    "capacity_limit": capacity_limit,
                }
                last_mark = entry_price

        if active is not None:
            close = float(
                market.exit_close_raw[absolute_idx, int(active["symbol_idx"])]
            )
            if math.isfinite(close) and close > 0.0:
                last_mark = close

        exit_trade = exits.get(absolute_idx)
        if exit_trade is not None and active is not None:
            if int(exit_trade.trade_id) != int(active["source_trade_id"]):
                raise AssertionError("finite dynamic oracle exit identity mismatch")
            symbol_idx = int(active["symbol_idx"])
            exit_price = float(market.exit_close_raw[absolute_idx, symbol_idx])
            if not bool(market.exit_sellable[absolute_idx, symbol_idx]):
                raise AssertionError("finite dynamic oracle selected an unsellable exit")
            proceeds, sell_cost, sell_notional = _sell_order(
                shares=int(active["shares"]),
                exit_price=exit_price,
                exit_date_idx=absolute_idx,
                date_values=market.date_values,
                contract=market.costs,
                slippage_multiplier=slippage_multiplier,
            )
            cash += proceeds
            fees_and_slippage += sell_cost
            turnover += sell_notional
            trade_rows.append(
                {
                    "cost_scenario": str(cost_scenario),
                    "starting_cash_cny": float(starting_cash),
                    "capacity_mode": str(capacity_mode),
                    "source_trade_id": int(active["source_trade_id"]),
                    "symbol_idx": symbol_idx,
                    "symbol": str(active["symbol"]),
                    "signal_date": str(
                        market.date_values[int(active["signal_date_idx"])]
                    ),
                    "entry_date": str(
                        market.date_values[int(active["entry_date_idx"])]
                    ),
                    "exit_date": str(market.date_values[absolute_idx]),
                    "signal_date_idx": int(active["signal_date_idx"]),
                    "entry_date_idx": int(active["entry_date_idx"]),
                    "exit_date_idx": absolute_idx,
                    "shares": int(active["shares"]),
                    "entry_price_raw": float(active["entry_price_raw"]),
                    "exit_price_raw": exit_price,
                    "buy_cash_cny": float(active["buy_cash"]),
                    "buy_notional_cny": float(active["buy_notional"]),
                    "sell_proceeds_cny": float(proceeds),
                    "sell_notional_cny": float(sell_notional),
                    "buy_cost_cny": float(active["buy_cost"]),
                    "sell_cost_cny": float(sell_cost),
                    "net_pnl_cny": float(proceeds - float(active["buy_cash"])),
                    "net_return_on_buy_cash": float(
                        proceeds / float(active["buy_cash"]) - 1.0
                    ),
                    "signal_amount_cny": float(active["signal_amount"]),
                    "capacity_limit_cny": float(active["capacity_limit"]),
                    "signal_amount_participation": (
                        float(active["buy_notional"] / active["signal_amount"])
                        if math.isfinite(float(active["signal_amount"]))
                        and float(active["signal_amount"]) > 0.0
                        else math.nan
                    ),
                }
            )
            active = None
            last_mark = math.nan

        position_value = (
            float(active["shares"]) * float(last_mark)
            if active is not None and math.isfinite(last_mark)
            else 0.0
        )
        equity = cash + position_value
        equity_rows.append(
            {
                "cost_scenario": str(cost_scenario),
                "starting_cash_cny": float(starting_cash),
                "capacity_mode": str(capacity_mode),
                "trade_date": str(market.date_values[absolute_idx]),
                "date_idx": absolute_idx,
                "cash": float(cash),
                "position_value": float(position_value),
                "equity": float(equity),
                "position_count": int(active is not None),
                "capital_utilization": (
                    float(position_value / equity) if equity > 0.0 else 0.0
                ),
            }
        )

    if active is not None:
        raise AssertionError("finite dynamic oracle ended with an open position")
    equity = pd.DataFrame(equity_rows)
    trades = pd.DataFrame(trade_rows)
    ending_equity = float(equity.iloc[-1]["equity"])
    summary = {
        "cost_scenario": str(cost_scenario),
        "starting_cash_cny": float(starting_cash),
        "capacity_mode": str(capacity_mode),
        "ending_equity_cny": ending_equity,
        "total_return": float(ending_equity / float(starting_cash) - 1.0),
        "log_growth": float(math.log(ending_equity / float(starting_cash))),
        "maximum_drawdown": _maximum_drawdown(equity["equity"].to_numpy(float)),
        "mean_capital_utilization": float(equity["capital_utilization"].mean()),
        "selected_trade_count": len(schedule),
        "executed_trade_count": len(trades),
        "missed_entry_count": int(missed_entries),
        "minimum_lot_failure_count": int(minimum_lot_failures),
        "capacity_capped_entry_count": int(capacity_capped),
        "fees_and_slippage_cny": float(fees_and_slippage),
        "turnover_notional_cny": float(turnover),
        "turnover_to_starting_cash": float(turnover / float(starting_cash)),
        "maximum_participation": (
            float(trades["signal_amount_participation"].max())
            if not trades.empty
            else 0.0
        ),
    }
    return summary, equity, trades


def _relaxed_summary(solution: RelaxedSolution) -> dict[str, Any]:
    trades = solution.selected_trades
    if trades.empty:
        duration = np.asarray([], dtype=np.float64)
        returns = np.asarray([], dtype=np.float64)
    else:
        duration = trades["occupied_sessions"].to_numpy(np.float64)
        returns = trades["proportional_log_return"].to_numpy(np.float64)
    return {
        "cost_scenario": str(solution.cost_scenario),
        "terminal_log_wealth": float(solution.terminal_log_wealth),
        "terminal_log10_wealth_multiplier": float(
            solution.terminal_log_wealth / math.log(10.0)
        ),
        "selected_trade_count": len(trades),
        "positive_trade_fraction": float((returns > 0.0).mean()) if len(returns) else 0.0,
        "mean_trade_log_return": float(returns.mean()) if len(returns) else 0.0,
        "median_occupied_sessions": float(np.median(duration)) if len(duration) else 0.0,
        "mean_occupied_sessions": float(duration.mean()) if len(duration) else 0.0,
        "cash_buy_action_dates": int((solution.cash_action_symbol >= 0).sum()),
        **solution.label_summary,
    }


def run_study(
    *,
    study_path: str | Path = DEFAULT_STUDY_PATH,
    output_root: str | Path | None = None,
) -> dict[str, Any]:
    started = time.time()
    study_file = _resolve(study_path)
    study = load_study(study_file)
    root = _resolve(output_root or study["output_root"])
    root.mkdir(parents=True, exist_ok=True)
    market, market_audit = replay.load_market(study)
    runtime = {
        "available_memory_mb_at_start": int(psutil.virtual_memory().available / 1024**2),
        "logical_cpu_count": int(psutil.cpu_count(logical=True) or 1),
        "adaptive_memory": bool(dict(study["resources"])["adaptive_memory"]),
        "reserve_memory_mb": int(dict(study["resources"])["reserve_memory_mb"]),
    }
    market_manifest = {
        "schema": "seq100_market_replay/1",
        "status": "completed",
        "study_id": STUDY_ID,
        "study": _file_record(study_file),
        "audit": market_audit,
        "execution_costs": asdict(market.costs),
        "future_path_exposed_only_to_named_oracle": True,
        "causal_features_materialized": False,
        "training_performed": False,
    }
    _write_json(root / "market_manifest.json", market_manifest)

    solutions: list[RelaxedSolution] = []
    relaxed_rows: list[dict[str, Any]] = []
    oracle_records: list[dict[str, Any]] = []
    for cost_scenario in replay.COST_SCENARIOS:
        solution = solve_relaxed_oracle(
            market=market,
            study=study,
            cost_scenario=cost_scenario,
            output_root=root,
            write_outputs=True,
        )
        solutions.append(solution)
        relaxed_rows.append(_relaxed_summary(solution))
        scenario_root = root / "relaxed_oracle" / f"cost={cost_scenario}"
        record = {
            "cost_scenario": cost_scenario,
            "terminal_log_wealth": float(solution.terminal_log_wealth),
            "daily_cash_actions": _file_record(
                scenario_root / "daily_cash_actions.parquet"
            ),
            "selected_trades": _file_record(
                scenario_root / "selected_trades.parquet"
            ),
            "cash_log_value": _file_record(scenario_root / "cash_log_value.npy"),
            "cash_action_symbol": _file_record(
                scenario_root / "cash_action_symbol.npy"
            ),
            "holding_exit_policy": _file_record(
                scenario_root / "holding_exit_policy.npy"
            ),
            "action_labels": list(solution.label_files),
            "label_summary": solution.label_summary,
        }
        oracle_records.append(record)
    relaxed_summary = pd.DataFrame(relaxed_rows)
    row_group_size = int(dict(study["resources"])["parquet_row_group_size"])
    _write_frame(
        root / "relaxed_summary.parquet",
        relaxed_summary,
        row_group_size=row_group_size,
    )

    finite_rows: list[dict[str, Any]] = []
    finite_equity: list[pd.DataFrame] = []
    finite_trades: list[pd.DataFrame] = []
    finite = dict(study["finite_replay"])
    for solution in solutions:
        for starting_cash in (float(value) for value in finite["starting_cash_cny"]):
            for capacity_mode in (str(value) for value in finite["capacity_modes"]):
                summary, equity, trades = replay_finite_account(
                    market=market,
                    selected_trades=solution.selected_trades,
                    cost_scenario=solution.cost_scenario,
                    starting_cash=starting_cash,
                    capacity_mode=capacity_mode,
                    maximum_signal_amount_fraction=float(
                        finite["maximum_signal_day_amount_fraction"]
                    ),
                )
                relaxed_log = float(solution.terminal_log_wealth)
                summary["relaxed_terminal_log_wealth"] = relaxed_log
                summary["discrete_and_capacity_log_loss"] = float(
                    relaxed_log - float(summary["log_growth"])
                )
                finite_rows.append(summary)
                finite_equity.append(equity)
                if not trades.empty:
                    finite_trades.append(trades)
    finite_summary = pd.DataFrame(finite_rows)
    equity_output = pd.concat(finite_equity, ignore_index=True)
    trades_output = (
        pd.concat(finite_trades, ignore_index=True)
        if finite_trades
        else pd.DataFrame()
    )
    _write_frame(
        root / "finite_account_summary.parquet",
        finite_summary,
        row_group_size=row_group_size,
    )
    _write_frame(
        root / "finite_equity.parquet",
        equity_output,
        row_group_size=row_group_size,
    )
    _write_frame(
        root / "finite_trades.parquet",
        trades_output,
        row_group_size=row_group_size,
    )

    manifest = {
        "schema": "seq100_dynamic_oracle/1",
        "status": "completed",
        "study_id": STUDY_ID,
        "schema_version": SCHEMA_VERSION,
        "study": _file_record(study_file),
        "market_manifest": _file_record(root / "market_manifest.json"),
        "runtime": {
            **runtime,
            "elapsed_seconds": float(time.time() - started),
            "available_memory_mb_at_end": int(
                psutil.virtual_memory().available / 1024**2
            ),
        },
        "audit": {
            "quality_rows": int(market.quality_rows),
            "label_rows": int(
                sum(
                    int(item["rows"])
                    for record in oracle_records
                    for item in record["action_labels"]
                )
            ),
            "cost_scenarios": len(oracle_records),
            "finite_account_tasks": len(finite_summary),
            "forbidden_2026_rows": 0,
        },
        "relaxed_oracles": oracle_records,
        "outputs": {
            "relaxed_summary": _file_record(root / "relaxed_summary.parquet"),
            "finite_account_summary": _file_record(
                root / "finite_account_summary.parquet"
            ),
            "finite_equity": _file_record(root / "finite_equity.parquet"),
            "finite_trades": _file_record(root / "finite_trades.parquet"),
        },
        "fixed_holding_horizon_used": False,
        "future_path_used": True,
        "training_performed": False,
        "causal_policy_evaluated": False,
        "profit_claim_allowed": False,
        "production_policy_selected": False,
        "report_generation_performed": False,
    }
    if int(manifest["audit"]["label_rows"]) != 2 * int(market.quality_rows):
        raise AssertionError("dynamic oracle aggregate label count mismatch")
    _write_json(root / "analysis_manifest.json", manifest)
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build the full-universe variable-duration dynamic oracle."
    )
    parser.add_argument("--study", default=str(DEFAULT_STUDY_PATH))
    parser.add_argument("--output-root", default=None)
    args = parser.parse_args(argv)
    result = run_study(study_path=args.study, output_root=args.output_root)
    print(
        json.dumps(
            {
                "status": result["status"],
                "quality_rows": result["audit"]["quality_rows"],
                "label_rows": result["audit"]["label_rows"],
                "finite_account_tasks": result["audit"]["finite_account_tasks"],
                "elapsed_seconds": result["runtime"]["elapsed_seconds"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
