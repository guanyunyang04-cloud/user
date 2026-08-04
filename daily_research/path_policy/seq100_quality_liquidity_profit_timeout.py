"""Finite-capital take-profit and timeout research for the quality pool model."""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from daily_research.path_policy import seq100_quality_liquidity_execution as base
from daily_research.path_policy import seq100_v4_economic_realizability as economic

WORKSPACE_ROOT = base.WORKSPACE_ROOT
STUDY_ID = "seq100_quality_liquidity_profit_timeout"
DEFAULT_STUDY_PATH = (
    WORKSPACE_ROOT
    / "daily_research/studies/seq100_quality_liquidity_profit_timeout.json"
)
DEFAULT_OUTPUT_ROOT = base.DEFAULT_OUTPUT_ROOT / "profit_timeout"
TASK_SCHEMA = "seq100_quality_liquidity_profit_timeout_task/1"
MANIFEST_SCHEMA = "seq100_quality_liquidity_profit_timeout_manifest/1"
AUDIT_SCHEMA = "seq100_quality_liquidity_profit_timeout_audit/1"
YEARS = (2023, 2024, 2025)
FAMILY = "dual_mfe_risk_veto"
SLOT_COUNTS = (6, 12)
TIMEOUT_DAYS = (10, 20)
TAKE_PROFIT_LEVELS: tuple[float | None, ...] = (None, 0.05, 0.08, 0.10)
COST_SCENARIOS = ("base", "stress")
STARTING_CASH_CNY = 1_000_000.0


def _resolve(value: str | Path) -> Path:
    return base._resolve(value)


def _load_json(path: Path) -> dict[str, Any]:
    return base._load_json(path)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    base._write_json(path, payload)


def _write_parquet(path: Path, frame: pd.DataFrame) -> None:
    base._write_parquet(path, frame)


def _file_record(path: Path, **extra: Any) -> dict[str, Any]:
    return base._file_record(path, **extra)


def _study_hash(path: Path) -> str:
    return economic._sha256(path)


def _emit(event: str, **payload: Any) -> None:
    base._emit(event, **payload)


def load_study(path: Path = DEFAULT_STUDY_PATH) -> dict[str, Any]:
    study = _load_json(path)
    if study.get("study_id") != STUDY_ID:
        raise ValueError(f"study_id must be {STUDY_ID}")
    period = dict(study["period"])
    if tuple(int(value) for value in period["signal_years"]) != YEARS:
        raise ValueError("signal years changed")
    if int(period["forbidden_year"]) != 2026:
        raise ValueError("2026 must remain forbidden")
    if str(period["last_entry_signal_date"]) != "2025-12-03":
        raise ValueError("last entry signal date changed")
    if str(period["last_mark_date"]) != "2025-12-31":
        raise ValueError("last mark date changed")
    signal = dict(study["signal"])
    if signal.get("family") != FAMILY or int(signal["feature_count"]) != 557:
        raise ValueError("canonical signal contract changed")
    if bool(signal["future_buyability_used_for_selection"]):
        raise ValueError("future buyability must not enter candidate selection")
    account = dict(study["account"])
    if tuple(int(value) for value in account["slot_counts"]) != SLOT_COUNTS:
        raise ValueError("slot counts changed")
    if not math.isclose(
        float(account["starting_cash_cny"]),
        STARTING_CASH_CNY,
        rel_tol=0.0,
        abs_tol=1.0e-9,
    ):
        raise ValueError("starting cash changed")
    policies = dict(study["policies"])
    if tuple(int(value) for value in policies["timeout_days"]) != TIMEOUT_DAYS:
        raise ValueError("timeout days changed")
    observed_levels = tuple(
        None if value is None else float(value)
        for value in policies["take_profit_levels"]
    )
    if observed_levels != TAKE_PROFIT_LEVELS or policies.get("stop_loss") is not None:
        raise ValueError("profit-timeout barrier contract changed")
    if int(policies["take_profit_active_from_day"]) != 2:
        raise ValueError("take profit must respect T+1")
    execution = dict(study["execution"])
    if tuple(str(value) for value in execution["cost_scenarios"]) != COST_SCENARIOS:
        raise ValueError("cost scenarios changed")
    expected = (
        len(SLOT_COUNTS)
        * len(TIMEOUT_DAYS)
        * len(TAKE_PROFIT_LEVELS)
        * len(COST_SCENARIOS)
    )
    if int(study["outputs"]["expected_task_count"]) != expected:
        raise ValueError("expected task count changed")
    return study


@dataclass(frozen=True)
class ProfitTimeoutSpec:
    slot_count: int
    timeout_day: int
    take_profit: float | None
    cost_scenario: str

    @property
    def policy_name(self) -> str:
        prefix = (
            "timeout_only"
            if self.take_profit is None
            else f"tp{round(self.take_profit * 100):02d}"
        )
        return f"{prefix}_d{self.timeout_day:02d}"

    @property
    def task_id(self) -> str:
        return (
            f"{FAMILY}__{self.policy_name}__k{self.slot_count:02d}"
            f"__{self.cost_scenario}"
        )

    def validate(self) -> None:
        if self.slot_count not in SLOT_COUNTS:
            raise ValueError(f"invalid slot count: {self.slot_count}")
        if self.timeout_day not in TIMEOUT_DAYS:
            raise ValueError(f"invalid timeout day: {self.timeout_day}")
        if self.take_profit not in TAKE_PROFIT_LEVELS:
            raise ValueError(f"invalid take profit: {self.take_profit}")
        if self.cost_scenario not in COST_SCENARIOS:
            raise ValueError(f"invalid cost scenario: {self.cost_scenario}")


@dataclass(frozen=True)
class PendingEntry:
    signal_day: int
    symbol_idx: int


@dataclass
class PendingExit:
    reason: str
    trigger_date_idx: int
    trigger_price: float
    blocked_days: int = 1


@dataclass(frozen=True)
class ExitDecision:
    reason: str
    trigger_date_idx: int
    trigger_price: float
    fill_price: float | None
    fill_phase: str


def task_specs(
    study_path: Path = DEFAULT_STUDY_PATH,
) -> list[ProfitTimeoutSpec]:
    load_study(study_path)
    specs = [
        ProfitTimeoutSpec(
            slot_count=slots,
            timeout_day=timeout,
            take_profit=take_profit,
            cost_scenario=cost,
        )
        for timeout in TIMEOUT_DAYS
        for take_profit in TAKE_PROFIT_LEVELS
        for slots in SLOT_COUNTS
        for cost in COST_SCENARIOS
    ]
    for spec in specs:
        spec.validate()
    if len(specs) != 32 or len({spec.task_id for spec in specs}) != len(specs):
        raise AssertionError("profit-timeout task inventory changed")
    return specs


def resolve_exit_decision(
    *,
    position: economic.Position,
    spec: ProfitTimeoutSpec,
    date_idx: int,
    adjusted_ohlc: np.ndarray,
    sellable: bool,
) -> ExitDecision | None:
    """Resolve a precommitted D2+ take-profit or vertical timeout decision."""

    if int(date_idx) <= int(position.entry_date_idx):
        return None
    values = np.asarray(adjusted_ohlc, dtype=np.float64)
    if values.shape != (4,):
        raise ValueError("adjusted_ohlc must contain open/high/low/close")
    open_price, high_price, _, close_price = values
    if spec.take_profit is not None:
        target = float(position.entry_adjusted_open) * (1.0 + spec.take_profit)
        if math.isfinite(open_price) and open_price >= target:
            return ExitDecision(
                reason="take_profit",
                trigger_date_idx=int(date_idx),
                trigger_price=float(open_price),
                fill_price=float(open_price) if sellable else None,
                fill_phase="gap_open" if sellable else "blocked",
            )
        if math.isfinite(high_price) and high_price >= target:
            return ExitDecision(
                reason="take_profit",
                trigger_date_idx=int(date_idx),
                trigger_price=float(target),
                fill_price=float(target) if sellable else None,
                fill_phase="standing_limit" if sellable else "blocked",
            )
    timeout_idx = int(position.entry_signal_date_idx) + int(spec.timeout_day)
    if int(date_idx) >= timeout_idx:
        fill_price = (
            float(close_price)
            if sellable and math.isfinite(close_price) and close_price > 0.0
            else None
        )
        return ExitDecision(
            reason="timeout",
            trigger_date_idx=int(date_idx),
            trigger_price=float(close_price),
            fill_price=fill_price,
            fill_phase="precommitted_close" if fill_price is not None else "blocked",
        )
    return None


def entry_candidates(
    *,
    book: base.SignalBook,
    day: int,
    held_symbols: set[int],
    limit: int,
) -> list[int]:
    """Select the dual-MFE Top-5% pool after the frozen risk veto."""

    if int(limit) <= 0:
        return []
    ordered = book.symbols_for_day(FAMILY, int(day))
    if len(ordered) == 0:
        return []
    top_fraction = float(book.study["policies"]["auxiliary_top_fraction"])
    retention = float(book.study["policies"]["auxiliary_retention_fraction"])
    pool_count = max(1, math.ceil(len(ordered) * top_fraction))
    pool = np.asarray(ordered[:pool_count], dtype=np.int32)
    safety = np.minimum(
        np.asarray(book.rank_panel[int(day), pool, 5], dtype=np.float64),
        np.asarray(book.rank_panel[int(day), pool, 6], dtype=np.float64),
    )
    if not bool(np.isfinite(safety).all()):
        raise ValueError("risk-veto rank contains non-finite values")
    retain_count = max(1, math.ceil(len(pool) * retention))
    risk_order = np.lexsort((pool, -safety))
    gate = {int(value) for value in pool[risk_order[:retain_count]]}
    output: list[int] = []
    for value in pool:
        symbol_idx = int(value)
        if symbol_idx in held_symbols or symbol_idx not in gate:
            continue
        output.append(symbol_idx)
        if len(output) == int(limit):
            break
    return output


def _trade_row(
    *,
    task_id: str,
    side: str,
    status: str,
    signal_date: str,
    execution_date: str,
    symbol_idx: int,
    symbol: str,
    reason: str,
    entry_signal_date_idx: int | None = None,
    entry_date_idx: int | None = None,
    exit_trigger_date_idx: int | None = None,
    execution_date_idx: int | None = None,
    fill_phase: str = "",
    details: Mapping[str, float] | None = None,
    holding_days: int | None = None,
    sell_delay_days: int | None = None,
    gross_return: float = math.nan,
    net_return: float = math.nan,
    participation: float = math.nan,
) -> dict[str, Any]:
    row = economic._attempt_row(
        task_id=task_id,
        side=side,
        status=status,
        signal_date=signal_date,
        execution_date=execution_date,
        symbol_idx=symbol_idx,
        symbol=symbol,
        reason=reason,
        details=details,
        holding_days=holding_days,
        sell_delay_days=sell_delay_days,
        participation=participation,
    )
    row.update(
        {
            "entry_signal_date_idx": (
                int(entry_signal_date_idx)
                if entry_signal_date_idx is not None
                else pd.NA
            ),
            "entry_date_idx": int(entry_date_idx)
            if entry_date_idx is not None
            else pd.NA,
            "exit_trigger_date_idx": (
                int(exit_trigger_date_idx)
                if exit_trigger_date_idx is not None
                else pd.NA
            ),
            "execution_date_idx": (
                int(execution_date_idx) if execution_date_idx is not None else pd.NA
            ),
            "fill_phase": str(fill_phase),
            "gross_return": float(gross_return),
            "net_return": float(net_return),
        }
    )
    return row


def simulate_task(
    *,
    book: base.SignalBook,
    spec: ProfitTimeoutSpec,
    study: Mapping[str, Any],
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    spec.validate()
    multiplier = float(
        dict(book.study["execution"]["cost_scenarios"])[spec.cost_scenario]
    )
    cash = float(study["account"]["starting_cash_cny"])
    gross_cash = cash
    positions: dict[int, economic.Position] = {}
    pending_entries: list[PendingEntry] = []
    pending_exits: dict[int, PendingExit] = {}
    total_cost = 0.0
    cost_parts = defaultdict(float)
    traded_notional = 0.0
    trades: list[dict[str, Any]] = []
    equity_rows: list[dict[str, Any]] = []
    holding_days: list[int] = []
    sell_delay_days: list[int] = []
    take_profit_hit_days: list[int] = []
    timeout_gross_returns: list[float] = []
    realized_contribution = defaultdict(float)
    counters = defaultdict(int)
    maximum_conservation_error = 0.0
    benchmark_wealth = STARTING_CASH_CNY
    previous_equity = STARTING_CASH_CNY
    previous_gross_equity = STARTING_CASH_CNY

    def absorb_costs(details: Mapping[str, float]) -> None:
        nonlocal total_cost, traded_notional
        total_cost += float(details["total_cost"])
        traded_notional += float(details["fill_notional"])
        for name in ("commission", "transfer_fee", "stamp_tax", "slippage"):
            cost_parts[name] += float(details[name])

    def fill_sell(
        *,
        symbol_idx: int,
        adjusted_price: float,
        date_idx: int,
        trade_date: str,
        reason: str,
        trigger_date_idx: int,
        fill_phase: str,
        delay_days: int,
    ) -> None:
        nonlocal cash, gross_cash
        position = positions[int(symbol_idx)]
        gross_entry = float(position.gross_entry_notional)
        net_entry = float(position.net_cash_outflow)
        proceeds, details = economic._sell_position(
            position=position,
            adjusted_open=float(adjusted_price),
            trade_date=trade_date,
            costs=book.costs,
            slippage_multiplier=multiplier,
        )
        cash += float(proceeds)
        gross_cash += float(details["gross_notional"])
        absorb_costs(details)
        duration = int(date_idx - position.entry_date_idx)
        holding_days.append(duration)
        if delay_days > 0:
            sell_delay_days.append(int(delay_days))
        gross_return = float(details["gross_notional"] / gross_entry - 1.0)
        net_return = float(proceeds / net_entry - 1.0)
        realized_contribution[int(symbol_idx)] += float(proceeds - net_entry)
        if reason == "take_profit":
            counters["take_profit_filled_count"] += 1
            take_profit_hit_days.append(
                int(trigger_date_idx - position.entry_signal_date_idx)
            )
        elif reason == "timeout":
            counters["timeout_filled_count"] += 1
            timeout_gross_returns.append(gross_return)
        counters["sell_count"] += 1
        trailing = book.trailing_amount(
            signal_date_idx=int(position.entry_signal_date_idx),
            symbol_idx=int(symbol_idx),
        )
        participation = (
            float(details["fill_notional"]) / trailing
            if math.isfinite(trailing) and trailing > 0.0
            else math.nan
        )
        trades.append(
            _trade_row(
                task_id=spec.task_id,
                side="sell",
                status="filled",
                signal_date=str(book.date_values[int(trigger_date_idx)]),
                execution_date=trade_date,
                symbol_idx=int(symbol_idx),
                symbol=str(book.symbol_values[int(symbol_idx)]),
                reason=reason,
                entry_signal_date_idx=int(position.entry_signal_date_idx),
                entry_date_idx=int(position.entry_date_idx),
                exit_trigger_date_idx=int(trigger_date_idx),
                execution_date_idx=int(date_idx),
                fill_phase=fill_phase,
                details=details,
                holding_days=duration,
                sell_delay_days=delay_days,
                gross_return=gross_return,
                net_return=net_return,
                participation=participation,
            )
        )
        positions.pop(int(symbol_idx))
        pending_exits.pop(int(symbol_idx), None)

    for day in range(book.day_count):
        date_idx = int(book.signal_date_idx[day])
        trade_date = book.date_text(day)
        if trade_date.startswith("2026"):
            raise ValueError("profit-timeout simulation attempted to read 2026")
        adjusted_open_row = np.asarray(book.adjusted_open[date_idx], dtype=np.float64)

        for symbol_idx, pending in list(pending_exits.items()):
            position = positions.get(int(symbol_idx))
            if position is None:
                pending_exits.pop(int(symbol_idx), None)
                continue
            open_sellable = day > 0 and bool(book.next_sellable[day - 1, symbol_idx])
            adjusted_open = float(adjusted_open_row[int(symbol_idx)])
            if (
                open_sellable
                and date_idx > int(position.entry_date_idx)
                and math.isfinite(adjusted_open)
                and adjusted_open > 0.0
            ):
                fill_sell(
                    symbol_idx=int(symbol_idx),
                    adjusted_price=adjusted_open,
                    date_idx=date_idx,
                    trade_date=trade_date,
                    reason=pending.reason,
                    trigger_date_idx=int(pending.trigger_date_idx),
                    fill_phase="delayed_open",
                    delay_days=int(pending.blocked_days),
                )
                continue
            pending.blocked_days += 1
            counters["failed_sell_count"] += 1
            counters["blocked_sell_count"] += 1
            trades.append(
                _trade_row(
                    task_id=spec.task_id,
                    side="sell",
                    status="failed",
                    signal_date=str(book.date_values[pending.trigger_date_idx]),
                    execution_date=trade_date,
                    symbol_idx=int(symbol_idx),
                    symbol=str(book.symbol_values[int(symbol_idx)]),
                    reason=pending.reason,
                    entry_signal_date_idx=int(position.entry_signal_date_idx),
                    entry_date_idx=int(position.entry_date_idx),
                    exit_trigger_date_idx=int(pending.trigger_date_idx),
                    execution_date_idx=date_idx,
                    fill_phase="blocked_open",
                )
            )

        equity_open = economic._portfolio_value(
            cash=cash,
            positions=positions,
            adjusted_prices=adjusted_open_row,
        )
        for order in pending_entries:
            if int(order.signal_day) != day - 1:
                raise AssertionError("pending entry skipped a signal day")
            symbol_idx = int(order.symbol_idx)
            signal_idx = int(book.signal_date_idx[order.signal_day])
            signal_date = book.date_text(order.signal_day)
            symbol = str(book.symbol_values[symbol_idx])
            if len(positions) >= spec.slot_count or symbol_idx in positions:
                counters["skipped_buy_slot_count"] += 1
                trades.append(
                    _trade_row(
                        task_id=spec.task_id,
                        side="buy",
                        status="skipped_slot",
                        signal_date=signal_date,
                        execution_date=trade_date,
                        symbol_idx=symbol_idx,
                        symbol=symbol,
                        reason="new_slot",
                        entry_signal_date_idx=signal_idx,
                        execution_date_idx=date_idx,
                    )
                )
                continue
            if not bool(book.next_buyable[order.signal_day, symbol_idx]):
                counters["failed_buy_count"] += 1
                trades.append(
                    _trade_row(
                        task_id=spec.task_id,
                        side="buy",
                        status="failed",
                        signal_date=signal_date,
                        execution_date=trade_date,
                        symbol_idx=symbol_idx,
                        symbol=symbol,
                        reason="new_slot",
                        entry_signal_date_idx=signal_idx,
                        execution_date_idx=date_idx,
                    )
                )
                continue
            allocation_spec = economic.TaskSpec(
                family=FAMILY,
                exposure_mode="target_full",
                slot_count=spec.slot_count,
                buffer_multiplier=0.0,
                cost_scenario=spec.cost_scenario,
            )
            allocation = economic._position_allocation(
                book=book,
                spec=allocation_spec,
                cash=cash,
                equity_open=equity_open,
            )
            position, details = economic._buy_position(
                available_cash=cash,
                allocated_cash=allocation,
                symbol_idx=symbol_idx,
                signal_date_idx=signal_idx,
                execution_date_idx=date_idx,
                raw_open=float(book.raw_open[date_idx, symbol_idx]),
                adjusted_open=float(adjusted_open_row[symbol_idx]),
                costs=book.costs,
                slippage_multiplier=multiplier,
            )
            if position is None:
                counters["failed_buy_cash_or_lot_count"] += 1
                trades.append(
                    _trade_row(
                        task_id=spec.task_id,
                        side="buy",
                        status="failed_cash_or_lot",
                        signal_date=signal_date,
                        execution_date=trade_date,
                        symbol_idx=symbol_idx,
                        symbol=symbol,
                        reason="new_slot",
                        entry_signal_date_idx=signal_idx,
                        execution_date_idx=date_idx,
                    )
                )
                continue
            cash += float(details["cash_flow"])
            gross_cash -= float(details["gross_notional"])
            absorb_costs(details)
            positions[symbol_idx] = position
            counters["buy_count"] += 1
            trailing = book.trailing_amount(
                signal_date_idx=signal_idx,
                symbol_idx=symbol_idx,
            )
            participation = (
                float(details["fill_notional"]) / trailing
                if math.isfinite(trailing) and trailing > 0.0
                else math.nan
            )
            trades.append(
                _trade_row(
                    task_id=spec.task_id,
                    side="buy",
                    status="filled",
                    signal_date=signal_date,
                    execution_date=trade_date,
                    symbol_idx=symbol_idx,
                    symbol=symbol,
                    reason="new_slot",
                    entry_signal_date_idx=signal_idx,
                    entry_date_idx=date_idx,
                    execution_date_idx=date_idx,
                    fill_phase="next_open",
                    details=details,
                    participation=participation,
                )
            )
        pending_entries = []

        for symbol_idx in list(positions):
            if symbol_idx in pending_exits:
                continue
            position = positions[symbol_idx]
            decision = resolve_exit_decision(
                position=position,
                spec=spec,
                date_idx=date_idx,
                adjusted_ohlc=np.asarray(
                    book.daily_raw[date_idx, symbol_idx, :4], dtype=np.float64
                ),
                sellable=bool(book.exit_sellable[date_idx, symbol_idx]),
            )
            if decision is None:
                continue
            counters[f"{decision.reason}_trigger_count"] += 1
            if decision.fill_price is not None:
                fill_sell(
                    symbol_idx=symbol_idx,
                    adjusted_price=float(decision.fill_price),
                    date_idx=date_idx,
                    trade_date=trade_date,
                    reason=decision.reason,
                    trigger_date_idx=decision.trigger_date_idx,
                    fill_phase=decision.fill_phase,
                    delay_days=0,
                )
                continue
            pending_exits[symbol_idx] = PendingExit(
                reason=decision.reason,
                trigger_date_idx=decision.trigger_date_idx,
                trigger_price=decision.trigger_price,
            )
            counters["failed_sell_count"] += 1
            counters["blocked_sell_count"] += 1
            trades.append(
                _trade_row(
                    task_id=spec.task_id,
                    side="sell",
                    status="failed",
                    signal_date=trade_date,
                    execution_date=trade_date,
                    symbol_idx=symbol_idx,
                    symbol=str(book.symbol_values[symbol_idx]),
                    reason=decision.reason,
                    entry_signal_date_idx=int(position.entry_signal_date_idx),
                    entry_date_idx=int(position.entry_date_idx),
                    exit_trigger_date_idx=decision.trigger_date_idx,
                    execution_date_idx=date_idx,
                    fill_phase="blocked_trigger",
                )
            )

        adjusted_close_row = np.asarray(book.adjusted_close[date_idx], dtype=np.float64)
        if day == book.day_count - 1:
            terminal_symbols = [
                int(symbol_idx)
                for symbol_idx in positions
                if bool(book.is_delisted[date_idx, int(symbol_idx)])
                or (
                    not bool(book.status_valid[date_idx, int(symbol_idx)])
                    and not bool(book.has_bar[date_idx, int(symbol_idx)])
                )
            ]
            for symbol_idx in terminal_symbols:
                position = positions.pop(symbol_idx)
                pending_exits.pop(symbol_idx, None)
                recovery = float(
                    position.gross_entry_notional * book.terminal_recovery_fraction
                )
                cash += recovery
                gross_cash += recovery
                duration = int(date_idx - position.entry_date_idx)
                holding_days.append(duration)
                realized_contribution[symbol_idx] += float(
                    recovery - position.net_cash_outflow
                )
                counters["terminal_recovery_count"] += 1
                counters["terminal_recovery_notional"] += recovery
                trades.append(
                    _trade_row(
                        task_id=spec.task_id,
                        side="terminal",
                        status="recovery",
                        signal_date=trade_date,
                        execution_date=trade_date,
                        symbol_idx=symbol_idx,
                        symbol=str(book.symbol_values[symbol_idx]),
                        reason="pack_terminal_lifecycle_recovery",
                        entry_signal_date_idx=int(position.entry_signal_date_idx),
                        entry_date_idx=int(position.entry_date_idx),
                        execution_date_idx=date_idx,
                        details={
                            "gross_notional": recovery,
                            "fill_notional": recovery,
                            "cash_flow": recovery,
                            "total_cost": 0.0,
                        },
                        holding_days=duration,
                    )
                )

        equity = economic._portfolio_value(
            cash=cash,
            positions=positions,
            adjusted_prices=adjusted_close_row,
        )
        gross_equity = economic._portfolio_value(
            cash=gross_cash,
            positions=positions,
            adjusted_prices=adjusted_close_row,
        )
        conservation_error = abs((gross_equity - equity) - total_cost)
        maximum_conservation_error = max(maximum_conservation_error, conservation_error)
        tolerance = max(1.0e-5, abs(gross_equity) * 1.0e-10)
        if conservation_error > tolerance:
            raise AssertionError(
                f"cashflow conservation failed by {conservation_error:.8f}"
            )
        daily_return = float(equity / previous_equity - 1.0)
        gross_daily_return = float(gross_equity / previous_gross_equity - 1.0)
        benchmark_return = float(book.benchmark_returns[day])
        benchmark_wealth *= 1.0 + benchmark_return
        excess_return = (
            float((1.0 + daily_return) / (1.0 + benchmark_return) - 1.0)
            if benchmark_return > -1.0
            else math.nan
        )
        equity_rows.append(
            {
                "task_id": spec.task_id,
                "trade_date": trade_date,
                "date_idx": date_idx,
                "cash": float(cash),
                "gross_cash": float(gross_cash),
                "position_count": len(positions),
                "pending_exit_count": len(pending_exits),
                "net_equity": float(equity),
                "gross_equity": float(gross_equity),
                "daily_net_return": daily_return,
                "daily_gross_return": gross_daily_return,
                "benchmark_return": benchmark_return,
                "daily_relative_excess_return": excess_return,
                "benchmark_wealth": float(benchmark_wealth),
                "cash_fraction": float(cash / equity) if equity > 0.0 else math.nan,
                "cumulative_cost": float(total_cost),
                "conservation_error": float(conservation_error),
            }
        )
        previous_equity = float(equity)
        previous_gross_equity = float(gross_equity)

        if day < book.day_count - 1 and book.trading_allowed(day):
            free_slots = max(spec.slot_count - len(positions), 0)
            selected = entry_candidates(
                book=book,
                day=day,
                held_symbols=set(positions),
                limit=free_slots,
            )
            pending_entries = [
                PendingEntry(signal_day=day, symbol_idx=symbol_idx)
                for symbol_idx in selected
            ]

    equity_frame = pd.DataFrame(equity_rows)
    equity_frame["rolling_63d_net_return"] = (
        1.0 + equity_frame["daily_net_return"].astype(float)
    ).rolling(63, min_periods=63).apply(np.prod, raw=True) - 1.0
    equity_frame["rolling_63d_relative_excess_return"] = (
        1.0 + equity_frame["daily_relative_excess_return"].astype(float)
    ).rolling(63, min_periods=63).apply(np.prod, raw=True) - 1.0
    trades_frame = pd.DataFrame(trades)
    if trades_frame.empty:
        trades_frame = pd.DataFrame(
            [
                _trade_row(
                    task_id=spec.task_id,
                    side="none",
                    status="none",
                    signal_date="",
                    execution_date="",
                    symbol_idx=0,
                    symbol="",
                    reason="none",
                )
            ]
        ).iloc[:0]
    monthly = economic._monthly_metrics(
        dates=equity_frame["trade_date"].to_numpy(dtype=str),
        equity=equity_frame["net_equity"].to_numpy(dtype=np.float64),
    )
    net_metrics = economic._annualized_metrics(
        equity=np.r_[STARTING_CASH_CNY, equity_frame["net_equity"].to_numpy()],
        daily_returns=np.r_[0.0, equity_frame["daily_net_return"].to_numpy()],
    )
    gross_metrics = economic._annualized_metrics(
        equity=np.r_[STARTING_CASH_CNY, equity_frame["gross_equity"].to_numpy()],
        daily_returns=np.r_[0.0, equity_frame["daily_gross_return"].to_numpy()],
    )
    annual = economic._year_metrics(
        dates=equity_frame["trade_date"].to_numpy(dtype=str),
        equity=equity_frame["net_equity"].to_numpy(dtype=np.float64),
        benchmark_wealth=equity_frame["benchmark_wealth"].to_numpy(dtype=np.float64),
    )
    final_date = str(equity_frame["trade_date"].iloc[-1])
    final_date_idx = int(equity_frame["date_idx"].iloc[-1])
    terminal_equity = float(cash)
    terminal_cost = 0.0
    contribution = defaultdict(float, realized_contribution)
    for symbol_idx, position in positions.items():
        adjusted_close = float(book.adjusted_close[final_date_idx, symbol_idx])
        proceeds, details = economic._sell_position(
            position=position,
            adjusted_open=adjusted_close,
            trade_date=final_date,
            costs=book.costs,
            slippage_multiplier=multiplier,
        )
        terminal_equity += float(proceeds)
        terminal_cost += float(details["total_cost"])
        contribution[symbol_idx] += float(proceeds - position.net_cash_outflow)
    absolute_contribution = np.asarray(
        sorted((abs(value) for value in contribution.values()), reverse=True),
        dtype=np.float64,
    )
    contribution_total = float(np.sum(absolute_contribution))
    top5_concentration = (
        float(np.sum(absolute_contribution[:5]) / contribution_total)
        if contribution_total > 0.0
        else 0.0
    )
    filled = trades_frame[trades_frame["status"].astype(str).eq("filled")]
    participation = pd.to_numeric(
        filled["participation_of_trailing20_median_amount"], errors="coerce"
    ).to_numpy(dtype=np.float64)
    participation = participation[np.isfinite(participation)]
    thresholds = tuple(
        float(value) for value in study["execution"]["amount_participation_thresholds"]
    )
    benchmark_end = float(equity_frame["benchmark_wealth"].iloc[-1])
    terminal_return = float(terminal_equity / STARTING_CASH_CNY - 1.0)
    terminal_excess = float(
        (terminal_equity / STARTING_CASH_CNY)
        / max(benchmark_end / STARTING_CASH_CNY, 1.0e-12)
        - 1.0
    )
    completed_exits = int(
        counters["take_profit_filled_count"] + counters["timeout_filled_count"]
    )
    result = {
        "schema": TASK_SCHEMA,
        "status": "completed",
        "completed_at": base._now(),
        "study_id": STUDY_ID,
        "task": asdict(spec),
        "policy_name": spec.policy_name,
        "task_id": spec.task_id,
        "signal_book_sha256": book.manifest["files"]["rank_panel"]["sha256"],
        "period": {
            "first_signal_date": str(equity_frame["trade_date"].iloc[0]),
            "last_entry_signal_date": str(study["period"]["last_entry_signal_date"]),
            "last_mark_date": final_date,
            "continuous_years": list(YEARS),
            "annual_reset": False,
            "forbidden_2026_row_count": 0,
            "research_semantics": str(study["period"]["research_semantics"]),
        },
        "metrics": {
            "net": net_metrics,
            "gross_same_trade_sequence": gross_metrics,
            "terminal_cost_accrued_equity": float(terminal_equity),
            "terminal_cost_accrued_return": terminal_return,
            "terminal_cost_accrued_relative_excess_return": terminal_excess,
            "terminal_accrued_exit_cost": float(terminal_cost),
            "benchmark_ending_wealth": benchmark_end,
            "turnover_to_starting_cash": float(traded_notional / STARTING_CASH_CNY),
            "average_cash_fraction": float(equity_frame["cash_fraction"].mean()),
            "maximum_cash_fraction": float(equity_frame["cash_fraction"].max()),
            "average_position_count": float(equity_frame["position_count"].mean()),
            "filled_buy_count": int(counters["buy_count"]),
            "filled_sell_count": int(counters["sell_count"]),
            "failed_buy_count": int(
                counters["failed_buy_count"] + counters["failed_buy_cash_or_lot_count"]
            ),
            "failed_sell_count": int(counters["failed_sell_count"]),
            "blocked_sell_count": int(counters["blocked_sell_count"]),
            "delayed_sell_count": len(sell_delay_days),
            "sell_delay_days_mean": (
                float(np.mean(sell_delay_days)) if sell_delay_days else 0.0
            ),
            "unresolved_blocked_sell_count": len(pending_exits),
            "take_profit_trigger_count": int(counters["take_profit_trigger_count"]),
            "take_profit_filled_count": int(counters["take_profit_filled_count"]),
            "timeout_trigger_count": int(counters["timeout_trigger_count"]),
            "timeout_filled_count": int(counters["timeout_filled_count"]),
            "take_profit_hit_rate": (
                float(counters["take_profit_filled_count"] / completed_exits)
                if completed_exits
                else 0.0
            ),
            "take_profit_hit_day_median": (
                float(np.median(take_profit_hit_days))
                if take_profit_hit_days
                else math.nan
            ),
            "timeout_gross_return_mean": (
                float(np.mean(timeout_gross_returns))
                if timeout_gross_returns
                else math.nan
            ),
            "holding_days_mean": (
                float(np.mean(holding_days)) if holding_days else math.nan
            ),
            "holding_days_median": (
                float(np.median(holding_days)) if holding_days else math.nan
            ),
            "holding_days_p90": (
                float(np.quantile(holding_days, 0.9)) if holding_days else math.nan
            ),
            "terminal_recovery_count": int(counters["terminal_recovery_count"]),
            "terminal_recovery_notional": float(counters["terminal_recovery_notional"]),
            "top5_absolute_pnl_contribution_fraction": top5_concentration,
            "maximum_conservation_error": float(maximum_conservation_error),
            "participation": {
                "finite_order_count": len(participation),
                **{
                    f"fraction_above_{threshold:g}": (
                        float(np.mean(participation > threshold))
                        if len(participation)
                        else math.nan
                    )
                    for threshold in thresholds
                },
            },
            "costs": {
                "total": float(total_cost),
                **{name: float(cost_parts[name]) for name in cost_parts},
            },
        },
        "annual": annual,
    }
    return result, equity_frame, trades_frame, monthly


def _task_dir(output_root: Path, spec: ProfitTimeoutSpec) -> Path:
    return output_root / "tasks" / spec.task_id


def _task_complete(
    *,
    output_root: Path,
    spec: ProfitTimeoutSpec,
    study_sha256: str,
    signal_sha256: str,
) -> bool:
    path = _task_dir(output_root, spec) / "task_result.json"
    if not path.is_file():
        return False
    try:
        result = _load_json(path)
        if (
            result.get("schema") != TASK_SCHEMA
            or result.get("status") != "completed"
            or result.get("study_id") != STUDY_ID
            or result.get("task_id") != spec.task_id
            or result.get("study_config_sha256") != study_sha256
            or result.get("signal_book_sha256") != signal_sha256
            or dict(result.get("task", {})) != asdict(spec)
        ):
            return False
        for record in dict(result["files"]).values():
            economic._verify_record(record)
    except (FileNotFoundError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return False
    return True


def _write_task(
    *,
    output_root: Path,
    result: dict[str, Any],
    equity: pd.DataFrame,
    trades: pd.DataFrame,
    monthly: pd.DataFrame,
    study_sha256: str,
) -> None:
    spec = ProfitTimeoutSpec(**dict(result["task"]))
    directory = _task_dir(output_root, spec)
    equity_path = directory / "equity.parquet"
    trades_path = directory / "trades.parquet"
    monthly_path = directory / "monthly.parquet"
    _write_parquet(equity_path, equity)
    _write_parquet(trades_path, trades)
    _write_parquet(monthly_path, monthly)
    result["study_config_sha256"] = study_sha256
    result["files"] = {
        "equity": _file_record(equity_path, row_count=len(equity)),
        "trades": _file_record(trades_path, row_count=len(trades)),
        "monthly": _file_record(monthly_path, row_count=len(monthly)),
    }
    _write_json(directory / "task_result.json", result)


def _source_material(
    study: Mapping[str, Any],
) -> tuple[dict[str, Any], Path, dict[str, Any]]:
    source = dict(study["sources"])
    base_study_path = _resolve(source["execution_study"])
    base_output = _resolve(source["execution_output"])
    base_study = base.load_study(base_study_path)
    source_audit = _load_json(base_output / "audit.json")
    if source_audit.get("status") != "ok":
        raise ValueError("source execution study is not audited")
    if not base._signal_complete(
        base_output, study_sha256=base._study_hash(base_study_path)
    ):
        raise ValueError("source execution signal book is incomplete")
    signal_manifest = _load_json(base_output / "signal_book/manifest.json")
    if (
        int(signal_manifest.get("feature_count", -1)) != 557
        or int(signal_manifest.get("forbidden_2026_row_count", -1)) != 0
    ):
        raise ValueError("source signal book contract changed")
    return base_study, base_output, signal_manifest


def status(
    *,
    study_path: Path = DEFAULT_STUDY_PATH,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
) -> dict[str, Any]:
    study = load_study(study_path)
    _, _, signal = _source_material(study)
    study_sha256 = _study_hash(study_path)
    signal_sha256 = str(signal["files"]["rank_panel"]["sha256"])
    specs = task_specs(study_path)
    completed = sum(
        _task_complete(
            output_root=output_root,
            spec=spec,
            study_sha256=study_sha256,
            signal_sha256=signal_sha256,
        )
        for spec in specs
    )
    return {
        "status": "completed" if completed == len(specs) else "in_progress",
        "completed": int(completed),
        "total": len(specs),
    }


def run(
    *,
    study_path: Path = DEFAULT_STUDY_PATH,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    max_tasks: int | None = None,
) -> dict[str, Any]:
    study = load_study(study_path)
    base_study, base_output, signal = _source_material(study)
    study_sha256 = _study_hash(study_path)
    signal_sha256 = str(signal["files"]["rank_panel"]["sha256"])
    specs = task_specs(study_path)
    book = base.SignalBook(study=base_study, output_root=base_output)
    completed = 0
    executed = 0
    started = time.monotonic()
    for spec in specs:
        if _task_complete(
            output_root=output_root,
            spec=spec,
            study_sha256=study_sha256,
            signal_sha256=signal_sha256,
        ):
            completed += 1
            continue
        if max_tasks is not None and executed >= int(max_tasks):
            break
        result, equity, trades, monthly = simulate_task(
            book=book, spec=spec, study=study
        )
        _write_task(
            output_root=output_root,
            result=result,
            equity=equity,
            trades=trades,
            monthly=monthly,
            study_sha256=study_sha256,
        )
        completed += 1
        executed += 1
        if completed % 4 == 0 or completed == len(specs):
            _emit(
                "profit_timeout_progress",
                completed=completed,
                total=len(specs),
                task_id=spec.task_id,
                elapsed_seconds=round(time.monotonic() - started, 2),
            )
    return {
        "status": "completed" if completed == len(specs) else "in_progress",
        "completed": int(completed),
        "total": len(specs),
        "executed_this_run": int(executed),
    }


def _task_result(output_root: Path, spec: ProfitTimeoutSpec) -> dict[str, Any]:
    return _load_json(_task_dir(output_root, spec) / "task_result.json")


def _optional_float(value: Any) -> float:
    return math.nan if value is None else float(value)


def _metric_row(result: Mapping[str, Any]) -> dict[str, Any]:
    task = dict(result["task"])
    metrics = dict(result["metrics"])
    net = dict(metrics["net"])
    gross = dict(metrics["gross_same_trade_sequence"])
    return {
        **task,
        "policy_name": str(result["policy_name"]),
        "task_id": str(result["task_id"]),
        "net_cumulative_return": float(net["cumulative_return"]),
        "gross_cumulative_return": float(gross["cumulative_return"]),
        "terminal_return": float(metrics["terminal_cost_accrued_return"]),
        "terminal_excess_return": float(
            metrics["terminal_cost_accrued_relative_excess_return"]
        ),
        "maximum_drawdown": float(net["maximum_drawdown"]),
        "sharpe": float(net["sharpe"]),
        "turnover": float(metrics["turnover_to_starting_cash"]),
        "average_cash_fraction": float(metrics["average_cash_fraction"]),
        "average_position_count": float(metrics["average_position_count"]),
        "holding_days_mean": float(metrics["holding_days_mean"]),
        "take_profit_hit_rate": float(metrics["take_profit_hit_rate"]),
        "take_profit_hit_day_median": _optional_float(
            metrics["take_profit_hit_day_median"]
        ),
        "timeout_gross_return_mean": _optional_float(
            metrics["timeout_gross_return_mean"]
        ),
        "filled_buy_count": int(metrics["filled_buy_count"]),
        "filled_sell_count": int(metrics["filled_sell_count"]),
        "blocked_sell_count": int(metrics["blocked_sell_count"]),
        "maximum_conservation_error": float(metrics["maximum_conservation_error"]),
    }


def evaluate(
    *,
    study_path: Path = DEFAULT_STUDY_PATH,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
) -> dict[str, Any]:
    study = load_study(study_path)
    _, _, signal = _source_material(study)
    study_sha256 = _study_hash(study_path)
    signal_sha256 = str(signal["files"]["rank_panel"]["sha256"])
    specs = task_specs(study_path)
    if not all(
        _task_complete(
            output_root=output_root,
            spec=spec,
            study_sha256=study_sha256,
            signal_sha256=signal_sha256,
        )
        for spec in specs
    ):
        raise RuntimeError("profit-timeout tasks are incomplete")
    results = [_task_result(output_root, spec) for spec in specs]
    metrics = pd.DataFrame([_metric_row(result) for result in results])
    annual_rows: list[dict[str, Any]] = []
    for result in results:
        task = dict(result["task"])
        for row in result["annual"]:
            annual_rows.append(
                {
                    **task,
                    "policy_name": str(result["policy_name"]),
                    "task_id": str(result["task_id"]),
                    **dict(row),
                }
            )
    annual = pd.DataFrame(annual_rows)

    paired_rows: list[dict[str, Any]] = []
    for row in metrics.itertuples(index=False):
        if pd.isna(row.take_profit):
            continue
        baseline = metrics[
            metrics["take_profit"].isna()
            & metrics["timeout_day"].eq(int(row.timeout_day))
            & metrics["slot_count"].eq(int(row.slot_count))
            & metrics["cost_scenario"].eq(str(row.cost_scenario))
        ].iloc[0]
        paired_rows.append(
            {
                "policy_name": str(row.policy_name),
                "timeout_day": int(row.timeout_day),
                "take_profit": float(row.take_profit),
                "slot_count": int(row.slot_count),
                "cost_scenario": str(row.cost_scenario),
                "terminal_return_delta": float(
                    row.terminal_return - baseline["terminal_return"]
                ),
                "terminal_excess_delta": float(
                    row.terminal_excess_return - baseline["terminal_excess_return"]
                ),
                "maximum_drawdown_delta": float(
                    row.maximum_drawdown - baseline["maximum_drawdown"]
                ),
                "turnover_delta": float(row.turnover - baseline["turnover"]),
            }
        )
    paired = pd.DataFrame(paired_rows)
    annual_paired_rows: list[dict[str, Any]] = []
    for row in annual.itertuples(index=False):
        if pd.isna(row.take_profit):
            continue
        baseline = annual[
            annual["take_profit"].isna()
            & annual["timeout_day"].eq(int(row.timeout_day))
            & annual["slot_count"].eq(int(row.slot_count))
            & annual["cost_scenario"].eq(str(row.cost_scenario))
            & annual["year"].eq(int(row.year))
        ].iloc[0]
        annual_paired_rows.append(
            {
                "policy_name": str(row.policy_name),
                "timeout_day": int(row.timeout_day),
                "take_profit": float(row.take_profit),
                "slot_count": int(row.slot_count),
                "cost_scenario": str(row.cost_scenario),
                "year": int(row.year),
                "net_return_delta": float(row.net_return - baseline["net_return"]),
                "relative_excess_delta": float(
                    row.relative_excess_return - baseline["relative_excess_return"]
                ),
            }
        )
    annual_paired = pd.DataFrame(annual_paired_rows)

    selection = dict(study["selection"])
    configuration_rows: list[dict[str, Any]] = []
    for timeout in TIMEOUT_DAYS:
        for take_profit in TAKE_PROFIT_LEVELS:
            for slots in SLOT_COUNTS:
                group = metrics[
                    metrics["timeout_day"].eq(timeout)
                    & metrics["slot_count"].eq(slots)
                    & (
                        metrics["take_profit"].isna()
                        if take_profit is None
                        else metrics["take_profit"].eq(take_profit)
                    )
                ]
                by_cost = group.set_index("cost_scenario")
                base_row = by_cost.loc["base"]
                stress_row = by_cost.loc["stress"]
                annual_group = annual[
                    annual["timeout_day"].eq(timeout)
                    & annual["slot_count"].eq(slots)
                    & (
                        annual["take_profit"].isna()
                        if take_profit is None
                        else annual["take_profit"].eq(take_profit)
                    )
                ]
                positive_absolute = {
                    cost: int(
                        annual_group[annual_group["cost_scenario"].eq(cost)][
                            "net_return"
                        ]
                        .gt(0.0)
                        .sum()
                    )
                    for cost in COST_SCENARIOS
                }
                positive_excess = {
                    cost: int(
                        annual_group[annual_group["cost_scenario"].eq(cost)][
                            "relative_excess_return"
                        ]
                        .gt(0.0)
                        .sum()
                    )
                    for cost in COST_SCENARIOS
                }
                paired_base = math.nan
                paired_stress = math.nan
                positive_paired = {cost: 0 for cost in COST_SCENARIOS}
                if take_profit is not None:
                    current_paired = paired[
                        paired["timeout_day"].eq(timeout)
                        & paired["slot_count"].eq(slots)
                        & paired["take_profit"].eq(take_profit)
                    ].set_index("cost_scenario")
                    paired_base = float(
                        current_paired.loc["base", "terminal_return_delta"]
                    )
                    paired_stress = float(
                        current_paired.loc["stress", "terminal_return_delta"]
                    )
                    for cost in COST_SCENARIOS:
                        positive_paired[cost] = int(
                            annual_paired[
                                annual_paired["timeout_day"].eq(timeout)
                                & annual_paired["slot_count"].eq(slots)
                                & annual_paired["take_profit"].eq(take_profit)
                                & annual_paired["cost_scenario"].eq(cost)
                            ]["net_return_delta"]
                            .gt(0.0)
                            .sum()
                        )
                absolute_pass = bool(
                    float(base_row["terminal_return"]) > 0.0
                    and float(stress_row["terminal_return"]) > 0.0
                    and float(base_row["terminal_excess_return"]) > 0.0
                    and float(stress_row["terminal_excess_return"]) > 0.0
                    and positive_absolute["base"]
                    >= int(selection["minimum_positive_absolute_years"])
                    and positive_absolute["stress"]
                    >= int(selection["minimum_positive_absolute_years"])
                    and positive_excess["base"]
                    >= int(selection["minimum_positive_excess_years"])
                    and positive_excess["stress"]
                    >= int(selection["minimum_positive_excess_years"])
                )
                incremental_pass = bool(
                    take_profit is not None
                    and paired_base > 0.0
                    and paired_stress > 0.0
                    and positive_paired["base"]
                    >= int(selection["minimum_positive_paired_years"])
                    and positive_paired["stress"]
                    >= int(selection["minimum_positive_paired_years"])
                )
                configuration_rows.append(
                    {
                        "policy_name": str(base_row["policy_name"]),
                        "timeout_day": timeout,
                        "take_profit": take_profit,
                        "slot_count": slots,
                        "base_terminal_return": float(base_row["terminal_return"]),
                        "stress_terminal_return": float(stress_row["terminal_return"]),
                        "base_terminal_excess": float(
                            base_row["terminal_excess_return"]
                        ),
                        "stress_terminal_excess": float(
                            stress_row["terminal_excess_return"]
                        ),
                        "base_maximum_drawdown": float(base_row["maximum_drawdown"]),
                        "stress_maximum_drawdown": float(
                            stress_row["maximum_drawdown"]
                        ),
                        "base_timeout_delta": paired_base,
                        "stress_timeout_delta": paired_stress,
                        "base_positive_years": positive_absolute["base"],
                        "stress_positive_years": positive_absolute["stress"],
                        "base_positive_excess_years": positive_excess["base"],
                        "stress_positive_excess_years": positive_excess["stress"],
                        "base_positive_paired_years": positive_paired["base"],
                        "stress_positive_paired_years": positive_paired["stress"],
                        "absolute_pass": absolute_pass,
                        "incremental_pass": incremental_pass,
                        "economic_pass": bool(absolute_pass and incremental_pass),
                    }
                )
    configurations = pd.DataFrame(configuration_rows)
    candidate_configs = configurations[configurations["take_profit"].notna()]
    robust_rows: list[dict[str, Any]] = []
    for (timeout, take_profit), group in candidate_configs.groupby(
        ["timeout_day", "take_profit"], sort=True
    ):
        passing_slots = sorted(
            int(value)
            for value in group.loc[group["economic_pass"], "slot_count"].tolist()
        )
        robust_rows.append(
            {
                "timeout_day": int(timeout),
                "take_profit": float(take_profit),
                "passing_slot_counts": json.dumps(passing_slots),
                "robust_region": passing_slots == list(SLOT_COUNTS),
            }
        )
    robust = pd.DataFrame(robust_rows)
    passing_count = int(candidate_configs["economic_pass"].sum())
    robust_count = int(robust["robust_region"].sum())
    ranked = configurations.sort_values(
        ["base_terminal_return", "stress_terminal_return", "base_maximum_drawdown"],
        ascending=[False, False, False],
        kind="stable",
    )
    best = ranked.iloc[0].to_dict()
    overall = (
        "profit_timeout_has_robust_research_candidate"
        if robust_count
        else (
            "profit_timeout_has_isolated_economic_cell"
            if passing_count
            else "profit_timeout_not_monetized"
        )
    )
    decision = {
        "status": "completed_without_production_policy_selection",
        "overall": overall,
        "research_semantics": str(study["period"]["research_semantics"]),
        "economic_passing_cell_count": passing_count,
        "robust_region_count": robust_count,
        "best_descriptive_configuration": economic._json_safe(best),
        "production_policy_selected": False,
        "next_step": (
            "inspect the robust policy before any forward use"
            if robust_count
            else "do not promote the profit-timeout execution policy"
        ),
    }
    evaluation_root = output_root / "evaluation"
    frames = {
        "task_metrics": metrics,
        "annual_metrics": annual,
        "paired_timeout_deltas": paired,
        "annual_paired_timeout_deltas": annual_paired,
        "configurations": configurations,
        "robust_regions": robust,
    }
    files: dict[str, dict[str, Any]] = {}
    for name, frame in frames.items():
        path = evaluation_root / f"{name}.parquet"
        _write_parquet(path, frame)
        files[name] = _file_record(path, row_count=len(frame))
    decision_path = evaluation_root / "decision.json"
    _write_json(decision_path, decision)
    files["decision"] = _file_record(decision_path)
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "status": "evaluated",
        "created_at": base._now(),
        "study_id": STUDY_ID,
        "study_config": _file_record(study_path),
        "study_config_sha256": study_sha256,
        "source_signal_book": _file_record(
            _resolve(study["sources"]["execution_output"]) / "signal_book/manifest.json"
        ),
        "signal_book_rank_sha256": signal_sha256,
        "feature_count": 557,
        "task_count": len(specs),
        "forbidden_2026_row_count": 0,
        "decision": decision,
        "files": files,
    }
    _write_json(output_root / "manifest.json", manifest)
    return decision


def audit(
    *,
    study_path: Path = DEFAULT_STUDY_PATH,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
) -> dict[str, Any]:
    study = load_study(study_path)
    _, base_output, signal = _source_material(study)
    study_sha256 = _study_hash(study_path)
    signal_sha256 = str(signal["files"]["rank_panel"]["sha256"])
    specs = task_specs(study_path)
    tasks_valid = all(
        _task_complete(
            output_root=output_root,
            spec=spec,
            study_sha256=study_sha256,
            signal_sha256=signal_sha256,
        )
        for spec in specs
    )
    manifest_path = output_root / "manifest.json"
    manifest = _load_json(manifest_path) if manifest_path.is_file() else {}
    output_files_valid = True
    try:
        for record in dict(manifest.get("files", {})).values():
            economic._verify_record(record)
    except (FileNotFoundError, KeyError, TypeError, ValueError):
        output_files_valid = False
    dates_valid = bool(tasks_valid)
    t_plus_one_valid = bool(tasks_valid)
    timeout_valid = bool(tasks_valid)
    accounting_valid = bool(tasks_valid)
    if tasks_valid:
        for spec in specs:
            result = _task_result(output_root, spec)
            equity = pd.read_parquet(
                economic._verify_record(result["files"]["equity"]),
                columns=["trade_date", "conservation_error"],
            )
            dates = equity["trade_date"].astype(str)
            dates_valid &= bool(
                len(dates) == int(signal["signal_date_count"])
                and dates.iloc[0] == "2023-01-03"
                and dates.iloc[-1] == "2025-12-31"
                and not dates.str.startswith("2026").any()
            )
            accounting_valid &= bool(
                equity["conservation_error"].astype(float).max()
                <= STARTING_CASH_CNY * 1.0e-9
            )
            trades = pd.read_parquet(economic._verify_record(result["files"]["trades"]))
            if not trades.empty:
                dates_valid &= bool(
                    not trades["signal_date"].astype(str).str.startswith("2026").any()
                    and not trades["execution_date"]
                    .astype(str)
                    .str.startswith("2026")
                    .any()
                )
                filled_sells = trades[
                    trades["side"].astype(str).eq("sell")
                    & trades["status"].astype(str).eq("filled")
                ].copy()
                if not filled_sells.empty:
                    entry_idx = pd.to_numeric(
                        filled_sells["entry_date_idx"], errors="raise"
                    )
                    execution_idx = pd.to_numeric(
                        filled_sells["execution_date_idx"], errors="raise"
                    )
                    t_plus_one_valid &= bool((execution_idx > entry_idx).all())
                    timeouts = filled_sells[
                        filled_sells["reason"].astype(str).eq("timeout")
                    ]
                    if not timeouts.empty:
                        timeout_valid &= bool(
                            (
                                pd.to_numeric(
                                    timeouts["exit_trigger_date_idx"], errors="raise"
                                )
                                >= pd.to_numeric(
                                    timeouts["entry_signal_date_idx"], errors="raise"
                                )
                                + int(spec.timeout_day)
                            ).all()
                        )
    source_audit = _load_json(base_output / "audit.json")
    checks = {
        "manifest_evaluated": manifest.get("status") in {"evaluated", "audited"},
        "source_execution_audited": source_audit.get("status") == "ok",
        "source_557_feature_signal_unchanged": int(signal.get("feature_count", -1))
        == 557,
        "task_count_exact": int(manifest.get("task_count", -1)) == 32,
        "task_files_valid": bool(tasks_valid),
        "evaluation_files_valid": bool(output_files_valid),
        "dates_and_2026_boundary_valid": bool(dates_valid),
        "t_plus_one_valid": bool(t_plus_one_valid),
        "timeout_boundary_valid": bool(timeout_valid),
        "accounting_conservation_valid": bool(accounting_valid),
        "stop_loss_absent": study["policies"].get("stop_loss") is None,
        "no_html_report": not any(output_root.rglob("*.html")),
    }
    payload = {
        "schema": AUDIT_SCHEMA,
        "study_id": STUDY_ID,
        "status": "ok" if all(checks.values()) else "failed",
        "created_at": base._now(),
        "checks": checks,
        "task_count": len(specs),
        "research_semantics": str(study["period"]["research_semantics"]),
    }
    _write_json(output_root / "audit.json", payload)
    if payload["status"] != "ok":
        raise RuntimeError("profit-timeout audit failed")
    manifest["status"] = "audited"
    manifest["audit"] = _file_record(output_root / "audit.json")
    _write_json(manifest_path, manifest)
    return payload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-path", type=Path, default=DEFAULT_STUDY_PATH)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument("--status", action="store_true")
    actions.add_argument("--run", action="store_true")
    actions.add_argument("--evaluate", action="store_true")
    actions.add_argument("--audit", action="store_true")
    parser.add_argument("--max-tasks", type=int, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.status:
        result = status(study_path=args.study_path, output_root=args.output_root)
    elif args.run:
        result = run(
            study_path=args.study_path,
            output_root=args.output_root,
            max_tasks=args.max_tasks,
        )
    elif args.evaluate:
        result = evaluate(study_path=args.study_path, output_root=args.output_root)
    else:
        result = audit(study_path=args.study_path, output_root=args.output_root)
    print(json.dumps(economic._json_safe(result), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
