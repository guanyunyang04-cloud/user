from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from daily_research.path_policy.seq100_candidate_execution import (
    DEFAULT_DAILY_COHORT_CASH_CNY,
    ExecutionCostContract,
    _cashflow,
    _resolve_plan,
    parse_execution_cost_contract,
)
from daily_research.path_policy.qdp_v2_sequence_path_pack import (
    assert_qdp_source_fresh,
    assert_sequence_continuity_contract,
)


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
SOURCE_STUDY_ROOT = Path(
    "daily_research/output/path_policy/studies/"
    "seq100_candidate_complete_development_walkforward_20260712_v3"
)
SOURCE_PACK_MANIFEST = None
DEFAULT_OUTPUT_ROOT = Path(
    "daily_research/output/path_policy/studies/"
    "seq100_candidate_complete_exit_policy_audit_2022_2025_v1"
)
QDP_ACTIVE_PATH = Path("quant_data_platform/data/qdp_v2/active/active.json")
ACTIVE_EXECUTION_PATH = Path("daily_research/output/active_execution_strategy.json")

YEARS = (2022, 2023, 2024, 2025)
PROFILES = ("baseline", "hard_st")
TOP_K_VALUES = (1, 3, 5, 10)
PRIMARY_FIXED_HORIZONS = (5, 10, 20, 40, 60)
DIAGNOSTIC_FIXED_HORIZONS = (2,)
FIXED_HORIZONS = DIAGNOSTIC_FIXED_HORIZONS + PRIMARY_FIXED_HORIZONS
PREDICTED_POLICY = "predicted_exit"
ORACLE_POLICY = "oracle_executable"
MAX_POSITIONS = 3
STARTING_CASH_CNY = 1_000_000.0


def _workspace_path(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else WORKSPACE_ROOT / value


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with _workspace_path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _read_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(_workspace_path(path).read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return payload


def _write_json(path: str | Path, payload: Mapping[str, Any]) -> Path:
    target = _workspace_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(
        json.dumps(dict(payload), ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, target)
    return target


def _write_csv(path: str | Path, frame: pd.DataFrame) -> Path:
    target = _workspace_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    frame.to_csv(temporary, index=False, encoding="utf-8-sig")
    os.replace(temporary, target)
    return target


def _open_memmap(meta: Mapping[str, Any], *, dtype: str) -> np.memmap:
    return np.memmap(
        Path(str(meta["path"])),
        dtype=dtype,
        mode="r",
        shape=tuple(int(value) for value in meta["shape"]),
    )


def _universe_hash(symbols: Sequence[str]) -> str:
    return hashlib.sha256("\n".join(sorted(str(value) for value in symbols)).encode("utf-8")).hexdigest()


def _policy_for_horizon(horizon: int) -> str:
    return f"fixed_h{int(horizon)}"


def _policy_role(policy: str) -> str:
    if policy == _policy_for_horizon(2):
        return "diagnostic_fixed"
    if policy.startswith("fixed_h"):
        return "primary_fixed"
    if policy == PREDICTED_POLICY:
        return "current_control"
    if policy == ORACLE_POLICY:
        return "diagnostic_oracle"
    raise ValueError(f"unknown policy: {policy}")


class CandidateCompleteAuditPack:
    def __init__(self, manifest_path: str | Path) -> None:
        self.manifest_path = _workspace_path(manifest_path).resolve()
        self.manifest = _read_json(self.manifest_path)
        assert_qdp_source_fresh(self.manifest)
        assert_sequence_continuity_contract(self.manifest)
        self.forward_days = int(self.manifest["forward_days"])
        self.execution_tail_days = int(self.manifest["execution_tail_days"])
        self.execution_days = self.forward_days + self.execution_tail_days
        if self.forward_days != 60 or self.execution_tail_days != 20:
            raise ValueError("exit-policy audit requires forward_days=60 and execution_tail_days=20")
        self.date_values = np.asarray(self.manifest["date_values"], dtype=object)
        self.symbol_values = np.asarray(self.manifest["symbol_values"], dtype=object)
        self.date_to_idx = {str(value): idx for idx, value in enumerate(self.date_values)}
        self.symbol_to_idx = {str(value): idx for idx, value in enumerate(self.symbol_values)}
        execution = dict(self.manifest["execution_arrays"])
        masks = dict(self.manifest["masks"])
        self.entry_open_raw = _open_memmap(execution["entry_open_raw"], dtype="float32")
        self.exit_close_raw = _open_memmap(execution["exit_close_raw"], dtype="float32")
        self.exit_sellable = _open_memmap(masks["exit_sellable"], dtype="bool")
        self.entry_filled = _open_memmap(masks["entry_filled"], dtype="bool")
        self.candidate_index_path = Path(str(self.manifest["candidate_index_path"])).resolve()
        self.contract = parse_execution_cost_contract(self.manifest)
        terminal = dict(self.manifest.get("terminal_execution_contract", {}) or {})
        self.terminal_recovery_fraction = float(
            terminal.get("recovery_fraction_of_entry_notional", 0.0) or 0.0
        )

    def candidates_for_year(self, year: int) -> pd.DataFrame:
        columns = [
            "candidate_id",
            "year",
            "trade_date",
            "date_idx",
            "symbol_idx",
            "symbol",
            "entry_filled",
        ]
        frame = pd.read_parquet(
            self.candidate_index_path,
            columns=columns,
            filters=[("year", "=", int(year))],
        )
        frame["trade_date"] = pd.to_datetime(frame["trade_date"], errors="raise").dt.strftime("%Y-%m-%d")
        frame["symbol"] = frame["symbol"].astype(str)
        frame["entry_filled"] = frame["entry_filled"].astype("boolean").fillna(False).astype(bool)
        frame = frame.sort_values(["date_idx", "symbol_idx"], kind="mergesort").reset_index(drop=True)
        if frame.empty:
            raise ValueError(f"candidate index contains no rows for {year}")
        if not bool(frame["year"].astype(int).eq(int(year)).all()):
            raise ValueError(f"candidate index year filter drifted for {year}")
        return frame

    def execution_paths(self, date_idx: int, symbol_idx: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        symbols = np.asarray(symbol_idx, dtype=np.int64)
        start = int(date_idx) + 1
        stop = start + self.execution_days
        if start < 0 or stop > len(self.date_values):
            raise ValueError(f"execution window is incomplete for date_idx={date_idx}")
        entry = np.asarray(self.entry_open_raw[start, symbols], dtype=np.float64).copy()
        close = np.asarray(self.exit_close_raw[start:stop, symbols], dtype=np.float64).T.copy()
        sellable = np.asarray(self.exit_sellable[start:stop, symbols], dtype=bool).T.copy()
        return entry, close, sellable


@dataclass(frozen=True)
class ResolvedPlanBatch:
    planned_day: np.ndarray
    exit_day: np.ndarray
    exit_date_idx: np.ndarray
    exit_price: np.ndarray
    terminal_recovery: np.ndarray


@dataclass(frozen=True)
class CashflowBatch:
    order_filled: np.ndarray
    shares: np.ndarray
    ending_cash: np.ndarray
    net_return: np.ndarray
    total_cost: np.ndarray
    cash_utilization: np.ndarray


def _next_valid_exit_indices(exit_prices: np.ndarray, exit_sellable: np.ndarray) -> np.ndarray:
    prices = np.asarray(exit_prices, dtype=np.float64)
    sellable = np.asarray(exit_sellable, dtype=bool)
    if prices.ndim != 2 or prices.shape != sellable.shape:
        raise ValueError("exit price and sellable arrays must have identical [candidate, day] shapes")
    valid = sellable & np.isfinite(prices) & (prices >= 0.0)
    result = np.full(valid.shape, -1, dtype=np.int16)
    next_index = np.full(valid.shape[0], -1, dtype=np.int16)
    for day_idx in range(valid.shape[1] - 1, -1, -1):
        next_index = np.where(valid[:, day_idx], np.int16(day_idx), next_index)
        result[:, day_idx] = next_index
    return result


def resolve_planned_exit_batch(
    *,
    signal_date_idx: int,
    entry_filled: np.ndarray,
    entry_prices: np.ndarray,
    exit_prices: np.ndarray,
    next_valid_exit_idx: np.ndarray,
    planned_days: int | np.ndarray,
    forward_days: int = 60,
    execution_days: int = 80,
    terminal_recovery_fraction: float = 0.0,
) -> ResolvedPlanBatch:
    entries = np.asarray(entry_prices, dtype=np.float64)
    filled = np.asarray(entry_filled, dtype=bool)
    count = int(entries.size)
    planned = np.broadcast_to(np.asarray(planned_days, dtype=np.float64), (count,)).copy()
    if not bool(np.isfinite(planned).all()):
        raise ValueError("planned exit days must be finite")
    planned = np.clip(np.rint(planned), 2, int(forward_days)).astype(np.int16)
    lookup = np.asarray(next_valid_exit_idx, dtype=np.int16)
    if lookup.shape != (count, int(execution_days)):
        raise ValueError("next-valid lookup has an invalid shape")
    requested = lookup[np.arange(count), planned.astype(np.int64) - 1]
    terminal = requested < 0
    actual_idx = np.where(terminal, int(execution_days) - 1, requested).astype(np.int16)
    prices = np.asarray(exit_prices, dtype=np.float64)
    resolved_price = prices[np.arange(count), actual_idx.astype(np.int64)]
    resolved_price = np.where(terminal, entries * float(terminal_recovery_fraction), resolved_price)
    valid_entry = filled & np.isfinite(entries) & (entries > 0.0)
    resolved_price = np.where(valid_entry, resolved_price, np.nan)
    exit_day = np.where(valid_entry, actual_idx.astype(np.int32) + 1, -1).astype(np.int16)
    exit_date_idx = np.where(
        valid_entry,
        int(signal_date_idx) + exit_day.astype(np.int32),
        -1,
    ).astype(np.int32)
    return ResolvedPlanBatch(
        planned_day=planned,
        exit_day=exit_day,
        exit_date_idx=exit_date_idx,
        exit_price=resolved_price,
        terminal_recovery=terminal & valid_entry,
    )


def _stamp_tax_bps_by_date_idx(
    date_idx: np.ndarray,
    *,
    date_values: np.ndarray,
    contract: ExecutionCostContract,
) -> np.ndarray:
    indices = np.asarray(date_idx, dtype=np.int64)
    clipped = np.clip(indices, 0, len(date_values) - 1)
    dates = np.asarray(date_values, dtype=object)[clipped].astype(str)
    rates = np.full(indices.shape, np.nan, dtype=np.float64)
    for effective_date, rate in contract.stamp_tax_schedule:
        rates = np.where(dates >= str(effective_date), float(rate), rates)
    if bool(np.isnan(rates[indices >= 0]).any()):
        raise ValueError("one or more exit dates predate the stamp-tax schedule")
    return np.nan_to_num(rates, nan=0.0)


def _buy_terms_batch(
    *,
    allocated_cash: float,
    entry_filled: np.ndarray,
    entry_prices: np.ndarray,
    contract: ExecutionCostContract,
    slippage_multiplier: float,
) -> dict[str, np.ndarray]:
    cash = float(allocated_cash)
    if not math.isfinite(cash) or cash <= 0.0:
        raise ValueError("allocated cash must be finite and positive")
    entries = np.asarray(entry_prices, dtype=np.float64)
    valid_entry = np.asarray(entry_filled, dtype=bool) & np.isfinite(entries) & (entries > 0.0)
    slippage_rate = contract.slippage_bps * float(slippage_multiplier) / 10_000.0
    buy_price = entries * (1.0 + slippage_rate)
    shares = np.where(
        valid_entry,
        np.floor(cash / (buy_price * contract.lot_size)) * contract.lot_size,
        0.0,
    ).astype(np.int64)
    commission_rate = contract.commission_bps / 10_000.0
    transfer_rate = contract.transfer_fee_bps / 10_000.0
    for _ in range(8):
        buy_notional = shares * buy_price
        buy_commission = np.maximum(contract.minimum_commission_cny, buy_notional * commission_rate)
        buy_transfer = buy_notional * transfer_rate
        buy_cash = buy_notional + buy_commission + buy_transfer
        unaffordable = (shares > 0) & (buy_cash > cash + 1.0e-9)
        if not bool(unaffordable.any()):
            break
        shares = np.where(unaffordable, shares - contract.lot_size, shares)
    shares = np.maximum(shares, 0).astype(np.int64)
    order_filled = valid_entry & (shares > 0)
    buy_notional = shares * buy_price
    buy_commission = np.where(
        order_filled,
        np.maximum(contract.minimum_commission_cny, buy_notional * commission_rate),
        0.0,
    )
    buy_transfer = np.where(order_filled, buy_notional * transfer_rate, 0.0)
    buy_cash = np.where(order_filled, buy_notional + buy_commission + buy_transfer, 0.0)
    return {
        "valid_entry": valid_entry,
        "order_filled": order_filled,
        "buy_price": buy_price,
        "shares": shares,
        "buy_notional": buy_notional,
        "buy_commission": buy_commission,
        "buy_transfer": buy_transfer,
        "buy_cash": buy_cash,
    }


def cashflow_batch(
    *,
    allocated_cash: float,
    entry_filled: np.ndarray,
    entry_prices: np.ndarray,
    plan: ResolvedPlanBatch,
    date_values: np.ndarray,
    contract: ExecutionCostContract,
    slippage_multiplier: float,
) -> CashflowBatch:
    cash = float(allocated_cash)
    entries = np.asarray(entry_prices, dtype=np.float64)
    terms = _buy_terms_batch(
        allocated_cash=cash,
        entry_filled=entry_filled,
        entry_prices=entries,
        contract=contract,
        slippage_multiplier=slippage_multiplier,
    )
    active = terms["order_filled"] & np.isfinite(plan.exit_price) & (plan.exit_date_idx >= 0)
    slippage_rate = contract.slippage_bps * float(slippage_multiplier) / 10_000.0
    sell_price = np.asarray(plan.exit_price, dtype=np.float64) * max(0.0, 1.0 - slippage_rate)
    sell_notional = terms["shares"] * sell_price
    commission_rate = contract.commission_bps / 10_000.0
    transfer_rate = contract.transfer_fee_bps / 10_000.0
    stamp_bps = _stamp_tax_bps_by_date_idx(
        plan.exit_date_idx,
        date_values=date_values,
        contract=contract,
    )
    sell_commission = np.where(
        active,
        np.maximum(contract.minimum_commission_cny, sell_notional * commission_rate),
        0.0,
    )
    sell_transfer = np.where(active, sell_notional * transfer_rate, 0.0)
    stamp_tax = np.where(active, sell_notional * stamp_bps / 10_000.0, 0.0)
    ending_cash = np.where(
        active,
        cash - terms["buy_cash"] + sell_notional - sell_commission - sell_transfer - stamp_tax,
        cash,
    )
    explicit_cost = terms["buy_commission"] + terms["buy_transfer"] + sell_commission + sell_transfer + stamp_tax
    slippage_cost = terms["shares"] * (
        (terms["buy_price"] - entries) + (np.asarray(plan.exit_price, dtype=np.float64) - sell_price)
    )
    total_cost = np.where(active, explicit_cost + slippage_cost, 0.0)
    utilization = np.where(active, terms["buy_cash"] / cash, 0.0)
    return CashflowBatch(
        order_filled=active,
        shares=terms["shares"],
        ending_cash=ending_cash,
        net_return=ending_cash / cash - 1.0,
        total_cost=total_cost,
        cash_utilization=utilization,
    )


def oracle_executable_outcome_batch(
    *,
    signal_date_idx: int,
    allocated_cash: float,
    entry_filled: np.ndarray,
    entry_prices: np.ndarray,
    exit_prices: np.ndarray,
    next_valid_exit_idx: np.ndarray,
    date_values: np.ndarray,
    contract: ExecutionCostContract,
    slippage_multiplier: float,
    forward_days: int = 60,
    execution_days: int = 80,
    terminal_recovery_fraction: float = 0.0,
) -> tuple[ResolvedPlanBatch, CashflowBatch]:
    cash = float(allocated_cash)
    entries = np.asarray(entry_prices, dtype=np.float64)
    filled = np.asarray(entry_filled, dtype=bool)
    prices = np.asarray(exit_prices, dtype=np.float64)
    lookup = np.asarray(next_valid_exit_idx, dtype=np.int16)
    count = int(entries.size)
    if prices.shape != (count, int(execution_days)):
        raise ValueError("exit price matrix has an invalid shape")
    if lookup.shape != prices.shape:
        raise ValueError("next-valid lookup has an invalid shape")

    # Planned days are columns.  This preserves the scalar implementation's
    # smallest-day tie break because np.argmax returns the first maximum.
    planned_options = np.arange(2, int(forward_days) + 1, dtype=np.int16)
    requested = lookup[:, planned_options.astype(np.int64) - 1]
    terminal = requested < 0
    actual_idx = np.where(terminal, int(execution_days) - 1, requested).astype(np.int16)
    row_idx = np.arange(count, dtype=np.int64)[:, None]
    resolved_prices = prices[row_idx, actual_idx.astype(np.int64)]
    resolved_prices = np.where(
        terminal,
        entries[:, None] * float(terminal_recovery_fraction),
        resolved_prices,
    )
    valid_entry = filled & np.isfinite(entries) & (entries > 0.0)
    resolved_prices = np.where(valid_entry[:, None], resolved_prices, np.nan)
    exit_days = np.where(valid_entry[:, None], actual_idx.astype(np.int32) + 1, -1).astype(np.int16)
    exit_date_indices = np.where(
        valid_entry[:, None],
        int(signal_date_idx) + exit_days.astype(np.int32),
        -1,
    ).astype(np.int32)

    terms = _buy_terms_batch(
        allocated_cash=cash,
        entry_filled=filled,
        entry_prices=entries,
        contract=contract,
        slippage_multiplier=slippage_multiplier,
    )
    active = terms["order_filled"][:, None] & np.isfinite(resolved_prices) & (exit_date_indices >= 0)
    slippage_rate = contract.slippage_bps * float(slippage_multiplier) / 10_000.0
    sell_prices = resolved_prices * max(0.0, 1.0 - slippage_rate)
    sell_notional = terms["shares"][:, None] * sell_prices
    commission_rate = contract.commission_bps / 10_000.0
    transfer_rate = contract.transfer_fee_bps / 10_000.0
    stamp_bps = _stamp_tax_bps_by_date_idx(
        exit_date_indices,
        date_values=date_values,
        contract=contract,
    )
    sell_commission = np.where(
        active,
        np.maximum(contract.minimum_commission_cny, sell_notional * commission_rate),
        0.0,
    )
    sell_transfer = np.where(active, sell_notional * transfer_rate, 0.0)
    stamp_tax = np.where(active, sell_notional * stamp_bps / 10_000.0, 0.0)
    ending_cash = np.where(
        active,
        cash
        - terms["buy_cash"][:, None]
        + sell_notional
        - sell_commission
        - sell_transfer
        - stamp_tax,
        cash,
    )
    net_returns = ending_cash / cash - 1.0
    explicit_cost = (
        terms["buy_commission"][:, None]
        + terms["buy_transfer"][:, None]
        + sell_commission
        + sell_transfer
        + stamp_tax
    )
    slippage_cost = terms["shares"][:, None] * (
        (terms["buy_price"] - entries)[:, None] + (resolved_prices - sell_prices)
    )
    total_cost = np.where(active, explicit_cost + slippage_cost, 0.0)
    utilization = np.where(active, terms["buy_cash"][:, None] / cash, 0.0)

    best_column = np.argmax(net_returns, axis=1)
    best = (np.arange(count, dtype=np.int64), best_column)
    chosen_planned = planned_options[best_column]
    return (
        ResolvedPlanBatch(
            planned_day=chosen_planned,
            exit_day=exit_days[best],
            exit_date_idx=exit_date_indices[best],
            exit_price=resolved_prices[best],
            terminal_recovery=terminal[best] & valid_entry,
        ),
        CashflowBatch(
            order_filled=active[best],
            shares=terms["shares"],
            ending_cash=ending_cash[best],
            net_return=net_returns[best],
            total_cost=total_cost[best],
            cash_utilization=utilization[best],
        ),
    )


def _mean(values: np.ndarray) -> float:
    numeric = np.asarray(values, dtype=np.float64)
    finite = numeric[np.isfinite(numeric)]
    return float(finite.mean()) if finite.size else math.nan


def _metric_row(
    *,
    profile: str,
    year: int,
    trade_date: str,
    policy: str,
    top_k: int,
    cost_scenario: str,
    selected_idx: np.ndarray,
    selected_symbols: Sequence[str],
    universe_hash: str,
    plan: ResolvedPlanBatch,
    cash: CashflowBatch,
) -> dict[str, Any]:
    selected = np.asarray(selected_idx, dtype=np.int64)
    universe = np.arange(cash.net_return.size, dtype=np.int64)
    selected_return = _mean(cash.net_return[selected])
    universe_return = _mean(cash.net_return[universe])
    selected_terminal = _mean((plan.terminal_recovery[selected] & cash.order_filled[selected]).astype(float))
    universe_terminal = _mean((plan.terminal_recovery[universe] & cash.order_filled[universe]).astype(float))
    selected_deferred = _mean(
        (
            (plan.exit_day[selected] > plan.planned_day[selected])
            & (~plan.terminal_recovery[selected])
            & cash.order_filled[selected]
        ).astype(float)
    )
    universe_deferred = _mean(
        (
            (plan.exit_day[universe] > plan.planned_day[universe])
            & (~plan.terminal_recovery[universe])
            & cash.order_filled[universe]
        ).astype(float)
    )
    return {
        "profile": profile,
        "development_year": int(year),
        "trade_date": str(trade_date),
        "policy": policy,
        "policy_role": _policy_role(policy),
        "top_k": int(top_k),
        "cost_scenario": cost_scenario,
        "selected_count": int(selected.size),
        "universe_count": int(universe.size),
        "universe_hash": universe_hash,
        "selected_symbols_json": json.dumps(list(selected_symbols), ensure_ascii=False, separators=(",", ":")),
        "selected_mean_net_return": selected_return,
        "universe_mean_net_return": universe_return,
        "alpha_mean_net_return": selected_return - universe_return,
        "selected_entry_fill_rate": _mean(cash.order_filled[selected].astype(float)),
        "universe_entry_fill_rate": _mean(cash.order_filled[universe].astype(float)),
        "selected_terminal_recovery_rate": selected_terminal,
        "universe_terminal_recovery_rate": universe_terminal,
        "selected_deferred_exit_rate": selected_deferred,
        "universe_deferred_exit_rate": universe_deferred,
        "selected_mean_planned_exit_day": _mean(plan.planned_day[selected]),
        "universe_mean_planned_exit_day": _mean(plan.planned_day[universe]),
        "selected_mean_resolved_exit_day": _mean(np.where(plan.exit_day[selected] >= 0, plan.exit_day[selected], np.nan)),
        "universe_mean_resolved_exit_day": _mean(np.where(plan.exit_day[universe] >= 0, plan.exit_day[universe], np.nan)),
        "selected_mean_cash_utilization": _mean(cash.cash_utilization[selected]),
        "universe_mean_cash_utilization": _mean(cash.cash_utilization[universe]),
        "selected_mean_execution_cost_cny": _mean(cash.total_cost[selected]),
        "universe_mean_execution_cost_cny": _mean(cash.total_cost[universe]),
    }


def _source_run_dir(source_root: str | Path, profile: str, year: int) -> Path:
    root = _workspace_path(source_root) / "runs" / "development"
    matches = sorted(root.glob(f"seq100_development_{profile}_{int(year)}_seed7_*"))
    if len(matches) != 1:
        raise ValueError(
            f"expected exactly one frozen source run for {profile}/{year}, found {len(matches)}"
        )
    return matches[0]


def _load_source_evidence(source_root: str | Path, profile: str, year: int) -> tuple[pd.DataFrame, pd.DataFrame, Path]:
    run_dir = _source_run_dir(source_root, profile, year)
    topk_path = run_dir / "topk_candidates.parquet"
    daily_path = run_dir / "daily_topk_metrics.csv"
    topk = pd.read_parquet(topk_path)
    daily = pd.read_csv(daily_path)
    topk["trade_date"] = pd.to_datetime(topk["trade_date"], errors="raise").dt.strftime("%Y-%m-%d")
    daily["trade_date"] = pd.to_datetime(daily["trade_date"], errors="raise").dt.strftime("%Y-%m-%d")
    topk["symbol"] = topk["symbol"].astype(str)
    if bool(topk.duplicated(["trade_date", "score_rank"]).any()):
        raise ValueError(f"source TopK ranks are duplicated for {profile}/{year}")
    if int(topk.groupby("trade_date").size().min()) < max(TOP_K_VALUES):
        raise ValueError(f"source TopK evidence does not cover Top{max(TOP_K_VALUES)} for {profile}/{year}")
    return topk, daily, run_dir


def _status_rate(raw_json: Any, status: str, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    try:
        payload = json.loads(str(raw_json))
    except json.JSONDecodeError:
        payload = {}
    return float(int(payload.get(status, 0) or 0) / denominator)


def _predicted_policy_rows(
    *,
    profile: str,
    year: int,
    daily: pd.DataFrame,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    source = daily.copy()
    if "score_column" in source.columns:
        source = source[source["score_column"].astype(str).eq("score")].copy()
    source = source[source["top_k"].astype(int).isin(TOP_K_VALUES)]
    for item in source.itertuples(index=False):
        values = item._asdict()
        selected_count = int(values["selected_count"])
        universe_count = int(values["universe_count"])
        for scenario in ("base", "stress"):
            suffix = "" if scenario == "base" else "_stress"
            selected_fill_key = "selected_entry_fill_rate" + suffix
            universe_fill_key = "universe_entry_fill_rate" + suffix
            selected_terminal_key = f"selected_terminal_recovery_count_{scenario}"
            universe_terminal_key = f"universe_terminal_recovery_count_{scenario}"
            selected_cost_key = f"selected_execution_cost_{scenario}_cny"
            universe_cost_key = f"universe_execution_cost_{scenario}_cny"
            selected_return_key = f"selected_net_realized_plan_return_{scenario}"
            universe_return_key = f"universe_net_realized_plan_return_{scenario}"
            selected_return = float(values[selected_return_key])
            universe_return = float(values[universe_return_key])
            rows.append(
                {
                    "profile": profile,
                    "development_year": int(year),
                    "trade_date": str(values["trade_date"]),
                    "policy": PREDICTED_POLICY,
                    "policy_role": _policy_role(PREDICTED_POLICY),
                    "top_k": int(values["top_k"]),
                    "cost_scenario": scenario,
                    "selected_count": selected_count,
                    "universe_count": universe_count,
                    "universe_hash": str(values["universe_hash"]),
                    "selected_symbols_json": str(values["selected_symbols_json"]),
                    "selected_mean_net_return": selected_return,
                    "universe_mean_net_return": universe_return,
                    "alpha_mean_net_return": selected_return - universe_return,
                    "selected_entry_fill_rate": float(values[selected_fill_key]),
                    "universe_entry_fill_rate": float(values[universe_fill_key]),
                    "selected_terminal_recovery_rate": float(values[selected_terminal_key]) / selected_count,
                    "universe_terminal_recovery_rate": float(values[universe_terminal_key]) / universe_count,
                    "selected_deferred_exit_rate": _status_rate(
                        values["selected_exit_status_counts_json"], "filled_deferred_exit", selected_count
                    ),
                    "universe_deferred_exit_rate": _status_rate(
                        values["universe_exit_status_counts_json"], "filled_deferred_exit", universe_count
                    ),
                    "selected_mean_planned_exit_day": float(values["selected_predicted_exit_day"]),
                    "universe_mean_planned_exit_day": float(values["universe_predicted_exit_day"]),
                    "selected_mean_resolved_exit_day": float(values["selected_realized_plan_exit_day"]),
                    "universe_mean_resolved_exit_day": float(values["universe_realized_plan_exit_day"]),
                    "selected_mean_cash_utilization": float(
                        values[f"selected_cash_utilization_{scenario}"]
                    ),
                    "universe_mean_cash_utilization": float(
                        values[f"universe_cash_utilization_{scenario}"]
                    ),
                    "selected_mean_execution_cost_cny": float(values[selected_cost_key]),
                    "universe_mean_execution_cost_cny": float(values[universe_cost_key]),
                }
            )
    return rows


def _aggregate_daily_metrics(daily: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    group_columns = ["profile", "development_year", "policy", "policy_role", "top_k", "cost_scenario"]
    mean_columns = [
        "selected_mean_net_return",
        "universe_mean_net_return",
        "alpha_mean_net_return",
        "selected_entry_fill_rate",
        "universe_entry_fill_rate",
        "selected_terminal_recovery_rate",
        "universe_terminal_recovery_rate",
        "selected_deferred_exit_rate",
        "universe_deferred_exit_rate",
        "selected_mean_planned_exit_day",
        "universe_mean_planned_exit_day",
        "selected_mean_resolved_exit_day",
        "universe_mean_resolved_exit_day",
        "selected_mean_cash_utilization",
        "universe_mean_cash_utilization",
        "selected_mean_execution_cost_cny",
        "universe_mean_execution_cost_cny",
    ]
    yearly = daily.groupby(group_columns, as_index=False, sort=True)[mean_columns].mean()
    date_counts = daily.groupby(group_columns, as_index=False, sort=True)["trade_date"].nunique().rename(
        columns={"trade_date": "date_count"}
    )
    yearly = yearly.merge(date_counts, on=group_columns, how="left", validate="one_to_one")
    positive = daily.assign(
        selected_positive=daily["selected_mean_net_return"].astype(float) > 0.0,
        alpha_positive=daily["alpha_mean_net_return"].astype(float) > 0.0,
    ).groupby(group_columns, as_index=False, sort=True)[["selected_positive", "alpha_positive"]].mean()
    positive = positive.rename(
        columns={"selected_positive": "positive_selected_day_rate", "alpha_positive": "positive_alpha_day_rate"}
    )
    yearly = yearly.merge(positive, on=group_columns, how="left", validate="one_to_one")

    equal_group = ["profile", "policy", "policy_role", "top_k", "cost_scenario"]
    equal_year = yearly.groupby(equal_group, as_index=False, sort=True)[mean_columns].mean()
    year_counts = yearly.assign(
        selected_positive_year=yearly["selected_mean_net_return"].astype(float) > 0.0,
        alpha_positive_year=yearly["alpha_mean_net_return"].astype(float) > 0.0,
    ).groupby(equal_group, as_index=False, sort=True).agg(
        development_year_count=("development_year", "nunique"),
        selected_positive_year_count=("selected_positive_year", "sum"),
        alpha_positive_year_count=("alpha_positive_year", "sum"),
        worst_year_selected_mean_net_return=("selected_mean_net_return", "min"),
        worst_year_alpha_mean_net_return=("alpha_mean_net_return", "min"),
    )
    equal_year = equal_year.merge(year_counts, on=equal_group, how="left", validate="one_to_one")
    return yearly, equal_year


@dataclass
class _PendingOrder:
    signal_date_idx: int
    symbol_idx: int
    symbol: str
    entry_filled: bool
    requested_planned_day: int | None


@dataclass
class _Position:
    signal_date_idx: int
    symbol_idx: int
    symbol: str
    shares: int
    entry_price_raw: float
    buy_cash: float
    planned_day: int
    last_mark_price: float
    entry_date_idx: int


def _buy_order(
    *,
    available_cash: float,
    allocated_cash: float,
    entry_price: float,
    contract: ExecutionCostContract,
    slippage_multiplier: float,
) -> tuple[int, float, float, float]:
    allocation = min(float(available_cash), float(allocated_cash))
    if not math.isfinite(entry_price) or entry_price <= 0.0 or allocation <= 0.0:
        return 0, 0.0, 0.0, 0.0
    terms = _buy_terms_batch(
        allocated_cash=allocation,
        entry_filled=np.asarray([True]),
        entry_prices=np.asarray([entry_price], dtype=np.float64),
        contract=contract,
        slippage_multiplier=slippage_multiplier,
    )
    if not bool(terms["order_filled"][0]):
        return 0, 0.0, 0.0, 0.0
    shares = int(terms["shares"][0])
    buy_cash = float(terms["buy_cash"][0])
    explicit_cost = float(terms["buy_commission"][0] + terms["buy_transfer"][0])
    slippage_cost = float(shares * (terms["buy_price"][0] - entry_price))
    return shares, buy_cash, explicit_cost + slippage_cost, float(terms["buy_notional"][0])


def _sell_order(
    *,
    shares: int,
    exit_price: float,
    exit_date_idx: int,
    date_values: np.ndarray,
    contract: ExecutionCostContract,
    slippage_multiplier: float,
) -> tuple[float, float, float]:
    slippage_rate = contract.slippage_bps * float(slippage_multiplier) / 10_000.0
    sell_price = float(exit_price) * max(0.0, 1.0 - slippage_rate)
    sell_notional = int(shares) * sell_price
    commission = max(contract.minimum_commission_cny, sell_notional * contract.commission_bps / 10_000.0)
    transfer = sell_notional * contract.transfer_fee_bps / 10_000.0
    stamp_bps = float(
        _stamp_tax_bps_by_date_idx(
            np.asarray([exit_date_idx]), date_values=date_values, contract=contract
        )[0]
    )
    stamp = sell_notional * stamp_bps / 10_000.0
    proceeds = sell_notional - commission - transfer - stamp
    slippage_cost = int(shares) * (float(exit_price) - sell_price)
    return float(proceeds), float(commission + transfer + stamp + slippage_cost), float(sell_notional)


def _portfolio_equity(
    *,
    cash: float,
    positions: Mapping[int, _Position],
    mark_panel: np.ndarray,
    date_idx: int,
) -> float:
    value = float(cash)
    for position in positions.values():
        mark = float(mark_panel[int(date_idx), int(position.symbol_idx)])
        if math.isfinite(mark) and mark >= 0.0:
            position.last_mark_price = mark
        value += int(position.shares) * float(position.last_mark_price)
    return float(value)


def _oracle_planned_day_scalar(
    *,
    pack: CandidateCompleteAuditPack,
    signal_date_idx: int,
    symbol_idx: int,
    entry_price: float,
    allocated_cash: float,
    slippage_multiplier: float,
) -> int:
    start = int(signal_date_idx) + 1
    stop = start + pack.execution_days
    prices = np.asarray(pack.exit_close_raw[start:stop, int(symbol_idx)], dtype=np.float64)
    sellable = np.asarray(pack.exit_sellable[start:stop, int(symbol_idx)], dtype=bool)

    def date_for_day(_row: int, day: int) -> str:
        return str(pack.date_values[int(signal_date_idx) + int(day)])

    best_day = 2
    best_return = -math.inf
    for planned_day in range(2, pack.forward_days + 1):
        plan = _resolve_plan(
            predicted_exit_day=float(planned_day),
            entry_filled=True,
            entry_price=float(entry_price),
            exit_prices=prices,
            exit_sellable=sellable,
            forward_days=pack.forward_days,
            horizon=pack.execution_days,
            terminal_recovery_fraction=pack.terminal_recovery_fraction,
            date_for_day=date_for_day,
            row=0,
        )
        cashflow = _cashflow(
            allocated_cash=float(allocated_cash),
            entry_filled=True,
            entry_price=float(entry_price),
            plan=plan,
            contract=pack.contract,
            slippage_multiplier=float(slippage_multiplier),
        )
        if float(cashflow.net_return) > best_return:
            best_return = float(cashflow.net_return)
            best_day = int(planned_day)
    return best_day


def _simulate_stateful_top3(
    *,
    pack: CandidateCompleteAuditPack,
    topk: pd.DataFrame,
    profile: str,
    year: int,
    policy: str,
    cost_scenario: str,
) -> tuple[dict[str, Any], pd.DataFrame]:
    multiplier = 1.0 if cost_scenario == "base" else pack.contract.stress_slippage_multiplier
    ranked = topk[topk["score_rank"].astype(int).le(3)].copy()
    ranked = ranked.sort_values(["trade_date", "score_rank"], kind="mergesort")
    ranked_by_date = {date: frame.copy() for date, frame in ranked.groupby("trade_date", sort=True)}
    signal_dates = sorted(ranked_by_date)
    if not signal_dates:
        raise ValueError(f"no Top3 signals for {profile}/{year}")
    first_signal_idx = int(pack.date_to_idx[signal_dates[0]])
    last_signal_idx = int(pack.date_to_idx[signal_dates[-1]])
    final_date_idx = last_signal_idx + pack.execution_days
    if final_date_idx >= len(pack.date_values):
        raise ValueError(f"stateful audit tail is incomplete for {profile}/{year}")

    cash = float(STARTING_CASH_CNY)
    positions: dict[int, _Position] = {}
    pending: list[_PendingOrder] = []
    equity_rows: list[dict[str, Any]] = []
    realized_pnls: list[float] = []
    holding_days: list[int] = []
    fees_paid = 0.0
    traded_notional = 0.0
    buy_count = 0
    sell_count = 0
    failed_entry_count = 0
    terminal_recovery_count = 0
    deferred_exit_count = 0

    for date_idx in range(first_signal_idx, final_date_idx + 1):
        trade_date = str(pack.date_values[date_idx])
        if pending:
            equity_open = _portfolio_equity(
                cash=cash,
                positions=positions,
                mark_panel=pack.entry_open_raw,
                date_idx=date_idx,
            )
            for order in pending:
                if len(positions) >= MAX_POSITIONS or order.symbol_idx in positions:
                    continue
                entry_price = float(pack.entry_open_raw[date_idx, order.symbol_idx])
                if not order.entry_filled or not math.isfinite(entry_price) or entry_price <= 0.0:
                    failed_entry_count += 1
                    continue
                allocation = min(cash, max(equity_open, 0.0) / MAX_POSITIONS)
                shares, buy_cash, buy_cost, buy_notional = _buy_order(
                    available_cash=cash,
                    allocated_cash=allocation,
                    entry_price=entry_price,
                    contract=pack.contract,
                    slippage_multiplier=multiplier,
                )
                if shares <= 0:
                    failed_entry_count += 1
                    continue
                planned_day = order.requested_planned_day
                if policy == ORACLE_POLICY:
                    planned_day = _oracle_planned_day_scalar(
                        pack=pack,
                        signal_date_idx=order.signal_date_idx,
                        symbol_idx=order.symbol_idx,
                        entry_price=entry_price,
                        allocated_cash=allocation,
                        slippage_multiplier=multiplier,
                    )
                if planned_day is None:
                    raise AssertionError("stateful order is missing a planned exit day")
                cash -= buy_cash
                fees_paid += buy_cost
                traded_notional += buy_notional
                buy_count += 1
                positions[order.symbol_idx] = _Position(
                    signal_date_idx=order.signal_date_idx,
                    symbol_idx=order.symbol_idx,
                    symbol=order.symbol,
                    shares=shares,
                    entry_price_raw=entry_price,
                    buy_cash=buy_cash,
                    planned_day=int(planned_day),
                    last_mark_price=entry_price,
                    entry_date_idx=date_idx,
                )
            pending = []

        for position in positions.values():
            mark = float(pack.exit_close_raw[date_idx, position.symbol_idx])
            if math.isfinite(mark) and mark >= 0.0:
                position.last_mark_price = mark

        for symbol_idx, position in list(positions.items()):
            planned_date_idx = position.signal_date_idx + position.planned_day
            terminal_date_idx = position.signal_date_idx + pack.execution_days
            if date_idx < planned_date_idx:
                continue
            exit_price = float(pack.exit_close_raw[date_idx, symbol_idx])
            sellable = bool(pack.exit_sellable[date_idx, symbol_idx]) and math.isfinite(exit_price) and exit_price >= 0.0
            terminal = date_idx >= terminal_date_idx and not sellable
            if not sellable and not terminal:
                continue
            if terminal:
                exit_price = position.entry_price_raw * pack.terminal_recovery_fraction
                terminal_recovery_count += 1
            elif date_idx > planned_date_idx:
                deferred_exit_count += 1
            proceeds, sell_cost, sell_notional = _sell_order(
                shares=position.shares,
                exit_price=exit_price,
                exit_date_idx=date_idx,
                date_values=pack.date_values,
                contract=pack.contract,
                slippage_multiplier=multiplier,
            )
            cash += proceeds
            fees_paid += sell_cost
            traded_notional += sell_notional
            sell_count += 1
            realized_pnls.append(float(proceeds - position.buy_cash))
            holding_days.append(int(date_idx - position.entry_date_idx))
            positions.pop(symbol_idx)

        equity = _portfolio_equity(
            cash=cash,
            positions=positions,
            mark_panel=pack.exit_close_raw,
            date_idx=date_idx,
        )
        capital_utilization = 0.0 if equity <= 0.0 else float(1.0 - cash / equity)
        equity_rows.append(
            {
                "profile": profile,
                "development_year": int(year),
                "policy": policy,
                "policy_role": _policy_role(policy),
                "cost_scenario": cost_scenario,
                "trade_date": trade_date,
                "date_idx": int(date_idx),
                "equity": float(equity),
                "cash": float(cash),
                "position_count": int(len(positions)),
                "capital_utilization": capital_utilization,
                "inside_signal_year": bool(date_idx <= last_signal_idx),
            }
        )

        date_rows = ranked_by_date.get(trade_date)
        if date_rows is not None:
            vacancies = max(MAX_POSITIONS - len(positions), 0)
            if vacancies:
                for item in date_rows.itertuples(index=False):
                    symbol = str(item.symbol)
                    symbol_idx = int(pack.symbol_to_idx[symbol])
                    if symbol_idx in positions:
                        continue
                    if len(pending) >= vacancies:
                        break
                    if policy.startswith("fixed_h"):
                        requested_day = int(policy.removeprefix("fixed_h"))
                    elif policy == PREDICTED_POLICY:
                        requested_day = int(np.clip(round(float(item.predicted_exit_day)), 2, pack.forward_days))
                    elif policy == ORACLE_POLICY:
                        requested_day = None
                    else:
                        raise ValueError(policy)
                    source_entry_filled = bool(pack.entry_filled[date_idx, symbol_idx])
                    recorded_entry_filled = bool(round(float(item.realized_plan_entry_filled)))
                    if source_entry_filled != recorded_entry_filled:
                        raise ValueError(
                            f"entry-fill identity drift for {profile}/{year}/{trade_date}/{symbol}"
                        )
                    pending.append(
                        _PendingOrder(
                            signal_date_idx=int(date_idx),
                            symbol_idx=symbol_idx,
                            symbol=symbol,
                            entry_filled=source_entry_filled,
                            requested_planned_day=requested_day,
                        )
                    )

    if positions or pending:
        raise AssertionError(f"stateful portfolio did not drain by D+80 for {profile}/{year}/{policy}")
    equity_frame = pd.DataFrame(equity_rows)
    equity_values = equity_frame["equity"].to_numpy(dtype=np.float64)
    path = np.r_[STARTING_CASH_CNY, equity_values]
    drawdown = path / np.maximum.accumulate(path) - 1.0
    signal_slice = equity_frame[equity_frame["inside_signal_year"]]
    final_equity = float(equity_values[-1])
    metric = {
        "profile": profile,
        "development_year": int(year),
        "policy": policy,
        "policy_role": _policy_role(policy),
        "cost_scenario": cost_scenario,
        "starting_cash_cny": float(STARTING_CASH_CNY),
        "ending_equity_cny": final_equity,
        "net_return": float(final_equity / STARTING_CASH_CNY - 1.0),
        "maximum_drawdown": float(drawdown.min()),
        "fees_and_slippage_cny": float(fees_paid),
        "turnover_notional_cny": float(traded_notional),
        "turnover_to_starting_cash": float(traded_notional / STARTING_CASH_CNY),
        "buy_count": int(buy_count),
        "sell_count": int(sell_count),
        "failed_entry_count": int(failed_entry_count),
        "terminal_recovery_count": int(terminal_recovery_count),
        "deferred_exit_count": int(deferred_exit_count),
        "closed_trade_count": int(len(realized_pnls)),
        "winning_trade_rate": float(np.mean(np.asarray(realized_pnls) > 0.0)) if realized_pnls else 0.0,
        "mean_holding_days": float(np.mean(holding_days)) if holding_days else 0.0,
        "mean_signal_year_position_count": float(signal_slice["position_count"].mean()),
        "mean_signal_year_capital_utilization": float(signal_slice["capital_utilization"].mean()),
        "maximum_position_count": int(equity_frame["position_count"].max()),
        "signal_date_count": int(len(signal_dates)),
        "portfolio_contract": "frozen_daily_top3_equal_third_new_positions_max3_no_unfilled_replacement",
    }
    return metric, equity_frame


def _stateful_equal_year(metrics: pd.DataFrame) -> pd.DataFrame:
    group = ["profile", "policy", "policy_role", "cost_scenario"]
    numeric = [
        "net_return",
        "maximum_drawdown",
        "fees_and_slippage_cny",
        "turnover_to_starting_cash",
        "winning_trade_rate",
        "mean_holding_days",
        "mean_signal_year_position_count",
        "mean_signal_year_capital_utilization",
    ]
    result = metrics.groupby(group, as_index=False, sort=True)[numeric].mean()
    counts = metrics.assign(
        positive_year=metrics["net_return"].astype(float) > 0.0,
        stress_survival=metrics["net_return"].astype(float) > -0.5,
    ).groupby(group, as_index=False, sort=True).agg(
        development_year_count=("development_year", "nunique"),
        positive_year_count=("positive_year", "sum"),
        worst_year_net_return=("net_return", "min"),
        best_year_net_return=("net_return", "max"),
        survival_year_count=("stress_survival", "sum"),
    )
    return result.merge(counts, on=group, how="left", validate="one_to_one")


def run_exit_policy_audit(
    *,
    source_study_root: str | Path = SOURCE_STUDY_ROOT,
    pack_manifest: str | Path | None = SOURCE_PACK_MANIFEST,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    years: Sequence[int] = YEARS,
    profiles: Sequence[str] = PROFILES,
    include_stateful: bool = True,
) -> dict[str, Any]:
    output = _workspace_path(output_root).resolve()
    output.mkdir(parents=True, exist_ok=True)
    normalized_years = tuple(int(value) for value in years)
    normalized_profiles = tuple(str(value) for value in profiles)
    if any(year not in YEARS for year in normalized_years):
        raise ValueError(f"unsupported development years: {normalized_years}")
    if any(profile not in PROFILES for profile in normalized_profiles):
        raise ValueError(f"unsupported profiles: {normalized_profiles}")

    qdp_active_before = _sha256_file(QDP_ACTIVE_PATH)
    active_execution_path = _workspace_path(ACTIVE_EXECUTION_PATH)
    active_execution_before = active_execution_path.exists()
    active_execution_sha256_before = (
        _sha256_file(active_execution_path) if active_execution_before else None
    )
    if pack_manifest is None:
        raise ValueError("run_exit_policy_audit requires an explicit pack_manifest")
    pack = CandidateCompleteAuditPack(pack_manifest)
    source_selection_path = _workspace_path(source_study_root) / "development_selection.json"
    source_selection = _read_json(source_selection_path)
    if source_selection.get("winner") is not None:
        raise ValueError("source v3 study unexpectedly has a winner")

    contract_payload = {
        "schema_version": 1,
        "audit_id": "seq100_candidate_complete_exit_policy_audit_2022_2025_v1",
        "audit_type": "zero_training_frozen_ranking_exit_policy_decomposition",
        "source_study": str(_workspace_path(source_study_root).resolve()),
        "source_selection_sha256": _sha256_file(source_selection_path),
        "source_pack_manifest": str(pack.manifest_path),
        "source_pack_manifest_sha256": _sha256_file(pack.manifest_path),
        "execution_cost_contract_sha256": pack.contract.semantic_sha256,
        "terminal_recovery_fraction": pack.terminal_recovery_fraction,
        "protected_qdp_active_sha256_at_start": qdp_active_before.upper(),
        "protected_active_execution_present_at_start": active_execution_before,
        "protected_active_execution_sha256_at_start": active_execution_sha256_before,
        "profiles": list(normalized_profiles),
        "development_years": list(normalized_years),
        "top_k_values": list(TOP_K_VALUES),
        "policies": [
            *[_policy_for_horizon(value) for value in FIXED_HORIZONS],
            PREDICTED_POLICY,
            ORACLE_POLICY,
        ],
        "primary_fixed_horizons": list(PRIMARY_FIXED_HORIZONS),
        "diagnostic_fixed_horizons": list(DIAGNOSTIC_FIXED_HORIZONS),
        "ranking": "frozen source Top100 score order; no future reranking",
        "entry": "D+1 raw open; source candidate entry_filled; no failed-entry replacement",
        "exit": "raw close; T+1 minimum; blocked exit retries through absolute D+80",
        "oracle": "per-candidate max exact scenario net return over planned days 2..60; diagnostic only",
        "costs": "manifest-bound lots, commissions, stamp schedule, transfer fee, base/double slippage",
        "stateless": "daily cohort Top1/3/5/10 with policy-specific full candidate universe benchmark",
        "stateful": "daily frozen Top3, max3, one-third target allocation, cash allowed, no unfilled replacement",
        "training": False,
        "checkpoint_selection": False,
        "active_execution_change_allowed": False,
        "qdp_active_change_allowed": False,
    }
    contract_payload["contract_sha256"] = _canonical_sha256(contract_payload)
    contract_payload["created_at"] = _now()
    _write_json(output / "audit_contract.json", contract_payload)

    daily_rows: list[dict[str, Any]] = []
    stateful_rows: list[dict[str, Any]] = []
    stateful_daily_frames: list[pd.DataFrame] = []
    source_artifacts: list[dict[str, Any]] = []
    expected_date_counts: dict[tuple[str, int], int] = {}

    for year in normalized_years:
        print(f"[exit-audit] loading {year}", file=sys.stderr, flush=True)
        candidates = pack.candidates_for_year(year)
        candidate_date_count = int(candidates["trade_date"].nunique())
        topk_by_profile: dict[str, pd.DataFrame] = {}
        for profile in normalized_profiles:
            topk, source_daily, run_dir = _load_source_evidence(source_study_root, profile, year)
            topk_by_profile[profile] = topk
            predicted_rows = _predicted_policy_rows(
                profile=profile,
                year=year,
                daily=source_daily,
            )
            expected_predicted_rows = candidate_date_count * len(TOP_K_VALUES) * 2
            if len(predicted_rows) != expected_predicted_rows:
                raise ValueError(
                    f"predicted-policy evidence is incomplete for {profile}/{year}: "
                    f"{len(predicted_rows)} != {expected_predicted_rows}"
                )
            daily_rows.extend(predicted_rows)
            expected_date_counts[(profile, int(year))] = candidate_date_count
            source_artifacts.append(
                {
                    "profile": profile,
                    "development_year": int(year),
                    "run_dir": str(run_dir.resolve()),
                    "topk_candidates_sha256": _sha256_file(run_dir / "topk_candidates.parquet"),
                    "daily_topk_metrics_sha256": _sha256_file(run_dir / "daily_topk_metrics.csv"),
                }
            )

        for trade_date, date_candidates in candidates.groupby("trade_date", sort=True):
            date_candidates = date_candidates.reset_index(drop=True)
            date_indices = date_candidates["date_idx"].astype(int).unique()
            if len(date_indices) != 1:
                raise ValueError(f"candidate date_idx is not unique for {trade_date}")
            date_idx = int(date_indices[0])
            symbols = date_candidates["symbol"].astype(str).tolist()
            symbol_idx = date_candidates["symbol_idx"].to_numpy(dtype=np.int64)
            universe_hash = _universe_hash(symbols)
            entry_filled = date_candidates["entry_filled"].to_numpy(dtype=bool)
            entry_prices, exit_prices, exit_sellable = pack.execution_paths(date_idx, symbol_idx)
            next_valid = _next_valid_exit_indices(exit_prices, exit_sellable)
            fixed_plans = {
                horizon: resolve_planned_exit_batch(
                    signal_date_idx=date_idx,
                    entry_filled=entry_filled,
                    entry_prices=entry_prices,
                    exit_prices=exit_prices,
                    next_valid_exit_idx=next_valid,
                    planned_days=horizon,
                    forward_days=pack.forward_days,
                    execution_days=pack.execution_days,
                    terminal_recovery_fraction=pack.terminal_recovery_fraction,
                )
                for horizon in FIXED_HORIZONS
            }
            offset_by_symbol = {symbol: offset for offset, symbol in enumerate(symbols)}
            ranked_offsets: dict[str, np.ndarray] = {}
            ranked_symbols: dict[str, list[str]] = {}
            for profile in normalized_profiles:
                date_topk = topk_by_profile[profile]
                date_topk = date_topk[date_topk["trade_date"].eq(str(trade_date))].sort_values(
                    "score_rank", kind="mergesort"
                )
                if date_topk.empty:
                    raise ValueError(f"source TopK is missing {profile}/{trade_date}")
                source_counts = date_topk["universe_count"].astype(int).unique()
                source_hashes = date_topk["universe_hash"].astype(str).unique()
                if len(source_counts) != 1 or int(source_counts[0]) != len(symbols):
                    raise ValueError(f"candidate count identity drift for {profile}/{trade_date}")
                if len(source_hashes) != 1 or str(source_hashes[0]) != universe_hash:
                    raise ValueError(f"candidate universe hash drift for {profile}/{trade_date}")
                selected_symbols = date_topk["symbol"].astype(str).tolist()
                missing = [symbol for symbol in selected_symbols if symbol not in offset_by_symbol]
                if missing:
                    raise ValueError(f"frozen TopK symbols are absent from candidate index: {missing[:5]}")
                ranked_offsets[profile] = np.asarray(
                    [offset_by_symbol[symbol] for symbol in selected_symbols], dtype=np.int64
                )
                ranked_symbols[profile] = selected_symbols

            for top_k in TOP_K_VALUES:
                allocation = float(DEFAULT_DAILY_COHORT_CASH_CNY) / int(top_k)
                for scenario, multiplier in (
                    ("base", 1.0),
                    ("stress", pack.contract.stress_slippage_multiplier),
                ):
                    outcomes: dict[str, tuple[ResolvedPlanBatch, CashflowBatch]] = {}
                    for horizon, plan in fixed_plans.items():
                        outcomes[_policy_for_horizon(horizon)] = (
                            plan,
                            cashflow_batch(
                                allocated_cash=allocation,
                                entry_filled=entry_filled,
                                entry_prices=entry_prices,
                                plan=plan,
                                date_values=pack.date_values,
                                contract=pack.contract,
                                slippage_multiplier=multiplier,
                            ),
                        )
                    outcomes[ORACLE_POLICY] = oracle_executable_outcome_batch(
                        signal_date_idx=date_idx,
                        allocated_cash=allocation,
                        entry_filled=entry_filled,
                        entry_prices=entry_prices,
                        exit_prices=exit_prices,
                        next_valid_exit_idx=next_valid,
                        date_values=pack.date_values,
                        contract=pack.contract,
                        slippage_multiplier=multiplier,
                        forward_days=pack.forward_days,
                        execution_days=pack.execution_days,
                        terminal_recovery_fraction=pack.terminal_recovery_fraction,
                    )
                    oracle_returns = outcomes[ORACLE_POLICY][1].net_return
                    for horizon in FIXED_HORIZONS:
                        fixed_returns = outcomes[_policy_for_horizon(horizon)][1].net_return
                        if bool(np.any(oracle_returns + 1.0e-12 < fixed_returns)):
                            raise AssertionError(
                                f"executable oracle failed to dominate fixed_h{horizon} "
                                f"for {trade_date}/Top{top_k}/{scenario}"
                            )
                    for profile in normalized_profiles:
                        selected = ranked_offsets[profile][: int(top_k)]
                        selected_symbols = ranked_symbols[profile][: int(top_k)]
                        for policy, (plan, cashflow) in outcomes.items():
                            daily_rows.append(
                                _metric_row(
                                    profile=profile,
                                    year=year,
                                    trade_date=str(trade_date),
                                    policy=policy,
                                    top_k=top_k,
                                    cost_scenario=scenario,
                                    selected_idx=selected,
                                    selected_symbols=selected_symbols,
                                    universe_hash=universe_hash,
                                    plan=plan,
                                    cash=cashflow,
                                )
                            )

        if include_stateful:
            policies = [
                *[_policy_for_horizon(value) for value in FIXED_HORIZONS],
                PREDICTED_POLICY,
                ORACLE_POLICY,
            ]
            for profile in normalized_profiles:
                for policy in policies:
                    for scenario in ("base", "stress"):
                        metric, equity = _simulate_stateful_top3(
                            pack=pack,
                            topk=topk_by_profile[profile],
                            profile=profile,
                            year=year,
                            policy=policy,
                            cost_scenario=scenario,
                        )
                        stateful_rows.append(metric)
                        stateful_daily_frames.append(equity)
        print(f"[exit-audit] completed {year}", file=sys.stderr, flush=True)

    daily = pd.DataFrame(daily_rows).sort_values(
        ["profile", "development_year", "policy", "top_k", "cost_scenario", "trade_date"],
        kind="mergesort",
    ).reset_index(drop=True)
    daily_key = [
        "profile",
        "development_year",
        "trade_date",
        "policy",
        "top_k",
        "cost_scenario",
    ]
    if bool(daily.duplicated(daily_key).any()):
        raise AssertionError("daily policy evidence contains duplicate audit keys")
    coverage = daily.groupby(
        ["profile", "development_year", "policy", "top_k", "cost_scenario"],
        sort=True,
    )["trade_date"].nunique()
    expected_group_count = (
        len(normalized_profiles)
        * len(normalized_years)
        * (len(FIXED_HORIZONS) + 2)
        * len(TOP_K_VALUES)
        * 2
    )
    if len(coverage) != expected_group_count:
        raise AssertionError(
            f"daily policy coverage has {len(coverage)} groups; expected {expected_group_count}"
        )
    for key, observed_dates in coverage.items():
        profile, year = str(key[0]), int(key[1])
        expected_dates = expected_date_counts[(profile, year)]
        if int(observed_dates) != expected_dates:
            raise AssertionError(
                f"daily policy coverage drift for {key}: {observed_dates} != {expected_dates}"
            )
    yearly, equal_year = _aggregate_daily_metrics(daily)
    _write_csv(output / "daily_policy_metrics.csv", daily)
    _write_csv(output / "year_policy_metrics.csv", yearly)
    _write_csv(output / "equal_year_policy_summary.csv", equal_year)

    stateful_metrics = pd.DataFrame(stateful_rows)
    stateful_equal = pd.DataFrame()
    if include_stateful:
        stateful_metrics = stateful_metrics.sort_values(
            ["profile", "development_year", "policy", "cost_scenario"], kind="mergesort"
        ).reset_index(drop=True)
        stateful_equal = _stateful_equal_year(stateful_metrics)
        stateful_daily = pd.concat(stateful_daily_frames, ignore_index=True).sort_values(
            ["profile", "development_year", "policy", "cost_scenario", "date_idx"],
            kind="mergesort",
        )
        _write_csv(output / "stateful_portfolio_metrics.csv", stateful_metrics)
        _write_csv(output / "stateful_equal_year_summary.csv", stateful_equal)
        _write_csv(output / "stateful_daily_equity.csv", stateful_daily)

    qdp_active_after = _sha256_file(QDP_ACTIVE_PATH)
    active_execution_after = active_execution_path.exists()
    active_execution_sha256_after = (
        _sha256_file(active_execution_path) if active_execution_after else None
    )
    if qdp_active_after != qdp_active_before:
        raise RuntimeError("QDP active manifest changed during a read-only audit")
    if (
        active_execution_after != active_execution_before
        or active_execution_sha256_after != active_execution_sha256_before
    ):
        raise RuntimeError("active execution state changed during a read-only audit")

    summary = {
        "schema_version": 1,
        "status": "completed",
        "audit_id": contract_payload["audit_id"],
        "contract_sha256": contract_payload["contract_sha256"],
        "created_at": contract_payload["created_at"],
        "completed_at": _now(),
        "training_performed": False,
        "source_winner": None,
        "profiles": list(normalized_profiles),
        "development_years": list(normalized_years),
        "daily_metric_row_count": int(len(daily)),
        "year_metric_row_count": int(len(yearly)),
        "equal_year_metric_row_count": int(len(equal_year)),
        "stateful_metric_row_count": int(len(stateful_metrics)),
        "source_artifacts": source_artifacts,
        "artifacts": {
            "contract": str((output / "audit_contract.json").resolve()),
            "daily_policy_metrics": str((output / "daily_policy_metrics.csv").resolve()),
            "year_policy_metrics": str((output / "year_policy_metrics.csv").resolve()),
            "equal_year_policy_summary": str((output / "equal_year_policy_summary.csv").resolve()),
            "stateful_portfolio_metrics": str((output / "stateful_portfolio_metrics.csv").resolve()) if include_stateful else None,
            "stateful_equal_year_summary": str((output / "stateful_equal_year_summary.csv").resolve()) if include_stateful else None,
            "stateful_daily_equity": str((output / "stateful_daily_equity.csv").resolve()) if include_stateful else None,
        },
        "protected_objects": {
            "qdp_active_sha256_before": qdp_active_before.upper(),
            "qdp_active_sha256_after": qdp_active_after.upper(),
            "qdp_active_changed": False,
            "active_execution_present_before": active_execution_before,
            "active_execution_present_after": active_execution_after,
            "active_execution_sha256_before": active_execution_sha256_before,
            "active_execution_sha256_after": active_execution_sha256_after,
            "active_execution_changed": False,
        },
    }
    _write_json(output / "study_summary.json", summary)
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Zero-training frozen-ranking exit-policy audit for candidate-complete Seq100 v3."
    )
    parser.add_argument("--source-study-root", type=Path, default=SOURCE_STUDY_ROOT)
    parser.add_argument("--pack-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--years", type=int, nargs="+", default=list(YEARS))
    parser.add_argument("--profiles", nargs="+", default=list(PROFILES), choices=list(PROFILES))
    parser.add_argument("--skip-stateful", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = run_exit_policy_audit(
        source_study_root=args.source_study_root,
        pack_manifest=args.pack_manifest,
        output_root=args.output_root,
        years=args.years,
        profiles=args.profiles,
        include_stateful=not bool(args.skip_stateful),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CandidateCompleteAuditPack",
    "CashflowBatch",
    "ResolvedPlanBatch",
    "cashflow_batch",
    "oracle_executable_outcome_batch",
    "resolve_planned_exit_batch",
    "run_exit_policy_audit",
]
