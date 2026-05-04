from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from daily_research.continuous_policy.allocation_optimizer import (
    AllocationOptimizerConstraints,
    solve_semidifferentiable_allocation,
)


DEFAULT_MAX_POSITIONS = 8
DEFAULT_MAX_POSITION_WEIGHT = 0.20
DEFAULT_TURNOVER_LIMIT = 0.60
DEFAULT_EXECUTION_DEADBAND_ABS = 0.0010
DEFAULT_EXECUTION_DEADBAND_REL = 0.04
EXECUTION_SEMANTICS_LEGACY = "legacy_weight_derived"
EXECUTION_SEMANTICS_SEMANTIC = "semantic_preserving_v1"
DEFAULT_EXECUTION_SEMANTICS = EXECUTION_SEMANTICS_SEMANTIC
EXECUTION_SEMANTICS_CHOICES = (EXECUTION_SEMANTICS_LEGACY, EXECUTION_SEMANTICS_SEMANTIC)
BUDGET_SEMANTICS_LEGACY = "legacy_total_candidate"
BUDGET_SEMANTICS_SPLIT = "action_budget_split_v1"
BUDGET_SEMANTICS_ALLOCATION_LAYER = "allocation_layer_v1"
DEFAULT_BUDGET_SEMANTICS = BUDGET_SEMANTICS_LEGACY
BUDGET_SEMANTICS_CHOICES = (BUDGET_SEMANTICS_LEGACY, BUDGET_SEMANTICS_SPLIT, BUDGET_SEMANTICS_ALLOCATION_LAYER)
BUDGET_CALIBRATION_NONE = "none"
BUDGET_CALIBRATION_CASH_EXIT = "cash_exit_guard_v1"
BUDGET_CALIBRATION_CASH_TRANSLATION = "cash_translation_guard_v2"
BUDGET_CALIBRATION_CASH_TRANSLATION_SELL = "cash_translation_sell_guard_v3"
BUDGET_CALIBRATION_CASH_CONSTRAINT = "cash_constraint_guard_v4"
BUDGET_CALIBRATION_CASH_CONSTRAINT_INTENT = "cash_constraint_intent_guard_v5"
BUDGET_CALIBRATION_CASH_CONSTRAINT_DEPLOY = "cash_constraint_deploy_guard_v6"
BUDGET_CALIBRATION_CASH_CONSTRAINT_SELL_SOURCE = "cash_constraint_sell_source_guard_v7"
BUDGET_CALIBRATION_CASH_CONSTRAINT_DIRECT_ACTION = "cash_constraint_direct_action_guard_v8"
BUDGET_CALIBRATION_CASH_CONSTRAINT_DIRECT_ACTION_REALLOCATION = "cash_constraint_direct_action_reallocation_guard_v9"
BUDGET_CALIBRATION_CASH_CONSTRAINT_DIRECT_ACTION_PAIR_REALLOCATION = "cash_constraint_direct_action_pair_reallocation_guard_v10"
BUDGET_CALIBRATION_CASH_CONSTRAINT_DIRECT_ACTION_PAIR_COST_GUARD = "cash_constraint_direct_action_pair_cost_guard_v11"
BUDGET_CALIBRATION_CASH_CONSTRAINT_PORTFOLIO_DAILY_RANKING = "cash_constraint_portfolio_daily_ranking_guard_v12"
BUDGET_CALIBRATION_CASH_CONSTRAINT_PORTFOLIO_DAILY_RANKING_CASH_AWARE = (
    "cash_constraint_portfolio_daily_ranking_cash_aware_guard_v13"
)
BUDGET_CALIBRATION_CASH_CONSTRAINT_PORTFOLIO_DAILY_RANKING_SOURCE_EXEC = (
    "cash_constraint_portfolio_daily_ranking_source_exec_guard_v14"
)
BUDGET_CALIBRATION_CASH_CONSTRAINT_PORTFOLIO_DAILY_RANKING_RECEIVER_EXEC = (
    "cash_constraint_portfolio_daily_ranking_receiver_exec_guard_v15"
)
BUDGET_CALIBRATION_END_TO_END_ALLOCATION_LAYER = "end_to_end_allocation_layer_v1"
BUDGET_CALIBRATION_PORTFOLIO_DAILY_RANKING_SET = (
    BUDGET_CALIBRATION_CASH_CONSTRAINT_PORTFOLIO_DAILY_RANKING,
    BUDGET_CALIBRATION_CASH_CONSTRAINT_PORTFOLIO_DAILY_RANKING_CASH_AWARE,
    BUDGET_CALIBRATION_CASH_CONSTRAINT_PORTFOLIO_DAILY_RANKING_SOURCE_EXEC,
    BUDGET_CALIBRATION_CASH_CONSTRAINT_PORTFOLIO_DAILY_RANKING_RECEIVER_EXEC,
)
DEFAULT_BUDGET_CALIBRATION = BUDGET_CALIBRATION_NONE
BUDGET_CALIBRATION_CHOICES = (
    BUDGET_CALIBRATION_NONE,
    BUDGET_CALIBRATION_CASH_EXIT,
    BUDGET_CALIBRATION_CASH_TRANSLATION,
    BUDGET_CALIBRATION_CASH_TRANSLATION_SELL,
    BUDGET_CALIBRATION_CASH_CONSTRAINT,
    BUDGET_CALIBRATION_CASH_CONSTRAINT_INTENT,
    BUDGET_CALIBRATION_CASH_CONSTRAINT_DEPLOY,
    BUDGET_CALIBRATION_CASH_CONSTRAINT_SELL_SOURCE,
    BUDGET_CALIBRATION_CASH_CONSTRAINT_DIRECT_ACTION,
    BUDGET_CALIBRATION_CASH_CONSTRAINT_DIRECT_ACTION_REALLOCATION,
    BUDGET_CALIBRATION_CASH_CONSTRAINT_DIRECT_ACTION_PAIR_REALLOCATION,
    BUDGET_CALIBRATION_CASH_CONSTRAINT_DIRECT_ACTION_PAIR_COST_GUARD,
    BUDGET_CALIBRATION_CASH_CONSTRAINT_PORTFOLIO_DAILY_RANKING,
    BUDGET_CALIBRATION_CASH_CONSTRAINT_PORTFOLIO_DAILY_RANKING_CASH_AWARE,
    BUDGET_CALIBRATION_CASH_CONSTRAINT_PORTFOLIO_DAILY_RANKING_SOURCE_EXEC,
    BUDGET_CALIBRATION_CASH_CONSTRAINT_PORTFOLIO_DAILY_RANKING_RECEIVER_EXEC,
    BUDGET_CALIBRATION_END_TO_END_ALLOCATION_LAYER,
)


def normalize_execution_semantics(value: str | None) -> str:
    text = str(value or DEFAULT_EXECUTION_SEMANTICS).strip().lower()
    aliases = {
        "legacy": EXECUTION_SEMANTICS_LEGACY,
        "legacy_weight": EXECUTION_SEMANTICS_LEGACY,
        "legacy_weight_derived": EXECUTION_SEMANTICS_LEGACY,
        "weight_derived": EXECUTION_SEMANTICS_LEGACY,
        "semantic": EXECUTION_SEMANTICS_SEMANTIC,
        "semantic_preserving": EXECUTION_SEMANTICS_SEMANTIC,
        "semantic_preserving_v1": EXECUTION_SEMANTICS_SEMANTIC,
    }
    if text not in aliases:
        raise ValueError(
            f"Unsupported execution semantics: {value!r}. "
            f"Available: {', '.join(EXECUTION_SEMANTICS_CHOICES)}"
        )
    return aliases[text]


def normalize_budget_semantics(value: str | None) -> str:
    text = str(value or DEFAULT_BUDGET_SEMANTICS).strip().lower()
    aliases = {
        "legacy": BUDGET_SEMANTICS_LEGACY,
        "legacy_total": BUDGET_SEMANTICS_LEGACY,
        "legacy_total_candidate": BUDGET_SEMANTICS_LEGACY,
        "total_candidate": BUDGET_SEMANTICS_LEGACY,
        "split": BUDGET_SEMANTICS_SPLIT,
        "action_split": BUDGET_SEMANTICS_SPLIT,
        "action_budget_split": BUDGET_SEMANTICS_SPLIT,
        "action_budget_split_v1": BUDGET_SEMANTICS_SPLIT,
        "allocation": BUDGET_SEMANTICS_ALLOCATION_LAYER,
        "allocation_layer": BUDGET_SEMANTICS_ALLOCATION_LAYER,
        "allocation_layer_v1": BUDGET_SEMANTICS_ALLOCATION_LAYER,
    }
    if text not in aliases:
        raise ValueError(
            f"Unsupported budget semantics: {value!r}. "
            f"Available: {', '.join(BUDGET_SEMANTICS_CHOICES)}"
        )
    return aliases[text]


def normalize_budget_calibration(value: str | None) -> str:
    text = str(value or DEFAULT_BUDGET_CALIBRATION).strip().lower()
    aliases = {
        "": BUDGET_CALIBRATION_NONE,
        "none": BUDGET_CALIBRATION_NONE,
        "off": BUDGET_CALIBRATION_NONE,
        "cash_exit": BUDGET_CALIBRATION_CASH_EXIT,
        "cash_exit_guard": BUDGET_CALIBRATION_CASH_EXIT,
        "cash_exit_guard_v1": BUDGET_CALIBRATION_CASH_EXIT,
        "cash_translation": BUDGET_CALIBRATION_CASH_TRANSLATION,
        "cash_translation_guard": BUDGET_CALIBRATION_CASH_TRANSLATION,
        "cash_translation_guard_v2": BUDGET_CALIBRATION_CASH_TRANSLATION,
        "cash_translation_sell": BUDGET_CALIBRATION_CASH_TRANSLATION_SELL,
        "cash_translation_sell_guard": BUDGET_CALIBRATION_CASH_TRANSLATION_SELL,
        "cash_translation_sell_guard_v3": BUDGET_CALIBRATION_CASH_TRANSLATION_SELL,
        "cash_translation_guard_v3": BUDGET_CALIBRATION_CASH_TRANSLATION_SELL,
        "sell_attribution_guard": BUDGET_CALIBRATION_CASH_TRANSLATION_SELL,
        "cash_constraint": BUDGET_CALIBRATION_CASH_CONSTRAINT,
        "cash_constraint_guard": BUDGET_CALIBRATION_CASH_CONSTRAINT,
        "cash_constraint_guard_v4": BUDGET_CALIBRATION_CASH_CONSTRAINT,
        "constraint_only_guard": BUDGET_CALIBRATION_CASH_CONSTRAINT,
        "cash_constraint_intent": BUDGET_CALIBRATION_CASH_CONSTRAINT_INTENT,
        "cash_constraint_intent_guard": BUDGET_CALIBRATION_CASH_CONSTRAINT_INTENT,
        "cash_constraint_intent_guard_v5": BUDGET_CALIBRATION_CASH_CONSTRAINT_INTENT,
        "intent_preserving_constraint": BUDGET_CALIBRATION_CASH_CONSTRAINT_INTENT,
        "cash_constraint_deploy": BUDGET_CALIBRATION_CASH_CONSTRAINT_DEPLOY,
        "cash_constraint_deploy_guard": BUDGET_CALIBRATION_CASH_CONSTRAINT_DEPLOY,
        "cash_constraint_deploy_guard_v6": BUDGET_CALIBRATION_CASH_CONSTRAINT_DEPLOY,
        "deploy_executability_constraint": BUDGET_CALIBRATION_CASH_CONSTRAINT_DEPLOY,
        "cash_constraint_sell_source": BUDGET_CALIBRATION_CASH_CONSTRAINT_SELL_SOURCE,
        "cash_constraint_sell_source_guard": BUDGET_CALIBRATION_CASH_CONSTRAINT_SELL_SOURCE,
        "cash_constraint_sell_source_guard_v7": BUDGET_CALIBRATION_CASH_CONSTRAINT_SELL_SOURCE,
        "sell_source_decoupled_constraint": BUDGET_CALIBRATION_CASH_CONSTRAINT_SELL_SOURCE,
        "cash_constraint_direct_action": BUDGET_CALIBRATION_CASH_CONSTRAINT_DIRECT_ACTION,
        "cash_constraint_direct_action_guard": BUDGET_CALIBRATION_CASH_CONSTRAINT_DIRECT_ACTION,
        "cash_constraint_direct_action_guard_v8": BUDGET_CALIBRATION_CASH_CONSTRAINT_DIRECT_ACTION,
        "direct_action_preserving_constraint": BUDGET_CALIBRATION_CASH_CONSTRAINT_DIRECT_ACTION,
        "cash_constraint_direct_action_reallocation": BUDGET_CALIBRATION_CASH_CONSTRAINT_DIRECT_ACTION_REALLOCATION,
        "cash_constraint_direct_action_reallocation_guard": BUDGET_CALIBRATION_CASH_CONSTRAINT_DIRECT_ACTION_REALLOCATION,
        "cash_constraint_direct_action_reallocation_guard_v9": BUDGET_CALIBRATION_CASH_CONSTRAINT_DIRECT_ACTION_REALLOCATION,
        "direct_action_reallocation_constraint": BUDGET_CALIBRATION_CASH_CONSTRAINT_DIRECT_ACTION_REALLOCATION,
        "cash_constraint_direct_action_pair_reallocation": BUDGET_CALIBRATION_CASH_CONSTRAINT_DIRECT_ACTION_PAIR_REALLOCATION,
        "cash_constraint_direct_action_pair_reallocation_guard": BUDGET_CALIBRATION_CASH_CONSTRAINT_DIRECT_ACTION_PAIR_REALLOCATION,
        "cash_constraint_direct_action_pair_reallocation_guard_v10": BUDGET_CALIBRATION_CASH_CONSTRAINT_DIRECT_ACTION_PAIR_REALLOCATION,
        "direct_action_pair_reallocation_constraint": BUDGET_CALIBRATION_CASH_CONSTRAINT_DIRECT_ACTION_PAIR_REALLOCATION,
        "cash_constraint_direct_action_pair_cost_guard": BUDGET_CALIBRATION_CASH_CONSTRAINT_DIRECT_ACTION_PAIR_COST_GUARD,
        "cash_constraint_direct_action_pair_cost_guard_v11": BUDGET_CALIBRATION_CASH_CONSTRAINT_DIRECT_ACTION_PAIR_COST_GUARD,
        "cash_constraint_direct_action_pair_reallocation_cost_guard": BUDGET_CALIBRATION_CASH_CONSTRAINT_DIRECT_ACTION_PAIR_COST_GUARD,
        "cash_constraint_direct_action_pair_reallocation_cost_guard_v11": BUDGET_CALIBRATION_CASH_CONSTRAINT_DIRECT_ACTION_PAIR_COST_GUARD,
        "direct_action_pair_cost_guard_constraint": BUDGET_CALIBRATION_CASH_CONSTRAINT_DIRECT_ACTION_PAIR_COST_GUARD,
        "cash_constraint_portfolio_daily_ranking": BUDGET_CALIBRATION_CASH_CONSTRAINT_PORTFOLIO_DAILY_RANKING,
        "cash_constraint_portfolio_daily_ranking_guard": BUDGET_CALIBRATION_CASH_CONSTRAINT_PORTFOLIO_DAILY_RANKING,
        "cash_constraint_portfolio_daily_ranking_guard_v12": BUDGET_CALIBRATION_CASH_CONSTRAINT_PORTFOLIO_DAILY_RANKING,
        "portfolio_daily_ranking_constraint": BUDGET_CALIBRATION_CASH_CONSTRAINT_PORTFOLIO_DAILY_RANKING,
        "portfolio_daily_ranking_guard_v12": BUDGET_CALIBRATION_CASH_CONSTRAINT_PORTFOLIO_DAILY_RANKING,
        "cash_constraint_portfolio_daily_ranking_cash_aware": BUDGET_CALIBRATION_CASH_CONSTRAINT_PORTFOLIO_DAILY_RANKING_CASH_AWARE,
        "cash_constraint_portfolio_daily_ranking_cash_aware_guard": BUDGET_CALIBRATION_CASH_CONSTRAINT_PORTFOLIO_DAILY_RANKING_CASH_AWARE,
        "cash_constraint_portfolio_daily_ranking_cash_aware_guard_v13": BUDGET_CALIBRATION_CASH_CONSTRAINT_PORTFOLIO_DAILY_RANKING_CASH_AWARE,
        "portfolio_daily_ranking_cash_aware": BUDGET_CALIBRATION_CASH_CONSTRAINT_PORTFOLIO_DAILY_RANKING_CASH_AWARE,
        "portfolio_daily_ranking_cash_aware_guard_v13": BUDGET_CALIBRATION_CASH_CONSTRAINT_PORTFOLIO_DAILY_RANKING_CASH_AWARE,
        "cash_constraint_portfolio_daily_ranking_source_exec": BUDGET_CALIBRATION_CASH_CONSTRAINT_PORTFOLIO_DAILY_RANKING_SOURCE_EXEC,
        "cash_constraint_portfolio_daily_ranking_source_exec_guard": BUDGET_CALIBRATION_CASH_CONSTRAINT_PORTFOLIO_DAILY_RANKING_SOURCE_EXEC,
        "cash_constraint_portfolio_daily_ranking_source_exec_guard_v14": BUDGET_CALIBRATION_CASH_CONSTRAINT_PORTFOLIO_DAILY_RANKING_SOURCE_EXEC,
        "portfolio_daily_ranking_source_exec": BUDGET_CALIBRATION_CASH_CONSTRAINT_PORTFOLIO_DAILY_RANKING_SOURCE_EXEC,
        "portfolio_daily_ranking_source_exec_guard_v14": BUDGET_CALIBRATION_CASH_CONSTRAINT_PORTFOLIO_DAILY_RANKING_SOURCE_EXEC,
        "cash_constraint_portfolio_daily_ranking_receiver_exec": BUDGET_CALIBRATION_CASH_CONSTRAINT_PORTFOLIO_DAILY_RANKING_RECEIVER_EXEC,
        "cash_constraint_portfolio_daily_ranking_receiver_exec_guard": BUDGET_CALIBRATION_CASH_CONSTRAINT_PORTFOLIO_DAILY_RANKING_RECEIVER_EXEC,
        "cash_constraint_portfolio_daily_ranking_receiver_exec_guard_v15": BUDGET_CALIBRATION_CASH_CONSTRAINT_PORTFOLIO_DAILY_RANKING_RECEIVER_EXEC,
        "portfolio_daily_ranking_receiver_exec": BUDGET_CALIBRATION_CASH_CONSTRAINT_PORTFOLIO_DAILY_RANKING_RECEIVER_EXEC,
        "portfolio_daily_ranking_receiver_exec_guard_v15": BUDGET_CALIBRATION_CASH_CONSTRAINT_PORTFOLIO_DAILY_RANKING_RECEIVER_EXEC,
        "allocation_layer": BUDGET_CALIBRATION_END_TO_END_ALLOCATION_LAYER,
        "end_to_end_allocation_layer": BUDGET_CALIBRATION_END_TO_END_ALLOCATION_LAYER,
        "end_to_end_allocation_layer_v1": BUDGET_CALIBRATION_END_TO_END_ALLOCATION_LAYER,
    }
    if text not in aliases:
        raise ValueError(
            f"Unsupported budget calibration: {value!r}. "
            f"Available: {', '.join(BUDGET_CALIBRATION_CHOICES)}"
        )
    return aliases[text]


def _weight_change_action(previous_weight: float, new_weight: float) -> str:
    if previous_weight <= 1e-8 and new_weight <= 1e-8:
        return "skip"
    if previous_weight <= 1e-8 and new_weight > 1e-8:
        return "open"
    if previous_weight > 1e-8 and new_weight <= 1e-8:
        return "exit"
    if new_weight > previous_weight + 1e-8:
        return "add"
    if new_weight < previous_weight - 1e-8:
        return "reduce"
    return "hold"


def _semantic_execution_action(
    *,
    model_action: str,
    previous_weight: float,
    new_weight: float,
    weight_action: str,
) -> tuple[str, str]:
    model_name = str(model_action or "skip").strip().lower()
    if model_name not in {"skip", "open", "hold", "add", "reduce", "exit"}:
        return weight_action, "unknown_model_action_fallback"
    if previous_weight <= 1e-8 and new_weight <= 1e-8:
        return "skip", "flat_no_order"
    if previous_weight <= 1e-8:
        if model_name in {"open", "add"}:
            return "open", "entry_intent_preserved"
        return weight_action, "entry_weight_change_fallback"
    if model_name == "skip":
        return "hold" if new_weight > 1e-8 else weight_action, "held_skip_translated_to_hold"
    if model_name == "open":
        return "add" if new_weight > 1e-8 else weight_action, "held_open_translated_to_add"
    if model_name in {"hold", "add", "reduce", "exit"}:
        return model_name, "model_lifecycle_intent_preserved"
    return weight_action, "weight_change_fallback"


@dataclass
class HoldingState:
    weight: float
    entry_price: float
    peak_price: float
    hold_days: int = 0


@dataclass
class StepResult:
    date: str
    weights: pd.Series
    actions: list[dict[str, Any]]
    diagnostics: dict[str, Any]


@dataclass
class PortfolioState:
    cash_weight: float = 1.0
    max_positions: int = DEFAULT_MAX_POSITIONS
    max_position_weight: float = DEFAULT_MAX_POSITION_WEIGHT
    turnover_limit: float = DEFAULT_TURNOVER_LIMIT
    holdings: dict[str, HoldingState] = field(default_factory=dict)
    last_buy_dates: dict[str, str] = field(default_factory=dict)
    last_sell_dates: dict[str, str] = field(default_factory=dict)
    last_reduce_dates: dict[str, str] = field(default_factory=dict)
    last_exit_dates: dict[str, str] = field(default_factory=dict)
    last_action_labels: dict[str, str] = field(default_factory=dict)
    recent_action_events: list[dict[str, Any]] = field(default_factory=list)
    recent_turnovers: list[float] = field(default_factory=list)
    recent_returns: list[float] = field(default_factory=list)
    recent_cash_weights: list[float] = field(default_factory=list)
    last_signal_date: str = ""

    def clone(self) -> "PortfolioState":
        return PortfolioState.from_snapshot(self.snapshot())

    def weight_map(self) -> dict[str, float]:
        return {stock: float(state.weight) for stock, state in self.holdings.items()}

    def entry_price_map(self) -> dict[str, float]:
        return {stock: float(state.entry_price) for stock, state in self.holdings.items()}

    def peak_price_map(self) -> dict[str, float]:
        return {stock: float(state.peak_price) for stock, state in self.holdings.items()}

    def hold_days_map(self) -> dict[str, int]:
        return {stock: int(state.hold_days) for stock, state in self.holdings.items()}

    def last_buy_date_map(self) -> dict[str, str]:
        return {str(stock): str(value or "") for stock, value in self.last_buy_dates.items()}

    def last_sell_date_map(self) -> dict[str, str]:
        return {str(stock): str(value or "") for stock, value in self.last_sell_dates.items()}

    def last_reduce_date_map(self) -> dict[str, str]:
        return {str(stock): str(value or "") for stock, value in self.last_reduce_dates.items()}

    def last_exit_date_map(self) -> dict[str, str]:
        return {str(stock): str(value or "") for stock, value in self.last_exit_dates.items()}

    def last_action_label_map(self) -> dict[str, str]:
        return {str(stock): str(value or "") for stock, value in self.last_action_labels.items()}

    def current_weights(self, universe: list[str] | pd.Index) -> pd.Series:
        return pd.Series({stock: float(self.holdings.get(str(stock), HoldingState(0.0, 0.0, 0.0)).weight) for stock in universe}, dtype=float)

    def portfolio_features(self) -> dict[str, float]:
        gross_exposure = float(sum(max(float(item.weight), 0.0) for item in self.holdings.values()))
        holding_count = int(sum(1 for item in self.holdings.values() if float(item.weight) > 1e-8))
        hhi = float(sum(float(item.weight) ** 2 for item in self.holdings.values()))
        hold_days = [float(item.hold_days) for item in self.holdings.values() if float(item.weight) > 1e-8]
        recent_returns = np.asarray(self.recent_returns[-20:], dtype=float) if self.recent_returns else np.asarray([], dtype=float)
        if recent_returns.size > 0:
            equity = np.cumprod(1.0 + recent_returns)
            drawdown_20d = float((equity / np.maximum.accumulate(equity) - 1.0).min())
            return_vol_20d = float(np.std(recent_returns, ddof=0))
        else:
            drawdown_20d = 0.0
            return_vol_20d = 0.0
        recent_cash_weights = self.recent_cash_weights[-5:] if self.recent_cash_weights else [float(self.cash_weight)]
        recent_cash_weights_20d = self.recent_cash_weights[-20:] if self.recent_cash_weights else [float(self.cash_weight)]
        recent_turnovers_20d = self.recent_turnovers[-20:] if self.recent_turnovers else [0.0]
        recent_turnover_5d = float(np.mean(self.recent_turnovers[-5:])) if self.recent_turnovers else 0.0
        recent_turnover_20d = float(np.mean(recent_turnovers_20d)) if recent_turnovers_20d else 0.0
        turnover_pressure = float(recent_turnover_5d / max(float(self.turnover_limit), 1e-6))
        recent_positive_share = float((recent_returns > 0).mean()) if recent_returns.size > 0 else 0.0
        recent_events = list(self.recent_action_events[-240:])
        recent_reversal_count = 0
        reversal_base = 0
        recent_reduce_count_10d = 0
        recent_exit_count_10d = 0
        recent_add_count_10d = 0
        recent_open_count_10d = 0
        if recent_events:
            events_by_stock: dict[str, list[dict[str, Any]]] = {}
            for event in recent_events:
                stock = str(event.get("stock", "") or "").strip().upper()
                if not stock:
                    continue
                events_by_stock.setdefault(stock, []).append(event)
                action = str(event.get("execution_action", "") or "").strip().lower()
                day_offset = int(event.get("days_ago", 999) or 999)
                if day_offset <= 10:
                    if action == "reduce":
                        recent_reduce_count_10d += 1
                    elif action == "exit":
                        recent_exit_count_10d += 1
                    elif action == "add":
                        recent_add_count_10d += 1
                    elif action == "open":
                        recent_open_count_10d += 1
            for stock_rows in events_by_stock.values():
                stock_rows = sorted(stock_rows, key=lambda item: (str(item.get("date", "")), str(item.get("stock", ""))))
                for idx in range(len(stock_rows) - 1):
                    current = stock_rows[idx]
                    nxt = stock_rows[idx + 1]
                    current_action = str(current.get("execution_action", "") or "").strip().lower()
                    next_action = str(nxt.get("execution_action", "") or "").strip().lower()
                    current_group = 1 if current_action in {"open", "add"} else -1 if current_action in {"reduce", "exit"} else 0
                    next_group = 1 if next_action in {"open", "add"} else -1 if next_action in {"reduce", "exit"} else 0
                    if current_group == 0 or next_group == 0:
                        continue
                    reversal_base += 1
                    try:
                        current_ts = pd.Timestamp(str(current.get("date", "") or ""))
                    except Exception:
                        current_ts = pd.NaT
                    try:
                        next_ts = pd.Timestamp(str(nxt.get("date", "") or ""))
                    except Exception:
                        next_ts = pd.NaT
                    if pd.notna(current_ts) and pd.notna(next_ts) and next_group != current_group and (next_ts - current_ts).days <= 3:
                        recent_reversal_count += 1
        recent_reversal_rate_20d = float(recent_reversal_count / reversal_base) if reversal_base > 0 else 0.0
        return {
            "cash_weight": float(self.cash_weight),
            "gross_exposure": gross_exposure,
            "holding_count": float(holding_count),
            "concentration_hhi": hhi,
            "recent_turnover_5d": recent_turnover_5d,
            "recent_turnover_20d": recent_turnover_20d,
            "recent_return_5d": float(np.sum(self.recent_returns[-5:])) if self.recent_returns else 0.0,
            "recent_return_20d": float(np.sum(self.recent_returns[-20:])) if self.recent_returns else 0.0,
            "average_hold_days": float(np.mean(hold_days)) if hold_days else 0.0,
            "cash_change_5d": float(recent_cash_weights[-1] - recent_cash_weights[0]) if len(recent_cash_weights) >= 2 else 0.0,
            "cash_weight_mean_20d": float(np.mean(recent_cash_weights_20d)) if recent_cash_weights_20d else float(self.cash_weight),
            "cash_weight_vol_20d": float(np.std(recent_cash_weights_20d, ddof=0)) if len(recent_cash_weights_20d) >= 2 else 0.0,
            "cash_deficit": float(max(0.0, 0.18 - float(self.cash_weight))),
            "turnover_pressure": turnover_pressure,
            "recent_positive_return_share_20d": recent_positive_share,
            "portfolio_drawdown_20d": drawdown_20d,
            "return_vol_20d": return_vol_20d,
            "recent_reversal_count_20d": float(recent_reversal_count),
            "recent_reversal_rate_20d": recent_reversal_rate_20d,
            "recent_reduce_count_10d": float(recent_reduce_count_10d),
            "recent_exit_count_10d": float(recent_exit_count_10d),
            "recent_add_count_10d": float(recent_add_count_10d),
            "recent_open_count_10d": float(recent_open_count_10d),
        }

    def record_realized_return(self, value: float) -> None:
        self.recent_returns.append(float(value))
        if len(self.recent_returns) > 60:
            self.recent_returns = self.recent_returns[-60:]

    def snapshot(self) -> dict[str, Any]:
        return {
            "cash_weight": float(self.cash_weight),
            "max_positions": int(self.max_positions),
            "max_position_weight": float(self.max_position_weight),
            "turnover_limit": float(self.turnover_limit),
            "holdings": {stock: asdict(state) for stock, state in sorted(self.holdings.items())},
            "last_buy_dates": {stock: str(value or "") for stock, value in sorted(self.last_buy_dates.items())},
            "last_sell_dates": {stock: str(value or "") for stock, value in sorted(self.last_sell_dates.items())},
            "last_reduce_dates": {stock: str(value or "") for stock, value in sorted(self.last_reduce_dates.items())},
            "last_exit_dates": {stock: str(value or "") for stock, value in sorted(self.last_exit_dates.items())},
            "last_action_labels": {stock: str(value or "") for stock, value in sorted(self.last_action_labels.items())},
            "recent_action_events": list(self.recent_action_events[-240:]),
            "recent_turnovers": [float(item) for item in self.recent_turnovers[-60:]],
            "recent_returns": [float(item) for item in self.recent_returns[-60:]],
            "recent_cash_weights": [float(item) for item in self.recent_cash_weights[-60:]],
            "last_signal_date": str(self.last_signal_date or ""),
        }

    @classmethod
    def from_snapshot(cls, payload: dict[str, Any] | None) -> "PortfolioState":
        data = payload if isinstance(payload, dict) else {}
        holdings_payload = data.get("holdings", {})
        holdings = {
            str(stock): HoldingState(
                weight=float((state or {}).get("weight", 0.0) or 0.0),
                entry_price=float((state or {}).get("entry_price", 0.0) or 0.0),
                peak_price=float((state or {}).get("peak_price", 0.0) or 0.0),
                hold_days=int((state or {}).get("hold_days", 0) or 0),
            )
            for stock, state in holdings_payload.items()
            if isinstance(state, dict) and float((state or {}).get("weight", 0.0) or 0.0) > 1e-8
        }
        return cls(
            cash_weight=float(data.get("cash_weight", 1.0) or 1.0),
            max_positions=int(data.get("max_positions", DEFAULT_MAX_POSITIONS) or DEFAULT_MAX_POSITIONS),
            max_position_weight=float(data.get("max_position_weight", DEFAULT_MAX_POSITION_WEIGHT) or DEFAULT_MAX_POSITION_WEIGHT),
            turnover_limit=float(data.get("turnover_limit", DEFAULT_TURNOVER_LIMIT) or DEFAULT_TURNOVER_LIMIT),
            holdings=holdings,
            last_buy_dates={str(stock): str(value or "") for stock, value in (data.get("last_buy_dates", {}) or {}).items()},
            last_sell_dates={str(stock): str(value or "") for stock, value in (data.get("last_sell_dates", {}) or {}).items()},
            last_reduce_dates={str(stock): str(value or "") for stock, value in (data.get("last_reduce_dates", {}) or {}).items()},
            last_exit_dates={str(stock): str(value or "") for stock, value in (data.get("last_exit_dates", {}) or {}).items()},
            last_action_labels={str(stock): str(value or "") for stock, value in (data.get("last_action_labels", {}) or {}).items()},
            recent_action_events=[
                {
                    "date": str((item or {}).get("date", "") or ""),
                    "stock": str((item or {}).get("stock", "") or "").strip().upper(),
                    "execution_action": str((item or {}).get("execution_action", "") or "").strip().lower(),
                    "days_ago": int((item or {}).get("days_ago", 999) or 999),
                }
                for item in (data.get("recent_action_events", []) or [])
                if isinstance(item, dict)
            ][-240:],
            recent_turnovers=[float(item) for item in data.get("recent_turnovers", [])][-60:],
            recent_returns=[float(item) for item in data.get("recent_returns", [])][-60:],
            recent_cash_weights=[float(item) for item in data.get("recent_cash_weights", [])][-60:],
            last_signal_date=str(data.get("last_signal_date", "") or ""),
        )

    @classmethod
    def from_account_snapshot(
        cls,
        *,
        account_snapshot: dict[str, Any],
        latest_prices: pd.Series,
        max_positions: int = DEFAULT_MAX_POSITIONS,
        max_position_weight: float = DEFAULT_MAX_POSITION_WEIGHT,
        turnover_limit: float = DEFAULT_TURNOVER_LIMIT,
    ) -> "PortfolioState":
        positions = account_snapshot.get("positions", []) if isinstance(account_snapshot, dict) else []
        cash_value = float(account_snapshot.get("available_cash", 0.0) or 0.0)
        position_values: dict[str, float] = {}
        entry_prices: dict[str, float] = {}
        for item in positions:
            stock = str((item or {}).get("stock", "") or "").strip().upper()
            shares = float((item or {}).get("shares", 0.0) or 0.0)
            if not stock or shares <= 0:
                continue
            market_price = float(latest_prices.get(stock, np.nan))
            if not np.isfinite(market_price) or market_price <= 0:
                market_price = float((item or {}).get("cost_price", 0.0) or 0.0)
            if market_price <= 0:
                continue
            position_values[stock] = shares * market_price
            entry_prices[stock] = float((item or {}).get("cost_price", market_price) or market_price)
        total_equity = cash_value + float(sum(position_values.values()))
        if total_equity <= 0:
            return cls(
                cash_weight=1.0,
                max_positions=max_positions,
                max_position_weight=max_position_weight,
                turnover_limit=turnover_limit,
            )

        holdings = {
            stock: HoldingState(
                weight=float(value / total_equity),
                entry_price=float(entry_prices.get(stock, 0.0) or 0.0),
                peak_price=float(max(latest_prices.get(stock, entry_prices.get(stock, 0.0)), entry_prices.get(stock, 0.0))),
                hold_days=0,
            )
            for stock, value in position_values.items()
            if float(value) > 0
        }
        gross = float(sum(item.weight for item in holdings.values()))
        cash_weight = max(0.0, 1.0 - gross)
        return cls(
            cash_weight=cash_weight,
            max_positions=max_positions,
            max_position_weight=max_position_weight,
            turnover_limit=turnover_limit,
            holdings=holdings,
        )

    def _allocate_with_cap(self, strengths: pd.Series, gross_exposure_target: float, *, position_cap: float | None = None) -> pd.Series:
        positive = strengths.clip(lower=0.0)
        if positive.sum() <= 0 or gross_exposure_target <= 0:
            return positive * 0.0
        normalized = positive / float(positive.sum())
        target = normalized * float(gross_exposure_target)
        cap = float(position_cap if position_cap is not None else self.max_position_weight)
        for _ in range(8):
            capped = target.clip(upper=cap)
            residual = float(gross_exposure_target) - float(capped.sum())
            if residual <= 1e-8:
                target = capped
                break
            free_mask = capped < cap - 1e-8
            if not bool(free_mask.any()):
                target = capped
                break
            free_strength = positive.where(free_mask, 0.0)
            if float(free_strength.sum()) <= 0:
                target = capped
                break
            target = capped + (free_strength / float(free_strength.sum())) * residual
        return target.clip(lower=0.0, upper=cap)

    def _shrink_to_target_by_priority(
        self,
        weights: pd.Series,
        *,
        target_total: float,
        floor: pd.Series,
        priority: pd.Series,
    ) -> tuple[pd.Series, pd.Series]:
        result = pd.to_numeric(weights, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(float).copy()
        floor_series = pd.to_numeric(floor.reindex(result.index), errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(float)
        priority_series = pd.to_numeric(priority.reindex(result.index), errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(float)
        guarded = pd.Series(False, index=result.index, dtype=bool)
        residual_excess = max(float(result.sum()) - float(target_total), 0.0)
        for _ in range(10):
            reducible = (result - floor_series).clip(lower=0.0)
            reducible_sum = float(reducible.sum())
            if residual_excess <= 1e-8 or reducible_sum <= 1e-8:
                break
            priority_mass = reducible * priority_series.clip(lower=0.0)
            if float(priority_mass.sum()) <= 1e-8:
                priority_mass = reducible
            step = (priority_mass / float(priority_mass.sum())) * residual_excess
            step = np.minimum(step.to_numpy(dtype=float), reducible.to_numpy(dtype=float))
            step_series = pd.Series(step, index=result.index, dtype=float)
            if float(step_series.sum()) <= 1e-8:
                break
            guarded = guarded | (step_series > 1e-8)
            result = (result - step_series).clip(lower=floor_series)
            residual_excess = max(float(result.sum()) - float(target_total), 0.0)
        return result, guarded

    def _lift_to_soft_floor_by_priority(
        self,
        weights: pd.Series,
        *,
        target_total: float,
        hard_floor: pd.Series,
        soft_floor: pd.Series,
        cap: pd.Series,
        priority: pd.Series,
    ) -> tuple[pd.Series, pd.Series]:
        result = pd.to_numeric(weights, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(float).copy()
        hard_floor_series = pd.to_numeric(hard_floor.reindex(result.index), errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(float)
        soft_floor_series = pd.to_numeric(soft_floor.reindex(result.index), errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(float)
        cap_series = pd.to_numeric(cap.reindex(result.index), errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(np.inf).astype(float)
        priority_series = pd.to_numeric(priority.reindex(result.index), errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(float)
        guarded = pd.Series(False, index=result.index, dtype=bool)
        result = result.clip(lower=hard_floor_series, upper=cap_series)
        residual_capacity = max(float(target_total) - float(result.sum()), 0.0)
        for _ in range(10):
            if residual_capacity <= 1e-8:
                break
            needed = (soft_floor_series - result).clip(lower=0.0)
            needed = np.minimum(needed.to_numpy(dtype=float), (cap_series - result).clip(lower=0.0).to_numpy(dtype=float))
            needed_series = pd.Series(needed, index=result.index, dtype=float)
            needed_sum = float(needed_series.sum())
            if needed_sum <= 1e-8:
                break
            priority_mass = needed_series * priority_series.clip(lower=0.0)
            if float(priority_mass.sum()) <= 1e-8:
                priority_mass = needed_series
            step = (priority_mass / float(priority_mass.sum())) * residual_capacity
            step = np.minimum(step.to_numpy(dtype=float), needed_series.to_numpy(dtype=float))
            step_series = pd.Series(step, index=result.index, dtype=float)
            if float(step_series.sum()) <= 1e-8:
                break
            guarded = guarded | (step_series > 1e-8)
            result = (result + step_series).clip(lower=hard_floor_series, upper=cap_series)
            residual_capacity = max(float(target_total) - float(result.sum()), 0.0)
        return result, guarded

    def _trim_delta_to_turnover_by_priority(
        self,
        delta: pd.Series,
        *,
        target_turnover: float,
        priority: pd.Series,
    ) -> tuple[pd.Series, pd.Series]:
        result = pd.to_numeric(delta, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(float).copy()
        priority_series = pd.to_numeric(priority.reindex(result.index), errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(float)
        guarded = pd.Series(False, index=result.index, dtype=bool)
        residual_excess = max(float(result.abs().sum()) - float(target_turnover), 0.0)
        for _ in range(10):
            if residual_excess <= 1e-8:
                break
            reducible = result.abs()
            reducible_sum = float(reducible.sum())
            if reducible_sum <= 1e-8:
                break
            trim_preference = reducible * (1.05 - priority_series.clip(lower=0.0, upper=1.0))
            trim_preference = trim_preference.clip(lower=0.05) * (reducible > 1e-8).astype(float)
            if float(trim_preference.sum()) <= 1e-8:
                trim_preference = reducible
            step = (trim_preference / float(trim_preference.sum())) * residual_excess
            step = np.minimum(step.to_numpy(dtype=float), reducible.to_numpy(dtype=float))
            step_series = pd.Series(step, index=result.index, dtype=float)
            if float(step_series.sum()) <= 1e-8:
                break
            guarded = guarded | (step_series > 1e-8)
            result = np.sign(result.to_numpy(dtype=float)) * np.maximum(reducible.to_numpy(dtype=float) - step_series.to_numpy(dtype=float), 0.0)
            result = pd.Series(result, index=delta.index, dtype=float)
            residual_excess = max(float(result.abs().sum()) - float(target_turnover), 0.0)
        return result, guarded

    def step(
        self,
        *,
        date: pd.Timestamp | str,
        prices: pd.Series,
        policy_frame: pd.DataFrame,
        global_targets: dict[str, Any] | None = None,
        source_label: str = "model",
        execution_semantics: str = DEFAULT_EXECUTION_SEMANTICS,
        budget_semantics: str = DEFAULT_BUDGET_SEMANTICS,
        budget_calibration: str = DEFAULT_BUDGET_CALIBRATION,
    ) -> StepResult:
        execution_semantics = normalize_execution_semantics(execution_semantics)
        budget_semantics = normalize_budget_semantics(budget_semantics)
        budget_calibration = normalize_budget_calibration(budget_calibration)
        signal_dt = pd.Timestamp(date).normalize()
        prices = prices.astype(float).copy()
        prices.index = prices.index.map(str)
        policy = policy_frame.copy()
        policy.index = policy.index.map(str)
        policy = policy.reindex(prices.index).fillna(
            {
                "action_label": "skip",
                "action_strength": 0.0,
                "target_delta_hint": 0.0,
                "hold_boost": 0.0,
                "exit_urgency": 0.0,
                "reduce_fraction": 0.0,
                "exit_hazard": 0.0,
                "sell_pressure": 0.0,
                "sell_attribution_score": 0.0,
                "sell_rank_score": 0.0,
                "lifecycle_sell_gate": 0.0,
                "large_upside_1d_target": 0.0,
                "alpha_opportunity_value": 0.0,
                "hold_continuation_value": 0.0,
                "sell_release_value": 0.0,
                "cash_defense_value": 0.0,
                "deployment_opportunity_cost": 0.0,
                "risk_adjusted_action_value": 0.0,
                "multi_horizon_forward_value": 0.0,
                "multi_horizon_forward_risk": 0.0,
                "multi_horizon_path_value": 0.0,
                "open_action_value": 0.0,
                "add_action_value": 0.0,
                "hold_action_value": 0.0,
                "reduce_action_value": 0.0,
                "exit_action_value": 0.0,
                "relative_opportunity_value": 0.0,
                "action_value_consistency_target": 0.5,
                "value_arbitration_target": 0.5,
                "deploy_value_target": 0.0,
                "release_value_target": 0.0,
                "defense_value_target": 0.0,
                "deploy_gate_target": 0.0,
                "release_gate_target": 0.0,
                "defense_gate_target": 0.0,
                "clipped_intent_risk": 0.0,
                "exit_timing_pressure": 0.0,
            }
        )
        for required_column, default_value in {
            "action_label": "skip",
            "action_strength": 0.0,
            "target_delta_hint": 0.0,
            "hold_boost": 0.0,
        }.items():
            if required_column not in policy.columns:
                policy[required_column] = default_value

        current = self.current_weights(prices.index)
        gross_exposure_target = float((global_targets or {}).get("gross_exposure_target", max(0.20, min(0.95, 1.0 - self.cash_weight))))
        candidate_budget = int((global_targets or {}).get("candidate_budget", self.max_positions) or self.max_positions)
        candidate_budget = max(1, min(candidate_budget, int(self.max_positions)))
        turnover_budget = float((global_targets or {}).get("turnover_budget", self.turnover_limit) or self.turnover_limit)
        gross_exposure_target_raw = float(gross_exposure_target)
        candidate_budget_raw = int(candidate_budget)
        turnover_budget_raw = float(turnover_budget)
        position_cap_target = float((global_targets or {}).get("max_position_weight_target", self.max_position_weight) or self.max_position_weight)
        position_cap_target = float(np.clip(position_cap_target, 0.05, 0.35))
        position_cap_target_raw = float(position_cap_target)
        hold_bias_target = float((global_targets or {}).get("hold_bias_target", 0.25) or 0.25)
        hold_bias_target = float(np.clip(hold_bias_target, 0.05, 0.98))
        reduce_bias_target = float((global_targets or {}).get("reduce_bias_target", 0.10) or 0.10)
        reduce_bias_target = float(np.clip(reduce_bias_target, 0.0, 0.65))
        exit_patience_target = float((global_targets or {}).get("exit_patience_target", 0.20) or 0.20)
        exit_patience_target = float(np.clip(exit_patience_target, 0.05, 0.95))
        reentry_guard_target = float((global_targets or {}).get("reentry_guard_target", 0.0) or 0.0)
        reentry_guard_target = float(np.clip(reentry_guard_target, 0.0, 0.45))
        budget_model_risk_signal = float((global_targets or {}).get("budget_model_risk_signal", 0.0) or 0.0)
        budget_model_deploy_signal = float((global_targets or {}).get("budget_model_deploy_signal", 0.0) or 0.0)
        budget_model_cash_timing_signal = float((global_targets or {}).get("budget_model_cash_timing_signal", 0.0) or 0.0)
        budget_model_alpha_focus_signal = float((global_targets or {}).get("budget_model_alpha_focus_signal", 0.0) or 0.0)
        budget_model_risk_deploy_gap = float((global_targets or {}).get("budget_model_risk_deploy_gap", 0.0) or 0.0)
        budget_model_value_arbitration_signal = float((global_targets or {}).get("budget_model_value_arbitration_signal", 0.5) or 0.5)
        budget_model_alpha_opportunity_signal = float((global_targets or {}).get("budget_model_alpha_opportunity_signal", 0.0) or 0.0)
        budget_model_cash_defense_signal = float((global_targets or {}).get("budget_model_cash_defense_signal", 0.0) or 0.0)
        budget_model_deploy_value_signal = float((global_targets or {}).get("budget_model_deploy_value_signal", 0.0) or 0.0)
        budget_model_release_value_signal = float((global_targets or {}).get("budget_model_release_value_signal", 0.0) or 0.0)
        budget_model_defense_value_signal = float((global_targets or {}).get("budget_model_defense_value_signal", 0.0) or 0.0)
        budget_model_deploy_gate_signal = float((global_targets or {}).get("budget_model_deploy_gate_signal", 0.0) or 0.0)
        budget_model_release_gate_signal = float((global_targets or {}).get("budget_model_release_gate_signal", 0.0) or 0.0)
        budget_model_defense_gate_signal = float((global_targets or {}).get("budget_model_defense_gate_signal", 0.0) or 0.0)
        budget_model_hierarchical_mode = float((global_targets or {}).get("budget_model_hierarchical_mode", 0.0) or 0.0)
        budget_model_constraint_only_mode = float((global_targets or {}).get("budget_model_constraint_only_mode", 0.0) or 0.0)
        hierarchical_budget_mode = budget_model_hierarchical_mode > 0.5
        portfolio_daily_calibration_mode = budget_calibration in BUDGET_CALIBRATION_PORTFOLIO_DAILY_RANKING_SET
        end_to_end_allocation_layer_mode = (
            budget_semantics == BUDGET_SEMANTICS_ALLOCATION_LAYER
            or budget_calibration == BUDGET_CALIBRATION_END_TO_END_ALLOCATION_LAYER
        )
        constraint_only_budget_mode = budget_model_constraint_only_mode > 0.5 or budget_calibration in {
            BUDGET_CALIBRATION_CASH_CONSTRAINT,
            BUDGET_CALIBRATION_CASH_CONSTRAINT_INTENT,
            BUDGET_CALIBRATION_CASH_CONSTRAINT_DEPLOY,
            BUDGET_CALIBRATION_CASH_CONSTRAINT_SELL_SOURCE,
            BUDGET_CALIBRATION_CASH_CONSTRAINT_DIRECT_ACTION,
            BUDGET_CALIBRATION_CASH_CONSTRAINT_DIRECT_ACTION_REALLOCATION,
            BUDGET_CALIBRATION_CASH_CONSTRAINT_DIRECT_ACTION_PAIR_REALLOCATION,
            BUDGET_CALIBRATION_CASH_CONSTRAINT_DIRECT_ACTION_PAIR_COST_GUARD,
            BUDGET_CALIBRATION_CASH_CONSTRAINT_PORTFOLIO_DAILY_RANKING,
        } or portfolio_daily_calibration_mode
        intent_preserving_constraint_mode = budget_calibration in {
            BUDGET_CALIBRATION_CASH_CONSTRAINT_INTENT,
            BUDGET_CALIBRATION_CASH_CONSTRAINT_DEPLOY,
            BUDGET_CALIBRATION_CASH_CONSTRAINT_SELL_SOURCE,
            BUDGET_CALIBRATION_CASH_CONSTRAINT_DIRECT_ACTION,
            BUDGET_CALIBRATION_CASH_CONSTRAINT_DIRECT_ACTION_REALLOCATION,
            BUDGET_CALIBRATION_CASH_CONSTRAINT_DIRECT_ACTION_PAIR_REALLOCATION,
            BUDGET_CALIBRATION_CASH_CONSTRAINT_DIRECT_ACTION_PAIR_COST_GUARD,
            BUDGET_CALIBRATION_CASH_CONSTRAINT_PORTFOLIO_DAILY_RANKING,
        } or portfolio_daily_calibration_mode
        deploy_executability_constraint_mode = budget_calibration in {
            BUDGET_CALIBRATION_CASH_CONSTRAINT_DEPLOY,
            BUDGET_CALIBRATION_CASH_CONSTRAINT_SELL_SOURCE,
            BUDGET_CALIBRATION_CASH_CONSTRAINT_DIRECT_ACTION,
            BUDGET_CALIBRATION_CASH_CONSTRAINT_DIRECT_ACTION_REALLOCATION,
            BUDGET_CALIBRATION_CASH_CONSTRAINT_DIRECT_ACTION_PAIR_REALLOCATION,
            BUDGET_CALIBRATION_CASH_CONSTRAINT_DIRECT_ACTION_PAIR_COST_GUARD,
            BUDGET_CALIBRATION_CASH_CONSTRAINT_PORTFOLIO_DAILY_RANKING,
        } or portfolio_daily_calibration_mode
        direct_action_preserving_mode = budget_calibration in {
            BUDGET_CALIBRATION_CASH_CONSTRAINT_DIRECT_ACTION,
            BUDGET_CALIBRATION_CASH_CONSTRAINT_DIRECT_ACTION_REALLOCATION,
            BUDGET_CALIBRATION_CASH_CONSTRAINT_DIRECT_ACTION_PAIR_REALLOCATION,
            BUDGET_CALIBRATION_CASH_CONSTRAINT_DIRECT_ACTION_PAIR_COST_GUARD,
            BUDGET_CALIBRATION_CASH_CONSTRAINT_PORTFOLIO_DAILY_RANKING,
        } or portfolio_daily_calibration_mode
        portfolio_daily_ranking_mode = portfolio_daily_calibration_mode or end_to_end_allocation_layer_mode
        direct_action_pair_cost_guard_mode = budget_calibration in {
            BUDGET_CALIBRATION_CASH_CONSTRAINT_DIRECT_ACTION_PAIR_COST_GUARD,
            BUDGET_CALIBRATION_CASH_CONSTRAINT_PORTFOLIO_DAILY_RANKING,
        } or portfolio_daily_calibration_mode
        direct_action_pair_reallocation_mode = budget_calibration in {
            BUDGET_CALIBRATION_CASH_CONSTRAINT_DIRECT_ACTION_PAIR_REALLOCATION,
            BUDGET_CALIBRATION_CASH_CONSTRAINT_DIRECT_ACTION_PAIR_COST_GUARD,
            BUDGET_CALIBRATION_CASH_CONSTRAINT_PORTFOLIO_DAILY_RANKING,
        } or portfolio_daily_calibration_mode
        direct_action_reallocation_mode = budget_calibration in {
            BUDGET_CALIBRATION_CASH_CONSTRAINT_DIRECT_ACTION_REALLOCATION,
            BUDGET_CALIBRATION_CASH_CONSTRAINT_DIRECT_ACTION_PAIR_REALLOCATION,
            BUDGET_CALIBRATION_CASH_CONSTRAINT_DIRECT_ACTION_PAIR_COST_GUARD,
            BUDGET_CALIBRATION_CASH_CONSTRAINT_PORTFOLIO_DAILY_RANKING,
        } or portfolio_daily_calibration_mode
        sell_source_decoupled_mode = budget_calibration in {
            BUDGET_CALIBRATION_CASH_CONSTRAINT_SELL_SOURCE,
            BUDGET_CALIBRATION_CASH_CONSTRAINT_DIRECT_ACTION,
            BUDGET_CALIBRATION_CASH_CONSTRAINT_DIRECT_ACTION_REALLOCATION,
            BUDGET_CALIBRATION_CASH_CONSTRAINT_DIRECT_ACTION_PAIR_REALLOCATION,
            BUDGET_CALIBRATION_CASH_CONSTRAINT_DIRECT_ACTION_PAIR_COST_GUARD,
            BUDGET_CALIBRATION_CASH_CONSTRAINT_PORTFOLIO_DAILY_RANKING,
        } or portfolio_daily_calibration_mode
        translation_guard_mode = budget_calibration in {
            BUDGET_CALIBRATION_CASH_TRANSLATION,
            BUDGET_CALIBRATION_CASH_TRANSLATION_SELL,
            BUDGET_CALIBRATION_CASH_CONSTRAINT_INTENT,
            BUDGET_CALIBRATION_CASH_CONSTRAINT_DEPLOY,
            BUDGET_CALIBRATION_CASH_CONSTRAINT_SELL_SOURCE,
            BUDGET_CALIBRATION_CASH_CONSTRAINT_DIRECT_ACTION,
            BUDGET_CALIBRATION_CASH_CONSTRAINT_DIRECT_ACTION_REALLOCATION,
            BUDGET_CALIBRATION_CASH_CONSTRAINT_DIRECT_ACTION_PAIR_REALLOCATION,
            BUDGET_CALIBRATION_CASH_CONSTRAINT_DIRECT_ACTION_PAIR_COST_GUARD,
            BUDGET_CALIBRATION_CASH_CONSTRAINT_PORTFOLIO_DAILY_RANKING,
        } or portfolio_daily_calibration_mode
        use_sell_priority_guard = budget_calibration in {
            BUDGET_CALIBRATION_CASH_TRANSLATION_SELL,
            BUDGET_CALIBRATION_CASH_CONSTRAINT,
            BUDGET_CALIBRATION_CASH_CONSTRAINT_INTENT,
            BUDGET_CALIBRATION_CASH_CONSTRAINT_DEPLOY,
            BUDGET_CALIBRATION_CASH_CONSTRAINT_SELL_SOURCE,
            BUDGET_CALIBRATION_CASH_CONSTRAINT_DIRECT_ACTION,
            BUDGET_CALIBRATION_CASH_CONSTRAINT_DIRECT_ACTION_REALLOCATION,
            BUDGET_CALIBRATION_CASH_CONSTRAINT_DIRECT_ACTION_PAIR_REALLOCATION,
            BUDGET_CALIBRATION_CASH_CONSTRAINT_DIRECT_ACTION_PAIR_COST_GUARD,
            BUDGET_CALIBRATION_CASH_CONSTRAINT_PORTFOLIO_DAILY_RANKING,
        } or portfolio_daily_calibration_mode
        portfolio_daily_source_exec_guard_mode = budget_calibration in {
            BUDGET_CALIBRATION_CASH_CONSTRAINT_PORTFOLIO_DAILY_RANKING_SOURCE_EXEC,
            BUDGET_CALIBRATION_CASH_CONSTRAINT_PORTFOLIO_DAILY_RANKING_RECEIVER_EXEC,
        }
        portfolio_daily_receiver_exec_guard_mode = (
            budget_calibration == BUDGET_CALIBRATION_CASH_CONSTRAINT_PORTFOLIO_DAILY_RANKING_RECEIVER_EXEC
        )

        action_names = policy["action_label"].astype(str).str.strip().str.lower()

        def _policy_numeric(name: str, default: float = 0.0) -> pd.Series:
            if name not in policy.columns:
                return pd.Series(float(default), index=policy.index, dtype=float)
            return pd.to_numeric(policy[name], errors="coerce").fillna(float(default)).astype(float)

        def _optional_policy_numeric(name: str) -> pd.Series | None:
            if name not in policy.columns:
                return None
            return pd.to_numeric(policy[name], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(float)

        def _masked_mean(values: pd.Series, mask: pd.Series) -> float:
            selected = values.loc[mask.reindex(values.index).fillna(False)]
            if selected.empty:
                return 0.0
            mean_value = float(selected.mean())
            return mean_value if np.isfinite(mean_value) else 0.0

        portfolio_context = self.portfolio_features()
        held_mask = current > 1e-8
        flat_mask = ~held_mask
        held_count_before = int(held_mask.sum())
        held_sell_action_share = float((action_names.loc[held_mask].isin({"reduce", "exit"})).mean()) if held_count_before else 0.0
        held_exit_action_share = float((action_names.loc[held_mask] == "exit").mean()) if held_count_before else 0.0
        held_add_action_share = float((action_names.loc[held_mask] == "add").mean()) if held_count_before else 0.0
        flat_entry_action_share = float((action_names.loc[flat_mask].isin({"open", "add"})).mean()) if int(flat_mask.sum()) else 0.0
        sell_pressure_series = _policy_numeric("sell_pressure")
        sell_attribution_series = _policy_numeric("sell_attribution_score")
        sell_rank_series = _policy_numeric("sell_rank_score")
        lifecycle_sell_gate_series = _policy_numeric("lifecycle_sell_gate")
        large_upside_series = _policy_numeric("large_upside_1d_target")
        alpha_opportunity_series = _policy_numeric("alpha_opportunity_value")
        hold_continuation_series = _policy_numeric("hold_continuation_value")
        sell_release_series = _policy_numeric("sell_release_value")
        cash_defense_series = _policy_numeric("cash_defense_value")
        deployment_opportunity_series = _policy_numeric("deployment_opportunity_cost")
        risk_adjusted_action_value_series = _policy_numeric("risk_adjusted_action_value")
        multi_horizon_forward_value_series = _policy_numeric("multi_horizon_forward_value")
        multi_horizon_forward_risk_series = _policy_numeric("multi_horizon_forward_risk")
        multi_horizon_path_value_series = _policy_numeric("multi_horizon_path_value")
        value_arbitration_series = _policy_numeric("value_arbitration_target", default=0.5)
        deploy_value_series = _policy_numeric("deploy_value_target")
        release_value_series = _policy_numeric("release_value_target")
        defense_value_series = _policy_numeric("defense_value_target")
        deploy_gate_series = _policy_numeric("deploy_gate_target")
        release_gate_series = _policy_numeric("release_gate_target")
        defense_gate_series = _policy_numeric("defense_gate_target")
        deploy_executability_series = _policy_numeric("deploy_executability_target")
        model_receiver_score_series = _optional_policy_numeric("portfolio_daily_receiver_score")
        model_receiver_executability_series = _optional_policy_numeric("portfolio_daily_receiver_executability")
        model_receiver_capacity_series = _optional_policy_numeric("portfolio_daily_receiver_add_capacity")
        model_source_score_series = _optional_policy_numeric("portfolio_daily_source_score")
        model_source_release_capacity_series = _optional_policy_numeric("portfolio_daily_source_release_capacity")
        model_source_forward_spread_score_series = _optional_policy_numeric("portfolio_daily_source_forward_spread_score")
        model_source_bad_forward_spread_risk_series = _optional_policy_numeric("portfolio_daily_source_bad_forward_spread_risk")
        model_source_economic_release_score_series = _optional_policy_numeric("portfolio_daily_source_economic_release_score")
        model_source_economic_block_risk_series = _optional_policy_numeric("portfolio_daily_source_economic_block_risk")
        model_source_forward_strength_brake_risk_series = _optional_policy_numeric(
            "portfolio_daily_source_forward_strength_brake_risk"
        )
        model_source_release_quality_series = _optional_policy_numeric("portfolio_daily_source_release_quality")
        model_source_opportunity_cost_series = _optional_policy_numeric("portfolio_daily_source_opportunity_cost")
        model_source_executability_series = _optional_policy_numeric("portfolio_daily_source_executability")
        model_cash_score_series = _optional_policy_numeric("portfolio_daily_cash_score")
        model_receiver_funding_coverage_series = _optional_policy_numeric("portfolio_daily_receiver_funding_coverage")
        model_funding_closure_score_series = _optional_policy_numeric("portfolio_daily_funding_closure_score")
        model_allocation_transfer_score_series = _optional_policy_numeric("portfolio_daily_allocation_transfer_score")
        model_allocation_dead_branch_risk_series = _optional_policy_numeric(
            "portfolio_daily_allocation_dead_branch_risk"
        )
        model_unified_receiver_score_series = _optional_policy_numeric("portfolio_daily_unified_receiver_score")
        model_unified_source_score_series = _optional_policy_numeric("portfolio_daily_unified_source_score")
        model_unified_cash_score_series = _optional_policy_numeric("portfolio_daily_unified_cash_score")
        model_source_positive_forward_penalty_series = _optional_policy_numeric(
            "portfolio_daily_source_positive_forward_penalty"
        )
        model_source_opportunity_cost_penalty_series = _optional_policy_numeric(
            "portfolio_daily_source_opportunity_cost_penalty"
        )
        model_receiver_source_spread_reward_series = _optional_policy_numeric(
            "portfolio_daily_receiver_source_spread_reward"
        )
        model_unified_allocation_objective_series = _optional_policy_numeric(
            "portfolio_daily_unified_allocation_objective"
        )
        unified_allocation_policy_mode = any(
            series is not None
            for series in (
                model_unified_receiver_score_series,
                model_unified_source_score_series,
                model_unified_cash_score_series,
                model_source_positive_forward_penalty_series,
                model_source_opportunity_cost_penalty_series,
                model_receiver_source_spread_reward_series,
                model_unified_allocation_objective_series,
            )
        )
        portfolio_daily_unified_receiver_score = (
            model_unified_receiver_score_series.clip(0.0, 1.0)
            if model_unified_receiver_score_series is not None
            else pd.Series(0.0, index=prices.index, dtype=float)
        )
        portfolio_daily_unified_source_score = (
            model_unified_source_score_series.clip(0.0, 1.0).where(held_mask, 0.0)
            if model_unified_source_score_series is not None
            else pd.Series(0.0, index=prices.index, dtype=float)
        )
        portfolio_daily_unified_cash_score = (
            model_unified_cash_score_series.clip(0.0, 1.0)
            if model_unified_cash_score_series is not None
            else pd.Series(0.0, index=prices.index, dtype=float)
        )
        portfolio_daily_unified_source_positive_forward_penalty = (
            model_source_positive_forward_penalty_series.clip(0.0, 1.0).where(held_mask, 0.0)
            if model_source_positive_forward_penalty_series is not None
            else pd.Series(0.0, index=prices.index, dtype=float)
        )
        portfolio_daily_unified_source_opportunity_cost_penalty = (
            model_source_opportunity_cost_penalty_series.clip(0.0, 1.0).where(held_mask, 0.0)
            if model_source_opportunity_cost_penalty_series is not None
            else pd.Series(0.0, index=prices.index, dtype=float)
        )
        portfolio_daily_unified_receiver_source_spread_reward = (
            model_receiver_source_spread_reward_series.clip(0.0, 1.0).where(held_mask, 0.0)
            if model_receiver_source_spread_reward_series is not None
            else pd.Series(0.0, index=prices.index, dtype=float)
        )
        portfolio_daily_unified_allocation_objective = (
            model_unified_allocation_objective_series.clip(0.0, 1.0)
            if model_unified_allocation_objective_series is not None
            else pd.Series(0.0, index=prices.index, dtype=float)
        )
        if model_receiver_executability_series is not None:
            deploy_executability_series = (
                0.72 * deploy_executability_series + 0.28 * model_receiver_executability_series.clip(0.0, 1.0)
            ).clip(0.0, 1.0)
        if hierarchical_budget_mode:
            decision_gate_denominator_series = (deploy_value_series + release_value_series + 1.0e-6).clip(lower=1.0e-6)
            decision_deploy_gate_series = (
                0.62 * deploy_gate_series
                + 0.38 * (deploy_value_series / decision_gate_denominator_series).clip(0.0, 1.0)
            ).clip(0.0, 1.0)
            decision_release_gate_series = (
                0.62 * release_gate_series
                + 0.38 * (release_value_series / decision_gate_denominator_series).clip(0.0, 1.0)
            ).clip(0.0, 1.0)
        else:
            decision_deploy_gate_series = deploy_gate_series
            decision_release_gate_series = release_gate_series
        clipped_intent_risk_series = _policy_numeric("clipped_intent_risk")
        exit_timing_pressure_series = _policy_numeric("exit_timing_pressure")
        exit_hazard_series = _policy_numeric("exit_hazard")
        entry_quality_series = _policy_numeric("entry_quality")
        direct_action_applied_series = _policy_numeric("direct_action_value_applied")
        direct_action_gap_series = _policy_numeric("direct_action_value_gap")
        direct_action_mode_series = (
            policy.get("policy_decision_mode", pd.Series("", index=policy.index))
            .astype(str)
            .str.lower()
            .eq("direct_action_value_v1")
            | (direct_action_applied_series > 0.5)
        )
        direct_action_keep_utility_series = pd.concat(
            [
                _policy_numeric("direct_action_utility_hold"),
                _policy_numeric("direct_action_utility_add"),
            ],
            axis=1,
        ).max(axis=1)
        direct_action_release_utility_series = pd.concat(
            [
                _policy_numeric("direct_action_utility_reduce"),
                _policy_numeric("direct_action_utility_exit"),
            ],
            axis=1,
        ).max(axis=1)
        direct_action_deploy_utility_series = pd.concat(
            [
                _policy_numeric("direct_action_utility_open"),
                _policy_numeric("direct_action_utility_add"),
            ],
            axis=1,
        ).max(axis=1)
        direct_action_release_advantage_series = (
            direct_action_release_utility_series - direct_action_keep_utility_series
        )
        direct_action_keep_advantage_series = (
            direct_action_keep_utility_series - direct_action_release_utility_series
        )
        direct_action_add_advantage_series = (
            _policy_numeric("direct_action_utility_add") - _policy_numeric("direct_action_utility_hold")
        )
        direct_action_open_advantage_series = (
            _policy_numeric("direct_action_utility_open") - _policy_numeric("direct_action_utility_skip")
        )
        direct_action_deploy_advantage_series = pd.concat(
            [
                direct_action_add_advantage_series.rename("add"),
                direct_action_open_advantage_series.rename("open"),
            ],
            axis=1,
        ).max(axis=1)
        direct_action_add_signal = (
            direct_action_reallocation_mode
            & direct_action_mode_series
            & held_mask
            & action_names.eq("add")
            & (
                (direct_action_add_advantage_series >= 0.045)
                | (
                    (direct_action_gap_series >= 0.060)
                    & (deploy_executability_series >= 0.42)
                )
                | (
                    (deploy_executability_series >= 0.62)
                    & (decision_deploy_gate_series >= decision_release_gate_series + 0.025)
                )
            )
            & (direct_action_release_advantage_series < 0.060)
        )
        direct_action_open_signal = (
            direct_action_reallocation_mode
            & direct_action_mode_series
            & flat_mask
            & action_names.eq("open")
            & (
                (direct_action_open_advantage_series >= 0.035)
                | (
                    (direct_action_gap_series >= 0.055)
                    & (deploy_executability_series >= 0.42)
                )
            )
        )
        direct_action_deploy_signal = direct_action_add_signal | direct_action_open_signal
        direct_action_deploy_rank_score = (
            direct_action_deploy_advantage_series.clip(lower=-0.25, upper=0.50) * 1.35
            + direct_action_gap_series.clip(lower=0.0, upper=0.35) * 0.64
            + deploy_executability_series.clip(0.0, 1.0) * 0.24
            + decision_deploy_gate_series.clip(0.0, 1.0) * 0.16
            + alpha_opportunity_series.clip(0.0, 1.0) * 0.08
            - current.clip(0.0, 0.25) * 0.16
            - cash_defense_series.clip(0.0, 1.0) * 0.08
        )
        portfolio_daily_receiver_pre_add_headroom = (
            pd.Series(float(position_cap_target), index=prices.index, dtype=float) - current
        )
        portfolio_daily_receiver_pre_min_add_delta = pd.concat(
            [
                pd.Series(DEFAULT_EXECUTION_DEADBAND_ABS * 2.0, index=prices.index, dtype=float),
                current.clip(lower=0.0) * 0.025,
                pd.Series(float(position_cap_target) * 0.018, index=prices.index, dtype=float),
            ],
            axis=1,
        ).max(axis=1)
        portfolio_daily_receiver_pre_add_capacity = (
            portfolio_daily_receiver_pre_add_headroom
            / portfolio_daily_receiver_pre_min_add_delta.clip(lower=1.0e-6)
        ).replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(0.0, 1.0)
        portfolio_daily_receiver_pre_add_capacity = portfolio_daily_receiver_pre_add_capacity.where(
            held_mask,
            1.0,
        )
        portfolio_daily_receiver_observable_pre_add_capacity = portfolio_daily_receiver_pre_add_capacity.copy()
        if model_receiver_capacity_series is not None:
            portfolio_daily_receiver_pre_add_capacity = (
                0.58 * model_receiver_capacity_series.clip(0.0, 1.0)
                + 0.42 * portfolio_daily_receiver_pre_add_capacity
            ).clip(0.0, 1.0)
            portfolio_daily_receiver_pre_add_capacity = pd.concat(
                [
                    portfolio_daily_receiver_pre_add_capacity.rename("model_blend"),
                    (
                        portfolio_daily_receiver_observable_pre_add_capacity
                        + pd.Series(0.16, index=prices.index, dtype=float).where(held_mask, 0.0)
                    ).clip(0.0, 1.0).rename("observable_cap"),
                ],
                axis=1,
            ).min(axis=1).clip(0.0, 1.0)
        portfolio_daily_receiver_score = (
            direct_action_deploy_rank_score.replace([np.inf, -np.inf], np.nan).fillna(0.0) * 0.52
            + deploy_value_series.clip(0.0, 1.0) * 0.18
            + decision_deploy_gate_series.clip(0.0, 1.0) * 0.16
            + deploy_executability_series.clip(0.0, 1.0) * 0.18
            + alpha_opportunity_series.clip(0.0, 1.0) * 0.14
            + value_arbitration_series.clip(0.0, 1.0) * 0.08
            - cash_defense_series.clip(0.0, 1.0) * 0.16
            - release_value_series.clip(0.0, 1.0) * 0.08
            - current.clip(0.0, 0.25) * 0.10
            - (1.0 - portfolio_daily_receiver_pre_add_capacity).clip(0.0, 1.0).where(held_mask, 0.0) * 0.22
        )
        if model_receiver_score_series is not None:
            receiver_capacity_bonus = (
                portfolio_daily_receiver_pre_add_capacity.clip(0.0, 1.0) * 0.08
            )
            portfolio_daily_receiver_score = (
                0.60 * model_receiver_score_series.clip(0.0, 1.0)
                + 0.40 * portfolio_daily_receiver_score
                + receiver_capacity_bonus
                - (1.0 - portfolio_daily_receiver_pre_add_capacity).clip(0.0, 1.0).where(held_mask, 0.0) * 0.18
            ).clip(lower=-0.25, upper=1.35)
        if model_unified_receiver_score_series is not None:
            portfolio_daily_receiver_score = (
                0.48 * portfolio_daily_receiver_score
                + 0.52 * portfolio_daily_unified_receiver_score
                + portfolio_daily_receiver_pre_add_capacity.clip(0.0, 1.0) * 0.05
                - cash_defense_series.clip(0.0, 1.0) * 0.04
            ).clip(lower=-0.25, upper=1.35)
        portfolio_daily_receiver_funding_coverage = pd.Series(0.0, index=prices.index, dtype=float)
        if model_receiver_funding_coverage_series is not None:
            portfolio_daily_receiver_funding_coverage = model_receiver_funding_coverage_series.clip(0.0, 1.0)
        portfolio_daily_funding_closure_score = pd.Series(0.0, index=prices.index, dtype=float)
        if model_funding_closure_score_series is not None:
            portfolio_daily_funding_closure_score = model_funding_closure_score_series.clip(0.0, 1.0)
        portfolio_daily_allocation_transfer_score = pd.Series(0.0, index=prices.index, dtype=float)
        if model_allocation_transfer_score_series is not None:
            portfolio_daily_allocation_transfer_score = model_allocation_transfer_score_series.clip(0.0, 1.0)
        portfolio_daily_allocation_dead_branch_risk = pd.Series(0.0, index=prices.index, dtype=float)
        if model_allocation_dead_branch_risk_series is not None:
            portfolio_daily_allocation_dead_branch_risk = model_allocation_dead_branch_risk_series.clip(0.0, 1.0)
        if model_receiver_funding_coverage_series is not None or model_funding_closure_score_series is not None:
            portfolio_daily_receiver_score = (
                portfolio_daily_receiver_score
                + portfolio_daily_receiver_funding_coverage.clip(0.0, 1.0) * 0.16
                + portfolio_daily_funding_closure_score.clip(0.0, 1.0) * 0.08
                + portfolio_daily_allocation_transfer_score.clip(0.0, 1.0) * 0.06
                - portfolio_daily_allocation_dead_branch_risk.clip(0.0, 1.0) * 0.12
            ).clip(lower=-0.25, upper=1.35)
        portfolio_daily_receiver_pre_exec_pass = (
            (~held_mask)
            | (portfolio_daily_receiver_pre_add_capacity >= 0.70)
            | (
                (portfolio_daily_receiver_pre_add_capacity >= 0.46)
                & (deploy_executability_series.clip(0.0, 1.0) >= 0.52)
                & (direct_action_add_advantage_series >= 0.060)
            )
        )
        flat_alpha_opportunity_value = _masked_mean(alpha_opportunity_series, flat_mask)
        flat_deployment_opportunity_cost = _masked_mean(deployment_opportunity_series, flat_mask)
        flat_deploy_executability = _masked_mean(deploy_executability_series, flat_mask)
        current_gross = float(current.sum())
        portfolio_daily_receiver_semantic_no_headroom = pd.Series(False, index=prices.index, dtype=bool)
        if portfolio_daily_receiver_exec_guard_mode:
            portfolio_daily_receiver_semantic_no_headroom = (
                direct_action_add_signal
                & held_mask
                & (
                    (portfolio_daily_receiver_pre_add_headroom < portfolio_daily_receiver_pre_min_add_delta)
                    | (portfolio_daily_receiver_pre_add_capacity < 0.46)
                )
            )
            if bool(portfolio_daily_receiver_semantic_no_headroom.any()):
                direct_action_add_signal = direct_action_add_signal & (~portfolio_daily_receiver_semantic_no_headroom)
        receiver_open_breadth_pressure = bool(
            portfolio_daily_ranking_mode
            and float(self.cash_weight) >= 0.28
            and held_count_before < int(self.max_positions)
            and float(gross_exposure_target) >= max(0.42, current_gross + 0.10)
            and (
                budget_model_deploy_signal >= 0.42
                or budget_model_alpha_focus_signal >= 0.40
                or flat_entry_action_share >= 0.08
                or float(
                    np.clip(
                        0.42 * flat_alpha_opportunity_value
                        + 0.34 * flat_deployment_opportunity_cost
                        + 0.24 * flat_deploy_executability,
                        0.0,
                        1.0,
                    )
                )
                >= 0.20
            )
        )
        portfolio_daily_receiver_open_breadth_candidate = (
            portfolio_daily_ranking_mode
            & receiver_open_breadth_pressure
            & flat_mask
            & (portfolio_daily_receiver_score >= 0.055)
            & (cash_defense_series < 0.660)
            & (exit_timing_pressure_series < 0.460)
            & (
                (deploy_executability_series >= 0.24)
                | (decision_deploy_gate_series >= decision_release_gate_series + 0.050)
                | (alpha_opportunity_series >= 0.28)
                | (deployment_opportunity_series >= 0.26)
            )
        )
        if bool(portfolio_daily_receiver_open_breadth_candidate.any()):
            direct_action_open_signal = direct_action_open_signal | portfolio_daily_receiver_open_breadth_candidate
        direct_action_deploy_signal = direct_action_add_signal | direct_action_open_signal
        portfolio_daily_receiver_funding_pass = pd.Series(True, index=prices.index, dtype=bool)
        if model_receiver_funding_coverage_series is not None or model_allocation_transfer_score_series is not None:
            funding_signal = (
                portfolio_daily_receiver_funding_coverage.clip(0.0, 1.0) * 0.64
                + portfolio_daily_allocation_transfer_score.clip(0.0, 1.0) * 0.24
                + portfolio_daily_funding_closure_score.clip(0.0, 1.0) * 0.12
            )
            portfolio_daily_receiver_funding_pass = (
                (current_gross < 0.72)
                | (cash_defense_series.clip(0.0, 1.0) < 0.54)
                | (funding_signal >= 0.12)
                | ((portfolio_daily_receiver_pre_add_capacity >= 0.82) & (portfolio_daily_receiver_score >= 0.22))
            )
        portfolio_daily_unified_receiver_candidate = pd.Series(False, index=prices.index, dtype=bool)
        if unified_allocation_policy_mode and model_unified_receiver_score_series is not None:
            unified_receiver_deploy_context = (
                (float(self.cash_weight) >= 0.18)
                or (float(gross_exposure_target) >= current_gross + 0.04)
                or (budget_model_deploy_signal >= 0.36)
                or (budget_model_alpha_focus_signal >= 0.34)
            )
            portfolio_daily_unified_receiver_candidate = (
                portfolio_daily_ranking_mode
                & portfolio_daily_receiver_pre_exec_pass
                & portfolio_daily_receiver_funding_pass
                & (
                    (portfolio_daily_unified_receiver_score >= 0.180)
                    | (portfolio_daily_receiver_score >= 0.240)
                )
                & (cash_defense_series < 0.760)
                & (exit_timing_pressure_series < 0.580)
                & (
                    (
                        flat_mask
                        & bool(unified_receiver_deploy_context)
                        & (
                            (deploy_executability_series.clip(0.0, 1.0) >= 0.10)
                            | (portfolio_daily_unified_receiver_score >= 0.260)
                        )
                    )
                    | (
                        held_mask
                        & (portfolio_daily_receiver_pre_add_capacity >= 0.46)
                        & (portfolio_daily_unified_receiver_score >= 0.240)
                    )
                )
            )
            direct_action_open_signal = direct_action_open_signal | (portfolio_daily_unified_receiver_candidate & flat_mask)
            direct_action_add_signal = direct_action_add_signal | (portfolio_daily_unified_receiver_candidate & held_mask)
            direct_action_deploy_signal = direct_action_add_signal | direct_action_open_signal
        portfolio_daily_receiver_candidate = (
            portfolio_daily_ranking_mode
            & direct_action_deploy_signal
            & portfolio_daily_receiver_pre_exec_pass
            & portfolio_daily_receiver_funding_pass
            & (
                (portfolio_daily_receiver_score >= 0.035)
                | (
                    (portfolio_daily_receiver_score >= 0.000)
                    & (deploy_executability_series.clip(0.0, 1.0) >= 0.45)
                    & (deploy_value_series.clip(0.0, 1.0) >= 0.56)
                    & (alpha_opportunity_series.clip(0.0, 1.0) >= 0.24)
                )
            )
            & (cash_defense_series < 0.720)
            & (exit_timing_pressure_series < 0.520)
        ) | portfolio_daily_unified_receiver_candidate
        portfolio_daily_receiver_target = pd.Series(False, index=prices.index, dtype=bool)
        direct_action_core_deploy_target = direct_action_deploy_signal.copy()
        paired_reallocation_pressure = False
        direct_action_add_rank = pd.Series(np.inf, index=prices.index, dtype=float)
        direct_action_open_rank = pd.Series(np.inf, index=prices.index, dtype=float)
        current_gross = float(current.sum())
        if direct_action_pair_reallocation_mode and bool(direct_action_deploy_signal.any()):
            deploy_signal_count = int(direct_action_deploy_signal.sum())
            add_signal_count = int(direct_action_add_signal.sum())
            open_signal_count = int(direct_action_open_signal.sum())
            if add_signal_count > 0:
                add_scores = direct_action_deploy_rank_score.where(direct_action_add_signal)
                direct_action_add_rank = add_scores.rank(method="first", ascending=False)
            if open_signal_count > 0:
                open_scores = direct_action_deploy_rank_score.where(direct_action_open_signal)
                direct_action_open_rank = open_scores.rank(method="first", ascending=False)
            portfolio_daily_receiver_count = int(portfolio_daily_receiver_candidate.sum())
            if portfolio_daily_ranking_mode and portfolio_daily_receiver_count > 0:
                direct_action_core_deploy_target = pd.Series(False, index=prices.index, dtype=bool)
                receiver_pressure = (
                    current_gross >= 0.84
                    or deploy_signal_count >= 2
                    or budget_model_deploy_signal >= 0.58
                    or budget_model_alpha_focus_signal >= 0.55
                )
                receiver_fraction = 0.44 if not receiver_pressure else 0.34
                receiver_funding_context_count = int(
                    (
                        held_mask
                        & (
                            (
                                (release_value_series.clip(0.0, 1.0) >= 0.34)
                                & (sell_release_series.clip(0.0, 1.0) >= 0.28)
                            )
                            | (
                                (decision_release_gate_series >= decision_deploy_gate_series + 0.14)
                                & (multi_horizon_forward_risk_series.clip(0.0, 1.0) >= 0.36)
                            )
                            | action_names.isin({"reduce", "exit"})
                        )
                    ).sum()
                )
                cash_slot_context = int(max(0.0, float(self.cash_weight)) / max(float(position_cap_target), 1.0e-6))
                receiver_funding_context_slots = receiver_funding_context_count + cash_slot_context
                receiver_slot_cap = int(self.max_positions)
                if portfolio_daily_ranking_mode and current_gross >= 0.84 and receiver_funding_context_slots < 1:
                    receiver_fraction = 0.0
                    receiver_slot_cap = 0
                elif portfolio_daily_ranking_mode and current_gross >= 0.84 and receiver_funding_context_slots < 2:
                    receiver_fraction = min(receiver_fraction, 0.12)
                    receiver_slot_cap = min(receiver_slot_cap, 1)
                elif portfolio_daily_ranking_mode and current_gross >= 0.80 and receiver_funding_context_slots < 3:
                    receiver_fraction = min(receiver_fraction, 0.18)
                    receiver_slot_cap = min(receiver_slot_cap, 2)
                elif portfolio_daily_ranking_mode and current_gross >= 0.72 and receiver_funding_context_slots < 4:
                    receiver_fraction = min(receiver_fraction, 0.22)
                    receiver_slot_cap = min(receiver_slot_cap, 3)
                if receiver_slot_cap <= 0 or receiver_fraction <= 0.0:
                    receiver_limit = 0
                else:
                    receiver_limit = min(
                        portfolio_daily_receiver_count,
                        max(1, min(receiver_slot_cap, int(np.ceil(float(self.max_positions) * receiver_fraction)))),
                    )
                if receiver_limit > 0:
                    receiver_rank = portfolio_daily_receiver_score.where(portfolio_daily_receiver_candidate).rank(
                        method="first",
                        ascending=False,
                    )
                    portfolio_daily_receiver_target = portfolio_daily_receiver_candidate & (
                        receiver_rank <= float(receiver_limit)
                    )
                direct_action_core_deploy_target = portfolio_daily_receiver_target.copy()
            else:
                paired_reallocation_pressure = current_gross >= 0.92 and deploy_signal_count >= 3
                if paired_reallocation_pressure:
                    direct_action_core_deploy_target = pd.Series(False, index=prices.index, dtype=bool)
                    if add_signal_count > 0:
                        add_limit = min(add_signal_count, max(1, min(3, int(np.ceil(add_signal_count * 0.35)))))
                        direct_action_core_deploy_target = direct_action_core_deploy_target | (
                            direct_action_add_signal & (direct_action_add_rank <= float(add_limit))
                        )
                    if open_signal_count > 0:
                        open_limit = min(open_signal_count, max(1, min(3, int(np.ceil(open_signal_count * 0.04)))))
                        direct_action_core_deploy_target = direct_action_core_deploy_target | (
                            direct_action_open_signal & (direct_action_open_rank <= float(open_limit))
                        )
        if portfolio_daily_receiver_exec_guard_mode:
            direct_action_core_deploy_target = portfolio_daily_receiver_target & portfolio_daily_receiver_candidate
        direct_action_executable_target = (
            direct_action_core_deploy_target
            if direct_action_pair_reallocation_mode
            else pd.Series(True, index=prices.index, dtype=bool)
        )
        direct_action_add_authorized = direct_action_add_signal & direct_action_executable_target
        direct_action_open_authorized = direct_action_open_signal & direct_action_executable_target
        direct_action_deploy_authorized = direct_action_add_authorized | direct_action_open_authorized
        direct_action_funding_release_authorized = (
            direct_action_preserving_mode
            & direct_action_mode_series
            & held_mask
            & (
                (
                    (direct_action_release_advantage_series >= 0.035)
                    & (direct_action_gap_series >= 0.030)
                )
                | (
                    (direct_action_release_advantage_series >= 0.010)
                    & (
                        (sell_release_series >= 0.44)
                        | (decision_release_gate_series >= decision_deploy_gate_series + 0.05)
                        | (exit_timing_pressure_series >= 0.38)
                    )
                )
            )
        )
        direct_action_funding_protected = (
            direct_action_preserving_mode
            & direct_action_mode_series
            & held_mask
            & (
                (direct_action_keep_advantage_series >= 0.030)
                | (
                    (direct_action_keep_utility_series >= 0.34)
                    & (direct_action_release_advantage_series < 0.015)
                )
                | (
                    (direct_action_gap_series < 0.030)
                    & (direct_action_release_advantage_series < 0.040)
                )
            )
        )
        direct_action_reallocation_source_candidate = (
            direct_action_reallocation_mode
            & bool(direct_action_deploy_authorized.any())
            & direct_action_mode_series
            & held_mask
            & action_names.isin({"hold", "skip"})
            & (direct_action_keep_advantage_series < 0.180)
            & (hold_continuation_series < 0.820)
            & (alpha_opportunity_series < 0.760)
            & (sell_pressure_series < 0.340)
            & (exit_timing_pressure_series < 0.420)
        )
        direct_action_reallocation_protection_override = (
            direct_action_reallocation_source_candidate
            & direct_action_funding_protected
            & (direct_action_keep_advantage_series < 0.140)
            & (hold_continuation_series < 0.780)
            & (alpha_opportunity_series < 0.720)
            & (deploy_executability_series < 0.760)
            & (direct_action_release_advantage_series <= 0.010)
        )
        direct_action_hold_reallocation_source = direct_action_reallocation_source_candidate & (
            (~direct_action_funding_protected) | direct_action_reallocation_protection_override
        )
        direct_action_pair_reallocation_source_candidate = (
            direct_action_pair_reallocation_mode
            & bool(direct_action_deploy_authorized.any())
            & bool(paired_reallocation_pressure)
            & direct_action_mode_series
            & held_mask
            & direct_action_add_signal
            & (~direct_action_add_authorized)
            & (current >= 0.015)
            & (direct_action_release_advantage_series < 0.040)
            & (sell_pressure_series < 0.380)
            & (exit_timing_pressure_series < 0.480)
            & (direct_action_add_rank > 1.0)
        )
        direct_action_pair_source_baseline_eligible = (
            direct_action_pair_reallocation_source_candidate
            & (direct_action_add_rank > 1.0)
            & (hold_continuation_series < 0.900)
            & (alpha_opportunity_series < 0.920)
            & (deploy_executability_series < 0.930)
        )
        direct_action_pair_core_reference_score = 0.0
        core_rank_scores = direct_action_deploy_rank_score.loc[direct_action_core_deploy_target].replace(
            [np.inf, -np.inf],
            np.nan,
        ).dropna()
        if bool(len(core_rank_scores)):
            direct_action_pair_core_reference_score = float(core_rank_scores.min())
        direct_action_pair_opportunity_spread = (
            pd.Series(direct_action_pair_core_reference_score, index=prices.index, dtype=float)
            - direct_action_deploy_rank_score.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        )
        direct_action_pair_source_opportunity_cost = (
            hold_continuation_series.clip(0.0, 1.0) * 0.38
            + alpha_opportunity_series.clip(0.0, 1.0) * 0.34
            + deploy_executability_series.clip(0.0, 1.0) * 0.22
            + (direct_action_keep_advantage_series.clip(lower=0.0, upper=0.18) / 0.18) * 0.18
            + large_upside_series.clip(0.0, 1.0) * 0.10
        ).clip(0.0, 1.0)
        pair_min_spread = float(
            np.clip(
                0.075
                + budget_model_alpha_focus_signal * 0.010
                + max(budget_model_deploy_signal - 0.55, 0.0) * 0.014,
                0.075,
                0.115,
            )
        )
        direct_action_pair_source_release_score = (
            direct_action_pair_opportunity_spread.clip(lower=-0.10, upper=0.20) * 1.35
            + (1.0 - direct_action_pair_source_opportunity_cost).clip(0.0, 1.0) * 0.72
            + direct_action_release_advantage_series.clip(lower=0.0, upper=0.12) * 0.28
            - current.clip(0.0, 0.25) * 0.18
            - large_upside_series.clip(0.0, 1.0) * 0.22
        )
        if direct_action_pair_cost_guard_mode:
            direct_action_pair_cost_guard_pass = (
                direct_action_pair_source_baseline_eligible
                & (direct_action_pair_opportunity_spread >= pair_min_spread)
                & (direct_action_pair_source_opportunity_cost <= 0.640)
                & (large_upside_series < 0.680)
                & (hold_continuation_series < 0.860)
                & (alpha_opportunity_series < 0.860)
                & (deploy_executability_series < 0.880)
            )
            guarded_pair_count = int(direct_action_pair_cost_guard_pass.sum())
            core_target_count = int(direct_action_core_deploy_target.sum())
            pair_source_limit = 0
            if guarded_pair_count > 0 and core_target_count > 0:
                pair_source_limit = min(guarded_pair_count, max(1, min(4, int(np.ceil(core_target_count * 1.25)))))
            if pair_source_limit > 0:
                pair_source_rank = direct_action_pair_source_release_score.where(
                    direct_action_pair_cost_guard_pass
                ).rank(method="first", ascending=False)
                direct_action_pair_reallocation_source = direct_action_pair_cost_guard_pass & (
                    pair_source_rank <= float(pair_source_limit)
                )
            else:
                direct_action_pair_reallocation_source = pd.Series(False, index=prices.index, dtype=bool)
            direct_action_pair_cost_guard_blocked = (
                direct_action_pair_reallocation_source_candidate
                & (~direct_action_pair_reallocation_source)
            )
        else:
            direct_action_pair_cost_guard_pass = direct_action_pair_source_baseline_eligible
            direct_action_pair_reallocation_source = direct_action_pair_source_baseline_eligible
            direct_action_pair_cost_guard_blocked = pd.Series(False, index=prices.index, dtype=bool)
        direct_action_reallocation_source = (
            direct_action_hold_reallocation_source | direct_action_pair_reallocation_source
        )
        held_sell_pressure = _masked_mean(sell_pressure_series, held_mask)
        held_sell_attribution = _masked_mean(sell_attribution_series, held_mask)
        held_sell_rank = _masked_mean(sell_rank_series, held_mask)
        held_lifecycle_sell_gate = _masked_mean(lifecycle_sell_gate_series, held_mask)
        held_hold_continuation_value = _masked_mean(hold_continuation_series, held_mask)
        held_sell_release_value = _masked_mean(sell_release_series, held_mask)
        held_cash_defense_value = _masked_mean(cash_defense_series, held_mask)
        held_release_value = _masked_mean(release_value_series, held_mask)
        held_release_gate = _masked_mean(decision_release_gate_series, held_mask)
        held_stock_defense_gate = _masked_mean(defense_gate_series, held_mask)
        flat_alpha_opportunity_value = _masked_mean(alpha_opportunity_series, flat_mask)
        flat_deployment_opportunity_cost = _masked_mean(deployment_opportunity_series, flat_mask)
        flat_deploy_value = _masked_mean(deploy_value_series, flat_mask)
        flat_deploy_gate = _masked_mean(decision_deploy_gate_series, flat_mask)
        flat_deploy_executability = _masked_mean(deploy_executability_series, flat_mask)
        held_deploy_executability = _masked_mean(deploy_executability_series, held_mask)
        avg_value_arbitration_target = float(value_arbitration_series.clip(0.0, 1.0).mean()) if len(value_arbitration_series) else 0.5
        avg_alpha_opportunity_value = float(alpha_opportunity_series.clip(0.0, 1.0).mean()) if len(alpha_opportunity_series) else 0.0
        avg_cash_defense_value = float(cash_defense_series.clip(0.0, 1.0).mean()) if len(cash_defense_series) else 0.0
        avg_deploy_value_target = float(deploy_value_series.clip(0.0, 1.0).mean()) if len(deploy_value_series) else 0.0
        avg_release_value_target = float(release_value_series.clip(0.0, 1.0).mean()) if len(release_value_series) else 0.0
        avg_defense_value_target = float(defense_value_series.clip(0.0, 1.0).mean()) if len(defense_value_series) else 0.0
        avg_deploy_gate_target = float(decision_deploy_gate_series.clip(0.0, 1.0).mean()) if len(decision_deploy_gate_series) else 0.0
        avg_release_gate_target = float(decision_release_gate_series.clip(0.0, 1.0).mean()) if len(decision_release_gate_series) else 0.0
        avg_stock_defense_gate_target = float(defense_gate_series.clip(0.0, 1.0).mean()) if len(defense_gate_series) else 0.0
        avg_deploy_executability_target = float(deploy_executability_series.clip(0.0, 1.0).mean()) if len(deploy_executability_series) else 0.0
        held_defense_gate = 0.0 if hierarchical_budget_mode else held_stock_defense_gate
        avg_defense_gate_target = 0.0 if hierarchical_budget_mode else avg_stock_defense_gate_target
        held_clipped_intent_risk = _masked_mean(clipped_intent_risk_series, held_mask)
        avg_clipped_intent_risk = float(clipped_intent_risk_series.clip(0.0, 1.0).mean()) if len(clipped_intent_risk_series) else 0.0
        held_exit_timing_pressure = _masked_mean(exit_timing_pressure_series, held_mask)
        held_exit_hazard = _masked_mean(exit_hazard_series, held_mask)
        flat_entry_quality = _masked_mean(entry_quality_series.clip(lower=0.0), flat_mask)
        current_gross_exposure = float(current.clip(lower=0.0).sum())
        portfolio_drawdown_20d = float(portfolio_context.get("portfolio_drawdown_20d", 0.0) or 0.0)
        recent_positive_share = float(portfolio_context.get("recent_positive_return_share_20d", 0.0) or 0.0)
        turnover_pressure = float(portfolio_context.get("turnover_pressure", 0.0) or 0.0)
        if constraint_only_budget_mode:
            budget_risk_off_score = float(
                np.clip(
                    0.34 * budget_model_defense_gate_signal
                    + 0.22 * budget_model_cash_timing_signal
                    + 0.14 * budget_model_risk_signal
                    + 0.10 * avg_cash_defense_value
                    + 0.08 * held_release_gate
                    + 0.06 * held_release_value
                    + max(-portfolio_drawdown_20d - 0.025, 0.0) / 0.09 * 0.14
                    + max(turnover_pressure - 0.65, 0.0) / 0.70 * 0.06
                    - recent_positive_share * 0.06
                    - budget_model_deploy_gate_signal * 0.08
                    - budget_model_deploy_value_signal * 0.04,
                    0.0,
                    1.0,
                )
            )
            budget_deploy_score = float(
                np.clip(
                    0.22 * flat_entry_action_share
                    + 0.14 * held_add_action_share
                    + max(flat_entry_quality, 0.0) / 0.24 * 0.18
                    + recent_positive_share * 0.14
                    + max(float(self.cash_weight) - 0.24, 0.0) / 0.45 * 0.16
                    + 0.18 * flat_alpha_opportunity_value
                    + 0.14 * flat_deployment_opportunity_cost
                    + 0.12 * flat_deploy_value
                    + 0.10 * flat_deploy_executability
                    + 0.14 * budget_model_deploy_gate_signal
                    + 0.10 * budget_model_deploy_value_signal
                    + 0.08 * budget_model_alpha_opportunity_signal
                    - budget_risk_off_score * 0.34
                    - budget_model_defense_gate_signal * 0.16
                    - budget_model_cash_timing_signal * 0.10
                    - held_clipped_intent_risk * 0.08,
                    0.0,
                    1.0,
                )
            )
        else:
            budget_risk_off_score = float(
                np.clip(
                    max(held_sell_pressure - 0.18, 0.0) / 0.45 * 0.30
                    + max(held_exit_timing_pressure - 0.22, 0.0) / 0.45 * 0.32
                    + max(held_exit_hazard - 0.22, 0.0) / 0.45 * 0.18
                    + max(held_lifecycle_sell_gate - 0.28, 0.0) / 0.55 * 0.18
                    + max(held_sell_rank - 0.55, 0.0) / 0.45 * 0.10
                    + held_sell_release_value * 0.16
                    + held_release_value * 0.12
                    + held_release_gate * 0.10
                    + held_defense_gate * 0.08
                    + budget_model_defense_gate_signal * (0.08 if hierarchical_budget_mode else 0.0)
                    + avg_cash_defense_value * 0.12
                    + held_sell_action_share * 0.26
                    + held_exit_action_share * 0.10
                    + max(-portfolio_drawdown_20d - 0.025, 0.0) / 0.09 * 0.18
                    + max(turnover_pressure - 0.65, 0.0) / 0.70 * 0.10
                    - recent_positive_share * 0.08
                    - avg_alpha_opportunity_value * 0.08
                    - avg_deploy_gate_target * 0.06,
                    0.0,
                    1.0,
                )
            )
            budget_deploy_score = float(
                np.clip(
                    flat_entry_action_share * 0.26
                    + held_add_action_share * 0.20
                    + max(flat_entry_quality, 0.0) / 0.24 * 0.20
                    + recent_positive_share * 0.16
                    + max(float(self.cash_weight) - 0.24, 0.0) / 0.45 * 0.18
                    + flat_alpha_opportunity_value * 0.18
                    + flat_deployment_opportunity_cost * 0.14
                    + flat_deploy_value * 0.14
                    + flat_deploy_gate * 0.12
                    + flat_deploy_executability * 0.10
                    + max(avg_value_arbitration_target - 0.50, 0.0) * 0.18
                    - budget_risk_off_score * 0.45
                    - avg_cash_defense_value * 0.14
                    - budget_model_defense_gate_signal * (0.12 if hierarchical_budget_mode else 0.0)
                    - avg_defense_gate_target * 0.10
                    - avg_release_gate_target * 0.08
                    - held_clipped_intent_risk * 0.14,
                    0.0,
                    1.0,
                )
            )
        portfolio_daily_receiver_target_count = int(portfolio_daily_receiver_target.sum())
        portfolio_daily_receiver_target_score = _masked_mean(
            portfolio_daily_receiver_score,
            portfolio_daily_receiver_target,
        )
        portfolio_daily_receiver_pressure = float(
            np.clip(
                portfolio_daily_receiver_target_count / max(float(self.max_positions), 1.0) * 0.50
                + max(portfolio_daily_receiver_target_score, 0.0) * 0.18
                + budget_deploy_score * 0.24
                + budget_model_alpha_focus_signal * 0.08,
                0.0,
                1.0,
            )
        )
        portfolio_daily_cash_score = float(
            np.clip(
                0.30 * budget_model_cash_timing_signal
                + 0.18 * budget_model_defense_gate_signal
                + 0.14 * budget_model_risk_signal
                + 0.14 * avg_cash_defense_value
                + 0.10 * max(-portfolio_drawdown_20d - 0.025, 0.0) / 0.09
                + 0.08 * max(turnover_pressure - 0.62, 0.0) / 0.70
                - 0.26 * portfolio_daily_receiver_pressure
                - 0.08 * recent_positive_share,
                0.0,
                1.0,
            )
        )
        if model_cash_score_series is not None and len(model_cash_score_series):
            model_cash_score_value = float(
                model_cash_score_series.replace([np.inf, -np.inf], np.nan).dropna().clip(0.0, 1.0).mean()
            )
            if np.isfinite(model_cash_score_value):
                portfolio_daily_cash_score = float(np.clip(0.60 * model_cash_score_value + 0.40 * portfolio_daily_cash_score, 0.0, 1.0))
        if model_unified_cash_score_series is not None and len(model_unified_cash_score_series):
            unified_cash_score_value = float(
                model_unified_cash_score_series.replace([np.inf, -np.inf], np.nan).dropna().clip(0.0, 1.0).mean()
            )
            if np.isfinite(unified_cash_score_value):
                portfolio_daily_cash_score = float(
                    np.clip(0.52 * unified_cash_score_value + 0.48 * portfolio_daily_cash_score, 0.0, 1.0)
                )
        if model_allocation_dead_branch_risk_series is not None or model_allocation_transfer_score_series is not None:
            allocation_dead_risk_value = float(
                portfolio_daily_allocation_dead_branch_risk.replace([np.inf, -np.inf], np.nan).dropna().clip(0.0, 1.0).mean()
            ) if len(portfolio_daily_allocation_dead_branch_risk) else 0.0
            allocation_transfer_value = float(
                portfolio_daily_allocation_transfer_score.replace([np.inf, -np.inf], np.nan).dropna().clip(0.0, 1.0).mean()
            ) if len(portfolio_daily_allocation_transfer_score) else 0.0
            portfolio_daily_cash_score = float(
                np.clip(
                    portfolio_daily_cash_score
                    + allocation_dead_risk_value * 0.10
                    - allocation_transfer_value * 0.08,
                    0.0,
                    1.0,
                )
            )
        if budget_calibration in {
            BUDGET_CALIBRATION_CASH_CONSTRAINT_PORTFOLIO_DAILY_RANKING_CASH_AWARE,
            BUDGET_CALIBRATION_CASH_CONSTRAINT_PORTFOLIO_DAILY_RANKING_SOURCE_EXEC,
            BUDGET_CALIBRATION_CASH_CONSTRAINT_PORTFOLIO_DAILY_RANKING_RECEIVER_EXEC,
        }:
            weak_receiver_quality = max(0.0, 0.18 - max(portfolio_daily_receiver_target_score, 0.0)) / 0.30
            weak_market_breadth = max(0.0, 0.52 - recent_positive_share) / 0.52
            high_exposure_pressure = max(0.0, current_gross - 0.70) / 0.30
            high_turnover_pressure = max(0.0, turnover_pressure - 0.45) / 0.55
            receiver_sparse_pressure = max(0.0, 2.0 - float(portfolio_daily_receiver_target_count)) / 2.0
            realized_drawdown_pressure = max(0.0, -portfolio_drawdown_20d - 0.005) / 0.08
            cash_competition_score = float(
                np.clip(
                    0.16 * weak_receiver_quality
                    + 0.14 * weak_market_breadth
                    + 0.12 * high_exposure_pressure
                    + 0.10 * high_turnover_pressure
                    + 0.10 * receiver_sparse_pressure
                    + 0.08 * realized_drawdown_pressure
                    + 0.08 * budget_model_defense_gate_signal
                    - 0.08 * portfolio_daily_receiver_pressure,
                    0.0,
                    1.0,
                )
            )
            portfolio_daily_cash_score = max(portfolio_daily_cash_score, cash_competition_score)
        portfolio_daily_cash_score_series = pd.Series(portfolio_daily_cash_score, index=prices.index, dtype=float)
        portfolio_daily_cash_threshold = (
            0.24
            if budget_calibration
            in {
                BUDGET_CALIBRATION_CASH_CONSTRAINT_PORTFOLIO_DAILY_RANKING_CASH_AWARE,
                BUDGET_CALIBRATION_CASH_CONSTRAINT_PORTFOLIO_DAILY_RANKING_SOURCE_EXEC,
                BUDGET_CALIBRATION_CASH_CONSTRAINT_PORTFOLIO_DAILY_RANKING_RECEIVER_EXEC,
            }
            else 0.58
        )
        portfolio_daily_cash_receiver_pressure_ceiling = (
            0.72
            if budget_calibration
            in {
                BUDGET_CALIBRATION_CASH_CONSTRAINT_PORTFOLIO_DAILY_RANKING_CASH_AWARE,
                BUDGET_CALIBRATION_CASH_CONSTRAINT_PORTFOLIO_DAILY_RANKING_SOURCE_EXEC,
                BUDGET_CALIBRATION_CASH_CONSTRAINT_PORTFOLIO_DAILY_RANKING_RECEIVER_EXEC,
            }
            else 0.48
        )
        portfolio_daily_cash_reserve_signal = bool(
            portfolio_daily_ranking_mode
            and portfolio_daily_cash_score >= portfolio_daily_cash_threshold
            and portfolio_daily_receiver_pressure < portfolio_daily_cash_receiver_pressure_ceiling
        )
        portfolio_daily_receiver_reference_score = 0.0
        portfolio_daily_receiver_scores = portfolio_daily_receiver_score.loc[portfolio_daily_receiver_target].replace(
            [np.inf, -np.inf],
            np.nan,
        ).dropna()
        if bool(len(portfolio_daily_receiver_scores)):
            portfolio_daily_receiver_reference_score = float(portfolio_daily_receiver_scores.min())
        portfolio_daily_source_gap = (
            pd.Series(portfolio_daily_receiver_reference_score, index=prices.index, dtype=float)
            - portfolio_daily_receiver_score.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        )
        portfolio_daily_source_min_release_delta = pd.concat(
            [
                pd.Series(0.0025, index=prices.index, dtype=float),
                current.clip(lower=0.0) * 0.018,
                pd.Series(position_cap_target * 0.012, index=prices.index, dtype=float),
            ],
            axis=1,
        ).max(axis=1)
        portfolio_daily_source_release_capacity = (
            current.clip(lower=0.0) / portfolio_daily_source_min_release_delta.clip(lower=1.0e-6)
        ).replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(0.0, 1.0).where(held_mask, 0.0)
        if model_source_release_capacity_series is not None:
            portfolio_daily_source_release_capacity = (
                0.58 * model_source_release_capacity_series.clip(0.0, 1.0)
                + 0.42 * portfolio_daily_source_release_capacity
            ).clip(0.0, 1.0)
        portfolio_daily_source_release_quality_observable = (
            release_value_series.clip(0.0, 1.0) * 0.24
            + sell_release_series.clip(0.0, 1.0) * 0.20
            + multi_horizon_forward_risk_series.clip(0.0, 1.0) * 0.18
            + decision_release_gate_series.clip(0.0, 1.0) * 0.12
            + portfolio_daily_cash_score_series.clip(0.0, 1.0) * 0.08
            + sell_rank_series.clip(0.0, 1.0) * 0.08
            + (1.0 - hold_continuation_series.clip(0.0, 1.0)) * 0.08
            - alpha_opportunity_series.clip(0.0, 1.0) * 0.14
            - large_upside_series.clip(0.0, 1.0) * 0.12
        ).clip(0.0, 1.0).where(held_mask, 0.0)
        portfolio_daily_source_release_quality = portfolio_daily_source_release_quality_observable.copy()
        if model_source_release_quality_series is not None:
            blended_source_release_quality = (
                0.66 * model_source_release_quality_series.clip(0.0, 1.0)
                + 0.34 * portfolio_daily_source_release_quality_observable
            ).clip(0.0, 1.0).where(held_mask, 0.0)
            observable_release_cap = (
                portfolio_daily_source_release_quality_observable
                + release_value_series.clip(0.0, 1.0) * 0.08
                + sell_release_series.clip(0.0, 1.0) * 0.06
                + decision_release_gate_series.clip(0.0, 1.0) * 0.05
                + portfolio_daily_cash_score_series.clip(0.0, 1.0) * 0.04
                + 0.12
            ).clip(0.0, 1.0).where(held_mask, 0.0)
            portfolio_daily_source_release_quality = pd.concat(
                [blended_source_release_quality, observable_release_cap],
                axis=1,
            ).min(axis=1).clip(0.0, 1.0).where(held_mask, 0.0)
        portfolio_daily_source_opportunity_cost = (
            direct_action_pair_source_opportunity_cost.clip(0.0, 1.0) * 0.34
            + hold_continuation_series.clip(0.0, 1.0) * 0.26
            + alpha_opportunity_series.clip(0.0, 1.0) * 0.22
            + deploy_value_series.clip(0.0, 1.0) * 0.12
            + deploy_executability_series.clip(0.0, 1.0) * 0.10
            + large_upside_series.clip(0.0, 1.0) * 0.10
            + (direct_action_keep_advantage_series.clip(lower=0.0, upper=0.18) / 0.18) * 0.08
            - release_value_series.clip(0.0, 1.0) * 0.10
            - portfolio_daily_cash_score_series.clip(0.0, 1.0) * 0.04
            - portfolio_daily_source_release_quality.clip(0.0, 1.0) * 0.12
        ).clip(0.0, 1.0).where(held_mask, 0.0)
        if model_source_opportunity_cost_series is not None:
            portfolio_daily_source_opportunity_cost = (
                0.64 * model_source_opportunity_cost_series.clip(0.0, 1.0)
                + 0.36 * portfolio_daily_source_opportunity_cost
            ).clip(0.0, 1.0).where(held_mask, 0.0)
        if unified_allocation_policy_mode:
            portfolio_daily_source_opportunity_cost = (
                portfolio_daily_source_opportunity_cost
                + portfolio_daily_unified_source_positive_forward_penalty * 0.20
                + portfolio_daily_unified_source_opportunity_cost_penalty * 0.16
                - portfolio_daily_unified_receiver_source_spread_reward * 0.10
            ).clip(0.0, 1.0).where(held_mask, 0.0)
        low_observable_source_release = (
            (0.22 - portfolio_daily_source_release_quality_observable).clip(lower=0.0, upper=0.22) / 0.22
        ).where(held_mask, 0.0)
        portfolio_daily_source_opportunity_cost = (
            portfolio_daily_source_opportunity_cost
            + low_observable_source_release * 0.18
            + hold_continuation_series.clip(0.0, 1.0) * 0.05
            + alpha_opportunity_series.clip(0.0, 1.0) * 0.06
            + large_upside_series.clip(0.0, 1.0) * 0.04
        ).clip(0.0, 1.0).where(held_mask, 0.0)
        portfolio_daily_source_executability = (
            direct_action_pair_source_release_score.replace([np.inf, -np.inf], np.nan).fillna(0.0) * 0.24
            + release_value_series.clip(0.0, 1.0) * 0.16
            + decision_release_gate_series.clip(0.0, 1.0) * 0.14
            + sell_release_series.clip(0.0, 1.0) * 0.12
            + portfolio_daily_source_gap.clip(lower=0.0, upper=0.24) * 0.40
            + portfolio_daily_source_release_capacity.clip(0.0, 1.0) * 0.06
            + (1.0 - portfolio_daily_source_opportunity_cost).clip(0.0, 1.0) * 0.16
            + portfolio_daily_source_release_quality.clip(0.0, 1.0) * 0.18
            + portfolio_daily_cash_score_series.clip(0.0, 1.0) * 0.06
            - hold_continuation_series.clip(0.0, 1.0) * 0.14
            - alpha_opportunity_series.clip(0.0, 1.0) * 0.12
            - deploy_executability_series.clip(0.0, 1.0) * 0.10
            - large_upside_series.clip(0.0, 1.0) * 0.06
        ).clip(0.0, 1.0).where(held_mask, 0.0)
        if model_source_executability_series is not None:
            portfolio_daily_source_executability = (
                0.58 * model_source_executability_series.clip(0.0, 1.0)
                + 0.42 * portfolio_daily_source_executability
            ).clip(0.0, 1.0)
        portfolio_daily_source_score = (
            portfolio_daily_source_executability.clip(0.0, 1.0) * 0.30
            + portfolio_daily_source_release_capacity.clip(0.0, 1.0) * 0.10
            + portfolio_daily_source_gap.clip(lower=-0.10, upper=0.24) * 0.94
            + (1.0 - portfolio_daily_source_opportunity_cost).clip(0.0, 1.0) * 0.30
            + portfolio_daily_source_release_quality.clip(0.0, 1.0) * 0.22
            + release_value_series.clip(0.0, 1.0) * 0.10
            + decision_release_gate_series.clip(0.0, 1.0) * 0.10
            + sell_release_series.clip(0.0, 1.0) * 0.08
            + exit_timing_pressure_series.clip(0.0, 1.0) * 0.06
            + portfolio_daily_cash_score_series.clip(0.0, 1.0) * 0.06
            - hold_continuation_series.clip(0.0, 1.0) * 0.14
            - alpha_opportunity_series.clip(0.0, 1.0) * 0.12
            - deploy_executability_series.clip(0.0, 1.0) * 0.10
            - large_upside_series.clip(0.0, 1.0) * 0.16
            - portfolio_daily_source_opportunity_cost.clip(0.0, 1.0) * 0.08
        )
        if model_source_score_series is not None:
            portfolio_daily_source_score = (
                0.54 * model_source_score_series.clip(0.0, 1.0)
                + 0.46 * portfolio_daily_source_score
                - 0.22 * low_observable_source_release
                - 0.10 * portfolio_daily_source_opportunity_cost.clip(0.0, 1.0)
            ).clip(lower=-0.20, upper=1.25)
        if model_unified_source_score_series is not None:
            unified_source_support = (
                portfolio_daily_unified_source_score
                - portfolio_daily_unified_source_positive_forward_penalty * 0.46
                - portfolio_daily_unified_source_opportunity_cost_penalty * 0.30
                + portfolio_daily_unified_receiver_source_spread_reward * 0.28
            ).clip(lower=-0.20, upper=1.25).where(held_mask, 0.0)
            portfolio_daily_source_score = (
                0.48 * portfolio_daily_source_score
                + 0.52 * unified_source_support
                + portfolio_daily_source_release_capacity.clip(0.0, 1.0) * 0.04
            ).clip(lower=-0.20, upper=1.25).where(held_mask, 0.0)
        fallback_source_forward_spread_score = (
            portfolio_daily_source_gap.clip(lower=0.0, upper=0.28) * 1.35
            + portfolio_daily_source_release_quality.clip(0.0, 1.0) * 0.20
            + (1.0 - portfolio_daily_source_opportunity_cost).clip(0.0, 1.0) * 0.16
            + multi_horizon_forward_risk_series.clip(0.0, 1.0) * 0.10
            + portfolio_daily_cash_score_series.clip(0.0, 1.0) * 0.08
            - hold_continuation_series.clip(0.0, 1.0) * 0.18
            - alpha_opportunity_series.clip(0.0, 1.0) * 0.16
            - large_upside_series.clip(0.0, 1.0) * 0.10
        ).clip(0.0, 1.0).where(held_mask, 0.0)
        if model_source_forward_spread_score_series is not None:
            portfolio_daily_source_forward_spread_score = (
                0.70 * model_source_forward_spread_score_series.clip(0.0, 1.0)
                + 0.30 * fallback_source_forward_spread_score
            ).clip(0.0, 1.0).where(held_mask, 0.0)
        else:
            portfolio_daily_source_forward_spread_score = (
                fallback_source_forward_spread_score * 0.55
            ).clip(0.0, 1.0).where(held_mask, 0.0)
        fallback_source_bad_forward_spread_risk = (
            portfolio_daily_source_opportunity_cost.clip(0.0, 1.0) * 0.24
            + hold_continuation_series.clip(0.0, 1.0) * 0.22
            + alpha_opportunity_series.clip(0.0, 1.0) * 0.20
            + large_upside_series.clip(0.0, 1.0) * 0.16
            + deploy_executability_series.clip(0.0, 1.0) * 0.08
            - portfolio_daily_source_release_quality.clip(0.0, 1.0) * 0.18
            - portfolio_daily_source_forward_spread_score.clip(0.0, 1.0) * 0.16
        ).clip(0.0, 1.0).where(held_mask, 0.0)
        if model_source_bad_forward_spread_risk_series is not None:
            portfolio_daily_source_bad_forward_spread_risk = (
                0.70 * model_source_bad_forward_spread_risk_series.clip(0.0, 1.0)
                + 0.30 * fallback_source_bad_forward_spread_risk
            ).clip(0.0, 1.0).where(held_mask, 0.0)
        else:
            portfolio_daily_source_bad_forward_spread_risk = (
                fallback_source_bad_forward_spread_risk + 0.10
            ).clip(0.0, 1.0).where(held_mask, 0.0)
        fallback_source_economic_release_score = (
            portfolio_daily_source_forward_spread_score.clip(0.0, 1.0) * 0.32
            + portfolio_daily_source_release_quality.clip(0.0, 1.0) * 0.20
            + portfolio_daily_source_score.clip(lower=0.0, upper=1.0) * 0.16
            + portfolio_daily_source_executability.clip(0.0, 1.0) * 0.12
            + (1.0 - portfolio_daily_source_opportunity_cost).clip(0.0, 1.0) * 0.10
            + portfolio_daily_funding_closure_score.clip(0.0, 1.0) * 0.06
            + portfolio_daily_cash_score_series.clip(0.0, 1.0) * 0.04
            - portfolio_daily_source_bad_forward_spread_risk.clip(0.0, 1.0) * 0.28
            - hold_continuation_series.clip(0.0, 1.0) * 0.12
            - alpha_opportunity_series.clip(0.0, 1.0) * 0.08
        ).clip(0.0, 1.0).where(held_mask, 0.0)
        if model_source_economic_release_score_series is not None:
            portfolio_daily_source_economic_release_score = (
                0.72 * model_source_economic_release_score_series.clip(0.0, 1.0)
                + 0.28 * fallback_source_economic_release_score
            ).clip(0.0, 1.0).where(held_mask, 0.0)
        else:
            portfolio_daily_source_economic_release_score = (
                fallback_source_economic_release_score * 0.60
            ).clip(0.0, 1.0).where(held_mask, 0.0)
        fallback_source_economic_block_risk = (
            portfolio_daily_source_bad_forward_spread_risk.clip(0.0, 1.0) * 0.34
            + portfolio_daily_source_opportunity_cost.clip(0.0, 1.0) * 0.20
            + hold_continuation_series.clip(0.0, 1.0) * 0.18
            + alpha_opportunity_series.clip(0.0, 1.0) * 0.14
            + large_upside_series.clip(0.0, 1.0) * 0.10
            - portfolio_daily_source_economic_release_score.clip(0.0, 1.0) * 0.22
            - portfolio_daily_source_forward_spread_score.clip(0.0, 1.0) * 0.14
        ).clip(0.0, 1.0).where(held_mask, 0.0)
        if model_source_economic_block_risk_series is not None:
            portfolio_daily_source_economic_block_risk = (
                0.72 * model_source_economic_block_risk_series.clip(0.0, 1.0)
                + 0.28 * fallback_source_economic_block_risk
            ).clip(0.0, 1.0).where(held_mask, 0.0)
        else:
            portfolio_daily_source_economic_block_risk = (
                fallback_source_economic_block_risk + 0.08
            ).clip(0.0, 1.0).where(held_mask, 0.0)
        fallback_source_forward_strength_brake_risk = (
            portfolio_daily_source_bad_forward_spread_risk.clip(0.0, 1.0) * 0.24
            + hold_continuation_series.clip(0.0, 1.0) * 0.22
            + alpha_opportunity_series.clip(0.0, 1.0) * 0.20
            + large_upside_series.clip(0.0, 1.0) * 0.16
            + multi_horizon_path_value_series.clip(0.0, 1.0) * 0.14
            + deploy_value_series.clip(0.0, 1.0) * 0.10
            + portfolio_daily_source_opportunity_cost.clip(0.0, 1.0) * 0.08
            - multi_horizon_forward_risk_series.clip(0.0, 1.0) * 0.18
            - portfolio_daily_source_forward_spread_score.clip(0.0, 1.0) * 0.16
            - portfolio_daily_source_release_quality.clip(0.0, 1.0) * 0.12
            - sell_release_series.clip(0.0, 1.0) * 0.08
            - portfolio_daily_cash_score_series.clip(0.0, 1.0) * 0.06
        ).clip(0.0, 1.0).where(held_mask, 0.0)
        if model_source_forward_strength_brake_risk_series is not None:
            portfolio_daily_source_forward_strength_brake_risk = (
                0.70 * model_source_forward_strength_brake_risk_series.clip(0.0, 1.0)
                + 0.30 * fallback_source_forward_strength_brake_risk
            ).clip(0.0, 1.0).where(held_mask, 0.0)
        else:
            portfolio_daily_source_forward_strength_brake_risk = (
                fallback_source_forward_strength_brake_risk + 0.08
            ).clip(0.0, 1.0).where(held_mask, 0.0)
        portfolio_daily_source_economic_block_risk = (
            portfolio_daily_source_economic_block_risk
            + portfolio_daily_source_forward_strength_brake_risk.clip(0.0, 1.0) * 0.16
            - portfolio_daily_source_economic_release_score.clip(0.0, 1.0) * 0.03
        ).clip(0.0, 1.0).where(held_mask, 0.0)
        portfolio_daily_source_economic_release_score = (
            portfolio_daily_source_economic_release_score
            - portfolio_daily_source_forward_strength_brake_risk.clip(0.0, 1.0) * 0.12
        ).clip(0.0, 1.0).where(held_mask, 0.0)
        portfolio_daily_source_opportunity_cost = (
            portfolio_daily_source_opportunity_cost
            + portfolio_daily_source_economic_block_risk.clip(0.0, 1.0) * 0.12
            + portfolio_daily_source_forward_strength_brake_risk.clip(0.0, 1.0) * 0.18
            - portfolio_daily_source_economic_release_score.clip(0.0, 1.0) * 0.06
        ).clip(0.0, 1.0).where(held_mask, 0.0)
        portfolio_daily_source_release_quality = (
            portfolio_daily_source_release_quality
            + portfolio_daily_source_economic_release_score.clip(0.0, 1.0) * 0.10
            - portfolio_daily_source_economic_block_risk.clip(0.0, 1.0) * 0.08
            - portfolio_daily_source_forward_strength_brake_risk.clip(0.0, 1.0) * 0.10
        ).clip(0.0, 1.0).where(held_mask, 0.0)
        portfolio_daily_source_score = (
            portfolio_daily_source_score
            + portfolio_daily_source_forward_spread_score.clip(0.0, 1.0) * 0.18
            + portfolio_daily_source_economic_release_score.clip(0.0, 1.0) * 0.24
            - portfolio_daily_source_bad_forward_spread_risk.clip(0.0, 1.0) * 0.18
            - portfolio_daily_source_economic_block_risk.clip(0.0, 1.0) * 0.22
            - portfolio_daily_source_forward_strength_brake_risk.clip(0.0, 1.0) * 0.22
        ).clip(lower=-0.20, upper=1.25).where(held_mask, 0.0)
        if model_allocation_transfer_score_series is not None or model_funding_closure_score_series is not None:
            portfolio_daily_source_opportunity_cost = (
                portfolio_daily_source_opportunity_cost
                + portfolio_daily_allocation_dead_branch_risk.clip(0.0, 1.0) * 0.06
                - portfolio_daily_funding_closure_score.clip(0.0, 1.0) * 0.04
            ).clip(0.0, 1.0).where(held_mask, 0.0)
            portfolio_daily_source_score = (
                portfolio_daily_source_score
                + portfolio_daily_allocation_transfer_score.clip(0.0, 1.0) * 0.10
                + portfolio_daily_funding_closure_score.clip(0.0, 1.0) * 0.06
                - portfolio_daily_allocation_dead_branch_risk.clip(0.0, 1.0) * 0.08
                - portfolio_daily_source_opportunity_cost.clip(0.0, 1.0) * 0.04
            ).clip(lower=-0.20, upper=1.25).where(held_mask, 0.0)
        allocation_source_funding_pressure = pd.Series(0.0, index=prices.index, dtype=float)
        if model_allocation_transfer_score_series is not None or model_funding_closure_score_series is not None:
            allocation_source_funding_pressure = (
                portfolio_daily_allocation_transfer_score.clip(0.0, 1.0) * 0.28
                + portfolio_daily_funding_closure_score.clip(0.0, 1.0) * 0.24
                + portfolio_daily_receiver_funding_coverage.clip(0.0, 1.0) * 0.14
                + pd.Series(portfolio_daily_receiver_pressure, index=prices.index, dtype=float).clip(0.0, 1.0) * 0.20
                + portfolio_daily_cash_score_series.clip(0.0, 1.0) * 0.08
                + portfolio_daily_source_gap.clip(lower=0.0, upper=0.28) * 0.68
                + portfolio_daily_source_economic_release_score.clip(0.0, 1.0) * 0.18
                - portfolio_daily_source_bad_forward_spread_risk.clip(0.0, 1.0) * 0.12
                - portfolio_daily_source_economic_block_risk.clip(0.0, 1.0) * 0.14
                - portfolio_daily_allocation_dead_branch_risk.clip(0.0, 1.0) * 0.08
            ).clip(0.0, 1.0).where(held_mask, 0.0)
            allocation_release_override = (
                (allocation_source_funding_pressure >= 0.08)
                & (portfolio_daily_source_gap >= -0.04)
                & (portfolio_daily_source_release_capacity >= 0.45)
                & (portfolio_daily_source_economic_release_score >= 0.08)
                & (portfolio_daily_source_economic_block_risk <= 0.80)
            )
            portfolio_daily_source_opportunity_cost = (
                portfolio_daily_source_opportunity_cost
                - allocation_source_funding_pressure * 0.22
                - portfolio_daily_source_gap.clip(lower=0.0, upper=0.28) * 0.30
            ).clip(0.0, 1.0).where(held_mask, 0.0)
            portfolio_daily_source_release_quality = pd.concat(
                [
                    portfolio_daily_source_release_quality,
                    (
                        allocation_source_funding_pressure * 0.48
                        + portfolio_daily_source_gap.clip(lower=0.0, upper=0.28) * 0.54
                        + portfolio_daily_cash_score_series.clip(0.0, 1.0) * 0.06
                    ).clip(0.0, 1.0).where(allocation_release_override, 0.0),
                ],
                axis=1,
            ).max(axis=1).clip(0.0, 1.0).where(held_mask, 0.0)
            portfolio_daily_source_executability = (
                portfolio_daily_source_executability
                + allocation_source_funding_pressure * 0.18
                + portfolio_daily_source_release_quality.clip(0.0, 1.0) * 0.08
                - portfolio_daily_source_opportunity_cost.clip(0.0, 1.0) * 0.04
            ).clip(0.0, 1.0).where(held_mask, 0.0)
        portfolio_daily_source_score = (
            portfolio_daily_source_score
            + allocation_source_funding_pressure * 0.36
            + portfolio_daily_source_gap.clip(lower=0.0, upper=0.28) * 0.76
            + portfolio_daily_source_release_quality.clip(0.0, 1.0) * 0.12
            + portfolio_daily_source_economic_release_score.clip(0.0, 1.0) * 0.18
            - portfolio_daily_source_bad_forward_spread_risk.clip(0.0, 1.0) * 0.12
            - portfolio_daily_source_economic_block_risk.clip(0.0, 1.0) * 0.16
            - portfolio_daily_source_opportunity_cost.clip(0.0, 1.0) * 0.02
        ).clip(lower=-0.20, upper=1.25).where(held_mask, 0.0)
        protected_source_release_override = (
            portfolio_daily_ranking_mode
            & held_mask
            & direct_action_funding_protected
            & (~direct_action_core_deploy_target)
            & (~portfolio_daily_receiver_target)
            & bool(portfolio_daily_receiver_target.any())
            & (current >= 0.012)
            & (direct_action_pair_opportunity_spread >= max(pair_min_spread, 0.200))
            & (direct_action_pair_source_release_score >= 0.160)
            & (direct_action_pair_source_opportunity_cost >= 0.800)
            & (portfolio_daily_source_gap >= 0.140)
            & (portfolio_daily_source_release_quality >= 0.21)
            & (portfolio_daily_source_economic_release_score >= 0.18)
            & (portfolio_daily_source_bad_forward_spread_risk <= 0.42)
            & (portfolio_daily_source_economic_block_risk <= 0.56)
            & (portfolio_daily_source_forward_strength_brake_risk <= 0.50)
            & (portfolio_daily_source_release_capacity >= 0.80)
            & (portfolio_daily_source_opportunity_cost >= 0.78)
            & (portfolio_daily_source_opportunity_cost <= 0.94)
            & (portfolio_daily_allocation_dead_branch_risk <= 0.30)
        )
        protected_source_release_boost = (
            direct_action_pair_opportunity_spread.clip(lower=0.0, upper=0.24) * 0.72
            + direct_action_pair_source_release_score.clip(lower=0.0, upper=0.32) * 0.38
            + portfolio_daily_source_gap.clip(lower=0.0, upper=0.28) * 0.24
            + allocation_source_funding_pressure.clip(0.0, 1.0) * 0.16
            - portfolio_daily_source_opportunity_cost.clip(lower=0.72, upper=1.0) * 0.08
        ).clip(0.0, 0.42).where(protected_source_release_override, 0.0)
        portfolio_daily_source_forward_strength_brake_pass = (
            (portfolio_daily_source_forward_strength_brake_risk <= 0.48)
            | (
                (portfolio_daily_source_forward_spread_score >= 0.34)
                & (portfolio_daily_source_gap >= 0.12)
                & (portfolio_daily_source_economic_release_score >= 0.18)
                & (portfolio_daily_source_forward_strength_brake_risk <= 0.62)
            )
            | (
                (portfolio_daily_cash_score_series.clip(0.0, 1.0) >= 0.58)
                & (multi_horizon_forward_risk_series.clip(0.0, 1.0) >= 0.50)
                & (portfolio_daily_source_economic_block_risk <= 0.62)
                & (portfolio_daily_source_forward_strength_brake_risk <= 0.62)
            )
            | (
                action_names.isin({"reduce", "exit"})
                & (release_value_series.clip(0.0, 1.0) >= 0.42)
                & (sell_release_series.clip(0.0, 1.0) >= 0.34)
                & (decision_release_gate_series >= decision_deploy_gate_series + 0.18)
                & (portfolio_daily_source_forward_strength_brake_risk <= 0.66)
            )
        )
        portfolio_daily_source_direct_release_relief_score = (
            direct_action_pair_source_release_score.clip(0.0, 1.0) * 0.54
            + (1.0 - direct_action_pair_source_opportunity_cost.clip(0.0, 1.0)) * 0.18
            + portfolio_daily_source_release_capacity.clip(0.0, 1.0) * 0.10
            + portfolio_daily_source_executability.clip(0.0, 1.0) * 0.08
            + ((0.36 - portfolio_daily_source_forward_strength_brake_risk).clip(lower=0.0, upper=0.36) / 0.36)
            * 0.10
            - portfolio_daily_source_economic_block_risk.clip(0.0, 1.0) * 0.12
            - portfolio_daily_source_bad_forward_spread_risk.clip(0.0, 1.0) * 0.08
        ).clip(0.0, 1.0).where(held_mask, 0.0)
        portfolio_daily_receiver_context_active = bool(portfolio_daily_receiver_target.any())
        portfolio_daily_source_direct_release_gap_pass = (
            (
                pd.Series(portfolio_daily_receiver_context_active, index=prices.index, dtype=bool)
                & (portfolio_daily_source_gap >= 0.020)
                & (portfolio_daily_source_bad_forward_spread_risk <= 0.46)
                & (portfolio_daily_source_economic_block_risk <= 0.60)
            )
            | (
                pd.Series(not portfolio_daily_receiver_context_active, index=prices.index, dtype=bool)
                & bool(portfolio_daily_cash_reserve_signal)
                & (portfolio_daily_source_gap >= -0.100)
                & (portfolio_daily_source_economic_release_score >= 0.20)
                & (portfolio_daily_source_bad_forward_spread_risk <= 0.42)
                & (portfolio_daily_source_economic_block_risk <= 0.52)
            )
        )
        portfolio_daily_source_forward_proxy_keep_risk = (
            multi_horizon_forward_value_series.clip(0.0, 1.0) * 0.34
            + multi_horizon_path_value_series.clip(0.0, 1.0) * 0.30
            + alpha_opportunity_series.clip(0.0, 1.0) * 0.18
            + deploy_value_series.clip(0.0, 1.0) * 0.12
            + release_value_series.clip(0.0, 1.0) * 0.06
        ).clip(0.0, 1.0).where(held_mask, 0.0)
        portfolio_daily_source_forward_proxy_pass = (
            (
                pd.Series(portfolio_daily_receiver_context_active, index=prices.index, dtype=bool)
                & (
                    (portfolio_daily_source_forward_proxy_keep_risk <= 0.245)
                    | (
                        (portfolio_daily_source_forward_proxy_keep_risk <= 0.280)
                        & (portfolio_daily_source_gap >= 0.450)
                        & (portfolio_daily_source_opportunity_cost <= 0.040)
                        & (portfolio_daily_source_economic_release_score >= 0.480)
                        & (portfolio_daily_source_release_capacity >= 0.920)
                    )
                )
            )
            | (
                pd.Series(not portfolio_daily_receiver_context_active, index=prices.index, dtype=bool)
                & (
                    (portfolio_daily_source_forward_proxy_keep_risk <= 0.320)
                    | (
                        bool(portfolio_daily_cash_reserve_signal)
                        & (portfolio_daily_source_forward_proxy_keep_risk <= 0.360)
                        & (portfolio_daily_source_gap >= 0.160)
                        & (portfolio_daily_source_economic_release_score >= 0.360)
                    )
                )
            )
            | (
                action_names.isin({"exit"})
                & (portfolio_daily_source_forward_proxy_keep_risk <= 0.300)
                & (decision_release_gate_series >= decision_deploy_gate_series + 0.220)
            )
        )
        portfolio_daily_source_direct_release_relief_pass = (
            held_mask
            & (~action_names.isin({"open", "add"}))
            & (direct_action_pair_source_release_score >= 0.56)
            & (direct_action_pair_source_opportunity_cost <= 0.12)
            & (portfolio_daily_source_forward_strength_brake_risk <= 0.38)
            & (portfolio_daily_source_release_capacity >= 0.70)
            & (portfolio_daily_source_direct_release_relief_score >= 0.42)
            & portfolio_daily_source_direct_release_gap_pass
        )
        portfolio_daily_source_release_quality = pd.concat(
            [
                portfolio_daily_source_release_quality,
                (portfolio_daily_source_direct_release_relief_score * 0.44).where(
                    portfolio_daily_source_direct_release_relief_pass,
                    0.0,
                ),
            ],
            axis=1,
        ).max(axis=1).clip(0.0, 1.0).where(held_mask, 0.0)
        portfolio_daily_source_economic_release_score = pd.concat(
            [
                portfolio_daily_source_economic_release_score,
                (portfolio_daily_source_direct_release_relief_score * 0.38).where(
                    portfolio_daily_source_direct_release_relief_pass,
                    0.0,
                ),
            ],
            axis=1,
        ).max(axis=1).clip(0.0, 1.0).where(held_mask, 0.0)
        portfolio_daily_source_economic_block_risk = (
            portfolio_daily_source_economic_block_risk
            - portfolio_daily_source_direct_release_relief_score.where(
                portfolio_daily_source_direct_release_relief_pass,
                0.0,
            )
            * 0.12
        ).clip(0.0, 1.0).where(held_mask, 0.0)
        direct_release_relief_cost_cap = (
            0.42 - portfolio_daily_source_direct_release_relief_score * 0.18
        ).clip(lower=0.12, upper=0.42)
        portfolio_daily_source_opportunity_cost = pd.concat(
            [
                portfolio_daily_source_opportunity_cost,
                direct_release_relief_cost_cap.where(
                    portfolio_daily_source_direct_release_relief_pass,
                    1.0,
                ),
            ],
            axis=1,
        ).min(axis=1).clip(0.0, 1.0).where(held_mask, 0.0)
        portfolio_daily_source_score = (
            portfolio_daily_source_score
            + portfolio_daily_source_direct_release_relief_score.where(
                portfolio_daily_source_direct_release_relief_pass,
                0.0,
            )
            * 0.44
        ).clip(lower=-0.20, upper=1.25).where(held_mask, 0.0)
        portfolio_daily_source_release_conviction = (
            portfolio_daily_source_score.clip(lower=-0.20, upper=1.25)
            + portfolio_daily_source_release_quality.clip(0.0, 1.0) * 0.24
            + portfolio_daily_source_economic_release_score.clip(0.0, 1.0) * 0.20
            + portfolio_daily_source_executability.clip(0.0, 1.0) * 0.12
            + portfolio_daily_source_release_capacity.clip(0.0, 1.0) * 0.08
            - portfolio_daily_source_opportunity_cost.clip(0.0, 1.0) * 0.35
            - portfolio_daily_source_forward_strength_brake_risk.clip(0.0, 1.0) * 0.24
            - portfolio_daily_source_bad_forward_spread_risk.clip(0.0, 1.0) * 0.22
            - portfolio_daily_source_economic_block_risk.clip(0.0, 1.0) * 0.18
            - portfolio_daily_source_forward_proxy_keep_risk.clip(0.0, 1.0) * 0.18
        ).clip(lower=-1.0, upper=1.0).where(held_mask, 0.0)
        portfolio_daily_source_release_conviction_pass = (
            (portfolio_daily_source_release_conviction >= 0.340)
            | (
                portfolio_daily_source_direct_release_relief_pass
                & (portfolio_daily_source_release_conviction >= 0.260)
            )
            | (
                direct_action_funding_release_authorized
                & (portfolio_daily_source_gap >= 0.080)
                & (portfolio_daily_source_release_conviction >= 0.300)
            )
            | (
                action_names.isin({"exit"})
                & (portfolio_daily_source_opportunity_cost <= 0.360)
                & (portfolio_daily_source_release_conviction >= 0.280)
            )
        )
        portfolio_daily_source_distribution_clean_pass = (
            (
                (portfolio_daily_source_gap >= 0.060)
                & (portfolio_daily_source_forward_strength_brake_risk <= 0.300)
                & (portfolio_daily_source_bad_forward_spread_risk <= 0.180)
                & (portfolio_daily_source_economic_block_risk <= 0.320)
                & (portfolio_daily_source_release_conviction >= 0.360)
            )
            | (
                (portfolio_daily_source_gap >= 0.025)
                & (portfolio_daily_source_bad_forward_spread_risk >= 0.100)
                & (portfolio_daily_source_bad_forward_spread_risk <= 0.180)
                & (portfolio_daily_source_forward_strength_brake_risk >= 0.180)
                & (portfolio_daily_source_forward_strength_brake_risk <= 0.300)
                & (portfolio_daily_source_economic_block_risk <= 0.320)
                & (portfolio_daily_source_release_conviction >= 0.360)
            )
            | (
                (portfolio_daily_source_gap >= 0.320)
                & (portfolio_daily_source_forward_spread_score >= 0.200)
                & (portfolio_daily_source_bad_forward_spread_risk <= 0.160)
                & (portfolio_daily_source_release_conviction >= 0.500)
            )
            | (
                portfolio_daily_source_direct_release_relief_pass
                & (portfolio_daily_source_gap >= 0.055)
                & (portfolio_daily_source_bad_forward_spread_risk <= 0.140)
                & (portfolio_daily_source_forward_strength_brake_risk <= 0.200)
                & (portfolio_daily_source_economic_block_risk <= 0.240)
                & (portfolio_daily_source_release_conviction >= 0.400)
            )
            | (
                action_names.isin({"exit"})
                & (decision_release_gate_series >= decision_deploy_gate_series + 0.220)
                & (portfolio_daily_source_gap >= 0.100)
                & (portfolio_daily_source_bad_forward_spread_risk <= 0.160)
                & (portfolio_daily_source_release_conviction >= 0.340)
            )
        )
        portfolio_daily_unified_source_candidate = pd.Series(False, index=prices.index, dtype=bool)
        if unified_allocation_policy_mode and model_unified_source_score_series is not None:
            portfolio_daily_unified_source_distribution_pass = (
                portfolio_daily_source_distribution_clean_pass
                | (
                    (portfolio_daily_source_gap >= 0.180)
                    & (portfolio_daily_unified_receiver_source_spread_reward >= 0.740)
                    & (portfolio_daily_unified_source_positive_forward_penalty <= 0.120)
                    & (portfolio_daily_unified_source_opportunity_cost_penalty <= 0.180)
                    & (portfolio_daily_source_bad_forward_spread_risk <= 0.120)
                    & (portfolio_daily_source_forward_strength_brake_risk <= 0.180)
                    & (portfolio_daily_source_forward_proxy_keep_risk <= 0.220)
                    & (portfolio_daily_source_economic_block_risk <= 0.240)
                    & (portfolio_daily_source_release_conviction >= 0.520)
                )
            )
            portfolio_daily_unified_source_candidate = (
                portfolio_daily_ranking_mode
                & held_mask
                & (~direct_action_core_deploy_target)
                & (current >= 0.012)
                & (
                    bool(portfolio_daily_receiver_target.any())
                    | portfolio_daily_cash_reserve_signal
                    | (budget_model_deploy_signal >= 0.46)
                    | (portfolio_daily_unified_receiver_source_spread_reward >= 0.10)
                )
                & (portfolio_daily_unified_source_score >= 0.160)
                & (portfolio_daily_source_score >= -0.040)
                & (portfolio_daily_source_release_capacity >= 0.45)
                & (portfolio_daily_source_opportunity_cost <= 0.740)
                & (portfolio_daily_source_economic_block_risk <= 0.780)
                & (portfolio_daily_source_forward_strength_brake_risk <= 0.520)
                & (portfolio_daily_source_forward_proxy_keep_risk <= 0.560)
                & (portfolio_daily_unified_source_positive_forward_penalty <= 0.360)
                & (portfolio_daily_unified_source_opportunity_cost_penalty <= 0.520)
                & portfolio_daily_unified_source_distribution_pass
            )
        portfolio_daily_source_low_keep_value_pass = (
            (
                (hold_continuation_series.clip(0.0, 1.0) <= 0.52)
                & (alpha_opportunity_series.clip(0.0, 1.0) <= 0.58)
            )
            | (
                (portfolio_daily_cash_score_series.clip(0.0, 1.0) >= 0.52)
                & (multi_horizon_forward_risk_series.clip(0.0, 1.0) >= 0.42)
            )
            | action_names.isin({"exit"})
        )
        portfolio_daily_source_observable_release_pass = (
            portfolio_daily_source_direct_release_relief_pass
            | (
                (portfolio_daily_source_direct_release_relief_score >= 0.42)
                & (direct_action_pair_source_opportunity_cost <= 0.12)
                & (portfolio_daily_source_forward_strength_brake_risk <= 0.38)
            )
            | (
                (portfolio_daily_source_release_quality_observable >= 0.24)
                & portfolio_daily_source_low_keep_value_pass
            )
            | (
                (release_value_series.clip(0.0, 1.0) >= 0.34)
                & (sell_release_series.clip(0.0, 1.0) >= 0.28)
                & (decision_release_gate_series >= decision_deploy_gate_series + 0.14)
                & portfolio_daily_source_low_keep_value_pass
            )
            | (
                action_names.isin({"reduce", "exit"})
                & (portfolio_daily_source_release_quality_observable >= 0.20)
                & (portfolio_daily_source_opportunity_cost <= 0.48)
                & portfolio_daily_source_low_keep_value_pass
            )
            | (
                (portfolio_daily_cash_score_series.clip(0.0, 1.0) >= 0.42)
                & (multi_horizon_forward_risk_series.clip(0.0, 1.0) >= 0.44)
                & (portfolio_daily_source_opportunity_cost <= 0.46)
                & (
                    (hold_continuation_series.clip(0.0, 1.0) <= 0.62)
                    | (portfolio_daily_cash_score_series.clip(0.0, 1.0) >= 0.60)
                )
            )
        )
        portfolio_daily_source_semantic_release_pass = (
            portfolio_daily_source_direct_release_relief_pass
            | action_names.isin({"reduce", "exit"})
            | (release_value_series.clip(0.0, 1.0) >= 0.30)
            | (sell_release_series.clip(0.0, 1.0) >= 0.32)
            | (
                (decision_release_gate_series >= decision_deploy_gate_series + 0.16)
                & (portfolio_daily_source_score >= 0.34)
                & (portfolio_daily_source_opportunity_cost <= 0.46)
            )
            | (
                (portfolio_daily_source_release_quality >= 0.30)
                & (portfolio_daily_source_score >= 0.18)
                & (portfolio_daily_source_executability >= 0.10)
                & (portfolio_daily_source_opportunity_cost <= 0.52)
            )
            | (
                (portfolio_daily_source_score >= 0.36)
                & (portfolio_daily_source_opportunity_cost <= 0.18)
                & (portfolio_daily_source_executability >= 0.15)
                & (portfolio_daily_source_release_capacity >= 0.80)
            )
            | (
                (allocation_source_funding_pressure >= 0.10)
                & (portfolio_daily_source_gap >= 0.04)
                & (portfolio_daily_source_release_quality >= 0.08)
                & (portfolio_daily_source_score >= -0.06)
                & (portfolio_daily_source_opportunity_cost <= 0.86)
                & (portfolio_daily_source_release_capacity >= 0.45)
                & (portfolio_daily_source_economic_release_score >= 0.08)
                & (portfolio_daily_source_economic_block_risk <= 0.80)
            )
        )
        portfolio_daily_source_quality_gap_pass = (
            portfolio_daily_source_direct_release_relief_pass
            | ((portfolio_daily_source_gap >= 0.10) & portfolio_daily_source_observable_release_pass)
            | (
                (portfolio_daily_source_release_quality >= 0.30)
                & portfolio_daily_source_observable_release_pass
                & (portfolio_daily_source_score >= 0.18)
                & (portfolio_daily_source_opportunity_cost <= 0.52)
            )
            | (
                (portfolio_daily_source_score >= 0.36)
                & portfolio_daily_source_observable_release_pass
                & (portfolio_daily_source_opportunity_cost <= 0.18)
                & (portfolio_daily_source_executability >= 0.15)
            )
            | (
                action_names.isin({"reduce", "exit"})
                & portfolio_daily_source_observable_release_pass
                & (portfolio_daily_source_score >= 0.28)
                & (portfolio_daily_source_opportunity_cost <= 0.66)
            )
            | (
                (allocation_source_funding_pressure >= 0.10)
                & (portfolio_daily_source_gap >= 0.04)
                & (portfolio_daily_source_release_quality >= 0.08)
                & (portfolio_daily_source_score >= -0.06)
                & (portfolio_daily_source_opportunity_cost <= 0.86)
                & (portfolio_daily_source_release_capacity >= 0.45)
                & (portfolio_daily_source_economic_release_score >= 0.08)
                & (portfolio_daily_source_economic_block_risk <= 0.80)
            )
        )
        portfolio_daily_source_recent_sell_days = pd.Series(
            {
                stock: (
                    float((signal_dt - pd.Timestamp(str(self.last_sell_dates.get(stock, "") or "")).normalize()).days)
                    if str(self.last_sell_dates.get(stock, "") or "").strip()
                    else 999.0
                )
                for stock in prices.index
            },
            index=prices.index,
            dtype=float,
        ).replace([np.inf, -np.inf], np.nan).fillna(999.0)
        portfolio_daily_source_repeat_release_pass = (
            (portfolio_daily_source_recent_sell_days > 4.0)
            | action_names.isin({"exit"})
            | (
                action_names.isin({"reduce"})
                & (portfolio_daily_source_gap >= 0.24)
                & (portfolio_daily_source_score >= 0.36)
                & (portfolio_daily_source_opportunity_cost <= 0.42)
            )
            | (
                portfolio_daily_source_distribution_clean_pass
                & (portfolio_daily_source_recent_sell_days > 2.0)
                & (portfolio_daily_source_score >= 0.320)
                & (portfolio_daily_source_opportunity_cost <= 0.340)
                & (portfolio_daily_source_release_capacity >= 0.50)
                & (portfolio_daily_source_economic_block_risk <= 0.320)
            )
        )
        portfolio_daily_source_candidate = (
            portfolio_daily_ranking_mode
            & held_mask
            & (~direct_action_core_deploy_target)
            & (current >= 0.012)
            & portfolio_daily_source_semantic_release_pass
            & portfolio_daily_source_observable_release_pass
            & portfolio_daily_source_quality_gap_pass
            & portfolio_daily_source_repeat_release_pass
            & portfolio_daily_source_forward_strength_brake_pass
            & portfolio_daily_source_forward_proxy_pass
            & portfolio_daily_source_release_conviction_pass
            & portfolio_daily_source_distribution_clean_pass
            & (
                (portfolio_daily_source_economic_release_score >= 0.10)
                | portfolio_daily_source_direct_release_relief_pass
                | (
                    portfolio_daily_cash_reserve_signal
                    & (portfolio_daily_source_economic_block_risk <= 0.68)
                    & (portfolio_daily_source_bad_forward_spread_risk <= 0.55)
                )
            )
            & (portfolio_daily_source_economic_block_risk <= 0.82)
            & (
                bool(portfolio_daily_receiver_target.any())
                | portfolio_daily_cash_reserve_signal
                | (budget_model_deploy_signal >= 0.58)
                | portfolio_daily_source_direct_release_relief_pass
            )
            & (
                ((portfolio_daily_source_score >= 0.165) & (portfolio_daily_source_opportunity_cost <= 0.600))
                | (
                    (portfolio_daily_source_release_quality >= 0.30)
                    & (portfolio_daily_source_score >= 0.18)
                    & (portfolio_daily_source_executability >= 0.10)
                    & (portfolio_daily_source_opportunity_cost <= 0.52)
                )
                | (
                    (portfolio_daily_source_executability >= 0.30)
                    & (portfolio_daily_source_release_capacity >= 0.50)
                    & (portfolio_daily_source_opportunity_cost <= 0.540)
                )
                | (action_names.isin({"reduce", "exit"}) & (portfolio_daily_source_opportunity_cost <= 0.680))
                | (
                    direct_action_funding_release_authorized
                    & (portfolio_daily_source_opportunity_cost <= 0.560)
                    & (portfolio_daily_source_gap >= 0.0)
                )
                | (
                    portfolio_daily_source_direct_release_relief_pass
                    & (portfolio_daily_source_score >= 0.12)
                    & (portfolio_daily_source_opportunity_cost <= 0.42)
                    & (portfolio_daily_source_economic_block_risk <= 0.58)
                )
                | (
                    (allocation_source_funding_pressure >= 0.10)
                    & (portfolio_daily_source_gap >= 0.04)
                    & (portfolio_daily_source_release_quality >= 0.08)
                    & (portfolio_daily_source_score >= -0.06)
                    & (portfolio_daily_source_opportunity_cost <= 0.86)
                    & (portfolio_daily_source_release_capacity >= 0.45)
                    & (portfolio_daily_source_economic_release_score >= 0.08)
                    & (portfolio_daily_source_economic_block_risk <= 0.80)
                )
            )
            & (
                (~direct_action_funding_protected)
                | ((portfolio_daily_source_score >= 0.255) & (portfolio_daily_source_opportunity_cost <= 0.560))
                | ((portfolio_daily_source_executability >= 0.38) & (portfolio_daily_source_opportunity_cost <= 0.520))
                | (action_names.isin({"reduce", "exit"}) & (portfolio_daily_source_opportunity_cost <= 0.620))
                | (
                    (allocation_source_funding_pressure >= 0.16)
                    & (portfolio_daily_source_gap >= 0.04)
                    & (portfolio_daily_source_opportunity_cost <= 0.82)
                    & (portfolio_daily_source_economic_release_score >= 0.12)
                    & (portfolio_daily_source_economic_block_risk <= 0.72)
                )
            )
            & (
                (portfolio_daily_source_opportunity_cost <= 0.720)
                | (
                    (allocation_source_funding_pressure >= 0.10)
                    & (portfolio_daily_source_gap >= 0.04)
                    & (portfolio_daily_source_opportunity_cost <= 0.86)
                )
            )
        ) | portfolio_daily_unified_source_candidate
        portfolio_daily_source_target = pd.Series(False, index=prices.index, dtype=bool)
        portfolio_daily_source_candidate_count = int(portfolio_daily_source_candidate.sum())
        if portfolio_daily_source_candidate_count > 0:
            source_limit_basis = portfolio_daily_receiver_target_count
            if portfolio_daily_cash_reserve_signal:
                source_limit_basis += 1
            source_limit = min(
                portfolio_daily_source_candidate_count,
                max(1, min(5, int(np.ceil(max(source_limit_basis, 1) * 1.35)))),
            )
            portfolio_daily_source_rank = portfolio_daily_source_score.where(portfolio_daily_source_candidate).rank(
                method="first",
                ascending=False,
            )
            portfolio_daily_source_target = portfolio_daily_source_candidate & (
                portfolio_daily_source_rank <= float(source_limit)
            )
        portfolio_daily_source_execution_pressure = pd.Series(0.0, index=prices.index, dtype=float)
        portfolio_daily_source_retention_floor = pd.Series(1.0, index=prices.index, dtype=float)
        if portfolio_daily_source_exec_guard_mode and bool(portfolio_daily_source_target.any()):
            portfolio_daily_source_execution_pressure = (
                pd.Series(0.52, index=prices.index, dtype=float)
                + portfolio_daily_source_gap.clip(lower=0.0, upper=0.24) * 1.55
                + portfolio_daily_source_score.clip(lower=0.0, upper=1.0) * 0.18
                + portfolio_daily_source_executability.clip(0.0, 1.0) * 0.14
                + portfolio_daily_source_release_capacity.clip(0.0, 1.0) * 0.05
                + portfolio_daily_source_release_quality.clip(0.0, 1.0) * 0.12
                + (1.0 - portfolio_daily_source_opportunity_cost).clip(0.0, 1.0) * 0.12
                + portfolio_daily_cash_score_series.clip(0.0, 1.0) * 0.08
                + pd.Series(portfolio_daily_receiver_pressure, index=prices.index, dtype=float).clip(0.0, 1.0) * 0.06
                - hold_continuation_series.clip(0.0, 1.0) * 0.08
                - alpha_opportunity_series.clip(0.0, 1.0) * 0.06
                - deploy_executability_series.clip(0.0, 1.0) * 0.04
                - portfolio_daily_source_opportunity_cost.clip(0.0, 1.0) * 0.12
            ).clip(lower=0.0, upper=1.0)
            portfolio_daily_source_execution_pressure = portfolio_daily_source_execution_pressure.where(
                portfolio_daily_source_target,
                0.0,
            )
            portfolio_daily_source_retention_floor = (
                pd.Series(0.84, index=prices.index, dtype=float)
                - portfolio_daily_source_execution_pressure.clip(0.0, 1.0) * 0.18
                - portfolio_daily_source_gap.clip(lower=0.0, upper=0.24) * 0.28
                - portfolio_daily_source_score.clip(lower=0.0, upper=1.0) * 0.035
                - portfolio_daily_source_executability.clip(0.0, 1.0) * 0.040
                - portfolio_daily_source_release_capacity.clip(0.0, 1.0) * 0.018
                - portfolio_daily_source_release_quality.clip(0.0, 1.0) * 0.030
                + portfolio_daily_source_opportunity_cost.clip(0.0, 1.0) * 0.080
                + hold_continuation_series.clip(0.0, 1.0) * 0.045
                + direct_action_keep_advantage_series.clip(lower=0.0, upper=0.18) * 0.080
                + alpha_opportunity_series.clip(0.0, 1.0) * 0.020
            ).clip(lower=0.64, upper=0.90)
            portfolio_daily_source_retention_floor = portfolio_daily_source_retention_floor.where(
                portfolio_daily_source_target,
                1.0,
            )
        if portfolio_daily_ranking_mode:
            direct_action_pair_reallocation_source = (
                direct_action_pair_reallocation_source | portfolio_daily_source_target
            )
            direct_action_reallocation_source = (
                direct_action_hold_reallocation_source | direct_action_pair_reallocation_source
            )
        if budget_calibration == BUDGET_CALIBRATION_CASH_EXIT:
            risk_cut = budget_risk_off_score * (0.08 + current_gross_exposure * 0.16)
            deploy_boost = budget_deploy_score * 0.055 if budget_risk_off_score < 0.35 else 0.0
            gross_exposure_target = float(
                np.clip(
                    gross_exposure_target - risk_cut + deploy_boost,
                    0.18,
                    min(0.92, max(gross_exposure_target_raw + 0.06, 0.34)),
                )
            )
            candidate_budget = int(
                np.clip(
                    round(candidate_budget - budget_risk_off_score * 2.0 + budget_deploy_score * 1.5),
                    1,
                    int(self.max_positions),
                )
            )
            turnover_budget = float(
                np.clip(
                    turnover_budget + budget_risk_off_score * 0.16 + held_exit_action_share * 0.06 - budget_deploy_score * 0.025,
                    0.08,
                    1.00,
                )
            )
            position_cap_target = float(
                np.clip(position_cap_target - budget_risk_off_score * 0.025 + budget_deploy_score * 0.006, 0.05, 0.35)
            )
        elif budget_calibration in {
            BUDGET_CALIBRATION_CASH_CONSTRAINT,
            BUDGET_CALIBRATION_CASH_CONSTRAINT_DEPLOY,
            BUDGET_CALIBRATION_CASH_CONSTRAINT_SELL_SOURCE,
        }:
            portfolio_constraint_pressure = float(
                np.clip(
                    0.42 * budget_model_defense_gate_signal
                    + 0.24 * budget_model_cash_timing_signal
                    + 0.12 * budget_model_risk_signal
                    + 0.10 * avg_cash_defense_value
                    + 0.08 * max(-portfolio_drawdown_20d - 0.025, 0.0) / 0.09,
                    0.0,
                    1.0,
                )
            )
            portfolio_release_pressure = float(
                np.clip(
                    0.40 * held_release_gate
                    + 0.26 * held_release_value
                    + 0.18 * held_sell_release_value
                    + 0.16 * max(held_exit_timing_pressure - 0.20, 0.0),
                    0.0,
                    1.0,
                )
            )
            portfolio_deploy_pressure = float(
                np.clip(
                    0.36 * budget_model_deploy_gate_signal
                    + 0.24 * budget_model_deploy_value_signal
                    + 0.18 * flat_alpha_opportunity_value
                    + 0.12 * flat_deployment_opportunity_cost
                    + (0.14 * flat_deploy_executability if deploy_executability_constraint_mode else 0.0)
                    + 0.10 * budget_model_alpha_opportunity_signal,
                    0.0,
                    1.0,
                )
            )
            risk_cut = (
                portfolio_constraint_pressure * (0.09 + current_gross_exposure * (0.14 if deploy_executability_constraint_mode else 0.16))
                + portfolio_release_pressure * 0.04
                - portfolio_deploy_pressure * (0.05 if deploy_executability_constraint_mode else 0.03)
            )
            deploy_boost = (
                portfolio_deploy_pressure * (0.070 if deploy_executability_constraint_mode else 0.055)
                + budget_model_alpha_focus_signal * 0.018
                if portfolio_constraint_pressure < 0.40 and budget_model_cash_timing_signal < 0.46
                else 0.0
            )
            gross_exposure_target = float(
                np.clip(
                    gross_exposure_target - risk_cut + deploy_boost,
                    0.20 if deploy_executability_constraint_mode else 0.18,
                    min(0.94 if deploy_executability_constraint_mode else 0.92, max(gross_exposure_target_raw + 0.05, 0.32)),
                )
            )
            candidate_budget = int(
                np.clip(
                    round(
                        candidate_budget
                        - portfolio_constraint_pressure * 2.2
                        - portfolio_release_pressure * 0.8
                        + portfolio_deploy_pressure * (2.2 if deploy_executability_constraint_mode else 1.8)
                        + budget_model_alpha_focus_signal * 0.5
                    ),
                    1,
                    int(self.max_positions),
                )
            )
            turnover_budget = float(
                np.clip(
                    turnover_budget
                    + portfolio_release_pressure * 0.12
                    + portfolio_constraint_pressure * 0.08
                    - portfolio_deploy_pressure * (0.01 if deploy_executability_constraint_mode else 0.02),
                    0.08,
                    1.00,
                )
            )
            position_cap_target = float(
                np.clip(
                    position_cap_target
                    - portfolio_constraint_pressure * 0.028
                    + portfolio_deploy_pressure * (0.014 if deploy_executability_constraint_mode else 0.010),
                    0.05,
                    0.34 if deploy_executability_constraint_mode else 0.32,
                )
            )
        elif portfolio_daily_ranking_mode:
            portfolio_source_pressure = float(
                np.clip(
                    int(portfolio_daily_source_target.sum()) / max(float(self.max_positions), 1.0) * 0.46
                    + max(_masked_mean(portfolio_daily_source_score, portfolio_daily_source_target), 0.0) * 0.22
                    + max(_masked_mean(portfolio_daily_source_release_quality, portfolio_daily_source_target), 0.0) * 0.18
                    + max(_masked_mean(portfolio_daily_source_executability, portfolio_daily_source_target), 0.0) * 0.18
                    + max(_masked_mean(portfolio_daily_source_gap, portfolio_daily_source_target), 0.0) * 2.8,
                    0.0,
                    1.0,
                )
            )
            if portfolio_daily_source_exec_guard_mode:
                portfolio_source_pressure = float(
                    np.clip(
                        portfolio_source_pressure
                        + max(_masked_mean(portfolio_daily_source_execution_pressure, portfolio_daily_source_target), 0.0)
                        * 0.12,
                        0.0,
                        1.0,
                    )
                )
            portfolio_constraint_pressure = float(
                np.clip(
                    0.34 * budget_model_cash_timing_signal
                    + 0.20 * budget_model_defense_gate_signal
                    + 0.14 * budget_model_risk_signal
                    + 0.12 * avg_cash_defense_value
                    + 0.10 * max(-portfolio_drawdown_20d - 0.025, 0.0) / 0.09
                    + 0.06 * max(turnover_pressure - 0.60, 0.0) / 0.70
                    + 0.04 * float(portfolio_daily_cash_reserve_signal),
                    0.0,
                    1.0,
                )
            )
            ranking_cash_cut = max(
                0.0,
                portfolio_constraint_pressure
                + portfolio_daily_cash_score * 0.38
                - portfolio_daily_receiver_pressure * 0.52
                - portfolio_source_pressure * 0.16,
            )
            risk_cut = (
                portfolio_constraint_pressure * (0.052 + current_gross_exposure * 0.094)
                + ranking_cash_cut * (0.046 + current_gross_exposure * 0.082)
            )
            deploy_boost = (
                portfolio_daily_receiver_pressure * 0.076
                + portfolio_source_pressure * 0.022
                if portfolio_daily_cash_score < 0.54 and budget_model_cash_timing_signal < 0.52
                else 0.0
            )
            gross_exposure_target = float(
                np.clip(
                    gross_exposure_target - risk_cut + deploy_boost,
                    0.20,
                    min(0.94, max(gross_exposure_target_raw + 0.050, 0.32)),
                )
            )
            candidate_budget = int(
                np.clip(
                    round(
                        candidate_budget
                        - portfolio_constraint_pressure * 1.6
                        - ranking_cash_cut * 1.1
                        + portfolio_daily_receiver_pressure * 2.4
                        + portfolio_source_pressure * 0.7
                    ),
                    1,
                    int(self.max_positions),
                )
            )
            turnover_budget = float(
                np.clip(
                    turnover_budget
                    + portfolio_source_pressure * 0.13
                    + portfolio_daily_receiver_pressure * 0.035
                    + portfolio_constraint_pressure * 0.035
                    - portfolio_daily_cash_score * 0.018,
                    0.08,
                    1.00,
                )
            )
            position_cap_target = float(
                np.clip(
                    position_cap_target
                    - portfolio_constraint_pressure * 0.016
                    - portfolio_daily_cash_score * 0.006
                    + portfolio_daily_receiver_pressure * 0.012,
                    0.05,
                    0.34,
                )
            )
        elif direct_action_pair_cost_guard_mode:
            pair_core_pressure = float(
                np.clip(
                    int(direct_action_core_deploy_target.sum()) / max(float(self.max_positions), 1.0) * 0.62
                    + budget_deploy_score * 0.30
                    + budget_model_alpha_focus_signal * 0.08,
                    0.0,
                    1.0,
                )
            )
            pair_guard_pass_count = int(direct_action_pair_cost_guard_pass.sum())
            pair_source_spread_mean = (
                float(direct_action_pair_opportunity_spread.loc[direct_action_pair_cost_guard_pass].mean())
                if pair_guard_pass_count > 0
                else 0.0
            )
            pair_source_pressure = float(
                np.clip(
                    pair_guard_pass_count / max(float(self.max_positions), 1.0) * 0.42
                    + _masked_mean(direct_action_pair_source_release_score, direct_action_pair_cost_guard_pass) * 0.30
                    + max(pair_source_spread_mean, 0.0) * 4.0,
                    0.0,
                    1.0,
                )
            )
            pair_constraint_pressure = float(
                np.clip(
                    0.34 * budget_model_cash_timing_signal
                    + 0.24 * budget_model_defense_gate_signal
                    + 0.14 * budget_model_risk_signal
                    + 0.12 * avg_cash_defense_value
                    + 0.10 * max(-portfolio_drawdown_20d - 0.025, 0.0) / 0.09
                    + 0.06 * max(turnover_pressure - 0.60, 0.0) / 0.70,
                    0.0,
                    1.0,
                )
            )
            pair_cash_cut = max(0.0, pair_constraint_pressure - pair_core_pressure * 0.46 - pair_source_pressure * 0.18)
            risk_cut = (
                pair_constraint_pressure * (0.060 + current_gross_exposure * 0.105)
                + pair_cash_cut * (0.040 + current_gross_exposure * 0.080)
            )
            deploy_boost = (
                pair_core_pressure * 0.072 + pair_source_pressure * 0.018
                if pair_constraint_pressure < 0.44 and budget_model_cash_timing_signal < 0.48
                else 0.0
            )
            gross_exposure_target = float(
                np.clip(
                    gross_exposure_target - risk_cut + deploy_boost,
                    0.20,
                    min(0.94, max(gross_exposure_target_raw + 0.045, 0.32)),
                )
            )
            candidate_budget = int(
                np.clip(
                    round(
                        candidate_budget
                        - pair_constraint_pressure * 1.8
                        - pair_cash_cut * 1.2
                        + pair_core_pressure * 2.1
                        + pair_source_pressure * 0.6
                    ),
                    1,
                    int(self.max_positions),
                )
            )
            turnover_budget = float(
                np.clip(
                    turnover_budget
                    + pair_source_pressure * 0.10
                    + pair_constraint_pressure * 0.05
                    - pair_core_pressure * 0.015,
                    0.08,
                    1.00,
                )
            )
            position_cap_target = float(
                np.clip(
                    position_cap_target
                    - pair_constraint_pressure * 0.018
                    + pair_core_pressure * 0.010,
                    0.05,
                    0.34,
                )
            )
        elif budget_calibration in {BUDGET_CALIBRATION_CASH_TRANSLATION, BUDGET_CALIBRATION_CASH_TRANSLATION_SELL}:
            positive_gap = max(budget_model_risk_deploy_gap, 0.0)
            lifecycle_sell_pressure = float(
                np.clip(
                    0.34 * held_sell_attribution
                    + 0.24 * held_lifecycle_sell_gate
                    + 0.14 * held_sell_rank
                    + 0.16 * held_sell_release_value
                    + 0.12 * held_cash_defense_value
                    - 0.10 * held_hold_continuation_value,
                    0.0,
                    1.0,
                )
            )
            sell_priority_bonus = lifecycle_sell_pressure * 0.08 if use_sell_priority_guard else 0.0
            risk_cut = (
                budget_risk_off_score * (0.08 + current_gross_exposure * 0.14)
                + budget_model_cash_timing_signal * 0.12
                + budget_model_cash_defense_signal * 0.05
                + budget_model_defense_gate_signal * 0.045
                + budget_model_release_gate_signal * 0.035
                + positive_gap * 0.08
                + sell_priority_bonus
                - budget_model_alpha_opportunity_signal * 0.035
                - budget_model_deploy_gate_signal * 0.050
                - budget_model_deploy_value_signal * 0.030
                - max(budget_model_value_arbitration_signal - 0.50, 0.0) * 0.045
            )
            deploy_boost = (
                budget_deploy_score * 0.045
                + budget_model_alpha_focus_signal * 0.015
                + budget_model_alpha_opportunity_signal * 0.020
                + budget_model_deploy_gate_signal * 0.035
                + budget_model_deploy_value_signal * 0.025
                + max(budget_model_value_arbitration_signal - 0.50, 0.0) * 0.025
                if budget_risk_off_score < 0.38 and budget_model_cash_timing_signal < 0.42 and budget_model_defense_gate_signal < 0.46
                else 0.0
            )
            gross_exposure_target = float(
                np.clip(
                    gross_exposure_target - risk_cut + deploy_boost,
                    0.16,
                    min(0.90, max(gross_exposure_target_raw + 0.04, 0.30)),
                )
            )
            candidate_budget = int(
                np.clip(
                    round(
                        candidate_budget
                        - budget_risk_off_score * 2.2
                        - budget_model_cash_timing_signal * 2.0
                        - positive_gap * 1.2
                        - lifecycle_sell_pressure * (0.8 if use_sell_priority_guard else 0.0)
                        + budget_deploy_score * 1.4
                        + budget_model_alpha_focus_signal * 0.6
                    ),
                    1,
                    int(self.max_positions),
                )
            )
            turnover_budget = float(
                np.clip(
                    turnover_budget
                    + budget_risk_off_score * 0.18
                    + budget_model_cash_timing_signal * 0.10
                    + held_exit_action_share * 0.07
                    + lifecycle_sell_pressure * (0.08 if use_sell_priority_guard else 0.0)
                    - budget_deploy_score * 0.03,
                    0.08,
                    1.00,
                )
            )
            position_cap_target = float(
                np.clip(
                    position_cap_target
                    - budget_risk_off_score * 0.030
                    - budget_model_cash_timing_signal * 0.012
                    - lifecycle_sell_pressure * (0.008 if use_sell_priority_guard else 0.0)
                    + budget_deploy_score * 0.008
                    + budget_model_alpha_focus_signal * 0.006,
                    0.05,
                    0.32,
                )
            )
        portfolio_daily_receiver_add_headroom = (
            pd.Series(float(position_cap_target), index=prices.index, dtype=float) - current
        )
        portfolio_daily_receiver_min_add_delta = pd.Series(0.0, index=prices.index, dtype=float)
        portfolio_daily_receiver_exec_guarded = pd.Series(False, index=prices.index, dtype=bool)
        portfolio_daily_receiver_exec_guard_reason = pd.Series("none", index=prices.index, dtype=object)
        if portfolio_daily_receiver_exec_guard_mode and bool(portfolio_daily_receiver_target.any()):
            portfolio_daily_receiver_min_add_delta = pd.concat(
                [
                    pd.Series(DEFAULT_EXECUTION_DEADBAND_ABS * 2.0, index=prices.index, dtype=float),
                    current.clip(lower=0.0) * 0.025,
                    pd.Series(float(position_cap_target) * 0.018, index=prices.index, dtype=float),
                    pd.Series(0.0025, index=prices.index, dtype=float),
                ],
                axis=1,
            ).max(axis=1)
            receiver_add_no_headroom = (
                portfolio_daily_receiver_target
                & held_mask
                & action_names.eq("add")
                & (portfolio_daily_receiver_add_headroom < portfolio_daily_receiver_min_add_delta)
            )
            if bool(receiver_add_no_headroom.any()):
                portfolio_daily_receiver_exec_guarded = receiver_add_no_headroom
                portfolio_daily_receiver_exec_guard_reason = portfolio_daily_receiver_exec_guard_reason.where(
                    ~receiver_add_no_headroom,
                    np.where(
                        portfolio_daily_receiver_add_headroom <= 1.0e-8,
                        "no_position_cap_headroom",
                        "insufficient_min_add_headroom",
                    ),
                )
                portfolio_daily_receiver_target = portfolio_daily_receiver_target & (~receiver_add_no_headroom)
                direct_action_core_deploy_target = direct_action_core_deploy_target & (~receiver_add_no_headroom)
                direct_action_add_authorized = direct_action_add_authorized & (~receiver_add_no_headroom)
                direct_action_deploy_authorized = direct_action_add_authorized | direct_action_open_authorized
                portfolio_daily_receiver_target_count = int(portfolio_daily_receiver_target.sum())
        execution_deadband_abs = float(
            np.clip(
                max(
                    DEFAULT_EXECUTION_DEADBAND_ABS,
                    min(position_cap_target * 0.02, turnover_budget * 0.01),
                ),
                DEFAULT_EXECUTION_DEADBAND_ABS,
                0.0030,
            )
        )
        execution_deadband_rel = float(
            np.clip(
                DEFAULT_EXECUTION_DEADBAND_REL
                + hold_bias_target * 0.015
                + exit_patience_target * 0.010,
                DEFAULT_EXECUTION_DEADBAND_REL,
                0.06,
            )
        )

        def _days_since(mapping: dict[str, str], stock: str) -> float:
            raw_value = str(mapping.get(stock, "") or "").strip()
            if not raw_value:
                return 999.0
            try:
                return float((signal_dt - pd.Timestamp(raw_value).normalize()).days)
            except Exception:
                return 999.0

        desired_strength = pd.Series(0.0, index=prices.index, dtype=float)
        forced_zero = pd.Series(False, index=prices.index, dtype=bool)
        protected_floor = pd.Series(0.0, index=prices.index, dtype=float)
        weak_tail_zero_candidate = pd.Series(False, index=prices.index, dtype=bool)
        exit_timing_pressure_values = pd.Series(0.0, index=prices.index, dtype=float)
        for stock in prices.index:
            action = str(policy.at[stock, "action_label"] or "skip").strip().lower()
            if bool(portfolio_daily_receiver_target.get(stock, False)):
                action = "add" if float(current.get(stock, 0.0)) > 1e-8 else "open"
            strength = float(policy.at[stock, "action_strength"] or 0.0)
            delta_hint = float(policy.at[stock, "target_delta_hint"] or 0.0)
            hold_boost = float(policy.at[stock, "hold_boost"] or 0.0)
            entry_quality = float(policy.at[stock, "entry_quality"] or 0.0) if "entry_quality" in policy.columns else 0.0
            hold_quality = float(policy.at[stock, "hold_quality"] or 0.0) if "hold_quality" in policy.columns else 0.0
            add_quality = float(policy.at[stock, "add_quality"] or 0.0) if "add_quality" in policy.columns else 0.0
            reduce_quality = float(policy.at[stock, "reduce_quality"] or 0.0) if "reduce_quality" in policy.columns else 0.0
            exit_urgency = float(policy.at[stock, "exit_urgency"] or 0.0) if "exit_urgency" in policy.columns else 0.0
            reduce_fraction = float(policy.at[stock, "reduce_fraction"] or 0.0) if "reduce_fraction" in policy.columns else max(-delta_hint, 0.0)
            exit_hazard = float(policy.at[stock, "exit_hazard"] or 0.0) if "exit_hazard" in policy.columns else float(np.clip(exit_urgency, 0.0, 1.0))
            sell_pressure = (
                float(policy.at[stock, "sell_pressure"] or 0.0)
                if "sell_pressure" in policy.columns
                else float(np.clip(0.58 * reduce_fraction + 0.42 * exit_hazard, 0.0, 1.0))
            )
            sell_attribution_score = (
                float(policy.at[stock, "sell_attribution_score"] or 0.0)
                if "sell_attribution_score" in policy.columns
                else float(np.clip(reduce_quality * 0.42 + exit_hazard * 0.20 - hold_quality * 0.16, 0.0, 1.0))
            )
            sell_rank_score = (
                float(policy.at[stock, "sell_rank_score"] or 0.0)
                if "sell_rank_score" in policy.columns
                else sell_attribution_score
            )
            lifecycle_sell_gate = (
                float(policy.at[stock, "lifecycle_sell_gate"] or 0.0)
                if "lifecycle_sell_gate" in policy.columns
                else float(np.clip(0.58 * sell_attribution_score + 0.22 * reduce_quality + 0.20 * exit_hazard, 0.0, 1.0))
            )
            clipped_intent_risk = (
                float(policy.at[stock, "clipped_intent_risk"] or 0.0)
                if "clipped_intent_risk" in policy.columns
                else 0.0
            )
            large_upside_1d_target = (
                float(policy.at[stock, "large_upside_1d_target"] or 0.0)
                if "large_upside_1d_target" in policy.columns
                else 0.0
            )
            alpha_opportunity_value = (
                float(policy.at[stock, "alpha_opportunity_value"] or 0.0)
                if "alpha_opportunity_value" in policy.columns
                else 0.0
            )
            hold_continuation_value = (
                float(policy.at[stock, "hold_continuation_value"] or 0.0)
                if "hold_continuation_value" in policy.columns
                else 0.0
            )
            sell_release_value = (
                float(policy.at[stock, "sell_release_value"] or 0.0)
                if "sell_release_value" in policy.columns
                else 0.0
            )
            cash_defense_value = (
                float(policy.at[stock, "cash_defense_value"] or 0.0)
                if "cash_defense_value" in policy.columns
                else 0.0
            )
            deployment_opportunity_cost = (
                float(policy.at[stock, "deployment_opportunity_cost"] or 0.0)
                if "deployment_opportunity_cost" in policy.columns
                else 0.0
            )
            risk_adjusted_action_value = (
                float(policy.at[stock, "risk_adjusted_action_value"] or 0.0)
                if "risk_adjusted_action_value" in policy.columns
                else 0.0
            )
            multi_horizon_forward_value = (
                float(policy.at[stock, "multi_horizon_forward_value"] or 0.0)
                if "multi_horizon_forward_value" in policy.columns
                else 0.0
            )
            multi_horizon_forward_risk = (
                float(policy.at[stock, "multi_horizon_forward_risk"] or 0.0)
                if "multi_horizon_forward_risk" in policy.columns
                else 0.0
            )
            multi_horizon_path_value = (
                float(policy.at[stock, "multi_horizon_path_value"] or 0.0)
                if "multi_horizon_path_value" in policy.columns
                else 0.0
            )
            open_action_value = (
                float(policy.at[stock, "open_action_value"] or 0.0)
                if "open_action_value" in policy.columns
                else 0.0
            )
            add_action_value = (
                float(policy.at[stock, "add_action_value"] or 0.0)
                if "add_action_value" in policy.columns
                else 0.0
            )
            hold_action_value = (
                float(policy.at[stock, "hold_action_value"] or 0.0)
                if "hold_action_value" in policy.columns
                else 0.0
            )
            reduce_action_value = (
                float(policy.at[stock, "reduce_action_value"] or 0.0)
                if "reduce_action_value" in policy.columns
                else 0.0
            )
            exit_action_value = (
                float(policy.at[stock, "exit_action_value"] or 0.0)
                if "exit_action_value" in policy.columns
                else 0.0
            )
            action_value_consistency_target = (
                float(policy.at[stock, "action_value_consistency_target"] or 0.5)
                if "action_value_consistency_target" in policy.columns
                else 0.5
            )
            value_arbitration_target = (
                float(policy.at[stock, "value_arbitration_target"] or 0.5)
                if "value_arbitration_target" in policy.columns
                else 0.5
            )
            deploy_value_target = (
                float(policy.at[stock, "deploy_value_target"] or 0.0)
                if "deploy_value_target" in policy.columns
                else float(np.clip(0.56 * deployment_opportunity_cost + 0.34 * alpha_opportunity_value + 0.10 * large_upside_1d_target, 0.0, 1.0))
            )
            release_value_target = (
                float(policy.at[stock, "release_value_target"] or 0.0)
                if "release_value_target" in policy.columns
                else float(np.clip(0.62 * sell_release_value + 0.24 * cash_defense_value + 0.14 * lifecycle_sell_gate, 0.0, 1.0))
            )
            defense_value_target = (
                float(policy.at[stock, "defense_value_target"] or 0.0)
                if "defense_value_target" in policy.columns
                else float(np.clip(0.68 * cash_defense_value + 0.20 * (1.0 - alpha_opportunity_value) + 0.12 * clipped_intent_risk, 0.0, 1.0))
            )
            gate_denominator = deploy_value_target + release_value_target + defense_value_target + 1.0e-6
            deploy_gate_target = (
                float(policy.at[stock, "deploy_gate_target"] or 0.0)
                if "deploy_gate_target" in policy.columns
                else float(np.clip(deploy_value_target / gate_denominator, 0.0, 1.0))
            )
            release_gate_target = (
                float(policy.at[stock, "release_gate_target"] or 0.0)
                if "release_gate_target" in policy.columns
                else float(np.clip(release_value_target / gate_denominator, 0.0, 1.0))
            )
            defense_gate_target = (
                float(policy.at[stock, "defense_gate_target"] or 0.0)
                if "defense_gate_target" in policy.columns
                else float(np.clip(defense_value_target / gate_denominator, 0.0, 1.0))
            )
            decision_gate_denominator = deploy_value_target + release_value_target + 1.0e-6
            decision_deploy_gate = float(
                np.clip(
                    0.62 * deploy_gate_target + 0.38 * np.clip(deploy_value_target / decision_gate_denominator, 0.0, 1.0),
                    0.0,
                    1.0,
                )
            ) if hierarchical_budget_mode else deploy_gate_target
            decision_release_gate = float(
                np.clip(
                    0.62 * release_gate_target + 0.38 * np.clip(release_value_target / decision_gate_denominator, 0.0, 1.0),
                    0.0,
                    1.0,
                )
            ) if hierarchical_budget_mode else release_gate_target
            decision_defense_signal = float(budget_model_defense_gate_signal if hierarchical_budget_mode else defense_gate_target)
            held_defense_weight = 0.0 if constraint_only_budget_mode else 1.0
            lifecycle_sell_pressure = float(
                np.clip(
                    0.24 * decision_release_gate
                    + 0.22 * release_value_target
                    + 0.18 * lifecycle_sell_gate
                    + 0.14 * sell_rank_score
                    + 0.12 * sell_attribution_score
                    + 0.08 * max(reduce_action_value, exit_action_value)
                    + 0.08 * decision_defense_signal * held_defense_weight
                    - 0.10 * hold_continuation_value
                    - 0.08 * max(add_action_value, hold_action_value)
                    - 0.08 * decision_deploy_gate,
                    0.0,
                    1.0,
                )
            )
            exit_timing_pressure = (
                float(policy.at[stock, "exit_timing_pressure"] or 0.0)
                if "exit_timing_pressure" in policy.columns
                else float(np.clip(exit_hazard * 0.62 + sell_pressure * 0.28 + exit_urgency * 0.10, 0.0, 1.0))
            )
            exit_timing_pressure_values.at[stock] = exit_timing_pressure
            planned_holding_days = float(policy.at[stock, "planned_holding_days"] or 0.0) if "planned_holding_days" in policy.columns else 0.0
            current_weight = float(current.get(stock, 0.0))
            current_hold_days = float(self.holdings.get(stock).hold_days) if stock in self.holdings else 0.0
            days_since_last_sell = _days_since(self.last_sell_dates, stock)
            days_since_last_reduce = _days_since(self.last_reduce_dates, stock)
            weak_tail_zero_candidate.at[stock] = bool(
                current_weight > 1e-8
                and current_hold_days >= 8.0
                and (
                    (action == "exit" and (exit_urgency >= 0.18 or exit_hazard >= 0.42 or exit_timing_pressure >= 0.50))
                    or (
                        action in {"hold", "reduce", "skip"}
                        and (
                            exit_urgency >= 0.26 + exit_patience_target * 0.06
                            or exit_hazard >= 0.34 + exit_patience_target * 0.04
                            or exit_timing_pressure >= 0.44 + exit_patience_target * 0.06
                            or lifecycle_sell_gate >= 0.60
                        )
                        and (
                            reduce_quality >= hold_quality + 0.04
                            or reduce_fraction >= 0.18
                            or exit_timing_pressure >= 0.52
                            or sell_rank_score >= 0.70
                            or sell_release_value >= 0.64
                            or max(reduce_action_value, exit_action_value) >= max(add_action_value, hold_action_value) + 0.16
                        )
                        and sell_pressure >= 0.20
                        and value_arbitration_target <= 0.58
                        and delta_hint <= max(0.01, hold_boost)
                        and add_quality <= hold_quality + 0.02
                    )
                )
            )
            if action == "exit":
                forced_zero.at[stock] = True
                continue
            if action == "reduce":
                target_reduce_fraction = float(
                    np.clip(
                        max(
                            reduce_fraction,
                            max(-delta_hint, 0.0),
                            0.06 + sell_pressure * 0.10 + exit_hazard * 0.08 + exit_timing_pressure * 0.14,
                        ),
                        0.06,
                        0.96,
                    )
                )
                keep_ratio = float(
                    np.clip(
                        1.0
                        - target_reduce_fraction
                        * (
                            0.82
                            + reduce_quality * 0.08
                            + sell_pressure * 0.12
                            + exit_hazard * 0.08
                            + exit_timing_pressure * 0.14
                            + sell_attribution_score * 0.10
                            + lifecycle_sell_gate * 0.12
                            + sell_rank_score * 0.08
                            + sell_release_value * 0.10
                            + reduce_action_value * 0.10
                            + cash_defense_value * (0.06 if not constraint_only_budget_mode else 0.0)
                            - hold_bias_target * 0.10
                            - hold_continuation_value * 0.06
                            - hold_action_value * 0.06
                            - alpha_opportunity_value * 0.04
                        ),
                        0.02 if (exit_hazard > 0.55 or exit_timing_pressure > 0.68) else 0.08,
                        0.92,
                    )
                )
                desired_strength.at[stock] = max(current_weight * keep_ratio, 0.0)
                if (
                    current_weight > 1e-8
                    and hold_boost > 0.02
                    and days_since_last_reduce <= 2.0
                    and target_reduce_fraction < 0.24
                    and exit_hazard < 0.22
                    and sell_pressure < 0.22
                    and exit_timing_pressure < 0.24
                    and sell_release_value < 0.36
                    and decision_deploy_gate > decision_release_gate + 0.02
                ):
                    protected_floor.at[stock] = max(
                        protected_floor.at[stock],
                        current_weight * np.clip(0.72 + hold_bias_target * 0.08 + hold_continuation_value * 0.08 - sell_attribution_score * 0.08, 0.56, 0.88),
                    )
                continue
            if action == "hold":
                hold_scale = (
                    1.0
                    + hold_bias_target * 0.06
                    + exit_patience_target * 0.04
                    + hold_continuation_value * 0.08
                    + hold_action_value * 0.08
                    + alpha_opportunity_value * 0.04
                    + multi_horizon_path_value * 0.04
                    + decision_deploy_gate * 0.04
                    + max(planned_holding_days - 3.0, 0.0) / 120.0
                    - sell_pressure * 0.18
                    - exit_hazard * 0.08
                    - exit_timing_pressure * 0.16
                    - sell_attribution_score * 0.07
                    - lifecycle_sell_gate * 0.08
                    - sell_rank_score * 0.04
                    - sell_release_value * 0.06
                    - cash_defense_value * (0.04 if not constraint_only_budget_mode else 0.0)
                )
                desired_strength.at[stock] = max(
                    current_weight * max(0.54 if exit_timing_pressure > 0.62 else 0.72, hold_scale),
                    current_weight
                    + max(
                        0.0,
                        hold_boost
                        + hold_quality
                        + hold_continuation_value * 0.28
                        + hold_action_value * 0.20
                        + alpha_opportunity_value * 0.12
                        + deploy_value_target * 0.08
                        + decision_deploy_gate * 0.06
                        - sell_pressure * 0.35
                        - exit_hazard * 0.18
                        - exit_timing_pressure * 0.22
                        - lifecycle_sell_pressure * 0.14,
                    )
                    * (0.025 + hold_bias_target * 0.030),
                )
                if current_weight > 1e-8:
                    protected_floor.at[stock] = max(
                        protected_floor.at[stock],
                        current_weight
                        * np.clip(
                            0.82
                            + hold_bias_target * 0.10
                            + exit_patience_target * 0.05
                            + hold_continuation_value * 0.08
                            + hold_action_value * 0.06
                            + alpha_opportunity_value * 0.04
                            + max(planned_holding_days - 3.0, 0.0) / 180.0,
                            0.54
                            + max(
                                0.0,
                                0.08
                                - sell_pressure * 0.08
                                - exit_timing_pressure * 0.08
                                - lifecycle_sell_pressure * 0.08,
                            ),
                            0.97,
                        ),
                    )
                continue
            if action == "add":
                add_increment = max(
                    0.0,
                    strength * 0.55
                    + add_quality * 0.15
                    + alpha_opportunity_value * 0.14
                    + deployment_opportunity_cost * 0.10
                    + add_action_value * 0.18
                    + multi_horizon_path_value * 0.08
                    + deploy_value_target * 0.10
                    + decision_deploy_gate * 0.08
                    + large_upside_1d_target * 0.06
                    + planned_holding_days / 300.0
                    - sell_pressure * 0.14
                    - exit_hazard * 0.10
                    - exit_timing_pressure * 0.18
                    - lifecycle_sell_pressure * 0.18
                    - sell_release_value * 0.10
                    - decision_release_gate * 0.08
                    - decision_defense_signal * (0.08 if not constraint_only_budget_mode else 0.14)
                    - clipped_intent_risk * 0.10,
                )
                desired_strength.at[stock] = (
                    current_weight
                    if (
                        sell_pressure > 0.26
                        or exit_hazard > 0.20
                        or exit_timing_pressure > 0.24
                        or lifecycle_sell_gate > 0.52
                        or max(reduce_action_value, exit_action_value) > max(add_action_value, hold_action_value) + 0.12
                        or clipped_intent_risk > 0.68
                        or ((not constraint_only_budget_mode) and decision_defense_signal > 0.46 and decision_deploy_gate < 0.34)
                    )
                    else max(current_weight + max(0.015, add_increment), current_weight)
                )
                if current_weight > 1e-8:
                    floor_ratio = float(
                        np.clip(
                            0.90 + hold_bias_target * 0.04 + hold_continuation_value * 0.04 - lifecycle_sell_pressure * 0.07,
                            0.78,
                            0.98,
                        )
                    )
                    if (
                        exit_urgency < 0.16
                        and exit_hazard < 0.18
                        and sell_pressure < 0.18
                        and exit_timing_pressure < 0.20
                        and lifecycle_sell_gate < 0.32
                        and clipped_intent_risk < 0.48
                        and sell_release_value < 0.30
                        and ((not constraint_only_budget_mode) or decision_defense_signal < 0.42)
                        and decision_deploy_gate > decision_release_gate + 0.02
                        and add_quality > max(0.10, hold_quality - 0.02)
                        and add_action_value >= hold_action_value - 0.04
                        and reduce_quality < hold_quality + 0.04
                    ):
                        floor_ratio = max(
                            floor_ratio,
                            float(np.clip(0.96 + hold_bias_target * 0.02, 0.94, 1.00)),
                        )
                    protected_floor.at[stock] = max(
                        protected_floor.at[stock],
                        current_weight * floor_ratio,
                    )
                continue
            if action == "open":
                reentry_penalty = np.clip((4.0 - min(days_since_last_sell, days_since_last_reduce)) / 4.0, 0.0, 1.0)
                open_clip_penalty = clipped_intent_risk * (0.18 + max(budget_model_risk_deploy_gap, 0.0) * 0.12) + cash_defense_value * 0.08
                desired_strength.at[stock] = max(
                    strength
                    * max(
                        0.25,
                        0.85
                        + hold_bias_target * 0.15
                        + alpha_opportunity_value * 0.10
                        + deployment_opportunity_cost * 0.08
                        + open_action_value * 0.16
                        + multi_horizon_path_value * 0.08
                        + deploy_value_target * 0.10
                        + decision_deploy_gate * 0.08
                        + risk_adjusted_action_value * 0.06
                        - reentry_guard_target * 0.30
                        - reentry_penalty * (0.22 + reentry_guard_target * 0.55)
                        - sell_pressure * 0.20
                        - exit_hazard * 0.12
                        - decision_defense_signal * (0.12 if constraint_only_budget_mode else 0.08)
                        - decision_release_gate * 0.05
                        - open_clip_penalty,
                    ),
                    max(delta_hint, 0.02 + entry_quality * 0.20 + alpha_opportunity_value * 0.05 + deployment_opportunity_cost * 0.04 + open_action_value * 0.06 + deploy_value_target * 0.05 + decision_deploy_gate * 0.04 + planned_holding_days / 320.0)
                    * max(0.55, 1.0 - sell_pressure * 0.35 - multi_horizon_forward_risk * 0.08 - clipped_intent_risk * 0.18 - decision_defense_signal * (0.14 if constraint_only_budget_mode else 0.10)),
                )
                continue
            desired_strength.at[stock] = current_weight * (1.0 + hold_bias_target * 0.02)

        model_release_signal = (
            (current > 1e-8)
            & (~action_names.isin({"reduce", "exit"}))
            & (
                (lifecycle_sell_gate_series >= 0.62)
                | (
                    (sell_attribution_series >= 0.62)
                    & (sell_rank_series >= 0.54)
                    & (sell_pressure_series >= 0.18)
                )
                | (
                    (decision_release_gate_series >= decision_deploy_gate_series + 0.08)
                    & (release_value_series >= deploy_value_series + 0.04)
                    & (sell_release_series >= 0.42)
                )
                | (
                    (exit_timing_pressure_values >= 0.40)
                    & (
                        (sell_pressure_series >= 0.22)
                        | (sell_release_series >= 0.48)
                        | (sell_rank_series >= 0.64)
                    )
                )
                | direct_action_funding_release_authorized
            )
        )
        sell_authorized_mask = (
            (current > 1e-8)
            & (
                action_names.isin({"reduce", "exit"})
                | model_release_signal
                | portfolio_daily_source_target
                | weak_tail_zero_candidate
                | forced_zero
            )
        )
        deploy_funding_weak_evidence_count = (
            (hold_continuation_series <= 0.46).astype(float)
            + (alpha_opportunity_series <= 0.30).astype(float)
            + (decision_deploy_gate_series <= decision_release_gate_series + 0.02).astype(float)
            + (current <= float(position_cap_target) * 0.70).astype(float)
        )
        deploy_funding_rebalance_signal = (
            (current > 1e-8)
            & (~sell_authorized_mask)
            & bool(sell_source_decoupled_mode)
            & action_names.isin({"hold", "skip"})
            & (
                (budget_deploy_score >= 0.12)
                | (budget_model_deploy_signal >= 0.78)
                | (
                    (flat_entry_action_share >= 0.10)
                    & (budget_deploy_score >= 0.09)
                )
            )
            & (deploy_funding_weak_evidence_count >= 2.0)
        )
        if direct_action_preserving_mode:
            direct_funding_evidence = (
                direct_action_funding_release_authorized
                | (
                    direct_action_mode_series
                    & held_mask
                    & (direct_action_release_advantage_series >= 0.020)
                    & (direct_action_gap_series >= 0.035)
                )
                | (
                    direct_action_mode_series
                    & held_mask
                    & (direct_action_release_advantage_series >= 0.0)
                    & (deploy_funding_weak_evidence_count >= 3.0)
                    & (
                        (sell_release_series >= 0.40)
                        | (decision_release_gate_series >= decision_deploy_gate_series + 0.03)
                    )
                )
            )
            deploy_funding_rebalance_signal = (
                deploy_funding_rebalance_signal
                & direct_funding_evidence
                & (~direct_action_funding_protected)
            )
        deploy_funding_retention_floor = pd.Series(1.0, index=prices.index, dtype=float)
        if sell_source_decoupled_mode:
            deploy_pressure = float(np.clip(budget_deploy_score, 0.0, 1.0))
            entry_pressure = float(np.clip(flat_entry_action_share, 0.0, 1.0))
            deploy_funding_retention_floor = (
                pd.Series(0.94 - deploy_pressure * 0.12 - entry_pressure * 0.06, index=prices.index, dtype=float)
                - (1.0 - hold_continuation_series.clip(0.0, 1.0)) * 0.05
                - (1.0 - alpha_opportunity_series.clip(0.0, 1.0)) * 0.03
                - (decision_release_gate_series - decision_deploy_gate_series).clip(lower=0.0, upper=1.0) * 0.04
            ).clip(lower=0.84, upper=0.94)
            if direct_action_preserving_mode:
                direct_release_boost = direct_action_release_advantage_series.clip(lower=0.0, upper=0.12)
                direct_keep_protection = direct_action_keep_advantage_series.clip(lower=0.0, upper=0.12)
                deploy_funding_retention_floor = (
                    deploy_funding_retention_floor
                    + direct_keep_protection * 0.45
                    - direct_release_boost * 0.30
                ).clip(lower=0.88, upper=0.98)
        sell_authorization_score = (
            action_names.isin({"reduce", "exit"}).astype(float) * 1.00
            + model_release_signal.astype(float) * 0.82
            + portfolio_daily_source_target.astype(float) * 0.64
            + deploy_funding_rebalance_signal.astype(float) * 0.46
            + weak_tail_zero_candidate.astype(float) * 0.74
            + lifecycle_sell_gate_series.clip(0.0, 1.0) * 0.24
            + sell_attribution_series.clip(0.0, 1.0) * 0.18
            + sell_rank_series.clip(0.0, 1.0) * 0.12
            + decision_release_gate_series.clip(0.0, 1.0) * 0.16
            + sell_release_series.clip(0.0, 1.0) * 0.12
            + exit_timing_pressure_values.clip(0.0, 1.0) * 0.12
            - decision_deploy_gate_series.clip(0.0, 1.0) * 0.08
            - hold_continuation_series.clip(0.0, 1.0) * 0.08
            - alpha_opportunity_series.clip(0.0, 1.0) * 0.06
        ).clip(lower=0.0, upper=1.0)

        budget_dropped = pd.Series(False, index=prices.index, dtype=bool)
        budget_released_from_hold = pd.Series(False, index=prices.index, dtype=bool)
        budget_entry_candidate_count = 0
        budget_entry_keep_count = 0
        budget_held_protected_count = 0
        budget_reclaimable_held_count = 0
        budget_released_held_count = 0
        if budget_semantics == BUDGET_SEMANTICS_LEGACY and candidate_budget < len(desired_strength):
            keep = desired_strength.nlargest(candidate_budget).index
            budget_dropped = pd.Series(~desired_strength.index.isin(keep), index=desired_strength.index, dtype=bool)
            desired_strength = desired_strength.where(desired_strength.index.isin(keep), 0.0)
            dropped_tail_zero = budget_dropped & weak_tail_zero_candidate
            forced_zero = forced_zero | dropped_tail_zero
            protected_floor = protected_floor.where(~dropped_tail_zero, 0.0)
        elif budget_semantics == BUDGET_SEMANTICS_SPLIT:
            held_lifecycle_mask = current > 1e-8
            held_survivor_mask = held_lifecycle_mask & (~forced_zero)
            entry_candidate_mask = (current <= 1e-8) & (desired_strength > 1e-12) & (~forced_zero)
            budget_entry_candidate_count = int(entry_candidate_mask.sum())
            budget_held_protected_count = int(held_lifecycle_mask.sum())
            candidate_limit = int(np.clip(candidate_budget, 1, int(self.max_positions)))
            small_held_weight_threshold = min(
                float(position_cap_target) * 0.60,
                max(float(gross_exposure_target) / max(candidate_limit, 1) * 1.10, 0.04),
            )
            reclaimable_held_mask = held_survivor_mask & (
                portfolio_daily_source_target
                | weak_tail_zero_candidate
                | (
                    action_names.isin({"hold", "add", "skip"})
                    & (current <= small_held_weight_threshold + 1.0e-12)
                    & (protected_floor <= current * 0.88 + 1.0e-12)
                    & (hold_continuation_series < 0.22)
                    & (alpha_opportunity_series < 0.18)
                    & (deploy_executability_series < 0.18)
                    & (decision_deploy_gate_series <= np.maximum(decision_release_gate_series + 0.02, 0.18))
                    & (sell_pressure_series < 0.18)
                    & (exit_timing_pressure_series < 0.22)
                )
            )
            if sell_source_decoupled_mode:
                reclaimable_held_mask = reclaimable_held_mask & (
                    sell_authorized_mask | deploy_funding_rebalance_signal
                )
            budget_reclaimable_held_count = int(reclaimable_held_mask.sum())
            fixed_held_keep_mask = held_survivor_mask & (~reclaimable_held_mask)
            remaining_slots = max(
                0,
                min(
                    int(self.max_positions) - int(fixed_held_keep_mask.sum()),
                    candidate_limit - int(fixed_held_keep_mask.sum()),
                ),
            )
            entry_keep_mask = pd.Series(False, index=prices.index, dtype=bool)
            reclaimable_held_keep_mask = pd.Series(False, index=prices.index, dtype=bool)
            competitive_mask = reclaimable_held_mask | entry_candidate_mask
            if remaining_slots > 0 and bool(competitive_mask.any()):
                competitive_priority = desired_strength.copy()
                competitive_priority = competitive_priority + protected_floor.clip(0.0, position_cap_target) * 0.30
                competitive_priority = competitive_priority + hold_continuation_series.clip(0.0, 1.0) * 0.025
                competitive_priority = competitive_priority + action_names.isin({"hold", "add"}).astype(float) * 0.015
                competitive_priority = competitive_priority + entry_candidate_mask.astype(float) * (
                    (
                        alpha_opportunity_series.clip(0.0, 1.0) * 0.30
                        + deployment_opportunity_series.clip(0.0, 1.0) * 0.22
                        + deploy_value_series.clip(0.0, 1.0) * 0.18
                        + decision_deploy_gate_series.clip(0.0, 1.0) * 0.16
                        + deploy_executability_series.clip(0.0, 1.0) * 0.14
                    ) * 0.050
                    + deploy_executability_series.clip(0.0, 1.0) * 0.030
                    + entry_quality_series.clip(0.0, 1.0) * 0.020
                )
                competitive_priority = competitive_priority - reclaimable_held_mask.astype(float) * (
                    (
                        sell_attribution_series.clip(0.0, 1.0) * 0.38
                        + lifecycle_sell_gate_series.clip(0.0, 1.0) * 0.24
                        + sell_rank_series.clip(0.0, 1.0) * 0.18
                        + sell_release_series.clip(0.0, 1.0) * 0.18
                        + cash_defense_series.clip(0.0, 1.0) * (0.04 if constraint_only_budget_mode else 0.08)
                        + sell_pressure_series.clip(0.0, 1.0) * 0.16
                        + exit_timing_pressure_series.clip(0.0, 1.0) * 0.12
                    ) * 0.040
                    + decision_release_gate_series.clip(0.0, 1.0) * 0.020
                )
                if portfolio_daily_source_exec_guard_mode:
                    competitive_priority = competitive_priority - portfolio_daily_source_target.astype(float) * (
                        portfolio_daily_source_execution_pressure.clip(0.0, 1.0) * 0.18
                        + portfolio_daily_source_gap.clip(lower=0.0, upper=0.24) * 0.70
                        + portfolio_daily_source_score.clip(lower=0.0, upper=1.0) * 0.035
                        + portfolio_daily_source_executability.clip(0.0, 1.0) * 0.050
                    )
                competitive_keep = competitive_priority.where(competitive_mask, -np.inf).nlargest(remaining_slots).index
                competitive_keep_mask = pd.Series(competitive_priority.index.isin(competitive_keep), index=competitive_priority.index, dtype=bool)
                entry_keep_mask = competitive_keep_mask & entry_candidate_mask
                reclaimable_held_keep_mask = competitive_keep_mask & reclaimable_held_mask
            budget_entry_keep_count = int(entry_keep_mask.sum())
            keep_mask = fixed_held_keep_mask | reclaimable_held_keep_mask | entry_keep_mask
            budget_dropped = entry_candidate_mask & (~entry_keep_mask)
            budget_released_from_hold = reclaimable_held_mask & (~reclaimable_held_keep_mask)
            budget_released_held_count = int(budget_released_from_hold.sum())
            desired_strength = desired_strength.where(keep_mask, 0.0)
            protected_floor = protected_floor.where(~budget_released_from_hold, 0.0)
        desired_strength = desired_strength.where(~forced_zero, 0.0)
        if portfolio_daily_source_exec_guard_mode and bool(portfolio_daily_source_target.any()):
            source_exec_cap = (current * portfolio_daily_source_retention_floor).clip(lower=0.0, upper=position_cap_target)
            desired_strength = desired_strength.where(
                ~portfolio_daily_source_target,
                pd.concat(
                    [desired_strength.rename("desired"), source_exec_cap.rename("source_exec_cap")],
                    axis=1,
                ).min(axis=1),
            )
            protected_floor = protected_floor.where(
                ~portfolio_daily_source_target,
                pd.concat(
                    [protected_floor.rename("protected"), source_exec_cap.rename("source_exec_cap")],
                    axis=1,
                ).min(axis=1),
            )
        sell_reduction_priority = (
            sell_attribution_series.clip(0.0, 1.0) * 0.38
            + lifecycle_sell_gate_series.clip(0.0, 1.0) * 0.24
            + sell_rank_series.clip(0.0, 1.0) * 0.18
            + sell_release_series.clip(0.0, 1.0) * 0.18
            + cash_defense_series.clip(0.0, 1.0) * (0.04 if constraint_only_budget_mode else 0.08)
            + sell_pressure_series.clip(0.0, 1.0) * 0.16
            + exit_timing_pressure_series.clip(0.0, 1.0) * 0.12
            + action_names.isin({"reduce", "exit"}).astype(float) * 0.14
            - hold_continuation_series.clip(0.0, 1.0) * 0.08
            - alpha_opportunity_series.clip(0.0, 1.0) * 0.08
            - value_arbitration_series.clip(0.0, 1.0) * 0.04
            - action_names.isin({"hold", "add"}).astype(float) * 0.06
            + (
                portfolio_daily_source_target.astype(float)
                * (
                    portfolio_daily_source_execution_pressure.clip(0.0, 1.0) * 0.44
                    + portfolio_daily_source_gap.clip(lower=0.0, upper=0.24) * 1.70
                    + portfolio_daily_source_score.clip(lower=0.0, upper=1.0) * 0.12
                    + portfolio_daily_source_executability.clip(0.0, 1.0) * 0.12
                )
                if portfolio_daily_source_exec_guard_mode
                else 0.0
            )
        ).clip(lower=0.0)
        deploy_intent_priority = (
            alpha_opportunity_series.clip(0.0, 1.0) * 0.30
            + deployment_opportunity_series.clip(0.0, 1.0) * 0.22
            + deploy_value_series.clip(0.0, 1.0) * 0.18
            + decision_deploy_gate_series.clip(0.0, 1.0) * 0.16
            + deploy_executability_series.clip(0.0, 1.0) * 0.14
            + hold_continuation_series.clip(0.0, 1.0) * 0.08
            + action_names.isin({"open", "add"}).astype(float) * 0.16
            + action_names.eq("hold").astype(float) * 0.08
            - sell_reduction_priority.clip(0.0, 1.0) * 0.10
            - cash_defense_series.clip(0.0, 1.0) * (0.06 if constraint_only_budget_mode else 0.10)
            - (
                portfolio_daily_source_target.astype(float)
                * (
                    portfolio_daily_source_execution_pressure.clip(0.0, 1.0) * 0.32
                    + portfolio_daily_source_gap.clip(lower=0.0, upper=0.24) * 0.95
                )
                if portfolio_daily_source_exec_guard_mode
                else 0.0
            )
        ).clip(lower=0.0)
        direct_action_reallocation_retention_floor = pd.Series(1.0, index=prices.index, dtype=float)
        if direct_action_reallocation_mode:
            direct_action_reallocation_retention_floor = (
                pd.Series(0.885, index=prices.index, dtype=float)
                + hold_continuation_series.clip(0.0, 1.0) * 0.045
                + direct_action_keep_advantage_series.clip(lower=0.0, upper=0.14) * 0.120
                + alpha_opportunity_series.clip(0.0, 1.0) * 0.025
                - deploy_intent_priority.clip(0.0, 1.0) * 0.030
            ).clip(lower=0.865, upper=0.945)
            if direct_action_pair_reallocation_mode:
                pair_retention_floor = (
                    pd.Series(0.800, index=prices.index, dtype=float)
                    + hold_continuation_series.clip(0.0, 1.0) * 0.045
                    + direct_action_keep_advantage_series.clip(lower=0.0, upper=0.18) * 0.090
                    + alpha_opportunity_series.clip(0.0, 1.0) * 0.020
                    - deploy_intent_priority.clip(0.0, 1.0) * 0.045
                ).clip(lower=0.780, upper=0.915)
                direct_action_reallocation_retention_floor = direct_action_reallocation_retention_floor.where(
                    ~direct_action_pair_reallocation_source,
                    pair_retention_floor,
                )
                if portfolio_daily_source_exec_guard_mode:
                    direct_action_reallocation_retention_floor = direct_action_reallocation_retention_floor.where(
                        ~portfolio_daily_source_target,
                        portfolio_daily_source_retention_floor,
                    )

        sell_source_floor_guarded = pd.Series(False, index=prices.index, dtype=bool)
        if sell_source_decoupled_mode:
            original_protected_floor = protected_floor.copy()
            sell_source_retention_floor = pd.Series(1.0, index=prices.index, dtype=float).where(
                ~deploy_funding_rebalance_signal,
                deploy_funding_retention_floor,
            )
            if direct_action_reallocation_mode:
                sell_source_retention_floor = sell_source_retention_floor.where(
                    ~direct_action_reallocation_source,
                    direct_action_reallocation_retention_floor,
                )
            sell_source_floor = (current * sell_source_retention_floor).where(
                (current > 1e-8) & (~sell_authorized_mask) & (~forced_zero),
                0.0,
            ).clip(lower=0.0, upper=position_cap_target)
            sell_source_floor_guarded = sell_source_floor > original_protected_floor + 1e-12
            protected_floor = pd.concat(
                [protected_floor.rename("protected"), sell_source_floor.rename("sell_source")],
                axis=1,
            ).max(axis=1)

        target_weights = self._allocate_with_cap(
            desired_strength,
            gross_exposure_target,
            position_cap=position_cap_target,
        )
        allocation_layer_expected_turnover = 0.0
        allocation_layer_cash_after = float(max(0.0, 1.0 - float(target_weights.sum())))
        allocation_layer_buy_turnover = 0.0
        allocation_layer_sell_turnover = 0.0
        allocation_layer_available_cash_to_deploy = 0.0
        allocation_layer_objective_value = 0.0
        allocation_layer_constraint_violations = 0.0
        allocation_layer_receiver_executable_candidate = pd.Series(False, index=prices.index, dtype=bool)
        allocation_layer_source_executable_candidate = pd.Series(False, index=prices.index, dtype=bool)
        if end_to_end_allocation_layer_mode:
            if "portfolio_daily_receiver_executable_candidate" in policy.columns:
                allocation_layer_receiver_executable_candidate = (
                    _policy_numeric("portfolio_daily_receiver_executable_candidate") > 0.5
                )
            else:
                allocation_layer_receiver_executable_candidate = (
                    portfolio_daily_receiver_candidate | portfolio_daily_unified_receiver_candidate
                )
            if "portfolio_daily_source_executable_candidate" in policy.columns:
                allocation_layer_source_executable_candidate = (
                    _policy_numeric("portfolio_daily_source_executable_candidate") > 0.5
                )
            else:
                allocation_layer_source_executable_candidate = (
                    portfolio_daily_source_candidate | portfolio_daily_unified_source_candidate
                )
            allocation_problem = policy.copy()
            allocation_problem["stock"] = prices.index.astype(str)
            allocation_problem["current_weight"] = current.reindex(prices.index).fillna(0.0).astype(float)
            allocation_problem["portfolio_daily_receiver_executable_candidate"] = (
                allocation_layer_receiver_executable_candidate.astype(float)
            )
            allocation_problem["portfolio_daily_source_executable_candidate"] = (
                allocation_layer_source_executable_candidate.astype(float)
            )
            allocation_solution = solve_semidifferentiable_allocation(
                allocation_problem,
                constraints=AllocationOptimizerConstraints(
                    cash_reserve_target=float((global_targets or {}).get("cash_reserve_target", 0.05) or 0.05),
                    turnover_limit=turnover_budget,
                    max_position_weight=position_cap_target,
                ),
            )
            target_weights = (
                allocation_solution.target_weight.reindex(prices.index)
                .replace([np.inf, -np.inf], np.nan)
                .fillna(0.0)
                .clip(lower=0.0, upper=position_cap_target)
                .astype(float)
            )
            desired_strength = target_weights.copy()
            protected_floor = pd.Series(0.0, index=prices.index, dtype=float)
            forced_zero = pd.Series(False, index=prices.index, dtype=bool)
            budget_dropped = pd.Series(False, index=prices.index, dtype=bool)
            budget_released_from_hold = pd.Series(False, index=prices.index, dtype=bool)
            direct_action_add_signal = pd.Series(False, index=prices.index, dtype=bool)
            direct_action_open_signal = pd.Series(False, index=prices.index, dtype=bool)
            direct_action_deploy_signal = pd.Series(False, index=prices.index, dtype=bool)
            direct_action_core_deploy_target = pd.Series(False, index=prices.index, dtype=bool)
            direct_action_add_authorized = pd.Series(False, index=prices.index, dtype=bool)
            direct_action_open_authorized = pd.Series(False, index=prices.index, dtype=bool)
            direct_action_deploy_authorized = pd.Series(False, index=prices.index, dtype=bool)
            direct_action_reallocation_source = pd.Series(False, index=prices.index, dtype=bool)
            direct_action_pair_reallocation_source = pd.Series(False, index=prices.index, dtype=bool)
            receiver_authorization_subset_violation_count = 0
            allocation_delta_preview = (target_weights - current).replace([np.inf, -np.inf], np.nan).fillna(0.0)
            receiver_delta_threshold = pd.concat(
                [
                    pd.Series(DEFAULT_EXECUTION_DEADBAND_ABS * 1.25, index=prices.index, dtype=float),
                    current.clip(lower=0.0) * 0.010,
                ],
                axis=1,
            ).max(axis=1)
            source_delta_threshold = pd.concat(
                [
                    pd.Series(DEFAULT_EXECUTION_DEADBAND_ABS * 1.25, index=prices.index, dtype=float),
                    current.clip(lower=0.0) * 0.010,
                ],
                axis=1,
            ).max(axis=1)
            portfolio_daily_receiver_target = (
                allocation_layer_receiver_executable_candidate
                & (allocation_delta_preview > receiver_delta_threshold)
            )
            portfolio_daily_source_target = (
                allocation_layer_source_executable_candidate
                & (allocation_delta_preview < -source_delta_threshold)
            )
            portfolio_daily_receiver_candidate = allocation_layer_receiver_executable_candidate.copy()
            portfolio_daily_source_candidate = allocation_layer_source_executable_candidate.copy()
            portfolio_daily_receiver_target_count = int(portfolio_daily_receiver_target.sum())
            portfolio_daily_source_target_count = int(portfolio_daily_source_target.sum())
            sell_authorized_mask = (
                (current > 1e-8)
                & (
                    portfolio_daily_source_target
                    | model_release_signal
                    | weak_tail_zero_candidate
                )
            )
            allocation_layer_expected_turnover = float(allocation_solution.expected_turnover)
            allocation_layer_cash_after = float(allocation_solution.cash_after)
            allocation_layer_buy_turnover = float(allocation_solution.buy_turnover)
            allocation_layer_sell_turnover = float(allocation_solution.sell_turnover)
            allocation_layer_available_cash_to_deploy = float(allocation_solution.available_cash_to_deploy)
            allocation_layer_objective_value = float(allocation_solution.allocation_objective_value)
            allocation_layer_constraint_violations = float(
                allocation_solution.diagnostics.get("constraint_violations", 0.0)
            )
        protected_floor = protected_floor.clip(lower=0.0, upper=position_cap_target)
        portfolio_daily_source_exec_cap_guarded = pd.Series(False, index=prices.index, dtype=bool)
        if portfolio_daily_source_exec_guard_mode and bool(portfolio_daily_source_target.any()):
            source_exec_cap = (current * portfolio_daily_source_retention_floor).clip(lower=0.0, upper=position_cap_target)
            source_cap_mask = portfolio_daily_source_target & (target_weights > source_exec_cap + 1.0e-12)
            target_weights = target_weights.where(~portfolio_daily_source_target, source_exec_cap)
            protected_floor = protected_floor.where(
                ~portfolio_daily_source_target,
                pd.concat(
                    [protected_floor.rename("protected"), source_exec_cap.rename("source_exec_cap")],
                    axis=1,
                ).min(axis=1),
            )
            portfolio_daily_source_exec_cap_guarded = portfolio_daily_source_exec_cap_guarded | source_cap_mask
        if float(protected_floor.sum()) > float(gross_exposure_target) > 0.0:
            if sell_source_decoupled_mode:
                gross_exposure_target = min(float(protected_floor.sum()), 0.999)
            else:
                protected_floor = protected_floor / float(protected_floor.sum()) * float(gross_exposure_target)
        sell_priority_guarded = pd.Series(False, index=prices.index, dtype=bool)
        if bool((protected_floor > 1e-8).any()):
            target_weights = target_weights.where(target_weights >= protected_floor, protected_floor)
            excess = float(target_weights.sum() - gross_exposure_target)
            if excess > 1e-8:
                if use_sell_priority_guard or constraint_only_budget_mode:
                    target_weights, shrink_guarded = self._shrink_to_target_by_priority(
                        target_weights,
                        target_total=float(gross_exposure_target),
                        floor=protected_floor,
                        priority=sell_reduction_priority,
                    )
                    sell_priority_guarded = sell_priority_guarded | shrink_guarded
                else:
                    reducible = (target_weights - protected_floor).clip(lower=0.0)
                    reducible_sum = float(reducible.sum())
                    if reducible_sum > 1e-8:
                        target_weights = target_weights - reducible / reducible_sum * excess
                    elif float(target_weights.sum()) > 1e-8:
                        target_weights = target_weights / float(target_weights.sum()) * float(gross_exposure_target)
        budget_split_bound_guarded = pd.Series(False, index=prices.index, dtype=bool)
        if execution_semantics == EXECUTION_SEMANTICS_SEMANTIC and budget_semantics == BUDGET_SEMANTICS_SPLIT:
            for stock in prices.index:
                previous_weight = float(current.get(stock, 0.0))
                if previous_weight <= 1e-8 or bool(forced_zero.get(stock, False)):
                    continue
                model_action_name = str(policy.at[stock, "action_label"] or "skip").strip().lower()
                if model_action_name != "reduce":
                    continue
                target_value = float(target_weights.get(stock, 0.0))
                if target_value > previous_weight + 1e-8:
                    target_weights.at[stock] = previous_weight
                    budget_split_bound_guarded.at[stock] = True
        translation_floor_guarded = pd.Series(False, index=prices.index, dtype=bool)
        translation_cap_guarded = pd.Series(False, index=prices.index, dtype=bool)
        translation_soft_lift_guarded = pd.Series(False, index=prices.index, dtype=bool)
        translation_floor = protected_floor.copy()
        translation_soft_floor = protected_floor.copy()
        translation_cap = pd.Series(position_cap_target, index=prices.index, dtype=float)
        if (
            execution_semantics == EXECUTION_SEMANTICS_SEMANTIC
            and budget_semantics == BUDGET_SEMANTICS_SPLIT
            and translation_guard_mode
        ):
            for stock in prices.index:
                previous_weight = float(current.get(stock, 0.0))
                target_value = float(target_weights.get(stock, 0.0))
                if bool(forced_zero.get(stock, False)):
                    translation_cap.at[stock] = 0.0
                    if target_value > 1e-12:
                        target_weights.at[stock] = 0.0
                        translation_cap_guarded.at[stock] = True
                    continue
                model_action_name = str(policy.at[stock, "action_label"] or "skip").strip().lower()
                if bool(portfolio_daily_receiver_target.get(stock, False)):
                    model_action_name = "add" if previous_weight > 1e-8 else "open"
                if model_action_name not in {"skip", "open", "hold", "add", "reduce", "exit"}:
                    continue
                deploy_executability_value = float(deploy_executability_series.get(stock, 0.0))
                decision_deploy_value = float(decision_deploy_gate_series.get(stock, 0.0))
                decision_release_value = float(decision_release_gate_series.get(stock, 0.0))
                strong_deploy_executable = bool(
                    deploy_executability_constraint_mode
                    and model_action_name in {"open", "add"}
                    and (
                        deploy_executability_value >= 0.52
                        or (
                            decision_deploy_value >= decision_release_value + 0.06
                            and float(deploy_value_series.get(stock, 0.0)) >= float(release_value_series.get(stock, 0.0))
                        )
                    )
                )
                direct_add_authorized = bool(direct_action_add_authorized.get(stock, False))
                direct_open_authorized = bool(direct_action_open_authorized.get(stock, False))
                if previous_weight > 1e-8:
                    deadband = max(execution_deadband_abs, previous_weight * execution_deadband_rel)
                    direct_reallocation_source_allowed = bool(
                        direct_action_reallocation_source.get(stock, False)
                    )
                    direct_pair_source_allowed = bool(
                        direct_action_pair_reallocation_source.get(stock, False)
                    )
                    portfolio_source_exec_allowed = bool(
                        portfolio_daily_source_exec_guard_mode
                        and portfolio_daily_source_target.get(stock, False)
                    )
                    if (
                        portfolio_source_exec_allowed
                    ):
                        source_pressure = float(portfolio_daily_source_execution_pressure.get(stock, 0.0))
                        source_retention = float(portfolio_daily_source_retention_floor.get(stock, 0.82))
                        min_source_reduce_delta = max(
                            deadband * 1.15,
                            previous_weight * (0.030 + source_pressure * 0.050),
                            0.0020,
                        )
                        source_exec_cap = max(
                            0.0,
                            min(previous_weight * source_retention, previous_weight - min_source_reduce_delta),
                        )
                        translation_floor.at[stock] = min(
                            float(translation_floor.get(stock, 0.0)),
                            source_exec_cap,
                        )
                        translation_soft_floor.at[stock] = min(
                            float(translation_soft_floor.get(stock, 0.0)),
                            source_exec_cap,
                        )
                        translation_cap.at[stock] = min(
                            float(translation_cap.get(stock, position_cap_target)),
                            source_exec_cap,
                        )
                        if target_value > source_exec_cap + 1.0e-12:
                            target_weights.at[stock] = source_exec_cap
                            translation_cap_guarded.at[stock] = True
                            portfolio_daily_source_exec_cap_guarded.at[stock] = True
                    elif (
                        sell_source_decoupled_mode
                        and not bool(sell_authorized_mask.get(stock, False))
                        and (
                            model_action_name in {"hold", "skip", "open"}
                            or (direct_reallocation_source_allowed and model_action_name == "add")
                        )
                    ):
                        if bool(deploy_funding_rebalance_signal.get(stock, False)):
                            funding_floor = max(
                                0.0,
                                previous_weight * float(deploy_funding_retention_floor.get(stock, 0.90)),
                            )
                        elif direct_reallocation_source_allowed and model_action_name in {"hold", "skip", "add"}:
                            source_retention = float(direct_action_reallocation_retention_floor.get(stock, 0.90))
                            funding_floor = max(0.0, previous_weight * source_retention)
                        else:
                            funding_floor = previous_weight
                        source_ceiling = funding_floor if direct_pair_source_allowed else previous_weight
                        translation_floor.at[stock] = max(float(translation_floor.get(stock, 0.0)), funding_floor)
                        translation_soft_floor.at[stock] = max(float(translation_soft_floor.get(stock, 0.0)), funding_floor)
                        translation_cap.at[stock] = min(float(translation_cap.get(stock, position_cap_target)), source_ceiling)
                        if target_value > source_ceiling + 1e-12:
                            target_weights.at[stock] = source_ceiling
                            translation_cap_guarded.at[stock] = True
                        elif target_value < funding_floor - 1e-12:
                            target_weights.at[stock] = funding_floor
                            translation_floor_guarded.at[stock] = True
                            if bool(deploy_funding_rebalance_signal.get(stock, False)):
                                sell_source_floor_guarded.at[stock] = True
                    elif model_action_name == "hold" and not (
                        sell_source_decoupled_mode and bool(sell_authorized_mask.get(stock, False))
                    ):
                        translation_floor.at[stock] = max(float(translation_floor.get(stock, 0.0)), previous_weight)
                        translation_soft_floor.at[stock] = max(float(translation_soft_floor.get(stock, 0.0)), previous_weight)
                        translation_cap.at[stock] = min(float(translation_cap.get(stock, position_cap_target)), previous_weight)
                        if target_value > previous_weight + 1e-12:
                            target_weights.at[stock] = previous_weight
                            translation_cap_guarded.at[stock] = True
                        elif target_value < previous_weight - 1e-12:
                            target_weights.at[stock] = previous_weight
                            translation_floor_guarded.at[stock] = True
                    elif model_action_name == "add" and not (
                        sell_source_decoupled_mode and bool(model_release_signal.get(stock, False))
                    ):
                        if direct_action_reallocation_mode and direct_add_authorized:
                            min_add_delta = max(
                                deadband * 1.10,
                                previous_weight * (0.020 + budget_model_deploy_signal * 0.015),
                                0.0025,
                            )
                        else:
                            min_add_delta = max(
                                deadband * 1.35,
                                previous_weight * (0.035 + budget_model_deploy_signal * 0.040),
                                0.0035,
                            )
                        min_add_weight = min(
                            position_cap_target,
                            previous_weight + min_add_delta,
                        )
                        hard_add_floor = (
                            min_add_weight
                            if strong_deploy_executable or direct_add_authorized
                            else previous_weight if intent_preserving_constraint_mode else min_add_weight
                        )
                        translation_floor.at[stock] = max(float(translation_floor.get(stock, 0.0)), hard_add_floor)
                        translation_soft_floor.at[stock] = max(float(translation_soft_floor.get(stock, 0.0)), min_add_weight)
                        translation_cap.at[stock] = max(float(translation_cap.get(stock, position_cap_target)), hard_add_floor)
                        if target_value < hard_add_floor - 1e-12:
                            target_weights.at[stock] = hard_add_floor
                            translation_floor_guarded.at[stock] = True
                    elif model_action_name == "reduce":
                        max_reduce_weight = max(
                            0.0,
                            previous_weight - max(deadband * 1.25, previous_weight * (0.040 + max(budget_model_cash_timing_signal, budget_risk_off_score) * 0.040), 0.0030),
                        )
                        translation_cap.at[stock] = min(float(translation_cap.get(stock, position_cap_target)), max_reduce_weight)
                        if target_value > max_reduce_weight + 1e-12:
                            target_weights.at[stock] = max_reduce_weight
                            translation_cap_guarded.at[stock] = True
                    elif model_action_name == "exit":
                        translation_cap.at[stock] = 0.0
                        if target_value > 1e-12:
                            target_weights.at[stock] = 0.0
                            translation_cap_guarded.at[stock] = True
                else:
                    if model_action_name in {"open", "add"} and target_value > 1e-12:
                        if direct_action_reallocation_mode and direct_open_authorized:
                            min_open_weight = min(
                                position_cap_target,
                                max(
                                    execution_deadband_abs * 2.2,
                                    0.010
                                    + budget_model_deploy_signal * 0.008
                                    + budget_model_alpha_focus_signal * 0.006,
                                ),
                            )
                        else:
                            min_open_weight = min(
                                position_cap_target,
                                max(execution_deadband_abs * 2.5, 0.012 + budget_model_deploy_signal * 0.010 + budget_model_alpha_focus_signal * 0.006),
                            )
                        hard_open_floor = (
                            min_open_weight
                            if strong_deploy_executable or direct_open_authorized
                            else 0.0 if intent_preserving_constraint_mode else min_open_weight
                        )
                        translation_floor.at[stock] = max(float(translation_floor.get(stock, 0.0)), hard_open_floor)
                        translation_soft_floor.at[stock] = max(float(translation_soft_floor.get(stock, 0.0)), min_open_weight)
                        if target_value < hard_open_floor - 1e-12:
                            target_weights.at[stock] = hard_open_floor
                            translation_floor_guarded.at[stock] = True
                    elif model_action_name in {"hold", "skip", "reduce", "exit"} and target_value > 1e-12:
                        translation_cap.at[stock] = 0.0
                        target_weights.at[stock] = 0.0
                        translation_cap_guarded.at[stock] = True
            target_weights = target_weights.clip(lower=translation_floor, upper=translation_cap)
            if intent_preserving_constraint_mode:
                target_weights, soft_guarded = self._lift_to_soft_floor_by_priority(
                    target_weights,
                    target_total=float(gross_exposure_target),
                    hard_floor=translation_floor,
                    soft_floor=translation_soft_floor,
                    cap=translation_cap,
                    priority=deploy_intent_priority,
                )
                translation_soft_lift_guarded = translation_soft_lift_guarded | soft_guarded
            effective_floor = pd.concat([protected_floor.rename("protected"), translation_floor.rename("translation")], axis=1).max(axis=1)
            excess = float(target_weights.sum() - gross_exposure_target)
            if excess > 1e-8:
                if use_sell_priority_guard or constraint_only_budget_mode:
                    target_weights, shrink_guarded = self._shrink_to_target_by_priority(
                        target_weights,
                        target_total=float(gross_exposure_target),
                        floor=effective_floor,
                        priority=sell_reduction_priority,
                    )
                    sell_priority_guarded = sell_priority_guarded | shrink_guarded
                else:
                    reducible = (target_weights - effective_floor).clip(lower=0.0)
                    reducible_sum = float(reducible.sum())
                    if reducible_sum > 1e-8:
                        target_weights = target_weights - reducible / reducible_sum * excess
                    elif float(target_weights.sum()) > 1e-8:
                        target_weights = target_weights / float(target_weights.sum()) * float(gross_exposure_target)
        delta = target_weights - current
        raw_turnover = float(delta.abs().sum())
        turnover_intent_guarded = pd.Series(False, index=prices.index, dtype=bool)
        if raw_turnover > turnover_budget > 0:
            forced_sell_delta = (-delta.where((forced_zero) & (delta < 0.0), 0.0)).clip(lower=0.0)
            forced_sell_turnover = float(forced_sell_delta.sum())
            if forced_sell_turnover >= turnover_budget > 0:
                delta = -forced_sell_delta / forced_sell_turnover * turnover_budget
            else:
                remaining_budget = float(max(turnover_budget - forced_sell_turnover, 0.0))
                residual_delta = delta.where(~((forced_zero) & (delta < 0.0)), 0.0)
                residual_turnover = float(residual_delta.abs().sum())
                if residual_turnover > remaining_budget > 0:
                    if intent_preserving_constraint_mode:
                        turnover_preservation_priority = pd.Series(
                            np.where(
                                residual_delta.to_numpy(dtype=float) < 0.0,
                                sell_reduction_priority.reindex(residual_delta.index).to_numpy(dtype=float),
                                deploy_intent_priority.reindex(residual_delta.index).to_numpy(dtype=float),
                            ),
                            index=residual_delta.index,
                            dtype=float,
                        )
                        residual_delta, turnover_guarded = self._trim_delta_to_turnover_by_priority(
                            residual_delta,
                            target_turnover=remaining_budget,
                            priority=turnover_preservation_priority,
                        )
                        turnover_intent_guarded = turnover_intent_guarded | turnover_guarded
                    else:
                        residual_delta = residual_delta * (remaining_budget / residual_turnover)
                elif remaining_budget <= 0:
                    residual_delta = residual_delta * 0.0
                delta = residual_delta
                if forced_sell_turnover > 0:
                    delta = delta.where(~((forced_zero) & (forced_sell_delta > 0.0)), -forced_sell_delta)
        semantic_delta_guarded = pd.Series(False, index=prices.index, dtype=bool)
        if execution_semantics == EXECUTION_SEMANTICS_SEMANTIC:
            for stock in prices.index:
                previous_weight = float(current.get(stock, 0.0))
                if previous_weight <= 1e-8 or bool(forced_zero.get(stock, False)):
                    continue
                model_action_name = str(policy.at[stock, "action_label"] or "skip").strip().lower()
                if bool(portfolio_daily_receiver_target.get(stock, False)):
                    model_action_name = "add" if previous_weight > 1e-8 else "open"
                delta_value = float(delta.get(stock, 0.0))
                if abs(delta_value) <= 1e-12:
                    continue
                deadband = max(execution_deadband_abs, previous_weight * execution_deadband_rel)
                direct_reallocation_source_allowed = bool(
                    direct_action_reallocation_source.get(stock, False)
                )
                exit_timing_pressure = float(exit_timing_pressure_values.get(stock, 0.0))
                protected_floor_value = float(protected_floor.get(stock, 0.0))
                hold_continuation_value = float(hold_continuation_series.get(stock, 0.0))
                alpha_opportunity_value = float(alpha_opportunity_series.get(stock, 0.0))
                deploy_executability_value = float(deploy_executability_series.get(stock, 0.0))
                deploy_gate_value = float(decision_deploy_gate_series.get(stock, 0.0))
                release_gate_value = float(decision_release_gate_series.get(stock, 0.0))
                sell_pressure = (
                    float(policy.at[stock, "sell_pressure"] or 0.0)
                    if "sell_pressure" in policy.columns
                    else 0.0
                )
                lifecycle_gate_value = float(lifecycle_sell_gate_series.get(stock, 0.0))
                sell_rank_value = float(sell_rank_series.get(stock, 0.0))
                protect_unauthorized_sell = (
                    sell_source_decoupled_mode
                    and delta_value < 0.0
                    and not bool(sell_authorized_mask.get(stock, False))
                    and not bool(deploy_funding_rebalance_signal.get(stock, False))
                    and not direct_reallocation_source_allowed
                    and abs(delta_value) <= max(deadband * 4.00, previous_weight * 0.25)
                )
                micro_negative_trim = delta_value < 0.0 and abs(delta_value) <= max(deadband * 1.10, previous_weight * 0.045)
                protect_hold_trim = (
                    model_action_name == "hold"
                    and micro_negative_trim
                    and not direct_reallocation_source_allowed
                    and not (
                        sell_source_decoupled_mode
                        and bool(deploy_funding_rebalance_signal.get(stock, False))
                    )
                    and exit_timing_pressure < 0.42
                    and sell_pressure < 0.34
                    and lifecycle_gate_value < 0.58
                    and sell_rank_value < 0.80
                    and (
                        protected_floor_value >= previous_weight * 0.90
                        or hold_continuation_value >= 0.22
                        or alpha_opportunity_value >= 0.18
                        or deploy_gate_value >= release_gate_value - 0.02
                    )
                )
                protect_add_trim = (
                    model_action_name == "add"
                    and micro_negative_trim
                    and not direct_reallocation_source_allowed
                    and exit_timing_pressure < 0.30
                    and sell_pressure < 0.28
                    and lifecycle_gate_value < 0.48
                    and sell_rank_value < 0.72
                    and (
                        protected_floor_value >= previous_weight * 0.94
                        or deploy_executability_value >= 0.46
                        or deploy_gate_value >= release_gate_value + 0.02
                        or alpha_opportunity_value >= 0.26
                    )
                )
                protect_hold_add = (
                    model_action_name == "hold"
                    and delta_value > 0.0
                    and exit_timing_pressure < 0.26
                    and sell_pressure < 0.24
                    and lifecycle_gate_value < 0.42
                    and abs(delta_value) <= max(deadband * 2.50, previous_weight * 0.08)
                )
                protect_reduce_add = (
                    model_action_name == "reduce"
                    and delta_value > 0.0
                    and abs(delta_value) <= max(deadband * 2.00, previous_weight * 0.08)
                )
                if protect_unauthorized_sell or protect_hold_trim or protect_add_trim or protect_hold_add or protect_reduce_add:
                    delta.at[stock] = 0.0
                    semantic_delta_guarded.at[stock] = True
                    if protect_unauthorized_sell:
                        sell_source_floor_guarded.at[stock] = True
        new_weights = (current + delta).clip(lower=0.0)
        if float(new_weights.sum()) > 0.999:
            if sell_source_decoupled_mode:
                excess_weight = float(new_weights.sum()) - 0.999
                positive_delta = (new_weights - current).clip(lower=0.0)
                positive_delta_sum = float(positive_delta.sum())
                if positive_delta_sum > 1e-8:
                    reduction = positive_delta / positive_delta_sum * min(excess_weight, positive_delta_sum)
                    new_weights = (new_weights - reduction).clip(lower=0.0)
            if float(new_weights.sum()) > 0.999:
                new_weights = new_weights / float(new_weights.sum())
        self.cash_weight = max(0.0, 1.0 - float(new_weights.sum()))
        if portfolio_daily_receiver_exec_guard_mode and bool(portfolio_daily_receiver_target.any()):
            receiver_final_delta = (new_weights - current).replace([np.inf, -np.inf], np.nan).fillna(0.0)
            receiver_final_min_delta = pd.concat(
                [
                    pd.Series(DEFAULT_EXECUTION_DEADBAND_ABS * 1.4, index=prices.index, dtype=float),
                    current.clip(lower=0.0) * 0.012,
                    portfolio_daily_receiver_min_add_delta.replace(0.0, np.nan).fillna(
                        DEFAULT_EXECUTION_DEADBAND_ABS * 1.4
                    )
                    * 0.50,
                ],
                axis=1,
            ).max(axis=1).clip(lower=1.0e-8)
            receiver_final_no_deploy = (
                portfolio_daily_receiver_target
                & (receiver_final_delta <= receiver_final_min_delta)
            )
            if bool(receiver_final_no_deploy.any()):
                receiver_final_reason = pd.Series("final_no_positive_delta", index=prices.index, dtype=object)
                receiver_final_reason = receiver_final_reason.where(
                    ~turnover_intent_guarded,
                    "turnover_budget_trim",
                )
                receiver_final_reason = receiver_final_reason.where(
                    ~budget_dropped,
                    "budget_slot_drop",
                )
                receiver_final_reason = receiver_final_reason.where(
                    ~(translation_cap_guarded | translation_floor_guarded | translation_soft_lift_guarded),
                    "weight_translation_guard",
                )
                receiver_final_reason = receiver_final_reason.where(
                    ~semantic_delta_guarded,
                    "semantic_delta_guard",
                )
                receiver_final_reason = receiver_final_reason.where(
                    ~sell_priority_guarded,
                    "gross_exposure_shrink",
                )
                receiver_final_reason = receiver_final_reason.where(
                    ~(
                        receiver_final_reason.eq("final_no_positive_delta")
                        & (receiver_final_delta <= 1.0e-8)
                    ),
                    "zero_delta_after_allocation",
                )
                portfolio_daily_receiver_exec_guarded = (
                    portfolio_daily_receiver_exec_guarded | receiver_final_no_deploy
                )
                portfolio_daily_receiver_exec_guard_reason = portfolio_daily_receiver_exec_guard_reason.where(
                    ~receiver_final_no_deploy,
                    receiver_final_reason,
                )
                portfolio_daily_receiver_target = portfolio_daily_receiver_target & (~receiver_final_no_deploy)
                direct_action_core_deploy_target = direct_action_core_deploy_target & (~receiver_final_no_deploy)
                direct_action_add_authorized = direct_action_add_authorized & (~receiver_final_no_deploy)
                direct_action_open_authorized = direct_action_open_authorized & (~receiver_final_no_deploy)
                direct_action_deploy_authorized = direct_action_add_authorized | direct_action_open_authorized
                portfolio_daily_receiver_target_count = int(portfolio_daily_receiver_target.sum())
        deploy_intent_candidate_mask = action_names.isin({"open", "add"})
        deploy_intent_candidate_count = int(deploy_intent_candidate_mask.sum())
        deploy_intent_candidate_realized_count = int(
            ((new_weights - current) > 1.0e-8).loc[deploy_intent_candidate_mask].sum()
        )
        deploy_intent_candidate_budget_drop_count = int(
            (
                deploy_intent_candidate_mask
                & (
                    budget_dropped.reindex(prices.index).fillna(False)
                    | forced_zero.reindex(prices.index).fillna(False)
                    | (desired_strength.reindex(prices.index).fillna(0.0) <= 1.0e-12)
                    | (new_weights.reindex(prices.index).fillna(0.0) <= 1.0e-8)
                )
            ).sum()
        )

        actions: list[dict[str, Any]] = []
        for stock in prices.index:
            previous_weight = float(current.get(stock, 0.0))
            new_weight = float(new_weights.get(stock, 0.0))
            model_action = str(policy.at[stock, "action_label"] or "skip")
            model_action_name = model_action.strip().lower()
            portfolio_daily_receiver_action_flag = bool(portfolio_daily_receiver_target.get(stock, False))
            portfolio_daily_receiver_semantic_no_headroom_flag = bool(
                portfolio_daily_receiver_semantic_no_headroom.get(stock, False)
            )
            portfolio_daily_receiver_exec_guarded_action_flag = bool(
                portfolio_daily_receiver_exec_guarded.get(stock, False)
            )
            if portfolio_daily_receiver_action_flag:
                model_action_name = "add" if previous_weight > 1e-8 else "open"
            elif (
                portfolio_daily_ranking_mode
                and model_action_name in {"open", "add"}
                and (
                    portfolio_daily_receiver_semantic_no_headroom_flag
                    or portfolio_daily_receiver_exec_guarded_action_flag
                    or not bool(direct_action_core_deploy_target.get(stock, False))
                )
            ):
                model_action_name = "hold" if previous_weight > 1e-8 else "skip"
            existing = self.holdings.get(stock)
            previous_hold_days = int(existing.hold_days) if existing is not None else 0
            previous_entry_price = float(existing.entry_price) if existing is not None else 0.0
            previous_peak_price = float(existing.peak_price) if existing is not None else 0.0
            current_price = float(prices.get(stock, np.nan))
            exit_timing_pressure = float(exit_timing_pressure_values.get(stock, 0.0))
            exit_urgency_value = float(policy.at[stock, "exit_urgency"] or 0.0) if "exit_urgency" in policy.columns else 0.0
            unrealized_pnl_before = (
                float(current_price / previous_entry_price - 1.0)
                if previous_weight > 1e-8 and previous_entry_price > 0 and np.isfinite(current_price)
                else 0.0
            )
            drawdown_from_peak_before = (
                float(current_price / previous_peak_price - 1.0)
                if previous_weight > 1e-8 and previous_peak_price > 0 and np.isfinite(current_price)
                else 0.0
            )
            if previous_weight <= 1e-8 and new_weight <= 1e-8:
                continue
            weight_change_action = _weight_change_action(previous_weight, new_weight)
            execution_action = weight_change_action
            state_update_action = weight_change_action
            delta_weight = float(new_weight - previous_weight)
            contradictory_micro_rebalance = False
            contradictory_intent_trim = False
            semantic_translation_reason = "legacy_weight_change"
            if previous_weight > 1e-8:
                deadband = max(execution_deadband_abs, previous_weight * execution_deadband_rel)
                effective_deadband = deadband
                if model_action_name == "hold" and delta_weight < 0.0:
                    effective_deadband = max(
                        deadband,
                        execution_deadband_abs * 1.75,
                        previous_weight * (execution_deadband_rel + 0.025),
                    )
                elif model_action_name == "add" and delta_weight < 0.0:
                    effective_deadband = max(
                        deadband,
                        execution_deadband_abs * 1.50,
                        previous_weight * (execution_deadband_rel + 0.020),
                    )
                elif model_action_name == "reduce" and delta_weight > 0.0:
                    effective_deadband = max(
                        deadband,
                        execution_deadband_abs * 1.25,
                        previous_weight * (execution_deadband_rel + 0.010),
                    )
                contradictory_micro_rebalance = abs(delta_weight) <= effective_deadband and (
                    model_action_name == "hold"
                    or (model_action_name == "add" and delta_weight < 0.0)
                    or (model_action_name == "reduce" and delta_weight > 0.0)
                )
                if exit_timing_pressure >= 0.28 and delta_weight < 0.0:
                    contradictory_micro_rebalance = False
                contradictory_intent_trim = (
                    model_action_name == "add"
                    and delta_weight < 0.0
                    and new_weight > 1e-8
                    and previous_hold_days < 18
                    and exit_urgency_value < 0.16
                    and exit_timing_pressure < 0.24
                    and drawdown_from_peak_before > -0.04
                    and unrealized_pnl_before > -0.02
                    and abs(delta_weight) <= max(effective_deadband * 2.5, previous_weight * 0.12)
                    and new_weight >= previous_weight * 0.86
                )
                if execution_semantics == EXECUTION_SEMANTICS_LEGACY:
                    if contradictory_micro_rebalance:
                        execution_action = "hold"
                        state_update_action = "hold"
                    elif contradictory_intent_trim:
                        execution_action = "hold"
                deadband = effective_deadband
            if execution_semantics == EXECUTION_SEMANTICS_SEMANTIC:
                execution_action, semantic_translation_reason = _semantic_execution_action(
                    model_action=model_action,
                    previous_weight=previous_weight,
                    new_weight=new_weight,
                    weight_action=weight_change_action,
                )

            budget_dropped_flag = bool(budget_dropped.get(stock, False))
            budget_released_from_hold_flag = bool(budget_released_from_hold.get(stock, False))
            forced_zero_flag = bool(forced_zero.get(stock, False))
            semantic_delta_guarded_flag = bool(semantic_delta_guarded.get(stock, False))
            budget_split_bound_guarded_flag = bool(budget_split_bound_guarded.get(stock, False))
            translation_floor_guarded_flag = bool(translation_floor_guarded.get(stock, False))
            translation_cap_guarded_flag = bool(translation_cap_guarded.get(stock, False))
            translation_soft_lift_guarded_flag = bool(translation_soft_lift_guarded.get(stock, False))
            sell_priority_guarded_flag = bool(sell_priority_guarded.get(stock, False))
            turnover_intent_guarded_flag = bool(turnover_intent_guarded.get(stock, False))
            sell_source_floor_guarded_flag = bool(sell_source_floor_guarded.get(stock, False))
            model_release_signal_flag = bool(model_release_signal.get(stock, False))
            deploy_funding_rebalance_signal_flag = bool(deploy_funding_rebalance_signal.get(stock, False))
            portfolio_daily_receiver_flag = bool(portfolio_daily_receiver_target.get(stock, False))
            portfolio_daily_receiver_exec_guarded_flag = bool(
                portfolio_daily_receiver_exec_guarded.get(stock, False)
            )
            portfolio_daily_receiver_exec_guard_reason_value = str(
                portfolio_daily_receiver_exec_guard_reason.get(stock, "none") or "none"
            )
            portfolio_daily_source_flag = bool(portfolio_daily_source_target.get(stock, False))
            portfolio_daily_source_exec_guard_flag = bool(
                portfolio_daily_source_exec_guard_mode and portfolio_daily_source_flag
            )
            portfolio_daily_source_exec_cap_guarded_flag = bool(
                portfolio_daily_source_exec_cap_guarded.get(stock, False)
            )
            direct_pair_reallocation_source_flag = bool(direct_action_pair_reallocation_source.get(stock, False))
            sell_authorized_by_model_flag = bool(sell_authorized_mask.get(stock, False))
            sell_authorization_score_value = float(sell_authorization_score.get(stock, 0.0))
            realized_sell = weight_change_action in {"reduce", "exit"}
            sell_intent = model_action_name in {"reduce", "exit"}
            sell_intent_suppressed = sell_intent and not realized_sell
            sell_execution_origin = "none"
            if realized_sell:
                if sell_intent:
                    sell_execution_origin = "model_sell_intent"
                elif model_release_signal_flag:
                    sell_execution_origin = "model_release_signal"
                elif portfolio_daily_source_flag:
                    sell_execution_origin = "portfolio_daily_ranking_source"
                elif deploy_funding_rebalance_signal_flag:
                    sell_execution_origin = "deploy_funding_rebalance"
                elif direct_pair_reallocation_source_flag:
                    sell_execution_origin = "direct_action_pair_reallocation"
                elif forced_zero_flag:
                    sell_execution_origin = "forced_zero"
                elif budget_released_from_hold_flag:
                    sell_execution_origin = "budget_slot_reclaim"
                elif sell_priority_guarded_flag:
                    sell_execution_origin = "budget_sell_priority"
                elif turnover_intent_guarded_flag:
                    sell_execution_origin = "turnover_budget_trim"
                elif translation_cap_guarded_flag:
                    sell_execution_origin = "translation_cap_guard"
                else:
                    sell_execution_origin = "weight_translation"
            portfolio_daily_source_target_not_sold_reason = "none"
            if portfolio_daily_source_flag and not realized_sell:
                if new_weight > previous_weight + 1.0e-8:
                    portfolio_daily_source_target_not_sold_reason = "weight_increased"
                elif float(target_weights.get(stock, new_weight)) >= previous_weight - 1.0e-8:
                    portfolio_daily_source_target_not_sold_reason = "target_weight_not_below_current"
                elif turnover_intent_guarded_flag:
                    portfolio_daily_source_target_not_sold_reason = "turnover_budget_trim"
                elif translation_floor_guarded_flag:
                    portfolio_daily_source_target_not_sold_reason = "translation_floor_guard"
                elif translation_soft_lift_guarded_flag:
                    portfolio_daily_source_target_not_sold_reason = "translation_soft_lift_guard"
                elif budget_split_bound_guarded_flag:
                    portfolio_daily_source_target_not_sold_reason = "budget_split_bound_guard"
                elif sell_source_floor_guarded_flag:
                    portfolio_daily_source_target_not_sold_reason = "sell_source_floor_guard"
                elif protected_floor.get(stock, 0.0) >= previous_weight - 1.0e-8:
                    portfolio_daily_source_target_not_sold_reason = "protected_floor_locked"
                elif abs(new_weight - previous_weight) <= 1.0e-8:
                    portfolio_daily_source_target_not_sold_reason = "zero_delta_after_translation"
                else:
                    portfolio_daily_source_target_not_sold_reason = "weight_translation"
            portfolio_daily_source_realized_reduction_weight = (
                max(previous_weight - new_weight, 0.0) if portfolio_daily_source_flag else 0.0
            )
            sell_suppression_origin = "none"
            if sell_intent_suppressed:
                if semantic_delta_guarded_flag:
                    sell_suppression_origin = "semantic_delta_guard"
                elif translation_floor_guarded_flag:
                    sell_suppression_origin = "translation_floor_guard"
                elif translation_soft_lift_guarded_flag:
                    sell_suppression_origin = "translation_soft_lift_guard"
                elif turnover_intent_guarded_flag:
                    sell_suppression_origin = "turnover_budget_trim"
                elif budget_split_bound_guarded_flag:
                    sell_suppression_origin = "budget_split_bound_guard"
                else:
                    sell_suppression_origin = "weight_translation"

            portfolio_daily_effective_model_action = model_action_name
            if portfolio_daily_receiver_flag:
                portfolio_daily_effective_model_action = "add" if previous_weight > 1e-8 else "open"
            elif (
                portfolio_daily_ranking_mode
                and model_action_name in {"open", "add"}
                and (
                    portfolio_daily_receiver_exec_guarded_flag
                    or not bool(direct_action_core_deploy_target.get(stock, False))
                )
            ):
                portfolio_daily_effective_model_action = str(weight_change_action).strip().lower()

            actions.append(
                {
                    "date": signal_dt.strftime("%Y-%m-%d"),
                    "stock": stock,
                    "source_label": source_label,
                    "model_action": model_action,
                    "portfolio_daily_effective_model_action": portfolio_daily_effective_model_action,
                    "policy_decision_mode": str(policy.at[stock, "policy_decision_mode"] or "") if "policy_decision_mode" in policy.columns else "",
                    "direct_action_value_label": str(policy.at[stock, "direct_action_value_label"] or "") if "direct_action_value_label" in policy.columns else "",
                    "direct_action_value_applied": float(policy.at[stock, "direct_action_value_applied"] or 0.0) if "direct_action_value_applied" in policy.columns else 0.0,
                    "direct_action_value_selected": float(policy.at[stock, "direct_action_value_selected"] or 0.0) if "direct_action_value_selected" in policy.columns else 0.0,
                    "direct_action_value_gap": float(policy.at[stock, "direct_action_value_gap"] or 0.0) if "direct_action_value_gap" in policy.columns else 0.0,
                    "direct_action_utility_skip": float(policy.at[stock, "direct_action_utility_skip"] or 0.0) if "direct_action_utility_skip" in policy.columns else 0.0,
                    "direct_action_utility_open": float(policy.at[stock, "direct_action_utility_open"] or 0.0) if "direct_action_utility_open" in policy.columns else 0.0,
                    "direct_action_utility_hold": float(policy.at[stock, "direct_action_utility_hold"] or 0.0) if "direct_action_utility_hold" in policy.columns else 0.0,
                    "direct_action_utility_add": float(policy.at[stock, "direct_action_utility_add"] or 0.0) if "direct_action_utility_add" in policy.columns else 0.0,
                    "direct_action_utility_reduce": float(policy.at[stock, "direct_action_utility_reduce"] or 0.0) if "direct_action_utility_reduce" in policy.columns else 0.0,
                    "direct_action_utility_exit": float(policy.at[stock, "direct_action_utility_exit"] or 0.0) if "direct_action_utility_exit" in policy.columns else 0.0,
                    "direct_action_keep_utility": float(direct_action_keep_utility_series.get(stock, 0.0)),
                    "direct_action_release_utility": float(direct_action_release_utility_series.get(stock, 0.0)),
                    "direct_action_deploy_utility": float(direct_action_deploy_utility_series.get(stock, 0.0)),
                    "direct_action_release_advantage": float(direct_action_release_advantage_series.get(stock, 0.0)),
                    "direct_action_deploy_advantage": float(direct_action_deploy_advantage_series.get(stock, 0.0)),
                    "direct_action_deploy_rank_score": float(direct_action_deploy_rank_score.get(stock, 0.0)),
                    "direct_action_add_signal": bool(direct_action_add_signal.get(stock, False)),
                    "direct_action_open_signal": bool(direct_action_open_signal.get(stock, False)),
                    "direct_action_deploy_signal": bool(direct_action_deploy_signal.get(stock, False)),
                    "direct_action_core_deploy_target": bool(direct_action_core_deploy_target.get(stock, False)),
                    "direct_action_add_authorized": bool(direct_action_add_authorized.get(stock, False)),
                    "direct_action_open_authorized": bool(direct_action_open_authorized.get(stock, False)),
                    "direct_action_deploy_authorized": bool(direct_action_deploy_authorized.get(stock, False)),
                    "direct_action_funding_release_authorized": bool(direct_action_funding_release_authorized.get(stock, False)),
                    "direct_action_funding_protected": bool(direct_action_funding_protected.get(stock, False)),
                    "direct_action_reallocation_source": bool(direct_action_reallocation_source.get(stock, False)),
                    "direct_action_pair_reallocation_source": direct_pair_reallocation_source_flag,
                    "direct_action_pair_opportunity_spread": float(direct_action_pair_opportunity_spread.get(stock, 0.0)),
                    "direct_action_pair_source_opportunity_cost": float(direct_action_pair_source_opportunity_cost.get(stock, 0.0)),
                    "direct_action_pair_cost_guard_pass": bool(direct_action_pair_cost_guard_pass.get(stock, False)),
                    "direct_action_pair_cost_guard_blocked": bool(direct_action_pair_cost_guard_blocked.get(stock, False)),
                    "direct_action_pair_source_release_score": float(direct_action_pair_source_release_score.get(stock, 0.0)),
                    "portfolio_daily_receiver_candidate": bool(portfolio_daily_receiver_candidate.get(stock, False)),
                    "portfolio_daily_receiver_semantic_no_headroom": bool(
                        portfolio_daily_receiver_semantic_no_headroom.get(stock, False)
                    ),
                    "portfolio_daily_receiver_open_breadth_candidate": bool(
                        portfolio_daily_receiver_open_breadth_candidate.get(stock, False)
                    ),
                    "portfolio_daily_unified_receiver_score": float(
                        portfolio_daily_unified_receiver_score.get(stock, 0.0)
                    ),
                    "portfolio_daily_unified_receiver_candidate": bool(
                        portfolio_daily_unified_receiver_candidate.get(stock, False)
                    ),
                    "portfolio_daily_receiver_score": float(portfolio_daily_receiver_score.get(stock, 0.0)),
                    "portfolio_daily_receiver_target": portfolio_daily_receiver_flag,
                    "portfolio_daily_receiver_exec_guarded": portfolio_daily_receiver_exec_guarded_flag,
                    "portfolio_daily_receiver_exec_guard_reason": portfolio_daily_receiver_exec_guard_reason_value,
                    "portfolio_daily_receiver_add_headroom": float(
                        portfolio_daily_receiver_add_headroom.get(stock, 0.0)
                    ),
                    "portfolio_daily_receiver_min_add_delta": float(
                        portfolio_daily_receiver_min_add_delta.get(stock, 0.0)
                    ),
                    "portfolio_daily_receiver_add_capacity": float(
                        np.clip(
                            (
                                portfolio_daily_receiver_add_headroom.get(stock, 0.0)
                                / max(float(portfolio_daily_receiver_min_add_delta.get(stock, 0.0)), 1.0e-6)
                            )
                            if bool(held_mask.get(stock, False))
                            else 1.0,
                            0.0,
                            1.0,
                        )
                    ),
                    "portfolio_daily_receiver_executability": float(deploy_executability_series.get(stock, 0.0)),
                    "portfolio_daily_receiver_funding_coverage": float(
                        portfolio_daily_receiver_funding_coverage.get(stock, 0.0)
                    ),
                    "portfolio_daily_funding_closure_score": float(
                        portfolio_daily_funding_closure_score.get(stock, 0.0)
                    ),
                    "portfolio_daily_allocation_transfer_score": float(
                        portfolio_daily_allocation_transfer_score.get(stock, 0.0)
                    ),
                    "portfolio_daily_allocation_dead_branch_risk": float(
                        portfolio_daily_allocation_dead_branch_risk.get(stock, 0.0)
                    ),
                    "portfolio_daily_source_gap": float(portfolio_daily_source_gap.get(stock, 0.0)),
                    "portfolio_daily_source_min_release_delta": float(
                        portfolio_daily_source_min_release_delta.get(stock, 0.0)
                    ),
                    "portfolio_daily_source_release_capacity": float(
                        portfolio_daily_source_release_capacity.get(stock, 0.0)
                    ),
                    "portfolio_daily_source_forward_spread_score": float(
                        portfolio_daily_source_forward_spread_score.get(stock, 0.0)
                    ),
                    "portfolio_daily_source_bad_forward_spread_risk": float(
                        portfolio_daily_source_bad_forward_spread_risk.get(stock, 0.0)
                    ),
                    "portfolio_daily_source_economic_release_score": float(
                        portfolio_daily_source_economic_release_score.get(stock, 0.0)
                    ),
                    "portfolio_daily_source_economic_block_risk": float(
                        portfolio_daily_source_economic_block_risk.get(stock, 0.0)
                    ),
                    "portfolio_daily_source_forward_strength_brake_risk": float(
                        portfolio_daily_source_forward_strength_brake_risk.get(stock, 0.0)
                    ),
                    "portfolio_daily_source_forward_strength_brake_pass": bool(
                        portfolio_daily_source_forward_strength_brake_pass.get(stock, False)
                    ),
                    "portfolio_daily_source_forward_proxy_keep_risk": float(
                        portfolio_daily_source_forward_proxy_keep_risk.get(stock, 0.0)
                    ),
                    "portfolio_daily_source_forward_proxy_pass": bool(
                        portfolio_daily_source_forward_proxy_pass.get(stock, False)
                    ),
                    "portfolio_daily_source_direct_release_relief_score": float(
                        portfolio_daily_source_direct_release_relief_score.get(stock, 0.0)
                    ),
                    "portfolio_daily_source_direct_release_relief_pass": bool(
                        portfolio_daily_source_direct_release_relief_pass.get(stock, False)
                    ),
                    "portfolio_daily_source_release_conviction": float(
                        portfolio_daily_source_release_conviction.get(stock, 0.0)
                    ),
                    "portfolio_daily_source_release_conviction_pass": bool(
                        portfolio_daily_source_release_conviction_pass.get(stock, False)
                    ),
                    "portfolio_daily_source_distribution_clean_pass": bool(
                        portfolio_daily_source_distribution_clean_pass.get(stock, False)
                    ),
                    "portfolio_daily_source_release_quality": float(
                        portfolio_daily_source_release_quality.get(stock, 0.0)
                    ),
                    "portfolio_daily_source_opportunity_cost": float(
                        portfolio_daily_source_opportunity_cost.get(stock, 0.0)
                    ),
                    "portfolio_daily_source_executability": float(
                        portfolio_daily_source_executability.get(stock, 0.0)
                    ),
                    "portfolio_daily_unified_source_score": float(
                        portfolio_daily_unified_source_score.get(stock, 0.0)
                    ),
                    "portfolio_daily_source_positive_forward_penalty": float(
                        portfolio_daily_unified_source_positive_forward_penalty.get(stock, 0.0)
                    ),
                    "portfolio_daily_source_opportunity_cost_penalty": float(
                        portfolio_daily_unified_source_opportunity_cost_penalty.get(stock, 0.0)
                    ),
                    "portfolio_daily_receiver_source_spread_reward": float(
                        portfolio_daily_unified_receiver_source_spread_reward.get(stock, 0.0)
                    ),
                    "portfolio_daily_unified_allocation_objective": float(
                        portfolio_daily_unified_allocation_objective.get(stock, 0.0)
                    ),
                    "portfolio_daily_source_score": float(portfolio_daily_source_score.get(stock, 0.0)),
                    "portfolio_daily_source_semantic_release_pass": bool(
                        portfolio_daily_source_semantic_release_pass.get(stock, False)
                    ),
                    "portfolio_daily_source_quality_gap_pass": bool(
                        portfolio_daily_source_quality_gap_pass.get(stock, False)
                    ),
                    "portfolio_daily_source_protected_release_override": bool(
                        protected_source_release_override.get(stock, False)
                    ),
                    "portfolio_daily_source_protected_release_boost": float(
                        protected_source_release_boost.get(stock, 0.0)
                    ),
                    "portfolio_daily_source_repeat_release_pass": bool(
                        portfolio_daily_source_repeat_release_pass.get(stock, False)
                    ),
                    "portfolio_daily_source_recent_sell_days": float(
                        portfolio_daily_source_recent_sell_days.get(stock, 999.0)
                    ),
                    "portfolio_daily_unified_source_candidate": bool(
                        portfolio_daily_unified_source_candidate.get(stock, False)
                    ),
                    "portfolio_daily_source_candidate": bool(portfolio_daily_source_candidate.get(stock, False)),
                    "portfolio_daily_source_target": portfolio_daily_source_flag,
                    "portfolio_daily_source_exec_guard": portfolio_daily_source_exec_guard_flag,
                    "portfolio_daily_source_execution_pressure": float(
                        portfolio_daily_source_execution_pressure.get(stock, 0.0)
                    ),
                    "portfolio_daily_source_retention_floor": float(
                        portfolio_daily_source_retention_floor.get(stock, 1.0)
                    ),
                    "portfolio_daily_source_exec_cap_guarded": portfolio_daily_source_exec_cap_guarded_flag,
                    "portfolio_daily_source_target_not_sold_reason": portfolio_daily_source_target_not_sold_reason,
                    "portfolio_daily_source_realized_reduction_weight": float(
                        portfolio_daily_source_realized_reduction_weight
                    ),
                    "portfolio_daily_unified_cash_score": float(
                        portfolio_daily_unified_cash_score.get(stock, 0.0)
                    ),
                    "portfolio_daily_cash_score": float(portfolio_daily_cash_score),
                    "portfolio_daily_cash_reserve_signal": bool(portfolio_daily_cash_reserve_signal),
                    "execution_action": execution_action,
                    "weight_change_action": weight_change_action,
                    "execution_semantics": execution_semantics,
                    "budget_semantics": budget_semantics,
                    "budget_calibration": budget_calibration,
                    "semantic_translation_reason": semantic_translation_reason,
                    "semantic_preserved": bool(
                        portfolio_daily_effective_model_action == str(execution_action).strip().lower()
                    ),
                    "budget_dropped": budget_dropped_flag,
                    "budget_released_from_hold": budget_released_from_hold_flag,
                    "forced_zero": forced_zero_flag,
                    "semantic_delta_guarded": semantic_delta_guarded_flag,
                    "budget_split_bound_guarded": budget_split_bound_guarded_flag,
                    "translation_floor_guarded": translation_floor_guarded_flag,
                    "translation_cap_guarded": translation_cap_guarded_flag,
                    "translation_soft_lift_guarded": translation_soft_lift_guarded_flag,
                    "sell_priority_guarded": sell_priority_guarded_flag,
                    "turnover_intent_guarded": turnover_intent_guarded_flag,
                    "sell_source_floor_guarded": sell_source_floor_guarded_flag,
                    "model_release_signal": model_release_signal_flag,
                    "deploy_funding_rebalance_signal": deploy_funding_rebalance_signal_flag,
                    "sell_authorized_by_model": sell_authorized_by_model_flag,
                    "sell_authorization_score": sell_authorization_score_value,
                    "sell_execution_origin": sell_execution_origin,
                    "sell_suppression_origin": sell_suppression_origin,
                    "desired_strength": float(desired_strength.get(stock, 0.0)),
                    "protected_floor": float(protected_floor.get(stock, 0.0)),
                    "current_weight": previous_weight,
                    "target_weight": new_weight,
                    "delta_weight": delta_weight,
                    "current_price": current_price,
                    "entry_price_before": previous_entry_price,
                    "peak_price_before": previous_peak_price,
                    "hold_days_before": previous_hold_days,
                    "hold_days_after": previous_hold_days + 1 if new_weight > 1e-8 else previous_hold_days,
                    "unrealized_pnl_before": unrealized_pnl_before,
                    "drawdown_from_peak_before": drawdown_from_peak_before,
                    "action_strength": float(policy.at[stock, "action_strength"] or 0.0),
                    "target_delta_hint": float(policy.at[stock, "target_delta_hint"] or 0.0),
                    "exit_urgency": exit_urgency_value,
                    "reduce_fraction": float(policy.at[stock, "reduce_fraction"] or 0.0) if "reduce_fraction" in policy.columns else 0.0,
                    "exit_hazard": float(policy.at[stock, "exit_hazard"] or 0.0) if "exit_hazard" in policy.columns else 0.0,
                    "sell_pressure": float(policy.at[stock, "sell_pressure"] or 0.0) if "sell_pressure" in policy.columns else 0.0,
                    "sell_attribution_score": float(policy.at[stock, "sell_attribution_score"] or 0.0) if "sell_attribution_score" in policy.columns else 0.0,
                    "sell_rank_score": float(policy.at[stock, "sell_rank_score"] or 0.0) if "sell_rank_score" in policy.columns else 0.0,
                    "lifecycle_sell_gate": float(policy.at[stock, "lifecycle_sell_gate"] or 0.0) if "lifecycle_sell_gate" in policy.columns else 0.0,
                    "large_upside_1d_target": float(policy.at[stock, "large_upside_1d_target"] or 0.0) if "large_upside_1d_target" in policy.columns else 0.0,
                    "alpha_opportunity_value": float(policy.at[stock, "alpha_opportunity_value"] or 0.0) if "alpha_opportunity_value" in policy.columns else 0.0,
                    "hold_continuation_value": float(policy.at[stock, "hold_continuation_value"] or 0.0) if "hold_continuation_value" in policy.columns else 0.0,
                    "sell_release_value": float(policy.at[stock, "sell_release_value"] or 0.0) if "sell_release_value" in policy.columns else 0.0,
                    "cash_defense_value": float(policy.at[stock, "cash_defense_value"] or 0.0) if "cash_defense_value" in policy.columns else 0.0,
                    "deployment_opportunity_cost": float(policy.at[stock, "deployment_opportunity_cost"] or 0.0) if "deployment_opportunity_cost" in policy.columns else 0.0,
                    "risk_adjusted_action_value": float(policy.at[stock, "risk_adjusted_action_value"] or 0.0) if "risk_adjusted_action_value" in policy.columns else 0.0,
                    "multi_horizon_forward_value": float(policy.at[stock, "multi_horizon_forward_value"] or 0.0) if "multi_horizon_forward_value" in policy.columns else 0.0,
                    "multi_horizon_forward_risk": float(policy.at[stock, "multi_horizon_forward_risk"] or 0.0) if "multi_horizon_forward_risk" in policy.columns else 0.0,
                    "multi_horizon_path_value": float(policy.at[stock, "multi_horizon_path_value"] or 0.0) if "multi_horizon_path_value" in policy.columns else 0.0,
                    "open_action_value": float(policy.at[stock, "open_action_value"] or 0.0) if "open_action_value" in policy.columns else 0.0,
                    "add_action_value": float(policy.at[stock, "add_action_value"] or 0.0) if "add_action_value" in policy.columns else 0.0,
                    "hold_action_value": float(policy.at[stock, "hold_action_value"] or 0.0) if "hold_action_value" in policy.columns else 0.0,
                    "reduce_action_value": float(policy.at[stock, "reduce_action_value"] or 0.0) if "reduce_action_value" in policy.columns else 0.0,
                    "exit_action_value": float(policy.at[stock, "exit_action_value"] or 0.0) if "exit_action_value" in policy.columns else 0.0,
                    "relative_opportunity_value": float(policy.at[stock, "relative_opportunity_value"] or 0.0) if "relative_opportunity_value" in policy.columns else 0.0,
                    "action_value_consistency_target": float(policy.at[stock, "action_value_consistency_target"] or 0.5) if "action_value_consistency_target" in policy.columns else 0.5,
                    "value_arbitration_target": float(policy.at[stock, "value_arbitration_target"] or 0.5) if "value_arbitration_target" in policy.columns else 0.5,
                    "deploy_value_target": float(policy.at[stock, "deploy_value_target"] or 0.0) if "deploy_value_target" in policy.columns else 0.0,
                    "release_value_target": float(policy.at[stock, "release_value_target"] or 0.0) if "release_value_target" in policy.columns else 0.0,
                    "defense_value_target": float(policy.at[stock, "defense_value_target"] or 0.0) if "defense_value_target" in policy.columns else 0.0,
                    "deploy_gate_target": float(policy.at[stock, "deploy_gate_target"] or 0.0) if "deploy_gate_target" in policy.columns else 0.0,
                    "release_gate_target": float(policy.at[stock, "release_gate_target"] or 0.0) if "release_gate_target" in policy.columns else 0.0,
                    "defense_gate_target": float(policy.at[stock, "defense_gate_target"] or 0.0) if "defense_gate_target" in policy.columns else 0.0,
                    "deploy_executability_target": float(policy.at[stock, "deploy_executability_target"] or 0.0) if "deploy_executability_target" in policy.columns else 0.0,
                    "clipped_intent_risk": float(policy.at[stock, "clipped_intent_risk"] or 0.0) if "clipped_intent_risk" in policy.columns else 0.0,
                    "exit_timing_pressure": float(policy.at[stock, "exit_timing_pressure"] or 0.0) if "exit_timing_pressure" in policy.columns else 0.0,
                    "execution_deadband": float(deadband if previous_weight > 1e-8 else 0.0),
                    "contradictory_micro_rebalance": bool(contradictory_micro_rebalance),
                    "state_update_action": state_update_action,
                }
            )

        next_holdings: dict[str, HoldingState] = {}
        next_last_buy_dates = dict(self.last_buy_dates)
        next_last_sell_dates = dict(self.last_sell_dates)
        next_last_reduce_dates = dict(self.last_reduce_dates)
        next_last_exit_dates = dict(self.last_exit_dates)
        next_last_action_labels = dict(self.last_action_labels)
        for stock, weight in new_weights.items():
            weight_value = float(weight)
            if weight_value <= 1e-8:
                continue
            price = float(prices.get(stock, np.nan))
            if not np.isfinite(price) or price <= 0:
                price = float(self.holdings.get(stock, HoldingState(weight_value, 0.0, 0.0)).entry_price or 0.0)
            existing = self.holdings.get(stock)
            if existing is None or float(existing.weight) <= 1e-8:
                next_holdings[stock] = HoldingState(
                    weight=weight_value,
                    entry_price=price,
                    peak_price=price,
                    hold_days=0,
                )
                continue
            next_holdings[stock] = HoldingState(
                weight=weight_value,
                entry_price=float(existing.entry_price or price),
                peak_price=max(float(existing.peak_price or price), price),
                hold_days=int(existing.hold_days) + 1,
            )

        signal_date_text = signal_dt.strftime("%Y-%m-%d")
        for item in actions:
            stock = str(item.get("stock", "") or "")
            execution_action = str(item.get("execution_action", "") or "").strip().lower()
            state_update_action = str(item.get("state_update_action", execution_action) or execution_action).strip().lower()
            if not stock or state_update_action not in {"open", "add", "reduce", "exit", "hold"}:
                continue
            next_last_action_labels[stock] = state_update_action
            if state_update_action in {"open", "add"}:
                next_last_buy_dates[stock] = signal_date_text
            if state_update_action in {"reduce", "exit"}:
                next_last_sell_dates[stock] = signal_date_text
            if state_update_action == "reduce":
                next_last_reduce_dates[stock] = signal_date_text
            if state_update_action == "exit":
                next_last_exit_dates[stock] = signal_date_text

        self.holdings = next_holdings
        self.last_buy_dates = next_last_buy_dates
        self.last_sell_dates = next_last_sell_dates
        self.last_reduce_dates = next_last_reduce_dates
        self.last_exit_dates = next_last_exit_dates
        self.last_action_labels = next_last_action_labels
        weight_delta = new_weights - current
        realized_turnover = float(weight_delta.abs().sum())
        buy_turnover = float(weight_delta.clip(lower=0.0).sum())
        sell_turnover = float((-weight_delta.clip(upper=0.0)).sum())
        action_count = max(len(actions), 1)

        def _intent_action(item: dict[str, Any]) -> str:
            return str(
                item.get("portfolio_daily_effective_model_action", item.get("model_action", "")) or ""
            ).strip().lower()

        semantic_conflict_count = sum(
            1
            for item in actions
            if _intent_action(item) != str(item.get("execution_action", "") or "").strip().lower()
        )
        order_translation_conflict_count = sum(
            1
            for item in actions
            if _intent_action(item) != str(item.get("weight_change_action", "") or "").strip().lower()
        )
        deploy_intent_items = [
            item
            for item in actions
            if _intent_action(item) in {"open", "add"}
        ]
        add_intent_items = [
            item
            for item in actions
            if _intent_action(item) == "add"
        ]
        deploy_realized_count = sum(
            1
            for item in deploy_intent_items
            if str(item.get("weight_change_action", "") or "").strip().lower() in {"open", "add"}
        )
        deploy_positive_delta_count = sum(
            1
            for item in deploy_intent_items
            if float(item.get("delta_weight", 0.0) or 0.0) > 1.0e-8
        )
        deploy_intent_unrealized_count = max(0, len(deploy_intent_items) - deploy_positive_delta_count)
        authorized_add_items = [
            item
            for item in actions
            if bool(item.get("direct_action_add_authorized", False))
        ]
        authorized_add_no_weight_change_count = sum(
            1
            for item in authorized_add_items
            if str(item.get("weight_change_action", "") or "").strip().lower() != "add"
            or float(item.get("delta_weight", 0.0) or 0.0) <= 1.0e-8
        )
        receiver_authorization_subset_violation_count = sum(
            1
            for item in actions
            if (
                bool(item.get("direct_action_add_authorized", False))
                or bool(item.get("direct_action_open_authorized", False))
            )
            and (
                not bool(item.get("portfolio_daily_receiver_target", False))
                or not bool(item.get("portfolio_daily_receiver_candidate", False))
            )
        )
        sell_intent_items = [
            item
            for item in actions
            if _intent_action(item) in {"reduce", "exit"}
        ]
        realized_sell_items = [
            item
            for item in actions
            if str(item.get("weight_change_action", "") or "").strip().lower() in {"reduce", "exit"}
        ]
        sell_intent_realized_count = sum(
            1
            for item in sell_intent_items
            if str(item.get("weight_change_action", "") or "").strip().lower() in {"reduce", "exit"}
        )
        sell_intent_hold_conflict_count = sum(
            1
            for item in sell_intent_items
            if str(item.get("weight_change_action", "") or "").strip().lower() == "hold"
        )
        sell_intent_suppressed_count = sum(
            1
            for item in sell_intent_items
            if str(item.get("sell_suppression_origin", "") or "").strip().lower() != "none"
        )
        budget_origin_sell_count = sum(
            1
            for item in realized_sell_items
            if str(item.get("sell_execution_origin", "") or "").strip().lower()
            not in {
                "model_sell_intent",
                "model_release_signal",
                "deploy_funding_rebalance",
                "direct_action_pair_reallocation",
                "portfolio_daily_ranking_source",
            }
        )
        model_release_signal_sell_count = sum(
            1
            for item in realized_sell_items
            if str(item.get("sell_execution_origin", "") or "").strip().lower() == "model_release_signal"
        )
        deploy_funding_rebalance_sell_count = sum(
            1
            for item in realized_sell_items
            if str(item.get("sell_execution_origin", "") or "").strip().lower() == "deploy_funding_rebalance"
        )
        direct_action_pair_reallocation_sell_count = sum(
            1
            for item in realized_sell_items
            if str(item.get("sell_execution_origin", "") or "").strip().lower() == "direct_action_pair_reallocation"
        )
        portfolio_daily_ranking_source_sell_count = sum(
            1
            for item in realized_sell_items
            if str(item.get("sell_execution_origin", "") or "").strip().lower() == "portfolio_daily_ranking_source"
        )
        budget_slot_reclaim_sell_count = sum(
            1
            for item in realized_sell_items
            if str(item.get("sell_execution_origin", "") or "").strip().lower() == "budget_slot_reclaim"
        )
        sell_priority_guard_sell_count = sum(
            1
            for item in realized_sell_items
            if str(item.get("sell_execution_origin", "") or "").strip().lower() == "budget_sell_priority"
        )
        turnover_trim_sell_count = sum(
            1
            for item in realized_sell_items
            if str(item.get("sell_execution_origin", "") or "").strip().lower() == "turnover_budget_trim"
        )
        forced_zero_sell_count = sum(
            1
            for item in realized_sell_items
            if str(item.get("sell_execution_origin", "") or "").strip().lower() == "forced_zero"
        )
        add_to_hold_conflict_count = sum(
            1
            for item in add_intent_items
            if str(item.get("weight_change_action", "") or "").strip().lower() == "hold"
        )
        deploy_hold_conflict_count = sum(
            1
            for item in deploy_intent_items
            if str(item.get("weight_change_action", "") or "").strip().lower() == "hold"
        )
        deploy_intent_dropped_count = sum(
            1
            for item in deploy_intent_items
            if bool(item.get("budget_dropped", False))
            or bool(item.get("forced_zero", False))
            or float(item.get("target_weight", 0.0) or 0.0) <= 1.0e-8
        )
        deploy_intent_count = len(deploy_intent_items)
        add_intent_count = len(add_intent_items)
        portfolio_daily_source_target_items = [
            item
            for item in actions
            if bool(item.get("portfolio_daily_source_target", False))
        ]
        portfolio_daily_source_realized_items = [
            item
            for item in portfolio_daily_source_target_items
            if str(item.get("weight_change_action", "") or "").strip().lower() in {"reduce", "exit"}
        ]
        portfolio_daily_source_not_sold_items = [
            item
            for item in portfolio_daily_source_target_items
            if str(item.get("weight_change_action", "") or "").strip().lower() not in {"reduce", "exit"}
        ]
        portfolio_daily_receiver_target_items = [
            item
            for item in actions
            if bool(item.get("portfolio_daily_receiver_target", False))
        ]
        portfolio_daily_receiver_realized_items = [
            item
            for item in portfolio_daily_receiver_target_items
            if str(item.get("weight_change_action", "") or "").strip().lower() in {"open", "add"}
        ]
        portfolio_daily_receiver_unrealized_items = [
            item
            for item in portfolio_daily_receiver_target_items
            if str(item.get("weight_change_action", "") or "").strip().lower() not in {"open", "add"}
        ]
        portfolio_daily_receiver_exec_guard_count = sum(
            1 for item in actions if bool(item.get("portfolio_daily_receiver_exec_guarded", False))
        )
        portfolio_daily_source_exec_guard_count = sum(
            1 for item in portfolio_daily_source_target_items if bool(item.get("portfolio_daily_source_exec_guard", False))
        )
        portfolio_daily_source_exec_cap_guard_count = sum(
            1
            for item in portfolio_daily_source_target_items
            if bool(item.get("portfolio_daily_source_exec_cap_guarded", False))
        )
        portfolio_daily_source_realized_reduction_weight = sum(
            float(item.get("portfolio_daily_source_realized_reduction_weight", 0.0) or 0.0)
            for item in portfolio_daily_source_target_items
        )
        portfolio_daily_source_target_not_sold_count = len(portfolio_daily_source_not_sold_items)
        portfolio_daily_source_target_count_from_actions = len(portfolio_daily_source_target_items)
        portfolio_daily_effective_capital_transfer_count = min(
            len(portfolio_daily_source_realized_items),
            len(portfolio_daily_receiver_realized_items),
        )
        self.recent_turnovers.append(realized_turnover)
        if len(self.recent_turnovers) > 60:
            self.recent_turnovers = self.recent_turnovers[-60:]
        self.recent_cash_weights.append(float(self.cash_weight))
        if len(self.recent_cash_weights) > 60:
            self.recent_cash_weights = self.recent_cash_weights[-60:]
        for event in self.recent_action_events:
            event["days_ago"] = int(event.get("days_ago", 999) or 999) + 1
        self.recent_action_events = [event for event in self.recent_action_events if int(event.get("days_ago", 999) or 999) <= 30]
        for item in actions:
            state_update_action = str(item.get("state_update_action", item.get("execution_action", "")) or "").strip().lower()
            if state_update_action not in {"open", "add", "reduce", "exit"}:
                continue
            self.recent_action_events.append(
                {
                    "date": signal_date_text,
                    "stock": str(item.get("stock", "") or "").strip().upper(),
                    "execution_action": state_update_action,
                    "days_ago": 0,
                }
            )
        self.recent_action_events = self.recent_action_events[-240:]
        self.last_signal_date = signal_date_text
        diagnostics = {
            "execution_semantics": execution_semantics,
            "budget_semantics": budget_semantics,
            "budget_calibration": budget_calibration,
            "gross_exposure_target_raw": gross_exposure_target_raw,
            "gross_exposure_target": gross_exposure_target,
            "candidate_budget_raw": candidate_budget_raw,
            "candidate_budget": candidate_budget,
            "turnover_budget_raw": turnover_budget_raw,
            "turnover_budget": turnover_budget,
            "max_position_weight_target_raw": position_cap_target_raw,
            "max_position_weight_target": position_cap_target,
            "hold_bias_target": hold_bias_target,
            "reduce_bias_target": reduce_bias_target,
            "exit_patience_target": exit_patience_target,
            "reentry_guard_target": reentry_guard_target,
            "budget_risk_off_score": budget_risk_off_score,
            "budget_deploy_score": budget_deploy_score,
            "budget_model_risk_signal": float((global_targets or {}).get("budget_model_risk_signal", 0.0) or 0.0),
            "budget_model_deploy_signal": float((global_targets or {}).get("budget_model_deploy_signal", 0.0) or 0.0),
            "budget_model_cash_timing_signal": float((global_targets or {}).get("budget_model_cash_timing_signal", 0.0) or 0.0),
            "budget_model_alpha_focus_signal": float((global_targets or {}).get("budget_model_alpha_focus_signal", 0.0) or 0.0),
            "budget_model_risk_deploy_gap": float((global_targets or {}).get("budget_model_risk_deploy_gap", 0.0) or 0.0),
            "budget_model_value_arbitration_signal": budget_model_value_arbitration_signal,
            "budget_model_alpha_opportunity_signal": budget_model_alpha_opportunity_signal,
            "budget_model_cash_defense_signal": budget_model_cash_defense_signal,
            "budget_model_deploy_value_signal": budget_model_deploy_value_signal,
            "budget_model_release_value_signal": budget_model_release_value_signal,
            "budget_model_defense_value_signal": budget_model_defense_value_signal,
            "budget_model_deploy_gate_signal": budget_model_deploy_gate_signal,
            "budget_model_release_gate_signal": budget_model_release_gate_signal,
            "budget_model_defense_gate_signal": budget_model_defense_gate_signal,
            "budget_head_layout": str((global_targets or {}).get("budget_head_layout", "") or ""),
            "held_sell_action_share": held_sell_action_share,
            "held_exit_action_share": held_exit_action_share,
            "held_add_action_share": held_add_action_share,
            "flat_entry_action_share": flat_entry_action_share,
            "held_sell_pressure": held_sell_pressure,
            "held_sell_attribution": held_sell_attribution,
            "held_sell_rank": held_sell_rank,
            "held_lifecycle_sell_gate": held_lifecycle_sell_gate,
            "held_hold_continuation_value": held_hold_continuation_value,
            "held_sell_release_value": held_sell_release_value,
            "held_cash_defense_value": held_cash_defense_value,
            "flat_alpha_opportunity_value": flat_alpha_opportunity_value,
            "flat_deployment_opportunity_cost": flat_deployment_opportunity_cost,
            "avg_value_arbitration_target": avg_value_arbitration_target,
            "avg_alpha_opportunity_value": avg_alpha_opportunity_value,
            "avg_cash_defense_value": avg_cash_defense_value,
            "avg_deploy_value_target": avg_deploy_value_target,
            "avg_release_value_target": avg_release_value_target,
            "avg_defense_value_target": avg_defense_value_target,
            "avg_deploy_gate_target": avg_deploy_gate_target,
            "avg_release_gate_target": avg_release_gate_target,
            "avg_defense_gate_target": avg_defense_gate_target,
            "flat_deploy_executability_target": flat_deploy_executability,
            "held_deploy_executability_target": held_deploy_executability,
            "avg_deploy_executability_target": avg_deploy_executability_target,
            "held_clipped_intent_risk": held_clipped_intent_risk,
            "avg_clipped_intent_risk": avg_clipped_intent_risk,
            "held_exit_timing_pressure": held_exit_timing_pressure,
            "held_exit_hazard": held_exit_hazard,
            "budget_entry_candidate_count": int(budget_entry_candidate_count),
            "budget_entry_keep_count": int(budget_entry_keep_count),
            "budget_held_protected_count": int(budget_held_protected_count),
            "budget_reclaimable_held_count": int(budget_reclaimable_held_count),
            "budget_released_held_count": int(budget_released_held_count),
            "model_release_signal_count": int(model_release_signal.sum()),
            "deploy_funding_rebalance_signal_count": int(deploy_funding_rebalance_signal.sum()),
            "direct_action_preserving_mode": float(bool(direct_action_preserving_mode)),
            "direct_action_pair_cost_guard_mode": float(bool(direct_action_pair_cost_guard_mode)),
            "portfolio_daily_ranking_mode": float(bool(portfolio_daily_ranking_mode)),
            "allocation_layer_primary_mode": float(bool(end_to_end_allocation_layer_mode)),
            "allocation_layer_expected_turnover": float(allocation_layer_expected_turnover),
            "allocation_layer_buy_turnover": float(allocation_layer_buy_turnover),
            "allocation_layer_sell_turnover": float(allocation_layer_sell_turnover),
            "allocation_layer_cash_after": float(allocation_layer_cash_after),
            "allocation_layer_available_cash_to_deploy": float(allocation_layer_available_cash_to_deploy),
            "allocation_layer_objective_value": float(allocation_layer_objective_value),
            "allocation_layer_constraint_violations": float(allocation_layer_constraint_violations),
            "allocation_layer_receiver_executable_candidate_count": int(
                allocation_layer_receiver_executable_candidate.sum()
            ),
            "allocation_layer_source_executable_candidate_count": int(
                allocation_layer_source_executable_candidate.sum()
            ),
            "allocation_layer_receiver_target_count": int(portfolio_daily_receiver_target.sum())
            if end_to_end_allocation_layer_mode
            else 0,
            "allocation_layer_source_target_count": int(portfolio_daily_source_target.sum())
            if end_to_end_allocation_layer_mode
            else 0,
            "direct_action_funding_release_authorized_count": int(direct_action_funding_release_authorized.sum()),
            "direct_action_funding_protected_count": int(direct_action_funding_protected.sum()),
            "direct_action_add_signal_count": int(direct_action_add_signal.sum()),
            "direct_action_open_signal_count": int(direct_action_open_signal.sum()),
            "direct_action_core_deploy_target_count": int(direct_action_core_deploy_target.sum()),
            "direct_action_add_authorized_count": int(direct_action_add_authorized.sum()),
            "direct_action_open_authorized_count": int(direct_action_open_authorized.sum()),
            "direct_action_authorization_subset_violation_count": int(
                receiver_authorization_subset_violation_count
            ),
            "direct_action_reallocation_source_count": int(direct_action_reallocation_source.sum()),
            "direct_action_pair_reallocation_source_count": int(direct_action_pair_reallocation_source.sum()),
            "direct_action_pair_cost_guard_pass_count": int(direct_action_pair_cost_guard_pass.sum()),
            "direct_action_pair_cost_guard_blocked_count": int(direct_action_pair_cost_guard_blocked.sum()),
            "direct_action_pair_source_spread_mean": _masked_mean(
                direct_action_pair_opportunity_spread,
                direct_action_pair_reallocation_source,
            ),
            "direct_action_pair_source_cost_mean": _masked_mean(
                direct_action_pair_source_opportunity_cost,
                direct_action_pair_reallocation_source,
            ),
            "portfolio_daily_unified_allocation_policy_mode": float(bool(unified_allocation_policy_mode)),
            "portfolio_daily_source_exec_guard_mode": float(bool(portfolio_daily_source_exec_guard_mode)),
            "portfolio_daily_receiver_exec_guard_mode": float(bool(portfolio_daily_receiver_exec_guard_mode)),
            "portfolio_daily_unified_receiver_candidate_count": int(portfolio_daily_unified_receiver_candidate.sum()),
            "portfolio_daily_unified_source_candidate_count": int(portfolio_daily_unified_source_candidate.sum()),
            "portfolio_daily_unified_receiver_score_mean": float(
                portfolio_daily_unified_receiver_score.replace([np.inf, -np.inf], np.nan).dropna().mean()
            )
            if len(portfolio_daily_unified_receiver_score.dropna())
            else 0.0,
            "portfolio_daily_unified_source_score_mean": _masked_mean(
                portfolio_daily_unified_source_score,
                held_mask,
            ),
            "portfolio_daily_unified_cash_score_mean": float(
                portfolio_daily_unified_cash_score.replace([np.inf, -np.inf], np.nan).dropna().mean()
            )
            if len(portfolio_daily_unified_cash_score.dropna())
            else 0.0,
            "portfolio_daily_source_positive_forward_penalty_mean": _masked_mean(
                portfolio_daily_unified_source_positive_forward_penalty,
                portfolio_daily_source_target,
            ),
            "portfolio_daily_source_opportunity_cost_penalty_mean": _masked_mean(
                portfolio_daily_unified_source_opportunity_cost_penalty,
                portfolio_daily_source_target,
            ),
            "portfolio_daily_receiver_source_spread_reward_mean": _masked_mean(
                portfolio_daily_unified_receiver_source_spread_reward,
                portfolio_daily_source_target,
            ),
            "portfolio_daily_unified_allocation_objective_mean": float(
                portfolio_daily_unified_allocation_objective.replace([np.inf, -np.inf], np.nan).dropna().mean()
            )
            if len(portfolio_daily_unified_allocation_objective.dropna())
            else 0.0,
            "portfolio_daily_receiver_candidate_count": int(portfolio_daily_receiver_candidate.sum()),
            "portfolio_daily_receiver_exec_guard_count": int(portfolio_daily_receiver_exec_guard_count),
            "portfolio_daily_receiver_semantic_no_headroom_count": int(
                portfolio_daily_receiver_semantic_no_headroom.sum()
            ),
            "portfolio_daily_receiver_open_breadth_candidate_count": int(
                portfolio_daily_receiver_open_breadth_candidate.sum()
            ),
            "portfolio_daily_receiver_add_headroom_mean": _masked_mean(
                portfolio_daily_receiver_add_headroom,
                portfolio_daily_receiver_exec_guarded,
            ),
            "portfolio_daily_receiver_min_add_delta_mean": _masked_mean(
                portfolio_daily_receiver_min_add_delta,
                portfolio_daily_receiver_exec_guarded,
            ),
            "portfolio_daily_receiver_target_count": int(portfolio_daily_receiver_target.sum()),
            "portfolio_daily_source_candidate_count": int(portfolio_daily_source_candidate.sum()),
            "portfolio_daily_source_target_count": int(portfolio_daily_source_target.sum()),
            "portfolio_daily_source_protected_release_override_count": int(
                protected_source_release_override.sum()
            ),
            "portfolio_daily_source_exec_guard_count": int(portfolio_daily_source_exec_guard_count),
            "portfolio_daily_source_exec_cap_guard_count": int(portfolio_daily_source_exec_cap_guard_count),
            "portfolio_daily_source_target_not_sold_count": int(portfolio_daily_source_target_not_sold_count),
            "portfolio_daily_source_target_not_sold_share": float(
                portfolio_daily_source_target_not_sold_count / portfolio_daily_source_target_count_from_actions
            )
            if portfolio_daily_source_target_count_from_actions
            else 0.0,
            "portfolio_daily_source_realized_reduction_weight": float(
                portfolio_daily_source_realized_reduction_weight
            ),
            "portfolio_daily_receiver_realized_deploy_count": int(len(portfolio_daily_receiver_realized_items)),
            "portfolio_daily_receiver_unrealized_deploy_count": int(len(portfolio_daily_receiver_unrealized_items)),
            "portfolio_daily_receiver_realized_deploy_rate": float(
                len(portfolio_daily_receiver_realized_items) / len(portfolio_daily_receiver_target_items)
            )
            if portfolio_daily_receiver_target_items
            else 0.0,
            "portfolio_daily_receiver_unrealized_deploy_share": float(
                len(portfolio_daily_receiver_unrealized_items) / len(portfolio_daily_receiver_target_items)
            )
            if portfolio_daily_receiver_target_items
            else 0.0,
            "portfolio_daily_effective_capital_transfer_count": int(portfolio_daily_effective_capital_transfer_count),
            "portfolio_daily_receiver_score_mean": _masked_mean(
                portfolio_daily_receiver_score,
                portfolio_daily_receiver_target,
            ),
            "portfolio_daily_source_score_mean": _masked_mean(
                portfolio_daily_source_score,
                portfolio_daily_source_target,
            ),
            "portfolio_daily_source_release_capacity_mean": _masked_mean(
                portfolio_daily_source_release_capacity,
                portfolio_daily_source_target,
            ),
            "portfolio_daily_source_forward_spread_score_mean": _masked_mean(
                portfolio_daily_source_forward_spread_score,
                portfolio_daily_source_target,
            ),
            "portfolio_daily_source_bad_forward_spread_risk_mean": _masked_mean(
                portfolio_daily_source_bad_forward_spread_risk,
                portfolio_daily_source_target,
            ),
            "portfolio_daily_source_economic_release_score_mean": _masked_mean(
                portfolio_daily_source_economic_release_score,
                portfolio_daily_source_target,
            ),
            "portfolio_daily_source_economic_block_risk_mean": _masked_mean(
                portfolio_daily_source_economic_block_risk,
                portfolio_daily_source_target,
            ),
            "portfolio_daily_source_forward_strength_brake_risk_mean": _masked_mean(
                portfolio_daily_source_forward_strength_brake_risk,
                portfolio_daily_source_target,
            ),
            "portfolio_daily_source_forward_proxy_keep_risk_mean": _masked_mean(
                portfolio_daily_source_forward_proxy_keep_risk,
                portfolio_daily_source_target,
            ),
            "portfolio_daily_source_forward_proxy_blocked_count": int(
                (
                    held_mask
                    & portfolio_daily_source_semantic_release_pass
                    & portfolio_daily_source_observable_release_pass
                    & portfolio_daily_source_quality_gap_pass
                    & portfolio_daily_source_repeat_release_pass
                    & portfolio_daily_source_forward_strength_brake_pass
                    & (~portfolio_daily_source_forward_proxy_pass)
                ).sum()
            ),
            "portfolio_daily_source_direct_release_relief_score_mean": _masked_mean(
                portfolio_daily_source_direct_release_relief_score,
                portfolio_daily_source_target,
            ),
            "portfolio_daily_source_release_conviction_mean": _masked_mean(
                portfolio_daily_source_release_conviction,
                portfolio_daily_source_target,
            ),
            "portfolio_daily_source_release_conviction_blocked_count": int(
                (
                    held_mask
                    & portfolio_daily_source_semantic_release_pass
                    & portfolio_daily_source_observable_release_pass
                    & portfolio_daily_source_quality_gap_pass
                    & portfolio_daily_source_repeat_release_pass
                    & portfolio_daily_source_forward_strength_brake_pass
                    & portfolio_daily_source_forward_proxy_pass
                    & (~portfolio_daily_source_release_conviction_pass)
                ).sum()
            ),
            "portfolio_daily_source_distribution_clean_blocked_count": int(
                (
                    held_mask
                    & portfolio_daily_source_semantic_release_pass
                    & portfolio_daily_source_observable_release_pass
                    & portfolio_daily_source_quality_gap_pass
                    & portfolio_daily_source_repeat_release_pass
                    & portfolio_daily_source_forward_strength_brake_pass
                    & portfolio_daily_source_forward_proxy_pass
                    & portfolio_daily_source_release_conviction_pass
                    & (~portfolio_daily_source_distribution_clean_pass)
                ).sum()
            ),
            "portfolio_daily_source_release_quality_mean": _masked_mean(
                portfolio_daily_source_release_quality,
                portfolio_daily_source_target,
            ),
            "portfolio_daily_source_executability_mean": _masked_mean(
                portfolio_daily_source_executability,
                portfolio_daily_source_target,
            ),
            "portfolio_daily_source_gap_mean": _masked_mean(
                portfolio_daily_source_gap,
                portfolio_daily_source_target,
            ),
            "portfolio_daily_receiver_funding_coverage_mean": _masked_mean(
                portfolio_daily_receiver_funding_coverage,
                portfolio_daily_receiver_target,
            ),
            "portfolio_daily_funding_closure_score_mean": _masked_mean(
                portfolio_daily_funding_closure_score,
                portfolio_daily_receiver_target | portfolio_daily_source_target,
            ),
            "portfolio_daily_allocation_transfer_score_mean": _masked_mean(
                portfolio_daily_allocation_transfer_score,
                portfolio_daily_receiver_target | portfolio_daily_source_target,
            ),
            "portfolio_daily_allocation_dead_branch_risk_mean": float(
                portfolio_daily_allocation_dead_branch_risk.mean()
            )
            if len(portfolio_daily_allocation_dead_branch_risk)
            else 0.0,
            "portfolio_daily_cash_score": float(portfolio_daily_cash_score),
            "portfolio_daily_cash_reserve_signal": float(bool(portfolio_daily_cash_reserve_signal)),
            "sell_authorized_held_count": int(sell_authorized_mask.sum()),
            "budget_translation_floor_guard_count": int(translation_floor_guarded.sum()),
            "budget_translation_cap_guard_count": int(translation_cap_guarded.sum()),
            "budget_translation_soft_lift_guard_count": int(translation_soft_lift_guarded.sum()),
            "budget_sell_priority_guard_count": int(sell_priority_guarded.sum()),
            "sell_source_floor_guard_count": int(sell_source_floor_guarded.sum()),
            "execution_deadband_abs": execution_deadband_abs,
            "execution_deadband_rel": execution_deadband_rel,
            "recent_reversal_rate_20d": float(self.portfolio_features().get("recent_reversal_rate_20d", 0.0)),
            "raw_turnover": raw_turnover,
            "realized_turnover": realized_turnover,
            "buy_turnover": buy_turnover,
            "sell_turnover": sell_turnover,
            "semantic_conflict_count": int(semantic_conflict_count),
            "semantic_conflict_rate": float(semantic_conflict_count / action_count),
            "order_translation_conflict_count": int(order_translation_conflict_count),
            "order_translation_conflict_rate": float(order_translation_conflict_count / action_count),
            "deploy_intent_action_count": int(deploy_intent_count),
            "deploy_intent_realized_count": int(deploy_realized_count),
            "deploy_intent_realized_rate": float(deploy_realized_count / deploy_intent_count) if deploy_intent_count else 0.0,
            "deploy_intent_unrealized_count": int(deploy_intent_unrealized_count),
            "deploy_intent_unrealized_share": float(deploy_intent_unrealized_count / deploy_intent_count) if deploy_intent_count else 0.0,
            "open_add_positive_weight_change_rate": float(deploy_positive_delta_count / deploy_intent_count) if deploy_intent_count else 0.0,
            "sell_intent_action_count": int(len(sell_intent_items)),
            "sell_intent_realized_count": int(sell_intent_realized_count),
            "sell_intent_realized_rate": float(sell_intent_realized_count / len(sell_intent_items)) if sell_intent_items else 0.0,
            "sell_intent_hold_conflict_count": int(sell_intent_hold_conflict_count),
            "sell_intent_hold_conflict_share": float(sell_intent_hold_conflict_count / len(sell_intent_items)) if sell_intent_items else 0.0,
            "sell_intent_suppressed_count": int(sell_intent_suppressed_count),
            "sell_intent_suppressed_share": float(sell_intent_suppressed_count / len(sell_intent_items)) if sell_intent_items else 0.0,
            "realized_sell_action_count": int(len(realized_sell_items)),
            "budget_origin_sell_count": int(budget_origin_sell_count),
            "budget_origin_sell_share": float(budget_origin_sell_count / len(realized_sell_items)) if realized_sell_items else 0.0,
            "model_release_signal_sell_count": int(model_release_signal_sell_count),
            "model_release_signal_sell_share": float(model_release_signal_sell_count / len(realized_sell_items)) if realized_sell_items else 0.0,
            "deploy_funding_rebalance_sell_count": int(deploy_funding_rebalance_sell_count),
            "deploy_funding_rebalance_sell_share": float(deploy_funding_rebalance_sell_count / len(realized_sell_items)) if realized_sell_items else 0.0,
            "direct_action_pair_reallocation_sell_count": int(direct_action_pair_reallocation_sell_count),
            "direct_action_pair_reallocation_sell_share": float(direct_action_pair_reallocation_sell_count / len(realized_sell_items)) if realized_sell_items else 0.0,
            "budget_slot_reclaim_sell_count": int(budget_slot_reclaim_sell_count),
            "budget_slot_reclaim_sell_share": float(budget_slot_reclaim_sell_count / len(realized_sell_items)) if realized_sell_items else 0.0,
            "sell_priority_guard_sell_count": int(sell_priority_guard_sell_count),
            "sell_priority_guard_sell_share": float(sell_priority_guard_sell_count / len(realized_sell_items)) if realized_sell_items else 0.0,
            "turnover_trim_sell_count": int(turnover_trim_sell_count),
            "turnover_trim_sell_share": float(turnover_trim_sell_count / len(realized_sell_items)) if realized_sell_items else 0.0,
            "forced_zero_sell_count": int(forced_zero_sell_count),
            "forced_zero_sell_share": float(forced_zero_sell_count / len(realized_sell_items)) if realized_sell_items else 0.0,
            "deploy_intent_dropped_count": int(deploy_intent_dropped_count),
            "deploy_intent_dropped_share": float(deploy_intent_dropped_count / deploy_intent_count) if deploy_intent_count else 0.0,
            "deploy_intent_candidate_count": int(deploy_intent_candidate_count),
            "deploy_intent_candidate_realized_count": int(deploy_intent_candidate_realized_count),
            "deploy_intent_candidate_realized_rate": float(deploy_intent_candidate_realized_count / deploy_intent_candidate_count) if deploy_intent_candidate_count else 0.0,
            "deploy_intent_candidate_budget_drop_count": int(deploy_intent_candidate_budget_drop_count),
            "deploy_intent_candidate_budget_drop_share": float(deploy_intent_candidate_budget_drop_count / deploy_intent_candidate_count) if deploy_intent_candidate_count else 0.0,
            "add_to_hold_conflict_count": int(add_to_hold_conflict_count),
            "add_to_hold_conflict_share": float(add_to_hold_conflict_count / add_intent_count) if add_intent_count else 0.0,
            "authorized_add_no_weight_change_count": int(authorized_add_no_weight_change_count),
            "authorized_add_no_weight_change_share": float(
                authorized_add_no_weight_change_count / len(authorized_add_items)
            )
            if authorized_add_items
            else 0.0,
            "deploy_intent_hold_conflict_share": float(deploy_hold_conflict_count / deploy_intent_count) if deploy_intent_count else 0.0,
            "semantic_delta_guard_count": int(semantic_delta_guarded.sum()),
            "turnover_intent_guard_count": int(turnover_intent_guarded.sum()),
            "budget_split_bound_guard_count": int(budget_split_bound_guarded.sum()),
            "forced_zero_count": int(forced_zero.sum()),
            "budget_drop_count": int(budget_dropped.sum()),
            "cash_weight": float(self.cash_weight),
            "holding_count": int(sum(1 for value in self.holdings.values() if value.weight > 1e-8)),
        }
        return StepResult(
            date=signal_dt.strftime("%Y-%m-%d"),
            weights=new_weights,
            actions=actions,
            diagnostics=diagnostics,
        )
