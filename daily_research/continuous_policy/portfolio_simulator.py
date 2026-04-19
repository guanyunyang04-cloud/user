from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd


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
DEFAULT_BUDGET_SEMANTICS = BUDGET_SEMANTICS_LEGACY
BUDGET_SEMANTICS_CHOICES = (BUDGET_SEMANTICS_LEGACY, BUDGET_SEMANTICS_SPLIT)
BUDGET_CALIBRATION_NONE = "none"
BUDGET_CALIBRATION_CASH_EXIT = "cash_exit_guard_v1"
BUDGET_CALIBRATION_CASH_TRANSLATION = "cash_translation_guard_v2"
BUDGET_CALIBRATION_CASH_TRANSLATION_SELL = "cash_translation_sell_guard_v3"
DEFAULT_BUDGET_CALIBRATION = BUDGET_CALIBRATION_NONE
BUDGET_CALIBRATION_CHOICES = (
    BUDGET_CALIBRATION_NONE,
    BUDGET_CALIBRATION_CASH_EXIT,
    BUDGET_CALIBRATION_CASH_TRANSLATION,
    BUDGET_CALIBRATION_CASH_TRANSLATION_SELL,
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
                "exit_timing_pressure": 0.0,
            }
        )

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
        use_sell_priority_guard = budget_calibration == BUDGET_CALIBRATION_CASH_TRANSLATION_SELL

        action_names = policy["action_label"].astype(str).str.strip().str.lower()

        def _policy_numeric(name: str, default: float = 0.0) -> pd.Series:
            if name not in policy.columns:
                return pd.Series(float(default), index=policy.index, dtype=float)
            return pd.to_numeric(policy[name], errors="coerce").fillna(float(default)).astype(float)

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
        exit_timing_pressure_series = _policy_numeric("exit_timing_pressure")
        exit_hazard_series = _policy_numeric("exit_hazard")
        entry_quality_series = _policy_numeric("entry_quality")
        held_sell_pressure = _masked_mean(sell_pressure_series, held_mask)
        held_sell_attribution = _masked_mean(sell_attribution_series, held_mask)
        held_exit_timing_pressure = _masked_mean(exit_timing_pressure_series, held_mask)
        held_exit_hazard = _masked_mean(exit_hazard_series, held_mask)
        flat_entry_quality = _masked_mean(entry_quality_series.clip(lower=0.0), flat_mask)
        current_gross_exposure = float(current.clip(lower=0.0).sum())
        portfolio_drawdown_20d = float(portfolio_context.get("portfolio_drawdown_20d", 0.0) or 0.0)
        recent_positive_share = float(portfolio_context.get("recent_positive_return_share_20d", 0.0) or 0.0)
        turnover_pressure = float(portfolio_context.get("turnover_pressure", 0.0) or 0.0)
        budget_risk_off_score = float(
            np.clip(
                max(held_sell_pressure - 0.18, 0.0) / 0.45 * 0.30
                + max(held_exit_timing_pressure - 0.22, 0.0) / 0.45 * 0.32
                + max(held_exit_hazard - 0.22, 0.0) / 0.45 * 0.18
                + held_sell_action_share * 0.26
                + held_exit_action_share * 0.10
                + max(-portfolio_drawdown_20d - 0.025, 0.0) / 0.09 * 0.18
                + max(turnover_pressure - 0.65, 0.0) / 0.70 * 0.10
                - recent_positive_share * 0.08,
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
                - budget_risk_off_score * 0.45,
                0.0,
                1.0,
            )
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
        elif budget_calibration in {BUDGET_CALIBRATION_CASH_TRANSLATION, BUDGET_CALIBRATION_CASH_TRANSLATION_SELL}:
            positive_gap = max(budget_model_risk_deploy_gap, 0.0)
            sell_priority_bonus = held_sell_attribution * 0.08 if use_sell_priority_guard else 0.0
            risk_cut = (
                budget_risk_off_score * (0.08 + current_gross_exposure * 0.14)
                + budget_model_cash_timing_signal * 0.12
                + positive_gap * 0.08
                + sell_priority_bonus
            )
            deploy_boost = (
                budget_deploy_score * 0.045
                + budget_model_alpha_focus_signal * 0.015
                if budget_risk_off_score < 0.32 and budget_model_cash_timing_signal < 0.32
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
                        - held_sell_attribution * (0.8 if use_sell_priority_guard else 0.0)
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
                    + held_sell_attribution * (0.08 if use_sell_priority_guard else 0.0)
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
                    - held_sell_attribution * (0.008 if use_sell_priority_guard else 0.0)
                    + budget_deploy_score * 0.008
                    + budget_model_alpha_focus_signal * 0.006,
                    0.05,
                    0.32,
                )
            )
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
                        )
                        and (reduce_quality >= hold_quality + 0.04 or reduce_fraction >= 0.18 or exit_timing_pressure >= 0.52)
                        and sell_pressure >= 0.20
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
                            + sell_attribution_score * 0.16
                            - hold_bias_target * 0.10
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
                ):
                    protected_floor.at[stock] = max(
                        protected_floor.at[stock],
                        current_weight * np.clip(0.72 + hold_bias_target * 0.08 - sell_attribution_score * 0.08, 0.56, 0.86),
                    )
                continue
            if action == "hold":
                hold_scale = (
                    1.0
                    + hold_bias_target * 0.06
                    + exit_patience_target * 0.04
                    + max(planned_holding_days - 3.0, 0.0) / 120.0
                    - sell_pressure * 0.18
                    - exit_hazard * 0.08
                    - exit_timing_pressure * 0.16
                    - sell_attribution_score * 0.10
                )
                desired_strength.at[stock] = max(
                    current_weight * max(0.54 if exit_timing_pressure > 0.62 else 0.72, hold_scale),
                    current_weight
                    + max(0.0, hold_boost + hold_quality - sell_pressure * 0.35 - exit_hazard * 0.18 - exit_timing_pressure * 0.22 - sell_attribution_score * 0.10)
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
                            + max(planned_holding_days - 3.0, 0.0) / 180.0,
                            0.56 + max(0.0, 0.08 - sell_pressure * 0.08 - exit_timing_pressure * 0.08 - sell_attribution_score * 0.06),
                            0.97,
                        ),
                    )
                continue
            if action == "add":
                add_increment = max(
                    0.0,
                    strength * 0.55 + add_quality * 0.15 + planned_holding_days / 300.0 - sell_pressure * 0.14 - exit_hazard * 0.10 - exit_timing_pressure * 0.18,
                )
                desired_strength.at[stock] = (
                    current_weight
                    if (sell_pressure > 0.26 or exit_hazard > 0.20 or exit_timing_pressure > 0.24)
                    else max(current_weight + max(0.015, add_increment), current_weight)
                )
                if current_weight > 1e-8:
                    floor_ratio = float(np.clip(0.90 + hold_bias_target * 0.04 - sell_attribution_score * 0.04, 0.82, 0.98))
                    if (
                        exit_urgency < 0.16
                        and exit_hazard < 0.18
                        and sell_pressure < 0.18
                        and exit_timing_pressure < 0.20
                        and add_quality > max(0.10, hold_quality - 0.02)
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
                desired_strength.at[stock] = max(
                    strength
                    * max(
                        0.25,
                        0.85
                        + hold_bias_target * 0.15
                        - reentry_guard_target * 0.30
                        - reentry_penalty * (0.22 + reentry_guard_target * 0.55)
                        - sell_pressure * 0.20
                        - exit_hazard * 0.12,
                    ),
                    max(delta_hint, 0.02 + entry_quality * 0.20 + planned_holding_days / 320.0) * max(0.65, 1.0 - sell_pressure * 0.35),
                )
                continue
            desired_strength.at[stock] = current_weight * (1.0 + hold_bias_target * 0.02)

        budget_dropped = pd.Series(False, index=prices.index, dtype=bool)
        budget_entry_candidate_count = 0
        budget_entry_keep_count = 0
        budget_held_protected_count = 0
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
            open_slots = max(
                0,
                min(
                    int(self.max_positions) - int(held_survivor_mask.sum()),
                    candidate_limit - int(held_survivor_mask.sum()),
                ),
            )
            entry_keep_mask = pd.Series(False, index=prices.index, dtype=bool)
            if open_slots > 0 and bool(entry_candidate_mask.any()):
                entry_keep = desired_strength.where(entry_candidate_mask, 0.0).nlargest(open_slots).index
                entry_keep_mask = pd.Series(desired_strength.index.isin(entry_keep), index=desired_strength.index, dtype=bool)
            budget_entry_keep_count = int(entry_keep_mask.sum())
            keep_mask = held_lifecycle_mask | entry_keep_mask
            budget_dropped = entry_candidate_mask & (~entry_keep_mask)
            desired_strength = desired_strength.where(keep_mask, 0.0)
        desired_strength = desired_strength.where(~forced_zero, 0.0)
        sell_reduction_priority = (
            sell_attribution_series.clip(0.0, 1.0) * 0.56
            + sell_pressure_series.clip(0.0, 1.0) * 0.22
            + exit_timing_pressure_series.clip(0.0, 1.0) * 0.14
            + action_names.isin({"reduce", "exit"}).astype(float) * 0.14
            - action_names.isin({"hold", "add"}).astype(float) * 0.06
        ).clip(lower=0.0)

        target_weights = self._allocate_with_cap(
            desired_strength,
            gross_exposure_target,
            position_cap=position_cap_target,
        )
        protected_floor = protected_floor.clip(lower=0.0, upper=position_cap_target)
        if float(protected_floor.sum()) > float(gross_exposure_target) > 0.0:
            protected_floor = protected_floor / float(protected_floor.sum()) * float(gross_exposure_target)
        sell_priority_guarded = pd.Series(False, index=prices.index, dtype=bool)
        if bool((protected_floor > 1e-8).any()):
            target_weights = target_weights.where(target_weights >= protected_floor, protected_floor)
            excess = float(target_weights.sum() - gross_exposure_target)
            if excess > 1e-8:
                if use_sell_priority_guard:
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
        translation_floor = protected_floor.copy()
        translation_cap = pd.Series(position_cap_target, index=prices.index, dtype=float)
        if (
            execution_semantics == EXECUTION_SEMANTICS_SEMANTIC
            and budget_semantics == BUDGET_SEMANTICS_SPLIT
            and budget_calibration in {BUDGET_CALIBRATION_CASH_TRANSLATION, BUDGET_CALIBRATION_CASH_TRANSLATION_SELL}
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
                if model_action_name not in {"skip", "open", "hold", "add", "reduce", "exit"}:
                    continue
                if previous_weight > 1e-8:
                    deadband = max(execution_deadband_abs, previous_weight * execution_deadband_rel)
                    if model_action_name == "hold":
                        translation_floor.at[stock] = max(float(translation_floor.get(stock, 0.0)), previous_weight)
                        translation_cap.at[stock] = min(float(translation_cap.get(stock, position_cap_target)), previous_weight)
                        if target_value > previous_weight + 1e-12:
                            target_weights.at[stock] = previous_weight
                            translation_cap_guarded.at[stock] = True
                        elif target_value < previous_weight - 1e-12:
                            target_weights.at[stock] = previous_weight
                            translation_floor_guarded.at[stock] = True
                    elif model_action_name == "add":
                        min_add_weight = min(
                            position_cap_target,
                            previous_weight + max(deadband * 1.35, previous_weight * (0.035 + budget_model_deploy_signal * 0.040), 0.0035),
                        )
                        translation_floor.at[stock] = max(float(translation_floor.get(stock, 0.0)), min_add_weight)
                        if target_value < min_add_weight - 1e-12:
                            target_weights.at[stock] = min_add_weight
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
                        min_open_weight = min(
                            position_cap_target,
                            max(execution_deadband_abs * 2.5, 0.012 + budget_model_deploy_signal * 0.010 + budget_model_alpha_focus_signal * 0.006),
                        )
                        translation_floor.at[stock] = max(float(translation_floor.get(stock, 0.0)), min_open_weight)
                        if target_value < min_open_weight - 1e-12:
                            target_weights.at[stock] = min_open_weight
                            translation_floor_guarded.at[stock] = True
                    elif model_action_name in {"hold", "skip", "reduce", "exit"} and target_value > 1e-12:
                        translation_cap.at[stock] = 0.0
                        target_weights.at[stock] = 0.0
                        translation_cap_guarded.at[stock] = True
            target_weights = target_weights.clip(lower=translation_floor, upper=translation_cap)
            effective_floor = pd.concat([protected_floor.rename("protected"), translation_floor.rename("translation")], axis=1).max(axis=1)
            excess = float(target_weights.sum() - gross_exposure_target)
            if excess > 1e-8:
                if use_sell_priority_guard:
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
                delta_value = float(delta.get(stock, 0.0))
                if abs(delta_value) <= 1e-12:
                    continue
                deadband = max(execution_deadband_abs, previous_weight * execution_deadband_rel)
                exit_timing_pressure = float(exit_timing_pressure_values.get(stock, 0.0))
                sell_pressure = (
                    float(policy.at[stock, "sell_pressure"] or 0.0)
                    if "sell_pressure" in policy.columns
                    else 0.0
                )
                protect_hold_trim = (
                    model_action_name == "hold"
                    and delta_value < 0.0
                    and exit_timing_pressure < 0.30
                    and sell_pressure < 0.28
                    and abs(delta_value) <= max(deadband * 2.75, previous_weight * 0.10)
                )
                protect_add_trim = (
                    model_action_name == "add"
                    and delta_value < 0.0
                    and exit_timing_pressure < 0.28
                    and sell_pressure < 0.26
                    and abs(delta_value) <= max(deadband * 3.00, previous_weight * 0.14)
                )
                protect_reduce_add = (
                    model_action_name == "reduce"
                    and delta_value > 0.0
                    and abs(delta_value) <= max(deadband * 2.00, previous_weight * 0.08)
                )
                if protect_hold_trim or protect_add_trim or protect_reduce_add:
                    delta.at[stock] = 0.0
                    semantic_delta_guarded.at[stock] = True
        new_weights = (current + delta).clip(lower=0.0)
        if float(new_weights.sum()) > 0.999:
            new_weights = new_weights / float(new_weights.sum())
        self.cash_weight = max(0.0, 1.0 - float(new_weights.sum()))

        actions: list[dict[str, Any]] = []
        for stock in prices.index:
            previous_weight = float(current.get(stock, 0.0))
            new_weight = float(new_weights.get(stock, 0.0))
            model_action = str(policy.at[stock, "action_label"] or "skip")
            model_action_name = model_action.strip().lower()
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

            actions.append(
                {
                    "date": signal_dt.strftime("%Y-%m-%d"),
                    "stock": stock,
                    "source_label": source_label,
                    "model_action": model_action,
                    "execution_action": execution_action,
                    "weight_change_action": weight_change_action,
                    "execution_semantics": execution_semantics,
                    "budget_semantics": budget_semantics,
                    "budget_calibration": budget_calibration,
                    "semantic_translation_reason": semantic_translation_reason,
                    "semantic_preserved": bool(model_action_name == str(execution_action).strip().lower()),
                    "budget_dropped": bool(budget_dropped.get(stock, False)),
                    "forced_zero": bool(forced_zero.get(stock, False)),
                    "semantic_delta_guarded": bool(semantic_delta_guarded.get(stock, False)),
                    "budget_split_bound_guarded": bool(budget_split_bound_guarded.get(stock, False)),
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
        semantic_conflict_count = sum(
            1
            for item in actions
            if str(item.get("model_action", "") or "").strip().lower()
            != str(item.get("execution_action", "") or "").strip().lower()
        )
        order_translation_conflict_count = sum(
            1
            for item in actions
            if str(item.get("model_action", "") or "").strip().lower()
            != str(item.get("weight_change_action", "") or "").strip().lower()
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
            "budget_head_layout": str((global_targets or {}).get("budget_head_layout", "") or ""),
            "held_sell_action_share": held_sell_action_share,
            "held_exit_action_share": held_exit_action_share,
            "held_add_action_share": held_add_action_share,
            "flat_entry_action_share": flat_entry_action_share,
            "held_sell_pressure": held_sell_pressure,
            "held_sell_attribution": held_sell_attribution,
            "held_exit_timing_pressure": held_exit_timing_pressure,
            "held_exit_hazard": held_exit_hazard,
            "budget_entry_candidate_count": int(budget_entry_candidate_count),
            "budget_entry_keep_count": int(budget_entry_keep_count),
            "budget_held_protected_count": int(budget_held_protected_count),
            "budget_translation_floor_guard_count": int(translation_floor_guarded.sum()),
            "budget_translation_cap_guard_count": int(translation_cap_guarded.sum()),
            "budget_sell_priority_guard_count": int(sell_priority_guarded.sum()),
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
            "semantic_delta_guard_count": int(semantic_delta_guarded.sum()),
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
