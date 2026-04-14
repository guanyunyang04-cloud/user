from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_MAX_POSITIONS = 8
DEFAULT_MAX_POSITION_WEIGHT = 0.20
DEFAULT_TURNOVER_LIMIT = 0.60


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
    last_action_labels: dict[str, str] = field(default_factory=dict)
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
            "last_action_labels": {stock: str(value or "") for stock, value in sorted(self.last_action_labels.items())},
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
            last_action_labels={str(stock): str(value or "") for stock, value in (data.get("last_action_labels", {}) or {}).items()},
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

    def step(
        self,
        *,
        date: pd.Timestamp | str,
        prices: pd.Series,
        policy_frame: pd.DataFrame,
        global_targets: dict[str, Any] | None = None,
        source_label: str = "model",
    ) -> StepResult:
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
            }
        )

        current = self.current_weights(prices.index)
        gross_exposure_target = float((global_targets or {}).get("gross_exposure_target", max(0.20, min(0.95, 1.0 - self.cash_weight))))
        candidate_budget = int((global_targets or {}).get("candidate_budget", self.max_positions) or self.max_positions)
        candidate_budget = max(1, min(candidate_budget, int(self.max_positions)))
        turnover_budget = float((global_targets or {}).get("turnover_budget", self.turnover_limit) or self.turnover_limit)
        position_cap_target = float((global_targets or {}).get("max_position_weight_target", self.max_position_weight) or self.max_position_weight)
        position_cap_target = float(np.clip(position_cap_target, 0.05, 0.35))
        hold_bias_target = float((global_targets or {}).get("hold_bias_target", 0.25) or 0.25)
        hold_bias_target = float(np.clip(hold_bias_target, 0.05, 0.98))

        desired_strength = pd.Series(0.0, index=prices.index, dtype=float)
        forced_zero = pd.Series(False, index=prices.index, dtype=bool)
        for stock in prices.index:
            action = str(policy.at[stock, "action_label"] or "skip").strip().lower()
            strength = float(policy.at[stock, "action_strength"] or 0.0)
            delta_hint = float(policy.at[stock, "target_delta_hint"] or 0.0)
            hold_boost = float(policy.at[stock, "hold_boost"] or 0.0)
            entry_quality = float(policy.at[stock, "entry_quality"] or 0.0) if "entry_quality" in policy.columns else 0.0
            hold_quality = float(policy.at[stock, "hold_quality"] or 0.0) if "hold_quality" in policy.columns else 0.0
            add_quality = float(policy.at[stock, "add_quality"] or 0.0) if "add_quality" in policy.columns else 0.0
            reduce_quality = float(policy.at[stock, "reduce_quality"] or 0.0) if "reduce_quality" in policy.columns else 0.0
            planned_holding_days = float(policy.at[stock, "planned_holding_days"] or 0.0) if "planned_holding_days" in policy.columns else 0.0
            current_weight = float(current.get(stock, 0.0))
            if action == "exit":
                forced_zero.at[stock] = True
                continue
            if action == "reduce":
                reduction_scale = max(0.12, 1.0 + delta_hint - reduce_quality * 0.20 + hold_bias_target * 0.08)
                desired_strength.at[stock] = max(current_weight * reduction_scale, 0.0)
                continue
            if action == "hold":
                hold_scale = 1.0 + hold_bias_target * 0.06 + max(planned_holding_days - 3.0, 0.0) / 120.0
                desired_strength.at[stock] = max(
                    current_weight * hold_scale,
                    current_weight + max(0.0, hold_boost + hold_quality) * (0.025 + hold_bias_target * 0.030),
                )
                continue
            if action == "add":
                desired_strength.at[stock] = max(
                    current_weight + max(0.015, strength * 0.55 + add_quality * 0.15 + planned_holding_days / 300.0),
                    current_weight,
                )
                continue
            if action == "open":
                desired_strength.at[stock] = max(
                    strength * (0.85 + hold_bias_target * 0.15),
                    max(delta_hint, 0.02 + entry_quality * 0.20 + planned_holding_days / 320.0),
                )
                continue
            desired_strength.at[stock] = current_weight * (1.0 + hold_bias_target * 0.02)

        if candidate_budget < len(desired_strength):
            keep = desired_strength.nlargest(candidate_budget).index
            desired_strength = desired_strength.where(desired_strength.index.isin(keep), 0.0)
        desired_strength = desired_strength.where(~forced_zero, 0.0)

        target_weights = self._allocate_with_cap(
            desired_strength,
            gross_exposure_target,
            position_cap=position_cap_target,
        )
        delta = target_weights - current
        raw_turnover = float(delta.abs().sum())
        if raw_turnover > turnover_budget > 0:
            delta = delta * (turnover_budget / raw_turnover)
        new_weights = (current + delta).clip(lower=0.0)
        if float(new_weights.sum()) > 0.999:
            new_weights = new_weights / float(new_weights.sum())
        self.cash_weight = max(0.0, 1.0 - float(new_weights.sum()))

        actions: list[dict[str, Any]] = []
        for stock in prices.index:
            previous_weight = float(current.get(stock, 0.0))
            new_weight = float(new_weights.get(stock, 0.0))
            model_action = str(policy.at[stock, "action_label"] or "skip")
            existing = self.holdings.get(stock)
            previous_hold_days = int(existing.hold_days) if existing is not None else 0
            previous_entry_price = float(existing.entry_price) if existing is not None else 0.0
            previous_peak_price = float(existing.peak_price) if existing is not None else 0.0
            current_price = float(prices.get(stock, np.nan))
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
            if previous_weight <= 1e-8 and new_weight > 1e-8:
                execution_action = "open"
            elif previous_weight > 1e-8 and new_weight <= 1e-8:
                execution_action = "exit"
            elif new_weight > previous_weight + 1e-8:
                execution_action = "add"
            elif new_weight < previous_weight - 1e-8:
                execution_action = "reduce"
            else:
                execution_action = "hold"

            actions.append(
                {
                    "date": signal_dt.strftime("%Y-%m-%d"),
                    "stock": stock,
                    "source_label": source_label,
                    "model_action": model_action,
                    "execution_action": execution_action,
                    "current_weight": previous_weight,
                    "target_weight": new_weight,
                    "delta_weight": new_weight - previous_weight,
                    "current_price": current_price,
                    "entry_price_before": previous_entry_price,
                    "peak_price_before": previous_peak_price,
                    "hold_days_before": previous_hold_days,
                    "hold_days_after": previous_hold_days + 1 if new_weight > 1e-8 else previous_hold_days,
                    "unrealized_pnl_before": unrealized_pnl_before,
                    "drawdown_from_peak_before": drawdown_from_peak_before,
                    "action_strength": float(policy.at[stock, "action_strength"] or 0.0),
                    "target_delta_hint": float(policy.at[stock, "target_delta_hint"] or 0.0),
                    "exit_urgency": float(policy.at[stock, "exit_urgency"] or 0.0),
                }
            )

        next_holdings: dict[str, HoldingState] = {}
        next_last_buy_dates = dict(self.last_buy_dates)
        next_last_sell_dates = dict(self.last_sell_dates)
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
            if not stock or execution_action not in {"open", "add", "reduce", "exit", "hold"}:
                continue
            next_last_action_labels[stock] = execution_action
            if execution_action in {"open", "add"}:
                next_last_buy_dates[stock] = signal_date_text
            if execution_action in {"reduce", "exit"}:
                next_last_sell_dates[stock] = signal_date_text

        self.holdings = next_holdings
        self.last_buy_dates = next_last_buy_dates
        self.last_sell_dates = next_last_sell_dates
        self.last_action_labels = next_last_action_labels
        weight_delta = new_weights - current
        realized_turnover = float(weight_delta.abs().sum())
        buy_turnover = float(weight_delta.clip(lower=0.0).sum())
        sell_turnover = float((-weight_delta.clip(upper=0.0)).sum())
        self.recent_turnovers.append(realized_turnover)
        if len(self.recent_turnovers) > 60:
            self.recent_turnovers = self.recent_turnovers[-60:]
        self.recent_cash_weights.append(float(self.cash_weight))
        if len(self.recent_cash_weights) > 60:
            self.recent_cash_weights = self.recent_cash_weights[-60:]
        self.last_signal_date = signal_date_text
        diagnostics = {
            "gross_exposure_target": gross_exposure_target,
            "candidate_budget": candidate_budget,
            "turnover_budget": turnover_budget,
            "max_position_weight_target": position_cap_target,
            "hold_bias_target": hold_bias_target,
            "raw_turnover": raw_turnover,
            "realized_turnover": realized_turnover,
            "buy_turnover": buy_turnover,
            "sell_turnover": sell_turnover,
            "cash_weight": float(self.cash_weight),
            "holding_count": int(sum(1 for value in self.holdings.values() if value.weight > 1e-8)),
        }
        return StepResult(
            date=signal_dt.strftime("%Y-%m-%d"),
            weights=new_weights,
            actions=actions,
            diagnostics=diagnostics,
        )
