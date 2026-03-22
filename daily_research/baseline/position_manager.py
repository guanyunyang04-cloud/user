from __future__ import annotations

from typing import Dict, List, Tuple

import pandas as pd

from daily_research.baseline.config import ResearchConfig


def _safe_series(row: pd.Series | None) -> pd.Series:
    if row is None:
        return pd.Series(dtype=float)
    return row.dropna().astype(float)


class PositionManager:
    def __init__(self, config: ResearchConfig):
        self.config = config
        self.weights = pd.Series(dtype=float)
        self.entry_price: Dict[str, float] = {}
        self.hold_days: Dict[str, int] = {}

    def step(
        self,
        date,
        prices: pd.Series,
        target_weights: pd.Series,
        target_scores: pd.Series | None = None,
    ) -> Tuple[pd.Series, List[Dict], Dict[str, float]]:
        prices = _safe_series(prices)
        target_weights = _safe_series(target_weights)
        target_scores = _safe_series(target_scores)

        current = self.weights.reindex(prices.index).fillna(0.0)
        desired = target_weights.reindex(prices.index).fillna(0.0)
        score_snapshot = target_scores.reindex(prices.index).fillna(0.0)

        for stock in current.index:
            if current[stock] > 0:
                self.hold_days[stock] = self.hold_days.get(stock, 0) + 1
            else:
                self.hold_days.pop(stock, None)
                self.entry_price.pop(stock, None)

        forced_reasons: Dict[str, str] = {}
        for stock, current_weight in current.items():
            if current_weight <= 0 or stock not in prices.index:
                continue
            entry = self.entry_price.get(stock, float(prices[stock]))
            ret = float(prices[stock] / entry - 1.0)
            if ret <= float(self.config.stop_loss):
                desired[stock] = 0.0
                forced_reasons[stock] = "止损清仓"
            elif ret >= float(self.config.take_profit):
                desired[stock] = min(float(desired.get(stock, 0.0)), current_weight * 0.5)
                forced_reasons[stock] = "止盈减仓"

        for stock, current_weight in current.items():
            if current_weight <= 0:
                continue
            if self.hold_days.get(stock, 0) < int(self.config.min_hold_days):
                if forced_reasons.get(stock) != "止损清仓":
                    desired[stock] = max(float(desired.get(stock, 0.0)), current_weight)

        delta = desired - current
        raw_turnover = float(delta.abs().sum())
        turnover_scale = 1.0
        if raw_turnover > float(self.config.turnover_limit) and raw_turnover > 0:
            turnover_scale = float(self.config.turnover_limit) / raw_turnover
            delta = delta * turnover_scale

        new_weights = (current + delta).clip(lower=0.0)
        if new_weights.sum() > 1.0:
            new_weights = new_weights / new_weights.sum()

        actions: List[Dict] = []
        for stock in new_weights.index:
            current_weight = float(current.get(stock, 0.0))
            new_weight = float(new_weights.get(stock, 0.0))
            action = None
            reason = None

            if current_weight == 0.0 and new_weight > 0.0:
                action = "建仓"
                reason = "新进入目标持仓"
                self.entry_price[stock] = float(prices.get(stock, 0.0))
                self.hold_days[stock] = 0
            elif new_weight > current_weight + 1e-8:
                action = "加仓"
                reason = "目标仓位上调"
            elif new_weight < current_weight - 1e-8 and new_weight > 0.0:
                action = "减仓"
                reason = forced_reasons.get(stock, "目标仓位下调")
            elif new_weight == 0.0 and current_weight > 0.0:
                action = "清仓"
                reason = forced_reasons.get(stock, "调出持仓名单")
                self.entry_price.pop(stock, None)
                self.hold_days.pop(stock, None)

            if action:
                actions.append(
                    {
                        "date": date,
                        "stock": stock,
                        "action": action,
                        "from_weight": current_weight,
                        "to_weight": new_weight,
                        "reason": reason,
                        "score": float(score_snapshot.get(stock, 0.0)),
                    }
                )

        self.weights = new_weights
        diagnostics = {
            "turnover": float((new_weights - current).abs().sum()),
            "raw_turnover": raw_turnover,
            "turnover_scale": turnover_scale,
            "holding_count": int((new_weights > 0).sum()),
        }
        return new_weights, actions, diagnostics
