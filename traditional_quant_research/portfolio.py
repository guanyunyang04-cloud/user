"""Portfolio construction baselines for traditional quant experiments."""

from __future__ import annotations

from typing import Mapping


def equal_weight(symbols: list[str]) -> dict[str, float]:
    """Return equal long-only weights for a non-empty symbol list."""
    if not symbols:
        return {}
    unique_symbols = list(dict.fromkeys(symbols))
    weight = 1.0 / len(unique_symbols)
    return {symbol: weight for symbol in unique_symbols}


def normalize_long_only(scores: Mapping[str, float]) -> dict[str, float]:
    """Normalize non-negative scores into long-only portfolio weights."""
    positive = {symbol: max(float(score), 0.0) for symbol, score in scores.items()}
    total = sum(positive.values())
    if total == 0:
        return equal_weight(list(positive))
    return {symbol: value / total for symbol, value in positive.items()}


def rank_long_short(scores: Mapping[str, float], leg_size: int) -> dict[str, float]:
    """Build a dollar-neutral rank portfolio from top and bottom score legs."""
    if leg_size <= 0:
        raise ValueError("leg_size must be positive")
    if len(scores) < leg_size * 2:
        raise ValueError("need at least two full legs of scored symbols")
    ranked = sorted(scores.items(), key=lambda item: (float(item[1]), item[0]))
    shorts = ranked[:leg_size]
    longs = ranked[-leg_size:]
    long_weight = 0.5 / leg_size
    short_weight = -0.5 / leg_size
    weights = {symbol: short_weight for symbol, _ in shorts}
    weights.update({symbol: long_weight for symbol, _ in longs})
    return weights
