"""Historical Intraday External Archive Repair: normalize responsibilities."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

from quantlab.data.qdp_v2.manifest import (
    EXPECTED_BAR_TIMES,
)

from .config import (
    _NUMERIC_COLUMNS,
    _PRICE_COLUMNS,
    AMOUNT_RELATIVE_TOLERANCE,
    PRICE_RELATIVE_TOLERANCE,
    VOLUME_RELATIVE_TOLERANCE,
)


def _positive_finite(value: Any) -> bool:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return bool(np.isfinite(number) and number > 0)


def _normalize_day(day: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    ordered = day.sort_values("bar_time", kind="stable").reset_index(drop=True)
    actual = tuple(ordered["bar_time"].astype(str))
    expected_colon = tuple(f"{item[:2]}:{item[2:4]}" for item in EXPECTED_BAR_TIMES)
    if len(ordered) == 48 and actual == expected_colon:
        normalized = ordered.copy()
    elif len(ordered) == 49 and actual == ("09:30", *expected_colon):
        auction = ordered.iloc[0]
        first = ordered.iloc[1]
        normalized = ordered.iloc[1:].copy().reset_index(drop=True)
        normalized.loc[0, "open"] = (
            float(auction["open"]) if _positive_finite(auction["open"]) else float(first["open"])
        )
        highs = [float(value) for value in (auction["high"], first["high"]) if _positive_finite(value)]
        lows = [float(value) for value in (auction["low"], first["low"]) if _positive_finite(value)]
        if not highs or not lows:
            return pd.DataFrame(), "auction_price_invalid"
        normalized.loc[0, "high"] = max(highs)
        normalized.loc[0, "low"] = min(lows)
        normalized.loc[0, "close"] = float(first["close"])
        normalized.loc[0, "volume"] = float(auction["volume"]) + float(first["volume"])
        normalized.loc[0, "amount"] = float(auction["amount"]) + float(first["amount"])
    else:
        if ordered["bar_time"].duplicated().any():
            return pd.DataFrame(), "duplicate_bar_time"
        return pd.DataFrame(), "invalid_bar_time_set"
    numeric = normalized.loc[:, _NUMERIC_COLUMNS].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(numeric.to_numpy(dtype="float64")).all():
        return pd.DataFrame(), "non_finite_numeric"
    if numeric.loc[:, _PRICE_COLUMNS].le(0).any().any():
        return pd.DataFrame(), "non_positive_price"
    if numeric.loc[:, ["volume", "amount"]].lt(0).any().any():
        return pd.DataFrame(), "negative_volume_or_amount"
    if (numeric["high"] < numeric[["open", "low", "close"]].max(axis=1)).any():
        return pd.DataFrame(), "invalid_high_relation"
    if (numeric["low"] > numeric[["open", "high", "close"]].min(axis=1)).any():
        return pd.DataFrame(), "invalid_low_relation"
    normalized.loc[:, _NUMERIC_COLUMNS] = numeric
    normalized["bar_time"] = list(EXPECTED_BAR_TIMES)
    return normalized.reset_index(drop=True), ""


def _relative_error(observed: float, reference: float) -> float:
    if not np.isfinite(observed) or not np.isfinite(reference):
        return float("inf")
    return abs(float(observed) - float(reference)) / max(abs(float(reference)), 1e-12)


def _validate_daily(
    day: pd.DataFrame,
    reference: Mapping[str, Any],
) -> tuple[bool, str, float, float, float]:
    price_errors = [
        _relative_error(
            float(day[column].iloc[0] if column == "open" else day[column].iloc[-1])
            if column in {"open", "close"}
            else float(day[column].max() if column == "high" else day[column].min()),
            float(reference[column]),
        )
        for column in _PRICE_COLUMNS
    ]
    price_error = max(price_errors)
    volume_error = _relative_error(float(day["volume"].sum()), float(reference["volume"]))
    amount_error = _relative_error(float(day["amount"].sum()), float(reference["amount"]))
    if price_error > PRICE_RELATIVE_TOLERANCE:
        reason = "daily_price_mismatch"
    elif volume_error > VOLUME_RELATIVE_TOLERANCE and amount_error > AMOUNT_RELATIVE_TOLERANCE:
        reason = "daily_volume_and_amount_mismatch"
    else:
        reason = ""
    return not reason, reason, price_error, volume_error, amount_error
