"""Recent Market Repair: validation responsibilities."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from quantlab.data.qdp_v2.manifest import (
    EXPECTED_BAR_TIMES,
)

from .config import (
    AMOUNT_UNIT_SCALES,
    DAILY_AMOUNT_RELATIVE_TOLERANCE,
    DAILY_COLUMNS,
    DAILY_PRICE_RELATIVE_TOLERANCE,
    DAILY_VOLUME_RELATIVE_TOLERANCE,
    INTRADAY_COLUMNS,
    NUMERIC_COLUMNS,
    VOLUME_UNIT_SCALES,
)


def _best_scale(observed: pd.Series, reference: pd.Series, candidates: Sequence[float]) -> float:
    left = pd.to_numeric(observed, errors="coerce").to_numpy(dtype="float64")
    right = pd.to_numeric(reference, errors="coerce").to_numpy(dtype="float64")
    valid = np.isfinite(left) & np.isfinite(right) & (left > 0) & (right > 0)
    if not valid.any():
        return 1.0
    scores = {
        float(scale): float(
            np.median(np.abs(np.log(np.maximum(left[valid] * float(scale), 1e-12) / np.maximum(right[valid], 1e-12))))
        )
        for scale in candidates
    }
    return min(scores, key=lambda item: (scores[item], abs(np.log(item))))


def _intraday_daily_aggregate(frame: pd.DataFrame, reference: pd.DataFrame) -> pd.DataFrame:
    aggregate = frame.groupby(["symbol", "trade_date"], as_index=False).agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        volume=("volume", "sum"),
        amount=("amount", "sum"),
    )
    ref = reference.copy()
    ref["symbol"] = ref["symbol"].astype(str).str.upper()
    ref["trade_date"] = ref["trade_date"].astype(str).str.slice(0, 10)
    return aggregate.merge(
        ref,
        on=["symbol", "trade_date"],
        how="left",
        suffixes=("_five", "_day"),
        validate="one_to_one",
    )


def _intraday_daily_errors(
    aggregate: pd.DataFrame,
    *,
    volume_scale: float,
    amount_scale: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.Series]:
    price_rel = np.full(len(aggregate), np.inf, dtype="float64")
    volume_rel = np.full(len(aggregate), np.inf, dtype="float64")
    amount_rel = np.full(len(aggregate), np.inf, dtype="float64")
    present = aggregate["open_day"].notna().to_numpy(dtype=bool)
    if present.any():
        price_rel[present] = np.column_stack(
            [
                (
                    aggregate.loc[present, f"{column}_five"].to_numpy(dtype="float64")
                    - aggregate.loc[present, f"{column}_day"].to_numpy(dtype="float64")
                ).__abs__()
                / aggregate.loc[present, f"{column}_day"].abs().clip(lower=1.0).to_numpy(dtype="float64")
                for column in ("open", "high", "low", "close")
            ]
        ).max(axis=1)
        volume_rel[present] = (aggregate.loc[present, "volume_five"] * volume_scale).sub(
            aggregate.loc[present, "volume_day"]
        ).abs().to_numpy(dtype="float64") / aggregate.loc[present, "volume_day"].abs().clip(
            lower=1.0
        ).to_numpy(dtype="float64")
        amount_rel[present] = (aggregate.loc[present, "amount_five"] * amount_scale).sub(
            aggregate.loc[present, "amount_day"]
        ).abs().to_numpy(dtype="float64") / aggregate.loc[present, "amount_day"].abs().clip(
            lower=1.0
        ).to_numpy(dtype="float64")
    volume_ratio = (aggregate["volume_five"] * volume_scale) / aggregate["volume_day"].replace(0.0, np.nan)
    volume_100x = volume_ratio.between(95.0, 105.0) | volume_ratio.between(0.0095, 0.0105)
    return price_rel, volume_rel, amount_rel, present, volume_100x


def _intraday_daily_rejections(
    aggregate: pd.DataFrame,
    *,
    accepted: np.ndarray,
    reference_present: np.ndarray,
    price_rel: np.ndarray,
    volume_100x: pd.Series,
) -> tuple[dict[tuple[str, str], str], dict[str, int]]:
    unresolved: dict[tuple[str, str], str] = {}
    counts: dict[str, int] = {}
    for index, row in aggregate.iterrows():
        if accepted[index]:
            continue
        if not reference_present[index]:
            reason = "missing_daily_reference"
        elif bool(volume_100x.iloc[index]):
            reason = "volume_100x_anomaly"
        elif price_rel[index] > DAILY_PRICE_RELATIVE_TOLERANCE:
            reason = "daily_price_mismatch"
        else:
            reason = "daily_volume_and_amount_mismatch"
        unresolved[(str(row["symbol"]), str(row["trade_date"]))] = reason
        counts[reason] = counts.get(reason, 0) + 1
    return unresolved, counts


def _scaled_accepted_intraday(
    frame: pd.DataFrame,
    aggregate: pd.DataFrame,
    *,
    accepted: np.ndarray,
    volume_scale: float,
    amount_scale: float,
    source_name: str,
) -> pd.DataFrame:
    accepted_pairs = pd.MultiIndex.from_frame(aggregate.loc[accepted, ["symbol", "trade_date"]])
    frame_pairs = pd.MultiIndex.from_frame(frame[["symbol", "trade_date"]])
    output = frame.loc[frame_pairs.isin(accepted_pairs)].copy()
    output["volume"] = pd.to_numeric(output["volume"], errors="coerce") * volume_scale
    output["amount"] = pd.to_numeric(output["amount"], errors="coerce") * amount_scale
    output["source"] = str(source_name)
    return output.reset_index(drop=True)


def _validate_intraday_against_daily(
    frame: pd.DataFrame,
    reference: pd.DataFrame,
    *,
    source_name: str = "validated_intraday",
) -> tuple[pd.DataFrame, dict[tuple[str, str], str], dict[str, Any]]:
    """Validate complete intraday days against one bucket-wide daily scale.

    A provider's volume and amount units are selected once from the whole
    bucket.  This prevents a per-symbol scale choice from silently accepting
    incompatible provider units.
    """
    if frame is None or frame.empty:
        return (
            pd.DataFrame(columns=INTRADAY_COLUMNS),
            {},
            {"rejection_counts": {}, "accepted_day_count": 0},
        )
    aggregate = _intraday_daily_aggregate(frame, reference)
    volume_scale = _best_scale(aggregate["volume_five"], aggregate["volume_day"], VOLUME_UNIT_SCALES)
    amount_scale = _best_scale(aggregate["amount_five"], aggregate["amount_day"], AMOUNT_UNIT_SCALES)
    price_rel, volume_rel, amount_rel, reference_present, volume_100x = _intraday_daily_errors(
        aggregate,
        volume_scale=volume_scale,
        amount_scale=amount_scale,
    )
    accepted = (
        reference_present
        & (price_rel <= DAILY_PRICE_RELATIVE_TOLERANCE)
        & ((volume_rel <= DAILY_VOLUME_RELATIVE_TOLERANCE) | (amount_rel <= DAILY_AMOUNT_RELATIVE_TOLERANCE))
        & ~volume_100x.fillna(False).to_numpy(dtype=bool)
    )
    unresolved, rejection_counts = _intraday_daily_rejections(
        aggregate,
        accepted=accepted,
        reference_present=reference_present,
        price_rel=price_rel,
        volume_100x=volume_100x,
    )
    output = _scaled_accepted_intraday(
        frame,
        aggregate,
        accepted=accepted,
        volume_scale=volume_scale,
        amount_scale=amount_scale,
        source_name=source_name,
    )
    return (
        output,
        unresolved,
        {
            "accepted_day_count": int(accepted.sum()),
            "rejection_counts": rejection_counts,
            "volume_scale": float(volume_scale),
            "amount_scale": float(amount_scale),
            "maximum_accepted_price_relative_error": float(price_rel[accepted].max()) if accepted.any() else None,
            "maximum_accepted_volume_relative_error": float(volume_rel[accepted].max()) if accepted.any() else None,
            "maximum_accepted_amount_relative_error": float(amount_rel[accepted].max()) if accepted.any() else None,
        },
    )


def _validate_daily_frame(
    frame: pd.DataFrame,
    wanted: Mapping[str, Sequence[str]],
) -> tuple[pd.DataFrame, dict[tuple[str, str], str]]:
    wanted_pairs = {(symbol, str(date)) for symbol, dates in wanted.items() for date in dates}
    if frame is None or frame.empty:
        return pd.DataFrame(columns=DAILY_COLUMNS), {item: "missing_provider_day" for item in wanted_pairs}
    data = frame.copy()
    data["symbol"] = data["symbol"].astype(str).str.upper()
    data["trade_date"] = data["trade_date"].astype(str).str.slice(0, 10)
    pair_index = pd.MultiIndex.from_frame(data[["symbol", "trade_date"]])
    data = data.loc[pair_index.isin(pd.MultiIndex.from_tuples(sorted(wanted_pairs)))].copy()
    if data.empty:
        return pd.DataFrame(columns=DAILY_COLUMNS), {item: "missing_provider_day" for item in wanted_pairs}
    numeric, row_valid = _numeric_validity(data)
    data.loc[:, list(NUMERIC_COLUMNS)] = numeric
    data["_row_valid"] = row_valid
    stats = data.groupby(["symbol", "trade_date"], sort=False).agg(
        rows=("symbol", "size"),
        valid=("_row_valid", "all"),
        volume=("volume", "sum"),
        amount=("amount", "sum"),
    )
    valid_pairs = {
        (str(symbol), str(trade_date))
        for (symbol, trade_date), row in stats.iterrows()
        if int(row["rows"]) == 1 and bool(row["valid"]) and (float(row["volume"]) != 0.0 or float(row["amount"]) != 0.0)
    }
    selected_index = pd.MultiIndex.from_frame(data[["symbol", "trade_date"]])
    valid_index = (
        pd.MultiIndex.from_tuples(sorted(valid_pairs), names=("symbol", "trade_date"))
        if valid_pairs
        else selected_index[:0]
    )
    out = data.loc[selected_index.isin(valid_index)].copy()
    out["source"] = "mootdx_online"
    out["adjusted_flag"] = "none"
    out = out.loc[:, DAILY_COLUMNS].reset_index(drop=True)
    unresolved = {pair: "missing_or_invalid_provider_day" for pair in wanted_pairs - valid_pairs}
    return out, unresolved


def _validate_intraday_frame(
    frame: pd.DataFrame,
    wanted: Mapping[str, Sequence[str]],
    *,
    source_name: str = "mootdx_online",
    detailed_reasons: bool = False,
) -> tuple[pd.DataFrame, dict[tuple[str, str], str]]:
    wanted_pairs = {(symbol, str(date)) for symbol, dates in wanted.items() for date in dates}
    if frame is None or frame.empty:
        return pd.DataFrame(columns=INTRADAY_COLUMNS), {
            item: ("provider_empty" if detailed_reasons else "missing_provider_day") for item in wanted_pairs
        }
    data = frame.copy()
    data["symbol"] = data["symbol"].astype(str).str.upper()
    data["trade_date"] = data["trade_date"].astype(str).str.slice(0, 10)
    data["bar_time"] = data["bar_time"].astype(str).str.replace(":", "", regex=False).str.slice(0, 4) + "00000"
    pair_index = pd.MultiIndex.from_frame(data[["symbol", "trade_date"]])
    data = data.loc[pair_index.isin(pd.MultiIndex.from_tuples(sorted(wanted_pairs)))].copy()
    if data.empty:
        return pd.DataFrame(columns=INTRADAY_COLUMNS), {
            item: ("provider_empty" if detailed_reasons else "missing_provider_day") for item in wanted_pairs
        }
    numeric, row_valid = _numeric_validity(data)
    data.loc[:, list(NUMERIC_COLUMNS)] = numeric
    data["_row_valid"] = row_valid
    data["_expected_time"] = data["bar_time"].isin(EXPECTED_BAR_TIMES)
    stats = data.groupby(["symbol", "trade_date"], sort=False).agg(
        rows=("bar_time", "size"),
        times=("bar_time", "nunique"),
        expected=("_expected_time", "all"),
        valid=("_row_valid", "all"),
        volume=("volume", "sum"),
        amount=("amount", "sum"),
    )
    valid_pairs = {
        (str(symbol), str(trade_date))
        for (symbol, trade_date), row in stats.iterrows()
        if int(row["rows"]) == 48
        and int(row["times"]) == 48
        and bool(row["expected"])
        and bool(row["valid"])
        and (float(row["volume"]) != 0.0 or float(row["amount"]) != 0.0)
    }
    selected_index = pd.MultiIndex.from_frame(data[["symbol", "trade_date"]])
    valid_index = (
        pd.MultiIndex.from_tuples(sorted(valid_pairs), names=("symbol", "trade_date"))
        if valid_pairs
        else selected_index[:0]
    )
    out = data.loc[selected_index.isin(valid_index)].copy()
    out["source"] = str(source_name)
    out["adjusted_flag"] = "none"
    out = out.loc[:, INTRADAY_COLUMNS].reset_index(drop=True)
    unresolved: dict[tuple[str, str], str] = {}
    if detailed_reasons:
        for pair in sorted(wanted_pairs - valid_pairs):
            pair_rows = data.loc[(data["symbol"] == pair[0]) & (data["trade_date"] == pair[1])]
            if pair_rows.empty:
                reason = "provider_empty"
            elif len(pair_rows) != 48:
                reason = "incomplete_48_bars"
            elif pair_rows["bar_time"].nunique() != 48 or not bool(pair_rows["_expected_time"].all()):
                reason = "unexpected_bar_time"
            elif not bool(pair_rows["_row_valid"].all()) or (
                float(pair_rows["volume"].sum()) == 0.0 and float(pair_rows["amount"].sum()) == 0.0
            ):
                reason = "invalid_numeric_or_ohlc"
            else:
                reason = "incomplete_48_bars"
            unresolved[pair] = reason
    else:
        unresolved = {pair: "missing_or_invalid_complete_48_bar_day" for pair in wanted_pairs - valid_pairs}
    return out, unresolved


def _numeric_validity(data: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    numeric = (
        data.loc[:, list(NUMERIC_COLUMNS)]
        .apply(pd.to_numeric, errors="coerce")
        .to_numpy(dtype="float64", na_value=np.nan, copy=True)
    )
    finite = np.isfinite(numeric).all(axis=1)
    positive_prices = (numeric[:, :4] > 0).all(axis=1)
    nonnegative_turnover = (numeric[:, 4:6] >= 0).all(axis=1)
    valid_ohlc = (numeric[:, 1] >= numeric[:, [0, 2, 3]].max(axis=1)) & (
        numeric[:, 2] <= numeric[:, [0, 1, 3]].min(axis=1)
    )
    return numeric, finite & positive_prices & nonnegative_turnover & valid_ohlc
