from __future__ import annotations

import hashlib
from typing import Any, Mapping

import numpy as np
import pandas as pd

from quant_data_platform.qdp_v3.constants import (
    EXPECTED_1M_BAR_ENDS,
    EXPECTED_5M_BAR_ENDS,
    QUALITY_PROVISIONAL,
    QUALITY_QUARANTINED,
    QUALITY_STRICT,
)
from quant_data_platform.qdp_v3.identity import normalize_symbol
from quant_data_platform.qdp_v3.intraday import _bar_end, is_complete_5m_day, normalize_provider_5m


ONE_MINUTE_COLUMNS = [
    "provider_symbol",
    "trade_date",
    "bar_end",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "amount",
    "source",
]

CANONICAL_ONE_MINUTE_COLUMNS = [
    "security_id",
    "trade_date",
    "bar_end",
    "symbol_on_date",
    "provider_symbol",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "amount",
    "source",
    "quality_tier",
    "quality_reason",
    "identity_mapping_status",
]


def _raw_timestamp(frame: pd.DataFrame) -> pd.Series:
    if "bar_time" in frame.columns and "trade_date" in frame.columns:
        dates = frame["trade_date"].astype(str).str.slice(0, 10)
        times = frame["bar_time"].astype(str)
        compact = times.str.replace(":", "", regex=False).str.replace(".", "", regex=False).str.replace("-", "", regex=False)
        normalized_time = compact.map(lambda value: f"{value[:2]}:{value[2:4]}:{value[4:6] if len(value) >= 6 else '00'}")
        return pd.to_datetime(dates + " " + normalized_time, errors="coerce")
    for column in ("trade_time", "datetime", "trade_datetime", "trade_date", "date", "time"):
        if column in frame.columns:
            return pd.to_datetime(frame[column], errors="coerce")
    return pd.Series(pd.NaT, index=frame.index, dtype="datetime64[ns]")


def normalize_provider_1m(
    frame: pd.DataFrame,
    *,
    provider_symbol: str,
    source: str,
    merge_0930_into_0931: bool,
) -> pd.DataFrame:
    """Normalize provider bars and apply the QDP 240-bar auction contract."""

    if frame is None or frame.empty:
        return pd.DataFrame(columns=ONE_MINUTE_COLUMNS)
    data = frame.copy()
    timestamps = _raw_timestamp(data)
    if timestamps.isna().any():
        raise ValueError(f"{source}_1m_timestamp_missing_or_invalid")
    data["trade_date"] = timestamps.dt.strftime("%Y-%m-%d")
    data["bar_end"] = timestamps.dt.strftime("%H:%M")
    symbol_column = "provider_symbol" if "provider_symbol" in data.columns else "ts_code" if "ts_code" in data.columns else "symbol" if "symbol" in data.columns else ""
    if symbol_column:
        data["provider_symbol"] = data[symbol_column].map(normalize_symbol)
    else:
        data["provider_symbol"] = normalize_symbol(provider_symbol)
    aliases = {"vol": "volume"}
    data = data.rename(columns={key: value for key, value in aliases.items() if key in data.columns and value not in data.columns})
    for column in ("open", "high", "low", "close", "volume", "amount"):
        data[column] = pd.to_numeric(data[column] if column in data.columns else np.nan, errors="coerce")
    data["source"] = str(source)
    data = data.loc[:, ONE_MINUTE_COLUMNS]
    duplicate = data.duplicated(["provider_symbol", "trade_date", "bar_end"], keep=False)
    if duplicate.any():
        raise ValueError(f"{source}_1m_duplicate_bar_timestamp")
    if merge_0930_into_0931:
        days: list[pd.DataFrame] = []
        for _, day in data.groupby(["provider_symbol", "trade_date"], sort=True, dropna=False):
            days.append(_merge_opening_auction_pair(day))
        data = pd.concat(days, ignore_index=True) if days else data.iloc[0:0]
    return data.sort_values(["provider_symbol", "trade_date", "bar_end"]).reset_index(drop=True)


def _merge_opening_auction_pair(day: pd.DataFrame) -> pd.DataFrame:
    times = set(day["bar_end"].astype(str))
    if "09:30" not in times:
        return day.copy()
    if "09:31" not in times:
        invalid = day.copy()
        invalid["bar_end"] = invalid["bar_end"].astype(str)
        return invalid
    opening = day.loc[day["bar_end"].eq("09:30")].iloc[0]
    first = day.loc[day["bar_end"].eq("09:31")].iloc[0].copy()
    first["open"] = opening["open"] if pd.notna(opening["open"]) and float(opening["open"]) > 0 else first["open"]
    first["high"] = np.nanmax([opening["high"], first["high"]])
    first["low"] = np.nanmin([opening["low"], first["low"]])
    first["close"] = first["close"]
    first["volume"] = float(pd.to_numeric(opening["volume"], errors="coerce") or 0.0) + float(
        pd.to_numeric(first["volume"], errors="coerce") or 0.0
    )
    first["amount"] = float(pd.to_numeric(opening["amount"], errors="coerce") or 0.0) + float(
        pd.to_numeric(first["amount"], errors="coerce") or 0.0
    )
    kept = day.loc[~day["bar_end"].isin(["09:30", "09:31"])].copy()
    return pd.concat([pd.DataFrame([first]), kept], ignore_index=True, sort=False)


def one_minute_day_errors(frame: pd.DataFrame) -> list[str]:
    errors: list[str] = []
    if frame is None:
        return ["frame_missing"]
    if len(frame) != 240:
        errors.append(f"bar_count:{len(frame)}")
    if "bar_end" not in frame.columns:
        return [*errors, "bar_end_missing"]
    times = frame["bar_end"].astype(str)
    if set(times) != set(EXPECTED_1M_BAR_ENDS):
        errors.append("bar_time_set_mismatch")
    if times.duplicated().any():
        errors.append("duplicate_bar_end")
    required = ["open", "high", "low", "close", "volume", "amount"]
    if not set(required).issubset(frame.columns):
        errors.append("numeric_columns_missing")
        return errors
    numeric = frame[required].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(numeric.to_numpy()).all():
        errors.append("non_finite_numeric")
    if numeric[["open", "high", "low", "close"]].le(0).any().any():
        errors.append("non_positive_price")
    if numeric[["volume", "amount"]].lt(0).any().any():
        errors.append("negative_flow")
    if (numeric["high"] < numeric[["open", "low", "close"]].max(axis=1)).any():
        errors.append("high_relation_invalid")
    if (numeric["low"] > numeric[["open", "high", "close"]].min(axis=1)).any():
        errors.append("low_relation_invalid")
    return errors


def is_complete_1m_day(frame: pd.DataFrame) -> bool:
    return not one_minute_day_errors(frame)


def aggregate_1m_to_5m(frame: pd.DataFrame) -> pd.DataFrame:
    if not is_complete_1m_day(frame):
        raise ValueError(f"one_minute_day_not_complete:{one_minute_day_errors(frame)}")
    day = frame.copy().sort_values("bar_end").reset_index(drop=True)
    expected_position = {bar_end: index for index, bar_end in enumerate(EXPECTED_1M_BAR_ENDS)}
    day["_position"] = day["bar_end"].map(expected_position)
    if day["_position"].isna().any():
        raise ValueError("one_minute_unexpected_bar_end")
    day["_bucket"] = day["_position"].astype(int) // 5
    rows: list[dict[str, Any]] = []
    for bucket, group in day.groupby("_bucket", sort=True):
        first = group.iloc[0]
        last = group.iloc[-1]
        rows.append(
            {
                "provider_symbol": str(first.get("provider_symbol", "")),
                "trade_date": str(first["trade_date"]),
                "bar_end": EXPECTED_5M_BAR_ENDS[int(bucket)],
                "open": float(first["open"]),
                "high": float(pd.to_numeric(group["high"], errors="coerce").max()),
                "low": float(pd.to_numeric(group["low"], errors="coerce").min()),
                "close": float(last["close"]),
                "volume": float(pd.to_numeric(group["volume"], errors="coerce").sum()),
                "amount": float(pd.to_numeric(group["amount"], errors="coerce").sum()),
                "source": f"{first.get('source', '')}_1m_aggregated",
            }
        )
    result = pd.DataFrame(rows)
    if not is_complete_5m_day(result):
        raise RuntimeError("aggregated_5m_contract_invalid")
    return result


def compare_aggregated_with_direct_5m(
    aggregated: pd.DataFrame,
    direct: pd.DataFrame,
    *,
    price_atol: float = 1e-10,
    flow_atol: float = 1e-6,
) -> dict[str, Any]:
    if not is_complete_5m_day(aggregated) or not is_complete_5m_day(direct):
        return {"comparable": False, "conflict": True, "reason": "one_or_both_5m_days_incomplete"}
    left = aggregated.sort_values("bar_end").set_index("bar_end")
    right = direct.sort_values("bar_end").set_index("bar_end")
    mismatches: dict[str, int] = {}
    max_abs: dict[str, float] = {}
    for column in ("open", "high", "low", "close", "volume", "amount"):
        lvalues = pd.to_numeric(left[column], errors="coerce").to_numpy(dtype=float)
        rvalues = pd.to_numeric(right[column], errors="coerce").to_numpy(dtype=float)
        atol = flow_atol if column in {"volume", "amount"} else price_atol
        equal = np.isclose(lvalues, rvalues, rtol=1e-12, atol=atol, equal_nan=False)
        mismatches[column] = int((~equal).sum())
        max_abs[column] = float(np.max(np.abs(lvalues - rvalues)))
    conflict = any(value for value in mismatches.values())
    return {
        "comparable": True,
        "conflict": bool(conflict),
        "mismatch_counts": mismatches,
        "max_absolute_difference": max_abs,
    }


def reconcile_1m_with_daily(day: pd.DataFrame, daily: Mapping[str, Any] | pd.Series) -> dict[str, Any]:
    if not is_complete_1m_day(day):
        return {"comparable": False, "conflict": True, "reason": "1m_day_incomplete"}
    ordered = day.sort_values("bar_end")
    minute = {
        "open": float(ordered.iloc[0]["open"]),
        "high": float(pd.to_numeric(ordered["high"], errors="coerce").max()),
        "low": float(pd.to_numeric(ordered["low"], errors="coerce").min()),
        "close": float(ordered.iloc[-1]["close"]),
        "volume": float(pd.to_numeric(ordered["volume"], errors="coerce").sum()),
        "amount": float(pd.to_numeric(ordered["amount"], errors="coerce").sum()),
    }
    reference = {key: float(pd.to_numeric(daily.get(key), errors="coerce")) for key in minute}
    if not all(np.isfinite(value) for value in reference.values()):
        return {"comparable": False, "conflict": False, "reason": "daily_reference_incomplete"}
    price_diff = {key: abs(minute[key] - reference[key]) for key in ("open", "high", "low", "close")}
    volume_diff = abs(minute["volume"] - reference["volume"])
    amount_diff = abs(minute["amount"] - reference["amount"])
    volume_relative = volume_diff / max(abs(reference["volume"]), 1.0)
    amount_relative = amount_diff / max(abs(reference["amount"]), 1.0)
    conflict = (
        price_diff["open"] > 1e-8
        or price_diff["close"] > 1e-8
        or price_diff["high"] > 0.010000001
        or price_diff["low"] > 0.010000001
        or volume_diff > max(100.0, 1e-5 * abs(reference["volume"]))
        or amount_diff > max(1_000.0, 0.001 * abs(reference["amount"]))
    )
    return {
        "comparable": True,
        "conflict": bool(conflict),
        "minute_aggregate": minute,
        "daily_reference": reference,
        "price_absolute_difference": price_diff,
        "volume_absolute_difference": volume_diff,
        "volume_relative_difference": volume_relative,
        "amount_absolute_difference": amount_diff,
        "amount_relative_difference": amount_relative,
    }


def classify_1m_stock_day(
    day: pd.DataFrame,
    *,
    daily_reference: Mapping[str, Any] | pd.Series | None,
    direct_5m: pd.DataFrame | None,
    identity_mapped: bool,
) -> tuple[str, str, dict[str, Any]]:
    structure_errors = one_minute_day_errors(day)
    evidence: dict[str, Any] = {"structure_errors": structure_errors}
    if structure_errors or not identity_mapped:
        reason = "structure_invalid" if structure_errors else "identity_unmapped_or_conflicted"
        return QUALITY_QUARANTINED, reason, evidence
    aggregated = aggregate_1m_to_5m(day)
    if daily_reference is None:
        daily_result = {"comparable": False, "conflict": False, "reason": "daily_reference_missing"}
    else:
        daily_result = reconcile_1m_with_daily(day, daily_reference)
    evidence["daily"] = daily_result
    if daily_result.get("conflict"):
        return QUALITY_QUARANTINED, "daily_reconciliation_conflict", evidence
    if direct_5m is None or direct_5m.empty:
        five_result = {"comparable": False, "conflict": False, "reason": "direct_5m_missing"}
    else:
        five_result = compare_aggregated_with_direct_5m(aggregated, direct_5m)
    evidence["direct_5m"] = five_result
    if five_result.get("conflict"):
        return QUALITY_QUARANTINED, "direct_1m_5m_conflict", evidence
    if not daily_result.get("comparable") or not five_result.get("comparable"):
        return QUALITY_PROVISIONAL, "single_source_or_incomplete_cross_check", evidence
    return QUALITY_STRICT, "structure_daily_and_direct_5m_consistent", evidence


def canonicalize_1m_day(
    day: pd.DataFrame,
    *,
    identity_registry: Any,
    quality_tier: str,
    quality_reason: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if day is None or day.empty:
        return pd.DataFrame(columns=CANONICAL_ONE_MINUTE_COLUMNS), pd.DataFrame()
    mapped = identity_registry.map_frame(
        day,
        provider_symbol_column="provider_symbol",
        date_column="trade_date",
    )
    mapped["quality_tier"] = str(quality_tier)
    mapped["quality_reason"] = str(quality_reason)
    invalid = mapped["identity_mapping_status"].ne("mapped")
    quarantine = mapped.loc[invalid].copy()
    canonical = mapped.loc[~invalid, CANONICAL_ONE_MINUTE_COLUMNS].copy()
    duplicate = canonical.duplicated(["security_id", "trade_date", "bar_end"], keep=False)
    if duplicate.any():
        duplicate_rows = canonical.loc[duplicate].copy()
        duplicate_rows["conflict_type"] = "identity_mapped_1m_duplicate"
        quarantine = pd.concat([quarantine, duplicate_rows], ignore_index=True, sort=False)
        canonical = canonical.loc[~duplicate].copy()
    return canonical.sort_values(["security_id", "trade_date", "bar_end"]).reset_index(drop=True), quarantine.reset_index(drop=True)


def stable_security_bucket(security_id: str, *, bucket_count: int = 64) -> int:
    digest = hashlib.sha256(str(security_id).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % int(bucket_count)


def normalize_direct_proxy_5m(frame: pd.DataFrame, *, provider_symbol: str) -> pd.DataFrame:
    data = frame.copy()
    timestamps = _raw_timestamp(data)
    if timestamps.isna().any():
        raise ValueError("tushare_proxy_5m_timestamp_missing_or_invalid")
    data["trade_date"] = timestamps.dt.strftime("%Y-%m-%d")
    data["bar_time"] = timestamps.dt.strftime("%H:%M")
    if "vol" in data.columns and "volume" not in data.columns:
        data = data.rename(columns={"vol": "volume"})
    return normalize_provider_5m(data, provider_symbol=provider_symbol, source="tushare_proxy")
