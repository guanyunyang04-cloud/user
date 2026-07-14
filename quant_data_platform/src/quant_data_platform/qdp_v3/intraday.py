from __future__ import annotations

import hashlib
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd

from quant_data_platform.core.json_io import read_json
from quant_data_platform.domains.contracts import DataDomain, DomainFetchRequest
from quant_data_platform.providers import BAOSTOCK_BATCH_WHEEL_SHA256, BaostockProvider, MootdxOnlineProvider
from quant_data_platform.progress import progress_write
from quant_data_platform.qdp_v3.constants import (
    EXPECTED_5M_BAR_ENDS,
    MOOTDX_VERSION,
    QUALITY_PROVISIONAL,
    QUALITY_QUARANTINED,
    QUALITY_STRICT,
    RAW_ADJUST_FACTOR_EVENT,
    RAW_DAILY_ASTOCK,
    RAW_INTRADAY_5M_BAOSTOCK,
    RAW_INTRADAY_5M_MOOTDX,
    RAW_INTRADAY_5M_SELECTED,
    RAW_SECURITY_MASTER,
)
from quant_data_platform.qdp_v3.identity import normalize_symbol
from quant_data_platform.qdp_v3.storage import get_raw_partition, iter_raw_partitions, read_raw_partition, read_raw_receipt, write_raw_partition


SELECTED_COLUMNS = [
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
    "quality_tier",
    "source_selection_reason",
]

CANONICAL_5M_COLUMNS = [
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
    "source_selection_reason",
    "identity_mapping_status",
]


def _bar_end(value: Any) -> str:
    raw = str(value or "").strip()
    if not raw:
        return ""
    if ":" in raw:
        matches = pd.Series([raw]).str.extract(r"(\d{2}:\d{2})(?::\d{2})?", expand=False)
        return str(matches.iloc[0]) if not pd.isna(matches.iloc[0]) else ""
    integer_text = raw.split(".", 1)[0]
    digits = "".join(item for item in integer_text if item.isdigit())
    if len(digits) >= 12 and digits[:4].isdigit() and 1990 <= int(digits[:4]) <= 2100:
        hour, minute = digits[8:10], digits[10:12]
    elif len(digits) >= 6:
        # BaoStock's normalizer can expose HHMMSSmmm rather than the original
        # YYYYMMDDHHMMSSmmm.  Time must be read from the left, not the trailing
        # millisecond zeros.
        hour, minute = digits[:2], digits[2:4]
    elif len(digits) >= 4:
        hour, minute = digits[:2], digits[2:4]
    else:
        return ""
    if int(hour) <= 23 and int(minute) <= 59:
        return f"{hour}:{minute}"
    return ""


def normalize_provider_5m(frame: pd.DataFrame, *, provider_symbol: str, source: str) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame(columns=SELECTED_COLUMNS[:-3] + ["source"])
    data = frame.copy()
    symbol_column = "symbol" if "symbol" in data.columns else "provider_symbol" if "provider_symbol" in data.columns else ""
    if symbol_column:
        data["provider_symbol"] = data[symbol_column].map(normalize_symbol)
    else:
        data["provider_symbol"] = normalize_symbol(provider_symbol)
    if "trade_date" not in data.columns:
        raise ValueError(f"{source}_5m_trade_date_missing")
    time_column = "bar_time" if "bar_time" in data.columns else "bar_end" if "bar_end" in data.columns else "time" if "time" in data.columns else ""
    if not time_column:
        raise ValueError(f"{source}_5m_bar_time_missing")
    data["trade_date"] = pd.to_datetime(data["trade_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    data["bar_end"] = data[time_column].map(_bar_end)
    for column in ("open", "high", "low", "close", "volume", "amount"):
        data[column] = pd.to_numeric(data[column] if column in data.columns else np.nan, errors="coerce")
    data["source"] = str(source)
    return data.loc[:, ["provider_symbol", "trade_date", "bar_end", "open", "high", "low", "close", "volume", "amount", "source"]].dropna(subset=["trade_date"]).reset_index(drop=True)


def normalize_tushare_proxy_5m(frame: pd.DataFrame, *, provider_symbol: str) -> pd.DataFrame:
    """Normalize ``stk_mins``; its raw volume/amount are already shares/CNY.

    The compatibility gate proves the unique 1.0 scales against Tushare daily
    volume (lots) and amount (thousand CNY) before bootstrap data is admitted.
    """

    if frame is None or frame.empty:
        return normalize_provider_5m(pd.DataFrame(), provider_symbol=provider_symbol, source="tushare_proxy")
    data = frame.copy()
    timestamp_column = next(
        (
            column
            for column in ("trade_time", "datetime", "trade_datetime", "date", "time")
            if column in data.columns
        ),
        "",
    )
    if timestamp_column:
        timestamps = pd.to_datetime(data[timestamp_column], errors="coerce")
    elif "trade_date" in data.columns and "bar_time" in data.columns:
        timestamps = pd.to_datetime(
            data["trade_date"].astype(str).str.slice(0, 10) + " " + data["bar_time"].astype(str),
            errors="coerce",
        )
    else:
        raise ValueError("tushare_proxy_5m_timestamp_column_missing")
    if timestamps.isna().any():
        raise ValueError("tushare_proxy_5m_timestamp_missing_or_invalid")
    data["trade_date"] = timestamps.dt.strftime("%Y-%m-%d")
    data["bar_time"] = timestamps.dt.strftime("%H:%M")
    if "vol" in data.columns and "volume" not in data.columns:
        data = data.rename(columns={"vol": "volume"})
    return normalize_provider_5m(data, provider_symbol=provider_symbol, source="tushare_proxy")


def is_complete_5m_day(frame: pd.DataFrame) -> bool:
    if frame is None or len(frame) != 48:
        return False
    if set(frame["bar_end"].astype(str)) != set(EXPECTED_5M_BAR_ENDS):
        return False
    if frame["bar_end"].duplicated().any():
        return False
    numeric = frame[["open", "high", "low", "close", "volume", "amount"]].apply(pd.to_numeric, errors="coerce")
    if numeric.isna().any().any() or numeric[["open", "high", "low", "close"]].le(0).any().any():
        return False
    if numeric[["volume", "amount"]].lt(0).any().any():
        return False
    if (numeric["high"] < numeric[["open", "low", "close"]].max(axis=1)).any():
        return False
    if (numeric["low"] > numeric[["open", "high", "close"]].min(axis=1)).any():
        return False
    return True


def reconcile_5m_with_daily(day: pd.DataFrame, daily: Mapping[str, Any] | pd.Series | None) -> dict[str, Any]:
    """Check one complete 5m stock-day against the unadjusted daily fact."""

    if not is_complete_5m_day(day):
        return {"comparable": False, "conflict": True, "reason": "5m_day_incomplete"}
    if daily is None:
        return {"comparable": False, "conflict": False, "reason": "daily_reference_missing"}
    ordered = day.sort_values("bar_end")
    intraday = {
        "open": float(ordered.iloc[0]["open"]),
        "high": float(pd.to_numeric(ordered["high"], errors="coerce").max()),
        "low": float(pd.to_numeric(ordered["low"], errors="coerce").min()),
        "close": float(ordered.iloc[-1]["close"]),
        "volume": float(pd.to_numeric(ordered["volume"], errors="coerce").sum()),
        "amount": float(pd.to_numeric(ordered["amount"], errors="coerce").sum()),
    }
    reference = {
        key: float(pd.to_numeric(daily.get(key), errors="coerce"))
        for key in intraday
    }
    if not all(np.isfinite(value) for value in reference.values()):
        return {"comparable": False, "conflict": False, "reason": "daily_reference_incomplete"}
    price_diff = {
        key: abs(intraday[key] - reference[key])
        for key in ("open", "high", "low", "close")
    }
    volume_diff = abs(intraday["volume"] - reference["volume"])
    amount_diff = abs(intraday["amount"] - reference["amount"])
    conflict = (
        any(value > 0.010000001 for value in price_diff.values())
        or volume_diff > max(100.0, 1e-5 * abs(reference["volume"]))
        or amount_diff > max(1_000.0, 0.001 * abs(reference["amount"]))
    )
    return {
        "comparable": True,
        "conflict": bool(conflict),
        "intraday_aggregate": intraday,
        "daily_reference": reference,
        "price_absolute_difference": price_diff,
        "volume_absolute_difference": volume_diff,
        "volume_relative_difference": volume_diff / max(abs(reference["volume"]), 1.0),
        "amount_absolute_difference": amount_diff,
        "amount_relative_difference": amount_diff / max(abs(reference["amount"]), 1.0),
    }


def stable_security_bucket(security_id: str, *, bucket_count: int = 64) -> int:
    digest = hashlib.sha256(str(security_id).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % int(bucket_count)


def compare_complete_5m_days(primary: pd.DataFrame, secondary: pd.DataFrame) -> dict[str, Any]:
    if not is_complete_5m_day(primary) or not is_complete_5m_day(secondary):
        return {"comparable": False, "conflict": True, "reason": "one_or_both_sources_incomplete"}
    left = primary.sort_values("bar_end").set_index("bar_end")
    right = secondary.sort_values("bar_end").set_index("bar_end")
    price_diff = (left[["open", "high", "low", "close"]] - right[["open", "high", "low", "close"]]).abs()
    open_close_conflict = bool(price_diff[["open", "close"]].gt(0.010000001).any().any())
    high_low_exceeds = price_diff[["high", "low"]].gt(0.010000001)
    high_low_rate = float(high_low_exceeds.to_numpy().mean())
    flow_rates: dict[str, float] = {}
    for column in ("volume", "amount"):
        denominator = right[column].abs().replace(0, np.nan)
        relative = (left[column] - right[column]).abs().div(denominator).fillna((left[column] - right[column]).abs())
        flow_rates[column] = float(relative.gt(0.01).mean())
        flow_rates[f"{column}_count"] = int(relative.gt(0.01).sum())
    conflict = open_close_conflict or high_low_rate > 0.001 or max(flow_rates["volume"], flow_rates["amount"]) > 0.005
    return {
        "comparable": True,
        "conflict": bool(conflict),
        "open_close_gt_one_tick": open_close_conflict,
        "open_close_gt_one_tick_count": int(price_diff[["open", "close"]].gt(0.010000001).sum().sum()),
        "high_low_gt_one_tick_rate": high_low_rate,
        "high_low_gt_one_tick_count": int(high_low_exceeds.sum().sum()),
        "high_low_comparison_count": int(high_low_exceeds.size),
        "volume_gt_one_percent_rate": flow_rates["volume"],
        "amount_gt_one_percent_rate": flow_rates["amount"],
        "volume_gt_one_percent_count": int(flow_rates["volume_count"]),
        "amount_gt_one_percent_count": int(flow_rates["amount_count"]),
        "flow_comparison_count": int(len(left)),
    }


def select_5m_day(
    *,
    mootdx: pd.DataFrame,
    baostock: pd.DataFrame,
    trade_date: str,
    compare_sources: bool = False,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    mootdx_day = mootdx.loc[mootdx["trade_date"].eq(str(trade_date))].copy() if not mootdx.empty else mootdx.copy()
    baostock_day = baostock.loc[baostock["trade_date"].eq(str(trade_date))].copy() if not baostock.empty else baostock.copy()
    mootdx_complete = is_complete_5m_day(mootdx_day)
    baostock_complete = is_complete_5m_day(baostock_day)
    comparison: dict[str, Any] = {}
    if compare_sources and mootdx_complete and baostock_complete:
        comparison = compare_complete_5m_days(mootdx_day, baostock_day)
        if comparison.get("conflict"):
            return pd.DataFrame(columns=SELECTED_COLUMNS), {
                "status": QUALITY_QUARANTINED,
                "reason": "unresolved_dual_source_conflict",
                "comparison": comparison,
            }
    if compare_sources and mootdx_complete and not baostock_complete:
        comparison = {
            "comparable": False,
            "conflict": False,
            "reason": "baostock_comparison_unavailable",
            "mootdx_bar_count": int(len(mootdx_day)),
            "baostock_bar_count": int(len(baostock_day)),
        }
    if mootdx_complete:
        chosen = mootdx_day.copy()
        reason = "complete_mootdx_preferred"
    elif baostock_complete:
        chosen = baostock_day.copy()
        reason = "mootdx_incomplete_complete_baostock_fallback"
    else:
        return pd.DataFrame(columns=SELECTED_COLUMNS), {
            "status": QUALITY_QUARANTINED,
            "reason": "both_sources_incomplete_no_stitching",
            "mootdx_bar_count": int(len(mootdx_day)),
            "baostock_bar_count": int(len(baostock_day)),
        }
    tier = QUALITY_STRICT
    chosen["quality_tier"] = tier
    chosen["source_selection_reason"] = reason
    return chosen.loc[:, SELECTED_COLUMNS].sort_values("bar_end").reset_index(drop=True), {
        "status": tier,
        "reason": reason,
        "source": str(chosen["source"].iloc[0]),
        "comparison": comparison,
    }


def deterministic_monthly_sample(symbols: Iterable[str], *, month: str, count: int = 64) -> set[str]:
    normalized = sorted({normalize_symbol(item) for item in symbols if normalize_symbol(item)})
    ranked = sorted(normalized, key=lambda symbol: hashlib.sha256(f"{month}|{symbol}".encode("utf-8")).hexdigest())
    return set(ranked[: min(int(count), len(ranked))])


def deterministic_stratified_monthly_sample(
    universe: pd.DataFrame,
    *,
    month: str,
    count: int = 64,
) -> set[str]:
    """Sample exchange/liquidity strata deterministically.

    ``universe`` accepts ``provider_symbol`` (or ``symbol``), optional
    ``exchange``, and optional ``liquidity``. Missing liquidity is kept in an
    explicit UNKNOWN stratum instead of being silently imputed.
    """

    data = _sampling_universe_with_strata(universe)
    if data.empty:
        return set()
    buckets: dict[tuple[str, str], list[str]] = {}
    for row in data.loc[:, ["provider_symbol", "exchange", "liquidity_quintile"]].itertuples(index=False):
        buckets.setdefault((str(row.exchange), str(row.liquidity_quintile)), []).append(str(row.provider_symbol))
    for key in buckets:
        buckets[key] = sorted(buckets[key], key=lambda symbol: hashlib.sha256(f"{month}|{key}|{symbol}".encode("utf-8")).hexdigest())
    selected: list[str] = []
    positions = {key: 0 for key in buckets}
    ordered_keys = sorted(buckets)
    while len(selected) < min(int(count), len(data)):
        progressed = False
        for key in ordered_keys:
            position = positions[key]
            if position >= len(buckets[key]):
                continue
            selected.append(buckets[key][position])
            positions[key] += 1
            progressed = True
            if len(selected) >= min(int(count), len(data)):
                break
        if not progressed:
            break
    return set(selected)


def _sampling_universe_with_strata(universe: pd.DataFrame | None) -> pd.DataFrame:
    if universe is None or universe.empty:
        return pd.DataFrame(columns=["provider_symbol", "exchange", "liquidity_quintile"])
    data = universe.copy()
    symbol_column = "provider_symbol" if "provider_symbol" in data.columns else "symbol" if "symbol" in data.columns else ""
    if not symbol_column:
        raise ValueError("intraday_sampling_symbol_column_missing")
    data["provider_symbol"] = data[symbol_column].map(normalize_symbol)
    data = data.loc[data["provider_symbol"].ne("")].drop_duplicates("provider_symbol", keep="last")
    if "exchange" not in data.columns:
        data["exchange"] = data["provider_symbol"].str.rsplit(".", n=1).str[-1]
    data["exchange"] = data["exchange"].fillna("UNKNOWN").astype(str).str.upper().replace("", "UNKNOWN")
    liquidity_source = data["liquidity"] if "liquidity" in data.columns else pd.Series(np.nan, index=data.index)
    liquidity = pd.to_numeric(liquidity_source, errors="coerce")
    data["liquidity_quintile"] = "UNKNOWN"
    valid = liquidity.notna()
    if valid.any():
        ranked = liquidity.loc[valid].rank(method="first", pct=True)
        data.loc[valid, "liquidity_quintile"] = np.minimum(np.ceil(ranked * 5), 5).astype(int).map(lambda value: f"Q{value}")
    return data.loc[:, ["provider_symbol", "exchange", "liquidity_quintile"]].reset_index(drop=True)


def _stored_intraday_audit_evidence(
    *,
    symbols: set[str],
    trade_dates: list[str],
    workspace_root: str | Path | None,
) -> tuple[
    set[tuple[str, str]],
    dict[tuple[str, str], tuple[float, float]],
    dict[str, set[str]],
    set[str],
]:
    """Collect forced stock-days without using future state as a feature.

    Company-action dates, listing first months, suspension-recovery dates, and
    daily open/close facts are audit controls only; they never become model
    inputs here.
    """

    date_set = set(trade_dates)
    if not date_set:
        return set(), {}, {}, set()
    start_date, end_date = min(date_set), max(date_set)
    forced: set[tuple[str, str]] = set()
    daily_reference: dict[tuple[str, str], tuple[float, float]] = {}
    required_dates_by_symbol: dict[str, set[str]] = {symbol: set() for symbol in symbols}
    covered_dates: set[str] = set()
    all_daily_refs = iter_raw_partitions(RAW_DAILY_ASTOCK, workspace_root=workspace_root, end_value=end_date)
    prior = [ref for ref in all_daily_refs if ref.partition_value < start_date]
    daily_refs = [*prior[-1:], *[ref for ref in all_daily_refs if ref.partition_value >= start_date]]
    previous_status: dict[str, str] = {}
    for ref in daily_refs:
        raw = read_raw_partition(ref)
        if raw.empty or "code" not in raw.columns:
            continue
        if ref.partition_value in date_set:
            covered_dates.add(ref.partition_value)
        work = raw.copy()
        work["_symbol"] = work["code"].map(normalize_symbol)
        work = work.loc[work["_symbol"].isin(symbols)]
        date_values = work.get("date", pd.Series(ref.partition_value, index=work.index)).astype(str).str.slice(0, 10)
        statuses = work.get("tradestatus", pd.Series("", index=work.index)).fillna("").astype(str).str.strip()
        opens = pd.to_numeric(work.get("open", pd.Series(np.nan, index=work.index)), errors="coerce")
        closes = pd.to_numeric(work.get("close", pd.Series(np.nan, index=work.index)), errors="coerce")
        for index in work.index:
            symbol = str(work.at[index, "_symbol"])
            trade_date = str(date_values.at[index])
            status = str(statuses.at[index])
            if trade_date in date_set and status == "1":
                required_dates_by_symbol.setdefault(symbol, set()).add(trade_date)
            if trade_date in date_set and previous_status.get(symbol) == "0" and status == "1":
                forced.add((symbol, trade_date))
            previous_status[symbol] = status
            open_value, close_value = float(opens.at[index]), float(closes.at[index])
            if trade_date in date_set and np.isfinite(open_value) and np.isfinite(close_value) and open_value > 0 and close_value > 0:
                daily_reference[(symbol, trade_date)] = (open_value, close_value)
    factor_refs = iter_raw_partitions(
        RAW_ADJUST_FACTOR_EVENT,
        workspace_root=workspace_root,
        start_value=start_date,
        end_value=end_date,
    )
    for ref in factor_refs:
        raw = read_raw_partition(ref)
        if "code" not in raw.columns:
            continue
        for symbol in raw["code"].map(normalize_symbol):
            if symbol in symbols and ref.partition_value in date_set:
                forced.add((symbol, ref.partition_value))
    master_refs = iter_raw_partitions(RAW_SECURITY_MASTER, workspace_root=workspace_root)
    if master_refs:
        master = read_raw_partition(master_refs[-1])
        symbol_column = "symbol" if "symbol" in master.columns else "provider_symbol" if "provider_symbol" in master.columns else ""
        if symbol_column and "list_date" in master.columns:
            for row in master.loc[:, [symbol_column, "list_date"]].itertuples(index=False, name=None):
                symbol = normalize_symbol(row[0])
                list_date = "" if pd.isna(row[1]) else str(row[1])[:10]
                if symbol in symbols and len(list_date) == 10:
                    forced.update((symbol, date) for date in trade_dates if date[:7] == list_date[:7])
    return forced, daily_reference, required_dates_by_symbol, covered_dates


def _latest_month_partition(raw_domain: str, partition_value: str, *, workspace_root: str | Path | None) -> Any | None:
    return get_raw_partition(
        raw_domain,
        partition_field="symbol_month",
        partition_value=partition_value,
        workspace_root=workspace_root,
    )


def canonicalize_selected_5m(
    frame: pd.DataFrame,
    *,
    identity_registry: Any,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Map provider-current symbols to stable identities without hiding conflicts."""

    if frame is None or frame.empty:
        return pd.DataFrame(columns=CANONICAL_5M_COLUMNS), pd.DataFrame()
    mapped = identity_registry.map_frame(frame, provider_symbol_column="provider_symbol", date_column="trade_date")
    mapped["bar_end"] = mapped["bar_end"].map(_bar_end)
    invalid = mapped["identity_mapping_status"].ne("mapped")
    quarantine = mapped.loc[invalid].copy()
    if not quarantine.empty:
        quarantine["conflict_type"] = quarantine["identity_mapping_status"]
    canonical = mapped.loc[~invalid, CANONICAL_5M_COLUMNS].copy()
    key = ["security_id", "trade_date", "bar_end"]
    duplicate = canonical.duplicated(key, keep=False)
    conflict_rows: list[pd.DataFrame] = []
    keep: list[int] = list(canonical.index[~duplicate])
    compare = ["open", "high", "low", "close", "volume", "amount", "source", "quality_tier"]
    for _, group in canonical.loc[duplicate].groupby(key, sort=False, dropna=False):
        normalized = group.loc[:, compare].astype("string").fillna("<QDP_NULL>").drop_duplicates()
        if len(normalized) == 1:
            keep.append(int(group.index[0]))
        else:
            conflict = group.copy()
            conflict["conflict_type"] = "identity_mapped_intraday_value_conflict"
            conflict_rows.append(conflict)
    canonical = canonical.loc[sorted(keep)].sort_values(key).reset_index(drop=True)
    if conflict_rows:
        quarantine = pd.concat([quarantine, *conflict_rows], ignore_index=True)
    return canonical, quarantine.reset_index(drop=True)


def _merge_month_partition(
    *,
    raw_domain: str,
    partition_value: str,
    frame: pd.DataFrame,
    key: list[str],
    receipt: dict[str, Any],
    workspace_root: str | Path | None,
    replace_trade_dates: Iterable[str] = (),
    existing_ref: Any | None = None,
    lookup_existing: bool = True,
) -> Any:
    if lookup_existing:
        direct = get_raw_partition(
            raw_domain,
            partition_field="symbol_month",
            partition_value=partition_value,
            workspace_root=workspace_root,
        )
        existing = [direct] if direct is not None else []
    else:
        existing = [existing_ref] if existing_ref is not None else []
    combined = frame.copy()
    if existing:
        try:
            prior = read_raw_partition(existing[-1])
        except RuntimeError as exc:
            if not str(exc).startswith("raw_content_hash_mismatch:"):
                raise
            # Early v3 smoke partitions hashed pre-Parquet dtypes.  Their
            # parquet byte hash was already verified before this specific
            # mismatch; read them only to produce a corrected immutable
            # successor under the persisted-content hash contract.
            prior = read_raw_partition(existing[-1], verify_hash=False)
        replace_dates = {str(item) for item in replace_trade_dates}
        if replace_dates and "trade_date" in prior.columns:
            prior = prior.loc[~prior["trade_date"].astype(str).isin(replace_dates)].copy()
        if combined.empty:
            combined = prior
        elif not prior.empty:
            combined = pd.concat([prior, combined], ignore_index=True)
    if not combined.empty:
        combined = combined.drop_duplicates(key, keep="last").sort_values(key).reset_index(drop=True)
    return write_raw_partition(
        raw_domain=raw_domain,
        partition_field="symbol_month",
        partition_value=partition_value,
        frame=combined,
        receipt=receipt,
        workspace_root=workspace_root,
    )[0]


def _raw_request_covers_dates(ref: Any, dates: list[str]) -> bool:
    if ref is None or not dates:
        return False
    receipt = read_raw_receipt(ref)
    if list(receipt.get("provider_errors", []) or []):
        return False
    request = dict(receipt.get("request", {}) or {})
    start_date = str(request.get("start_date", "") or "")[:10]
    end_date = str(request.get("end_date", "") or "")[:10]
    return bool(start_date and end_date and start_date <= dates[0] and end_date >= dates[-1])


def _prefetch_mootdx_symbol_ranges(
    *,
    symbols: list[str],
    trade_dates: list[str],
    required_dates_by_symbol: Mapping[str, set[str]],
    daily_covered_dates: set[str],
    source: MootdxOnlineProvider,
    workspace_root: str | Path | None,
    refresh: bool,
) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]], dict[str, int]]:
    """Download each symbol once, then split its response into monthly raw partitions.

    TDX bar pagination walks backward from the latest quote. Querying one old
    month at a time therefore downloads the same newer pages repeatedly. A
    single earliest-to-latest request per symbol removes that quadratic network
    pattern while preserving the existing immutable symbol-month raw contract.
    """

    existing: dict[str, Any] = {}
    refs: dict[str, Any] = {}
    errors_by_symbol: dict[str, list[dict[str, Any]]] = {}
    metrics = {
        "range_prefetch_network_request_count": 0,
        "range_prefetch_downloaded_symbol_count": 0,
        "range_prefetch_reused_partition_count": 0,
        "range_prefetch_written_partition_count": 0,
    }
    for index, symbol in enumerate(symbols, start=1):
        symbol_dates = [
            date
            for date in trade_dates
            if date not in daily_covered_dates or date in required_dates_by_symbol.get(symbol, set())
        ]
        if not symbol_dates:
            continue
        dates_by_month = {
            month: [date for date in symbol_dates if date.startswith(month)]
            for month in sorted({date[:7] for date in symbol_dates})
        }
        partition_by_month = {month: f"{symbol}_{month.replace('-', '')}" for month in dates_by_month}
        for partition_value in partition_by_month.values():
            ref = get_raw_partition(
                RAW_INTRADAY_5M_MOOTDX,
                partition_field="symbol_month",
                partition_value=partition_value,
                workspace_root=workspace_root,
            )
            if ref is not None:
                existing[partition_value] = ref
                refs[partition_value] = ref
        if not refresh and all(
            _raw_request_covers_dates(existing.get(partition_by_month[month]), dates_by_month[month])
            for month in dates_by_month
        ):
            metrics["range_prefetch_reused_partition_count"] += len(dates_by_month)
            continue
        if index == 1 or index % 50 == 0 or index == len(symbols):
            progress_write(
                f"mootdx_5m_range_prefetch={index}/{len(symbols)} symbol={symbol} "
                f"range={symbol_dates[0]}..{symbol_dates[-1]}"
            )
        metrics["range_prefetch_network_request_count"] += 1
        try:
            result = source.fetch_domain(
                DomainFetchRequest(
                    domain=DataDomain.MARKET_INTRADAY_5M,
                    symbols=(symbol,),
                    start_date=symbol_dates[0],
                    end_date=symbol_dates[-1],
                    adjusted_flag="none",
                )
            )
            provider_errors = [
                dict(item)
                for item in list(result.error_report or [])
                if isinstance(item, Mapping)
            ]
            if provider_errors:
                errors_by_symbol[symbol] = provider_errors
                continue
            normalized = normalize_provider_5m(result.data, provider_symbol=symbol, source="mootdx")
            normalized = normalized.loc[
                normalized["provider_symbol"].eq(symbol) & normalized["trade_date"].isin(symbol_dates)
            ].copy()
        except Exception as exc:
            errors_by_symbol[symbol] = [
                {"provider": "mootdx", "symbol": symbol, "error": f"{type(exc).__name__}: {exc}"}
            ]
            continue
        metrics["range_prefetch_downloaded_symbol_count"] += 1
        for month, month_dates in dates_by_month.items():
            partition_value = partition_by_month[month]
            month_frame = normalized.loc[normalized["trade_date"].isin(month_dates)].copy()
            ref = _merge_month_partition(
                raw_domain=RAW_INTRADAY_5M_MOOTDX,
                partition_value=partition_value,
                frame=month_frame,
                key=["provider_symbol", "trade_date", "bar_end", "source"],
                receipt={
                    "provider": "mootdx",
                    "endpoint": "Quotes.bars",
                    "package_version": MOOTDX_VERSION,
                    "quality_tier": QUALITY_PROVISIONAL,
                    "download_strategy": "single_symbol_range_split_monthly",
                    "request": {
                        "symbol": symbol,
                        "month": month,
                        "start_date": month_dates[0],
                        "end_date": month_dates[-1],
                        "range_start_date": symbol_dates[0],
                        "range_end_date": symbol_dates[-1],
                    },
                    "provider_errors": [],
                },
                workspace_root=workspace_root,
                replace_trade_dates=month_dates,
                existing_ref=existing.get(partition_value),
                lookup_existing=False,
            )
            refs[partition_value] = ref
            metrics["range_prefetch_written_partition_count"] += 1
    return refs, errors_by_symbol, metrics


def _prefetch_baostock_symbol_range(
    *,
    symbol: str,
    symbol_dates: list[str],
    source: BaostockProvider,
    workspace_root: str | Path | None,
    refresh: bool,
    known_refs: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, int]]:
    """Fetch one BaoStock 5m history range and materialize its monthly shards."""

    metrics = {
        "baostock_range_prefetch_network_request_count": 0,
        "baostock_range_prefetch_reused_partition_count": 0,
        "baostock_range_prefetch_written_partition_count": 0,
    }
    if not symbol_dates:
        return known_refs, [], metrics
    dates_by_month = {
        month: [date for date in symbol_dates if date.startswith(month)]
        for month in sorted({date[:7] for date in symbol_dates})
    }
    partition_by_month = {month: f"{symbol}_{month.replace('-', '')}" for month in dates_by_month}
    for partition_value in partition_by_month.values():
        if partition_value in known_refs:
            continue
        ref = get_raw_partition(
            RAW_INTRADAY_5M_BAOSTOCK,
            partition_field="symbol_month",
            partition_value=partition_value,
            workspace_root=workspace_root,
        )
        if ref is not None:
            known_refs[partition_value] = ref
    if not refresh and all(
        _raw_request_covers_dates(known_refs.get(partition_by_month[month]), dates_by_month[month])
        for month in dates_by_month
    ):
        metrics["baostock_range_prefetch_reused_partition_count"] = len(dates_by_month)
        return known_refs, [], metrics
    metrics["baostock_range_prefetch_network_request_count"] = 1
    try:
        result = source.fetch_domain(
            DomainFetchRequest(
                domain=DataDomain.MARKET_INTRADAY_5M,
                symbols=(symbol,),
                start_date=symbol_dates[0],
                end_date=symbol_dates[-1],
                adjusted_flag="none",
            )
        )
        provider_errors = [
            dict(item)
            for item in list(result.error_report or [])
            if isinstance(item, Mapping)
        ]
        if provider_errors:
            return known_refs, provider_errors, metrics
        normalized = normalize_provider_5m(result.data, provider_symbol=symbol, source="baostock")
        normalized = normalized.loc[
            normalized["provider_symbol"].eq(symbol) & normalized["trade_date"].isin(symbol_dates)
        ].copy()
    except Exception as exc:
        return known_refs, [{"provider": "baostock", "symbol": symbol, "error": f"{type(exc).__name__}: {exc}"}], metrics
    for month, month_dates in dates_by_month.items():
        partition_value = partition_by_month[month]
        month_frame = normalized.loc[normalized["trade_date"].isin(month_dates)].copy()
        ref = _merge_month_partition(
            raw_domain=RAW_INTRADAY_5M_BAOSTOCK,
            partition_value=partition_value,
            frame=month_frame,
            key=["provider_symbol", "trade_date", "bar_end", "source"],
            receipt={
                "provider": "baostock",
                "endpoint": "query_history_k_data_plus(frequency=5)",
                "package_version": importlib_metadata.version("baostock"),
                "wheel_sha256": BAOSTOCK_BATCH_WHEEL_SHA256,
                "quality_tier": QUALITY_PROVISIONAL,
                "download_strategy": "single_symbol_range_split_monthly",
                "request": {
                    "symbol": symbol,
                    "month": month,
                    "start_date": month_dates[0],
                    "end_date": month_dates[-1],
                    "range_start_date": symbol_dates[0],
                    "range_end_date": symbol_dates[-1],
                },
                "provider_errors": [],
            },
            workspace_root=workspace_root,
            replace_trade_dates=month_dates,
            existing_ref=known_refs.get(partition_value),
            lookup_existing=False,
        )
        known_refs[partition_value] = ref
        metrics["baostock_range_prefetch_written_partition_count"] += 1
    return known_refs, [], metrics


def ingest_intraday_5m(
    *,
    symbols: Iterable[str],
    trade_dates: Iterable[str],
    workspace_root: str | Path | None = None,
    mootdx_provider: MootdxOnlineProvider | None = None,
    baostock_provider: BaostockProvider | None = None,
    comparison_sample_count: int = 64,
    sampling_universe: pd.DataFrame | None = None,
    force_compare_stock_days: Iterable[tuple[str, str]] = (),
    refresh: bool = False,
) -> dict[str, Any]:
    symbol_list = sorted({normalize_symbol(item) for item in symbols if normalize_symbol(item)})
    dates = sorted({str(item)[:10] for item in trade_dates if str(item).strip()})
    if not symbol_list or not dates:
        raise ValueError("intraday_5m_symbols_and_dates_required")
    if baostock_provider is None or isinstance(baostock_provider, BaostockProvider):
        from quant_data_platform.qdp_v3.compatibility import assert_baostock_compatibility_gate

        assert_baostock_compatibility_gate(workspace_root)
    mootdx_source = mootdx_provider or MootdxOnlineProvider()
    baostock_source = baostock_provider or BaostockProvider(_reuse_symbol_range_session=True)
    sample_frame = sampling_universe.copy() if isinstance(sampling_universe, pd.DataFrame) else pd.DataFrame({"provider_symbol": symbol_list})
    stored_forced, daily_reference, required_dates_by_symbol, daily_covered_dates = _stored_intraday_audit_evidence(
        symbols=set(symbol_list),
        trade_dates=dates,
        workspace_root=workspace_root,
    )
    range_prefetch_enabled = bool(getattr(mootdx_source, "supports_backward_range_prefetch", False))
    if range_prefetch_enabled:
        mootdx_prefetch_refs, mootdx_prefetch_errors, prefetch_metrics = _prefetch_mootdx_symbol_ranges(
            symbols=symbol_list,
            trade_dates=dates,
            required_dates_by_symbol=required_dates_by_symbol,
            daily_covered_dates=daily_covered_dates,
            source=mootdx_source,
            workspace_root=workspace_root,
            refresh=refresh,
        )
    else:
        mootdx_prefetch_refs = {}
        mootdx_prefetch_errors = {}
        prefetch_metrics = {
            "range_prefetch_network_request_count": 0,
            "range_prefetch_downloaded_symbol_count": 0,
            "range_prefetch_reused_partition_count": 0,
            "range_prefetch_written_partition_count": 0,
        }
    date_set = set(dates)
    explicit_forced = {
        (normalize_symbol(symbol), str(trade_date)[:10])
        for symbol, trade_date in force_compare_stock_days
        if normalize_symbol(symbol) and str(trade_date)[:10] in date_set
    }
    forced_stock_days = stored_forced | explicit_forced
    results: list[dict[str, Any]] = []
    selected_refs: list[str] = []
    escalated_records: list[dict[str, Any]] = []
    active_symbol_month_count = 0
    excluded_nontrading_stock_day_count = 0
    empty_columns = ["provider_symbol", "trade_date", "bar_end", "open", "high", "low", "close", "volume", "amount", "source"]
    all_symbol_dates_by_symbol = {
        symbol: [
            date
            for date in dates
            if date not in daily_covered_dates or date in required_dates_by_symbol.get(symbol, set())
        ]
        for symbol in symbol_list
    }
    baostock_range_prefetch_enabled = bool(getattr(baostock_source, "supports_symbol_range_prefetch", False))
    baostock_prefetched_symbols: set[str] = set()
    baostock_prefetch_refs: dict[str, Any] = {}
    baostock_prefetch_errors: dict[str, list[dict[str, Any]]] = {}
    baostock_prefetch_metrics = {
        "baostock_range_prefetch_network_request_count": 0,
        "baostock_range_prefetch_reused_partition_count": 0,
        "baostock_range_prefetch_written_partition_count": 0,
    }

    def fetch_month(source: Any, *, source_name: str, symbol: str, month_dates: list[str]) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
        if source_name == "mootdx" and range_prefetch_enabled:
            if symbol in mootdx_prefetch_errors:
                return pd.DataFrame(columns=empty_columns), list(mootdx_prefetch_errors[symbol])
            partition_value = f"{symbol}_{month_dates[0][:7].replace('-', '')}"
            ref = mootdx_prefetch_refs.get(partition_value)
            if ref is None:
                return pd.DataFrame(columns=empty_columns), [
                    {
                        "provider": "mootdx",
                        "symbol": symbol,
                        "error": "range_prefetch_partition_missing",
                    }
                ]
            frame = read_raw_partition(ref)
            frame = frame.loc[
                frame["provider_symbol"].eq(symbol) & frame["trade_date"].isin(month_dates)
            ].copy()
            return frame, []
        if source_name == "baostock" and baostock_range_prefetch_enabled:
            if symbol not in baostock_prefetched_symbols:
                refs, errors, metrics = _prefetch_baostock_symbol_range(
                    symbol=symbol,
                    symbol_dates=all_symbol_dates_by_symbol[symbol],
                    source=baostock_source,
                    workspace_root=workspace_root,
                    refresh=refresh,
                    known_refs=baostock_prefetch_refs,
                )
                baostock_prefetch_refs.update(refs)
                if errors:
                    baostock_prefetch_errors[symbol] = errors
                for key, value in metrics.items():
                    baostock_prefetch_metrics[key] += int(value)
                baostock_prefetched_symbols.add(symbol)
            if symbol in baostock_prefetch_errors:
                return pd.DataFrame(columns=empty_columns), list(baostock_prefetch_errors[symbol])
            partition_value = f"{symbol}_{month_dates[0][:7].replace('-', '')}"
            ref = baostock_prefetch_refs.get(partition_value)
            if ref is None:
                return pd.DataFrame(columns=empty_columns), [
                    {"provider": "baostock", "symbol": symbol, "error": "range_prefetch_partition_missing"}
                ]
            frame = read_raw_partition(ref)
            frame = frame.loc[
                frame["provider_symbol"].eq(symbol) & frame["trade_date"].isin(month_dates)
            ].copy()
            return frame, []
        request = DomainFetchRequest(
            domain=DataDomain.MARKET_INTRADAY_5M,
            symbols=(symbol,),
            start_date=month_dates[0],
            end_date=month_dates[-1],
            adjusted_flag="none",
        )
        try:
            provider_result = source.fetch_domain(request)
            errors = [dict(item) for item in list(provider_result.error_report or []) if isinstance(item, Mapping)]
            if errors:
                return pd.DataFrame(columns=empty_columns), errors
            frame = normalize_provider_5m(provider_result.data, provider_symbol=symbol, source=source_name)
            frame = frame.loc[
                frame["provider_symbol"].eq(symbol) & frame["trade_date"].isin(month_dates)
            ].copy()
            return frame, []
        except Exception as exc:
            return pd.DataFrame(columns=empty_columns), [{"provider": source_name, "error": f"{type(exc).__name__}: {exc}"}]

    for month in sorted({date[:7] for date in dates}):
        month_dates = [date for date in dates if date.startswith(month)]
        month_key = month.replace("-", "")
        month_frame = sample_frame.loc[sample_frame["month"].astype(str).eq(month)].copy() if "month" in sample_frame.columns else sample_frame.copy()
        symbol_dates_by_symbol = {
            symbol: [date for date in all_symbol_dates_by_symbol[symbol] if date.startswith(month)]
            for symbol in symbol_list
        }
        active_symbols = [symbol for symbol in symbol_list if symbol_dates_by_symbol[symbol]]
        active_symbol_month_count += len(active_symbols)
        excluded_nontrading_stock_day_count += sum(len(month_dates) - len(symbol_dates_by_symbol[symbol]) for symbol in symbol_list)
        if not active_symbols:
            continue
        if not month_frame.empty:
            month_symbol_column = "provider_symbol" if "provider_symbol" in month_frame.columns else "symbol" if "symbol" in month_frame.columns else ""
            if month_symbol_column:
                month_frame = month_frame.loc[month_frame[month_symbol_column].map(normalize_symbol).isin(active_symbols)].copy()
        samples = deterministic_stratified_monthly_sample(month_frame, month=month, count=comparison_sample_count)
        samples = (samples & set(active_symbols)) or deterministic_monthly_sample(active_symbols, month=month, count=comparison_sample_count)
        strata_frame = _sampling_universe_with_strata(month_frame)
        stratum_by_symbol = {
            str(row.provider_symbol): f"{row.exchange}|{row.liquidity_quintile}"
            for row in strata_frame.itertuples(index=False)
        }
        for symbol in active_symbols:
            stratum_by_symbol.setdefault(symbol, f"{symbol.rsplit('.', 1)[-1]}|UNKNOWN")
        forced_by_symbol = {
            symbol: {
                date
                for item_symbol, date in forced_stock_days
                if item_symbol == symbol and date in symbol_dates_by_symbol[symbol]
            }
            for symbol in active_symbols
        }
        preflight_symbols = sorted(set(samples) | {symbol for symbol, forced_dates in forced_by_symbol.items() if forced_dates})
        cache: dict[str, tuple[pd.DataFrame, list[dict[str, Any]], pd.DataFrame, list[dict[str, Any]]]] = {}
        stratum_stats: dict[str, dict[str, Any]] = {}

        def stat_for(stratum: str) -> dict[str, Any]:
            return stratum_stats.setdefault(
                stratum,
                {
                    "required_stock_day_count": 0,
                    "comparable_stock_day_count": 0,
                    "open_close_gt_one_tick_count": 0,
                    "high_low_gt_one_tick_count": 0,
                    "high_low_comparison_count": 0,
                    "volume_gt_one_percent_count": 0,
                    "amount_gt_one_percent_count": 0,
                    "flow_comparison_count": 0,
                    "provider_error_count": 0,
                },
            )

        for symbol in preflight_symbols:
            symbol_dates = symbol_dates_by_symbol[symbol]
            mootdx, mootdx_errors = fetch_month(mootdx_source, source_name="mootdx", symbol=symbol, month_dates=symbol_dates)
            baostock, baostock_errors = fetch_month(baostock_source, source_name="baostock", symbol=symbol, month_dates=symbol_dates)
            cache[symbol] = (mootdx, mootdx_errors, baostock, baostock_errors)
            required_dates = set(symbol_dates) if symbol in samples else set(forced_by_symbol.get(symbol, set()))
            stats = stat_for(stratum_by_symbol[symbol])
            stats["provider_error_count"] += int(bool(mootdx_errors or baostock_errors))
            for trade_date in sorted(required_dates):
                stats["required_stock_day_count"] += 1
                left = mootdx.loc[mootdx["trade_date"].eq(trade_date)]
                right = baostock.loc[baostock["trade_date"].eq(trade_date)]
                if not is_complete_5m_day(left) or not is_complete_5m_day(right):
                    continue
                comparison = compare_complete_5m_days(left, right)
                stats["comparable_stock_day_count"] += 1
                for field in (
                    "open_close_gt_one_tick_count",
                    "high_low_gt_one_tick_count",
                    "high_low_comparison_count",
                    "volume_gt_one_percent_count",
                    "amount_gt_one_percent_count",
                    "flow_comparison_count",
                ):
                    stats[field] += int(comparison.get(field, 0) or 0)

        escalated_strata: set[str] = set()
        for stratum, stats in sorted(stratum_stats.items()):
            high_low_rate = stats["high_low_gt_one_tick_count"] / max(stats["high_low_comparison_count"], 1)
            volume_rate = stats["volume_gt_one_percent_count"] / max(stats["flow_comparison_count"], 1)
            amount_rate = stats["amount_gt_one_percent_count"] / max(stats["flow_comparison_count"], 1)
            incomplete_proof = stats["comparable_stock_day_count"] < stats["required_stock_day_count"]
            reasons = []
            if stats["open_close_gt_one_tick_count"]:
                reasons.append("open_close_gt_one_tick")
            if high_low_rate > 0.001:
                reasons.append("high_low_rate_gt_0_1_percent")
            if volume_rate > 0.005 or amount_rate > 0.005:
                reasons.append("volume_or_amount_rate_gt_0_5_percent")
            if stats["provider_error_count"] or incomplete_proof:
                reasons.append("sample_or_forced_proof_incomplete")
            if reasons:
                escalated_strata.add(stratum)
            escalated_records.append(
                {
                    "month": month,
                    "stratum": stratum,
                    **stats,
                    "high_low_gt_one_tick_rate": high_low_rate,
                    "volume_gt_one_percent_rate": volume_rate,
                    "amount_gt_one_percent_rate": amount_rate,
                    "escalated": bool(reasons),
                    "reasons": reasons,
                }
            )

        for symbol in active_symbols:
            symbol_dates = symbol_dates_by_symbol[symbol]
            partition_value = f"{symbol}_{month_key}"
            existing_selected = _latest_month_partition(RAW_INTRADAY_5M_SELECTED, partition_value, workspace_root=workspace_root)
            if not refresh and symbol not in preflight_symbols and stratum_by_symbol[symbol] not in escalated_strata and existing_selected is not None:
                existing_receipt = read_raw_receipt(existing_selected)
                prior_results = [dict(item) for item in list(existing_receipt.get("day_results", []) or []) if isinstance(item, Mapping)]
                if set(symbol_dates).issubset({str(item.get("trade_date", "")) for item in prior_results}):
                    results.extend({"symbol": symbol, "month": month, "resumed": True, **item} for item in prior_results if str(item.get("trade_date", "")) in set(symbol_dates))
                    selected_refs.append(existing_selected.content_sha256)
                    continue
            if symbol in cache:
                mootdx_month, mootdx_errors, baostock_month, baostock_errors = cache[symbol]
            else:
                mootdx_month, mootdx_errors = fetch_month(mootdx_source, source_name="mootdx", symbol=symbol, month_dates=symbol_dates)
                baostock_month, baostock_errors = pd.DataFrame(columns=empty_columns), []
            incomplete_dates = [date for date in symbol_dates if not is_complete_5m_day(mootdx_month.loc[mootdx_month["trade_date"].eq(date)])]
            compare_dates = set(symbol_dates) if symbol in samples else set(forced_by_symbol.get(symbol, set()))
            if stratum_by_symbol[symbol] in escalated_strata:
                compare_dates = set(symbol_dates)
            for index, trade_date in enumerate(symbol_dates):
                if trade_date in incomplete_dates:
                    if index > 0:
                        compare_dates.add(symbol_dates[index - 1])
                    if index + 1 < len(symbol_dates):
                        compare_dates.add(symbol_dates[index + 1])
                day = mootdx_month.loc[mootdx_month["trade_date"].eq(trade_date)]
                reference = daily_reference.get((symbol, trade_date))
                if reference and is_complete_5m_day(day):
                    ordered = day.sort_values("bar_end")
                    if abs(float(ordered.iloc[0]["open"]) - reference[0]) > 0.010000001 or abs(float(ordered.iloc[-1]["close"]) - reference[1]) > 0.010000001:
                        compare_dates.add(trade_date)
            need_baostock = bool(incomplete_dates or compare_dates)
            if need_baostock and symbol not in cache:
                baostock_month, baostock_errors = fetch_month(baostock_source, source_name="baostock", symbol=symbol, month_dates=symbol_dates)
            provider_errors = [*mootdx_errors, *baostock_errors]
            mootdx_ref = None
            if not mootdx_errors:
                mootdx_ref = mootdx_prefetch_refs.get(partition_value) if range_prefetch_enabled else None
                if mootdx_ref is None:
                    mootdx_ref = _merge_month_partition(
                        raw_domain=RAW_INTRADAY_5M_MOOTDX,
                        partition_value=partition_value,
                        frame=mootdx_month,
                        key=["provider_symbol", "trade_date", "bar_end", "source"],
                        receipt={
                            "provider": "mootdx",
                            "endpoint": "Quotes.bars",
                            "package_version": MOOTDX_VERSION,
                            "quality_tier": QUALITY_PROVISIONAL,
                            "request": {"symbol": symbol, "month": month, "start_date": symbol_dates[0], "end_date": symbol_dates[-1]},
                            "provider_errors": [],
                        },
                        workspace_root=workspace_root,
                        replace_trade_dates=month_dates,
                    )
            baostock_ref = None
            if need_baostock and not baostock_errors:
                baostock_ref = baostock_prefetch_refs.get(partition_value) if baostock_range_prefetch_enabled else None
                if baostock_ref is None:
                    baostock_ref = _merge_month_partition(
                        raw_domain=RAW_INTRADAY_5M_BAOSTOCK,
                        partition_value=partition_value,
                        frame=baostock_month,
                        key=["provider_symbol", "trade_date", "bar_end", "source"],
                        receipt={
                            "provider": "baostock",
                            "endpoint": "query_history_k_data_plus(frequency=5)",
                            "package_version": importlib_metadata.version("baostock"),
                            "wheel_sha256": BAOSTOCK_BATCH_WHEEL_SHA256,
                            "quality_tier": QUALITY_PROVISIONAL,
                            "request": {"symbol": symbol, "month": month, "start_date": symbol_dates[0], "end_date": symbol_dates[-1]},
                            "provider_errors": [],
                        },
                        workspace_root=workspace_root,
                        replace_trade_dates=month_dates,
                    )
            selected_days: list[pd.DataFrame] = []
            day_results: list[dict[str, Any]] = []
            for date in symbol_dates:
                mootdx_complete = is_complete_5m_day(mootdx_month.loc[mootdx_month["trade_date"].eq(date)])
                selected, evidence = select_5m_day(
                    mootdx=mootdx_month,
                    baostock=baostock_month,
                    trade_date=date,
                    compare_sources=date in compare_dates and mootdx_complete,
                )
                scopes = []
                if symbol in samples:
                    scopes.append("monthly_stratified_sample")
                if date in forced_by_symbol.get(symbol, set()):
                    scopes.append("event_listing_or_resume")
                if stratum_by_symbol[symbol] in escalated_strata:
                    scopes.append("stratum_full_review")
                if date in compare_dates and not scopes:
                    scopes.append("source_switch_or_open_close_anomaly")
                day_results.append(
                    {
                        "trade_date": date,
                        "stratum": stratum_by_symbol[symbol],
                        "comparison_scope": scopes,
                        "provider_errors": provider_errors,
                        **evidence,
                    }
                )
                if not selected.empty:
                    selected_days.append(selected)
            selected_month = pd.concat(selected_days, ignore_index=True) if selected_days else pd.DataFrame(columns=SELECTED_COLUMNS)
            failed_days = [item for item in day_results if item.get("status") == QUALITY_QUARANTINED]
            partition_tier = (
                QUALITY_QUARANTINED
                if failed_days
                else QUALITY_PROVISIONAL
                if any(item.get("status") == QUALITY_PROVISIONAL for item in day_results)
                else QUALITY_STRICT
            )
            ref = _merge_month_partition(
                raw_domain=RAW_INTRADAY_5M_SELECTED,
                partition_value=partition_value,
                frame=selected_month,
                key=["provider_symbol", "trade_date", "bar_end"],
                receipt={
                    "provider": "qdp_v3_source_selector",
                    "endpoint": "mootdx_then_baostock_no_stitch",
                    "quality_tier": partition_tier,
                    "request": {"symbol": symbol, "month": month, "start_date": symbol_dates[0], "end_date": symbol_dates[-1]},
                    "day_results": day_results,
                    "quarantined_day_count": len(failed_days),
                    "stratum": stratum_by_symbol[symbol],
                    "stratum_escalated": stratum_by_symbol[symbol] in escalated_strata,
                    "inputs": [
                        *([{"raw_domain": mootdx_ref.raw_domain, "content_sha256": mootdx_ref.content_sha256}] if mootdx_ref else []),
                        *([{"raw_domain": baostock_ref.raw_domain, "content_sha256": baostock_ref.content_sha256}] if baostock_ref else []),
                    ],
                    "provider_errors": provider_errors,
                },
                workspace_root=workspace_root,
                replace_trade_dates=month_dates,
            )
            selected_refs.append(ref.content_sha256)
            results.extend({"symbol": symbol, "month": month, **item} for item in day_results)
    failed = [item for item in results if item.get("status") == QUALITY_QUARANTINED]
    if mootdx_provider is None:
        mootdx_source.close()
    if baostock_provider is None:
        baostock_source.close()
    return {
        "status": "partial" if failed else "completed",
        "symbol_count": len(symbol_list),
        "trade_date_count": len(dates),
        "stock_day_count": len(results),
        "active_symbol_month_count": active_symbol_month_count,
        "excluded_nontrading_stock_day_count": excluded_nontrading_stock_day_count,
        "strict_stock_day_count": sum(item.get("status") == QUALITY_STRICT for item in results),
        "provisional_stock_day_count": sum(item.get("status") == QUALITY_PROVISIONAL for item in results),
        "quarantined_stock_day_count": len(failed),
        "selected_partition_hashes": sorted(set(selected_refs)),
        "mootdx_download_strategy": "single_symbol_range_split_monthly" if range_prefetch_enabled else "symbol_month",
        "baostock_download_strategy": "single_symbol_range_split_monthly" if baostock_range_prefetch_enabled else "symbol_month",
        **prefetch_metrics,
        **baostock_prefetch_metrics,
        "forced_comparison_stock_day_count": sum(
            1
            for symbol, trade_date in forced_stock_days
            if trade_date not in daily_covered_dates or trade_date in required_dates_by_symbol.get(symbol, set())
        ),
        "escalated_strata": escalated_records,
        "failed": failed[:100],
    }
