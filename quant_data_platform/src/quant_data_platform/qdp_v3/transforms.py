from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from quant_data_platform.qdp_v3.identity import SecurityIdentityRegistry, board_for_symbol, normalize_symbol


@dataclass(frozen=True)
class DailyDomainFrames:
    market_daily_raw: pd.DataFrame
    security_status_daily: pd.DataFrame
    valuation_daily: pd.DataFrame
    quarantine: pd.DataFrame


def _numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(np.nan, index=frame.index, dtype="float64")
    return pd.to_numeric(frame[column].replace("", pd.NA), errors="coerce")


def _deduplicate_mapped_rows(
    frame: pd.DataFrame,
    *,
    key: list[str],
    compare_columns: list[str],
    conflict_type: str,
    prefer_symbol_on_date: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if frame.empty:
        return frame.copy(), pd.DataFrame()
    duplicate = frame.duplicated(key, keep=False)
    if not duplicate.any():
        return frame.reset_index(drop=True), pd.DataFrame()
    keep_indices: list[int] = list(frame.index[~duplicate])
    conflicts: list[dict[str, Any]] = []
    for key_values, group in frame.loc[duplicate].groupby(key, sort=False, dropna=False):
        preferred = group.iloc[0:0]
        if prefer_symbol_on_date and {"provider_symbol", "symbol_on_date"}.issubset(group.columns):
            preferred = group.loc[
                group["symbol_on_date"].astype(str).ne("")
                & group["provider_symbol"].astype(str).eq(group["symbol_on_date"].astype(str))
            ]
        comparison = group.loc[:, compare_columns].copy()
        comparison = comparison.astype("string").fillna("<QDP_NULL>").drop_duplicates()
        if len(comparison) == 1:
            selected = preferred.iloc[0] if len(preferred) == 1 else group.iloc[0]
            keep_indices.append(int(selected.name))
            continue
        values = key_values if isinstance(key_values, tuple) else (key_values,)
        conflict = {column: value for column, value in zip(key, values)}
        differing_columns = [
            column
            for column in compare_columns
            if group[column].astype("string").fillna("<QDP_NULL>").nunique(dropna=False) > 1
        ]
        resolved = len(preferred) == 1
        if resolved:
            keep_indices.append(int(preferred.index[0]))
        conflict.update(
            {
                "conflict_type": conflict_type,
                "provider_symbols": sorted(set(group.get("provider_symbol", pd.Series(dtype=str)).astype(str))),
                "row_count": int(len(group)),
                "differing_columns": differing_columns,
                "resolution_status": "resolved" if resolved else "unresolved",
                "resolution_method": "official_pit_symbol_on_date" if resolved else "",
                "selected_provider_symbol": str(preferred.iloc[0]["provider_symbol"]) if resolved else "",
            }
        )
        conflicts.append(conflict)
    return frame.loc[sorted(keep_indices)].reset_index(drop=True), pd.DataFrame(conflicts)


def derive_daily_domains(
    raw: pd.DataFrame,
    *,
    query_date: str,
    identity_registry: SecurityIdentityRegistry,
) -> DailyDomainFrames:
    required = {
        "date",
        "code",
        "open",
        "high",
        "low",
        "close",
        "preclose",
        "volume",
        "amount",
        "adjustflag",
        "turn",
        "tradestatus",
        "pctChg",
        "peTTM",
        "pbMRQ",
        "psTTM",
        "pcfNcfTTM",
        "isST",
    }
    missing = sorted(required - set(raw.columns))
    if missing:
        raise ValueError(f"baostock_daily_raw_missing_fields:{missing}")
    base = pd.DataFrame(
        {
            "trade_date": raw["date"].fillna("").astype(str).str.strip(),
            "provider_symbol": raw["code"].map(normalize_symbol),
            "open": _numeric(raw, "open"),
            "high": _numeric(raw, "high"),
            "low": _numeric(raw, "low"),
            "close": _numeric(raw, "close"),
            "preclose": _numeric(raw, "preclose"),
            "volume": _numeric(raw, "volume"),
            "amount": _numeric(raw, "amount"),
            "pct_chg": _numeric(raw, "pctChg"),
            "adjustflag": raw["adjustflag"].fillna("").astype(str).str.strip(),
            "tradestatus": raw["tradestatus"].fillna("").astype(str).str.strip(),
            "is_st": raw["isST"].fillna("").astype(str).str.strip().eq("1"),
            "turnover_rate": _numeric(raw, "turn"),
            "pe_ttm": _numeric(raw, "peTTM"),
            "pb_mrq": _numeric(raw, "pbMRQ"),
            "ps_ttm": _numeric(raw, "psTTM"),
            "pcf_ncf_ttm": _numeric(raw, "pcfNcfTTM"),
        }
    )
    bad_date = base["trade_date"].ne(str(query_date))
    if bad_date.any():
        raise ValueError(f"baostock_daily_query_date_mismatch:{sorted(base.loc[bad_date, 'trade_date'].unique())[:10]}")
    base = identity_registry.map_frame(base)
    suspended = base["tradestatus"].ne("1")
    price_columns = ["open", "high", "low", "close", "preclose"]
    for column in price_columns:
        base.loc[suspended & base[column].eq(0), column] = np.nan
    market_columns = [
        "security_id",
        "trade_date",
        "symbol_on_date",
        "provider_symbol",
        "open",
        "high",
        "low",
        "close",
        "preclose",
        "volume",
        "amount",
        "pct_chg",
        "tradestatus",
        "identity_mapping_status",
    ]
    market = base.loc[:, market_columns].copy()
    market["source"] = "baostock.query_daily_history_k_AStock"
    market, market_conflicts = _deduplicate_mapped_rows(
        market,
        key=["security_id", "trade_date"],
        compare_columns=["open", "high", "low", "close", "preclose", "volume", "amount", "pct_chg", "tradestatus"],
        conflict_type="identity_mapped_daily_value_conflict",
        prefer_symbol_on_date=True,
    )
    status = base.loc[
        :,
        [
            "security_id",
            "trade_date",
            "symbol_on_date",
            "provider_symbol",
            "tradestatus",
            "is_st",
            "identity_mapping_status",
        ],
    ].copy()
    status["is_suspended"] = status["tradestatus"].ne("1")
    status["list_status"] = "listed"
    status["status_source"] = "baostock.query_daily_history_k_AStock"
    status, status_conflicts = _deduplicate_mapped_rows(
        status,
        key=["security_id", "trade_date"],
        compare_columns=["tradestatus", "is_st", "is_suspended", "list_status"],
        conflict_type="identity_mapped_status_conflict",
        prefer_symbol_on_date=True,
    )
    valuation = base.loc[
        :,
        [
            "security_id",
            "trade_date",
            "symbol_on_date",
            "provider_symbol",
            "turnover_rate",
            "pe_ttm",
            "pb_mrq",
            "ps_ttm",
            "pcf_ncf_ttm",
            "identity_mapping_status",
        ],
    ].copy()
    valuation["source"] = "baostock.query_daily_history_k_AStock"
    valuation, valuation_conflicts = _deduplicate_mapped_rows(
        valuation,
        key=["security_id", "trade_date"],
        compare_columns=["turnover_rate", "pe_ttm", "pb_mrq", "ps_ttm", "pcf_ncf_ttm"],
        conflict_type="identity_mapped_valuation_conflict",
        prefer_symbol_on_date=True,
    )
    quarantine = pd.concat(
        [item for item in (market_conflicts, status_conflicts, valuation_conflicts) if not item.empty],
        ignore_index=True,
    ) if any(not item.empty for item in (market_conflicts, status_conflicts, valuation_conflicts)) else pd.DataFrame()
    return DailyDomainFrames(
        market_daily_raw=market.sort_values(["trade_date", "security_id"]).reset_index(drop=True),
        security_status_daily=status.sort_values(["trade_date", "security_id"]).reset_index(drop=True),
        valuation_daily=valuation.sort_values(["trade_date", "security_id"]).reset_index(drop=True),
        quarantine=quarantine,
    )


def derive_adjust_factor_events(
    raw: pd.DataFrame,
    *,
    query_date: str,
    identity_registry: SecurityIdentityRegistry,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    columns = [
        "security_id",
        "divid_operate_date",
        "symbol_on_date",
        "provider_symbol",
        "fore_adjust_factor",
        "back_adjust_factor",
        "adjust_factor",
        "query_date",
        "source_method",
        "verification_status",
        "identity_mapping_status",
        "source",
    ]
    if raw.empty:
        return pd.DataFrame(columns=columns), pd.DataFrame()
    required = {"code", "dividOperateDate", "foreAdjustFactor", "backAdjustFactor"}
    missing = sorted(required - set(raw.columns))
    if missing:
        raise ValueError(f"baostock_factor_event_missing_fields:{missing}")
    factor_fields = [item for item in ("adjustFacto", "adjustFactor", "adjust_factor") if item in raw.columns]
    if not factor_fields:
        raise ValueError("baostock_factor_event_adjust_field_missing")
    if len(factor_fields) > 1:
        first = raw[factor_fields[0]].astype(str)
        if any(not first.equals(raw[item].astype(str)) for item in factor_fields[1:]):
            raise ValueError(f"baostock_factor_event_alias_conflict:{factor_fields}")
    event_dates = raw["dividOperateDate"].fillna("").astype(str).str.strip()
    bad = event_dates.ne(str(query_date))
    if bad.any():
        raise ValueError(f"baostock_factor_event_date_mismatch:{sorted(event_dates.loc[bad].unique())[:10]}")
    base = pd.DataFrame(
        {
            "trade_date": event_dates,
            "provider_symbol": raw["code"].map(normalize_symbol),
            "fore_adjust_factor": _numeric(raw, "foreAdjustFactor"),
            "back_adjust_factor": _numeric(raw, "backAdjustFactor"),
            "adjust_factor": _numeric(raw, factor_fields[0]),
        }
    )
    mapped = identity_registry.map_frame(base)
    mapped = mapped.rename(columns={"trade_date": "divid_operate_date"})
    mapped["query_date"] = str(query_date)
    mapped["source_method"] = "date_batch"
    mapped["verification_status"] = "batch_unreconciled"
    mapped["source"] = "baostock.query_daily_adjust_factor"
    mapped = mapped.loc[:, columns]
    events, conflicts = _deduplicate_mapped_rows(
        mapped,
        key=["security_id", "divid_operate_date"],
        compare_columns=["fore_adjust_factor", "back_adjust_factor", "adjust_factor"],
        conflict_type="identity_mapped_factor_event_conflict",
        prefer_symbol_on_date=True,
    )
    return events.sort_values(["divid_operate_date", "security_id"]).reset_index(drop=True), conflicts


def derive_symbol_adjust_factor_events(
    raw: pd.DataFrame,
    *,
    provider_symbol: str,
    identity_registry: SecurityIdentityRegistry,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    columns = [
        "security_id",
        "divid_operate_date",
        "symbol_on_date",
        "provider_symbol",
        "fore_adjust_factor",
        "back_adjust_factor",
        "adjust_factor",
        "query_date",
        "source_method",
        "verification_status",
        "identity_mapping_status",
        "source",
    ]
    if raw is None or raw.empty:
        return pd.DataFrame(columns=columns), pd.DataFrame()
    required = {"trade_date", "fore_adjust_factor", "back_adjust_factor", "adjust_factor"}
    missing = sorted(required - set(raw.columns))
    if missing:
        raise ValueError(f"baostock_symbol_factor_history_missing_fields:{missing}")
    base = pd.DataFrame(
        {
            "trade_date": raw["trade_date"].fillna("").astype(str).str.slice(0, 10),
            "provider_symbol": normalize_symbol(provider_symbol),
            "fore_adjust_factor": _numeric(raw, "fore_adjust_factor"),
            "back_adjust_factor": _numeric(raw, "back_adjust_factor"),
            "adjust_factor": _numeric(raw, "adjust_factor"),
        }
    )
    mapped = identity_registry.map_frame(base)
    mapped = mapped.rename(columns={"trade_date": "divid_operate_date"})
    mapped["query_date"] = ""
    mapped["source_method"] = "symbol_history"
    mapped["verification_status"] = "symbol_history_unreconciled"
    mapped["source"] = "baostock.query_adjust_factor"
    mapped = mapped.loc[:, columns]
    events, conflicts = _deduplicate_mapped_rows(
        mapped,
        key=["security_id", "divid_operate_date"],
        compare_columns=["fore_adjust_factor", "back_adjust_factor", "adjust_factor"],
        conflict_type="identity_mapped_symbol_factor_history_conflict",
        prefer_symbol_on_date=True,
    )
    return events.sort_values(["divid_operate_date", "security_id"]).reset_index(drop=True), conflicts


def reconcile_adjust_factor_events(
    batch_events: pd.DataFrame,
    symbol_events: pd.DataFrame,
    *,
    comparable_start: str,
    comparable_end: str,
    relative_tolerance: float = 1e-6,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Accept only identical event keys/values from the two BaoStock paths."""

    key = ["security_id", "divid_operate_date"]
    factor_columns = ["fore_adjust_factor", "back_adjust_factor", "adjust_factor"]
    canonical_columns = [
        "security_id",
        "divid_operate_date",
        "symbol_on_date",
        "provider_symbol",
        *factor_columns,
        "query_date",
        "source_method",
        "verification_status",
        "identity_mapping_status",
        "source",
    ]
    left = batch_events.copy()
    right = symbol_events.copy()
    for data in (left, right):
        for column in factor_columns:
            data[column] = pd.to_numeric(data[column], errors="coerce")
    merged = left.merge(right, on=key, how="outer", suffixes=("_batch", "_symbol"), indicator=True)
    accepted_rows: list[dict[str, Any]] = []
    disputes: list[dict[str, Any]] = []
    mismatch_count = 0
    for row in merged.to_dict("records"):
        presence = str(row.get("_merge", ""))
        event_date = str(row.get("divid_operate_date", ""))
        if presence == "both":
            close = True
            relative_errors: dict[str, float] = {}
            for column in factor_columns:
                batch_value = float(row.get(f"{column}_batch", np.nan))
                symbol_value = float(row.get(f"{column}_symbol", np.nan))
                denominator = max(abs(batch_value), abs(symbol_value), np.finfo(float).tiny)
                relative_error = abs(batch_value - symbol_value) / denominator
                relative_errors[column] = float(relative_error)
                close = close and np.isfinite(batch_value) and np.isfinite(symbol_value) and relative_error <= float(relative_tolerance)
            if close:
                accepted_rows.append(
                    {
                        "security_id": row["security_id"],
                        "divid_operate_date": event_date,
                        "symbol_on_date": row.get("symbol_on_date_batch", ""),
                        "provider_symbol": row.get("provider_symbol_batch", ""),
                        **{column: row.get(f"{column}_batch") for column in factor_columns},
                        "query_date": row.get("query_date_batch", event_date),
                        "source_method": "date_batch+symbol_history",
                        "verification_status": "verified_dual_path",
                        "identity_mapping_status": row.get("identity_mapping_status_batch", "mapped"),
                        "source": "baostock.query_daily_adjust_factor+query_adjust_factor",
                    }
                )
            else:
                mismatch_count += 1
                disputes.append({**row, "dispute_reason": "factor_values_mismatch", "relative_errors": relative_errors})
        else:
            if presence == "right_only" and event_date < str(comparable_start):
                reason = "pre_batch_baseline_single_source_unverified"
            elif event_date < str(comparable_start) or event_date > str(comparable_end):
                reason = "outside_batch_comparison_range_single_source_unverified"
            else:
                reason = "batch_only_event" if presence == "left_only" else "symbol_history_only_event"
            disputes.append({**row, "dispute_reason": reason})
    accepted = pd.DataFrame(accepted_rows, columns=canonical_columns)
    disputed = pd.DataFrame(disputes)
    metrics = {
        "batch_event_count": int(len(left)),
        "symbol_history_event_count": int(len(right)),
        "verified_event_count": int(len(accepted)),
        "disputed_event_count": int(len(disputed)),
        "value_mismatch_count": int(mismatch_count),
        "comparable_start": str(comparable_start),
        "comparable_end": str(comparable_end),
        "relative_tolerance": float(relative_tolerance),
    }
    return accepted.sort_values(key).reset_index(drop=True), disputed.reset_index(drop=True), metrics


def build_adjust_factor_daily(market_daily: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "security_id",
        "trade_date",
        "symbol_on_date",
        "fore_adjust_factor",
        "back_adjust_factor",
        "adjust_factor",
        "factor_event_date",
        "baseline_status",
        "source",
    ]
    if market_daily.empty:
        return pd.DataFrame(columns=columns)
    event_columns = ["divid_operate_date", "fore_adjust_factor", "back_adjust_factor", "adjust_factor"]
    rows: list[pd.DataFrame] = []
    event_groups = {key: group.copy() for key, group in events.groupby("security_id", sort=False)} if not events.empty else {}
    for security_id, market_group in market_daily.groupby("security_id", sort=False):
        market = market_group.loc[:, ["security_id", "trade_date", "symbol_on_date"]].sort_values("trade_date").copy()
        market["_trade_ts"] = pd.to_datetime(market["trade_date"], errors="raise")
        security_events = event_groups.get(security_id)
        if security_events is None or security_events.empty:
            for column in ("fore_adjust_factor", "back_adjust_factor", "adjust_factor"):
                market[column] = np.nan
            market["factor_event_date"] = ""
            market["baseline_status"] = "baseline_unproven"
            rows.append(market)
            continue
        event = security_events.loc[:, event_columns].sort_values("divid_operate_date").copy()
        event["_event_ts"] = pd.to_datetime(event["divid_operate_date"], errors="raise")
        merged = pd.merge_asof(
            market.sort_values("_trade_ts"),
            event.sort_values("_event_ts"),
            left_on="_trade_ts",
            right_on="_event_ts",
            direction="backward",
            allow_exact_matches=True,
        )
        merged["factor_event_date"] = merged["divid_operate_date"].fillna("").astype(str)
        merged["baseline_status"] = np.where(
            merged["factor_event_date"].eq(""),
            "baseline_unproven",
            np.where(
                merged["factor_event_date"].lt(str(market["trade_date"].min())),
                "proven_from_prior_event",
                "event_initialized",
            ),
        )
        rows.append(merged)
    out = pd.concat(rows, ignore_index=True)
    out["source"] = "qdp_v3_forward_fill_validated_events"
    return out.loc[:, columns].sort_values(["trade_date", "security_id"]).reset_index(drop=True)


def trading_calendar_from_snapshot_dates(dates: list[str], *, source: str) -> pd.DataFrame:
    normalized = sorted({str(item)[:10] for item in dates if str(item).strip()})
    return pd.DataFrame(
        {
            "trade_date": normalized,
            "is_open": [True] * len(normalized),
            "exchange": ["SSE/SZSE"] * len(normalized),
            "source": [str(source)] * len(normalized),
        }
    )


def build_eligible_signal_view(security_status: pd.DataFrame) -> pd.DataFrame:
    status = security_status.copy().sort_values(["security_id", "trade_date"])
    board = status["symbol_on_date"].map(board_for_symbol)
    listed = status["list_status"].fillna("").astype(str).str.lower().isin({"listed", "l", "1"})
    non_st = ~status["is_st"].fillna(False).astype(bool)
    status["is_eligible_signal"] = board.eq("MainBoard") & listed & non_st
    status["eligibility_reason"] = np.select(
        [board.ne("MainBoard"), ~listed, ~non_st],
        ["not_mainboard_on_date", "not_listed_on_date", "st_on_date"],
        default="eligible_mainboard_non_st",
    )
    status["source"] = "qdp_v3_pit_status_on_signal_date"
    return status.loc[
        :,
        ["security_id", "trade_date", "symbol_on_date", "is_eligible_signal", "eligibility_reason", "source"],
    ].reset_index(drop=True)


def build_tradable_open_view(market_daily: pd.DataFrame) -> pd.DataFrame:
    market = market_daily.copy().sort_values(["security_id", "trade_date"])
    grouped = market.groupby("security_id", sort=False)
    market["next_trade_date"] = grouped["trade_date"].shift(-1)
    market["next_symbol_on_date"] = grouped["symbol_on_date"].shift(-1)
    market["open_d1"] = pd.to_numeric(grouped["open"].shift(-1), errors="coerce")
    market["tradestatus_d1"] = grouped["tradestatus"].shift(-1).fillna("").astype(str)
    market["tradable_open_d1"] = market["tradestatus_d1"].eq("1") & market["open_d1"].gt(0)
    market["tradability_reason"] = np.select(
        [market["next_trade_date"].isna(), market["tradestatus_d1"].ne("1"), market["open_d1"].isna() | market["open_d1"].le(0)],
        ["next_observation_unavailable", "suspended_on_d1", "raw_open_d1_unavailable"],
        default="raw_open_d1_available",
    )
    market["next_trade_date"] = market["next_trade_date"].fillna("").astype(str)
    market["next_symbol_on_date"] = market["next_symbol_on_date"].fillna("").astype(str)
    market["source"] = "qdp_v3_raw_next_observation_label_only"
    return market.loc[
        :,
        [
            "security_id",
            "trade_date",
            "symbol_on_date",
            "next_trade_date",
            "next_symbol_on_date",
            "open_d1",
            "tradable_open_d1",
            "tradability_reason",
            "source",
        ],
    ].reset_index(drop=True)


def build_signal_and_open_pit_views(
    market_daily: pd.DataFrame,
    security_status: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    return build_eligible_signal_view(security_status), build_tradable_open_view(market_daily)
