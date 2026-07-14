from __future__ import annotations

from bisect import bisect_right
from typing import Any, Iterable

import pandas as pd

from quant_data_platform.qdp_v3.constants import (
    DOMAIN_FINANCIAL_QUARTERLY,
    DOMAIN_PERFORMANCE_EXPRESS,
    DOMAIN_PERFORMANCE_FORECAST,
    QUALITY_PROVISIONAL,
)
from quant_data_platform.qdp_v3.identity import SecurityIdentityRegistry, normalize_symbol
from quant_data_platform.qdp_v3.manifest import stable_hash
from quant_data_platform.qdp_v3.storage import RawPartitionRef, read_raw_partition


API_DOMAIN = {
    "income": DOMAIN_FINANCIAL_QUARTERLY,
    "balancesheet": DOMAIN_FINANCIAL_QUARTERLY,
    "cashflow": DOMAIN_FINANCIAL_QUARTERLY,
    "fina_indicator": DOMAIN_FINANCIAL_QUARTERLY,
    "forecast": DOMAIN_PERFORMANCE_FORECAST,
    "express": DOMAIN_PERFORMANCE_EXPRESS,
}


def _date_text(series: pd.Series) -> pd.Series:
    text = series.fillna("").astype(str).str.strip().str.slice(0, 10)
    compact = text.str.fullmatch(r"\d{8}")
    text.loc[compact] = (
        text.loc[compact].str.slice(0, 4)
        + "-"
        + text.loc[compact].str.slice(4, 6)
        + "-"
        + text.loc[compact].str.slice(6, 8)
    )
    return text


def _next_open(date: str, open_dates: list[str]) -> str:
    if not date:
        return ""
    index = bisect_right(open_dates, str(date))
    return open_dates[index] if index < len(open_dates) else ""


def _api_name(ref: RawPartitionRef) -> str:
    value = str(ref.partition_value or "")
    return value.split("__", 1)[0].strip().lower()


def canonicalize_proxy_financial(
    refs: Iterable[RawPartitionRef],
    *,
    identity_registry: SecurityIdentityRegistry,
    calendar: pd.DataFrame,
) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame], dict[str, list[RawPartitionRef]]]:
    """Build source-prefixed, revision-preserving PIT disclosure events."""

    frames_by_domain: dict[str, list[pd.DataFrame]] = {domain: [] for domain in set(API_DOMAIN.values())}
    conflicts_by_domain: dict[str, list[pd.DataFrame]] = {domain: [] for domain in set(API_DOMAIN.values())}
    refs_by_domain: dict[str, list[RawPartitionRef]] = {domain: [] for domain in set(API_DOMAIN.values())}
    open_dates = sorted(
        set(
            calendar.loc[
                calendar["is_open"].astype(str).str.lower().isin({"1", "true", "t", "yes"}),
                "trade_date",
            ].astype(str).str.slice(0, 10)
        )
    ) if calendar is not None and not calendar.empty else []
    for ref in refs:
        api_name = _api_name(ref)
        domain = API_DOMAIN.get(api_name)
        if domain is None:
            continue
        refs_by_domain[domain].append(ref)
        raw = read_raw_partition(ref)
        if raw.empty:
            continue
        required = {"ts_code", "end_date"}
        if not required.issubset(raw.columns):
            conflicts_by_domain[domain].append(
                pd.DataFrame(
                    [
                        {
                            "partition_value": ref.partition_value,
                            "api_name": api_name,
                            "conflict_type": "tushare_proxy_financial_schema_invalid",
                            "missing": sorted(required - set(raw.columns)),
                        }
                    ]
                )
            )
            continue
        work = raw.copy()
        report_date = _date_text(work["end_date"])
        ann = _date_text(work.get("ann_date", pd.Series("", index=work.index)))
        final_ann = _date_text(work.get("f_ann_date", pd.Series("", index=work.index)))
        publish_date = final_ann.where(final_ann.ne(""), ann)
        base = pd.DataFrame(
            {
                "report_date": report_date,
                "provider_symbol": work["ts_code"].map(normalize_symbol),
            }
        )
        mapped = identity_registry.map_frame(
            base,
            provider_symbol_column="provider_symbol",
            date_column="report_date",
        )
        mapped["publish_date"] = publish_date
        mapped["publish_date_source"] = final_ann.ne("").map(
            {True: "f_ann_date", False: "ann_date"}
        )
        mapped["available_date"] = publish_date.map(lambda value: _next_open(value, open_dates))
        mapped["availability_status"] = "publish_date_unproven"
        known = publish_date.ne("") & mapped["available_date"].ne("")
        mapped.loc[known, "availability_status"] = "next_trading_day_after_publish_date"
        mapped.loc[publish_date.ne("") & mapped["available_date"].eq(""), "availability_status"] = "calendar_horizon_missing"
        mapped["quality_tier"] = QUALITY_PROVISIONAL
        invalid = mapped["identity_mapping_status"].ne("mapped") | mapped["report_date"].eq("")
        if invalid.any():
            bad = mapped.loc[invalid].copy()
            bad["api_name"] = api_name
            bad["conflict_type"] = "tushare_proxy_financial_identity_or_report_date_invalid"
            conflicts_by_domain[domain].append(bad)
        mapped = mapped.loc[~invalid].copy()
        if mapped.empty:
            continue
        identity_columns = {"ts_code", "end_date", "ann_date", "f_ann_date"}
        payload_columns = [column for column in work.columns if column not in identity_columns]
        for column in payload_columns:
            mapped[f"tushare_proxy_{api_name}__{column}"] = work.loc[mapped.index, column]
        # Preserve every disclosed revision.  Source remains human-readable;
        # its suffix is a content-stable record id so equal duplicates collapse
        # while distinct same-day revisions remain separate PIT events.
        record_payload_columns = [column for column in mapped.columns if column.startswith(f"tushare_proxy_{api_name}__")]
        record_ids = []
        for index, row in mapped.iterrows():
            payload = {
                "security_id": str(row["security_id"]),
                "report_date": str(row["report_date"]),
                "publish_date": str(row["publish_date"]),
                **{column: None if pd.isna(row[column]) else str(row[column]) for column in record_payload_columns},
            }
            record_ids.append(stable_hash(payload, length=16))
        mapped["source"] = [f"tushare_proxy.{api_name}:{record_id}" for record_id in record_ids]
        prefix = [
            "security_id",
            "report_date",
            "symbol_on_date",
            "provider_symbol",
            "publish_date",
            "publish_date_source",
            "available_date",
            "availability_status",
            "identity_mapping_status",
            "quality_tier",
            "source",
        ]
        mapped = mapped.loc[:, [*prefix, *record_payload_columns]]
        mapped = mapped.drop_duplicates(
            ["security_id", "report_date", "publish_date", "source"], keep="last"
        )
        frames_by_domain[domain].append(mapped)

    frames = {
        domain: (
            pd.concat(items, ignore_index=True, sort=False).sort_values(
                ["report_date", "security_id", "publish_date", "source"], kind="mergesort"
            ).reset_index(drop=True)
            if items
            else pd.DataFrame()
        )
        for domain, items in frames_by_domain.items()
    }
    conflicts = {
        domain: (
            pd.concat(items, ignore_index=True, sort=False).reset_index(drop=True)
            if items
            else pd.DataFrame()
        )
        for domain, items in conflicts_by_domain.items()
    }
    return frames, conflicts, refs_by_domain


def canonicalize_proxy_daily_basic_capital(
    raw: pd.DataFrame,
    *,
    query_date: str,
    identity_registry: SecurityIdentityRegistry,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    columns = [
        "security_id",
        "trade_date",
        "symbol_on_date",
        "provider_symbol",
        "total_share",
        "float_share",
        "free_share",
        "restricted_share",
        "reported_total_mv",
        "reported_circ_mv",
        "baseline_status",
        "identity_mapping_status",
        "source",
    ]
    if raw is None or raw.empty:
        return pd.DataFrame(columns=columns), pd.DataFrame()
    required = {"ts_code", "trade_date", "total_share", "float_share"}
    if not required.issubset(raw.columns):
        return pd.DataFrame(columns=columns), pd.DataFrame(
            [
                {
                    "trade_date": str(query_date),
                    "conflict_type": "tushare_proxy_daily_basic_capital_schema_invalid",
                    "missing": sorted(required - set(raw.columns)),
                }
            ]
        )
    base = pd.DataFrame(
        {
            "trade_date": _date_text(raw["trade_date"]),
            "provider_symbol": raw["ts_code"].map(normalize_symbol),
        }
    )
    mapped = identity_registry.map_frame(
        base,
        provider_symbol_column="provider_symbol",
        date_column="trade_date",
    )
    # Tushare daily_basic share and market-value fields are reported in ten
    # thousand shares / ten thousand CNY respectively.
    mapped["total_share"] = pd.to_numeric(raw["total_share"], errors="coerce") * 10_000.0
    mapped["float_share"] = pd.to_numeric(raw["float_share"], errors="coerce") * 10_000.0
    mapped["free_share"] = pd.to_numeric(raw.get("free_share"), errors="coerce") * 10_000.0
    mapped["restricted_share"] = mapped["total_share"] - mapped["float_share"]
    mapped["reported_total_mv"] = pd.to_numeric(raw.get("total_mv"), errors="coerce") * 10_000.0
    mapped["reported_circ_mv"] = pd.to_numeric(raw.get("circ_mv"), errors="coerce") * 10_000.0
    mapped["baseline_status"] = "daily_observed"
    mapped["source"] = "tushare_proxy.daily_basic"
    invalid = (
        mapped["identity_mapping_status"].ne("mapped")
        | ~mapped["total_share"].gt(0)
        | ~mapped["float_share"].gt(0)
        | mapped["float_share"].gt(mapped["total_share"])
        | (mapped["free_share"].notna() & (mapped["free_share"].lt(0) | mapped["free_share"].gt(mapped["float_share"])))
    )
    conflicts = mapped.loc[invalid].copy()
    if not conflicts.empty:
        conflicts["conflict_type"] = "tushare_proxy_daily_basic_capital_invalid"
    accepted = mapped.loc[~invalid, columns].copy()
    duplicate = accepted.duplicated(["security_id", "trade_date"], keep=False)
    if duplicate.any():
        duplicated = accepted.loc[duplicate].copy()
        duplicated["conflict_type"] = "tushare_proxy_daily_basic_capital_identity_duplicate"
        conflicts = pd.concat([conflicts, duplicated], ignore_index=True, sort=False)
        accepted = accepted.loc[~duplicate]
    return accepted.sort_values(["security_id", "trade_date"], kind="mergesort").reset_index(drop=True), conflicts.reset_index(drop=True)
