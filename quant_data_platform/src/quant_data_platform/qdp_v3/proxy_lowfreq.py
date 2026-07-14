from __future__ import annotations

from typing import Any, Iterable

import numpy as np
import pandas as pd

from quant_data_platform.qdp_v3.constants import DOMAIN_MARKET_DAILY_RAW
from quant_data_platform.qdp_v3.identity import normalize_symbol
from quant_data_platform.qdp_v3.quality import QualityFinding
from quant_data_platform.qdp_v3.storage import RawPartitionRef, read_raw_partition


def _map_daily_identity(frame: pd.DataFrame, *, registry: Any, symbol_column: str, date_column: str) -> pd.DataFrame:
    data = frame.copy()
    data["provider_symbol"] = data[symbol_column].map(normalize_symbol)
    data["trade_date"] = _date_text(data[date_column])
    mapped = registry.map_frame(data, provider_symbol_column="provider_symbol", date_column="trade_date")
    return mapped


def audit_proxy_daily_against_baostock(
    *,
    proxy_refs: Iterable[RawPartitionRef],
    baostock_refs: Iterable[RawPartitionRef],
    registry: Any,
) -> tuple[list[QualityFinding], dict[str, Any], list[dict[str, Any]]]:
    """Cross-check daily bars without assigning either provider blanket priority."""

    proxy_by_date = {ref.partition_value: ref for ref in proxy_refs}
    bao_by_date = {ref.partition_value: ref for ref in baostock_refs}
    findings: list[QualityFinding] = []
    conflicts: list[dict[str, Any]] = []
    conflict_count = 0
    compared_rows = 0
    proxy_only_rows = 0
    bao_traded_missing_rows = 0
    unmapped_rows = 0
    for trade_date in sorted(set(proxy_by_date).intersection(bao_by_date)):
        proxy_raw = read_raw_partition(proxy_by_date[trade_date])
        bao_raw = read_raw_partition(bao_by_date[trade_date])
        if proxy_raw.empty or bao_raw.empty:
            continue
        if not {"ts_code", "trade_date"}.issubset(proxy_raw.columns) or not {"code", "date"}.issubset(bao_raw.columns):
            findings.append(
                QualityFinding(
                    code="proxy_daily_raw_schema_invalid",
                    severity="blocker",
                    message="Proxy or BaoStock daily raw lacks identity/date fields.",
                    domain=DOMAIN_MARKET_DAILY_RAW,
                    sample=[{"trade_date": trade_date}],
                )
            )
            continue
        proxy = _map_daily_identity(proxy_raw, registry=registry, symbol_column="ts_code", date_column="trade_date")
        bao = _map_daily_identity(bao_raw, registry=registry, symbol_column="code", date_column="date")
        unmapped_rows += int(proxy["identity_mapping_status"].ne("mapped").sum())
        proxy = proxy.loc[proxy["identity_mapping_status"].eq("mapped")].copy()
        bao = bao.loc[bao["identity_mapping_status"].eq("mapped")].copy()
        proxy = proxy.rename(columns={"pre_close": "preclose", "vol": "volume"})
        proxy["volume"] = pd.to_numeric(proxy.get("volume"), errors="coerce") * 100.0
        proxy["amount"] = pd.to_numeric(proxy.get("amount"), errors="coerce") * 1_000.0
        for column in ("open", "high", "low", "close", "preclose"):
            proxy[column] = pd.to_numeric(proxy.get(column), errors="coerce")
            bao[column] = pd.to_numeric(bao.get(column), errors="coerce")
        bao["volume"] = pd.to_numeric(bao.get("volume"), errors="coerce")
        bao["amount"] = pd.to_numeric(bao.get("amount"), errors="coerce")
        bao["tradestatus"] = bao.get("tradestatus", pd.Series("", index=bao.index)).astype(str)
        proxy_duplicate = proxy.duplicated(["security_id", "trade_date"], keep=False)
        bao_duplicate = bao.duplicated(["security_id", "trade_date"], keep=False)
        if proxy_duplicate.any() or bao_duplicate.any():
            findings.append(
                QualityFinding(
                    code="proxy_daily_identity_duplicate",
                    severity="blocker",
                    message="A provider exposes duplicate daily rows after stable-identity mapping.",
                    domain=DOMAIN_MARKET_DAILY_RAW,
                    count=int(proxy_duplicate.sum() + bao_duplicate.sum()),
                    sample=[{"trade_date": trade_date}],
                )
            )
            continue
        keys = ["security_id", "trade_date"]
        columns = [*keys, "open", "high", "low", "close", "preclose", "volume", "amount"]
        compared = bao.loc[bao["tradestatus"].eq("1"), columns].merge(
            proxy.loc[:, columns],
            on=keys,
            how="outer",
            suffixes=("_bao", "_proxy"),
            indicator=True,
        )
        proxy_only = compared["_merge"].eq("right_only")
        bao_missing = compared["_merge"].eq("left_only")
        proxy_only_rows += int(proxy_only.sum())
        bao_traded_missing_rows += int(bao_missing.sum())
        both = compared.loc[compared["_merge"].eq("both")].copy()
        compared_rows += int(len(both))
        bad = pd.Series(False, index=both.index)
        details: dict[int, list[str]] = {int(index): [] for index in both.index}
        for column in ("open", "high", "low", "close", "preclose"):
            left = both[f"{column}_bao"]
            right = both[f"{column}_proxy"]
            difference = (left - right).abs()
            mismatch = left.isna().ne(right.isna()) | difference.gt(1e-8)
            bad |= mismatch
            for index in both.index[mismatch]:
                details[int(index)].append(column)
        volume_difference = (both["volume_bao"] - both["volume_proxy"]).abs()
        volume_bad = both["volume_bao"].isna().ne(both["volume_proxy"].isna()) | volume_difference.gt(1.0)
        bad |= volume_bad
        amount_difference = (both["amount_bao"] - both["amount_proxy"]).abs()
        amount_limit = np.maximum(10.0, both["amount_bao"].abs() * 1e-6)
        amount_bad = both["amount_bao"].isna().ne(both["amount_proxy"].isna()) | amount_difference.gt(amount_limit)
        bad |= amount_bad
        for index in both.index[volume_bad]:
            details[int(index)].append("volume")
        for index in both.index[amount_bad]:
            details[int(index)].append("amount")
        conflict_count += int(bad.sum())
        for index, row in both.loc[bad].head(max(0, 100 - len(conflicts))).iterrows():
            conflicts.append(
                {
                    "security_id": str(row["security_id"]),
                    "trade_date": str(row["trade_date"]),
                    "conflicting_fields": details[int(index)],
                }
            )
    missing_proxy_dates = sorted(set(bao_by_date) - set(proxy_by_date))
    if missing_proxy_dates:
        findings.append(
            QualityFinding(
                code="proxy_daily_trade_dates_missing",
                severity="blocker",
                message="Proxy daily cross-source raw is missing for BaoStock snapshot dates.",
                domain=DOMAIN_MARKET_DAILY_RAW,
                count=len(missing_proxy_dates),
                sample=missing_proxy_dates[:20],
            )
        )
    if conflict_count:
        findings.append(
            QualityFinding(
                code="proxy_baostock_daily_value_conflicts",
                severity="blocker",
                message="Daily values differ beyond the explicit unit/rounding contract and require quarantine.",
                domain=DOMAIN_MARKET_DAILY_RAW,
                count=conflict_count,
                sample=conflicts[:20],
            )
        )
    if bao_traded_missing_rows:
        findings.append(
            QualityFinding(
                code="proxy_daily_traded_rows_missing",
                severity="blocker",
                message="BaoStock traded rows are absent from proxy daily history.",
                domain=DOMAIN_MARKET_DAILY_RAW,
                count=bao_traded_missing_rows,
            )
        )
    if proxy_only_rows:
        findings.append(
            QualityFinding(
                code="proxy_only_daily_bars_provisional",
                severity="warning",
                message="Daily bars present only in the proxy require independent repair before strict visibility.",
                domain=DOMAIN_MARKET_DAILY_RAW,
                count=proxy_only_rows,
                provisional=True,
            )
        )
    if unmapped_rows:
        findings.append(
            QualityFinding(
                code="proxy_daily_identity_unmapped",
                severity="blocker",
                message="Proxy daily rows cannot map to stable identities.",
                domain=DOMAIN_MARKET_DAILY_RAW,
                count=unmapped_rows,
            )
        )
    metrics = {
        "proxy_daily_partition_count": len(proxy_by_date),
        "baostock_daily_partition_count": len(bao_by_date),
        "compared_row_count": compared_rows,
        "value_conflict_count": conflict_count,
        "proxy_only_row_count": proxy_only_rows,
        "baostock_traded_missing_row_count": bao_traded_missing_rows,
        "unmapped_row_count": unmapped_rows,
    }
    return findings, metrics, conflicts


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


def audit_proxy_calendar(
    *,
    proxy_refs: Iterable[RawPartitionRef],
    canonical_calendar: pd.DataFrame,
) -> tuple[list[QualityFinding], dict[str, Any]]:
    findings: list[QualityFinding] = []
    frames = [read_raw_partition(ref) for ref in proxy_refs]
    raw = pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()
    if raw.empty:
        return findings, {"proxy_calendar_row_count": 0}
    if not {"cal_date", "is_open"}.issubset(raw.columns):
        return [
            QualityFinding(
                code="tushare_proxy_calendar_schema_invalid",
                severity="blocker",
                message="Proxy trade calendar lacks cal_date/is_open.",
                domain="trading_calendar",
            )
        ], {"proxy_calendar_row_count": int(len(raw))}
    proxy = pd.DataFrame(
        {
            "trade_date": _date_text(raw["cal_date"]),
            "proxy_is_open": raw["is_open"].astype(str).str.lower().isin({"1", "true", "t", "yes"}),
        }
    )
    inconsistent = proxy.groupby("trade_date")["proxy_is_open"].nunique().gt(1)
    if inconsistent.any():
        findings.append(
            QualityFinding(
                code="tushare_proxy_calendar_internal_conflict",
                severity="blocker",
                message="Proxy calendar reports conflicting open states for a date.",
                domain="trading_calendar",
                count=int(inconsistent.sum()),
                sample=inconsistent.loc[inconsistent].index.tolist()[:20],
            )
        )
    proxy = proxy.drop_duplicates("trade_date", keep="last")
    canonical = canonical_calendar.copy()
    canonical["trade_date"] = canonical["trade_date"].astype(str).str.slice(0, 10)
    canonical["canonical_is_open"] = canonical["is_open"].astype(str).str.lower().isin({"1", "true", "t", "yes"})
    canonical = canonical.groupby("trade_date", as_index=False)["canonical_is_open"].max()
    compared = canonical.merge(proxy, on="trade_date", how="left")
    mismatched = compared["proxy_is_open"].notna() & compared["canonical_is_open"].ne(compared["proxy_is_open"])
    missing_open = compared["canonical_is_open"] & compared["proxy_is_open"].isna()
    if mismatched.any():
        findings.append(
            QualityFinding(
                code="tushare_proxy_baostock_calendar_conflict",
                severity="blocker",
                message="Proxy and BaoStock disagree on trading-day state.",
                domain="trading_calendar",
                count=int(mismatched.sum()),
                sample=compared.loc[mismatched].head(20).to_dict("records"),
            )
        )
    if missing_open.any():
        findings.append(
            QualityFinding(
                code="tushare_proxy_calendar_open_dates_missing",
                severity="blocker",
                message="Proxy calendar omits one or more BaoStock open dates in the bootstrap range.",
                domain="trading_calendar",
                count=int(missing_open.sum()),
                sample=compared.loc[missing_open, "trade_date"].head(20).tolist(),
            )
        )
    return findings, {
        "proxy_calendar_row_count": int(len(proxy)),
        "compared_date_count": int(compared["proxy_is_open"].notna().sum()),
        "mismatch_count": int(mismatched.sum()),
        "missing_open_date_count": int(missing_open.sum()),
    }


def audit_proxy_daily_basic(
    *,
    proxy_refs: Iterable[RawPartitionRef],
    baostock_refs: Iterable[RawPartitionRef],
    registry: Any,
    relative_tolerance: float = 1e-5,
) -> tuple[list[QualityFinding], dict[str, Any], list[dict[str, Any]]]:
    proxy_by_date = {ref.partition_value: ref for ref in proxy_refs}
    bao_by_date = {ref.partition_value: ref for ref in baostock_refs}
    findings: list[QualityFinding] = []
    samples: list[dict[str, Any]] = []
    conflict_count = 0
    missing_count = 0
    internal_mv_conflicts = 0
    compared_count = 0
    mappings = {
        "turnover_rate": "turn",
        "pe_ttm": "peTTM",
        "pb": "pbMRQ",
        "ps_ttm": "psTTM",
    }
    for trade_date in sorted(set(proxy_by_date).intersection(bao_by_date)):
        raw_proxy = read_raw_partition(proxy_by_date[trade_date])
        raw_bao = read_raw_partition(bao_by_date[trade_date])
        if raw_proxy.empty:
            continue
        required = {"ts_code", "trade_date"}
        if not required.issubset(raw_proxy.columns):
            findings.append(
                QualityFinding(
                    code="tushare_proxy_daily_basic_schema_invalid",
                    severity="blocker",
                    message="Proxy daily_basic lacks ts_code/trade_date.",
                    domain="valuation_daily",
                    sample=[trade_date],
                )
            )
            continue
        proxy = _map_daily_identity(raw_proxy, registry=registry, symbol_column="ts_code", date_column="trade_date")
        bao = _map_daily_identity(raw_bao, registry=registry, symbol_column="code", date_column="date")
        proxy = proxy.loc[proxy["identity_mapping_status"].eq("mapped")].copy()
        bao = bao.loc[bao["identity_mapping_status"].eq("mapped")].copy()
        if proxy.duplicated(["security_id", "trade_date"]).any():
            findings.append(
                QualityFinding(
                    code="tushare_proxy_daily_basic_identity_duplicate",
                    severity="blocker",
                    message="daily_basic duplicates a stable identity/date.",
                    domain="valuation_daily",
                    sample=[trade_date],
                )
            )
            continue
        for proxy_column, bao_column in mappings.items():
            proxy[proxy_column] = pd.to_numeric(proxy.get(proxy_column), errors="coerce")
            bao[bao_column] = pd.to_numeric(bao.get(bao_column), errors="coerce")
        bao["tradestatus"] = bao.get("tradestatus", pd.Series("", index=bao.index)).astype(str)
        left = bao.loc[bao["tradestatus"].eq("1"), ["security_id", "trade_date", *mappings.values()]]
        right = proxy.loc[:, ["security_id", "trade_date", *mappings.keys()]]
        compared = left.merge(right, on=["security_id", "trade_date"], how="outer", indicator=True)
        missing_count += int(compared["_merge"].eq("left_only").sum())
        both = compared.loc[compared["_merge"].eq("both")]
        compared_count += int(len(both))
        row_bad = pd.Series(False, index=both.index)
        row_fields: dict[int, list[str]] = {int(index): [] for index in both.index}
        for proxy_column, bao_column in mappings.items():
            proxy_value = both[proxy_column]
            bao_value = both[bao_column]
            denominator = pd.concat([proxy_value.abs(), bao_value.abs()], axis=1).max(axis=1).clip(lower=1.0)
            bad = proxy_value.isna().ne(bao_value.isna()) | (proxy_value - bao_value).abs().div(denominator).gt(relative_tolerance)
            row_bad |= bad
            for index in both.index[bad]:
                row_fields[int(index)].append(proxy_column)
        conflict_count += int(row_bad.sum())
        for index, row in both.loc[row_bad].head(max(0, 100 - len(samples))).iterrows():
            samples.append(
                {
                    "security_id": str(row["security_id"]),
                    "trade_date": str(row["trade_date"]),
                    "conflicting_fields": row_fields[int(index)],
                }
            )
        for share_column, mv_column in (("total_share", "total_mv"), ("float_share", "circ_mv")):
            if {"close", share_column, mv_column}.issubset(proxy.columns):
                close = pd.to_numeric(proxy["close"], errors="coerce")
                shares = pd.to_numeric(proxy[share_column], errors="coerce")
                market_value = pd.to_numeric(proxy[mv_column], errors="coerce")
                denominator = market_value.abs().clip(lower=1.0)
                bad_mv = close.mul(shares).sub(market_value).abs().div(denominator).gt(0.002)
                internal_mv_conflicts += int(bad_mv.sum())
    missing_dates = sorted(set(bao_by_date) - set(proxy_by_date))
    if missing_dates:
        findings.append(
            QualityFinding(
                code="tushare_proxy_daily_basic_dates_missing",
                severity="blocker",
                message="daily_basic bootstrap partitions are missing for BaoStock dates.",
                domain="valuation_daily",
                count=len(missing_dates),
                sample=missing_dates[:20],
            )
        )
    if missing_count:
        findings.append(
            QualityFinding(
                code="tushare_proxy_daily_basic_traded_rows_missing",
                severity="blocker",
                message="daily_basic omits BaoStock traded identities.",
                domain="valuation_daily",
                count=missing_count,
            )
        )
    if conflict_count:
        findings.append(
            QualityFinding(
                code="tushare_proxy_baostock_valuation_conflicts",
                severity="blocker",
                message="Shared valuation fields differ beyond tolerance.",
                domain="valuation_daily",
                count=conflict_count,
                sample=samples[:20],
            )
        )
    if internal_mv_conflicts:
        findings.append(
            QualityFinding(
                code="tushare_proxy_daily_basic_market_cap_identity_failed",
                severity="blocker",
                message="price × share capital does not reconstruct reported market value within 0.2%.",
                domain="valuation_daily",
                count=internal_mv_conflicts,
            )
        )
    metrics = {
        "partition_count": len(proxy_by_date),
        "compared_row_count": compared_count,
        "missing_traded_row_count": missing_count,
        "value_conflict_count": conflict_count,
        "market_cap_identity_conflict_count": internal_mv_conflicts,
    }
    return findings, metrics, samples


def audit_proxy_suspend_status(
    *,
    proxy_refs: Iterable[RawPartitionRef],
    baostock_refs: Iterable[RawPartitionRef],
    registry: Any,
) -> tuple[list[QualityFinding], dict[str, Any], list[dict[str, Any]]]:
    bao_by_date = {ref.partition_value: ref for ref in baostock_refs}
    conflicts: list[dict[str, Any]] = []
    full_day_count = 0
    for ref in proxy_refs:
        raw = read_raw_partition(ref)
        if raw.empty or not {"ts_code", "trade_date"}.issubset(raw.columns):
            continue
        timing = raw.get("suspend_timing", pd.Series("", index=raw.index)).fillna("").astype(str).str.strip()
        suspend_type = raw.get("suspend_type", pd.Series("S", index=raw.index)).fillna("S").astype(str).str.upper()
        # A blank timing with suspension type S represents a whole-day halt;
        # partial intraday suspensions are evidence but do not contradict a
        # BaoStock daily traded status.
        full_day = suspend_type.str.startswith("S") & timing.eq("")
        if not full_day.any():
            continue
        mapped = _map_daily_identity(raw.loc[full_day], registry=registry, symbol_column="ts_code", date_column="trade_date")
        full_day_count += int(len(mapped))
        for trade_date, group in mapped.groupby("trade_date", sort=False):
            bao_ref = bao_by_date.get(str(trade_date))
            if bao_ref is None:
                conflicts.extend(
                    {"security_id": str(row.security_id), "trade_date": str(trade_date), "conflict_type": "baostock_status_date_missing"}
                    for row in group.itertuples(index=False)
                )
                continue
            bao = _map_daily_identity(read_raw_partition(bao_ref), registry=registry, symbol_column="code", date_column="date")
            statuses = dict(zip(bao["security_id"].astype(str), bao["tradestatus"].astype(str)))
            for row in group.itertuples(index=False):
                if str(row.identity_mapping_status) != "mapped" or statuses.get(str(row.security_id)) != "0":
                    conflicts.append(
                        {
                            "security_id": str(row.security_id),
                            "trade_date": str(trade_date),
                            "baostock_tradestatus": statuses.get(str(row.security_id), "missing"),
                            "conflict_type": "proxy_full_day_suspend_baostock_not_suspended",
                        }
                    )
    findings: list[QualityFinding] = []
    if conflicts:
        findings.append(
            QualityFinding(
                code="tushare_proxy_baostock_suspend_conflicts",
                severity="blocker",
                message="A proxy whole-day suspension conflicts with BaoStock daily trading status.",
                domain="security_status_daily",
                count=len(conflicts),
                sample=conflicts[:20],
            )
        )
    return findings, {"proxy_full_day_suspend_count": full_day_count, "conflict_count": len(conflicts)}, conflicts
