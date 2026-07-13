from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd

from quant_data_platform.qdp_v3.constants import (
    BAOSTOCK_DAILY_FIELDS,
    EXPECTED_5M_BAR_ENDS,
    QUALITY_PROVISIONAL,
    QUALITY_QUARANTINED,
    QUALITY_STRICT,
)
from quant_data_platform.qdp_v3.identity import normalize_symbol


@dataclass(frozen=True)
class QualityFinding:
    code: str
    severity: str
    message: str
    domain: str = ""
    count: int = 0
    sample: list[Any] = field(default_factory=list)
    provisional: bool = False

    @property
    def blocks_release(self) -> bool:
        return self.severity == "blocker"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class QualityReport:
    domain: str
    quality_tier: str
    findings: list[QualityFinding]
    metrics: dict[str, Any] = field(default_factory=dict)

    @property
    def blockers(self) -> list[QualityFinding]:
        return [item for item in self.findings if item.blocks_release]

    def to_dict(self) -> dict[str, Any]:
        return {
            "domain": self.domain,
            "quality_tier": self.quality_tier,
            "findings": [item.to_dict() for item in self.findings],
            "blockers": [item.to_dict() for item in self.blockers],
            "metrics": dict(self.metrics),
        }


def report_for(domain: str, findings: Iterable[QualityFinding], *, metrics: dict[str, Any] | None = None) -> QualityReport:
    items = list(findings)
    if any(item.blocks_release for item in items):
        tier = QUALITY_QUARANTINED
    elif any(item.provisional for item in items):
        tier = QUALITY_PROVISIONAL
    else:
        tier = QUALITY_STRICT
    return QualityReport(domain=domain, quality_tier=tier, findings=items, metrics=dict(metrics or {}))


def _finding(
    code: str,
    message: str,
    *,
    domain: str,
    severity: str = "blocker",
    count: int = 0,
    sample: Iterable[Any] = (),
    provisional: bool = False,
) -> QualityFinding:
    return QualityFinding(
        code=code,
        severity=severity,
        message=message,
        domain=domain,
        count=int(count),
        sample=list(sample)[:20],
        provisional=bool(provisional),
    )


def _clean_date(value: Any) -> str:
    if value is None or (not isinstance(value, (list, tuple, dict, set)) and pd.isna(value)):
        return ""
    text = str(value).strip()[:10]
    return "" if text.lower() in {"", "nan", "nat", "none"} else text


def _is_true(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    return str(value or "").strip().lower() in {"1", "true", "t", "yes"}


def audit_baostock_daily_raw(
    frame: pd.DataFrame,
    *,
    query_date: str,
    raw_domain: str = "baostock_daily_astock_raw",
    expected_codes: Iterable[str] | None = None,
    expected_code_evidence: pd.DataFrame | None = None,
    security_master: pd.DataFrame | None = None,
    neighbor_row_count: int | None = None,
    neighbor_expected_codes: Iterable[str] | None = None,
    neighbor_date: str = "",
    count_change_explained: bool = False,
    identity_security_by_symbol: Mapping[str, str] | None = None,
) -> QualityReport:
    domain = str(raw_domain or "baostock_daily_astock_raw")
    findings: list[QualityFinding] = []
    columns = [str(item) for item in frame.columns]
    missing_fields = [item for item in BAOSTOCK_DAILY_FIELDS if item not in columns]
    unexpected_fields = [item for item in columns if item not in BAOSTOCK_DAILY_FIELDS]
    if missing_fields:
        findings.append(_finding("daily_raw_fields_missing", f"Missing BaoStock daily fields: {missing_fields}", domain=domain, count=len(missing_fields), sample=missing_fields))
    if unexpected_fields:
        findings.append(_finding("daily_raw_fields_unexpected", f"Unexpected BaoStock daily fields: {unexpected_fields}", domain=domain, severity="warning", count=len(unexpected_fields), sample=unexpected_fields, provisional=True))
    row_count = int(len(frame))
    if row_count <= 0:
        findings.append(_finding("daily_raw_empty", "A trading-day A-share snapshot must contain rows.", domain=domain))
    if row_count >= 20_000:
        findings.append(_finding("daily_raw_potential_truncation", "Bulk response reached its 20,000 row capacity.", domain=domain, count=row_count))
    if "date" in frame.columns:
        dates = frame["date"].fillna("").astype(str).str.strip()
        bad = dates.ne(str(query_date))
        if bad.any():
            findings.append(_finding("daily_raw_date_mismatch", "Returned dates differ from query date.", domain=domain, count=int(bad.sum()), sample=dates.loc[bad].unique()))
    if {"date", "code"}.issubset(frame.columns):
        duplicate = frame.duplicated(["date", "code"], keep=False)
        if duplicate.any():
            sample = frame.loc[duplicate, ["date", "code"]].head(20).to_dict("records")
            findings.append(_finding("daily_raw_duplicate_key", "(date, code) is not unique.", domain=domain, count=int(duplicate.sum()), sample=sample))
    if "adjustflag" in frame.columns:
        flags = frame["adjustflag"].fillna("").astype(str).str.strip()
        bad = flags.ne("3")
        if bad.any():
            findings.append(_finding("daily_raw_not_unadjusted", "All batch daily rows must have adjustflag=3.", domain=domain, count=int(bad.sum()), sample=flags.loc[bad].unique()))
    batch_codes: set[str] = set()
    if "code" in frame.columns:
        batch_codes = {normalize_symbol(item) for item in frame["code"] if normalize_symbol(item)}
        invalid = [item for item in batch_codes if not item.endswith((".SH", ".SZ", ".BJ"))]
        if invalid:
            findings.append(_finding("daily_raw_invalid_code", "Snapshot includes invalid security codes.", domain=domain, count=len(invalid), sample=invalid))
    missing_codes: list[str] = []
    extra_codes: list[str] = []
    identity_restatements: list[dict[str, str]] = []
    gap_classifications: dict[str, list[str]] = {}
    master_rows: dict[str, dict[str, Any]] = {}
    evidence_rows: dict[str, dict[str, Any]] = {}
    if isinstance(security_master, pd.DataFrame) and not security_master.empty:
        master_symbol_column = (
            "symbol"
            if "symbol" in security_master.columns
            else "provider_symbol"
            if "provider_symbol" in security_master.columns
            else ""
        )
        if master_symbol_column:
            for row in security_master.to_dict("records"):
                symbol = normalize_symbol(row.get(master_symbol_column, ""))
                if symbol:
                    master_rows[symbol] = row
    if isinstance(expected_code_evidence, pd.DataFrame) and not expected_code_evidence.empty:
        evidence_symbol_column = (
            "symbol"
            if "symbol" in expected_code_evidence.columns
            else "provider_symbol"
            if "provider_symbol" in expected_code_evidence.columns
            else ""
        )
        if evidence_symbol_column:
            for row in expected_code_evidence.to_dict("records"):
                symbol = normalize_symbol(row.get(evidence_symbol_column, ""))
                if symbol:
                    evidence_rows[symbol] = row
    def active_master_codes(at_date: str) -> set[str]:
        active: set[str] = set()
        for symbol, row in master_rows.items():
            list_date = _clean_date(row.get("list_date", ""))
            delist_date = _clean_date(row.get("delist_date", ""))
            if list_date and str(at_date) < list_date:
                continue
            # BaoStock outDate is the first date on which the old listing/code
            # is no longer effective (for example 300114 -> 302132).
            if delist_date and str(at_date) >= delist_date:
                continue
            active.add(symbol)
        return active
    query_all_codes: set[str] = set()
    master_active_codes = active_master_codes(str(query_date))
    expected: set[str] = set()
    if expected_codes is None:
        findings.append(
            _finding(
                "daily_raw_code_set_crosscheck_missing",
                "query_all_stock/security-master code-set evidence was not supplied.",
                domain=domain,
                severity="warning",
                provisional=True,
            )
        )
    else:
        query_all_codes = {normalize_symbol(item) for item in expected_codes if normalize_symbol(item)}
        # query_all_stock omits some valid types (for example CDR 689009) and
        # can lag official code changes.  The audit contract is therefore the
        # union of its date snapshot and stock-basic rows active on that date.
        expected = query_all_codes | master_active_codes
        missing_codes = sorted(expected - batch_codes)
        extra_codes = sorted(batch_codes - expected)
        identity_map = {normalize_symbol(key): str(value) for key, value in dict(identity_security_by_symbol or {}).items() if normalize_symbol(key) and str(value)}
        missing_by_identity = {identity_map.get(symbol, ""): symbol for symbol in missing_codes if identity_map.get(symbol, "")}
        extra_by_identity = {identity_map.get(symbol, ""): symbol for symbol in extra_codes if identity_map.get(symbol, "")}
        paired_ids = sorted(set(missing_by_identity) & set(extra_by_identity))
        identity_restatements = [
            {"security_id": security_id, "expected_symbol": missing_by_identity[security_id], "provider_symbol": extra_by_identity[security_id]}
            for security_id in paired_ids
        ]
        unresolved_missing = [symbol for symbol in missing_codes if identity_map.get(symbol, "") not in paired_ids]
        unresolved_extra = [symbol for symbol in extra_codes if identity_map.get(symbol, "") not in paired_ids]
        for symbol in unresolved_missing:
            master_row = master_rows.get(symbol, {})
            evidence_row = evidence_rows.get(symbol, {})
            list_date = _clean_date(master_row.get("list_date", ""))
            delist_date = _clean_date(master_row.get("delist_date", ""))
            trade_status = str(evidence_row.get("trade_status", evidence_row.get("tradeStatus", "")) or "").strip()
            if not identity_map.get(symbol, ""):
                category = "identity_mapping_problem"
            elif list_date and str(query_date) < list_date:
                category = "not_yet_listed"
            elif delist_date and str(query_date) >= delist_date:
                category = "already_delisted"
            elif _is_true(evidence_row.get("is_suspended", False)) or trade_status == "0":
                category = "suspended"
            else:
                category = "provider_gap"
            gap_classifications.setdefault(category, []).append(symbol)
        for category in ("not_yet_listed", "already_delisted", "suspended"):
            symbols = gap_classifications.get(category, [])
            if symbols:
                findings.append(
                    _finding(
                        f"daily_raw_code_gap_{category}",
                        f"Missing query_all_stock codes were classified as {category} from lifecycle/status evidence.",
                        domain=domain,
                        severity="warning",
                        count=len(symbols),
                        sample=symbols,
                    )
                )
        for category in ("identity_mapping_problem", "provider_gap"):
            symbols = gap_classifications.get(category, [])
            if symbols:
                findings.append(
                    _finding(
                        f"daily_raw_code_gap_{category}",
                        f"Missing query_all_stock codes remain blocked as {category}.",
                        domain=domain,
                        count=len(symbols),
                        sample=symbols,
                    )
                )
        if unresolved_extra:
            findings.append(_finding("daily_raw_code_set_extra", "Batch codes are absent from query_all_stock/security master.", domain=domain, count=len(unresolved_extra), sample=unresolved_extra))
        if identity_restatements:
            findings.append(
                _finding(
                    "daily_raw_official_identity_restatement",
                    "Provider-current symbols replaced historical symbols and were reconciled through official symbol history.",
                    domain=domain,
                    severity="warning",
                    count=len(identity_restatements),
                    sample=identity_restatements,
                )
            )
    neighbor_change_is_explained = bool(count_change_explained)
    if neighbor_expected_codes is not None and expected_codes is not None:
        current_expected = expected
        previous_expected = {
            normalize_symbol(item) for item in neighbor_expected_codes if normalize_symbol(item)
        } | active_master_codes(str(neighbor_date))
        added = set(current_expected - previous_expected)
        removed = set(previous_expected - current_expected)
        identity_map = {
            normalize_symbol(key): str(value)
            for key, value in dict(identity_security_by_symbol or {}).items()
            if normalize_symbol(key) and str(value)
        }
        added_by_identity = {identity_map.get(symbol, ""): symbol for symbol in added if identity_map.get(symbol, "")}
        removed_by_identity = {identity_map.get(symbol, ""): symbol for symbol in removed if identity_map.get(symbol, "")}
        changed_ids = set(added_by_identity) & set(removed_by_identity)
        added = {symbol for symbol in added if identity_map.get(symbol, "") not in changed_ids}
        removed = {symbol for symbol in removed if identity_map.get(symbol, "") not in changed_ids}
        lifecycle_added = {
            symbol
            for symbol in added
            if _clean_date(master_rows.get(symbol, {}).get("list_date", ""))
            and (not neighbor_date or _clean_date(master_rows[symbol].get("list_date", "")) > str(neighbor_date))
            and _clean_date(master_rows[symbol].get("list_date", "")) <= str(query_date)
        }
        lifecycle_removed = {
            symbol
            for symbol in removed
            if _clean_date(master_rows.get(symbol, {}).get("delist_date", ""))
            and (not neighbor_date or _clean_date(master_rows[symbol].get("delist_date", "")) > str(neighbor_date))
            and _clean_date(master_rows[symbol].get("delist_date", "")) <= str(query_date)
        }
        expected_count_delta = len(current_expected) - len(previous_expected)
        observed_count_delta = row_count - int(neighbor_row_count) if neighbor_row_count is not None else expected_count_delta
        neighbor_change_is_explained = (
            observed_count_delta == expected_count_delta
            and not (added - lifecycle_added)
            and not (removed - lifecycle_removed)
        )
        if changed_ids:
            gap_classifications["official_code_change"] = sorted(changed_ids)
        if lifecycle_added:
            gap_classifications["new_listing"] = sorted(lifecycle_added)
        if lifecycle_removed:
            gap_classifications["delisting"] = sorted(lifecycle_removed)
    if neighbor_row_count is not None:
        delta = abs(row_count - int(neighbor_row_count))
        threshold = max(20, int(np.ceil(max(row_count, int(neighbor_row_count)) * 0.01)))
        if delta > threshold and not neighbor_change_is_explained:
            findings.append(
                _finding(
                    "daily_raw_unexplained_neighbor_count_jump",
                    f"Adjacent snapshot count changed by {delta}, above threshold {threshold}.",
                    domain=domain,
                    count=delta,
                )
            )
    return report_for(
        domain,
        findings,
        metrics={
            "row_count": row_count,
            "unique_code_count": len(batch_codes),
            "query_all_code_count": len(query_all_codes),
            "security_master_active_code_count": len(master_active_codes),
            "expected_code_count": len(expected),
            "missing_code_count": len(missing_codes),
            "extra_code_count": len(extra_codes),
            "identity_restatement_count": len(identity_restatements),
            "code_gap_classification_counts": {key: len(value) for key, value in sorted(gap_classifications.items())},
            "neighbor_count_change_explained": bool(neighbor_change_is_explained),
            "query_date": str(query_date),
        },
    )


def audit_baostock_factor_event_raw(frame: pd.DataFrame, *, query_date: str) -> QualityReport:
    domain = "baostock_adjust_factor_event_raw"
    findings: list[QualityFinding] = []
    if len(frame) >= 20_000:
        findings.append(_finding("factor_event_potential_truncation", "Factor event response reached bulk capacity.", domain=domain, count=len(frame)))
    date_field = "dividOperateDate" if "dividOperateDate" in frame.columns else "divid_operate_date" if "divid_operate_date" in frame.columns else ""
    if not date_field and not frame.empty:
        findings.append(_finding("factor_event_date_field_missing", "Factor event date field is missing.", domain=domain))
    if date_field:
        dates = frame[date_field].fillna("").astype(str).str.strip()
        bad = dates.ne(str(query_date))
        if bad.any():
            findings.append(_finding("factor_event_query_date_mismatch", "dividOperateDate must equal query date.", domain=domain, count=int(bad.sum()), sample=dates.loc[bad].unique()))
    factor_fields = [item for item in ("adjustFacto", "adjustFactor", "adjust_factor") if item in frame.columns]
    if not factor_fields and not frame.empty:
        findings.append(_finding("factor_event_adjust_field_missing", "No accepted explicit adjustment-factor field exists.", domain=domain))
    if {"code", date_field}.issubset(frame.columns) if date_field else False:
        duplicate = frame.duplicated(["code", date_field], keep=False)
        if duplicate.any():
            findings.append(_finding("factor_event_duplicate_raw_key", "(code, event date) is not unique.", domain=domain, count=int(duplicate.sum())))
    return report_for(domain, findings, metrics={"row_count": int(len(frame)), "query_date": str(query_date), "zero_event_day": bool(frame.empty)})


def audit_canonical_daily(frame: pd.DataFrame) -> QualityReport:
    domain = "market_daily_raw"
    findings: list[QualityFinding] = []
    required = {"security_id", "trade_date", "symbol_on_date", "provider_symbol", "open", "high", "low", "close", "volume", "amount"}
    missing = sorted(required - set(frame.columns))
    if missing:
        findings.append(_finding("canonical_daily_fields_missing", f"Missing fields: {missing}", domain=domain, count=len(missing), sample=missing))
        return report_for(domain, findings, metrics={"row_count": int(len(frame))})
    null_key = frame[["security_id", "trade_date"]].isna().any(axis=1) | frame["security_id"].astype(str).eq("") | frame["trade_date"].astype(str).eq("")
    if null_key.any():
        findings.append(_finding("canonical_daily_null_key", "Canonical daily primary key is null/blank.", domain=domain, count=int(null_key.sum())))
    duplicate = frame.duplicated(["security_id", "trade_date"], keep=False)
    if duplicate.any():
        findings.append(_finding("canonical_daily_duplicate_key", "(security_id, trade_date) is not unique.", domain=domain, count=int(duplicate.sum()), sample=frame.loc[duplicate, ["security_id", "trade_date"]].head(20).to_dict("records")))
    if "identity_mapping_status" in frame.columns:
        unresolved = frame["identity_mapping_status"].ne("mapped")
        if unresolved.any():
            findings.append(_finding("canonical_daily_identity_unresolved", "Some rows lack a date-local symbol mapping.", domain=domain, count=int(unresolved.sum()), sample=frame.loc[unresolved, ["provider_symbol", "trade_date", "identity_mapping_status"]].head(20).to_dict("records")))
    numeric = frame[["open", "high", "low", "close", "volume", "amount"]].apply(pd.to_numeric, errors="coerce")
    has_price = numeric[["open", "high", "low", "close"]].notna().any(axis=1)
    partial_price = has_price & numeric[["open", "high", "low", "close"]].isna().any(axis=1)
    if partial_price.any():
        findings.append(_finding("canonical_daily_partial_ohlc", "OHLC is partially null.", domain=domain, count=int(partial_price.sum())))
    complete = numeric[["open", "high", "low", "close"]].notna().all(axis=1)
    invalid_nonpositive = complete & numeric[["open", "high", "low", "close"]].le(0).any(axis=1)
    invalid_relation = complete & (
        numeric["high"].lt(numeric[["open", "close", "low"]].max(axis=1))
        | numeric["low"].gt(numeric[["open", "close", "high"]].min(axis=1))
    )
    if invalid_nonpositive.any():
        findings.append(_finding("canonical_daily_nonpositive_price", "Trading prices must be positive.", domain=domain, count=int(invalid_nonpositive.sum())))
    if invalid_relation.any():
        findings.append(_finding("canonical_daily_invalid_ohlc", "OHLC ordering is invalid.", domain=domain, count=int(invalid_relation.sum())))
    negative_flow = numeric[["volume", "amount"]].lt(0).any(axis=1)
    if negative_flow.any():
        findings.append(_finding("canonical_daily_negative_flow", "Volume/amount cannot be negative.", domain=domain, count=int(negative_flow.sum())))
    suspended = frame.get("tradestatus", pd.Series("1", index=frame.index)).astype(str).ne("1")
    zero_suspended = suspended & numeric[["open", "high", "low", "close"]].eq(0).any(axis=1)
    if zero_suspended.any():
        findings.append(_finding("canonical_daily_suspension_zero_fill", "Suspended rows must not be manufactured as zero-price bars.", domain=domain, count=int(zero_suspended.sum())))
    return report_for(domain, findings, metrics={"row_count": int(len(frame)), "priced_row_count": int(complete.sum()), "suspended_row_count": int(suspended.sum())})


def audit_table_contract(frame: pd.DataFrame, *, domain: str, primary_key: list[str]) -> QualityReport:
    findings: list[QualityFinding] = []
    missing = sorted(set(primary_key) - set(frame.columns))
    if missing:
        findings.append(_finding("table_primary_key_fields_missing", f"Missing primary-key fields: {missing}", domain=domain, count=len(missing), sample=missing))
        return report_for(domain, findings, metrics={"row_count": int(len(frame))})
    if frame.empty:
        findings.append(_finding("table_empty", "Canonical table is empty.", domain=domain))
        return report_for(domain, findings, metrics={"row_count": 0})
    keys = frame.loc[:, primary_key]
    blank = keys.isna().any(axis=1)
    for column in primary_key:
        if pd.api.types.is_object_dtype(keys[column]) or pd.api.types.is_string_dtype(keys[column]):
            blank |= keys[column].astype(str).str.strip().eq("")
    if blank.any():
        findings.append(_finding("table_primary_key_null", "Primary key contains null/blank values.", domain=domain, count=int(blank.sum())))
    duplicate = frame.duplicated(primary_key, keep=False)
    if duplicate.any():
        findings.append(_finding("table_primary_key_duplicate", "Primary key is not unique.", domain=domain, count=int(duplicate.sum()), sample=frame.loc[duplicate, primary_key].head(20).to_dict("records")))
    if "identity_mapping_status" in frame.columns:
        unresolved = frame["identity_mapping_status"].ne("mapped")
        if unresolved.any():
            findings.append(_finding("table_identity_unresolved", "Rows lack stable identity/PIT symbol mapping.", domain=domain, count=int(unresolved.sum())))
    return report_for(domain, findings, metrics={"row_count": int(len(frame))})


def audit_factor_semantics(events: pd.DataFrame, daily: pd.DataFrame) -> QualityReport:
    domain = "adjust_factor_daily"
    findings: list[QualityFinding] = []
    event_key = ["security_id", "divid_operate_date"]
    daily_key = ["security_id", "trade_date"]
    for frame, key, label in ((events, event_key, "event"), (daily, daily_key, "daily")):
        if not set(key).issubset(frame.columns):
            findings.append(_finding(f"factor_{label}_key_missing", f"Missing key fields {key}.", domain=domain))
            continue
        duplicate = frame.duplicated(key, keep=False)
        if duplicate.any():
            findings.append(_finding(f"factor_{label}_duplicate_key", f"{label} key is not unique.", domain=domain, count=int(duplicate.sum())))
    factor_columns = ["fore_adjust_factor", "back_adjust_factor", "adjust_factor"]
    if set(factor_columns).issubset(events.columns):
        numeric_events = events[factor_columns].apply(pd.to_numeric, errors="coerce")
        invalid = ~np.isfinite(numeric_events.to_numpy(dtype="float64")).all(axis=1) | numeric_events.le(0).any(axis=1)
        if invalid.any():
            findings.append(_finding("factor_event_nonpositive_or_nonfinite", "All event factors must be finite and positive.", domain=domain, count=int(invalid.sum())))
    if "verification_status" in events.columns:
        allowed = {"verified_dual_path", "official_arbitrated", "xdxr_official_arbitrated"}
        unverified = ~events["verification_status"].fillna("").astype(str).isin(allowed)
        if unverified.any():
            findings.append(
                _finding(
                    "factor_event_not_verified_or_arbitrated",
                    "Canonical factor events must agree across both paths or carry official arbitration.",
                    domain=domain,
                    count=int(unverified.sum()),
                    sample=events.loc[unverified, event_key + ["verification_status"]].head(20).to_dict("records"),
                )
            )
    if set(factor_columns + daily_key).issubset(daily.columns) and not daily.empty:
        ordered = daily.sort_values(daily_key).copy()
        numeric_daily = ordered[factor_columns].apply(pd.to_numeric, errors="coerce")
        invalid = numeric_daily.notna().any(axis=1) & (~np.isfinite(numeric_daily.fillna(1).to_numpy(dtype="float64")).all(axis=1) | numeric_daily.le(0).any(axis=1))
        if invalid.any():
            findings.append(_finding("factor_daily_nonpositive_or_nonfinite", "Known daily factors must be finite and positive.", domain=domain, count=int(invalid.sum())))
        changed = ordered.groupby("security_id", sort=False)[factor_columns].transform(lambda item: item.ne(item.shift()) & item.notna() & item.shift().notna()).any(axis=1)
        event_keys = set(zip(events.get("security_id", []), events.get("divid_operate_date", [])))
        changed_keys = list(zip(ordered.loc[changed, "security_id"], ordered.loc[changed, "trade_date"]))
        illegal = [key for key in changed_keys if key not in event_keys]
        if illegal:
            findings.append(_finding("factor_non_event_day_jump", "Daily factors changed without a validated event.", domain=domain, count=len(illegal), sample=illegal))
            regression = [key for key in illegal if "600076" in str(key[0])]
            if regression:
                findings.append(_finding("factor_600076_non_event_jump_regression", "600076.SH reproduced the known non-event factor-jump failure.", domain=domain, count=len(regression), sample=regression))
        if not events.empty:
            event_values = events.loc[:, event_key + factor_columns].rename(columns={"divid_operate_date": "trade_date"})
            daily_on_event = ordered.loc[:, daily_key + factor_columns].merge(event_values, on=daily_key, how="inner", suffixes=("_daily", "_event"))
            mismatched: list[dict[str, Any]] = []
            for column in factor_columns:
                left = pd.to_numeric(daily_on_event[f"{column}_daily"], errors="coerce")
                right = pd.to_numeric(daily_on_event[f"{column}_event"], errors="coerce")
                denominator = pd.concat([left.abs(), right.abs()], axis=1).max(axis=1).replace(0, np.nan)
                relative = (left - right).abs().div(denominator).fillna((left - right).abs())
                bad = relative.gt(1e-6) | left.isna() | right.isna()
                if bad.any():
                    for record in daily_on_event.loc[bad, daily_key].head(20).to_dict("records"):
                        record.update({"factor_column": column, "relative_error": float(relative.loc[bad].iloc[0])})
                        mismatched.append(record)
            if mismatched:
                findings.append(_finding("factor_event_daily_value_mismatch", "Daily factor on an event date differs from the verified event by more than 1e-6.", domain=domain, count=len(mismatched), sample=mismatched[:20]))
    if "baseline_status" in daily.columns:
        unproven = daily["baseline_status"].eq("baseline_unproven")
        if unproven.any():
            findings.append(
                _finding(
                    "factor_2010_baseline_unproven",
                    "Pre-2010 baseline evidence is missing; adjusted returns cannot be strict.",
                    domain=domain,
                    severity="warning",
                    count=int(unproven.sum()),
                    provisional=True,
                )
            )
    return report_for(domain, findings, metrics={"event_rows": int(len(events)), "daily_rows": int(len(daily))})


def audit_intraday_5m(frame: pd.DataFrame) -> QualityReport:
    domain = "market_intraday_5m"
    findings: list[QualityFinding] = []
    required = {"security_id", "trade_date", "bar_end", "open", "high", "low", "close", "volume", "amount", "source"}
    missing = sorted(required - set(frame.columns))
    if missing:
        findings.append(_finding("intraday_5m_fields_missing", f"Missing fields: {missing}", domain=domain, count=len(missing), sample=missing))
        return report_for(domain, findings, metrics={"row_count": int(len(frame))})
    duplicate = frame.duplicated(["security_id", "trade_date", "bar_end"], keep=False)
    if duplicate.any():
        findings.append(_finding("intraday_5m_duplicate_key", "5m primary key is not unique.", domain=domain, count=int(duplicate.sum())))
    expected = set(EXPECTED_5M_BAR_ENDS)
    bad_days: list[dict[str, Any]] = []
    for (security_id, trade_date), group in frame.groupby(["security_id", "trade_date"], sort=False):
        bars = group["bar_end"].astype(str).str.slice(-5).tolist()
        if len(bars) != 48 or set(bars) != expected:
            bad_days.append({"security_id": security_id, "trade_date": trade_date, "bar_count": len(bars)})
    if bad_days:
        findings.append(_finding("intraday_5m_not_48_complete_bars", "Strict stock-days require the exact 48 right-closed bars.", domain=domain, count=len(bad_days), sample=bad_days))
    return report_for(domain, findings, metrics={"row_count": int(len(frame)), "stock_day_count": int(frame.groupby(["security_id", "trade_date"]).ngroups), "invalid_stock_day_count": len(bad_days)})
