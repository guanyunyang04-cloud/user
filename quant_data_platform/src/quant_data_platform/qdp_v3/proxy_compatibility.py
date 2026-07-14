from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping

import pandas as pd

from quant_data_platform.core.json_io import read_json
from quant_data_platform.domains.contracts import HistoryPageFetchRequest, latest_completed_business_date
from quant_data_platform.qdp_v3.manifest import atomic_write_json, stable_hash, utc_now
from quant_data_platform.qdp_v3.paths import ensure_qdp_v3_layout, qdp_v3_paths
from quant_data_platform.tushare_proxy import (
    TUSHARE_PROXY_DEFAULT_MINUTE_DAILY_LIMIT,
    TUSHARE_PROXY_HISTORY_PAGE_SIZE,
    TushareProxyClient,
    TushareProxyError,
    redact_secrets,
)


REQUIRED_PROXY_APIS = (
    "stock_basic",
    "trade_cal",
    "daily",
    "daily_basic",
    "adj_factor",
    "namechange",
    "suspend_d",
    "stk_limit",
    "dividend",
    "income",
    "balancesheet",
    "cashflow",
    "fina_indicator",
    "forecast",
    "express",
    "stk_mins",
)

ESTIMATED_BOOTSTRAP_MINUTE_REQUESTS = 65_613
UNIT_CONTRACT_SAMPLE_DATE = "2010-01-04"
UNIT_CONTRACT_SAMPLE_SYMBOLS = ("600000.SH", "000001.SZ")


def _evaluate_5m_unit_contract(minute: pd.DataFrame, daily: pd.DataFrame) -> dict[str, Any]:
    if minute.empty or daily.empty:
        raise RuntimeError("stk_mins_unit_contract_sample_missing")
    volume_column = "vol" if "vol" in minute.columns else "volume" if "volume" in minute.columns else ""
    if not volume_column or "amount" not in minute.columns or "vol" not in daily.columns or "amount" not in daily.columns:
        raise RuntimeError("stk_mins_unit_contract_fields_missing")
    minute_volume = float(pd.to_numeric(minute[volume_column], errors="coerce").sum())
    minute_amount = float(pd.to_numeric(minute["amount"], errors="coerce").sum())
    daily_volume = float(pd.to_numeric(daily.iloc[0]["vol"], errors="coerce")) * 100.0
    daily_amount = float(pd.to_numeric(daily.iloc[0]["amount"], errors="coerce")) * 1_000.0

    volume_scales = [
        scale
        for scale in (1.0, 100.0)
        if abs(minute_volume * scale - daily_volume) <= max(100.0, 1e-5 * abs(daily_volume))
    ]
    amount_scales = [
        scale
        for scale in (1.0, 1_000.0)
        if abs(minute_amount * scale - daily_amount) <= max(1_000.0, 0.001 * abs(daily_amount))
    ]
    if volume_scales != [1.0] or amount_scales != [1.0]:
        raise RuntimeError(
            f"stk_mins_unit_contract_not_unique:volume={volume_scales}:amount={amount_scales}"
        )
    return {
        "status": "passed",
        "raw_volume_unit": "share",
        "raw_amount_unit": "CNY",
        "canonical_volume_scale": 1.0,
        "canonical_amount_scale": 1.0,
        "volume_relative_difference": abs(minute_volume - daily_volume) / max(abs(daily_volume), 1.0),
        "amount_relative_difference": abs(minute_amount - daily_amount) / max(abs(daily_amount), 1.0),
    }


def _probe_contracts(as_of_date: str) -> list[dict[str, Any]]:
    compact = str(as_of_date).replace("-", "")
    return [
        {"api_name": "stock_basic", "params": {"list_status": "L"}, "fields": "ts_code,name,market,list_status,list_date,delist_date"},
        {"api_name": "trade_cal", "params": {"start_date": compact, "end_date": compact}, "fields": "exchange,cal_date,is_open,pretrade_date"},
        {"api_name": "daily", "params": {"trade_date": "20100104"}, "fields": "ts_code,trade_date,open,high,low,close,pre_close,vol,amount"},
        {"api_name": "daily_basic", "params": {"trade_date": "20100104"}, "fields": "ts_code,trade_date,total_share,float_share,total_mv,circ_mv,turnover_rate,pe_ttm,pb"},
        {"api_name": "adj_factor", "params": {"ts_code": "600000.SH", "start_date": "20050101", "end_date": compact}, "fields": "ts_code,trade_date,adj_factor"},
        {"api_name": "namechange", "params": {"ts_code": "300114.SZ"}, "fields": "ts_code,name,start_date,end_date,ann_date,change_reason"},
        {"api_name": "suspend_d", "params": {"trade_date": "20100104"}, "fields": "ts_code,trade_date,suspend_timing,suspend_type"},
        {"api_name": "stk_limit", "params": {"trade_date": "20100104"}, "fields": "trade_date,ts_code,pre_close,up_limit,down_limit"},
        {"api_name": "dividend", "params": {"ts_code": "600000.SH"}, "fields": "ts_code,end_date,ann_date,div_proc,stk_div,cash_div,record_date,ex_date,pay_date"},
        {"api_name": "income", "params": {"ts_code": "600000.SH", "start_date": "20100101", "end_date": "20101231"}, "fields": "ts_code,ann_date,f_ann_date,end_date,report_type,total_revenue,n_income_attr_p"},
        {"api_name": "balancesheet", "params": {"ts_code": "600000.SH", "start_date": "20100101", "end_date": "20101231"}, "fields": "ts_code,ann_date,f_ann_date,end_date,total_assets,total_liab"},
        {"api_name": "cashflow", "params": {"ts_code": "600000.SH", "start_date": "20100101", "end_date": "20101231"}, "fields": "ts_code,ann_date,f_ann_date,end_date,n_cashflow_act"},
        {"api_name": "fina_indicator", "params": {"ts_code": "600000.SH", "start_date": "20100101", "end_date": "20101231"}, "fields": "ts_code,ann_date,end_date,eps,roe,assets_turn"},
        {"api_name": "forecast", "params": {"ts_code": "600000.SH", "start_date": "20100101", "end_date": "20101231"}, "fields": "ts_code,ann_date,end_date,type,p_change_min,p_change_max"},
        {"api_name": "express", "params": {"ts_code": "600000.SH", "start_date": "20100101", "end_date": "20101231"}, "fields": "ts_code,ann_date,end_date,revenue,n_income"},
    ]


def _no_secret_assertion(payload: Mapping[str, Any], *, token: str) -> None:
    serialized = json.dumps(payload, ensure_ascii=False, default=str)
    if token and token in serialized:
        raise RuntimeError("tushare_proxy_secret_leak_in_compatibility_payload")
    if "/token=" in serialized.lower():
        raise RuntimeError("tushare_proxy_mcp_url_leak_in_compatibility_payload")


def run_tushare_proxy_compatibility_gate(
    *,
    workspace_root: str | Path | None = None,
    as_of_date: str = "",
    client: TushareProxyClient | None = None,
    smoke: bool = False,
) -> dict[str, Any]:
    paths = ensure_qdp_v3_layout(workspace_root)
    source = client or TushareProxyClient()
    probe_date = str(as_of_date or latest_completed_business_date())[:10]
    probes: list[dict[str, Any]] = []
    blockers: list[dict[str, Any]] = []
    entitlement: dict[str, Any] = {}
    try:
        entitlement = source.fetch_key_status()
    except Exception as exc:
        blockers.append(
            {
                "code": "key_status_probe_failed",
                "error": str(redact_secrets(exc, secrets=(source.config.token,))),
            }
        )
    for contract in _probe_contracts(probe_date):
        api_name = str(contract["api_name"])
        try:
            result = source.fetch_frame(
                api_name=api_name,
                params=dict(contract["params"]),
                fields=str(contract["fields"]),
            )
            probes.append(
                {
                    "api_name": api_name,
                    "status": "passed",
                    "row_count": int(len(result.frame)),
                    "fields": list(result.fields),
                    "response_sha256": result.response_sha256,
                    "elapsed_seconds": result.elapsed_seconds,
                }
            )
        except Exception as exc:
            probes.append(
                {
                    "api_name": api_name,
                    "status": "failed",
                    "error": str(redact_secrets(exc, secrets=(source.config.token,))),
                }
            )
            blockers.append({"code": "required_api_probe_failed", "api_name": api_name})
    for frequency in ("5m",):
        start_at = "2010-01-01 00:00:00" if not smoke else "2010-01-04 00:00:00"
        end_at = f"{probe_date} 23:59:59" if not smoke else "2010-01-04 23:59:59"
        try:
            first = source.fetch_history_page(
                HistoryPageFetchRequest(
                    provider_symbol="600000.SH",
                    start_at=start_at,
                    end_at=end_at,
                    page_size=TUSHARE_PROXY_HISTORY_PAGE_SIZE,
                )
            )
            pagination = {
                "api_name": "stk_mins",
                "frequency": frequency,
                "status": "passed",
                "first_page_row_count": first.row_count,
                "first_page_min_timestamp": first.min_timestamp,
                "first_page_max_timestamp": first.max_timestamp,
                "first_page_response_sha256": first.response_sha256,
                "is_complete": first.is_complete,
            }
            if not smoke:
                if first.row_count != TUSHARE_PROXY_HISTORY_PAGE_SIZE or first.is_complete or not first.next_end_at:
                    raise RuntimeError(f"stk_mins_full_page_truncation_contract_not_observed:{frequency}")
                second = source.fetch_history_page(
                    HistoryPageFetchRequest(
                        provider_symbol="600000.SH",
                        start_at=start_at,
                        end_at=first.next_end_at,
                        page_size=TUSHARE_PROXY_HISTORY_PAGE_SIZE,
                    )
                )
                if second.max_timestamp and pd.Timestamp(second.max_timestamp) >= pd.Timestamp(first.min_timestamp):
                    raise RuntimeError(f"stk_mins_reverse_pagination_overlap:{frequency}")
                pagination.update(
                    {
                        "second_page_row_count": second.row_count,
                        "second_page_max_timestamp": second.max_timestamp,
                        "second_page_response_sha256": second.response_sha256,
                    }
                )
            probes.append(pagination)
        except Exception as exc:
            probes.append(
                {
                    "api_name": "stk_mins",
                    "frequency": frequency,
                    "status": "failed",
                    "error": str(redact_secrets(exc, secrets=(source.config.token,))),
                }
            )
            blockers.append({"code": "stk_mins_pagination_probe_failed", "frequency": frequency})
    try:
        compact_unit_date = UNIT_CONTRACT_SAMPLE_DATE.replace("-", "")
        unit_daily = source.fetch_frame(
            api_name="daily",
            params={"trade_date": compact_unit_date},
            fields="ts_code,trade_date,open,high,low,close,vol,amount",
        )
        unit_samples: list[dict[str, Any]] = []
        for symbol in UNIT_CONTRACT_SAMPLE_SYMBOLS:
            unit_page = source.fetch_history_page(
                HistoryPageFetchRequest(
                    provider_symbol=symbol,
                    start_at=f"{UNIT_CONTRACT_SAMPLE_DATE} 00:00:00",
                    end_at=f"{UNIT_CONTRACT_SAMPLE_DATE} 23:59:59",
                    page_size=TUSHARE_PROXY_HISTORY_PAGE_SIZE,
                )
            )
            if unit_page.row_count != 48 or not unit_page.max_timestamp.endswith("15:00:00"):
                raise RuntimeError(f"stk_mins_unit_contract_day_incomplete:{symbol}")
            daily_sample = unit_daily.frame.loc[unit_daily.frame["ts_code"].astype(str).str.upper().eq(symbol)]
            evaluation = _evaluate_5m_unit_contract(unit_page.raw_data, daily_sample)
            unit_samples.append(
                {
                    "provider_symbol": symbol,
                    "trade_date": UNIT_CONTRACT_SAMPLE_DATE,
                    "minute_response_sha256": unit_page.response_sha256,
                    "daily_response_sha256": unit_daily.response_sha256,
                    **evaluation,
                }
            )
        probes.append(
            {
                "api_name": "stk_mins_unit_contract",
                "status": "passed",
                "samples": unit_samples,
                "raw_volume_unit": "share",
                "raw_amount_unit": "CNY",
                "canonical_volume_scale": 1.0,
                "canonical_amount_scale": 1.0,
            }
        )
    except Exception as exc:
        probes.append(
            {
                "api_name": "stk_mins_unit_contract",
                "status": "failed",
                "error": str(redact_secrets(exc, secrets=(source.config.token,))),
            }
        )
        blockers.append({"code": "stk_mins_unit_contract_probe_failed"})
    observed_apis = {str(item.get("api_name", "")) for item in probes if item.get("status") == "passed"}
    missing = sorted(set(REQUIRED_PROXY_APIS) - observed_apis)
    if missing:
        blockers.append({"code": "required_api_coverage_missing", "apis": missing})
    minute_daily_limit = int(
        entitlement.get("stk_mins_daily_limit", TUSHARE_PROXY_DEFAULT_MINUTE_DAILY_LIMIT)
        or TUSHARE_PROXY_DEFAULT_MINUTE_DAILY_LIMIT
    )
    estimated_days = int(math.ceil(ESTIMATED_BOOTSTRAP_MINUTE_REQUESTS / max(minute_daily_limit, 1)))
    payload: dict[str, Any] = {
        "status": "passed" if not blockers else "blocked",
        "provider": "tushare_proxy",
        "protocol": "tushare_compatible_http",
        "endpoint": source.config.url,
        "upstream_provenance": "not_exposed",
        "production_role": "historical_bootstrap",
        "probe_date": probe_date,
        "smoke": bool(smoke),
        "entitlement": entitlement,
        "probes": probes,
        "metrics": source.operational_metrics(),
        "estimated_bootstrap_minute_requests": ESTIMATED_BOOTSTRAP_MINUTE_REQUESTS,
        "minute_daily_limit": minute_daily_limit,
        "estimated_minimum_calendar_days_at_daily_limit": estimated_days,
        "blockers": blockers,
        "created_at": utc_now(),
    }
    _no_secret_assertion(payload, token=source.config.token)
    report_id = stable_hash(payload, length=24)
    report_path = paths.compatibility / f"tushare_proxy__{report_id}.json"
    atomic_write_json(report_path, payload)
    payload["report_path"] = str(report_path.resolve())
    return payload


def lock_bootstrap_cutoff(
    *,
    requested_cutoff: str = "",
    sample_symbols: Iterable[str] = ("600000.SH",),
    workspace_root: str | Path | None = None,
    client: TushareProxyClient | None = None,
) -> dict[str, Any]:
    """Validate and immutably lock the historical bootstrap end date."""

    paths = ensure_qdp_v3_layout(workspace_root)
    cutoff_path = paths.metadata / "tushare_proxy_bootstrap_cutoff.json"
    five_minute_proof_path = paths.metadata / "tushare_proxy_bootstrap_cutoff_5m_proof.json"
    existing = read_json(cutoff_path)
    if existing:
        locked = str(existing.get("bootstrap_cutoff", "") or "")
        if requested_cutoff and str(requested_cutoff)[:10] != locked:
            raise RuntimeError(f"bootstrap_cutoff_already_locked:{locked}")
        existing_proof = read_json(five_minute_proof_path)
        existing_minute_proofs = list(existing_proof.get("minute_proofs", []) or [])
        if (
            str(existing_proof.get("bootstrap_cutoff", "") or "") == locked
            and str(existing_proof.get("frequency", "") or "") == "5m"
            and existing_minute_proofs
            and all(
                int(item.get("row_count", 0) or 0) == 48
                and str(item.get("max_timestamp", "") or "").endswith("15:00:00")
                for item in existing_minute_proofs
                if isinstance(item, dict)
            )
            and all(isinstance(item, dict) for item in existing_minute_proofs)
        ):
            return {
                **existing_proof,
                "status": "already_locked_5m_proved",
                "path": str(cutoff_path.resolve()),
                "five_minute_proof_path": str(five_minute_proof_path.resolve()),
            }
        cutoff = locked
    else:
        cutoff = str(requested_cutoff or latest_completed_business_date())[:10]
    if pd.Timestamp(cutoff) > pd.Timestamp(latest_completed_business_date()):
        raise ValueError(f"bootstrap_cutoff_not_completed:{cutoff}")
    source = client or TushareProxyClient()
    compact = cutoff.replace("-", "")
    calendar = source.fetch_frame(
        api_name="trade_cal",
        params={"start_date": compact, "end_date": compact},
        fields="exchange,cal_date,is_open,pretrade_date",
    )
    if calendar.frame.empty or not calendar.frame["is_open"].astype(str).isin({"1", "true", "True"}).any():
        raise RuntimeError(f"bootstrap_cutoff_not_open_trade_date:{cutoff}")
    daily = source.fetch_frame(
        api_name="daily",
        params={"trade_date": compact},
        fields="ts_code,trade_date,open,high,low,close,pre_close,vol,amount",
    )
    if daily.frame.empty:
        raise RuntimeError(f"bootstrap_cutoff_daily_missing:{cutoff}")
    minute_proofs: list[dict[str, Any]] = []
    for symbol in sorted({str(item).strip().upper() for item in sample_symbols if str(item).strip()}):
        page = source.fetch_history_page(
            HistoryPageFetchRequest(
                provider_symbol=symbol,
                start_at=f"{cutoff} 00:00:00",
                end_at=f"{cutoff} 23:59:59",
                page_size=TUSHARE_PROXY_HISTORY_PAGE_SIZE,
            )
        )
        if page.row_count != 48 or not page.max_timestamp.endswith("15:00:00"):
            raise RuntimeError(f"bootstrap_cutoff_sample_5m_incomplete:{symbol}:{cutoff}")
        minute_proofs.append(
            {
                "provider_symbol": symbol,
                "row_count": page.row_count,
                "max_timestamp": page.max_timestamp,
                "response_sha256": page.response_sha256,
            }
        )
    payload = {
        "status": "revalidated_existing_lock" if existing else "locked",
        "provider": "tushare_proxy",
        "frequency": "5m",
        "bootstrap_cutoff": cutoff,
        "calendar_response_sha256": calendar.response_sha256,
        "daily_response_sha256": daily.response_sha256,
        "minute_proofs": minute_proofs,
        "provider_metadata": source.config.public_metadata(),
        "locked_at": utc_now(),
    }
    _no_secret_assertion(payload, token=source.config.token)
    if existing:
        atomic_write_json(five_minute_proof_path, payload)
        return {
            **payload,
            "path": str(cutoff_path.resolve()),
            "five_minute_proof_path": str(five_minute_proof_path.resolve()),
        }
    atomic_write_json(cutoff_path, payload)
    atomic_write_json(five_minute_proof_path, payload)
    return {
        **payload,
        "path": str(cutoff_path.resolve()),
        "five_minute_proof_path": str(five_minute_proof_path.resolve()),
    }
