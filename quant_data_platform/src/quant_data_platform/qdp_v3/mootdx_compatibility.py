from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from quant_data_platform.domains.contracts import DataDomain, DomainFetchRequest, latest_completed_business_date
from quant_data_platform.providers import MootdxOnlineProvider, _mootdx_protocol_servers_snapshot
from quant_data_platform.qdp_v3.intraday import is_complete_5m_day, normalize_provider_5m
from quant_data_platform.qdp_v3.manifest import atomic_write_json, stable_hash, utc_now
from quant_data_platform.qdp_v3.paths import ensure_qdp_v3_layout


MOOTDX_COMPATIBILITY_SYMBOLS = ("600000.SH", "000001.SZ", "600076.SH")


def _latest_complete_days(frame: pd.DataFrame, symbols: Iterable[str], *, as_of_date: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for symbol in symbols:
        subset = frame.loc[
            frame["provider_symbol"].eq(str(symbol))
            & frame["trade_date"].le(str(as_of_date))
        ]
        proof: dict[str, Any] = {
            "provider_symbol": str(symbol),
            "status": "failed",
            "trade_date": "",
            "bar_count": 0,
            "last_bar_end": "",
        }
        for trade_date in sorted(set(subset["trade_date"].astype(str)), reverse=True):
            day = subset.loc[subset["trade_date"].eq(trade_date)].copy()
            if not is_complete_5m_day(day):
                continue
            proof.update(
                {
                    "status": "passed",
                    "trade_date": str(trade_date),
                    "bar_count": int(len(day)),
                    "last_bar_end": str(day["bar_end"].max()),
                }
            )
            break
        rows.append(proof)
    return rows


def run_mootdx_5m_compatibility_gate(
    *,
    workspace_root: str | Path | None = None,
    as_of_date: str = "",
    symbols: Iterable[str] = MOOTDX_COMPATIBILITY_SYMBOLS,
    provider: MootdxOnlineProvider | None = None,
) -> dict[str, Any]:
    paths = ensure_qdp_v3_layout(workspace_root)
    probe_date = str(as_of_date or latest_completed_business_date())[:10]
    start_date = (pd.Timestamp(probe_date) - pd.Timedelta(days=20)).strftime("%Y-%m-%d")
    symbol_list = tuple(dict.fromkeys(str(item).strip().upper() for item in symbols if str(item).strip()))
    source = provider or MootdxOnlineProvider()
    blockers: list[dict[str, Any]] = []
    started = time.perf_counter()
    result = source.fetch_domain(
        DomainFetchRequest(
            domain=DataDomain.MARKET_INTRADAY_5M,
            symbols=symbol_list,
            start_date=start_date,
            end_date=probe_date,
            adjusted_flag="none",
        )
    )
    cold_seconds = time.perf_counter() - started
    normalized = normalize_provider_5m(result.data, provider_symbol="", source="mootdx")
    proofs = _latest_complete_days(normalized, symbol_list, as_of_date=probe_date)
    if list(result.error_report or []):
        blockers.append(
            {
                "code": "mootdx_5m_provider_errors",
                "errors": list(result.error_report or [])[:10],
            }
        )
    failed = [item for item in proofs if item["status"] != "passed"]
    if failed:
        blockers.append({"code": "mootdx_5m_complete_day_missing", "sample": failed})
    if cold_seconds > 10.0:
        blockers.append(
            {
                "code": "mootdx_5m_cold_start_exceeds_10_seconds",
                "elapsed_seconds": cold_seconds,
            }
        )

    hot_started = time.perf_counter()
    hot_result = source.fetch_domain(
        DomainFetchRequest(
            domain=DataDomain.MARKET_INTRADAY_5M,
            symbols=symbol_list[:1],
            start_date=start_date,
            end_date=probe_date,
            adjusted_flag="none",
        )
    )
    hot_seconds = time.perf_counter() - hot_started
    if list(hot_result.error_report or []):
        blockers.append(
            {
                "code": "mootdx_5m_hot_start_failed",
                "errors": list(hot_result.error_report or [])[:5],
            }
        )
    if hot_seconds > 1.0:
        blockers.append(
            {
                "code": "mootdx_5m_hot_start_exceeds_1_second",
                "elapsed_seconds": hot_seconds,
            }
        )
    healthy_servers = [
        {"host": host, "port": port}
        for host, port in _mootdx_protocol_servers_snapshot()
    ]
    if not healthy_servers and provider is None:
        blockers.append({"code": "mootdx_5m_healthy_server_pool_empty"})
    payload = {
        "status": "passed" if not blockers else "blocked",
        "provider": "mootdx_online",
        "protocol": "tdx_quote_protocol",
        "frequency": "5m",
        "probe_date": probe_date,
        "symbols": list(symbol_list),
        "proofs": proofs,
        "cold_start_seconds": cold_seconds,
        "hot_start_seconds": hot_seconds,
        "healthy_servers": healthy_servers,
        "provider_errors": list(result.error_report or []),
        "blockers": blockers,
        "created_at": utc_now(),
    }
    report_id = stable_hash(payload, length=24)
    report_path = paths.compatibility / f"mootdx_5m__{report_id}.json"
    atomic_write_json(report_path, payload)
    source.close()
    return {**payload, "report_path": str(report_path.resolve())}
