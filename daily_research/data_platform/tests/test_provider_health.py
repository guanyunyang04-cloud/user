from __future__ import annotations

from pathlib import Path

import pandas as pd

from daily_research.data_platform.contracts import DataDomain
from daily_research.data_platform.manager import InMemoryDomainProvider
from daily_research.data_platform.provider_health import ProviderHealthConfig, run_provider_health


def test_provider_health_is_read_only_and_reports_domain_rows(tmp_path: Path) -> None:
    provider = InMemoryDomainProvider(
        "baostock",
        {
            DataDomain.MARKET_DAILY: pd.DataFrame(
                {
                    "symbol": ["000001.SZ"],
                    "trade_date": ["2026-05-22"],
                    "open": [10.0],
                    "high": [10.2],
                    "low": [9.8],
                    "close": [10.1],
                    "volume": [1000],
                    "amount": [10100],
                    "source": ["baostock"],
                    "adjusted_flag": ["none"],
                }
            ),
            DataDomain.TRADING_CALENDAR: pd.DataFrame(
                {
                    "trade_date": ["2026-05-22"],
                    "is_open": [True],
                    "exchange": ["SSE"],
                    "source": ["baostock"],
                }
            ),
        },
    )

    before = sorted(tmp_path.rglob("*"))
    payload = run_provider_health(
        ProviderHealthConfig(
            provider_plan="formal_free_v3",
            as_of_date="2026-05-22",
            domains=(DataDomain.MARKET_DAILY, DataDomain.TRADING_CALENDAR),
            symbols=("000001.SZ",),
        ),
        providers=[provider],
    )
    after = sorted(tmp_path.rglob("*"))

    assert before == after
    assert payload["status"] == "ok"
    assert payload["provider_plan"] == "formal_free_v3"
    assert payload["summary"]["checked_domain_count"] == 2
    assert payload["summary"]["ok_domain_count"] == 2
    assert payload["providers"][0]["provider"] == "baostock"
    assert payload["providers"][0]["domains"]["market_daily"]["status"] == "ok"


def test_provider_health_marks_required_domain_missing_without_throwing() -> None:
    provider = InMemoryDomainProvider("baostock", {})

    payload = run_provider_health(
        ProviderHealthConfig(
            provider_plan="formal_free_v3",
            as_of_date="2026-05-22",
            domains=(DataDomain.MARKET_DAILY,),
            symbols=("000001.SZ",),
        ),
        providers=[provider],
    )

    assert payload["status"] == "degraded"
    assert payload["providers"][0]["domains"]["market_daily"]["status"] in {"error", "no_data", "unsupported"}
    assert payload["summary"]["error_count"] >= 1


def test_provider_health_degrades_when_any_requested_required_domain_has_no_ok_provider() -> None:
    provider = InMemoryDomainProvider(
        "baostock",
        {
            DataDomain.MARKET_DAILY: pd.DataFrame(
                {
                    "symbol": ["000001.SZ"],
                    "trade_date": ["2026-05-22"],
                    "open": [10.0],
                    "high": [10.2],
                    "low": [9.8],
                    "close": [10.1],
                    "volume": [1000],
                    "amount": [10100],
                    "source": ["baostock"],
                    "adjusted_flag": ["none"],
                }
            )
        },
    )

    payload = run_provider_health(
        ProviderHealthConfig(
            provider_plan="formal_free_v3",
            as_of_date="2026-05-22",
            domains=(DataDomain.MARKET_DAILY, DataDomain.TRADING_CALENDAR),
            symbols=("000001.SZ",),
        ),
        providers=[provider],
    )

    assert payload["status"] == "degraded"
    assert payload["summary"]["required_domain_status"]["market_daily"] == "ok"
    assert payload["summary"]["required_domain_status"]["trading_calendar"] == "missing"
