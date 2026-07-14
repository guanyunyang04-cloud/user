from __future__ import annotations

import pandas as pd

from quant_data_platform.qdp_v3.audit import (
    _audit_market_status_key_alignment,
    _audit_symbol_history_name_evidence,
)
from quant_data_platform.qdp_v3.build import _stage_daily_domains
from quant_data_platform.qdp_v3.constants import (
    RAW_TUSHARE_PROXY_DAILY,
    RAW_TUSHARE_PROXY_NAMECHANGE,
    RAW_TUSHARE_PROXY_SUSPEND,
)
from quant_data_platform.qdp_v3.identity import SecurityIdentityRegistry
from quant_data_platform.qdp_v3.storage import write_raw_partition
from quant_data_platform.qdp_v3.transforms import derive_trusted_proxy_suspend_status


def test_trusted_suspend_facts_create_status_without_market_prices() -> None:
    registry = SecurityIdentityRegistry.from_sources(provider_symbols=["600000.SH"])

    status, conflicts = derive_trusted_proxy_suspend_status(
        pd.DataFrame(
            {
                "ts_code": ["600000.SH"],
                "trade_date": ["20260710"],
                "suspend_timing": ["09:30"],
                "suspend_type": ["S"],
            }
        ),
        query_date="2026-07-10",
        identity_registry=registry,
    )

    assert conflicts.empty
    assert len(status) == 1
    assert status.loc[0, "tradestatus"] == "0"
    assert bool(status.loc[0, "is_suspended"]) is True
    assert status.loc[0, "status_source"] == "tushare_proxy.suspend_d"


def test_market_status_audit_allows_only_explicit_suspension_rows(tmp_path) -> None:
    market_path = tmp_path / "market.parquet"
    status_path = tmp_path / "status.parquet"
    pd.DataFrame(
        [{"security_id": "SEC-1", "trade_date": "2026-07-09"}]
    ).to_parquet(market_path, index=False)
    pd.DataFrame(
        [
            {
                "security_id": "SEC-1",
                "trade_date": "2026-07-09",
                "tradestatus": "1",
                "is_suspended": False,
            },
            {
                "security_id": "SEC-1",
                "trade_date": "2026-07-10",
                "tradestatus": "0",
                "is_suspended": True,
            },
        ]
    ).to_parquet(status_path, index=False)

    assert _audit_market_status_key_alignment(market_path, status_path, year="2026") == []

    broken = pd.read_parquet(status_path)
    broken.loc[1, "is_suspended"] = False
    broken.to_parquet(status_path, index=False)
    findings = _audit_market_status_key_alignment(market_path, status_path, year="2026")
    assert [item["code"] for item in findings] == ["status_only_row_not_suspended"]


def test_daily_stage_merges_explicit_suspend_status_without_zero_price_row(tmp_path) -> None:
    registry = SecurityIdentityRegistry.from_sources(provider_symbols=["600000.SH", "600001.SH"])
    daily_ref, _ = write_raw_partition(
        raw_domain=RAW_TUSHARE_PROXY_DAILY,
        partition_value="2026-07-10",
        frame=pd.DataFrame(
            {
                "ts_code": ["600000.SH"],
                "trade_date": ["20260710"],
                "open": [10.0],
                "high": [10.2],
                "low": [9.9],
                "close": [10.1],
                "pre_close": [10.0],
                "pct_chg": [1.0],
                "vol": [100.0],
                "amount": [1.0],
            }
        ),
        receipt={"quality_tier": "strict", "provider": "tushare_proxy"},
        workspace_root=tmp_path,
    )
    suspend_ref, _ = write_raw_partition(
        raw_domain=RAW_TUSHARE_PROXY_SUSPEND,
        partition_value="2026-07-10",
        frame=pd.DataFrame({"ts_code": ["600001.SH"], "trade_date": ["20260710"]}),
        receipt={"quality_tier": "strict", "provider": "tushare_proxy"},
        workspace_root=tmp_path,
    )

    staged = _stage_daily_domains(
        refs=[daily_ref],
        registry=registry,
        staging_root=tmp_path / "stage",
        suspend_refs=[suspend_ref],
    )
    market = pd.read_parquet(staged["staged"]["market_daily_raw"][0][1])
    status = pd.read_parquet(staged["staged"]["security_status_daily"][0][1])

    assert len(market) == 1
    assert len(status) == 2
    suspended = status.loc[status["provider_symbol"].eq("600001.SH")].iloc[0]
    assert suspended["tradestatus"] == "0"
    assert bool(suspended["is_suspended"]) is True


def test_semantic_name_audit_accepts_trusted_namechange_lineage(tmp_path) -> None:
    history = pd.DataFrame(
        [
            {
                "symbol": "600000.SH",
                "effective_from": "2010-01-01",
                "name_on_date": "浦发银行",
                "evidence_source": "tushare_proxy.namechange",
                "official_document_hash": "",
            }
        ]
    )
    candidate = type(
        "Candidate",
        (),
        {"raw_partitions": [{"raw_domain": RAW_TUSHARE_PROXY_NAMECHANGE}]},
    )()

    assert _audit_symbol_history_name_evidence(
        history,
        candidate,
        root=tmp_path,
        workspace_root=tmp_path,
    ) == []
