from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from quant_data_platform.qdp_v3.constants import (
    BOOTSTRAP_CUTOFF,
    MANIFEST_VERSION,
    QDP_V3_CONTRACT_VERSION,
    RAW_DAILY_ASTOCK,
    RAW_TUSHARE_PROXY_DAILY,
    SCHEMA_VERSION,
    STRICT_RELEASE_DOMAINS,
)
from quant_data_platform.qdp_v3.build import (
    _compose_calendar_frames,
    _compose_security_master_inventory,
    _compose_trusted_daily_refs,
    _stage_daily_domains,
    _watermark_contract_findings,
)
from quant_data_platform.qdp_v3.corrections import (
    CORRECTIONS_SCHEMA_VERSION,
    CorrectionSet,
    apply_corrections,
)
from quant_data_platform.qdp_v3.identity import SecurityIdentityRegistry
from quant_data_platform.qdp_v3.proxy_factor import (
    build_trusted_factor_daily,
    continue_trusted_factors_with_baostock,
    normalize_trusted_proxy_factors,
)
from quant_data_platform.qdp_v3.quality import (
    QualityFinding,
    audit_strict_5m_coverage,
    audit_trusted_factor_daily,
    report_for,
)
from quant_data_platform.qdp_v3.transforms import derive_trusted_proxy_daily_domains
from quant_data_platform.qdp_v3.storage import write_raw_partition


def _identity_config() -> Path:
    return Path(__file__).resolve().parents[2] / "configs" / "qdp_v3_symbol_history.json"


def _correction(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "correction_id": "fix-1",
        "provider": "tushare_proxy",
        "domain": "market_daily_raw",
        "key": {"security_id": "SEC-1"},
        "field": "close",
        "old_value": 10.0,
        "new_value": 10.1,
        "effective_from": "2024-01-02",
        "effective_to": "2024-01-02",
        "reason": "curated provider typo",
    }
    payload.update(overrides)
    return payload


def test_trusted_source_contract_has_exact_nine_core_domains() -> None:
    assert QDP_V3_CONTRACT_VERSION == "qdp_v3_20260715_trusted_source_5m"
    assert SCHEMA_VERSION == "3.3.0"
    assert MANIFEST_VERSION == 4
    assert len(STRICT_RELEASE_DOMAINS) == 9
    assert "valuation_daily" not in STRICT_RELEASE_DOMAINS
    assert "adjust_factor_event" not in STRICT_RELEASE_DOMAINS
    assert "market_intraday_1m" not in STRICT_RELEASE_DOMAINS


def test_new_quality_reports_do_not_emit_provisional() -> None:
    report = report_for(
        "example",
        [QualityFinding(code="legacy_evidence", severity="warning", message="legacy", provisional=True)],
    )
    assert report.quality_tier == "strict"


@pytest.mark.parametrize(
    ("covered", "code", "severity"),
    [
        (979, "strict_5m_coverage_below_98_percent", "blocker"),
        (985, "strict_5m_coverage_below_99_percent", "warning"),
        (990, "", ""),
    ],
)
def test_lean_5m_coverage_boundaries(covered: int, code: str, severity: str) -> None:
    findings = audit_strict_5m_coverage(
        {
            "expected_stock_day_count": 1_000,
            "strict_covered_stock_day_count": covered,
            "strict_coverage_rate": covered / 1_000,
        }
    )
    assert ([item.code for item in findings] or [""]) == [code]
    assert ([item.severity for item in findings] or [""]) == [severity]


def test_corrections_apply_to_canonical_copy_and_validate_old_value() -> None:
    rules = CorrectionSet.from_mapping(
        {"schema_version": CORRECTIONS_SCHEMA_VERSION, "regressions": [], "corrections": [_correction()]}
    )
    raw = pd.DataFrame([{"security_id": "SEC-1", "trade_date": "2024-01-02", "close": 10.0}])
    corrected, applied = apply_corrections(
        raw,
        provider="tushare_proxy",
        domain="market_daily_raw",
        corrections=rules,
    )
    assert raw.loc[0, "close"] == 10.0
    assert corrected.loc[0, "close"] == pytest.approx(10.1)
    assert applied[0]["correction_id"] == "fix-1"
    with pytest.raises(ValueError, match="correction_old_value_mismatch"):
        apply_corrections(
            raw.assign(close=9.9),
            provider="tushare_proxy",
            domain="market_daily_raw",
            corrections=rules,
        )


def test_correction_overlap_is_rejected() -> None:
    with pytest.raises(ValueError, match="correction_interval_overlap"):
        CorrectionSet.from_mapping(
            {
                "schema_version": CORRECTIONS_SCHEMA_VERSION,
                "regressions": [],
                "corrections": [
                    _correction(),
                    _correction(
                        correction_id="fix-2",
                        effective_from="2024-01-02",
                        effective_to="2024-01-03",
                        new_value=10.2,
                    ),
                ],
            }
        )


def test_tushare_daily_units_and_rested_symbol_are_canonicalized() -> None:
    registry = SecurityIdentityRegistry.from_sources(
        provider_symbols=["302132.SZ"],
        config_path=_identity_config(),
    )
    raw = pd.DataFrame(
        [
            {
                "ts_code": "302132.SZ",
                "trade_date": "20160104",
                "open": 10.0,
                "high": 11.0,
                "low": 9.0,
                "close": 10.5,
                "pre_close": 10.0,
                "change": 0.5,
                "pct_chg": 5.0,
                "vol": 123.45,
                "amount": 456.78,
            }
        ]
    )
    derived = derive_trusted_proxy_daily_domains(raw, query_date="2016-01-04", identity_registry=registry)
    row = derived.market_daily_raw.iloc[0]
    assert row["provider_symbol"] == "302132.SZ"
    assert row["symbol_on_date"] == "300114.SZ"
    assert row["volume"] == pytest.approx(12_345.0)
    assert row["amount"] == pytest.approx(456_780.0)
    assert row["source"] == "tushare_proxy.daily"


def test_trusted_daily_partitions_switch_once_at_cutoff() -> None:
    def ref(raw_domain: str, trade_date: str) -> SimpleNamespace:
        return SimpleNamespace(raw_domain=raw_domain, partition_value=trade_date)

    selected, metrics = _compose_trusted_daily_refs(
        [
            ref(RAW_TUSHARE_PROXY_DAILY, "2026-07-10"),
            ref(RAW_TUSHARE_PROXY_DAILY, BOOTSTRAP_CUTOFF),
            ref(RAW_TUSHARE_PROXY_DAILY, "2026-07-14"),
        ],
        [
            ref(RAW_DAILY_ASTOCK, BOOTSTRAP_CUTOFF),
            ref(RAW_DAILY_ASTOCK, "2026-07-14"),
            ref(RAW_DAILY_ASTOCK, "2026-07-15"),
        ],
        start_date="2026-07-10",
        end_date="2026-07-15",
    )

    assert [(item.raw_domain, item.partition_value) for item in selected] == [
        (RAW_TUSHARE_PROXY_DAILY, "2026-07-10"),
        (RAW_TUSHARE_PROXY_DAILY, BOOTSTRAP_CUTOFF),
        (RAW_DAILY_ASTOCK, "2026-07-14"),
        (RAW_DAILY_ASTOCK, "2026-07-15"),
    ]
    assert metrics["ignored_tushare_post_cutoff_count"] == 1
    assert metrics["ignored_baostock_through_cutoff_count"] == 1


def test_mixed_trusted_daily_stage_uses_each_partition_provider_schema(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    proxy_ref, _ = write_raw_partition(
        raw_domain=RAW_TUSHARE_PROXY_DAILY,
        partition_field="trade_date",
        partition_value=BOOTSTRAP_CUTOFF,
        frame=pd.DataFrame(
            [
                {
                    "ts_code": "600000.SH",
                    "trade_date": "20260713",
                    "open": 10.0,
                    "high": 10.5,
                    "low": 9.8,
                    "close": 10.2,
                    "pre_close": 10.0,
                    "pct_chg": 2.0,
                    "vol": 100.0,
                    "amount": 1.0,
                }
            ]
        ),
        receipt={"provider": "tushare_proxy", "quality_tier": "strict"},
        workspace_root=workspace,
    )
    baostock_ref, _ = write_raw_partition(
        raw_domain=RAW_DAILY_ASTOCK,
        partition_field="query_date",
        partition_value="2026-07-14",
        frame=pd.DataFrame(
            [
                {
                    "date": "2026-07-14",
                    "code": "sh.600000",
                    "open": "10.2",
                    "high": "10.8",
                    "low": "10.1",
                    "close": "10.6",
                    "preclose": "10.2",
                    "volume": "20000",
                    "amount": "210000",
                    "adjustflag": "3",
                    "turn": "0.1",
                    "tradestatus": "1",
                    "pctChg": "3.9215686",
                    "peTTM": "10",
                    "pbMRQ": "1",
                    "psTTM": "2",
                    "pcfNcfTTM": "3",
                    "isST": "0",
                }
            ]
        ),
        receipt={"provider": "baostock", "quality_tier": "strict"},
        workspace_root=workspace,
    )
    registry = SecurityIdentityRegistry.from_sources(
        provider_symbols=["600000.SH"],
        security_master=pd.DataFrame(
            [{"symbol": "600000.SH", "list_date": "1999-11-10", "name": "浦发银行"}]
        ),
        workspace_root=workspace,
    )

    staged = _stage_daily_domains(
        refs=[proxy_ref, baostock_ref],
        registry=registry,
        staging_root=tmp_path / "stage",
    )
    market_path = staged["staged"]["market_daily_raw"][0][1]
    market = pd.read_parquet(market_path, engine="pyarrow")

    assert market["trade_date"].tolist() == [BOOTSTRAP_CUTOFF, "2026-07-14"]
    assert market["source"].tolist() == ["tushare_proxy.daily", "baostock.query_daily_history_k_AStock"]


def test_post_cutoff_calendar_and_new_listing_extend_trusted_inventory() -> None:
    proxy_master = pd.DataFrame(
        [{"symbol": "600000.SH", "name": "浦发银行", "list_date": "1999-11-10"}]
    )
    baostock_master = pd.DataFrame(
        [
            {"symbol": "600000.SH", "name": "浦发银行", "list_date": "1999-11-10"},
            {"symbol": "001999.SZ", "name": "新增证券", "list_date": "2026-07-14"},
        ]
    )
    inventory = _compose_security_master_inventory(
        proxy_master,
        baostock_master,
        baostock_as_of_date="2026-07-15",
    )
    assert set(inventory["symbol"]) == {"600000.SH", "001999.SZ"}
    assert inventory.loc[inventory["symbol"].eq("001999.SZ"), "identity_source"].item() == "baostock.query_stock_basic"

    proxy_calendar = pd.DataFrame(
        {
            "trade_date": ["2026-07-12", BOOTSTRAP_CUTOFF],
            "is_open": [False, True],
            "exchange": ["SSE/SZSE", "SSE/SZSE"],
            "source": ["tushare_proxy.trade_cal"] * 2,
        }
    )
    baostock_calendar = pd.DataFrame(
        {
            "trade_date": [BOOTSTRAP_CUTOFF, "2026-07-14", "2026-07-15"],
            "is_open": [False, True, True],
            "exchange": ["SSE/SZSE"] * 3,
            "source": ["baostock.query_trade_dates"] * 3,
        }
    )
    calendar = _compose_calendar_frames(
        proxy_calendar,
        baostock_calendar,
        start_date="2026-07-12",
        end_date="2026-07-15",
    )
    assert calendar["trade_date"].tolist() == ["2026-07-12", BOOTSTRAP_CUTOFF, "2026-07-14", "2026-07-15"]
    assert calendar.loc[calendar["trade_date"].eq(BOOTSTRAP_CUTOFF), "is_open"].item() is True


def test_requested_end_rejects_stale_daily_and_intraday_watermarks() -> None:
    calendar = pd.DataFrame(
        {
            "trade_date": [BOOTSTRAP_CUTOFF, "2026-07-14", "2026-07-15"],
            "is_open": [True, True, True],
        }
    )
    findings, required = _watermark_contract_findings(
        calendar=calendar,
        snapshot_dates=[BOOTSTRAP_CUTOFF, "2026-07-14"],
        intraday_watermark="2026-07-14",
        requested_end_date="2026-07-15",
    )
    assert required == "2026-07-15"
    assert {item["code"] for item in findings} == {
        "market_daily_watermark_not_at_requested_end",
        "market_intraday_5m_watermark_not_at_requested_end",
    }


def test_curated_alias_evidence_no_longer_requires_document_hash() -> None:
    registry = SecurityIdentityRegistry.from_sources(
        provider_symbols=[],
        config_path=_identity_config(),
    )
    identities, history = registry.identity_frames()
    history["official_document_hash"] = ""
    rebuilt = SecurityIdentityRegistry(identities, history)
    security_id = rebuilt.security_id_for_provider_symbol("302132.SZ")
    assert rebuilt.symbol_for_date(security_id, "2016-01-04") == "300114.SZ"


def test_trusted_factor_normalization_and_post_cutoff_ratio_continuation() -> None:
    proxy = normalize_trusted_proxy_factors(
        pd.DataFrame(
            [
                {"security_id": "SEC-1", "trade_date": "20100104", "adj_factor": 2.0},
                {"security_id": "SEC-1", "trade_date": "20260713", "adj_factor": 4.0},
            ]
        )
    )
    assert proxy["adjust_factor"].tolist() == pytest.approx([1.0, 2.0])
    continuation = continue_trusted_factors_with_baostock(
        proxy,
        pd.DataFrame(
            [
                {"security_id": "SEC-1", "divid_operate_date": "2026-07-13", "adjust_factor": 4.0},
                {"security_id": "SEC-1", "divid_operate_date": "2026-07-15", "adjust_factor": 6.0},
            ]
        ),
        cutoff="2026-07-13",
    )
    assert continuation.loc[0, "adjust_factor"] == pytest.approx(3.0)
    daily = build_trusted_factor_daily(
        pd.DataFrame(
            [
                {"security_id": "SEC-1", "trade_date": "2010-01-04"},
                {"security_id": "SEC-1", "trade_date": "2026-07-14"},
                {"security_id": "SEC-1", "trade_date": "2026-07-15"},
            ]
        ),
        proxy,
        continuation,
    )
    assert daily["adjust_factor"].tolist() == pytest.approx([1.0, 2.0, 3.0])
    assert audit_trusted_factor_daily(daily).blockers == []


def test_trusted_factor_continuation_uses_cutoff_absolute_anchor_without_double_counting() -> None:
    proxy = normalize_trusted_proxy_factors(
        pd.DataFrame(
            [
                {"security_id": "SEC-1", "trade_date": "2010-01-04", "adj_factor": 1.0},
                {"security_id": "SEC-1", "trade_date": "2025-01-02", "adj_factor": 4.0},
                {"security_id": "SEC-1", "trade_date": "2026-07-13", "adj_factor": 8.0},
            ]
        )
    )
    batch = pd.DataFrame(
        [
            {
                "security_id": "SEC-1",
                "divid_operate_date": "2026-07-15",
                "adjust_factor": 300.0,
                "symbol_on_date": "600000.SH",
                "provider_symbol": "600000.SH",
            }
        ]
    )
    absolute_history = pd.DataFrame(
        [
            {"security_id": "SEC-1", "divid_operate_date": "2019-01-02", "adjust_factor": 100.0},
            {"security_id": "SEC-1", "divid_operate_date": "2025-01-02", "adjust_factor": 200.0},
            {"security_id": "SEC-1", "divid_operate_date": "2026-07-15", "adjust_factor": 300.0},
        ]
    )

    continuation = continue_trusted_factors_with_baostock(
        proxy,
        batch,
        cutoff="2026-07-13",
        baostock_history=absolute_history,
    )

    assert continuation["trade_date"].tolist() == ["2026-07-15"]
    assert continuation["adjust_factor"].tolist() == pytest.approx([12.0])


def test_trusted_factor_continuation_requires_absolute_history_for_first_post_cutoff_event() -> None:
    proxy = normalize_trusted_proxy_factors(
        pd.DataFrame(
            [{"security_id": "SEC-1", "trade_date": "2026-07-13", "adj_factor": 2.0}]
        )
    )
    batch = pd.DataFrame(
        [{"security_id": "SEC-1", "divid_operate_date": "2026-07-15", "adjust_factor": 3.0}]
    )

    with pytest.raises(ValueError, match="trusted_factor_baostock_ratio_anchor_missing"):
        continue_trusted_factors_with_baostock(
            proxy,
            batch,
            cutoff="2026-07-13",
            baostock_history=batch,
        )


def test_trusted_factor_continuation_supports_post_cutoff_listing_baseline() -> None:
    listing_baseline = pd.DataFrame(
        [
            {
                "security_id": "NEW-SEC",
                "trade_date": "2026-08-01",
                "symbol_on_date": "600999.SH",
                "provider_symbol": "600999.SH",
                "fore_adjust_factor": 1.0,
                "back_adjust_factor": 1.0,
                "adjust_factor": 1.0,
                "baseline_status": "post_cutoff_listing_baseline",
                "source": "baostock.security_master_listing_baseline",
            }
        ]
    )
    batch = pd.DataFrame(
        [
            {
                "security_id": "NEW-SEC",
                "divid_operate_date": "2026-10-10",
                "adjust_factor": 1.5,
            }
        ]
    )
    history = pd.DataFrame(
        [
            {
                "security_id": "NEW-SEC",
                "divid_operate_date": "2026-08-01",
                "adjust_factor": 1.0,
            },
            {
                "security_id": "NEW-SEC",
                "divid_operate_date": "2026-10-10",
                "adjust_factor": 1.5,
            },
        ]
    )

    continuation = continue_trusted_factors_with_baostock(
        listing_baseline,
        batch,
        cutoff="2026-07-13",
        baostock_history=history,
    )

    assert continuation["adjust_factor"].tolist() == pytest.approx([1.5])


def test_default_corrections_config_keeps_fixed_regressions() -> None:
    path = Path(__file__).resolve().parents[2] / "configs" / "qdp_v3_corrections.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == CORRECTIONS_SCHEMA_VERSION
    assert set(payload["regressions"]) == {
        "300114.SZ/302132.SZ",
        "000022.SZ/001872.SZ",
        "000043.SZ/001914.SZ",
        "600076.SH/2024",
    }
