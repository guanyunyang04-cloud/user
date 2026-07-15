from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from quant_data_platform.domains.contracts import ProviderResult
from quant_data_platform.providers import MootdxOnlineProvider
from quant_data_platform.qdp_v3.audit import _audit_dataset_manifest_and_shards, _audit_symbol_history_name_evidence
from quant_data_platform.qdp_v3.constants import (
    DOMAIN_FINANCIAL_QUARTERLY,
    DOMAIN_MARKET_INTRADAY_5M,
    EXPECTED_5M_BAR_ENDS,
    RAW_DAILY_ASTOCK,
    RAW_FINANCIAL_QUARTERLY,
)
from quant_data_platform.qdp_v3.corporate_actions import (
    OFFICIAL_FACTOR_EVIDENCE_COLUMNS,
    arbitrate_adjust_factor_disputes,
    build_share_capital_daily,
    canonicalize_mootdx_xdxr,
    reconstruct_xdxr_reference_prices,
)
from quant_data_platform.qdp_v3.identity import (
    IDENTITY_COLUMNS,
    SYMBOL_HISTORY_COLUMNS,
    SecurityIdentityRegistry,
    board_for_symbol,
)
from quant_data_platform.qdp_v3.intraday import (
    CANONICAL_5M_COLUMNS,
    canonicalize_selected_5m,
    deterministic_stratified_monthly_sample,
    ingest_intraday_5m,
    is_complete_5m_day,
    normalize_provider_5m,
    select_5m_day,
    stable_security_bucket,
)
from quant_data_platform.qdp_v3.intraday_build import _choose_provider_day, _frames_from_bucketed_stage
from quant_data_platform.qdp_v3.datasets import write_partitioned_dataset
from quant_data_platform.qdp_v3.manifest import manifest_sha256, sha256_file
from quant_data_platform.qdp_v3.quality import report_for
from quant_data_platform.qdp_v3.storage import iter_raw_partitions, read_raw_partition, read_raw_receipt, write_raw_partition
from quant_data_platform.qdp_v3.secondary import canonicalize_secondary_domain, ingest_baostock_report_domain
from quant_data_platform.qdp_v3.transforms import (
    build_adjust_factor_daily,
    build_signal_and_open_pit_views,
    derive_daily_domains,
    reconcile_adjust_factor_events,
)


def _workspace(tmp_path: Path) -> Path:
    workspace = tmp_path / "workspace"
    path = workspace / "brain" / "brain_manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"schema_version": 1, "brain_type": "main"}), encoding="utf-8")
    return workspace


def _identity_config() -> Path:
    return Path(__file__).resolve().parents[2] / "configs" / "qdp_v3_symbol_history.json"


def _registry() -> SecurityIdentityRegistry:
    return SecurityIdentityRegistry.from_sources(
        provider_symbols=["302132.SZ", "600000.SH"],
        config_path=_identity_config(),
    )


def _daily_raw(*, code: str = "sz.302132", trade_date: str = "2016-01-04", status: str = "1") -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": [trade_date],
            "code": [code],
            "open": ["10" if status == "1" else "0"],
            "high": ["11" if status == "1" else "0"],
            "low": ["9" if status == "1" else "0"],
            "close": ["10.5" if status == "1" else "0"],
            "preclose": ["10" if status == "1" else "0"],
            "volume": ["100" if status == "1" else "0"],
            "amount": ["1000" if status == "1" else "0"],
            "adjustflag": ["3"],
            "turn": ["1"],
            "tradestatus": [status],
            "pctChg": ["5"],
            "peTTM": ["10"],
            "pbMRQ": ["2"],
            "psTTM": ["3"],
            "pcfNcfTTM": ["4"],
            "isST": ["0"],
        }
    )


def test_provider_current_code_restores_pit_symbol() -> None:
    registry = _registry()

    frames = derive_daily_domains(_daily_raw(), query_date="2016-01-04", identity_registry=registry)

    assert frames.market_daily_raw.loc[0, "provider_symbol"] == "302132.SZ"
    assert frames.market_daily_raw.loc[0, "symbol_on_date"] == "300114.SZ"
    assert registry.symbol_for_date(frames.market_daily_raw.loc[0, "security_id"], "2025-02-17") == "302132.SZ"


def test_official_main_board_code_changes_restore_historical_symbols() -> None:
    registry = SecurityIdentityRegistry.from_sources(
        provider_symbols=["001872.SZ", "001914.SZ"],
        config_path=_identity_config(),
    )

    port_id = registry.security_id_for_provider_symbol("001872.SZ")
    property_id = registry.security_id_for_provider_symbol("001914.SZ")
    assert registry.symbol_for_date(port_id, "2012-09-10") == "000022.SZ"
    assert registry.symbol_for_date(port_id, "2018-12-26") == "001872.SZ"
    assert registry.symbol_for_date(property_id, "2012-09-10") == "000043.SZ"
    assert registry.symbol_for_date(property_id, "2019-12-16") == "001914.SZ"


def test_provider_current_duplicate_prefers_official_symbol_on_date() -> None:
    old_symbol = _daily_raw(code="sz.300114", trade_date="2010-09-08")
    current_symbol = _daily_raw(code="sz.302132", trade_date="2010-09-08")
    current_symbol.loc[0, "turn"] = "1.0001"
    current_symbol.loc[0, "pctChg"] = "5.0001"

    frames = derive_daily_domains(
        pd.concat([old_symbol, current_symbol], ignore_index=True),
        query_date="2010-09-08",
        identity_registry=_registry(),
    )

    assert len(frames.market_daily_raw) == 1
    assert frames.market_daily_raw.loc[0, "provider_symbol"] == "300114.SZ"
    assert frames.market_daily_raw.loc[0, "symbol_on_date"] == "300114.SZ"
    assert len(frames.valuation_daily) == 1
    assert frames.valuation_daily.loc[0, "provider_symbol"] == "300114.SZ"
    assert set(frames.quarantine["resolution_status"]) == {"resolved"}
    assert set(frames.quarantine["resolution_method"]) == {"official_pit_symbol_on_date"}


def test_current_master_name_is_not_backfilled_before_date_local_observation() -> None:
    master = pd.DataFrame(
        {
            "symbol": ["600000.SH"],
            "name": ["CURRENT NAME"],
            "list_date": ["2000-01-01"],
            "delist_date": [""],
            "board": ["1"],
        }
    )
    registry = SecurityIdentityRegistry.from_sources(provider_symbols=["600000.SH"], security_master=master)
    base_history = registry.identity_frames()[1]
    base_history = base_history.loc[base_history["symbol"].eq("600000.SH")].reset_index(drop=True)
    assert base_history.loc[0, "name_on_date"] == ""
    assert base_history.loc[0, "board_on_date"] == "MainBoard"

    observations = pd.DataFrame(
        {
            "symbol": ["600000.SH", "600000.SH", "600000.SH"],
            "trade_date": ["2010-01-04", "2014-12-31", "2015-01-05"],
            "name": ["OLD NAME", "OLD NAME", "NEW NAME"],
        }
    )
    history = registry.with_pit_name_observations(observations).identity_frames()[1]
    history = history.loc[history["symbol"].eq("600000.SH")].reset_index(drop=True)

    assert history[["effective_from", "effective_to", "name_on_date"]].to_dict("records") == [
        {"effective_from": "2000-01-01", "effective_to": "2010-01-03", "name_on_date": ""},
        {"effective_from": "2010-01-04", "effective_to": "2015-01-04", "name_on_date": "OLD NAME"},
        {"effective_from": "2015-01-05", "effective_to": "9999-12-31", "name_on_date": "NEW NAME"},
    ]
    assert "query_all_stock" in history.iloc[-1]["evidence_source"]


def test_board_and_terminal_out_date_semantics_are_pit_safe() -> None:
    assert board_for_symbol("689009.SH") == "STAR"
    master = pd.DataFrame(
        {
            "symbol": ["600001.SH"],
            "name": ["DELISTED"],
            "list_date": ["1998-01-22"],
            "delist_date": ["2009-12-29"],
            "board": ["1"],
        }
    )
    history = SecurityIdentityRegistry.from_sources(
        provider_symbols=["600001.SH"], security_master=master
    ).identity_frames()[1]
    assert history.loc[0, "effective_to"] == "2009-12-29"
    assert SecurityIdentityRegistry.from_sources(
        provider_symbols=["302132.SZ"], config_path=_identity_config()
    ).symbol_for_date("QDP-CN-SZSE-AVICCAC-20100827", "2025-02-17") == "302132.SZ"


def test_semantic_audit_rejects_current_name_backfill(tmp_path: Path) -> None:
    history = pd.DataFrame(
        {
            "security_id": ["S1"],
            "symbol": ["600000.SH"],
            "effective_from": ["2000-01-01"],
            "effective_to": ["9999-12-31"],
            "name_on_date": ["CURRENT NAME"],
            "board_on_date": ["MainBoard"],
            "evidence_source": ["baostock.query_stock_basic"],
            "official_document_hash": [""],
        }
    )
    candidate = type("Candidate", (), {"raw_partitions": []})()

    findings = _audit_symbol_history_name_evidence(history, candidate, root=tmp_path)

    assert any(item["code"] == "symbol_history_name_without_pit_evidence" for item in findings)


def test_multi_symbol_identity_requires_official_document_evidence() -> None:
    identity = pd.DataFrame(
        [
            {
                "security_id": "S1",
                "official_org_id": "",
                "issuer_name": "issuer",
                "exchange": "SZSE",
                "list_date": "2020-01-01",
                "current_symbol": "000002.SZ",
                "identity_source": "manual",
            }
        ],
        columns=IDENTITY_COLUMNS,
    )
    history = pd.DataFrame(
        [
            {
                "security_id": "S1",
                "symbol": "000001.SZ",
                "effective_from": "2020-01-01",
                "effective_to": "2024-12-31",
                "name_on_date": "old",
                "board_on_date": "MainBoard",
                "evidence_source": "",
                "official_document_hash": "",
            },
            {
                "security_id": "S1",
                "symbol": "000002.SZ",
                "effective_from": "2025-01-01",
                "effective_to": "9999-12-31",
                "name_on_date": "new",
                "board_on_date": "MainBoard",
                "evidence_source": "https://example.invalid/official.pdf",
                "official_document_hash": "a" * 64,
            },
        ],
        columns=SYMBOL_HISTORY_COLUMNS,
    )

    with pytest.raises(ValueError, match="multi_symbol_identity_official_evidence_missing"):
        SecurityIdentityRegistry(identity, history)


def test_suspension_does_not_manufacture_zero_prices() -> None:
    registry = _registry()

    frames = derive_daily_domains(_daily_raw(code="sh.600000", trade_date="2026-01-05", status="0"), query_date="2026-01-05", identity_registry=registry)

    assert frames.market_daily_raw[["open", "high", "low", "close", "preclose"]].isna().all(axis=None)
    assert bool(frames.security_status_daily.loc[0, "is_suspended"])


def test_pit_signal_and_d1_open_views_use_the_correct_date() -> None:
    market = pd.DataFrame(
        {
            "security_id": ["S1", "S1"],
            "trade_date": ["2026-01-05", "2026-01-06"],
            "symbol_on_date": ["600000.SH", "600000.SH"],
            "open": [10.0, 11.0],
            "tradestatus": ["1", "1"],
        }
    )
    status = pd.DataFrame(
        {
            "security_id": ["S1", "S1"],
            "trade_date": ["2026-01-05", "2026-01-06"],
            "symbol_on_date": ["600000.SH", "600000.SH"],
            "list_status": ["listed", "listed"],
            "is_st": [False, True],
        }
    )

    eligible, tradable = build_signal_and_open_pit_views(market, status)

    assert bool(eligible.loc[0, "is_eligible_signal"])
    assert not bool(eligible.loc[1, "is_eligible_signal"])
    assert tradable.loc[0, "next_trade_date"] == "2026-01-06"
    assert tradable.loc[0, "open_d1"] == pytest.approx(11.0)
    assert bool(tradable.loc[0, "tradable_open_d1"])
    assert tradable.loc[1, "tradability_reason"] == "next_observation_unavailable"


def _event_frame(method: str, value: float = 1.25) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "security_id": ["QDP-CN-SSE-600000"],
            "divid_operate_date": ["2026-06-26"],
            "symbol_on_date": ["600000.SH"],
            "provider_symbol": ["600000.SH"],
            "fore_adjust_factor": [value],
            "back_adjust_factor": [value * 2],
            "adjust_factor": [value * 3],
            "query_date": ["2026-06-26" if method == "date_batch" else ""],
            "source_method": [method],
            "verification_status": [f"{method}_unreconciled"],
            "identity_mapping_status": ["mapped"],
            "source": ["baostock"],
        }
    )


def test_factor_dual_path_accepts_only_matching_keys_and_values() -> None:
    accepted, disputed, metrics = reconcile_adjust_factor_events(
        _event_frame("date_batch"),
        _event_frame("symbol_history", value=1.25 * (1 + 5e-7)),
        comparable_start="2010-01-01",
        comparable_end="2026-12-31",
    )
    assert len(accepted) == 1
    assert disputed.empty
    assert accepted.loc[0, "verification_status"] == "verified_dual_path"
    assert metrics["verified_event_count"] == 1

    accepted, disputed, _ = reconcile_adjust_factor_events(
        _event_frame("date_batch"),
        _event_frame("symbol_history", value=2.0),
        comparable_start="2010-01-01",
        comparable_end="2026-12-31",
    )
    assert accepted.empty
    assert disputed.loc[0, "dispute_reason"] == "factor_values_mismatch"


def test_xdxr_is_lossless_pit_evidence_but_not_factor_authority() -> None:
    raw = pd.DataFrame(
        {
            "year": [2026, 2026],
            "month": [6, 7],
            "day": [26, 1],
            "category": [1, 5],
            "name": ["除权除息", "股本变化"],
            "fenhong": [1.0, np.nan],
            "peigujia": [4.0, np.nan],
            "songzhuangu": [2.0, np.nan],
            "peigu": [1.0, np.nan],
            "panqianliutong": [np.nan, 100.0],
            "panhouliutong": [np.nan, 120.0],
            "qianzongguben": [np.nan, 200.0],
            "houzongguben": [np.nan, 220.0],
        }
    )
    actions, capitals, conflicts = canonicalize_mootdx_xdxr(
        [("600000.SH", raw)],
        identity_registry=_registry(),
    )

    assert conflicts.empty
    assert actions.loc[0, "verification_status"] == "xdxr_unofficial_evidence"
    assert actions.loc[0, "bonus_transfer_per_10"] == pytest.approx(2.0)
    assert capitals.loc[0, "total_share_after"] == pytest.approx(2_200_000.0)


def test_official_factor_arbitration_requires_matching_values() -> None:
    _, disputed, _ = reconcile_adjust_factor_events(
        _event_frame("date_batch", value=1.25),
        _event_frame("symbol_history", value=2.0),
        comparable_start="2010-01-01",
        comparable_end="2026-12-31",
    )
    xdxr = pd.DataFrame({"security_id": ["QDP-CN-SSE-600000"], "event_date": ["2026-06-26"]})
    official = pd.DataFrame(
        [
            {
                "security_id": "QDP-CN-SSE-600000",
                "event_date": "2026-06-26",
                "provider_symbol": "600000.SH",
                "official_source": "sse",
                "official_document_url": "https://example.invalid/official.pdf",
                "official_document_sha256": "a" * 64,
                "fore_adjust_factor": 1.25,
                "back_adjust_factor": 2.5,
                "adjust_factor": 3.75,
                "notes": "fixture",
            }
        ],
        columns=OFFICIAL_FACTOR_EVIDENCE_COLUMNS,
    )

    accepted, remaining, metrics = arbitrate_adjust_factor_disputes(
        disputed,
        xdxr_events=xdxr,
        official_evidence=official,
    )

    assert remaining.empty
    assert accepted.loc[0, "verification_status"] == "xdxr_official_arbitrated"
    assert accepted.loc[0, "source_method"] == "official_arbitration+date_batch"
    assert metrics["arbitrated_count"] == 1

    accepted, remaining, _ = arbitrate_adjust_factor_disputes(
        disputed,
        xdxr_events=xdxr,
        official_evidence=official.assign(adjust_factor=9.0),
    )
    assert accepted.empty
    assert remaining.loc[0, "arbitration_status"] == "official_values_match_no_unique_candidate"


def test_share_capital_has_no_future_backfill_and_reference_price_is_checked() -> None:
    market = pd.DataFrame(
        {
            "security_id": ["S1", "S1", "S1"],
            "trade_date": ["2026-06-25", "2026-06-26", "2026-07-01"],
            "symbol_on_date": ["600000.SH"] * 3,
            "close": [10.0, 8.0, 8.2],
            "preclose": [9.9, 8.0, 8.0],
        }
    )
    capital = pd.DataFrame(
        {
            "security_id": ["S1"],
            "event_date": ["2026-07-01"],
            "total_share_after": [2_000_000.0],
            "float_share_after": [1_500_000.0],
        }
    )
    daily = build_share_capital_daily(market, capital)
    assert pd.isna(daily.loc[daily["trade_date"].eq("2026-06-26"), "total_share"]).all()
    assert daily.loc[daily["trade_date"].eq("2026-07-01"), "total_share"].iloc[0] == pytest.approx(2_000_000.0)

    action = pd.DataFrame(
        {
            "security_id": ["S1"],
            "event_date": ["2026-06-26"],
            "cash_dividend_per_10": [0.0],
            "rights_share_per_10": [0.0],
            "bonus_transfer_per_10": [2.5],
            "rights_price": [np.nan],
        }
    )
    proof = reconstruct_xdxr_reference_prices(market, action)
    assert proof.loc[0, "reconstructed_reference"] == pytest.approx(8.0)
    assert bool(proof.loc[0, "within_one_tick"])


def test_report_pit_uses_next_exchange_day_and_keeps_unknown_unavailable(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    report = pd.DataFrame(
        {
            "symbol": ["302132.SZ", "302132.SZ"],
            "trade_date": ["2016-01-08", "2016-03-31"],
            "report_date": ["2015-12-31", "2016-03-31"],
            "fiscal_year": [2015, 2016],
            "fiscal_quarter": [4, 1],
            "publish_date": ["2016-01-08", "2016-03-31"],
            "lag_policy": ["publish_date_plus_1d_in_features", "conservative_report_date_plus_90bd_plus_1d_in_features"],
            "source": ["baostock", "baostock"],
        }
    )
    ref, _ = write_raw_partition(
        raw_domain=RAW_FINANCIAL_QUARTERLY,
        partition_field="provider_symbol",
        partition_value="302132.SZ",
        frame=report,
        receipt={"quality_tier": "provisional"},
        workspace_root=workspace,
    )
    calendar = pd.DataFrame(
        {
            "trade_date": ["2016-01-08", "2016-01-11", "2016-03-31", "2016-04-01"],
            "is_open": [True, True, True, True],
        }
    )

    canonical, conflicts = canonicalize_secondary_domain(
        domain=DOMAIN_FINANCIAL_QUARTERLY,
        refs=[ref],
        identity_registry=_registry(),
        calendar=calendar,
    )

    assert conflicts.empty
    assert canonical.loc[0, "symbol_on_date"] == "300114.SZ"
    assert canonical.loc[0, "available_date"] == "2016-01-11"
    assert canonical.loc[1, "available_date"] == ""
    assert canonical.loc[1, "availability_status"] == "publish_date_unproven"


def _bars(source: str, *, complete: bool = True, trade_date: str = "2026-06-26") -> pd.DataFrame:
    ends = EXPECTED_5M_BAR_ENDS if complete else EXPECTED_5M_BAR_ENDS[:-1]
    return pd.DataFrame(
        {
            "provider_symbol": ["600000.SH"] * len(ends),
            "trade_date": [trade_date] * len(ends),
            "bar_end": list(ends),
            "open": [10.0] * len(ends),
            "high": [10.1] * len(ends),
            "low": [9.9] * len(ends),
            "close": [10.0] * len(ends),
            "volume": [100.0] * len(ends),
            "amount": [1000.0] * len(ends),
            "source": [source] * len(ends),
        }
    )


def test_intraday_stage_coalesces_natural_year_security_bucket_and_full_audit_accepts_it(
    tmp_path: Path,
) -> None:
    first_by_bucket: dict[int, str] = {}
    security_ids: tuple[str, str] | None = None
    bucket = -1
    for index in range(1024):
        security_id = f"security_{index:04d}"
        bucket = stable_security_bucket(security_id)
        if bucket in first_by_bucket:
            security_ids = (first_by_bucket[bucket], security_id)
            break
        first_by_bucket[bucket] = security_id
    assert security_ids is not None

    staged: list[tuple[str, int, Path]] = []
    for index, security_id in enumerate(security_ids):
        frame = _bars("tushare_proxy", trade_date="2026-07-13").assign(
            security_id=security_id,
            symbol_on_date=f"60000{index}.SH",
            provider_symbol=f"60000{index}.SH",
            quality_tier="strict",
            source_selection_reason="unit",
            identity_mapping_status="mapped",
        )
        frame = frame.loc[:, CANONICAL_5M_COLUMNS]
        path = tmp_path / f"{security_id}.parquet"
        frame.to_parquet(path, index=False)
        staged.append(("2026", bucket, path))

    partition_frames = list(_frames_from_bucketed_stage(staged))
    assert len(partition_frames) == 1
    shard_key, partition_value, combined = partition_frames[0]
    assert shard_key == f"2026_b{bucket:02d}"
    assert partition_value == f"2026/bucket={bucket:02d}"
    assert set(combined["security_id"]) == set(security_ids)

    root = tmp_path / "qdp_v3"
    manifest = write_partitioned_dataset(
        root=root,
        domain=DOMAIN_MARKET_INTRADAY_5M,
        partition_frames=partition_frames,
        layer="canonical_tiered",
        frequency="5m",
        primary_key=["security_id", "trade_date", "bar_end"],
        quality_report=report_for(DOMAIN_MARKET_INTRADAY_5M, []),
        partitioning="natural_year_security_bucket",
    )
    manifest_path = root / "datasets" / DOMAIN_MARKET_INTRADAY_5M / manifest.dataset_id / "dataset.json"
    findings, audited_manifest, _ = _audit_dataset_manifest_and_shards(
        root=root,
        domain=DOMAIN_MARKET_INTRADAY_5M,
        dataset_id=manifest.dataset_id,
        expected_manifest_sha=manifest_sha256(manifest_path),
        full=True,
    )

    assert audited_manifest is not None
    assert len(audited_manifest.shards) == 1
    assert "intraday_shard_partition_contract_invalid" not in {item["code"] for item in findings}
    assert "intraday_cross_shard_partition_duplicate" not in {item["code"] for item in findings}


def test_5m_source_policy_never_stitches() -> None:
    selected, evidence = select_5m_day(mootdx=_bars("mootdx", complete=False), baostock=_bars("baostock"), trade_date="2026-06-26")
    assert len(selected) == 48
    assert set(selected["source"]) == {"baostock"}
    assert evidence["status"] == "strict"

    selected, evidence = select_5m_day(mootdx=_bars("mootdx", complete=False), baostock=_bars("baostock", complete=False), trade_date="2026-06-26")
    assert selected.empty
    assert evidence["status"] == "quarantined"


def test_5m_quality_tier_is_evidence_driven_before_2020() -> None:
    trade_date = "2010-01-04"
    selected, evidence = select_5m_day(
        mootdx=_bars("mootdx", trade_date=trade_date),
        baostock=_bars("baostock", complete=False, trade_date=trade_date),
        trade_date=trade_date,
    )

    assert len(selected) == 48
    assert selected["quality_tier"].eq("strict").all()
    assert evidence["status"] == "strict"


def test_5m_complete_mootdx_survives_unavailable_baostock_audit() -> None:
    selected, evidence = select_5m_day(
        mootdx=_bars("mootdx"),
        baostock=_bars("baostock", complete=False),
        trade_date="2026-06-26",
        compare_sources=True,
    )

    assert len(selected) == 48
    assert evidence["status"] == "strict"
    assert evidence["comparison"]["reason"] == "baostock_comparison_unavailable"


def test_provider_code_restatement_conflict_is_quarantined_before_pit_preference() -> None:
    registry = _registry()
    old_code = _bars("tushare_proxy", trade_date="2016-01-04")
    old_code["provider_symbol"] = "300114.SZ"
    new_code = old_code.copy()
    new_code["provider_symbol"] = "302132.SZ"
    new_code.loc[new_code.index[0], "close"] = 10.02

    selected, reason = _choose_provider_day(
        pd.concat([old_code, new_code], ignore_index=True),
        security_id=str(registry.security_id_for_provider_symbol("300114.SZ")),
        trade_date="2016-01-04",
        registry=registry,
    )

    assert selected.empty
    assert reason == "provider_code_restatement_value_conflict"


@pytest.mark.parametrize(
    ("provider_time", "expected"),
    [("20260522093500000", "09:35"), ("093500000", "09:35"), ("2026-05-22 09:35:00", "09:35")],
)
def test_5m_bar_end_parses_full_and_time_only_baostock_encodings(provider_time: str, expected: str) -> None:
    raw = pd.DataFrame(
        {
            "symbol": ["600000.SH"],
            "trade_date": ["2026-05-22"],
            "bar_time": [provider_time],
            "open": [10.0],
            "high": [10.1],
            "low": [9.9],
            "close": [10.0],
            "volume": [100.0],
            "amount": [1000.0],
        }
    )

    normalized = normalize_provider_5m(raw, provider_symbol="600000.SH", source="baostock")

    assert normalized.loc[0, "bar_end"] == expected


def test_5m_identity_mapping_and_stratified_sample() -> None:
    selected = _bars("mootdx")
    selected["quality_tier"] = "strict"
    selected["source_selection_reason"] = "complete_mootdx_preferred"

    canonical, quarantine = canonicalize_selected_5m(selected, identity_registry=_registry())

    assert quarantine.empty
    assert is_complete_5m_day(canonical)
    universe = pd.DataFrame(
        {
            "provider_symbol": ["600000.SH", "600001.SH", "000001.SZ", "000002.SZ"],
            "exchange": ["SH", "SH", "SZ", "SZ"],
            "liquidity": [1, 100, 2, 200],
        }
    )
    sample = deterministic_stratified_monthly_sample(universe, month="2026-06", count=4)
    assert sample == set(universe["provider_symbol"])


class _FakeIntradayProvider:
    def __init__(self, dates: list[str], *, source: str, conflict_symbol: str = "") -> None:
        self.dates = dates
        self.source = source
        self.conflict_symbol = conflict_symbol
        self.calls = []

    def fetch_domain(self, request):
        self.calls.append(request)
        symbol = str(request.symbols[0])
        frames = []
        for trade_date in self.dates:
            if request.start_date <= trade_date <= request.end_date:
                day = _bars(self.source, trade_date=trade_date).assign(provider_symbol=symbol)
                if symbol == self.conflict_symbol:
                    day.loc[day.index[0], "open"] += 0.02
                frames.append(day)
        return ProviderResult(provider=self.source, data=pd.concat(frames, ignore_index=True) if frames else pd.DataFrame())


def test_fast_5m_ingest_makes_zero_baostock_requests_when_mootdx_is_complete(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    dates = ["2026-07-13", "2026-07-14"]
    mootdx = _FakeIntradayProvider(dates, source="mootdx")
    baostock = _FakeIntradayProvider(dates, source="baostock")

    result = ingest_intraday_5m(
        symbols=["600000.SH"],
        trade_dates=dates,
        workspace_root=workspace,
        mootdx_provider=mootdx,
        baostock_provider=baostock,
        refresh=True,
    )

    assert result["status"] == "completed"
    assert result["mode"] == "trusted_source_fast"
    assert result["baostock_fallback_stock_day_count"] == 0
    assert result["baostock_fallback_request_count"] == 0
    assert baostock.calls == []


def test_fast_5m_ingest_requests_baostock_only_for_incomplete_mootdx_dates(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    dates = ["2026-07-13", "2026-07-14"]

    class PartiallyIncompleteMootdx(_FakeIntradayProvider):
        def fetch_domain(self, request):
            result = super().fetch_domain(request)
            data = result.data
            incomplete = data["trade_date"].eq("2026-07-14")
            drop_index = data.loc[incomplete].index[-1]
            return ProviderResult(provider="mootdx", data=data.drop(index=drop_index), error_report=[])

    mootdx = PartiallyIncompleteMootdx(dates, source="mootdx")
    baostock = _FakeIntradayProvider(dates, source="baostock")
    result = ingest_intraday_5m(
        symbols=["600000.SH"],
        trade_dates=dates,
        workspace_root=workspace,
        mootdx_provider=mootdx,
        baostock_provider=baostock,
        refresh=True,
    )

    assert result["status"] == "completed"
    assert result["strict_stock_day_count"] == 2
    assert result["baostock_fallback_stock_day_count"] == 1
    assert result["baostock_fallback_request_count"] == 1
    assert [(call.start_date, call.end_date) for call in baostock.calls] == [("2026-07-14", "2026-07-14")]
    selected_ref = iter_raw_partitions("qdp_intraday_5m_selected_raw", workspace_root=workspace)[0]
    selected = read_raw_partition(selected_ref)
    source_by_date = selected.groupby("trade_date")["source"].unique().map(set).to_dict()
    assert source_by_date == {"2026-07-13": {"mootdx"}, "2026-07-14": {"baostock"}}


def test_5m_ingest_uses_symbol_month_tasks_pit_trading_days_and_stratum_escalation(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    dates = ["2026-06-25", "2026-06-26"]
    trading_symbols = ["600000.SH", "600001.SH"]
    suspended_symbol = "600002.SH"
    for trade_date in dates:
        daily = pd.DataFrame(
            {
                "date": [trade_date] * 3,
                "code": ["sh.600000", "sh.600001", "sh.600002"],
                "tradestatus": ["1", "1", "0"],
                "open": [10.0, 10.0, np.nan],
                "close": [10.0, 10.0, np.nan],
            }
        )
        write_raw_partition(
            raw_domain=RAW_DAILY_ASTOCK,
            partition_field="query_date",
            partition_value=trade_date,
            frame=daily,
            receipt={"quality_tier": "strict"},
            workspace_root=workspace,
        )
    universe = pd.DataFrame(
        {
            "provider_symbol": [*trading_symbols, suspended_symbol],
            "exchange": ["SH", "SH", "SH"],
            "liquidity": [np.nan, np.nan, np.nan],
        }
    )
    sampled = next(iter(deterministic_stratified_monthly_sample(universe.iloc[:2], month="2026-06", count=1)))
    mootdx = _FakeIntradayProvider(dates, source="mootdx")
    baostock = _FakeIntradayProvider(dates, source="baostock", conflict_symbol=sampled)

    result = ingest_intraday_5m(
        symbols=[*trading_symbols, suspended_symbol],
        trade_dates=dates,
        workspace_root=workspace,
        mootdx_provider=mootdx,
        baostock_provider=baostock,
        comparison_sample_count=1,
        sampling_universe=universe,
        audit_cross_sources=True,
    )

    assert result["status"] == "partial"
    assert result["stock_day_count"] == 4
    assert result["excluded_nontrading_stock_day_count"] == 2
    assert any(item["escalated"] and item["stratum"] == "SH|UNKNOWN" for item in result["escalated_strata"])
    assert {call.symbols[0] for call in baostock.calls} == set(trading_symbols)
    assert suspended_symbol not in {call.symbols[0] for call in mootdx.calls}
    assert all(call.start_date[:7] == call.end_date[:7] == "2026-06" for call in [*mootdx.calls, *baostock.calls])
    assert result["strict_stock_day_count"] == 2
    assert result["quarantined_stock_day_count"] == 2


def test_5m_ingest_prefetches_one_mootdx_range_and_splits_monthly(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    dates = ["2026-01-05", "2026-02-02"]
    instances = []

    class Client:
        def __init__(self) -> None:
            self.calls = []
            self.closed = 0
            instances.append(self)

        def bars(self, *, symbol: str, frequency: int, start: int, offset: int) -> pd.DataFrame:
            self.calls.append((symbol, frequency, start, offset))
            return pd.DataFrame(
                [
                    {
                        "datetime": f"{trade_date} {bar_end}:00",
                        "open": 10.0,
                        "high": 10.1,
                        "low": 9.9,
                        "close": 10.0,
                        "volume": 100.0,
                        "amount": 1000.0,
                    }
                    for trade_date in dates
                    for bar_end in EXPECTED_5M_BAR_ENDS
                ]
            )

        def close(self) -> None:
            self.closed += 1

    mootdx = MootdxOnlineProvider(_client_factory=Client)
    baostock = _FakeIntradayProvider(dates, source="baostock")
    first = ingest_intraday_5m(
        symbols=["600000.SH"],
        trade_dates=dates,
        workspace_root=workspace,
        mootdx_provider=mootdx,
        baostock_provider=baostock,
        comparison_sample_count=0,
    )

    assert first["status"] == "completed"
    assert first["mootdx_download_strategy"] == "single_symbol_range_split_monthly"
    assert first["range_prefetch_network_request_count"] == 1
    assert first["range_prefetch_written_partition_count"] == 2
    assert len(instances) == 1
    assert len(instances[0].calls) == 1
    assert baostock.calls == []

    second = ingest_intraday_5m(
        symbols=["600000.SH"],
        trade_dates=dates,
        workspace_root=workspace,
        mootdx_provider=mootdx,
        baostock_provider=baostock,
        comparison_sample_count=0,
    )
    assert second["range_prefetch_network_request_count"] == 0
    assert second["range_prefetch_reused_partition_count"] == 2
    assert len(instances[0].calls) == 1
    mootdx.close()
    assert instances[0].closed == 1


def test_5m_ingest_prefetches_one_baostock_fallback_range(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    dates = ["2026-01-05", "2026-02-02"]

    class MissingMootdx:
        def fetch_domain(self, request):
            return ProviderResult(
                provider="mootdx",
                data=pd.DataFrame(),
                error_report=[{"symbol": request.symbols[0], "message": "unavailable"}],
            )

    class RangeBaostock:
        supports_symbol_range_prefetch = True

        def __init__(self) -> None:
            self.calls = []

        def fetch_domain(self, request):
            self.calls.append(request)
            frames = [
                _bars("baostock", trade_date=trade_date).assign(provider_symbol=request.symbols[0])
                for trade_date in dates
                if request.start_date <= trade_date <= request.end_date
            ]
            return ProviderResult(provider="baostock", data=pd.concat(frames, ignore_index=True), error_report=[])

    baostock = RangeBaostock()
    result = ingest_intraday_5m(
        symbols=["600000.SH"],
        trade_dates=dates,
        workspace_root=workspace,
        mootdx_provider=MissingMootdx(),
        baostock_provider=baostock,
        comparison_sample_count=0,
    )

    assert result["status"] == "completed"
    assert result["baostock_download_strategy"] == "fallback_dates_only"
    assert result["baostock_range_prefetch_network_request_count"] == 1
    assert result["baostock_range_prefetch_written_partition_count"] == 2
    assert len(baostock.calls) == 1
    assert baostock.calls[0].start_date == dates[0]
    assert baostock.calls[0].end_date == dates[-1]


def test_financial_ingest_clips_prelisting_history_and_batches_by_quarter(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)

    class Provider:
        def __init__(self) -> None:
            self.calls = []

        def fetch_domain(self, request):
            self.calls.append(request)
            return ProviderResult(provider="baostock", data=pd.DataFrame(), error_report=[])

    provider = Provider()
    result = ingest_baostock_report_domain(
        domain=DOMAIN_FINANCIAL_QUARTERLY,
        symbols=["600000.SH", "600001.SH", "600002.SH"],
        start_date="2010-01-01",
        end_date="2026-06-26",
        workspace_root=workspace,
        provider=provider,
        chunk_size=8,
        symbol_lifecycle_ranges={
            "600000.SH": ("2025-01-10", ""),
            "600001.SH": ("2025-02-10", ""),
            "600002.SH": ("2010-01-01", ""),
        },
    )

    assert result["status"] == "completed"
    assert result["provider_query_chunk_count"] == 2
    assert result["lifecycle_clipped_symbol_count"] == 2
    grouped = {tuple(call.symbols): call.start_date for call in provider.calls}
    assert grouped[("600000.SH", "600001.SH")] == "2023-07-01"
    assert grouped[("600002.SH",)] == "2010-01-01"


def test_raw_partition_is_idempotent_and_preserves_revision(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    first = pd.DataFrame({"value": [1]})
    ref1, created1 = write_raw_partition(raw_domain="unit_raw", partition_value="2026-01-05", frame=first, receipt={"quality_tier": "strict"}, workspace_root=workspace)
    ref2, created2 = write_raw_partition(raw_domain="unit_raw", partition_value="2026-01-05", frame=first, receipt={"quality_tier": "strict"}, workspace_root=workspace)
    ref3, created3 = write_raw_partition(raw_domain="unit_raw", partition_value="2026-01-05", frame=pd.DataFrame({"value": [2]}), receipt={"quality_tier": "strict"}, workspace_root=workspace)
    ref4, created4 = write_raw_partition(
        raw_domain="unit_raw",
        partition_value="2026-01-05",
        frame=pd.DataFrame({"value": [2]}),
        receipt={
            "quality_tier": "quarantined",
            "quality_report": {"blockers": [{"code": "new_semantic_blocker"}]},
        },
        workspace_root=workspace,
    )

    assert created1 and not created2 and created3 and not created4
    assert ref1.content_sha256 == ref2.content_sha256
    assert ref3.revision_of == ref1.content_sha256
    assert ref4.content_sha256 == ref3.content_sha256
    latest = iter_raw_partitions("unit_raw", workspace_root=workspace)[0]
    assert latest.quality_tier == "quarantined"
    assert read_raw_receipt(latest)["quality_report"]["blockers"][0]["code"] == "new_semantic_blocker"
    assert len(list(ref3.receipt_path.parent.glob("quality_assessments/*.json"))) == 2
    assert read_raw_partition(ref1).loc[0, "value"] == 1
    assert read_raw_partition(ref3).loc[0, "value"] == 2


def test_raw_partition_recovers_payload_only_version_after_hard_stop(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    frame = pd.DataFrame({"value": [1, 2, 3]})
    first, created = write_raw_partition(
        raw_domain="unit_recover_raw",
        partition_value="2026-01-05",
        frame=frame,
        receipt={"quality_tier": "strict", "provider": "unit"},
        workspace_root=workspace,
    )
    assert created
    payload_sha = sha256_file(first.payload_path)
    latest_path = first.payload_path.parents[2] / "latest.json"
    first.receipt_path.unlink()
    latest_path.unlink()

    recovered, recovered_created = write_raw_partition(
        raw_domain="unit_recover_raw",
        partition_value="2026-01-05",
        frame=frame,
        receipt={"quality_tier": "strict", "provider": "unit"},
        workspace_root=workspace,
    )

    assert recovered_created
    assert recovered.content_sha256 == first.content_sha256
    assert sha256_file(recovered.payload_path) == payload_sha
    assert read_raw_receipt(recovered)["provider"] == "unit"
    assert read_raw_partition(recovered)["value"].tolist() == [1, 2, 3]


def test_raw_content_hash_uses_persisted_parquet_representation(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    frame = pd.DataFrame(
        {
            "text": pd.Series(["value"], dtype="string"),
            "nullable_integer": pd.Series([1], dtype="Int64"),
        }
    )

    ref, created = write_raw_partition(
        raw_domain="unit_roundtrip_raw",
        partition_value="2026-01-05",
        frame=frame,
        receipt={"quality_tier": "strict"},
        workspace_root=workspace,
    )

    assert created
    assert read_raw_partition(ref).to_dict("records") == [{"text": "value", "nullable_integer": 1}]
