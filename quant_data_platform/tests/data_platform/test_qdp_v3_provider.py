from __future__ import annotations

import pandas as pd
import pytest

from quant_data_platform.domains.contracts import DataDomain, DatePartitionFetchRequest, DomainFetchRequest
from quant_data_platform.providers import (
    BAOSTOCK_BULK_PER_PAGE_COUNT,
    BaostockProvider,
    assert_baostock_batch_runtime,
    _baostock_bulk_adjust_factor_event_frame,
    _baostock_bulk_query_to_frame,
    _quiet_baostock_call,
    MootdxOnlineProvider,
)
from quant_data_platform.qdp_v3.compatibility import _compare_fixture
from quant_data_platform.qdp_v3.constants import BAOSTOCK_DAILY_FIELDS
from quant_data_platform.qdp_v3.quality import audit_baostock_daily_raw


class BulkResult:
    def __init__(self, size: int, *, per_page_count: int = BAOSTOCK_BULK_PER_PAGE_COUNT) -> None:
        self.error_code = "0"
        self.error_msg = ""
        self.fields = ["value"]
        self.data = [[str(index)] for index in range(size)]
        self.per_page_count = per_page_count
        self.next_called = False

    def next(self) -> bool:
        self.next_called = True
        raise AssertionError("bulk parser must not call next")


def test_baostock_console_noise_is_suppressed(capsys: pytest.CaptureFixture[str]) -> None:
    def noisy_operation() -> str:
        print("login success!")
        return "ok"

    assert _quiet_baostock_call(noisy_operation) == "ok"
    assert capsys.readouterr().out == ""


def test_baostock_runtime_rejects_recorded_wrong_wheel_hash(monkeypatch: pytest.MonkeyPatch) -> None:
    from quant_data_platform import providers

    class Distribution:
        @staticmethod
        def read_text(_name: str) -> str:
            return '{"archive_info":{"hashes":{"sha256":"deadbeef"}}}'

    monkeypatch.setattr(providers, "baostock_runtime_version", lambda: "0.9.3")
    monkeypatch.setattr(providers, "package_distribution", lambda _name: Distribution())

    with pytest.raises(RuntimeError, match="wheel_hash_mismatch"):
        assert_baostock_batch_runtime()


def _daily_raw(symbols: list[str], trade_date: str = "2026-06-26") -> pd.DataFrame:
    defaults = {field: "1" for field in BAOSTOCK_DAILY_FIELDS}
    defaults.update({"date": trade_date, "adjustflag": "3", "tradestatus": "1", "isST": "0"})
    return pd.DataFrame([{**defaults, "code": symbol} for symbol in symbols], columns=BAOSTOCK_DAILY_FIELDS)


@pytest.mark.parametrize("size", [1999, 2000, 2001])
def test_bulk_parser_never_uses_ordinary_pagination(size: int) -> None:
    result = BulkResult(size)

    frame = _baostock_bulk_query_to_frame(result, "unit")

    assert len(frame) == size
    assert not result.next_called


def test_bulk_parser_blocks_capacity_and_wrong_protocol() -> None:
    with pytest.raises(RuntimeError, match="potential_truncation"):
        _baostock_bulk_query_to_frame(BulkResult(20_000), "unit")
    with pytest.raises(RuntimeError, match="unexpected_per_page_count"):
        _baostock_bulk_query_to_frame(BulkResult(1, per_page_count=2_000), "unit")
    malformed = BulkResult(1)
    malformed.fields = ["a", "b"]
    with pytest.raises(RuntimeError, match="field_width_mismatch"):
        _baostock_bulk_query_to_frame(malformed, "unit")


def test_bulk_parser_does_not_treat_provider_error_as_empty() -> None:
    result = BulkResult(0)
    result.error_code = "1001"
    result.error_msg = "broken"

    with pytest.raises(RuntimeError, match="query_error:1001"):
        _baostock_bulk_query_to_frame(result, "unit")


@pytest.mark.parametrize("alias", ["adjustFacto", "adjustFactor", "adjust_factor"])
def test_factor_batch_aliases_are_explicit(alias: str) -> None:
    raw = pd.DataFrame(
        {
            "code": ["sh.600000"],
            "dividOperateDate": ["2026-06-26"],
            "foreAdjustFactor": ["1.2"],
            "backAdjustFactor": ["3.4"],
            alias: ["5.6"],
        }
    )

    frame = _baostock_bulk_adjust_factor_event_frame(raw, query_date="2026-06-26")

    assert frame.loc[0, "adjust_factor"] == pytest.approx(5.6)
    assert frame.loc[0, "provider_symbol"] == "600000.SH"


def test_factor_batch_rejects_wrong_event_date_and_alias_conflict() -> None:
    raw = pd.DataFrame(
        {
            "code": ["sh.600000"],
            "dividOperateDate": ["2026-06-25"],
            "foreAdjustFactor": ["1"],
            "backAdjustFactor": ["1"],
            "adjustFacto": ["1"],
        }
    )
    with pytest.raises(RuntimeError, match="event_date_mismatch"):
        _baostock_bulk_adjust_factor_event_frame(raw, query_date="2026-06-26")

    raw["dividOperateDate"] = "2026-06-26"
    raw["adjustFactor"] = "2"
    with pytest.raises(RuntimeError, match="conflicting_alias_fields"):
        _baostock_bulk_adjust_factor_event_frame(raw, query_date="2026-06-26")


def test_date_partition_request_contract() -> None:
    request = DatePartitionFetchRequest(
        domain=DataDomain.MARKET_DAILY,
        trade_date="2026-06-26",
        universe_kind="all_a",
        fetch_mode="date_snapshot",
    ).normalized()
    assert request.trade_date == "2026-06-26"

    with pytest.raises(ValueError, match="date_events does not support domain"):
        DatePartitionFetchRequest(
            domain=DataDomain.MARKET_DAILY,
            trade_date="2026-06-26",
            universe_kind="all_a",
            fetch_mode="date_events",
        ).normalized()
    with pytest.raises(ValueError, match="ETF date partition only supports"):
        DatePartitionFetchRequest(
            domain=DataDomain.ADJUST_FACTOR_EVENT,
            trade_date="2026-06-26",
            universe_kind="etf",
            fetch_mode="date_events",
        ).normalized()


def test_combined_daily_and_all_stock_fetch_is_one_persistent_command(monkeypatch: pytest.MonkeyPatch) -> None:
    from quant_data_platform import providers

    raw = _daily_raw(["sh.600000"])
    audit = pd.DataFrame({"symbol": ["600000.SH"], "trade_date": ["2026-06-26"]})
    commands: list[dict[str, object]] = []
    provider = BaostockProvider()

    def fake_request(payload: dict[str, object], *, timeout_seconds: int):
        commands.append({**payload, "timeout_seconds": timeout_seconds})
        return {
            "data": raw,
            "audit_data": audit,
            "meta": {"error_code": "0", "per_page_count": 20_000},
        }, 1, []

    monkeypatch.setattr(providers, "assert_baostock_batch_runtime", lambda: "0.9.3")
    monkeypatch.setattr(provider, "_persistent_date_request", fake_request)
    request = DatePartitionFetchRequest(
        domain=DataDomain.MARKET_DAILY,
        trade_date="2026-06-26",
        universe_kind="all_a",
        fetch_mode="date_snapshot",
    )

    result, universe = provider.fetch_date_partition_with_all_stock(request)

    assert commands == [
        {
            "kind": "bulk_with_all_stock",
            "endpoint": "query_daily_history_k_AStock",
            "trade_date": "2026-06-26",
            "timeout_seconds": 120,
        }
    ]
    assert result.coverage_report["combined_all_stock_audit"] is True
    assert result.raw_data.equals(raw)
    assert universe.equals(audit)


def test_mootdx_xdxr_raw_and_domain_adapter_preserve_semantics() -> None:
    class Client:
        def xdxr(self, *, symbol: str) -> pd.DataFrame:
            assert symbol == "600000"
            return pd.DataFrame(
                [
                    {
                        "year": 2026,
                        "month": 6,
                        "day": 26,
                        "category": 1,
                        "name": "除权除息",
                        "fenhong": 1.0,
                        "songzhuangu": 2.0,
                    }
                ]
            )

        def close(self) -> None:
            return None

    provider = MootdxOnlineProvider(_client_factory=Client)
    raw = provider.fetch_xdxr_raw(["600000.SH"])
    assert raw.error_report == []
    assert raw.data.loc[0, "provider_symbol"] == "600000.SH"
    assert raw.data.loc[0, "songzhuangu"] == pytest.approx(2.0)

    normalized = provider.fetch_domain(
        DomainFetchRequest(
            domain=DataDomain.CORPORATE_ACTIONS,
            symbols=("600000.SH",),
            start_date="2026-01-01",
            end_date="2026-12-31",
        )
    )
    assert len(normalized.data) == 1
    assert pd.isna(normalized.data.loc[0, "bonus_share_per_10"])
    assert "combined" in normalized.data.loc[0, "description"]


def test_batch_compatibility_uses_pit_alias_not_current_code_restatement() -> None:
    fields = {
        "date": "2018-12-28",
        "open": "10",
        "high": "11",
        "low": "9",
        "close": "10.5",
        "preclose": "10.6",
        "volume": "100",
        "amount": "1000",
        "adjustflag": "3",
        "tradestatus": "1",
        "isST": "0",
        "peTTM": "10",
        "pbMRQ": "2",
        "psTTM": "3",
        "pcfNcfTTM": "4",
    }
    old = {**fields, "code": "sz.300114", "requested_symbol": "300114.SZ", "pctChg": "-0.419583", "turn": "1.0"}
    restated = {**fields, "code": "sz.302132", "requested_symbol": "302132.SZ", "pctChg": "-0.419600", "turn": "1.1"}
    golden = pd.DataFrame([old, restated])
    current = golden.copy()
    batch = pd.DataFrame([{**old, "code": "sz.302132"}]).drop(columns="requested_symbol")

    issues = _compare_fixture(
        golden,
        current,
        batch,
        trade_date="2018-12-28",
        workspace_root=__import__("pathlib").Path(__file__).resolve().parents[2],
    )

    assert issues == []


def test_daily_code_gap_requires_explicit_lifecycle_or_status_classification() -> None:
    raw = _daily_raw(["sh.600000"])
    expected = ["600000.SH", "600001.SH"]
    identity = {symbol: f"security-{symbol}" for symbol in expected}
    master = pd.DataFrame(
        [
            {"symbol": "600000.SH", "list_date": "1999-11-10", "delist_date": ""},
            {"symbol": "600001.SH", "list_date": "2000-01-01", "delist_date": ""},
        ]
    )
    suspended = pd.DataFrame(
        [
            {"symbol": "600000.SH", "trade_status": "1", "is_suspended": False},
            {"symbol": "600001.SH", "trade_status": "0", "is_suspended": True},
        ]
    )

    accepted = audit_baostock_daily_raw(
        raw,
        query_date="2026-06-26",
        expected_codes=expected,
        expected_code_evidence=suspended,
        security_master=master,
        identity_security_by_symbol=identity,
    )
    assert accepted.quality_tier == "strict"
    assert accepted.metrics["code_gap_classification_counts"] == {"suspended": 1}

    blocked = audit_baostock_daily_raw(
        raw,
        query_date="2026-06-26",
        expected_codes=expected,
        expected_code_evidence=suspended.assign(trade_status="1", is_suspended=False),
        security_master=master,
        identity_security_by_symbol=identity,
    )
    assert blocked.quality_tier == "quarantined"
    assert {finding.code for finding in blocked.blockers} == {"daily_raw_code_gap_provider_gap"}


def test_daily_code_set_uses_pit_symbol_intervals_instead_of_current_code_list_date() -> None:
    raw = _daily_raw(["sz.000022", "sz.000043", "sz.300114"], trade_date="2012-09-10")
    expected = ["000022.SZ", "000043.SZ", "300114.SZ"]
    master = pd.DataFrame(
        [
            {"symbol": "000022.SZ", "list_date": "1993-05-05", "delist_date": "2018-12-26"},
            {"symbol": "001872.SZ", "list_date": "1993-05-05", "delist_date": ""},
            {"symbol": "000043.SZ", "list_date": "1994-09-28", "delist_date": "2019-12-16"},
            {"symbol": "001914.SZ", "list_date": "1994-09-28", "delist_date": ""},
            {"symbol": "300114.SZ", "list_date": "2010-08-27", "delist_date": "2025-02-17"},
            {"symbol": "302132.SZ", "list_date": "2010-08-27", "delist_date": ""},
        ]
    )
    history = pd.DataFrame(
        [
            {"symbol": "000022.SZ", "effective_from": "1993-05-05", "effective_to": "2018-12-25"},
            {"symbol": "001872.SZ", "effective_from": "2018-12-26", "effective_to": "9999-12-31"},
            {"symbol": "000043.SZ", "effective_from": "1994-09-28", "effective_to": "2019-12-15"},
            {"symbol": "001914.SZ", "effective_from": "2019-12-16", "effective_to": "9999-12-31"},
            {"symbol": "300114.SZ", "effective_from": "2010-08-27", "effective_to": "2025-02-16"},
            {"symbol": "302132.SZ", "effective_from": "2025-02-17", "effective_to": "9999-12-31"},
        ]
    )
    identity = {
        "000022.SZ": "port",
        "001872.SZ": "port",
        "000043.SZ": "property",
        "001914.SZ": "property",
        "300114.SZ": "avic",
        "302132.SZ": "avic",
    }

    report = audit_baostock_daily_raw(
        raw,
        query_date="2012-09-10",
        expected_codes=expected,
        expected_code_evidence=pd.DataFrame({"symbol": expected}),
        security_master=master,
        identity_security_by_symbol=identity,
        symbol_history=history,
    )

    assert report.quality_tier == "strict"
    assert report.metrics["security_master_active_code_count"] == 3
    assert report.metrics["missing_code_count"] == 0


def test_neighbor_count_jump_only_passes_with_listing_lifecycle_evidence() -> None:
    symbols = [f"{600000 + index:06d}.SH" for index in range(30)]
    raw = _daily_raw([f"sh.{symbol[:6]}" for symbol in symbols])
    master = pd.DataFrame(
        [
            {
                "symbol": symbol,
                "list_date": "2000-01-01" if index == 0 else "2026-06-26",
                "delist_date": "",
            }
            for index, symbol in enumerate(symbols)
        ]
    )
    identity = {symbol: f"security-{symbol}" for symbol in symbols}
    report = audit_baostock_daily_raw(
        raw,
        query_date="2026-06-26",
        expected_codes=symbols,
        expected_code_evidence=pd.DataFrame({"symbol": symbols}),
        security_master=master,
        neighbor_row_count=1,
        neighbor_expected_codes=symbols[:1],
        neighbor_date="2026-06-25",
        identity_security_by_symbol=identity,
    )

    assert report.quality_tier == "strict"
    assert report.metrics["neighbor_count_change_explained"] is True
    assert report.metrics["code_gap_classification_counts"]["new_listing"] == 29
